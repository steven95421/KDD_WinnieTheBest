#!/usr/bin/env python
# coding: utf-8

# ### imports
# ***

# In[1]:


import torch, math
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torch import optim
from torch.optim.lr_scheduler import LambdaLR

from transformers import get_linear_schedule_with_warmup,get_cosine_with_hard_restarts_schedule_with_warmup
from transformers import AutoTokenizer, AutoModel

from gensim import corpora, similarities, models
from gensim.matutils import corpus2csc
from sklearn.model_selection import KFold
from lightgbm import LGBMClassifier

import pandas as pd 
import numpy as np
from collections import Counter
import base64
from tqdm.auto import tqdm
import pickle, random
import matplotlib.pyplot as plt
from multiprocessing import Pool
import sys, csv, json, os, gc, time


# ### parameters
# ***

# In[2]:


no = sys.argv[1]
device = torch.device('cuda:'+no) if torch.cuda.is_available() else torch.device('cpu')
print(device)

k = 10
lr = 1e-5
# true batch size = batch_size * grad_step
batch_size = 64
margin = 8
grad_step = 1
max_img_len = 30
epochs = 1 # only read data once
MOD = 20000
shuffle_fold = True
workers = 48
seed = 9115
random.seed(seed)
torch.manual_seed(seed)


# In[3]:


class params:
    LABEL_SIZE = 32
    IMG_FEAT_SIZE = 2048+6+LABEL_SIZE
    WORD_EMBED_SIZE = 1024
    LAYER = 12
    HIDDEN_SIZE = 1024
    MULTI_HEAD = 16
    DROPOUT_R = 0.1
    FLAT_MLP_SIZE = 512
    FLAT_GLIMPSES = 1
    FLAT_OUT_SIZE = 2048
    FF_SIZE = HIDDEN_SIZE*4
    HIDDEN_SIZE_HEAD = HIDDEN_SIZE // MULTI_HEAD
    OPT_BETAS = (0.9, 0.98)
    OPT_EPS = 1e-9
    TRAIN_SIZE = 3000000

__C = params()


# ### load data
# ***

# In[4]:


trash = {'!', '$', "'ll", "'s", ',', '&', ':', 'and', 'cut', 'is', 'are', 'was'}
trash_replace = ['"hey siri, play some', 'however, ', 'yin and yang, ',
                 'shopping mall/']

def process(x):
    tmp = x.split()
    if tmp[0] in trash: x = ' '.join(tmp[1:])
    if tmp[0][0] == '-': x = x[1:]
    for tr in trash_replace:
        x = x.replace(tr, '')
    return x

def normalize(x):
    ret = x['boxes'].copy()
    ret[:,0] /= x['image_h']
    ret[:,1] /= x['image_w']
    ret[:,2] /= x['image_h']
    ret[:,3] /= x['image_w']
    wh = (ret[:,2]-ret[:,0]) * (ret[:,3]-ret[:,1])
    wh2 = (ret[:,2]-ret[:,0]) / (ret[:,3]-ret[:,1]+1e-6)
    ret = np.hstack((ret, wh.reshape(-1,1), wh2.reshape(-1,1)))
    return ret

def load_data(file_name, reset=False, decode=True):
    ret = pd.read_csv(file_name, sep='\t')
    if decode:
        ret['boxes'] = ret['boxes'].apply(lambda x: np.frombuffer(base64.b64decode(x), dtype=np.float32).reshape(-1, 4))
        ret['features'] = ret['features'].apply(lambda x: np.frombuffer(base64.b64decode(x), dtype=np.float32).reshape(-1, 2048))
        ret['class_labels'] = ret['class_labels'].apply(lambda x: np.frombuffer(base64.b64decode(x), dtype=np.int64).reshape(-1, 1))
        ret['boxes'] = ret.apply(lambda x: normalize(x), axis=1)
        ret['features'] = ret.apply(lambda x: np.concatenate((x['class_labels'], x['features'], x['boxes']), axis=1)[:max_img_len], axis=1)
    ret['query'] = ret['query'].apply(lambda x: process(x))
    # reset query_id
    if reset:
        query2qid = {query: qid for qid, (query, _) in enumerate(tqdm(ret.groupby('query')))}
        ret['query_id'] = ret.apply(lambda x: query2qid[x['query']], axis=1)
    return ret


# In[5]:


data_path = '../data/'
model_path = '../user_data/model_data/'
pred_path = '../prediction_result/'
test = load_data(data_path+'valid/valid.tsv')
testB = load_data(data_path+'testB/testB.tsv')
answers = json.loads(open(data_path+'valid/valid_answer.json', 'r').read())
test['target'] = test.apply(lambda x: 1 if x['product_id'] in answers[str(x['query_id'])] else 0, axis=1)


# ### preprocess
# ***

# In[6]:


# load pre-trained model
take = 'roberta-large'
emb_size = __C.WORD_EMBED_SIZE
tokenizer = AutoTokenizer.from_pretrained(take)
pretrained_emb = AutoModel.from_pretrained(take)
pad_id = tokenizer.pad_token_id

qid2token = {qid: tokenizer.encode(group['query'].values[0]) for qid, group in tqdm(test.groupby('query_id'))}
test['token'] = test['query_id'].apply(lambda x: qid2token[x])
qid2token = {qid: tokenizer.encode(group['query'].values[0]) for qid, group in tqdm(testB.groupby('query_id'))}
testB['token'] = testB['query_id'].apply(lambda x: qid2token[x])


# ### model
# ***

# In[7]:


class FC(nn.Module):
    def __init__(self, in_size, out_size, dropout_r=0., use_relu=True):
        super(FC, self).__init__()
        self.dropout_r = dropout_r
        self.use_relu = use_relu

        self.linear = nn.Linear(in_size, out_size)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_r)

    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class MLP(nn.Module):
    def __init__(self, in_size, mid_size, out_size, dropout_r=0., use_relu=True):
        super(MLP, self).__init__()

        self.fc = FC(in_size, mid_size, dropout_r=dropout_r, use_relu=use_relu)
        self.linear = nn.Linear(mid_size, out_size)

    def forward(self, x):
        return self.linear(self.fc(x))


class LayerNorm(nn.Module):
    def __init__(self, size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps

        self.a_2 = nn.Parameter(torch.ones(size))
        self.b_2 = nn.Parameter(torch.zeros(size))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x-mean) / (std+self.eps) + self.b_2
    
class MHAtt(nn.Module):
    def __init__(self, __C):
        super(MHAtt, self).__init__()
        self.__C = __C

        self.linear_v = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_k = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_q = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_merge = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.dropout = nn.Dropout(__C.DROPOUT_R)

    def forward(self, v, k, q, mask):
        n_batches = q.size(0)
        v = self.linear_v(v).view(n_batches,
                                  -1,
                                  self.__C.MULTI_HEAD,
                                  self.__C.HIDDEN_SIZE_HEAD).transpose(1, 2)
        k = self.linear_k(k).view(n_batches,
                                  -1,
                                  self.__C.MULTI_HEAD,
                                  self.__C.HIDDEN_SIZE_HEAD).transpose(1, 2)
        q = self.linear_q(q).view(n_batches,
                                  -1,
                                  self.__C.MULTI_HEAD,
                                  self.__C.HIDDEN_SIZE_HEAD).transpose(1, 2)

        atted = self.att(v, k, q, mask)
        atted = atted.transpose(1, 2).contiguous().view(n_batches, -1, self.__C.HIDDEN_SIZE)
        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1))/math.sqrt(d_k)
        scores = scores.masked_fill(mask, -1e9)
        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)
        return torch.matmul(att_map, value)

# ---------------------------
# ---- Feed Forward Nets ----
# ---------------------------

class FFN(nn.Module):
    def __init__(self, __C):
        super(FFN, self).__init__()

        self.mlp = MLP(in_size=__C.HIDDEN_SIZE,
                       mid_size=__C.FF_SIZE,
                       out_size=__C.HIDDEN_SIZE,
                       dropout_r=__C.DROPOUT_R,
                       use_relu=True)

    def forward(self, x):
        return self.mlp(x)

# ------------------------
# ---- Self Attention ----
# ------------------------

class SA(nn.Module):
    def __init__(self, __C):
        super(SA, self).__init__()

        self.mhatt = MHAtt(__C)
        self.ffn = FFN(__C)

        self.dropout1 = nn.Dropout(__C.DROPOUT_R)
        self.norm1 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout2 = nn.Dropout(__C.DROPOUT_R)
        self.norm2 = LayerNorm(__C.HIDDEN_SIZE)

    def forward(self, x, x_mask):
        x = self.norm1(x + self.dropout1(self.mhatt(x, x, x, x_mask)))
        x = self.norm2(x + self.dropout2(self.ffn(x)))
        return x

# -------------------------------
# ---- Self Guided Attention ----
# -------------------------------

class SGA(nn.Module):
    def __init__(self, __C):
        super(SGA, self).__init__()

        self.mhatt1 = MHAtt(__C)
        self.mhatt2 = MHAtt(__C)
        self.ffn = FFN(__C)

        self.dropout1 = nn.Dropout(__C.DROPOUT_R)
        self.norm1 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout2 = nn.Dropout(__C.DROPOUT_R)
        self.norm2 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout3 = nn.Dropout(__C.DROPOUT_R)
        self.norm3 = LayerNorm(__C.HIDDEN_SIZE)

    def forward(self, x, y, x_mask, y_mask):
        x = self.norm1(x + self.dropout1(self.mhatt1(x, x, x, x_mask)))
        x = self.norm2(x + self.dropout2(self.mhatt2(y, y, x, y_mask)))
        x = self.norm3(x + self.dropout3(self.ffn(x)))
        return x
    
class GA(nn.Module):
    def __init__(self, __C):
        super(GA, self).__init__()

        self.mhatt2 = MHAtt(__C)
        self.ffn = FFN(__C)

        self.dropout2 = nn.Dropout(__C.DROPOUT_R)
        self.norm2 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout3 = nn.Dropout(__C.DROPOUT_R)
        self.norm3 = LayerNorm(__C.HIDDEN_SIZE)

    def forward(self, x, y, x_mask, y_mask):
        x = self.norm2(x + self.dropout2(self.mhatt2(y, y, x, y_mask)))
        x = self.norm3(x + self.dropout3(self.ffn(x)))
        return x
        
# ------------------------------------------------
# ---- MAC Layers Cascaded by Encoder-Decoder ----
# ------------------------------------------------

class MCA_ED(nn.Module):
    def __init__(self, __C):
        super(MCA_ED, self).__init__()

        self.enc_list = nn.ModuleList([SA(__C) for _ in range(__C.LAYER)])
        self.dec_list = nn.ModuleList([SGA(__C) for _ in range(__C.LAYER)])

    def forward(self, x, y, x_mask, y_mask):
        # Get hidden vector
        for enc in self.enc_list:
            x = enc(x, x_mask)
        for dec in self.dec_list:
            y = dec(y, x, y_mask, x_mask)
        return x, y
      
    
class AttFlat(nn.Module):
    def __init__(self, __C):
        super(AttFlat, self).__init__()
        self.__C = __C

        self.mlp = MLP(in_size=__C.HIDDEN_SIZE,
                       mid_size=__C.FLAT_MLP_SIZE,
                       out_size=__C.FLAT_GLIMPSES,
                       dropout_r=__C.DROPOUT_R,
                       use_relu=True)
        self.linear_merge = nn.Linear(__C.HIDDEN_SIZE*__C.FLAT_GLIMPSES, __C.FLAT_OUT_SIZE)

    def forward(self, x, x_mask):
        att = self.mlp(x)
        att = att.masked_fill(x_mask.squeeze(1).squeeze(1).unsqueeze(2), -1e9)
        att = F.softmax(att, dim=1)

        att_list = []
        for i in range(self.__C.FLAT_GLIMPSES):
            att_list.append(torch.sum(att[:,:,i:i+1]*x, dim=1))

        x_atted = torch.cat(att_list, dim=1)
        x_atted = self.linear_merge(x_atted)
        return x_atted

# -------------------------
# ---- Main MCAN Model ----
# -------------------------

class Net(nn.Module):
    def __init__(self, __C, pretrained_emb, answer_size):
        super(Net, self).__init__()

        self.embedding = pretrained_emb.embeddings
        self.label_emb = nn.Embedding(33, __C.LABEL_SIZE)
        self.img_feat_linear = MLP(__C.IMG_FEAT_SIZE, __C.IMG_FEAT_SIZE, __C.HIDDEN_SIZE)
        self.lang_feat_linear = nn.Linear(__C.WORD_EMBED_SIZE, __C.HIDDEN_SIZE)
        self.backbone = MCA_ED(__C)

        self.attflat_img = AttFlat(__C)
        self.attflat_lang = AttFlat(__C)
        
        self.proj_norm_lang = LayerNorm(__C.FLAT_OUT_SIZE)
        self.proj_norm_img = LayerNorm(__C.FLAT_OUT_SIZE)
        self.proj_norm_mul = LayerNorm(__C.FLAT_OUT_SIZE)
        self.proj_norm_dis = LayerNorm(__C.FLAT_OUT_SIZE)
        self.proj = MLP(__C.FLAT_OUT_SIZE*4, __C.FLAT_OUT_SIZE*2, answer_size)

    def forward(self, ques_ix, img_feats):
        proj_feats = []
        for img_feat in img_feats:
            # Make mask
            lang_feat_mask = self.make_mask(ques_ix.unsqueeze(2), pad_id)
            img_feat_mask = self.make_mask(img_feat, 0)

            # Pre-process Language Feature
            lang_feat = self.embedding(ques_ix)
            lang_feat = self.lang_feat_linear(lang_feat)

            # Pre-process Image Feature
            label_feat = self.label_emb(img_feat[:,:,0].long())
            img_feat = torch.cat((img_feat[:,:,1:], label_feat), dim=2)
            img_feat = self.img_feat_linear(img_feat)

            # Backbone Framework
            lang_feat, img_feat = self.backbone(lang_feat, img_feat, lang_feat_mask, img_feat_mask)
            lang_feat = self.attflat_lang(lang_feat, lang_feat_mask)
            img_feat = self.attflat_img(img_feat, img_feat_mask)
            distance = torch.abs(lang_feat-img_feat)
            
            proj_feat = torch.cat((self.proj_norm_lang(lang_feat),
                                   self.proj_norm_img(img_feat),
                                   self.proj_norm_mul(lang_feat*img_feat),
                                   self.proj_norm_dis(distance)
                                  ), dim=1)
            proj_feat = self.proj(proj_feat)
            proj_feats.append(proj_feat)
        return proj_feats

    # Masking
    def make_mask(self, feature, target):
        return (torch.sum(torch.abs(feature), dim=-1) == target).unsqueeze(1).unsqueeze(2)


# ### train
# ***


# In[9]:


class CustomDataset(data.Dataset):
    def __init__(self, train_x):
        self.train_x = train_x
        
    def __getitem__(self, index):
        tokens, features = self.train_x[index][0], self.train_x[index][1]
        return [tokens, features]
    
    def __len__(self):
        return len(self.train_x)
    
def collate_fn(batch):
    tokens, features = zip(*batch)
    max_len_t, max_len_f = len(max(tokens, key=lambda x: len(x))), len(max(features, key=lambda x: len(x)))
    tokens, features = [token+[pad_id]*(max_len_t-len(token)) for token in tokens], [np.concatenate((feature, np.zeros((max_len_f-feature.shape[0], feature.shape[1]))), axis=0) for feature in features]
    return torch.LongTensor(tokens), torch.FloatTensor(features)

def custom_schedule(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, amplitude=0.1, last_epoch=-1):
    
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = 2.0 * math.pi * float(num_cycles) * float(current_step-num_warmup_steps) / float(max(1, num_training_steps-num_warmup_steps))
        linear = float(num_training_steps-current_step) / float(max(1, num_training_steps-num_warmup_steps))
        return abs(linear + math.sin(progress)*linear*amplitude)

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def shuffle(x):
    idxs = [i for i in range(x.shape[0])]
    random.shuffle(idxs)
    return x[idxs]

def nDCG_score(preds, answers):
    iDCG = sum([sum([np.log(2)/np.log(i+2) for i in range(min(len(answer), 5))])                 for answer in list(answers.values())])
    DCG = sum([sum([np.log(2)/np.log(i+2) if preds[qid][i] in answers[str(qid)] else 0                     for i in range(len(preds[qid]))]) for qid in list(preds.keys())])
    return DCG/iDCG

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


# In[10]:


print('initializing model...')
nDCGs = []
best_nDCG = 0.0
model = Net(__C, pretrained_emb, 1).to(device)


# ### prediction
# ***

# In[11]:


def extract_embedding(model, ques_ix, img_feat):
    # Make mask
    lang_feat_mask = model.make_mask(ques_ix.unsqueeze(2), pad_id)
    img_feat_mask = model.make_mask(img_feat, 0)

    # Pre-process Language Feature
    lang_feat = model.embedding(ques_ix)
    lang_feat = model.lang_feat_linear(lang_feat)

    # Pre-process Image Feature
    label_feat = model.label_emb(img_feat[:,:,0].long())
    img_feat = torch.cat((img_feat[:,:,1:], label_feat), dim=2)
    img_feat = model.img_feat_linear(img_feat)

    # Backbone Framework
    lang_feat, img_feat = model.backbone(lang_feat, img_feat, lang_feat_mask, img_feat_mask)
    lang_feat = model.attflat_lang(lang_feat, lang_feat_mask)
    img_feat = model.attflat_img(img_feat, img_feat_mask)
    distance = torch.abs(lang_feat-img_feat)

    proj_feat = torch.cat((model.proj_norm_lang(lang_feat),
                           model.proj_norm_img(img_feat),
                           model.proj_norm_mul(lang_feat*img_feat),
                           model.proj_norm_dis(distance)
                          ), dim=1)
    return proj_feat


# In[12]:


def get_cls(model, n_splits):
    model.eval()
    qids = [[qid] for qid, _ in test.groupby('query_id')]
    kf = KFold(n_splits=n_splits, shuffle=True)
    train_x, train_y = [], []
    test_x, test_y = [], []
    qid2fold = {qids[idx][0]: i                 for i, (train_index, test_index) in enumerate(kf.split(qids))                 for idx in test_index}
    
    with torch.no_grad():
        for qid, group in tqdm(test.groupby('query_id')):
            # prepare batch
            tokens, features = group['token'].values.tolist(), group['features'].values.tolist()
            max_len_f = len(max(features, key=lambda x: len(x)))
            features = [np.concatenate((feature, np.zeros((max_len_f-feature.shape[0], feature.shape[1]))), axis=0) for feature in features]
            # # to tensor
            tokens = torch.LongTensor(tokens).to(device)
            features = torch.FloatTensor(features).to(device)
            # predict
            tmp_x = extract_embedding(model, tokens, features).tolist()
            tmp_y = group['target'].values.tolist()
            # only use first fold
            if qid2fold[qid]:
                train_x += tmp_x
                train_y += tmp_y
            else:
                test_x += tmp_x
                test_y += tmp_y
    
    train_x, train_y = np.array(train_x), np.array(train_y)
    test_x, test_y = np.array(test_x), np.array(test_y)
    print('train:test = {}:{}'.format(train_x.shape[0], test_x.shape[0]))
    cls = LGBMClassifier(random_state=0, n_jobs=int(sys.argv[2]))
    cls.fit(train_x, train_y,
            eval_set=[(test_x, test_y)],
            early_stopping_rounds=100,
            verbose=100)
        
    return cls


# In[14]:


def predict_all(model, test, pad_len, cls):
    model.eval()
    preds = {}
    
    with torch.no_grad():
        for qid, group in tqdm(test.groupby('query_id')):
            # prepare batch
            tokens, features = group['token'].values.tolist(), group['features'].values.tolist()
            max_len_f = len(max(features, key=lambda x: len(x)))
            features = [np.concatenate((feature, np.zeros((max_len_f-feature.shape[0], feature.shape[1]))), axis=0) for feature in features]
            # # to tensor
            tokens = torch.LongTensor(tokens).to(device)
            features = torch.FloatTensor(features).to(device)
            # predict
            embeddings = np.array(extract_embedding(model, tokens, features).tolist())
            out = cls.predict_proba(embeddings)[:,1]
            pred = [(pid, val) for pid, val in zip(group['product_id'].values.tolist(), out.tolist())]
            pred.sort(key=lambda x: x[1], reverse=True)
            assert len(pred) <= pad_len
            pid, score = [p for p, s in pred], [s for p, s in pred]
            pid, score = pid+[np.nan]*(pad_len-len(pred)), score+[np.nan]*(pad_len-len(pred))
            preds[qid] = pid+score
            
    return preds


# In[ ]:


seeds = {5119: 38, 519: 39, 123: 36, 1234: 36, 12345: 31, 1213: 34, 207: 34,
         1333: 37, 2020: 36, 1115: 36, 666: 32, 2574: 39, 89983: 38, 46555: 38, 86031: 39,
         7414: 35, 71438: 38, 777: 35, 87: 35, 8787: 32, 878787: 30, 800061: 31, 856710: 31,
         42: 38, 426: 36, 64: 38, 8864: 36, 26: 39, 7382: 39, 1010: 39, 1001: 36,
         2330: 37, 612: 39, 24: 38, 25: 32, 2077: 35, 2049: 39, 2045: 39, 1917: 36,
         78667: 36, 68654: 34, 56474: 33, 56464: 36, 54367: 37, 4547: 32, 437: 36, 485: 38,
         132: 38, 257: 37, 584: 35, 931: 37, 792: 33, 603: 39, 746: 39, 480: 35}
folds = [i for i in range(30, 40)]
pad_len = 30
thd = 9999
n_splits = 10

for seed in list(seeds.keys()):
    t0 = time.time()
    for fold in folds:
        print('seed: {}; fold: {}'.format(seed, fold))
        # load model weights
        try: model.load_state_dict(torch.load(model_path+'model_MCAN-RoBERTa_pair-cat_box_tfidf-neg_focal_all_shared_{}_{}'.format(seed, fold), map_location=device))
        except: continue
        # train cls
        cls = get_cls(model, n_splits)
        # test
        preds = predict_all(model, testB, pad_len, cls)
        # write to file
        header = ['qid'] + ['p'+str(i) for i in range(pad_len)] + ['s'+str(i) for i in range(pad_len)]
        with open(pred_path+'prediction_all_cls_{}_{}.csv'.format(seed, fold), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(header)
            for qid in sorted(list(preds.keys())):
                w.writerow([qid]+preds[qid])
    t = round(time.time()-t0)
    print('time consumed: {} min {} sec'.format(t//60, t%60))

