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

from transformers import AdamW, get_linear_schedule_with_warmup
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
max_lang_len = 15
max_img_len = 30
epochs = 1
MOD = 20000
shuffle_fold = True
seed = 9527
random.seed(seed)
torch.manual_seed(seed)


# In[3]:


class params:
    IMG_FEAT_SIZE = 2048+6
    WORD_EMBED_SIZE = 1024
    LAYER = 6
    HIDDEN_SIZE = 1024
    MULTI_HEAD = 8
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

def sort_by_area(x):
    return np.array(sorted(x.tolist(), key=lambda x: x[-1], reverse=True))

def load_data(file_name, reset=False, decode=True):
    ret = pd.read_csv(file_name, sep='\t')
    if decode:
        ret['boxes'] = ret['boxes'].apply(lambda x: np.frombuffer(base64.b64decode(x), dtype=np.float32).reshape(-1, 4))
        ret['features'] = ret['features'].apply(lambda x: np.frombuffer(base64.b64decode(x), dtype=np.float32).reshape(-1, 2048))
        ret['class_labels'] = ret['class_labels'].apply(lambda x: np.frombuffer(base64.b64decode(x), dtype=np.int64).reshape(-1, 1))
        ret['boxes'] = ret.apply(lambda x: normalize(x), axis=1)
        ret['features'] = ret.apply(lambda x: np.concatenate((x['features'], x['boxes']), axis=1)[:max_img_len], axis=1)
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

# In[ ]:


# load pre-trained model
take = 'bert-large-uncased'
emb_size = __C.WORD_EMBED_SIZE
tokenizer = AutoTokenizer.from_pretrained(take)
pretrained_emb = AutoModel.from_pretrained(take)
pad_id = tokenizer.pad_token_id
sep_id = tokenizer.sep_token_id

qid2token = {qid: tokenizer.encode(group['query'].values[0]) for qid, group in tqdm(test.groupby('query_id'))}
test['token'] = test['query_id'].apply(lambda x: qid2token[x])
test['token'] = test['token'].apply(lambda x: x[:max_lang_len-1]+[sep_id] if len(x) > max_lang_len else x)
qid2token = {qid: tokenizer.encode(group['query'].values[0]) for qid, group in tqdm(testB.groupby('query_id'))}
testB['token'] = testB['query_id'].apply(lambda x: qid2token[x])
testB['token'] = testB['token'].apply(lambda x: x[:max_lang_len-1]+[sep_id] if len(x) > max_lang_len else x)


# ### model
# ***

# In[ ]:


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


class VisualBERT(nn.Module):
    def __init__(self, __C, bert):
        super(VisualBERT, self).__init__()
        
        self.bert = bert
        self.linear_img = nn.Linear(__C.IMG_FEAT_SIZE, __C.WORD_EMBED_SIZE)
        self.out = MLP(__C.WORD_EMBED_SIZE, __C.WORD_EMBED_SIZE//2, 1)
        
    def forward(self, ques_ix, img_feats):
        proj_feats = []
        for img_feat in img_feats:
            # Make mask & token type ids
            mask = self.make_mask(ques_ix.unsqueeze(2), img_feat, pad_id)
            token = self.get_token_type(ques_ix, img_feat)
            # Preprocess features
            lang_feat = self.bert.embeddings.word_embeddings(ques_ix)
            img_feat = self.linear_img(img_feat)
            combine_feat = torch.cat((lang_feat, img_feat), dim=1)
            # Token embeddings & position embeddings
            position_ids = torch.arange(token.size(1), dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(token.size())
            position_embeddings = self.bert.embeddings.position_embeddings(position_ids)
            token_type_embeddings = self.bert.embeddings.token_type_embeddings(token)
            # Add all
            embeddings = combine_feat+position_embeddings+token_type_embeddings
            embeddings = self.bert.embeddings.dropout(self.bert.embeddings.LayerNorm(embeddings))
            # Go through the rest of BERT
            head_mask = self.bert.get_head_mask(None, self.bert.config.num_hidden_layers)
            extended_attention_mask = self.bert.get_extended_attention_mask(mask, mask.size(), device)
            encoder_outputs = self.bert.encoder(embeddings,
                                                attention_mask=extended_attention_mask,
                                                head_mask=head_mask,
                                                encoder_hidden_states=None,
                                                encoder_attention_mask=None)
            # CLS embedding & output value
            outputs = encoder_outputs[0][:,0,:]
            proj_feats.append(self.out(outputs))
        return proj_feats
            
    # Masking
    def make_mask(self, lang_feat, img_feat, target):
        # 1 for NOT masked; 0 for masked
        # [batch, len]
        lang_mask = (torch.sum(torch.abs(lang_feat), dim=-1) != target).long()
        img_mask = (torch.sum(torch.abs(img_feat), dim=-1) != 0).long()
        return torch.cat((lang_mask, img_mask), dim=1)
    
    # Token type ids
    def get_token_type(self, lang_feat, img_feat):
        #    lang      img
        # 0 0 0 0 0 0 1 1 1 1 1
        lang_token = torch.zeros(lang_feat.size(0), lang_feat.size(1)).to(device)
        img_token = torch.ones(img_feat.size(0), img_feat.size(1)).to(device)
        return torch.cat((lang_token, img_token), dim=1).long()


# ### train
# ***

# In[ ]:


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


# In[ ]:


print('initializing model...')
nDCGs = []
best_nDCG = 0.0
model = VisualBERT(__C, pretrained_emb).to(device)


# ### prediction
# ***

# In[ ]:


def extract_embedding(model, ques_ix, img_feat):
    # Make mask & token type ids
    mask = model.make_mask(ques_ix.unsqueeze(2), img_feat, pad_id)
    token = model.get_token_type(ques_ix, img_feat)
    # Preprocess features
    lang_feat = model.bert.embeddings.word_embeddings(ques_ix)
    img_feat = model.linear_img(img_feat)
    combine_feat = torch.cat((lang_feat, img_feat), dim=1)
    # Token embeddings & position embeddings
    position_ids = torch.arange(token.size(1), dtype=torch.long, device=device)
    position_ids = position_ids.unsqueeze(0).expand(token.size())
    position_embeddings = model.bert.embeddings.position_embeddings(position_ids)
    token_type_embeddings = model.bert.embeddings.token_type_embeddings(token)
    # Add all
    embeddings = combine_feat+position_embeddings+token_type_embeddings
    embeddings = model.bert.embeddings.dropout(model.bert.embeddings.LayerNorm(embeddings))
    # Go through the rest of BERT
    head_mask = model.bert.get_head_mask(None, model.bert.config.num_hidden_layers)
    extended_attention_mask = model.bert.get_extended_attention_mask(mask, mask.size(), device)
    encoder_outputs = model.bert.encoder(embeddings,
                                         attention_mask=extended_attention_mask,
                                         head_mask=head_mask,
                                         encoder_hidden_states=None,
                                         encoder_attention_mask=None)
    # CLS embedding & output value
    outputs = encoder_outputs[0][:,0,:]
    return outputs


# In[ ]:


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


# In[ ]:


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


seeds = {413: 32, 807: 37, 9527: 38, 713: 36, 625: 38, 1324: 38,
         987: 34, 116: 39, 41: 30, 145: 39,
         7328: 32, 62: 35, 3951: 37, 9736: 38}
folds = [i for i in range(30, 40)]
pad_len = 30
thd = 9999
n_splits = 10

for seed in list(seeds.keys()):
    if not seeds[seed]: continue
    t0 = time.time()
    for fold in folds:
        print('seed: {}; fold: {}'.format(seed, fold))
        # load model weights
        try: model.load_state_dict(torch.load(model_path+'model_Visual-BERT_pair_box_tfidf-neg_focal_all_{}_{}'.format(seed, fold), map_location=device))
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

