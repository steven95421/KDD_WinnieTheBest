#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from collections import Counter
from os import listdir
import base64, json, csv
import matplotlib.pyplot as plt


# In[2]:


def load_data(file_name, reset=False):
    ret = pd.read_csv(file_name, sep='\t')
    return ret


# In[3]:


data_path = '../data/'
pred_path = '../prediction_result/'
valid = load_data(data_path+'valid/valid.tsv')
testA = load_data(data_path+'testA/testA.tsv')
testB = load_data(data_path+'testB/testB.tsv')


# In[4]:


def sig(x):
    return 1/(1+np.e**(-x))

def load_one_file(filename, na_val, pad_len, method):
    preds = pd.read_csv(filename).fillna(na_val)
    qid2score = {}
    
    for row in preds.values:
        pid, score = row[1:pad_len+1], row[pad_len+1:]
        l = len([x for x in pid if x != na_val])
        if method == 'vote': qid2score[row[0]] = {pid[i]: weight[i] for i in range(l)}
        else: qid2score[row[0]] = {pid[i]: score[i] for i in range(l)}
        
    return qid2score

def ensemble(filenames, na_val, pad_len, thd, method='vote'):
    counts = Counter(valid['product_id'].values.tolist()+testA['product_id'].values.tolist()+testB['product_id'].values.tolist())
    print('loading files...')
    # first file
    qid2score = load_one_file(pred_path+filenames[0], na_val, pad_len, method)
    # other files
    for filename in tqdm(filenames[1:]):
        q2s = load_one_file(pred_path+filename, na_val, pad_len, method)
        # update score
        for qid in list(qid2score.keys()):
            for pid in list(qid2score[qid].keys()):
                qid2score[qid][pid] += q2s[qid][pid]
    # ensemble
    print('ensembling...')
    preds = {}
    for qid in tqdm(list(qid2score.keys())):
        pred = [(pid, qid2score[qid][pid]) for pid in list(qid2score[qid].keys())]
        tmp_thd = thd
        pred2 = [x for x in pred if counts[x[0]] <= tmp_thd]
        while len(pred2) < 5:
            tmp_thd += 1
            pred2 = [x for x in pred if counts[x[0]] <= tmp_thd]
        pred2.sort(key=lambda x: x[1], reverse=True)
        preds[int(qid)] = [int(pid) for pid, _ in pred2[:5]]
    return preds


# In[5]:


mcans = {5119: 38, 519: 39, 123: 36, 1234: 36, 12345: 31, 1213: 34, 207: 34,
         1333: 37, 2020: 36, 1115: 36, 666: 32, 2574: 39, 89983: 38, 46555: 38, 86031: 39,
         7414: 35, 71438: 38, 777: 35, 87: 35, 8787: 32, 878787: 30, 800061: 31, 856710: 31,
         42: 38, 426: 36, 64: 38, 8864: 36, 26: 39, 7382: 39, 1010: 39, 1001: 36,
         2330: 37, 612: 39, 24: 38, 25: 32, 2077: 35, 2049: 39, 2045: 39, 1917: 36,
         78667: 36, 68654: 34, 56474: 33, 56464: 36, 54367: 37, 4547: 32, 437: 36, 485: 38,
         132: 38, 257: 37, 584: 35, 931: 37, 792: 33, 603: 39, 746: 39, 480: 35}
visuals = {413: 32, 807: 37, 9527: 38, 713: 36, 625: 38, 1324: 38,
           987: 34, 116: 39, 41: 30, 145: 39,
           7328: 32, 62: 35, 3951: 37, 9736: 38}
f_m = ['prediction_all_cls_{}_{}.csv'.format(seed, mcans[seed])        for seed in list(mcans.keys()) if mcans[seed]]
f_v = ['prediction_all_cls_{}_{}.csv'.format(seed, visuals[seed])        for seed in list(visuals.keys()) if visuals[seed]]
add_visual = True
filenames = f_m+f_v if add_visual else f_m
weight = [1/i for i in range(1, 31)]
na_val = -1e10
pad_len = 30
thd = 1
method = 'sum'

# emsemble
preds = ensemble(filenames, na_val, pad_len, thd, method)
# write to file
header = ['query-id', 'product1', 'product2', 'product3', 'product4', 'product5']
with open(pred_path+'submission.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(header)
    for qid in sorted(list(preds.keys())):
        w.writerow([qid]+preds[qid])