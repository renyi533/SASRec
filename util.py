from __future__ import print_function
import sys
import copy
import random
import numpy as np
from collections import defaultdict
import pytest

def data_partition(fname, exp_len):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    Item = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    f = open('data/%s.txt' % fname, 'r')
    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)
        Item[i].append(u)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < exp_len:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    return [user_train, user_valid, user_test, usernum, itemnum, User, Item]

def gini(x):
    x = np.array(x)
    total = 0
    for i, xi in enumerate(x[:-1], 1):
        total += np.sum(np.abs(xi - x[i:]))
    return total / (len(x)**2 * np.mean(x))

def evaluate(model, dataset, args, sess, full_test=True):
    [train, valid, test, usernum, itemnum, User, Item] = copy.deepcopy(dataset)

    NDCG = 0.0
    W_NDCG = 0.0
    HT = 0.0
    W_HT = 0.0
    valid_user = 0.0
    total_w = 0.0
    if usernum>10000 and (not full_test):
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    set_u =[]
    set_w = []
    set_seq = []
    set_test_item = []
    for u in users:

        if len(train[u]) < 1 or len(test[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        seq[idx] = valid[u][0]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        rated = set(train[u])
        rated.add(0)
        item_idx = [test[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)
        
        w = 1.0 / len(Item[test[u][0]])
        total_w += w
        set_u.append(u)
        set_w.append(w)
        set_seq.append(seq)
        set_test_item.append(item_idx)
        
        valid_user += 1

        if valid_user % 1000 == 0:
            ndcg, w_ndcg, ht, w_ht = compute_metric(model, sess, set_u, set_w, set_seq, set_test_item)
            set_u.clear()
            set_w.clear()
            set_seq.clear()
            set_test_item.clear()
            NDCG += ndcg
            W_NDCG += w_ndcg
            HT += ht
            W_HT += w_ht
            print ('.', end="")
            sys.stdout.flush()
            
    if len(set_u) > 0:
        ndcg, w_ndcg, ht, w_ht = compute_metric(model, sess, set_u, set_w, set_seq, set_test_item)
        NDCG += ndcg
        W_NDCG += w_ndcg
        HT += ht
        W_HT += w_ht       
    
    return NDCG / valid_user, HT / valid_user, W_NDCG / total_w, W_HT/total_w

def compute_metric(model, sess, set_u, set_w, set_seq, set_test_item):
    predictions = -model.predict(sess, set_u, set_seq, set_test_item)
    ranks = predictions.argsort().argsort()[:, 0]
    ndcgs = 1 / np.log2(ranks + 2)
    ones = np.ones_like(ndcgs, dtype=float)
    zeros = np.zeros_like(ndcgs)
    
    ndcg_10 = np.where(ranks < 10, ndcgs, zeros)
    ht_10 = np.where(ranks <10, ones, zeros)
    
    w_ndcg_10 = set_w * ndcg_10
    w_ht_10 = set_w * ht_10
    NDCG_10 = np.sum(ndcg_10)
    W_NDCG_10 = np.sum(w_ndcg_10)
    HT_10 = np.sum(ht_10)
    W_HT_10 = np.sum(w_ht_10)
    return NDCG_10, W_NDCG_10, HT_10, W_HT_10

    #NDCG = 0.0
    #W_NDCG = 0.0
    #HT = 0.0
    #W_HT = 0.0    
    #for i in range(len(predictions)):
    #    prediction = predictions[i]
    #    rank = prediction.argsort().argsort()[0]
    #    if rank < 10:
    #        NDCG += 1 / np.log2(rank + 2)
    #        W_NDCG += set_w[i] / np.log2(rank + 2)
    #        HT += 1
    #        W_HT += set_w[i]   
    #print(NDCG_10, NDCG)
    #print(W_NDCG_10, W_NDCG)
    #print(HT_10, HT)
    #print(W_HT_10, W_HT)
    #assert [NDCG_10, W_NDCG_10, HT_10,W_HT_10]  == pytest.approx([NDCG, W_NDCG, HT, W_HT])
    #return NDCG_10, W_NDCG_10, HT_10, W_HT_10

def evaluate_valid(model, dataset, args, sess, full_test=True):
    [train, valid, test, usernum, itemnum, User, Item] = copy.deepcopy(dataset)

    NDCG = 0.0
    W_NDCG = 0.0
    HT = 0.0
    W_HT = 0.0
    valid_user = 0.0
    total_w = 0.0
    if usernum>10000 and (not full_test):
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
        
    set_u =[]
    set_w = []
    set_seq = []
    set_test_item = []
    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        rated = set(train[u])
        rated.add(0)
        item_idx = [valid[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        w = 1.0 / len(Item[valid[u][0]])
        total_w += w
        set_u.append(u)
        set_w.append(w)
        set_seq.append(seq)
        set_test_item.append(item_idx)
        valid_user += 1

        if valid_user % 1000 == 0:
            ndcg, w_ndcg, ht, w_ht = compute_metric(model, sess, set_u, set_w, set_seq, set_test_item)
            set_u.clear()
            set_w.clear()
            set_seq.clear()
            set_test_item.clear()
            NDCG += ndcg
            W_NDCG += w_ndcg
            HT += ht
            W_HT += w_ht
            print ('.', end="")
            sys.stdout.flush()
            
    if len(set_u) > 0:
        ndcg, w_ndcg, ht, w_ht = compute_metric(model, sess, set_u, set_w, set_seq, set_test_item)
        NDCG += ndcg
        W_NDCG += w_ndcg
        HT += ht
        W_HT += w_ht  

    return NDCG / valid_user, HT / valid_user, W_NDCG / total_w, W_HT/total_w
