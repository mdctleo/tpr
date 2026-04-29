import sys
from copy import deepcopy
from time import time
from numpy.core.fromnumeric import argmax

import pandas as pd 
from sklearn.metrics import \
    roc_auc_score as auc_score, \
    f1_score, average_precision_score as ap_score
import torch 
from tqdm import tqdm
from attack_lib import evasion_attack,slice_data
import json

from models.recurrent import GRU 
from model_1 import VGRNN
import loaders.load_0501 as ld 
from loaders.tdata import TData
import random
import numpy as np
import pickle
import json
import random
import numpy as np
seed = 0
random.seed(seed) # python random generator
np.random.seed(seed) # numpy random generator

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

MODEL_PATH='VGRNN_0501.pkl'
CUTOFF = 0.92489093542099
DEVICE = 'cpu'#'cuda'

def score_stats(title, scores, labels,  cutoff, ctime):
    # Cat scores from timesteps together bc separation 
    # is no longer necessary 
    scores = torch.cat(scores, dim=0).detach()
    labels = torch.cat(labels, dim=0).clamp(max=1)
    # weights = torch.cat(weights, dim=0)

    # Classify using cutoff from earlier
    classified = torch.zeros(labels.size())
    classified[scores <= cutoff] = 1

    # Calculate TPR
    p = classified[labels==1]
    tpr = p.mean()
    tp = p.sum()
    del p

    # Calculate FPR
    f = classified[labels==0]
    fp = f.sum()
    fpr = f.mean()
    del f 
    
    # Because a low score correlates to a 1 lable, sub from 1 to get
    # accurate AUC/AP scores
    scores = 1-scores

    # Get metrics
    auc = auc_score(labels, scores)
    ap = ap_score(labels, scores)
    f1 = f1_score(labels, classified)

    print(title)
    print("Learned Cutoff %0.4f" % cutoff)
    print("TPR: %0.4f, FPR: %0.4f" % (tpr, fpr))
    print("TP: %d  FP: %d" % (tp, fp))
    print("F1: %0.8f" % f1)
    print("AUC: %0.4f  AP: %0.4f\n" % (auc,ap))

    return {
        'Model': title,
        'TPR':tpr.item(), 
        'FPR':fpr.item(), 
        'TP':tp.item(), 
        'FP':fp.item(), 
        'F1':f1, 
        'AUC':auc, 
        'AP': ap,
        'FwdTime':ctime
    }


def main():

    states = pickle.load(open(MODEL_PATH, 'rb'))
    DELTA= int(60* 1000*2.5)
    print(DELTA)
    TR___=72378559
    TR_START=0
    #TR_END=TR___-DELTA*3 
    TR_END=TR___-max((TR___ - TR_START) // 20, DELTA*2)
    VAL_START=TR_END
    VAL_END=TR___

    TE_START=TR___
    TE_END = 86674422
    data = ld.load_optc_dist (8, start=TR___, end=TE_END, delta=DELTA, is_test=True)
    model = VGRNN(8746, 32, 16, pred=False)
    print(states['states'])
    model.load_state_dict(states['states'].state_dict())
    model.recurrent.weight_xz[0].load_state_dict(states['states'].recurrent.weight_xz[0].state_dict())
    model.recurrent.weight_hz[0].load_state_dict(states['states'].recurrent.weight_hz[0].state_dict())
    model.recurrent.weight_xr[0].load_state_dict(states['states'].recurrent.weight_xr[0].state_dict())
    model.recurrent.weight_hr[0].load_state_dict(states['states'].recurrent.weight_hr[0].state_dict())
    model.recurrent.weight_xh[0].load_state_dict(states['states'].recurrent.weight_xh[0].state_dict())
    model.recurrent.weight_hh[0].load_state_dict(states['states'].recurrent.weight_hh[0].state_dict())
    h0=states['h0']
    # print(h0)
    model.eval()
    # preds = []
    # labels = []
    # cnts = []
    # with torch.no_grad():
    #     for t in tqdm(range(data.T)):
    #         data_t = slice_data(data, t)
    #         zs, _,_ = model.forward(data_t, TData.ALL,h0=h0)
    #         pred = model.decode(data_t.eis[0][0], data_t.eis[0][1], zs[0])
    #         preds.append(pred.cpu())
    #         labels.append(data_t.ys[0].cpu())
    #         # cnts.append(data_t.cnt[0].cpu())
    #     score_stats('TITLE', preds, labels,  cutoff=CUTOFF, ctime=None)
    
    
    # With attack
    all_mals = []
    for t, (one_es, one_ys) in enumerate(zip(data.eis, data.ys)):
        all_mals.extend( [(t, u.item(),v.item()) for u, v, y in zip(one_es[0], one_es[1], one_ys) if y==1] )
    print (all_mals)
    print (len(all_mals))
    ALL_KS = [2,5,10,20,50]
    all_final = {k:[] for k in ALL_KS}
    all_cvrs = {k:[] for k in ALL_KS}
    atk_to_save = {'goal':[], 'edge':[]}
    all_init = []
    results={}
    #for tgt_t,tgt_u,tgt_v in all_mals[:2]:
    i=0
    mise=[]
    for tgt_t,tgt_u,tgt_v in all_mals:
        #tgt_t,tgt_u,tgt_v=457,173,810
        data_t = slice_data(data, tgt_t)
        print (i,tgt_t, tgt_u, tgt_v)
        i+=1
        enc_out ,_,g= model.forward(data_t, mask_enum=2, h0=h0)
        print(g)
        init_pred = model.decode([tgt_u], [tgt_v], enc_out[0]).item()
        all_init.append(init_pred)

        #extra_edges = evasion_attack(enc, rnn, rnn_h0, data, tgt_t, tgt_u, tgt_v, CUTOFF, K=5, N_choice=10)
        all_extra_edges = evasion_attack(model, h0, data_t, 0, tgt_u, tgt_v, CUTOFF, K=max(ALL_KS), N_choice=100000)
        if len(all_extra_edges) == 0:
                mise.append((tgt_t, tgt_u, tgt_v))
                continue
        atk_to_save['goal'].append((tgt_t, tgt_u, tgt_v))
        atk_to_save['edge'].append(all_extra_edges)
        print(all_extra_edges)
        for k in ALL_KS:
            extra_edges = {t:[] for t in range(data_t.T)}
            for one_t, one_u, one_v in all_extra_edges[:k]:
                    extra_edges[one_t].append((one_u, one_v))
            enc_out,_,_= model.forward(data_t, mask_enum=2, h0=h0 ,extra_edges=extra_edges)
            final_pred = model.decode([tgt_u], [tgt_v], enc_out[0]).item()
            cover_preds = []
            for prev_t, es in extra_edges.items():
                for prev_u, prev_v in es:
                    cover_preds.append(model.decode([prev_u], [prev_v], enc_out[prev_t]).item())
            all_final[k].append(final_pred)
            all_cvrs[k].append(cover_preds)
            print ("Init pred: %.6f; Final pred: %.6f; Avg cover pred: %.6f; Avg cover stealthy: %.6f;"%(init_pred, final_pred, np.mean(cover_preds), np.mean([p>CUTOFF for p in cover_preds])))
        #break
    print ("Avg init: %.6f; stealthy: %.6f"%(np.mean(all_init), np.mean([p>CUTOFF for p in all_init])))
    for k in ALL_KS:
        print ("K=%d,Avg final: %.6f; stealthy: %.6f"%(k, np.mean(all_final[k]), np.mean([p>CUTOFF for p in all_final[k]])))
        flatten_cvrs = [v for cvr in all_cvrs[k] for v in cvr]
        print ("K=%d,Avg cvrs: %.6f; stealthy: %.6f"%(k, np.mean(flatten_cvrs), np.mean([p>CUTOFF for p in flatten_cvrs])))
        final_succeed = []
        for i in range(len(all_final[k])):
            is_succeed = (all_final[k][i] > CUTOFF)
            for cvr_pred in all_cvrs[k][i]:
                if not is_succeed:
                    break
                is_succeed = is_succeed and (cvr_pred > CUTOFF)
            final_succeed.append(is_succeed)
        print ("K=%d,Final atk succeed: %.4f"%(k, np.mean(final_succeed)))
        results[k]={'result':[np.mean([p>CUTOFF for p in all_init]),np.mean([p>CUTOFF for p in all_final[k]]),np.mean([p>CUTOFF for p in flatten_cvrs]),np.mean(final_succeed),np.mean(final_succeed)],'init':all_init,'final':all_final[k],'cover':all_cvrs[k]} 
    with open('./saved_attack.json','w') as outf:
        json.dump(atk_to_save, outf)
    f = open('euler_optc_eva.pkl', 'wb+')
    pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    torch.set_num_threads(6)
    main()#k_num为逃逸边数量

'''
VGRNN(
  (phi_x): Sequential(
    (0): Linear(in_features=8746, out_features=32, bias=True)
    (1): ReLU()
  )
  (encoder): VGAE_Prior(
    (c1): GCNConv(64, 32)
    (relu): ReLU()
    (drop): Dropout(p=0.25, inplace=False)
    (mean): GCNConv(32, 16)
    (std): GCNConv(32, 16)
    (soft): Softplus(beta=1, threshold=20)
  )
  (phi_z): Sequential(
    (0): Linear(in_features=16, out_features=32, bias=True)
    (1): ReLU()
  )
  (recurrent): GraphGRU(
    (sig): Sigmoid()
    (tanh): Tanh()
    (drop): Dropout(p=0.25, inplace=False)
  )
  (prior): Sequential(
    (0): Linear(in_features=32, out_features=32, bias=True)
    (1): ReLU()
  )
  (prior_mean): Sequential(
    (0): Linear(in_features=32, out_features=16, bias=True)
  )
  (prior_std): Sequential(
    (0): Linear(in_features=32, out_features=16, bias=True)
    (1): Softplus(beta=1, threshold=20)
  )
)'''