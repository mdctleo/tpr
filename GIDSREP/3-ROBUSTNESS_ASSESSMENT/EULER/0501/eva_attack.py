import numpy as np
import torch
import torch.nn as nn
import pickle

from models.recurrent import GRU,EmptyModel
from models.embedders import DropEdge
import loaders.load_0501 as lanl
from tqdm import tqdm
from attack_lib import evasion_attack,slice_data
import json

from sklearn.metrics import \
    roc_auc_score as auc_score, \
    f1_score, average_precision_score as ap_score

MODEL_PATH='model_save_0501.pkl'
CUTOFF = 0.5857
DEVICE = 'cpu'#'cuda'

#from torch_geometric.nn import GCNConv
from torch_geometric.nn.dense.linear import Linear
class GCNConv(nn.Module):
    # Customized GCNConv so that gradient can propagate
    def __init__(self, in_channels, out_channels, add_self_loops=True, bias=True):
        super(GCNConv, self).__init__()
        assert add_self_loops
        self.lin = Linear(in_channels, out_channels, bias=False, weight_initializer='glorot')
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
    def forward(self, x, adj):
        x = self.lin(x)
        out = adj @ x
        if self.bias is not None:
            out += self.bias
        return out

class GCN_local(nn.Module):
    '''
    2-layer GCN implimenting the Euler Embed Unit interface
    '''

    def __init__(self, h_dim, z_dim, device='cpu'):
        '''
        Constructor for the model 

        parameters
        ----------
        data_load : callable[..., loaders.TGraph]
            Function to load data onto this worker. Must return a loaders.TGraph object
        data_kws : dict 
            Dictionary of keyword args for the loader function 
        h_dim : int 
            The dimension of the hidden layer
        z_dim : int 
            The dimension of the output layer

        attributes
        ----------
        data : loaders.TGraph 
            A TGraph object holding data for all snapshots loaded into this model 
        '''
        super(GCN_local, self).__init__()
        self.device = device

        # Params 
        #self.c1 = GCNConv(data.x_dim, h_dim, add_self_loops=True)
        self.c1 = GCNConv(8746, h_dim, add_self_loops=True)
        self.relu = nn.ReLU()
        self.c2 = GCNConv(h_dim, z_dim, add_self_loops=True)
        self.drop = nn.Dropout(0.25)
        self.tanh = nn.Tanh()
        self.de = DropEdge(0.8)

    def forward(self, data, mask_enum, no_grad, ret_adj=False, extra_edges=None):
        if no_grad:
            self.eval()
            with torch.no_grad():
                return self.inner_forward(data, mask_enum, ret_adj=ret_adj, extra_edges=extra_edges)
        return self.inner_forward(data, mask_enum, ret_adj=ret_adj, extra_edges=extra_edges)

    def inner_forward(self, data, mask_enum, ret_adj=False, extra_edges=None):
        '''
        Override parent's abstract inner_forward method

        mask_enum : int
            enum representing train, validation, test used to mask which 
            edges are sent into the model
        '''
        zs = []
        adjs = []
        #for i in tqdm(range(data.T)):
        for i in range(data.T):
            # Small optimization. Running each loop step as its own thread
            # is a tiny bit faster. 
            extra = extra_edges[i] if extra_edges is not None else None
            if ret_adj:
                z, adj = self.forward_once(data, mask_enum, i, ret_adj=ret_adj, extra=extra)
                zs.append(z)
                adjs.append(adj)
            else:
                zs.append(
                    #torch.jit._fork(self.forward_once, mask_enum, i)
                    self.forward_once(data, mask_enum, i, ret_adj=ret_adj, extra=extra)
                )

        #return torch.stack([torch.jit._wait(z) for z in zs])
        if ret_adj:
            return torch.stack(zs), adjs
        else:
            return torch.stack(zs)

    def forward_once(self, data, mask_enum, i, ret_adj=False, extra=None):
        '''
        Helper function to make inner_forward a little more readable 
        Just passes each time step through a 2-layer GCN with final tanh activation

        mask_enum : int 
            enum representing train, validation, test 
            used to mask edges passed into model 
        i : int
            The index of the snapshot being processed
        '''
        if data.dynamic_feats:
            x = data.xs[i]
        else:
            x = data.xs 
        ei = data.ei_masked(mask_enum, i)
        ew = data.ew_masked(mask_enum, i)
        if self.device != 'cpu':
            x = x.to(self.device)
            ei = ei.to(self.device)
            ew = ew.to(self.device)
        ei, ew = self.de(ei, ew=ew)
        if extra is not None and len(extra) > 0:
            ei = torch.cat( (ei, torch.LongTensor(extra).to(self.device).T), 1)
            ew = torch.cat( (ew, torch.ones(len(extra), device=self.device)*ew.mean()), 0)
        # Simple 2-layer GCN. Tweak if desired
        raw_adj, normed_adj = self.gcn_norm_adj(ei, ew, len(x), need_grad=ret_adj)
        x = self.c1(x, normed_adj)
        #x = self.c1(x, ei, edge_weight=ew)
        #print ('x1', x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.c2(x, normed_adj)
        #x = self.c2(x, ei, edge_weight=ew)
        #print ('x2', x)
        # Experiments have shown this is the best activation for GCN+GRU
        if ret_adj:
            return self.tanh(x), raw_adj
        else:
            return self.tanh(x)

    def gcn_norm_adj(self, ei, ew, N_node, need_grad):
        import time
        raw_adj = torch.eye(N_node, device=self.device)
        raw_adj[ei[0],ei[1]] = ew
        if need_grad:
            raw_adj.requires_grad_()  # TODO: hard-coded here for gradient propagation!
        deg_inv_sqrt = raw_adj.sum(0).pow_(-0.5)
        normed_adj = (raw_adj * deg_inv_sqrt.unsqueeze(0) * deg_inv_sqrt.unsqueeze(1)).T
        return raw_adj, normed_adj

    def decode(self, src, dst, z):
        return torch.sigmoid(
            (z[src] * z[dst]).sum(dim=1)
        )

def score_stats(title, scores, labels, weights, cutoff, ctime):
    # Cat scores from timesteps together bc separation 
    # is no longer necessary 
    scores = torch.cat(scores, dim=0).detach()
    labels = torch.cat(labels, dim=0).clamp(max=1)
    weights = torch.cat(weights, dim=0)

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
 
    data = lanl.load_partial_lanl(start=72378559, end=86674422, delta=150000, is_test=True)

    enc = GCN_local(32, 32, device=DEVICE).to(DEVICE)
    gcn_state = {}
    for k, v in states['states'][0].items():
        gcn_state[k[7:]] = v  # 'module.xxx' -> 'xxx'
    enc.load_state_dict(gcn_state)
    enc.eval()
    for p in enc.parameters():
        p.requires_grad = False
    rnn = GRU(32, 32, 16).to(DEVICE)
    rnn_state = {}
    for k, v in states['states'][1].items():
        rnn_state[k[4:]] = v # 'rnn.xxx' -> 'xxx'
    rnn.load_state_dict(rnn_state)
    rnn.train() # cudnn rnn backward only support training
    for p in rnn.parameters():
        p.requires_grad = False
    rnn.eval()
    rnn_h0 = states['h0'].to(DEVICE)


    ### Without attack
    # trial_enc_out = enc.forward(data, mask_enum=2, no_grad=True)
    # print (trial_enc_out.shape)
    # trial_rnn_out = rnn.forward(trial_enc_out, rnn_h0)
    # #print(trial_rnn_out)
    # print (trial_rnn_out.shape)
    preds = []
    labels = []
    cnts = []
    for t in tqdm(range(data.T)):
        data_t = slice_data(data, t)
        trial_enc_out = enc.forward(data_t, mask_enum=2, no_grad=True)
        trial_rnn_out = rnn.forward(trial_enc_out, rnn_h0)
        pred = enc.decode(data_t.eis[0][0], data_t.eis[0][1], trial_rnn_out[0])
        preds.append(pred.cpu())
        labels.append(data_t.ys[0].cpu())
        cnts.append(data_t.cnt[0].cpu())
    #from spinup import score_stats
    score_stats('TITLE', preds, labels, cnts, cutoff=CUTOFF, ctime=None)
    #assert 0
    #'''
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

        rnn.eval()
        enc_out, raw_adj = enc.forward(data_t, mask_enum=2, no_grad=False, ret_adj=True)
        rnn_out = rnn.forward(enc_out, rnn_h0)
        init_pred = enc.decode([tgt_u], [tgt_v], rnn_out[0]).item()
        all_init.append(init_pred)

        #extra_edges = evasion_attack(enc, rnn, rnn_h0, data, tgt_t, tgt_u, tgt_v, CUTOFF, K=5, N_choice=10)
        all_extra_edges = evasion_attack(enc, rnn, rnn_h0, data_t, 0, tgt_u, tgt_v, CUTOFF, K=max(ALL_KS), N_choice=100000)
        if len(all_extra_edges) == 0:
                mise.append((tgt_t, tgt_u, tgt_v))
                continue
        atk_to_save['goal'].append((tgt_t, tgt_u, tgt_v))
        atk_to_save['edge'].append(all_extra_edges)
        rnn.eval()
        print(all_extra_edges)
        for k in ALL_KS:
            extra_edges = {t:[] for t in range(data_t.T)}
            for one_t, one_u, one_v in all_extra_edges[:k]:
                    extra_edges[one_t].append((one_u, one_v))
            enc_out, raw_adj = enc.forward(data_t, mask_enum=2, no_grad=False, ret_adj=True, extra_edges=extra_edges)
            rnn_out = rnn.forward(enc_out, rnn_h0)
            final_pred = enc.decode([tgt_u], [tgt_v], rnn_out[0]).item()
            cover_preds = []
            for prev_t, es in extra_edges.items():
                for prev_u, prev_v in es:
                    cover_preds.append(enc.decode([prev_u], [prev_v], rnn_out[prev_t]).item())
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
    main()#k_num为逃逸边数量
