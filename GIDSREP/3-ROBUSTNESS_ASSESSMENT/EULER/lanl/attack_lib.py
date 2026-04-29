import numpy as np
import torch
from tqdm import tqdm
from loaders.tdata import TData
import pickle

def load_from(path, enc, rnn, DEVICE):
    states = pickle.load(open(path, 'rb'))
    gcn_state = {}
    for k, v in states['states'][0].items():
        gcn_state[k[7:]] = v  # 'module.xxx' -> 'xxx'
    enc.load_state_dict(gcn_state)
    rnn_state = {}
    for k, v in states['states'][1].items():
        rnn_state[k[4:]] = v # 'rnn.xxx' -> 'xxx'
    rnn.load_state_dict(rnn_state)
    rnn_h0 = states['h0'].to(DEVICE)
    return enc, rnn, rnn_h0

def slice_data(data, t):
    eis_t = [data.eis[t]]
    xs_t = data.xs
    ys_t = [data.ys[t]] if hasattr(data, 'ys') else None
    masks_t = data.masks[t] if len(data.masks) > 0 else data.masks
    ews_t = [data.ews[t]]
    cnt_t = [data.cnt[t]]
    new_data = TData(eis_t, xs_t, ys_t, masks_t, ews=ews_t, cnt=cnt_t, node_map=None)
    #print (data)
    #print (new_data)
    return new_data

def evasion_attack(enc, rnn, rnn_h0, data, tgt_t, tgt_u, tgt_v, CUTOFF, K=50, N_choice=10000):
    rnn.train()
    enc.eval()
    extra_edges = {t:[] for t in range(data.T)}
    all_extra_edges = []
    for num_edges in tqdm(range(K)):
        trial_enc_out, raw_adj = enc.forward(data, mask_enum=2, no_grad=False, ret_adj=True, extra_edges=extra_edges)
        trial_rnn_out = rnn.forward(trial_enc_out, rnn_h0)
        pred = enc.decode([tgt_u], [tgt_v], trial_rnn_out[tgt_t])
        print (tgt_t,tgt_u,tgt_v,pred,'added:%d'%num_edges)
        pred.backward()
        all_grad = -torch.stack([raw_adj[t].grad.detach() for t in range(len(raw_adj))])
        print (all_grad.shape)
        N_t, N_u, _ = all_grad.shape
        for num_index, index in enumerate(torch.topk(all_grad.view(-1), N_choice, largest=False).indices):
            new_t = (index // N_u // N_u).item()
            new_u = ((index // N_u) % N_u).item()
            new_v = (index % N_u).item()
            #if True:
            #print ("NEW", new_t, new_u, new_v)
            if raw_adj[new_t][new_u,new_v] == 0: # test - not already exist
                new_score = enc.decode([new_u],[new_v],trial_rnn_out[new_t]).item()
                #print ("NEWSCORE", new_score)
                #if new_score < CUTOFF: # test - stealthy
                if new_score > CUTOFF: # test - stealthy
                    can_use = True
                    for prev_t, es in extra_edges.items(): # test - previous stealthy
                        for prev_u, prev_v in es:
                            prev_score = enc.decode([prev_u], [prev_v], trial_rnn_out[prev_t]).item()
                    ##        if prev_score > CUTOFF:
                            if prev_score < CUTOFF:
                                can_use = False
                    if can_use:
                        #print ("USE")
                        extra_edges[new_t].append((new_u, new_v))
                        all_extra_edges.append((new_t, new_u, new_v))
                        break
        #print (extra_edges)
        #print (all_extra_edges)
        del all_grad
        for t in range(len(raw_adj)):
            del raw_adj[t].grad
        del raw_adj
        if num_index >= N_choice:
            break

    #print (all_extra_edges)
    #print (extra_edges)
    return all_extra_edges,extra_edges

def poison_attack(train_emb, u_emb, v_emb, K=100, delta=1800):
    u_sims = ((u_emb[None,None]-train_emb)**2).sum(2)
    sim_uids = u_sims.argsort(axis=1)
    v_sims = ((v_emb[None,None]-train_emb)**2).sum(2)
    sim_vids = v_sims.argsort(axis=1)

    chosen_edges = []
    cur_ptrs = {t:[0,0] for t in range(len(train_emb))}

    while len(chosen_edges) < K:
        all_scores = [ u_sims[t,sim_uids[t][uid]]+v_sims[t,sim_vids[t][vid]] for t,(uid,vid) in cur_ptrs.items() ]
        chosen_t = np.argmin(all_scores)
        chosen_edges.append((((chosen_t+0.5)*delta).item(), sim_uids[chosen_t][cur_ptrs[chosen_t][0]].item(), sim_vids[chosen_t][cur_ptrs[chosen_t][1]].item()))
        if np.random.uniform() > 0.5:
            cur_ptrs[chosen_t][0] += 1
        else:
            cur_ptrs[chosen_t][1] += 1
    return chosen_edges

from loaders.load_utils import std_edge_w
from loaders.load_lanl import load_lanl_dist
class ExtraLoader:
    def __init__(self, extra_edges):
        self.extra_edges = extra_edges
    def load_lanl_with_extra(self, workers, start=0, end=635015, delta=8640, is_test=False, ew_fn=std_edge_w):
        print ("LOADING EXTRA WITH %d EDGES"%len(self.extra_edges))
        init_data = load_lanl_dist(workers=workers, start=start, end=end, delta=delta, is_test=is_test, ew_fn=ew_fn)
        for new_t, new_u, new_v in self.extra_edges:
            if new_t < start or new_t > end:
                continue
            new_idx = int((new_t-start) // delta)
            init_data.eis[new_idx] = torch.cat([init_data.eis[new_idx], torch.LongTensor([[new_u],[new_v]])], 1)
            #init_data.ews[new_idx] = torch.cat([init_data.ews[new_idx], init_data.ews[new_idx].mean(0,keepdim=True)], 0)
            init_data.ews[new_idx] = torch.cat([init_data.ews[new_idx], torch.FloatTensor([1.0])], 0)
            init_data.masks[new_idx] = torch.cat([init_data.masks[new_idx], torch.BoolTensor([True])], 0)
        init_data.ei_sizes = [
            (
                init_data.ei_masked(init_data.TRAIN, t).size(1), 
                init_data.ei_masked(init_data.VAL, t).size(1),
                init_data.eis[t].size(1)
            ) 
            for t in range(init_data.T)
        ]
        return init_data

#def construct_loader_with_extra(extra_edges):
#    from loaders.load_lanl import load_lanl_dist
#    from loaders.load_utils import std_edge_w
#    def load_lanl_with_extra(workers, start=0, end=635015, delta=8640, is_test=False, ew_fn=std_edge_w):
#        init_data = load_lanl_dist(workers=workers, start=start, end=end, delta=delta, is_test=is_test, ew_fn=ew_fn)
#        for new_t, new_u, new_v in extra_edges:
#            print ("========")
#            print (new_t, new_u, new_v)
#            print (init_data.eis[new_t].shape)
#            init_data.eis[new_t] = torch.cat([init_data.eis[new_t], torch.LongTensor([[new_u],[new_v]])], 1)
#            init_data.ews[new_t] = torch.cat([init_data.ews[new_t], init_data.ews[new_t].mean(0,keepdim=True)], 0)
#            init_data.masks[new_t] = torch.cat([init_data.masks[new_t], torch.BoolTensor([True])], 0)
#            print (init_data.eis[new_t].shape)
#        init_data.ei_sizes = [
#            (
#                init_data.ei_masked(init_data.TRAIN, t).size(1), 
#                init_data.ei_masked(init_data.VAL, t).size(1),
#                init_data.eis[t].size(1)
#            ) 
#            for t in range(init_data.T)
#        ]
#        return init_data
#
#    return load_lanl_with_extra
