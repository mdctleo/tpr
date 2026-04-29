import numpy as np
import torch
from tqdm import tqdm
from loaders.tdata import TData
import pickle
import copy
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
    #rnn.train()
    extra_edges = {t:[] for t in range(data.T)}
    all_extra_edges = []
    # print(data.eis)
    # print(data.ews)
    # trial_enc_out, raw_adj1 = enc.forward(data, mask_enum=2, no_grad=False, ret_adj=True, extra_edges=extra_edges)
    # trial_rnn_out = rnn.forward(trial_enc_out, rnn_h0)
    # print(raw_adj1[0][173][1])
    # print(enc.decode([tgt_u], [tgt_v], trial_rnn_out[tgt_t]).item())
    # trial_enc_out, raw_adj2 = enc.forward(data, mask_enum=2, no_grad=False, ret_adj=True, extra_edges={0:[(173,1)]})
    # trial_rnn_out = rnn.forward(trial_enc_out, rnn_h0)
    # print(raw_adj2[0][173][1])
    # print(enc.decode([tgt_u], [tgt_v], trial_rnn_out[tgt_t]).item())
    # assert 0
    for num_edges in tqdm(range(K)):
        rnn.train()
        #rnn.eval()
        trial_enc_out, raw_adj = enc.forward(data, mask_enum=2, no_grad=False, ret_adj=True, extra_edges=extra_edges)
        trial_rnn_out = rnn.forward(trial_enc_out, rnn_h0)
        pred = enc.decode([tgt_u], [tgt_v], trial_rnn_out[tgt_t])
        #print (tgt_t,tgt_u,tgt_v,pred,'added:%d'%num_edges)
        pred.backward()
        all_grad = -torch.stack([raw_adj[t].grad.detach() for t in range(len(raw_adj))])
        #print (all_grad.shape)
        N_t, N_u, _ = all_grad.shape
        for num_index, index in enumerate(torch.topk(all_grad.view(-1), N_choice, largest=False).indices):
            new_t = (index // N_u // N_u).item()
            new_u = ((index // N_u) % N_u).item()
            new_v = (index % N_u).item()
            #if True:
            #print ("NEW", new_t, new_u, new_v)
            rnn.eval()
            if raw_adj[new_t][new_u,new_v] == 0 and (new_u,new_v) not in extra_edges[new_t]: # test - not already exist
                new_score = enc.decode([new_u],[new_v],trial_rnn_out[new_t]).item()
                #print ("NEWSCORE", new_score)
                #if new_score < CUTOFF: # test - stealthy
                if new_score > CUTOFF: # test - stealthy
                    #print(enc.decode([tgt_u],[tgt_v],trial_rnn_out[tgt_t]).item())
                    can_use = True
                    # tmp_edges=copy.deepcopy(extra_edges)
                    # tmp_edges[new_t].append((new_u, new_v))
                    # rnn.eval()
                    # for prev_t, es in tmp_edges.items():
                    #     for prev_u, prev_v in es:
                    #         tmp_enc_out, _ = enc.forward(data, mask_enum=2, no_grad=False, ret_adj=True, extra_edges=tmp_edges)
                    #         tmp_rnn_out = rnn.forward(tmp_enc_out, rnn_h0)
                    #         prev_score = enc.decode([prev_u], [prev_v], tmp_rnn_out[prev_t]).item()
                    #         if prev_score < CUTOFF:
                    #             can_use = False
                    #for prev_t, es in extra_edges.items(): # test - previous stealthy
                    #    for prev_u, prev_v in es:
                    #        prev_score = enc.decode([prev_u], [prev_v], trial_rnn_out[prev_t])
                    ##        if prev_score > CUTOFF:
                    #        if prev_score < CUTOFF:
                    #            can_use = False
                    if can_use:
                        #print ("USE")
                        #print(all_grad.view(-1)[index])
                        #print(enc.decode([tgt_u],[tgt_v], tmp_rnn_out[prev_t]).item())
                        #print(enc.decode([173],[810], tmp_rnn_out[prev_t]).item())
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
    return all_extra_edges