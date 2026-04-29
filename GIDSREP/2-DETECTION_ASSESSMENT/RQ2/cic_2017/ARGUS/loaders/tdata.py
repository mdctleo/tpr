import numpy as np
import pandas as pd
import pickle
import torch
from torch_geometric.data import Data
from torch_geometric.utils import structured_negative_sampling
'''
Special data object that the dist_framework uses
'''
class TData(Data):
    TRAIN = 0
    VAL = 1
    TEST = 2
    ALL = 2

    #eas for edge attributes
    def __init__(self, slices, eis, xs, ys, masks, ews=None, eas=None, use_flows=False, nmap=None, kde_dict=None, **kwargs):
        super(TData, self).__init__(**kwargs)

        # Required fields for models to use this
        self.slices = slices
        self.eis = eis
        self.T = len(eis)
        self.xs = xs
        self.masks = masks
        self.dynamic_feats = isinstance(xs, list)
        self.ews = ews
        self.eas = eas
        self.ys = ys
        self.is_test = not isinstance(ys, None.__class__)
        self.nmap = nmap
        # KDE dict stored in __dict__ directly (PyG GlobalStorage drops None values)
        self.__dict__['kde_dict'] = kde_dict
        if kde_dict is not None and len(kde_dict) > 0:
            _sample = next(iter(kde_dict.values()))
            self.__dict__['kde_dim'] = len(_sample)
        else:
            self.__dict__['kde_dim'] = 0

        # Makes finding sizes of positive samples a little easier
        self.ei_sizes = [
            (
                self.ei_masked(self.TRAIN, t).size(1),
                self.ei_masked(self.VAL, t).size(1),
                self.eis[t].size(1)
            )
            for t in range(self.T)
        ]

        if self.dynamic_feats:
            self.num_nodes = max([x.size(0) for x in xs])
            self.x_dim = xs[0].size(1)
        else:
            self.num_nodes = xs.size(0)
            self.x_dim = xs.size(1)

        #number of edge features — always 6-dim flow (KDE is now at decode step)
        if not isinstance(eas, None.__class__) and len(eas) > 0:
            self.ea_dim = eas[0].size(0)
        elif use_flows:
            self.ea_dim = 5  # legacy fallback
        else:
            self.ea_dim = 0

    '''
    Returns masked ei/ew/ea at timestep t
    Assumes it will only be called on tr or val data
    (i.e. test data is the entirity of certain time steps)
    '''
    def ei_masked(self, enum, t):
        if enum == self.TEST or self.is_test:
            return self.eis[t]
        if enum == self.TRAIN:
            return self.eis[t][:, self.masks[t]]
        else:
            return self.eis[t][:, ~self.masks[t]]

    def ew_masked(self, enum, t):
        if isinstance(self.ews, None.__class__):
            return None

        if enum == self.TEST or self.is_test:
            return self.ews[t]

        return self.ews[t][self.masks[t]] if enum == self.TRAIN \
            else self.ews[t][~self.masks[t]]

    def ea_masked(self, enum, t):
        if isinstance(self.eas, None.__class__):
            return None

        if enum == self.TEST or self.is_test:
            return self.eas[t]

        #To implement, eas is edge attr and have different dimensions
        return self.eas[t][:, self.masks[t]] if enum == self.TRAIN \
            else self.eas[t][:, ~self.masks[t]]


    def get_kde_tensor(self, src_t, dst_t, fallback=None):
        """Return (E, kde_dim) float tensor of KDE vectors for edge list.

        Improvement 3: When `fallback` is an nn.Parameter, unseen edges
        receive the learned fallback vector instead of hard zeros.
        src_t, dst_t: 1-D integer tensors of node IDs.
        """
        if self.kde_dim == 0 or self.kde_dict is None:
            return None
        vecs = []
        for s, d in zip(src_t.tolist(), dst_t.tolist()):
            v = self.kde_dict.get((int(s), int(d)))
            if v is not None:
                vecs.append(torch.tensor(list(v), dtype=torch.float32))
            elif fallback is not None:
                vecs.append(fallback)  # nn.Parameter, stays in graph
            else:
                vecs.append(torch.zeros(self.kde_dim))
        return torch.stack(vecs)

    def kde_ea_masked(self, enum, t):
        """Improvement 5: Build (kde_dim, E_masked) KDE edge-attr tensor for snapshot t.
        Returns None if no KDE dict is available.
        """
        if self.kde_dim == 0 or self.kde_dict is None:
            return None
        ei = self.ei_masked(enum, t)
        src_t, dst_t = ei[0], ei[1]
        # get_kde_tensor returns (E, kde_dim); transpose to (kde_dim, E) for NNConv
        kde_t = self.get_kde_tensor(src_t, dst_t)  # no fallback here — zeros for NNConv
        return kde_t.t() if kde_t is not None else None

    def get_negative_edges(self, enum, nratio=1, start=0):
        negs = []
        size = []
        for t in range(start, self.T):
            if enum == self.TRAIN:
                pos = self.ei_masked(enum, t)
            else:
                pos = self.eis[t]

            num_pos = self.ei_sizes[t][enum]
            negs.append(fast_negative_sampling(pos, int(num_pos*nratio),self.num_nodes))
            size.append(negs[-1].size(1))
        size = sum(size)
        return negs



    def get_val_repr(self, scores, delta=1):
        pairs = []
        for i in range(len(scores)):
            score = scores[i]
            ei = self.eis[i]

            for j in range(ei.size(1)):
                if self.nmap is not None:
                    src, dst = self.nmap[ei[0,j]], self.nmap[ei[1,j]]
                else:
                    src, dst = ei[0,j], ei[1,j]
                if self.hr:
                    ts = self.hr[i]
                else:
                    ts = '%d-%d' % (i*delta, (i+1)*delta)

                s = '%s\t%s\t%s' % (src, dst, ts)
                pairs.append((score[j], s))

        pairs.sort(key=lambda x : x[0])
        return pairs

'''
Uses Kipf-Welling pull #25 to quickly find negative edges
(For some reason, this works a touch better than the builtin
torch geo method)
'''
def fast_negative_sampling(edge_list, batch_size, num_nodes, oversample=1.25):
    # For faster membership checking
    el_hash = lambda x : x[0,:] + x[1,:]*num_nodes

    el1d = el_hash(edge_list).cpu().numpy()
    neg = np.array([[],[]])

    while(neg.shape[1] < batch_size):
        maybe_neg = np.random.randint(0,num_nodes, (2, int(batch_size*oversample))) #generates a 2d matrix
        neg_hash = el_hash(maybe_neg)

        neg = np.concatenate(
            [neg, maybe_neg[:, ~np.in1d(neg_hash, el1d)]],
            axis=1
        )
    neg = neg[:, :batch_size]
    return torch.tensor(neg).long()
