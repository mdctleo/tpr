from copy import deepcopy
import torch
from torch import nn
import numpy as np
from torch_geometric.utils import to_dense_adj
import random
from libauc.losses import APLoss
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GCNConv, GATConv, NNConv, SAGEConv
from torch_geometric.nn.conv.message_passing import MessagePassing


class Euler_Embed_Unit(nn.Module):
    def inner_forward(self, mask_enum):
        raise NotImplementedError

    def forward(self, mask_enum, no_grad):
        if no_grad:
            self.eval()
            with torch.no_grad():
                return self.inner_forward(mask_enum)
        return self.inner_forward(mask_enum)


    def decode(self, src, dst, z):
        return torch.sigmoid(
            (z[src] * z[dst]).sum(dim=1)
        )

    #LoF to get the anomaly scores, clf is the LoF class object with initialized params
    def lof(self, src, dst, z, clf):
        edge_z = z[src] * z[dst]
        lof_labels = clf.fit_predict(edge_z)
        lof_scores = clf.negative_outlier_factor_
        return lof_labels,lof_scores

    #Adjust the edge prediction score based on other edges sharing the same src (excluding the edge)
    def get_src_score(self, src, dst, preds):
        src_dict = {}
        for i in range(0, len(src)):
            k = int(src[i])
            if not k in src_dict:
                src_dict[k] = [float(preds[i])]
            else:
                src_dict[k].append(float(preds[i]))
        preds_src = []
        #weights for edge score and neighborhood score
        lambda1 = 0.5
        lambda2 = 0.5
        for i in range(0, len(src)):
            k = int(src[i])
            preds_src.append(lambda1 * float(preds[i]) + lambda2 * np.mean(src_dict[k]))
        return torch.tensor(preds_src)


class GCN(Euler_Embed_Unit):
    def __init__(self, data_load, data_kws, h_dim, z_dim, device):
        super(GCN, self).__init__()
        self.device = device
        self.data = data_load(**data_kws).to(self.device)
        self.c1 = GCNConv(self.data.x_dim, h_dim, add_self_loops=True)
        self.relu = nn.ReLU()
        self.c2 = GCNConv(h_dim, z_dim, add_self_loops=True)
        self.drop = nn.Dropout(0.25)
        self.tanh = nn.Tanh()
        self.de = DropEdge(0.5)


    def inner_forward(self, mask_enum):
        zs = []
        for i in range(self.data.T):
            zs.append(
                self.forward_once(mask_enum, i)#.cpu().detach()
            )
        return torch.stack(zs)


    def forward_once(self, mask_enum, i):
        if self.data.dynamic_feats:
            x = self.data.xs[i].to(self.device)
        else:
            x = self.data.xs.to(self.device)

        ei = self.data.ei_masked(mask_enum, i).to(self.device)
        ew = self.data.ew_masked(mask_enum, i).to(self.device)

        x = self.c1(x, ei, edge_weight=ew)
        x = self.relu(x)
        x = self.drop(x)
        x = self.c2(x, ei, edge_weight=ew)

        # Experiments have shown this is the best activation for GCN+GRU
        return self.tanh(x)

class Argus_OPTC(GCN):
    def __init__(self, data_load, data_kws, h_dim, z_dim, device):
        super().__init__(data_load, data_kws, h_dim, z_dim, device)
        self.de = DropEdge(0.5)
        self.c1 = GCNConv(self.data.x_dim, h_dim, add_self_loops=True)
        self.c2 = GCNConv(h_dim, h_dim, add_self_loops=True)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.1)
        self.c3 = GCNConv(h_dim, z_dim, add_self_loops=True)
        self.c4 = GCNConv(h_dim, z_dim, add_self_loops=True)
        self.ac = nn.Tanh()

    def forward_once(self, mask_enum, i):
        if self.data.dynamic_feats:
            x = self.data.xs[i].to(self.device)
        else:
            x = self.data.xs.to(self.device)
        ei = self.data.ei_masked(mask_enum, i).to(self.device)
        ew = self.data.ew_masked(mask_enum, i).to(self.device)
        ei, ew = self.de(ei, ewa=ew) # increase 2%
        x = self.c1(x, ei, edge_weight=ew)
        x = self.c2(x, ei, edge_weight=ew)
        x = self.relu(x)
        x = self.drop(x)
        x = self.c3(x, ei, edge_weight=ew)
        x = self.relu(x)
        x = self.drop(x)
        x = self.c4(x, ei, edge_weight=ew)
        return self.ac(x)


def detector_optc_rref(loader, kwargs, h_dim, z_dim, **kws):
    device = kwargs.pop('device')
    return DetectorEncoder(Argus_OPTC(loader, kwargs, h_dim, z_dim, device), device, 'OPTC')


class Argus_LANL(GCN):
    def __init__(self, data_load, data_kws, h_dim, z_dim, device):
        super().__init__(data_load, data_kws, h_dim, z_dim, device)
        self.c1 = GCNConv(self.data.x_dim, h_dim, add_self_loops=True)
        self.relu = nn.ReLU()
        self.c2 = GCNConv(h_dim, h_dim, add_self_loops=True)
        self.drop = nn.Dropout(0.1)
        self.ac = nn.Tanh()
        self.c3 = GCNConv(h_dim, z_dim, add_self_loops=True)
        # NNConv always uses 6-dim flow features only (KDE handled at decode)
        nn4 = nn.Sequential(nn.Linear(6, 8), nn.ReLU(),
                            nn.Linear(8, h_dim * z_dim))
        self.c4 = NNConv(h_dim, z_dim, nn4, aggr='mean')


    def forward_once(self, mask_enum, i):
        if self.data.dynamic_feats:
            x = self.data.xs[i].to(self.device)
        else:
            x = self.data.xs.to(self.device)

        ei = self.data.ei_masked(mask_enum, i).to(self.device)
        ea = self.data.ea_masked(mask_enum, i).to(self.device)
        ew = self.data.ew_masked(mask_enum, i).to(self.device)
        ea = torch.transpose(ea, 0, 1)
        x1 = self.c1(x, ei, edge_weight=ew)
        x = self.c2(x1, ei, edge_weight=ew)
        x = self.relu(x)
        x = self.drop(x)
        x = self.c3(x, ei, edge_weight=ew)
        x = self.relu(x)
        x = self.drop(x)
        x = self.c4(x, ei, edge_attr=ea)
        return self.ac(x)

def detector_lanl_rref(loader, kwargs, h_dim, z_dim, **kws):
    device = kwargs.pop('device')
    return DetectorEncoder(Argus_LANL(loader, kwargs, h_dim, z_dim, device), device, 'LANL')

class DetectorEncoder(Euler_Embed_Unit):
    def __init__(self, module: Euler_Embed_Unit, device, dataset, **kwargs):
        super().__init__(**kwargs)
        self.module = module
        self.sample_num = 5
        self.zdim = 16
        self.out = nn.Sequential(nn.Linear(self.zdim, self.zdim), nn.Softmax(dim=1))
        if dataset.startswith('L'):
            self.num_nodes = 19129
            self.ap_loss = APLoss(data_len=2830742, margin=0.8, gamma=0.1, surr_loss='squared_hinge', device=device)
        elif dataset.startswith('O'):
            self.num_nodes = 19129
            self.ap_loss = APLoss(data_len=2830742, margin=0.8, gamma=0.1, surr_loss='squared_hinge', device=device)
        # Option B (additive): KDE_MLP provides a scalar adjustment added to
        # the baseline dot-product score.  The MLP sees ONLY the KDE vector,
        # NOT the z embeddings, so it cannot memorise the link-prediction task.
        #   score = sigmoid( z_src · z_dst  +  KDE_MLP(kde_vec) )
        kde_dim = getattr(module.data, 'kde_dim', 0) if hasattr(module, 'data') else 0
        self._kde_dim = kde_dim
        if kde_dim > 0:
            mlp_h = max(kde_dim // 2, 8)          # 10 → 8
            self.kde_mlp = nn.Sequential(
                nn.Linear(kde_dim, mlp_h),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(mlp_h, 1)
            )
            # Fix 4: zero-init output layer so KDE starts contributing nothing
            nn.init.zeros_(self.kde_mlp[-1].weight)
            nn.init.zeros_(self.kde_mlp[-1].bias)
    def train(self, mode=True):
        self.module.train()


    def load_new_data(self, loader, kwargs):
        self.module.data = loader(**kwargs)
        #import pickle
        #pickle.dump(self.module.data, open('optc_data.pickle', 'wb+'), protocol=pickle.HIGHEST_PROTOCOL)
        return True

    def get_data_field(self, field):
        return self.module.data.__getattribute__(field)


    def get_data(self):
        return self.module.data


    def run_arbitrary_fn(self, fn, *args, **kwargs):
        return fn(*args, **kwargs)


    def decode(self, e, z, no_grad):
        src, dst = e
        # Raw dot-product (same as baseline) — wider score distribution than
        # cosine+tau, which compressed everything into a tiny band near 1.0.
        base_score = (z[src] * z[dst]).sum(dim=1)                         # (E,)
        # Option B additive: KDE MLP provides a small scalar adjustment.
        if self._kde_dim > 0:
            kde_vecs = self.module.data.get_kde_tensor(src, dst)          # (E, K)
            if kde_vecs is not None:
                kde_vecs = kde_vecs.to(z.device)
                kde_adj = self.kde_mlp(kde_vecs).squeeze(1)              # (E,)
                kde_adj = torch.clamp(kde_adj, -1.0, 1.0)               # Fix 4: prevent extreme shifts
                return torch.sigmoid(base_score + kde_adj)
        return torch.sigmoid(base_score)


    def inner_forward_sm(self, z, s):
        temp = torch.matmul(s, z)
        return self.out(1 / float(self.sample_num) * temp)

    def lof(self, e,z,clf):
        src,dst = e
        return self.module.lof(src,dst,z,clf)

    def get_src_score(self, e, preds):
        src,dst = e
        return self.module.get_src_score(src,dst,preds)

    def bce(self, t_scores, f_scores):
        EPS = 1e-6
        pos_loss = -torch.log(t_scores+EPS).mean()
        neg_loss = -torch.log(1-f_scores+EPS).mean()
        return (pos_loss + neg_loss) * 0.5

    def decode_all(self, zs, unsqueeze=False):
        assert not zs.size(0) < self.module.data.T, \
            "%s was given fewer embeddings than it has time slices"\

        assert not zs.size(0) > self.module.data.T, \
            "%s was given more embeddings than it has time slices"\

        preds = []
        ys = []
        cnts = []
        eis_out = []
        for i in range(self.module.data.T):
            preds_i = self.decode(self.module.data.eis[i], zs[i], False)
            preds.append(preds_i.detach().cpu().numpy())

            ys.append(self.module.data.ys[i].detach().cpu().numpy())
            cnts.append(self.module.data.cnt[i].detach().cpu().numpy())
            eis_out.append(self.module.data.eis[i].detach().cpu().numpy())

        return preds, ys, cnts, eis_out

    def score_edges(self, z, partition, nratio): # for validation
        n = self.module.data.get_negative_edges(partition, nratio)

        p_scores = []
        n_scores = []

        for i in range(len(z)):
            p = self.module.data.ei_masked(partition, i)
            if p.size(1) == 0:
                continue

            p_scores.append(self.decode(p, z[i], False))
            n_scores.append(self.decode(n[i], z[i], False))

        p_scores = torch.cat(p_scores, dim=0)
        n_scores = torch.cat(n_scores, dim=0)

        return p_scores, n_scores

    def calc_loss(self, z, partition, nratio, device):
        tot_loss = torch.zeros(1).to(device)
        ns = self.module.data.get_negative_edges(partition, nratio)

        for i in range(len(z)):
            ps = self.module.data.ei_masked(partition, i)
            if ps.size(1) == 0:
                continue
            tot_loss += self.bce(
                self.decode(ps, z[i], False),
                self.decode(ns[i], z[i], False))
        return tot_loss.true_divide(len(z))


    def calc_loss_argus(self, z, partition, nratio, device):
        tot_loss = torch.zeros(1).to(device)
        ns = self.module.data.get_negative_edges(partition, nratio)
        for i in range(len(z)):
            ps = self.module.data.ei_masked(partition, i)
            if ps.size(1) == 0:
                continue
            t_index = torch.arange(0, ps.shape[1], dtype=torch.int64).to(device).detach()
            pos_pred = self.decode(ps, z[i], False)
            neg_pred = self.decode(ns[i], z[i], False)

            tot_loss += self.ap_loss(
                torch.cat((pos_pred, neg_pred), 0),
                torch.cat((torch.ones(pos_pred.shape[0]),torch.zeros(neg_pred.shape[0])), 0).to(self.module.device).detach(),
                t_index)
        return tot_loss.true_divide(len(z))

class DropEdge(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, ei, ewa=None):
        #either edge weights or edge attributes (ewa can be either ew or ea)
        if self.training and self.p > 0:
            mask = torch.rand(ei.size(1))
            if ewa is None:
                return ei[:, mask > self.p]
            else:
                #accommodate edge weights and features
                if len(ewa.size()) == 1:
                    return ei[:, mask > self.p], ewa[mask > self.p]
                else:
                    return ei[:, mask > self.p], ewa[:, mask > self.p]

        if ewa is None:
            return ei
        else:
            return ei, ewa

class Argus(nn.Module):
    def __init__(self, rnn: nn.Module, encoder: nn.Module, device):
        super(Argus, self).__init__()
        self.gcns = encoder
        self.rnn = rnn
        self.device = device
        self.len_from_each = []


    # def forward(self, mask_enum, include_h=False, h0=None, no_grad=False):
    #     futs = self.encode(mask_enum, no_grad)
    #     zs = []
    #     zs, h0 = self.rnn(futs.to(self.device), h0, include_h=True)
    #     self.len_from_each = [embed.size(0) for embed in zs]
    #     self.z_dim = zs.size(-1)
    #     if include_h:
    #         return zs, h0
    #     else:
    #         return zs
    def forward(self, mask_enum, include_h=False, h0=None, no_grad=False):
        futs = self.encode(mask_enum, no_grad)
        zs = []
        for f in futs:
            z, h0 = self.rnn(f.unsqueeze(0).to(self.device), h0, include_h=True)
            zs.append(z)
        self.len_from_each = [embed.size(0) for embed in zs]
        zs = torch.cat(zs, dim=0)
        self.z_dim = zs.size(-1)
        if include_h:
            return zs, h0
        else:
            return zs


    def encode(self, mask_enum, no_grad):
        return self.gcns.module(mask_enum, no_grad)

    def save_states(self):
        gcn = self.gcns
        return gcn, deepcopy(self.state_dict())

    def load_states(self, gcn_state_dict, rnn_state_dict):
        self.load_state_dict(rnn_state_dict)

    def train(self, mode=True):
        super(Argus, self).train()

    def eval(self):
        super(Argus, self).train(False)

    def score_all(self, zs, unsqueeze=False):
        futs = []
        futs = DetectorEncoder.decode_all(self.gcns, zs, unsqueeze)

        obj = [futs] #[f.wait() for f in futs]
        scores, ys, cnts, eis = zip(*obj)

        # Compress into single list of snapshots
        scores = sum(scores, [])
        ys = sum(ys, [])
        cnts = sum(cnts, [])
        eis = sum(eis, [])

        return scores, ys, cnts, eis


    def loss_fn(self, zs, partition, nratio=1, device=None, encoder_name=None):
        futs = []
        start = 0
        if encoder_name == 'ARGUS':
            futs = DetectorEncoder.calc_loss_argus(self.gcns, zs, partition, nratio, device)
        else:
            futs = DetectorEncoder.calc_loss(self.gcns, zs, partition, nratio, device)

        tot_loss = torch.zeros(1).to(device)

        for f in futs:
            tot_loss += f

        return tot_loss


    def score_edges(self, zs, partition, nratio=1):
        futs = []
        futs = DetectorEncoder.score_edges(self.gcns, zs, partition, nratio)
        pos, neg = zip(*[futs])
        return torch.cat(pos, dim=0), torch.cat(neg, dim=0)
