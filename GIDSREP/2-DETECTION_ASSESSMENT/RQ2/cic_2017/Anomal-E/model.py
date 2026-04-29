from urllib.parse import urlparse
def get_host(url: str) -> (str, str, str):
	parsed_url = urlparse(url)
	return parsed_url.scheme, parsed_url.netloc, parsed_url.path
import urllib3
urllib3.get_host = get_host

import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import torch
import math
#import tqdm
#import gc

class SAGELayer(nn.Module):
    def __init__(self, ndim_in, edims, ndim_out, activation):
      super(SAGELayer, self).__init__()
      self.W_apply = nn.Linear(ndim_in + edims , ndim_out)
      self.activation = F.relu
      self.W_edge = nn.Linear(ndim_out*2, ndim_out*2)
      self.reset_parameters()

    def reset_parameters(self):
      gain = nn.init.calculate_gain('relu')
      nn.init.xavier_uniform_(self.W_apply.weight, gain=gain)

    def message_func(self, edges):
      return {'m':  edges.data['h']}

    def forward(self, g_dgl, nfeats, efeats):
      with g_dgl.local_scope():
        g = g_dgl
        g.ndata['h'] = nfeats
        g.edata['h'] = efeats
        g.update_all(self.message_func, fn.mean('m', 'h_neigh'))
        g.ndata['h'] = F.relu(self.W_apply(torch.cat([g.ndata['h'], g.ndata['h_neigh']], 2)))

        # Compute edge embeddings
        u, v = g.edges()
        edge = self.W_edge(torch.cat((g.srcdata['h'][u], g.dstdata['h'][v]), 2))
        return g.ndata['h'], edge

class SAGE(nn.Module):
    def __init__(self, ndim_in, ndim_out, edim,  activation):
      super(SAGE, self).__init__()
      self.layers = nn.ModuleList()
      self.layers.append(SAGELayer(ndim_in, edim, ndim_out, F.relu))

    def forward(self, g, nfeats, efeats, corrupt=False):
      if corrupt:
        e_perm = torch.randperm(g.number_of_edges())
        #n_perm = torch.randperm(g.number_of_nodes())
        efeats = efeats[e_perm]
        #nfeats = nfeats[n_perm]
      for i, layer in enumerate(self.layers):
        #nfeats = layer(g, nfeats, efeats)
        nfeats, e_feats = layer(g, nfeats, efeats)
      #return nfeats.sum(1)
      return nfeats.sum(1), e_feats.sum(1)

class Discriminator(nn.Module):
    def __init__(self, n_hidden):
      super(Discriminator, self).__init__()
      self.weight = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
      self.reset_parameters()

    def uniform(self, size, tensor):
      bound = 1.0 / math.sqrt(size)
      if tensor is not None:
        tensor.data.uniform_(-bound, bound)

    def reset_parameters(self):
      size = self.weight.size(0)
      self.uniform(size, self.weight)

    def forward(self, features, summary):
      features = torch.matmul(features, torch.matmul(self.weight, summary))
      return features

class DGI(nn.Module):
    def __init__(self, ndim_in, ndim_out, edim, activation):
      super(DGI, self).__init__()
      self.encoder = SAGE(ndim_in, ndim_out, edim,  F.relu)
      #self.discriminator = Discriminator(128)
      self.discriminator = Discriminator(ndim_out*2)
      self.loss = nn.BCEWithLogitsLoss()

    def forward(self, g, n_features, e_features):
      positive = self.encoder(g, n_features, e_features, corrupt=False)
      negative = self.encoder(g, n_features, e_features, corrupt=True)
      self.loss = nn.BCEWithLogitsLoss()

    def forward(self, g, n_features, e_features):
      positive = self.encoder(g, n_features, e_features, corrupt=False)
      negative = self.encoder(g, n_features, e_features, corrupt=True)

      positive = positive[1]
      negative = negative[1]

      summary = torch.sigmoid(positive.mean(dim=0))

      positive = self.discriminator(positive, summary)
      negative = self.discriminator(negative, summary)

      l1 = self.loss(positive, torch.ones_like(positive))
      l2 = self.loss(negative, torch.zeros_like(negative))

      return l1 + l2