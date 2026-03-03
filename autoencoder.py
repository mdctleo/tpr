import torch
import torch.nn as nn
from torch_geometric.nn import NNConv


class EdgeAutoencoder(nn.Module):
    def __init__(self, edge_dim, hidden_dim):
        super(EdgeAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, edge_dim),
            # nn.Sigmoid()
        )

    def forward(self, edge_attr):
        encoded = self.encoder(edge_attr)
        decoded = self.decoder(encoded)
        return decoded


class GraphAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, edge_dim, output_dim):
        super(GraphAutoencoder, self).__init__()
        self.encoder = NNConv(input_dim, hidden_dim, nn=nn.Sequential(
            nn.Linear(edge_dim, hidden_dim * input_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim * input_dim, hidden_dim * input_dim)
        ))
        self.decoder = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.encoder(x, edge_index, edge_attr)
        x = torch.relu(x)
        x = self.decoder(x)
        return x