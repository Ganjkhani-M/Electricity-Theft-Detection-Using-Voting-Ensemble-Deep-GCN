# 12-layer Graph Convolutional Network (No Residual Connections)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class UserGCN(nn.Module):
  def __init__(self, in_channels, hidden=32, out_embed=32, num_layers = 12):
    super().__init__()

    self.convs = nn.ModuleList()
    self.bns = nn.ModuleList()
    self.convs.append(GCNConv(in_channels, hidden))
    self.bns.append(nn.BatchNorm1d(hidden))
    for _ in range(num_layers -1):
      self.convs.append(GCNConv(hidden, hidden))
      self.bns.append(nn.BatchNorm1d(hidden))

    self.fc_embed = nn.Linear(hidden, out_embed)
    self.fc_out = nn.Linear(out_embed, 1)
    self.dropout = nn.Dropout(0.5)
    
  def forward(self, data, return_embeddings = False):
    x , edge_index = data.x , data.edge_index

    for i , (conv , bn) in enumerate(zip(self.convs, self.bns)):
      x = conv(x, edge_index)
      x = bn(x)
      x = F.relu(x)
      if i < len(self.convs) - 1: # Using the dropout for all layers except the last one
        x = self.dropout(x)

    emb = self.fc_embed(x)
    if return_embeddings:
      return emb
    else:
      return self.fc_out(emb).squeeze()

    
    
