# 12-layer Graph Convolutional Network (No Residual Connections)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class UserGCN(nn.Module):
  def __init__(self, in_channels, hidden=32, out_embed=32):
    super().__init__()

    self.conv1 = GCNConv(in_channels, hidden)
    self.bn1 = nn.BatchNorm1d(hidden)
    self.conv2 = GCNConv(hidden, hidden)
    self.bn2 = nn.BatchNorm1d(hidden)
    self.conv3 = GCNConv(hidden, hidden)
    self.bn3 = nn.BatchNorm1d(hidden)
    self.conv4 = GCNConv(hidden, hidden)
    self.bn4 = nn.BatchNorm1d(hidden)
    self.conv5 = GCNConv(hidden, hidden)
    self.bn5 = nn.BatchNorm1d(hidden)
    self.conv6 = GCNConv(hidden, hidden)
    self.bn6 = nn.BatchNorm1d(hidden)
    self.conv7 = GCNConv(hidden, hidden)
    self.bn7 = nn.BatchNorm1d(hidden)
    self.conv8 = GCNConv(hidden, hidden)
    self.bn8 = nn.BatchNorm1d(hidden)
    self.conv9 = GCNConv(hidden, hidden)
    self.bn9 = nn.BatchNorm1d(hidden)
    self.conv10 = GCNConv(hidden, hidden)
    self.bn10 = nn.BatchNorm1d(hidden)
    self.conv11 = GCNConv(hidden, hidden)
    self.bn11 = nn.BatchNorm1d(hidden)
    self.conv12 = GCNConv(hidden, hidden)
    self.bn12 = nn.BatchNorm1d(hidden)

    self.fc_embed = nn.Linear(hidden, out_embed)
    self.fc_out = nn.Linear(out_embed, 1)
    self.dropout = nn.Dropout(0.5)

  def forward(self, data, return_embeddings = False):
    x , edge_index = data.x , data.edge_index
    
    # layer 1
    x = self.conv1(x, edge_index)
    x = self.bn1(x)
    x = F.relu(x)
    x = self.dropout(x)

    # layer 2
    x = self.conv2(x, edge_index)
    x = self.bn2(x)
    x = F.relu(x)
    x = self.dropout(x)

    # layer 3
    x = self.conv3(x, edge_index)
    x = self.bn3(x)
    x = F.relu(x)
    x = self.dropout(x)

    # layer 4
    x = self.conv4(x, edge_index)
    x = self.bn4(x)
    x = F.relu(x)
    x = self.dropout(x)

    # layer 5
    x = self.conv5(x, edge_index)
    x = self.bn5(x)
    x = F.relu(x)
    x = self.dropout(x)

    # layer 6
    x = self.conv6(x, edge_index)
    x = self.bn6(x)
    x = F.relu(x)
    x = self.dropout(x)

    # layer 7
    x = self.conv7(x, edge_index)
    x = self.bn7(x)
    x = F.relu(x)
    x = self.dropout(x)

    # layer 8
    x = self.conv8(x, edge_index)
    x = self.bn8(x)
    x = F.relu(x)
    x = self.dropout(x)

    # layer 9
    x = self.conv9(x, edge_index)
    x = self.bn9(x)
    x = F.relu(x)
    x = self.dropout(x)

    # layer 10
    x = self.conv10(x, edge_index)
    x = self.bn10(x)
    x = F.relu(x)
    x = self.dropout(x)

    # layer 11
    x = self.conv11(x, edge_index)
    x = self.bn11(x)
    x = F.relu(x)
    x = self.dropout(x)

    # layer 12
    x = self.conv12(x, edge_index)
    x = self.bn12(x)
    x = F.relu(x)

    emb = self.fc_embed(x)
    if return_embeddings:
      return emb
    else:
      return self.fc_out(emb).squeeze()

    
    
