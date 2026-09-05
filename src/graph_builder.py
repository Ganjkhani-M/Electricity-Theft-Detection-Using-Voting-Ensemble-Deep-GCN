# User-user graph construction using KNN

import torch
from sklearn.neighbors import NearestNeighbors

def build_user_graph(features, k=70, metric='cosine'):
  # building user-user unidirectional graph based on user features similarites 
  nbrs = NearestNeighbors(n_neighbors= k+1, metric = metric, n_jobs=-1)
  nbrs.fit(features)
  distance , indices = nbrs.kneighbors(features)

  row, col = [] , []
  
  for i in range(len(features)):
    for j in indices[i][1:]:
      if i < j:
        row.append(i)
        col.append(j)
    
  edge_index = torch.tensor([row+col, col+row], dtype=torch.long)
  return edge_index 

