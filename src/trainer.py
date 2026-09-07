# Training loop for the GCN model

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score
from torch_geometric.data import Data as PyGData
import matplotlib.pyplot as plt

from .models import UserGCN

def train_gcn(x, y, train_mask, val_mask, edge_index, hidden=32, embed_dim=32, lr=1e-4,weight_decay=1e-4,
              num_layers = 12, use_residual=False, epochs=150, patience = 30, print_loss_every=5, plot_loss=False, k_neighbors=70):

                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

                data = PyGData(x = torch.tensor(x, dtype=torch.float32),
                                edge_index = edge_index, 
                                y = torch.tensor(y, dtype = torch.float32))

                data.train_mask = train_mask
                data.val_mask = val_mask
                data = data.to(device)

                model = UserGCN(in_channels = x.shape[1], hidden = hidden, out_embed = embed_dim, num_layers = num_layers, use_residual = use_residual).to(device)

                # Positive class weighting to deal with data imbalance
                pos_weight = torch.tensor([
                                          (y[train_mask.cpu()] == 0).sum() / max((y[train_mask.cpu()] == 1).sum(), 1)]).to(device)
                criterion = nn.BCEWithLogitsLoss(pos_weight = pos_weight)
                optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience = 10, factor=0.5)
                best_val_auc = 0
                best_state = None
                trigger = 0
                loss_history = [] 

                print(f'Training GCN on {device}.')
                print(f'K = {k_neighbors}')
                print(f"{'Epoch' : >6} | {'Train Loss' : >12} | {'Val AUC' : >10}")
                print("-" * 40)


                for epoch in range(epochs):
                  model.train()
                  optimizer.zero_grad()
                  out = model(data, return_embeddings = False)
                  loss = criterion(out[data.train_mask], data.y[data.train_mask])

                  if torch.isnan(loss):
                    break

                  loss.backward()
                  torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                  optimizer.step()
                  loss_history.append(loss.item())

                  # Evaluation of the model
                  model.eval()
                  with torch.no_grad():
                    val_scores = torch.sigmoid(model(data, return_embeddings = False))[data.val_mask].cpu().numpy()
                    val_true = data.y[data.val_mask].cpu().numpy()
                    val_auc = roc_auc_score(val_true, val_scores) if len(np.unique(val_scores)) > 1 else 0.5

                  scheduler.step(val_auc)

                  if (epoch + 1) % print_loss_every == 0 or epoch == 0 :
                    print(f"{epoch + 1: 6d} | {loss.item(): 12.6f} | {val_auc:10.4f}")

                  if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    trigger = 0
                    best_state = {k : v.cpu().clone() for k, v in model.state_dict().items()}
                  else:
                    trigger += 1
                    if trigger >= patience:
                      print(f"Early stopping at epoch {epoch + 1}")
                      break

                # choosing the best weights
                model.load_state_dict(best_state)
                model.eval()
                with torch.no_grad():
                  embeddings = model(data, return_embeddings = True).cpu().numpy()

                # Plotting the loss
                if plot_loss:
                  plt.figure(figsize=(8,5))
                  plt.plot(range(1, len(loss_history) + 1) , loss_history , color= 'blue', linewidth = 1.5)
                  plt.xlabel('Epoch')
                  plt.ylabel('Training Loss')
                  plt.title(f"GCN with K={k_neighbors} / Training Loss")
                  plt.grid(alpha = 0.4)
                  plt.tight_layout()
                  plt.savefig(f'results/gcn_training_loss_12layers_K{k_neighbors}.png', dpi=200)
                  plt.show()
                  print(f"Training loss plot saved as 'results/gcn_training_loss_12layers_K{k_neighbors}.png'")
                return embeddings , loss_history


                  

                
