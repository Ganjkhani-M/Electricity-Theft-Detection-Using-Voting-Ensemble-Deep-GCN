# Utility functions: metrics, oversampling , and splitting

import numpy as np

def oversample_features(x, y, factor):
  # Oversampling the positive class 
  x_norm = x[y==0]
  x_theft = x[y==1]
  y_norm = np.zeros(len(x_norm))
  y_theft = np.ones(len(x_theft))

  x_dup = np.repeat(x_theft , factor , axis = 0)
  y_dup = np.repeat(y_theft , factor , axis = 0)

  return np.vstack([x_norm, x_theft , x_dup]) , np.hstack([y_norm, y_theft, y_dup])


def map_at_k(labels, scores, k):
  sorted_idx = np.argsort(scores)[::-1]
  sorted_labels = labels[sorted_idx]
  pos_idx = np.where(sorted_labels == 1)[0]
  pos_in_top_k = pos_idx[pos_idx < k]

  if len(pos_in_top_k) == 0:
    return 0

  precisions = [(i+1) / (idx + 1) for i, idx in enumerate(pos_in_top_k)]
  return np.mean(precisions)
