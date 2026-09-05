# Main entry point to run the full Hybrid GCN + Voting Ensemble 

import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
import warnings 
warnings.filterwarnings('ignore')

import config 
from src.preprocess import preprocess_sgcc_data, extract_advanced_features
from src.graph_builder import build_user_graph
from src.trainer import train_gcn
from src.ensemble import train_voting_ensemble
from src.utils import map_at_k
from sklearn.metrics import roc_auc_score


def main():
  print("\n" + "=" * 50)
  print("Configuration (GCN with 12 layers - No residual connections)")
  print("=" * 50)
  print(f"CSV path           : {config.csv_path}")
  print(f"Fraction           : {config.fraction: .2f}")
  print(f"Train Ratio        : {config.train_ratio: .2f}")
  print(f"Oversample Factor  : {config.oversample_factor}")
  print(f"K neighbors        : {config.k_neighbors}")

  
