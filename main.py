"""Main entry point to run the full Hybrid GCN + Ensemble pipeline."""

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
    print("\n" + "=" * 60)
    print("CONFIGURATION (GCN 12 layers – NO residual)")
    print("=" * 60)
    print(f"CSV Path         : {config.csv_path}")
    print(f"Fraction         : {config.fraction:.2f}")
    print(f"Train Ratio      : {config.train_ratio:.2f}")
    print(f"Oversample factor: {config.oversample_factor}")
    print(f"K neighbors      : {config.k_neighbors}")
    print("=" * 60 + "\n")

    # preprocessing
    x_clean, y_clean = preprocess_sgcc_data(config.csv_path)
    x_feat = extract_advanced_features(x_clean)
    x_combined_raw = np.hstack([x_clean, x_feat])

    # Spliting Data
    indices = np.arange(len(y_clean))
    train_val_idx, test_idx, y_train_val, y_test = train_test_split(
        indices, y_clean, test_size=1 - config.train_ratio,
        stratify=y_clean, random_state=42
    )
    x_train_val = x_combined_raw[train_val_idx]
    y_train_val = y_train_val
    x_test = x_combined_raw[test_idx]
    y_test = y_test

    train_idx, val_idx, y_train, y_val = train_test_split(
        np.arange(len(y_train_val)), y_train_val,
        test_size=0.2, stratify=y_train_val, random_state=42
    )
    x_train = x_train_val[train_idx]
    y_train = y_train
    x_val = x_train_val[val_idx]
    y_val = y_val

    print(f"Total samples    : {len(y_clean)}")
    print(f"Train+val        : {len(y_train_val)}, Test: {len(y_test)}")
    print(f"Train (pre-frac) : {len(y_train)}, Val: {len(y_val)}")

    # Fraction (if required)
    if config.fraction < 1.0:
        subset_size = int(config.fraction * len(x_train))
        sub_idx, _ = train_test_split(
            np.arange(len(x_train)), train_size=subset_size,
            stratify=y_train, random_state=42
        )
        train_sub_global_idx = train_val_idx[train_idx][sub_idx]
        x_train = x_train[sub_idx]
        y_train = y_train[sub_idx]
    else:
        train_sub_global_idx = train_val_idx[train_idx]

    print(f"Train (after frac): {len(y_train)}")

    # Building the User Graph
    scaler_std = StandardScaler()
    x_scaled_all = scaler_std.fit_transform(x_combined_raw)
    edge_index_user = build_user_graph(x_scaled_all, k=config.k_neighbors, metric='cosine')

    train_mask = torch.zeros(len(y_clean), dtype=torch.bool)
    val_mask = torch.zeros(len(y_clean), dtype=torch.bool)
    train_mask[train_val_idx[train_idx]] = True
    val_mask[train_val_idx[val_idx]] = True

    # Training the GCN
    print("\n--- Starting GCN Training (12 layers, no residual) ---")
    embeddings, loss_history = train_gcn(
        x_scaled_all, y_clean, train_mask, val_mask, edge_index_user,
        hidden=config.hidden_dim, embed_dim=config.embed_dim,
        lr=config.learning_rate,weight_decay = config.weight_decay,
        num_layers = config.GCN_layers,
        use_residual = config.use_residual,
        epochs=config.epochs, patience=config.patience,
        print_loss_every=config.print_loss_every,
        plot_loss=config.plot_loss,
        k_neighbors=config.k_neighbors
    )
    print(f"GCN embeddings shape: {embeddings.shape}")

    # Features
    x_full = np.hstack([x_clean, x_feat, embeddings])
    x_train_full = x_full[train_sub_global_idx]
    y_train_full = y_train
    x_val_full = x_full[train_val_idx[val_idx]]
    y_val_full = y_val
    x_test_full = x_full[test_idx]
    y_test_full = y_test

    # Training the Voting Ensemble
    print("\n--- Training Voting Ensemble ---")
    ensemble = train_voting_ensemble(
        x_train_full, y_train_full,
        x_val_full, y_val_full,
        factor=config.oversample_factor,
        map_k=config.MAP_K
    )

    # Final Evaluation
    probs = ensemble.predict_proba(x_test_full)[:, 1]
    auc = roc_auc_score(y_test_full, probs)
    map100 = map_at_k(y_test_full, probs, 100)
    map200 = map_at_k(y_test_full, probs, 200)

    print("\n" + "=" * 60)
    print(f" FINAL RESULTS (12‑Layer GCN (no residual) + Voting Ensemble, K={config.k_neighbors})")
    print("=" * 60)
    print(f"Train ratio  : {config.train_ratio:.2f}")
    print(f"Fraction     : {config.fraction:.2f}")
    print(f"Test size    : {len(y_test_full)}")
    print(f"AUC          : {auc:.4f}")
    print(f"MAP@100      : {map100:.4f}")
    print(f"MAP@200      : {map200:.4f}")
    print("=" * 60)

    # Saving the Results
    if config.save_results:
        os.makedirs("results", exist_ok=True)
        result_row = {
            'K': config.k_neighbors,
            'training_ratio': config.train_ratio,
            'fraction': config.fraction,
            'AUC_with_GCN': auc,
            'MAP100_with_GCN': map100,
            'MAP200_with_GCN': map200
        }
        df_result = pd.DataFrame([result_row])
        if os.path.exists(config.results_file):
            df_existing = pd.read_csv(config.results_file)
            df_existing = df_existing[~((df_existing['K'] == config.k_neighbors) &
                                        (df_existing['training_ratio'] == config.train_ratio) &
                                        (df_existing['fraction'] == config.fraction))]
            df_combined = pd.concat([df_existing, df_result], ignore_index=True)
        else:
            df_combined = df_result
        df_combined.to_csv(config.results_file, index=False)
        print(f"Results saved to {config.results_file}")
        


if __name__ == "__main__":
    main()
