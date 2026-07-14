"""Step 1 main script: download data -> build PPI graph -> train GNN -> evaluate -> plot.

Usage:
    python scripts/run_step1_train_gnn.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import torch

from src import config
from src.data import build_ppi_graph, custom_link_split, download_dataset, load_esm_model
from src.models import LinkPredictor
from src.train_gnn import evaluate_model, train_and_validate
from src.visualize import plot_top_predictions, plot_training_history, plot_tsne_edge_embeddings


def main():
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    device = config.DEVICE

    # 1. Download the data + load ESM-2
    pos_path, neg_path = download_dataset()
    esm_model, esm_tokenizer = load_esm_model(config.ESM_MODEL_NAME, device)

    # 2. Build the PPI graph (ESM embeddings as node features)
    x, pos_edges, neg_edges, sequence_mapping = build_ppi_graph(
        pos_path, neg_path, esm_model, esm_tokenizer, device
    )

    pos_df = pd.read_csv(pos_path)
    neg_df = pd.read_csv(neg_path)
    print('Edges counts: ', pos_df.shape[0] + neg_df.shape[0])

    # 3. Split into train/val/test while preventing message-passing/supervision leakage
    train_data, val_data, test_data = custom_link_split(
        x=x, pos_edges=pos_edges, neg_edges=neg_edges,
        val_ratio=config.GNN_VAL_RATIO, test_ratio=config.GNN_TEST_RATIO,
        disjoint_train_ratio=config.GNN_DISJOINT_TRAIN_RATIO,
    )
    train_data, val_data, test_data = train_data.to(device), val_data.to(device), test_data.to(device)
    print("Train positive samples:", (train_data.edge_label == 1).sum().item())
    print("Train negative samples:", (train_data.edge_label == 0).sum().item())

    # 4. Train the GNN (LinkPredictor)
    gnn_model = LinkPredictor(in_channels=config.ESM_FEATURE_DIM, hidden_channels=config.GNN_HIDDEN_CHANNELS,
                               out_channels=config.GNN_OUT_CHANNELS).to(device)
    optimizer = torch.optim.Adam(gnn_model.parameters(), lr=config.GNN_LR)
    criterion = torch.nn.BCEWithLogitsLoss()

    trained_model, training_history = train_and_validate(
        gnn_model, optimizer, criterion, train_data, val_data,
        epochs=config.GNN_EPOCHS, checkpoint_path=config.GNN_MODEL_PATH,
    )

    # 5. Do the final blind test using the checkpoint with the highest validation AUC
    #    (rather than the weights from the last epoch)
    trained_model.load_state_dict(torch.load(config.GNN_MODEL_PATH, map_location=device))
    final_test_auc = evaluate_model(trained_model, test_data)
    print('\n================================')
    print(f'Final blind-test result! Final Test AUC: {final_test_auc:.4f}')
    print('================================')

    plot_training_history(training_history)
    plot_top_predictions(trained_model, test_data, top_k=50)
    plot_tsne_edge_embeddings(trained_model, test_data, num_samples=2000, method="t-SNE")
    plot_tsne_edge_embeddings(trained_model, test_data, num_samples=2000, method="UMAP")

    # 6. Save the graph data that Step 3 needs to reuse (so we don't have to rerun ESM embedding every time)
    torch.save({
        "x": x,
        "pos_edges": pos_edges,
        "neg_edges": neg_edges,
        "sequence_mapping": sequence_mapping,
        "pos_path": pos_path,
        "neg_path": neg_path,
    }, config.GRAPH_DATA_PATH)
    print(f"\nGraph data saved to {config.GRAPH_DATA_PATH} — the Step 2 / Step 3 scripts can load it directly.")


if __name__ == "__main__":
    main()
