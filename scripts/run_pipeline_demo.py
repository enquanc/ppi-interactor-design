"""End-to-end demo that closes out Step 2: load the graph data saved by Step 1 -> Generator
creates one new sequence -> GNN scores it.

Prerequisites: run run_step1_train_gnn.py first (needs checkpoints/graph_data.pt and
               best_model.pth), and run_step2_train_generator.py (needs
               checkpoints/best_generator.pth)

Usage:
    python scripts/run_pipeline_demo.py --target-node-id 809
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from torch_geometric.data import Data

from src import config
from src.data import load_esm_model
from src.pipeline import run_end_to_end_pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-node-id", type=int, default=809,
                         help="Which existing node (node_id) to test the new sequence against for interaction")
    args = parser.parse_args()

    device = config.DEVICE
    graph_data = torch.load(config.GRAPH_DATA_PATH, map_location=device)
    x, pos_edges = graph_data["x"], graph_data["pos_edges"]

    # run_end_to_end_pipeline needs a graph object carrying x / edge_index
    inference_graph = Data(x=x, edge_index=pos_edges)

    esm_model, esm_tokenizer = load_esm_model(config.ESM_MODEL_NAME, device)

    run_end_to_end_pipeline(
        inference_graph, target_node_id=args.target_node_id,
        esm_model=esm_model, tokenizer=esm_tokenizer, device=device,
        generator_weight=config.GENERATOR_MODEL_PATH, gnn_weight=config.GNN_MODEL_PATH,
    )


if __name__ == "__main__":
    main()
