"""Dataset download and PPI graph construction (corresponds to the data preprocessing part of Step 1 in the notebook)."""
import glob
import os

import kagglehub
import pandas as pd
import torch
from torch_geometric.data import Data
from tqdm import tqdm
from transformers import AutoTokenizer, EsmModel

from . import config


def download_dataset():
    """Download the dataset via kagglehub, return (positive CSV path, negative CSV path)."""
    dataset_dir = kagglehub.dataset_download(config.KAGGLE_DATASET)
    print("Dataset path:", dataset_dir)

    pos_path = glob.glob(os.path.join(dataset_dir, "**", "positive_protein_sequences.csv"), recursive=True)[0]
    neg_path = glob.glob(os.path.join(dataset_dir, "**", "negative_protein_sequences.csv"), recursive=True)[0]
    print("Positive-sample CSV:", pos_path)
    print("Negative-sample CSV:", neg_path)
    return pos_path, neg_path


def load_esm_model(model_name=config.ESM_MODEL_NAME, device=config.DEVICE):
    """Load the ESM-2 model and tokenizer, and set the model to eval mode."""
    print(f"Loading model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = EsmModel.from_pretrained(model_name).to(device)
    model.eval()
    return model, tokenizer


def generate_esm_embeddings(seq_to_id, model, tokenizer, device):
    """
    Input: seq_to_id (dict mapping sequence string -> node ID), model
    Output: x (Tensor of shape [num_nodes, feature_dim])
    """
    model.eval()

    num_nodes = len(seq_to_id)
    feature_dim = model.config.hidden_size  # automatically read the model's hidden dim (e.g. 480)

    # Prepare an all-zero matrix to hold the result
    x = torch.zeros((num_nodes, feature_dim))

    print(f"Extracting features for {num_nodes} protein sequences...")

    # Disable gradient tracking to save memory
    with torch.no_grad():
        for seq, idx in tqdm(seq_to_id.items()):
            # 2. Tokenization: convert the amino-acid string into model IDs
            # Truncate with max_length to avoid extremely long sequences blowing up memory
            inputs = tokenizer(seq, return_tensors="pt", truncation=True, max_length=1024)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # 3. Forward pass through the ESM model
            outputs = model(**inputs)

            # last_hidden_state has shape [1, sequence_length, feature_dim]
            last_hidden_state = outputs.last_hidden_state

            # 4. Pooling
            # Since each protein has a different length, we need to collapse the sequence_length dim
            # Standard approach: mean pooling. Remember to drop the special tokens ([CLS], [EOS]) at both ends
            seq_len = inputs['attention_mask'][0].sum().item()
            # The actual amino acids sit between index 1 and seq_len-2
            # Guard: if the sequence is empty (only [CLS]+[EOS], seq_len<=2), slicing between head/tail
            # gives an empty slice and mean() returns NaN. Fall back to averaging all valid tokens instead.
            if seq_len > 2:
                protein_vector = last_hidden_state[0, 1:seq_len - 1].mean(dim=0)
            else:
                protein_vector = last_hidden_state[0, :seq_len].mean(dim=0)

            # 5. Store into the corresponding node ID slot, and move back to CPU
            x[idx] = protein_vector.cpu()

    return x


def build_ppi_graph(pos_csv_path, neg_csv_path, model, tokenizer, device):
    print("1. Reading data...")
    pos_df = pd.read_csv(pos_csv_path)
    neg_df = pd.read_csv(neg_csv_path)

    print("2. Extracting all unique nodes (protein sequences)...")
    # Union all sequences from the positive and negative samples to find the unique set
    all_sequences = set(pos_df['protein_sequences_1']).union(set(pos_df['protein_sequences_2']))
    all_sequences = all_sequences.union(set(neg_df['protein_sequences_1'])).union(set(neg_df['protein_sequences_2']))

    # Build the sequence -> node ID (integer) mapping dict
    # Note: must sorted() before enumerate() — otherwise set iteration order is not stable
    # across processes (Python string hash randomization), which would make node IDs different
    # on every run and non-reproducible.
    all_sequences = sorted(all_sequences)
    seq_to_id = {seq: i for i, seq in enumerate(all_sequences)}
    num_nodes = len(all_sequences)
    print(f"-> Found {num_nodes} unique protein nodes.")

    print("3. Building the edge index...")
    # Positive samples (interaction exists, label = 1)
    pos_edge_1 = [seq_to_id[seq] for seq in pos_df['protein_sequences_1']]
    pos_edge_2 = [seq_to_id[seq] for seq in pos_df['protein_sequences_2']]
    pos_edges = torch.tensor([pos_edge_1, pos_edge_2], dtype=torch.long)

    # Negative samples (no interaction, label = 0)
    neg_edge_1 = [seq_to_id[seq] for seq in neg_df['protein_sequences_1']]
    neg_edge_2 = [seq_to_id[seq] for seq in neg_df['protein_sequences_2']]
    neg_edges = torch.tensor([neg_edge_1, neg_edge_2], dtype=torch.long)

    print("4. Building real node features with ESM-2...")
    x = generate_esm_embeddings(seq_to_id, model, tokenizer, device)

    return x, pos_edges, neg_edges, seq_to_id


def custom_link_split(x, pos_edges, neg_edges, val_ratio=0.1, test_ratio=0.2,
                       disjoint_train_ratio=0.3):
    """
    Custom link-prediction data split that fully preserves the user-defined positive/negative
    samples while preventing information leakage.

    Args:
      x (Tensor): node features (produced by ESM)
      pos_edges (Tensor): positive-sample edge_index, shape [2, num_pos]
      neg_edges (Tensor): negative-sample edge_index, shape [2, num_neg]
      disjoint_train_ratio (float): fraction of the training positive edges used as the
          "supervision target"; the rest are used for message passing. The two subsets are
          mutually exclusive, so the GNN never sees the very edge it's being asked to predict
          during message passing (training link leakage).
    """
    # 1. Shuffle and split the positive samples
    num_pos = pos_edges.size(1)
    pos_idx = torch.randperm(num_pos)
    pos_val_end = int(val_ratio * num_pos)
    pos_test_end = pos_val_end + int(test_ratio * num_pos)

    val_pos_edges = pos_edges[:, pos_idx[:pos_val_end]]
    test_pos_edges = pos_edges[:, pos_idx[pos_val_end:pos_test_end]]
    train_pos_edges = pos_edges[:, pos_idx[pos_test_end:]]

    # 1b. Split the training positive edges further into two disjoint subsets:
    # "message passing" and "supervision"
    num_train_pos = train_pos_edges.size(1)
    tp_idx = torch.randperm(num_train_pos)
    num_sup = int(disjoint_train_ratio * num_train_pos)
    sup_pos_edges = train_pos_edges[:, tp_idx[:num_sup]]  # used as the prediction target
    mp_pos_edges = train_pos_edges[:, tp_idx[num_sup:]]  # used for message passing

    # 2. Shuffle and split the negative samples
    num_neg = neg_edges.size(1)
    neg_idx = torch.randperm(num_neg)
    neg_val_end = int(val_ratio * num_neg)
    neg_test_end = neg_val_end + int(test_ratio * num_neg)

    val_neg_edges = neg_edges[:, neg_idx[:neg_val_end]]
    test_neg_edges = neg_edges[:, neg_idx[neg_val_end:neg_test_end]]
    train_neg_edges = neg_edges[:, neg_idx[neg_test_end:]]

    # --- 3. Build the PyG Data objects ---

    # [Train Data]
    # message passing: only mp_pos_edges (the model learns the graph structure from these)
    train_edge_index = mp_pos_edges
    # supervision: sup_pos_edges (disjoint from message passing) + the train negative samples
    train_label_index = torch.cat([sup_pos_edges, train_neg_edges], dim=1)
    train_label = torch.cat([torch.ones(sup_pos_edges.size(1)), torch.zeros(train_neg_edges.size(1))], dim=0)

    train_data = Data(x=x, edge_index=train_edge_index, edge_label_index=train_label_index, edge_label=train_label)

    # [Validation Data]
    # message passing: to prevent leakage, only the full set of train positive edges can be used
    # (the val edges have not been observed yet at this point)
    val_edge_index = train_pos_edges
    # supervision: val positive samples + val negative samples
    val_label_index = torch.cat([val_pos_edges, val_neg_edges], dim=1)
    val_label = torch.cat([torch.ones(val_pos_edges.size(1)), torch.zeros(val_neg_edges.size(1))], dim=0)

    val_data = Data(x=x, edge_index=val_edge_index, edge_label_index=val_label_index, edge_label=val_label)

    # [Test Data]
    # message passing: can include both train + val positive edges (gives the test-time graph
    # a more complete structure)
    test_edge_index = torch.cat([train_pos_edges, val_pos_edges], dim=1)
    # supervision: test positive samples + test negative samples
    test_label_index = torch.cat([test_pos_edges, test_neg_edges], dim=1)
    test_label = torch.cat([torch.ones(test_pos_edges.size(1)), torch.zeros(test_neg_edges.size(1))], dim=0)

    test_data = Data(x=x, edge_index=test_edge_index, edge_label_index=test_label_index, edge_label=test_label)

    return train_data, val_data, test_data
