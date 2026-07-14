"""Step 2 main script: train the autoregressive protein sequence generator (ProteinMiniGPT).

Usage:
    python scripts/run_step2_train_generator.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

from src import config
from src.data import download_dataset
from src.models import CHAR_TO_IDX, ProteinMiniGPT, VOCAB_SIZE
from src.train_generator import get_dataloader, prepare_sequences_from_csv, train_generator
from src.visualize import plot_generator_history


def main():
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    device = config.DEVICE

    # 1. Get the data (same kagglehub dataset used in Step 1)
    pos_path, neg_path = download_dataset()
    sequences = prepare_sequences_from_csv(pos_path, neg_path)

    # 2. Split the dataset (90% train, 10% validation)
    train_seqs, val_seqs = train_test_split(sequences, test_size=0.1, random_state=42)
    print(f"Total sequences: {len(sequences)} | Train: {len(train_seqs)} | Val: {len(val_seqs)}")

    train_loader = get_dataloader(train_seqs, batch_size=config.GEN_BATCH_SIZE,
                                   max_length=config.GEN_MAX_SEQ_LEN, drop_last=True)
    val_loader = get_dataloader(val_seqs, batch_size=config.GEN_BATCH_SIZE,
                                 max_length=config.GEN_MAX_SEQ_LEN, drop_last=False)

    # 3. Initialize the model, optimizer, and loss function
    generator = ProteinMiniGPT(vocab_size=VOCAB_SIZE, d_model=config.GEN_D_MODEL,
                                nhead=config.GEN_NHEAD, num_layers=config.GEN_NUM_LAYERS).to(device)

    # Key detail: set ignore_index so the model automatically skips the loss for the padding
    # character '_'
    criterion = nn.CrossEntropyLoss(ignore_index=CHAR_TO_IDX['_'])
    # AdamW is generally recommended for training language models
    optimizer = torch.optim.AdamW(generator.parameters(), lr=config.GEN_LR, weight_decay=0.01)

    # 4. Train
    trained_generator, gen_history = train_generator(
        model=generator,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        epochs=config.GEN_EPOCHS,
        device=device,
        checkpoint_path=config.GENERATOR_MODEL_PATH,
    )

    plot_generator_history(gen_history)


if __name__ == "__main__":
    main()
