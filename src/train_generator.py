"""Step 2: data prep, training, and generation logic for the protein sequence generator (ProteinMiniGPT)."""
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from . import config
from .models import CHAR_TO_IDX, IDX_TO_CHAR, VOCAB_SIZE, ProteinSeqDataset


def prepare_sequences_from_csv(pos_csv, neg_csv):
    """Extract all unique real protein sequences from the two CSV files."""
    pos_df = pd.read_csv(pos_csv)
    neg_df = pd.read_csv(neg_csv)

    all_seqs = set(pos_df['protein_sequences_1']).union(set(pos_df['protein_sequences_2']))
    all_seqs = all_seqs.union(set(neg_df['protein_sequences_1'])).union(set(neg_df['protein_sequences_2']))
    return list(all_seqs)


def get_dataloader(sequences, batch_size=64, max_length=128, drop_last=True):
    dataset = ProteinSeqDataset(sequences, max_length=max_length)
    # shuffle=True ensures the sample order is randomized every epoch
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last)
    return dataloader


def train_generator(model, train_loader, val_loader, optimizer, criterion, epochs=20, device='cpu',
                     checkpoint_path=config.GENERATOR_MODEL_PATH):
    print(f"Starting Generator training (device: {device})...")

    history = {
        'train_loss': [],
        'val_loss': []
    }

    # Track the best validation loss so far (start at infinity)
    best_val_loss = float('inf')

    for epoch in range(1, epochs + 1):
        # ==========================================
        # Phase 1: Training
        # ==========================================
        model.train()
        train_epoch_loss = 0.0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{epochs:02d} [Train]")
        for batch_x, batch_y in train_bar:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out.view(-1, VOCAB_SIZE), batch_y.view(-1))
            loss.backward()

            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_epoch_loss += loss.item()
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = train_epoch_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        # ==========================================
        # Phase 2: Validation
        # ==========================================
        model.eval()
        val_epoch_loss = 0.0

        # Disable gradient tracking during validation to save memory and speed things up
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                out = model(batch_x)
                loss = criterion(out.view(-1, VOCAB_SIZE), batch_y.view(-1))
                val_epoch_loss += loss.item()

        avg_val_loss = val_epoch_loss / len(val_loader)
        history['val_loss'].append(avg_val_loss)

        # Print the summary for this epoch
        print(f"Epoch {epoch:02d} -> Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Check whether this is the best model so far (lower validation loss is better)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Found a better model! Saved to {checkpoint_path}")

        print("-" * 50)

    print("Generator training finished!")
    return model, history


@torch.no_grad()
def generate_new_protein(model, max_len=100, temperature=0.8, top_k=5):
    model.eval()
    device = next(model.parameters()).device

    # Special tokens that must never be generated: padding '_' and the start token '<bos>'
    # (these aren't amino acids; sampling them would corrupt the sequence string and feed ESM
    # an invalid input)
    forbidden_ids = [CHAR_TO_IDX['_'], CHAR_TO_IDX['<bos>']]

    # 1. Start with just the <bos> token
    current_seq = [CHAR_TO_IDX['<bos>']]
    input_tensor = torch.tensor([current_seq], dtype=torch.long, device=device)

    for _ in range(max_len):
        # 2. Have the model predict the logits for the next token
        logits = model(input_tensor)
        next_token_logits = logits[0, -1, :] / temperature  # apply temperature

        # 2.5 Block the non-amino-acid special tokens from being sampled
        for fid in forbidden_ids:
            next_token_logits[fid] = -float('Inf')

        # 3. Top-k sampling to avoid nonsense: keep only the top-k most likely tokens, set the rest to -inf
        v, _ = torch.topk(next_token_logits, top_k)
        next_token_logits[next_token_logits < v[-1]] = -float('Inf')

        # 4. Convert to a probability distribution and sample
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()

        # 5. Stop generating if the end token is sampled
        if next_token == CHAR_TO_IDX['<eos>']:
            break

        current_seq.append(next_token)
        input_tensor = torch.tensor([current_seq], dtype=torch.long, device=device)

    # 6. Convert the numeric ID sequence back into an amino-acid string
    generated_amino_acids = [IDX_TO_CHAR[idx] for idx in current_seq[1:]]  # skip <bos>
    return "".join(generated_amino_acids)
