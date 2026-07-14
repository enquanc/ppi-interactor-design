"""Model definitions: the Step 1 GNN (Encoder/Decoder) and the Step 2 autoregressive sequence generator."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.nn import SAGEConv

# ============================================================
# Step 1: Link-prediction GNN
# ============================================================


class GNNEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        # Use GraphSAGE — it performs well on large graphs and with unseen nodes
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        # Layer 1: aggregate features, then ReLU + Dropout against overfitting
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        # Layer 2: output the final node embedding (reduced dimension)
        x = self.conv2(x, edge_index)
        return x


class EdgeDecoder(torch.nn.Module):
    """
    A small MLP replacing the original "raw dot product" decoder.

    The original (src * dst).sum() dot product, on unnormalized 128-dim vectors, easily
    produces very large logits that saturate the sigmoid near 1 — so almost every pair gets
    predicted at ~100% (the probability becomes meaningless). Here we instead feed a symmetric
    feature [src*dst, |src-dst|] into an MLP:
      - symmetric (swapping src/dst gives the same result), matching the undirected nature of
        protein-protein interactions
      - the MLP learns a better-calibrated score
    """

    def __init__(self, in_channels, hidden_channels=128):
        super().__init__()
        self.lin1 = torch.nn.Linear(in_channels * 2, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, 1)

    def forward(self, z, edge_label_index):
        # z holds all node embeddings produced by the Encoder
        # Gather the src/dst node features according to edge_label_index
        src = z[edge_label_index[0]]
        dst = z[edge_label_index[1]]

        # Symmetric edge feature: element-wise product + absolute difference
        h = torch.cat([src * dst, (src - dst).abs()], dim=-1)
        h = self.lin1(h).relu()
        # Return the raw logit (no sigmoid applied), paired with BCEWithLogitsLoss
        return self.lin2(h).squeeze(-1)


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.encoder = GNNEncoder(in_channels, hidden_channels, out_channels)
        self.decoder = EdgeDecoder(out_channels)

    def forward(self, x, edge_index, edge_label_index):
        # 1. Get the updated node features z from the Encoder
        z = self.encoder(x, edge_index)
        # 2. Have the Decoder predict the connection probability score for the labeled edges
        return self.decoder(z, edge_label_index)


# ============================================================
# Step 2: Autoregressive protein sequence generator
# ============================================================

# 20 standard amino acids plus special tokens
AMINO_ACIDS = ['_', '<bos>', '<eos>', 'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G',
               'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
CHAR_TO_IDX = {char: idx for idx, char in enumerate(AMINO_ACIDS)}
IDX_TO_CHAR = {idx: char for idx, char in enumerate(AMINO_ACIDS)}
VOCAB_SIZE = len(AMINO_ACIDS)


class ProteinSeqDataset(Dataset):
    def __init__(self, sequences, max_length=256):
        self.sequences = sequences
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        # Truncate overly long sequences, leaving room for <bos> and <eos>
        seq = seq[:self.max_length - 2]

        # Assemble the sequence: <bos> + amino acids + <eos>
        full_seq = ['<bos>'] + list(seq) + ['<eos>']
        numerical_seq = [CHAR_TO_IDX.get(c, 0) for c in full_seq]  # unknown letters map to '_'

        # Pad to a uniform length
        pad_len = self.max_length - len(numerical_seq)
        numerical_seq += [CHAR_TO_IDX['_']] * pad_len

        # Autoregressive input/target alignment: input is 0..N-1, target is 1..N
        token_tensor = torch.tensor(numerical_seq, dtype=torch.long)
        x = token_tensor[:-1]
        y = token_tensor[1:]
        return x, y


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x):
        positions = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)
        return x + self.pe(positions)


class ProteinMiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=3, dim_feedforward=256, dropout=0.3):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)

        # Build the Transformer layer (dropout is passed explicitly)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            activation='gelu',
            dropout=dropout  # <--- raise this to fight overfitting
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Final language-modeling head
        self.ln = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        sz = x.size(1)
        mask = nn.Transformer.generate_square_subsequent_mask(sz, device=x.device)

        out = self.token_emb(x)
        out = self.pos_emb(out)

        out = self.transformer(out, mask=mask, is_causal=True)
        out = self.ln(out)
        logits = self.lm_head(out)
        return logits
