"""Step 3: De novo interactor design (generate -> screen -> rank).

Reuses the Step 1 GNN (judge) and the Step 2 Generator to form a "design-and-screen"
generative pipeline:

1. Pick a high-degree protein from the dataset as the target (the GNN's embedding for it is
   the best-learned, so scores against it are the most trustworthy).
2. The Generator produces a large batch of brand-new sequences as candidate binders.
3. Filter out ones that are too short / duplicates, and flag which are novel (not in the
   training set).
4. Turn the candidates into features with ESM-2, insert them as isolated new nodes into the
   graph, and score them all with the GNN in one pass.
5. Rank by predicted interaction probability and export the top candidates to FASTA.
"""
import torch
from tqdm import tqdm

from . import config
from .models import LinkPredictor, ProteinMiniGPT, VOCAB_SIZE
from .train_generator import generate_new_protein


def select_target_by_degree(pos_edges, sequence_mapping, top_n=10):
    """Pick the target by ranking nodes on degree.

    The higher a node's degree, the more reliable the embedding the GNN learned for it via
    message passing — so it makes the most trustworthy target for scoring against.
    """
    num_nodes = len(sequence_mapping)
    # Count both rows of pos_edges (undirected), i.e. how many times each node appears = degree
    deg = torch.bincount(pos_edges.reshape(-1), minlength=num_nodes)

    topk = torch.topk(deg, top_n)
    print(f"Top {top_n} nodes by degree (good target candidates):")
    for rank, (nid, d) in enumerate(zip(topk.indices.tolist(), topk.values.tolist()), 1):
        print(f"{rank:2d}. node_id = {nid:5d} | degree = {d}")

    # node_id -> sequence lookup table (useful later for BLAST-identifying the target protein)
    id_to_seq = {idx: seq for seq, idx in sequence_mapping.items()}

    # By default, pick the highest-degree node as the target; you can also manually pick another
    # node_id from the printed list
    target_node_id = int(topk.indices[0])
    print(f"\n>>> Selected target node_id = {target_node_id} (degree = {int(deg[target_node_id])})")
    print("Target sequence (paste into NCBI BLAST to identify the protein and its known interactors):")
    print(id_to_seq[target_node_id])

    return target_node_id, id_to_seq


@torch.no_grad()
def embed_sequences_with_esm(seqs, esm_model, tokenizer, device):
    """Turn a batch of sequences into [N, 480] mean-pooled ESM-2 embeddings."""
    esm_model.eval()
    feats = []
    for s in seqs:
        inputs = tokenizer(s, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        out = esm_model(**inputs)
        seq_len = inputs['attention_mask'][0].sum().item()
        if seq_len > 2:
            v = out.last_hidden_state[0, 1:seq_len - 1].mean(dim=0)
        else:
            v = out.last_hidden_state[0, :seq_len].mean(dim=0)
        feats.append(v)
    return torch.stack(feats, dim=0)  # [N, 480]


def design_binders(target_node_id, x, pos_edges, sequence_mapping, esm_model, esm_tokenizer, device,
                    n_candidates=200,
                    temperatures=(0.8, 1.0, 1.2),
                    top_k=15,          # raised from 5 to 15: widen the sampling pool for more diversity, curb mode collapse
                    gen_max_len=160,   # raised from 100 to 160: past the ~127 position where <eos> tends to appear during training, so sequences can end naturally
                    min_len=20,
                    keep_top=10,
                    fasta_path=config.DESIGNED_BINDERS_FASTA,
                    generator_weight=config.GENERATOR_MODEL_PATH,
                    gnn_weight=config.GNN_MODEL_PATH,
                    seed=42):
    print(f"=== Step 3: designing a de novo interactor for target node {target_node_id} ===")
    torch.manual_seed(seed)

    # 1) Load the Step 1 GNN and Step 2 Generator (their respective best checkpoints)
    gen = ProteinMiniGPT(vocab_size=VOCAB_SIZE, d_model=config.GEN_D_MODEL,
                          nhead=config.GEN_NHEAD, num_layers=config.GEN_NUM_LAYERS).to(device)
    gen.load_state_dict(torch.load(generator_weight, map_location=device))
    gen.eval()

    gnn = LinkPredictor(in_channels=config.ESM_FEATURE_DIM, hidden_channels=config.GNN_HIDDEN_CHANNELS,
                         out_channels=config.GNN_OUT_CHANNELS).to(device)
    gnn.load_state_dict(torch.load(gnn_weight, map_location=device))
    gnn.eval()

    # 2) Generate N new sequences (cycling through temperatures for diversity)
    print(f"\n[1/4] Generating {n_candidates} candidate sequences (top_k={top_k}, max_len={gen_max_len})...")
    cand_seqs = []
    for i in tqdm(range(n_candidates)):
        T = temperatures[i % len(temperatures)]
        s = generate_new_protein(gen, max_len=gen_max_len, temperature=T, top_k=top_k)
        cand_seqs.append(s)

    # 3) Basic filtering: drop ones that are too short, dedupe candidates, flag novelty
    print("[2/4] Filtering (length + dedupe + novelty flag)...")
    train_seq_set = set(sequence_mapping.keys())
    seen, filtered = set(), []
    for s in cand_seqs:
        if len(s) < min_len:  # too short
            continue
        if s in seen:  # duplicate among candidates
            continue
        seen.add(s)
        is_novel = s not in train_seq_set  # True = brand-new (not present in the training set)
        filtered.append((s, is_novel))
    n_novel = sum(1 for _, nov in filtered if nov)
    print(f"   -> Kept {len(filtered)} candidates ({n_novel} of which are novel)")
    if len(filtered) == 0:
        print("No candidates survived filtering — try raising n_candidates or adjusting temperature.")
        return None

    # 4) Convert to ESM-2 features, insert all candidates as isolated new nodes at once, score with the GNN
    print("[3/4] Converting to ESM-2 features + scoring with the GNN...")
    cand_only_seqs = [s for s, _ in filtered]
    cand_feats = embed_sequences_with_esm(cand_only_seqs, esm_model, esm_tokenizer, device)  # [M,480]

    num_nodes = x.size(0)
    x_ext = torch.cat([x.to(device), cand_feats], dim=0)  # [num_nodes+M, 480]
    edge_index = pos_edges.to(device)  # candidates are all isolated nodes (no edges)
    z_all = gnn.encoder(x_ext, edge_index)  # run the encoder just once

    M = cand_feats.size(0)
    cand_ids = torch.arange(num_nodes, num_nodes + M, device=device)
    tgt_ids = torch.full((M,), target_node_id, dtype=torch.long, device=device)
    edge_lab = torch.stack([cand_ids, tgt_ids], dim=0)
    probs = torch.sigmoid(gnn.decoder(z_all, edge_lab)).cpu()

    # 5) Rank by score and export the top keep_top to FASTA
    print("[4/4] Ranking and writing the FASTA output...")
    order = torch.argsort(probs, descending=True).tolist()
    print(f"\n--- Top {min(keep_top, len(order))} design results (target node {target_node_id}) ---")
    with open(fasta_path, 'w') as f:
        for rank, j in enumerate(order[:keep_top], 1):
            s, is_novel = filtered[j]
            p = probs[j].item()
            tag = "novel" if is_novel else "in_train"
            f.write(f">cand{rank}_target{target_node_id}_score{p:.3f}_{tag}_len{len(s)}\n{s}\n")
            print(f"#{rank:2d} | score={p:.3f} | {tag:8s} | len={len(s)}")
    print(f"\nSaved the top {keep_top} candidates to FASTA: {fasta_path}")
    return [(r + 1, probs[j].item(), filtered[j][1], filtered[j][0]) for r, j in enumerate(order)]
