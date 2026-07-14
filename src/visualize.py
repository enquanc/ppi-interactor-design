"""All plotting and statistical-check functions (training curves, t-SNE/UMAP, prediction network graph, candidate sequence sanity checks)."""
from collections import Counter

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import umap
from sklearn.manifold import TSNE

# 20 standard amino acids, used for sequence composition sanity checks
STANDARD_AMINO_ACIDS = list("ARNDCQEGHILKMFPSTWYV")


# ============================================================
# Step 1: GNN training / prediction visualization
# ============================================================

def plot_training_history(history):
    # Figure size (width 12, height 5)
    plt.figure(figsize=(12, 5))

    # --- Plot the training loss curve (left panel) ---
    plt.subplot(1, 2, 1)
    epochs_range = range(1, len(history['train_loss']) + 1)
    plt.plot(epochs_range, history['train_loss'], label='Train Loss', color='blue')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (BCE)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    # --- Plot the validation AUC curve (right panel) ---
    plt.subplot(1, 2, 2)
    plt.plot(history['val_epochs'], history['val_auc'], label='Validation AUC', color='orange', marker='o')
    plt.title('Validation AUC Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('AUC Score')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    plt.tight_layout()
    plt.show()


@torch.no_grad()
def plot_top_predictions(model, data, top_k=50):
    model.eval()

    # 1. Get the predicted probabilities on the test set
    out = model(data.x, data.edge_index, data.edge_label_index)
    probs = torch.sigmoid(out).cpu().numpy()

    # 2. Get the ground-truth labels and the corresponding node IDs
    labels = data.edge_label.cpu().numpy()
    edges = data.edge_label_index.cpu().numpy()

    # 3. Find the indices of the top_k edges with the highest predicted probability
    top_indices = np.argsort(probs)[-top_k:][::-1]

    # 4. Build the NetworkX graph
    G = nx.Graph()
    for idx in top_indices:
        src = edges[0, idx]
        dst = edges[1, idx]
        prob = probs[idx]
        is_true = labels[idx] == 1.0  # check whether this prediction was actually correct
        edge_color = 'green' if is_true else 'red'
        G.add_edge(src, dst, weight=prob, color=edge_color)

    # 5. Draw the plot
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, k=0.8, iterations=100, seed=42)
    edge_colors = [G[u][v]['color'] for u, v in G.edges()]
    edge_widths = [G[u][v]['weight'] * 3 for u, v in G.edges()]

    nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=100, edgecolors='black')
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths, alpha=0.8)
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')

    plt.title(f"Top {top_k} Predicted Protein Interactions", fontsize=16)

    import matplotlib.lines as mlines
    green_line = mlines.Line2D([], [], color='green', linewidth=2, label='True Positive (Correct)')
    red_line = mlines.Line2D([], [], color='red', linewidth=2, label='False Positive (Wrong)')
    plt.legend(handles=[green_line, red_line], loc='best')

    plt.axis('off')
    plt.tight_layout()
    plt.show()


@torch.no_grad()
def plot_tsne_edge_embeddings(model, data, num_samples=2000, method="UMAP"):
    """Reduce the test-set edge features to 2D with t-SNE/UMAP and visualize them."""
    model.eval()

    # 1. Get the latent features of all nodes via the Encoder
    z = model.encoder(data.x, data.edge_index)

    # 2. Get the target edges and ground-truth labels
    edges = data.edge_label_index
    labels = data.edge_label.cpu().numpy()

    # 3. Build the edge features (element-wise product of src and dst vectors)
    src = z[edges[0]]
    dst = z[edges[1]]
    edge_features = (src * dst).cpu().numpy()

    # 4. Random subsampling (running dimensionality reduction on everything would be too slow and too crowded)
    np.random.seed(42)
    indices = np.random.choice(len(labels), size=min(num_samples, len(labels)), replace=False)
    sampled_features = edge_features[indices]
    sampled_labels = labels[indices]

    # 5. Run dimensionality reduction (from 128 dims down to 2)
    if method == "UMAP":
        print("Computing UMAP...")
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        reduced_features = reducer.fit_transform(sampled_features)
    elif method == "t-SNE":
        print("Computing t-SNE, this may take a few tens of seconds...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        reduced_features = tsne.fit_transform(sampled_features)
    else:
        print("Error!!!")
        return

    # 6. Draw the scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(reduced_features[sampled_labels == 0, 0],
                reduced_features[sampled_labels == 0, 1],
                c='tomato', label='No Interaction', alpha=0.6, s=15, edgecolors='none')
    plt.scatter(reduced_features[sampled_labels == 1, 0],
                reduced_features[sampled_labels == 1, 1],
                c='mediumseagreen', label='Interaction', alpha=0.6, s=15, edgecolors='none')

    plt.title(f'{method} Visualization of Protein Pair Embeddings', fontsize=16)
    plt.legend(markerscale=3)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


# ============================================================
# Step 2: Generator training visualization
# ============================================================

def plot_generator_history(history):
    plt.figure(figsize=(8, 5))
    epochs_range = range(1, len(history['train_loss']) + 1)

    plt.plot(epochs_range, history['train_loss'], color='purple', label='Train Loss', linestyle='-')
    plt.plot(epochs_range, history['val_loss'], color='teal', label='Val Loss', linestyle='--')

    plt.title('Generator Loss History')
    plt.xlabel('Epochs')
    plt.ylabel('Cross Entropy Loss')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.show()


# ============================================================
# Step 3: candidate sequence sanity check (do the generated sequences look like real proteins?)
# ============================================================

def aa_frequency(seqs):
    c, total = Counter(), 0
    for s in seqs:
        c.update(s)
        total += len(s)
    return np.array([c.get(a, 0) / max(total, 1) for a in STANDARD_AMINO_ACIDS])


def seq_stats(s):
    cnt = Counter(s)
    n = len(s)
    distinct = len(cnt)  # how many distinct amino acids are used (more = more protein-like)
    max_frac = max(cnt.values()) / n  # the highest single-amino-acid fraction (higher = more likely degenerate)
    return distinct, max_frac


def safe_corr(a, b):
    """corrcoef returns nan when variance is 0 (e.g. all lengths identical); return None instead."""
    if np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def plot_candidate_sanity_check(results, real_seqs):
    """Before spending a lot of compute on ESMFold, first confirm the generated sequences aren't
    degenerate/nonsensical, and check whether the GNN score is just riding a shortcut based on
    length/composition (reward shortcut).

    Args:
      results: the return value of design_binders(), a list of (rank, score, is_novel, seq)
      real_seqs: real sequences from the training data (list[str]), used as a reference
    """
    cand_seqs = [r[3] for r in results]
    real_freq, gen_freq = aa_frequency(real_seqs), aa_frequency(cand_seqs)

    scores = np.array([r[1] for r in results])
    lengths = np.array([len(r[3]) for r in results])
    distinct = np.array([seq_stats(r[3])[0] for r in results])
    max_frac = np.array([seq_stats(r[3])[1] for r in results])
    real_len = np.array([len(s) for s in real_seqs])

    print(f"Number of candidates: {len(results)}")
    print(f"Length             : generated mean {lengths.mean():.1f}  | real protein mean {real_len.mean():.1f}")
    print(f"Distinct length values: {len(np.unique(lengths))} (if =1, all candidates are the same length — generator is barely emitting <eos>)")
    print(f"Amino acids used   : mean {distinct.mean():.1f} / 20")
    print(f"Max single-AA fraction: mean {max_frac.mean():.2f}  (>0.4 usually indicates low complexity/degeneracy)")
    print(f"Suspected degenerate sequences (max_frac>0.4): {(max_frac > 0.4).sum()} / {len(results)}")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    xpos = np.arange(len(STANDARD_AMINO_ACIDS))
    axes[0, 0].bar(xpos - 0.2, real_freq, width=0.4, label='Real', color='steelblue')
    axes[0, 0].bar(xpos + 0.2, gen_freq, width=0.4, label='Generated', color='coral')
    axes[0, 0].set_xticks(xpos)
    axes[0, 0].set_xticklabels(STANDARD_AMINO_ACIDS)
    axes[0, 0].set_title('Amino-acid composition')
    axes[0, 0].legend()

    axes[0, 1].hist(real_len, bins=40, range=(0, 400), alpha=0.6, density=True, label='Real', color='steelblue')
    axes[0, 1].hist(lengths, bins=20, alpha=0.6, density=True, label='Generated', color='coral')
    axes[0, 1].set_title('Length distribution')
    axes[0, 1].set_xlabel('length')
    axes[0, 1].legend()

    axes[1, 0].scatter(lengths, scores, s=12, alpha=0.5)
    axes[1, 0].set_xlabel('length')
    axes[1, 0].set_ylabel('GNN score')
    axes[1, 0].set_title('Score vs length (ideally no clear trend)')

    axes[1, 1].scatter(max_frac, scores, s=12, alpha=0.5, color='purple')
    axes[1, 1].set_xlabel('max single-AA fraction')
    axes[1, 1].set_ylabel('GNN score')
    axes[1, 1].set_title('Score vs degeneracy')

    plt.tight_layout()
    plt.show()

    c_len = safe_corr(lengths, scores)
    c_deg = safe_corr(max_frac, scores)
    len_msg = "N/A (all candidates have the same length — generator is barely emitting the <eos> stop token)" if c_len is None else f"{c_len:+.3f}"
    deg_msg = "N/A" if c_deg is None else f"{c_deg:+.3f}"
    print(f"\nCorrelation of GNN score vs. length    : {len_msg}")
    print(f"Correlation of GNN score vs. degeneracy: {deg_msg}")
    print("(Closer to 0 is better; a large value suggests the GNN may just be scoring on these surface features.)")
