"""Step 3 validation (2): use ESMFold to check candidate foldability (pLDDT).

pLDDT > 70: confident structure; 50-70: moderate; < 50: low confidence, likely doesn't fold at all.
The ideal candidate has both a high GNN score and a high pLDDT.
Requires a GPU; facebook/esmfold_v1 is ~2.8GB and will be downloaded on first run.
"""
import os
import random

import torch
from transformers import AutoTokenizer, EsmForProteinFolding
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37
from transformers.models.esm.openfold_utils.protein import Protein as OFProtein
from transformers.models.esm.openfold_utils.protein import to_pdb

from . import config

# Amino-acid alphabet used to build the random-shuffle negative control
AMINO_ACID_LETTERS = "ARNDCQEGHILKMFPSTWYV"


def load_esmfold_model(device, model_name=config.ESMFOLD_MODEL_NAME):
    print(f"Loading ESMFold ({model_name}); the first run downloads ~2.8GB, please wait...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = EsmForProteinFolding.from_pretrained(model_name, low_cpu_mem_usage=True)
    model = model.to(device)
    model.esm = model.esm.half()  # run the ESM backbone in half precision to save VRAM
    model.trunk.set_chunk_size(64)  # chunked computation to lower peak VRAM usage
    model.eval()
    print("ESMFold loaded.")
    return model, tokenizer


def convert_outputs_to_pdb(outputs):
    """Convert ESMFold outputs into a PDB string (adapted from the HuggingFace official example)."""
    final_atom_positions = atom14_to_atom37(outputs["positions"][-1], outputs)
    outputs = {k: v.to("cpu").numpy() for k, v in outputs.items()}
    final_atom_positions = final_atom_positions.cpu().numpy()
    final_atom_mask = outputs["atom37_atom_exists"]
    pdbs = []
    for i in range(outputs["aatype"].shape[0]):
        pred = OFProtein(
            aatype=outputs["aatype"][i],
            atom_positions=final_atom_positions[i],
            atom_mask=final_atom_mask[i],
            residue_index=outputs["residue_index"][i] + 1,
            b_factors=outputs["plddt"][i],
            chain_index=outputs["chain_index"][i] if "chain_index" in outputs else None,
        )
        pdbs.append(to_pdb(pred))
    return pdbs


@torch.no_grad()
def fold_sequence(seq, esmfold_model, esmfold_tokenizer, device):
    tok = esmfold_tokenizer([seq], return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    out = esmfold_model(tok)
    plddt = out["plddt"]  # [1, L, 37] per-atom pLDDT
    mask = out["atom37_atom_exists"]  # [1, L, 37] whether that atom exists
    mean_plddt = float((plddt * mask).sum() / mask.sum())  # average only over atoms that actually exist
    # Guard: some transformers/model versions emit pLDDT on a 0-1 scale instead of the
    # standard 0-100 scale. Detect this from the raw max value and rescale, so the 0-100
    # confidence thresholds used downstream (>70 confident, >50 moderate) stay meaningful.
    if float(plddt.max()) <= 1.5:
        mean_plddt *= 100.0
    pdb_str = convert_outputs_to_pdb(out)[0]
    return mean_plddt, pdb_str


def fold_top_candidates(results, esmfold_model, esmfold_tokenizer, device, target_node_id,
                         top_n=5, output_dir=config.FOLDED_PDB_DIR):
    """Fold the top_n ranked results returned by design_binders() and save them as PDB files."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"Folding the top {top_n} candidates (target node {target_node_id})...\n")
    fold_results = []
    for rank, score, is_novel, seq in results[:top_n]:
        mean_plddt, pdb_str = fold_sequence(seq, esmfold_model, esmfold_tokenizer, device)
        pdb_path = os.path.join(output_dir, f"cand{rank}_plddt{mean_plddt:.1f}.pdb")
        with open(pdb_path, "w") as f:
            f.write(pdb_str)
        fold_results.append((rank, score, mean_plddt, is_novel, len(seq), pdb_path))
        tag = "novel" if is_novel else "in_train"
        verdict = "confident fold" if mean_plddt > 70 else ("moderate" if mean_plddt > 50 else "low confidence / likely unfolded")
        print(f"#{rank:2d} | GNN={score:.3f} | pLDDT={mean_plddt:5.1f} ({verdict}) | {tag} | len={len(seq)} -> {pdb_path}")

    print("\n=== Summary ===")
    print(f"{'rank':>4} {'GNN':>6} {'pLDDT':>6} {'len':>4}  novelty")
    for rank, score, mean_plddt, is_novel, L, _ in fold_results:
        print(f"{rank:>4} {score:>6.3f} {mean_plddt:>6.1f} {L:>4}  {'novel' if is_novel else 'in_train'}")
    print(f"\nPDB files saved to {output_dir}/ — download and view them with PyMOL/ChimeraX, or upload to https://molstar.org/viewer.")
    print("Interpretation: pLDDT>70 means a confident structure; candidates with both a high GNN score and a high pLDDT are the best ones to follow up with complex prediction against the target (AlphaFold-Multimer).")
    return fold_results


def validate_plddt_scale(esmfold_model, esmfold_tokenizer, device, real_seq,
                          output_dir=config.FOLDED_PDB_DIR, seed=0):
    """Sanity-check the pLDDT scale and pipeline: a real protein as the positive control, a
    random shuffle as the negative control.

    - The real protein should score clearly higher than the random shuffle (typically ~80-90 vs ~30-40).
    - If so -> the pLDDT scale and pooling are correct, and the earlier candidate scores are trustworthy.
    - If the real protein also scores low -> it's a pooling/scale bug, not bad sequences — go back and fix it.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1) Inspect the raw pLDDT range directly -> determine the scale (0-1 or 0-100)
    tok = esmfold_tokenizer([real_seq], return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    with torch.no_grad():
        out = esmfold_model(tok)
    p = out["plddt"]
    print(f"raw plddt  shape={tuple(p.shape)}  min={float(p.min()):.3f}  max={float(p.max()):.3f}  mean={float(p.mean()):.3f}")
    print("-> if max ≈ 1, the scale is 0-1 (fold_sequence() will auto-correct this with x100)")
    print("-> if max ≈ 100, the scale is already 0-100\n")

    # 2) Use fold_sequence (with its automatic scaling) to compute the positive control
    real_plddt, real_pdb = fold_sequence(real_seq, esmfold_model, esmfold_tokenizer, device)
    with open(os.path.join(output_dir, "REAL_control.pdb"), "w") as f:
        f.write(real_pdb)
    print(f"Real protein (positive control) mean_plddt = {real_plddt:5.1f}  (len={len(real_seq)})")

    # 3) Negative control: a random shuffle of the same length
    random.seed(seed)
    junk = "".join(random.choice(AMINO_ACID_LETTERS) for _ in range(len(real_seq)))
    junk_plddt, _ = fold_sequence(junk, esmfold_model, esmfold_tokenizer, device)
    print(f"Random shuffle (negative control) mean_plddt = {junk_plddt:5.1f}  (len={len(junk)})")

    return real_plddt, junk_plddt
