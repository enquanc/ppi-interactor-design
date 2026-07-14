"""Step 3 main script: design de novo interactors -> quality sanity check -> ESMFold structural validation.

Prerequisites: run run_step1_train_gnn.py and run_step2_train_generator.py first;
               checkpoints/ must contain graph_data.pt, best_model.pth, best_generator.pth.

Usage:
    python scripts/run_step3_design_and_fold.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from src import config
from src.data import load_esm_model
from src.design_binders import design_binders, select_target_by_degree
from src.fold_validation import fold_top_candidates, load_esmfold_model, validate_plddt_scale
from src.visualize import plot_candidate_sanity_check

# Real protein sequence used as the positive control for validating the pLDDT scale
# (can be swapped for any protein known to fold well)
REAL_CONTROL_SEQUENCE = (
    "MARPHPWWLCVLGTLVGLSATPAPKSCPERHYWAQGKLCCQMCEPGTFLVKDCDQHRKAAQCDPCIPGVSFSPDHHTRPHCESCRHCNSGLLVRN"
    "CTITANAECACRNGWQCRDKECTECDPLPNPSLTARSSQALSPHPQPTHLPYVSEMLEARTAGHMQTLADFRQLPARTLSTHWPPQRSLCSSDFIRIL"
    "VIFSGMFLVFTLAGALFLHQRRKYRSNKGESPVEPAEPCRYSCPREEEGSTIPIQEDYRKPEPACSP"
)


def main():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    device = config.DEVICE

    graph_data = torch.load(config.GRAPH_DATA_PATH, map_location=device)
    x = graph_data["x"]
    pos_edges = graph_data["pos_edges"]
    sequence_mapping = graph_data["sequence_mapping"]

    esm_model, esm_tokenizer = load_esm_model(config.ESM_MODEL_NAME, device)

    # 1. Pick the target (the node with the highest degree)
    target_node_id, _ = select_target_by_degree(pos_edges, sequence_mapping, top_n=10)

    # 2. Generate candidates -> filter -> rank with the GNN -> export FASTA
    results = design_binders(
        target_node_id=target_node_id,
        x=x, pos_edges=pos_edges, sequence_mapping=sequence_mapping,
        esm_model=esm_model, esm_tokenizer=esm_tokenizer, device=device,
        n_candidates=200, keep_top=10,
        fasta_path=config.DESIGNED_BINDERS_FASTA,
    )
    if not results:
        return

    # 3. Candidate sequence sanity check (composition/length/degeneracy, to catch the GNN score
    #    just taking a shortcut)
    real_seqs = list(sequence_mapping.keys())
    plot_candidate_sanity_check(results, real_seqs)

    # 4. ESMFold structural validation (pLDDT): fold the top 5 candidates
    esmfold_model, esmfold_tokenizer = load_esmfold_model(device)
    fold_top_candidates(results, esmfold_model, esmfold_tokenizer, device,
                         target_node_id=target_node_id, top_n=5, output_dir=config.FOLDED_PDB_DIR)

    # 5. Validate the pLDDT scale: real protein (positive control) vs. random shuffle (negative control)
    validate_plddt_scale(esmfold_model, esmfold_tokenizer, device, REAL_CONTROL_SEQUENCE,
                          output_dir=config.FOLDED_PDB_DIR)


if __name__ == "__main__":
    main()
