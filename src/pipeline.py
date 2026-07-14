"""End-to-end demo pipeline that closes out Step 2: Generator produces a new sequence -> ESM turns it
into features -> GNN scores the interaction probability."""
import torch

from . import config
from .models import LinkPredictor, ProteinMiniGPT, VOCAB_SIZE
from .train_generator import generate_new_protein


def run_end_to_end_pipeline(original_graph_data, target_node_id, esm_model, tokenizer, device,
                             generator_weight=config.GENERATOR_MODEL_PATH,
                             gnn_weight=config.GNN_MODEL_PATH):
    print("=== Launching the GenAI + Bio-Informatics end-to-end pipeline ===")

    # ---------------------------------------------------------
    # Load the pretrained Generator and the GNN judge
    # ---------------------------------------------------------
    print("\nLoading AI model weights...")

    # Initialize and load the Mini-GPT (Generator)
    generator = ProteinMiniGPT(vocab_size=VOCAB_SIZE, d_model=config.GEN_D_MODEL,
                                nhead=config.GEN_NHEAD, num_layers=config.GEN_NUM_LAYERS).to(device)
    generator.load_state_dict(torch.load(generator_weight, map_location=device))

    # Initialize and load the GNN (Discriminator / Evaluator)
    gnn_model = LinkPredictor(in_channels=config.ESM_FEATURE_DIM, hidden_channels=config.GNN_HIDDEN_CHANNELS,
                               out_channels=config.GNN_OUT_CHANNELS).to(device)
    gnn_model.load_state_dict(torch.load(gnn_weight, map_location=device))

    # ---------------------------------------------------------
    # Generator creates a brand-new protein sequence
    # ---------------------------------------------------------
    print("\nCalling the Generator to produce a candidate sequence...")
    new_sequence = generate_new_protein(generator, max_len=100, temperature=0.8, top_k=5)
    print(f"Successfully created a brand-new protein sequence:\n{new_sequence}")

    # Guard: abort immediately on an empty sequence, otherwise the ESM pooling below produces NaN
    if len(new_sequence) == 0:
        print("Generator produced an empty sequence, cannot evaluate — stopping early.")
        return

    with torch.no_grad():
        inputs = tokenizer(new_sequence, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = esm_model(**inputs)

        # Mean pooling to get the 480-dim vector
        seq_len = inputs['attention_mask'][0].sum().item()
        # Guard: if the sequence is too short (only special tokens left), average over all valid
        # tokens instead, to avoid an empty slice producing NaN
        if seq_len > 2:
            new_node_feature = outputs.last_hidden_state[0, 1:seq_len - 1].mean(dim=0)  # shape: [480]
        else:
            new_node_feature = outputs.last_hidden_state[0, :seq_len].mean(dim=0)

    # ---------------------------------------------------------
    # Dynamic graph manipulation
    # ---------------------------------------------------------
    print("\nDynamically inserting the new node into the original PyG graph topology...")
    original_x = original_graph_data.x.to(device)

    # Append the new node's feature to the end of the original feature matrix
    x_extended = torch.cat([original_x, new_node_feature.unsqueeze(0)], dim=0)
    new_node_id = x_extended.size(0) - 1  # the new node's ID is the last index

    # Build the test edge (edge_label_index): [new node ID, target node ID]
    # We want to predict whether the newly created protein interacts with the given target_node_id
    test_edge_index = torch.tensor([[new_node_id], [target_node_id]], dtype=torch.long, device=device)

    # ---------------------------------------------------------
    # The GNN judge makes the final call
    # ---------------------------------------------------------
    print("Asking the GNN judge for the final score...")
    with torch.no_grad():
        # Pass in the extended x_extended, but keep edge_index as the original topology (the new
        # node has no established connections in the graph yet)
        out = gnn_model(x_extended, original_graph_data.edge_index.to(device), test_edge_index)
        interaction_prob = torch.sigmoid(out).item()

    print("\n==================================================")
    print("Evaluation complete!")
    print(f"Predicted interaction probability between the new sequence and the target: {interaction_prob * 100:.2f}%")
    print("==================================================")

    return new_sequence, interaction_prob
