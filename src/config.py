"""Global settings: paths, device, shared hyperparameters."""
import os
import torch

# --- Paths ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
FOLDED_PDB_DIR = os.path.join(OUTPUT_DIR, "folded_pdb")

GNN_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")
GENERATOR_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_generator.pth")
GRAPH_DATA_PATH = os.path.join(CHECKPOINT_DIR, "graph_data.pt")
DESIGNED_BINDERS_FASTA = os.path.join(OUTPUT_DIR, "designed_binders.fasta")

# --- Kaggle dataset ---
KAGGLE_DATASET = "spandansureja/ppi-dataset"

# --- Device ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- ESM-2 (Step 1 feature extraction / Step 3 candidate features) ---
ESM_MODEL_NAME = "facebook/esm2_t12_35M_UR50D"
ESM_FEATURE_DIM = 480  # hidden_size of esm2_t12_35M_UR50D

# --- GNN (Step 1) ---
GNN_HIDDEN_CHANNELS = 256
GNN_OUT_CHANNELS = 128
GNN_LR = 5e-4
GNN_EPOCHS = 500
GNN_VAL_RATIO = 0.1
GNN_TEST_RATIO = 0.2
GNN_DISJOINT_TRAIN_RATIO = 0.3

# --- Generator (Step 2) ---
GEN_BATCH_SIZE = 64
GEN_MAX_SEQ_LEN = 128
GEN_EPOCHS = 50
GEN_LR = 5e-4
GEN_D_MODEL = 128
GEN_NHEAD = 4
GEN_NUM_LAYERS = 3

# --- ESMFold (Step 3 structural validation) ---
ESMFOLD_MODEL_NAME = "facebook/esmfold_v1"
