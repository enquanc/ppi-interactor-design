"""Step 1: training and evaluation logic for the GNN (LinkPredictor)."""
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from . import config


def train_and_validate(model, optimizer, criterion, train_data, val_data, epochs=100,
                        checkpoint_path=config.GNN_MODEL_PATH):
    print("Starting model training...")
    best_val_auc = 0.0

    # Dict to store the training history
    history = {
        'train_loss': [],
        'val_epochs': [],  # validation isn't run every epoch, so track the epoch separately
        'val_auc': []
    }

    for epoch in tqdm(range(1, epochs + 1)):
        # --- [Training phase] ---
        model.train()
        optimizer.zero_grad()

        out = model(train_data.x, train_data.edge_index, train_data.edge_label_index)
        loss = criterion(out, train_data.edge_label.float())

        loss.backward()
        optimizer.step()

        # Record the train loss for this epoch
        history['train_loss'].append(loss.item())

        # --- [Validation phase] ---
        if epoch % 10 == 0:
            val_auc = evaluate_model(model, val_data)
            print(f'Epoch: {epoch:03d} | Train Loss: {loss.item():.4f} | Validation AUC: {val_auc:.4f}')

            # Record the validation AUC and its corresponding epoch
            history['val_epochs'].append(epoch)
            history['val_auc'].append(val_auc)

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                torch.save(model.state_dict(), checkpoint_path)

    print("Training finished!")
    # Return the history alongside the model
    return model, history


@torch.no_grad()
def evaluate_model(model, data):
    model.eval()
    out = model(data.x, data.edge_index, data.edge_label_index)
    pred = torch.sigmoid(out).cpu().numpy()
    target = data.edge_label.cpu().numpy()
    return roc_auc_score(target, pred)
