import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, f1_score, hamming_loss, roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import joblib
from src.models.utils import load_dataset, setup_predict
from src.config import CACHE_DIR, ENC_PATH, MLB_PATH, PREDICTIONS_DIR, TACTIC_WHITELIST

# ---- Config ----
MODEL_PATH = Path(os.path.join(CACHE_DIR, 'mlp_model.pt'))

# ---- Model ----
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))


# ---- Train Model ----
def train_mlp_kfold(epochs=10, batch_size=32, lr=1e-3, n_splits=5):
    X_all, y_all, df_all, enc, mlb = load_dataset("concat", merge_train_val=True)

    input_dim = X_all.shape[1]
    output_dim = y_all.shape[1]
    oof_preds = np.zeros_like(y_all, dtype=np.float32)
    metrics_log = {}

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_all)):
        print(f"\n=== Fold {fold+1}/{n_splits} ===")

        model = MLP(input_dim, output_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.BCELoss()

        X_train, y_train = X_all[train_idx], y_all[train_idx]
        X_val, y_val = X_all[val_idx], y_all[val_idx]

        train_loader = DataLoader(
            TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)),
            batch_size=batch_size,
            shuffle=True
        )

        # Train model
        for _ in range(epochs):
            model.train()
            for xb, yb in train_loader:
                optimizer.zero_grad()
                preds = model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()

        # Validation predictions for OOF
        model.eval()
        with torch.no_grad():
            val_inputs = torch.tensor(X_val, dtype=torch.float32)
            val_outputs = model(val_inputs).numpy()
            oof_preds[val_idx] = val_outputs

            val_preds_bin = (val_outputs > 0.5).astype(int)
            print(f"[Fold {fold+1}] Classification Report:")
            print(classification_report(y_val, val_preds_bin, target_names=mlb.classes_))

            # Compute metrics
            fold_metrics = {}
            fold_metrics["f1_micro"] = f1_score(y_val, val_preds_bin, average="micro", zero_division=0)
            fold_metrics["f1_macro"] = f1_score(y_val, val_preds_bin, average="macro", zero_division=0)
            fold_metrics["hamming_loss"] = hamming_loss(y_val, val_preds_bin)

            try:
                fold_metrics["auc_macro"] = roc_auc_score(y_val, val_outputs, average="macro")
                fold_metrics["auc_micro"] = roc_auc_score(y_val, val_outputs, average="micro")
            except ValueError as e:
                print(f"[Fold {fold+1}] ROC AUC could not be computed: {e}")
                fold_metrics["auc_macro"] = None
                fold_metrics["auc_micro"] = None

            metrics_log[f"fold_{fold+1}"] = fold_metrics
            print(f"[Fold {fold+1}] Metrics:", json.dumps(fold_metrics, indent=2))
            
    # Save final model trained on full data
    print("\n=== Training final model on full data ===")
    final_model = MLP(input_dim, output_dim)
    optimizer = torch.optim.Adam(final_model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    full_loader = DataLoader(
        TensorDataset(torch.tensor(X_all, dtype=torch.float32), torch.tensor(y_all, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=True
    )
    for _ in range(epochs):
        final_model.train()
        for xb, yb in full_loader:
            optimizer.zero_grad()
            preds = final_model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

    torch.save(final_model.state_dict(), MODEL_PATH)
    joblib.dump(mlb, MLB_PATH)
    joblib.dump(enc, ENC_PATH)

    # Save OOF predictions
    oof_df = pd.DataFrame(oof_preds, columns=mlb.classes_)
    oof_df.insert(0, "url", df_all["url"])
    oof_df.insert(0, "id", df_all.index)
    oof_path = (Path(PREDICTIONS_DIR) / MODEL_PATH.stem).with_suffix(".csv")
    oof_df.to_csv(oof_path, index=False)
    print(f"\nOOF predictions saved to: {oof_path}")

    # Save metrics
    with open((Path(PREDICTIONS_DIR) / MODEL_PATH.stem).with_suffix(".metrics.kfold.json"), "w") as metrics_file:
        json.dump(metrics_log, metrics_file, indent=4)
    return final_model, enc, mlb

# ---- Train Model ----
def train_mlp(epochs=10, batch_size=32, lr=1e-3):
    X_all, y_all, df_all, enc, mlb = load_dataset("concat", merge_train_val=True)

    input_dim = X_all.shape[1]
    output_dim = y_all.shape[1]

    train_loader = DataLoader(TensorDataset(torch.tensor(X_all, dtype=torch.float32), torch.tensor(y_all, dtype=torch.float32)), batch_size=batch_size, shuffle=True)

    model = MLP(input_dim, output_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

        model.eval()
    torch.save(model.state_dict(), MODEL_PATH)
    joblib.dump(mlb, MLB_PATH)
    joblib.dump(enc, ENC_PATH)
    return model, enc, mlb

# ---- Inference ----
def predict_mlp(df, embed_features, model=None, enc=None, mlb=None, return_probs=False):
    X_all, enc, mlb = setup_predict(df, embed_features, enc, mlb)

    input_dim = X_all.shape[1]
    output_dim = len(mlb.classes_)

    if model is None:
        model = MLP(input_dim, output_dim)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        model.eval()

    with torch.no_grad():
        inputs = torch.tensor(X_all, dtype=torch.float32)
        outputs = model(inputs)
        probs = outputs.numpy()
        if return_probs:
            preds = (probs > 0.5).astype(int)
            return probs, preds
        return mlb.inverse_transform((probs > 0.5).astype(int))

if __name__ == '__main__':
    train_mlp()