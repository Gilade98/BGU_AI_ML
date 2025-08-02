# ---- Config ----
import json
import os
import joblib
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder

from src.config import PROCESSED_DIR, CACHE_DIR, CAT_FEATURES, TACTIC_WHITELIST, ENC_PATH, MLB_PATH, PREDICTIONS_DIR
from pathlib import Path

from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, f1_score, hamming_loss, roc_auc_score


def load_dataset(embedding: str = "concat", merge_train_val: bool = False):
    """
    embedding: one of ['attackbert', 'secbert', 'tfidf', 'concat']
    returns: 
        if merge_train_val:
            X_all, y_all, df_all, enc, mlb
        else:
            X_train, X_val, X_test, y_train, y_val, y_test, df_train, df_val, df_test, enc, mlb
    """
    # Load raw dataframes
    df_train = pd.read_parquet(os.path.join(PROCESSED_DIR, "train.parquet"))
    df_val = pd.read_parquet(os.path.join(PROCESSED_DIR, "val.parquet"))
    df_test = pd.read_parquet(os.path.join(PROCESSED_DIR, "test.parquet"))

    def load_embed(name):
        return (
            np.load(os.path.join(CACHE_DIR, f"train_{name}.npy")),
            np.load(os.path.join(CACHE_DIR, f"val_{name}.npy")),
            np.load(os.path.join(CACHE_DIR, f"test_{name}.npy"))
        )

    # Load embeddings
    if embedding == "concat":
        X_train_list, X_val_list, X_test_list = [], [], []
        for name in ["attackbert", "secbert", "tfidf_svd"]:
            X_tr, X_vl, X_te = load_embed(name)
            X_train_list.append(X_tr)
            X_val_list.append(X_vl)
            X_test_list.append(X_te)
        X_train = np.concatenate(X_train_list, axis=1)
        X_val = np.concatenate(X_val_list, axis=1)
        X_test = np.concatenate(X_test_list, axis=1)
    else:
        if embedding == "tfidf":
            embedding = "tfidf_svd"
        X_train, X_val, X_test = load_embed(embedding)

    # Metadata encoding
    for df in [df_train, df_val, df_test]:
        df[CAT_FEATURES] = df[CAT_FEATURES].fillna("missing")

    enc_path = os.path.join(CACHE_DIR, "meta_encoder.pkl")
    if os.path.exists(enc_path):
        enc = joblib.load(enc_path)
    else:
        enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        enc.fit(df_train[CAT_FEATURES])
        joblib.dump(enc, enc_path)

    X_train_meta = enc.transform(df_train[CAT_FEATURES])
    X_val_meta = enc.transform(df_val[CAT_FEATURES])
    X_test_meta = enc.transform(df_test[CAT_FEATURES])

    X_train = np.concatenate([X_train, X_train_meta], axis=1) #type: ignore
    X_val = np.concatenate([X_val, X_val_meta], axis=1) #type: ignore
    X_test = np.concatenate([X_test, X_test_meta], axis=1) #type: ignore

    # Multi-label targets
    mlb_path = os.path.join(CACHE_DIR, "mlb_labels.pkl")
    y_field = "tactic_ids"
    
    if os.path.exists(mlb_path):
        mlb = joblib.load(mlb_path)
    else:
        mlb = MultiLabelBinarizer(classes=TACTIC_WHITELIST)
        mlb.fit([[]]) # Fit on empty list to initialize all classes
        joblib.dump(mlb, mlb_path)

    print("Loaded classes:", mlb.classes_)

    y_train = mlb.transform(df_train[y_field])
    y_val = mlb.transform(df_val[y_field])
    y_test = mlb.transform(df_test[y_field])

    label_sums = y_train.sum(axis=0)  # Total number of positives per class
    label_names = mlb.classes_ if hasattr(mlb, "classes_") else [f"label_{i}" for i in range(y_train.shape[1])]

    for i, total in enumerate(label_sums):
        if total == 0:
            print(f"[WARN] Label '{label_names[i]}' has no positive examples in y_train.")
        elif total == y_train.shape[0]:
            print(f"[WARN] Label '{label_names[i]}' has only positive examples in y_train.")

    if merge_train_val:
        X_all = np.concatenate([X_train, X_val], axis=0)
        y_all = np.concatenate([y_train, y_val], axis=0)
        df_all = pd.concat([df_train, df_val], axis=0).reset_index(drop=True)
        return X_all, y_all, df_all, enc, mlb
    
    return X_train, X_val, X_test, y_train, y_val, y_test, df_train, df_val, df_test, enc, mlb


def load_embed_only(feature_type="concat", split="train"):
    """
    Load embeddings by type: 'attackbert', 'secbert', 'tfidf_svd', or 'concat'.
    """
    split = split.lower()
    assert split in ["train", "val", "test"], f"Invalid split: {split}"

    if feature_type == "concat":
        embed_list = []
        for sub_type in ["attackbert", "secbert", "tfidf_svd"]:
            path = os.path.join(CACHE_DIR, f"{split}_{sub_type}.npy")
            assert os.path.exists(path), f"Missing: {path}"
            embed_list.append(np.load(path))
        return np.concatenate(embed_list, axis=1)

    else:
        path = os.path.join(CACHE_DIR, f"{split}_{feature_type}.npy")
        assert os.path.exists(path), f"Missing: {path}"
        return np.load(path)


def setup_predict(df, embed_features, enc=None, mlb=None):
    if enc is None:
        enc = joblib.load(ENC_PATH)
    if mlb is None:
        mlb = joblib.load(MLB_PATH)

    df[CAT_FEATURES] = df[CAT_FEATURES].fillna("missing")
    X_meta = enc.transform(df[CAT_FEATURES])
    X_all = np.concatenate([embed_features, X_meta], axis=1)

    return X_all, enc, mlb

def predict_model(df, embed_features, model, enc=None, mlb=None, return_probs=False):
    X_all, enc, mlb = setup_predict(df, embed_features, enc, mlb)

    if return_probs:
        probs_matrix = model.predict_proba(X_all)  # List of arrays, one per class

        # Convert list of (n_samples, ) arrays to (n_samples, n_classes)
        # If it's a list of (n_samples, 2) arrays per class → stack
        if isinstance(probs_matrix, list) and hasattr(probs_matrix[0], 'shape') and probs_matrix[0].shape[1] == 2:
            probs_matrix = np.stack([p[:, 1] for p in probs_matrix], axis=1)
        elif isinstance(probs_matrix, np.ndarray):
            # Already in shape (n_samples, n_classes), so we keep as-is
            pass
        else:
            raise ValueError("Unexpected output format from predict_proba.")

        preds_matrix = (probs_matrix > 0.5).astype(int)
        return probs_matrix, preds_matrix

    # Default: return inverse-transformed label sets (used for ensemble)
    preds_matrix = model.predict(X_all)
    return mlb.inverse_transform((preds_matrix > 0.5).astype(int))

def train_model(model, model_path:Path):
    X_train, X_val, _, y_train, y_val, _, _, _, _, enc, mlb = load_dataset("concat")

    model.fit(X_train, y_train)

    val_preds = model.predict(X_val)
    print(f"Model ({model_path.name}) Validation Report:")
    print(classification_report(y_val, val_preds, target_names=mlb.classes_))

    if getattr(model, "save_model", None):
        model.save_model(model_path)
    else:
        joblib.dump(model, model_path)
    joblib.dump(mlb, MLB_PATH)
    joblib.dump(enc, ENC_PATH)
    return model, enc, mlb

def train_model_with_oof(model, model_path:Path, n_splits=5):
    X_all, y_all, df_all, enc, mlb = load_dataset("concat", merge_train_val=True)

    oof_preds = np.zeros_like(y_all, dtype=float)
    metrics_log = {}
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_all)):
        print(f"\n=== Fold {fold + 1}/{n_splits} ===")
        X_train, X_val = X_all[train_idx], X_all[val_idx]
        y_train, y_val = y_all[train_idx], y_all[val_idx]

        model.fit(X_train, y_train)

        if hasattr(model, "predict_proba"):
            preds_proba = model.predict_proba(X_val)

            if isinstance(preds_proba, list):
                # For sklearn's MultiOutputClassifier
                preds_matrix = np.stack([p[:, 1] for p in preds_proba], axis=1)
            else:
                preds_matrix = preds_proba
        else:
            preds_matrix = model.predict(X_val)
            if preds_matrix.ndim == 1:
                preds_matrix = preds_matrix[:, np.newaxis]

        oof_preds[val_idx] = preds_matrix

        y_val_bin = (preds_matrix > 0.5).astype(int)
        print(classification_report(y_val, y_val_bin, target_names=mlb.classes_))

        # Compute and store metrics
        fold_metrics = {
            "f1_micro": f1_score(y_val, y_val_bin, average="micro", zero_division=0),
            "f1_macro": f1_score(y_val, y_val_bin, average="macro", zero_division=0),
            "hamming_loss": hamming_loss(y_val, y_val_bin),
        }

        try:
            fold_metrics["auc_macro"] = roc_auc_score(y_val, preds_matrix, average="macro")
            fold_metrics["auc_micro"] = roc_auc_score(y_val, preds_matrix, average="micro")
        except ValueError as e:
            print(f"[Fold {fold+1}] ROC AUC could not be computed: {e}")
            fold_metrics["auc_macro"] = None
            fold_metrics["auc_micro"] = None

        metrics_log[f"fold_{fold + 1}"] = fold_metrics
    # Save model trained on all data
    print("\n=== Training final model on all data ===")
    model.fit(X_all, y_all)

    if getattr(model, "save_model", None):
        model.save_model(model_path)
    else:
        joblib.dump(model, model_path)

    joblib.dump(enc, ENC_PATH)
    joblib.dump(mlb, MLB_PATH)

    # Save OOF predictions
    oof_df = pd.DataFrame(oof_preds, columns=mlb.classes_)
    oof_df.insert(0, "url", df_all["url"])
    oof_df.insert(0, "id", df_all.index)
    oof_path = (Path(PREDICTIONS_DIR) / f"oof_predictions_{model_path.stem}").with_suffix(".csv")
    oof_df.to_csv(oof_path, index=False)
    print(f"\nOOF predictions saved to: {oof_path}")

    # Save metrics
    metrics_path = (Path(PREDICTIONS_DIR) / model_path.stem).with_suffix(".metrics.kfold.json")
    with open(metrics_path, "w") as metrics_file:
        json.dump(metrics_log, metrics_file, indent=4)
        print(f"Metrics saved to: {metrics_path}")

    return model, enc, mlb

def create_graphs():
    metrics_to_plot = ["f1_macro", "f1_micro", "hamming_loss", "auc_macro", "auc_micro"]
    metrics_dir = Path(PREDICTIONS_DIR) / "metrics"
    all_models_folds = {}

    for json_file in metrics_dir.glob("*.metrics.kfold.json"):
        model_name = json_file.stem.split("_")[0]

        with json_file.open("r") as f:
            data = json.load(f)
        
        # Convert folds to DataFrame
        df = pd.DataFrame(data).T.astype(float)  # folds as rows
        all_models_folds[model_name] = df

    # --- Plot line graph per metric ---
    for metric in metrics_to_plot:
        plt.figure(figsize=(10, 6))
        
        for model_name, df in all_models_folds.items():
            plt.plot(df.index, df[metric], marker='o', label=model_name)
        
        plt.title(f"K-Fold {metric} per Model")
        plt.xlabel("Fold")
        plt.ylabel(metric)
        plt.legend(title="Model")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.xticks(rotation=45)
        plt.tight_layout()
        # Save the figure as PNG in metrics_dir
        output_path = metrics_dir / f"{metric}.png"
        plt.savefig(output_path)
        plt.close()

if __name__ == "__main__":
    create_graphs()

    