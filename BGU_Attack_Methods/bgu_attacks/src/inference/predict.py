import os
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from src.config import PREDICTIONS_DIR, CACHE_DIR
from src.models.train_catboost import predict_catboost
from src.models.train_logres import predict_logistic
from src.models.train_rf import predict_rf
from src.models.train_lgbm import predict_lgbm
from src.models.train_mlp import predict_mlp
from src.models.train_knn import predict_knn
from src.models.utils import load_embed_only
from sklearn.metrics import classification_report, precision_score, roc_auc_score
import joblib
from sklearn.metrics import roc_curve, auc

MODEL_PREDICTORS = {
    "catboost": predict_catboost,
    "knn": predict_knn,
    "lgbm": predict_lgbm,
    "logistic": predict_logistic,
    "mlp": predict_mlp,
    "rf": predict_rf,
}

def run_model_predictions(df, mlb_classes):
    embed_features = load_embed_only("concat", split="test")
    results = []

    for model_name, predict_fn in MODEL_PREDICTORS.items():
        probs, preds = predict_fn(df, embed_features, return_probs=True)

        model_result = pd.DataFrame({
            f"{model_name}.attack.{cls}": probs[:, i]
            for i, cls in enumerate(mlb_classes)
        })

        model_result[[f"{model_name}.vote.{cls}" for cls in mlb_classes]] = preds
        model_result[[f"{model_name}.final.{cls}" for cls in mlb_classes]] = preds  # final = vote for now
        results.append(model_result)

        # Save raw probability predictions to CSV
        probas_df = pd.DataFrame(probs, columns=mlb_classes)
        probas_df.insert(0, "url", df["url"])
        probas_df.insert(0, "id", df.index)
        
        os.makedirs(PREDICTIONS_DIR, exist_ok=True)
        probas_path = os.path.join(PREDICTIONS_DIR, f"{model_name}_probas.csv")
        probas_df.to_csv(probas_path, index=False)
        print(f"[✓] Saved {model_name} probabilities to {probas_path}")


    return pd.concat(results, axis=1)

def calc_micro_roc_auc(df, mlb):
    embed_features = load_embed_only("concat", split="test")

    # Store results
    results = {}
    y_true = mlb.transform(df["tactic_ids"])

    plt.figure(figsize=(10,8))

    # Evaluate micro-averaged ROC AUC for each model
    for model_name, predict_fn in MODEL_PREDICTORS.items():
        probs, preds = predict_fn(df, embed_features, return_probs=True)

        # Ensure probs is a 2D array: shape (n_samples, n_labels)
        probs = pd.DataFrame(probs).values
        
        # --- Compute micro-average ROC AUC ---
        auc_micro = roc_auc_score(y_true, probs, average="micro")
        results[model_name] = auc_micro

        # --- Compute ROC curve and AUC (micro average) ---
        fpr, tpr, _ = roc_curve(y_true.ravel(), probs.ravel())
        roc_auc = auc(fpr, tpr)

        # --- Plot ---
        plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.2f})")

    # --- Finalize plot ---
    plt.plot([0, 1], [0, 1], 'k--', label="Random Classifier")
    plt.title("Micro-Averaged ROC Curve (Test Set)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()

    # --- Save plot ---
    metrics_dir = Path(PREDICTIONS_DIR) / "metrics"
    output_path = metrics_dir / "test_auc_micro_curve.png"
    plt.savefig(output_path)
    plt.close()

    # --- Plotting ---
    plt.figure(figsize=(10, 6))
    plt.bar(results.keys(), results.values(), color='mediumseagreen')
    plt.title("Micro ROC AUC per Model on Test Set")
    plt.ylabel("Micro ROC AUC")
    plt.ylim(0.0, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save plot
    output_path = metrics_dir / "test_auc_micro.png"
    plt.savefig(output_path)
    plt.close()

if __name__ == "__main__":
    df = pd.read_parquet("data/processed/test.parquet")
    mlb = joblib.load(os.path.join(CACHE_DIR, "mlb_labels.pkl"))
    calc_micro_roc_auc(df, mlb)

    base_df = pd.DataFrame({
        "id": range(1, len(df) + 1),
        "url": df["url"]
    })

    pred_df = run_model_predictions(df, mlb.classes_)
    output_df = pd.concat([base_df, pred_df], axis=1)

    # Save prediction output
    os.makedirs("data/predictions", exist_ok=True)
    output_df.to_csv("data/predictions/all_model_outputs.csv", index=False)
    print("Saved model outputs to data/predictions/all_model_outputs.csv")

    # Optional evaluation (ensemble-style)
    if "tactic_ids" in df.columns:
        y_true = mlb.transform(df["tactic_ids"])

        for model_name in MODEL_PREDICTORS:
            y_pred = output_df[[f"{model_name}.final.{cls}" for cls in mlb.classes_]].values
            print(f"\n=== {model_name.upper()} Classification Report ===")
            print(classification_report(y_true, y_pred, target_names=mlb.classes_))