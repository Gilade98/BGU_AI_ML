import pandas as pd
import numpy as np
from src.config import PREDICTIONS_DIR, PROCESSED_DIR
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


# Load meta model probs
meta_df = pd.read_csv(Path(PREDICTIONS_DIR) / "xgb_meta_probas.csv")
label_names = [col for col in meta_df.columns if col not in ("id", "url")]

# Load LLM probs and align columns (drop 'reconnaissance')
llm_df = pd.read_csv(Path(PREDICTIONS_DIR) / "llm_output.csv")
llm_col_map = {col.replace("llm_prob_attack.", ""): col
               for col in llm_df.columns if col.startswith("llm_prob_attack.")}
filtered_labels = [l for l in label_names if l in llm_col_map]
meta_probs = meta_df[filtered_labels].values
llm_probs = llm_df[[llm_col_map[l] for l in filtered_labels]].values

llm_processed = llm_df[[llm_col_map[l] for l in filtered_labels]]
llm_processed.columns = filtered_labels  # Remove the 'llm_prob_attack.' prefix for clarity

llm_processed.to_csv("llm_probs_processed.csv", index=False)

ensemble_probs = 0.5 * meta_probs + 0.5 * llm_probs

test_labels_df = pd.read_csv(Path(PROCESSED_DIR) / "test_labels_gt.csv")
# Make sure to match filtered_labels and skip 'reconnaissance'
y_true = test_labels_df[filtered_labels].values


def plot_micro_roc_curves(y_true, meta_probs, llm_probs, ensemble_probs, label='Micro-ROC-AUC'):
    plt.figure(figsize=(8, 8))
    # Meta model
    fpr_meta, tpr_meta, _ = roc_curve(y_true.ravel(), meta_probs.ravel())
    auc_meta = auc(fpr_meta, tpr_meta)
    plt.plot(fpr_meta, tpr_meta, label=f'Meta model (AUC={auc_meta:.3f})', lw=2)

    # LLM
    fpr_llm, tpr_llm, _ = roc_curve(y_true.ravel(), llm_probs.ravel())
    auc_llm = auc(fpr_llm, tpr_llm)
    plt.plot(fpr_llm, tpr_llm, label=f'LLM (AUC={auc_llm:.3f})', lw=2)

    # Ensemble
    fpr_ens, tpr_ens, _ = roc_curve(y_true.ravel(), ensemble_probs.ravel())
    auc_ens = auc(fpr_ens, tpr_ens)
    plt.plot(fpr_ens, tpr_ens, label=f'Ensemble (AUC={auc_ens:.3f})', lw=2)

    # Baseline
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(label)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("micro_roc_comparison.png")
    plt.show()

plot_micro_roc_curves(y_true, meta_probs, llm_probs, ensemble_probs)

