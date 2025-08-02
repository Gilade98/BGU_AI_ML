import pandas as pd
import numpy as np
from pathlib import Path
import re
from xgboost import XGBClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import f1_score, hamming_loss, roc_auc_score
from src.config import PREDICTIONS_DIR, PROCESSED_DIR
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold



def prepare_meta_data(pred_dir, gt_paths, labels_to_drop=['reconnaissance']):
    """
    Loads and horizontally stacks base model predictions and loads/aligns ground-truth labels.
    Returns X_meta, y_meta, column_names, label_names.
    """
    dfs = []
    column_names = []
    model_names = []

    pattern = re.compile(r'oof_predictions_(.+?)_model\.csv')
    csv_files = list(Path(pred_dir).glob('*.csv'))

    for f in csv_files:
        # Extract model name from filename
        match = pattern.match(f.name)
        if match:
            model_name = match.group(1)
        else:
            model_name = f.stem  # fallback

        df = pd.read_csv(f)
        # Drop 'url' and 'id'
        df_no_ids = df.drop(columns=['url', 'id'])
        # Prefix columns
        df_no_ids = df_no_ids.add_prefix(f"{model_name}_")
        dfs.append(df_no_ids)
        column_names.extend(df_no_ids.columns)
        model_names.append(model_name)

    # Stack all model predictions horizontally
    X_meta = np.hstack([df.values for df in dfs])

    X_meta_df = pd.DataFrame(X_meta, columns=column_names)

    # Load the label CSVs
    # Load and align ground truth
    gt_dfs = [pd.read_csv(Path(p)) for p in gt_paths]

    # Get common label columns (excluding id/url and any to drop)
    labels_cols = [c for c in gt_dfs[0].columns if c not in ('id', 'url') and c not in labels_to_drop]
    label_names = labels_cols.copy()
    gt_arrays = [df[labels_cols].values for df in gt_dfs]
    y_meta = np.concatenate(gt_arrays, axis=0)
    return X_meta, y_meta, column_names, label_names

def prepare_inference_meta_data(pred_dir, labels_to_drop=['reconnaissance'],ignore_files=None): # in our test set impact column is all 0's.
    if ignore_files is None:
        ignore_files = []
    pattern = re.compile(r'(.+?)_probas\.csv')
    csv_files = [
        f for f in Path(pred_dir).glob('*.csv')
        if f.name not in ignore_files
    ]
    dfs = []
    column_names = []
    ids, urls = None, None

    for f in csv_files:
        match = pattern.match(f.name)
        model_name = match.group(1) if match else f.stem
        df = pd.read_csv(f)
        if ids is None:
            ids = df['id'].values
            urls = df['url'].values
        # Drop id, url, and any labels to drop
        cols_to_drop = ['id', 'url'] + labels_to_drop
        keep_cols = [c for c in df.columns if c not in cols_to_drop]
        df_no_ids = df[keep_cols].add_prefix(f"{model_name}_")
        dfs.append(df_no_ids)
        column_names.extend(df_no_ids.columns)
    X_meta = np.hstack([df.values for df in dfs])
    return X_meta, column_names, ids, urls


def train_meta_model(X_meta, y_meta ,n_iter=30, random_state=42):
    param_distributions = {
        'estimator__n_estimators': [100, 200, 300],
        'estimator__max_depth': [3, 5, 7],
        'estimator__learning_rate': [0.01, 0.05, 0.1],
        'estimator__subsample': [0.8, 1.0],
        'estimator__colsample_bytree': [0.7, 1.0],
    }
    xgb = XGBClassifier(eval_metric='logloss', n_jobs=-1, verbosity=1)
    meta_model = MultiOutputClassifier(xgb)
    mskf = MultilabelStratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        meta_model,
        param_distributions=param_distributions,
        n_iter=n_iter,  # Number of random parameter sets to try
        scoring='roc_auc',
        cv=mskf,
        n_jobs=-1,
        verbose=2,
        random_state=random_state
    )
    search.fit(X_meta, y_meta)
    print("Best parameters:", search.best_params_)
    return search.best_estimator_

def predict_and_eval(meta_model, X, y_true, label_names=None, threshold=0.5, prefix="", labels_to_skip=['impact']):
    """
    Predicts and evaluates multilabel metrics.
    """
    y_pred_proba = np.column_stack([
        est.predict_proba(X)[:, 1] for est in meta_model.estimators_
    ])
    y_pred = (y_pred_proba >= threshold).astype(int)

    if labels_to_skip is None:
        labels_to_skip = []
    valid_label_indices = [
        i for i, label in enumerate(label_names)
        if label not in labels_to_skip and len(np.unique(y_true[:, i])) > 1
    ]

    results = {
        f"{prefix}f1_micro": f1_score(y_true, y_pred, average='micro'),
        f"{prefix}f1_macro": f1_score(y_true[:, valid_label_indices], y_pred[:, valid_label_indices], average='macro'),
        f"{prefix}hamming_loss": hamming_loss(y_true, y_pred),
        f"{prefix}auc_macro": roc_auc_score(y_true[:, valid_label_indices], y_pred_proba[:, valid_label_indices], average='macro'),
        f"{prefix}auc_micro": roc_auc_score(y_true, y_pred_proba, average='micro'),
    }

    print(f"\n{prefix}Results:")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")
    return results, y_pred_proba

def plot(y_test_meta,y_test_pred_proba):
    fpr_micro, tpr_micro, _ = roc_curve(
        y_test_meta.ravel(), y_test_pred_proba.ravel()
    )
    roc_auc_micro = auc(fpr_micro, tpr_micro)

    plt.figure(figsize=(7, 7))
    plt.plot(fpr_micro, tpr_micro, color='blue',
             label=f'XGBoost meta-model (AUC = {roc_auc_micro:.3f})')
    plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--', label='Random Clasifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Micro-Average ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("micro_average_roc_curve.png")
    plt.show()




if __name__ == "__main__":
    oof_pred_dir = Path(PREDICTIONS_DIR) / 'oof_predictions'
    train_labels_path = Path(PROCESSED_DIR) / 'train_labels_gt.csv'
    val_labels_path = Path(PROCESSED_DIR) / 'val_labels_gt.csv'
    X_meta, y_meta, column_names, label_names = prepare_meta_data(
        oof_pred_dir, [train_labels_path, val_labels_path]
    )
    print("X_meta shape:", X_meta.shape)
    print("y_meta shape:", y_meta.shape)

    meta_model = train_meta_model(X_meta, y_meta)

    # Prepare test meta data (using base model predictions on test set)
    test_pred_dir = Path(PREDICTIONS_DIR)
    X_test_meta, test_column_names, test_ids, test_urls = prepare_inference_meta_data(test_pred_dir,
                                                                                      ignore_files=[
                                                                                          'all_model_outputs.csv',
                                                                                          'llm_output.csv']
                                                                                      )

    test_labels_gt = pd.read_csv(Path(PROCESSED_DIR) / 'test_labels_gt.csv')
    labels_cols = [c for c in test_labels_gt.columns if c not in ('id', 'url', 'reconnaissance')]
    y_test_meta = test_labels_gt[labels_cols].values

    print("X_test_meta shape:", X_test_meta.shape)
    print("y_test_meta shape:", y_test_meta.shape)

    constant_labels = [i for i in range(y_test_meta.shape[1]) if len(np.unique(y_test_meta[:, i])) == 1]
    for idx in constant_labels:
        print(f"Constant label at index {idx}:", label_names[idx])

    # Predict/evaluate on test
    results, y_test_pred_proba = predict_and_eval(meta_model, X_test_meta, y_test_meta, label_names=labels_cols, prefix="test_")

    plot(y_test_meta, y_test_pred_proba)

    # Create DataFrame with predicted probabilities and label names as columns
    df_pred_proba = pd.DataFrame(y_test_pred_proba, columns=label_names)

    # (Optional) Add id/url columns if you have them
    if test_ids is not None:
        df_pred_proba.insert(0, "id", test_ids)
    if test_urls is not None:
        df_pred_proba.insert(1, "url", test_urls)

    # Save to CSV
    df_pred_proba.to_csv("xgb_meta_probas.csv", index=False)
    print("Saved test prediction probabilities to y_test_pred_proba.csv")

