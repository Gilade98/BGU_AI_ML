import os

import joblib
import pandas as pd
pd.set_option('display.max_columns', None)

from src.config import TRAIN_PARQUET, TEST_PARQUET, CACHE_DIR, PROCESSED_DIR, VAL_PARQUET


def create_gt(df, mlb):
    y = mlb.transform(df["tactic_ids"])  # shape: (samples, n_tactics)
    tactic_cols = list(mlb.classes_)
    y_df = pd.DataFrame(y, columns=tactic_cols)
    if "reconnaissance" not in y_df.columns:
        y_df["reconnaissance"] = 0  # Adds an all-zero column
    if tactic_cols[-1] != "reconnaissance":
        cols = [c for c in tactic_cols if c != "reconnaissance"] + ["reconnaissance"]
        y_df = y_df[cols]
    result_df = pd.concat([df[['id']].reset_index(drop=True), y_df], axis=1)

    return result_df

if __name__ == '__main__':
    split_files = {
        "train": TRAIN_PARQUET,
        "val": VAL_PARQUET,
        "test": TEST_PARQUET
    }

    mlb_path = os.path.join(CACHE_DIR, "mlb_labels.pkl")
    mlb = joblib.load(mlb_path)

    for split, parquet_path in split_files.items():
        df = pd.read_parquet(parquet_path)
        gt = create_gt(df, mlb)
        save_path = os.path.join(PROCESSED_DIR, f"{split}_labels_gt.csv")
        gt.to_csv(save_path, index=False)
        print(f"Saved {split} labels to {save_path}")

