import os

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from src.config import REPO_ROOT,TACTIC_WHITELIST


def all_labels_present(df_split, label_col, all_labels):
    split_labels = set(label for labels in df_split[label_col] for label in labels)
    return set(all_labels).issubset(split_labels)

def group_split(
    df,
    label_col='tactic_ids',
    group_col='url',
    val_size=0.15,
    test_size=0.15,
    seed=42,
    max_tries=50
):
    all_labels = set(label for labels in df[label_col] for label in labels)

    # 1st split: train+val vs test
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_val_idx, test_idx = next(splitter.split(df, groups=df[group_col]))
    df_train_val = df.iloc[train_val_idx].copy()
    df_test = df.iloc[test_idx].copy()

    # 2nd split: train vs val (size is val_size relative to total)
    rel_val_size = val_size / (1 - test_size)
    splitter = GroupShuffleSplit(n_splits=1, test_size=rel_val_size, random_state=seed + 1)
    train_idx, val_idx = next(splitter.split(df_train_val, groups=df_train_val[group_col]))
    df_train = df_train_val.iloc[train_idx].copy()
    df_val = df_train_val.iloc[val_idx].copy()

    return df_train, df_val, df_test

    # for attempt in range(max_tries):
    #     splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed + attempt)
    #     train_val_idx, test_idx = next(splitter.split(df, groups=df[group_col]))
    #     df_train_val = df.iloc[train_val_idx].copy()
    #     df_test = df.iloc[test_idx].copy()
    #
    #     rel_val_size = val_size / (1 - test_size)
    #     splitter = GroupShuffleSplit(n_splits=1, test_size=rel_val_size, random_state=seed + 100 + attempt)
    #     train_idx, val_idx = next(splitter.split(df_train_val, groups=df_train_val[group_col]))
    #     df_train = df_train_val.iloc[train_idx].copy()
    #     df_val = df_train_val.iloc[val_idx].copy()
    #
    #     for df in [df_train, df_val, df_test]:
    #         print(set(label for labels in df_train[label_col] for label in labels))
    #     if all(map(lambda d: all_labels_present(d, label_col, all_labels), [df_train, df_val, df_test])):
    #         print(f"Successful split on attempt {attempt + 1}")
    #         return df_train, df_val, df_test
    #
    # raise ValueError("Failed to create a split where all splits contain all labels after max tries.")


if __name__ == '__main__':
    # Set up paths
    PARQUET_PATH = os.path.join(REPO_ROOT, "data", "processed", "clean_data.parquet")
    OUTPUT_DIR = os.path.join(REPO_ROOT, "data", "processed")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load data
    df = pd.read_parquet(PARQUET_PATH)

    # Split
    df_train, df_val, df_test = group_split(
        df,
        group_col="url",
        val_size=0.15,
        test_size=0.15,
        seed=42
    )

    # Save
    df_train.to_parquet(os.path.join(OUTPUT_DIR, "train.parquet"), index=False)
    df_val.to_parquet(os.path.join(OUTPUT_DIR, "val.parquet"), index=False)
    df_test.to_parquet(os.path.join(OUTPUT_DIR, "test.parquet"), index=False)
    print(f"Saved splits: train={len(df_train)}, val={len(df_val)}, test={len(df_test)}")