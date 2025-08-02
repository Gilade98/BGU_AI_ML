import os
from pathlib import Path

from catboost import CatBoostClassifier
from src.config import CACHE_DIR
from src.models.utils import train_model, train_model_with_oof, predict_model

# ---- Config ----
MODEL_PATH = os.path.join(CACHE_DIR, 'catboost_model.pt')

# ---- Train CatBoost ----
def train_catboost():
    model = CatBoostClassifier(verbose=100,  # Print progress every 100 iterations
        loss_function='MultiLogloss',
        iterations=300,
        early_stopping_rounds=20
    )

    return train_model(model, Path(MODEL_PATH))

def train_catboost_kfold():
    model = CatBoostClassifier(verbose=100,  # Print progress every 100 iterations
        loss_function='MultiLogloss',
        iterations=300,
        early_stopping_rounds=20
    )

    return train_model_with_oof(model, Path(MODEL_PATH))

# ---- Inference CatBoost ----
def predict_catboost(df, embed_features, model=None, enc=None, mlb=None, return_probs=False):
    if model is None:
        model = CatBoostClassifier()
        model.load_model(MODEL_PATH)

    return predict_model(df, embed_features, model, enc, mlb, return_probs)

if __name__ == '__main__':
    train_catboost()
