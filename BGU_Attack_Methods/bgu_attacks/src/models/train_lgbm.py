import os
import warnings
from collections import Counter
from pathlib import Path

import joblib
from lightgbm import LGBMClassifier
from sklearn.multioutput import MultiOutputClassifier

from src.config import CACHE_DIR
from src.models.utils import train_model, train_model_with_oof, predict_model

warnings.filterwarnings("ignore", category=UserWarning, message="X does not have valid feature names")

# ---- Config ----
MODEL_PATH = os.path.join(CACHE_DIR, 'lgbm_model.pkl')

# ---- Train Model ----
def train_lgbm():
    model = MultiOutputClassifier(LGBMClassifier(n_estimators=200, max_depth=7, random_state=42))
    
    return train_model(model, Path(MODEL_PATH))

def train_lgbm_kfold():
    model = MultiOutputClassifier(LGBMClassifier(n_estimators=200, max_depth=7, random_state=42))
    
    return train_model_with_oof(model, Path(MODEL_PATH))

# ---- Inference ----
def predict_lgbm(df, embed_features, model=None, enc=None, mlb=None, return_probs=False):
    if model is None:
        model = joblib.load(MODEL_PATH)

    return predict_model(df, embed_features, model, enc, mlb, return_probs)


if __name__ == '__main__':
    train_lgbm()
