import os
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
import joblib

from src.config import CACHE_DIR
from src.models.utils import train_model, train_model_with_oof, predict_model

# ---- Config ----
MODEL_PATH = os.path.join(CACHE_DIR, 'rf_model.pt')


# ---- Train Model ----
def train_rf():
    model = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
    
    return train_model(model, Path(MODEL_PATH))

def train_rf_kfold():
    model = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
    
    return train_model_with_oof(model, Path(MODEL_PATH))

# ---- Inference ----
def predict_rf(df, embed_features, model=None, enc=None, mlb=None, return_probs=False):
    if model is None:
        model = joblib.load(MODEL_PATH)
    
    return predict_model(df, embed_features, model, enc, mlb, return_probs)


if __name__ == '__main__':
    train_rf()