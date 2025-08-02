import os
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
import joblib

from src.config import CACHE_DIR
from src.models.utils import train_model, train_model_with_oof, predict_model

# ---- Config ----
MODEL_PATH = os.path.join(CACHE_DIR, 'logreg_model.pt')


# ---- Train Model ----
def train_logistic():
    model = MultiOutputClassifier(LogisticRegression(max_iter=1000))
    
    return train_model(model, Path(MODEL_PATH))

def train_logistic_kfold():
    model = MultiOutputClassifier(LogisticRegression(max_iter=1000))
    
    return train_model_with_oof(model, Path(MODEL_PATH))

# ---- Inference ----
def predict_logistic(df, embed_features, model=None, enc=None, mlb=None, return_probs=False):
    if model is None:
        model = joblib.load(MODEL_PATH)
    
    return predict_model(df, embed_features, model, enc, mlb, return_probs)


if __name__ == '__main__':
    train_logistic()