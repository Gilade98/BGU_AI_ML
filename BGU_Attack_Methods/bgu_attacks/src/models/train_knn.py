import os
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multioutput import MultiOutputClassifier
import joblib

from src.config import CACHE_DIR
from src.models.utils import train_model, train_model_with_oof, predict_model

# ---- Config ----
MODEL_PATH = os.path.join(CACHE_DIR, 'knn_model.pkl')

# ---- Train Model ----
def train_knn():
    model = MultiOutputClassifier(KNeighborsClassifier(n_neighbors=5, metric='cosine'))
    
    return train_model(model, Path(MODEL_PATH))

def train_knn_kfold():
    model = MultiOutputClassifier(KNeighborsClassifier(n_neighbors=5, metric='cosine'))
    
    return train_model_with_oof(model, Path(MODEL_PATH))

# ---- Inference ----
def predict_knn(df, embed_features, model=None, enc=None, mlb=None, return_probs=False):
    if model is None:
        model = joblib.load(MODEL_PATH)

    return predict_model(df, embed_features, model, enc, mlb, return_probs)


if __name__ == '__main__':
    train_knn()