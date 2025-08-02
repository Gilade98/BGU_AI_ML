import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import joblib
import torch
from transformers import AutoTokenizer, AutoModel
from src.config import DATA_DIR,CACHE_DIR

# ---- 1. Utility: Combine text columns ----
def get_texts(df):
    # Use .get to handle missing columns safely
    text = df['markdown'].fillna('')
    if 'title' in df.columns:
        text += ' ' + df['title'].fillna('')
    if 'description' in df.columns:
        text += ' ' + df['description'].fillna('')
    return text

# ---- 2. Fit and transform TF-IDF + SVD ----
def fit_tfidf_svd(
    df,
    tfidf_max_features=5000,
    tfidf_min_df=2,
    svd_components=128,
    tfidf_path=None,
    svd_path=None,
    matrix_path=None
):
    texts = get_texts(df)
    tfidf = TfidfVectorizer(max_features=tfidf_max_features, min_df=tfidf_min_df, stop_words='english')
    X_tfidf = tfidf.fit_transform(texts)
    svd = TruncatedSVD(n_components=svd_components, random_state=42)
    X_svd = svd.fit_transform(X_tfidf)

    # Save caches
    if tfidf_path:
        joblib.dump(tfidf, tfidf_path)
    if svd_path:
        joblib.dump(svd, svd_path)
    if matrix_path:
        np.save(matrix_path, X_svd)
    return tfidf, svd, X_svd

# ---- 3. Transform new data using cached models ----
def transform_tfidf_svd(df, tfidf_path, svd_path, matrix_path=None):
    tfidf = joblib.load(tfidf_path)
    svd = joblib.load(svd_path)
    texts = get_texts(df)
    X_tfidf = tfidf.transform(texts)
    X_svd = svd.transform(X_tfidf)
    if matrix_path:
        np.save(matrix_path, X_svd)
    return X_svd

# ---- 4. BERT Embeddings (Optional, uncomment if needed) ----
def get_bert_embeddings(
    df,
    text_col='markdown',
    model_name='sentence-transformers/all-MiniLM-L6-v2',
    batch_size=16,
    device=None,
    cache_path=None,
    max_length=256
):

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    if device is not None:
        model.to(device)
    model.eval()

    texts = df[text_col].fillna("").tolist()
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
            if device is not None:
                inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # CLS pooling
            embeddings.append(emb)
    embeddings = np.concatenate(embeddings, axis=0)
    if cache_path:
        np.save(cache_path, embeddings)
    return embeddings

if __name__ == "__main__":

    # ----- TF-IDF + SVD -----
    print("Loading train/val/test splits...")
    df_train = pd.read_parquet(os.path.join(DATA_DIR, "train.parquet"))
    df_val = pd.read_parquet(os.path.join(DATA_DIR, "val.parquet"))
    df_test = pd.read_parquet(os.path.join(DATA_DIR, "test.parquet"))

    print("Fitting TF-IDF + SVD on train...")
    tfidf_path = os.path.join(CACHE_DIR, "tfidf_vectorizer.joblib")
    svd_path = os.path.join(CACHE_DIR, "tfidf_svd.joblib")
    train_matrix_path = os.path.join(CACHE_DIR, "train_tfidf_svd.npy")
    tfidf, svd, X_train = fit_tfidf_svd(
        df_train,
        tfidf_max_features=5000,
        tfidf_min_df=2,
        svd_components=128,
        tfidf_path=tfidf_path,
        svd_path=svd_path,
        matrix_path=train_matrix_path
    )

    print("Transforming val and test splits...")
    X_val = transform_tfidf_svd(df_val, tfidf_path, svd_path, matrix_path=os.path.join(CACHE_DIR, "val_tfidf_svd.npy"))
    X_test = transform_tfidf_svd(df_test, tfidf_path, svd_path, matrix_path=os.path.join(CACHE_DIR, "test_tfidf_svd.npy"))

    print(f"Shapes: train {X_train.shape}, val {X_val.shape}, test {X_test.shape}")

    # ---- SecBERT ----
    print("Generating SecBERT embeddings...")
    secbert_model = "jackaduma/SecBERT"
    train_secbert = get_bert_embeddings(
        df_train,
        text_col='markdown',
        model_name=secbert_model,
        cache_path=os.path.join(CACHE_DIR, "train_secbert.npy")
    )
    val_secbert = get_bert_embeddings(
        df_val,
        text_col='markdown',
        model_name=secbert_model,
        cache_path=os.path.join(CACHE_DIR, "val_secbert.npy")
    )
    test_secbert = get_bert_embeddings(
        df_test,
        text_col='markdown',
        model_name=secbert_model,
        cache_path=os.path.join(CACHE_DIR, "test_secbert.npy")
    )

    print(f"Shapes: train {train_secbert.shape}, val {val_secbert.shape}, test {test_secbert.shape}")

    # ----CTI-BERT ----
    print("Generating CTI-BERT embeddings...")
    attackbert_model = "ibm-research/CTI-BERT"
    train_attackbert = get_bert_embeddings(
        df_train,
        text_col='markdown',
        model_name=attackbert_model,
        cache_path=os.path.join(CACHE_DIR, "train_attackbert.npy")
    )
    val_attackbert = get_bert_embeddings(
        df_val,
        text_col='markdown',
        model_name=attackbert_model,
        cache_path=os.path.join(CACHE_DIR, "val_attackbert.npy")
    )
    test_attackbert = get_bert_embeddings(
        df_test,
        text_col='markdown',
        model_name=attackbert_model,
        cache_path=os.path.join(CACHE_DIR, "test_attackbert.npy")
    )
    print(f"Shapes: train {train_attackbert.shape}, val {val_attackbert.shape}, test {test_attackbert.shape}")

