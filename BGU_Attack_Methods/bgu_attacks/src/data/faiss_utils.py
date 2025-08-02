import faiss
from src.config import *

assert len(train_df) == train_emb.shape[0], "Mismatch between train data and embeddings"

# Build and save FAISS index
index = faiss.IndexFlatL2(train_emb.shape[1])
index.add(train_emb)
faiss.write_index(index, "secbert_train.faiss")
