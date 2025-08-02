import os
import pandas as pd
import numpy as np
import faiss
from transformers import AutoTokenizer, AutoModel
import torch
import logging

logging.basicConfig(level=logging.INFO)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)


# Load train data and embeddings
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(REPO_ROOT, "data", "caches")
DATA_DIR = os.path.join(REPO_ROOT, "data", "processed")
PROCESSED_DIR = os.path.join(REPO_ROOT, "data", "processed")
PREDICTIONS_DIR = os.path.join(REPO_ROOT, "data", "predictions")
TRAIN_PARQUET = os.path.join(REPO_ROOT, "data/processed/train.parquet")
TEST_PARQUET = os.path.join(REPO_ROOT, "data/processed/test.parquet")
TRAIN_JSON = os.path.join(REPO_ROOT, "data/raw/splits/train.json")
TEST_JSON = os.path.join(REPO_ROOT, "data/raw/splits/test.json")
VAL_PARQUET = os.path.join(REPO_ROOT, "data/processed/val.parquet")
VAL_JSON = os.path.join(REPO_ROOT, "data/raw/splits/val.json")
RAW_PATH = os.path.join(REPO_ROOT, "data", "raw", "enrichment_dataset.json")
OUT_PATH = os.path.join(REPO_ROOT, "data", "processed", "clean_data.parquet")
REASONINGS_PATH = "all_reasonings.json"

LLM_OUTPUT_PATH = os.path.join(CACHE_DIR, 'llm_output.csv')
BATCH_SIZE = 1

NUM_DIRECT_WORKERS = 5      # set to however many Send() you do
NUM_RETRIEVAL_WORKERS = 5   # set to your retrieval Send() count

train_emb = np.load(os.path.join(REPO_ROOT,"data/caches/train_secbert.npy"))  # shape: (num_train, emb_dim)
train_df = pd.read_parquet(TRAIN_PARQUET)

FAISS_INDEX_PATH = os.path.join(REPO_ROOT, "src/ai_agent/secbert_train.faiss")
SEC_BERT_PATH = "jackaduma/SecBERT"
faiss_index = faiss.read_index(FAISS_INDEX_PATH)

tokenizer = AutoTokenizer.from_pretrained(SEC_BERT_PATH)
model = AutoModel.from_pretrained(SEC_BERT_PATH)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

RETRIEVAL_K = 3

# TACTIC_WHITELIST = [
#     "attack.initial_access",
#     "attack.execution",
#     "attack.persistence",
#     "attack.privilege_escalation",
#     "attack.defense_evasion",
#     "attack.credential_access",
#     "attack.discovery",
#     "attack.lateral_movement",
#     "attack.collection",
#     "attack.command_and_control",
#     "attack.exfiltration",
#     "attack.impact",
#     "attack.resource_development",
# ]

TACTIC_WHITELIST = [
    "initial_access", 
    "execution",
    "persistence", 
    "privilege_escalation", 
    "defense_evasion", 
    "credential_access", 
    "discovery",
    "lateral_movement",
    "collection",
    "command_and_control",
    "exfiltration",
    "impact",
    "resource_development",
    # "reconnaissance" # does not exist in the data
]

# Metadata encoding
CAT_FEATURES = ['level', 'status', 'log_cat', 'log_product']
MLB_PATH = os.path.join(CACHE_DIR, 'mlb_labels.pkl')
ENC_PATH = os.path.join(CACHE_DIR, 'meta_encoder.pkl')
