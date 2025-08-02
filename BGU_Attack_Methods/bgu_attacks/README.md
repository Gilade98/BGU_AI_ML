```text
mitre_classifier/
├── src/                        # All core source code modules
│   ├── __init__.py
│   ├── config.py               # Central config: paths, model names, constants
│   ├── data/                   # Data handling
│   │   ├── ingest.py           # Load & clean JSON, initial EDA
│   │   ├── split.py            # Group-aware train/val/test splits
│   │   └── features.py         # TF-IDF, embeddings extraction/caching
│   ├── kb/                     # MITRE ATT&CK knowledge base
│   │   └── build_kb.py         # Crawl, embed, and index with FAISS
│   ├── rag_graph/              # Agentic RAG workflow (LangGraph)
│   │   ├── schema.py           # Pydantic data schemas for LLM calls
│   │   ├── nodes.py            # Nodes: Embedder, Retriever, Reasoner, etc.
│   │   └── build_graph.py      # Assemble LangGraph DAG workflow
│   ├── models/                 # All model training scripts
│   │   ├── train_logreg.py     # Logistic Regression (TF-IDF)
│   │   ├── train_lgbm.py       # LightGBM (all features)
│   │   ├── train_mlp.py        # MLP (embeddings)
│   │   ├── train_knn.py        # k-NN (ATT&CK-BERT)
│   │   └── train_meta.py       # XGBoost meta-learner, threshold tuning
│   ├── inference/              # Inference pipeline for new/unseen data
│   │   └── predict.py
│   └── eval/                   # Evaluation metrics & plotting
│       └── metrics.py
├── notebooks/                  # Jupyter/Colab notebooks for EDA, prototyping
├── data/
│   ├── raw/                    # Raw input JSON, untouched
│   ├── caches/ 
│   └── processed/              # Cleaned train/val/test splits (Parquet/NPY)
├── kb/
│   └── mitre_faiss.index       # Precomputed FAISS index of MITRE technique vectors
├── models/                     # Trained model files (.pkl, .pt, thresholds)
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation (this file)
└── Makefile                    # One-click targets: train, evaluate, predict, etc.
```
