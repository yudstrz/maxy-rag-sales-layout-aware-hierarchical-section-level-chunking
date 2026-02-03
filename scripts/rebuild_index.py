import sys
import os
import time

# Add parent dir to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import HybridRAGConfig
from src.rag_engine import LayoutAwareChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

import pickle
from langchain_community.retrievers import BM25Retriever

def build_index():
    print("[INFO] Starting Index Build Process...")
    start_time = time.time()
    
    config = HybridRAGConfig()
    print(f"[INFO] Config loaded. Target Path: {config.FAISS_INDEX_PATH}")
    
    # 1. Load Data
    print("[INFO] Loading data from JSONL files...")
    chunker = LayoutAwareChunker(config)
    sections = chunker.process_all()
    print(f"[INFO] Loaded {len(sections)} sections.")
    
    if not sections:
        print("[ERROR] No data found! Aborting.")
        return

    # SAVE DATA CACHE
    print(f"[INFO] Saving Data Cache to {config.PREPROCESSED_DATA_PATH}...")
    with open(config.PREPROCESSED_DATA_PATH, "wb") as f:
        pickle.dump(sections, f)

    # 2. Load Embedding Model
    print("[INFO] Loading Embedding Model (this may take a while)...")
    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
    print("[INFO] Embedding model loaded.")
    
    # 3. Build BM25
    print("[INFO] Building BM25 Index...")
    abstract_docs = [s.to_langchain_doc(use_abstract=True) for s in sections]
    abstract_bm25 = BM25Retriever.from_documents(abstract_docs)
    abstract_bm25.k = config.TOP_K_ABSTRACT_BM25
    
    # SAVE BM25 CACHE
    print(f"[INFO] Saving BM25 Index to {config.BM25_INDEX_PATH}...")
    with open(config.BM25_INDEX_PATH, "wb") as f:
        pickle.dump(abstract_bm25, f)

    # 4. Build FAISS
    print("[INFO] Building FAISS Index (generating embeddings)...")
    full_docs = [s.to_langchain_doc(use_abstract=False) for s in sections]
    vectorstore = FAISS.from_documents(full_docs, embeddings)
    print("[INFO] FAISS Index built.")
    
    # 5. Save FAISS
    print(f"[INFO] Saving FAISS to {config.FAISS_INDEX_PATH}...")
    vectorstore.save_local(config.FAISS_INDEX_PATH)
    
    elapsed = time.time() - start_time
    print(f"[SUCCESS] Done in {elapsed:.2f} seconds!")

if __name__ == "__main__":
    build_index()
