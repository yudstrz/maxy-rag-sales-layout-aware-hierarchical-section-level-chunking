import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

def get_groq_api_key():
    """Get Groq API key from: 1) session_state (UI input), 2) st.secrets, 3) .env"""
    # 1. Check session_state (UI input)
    if "groq_api_key" in st.session_state and st.session_state.groq_api_key:
        return st.session_state.groq_api_key
    # 2. Check Streamlit Secrets (Cloud)
    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except:
        pass
    # 3. Check environment variable (.env)
    return os.getenv("GROQ_API_KEY", "")

class GroqConfig:
    BASE_URL = "https://api.groq.com/openai/v1"
    MODEL_LIST = [
        "llama-3.3-70b-versatile",
        "mixtral-8x7b-32768",
        "gemma2-9b-it",
    ]
    
    @classmethod
    def get_api_key(cls):
        return get_groq_api_key()



class HybridRAGConfig:
    # Dynamic Project Root
    # c:\Users\...\src\config.py -> c:\Users\...\
    BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    BOOTCAMP_DATASET = os.path.join(BASE_PATH, "MAXY_Bootcamp_Dataset.jsonl")
    CURRICULUM_DATASET = os.path.join(BASE_PATH, "MAXY_Curriculum_Syllabus.jsonl")
    COMPANY_INFO = os.path.join(BASE_PATH, "MAXY_Company_Info.jsonl")
    
    FAISS_INDEX_PATH = os.path.join(BASE_PATH, "faiss_index")
    BM25_INDEX_PATH = os.path.join(BASE_PATH, "bm25_index.pkl")
    PREPROCESSED_DATA_PATH = os.path.join(BASE_PATH, "data_cache.pkl")

    EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" # ~470MB (Best Quality)
    # EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2" # ~80MB (Best Speed/Size)
    # Use smaller reranker for Streamlit Cloud (bge-reranker-v2-m3 is too large ~560MB)
    RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # ~80MB, much faster

    TOP_K_ABSTRACT_BM25 = 30
    TOP_K_DENSE = 15
    TOP_K_RERANK = 10
    TOP_K_FINAL = 5
