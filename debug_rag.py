
import os
import sys
from Maxy_RAG_Streamlit import load_rag_system, HybridRAGConfig

# Mock streamlit cache_resource to just return the function
import streamlit as st
def mock_cache(func):
    return func
st.cache_resource = mock_cache

def main():
    print("Starting debug run...")
    try:
        # Create a dummy progress callback
        def progress_callback(pct, msg):
            print(f"[{pct}%] {msg.encode('ascii', 'ignore').decode('ascii')}")

        print("Initializing RAG Config...")
        config = HybridRAGConfig()
        print(f"Base Path: {config.BASE_PATH}")
        print(f"Bootcamp Path: {config.BOOTCAMP_DATASET}")
        
        print("Calling load_rag_system...")
        rag = load_rag_system(_progress_callback=progress_callback)
        
        if rag:
            print("RAG System loaded successfully!")
        else:
            print("RAG System returned None (Sections might be empty)")
            
    except Exception as e:
        print(f"CAUGHT EXCEPTION: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
