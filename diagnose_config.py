import streamlit as st
import os
import sys

# Mock streamlit secrets if running as script
if not hasattr(st, "secrets"):
    try:
        import toml
        secrets_path = ".streamlit/secrets.toml"
        if os.path.exists(secrets_path):
             st.secrets = toml.load(secrets_path)
             print("Loaded secrets from file.")
        else:
             st.secrets = {}
             print("Secrets file not found.")
    except ImportError:
        print("toml module not found, cannot mock secrets.")
        st.secrets = {}

from src.config import get_groq_api_key, GROQ_API_KEY, ZAI_API_KEY

print(f"GROQ_API_KEY from secrets: {'Found' if 'GROQ_API_KEY' in st.secrets else 'Missing'}")
key = get_groq_api_key()
print(f"Computed Key: {key[:5]}...{key[-5:] if key else ''}")
print(f"Secret Content Match: {key == st.secrets.get('GROQ_API_KEY', 'x')}")
