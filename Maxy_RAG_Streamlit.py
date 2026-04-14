# -*- coding: utf-8 -*-
"""Maxy_RAG_Streamlit.py

Maxy Academy RAG System V3 - Universal Version
"""

import sys
import os
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
from src.config import get_api_key
from src.ui import load_custom_css
from src.rag_engine import load_rag_system

# Page Config
st.set_page_config(
    page_title="MinMax - AI Assistant",
    layout="centered",
    initial_sidebar_state="collapsed"
)

def main():
    load_custom_css()
    
    if "language" not in st.session_state:
        st.session_state.language = "Indonesia"

    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("MinMax")
        st.caption("AI Consultant - Maxy Academy")
    with col2:
        st.markdown('<div class="status-badge"><div class="status-dot"></div>Online</div>', unsafe_allow_html=True)
    
    st.divider()
    
    # Check API Key
    api_key = get_api_key()

    if not api_key:
        st.error("API Key tidak ditemukan. Pastikan `OPENAI_API_KEY` telah diset di file `.env`.")
        st.stop()
    
    # Initialize messages
    if "messages" not in st.session_state or len(st.session_state.messages) == 0:
        if st.session_state.language == "English":
            welcome_msg = """**Hello! Welcome to Maxy Academy AI Assistant.**

I am **MinMax**, ready to help you find your dream career program or answer questions about Maxy Academy.

You can ask anything, for example:
- "What is the Fast Track program?"
- "Is there a Data Science bootcamp?"
- "How do I register for an internship?"

Please ask your question today."""
        else:
            welcome_msg = """**Halo! Selamat datang di Maxy Academy AI Assistant.**

Saya **MinMax**, siap membantu Anda menemukan program karir impian atau menjawab pertanyaan seputar Maxy Academy.

Anda bisa bertanya apa saja, misalnya:
- "Apa itu program Fast Track?"
- "Apakah ada bootcamp Data Science?"
- "Bagaimana cara mendaftar magang?"

Silakan ajukan pertanyaan Anda hari ini."""
            
        st.session_state.messages = [
            {"role": "assistant", "content": welcome_msg}
        ]
    
    # Load RAG
    if "rag_loaded" not in st.session_state:
        status_label = "Preparing MinMax..." if st.session_state.language == "English" else "Mempersiapkan MinMax..."
        with st.status(status_label, expanded=True) as status:
            st.write("Loading data & core system..." if st.session_state.language == "English" else "Memuat data & core system...")
            rag = load_rag_system()
            
            if rag is None:
                err_label = "Failed to load AI system" if st.session_state.language == "English" else "Gagal memuat sistem AI"
                status.update(label=err_label, state="error", expanded=True)
                if st.session_state.language == "English":
                    st.error("Please refresh the page or check internet connection.")
                else:
                    st.error("Silakan refresh halaman atau cek koneksi internet.")
                st.stop()
            
            ok_label = "MinMax is ready!" if st.session_state.language == "English" else "MinMax siap!"
            status.update(label=ok_label, state="complete", expanded=False)
            
        st.session_state.rag_system = rag
        st.session_state.rag_loaded = True
        st.rerun()
    
    # Display Messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Quick Actions
    if len(st.session_state.messages) <= 2:
        if st.session_state.language == "English":
            st.markdown("**Popular Questions:**")
            quick_qs = [
                "Fast Track package info",
                "5-week Bootcamp info",
                "8-week Bootcamp info",
                "ISA Scholarship info",
                "Request custom program",
                "Maxy Academy social media accounts"
            ]
        else:
            st.markdown("**Pertanyaan Populer:**")
            quick_qs = [
                "Info paket Fast Track",
                "Info paket Bootcamp 5 minggu",
                "Info paket Bootcamp 8 minggu",
                "Info paket Beasiswa Income Sharing Agreement (ISA)",
                "Request program custom",
                "Apa saja daftar lengkap akun media sosial Maxy Academy?"
            ]
        cols = st.columns(2)
        for i, q in enumerate(quick_qs):
            with cols[i % 2]:
                if st.button(q, key=f"quick_{i}", width='stretch'):
                    st.session_state.messages.append({"role": "user", "content": q})
                    st.rerun()
    
    # Chat Input
    chat_placeholder = "Ketik pertanyaan kamu..." if st.session_state.language == "Indonesia" else "Type your question..."
    if prompt := st.chat_input(chat_placeholder):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.rerun()

    # Generate Response
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        prompt = st.session_state.messages[-1]["content"]
        
        with st.chat_message("assistant"):
            spinner_text = "MinMax sedang memproses..." if st.session_state.language == "Indonesia" else "MinMax is processing..."
            with st.spinner(spinner_text):
                if "rag_system" not in st.session_state or st.session_state.rag_system is None:
                     error_msg = "Sistem AI belum siap. Silakan refresh halaman." if st.session_state.language == "Indonesia" else "AI System not ready yet. Please refresh the page."
                     answer = error_msg
                     response = {"answer": answer}
                else:
                     response = st.session_state.rag_system.query(prompt, st.session_state.messages[:-1], language=st.session_state.language)
                     answer = response["answer"]
            
            st.markdown(answer)
        
        st.session_state.messages.append({"role": "assistant", "content": answer})
    
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/150x50/E67E22/FFFFFF?text=MAXY", width='stretch')
        
        st.markdown("### 🌐 Bahasa / Language")
        selected_lang = st.radio(
            "Language options",
            ["Indonesia", "English"],
            index=0 if st.session_state.get("language", "Indonesia") == "Indonesia" else 1,
            label_visibility="collapsed"
        )
        
        if selected_lang != st.session_state.get("language", "Indonesia"):
            st.session_state.language = selected_lang
            # Clear chat when changing language to avoid mixing
            st.session_state.messages = []
            st.rerun()
            
        st.divider()
        
        if st.session_state.language == "English":
            st.markdown("### Example Questions")
            examples = [
                "Fast Track package info",
                "5-week Bootcamp info",
                "8-week Bootcamp info",
                "ISA Scholarship info",
                "Request custom program",
                "Maxy Academy social media accounts"
            ]
        else:
            st.markdown("### Contoh Pertanyaan")
            examples = [
                "Info paket Fast Track",
                "Info paket Bootcamp 5 minggu",
                "Info paket Bootcamp 8 minggu",
                "Info paket Beasiswa Income Sharing Agreement (ISA)",
                "Request program custom",
                "Apa saja daftar lengkap akun media sosial Maxy Academy?"
            ]
        
        for ex in examples:
            if st.button(ex, key=f"sidebar_{ex}"):
                st.session_state.messages.append({"role": "user", "content": ex})
                st.rerun()
        
        st.divider()
        
        clear_btn_text = "Hapus Obrolan" if st.session_state.language == "Indonesia" else "Clear Chat"
        if st.button(clear_btn_text, width='stretch'):
            st.session_state.messages = [] # Reset to welcome

            st.rerun()
        
        st.divider()
        
        st.markdown("**Status:**")
        api_key_val = get_api_key()
        
        if api_key_val:
            if st.session_state.language == "English":
                st.caption(f"Active Connection: OpenAI")
            else:
                st.caption(f"Koneksi Aktif: OpenAI")
        else:
            if st.session_state.language == "English":
                st.caption("Status: API Key not set")
            else:
                st.caption("Status: API Key belum diset")

        st.divider()
        st.caption("Powered by **OpenAI**")
        st.caption("© 2026 Maxy Academy")

if __name__ == "__main__":
    main()
