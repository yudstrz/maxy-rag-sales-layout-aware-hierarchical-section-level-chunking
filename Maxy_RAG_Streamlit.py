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
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": """**Halo! Selamat datang di Maxy Academy AI Assistant.**

Saya **MinMax**, siap membantu Anda menemukan program karir impian atau menjawab pertanyaan seputar Maxy Academy.

Anda bisa bertanya apa saja, misalnya:
- "Apa itu program Fast Track?"
- "Apakah ada bootcamp Data Science?"
- "Bagaimana cara mendaftar magang?"

Silakan ajukan pertanyaan Anda hari ini."""}
        ]
    
    # Load RAG
    if "rag_loaded" not in st.session_state:
        with st.status("Mempersiapkan MinMax...", expanded=True) as status:
            st.write("Memuat data & core system...")
            rag = load_rag_system()
            
            if rag is None:
                status.update(label="Gagal memuat sistem AI", state="error", expanded=True)
                st.error("Silakan refresh halaman atau cek koneksi internet.")
                st.stop()
            
            status.update(label="MinMax siap!", state="complete", expanded=False)
            
        st.session_state.rag_system = rag
        st.session_state.rag_loaded = True
        st.rerun()
    
    # Display Messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Quick Actions
    if len(st.session_state.messages) <= 2:
        st.markdown("**Pertanyaan Populer:**")
        cols = st.columns(2)
        quick_qs = [
            "Info paket Fast Track",
            "Info paket Bootcamp 5 minggu",
            "Info paket Bootcamp 8 minggu",
            "Info paket Beasiswa Income Sharing Agreement (ISA)",
            "Request program custom",
            "Apa saja daftar lengkap akun media sosial Maxy Academy?"
        ]
        for i, q in enumerate(quick_qs):
            with cols[i % 2]:
                if st.button(q, key=f"quick_{i}", width='stretch'):
                    st.session_state.messages.append({"role": "user", "content": q})
                    st.rerun()
    
    # Chat Input
    if prompt := st.chat_input("Ketik pertanyaan kamu..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.rerun()

    # Generate Response
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        prompt = st.session_state.messages[-1]["content"]
        
        with st.chat_message("assistant"):
            with st.spinner("MinMax sedang memproses..."):
                if "rag_system" not in st.session_state or st.session_state.rag_system is None:
                     answer = "Sistem AI belum siap. Silakan refresh halaman."
                     response = {"answer": answer}
                else:
                     response = st.session_state.rag_system.query(prompt, st.session_state.messages[:-1])
                     answer = response["answer"]
            
            st.markdown(answer)
        
        st.session_state.messages.append({"role": "assistant", "content": answer})
    
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/150x50/E67E22/FFFFFF?text=MAXY", width='stretch')
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
        
        if st.button("Hapus Obrolan", width='stretch'):
            st.session_state.messages = [st.session_state.messages[0]] # Reset to welcome
            st.rerun()
        
        st.divider()
        
        st.markdown("**Status:**")
        api_key_val = get_api_key()
        
        if api_key_val:
            st.caption(f"Koneksi Aktif: OpenAI")
        else:
            st.caption("Status: API Key belum diset")

        st.divider()
        st.caption("Powered by **OpenAI**")
        st.caption("© 2026 Maxy Academy")

if __name__ == "__main__":
    main()
