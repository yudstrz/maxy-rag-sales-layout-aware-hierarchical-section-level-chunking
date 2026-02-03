# -*- coding: utf-8 -*-
"""Maxy_RAG_Streamlit.py

Maxy Academy RAG System V3 - Cleaned & Modularized
"""

import sys
import os
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
from src.config import get_groq_api_key, ZaiConfig
from src.ui import load_custom_css
from src.rag_engine import load_rag_system

# Page Config
st.set_page_config(
    page_title="MinMax - AI Assistant",
    page_icon="🚀",
    layout="centered",
    initial_sidebar_state="collapsed"
)

def main():
    load_custom_css()
    
    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("🚀 MinMax")
        st.caption("AI Consultant - Maxy Academy")
    with col2:
        st.markdown('<div class="status-badge"><div class="status-dot"></div>Online</div>', unsafe_allow_html=True)
    
    st.divider()
    
    # Check API Key
    groq_key = get_groq_api_key()
    
    # Validasi: Minimal satu key harus ada untuk jalan
    groq_key = get_groq_api_key()
    zai_key = getattr(ZaiConfig, 'get_api_key', lambda: "")()

    if not groq_key and not zai_key:
        st.warning("⚠️ Belum ada API Key yang terdeteksi. Silakan masukkan salah satu atau keduanya.")
        
        with st.form("api_key_form"):
            st.markdown("### 🔑 Masukkan API Key")
            st.caption("Masukkan minimal satu key. Jika keduanya diisi, sistem akan auto-switch ke Z.ai jika Groq error.")
            
            col1, col2 = st.columns(2)
            with col1:
                groq_input = st.text_input("Groq API Key", type="password", placeholder="gsk_...", help="Dapatkan di console.groq.com")
            with col2:
                zai_input = st.text_input("Z.ai (GLM) API Key", type="password", placeholder="Key Z.ai...", help="Dapatkan di open.bigmodel.cn")
            
            st.markdown("Belum punya API Key? [Dapatkan Groq Key](https://console.groq.com/keys) atau [Z.ai Key](https://open.bigmodel.cn/)")
            
            submit = st.form_submit_button("✅ Simpan & Mulai", use_container_width=True)
            
            if submit:
                if groq_input or zai_input:
                    if groq_input:
                        st.session_state.groq_api_key = groq_input
                    if zai_input:
                        st.session_state.zai_api_key = zai_input
                    st.success("✅ API Key tersimpan!")
                    st.rerun()
                else:
                    st.error("❌ Masukkan setidaknya satu API Key!")
        
        st.stop()
    
    # Initialize messages
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": """**Halo kak! 👋 Selamat datang di Maxy Academy AI Assistant!**

Aku **MinMax**, siap bantu kakak menemukan program karir impian atau jawab pertanyaan seputar Maxy Academy. 🚀

Kakak bisa tanya apa saja, misalnya:
- "Apa itu program Fast Track?"
- "Ada bootcamp Data Science ga?"
- "Cara daftar magang gimana kak?"

Yuk, mau tanya apa hari ini? 😊"""}
        ]
    
    # Load RAG
    if "rag_loaded" not in st.session_state:
        with st.status("🔄 Mempersiapkan MinMax...", expanded=True) as status:
            st.write("📂 Memuat data & core system...")
            rag = load_rag_system()
            
            if rag is None:
                status.update(label="❌ Gagal memuat sistem AI", state="error", expanded=True)
                st.error("Silakan refresh halaman atau cek koneksi internet.")
                st.stop()
            
            status.update(label="✅ MinMax siap!", state="complete", expanded=False)
            
        st.session_state.rag_system = rag
        st.session_state.rag_loaded = True
        st.rerun()
    
    # Display Messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar="🤖" if message["role"] == "assistant" else "👤"):
            st.markdown(message["content"])
    
    # Quick Actions
    if len(st.session_state.messages) <= 2:
        st.markdown("**💡 Pertanyaan Populer:**")
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
        
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("MinMax sedang berpikir... 💭"):
                if "rag_system" not in st.session_state or st.session_state.rag_system is None:
                     answer = "⚠️ Sistem AI belum siap. Silakan refresh halaman."
                     response = {"answer": answer}
                else:
                     response = st.session_state.rag_system.query(prompt, st.session_state.messages[:-1])
                     answer = response["answer"]
            
            st.markdown(answer)
            
            # REFERENCE REMOVED BY USER REQUEST
            # if response.get("sources"):
            #     with st.expander("📚 Sumber Referensi"):
            #         for src in response["sources"]:
            #             st.caption(f"• {src}")
        
        st.session_state.messages.append({"role": "assistant", "content": answer})
    
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/150x50/E67E22/FFFFFF?text=MAXY", width='stretch')
        st.markdown("### 🎯 Contoh Pertanyaan")
        
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
        
        if st.button("🗑️ Hapus Chat", width='stretch'):
            st.session_state.messages = [st.session_state.messages[0]] # Reset to welcome
            st.rerun()
        
        st.divider()
        
        st.markdown("### ⚙️ Settings")
        groq_input = st.text_input(
            "Groq API Key",
            value=st.session_state.get("groq_api_key", ""),
            type="password",
            help="API Key dari Groq",
            key="groq_key_input_field"
        )
        
        if st.button("💾 Simpan API Key", use_container_width=True):
            st.session_state.groq_api_key = groq_input
            if "rag_loaded" in st.session_state:
                del st.session_state.rag_loaded
            if "rag_system" in st.session_state:
                del st.session_state.rag_system
            st.success("API Key tersimpan! Reload sistem...")
            st.rerun()
        
        st.markdown("**Status:**")
        groq_key = get_groq_api_key()
        zai_key = getattr(st.secrets, "ZAI_API_KEY", None) or os.getenv("ZAI_API_KEY")
        
        if groq_key:
            st.caption(f"✅ Groq: ...{groq_key[-8:]}")
        elif zai_key:
            st.caption(f"✅ Z.ai: ...{zai_key[-8:]}")
        else:
            st.caption("⚠️ API Key: Belum diset")

        st.divider()
        st.caption("Powered by **Groq LPU**")
        st.caption("© 2026 Maxy Academy")

if __name__ == "__main__":
    main()
