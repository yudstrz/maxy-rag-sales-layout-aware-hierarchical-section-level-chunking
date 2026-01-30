# -*- coding: utf-8 -*-
"""Maxy_RAG_Streamlit.py

Maxy Academy RAG System V3 - Streamlit Edition
Premium Chat UI dengan Native Streamlit Components
"""

import os
import sys
import time
import json
import hashlib
import warnings
import traceback

# Suppress warnings early
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("[STARTUP] Importing core libraries...", flush=True)

import requests
import openai
import streamlit as st
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

# ==========================================
# Configuration
# ==========================================
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
    if os.path.exists("d:/MAXY ACADEMY/Maxy-RAG"):
        BASE_PATH = "d:/MAXY ACADEMY/Maxy-RAG/"
    elif os.path.exists("/content/"):
        BASE_PATH = "/content/"
    else:
        BASE_PATH = "./"

    BOOTCAMP_DATASET = os.path.join(BASE_PATH, "MAXY_Bootcamp_Dataset.jsonl")
    CURRICULUM_DATASET = os.path.join(BASE_PATH, "MAXY_Curriculum_Syllabus.jsonl")
    COMPANY_INFO = os.path.join(BASE_PATH, "MAXY_Company_Info.jsonl")

    EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    # Use smaller reranker for Streamlit Cloud (bge-reranker-v2-m3 is too large ~560MB)
    RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # ~80MB, much faster

    TOP_K_ABSTRACT_BM25 = 30
    TOP_K_DENSE = 15
    TOP_K_RERANK = 10
    TOP_K_FINAL = 5

# ==========================================
# Section Chunk Data Structure
# ==========================================
@dataclass
class SectionChunk:
    section_id: str
    section_path: str
    title: str
    full_text: str
    abstract: str
    source_type: str
    position: int
    metadata: dict = field(default_factory=dict)
    
    def to_langchain_doc(self, use_abstract: bool = False):
        from langchain_core.documents import Document
        content = self.abstract if use_abstract else self.full_text
        return Document(
            page_content=content,
            metadata={
                'section_id': self.section_id,
                'section_path': self.section_path,
                'title': self.title,
                'source_type': self.source_type,
                'position': self.position,
                **self.metadata
            }
        )

# ==========================================
# Layout-Aware Chunker
# ==========================================
class LayoutAwareChunker:
    def __init__(self, config: HybridRAGConfig):
        self.config = config
        self.position_counter = 0
    
    def _generate_id(self, content: str) -> str:
        return hashlib.md5(content.encode('utf-8')).hexdigest()[:12]
    
    def _create_abstract(self, full_text: str, title: str = "") -> str:
        sentences = full_text.replace('\n', ' ').split('.')
        sentences = [s.strip() for s in sentences if s.strip()]
        abstract_parts = []
        if title:
            abstract_parts.append(title)
        for s in sentences[:2]:
            if len(s) > 20:
                abstract_parts.append(s + '.')
        abstract = ' '.join(abstract_parts)
        if len(abstract) > 800:
            abstract = abstract[:800] + '...'
        return abstract
    
    def process_bootcamp(self, path: str) -> List[SectionChunk]:
        sections = []
        if not os.path.exists(path):
            return sections
        with open(path, 'r', encoding='utf-8-sig') as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                kategori = item.get('Kategori', 'Umum')
                nama = item.get('nama', item.get('program_name', 'Unknown'))
                details = item.get('details', '')
                url = item.get('url bootcamp', item.get('url_bootcamp', ''))
                section_path = f"Bootcamp > {kategori} > {nama}"
                full_text = f"**Program: {nama}**\nKategori: {kategori}\n\n{details}\n"
                if url:
                    full_text += f"\nLink Program: {url}"
                abstract = self._create_abstract(full_text, title=nama)
                sections.append(SectionChunk(
                    section_id=self._generate_id(full_text),
                    section_path=section_path,
                    title=nama,
                    full_text=full_text,
                    abstract=abstract,
                    source_type='bootcamp',
                    position=self.position_counter,
                    metadata={'kategori': kategori, 'url': url}
                ))
                self.position_counter += 1
        return sections
    
    def process_curriculum(self, path: str) -> List[SectionChunk]:
        sections = []
        if not os.path.exists(path):
            return sections
        with open(path, 'r', encoding='utf-8-sig') as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                program = item.get('program_name', item.get('course', 'Unknown'))
                module = item.get('module', '')
                topic = item.get('topic', item.get('session', ''))
                day = item.get('day', '')
                overview = item.get('overview', '')
                duration = item.get('duration', '')
                tools = item.get('resources_tools', '')
                
                path_parts = [program]
                if day:
                    path_parts.append(f"Hari {day}")
                if topic:
                    path_parts.append(topic)
                section_path = " > ".join(path_parts)
                
                full_text = f"**{topic or module}**\nProgram: {program}\n"
                if day:
                    full_text += f"Hari ke-{day}\n"
                if duration:
                    full_text += f"Durasi: {duration}\n"
                full_text += f"\n{overview}\n"
                if tools:
                    full_text += f"\nTools: {tools}\n"
                
                abstract = self._create_abstract(full_text, title=topic or module)
                sections.append(SectionChunk(
                    section_id=self._generate_id(full_text),
                    section_path=section_path,
                    title=topic or module or program,
                    full_text=full_text,
                    abstract=abstract,
                    source_type='curriculum',
                    position=self.position_counter,
                    metadata={'program_name': program, 'day': day}
                ))
                self.position_counter += 1
        return sections
    
    def process_company(self, path: str) -> List[SectionChunk]:
        sections = []
        if not os.path.exists(path):
            return sections
        with open(path, 'r', encoding='utf-8-sig') as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                kategori = item.get('kategori', 'Info')
                pertanyaan = item.get('pertanyaan', '')
                konten = item.get('konten', '')
                section_path = f"Maxy Academy > {kategori}"
                title = pertanyaan if pertanyaan else kategori
                full_text = f"**{kategori}**\n"
                if pertanyaan:
                    full_text += f"Q: {pertanyaan}\n"
                full_text += f"\n{konten}"
                abstract = self._create_abstract(full_text, title=title)
                sections.append(SectionChunk(
                    section_id=self._generate_id(full_text),
                    section_path=section_path,
                    title=title,
                    full_text=full_text,
                    abstract=abstract,
                    source_type='company',
                    position=self.position_counter,
                    metadata={'kategori': kategori}
                ))
                self.position_counter += 1
        return sections
    
    def process_all(self) -> List[SectionChunk]:
        all_sections = []
        all_sections.extend(self.process_bootcamp(self.config.BOOTCAMP_DATASET))
        all_sections.extend(self.process_company(self.config.COMPANY_INFO))
        all_sections.extend(self.process_curriculum(self.config.CURRICULUM_DATASET))
        return all_sections

# ==========================================
# Multi-LLM (Gemini + OpenRouter)
# ==========================================


class GroqLLM:
    def __init__(self, api_key: str):
        self.client = None
        if api_key:
            self.client = openai.OpenAI(base_url="https://api.groq.com/openai/v1", api_key=api_key)
        self.models = GroqConfig.MODEL_LIST

    def generate(self, prompt: str, system_prompt: str = "") -> Optional[str]:
        if not self.client:
            return None
        for model in self.models:
            try:
                print(f"[GROQ] Trying model: {model}", flush=True)
                completion = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=500,
                )
                print(f"[GROQ] Success with model: {model}", flush=True)
                return completion.choices[0].message.content
            except Exception as e:
                print(f"[GROQ] Error with {model}: {str(e)}", flush=True)
                continue
        return None



# ==========================================
# Streamlit UI - Native Components
# ==========================================
# Streamlit UI - Native Components
# ==========================================

# Page Config
st.set_page_config(
    page_title="Kak Maxy - AI Assistant",
    page_icon="🚀",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS - Compatible with Streamlit
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* Main App */
.stApp {
    font-family: 'Inter', sans-serif;
}

/* Hide Streamlit Elements */
#MainMenu, footer, header {visibility: hidden;}

/* Chat Input Styling */
.stChatInput > div {
    border-radius: 25px !important;
    border: 2px solid #E67E22 !important;
}

.stChatInput textarea {
    font-family: 'Inter', sans-serif !important;
}

/* Chat Message Styling */
.stChatMessage {
    background: transparent !important;
    border: none !important;
}

/* Assistant Message Bubble */
[data-testid="stChatMessageContent"] {
    background: #ffffff !important; /* Force white background for assistant */
    color: #2c3e50 !important; /* Force dark text for assistant */
    border-radius: 18px !important;
    padding: 12px 16px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1) !important;
    border: 1px solid #e0e0e0 !important;
}

/* User Message Bubble */
[data-testid="stChatMessageContent"][class*="user"] {
    background: linear-gradient(135deg, #3498DB, #2980B9) !important;
    color: white !important;
    border: none !important;
}

/* Make sure text inside bubbles is readable */
[data-testid="stChatMessageContent"] p, 
[data-testid="stChatMessageContent"] div {
    color: inherit !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #f8f9fa !important;
}

/* Fix sidebar text color in dark mode */
section[data-testid="stSidebar"] p, 
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #2c3e50 !important;
}

section[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    border-radius: 12px;
    border: 1.5px solid #e0e0e0;
    background: white !important;
    color: #2c3e50 !important;
    font-weight: 500;
    padding: 10px 16px;
    margin-bottom: 8px;
    transition: all 0.2s;
    text-align: left;
}

section[data-testid="stSidebar"] .stButton > button:hover {
    background: #E67E22 !important;
    color: white !important;
    border-color: #E67E22 !important;
    transform: translateX(4px);
}

/* Expander - Fix for Dark Mode */
.streamlit-expanderHeader {
    background: #ffffff !important;
    color: #2c3e50 !important;
    border-radius: 10px !important;
    border: 1px solid #e0e0e0 !important;
}

.streamlit-expanderContent {
    background: #ffffff !important;
    color: #2c3e50 !important;
    border-radius: 0 0 10px 10px !important;
    border-top: none !important;
}

/* Title styling */
h1 {
    background: linear-gradient(135deg, #E67E22, #D35400);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

/* Quick buttons in main area */
.stButton > button {
    border-radius: 20px;
    border: 1.5px solid #3498DB;
    background: transparent;
    color: #3498DB;
    font-weight: 500;
    transition: all 0.2s;
}

.stButton > button:hover {
    background: #3498DB;
    color: white;
}

/* Status badge */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #e8f5e9;
    color: #2e7d32;
    padding: 4px 12px;
    border-radius: 15px;
    font-size: 13px;
    font-weight: 500;
    border: 1px solid #c8e6c9;
}

.status-dot {
    width: 8px;
    height: 8px;
    background: #2e7d32;
    border-radius: 50%;
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_rag_system():
    """Load RAG system with progress tracking."""
    print("[RAG] Starting load_rag_system...", flush=True)
    
    try:
        print("[RAG] Importing LangChain components...", flush=True)
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_community.vectorstores import FAISS
        from langchain_community.retrievers import BM25Retriever
        from sentence_transformers import CrossEncoder
        print("[RAG] LangChain imports successful!", flush=True)
        
        config = HybridRAGConfig()
        print(f"[RAG] Config loaded. BASE_PATH={config.BASE_PATH}", flush=True)
        
        # Step 1: Load Data
        print("[RAG] Loading data...", flush=True)
        
        chunker = LayoutAwareChunker(config)
        sections = chunker.process_all()
        print(f"[RAG] Loaded {len(sections)} sections", flush=True)
    
        if not sections:
            return None
        
        # Step 2: Load Embedding Model
        print("[RAG] Loading embedding model...", flush=True)
        embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
        
        # Step 3: Build BM25 Index
        print("[RAG] Building BM25 index...", flush=True)
        section_map = {s.section_id: s for s in sections}
        abstract_docs = [s.to_langchain_doc(use_abstract=True) for s in sections]
        abstract_bm25 = BM25Retriever.from_documents(abstract_docs)
        abstract_bm25.k = config.TOP_K_ABSTRACT_BM25
        
        # Step 4: Build FAISS Vector Store
        print("[RAG] Building FAISS vector store...", flush=True)
        full_docs = [s.to_langchain_doc(use_abstract=False) for s in sections]
        vectorstore = FAISS.from_documents(full_docs, embeddings)
        
        # Step 5: Load Reranker Model
        print("[RAG] Loading reranker model...", flush=True)
        reranker = CrossEncoder(config.RERANKER_MODEL)
        
        # Step 6: Initialize LLM
        print("[RAG] Initializing Groq LLM...", flush=True)
        llm = GroqLLM(GroqConfig.get_api_key())
        
        # Create RAG object with pre-loaded components
        rag = HybridRAGPreloaded(
            config=config,
            sections=sections,
            section_map=section_map,
            embeddings=embeddings,
            abstract_bm25=abstract_bm25,
            vectorstore=vectorstore,
            reranker=reranker,
            llm=llm
        )
        
        print("[RAG] System loaded successfully!", flush=True)
    
    except Exception as e:
        print(f"[ERROR] Failed to load RAG system: {str(e)}", flush=True)
        traceback.print_exc()
        return None

    return rag

class HybridRAGPreloaded:
    """RAG System with pre-loaded components (for progress tracking)."""
    
    def __init__(self, config, sections, section_map, embeddings, abstract_bm25, vectorstore, reranker, llm):
        self.config = config
        self.sections = sections
        self.section_map = section_map
        self.embeddings = embeddings
        self.abstract_bm25 = abstract_bm25
        self.vectorstore = vectorstore
        self.reranker = reranker
        self.llm = llm
    
    def retrieve(self, query: str) -> List[SectionChunk]:
        abstract_results = self.abstract_bm25.invoke(query)
        candidate_ids = [doc.metadata['section_id'] for doc in abstract_results]
        
        if not candidate_ids:
            return []
        
        faiss_results = self.vectorstore.similarity_search_with_score(query, k=min(len(candidate_ids), self.config.TOP_K_DENSE * 2))
        dense_ids = [doc.metadata['section_id'] for doc, _ in faiss_results if doc.metadata['section_id'] in candidate_ids][:self.config.TOP_K_DENSE]
        
        if not dense_ids:
            dense_ids = candidate_ids[:self.config.TOP_K_DENSE]
        
        sections = [self.section_map[sid] for sid in dense_ids if sid in self.section_map]
        if not sections:
            return []
        
        pairs = [[query, s.full_text] for s in sections]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(sections, scores), key=lambda x: x[1], reverse=True)
        
        seen = set()
        unique = []
        for s, _ in ranked:
            if s.section_id not in seen:
                seen.add(s.section_id)
                unique.append(s)
        
        return unique[:self.config.TOP_K_FINAL]
    
    def query(self, question: str) -> Dict:
        sections = self.retrieve(question)
        q_lower = question.lower()
        
        custom_keywords = ["custom", "by request", "sesuai kebutuhan", "khusus"]
        if any(kw in q_lower for kw in custom_keywords):
            return {
                "answer": """Wah, keren nih kak! 🙌

Maxy Academy memang open untuk program **custom by request** sesuai kebutuhan perusahaan kakak.

Boleh cerita dulu kak:
- **Bidang apa** yang mau dipelajari?
- **Berapa orang** yang akan ikut?
- **Durasi** yang diinginkan?

👉 [Chat Admin untuk Request Program Custom](https://wa.me/62811355993?text=Halo%20Admin%20Maxy)""",
                "sources": []
            }
        
        if not sections:
            return {
                "answer": """Hmm, aku belum nemuin info spesifik soal itu kak 😅

Tapi kakak bisa langsung tanya ke Admin Maxy:

👉 [Chat Admin via WhatsApp](https://wa.me/62811355993)""",
                "sources": []
            }

        context_text = ""
        for i, section in enumerate(sections):
            context_text += f"[SUMBER {i+1}] {section.section_path}\n{section.full_text}\n\n"

        system_prompt = """Kamu adalah 'Kak Maxy', AI Consultant Maxy Academy yang ramah, natural, dan helpful! 🚀

PERSONALITY:
- Panggil user dengan "Kak" atau "Kakak"
- Gunakan bahasa casual tapi tetap profesional
- Boleh pakai emoji secukupnya (1-3 per pesan)
- Kalau ada pertanyaan yang kurang jelas, TANYA BALIK dulu sebelum kasih rekomendasi

ATURAN UTAMA:
1. Kamu BOLEH menyebutkan sumber dengan cara natural, contoh:
   - "Di materi AI Day 1 tentang Ethics..."
   - "Menurut info program Data Science..."
   - "Berdasarkan kurikulum Bootcamp ML..."
2. JANGAN menyebutkan "[SUMBER 1]" atau "[INFO 1]" secara langsung - gunakan nama program/topik.
3. JANGAN mengarang info yang tidak ada di konteks.

ATURAN BERDASARKAN JENIS PERTANYAAN:

1. Untuk SAPAAN/BASA-BASI/CASUAL CHAT (halo, selamat pagi, apa kabar, pujian, candaan, dll):
   - Balas dengan ramah dan natural seperti manusia
   - Contoh sapaan: "Halo kak! Selamat pagi juga! Ada yang bisa Kak Maxy bantu hari ini? 😊"
   - Contoh pujian: "Wah makasih kak! Kakak juga pasti keren. Ada yang mau ditanyakan tentang program Maxy?"
   - JANGAN PERNAH langsung kasih "Rekomendasi Program:" untuk chat casual seperti ini!

2. Untuk pertanyaan INFO PERUSAHAAN (Instagram, Facebook, CEO, alamat, harga, dll):
   - Jawab langsung dan singkat tanpa rekomendasi program
   - Contoh: "Instagram Maxy Academy: https://www.instagram.com/maxy.academy"

3. Untuk pertanyaan BELAJAR/KARIR/BOOTCAMP (ingin belajar X, rekomendasi program, dll):
   - Jawab dengan penjelasan + rekomendasi 1-3 program relevan
   - Sertakan nama program, kategori, apa yang dipelajari, dan link jika ada
   - PENTING: HANYA rekomendasikan program Maxy Academy yang ada di konteks. JANGAN menyebut bootcamp dari luar Maxy Academy.
   - Format:
     **Rekomendasi Program:**
     1. **Nama Program** (Kategori)
        - Apa yang dipelajari
        - Link: [jika ada di konteks]

4. Untuk pertanyaan DETAIL/KURIKULUM/SILABUS BOOTCAMP (materi apa saja, hari ke berapa, tools apa):
   - Jelaskan detail materi per hari/session jika ada di konteks
   - Sertakan: topik, durasi, tools yang digunakan, dan task/studi kasus
   - Contoh: "Di Hari ke-3, kamu akan belajar HTML & CSS (1 jam) dengan tools VSCode..."

5. Untuk hal yang TIDAK DIKETAHUI:
   - Bilang: "Aku kurang tau soal itu kak, tapi kakak bisa tanya langsung ke Admin Maxy ya!"
   - Arahkan ke WhatsApp Admin

6. STRATEGI SALES CONSULTANT:
   - Posisikan diri sebagai "Konsultan Karir", bukan admin/robot
   - Fokus pada 'Kecepatan Penyaluran Kerja', bukan fitur/materi
   - Pahami pain point: "Experience Paradox" (butuh kerja tapi butuh pengalaman)
   - Prioritaskan Fast Track (Rp8jt) sebagai solusi utama
   - Gunakan Reguler (Rp5jt) sebagai pembanding (Decoy Effect)

7. HANDLING OBJECTION "MAHAL":
   - JIKA user bilang "mahal" TANPA konteks → tanya dulu: "Mahal yang mana kak? Boleh tau paket yang dimaksud?"
   - JIKA user sudah sebut paket spesifik (misal "Fast Track mahal"):
     * Jangan minta maaf soal harga
     * Jelaskan ROI: Investasi 8jt, magang gaji UMR 4.5jt/bulan -> 2 bulan balik modal
     * Tekankan: "Ini investasi karir, bukan pengeluaran"
   - JIKA user tetap keberatan → tawarkan opsi cicilan atau paket Reguler

8. Akhiri dengan pertanyaan lanjutan atau link WA jika relevan.

INGAT: Untuk sapaan dan basa-basi, JANGAN tulis "Rekomendasi Program:" sama sekali!"""

        full_prompt = f"""Konteks:
{context_text}

Pertanyaan: {question}

Jawaban (singkat, relevan):"""

        answer = self.llm.generate(full_prompt, system_prompt)
        
        wa_link = "\n\n👉 [Chat Admin via WhatsApp](https://wa.me/62811355993)"
        high_intent = ["daftar", "biaya", "bayar", "gabung", "join"]
        if any(kw in q_lower for kw in high_intent) and "wa.me" not in answer:
            answer += wa_link
        
        return {"answer": answer, "sources": [s.section_path for s in sections]}

def main():
    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("🚀 Kak Maxy")
        st.caption("AI Consultant - Maxy Academy")
    with col2:
        st.markdown('<div class="status-badge"><div class="status-dot"></div>Online</div>', unsafe_allow_html=True)
    
    st.divider()
    
    # Check API Key - show input if not set
    # Check API Key - show input if not set
    groq_key = get_groq_api_key()
    
    if not groq_key:
        st.warning("⚠️ Groq API Key belum diset. Masukkan API Key untuk mulai chat.")
        
        with st.form("api_key_form"):
            st.markdown("**🔑 Masukkan Groq API Key:**")
            api_key_input = st.text_input(
                "Groq API Key",
                type="password",
                placeholder="gsk_xxxxxxxxxxxxxxxxxxxxxxxx",
                help="Dapatkan gratis di console.groq.com"
            )
            st.markdown("[📎 Dapatkan Groq API Key di sini](https://console.groq.com/keys)")
            
            submit = st.form_submit_button("✅ Simpan & Mulai", use_container_width=True)
            
            if submit:
                if api_key_input:
                    st.session_state.groq_api_key = api_key_input
                    st.success("✅ API Key tersimpan!")
                    st.rerun()
                else:
                    st.error("❌ Masukkan Groq API Key!")
        
        st.stop()
    
    # Initialize messages
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Halo kak! 👋 Aku Kak Maxy, AI Consultant dari Maxy Academy.\n\nAda yang bisa aku bantu hari ini?"}
        ]
    
    if "rag_loaded" not in st.session_state:
        # Show detailed loading progress using st.status
        with st.status("🔄 Mempersiapkan Kak Maxy...", expanded=True) as status:
            st.write("📂 Memuat data bootcamp & curriculum...")
            st.write("🧠 Mengunduh model embedding (pertama kali bisa lama ~500MB)...")
            st.write("🔍 Membangun BM25 search index...")
            st.write("📊 Membangun vector database (FAISS)...")
            st.write("⚖️ Memuat reranker model (~80MB)...")
            st.write("🤖 Menginisialisasi Groq AI...")
            
            # Actually load the RAG system
            rag = load_rag_system()
            
            if rag is None:
                status.update(label="❌ Gagal memuat sistem AI", state="error", expanded=True)
                st.error("Silakan refresh halaman atau cek koneksi internet.")
                st.stop()
            
            status.update(label="✅ Kak Maxy siap!", state="complete", expanded=False)
            
        st.session_state.rag_system = rag
        st.session_state.rag_loaded = True
        st.rerun()
    
    # Display Messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar="🤖" if message["role"] == "assistant" else "👤"):
            st.markdown(message["content"])
    
    # Quick Actions (only at start)
    if len(st.session_state.messages) <= 2:
        st.markdown("**💡 Pertanyaan Populer:**")
        cols = st.columns(2)
        quick_qs = [
            "Apa saja daftar lengkap akun media sosial Maxy Academy?",
            "Info paket Fast Track",
            "Info paket Bootcamp 5 minggu",
            "Info paket Bootcamp 8 minggu"
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

    # Generate Response Implementation (Triggered if last message is from user)
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        prompt = st.session_state.messages[-1]["content"]
        
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Kak Maxy sedang berpikir... 💭"):
                # Ensure RAG system is loaded
                if "rag_system" not in st.session_state or st.session_state.rag_system is None:
                     answer = "⚠️ Sistem AI belum siap. Silakan refresh halaman."
                     response = {"answer": answer}
                else:
                     response = st.session_state.rag_system.query(prompt)
                     answer = response["answer"]
            
            st.markdown(answer)
            
            # Show sources
            if response.get("sources"):
                with st.expander("📚 Sumber Referensi"):
                    for src in response["sources"]:
                        st.caption(f"• {src}")
        
        st.session_state.messages.append({"role": "assistant", "content": answer})
        # Optional: Rerun to update the UI specifically if you want the "assistant" message 
        # to be part of the main display loop next time, but it's already rendered above.
        # No rerun needed here as it just finished rendering.
    
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/150x50/E67E22/FFFFFF?text=MAXY", width='stretch')
        st.markdown("### 🎯 Contoh Pertanyaan")
        
        examples = [
            "Apa saja daftar lengkap akun media sosial Maxy Academy?",
            "Info paket Fast Track",
            "Info paket Bootcamp 5 minggu",
            "Info paket Bootcamp 8 minggu",
            "Siapa CEO Maxy?",
            "Request program custom",
        ]
        
        for ex in examples:
            if st.button(ex, key=f"sidebar_{ex}"):
                st.session_state.messages.append({"role": "user", "content": ex})
                st.rerun()
        
        st.divider()
        
        if st.button("🗑️ Hapus Chat", width='stretch'):
            st.session_state.messages = [
                {"role": "assistant", "content": "Halo kak! 👋 Aku Kak Maxy, AI Consultant dari Maxy Academy.\n\nAda yang bisa aku bantu hari ini?"}
            ]
            st.rerun()
        
        st.divider()
        
        # API Key Settings
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
            # Clear cached RAG so it reloads with new key
            if "rag_loaded" in st.session_state:
                del st.session_state.rag_loaded
            if "rag_system" in st.session_state:
                del st.session_state.rag_system
            st.success("API Key tersimpan! Reload sistem...")
            st.rerun()
        
        # Show current API key status
        st.markdown("**Status:**")
        groq_key = get_groq_api_key()
        
        if groq_key:
            st.caption(f"✅ Groq: ...{groq_key[-8:]}")
        else:
            st.caption("⚠️ Groq: Belum diset")

        st.divider()
        st.caption("Powered by **Groq LPU**")
        st.caption("© 2026 Maxy Academy")

if __name__ == "__main__":
    main()
