import os
import streamlit as st
import json
import hashlib
import traceback
from typing import List, Dict

# External RAG libs
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder

# Local imports
from src.config import HybridRAGConfig, GroqConfig
from src.models import SectionChunk
from src.llm import GroqLLM

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
    
    def query(self, question: str, chat_history: List[Dict] = []) -> Dict:
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
        
        # Build history text (last 5 turns / 10 messages)
        history_text = ""
        if chat_history:
            recent = chat_history[-10:]
            for msg in recent:
                role = "User" if msg["role"] == "user" else "Kak Maxy"
                history_text += f"{role}: {msg['content']}\n"
        
        # If no sections found AND no history, return fallback.
        if not sections and not history_text:
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
4. **PENTING: PERHATIKAN RIWAYAT CHAT!**
   - Jika user mengajukan pertanyaan lanjutan (misal "Harganya berapa?" atau "Cara daftarnya?"), kamu HARUS menjawab berdasarkan konteks program/topik yang sedang dibahas di chat sebelumnya.
   - Jika user menggunakan kata ganti seperti "itu", "program tersebut", atau "bootcamp tadi", rujuklah ke topik terakhir di riwayat chat.

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
   - **WAJIB MENYERTAKAN LINK:** Jika ada info "Link Program" atau URL di konteks, KAMU HARUS MENULISNYA di jawaban.
   - Format:
     **Rekomendasi Program:**
     1. **Nama Program** (Kategori)
        - Apa yang dipelajari
        - 🔗 **Link Pendaftaran:** [Tulis URL disini]

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

        full_prompt = f"""=== RIWAYAT CHAT (PENTING: Gunakan ini untuk konteks pertanyaan lanjutan) ===
{history_text}

=== INFORMASI PENDUKUNG (RAG) ===
{context_text}

=== PERTANYAAN USER SAAT INI ===
{question}

INSTRUKSI:
Jawablah pertanyaan user {question} dengan ramah.
Jika pertanyaan user tidak jelas atau merujuk ke "itu/ini" (seperti "berapa harganya?"), LIHAT "RIWAYAT CHAT" untuk mengetahui apa yang sedang dibahas sebelumnya.
Gunakan "INFORMASI PENDUKUNG" untuk fakta dan data.

Jawaban (singkat, relevan, perhatikan riwayat chatting):"""

        answer = self.llm.generate(full_prompt, system_prompt)
        
        if not answer:
            answer = "⚠️ Maaf, terjadi kesalahan pada koneksi AI. Pastikan API Key sudah benar."

        wa_link = "\n\n👉 [Chat Admin via WhatsApp](https://wa.me/62811355993)"
        high_intent = ["daftar", "biaya", "bayar", "gabung", "join", "info", "tanya", "program", "bootcamp"]
        if any(kw in q_lower for kw in high_intent) and "wa.me" not in answer:
            answer += wa_link
        
        return {"answer": answer, "sources": [s.section_path for s in sections]}

@st.cache_resource
def load_rag_system():
    """Load RAG system with progress tracking."""
    print("[RAG] Starting load_rag_system...", flush=True)
    
    try:
        print("[RAG] Importing LangChain components...", flush=True)
        # Verify imports inside function to avoid caching issues if imports fail
        
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
