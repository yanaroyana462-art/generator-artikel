import os
import shutil
import json
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor
from fastembed import TextEmbedding
from google import genai
from google.genai import types

# Memuat variabel lingkungan
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY belum dikonfigurasi di berkas .env")

# Inisialisasi Google GenAI Client
client = genai.Client(api_key=GEMINI_API_KEY)

# Inisialisasi Model Embedding (FastEmbed - BGE Small)
embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

app = FastAPI(
    title="RAG Content Engine API",
    description="Backend API untuk Ingest Kitab PDF dan Generasi Artikel Berbasis Vector Search & Gemini AI",
    version="1.0.0"
)

# Konfigurasi CORS agar frontend Vercel bisa mengakses API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Anda bisa mengganti dengan domain Vercel spesifik
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- DATABASE INITIALIZATION ---
@app.on_event("startup")
def startup_db():
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            # Tabel untuk menyimpan artikel yang siap dibagikan lewat tautan unik
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS shared_articles (
                    id SERIAL PRIMARY KEY,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    references_json JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            cursor.execute("ALTER TABLE shared_articles ADD COLUMN IF NOT EXISTS article_type TEXT;")
            # Tabel pelacak aktivitas pembaca (Leads Database)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS readers (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    phone VARCHAR(50),
                    article_id INT REFERENCES shared_articles(id),
                    viewed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            conn.commit()
    finally:
        conn.close()

# --- HELPER FUNCTIONS ---

def get_db_connection():
    try:
        conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
        return conn
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Koneksi Database Gagal: {str(e)}")

def search_relevant_chunks(topic: str, selected_kitab_ids: List[int], top_k: int = 5):
    """
    Melakukan pencarian semantik (Vector Similarity Search) di PostgreSQL
    """
    conn = get_db_connection()
    try:
        # Generate embedding untuk topik query
        query_embedding = list(embedding_model.embed([topic]))[0].tolist()
        
        with conn.cursor() as cursor:
            # Query Cosine Distance (<->) menggunakan pgvector
            if selected_kitab_ids:
                query = """
                    SELECT c.id, c.content, c.page_number, k.title as kitab_title
                    FROM chunks c
                    JOIN kitabs k ON c.kitab_id = k.id
                    WHERE c.kitab_id = ANY(%s)
                    ORDER BY c.embedding <-> %s::vector
                    LIMIT %s;
                """
                cursor.execute(query, (selected_kitab_ids, query_embedding, top_k))
            else:
                query = """
                    SELECT c.id, c.content, c.page_number, k.title as kitab_title
                    FROM chunks c
                    JOIN kitabs k ON c.kitab_id = k.id
                    ORDER BY c.embedding <-> %s::vector
                    LIMIT %s;
                """
                cursor.execute(query, (query_embedding, top_k))
                
            results = cursor.fetchall()
            return results
    finally:
        conn.close()

# --- SCHEMAS ---

class GenerateRequest(BaseModel):
    topic: str
    target_audience: Optional[str] = "Umum"
    article_length: Optional[str] = "Sedang"
    article_type: Optional[str] = "Sedang" # Ringan, Sedang, Ilmiah/Akademis
    article_format: Optional[str] = "Artikel Umum" # Khutbah, Diskusi, Makalah
    article_tone: Optional[str] = "Formal" # Formal, Semi-Formal, Santai
    selected_kitab_ids: Optional[List[int]] = []

class SaveArticleRequest(BaseModel):
    title: str
    content: str
    references: Optional[List[dict]] = []
    article_type: Optional[str] = "Umum"

class TrackReaderRequest(BaseModel):
    name: str
    phone: Optional[str] = ""
    article_id: int

# --- API ENDPOINTS ---

@app.get("/")
def read_root():
    return {"status": "online", "message": "RAG Content Engine Backend API Siap Digunakan"}

@app.get("/api/kitab")
def list_kitab():
    """Mengambil daftar semua kitab yang terdaftar di database"""
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT id, title, description, created_at FROM kitabs ORDER BY created_at DESC;")
            kitabs = cursor.fetchall()
            return {"success": True, "data": kitabs}
    finally:
        conn.close()

@app.post("/api/upload")
async def upload_kitab(
    title: str = Form(...),
    description: Optional[str] = Form(""),
    file: UploadFile = File(...)
):
    """
    Endpoint untuk menerima file PDF kitab, memproses ekstraksi & chunking,
    lalu memasukkan vektor ke PostgreSQL.
    """
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Hanya berkas format .pdf yang diperbolehkan.")

    # Simpan sementara berkas PDF di server
    os.makedirs("temp_uploads", exist_ok=True)
    file_path = os.path.join("temp_uploads", file.filename)
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Panggil modul ingest internal untuk ekstraksi dan embedding
        from ingest.main import process_and_ingest_pdf
        
        kitab_id = process_and_ingest_pdf(
            pdf_path=file_path,
            title=title,
            description=description
        )

        return {
            "success": True,
            "message": f"Kitab '{title}' berhasil diunggah dan diindeks ke dalam vector database.",
            "data": {
                "kitab_id": kitab_id,
                "title": title,
                "filename": file.filename
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal memproses berkas PDF: {str(e)}")
    finally:
        # Bersihkan berkas sementara
        if os.path.exists(file_path):
            os.remove(file_path)

@app.post("/api/generate")
async def generate_article(payload: GenerateRequest):
    """
    Endpoint RAG utama:
    1. Melakukan pencarian vektor (Cosine Similarity) berdasarkan topik.
    2. Menyusun Konteks Kutipan Kitab.
    3. Mengirimkan prompt kontekstual ke Gemini 2.5 Flash untuk penyusunan naskah ideologis kaffah.
    """
    if not payload.topic.strip():
        raise HTTPException(status_code=400, detail="Topik tidak boleh kosong.")

    # 1. Retrieval: Cari potongan teks yang relevan dari database
    context_chunks = search_relevant_chunks(
        topic=payload.topic,
        selected_kitab_ids=payload.selected_kitab_ids,
        top_k=5
    )

    if not context_chunks:
        retrieved_context_text = "Tidak ditemukan rujukan teks kitab yang spesifik. Susun naskah berdasarkan pemahaman umum Islam yang mendalam."
        references = []
    else:
        context_list = []
        references = []
        for idx, chunk in enumerate(context_chunks, 1):
            ref_str = f"[{idx}] Kitab: {chunk['kitab_title']} (Hal. {chunk['page_number']})"
            context_list.append(f"{ref_str}\nKutipan: {chunk['content']}")
            references.append({
                "kitab": chunk['kitab_title'],
                "page": chunk['page_number'],
                "excerpt": chunk['content'][:150] + "..."
            })
        retrieved_context_text = "\n\n".join(context_list)

    # 2. Prompt Engineering: Menginstruksikan Gemini dengan kaffah/ideologis tone
    system_instruction = (
        "Anda adalah seorang konseptor dan penulis narasi Islam ideologis (Kaffah) yang sangat mendalam, rasional, dan terstruktur. "
        "Tugas Anda adalah menulis artikel/naskah komprehensif berdasarkan rujukan kitab-kitab yang diberikan. "
        "Gunakan bahasa Indonesia yang lugas, tegas, berbasis argumen sahih, dan terstruktur rapi menggunakan format Markdown."
    )

    prompt = f"""
Sajikan sebuah artikel ideologis berbobot tinggi mengenai topik berikut:

TOPIK/TEMA: {payload.topic}
TARGET AUDIENS: {payload.target_audience}
PANJANG NASKAH: {payload.article_length}
GAYA PENULISAN: {payload.article_type}
FORMAT NASKAH: {payload.article_format}
GAYA BAHASA/TONE: {payload.article_tone}

---
RUJUKAN KITAB TERKASIH (KONTEKS RAG):
{retrieved_context_text}
---

PETUNJUK PENULISAN:
1. Buatlah Judul yang kuat, menggugah pemikiran, dan mencerminkan pandangan Islam Kaffah.
2. Tuliskan Pendahuluan yang mengurai problematika secara mendasar (First Principles).
3. Sertakan Pembahasan Utama dengan menyisipkan argumen ilmiah/syar'i dari rujukan kitab di atas. Gunakan catatan rujukan berbentuk [1], [2] pada kalimat yang relevan.
4. Akhiri dengan Kesimpulan Konkrit dan Strategis.
5. Sajikan output langsung dalam bentuk format HTML/Markdown bersih siap pakai.
"""

    try:
        # Panggil Gemini API (menggunakan Gemini 2.5 Flash)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.3, # Menjaga respon tetap terfokus dan presisi
            )
        )

        return {
            "success": True,
            "topic": payload.topic,
            "content": response.text,
            "references": references
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal menghasilkan artikel via Gemini API: {str(e)}")

@app.post("/api/articles")
def save_article(payload: SaveArticleRequest):
    """Menyimpan konten artikel agar bisa diakses lewat tautan publik unik"""
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                "INSERT INTO shared_articles (title, content, references_json, article_type) VALUES (%s, %s, %s, %s) RETURNING id;",
                (payload.title, payload.content, json.dumps(payload.references), payload.article_type)
            )
            article_id = cursor.fetchone()['id']
            conn.commit()
            return {"success": True, "article_id": article_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal menyimpan artikel: {str(e)}")
    finally:
        conn.close()

@app.get("/api/articles")
def list_articles():
    """Mengambil daftar semua artikel publik yang disimpan"""
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT id, title, article_type, created_at FROM shared_articles ORDER BY created_at DESC;")
            articles = cursor.fetchall()
            return {"success": True, "data": articles}
    finally:
        conn.close()

@app.get("/api/articles/{article_id}")
def get_article(article_id: int):
    """Mengambil data artikel publik berdasarkan ID"""
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT id, title, content, references_json, article_type FROM shared_articles WHERE id = %s;", (article_id,))
            article = cursor.fetchone()
            if not article:
                raise HTTPException(status_code=404, detail="Artikel tidak ditemukan")
            return {"success": True, "data": article}
    finally:
        conn.close()

@app.post("/api/readers/track")
def track_reader(payload: TrackReaderRequest):
    """Mencatat data pembaca baru ke dalam database pembaca"""
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                "INSERT INTO readers (name, phone, article_id) VALUES (%s, %s, %s) RETURNING id;",
                (payload.name, payload.phone, payload.article_id)
            )
            conn.commit()
            return {"success": True, "message": "Aktivitas pembaca berhasil dicatat"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal mencatat pembaca: {str(e)}")
    finally:
        conn.close()

@app.get("/api/readers")
def list_readers():
    """Mengambil semua daftar pembaca yang masuk lewat tautan publik"""
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT r.id, r.name, r.phone, r.viewed_at, a.title as article_title 
                FROM readers r
                LEFT JOIN shared_articles a ON r.article_id = a.id
                ORDER BY r.viewed_at DESC;
            """)
            readers = cursor.fetchall()
            return {"success": True, "data": readers}
    finally:
        conn.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)