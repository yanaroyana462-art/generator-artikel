import os
import sys
import hashlib
import pdfplumber
from fastembed import TextEmbedding

# Tambahkan path root ke sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backend.app.core.database import get_db_connection, init_db

# Model Embedding Resmi FastEmbed
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

def generate_hash(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def extract_pdf_pages(file_path: str):
    pages_data = []
    with pdfplumber.open(file_path) as pdf:
        for index, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            pages_data.append({
                "page_num": index + 1,
                "text": text.strip()
            })
    return pages_data

def chunk_text(pages_data, chunk_size=400, overlap=50):
    chunks = []
    current_chunk = []
    current_tokens = 0
    start_page = 1

    for p in pages_data:
        words = p["text"].split()
        if not words:
            continue
            
        if not current_chunk:
            start_page = p["page_num"]

        for word in words:
            current_chunk.append(word)
            current_tokens += 1

            if current_tokens >= chunk_size:
                chunk_text_str = " ".join(current_chunk)
                chunks.append({
                    "halaman_awal": start_page,
                    "halaman_akhir": p["page_num"],
                    "teks_asli": chunk_text_str,
                    "hash_teks": generate_hash(chunk_text_str)
                })
                current_chunk = current_chunk[-overlap:]
                current_tokens = len(current_chunk)
                start_page = p["page_num"]

    if current_chunk:
        chunk_text_str = " ".join(current_chunk)
        chunks.append({
            "halaman_awal": start_page,
            "halaman_akhir": pages_data[-1]["page_num"] if pages_data else 1,
            "teks_asli": chunk_text_str,
            "hash_teks": generate_hash(chunk_text_str)
        })

    return chunks

def process_and_ingest(file_path: str, kitab_judul: str):
    print(f"\n---> Mengindeks Kitab: {kitab_judul}")
    
    print("Menyiapkan Model Embedding...")
    embedding_model = TextEmbedding(model_name=EMBEDDING_MODEL_NAME)
    
    # Inisialisasi Database
    init_db()

    pages = extract_pdf_pages(file_path)
    if not pages:
        print(f"ERROR: Tidak ada teks yang diekstrak dari {file_path}")
        return

    chunks = chunk_text(pages)
    print(f"Total Halaman: {len(pages)} | Total Chunk Dihasilkan: {len(chunks)}")

    print("Menghasilkan Embedding Vektor...")
    texts = [c["teks_asli"] for c in chunks]
    embeddings = list(embedding_model.embed(texts))

    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO kitab (judul, status_indeks, total_halaman, total_chunk)
        VALUES (%s, 'Terindeks', %s, %s)
        ON CONFLICT (judul) DO UPDATE 
        SET status_indeks = 'Terindeks', total_halaman = %s, total_chunk = %s;
    """, (kitab_judul, len(pages), len(chunks), len(pages), len(chunks)))

    inserted_count = 0
    for chunk, emb in zip(chunks, embeddings):
        try:
            cur.execute("""
                INSERT INTO chunks (kitab_judul, halaman_awal, halaman_akhir, teks_asli, hash_teks, embedding)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (hash_teks) DO NOTHING;
            """, (
                kitab_judul,
                chunk["halaman_awal"],
                chunk["halaman_akhir"],
                chunk["teks_asli"],
                chunk["hash_teks"],
                emb.tolist()
            ))
            if cur.rowcount > 0:
                inserted_count += 1
        except Exception as e:
            print(f"Gagal memasukkan chunk: {e}")

    conn.commit()
    cur.close()
    conn.close()
    print(f"✓ Ingest Selesai. {inserted_count} chunk baru berhasil disimpan ke database.\n")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Penggunaan: python -m ingest.main <path_pdf> <nama_kitab>")
    else:
        process_and_ingest(sys.argv[1], sys.argv[2])
