import os
import psycopg2
from psycopg2.extras import RealDictCursor
from pgvector.psycopg2 import register_vector

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/rag_artikel")

def get_db_connection():
    conn = psycopg2.connect(DATABASE_URL)
    try:
        register_vector(conn)
    except psycopg2.ProgrammingError:
        pass
    return conn

def init_db():
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    conn.commit()
    
    register_vector(conn)
    
    cur.execute("""
        CREATE TABLE IF NOT EXISTS kitab (
            id SERIAL PRIMARY KEY,
            judul VARCHAR(255) UNIQUE NOT NULL,
            status_indeks VARCHAR(50) DEFAULT 'Terindeks',
            total_halaman INT DEFAULT 0,
            total_chunk INT DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    
    # Reset / Buat tabel chunks dengan dimensi 384
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id SERIAL PRIMARY KEY,
            kitab_judul VARCHAR(255) REFERENCES kitab(judul) ON DELETE CASCADE,
            jilid INT DEFAULT 1,
            bab VARCHAR(255) DEFAULT 'Umum',
            halaman_awal INT NOT NULL,
            halaman_akhir INT NOT NULL,
            teks_asli TEXT NOT NULL,
            hash_teks VARCHAR(64) UNIQUE NOT NULL,
            embedding vector(384),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_chunks_kitab ON chunks(kitab_judul);
    """)
    
    conn.commit()
    cur.close()
    conn.close()
    print("✓ Skema database & ekstensi pgvector berhasil diinisialisasi.")

if __name__ == "__main__":
    init_db()
