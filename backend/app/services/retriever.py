import os
import re
from typing import List, Dict, Any
from fastembed import TextEmbedding
from backend.app.core.database import get_db_connection

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.35"))

class RAGRetriever:
    def __init__(self):
        self.embedding_model = TextEmbedding(model_name=EMBEDDING_MODEL_NAME)

    def retrieve(self, query: str, kitab_list: List[str], top_k: int = 10) -> List[Dict[str, Any]]:
        # 1. Generate embedding untuk kueri pencarian
        query_embeddings = list(self.embedding_model.embed([query]))
        query_vector = query_embeddings[0].tolist()

        conn = get_db_connection()
        cur = conn.cursor()

        # 2. Query ke PostgreSQL + pgvector menggunakan Cosine Distance (<->)
        # Filter berdasarkan daftar kitab yang dipilih pengguna
        sql = """
            SELECT id, kitab_judul, bab, halaman_awal, halaman_akhir, teks_asli,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM chunks
            WHERE kitab_judul = ANY(%s)
            ORDER BY similarity DESC
            LIMIT %s;
        """
        
        cur.execute(sql, (query_vector, kitab_list, top_k))
        rows = cur.fetchall()
        cur.close()
        conn.close()

        results = []
        for row in rows:
            similarity = float(row[6])
            if similarity >= SIMILARITY_THRESHOLD:
                results.append({
                    "id": row[0],
                    "kitab": row[1],
                    "bab": row[2],
                    "halaman_awal": row[3],
                    "halaman_akhir": row[4],
                    "teks_asli": row[5],
                    "similarity": similarity
                })

        return results

retriever_service = RAGRetriever()
