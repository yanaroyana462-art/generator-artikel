import os
import json
from typing import List, Dict, Any

try:
    from google import genai
    from google.genai import types
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

class ArticleGenerator:
    def __init__(self):
        if HAS_GENAI and GEMINI_API_KEY:
            self.client = genai.Client(api_key=GEMINI_API_KEY)
        else:
            self.client = None

    def build_prompt(self, topik: str, target_pembaca: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
        context_str = ""
        for index, c in enumerate(retrieved_chunks, start=1):
            context_str += f"\n--- [Konteks #{index} | ID Chunk: {c['id']}] ---\n"
            context_str += f"Kitab: {c['kitab']} | Bab: {c['bab']} | Halaman: {c['halaman_awal']}-{c['halaman_akhir']}\n"
            context_str += f"Teks Asli: \"{c['teks_asli']}\"\n"

        prompt = f"""
Anda adalah seorang Redaktur Senior dan Penulis Ideologis untuk KaffahMedia.
Tugas Anda adalah menulis artikel mendalam, lugas, dan terstruktur berdasarkan referensi kitab yang diberikan.

TOPIK ARTIKEL: {topik}
TARGET PEMBACA: {target_pembaca}

KONTEKS KITAB REFERENSI:
{context_str if context_str else "Tidak ada referensi spesifik yang ditemukan."}

INSTRUKSI PENULISAN:
1. Gunakan gaya bahasa KaffahMedia: lugas, tajam, ideologis, dan sistematis.
2. Setiap kali merujuk atau mengutip teks dari Konteks Kitab, Anda WAJIB menyertakan `sumber_id` (ID Chunk yang sesuai).
3. Untuk kutipan langsung, gunakan teks eksak yang ada di referensi agar lolos verifikasi string-matching.
4. Jangan membuat informasi atau klaim tanpa dukungan konteks jika konteks tidak mencukupi.

Gunakan format JSON persis seperti berikut:
{{
  "judul": "Judul Artikel yang Menarik dan Tajam",
  "ringkasan": "Ringkasan eksekutif 2-3 kalimat",
  "bagian": [
    {{
      "subjudul": "Subjudul Bagian 1",
      "paragraf": [
        {{
          "teks": "Teks paragraf narasi...",
          "kutipan": [
            {{
              "teks": "Potongan kutipan langsung dari kitab",
              "sumber_id": 1
            }}
          ]
        }}
      ]
    }}
  ]
}}
"""
        return prompt

    def generate(self, topik: str, target_pembaca: str, retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not self.client:
            return {
                "judul": f"Analisis Ideologis: {topik}",
                "ringkasan": "Artikel simulasi (Mode Offline / Fallback).",
                "bagian": [
                    {
                        "subjudul": "Pendahuluan",
                        "paragraf": [
                            {
                                "teks": f"Pembahasan mengenai {topik} memerlukan pemahaman mendasar terhadap fakta dan realitas.",
                                "kutipan": [
                                    {
                                        "teks": retrieved_chunks[0]["teks_asli"] if retrieved_chunks else "Akal manusia adalah alat",
                                        "sumber_id": retrieved_chunks[0]["id"] if retrieved_chunks else 1
                                    }
                                ]
                            }
                        ]
                    }
                ]
            }

        prompt = self.build_prompt(topik, target_pembaca, retrieved_chunks)
        
        response = self.client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.2,
            )
        )
        
        try:
            return json.loads(response.text)
        except Exception:
            return {"error": "Gagal memproses JSON dari LLM", "raw_response": response.text}

generator_service = ArticleGenerator()
