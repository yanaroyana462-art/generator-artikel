import re
from typing import List, Dict, Any

def normalize_text(text: str) -> str:
    """Normalisasi teks untuk pencocokan string: hapus harakat, tanda baca ganda, dan spasi berlebih."""
    if not text:
        return ""
    # Hapus harakat Arab sederhana
    text = re.sub(r'[\u064B-\u0652]', '', text)
    # Hapus karakter non-alphanumeric selain spasi
    text = re.sub(r'[^\w\s]', '', text)
    # Lowercase & gabung spasi
    return " ".join(text.lower().split())

class QuoteVerifier:
    @staticmethod
    def verify_quotes(generated_article: Dict[str, Any], retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Memverifikasi bahwa setiap kutipan langsung dari LLM benar-benar ada di teks chunk sumber.
        Jika tidak cocok, tandai sebagai parafrase.
        """
        chunk_map = {c["id"]: c["teks_asli"] for c in retrieved_chunks}
        
        verified_quotes = []
        failed_quotes = 0
        total_quotes = 0

        sections = generated_article.get("bagian", [])
        for section in sections:
            paragraphs = section.get("paragraf", [])
            for p in paragraphs:
                quotes = p.get("kutipan", [])
                for q in quotes:
                    total_quotes += 1
                    sumber_id = q.get("sumber_id")
                    teks_kutipan = q.get("teks", "")
                    
                    raw_source_text = chunk_map.get(sumber_id, "")
                    norm_quote = normalize_text(teks_kutipan)
                    norm_source = normalize_text(raw_source_text)

                    # String matching check
                    if norm_quote in norm_source and len(norm_quote) > 5:
                        q["status"] = "terverifikasi"
                        verified_quotes.append(q)
                    else:
                        # Jika gagal match persis, paksa status ke parafrase
                        q["status"] = "parafrase"
                        failed_quotes += 1

        return {
            "verified_count": len(verified_quotes),
            "failed_count": failed_quotes,
            "total_quotes": total_quotes,
            "pass_rate": (len(verified_quotes) / total_quotes * 100) if total_quotes > 0 else 100.0
        }

verifier_service = QuoteVerifier()
