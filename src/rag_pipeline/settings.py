import os 
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Settings:
    pdf_dir: str = os.getenv('PDF_DIR', 'data/raw/pdfs')
    ingested_cache: str = os.getenv("INGESTED_CACHE", 'data/ingested_files.txt')
    runs_dir: str = os.getenv("RUNS_DIR", 'data/runs')


    embedding_model: str = os.getenv("EMBEDDING_MODEL", 'sentence-transformers/all-MiniLM-L6-v2')
    embed_batch_size: int = int(os.getenv("EMBED_BATCH_SIZE", 32))


    qdrant_host: str = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port: int = int(os.getenv("QDRANT_PORT", "6333"))
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "rag_collection")
    upsert_batch_size: int = int(os.getenv("UPSERT_BATCH_SIZE", "64"))

    chunk_chars: int = int(os.getenv("CHUNK_CHARS", "1200"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))

    # Stage-2 knobs
    max_files: int = int(os.getenv("MAX_FILES", "0"))  # 0 = all
    only_match: str = os.getenv("ONLY_MATCH", "").strip()  # substring filter
    max_chunks_per_doc: int = int(os.getenv("MAX_CHUNKS_PER_DOC", "0"))