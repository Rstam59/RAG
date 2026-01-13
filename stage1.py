import os
import glob
import re
import hashlib
import uuid

from dotenv import load_dotenv
from pypdf import PdfReader
from pypdf.errors import DependencyError, PdfReadError
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

load_dotenv()

# -----------------------
# Config (Stage 1 = simple)
# -----------------------
PDF_DIR = os.getenv("PDF_DIR", "data/raw/pdfs")
INGESTED_CACHE = os.getenv("INGESTED_CACHE", "data/ingested_files.txt")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "rag_collection")

CHUNK_CHARS = int(os.getenv("CHUNK_CHARS", "1200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "32"))
UPSERT_BATCH_SIZE = int(os.getenv("UPSERT_BATCH_SIZE", "64"))

MAX_FILES = int(os.getenv("MAX_FILES", "0"))  # 0 means all

_ws = re.compile(r"\s+")


def clean_text(t: str) -> str:
    t = t.replace("\x00", " ")
    t = _ws.sub(" ", t).strip()
    return t


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stable_id(file_hash: str, chunk_idx: int) -> str:
    # deterministic point id (re-run overwrites same chunks for same file)
    ns = uuid.UUID("12345678-1234-5678-1234-567812345678")
    return str(uuid.uuid5(ns, f"{file_hash}:{chunk_idx}"))


# def read_pdf_text(path: str) -> str:
#     reader = PdfReader(path)
#     parts = []
#     for page in reader.pages:
#         txt = clean_text(page.extract_text() or "")
#         if txt:
#             parts.append(txt)
#     return "\n\n".join(parts)



def read_pdf_text(path: str) -> str:
    """
    Best-effort PDF text extraction:
    - Skips unreadable/encrypted PDFs
    - Skips pages that crash extraction (broken fonts, etc.)
    """
    try:
        reader = PdfReader(path)

        # Handle encryption if present
        if getattr(reader, "is_encrypted", False):
            try:
                reader.decrypt("")  # common case: empty password
            except Exception:
                return ""

        parts = []
        for _, page in enumerate(reader.pages, start=1):
            try:
                raw = page.extract_text() or ""
            except Exception:
                # broken fonts / weird metadata / parser edge cases -> skip this page
                continue

            txt = clean_text(raw)
            if txt:
                parts.append(txt)

        return "\n\n".join(parts)

    except (DependencyError, PdfReadError):
        return ""
    except Exception:
        return ""


def load_ingested_cache() -> set[str]:
    if not os.path.exists(INGESTED_CACHE):
        return set()
    with open(INGESTED_CACHE, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def mark_ingested(file_hash: str) -> None:
    # ensure parent dir exists
    parent = os.path.dirname(INGESTED_CACHE)
    if parent:
        os.makedirs(parent, exist_ok=True)

    # avoid duplicating entries
    with open(INGESTED_CACHE, "a", encoding="utf-8") as f:
        f.write(file_hash + "\n")


def chunk_text(text: str, chunk_chars: int, overlap: int) -> list[str]:
    if not text:
        return []
    step = max(1, chunk_chars - overlap)
    chunks = []
    for start in range(0, len(text), step):
        end = min(len(text), start + chunk_chars)
        c = text[start:end].strip()
        if c:
            chunks.append(c)
        if end >= len(text):
            break
    return chunks


def ensure_collection(client: QdrantClient, collection: str, dim: int) -> None:
    existing = {c.name for c in client.get_collections().collections}
    if collection not in existing:
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )


def main() -> None:
    ingested = load_ingested_cache()
    print(f"[stage1] Already ingested files: {len(ingested)}")

    pdfs = sorted(glob.glob(os.path.join(PDF_DIR, "*.pdf")))
    if not pdfs:
        raise SystemExit(f"No PDFs found in: {PDF_DIR}")

    if MAX_FILES > 0:
        pdfs = pdfs[:MAX_FILES]
        print(f"[stage1] Limiting to first {MAX_FILES} PDFs")

    print(f"[stage1] Found {len(pdfs)} PDFs in {PDF_DIR}")
    print(f"[stage1] Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    dim = model.get_sentence_embedding_dimension()
    print(f"[stage1] Embedding dim: {dim}")

    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    ensure_collection(client, QDRANT_COLLECTION, dim)

    total = 0

    for path in pdfs:
        file_name = os.path.basename(path)
        file_hash = sha256_file(path)

        if file_hash in ingested:
            print(f"[stage1] Skipping already ingested: {file_name}")
            continue

        print(f"\n[stage1] Reading: {file_name}")
        text = read_pdf_text(path)
        if not text:
            print("[stage1]  -> No extractable text / encrypted / unreadable. Skipping.")
            continue

        chunks = chunk_text(text, CHUNK_CHARS, CHUNK_OVERLAP)
        print(f"[stage1]  -> {len(chunks)} chunks")

        vectors = model.encode(
            chunks,
            batch_size=EMBED_BATCH_SIZE,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

        points = []
        for idx, (chunk, vec) in enumerate(zip(chunks, vectors)):
            points.append(
                PointStruct(
                    id=stable_id(file_hash, idx),
                    vector=vec.tolist(),
                    payload={
                        "source_path": path,
                        "file_name": file_name,
                        "file_hash": file_hash,
                        "chunk_index": idx,
                        "text": chunk,
                    },
                )
            )

        # Upsert in smaller batches to avoid timeouts
        try:
            for i in range(0, len(points), UPSERT_BATCH_SIZE):
                batch = points[i : i + UPSERT_BATCH_SIZE]
                client.upsert(collection_name=QDRANT_COLLECTION, points=batch)
        except Exception as e:
            print(f"[stage1]  -> Upsert failed for {file_name}: {e}. Skipping.")
            continue

        # Mark ingested only after the entire file is successfully indexed
        mark_ingested(file_hash)
        ingested.add(file_hash)

        print(f"[stage1]  -> Upserted {len(points)} vectors to {QDRANT_COLLECTION}")
        total += len(points)

    print(f"\n[stage1] ✅ Done. Total vectors upserted this run: {total}")
    print(f"[stage1] Qdrant running at: http://{QDRANT_HOST}:{QDRANT_PORT}")


if __name__ == "__main__":
    main()
