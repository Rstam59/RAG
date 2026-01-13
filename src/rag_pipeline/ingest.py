import os
import glob
import json
import time
from datetime import datetime, timezone

from rag_pipeline.settings import Settings
from rag_pipeline.loaders.pdf_loader import read_pdf_text_best_effort
from rag_pipeline.chunking.chunker import chunk_text
from rag_pipeline.embedding.embedder import Embedder
from rag_pipeline.indexing.qdrant_index import QdrantIndex, sha256_file


def load_ingested_cache(path: str) -> set[str]:
    if not os.path.exists(path):
        return set()
    with open(path, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def mark_ingested(path: str, file_hash: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(file_hash + "\n")


def write_run_manifest(runs_dir: str, payload: dict) -> str:
    os.makedirs(runs_dir, exist_ok=True)
    run_id = payload["run_id"]
    out = os.path.join(runs_dir, f"{run_id}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return out


def main():
    s = Settings()

    ingested = load_ingested_cache(s.ingested_cache)
    print(f"[stage2] Already ingested files: {len(ingested)}")

    pdfs = sorted(glob.glob(os.path.join(s.pdf_dir, "*.pdf")))
    if not pdfs:
        raise SystemExit(f"No PDFs found in: {s.pdf_dir}")

    if s.only_match:
        pdfs = [p for p in pdfs if s.only_match.lower() in os.path.basename(p).lower()]
        print(f"[stage2] Filter ONLY_MATCH='{s.only_match}' -> {len(pdfs)} PDFs")

    if s.max_files > 0:
        pdfs = pdfs[: s.max_files]
        print(f"[stage2] Limiting to first {s.max_files} PDFs")

    print(f"[stage2] Found {len(pdfs)} PDFs")

    embedder = Embedder(s.embedding_model, batch_size=s.embed_batch_size)
    index = QdrantIndex(s.qdrant_host, s.qdrant_port, s.qdrant_collection)
    index.ensure_collection(embedder.dim)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    started = time.time()

    run_docs = []
    total_vectors = 0
    skipped = 0

    for path in pdfs:
        file_name = os.path.basename(path)
        file_hash = sha256_file(path)

        if file_hash in ingested:
            print(f"[stage2] Skip cached: {file_name}")
            continue

        print(f"\n[stage2] Reading: {file_name}")
        text = read_pdf_text_best_effort(path)
        if not text:
            print("[stage2]  -> unreadable/encrypted/no-text. Skipping.")
            skipped += 1
            continue

        chunks = chunk_text(
            text,
            chunk_chars=s.chunk_chars,
            overlap=s.chunk_overlap,
            max_chunks=s.max_chunks_per_doc,
        )
        if not chunks:
            print("[stage2]  -> produced 0 chunks. Skipping.")
            skipped += 1
            continue

        print(f"[stage2]  -> {len(chunks)} chunks")
        vectors = embedder.encode(chunks)

        points = index.make_points(
            file_hash=file_hash,
            file_name=file_name,
            source_path=path,
            chunks=chunks,
            vectors=vectors,
        )

        try:
            index.upsert_batched(points, batch_size=s.upsert_batch_size)
        except Exception as e:
            print(f"[stage2]  -> upsert failed: {e}. Skipping file.")
            skipped += 1
            continue

        mark_ingested(s.ingested_cache, file_hash)
        ingested.add(file_hash)

        total_vectors += len(points)
        run_docs.append({"file_name": file_name, "file_hash": file_hash, "chunks": len(points)})

        print(f"[stage2]  -> upserted {len(points)} vectors")

    manifest = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "collection": s.qdrant_collection,
        "embedding_model": s.embedding_model,
        "chunk_chars": s.chunk_chars,
        "chunk_overlap": s.chunk_overlap,
        "max_chunks_per_doc": s.max_chunks_per_doc,
        "total_vectors_upserted": total_vectors,
        "docs_indexed": len(run_docs),
        "docs_skipped": skipped,
        "elapsed_seconds": round(time.time() - started, 2),
        "docs": run_docs,
    }

    out = write_run_manifest(s.runs_dir, manifest)

    print(f"\n[stage2] ✅ Done. vectors upserted this run: {total_vectors}")
    print(f"[stage2] Manifest: {out}")
    print(f"[stage2] Qdrant: http://{s.qdrant_host}:{s.qdrant_port}")


if __name__ == "__main__":
    main()
