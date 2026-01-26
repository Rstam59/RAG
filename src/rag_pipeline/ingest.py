import glob
import logging
import os
import time
from typing import Any, Dict, List, Tuple

from rag_pipeline.logging_setup import setup_logging, log
from rag_pipeline.manifest import utc_now_iso, write_run_manifest
from rag_pipeline.settings import InfraSettings, load_pipeline_config
from rag_pipeline.versioning import sha256_file, stable_doc_id, config_fingerprint, corpus_version

from rag_pipeline.loaders.pdf_loader import read_pdf_text_best_effort
from rag_pipeline.chunking.chunker import chunk_text
from rag_pipeline.embedding.embedder import Embedder
from rag_pipeline.indexing.qdrant_index import QdrantIndex


def _cache_path(ingested_dir: str, corpus_ver: str) -> str:
    os.makedirs(ingested_dir, exist_ok=True)
    return os.path.join(ingested_dir, f"ingested_{corpus_ver}.txt")


def _load_cache(path: str) -> set[str]:
    if not os.path.exists(path):
        return set()
    with open(path, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def _append_cache(path: str, doc_id: str) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(doc_id + "\n")


def main() -> None:
    infra = InfraSettings()
    setup_logging(infra.log_level)
    logger = logging.getLogger("rag_pipeline.ingest")

    pipe, raw_cfg = load_pipeline_config(infra.pipeline_config)
    cfg_fp = config_fingerprint(raw_cfg)

    pdfs = sorted(glob.glob(os.path.join(infra.pdf_dir, "*.pdf")))
    if pipe.only_match:
        pdfs = [p for p in pdfs if pipe.only_match.lower() in os.path.basename(p).lower()]
    if pipe.max_files > 0:
        pdfs = pdfs[: pipe.max_files]
    if not pdfs:
        raise SystemExit(f"No PDFs found after filtering in {infra.pdf_dir}")

    # Compute doc_ids up front -> stable corpus_version
    docs: List[Tuple[str, str]] = []
    for p in pdfs:
        file_hash = sha256_file(p)
        docs.append((p, stable_doc_id(file_hash)))

    corpus_ver = corpus_version((d for _, d in docs), cfg_fp)
    cache_path = _cache_path(infra.ingested_dir, corpus_ver)
    ingested = _load_cache(cache_path)

    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    started = time.time()

    log(logger, "stage3_start",
        env=infra.env,
        run_id=run_id,
        pdf_dir=infra.pdf_dir,
        pdf_count=len(docs),
        qdrant_url=infra.qdrant_url,
        collection=infra.qdrant_collection,
        pipeline_version=pipe.pipeline_version,
        config_fingerprint=cfg_fp,
        corpus_version=corpus_ver,
        cache_path=cache_path,
        already_ingested=len(ingested),
    )

    embedder = Embedder(pipe.embed_model, pipe.embed_batch_size, pipe.embed_normalize)
    index = QdrantIndex(infra.qdrant_url, infra.qdrant_collection)
    index.ensure_collection(embedder.dim)

    docs_indexed = 0
    docs_skipped = 0
    vectors_upserted = 0
    failures: List[Dict[str, Any]] = []

    payload_meta = {
        "corpus_version": corpus_ver,
        "pipeline_version": pipe.pipeline_version,
        "config_fingerprint": cfg_fp,
        "cleaner_version": pipe.cleaner_version,
        "chunker_version": pipe.chunker_version,
    }

    for path, doc_id in docs:
        file_name = os.path.basename(path)

        if doc_id in ingested:
            log(logger, "skip_cached", file=file_name, doc_id=doc_id)
            continue

        t0 = time.time()
        text = read_pdf_text_best_effort(path)
        if not text:
            docs_skipped += 1
            failures.append({"file": file_name, "doc_id": doc_id, "reason": "unreadable_or_no_text"})
            log(logger, "skip_unreadable", file=file_name, doc_id=doc_id)
            continue

        chunks = chunk_text(text, pipe.chunk_chars, pipe.chunk_overlap, pipe.max_chunks_per_doc)
        if not chunks:
            docs_skipped += 1
            failures.append({"file": file_name, "doc_id": doc_id, "reason": "zero_chunks"})
            log(logger, "skip_zero_chunks", file=file_name, doc_id=doc_id)
            continue

        t_embed0 = time.time()
        vectors = embedder.encode(chunks)
        embed_s = time.time() - t_embed0

        points = index.make_points(
            doc_id=doc_id,
            file_name=file_name,
            source_path=path,
            chunks=chunks,
            vectors=vectors,
            payload_meta=payload_meta,
        )

        try:
            t_up0 = time.time()
            index.upsert_batched(points, pipe.upsert_batch_size)
            upsert_s = time.time() - t_up0
        except Exception as e:
            docs_skipped += 1
            failures.append({"file": file_name, "doc_id": doc_id, "reason": f"upsert_failed: {e}"})
            log(logger, "upsert_failed", file=file_name, doc_id=doc_id, error=str(e))
            continue

        _append_cache(cache_path, doc_id)
        ingested.add(doc_id)

        docs_indexed += 1
        vectors_upserted += len(points)

        log(logger, "doc_indexed",
            file=file_name,
            doc_id=doc_id,
            chunks=len(chunks),
            vectors=len(points),
            embed_s=round(embed_s, 3),
            upsert_s=round(upsert_s, 3),
            total_s=round(time.time() - t0, 3),
        )

    elapsed = round(time.time() - started, 2)
    manifest = {
        "run_id": run_id,
        "created_at_utc": utc_now_iso(),
        "env": infra.env,

        "qdrant_url": infra.qdrant_url,
        "collection": infra.qdrant_collection,

        "pipeline_version": pipe.pipeline_version,
        "config_fingerprint": cfg_fp,
        "corpus_version": corpus_ver,

        "pdf_dir": infra.pdf_dir,
        "pdf_count": len(docs),

        "docs_indexed": docs_indexed,
        "docs_skipped": docs_skipped,
        "vectors_upserted": vectors_upserted,
        "elapsed_seconds": elapsed,

        "cache_path": cache_path,
        "failures": failures[:200],
    }

    out = write_run_manifest(infra.runs_dir, run_id, manifest)

    log(logger, "stage3_done",
        run_id=run_id,
        docs_indexed=docs_indexed,
        docs_skipped=docs_skipped,
        vectors_upserted=vectors_upserted,
        elapsed_s=elapsed,
        manifest_path=out,
        corpus_version=corpus_ver,
    )

    print(f"\n[ingest] ✅ Done run_id={run_id}")
    print(f"[ingest] corpus_version={corpus_ver}")
    print(f"[ingest] manifest={out}")


if __name__ == "__main__":
    main()
