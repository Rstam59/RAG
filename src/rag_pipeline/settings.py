import os
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import yaml
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class InfraSettings:
    env: str = os.getenv("ENV", "dev")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333").rstrip("/")
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "rag_collection")

    pdf_dir: str = os.getenv("PDF_DIR", "data/raw/pdfs")
    runs_dir: str = os.getenv("RUNS_DIR", "data/runs")
    ingested_dir: str = os.getenv("INGESTED_DIR", "data/ingested")

    pipeline_config: str = os.getenv("PIPELINE_CONFIG", "configs/pipeline.yaml")
    eval_dir: str = os.getenv("EVAL_DIR", "data/eval")


@dataclass(frozen=True)
class PipelineConfig:
    pipeline_version: str

    embed_model: str
    embed_batch_size: int
    embed_normalize: bool

    chunk_chars: int
    chunk_overlap: int
    max_chunks_per_doc: int

    max_files: int
    only_match: str
    upsert_batch_size: int

    cleaner_version: str
    chunker_version: str


def load_pipeline_config(path: str) -> Tuple[PipelineConfig, Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f)

    cfg = PipelineConfig(
        pipeline_version=str(raw["pipeline_version"]),

        embed_model=str(raw["embedding"]["model"]),
        embed_batch_size=int(raw["embedding"].get("batch_size", 32)),
        embed_normalize=bool(raw["embedding"].get("normalize", True)),

        chunk_chars=int(raw["chunking"]["chunk_chars"]),
        chunk_overlap=int(raw["chunking"]["overlap"]),
        max_chunks_per_doc=int(raw["chunking"].get("max_chunks_per_doc", 0)),

        max_files=int(raw["ingest"].get("max_files", 0)),
        only_match=str(raw["ingest"].get("only_match", "")).strip(),
        upsert_batch_size=int(raw["ingest"].get("upsert_batch_size", 64)),

        cleaner_version=str(raw.get("cleaner_version", "clean_v1")),
        chunker_version=str(raw.get("chunker_version", "chars_v1")),
    )

    return cfg, raw
