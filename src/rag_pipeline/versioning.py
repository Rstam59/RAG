import hashlib
import json
from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineVersions:
    pipeline_version: str
    chunker_id: str
    embedder_id: str
    ocr_id: str  # "none" or "tesseract_v1"


def corpus_version(doc_hashes: list[str], versions: PipelineVersions) -> str:
    """
    corpus_version should change if:
      - docs change
      - chunker/embedder/ocr changes (because resulting index semantics change)
    """
    payload = {
        "docs": sorted(doc_hashes),
        "pipeline_version": versions.pipeline_version,
        "chunker_id": versions.chunker_id,
        "embedder_id": versions.embedder_id,
        "ocr_id": versions.ocr_id,
    }
    s = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()
