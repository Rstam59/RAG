import hashlib
import json
from typing import Any, Dict, Iterable


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stable_doc_id(file_hash: str) -> str:
    # doc_id is the content-hash of PDF bytes
    return file_hash


def config_fingerprint(cfg: Dict[str, Any]) -> str:
    blob = json.dumps(cfg, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return sha256_bytes(blob)


def corpus_version(doc_ids: Iterable[str], cfg_fp: str) -> str:
    h = hashlib.sha256()
    h.update(cfg_fp.encode("utf-8"))
    for d in sorted(doc_ids):
        h.update(d.encode("utf-8"))
    return h.hexdigest()
