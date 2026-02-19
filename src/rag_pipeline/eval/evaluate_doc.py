import json
import os
import time
from typing import Dict, List, Optional

import yaml
from sentence_transformers import SentenceTransformer

from rag_pipeline.settings import InfraSettings, load_pipeline_config
from rag_pipeline.http.qdrant_http import QdrantHttp
from rag_pipeline.manifest import write_json
from rag_pipeline.versioning import config_fingerprint


def read_jsonl(path: str) -> List[dict]:
    out: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def precision_at_k(retrieved: List[str], gold: List[str], k: int) -> float:
    if k <= 0 or not retrieved:
        return 0.0
    topk = retrieved[:k]
    g = set(gold)
    rel = sum(1 for d in topk if d in g)
    return rel / min(k, len(topk))


def recall_at_k(retrieved: List[str], gold: List[str], k: int) -> float:
    if k <= 0 or not gold or not retrieved:
        return 0.0
    topk = retrieved[:k]
    g = set(gold)
    rel = sum(1 for d in topk if d in g)
    return rel / len(gold)


def average_precision_at_k(retrieved: List[str], gold: List[str], k: int) -> float:
    if k <= 0 or not gold or not retrieved:
        return 0.0

    g = set(gold)
    topk = retrieved[:k]

    num_relevant = 0
    ap_sum = 0.0
    for i, doc_id in enumerate(topk, start=1):
        if doc_id in g:
            num_relevant += 1
            ap_sum += num_relevant / i  # precision@i

    denom = min(len(gold), k)
    return ap_sum / denom if denom > 0 else 0.0


def reciprocal_rank_at_k(retrieved: List[str], gold: List[str], k: int) -> float:
    if k <= 0 or not gold or not retrieved:
        return 0.0
    g = set(gold)
    topk = retrieved[:k]
    for i, doc_id in enumerate(topk, start=1):
        if doc_id in g:
            return 1.0 / i
    return 0.0


def pctl(sorted_vals: List[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = int(p * (len(sorted_vals) - 1))
    return sorted_vals[idx]


def main() -> None:
    infra = InfraSettings()

    # Load pipeline config (so eval uses the exact same semantic config)
    pipe, raw_cfg = load_pipeline_config(infra.pipeline_config)
    cfg_fp = config_fingerprint(raw_cfg)

    # Optional: keep CORPUS_VERSION for backwards compat / tighter snapshot filtering
    corpus_ver = os.getenv("CORPUS_VERSION", "").strip() or None

    # Load embedding settings (same as indexing)
    with open(infra.pipeline_config, "r", encoding="utf-8") as f:
        raw_yaml = yaml.safe_load(f)
    embed_model = raw_yaml["embedding"]["model"]
    normalize = bool(raw_yaml["embedding"].get("normalize", True))

    # Load eval data
    queries_path = os.path.join(infra.eval_dir, "queries.jsonl")
    labels_path = os.path.join(infra.eval_dir, "labels.jsonl")
    queries = read_jsonl(queries_path)
    labels_rows = read_jsonl(labels_path)

    labels: Dict[str, List[str]] = {r["qid"]: list(r["gold_doc_ids"]) for r in labels_rows}

    model = SentenceTransformer(embed_model)
    qdrant = QdrantHttp(infra.qdrant_url, timeout_s=20.0, max_retries=3)

    # ✅ Filter by dataset_id + config_fingerprint everywhere
    must_filters: List[dict] = [
        {"key": "dataset_id", "match": {"value": pipe.dataset_id}},
        {"key": "config_fingerprint", "match": {"value": cfg_fp}},
    ]
    # Optional extra filter if you still store corpus_version/snapshot_id in payloads
    if corpus_ver:
        must_filters.append({"key": "corpus_version", "match": {"value": corpus_ver}})

    filter_payload = {"must": must_filters}

    ks = [1, 3, 5, 10]
    max_k = max(ks)

    sums = {
        "precision": {k: 0.0 for k in ks},
        "recall": {k: 0.0 for k in ks},
        "map": {k: 0.0 for k in ks},
        "mrr": {k: 0.0 for k in ks},
    }

    latencies: List[float] = []
    n = 0
    skipped_no_gold = 0

    for q in queries:
        qid = q["qid"]
        query = q["query"]

        gold = labels.get(qid, [])
        if not gold:
            skipped_no_gold += 1
            continue

        t0 = time.time()
        q_vec = model.encode(query, normalize_embeddings=normalize).tolist()

        hits = qdrant.search(
            collection=infra.qdrant_collection,
            vector=q_vec,
            limit=max_k,
            with_payload=True,
            filter_payload=filter_payload,
        )
        latencies.append(time.time() - t0)

        # doc-level: unique doc_ids in rank order
        retrieved: List[str] = []
        seen = set()
        for h in hits:
            payload = h.get("payload") or {}
            doc_id = payload.get("doc_id")
            if not doc_id or doc_id in seen:
                continue
            seen.add(doc_id)
            retrieved.append(doc_id)

        for k in ks:
            sums["precision"][k] += precision_at_k(retrieved, gold, k)
            sums["recall"][k] += recall_at_k(retrieved, gold, k)
            sums["map"][k] += average_precision_at_k(retrieved, gold, k)
            sums["mrr"][k] += reciprocal_rank_at_k(retrieved, gold, k)

        n += 1

    qdrant.close()

    if n == 0:
        raise SystemExit(
            "No evaluable queries. Check queries.jsonl / labels.jsonl, or your Qdrant payload filters."
        )

    latencies.sort()

    report = {
        "dataset_id": pipe.dataset_id,
        "config_fingerprint": cfg_fp,
        "corpus_version": corpus_ver,  # may be None
        "collection": infra.qdrant_collection,
        "embed_model": embed_model,
        "num_queries": n,
        "skipped_no_gold": skipped_no_gold,
        "metrics": {
            **{f"precision@{k}": round(sums["precision"][k] / n, 4) for k in ks},
            **{f"recall@{k}": round(sums["recall"][k] / n, 4) for k in ks},
            **{f"map@{k}": round(sums["map"][k] / n, 4) for k in ks},
            **{f"mrr@{k}": round(sums["mrr"][k] / n, 4) for k in ks},
        },
        "latency_seconds": {
            "p50": round(pctl(latencies, 0.50), 4),
            "p95": round(pctl(latencies, 0.95), 4),
            "p99": round(pctl(latencies, 0.99), 4),
        },
    }

    tag = (corpus_ver[:8] if corpus_ver else cfg_fp[:8])
    out = os.path.join(infra.runs_dir, f"eval_{tag}.json")
    write_json(out, report)

    print(json.dumps(report, indent=2))
    print(f"\n[eval] wrote: {out}")


if __name__ == "__main__":
    main()
