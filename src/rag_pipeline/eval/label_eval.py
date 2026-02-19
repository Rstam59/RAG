import json
import os
from typing import Dict, List, Optional, Set

import yaml
from sentence_transformers import SentenceTransformer

from rag_pipeline.settings import InfraSettings, load_pipeline_config
from rag_pipeline.http.qdrant_http import QdrantHttp
from rag_pipeline.versioning import config_fingerprint


def read_jsonl(path: str) -> List[dict]:
    if not os.path.exists(path):
        return []
    out: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write_jsonl_atomic(path: str, rows: List[dict]) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def load_labels_map(path: str) -> Dict[str, List[str]]:
    rows = read_jsonl(path)
    mp: Dict[str, List[str]] = {}
    for r in rows:
        qid = r.get("qid")
        gold = r.get("gold_doc_ids")
        if isinstance(qid, str) and isinstance(gold, list):
            mp[qid] = [str(x) for x in gold]
    return mp


def parse_indices(s: str, max_n: int) -> List[int]:
    """
    Accept: "1,3,5" or "1 3 5"
    Returns 0-based indices.
    """
    s = s.replace(",", " ").strip()
    if not s:
        return []
    out: List[int] = []
    for tok in s.split():
        if not tok.isdigit():
            continue
        i = int(tok)
        if 1 <= i <= max_n:
            out.append(i - 1)
    return sorted(set(out))


def dedup_doc_hits(hits: List[dict]) -> List[dict]:
    """
    Qdrant returns chunk-level points. We want doc-level candidates:
    keep first occurrence of each doc_id (highest scored chunk).
    """
    out: List[dict] = []
    seen: Set[str] = set()
    for h in hits:
        payload = h.get("payload") or {}
        doc_id = payload.get("doc_id")
        if not doc_id or doc_id in seen:
            continue
        seen.add(doc_id)
        out.append(h)
    return out


def build_filter(
    *,
    dataset_id: str,
    cfg_fp: str,
    corpus_ver: Optional[str] = None,
) -> dict:
    must: List[dict] = [
        {"key": "dataset_id", "match": {"value": dataset_id}},
        {"key": "config_fingerprint", "match": {"value": cfg_fp}},
    ]
    if corpus_ver:
        must.append({"key": "corpus_version", "match": {"value": corpus_ver}})
    return {"must": must}


def main() -> None:
    infra = InfraSettings()

    # Load embed config (must match indexing)
    with open(infra.pipeline_config, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    embed_model = raw["embedding"]["model"]
    normalize = bool(raw["embedding"].get("normalize", True))

    # Load pipeline config to get dataset_id + config fingerprint
    pipe, raw_cfg = load_pipeline_config(infra.pipeline_config)
    cfg_fp = config_fingerprint(raw_cfg)

    # Optional: keep CORPUS_VERSION as an extra narrowing filter
    corpus_ver = os.getenv("CORPUS_VERSION", "").strip() or None

    # I/O files
    queries_path = os.path.join(infra.eval_dir, "queries.jsonl")
    labels_path = os.path.join(infra.eval_dir, "labels.jsonl")

    queries = read_jsonl(queries_path)
    if not queries:
        raise SystemExit(f"No queries found at: {queries_path}")

    labels_map = load_labels_map(labels_path)

    # Runtime knobs
    top_k = int(os.getenv("LABEL_TOPK", "10"))
    show_chars = int(os.getenv("LABEL_SNIPPET_CHARS", "350"))

    # Services
    model = SentenceTransformer(embed_model)
    qdrant = QdrantHttp(infra.qdrant_url, timeout_s=20.0, max_retries=3)

    # ✅ Filter by dataset_id + config_fingerprint (and optionally corpus_version)
    filter_payload = build_filter(dataset_id=pipe.dataset_id, cfg_fp=cfg_fp, corpus_ver=corpus_ver)

    print("\n--- RAG Labeling Tool (doc-level) ---")
    print(f"Qdrant:      {infra.qdrant_url}")
    print(f"Collection:  {infra.qdrant_collection}")
    print(f"Embed model: {embed_model}")
    print(f"Dataset:     {pipe.dataset_id}")
    print(f"Cfg FP:      {cfg_fp[:12]}...")
    if corpus_ver:
        print(f"Corpus:      {corpus_ver[:12]}...")
    else:
        print("Corpus:      (none)")
    print(f"TopK:        {top_k}")
    print(f"Queries:     {len(queries)}")
    print(f"Labels:      {len(labels_map)} already labeled")
    print("-------------------------------------")

    for qi, q in enumerate(queries, start=1):
        qid = q.get("qid")
        query = q.get("query")
        if not isinstance(qid, str) or not isinstance(query, str):
            continue

        already = labels_map.get(qid)

        print(f"\n[{qi}/{len(queries)}] qid={qid}")
        print(f"Q: {query}")

        if already is not None:
            print(f"(Already labeled: {len(already)} doc_ids)  Enter 'e' to edit, or just press Enter to skip.")
            cmd = input("Action [Enter=skip, e=edit]: ").strip().lower()
            if cmd != "e":
                continue

        # Retrieve candidates
        q_vec = model.encode(query, normalize_embeddings=normalize).tolist()
        hits = qdrant.search(
            collection=infra.qdrant_collection,
            vector=q_vec,
            limit=max(top_k * 3, top_k),  # overfetch to dedup docs
            with_payload=True,
            filter_payload=filter_payload,
        )
        hits = dedup_doc_hits(hits)[:top_k]

        if not hits:
            print("No hits. Likely causes: wrong filters, old points missing dataset_id/config_fingerprint, or empty collection.")
            cmd = input("Action [Enter=continue, m=manual doc_id]: ").strip().lower()
            if cmd == "m":
                manual = input("Paste doc_id(s) separated by comma: ").strip()
                docs = [x.strip() for x in manual.split(",") if x.strip()]
                if docs:
                    labels_map[qid] = docs
                    write_jsonl_atomic(labels_path, [{"qid": k, "gold_doc_ids": v} for k, v in labels_map.items()])
                    print(f"Saved {len(docs)} manual doc_id(s).")
            continue

        # Display candidates
        print("\nCandidates:")
        for i, h in enumerate(hits, start=1):
            score = float(h.get("score", 0.0) or 0.0)
            payload = h.get("payload") or {}
            file_name = payload.get("file_name", "(unknown)")
            doc_id = payload.get("doc_id", "")
            chunk_idx = payload.get("chunk_index", None)
            text = (payload.get("text") or "").replace("\n", " ").strip()
            snippet = text[:show_chars] + ("..." if len(text) > show_chars else "")
            print(f"\n {i}) score={score:.4f} | file={file_name} | chunk={chunk_idx}")
            print(f"    doc_id={str(doc_id)[:12]}...")
            print(f"    {snippet}")

        print("\nLabeling:")
        print(" - Type indices of relevant docs, e.g. 1,3,5")
        print(" - Enter = skip (no label change)")
        print(" - 'm' = manually paste doc_id(s)")
        print(" - 'q' = quit")
        ans = input("Relevant: ").strip().lower()

        if ans == "q":
            break

        if ans == "":
            continue

        if ans == "m":
            manual = input("Paste doc_id(s) separated by comma: ").strip()
            docs = [x.strip() for x in manual.split(",") if x.strip()]
            if docs:
                labels_map[qid] = docs
                write_jsonl_atomic(labels_path, [{"qid": k, "gold_doc_ids": v} for k, v in labels_map.items()])
                print(f"Saved {len(docs)} manual doc_id(s).")
            continue

        idxs = parse_indices(ans, max_n=len(hits))
        chosen: List[str] = []
        for j in idxs:
            payload = hits[j].get("payload") or {}
            doc_id = payload.get("doc_id")
            if doc_id:
                chosen.append(str(doc_id))

        if not chosen:
            print("No valid selections parsed. Skipping.")
            continue

        labels_map[qid] = chosen
        write_jsonl_atomic(labels_path, [{"qid": k, "gold_doc_ids": v} for k, v in labels_map.items()])
        print(f"✅ Saved labels for {qid}: {len(chosen)} doc_id(s)")

    qdrant.close()
    print(f"\nDone. labels.jsonl at: {labels_path}")


if __name__ == "__main__":
    main()
