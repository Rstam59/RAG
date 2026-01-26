import os
from collections import Counter
from typing import Dict, Optional, Set

from rag_pipeline.settings import InfraSettings
from rag_pipeline.http.qdrant_http import QdrantHttp


def main() -> None:
    infra = InfraSettings()
    qdrant = QdrantHttp(infra.qdrant_url, timeout_s=20.0, max_retries=3)

    corpus_ver = os.getenv("CORPUS_VERSION", "").strip() or None
    max_docs = int(os.getenv("MAX_DOCS", "200"))
    page_size = int(os.getenv("PAGE_SIZE", "128"))

    filter_payload = None
    if corpus_ver:
        filter_payload = {"must": [{"key": "corpus_version", "match": {"value": corpus_ver}}]}

    seen: Set[str] = set()
    names: Dict[str, str] = {}
    counts = Counter()

    offset: Optional[Dict[str, object]] = None

    print(f"[list_docs] collection={infra.qdrant_collection}")
    print(f"[list_docs] corpus_filter={corpus_ver if corpus_ver else '(none)'}")

    while True:
        points, offset = qdrant.scroll(
            collection=infra.qdrant_collection,
            limit=page_size,
            offset=offset,
            filter_payload=filter_payload,
            with_payload=True,
            with_vectors=False,
        )

        if not points:
            break

        for p in points:
            payload = p.get("payload") or {}
            doc_id = payload.get("doc_id")
            if not doc_id:
                continue
            counts[doc_id] += 1
            if doc_id not in seen:
                seen.add(doc_id)
                fn = payload.get("file_name")
                if fn:
                    names[doc_id] = fn
            if len(seen) >= max_docs:
                break

        if len(seen) >= max_docs:
            break

        if offset is None:
            break

    qdrant.close()

    print(f"\n[list_docs] ✅ Found {len(seen)} unique docs (showing up to {max_docs})\n")

    # print sorted by chunk count desc (more informative)
    for i, (doc_id, n_chunks) in enumerate(counts.most_common(max_docs), 1):
        fn = names.get(doc_id, "(unknown)")
        print(f"{i:>3}. {fn} | doc_id={doc_id[:12]}... | chunks_seen={n_chunks}")


if __name__ == "__main__":
    main()
