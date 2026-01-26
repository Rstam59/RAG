import os
import yaml
from sentence_transformers import SentenceTransformer

from rag_pipeline.settings import InfraSettings
from rag_pipeline.http.qdrant_http import QdrantHttp


def main() -> None:
    infra = InfraSettings()

    corpus_ver = os.getenv("CORPUS_VERSION", "").strip() or None

    with open(infra.pipeline_config, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    embed_model = raw["embedding"]["model"]
    normalize = bool(raw["embedding"].get("normalize", True))

    model = SentenceTransformer(embed_model)
    qdrant = QdrantHttp(infra.qdrant_url, timeout_s=20.0, max_retries=3)

    filter_payload = None
    if corpus_ver:
        filter_payload = {"must": [{"key": "corpus_version", "match": {"value": corpus_ver}}]}

    print(f"\n[search] Qdrant={infra.qdrant_url} collection={infra.qdrant_collection}")
    print(f"[search] embed_model={embed_model}")
    print(f"[search] corpus_filter={corpus_ver if corpus_ver else '(none)'}")

    while True:
        q = input("\nQuery (or 'q'): ").strip()
        if q.lower() == "q":
            break

        q_vec = model.encode(q, normalize_embeddings=normalize).tolist()
        hits = qdrant.search(
            collection=infra.qdrant_collection,
            vector=q_vec,
            limit=8,
            with_payload=True,
            filter_payload=filter_payload,
        )

        print("\n--- Top hits ---")
        for i, h in enumerate(hits, 1):
            payload = h.get("payload") or {}
            print(f"\n#{i} score={h.get('score'):.4f}")
            print("file:", payload.get("file_name"))
            print("doc_id:", (payload.get("doc_id") or "")[:12] + "...")
            print("chunk:", payload.get("chunk_index"))
            text = (payload.get("text") or "").replace("\n", " ")
            print(text[:500], "...")

    qdrant.close()


if __name__ == "__main__":
    main()
