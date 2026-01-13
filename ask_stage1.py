import httpx
from sentence_transformers import SentenceTransformer

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION = "rag_collection"

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def qdrant_search(query_vector, limit=5):
    url = f"http://{QDRANT_HOST}:{QDRANT_PORT}/collections/{COLLECTION}/points/search"
    payload = {
        "vector": query_vector,
        "limit": limit,
        "with_payload": True,
    }

    with httpx.Client(timeout=60.0) as client:
        r = client.post(url, json=payload)
        r.raise_for_status()
        data = r.json()
        return data.get("result", [])


def main():
    while True:
        q = input("\nAsk something (or 'q' to quit): ").strip()
        if q.lower() == "q":
            break

        q_vec = model.encode(q, normalize_embeddings=True).tolist()

        try:
            hits = qdrant_search(q_vec, limit=5)
        except Exception as e:
            print(f"\n[error] Qdrant search failed: {e}")
            print("Check that Qdrant is running and the collection exists.")
            continue

        print("\n--- Top results ---")
        for i, h in enumerate(hits, 1):
            score = h.get("score")
            payload = h.get("payload") or {}
            file_name = payload.get("file_name")
            text = payload.get("text", "")
            print(f"\n#{i}  score={score:.4f}")
            print(f"file: {file_name}")
            print(text[:500], "...")


if __name__ == "__main__":
    main()
