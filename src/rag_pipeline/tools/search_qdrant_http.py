import os 
import httpx
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION = os.getenv("QDRANT_COLLECTION", "rag_collection")
MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


def qdrant_search_http(vector, limit: int = 5):
    url = f'http://{QDRANT_HOST}:{QDRANT_PORT}/collections/{COLLECTION}/points/search'
    payload = {'vector': vector, 'limit': limit, 'with_payload': True}
    with httpx.Client(timeout = 60.0) as c:
        r = c.post(url, json = payload)
        r.raise_for_status()
        return r.json().get('result', [])
    

def main():
    model = SentenceTransformer(MODEL)

    while True:
        q = input('\nAsk (or q): ').strip()
        if q.lower() == 'q':
            break

        q_vec = model.encode(q, normalize_embeddings= True).tolist()
        hits = qdrant_search_http(q_vec, limit = 5)

        print("\n--- Top results ---")
        for i, h in enumerate(hits, 1):
            payload = h.get('payload') or {}
            print(f"\n#{i} score={h.get('score'):.4f}")
            print("file:", payload.get("file_name"))
            print((payload.get("text") or "")[:500], "...")


if __name__ == "__main__":
    main()

