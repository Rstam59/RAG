import os 
import numpy as np 
from sentence_transformers import SentenceTransformer
from google import genai
from rank_bm25 import BM25Okapi

api_key = os.getenv("GOOGLE_API_KEY1")

if not api_key:
    raise ValueError("NO API KEY found. Run export GOOGLE_API_KEY1='your_api_key'")

client = genai.Client(api_key= api_key)

MODEL_NAME = "gemini-2.5-flash" 

docs = [
    "RAG stands for Retrieval-Augmented Generation: retrieve relevant context then generate an answer.",
    "SentenceTransformers creates dense embeddings for sentences and paragraphs for semantic search.",
    "Cosine similarity measures the angle between vectors; higher means more similar.",
    "Chunking text (e.g., 200-500 tokens) helps retrieval and reduces irrelevant context.",
    "Gemini is a family of Google models usable via the google-generativeai SDK.",
]

embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
doc_emb = embedder.encode(docs, normalize_embeddings=True)

tokenized_corpus = [d.lower().split() for d in docs]
bm25 = BM25Okapi(tokenized_corpus)

def semantic_scores(query: str, top_k: int = 1):
    q = embedder.encode([query], normalize_embeddings=True)[0]
    return doc_emb @ q  # cosine because normalized


def bm25_scores(query: str) -> np.ndarray:
    return np.array(bm25.get_scores(query), dtype=float)


def rank_from_scores(scores: np.ndarray):
    return np.argsort(-scores).tolist()


def rrf_fuse(rank_lists, k: int = 60):
    scores  = {}
    for ranks in rank_lists:
        for r, idx in enumerate(ranks, start=1):  # rank starts at 1
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + r)
    return scores


def retrieve_hybrid(query: str, top_k: int = 3, rrf_k: int = 60):
    sem = semantic_scores(query)
    kw = bm25_scores(query)

    sem_rank = rank_from_scores(sem)
    kw_rank = rank_from_scores(kw)

    rrf = rrf_fuse([sem_rank, kw_rank], k=rrf_k)

    # final ranking by fused score
    fused_rank = sorted(rrf.items(), key=lambda x: x[1], reverse=True)
    fused_rank = fused_rank[:top_k]

    out = []

    for idx, rrf_score in fused_rank:
        out.append({
            "idx": idx,
            "text": docs[idx],
            "semantic_score": float(sem[idx]),
            "bm25_score": float(kw[idx]),
            "rrf_score": float(rrf_score),
        })

    return out

def run_rag(query: str, top_k: int = 3) -> str:
    hits = retrieve_hybrid(query, top_k=top_k)

    context = "\n\n".join(
        f"[{i+1}] {h['text']}"
        for i, h in enumerate(hits)
    )

    prompt = f"""Use ONLY the context to answer.
If the question uses different casing or refers to an acronym that is clearly defined in the context, treat them as the same term.
If the answer is not in the context, say: "I don't know from the provided context."

Context:
{context}

Question: {query}

Answer:"""

    resp = client.models.generate_content(model=MODEL_NAME, contents=prompt)
    return resp.text


if __name__ == "__main__":
    q = "What is cosine similarity?"
    print(run_rag(q, top_k=3))



