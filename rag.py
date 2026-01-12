#Windows
# python -m venv myenv
# source myenv\Scripts\activate

#MacOS/Linux
# python3 -m venv myenv
# source myenv/bin/activate


# pip install numpy sentence-transformers google-genai


import os 
import numpy as np 
from sentence_transformers import SentenceTransformer
from google import genai


api_key = os.getenv('GOOGLE_API_KEY1')

if not api_key:
    raise ValueError('API not found. Run export GOOGLE_API_KEY1=your_api_key')


client = genai.Client(api_key = api_key)

MODEL_NAME = "gemini-2.5-flash" 

docs = [
    "RAG stands for Retrieval-Augmented Generation: retrieve relevant context then generate an answer.",
    "SentenceTransformers creates dense embeddings for sentences and paragraphs for semantic search.",
    "Cosine similarity measures the angle between vectors; higher means more similar.",
    "Chunking text (e.g., 200-500 tokens) helps retrieval and reduces irrelevant context.",
    "Gemini is a family of Google models usable via the google-generativeai SDK.",
]

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
doc_emb = embedder.encode(docs, normalize_embeddings=True)

def retrieve(query:str, top_k:int)->list:
    query_emb = embedder.encode([query], normalize_embeddings=True)[0]
    scores = doc_emb @ query_emb #Cosine similarity a @ b = |a||b|cos(theta); if normalized |a|=|b|=1
    idx = np.argsort(-scores)[:top_k]
    return [(docs[i], scores[i]) for i in idx]


def rag_answer(query:str, top_k:int = 1):
    retriever = retrieve(query, top_k)
    context = '\n\n'.join(f'[{i + 1}] {text} {scores}' for i, (text, scores) in enumerate(retriever))
    prompt = f'''Answer ONLY using context. if context doesnt contain the 
                 answer then say i dont know. Context: {context}. \n\n Question: {query}. 
                 '''
    
    try:
        response = client.models.generate_content(
            model = MODEL_NAME,
            contents= prompt
        )
        return response.text 
    except Exception as e:
        return f"Error calling Gemini: {e}"
    

if __name__ == '__main__':
    print(rag_answer("What is RAG"))
    





