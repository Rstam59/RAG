flowchart TD
  %% ---------------------------
  %% Inputs
  %% ---------------------------
  A[PDF Corpus<br/>data/raw/pdfs/*.pdf]
  B[.env<br/>Infra wiring<br/>QDRANT_URL, PDF_DIR, etc]
  C[configs/pipeline.yaml<br/>Versioned algorithm config<br/>chunking, embed model, versions]

  %% ---------------------------
  %% Ingestion
  %% ---------------------------
  subgraph INGESTION[Ingestion Pipeline (Stage 3)]
    D[Discover PDFs<br/>glob *.pdf]
    E[Versioning<br/>file_hash=sha256(pdf bytes)<br/>doc_id=file_hash<br/>cfg_fp=hash(config)<br/>corpus_version=hash(cfg_fp + doc_ids)]
    F[Resume Cache<br/>data/ingested/ingested_{corpus_version}.txt<br/>skip already ingested docs]
    G[PDF Loader<br/>best-effort extract_text<br/>decrypt if possible<br/>skip broken pages]
    H[Chunking<br/>chunk_text(chunk_chars, overlap)]
    I[Embedding<br/>SentenceTransformer.encode(chunks)]
    J[Indexing to Qdrant<br/>point_id=uuid5(doc_id:chunk_index)<br/>payload={doc_id,file,text,...,versions}<br/>upsert batches]
    K[Tracking<br/>JSON logs + run manifest<br/>data/runs/{run_id}.json]
  end

  %% ---------------------------
  %% Retrieval
  %% ---------------------------
  subgraph RETRIEVAL[Retrieval]
    Q[User Query]
    R[Embed Query<br/>(same embedding model)]
    S[Qdrant Search (HTTP)<br/>filter corpus_version]
    T[Top Hits<br/>(chunks)]
    U[Deduplicate by doc_id<br/>rank docs]
  end

  %% ---------------------------
  %% Evaluation
  %% ---------------------------
  subgraph EVAL[Evaluation (Doc-Level)]
    V[data/eval/queries.jsonl<br/>qid + query]
    W[data/eval/labels.jsonl<br/>qid → gold_doc_ids]
    X[Metrics<br/>recall@k + latency p50/p95]
    Y[Eval Report<br/>data/runs/eval_{corpus_version}.json]
  end

  %% ---------------------------
  %% Links
  %% ---------------------------
  A --> D
  B --> INGESTION
  C --> INGESTION

  D --> E --> F --> G --> H --> I --> J --> K

  Q --> R --> S --> T --> U
  J --> S

  V --> R
  W --> X
  U --> X --> Y
