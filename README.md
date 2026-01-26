flowchart TD
  A[PDF Corpus<br/>data/raw/pdfs/*.pdf] --> B[Ingestion Pipeline<br/>(stage1 → stage3)]
  B --> C[(Qdrant Vector DB<br/>rag_collection)]
  C --> D[Retrieval<br/>query → embed → search]
  D --> E[Evaluation<br/>queries + doc-labels → metrics]




flowchart TD
  subgraph CFG[Configuration]
    ENV[.env<br/>(infra wiring only)] --> INFRA[InfraSettings]
    YAML[configs/pipeline.yaml<br/>(versioned algorithm config)] --> PIPE[PipelineConfig]
  end

  subgraph DISC[Discovery + Versioning]
    PDFs[Find PDFs<br/>glob PDF_DIR] --> HASH[doc_id = sha256(pdf_bytes)]
    PIPE --> CFP[config_fingerprint(hash of yaml)]
    HASH --> CV[corpus_version = hash(cfg_fp + sorted(doc_ids))]
    CFP --> CV
  end

  subgraph RESUME[Resume / Idempotency]
    CV --> CACHE[data/ingested/ingested_<corpus_version>.txt]
    CACHE --> SKIP{doc_id already ingested?}
  end

  subgraph DATA[Data Processing]
    SKIP -- No --> LOAD[Loader<br/>read_pdf_text_best_effort]
    LOAD --> CHUNK[Chunker<br/>chunk_text(chunk_chars, overlap)]
    CHUNK --> EMB[Embedder<br/>encode(chunks) → vectors]
  end

  subgraph INDEX[Indexing]
    EMB --> PTS[Build Points<br/>id=uuid5(doc_id:chunk_index)<br/>payload includes doc_id, text, corpus_version...]
    PTS --> UPSERT[Qdrant upsert<br/>batched]
    UPSERT --> MARK[Append doc_id to cache<br/>(only after success)]
  end

  subgraph TRACK[Tracking]
    MARK --> MANIFEST[data/runs/<run_id>.json<br/>counts + timings + failures<br/>corpus_version + config_fingerprint]
  end

  SKIP -- Yes --> DONE[(skip)]
  ENV --> INFRA --> PDFs
  PIPE --> PDFs



flowchart TD
  QSET[data/eval/queries.jsonl<br/>qid + query] --> QEMB[Embed query<br/>same embed model]
  QEMB --> SEARCH[Qdrant HTTP search<br/>filter: corpus_version]
  SEARCH --> HITS[Top hits = chunks<br/>payload contains doc_id]
  HITS --> DEDUP[Deduplicate doc_id<br/>rank order → doc list]

  LSET[data/eval/labels.jsonl<br/>qid → gold_doc_ids] --> METRICS[Compute metrics<br/>recall@k, MRR(optional), latency p50/p95]
  DEDUP --> METRICS

  METRICS --> REPORT[data/runs/eval_<corpus>.json<br/>metrics + corpus_version]





flowchart LR
  S1[Stage 1<br/>Make it work] --> S2[Stage 2<br/>Make it modular]
  S2 --> S3[Stage 3<br/>Make it measurable + reliable]

  S1 --> A1[Single script<br/>no tracking]
  S2 --> A2[Loaders/Chunking/Embedding/Indexing modules]
  S3 --> A3[corpus_version + manifests + cache per corpus + doc-level eval]
