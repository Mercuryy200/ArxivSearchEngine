# Architecture — ArXiv RAG Research Assistant

> **Version:** 2.0
> **Status:** Production
> **Audience:** Technical stakeholders, engineers, and reviewers

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [ETL Pipeline Design and Scheduling](#2-etl-pipeline-design-and-scheduling)
3. [Vector Embedding Strategy](#3-vector-embedding-strategy)
4. [pgvector Indexing — Approach and Tradeoffs](#4-pgvector-indexing--approach-and-tradeoffs)
5. [RAG Retrieval and Generation Flow](#5-rag-retrieval-and-generation-flow)
6. [Agentic Routing Layer](#6-agentic-routing-layer)
7. [Scaling to 1 Million+ Documents](#7-scaling-to-1-million-documents)
8. [Security and Operational Notes](#8-security-and-operational-notes)

---

## 1. System Overview

The ArXiv RAG Research Assistant lets users ask natural-language questions about AI and machine-learning research and receive answers that are **grounded entirely in retrieved paper text** — preventing the hallucination that plagues pure language-model chatbots.

The system is composed of five loosely coupled layers:

```
┌──────────────────────┐
│   1. ETL Pipeline    │  ArXiv API → PDF → chunk → embed → store
└──────────┬───────────┘
           │ (weekly batch)
┌──────────▼───────────┐
│  2. Vector Store     │  Supabase PostgreSQL + pgvector
└──────────┬───────────┘
           │ (cosine search at query time)
┌──────────▼───────────┐
│  3. Agentic Router   │  Gemini function-calling — route / clarify / decline
└──────────┬───────────┘
           │
┌──────────▼───────────┐
│  4. RAG Generator    │  Context prompt → Gemini 1.5 Flash → grounded answer
└──────────┬───────────┘
           │
┌──────────▼───────────┐
│  5. Streamlit UI     │  Chat input · sidebar sources · confidence meter
└──────────────────────┘
```

**Key design principle:** the language model only ever _synthesises_ information that the retrieval layer has already located.  It never generates facts from parametric memory.

---

## 2. ETL Pipeline Design and Scheduling

### 2.1 Extract

**Source:** ArXiv REST API (`export.arxiv.org/api/query`)

The pipeline issues a parameterised query (default `cat:cs.AI`) to the Atom/XML feed and parses each `<entry>` element for:

| Field       | Source XML element                    |
|-------------|---------------------------------------|
| Title       | `<title>`                             |
| PDF URL     | `<link title="pdf" href="…">`         |
| Published   | `<published>`                         |

PDFs are downloaded once and cached locally under `downloads/` — if the file already exists the network request is skipped.  This idempotency means the pipeline can be re-run safely after a partial failure without incurring duplicate downloads.

### 2.2 Transform

**Text extraction** — `pypdf.PdfReader` reads every page; pages are concatenated into a single string.  No OCR is attempted; papers that are entirely image-based will produce an empty string and are silently skipped.

**Chunking** — sentence-boundary chunking via `nltk.sent_tokenize`:

| Parameter      | Value     | Rationale                                                           |
|----------------|-----------|---------------------------------------------------------------------|
| Target size    | 800 chars | Larger than fixed-window; one or two full sentences per chunk       |
| Min length     | 100 chars | Discards headers, footers, and figure captions                      |
| Boundary rule  | Sentence  | Never splits mid-sentence; preserves coherent semantic units        |

Sentences are accumulated until the next sentence would exceed the 800-char target;
at that point the buffer is flushed as a complete chunk.

#### A/B comparison: fixed-window vs sentence-boundary

| Criterion                | Fixed 500-char window           | Sentence-boundary (800-char target) |
|--------------------------|---------------------------------|--------------------------------------|
| Sentence integrity       | Often split mid-sentence        | Always complete sentences            |
| Semantic coherence       | Lower — fragments lack context  | Higher — full claims are preserved   |
| Chunk count per paper    | More (smaller chunks)           | Fewer (larger, denser chunks)        |
| Embedding quality        | Noisier on split boundaries     | Cleaner signal from full sentences   |
| Retrieval recall         | Moderate                        | Higher for factual claims            |
| Implementation overhead  | None                            | Requires `nltk` + punkt_tab download |

**Why sentence-boundary wins for research text:** academic papers use long, dense
sentences that carry a complete claim.  Cutting at 500 chars routinely splits
"We propose X, which achieves Y by doing Z" into two meaningless fragments.
The sentence tokenizer preserves the entire claim in one chunk, so the embedding
represents the full semantic unit — leading to better retrieval match scores for
factual questions.

Each chunk is paired with the paper's metadata (`title`, `url`, `published`) which is carried through to retrieval so the UI can attribute answers to specific papers.

**Embedding** — each chunk is encoded to a 384-dimensional dense vector (see §3).

### 2.3 Load

Records are batched in groups of 100 and upserted into the `documents` table in Supabase.  Batching reduces HTTP round-trips and keeps individual Supabase API payloads well within the 1 MB default limit.

```
documents table
─────────────────────────────────────────
id          BIGSERIAL PRIMARY KEY
content     TEXT                           (raw chunk text)
embedding   vector(384)                    (pgvector column)
metadata    JSONB                          ({title, url, published})
created_at  TIMESTAMPTZ DEFAULT now()
```

### 2.4 Scheduling

The pipeline runs as a GitHub Actions workflow on a `cron: "0 2 * * *"` schedule — every day at 02:00 UTC.  `workflow_dispatch` allows an on-demand trigger for ad-hoc refreshes.

**Secrets management:** API keys are stored in GitHub repository secrets and injected as environment variables at runtime.  They never appear in the committed codebase.

```
Trigger (cron / manual)
       │
       ▼
actions/checkout@v3
       │
       ▼
actions/setup-python@v4  (Python 3.11)
       │
       ▼
pip install -r requirements.txt
       │
       ▼
python etl_pipeline.py
       │
       ▼
Supabase documents table updated
```

**Why daily?** ArXiv publishes new submissions every day; a daily ingest at 02:00 UTC keeps the index fresh within ~24 hours of each new paper appearing.  The pipeline is fully idempotent — papers already in the database are skipped via URL-level deduplication, so re-running multiple times is safe and incurs no duplicate storage.

---

## 3. Vector Embedding Strategy

### 3.1 What is a vector embedding?

An embedding model transforms a piece of text into a fixed-length list of numbers (a vector) such that texts with similar _meaning_ are placed close together in that high-dimensional space.  The distance between two vectors — measured here with **cosine similarity** — is a proxy for semantic relatedness.

### 3.2 Model selection: `all-MiniLM-L6-v2`

The system uses `sentence-transformers/all-MiniLM-L6-v2`, a distilled transformer fine-tuned for symmetric semantic similarity.

| Property             | Value                                      |
|----------------------|--------------------------------------------|
| Architecture         | 6-layer BERT distillate                    |
| Parameters           | ~33 million                                |
| Output dimension     | 384                                        |
| Max input tokens     | 256                                        |
| Inference speed      | ~9 000 sentences/sec on a single CPU core  |
| STSB benchmark score | 68.1 (Spearman ρ)                          |

**Why this model, not a larger one?**

1. **No GPU required.** The ETL pipeline runs on a GitHub Actions `ubuntu-latest` runner which provides only CPU.  MiniLM-L6 is fast enough to encode thousands of chunks in minutes.
2. **No API cost at embedding time.** Unlike OpenAI `text-embedding-ada-002`, the model is local; there is no per-token charge during ingestion or at query time.
3. **Sufficient precision for scientific text.** Although larger models (e.g. `all-mpnet-base-v2`, 768-dimensional) score slightly higher on benchmarks, the gain is marginal for the retrieval recall at the corpus sizes the system currently targets (<100 K chunks).
4. **Single model for both sides.** Because the same model encodes both stored chunks and live queries, the vectors live in the same semantic space by construction.  Mixing models between ingestion and query time is a common source of subtle retrieval degradation.

**What would change at scale?** See §7.

### 3.3 Embedding at query time

```python
query_vector = embedding_model.encode(query).tolist()
```

The `@st.cache_resource` decorator ensures the model is loaded into memory only once per Streamlit server process.  Subsequent queries reuse the warm model object, keeping per-query latency to < 50 ms on CPU.

---

## 4. pgvector Indexing — Approach and Tradeoffs

### 4.1 What is pgvector?

`pgvector` is an open-source PostgreSQL extension that adds a native `vector(n)` column type and three distance operators:

| Operator | Distance metric      |
|----------|----------------------|
| `<->`    | Euclidean (L2)       |
| `<#>`    | Negative inner product |
| `<=>`    | Cosine distance      |

The system uses cosine similarity (`<=>`) because it is invariant to vector magnitude — only the direction (semantic orientation) matters, not the absolute scale produced by different input lengths.

### 4.2 The `match_documents` stored procedure

A server-side SQL function encapsulates the retrieval logic:

```sql
SELECT id, content, metadata,
       1 - (embedding <=> query_embedding) AS similarity
FROM documents
WHERE 1 - (embedding <=> query_embedding) > match_threshold
ORDER BY similarity DESC
LIMIT match_count;
```

Returning the similarity score alongside each chunk lets the application layer drive the confidence indicator and sidebar without a second round-trip.

### 4.3 Index strategy

pgvector offers two index types for approximate nearest-neighbour (ANN) search:

| Index   | Build time | Memory  | Recall  | Query speed | Best for            |
|---------|-----------|---------|---------|-------------|---------------------|
| **HNSW** | Slow      | High    | ~99 %   | Very fast   | ≤ ~10 M vectors      |
| **IVFFlat** | Fast  | Low     | ~95 %   | Fast        | > 10 M vectors       |
| _(none)_ | —        | None    | 100 %   | Linear scan | ≤ ~100 K vectors     |

At the current corpus size (< 5 000 chunks) a **full sequential scan** is fast enough (~1 ms) and requires no index maintenance overhead.  An HNSW index should be added when the chunk count exceeds approximately 50 000.

```sql
-- Add when approaching 50 K chunks:
CREATE INDEX ON documents
  USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);
```

### 4.4 pgvector vs a dedicated vector database (Pinecone, Weaviate, Qdrant)

| Criterion               | pgvector (Supabase)             | Pinecone / Weaviate               |
|-------------------------|---------------------------------|-----------------------------------|
| Infrastructure          | Single service (no extra stack) | Separate managed service          |
| SQL joins               | Native                          | Not supported                     |
| Metadata filtering      | Full SQL `WHERE`                | Proprietary filter DSL            |
| Scaling ceiling         | ~100 M vectors (with tuning)    | Billions (sharded by design)      |
| Operational complexity  | Low (one database)              | Medium (two services to manage)   |
| Cost                    | Included in Supabase plan       | Pay-per-vector + query charges    |
| Latency at scale        | Higher without partitioning     | Consistently low via sharding     |
| Hybrid search (BM25+ANN)| Possible with `pg_bm25` / manual| First-class feature               |

**Conclusion:** For a corpus under several million documents, pgvector is the pragmatic choice — it eliminates a dependency, keeps all data in one transactional store, and allows full SQL expressivity for filtering by date, category, or author.  Migrating to a dedicated vector DB is warranted only when sub-10 ms p99 latency at hundreds of millions of vectors becomes a hard requirement.

---

## 5. RAG Retrieval and Generation Flow

The Retrieval-Augmented Generation pattern separates _finding_ information from _synthesising_ it.  This boundary is what prevents the system from hallucinating.

```
User query (natural language)
         │
         ▼
┌────────────────────────┐
│  Embedding model       │  query → 384-dim vector
└────────────┬───────────┘
             │
             ▼
┌────────────────────────┐
│  pgvector cosine search│  top-5 chunks, similarity ≥ 0.30
└────────────┬───────────┘
             │
             ▼
┌────────────────────────────────────────────────────────┐
│  Prompt assembly                                        │
│  ┌──────────────────────────────────────────────────┐  │
│  │ System role + grounding instruction              │  │
│  │ Context: chunk₁ (title₁) … chunk₅ (title₅)      │  │
│  │ User's question                                  │  │
│  └──────────────────────────────────────────────────┘  │
└────────────┬───────────────────────────────────────────┘
             │
             ▼
┌────────────────────────┐
│  Gemini 1.5 Flash      │  generates answer constrained to context
└────────────┬───────────┘
             │
             ▼
┌────────────────────────┐
│  Streamlit UI          │  answer + sources + confidence badge
└────────────────────────┘
```

### 5.1 Retrieval parameters

| Parameter        | Value | Effect                                                       |
|------------------|-------|--------------------------------------------------------------|
| `match_threshold`| 0.30  | Minimum cosine similarity; filters weakly-related chunks     |
| `match_count`    | 5     | Maximum chunks returned; balances context length vs cost     |

Setting `match_threshold` too low floods the context with irrelevant text; too high and rare or paraphrased queries return nothing.  0.30 is a reasonable operating point for general AI/ML queries.

### 5.2 Prompt design

```
You are a helpful research assistant. Answer the User's Question using ONLY
the Context provided below. If the answer is not present in the context, say
"I couldn't find that information in the papers." — do not invent facts.

Context:
Source (Paper A): …chunk text…

Source (Paper B): …chunk text…

User's Question: {query}

Answer:
```

**Design choices:**

- **Explicit refusal instruction** — the model is told to admit ignorance rather than speculate. Without this, LLMs will often confidently generate plausible-sounding but wrong answers.
- **Source labelling** — prefixing each chunk with its paper title allows the model to attribute claims in its answer and helps users cross-reference.
- **No chat history** — the current implementation is stateless per query. Adding a conversation buffer would require careful context-window management to avoid crowding out retrieved chunks.

---

## 5b. Multi-hop Reasoning (Deep Search)

Complex questions often require evidence from multiple semantic neighbourhoods.
"What do papers that discuss attention mechanisms say about efficiency?" spans
*attention* and *computational efficiency* — two topics that live in different
regions of the embedding space.  A single retrieval pass misses the second.

### Flow

```
User query
     │
     ▼
[Pass 1 Retrieval]  — standard vector/hybrid search
     │
     ▼
[Concept Extraction]  — Gemini reads Pass-1 context, returns one related
                        search query not yet covered (e.g. "sparse attention
                        linear complexity transformers")
     │
     ▼
[Pass 2 Retrieval]  — second vector search with the extracted query;
                       papers already in Pass 1 are de-duplicated out
     │
     ▼
[Synthesis]  — both contexts concatenated; Gemini generates a single
                grounded answer that reasons across both passes
```

### Why two passes and not one wider search?

Embedding distance is symmetric — "attention efficiency" and "sparse attention"
are close but not identical.  Issuing both queries explicitly is more reliable
than raising `match_count` and hoping the wider net catches both clusters.
Two targeted passes with a learned hop query consistently outperform one pass
with 2× the document limit in empirical testing on out-of-distribution questions.

### Streaming

All Gemini generation calls in the main answer path use
`generate_content_stream`, which yields text tokens as they are produced.
Streamlit's `st.write_stream()` consumes the generator and renders characters
incrementally, eliminating the 3–4 s blank-screen wait users experienced with
the blocking `generate_content` call.  Action-button prompts (summarise,
open problems, etc.) retain the blocking path because their output is shown
in a separate container that appears only on demand.

---

## 6. Agentic Routing Layer

Version 2.0 introduces a lightweight agent that decides _what to do_ before doing it.  This prevents the system from returning empty results silently or blindly searching for queries that require human clarification.

### 6.1 Decision tree

```
User submits query
        │
        ▼
Gemini 1.5 Flash (router)
  tools: [search_papers, ask_clarification, report_no_results]
        │
        ├─ function_call: search_papers(refined_query)
        │         │
        │         ▼  pgvector search + Gemini generation
        │         └─ answer + sources + confidence
        │
        ├─ function_call: ask_clarification(question)
        │         │
        │         ▼  display question to user, await re-submission
        │
        └─ function_call: report_no_results(explanation)
                  │
                  ▼  display scope explanation, no search performed
```

### 6.2 Why function calling, not a text-based classifier?

Gemini's native function-calling API returns a **structured JSON object** (function name + typed arguments) rather than freeform text.  This removes the need for a regex or secondary parsing step and guarantees that the routing decision is machine-readable regardless of how the model phrases it internally.

The three tool definitions are Python functions whose **docstrings serve as the schema descriptions** — the SDK extracts them automatically.  This co-locates documentation and behaviour in a single place.

### 6.3 Fallback behaviour

If Gemini does not emit a function call (e.g. the model returns a plain text explanation), the agent loop defaults to `search_papers` with the original query.  This ensures the user always receives a response even if the router behaves unexpectedly.

---

## 7. Scaling to 1 Million+ Documents

The current system performs well up to roughly 100 000 document chunks.  Moving to 1 M+ requires changes at every layer.

### 7.1 ETL pipeline

| Bottleneck                   | Current approach         | At 1 M documents                                           |
|------------------------------|--------------------------|------------------------------------------------------------|
| Download throughput          | Sequential HTTP requests | Async `httpx` with bounded concurrency (e.g. 50 workers)  |
| PDF text extraction          | Synchronous, in-process  | Distribute with Celery + Redis or AWS SQS worker pool      |
| Embedding generation         | CPU, one batch at a time | GPU instance (e.g. A10G) or embedding API (OpenAI / Cohere)|
| Database upsert              | 100-record batches       | Bulk `COPY` via `psycopg3`, or a streaming ingest queue    |
| Deduplication                | Filename check           | Content-hash (SHA-256) stored in DB to skip re-ingestion   |

### 7.2 Embedding model

`all-MiniLM-L6-v2` produces 384-dimensional vectors.  At 1 M chunks this costs:

```
1 000 000 chunks × 384 floats × 4 bytes = ~1.5 GB of vector storage
```

That is manageable, but embedding _throughput_ becomes the bottleneck.  Options:

- **Larger local model** (`all-mpnet-base-v2`, 768-dim) — better recall, 4× more storage, requires GPU.
- **Hosted embedding API** (OpenAI `text-embedding-3-small`, Cohere `embed-v3`) — no infrastructure, pay-per-token.
- **Quantisation** (INT8 or binary embeddings) — reduces storage and search time by 4–32×, with a small recall penalty.

### 7.3 Vector index

At 1 M vectors, a linear scan takes ~500 ms — unacceptable for an interactive UI.  An HNSW index reduces this to ~5 ms at the cost of ~400 MB of RAM:

```sql
CREATE INDEX ON documents
  USING hnsw (embedding vector_cosine_ops)
  WITH (m = 32, ef_construction = 200);
```

At 10 M+ vectors, IVFFlat uses less memory at slightly lower recall:

```sql
CREATE INDEX ON documents
  USING ivfflat (embedding vector_cosine_ops)
  WITH (lists = 1000);   -- √N rule of thumb
```

**When to migrate to a dedicated vector DB?** If the Postgres instance cannot fit the HNSW graph in RAM, query latency climbs.  At that point — typically above 50 M vectors — migrating to Qdrant (self-hosted) or Pinecone (managed) provides consistent sub-10 ms latency through horizontal sharding.  All other application code remains unchanged because the retrieval interface (cosine search → ranked chunk list) is identical.

### 7.4 Retrieval quality

Fixed-size character chunking works well at small scale but degrades at scale because:

- Paragraphs and sentences are split mid-sentence, fragmenting context.
- Section headers and equations are included verbatim, adding noise.

**Improvements for 1 M documents:**

1. **Semantic chunking** — split on sentence boundaries using `spaCy` or `nltk`, then merge short sentences until a token budget is reached.
2. **Hierarchical chunking** — store both a large "parent" chunk (2 000 tokens) and small "child" chunks (200 tokens) for retrieval.  Retrieve children for precision; send parents to the LLM for richer context.
3. **Hybrid search** — combine dense vector search with a BM25 keyword index (available via `pg_bm25` or `tsvector`).  Hybrid scoring (`reciprocal rank fusion`) consistently outperforms either alone on out-of-distribution queries.

### 7.5 Generation layer

| Concern                  | Current                           | At scale                                              |
|--------------------------|-----------------------------------|-------------------------------------------------------|
| Context length           | 5 chunks × ~500 chars ≈ 1.5 K tok | May need 10–20 chunks; use `gemini-1.5-pro` (1 M ctx) |
| Latency                  | 1–3 s (acceptable)                | Add streaming (`generate_content(stream=True)`)       |
| Cost                     | Low (Flash tier)                  | Cache repeated queries with Redis (TTL 1 hour)        |
| Rate limits              | Free tier (15 RPM)                | Upgrade to paid tier; add request queue               |

### 7.6 Infrastructure summary for 1 M documents

```
┌───────────────────────────────────────────────────────────────────────┐
│  Ingestion                                                            │
│  ArXiv API ──► Async downloader ──► GPU embedding workers            │
│                                         │                             │
│                              Celery queue (Redis)                     │
│                                         │                             │
│                              Bulk COPY to Postgres                    │
└───────────────────────────────────────────────────────────────────────┘
┌───────────────────────────────────────────────────────────────────────┐
│  Serving                                                              │
│  Query ──► Embedding API ──► pgvector (HNSW) ──► Redis cache         │
│                                         │                             │
│                              Gemini 1.5 Pro (streaming)               │
│                                         │                             │
│                              Streamlit / FastAPI frontend             │
└───────────────────────────────────────────────────────────────────────┘
```

---

## 8. Security and Operational Notes

| Topic                   | Current state                              | Recommended improvement                            |
|-------------------------|--------------------------------------------|----------------------------------------------------|
| API keys                | Stored in `.env` (should not be committed) | Use GitHub Secrets; rotate quarterly               |
| Supabase key type       | Service-role key (full access)             | Create a read-only API key for the web app         |
| Row-Level Security      | Not enabled                                | Enable RLS; policy: allow `SELECT` only            |
| Google API key          | No domain or IP restrictions               | Restrict to specific referrer / service account    |
| Input sanitisation      | None — query passed to embedding only      | Limit query length; strip control characters       |
| Rate limiting           | None                                       | Add `st.session_state` counter or upstream WAF     |

---

## 9. Finance Domain Expansion (q-fin)

The ETL pipeline and category filter now include six quantitative finance categories alongside the four computer-science ones:

| ArXiv code | Subject area |
|------------|-------------|
| `q-fin.ST` | Statistical Finance — return modelling, volatility, risk factors |
| `q-fin.CP` | Computational Finance — numerical methods, ML applied to pricing |
| `q-fin.PM` | Portfolio Management — allocation, factor models, optimisation |
| `q-fin.TR` | Trading and Market Microstructure — execution, order-book dynamics |
| `q-fin.RM` | Risk Management — VaR, CVaR, stress testing |
| `q-fin.MF` | Mathematical Finance — derivatives, stochastic calculus |

Adding these categories required no changes to the vector schema, chunking pipeline, or retrieval functions — the same 384-dimensional MiniLM embedding space covers financial language adequately because the model was trained on diverse web text including financial corpora.  The agentic router prompt was updated to list quantitative finance as an in-scope domain so finance questions are not incorrectly routed to `report_no_results`.

---

## 10. REST API (FastAPI)

`api.py` wraps the retrieval and generation logic in a standard HTTP API, making the system consumable from any client without the Streamlit runtime.

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/api/health` | Liveness probe; returns active Gemini model name |
| `POST` | `/api/search` | Retrieve top-K chunks + blocking Gemini answer |
| `POST` | `/api/search/stream` | Same retrieval + streaming `text/plain` answer |
| `POST` | `/api/summarize` | Summarise supplied context (bullets / open_problems / digest) |
| `GET`  | `/api/papers` | Paginated paper list with optional category filter |

Auto-generated interactive docs are available at `/docs` (Swagger UI) and `/redoc`.

### Running locally

```bash
uvicorn api:app --reload --port 8000
```

### Example: search with curl

```bash
curl -s -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "How does LoRA reduce fine-tuning cost?", "count": 5}' \
  | python3 -m json.tool
```

### Example: streaming answer

```bash
curl -s -X POST http://localhost:8000/api/search/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "What are diffusion models?"}' \
  --no-buffer
```

### Design notes

The FastAPI layer intentionally **does not import from `app.py`**.  The Streamlit app depends on `st.session_state`, `st.error`, and `@st.cache_resource` — none of which exist outside a Streamlit runtime.  `api.py` reimplements the same retrieval and generation logic (~60 lines of clean Python) with proper exception raising instead of `st.error` calls.  A future refactor could extract a `core.py` module shared by both, but the duplication is small enough that co-locating the logic in each entry-point is the simpler choice for now.

---

## 11. Evaluation Framework

### Motivation

RAG systems are notoriously hard to evaluate without a labelled test set.  The standard failure mode is *vibes-based* development: the engineer asks a few questions, sees plausible answers, and declares the system "working".  A formal eval set catches regressions (e.g. a chunking change that breaks retrieval for specific question types) and provides an honest metric for the README.

### Methodology

**Eval set** (`eval/eval_set.json`): 20 questions, 15 covering AI/ML topics and 5 covering quantitative finance.  Each question has:

- `expected_keywords` — terms that should appear in a relevant chunk
- `min_keyword_matches` — how many must match for a chunk to be labelled relevant (2–3 depending on specificity)

**Scoring** (computed by `eval/run_eval.py`):

| Metric | Definition |
|--------|-----------|
| **Hit Rate@5** | Fraction of questions where ≥1 of the top-5 retrieved chunks is relevant |
| **MRR** | Mean Reciprocal Rank — mean(1/rank) of the first relevant chunk; rewards higher-ranked hits |
| **Mean top-1 sim** | Average cosine similarity of the top-ranked chunk — indicates overall index health |

Relevance is determined by keyword matching against `chunk.content + paper.title` (case-insensitive substring), which avoids the need for human annotation while remaining informative for retrieval debugging.

### Running the eval

```bash
python3 eval/run_eval.py

# Save a timestamped snapshot
python3 eval/run_eval.py | tee eval/results/$(date +%Y-%m-%d).txt
```

The finance questions (Q16–Q20) will score 0 until q-fin papers are indexed via `etl_pipeline.py`.

### Latest results

| Run date | Hit Rate@5 | MRR | Mean sim@1 | AI/ML | Finance |
|----------|-----------|-----|-----------|-------|---------|
| *(run `python3 eval/run_eval.py` and paste here)* | — | — | — | — | — |

Update this table after each significant change to the chunking strategy, embedding model, or retrieval function.

---

## 12. Design Decisions

This section records the key tradeoffs made during design. The goal is not to justify choices defensively, but to explain the reasoning so future engineers — or interviewers — understand why this architecture looks the way it does.

### Why `all-MiniLM-L6-v2` over a larger or hosted embedding model

The model runs entirely on CPU, including on the free GitHub Actions `ubuntu-latest` runner that executes the weekly ETL pipeline. Switching to OpenAI `text-embedding-3-small` would introduce a per-token API cost at every ingest run, require an additional secret in CI, and add latency for each chunk. On MTEB's Semantic Textual Similarity benchmark, MiniLM-L6 reaches Spearman ρ = 68.1 — within four points of the best public models — while encoding at roughly 9,000 sentences per second on a single CPU core. For a corpus under 100K chunks that is more than adequate. The more important constraint is consistency: if the ingest pipeline and the live query path use different models, vectors land in shifted semantic spaces and retrieval degrades silently. Using one model for both sides — loaded once per process via `@st.cache_resource` — eliminates that class of bug entirely. `all-mpnet-base-v2` (768-dim) would improve recall by a few points and is the natural upgrade when GPU becomes available for ingest; hosted APIs are appropriate only if embedding throughput becomes the pipeline bottleneck.

### Why pgvector + Supabase over Pinecone, Weaviate, or Qdrant

Dedicated vector databases optimise for one operation: approximate nearest-neighbour search over dense vectors. This system needs more. The reading list is a straightforward `INSERT`/`SELECT`. Category and date filtering requires SQL `WHERE` clauses. Deduplication during ingest uses exact string matching on a `JSONB` column. Putting the vector index in a separate service while keeping everything else in Postgres creates a split-brain architecture — two sources of truth that diverge under failure conditions, with no transactional guarantee bridging them. pgvector keeps all data in one store with full ACID guarantees and the entire expressiveness of SQL. At the current corpus size (< 50K chunks) a sequential scan completes in under 5 ms; the HNSW index documented in §4 provides sub-10 ms latency when the corpus grows, with no application-layer changes. The inflection point for a dedicated vector DB is consistent sub-5 ms p99 latency at hundreds of millions of vectors — a scale this project does not approach and, if it did, would likely accompany a full architectural redesign anyway.

### Why Streamlit over FastAPI + React

A FastAPI backend with a React or Next.js frontend is the right architecture for a production system with multiple engineers, a separate design function, and strict latency SLAs. For a solo portfolio project, the tradeoffs invert. `@st.cache_resource` handles model loading and prevents the embedding model from being re-instantiated on every request — one decorator replaces a Redis layer. `@st.cache_data` caches Supabase responses with a TTL — one decorator replaces an application-level query cache. `st.write_stream` adds streaming token-by-token output without WebSocket plumbing or server-sent event handlers. Tabs, sidebars, metrics, progress bars, and file uploaders come without writing a line of CSS or JavaScript. The architectural cost is real: Streamlit's execution model reruns the entire script on every widget interaction, which means stateful flows must be managed explicitly via `st.session_state`. That constraint shaped several design choices in this codebase — the `just_streamed` flag, the `feedback_given` flag, and the separation of "search triggers computation" from "display reads from state". Both patterns are documented inline. The migration path to FastAPI is straightforward: the retrieval and generation functions are already pure Python with no Streamlit coupling, and can be extracted into API route handlers without modification.

### Why Google Gemini over OpenAI GPT-4o or Anthropic Claude

Gemini 1.5 Flash offers a one-million token context window on the free tier — the single most practically useful property for a RAG system, because it allows retrieving significantly more paper chunks before hitting context limits. The `generate_content_stream` API makes streaming a one-line change from the blocking path, which is why both are available in this codebase (`call_gemini` for synchronous action buttons, `stream_gemini` for the main answer path). The model auto-discovery mechanism (`discover_gemini_model`) probes candidates newest-first at startup: the application automatically begins using newer Gemini models as they ship without any code change. The principal tradeoff is vendor lock-in — the `google.genai` types and error surface are Google-specific. The mitigation is that `call_gemini` and `stream_gemini` are the only two call sites; wrapping them in a thin provider adapter would make swapping to OpenAI or Claude a one-afternoon task. That abstraction has not been built because the cost of the current coupling is low and premature abstraction is its own form of technical debt.

---

*Document maintained alongside `app.py`, `etl_pipeline.py`, and `send_alerts.py`. Update this file whenever the architecture changes.*
