---
title: ArXiv RAG Research Assistant
emoji: 🔬
colorFrom: indigo
colorTo: gray
sdk: streamlit
sdk_version: 1.53.1
app_file: app.py
pinned: true
---

# ArXiv RAG Research Assistant

![Papers Indexed](https://img.shields.io/badge/papers%20indexed-1%2C200%2B-blue)
![Python](https://img.shields.io/badge/python-3.11-blue)
![Streamlit](https://img.shields.io/badge/built%20with-Streamlit-ff4b4b)
![License](https://img.shields.io/badge/license-MIT-green)

A **full-stack AI application** that turns the ArXiv research database into a conversational research assistant. Ask any question about AI/ML research and the system retrieves the most semantically relevant paper chunks from a vector database, then uses **Google Gemini** to generate an answer grounded entirely in those papers — no hallucination, no guesswork.

Built to demonstrate production RAG patterns: streaming, multi-hop reasoning, agentic routing, sentence-boundary chunking, PDF upload, cosine-similarity recommendations, email alerts, and answer quality feedback — all in a single deployable app.

![App Demo](/images/demo.png)

---

## Features

| Feature | Description |
|---|---|
| **Streaming answers** | Gemini responses stream word-by-word via `generate_content_stream` + `st.write_stream` — no blocking spinner |
| **Multi-hop reasoning** | "Deep Search" mode: Pass 1 retrieves papers → Gemini identifies a related concept → Pass 2 retrieves additional papers → synthesised answer across both passes |
| **Agentic routing** | Gemini decides whether to search, ask for clarification, or decline out-of-scope queries before touching the database |
| **Sentence-boundary chunking** | NLTK `sent_tokenize` preserves full sentences in each chunk (800-char target) — better embeddings than fixed-window splits |
| **Semantic + hybrid search** | Queries encoded with `all-MiniLM-L6-v2`; matched via pgvector cosine similarity with BM25 hybrid fallback (RRF scoring) |
| **RAG generation** | Top retrieved chunks assembled into a grounded prompt; the LLM can only cite what the papers say |
| **Multi-language support** | Ask in any language — Gemini responds in the same language as the question |
| **PDF Upload** | Upload any PDF (not just ArXiv); same chunking and embedding pipeline applied in-session; nothing written to the database |
| **Paper recommendations** | Saved reading list → embedding centroid → cosine search → top-8 papers you haven't seen yet |
| **Answer quality feedback** | 👍 / 👎 buttons write to a Supabase `feedback` table; Analytics tab shows ratings over time |
| **Weekly paper alerts** | Subscribe with email + topics; GitHub Actions sends a personalised HTML digest every Monday via SendGrid |
| **Confidence indicator** | Cosine similarity scores drive a High / Medium / Low confidence badge shown with every answer |
| **Comparison mode** | Run two queries side-by-side; Gemini contrasts what the papers say about each topic |
| **Action buttons** | After any answer: summarise in 3 bullets, find open problems, explain for students, explore related concepts |
| **Student Mode** | Sidebar toggle appends an undergraduate-friendly instruction to every Gemini prompt |
| **Category filter** | Sidebar multiselect (cs.AI / cs.LG / cs.CL / cs.CV) filters retrieval and the Papers Database tab |
| **Reading List** | Save papers with one click; paper recommendations appear once ≥2 papers are saved; export all as BibTeX |
| **Trending This Week** | Papers indexed in the last 7 days (falls back to 30), category bar chart, one-click weekly digest |
| **Live hero stats** | Landing page shows papers indexed, queries this month, and answers rated — pulled from Supabase in real time |
| **Session Analytics** | Query history table, confidence chart, category usage, feedback summary metrics |
| **Auto model discovery** | Probes Gemini models newest-first at startup; survives Google deprecations without code changes |
| **Idempotent ETL** | Re-running the pipeline skips already-indexed papers; URL-level deduplication in the database |
| **Finance domain (q-fin)** | Indexes quantitative finance papers (q-fin.ST · CP · PM · TR · RM · MF) alongside AI/ML — relevant to fintech and quant roles |
| **REST API (FastAPI)** | `POST /api/search`, `/api/search/stream`, `POST /api/summarize`, `GET /api/papers` — same logic as the UI, consumable from any client |
| **Evaluation framework** | 20-question eval set with Hit Rate@5 and MRR; `eval/run_eval.py` runs the full pipeline and reports per-domain results |
| **Daily ETL automation** | GitHub Actions cron runs every day at 02:00 UTC and pushes new papers into the vector store |

---

## Architecture

```
ArXiv API  ──(weekly cron)──►  ETL Pipeline
                                    │
                          PDF → sentence-boundary chunks (NLTK)
                          → embeddings (all-MiniLM-L6-v2, 384-dim)
                          → Supabase / pgvector
                                    │
User query ──► Gemini router  ──►  embed query  ──►  hybrid search (cosine + BM25)
                                    │
                          [Standard path]  top-5 diverse chunks
                                    │
                          [Deep Search]    Pass 1 chunks
                                    │     → Gemini extracts related concept
                                    │     → Pass 2 chunks (deduplicated)
                                    │
                          Gemini (streaming)  ──►  grounded answer + sources
```

1. **Extract** — ArXiv REST API, sorted by submission date descending, categories `cs.AI OR cs.LG OR cs.CL OR cs.CV OR q-fin.ST OR q-fin.CP OR q-fin.PM OR q-fin.TR OR q-fin.RM OR q-fin.MF`
2. **Transform** — `pypdf` text extraction → NLTK sentence-boundary chunks (~800 chars, min 100 chars) → `all-MiniLM-L6-v2` embeddings; category stored in metadata
3. **Load** — Supabase PostgreSQL + `pgvector`; URL-level deduplication skips already-indexed papers on re-runs
4. **Route** — Gemini classifies each query as *search / clarify / out-of-scope* via a structured JSON prompt
5. **Retrieve** — pgvector hybrid search (cosine + BM25 with RRF); results deduplicated and diversified across papers; optionally filtered by category
6. **Generate** — Retrieved chunks + question → streaming Gemini prompt → answer grounded in paper text

See [ARCHITECTURE.md](ARCHITECTURE.md) for indexing strategy, scaling, FastAPI design, eval methodology, and design decision rationale.

---

## App Layout (6 tabs)

| Tab | Description |
|---|---|
| **Ask a Question** | Agentic search with Deep Search (multi-hop), comparison mode, 4 action buttons, feedback, Save + BibTeX per source |
| **Trending This Week** | Recent papers, category breakdown chart, one-click weekly digest |
| **Reading List** | Saved papers, paper recommendations, individual Remove, Export All as BibTeX, Clear All |
| **Papers Database** | Full paper index with title search, year-range slider, category filter |
| **Analytics** | Query history, confidence chart, category usage, answer quality ratings summary |
| **Upload PDF** | Upload any PDF, ask questions, view source passages — in-session, nothing stored to the database |

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM & routing | Google Gemini (`google-genai` SDK — streaming + blocking, auto-discovers model) |
| Vector store | Supabase — PostgreSQL + `pgvector` (hybrid search: cosine + BM25 / RRF) |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` (384-dim, CPU, ~9K sentences/sec) |
| Chunking | NLTK `sent_tokenize` — sentence-boundary chunks, ~800 char target |
| Frontend | Streamlit (streaming, session state, `st.cache_resource` / `st.cache_data`) |
| Email | SendGrid free tier (weekly digest via GitHub Actions) |
| ETL automation | GitHub Actions (cron weekly ETL + cron weekly email alerts) |
| Language | Python 3.11 |

---

## How to Run Locally

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/arxiv-search-engine.git
cd arxiv-search-engine
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Environment variables

Create a `.env` file:

```
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-supabase-service-role-key
GOOGLE_API_KEY=your-google-ai-studio-key

# Optional — only needed to send email alerts
SENDGRID_API_KEY=your-sendgrid-api-key
ALERT_FROM_EMAIL=alerts@yourdomain.com
```

### 3. Set up Supabase

Run the following once in the **Supabase SQL Editor** (Project → SQL Editor):

```sql
-- Enable pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- Documents table
CREATE TABLE IF NOT EXISTS documents (
    id         BIGSERIAL PRIMARY KEY,
    content    TEXT,
    embedding  vector(384),
    metadata   JSONB,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Cosine similarity search function
CREATE OR REPLACE FUNCTION match_documents(
    query_embedding vector(384),
    match_threshold float,
    match_count     int
)
RETURNS TABLE (id bigint, content text, metadata jsonb, similarity float)
LANGUAGE sql STABLE AS $$
    SELECT id, content, metadata,
           1 - (embedding <=> query_embedding) AS similarity
    FROM documents
    WHERE 1 - (embedding <=> query_embedding) > match_threshold
    ORDER BY similarity DESC
    LIMIT match_count;
$$;
```

Then run **`supabase_migrations.sql`** in the same editor to create the `feedback`, `query_log`, and `paper_alerts` tables.

### 4. Populate the database

```bash
python3 etl_pipeline.py
```

### 5. Run the app

```bash
streamlit run app.py
```

### 6. (Optional) Run the REST API

```bash
uvicorn api:app --reload --port 8000
# Swagger UI: http://localhost:8000/docs
```

### 7. (Optional) Run the evaluation

```bash
python3 eval/run_eval.py
```

Requires the database to be populated. Finance questions (Q16–Q20) need q-fin papers indexed first.

### 8. (Optional) Set up email alerts

Add `SENDGRID_API_KEY` and `ALERT_FROM_EMAIL` to your GitHub repository secrets. The workflow in `.github/workflows/paper_alerts.yml` triggers every Monday at 08:00 UTC. Run manually at any time from the Actions tab.

---

## Project Structure

```
├── app.py                         # Streamlit UI — 6 tabs, hero, sidebar, full RAG pipeline
├── api.py                         # FastAPI REST layer — /api/search, /api/summarize, /api/papers
├── etl_pipeline.py                # Extract → Transform → Load (ArXiv + q-fin → pgvector, daily)
├── send_alerts.py                 # Weekly email digest sender (SendGrid)
├── supabase_migrations.sql        # DDL for feedback, query_log, paper_alerts tables
├── check_models.py                # Lists available Gemini models for debugging
├── requirements.txt               # Pinned dependencies
├── ARCHITECTURE.md                # Deep-dive: indexing, scaling, eval, design decisions
├── CONTRIBUTING.md                # Contribution guidelines
├── eval/
│   ├── eval_set.json              # 20 questions with expected keywords (15 AI/ML + 5 finance)
│   └── run_eval.py                # Evaluation runner — Hit Rate@5, MRR, per-domain breakdown
├── .github/
│   └── workflows/
│       ├── weekly_update.yml      # ETL cron — every day 02:00 UTC
│       └── paper_alerts.yml       # Email alerts cron — every Monday 08:00 UTC
└── downloads/                     # Cached PDFs (git-ignored)
```

---

## Design Decisions

See [ARCHITECTURE.md §12](ARCHITECTURE.md#12-design-decisions) for detailed rationale on:

- Why `all-MiniLM-L6-v2` over OpenAI `text-embedding-3-small` or `all-mpnet-base-v2`
- Why pgvector + Supabase over Pinecone, Weaviate, or Qdrant
- Why Streamlit over FastAPI + React
- Why Google Gemini over OpenAI GPT-4o or Anthropic Claude
