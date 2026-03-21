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
![Python](https://img.shields.io/badge/python-3.13-blue)
![Streamlit](https://img.shields.io/badge/built%20with-Streamlit-ff4b4b)
![License](https://img.shields.io/badge/license-MIT-green)

A **full-stack AI application** that turns the ArXiv research database into a conversational assistant. Ask a question about AI/ML or quantitative finance research and the system retrieves the most relevant paper chunks using **hybrid search**, then uses **Google Gemini** to generate a grounded, streamed answer — no hallucination.

![App Demo](/images/demo.png)

---

## Features

| Feature                    | Description                                                                                                              |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| **Agentic routing**        | Gemini decides whether to search, ask for clarification, or decline out-of-scope queries — before touching the database  |
| **Hybrid search**          | Queries are matched via a `hybrid_search` RPC combining pgvector cosine similarity with full-text search (RRF fusion)   |
| **RAG generation**         | Top retrieved chunks are assembled into a grounded prompt; the LLM can only answer from what the papers say             |
| **Streaming answers**      | Responses are streamed token-by-token via `generate_content_stream` for a real-time feel                                |
| **Multi-hop reasoning**    | Two-pass retrieval: Gemini extracts a related concept from pass-1 results and runs a second retrieval to deepen coverage |
| **Confidence indicator**   | Cosine similarity scores drive a High/Medium/Low confidence label shown with every answer                               |
| **Comparison mode**        | Run two queries side-by-side; Gemini contrasts what the papers say about each topic                                     |
| **Action buttons**         | After any answer: summarise in 3 bullets, find open problems, explain for students, or explore related concepts         |
| **Student Mode**           | Sidebar toggle that appends an undergraduate-friendly explanation request to every Gemini prompt                        |
| **Category filter**        | Filter retrieval and the Papers Database by `cs.AI / cs.LG / cs.CL / cs.CV` or q-fin categories                        |
| **PDF upload**             | Upload any PDF for in-memory Q&A — sentence-chunked and embedded the same way as indexed papers                         |
| **Reading List**           | Save papers with one click; export all as BibTeX or clear with confirmation                                             |
| **Paper recommendations**  | Reading list centroid embedding → `match_documents` RPC surfaces similar unsaved papers                                 |
| **BibTeX export**          | Per-paper and bulk BibTeX generation (`@misc{arxiv_YEAR_slug}` format)                                                  |
| **Trending This Week**     | Shows papers indexed in the last 7 days (falls back to 30), with a category bar chart                                   |
| **Weekly Digest**          | One-click Gemini summary of what's new in AI/ML this week based on recent paper titles                                  |
| **Weekly email alerts**    | Subscribers receive a personalised digest of matching new papers every Monday via SendGrid                              |
| **User feedback**          | Thumbs-up / thumbs-down ratings logged to Supabase `feedback` table after every answer                                  |
| **Session Analytics**      | Query history table, confidence line chart, category usage bar chart, summary metrics                                   |
| **Top 3 sources sidebar**  | Most relevant papers shown in the sidebar with similarity bars and ArXiv abstract links                                  |
| **Auto model discovery**   | Probes Gemini models newest-first at startup; survives Google deprecations automatically                                 |
| **Idempotent ETL**         | Re-running the pipeline skips already-indexed papers; fetches cs.AI, cs.LG, cs.CL, cs.CV + q-fin subcategories         |
| **Daily automation**       | GitHub Actions cron runs the ETL every day at 02:00 UTC and pushes new papers into the vector store                    |

---

## Architecture

```
ArXiv API  ──(daily)──►  ETL Pipeline  ──►  Supabase / pgvector
                              │
                     PDF → sentence-boundary chunks (~800 char)
                     → embeddings (384-dim)
                     → category stored in metadata
                              │
User query ──► Gemini router  ──► embed query ──► hybrid search (vector + full-text)
                              │
                    top-5 diverse chunks (deduplicated, one per paper first)
                    optionally category-filtered
                              │
            ┌─────── multi-hop? ──────────┐
            │ pass 1 context              │
            │ Gemini extracts concept     │
            │ pass 2 retrieval            │
            └─────────────────────────────┘
                              │
                    Gemini streaming  ──►  grounded answer + sources
```

1. **Extract** — ArXiv REST API, sorted by submission date descending; categories `cs.AI OR cs.LG OR cs.CL OR cs.CV OR q-fin.ST OR q-fin.CP OR q-fin.PM OR q-fin.TR OR q-fin.RM OR q-fin.MF`; exponential-backoff retry on timeouts and 503s
2. **Transform** — `pypdf` text extraction → NLTK sentence-boundary chunking (target 800 chars, min 100 chars) → `all-MiniLM-L6-v2` embeddings (384-dim); primary category stored in metadata
3. **Load** — Supabase PostgreSQL with `pgvector`; duplicate chunks are skipped on re-runs via RPC batch insert
4. **Route** — Gemini classifies each query as _search / clarify / out-of-scope_ using a structured JSON prompt
5. **Retrieve** — `hybrid_search` RPC (pgvector cosine + full-text, RRF fusion); falls back to pure vector `match_documents`; results deduplicated and diversified across papers; optionally filtered by category
6. **Generate** — Retrieved chunks + question → Gemini streaming prompt → answer grounded in paper text; Student Mode injects an undergraduate-friendly suffix into every prompt

---

## App Layout (5 tabs)

| Tab                    | Description                                                                      |
| ---------------------- | -------------------------------------------------------------------------------- |
| **Ask a Question**     | Agentic search with comparison mode, multi-hop toggle, 4 action buttons, feedback, Save + BibTeX per source |
| **Trending This Week** | Recent papers, category breakdown chart, one-click Weekly Digest                 |
| **Reading List**       | Saved papers, paper recommendations, individual Remove, Export All as BibTeX, Clear All |
| **Papers Database**    | Full paper index with title search, year-range slider, category filter           |
| **Analytics**          | Query history, confidence line chart, category usage chart, summary metrics      |

---

## Tech Stack

| Layer             | Technology                                                           |
| ----------------- | -------------------------------------------------------------------- |
| LLM & routing     | Google Gemini (`google-genai` SDK, auto-discovers available model)   |
| Vector store      | Supabase — PostgreSQL + `pgvector`                                   |
| Search            | `hybrid_search` RPC (pgvector + full-text, RRF); fallback vector RPC |
| Embedding model   | `sentence-transformers/all-MiniLM-L6-v2` (384-dim, runs on CPU)     |
| Frontend          | Streamlit                                                            |
| Email alerts      | SendGrid (weekly digest, Monday 08:00 UTC)                           |
| ETL automation    | GitHub Actions (cron, daily at 02:00 UTC)                            |
| Language          | Python 3.13                                                          |

---

## How to Run Locally

1. **Clone the repo**

   ```bash
   git clone https://github.com/RimaNafougui/ArxivSearchEngine
   cd ArxivSearchEngine
   ```

2. **Install dependencies**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Create a `.env` file** with your credentials

   ```
   SUPABASE_URL=https://your-project.supabase.co
   SUPABASE_KEY=your-supabase-service-role-key
   GOOGLE_API_KEY=your-google-ai-studio-key

   # Optional — only needed for weekly email alerts
   SENDGRID_API_KEY=your-sendgrid-api-key
   ALERT_FROM_EMAIL=alerts@yourdomain.com
   ```

4. **Set up the Supabase database** — run `supabase_migrations.sql` once in the Supabase SQL editor, or apply the minimum manually:

   ```sql
   -- Ensure correct vector dimension
   ALTER TABLE public.documents
     ALTER COLUMN embedding TYPE vector(384);

   -- Pure vector similarity search (fallback)
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

   The `hybrid_search` RPC (vector + full-text with RRF fusion) is defined in `supabase_migrations.sql`.

5. **Run the ETL pipeline** (populates the database)

   ```bash
   python3 etl_pipeline.py
   ```

6. **Start the app**
   ```bash
   streamlit run app.py
   ```

---

## Project Structure

```
├── app.py                    # Streamlit UI — 5 tabs, sidebar, RAG pipeline
├── etl_pipeline.py           # Extract → Transform → Load (ArXiv → pgvector)
├── send_alerts.py            # Weekly email digest via SendGrid
├── check_models.py           # Lists available Gemini models for debugging
├── supabase_migrations.sql   # DB setup: feedback, paper_alerts tables, hybrid_search RPC
├── requirements.txt
├── ARCHITECTURE.md           # Deep-dive technical design document
├── .streamlit/
│   └── config.toml           # Dark academic theme (Hugging Face Spaces)
├── .github/workflows/
│   ├── weekly_update.yml     # GitHub Actions ETL cron (daily, 02:00 UTC)
│   └── paper_alerts.yml      # GitHub Actions email alerts cron (Monday, 08:00 UTC)
└── .devcontainer/
    └── devcontainer.json     # GitHub Codespaces config
```
