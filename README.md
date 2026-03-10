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

# ArXiv RAG Assistant

A **full-stack AI application** that turns the ArXiv research database into a conversational assistant. Ask a question about AI/ML research and the system retrieves the most relevant paper chunks, then uses **Google Gemini** to generate a grounded answer — no hallucination.

![App Demo](/images/demo.png)

---

## Features

| Feature | Description |
|---|---|
| **Agentic routing** | Gemini decides whether to search, ask a clarifying question, or politely decline out-of-scope queries — before touching the database |
| **Semantic search** | Queries are encoded with `all-MiniLM-L6-v2` and matched against stored embeddings using cosine similarity (pgvector) |
| **RAG generation** | Top retrieved chunks are assembled into a grounded prompt; the LLM can only answer from what the papers say |
| **Confidence indicator** | Cosine similarity scores drive a 🟢/🟡/🔴 confidence badge shown with every answer |
| **Top 3 sources sidebar** | The three most relevant papers appear in the sidebar with similarity bars and direct ArXiv abstract links |
| **Papers Database tab** | Browse or search the full list of indexed papers by title, sorted newest first, with year and links |
| **Auto model discovery** | At startup the app probes a list of Gemini models newest-first and uses the first one that responds — survives Google deprecations automatically |
| **Idempotent ETL** | Re-running the pipeline skips already-indexed papers; newest ArXiv papers are fetched first (`sortBy=submittedDate`) |
| **Weekly automation** | GitHub Actions cron runs the ETL every Sunday and pushes new papers into the vector store |

---

## Architecture

```
ArXiv API  ──(weekly)──►  ETL Pipeline  ──►  Supabase / pgvector
                              │
                     PDF → chunks (500 char)
                     → embeddings (384-dim)
                              │
User query ──► Gemini router  ──► embed query ──► cosine search
                              │
                    top-5 diverse chunks
                              │
                    Gemini generation  ──►  grounded answer + sources
```

1. **Extract** — ArXiv REST API, sorted by submission date descending, category `cs.AI`
2. **Transform** — `pypdf` text extraction → 500-char chunks (50-char overlap) → `all-MiniLM-L6-v2` embeddings
3. **Load** — Supabase PostgreSQL with `pgvector`; duplicate chunks are skipped on re-runs
4. **Route** — Gemini classifies each query as *search / clarify / out-of-scope* using a structured JSON prompt
5. **Retrieve** — pgvector cosine similarity search; results are deduplicated and diversified across papers
6. **Generate** — Retrieved chunks + question → Gemini prompt → answer grounded in paper text

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM & routing | Google Gemini (`google-genai` SDK, auto-discovers available model) |
| Vector store | Supabase — PostgreSQL + `pgvector` |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` (384-dim, runs on CPU) |
| Frontend | Streamlit |
| ETL automation | GitHub Actions (cron, weekly) |
| Language | Python 3.13 |

---

## How to Run Locally

1. **Clone the repo**
   ```bash
   git clone https://github.com/YOUR_USERNAME/arxiv-search-engine.git
   cd arxiv-search-engine
   ```

2. **Install dependencies**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Create a `.env` file** with your credentials
   ```
   SUPABASE_URL=https://your-project.supabase.co
   SUPABASE_KEY=your-supabase-service-role-key
   GOOGLE_API_KEY=your-google-ai-studio-key
   ```

4. **Set up the Supabase database** — run this once in the Supabase SQL editor:
   ```sql
   -- Fix vector dimension
   ALTER TABLE public.documents
     ALTER COLUMN embedding TYPE vector(384);

   -- Similarity search function used by the app
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

5. **Run the ETL pipeline** (populates the database)
   ```bash
   python etl_pipeline.py
   ```

6. **Start the app**
   ```bash
   streamlit run app.py
   ```

---

## Project Structure

```
├── app.py                  # Streamlit UI + agentic router + RAG pipeline
├── etl_pipeline.py         # Extract → Transform → Load (ArXiv → pgvector)
├── check_models.py         # Lists available Gemini models for debugging
├── requirements.txt
├── ARCHITECTURE.md         # Deep-dive technical design document
├── .github/workflows/
│   └── weekly_update.yml   # GitHub Actions ETL cron
└── .devcontainer/
    └── devcontainer.json   # GitHub Codespaces config
```
