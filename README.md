# ArXiv Semantic Search Engine

An automated **Data Engineering pipeline** that ingests raw research papers, transforms them into vector embeddings, and enables "meaning-based" search using AI.

### Architecture
* **Extract:** Automated scraper fetches PDFs from ArXiv API (via **GitHub Actions** cron job).
* **Transform:** Chunks text and converts it to 384-dimensional vectors using `sentence-transformers`.
* **Load:** Stores vectors in **Supabase** (PostgreSQL + `pgvector`).
* **Serve:** A **Streamlit** UI that performs cosine similarity search to find relevant answers.

### Tech Stack
* **Language:** Python 3.13
* **Database:** Supabase (PostgreSQL)
* **AI Model:** all-MiniLM-L6-v2 (HuggingFace)
* **Orchestration:** GitHub Actions (Serverless Automation)
* **Frontend:** Streamlit Cloud
