# ArXiv RAG Assistant (Retrieval-Augmented Generation)

A **full-stack AI application** that turns the ArXiv research database into a conversational assistant. It doesn't just find papers; it reads them and uses **Google Gemini** to answer questions with citations.

### Features
* **Automated Data Pipeline:** A self-healing ETL pipeline runs weekly via **GitHub Actions** to fetch, chunk, and embed new AI research papers.
* **Semantic Search:** Uses **Vector Embeddings** (Supabase + pgvector) to understand the *meaning* of a query, not just keywords.
* **RAG Architecture:** Retrieves relevant technical context and feeds it to **Google Gemini Pro** to generate accurate, cited answers.
* **Interactive UI:** A deployed **Streamlit** interface for real-time Q&A.

### Architecture


1.  **Ingestion (ETL):** Python script fetches PDFs from ArXiv $\rightarrow$ Chunks text $\rightarrow$ Generates Embeddings (`all-MiniLM-L6-v2`).
2.  **Storage:** Vectors stored in **Supabase** (PostgreSQL) with `pgvector` indexing.
3.  **Retrieval:** User asks a question $\rightarrow$ System performs Cosine Similarity search to find top 5 relevant chunks.
4.  **Generation:** Top chunks + Question are sent to **Gemini Pro LLM** $\rightarrow$ AI summarizes the answer based *only* on the provided papers.

### 🛠 Tech Stack
* **LLM:** Google Gemini Pro
* **Vector DB:** Supabase (PostgreSQL + pgvector)
* **Orchestration:** GitHub Actions (Cron Automation)
* **Embedding Model:** SentenceTransformers (HuggingFace)
* **Frontend:** Streamlit
* **Language:** Python 3.13

### How to Run Locally
1.  **Clone the repo:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/arxiv-search-engine.git](https://github.com/YOUR_USERNAME/arxiv-search-engine.git)
    cd arxiv-search-engine
    ```
2.  **Install dependencies:**
    ```bash
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```
3.  **Set up secrets:**
    Create a `.env` file with `SUPABASE_URL`, `SUPABASE_KEY`, and `GOOGLE_API_KEY`.
4.  **Run the App:**
    ```bash
    streamlit run app.py
    ```
