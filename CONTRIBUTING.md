# Contributing to ArXiv RAG Research Assistant

Thank you for your interest in contributing! This is an open project and contributions of all kinds are welcome — bug fixes, new features, documentation improvements, and more.

---

## Getting Started

1. **Fork** the repository and clone your fork:

   ```bash
   git clone https://github.com/RimaNafougui/arxivsearchengine.git
   cd arxivsearchengine
   ```

2. **Create a virtual environment** and install dependencies:

   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Copy the environment template** and fill in your credentials:

   ```bash
   cp .env.example .env
   # Edit .env with your Supabase URL, Supabase key, and Google API key
   ```

4. **Create a feature branch** from `main`:
   ```bash
   git checkout -b feat/your-feature-name
   ```

---

## What to Work On

Check the [Issues](../../issues) tab for open tasks. Contributions are especially welcome in these areas:

| Area                 | Examples                                                                    |
| -------------------- | --------------------------------------------------------------------------- |
| **New data sources** | Add support for Semantic Scholar, PubMed, or other preprint servers         |
| **Search quality**   | Experiment with different embedding models or hybrid search (BM25 + vector) |
| **UI improvements**  | Better paper cards, keyboard shortcuts, dark/light theme toggle             |
| **ETL robustness**   | Better error handling, retry logic, incremental updates                     |
| **Testing**          | Unit tests for retrieval logic, ETL pipeline, and prompt formatting         |
| **Documentation**    | Tutorials, architecture diagrams, example notebooks                         |

---

## Development Guidelines

### Code style

- Follow the existing style (PEP 8, no unnecessary type annotations on unchanged code)
- Keep functions small and focused — the existing helpers (`retrieve_documents`, `build_answer`, etc.) are good references
- Add comments only where the logic isn't self-evident

### Streamlit conventions

- Persist results in `st.session_state` so the UI doesn't reset on every interaction
- Use `@st.cache_data` for any function that fetches data from Supabase or an external API
- Avoid calling `st.rerun()` unless strictly necessary

### Commits

- Use imperative commit messages: `Add BibTeX export for reading list`, not `Added BibTeX export`
- Keep commits focused — one logical change per commit

---

## Pull Request Process

1. Make sure the app runs without errors: `streamlit run app.py`
2. If you changed the ETL pipeline, test it against a small batch first
3. Update `README.md` if your change adds or removes a feature
4. Open a pull request against `main` with a clear description of what you changed and why

PRs will be reviewed within a few days. Feedback will be constructive — if a change needs adjustment, the review will say why.

---

## Reporting Bugs

Open an [issue](../../issues/new) and include:

- A clear description of the problem
- Steps to reproduce it
- What you expected vs. what happened
- Relevant error messages or screenshots

---

## Environment Variables Reference

| Variable         | Description                                           |
| ---------------- | ----------------------------------------------------- |
| `SUPABASE_URL`   | Your Supabase project URL (`https://xxx.supabase.co`) |
| `SUPABASE_KEY`   | Supabase service role key (keep this secret)          |
| `GOOGLE_API_KEY` | Google AI Studio key for Gemini access                |

---

## Code of Conduct

Be respectful. This project follows the [Contributor Covenant](https://www.contributor-covenant.org/version/2/1/code_of_conduct/) code of conduct.
