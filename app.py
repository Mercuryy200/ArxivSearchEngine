import streamlit as st
from sentence_transformers import SentenceTransformer
from supabase import create_client
from google import genai
from google.genai import types
import os
import json
import re
from dotenv import load_dotenv

# ── 1. SETUP ──────────────────────────────────────────────────────────────────
load_dotenv()
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key  = os.environ.get("SUPABASE_KEY")
google_key    = os.environ.get("GOOGLE_API_KEY")

supabase = create_client(supabase_url, supabase_key)
gemini   = genai.Client(api_key=google_key)

# Candidates tried newest-first.  The first one that accepts a live
# generation call (not just appears in models.list) is used.
_MODEL_CANDIDATES = [
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash-exp",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    "gemini-1.5-flash-latest",
]

# ── 2. CACHED RESOURCES ───────────────────────────────────────────────────────
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')


@st.cache_resource
def discover_gemini_model() -> str:
    """
    Return the first model in _MODEL_CANDIDATES that actually accepts a
    generation call for this API key.

    models.list() is intentionally skipped: deprecated models still appear
    in the list but return 404 on generate_content.  A cheap 1-token probe
    is the only reliable test.  Result is cached for the app's lifetime.
    """
    for name in _MODEL_CANDIDATES:
        try:
            gemini.models.generate_content(
                model=name,
                contents="hi",
                config=types.GenerateContentConfig(max_output_tokens=1),
            )
            return name
        except Exception:
            continue
    return _MODEL_CANDIDATES[-1]


embedding_model = load_embedding_model()
GEMINI_MODEL    = discover_gemini_model()

# ── 3. HELPER FUNCTIONS ───────────────────────────────────────────────────────

def retrieve_documents(query: str, threshold: float = 0.3, count: int = 5) -> list:
    """
    Cosine-similarity search with deduplication and paper-level diversity.

    Strategy (all from a single over-fetched batch):
      1. Remove exact-content duplicates (repeated ETL runs can insert the
         same chunk multiple times).
      2. Pick the best-scoring chunk from each unique paper first, then fill
         any remaining slots with the next-best chunks regardless of paper.
    This ensures the context window spans multiple papers when they exist,
    rather than returning N chunks from the single closest paper.
    """
    query_vector = embedding_model.encode(query).tolist()
    resp = supabase.rpc("match_documents", {
        "query_embedding": query_vector,
        "match_threshold": threshold,
        "match_count": count * 8,
    }).execute()

    # Step 1 — deduplicate by content fingerprint
    seen_fp: set[str] = set()
    deduped: list     = []
    for doc in (resp.data or []):
        fp = doc["content"][:150]
        if fp not in seen_fp:
            seen_fp.add(fp)
            deduped.append(doc)

    # Step 2 — one best chunk per paper, then overflow chunks
    best_per_paper: dict[str, dict] = {}
    overflow:       list            = []
    for doc in deduped:
        title = doc["metadata"]["title"]
        if title not in best_per_paper:
            best_per_paper[title] = doc
        else:
            overflow.append(doc)

    return (list(best_per_paper.values()) + overflow)[:count]


def build_answer(user_query: str, matches: list) -> str:
    """Ask Gemini to synthesise a grounded answer from retrieved context."""
    context_text = "\n\n".join(
        f"[Source: {doc['metadata']['title']}]\n{doc['content']}"
        for doc in matches
    )
    prompt = f"""You are a helpful AI research assistant. Answer the Question below \
using the research paper excerpts in the Context.

Rules:
- Synthesise an answer from whatever relevant information exists in the Context.
- If the Context only partially addresses the question, share what IS there and \
note what is missing.
- Only reply "I couldn't find relevant information in the retrieved papers." if \
the Context contains absolutely nothing related to the question.
- Never invent facts that are not supported by the Context.

Context:
{context_text}

Question: {user_query}

Answer:"""
    return gemini.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    ).text


def run_agent(user_query: str) -> tuple[str, dict]:
    """
    Agentic routing via a structured JSON prompt.

    Asks Gemini to return exactly one of:
      {"action":"search",    "query":"<refined query>"}
      {"action":"clarify",   "question":"<clarifying question>"}
      {"action":"no_results","reason":"<scope explanation>"}

    Returns (action_name, action_args) where action_name is one of:
      "search_papers" | "ask_clarification" | "report_no_results"
    """
    routing_prompt = f"""You are a router for an ArXiv AI/ML research assistant.
The database covers: deep learning, neural networks, transformers, NLP,
computer vision, reinforcement learning, generative models, LLMs, diffusion models.

Respond with EXACTLY ONE JSON object — no extra text, no markdown fences:

{{"action":"search",    "query":"<optimised search query>"}}
{{"action":"clarify",   "question":"<specific clarifying question>"}}
{{"action":"no_results","reason":"<why out of scope + what IS covered>"}}

Rules:
- "search"     → query is specific and related to AI/ML research
- "clarify"    → query is vague or ambiguous
- "no_results" → query is clearly outside AI/ML scope (cooking, sports, etc.)

User query: "{user_query}"

JSON:"""

    try:
        raw = gemini.models.generate_content(
            model=GEMINI_MODEL,
            contents=routing_prompt,
        ).text
    except Exception:
        return "search_papers", {"refined_query": user_query}

    match = re.search(r'\{[^{}]+\}', raw, re.DOTALL)
    if match:
        try:
            data   = json.loads(match.group())
            action = data.get("action", "search")
            if action == "search":
                return "search_papers", {"refined_query": data.get("query", user_query)}
            if action == "clarify":
                return "ask_clarification", {"question": data.get("question", "")}
            if action == "no_results":
                return "report_no_results", {"explanation": data.get("reason", "")}
        except (json.JSONDecodeError, KeyError):
            pass

    return "search_papers", {"refined_query": user_query}


def arxiv_abstract_url(pdf_url: str) -> str:
    """Convert an ArXiv PDF URL to its abstract page URL."""
    url = pdf_url.replace("arxiv.org/pdf/", "arxiv.org/abs/")
    if url.endswith(".pdf"):
        url = url[:-4]
    return url


def confidence_badge(matches: list) -> tuple[str, float, float, str]:
    """Return (label, max_sim, avg_sim, description) from similarity scores."""
    if not matches:
        return "⚪ No Signal", 0.0, 0.0, "No documents retrieved."

    sims    = [m["similarity"] for m in matches]
    max_sim = max(sims)
    avg_sim = sum(sims) / len(sims)
    desc    = f"Best match: {max_sim:.2f} · Average: {avg_sim:.2f} across {len(matches)} chunks"

    if max_sim >= 0.70:
        label = "🟢 High Confidence"
    elif max_sim >= 0.50:
        label = "🟡 Medium Confidence"
    else:
        label = "🔴 Low Confidence"

    return label, max_sim, avg_sim, desc


@st.cache_data(ttl=300)
def fetch_all_papers() -> list[dict]:
    """
    Return one metadata dict per unique paper in the database.

    Paginates through ALL rows in chunks of 1 000 so that the 1 000-row
    default Supabase limit never silently drops papers whose chunks happen
    to fall outside the first page.
    Only the metadata column is fetched (no content, no embeddings).
    Cached for 5 minutes so repeated tab switches are instant.
    """
    seen:      set[str]   = set()
    papers:    list[dict] = []
    page_size: int        = 1000
    offset:    int        = 0

    while True:
        resp = (
            supabase.table("documents")
            .select("metadata")
            .range(offset, offset + page_size - 1)
            .execute()
        )
        batch = resp.data or []
        for row in batch:
            meta  = row.get("metadata") or {}
            title = meta.get("title", "").strip()
            if title and title not in seen:
                seen.add(title)
                papers.append(meta)
        if len(batch) < page_size:
            break                       # last page reached
        offset += page_size

    # Sort newest-first.  ISO date strings ("2024-11-02T...") are
    # lexicographically comparable, so reverse string sort = newest first.
    # Papers with a missing date fall to the end.
    return sorted(papers, key=lambda p: p.get("published", ""), reverse=True)


# ── 4. PAGE CONFIG ────────────────────────────────────────────────────────────
st.set_page_config(page_title="ArXiv RAG Assistant", layout="wide")
st.title("🤖 ArXiv Research Assistant")

tab_chat, tab_papers = st.tabs(["💬 Ask a Question", "📄 Papers Database"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CHAT
# ══════════════════════════════════════════════════════════════════════════════
with tab_chat:
    st.markdown(
        """
Ask anything about AI/ML research. The assistant will:
1. **Decide** the best action — search, ask for clarification, or explain why it can't help
2. **Retrieve** semantically relevant paper chunks from the vector database
3. **Answer** based *only* on those papers — no hallucination
"""
    )

    query = st.text_input(
        "Ask a question about AI/ML research:",
        placeholder="e.g., How does multi-head attention work in Transformers?",
    )

    if query:

        with st.spinner("Analysing your query…"):
            action, args = run_agent(query)

        if action == "ask_clarification":
            st.info(
                f"**Before I search, could you clarify?**\n\n"
                f"{args.get('question', 'Could you provide more detail about what you are looking for?')}"
            )

        elif action == "report_no_results":
            st.warning(
                f"**This topic appears to be outside my knowledge base.**\n\n"
                f"{args.get('explanation', 'The database covers AI/ML research papers only.')}"
            )

        else:
            refined_query = args.get("refined_query", query)

            with st.spinner("Searching the vector database…"):
                matches = retrieve_documents(refined_query)

            if not matches:
                st.warning(
                    "No relevant papers were found for that query. "
                    "Try rephrasing, or ask about a different AI/ML topic."
                )

            else:
                # ── Sidebar: top-3 source papers ─────────────────────────────
                with st.sidebar:
                    st.header("📚 Top Sources")
                    st.caption("Papers most relevant to your query")

                    seen: dict[str, dict] = {}
                    for m in matches:
                        t = m["metadata"]["title"]
                        if t not in seen or m["similarity"] > seen[t]["similarity"]:
                            seen[t] = m
                    top_papers = sorted(
                        seen.values(), key=lambda x: x["similarity"], reverse=True
                    )[:3]

                    for i, paper in enumerate(top_papers, 1):
                        title   = paper["metadata"]["title"]
                        pdf_url = paper["metadata"].get("url", "")
                        abs_url = arxiv_abstract_url(pdf_url)
                        sim     = paper["similarity"]

                        display_title = title if len(title) <= 55 else title[:52] + "…"
                        st.markdown(f"**{i}. {display_title}**")
                        st.progress(float(sim), text=f"Similarity: {sim:.2f}")
                        if abs_url:
                            st.markdown(f"[View Abstract ↗]({abs_url})")
                        if i < len(top_papers):
                            st.divider()

                # ── Confidence indicator ──────────────────────────────────────
                label, max_sim, avg_sim, desc = confidence_badge(matches)
                col_badge, col_desc = st.columns([1, 3])
                with col_badge:
                    st.metric(
                        label="Retrieval Confidence",
                        value=f"{max_sim:.2f}",
                        delta=f"{avg_sim:.2f} avg",
                        help="Cosine similarity between your query and the best-matching "
                             "paper chunk. Higher = stronger semantic match.",
                    )
                with col_desc:
                    st.markdown(f"**{label}**")
                    st.caption(desc)

                # ── Generate answer ───────────────────────────────────────────
                with st.spinner("Reading papers and generating answer…"):
                    try:
                        answer = build_answer(query, matches)
                        st.success("Answer generated from research papers:")
                        st.markdown(f"### 💡 {answer}")
                    except Exception as e:
                        st.error(f"Error generating answer: {e}")

                # ── Expandable full source listing ────────────────────────────
                with st.expander("📖 View All Source Documents"):
                    for match in matches:
                        sim     = match["similarity"]
                        title   = match["metadata"]["title"]
                        pdf_url = match["metadata"].get("url", "")
                        abs_url = arxiv_abstract_url(pdf_url)

                        st.markdown(f"**{title}** — Similarity: `{sim:.2f}`")
                        st.progress(float(sim))
                        st.info(match["content"])

                        link_col1, link_col2 = st.columns(2)
                        with link_col1:
                            if pdf_url:
                                st.markdown(f"[📄 PDF]({pdf_url})")
                        with link_col2:
                            if abs_url:
                                st.markdown(f"[🔗 Abstract]({abs_url})")
                        st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PAPERS DATABASE
# ══════════════════════════════════════════════════════════════════════════════
with tab_papers:
    st.subheader("All Papers in the Database")

    with st.spinner("Loading papers…"):
        all_papers = fetch_all_papers()

    # ── Search input ──────────────────────────────────────────────────────────
    search = st.text_input(
        "🔍 Search by title",
        placeholder="e.g., transformer, diffusion, reinforcement…",
    )

    filtered = (
        [p for p in all_papers if search.strip().lower() in p.get("title", "").lower()]
        if search.strip()
        else all_papers
    )

    st.caption(f"Showing **{len(filtered)}** of **{len(all_papers)}** papers")
    st.divider()

    # ── Paper list ────────────────────────────────────────────────────────────
    if not filtered:
        st.info("No papers match your search. Try a different keyword.")
    else:
        for paper in filtered:
            title     = paper.get("title", "Untitled")
            pdf_url   = paper.get("url", "")
            published = paper.get("published", "")
            abs_url   = arxiv_abstract_url(pdf_url) if pdf_url else ""

            # Published year extracted from ISO date string (e.g. "2023-11-02T00:00:00Z")
            year = published[:4] if published else "Unknown"

            col_title, col_year, col_links = st.columns([5, 1, 2])
            with col_title:
                st.markdown(f"**{title}**")
            with col_year:
                st.markdown(f"📅 {year}")
            with col_links:
                links = []
                if abs_url:
                    links.append(f"[Abstract ↗]({abs_url})")
                if pdf_url:
                    links.append(f"[PDF ↗]({pdf_url})")
                st.markdown(" · ".join(links))

            st.divider()
