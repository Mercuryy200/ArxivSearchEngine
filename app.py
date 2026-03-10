import streamlit as st
from sentence_transformers import SentenceTransformer
from supabase import create_client
from google import genai
from google.genai import types
import os
import json
import re
import datetime
import pandas as pd
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

def student_suffix() -> str:
    """Return the student-mode prompt injection when toggled on."""
    if st.session_state.get("student_mode"):
        return (
            " Explain this clearly for an undergraduate student"
            " with no prior background."
        )
    return ""


def call_gemini(prompt: str) -> str | None:
    """Single reusable Gemini call. Surfaces errors with st.error()."""
    try:
        return gemini.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        ).text
    except Exception as e:
        st.error(f"Gemini error: {e}")
        return None


def _bibtex_slug(title: str, year: str) -> str:
    words = title.lower().split()[:4]
    slug  = "_".join(w.strip(".,;:!?\"'") for w in words)
    return f"arxiv_{year}_{slug}"


def make_bibtex(paper: dict) -> str:
    title     = paper.get("title", "Unknown")
    url       = paper.get("url", "")
    published = paper.get("published", paper.get("date", ""))
    year      = published[:4] if published else "0000"
    slug      = _bibtex_slug(title, year)
    return (
        f"@misc{{{slug},\n"
        f"  title={{{title}}},\n"
        f"  year={{{year}}},\n"
        f"  url={{{url}}},\n"
        f"  note={{ArXiv preprint}}\n"
        f"}}"
    )


def save_to_reading_list(paper: dict) -> None:
    existing = {p["title"] for p in st.session_state.reading_list}
    if paper.get("title") not in existing:
        st.session_state.reading_list.append(paper)
        st.toast(f"Saved: {paper['title'][:55]}")
    else:
        st.toast("Already in reading list.")


def retrieve_documents(
    query: str,
    threshold: float = 0.3,
    count: int = 5,
    category_filter: list | None = None,
) -> list:
    """
    Cosine-similarity search with deduplication and paper-level diversity.

    Strategy (all from a single over-fetched batch):
      1. Remove exact-content duplicates (repeated ETL runs can insert the
         same chunk multiple times).
      2. Pick the best-scoring chunk from each unique paper first, then fill
         any remaining slots with the next-best chunks regardless of paper.
    This ensures the context window spans multiple papers when they exist,
    rather than returning N chunks from the single closest paper.

    Optional category_filter post-filters results to selected categories.
    """
    try:
        query_vector = embedding_model.encode(query).tolist()
        resp = supabase.rpc("match_documents", {
            "query_embedding": query_vector,
            "match_threshold": threshold,
            "match_count": count * 8,
        }).execute()
    except Exception as e:
        st.error(f"Retrieval error: {e}")
        return []

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

    result = (list(best_per_paper.values()) + overflow)[:count]

    # Step 3 — optional category filter (post-retrieval)
    if category_filter and "All" not in category_filter:
        result = [
            m for m in result
            if m.get("metadata", {}).get("category", "") in category_filter
        ]

    return result


def build_answer(user_query: str, context_text: str) -> str | None:
    """Ask Gemini to synthesise a grounded answer from retrieved context."""
    prompt = (
        "You are a helpful AI research assistant. Answer the Question below "
        "using the research paper excerpts in the Context."
        f"{student_suffix()}\n\n"
        "Rules:\n"
        "- Synthesise an answer from whatever relevant information exists in the Context.\n"
        "- If the Context only partially addresses the question, share what IS there "
        "and note what is missing.\n"
        "- Only reply \"I couldn't find relevant information in the retrieved papers.\" "
        "if the Context contains absolutely nothing related to the question.\n"
        "- Never invent facts that are not supported by the Context.\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {user_query}\n\nAnswer:"
    )
    return call_gemini(prompt)


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


def avg_confidence(matches: list) -> float:
    if not matches:
        return 0.0
    return sum(m.get("similarity", 0) for m in matches) / len(matches)


def context_from_matches(matches: list) -> str:
    return "\n\n".join(
        f"[Source: {m['metadata']['title']}]\n{m['content']}"
        for m in matches
    )


def _categories_from_matches(matches: list) -> list:
    return list({
        m.get("metadata", {}).get("category", "")
        for m in matches
        if m.get("metadata", {}).get("category")
    })


def render_source_paper(match: dict, key_prefix: str) -> None:
    """Render one source paper card with Save and BibTeX buttons."""
    meta     = match.get("metadata", {})
    title    = meta.get("title", "Unknown")
    pdf_url  = meta.get("url", "")
    abs_url  = arxiv_abstract_url(pdf_url) if pdf_url else ""
    sim      = match.get("similarity", 0.0)
    pub      = meta.get("published", "")
    category = meta.get("category", "")

    st.markdown(f"**{title}** — Similarity: `{sim:.2f}`")
    st.caption(f"{category} · {pub[:10]}")
    st.progress(float(sim))
    st.info(match.get("content", "")[:300] + "…")

    link_col1, link_col2 = st.columns(2)
    with link_col1:
        if pdf_url:
            st.markdown(f"[📄 PDF]({pdf_url})")
    with link_col2:
        if abs_url:
            st.markdown(f"[🔗 Abstract]({abs_url})")

    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        if st.button("📌 Save to Reading List", key=f"save_{key_prefix}_{title[:18]}"):
            save_to_reading_list({
                "title":     title,
                "url":       pdf_url,
                "published": pub,
                "category":  category,
            })
    with btn_col2:
        if st.button("📋 Copy BibTeX", key=f"bib_{key_prefix}_{title[:18]}"):
            st.code(
                make_bibtex({"title": title, "url": pdf_url, "published": pub}),
                language="bibtex",
            )
    st.divider()


@st.cache_data(ttl=300)
def fetch_trending_papers(days: int = 7) -> list:
    """Fetch documents published within `days` days. Cached for 5 minutes."""
    try:
        cutoff = (
            datetime.datetime.utcnow() - datetime.timedelta(days=days)
        ).date().isoformat()
        resp = (
            supabase.table("documents")
            .select("metadata")
            .gte("metadata->>published", cutoff)
            .execute()
        )
        return resp.data or []
    except Exception as e:
        st.error(f"Error fetching trending papers: {e}")
        return []


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
        try:
            resp = (
                supabase.table("documents")
                .select("metadata")
                .range(offset, offset + page_size - 1)
                .execute()
            )
        except Exception as e:
            st.error(f"Database error: {e}")
            break
        batch = resp.data or []
        for row in batch:
            meta  = row.get("metadata") or {}
            title = meta.get("title", "").strip()
            if title and title not in seen:
                seen.add(title)
                papers.append(meta)
        if len(batch) < page_size:
            break
        offset += page_size

    return sorted(papers, key=lambda p: p.get("published", ""), reverse=True)


# ── 4. SESSION STATE ──────────────────────────────────────────────────────────
_defaults = {
    "reading_list":   [],    # list[dict] — paper metadata
    "query_history":  [],    # list[dict] — {query, action, confidence, timestamp, categories}
    "student_mode":   False,
    # Tab 1 — regular search result cache (persists across reruns)
    "active_query":   "",
    "active_matches": [],
    "active_context": "",
    "active_answer":  "",
    "action_result":  "",
    "action_label":   "",
    # Tab 1 — comparison mode result cache
    "cmp_answer":     "",
    "cmp_matches1":   [],
    "cmp_matches2":   [],
    # Tab 3 — confirm clear
    "confirm_clear":  False,
}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ── 5. PAGE CONFIG ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ArXiv RAG Research Assistant",
    page_icon="🔬",
    layout="wide",
)


# ── 6. SIDEBAR ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔬 ArXiv RAG")
    st.caption(f"Model: `{GEMINI_MODEL}`")
    st.divider()

    # Student Mode toggle
    st.session_state.student_mode = st.toggle(
        "🎓 Student Mode",
        value=st.session_state.student_mode,
        help=(
            "Appends 'Explain this clearly for an undergraduate student "
            "with no prior background.' to every Gemini prompt."
        ),
    )

    # Category filter
    st.subheader("Category Filter")
    CATEGORIES = ["All", "cs.AI", "cs.LG", "cs.CL", "cs.CV"]
    selected_categories: list = st.multiselect(
        "Show categories", CATEGORIES, default=["All"]
    ) or ["All"]

    st.divider()

    # Session stats
    st.subheader("📊 Session Stats")
    _n_q  = len(st.session_state.query_history)
    _n_sv = len(st.session_state.reading_list)
    _a_cf = (
        sum(_h["confidence"] for _h in st.session_state.query_history) / _n_q
        if _n_q else 0.0
    )
    st.metric("Queries run",    _n_q)
    st.metric("Papers saved",   _n_sv)
    st.metric("Avg confidence", f"{_a_cf:.2f}")

    st.divider()

    # Last 5 queries
    st.subheader("🕑 Recent Queries")
    _recent = st.session_state.query_history[-5:][::-1]
    if _recent:
        for _h in _recent:
            _qt = _h["query"]
            st.caption(f"• {_qt[:48]}…" if len(_qt) > 48 else f"• {_qt}")
    else:
        st.caption("No queries yet.")


# ── 7. MAIN TITLE & TABS ──────────────────────────────────────────────────────
st.title("🔬 ArXiv RAG Research Assistant")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "💬 Ask a Question",
    "📈 Trending This Week",
    "📚 Reading List",
    "🗃 Papers Database",
    "📊 Analytics",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Ask a Question
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown(
        """
Ask anything about AI/ML research. The assistant will:
1. **Decide** the best action — search, ask for clarification, or explain why it can't help
2. **Retrieve** semantically relevant paper chunks from the vector database
3. **Answer** based *only* on those papers — no hallucination
"""
    )

    comparison_mode = st.checkbox("🔀 Comparison Mode")

    # ── Comparison mode ───────────────────────────────────────────────────────
    if comparison_mode:
        col_q1, col_q2 = st.columns(2)
        with col_q1:
            query1 = st.text_input(
                "Query 1",
                placeholder="e.g., How does attention work?",
                key="q1_input",
            )
        with col_q2:
            query2 = st.text_input(
                "Query 2",
                placeholder="e.g., How does convolution work?",
                key="q2_input",
            )

        if st.button("⚡ Compare", type="primary"):
            if query1 and query2:
                with st.spinner("Retrieving papers for both queries…"):
                    m1 = retrieve_documents(query1, category_filter=selected_categories)
                    m2 = retrieve_documents(query2, category_filter=selected_categories)

                ctx1 = context_from_matches(m1)
                ctx2 = context_from_matches(m2)

                cmp_prompt = (
                    f"Compare and contrast what these papers say about "
                    f'"{query1}" versus "{query2}".'
                    f"{student_suffix()}\n\n"
                    f'Papers about "{query1}":\n{ctx1}\n\n'
                    f'Papers about "{query2}":\n{ctx2}\n\n'
                    "Comparison:"
                )
                with st.spinner("Generating comparison…"):
                    cmp_ans = call_gemini(cmp_prompt)

                st.session_state.cmp_answer   = cmp_ans or ""
                st.session_state.cmp_matches1 = m1
                st.session_state.cmp_matches2 = m2

                conf = (avg_confidence(m1) + avg_confidence(m2)) / 2
                st.session_state.query_history.append({
                    "query":      f"{query1} vs {query2}",
                    "action":     "comparison",
                    "confidence": conf,
                    "timestamp":  datetime.datetime.now().isoformat(),
                    "categories": _categories_from_matches(m1 + m2),
                })

        if st.session_state.cmp_answer:
            st.markdown("### 🔀 Comparison")
            st.markdown(st.session_state.cmp_answer)

            with st.expander("📚 Source Documents — Query 1"):
                for _i, _m in enumerate(st.session_state.cmp_matches1):
                    render_source_paper(_m, f"cmp1_{_i}")
            with st.expander("📚 Source Documents — Query 2"):
                for _i, _m in enumerate(st.session_state.cmp_matches2):
                    render_source_paper(_m, f"cmp2_{_i}")

    # ── Regular search ────────────────────────────────────────────────────────
    else:
        query = st.text_input(
            "Ask a question about AI/ML research:",
            placeholder="e.g., How does multi-head attention work in Transformers?",
        )

        if st.button("🔍 Search", type="primary") and query:
            with st.spinner("Analysing your query…"):
                action, args = run_agent(query)

            if action == "ask_clarification":
                st.info(
                    f"**Before I search, could you clarify?**\n\n"
                    f"{args.get('question', 'Could you provide more detail?')}"
                )
                st.session_state.active_answer = ""

            elif action == "report_no_results":
                st.warning(
                    f"**This topic appears to be outside my knowledge base.**\n\n"
                    f"{args.get('explanation', 'The database covers AI/ML research papers only.')}"
                )
                st.session_state.active_answer = ""

            else:
                refined = args.get("refined_query", query)
                with st.spinner("Searching the vector database…"):
                    matches = retrieve_documents(
                        refined, category_filter=selected_categories
                    )

                if not matches:
                    st.warning(
                        "No relevant papers found. Try rephrasing, "
                        "or ask about a different AI/ML topic."
                    )
                    st.session_state.active_answer = ""
                else:
                    ctx = context_from_matches(matches)
                    with st.spinner("Reading papers and generating answer…"):
                        answer = build_answer(query, ctx)

                    st.session_state.active_query   = query
                    st.session_state.active_matches  = matches
                    st.session_state.active_context  = ctx
                    st.session_state.active_answer   = answer or ""
                    st.session_state.action_result   = ""
                    st.session_state.action_label    = ""

                    conf = avg_confidence(matches)
                    st.session_state.query_history.append({
                        "query":      query,
                        "action":     action,
                        "confidence": conf,
                        "timestamp":  datetime.datetime.now().isoformat(),
                        "categories": _categories_from_matches(matches),
                    })

        # Display persisted answer + UI
        if st.session_state.active_answer:
            # ── Confidence indicator ──────────────────────────────────────────
            label, max_sim, avg_sim, desc = confidence_badge(
                st.session_state.active_matches
            )
            col_badge, col_desc = st.columns([1, 3])
            with col_badge:
                st.metric(
                    label="Retrieval Confidence",
                    value=f"{max_sim:.2f}",
                    delta=f"{avg_sim:.2f} avg",
                )
            with col_desc:
                st.markdown(f"**{label}**")
                st.caption(desc)

            st.success("Answer generated from research papers:")
            st.markdown(f"### 💡 {st.session_state.active_answer}")

            # ── Action buttons ────────────────────────────────────────────────
            st.markdown("---")
            b1, b2, b3, b4 = st.columns(4)
            _ctx = st.session_state.active_context
            _q   = st.session_state.active_query

            with b1:
                if st.button("📝 Summarize in 3 bullets"):
                    with st.spinner("Summarizing…"):
                        res = call_gemini(
                            f"Summarize the following research context in exactly "
                            f"3 concise bullet points.{student_suffix()}"
                            f"\n\nContext:\n{_ctx}\n\nBullet summary:"
                        )
                    st.session_state.action_result = res or ""
                    st.session_state.action_label  = "📝 3-Bullet Summary"

            with b2:
                if st.button("🔍 Find open problems"):
                    with st.spinner("Identifying open problems…"):
                        res = call_gemini(
                            "Based on the following research papers, what are the main "
                            f"unsolved problems and open challenges identified?{student_suffix()}"
                            f"\n\nContext:\n{_ctx}\n\nOpen problems:"
                        )
                    st.session_state.action_result = res or ""
                    st.session_state.action_label  = "🔍 Open Problems"

            with b3:
                # Always uses student-mode phrasing regardless of toggle
                if st.button("🎓 Explain for students"):
                    with st.spinner("Simplifying…"):
                        res = call_gemini(
                            "Explain the following research in simple terms for an "
                            "undergraduate student with no prior background.\n\n"
                            f"Context:\n{_ctx}\n\n"
                            f"Question: {_q}\n\nSimple explanation:"
                        )
                    st.session_state.action_result = res or ""
                    st.session_state.action_label  = "🎓 Student Explanation"

            with b4:
                if st.button("🌐 Related concepts"):
                    with st.spinner("Finding related concepts…"):
                        res = call_gemini(
                            "Based on the following research papers, what adjacent "
                            "topics, related concepts, and further reading areas do "
                            f"these papers point toward?{student_suffix()}"
                            f"\n\nContext:\n{_ctx}\n\nRelated concepts:"
                        )
                    st.session_state.action_result = res or ""
                    st.session_state.action_label  = "🌐 Related Concepts"

            if st.session_state.action_result:
                st.markdown(f"#### {st.session_state.action_label}")
                st.markdown(st.session_state.action_result)

            # ── Top-3 sources in sidebar ──────────────────────────────────────
            with st.sidebar:
                st.header("📚 Top Sources")
                st.caption("Papers most relevant to your query")
                _seen_s: dict[str, dict] = {}
                for _m in st.session_state.active_matches:
                    _t = _m["metadata"]["title"]
                    if _t not in _seen_s or _m["similarity"] > _seen_s[_t]["similarity"]:
                        _seen_s[_t] = _m
                _top3 = sorted(
                    _seen_s.values(), key=lambda x: x["similarity"], reverse=True
                )[:3]
                for _i, _paper in enumerate(_top3, 1):
                    _pt  = _paper["metadata"]["title"]
                    _pu  = _paper["metadata"].get("url", "")
                    _au  = arxiv_abstract_url(_pu)
                    _sim = _paper["similarity"]
                    st.markdown(
                        f"**{_i}. {_pt[:55]}{'…' if len(_pt) > 55 else ''}**"
                    )
                    st.progress(float(_sim), text=f"Similarity: {_sim:.2f}")
                    if _au:
                        st.markdown(f"[View Abstract ↗]({_au})")
                    if _i < len(_top3):
                        st.divider()

            # ── Source documents expander ─────────────────────────────────────
            with st.expander("📖 View All Source Documents"):
                for _i, _match in enumerate(st.session_state.active_matches):
                    render_source_paper(_match, f"qa_{_i}")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Trending This Week
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Trending This Week")

    trending_raw  = fetch_trending_papers(7)
    fallback_used = len(trending_raw) < 5

    if fallback_used:
        st.info("Fewer than 5 papers in the last 7 days — showing last 30 days instead.")
        trending_raw = fetch_trending_papers(30)

    # Deduplicate by title
    _seen_t: set  = set()
    trending: list = []
    for _row in trending_raw:
        _t = _row.get("metadata", {}).get("title", "")
        if _t and _t not in _seen_t:
            _seen_t.add(_t)
            trending.append(_row)

    st.metric(
        "Papers found",
        len(trending),
        delta="last 30 days" if fallback_used else "last 7 days",
    )

    for _row in trending:
        _meta  = _row.get("metadata", {})
        _title = _meta.get("title", "Unknown")
        _url   = _meta.get("url", "#")
        _cat   = _meta.get("category", "")
        _pub   = _meta.get("published", "")[:10]
        st.markdown(f"- **[{_title}]({_url})** `{_cat}` _{_pub}_")

    # Category breakdown bar chart
    if trending:
        st.subheader("Category Breakdown")
        _cat_cnt: dict = {}
        for _row in trending:
            _c = _row.get("metadata", {}).get("category", "unknown")
            _cat_cnt[_c] = _cat_cnt.get(_c, 0) + 1
        if _cat_cnt:
            st.bar_chart(pd.DataFrame({"Papers": _cat_cnt}))

    # Weekly Digest
    st.divider()
    st.subheader("Weekly Digest")
    if st.button("Generate Weekly Digest"):
        if not trending:
            st.warning("No recent papers to summarize.")
        else:
            _titles_txt = "\n".join(
                f"- {_r.get('metadata', {}).get('title', 'Unknown')}"
                for _r in trending
            )
            _digest_prompt = (
                "Write a 150-word research digest summarizing what is new this week "
                f"in AI/ML based on these paper titles.{student_suffix()}\n\n"
                f"Papers:\n{_titles_txt}\n\nDigest:"
            )
            with st.spinner("Writing digest…"):
                _digest = call_gemini(_digest_prompt)
            if _digest:
                st.markdown(_digest)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Reading List
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("My Reading List")

    if not st.session_state.reading_list:
        st.info(
            "No papers saved yet. Use the '📌 Save to Reading List' buttons "
            "when viewing search results."
        )
    else:
        for _i, _paper in enumerate(list(st.session_state.reading_list)):
            _c_info, _c_rm = st.columns([5, 1])
            with _c_info:
                st.markdown(f"**{_paper.get('title', 'Unknown')}**")
                _pub = _paper.get("published", "")[:10]
                _cat = _paper.get("category", "")
                _url = _paper.get("url", "#")
                st.caption(f"{_cat} · {_pub} · [PDF]({_url})")
            with _c_rm:
                if st.button("🗑", key=f"rm_{_i}", help="Remove"):
                    st.session_state.reading_list.pop(_i)
                    st.rerun()
            st.divider()

        st.subheader("Export")
        if st.button("📤 Export All as BibTeX"):
            _all_bib = "\n\n".join(
                make_bibtex(_p) for _p in st.session_state.reading_list
            )
            st.code(_all_bib, language="bibtex")

        # Clear all — requires confirmation step
        if st.button("🗑 Clear All"):
            st.session_state.confirm_clear = True

        if st.session_state.confirm_clear:
            st.warning("This will remove all saved papers. Are you sure?")
            _ca, _cb = st.columns(2)
            with _ca:
                if st.button("✅ Yes, clear all", type="primary"):
                    st.session_state.reading_list  = []
                    st.session_state.confirm_clear = False
                    st.rerun()
            with _cb:
                if st.button("❌ Cancel"):
                    st.session_state.confirm_clear = False
                    st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Papers Database
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("All Papers in the Database")

    with st.spinner("Loading papers…"):
        all_papers = fetch_all_papers()

    _search_title = st.text_input(
        "🔍 Search by title",
        placeholder="e.g., transformer, diffusion, reinforcement…",
    )
    _year_min, _year_max = st.slider("Year range", 2018, 2025, (2020, 2025))

    # Apply title search
    _filtered = (
        [p for p in all_papers if _search_title.strip().lower() in p.get("title", "").lower()]
        if _search_title.strip()
        else list(all_papers)
    )

    # Apply year filter
    _filtered = [
        p for p in _filtered
        if _year_min <= int(p.get("published", "0000")[:4] or "0") <= _year_max
    ]

    # Apply category filter (uses sidebar selection)
    if "All" not in selected_categories:
        _filtered = [
            p for p in _filtered
            if p.get("category", "") in selected_categories
        ]

    st.caption(f"Showing **{len(_filtered)}** of **{len(all_papers)}** papers")
    st.divider()

    if not _filtered:
        st.info("No papers match your filters. Try adjusting the search or year range.")
    else:
        for _paper in _filtered:
            _title   = _paper.get("title", "Untitled")
            _pdf_url = _paper.get("url", "")
            _pub     = _paper.get("published", "")
            _cat     = _paper.get("category", "")
            _abs_url = arxiv_abstract_url(_pdf_url) if _pdf_url else ""
            _year    = _pub[:4] if _pub else "Unknown"

            _col_t, _col_y, _col_l, _col_s = st.columns([5, 1, 2, 1])
            with _col_t:
                st.markdown(f"**{_title}**")
                if _cat:
                    st.caption(_cat)
            with _col_y:
                st.markdown(f"📅 {_year}")
            with _col_l:
                _links = []
                if _abs_url:
                    _links.append(f"[Abstract ↗]({_abs_url})")
                if _pdf_url:
                    _links.append(f"[PDF ↗]({_pdf_url})")
                st.markdown(" · ".join(_links))
            with _col_s:
                if st.button("📌", key=f"db_{_title[:22]}", help="Save to reading list"):
                    save_to_reading_list({
                        "title":     _title,
                        "url":       _pdf_url,
                        "published": _pub,
                        "category":  _cat,
                    })
            st.divider()


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Analytics
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("Session Analytics")

    _history = st.session_state.query_history

    if not _history:
        st.info("No queries yet. Start asking questions in the 'Ask a Question' tab.")
    else:
        # Summary metrics row
        _total_q = len(_history)
        _avg_cf  = sum(_h["confidence"] for _h in _history) / _total_q
        _action_cts: dict = {}
        for _h in _history:
            _a = _h["action"]
            _action_cts[_a] = _action_cts.get(_a, 0) + 1
        _top_action = max(_action_cts, key=_action_cts.get)

        _ma, _mb, _mc = st.columns(3)
        _ma.metric("Total Queries",      _total_q)
        _mb.metric("Avg Confidence",     f"{_avg_cf:.2f}")
        _mc.metric("Most Common Action", _top_action)

        # Query history table
        st.subheader("Query History")
        st.dataframe(
            [
                {
                    "Query":      _h["query"],
                    "Action":     _h["action"],
                    "Confidence": round(_h["confidence"], 3),
                    "Timestamp":  _h["timestamp"],
                }
                for _h in _history
            ],
            use_container_width=True,
        )

        # Confidence over time (line chart)
        st.subheader("Confidence Over Time")
        st.line_chart(
            pd.DataFrame({"Confidence": [_h["confidence"] for _h in _history]})
        )

        # Category usage (bar chart — pie chart fallback per spec)
        st.subheader("Category Usage")
        _cat_cts: dict = {}
        for _h in _history:
            for _c in _h.get("categories", []):
                if _c:
                    _cat_cts[_c] = _cat_cts.get(_c, 0) + 1

        if _cat_cts:
            st.bar_chart(pd.DataFrame({"Queries returning category": _cat_cts}))
        else:
            st.caption("(No category data yet — showing action breakdown instead.)")
            st.bar_chart(pd.DataFrame({"Queries": _action_cts}))
