import streamlit as st
from sentence_transformers import SentenceTransformer
from supabase import create_client
import google.generativeai as genai
import os
from dotenv import load_dotenv

# ── 1. SETUP ──────────────────────────────────────────────────────────────────
load_dotenv()
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key  = os.environ.get("SUPABASE_KEY")
google_key    = os.environ.get("GOOGLE_API_KEY")

supabase = create_client(supabase_url, supabase_key)
genai.configure(api_key=google_key)

# ── 2. CACHED RESOURCES ───────────────────────────────────────────────────────
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

embedding_model = load_embedding_model()

# ── 3. AGENT TOOL DEFINITIONS (Gemini function-calling schemas) ────────────────
# These Python functions are passed to Gemini as tools.  The LLM chooses which
# one to call based on the user query.  The actual logic lives in the app loop.

def search_papers(refined_query: str) -> str:
    """Search the ArXiv AI/ML paper database and generate a grounded answer.

    Use this when the query is specific and relevant to AI/ML research topics
    such as deep learning, transformers, NLP, computer vision, reinforcement
    learning, or generative models.

    Args:
        refined_query: The user's query, optionally rephrased for better
                       semantic retrieval from the vector database.
    """
    return "search"


def ask_clarification(question: str) -> str:
    """Ask the user a clarifying question before searching.

    Use this when the query is vague, ambiguous, or could refer to several
    different research areas, making it hard to retrieve the right papers.

    Args:
        question: A concise, targeted question that will help narrow down
                  what the user actually wants to find.
    """
    return "clarify"


def report_no_results(explanation: str) -> str:
    """Tell the user that no relevant papers can be found for their query.

    Use this when the query is clearly outside the scope of the ArXiv AI/ML
    database (e.g. cooking, sports, general medical advice).  Never fabricate
    an answer — it is better to be honest about scope limitations.

    Args:
        explanation: A friendly explanation of why results cannot be found and
                     a brief description of what topics the database does cover.
    """
    return "no_results"


# ── 4. HELPER FUNCTIONS ───────────────────────────────────────────────────────

def retrieve_documents(query: str, threshold: float = 0.3, count: int = 5) -> list:
    """Run a cosine-similarity search against the pgvector database."""
    query_vector = embedding_model.encode(query).tolist()
    resp = supabase.rpc("match_documents", {
        "query_embedding": query_vector,
        "match_threshold": threshold,
        "match_count": count,
    }).execute()
    return resp.data or []


def build_answer(user_query: str, matches: list) -> str:
    """Ask Gemini to synthesise a grounded answer from retrieved context."""
    context_text = "\n\n".join(
        f"Source ({doc['metadata']['title']}): {doc['content']}"
        for doc in matches
    )
    gen_model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"""You are a helpful research assistant. Answer the User's Question using ONLY
the Context provided below. If the answer is not present in the context, say
"I couldn't find that information in the papers." — do not invent facts.

Context:
{context_text}

User's Question: {user_query}

Answer:"""
    return gen_model.generate_content(prompt).text


def run_agent(user_query: str) -> tuple[str, dict]:
    """
    Agentic routing loop using Gemini function calling.

    Sends the user query to a Gemini model equipped with three tools and
    returns (action_name, action_args) so the UI loop can act on the
    decision without the LLM needing to know anything about Streamlit.

    Possible returns:
      ("search_papers",     {"refined_query": str})
      ("ask_clarification", {"question": str})
      ("report_no_results", {"explanation": str})
    """
    router = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        tools=[search_papers, ask_clarification, report_no_results],
    )

    routing_prompt = f"""You are a smart router for an ArXiv AI/ML research assistant.

The database contains papers about: deep learning, neural networks, transformers,
NLP, computer vision, reinforcement learning, generative models, LLMs, diffusion
models, and related AI/ML topics.

Analyse the following user query and call the most appropriate tool:
  • search_papers     — query is specific and relevant to AI/ML research
  • ask_clarification — query is too vague or could mean multiple different things
  • report_no_results — query is clearly outside AI/ML scope

User query: "{user_query}"
"""
    response = router.generate_content(routing_prompt)

    # Walk response parts looking for a function call
    for part in response.parts:
        fc = getattr(part, "function_call", None)
        if fc and fc.name:
            return fc.name, dict(fc.args)

    # Graceful fallback: default to search so the user always gets something
    return "search_papers", {"refined_query": user_query}


def arxiv_abstract_url(pdf_url: str) -> str:
    """Convert an ArXiv PDF URL to its abstract page URL."""
    url = pdf_url.replace("arxiv.org/pdf/", "arxiv.org/abs/")
    if url.endswith(".pdf"):
        url = url[:-4]
    return url


def confidence_badge(matches: list) -> tuple[str, float, float, str]:
    """
    Derive a confidence label from retrieved chunk similarities.

    Returns (label, max_sim, avg_sim, description).
    """
    if not matches:
        return "⚪ No Signal", 0.0, 0.0, "No documents retrieved."

    sims = [m["similarity"] for m in matches]
    max_sim = max(sims)
    avg_sim = sum(sims) / len(sims)
    desc = f"Best match: {max_sim:.2f} · Average: {avg_sim:.2f} across {len(matches)} chunks"

    if max_sim >= 0.70:
        label = "🟢 High Confidence"
    elif max_sim >= 0.50:
        label = "🟡 Medium Confidence"
    else:
        label = "🔴 Low Confidence"

    return label, max_sim, avg_sim, desc


# ── 5. PAGE CONFIG ────────────────────────────────────────────────────────────
st.set_page_config(page_title="ArXiv RAG Assistant", layout="wide")
st.title("🤖 ArXiv Research Assistant")
st.markdown(
    """
Ask anything about AI/ML research. The assistant will:
1. **Decide** the best action — search, ask for clarification, or explain why it can't help
2. **Retrieve** semantically relevant paper chunks from the vector database
3. **Answer** based *only* on those papers — no hallucination
"""
)

# ── 6. QUERY INPUT ────────────────────────────────────────────────────────────
query = st.text_input(
    "Ask a question about AI/ML research:",
    placeholder="e.g., How does multi-head attention work in Transformers?",
)

# ── 7. AGENTIC LOOP ───────────────────────────────────────────────────────────
if query:

    # ── Phase 1: Agent routing decision ──────────────────────────────────────
    with st.spinner("Analysing your query…"):
        action, args = run_agent(query)

    # ── Action A: Clarification needed ───────────────────────────────────────
    if action == "ask_clarification":
        st.info(
            f"**Before I search, could you clarify?**\n\n"
            f"{args.get('question', 'Could you provide more detail about what you are looking for?')}"
        )

    # ── Action B: Out of scope ────────────────────────────────────────────────
    elif action == "report_no_results":
        st.warning(
            f"**This topic appears to be outside my knowledge base.**\n\n"
            f"{args.get('explanation', 'The database covers AI/ML research papers only.')}"
        )

    # ── Action C: Search and answer ───────────────────────────────────────────
    else:
        refined_query = args.get("refined_query", query)

        with st.spinner("Searching the vector database…"):
            matches = retrieve_documents(refined_query)

        # ── No results after search ───────────────────────────────────────────
        if not matches:
            st.warning(
                "No relevant papers were found for that query.  "
                "Try rephrasing, or ask about a different AI/ML topic."
            )

        # ── Results found ─────────────────────────────────────────────────────
        else:
            # ── Sidebar: top-3 source papers ─────────────────────────────────
            with st.sidebar:
                st.header("📚 Top Sources")
                st.caption("Papers most relevant to your query")

                # Deduplicate by title, keep best similarity per paper
                seen: dict[str, dict] = {}
                for m in matches:
                    t = m["metadata"]["title"]
                    if t not in seen or m["similarity"] > seen[t]["similarity"]:
                        seen[t] = m
                top_papers = sorted(seen.values(), key=lambda x: x["similarity"], reverse=True)[:3]

                for i, paper in enumerate(top_papers, 1):
                    title    = paper["metadata"]["title"]
                    pdf_url  = paper["metadata"].get("url", "")
                    abs_url  = arxiv_abstract_url(pdf_url)
                    sim      = paper["similarity"]

                    display_title = title if len(title) <= 55 else title[:52] + "…"
                    st.markdown(f"**{i}. {display_title}**")
                    st.progress(float(sim), text=f"Similarity: {sim:.2f}")
                    if abs_url:
                        st.markdown(f"[View Abstract ↗]({abs_url})")
                    if i < len(top_papers):
                        st.divider()

            # ── Confidence indicator ──────────────────────────────────────────
            label, max_sim, avg_sim, desc = confidence_badge(matches)
            col_badge, col_desc = st.columns([1, 3])
            with col_badge:
                st.metric(
                    label="Retrieval Confidence",
                    value=f"{max_sim:.2f}",
                    delta=f"{avg_sim:.2f} avg",
                    help="Cosine similarity between your query and the best-matching paper chunk. "
                         "Higher = stronger semantic match.",
                )
            with col_desc:
                st.markdown(f"**{label}**")
                st.caption(desc)

            # ── Generate answer ───────────────────────────────────────────────
            with st.spinner("Reading papers and generating answer…"):
                try:
                    answer = build_answer(query, matches)
                    st.success("Answer generated from research papers:")
                    st.markdown(f"### 💡 {answer}")
                except Exception as e:
                    st.error(f"Error generating answer: {e}")

            # ── Expandable full source listing ────────────────────────────────
            with st.expander("📖 View All Source Documents"):
                for match in matches:
                    sim      = match["similarity"]
                    title    = match["metadata"]["title"]
                    pdf_url  = match["metadata"].get("url", "")
                    abs_url  = arxiv_abstract_url(pdf_url)

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
