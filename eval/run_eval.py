"""ArXiv RAG — Retrieval Evaluation Runner.

Metrics
-------
Hit Rate@5 (primary)
    For each question, retrieve the top-5 chunks.  A chunk is "relevant" if it
    contains >= min_keyword_matches of the question's expected_keywords
    (case-insensitive substring match against chunk content + paper title).
    A question is "answered correctly" when at least 1 top-5 chunk is relevant.
    Hit Rate@5 = # correct / total questions.

MRR (secondary)
    Mean Reciprocal Rank: mean(1/rank) where rank is the position (1-indexed)
    of the first relevant chunk.  Questions with no relevant chunk contribute 0.

Both metrics are reported overall and split by domain (AI/ML vs. q-fin).

Usage
-----
    cd /path/to/arxiv-search-engine
    python3 eval/run_eval.py

Save results:
    python3 eval/run_eval.py | tee eval/results/$(date +%Y-%m-%d).txt
"""

import datetime
import json
import os
import sys

# Allow imports from project root when run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import nltk
nltk.download("punkt_tab", quiet=True)

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from supabase import create_client

load_dotenv()

_supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
_embed    = SentenceTransformer("all-MiniLM-L6-v2")

EVAL_SET  = os.path.join(os.path.dirname(__file__), "eval_set.json")
TOP_K     = 5


# ── retrieval (mirrors api.py / app.py logic, no Streamlit dependency) ────────

def retrieve(query: str, count: int = TOP_K) -> list[dict]:
    vec = _embed.encode(query).tolist()
    try:
        resp = _supabase.rpc("hybrid_search", {
            "query_text":      query,
            "query_embedding": vec,
            "match_count":     count * 6,
            "match_threshold": 0.2,
            "rrf_k":           60,
        }).execute()
    except Exception:
        resp = _supabase.rpc("match_documents", {
            "query_embedding": vec,
            "match_threshold": 0.2,
            "match_count":     count * 6,
        }).execute()

    seen: set[str] = set()
    out:  list[dict] = []
    for doc in (resp.data or []):
        fp = doc["content"][:150]
        if fp not in seen:
            seen.add(fp)
            out.append(doc)
    return out[:count]


# ── scoring ───────────────────────────────────────────────────────────────────

def is_relevant(chunk: dict, keywords: list[str], min_matches: int) -> bool:
    """True when the chunk (content + title) contains >= min_matches keywords."""
    haystack = (
        chunk.get("content", "")
        + " "
        + chunk.get("metadata", {}).get("title", "")
    ).lower()
    return sum(1 for kw in keywords if kw.lower() in haystack) >= min_matches


def reciprocal_rank(chunks: list[dict], keywords: list[str], min_matches: int) -> float:
    for rank, chunk in enumerate(chunks, 1):
        if is_relevant(chunk, keywords, min_matches):
            return 1.0 / rank
    return 0.0


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    with open(EVAL_SET) as f:
        data = json.load(f)
    questions = data["questions"]

    hits:   int   = 0
    rr_sum: float = 0.0

    # domain → [hit, hit, ...]
    domain_hits: dict[str, list[bool]] = {}

    W = 70  # line width

    print()
    print("ArXiv RAG — Retrieval Evaluation Report")
    print("=" * W)
    print(f"  Date:      {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  Eval set:  {EVAL_SET}")
    print(f"  Questions: {len(questions)}")
    print(f"  Top-K:     {TOP_K}")
    print(f"  Metric:    Hit Rate@{TOP_K} (primary)  ·  MRR (secondary)")
    print("=" * W)
    print()

    for q in questions:
        qid      = q["id"]
        question = q["question"]
        keywords = q["expected_keywords"]
        min_k    = q["min_keyword_matches"]
        domain   = q["domain"]
        topic    = q["topic"]

        chunks = retrieve(question)

        hit = any(is_relevant(c, keywords, min_k) for c in chunks)
        rr  = reciprocal_rank(chunks, keywords, min_k)

        hits   += int(hit)
        rr_sum += rr
        domain_hits.setdefault(domain, []).append(hit)

        # Top-1 info
        if chunks:
            t1_title = chunks[0]["metadata"]["title"][:52]
            t1_sim   = f"{chunks[0]['similarity']:.3f}"
            # Which keywords matched in top-1?
            found = [kw for kw in keywords if kw.lower() in (
                chunks[0].get("content", "") + chunks[0]["metadata"].get("title", "")
            ).lower()]
        else:
            t1_title = "—"
            t1_sim   = "—"
            found    = []

        mark = "✓" if hit else "✗"
        print(f"  [{mark}] Q{qid:02d} · {topic} ({domain})")
        print(f"       {question[:W - 7]}" + ("…" if len(question) > W - 7 else ""))
        print(f"       Top-1: {t1_title!r}  sim={t1_sim}  RR={rr:.2f}")
        if found:
            print(f"       Keywords matched: {', '.join(found)}")
        print()

    # ── aggregate results ─────────────────────────────────────────────────────
    n        = len(questions)
    hit_rate = hits / n
    mrr      = rr_sum / n

    top1_sims = []
    for q in questions:
        chunks = retrieve(q["question"], count=1)
        if chunks:
            top1_sims.append(chunks[0]["similarity"])
    avg_sim = sum(top1_sims) / len(top1_sims) if top1_sims else 0.0

    def _domain_summary(prefix: str) -> tuple[int, int]:
        hs = [h for d, hits_list in domain_hits.items() if d.startswith(prefix)
              for h in hits_list]
        return sum(hs), len(hs)

    ai_h,  ai_t  = _domain_summary("cs.")
    fin_h, fin_t = _domain_summary("q-fin")

    print("=" * W)
    print("SUMMARY")
    print("=" * W)
    print(f"  Hit Rate@{TOP_K}         {hits}/{n} = {hit_rate:.1%}")
    print(f"  MRR                  {mrr:.3f}")
    print(f"  Mean top-1 sim       {avg_sim:.3f}")
    print()
    if ai_t:
        print(f"  AI/ML  ({ai_t:2d} Qs)      {ai_h}/{ai_t} = {ai_h/ai_t:.1%}")
    if fin_t:
        print(f"  Finance ({fin_t:2d} Qs)     {fin_h}/{fin_t} = {fin_h/fin_t:.1%}")
    if fin_t and fin_h == 0:
        print()
        print("  0 finance hits — index q-fin papers first:")
        print("     python3 etl_pipeline.py")
    print("=" * W)
    print()
    print("  Paste these numbers into ARCHITECTURE.md §11 after each run.")
    print()


if __name__ == "__main__":
    main()
