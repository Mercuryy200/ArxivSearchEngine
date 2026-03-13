"""ArXiv RAG — FastAPI REST layer.

Exposes the same retrieval and generation logic as the Streamlit UI via a
clean HTTP API, making the project consumable by any client (CLI, web app,
other services, or automated eval pipelines).

Run locally:
    uvicorn api:app --reload --port 8000

Interactive docs (auto-generated):
    http://localhost:8000/docs   (Swagger UI)
    http://localhost:8000/redoc  (ReDoc)

Endpoints:
    GET  /api/health              — liveness probe + active Gemini model
    POST /api/search              — retrieve papers + generate blocking answer
    POST /api/search/stream       — retrieve papers + stream answer token-by-token
    POST /api/summarize           — summarise supplied context text
    GET  /api/papers              — paginated paper list with optional category filter
"""

import os
from contextlib import asynccontextmanager
from typing import Generator

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from supabase import create_client

load_dotenv()

# ── Clients ───────────────────────────────────────────────────────────────────

_supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
_gemini   = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
_embed    = SentenceTransformer("all-MiniLM-L6-v2")

_MODEL_CANDIDATES = [
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash-exp",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    "gemini-1.5-flash-latest",
]
_GEMINI_MODEL = _MODEL_CANDIDATES[-1]  # overwritten at startup


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Probe Gemini models at startup; use the first one that responds."""
    global _GEMINI_MODEL
    for name in _MODEL_CANDIDATES:
        try:
            _gemini.models.generate_content(
                model=name,
                contents="hi",
                config=types.GenerateContentConfig(max_output_tokens=1),
            )
            _GEMINI_MODEL = name
            break
        except Exception:
            continue
    yield


app = FastAPI(
    title="ArXiv RAG Research API",
    description=(
        "Semantic search and LLM-grounded question answering over ArXiv papers "
        "(cs.AI · cs.LG · cs.CL · cs.CV · q-fin.*). "
        "Powered by pgvector + Gemini."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response schemas ────────────────────────────────────────────────

class SearchRequest(BaseModel):
    query:      str         = Field(...,  description="Natural-language research question")
    count:      int         = Field(5,    ge=1, le=20, description="Number of chunks to retrieve")
    threshold:  float       = Field(0.3,  ge=0.0, le=1.0, description="Minimum cosine similarity")
    categories: list[str] | None = Field(
        None,
        description="Optional category filter, e.g. ['cs.AI', 'q-fin.ST']. "
                    "Omit or pass null to search all categories.",
    )


class ChunkResult(BaseModel):
    title:      str
    url:        str
    published:  str
    category:   str
    similarity: float
    excerpt:    str   = Field(..., description="First 300 characters of the chunk")


class SearchResponse(BaseModel):
    query:      str
    answer:     str
    confidence: float = Field(..., description="Max cosine similarity across retrieved chunks")
    sources:    list[ChunkResult]
    model:      str


class SummarizeRequest(BaseModel):
    context: str = Field(..., description="Raw text to summarise (e.g. concatenated chunk content)")
    style:   str = Field(
        "bullets",
        description="One of: 'bullets' (3-point summary), 'open_problems', 'digest' (150-word)",
    )


class SummarizeResponse(BaseModel):
    style:  str
    result: str


class PapersResponse(BaseModel):
    returned: int
    papers:   list[dict]


# ── Core helpers (no Streamlit dependency) ────────────────────────────────────

def _retrieve(
    query:      str,
    count:      int,
    threshold:  float,
    categories: list[str] | None,
) -> list[dict]:
    """Return deduplicated, diversity-capped chunks for *query*."""
    vec = _embed.encode(query).tolist()

    try:
        resp = _supabase.rpc("hybrid_search", {
            "query_text":      query,
            "query_embedding": vec,
            "match_count":     count * 8,
            "match_threshold": threshold,
            "rrf_k":           60,
        }).execute()
    except Exception:
        resp = _supabase.rpc("match_documents", {
            "query_embedding": vec,
            "match_threshold": threshold,
            "match_count":     count * 8,
        }).execute()

    seen_fp:  set[str]       = set()
    best:     dict[str, dict] = {}
    overflow: list[dict]      = []

    for doc in (resp.data or []):
        fp = doc["content"][:150]
        if fp in seen_fp:
            continue
        seen_fp.add(fp)
        title = doc["metadata"]["title"]
        if title not in best:
            best[title] = doc
        else:
            overflow.append(doc)

    result = (list(best.values()) + overflow)[:count]

    if categories and "All" not in categories:
        result = [
            m for m in result
            if m.get("metadata", {}).get("category", "") in categories
        ]

    return result


def _build_prompt(query: str, context: str) -> str:
    return (
        "You are a helpful AI research assistant. Answer the Question below "
        "using ONLY the research paper excerpts in the Context.\n\n"
        "Rules:\n"
        "- Synthesise an answer from whatever relevant information exists.\n"
        "- If the Context only partially addresses the question, share what IS there.\n"
        "- Never invent facts not supported by the Context.\n"
        "- Respond in the same language as the Question.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\nAnswer:"
    )


def _context_from(matches: list[dict]) -> str:
    return "\n\n".join(
        f"[Source: {m['metadata']['title']}]\n{m['content']}"
        for m in matches
    )


def _call_gemini(prompt: str) -> str:
    return _gemini.models.generate_content(
        model=_GEMINI_MODEL, contents=prompt
    ).text or ""


def _stream_gemini(prompt: str) -> Generator[str, None, None]:
    for chunk in _gemini.models.generate_content_stream(
        model=_GEMINI_MODEL, contents=prompt
    ):
        if chunk.text:
            yield chunk.text


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/api/health", tags=["Meta"])
def health():
    """Liveness probe. Returns the active Gemini model name."""
    return {"status": "ok", "model": _GEMINI_MODEL}


@app.post("/api/search", response_model=SearchResponse, tags=["Search"])
def search(req: SearchRequest):
    """Retrieve the most relevant paper chunks and return a grounded answer.

    The answer is generated synchronously (blocking). For streaming output use
    `POST /api/search/stream` instead.
    """
    matches = _retrieve(req.query, req.count, req.threshold, req.categories)
    if not matches:
        raise HTTPException(status_code=404, detail="No relevant papers found.")

    context = _context_from(matches)
    answer  = _call_gemini(_build_prompt(req.query, context))
    sims    = [m["similarity"] for m in matches]

    return SearchResponse(
        query=req.query,
        answer=answer,
        confidence=round(max(sims), 4),
        sources=[
            ChunkResult(
                title=m["metadata"]["title"],
                url=m["metadata"].get("url", ""),
                published=m["metadata"].get("published", ""),
                category=m["metadata"].get("category", ""),
                similarity=round(m["similarity"], 4),
                excerpt=m["content"][:300],
            )
            for m in matches
        ],
        model=_GEMINI_MODEL,
    )


@app.post("/api/search/stream", tags=["Search"])
def search_stream(req: SearchRequest):
    """Retrieve papers and stream the Gemini answer token-by-token.

    Response is `text/plain`; consume it as a server-sent stream.
    The `sources` metadata is not included — call `POST /api/search` if you
    need structured source attribution alongside the answer.
    """
    matches = _retrieve(req.query, req.count, req.threshold, req.categories)
    if not matches:
        raise HTTPException(status_code=404, detail="No relevant papers found.")

    context = _context_from(matches)
    return StreamingResponse(
        _stream_gemini(_build_prompt(req.query, context)),
        media_type="text/plain",
    )


@app.post("/api/summarize", response_model=SummarizeResponse, tags=["Generation"])
def summarize(req: SummarizeRequest):
    """Summarise or analyse a block of text without performing retrieval.

    Useful when you already have the context (e.g. from a prior `/api/search`
    call) and want a different view of the same material.
    """
    _STYLE_PROMPTS = {
        "bullets": (
            "Summarise the following research context in exactly 3 concise bullet points.\n\n"
            f"Context:\n{req.context}\n\nBullet summary:"
        ),
        "open_problems": (
            "Based on the following research papers, what are the main unsolved problems "
            f"and open challenges identified?\n\nContext:\n{req.context}\n\nOpen problems:"
        ),
        "digest": (
            "Write a 150-word research digest summarising the key findings "
            f"and contributions.\n\nContext:\n{req.context}\n\nDigest:"
        ),
    }
    if req.style not in _STYLE_PROMPTS:
        raise HTTPException(
            status_code=400,
            detail=f"style must be one of {sorted(_STYLE_PROMPTS)}",
        )
    return SummarizeResponse(
        style=req.style,
        result=_call_gemini(_STYLE_PROMPTS[req.style]),
    )


@app.get("/api/papers", response_model=PapersResponse, tags=["Papers"])
def list_papers(
    limit:    int = Query(50,   ge=1, le=500),
    offset:   int = Query(0,    ge=0),
    category: str | None = Query(None, description="e.g. 'cs.AI' or 'q-fin.ST'"),
):
    """Return a paginated list of unique papers in the database.

    Metadata only — no content or embeddings are returned.
    """
    try:
        resp = (
            _supabase.table("documents")
            .select("metadata")
            .range(offset, offset + limit * 4 - 1)   # over-fetch to fill after dedup
            .execute()
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    seen:   set[str]  = set()
    papers: list[dict] = []
    for row in (resp.data or []):
        meta  = row.get("metadata") or {}
        title = meta.get("title", "").strip()
        if not title or title in seen:
            continue
        if category and meta.get("category") != category:
            continue
        seen.add(title)
        papers.append(meta)
        if len(papers) >= limit:
            break

    return PapersResponse(returned=len(papers), papers=papers)
