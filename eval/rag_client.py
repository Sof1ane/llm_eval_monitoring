"""HTTP client for the sovereign RAG API."""
from __future__ import annotations

import time
from dataclasses import dataclass

import httpx
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from eval.config import get_settings

logger = structlog.get_logger(__name__)


@dataclass
class RAGResult:
    question: str
    answer: str
    contexts: list[str]      # raw excerpts from retrieved chunks
    sources: list[str]       # source filenames
    confidence: float
    latency_ms: float
    tokens_used: int | None  # available if RAG logs it


def _build_client() -> httpx.Client:
    s = get_settings()
    return httpx.Client(base_url=s.rag_base_url, timeout=s.rag_timeout_s)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def query_rag(question: str, top_k: int = 5) -> RAGResult:
    """Call POST /query on the RAG API and return a structured result."""
    settings = get_settings()
    payload = {"question": question, "top_k": top_k}

    t0 = time.perf_counter()
    with _build_client() as client:
        resp = client.post("/query", json=payload)
        resp.raise_for_status()
    latency_ms = (time.perf_counter() - t0) * 1000

    data = resp.json()
    sources_raw: list[dict] = data.get("sources", [])

    return RAGResult(
        question=question,
        answer=data.get("answer", ""),
        contexts=[s.get("excerpt", "") for s in sources_raw],
        sources=[s.get("source", "") for s in sources_raw],
        confidence=data.get("confidence", 0.0),
        latency_ms=round(latency_ms, 1),
        tokens_used=None,  # RAG API doesn't expose this yet; tracked via judge tokens
    )
