"""Main evaluation loop: testset → RAG → Ragas → store results."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import structlog
from datasets import Dataset
from ragas import evaluate

from eval.config import get_settings
from eval.judge_cost import estimate_cost
from eval.metrics import METRIC_NAMES, get_ragas_metrics
from eval.rag_client import RAGResult, query_rag
from eval.testset.generator import TESTSET_PATH, load_testset
from store.writer import write_eval_run

logger = structlog.get_logger(__name__)


@dataclass
class EvalRun:
    run_id: str
    started_at: datetime
    model_version: str
    config_snapshot: dict
    results: list[dict] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)
    total_latency_ms: float = 0.0
    question_count: int = 0
    judge_tokens_used: int = 0
    judge_cost_usd: float = 0.0


def run_evaluation(
    testset_path: Path = TESTSET_PATH,
    model_version: str = "qwen2.5",
    run_id: str | None = None,
    top_k: int = 5,
) -> EvalRun:
    """
    Full evaluation pipeline:
    1. Load testset
    2. Query RAG for each question
    3. Score with Ragas
    4. Persist to Postgres
    """
    settings = get_settings()
    run_id = run_id or str(uuid.uuid4())
    started_at = datetime.now(timezone.utc)

    config_snapshot = {
        "rag_base_url": settings.rag_base_url,
        "judge_backend": settings.judge_backend,
        "judge_model": settings.judge_model,
        "top_k": top_k,
        "eval_testset_size": settings.eval_testset_size,
    }

    logger.info("eval_run_start", run_id=run_id, model_version=model_version)

    testset = load_testset(testset_path)
    rag_results: list[RAGResult] = []

    logger.info("querying_rag", questions=len(testset))
    for i, row in enumerate(testset):
        question = row.get("question") or row.get("user_input", "")
        if not question:
            continue
        try:
            result = query_rag(question, top_k=top_k)
            rag_results.append(result)
        except Exception as exc:
            logger.warning("rag_query_failed", question=question[:80], error=str(exc))

        if (i + 1) % 10 == 0:
            logger.info("rag_progress", done=i + 1, total=len(testset))

    if not rag_results:
        raise RuntimeError("No RAG results obtained — is the RAG API running?")

    # Build Ragas dataset
    # ground_truths from testset; contexts from RAG retrieval
    ground_truths = []
    for row in testset[: len(rag_results)]:
        gt = row.get("ground_truth") or row.get("reference") or row.get("answer", "")
        ground_truths.append(gt)

    ragas_dataset = Dataset.from_dict(
        {
            "question": [r.question for r in rag_results],
            "answer": [r.answer for r in rag_results],
            "contexts": [r.contexts for r in rag_results],
            "ground_truth": ground_truths,
        }
    )

    logger.info("ragas_scoring_start", samples=len(rag_results))
    metrics = get_ragas_metrics()
    ragas_result = evaluate(ragas_dataset, metrics=metrics)
    scores_df = ragas_result.to_pandas()

    # Ragas ≥0.2 exposes token usage on the result object
    ragas_tokens: int | None = getattr(ragas_result, "total_tokens", None)
    judge_tokens, judge_cost = estimate_cost(
        model=settings.judge_model,
        total_tokens=ragas_tokens,
        question_count=len(rag_results),
    )

    # Aggregate scores
    aggregate_scores: dict[str, float] = {}
    for metric in METRIC_NAMES:
        if metric in scores_df.columns:
            aggregate_scores[metric] = round(float(scores_df[metric].mean()), 4)

    total_latency = sum(r.latency_ms for r in rag_results)
    avg_latency = total_latency / len(rag_results)

    per_question_results = []
    for idx, rag_r in enumerate(rag_results):
        row_scores = {m: round(float(scores_df.iloc[idx][m]), 4) for m in METRIC_NAMES if m in scores_df.columns}
        per_question_results.append(
            {
                "question": rag_r.question,
                "answer": rag_r.answer,
                "sources": rag_r.sources,
                "confidence": rag_r.confidence,
                "latency_ms": rag_r.latency_ms,
                "ground_truth": ground_truths[idx],
                **row_scores,
            }
        )

    eval_run = EvalRun(
        run_id=run_id,
        started_at=started_at,
        model_version=model_version,
        config_snapshot=config_snapshot,
        results=per_question_results,
        scores=aggregate_scores,
        total_latency_ms=total_latency,
        question_count=len(rag_results),
        judge_tokens_used=judge_tokens,
        judge_cost_usd=judge_cost,
    )

    logger.info(
        "eval_run_complete",
        run_id=run_id,
        scores=aggregate_scores,
        avg_latency_ms=round(avg_latency, 1),
        judge_tokens=judge_tokens,
        judge_cost_usd=judge_cost,
    )

    write_eval_run(eval_run)
    return eval_run
