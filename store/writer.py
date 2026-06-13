"""Persist eval results to Postgres."""
from __future__ import annotations

from datetime import datetime, timezone

import structlog

from store.db import get_session, init_db
from store.models import EvalResult, EvalRun as EvalRunORM, ModelRegistry, QualityBaseline

logger = structlog.get_logger(__name__)


def _ensure_model_registry(session, model_version: str, config_snapshot: dict) -> int | None:
    entry = session.query(ModelRegistry).filter_by(model_version=model_version).first()
    if entry is None:
        entry = ModelRegistry(
            model_version=model_version,
            served_model_name=model_version,
            registered_at=datetime.now(timezone.utc),
            config_snapshot=config_snapshot,
        )
        session.add(entry)
        session.flush()
    return entry.id


def write_eval_run(eval_run) -> None:
    """
    Persist an EvalRun dataclass (from eval.runner) to the database.
    Creates tables on first call if they don't exist.
    """
    init_db()

    with get_session() as session:
        model_registry_id = _ensure_model_registry(
            session,
            eval_run.model_version,
            eval_run.config_snapshot,
        )

        question_count = eval_run.question_count
        avg_latency = (
            eval_run.total_latency_ms / question_count if question_count else None
        )

        orm_run = EvalRunORM(
            run_id=eval_run.run_id,
            started_at=eval_run.started_at,
            finished_at=datetime.now(timezone.utc),
            model_version=eval_run.model_version,
            model_registry_id=model_registry_id,
            question_count=question_count,
            faithfulness=eval_run.scores.get("faithfulness"),
            answer_relevancy=eval_run.scores.get("answer_relevancy"),
            context_precision=eval_run.scores.get("context_precision"),
            context_recall=eval_run.scores.get("context_recall"),
            avg_latency_ms=round(avg_latency, 1) if avg_latency else None,
            total_latency_ms=round(eval_run.total_latency_ms, 1),
            judge_tokens_used=getattr(eval_run, "judge_tokens_used", None),
            judge_cost_usd=getattr(eval_run, "judge_cost_usd", None),
            config_snapshot=eval_run.config_snapshot,
            tags={},
        )
        session.add(orm_run)
        session.flush()

        for row in eval_run.results:
            result = EvalResult(
                run_id=eval_run.run_id,
                question=row["question"],
                answer=row["answer"],
                ground_truth=row.get("ground_truth"),
                sources=row.get("sources", []),
                rag_confidence=row.get("confidence"),
                latency_ms=row.get("latency_ms"),
                faithfulness=row.get("faithfulness"),
                answer_relevancy=row.get("answer_relevancy"),
                context_precision=row.get("context_precision"),
                context_recall=row.get("context_recall"),
            )
            session.add(result)

        session.commit()
        logger.info("eval_run_persisted", run_id=eval_run.run_id, questions=question_count)


def write_baseline(eval_run, notes: str | None = None) -> None:
    """Pin the current eval run as the active quality baseline."""
    init_db()

    with get_session() as session:
        # Deactivate previous baselines
        session.query(QualityBaseline).filter_by(is_active=True).update({"is_active": False})

        baseline = QualityBaseline(
            captured_at=datetime.now(timezone.utc),
            run_id=eval_run.run_id,
            model_version=eval_run.model_version,
            faithfulness=eval_run.scores.get("faithfulness", 0.0),
            answer_relevancy=eval_run.scores.get("answer_relevancy", 0.0),
            context_precision=eval_run.scores.get("context_precision", 0.0),
            context_recall=eval_run.scores.get("context_recall", 0.0),
            notes=notes,
            is_active=True,
        )
        session.add(baseline)
        session.commit()
        logger.info("baseline_captured", run_id=eval_run.run_id, model=eval_run.model_version)


def get_active_baseline() -> QualityBaseline | None:
    """Return the currently active baseline, or None."""
    init_db()
    with get_session() as session:
        return session.query(QualityBaseline).filter_by(is_active=True).first()
