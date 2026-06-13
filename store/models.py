"""SQLAlchemy ORM models for the eval database."""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class ModelRegistry(Base):
    """Tracks every model version that has been evaluated — AI Act Article 12."""
    __tablename__ = "model_registry"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_version: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    served_model_name: Mapped[str] = mapped_column(String(128), nullable=False)
    registered_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    config_snapshot: Mapped[dict] = mapped_column(JSON, nullable=False, default={})
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    runs: Mapped[list["EvalRun"]] = relationship("EvalRun", back_populates="model_entry")


class EvalRun(Base):
    """One evaluation run = one execution of run_evaluation()."""
    __tablename__ = "eval_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(36), unique=True, nullable=False)  # UUID
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    model_version: Mapped[str] = mapped_column(String(128), nullable=False)
    model_registry_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("model_registry.id"), nullable=True
    )
    question_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # Aggregate scores
    faithfulness: Mapped[float | None] = mapped_column(Float, nullable=True)
    answer_relevancy: Mapped[float | None] = mapped_column(Float, nullable=True)
    context_precision: Mapped[float | None] = mapped_column(Float, nullable=True)
    context_recall: Mapped[float | None] = mapped_column(Float, nullable=True)
    avg_latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    total_latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Cost tracking
    judge_tokens_used: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    judge_cost_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Metadata
    is_baseline: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    ci_triggered: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    config_snapshot: Mapped[dict] = mapped_column(JSON, nullable=False, default={})
    tags: Mapped[dict] = mapped_column(JSON, nullable=False, default={})

    model_entry: Mapped[ModelRegistry | None] = relationship("ModelRegistry", back_populates="runs")
    results: Mapped[list["EvalResult"]] = relationship("EvalResult", back_populates="run")


class EvalResult(Base):
    """One row per question in an eval run — full traceability."""
    __tablename__ = "eval_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(36), ForeignKey("eval_runs.run_id"), nullable=False)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    ground_truth: Mapped[str | None] = mapped_column(Text, nullable=True)
    sources: Mapped[list] = mapped_column(JSON, nullable=False, default=[])
    rag_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Per-question Ragas scores
    faithfulness: Mapped[float | None] = mapped_column(Float, nullable=True)
    answer_relevancy: Mapped[float | None] = mapped_column(Float, nullable=True)
    context_precision: Mapped[float | None] = mapped_column(Float, nullable=True)
    context_recall: Mapped[float | None] = mapped_column(Float, nullable=True)

    run: Mapped[EvalRun] = relationship("EvalRun", back_populates="results")


class QualityBaseline(Base):
    """Pinned baseline scores used by quality_gate.py in CI."""
    __tablename__ = "quality_baselines"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    captured_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    run_id: Mapped[str] = mapped_column(String(36), nullable=False)
    model_version: Mapped[str] = mapped_column(String(128), nullable=False)
    faithfulness: Mapped[float] = mapped_column(Float, nullable=False)
    answer_relevancy: Mapped[float] = mapped_column(Float, nullable=False)
    context_precision: Mapped[float] = mapped_column(Float, nullable=False)
    context_recall: Mapped[float] = mapped_column(Float, nullable=False)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
