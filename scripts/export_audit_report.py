"""
Export an AI Act compliance audit report (Articles 9, 12, 13).

Output: JSON + optional Markdown summary.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import typer
from rich.console import Console

app = typer.Typer(add_completion=False)
console = Console()


def _build_report(since: datetime | None = None) -> dict:
    from store.db import get_session, init_db
    from store.models import EvalResult, EvalRun, ModelRegistry, QualityBaseline

    init_db()

    with get_session() as session:
        # ── Article 12 — Logging & traceability ─────────────────────────────
        models_q = session.query(ModelRegistry).order_by(ModelRegistry.registered_at)
        models = [
            {
                "model_version": m.model_version,
                "served_model_name": m.served_model_name,
                "registered_at": m.registered_at.isoformat(),
                "notes": m.notes,
            }
            for m in models_q
        ]

        # ── Article 9 — Risk management via CI gates ─────────────────────────
        baselines_q = session.query(QualityBaseline).order_by(QualityBaseline.captured_at)
        baselines = [
            {
                "captured_at": b.captured_at.isoformat(),
                "run_id": b.run_id,
                "model_version": b.model_version,
                "scores": {
                    "faithfulness": b.faithfulness,
                    "answer_relevancy": b.answer_relevancy,
                    "context_precision": b.context_precision,
                    "context_recall": b.context_recall,
                },
                "is_active": b.is_active,
                "notes": b.notes,
            }
            for b in baselines_q
        ]

        # ── Article 13 — Transparency ─────────────────────────────────────────
        runs_q = session.query(EvalRun).order_by(EvalRun.started_at)
        if since:
            runs_q = runs_q.filter(EvalRun.started_at >= since)

        eval_runs = [
            {
                "run_id": r.run_id,
                "started_at": r.started_at.isoformat(),
                "finished_at": r.finished_at.isoformat() if r.finished_at else None,
                "model_version": r.model_version,
                "question_count": r.question_count,
                "aggregate_scores": {
                    "faithfulness": r.faithfulness,
                    "answer_relevancy": r.answer_relevancy,
                    "context_precision": r.context_precision,
                    "context_recall": r.context_recall,
                },
                "avg_latency_ms": r.avg_latency_ms,
                "judge_cost_usd": r.judge_cost_usd,
                "is_baseline": r.is_baseline,
                "ci_triggered": r.ci_triggered,
                "config_snapshot": r.config_snapshot,
            }
            for r in runs_q
        ]

    return {
        "report_generated_at": datetime.now(timezone.utc).isoformat(),
        "report_version": "1.0",
        "ai_act_articles": {
            "article_9_risk_management": {
                "description": "Quality gates enforced in CI — regression blocks deployment",
                "quality_baselines": baselines,
            },
            "article_12_logging": {
                "description": "All model versions and eval runs are logged with full config snapshots",
                "model_registry": models,
            },
            "article_13_transparency": {
                "description": "Evaluation results exportable on demand for human review",
                "eval_runs": eval_runs,
            },
        },
        "summary": {
            "total_models_registered": len(models),
            "total_eval_runs": len(eval_runs),
            "total_baselines": len(baselines),
            "active_baseline": next((b for b in baselines if b["is_active"]), None),
        },
    }


def _render_markdown(report: dict) -> str:
    ts = report["report_generated_at"]
    summary = report["summary"]
    active_bl = summary.get("active_baseline")

    lines = [
        "# AI Act Compliance Audit Report",
        f"\n**Generated:** {ts}",
        f"**Report version:** {report['report_version']}",
        "",
        "## Executive Summary",
        "",
        f"| | |",
        f"|---|---|",
        f"| Models registered | {summary['total_models_registered']} |",
        f"| Evaluation runs | {summary['total_eval_runs']} |",
        f"| Quality baselines | {summary['total_baselines']} |",
        "",
    ]

    if active_bl:
        scores = active_bl["scores"]
        lines += [
            "**Active quality baseline:**",
            "",
            f"| Metric | Score |",
            f"|---|---|",
            f"| Faithfulness | {scores['faithfulness']:.4f} |",
            f"| Answer Relevancy | {scores['answer_relevancy']:.4f} |",
            f"| Context Precision | {scores['context_precision']:.4f} |",
            f"| Context Recall | {scores['context_recall']:.4f} |",
            "",
        ]

    lines += [
        "## Article 9 — Risk Management",
        "",
        "Quality gates are integrated into the CI/CD pipeline. "
        "Any regression greater than 5% on Ragas metrics blocks deployment.",
        "",
        "## Article 12 — Logging & Record-Keeping",
        "",
        "All model versions are recorded in `model_registry` with full config snapshots. "
        "Every evaluation run is persisted with timestamps, scores, and judge configuration.",
        "",
        "## Article 13 — Transparency",
        "",
        "Evaluation results are exportable on demand (this report). "
        "The Grafana dashboard provides real-time visibility into quality trends.",
        "",
        "---",
        "_This report was generated automatically by `scripts/export_audit_report.py`._",
    ]

    return "\n".join(lines)


@app.command()
def main(
    output: Path = typer.Option(
        Path("data/audit_report.json"),
        "--output",
        "-o",
        help="Output path for the JSON audit report",
    ),
    markdown: bool = typer.Option(True, "--markdown/--no-markdown", help="Also write a Markdown summary"),
    since: str = typer.Option(
        "",
        "--since",
        help="ISO date to filter eval runs (e.g. 2026-01-01)",
    ),
) -> None:
    """Export an AI Act compliance audit report covering Articles 9, 12, and 13."""
    since_dt = datetime.fromisoformat(since).replace(tzinfo=timezone.utc) if since else None

    report = _build_report(since=since_dt)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    console.print(f"[green]JSON report saved:[/green] {output}")

    if markdown:
        md_path = output.with_suffix(".md")
        md_path.write_text(_render_markdown(report), encoding="utf-8")
        console.print(f"[green]Markdown summary saved:[/green] {md_path}")

    s = report["summary"]
    console.print(
        f"\n  models registered : {s['total_models_registered']}\n"
        f"  eval runs         : {s['total_eval_runs']}\n"
        f"  baselines         : {s['total_baselines']}"
    )


if __name__ == "__main__":
    app()
