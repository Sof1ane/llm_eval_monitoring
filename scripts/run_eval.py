"""Run the full evaluation pipeline and store results."""
from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(
    testset: Path = typer.Option(
        Path("data/testset.json"),
        "--testset",
        "-t",
        help="Path to the testset JSON file",
    ),
    model_version: str = typer.Option(
        "qwen2.5",
        "--model",
        "-m",
        help="Model version identifier (stored in DB for traceability)",
    ),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of RAG contexts to retrieve"),
    run_id: str = typer.Option("", "--run-id", help="Custom run ID (default: auto UUID)"),
    ci: bool = typer.Option(False, "--ci", help="Mark this run as CI-triggered"),
) -> None:
    """Run RAG evaluation and persist results to Postgres."""
    from eval.runner import run_evaluation

    console.print("[bold blue]Starting evaluation run[/bold blue]")
    console.print(f"  testset       : {testset}")
    console.print(f"  model_version : {model_version}")
    console.print(f"  top_k         : {top_k}")

    eval_run = run_evaluation(
        testset_path=testset,
        model_version=model_version,
        run_id=run_id or None,
        top_k=top_k,
    )

    if ci:
        # Mark CI flag in DB
        from store.db import get_session
        from store.models import EvalRun as EvalRunORM
        with get_session() as session:
            orm = session.query(EvalRunORM).filter_by(run_id=eval_run.run_id).first()
            if orm:
                orm.ci_triggered = True
                session.commit()

    table = Table(title=f"Eval Run — {eval_run.run_id[:8]}")
    table.add_column("Metric", style="cyan")
    table.add_column("Score", style="green")

    for metric, score in eval_run.scores.items():
        table.add_row(metric, f"{score:.4f}")

    avg_lat = eval_run.total_latency_ms / eval_run.question_count if eval_run.question_count else 0
    table.add_row("avg_latency_ms", f"{avg_lat:.1f}")
    table.add_row("questions", str(eval_run.question_count))
    table.add_row("judge_tokens", str(eval_run.judge_tokens_used))
    table.add_row("judge_cost_usd", f"${eval_run.judge_cost_usd:.4f}")

    console.print(table)
    console.print(f"\n[bold green]Run ID:[/bold green] {eval_run.run_id}")


if __name__ == "__main__":
    app()
