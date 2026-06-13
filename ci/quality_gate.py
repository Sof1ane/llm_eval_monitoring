"""
CI Quality Gate — compares the latest eval run against the active baseline.

Exit codes:
  0  — all metrics pass (or no baseline exists yet)
  1  — one or more metrics regressed beyond threshold
  2  — configuration / connectivity error
"""
from __future__ import annotations

import sys
from pathlib import Path

import structlog
import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(add_completion=False)
console = Console()
logger = structlog.get_logger(__name__)

METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


@app.command()
def main(
    testset: Path = typer.Option(Path("data/testset.json"), "--testset", "-t"),
    model_version: str = typer.Option("qwen2.5", "--model", "-m"),
    top_k: int = typer.Option(5, "--top-k", "-k"),
    threshold: float = typer.Option(
        0.05,
        "--threshold",
        help="Max allowed relative regression (0.05 = 5%)",
    ),
    fail_fast: bool = typer.Option(True, "--fail-fast/--no-fail-fast"),
) -> None:
    """Run evaluation and gate on quality regression vs. active baseline."""
    try:
        from eval.runner import run_evaluation
        from store.writer import get_active_baseline
    except ImportError as e:
        console.print(f"[red]Import error:[/red] {e}")
        raise typer.Exit(2)

    # 1. Load baseline
    baseline = get_active_baseline()
    if baseline is None:
        console.print(
            "[yellow]No active baseline found — skipping quality gate.[/yellow]\n"
            "Run `python scripts/capture_baseline.py` first."
        )
        raise typer.Exit(0)

    console.print(f"[bold blue]Quality Gate[/bold blue] — baseline from run {baseline.run_id[:8]}")
    console.print(f"  model : {baseline.model_version}")
    console.print(f"  threshold : {threshold:.0%} regression allowed\n")

    # 2. Run current evaluation
    try:
        eval_run = run_evaluation(
            testset_path=testset,
            model_version=model_version,
            top_k=top_k,
        )
    except Exception as exc:
        console.print(f"[red]Evaluation failed:[/red] {exc}")
        raise typer.Exit(2)

    # 3. Compare
    failures: list[str] = []
    table = Table(title="Quality Gate Results")
    table.add_column("Metric")
    table.add_column("Baseline", style="cyan")
    table.add_column("Current", style="yellow")
    table.add_column("Delta", style="white")
    table.add_column("Status")

    for metric in METRICS:
        baseline_score: float = getattr(baseline, metric, 0.0)
        current_score: float = eval_run.scores.get(metric, 0.0)

        if baseline_score > 0:
            relative_change = (current_score - baseline_score) / baseline_score
        else:
            relative_change = 0.0

        delta_str = f"{relative_change:+.1%}"
        passed = relative_change >= -threshold

        if not passed:
            failures.append(metric)
            status = "[red]FAIL[/red]"
        else:
            status = "[green]PASS[/green]"

        table.add_row(metric, f"{baseline_score:.4f}", f"{current_score:.4f}", delta_str, status)

    console.print(table)

    if failures:
        console.print(
            f"\n[bold red]GATE FAILED[/bold red] — {len(failures)} metric(s) regressed "
            f"by more than {threshold:.0%}:\n  " + ", ".join(failures)
        )
        logger.error(
            "quality_gate_failed",
            failures=failures,
            threshold=threshold,
            run_id=eval_run.run_id,
            baseline_run_id=baseline.run_id,
        )
        raise typer.Exit(1)

    console.print(f"\n[bold green]GATE PASSED[/bold green] — all metrics within {threshold:.0%} of baseline.")
    logger.info("quality_gate_passed", run_id=eval_run.run_id)


if __name__ == "__main__":
    app()
