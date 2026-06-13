"""Pin an eval run as the active quality baseline for CI gates."""
from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(
    testset: Path = typer.Option(Path("data/testset.json"), "--testset", "-t"),
    model_version: str = typer.Option("qwen2.5", "--model", "-m"),
    top_k: int = typer.Option(5, "--top-k", "-k"),
    notes: str = typer.Option("", "--notes", "-n", help="Free-text note stored with baseline"),
) -> None:
    """
    Run an evaluation pass and pin it as the active quality baseline.

    This sets the reference scores that quality_gate.py will compare against in CI.
    Run this manually after a validated model upgrade or prompt change.
    """
    from eval.runner import run_evaluation
    from store.writer import write_baseline

    console.print("[bold blue]Capturing baseline[/bold blue]")

    eval_run = run_evaluation(
        testset_path=testset,
        model_version=model_version,
        top_k=top_k,
    )

    write_baseline(eval_run, notes=notes or None)

    console.print("\n[bold green]Baseline captured[/bold green]")
    for metric, score in eval_run.scores.items():
        console.print(f"  {metric:<25} {score:.4f}")
    console.print(f"\n  run_id : {eval_run.run_id}")


if __name__ == "__main__":
    app()
