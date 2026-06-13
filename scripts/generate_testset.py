"""Generate a synthetic testset from CUAD documents using Ragas + BGE-M3."""
from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(
    data_dir: Path = typer.Option(
        Path("../rag_souverain/data/documents"),
        "--data-dir",
        "-d",
        help="Directory containing CUAD .txt documents",
    ),
    output: Path = typer.Option(
        Path("data/testset.json"),
        "--output",
        "-o",
        help="Where to save the generated testset",
    ),
    max_docs: int = typer.Option(
        20,
        "--max-docs",
        "-n",
        help="Max CUAD documents to use (keep low on CPU; 20 ≈ ~50 questions)",
    ),
    testset_size: int = typer.Option(
        0,
        "--size",
        "-s",
        help="Target number of Q/A pairs (0 = use EVAL_TESTSET_SIZE from config)",
    ),
) -> None:
    """Generate a Ragas synthetic testset from CUAD contract documents."""
    from eval.config import get_settings
    from eval.testset.generator import generate_testset

    settings = get_settings()
    if testset_size == 0:
        testset_size = settings.eval_testset_size

    console.print(f"[bold blue]Generating testset[/bold blue]")
    console.print(f"  data_dir  : {data_dir}")
    console.print(f"  max_docs  : {max_docs}")
    console.print(f"  size      : {testset_size}")
    console.print(f"  output    : {output}")

    output.parent.mkdir(parents=True, exist_ok=True)

    testset = generate_testset(
        data_dir=data_dir,
        output_path=output,
        max_docs=max_docs,
        testset_size=testset_size,
    )

    console.print(f"\n[bold green]Done.[/bold green] {len(testset)} questions saved to {output}")


if __name__ == "__main__":
    app()
