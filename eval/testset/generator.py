"""Generate a Q/A testset from CUAD contracts using Ragas TestsetGenerator."""
from __future__ import annotations

import json
from pathlib import Path

import structlog
from ragas.testset import TestsetGenerator
from ragas.testset.evolutions import multi_context, reasoning, simple

from eval.config import get_settings
from eval.judge_client import get_judge_llm
from eval.testset.cuad_adapter import load_cuad_documents

logger = structlog.get_logger(__name__)

TESTSET_PATH = Path("data/testset.json")


def generate_testset(
    data_dir: str | Path,
    output_path: Path = TESTSET_PATH,
    size: int | None = None,
    max_docs: int = 30,
) -> list[dict]:
    """Generate synthetic Q/A pairs from CUAD contracts and save to JSON."""
    settings = get_settings()
    n = size or settings.eval_testset_size

    docs = load_cuad_documents(data_dir, max_docs=max_docs)
    if not docs:
        raise ValueError(f"No documents found in {data_dir}")

    llm = get_judge_llm()

    generator = TestsetGenerator.from_langchain(
        generator_llm=llm,
        critic_llm=llm,
        embeddings=_get_embeddings(),
    )

    logger.info("testset_generation_start", size=n, docs=len(docs))
    testset = generator.generate_with_langchain_docs(
        documents=docs,
        test_size=n,
        distributions={simple: 0.5, reasoning: 0.25, multi_context: 0.25},
        raise_exceptions=False,
    )

    rows = testset.to_pandas().to_dict(orient="records")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("testset_saved", path=str(output_path), size=len(rows))
    return rows


def load_testset(path: Path = TESTSET_PATH) -> list[dict]:
    """Load a previously generated testset from disk."""
    if not path.exists():
        raise FileNotFoundError(f"Testset not found at {path}. Run generate_testset first.")
    rows = json.loads(path.read_text(encoding="utf-8"))
    logger.info("testset_loaded", path=str(path), size=len(rows))
    return rows


def _get_embeddings():
    """BGE-M3 embeddings — same model as the RAG pipeline for consistency."""
    from langchain_community.embeddings import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
