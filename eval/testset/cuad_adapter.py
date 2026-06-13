"""Convert CUAD contracts in data/documents/ into Ragas-compatible documents."""
from __future__ import annotations

import json
from pathlib import Path

import structlog
from langchain_core.documents import Document

logger = structlog.get_logger(__name__)


def load_cuad_documents(data_dir: str | Path, max_docs: int = 0) -> list[Document]:
    """Load CUAD .txt contracts with their metadata into LangChain Documents."""
    data_dir = Path(data_dir)
    txt_files = sorted(data_dir.glob("*.txt"))

    if max_docs:
        txt_files = txt_files[:max_docs]

    docs: list[Document] = []
    for txt_path in txt_files:
        text = txt_path.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            continue

        meta: dict = {"source": txt_path.name, "dataset": "CUAD"}
        meta_path = txt_path.with_suffix(".meta.json")
        if meta_path.exists():
            try:
                meta.update(json.loads(meta_path.read_text(encoding="utf-8")))
            except json.JSONDecodeError:
                pass

        docs.append(Document(page_content=text, metadata=meta))

    logger.info("cuad_docs_loaded", count=len(docs), dir=str(data_dir))
    return docs
