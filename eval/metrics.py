"""Ragas metric definitions and quality thresholds."""
from __future__ import annotations

from dataclasses import dataclass

from ragas.metrics import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

from eval.config import get_settings
from eval.judge_client import get_judge_llm


def get_ragas_metrics() -> list:
    """Return configured Ragas metrics using the project LLM judge."""
    llm = get_judge_llm()
    return [
        Faithfulness(llm=llm),
        AnswerRelevancy(llm=llm),
        ContextPrecision(llm=llm),
        ContextRecall(llm=llm),
    ]


METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


@dataclass
class QualityThresholds:
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float

    def as_dict(self) -> dict[str, float]:
        return {
            "faithfulness": self.faithfulness,
            "answer_relevancy": self.answer_relevancy,
            "context_precision": self.context_precision,
            "context_recall": self.context_recall,
        }

    @classmethod
    def from_settings(cls) -> "QualityThresholds":
        s = get_settings()
        return cls(
            faithfulness=s.gate_faithfulness_min,
            answer_relevancy=s.gate_answer_relevancy_min,
            context_precision=s.gate_context_precision_min,
            context_recall=s.gate_context_recall_min,
        )
