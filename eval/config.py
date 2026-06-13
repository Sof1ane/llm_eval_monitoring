"""Central settings — loaded once, shared across all eval modules."""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, PostgresDsn, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # RAG
    rag_base_url: str = "http://localhost:8080"
    rag_timeout_s: int = 30

    # Judge
    judge_backend: Literal["anthropic", "openai_compat"] = "anthropic"
    anthropic_api_key: str = ""
    judge_model: str = "claude-haiku-4-5-20251001"
    judge_base_url: str = "http://localhost:8000/v1"

    # Postgres
    eval_postgres_host: str = "localhost"
    eval_postgres_port: int = 5433
    eval_postgres_user: str = "eval"
    eval_postgres_password: str = "changeme"
    eval_postgres_db: str = "llm_eval"

    @computed_field
    @property
    def eval_db_url(self) -> str:
        return (
            f"postgresql+psycopg2://{self.eval_postgres_user}:{self.eval_postgres_password}"
            f"@{self.eval_postgres_host}:{self.eval_postgres_port}/{self.eval_postgres_db}"
        )

    # Eval params
    eval_batch_size: int = 10
    eval_testset_size: int = 50
    eval_ragas_timeout_s: int = 120

    # Quality gates
    gate_faithfulness_min: float = 0.75
    gate_answer_relevancy_min: float = 0.70
    gate_context_precision_min: float = 0.65
    gate_context_recall_min: float = 0.60
    gate_max_regression_delta: float = 0.05

    # Grafana
    grafana_admin_password: str = "admin"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
