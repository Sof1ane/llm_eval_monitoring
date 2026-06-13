"""LLM judge abstraction — swappable between cloud API and local OpenAI-compat endpoint."""
from __future__ import annotations

from functools import lru_cache

import structlog
from langchain_core.language_models import BaseChatModel

from eval.config import get_settings

logger = structlog.get_logger(__name__)


@lru_cache(maxsize=1)
def get_judge_llm() -> BaseChatModel:
    settings = get_settings()

    if settings.judge_backend == "anthropic":
        from langchain_anthropic import ChatAnthropic

        logger.info("judge_init", backend="anthropic", model=settings.judge_model)
        return ChatAnthropic(
            model=settings.judge_model,
            api_key=settings.anthropic_api_key,
            temperature=0,
            max_tokens=1024,
        )

    if settings.judge_backend == "ollama":
        from langchain_ollama import ChatOllama

        logger.info("judge_init", backend="ollama", model=settings.judge_model, base_url=settings.ollama_base_url)
        return ChatOllama(
            model=settings.judge_model,
            base_url=settings.ollama_base_url,
            temperature=0,
            num_predict=1024,
        )

    # openai_compat — vLLM, LM Studio, or any OpenAI-compatible endpoint
    from langchain_openai import ChatOpenAI

    logger.info("judge_init", backend="openai_compat", model=settings.judge_model)
    return ChatOpenAI(
        model=settings.judge_model,
        base_url=settings.judge_base_url,
        api_key="not-used",
        temperature=0,
        max_tokens=1024,
    )
