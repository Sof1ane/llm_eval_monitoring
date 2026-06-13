"""
Estimate judge LLM cost in USD.

Pricing is hardcoded for known models; defaults to 0 for unknowns.
Keeping this simple: real deployments should instrument at the LLM layer,
but this gives a useful order-of-magnitude figure for the dashboard.
"""
from __future__ import annotations

# USD per 1M tokens (input, output)
_PRICING: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5-20251001": (0.80, 4.00),
    "claude-haiku-4-5": (0.80, 4.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-opus-4-8": (15.00, 75.00),
    "gpt-4o-mini": (0.15, 0.60),
    "mistral": (0.0, 0.0),
}

# Ragas scores ~4 metrics × N questions, each needing ~3 LLM calls.
# Average prompt ~800 tokens, completion ~200 tokens — rough constants.
_PROMPT_TOKENS_PER_CALL = 800
_COMPLETION_TOKENS_PER_CALL = 200
_CALLS_PER_QUESTION = 3  # faithfulness + relevancy + precision/recall share calls


def estimate_cost(
    model: str,
    total_tokens: int | None,
    question_count: int,
) -> tuple[int, float]:
    """
    Returns (estimated_tokens, estimated_cost_usd).

    If Ragas exposes total_tokens directly, use it; otherwise estimate from
    question count and per-call constants.
    """
    if total_tokens is None or total_tokens == 0:
        calls = question_count * _CALLS_PER_QUESTION
        prompt_t = calls * _PROMPT_TOKENS_PER_CALL
        completion_t = calls * _COMPLETION_TOKENS_PER_CALL
        total_tokens = prompt_t + completion_t
        input_tokens = prompt_t
        output_tokens = completion_t
    else:
        # Assume 80/20 split when we only have total
        input_tokens = int(total_tokens * 0.8)
        output_tokens = total_tokens - input_tokens

    # Match by prefix for versioned names like "claude-haiku-4-5-20251001"
    input_price, output_price = 0.0, 0.0
    for key, (ip, op) in _PRICING.items():
        if model.lower().startswith(key.lower()) or key.lower() in model.lower():
            input_price, output_price = ip, op
            break

    cost = (input_tokens / 1_000_000) * input_price + (output_tokens / 1_000_000) * output_price
    return total_tokens, round(cost, 6)
