"""Estimate request cost from a profile's pricing configuration."""

from __future__ import annotations

from agent_harness.config.loader import ModelPricing
from agent_harness.domain import Usage


def estimate_cost_usd(
    pricing: ModelPricing | None, usage: Usage, latency_seconds: float
) -> float | None:
    """Return an estimated cost, or ``None`` when nothing could be priced.

    Token prices apply only to reported token counts. GPU pricing multiplies
    wall-clock request latency (including retries and cold start) by the rate;
    with concurrent requests sharing a container this is an upper bound.
    """

    if pricing is None:
        return None
    total = 0.0
    priced = False
    if pricing.input_usd_per_million_tokens is not None and usage.input_tokens is not None:
        cached = 0
        if pricing.cached_input_usd_per_million_tokens is not None and usage.cached_input_tokens:
            cached = min(usage.cached_input_tokens, usage.input_tokens)
            total += cached * pricing.cached_input_usd_per_million_tokens / 1_000_000
        total += (usage.input_tokens - cached) * pricing.input_usd_per_million_tokens / 1_000_000
        priced = True
    if pricing.output_usd_per_million_tokens is not None and usage.output_tokens is not None:
        total += usage.output_tokens * pricing.output_usd_per_million_tokens / 1_000_000
        priced = True
    if pricing.gpu_usd_per_second is not None:
        total += max(0.0, latency_seconds) * pricing.gpu_usd_per_second
        priced = True
    return total if priced else None
