"""Validation pipelines: public-benchmark runs, regression gates, and stages.

Entry point: ``agent-harness-bench`` (or ``python -m agent_harness.bench``).
See ``docs/VALIDATION.md``.
"""

from .compare import GateResult, compare_reports
from .data import load_records
from .core import BenchConfigError, BenchContext, ItemResult
from .runner import run_benchmark, summarize
from .suites import SUITES, BenchItem, Suite

__all__ = [
    "SUITES",
    "BenchConfigError",
    "BenchContext",
    "BenchItem",
    "GateResult",
    "ItemResult",
    "Suite",
    "compare_reports",
    "load_records",
    "run_benchmark",
    "summarize",
]
