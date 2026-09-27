"""Regression gates: compare a benchmark report with a saved baseline."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class Thresholds:
    max_accuracy_drop: float = 0.03
    min_accuracy: float | None = None
    max_error_rate: float = 0.10
    max_latency_increase: float | None = None  # relative, e.g. 0.5 = +50% mean latency
    max_cost_increase: float | None = None  # relative

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "Thresholds":
        known = {name: values[name] for name in cls.__dataclass_fields__ if name in values}
        return cls(**known)


@dataclass(slots=True)
class GateResult:
    passed: bool
    checks: list[tuple[str, bool, str]] = field(default_factory=list)

    def lines(self) -> list[str]:
        return [f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}" for name, ok, detail in self.checks]


def compare_reports(
    current: Mapping[str, Any],
    baseline: Mapping[str, Any] | None,
    thresholds: Thresholds,
) -> GateResult:
    """Check absolute floors, then (with a baseline) regressions against it."""

    metrics = current["metrics"]
    checks: list[tuple[str, bool, str]] = []
    accuracy = float(metrics["accuracy"])
    if thresholds.min_accuracy is not None:
        checks.append((
            "min_accuracy",
            accuracy >= thresholds.min_accuracy,
            f"{accuracy:.3f} >= {thresholds.min_accuracy:.3f}",
        ))
    error_rate = float(metrics["error_rate"])
    checks.append(("max_error_rate", error_rate <= thresholds.max_error_rate,
                   f"{error_rate:.3f} <= {thresholds.max_error_rate:.3f}"))

    if baseline is not None:
        base = baseline["metrics"]
        same_data = current.get("dataset", {}).get("sha256") == baseline.get("dataset", {}).get("sha256")
        same_items = current.get("settings", {}).get("item_ids_sha256") == baseline.get(
            "settings", {}
        ).get("item_ids_sha256")
        checks.append(("same_dataset", same_data and same_items,
                       "dataset file and selected items match the baseline"
                       if same_data and same_items else "dataset or item selection differs"))
        drop = float(base["accuracy"]) - accuracy
        checks.append((
            "max_accuracy_drop",
            drop <= thresholds.max_accuracy_drop,
            f"{accuracy:.3f} vs baseline {float(base['accuracy']):.3f} "
            f"(drop {drop:+.3f}, allowed {thresholds.max_accuracy_drop:.3f})",
        ))
        for name, key, limit in (
            ("max_latency_increase", "mean_latency_seconds", thresholds.max_latency_increase),
            ("max_cost_increase", "total_cost_usd", thresholds.max_cost_increase),
        ):
            if limit is None or not float(base.get(key) or 0):
                continue
            change = float(metrics.get(key) or 0) / float(base[key]) - 1
            checks.append((name, change <= limit, f"{change:+.1%} (allowed +{limit:.0%})"))
    return GateResult(all(ok for _, ok, _ in checks), checks)
