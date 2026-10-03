"""Per-feature validation (Phase 15c): ``[[feature]]`` entries in a pipeline plan.

Each feature names the checks that validate it::

    [[feature]]
    name = "guardrails"
    description = "Input, output, and tool-output guards; taint mode"
    tests = ["test_guardrails"]                 # unit-test modules in tests/
    scenarios = ["guardrail_refusal"]           # files <scenarios.path>/<name>.toml
    smoke = []                                  # [[smoke]] profiles
    benchmarks = ["guardrails"]                 # [[benchmark]] names

``agent-harness-bench pipeline PLAN --feature guardrails`` runs only these checks;
without ``--feature`` the pipeline runs everything, as before.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .core import BenchConfigError

_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_LISTS = ("tests", "scenarios", "smoke", "benchmarks")


@dataclass(frozen=True, slots=True)
class Feature:
    name: str
    description: str = ""
    tests: tuple[str, ...] = ()
    scenarios: tuple[str, ...] = ()
    smoke: tuple[str, ...] = ()
    benchmarks: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class FeatureSelection:
    """The union of the selected features' checks, in first-seen order."""

    names: tuple[str, ...]
    tests: tuple[str, ...]
    scenarios: tuple[str, ...]
    smoke: tuple[str, ...]
    benchmarks: tuple[str, ...]

    @property
    def label(self) -> str:
        return ",".join(self.names)


def parse_features(plan: Mapping[str, Any]) -> dict[str, Feature]:
    """Validate the plan's ``[[feature]]`` entries against its smoke and benchmark entries."""

    raw = plan.get("feature", [])
    if not isinstance(raw, list) or not all(isinstance(item, dict) for item in raw):
        raise BenchConfigError("feature must be an array of tables ([[feature]])")
    smoke_profiles = {str(entry.get("profile")) for entry in plan.get("smoke", [])}
    benchmark_names = {
        str(entry.get("name") or f"{entry.get('suite')}-{entry.get('profile', 'default')}")
        for entry in plan.get("benchmark", [])
    }
    features: dict[str, Feature] = {}
    for item in raw:
        unknown = set(item) - {"name", "description", *_LISTS}
        if unknown:
            raise BenchConfigError(f"unknown fields in feature: {', '.join(sorted(unknown))}")
        name = item.get("name")
        if not isinstance(name, str) or not _NAME.fullmatch(name):
            raise BenchConfigError(f"invalid feature name: {name!r}")
        if name in features:
            raise BenchConfigError(f"feature {name!r} is defined twice")
        values: dict[str, tuple[str, ...]] = {}
        for key in _LISTS:
            value = item.get(key, [])
            if not isinstance(value, list) or not all(isinstance(v, str) and v for v in value):
                raise BenchConfigError(f"feature {name!r}: {key} must be a list of names")
            values[key] = tuple(value)
        missing = [p for p in values["smoke"] if p not in smoke_profiles]
        missing += [b for b in values["benchmarks"] if b not in benchmark_names]
        if missing:
            raise BenchConfigError(f"feature {name!r} names unknown smoke/benchmark entries: {missing}")
        features[name] = Feature(name=name, description=str(item.get("description", "")), **values)
    return features


def select_features(features: Mapping[str, Feature], names: Sequence[str]) -> FeatureSelection:
    unknown = [name for name in names if name not in features]
    if unknown:
        raise BenchConfigError(f"unknown feature(s) {unknown}; choose from {sorted(features)}")
    chosen = [features[name] for name in dict.fromkeys(names)]

    def union(key: str) -> tuple[str, ...]:
        return tuple(dict.fromkeys(value for feature in chosen for value in getattr(feature, key)))

    return FeatureSelection(
        names=tuple(feature.name for feature in chosen),
        tests=union("tests"),
        scenarios=union("scenarios"),
        smoke=union("smoke"),
        benchmarks=union("benchmarks"),
    )


def uncovered(features: Iterable[Feature], key: str, available: Iterable[str]) -> list[str]:
    """Items in ``available`` that no feature lists under ``key`` (for coverage checks)."""

    covered = {value for feature in features for value in getattr(feature, key)}
    return sorted(set(available) - covered)
