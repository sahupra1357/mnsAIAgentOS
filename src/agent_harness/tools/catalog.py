"""The persistent tool catalog: what every tool source provides, and the operator's switches.

Handlers are code and cannot be stored, so the catalog records *definitions*
(spec, source, hash, trust) and the operator's enabled flag. At startup the
application re-attaches handlers from their sources and then calls
``ToolCatalog.sync`` to record what is available now and to hide tools an
operator disabled. The catalog keeps rows for tools that disappeared, so older
runs can still resolve the names they used.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict

from agent_harness.domain import (
    ToolCatalogEntry,
    ToolCatalogStore,
    ToolRisk,
    ToolSpec,
    ToolTrust,
)
from agent_harness.tools.registry import ToolRegistry
from agent_harness.orchestration.versioning import spec_sha256


def spec_to_json(spec: ToolSpec) -> str:
    data = asdict(spec)
    data["risk"] = spec.risk.value
    data["trust"] = spec.trust.value
    data["writes_paths"] = list(spec.writes_paths)
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def spec_from_json(value: str) -> ToolSpec:
    data = json.loads(value)
    return ToolSpec(
        name=data["name"],
        description=data["description"],
        input_schema=data["input_schema"],
        timeout_seconds=float(data.get("timeout_seconds", 30.0)),
        risk=ToolRisk(data.get("risk", ToolRisk.READ_ONLY.value)),
        idempotent=bool(data.get("idempotent", True)),
        writes_paths=tuple(data.get("writes_paths", ())),
        trust=ToolTrust(data.get("trust", ToolTrust.TRUSTED.value)),
    )


def entry_for(spec: ToolSpec, *, source: str, source_kind: str) -> ToolCatalogEntry:
    return ToolCatalogEntry(
        name=spec.name,
        source=source,
        source_kind=source_kind,
        spec_json=spec_to_json(spec),
        spec_sha256=spec_sha256(spec),
        description=spec.description,
        risk=spec.risk,
        trust=spec.trust,
    )


class ToolCatalog:
    """Keep a registry and its durable catalog in step."""

    def __init__(self, store: ToolCatalogStore, registry: ToolRegistry) -> None:
        self._store = store
        self._registry = registry

    def sync(self, *, sources: Iterable[str] | None = None) -> None:
        """Record the registry's tools, then hide tools the catalog says are off.

        ``sources`` names the sources that were fully enumerated just now; their
        tools that are missing are marked unavailable. By default every source
        currently in the registry counts as enumerated.
        """

        registered = self._registry.registered()
        selected = set(sources) if sources is not None else {t.source for t in registered.values()}
        entries = [
            entry_for(tool.spec, source=tool.source, source_kind=tool.source_kind)
            for tool in registered.values()
            if tool.source in selected
        ]
        self._store.sync_catalog(entries, sources=sorted(selected))
        self.apply_switches()

    def apply_switches(self) -> None:
        """Hide registered tools that are disabled or unavailable in the catalog."""

        catalog = {entry.name: entry for entry in self._store.catalog_entries()}
        for name in self._registry.registered():
            entry = catalog.get(name)
            self._registry.set_hidden(name, entry is not None and not (entry.enabled and entry.available))

    def set_enabled(self, name: str, enabled: bool) -> bool:
        changed = self._store.set_tool_enabled(name, enabled)
        if changed:
            self.apply_switches()
        return changed

    def entries(self) -> Sequence[ToolCatalogEntry]:
        return self._store.catalog_entries()

    def cached_specs(self, source: str) -> list[ToolSpec]:
        """Last known specs of an available source (used before it is reachable)."""

        return [
            spec_from_json(entry.spec_json)
            for entry in self._store.catalog_entries()
            if entry.source == source and entry.available
        ]
