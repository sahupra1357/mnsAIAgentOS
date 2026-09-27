"""Composition helpers for building a runnable harness instance."""

from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from pathlib import Path

from .coding import create_coding_tool_runtime
from .config import HarnessConfig, load_config
from .context import BudgetedContextBuilder, ContextPolicy, limits_from_profiles
from .model_gateway import ModelGateway
from .orchestrator import AgentOrchestrator, OrchestratorSettings
from .persistence import SQLiteRunStore
from .tools import ToolRuntime, create_default_runtime


@dataclass(slots=True)
class HarnessApplication:
    """The composed runtime and its replaceable boundaries."""

    config: HarnessConfig
    models: ModelGateway
    tools: ToolRuntime
    store: SQLiteRunStore
    orchestrator: AgentOrchestrator

    async def close(self) -> None:
        await self.store.close()

    async def __aenter__(self) -> "HarnessApplication":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()


def build_application(
    *,
    config_path: str | PathLike[str],
    database_path: str | PathLike[str],
    tools: ToolRuntime | None = None,
    settings: OrchestratorSettings | None = None,
    context_policy: ContextPolicy | None = None,
) -> HarnessApplication:
    """Compose the default local application from replaceable implementations."""

    config = load_config(config_path)
    models = ModelGateway(config)
    tool_runtime = tools
    if tool_runtime is None:
        if config.coding.enabled:
            tool_runtime = create_coding_tool_runtime(
                config.coding,
                base_path=Path(config_path).parent,
            )
        else:
            tool_runtime = create_default_runtime()
    store = SQLiteRunStore(Path(database_path))
    orchestrator = AgentOrchestrator(
        models=models,
        tools=tool_runtime,
        store=store,
        settings=settings,
        context=BudgetedContextBuilder(
            limits_from_profiles(config.models), policy=context_policy
        ),
    )
    return HarnessApplication(
        config=config,
        models=models,
        tools=tool_runtime,
        store=store,
        orchestrator=orchestrator,
    )
