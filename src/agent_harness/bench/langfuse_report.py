"""Benchmark results in Langfuse: a dataset, an experiment, and a trace with scores per item.

``agent-harness-bench run ... --langfuse`` (or ``langfuse = true`` on a ``[[benchmark]]``):

* **Dataset** ``agent-harness/<suite>/<data file>``: every selected item is upserted with its
  input and expected output (stable IDs, so re-runs reuse the items).
* **Experiment** ``--langfuse-run`` (default ``<profile> <UTC time>``): each item is one trace
  whose spans carry Langfuse's experiment attributes (experiment ID and name, dataset ID, item
  ID, root observation), so Langfuse groups the run as an experiment on that dataset and
  compares runs side by side. This is the v4 way; the older dataset-run-items API is deprecated.
* **Trace** per item (``bench <suite>``, environment ``experiment``): a root span with the item
  input, the answer, and the expected output, holding the item's whole agent run
  (generations, tools, subagents) or its direct model calls (bfcl, judge calls).
* The expected output is also in the root's metadata (``expected_output``), so a Langfuse
  LLM-as-a-judge can map ``{{ground_truth}}`` to it (its Expected Output field is only offered
  for experiments started in Langfuse's UI).
* **Scores** on the root observation: ``correct`` (boolean) and every boolean the suite reports
  in its details (``answer_ok``, ``behavior_ok``, ...), with the error or verdict as the comment.

Uses the ``[langfuse]`` content policy and keys; ``[langfuse] enabled`` is only for live runs.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

from agent_harness.domain import ModelRequest, ModelResponse, RunEvent, RunRecord
from agent_harness.observability.langfuse import (
    ContentPolicy,
    LangfuseBatcher,
    LangfuseClient,
    Span,
    TraceContext,
    _json,
    _metadata,
    _observation,
    build_turn,
    credentials,
    root_span_id,
    score_body,
    span_id,
)

from .core import BenchConfigError, BenchItem, ItemResult

logger = logging.getLogger(__name__)
_NAMESPACE = uuid.UUID("0f6b1e2a-3c4d-4e5f-8a9b-0c1d2e3f4a5b")
EXPERIMENT_ENVIRONMENT = "experiment"


def dataset_item_id(dataset: str, item_id: str) -> str:
    return uuid.uuid5(_NAMESPACE, f"{dataset}:{item_id}").hex


class BenchLangfuse:
    """Collects each item's run or model calls, then reports the item once it is scored."""

    def __init__(self, client: LangfuseClient, context: TraceContext, *, suite: str, profile: str,
                 dataset: str, run_name: str, description: str = "",
                 batcher: LangfuseBatcher | None = None) -> None:
        self.client = client
        self.context = context
        self.suite = suite
        self.profile = profile
        self.dataset = dataset
        self.run_name = run_name
        self.description = description
        self.experiment_id = uuid.uuid5(_NAMESPACE, f"experiment:{dataset}:{run_name}").hex
        self.dataset_id: str | None = None
        self.batcher = batcher or LangfuseBatcher(client, flush_seconds=1.0)
        self._runs: dict[str, tuple[RunRecord, Sequence[RunEvent]]] = {}
        self._calls: dict[str, list[tuple[ModelRequest, ModelResponse | None, float, str | None]]] = {}
        self.items = 0
        self.dataset_errors = 0

    async def prepare(self) -> None:
        self.dataset_id = await asyncio.to_thread(self.client.ensure_dataset, self.dataset,
                                                  f"agent-harness {self.suite} benchmark items", {"suite": self.suite})

    def capture(self, item_id: str, run: RunRecord, events: Sequence[RunEvent]) -> None:
        self._runs[item_id] = (run, list(events))

    def capture_call(self, item_id: str, request: ModelRequest, response: ModelResponse | None,
                     duration: float, error: str | None) -> None:
        self._calls.setdefault(item_id, []).append((request, response, duration, error))

    def trace_id(self, item_id: str) -> str:
        return uuid.uuid5(_NAMESPACE, f"trace:{self.run_name}:{self.dataset}:{item_id}").hex

    async def report(self, item: BenchItem, result: ItemResult) -> None:
        """Upsert the item into the dataset, then send its experiment trace and scores."""

        item_id = dataset_item_id(self.dataset, item.id)
        await asyncio.to_thread(self._upsert, item, item_id)
        trace = self.trace_id(item.id)
        root = root_span_id(trace)
        policy = self.context.policy
        experiment = {
            "langfuse.experiment.id": self.experiment_id,
            "langfuse.experiment.name": self.run_name,
            "langfuse.experiment.dataset.id": self.dataset_id,
            "langfuse.experiment.item.id": item_id,
            "langfuse.experiment.item.root_observation_id": root,
        }
        expected = policy.encoded(item.expected)
        root_extra = {"langfuse.experiment.description": self.description or None,
                      "langfuse.experiment.item.expected_output": expected,
                      # Also as root metadata: Langfuse evaluators on OTel-ingested experiments can map an
                      # observation's input, output, and metadata, but not the experiment's Expected Output.
                      "langfuse.observation.metadata.expected_output": expected,
                      **_metadata("langfuse.experiment.item.metadata.", {"category": item.category,
                                                                         "suite_item_id": item.id})}
        metadata = {"suite": self.suite, "item_id": item.id, "category": item.category, "status": result.status,
                    "correct": result.correct, "steps": result.steps, "tokens": result.tokens,
                    "cost_usd": result.cost_usd, "latency_seconds": round(result.latency_seconds, 3),
                    "error": policy.text(result.error)}
        tags = ["bench", self.suite]
        captured = self._runs.pop(item.id, None)
        calls = self._calls.pop(item.id, [])
        if captured is not None:
            run, events = captured
            spans = build_turn(run, events, 0, self.context, trace=trace, name=f"bench {self.suite}",
                               session_id=self.run_name, tags=tags, metadata=metadata, extra=experiment,
                               root_extra=root_extra, root_input=dict(item.input),
                               root_output=result.predicted or None, whole_run=True)
            if calls:  # e.g. the rubric judge's calls, next to the agent's own
                spans.extend(self._call_spans(trace, root, item, result, calls, tags, metadata, experiment,
                                              root_extra)[1:])
        else:
            spans = self._call_spans(trace, root, item, result, calls, tags, metadata, experiment, root_extra)
        self.batcher.add_spans(spans)
        comment = policy.text(result.error or _verdict(result.details))
        self.batcher.add_score(score_body(trace, "correct", bool(result.correct), observation=root, comment=comment,
                                          environment=self.context.environment))
        for key, value in sorted(result.details.items()):
            if isinstance(value, bool):
                self.batcher.add_score(score_body(trace, key, value, observation=root,
                                                  environment=self.context.environment))
        self.items += 1

    def _upsert(self, item: BenchItem, item_id: str) -> None:
        policy = self.context.policy
        try:
            self.client.upsert_dataset_item(self.dataset, item_id, policy.value(dict(item.input)),
                                            policy.value(item.expected),
                                            {"item_id": item.id, "category": item.category, "suite": self.suite})
        except Exception as exc:  # noqa: BLE001 - the benchmark result stands without Langfuse
            self.dataset_errors += 1
            logger.warning("Langfuse: dataset item %s: %s", item.id, exc)

    def _call_spans(self, trace: str, root: str, item: BenchItem, result: ItemResult,
                    calls: Sequence[tuple[ModelRequest, ModelResponse | None, float, str | None]],
                    tags: Sequence[str], metadata: Mapping[str, Any], experiment: Mapping[str, Any],
                    root_extra: Mapping[str, Any]) -> list[Span]:
        """A root span plus one generation per direct model call (suites without an agent run)."""

        policy = self.context.policy
        end = datetime.now(UTC)
        start = end - timedelta(seconds=max(result.latency_seconds, sum(c[2] for c in calls)))
        shared = {
            "langfuse.trace.name": f"bench {self.suite}", "langfuse.session.id": self.run_name,
            "langfuse.trace.tags": sorted({self.profile, *tags}), "langfuse.environment": self.context.environment,
            "langfuse.release": self.context.release, "langfuse.version": self.context.release,
            **_metadata("langfuse.trace.metadata.", {**metadata, "model_profile": self.profile}), **experiment,
        }
        spans = [Span(trace, root, f"bench {self.suite}", start, end, None, {
            **shared, **root_extra, **_observation(
                "agent", input=policy.encoded(dict(item.input)), output=policy.text(result.predicted) or None,
                level="DEFAULT" if result.status not in ("error", "failed") else "WARNING")})]
        cursor = start
        for number, (request, response, duration, error) in enumerate(calls):
            usage: dict[str, int] = {}
            if response is not None:
                for name, value in (("input", response.usage.input_tokens), ("output", response.usage.output_tokens)):
                    if value is not None:
                        usage[name] = value
            output = None
            if response is not None and policy.mode != "none":
                output = _json({"content": policy.text(response.content),
                                "tool_calls": [{"name": c.name, "arguments": policy.value(dict(c.arguments))}
                                               for c in response.tool_calls]})
            attributes = {**shared, **_observation(
                "generation", output=output, level="DEFAULT" if error is None else "ERROR",
                status=policy.text(error),
                input=_json([{"role": m.role.value, "content": policy.text(m.content)} for m in request.messages])
                if policy.mode != "none" else None)}
            attributes["langfuse.observation.model.name"] = self.context.model_names.get(self.profile, self.profile)
            if usage:
                attributes["langfuse.observation.usage_details"] = _json({**usage, "total": sum(usage.values())})
            if response is not None and response.usage.cost_usd is not None:
                attributes["langfuse.observation.cost_details"] = _json({"total": response.usage.cost_usd})
            finish = cursor + timedelta(seconds=duration)
            spans.append(Span(trace, span_id(trace, "call", number), f"model {self.profile}", cursor, finish,
                              root, attributes, policy.text(error)))
            cursor = finish
        return spans

    def close(self) -> dict[str, Any]:
        self.batcher.close()
        return {"dataset": self.dataset, "experiment": self.run_name, "items": self.items,
                "spans_sent": self.batcher.sent_spans, "scores_sent": self.batcher.sent_scores,
                "failed": self.batcher.failed, "dataset_errors": self.dataset_errors,
                "last_error": self.batcher.last_error}


def _verdict(details: Mapping[str, Any]) -> str | None:
    for key in ("judge_reason", "reason", "verdict", "failures"):
        value = details.get(key)
        if value:
            return str(value)
    return None


def bench_reporter(config: Any, environ: Mapping[str, str], *, suite: str, profile: str, data_name: str,
                   run_name: str | None = None, redactor: Any = None) -> BenchLangfuse:
    """The reporter for ``--langfuse``; raises BenchConfigError when the keys are missing."""

    from agent_harness import __version__
    from agent_harness.observability.redaction import Redactor

    settings = config.langfuse
    found = credentials(settings, environ)
    if found is None:
        raise BenchConfigError(f"--langfuse needs {settings.base_url_env}, {settings.public_key_env}, and "
                               f"{settings.secret_key_env} (put them in .env)")
    client = LangfuseClient(*found, timeout_seconds=settings.timeout_seconds, version=__version__)
    context = TraceContext(
        policy=ContentPolicy(settings.content, redactor=redactor or Redactor(config.redaction.patterns),
                             max_chars=settings.max_text_chars),
        model_names={name: p.model for name, p in config.models.items()},
        environment=EXPERIMENT_ENVIRONMENT, release=__version__,
    )
    name = run_name or f"{profile} {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')}"
    return BenchLangfuse(client, context, suite=suite, profile=profile,
                         dataset=f"agent-harness/{suite}/{data_name}", run_name=name,
                         description=f"agent-harness {__version__}, profile {profile}")
