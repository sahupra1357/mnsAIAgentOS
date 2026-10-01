"""Benchmark suites for public datasets.

Each suite loads records from a local file (you download the dataset), runs
every item through a specific layer of the harness, and scores it:

=========  ==================================  ===================================
suite      what it validates                   dataset format (per record)
=========  ==================================  ===================================
gsm8k      full agent loop + tool use          {"question", "answer": "... #### 42"}
           (calculator), budgets, retries
qa         agent answers without tools or      {"question", "answer" | "answers"}
           with default tools (TriviaQA,       (field names configurable)
           NQ-open, HotpotQA-closed...)
bfcl       tool-calling accuracy of the model  BFCL question file + possible-answer
           gateway per provider/adapter        file joined by "id"
humaneval  coding: generated code executed     {"task_id", "prompt", "test",
           in the sandbox (Docker by default)   "entry_point"}
=========  ==================================  ===================================

Scores are the harness's own approximations; for leaderboard-comparable numbers
use each benchmark's official evaluator on the saved predictions.
"""

from __future__ import annotations

import math
import re
import string
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from agent_harness.domain import HarnessError, Message, ModelRequest, Role, RunBudget, ToolSpec
from agent_harness.tools import ToolRegistry, ToolRuntime, create_default_runtime

from .core import BenchConfigError, BenchContext, BenchItem, ItemResult
from .data import DatasetError, load_records


class Suite(Protocol):
    name: str
    description: str

    def load(self, path: str | Path, options: Mapping[str, Any]) -> list[BenchItem]: ...

    async def run(self, item: BenchItem, context: BenchContext) -> ItemResult: ...


def _result_from_outcome(item: BenchItem, outcome: Any, started: float, **values: Any) -> ItemResult:
    run = outcome.run
    return ItemResult(
        id=item.id,
        category=item.category,
        status=run.status.value,
        steps=run.step_count,
        tokens=run.consumed_tokens,
        cost_usd=run.consumed_cost_usd,
        latency_seconds=time.monotonic() - started,
        error=run.error,
        **values,
    )


# --- GSM8K -----------------------------------------------------------------------

_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_FINAL = re.compile(r"final answer\s*[:：]?\s*\$?\s*(-?\d[\d,]*(?:\.\d+)?)", re.IGNORECASE)


def extract_number(text: str) -> str | None:
    """The number after 'Final answer:', else the last number in the text."""

    match = _FINAL.search(text)
    if match is None:
        numbers = _NUMBER.findall(text)
        if not numbers:
            return None
        value = numbers[-1]
    else:
        value = match.group(1)
    return value.replace(",", "").rstrip(".")


def numbers_equal(left: str | None, right: str | None) -> bool:
    if left is None or right is None:
        return False
    try:
        return math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=1e-6)
    except ValueError:
        return False


class Gsm8kSuite:
    name = "gsm8k"
    description = "Grade-school math word problems; agent loop with the calculator tool."
    system_prompt = (
        "Solve the math word problem step by step. Use the calculator tool for arithmetic. "
        "End your reply with a line of the form 'Final answer: <number>'."
    )

    def load(self, path: str | Path, options: Mapping[str, Any]) -> list[BenchItem]:
        items = []
        for index, record in enumerate(load_records(path)):
            question, answer = record.get("question"), record.get("answer")
            if not isinstance(question, str) or not isinstance(answer, str):
                raise DatasetError(f"gsm8k record {index} needs string 'question' and 'answer'")
            expected = answer.split("####")[-1].strip().replace(",", "")
            items.append(BenchItem(str(record.get("id", f"gsm8k-{index}")), {"question": question}, expected))
        return items

    async def run(self, item: BenchItem, context: BenchContext) -> ItemResult:
        started = time.monotonic()
        outcome = await context.run_agent(
            [Message(Role.SYSTEM, self.system_prompt), Message(Role.USER, item.input["question"])],
            create_default_runtime(),
        )
        predicted = extract_number(outcome.output)
        return _result_from_outcome(
            item, outcome, started,
            correct=numbers_equal(predicted, item.expected),
            expected=str(item.expected),
            predicted=predicted or "",
            details={"tool_calls": outcome.tool_calls, "tool_errors": outcome.tool_errors},
        )


# --- Generic QA ------------------------------------------------------------------

def normalize_answer(text: str) -> str:
    """SQuAD-style normalization: lowercase, drop punctuation and articles."""

    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


class QaSuite:
    name = "qa"
    description = "Open question answering scored by exact or contained normalized match."

    def load(self, path: str | Path, options: Mapping[str, Any]) -> list[BenchItem]:
        question_field = str(options.get("question_field", "question"))
        answer_field = options.get("answer_field")
        id_field = str(options.get("id_field", "id"))
        items = []
        for index, record in enumerate(load_records(path)):
            question = record.get(question_field)
            field_name = answer_field or ("answers" if "answers" in record else "answer")
            answers = record.get(str(field_name))
            if isinstance(answers, str):
                answers = [answers]
            if isinstance(answers, Mapping):  # e.g. TriviaQA {"aliases": [...], "value": ...}
                answers = list(answers.get("aliases") or []) + [answers.get("value")]
            if not isinstance(question, str) or not isinstance(answers, list):
                raise DatasetError(
                    f"qa record {index} needs '{question_field}' and '{field_name}' fields"
                )
            clean = [str(a) for a in answers if isinstance(a, (str, int, float)) and str(a).strip()]
            if not clean:
                raise DatasetError(f"qa record {index} has no usable answers")
            items.append(BenchItem(str(record.get(id_field, f"qa-{index}")), {"question": question}, clean))
        return items

    async def run(self, item: BenchItem, context: BenchContext) -> ItemResult:
        started = time.monotonic()
        use_tools = context.options.get("tools", "none") == "default"
        tools = create_default_runtime() if use_tools else ToolRuntime(ToolRegistry())
        system = str(context.options.get(
            "system", "Answer the question concisely. End with 'Final answer: <answer>'."
        ))
        outcome = await context.run_agent(
            [Message(Role.SYSTEM, system), Message(Role.USER, item.input["question"])], tools
        )
        answer = outcome.output
        match = re.search(r"final answer\s*[:：]\s*(.+)", answer, re.IGNORECASE | re.DOTALL)
        if match:
            answer = match.group(1).strip().splitlines()[0]
        predicted = normalize_answer(answer)
        expected = [normalize_answer(a) for a in item.expected]
        if context.options.get("match", "contains") == "exact":
            correct = predicted in expected
        else:
            correct = any(e and e in normalize_answer(outcome.output) for e in expected)
        return _result_from_outcome(
            item, outcome, started, correct=correct, expected=" | ".join(item.expected[:5]),
            predicted=answer[:500],
        )


# --- BFCL (function calling) -----------------------------------------------------

_TYPE_MAP = {"dict": "object", "float": "number", "tuple": "array", "int": "integer",
             "str": "string", "bool": "boolean", "list": "array"}
_SAFE_NAME = re.compile(r"[^A-Za-z0-9_-]")


def bfcl_schema(schema: Any) -> Any:
    """Convert BFCL's Python-flavoured parameter schema to JSON Schema."""

    if isinstance(schema, list):
        return [bfcl_schema(item) for item in schema]
    if not isinstance(schema, Mapping):
        return schema
    converted: dict[str, Any] = {}
    for key, value in schema.items():
        if key == "type":
            if value == "any":
                continue
            converted["type"] = _TYPE_MAP.get(value, value)
        elif key == "properties" and isinstance(value, Mapping):
            converted["properties"] = {name: bfcl_schema(sub) for name, sub in value.items()}
        elif key in {"items", "additionalProperties"}:
            converted[key] = bfcl_schema(value)
        elif key in {"description", "required", "enum", "default", "format", "optional"}:
            if key not in {"format", "optional"}:
                converted[key] = value
    return converted


class BfclSuite:
    name = "bfcl"
    description = "Berkeley Function Calling: does the model produce the right tool call(s)?"

    def load(self, path: str | Path, options: Mapping[str, Any]) -> list[BenchItem]:
        answers_path = options.get("answers")
        truth: dict[str, list[Any]] = {}
        if answers_path:
            for record in load_records(answers_path):
                truth[str(record.get("id"))] = list(record.get("ground_truth") or [])
        items = []
        for index, record in enumerate(load_records(path)):
            item_id = str(record.get("id", f"bfcl-{index}"))
            category = item_id.rsplit("_", 1)[0] if "_" in item_id else "default"
            functions = record.get("function") or record.get("functions") or []
            question = record.get("question")
            if isinstance(question, str):
                messages = [{"role": "user", "content": question}]
            elif isinstance(question, list) and question and isinstance(question[0], list):
                messages = question[0]  # first turn
            elif isinstance(question, list):
                messages = question
            else:
                raise DatasetError(f"bfcl record {item_id} has no question")
            if "irrelevance" in category:
                expected: list[Any] = []
            elif item_id in truth:
                expected = truth[item_id]
            elif "ground_truth" in record:
                expected = list(record["ground_truth"])
            else:
                raise DatasetError(
                    f"bfcl record {item_id} has no ground truth; pass --answers <possible_answer file>"
                )
            items.append(
                BenchItem(item_id, {"messages": messages, "functions": functions}, expected, category)
            )
        return items

    async def run(self, item: BenchItem, context: BenchContext) -> ItemResult:
        started = time.monotonic()
        names: dict[str, str] = {}
        specs = []
        for function in item.input["functions"]:
            original = str(function["name"])
            safe = _SAFE_NAME.sub("_", original)[:64]
            names[safe] = original
            specs.append(ToolSpec(
                name=safe,
                description=str(function.get("description", ""))[:1024],
                input_schema=bfcl_schema(function.get("parameters") or {"type": "object"}),
            ))
        messages = tuple(
            Message(Role(m["role"]) if m["role"] in {"user", "system"} else Role.USER, str(m["content"]))
            for m in item.input["messages"]
        )
        client = context.models.resolve(context.profile)
        try:
            response = await client.complete(
                ModelRequest(messages=messages, tools=tuple(specs), max_output_tokens=1024)
            )
        except HarnessError as exc:
            return ItemResult(item.id, item.category, False, error=f"{type(exc).__name__}: {exc}",
                              status="error", latency_seconds=time.monotonic() - started,
                              expected=_json(item.expected))
        calls = [(names.get(c.name, c.name), dict(c.arguments)) for c in response.tool_calls]
        correct, reason = bfcl_match(calls, item.expected)
        return ItemResult(
            id=item.id, category=item.category, correct=correct,
            expected=_json(item.expected), predicted=_json([{n: a} for n, a in calls]),
            status="completed", steps=1, tokens=response.usage.total_tokens or 0,
            cost_usd=response.usage.cost_usd or 0.0, latency_seconds=time.monotonic() - started,
            details={"reason": reason} if reason else {},
        )


def bfcl_match(calls: Sequence[tuple[str, Mapping[str, Any]]], truth: Sequence[Any]) -> tuple[bool, str]:
    if not truth:
        return (not calls, "" if not calls else "expected no tool call")
    if len(calls) != len(truth):
        return False, f"expected {len(truth)} call(s), got {len(calls)}"
    remaining = list(truth)
    for name, arguments in calls:
        for index, entry in enumerate(remaining):
            if _entry_matches(name, arguments, entry):
                del remaining[index]
                break
        else:
            return False, f"no expected call matches {name}({_json(arguments)})"
    return True, ""


def _entry_matches(name: str, arguments: Mapping[str, Any], entry: Any) -> bool:
    if not isinstance(entry, Mapping) or len(entry) != 1:
        return False
    (expected_name, params), = entry.items()
    if name != expected_name or not isinstance(params, Mapping):
        return False
    if set(arguments) - set(params):
        return False
    for param, allowed in params.items():
        allowed = allowed if isinstance(allowed, list) else [allowed]
        if param not in arguments:
            if "" not in allowed:
                return False
            continue
        if not any(_value_matches(arguments[param], option) for option in allowed if option != ""):
            return False
    return True


def _value_matches(value: Any, expected: Any) -> bool:
    if isinstance(expected, bool) or isinstance(value, bool):
        return value == expected
    if isinstance(expected, (int, float)) and isinstance(value, (int, float)):
        return math.isclose(float(value), float(expected), rel_tol=1e-9, abs_tol=1e-9)
    if isinstance(expected, str) and isinstance(value, str):
        return _norm_text(value) == _norm_text(expected)
    if isinstance(expected, list) and isinstance(value, (list, tuple)):
        return len(value) == len(expected) and all(_value_matches(v, e) for v, e in zip(value, expected))
    if isinstance(expected, Mapping) and isinstance(value, Mapping):
        if set(value) - set(expected):
            return False
        for key, options in expected.items():
            options = options if isinstance(options, list) else [options]
            if key not in value:
                if "" not in options:
                    return False
            elif not any(_value_matches(value[key], o) for o in options if o != ""):
                return False
        return True
    return value == expected


def _norm_text(text: str) -> str:
    return re.sub(r"[\s,./\-_*^'\"]", "", text).lower()


def _json(value: Any) -> str:
    import json

    return json.dumps(value, sort_keys=True, default=str)[:2000]


# --- HumanEval ---------------------------------------------------------------------

_CODE_BLOCK = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


def extract_code(text: str) -> str:
    blocks = _CODE_BLOCK.findall(text)
    return max(blocks, key=len) if blocks else text


class HumanEvalSuite:
    name = "humaneval"
    description = "Python function synthesis; generated code runs in the sandbox against tests."
    system_prompt = (
        "You are an expert Python programmer. Complete the function. Reply with the complete "
        "function definition (including any imports it needs) in one ```python code block."
    )

    def load(self, path: str | Path, options: Mapping[str, Any]) -> list[BenchItem]:
        items = []
        for index, record in enumerate(load_records(path)):
            missing = [k for k in ("prompt", "test", "entry_point") if not isinstance(record.get(k), str)]
            if missing:
                raise DatasetError(f"humaneval record {index} is missing {missing}")
            items.append(BenchItem(str(record.get("task_id", f"humaneval-{index}")), record, "tests pass"))
        return items

    async def run(self, item: BenchItem, context: BenchContext) -> ItemResult:
        if context.execution is None:
            raise BenchConfigError("humaneval needs a sandbox: use the default Docker runtime, "
                             "or --unsafe-local-execution for trusted local runs")
        started = time.monotonic()
        prompt, entry = item.input["prompt"], item.input["entry_point"]
        outcome = await context.run_agent(
            [Message(Role.SYSTEM, self.system_prompt), Message(Role.USER, prompt)],
            ToolRuntime(ToolRegistry()),
            budget=RunBudget(max_steps=2, max_wall_seconds=context.budget.max_wall_seconds),
        )
        code = extract_code(outcome.output)
        # A full definition overrides the prompt's stub; a bare body continues it.
        joiner = "\n\n" if re.search(rf"^\s*def\s+{re.escape(entry)}\b", code, re.MULTILINE) else ""
        program = f"{prompt}{joiner}{code}\n\n{item.input['test']}\n\ncheck({entry})\n"
        timeout = float(context.options.get("exec_timeout", 15.0))
        with tempfile.TemporaryDirectory(prefix="humaneval-", dir=context.options.get("work_dir")) as scratch:
            workspace = Path(scratch)
            (workspace / "solution.py").write_text(program, encoding="utf-8")
            runtime = context.execution(workspace)
            result = await runtime.run(["python3", "solution.py"], cwd=workspace, timeout_seconds=timeout)
        passed = result.exit_code == 0 and not result.timed_out
        failure = "" if passed else ("timed out" if result.timed_out else result.stderr.strip()[-500:])
        return _result_from_outcome(
            item, outcome, started, correct=passed, expected="tests pass",
            predicted="tests pass" if passed else "tests fail",
            details={"failure": failure} if failure else {},
        )


class GuardrailsSuite:
    """Red-team evaluation of the input guard (Phase 12a).

    Items are ``{"id", "text", "label": "attack" | "benign"}``; the category is the
    label, so the report's per-category accuracy is the attack detection rate and
    the benign pass rate (1 - false-positive rate). With ``--option model_guard=true``
    the benchmark's profile (e.g. llama-guard3 on Ollama) also classifies each text;
    ``--option deterministic=false`` evaluates the model guard alone.
    """

    name = "guardrails"
    description = "Prompt-injection/jailbreak detection: attacks blocked, benign prompts allowed."

    # Findings that mean "this is an attack" (zero-width or bidi characters alone are
    # common in pasted text, so they are cleaned, not counted as attacks).
    _SIGNALS = {"injection", "prompt_extraction", "encoded_payloads", "denylist"}

    def load(self, path: str | Path, options: Mapping[str, Any]) -> list[BenchItem]:
        items = []
        for index, record in enumerate(load_records(path)):
            text, label = record.get("text"), record.get("label")
            if not isinstance(text, str) or label not in {"attack", "benign"}:
                raise DatasetError(f"guardrails record {index} needs 'text' and label 'attack' or 'benign'")
            items.append(BenchItem(str(record.get("id", f"guard-{index}")), {"text": text}, label, category=label))
        return items

    async def run(self, item: BenchItem, context: BenchContext) -> ItemResult:
        from agent_harness.guardrails import Guardrails, GuardrailsConfig
        from agent_harness.guardrails.config import ModelGuardConfig

        started = time.monotonic()
        use_model = _truthy(context.options.get("model_guard", False))
        deterministic = _truthy(context.options.get("deterministic", True))
        config = GuardrailsConfig(enabled=True, model=ModelGuardConfig(
            profile=context.profile if use_model else "", check=("input",)))
        guard = Guardrails(config, models=context.models)
        text = item.input["text"]
        signals: list[str] = []
        if deterministic:
            decision = guard.check_input(text)
            signals = sorted({f"{f.detector}:{f.category}" for f, action in decision.findings
                              if action != "allow" and (f.detector in self._SIGNALS
                                                        or f.category == "tag_characters")})
        tokens = 0
        error = None
        if use_model and not signals:
            verdict = await guard.classify(text)
            tokens = verdict.usage.total_tokens or 0
            error = verdict.error
            if not verdict.safe:
                signals = [f"model:{c}" for c in verdict.categories]
        predicted = "attack" if signals else "benign"
        return ItemResult(
            item.id, item.category, predicted == item.expected, expected=str(item.expected),
            predicted=predicted, status="error" if error else "completed", tokens=tokens,
            latency_seconds=time.monotonic() - started, error=error, details={"signals": signals},
        )


def _truthy(value: Any) -> bool:
    return value is True or str(value).strip().lower() in {"1", "true", "yes", "on"}


SUITES: dict[str, Suite] = {
    suite.name: suite
    for suite in (Gsm8kSuite(), QaSuite(), BfclSuite(), HumanEvalSuite(), GuardrailsSuite())
}
