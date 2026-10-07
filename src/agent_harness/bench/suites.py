"""Benchmark suites for public datasets.

Each suite loads records from a local file (you download the dataset), runs
every item through a specific layer of the harness, and scores it:

=========  ==================================  ===================================
suite      what it validates                   dataset format (per record)
=========  ==================================  ===================================
gsm8k      full agent loop + tool use          {"question", "answer": "... #### 42"}
           (calculator), budgets, retries
qa         agent answers without tools or      {"question", "answer" | "answers"}
           with default tools (TriviaQA,       (field names configurable; optional
           NQ-open, HotpotQA-closed...);       "rubric" for the judge)
           match = "judge" grades open-ended
           answers with a model
bfcl       tool-calling accuracy of the model  BFCL question file + possible-answer
           gateway per provider/adapter        file joined by "id"
humaneval  coding: generated code executed     {"task_id", "prompt", "test",
           in the sandbox (Docker by default)   "entry_point"}
context    long conversations past the         {"needle", "question", "answer",
           profile's context window: trimming, "filler": prose|json|code}
           server token counts, overflow       (synthetic; no download)
           recovery
memory     long-term memory: notes saved, then {"notes": [...], "question",
           found with memory_search (keyword,  "answer"}
           semantic, or hybrid search)
agents     delegation when told to delegate    {"notes", "question", "answer"}
           (only the agent tool is offered)    (synthetic; no download)
tasks      real tasks in the full application  {"prompt", "facts", "rubric",
           (the config under test): answer,    "expect", "workspace"}
           and how the agent chose to work     (see TasksSuite)
=========  ==================================  ===================================

Scores are the harness's own approximations; for leaderboard-comparable numbers
use each benchmark's official evaluator on the saved predictions.
"""

from __future__ import annotations

import asyncio
import json
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
            item_input: dict[str, Any] = {"question": question}
            if isinstance(record.get("rubric"), str):
                item_input["rubric"] = record["rubric"]
            items.append(BenchItem(str(record.get(id_field, f"qa-{index}")), item_input, clean))
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
        match_mode = context.options.get("match", "contains")
        details: dict[str, Any] = {}
        if match_mode == "judge":
            verdict = await judge_answer(context, item.input["question"], item.expected, outcome.output,
                                         rubric=item.input.get("rubric"))
            correct = verdict["correct"] is True
            details["judge"] = verdict
        elif match_mode == "exact":
            correct = predicted in expected
        else:
            correct = any(e and e in normalize_answer(outcome.output) for e in expected)
        result = _result_from_outcome(
            item, outcome, started, correct=correct, expected=" | ".join(item.expected[:5]),
            predicted=answer[:500],
        )
        result.details.update(details)
        return result


JUDGE_INSTRUCTIONS = (
    "You grade an answer to a question against reference answers. The answer is correct when it "
    "states the same fact or result as a reference, even in other words; extra explanation is fine, "
    "but a contradiction, a different result, or no answer is incorrect. Follow the rubric if one is "
    'given. Reply with JSON only: {"correct": true or false, "reason": "<one sentence>"}'
)


async def judge_answer(
    context: BenchContext,
    question: str,
    references: Sequence[str],
    answer: str,
    *,
    rubric: str | None = None,
) -> dict[str, Any]:
    """Ask a model (``judge_profile``, default the profile under test) whether ``answer`` is right.

    Returns ``{"correct": bool | None, "reason": str, "profile": str}``; ``None`` means the
    judge's reply could not be read (counted as incorrect).
    """

    profile = str(context.options.get("judge_profile") or context.profile)
    parts = [f"Question:\n{question}", "Reference answers:\n" + "\n".join(f"- {r}" for r in references)]
    if rubric:
        parts.append(f"Rubric:\n{rubric}")
    parts.append(f"Answer to grade:\n{answer[:8_000]}")
    request = ModelRequest(
        messages=(Message(Role.SYSTEM, JUDGE_INSTRUCTIONS), Message(Role.USER, "\n\n".join(parts))),
        temperature=0.0,
        max_output_tokens=int(context.options.get("judge_max_tokens", 1024)),
    )
    try:
        reply = await context.models.resolve(profile).complete(request)
    except HarnessError as exc:
        return {"correct": None, "reason": f"judge call failed: {exc}", "profile": profile}
    correct, reason = parse_verdict(reply.content)
    return {"correct": correct, "reason": reason, "profile": profile}


def parse_verdict(text: str) -> tuple[bool | None, str]:
    """Read a judge reply: JSON ``{"correct": ...}``, else a ``correct: true`` / CORRECT word."""

    for candidate in re.findall(r"\{[^{}]*\}", text, re.DOTALL):
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and isinstance(value.get("correct"), bool):
            return value["correct"], str(value.get("reason", ""))[:500]
    match = re.search(r'"?correct"?\s*[:=]\s*(true|false)', text, re.IGNORECASE)
    if match:
        return match.group(1).lower() == "true", ""
    words = re.findall(r"\b(incorrect|correct)\b", text, re.IGNORECASE)
    if len({word.lower() for word in words}) == 1:
        return words[0].lower() == "correct", ""
    return None, "the judge's reply could not be read"


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
        try:
            # One direct call: BFCL scores raw function calling, not the agent loop.
            response = await context.call_model(
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


class SandboxError(RuntimeError):
    """The code sandbox (Docker) failed; the item is an error, not a wrong answer."""


# docker run exits 125 when the daemon or the run itself failed (not the program).
_DOCKER_FAILURE_EXIT = 125
_DOCKER_FAILURE_TEXT = re.compile(
    r"Cannot connect to the Docker daemon|error during connect|Error response from daemon|"
    r"docker: .*(?:error|denied)|Unable to find image|pull access denied",
    re.IGNORECASE,
)


async def docker_problem(binary: str = "docker", timeout_seconds: float = 15.0) -> str | None:
    """Why Docker cannot run sandboxes right now (``None`` when ``docker info`` answers)."""

    try:
        process = await asyncio.create_subprocess_exec(
            binary, "info", "--format", "{{.ServerVersion}}",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
    except OSError as exc:
        return f"cannot start {binary!r}: {exc}"
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout_seconds)
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()
        return f"`{binary} info` did not answer within {timeout_seconds:.0f}s (is the daemon stuck?)"
    if process.returncode != 0 or not stdout.strip():
        detail = (stderr or stdout).decode("utf-8", "replace").strip().splitlines()
        return f"`{binary} info` failed: {detail[-1] if detail else f'exit {process.returncode}'}"
    return None


async def _raise_for_sandbox_failure(result: Any, runtime: Any) -> None:
    """Docker sandbox failures are errors: they say nothing about the generated code."""

    from agent_harness.coding.execution import DockerExecutionRuntime

    if not isinstance(runtime, DockerExecutionRuntime):
        return  # local execution (or a test double): exit codes are the program's own
    if result.exit_code == _DOCKER_FAILURE_EXIT or (
        result.exit_code != 0 and _DOCKER_FAILURE_TEXT.search(result.stderr or "")
    ):
        raise SandboxError(f"sandbox failed (exit {result.exit_code}): {(result.stderr or '').strip()[-300:]}")
    if result.timed_out:
        # A stuck daemon also looks like a timeout; only a healthy Docker makes it the code's fault.
        problem = await docker_problem(runtime.config.docker_binary)
        if problem is not None:
            raise SandboxError(f"sandbox timed out and Docker is unavailable: {problem}")


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
        await _raise_for_sandbox_failure(result, runtime)
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


# --- Context: conversations longer than the window -------------------------------------


def filler_text(kind: str, seed: str, chars: int) -> str:
    """Deterministic filler. JSON and code use more tokens per character than prose,
    which is where a character-based estimate undercounts."""

    import random

    rng = random.Random(seed)
    words = ("alpha", "river", "stone", "market", "signal", "garden", "copper", "window", "harbor",
             "lantern", "meadow", "engine", "silver", "orbit", "canvas", "timber", "valley", "ember")
    lines: list[str] = []
    while sum(len(line) + 1 for line in lines) < chars:
        if kind == "json":
            lines.append(json.dumps({"id": rng.randint(1000, 99999), "sku": f"{rng.choice('ABCDEFGH')}"
                                     f"{rng.randint(10, 99)}-{rng.randint(100, 999)}", "qty": rng.randint(1, 40),
                                     "price": round(rng.uniform(1, 500), 2), "tag": rng.choice(words)}))
        elif kind == "code":
            name = f"{rng.choice(words)}_{rng.randint(1, 99)}"
            lines.append(f"def {name}(x, y={rng.randint(0, 9)}):  return [v * {rng.randint(2, 9)} for v in "
                         f"range(x) if v % {rng.randint(2, 7)} == y]")
        else:
            lines.append(" ".join(rng.choice(words) for _ in range(14)).capitalize() + ".")
    return "\n".join(lines)[:chars]


class ContextSuite:
    name = "context"
    description = "Long conversations past the context window: trimming, server token counts, overflow recovery."
    system_prompt = ("Answer the user's question from the conversation. "
                     "End your reply with a line of the form 'Final answer: <answer>'.")

    def load(self, path: str | Path, options: Mapping[str, Any]) -> list[BenchItem]:
        items = []
        for index, record in enumerate(load_records(path)):
            fields = [record.get(key) for key in ("needle", "question", "answer")]
            if not all(isinstance(value, str) and value for value in fields):
                raise DatasetError(f"context record {index} needs 'needle', 'question', and 'answer'")
            filler = str(record.get("filler", "prose"))
            if filler not in {"prose", "json", "code"}:
                raise DatasetError(f"context record {index}: filler must be prose, json, or code")
            items.append(BenchItem(str(record.get("id", f"context-{index}")),
                                   {"needle": fields[0], "question": fields[1], "filler": filler},
                                   fields[2], category=filler))
        return items

    async def run(self, item: BenchItem, context: BenchContext) -> ItemResult:
        started = time.monotonic()
        turns = int(context.options.get("turns", 14))
        chars = int(context.options.get("chars_per_turn", 1_500))
        messages = [Message(Role.SYSTEM, self.system_prompt),
                    Message(Role.USER, "I will send records in parts. Reply 'noted.' to each; then I will ask a question.")]
        for turn in range(turns):
            text = filler_text(item.input["filler"], f"{item.id}-{turn}", chars)
            if turn == turns - 1:  # the newest part holds the fact, so a correct trim keeps it
                text = f"{text}\nNote: {item.input['needle']}"
            messages += [Message(Role.USER, f"Part {turn + 1}:\n{text}"), Message(Role.ASSISTANT, "noted.")]
        messages.append(Message(Role.USER, item.input["question"]))
        outcome = await context.run_agent(messages, ToolRuntime(ToolRegistry()))
        answer = outcome.output
        match = re.search(r"final answer\s*[:：]\s*(.+)", answer, re.IGNORECASE | re.DOTALL)
        predicted = match.group(1).strip().splitlines()[0] if match else answer.strip()[-200:]
        correct = normalize_answer(str(item.expected)) in normalize_answer(answer)
        responded = [e.payload for e in outcome.events if e.type.value == "model_responded"]
        details = {
            "messages_in": len(messages),
            "omitted_messages": max((p.get("context", {}).get("omitted_messages", 0) for p in responded), default=0),
            "context_capped": sum(e.type.value == "context_capped" for e in outcome.events),
            "context_overflows": sum(p.get("context_overflow") is True for p in responded),
        }
        return _result_from_outcome(item, outcome, started, correct=correct, expected=str(item.expected),
                                    predicted=predicted[:200], details=details)


# --- Memory: save notes, then find them with memory_search ------------------------------


class MemorySuite:
    name = "memory"
    description = "Long-term memory: notes are saved, then found with memory_search (keyword, semantic, hybrid)."
    system_prompt = ("You have a memory_search tool over saved notes. Search the notes before answering; "
                     "the question may use different words than the notes. "
                     "End your reply with a line of the form 'Final answer: <answer>'.")

    def load(self, path: str | Path, options: Mapping[str, Any]) -> list[BenchItem]:
        items = []
        for index, record in enumerate(load_records(path)):
            notes, question, answer = record.get("notes"), record.get("question"), record.get("answer")
            if (not isinstance(notes, list) or not all(isinstance(n, str) and n for n in notes)
                    or not isinstance(question, str) or not isinstance(answer, str)):
                raise DatasetError(f"memory record {index} needs 'notes' (strings), 'question', and 'answer'")
            items.append(BenchItem(str(record.get("id", f"memory-{index}")),
                                   {"notes": notes, "question": question}, answer))
        return items

    async def run(self, item: BenchItem, context: BenchContext) -> ItemResult:
        from agent_harness.knowledge.memory import MEMORY_SEARCH_SPEC, SQLiteMemoryStore, memory_tool_handlers
        from agent_harness.knowledge.semantic_memory import SemanticMemoryStore, VectorIndex

        started = time.monotonic()
        mode = str(context.options.get("search", "hybrid"))
        if mode not in {"keyword", "semantic", "hybrid"}:
            raise BenchConfigError("memory: options.search must be keyword, semantic, or hybrid")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.sqlite3"
            store: Any = SQLiteMemoryStore(path, namespace="bench")
            try:
                if mode != "keyword":
                    # Default: the profile under test embeds with its own embedding_model.
                    profile = context.options.get("embedding_profile") or context.embedding_profile
                    if not profile:
                        raise BenchConfigError("memory: options.embedding_profile (or embedding_model on "
                                               f"the profile under test) is required for search = {mode!r}")
                    embed = getattr(context.models.resolve(str(profile)), "embed", None)
                    if not callable(embed):
                        raise BenchConfigError(f"memory: profile {profile!r} has no embeddings endpoint")
                    store = SemanticMemoryStore(store, embed, VectorIndex(path), model=str(profile), mode=mode)
                for note in item.input["notes"]:
                    await store.put(note)
                if mode != "keyword":
                    await store.reindex()  # raises if the embedding endpoint does not work
                runtime = ToolRuntime(ToolRegistry())
                search, _ = memory_tool_handlers(store)
                runtime.register(MEMORY_SEARCH_SPEC, search)
                outcome = await context.run_agent(
                    [Message(Role.SYSTEM, self.system_prompt), Message(Role.USER, item.input["question"])], runtime
                )
            finally:
                await store.close()
        correct = normalize_answer(str(item.expected)) in normalize_answer(outcome.output)
        match = re.search(r"final answer\s*[:：]\s*(.+)", outcome.output, re.IGNORECASE | re.DOTALL)
        predicted = match.group(1).strip().splitlines()[0] if match else outcome.output.strip()[-200:]
        return _result_from_outcome(item, outcome, started, correct=correct, expected=str(item.expected),
                                    predicted=predicted[:200],
                                    details={"search": mode, "tool_calls": outcome.tool_calls,
                                             "searched": "memory_search" in outcome.tool_calls})


class AgentsSuite:
    """Delegation to subagents: the top-level agent can only use the ``agent`` tool.

    Each item has notes, a question with independent parts, and the expected facts. The
    parent must delegate to the ``researcher`` subagent, which can list and read the notes
    (in parallel when it calls ``agent`` several times in one reply), then combine the
    answers. An item is correct when the final answer contains every expected fact; how it
    delegated is reported, not scored (details: subagents, max_parallel, facts_found).
    """

    name = "agents"
    description = "Subagents: delegate independent parts to a researcher subagent, then combine the answers."
    system_prompt = (
        "You coordinate other agents. You cannot read the notes yourself: use the agent tool to "
        "delegate each independent part of the question to the researcher subagent. Delegate "
        "independent parts in the same reply so they run in parallel. Then combine the answers. "
        "End your reply with a line of the form 'Final answer: <answer>'."
    )
    researcher_prompt = (
        "Answer the task from the notes. Use list_notes to see what exists and read_note to read a "
        "note. Reply with the fact asked for and the note it came from; do not guess."
    )

    def load(self, path: str | Path, options: Mapping[str, Any]) -> list[BenchItem]:
        items = []
        for index, record in enumerate(load_records(path)):
            notes, question, answer = record.get("notes"), record.get("question"), record.get("answer")
            if isinstance(answer, str):
                answer = [answer]
            if (not isinstance(notes, dict) or not notes
                    or not all(isinstance(k, str) and isinstance(v, str) for k, v in notes.items())
                    or not isinstance(question, str) or not isinstance(answer, list)
                    or not answer or not all(isinstance(a, str) and a for a in answer)):
                raise DatasetError(f"agents record {index} needs notes (name -> text), question, and answer")
            items.append(BenchItem(str(record.get("id", f"agents-{index}")),
                                   {"notes": notes, "question": question},
                                   answer, str(record.get("category", "default"))))
        return items

    async def run(self, item: BenchItem, context: BenchContext) -> ItemResult:
        from agent_harness.orchestration.agents import AGENT_TOOL, SubagentDefinition
        from agent_harness.domain import EventType, ToolRisk

        started = time.monotonic()
        notes: Mapping[str, str] = item.input["notes"]
        runtime = ToolRuntime(ToolRegistry())

        async def list_notes(arguments: Mapping[str, Any]) -> dict[str, Any]:
            return {"notes": sorted(notes)}

        async def read_note(arguments: Mapping[str, Any]) -> dict[str, Any]:
            name = str(arguments.get("name", ""))
            if name not in notes:
                return {"error": f"no note named {name!r}", "notes": sorted(notes)}
            return {"name": name, "text": notes[name]}

        runtime.register(ToolSpec(name="list_notes", description="List the names of the notes.",
                                  input_schema={"type": "object", "properties": {}},
                                  risk=ToolRisk.READ_ONLY), list_notes)
        runtime.register(ToolSpec(name="read_note", description="Read one note by name.",
                                  input_schema={"type": "object", "required": ["name"],
                                                "properties": {"name": {"type": "string"}}},
                                  risk=ToolRisk.READ_ONLY), read_note)
        researcher = SubagentDefinition(
            name="researcher", description="Finds one fact in the notes and reports it with its source.",
            prompt=self.researcher_prompt, tools=("list_notes", "read_note"), max_steps=6,
        )
        outcome = await context.run_agent(
            [Message(Role.SYSTEM, self.system_prompt), Message(Role.USER, item.input["question"])],
            runtime, subagents=(researcher,), allowed_tools=(AGENT_TOOL,),
        )
        started_events = [e for e in outcome.events if e.type is EventType.SUBAGENT_STARTED]
        per_step: dict[int, int] = {}
        for event in started_events:
            per_step[event.step] = per_step.get(event.step, 0) + 1
        subagents = len(started_events)
        found = [fact for fact in item.expected if normalize_answer(fact) in normalize_answer(outcome.output)]
        correct = len(found) == len(item.expected)
        reason = "" if correct else f"missing {sorted(set(item.expected) - set(found))}"
        result = _result_from_outcome(item, outcome, started, correct=correct,
                                      expected="; ".join(item.expected), predicted=outcome.output.strip()[-200:])
        result.details.update({
            "subagents": subagents,
            "max_parallel": max(per_step.values(), default=0),
            "facts_found": len(found),
            "reason": reason,
        })
        return result


_TASK_EXPECT = {"plan", "delegate", "plan_or_delegate", "max_steps", "tools", "no_tools"}
# Said to the agent when it asks the user a question during an evaluation.
_TASK_ANSWER = "No one can answer questions during this evaluation; use your best judgment and continue."
_TASK_DENIAL = "Not allowed during this evaluation; continue without it."


class TasksSuite:
    """Real tasks in the full application: the model chooses how to work.

    Unlike the other suites, nothing is set up for the model: each item runs in the
    application built from the config under test (its tools, instructions, subagents,
    planning mode, guardrails, context policy, and summaries), with no instruction to
    plan or delegate. An item is graded on its answer (``facts`` that must appear,
    and an optional ``rubric`` for a judge model) and on how it worked (``expect``:
    ``plan``, ``delegate``, ``plan_or_delegate``, ``max_steps``, ``tools``,
    ``no_tools``). ``options.score = "answer"`` keeps how it worked report-only.
    ``options.repeats = k`` runs each task k times (pass^k: every repeat correct).

    ``workspace`` is ``"repo"`` (default: the config's own coding workspace, used
    read-only: write and command approvals are denied) or a table of files written to
    a temporary workspace (``approvals = "allow"`` lets the agent change them).
    """

    name = "tasks"
    description = "Real tasks in the full application, graded on the answer and on how the agent worked."

    def load(self, path: str | Path, options: Mapping[str, Any]) -> list[BenchItem]:
        repeats = options.get("repeats", 1)
        if not isinstance(repeats, int) or isinstance(repeats, bool) or repeats < 1:
            raise BenchConfigError("tasks: options.repeats must be a positive integer")
        items = []
        for index, record in enumerate(load_records(path)):
            prompt, facts, rubric = record.get("prompt"), record.get("facts", []), record.get("rubric")
            expect, workspace = record.get("expect", {}), record.get("workspace", "repo")
            approvals = record.get("approvals", "deny")
            if isinstance(facts, str):
                facts = [facts]
            problems = []
            if not isinstance(prompt, str) or not prompt.strip():
                problems.append("a prompt")
            if not isinstance(facts, list) or not all(isinstance(f, str) and f for f in facts):
                problems.append("facts as a list of strings")
            if rubric is not None and not isinstance(rubric, str):
                problems.append("a rubric string")
            if not facts and not rubric:
                problems.append("facts or a rubric to grade the answer")
            if not isinstance(expect, dict) or set(expect) - _TASK_EXPECT:
                problems.append(f"expect with only {', '.join(sorted(_TASK_EXPECT))}")
            if workspace != "repo" and not (isinstance(workspace, dict) and all(
                    isinstance(k, str) and isinstance(v, str) for k, v in workspace.items())):
                problems.append('workspace "repo" or a table of file name -> text')
            if approvals not in {"deny", "allow"}:
                problems.append('approvals "deny" or "allow"')
            if problems:
                raise DatasetError(f"tasks record {index} needs " + "; ".join(problems))
            task_id = str(record.get("id", f"task-{index}"))
            task_input = {"task": task_id, "prompt": prompt, "rubric": rubric, "expect": dict(expect),
                          "workspace": workspace, "approvals": approvals}
            category = str(record.get("category", "default"))
            for repeat in range(1, repeats + 1):
                item_id = task_id if repeats == 1 else f"{task_id}#{repeat}"
                items.append(BenchItem(item_id, task_input, list(facts), category))
        return items

    async def run(self, item: BenchItem, context: BenchContext) -> ItemResult:
        from agent_harness.application import build_application
        from agent_harness.coding import CODING_AGENT_SYSTEM_PROMPT
        from agent_harness.domain import RunStatus

        if context.config_path is None:
            raise BenchConfigError("tasks: needs the config under test (run it from a benchmark spec)")
        started = time.monotonic()
        workspace = item.input["workspace"]
        with tempfile.TemporaryDirectory(prefix="agent-harness-task-") as scratch:
            root = Path(scratch)
            overrides: dict[str, Any] = {"artifacts": {"directory": str(root / "artifacts")}}
            if isinstance(workspace, Mapping):
                files = root / "workspace"
                for name, text in workspace.items():
                    target = files / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(text, encoding="utf-8")
                files.mkdir(exist_ok=True)
                overrides["coding"] = {"workspace": str(files)}
            app = build_application(
                config_path=context.config_path, database_path=root / "runs.sqlite3",
                config_overrides=overrides, model_resolver=context.models,
                # The app closes its telemetry; the benchmark's sink must outlive each task.
                telemetry=_Unclosed(context.telemetry) if context.telemetry is not None else None,
            )
            async with app:
                await app.start_tools()
                orchestrator = app.orchestrator
                messages = []
                if app.config.coding.enabled:
                    messages.append(Message(Role.SYSTEM, CODING_AGENT_SYSTEM_PROMPT))
                messages.append(Message(Role.USER, item.input["prompt"]))
                run = await orchestrator.create_run(
                    model_profile=context.profile, messages=messages, budget=context.budget,
                    versions=app.run_versions(context.profile),
                )
                run = await orchestrator.run(run.id)
                decisions = 0
                while run.status is RunStatus.AWAITING_INPUT and decisions < 20:
                    pending = await orchestrator.pending_approval(run.id)
                    if pending is None:
                        break
                    decisions += 1
                    if pending.kind == "question":
                        await orchestrator.resolve_approval(run.id, call_id=pending.call_id, approved=True,
                                                            answer=_TASK_ANSWER, actor="bench")
                    else:
                        allow = item.input["approvals"] == "allow"
                        await orchestrator.resolve_approval(
                            run.id, call_id=pending.call_id, approved=allow,
                            feedback=None if allow else _TASK_DENIAL, actor="bench")
                    run = await orchestrator.resume(run.id)
                events = list(await app.store.list_events(run.id))
        context.record_transcript(run, events)
        output = next((m.content for m in reversed(run.messages)
                       if m.role is Role.ASSISTANT and not m.tool_calls and m.content.strip()), "")
        behavior = task_behavior(events, run.step_count)
        behavior["decisions"] = decisions
        found = [fact for fact in item.expected if normalize_answer(fact) in normalize_answer(output)]
        answer_ok = run.status is RunStatus.COMPLETED and len(found) == len(item.expected)
        details: dict[str, Any] = {"task": item.input["task"], "facts_found": len(found),
                                   "missing_facts": [f for f in item.expected if f not in found], **behavior}
        if item.input["rubric"] and run.status is RunStatus.COMPLETED:
            verdict = await judge_answer(context, item.input["prompt"], item.expected or ["(see the rubric)"],
                                         output, rubric=item.input["rubric"])
            details["judge"] = verdict
            answer_ok = answer_ok and verdict["correct"] is True
        elif item.input["rubric"]:
            answer_ok = False
        failures = behavior_failures(item.input["expect"], behavior)
        behavior_ok = not failures
        details.update(answer_ok=answer_ok, behavior_ok=behavior_ok, behavior_failures=failures)
        scored = context.options.get("score", "all")
        if scored not in {"all", "answer"}:
            raise BenchConfigError('tasks: options.score must be "all" or "answer"')
        correct = answer_ok and (behavior_ok or scored == "answer")
        usage_tokens = run.consumed_tokens
        return ItemResult(
            id=item.id, category=item.category, correct=correct,
            expected="; ".join(item.expected) or "(rubric)", predicted=output.strip()[-300:],
            status=run.status.value, steps=run.step_count, tokens=usage_tokens,
            cost_usd=run.consumed_cost_usd or 0.0, latency_seconds=time.monotonic() - started,
            error=run.error, details=details,
        )


class _Unclosed:
    """A telemetry sink that ignores close() (one shared sink, one application per task)."""

    def __init__(self, sink: Any) -> None:
        self._sink = sink

    def record(self, record: Any) -> None:
        self._sink.record(record)


def task_behavior(events: Sequence[Any], steps: int) -> dict[str, Any]:
    """How a run worked, read from its events (the ``tasks`` suite's behavior facts)."""

    from agent_harness.domain import EventType

    tools = [str(e.payload.get("tool")) for e in events if e.type is EventType.TOOL_REQUESTED]
    return {
        "steps": steps,
        "tools_used": sorted(set(tools)),
        "tool_calls": len(tools),
        "planned": any(e.type is EventType.PLAN_CREATED for e in events),
        "subagents": sum(e.type is EventType.SUBAGENT_STARTED for e in events),
        "summaries": sum(e.type is EventType.CONTEXT_SUMMARIZED and e.payload.get("ok") is True for e in events),
        "reminded": any(e.type is EventType.PLAN_STEP_UPDATED and e.payload.get("reminder") == "auto_plan"
                        for e in events),
        "guard_flags": sum(e.type is EventType.GUARDRAIL_DECIDED and e.payload.get("action") == "flag"
                           for e in events),
    }


def behavior_failures(expect: Mapping[str, Any], behavior: Mapping[str, Any]) -> list[str]:
    """Each way the run's behavior missed the item's ``expect`` table (empty: as expected)."""

    failures = []
    delegated = behavior["subagents"] > 0
    if "plan" in expect and bool(expect["plan"]) != behavior["planned"]:
        failures.append("made a plan" if behavior["planned"] else "made no plan")
    if "delegate" in expect and bool(expect["delegate"]) != delegated:
        failures.append("delegated" if delegated else "did not delegate")
    if expect.get("plan_or_delegate") and not (behavior["planned"] or delegated):
        failures.append("neither planned nor delegated")
    if expect.get("max_steps") is not None and behavior["steps"] > int(expect["max_steps"]):
        failures.append(f"took {behavior['steps']} steps (expected at most {expect['max_steps']})")
    for tool in expect.get("tools", []):
        if tool not in behavior["tools_used"]:
            failures.append(f"did not use {tool}")
    for tool in expect.get("no_tools", []):
        if tool in behavior["tools_used"]:
            failures.append(f"used {tool}")
    return failures


SUITES: dict[str, Suite] = {
    suite.name: suite
    for suite in (Gsm8kSuite(), QaSuite(), BfclSuite(), HumanEvalSuite(), GuardrailsSuite(), ContextSuite(),
                  MemorySuite(), AgentsSuite(), TasksSuite())
}
