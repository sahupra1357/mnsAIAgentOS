"""Deterministic guardrail detectors: fast, free, and explainable.

Each detector returns ``Finding``s with the span it matched, so an action can
flag, redact (replace the span), or block. Pattern detectors are a first layer
only: a determined attacker can rephrase around them. Least-privilege tools,
approvals, and taint mode are the stronger defenses.
"""

from __future__ import annotations

import base64
import binascii
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from agent_harness.observability.redaction import BUILT_IN_PATTERNS


@dataclass(frozen=True, slots=True)
class Finding:
    detector: str  # e.g. "injection", "pii"
    category: str  # e.g. "instruction_override", "email"
    start: int = 0
    end: int = 0

    @property
    def span(self) -> tuple[int, int]:
        return self.start, self.end


def _compile(patterns: Iterable[str]) -> tuple[re.Pattern[str], ...]:
    return tuple(re.compile(p, re.IGNORECASE) for p in patterns)


# --- prompt injection / jailbreak --------------------------------------------------

_INJECTION: dict[str, tuple[re.Pattern[str], ...]] = {
    "instruction_override": _compile((
        r"\b(?:ignore|disregard|forget|override|bypass|skip)\s+(?:all\s+|any\s+|the\s+|your\s+|my\s+)*"
        r"(?:previous|prior|above|earlier|preceding|initial|original|system)?\s*"
        r"(?:instructions?|prompts?|rules?|directions?|guidelines?|guardrails?|context|constraints)\b",
        r"\bforget\s+(?:everything|all)\s+(?:you\s+(?:were|have\s+been)\s+told|above|before)\b",
        r"\b(?:new|updated|real|actual)\s+(?:system\s+)?(?:instructions?|prompt|rules)\s*:",
        r"\bfrom\s+now\s+on,?\s+you\s+(?:will|must|are|have)\s+(?:no|not|ignore|answer\s+without)",
    )),
    "role_play_jailbreak": _compile((
        r"\byou\s+are\s+(?:now\s+)?(?:DAN|STAN|DUDE|AIM|jailbroken|unfiltered|uncensored|unrestricted)\b",
        r"\bdo\s+anything\s+now\b",
        r"\b(?:developer|god|jailbreak|unrestricted|DAN)\s+mode\s+(?:enabled|on|activated|engaged)\b",
        r"\b(?:enable|activate|enter)\s+(?:developer|god|jailbreak|DAN|unrestricted)\s+mode\b",
        r"\b(?:act|behave|respond)\s+as\s+(?:an?\s+)?(?:unfiltered|uncensored|unrestricted|amoral|evil)\b",
        r"\bpretend\s+(?:that\s+)?you\s+(?:have\s+no|are\s+not\s+bound\s+by|don'?t\s+have|are\s+free\s+of)\s+"
        r"(?:rules|restrictions|guidelines|filters|limits|policies)\b",
        r"\bwithout\s+(?:any\s+)?(?:ethical|moral|safety)\s+(?:restrictions|guidelines|filters|limits)\b",
    )),
    "role_tag_spoofing": _compile((
        r"<\|im_start\|>\s*system", r"<\|start_header_id\|>\s*system", r"\[/?INST\]", r"<<SYS>>",
        r"^\s*#{2,}\s*(?:system|instruction)s?\s*:", r"^\s*system\s*:\s*you\s+(?:are|must)",
    )),
}

_PROMPT_EXTRACTION = _compile((
    r"\b(?:reveal|show|print|display|output|repeat|leak|dump|give\s+me|tell\s+me|what\s+(?:is|are|was|were))\s+"
    r"(?:me\s+)?(?:your|the)\s+(?:full\s+|entire\s+|exact\s+|complete\s+)?"
    r"(?:system|initial|hidden|original|secret|internal|developer)\s+(?:prompt|instructions?|message|rules)\b",
    r"\brepeat\s+(?:the\s+)?(?:text|words|everything|all)\s+(?:above|before|preceding)\b",
    r"\bwhat\s+(?:were|are)\s+you\s+told\s+(?:before|at\s+the\s+(?:start|beginning))\b",
    r"\b(?:start|begin)\s+your\s+(?:reply|answer|response)\s+with\s+[\"']?you\s+are\b",
))


def detect_injection(text: str) -> list[Finding]:
    return [Finding("injection", category, m.start(), m.end())
            for category, patterns in _INJECTION.items() for p in patterns for m in p.finditer(text)]


def detect_prompt_extraction(text: str) -> list[Finding]:
    return [Finding("prompt_extraction", "system_prompt_request", m.start(), m.end())
            for p in _PROMPT_EXTRACTION for m in p.finditer(text)]


# --- hidden characters --------------------------------------------------------------

_HIDDEN = re.compile(
    "[\U000E0000-\U000E007F"       # Unicode tag characters (invisible "ASCII smuggling")
    "\u200b-\u200f"                # zero-width space/joiners, LRM/RLM
    "\u2060-\u2064\ufeff"          # word joiner, invisible operators, BOM
    "\u202a-\u202e\u2066-\u2069]"  # bidirectional overrides/isolates
)
_BIDI = frozenset("\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069")


def detect_hidden_characters(text: str) -> list[Finding]:
    findings: list[Finding] = []
    for match in _HIDDEN.finditer(text):
        char = match.group()
        if "\U000E0000" <= char <= "\U000E007F":
            category = "tag_characters"
        elif char in _BIDI:
            category = "bidi_control"
        else:
            category = "zero_width"
        if findings and findings[-1].end == match.start() and findings[-1].category == category:
            findings[-1] = Finding("hidden_characters", category, findings[-1].start, match.end())
        else:
            findings.append(Finding("hidden_characters", category, match.start(), match.end()))
    return findings


# --- encoded payloads -------------------------------------------------------------

_BASE64 = re.compile(r"(?<![A-Za-z0-9+/=])[A-Za-z0-9+/]{24,}={0,2}(?![A-Za-z0-9+/=])")
_HEX = re.compile(r"\b(?:[0-9a-fA-F]{2}){16,}\b")


def detect_encoded_payloads(text: str) -> list[Finding]:
    """Base64/hex blobs that decode to text containing an injection or extraction."""

    findings = []
    for pattern, decode, category in ((_BASE64, _b64, "base64"), (_HEX, _hex, "hex")):
        for match in pattern.finditer(text):
            decoded = decode(match.group())
            if decoded and (detect_injection(decoded) or detect_prompt_extraction(decoded)):
                findings.append(Finding("encoded_payloads", category, match.start(), match.end()))
    return findings


def _b64(blob: str) -> str | None:
    try:
        raw = base64.b64decode(blob + "=" * (-len(blob) % 4), validate=True)
    except (binascii.Error, ValueError):
        return None
    return _printable(raw)


def _hex(blob: str) -> str | None:
    try:
        return _printable(bytes.fromhex(blob))
    except ValueError:
        return None


def _printable(raw: bytes) -> str | None:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None
    return text if text and sum(c.isprintable() or c.isspace() for c in text) / len(text) > 0.9 else None


# --- flooding -----------------------------------------------------------------------

_REPEAT = re.compile(r"(.{2,40}?)\1{30,}", re.DOTALL)


def detect_flooding(text: str, max_chars: int) -> list[Finding]:
    findings = []
    if max_chars and len(text) > max_chars:
        findings.append(Finding("flooding", "too_long", max_chars, len(text)))
    for match in _REPEAT.finditer(text):
        findings.append(Finding("flooding", "repetition", match.start(), match.end()))
    return findings


# --- personal data ------------------------------------------------------------------

_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
# A phone number is written like one: a +country code, an (area code), or the same separator between
# its groups. A bare run of ten digits is not one: on 2026-10-11 the Fibonacci number 7778742049 in an
# answer was redacted as [REDACTED:phone] (computed numbers, ids, and timestamps look the same).
_PHONE = re.compile(
    r"(?<![\w+])(?:"
    r"\+\d{1,3}[\s.-]?(?:\(\d{3}\)|\d{3})[\s.-]?\d{3}[\s.-]?\d{4}"  # +1 415 555 0100, +14155550100
    r"|\(\d{3}\)[\s.-]?\d{3}[\s.-]?\d{4}"                         # (415) 555-0100
    r"|\d{3}([\s.-])\d{3}\1\d{4}"                                  # 415-555-0100, 415.555.0100
    r")(?![\w])"
)
_CARD = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")
_SSN = re.compile(r"(?<!\d)(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}(?!\d)")
_IBAN = re.compile(r"\b[A-Z]{2}\d{2}(?:\s?[A-Z0-9]{4}){2,7}(?:\s?[A-Z0-9]{1,4})?\b")


def detect_pii(text: str, kinds: Sequence[str] = ("email", "phone", "card", "ssn", "iban")) -> list[Finding]:
    findings: list[Finding] = []
    checks = {
        "email": (_EMAIL, None),
        "phone": (_PHONE, None),
        "card": (_CARD, lambda s: _luhn(re.sub(r"\D", "", s))),
        "ssn": (_SSN, None),
        "iban": (_IBAN, _iban_ok),
    }
    for kind in kinds:
        pattern, valid = checks[kind]
        for match in pattern.finditer(text):
            if valid is None or valid(match.group()):
                findings.append(Finding("pii", kind, match.start(), match.end()))
    return _drop_overlaps(findings)


def _luhn(digits: str) -> bool:
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for index, char in enumerate(reversed(digits)):
        value = int(char) * (2 if index % 2 else 1)
        total += value - 9 if value > 9 else value
    return total % 10 == 0


def _iban_ok(candidate: str) -> bool:
    iban = candidate.replace(" ", "")
    if not 15 <= len(iban) <= 34:
        return False
    rearranged = iban[4:] + iban[:4]
    try:
        return int("".join(str(int(c, 36)) for c in rearranged)) % 97 == 1
    except ValueError:
        return False


# --- secrets, denylist, canary --------------------------------------------------------

_SECRETS = _compile(BUILT_IN_PATTERNS)


def detect_secrets(text: str) -> list[Finding]:
    return _drop_overlaps([Finding("secrets", "credential", m.start(), m.end())
                           for p in _SECRETS for m in p.finditer(text)])


def detect_denylist(text: str, patterns: Sequence[re.Pattern[str]]) -> list[Finding]:
    return [Finding("denylist", p.pattern[:40], m.start(), m.end()) for p in patterns for m in p.finditer(text)]


CANARY_PATTERN = re.compile(r"ah-canary-[0-9a-f]{16}")


def detect_canary(text: str, canary: str | None) -> list[Finding]:
    if not canary:
        return []
    return [Finding("canary", "system_prompt_leak", m.start(), m.end())
            for m in re.finditer(re.escape(canary), text)]


def _drop_overlaps(findings: list[Finding]) -> list[Finding]:
    kept: list[Finding] = []
    for finding in sorted(findings, key=lambda f: (f.start, -(f.end - f.start))):
        if not kept or finding.start >= kept[-1].end:
            kept.append(finding)
    return kept
