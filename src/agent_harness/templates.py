"""Prompt templates: reusable prompts invoked as ``/<name> arguments``.

A template is ``<templates dir>/<name>.md``, optionally with frontmatter
(``description``). Placeholders:

* ``$ARGUMENTS`` - everything after the template name
* ``$1`` .. ``$9`` - whitespace-separated positional arguments
* ``${key}`` - named values (``--arg key=value`` on the CLI, ``template_args``
  in the API); a missing named value is an error

Expansion happens at the CLI/API boundary; the run records the template name
and content hash.
"""

from __future__ import annotations

import hashlib
import re
import shlex
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from .instructions import frontmatter, strip_frontmatter

_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_NAMED = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_POSITIONAL = re.compile(r"\$([1-9])")


class TemplateError(ValueError):
    """A template is missing or cannot be expanded."""


@dataclass(frozen=True, slots=True)
class Template:
    name: str
    description: str
    body: str
    path: Path

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.body.encode("utf-8")).hexdigest()

    def expand(self, arguments: str = "", named: Mapping[str, str] | None = None) -> str:
        values = dict(named or {})
        try:
            positional = shlex.split(arguments)
        except ValueError:
            positional = arguments.split()

        def named_value(match: re.Match[str]) -> str:
            key = match.group(1)
            if key not in values:
                raise TemplateError(f"template {self.name!r} needs a value for {key!r}")
            return values[key]

        text = _NAMED.sub(named_value, self.body)
        text = _POSITIONAL.sub(
            lambda m: positional[int(m.group(1)) - 1] if int(m.group(1)) <= len(positional) else "",
            text,
        )
        return text.replace("$ARGUMENTS", arguments).strip()


def discover_templates(directories: Iterable[Path]) -> dict[str, Template]:
    templates: dict[str, Template] = {}
    for directory in directories:
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.md")):
            name = path.stem
            if not _NAME.fullmatch(name) or name in templates:
                continue
            raw = path.read_text(encoding="utf-8")
            body = strip_frontmatter(raw).strip()
            if body:
                templates[name] = Template(name, frontmatter(raw).get("description", ""), body, path)
    return templates


def expand_prompt(
    prompt: str, templates: Mapping[str, Template], named: Mapping[str, str] | None = None
) -> tuple[str, Template | None]:
    """Expand ``/name args`` when ``name`` is a known template; else return as is."""

    if not prompt.startswith("/"):
        return prompt, None
    head, _, rest = prompt[1:].partition(" ")
    template = templates.get(head)
    if template is None:
        return prompt, None
    return template.expand(rest.strip(), named), template
