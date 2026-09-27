"""Load local dataset files: JSON Lines, JSON arrays, optionally gzip-compressed.

Datasets are never downloaded by the harness; point it at files you fetched.
Parquet is not read (the core is stdlib-only); convert with e.g.
``python -c "import pandas as p; p.read_parquet('x.parquet').to_json('x.jsonl', orient='records', lines=True)"``.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any


class DatasetError(ValueError):
    """A dataset file is missing or malformed."""


def load_records(path: str | Path) -> list[dict[str, Any]]:
    file = Path(path)
    if not file.is_file():
        raise DatasetError(f"dataset file not found: {file}")
    raw = file.read_bytes()
    if file.suffix == ".gz":
        raw = gzip.decompress(raw)
    text = raw.decode("utf-8-sig")
    stripped = text.lstrip()
    if stripped.startswith("["):
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise DatasetError(f"{file}: invalid JSON: {exc}") from exc
        records = data
    else:
        records = []
        for number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise DatasetError(f"{file}:{number}: invalid JSON line: {exc}") from exc
    if not all(isinstance(record, dict) for record in records):
        raise DatasetError(f"{file}: every record must be a JSON object")
    return records


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
