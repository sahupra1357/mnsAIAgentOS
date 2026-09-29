"""Workspace file snapshots for rewind.

Pre-images are stored content-addressed in an artifact store, so identical
content is kept once and snapshots survive restarts. Paths are validated
against the workspace root: relative only, no parent traversal, no writes
through symlinks.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from agent_harness.domain import ArtifactStore


class SnapshotError(ValueError):
    """A path cannot be snapshotted or restored."""


class WorkspaceSnapshotter:
    def __init__(self, root: Path, artifacts: ArtifactStore, *, max_file_bytes: int = 5_000_000) -> None:
        self.root = root.resolve(strict=True)
        self._artifacts = artifacts
        self._max = max_file_bytes

    def _path(self, relative: str) -> Path:
        raw = Path(relative)
        if not relative.strip() or raw.is_absolute() or ".." in raw.parts:
            raise SnapshotError(f"path must stay inside the workspace: {relative!r}")
        candidate = self.root / raw
        parent = candidate.parent.resolve()
        if not parent.is_relative_to(self.root):
            raise SnapshotError(f"path escapes the workspace: {relative!r}")
        if candidate.is_symlink():
            raise SnapshotError(f"refusing to snapshot a symlink: {relative!r}")
        return parent / candidate.name

    async def capture(self, paths: Sequence[str]) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        for relative in dict.fromkeys(paths):  # de-duplicate, keep order
            try:
                path = self._path(relative)
            except SnapshotError as exc:
                entries.append({"path": relative, "error": str(exc)})
                continue
            if not path.exists():
                entries.append({"path": relative, "existed": False})
            elif not path.is_file():
                entries.append({"path": relative, "error": "not a regular file"})
            elif path.stat().st_size > self._max:
                entries.append({"path": relative, "existed": True, "skipped": "too_large"})
            else:
                ref = await self._artifacts.put(path.read_bytes(), "application/octet-stream")
                entries.append({"path": relative, "existed": True, "artifact_id": ref.id, "size": ref.size})
        return entries

    async def restore(self, entries: Sequence[Mapping[str, Any]]) -> list[str]:
        notes: list[str] = []
        for entry in entries:
            relative = str(entry.get("path", ""))
            if entry.get("error") or entry.get("skipped"):
                notes.append(f"not restored (no snapshot): {relative}")
                continue
            path = self._path(relative)
            if not entry.get("existed"):
                if path.exists():
                    path.unlink()
                    notes.append(f"deleted {relative}")
                continue
            data = await self._artifacts.get(str(entry.get("artifact_id", "")))
            if data is None:
                notes.append(f"not restored (snapshot missing): {relative}")
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=".rewind-")
            try:
                with os.fdopen(handle, "wb") as file:
                    file.write(data)
                os.replace(temporary, path)
            except BaseException:
                Path(temporary).unlink(missing_ok=True)
                raise
            notes.append(f"restored {relative}")
        return notes
