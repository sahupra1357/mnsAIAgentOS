"""Workspace-confined file inspection, search, and atomic patch application."""

from __future__ import annotations

import fnmatch
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class WorkspaceViolation(ValueError):
    """A requested file operation escaped or violated the workspace policy."""


_SKIP_DIRECTORIES = {".git", ".venv", "node_modules", "__pycache__"}
_SENSITIVE_NAMES = {".env", ".npmrc", ".pypirc", "id_rsa", "id_ed25519"}
_SENSITIVE_SUFFIXES = {".key", ".pem", ".p12", ".pfx"}
_DOCUMENT_SUFFIXES = {
    ".md",
    ".mdx",
    ".rst",
    ".txt",
    ".adoc",
    ".toml",
    ".yaml",
    ".yml",
}


@dataclass(slots=True)
class WorkspaceBoundary:
    root: Path
    max_file_bytes: int = 1_000_000
    max_search_results: int = 100

    def __post_init__(self) -> None:
        self.root = self.root.resolve(strict=True)
        if not self.root.is_dir():
            raise WorkspaceViolation("workspace root must be a directory")
        if self.max_file_bytes < 1 or self.max_search_results < 1:
            raise ValueError("workspace limits must be positive")

    def read_file(
        self, relative_path: str, *, start_line: int = 1, max_lines: int = 400
    ) -> dict[str, Any]:
        path = self._resolve(relative_path, must_exist=True)
        self._assert_readable_file(path)
        if start_line < 1 or not 1 <= max_lines <= 2_000:
            raise WorkspaceViolation("line bounds are invalid")
        text = self._read_text(path)
        lines = text.splitlines()
        selected = lines[start_line - 1 : start_line - 1 + max_lines]
        return {
            "path": path.relative_to(self.root).as_posix(),
            "start_line": start_line,
            "end_line": start_line + len(selected) - 1 if selected else start_line - 1,
            "content": "\n".join(selected),
            "truncated": start_line - 1 + len(selected) < len(lines),
        }

    def search(
        self,
        query: str,
        *,
        glob: str = "*",
        documentation_only: bool = False,
        case_sensitive: bool = False,
    ) -> dict[str, Any]:
        if not query or len(query) > 1_000:
            raise WorkspaceViolation("search query must contain 1-1000 characters")
        needle = query if case_sensitive else query.casefold()
        matches: list[dict[str, Any]] = []
        scanned = 0
        for path in self._iter_files():
            relative = path.relative_to(self.root).as_posix()
            if not fnmatch.fnmatch(relative, glob) and not fnmatch.fnmatch(path.name, glob):
                continue
            if documentation_only and path.suffix.casefold() not in _DOCUMENT_SUFFIXES:
                continue
            try:
                text = self._read_text(path)
            except (UnicodeDecodeError, WorkspaceViolation):
                continue
            scanned += 1
            for line_number, line in enumerate(text.splitlines(), 1):
                haystack = line if case_sensitive else line.casefold()
                column = haystack.find(needle)
                if column < 0:
                    continue
                matches.append(
                    {
                        "path": relative,
                        "line": line_number,
                        "column": column + 1,
                        "text": line[:500],
                    }
                )
                if len(matches) >= self.max_search_results:
                    return {"matches": matches, "truncated": True, "files_scanned": scanned}
        return {"matches": matches, "truncated": False, "files_scanned": scanned}

    def apply_patch(
        self,
        relative_path: str,
        *,
        old_text: str,
        new_text: str,
        create: bool = False,
    ) -> dict[str, Any]:
        path = self._resolve(relative_path, must_exist=not create)
        self._assert_writable_target(path)
        if len(new_text.encode("utf-8")) > self.max_file_bytes:
            raise WorkspaceViolation("resulting file would exceed the size limit")

        if create:
            if path.exists():
                raise WorkspaceViolation("create requested but the file already exists")
            if old_text:
                raise WorkspaceViolation("old_text must be empty when creating a file")
            updated = new_text
            mode = 0o644
        else:
            self._assert_readable_file(path)
            current = self._read_text(path)
            if not old_text:
                raise WorkspaceViolation("old_text must be non-empty when updating a file")
            occurrences = current.count(old_text)
            if occurrences != 1:
                raise WorkspaceViolation(
                    f"old_text must match exactly once; found {occurrences} matches"
                )
            updated = current.replace(old_text, new_text, 1)
            if len(updated.encode("utf-8")) > self.max_file_bytes:
                raise WorkspaceViolation("resulting file would exceed the size limit")
            mode = path.stat().st_mode & 0o777

        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_name = ""
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_name = temporary.name
                temporary.write(updated)
                temporary.flush()
                os.fsync(temporary.fileno())
            os.chmod(temporary_name, mode)
            os.replace(temporary_name, path)
        finally:
            if temporary_name and os.path.exists(temporary_name):
                os.unlink(temporary_name)
        return {
            "path": path.relative_to(self.root).as_posix(),
            "created": create,
            "bytes": len(updated.encode("utf-8")),
        }

    def resolve_directory(self, relative_path: str) -> Path:
        path = self._resolve(relative_path or ".", must_exist=True)
        if not path.is_dir():
            raise WorkspaceViolation("working directory is not a directory")
        return path

    def validate_relative_file(self, relative_path: str) -> str:
        path = self._resolve(relative_path, must_exist=True)
        self._assert_readable_file(path)
        return path.relative_to(self.root).as_posix()

    def _resolve(self, relative_path: str, *, must_exist: bool) -> Path:
        if not isinstance(relative_path, str) or not relative_path.strip():
            raise WorkspaceViolation("path must be a non-empty relative string")
        raw = Path(relative_path)
        if raw.is_absolute() or ".." in raw.parts:
            raise WorkspaceViolation("path must stay inside the workspace")
        candidate = self.root / raw
        if must_exist:
            resolved = candidate.resolve(strict=True)
        else:
            parent = candidate.parent.resolve(strict=True)
            resolved = parent / candidate.name
        if not resolved.is_relative_to(self.root):
            raise WorkspaceViolation("path escapes the workspace")
        return resolved

    def _assert_readable_file(self, path: Path) -> None:
        if not path.is_file() or path.is_symlink():
            raise WorkspaceViolation("path must be a regular non-symlink file")
        relative = path.relative_to(self.root)
        if any(part in _SKIP_DIRECTORIES for part in relative.parts[:-1]):
            raise WorkspaceViolation("reads from protected directories are forbidden")
        if path.name in _SENSITIVE_NAMES or path.suffix.casefold() in _SENSITIVE_SUFFIXES:
            raise WorkspaceViolation("reads from credential-like files are forbidden")
        if path.stat().st_size > self.max_file_bytes:
            raise WorkspaceViolation("file exceeds the configured size limit")

    def _assert_writable_target(self, path: Path) -> None:
        relative = path.relative_to(self.root)
        if any(part in _SKIP_DIRECTORIES for part in relative.parts[:-1]):
            raise WorkspaceViolation("writes to protected directories are forbidden")
        if path.name in _SENSITIVE_NAMES or path.suffix.casefold() in _SENSITIVE_SUFFIXES:
            raise WorkspaceViolation("writes to credential-like files are forbidden")
        if path.exists() and path.is_symlink():
            raise WorkspaceViolation("writes through symlinks are forbidden")

    def _read_text(self, path: Path) -> str:
        data = path.read_bytes()
        if len(data) > self.max_file_bytes:
            raise WorkspaceViolation("file exceeds the configured size limit")
        if b"\x00" in data:
            raise UnicodeDecodeError("utf-8", data, 0, 1, "binary file")
        return data.decode("utf-8")

    def _iter_files(self):
        for directory, names, files in os.walk(self.root, followlinks=False):
            names[:] = [name for name in names if name not in _SKIP_DIRECTORIES]
            base = Path(directory)
            for name in files:
                path = base / name
                if path.is_symlink() or name in _SENSITIVE_NAMES:
                    continue
                if path.suffix.casefold() in _SENSITIVE_SUFFIXES:
                    continue
                yield path
