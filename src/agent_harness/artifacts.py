"""Content-addressed storage for large tool outputs and a tool to page through them."""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
import tempfile
from pathlib import Path

from agent_harness.domain import ArtifactRef, ArtifactStore, JsonObject, ToolRisk, ToolSpec

_ARTIFACT_ID = re.compile(r"^[0-9a-f]{64}$")
MAX_READ_CHARS = 8_000


class InMemoryArtifactStore:
    def __init__(self) -> None:
        self._blobs: dict[str, tuple[bytes, str]] = {}

    async def put(self, data: bytes, media_type: str) -> ArtifactRef:
        artifact_id = hashlib.sha256(data).hexdigest()
        self._blobs.setdefault(artifact_id, (bytes(data), media_type))
        return ArtifactRef(artifact_id, len(data), media_type)

    async def get(self, artifact_id: str) -> bytes | None:
        if not _ARTIFACT_ID.fullmatch(artifact_id):
            return None
        blob = self._blobs.get(artifact_id)
        return blob[0] if blob else None


class LocalArtifactStore:
    """Store blobs under ``root/ab/<sha256>``; identical content is stored once."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root)

    def _path(self, artifact_id: str) -> Path:
        return self.root / artifact_id[:2] / artifact_id

    async def put(self, data: bytes, media_type: str) -> ArtifactRef:
        artifact_id = hashlib.sha256(data).hexdigest()
        await asyncio.to_thread(self._write, artifact_id, bytes(data))
        return ArtifactRef(artifact_id, len(data), media_type)

    def _write(self, artifact_id: str, data: bytes) -> None:
        target = self._path(artifact_id)
        if target.exists():
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        # Write then rename so readers never observe a partial blob.
        handle, temporary = tempfile.mkstemp(dir=target.parent, prefix=".tmp-")
        try:
            with os.fdopen(handle, "wb") as file:
                file.write(data)
            os.replace(temporary, target)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise

    async def get(self, artifact_id: str) -> bytes | None:
        if not _ARTIFACT_ID.fullmatch(artifact_id):
            return None
        path = self._path(artifact_id)
        if not path.is_file():
            return None
        return await asyncio.to_thread(path.read_bytes)


READ_ARTIFACT_SPEC = ToolSpec(
    name="read_artifact",
    description=(
        "Read part of a stored large tool output. Use the artifact id from an earlier "
        "tool result; page with offset (characters) and length."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "id": {"type": "string", "minLength": 64, "maxLength": 64},
            "offset": {"type": "integer", "minimum": 0},
            "length": {"type": "integer", "minimum": 1, "maximum": MAX_READ_CHARS},
        },
        "required": ["id"],
        "additionalProperties": False,
    },
    risk=ToolRisk.READ_ONLY,
)


def read_artifact_handler(store: ArtifactStore):
    async def read_artifact(arguments: JsonObject) -> JsonObject:
        data = await store.get(str(arguments["id"]))
        if data is None:
            return {"error": "unknown artifact"}
        text = data.decode("utf-8", errors="replace")
        offset = int(arguments.get("offset", 0))
        length = int(arguments.get("length", MAX_READ_CHARS))
        chunk = text[offset : offset + length]
        next_offset = offset + len(chunk)
        return {
            "text": chunk,
            "offset": offset,
            "next_offset": next_offset,
            "total_chars": len(text),
            "done": next_offset >= len(text),
        }

    return read_artifact
