"""Images inside a prompt: ``agent-harness "what is wrong here" error.png "and how do I fix it"``.

An argument, or a word inside one, that names an existing image file becomes an attachment
for vision models, and its place in the text becomes ``[image N]`` so the model knows which
part of the request each image belongs to. A whole argument that is an ``https://`` URL ending
in an image extension is attached too. Anything else stays text, so a file that does not exist
(or is not an image) is never read.
"""

from __future__ import annotations

import base64
import re
from collections.abc import Sequence
from pathlib import Path
from urllib.parse import urlsplit

IMAGE_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
               ".gif": "image/gif", ".webp": "image/webp"}
MAX_IMAGES = 16  # the API's limit per message
MAX_IMAGE_BYTES = 15 * 1024 * 1024
# Punctuation that often sticks to a path written in a sentence: "see (error.png)," or "'a.png'".
_LEADING = "\"'(<["
_TRAILING = "\"')>],.;:!?"


def image_url(item: str) -> str:
    """An https:// URL as is, or a local image file as a base64 data: URL."""

    if item.startswith("https://"):
        return item
    path = Path(item).expanduser()
    media_type = IMAGE_TYPES.get(path.suffix.lower())
    if media_type is None:
        raise ValueError(f"{item}: images must be .png, .jpg, .jpeg, .gif, or .webp files (or https:// URLs)")
    data = path.read_bytes()
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError(f"{item}: images must be under 15 MB")
    return f"data:{media_type};base64," + base64.b64encode(data).decode("ascii")


def split_images(parts: Sequence[str], *, cwd: Path | None = None) -> tuple[str, list[str]]:
    """Text and image attachments from prompt arguments; images are replaced by ``[image N]``.

    Returns the prompt text (arguments joined with spaces) and the images as URLs ready for
    ``ContentPart`` (local files as base64 ``data:`` URLs). A prompt without images comes back
    exactly as given, joined with spaces.
    """

    base = cwd or Path.cwd()
    images: list[str] = []
    texts: list[str] = []

    def attach(source: str) -> str:
        if len(images) >= MAX_IMAGES:
            raise ValueError(f"at most {MAX_IMAGES} images per message")
        images.append(image_url(source))
        return f"[image {len(images)}]"

    for part in parts:
        if _is_image_link(part):
            texts.append(attach(part))
            continue
        whole = _local_image(part.strip().strip(_LEADING + _TRAILING), base)
        if whole is not None:  # the whole argument is one path (it may contain spaces)
            texts.append(attach(str(whole)))
            continue
        pieces = re.split(r"(\s+)", part)
        for index, piece in enumerate(pieces):
            if not piece or piece.isspace():
                continue
            core = piece.lstrip(_LEADING)
            lead = piece[: len(piece) - len(core)]
            stripped = core.rstrip(_TRAILING)
            trail = core[len(stripped):]
            path = _local_image(stripped, base)
            if path is not None:
                pieces[index] = lead + attach(str(path)) + trail
        texts.append("".join(pieces))
    return " ".join(text for text in texts if text), images


def _local_image(word: str, base: Path) -> Path | None:
    if not word or Path(word).suffix.lower() not in IMAGE_TYPES:
        return None
    path = Path(word).expanduser()
    if not path.is_absolute():
        path = base / path
    return path if path.is_file() else None


def _is_image_link(part: str) -> bool:
    if not part.startswith("https://") or any(ch.isspace() for ch in part):
        return False
    return Path(urlsplit(part).path).suffix.lower() in IMAGE_TYPES
