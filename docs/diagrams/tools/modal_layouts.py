"""Draw docs/diagrams/10-modal-layouts.svg: three Modal layouts compared.

    python docs/diagrams/tools/modal_layouts.py docs/diagrams/10-modal-layouts.svg

All numbers are in DATA below, each marked measured (M) or estimated (E) with its source, so the
chart can be redrawn when they change. Standard library only.

Sources (2026-10-10/11, vLLM 0.30, `.agent-harness/telemetry.jsonl` and `modal app logs`):
- Chat model calls: telemetry `model.complete` durations. H200 (`modal`, Qwen 3.6 35B-A3B): 29
  calls, median 3.1 s, p90 13.9 s, cold first call 307.5 s. Shared endpoint (`modal_shared`,
  DeepSeek V4.1 Flash): 58 calls, median 1.3 s, p90 3.2 s.
- Cost per chat call: telemetry `cost_usd`, median $0.0027 (H200, request GPU-seconds only) and
  $0.0017 (shared endpoint, tokens; 72% of input tokens cached).
- Cold starts: container boot plus vLLM import 47 s; separate embed server 71-85 s after that
  (3 starts); guard 8B 169 s inside the shared container (the one separate 8B start in the logs
  was the first, downloading 16 GB, so the separate 8B is estimated as 47 + 169 s); shared L4
  5 m 31 s and 5 m 34 s (two starts).
- Guard check: 0.4 s warm (doctor on the shared L4); it runs alongside the chat call for input.
- Prices: H200 $0.001261/s (models.toml), L4 $0.000222/s (Modal's list price, same table),
  DeepSeek V4.1 Flash $0.30 / $0.03 cached / $1.20 per million tokens; about $0.01 per run.
  GPU only: Modal also bills container CPU and memory (small next to the GPU).
- Session: wake from idle, 10 minutes of use (about 10 runs), then the 20-minute idle tail
  (SCALEDOWN_WINDOW_SECONDS) before the container stops.
"""

from __future__ import annotations

import sys
from pathlib import Path
from xml.sax.saxutils import escape

H200_PER_MIN = 0.001261 * 60
L4_PER_MIN = 0.000222 * 60
USE_MIN, IDLE_MIN = 10, 20
TOKENS_PER_SESSION = 0.10  # about 10 runs at about $0.01


def l4_session(start_min: float) -> float:
    return (start_min + USE_MIN + IDLE_MIN) * L4_PER_MIN


EMBED_COLD = (47 + 78) / 60   # M: boot + 71-85 s
GUARD_COLD = (47 + 169) / 60  # E: boot + the measured guard phase
H200_COLD = 307.5 / 60        # M: the cold first call (guard started alongside)
SHARED_L4_COLD = 332 / 60     # M: 5 m 31 s / 5 m 34 s
EMBED_LATE = (EMBED_COLD + 5 + IDLE_MIN) * L4_PER_MIN  # embed woken mid-session

DATA = [
    {
        "title": "1. Dedicated H200 + 2 L4",
        "boxes": [("Chat", "Qwen 3.6 35B-A3B", "H200, $4.54/h"), ("Guard", "Llama Guard 3 8B", "L4, $0.80/h"),
                  ("Embed", "nomic-embed", "L4, $0.80/h")],
        "cold": H200_COLD, "cold_tag": "M", "embed_extra": EMBED_COLD,
        "median": 3.1, "p90": 13.9, "call_cost": "$0.0027",
        "session": (H200_COLD + USE_MIN + IDLE_MIN) * H200_PER_MIN + l4_session(H200_COLD),
        "session_embed": EMBED_LATE,
        "facts": ["GPU billed while up, idle tail included", "20 min idle tail: H200 $1.51 + L4 $0.27",
                  "own model; guard KV cache about 3 GiB", "each app scales on its own"],
    },
    {
        "title": "2. Shared endpoint + 2 L4",
        "boxes": [("Chat", "DeepSeek V4.1 Flash", "per token, warm"), ("Guard", "Llama Guard 3 8B", "L4, $0.80/h"),
                  ("Embed", "nomic-embed", "L4, $0.80/h")],
        "cold": GUARD_COLD, "cold_tag": "E", "embed_extra": EMBED_COLD,
        "median": 1.3, "p90": 3.2, "call_cost": "$0.0017",
        "session": l4_session(GUARD_COLD) + TOKENS_PER_SESSION,
        "session_embed": EMBED_LATE,
        "facts": ["chat per token: $0.30 / $0.03 / $1.20 per M", "20 min idle tail: L4 $0.27 (chat: none)",
                  "429 when the shared pool is busy (retried)", "guard KV cache about 3 GiB"],
    },
    {
        "title": "3. Shared endpoint + 1 shared L4",
        "boxes": [("Chat", "DeepSeek V4.1 Flash", "per token, warm"),
                  ("Guard + Embed", "router + 2 vLLM", "one L4, $0.80/h")],
        "cold": SHARED_L4_COLD, "cold_tag": "M", "embed_extra": 0.0,
        "median": 1.3, "p90": 3.2, "call_cost": "$0.0017",
        "session": l4_session(SHARED_L4_COLD) + TOKENS_PER_SESSION,
        "session_embed": 0.0,
        "facts": ["one L4 for both companions", "20 min idle tail: L4 $0.27 (chat: none)",
                  "guard KV cache 2.55 GiB (2.5 checks of 8k)", "embed warm whenever the guard is"],
    },
]

COLORS = ["#4a9eed", "#f59e0b", "#22c55e"]
FILLS = ["#a5d8ff", "#fff3bf", "#b2f2bb"]
INK, MUTED, GRID = "#1e1e1e", "#6b7280", "#e5e7eb"
W = 1500


def text(x: float, y: float, s: str, size: int = 16, *, weight: str = "normal", fill: str = INK,
         anchor: str = "start") -> str:
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" fill="{fill}" '
            f'text-anchor="{anchor}">{escape(s)}</text>')


def rect(x: float, y: float, w: float, h: float, fill: str, stroke: str = "none", rx: int = 6,
         opacity: float = 1.0, dash: str = "") -> str:
    extra = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(w, 0):.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="2" opacity="{opacity}"{extra}/>')


def cards(y: float) -> list[str]:
    out: list[str] = []
    card_w, gap = 450, 25
    for i, group in enumerate(DATA):
        x = 40 + i * (card_w + gap)
        out.append(rect(x, y, card_w, 200, "#ffffff", COLORS[i], rx=12))
        out.append(text(x + 18, y + 32, group["title"], 20, weight="bold"))
        boxes = group["boxes"]
        bw = (card_w - 36 - 10 * (len(boxes) - 1)) / len(boxes)
        for j, (name, model, where) in enumerate(boxes):
            bx = x + 18 + j * (bw + 10)
            out.append(rect(bx, y + 52, bw, 128, FILLS[i], COLORS[i], rx=10))
            cx = bx + bw / 2
            out.append(text(cx, y + 86, name, 17, weight="bold", anchor="middle"))
            out.append(text(cx, y + 118, model, 13, anchor="middle"))
            out.append(text(cx, y + 146, where, 13, fill=MUTED, anchor="middle"))
    return out


def bar_chart(y: float, title: str, note: str, rows: list[tuple[float, float, str, str]], unit_max: float,
              ticks: list[float], fmt, extra_label: str) -> list[str]:
    """rows: (main value, extra value, label for the main bar, tag)."""

    out = [text(40, y, title, 22, weight="bold"), text(40, y + 24, note, 14, fill=MUTED)]
    x0, x1 = 360, W - 500  # room on the right for the value labels
    scale = (x1 - x0) / unit_max
    top = y + 44
    row_h = 46
    for t in ticks:
        gx = x0 + t * scale
        out.append(f'<line x1="{gx:.1f}" y1="{top - 4}" x2="{gx:.1f}" y2="{top + row_h * 3}" stroke="{GRID}" stroke-width="1"/>')
        out.append(text(gx, top + row_h * 3 + 18, fmt(t), 13, fill=MUTED, anchor="middle"))
    for i, (value, extra, label, tag) in enumerate(rows):
        ry = top + i * row_h
        out.append(text(x0 - 14, ry + 25, DATA[i]["title"], 16, anchor="end"))
        out.append(rect(x0, ry + 6, value * scale, 28, COLORS[i], rx=5))
        end = x0 + value * scale
        if extra:
            out.append(rect(end, ry + 6, extra * scale, 28, FILLS[i], COLORS[i], rx=5, dash="5 4"))
            end += extra * scale
        out.append(text(end + 10, ry + 26, label + (f"  ({tag})" if tag else ""), 15, weight="bold"))
    legend_y = top + row_h * 3 + 40
    out.append(rect(x0, legend_y - 12, 22, 14, "#ffffff", MUTED, rx=3, dash="5 4"))
    out.append(text(x0 + 30, legend_y, extra_label, 13, fill=MUTED))
    return out


def whisker_chart(y: float) -> list[str]:
    out = [text(40, y, "Warm response: one chat model call", 22, weight="bold"),
           text(40, y + 24, "Telemetry median (bar) and p90 (line). The guard check (0.4 s) runs alongside the "
                "chat call for new input, and once (+0.4 s) on each final answer.", 14, fill=MUTED)]
    x0, x1 = 360, W - 500  # room on the right for the value labels
    unit_max = 15.0
    scale = (x1 - x0) / unit_max
    top = y + 44
    row_h = 46
    for t in (0, 3, 6, 9, 12, 15):
        gx = x0 + t * scale
        out.append(f'<line x1="{gx:.1f}" y1="{top - 4}" x2="{gx:.1f}" y2="{top + row_h * 3}" stroke="{GRID}" stroke-width="1"/>')
        out.append(text(gx, top + row_h * 3 + 18, f"{t:g} s", 13, fill=MUTED, anchor="middle"))
    for i, group in enumerate(DATA):
        ry = top + i * row_h
        out.append(text(x0 - 14, ry + 25, group["title"], 16, anchor="end"))
        out.append(rect(x0, ry + 6, group["median"] * scale, 28, COLORS[i], rx=5))
        px = x0 + group["p90"] * scale
        out.append(f'<line x1="{x0 + group["median"] * scale:.1f}" y1="{ry + 20}" x2="{px:.1f}" y2="{ry + 20}" '
                   f'stroke="{COLORS[i]}" stroke-width="3"/>')
        out.append(f'<line x1="{px:.1f}" y1="{ry + 10}" x2="{px:.1f}" y2="{ry + 30}" stroke="{COLORS[i]}" stroke-width="3"/>')
        out.append(text(px + 10, ry + 26, f'{group["median"]:g} s median, {group["p90"]:g} s p90, '
                        f'{group["call_cost"]} per call (M)', 15, weight="bold"))
    return out


def fact_table(y: float) -> list[str]:
    out = [text(40, y, "At a glance", 22, weight="bold")]
    card_w, gap = 450, 25
    for i, group in enumerate(DATA):
        x = 40 + i * (card_w + gap)
        out.append(rect(x, y + 14, card_w, 30 + 26 * len(group["facts"]), "#f8fafc", COLORS[i], rx=10))
        for j, fact in enumerate(group["facts"]):
            out.append(text(x + 18, y + 44 + j * 26, "• " + fact, 15))
    return out


def build() -> str:
    parts: list[str] = []
    parts.append(text(40, 50, "Modal layouts: cold start, response time, and price", 30, weight="bold"))
    parts.append(text(40, 78, "Measured 2026-10-10/11 on vLLM 0.30 (M), or estimated from measured parts (E). "
                      "Sources and arithmetic: docs/diagrams/tools/modal_layouts.py", 15, fill=MUTED))
    parts += cards(100)
    parts += bar_chart(
        360, "First turn after idle: waiting for the cold start",
        "The turn waits for the slowest container it needs. Group 1 starts the H200 and the guard together "
        "(parallel = true); the embed L4 starts only when a run uses memory.",
        [(g["cold"], g["embed_extra"], f'{g["cold"]:.1f} min' + (f' + {g["embed_extra"]:.1f} on first memory use'
                                                                    if g["embed_extra"] else ", embed included"),
          g["cold_tag"]) for g in DATA],
        7.5, [0, 1, 2, 3, 4, 5, 6, 7], lambda t: f"{t:g} min",
        "dashed: extra wait for the separate embed L4, only when a run first calls memory_search / memory_write")
    parts += whisker_chart(640)
    parts += bar_chart(
        890, "Price of one session (GPU and tokens)",
        f"Wake from idle, {USE_MIN} minutes of use (about 10 runs), then the {IDLE_MIN}-minute idle tail "
        "Modal bills before the container stops. Modal's CPU and memory charges are left out.",
        [(g["session"], g["session_embed"], f'${g["session"]:.2f}' + (f' + ${g["session_embed"]:.2f} if memory is used'
                                                                      if g["session_embed"] else ", embed included"),
          "E") for g in DATA],
        3.6, [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5], lambda t: f"${t:g}",
        "dashed: the embed L4 woken mid-session (cold start, 5 minutes of use, idle tail)")
    parts += fact_table(1170)
    height = 1170 + 44 + 26 * 4 + 40
    head = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {height}" width="{W}" height="{height}" '
            f'font-family="Helvetica, Arial, sans-serif" role="img" '
            f'aria-label="Modal layouts compared: cold start, response time, and price">')
    return "\n".join([head, f'<rect width="{W}" height="{height}" fill="#ffffff"/>', *parts, "</svg>\n"])


if __name__ == "__main__":
    target = Path(sys.argv[1] if len(sys.argv) > 1 else "docs/diagrams/10-modal-layouts.svg")
    target.write_text(build(), encoding="utf-8")
    print(f"wrote {target}")
