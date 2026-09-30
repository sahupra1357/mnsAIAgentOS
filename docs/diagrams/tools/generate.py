"""Generate static SVG, animated SVG, .excalidraw, and tour data for the harness diagrams.

Usage (from the repository root):
    python docs/diagrams/tools/generate.py docs/diagrams build/tours.json docs/diagrams/tools/page_template.html

The optional third argument also writes the interactive walkthrough page
(build/tours.html) from the template. Diagrams, tours, and captions are all
defined in this file; edit a diagram function and regenerate.
"""

from __future__ import annotations

import json
import math
import random
import sys
from html import escape
from pathlib import Path

COLORS = {
    "blue": ("#a5d8ff", "#4a9eed"),
    "purple": ("#d0bfff", "#8b5cf6"),
    "green": ("#b2f2bb", "#22c55e"),
    "teal": ("#c3fae8", "#22c55e"),
    "orange": ("#ffd8a8", "#f59e0b"),
    "yellow": ("#fff3bf", "#f59e0b"),
    "red": ("#ffc9c9", "#ef4444"),
    "pink": ("#eebefa", "#ec4899"),
}
ZONES = {
    "blue": ("#dbe4ff", "#4a9eed", "#2563eb"),
    "purple": ("#e5dbff", "#8b5cf6", "#6d28d9"),
    "green": ("#d3f9d8", "#22c55e", "#15803d"),
    "yellow": ("#fff3bf", "#f59e0b", "#b45309"),
    "pink": ("#eebefa", "#ec4899", "#be185d"),
}
ACCENT = "#1e3a8a"  # navy tour highlight outline, same as the dot
DOT = "#1e3a8a"  # navy dot that travels along the arrows
CAPTION_BAND = 110


class Diagram:
    def __init__(self, name: str, title: str, width: int, height: int) -> None:
        self.name, self.title, self.width, self.height = name, title, width, height
        self.items: list[dict] = []
        self.nodes: dict[str, dict] = {}
        self.edges: dict[str, dict] = {}
        self.steps: list[dict] = []
        self._auto = 0

    def _id(self, given):
        if given:
            return given
        self._auto += 1
        return f"_{self._auto}"

    def zone(self, x, y, w, h, title, color):
        fill, stroke, text = ZONES[color]
        self.items.append(dict(kind="zone", x=x, y=y, w=w, h=h, fill=fill, stroke=stroke))
        self.text(x + 15, y + 10, title, 22, text)

    def box(self, x, y, w, h, label, color, size=18, shape="rectangle", id=None):
        fill, stroke = COLORS[color]
        item = dict(kind=shape, id=self._id(id), x=x, y=y, w=w, h=h, fill=fill,
                    stroke=stroke, label=label, size=size)
        self.items.append(item)
        self.nodes[item["id"]] = item

    def diamond(self, x, y, w, h, label, color="yellow", size=18, id=None):
        self.box(x, y, w, h, label, color, size, shape="diamond", id=id)

    def text(self, x, y, text, size=18, color="#1e1e1e"):
        self.items.append(dict(kind="text", x=x, y=y, text=text, size=size, color=color))

    def arrow(self, points, label=None, color="#1e1e1e", dashed=False, both=False, width=2, id=None):
        item = dict(kind="arrow", id=self._id(id), points=points, label=label, color=color,
                    dashed=dashed, both=both, width=width)
        self.items.append(item)
        self.edges[item["id"]] = item

    def step(self, node, via, title, detail, events=()):
        """``via``: None (straight line), "jump" (teleport), or edge ids ("-id" = reversed)."""
        self.steps.append(dict(node=node, via=via, title=title, detail=detail, events=list(events)))

    # ---- tour geometry --------------------------------------------------------
    def center(self, node_id):
        """Where the token rests on a node: the middle of its top edge, clear of the label."""
        n = self.nodes[node_id]
        return (n["x"] + n["w"] / 2, n["y"])

    def routes(self):
        """Per step: the arrows ("legs") the token travels along to reach it.

        The token only ever moves on arrows, never across a box: each leg is
        one arrow's polyline (reversed for "-id"). An empty list means the
        step has no arrow to travel (the first step or a jump).
        """
        out = []
        for index, step in enumerate(self.steps):
            vias = step["via"]
            if index == 0 or vias in (None, "jump"):
                out.append([])
                continue
            if isinstance(vias, str):
                vias = [vias]
            legs = []
            for via in vias:
                edge = self.edges[via.lstrip("-")]["points"]
                points = list(reversed(edge)) if via.startswith("-") else list(edge)
                legs.append([[round(x, 1), round(y, 1)] for x, y in points])
            out.append(legs)
        return out

    @staticmethod
    def leg_seconds(leg, speed=240.0):
        length = sum(math.dist(a, b) for a, b in zip(leg, leg[1:]))
        return min(1.8, max(0.5, length / speed))

    def timeline(self, dwell=2.6, gap=0.3):
        """Per step: (travel_start, arrive, leave, [(leg_start, leg_end), ...]) and the total.

        Between two legs of one step the token is hidden for ``gap`` seconds
        while the path passes through an intermediate box.
        """
        t = 0.0
        times = []
        for legs in self.routes():
            start = t
            windows = []
            for number, leg in enumerate(legs):
                if number:
                    t += gap
                windows.append((t, t + self.leg_seconds(leg)))
                t = windows[-1][1]
            arrive = t
            t += dwell
            times.append((start, arrive, t, windows))
        return times, t + 1.0

    # ---- excalidraw ---------------------------------------------------------
    def excalidraw(self) -> dict:
        rng = random.Random(self.name)
        elements: list[dict] = []
        counter = 0

        def base(kind, x, y, w, h, **extra):
            nonlocal counter
            counter += 1
            element = {
                "id": f"{self.name}-{counter}", "type": kind, "x": x, "y": y,
                "width": w, "height": h, "angle": 0, "strokeColor": "#1e1e1e",
                "backgroundColor": "transparent", "fillStyle": "solid", "strokeWidth": 2,
                "strokeStyle": "solid", "roughness": 1, "opacity": 100, "groupIds": [],
                "frameId": None, "roundness": None, "seed": rng.randint(1, 2**31),
                "version": 1, "versionNonce": rng.randint(1, 2**31), "isDeleted": False,
                "boundElements": [], "updated": 1, "link": None, "locked": False,
            }
            element.update(extra)
            elements.append(element)
            return element

        def text_el(x, y, text, size, color, container=None, align="left"):
            lines = text.split("\n")
            w = max(len(line) for line in lines) * size * 0.55
            h = len(lines) * size * 1.25
            if container is not None:
                cx = container["x"] + container["width"] / 2
                cy = container["y"] + container["height"] / 2
                x, y = cx - w / 2, cy - h / 2
            return base("text", x, y, w, h, strokeColor=color, text=text, originalText=text,
                        fontSize=size, fontFamily=1, textAlign=align,
                        verticalAlign="middle" if container else "top",
                        containerId=container["id"] if container else None,
                        lineHeight=1.25, autoResize=True)

        for item in self.items:
            kind = item["kind"]
            if kind == "zone":
                base("rectangle", item["x"], item["y"], item["w"], item["h"],
                     backgroundColor=item["fill"], strokeColor=item["stroke"], strokeWidth=1,
                     opacity=30, roundness={"type": 3})
            elif kind in ("rectangle", "diamond"):
                shape = base(kind, item["x"], item["y"], item["w"], item["h"],
                             backgroundColor=item["fill"], strokeColor=item["stroke"],
                             roundness={"type": 3} if kind == "rectangle" else {"type": 2})
                label = text_el(0, 0, item["label"], item["size"], "#1e1e1e", shape, "center")
                shape["boundElements"] = [{"type": "text", "id": label["id"]}]
            elif kind == "text":
                text_el(item["x"], item["y"], item["text"], item["size"], item["color"])
            elif kind == "arrow":
                (x0, y0) = item["points"][0]
                rel = [[px - x0, py - y0] for px, py in item["points"]]
                xs = [p[0] for p in rel]
                ys = [p[1] for p in rel]
                arrow = base("arrow", x0, y0, max(xs) - min(xs), max(ys) - min(ys),
                             strokeColor=item["color"], strokeWidth=item["width"],
                             strokeStyle="dashed" if item["dashed"] else "solid",
                             roundness=None, points=rel, lastCommittedPoint=None,
                             startBinding=None, endBinding=None,
                             startArrowhead="arrow" if item["both"] else None,
                             endArrowhead="arrow", elbowed=False)
                if item["label"]:
                    mx, my = _midpoint(item["points"])
                    label = text_el(mx, my, item["label"], 16, "#1e1e1e", None, "center")
                    label["containerId"] = arrow["id"]
                    label["verticalAlign"] = "middle"
                    label["x"] -= label["width"] / 2
                    label["y"] -= label["height"] / 2
                    arrow["boundElements"] = [{"type": "text", "id": label["id"]}]
        return {"type": "excalidraw", "version": 2, "source": "https://excalidraw.com",
                "elements": elements,
                "appState": {"viewBackgroundColor": "#ffffff", "gridSize": None}, "files": {}}

    # ---- svg ------------------------------------------------------------------
    def svg_body(self) -> list[str]:
        out: list[str] = []
        labels: list[str] = []
        for item in self.items:
            kind = item["kind"]
            if kind == "zone":
                out.append(f'<rect x="{item["x"]}" y="{item["y"]}" width="{item["w"]}" '
                           f'height="{item["h"]}" rx="16" fill="{item["fill"]}" fill-opacity="0.3" '
                           f'stroke="{item["stroke"]}" stroke-width="1"/>')
            elif kind in ("rectangle", "diamond"):
                out.append(f'<g class="node" id="n-{item["id"]}">'
                           + _shape(item, item["fill"], item["stroke"], 2)
                           + _svg_lines(item["x"] + item["w"] / 2, item["y"] + item["h"] / 2,
                                        item["label"], item["size"])
                           + "</g>")
            elif kind == "text":
                lines = item["text"].split("\n")
                spans = "".join(
                    f'<tspan x="{item["x"]}" dy="{0 if i == 0 else item["size"] * 1.25}">'
                    f"{escape(line)}</tspan>" for i, line in enumerate(lines))
                out.append(f'<text x="{item["x"]}" y="{item["y"] + item["size"]}" '
                           f'font-size="{item["size"]}" fill="{item["color"]}" '
                           f'font-weight="{"bold" if item["size"] >= 22 else "normal"}">{spans}</text>')
            elif kind == "arrow":
                pts = item["points"]
                dash = ' stroke-dasharray="8 6"' if item["dashed"] else ""
                parts = [f'<polyline points="{" ".join(f"{px},{py}" for px, py in pts)}" '
                         f'fill="none" stroke="{item["color"]}" stroke-width="{item["width"]}"{dash}/>',
                         _head(pts[-2], pts[-1], item["color"])]
                if item["both"]:
                    parts.append(_head(pts[1], pts[0], item["color"]))
                out.append(f'<g class="edge" id="e-{item["id"]}">' + "".join(parts) + "</g>")
                if item["label"]:
                    mx, my = _midpoint(pts)
                    lines = item["label"].split("\n")
                    w = max(len(line) for line in lines) * 16 * 0.56 + 12
                    h = len(lines) * 20 + 6
                    labels.append(f'<rect x="{mx - w / 2}" y="{my - h / 2}" width="{w}" height="{h}" '
                                  'rx="4" fill="#ffffff" fill-opacity="0.92"/>')
                    labels.append(_svg_lines(mx, my, item["label"], 16))
        return out + labels

    def svg(self) -> str:
        return "\n".join([
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.width} {self.height}" '
            f'width="{self.width}" height="{self.height}" font-family="Helvetica, Arial, sans-serif">',
            f'<rect width="{self.width}" height="{self.height}" fill="#ffffff"/>',
            *self.svg_body(), "</svg>"])

    def interactive_svg(self) -> str:
        """Inline SVG for the web page: no fixed size, plus a token the page moves."""
        return "\n".join([
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.width} {self.height}" '
            f'font-family="Helvetica, Arial, sans-serif" role="img" aria-label="{escape(self.title)}">',
            f'<rect width="{self.width}" height="{self.height}" fill="#ffffff"/>',
            *self.svg_body(),
            f'<g class="token" opacity="0"><circle r="11" fill="{DOT}" fill-opacity="0.2"/>'
            f'<circle r="6" fill="{DOT}" stroke="#ffffff" stroke-width="2"/></g>',
            "</svg>"])

    def animated_svg(self) -> str:
        """Self-running animation (SMIL): works in browsers and on GitHub, no script."""
        times, total = self.timeline()
        routes = self.routes()
        height = self.height + CAPTION_BAND
        out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.width} {height}" '
               f'width="{self.width}" height="{height}" font-family="Helvetica, Arial, sans-serif">',
               f'<rect width="{self.width}" height="{height}" fill="#ffffff"/>',
               *self.svg_body()]
        dur = f'dur="{total:.2f}s" repeatCount="indefinite"'

        def window(start, end):
            """Discrete opacity animation: visible only within [start, end)."""
            values, keys = [], []
            if start > 0:
                values.append("0"), keys.append(0.0)
            values.append("1"), keys.append(start / total)
            values.append("0"), keys.append(end / total)
            return (f'<animate attributeName="opacity" {dur} calcMode="discrete" '
                    f'values="{";".join(values)}" keyTimes="{";".join(f"{k:.4f}" for k in keys)}"/>')

        # The box is outlined only while the tour rests on it; it goes dark as
        # the dot leaves, and the next box lights up when the dot arrives.
        for step, (start, arrive, leave, _) in zip(self.steps, times):
            node = self.nodes[step["node"]]
            out.append(f'<g opacity="0">{_shape(node, "none", ACCENT, 6, pad=6)}'
                       f'{window(arrive, leave)}</g>')

        # One dot per arrow travelled. It is visible only while moving along
        # that arrow, so it never sits on or crosses a box.
        for legs, (_, _, _, windows) in zip(routes, times):
            for leg, (leg_start, leg_end) in zip(legs, windows):
                path = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in leg)
                key_times = f"0;{leg_start / total:.4f};{leg_end / total:.4f};1"
                out.append(
                    f'<g opacity="0">{window(leg_start, leg_end)}'
                    f'<g><circle r="11" fill="{DOT}" fill-opacity="0.2"/>'
                    f'<circle r="6" fill="{DOT}" stroke="#ffffff" stroke-width="2"/>'
                    f'<animateMotion {dur} calcMode="linear" path="{path}" '
                    f'keyTimes="{key_times}" keyPoints="0;0;1;1"/></g></g>')

        # Caption band: switches when the dot arrives at the next box.
        band_y = self.height
        out.append(f'<rect x="0" y="{band_y}" width="{self.width}" height="{CAPTION_BAND}" fill="#1e1b2e"/>')
        count = len(self.steps)
        arrivals = [arrive for _, arrive, _, _ in times] + [total]
        for number, step in enumerate(self.steps, 1):
            head = f"Step {number} of {count} · {step['title']}"
            out.append(
                f'<g opacity="0">{window(arrivals[number - 1], arrivals[number])}'
                f'<text x="40" y="{band_y + 42}" font-size="24" font-weight="bold" fill="#ffffff">'
                f'{escape(head)}</text>'
                f'<text x="40" y="{band_y + 80}" font-size="19" fill="#d6d3e4">{escape(step["detail"])}</text></g>')
        out.append("</svg>")
        return "\n".join(out)

    def tour(self) -> dict:
        return {
            "id": self.name, "title": self.title,
            "svg": self.interactive_svg(),
            "steps": [dict(step, route=route, dock=[round(v, 1) for v in self.center(step["node"])])
                      for step, route in zip(self.steps, self.routes())],
        }


def _shape(item, fill, stroke, width, pad=0):
    x, y, w, h = item["x"] - pad, item["y"] - pad, item["w"] + 2 * pad, item["h"] + 2 * pad
    if item["kind"] == "diamond":
        pts = f"{x + w / 2},{y} {x + w},{y + h / 2} {x + w / 2},{y + h} {x},{y + h / 2}"
        return f'<polygon points="{pts}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>'
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{width}"/>')


def _midpoint(points):
    lengths = [math.dist(a, b) for a, b in zip(points, points[1:])]
    half = sum(lengths) / 2
    for (a, b), length in zip(zip(points, points[1:]), lengths):
        if half <= length and length:
            t = half / length
            return a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t
        half -= length
    return points[-1]


def _head(a, b, color):
    angle = math.atan2(b[1] - a[1], b[0] - a[0])
    size = 12
    left = (b[0] - size * math.cos(angle - 0.45), b[1] - size * math.sin(angle - 0.45))
    right = (b[0] - size * math.cos(angle + 0.45), b[1] - size * math.sin(angle + 0.45))
    return (f'<polygon points="{b[0]},{b[1]} {left[0]:.1f},{left[1]:.1f} '
            f'{right[0]:.1f},{right[1]:.1f}" fill="{color}"/>')


def _svg_lines(cx, cy, text, size):
    lines = text.split("\n")
    step = size * 1.25
    top = cy - step * (len(lines) - 1) / 2 + size * 0.35
    spans = "".join(f'<tspan x="{cx}" y="{top + i * step:.1f}">{escape(line)}</tspan>'
                    for i, line in enumerate(lines))
    return f'<text font-size="{size}" text-anchor="middle" fill="#1e1e1e">{spans}</text>'


# ---------------------------------------------------------------------------
def end_to_end() -> Diagram:
    d = Diagram("01-end-to-end", "End-to-end flow", 1610, 1180)
    d.text(545, 8, "agent_harness: end-to-end flow", 32)
    d.zone(10, 55, 310, 1110, "1. Entry and setup", "blue")
    d.box(25, 105, 280, 105, "CLI\nrun, continue, approve,\nresume, rewind, cancel", "blue", id="cli")
    d.box(25, 230, 280, 80, "HTTP API: keys, roles,\ntenants, quotas, SSE", "blue", id="api")
    d.box(25, 330, 280, 80, "Queue workers (leases;\nPostgres SKIP LOCKED)", "blue", id="workers")
    d.arrow([(165, 410), (165, 470)], id="to_comp")
    d.box(25, 470, 280, 250, "build_application\n\nlayered settings, profiles\ntool sources + MCP\n"
          "tool catalog sync\nplugins, hooks, subagents\nAGENTS.md, skills, planning", "purple", id="comp")
    d.arrow([(165, 720), (165, 770)], id="to_create")
    d.box(25, 770, 280, 140, "quota check, create_run\nsystem prompt + hashes\ntenant, planning\nRUN_CREATED (queued)",
          "blue", id="create")
    d.box(25, 1000, 280, 120, "Follow-ups re-enter:\ncontinue, approve,\nresume, rewind", "yellow", id="follow")

    d.zone(340, 55, 660, 1110, "2. AgentOrchestrator loop (durable)", "purple")
    d.arrow([(305, 840), (350, 840), (350, 145), (370, 145)], "run()", "#8b5cf6", id="run")
    d.box(370, 105, 280, 85, "Step guard: budgets, steer,\nplan step / delegation", "purple", id="guard")
    d.arrow([(510, 190), (510, 220)], id="g_pm")
    d.box(370, 220, 280, 60, "pre_model hook", "purple", id="pre_model")
    d.arrow([(510, 280), (510, 310)], id="pm_ctx")
    d.box(370, 310, 280, 110, "ContextBuilder\nbounded view + summary\n(pre_compact hook)", "purple", id="ctx")
    d.arrow([(510, 420), (510, 450)], id="ctx_model")
    d.box(370, 450, 280, 80, "Model call\nModelRequest", "purple", id="model")
    d.arrow([(510, 530), (510, 565)], id="model_dec")
    d.diamond(400, 565, 220, 140, "tool\ncalls?", id="dec")
    d.arrow([(510, 705), (510, 760)], "no", id="no")
    d.box(370, 760, 280, 80, "stop hook\n-> COMPLETED", "green", id="done")
    d.arrow([(620, 635), (690, 635)], "yes", id="yes")
    d.box(690, 580, 280, 100, "pre_tool hook\npermission rules\n+ permission mode", "purple", id="pre_tool")
    d.arrow([(830, 580), (830, 540)], id="pt_appr")
    d.box(690, 445, 280, 95, "Needs approval?\nyes -> AWAITING_INPUT", "orange", id="approval")
    d.arrow([(830, 445), (830, 410)], "no", id="appr_rt")
    d.box(690, 275, 280, 135, "ToolRuntime\nvalidation, timeout,\nidempotency, snapshot\n(read-only in parallel)",
          "purple", id="runtime")
    d.arrow([(830, 275), (830, 240)], id="rt_post")
    d.box(690, 105, 280, 135, "post_tool hook\nuntrusted output wrapped\nlarge output -> artifact", "purple",
          id="post_tool")
    d.arrow([(690, 148), (650, 148)], color="#8b5cf6", width=3, id="loop")
    d.box(690, 720, 280, 130, "Orchestrator tools:\nagent, todo_write, exit_plan,\nsearch/load_tools,\ncreate/update/revise_plan",
          "yellow", id="builtins", size=16)
    d.box(690, 900, 280, 90, "AWAITING_INPUT\napproval / plan review", "orange", id="awaiting")
    d.arrow([(970, 492), (990, 492), (990, 945), (970, 945)], color="#f59e0b", dashed=True, id="pause")
    d.box(370, 900, 280, 90, "FAILED / CANCELLED /\nBUDGET_EXHAUSTED", "red", id="failed")
    d.text(400, 1000, "any step can end here", 16, "#b91c1c")
    d.arrow([(370, 960), (305, 1050)], color="#f59e0b", dashed=True, id="refollow")

    d.zone(1020, 55, 300, 395, "4. Durable state", "green")
    d.box(1040, 100, 260, 120, "Run store: SQLite / Postgres\nappend-only events (masked)\nqueue, idempotency", "teal", id="store", size=16)
    d.arrow([(970, 165), (1040, 160)], color="#22c55e", dashed=True, id="checkpoint")
    d.box(1040, 240, 260, 80, "Artifacts\nsnapshots, big outputs", "teal", id="artifacts")
    d.arrow([(970, 340), (1040, 280)], color="#22c55e", dashed=True, id="snap")
    d.box(1040, 340, 260, 80, "Tool catalog (persistent)\n+ memory (SQLite FTS)", "teal", id="memory", size=16)
    d.zone(1340, 55, 255, 410, "5. Observability, controls", "pink")
    d.box(1355, 100, 225, 110, "Telemetry\nJSON logs, spans,\nmetrics, cache hits", "pink", id="telemetry")
    d.arrow([(1300, 160), (1355, 155)], color="#ec4899", dashed=True, id="tel")
    d.box(1355, 230, 225, 95, "Version record\nprompt, tools,\nprofile hashes", "pink", id="versions")
    d.box(1355, 345, 225, 95, "Redaction, egress,\nsecrets, gc; evals", "pink", id="evals")

    d.zone(1020, 465, 300, 700, "3. Model gateway", "yellow")
    d.arrow([(650, 490), (670, 490), (670, 550), (1040, 550)], "request / response",
            "#f59e0b", both=True, width=3, id="req")
    d.box(1040, 515, 260, 70, "Route -> profile\n(fast / strong)", "orange", id="route")
    d.arrow([(1170, 585), (1170, 615)], id="route_ad")
    d.box(1040, 615, 260, 215, "Adapters\nOllama\nllama.cpp\nModal (vLLM / SGLang)\n"
          "OpenAI-compatible\nLiteLLM (optional)", "orange", id="adapters")
    d.arrow([(1170, 830), (1170, 860)], id="ad_retry")
    d.box(1040, 860, 260, 120, "One retry budget\nfallback chain\ncost + cached tokens", "orange", id="retry")
    d.arrow([(1170, 980), (1170, 1010)], id="retry_stream")
    d.box(1040, 1010, 260, 100, "Streaming (SSE)\nonly final response\nis checkpointed", "orange", id="stream")
    d.zone(1340, 480, 255, 685, "6. Model servers", "blue")
    d.box(1355, 540, 225, 75, "Ollama\n:11434/v1", "blue", id="ollama")
    d.box(1355, 640, 225, 75, "llama-server\n:8080/v1", "blue", id="llamacpp")
    d.box(1355, 740, 225, 100, "Modal\nvLLM / SGLang\nbearer + proxy keys", "blue", id="modal")
    d.box(1355, 865, 225, 80, "Other providers\nvia LiteLLM", "blue", id="litellm")
    d.box(1355, 965, 225, 80, "MCP servers\nstdio / HTTP (tools)", "blue", id="mcpservers")
    for name, (y0, y1) in zip(["s_ollama", "s_llama", "s_modal", "s_litellm"],
                              [(690, 578), (715, 678), (745, 790), (775, 905)]):
        d.arrow([(1300, y0), (1355, y1)], color="#4a9eed", id=name)
    d.text(1360, 1060, "Models: OpenAI-style\n/chat/completions\nTests use fakes only", 16, "#555555")

    d.step("cli", None, "A run starts",
           "From the CLI, the HTTP API (key -> tenant and role), or a queue worker that leases runs.")
    d.step("comp", "to_comp", "Compose the application",
           "Settings, profiles, tool sources and MCP servers, catalog sync, hooks, subagents, planning.")
    d.step("create", "to_create", "Create the run",
           "Tenant quota checked; system prompt, tool hashes, tenant and planning recorded; queued.",
           ["RUN_CREATED"])
    d.step("guard", "run", "Step guard",
           "Budgets and cancel checked, steering applied; in plan mode the next step or delegation.",
           ["RUN_STARTED", "STEER_APPLIED", "PLAN_STEP_UPDATED"])
    d.step("pre_model", "g_pm", "pre_model hook", "A hook can deny the call or add context for this request.")
    d.step("ctx", "pm_ctx", "Build the context view",
           "Bounded view of the history; older turns are summarized or omitted when it gets too long.",
           ["CONTEXT_SUMMARIZED"])
    d.step("model", "ctx_model", "Call the model", "The request is checkpointed before the provider is contacted.",
           ["MODEL_REQUESTED"])
    d.step("route", "req", "Route to a profile", "A route like fast or strong resolves to a configured model profile.")
    d.step("adapters", "route_ad", "Provider adapter",
           "The adapter handles the provider's quirks, headers, and health checks.")
    d.step("modal", "s_modal", "Model server", "For example Modal running vLLM, with a bearer key and proxy headers.")
    d.step("retry", ["-s_modal", "ad_retry"], "Retries, fallback, cost",
           "One retry budget per step; after it runs out the next fallback profile is tried. Cost is estimated.",
           ["MODEL_FALLBACK"])
    d.step("model", ["-ad_retry", "-route_ad", "-req"], "Response returns",
           "Only the final response is saved, even when tokens were streamed.", ["MODEL_RESPONDED"])
    d.step("dec", "model_dec", "Did the model ask for tools?", "This time it did, so the tool path runs.")
    d.step("pre_tool", "yes", "Check permissions",
           "pre_tool hook, then deny, mode, ask, and allow rules, then the tool's risk policy.", ["HOOK_DECIDED"])
    d.step("approval", "pt_appr", "Approval if needed",
           "Risky calls pause the run as AWAITING_INPUT until a person approves or denies.",
           ["APPROVAL_REQUESTED", "APPROVAL_RESOLVED"])
    d.step("runtime", "appr_rt", "Run the tool",
           "Validation, timeout, idempotency, snapshot; several read-only calls run in parallel.",
           ["TOOL_REQUESTED", "FILE_SNAPSHOT"])
    d.step("post_tool", "rt_post", "Record the result",
           "post_tool hook; untrusted output wrapped as data; large outputs become artifacts.",
           ["TOOL_RESPONDED"])
    d.step("store", "checkpoint", "Durable checkpoint",
           "Run and events (secrets masked) in one SQLite or Postgres transaction; a crash resumes here.")
    d.step("guard", ["-checkpoint", "loop"], "Next step", "The loop starts again from the step guard.")
    d.step("model", ["g_pm", "pm_ctx", "ctx_model"], "Model call again",
           "Hook, context view, and model call as before, now with the tool result in the history.",
           ["MODEL_REQUESTED", "MODEL_RESPONDED"])
    d.step("dec", "model_dec", "No tool calls", "The model answered with a final message.")
    d.step("done", "no", "Completed", "The stop hook runs and the run is marked completed.", ["RUN_COMPLETED"])
    d.step("follow", "jump", "Follow-ups",
           "Continue with a new message, approve a paused run, resume after a crash, or rewind to a step.",
           ["USER_MESSAGE_ADDED", "REWOUND"])
    return d


def permissions() -> Diagram:
    d = Diagram("02-permissions-approval", "Permissions and approval", 1660, 1010)
    d.text(40, 20, "Tool call: permissions, approval and plan mode", 30)
    names = [("call", "Tool call\nfrom the model"), ("hook", "pre_tool hook"), ("deny", "1. deny rules"),
             ("mode", "2. permission\nmode"), ("ask", "3. ask rules"), ("allow", "4. allow rules"),
             ("risk", "5. tool risk\npolicy")]
    for i, (node_id, name) in enumerate(names):
        x = 40 + i * 230
        d.box(x, 90, 200, 80, name, "blue" if i == 0 else "purple", id=node_id)
        if i < len(names) - 1:
            d.arrow([(x + 200, 130), (x + 230, 130)], id=f"to_{names[i + 1][0]}")
    tags = [
        (270, 200, 50, "deny", "red"), (270, 260, 50, "modify arguments", "yellow"),
        (500, 200, 50, "match -> deny", "red"),
        (730, 200, 60, "plan: non-read-only\n-> deny", "red"),
        (730, 275, 60, "bypass / accept_edits\n-> allow", "green"),
        (960, 200, 50, "match -> ask", "orange"),
        (1190, 200, 50, "match -> allow", "green"),
        (1420, 200, 60, "high_risk / destructive\n-> ask", "orange"),
        (1420, 275, 50, "otherwise -> allow", "green"),
    ]
    for x, y, h, label, color in tags:
        d.box(x, y, 200, h, label, color, 16)
    d.text(40, 365, "The first decision wins, left to right. Guardrails inside the tools "
                    "(workspace paths, command rules) always apply, even in bypass mode.", 18, "#555555")
    d.box(40, 430, 420, 100, "Refused\nerror result returned to the model", "red", id="refused")
    d.box(560, 430, 480, 100, "APPROVAL_REQUESTED\nrun pauses: AWAITING_INPUT", "orange", id="requested")
    d.box(1140, 430, 480, 100, "ToolRuntime executes\nvalidate, timeout, idempotency,\nfile snapshot",
          "green", id="execute")
    d.arrow([(800, 530), (800, 600)], id="to_human")
    d.box(560, 600, 480, 110, "Human decides\nagent-harness approve RUN_ID --allow / --deny\n"
          "or POST /runs/{id}/approval", "yellow", 16, id="human")
    d.arrow([(1040, 640), (1090, 640), (1090, 480), (1140, 480)], "allow", "#22c55e", id="approve")
    d.arrow([(560, 655), (250, 655), (250, 530)], "deny (+ feedback)", "#ef4444", id="reject")
    d.arrow([(1380, 530), (1380, 600)], id="to_post")
    d.box(1140, 600, 480, 90, "post_tool hook\nresult checkpointed, back to the model", "purple", id="post")

    d.zone(20, 755, 1620, 245, "Plan mode", "purple")
    d.box(40, 810, 300, 100, "Run in plan mode\nread-only tools only", "purple", id="plan")
    d.arrow([(340, 860), (400, 860)], id="p1")
    d.box(400, 810, 300, 100, "Model calls exit_plan\nwith its plan", "purple", id="exit_plan")
    d.arrow([(700, 860), (760, 860)], id="p2")
    d.box(760, 810, 300, 100, "AWAITING_INPUT\nkind = plan", "orange", id="plan_wait")
    d.arrow([(1060, 845), (1120, 825)], color="#22c55e", id="p_ok")
    d.box(1120, 790, 500, 70, "approve --mode default / accept_edits\n-> mode switches, run continues",
          "green", 16, id="plan_ok")
    d.arrow([(1060, 875), (1120, 915)], color="#ef4444", id="p_no")
    d.box(1120, 880, 500, 70, "reject with feedback\n-> the model revises the plan", "red", 16, id="plan_no")
    d.arrow([(1370, 950), (1370, 985), (550, 985), (550, 910)], "revise", "#ef4444", dashed=True, id="revise")

    d.step("call", None, "A tool call arrives", "The model asked to run a tool with some arguments.")
    d.step("hook", "to_hook", "pre_tool hook",
           "Hooks run first. They can deny the call or rewrite its arguments.", ["HOOK_DECIDED"])
    d.step("deny", "to_deny", "Deny rules", "Deny rules always win, for example run_command(argv[0]=rm).")
    d.step("mode", "to_mode", "Permission mode",
           "Plan mode refuses anything that is not read-only; bypass and accept_edits pre-approve.")
    d.step("ask", "to_ask", "Ask rules", "A matching ask rule pauses for a person to decide.")
    d.step("allow", "to_allow", "Allow rules", "A matching allow rule runs the tool without asking.")
    d.step("risk", "to_risk", "Tool risk policy",
           "With no rule matched, high-risk and destructive tools ask; everything else runs.")
    d.step("requested", "jump", "Approval requested",
           "This call needs approval, so the run pauses with status AWAITING_INPUT.",
           ["APPROVAL_REQUESTED"])
    d.step("human", "to_human", "A person decides",
           "From the CLI (approve --allow or --deny) or the HTTP API, optionally with feedback.",
           ["APPROVAL_RESOLVED"])
    d.step("execute", "approve", "Approved: the tool runs",
           "Validation, timeout, idempotency, and a file snapshot so the edit can be rewound.",
           ["TOOL_REQUESTED", "TOOL_RESPONDED"])
    d.step("post", "to_post", "Back to the model", "post_tool hook runs and the result is checkpointed.")
    d.step("plan", "jump", "Plan mode", "A run in plan mode may only read. It explores, then proposes a plan.")
    d.step("exit_plan", "p1", "exit_plan", "The model submits its plan with the exit_plan tool.")
    d.step("plan_wait", "p2", "Plan review", "The run pauses so a person can review the plan.")
    d.step("plan_no", "p_no", "Rejected with feedback", "The feedback goes back to the model.")
    d.step("exit_plan", "revise", "Revised plan", "The model revises the plan and submits it again.")
    d.step("plan_wait", "p2", "Review again", "The run pauses for review once more.")
    d.step("plan_ok", "p_ok", "Approved",
           "The mode switches (default or accept_edits) and the run continues with write access.",
           ["MODE_CHANGED"])
    return d


def subagents() -> Diagram:
    d = Diagram("03-subagents", "Subagents", 1460, 900)
    d.text(40, 20, "Subagents: the agent tool spawns a child run", 30)
    d.zone(20, 70, 1420, 330, "Parent run", "purple")
    d.box(40, 150, 240, 100, "Model calls the\nagent tool\n(name, task)", "blue", id="call")
    d.arrow([(280, 200), (320, 200)], id="a1")
    d.box(320, 150, 240, 100, "Check limits\ndepth < max_depth\nspawned < max_per_run", "purple", 16, id="limits")
    d.box(320, 280, 240, 60, "otherwise: refused,\nerror result to model", "red", 16, id="refused")
    d.arrow([(560, 200), (600, 200)], id="a2")
    d.box(600, 150, 240, 100, "SUBAGENT_STARTED\nchild id = hash of\nparent, step, call", "purple", 16,
          id="started")
    d.arrow([(840, 200), (880, 200)], id="a3")
    d.box(880, 150, 240, 100, "Parent pauses\nSUBAGENT_WAITING", "orange", id="waiting")
    d.box(1160, 150, 260, 100, "SUBAGENT_FINISHED\nfinal answer -> tool result\nusage rolls up", "green", 16,
          id="finished")
    d.zone(20, 430, 1420, 340, "Child run (same orchestrator, same store)", "blue")
    d.arrow([(720, 250), (720, 500)], "create", id="create")
    d.box(600, 500, 240, 120, "Child run\nown prompt, tool subset,\nprofile, budget\n<= what parent has left",
          "purple", 16, id="child")
    d.arrow([(840, 560), (880, 560)], id="a4")
    d.box(880, 510, 240, 100, "Child loop\nmodel + tools", "purple", id="loop")
    d.arrow([(1120, 560), (1160, 560)], id="a5")
    d.box(1160, 510, 260, 100, "Child ends\ncompleted / failed", "green", id="ends")
    d.arrow([(1290, 510), (1290, 250)], "result", "#22c55e", id="result")
    d.arrow([(1000, 610), (1000, 650)], color="#f59e0b", dashed=True, id="needs")
    d.box(880, 650, 260, 100, "Child needs approval\n-> parent AWAITING_INPUT\napprove PARENT_ID",
          "orange", 16, id="approval")
    d.box(40, 800, 1400, 70, "Cancel the parent -> its children are cancelled.   Resume the parent -> "
          "the unfinished child resumes.   Children inherit the parent's permission mode.",
          "yellow", 16, id="notes")

    d.step("call", None, "The parent delegates", "The parent's model calls the agent tool with a subagent name and task.")
    d.step("limits", "a1", "Check limits",
           "Depth must stay under max_depth and the run under max_per_run subagents.")
    d.step("started", "a2", "Child id is derived",
           "The id is a hash of parent, step, and call, so a resumed parent finds the same child.",
           ["SUBAGENT_STARTED"])
    d.step("waiting", "a3", "Parent pauses", "The parent's agent call stays pending while the child works.",
           ["SUBAGENT_WAITING"])
    d.step("child", ["-a3", "create"], "Child run is created",
           "Its own system prompt, tool subset, and profile; its budget is capped by what the parent has left.",
           ["RUN_CREATED"])
    d.step("loop", "a4", "Child works", "The child runs the normal loop on the same orchestrator and store.")
    d.step("approval", "needs", "Child needs approval",
           "The parent also shows AWAITING_INPUT; approve PARENT_ID decides the child's call.")
    d.step("loop", "-needs", "Child continues", "After the decision the child resumes its loop.")
    d.step("ends", "a5", "Child ends", "The child completes (or fails) with a final answer.")
    d.step("finished", "result", "Result returns",
           "Only the final answer returns as the tool result; the child's tokens and cost roll up.",
           ["SUBAGENT_FINISHED", "SUBAGENT_RESUMED"])
    d.step("notes", "jump", "Lifecycle rules",
           "Cancelling a parent cancels its children, and resuming a parent resumes its unfinished child.")
    return d


def summarization() -> Diagram:
    d = Diagram("04-summarization", "Context summarization", 1500, 960)
    d.text(40, 20, "Context window: trimming and summarization", 30)
    d.box(40, 90, 300, 90, "Build view\nBudgetedContextBuilder", "purple", id="build")
    d.arrow([(190, 180), (190, 220)], id="a1")
    d.diamond(60, 220, 260, 140, "messages\nomitted?", id="omitted")
    d.arrow([(190, 360), (190, 420)], "no", id="no1")
    d.box(40, 420, 300, 80, "Send view to model", "green", id="send")
    d.arrow([(320, 290), (430, 290)], "yes", id="yes1")
    d.diamond(430, 220, 260, 140, "summarizer\nenabled?", id="enabled")
    d.arrow([(560, 360), (560, 420)], "no", id="no2")
    d.box(430, 420, 260, 90, "Plain omission:\nnotice in first message", "orange", id="omit")
    d.arrow([(430, 465), (340, 460)], id="omit_send")
    d.arrow([(690, 290), (800, 290)], "yes", id="yes2")
    d.diamond(800, 220, 260, 140, "same trim point\nfailed before?", size=16, id="failed_before")
    d.arrow([(930, 360), (930, 390), (640, 390), (640, 420)], "yes", id="yes3")
    d.arrow([(1060, 290), (1170, 290)], "no", id="no3")
    d.box(1170, 235, 280, 110, "pre_compact hook", "purple", id="hook")
    d.arrow([(1170, 330), (1080, 560)], "deny", "#ef4444", id="deny")
    d.arrow([(1310, 345), (1310, 400)], "allow", id="allow")
    d.box(1170, 400, 280, 110, "Summary model call\nprevious summary +\nomitted transcript", "blue", id="call")
    d.arrow([(1310, 510), (1310, 550)], id="a2")
    d.diamond(1180, 550, 260, 140, "ok?", id="ok")
    d.arrow([(1180, 620), (1080, 620)], "no", id="not_ok")
    d.box(800, 560, 280, 110, "CONTEXT_SUMMARIZED\nok: false\n-> plain omission", "red", 16, id="fail")
    d.arrow([(800, 615), (560, 615), (560, 510)], id="fail_omit")
    d.arrow([(1310, 690), (1310, 740)], "yes", id="is_ok")
    d.box(1170, 740, 280, 130, "Store context_summary\nCONTEXT_SUMMARIZED ok\nusage -> run budget", "teal",
          id="store")
    d.arrow([(1170, 805), (1100, 815)], id="a3")
    d.box(800, 760, 300, 110, "Rebuild view:\nhead + summary\n+ newer turns", "purple", id="rebuild")
    d.arrow([(800, 815), (190, 815), (190, 500)], "send", id="send_arrow")
    d.text(40, 900, "Resume reuses the stored summary.  Rewinding behind it clears it.  "
                    "Views stay append-only between trim points (prefix cache friendly).", 16, "#555555")

    d.step("build", None, "Build the view", "Every model call gets a bounded view of the stored history.")
    d.step("omitted", "a1", "Too long?",
           "Past 85% of the input budget, the oldest turns are cut back to 60% in one chunk.")
    d.step("enabled", "yes1", "Summarization on?", "Only when [summarization] enabled = true.")
    d.step("failed_before", "yes2", "Already failed here?",
           "A failed summary is not retried until the trim point moves, so no retry storm.")
    d.step("hook", "no3", "pre_compact hook", "A hook can veto the summary.", ["HOOK_DECIDED"])
    d.step("call", "allow", "Write the summary",
           "One model call folds the previous summary and the omitted messages into a new summary.")
    d.step("ok", "a2", "Did it work?", "An empty or failed response counts as a failure.")
    d.step("store", "is_ok", "Store it durably",
           "Saved on the run and in an event; its tokens and cost count against the run budget.",
           ["CONTEXT_SUMMARIZED"])
    d.step("rebuild", "a3", "Rebuild the view",
           "System prompt, first user message with the summary, then the newer turns.")
    d.step("send", "send_arrow", "Send to the model",
           "Until the next trim point, each view only grows at the end, so prefix caches keep hitting.")
    d.step("fail", "jump", "If it fails or is denied",
           "An ok: false event is recorded and the run carries on.", ["CONTEXT_SUMMARIZED"])
    d.step("omit", "fail_omit", "Plain omission",
           "The old turns are left out with a short notice in the first user message.")
    d.step("send", "omit_send", "Send to the model", "The run never fails because a summary failed.")
    return d


def tool_platform() -> Diagram:
    d = Diagram("05-tool-platform", "Tool platform", 1500, 960)
    d.text(40, 20, "Tool platform: sources, persistent catalog, dynamic loading", 30)
    d.box(40, 90, 260, 90, "[[tools]] kind = python\nmodule:register", "purple", id="py")
    d.box(330, 90, 260, 90, "[[tools]] kind = http\ndeclarative JSON API", "purple", id="http")
    d.box(620, 90, 260, 90, "web_search (Tavily)\nkey from env or file", "purple", id="web")
    d.box(910, 90, 260, 90, "[[mcp_servers]]\nstdio / streamable HTTP", "purple", id="mcp")
    d.box(1200, 90, 260, 90, "MCP server\nprocess or endpoint", "blue", id="server")
    d.arrow([(1170, 135), (1200, 135)], color="#4a9eed", both=True, id="rpc")
    d.box(330, 260, 840, 90, "ToolRuntime registry: handlers re-attached at every start\n"
          "validation, timeouts, idempotency, approval by risk", "purple", id="registry")
    d.arrow([(170, 180), (430, 260)], id="a_py")
    d.arrow([(460, 180), (560, 260)], id="a_http")
    d.arrow([(750, 180), (750, 260)], id="a_web")
    d.arrow([(1040, 180), (1000, 260)], label="cached specs\nuntil sync", id="a_mcp")
    d.box(330, 420, 840, 100, "tool_catalog table (SQLite / Postgres)\n"
          "source, spec hash, risk, trust, enabled (switch), available", "teal", id="catalog")
    d.arrow([(750, 350), (750, 420)], "sync", id="a_sync")
    d.box(40, 420, 250, 100, "tools list / sync\nenable / disable\n(persists)", "yellow", id="admin")
    d.arrow([(290, 470), (330, 470)], id="a_admin")
    d.diamond(620, 580, 260, 130, "tool_loading\nmode?", id="mode")
    d.arrow([(750, 520), (750, 580)], "enabled + available", id="a_offer")
    d.box(250, 760, 330, 100, "mode = all\nevery enabled tool is offered", "green", id="all")
    d.arrow([(620, 645), (415, 645), (415, 760)], "all", id="a_all")
    d.box(920, 760, 470, 120, "mode = dynamic\ncore tools + search_tools + load_tools\n"
          "+ loaded tools (appended, saved on the run)", "purple", id="dynamic", size=16)
    d.arrow([(880, 645), (1155, 645), (1155, 760)], "dynamic", id="a_dyn")
    d.box(1200, 420, 270, 100, "Untrusted output\n(web, HTTP, MCP) wrapped\nas data", "red", id="untrusted", size=16)
    d.text(40, 905, "MCP tools are named server__tool, ask for approval unless read_only, and see only "
                    "the env vars named in env. Egress allowlists apply to HTTP, web search, and MCP.",
           16, "#555555")

    d.step("py", None, "Python tools", "A module function registers ToolSpec + handler pairs under its source name.")
    d.step("registry", "a_py", "Registered", "Every source's tools go into the runtime registry.")
    d.step("http", "jump", "HTTP tools", "Declarative JSON APIs: GET sends a query string, other methods a JSON body.")
    d.step("web", "jump", "Web search", "Built-in Tavily search; the key comes from an env var or a secret file.")
    d.step("mcp", "jump", "MCP servers", "Configured by command (stdio) or URL (HTTP); names become server__tool.")
    d.step("server", "rpc", "JSON-RPC", "initialize, tools/list (all pages), tools/call. Servers start on first use.",
           [])
    d.step("registry", ["-rpc", "a_mcp"], "Cached until sync",
           "Until tools sync reaches the server, its tools come from the last catalog sync.")
    d.step("catalog", "a_sync", "Persistent catalog",
           "Spec hashes, risk, trust, the operator's switch, and whether the source still provides it.")
    d.step("admin", "-a_admin", "Operator switches", "tools disable NAME hides a tool; the switch survives restarts.")
    d.step("mode", ["a_admin", "a_offer"], "What the model sees", "Only enabled and available tools are offered.")
    d.step("all", "a_all", "All tools", "The default: every enabled tool is in every request.")
    d.step("dynamic", ["-a_all", "a_dyn"], "Dynamic loading",
           "For hundreds of tools: search_tools, then load_tools; loaded tools stay for the run.",
           ["TOOLS_LOADED"])
    d.step("untrusted", "jump", "Untrusted output",
           "Results from web, HTTP, and MCP tools are wrapped as data (prompt-injection defence).")
    return d


def plan_execute() -> Diagram:
    d = Diagram("06-plan-and-execute", "Plan and execute", 1500, 960)
    d.text(40, 20, "Plan-and-execute: the orchestrator owns and drives the plan", 30)
    d.box(40, 90, 250, 90, "Task + planner\ninstructions", "blue", id="task")
    d.arrow([(290, 135), (340, 135)], id="a1")
    d.box(340, 90, 260, 90, "Explore\n(read-only tools only)", "purple", id="explore")
    d.arrow([(600, 135), (650, 135)], id="a2")
    d.box(650, 90, 260, 90, "create_plan\ngoal + steps", "purple", id="create")
    d.arrow([(910, 135), (960, 135)], id="a3")
    d.diamond(960, 70, 300, 130, "valid? ids, deps,\nno cycles, agents", size=16, id="valid")
    d.arrow([(1260, 135), (1320, 135), (1320, 60), (780, 60), (780, 90)], "invalid: error to model",
            "#ef4444", dashed=True, id="invalid")
    d.box(1000, 250, 260, 80, "PLAN_CREATED\nplan saved on the run", "teal", id="stored")
    d.arrow([(1110, 200), (1110, 250)], "yes", id="a4")
    d.box(560, 250, 360, 90, "Orchestrator picks\nready steps (deps done)", "yellow", id="loop")
    d.arrow([(1000, 290), (920, 290)], id="a5")
    d.box(40, 420, 440, 120, "Agent steps, ready together\nchild runs started by the orchestrator\n"
          "IN PARALLEL (budget split)", "green", id="agents", size=16)
    d.arrow([(620, 340), (300, 420)], "agent steps", id="a_agents")
    d.box(40, 590, 200, 70, "researcher\nchild run", "blue", id="child1")
    d.box(280, 590, 200, 70, "writer\nchild run", "blue", id="child2")
    d.arrow([(140, 540), (140, 590)], id="a_c1")
    d.arrow([(380, 540), (380, 590)], id="a_c2")
    d.box(40, 720, 440, 90, "Results collected (plan outbox)\nno model turn needed", "yellow", id="results")
    d.arrow([(140, 660), (140, 720)], id="a_r1")
    d.arrow([(380, 660), (380, 720)], id="a_r2")
    d.arrow([(480, 765), (510, 765), (510, 315), (560, 315)], id="a_back")
    d.box(560, 420, 420, 120, "Main-model step\n\"[Plan] Execute step ...\"\nupdate_plan or a final answer", "purple",
          id="main", size=16)
    d.arrow([(760, 340), (760, 420)], "own steps", id="a_main")
    d.box(1040, 420, 420, 120, "Step failed\n\"[Plan] ... failed\" -> revise_plan\n(at most max_replans)", "red",
          id="failed", size=16)
    d.arrow([(980, 480), (1040, 480)], "failed", id="a_fail")
    d.box(1040, 720, 420, 90, "revise_plan\ncompleted steps are kept", "purple", id="revise")
    d.arrow([(1250, 540), (1250, 720)], id="a_rev")
    d.arrow([(1040, 765), (1010, 765), (1010, 330), (920, 330)], id="a_rev_back")
    d.box(560, 720, 420, 90, "All steps finished\n-> final answer -> completed", "green", id="final")
    d.arrow([(760, 540), (760, 720)], "all done", id="a_final")
    d.text(40, 860, "Plans are stored on the run with plan_* events. Approvals inside a child pause the parent; "
                    "a crash resumes at the same step.", 16, "#555555")

    d.step("task", None, "Plan-and-execute", "run --plan, 'planning': true, or [planning] enabled = true.")
    d.step("explore", "a1", "Explore first", "Before a plan exists only read-only tools and create_plan are offered.")
    d.step("create", "a2", "Propose the plan", "Steps have ids, depends_on, and an optional agent to delegate to.")
    d.step("valid", "a3", "Validate", "Unique ids, known dependencies, no cycles, known agents; errors go back.")
    d.step("stored", "a4", "Plan stored", "The orchestrator now owns the plan.", ["PLAN_CREATED"])
    d.step("loop", "a5", "Pick ready steps", "Steps whose dependencies are completed.")
    d.step("agents", "a_agents", "Delegate agent steps",
           "The orchestrator starts the child runs itself; no model turn needed.", ["SUBAGENT_STARTED"])
    d.step("child1", "a_c1", "Children in parallel", "researcher and writer run at the same time.")
    d.step("child2", ["-a_c1", "a_c2"], "Budget split", "The parent's remaining tokens and cost are shared.")
    d.step("results", "a_r2", "Collect results", "Results wait in the plan's outbox for the next message.",
           ["SUBAGENT_FINISHED", "PLAN_STEP_UPDATED"])
    d.step("loop", "a_back", "Next boundary", "Results and the next instruction go out as one message.")
    d.step("main", "a_main", "Main-model step", "The model does its own steps one at a time.")
    d.step("failed", "a_fail", "A step fails", "The model is asked to revise the plan.")
    d.step("revise", "a_rev", "Replan", "Unfinished steps are replaced; completed ones kept.", ["PLAN_REVISED"])
    d.step("loop", "a_rev_back", "Continue", "Execution continues with the revised plan.")
    d.step("final", ["a_main", "a_final"], "Finish", "All steps done: the final answer completes the run.",
           ["RUN_COMPLETED"])
    return d


def api_access() -> Diagram:
    d = Diagram("07-api-access", "API access and tenancy", 1500, 860)
    d.text(40, 20, "API access: keys, roles, tenants, quotas", 30)
    d.box(40, 100, 220, 80, "HTTP request\nBearer <key>", "blue", id="req")
    d.arrow([(260, 140), (300, 140)], id="b1")
    d.diamond(300, 75, 220, 130, "known key?", id="auth")
    d.box(300, 290, 220, 70, "401", "red", id="e401")
    d.arrow([(410, 205), (410, 290)], "no", "#ef4444", id="n401")
    d.arrow([(520, 140), (560, 140)], "yes", id="b2")
    d.box(560, 100, 240, 80, "Principal\nname, tenant, role", "purple", id="principal")
    d.arrow([(800, 140), (840, 140)], id="b3")
    d.diamond(840, 75, 220, 130, "role allows?\nviewer<operator\n<admin", size=16, id="role")
    d.box(840, 290, 220, 70, "403", "red", id="e403")
    d.arrow([(950, 205), (950, 290)], "no", "#ef4444", id="n403")
    d.arrow([(1060, 140), (1100, 140)], "yes", id="b4")
    d.diamond(1100, 75, 240, 130, "run in my\ntenant?", id="tenant")
    d.box(1360, 105, 120, 70, "404", "red", id="e404")
    d.arrow([(1340, 140), (1360, 140)], color="#ef4444", id="n404")
    d.diamond(1100, 400, 240, 130, "quota OK?\n(POST /runs)", size=16, id="quota")
    d.arrow([(1220, 205), (1220, 400)], "yes", id="b5")
    d.box(1360, 430, 120, 70, "429", "red", id="e429")
    d.arrow([(1340, 465), (1360, 465)], color="#ef4444", id="n429")
    d.box(780, 425, 260, 80, "create_run\ntenant + created_by", "green", id="create")
    d.arrow([(1100, 465), (1040, 465)], "yes", id="b6")
    d.box(460, 425, 260, 80, "Queue\n(Postgres: SKIP LOCKED)", "teal", id="queue")
    d.arrow([(780, 465), (720, 465)], id="b7")
    d.box(140, 425, 260, 80, "Worker\nlease, run, checkpoint", "purple", id="worker")
    d.arrow([(460, 465), (400, 465)], id="b8")
    d.box(140, 600, 580, 90, "Run store (SQLite / Postgres): migrations on open,\nevents masked by redaction",
          "teal", id="store", size=16)
    d.arrow([(270, 505), (270, 600)], id="b9")
    d.box(780, 600, 260, 90, "Approve / mode change\nactor recorded", "orange", id="actor")
    d.box(1100, 600, 380, 90, "/healthz  /readyz\n(no key; store + schema version)", "yellow", id="ready")
    d.text(40, 750, "Keys come from [[api.keys]] by SHA-256 or env var; tenant '*' sees every tenant. "
                    "Quotas: runs, tokens, cost per 24h, and concurrent runs.", 16, "#555555")

    d.step("req", None, "A request arrives", "Every route except /healthz and /readyz needs a bearer key.")
    d.step("auth", "b1", "Authenticate", "The key's SHA-256 is compared in constant time; unknown -> 401.")
    d.step("principal", "b2", "Principal", "The key names a principal with a tenant and a role.")
    d.step("role", "b3", "Authorize", "viewer reads; operator acts on runs; admin changes modes and rewinds.")
    d.step("tenant", "b4", "Tenant isolation", "Runs of other tenants answer 404, so their existence is not leaked.")
    d.step("quota", "b5", "Quotas", "Runs, tokens, and cost per day and concurrent runs; over the limit -> 429.")
    d.step("create", "b6", "Create the run", "Stamped with the tenant and the key's name.", ["RUN_CREATED"])
    d.step("queue", "b7", "Enqueue", "Workers on any host claim runs with leases.")
    d.step("worker", "b8", "Execute", "A worker leases the run, executes it, and checkpoints every step.")
    d.step("store", "b9", "Durable and masked", "Secrets are masked in events as they are written.")
    d.step("actor", "jump", "Accountable decisions", "Approvals and mode changes record which key acted.",
           ["APPROVAL_RESOLVED", "MODE_CHANGED"])
    d.step("ready", "jump", "Health", "/readyz checks the store and reports the schema version.")
    return d


DIAGRAMS = (end_to_end, permissions, subagents, summarization, tool_platform, plan_execute, api_access)

if __name__ == "__main__":
    target = Path(sys.argv[1])
    data_out = Path(sys.argv[2])
    target.mkdir(parents=True, exist_ok=True)
    tours = []
    for factory in DIAGRAMS:
        diagram = factory()
        (target / f"{diagram.name}.excalidraw").write_text(
            json.dumps(diagram.excalidraw(), indent=1), encoding="utf-8")
        (target / f"{diagram.name}.svg").write_text(diagram.svg(), encoding="utf-8")
        (target / f"{diagram.name}.animated.svg").write_text(diagram.animated_svg(), encoding="utf-8")
        tours.append(diagram.tour())
        print("wrote", diagram.name, len(diagram.steps), "steps", f"{diagram.timeline()[1]:.0f}s loop")
    data_out.write_text(json.dumps(tours), encoding="utf-8")
    if len(sys.argv) > 3:  # optional: page template -> interactive walkthrough page
        template = Path(sys.argv[3]).read_text(encoding="utf-8")
        page = data_out.with_suffix(".html")
        page.write_text(template.replace("__TOURS__", json.dumps(tours).replace("</", "<\\/")), encoding="utf-8")
        print("wrote", page)
