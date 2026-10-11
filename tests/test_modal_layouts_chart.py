"""docs/diagrams/10-modal-layouts.svg is generated from tools/modal_layouts.py and kept in step."""

from __future__ import annotations

import importlib.util
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "docs" / "diagrams" / "tools" / "modal_layouts.py"
CHART = ROOT / "docs" / "diagrams" / "10-modal-layouts.svg"

_spec = importlib.util.spec_from_file_location("modal_layouts", TOOL)
assert _spec is not None and _spec.loader is not None
chart = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(chart)


class ModalLayoutsChartTests(unittest.TestCase):
    def test_committed_svg_matches_the_generator(self) -> None:
        # Edit DATA in the tool, then run it: python docs/diagrams/tools/modal_layouts.py <svg>.
        self.assertEqual(CHART.read_text(encoding="utf-8"), chart.build())

    def test_svg_is_well_formed_and_names_all_three_layouts(self) -> None:
        root = ET.fromstring(chart.build())
        texts = " ".join(t.text or "" for t in root.iter("{http://www.w3.org/2000/svg}text"))
        for group in chart.DATA:
            self.assertIn(group["title"], texts)

    def test_session_prices_follow_the_rates(self) -> None:
        h200, l4 = 0.001261 * 60, 0.000222 * 60
        dedicated, shared_two, shared_one = chart.DATA
        self.assertAlmostEqual(dedicated["session"], (chart.H200_COLD + 30) * (h200 + l4))
        self.assertAlmostEqual(shared_two["session"], (chart.GUARD_COLD + 30) * l4 + chart.TOKENS_PER_SESSION)
        self.assertAlmostEqual(shared_one["session"], (chart.SHARED_L4_COLD + 30) * l4 + chart.TOKENS_PER_SESSION)
        self.assertEqual(shared_one["embed_extra"], 0)  # the shared L4 carries the embed model


if __name__ == "__main__":
    unittest.main()
