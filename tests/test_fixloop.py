"""Fix-loop helper: worst-gate ranking, declaration blanks, section rewriting."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("fixloop_loop", ROOT / "fixloop" / "loop.py")
loop = importlib.util.module_from_spec(spec)
spec.loader.exec_module(loop)


def _metrics():
    return {
        "gates": [
            {"name": "wall lengths", "tier": "lidar", "threshold": "t", "value": "11/12 within", "passed": False,
             "detail": "", "score": 11 / 12},
            {"name": "opening widths", "tier": "lidar", "threshold": "t", "value": "50% (4/8)", "passed": False,
             "detail": "missed 1", "score": 0.5 / 0.85},
            {"name": "ceiling height", "tier": "lidar", "threshold": "t", "value": "3/3 rooms", "passed": True,
             "detail": "", "score": 1.0},
            {"name": "wall lengths", "tier": "photo", "threshold": "t", "value": "n/a", "passed": None, "score": None},
        ],
        "repeatability": [{"pair": ["a", "b"], "tier": "lidar", "passed": False,
                           "rows": [{"ok": True}, {"ok": True}, {"ok": False}, {"ok": True}]}],
    }


def test_worst_gate_is_the_furthest_from_its_requirement():
    fails = loop.failing_gates(_metrics())
    assert [f["name"] for f in fails] == ["opening widths", "repeatability (a vs b)", "wall lengths"]


def test_template_has_blanks_and_a_filled_declaration_has_none():
    template = (ROOT / "fixloop" / "DECLARATION.md").read_text(encoding="utf-8")
    assert loop.blanks(template)
    fails = loop.failing_gates(_metrics())
    text = loop.replace_section(template, 1, loop.section1(fails[0], fails[1:], "benchmark/results/abc", "abc"))
    text = loop.replace_section(text, 2, "## 2. Root-cause hypothesis and evidence\n\n- Hypothesis: jambs from "
                                         "coarse depth\n- Evidence:\n  1. errors are +-1 depth pixel\n\n")
    text = loop.replace_section(text, 3, "## 3. The fix and the predicted number\n\n- Fix: RGB edges\n"
                                         "- Predicted value: 85 %\n\n")
    assert loop.blanks(text) == []
    assert "## 4." in text and "After-run value" in text  # section 4 left for later
    assert loop.declared_gate(text) == ("opening widths", "lidar")
