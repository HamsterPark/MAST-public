""" — `show_plan_on_map` 收下「米」量级坐标并回 success。

17 show / 11 clear calls, 18 impossible ``[xywh]_m`` values
(``x_m=6.0`` — a one-metre scan box six metres away), every one answered
``{"success": true}``. The agent re-published the same nonsense five times in
two minutes because nothing ever told it the values were wrong.

The fix must distinguish TWO things, and the tests below pin both:

  * **量级荒谬** — the number is not an STM-scale length. Reject.
  * **单位** — NOT the problem. Every skill parameter in this repo is SI and
    ``_m`` means metres, so ``0.5`` really is the correct SI spelling of half a
    metre. The message must say so, and must not accuse the caller of dropping
    an exponent (a sibling guard did exactly that to correctly-written values).

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest \
        tests/v2/unit/agents/test_show_plan_on_map_magnitude.py -x -v
"""
from __future__ import annotations

# ── path setup BEFORE any mast.* imports (canonical block for tests/v2/) ──
import sys
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[4] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _name in list(sys.modules):
    if _name == "mast" or _name.startswith("mast."):
        _f = getattr(sys.modules[_name], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_name]

import json

import pytest

from mast.agents._shared import meta_tools as mt
from mast.io.plan_overlay import get_plan_overlay


# The three field payloads, verbatim from the forensics report.
FIELD_STEPS = [
    {"kind": "scan", "label": "P1", "x_m": 6.0, "y_m": 1.4, "w_m": 1.0, "h_m": 1.0},
    {"kind": "scan", "label": "overview 500nm",
     "x_m": 1.253, "y_m": 1.953, "w_m": 5.0, "h_m": 5.0},
    {"kind": "scan", "label": "A 100nm",
     "x_m": 1.0025, "y_m": 1.7031, "w_m": 1.5, "h_m": 1.5},
]

# The same intent written at STM scale: a 100 nm box 200 nm off centre.
SANE_STEPS = [
    {"kind": "scan", "label": "A 100nm", "x_m": 2e-7, "y_m": 1.4e-7,
     "w_m": 1e-7, "h_m": 1e-7},
    {"kind": "sts", "label": "point 1", "x_m": 2.1e-7, "y_m": 1.4e-7},
]


@pytest.fixture
def tools():
    """The real meta-tool list, bound to an empty context (no live core needed)."""
    get_plan_overlay().clear()
    yield {t.name: t for t in mt.make_meta_tools(lambda: {})}
    get_plan_overlay().clear()


def _show(tools, steps, title=""):
    return json.loads(tools["show_plan_on_map"].invoke(
        {"steps": steps, "title": title}))


# ──────────────────────────────────────────────────────────────────────
# The rejection itself
# ──────────────────────────────────────────────────────────────────────

class TestAbsurdMagnitudeRejected:

    def test_field_payload_is_rejected(self, tools):
        out = _show(tools, FIELD_STEPS)
        assert out["success"] is False
        assert out["problems"]

    def test_nothing_is_published_when_any_step_is_absurd(self, tools):
        """Reject the WHOLE route — a partial publish misreports the screen."""
        _show(tools, [SANE_STEPS[0], FIELD_STEPS[0]])
        assert get_plan_overlay().snapshot() == []

    def test_each_bad_field_is_named(self, tools):
        out = _show(tools, [FIELD_STEPS[0]])
        blob = " ".join(out["problems"])
        for field in ("x_m", "y_m", "w_m", "h_m"):
            assert field in blob, f"{field} not reported"

    def test_step_is_identified_by_index_and_label(self, tools):
        out = _show(tools, [SANE_STEPS[0], FIELD_STEPS[1]])
        blob = " ".join(out["problems"])
        assert "第 2 步" in blob
        assert "overview 500nm" in blob

    def test_metre_scale_extent_rejected_even_at_a_sane_position(self, tools):
        """w_m is checked against the extent cap, not the position cap."""
        out = _show(tools, [{"kind": "scan", "x_m": 1e-7, "y_m": 1e-7,
                             "w_m": 1.0, "h_m": 1.0}])
        assert out["success"] is False
        assert any("w_m" in p for p in out["problems"])

    @pytest.mark.parametrize("bad", [float("nan"), float("inf")])
    def test_non_finite_rejected(self, tools, bad):
        out = _show(tools, [{"kind": "scan", "x_m": bad, "y_m": 1e-7}])
        assert out["success"] is False


# ──────────────────────────────────────────────────────────────────────
# What the message may and may not say
# ──────────────────────────────────────────────────────────────────────

class TestMessageDiagnosesMagnitudeNotUnits:

    def test_says_the_unit_is_correct(self, tools):
        """The caller must not "fix" this by switching away from metres."""
        blob = " ".join(_show(tools, [FIELD_STEPS[0]])["problems"])
        assert "单位没错" in blob
        assert "量级" in blob

    def test_states_the_working_envelope(self, tools):
        out = _show(tools, [FIELD_STEPS[0]])
        assert "1.5e-06" in out["expected"] or "1.5e-06" in " ".join(out["problems"])

    def test_quantifies_how_far_out(self, tools):
        """A ratio, not an adjective — 6 m is 4e6 × the piezo range."""
        blob = " ".join(_show(tools, [FIELD_STEPS[0]])["problems"])
        assert "倍" in blob
        assert "4e+06" in blob or "4.0e+06" in blob or "4e6" in blob

    def test_offers_conversions_without_asserting_intent(self, tools):
        """Arithmetic, explicitly disclaimed — NOT "you dropped an exponent"."""
        blob = " ".join(_show(tools, [FIELD_STEPS[0]])["problems"])
        assert "6e-06" in blob and "6e-09" in blob      # µm and nm readings
        assert "无法判定你的本意" in blob
        # The phrasing that misdiagnosed correctly-written values elsewhere.
        assert "指数被吞" not in blob
        assert "漏了指数" not in blob


class TestHintArithmetic:
    """`_plan_magnitude_hint` is pure — check its arithmetic directly."""

    def test_offers_only_in_range_readings(self):
        hint = mt._plan_magnitude_hint(6.0)
        assert "6e-06" in hint and "6e-09" in hint

    def test_drops_a_reading_that_is_still_out_of_range(self):
        """1e6 µm = 1 m — still absurd, so don't offer it."""
        hint = mt._plan_magnitude_hint(1e6)
        assert "µm" not in hint
        assert "0.001" in hint or "1e-03" in hint     # the nm reading survives

    def test_no_hint_when_nothing_lands_in_range(self):
        assert mt._plan_magnitude_hint(1e12) == ""


# ──────────────────────────────────────────────────────────────────────
# Real STM values must still go through — the guard must not overfire
# ──────────────────────────────────────────────────────────────────────

class TestRealValuesStillPublish:

    def test_stm_scale_plan_publishes(self, tools):
        out = _show(tools, SANE_STEPS, title="plan A")
        assert out["success"] is True
        assert out["steps"] == 2
        assert "warnings" not in out
        assert len(get_plan_overlay().snapshot()) == 2

    def test_full_piezo_range_is_accepted(self, tools):
        """±1.5 µm is the working area, not an error."""
        out = _show(tools, [{"kind": "scan", "x_m": 1.5e-6, "y_m": -1.5e-6,
                             "w_m": 1e-7, "h_m": 1e-7}])
        assert out["success"] is True
        assert "warnings" not in out

    def test_steps_without_extent_are_fine(self, tools):
        out = _show(tools, [{"kind": "sts", "x_m": 1e-7, "y_m": 1e-7}])
        assert out["success"] is True

    def test_coarse_motion_offset_warns_but_publishes(self, tools):
        """Conceivable after coarse motion — warn about the display cost, allow it.

        Measured: ScanMapCanvas auto-fits over every marker, so a distant step
        stretches the shared viewport and squashes the nanometre-scale markers.
        That is a display consequence worth saying out loud, not a rejection.
        """
        out = _show(tools, [{"kind": "scan", "x_m": 5e-5, "y_m": 0.0,
                             "w_m": 1e-7, "h_m": 1e-7}])
        assert out["success"] is True
        assert out["warnings"]
        assert "压电量程" in " ".join(out["warnings"])
        assert len(get_plan_overlay().snapshot()) == 1


# ──────────────────────────────────────────────────────────────────────
# 推测-2: do absurd coordinates push the route OFF the canvas?
# ──────────────────────────────────────────────────────────────────────

class TestPlanRouteRenderingReach:
    """Replicates ScanMapCanvas.tsx `collectBounds` + `view` (metres → pixels).

    推测-2 said the metre-scale coordinates were the reason the route did not
    display, on the theory that they fall outside the visible area. They do not:
    the canvas AUTO-FITS over every marker, so the offending steps define the
    viewport instead of escaping it. What they actually destroy is everything
    else on the map.
    """

    CANVAS_W, CANVAS_H, PAD, FIXED = 620, 460, 44, 1e-7

    @classmethod
    def _view(cls, points):
        xs, ys = [], []
        for x, y, w, h in points:
            hw, hh = abs(w or 0) / 2, abs(h or 0) / 2
            xs += [x - hw, x + hw]
            ys += [y - hh, y + hh]
        minX, maxX, minY, maxY = min(xs), max(xs), min(ys), max(ys)
        cx, cy = (minX + maxX) / 2, (minY + maxY) / 2
        half = max(maxX - minX, maxY - minY, cls.FIXED) / 2
        pad = half * 0.14
        minX, maxX = cx - half - pad, cx + half + pad
        minY, maxY = cy - half - pad, cy + half + pad
        spanX, spanY = maxX - minX, maxY - minY
        s = min((cls.CANVAS_W - 2 * cls.PAD) / spanX,
                (cls.CANVAS_H - 2 * cls.PAD) / spanY)
        offX = cls.PAD + (cls.CANVAS_W - 2 * cls.PAD - spanX * s) / 2
        offY = cls.PAD + (cls.CANVAS_H - 2 * cls.PAD - spanY * s) / 2
        return s, (lambda xm, ym: (offX + (xm - minX) * s,
                                   cls.CANVAS_H - (offY + (ym - minY) * s)))

    PLAN = [(6.0, 1.4, 1.0, 1.0), (1.253, 1.953, 5.0, 5.0), (1.0025, 1.7031, 1.5, 1.5)]
    REAL = [(0.0, 0.0, 5e-7, 5e-7), (1.43e-6, 1.35e-6, 0, 0), (2e-7, -1e-7, 1e-7, 1e-7)]

    def _onscreen(self, p):
        return 0 <= p[0] <= self.CANVAS_W and 0 <= p[1] <= self.CANVAS_H

    def test_absurd_plan_steps_land_ON_the_canvas(self):
        """推测-2 disproved: auto-fit means they cannot fall outside."""
        _s, to_px = self._view(self.PLAN)
        for step in self.PLAN:
            assert self._onscreen(to_px(step[0], step[1]))

    def test_absurd_plan_steps_collapse_the_real_data(self):
        """The real harm: every real marker lands on ONE pixel."""
        _s, to_px = self._view(self.PLAN + self.REAL)
        pixels = {tuple(round(c) for c in to_px(r[0], r[1])) for r in self.REAL}
        assert len(pixels) == 1, "real markers should be indistinguishable"

    def test_real_data_alone_is_legible(self):
        """Control: without the absurd steps the same markers are distinct."""
        s, to_px = self._view(self.REAL)
        pixels = {tuple(round(c) for c in to_px(r[0], r[1])) for r in self.REAL}
        assert len(pixels) == len(self.REAL)
        assert self.REAL[0][2] * s > 10, "500 nm frame should be tens of pixels"
