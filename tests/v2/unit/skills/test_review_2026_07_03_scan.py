"""Regression pins for the 2026-07-03 review — scan / imaging cluster.

Covers:
  * parse_frame_grab handles the real heterogeneous Nanonis body
    [name_len, name, rows, cols, data_2D, dir] (bare np.asarray raised
    "inhomogeneous shape" on every real scan).
  * WaitScanComplete stops the scan on timeout (stop_on_timeout default True).
  * ConfigureScan preserves the current scan angle when none is supplied.
  * SetScanSpeed keep_const matches Nanonis (0/1/2).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

_MASTV2_ROOT = str(Path(__file__).resolve().parents[4] / "MASTv2")
if sys.path and sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _n in list(sys.modules):
    if _n == "mast" or _n.startswith("mast."):
        _f = getattr(sys.modules[_n], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_n]

from mast.core.types import NanonisCallRecord
from mast.io.nanonis_files import parse_frame_grab
from mast.skills.builtins.imaging import ConfigureScan, SetScanSpeed


@dataclass
class FakeCtx:
    canned: dict[str, Any] = field(default_factory=dict)
    calls: list = field(default_factory=list)

    def safe_call(self, method, *args, role="main"):
        self.calls.append((method, args))
        e = self.canned.get(method, {})
        return NanonisCallRecord(method=method, args=args,
                                 return_value=e.get("return_value"),
                                 error=e.get("error", ""))


def test_parse_frame_grab_heterogeneous_body():
    # Real Nanonis body: [name_len, name(str), rows, cols, data_2D(ndarray), dir]
    img = np.arange(6, dtype=np.float64).reshape(2, 3)
    body = [1, "Z (m)", 2, 3, img, 1]
    parsed = ("", b"", body)
    flat = parse_frame_grab(parsed)
    assert flat is not None
    assert flat.tolist() == [0, 1, 2, 3, 4, 5]
    grid = parse_frame_grab(parsed, shape_2d=True)
    assert grid is not None and grid.shape == (2, 3)


def test_parse_frame_grab_flat_stub():
    # Flat-list stub (no name string) still works.
    parsed = ("", b"", [0.0, 1.0, 2.0, 3.0])
    flat = parse_frame_grab(parsed)
    assert flat is not None and flat.tolist() == [0.0, 1.0, 2.0, 3.0]


def test_configure_scan_preserves_angle_when_unspecified():
    # Current frame angle = 30°; ConfigureScan without angle_deg must keep it.
    frame = ("", b"", [0.0, 0.0, 50e-9, 50e-9, 30.0])
    ctx = FakeCtx(canned={"Scan_FrameGet": {"return_value": frame},
                          "Scan_FrameSet": {"return_value": ("", b"", [])}})
    ConfigureScan().execute(ctx, {"center_x_m": 0.0, "center_y_m": 0.0,
                                  "width_m": 50e-9, "height_m": 50e-9})
    frameset = next(args for m, args in ctx.calls if m == "Scan_FrameSet")
    assert frameset[4] == 30.0, "angle must be preserved, not reset to 0"


def _no_frameget_before_frameset(ctx):
    """FrameSet 之前有没有读过帧。

    2026-08-28：ConfigureScan 在 FrameSet **之后**加了一次读回校验
    （仪器把框夹到量程内时，上层原来完全看不出来）。所以「FrameGet 一次
    都不调」这个断言过强了 —— 它会把那道新闸门也一并禁掉。
    原意是「别靠读来决定写什么」，精确的写法是**读不能发生在写之前**。
    """
    for m, _ in ctx.calls:
        if m == "Scan_FrameSet":
            return True
        if m == "Scan_FrameGet":
            return False
    return True


def test_configure_scan_uses_explicit_angle():
    ctx = FakeCtx(canned={
        "Scan_FrameSet": {"return_value": ("", b"", [])},
        # 回显刚设置的帧，供 FrameSet 后的读回校验比较。
        "Scan_FrameGet": {"return_value": ("", b"", [0.0, 0.0, 50e-9, 50e-9, 15.0])}})
    ConfigureScan().execute(ctx, {"center_x_m": 0.0, "center_y_m": 0.0,
                                  "width_m": 50e-9, "height_m": 50e-9,
                                  "angle_deg": 15.0})
    frameset = next(args for m, args in ctx.calls if m == "Scan_FrameSet")
    assert frameset[4] == 15.0
    # 显式角度时不该**先读再写**。（FrameSet 之后仍有一次读回校验，
    # 那是另一件事 —— 见 2026-08-28 的帧读回闸门。）
    assert _no_frameget_before_frameset(ctx)


# ---------------------------------------------------------------------------
# 2026-08-05: unreadable current angle must REFUSE, not fabricate 0°.
#
# "Preserve the current angle" (the pin above) had four ways to fail back to
# 0.0, and only ONE of them raised. The other three were silent — including
# the likeliest one in the field: safe_call *records* a comms error instead of
# raising, and nobody ever read `rec_fg.error`. A fabricated 0° then rotates
# the frame flat, which is exactly what the caller asked NOT to do.
#
# What these tests pin is "NOT A SINGLE HARDWARE WRITE happened" — returning
# an error but ramming Scan_FrameSet through anyway would be no fix at all
# (same yardstick as the SetBiasRamp pins, bb69972).
# ---------------------------------------------------------------------------

_GEOM = {"center_x_m": 0.0, "center_y_m": 0.0, "width_m": 50e-9, "height_m": 50e-9}


def _assert_refused_no_write(ctx, result):
    assert result.success is False
    assert not any(m == "Scan_FrameSet" for m, _ in ctx.calls), \
        "refusal must mean ZERO hardware writes, not error-plus-write"
    # The message must hand the caller the way out — an unexplained refusal
    # reads as over-strictness and gets deleted by the next reviewer.
    assert "angle_deg=0.0" in (result.error or "")


def test_configure_scan_refuses_when_frameget_records_error():
    # safe_call returns an error RECORD (does not raise) — the silent path a
    # real comms fault takes, and the one the old code never looked at.
    ctx = FakeCtx(canned={"Scan_FrameGet": {"return_value": None,
                                            "error": "timeout"}})
    result = ConfigureScan().execute(ctx, dict(_GEOM))
    _assert_refused_no_write(ctx, result)


def test_configure_scan_refuses_when_body_is_not_a_sequence():
    ctx = FakeCtx(canned={"Scan_FrameGet": {"return_value": None}})
    result = ConfigureScan().execute(ctx, dict(_GEOM))
    _assert_refused_no_write(ctx, result)


def test_configure_scan_refuses_when_frame_body_too_short():
    # Variables carry only 4 values — no angle at index 4.
    frame = ("", b"", [0.0, 0.0, 50e-9, 50e-9])
    ctx = FakeCtx(canned={"Scan_FrameGet": {"return_value": frame}})
    result = ConfigureScan().execute(ctx, dict(_GEOM))
    _assert_refused_no_write(ctx, result)


def test_configure_scan_refuses_nan_angle():
    # float("nan") is a legal float: without isfinite it would sail through
    # and Scan_FrameSet would receive NaN as the angle.
    frame = ("", b"", [0.0, 0.0, 50e-9, 50e-9, float("nan")])
    ctx = FakeCtx(canned={"Scan_FrameGet": {"return_value": frame}})
    result = ConfigureScan().execute(ctx, dict(_GEOM))
    _assert_refused_no_write(ctx, result)


def test_configure_scan_refuses_when_frameget_raises():
    class RaisingCtx(FakeCtx):
        def safe_call(self, method, *args, role="main"):
            if method == "Scan_FrameGet":
                raise RuntimeError("boom")
            return super().safe_call(method, *args, role=role)

    ctx = RaisingCtx()
    result = ConfigureScan().execute(ctx, dict(_GEOM))
    _assert_refused_no_write(ctx, result)


def test_configure_scan_explicit_zero_is_honoured():
    # The refusal message tells callers "pass angle_deg=0.0 if you mean 0°" —
    # so a DELIBERATE 0° must work, and must not even consult FrameGet.
    ctx = FakeCtx(canned={
        "Scan_FrameSet": {"return_value": ("", b"", [])},
        "Scan_FrameGet": {"return_value": ("", b"", [
            _GEOM["center_x_m"], _GEOM["center_y_m"],
            _GEOM["width_m"], _GEOM["height_m"], 0.0])}})
    result = ConfigureScan().execute(ctx, {**_GEOM, "angle_deg": 0.0})
    assert result.success is True
    frameset = next(args for m, args in ctx.calls if m == "Scan_FrameSet")
    assert frameset[4] == 0.0
    assert _no_frameget_before_frameset(ctx)


def test_configure_scan_angle_schema_default_must_stay_none():
    # Pin the SCHEMA, not just the code: wrap_skill materialises
    # ParameterSpec.default into the pydantic field (same mechanism the
    # full_scan.line_time_s comment documents), so `default=0.0` here would
    # make the LLM path deliver an EXPLICIT 0.0 whenever the model omits the
    # argument — "preserve current angle" and the refusal above would both
    # become dead code on that path, silently. If you want a default back,
    # answer first: how does the adapter tell "omitted" from "chose 0°"?
    md = ConfigureScan().metadata()
    spec = {p.name: p for p in md.parameters}["angle_deg"]
    assert spec.default is None
    assert spec.required is False


def test_set_scan_speed_keep_const_allows_nanonis_values():
    md = SetScanSpeed().metadata()
    spec = {p.name: p for p in md.parameters}["keep_const"]
    assert spec.allowed_values == [0, 1, 2]
