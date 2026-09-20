"""Numeric spectrum data for the Data tab's interactive curve.

A Nanonis point spectrum reached the frontend as ONE thing: a small PNG of the
file's first two columns, no axes, no labels, no zoom. Every other column —
the backward sweep, the lock-in channel that IS the dI/dV — was thrown away
before the wire. This module hands over the numbers instead, so the client can
draw an I-V or a dI/dV curve the operator can zoom into.

Design notes:

* **Nothing is dropped.** Columns whose role we cannot name still ship (the
  ``columns`` list always holds every name, and an unrecognisable file falls back
  to "sweep = first column, series = the rest"). A viewer that silently shows a
  subset of what is in the file is the failure this codebase keeps recording from
  the other direction.
* **dI/dV has three states, not two** — a real lock-in column, a numeric
  derivative of I(V), or nothing. They must not look alike on screen, so the
  response says which one it is and the client labels the numeric one. Silently
  differentiating and calling the result "dI/dV" would put a noisy derivative
  next to genuine lock-in data under the same name.
* The column-role patterns are COPIED from the skills that own them
  (``skills/builtins/spectrum_assess.py`` and ``tip_spectro_assess.py``) rather
  than imported: ``webui`` must not pull in the skill machinery to draw a curve.
  A test pins the copies against the originals so the two cannot drift apart
  unnoticed.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# ── column roles ───────────────────────────────────────────────────────
#
# COPIES. Keep byte-identical to the originals; tests/v2/unit/api/
# test_spectrum_data.py asserts they match.

#: 反扫列的标记。Nanonis 真机列名是 ``Current [bwd] (A)``。
#: (from skills/builtins/spectrum_assess.py)
_BWD_MARKERS: tuple[str, ...] = ("[bwd]", "bwd", "backward")

_BIAS_PATTERNS: tuple[tuple[str, ...], ...] = (("bias",), ("voltage",), ("v (v)",))
_CURRENT_PATTERNS: tuple[tuple[str, ...], ...] = (("current",), ("i (a)",))
_Z_PATTERNS: tuple[tuple[str, ...], ...] = (("z rel",), ("z (m)",), ("z spectr",))

#: dI/dV 列名的候选,按优先级。
#: (from skills/builtins/tip_spectro_assess.py)
_DIDV_PATTERNS: tuple[tuple[str, ...], ...] = (
    ("li", "demod", "x"),        # "LI Demod 1 X (A)"
    ("lix",),                    # "LIX 1 omega (A)"
    ("demod", "x"),
    ("didv",),
    ("di/dv",),
)

#: Points above which the curve is decimated. uPlot draws a few thousand points
#: without complaint; a 3DS-sized sweep would be megabytes of JSON for pixels
#: that land on top of each other.
_MAX_POINTS = 20000

#: Ceiling on unnamed fallback columns. A file we cannot interpret still shows
#: its data, but a 60-column export must not become 60 lines on one chart.
_MAX_FALLBACK_SERIES = 8

#: Below this the bias column is not sweeping and d/dV is meaningless — dividing
#: by it produces a vertical wall, not a spectrum. Same physical floor
#: ``vision.spectroscopy.resolve_spectrum_kind`` uses to decide I(V) at all.
_MIN_BIAS_SPAN_V = 1e-6


def _is_backward(name: str) -> bool:
    low = str(name).lower()
    return any(m in low for m in _BWD_MARKERS)


def _pick_directional(columns, patterns, *, backward: bool) -> str | None:
    """按子串组合挑一列,**显式**区分正/反扫。

    Copied from ``spectrum_assess``; the reason it exists rather than "first
    column containing the substring" is that forward and backward are both
    called Current, both the same order of magnitude, and picking the wrong one
    by dict order raises no alarm anywhere downstream."""
    for pats in patterns:
        for name in columns:
            low = str(name).lower()
            if all(p in low for p in pats) and _is_backward(name) is backward:
                return name
    return None


def _pick_column(columns, patterns) -> str | None:
    """按子串组合挑一列(全部子串都要出现,不区分大小写)。"""
    low = {name: str(name).lower() for name in columns}
    for pats in patterns:
        for name, lname in low.items():
            if all(p in lname for p in pats):
                return name
    return None


def _clean(values) -> list[float]:
    """ndarray → JSON-safe list. NaN/Inf become ``None``.

    JSON has no NaN; letting one through produces a body the browser's parser
    rejects outright. ``None`` is also what the chart needs to BREAK the line —
    a gap in a spectrum must not be drawn as a straight segment across it
    ("图上空档不能连线")."""
    import math

    out: list[float] = []
    for v in values:
        f = float(v)
        out.append(f if math.isfinite(f) else None)  # type: ignore[arg-type]
    return out


def extract_spectrum(path: str) -> dict:
    """Read one .dat/.txt spectrum into plottable series.

    Returns a dict matching ``api.schemas_records_export.SpectrumDataResponse``:
    ``found / kind / kind_evidence / sweep_name / sweep / series / columns /
    didv_source / n_points / decimated / degraded / detail``.

    Never raises: a file we cannot read comes back ``degraded=True`` with a
    reason, because the caller is an HTTP handler that must not 500."""
    import numpy as np

    out: dict = {
        "path": str(path), "found": False, "kind": "", "kind_evidence": "",
        "sweep_name": "", "sweep": [], "series": [], "columns": [],
        "didv_source": None, "n_points": 0, "decimated": False,
        "degraded": False, "detail": None,
    }
    p = Path(path)
    if not p.exists():
        out["degraded"] = True
        out["detail"] = "file not found"
        return out
    out["found"] = True

    try:
        from mast.data.loaders import load_spectrum_named

        matrix, names = load_spectrum_named(str(p))
        matrix = np.asarray(matrix, dtype=np.float64)
    except Exception as exc:
        logger.info("spectrum read failed for %s: %s", path, exc)
        out["degraded"] = True
        out["detail"] = str(exc)
        return out

    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 2:
        out["degraded"] = True
        out["detail"] = "need at least 2 columns and 2 rows to draw a curve"
        return out
    if not names or len(names) != matrix.shape[1]:
        # Unnamed format (.npy/.3ds) or a name/width mismatch. Placeholders are
        # marked as such: `load_spectrum_named` documents that inventing names
        # here is how an earlier bug came to label every trace "I (A)".
        names = [f"col{i}" for i in range(matrix.shape[1])]
    out["columns"] = list(names)

    cols = {name: matrix[:, i] for i, name in enumerate(names)}

    # Decimate BEFORE building series so every array stays aligned with `sweep`.
    step = 1
    n_rows = int(matrix.shape[0])
    if n_rows > _MAX_POINTS:
        step = -(-n_rows // _MAX_POINTS)      # ceil
        out["decimated"] = True
    take = slice(None, None, step)

    bias_name = _pick_directional(names, _BIAS_PATTERNS, backward=False) or _pick_column(
        names, _BIAS_PATTERNS)
    z_name = _pick_directional(names, _Z_PATTERNS, backward=False) or _pick_column(
        names, _Z_PATTERNS)

    try:
        from mast.vision.spectroscopy import resolve_spectrum_kind

        kind, evidence = resolve_spectrum_kind(
            bias_v=cols[bias_name] if bias_name else None,
            z_m=cols[z_name] if z_name else None,
        )
    except Exception as exc:  # noqa: BLE001 — a curve is still drawable
        kind, evidence = "", f"判据不可用：{exc}"
    out["kind"], out["kind_evidence"] = kind, evidence

    # The x axis is whichever column is actually being swept — the same question
    # `resolve_spectrum_kind` answers, and the reason it does not trust the
    # header ("字段标签会说谎").
    if kind == "iz" and z_name:
        sweep_name = z_name
    elif bias_name:
        sweep_name = bias_name
    else:
        sweep_name = z_name or names[0]
    out["sweep_name"] = sweep_name
    out["sweep"] = _clean(cols[sweep_name][take])
    out["n_points"] = len(out["sweep"])

    series: list[dict] = []
    used: set[str] = {sweep_name}

    for backward in (False, True):
        cur = _pick_directional(names, _CURRENT_PATTERNS, backward=backward)
        if cur and cur not in used:
            series.append({
                "id": "current_bwd" if backward else "current",
                "name": cur, "values": _clean(cols[cur][take]), "source": "file",
            })
            used.add(cur)

    didv_fwd = _pick_column(
        [n for n in names if not _is_backward(n)], _DIDV_PATTERNS)
    didv_bwd = _pick_column([n for n in names if _is_backward(n)], _DIDV_PATTERNS)
    for sid, name in (("didv", didv_fwd), ("didv_bwd", didv_bwd)):
        if name and name not in used:
            series.append({"id": sid, "name": name,
                           "values": _clean(cols[name][take]), "source": "file"})
            used.add(name)
            out["didv_source"] = "lockin"

    # No lock-in channel: differentiate I(V) numerically — but SAY SO. A numeric
    # derivative of a noisy current is not the same measurement as a lock-in
    # trace, and drawing them under one unlabelled name would let the operator
    # read modulation-free noise as spectroscopy.
    if out["didv_source"] is None and kind == "iv" and bias_name:
        cur_name = _pick_directional(names, _CURRENT_PATTERNS, backward=False)
        if cur_name:
            bias = np.asarray(cols[bias_name][take], dtype=np.float64)
            cur = np.asarray(cols[cur_name][take], dtype=np.float64)
            finite = np.isfinite(bias) & np.isfinite(cur)
            span = (float(np.max(bias[finite]) - np.min(bias[finite]))
                    if finite.sum() >= 2 else 0.0)
            if span >= _MIN_BIAS_SPAN_V:
                try:
                    grad = np.gradient(cur, bias)
                    series.append({
                        "id": "didv", "name": f"dI/dV（{cur_name} 的数值微分）",
                        "values": _clean(grad), "source": "numeric",
                    })
                    out["didv_source"] = "numeric"
                except Exception as exc:  # noqa: BLE001
                    out["detail"] = f"数值微分失败：{exc}"
            else:
                out["detail"] = (
                    f"偏压跨度 {span:.3g} V 太小，不做数值微分（没有 lock-in 列）")

    # Whatever is left. A file whose roles we could not name still shows its
    # data — "重排不是过滤" applies to columns exactly as it does to files.
    if not series:
        for i, name in enumerate(names):
            if name in used or len(series) >= _MAX_FALLBACK_SERIES:
                continue
            series.append({"id": f"col{i}", "name": name,
                           "values": _clean(cols[name][take]), "source": "file"})
        if len(names) - 1 > _MAX_FALLBACK_SERIES:
            out["detail"] = (
                f"未能识别通道角色，显示前 {_MAX_FALLBACK_SERIES} 列"
                f"（共 {len(names)} 列，全部列名在 columns 里）")

    out["series"] = series
    return out
