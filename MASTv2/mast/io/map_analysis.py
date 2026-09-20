"""Scan-map analysis — what the recorded markers MEAN for where the tip may go.

The scan map is not decoration. It is the only place the system knows which
patches of surface are already imaged, which have been ruined by tip forming, a
bias pulse or a crash, and therefore where the next scan should go and when the
surface is spent. Those judgements are made HERE, in code, from the recorded
markers — never by asking a language model to look at a picture and guess. The
agent reads the conclusions through tools; the operator reads the same
conclusions through a button next to the map. One computation, two audiences.

Pure and offline: input is the row dicts from ``ExperimentStorage.get_markers``
plus an explicit ``AnalysisConfig``; output is frozen dataclasses. No hardware, no
database, no agent state, no global settings lookups — every threshold arrives as
a parameter so a test can state the world it means to test.

Coordinate frame: Nanonis stage frame, METRES, same as ``exp_map``. Analysis is
confined to ONE coordinate generation (``coord_epoch``): after a lateral coarse
move the old numbers address different surface, so mixing generations would mark
fresh surface as already-scanned.

Two semantics worth stating up front, because they are easy to get backwards:

  * **Already scanned is not unusable.** Re-imaging a good area, or returning to
    take spectra where the surface was clean, is ordinary work. Only DAMAGE
    subtracts from the usable area. ``usable_unscanned_frac`` is the one that
    answers "is there anywhere new left".

  * **The candidate sequence is stateless.** Where the Nth scan should go depends
    only on the configuration, never on history; history only filters out
    candidates already used or ruined. That is what lets an interrupted survey
    resume correctly without anyone having stored a cursor.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

from mast.io.exp_map import MapMarker, epoch_of_row, markers_from_rows

logger = logging.getLogger(__name__)


# ── Configuration ───────────────────────────────────────────────────────────

#: Marker kinds that damage or contaminate the surface, each with the
#: ``AnalysisConfig`` field holding its avoidance radius. ``manual`` is absent on
#: purpose: an operator-placed keep-out carries its own radius in ``meta``.
DAMAGE_KINDS: dict[str, str] = {
    "tip_shape": "tip_shape_r_m",
    "pulse": "pulse_r_m",
    "crash": "crash_r_m",
    "approach": "approach_r_m",
}

#: ``meta`` keys written by the ``mark_area_used`` tool.
META_AVOID_RADIUS = "avoid_radius_m"
META_USED_RADIUS = "used_radius_m"


@dataclass(frozen=True)
class AnalysisConfig:
    """Every threshold the analysis uses, in one explicit place.

    Assembled from ``instrument_profile`` + ``config.SafetyLimits`` by the
    caller (see ``mast.core.runtime``), never read from globals here — that is
    what makes the analysis testable and what keeps a test from having to reach
    into private module state to state its premises."""

    #: Half-width of the reachable piezo square (the analysis universe).
    piezo_half_range_m: float = 1.5e-6
    #: Raster cell for the coverage/damage grids. 25 nm over ±1.5 µm is 120×120.
    grid_cell_m: float = 25e-9

    # Avoidance radii — how much surface each event costs.
    tip_shape_r_m: float = 30e-9
    pulse_r_m: float = 150e-9
    crash_r_m: float = 150e-9
    approach_r_m: float = 200e-9
    #: False only when the operator has confirmed this rig's approach does not
    #: touch the surface; "unknown" resolves to True upstream.
    approach_damages: bool = True

    #: ``center_first`` | ``perimeter_inward`` — already resolved, never "auto".
    strategy: str = "center_first"
    #: Side of the scan frame we are placing. Drives candidate spacing.
    frame_size_m: float = 100e-9
    #: Ring width, and spacing along a ring/spiral, as multiples of the frame.
    #: Slightly over 1 leaves a seam so neighbouring frames do not overlap.
    ring_width_factor: float = 1.2
    point_spacing_factor: float = 1.2
    #: Keep candidates off the very edge of the piezo range.
    edge_margin_frac: float = 0.06
    #: A candidate overlapping the already-scanned area by more than this is
    #: treated as "been there".
    reuse_overlap_frac: float = 0.30

    # Coarse-move advice thresholds.
    min_usable_unscanned_frac: float = 0.15
    #: Radius of the "centre zone" as a fraction of the piezo half-range, used
    #: to notice that centre-first has lost its point.
    center_zone_frac: float = 0.25
    center_zone_blocked_frac: float = 0.50
    #: True when the rig can actually relocate; gates every coarse suggestion.
    has_xy_coarse_motion: bool = True

    #: 有 XY 粗动时,**只用压电范围最中间的这一块**(边长,不是半径)。
    #: ``None`` = 不设限,用满整个可用范围(没有 XY 粗动的仪器就是这样)。
    #:
    #: 这个数与 ``pulse_r_m`` 是一对,合起来产生一个几何后果:
    #: 中心区 500 nm 见方、而一发脉冲的避让半径也是 500 nm ⇒ **打完一发,
    #: 中心区被整个盖住,下一次找干净地方必然找不到,于是被迫粗动换区。**
    #:
    #: 2026-08-13 判据:这样一发之后是**被迫**换地方,而不是
    #: **主动**要求换地方。这是本仓反复要求的那种修法 —— 移除「原地再来一发」这个
    #: 可能性,而不是在别处加一条「请你换地方」的规则去说服它。
    #:
    #: 没有 XY 粗动的仪器走另一条路:不设中心区(整片表面都得用上)、脉冲避让
    #: 收到 200 nm、路径策略用 ``perimeter_inward`` 从外圈往里吃 —— 一发脉冲
    #: 照样会换地方,但不会把整个可扫区域一次废掉。
    center_zone_side_m: "float | None" = None

    #: Caps on what leaves this module for an LLM or a browser.
    max_avoid_circles: int = 400
    max_sts_points: int = 500
    #: Ceiling on the candidate route length. A very small frame inside a large
    #: piezo range implies tens of thousands of positions; enumerating them all
    #: would make an interactive button slow for no benefit, since only the
    #: first acceptable one is acted on. Truncation is reported, never silent
    #: (``route_truncated`` in the result).
    max_candidates: int = 2000

    def radius_for(self, kind: str) -> float | None:
        """Avoidance radius for a damage kind, or None if it does not damage."""
        field_name = DAMAGE_KINDS.get(kind)
        if field_name is None:
            return None
        if kind == "approach" and not self.approach_damages:
            return None
        r = float(getattr(self, field_name, 0.0) or 0.0)
        return r if r > 0 else None

    @property
    def effective_half_range_m(self) -> float:
        """Half-range with the edge margin removed — where candidates may sit.

        ``center_zone_side_m`` 再往里收一层(有 XY 粗动时):落点只许出现在压电
        范围最中间那一小块里。**取两者的较小值**,不是替换 —— 中心区比压电范围
        还大的仪器上,那个数不该反过来把可用区域撑开。
        """
        base = self.piezo_half_range_m * (
            1.0 - max(0.0, min(0.9, self.edge_margin_frac)))
        side = self.center_zone_side_m
        if side is None:
            return base
        try:
            half = float(side) / 2.0
        except (TypeError, ValueError):
            return base
        return min(base, half) if half > 0 else base


# ── Results ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class AvoidCircle:
    """A keep-out disc: somewhere the surface is damaged or contaminated."""
    x_m: float
    y_m: float
    radius_m: float
    kind: str
    label: str = ""


@dataclass(frozen=True)
class NextPosition:
    """A recommended scan-frame centre, with the reasoning that chose it."""
    x_m: float
    y_m: float
    strategy: str
    reason: str
    ring_index: int | None = None
    candidates_left: int = 0


@dataclass(frozen=True)
class CoarseMoveAdvice:
    """Whether to give up on this patch of surface and relocate."""
    suggest: bool
    reasons: list[str] = field(default_factory=list)
    usable_unscanned_frac: float = 0.0
    candidates_left: int = 0


@dataclass(frozen=True)
class MapAnalysisResult:
    current_epoch: int
    markers_total: int
    markers_current_epoch: int
    coverage_frac: float
    usable_frac: float
    usable_unscanned_frac: float
    avoid_circles: list[AvoidCircle]
    damage_counts: dict[str, int]
    next_position: NextPosition | None
    coarse_advice: CoarseMoveAdvice
    sts_points: list[tuple[float, float]]
    sts_total: int
    strategy: str
    survey: dict[str, Any]
    #: True when the candidate route hit ``AnalysisConfig.max_candidates``. The
    #: recommended position is still valid (it is the first acceptable one); only
    #: ``candidates_left`` is then a floor rather than a count. Surfaced so a
    #: bounded search never reads as "the surface is nearly full".
    route_truncated: bool = False
    #: The raster this result's ``coverage_frac`` / ``usable_frac`` were computed
    #: on: cells per axis, and the cell size in metres.
    #:
    #: **Two coverage numbers are only comparable when these match.** The cell is
    #: sized to resolve one scan frame (see ``_grid_cell_m``), so it depends on
    #: ``AnalysisConfig.frame_size_m`` — which ``map_scope.analysis_config()``
    #: takes from the LIVE scan frame, i.e. from whatever the last skill happened
    #: to leave configured. a ``ForgeAuTip`` run left the frame
    #: at 50 nm instead of 100 nm, the grid went 240×240 → 480×480, and coverage
    #: read 2.98% → 2.90% — **down, after imaging**, with not one scan footprint
    #: added or removed. Nothing was miscounted; the two numbers were quantised
    #: differently and were never comparable in the first place.
    #:
    #: Surfaced rather than silently fixed because the frame coupling is
    #: load-bearing: at a 25 nm cell a 100 nm frame is 4 cells across and three
    #: frames read 0.44% instead of 0.33% (``test_coverage_is_accurate_at_the_
    #: real_piezo_scale`` pins that incident). Making the cell stable AND keeping
    #: that accuracy means deriving it from the recorded footprints instead of the
    #: live frame, and threading it through the four other ``_grid_axis(cfg)``
    #: call sites that must stay index-consistent with the raster — a real change,
    #: not a tweak. Until then: compare coverage only within one ``grid_cells``.
    grid_cells: int = 0
    grid_cell_m: float = 0.0
    #: The next few positions the current strategy would walk, first one first —
    #: empty unless ``analyze_map`` was asked for them. ``upcoming[0]`` is the
    #: same point as :attr:`next_position` (both take the first candidate that
    #: passes the same filters); the rest additionally avoid each other, so the
    #: list is a route that could actually be executed, not N copies of one
    #: recommendation. Shorter than requested means the surface really has that
    #: few positions left, which the caller must report honestly.
    upcoming: list[NextPosition] = field(default_factory=list)


# ── Epoch handling ──────────────────────────────────────────────────────────

def epoch_series(rows: Sequence[dict]) -> list[int]:
    """Each row's coordinate generation, derived from the row order.

    The stored column is authoritative (``log_marker`` stamped it inside the
    INSERT transaction). This exists to verify that — and to keep working on a
    database whose rows predate the column, where the count is the definition:
    the number of ``coarse_move`` rows that came before."""
    out: list[int] = []
    seen = 0
    for r in rows or []:
        out.append(seen)
        if (r or {}).get("kind") == "coarse_move":
            seen += 1
    return out


def current_epoch_of(rows: Sequence[dict]) -> int:
    """The generation the LAST row belongs to — i.e. the live one."""
    return sum(1 for r in (rows or []) if (r or {}).get("kind") == "coarse_move")


def filter_epoch(rows: Sequence[dict], epoch: int) -> list[dict]:
    """Rows belonging to one generation, tolerating a missing/NULL column."""
    derived = epoch_series(rows)
    out: list[dict] = []
    for r, d in zip(rows or [], derived):
        stored = (r or {}).get("coord_epoch")
        e = epoch_of_row(r) if stored is not None else d
        if e == epoch:
            out.append(r)
    return out


# ── Avoidance ───────────────────────────────────────────────────────────────

def build_avoid_circles(
    markers: Iterable[MapMarker], cfg: AnalysisConfig
) -> list[AvoidCircle]:
    """Keep-out discs implied by the markers.

    Deliberately does NOT merge or de-duplicate overlapping discs: five plunges
    in one spot really is five events, the count is meaningful to a human reading
    the map, and the geometry tests below are cheap enough not to care."""
    out: list[AvoidCircle] = []
    for m in markers:
        if not m.has_xy:
            continue
        r = cfg.radius_for(m.kind)
        if r is not None:
            out.append(AvoidCircle(float(m.x_m), float(m.y_m), r, m.kind,
                                   m.label or m.kind))
            continue
        # Operator-placed keep-out (mark_area_used with forbidden=True).
        if m.kind == "manual":
            mr = _finite(m.meta.get(META_AVOID_RADIUS)) if isinstance(m.meta, dict) else None
            if mr is not None and mr > 0:
                out.append(AvoidCircle(float(m.x_m), float(m.y_m), float(mr),
                                       "manual_avoid", m.label or "人工标避让区"))
        if len(out) >= cfg.max_avoid_circles:
            logger.debug("avoid-circle cap %d reached", cfg.max_avoid_circles)
            break
    return out


def used_discs(markers: Iterable[MapMarker]) -> list[tuple[float, float, float]]:
    """Discs an operator marked as "been here" without forbidding them."""
    out: list[tuple[float, float, float]] = []
    for m in markers:
        if m.kind != "manual" or not m.has_xy or not isinstance(m.meta, dict):
            continue
        r = _finite(m.meta.get(META_USED_RADIUS))
        if r is not None and r > 0:
            out.append((float(m.x_m), float(m.y_m), float(r)))
    return out


# ── Rasterisation ───────────────────────────────────────────────────────────

#: Hard ceiling on cells per axis. 512² booleans is a quarter-megabyte and fills
#: in milliseconds, so this bounds the work without ever being the binding
#: constraint at realistic scan sizes.
_MAX_GRID_CELLS_PER_AXIS = 512
#: Minimum cells across one scan footprint. Coverage is decided by whether a
#: cell CENTRE falls inside a footprint, so a frame only ~4 cells wide gains or
#: loses a whole row of cells depending on where it happens to sit — a ~25%
#: error in a coverage number the agent acts on. Eight is enough to keep the
#: quantisation error well under the thresholds that trigger decisions.
_MIN_CELLS_PER_FRAME = 8


def _grid_cell_m(cfg: AnalysisConfig) -> float:
    """Raster cell size actually used: fine enough to resolve one scan frame,
    coarse enough to keep the grid bounded."""
    r = float(cfg.piezo_half_range_m)
    cell = float(cfg.grid_cell_m)
    frame = float(cfg.frame_size_m or 0.0)
    if frame > 0:
        cell = min(cell, frame / _MIN_CELLS_PER_FRAME)
    return max(cell, 2.0 * r / _MAX_GRID_CELLS_PER_AXIS)


def _grid_axis(cfg: AnalysisConfig) -> np.ndarray:
    """Cell-centre coordinates along one axis of the analysis grid."""
    r = float(cfg.piezo_half_range_m)
    cell = _grid_cell_m(cfg)
    # round(), not ceil(): 2r/cell is frequently a whole number that floating
    # point renders as 120.00000000000001, and ceil() then adds a spurious cell
    # which shifts every cell centre and skews the areas.
    n = max(2, int(round(2.0 * r / cell)))
    edges = np.linspace(-r, r, n + 1)
    return 0.5 * (edges[:-1] + edges[1:])


def rasterize(
    markers: Iterable[MapMarker], cfg: AnalysisConfig
) -> tuple[np.ndarray, np.ndarray]:
    """``(covered, blocked)`` boolean grids over the piezo square.

    ``covered`` is the union of completed scan footprints (plus operator
    "used" discs); ``blocked`` is the union of avoidance discs. Rotated
    footprints are tested properly rather than approximated by their bounding
    box — a 45°-rotated frame's bounding box is twice its area, and claiming
    coverage we do not have is the one error that makes the survey skip real
    surface. Each footprint is filled inside its own bounding box, so cost
    scales with the area actually covered, not with grid size × marker count."""
    xs = _grid_axis(cfg)
    ys = _grid_axis(cfg)
    covered = np.zeros((ys.size, xs.size), dtype=bool)
    blocked = np.zeros((ys.size, xs.size), dtype=bool)

    marker_list = list(markers)

    for m in marker_list:
        if m.kind != "scan" or m.status != "done" or not m.has_footprint:
            continue
        _fill_rect(covered, xs, ys, float(m.x_m), float(m.y_m),
                   float(m.w_m), float(m.h_m), float(m.angle_deg or 0.0))

    for cx, cy, r in used_discs(marker_list):
        _fill_disc(covered, xs, ys, cx, cy, r)

    for c in build_avoid_circles(marker_list, cfg):
        _fill_disc(blocked, xs, ys, c.x_m, c.y_m, c.radius_m)

    return covered, blocked


def _bbox_slice(axis: np.ndarray, lo: float, hi: float) -> tuple[int, int]:
    """Index range of cell centres within ``[lo, hi]`` (may be empty)."""
    i0 = int(np.searchsorted(axis, lo, side="left"))
    i1 = int(np.searchsorted(axis, hi, side="right"))
    return max(0, i0), min(axis.size, i1)


def _fill_rect(grid: np.ndarray, xs: np.ndarray, ys: np.ndarray,
               cx: float, cy: float, w: float, h: float, angle_deg: float) -> None:
    hw, hh = abs(w) / 2.0, abs(h) / 2.0
    if hw <= 0 or hh <= 0:
        return
    ang = math.radians(angle_deg or 0.0)
    if abs(angle_deg or 0.0) < 0.5:
        ix0, ix1 = _bbox_slice(xs, cx - hw, cx + hw)
        iy0, iy1 = _bbox_slice(ys, cy - hh, cy + hh)
        if ix1 > ix0 and iy1 > iy0:
            grid[iy0:iy1, ix0:ix1] = True
        return
    # Rotated: bound by the circumscribed box, then test each centre in the
    # rectangle's own frame.
    reach = math.hypot(hw, hh)
    ix0, ix1 = _bbox_slice(xs, cx - reach, cx + reach)
    iy0, iy1 = _bbox_slice(ys, cy - reach, cy + reach)
    if ix1 <= ix0 or iy1 <= iy0:
        return
    gx, gy = np.meshgrid(xs[ix0:ix1] - cx, ys[iy0:iy1] - cy)
    ca, sa = math.cos(-ang), math.sin(-ang)
    lx = gx * ca - gy * sa
    ly = gx * sa + gy * ca
    grid[iy0:iy1, ix0:ix1] |= (np.abs(lx) <= hw) & (np.abs(ly) <= hh)


def _fill_disc(grid: np.ndarray, xs: np.ndarray, ys: np.ndarray,
               cx: float, cy: float, r: float) -> None:
    if r <= 0:
        return
    ix0, ix1 = _bbox_slice(xs, cx - r, cx + r)
    iy0, iy1 = _bbox_slice(ys, cy - r, cy + r)
    if ix1 <= ix0 or iy1 <= iy0:
        return
    gx, gy = np.meshgrid(xs[ix0:ix1] - cx, ys[iy0:iy1] - cy)
    grid[iy0:iy1, ix0:ix1] |= (gx * gx + gy * gy) <= (r * r)


def coverage_stats(covered: np.ndarray, blocked: np.ndarray) -> tuple[float, float, float]:
    """``(coverage_frac, usable_frac, usable_unscanned_frac)``.

    Note what ``usable`` does NOT subtract: already-scanned area. Going back to a
    clean, already-imaged spot to take spectra, or re-imaging it at higher
    resolution, is normal work — it is not consumed surface. Only damage removes
    surface from the budget. ``usable_unscanned_frac`` is the survey metric: how
    much of the reachable surface is both intact and never yet imaged.

    **These fractions are quantised by the grid they arrive on, so two of them are
    comparable only when computed on the same grid** — see
    ``MapAnalysisResult.grid_cells``. Differencing two coverage numbers taken at
    different times is exactly the use this does not support today.

    One diagnostic worth knowing: ``coverage_frac + usable_unscanned_frac == 1``
    EXACTLY means ``blocked`` is empty — i.e. the analysis found NO damage at all.
    On a surface that has been pulsed or plunged, that is evidence the damage
    markers never reached the map, not evidence the surface is intact."""
    total = float(covered.size) or 1.0
    usable_mask = ~blocked
    return (float(covered.sum()) / total,
            float(usable_mask.sum()) / total,
            float((usable_mask & ~covered).sum()) / total)


# ── Candidate positions ─────────────────────────────────────────────────────

def candidate_positions(cfg: AnalysisConfig) -> list[tuple[float, float, int]]:
    """The route this rig should follow, as ``(x, y, ring_index)`` in order.

    Depends on nothing but the configuration. That is the design: an
    interrupted survey resumes by walking this same sequence and skipping what
    the record shows is already done, so no cursor has to be stored, kept in
    sync, or recovered after a restart.

    ``center_first`` — a square spiral out from (0, 0). The scan tube creeps
    least near the middle of its range, so on a rig that can coarse-move to
    fresh surface the best image is always the one taken at the centre.

    ``perimeter_inward`` — outermost ring first, working in. On a rig that
    cannot relocate, tip-forming debris eats the surface around wherever you
    have been working; consuming the edge first keeps the largest contiguous
    clean region — and the low-creep centre — available for as long as possible.
    """
    r_eff = cfg.effective_half_range_m
    step = max(float(cfg.frame_size_m) * float(cfg.point_spacing_factor), 1e-12)
    limit = max(1, int(cfg.max_candidates))
    if cfg.strategy == "perimeter_inward":
        return _perimeter_candidates(r_eff, step, cfg, limit)
    return _spiral_candidates(r_eff, step, limit)


def _ring_cells(ring: int) -> Iterable[tuple[int, int]]:
    """Grid cells exactly ``ring`` steps from the origin (Chebyshev ring).

    Generated as four edges rather than by filtering the enclosing square, so
    the whole spiral costs O(number of positions) instead of O(rings³) — with a
    small scan frame in a large piezo range the difference is millions of
    iterations."""
    if ring == 0:
        yield (0, 0)
        return
    for gx in range(-ring, ring + 1):
        yield (gx, -ring)
        yield (gx, ring)
    for gy in range(-ring + 1, ring):
        yield (-ring, gy)
        yield (ring, gy)


def _spiral_candidates(
    r_eff: float, step: float, limit: int
) -> list[tuple[float, float, int]]:
    """Square spiral from the origin outward, ring index = Chebyshev ring."""
    out: list[tuple[float, float, int]] = []
    n_rings = int(r_eff / step) if step > 0 else 0
    for ring in range(0, n_rings + 1):
        for gx, gy in _ring_cells(ring):
            x, y = gx * step, gy * step
            if abs(x) <= r_eff and abs(y) <= r_eff:
                out.append((x, y, ring))
        if len(out) >= limit:
            break
    return out[:limit]


def _perimeter_candidates(
    r_eff: float, step: float, cfg: AnalysisConfig, limit: int
) -> list[tuple[float, float, int]]:
    """Concentric rings, outermost first, points evenly spaced along each."""
    ring_w = max(float(cfg.frame_size_m) * float(cfg.ring_width_factor), 1e-12)
    out: list[tuple[float, float, int]] = []
    ring = 0
    r = r_eff - ring_w / 2.0
    while r > 0 and len(out) < limit:
        if r < ring_w / 2.0:
            out.append((0.0, 0.0, ring))     # innermost disc collapses to centre
            break
        n = max(1, int(math.floor(2.0 * math.pi * r / step)))
        for k in range(n):
            th = 2.0 * math.pi * k / n
            x, y = r * math.cos(th), r * math.sin(th)
            # Rings are circular but the piezo range is square: drop the parts
            # of a ring that fall outside the reachable box.
            if abs(x) <= r_eff and abs(y) <= r_eff:
                out.append((x, y, ring))
        ring += 1
        r -= ring_w
    return out[:limit]


# ── Position selection ──────────────────────────────────────────────────────

def _frame_hits_circle(cx: float, cy: float, half: float, c: AvoidCircle) -> bool:
    """Does an axis-aligned frame of half-side ``half`` reach into the disc?"""
    dx = max(abs(c.x_m - cx) - half, 0.0)
    dy = max(abs(c.y_m - cy) - half, 0.0)
    return (dx * dx + dy * dy) <= (c.radius_m * c.radius_m)


def _frame_overlap_frac(covered: np.ndarray, xs: np.ndarray, ys: np.ndarray,
                        cx: float, cy: float, half: float) -> float:
    ix0, ix1 = _bbox_slice(xs, cx - half, cx + half)
    iy0, iy1 = _bbox_slice(ys, cy - half, cy + half)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    window = covered[iy0:iy1, ix0:ix1]
    return float(window.sum()) / float(window.size or 1)


def pick_next_position(
    markers: Iterable[MapMarker],
    cfg: AnalysisConfig,
    *,
    covered: np.ndarray | None = None,
    circles: list[AvoidCircle] | None = None,
    candidates: Sequence[tuple[float, float, int]] | None = None,
) -> tuple[NextPosition | None, int]:
    """First candidate that is intact, unvisited and reachable.

    Returns ``(position, candidates_left)``; ``(None, 0)`` means this patch of
    surface is spent under the current strategy, which is what turns into a
    coarse-move recommendation."""
    marker_list = list(markers)
    if circles is None:
        circles = build_avoid_circles(marker_list, cfg)
    if covered is None:
        covered, _ = rasterize(marker_list, cfg)
    if candidates is None:
        candidates = candidate_positions(cfg)
    xs = _grid_axis(cfg)
    ys = _grid_axis(cfg)
    half = max(float(cfg.frame_size_m) / 2.0, 0.0)

    chosen: tuple[float, float, int] | None = None
    hit_circles = 0
    n_ok = 0
    for (x, y, ring) in candidates:
        if abs(x) + half > cfg.piezo_half_range_m or abs(y) + half > cfg.piezo_half_range_m:
            continue                      # frame would leave the piezo range
        blocked_by = [c for c in circles if _frame_hits_circle(x, y, half, c)]
        if blocked_by:
            if chosen is None:
                hit_circles += len(blocked_by)
            continue
        if _frame_overlap_frac(covered, xs, ys, x, y, half) >= cfg.reuse_overlap_frac:
            continue
        n_ok += 1
        if chosen is None:
            chosen = (x, y, ring)
    if chosen is None:
        return None, 0

    x, y, ring = chosen
    if cfg.strategy == "perimeter_inward":
        where = f"外圈向内第 {ring + 1} 环"
    else:
        where = "压电中心" if ring == 0 else f"中心螺旋第 {ring} 圈"
    bits = [where, f"({x * 1e9:.0f}, {y * 1e9:.0f}) nm"]
    if hit_circles:
        bits.append(f"跳过 {hit_circles} 个撞上避让区的候选点")
    reason = "；".join(bits)
    return NextPosition(x_m=x, y_m=y, strategy=cfg.strategy, reason=reason,
                        ring_index=ring, candidates_left=max(0, n_ok - 1)), \
        max(0, n_ok - 1)


def pick_next_positions(
    markers: Iterable[MapMarker],
    cfg: AnalysisConfig,
    count: int,
    *,
    covered: np.ndarray | None = None,
    circles: list[AvoidCircle] | None = None,
    candidates: Sequence[tuple[float, float, int]] | None = None,
) -> list[NextPosition]:
    """Up to *count* next positions, in strategy order, none overlapping another.

    多图规划要一次拿一串位置,而 :func:`pick_next_position` 只给第一个。直接连着
    调 N 次也不行:它只拿**历史**覆盖做过滤,不知道这一批里前面几个已经占了哪儿,
    于是同一片表面会被排进计划好几次。

    返回可能少于 ``count``(这片表面在当前策略下就这么多位置了)—— 调用方必须
    如实报告缺口,而不是把少给的那几张悄悄吞掉。
    """
    marker_list = list(markers)
    if circles is None:
        circles = build_avoid_circles(marker_list, cfg)
    if covered is None:
        covered, _ = rasterize(marker_list, cfg)
    if candidates is None:
        candidates = candidate_positions(cfg)

    xs = _grid_axis(cfg)
    ys = _grid_axis(cfg)
    half = max(float(cfg.frame_size_m) / 2.0, 0.0)
    want = max(0, int(count))

    picked: list[NextPosition] = []
    taken: list[tuple[float, float]] = []
    for (x, y, ring) in candidates:
        if len(picked) >= want:
            break
        if abs(x) + half > cfg.piezo_half_range_m or abs(y) + half > cfg.piezo_half_range_m:
            continue
        if any(_frame_hits_circle(x, y, half, c) for c in circles):
            continue
        if _frame_overlap_frac(covered, xs, ys, x, y, half) >= cfg.reuse_overlap_frac:
            continue
        # 与本批已选的帧互斥:边长 ≥ 一个帧宽的间距 = 两帧不相交。
        if any(abs(x - px) < cfg.frame_size_m and abs(y - py) < cfg.frame_size_m
               for px, py in taken):
            continue
        taken.append((x, y))
        where = (f"外圈向内第 {ring + 1} 环" if cfg.strategy == "perimeter_inward"
                 else ("压电中心" if ring == 0 else f"中心螺旋第 {ring} 圈"))
        picked.append(NextPosition(
            x_m=x, y_m=y, strategy=cfg.strategy,
            reason=f"{where}；({x * 1e9:.0f}, {y * 1e9:.0f}) nm",
            ring_index=ring, candidates_left=0))
    return picked


# ── Nearby spot selection (point actions, not scan frames) ──────────────────

@dataclass(frozen=True)
class CleanSpot:
    """A point the tip may be taken to for a **spot action** — a bias pulse or a
    plunge — with how far it is from where the tip stands now."""
    x_m: float
    y_m: float
    distance_m: float


def nearest_clean_from(
    markers: Iterable[MapMarker],
    cfg: AnalysisConfig,
    x_m: float,
    y_m: float,
    *,
    spot_r_m: float,
    count: int = 16,
    circles: list[AvoidCircle] | None = None,
    exclude: Sequence[tuple[float, float]] | None = None,
    max_distance_m: float | None = None,
    frame_m: float | None = None,
) -> list[CleanSpot]:
    """Undamaged spots near ``(x_m, y_m)``, nearest first.

    A different question from :func:`pick_next_position`, and the difference is
    the whole point. That one walks a fixed route from the origin: it answers
    "where should the Nth survey frame go", and after a pulse it would happily
    send the tip back across the piezo range to the centre of the spiral. This
    one answers "I just fired here and must not fire again on the same spot —
    where is the nearest place I may". Every large excursion re-excites piezo
    creep, so on a conditioning run that fires ten times, travel is not a
    cosmetic concern.

    Being scanned is NOT a reason to skip a spot (see the module docstring):
    only damage is. ``exclude`` carries the points used earlier in this run,
    which the caller holds because a marker written moments ago may not have
    reached the store yet — a stale read must never let two pulses land on one
    spot.

    The tip's current position is itself a candidate when it is still clean, and
    comes back first at distance 0. "Move somewhere else after every shot" is
    expressed by the caller adding each fired point to ``exclude``, not by this
    function assuming the tip is always standing on spent surface — that
    assumption would cost a pointless move, and a fresh dose of creep, at the
    start of every run.

    Stateless, like the survey route: hand it the same tip position and the same
    history and it returns the same list, so an interrupted run resumes without
    anyone having stored a cursor.
    """
    marker_list = list(markers)
    if circles is None:
        circles = build_avoid_circles(marker_list, cfg)
    r_spot = max(float(spot_r_m), 0.0)
    want = max(0, int(count))
    if want == 0:
        return []

    # Neighbouring spots must not overlap, so the lattice step is one diameter.
    step = max(2.0 * r_spot, float(cfg.grid_cell_m), 1e-12)

    # 选点使用 effective_half_range_m，使中心区与边距约束真正作用于候选区域。
    # 硬压电范围必须来自有效仪器范围，不能用静态安全配置冒充实际可达范围。
    # 需要完整范围的调用方通过 release_center_zone 显式解除中心区约束。
    # 
    # 中心区大小、避让半径与同点重复预算共同决定是否需要粗动换区；参数需协调配置。
    # 没有可用落点时返回 surface_spent，由外层站点循环决定换区，不能伪造可达坐标。
    # 
    # 边距按要在落点扫描的帧计算，而非按损伤盘半径计算。
    # 损伤盘伸出可达区并不意味着针尖要到区外；frame_m=0 表示该落点无需扫描帧边距。
    # 修正真实压电范围时，也须保持这两个几何量分离，避免合法候选被重复扣除。
    frame_margin = max(float(cfg.frame_size_m if frame_m is None else frame_m),
                       0.0) / 2.0
    reach = max(cfg.effective_half_range_m - frame_margin, 0.0)

    # 中心区必须至少容纳一个所需净空的落点；几何上无解与表面已用完不同。
    # 进针痕迹也参与避让，不能假设新区只有之后产生的脉冲损伤。
    # 约束本身无解时退回有效压电范围并说明，避免反复粗动仍得到同一无解条件。
    # 避让半径和中心区尺寸应协调配置，不能把无解报告成已经耗尽的表面。
    zone_released = False
    if reach < r_spot:
        zone_released = True
        base = cfg.piezo_half_range_m * (
            1.0 - max(0.0, min(0.9, cfg.edge_margin_frac)))
        reach = max(base - frame_margin, 0.0)
        logger.warning(
            "可用区半程 %.0f nm 装不下一个 %.0f nm 的落点 —— 这个区无解,"
            "已退回压电范围 ±%.0f nm。要「一发就换区」请调 avoid_radius_pulse_nm,"
            "不要把中心区收到比避让半径还小。",
            cfg.effective_half_range_m * 1e9, r_spot * 1e9, reach * 1e9)
    excl = [(float(px), float(py)) for px, py in (exclude or [])]

    # 候选格点锚定在可用区中心，使合法落点集合只取决于表面区域。
    # 针尖位置只影响距离排序，不能改变候选集合；否则区域内有效落点可能因针尖
    # 移位而消失，粗动后的压电坐标也不能修复这种锚点错误。
    found: list[CleanSpot] = []
    # 环数上限跟着 reach 收 —— 按硬边界枚举会在中心区之外白跑很多圈。
    # 从区心起算,所以要盖住 reach 一圈就够(再加一环兜住取整)。
    # 环数盖住整个可用区(针尖可能站在区外,所以不能按「离针尖多远」算)。
    max_ring = int(reach / step) + 2
    # 针尖离区心多远 —— 提前退出要用它(见下)。
    d_tip = math.hypot(float(x_m), float(y_m))
    for ring in range(0, max_ring + 1):
        # 按三角不等式，第 r 环候选距针尖至少 r*step-d_tip。
        # 只有该下界超过已收集的第 want 个结果才可提前退出；
        # 区心锚定后的 r*step 不能直接当成到针尖的距离下界。
        if len(found) >= want:
            found.sort(key=lambda s: s.distance_m)
            if ring * step - d_tip > found[want - 1].distance_m:
                break
        for gx, gy in _ring_cells(ring):
            cx, cy = gx * step, gy * step
            if abs(cx) > reach or abs(cy) > reach:
                continue
            d = math.hypot(cx - x_m, cy - y_m)
            if max_distance_m is not None and d > max_distance_m:
                continue
            # Disc-vs-disc, not the frame test used for scans: a pulse or a
            # plunge acts on a round patch, and squaring it off would throw away
            # usable surface on every one of a dozen shots.
            if any(math.hypot(c.x_m - cx, c.y_m - cy) <= c.radius_m + r_spot
                   for c in circles):
                continue
            if any(math.hypot(px - cx, py - cy) < 2.0 * r_spot for px, py in excl):
                continue
            found.append(CleanSpot(cx, cy, d))

    found.sort(key=lambda s: s.distance_m)
    return found[:want]


# ── Coarse-move advice ──────────────────────────────────────────────────────

def coarse_move_advice(
    cfg: AnalysisConfig,
    *,
    next_position: NextPosition | None,
    usable_unscanned_frac: float,
    candidates_left: int,
    blocked: np.ndarray | None = None,
) -> CoarseMoveAdvice:
    """Should we relocate to a fresh patch of surface, and why.

    Reasons are written once, in Chinese, and shown verbatim to both the agent
    and the operator — so that when the two disagree about what to do next, at
    least they are disagreeing about the same sentence."""
    if not cfg.has_xy_coarse_motion:
        return CoarseMoveAdvice(
            suggest=False,
            reasons=["本仪器没有 XY 粗动马达：换区只能靠插拔样品，"
                     "当前压电范围内的表面要省着用。"],
            usable_unscanned_frac=usable_unscanned_frac,
            candidates_left=candidates_left)

    reasons: list[str] = []
    if next_position is None:
        reasons.append("当前策略下已没有可用的候选位置（全部被避让区占据或已扫过）。")
    if usable_unscanned_frac < cfg.min_usable_unscanned_frac:
        reasons.append(
            f"可用且未扫的面积只剩 {usable_unscanned_frac * 100:.1f}%"
            f"（低于 {cfg.min_usable_unscanned_frac * 100:.0f}% 阈值）。")
    if blocked is not None and cfg.strategy == "center_first":
        frac = _center_zone_blocked_frac(blocked, cfg)
        if frac > cfg.center_zone_blocked_frac:
            reasons.append(
                f"中心区已有 {frac * 100:.0f}% 被破坏区覆盖，"
                "「中心优先」策略在这里已失去意义（中心本是压电蠕变最小的地方）。")
    return CoarseMoveAdvice(suggest=bool(reasons), reasons=reasons,
                            usable_unscanned_frac=usable_unscanned_frac,
                            candidates_left=candidates_left)


def _center_zone_blocked_frac(blocked: np.ndarray, cfg: AnalysisConfig) -> float:
    xs = _grid_axis(cfg)
    ys = _grid_axis(cfg)
    r = cfg.piezo_half_range_m * max(0.01, min(1.0, cfg.center_zone_frac))
    gx, gy = np.meshgrid(xs, ys)
    zone = (gx * gx + gy * gy) <= (r * r)
    n = int(zone.sum())
    if n == 0:
        return 0.0
    return float((blocked & zone).sum()) / float(n)


# ── Queries ─────────────────────────────────────────────────────────────────

def sts_coverage(
    markers: Iterable[MapMarker], cfg: AnalysisConfig
) -> tuple[list[tuple[float, float]], int]:
    """``(points, total)`` for spectroscopy in this generation.

    ``total`` is the honest count even when the returned list is truncated, so a
    caller never mistakes a display cap for the real number."""
    pts: list[tuple[float, float]] = []
    total = 0
    for m in markers:
        if not m.has_xy:
            continue
        is_sts = m.kind == "sts" or (
            m.kind == "manual" and "谱" in (m.label or ""))
        if not is_sts:
            continue
        total += 1
        if len(pts) < cfg.max_sts_points:
            pts.append((float(m.x_m), float(m.y_m)))
    return pts, total


def markers_near(
    markers: Iterable[MapMarker], x_m: float, y_m: float, radius_m: float,
    *, limit: int = 20, current_epoch: int | None = None,
) -> list[dict]:
    """What has already happened near a point, nearest first.

    Searches EVERY generation on purpose: "has anyone worked here before" is a
    question about history, and a stale-coordinate hit is still worth seeing.
    Each entry carries its ``coord_epoch`` so the caller can tell whether the
    coordinate still means anything, plus a flag when it does not."""
    hits: list[tuple[float, dict]] = []
    for m in markers:
        if not m.has_xy:
            continue
        d = math.hypot(float(m.x_m) - x_m, float(m.y_m) - y_m)
        if d > radius_m:
            continue
        hits.append((d, {
            "kind": m.kind,
            "label": m.label,
            "skill": m.skill_name,
            "dist_nm": round(d * 1e9, 1),
            "x_nm": round(float(m.x_m) * 1e9, 1),
            "y_nm": round(float(m.y_m) * 1e9, 1),
            "status": m.status,
            "timestamp": m.timestamp,
            "coord_epoch": m.coord_epoch,
            "stale_coords": (current_epoch is not None
                             and m.coord_epoch is not None
                             and m.coord_epoch != current_epoch),
        }))
    hits.sort(key=lambda t: t[0])
    return [h[1] for h in hits[:limit]]


def survey_progress(
    plan_markers: Sequence[MapMarker] | None,
    scanned: Iterable[MapMarker],
    tol_m: float,
) -> dict[str, Any]:
    """Planned-route progress, with an explicit admission of what it cannot say.

    There is no completion percentage here and there should not be: the plan
    overlay CONSUMES a step once the tip reaches it, so the original total is
    genuinely unrecoverable from what remains. Inventing a denominator would
    make the number move for reasons that have nothing to do with progress. The
    survey metric is ``coverage_frac``; this reports the pending route only, plus
    a count of pending steps that a completed scan already sits on (which means
    the overlay's own advance tolerance is not matching reality)."""
    pending = [m for m in (plan_markers or []) if m.has_xy]
    done_xy = [(float(m.x_m), float(m.y_m)) for m in scanned
               if m.has_xy and m.kind == "scan" and m.status == "done"]
    matched = 0
    for p in pending:
        if any(math.hypot(px - float(p.x_m), py - float(p.y_m)) <= tol_m
               for px, py in done_xy):
            matched += 1
    return {"pending_plan_steps": len(pending),
            "pending_already_scanned": matched}


# ── Top level ───────────────────────────────────────────────────────────────

def analyze_map(
    rows: Sequence[dict],
    cfg: AnalysisConfig,
    *,
    plan_markers: Sequence[MapMarker] | None = None,
    current_epoch: int | None = None,
    upcoming_count: int = 0,
) -> MapAnalysisResult:
    """The one entry point. Rows in, conclusions out.

    Accepts the FULL row list and filters generations itself, so the analysis is
    reproducible from a plain database dump; a caller that would rather filter in
    SQL may pass pre-filtered rows, since selecting the live generation twice is
    idempotent.

    ``current_epoch`` — PASS IT IF YOU HAVE IT. Deriving the generation from
    ``rows`` is only correct when ``rows`` is the whole scope, and
    ``ExperimentStorage.get_markers`` truncates to its ``limit`` (2000 by
    default, newest first). On a scope with more markers than that, the count
    inside the window is SMALLER than the real generation — and because
    ``filter_epoch`` trusts the stored ``coord_epoch`` column, the mismatch does
    not degrade to "no rows": it selects a genuinely older generation and
    analyses the WRONG PATCH OF SURFACE, confidently. ``storage.current_epoch()``
    counts the whole scope through the index in microseconds; there is no reason
    to re-derive it here. The derived value stays as the fallback for callers
    holding a plain dump (2026-07-31).

    ``upcoming_count`` — how many positions BEYOND the recommendation to walk
    out, for a caller that wants to draw the route rather than just the next
    step. Zero (the default) costs nothing: agent tools ask for a decision, not
    a route, and every position in the list is tokens they would pay for."""
    rows = list(rows or [])
    epoch = current_epoch_of(rows) if current_epoch is None else int(current_epoch)
    live_rows = filter_epoch(rows, epoch)
    markers = markers_from_rows(live_rows)

    circles = build_avoid_circles(markers, cfg)
    covered, blocked = rasterize(markers, cfg)
    coverage_frac, usable_frac, usable_unscanned = coverage_stats(covered, blocked)

    route = candidate_positions(cfg)
    nxt, left = pick_next_position(markers, cfg, covered=covered, circles=circles,
                                   candidates=route)
    advice = coarse_move_advice(
        cfg, next_position=nxt, usable_unscanned_frac=usable_unscanned,
        candidates_left=left, blocked=blocked)
    # Reuses the circles/grid/route already computed above — asking for the
    # route costs one more walk of the candidate list, not a second analysis.
    upcoming = (pick_next_positions(markers, cfg, int(upcoming_count),
                                    covered=covered, circles=circles,
                                    candidates=route)
                if int(upcoming_count) > 0 else [])

    sts_pts, sts_total = sts_coverage(markers, cfg)
    damage_counts: dict[str, int] = {}
    for c in circles:
        damage_counts[c.kind] = damage_counts.get(c.kind, 0) + 1

    return MapAnalysisResult(
        current_epoch=epoch,
        markers_total=len(rows),
        markers_current_epoch=len(live_rows),
        coverage_frac=coverage_frac,
        usable_frac=usable_frac,
        usable_unscanned_frac=usable_unscanned,
        avoid_circles=circles,
        damage_counts=damage_counts,
        next_position=nxt,
        coarse_advice=advice,
        sts_points=sts_pts,
        sts_total=sts_total,
        strategy=cfg.strategy,
        survey=survey_progress(
            plan_markers, markers,
            tol_m=max(float(cfg.frame_size_m) / 2.0, 1e-12)),
        route_truncated=len(route) >= max(1, int(cfg.max_candidates)),
        grid_cells=int(covered.shape[0]),
        grid_cell_m=_grid_cell_m(cfg),
        upcoming=upcoming,
    )


def _finite(v: Any) -> float | None:
    try:
        if v is None or isinstance(v, bool):
            return None
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


__all__ = [
    "DAMAGE_KINDS",
    "META_AVOID_RADIUS",
    "META_USED_RADIUS",
    "AnalysisConfig",
    "AvoidCircle",
    "NextPosition",
    "CleanSpot",
    "CoarseMoveAdvice",
    "MapAnalysisResult",
    "epoch_series",
    "current_epoch_of",
    "filter_epoch",
    "build_avoid_circles",
    "used_discs",
    "rasterize",
    "coverage_stats",
    "candidate_positions",
    "pick_next_position",
    "pick_next_positions",
    "nearest_clean_from",
    "coarse_move_advice",
    "sts_coverage",
    "markers_near",
    "survey_progress",
    "analyze_map",
]
