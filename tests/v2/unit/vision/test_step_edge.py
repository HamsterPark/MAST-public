"""``locate_step_edge``: where the step in a topography frame runs.

The number that matters is the direction. A line of spectra is laid out along the normal to
this edge and their distances are measured against it, so a few degrees of error becomes a
scale error on every distance — and a wrong edge entirely makes every spectrum land somewhere
nobody chose, while each one still "succeeds".
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mast.vision.step_edge import locate_step_edge  # noqa: E402

STEP_M = 208e-12                       # one Cu(111) layer


def frame(angle_deg: float, *, n: int = 256, step_m: float = STEP_M, noise_m: float = 5e-12,
          n_steps: int = 1, tilt_m: float = 0.0, seed: int = 0) -> np.ndarray:
    """A staircase of ``n_steps`` edges running at ``angle_deg`` in image coordinates."""
    i, j = np.mgrid[0:n, 0:n].astype(float)
    th = math.radians(angle_deg)
    d = -(j - n / 2) * math.sin(th) + (i - n / 2) * math.cos(th)
    pitch = n / (n_steps + 1)
    z = step_m * np.floor(d / pitch + 0.5)
    if tilt_m:
        z = z + tilt_m * (j / n)
    return z + np.random.default_rng(seed).normal(0.0, noise_m, (n, n))


def angle_error(got: float | None, want: float) -> float:
    assert got is not None
    e = abs(got - want) % 180.0
    return min(e, 180.0 - e)


@pytest.mark.parametrize("angle", [0.0, 20.0, 45.0, 70.0, 110.0, 155.0])
def test_it_finds_a_single_step_at_any_angle(angle):
    res = locate_step_edge(frame(angle))
    assert res.verdict == "step_edge", res.reasons
    assert angle_error(res.angle_deg, angle) < 2.0
    assert res.step_height_m == pytest.approx(STEP_M, rel=0.10)


def test_the_scan_frame_angle_is_the_mirror_of_the_image_angle():
    """The reader puts row 0 at the high-y edge, so the array's slow axis runs along −y and the
    two conventions differ by a sign. Both are reported precisely so nobody has to remember."""
    res = locate_step_edge(frame(35.0))
    assert res.verdict == "step_edge"
    assert res.angle_scan_deg == pytest.approx((-res.angle_deg) % 180.0, abs=1e-9)


def test_a_staircase_gives_one_of_its_edges_not_an_average_of_them():
    """Several parallel steps in one frame is the normal case on Cu(111) at 60 nm. Fitting a
    line through all of their pixels at once would give the right direction and a point in
    between them, which is on no edge at all."""
    res = locate_step_edge(frame(30.0, n_steps=4))
    assert res.verdict == "step_edge", res.reasons
    assert angle_error(res.angle_deg, 30.0) < 3.0
    # the point it returns sits on one edge: the frame is a staircase of four, and the pixel
    # count of the chosen edge is about one edge's worth, not four
    assert res.n_edge_px < 4 * 256


def test_a_tilt_is_not_a_step():
    """A ramp has a height difference across the frame too. Telling them apart is the point:
    a plane fitted to a real staircase absorbs the staircase, so the distinction cannot be
    made on heights, only on how sudden the change is."""
    n = 256
    j = np.mgrid[0:n, 0:n][1].astype(float)
    z = 2e-9 * j / n + np.random.default_rng(1).normal(0.0, 5e-12, (n, n))
    res = locate_step_edge(z)
    assert res.verdict == "no_step", (res.verdict, res.angle_deg)


def test_a_flat_terrace_has_no_edge():
    z = np.random.default_rng(2).normal(0.0, 5e-12, (256, 256))
    assert locate_step_edge(z).verdict == "no_step"


def test_a_step_on_a_ramp_is_still_found():
    """Real frames arrive tilted; the sample plane is not the measurement."""
    res = locate_step_edge(frame(50.0, tilt_m=1e-9))
    assert res.verdict == "step_edge", res.reasons
    assert angle_error(res.angle_deg, 50.0) < 3.0


def test_a_curved_island_gives_a_straight_piece_of_its_own_boundary():
    """An island edge is a circle, and no single line describes all of it. What comes back is
    one locally straight arc — the point on the boundary and the tangent there — which is what
    a normal has to be built from. What must NOT come back is a line through the middle of the
    island, fitted to two opposite arcs at once."""
    n = 256
    i, j = np.mgrid[0:n, 0:n].astype(float)
    radius = n / 3
    z = np.where(np.hypot(i - n / 2, j - n / 2) < radius, STEP_M, 0.0)
    z = z + np.random.default_rng(3).normal(0.0, 5e-12, (n, n))
    res = locate_step_edge(z)
    assert res.verdict == "step_edge", res.reasons
    # the point sits on the circle, not inside it
    assert math.hypot(res.y_px - n / 2, res.x_px - n / 2) == pytest.approx(radius, rel=0.06)
    # and the direction is the tangent there, which is perpendicular to the radius
    radial = math.degrees(math.atan2(res.y_px - n / 2, res.x_px - n / 2)) % 180.0
    assert angle_error(res.angle_deg, (radial + 90.0) % 180.0) < 12.0
    assert res.straightness < 0.02


def test_a_tiny_or_empty_frame_is_undecidable_not_an_exception():
    assert locate_step_edge(np.zeros((8, 8))).verdict == "undecidable"
    assert locate_step_edge(np.full((64, 64), np.nan)).verdict == "undecidable"
