"""2026-07-28 artefact audit — a step edge was published as 109 point defects.

Six imaging reports had been produced and every check was green. Reading the
產物 against a KNOWN ground truth showed two of its numbers meant something other
than what the report said they meant.

The synthetic surface (scratchpad/e2e_loop*.py ``_make_scan``) is exactly:

    lattice  sin(0.55x)·sin(0.55y) × 3e-11      (30 pm amplitude)
    tilt     x·2e-13 + y·1.1e-13                (0.20 / 0.11 pm per px)
    step     y > 150 → 2.4e-10                  (ONE 240 pm monoatomic step)
    noise    σ = 4e-12                          (4 pm)
    ⇒ ZERO defects. None. Not one.

What the report published, and what is actually true:

* "After first-order plane subtraction the RMS roughness is 63.8 pm" and
  "a moderately rough surface (RMS 63.8 pm)". 63.757 pm is the arithmetically
  correct RMS — but the true surface roughness is **15.4 pm**; the other 4× is
  the step. A first-order plane cannot remove a step, so what was called
  roughness is a structural feature.
* "27 bright protrusions and 82 dark spots, a dark:bright ratio of
  approximately 3:1 … consistent with depressions (e.g. vacancies or pits)".
  All 109 lie in rows 123–168 — a band straddling the step at y = 150. They are
  ONE step edge, chopped up by the lattice crossing the ±2σ contour. On a
  defect-free flat surface the same code correctly returns 0/0, and with 6 real
  vacancies + a step it returns 26/87 — burying the 6 it should have found.

Neither tool computed a wrong number; both answered a question whose
precondition had silently failed. So the fix is not a new algorithm — it is
making the tools say when that precondition fails.

Two guards, both measured against ground truth rather than guessed:

  * structure dominance — global σ ÷ median per-tile σ. Flat / gently curved /
    genuinely defective surfaces all sit at 1.00–1.19; one step gives 3.19 in
    ANY orientation, two steps 2.43. Cut at 1.8.
  * band shape — min(row-span, col-span) of the detections. Steps span 41–46 of
    256 in their narrow direction; 6, 20 and 40 scattered vacancies all span
    253–254 in both. Cut at n//4.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest \
        tests/v2/unit/agents/test_step_not_defects_20260728.py -q
"""
from __future__ import annotations

# ── path bootstrap ───────────────────────────────────────────────────────────
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.modules.setdefault("nanonis_spm", MagicMock())


def _find_mastv2_root() -> str:
    p = Path(__file__).resolve()
    while p.parent != p:
        if (p / "MASTv2").is_dir():
            return str(p / "MASTv2")
        p = p.parent
    raise RuntimeError("MASTv2 dir not found")


_MASTV2_ROOT = _find_mastv2_root()
if sys.path and sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _n in list(sys.modules):
    if _n == "mast" or _n.startswith("mast."):
        _f = getattr(sys.modules[_n], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_n]

import numpy as np  # noqa: E402
import pytest  # noqa: E402

N = 256
TRUE_STEP_M = 2.4e-10          # 240 pm
TRUE_LOCAL_ROUGHNESS_PM = 15.4  # lattice(30 pm amp) + noise(4 pm), measured


# ── the ground truth, verbatim from the e2e harness ──────────────────────────

def _surface(*, step: str = "horizontal", n_defects: int = 0,
             seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:N, 0:N]
    im = (np.sin(x * 0.55) * np.sin(y * 0.55)) * 3e-11
    im = im + x * 2e-13 + y * 1.1e-13
    if step == "horizontal":
        im = im + np.where(y > 150, TRUE_STEP_M, 0.0)
    elif step == "vertical":
        im = im + np.where(x > 150, TRUE_STEP_M, 0.0)
    elif step != "none":  # pragma: no cover - guard against a typo in a test
        raise ValueError(step)
    im = im + rng.normal(0, 4e-12, (N, N))
    for _ in range(n_defects):
        cy, cx = rng.integers(15, 240), rng.integers(15, 240)
        im = im - (((y - cy) ** 2 + (x - cx) ** 2) <= 9) * 6e-11
    return im.astype(np.float32)


@pytest.fixture()
def npy(tmp_path):
    def _write(**kw):
        p = tmp_path / f"s_{kw.get('step','horizontal')}_{kw.get('n_defects',0)}.npy"
        np.save(p, _surface(**kw))
        return str(p)
    return _write


# ── the ground truth itself, so a broken generator cannot fake a pass ────────

def test_the_synthetic_surface_really_has_no_defects_and_one_step():
    im = _surface().astype(np.float64)
    y, x = np.mgrid[0:N, 0:N]
    lower_m = y <= 150

    # Fit the tilt on the LOWER terrace only, then extrapolate it across the
    # whole frame. Taking a bare mean difference between terraces would fold the
    # tilt into the answer: the upper terrace sits ~128 rows higher, so
    # 128 × 1.1e-13 = 14 pm of the raw 254 pm difference is tilt, not step.
    A_low = np.column_stack([x[lower_m].ravel(), y[lower_m].ravel(),
                             np.ones(int(lower_m.sum()))])
    c, *_ = np.linalg.lstsq(A_low, im[lower_m].ravel(), rcond=None)
    assert c[0] * 1e12 == pytest.approx(0.20, rel=0.05), "true x tilt 0.20 pm/px"
    assert c[1] * 1e12 == pytest.approx(0.11, rel=0.10), "true y tilt 0.11 pm/px"

    A_all = np.column_stack([x.ravel(), y.ravel(), np.ones(im.size)])
    detilted = im - (A_all @ c).reshape(im.shape)
    assert detilted[~lower_m].mean() == pytest.approx(TRUE_STEP_M, rel=0.02), (
        "the step is 240 pm once the tilt is taken out")
    assert detilted[lower_m].std() * 1e12 == pytest.approx(
        TRUE_LOCAL_ROUGHNESS_PM, rel=0.05), "terrace roughness is 15.4 pm"


# ── detect_defects: 109 phantom defects must not be reported bare ────────────

def test_a_step_edge_is_not_reported_as_a_defect_population(npy):
    """THE REGRESSION. Ground truth: zero defects. The counts themselves are a
    property of the ±2σ contour and are kept for continuity — what must never
    happen again is emitting them with nothing saying they are a step."""
    from mast.agents.data_processing.tools import detect_defects

    out = detect_defects.invoke({"path": npy(step="horizontal")})
    assert "bright protrusions: 27" in out, "the counts themselves are unchanged"
    assert "dark spots: 82" in out
    # …but they no longer stand alone.
    assert "台阶" in out, (
        "a defect-free stepped surface still reports 109 defects with no "
        "indication that they are one step edge — the report published them "
        "as a 3:1 vacancy excess")
    assert "不是点缺陷" in out
    assert "大尺度结构" in out


def test_the_warning_is_direction_agnostic(npy):
    """The step in the audited frame was horizontal; a vertical one is the same
    physics. An earlier draft of this guard only looked at row-span and silently
    missed it."""
    from mast.agents.data_processing.tools import detect_defects

    out = detect_defects.invoke({"path": npy(step="vertical")})
    assert "窄带" in out, "a vertical step edge slipped through the band check"
    assert "大尺度结构" in out


def test_a_genuine_defect_population_is_not_wrongly_flagged(npy):
    """The other side: 40 real vacancies on a flat surface must report cleanly.
    A guard that fires on everything would be worse than no guard."""
    from mast.agents.data_processing.tools import detect_defects

    out = detect_defects.invoke({"path": npy(step="none", n_defects=40)})
    # not exactly 40 — randomly placed disks can overlap and merge into one
    # component. The point is that they ARE found and NOT explained away.
    n_dark = int(out.split("dark spots: ")[1].split(" ")[0])
    assert 30 <= n_dark <= 40, out
    assert "窄带" not in out, "real scattered defects flagged as a band"
    assert "大尺度结构" not in out, "a flat surface flagged as terraced"


def test_a_clean_flat_surface_stays_clean(npy):
    """No step, no defects → nothing detected and nothing warned."""
    from mast.agents.data_processing.tools import detect_defects

    out = detect_defects.invoke({"path": npy(step="none")})
    assert "bright protrusions: 0" in out and "dark spots: 0" in out
    assert "⚠" not in out


# ── plane_subtract: a step is not roughness ──────────────────────────────────

def test_rms_over_a_step_is_not_called_roughness_bare(npy):
    """The report's headline was "a moderately rough surface (RMS 63.8 pm)".
    The real surface roughness is 15.4 pm."""
    from mast.agents.data_processing.tools import plane_subtract

    out = plane_subtract.invoke({"path": npy(step="horizontal"), "order": 1})
    assert "6.376e-11" in out, "the RMS itself is arithmetically right, keep it"
    assert "不是表面粗糙度" in out, (
        "RMS over a stepped surface is still presented as plain roughness")
    # and the honest local number is offered alongside
    assert "局部粗糙度" in out
    # the slope caveat too: the fitted y slope was 13× the true tilt
    assert "斜率" in out


def test_plane_subtract_on_a_flat_surface_is_unchanged(npy):
    from mast.agents.data_processing.tools import plane_subtract

    out = plane_subtract.invoke({"path": npy(step="none"), "order": 1})
    assert "RMS roughness after subtraction" in out
    assert "⚠" not in out, "a flat surface must not be warned about"


# ── the discriminator itself, measured not guessed ───────────────────────────

@pytest.mark.parametrize(
    "kw, should_fire",
    [
        ({"step": "horizontal"}, True),
        ({"step": "vertical"}, True),
        ({"step": "none"}, False),
        ({"step": "none", "n_defects": 6}, False),
        ({"step": "none", "n_defects": 40}, False),
    ],
)
def test_structure_dominance_separates_steps_from_texture(kw, should_fire):
    from mast.agents.data_processing.tools import (
        _STRUCTURE_RATIO,
        _structure_dominance,
    )

    im = _surface(**kw).astype(np.float64)
    yy, xx = np.indices(im.shape)
    A = np.column_stack([xx.ravel(), yy.ravel(), np.ones(im.size)])
    c, *_ = np.linalg.lstsq(A, im.ravel(), rcond=None)
    flat = im - (A @ c).reshape(im.shape)

    _g, local, ratio = _structure_dominance(flat)
    assert (ratio >= _STRUCTURE_RATIO) is should_fire, (
        f"{kw} → ratio {ratio:.2f} (cut {_STRUCTURE_RATIO})")
    if not should_fire:
        # on an unstepped surface the local σ IS the surface roughness
        assert local * 1e12 == pytest.approx(TRUE_LOCAL_ROUGHNESS_PM, rel=0.25)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
