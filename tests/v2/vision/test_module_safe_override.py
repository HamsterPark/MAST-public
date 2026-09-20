"""SAFE-mode override at the VisionModule facade.

SAFE's contract is "the tip is fine, keep running experiments". Until 2026-08-01
that was enforced only by a belief block in the agent's prompt while these
methods kept returning "bad" — so the model argued with its own tool results and
a bad verdict still halted the running experiment through the buffer's CRITICAL
path. These tests pin the override *and* its boundaries:

  - the tip VERDICT flips (that is the tip-repair incentive),
  - scan ARTIFACTS and raw MEASUREMENTS do not (those drive re-tune/re-scan, and
    are facts about the image rather than claims about the apex),
  - an unbound holder / SEMI / AUTO changes nothing at all.
"""
from __future__ import annotations

import sys
from pathlib import Path


def _find_mastv2_root() -> str:
    p = Path(__file__).resolve()
    while p.parent != p:
        if (p / "MASTv2").is_dir():
            return str(p / "MASTv2")
        p = p.parent
    raise RuntimeError("MASTv2 dir not found")


_MASTV2_ROOT = _find_mastv2_root()
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _name in list(sys.modules):
    if _name == "mast" or _name.startswith("mast."):
        _f = getattr(sys.modules[_name], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_name]

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from mast.core.operating_mode import bind_mode_source  # noqa: E402
from mast.vision.module import VisionModule  # noqa: E402


@pytest.fixture(autouse=True)
def _unbind_after():
    """Fail-closed cleanup. A leaked SAFE binding would silently rewrite tip
    verdicts in every later test in this process; a leaked mock singleton would
    hand them a backend that always says "bad"."""
    yield
    bind_mode_source(None)
    VisionModule._instance = None


def _safe():
    bind_mode_source(lambda: "safe")


def _mock_vm(monkeypatch):
    """A VisionModule on the mock backend, which returns label='bad' /
    is_usable=False unconditionally (a deliberate fail-safe: no model must never
    report 'good'). That makes it the ideal fixture for the override."""
    monkeypatch.setenv("MAST_VISION_BACKEND", "mock")
    VisionModule._instance = None            # force a fresh backend pick-up
    vm = VisionModule.get()
    return vm


def _noise(seed=0, n=64):
    return np.random.RandomState(seed).randn(n, n).astype(np.float32)


def _lattice(H=192, W=192, period=7.3, seed=0):
    """Rotated incommensurate lattice — a clean, resolvable surface."""
    yy, xx = np.mgrid[0:H, 0:W]
    th = 0.3
    x2 = xx * np.cos(th) + yy * np.sin(th)
    y2 = -xx * np.sin(th) + yy * np.cos(th)
    return (np.sin(x2 * 2 * np.pi / period + 0.7)
            * np.sin(y2 * 2 * np.pi / (period * 1.13) + 1.1)
            + 0.05 * np.random.RandomState(seed).randn(H, W)).astype(np.float32)


# ── assess_tip_coarse: the only producer of a TipStatus ──────────────────────

def test_coarse_bad_becomes_good_in_safe(monkeypatch):
    vm = _mock_vm(monkeypatch)
    img = _noise()
    before = vm.assess_tip_coarse(img)
    assert before.label == "bad" and before.safe_mode_raw is None

    _safe()
    after = vm.assess_tip_coarse(img)
    assert after.label == "good"
    assert after.safe_mode_raw == {"label": "bad", "confidence": before.confidence}


def test_coarse_confidence_stays_above_half_in_safe(monkeypatch):
    """Downstream consumers re-derive good/bad from the score (paper skills use
    0.5, PreScanCheck 0.8). A rewritten 'good' with confidence 0.2 would be read
    straight back as 'bad', so the override must not hand them one."""
    vm = _mock_vm(monkeypatch)
    _safe()
    assert vm.assess_tip_coarse(_noise()).confidence > 0.5


@pytest.mark.parametrize("mode", ["semi", "auto"])
def test_coarse_untouched_outside_safe(monkeypatch, mode):
    vm = _mock_vm(monkeypatch)
    bind_mode_source(lambda: mode)
    res = vm.assess_tip_coarse(_noise())
    assert res.label == "bad"
    assert res.safe_mode_raw is None


def test_coarse_untouched_when_unbound(monkeypatch):
    """The regression guard for the whole feature: no runtime bound → the old
    behaviour, byte for byte."""
    vm = _mock_vm(monkeypatch)
    bind_mode_source(None)
    res = vm.assess_tip_coarse(_noise())
    assert res.label == "bad"
    assert res.safe_mode_raw is None


# ── assess_tip_fine ─────────────────────────────────────────────────────────

def test_fine_unusable_becomes_usable_in_safe(monkeypatch):
    vm = _mock_vm(monkeypatch)
    assert vm.assess_tip_fine(_noise()).is_usable is False
    _safe()
    res = vm.assess_tip_fine(_noise())
    assert res.is_usable is True
    assert res.safe_mode_raw == {"is_usable": False, "label": "unknown"}


# ── assess_tip_quality: the transparent classical verdict ────────────────────

def test_classical_bad_becomes_good_and_reasons_cleared(monkeypatch):
    """Pure noise → the classical verdict is 'bad' with a stated reason. In SAFE
    the reasons must go too: they ARE the argument for repairing the tip, and
    leaving them would put that argument back into the agent's context."""
    vm = _mock_vm(monkeypatch)
    img = _noise(seed=3, n=192)
    before = vm.assess_tip_quality(img)
    assert before.label == "bad" and before.reasons

    _safe()
    after = vm.assess_tip_quality(img)
    assert after.label == "good"
    assert after.reasons == []
    assert after.confidence > 0.5
    assert after.safe_mode_raw["label"] == "bad"
    assert after.safe_mode_raw["reasons"] == list(before.reasons)


def test_classical_measurements_stay_truthful_in_safe(monkeypatch):
    """SAFE suppresses the verdict, not the physics. The sub-signals are
    measurements an operator (or a later analysis) still needs."""
    vm = _mock_vm(monkeypatch)
    img = _noise(seed=5, n=192)
    before = vm.assess_tip_quality(img)
    _safe()
    after = vm.assess_tip_quality(img)
    assert after.fft_sharpness == before.fft_sharpness
    assert after.has_lattice == before.has_lattice
    assert after.is_double == before.is_double
    assert after.tip_changed == before.tip_changed
    assert after.z_noise == before.z_noise


def test_classical_good_verdict_untouched_in_safe(monkeypatch):
    """A genuinely good tip needs no override — and must not grow a bogus
    safe_mode_raw that would read as 'this was overridden'."""
    vm = _mock_vm(monkeypatch)
    img = _lattice()
    if vm.assess_tip_quality(img).label != "good":
        pytest.skip("fixture lattice not classified good on this build")
    _safe()
    after = vm.assess_tip_quality(img)
    assert after.label == "good"
    assert after.safe_mode_raw is None


# ── assess(): the fused verdict ─────────────────────────────────────────────

def test_assess_overall_good_in_safe_when_only_tip_channels_fired(monkeypatch):
    vm = _mock_vm(monkeypatch)
    img = _noise(seed=7, n=192)
    assert vm.assess(img, use_learned=False).overall == "bad"
    _safe()
    res = vm.assess(img, use_learned=False)
    assert res.overall == "good"
    assert res.reasons == []
    assert res.safe_mode_raw is not None            # the audit trail is attached
    assert res.classical.safe_mode_raw["label"] == "bad"


def test_assess_scan_artifacts_still_pull_verdict_down_in_safe(monkeypatch):
    """The boundary that keeps SAFE honest: feedback ringing is a SCAN problem
    (retune the loop / re-scan), not a tip-repair prompt. SAFE must not hide it —
    hiding it would make the agent blind to a fixable acquisition fault."""
    vm = _mock_vm(monkeypatch)
    H = 192
    xx = np.mgrid[0:H, 0:H][1]
    ringing = (1.5 * np.sin(xx * 2 * np.pi / 6)
               + 0.05 * np.random.RandomState(0).randn(H, H)).astype(np.float32)
    _safe()
    res = vm.assess(ringing, use_learned=False)
    assert res.artifacts.oscillation is True
    assert res.overall == "bad"
    assert any("oscillation" in r for r in res.reasons)


def test_assess_iz_iv_tip_probes_suppressed_in_safe(monkeypatch):
    """I(z)/I(V) are tip-quality probes ("blunt/unstable tip"), so they follow
    the same rule as the classical verdict — but their raw sub-results stay
    attached for anyone who asks."""
    vm = _mock_vm(monkeypatch)
    z_nm = np.linspace(0.0, 1.0, 64)
    noisy_iz = np.random.RandomState(0).rand(64) * 1e-9      # not an exponential
    bias = np.linspace(-1.0, 1.0, 64)
    spiky_iv = np.random.RandomState(1).randn(64) * 1e-9     # unstable

    _safe()
    res = vm.assess(_lattice(), iz=(z_nm, noisy_iz), iv=(bias, spiky_iv),
                    use_learned=False)
    assert res.iz is not None and res.iv is not None          # raw probes attached
    assert not any("I(z)" in r or "I(V)" in r for r in res.reasons)
    assert res.overall in ("good", "usable")                 # not dragged to bad


def test_assess_never_reports_bad_without_a_reason(monkeypatch):
    """`has_artifact` is the OR of the detector's flags; the reason list is a
    hand-written mirror of it. When they disagree the verdict is "bad" with an
    empty `reasons` — the least answerable kind of bad news, and in SAFE the one
    remaining way to imply "repair the tip" without saying anything falsifiable.
    Spike-only frames were exactly that gap."""
    vm = _mock_vm(monkeypatch)
    rs = np.random.RandomState(4)
    img = 0.02 * rs.randn(192, 192).astype(np.float32)
    idx = rs.choice(192 * 192, size=int(0.05 * 192 * 192), replace=False)
    img.flat[idx] += 8.0                       # sparse spikes, no ringing/bad rows

    _safe()
    res = vm.assess(img, use_learned=False)
    if res.overall == "bad":
        assert res.reasons, "overall=bad with an empty reason list"
    assert res.artifacts.spike_frac >= 0.0     # the measurement itself is intact


def test_assess_confidence_stays_in_range_in_safe(monkeypatch):
    """The rewritten confidence still has to satisfy the model's own field
    constraint (ge=0, le=1) — model_copy skips validation, so nothing else
    would catch an out-of-range value."""
    vm = _mock_vm(monkeypatch)
    _safe()
    for img in (_noise(seed=1, n=192), _lattice()):
        r = vm.assess(img, use_learned=False)
        assert 0.0 <= r.confidence <= 1.0
        assert 0.0 <= vm.assess_tip_quality(img).confidence <= 1.0
        assert 0.0 <= vm.assess_tip_coarse(img).confidence <= 1.0


def test_assess_reason_order_unchanged_outside_safe(monkeypatch):
    """Splitting reasons into tip/artifact/probe groups for the SAFE branch must
    not reorder them on the normal path — the list is read by humans and quoted
    into agent context, and the historical order is classical → scan artifacts →
    spectroscopic/learned probes."""
    vm = _mock_vm(monkeypatch)
    H = 192
    xx = np.mgrid[0:H, 0:H][1]
    ringing = (1.5 * np.sin(xx * 2 * np.pi / 6)
               + 0.05 * np.random.RandomState(0).randn(H, H)).astype(np.float32)
    z_nm = np.linspace(0.0, 1.0, 64)
    noisy_iz = np.random.RandomState(0).rand(64) * 1e-9      # not an exponential

    bind_mode_source(None)
    reasons = vm.assess(ringing, iz=(z_nm, noisy_iz), use_learned=False).reasons

    osc_i = next(i for i, r in enumerate(reasons) if "oscillation" in r)
    iz_i = next(i for i, r in enumerate(reasons) if "I(z)" in r)
    assert osc_i < iz_i, f"scan artifacts must precede the probes: {reasons}"


@pytest.mark.parametrize("mode", ["semi", "auto"])
def test_assess_untouched_outside_safe(monkeypatch, mode):
    vm = _mock_vm(monkeypatch)
    img = _noise(seed=11, n=192)
    bind_mode_source(None)
    baseline = vm.assess(img, use_learned=False)
    bind_mode_source(lambda: mode)
    res = vm.assess(img, use_learned=False)
    assert res.overall == baseline.overall == "bad"
    assert res.reasons == baseline.reasons
    assert res.safe_mode_raw is None
