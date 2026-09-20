"""Alert adjudication: what may escalate, what may not, and the payload contract.

Two classes of assertion here:

* the escalation gate — only three instrument-level rules may reach CRITICAL,
  each needs consecutive confirmation, and a missing feature never fires
  anything;
* the buffer payload — its key set is a contract with two consumers that live
  outside this package (the composite halt hook and the HITL middleware). If a
  key is renamed here, those consumers silently stop reacting, so the keys are
  pinned by test.
"""
from __future__ import annotations

# ── path setup BEFORE any mast.* imports (see tests/v2/conftest.py) ──
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

import numpy as np
import pytest

from mast.monitoring import alerts as A
from mast.monitoring.pump import Segment
from mast.monitoring.thresholds import MonitorThresholds

FS = 20000.0


def _engine(**overrides) -> A.AlertEngine:
    th = MonitorThresholds.from_mapping(overrides)
    return A.AlertEngine(lambda: th)


def _clean() -> dict:
    return {"sat_frac": 0.0, "frozen": 0, "spike_max_sigma": 3.0,
            "max_step_a": 1e-13, "rms_detrended_a": 2e-12, "rtn_score": 0.05,
            "line_ratio": 2.0, "jump_rate_hz": 0.5}


# ── per-segment rules ────────────────────────────────────────────────────────

def test_clean_segment_is_ok():
    assert _engine().evaluate(_clean()).level == "ok"


def test_saturation_is_a_critical_candidate():
    f = _clean() | {"sat_frac": 0.5}
    v = _engine().evaluate(f)
    assert v.level == "critical_candidate" and "saturation" in v.rules


def test_frozen_readout_is_a_critical_candidate():
    v = _engine().evaluate(_clean() | {"frozen": 1})
    assert v.level == "critical_candidate" and "freeze" in v.rules


def test_giant_spike_needs_a_real_step_not_just_a_big_z_score():
    """On a very quiet trace the robust sigma is tiny and ordinary noise scores
    hundreds of sigma — the step size is what makes it physical."""
    tiny_step = _clean() | {"spike_max_sigma": 200.0, "max_step_a": 1e-15,
                            "rms_detrended_a": 1e-12}
    v = _engine().evaluate(tiny_step)
    assert v.level == "warn" and "spike_warn" in v.rules

    real = _clean() | {"spike_max_sigma": 200.0, "max_step_a": 5e-10,
                       "rms_detrended_a": 1e-12}
    v2 = _engine().evaluate(real)
    assert v2.level == "critical_candidate" and "giant_spike" in v2.rules


@pytest.mark.parametrize("feats,rule", [
    ({"rms_detrended_a": 100e-12}, "rms_high"),
    ({"rtn_score": 0.9}, "rtn_bistable"),
    ({"line_ratio": 50.0}, "line_hum"),
    ({"jump_rate_hz": 40.0}, "jump_burst"),
    ({"spike_max_sigma": 12.0}, "spike_warn"),
])
def test_tip_quality_signals_stay_advisory(feats, rule):
    """Noise / RTN / hum / jumps are the interesting numbers, but their
    operating points are not calibrated against real tips yet — none of them
    may halt a scan."""
    v = _engine().evaluate(_clean() | feats)
    assert v.level == "warn"
    assert rule in v.rules
    assert not set(v.rules) & set(A.CRIT_RULES)


# ── scanning context ─────────────────────────────────────────────────────────
#
# Synthetic paired contexts distinguish scan suppression from a clean verdict.

_SCANNING = {"ctx_scanning": True}


def test_jump_burst_is_recorded_but_not_judged_while_scanning():
    """Crossing a step edge IS an abrupt current change. Judging jump_rate
    during a scan judges the sample, not the instrument."""
    v = _engine().evaluate(_clean() | {"jump_rate_hz": 40.0}, _SCANNING)
    assert v.level == "suppressed"
    assert v.rules == []                       # nothing to alert on
    assert v.suppressed_rules == ["jump_burst"]   # ...but it is written down


def test_spike_warn_is_recorded_but_not_judged_while_scanning():
    v = _engine().evaluate(_clean() | {"spike_max_sigma": 40.0,
                                       "max_step_a": 1e-15}, _SCANNING)
    assert v.level == "suppressed"
    assert v.suppressed_rules == ["spike_warn"]


def test_suppressed_is_not_ok():
    """'ok' claims we looked and found nothing wrong. We declined to look."""
    quiet = _engine().evaluate(_clean(), _SCANNING)
    assert quiet.level == "ok"                 # genuinely nothing fired
    jumpy = _engine().evaluate(_clean() | {"jump_rate_hz": 40.0}, _SCANNING)
    assert jumpy.level != "ok"


@pytest.mark.parametrize("ctx", [
    {},                                # no snapshot at all
    {"ctx_scanning": None},            # snapshot present, field unknown
    {"ctx_scanning": False},           # explicitly parked
    {"ctx_scanning": 0},               # ...as SQLite hands it back
])
def test_only_an_explicit_scan_suppresses(ctx):
    """Polarity check. This gate REMOVES judgement, so unknown must mean 'do
    not suppress' — an install without a live InstrumentState keeps the rules
    it has today instead of silently losing two of them."""
    v = _engine().evaluate(_clean() | {"jump_rate_hz": 40.0}, ctx)
    assert v.level == "warn" and "jump_burst" in v.rules
    assert v.suppressed_rules == []


def test_scanning_as_an_sqlite_int_still_suppresses():
    """`1 is True` is False in Python. An `is True` test here would make the
    gate a no-op on every path that round-trips through the store — the exact
    bug that once emptied a calibration group (commission._tri_bool)."""
    v = _engine().evaluate(_clean() | {"jump_rate_hz": 40.0}, {"ctx_scanning": 1})
    assert v.level == "suppressed"


@pytest.mark.parametrize("feats,rule", [
    ({"rms_detrended_a": 100e-12}, "rms_high"),
    ({"line_ratio": 50.0}, "line_hum"),
    ({"rtn_score": 0.9}, "rtn_bistable"),
])
def test_rules_outside_the_scan_set_still_fire_while_scanning(feats, rule):
    """Rules excluded from scan suppression must continue warning on the same synthetic feature values."""
    v = _engine().evaluate(_clean() | feats, _SCANNING)
    assert v.level == "warn" and rule in v.rules


def test_an_unsuppressed_warn_outranks_a_suppressed_one():
    v = _engine().evaluate(
        _clean() | {"jump_rate_hz": 40.0, "line_ratio": 50.0}, _SCANNING)
    assert v.level == "warn"
    assert v.rules == ["line_hum"]
    assert v.suppressed_rules == ["jump_burst"]


@pytest.mark.parametrize("feats,rule", [
    ({"sat_frac": 0.5}, "saturation"),
    ({"frozen": 1}, "freeze"),
    ({"spike_max_sigma": 200.0, "max_step_a": 5e-10,
      "rms_detrended_a": 1e-12}, "giant_spike"),
])
def test_the_three_critical_rules_are_untouched_by_scanning(feats, rule):
    """They are statements about the INSTRUMENT — the preamp is railed, nobody
    is measuring, something discharged — and they hold while scanning exactly
    as they hold at rest."""
    v = _engine().evaluate(_clean() | feats, _SCANNING)
    assert v.level == "critical_candidate" and rule in v.rules


def test_suppressing_spike_warn_does_not_blind_giant_spike():
    """Both rules read spike_max_sigma. The advisory goes quiet during a scan;
    the alarm — which additionally demands a step > 10×RMS — does not."""
    feats = _clean() | {"spike_max_sigma": 200.0, "max_step_a": 5e-10,
                        "rms_detrended_a": 1e-12}
    v = _engine().evaluate(feats, _SCANNING)
    assert "giant_spike" in v.rules
    assert "spike_warn" not in v.suppressed_rules   # crit consumed the sigma


@pytest.mark.parametrize("jump_hz", [30.0, 75.0, 180.0])
def test_synthetic_scanning_rates_raise_no_warning(jump_hz):
    """Independent synthetic rates above the chosen threshold are suppressed during a scan and warned on at rest."""
    eng = _engine(cm_jump_rate_warn_hz=24.0)
    assert eng.evaluate(_clean() | {"jump_rate_hz": jump_hz},
                        _SCANNING).level == "suppressed"
    assert eng.evaluate(_clean() | {"jump_rate_hz": jump_hz},
                        {"ctx_scanning": False}).level == "warn"


def test_a_suppressed_verdict_breaks_a_critical_streak():
    """Same reasoning as a clean segment: a saturation before a scan and one
    after it are not consecutive."""
    eng = _engine(cm_crit_consecutive=3)
    bad = eng.evaluate(_clean() | {"sat_frac": 0.5})
    eng.confirm(bad, now=1000.0)
    eng.confirm(bad, now=1001.0)
    parked = eng.evaluate(_clean() | {"jump_rate_hz": 40.0}, _SCANNING)
    assert parked.level == "suppressed"
    assert eng.confirm(parked, now=1002.0) is None
    assert eng.confirm(bad, now=1003.0) is None         # streak restarted


def test_the_scan_suppressed_set_holds_no_critical_rule():
    """A structural guard: adding a rule to SCAN_SUPPRESSED_RULES must never be
    able to silence something that can halt a scan."""
    assert not set(A.SCAN_SUPPRESSED_RULES) & set(A.CRIT_RULES)
    assert set(A.SCAN_SUPPRESSED_RULES) <= set(A.WARN_RULES)


def test_missing_features_fire_nothing():
    """Fail-closed: a feature that could not be computed is not evidence."""
    v = _engine().evaluate({k: None for k in _clean()})
    assert v.level == "ok" and v.rules == []


def test_nan_features_fire_nothing():
    v = _engine().evaluate(_clean() | {"sat_frac": float("nan"),
                                       "rms_detrended_a": float("nan")})
    assert v.level == "ok"


# ── confirmation ─────────────────────────────────────────────────────────────

def test_critical_needs_n_consecutive_segments():
    eng = _engine(cm_crit_consecutive=3)
    v = eng.evaluate(_clean() | {"sat_frac": 0.5})
    assert eng.confirm(v, now=1000.0) is None
    assert eng.confirm(v, now=1001.0) is None
    assert eng.confirm(v, now=1002.0) == "saturation"


def test_one_clean_segment_resets_the_streak():
    """An intermittent rail touch is an oddity; a sustained one is a crash."""
    eng = _engine(cm_crit_consecutive=3)
    bad = eng.evaluate(_clean() | {"sat_frac": 0.5})
    eng.confirm(bad, now=1000.0)
    eng.confirm(bad, now=1001.0)
    eng.confirm(eng.evaluate(_clean()), now=1002.0)      # clean segment
    assert eng.confirm(bad, now=1003.0) is None          # streak restarted
    assert eng.confirm(bad, now=1004.0) is None
    assert eng.confirm(bad, now=1005.0) == "saturation"


def test_switching_rule_resets_the_streak():
    eng = _engine(cm_crit_consecutive=2)
    sat = eng.evaluate(_clean() | {"sat_frac": 0.5})
    frz = eng.evaluate(_clean() | {"frozen": 1})
    assert eng.confirm(sat, now=1.0) is None
    assert eng.confirm(frz, now=2.0) is None             # different rule
    assert eng.confirm(frz, now=3.0) == "freeze"


def test_cooldown_suppresses_a_repeat():
    eng = _engine(cm_crit_consecutive=1, cm_alert_cooldown_s=120.0)
    v = eng.evaluate(_clean() | {"frozen": 1})
    assert eng.confirm(v, now=1000.0) == "freeze"
    assert eng.confirm(v, now=1050.0) is None
    assert eng.confirm(v, now=1200.0) == "freeze"


def test_warn_debounce_is_per_rule():
    eng = _engine(cm_alert_cooldown_s=100.0)
    assert eng.should_emit_warn("rms_high", now=1000.0) is True
    assert eng.should_emit_warn("rms_high", now=1050.0) is False
    assert eng.should_emit_warn("line_hum", now=1050.0) is True    # independent
    assert eng.should_emit_warn("rms_high", now=1200.0) is True


# ── summaries ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("rule", A.CRIT_RULES + A.WARN_RULES)
def test_every_rule_has_a_chinese_summary_with_numbers(rule):
    text = A.summarize_zh(rule, _clean() | {"sat_frac": 0.4, "rtn_score": 0.9,
                                            "rtn_rate_hz": 12.0, "rtn_gap_a": 2e-11,
                                            "line_ratio": 50.0, "jump_rate_hz": 9.0,
                                            "spike_max_sigma": 40.0,
                                            "max_step_a": 5e-10}, {})
    assert text and "触发" not in text          # not the generic fallback
    assert any("一" <= ch <= "鿿" for ch in text)


# ── buffer payload contract ──────────────────────────────────────────────────

def test_emit_critical_payload_matches_what_the_consumers_read(monkeypatch):
    captured = {}

    class FakeBuf:
        def next_seq(self):
            return 7

        def emit_event(self, ev):
            captured["ev"] = ev

    monkeypatch.setattr("mast.buffer.active.get_active_buffer", lambda: FakeBuf())
    ok = A.emit_critical("saturation", "电流饱和", {"sat_frac": 0.5},
                         "/tmp/ev.png", 42, scan_id="scan-1")
    assert ok is True

    ev = captured["ev"]
    from mast.buffer.schemas import Severity, VisionEventType
    assert ev.kind is VisionEventType.TIP_QUALITY_DROP     # halt hook gates on this
    assert ev.severity is Severity.CRITICAL                # ...and on this
    assert ev.cause_ref == "current_monitor#42"

    p = ev.payload
    # summary_zh: read by make_tip_halt_hook. frame_path: the GUI thumbnail key.
    for key in ("signal", "scan_id", "summary_zh", "frame_path", "features",
                "recommend", "source"):
        assert key in p, f"consumers depend on payload[{key!r}]"
    assert p["summary_zh"] == "电流饱和"
    assert p["frame_path"] == "/tmp/ev.png"
    assert p["source"] == "current_monitor"
    assert isinstance(p["features"], dict)
    assert all(isinstance(v, float) for v in p["features"].values())


def test_emit_critical_without_a_buffer_is_false_not_an_exception(monkeypatch):
    """Standalone API mode has no buffer; the alert still gets recorded."""
    monkeypatch.setattr("mast.buffer.active.get_active_buffer", lambda: None)
    assert A.emit_critical("freeze", "读数冻结", {}, None, 1) is False


def test_emit_critical_survives_a_broken_buffer(monkeypatch):
    class Exploding:
        def next_seq(self):
            raise RuntimeError("boom")

        def emit_event(self, ev):
            pass

    monkeypatch.setattr("mast.buffer.active.get_active_buffer", lambda: Exploding())
    assert A.emit_critical("freeze", "读数冻结", {}, None, 1) is False


# ── evidence ─────────────────────────────────────────────────────────────────

def test_evidence_png_is_written(tmp_path):
    y = 100e-12 + np.random.default_rng(0).normal(0, 2e-12, int(FS))
    seg = Segment(t_start=0, t_end=1, osci_t0=0, fs_hz=FS, runs=[y],
                  n_samples=y.size)
    path = A.render_evidence_png(seg, {"rms_detrended_a": 2e-12}, tmp_path, "saturation")
    assert path is not None
    p = Path(path)
    assert p.is_file() and p.stat().st_size > 1000
    assert "saturation" in p.name
    assert p.read_bytes()[:4] == b"\x89PNG"          # a real PNG, not a stub


def test_evidence_png_leaves_no_partial_file(tmp_path):
    """Rendering runs on a daemon thread. A shutdown mid-write must not leave a
    truncated image at the path the alert row already points at — hence the
    temp-file-then-rename."""
    y = 100e-12 + np.random.default_rng(0).normal(0, 2e-12, int(FS))
    seg = Segment(t_start=0, t_end=1, osci_t0=0, fs_hz=FS, runs=[y],
                  n_samples=y.size)
    A.render_evidence_png(seg, {}, tmp_path, "freeze")
    assert not list(Path(tmp_path).glob("*.part"))


def test_evidence_png_returns_none_for_an_empty_segment(tmp_path):
    seg = Segment(t_start=0, t_end=0, osci_t0=0, fs_hz=FS, runs=[], n_samples=0)
    assert A.render_evidence_png(seg, {}, tmp_path) is None
