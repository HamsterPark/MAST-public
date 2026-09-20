"""Commissioning report: it must not hand back a number that hurts if applied.

The point of this tool is to replace synthetic thresholds with ones measured on
the instrument, so its failure mode is not "no output" — it is "a plausible
number that makes the monitor useless". Both directions are pinned here.
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

import pytest

from mast.monitoring.commission import _rows_from_api, analyse, render
from mast.monitoring.thresholds import MonitorThresholds


def _row(rms, *, scanning=False, skill="", zctrl=True, spike=4.0,
         line=2.0, jump=0.0, rtn=0.0, ts=0.0) -> dict:
    return {"ts": ts, "verdict": "ok", "scanning": scanning, "skill": skill,
            "z_ctrl_on": zctrl,
            "metrics": {"rms_detrended_a": rms, "spike_max_sigma": spike,
                        "line_ratio": line, "jump_rate_hz": jump,
                        "rtn_score": rtn}}


def _quiet(n=60, rms=3e-12, **kw):
    return [_row(rms, ts=float(i), **kw) for i in range(n)]


THRESHOLDS = MonitorThresholds().to_mapping()


def test_a_detector_that_never_fires_gets_no_numeric_suggestion():
    """jump_rate and rtn_score are zero on a healthy tip BY DESIGN. p99 is then
    zero, and zero times any headroom is still zero — a threshold of 0 turns
    every later segment into an alert. Applying that suggestion would be worse
    than leaving the synthetic default."""
    rep = analyse(_quiet(), THRESHOLDS)
    by_knob = {s["knob"]: s for s in rep["suggestions"]}

    for knob in ("cm_jump_rate_warn_hz", "cm_rtn_score_warn"):
        s = by_knob[knob]
        assert s["n"] > 0, "the data was there"
        assert s["suggested"] is None, f"{knob}: suggested a threshold of zero"
        assert s["note"]
        assert s["current"] == THRESHOLDS[knob]

    text = render(rep, {})
    assert "建议            0" not in text
    assert "保留当前值" in text


def test_a_detector_with_a_real_spread_gets_a_suggestion_above_the_baseline():
    rows = [_row(2e-12 + (i % 10) * 1e-13, ts=float(i)) for i in range(60)]
    rep = analyse(rows, THRESHOLDS)
    s = {x["knob"]: x for x in rep["suggestions"]}["cm_rms_warn_a"]
    assert s["suggested"] is not None
    assert s["suggested"] > s["baseline_p99"], "no headroom over the baseline"
    assert s["suggested"] > max(r["metrics"]["rms_detrended_a"] for r in rows)


def test_a_current_threshold_that_would_misfire_is_reported():
    """The reason to run this at all: finding out a shipped default is wrong for
    this instrument."""
    rows = [_row(3e-12, line=50.0, ts=float(i)) for i in range(60)]
    rep = analyse(rows, THRESHOLDS)      # cm_line_ratio_warn default is 10
    s = {x["knob"]: x for x in rep["suggestions"]}["cm_line_ratio_warn"]
    assert s["would_fire_pct"] == 100.0
    assert "当前阈值会在基线上误报" in render(rep, {})


def test_scanning_and_tip_work_are_kept_out_of_the_baseline():
    """A segment taken while scanning, or while a tip-shaping skill held the
    instrument, is a different population. Calibrating across all of them
    produces a threshold that fits none."""
    rows = (_quiet(40, rms=3e-12)
            + [_row(30e-12, scanning=True, ts=100.0 + i) for i in range(30)]
            + [_row(500e-12, skill="TipShape", ts=200.0 + i) for i in range(20)])
    rep = analyse(rows, THRESHOLDS)

    assert rep["baseline_group"] == "quiet"
    assert rep["group_sizes"] == {"quiet": 40, "scanning": 30, "skill": 20,
                                  "no_junction": 0}
    s = {x["knob"]: x for x in rep["suggestions"]}["cm_rms_warn_a"]
    assert s["suggested"] < 10e-12, "the loud groups leaked into the baseline"


def test_the_scanning_fallback_will_not_calibrate_a_scan_suppressed_rule():
    """Scanning-only data must not suggest thresholds for rules suppressed during scanning; eligible rules still receive suggestions."""
    rows = [_row(3e-12, spike=12.0 + (i % 6), line=4.0 + (i % 4) * 0.2,
                 jump=35.0 + (i % 8), scanning=True, ts=float(i))
            for i in range(60)]
    rep = analyse(rows, THRESHOLDS)
    assert rep["baseline_group"] == "scanning"

    by_knob = {s["knob"]: s for s in rep["suggestions"]}
    for knob in ("cm_jump_rate_warn_hz", "cm_spike_sigma_warn"):
        s = by_knob[knob]
        assert s["suggested"] is None, f"{knob}: calibrated off scanning data"
        assert "扫描" in s["note"]
    # ...while a feature that IS judged during a scan still gets its number.
    assert by_knob["cm_line_ratio_warn"]["suggested"] is not None


def test_segments_without_a_junction_are_their_own_group():
    rows = _quiet(40) + [_row(9e-9, zctrl=False, ts=100.0 + i) for i in range(15)]
    rep = analyse(rows, THRESHOLDS)
    assert rep["group_sizes"]["no_junction"] == 15
    assert rep["baseline_group"] == "quiet"


def test_too_little_data_refuses_to_calibrate():
    """Below a usable sample count it must say so, not extrapolate from five
    segments."""
    rep = analyse(_quiet(5), THRESHOLDS)
    assert rep["baseline_group"] == ""
    for s in rep["suggestions"]:
        assert s["suggested"] is None
    assert "阈值建议不可用" in render(rep, {})


def test_render_survives_an_empty_status_and_no_rows():
    text = render(analyse([], THRESHOLDS), {})
    assert "真机标定报告" in text
    assert "阈值建议不可用" in text


def test_render_shows_the_facts_only_the_instrument_can_answer():
    # Independent synthetic status; these values are not an instrument's timebase table.
    status = {"state": "running", "strategy": "osci1t", "fs_hz": 8000.0,
              "rt_freq_hz": 4000.0, "channel_name": "Current (A)",
              "n_buffer": 512, "timebases_s": [1.25e-4, 2.5e-4], "timebase_index": 0,
              "pump_stats": {"fresh": 80, "duplicate": 3, "gap": 2},
              "segments_done": 8, "gaps_total_s": 0.4}
    text = render(analyse(_quiet(), THRESHOLDS), status)
    assert "512" in text
    assert "125µs" in text
    assert "8000 Hz" in text
    assert "Current (A)" in text



def test_missing_timebases_points_at_the_patch():
    """If the table is empty the likely cause is the nanonis_spm binding that
    sends the setter — say so rather than printing a bare dash."""
    text = render(analyse(_quiet(), THRESHOLDS), {"fs_hz": 20000.0})
    assert "nanonis_patch" in text


# ════════════════════════════════════════════════════════════════════════════
# 合成轮询计数的诊断边界
# ════════════════════════════════════════════════════════════════════════════
def _status(fresh: int, duplicate: int, segments_done: int) -> dict:
    return {
        "running": True, "strategy": "osci1t", "fs_hz": 1000.0,
        "rt_freq_hz": 10000.0, "channel_name": "Current (A)", "n_buffer": 100,
        "timebases_s": [0.1], "timebase_index": 0,
        "pump_stats": {"fresh": fresh, "duplicate": duplicate, "gap": 0,
                       "busy": 0, "discontinuity": 0},
        "segments_done": segments_done, "gaps_total_s": 0.0,
    }


def test_near_total_duplicates_are_not_reported_as_normal() -> None:
    """重复很多且长期没有新段落时必须提示判新异常，不能解释为正常提前轮询。"""
    out = render(analyse([], {}), _status(fresh=1, duplicate=4000, segments_done=0))
    assert "判新判不出来" in out, "病态占比下报告仍然只说「正常」"
    assert "t0" in out, "没有指向可查的方向，用户只能去查网络"
    assert "已落盘 0 段" in out, "没有把「零段落盘」这个决定性事实摆出来"


def test_a_healthy_pump_does_not_get_the_warning() -> None:
    """合成健康计数：有持续的新数据且没有重复，不应触发判新异常提示。"""
    out = render(analyse([], {}), _status(fresh=1200, duplicate=0, segments_done=150))
    assert "判新判不出来" not in out


def test_a_small_sample_does_not_trigger_the_warning() -> None:
    """刚启动时样本还太少，占比不可信 —— 不能在头几次轮询就报警。"""
    out = render(analyse([], {}), _status(fresh=0, duplicate=8, segments_done=0))
    assert "判新判不出来" not in out


def test_rows_from_api_verifies_certificates_by_default() -> None:
    """``insecure`` 是安全开关，默认必须是校验。

    即使服务器可能采用自签名证书，也不能将校验默认关闭。
    """
    import inspect

    sig = inspect.signature(_rows_from_api)
    assert sig.parameters["insecure"].default is False


def test_sat_default_matches_thresholds() -> None:
    """本模块抄的那份出厂默认必须和真源一致 —— 漂开了这条警告就再也不响。"""
    from mast.monitoring.commission import _SAT_SHIPPED_DEFAULT
    from mast.monitoring.thresholds import MonitorThresholds

    assert _SAT_SHIPPED_DEFAULT == MonitorThresholds().cm_sat_current_a


def test_an_unrevised_saturation_threshold_is_called_out() -> None:
    """饱和是三条 CRITICAL 之一，而它的出厂值是个猜测（"usual 100 nA preamp"）。

    判据是 ``|I| >= cm_sat_current_a``。阈值高于真实量程时，前放贴轨的读数停在真实
    量程上，**永远够不到阈值** —— 这条 CRITICAL 被静默关掉，而没有任何东西会说。
    测试只检查出厂占位值的提示语义，不复用任何仪器的实测量程。
    """
    from mast.monitoring.commission import _SAT_SHIPPED_DEFAULT

    out = render(analyse([], {"cm_sat_current_a": _SAT_SHIPPED_DEFAULT}), {})
    assert "出厂猜测值" in out
    assert "静默关掉" in out


def test_a_rig_specific_saturation_threshold_is_not_nagged() -> None:
    """按本机前放量程登记过之后就不该再喊 —— 否则用户会学会忽略它。"""
    out = render(analyse([], {"cm_sat_current_a": 2e-8}), {})
    assert "出厂猜测值" not in out
