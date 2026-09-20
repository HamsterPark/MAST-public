"""Store round-trips, retention and the pin exemption.

The retention tests are the ones that matter: a sweep that deletes a pinned
segment silently destroys training data, and a sweep that never runs fills the
disk. Both failure modes are quiet, so they get explicit coverage.
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

import time

import numpy as np
import pytest

from mast.monitoring import features as F
from mast.monitoring.store import (
    CurrentMonitorStore, StoreQueryFailed, get_store, segment_npy_path,
    set_store_for_test, trace_gap_marks,
)

FS = 20000.0


@pytest.fixture()
def store(tmp_path):
    s = CurrentMonitorStore(tmp_path / "monitor.sqlite", tmp_path)
    set_store_for_test(s)
    yield s
    set_store_for_test(None)
    s.close()


def _write_segment(store: CurrentMonitorStore, *, t_start: float, dur: float = 1.0,
                   amp: float = 1e-12, pinned: bool = False,
                   with_file: bool = True, level: str = "ok",
                   ctx: dict | None = None) -> int:
    n = int(FS * dur)
    rng = np.random.default_rng(int(t_start) % 10000)
    y = 100e-12 + rng.normal(0, amp, n)
    npy_path = None
    nbytes = 0
    if with_file:
        p = segment_npy_path(store.data_dir, t_start, FS)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(p, y.astype(np.float32))
        npy_path = str(p)
        nbytes = p.stat().st_size
    env = F.envelope(y, FS, 100)
    sid = store.add_segment(
        {"t_start": t_start, "t_end": t_start + dur, "fs_hz": FS, "n_samples": n,
         "npy_path": npy_path, "npy_bytes": nbytes, "channel_name": "Current (A)",
         "pinned": pinned, "pin_reason": "test" if pinned else None},
        env.tobytes(), 0.01,
    )
    feats = F.compute_segment_features([y], FS)
    feats["t_start"] = t_start
    store.add_features(sid, feats, ctx or {}, level)
    return sid


# ── round-trip ───────────────────────────────────────────────────────────────

def test_segment_and_features_round_trip(store):
    sid = _write_segment(store, t_start=time.time())
    assert sid is not None
    meta = store.segment_meta(sid)
    assert meta["fs_hz"] == FS
    assert meta["has_file"] is True
    latest = store.latest_feature()
    assert latest["segment_id"] == sid
    assert latest["rms_detrended_a"] is not None


def test_every_feature_column_survives_the_round_trip(store):
    """The store must not silently drop a feature the extractor produces."""
    sid = _write_segment(store, t_start=time.time())
    row = store.latest_feature()
    missing = [c for c in F.FEATURE_COLUMNS if c not in row]
    assert not missing, f"columns not persisted: {missing}"
    assert row["segment_id"] == sid


def test_context_labels_are_stored(store):
    sid = _write_segment(store, t_start=time.time(), ctx={
        "ctx_scanning": True, "ctx_bias_v": -1.5, "ctx_setpoint_a": 100e-12,
        "ctx_z_m": -1.2e-9, "ctx_zctrl_on": True, "ctx_stale": False,
        "ctx_skill": "TipShape",
    })
    row = store.latest_feature()
    assert row["ctx_scanning"] == 1
    assert row["ctx_bias_v"] == pytest.approx(-1.5)
    assert row["ctx_skill"] == "TipShape"
    assert row["segment_id"] == sid


def test_unique_paths_for_same_millisecond(store):
    """Same-millisecond writes must not collide — this project has lost frames
    to a shared path twice already."""
    t = time.time()
    a = segment_npy_path(store.data_dir, t, FS)
    a.parent.mkdir(parents=True, exist_ok=True)
    a.write_bytes(b"x")
    b = segment_npy_path(store.data_dir, t, FS)
    assert a != b


# ── decimation / envelope fallback ───────────────────────────────────────────

def test_read_segment_decimated_preserves_a_spike(store):
    t = time.time()
    n = int(FS)
    y = np.full(n, 100e-12)
    y[12345] = 5e-9
    p = segment_npy_path(store.data_dir, t, FS)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.save(p, y.astype(np.float32))
    sid = store.add_segment(
        {"t_start": t, "t_end": t + 1, "fs_hz": FS, "n_samples": n,
         "npy_path": str(p), "npy_bytes": p.stat().st_size},
        F.envelope(y, FS, 100).tobytes(), 0.01)

    out = store.read_segment_decimated(sid, max_points=1000)
    assert out["source"] == "raw"
    assert out["decimated"] is True
    assert len(out["i_a"]) <= 1000
    assert max(out["i_a"]) == pytest.approx(5e-9, rel=1e-4)   # spike survived


def test_decimation_is_a_noop_below_the_cap(store):
    sid = _write_segment(store, t_start=time.time(), dur=0.01)   # 200 samples
    out = store.read_segment_decimated(sid, max_points=4000)
    assert out["decimated"] is False
    assert len(out["i_a"]) == 200


def test_falls_back_to_envelope_after_the_raw_file_is_swept(store):
    t = time.time() - 100 * 3600
    sid = _write_segment(store, t_start=t)
    store.retention_sweep(keep_hours=1.0, keep_gb=100.0)

    assert store.segment_meta(sid)["has_file"] is False
    out = store.read_segment_decimated(sid, max_points=1000)
    assert out is not None and out["source"] == "envelope"
    assert len(out["i_a"]) > 0                 # still drawable
    assert store.latest_feature()["segment_id"] == sid   # features survived


def test_psd_is_none_without_raw_samples(store):
    t = time.time() - 100 * 3600
    sid = _write_segment(store, t_start=t)
    assert store.segment_psd(sid) is not None
    store.retention_sweep(keep_hours=1.0, keep_gb=100.0)
    assert store.segment_psd(sid) is None       # honest, not a fabricated spectrum


def test_segment_psd_shape_matches_fft_contract(store):
    sid = _write_segment(store, t_start=time.time())
    psd = store.segment_psd(sid)
    for key in ("freqs_hz", "spectrum", "fs_hz", "nyquist_hz", "df_hz", "output"):
        assert key in psd
    assert psd["output"] == "power"


# ── retention ────────────────────────────────────────────────────────────────

def test_age_sweep_drops_old_raw_but_keeps_rows(store):
    old = _write_segment(store, t_start=time.time() - 48 * 3600)
    new = _write_segment(store, t_start=time.time())
    res = store.retention_sweep(keep_hours=24.0, keep_gb=100.0)

    assert res["removed"] == 1
    assert res["freed_bytes"] > 0
    assert store.segment_meta(old)["has_file"] is False
    assert store.segment_meta(new)["has_file"] is True
    assert store.storage_stats()["segments"] == 2      # rows are permanent


def test_pinned_segments_are_exempt_from_the_age_sweep(store):
    pinned = _write_segment(store, t_start=time.time() - 48 * 3600, pinned=True)
    plain = _write_segment(store, t_start=time.time() - 48 * 3600)
    store.retention_sweep(keep_hours=24.0, keep_gb=100.0)

    assert store.segment_meta(pinned)["has_file"] is True
    assert store.segment_meta(plain)["has_file"] is False


def test_size_sweep_drops_oldest_first_and_spares_pinned(store):
    now = time.time()
    oldest = _write_segment(store, t_start=now - 300, pinned=True)
    mid = _write_segment(store, t_start=now - 200)
    newest = _write_segment(store, t_start=now - 100)

    total = store.storage_stats()["store_bytes"]
    # budget for ~2.5 of the 3 segments: exactly one unpinned file must go
    res = store.retention_sweep(keep_hours=1000.0, keep_gb=(total * 5 / 6) / (1 << 30))

    assert res["over_budget"] is False
    assert store.segment_meta(oldest)["has_file"] is True    # pinned survives
    assert store.segment_meta(mid)["has_file"] is False      # oldest unpinned goes
    assert store.segment_meta(newest)["has_file"] is True


def test_pinning_past_the_budget_is_reported_not_silently_endless(store):
    """Pinned bytes count towards the budget but cannot be reclaimed. When they
    alone exceed it, the sweep must say so instead of looking successful while
    the disk keeps growing."""
    now = time.time()
    pinned = _write_segment(store, t_start=now - 200, pinned=True)
    plain = _write_segment(store, t_start=now - 100)
    total = store.storage_stats()["store_bytes"]

    res = store.retention_sweep(keep_hours=1000.0, keep_gb=(total * 0.25) / (1 << 30))

    assert res["over_budget"] is True
    assert res["pinned_bytes"] > 0
    assert store.segment_meta(pinned)["has_file"] is True    # still exempt
    assert store.segment_meta(plain)["has_file"] is False    # everything else went


def test_over_budget_warns_once_not_every_sweep(store, caplog):
    """The condition persists until a human acts on it; repeating the warning
    every ten minutes would bury everything else in the log."""
    import logging

    now = time.time()
    _write_segment(store, t_start=now - 200, pinned=True)
    _write_segment(store, t_start=now - 100)
    tiny_gb = (store.storage_stats()["store_bytes"] * 0.25) / (1 << 30)

    with caplog.at_level(logging.WARNING, logger="mast.monitoring.store"):
        for _ in range(5):
            res = store.retention_sweep(keep_hours=1000.0, keep_gb=tiny_gb)
            assert res["over_budget"] is True
    warnings = [r for r in caplog.records if "retention budget" in r.getMessage()]
    assert len(warnings) == 1, f"warned {len(warnings)} times for one condition"


def test_sweep_is_idempotent(store):
    _write_segment(store, t_start=time.time() - 48 * 3600)
    first = store.retention_sweep(keep_hours=24.0, keep_gb=100.0)
    second = store.retention_sweep(keep_hours=24.0, keep_gb=100.0)
    assert first["removed"] == 1
    assert second["removed"] == 0


# ── pin / label ──────────────────────────────────────────────────────────────

def test_pin_range_covers_overlapping_segments(store):
    now = time.time()
    a = _write_segment(store, t_start=now - 60)
    b = _write_segment(store, t_start=now - 30)
    c = _write_segment(store, t_start=now)

    n = store.pin_range(now - 31, now - 29, "skill:TipShape")
    assert n == 1
    assert store.segment_meta(b)["pinned"] == 1
    assert store.segment_meta(a)["pinned"] == 0
    assert store.segment_meta(c)["pinned"] == 0


def test_pin_range_boundary_touch_still_pins(store):
    now = time.time()
    sid = _write_segment(store, t_start=now, dur=1.0)
    # window that only touches the segment's trailing edge
    assert store.pin_range(now + 1.0, now + 2.0, "edge") == 1
    assert store.segment_meta(sid)["pinned"] == 1


def test_labels_round_trip_and_filter(store):
    now = time.time()
    good = _write_segment(store, t_start=now - 10)
    bad = _write_segment(store, t_start=now)
    store.add_label(t_start=now - 10, t_end=now - 9, label="good", segment_id=good)
    store.add_label(t_start=now, t_end=now + 1, label="bad", segment_id=bad,
                    note="tip crashed")

    res = store.segments_query(label="bad")
    assert [r["id"] for r in res["segments"]] == [bad]
    assert res["segments"][0]["label_note"] == "tip crashed"
    assert res["labeled_count"] == 2

    unl = store.segments_query(label="unlabeled")
    assert unl["segments"] == []


def test_clear_labels_only_touches_human_source(store):
    now = time.time()
    sid = _write_segment(store, t_start=now)
    store.add_label(t_start=now, t_end=now + 1, label="good", segment_id=sid)
    store.add_label(t_start=now, t_end=now + 1, label="bad", segment_id=sid,
                    source="weak")
    store.clear_labels(sid, source="human")

    remaining = store.labels_query()
    assert [r["source"] for r in remaining] == ["weak"]


# ── queries ──────────────────────────────────────────────────────────────────

def test_features_query_range_and_pagination(store):
    now = time.time()
    for i in range(10):
        _write_segment(store, t_start=now + i)
    rows, total, thinned = store.features_query(since=now + 2, until=now + 7)
    assert total == 5 and len(rows) == 5 and thinned is False

    page, total2, _ = store.features_query(since=now, limit=3, offset=3)
    assert total2 == 10 and len(page) == 3
    assert page[0]["t_start"] == pytest.approx(now + 3)


def test_thinning_keeps_the_worst_verdict_in_each_bucket(store):
    """Downsampling a chart must never hide an alert that fired."""
    now = time.time()
    for i in range(20):
        _write_segment(store, t_start=now + i,
                       level="critical" if i == 7 else "ok")
    rows, total, thinned = store.features_query(since=now, max_points=5)
    assert thinned is True and total == 20
    assert any(r["alert_level"] == "critical" for r in rows)


def test_live_tail_stitches_envelopes(store):
    now = time.time()
    for i in range(3):
        _write_segment(store, t_start=now - 3 + i)
    out = store.live_tail(window_s=60, max_points=1000)
    assert out["n_segments"] == 3
    assert len(out["t_s"]) == len(out["i_min_a"]) == len(out["i_max_a"]) > 0
    assert all(lo <= hi for lo, hi in zip(out["i_min_a"], out["i_max_a"]))


def test_live_tail_respects_max_points(store):
    now = time.time()
    for i in range(20):
        _write_segment(store, t_start=now - 20 + i)
    out = store.live_tail(window_s=600, max_points=50)
    assert len(out["t_s"]) <= 50


# ── 采集中断：曲线必须断开，不能连过去 ───────────────────────────────────────
#
# 反馈：「辅助通道在数据中断的时候不会自动正确显示中断，而是强行连线。」
# 同一个毛病在实时电流这张图上更隐蔽，因为它画的是一条带子 —— 一片连续的色块比
# 一条直线更像「这段时间在测量」。
#
# 判据的**两条腿各自被一条测试钉住**，而且每一条都是在杀掉一个更简单的写法：
# 只按段时长判 → 低占空比配置上每个段边界都断；只按典型空档判 → 窗口里只有两段时
# 那个空档自己成了「典型」，最该断的地方一处都不断。

def test_live_tail_breaks_the_band_at_an_acquisition_gap(store):
    now = time.time()
    for i in range(3):
        _write_segment(store, t_start=now - 200 + i)
    for i in range(3):                       # 60 秒什么都没采
        _write_segment(store, t_start=now - 137 + i)

    out = store.live_tail(window_s=600, max_points=5000)
    assert out["n_gaps"] == 1
    # 断点在两条边缘上都要有，否则线断了、中间那片带子照样糊过去
    # （uPlot 用两条边缘各自的 gap clip 去裁 band 填充）。
    holes = [i for i, v in enumerate(out["i_min_a"]) if v is None]
    assert len(holes) == 1
    assert out["i_max_a"][holes[0]] is None
    # x 轴仍严格递增 —— 插错位置的话 uPlot 会画出一条往回走的线。
    assert all(b > a for a, b in zip(out["t_s"], out["t_s"][1:]))
    # 断点落在空档的**左边缘**（最后一段结束的那一刻），不在数据上。
    # 空洞就该从数据停下来的地方开始，而不是从空档正中开始。
    assert out["t_s"][holes[0]] == pytest.approx(now - 197)
    # last_ts 是最后一个**真实**读数：页面拿它算「停更多久」，
    # 它变成 None 的话「停更」这件事本身就没了。
    assert out["last_ts"] is not None
    assert out["last_ts"] == pytest.approx(out["t_s"][-1])


def test_live_tail_does_not_break_on_the_normal_inter_segment_pause(store):
    """段间那 0.31 s 是特征提取，是设计如此 —— 不是中断。

    这条测试杀掉「相邻点距超过 N 倍中位数就断」那个前端写法：段**内部**的点距是
    包络步长（这里 0.01 s），按它判的话每一个段边界都会被划成断点，一条好好的
    曲线画成上千道虚线。
    """
    now = time.time()
    for i in range(8):
        _write_segment(store, t_start=now - 60 + i * 1.31)
    out = store.live_tail(window_s=600, max_points=5000)
    assert out["n_gaps"] == 0
    assert all(v is not None for v in out["i_min_a"])


def test_trace_gap_marks_sees_a_hole_between_just_two_segments():
    """窗口里只有两段、中间隔了六小时 —— 最该断的场合。

    这条杀掉「阈值 = 典型空档 × 3」那个只有一条腿的写法：一个样本的中位数就是它
    自己，六小时会被当成「典型」，于是一处断点都划不出来。
    """
    assert trace_gap_marks([(0.0, 1.0), (21600.0, 21601.0)]) == [1.0]


def test_trace_gap_marks_tolerates_a_rig_whose_pause_exceeds_its_segment():
    """``cm_segment_s`` 可以调到 0.2 s，而提取那 0.31 s 的停顿不跟着变。

    这条杀掉「阈值 = 一整段时长」那个只有另一条腿的写法：正常停顿比整段还长，
    于是**每一个**段边界都成了「中断」。
    """
    spans = [(i * 0.51, i * 0.51 + 0.2) for i in range(12)]
    assert trace_gap_marks(spans) == []
    # 同一台机器上真出现 60 s 停摆时照样要报。
    spans.append((spans[-1][1] + 60.0, spans[-1][1] + 60.2))
    assert trace_gap_marks(spans) == [pytest.approx(spans[-2][1])]


def test_trace_gap_marks_says_nothing_when_it_cannot_tell():
    """判不出来就一处不划 —— 凭空一道断点会让人去查一个没坏的东西。"""
    assert trace_gap_marks([]) == []
    assert trace_gap_marks([(0.0, 1.0)]) == []
    assert trace_gap_marks([(0.0, 0.0), (5.0, 5.0)]) == []   # 零长度段，判不了


def test_alerts_round_trip_and_evidence(store, tmp_path):
    png = tmp_path / "ev.png"
    png.write_bytes(b"\x89PNG fake")
    aid = store.add_alert(ts=time.time(), level="critical", rule="saturation",
                          summary_zh="电流持续饱和", evidence_png=str(png),
                          features={"sat_frac": 0.4}, emitted_buffer=True)
    rows, total = store.alerts_query()
    assert total == 1
    assert rows[0]["rule"] == "saturation"
    assert rows[0]["evidence_available"] == 1
    assert rows[0]["emitted_buffer"] == 1
    assert store.alert_evidence_png(aid) == b"\x89PNG fake"


def test_alert_evidence_missing_file_is_none_not_a_crash(store):
    aid = store.add_alert(ts=time.time(), level="warn", rule="rms_high",
                          summary_zh="噪声偏高",
                          evidence_png=str(store.evidence_dir / "gone.png"))
    assert store.alert_evidence_png(aid) is None


# ── fail-safe ────────────────────────────────────────────────────────────────

def test_writes_swallow_errors_after_close(store):
    """Acquisition must survive a broken store, not crash with it."""
    store.close()
    assert store.add_segment({"t_start": 1.0, "t_end": 2.0, "fs_hz": FS,
                              "n_samples": 10}) is None
    store.add_features(1, {"t_start": 1.0, "fs_hz": FS})       # no raise
    assert store.add_alert(ts=1.0, level="warn", rule="x", summary_zh="y") is None
    assert store.add_label(t_start=1.0, t_end=2.0, label="good") is None
    assert store.set_pin(1, True) is False
    assert store.pin_range(1.0, 2.0, "r") == 0
    assert store.latest_feature() is None
    # ⚠️ ``storage_stats`` 曾经在这里被断言成 ``["segments"] == 0`` —— 一个**读**
    # 混进了一条关于**写**的豁免。写吞异常是为了不让采集线程跟着库一起倒；
    # 而一个关着的库回「0 段、0 字节」，说的是一句关于磁盘的正面断言。
    # 现在它抛,见 ``test_reads_do_not_fold_a_failure_into_a_count``。


def test_reads_on_empty_store_are_empty_not_errors(store):
    """空库 ≠ 坏库。这里每一个 0 都是**答得上来**的 0。"""
    assert store.latest_feature() is None
    assert store.features_query() == ([], 0, False)
    assert store.segments_query()["segments"] == []
    assert store.alerts_query() == ([], 0)
    assert store.labels_query() == []
    assert store.live_tail()["n_segments"] == 0
    assert store.storage_stats()["segments"] == 0
    assert store.read_segment_decimated(999) is None
    assert store.segment_psd(999) is None


# ── 读失败不许折成一个计数 ────────────────────────────────────────────────
#
# 「读不到」被折叠成一个具体的值，是本仓记账最多的那族缺陷。写的豁免(上面那条
# 测试)不适用于读：一个 ``([], 0)`` 同时是「查过了，零条」和「我根本没查成」，
# 而这两句话对「要不要接着扫一整夜」给的是相反的答案。
#
# 触发方式是关掉库 —— 这正是真机上会发生的那一类失败(库被换掉/盘掉了/连接坏了)。

@pytest.mark.parametrize("call,what", [
    (lambda s: s.alerts_query(), "alerts_query"),
    (lambda s: s.labels_query(), "labels_query"),
    (lambda s: s.aux_query(), "aux_query"),
    (lambda s: s.features_query(), "features_query"),
    (lambda s: s.segments_query(), "segments_query"),
    (lambda s: s.live_tail(), "live_tail"),
    (lambda s: s.storage_stats(), "storage_stats"),
])
def test_reads_do_not_fold_a_failure_into_a_count(store, call, what):
    store.close()
    with pytest.raises(StoreQueryFailed) as exc:
        call(store)
    assert what in str(exc.value), "错误里要说是哪一个查询炸了"
    assert exc.value.__cause__ is not None, "原因要留着 —— 排障靠的是它"


@pytest.mark.parametrize("call", [
    lambda s: s.latest_aux(),
    lambda s: s.latest_feature(),
    lambda s: s.feature_row(1),
    lambda s: s.segment_meta(1),
    lambda s: s.alert_evidence_png(1),
    lambda s: s.read_segment_decimated(1),
])
def test_reads_that_answer_with_none_stay_that_way(store, call):
    """⚠️ 这一条钉的是**没有一刀切**。

    回 ``None`` 的那些读不是同一个问题:``None`` 不是正面断言,调用方一律
    ``or {}``。把它们一起改成抛,只会让一堆「本来就没有这一行」的正常情况开始
    抛异常 —— 而那才是真的噪声。
    """
    store.close()
    assert call(store) is None


def test_delivery_queries_still_treat_unreadable_as_none(store):
    """``undelivered_alerts`` / ``critical_alerts_since`` **刻意**保持吞。

    两个调用方各自写下过理由:``forge_au_tip._critical_since`` 说得最清楚 ——
    它是**额外**加的一道网,不是唯一一道(贴轨/冻结照样落库、进面板、经
    ``AlertDeliveryMiddleware`` 到 agent 眼前),让一个可选子系统读不到就卡死
    整条修针流程,是拿一件小事去换一件大事。
    **这条测试变红 = 有人顺手一刀切了**,请先去读那两个调用方的注释。
    """
    store.close()
    assert store.undelivered_alerts(0.0) == []
    assert store.critical_alerts_since(0.0) == []


def test_singleton_seam(tmp_path):
    s = CurrentMonitorStore(tmp_path / "x.sqlite", tmp_path)
    set_store_for_test(s)
    assert get_store() is s
    set_store_for_test(None)
    s.close()
