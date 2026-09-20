"""通过临时 SQLite 验证脉冲落点写入、读回与选点避让之间的信息传递。"""
from __future__ import annotations

import sys
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[5] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _name in list(sys.modules):
    if _name == "mast" or _name.startswith("mast."):
        _f = getattr(sys.modules[_name], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_name]

import pytest  # noqa: E402

from mast.core import map_scope  # noqa: E402
from mast.io.map_analysis import (  # noqa: E402
    DAMAGE_KINDS,
    AnalysisConfig,
    build_avoid_circles,
    nearest_clean_from,
)
from mast.logging.storage import ExperimentStorage  # noqa: E402


class _Log:
    """``get_active_log()`` 返回的那个对象,只保留 map_scope 会碰的三个属性。

    属性名照 ``load_markers`` 里的写法取(``_storage`` /
    ``current_experiment_id`` / ``current_sample_id``)—— 名字写错的话
    ``getattr(..., None)`` 会静默给 None,于是 ``load_markers`` 返回
    ``([], 0, False)``,而这一组测试会「通过」得莫名其妙。
    """

    def __init__(self, storage):
        self._storage = storage
        self.current_experiment_id = "E1"
        self.current_sample_id = "S1"


@pytest.fixture
def live_map(tmp_path, monkeypatch):
    """一个真的空地图。返回 ``(storage, log)``。"""
    st = ExperimentStorage(tmp_path / "exp.db")
    log = _Log(st)
    monkeypatch.setattr("mast.logging.experiment_log.get_active_log",
                        lambda: log, raising=True)
    return st, log


def _cfg(**over):
    base = dict(piezo_half_range_m=1.8e-6, pulse_r_m=400e-9,
                tip_shape_r_m=25e-9, frame_size_m=100e-9,
                center_zone_side_m=None, has_xy_coarse_motion=False)
    base.update(over)
    return AnalysisConfig(**base)


# ── 往返 ────────────────────────────────────────────────────────────────────

def test_the_crater_comes_back_out_of_the_map(live_map):
    """A marker written through the public helper must be returned by the map reader."""
    assert map_scope.load_markers()[0] == [], "起点应当是一张空地图"

    row_id = map_scope.record_damage_marker(1e-6, -0.5e-6, kind="pulse",
                                            skill_name="ForgeAuTip")
    assert row_id is not None, "没写进去"

    markers, _epoch, available = map_scope.load_markers()
    assert available is True
    assert len(markers) == 1
    m = markers[0]
    assert m.kind == "pulse"
    assert m.x_m == pytest.approx(1e-6)
    assert m.y_m == pytest.approx(-0.5e-6)


def test_the_crater_actually_blocks_the_next_pulse(live_map):
    """读回来还不够 —— 它得**真的挡住**下一发。

    往返成立、``build_avoid_circles`` 却给不出圈,就等于没标记。
    这两件事分开测,是因为它们会各自坏掉:字段名错了坏第一条,
    ``kind`` 不在 ``DAMAGE_KINDS`` 里坏第二条,而症状一模一样。
    """
    cfg = _cfg()
    before = nearest_clean_from([], cfg, 0.0, 0.0, spot_r_m=400e-9, count=9)
    assert any(s.x_m == 0.0 and s.y_m == 0.0 for s in before), "原点本该可用"

    map_scope.record_damage_marker(0.0, 0.0, kind="pulse", skill_name="ForgeAuTip")
    markers, _e, _a = map_scope.load_markers()

    circles = build_avoid_circles(markers, cfg)
    assert circles, "标记读回来了,却产生不了避让圈 —— 那等于没标记"
    assert circles[0].radius_m == pytest.approx(cfg.pulse_r_m), (
        "避让半径不是 pulse_r_m —— 有人在这条路上塞了一个私有半径")

    after = nearest_clean_from(markers, cfg, 0.0, 0.0, spot_r_m=400e-9, count=9)
    assert not any(s.x_m == 0.0 and s.y_m == 0.0 for s in after), (
        "The marked origin must be excluded from subsequent candidate points.")


def test_the_two_kinds_keep_their_own_radii(live_map):
    """不同标记类别必须分别使用配置的避让半径。
    本测试独立选择两种半径，核验类别到配置键的映射，不表达实验损伤尺寸。"""
    cfg = _cfg()
    map_scope.record_damage_marker(0.0, 0.0, kind="pulse")
    map_scope.record_damage_marker(600e-9, 0.0, kind="tip_shape")
    markers, _e, _a = map_scope.load_markers()

    by_kind = {c.kind: c.radius_m for c in build_avoid_circles(markers, cfg)}
    assert by_kind.get("pulse") == pytest.approx(cfg.pulse_r_m)
    assert by_kind.get("tip_shape") == pytest.approx(cfg.tip_shape_r_m)
    # 半径的真源是 DAMAGE_KINDS,不是这里的字面量。
    assert DAMAGE_KINDS["pulse"] == "pulse_r_m"
    assert DAMAGE_KINDS["tip_shape"] == "tip_shape_r_m"


# ── 失败要说出来 ────────────────────────────────────────────────────────────

def test_an_unknown_kind_is_refused_rather_than_silently_useless(live_map):
    """``kind`` 不在 ``DAMAGE_KINDS`` 里 ⇒ **当场拒绝**,不写一条没有效力的记录。

    写了个寂寞比没写更坏:报文会说「已标记」,而那个圈半径为 None,
    几何上等于不存在 —— 下一发照旧打在同一个地方。
    """
    assert map_scope.record_damage_marker(0.0, 0.0, kind="scorch_mark") is None
    assert map_scope.load_markers()[0] == [], "不该留下一条没有效力的记录"


def test_a_dead_map_costs_the_marker_not_the_run(monkeypatch):
    """地图缺失或存储异常时返回 None，不应把记录失败升级成执行异常。"""
    monkeypatch.setattr("mast.logging.experiment_log.get_active_log",
                        lambda: None, raising=True)
    assert map_scope.record_damage_marker(0.0, 0.0, kind="pulse") is None

    def boom():
        raise RuntimeError("storage down")

    monkeypatch.setattr("mast.logging.experiment_log.get_active_log",
                        boom, raising=True)
    assert map_scope.record_damage_marker(0.0, 0.0, kind="pulse") is None


# ── 调用方那一半 ────────────────────────────────────────────────────────────

def test_mark_dirty_writes_both_places(live_map):
    """``_mark_dirty`` 两处都写:本跑的共享表 **和** 地图。

    两处都要,不是冗余 —— ``nearest_clean_from`` 的自述写着:
    「a marker written moments ago may not have reached the store yet」。
    而那个窗口正好是「连打两发」那段时间。
    """
    from mast.skills.composite._tip_phases import _dirty, _mark_dirty

    class _Ex:
        class progress:
            partial_data: dict = {}

    ex = _Ex()
    ex.progress.partial_data = {}

    _mark_dirty(ex, 1e-6, 2e-6, kind="pulse")
    assert _dirty(ex) == [(1e-6, 2e-6)], "本跑的共享表没记上"
    assert len(map_scope.load_markers()[0]) == 1, "地图上没记上"

    # 幂等:同一个点再来一次,两边都不该多出东西。
    _mark_dirty(ex, 1e-6, 2e-6, kind="pulse")
    assert len(_dirty(ex)) == 1
    assert len(map_scope.load_markers()[0]) == 1, "同一个点写了两条地图标记"


def test_a_failed_marker_is_counted_so_the_report_can_say_so(monkeypatch):
    """写入失败必须累计可报告的计数与原因，不能静默丢失记录。"""
    from mast.skills.composite._tip_phases import _mark_dirty

    monkeypatch.setattr("mast.logging.experiment_log.get_active_log",
                        lambda: None, raising=True)

    class _Ex:
        class progress:
            partial_data: dict = {}

    ex = _Ex()
    ex.progress.partial_data = {}
    _mark_dirty(ex, 0.0, 0.0, kind="pulse")
    assert ex.progress.partial_data.get("map_marker_failures") == 1
    assert ex.progress.partial_data.get("map_marker_failure_why")
