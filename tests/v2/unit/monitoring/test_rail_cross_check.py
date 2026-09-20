"""观测到的长时间恒值平台应与配置满量程交叉核对。

railed_frac 直接来自相同样本占比，不依赖待核查的饱和阈值。
配置过大而 sat_frac 为零时，这一独立证据仍应能提示量程不一致。"""
from __future__ import annotations

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

import logging  # noqa: E402

# 源码级断言一律走它,不用 ``inspect.getsource``(2026-08-15)——
# 后者按 import 那一刻的行号切当前文件,别人同时在改就返回错位切片:
# ``in`` 那半给假红(吵、会被查),``not in`` 那半给**假绿**(不吵、没人会查)。
from tests.v2.srcref import source_of  # noqa: E402

from mast.monitoring.service import CurrentMonitorService  # noqa: E402
from mast.monitoring.thresholds import MonitorThresholds  # noqa: E402

#: 独立设定的合成电流轨值，用于量程配置的正反对照。
SYNTHETIC_RAIL_A = 20e-9

LOGGER = "mast.monitoring.service"


def _svc():
    return CurrentMonitorService(pool_getter=lambda: None)


def _feats(*, railed_frac, max_a, min_a=0.0):
    return {"railed_frac": railed_frac, "max_a": max_a, "min_a": min_a}


def test_a_pin_far_below_the_configured_rail_is_shouted_about(caplog):
    """独立构造的贴轨电流与错误量程配置明显不符时，应报告配置不匹配。"""
    svc = _svc()
    th = MonitorThresholds()          # 出厂 cm_sat_current_a = 90 nA
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        svc._cross_check_rail(_feats(railed_frac=1.0, max_a=SYNTHETIC_RAIL_A), th)

    msgs = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert msgs, "信号被钉在一个远低于配置满量程的值上,却一个字都没说"
    joined = "\n".join(msgs)
    assert "cm_sat_current_a" in joined, "没有点名该去改哪个键"
    assert "9.0" in joined or "9e-08" in joined or "9.000000e-08" in joined


def test_it_does_not_use_sat_frac_which_is_computed_from_the_suspect_number(caplog):
    """判据必须与阈值无关。

    配置 90 nA 时,10 nA 的贴轨 `sat_frac` 是 **0** —— 用它去查等于用被怀疑的那个量
    检查它自己。这条测试连 `sat_frac` 都不传,而对账仍然必须成立。
    """
    svc = _svc()
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        svc._cross_check_rail(
            {"railed_frac": 1.0, "max_a": SYNTHETIC_RAIL_A, "min_a": 0.0},  # 无 sat_frac
            MonitorThresholds())
    assert any(r.levelno >= logging.WARNING for r in caplog.records)


def test_a_correctly_configured_rail_says_nothing(caplog):
    """标定对了就闭嘴 —— 否则这条自己变成噪声。"""
    svc = _svc()
    th = MonitorThresholds.from_mapping({"cm_sat_current_a": 20e-9})   # 与合成轨值匹配的配置
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        svc._cross_check_rail(_feats(railed_frac=1.0, max_a=SYNTHETIC_RAIL_A), th)
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_an_unpinned_segment_says_nothing(caplog):
    """普通测量:相邻采样几乎不可能逐位相同 ⇒ railed_frac 低 ⇒ 不该说话。"""
    svc = _svc()
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        svc._cross_check_rail(_feats(railed_frac=0.05, max_a=1.2e-10),
                              MonitorThresholds())
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_it_only_says_it_once_per_episode(caplog):
    """边沿触发:一段持续贴轨不该每秒刷一行。"""
    svc = _svc()
    th = MonitorThresholds()
    f = _feats(railed_frac=1.0, max_a=SYNTHETIC_RAIL_A)
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        for _ in range(5):
            svc._cross_check_rail(f, th)
    assert len([r for r in caplog.records if r.levelno >= logging.WARNING]) == 1

    # 恢复正常之后再贴轨,应当再说一次(否则第二次事故会静默)。
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        svc._cross_check_rail(_feats(railed_frac=0.0, max_a=1e-10), th)
        svc._cross_check_rail(f, th)
    assert len([r for r in caplog.records if r.levelno >= logging.WARNING]) == 2


def test_a_negative_rail_is_detected_too(caplog):
    """贴的是负轨也一样 —— 判据取绝对值。"""
    svc = _svc()
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        svc._cross_check_rail(
            _feats(railed_frac=1.0, max_a=0.0, min_a=-SYNTHETIC_RAIL_A),
            MonitorThresholds())
    assert any(r.levelno >= logging.WARNING for r in caplog.records)


def test_the_cross_check_is_actually_reached_from_the_segment_path():
    """**可达性**:每段都要真的调它,而不是「这个方法自己能工作」。

    ⚠️ 上面那些测试全都**直接调** ``_cross_check_rail``。把 ``_on_segment`` 里那一行
    删掉,它们**全都照绿** —— 变异 R1 当场证明了这一点。这是同一天里第四次
    「测了原语没测调用点」(前三次:`_CriticalWatch` / `_rail_held_for` / `idle_s`)。

    所以这条对着**调用点**核。它不漂亮,但它是唯一能让 R1 变红的那条。
    """
    import inspect

    src = source_of(CurrentMonitorService._on_segment)
    assert "_cross_check_rail" in src, (
        "每段的处理路径里没有调用对账 —— 这条自检永远不会跑")

    # 自检:确认这条闸门认得出「被删掉」的样子(否则它的绿可能只是扫不到东西)。
    other = source_of(CurrentMonitorService._emit_aux_warns)
    assert "def _emit_aux_warns" in other, "取源坏了,这条自检本身失效"
    assert "_cross_check_rail" not in other


def test_garbage_features_never_raise():
    """对账绝不能反噬采集。"""
    svc = _svc()
    th = MonitorThresholds()
    for bad in ({}, {"railed_frac": "x"}, {"railed_frac": 1.0, "max_a": None},
                {"railed_frac": None}, {"railed_frac": 1.0, "max_a": float("nan")}):
        svc._cross_check_rail(bad, th)   # 不抛就算过
