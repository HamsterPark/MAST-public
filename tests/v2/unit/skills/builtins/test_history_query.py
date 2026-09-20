# -*- coding: utf-8 -*-
"""长期监控历史的查询入口。

这个技能存在的理由是一个**缺口**，不是一个便利：2026-08-20 查下来，数据处理
agent 的三十来个工具全是扫描图与谱的处理，两个 agent 的提示词里一次都没提过
``aux`` 与 ``env_history`` 这两套数据存在。一台连续记录了半个月环境参数的仪器，
它自己的 agent 够不着那些数据 —— 于是「昨晚成像变差时温度在做什么」这类问题
连提都提不出来。

所以这里测的重点不是"能不能返回数字"，而是三件更要紧的事：

1. 查不到时**说得出为什么**，不能把"没数据"和"一切正常"混在一起；
2. 报出**这套数据能回答什么频率** —— env 是 60 s 桶，Nyquist 120 s，
   拿它去找 60 秒周期的东西会一无所获且不报错；
3. 数据里混着扫描段时要**出声**（扫描时 Z 装的是形貌，不是环境起伏）。
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from mast.skills.builtins.history_query import QueryMonitorHistory


def _store_with_aux(tmp_path, n=600, dt=0.25, scanning_frac=0.0,
                    period_s=60.0, amp_m=5e-12):
    """造一个带 aux 行的临时 store。"""
    from mast.monitoring.store import CurrentMonitorStore

    st = CurrentMonitorStore(tmp_path / "m.sqlite", tmp_path)
    t0 = time.time() - n * dt
    n_scan = int(n * scanning_frac)
    for i in range(n):
        ts = t0 + i * dt
        z = amp_m * np.sin(2 * np.pi * (i * dt) / period_s) + 1e-9
        st.add_aux_sample(
            ts, {"z_m": float(z)},
            # ctx 的键**带 ctx_ 前缀**（ctx.get("ctx_scanning")）。
            # 传 "scanning" 会静默存成 NULL，读回是 False —— 于是一段一半在
            # 扫描的数据看起来完全干净，而这正是这个字段要防的事。
            {"ctx_scanning": i < n_scan, "ctx_zctrl_on": True})
    return st


def test_missing_store_says_why_instead_of_returning_nothing(monkeypatch):
    """monitoring 没在跑时，要说「没在跑」，不能返回一个空的成功。"""
    monkeypatch.setattr("mast.monitoring.store.get_store_if_exists",
                        lambda: None, raising=False)
    r = QueryMonitorHistory().execute(None, {"source": "aux", "column": "z_m"})
    assert not r.success
    assert "没在跑" in (r.error or "") or "不可用" in (r.error or "")


def test_unknown_source_is_refused_with_a_pointer(monkeypatch):
    """source 写错时要告诉调用方该按**时间尺度**选，而不是只说"非法值"。"""
    r = QueryMonitorHistory().execute(None, {"source": "sensors"})
    assert not r.success
    assert "aux" in (r.error or "") and "env" in (r.error or "")


def test_aux_query_reports_the_resolution_it_can_answer(tmp_path, monkeypatch):
    """必须报出 Nyquist 与频率分辨率 —— 「这套数据能回答什么」是前提，不是脚注。"""
    st = _store_with_aux(tmp_path, n=800, dt=0.25)
    monkeypatch.setattr("mast.monitoring.store.get_store_if_exists",
                        lambda: st, raising=False)
    r = QueryMonitorHistory().execute(
        None, {"source": "aux", "column": "z_m", "hours": 1.0})
    assert r.success, r.error
    stats = r.data["stats"]
    assert stats["n"] > 100
    assert stats["dt_median_s"] == pytest.approx(0.25, rel=0.2)
    assert stats["nyquist_period_s"] == pytest.approx(0.5, rel=0.3)
    assert stats["freq_resolution_hz"] > 0
    assert any("最快周期" in n for n in r.data["notes"])


def test_scanning_contamination_is_announced(tmp_path, monkeypatch):
    """段里混着扫描时要出声 —— 扫描时 Z 装的是形貌与慢轴运动，不是环境起伏。"""
    st = _store_with_aux(tmp_path, n=800, dt=0.25, scanning_frac=0.5)
    monkeypatch.setattr("mast.monitoring.store.get_store_if_exists",
                        lambda: st, raising=False)
    r = QueryMonitorHistory().execute(
        None, {"source": "aux", "column": "z_m", "hours": 1.0})
    assert r.success
    assert r.data["scanning_fraction"] > 0.3
    assert any("扫描中" in n for n in r.data["notes"])


def test_a_quiet_segment_is_not_flagged(tmp_path, monkeypatch):
    """反过来也要成立：安静的段不能被误报成脏的，否则这个提示就没人看了。"""
    st = _store_with_aux(tmp_path, n=800, dt=0.25, scanning_frac=0.0)
    monkeypatch.setattr("mast.monitoring.store.get_store_if_exists",
                        lambda: st, raising=False)
    r = QueryMonitorHistory().execute(
        None, {"source": "aux", "column": "z_m", "hours": 1.0})
    assert r.data["scanning_fraction"] == 0.0
    assert not any("扫描中" in n for n in r.data["notes"])


def test_unknown_column_lists_what_is_available(tmp_path, monkeypatch):
    """列名写错时列出有哪些 —— 猜列名是这个入口最常见的用法错误。"""
    st = _store_with_aux(tmp_path, n=200, dt=0.25)
    monkeypatch.setattr("mast.monitoring.store.get_store_if_exists",
                        lambda: st, raising=False)
    r = QueryMonitorHistory().execute(
        None, {"source": "aux", "column": "temperature_k", "hours": 1.0})
    assert not r.success
    assert "可用的列" in (r.error or "")


def test_series_is_withheld_unless_asked(tmp_path, monkeypatch):
    """默认只回摘要 —— 几万个点塞进对话没有意义，而且会挤掉真正的结论。"""
    st = _store_with_aux(tmp_path, n=800, dt=0.25)
    monkeypatch.setattr("mast.monitoring.store.get_store_if_exists",
                        lambda: st, raising=False)
    lean = QueryMonitorHistory().execute(
        None, {"source": "aux", "column": "z_m", "hours": 1.0}).data
    assert "values" not in lean and "t_s" not in lean

    full = QueryMonitorHistory().execute(
        None, {"source": "aux", "column": "z_m", "hours": 1.0,
               "include_series": True}).data
    assert len(full["values"]) == full["stats"]["n"]


def test_env_source_warns_about_its_own_nyquist(monkeypatch):
    """env 是 60 s 桶 —— **拿它找 60 秒周期的东西会一无所获且不报错**。

    这条提示是这个技能最该说的一句话：不说的话，一个空手而归的查询会被读成
    「那里没有这个扰动」，而真相是这套数据在原理上就看不见它。
    """
    class _FakeEnvStore:
        def series(self, sensor, since=None, until=None, max_points=None):
            t0 = time.time() - 3600
            return {"sensor": sensor, "unit": "K",
                    "points": [{"ts": t0 + 60 * i, "mean": 77.7 + 1e-3 * i}
                               for i in range(60)],
                    "bucket_s_effective": 60.0, "thinned": False, "total": 60}

        def sensors(self):
            return [{"sensor": "SPM (COM13)"}]

    monkeypatch.setattr("mast.envhistory.store.get_store",
                        lambda: _FakeEnvStore(), raising=False)
    r = QueryMonitorHistory().execute(
        None, {"source": "env", "column": "SPM (COM13)", "hours": 1.0})
    assert r.success, r.error
    assert r.data["stats"]["n"] == 60
    notes = " ".join(r.data["notes"])
    assert "60" in notes and "120" in notes
    assert "aux" in notes, "没有告诉调用方想看更快的东西该去哪"
