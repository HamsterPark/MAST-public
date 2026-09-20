"""`CoreRuntime._wake_goal_check` —— 唤醒准入读的那份判据。

这是唤醒环上**唯一**能回答「这份等待还该不该醒」的地方，而它在实现之初一条
测试都没有（自审 2026-08-28 点名）。它错的两个方向后果完全不对称：

* 该醒的没醒 ⇒ 一份**永远不醒**的 park，只能等 24 h 超时浮出来；
* 不该醒的醒了 ⇒ 多花一次唤醒，而那有配额兜着。

所以这里的每一条都在确认它**偏向唤醒**：读不到、没写判据、写坏了、纲领不存在，
一律回 ``None``（= 照旧唤醒），只有真的判出 done 才关 park。
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

import pytest  # noqa: E402

from mast.core.runtime import CoreRuntime  # noqa: E402


@pytest.fixture()
def v2_store(tmp_path, monkeypatch):
    monkeypatch.setenv("MAST_DATA_DIR", str(tmp_path))
    from mast.agents._shared.data_paths import v2_experiment_db_path

    assert str(tmp_path) in str(v2_experiment_db_path()), "重定向没生效，不许继续"
    return tmp_path


def _check(park: dict):
    """按未绑定方法调 —— 不去构造一整个 CoreRuntime。

    ``_wake_goal_check`` 只用到 ``self`` 的日志，不碰实例状态；用 ``None`` 当
    self 是诚实的（它一旦开始读实例字段，这条测试会当场 AttributeError 而不是
    悄悄绿着）。
    """
    return CoreRuntime._wake_goal_check(None, park)


def _campaign(goal: dict | None = None, status: str = "running") -> str:
    from mast.agents._shared.data_paths import v2_experiment_db_path
    from mast.logging.v2.repos import build_repos
    from mast.logging.v2.storage import open_store

    repos = build_repos(open_store(v2_experiment_db_path()))
    cid = repos.campaigns.create(title="t", hypothesis="h",
                                 hypothesis_kind="confirmatory",
                                 goal=goal or {}, created_by="test")
    if status != "draft":
        repos.campaigns.set_status(cid, status)
    return cid


# ── 偏向唤醒的那几条 ────────────────────────────────────────────────

def test_a_park_without_a_campaign_wakes_as_before(v2_store):
    """归属只认 park 上冻结的那一个。空串 ⇒ 不判、照旧唤醒 —— **不猜**。"""
    assert _check({}) is None
    assert _check({"campaign_id": ""}) is None
    assert _check({"campaign_id": "   "}) is None


def test_an_unknown_campaign_id_wakes_as_before(v2_store):
    assert _check({"campaign_id": "01NOSUCHCAMPAIGN"}) is None


def test_a_campaign_without_done_when_wakes_as_before(v2_store):
    """没写判据 ⇒ 照旧唤醒。**不是** done —— 「没人写过什么算答完」不是答完了。"""
    cid = _campaign({"question": "还没写判据"})
    assert _check({"campaign_id": cid}) is None


def test_an_unparsable_done_when_wakes_as_before(v2_store):
    """判据写坏了 ⇒ 照旧唤醒，而不是把这份 park 永远关掉。"""
    cid = _campaign({"done_when": [{"kind": "vibes"}]})
    assert _check({"campaign_id": cid}) is None


def test_an_unsatisfied_criterion_does_not_close_the_park(v2_store):
    cid = _campaign({"done_when": {"all": [
        {"kind": "claims_supported", "min_count": 5}]}})
    v = _check({"campaign_id": cid})
    assert v is not None and v.verdict == "not_done"


# ── 真的该关的那一条 ────────────────────────────────────────────────

def test_a_completed_campaign_stops_its_parks(v2_store):
    """人或 RD 已经把这条纲领收口了 ⇒ 它下面的 park 不该再花钱醒。

    这是**唯一**不需要 done_when 就生效的行为变化，所以单独钉一条。
    """
    for status in ("completed", "aborted"):
        cid = _campaign({"question": "q"}, status=status)
        v = _check({"campaign_id": cid})
        assert v is not None and v.verdict == "done", status
        assert status in v.reason, v.reason
        assert "不是判据" in v.reason or "状态" in v.reason


def test_a_paused_campaign_still_wakes(v2_store):
    """``paused`` 不是终态 —— 暂停一条纲领不该顺手掐掉它下面所有的等待。"""
    cid = _campaign({"question": "q"}, status="paused")
    assert _check({"campaign_id": cid}) is None


# ── 坏掉的世界里也不许把 park 关掉 ──────────────────────────────────

def test_a_goal_json_that_is_not_an_object_wakes_as_before(v2_store):
    """``goal_json`` 合法但不是对象（一个列表 / 一个字符串）。

    列上有 ``CHECK (json_valid(goal_json))``，所以「坏 JSON」进不了库 —— 真正
    可能的坏形状是这一种。读到它要照旧唤醒，而不是拿一个 ``AttributeError``
    杀掉整趟 tick。
    """
    from mast.agents._shared.data_paths import v2_experiment_db_path
    from mast.logging.v2.repos import build_repos
    from mast.logging.v2.storage import open_store

    cid = _campaign({"done_when": [{"kind": "claims_supported", "min_count": 1}]})
    repos = build_repos(open_store(v2_experiment_db_path()))
    for bad in ('["不是对象"]', '"就是一个字符串"', "123"):
        with repos.campaigns.store.connect() as conn:
            conn.execute("UPDATE campaigns SET goal_json=? WHERE id=?", (bad, cid))
        assert _check({"campaign_id": cid}) is None, bad


def test_the_evaluator_blowing_up_wakes_as_before(v2_store, monkeypatch):
    cid = _campaign({"done_when": [{"kind": "claims_supported", "min_count": 1}]})

    def _boom(*a, **k):
        raise RuntimeError("求值器炸了")

    monkeypatch.setattr("mast.goals.evaluate_done_when", _boom)
    assert _check({"campaign_id": cid}) is None


def test_it_never_asks_the_operator(v2_store):
    """空闲进程里没人能回答问题 —— ``operator_confirmed`` 必须是判不了，
    而不是「没确认」（后者会让这份 park 永远醒不了……不对，是永远醒着；
    真正的问题是它会被当成一个**读得到的否定**，掩盖掉「其实问不出去」）。"""
    cid = _campaign({"done_when": [{"kind": "operator_confirmed"}]})
    v = _check({"campaign_id": cid})
    assert v is not None and v.verdict == "unknown", v.verdict
    assert "提问" in v.reason or "checkpointer" in v.reason
