"""订阅 store：落盘、冷启动、以及「读不出来」到底意味着什么。

这个文件回答的是 [[producer_wired_consumer_absent]] 那个问题 ——「已经记下来了」是
生产方的话，**谁读它**？订阅 holder 是惰性自读的（第一次被问到时从盘上加载），所以
读端与写端在同一个模块里、由同一组测试证明接上了：写一次 → 丢掉进程态 → 再读，拿到
的必须是盘上那份。这比「断言 runtime 启动序列里有那一行」更强，因为它验的是行为
而不是形状。

另一半是失败语义。这个 store 有三种「没有订阅列表」，它们**不是**一回事：

* 文件不存在        → 还没人定制过（正常，全订阅）
* 文件读不出来      → 有人配了但我们没看懂（全订阅 + **说出来**）
* 文件里是空 entries → 他真的退订了一切（那就是空）

把前两种折叠成第三种，就是 [[unknown_is_not_an_answer]] 那一族事故：agent 一夜之间
失去全部技能，而界面上一切正常。
"""

from __future__ import annotations

import json

import pytest

from mast.skills import subscription as sub


# ─────────────────────────────────────────────────────────────────────────────
# 三种「没有订阅列表」互不相同
# ─────────────────────────────────────────────────────────────────────────────

def test_missing_file_is_uncustomised_not_an_error(subscription_store):
    assert not subscription_store.exists()
    assert sub.is_customised() is False
    assert sub.unreadable_reason() == ""
    assert sub.unloaded_skill_names({"A", "B"}) == frozenset()


def test_empty_entries_really_means_everything_off(subscription_store):
    """他真的退订了一切 —— 这一种必须与上面两种区分开。"""
    sub.set_subscribed(set())
    sub.reset_default_store()
    assert sub.is_customised() is True
    assert sub.unreadable_reason() == ""
    assert sub.unloaded_skill_names({"A", "B"}) == {"A", "B"}


def test_unreadable_is_distinguishable_from_missing(subscription_store):
    subscription_store.parent.mkdir(parents=True, exist_ok=True)
    subscription_store.write_text("{ broken", encoding="utf-8")
    sub.reset_default_store()

    assert sub.unreadable_reason(), "读不出来必须留下原因"
    assert sub.is_customised() is False
    assert sub.unloaded_skill_names({"A", "B"}) == frozenset()


def test_unreadable_never_shrinks_the_tool_face(subscription_store):
    """核心不变量：无论文件坏成什么样，工具面**不会变小**。

    manifest.py 那条「已启用的条目没有被停用」在这里的等价物。方向一致：一个我们
    读不懂的文件，不该有权力关掉任何东西。
    """
    universe = {f"S{i}" for i in range(20)}
    sub.set_subscribed(universe - {"S0"})
    before = sub.unloaded_skill_names(universe)
    assert before == {"S0"}, "前提没立住"

    subscription_store.write_text("！ 不是 JSON ！", encoding="utf-8")
    sub.reset_default_store()

    after = sub.unloaded_skill_names(universe)
    assert not (after - before), (
        f"订阅文件坏掉之后，工具面反而少了 {sorted(after - before)} —— "
        "一个读不懂的文件不该有权力关掉任何东西")
    assert after == frozenset()


def test_a_corrupt_file_is_not_re_read_behind_your_back(subscription_store):
    """进程态是热的：盘上的文件被别人写坏，不会在下一次读时静默改变行为。"""
    sub.set_subscribed({"A"})
    subscription_store.write_text("garbage", encoding="utf-8")
    assert sub.subscribed_names() == {"A"}, "有人在背后换掉了活着的订阅状态"


# ─────────────────────────────────────────────────────────────────────────────
# 落盘 / 冷启动
# ─────────────────────────────────────────────────────────────────────────────

def test_roundtrip_lands_on_disk_and_comes_back(subscription_store):
    """变异必须证明自己动了手：文件 / holder / 冷启动三处一致。"""
    sub.set_subscribed({"A", "B"})

    assert subscription_store.is_file(), "写了却没落盘"
    doc = json.loads(subscription_store.read_text(encoding="utf-8"))
    assert doc["customised"] is True
    assert sorted(doc["entries"]) == ["A", "B"]
    assert doc["schema_version"] == sub.SCHEMA_VERSION

    sub.reset_default_store()                       # 冷启动
    assert sub.subscribed_names() == {"A", "B"}


def test_write_is_atomic_and_leaves_no_part_file(subscription_store):
    sub.set_subscribed({"A"})
    leftovers = list(subscription_store.parent.glob("*.part"))
    assert not leftovers, f"原子写留下了半截文件：{leftovers}"


def test_store_path_is_lazy_and_follows_the_project_root(tmp_path, monkeypatch):
    """惰性解析 —— 模块级常量 + 私有 parents[N] 走法正是 wishlist 那次的成因。"""
    monkeypatch.setenv("MAST2_PROJECT_ROOT", str(tmp_path))
    sub.reset_default_store()
    try:
        assert sub.store_path().parent.parent == tmp_path
    finally:
        sub.reset_default_store()


def test_pending_and_audit_survive_a_reload(subscription_store):
    sub.set_subscribed({"A"})
    rec = sub.add_recommendation("B", by_agent="dp", reason="因为")["recommendation"]
    sub.reset_default_store()

    pend = sub.pending_recommendations()
    assert [p["id"] for p in pend] == [rec["id"]]
    assert pend[0]["reason"] == "因为"
    assert sub.audit_tail(), "audit 没落盘 —— 「谁在什么时候改的」全丢了"


def test_audit_records_who_did_it(subscription_store):
    sub.subscribe(["B"], all_names={"A", "B"}, via=sub.VIA_IMPORT)
    actions = [(a["action"], a["via"]) for a in sub.audit_tail()]
    assert (sub.VIA_MATERIALISE, sub.VIA_IMPORT) in actions, (
        "第一次定制没有留下 materialise 记录 —— 那之后没人说得清「全订阅」是"
        "什么时候变成明确名单的")


def test_pending_is_capped(subscription_store):
    sub.set_subscribed({"A"})
    for i in range(sub.MAX_PENDING + 10):
        sub.add_recommendation(f"S{i}")
    assert len(sub.pending_recommendations()) <= sub.MAX_PENDING, (
        "一个跑飞的 agent 能把这个文件刷到多大？")


@pytest.mark.parametrize("doc", [
    {"customised": True, "entries": ["A", None, 3, "", "  "]},
    {"customised": True, "entries": ["A"], "pending": ["not a dict", {}]},
    {"customised": True, "entries": ["A"], "audit": "nope"},
])
def test_partially_broken_docs_keep_what_is_readable(subscription_store, doc):
    """看得懂的部分留下，看不懂的丢掉 —— 整份拒绝会把一份好清单因为一行脏数据废掉。"""
    subscription_store.parent.mkdir(parents=True, exist_ok=True)
    subscription_store.write_text(json.dumps(doc), encoding="utf-8")
    sub.reset_default_store()
    assert sub.subscribed_names() == {"A"}
    assert sub.unreadable_reason() == ""
