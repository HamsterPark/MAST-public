"""旁白**没接上的时候什么都不做**，接上了也绝不让调用方等。

设计文档 ``docs/v2/design/chat_narration_sidechannel.md`` §Q8 要求的不是「跑一遍没崩」，
是「证明它不可能弄坏正在跑的实验」。这份文件钉三件事，每一件都对应一条真实的坏法：

1. **三种 no-op** —— 开关关 / 没有会话 id / 进程里没有转录存储。任何一种下都
   **一个字节都不写**。这是「关掉之后系统行为逐字节相同」那句承诺的兑现方式。
2. **绝不阻塞** —— 存储卡死 2 秒，``narrate()`` 仍然毫秒级返回。仪器线程等一次
   SQLite 写就是让针尖在原地多停一会儿。
3. **满了丢最旧** —— 灌一万条，队列有界、调用方不卡、进程不涨。

这里**没有一个手写的 store 替身**：写的是真的 ``ConversationStore``，只是建在
``tmp_path`` 上。理由有两条，缺一不可：

* 替身会说真组件不会说的话。本仓今晚刚被这件事咬过 —— 一份测试用
  ``sharpness(verdict="good")`` 让一条真机上结构不可达的快乐路径绿了几个月。
  没有替身，就没有词汇表可以说错。
* 「测试污染真实用户数据」在本仓已经发生**五次**，而旁白写的正是用户真实会话
  所在的那个库。所以这里不但把库指到 tmp，还**断言真实库那一侧没有被创建**
  （光重定向不够 —— 第五次事故的形态就是「复位把逃生门打开了」）。

    .venv-v2-py313/Scripts/python.exe -m pytest tests/v2/unit/chat/test_narration_noop.py -q
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

sys.modules.setdefault("nanonis_spm", MagicMock())


def _find_mastv2_root() -> str:
    p = Path(__file__).resolve()
    while p.parent != p:
        candidate = p / "MASTv2"
        if candidate.is_dir():
            return str(candidate)
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

import pytest  # noqa: E402

from mast.chat import narration  # noqa: E402
from mast.chat.store import ConversationStore  # noqa: E402
from mast.core.turn_context import turn_scope  # noqa: E402

CID = "conv-test-narration"


@pytest.fixture()
def store(tmp_path):
    """真的 ConversationStore，建在 tmp 上。"""
    narration.reset_for_tests()
    st = ConversationStore(tmp_path / "chat" / "mast_conversations.db")
    narration.set_store(st)
    yield st
    narration.flush(2.0)
    narration.reset_for_tests()


def _rows(store: ConversationStore) -> list[dict]:
    return [r for r in store.messages_since(CID, 0)
            if r["kind"] == narration.NARRATION_KIND]


def _all_narration_rows(store: ConversationStore) -> list[dict]:
    """整张表里的旁白行 —— **不按 conversation_id 过滤**。

    只查 CID 的话，「旁白挂到了别的会话上」这种坏法会被读成「什么都没写」。
    那正是这个模块最该防住的错（``turn_context`` 第三条纪律：只读不猜），
    所以断言不能自己把它藏起来。
    """
    import sqlite3
    conn = sqlite3.connect(str(store._db_path))
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute(
            "SELECT * FROM conversation_messages WHERE kind=?",
            (narration.NARRATION_KIND,)).fetchall()]
    finally:
        conn.close()


def _narrate_and_settle(kind: str, **kw) -> None:
    # ``kind`` 分出来单独收:``narrate()`` 的第一个形参是**位置限定**的
    # (2026-08-18)。不然任何一条数据字段里带 ``kind`` 的旁白 —— 比如
    # ``poke_decision`` 的决策码 —— 都会撞上那个形参名。
    with turn_scope(conversation_id=CID, run_id="run-1"):
        narration.narrate(kind, **kw)
    narration.flush(2.0)


# ── 1. 三种 no-op ───────────────────────────────────────────────────────


def test_switch_off_writes_nothing(store, monkeypatch):
    monkeypatch.setenv("MAST_CHAT_NARRATION", "0")
    assert narration.enabled() is False
    _narrate_and_settle(kind="scan_start")
    assert _all_narration_rows(store) == []
    assert narration.stats()["off"] == 1
    assert narration.stats()["emitted"] == 0


def test_no_conversation_id_writes_nothing(store):
    """没有会话 id ⇒ **不发**，而不是发给「最近那个会话」。

    ``turn_context`` 的第三条纪律是「只读不猜」。一条挂错会话的旁白会让用户
    相信一条不存在的因果链，那比没有旁白坏。

    ``turn_scope(conversation_id="")`` 是显式清空（``set_turn`` 把空串写成 None），
    不是「大概没人设过」。整个目录一起跑的时候，前面的用例**确实**会在
    ContextVar 里留下一个 conversation_id —— 第一版这条断言就是这么假绿的：
    它只查了 CID 那一个会话，而那条旁白挂到了泄漏来的那个会话上。
    """
    with turn_scope(conversation_id=""):
        narration.narrate("scan_start")
    narration.flush(2.0)
    assert _all_narration_rows(store) == []
    assert narration.stats()["no_cid"] == 1


def test_no_store_writes_nothing_and_starts_no_thread():
    """进程里没有转录存储（standalone dev / 单测）⇒ no-op，且**不起写线程**。

    「起了线程只是没东西写」和「根本没起」在功能上一样，在资源上不一样 ——
    每个导入本模块的单测进程都多一根线程是没有理由的。

    前置条件要**自己建立**，不能假设。``narration.current_store()`` 在没有显式
    注册时会去问进程级的 ``api.context``，而任何一个先跑过的用例只要建过 app
    就会在那里留下一份接好线的 ctx（``create_app`` 内部 ``set_context``）——
    第一版这条断言就是这么红的，而它红得对：那种残留在真机上意味着旁白会写进
    另一个 store。
    """
    from mast.api.context import AppContext, get_context, set_context

    narration.reset_for_tests()
    prev = get_context()
    set_context(AppContext())
    try:
        assert narration.current_store() is None, \
            "这个进程里居然接着一个真的 ConversationStore —— 测试会往真库里写"
        import threading
        before = sum(1 for t in threading.enumerate() if t.name == "mast-narration")
        with turn_scope(conversation_id=CID, run_id="run-1"):
            narration.narrate("scan_start")
        after = sum(1 for t in threading.enumerate() if t.name == "mast-narration")
        assert narration.stats()["no_store"] == 1
        assert narration.stats()["emitted"] == 0
        assert after == before
    finally:
        set_context(prev)


def test_happy_path_actually_writes_into_the_temp_db(store, tmp_path):
    """反证：上面三条 no-op 不是因为整条链路本来就不通。

    没有这一条，前三个断言可以在一个彻底坏掉的模块上全绿。顺带钉住
    §10.10 —— 断言那一行**确实落在 tmp 里**，而不是只把路径指过去。
    """
    _narrate_and_settle(kind="scan_start")
    rows = _rows(store)
    assert len(rows) == 1
    assert rows[0]["agent_id"] == "", "旁白不属于任何 agent（见 narration._write_one）"
    assert "扫" in rows[0]["text"]
    written = list((tmp_path / "chat").glob("mast_conversations.db*"))
    assert written, "行落库了，但不在 tmp 里 —— 那它落在用户的真库里了"


# ── 2. 绝不阻塞 ─────────────────────────────────────────────────────────


class _GlacialStore(ConversationStore):
    """真 store 的子类：只把写变慢。

    刻意用**继承**而不是重写一个类：``append_message`` 的签名、seq 分配、
    裁剪规则全部原样继承，所以这个替身不可能在签名上和真的分岔 ——
    它替换的只有「慢」这一件事。
    """

    def append_message(self, *a, **kw):  # type: ignore[override]
        time.sleep(2.0)
        return super().append_message(*a, **kw)


def test_narrate_returns_immediately_even_when_the_store_is_stuck(tmp_path):
    narration.reset_for_tests()
    narration.set_store(_GlacialStore(tmp_path / "slow" / "c.db"))
    try:
        with turn_scope(conversation_id=CID, run_id="run-1"):
            t0 = time.perf_counter()
            for _ in range(20):
                narration.narrate("scan_start")
            elapsed = time.perf_counter() - t0
        # 20 条总共 10 ms 都用不掉;真出问题时这里会是 40 秒。
        assert elapsed < 0.5, f"narrate() 阻塞了 {elapsed:.3f}s —— 那是仪器线程在等 SQLite"
    finally:
        narration.reset_for_tests()


# ── 3. 满了丢最旧 ───────────────────────────────────────────────────────


def test_flooding_drops_oldest_and_never_blocks(tmp_path):
    """一万条灌进去：队列有界、调用方不卡、真的丢掉了一些。

    钉的是「丢**最旧**」这个方向。丢最新会让旁白停在几分钟以前，而用户看旁白
    是为了知道**现在**在干什么。
    """
    narration.reset_for_tests()
    narration.set_store(_GlacialStore(tmp_path / "flood" / "c.db"))
    try:
        with turn_scope(conversation_id=CID, run_id="run-1"):
            t0 = time.perf_counter()
            for _ in range(10_000):
                narration.narrate("scan_start")
            elapsed = time.perf_counter() - t0
        assert elapsed < 5.0, f"灌一万条用了 {elapsed:.1f}s —— 说明有一步在等"
        assert narration._QUEUE.qsize() <= narration._QUEUE_MAX
        assert narration.stats()["dropped"] > 0, \
            "一万条没丢一条 —— 队列上限没生效，内存会一直涨"
    finally:
        narration.reset_for_tests()


# ── 4. anchor ───────────────────────────────────────────────────────────


def test_anchor_is_taken_when_the_line_is_emitted_not_when_it_is_written(store):
    """anchor 必须在**入队时**取。

    写线程可能几百毫秒后才落库，那时对话已经多了两条消息 —— 用落库时刻的
    anchor 会把旁白插到它根本没发生过的位置。这条用「发完之后立刻改 anchor」
    来暴露那个写法。
    """
    narration.set_anchor(CID, 3)
    with turn_scope(conversation_id=CID, run_id="run-1"):
        narration.narrate("scan_start")
    narration.set_anchor(CID, 99)          # 落库之前就变了
    narration.flush(2.0)
    import json
    meta = json.loads(_rows(store)[0]["meta"])
    assert meta["anchor"] == 3
    assert meta["nk"] == "scan_start"
    assert meta["run_id"] == "run-1"


def test_no_anchor_means_append_to_the_end(store):
    """``-1`` = 没有活跃回合（唤醒调度 / 群跑 / 手动触发）。"""
    with turn_scope(conversation_id="conv-anchorless", run_id=""):
        narration.narrate("scan_start")
    narration.flush(2.0)
    import json
    rows = store.messages_since("conv-anchorless", 0)
    assert json.loads(rows[0]["meta"])["anchor"] == -1
