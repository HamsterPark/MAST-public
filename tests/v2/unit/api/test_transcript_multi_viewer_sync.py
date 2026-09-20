"""两台电脑看同一个群聊时，一台写了另一台要能看到。

2026-07-25 要求：「多个电脑上打开远程窗口时，对话不同步显示」。

转录只在面板挂载时读一次，之后只有点「刷新进展」才再读 —— 所以第二台机器上
的同一个群聊**永远**不会自己出现新消息。今天的 SSE 重连工作解决的是「自己这条流
断了怎么办」，不是「多端看到同一份内容」。

现在 ``ConversationStore.append_message`` 往 EventBus 广播一个**游标**，
``/ws/events`` 推给所有连着的浏览器，面板据此重新拉取。

这份测试钉住的是那条链路里**容易做错**的几处，不是「有没有代码」：

* 广播里**没有正文** —— 只有 conversation_id + seq + kind/agent。谁连上
  ``/ws/events`` 谁就能读到对话内容，是把一个鉴权读取变成广播；要正文的客户端
  仍然走原来的端点。
* 只影响**同一个会话** —— 用户同时开着两个群聊时，一个动不能让另一个刷新。
* 通知失败**绝不能损失消息本身** —— 掉一次通知代价是晚一点刷新，抛出来的代价是
  这条消息没写进去。

从仓库根运行::

    .venv-v2-py313/Scripts/python.exe -m pytest \
        tests/v2/unit/api/test_transcript_multi_viewer_sync.py -q
"""
from __future__ import annotations

# ── path bootstrap ───────────────────────────────────────────────────────────
import sys
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

import tempfile  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from mast.api.app import create_app  # noqa: E402
from mast.chat.store import ConversationStore  # noqa: E402
from mast.core.events import EventBus  # noqa: E402


@pytest.fixture()
def store() -> ConversationStore:
    return ConversationStore(str(Path(tempfile.mkdtemp()) / "c.db"))


def _new_group(store, title: str) -> str:
    conv = store.create("_supervisor", kind="group", title=title)
    return conv["conversation_id"] if isinstance(conv, dict) else conv


def _cursor() -> int:
    _, latest = EventBus.get().recent_events_since(0)
    return int(latest)


def _collect_transcript(ws, want: int, tries: int = 60) -> list[dict]:
    out: list[dict] = []
    for _ in range(tries):
        f = ws.receive_json()
        d = f.get("data") or {}
        if d.get("scope") == "transcript":
            out.append(d)
            if len(out) >= want:
                break
    return out


# ════════════════════════════════════════════════════════════════════════════
# 第二台电脑确实会被通知
# ════════════════════════════════════════════════════════════════════════════

def test_a_second_viewer_is_told_a_message_landed(store):
    cid = _new_group(store, "synthetic_sample 群聊")
    client = TestClient(create_app())
    with client.websocket_connect(f"/ws/events?since={_cursor()}") as viewer2:
        def machine1():
            time.sleep(0.15)
            store.append_message(cid, kind="message",
                                 agent_id="instrument_control",
                                 role="assistant", text="扫描完成，共 512 行")
        threading.Thread(target=machine1, daemon=True).start()
        got = _collect_transcript(viewer2, 1)
    assert got, "第二台电脑没有收到任何通知 —— 对话仍然不同步"
    d = got[0]
    assert d["conversation_id"] == cid
    assert d["seq"] == 1 and d["kind"] == "message"
    assert d["agent_id"] == "instrument_control"


def test_the_broadcast_carries_no_message_text(store):
    """谁连上 /ws/events 谁就能读对话内容 —— 那是把鉴权读取变成了广播。"""
    cid = _new_group(store, "群聊")
    client = TestClient(create_app())
    with client.websocket_connect(f"/ws/events?since={_cursor()}") as ws:
        def w():
            time.sleep(0.15)
            store.append_message(cid, kind="message", text="机密：偏压 +7 V")
        threading.Thread(target=w, daemon=True).start()
        d = _collect_transcript(ws, 1)[0]
    assert "text" not in d
    assert "机密" not in str(d)


def test_another_conversation_does_not_trigger_this_one(store):
    """用户同时开着两个群聊时，一个动不能让另一个刷新。"""
    cid_a = _new_group(store, "本会话")
    cid_b = _new_group(store, "另一个群聊")
    client = TestClient(create_app())
    with client.websocket_connect(f"/ws/events?since={_cursor()}") as ws:
        def w():
            time.sleep(0.15)
            store.append_message(cid_b, kind="message", text="别的会话")
            store.append_message(cid_a, kind="message", text="本会话")
        threading.Thread(target=w, daemon=True).start()
        got = _collect_transcript(ws, 2)
    # 两条都会广播（服务端不知道谁在看什么），过滤是前端按 conversation_id 做的。
    mine = [d for d in got if d["conversation_id"] == cid_a]
    assert len(mine) == 1, "本会话应恰好收到一条"
    assert any(d["conversation_id"] == cid_b for d in got), (
        "另一个会话的事件也该广播 —— 过滤是客户端的事，服务端不该替它决定")


def test_every_kind_of_row_is_announced(store):
    """状态行 / 中断 / 压缩标记也要同步 —— 只同步 message 会让第二台看到一份
    缺了上下文的历史。"""
    cid = _new_group(store, "群聊")
    client = TestClient(create_app())
    kinds = ["status", "interrupt", "compaction"]
    with client.websocket_connect(f"/ws/events?since={_cursor()}") as ws:
        def w():
            time.sleep(0.15)
            for k in kinds:
                store.append_message(cid, kind=k, text=f"{k} row")
        threading.Thread(target=w, daemon=True).start()
        got = _collect_transcript(ws, len(kinds))
    assert [d["kind"] for d in got] == kinds


# ════════════════════════════════════════════════════════════════════════════
# 通知不能反过来伤到写入
# ════════════════════════════════════════════════════════════════════════════

def test_a_broken_notifier_never_costs_the_message(store, monkeypatch):
    from mast.chat import store as S

    cid = _new_group(store, "群聊")
    monkeypatch.setattr(S, "_publish_transcript_append",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("bus down")))
    # 通知炸了，消息仍然必须写进去
    seq = store.append_message(cid, kind="message", text="仍然要写进去")
    assert seq == 1
    assert store.messages_for(cid)[-1]["text"] == "仍然要写进去"


def test_the_seq_is_the_one_the_client_can_use_as_a_cursor(store):
    """广播的 seq 必须等于这行在会话里的真实 seq，否则客户端拿它做去重会错。"""
    cid = _new_group(store, "群聊")
    client = TestClient(create_app())
    with client.websocket_connect(f"/ws/events?since={_cursor()}") as ws:
        def w():
            time.sleep(0.15)
            for i in range(3):
                store.append_message(cid, kind="message", text=f"m{i}")
        threading.Thread(target=w, daemon=True).start()
        got = _collect_transcript(ws, 3)
    assert [d["seq"] for d in got] == [1, 2, 3]
    assert [r["seq"] for r in store.messages_for(cid)] == [1, 2, 3]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))


# ════════════════════════════════════════════════════════════════════════════
# 私聊也要有这条通道（，2026-08-06）
#
# 上面那条链路挂在 ``ConversationStore.append_message`` 上，而**私聊的正文根本
# 不走这张表** —— 它落在 LangGraph 的 checkpointer 里（``/agents/{id}/messages``
# 读的就是那里）。所以群聊转录有推送、私聊一条都没有，「代理对话」那一页只在
# 本浏览器自己发完消息之后才刷新一次。判据:「智能体代理对话不会自动刷新」。
#
# 同一个缺陷，同一条纪律，第二个存储位置 —— 这正是「两处各写一份」的形状，
# 所以这一组写在同一个文件里，让下一个人一眼看到两条路必须一起想。
# ════════════════════════════════════════════════════════════════════════════

def _collect_scope(ws, scope: str, want: int, tries: int = 60) -> list[dict]:
    out: list[dict] = []
    for _ in range(tries):
        f = ws.receive_json()
        d = f.get("data") or {}
        if d.get("scope") == scope:
            out.append(d)
            if len(out) >= want:
                break
    return out


def test_a_finished_private_turn_reaches_a_second_viewer():
    """私聊回合写完，别的窗口要能知道。"""
    from mast.chat.store import publish_private_turn_finished

    client = TestClient(create_app())
    with client.websocket_connect(f"/ws/events?since={_cursor()}") as viewer2:
        def machine1():
            time.sleep(0.15)
            publish_private_turn_finished("conv-abc", "instrument_control")
        threading.Thread(target=machine1, daemon=True).start()
        got = _collect_scope(viewer2, "private_turn", 1)
    assert got, "私聊回合没有任何通知 —— 「代理对话」仍然不会自动刷新"
    assert got[0]["conversation_id"] == "conv-abc"
    assert got[0]["agent_id"] == "instrument_control"


def test_the_private_turn_broadcast_carries_no_text():
    """与群聊那条同一条纪律：只推游标，正文走已鉴权的读端点。

    私聊尤其不能松 —— 群聊本来就是多方可见的，私聊不是。
    """
    from mast.chat.store import publish_private_turn_finished

    client = TestClient(create_app())
    with client.websocket_connect(f"/ws/events?since={_cursor()}") as ws:
        def w():
            time.sleep(0.15)
            publish_private_turn_finished("conv-secret", "paper_writing")
        threading.Thread(target=w, daemon=True).start()
        d = _collect_scope(ws, "private_turn", 1)[0]
    assert set(d) == {"scope", "conversation_id", "agent_id"}, d


def test_private_turn_uses_its_own_scope_not_the_group_one():
    """``scope`` 刻意不是 ``"transcript"``。

    前端 ``RunTaskPanel`` 拿 ``d.scope !== "transcript"`` 做第一道过滤，然后按
    ``seq`` 去重。借用那个值的话，私聊事件会走进群聊那条路，而它没有 seq ——
    ``Number(undefined ?? 0) === 0``，去重条件恒真，症状是「一条都不刷」，
    而不是任何看得见的错误。一个新语义配一个新名字。
    """
    from mast.chat.store import publish_private_turn_finished

    client = TestClient(create_app())
    with client.websocket_connect(f"/ws/events?since={_cursor()}") as ws:
        def w():
            time.sleep(0.15)
            publish_private_turn_finished("conv-x", "literature")
        threading.Thread(target=w, daemon=True).start()
        d = _collect_scope(ws, "private_turn", 1)[0]
    assert d["scope"] == "private_turn"
    assert "seq" not in d, "带了 seq 就会被误认成群聊游标"


def test_a_broken_bus_never_costs_the_turn(monkeypatch):
    """掉一次通知的代价是晚几秒刷新；抛一次的代价是用户的那轮对话。"""
    from mast.chat import store as chat_store

    class Boom:
        @staticmethod
        def get():
            raise RuntimeError("bus is down")

    monkeypatch.setattr("mast.core.events.EventBus", Boom)
    chat_store.publish_private_turn_finished("c", "ic")      # 不抛就是通过
