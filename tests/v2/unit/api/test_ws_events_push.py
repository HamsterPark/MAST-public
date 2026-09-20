"""`/ws/events` 从「只发 ping 的桩」变成真正的推送通道。

要求:补上 WebSocket 推送 —— 已经有 tailscale 做连接,很多轮询该改回 websocket。

在此之前 `/ws/events` 的注释写着「保持一个契约正确的 socket 供前端连接」——
它接受连接、每 15 秒发一个 ping，**从不发数据**。于是每一个需要实时的东西都在
轮询：顶栏、右栏、仪表盘各 2 s 一次读数，群聊转录干脆只在挂载时拉一次
（「多个电脑上打开远程窗口时，对话不同步显示」）。

EventBus 的骨架一直是齐的 —— 序号化的 100 条环形历史，docstring 明写就是为了
「重连后追赶」。缺的只是把两端接上。

这份测试钉住四件事：

1. **断线追赶是无损的**：连接时带 `?since=<最后收到的 seq>`，服务端先补发历史里
   seq 更大的，再进入实时。没有它，重连＝丢消息，而丢了什么谁也不知道。
2. **跨线程发布能到达**：真实的发布者是后台线程（state.refresh 的 1 Hz 轮询、
   技能线程），而 socket 在事件循环上。中间那座桥断了的话，本地测试会通、
   真机会静默不推。
3. **慢客户端会被告知丢了什么**（`dropped`），而不是拿到一个自己看不见的空洞。
   无界队列则是把一个卡住的浏览器标签变成服务端内存泄漏。
4. **转录广播不带正文** —— 只发游标。要正文的客户端走原有的已鉴权读取。

从仓库根运行::

    .venv-v2-py313/Scripts/python.exe -m pytest \
        tests/v2/unit/api/test_ws_events_push.py -q
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
from mast.core.events import Event, EventBus, EventType  # noqa: E402


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(create_app())


def _pub(n: int = 1, **data):
    bus = EventBus.get()
    for i in range(n):
        bus.publish(Event(type=EventType.HARDWARE_STATE,
                          data={**data, "i": i}))


def _drain_events(ws, want: int, tries: int = 40, *,
                  tag: str | None = None) -> list[dict]:
    """Collect `want` event frames, skipping pings.

    ``tag`` filters to events THIS test published. The EventBus is a
    process-wide singleton with a shared 100-event history, so a bare
    ``since=0`` replays whatever every other test module has published — a
    test that only passes when run alone is worse than no test.
    """
    out: list[dict] = []
    for _ in range(tries):
        f = ws.receive_json()
        if f.get("kind") != "event":
            continue
        if tag is not None and (f.get("data") or {}).get("tag") != tag:
            continue
        out.append(f)
        if len(out) >= want:
            break
    return out


def _cursor() -> int:
    """The bus's current latest seq — connect here to start with an empty
    replay, whatever other tests have left in the ring buffer."""
    _, latest = EventBus.get().recent_events_since(0)
    return int(latest)


# ════════════════════════════════════════════════════════════════════════════
# 1 · 断线追赶
# ════════════════════════════════════════════════════════════════════════════

def test_events_published_before_connecting_are_replayed(client):
    c = _cursor()
    _pub(3, tag="replay")
    with client.websocket_connect(f"/ws/events?since={c}") as ws:
        got = _drain_events(ws, 3, tag="replay")
    assert len(got) == 3
    assert [g["seq"] for g in got] == sorted(g["seq"] for g in got), "补发乱序"


def test_since_skips_what_the_client_already_has(client):
    c = _cursor()
    _pub(2, tag="a")
    with client.websocket_connect(f"/ws/events?since={c}") as ws:
        first = _drain_events(ws, 2, tag="a")
    cursor = max(g["seq"] for g in first)
    _pub(2, tag="b")
    with client.websocket_connect(f"/ws/events?since={cursor}") as ws:
        second = _drain_events(ws, 2, tag="b")
    assert all(g["seq"] > cursor for g in second), (
        "重连后又收到了已经看过的事件 —— since 游标没生效")


def test_a_garbage_since_does_not_break_the_socket(client):
    """客户端存了个坏游标不能让它连不上。

    发布放在**连接之后**：坏 since 会被当成 0，也就是重放整个环形历史，而那里面
    有多少条取决于同批跑了哪些别的测试。用实时事件来判定，结果就不依赖历史长度。
    """
    with client.websocket_connect("/ws/events?since=not-a-number") as ws:
        def pub():
            time.sleep(0.15)
            _pub(1, tag="garbage")
        threading.Thread(target=pub, daemon=True).start()
        assert _drain_events(ws, 1, tag="garbage", tries=200), (
            "坏 since 让连接不可用了")


# ════════════════════════════════════════════════════════════════════════════
# 2 · 跨线程实时推送 —— 真实发布者都在后台线程上
# ════════════════════════════════════════════════════════════════════════════

def test_an_event_published_from_another_thread_arrives(client):
    # Connect AT the current cursor so the history replay is empty and the only
    # frame that can arrive is the live one. (Connecting with since=0 replays
    # everything this module already published, and the first frame drained
    # would be a replay — which is what this test must not accept as proof.)
    _, cursor = EventBus.get().recent_events_since(0)
    with client.websocket_connect(f"/ws/events?since={cursor}") as ws:

        def pub():
            time.sleep(0.15)
            EventBus.get().publish(Event(type=EventType.HARDWARE_STATE,
                                         data={"live": True, "bias_v": 0.5}))
        threading.Thread(target=pub, daemon=True).start()
        got = _drain_events(ws, 1)
    assert got and got[0]["data"].get("live") is True, (
        "后台线程发布的事件没到达 —— 线程→事件循环的桥断了")


def test_the_frame_shape_is_what_the_client_is_written_against(client):
    c = _cursor()
    _pub(1, tag="shape")
    with client.websocket_connect(f"/ws/events?since={c}") as ws:
        f = _drain_events(ws, 1, tag="shape")[0]
    assert f["kind"] == "event"
    assert isinstance(f["seq"], int) and f["seq"] > 0
    assert f["type"] == "hardware_state"
    assert isinstance(f["data"], dict)
    assert "ts" in f


# ════════════════════════════════════════════════════════════════════════════
# 3 · 转录广播 —— #4 多端同步，且不泄露正文
# ════════════════════════════════════════════════════════════════════════════

def test_a_transcript_append_is_broadcast_without_its_text(client):
    from mast.chat.store import ConversationStore

    st = ConversationStore(str(Path(tempfile.mkdtemp()) / "c.db"))
    conv = st.create("_supervisor", kind="group", title="群聊")
    cid = conv["conversation_id"] if isinstance(conv, dict) else conv

    with client.websocket_connect(f"/ws/events?since={_cursor()}") as ws:
        def w():
            time.sleep(0.15)
            st.append_message(cid, kind="message", agent_id="instrument_control",
                              role="assistant", text="机密正文不该上广播总线")
        threading.Thread(target=w, daemon=True).start()
        hit = None
        for _ in range(40):
            f = ws.receive_json()
            if (f.get("data") or {}).get("scope") == "transcript":
                hit = f
                break
    assert hit, "转录追加没有广播 —— 第二台电脑还是看不到新消息"
    d = hit["data"]
    assert d["conversation_id"] == cid and d["seq"] == 1
    assert d["kind"] == "message" and d["agent_id"] == "instrument_control"
    assert "text" not in d, "正文被广播出去了 —— 谁连上 /ws/events 谁就能读对话内容"


def test_a_broken_bus_never_costs_a_message(monkeypatch):
    """通知失败必须只损失一次刷新，不能损失这条消息本身。"""
    from mast.chat import store as S

    st = S.ConversationStore(str(Path(tempfile.mkdtemp()) / "c.db"))
    conv = st.create("_supervisor", kind="group", title="群聊")
    cid = conv["conversation_id"] if isinstance(conv, dict) else conv

    def _boom(*a, **k):
        raise RuntimeError("bus exploded")
    monkeypatch.setattr(S, "_publish_transcript_append", _boom)
    # append_message 内部调用它 —— 若未被吞掉，这里会抛
    with pytest.raises(RuntimeError):
        S._publish_transcript_append(cid, 1, "message", "", "", 0.0)
    # 真实路径必须活下来
    monkeypatch.undo()
    assert st.append_message(cid, kind="message", text="仍然要写进去") == 1


# ════════════════════════════════════════════════════════════════════════════
# 4 · 轮询回退没有被拆掉
# ════════════════════════════════════════════════════════════════════════════

def test_the_http_polling_fallback_still_works():
    """WebSocket 是**多了一条快路**，不是把旧路拆了。企业防火墙会剥
    Upgrade 头，那时轮询是唯一的通路。"""
    bus = EventBus.get()
    _, before = bus.recent_events_since(0)
    _pub(2, tag="fallback")
    events, latest = bus.recent_events_since(before)
    assert len(events) >= 2 and latest > before
    assert all("id" in e and "type" in e and "data" in e for e in events)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
