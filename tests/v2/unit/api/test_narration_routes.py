"""``GET /api/chat/narration`` —— 读端点。

三件事：

1. **路由遮蔽回归。** 断言的是「这个路径解析到了**这个函数**」，不是「返回了 200」。
   本仓两次遮蔽事故（``orchestrator.py:1934-1939``、``documents.py:38-41``）的症状
   都是 **200 + 另一个 handler 的数据** —— 一条只看状态码的测试会全程绿着。
2. **降级口径。** 没接存储 → ``degraded=true`` + 空列表，不是 500。
3. **游标能往前走。** ``latest_seq`` 必须跨过中间那些**非旁白**行；只取旁白行的
   最大 seq 会让轮询每次都把它们重读一遍，且永远读不完。

用的是真的 ``ConversationStore``（建在 ``tmp_path``）+ 真的 ``narrate()`` 写入 ——
没有手写替身，所以没有词汇表可以说错。

    .venv-v2-py313/Scripts/python.exe -m pytest tests/v2/unit/api/test_narration_routes.py -q
"""
from __future__ import annotations

import json
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

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from mast.api.app import create_app  # noqa: E402
from mast.api.routes import chat_narration  # noqa: E402
from mast.chat import narration  # noqa: E402
from mast.chat.store import ConversationStore  # noqa: E402
from mast.core.turn_context import turn_scope  # noqa: E402

CID = "conv-route-test"


@pytest.fixture()
def wired(tmp_path):
    """接了 ConversationStore 的 app（store 在 tmp 上）。

    ``create_app`` 会 ``set_context`` 到**进程级**单例，所以离开时必须还回去 ——
    否则下一份文件里那句「这个进程里没有接存储」的前置条件会莫名其妙不成立
    （已经因此红过一次，而它红得对）。
    """
    from mast.api.context import get_context, set_context

    prev = get_context()
    narration.reset_for_tests()
    store = ConversationStore(tmp_path / "chat" / "mast_conversations.db")
    narration.set_store(store)
    app = create_app(dev_cors=False)
    app.state.ctx.conversation_store = store
    client = TestClient(app)
    yield client, store
    narration.flush(2.0)
    narration.reset_for_tests()
    set_context(prev)


def _write(kind: str, **data) -> None:
    with turn_scope(conversation_id=CID, run_id="run-route"):
        narration.narrate(kind, **data)
    assert narration.flush(3.0)


# ── 1. 路由遮蔽 ─────────────────────────────────────────────────────────


def test_the_path_resolves_to_this_module_not_some_other_handler():
    """遮蔽时错的 handler **也返回 200** —— 所以这里断言的是解析目标本身。"""
    app = create_app(dev_cors=False)
    matches = [r for r in app.routes
               if getattr(r, "path", "") == "/api/chat/narration"]
    assert len(matches) == 1, f"路径注册了 {len(matches)} 次：{matches}"
    endpoint = matches[0].endpoint
    assert endpoint is chat_narration.get_narration, (
        f"/api/chat/narration 解析到了 {endpoint!r}，"
        f"不是 chat_narration.get_narration —— 被别的路由器遮蔽了")


def test_nothing_registered_before_it_can_swallow_this_path():
    """再钉一层：**排在它前面**的路由里，没有一条能匹配 ``/api/chat/narration``。

    上一条断言「现在解析对了」。这一条断言「以后也不会被人在前面插一个
    ``/chat/{something}`` 悄悄夺走」—— 那正是 orchestrator 那次事故的形状
    （被夺走之后，返回的是绑了 ``agent_id="run-task"`` 的另一个 handler）。

    判据是**注册顺序**，因为 Starlette 就是按顺序取第一个匹配的。排在后面的
    带参路由无害 —— SPA 的 ``/{full_path:path}`` 兜底就是刻意注册在最后的，
    把它算成威胁会让这条断言永远红，然后被人删掉。
    """
    app = create_app(dev_cors=False)
    idx = next(i for i, r in enumerate(app.routes)
               if getattr(r, "path", "") == "/api/chat/narration")
    greedy = [r.path for r in app.routes[:idx]
              if getattr(r, "path_regex", None) is not None
              and r.path_regex.match("/api/chat/narration")]
    assert not greedy, f"这些先注册的路由会吃掉本路径：{greedy}"


# ── 1.5 写端在真实接线下找得到存储 ──────────────────────────────────────


def test_the_writer_finds_the_store_through_the_real_app_wiring(tmp_path):
    """未显式 set_store 时，旁白应取得 API 上下文中的同一个 ConversationStore。
    通过对象身份验证启动接线，防止静默无存储或误建第二份历史库。"""
    narration.reset_for_tests()
    assert narration.current_store() is None
    live = ConversationStore(tmp_path / "live" / "mast_conversations.db")
    from mast.api.context import AppContext

    from mast.api.context import get_context, set_context

    prev = get_context()
    ctx = AppContext()
    ctx.conversation_store = live            # bootstrap 里就是这一行
    try:
        create_app(context=ctx, dev_cors=False)
        assert narration.current_store() is live, (
            "写端找不到真实接线里的 ConversationStore —— 旁白会在运行时"
            "静默地一条都不发")
    finally:
        narration.reset_for_tests()
        set_context(prev)   # 进程级 ctx 是全局的：不还回去会污染后面的用例


# ── 2. 降级 ─────────────────────────────────────────────────────────────


def test_unwired_process_degrades_instead_of_500():
    client = TestClient(create_app(dev_cors=False))
    r = client.get("/api/chat/narration", params={"conversation_id": CID})
    assert r.status_code == 200
    body = r.json()
    assert body["degraded"] is True
    assert body["items"] == []


def test_empty_conversation_is_not_degraded():
    """「这个会话还没有旁白」和「这个进程读不到旁白」是两句话，不能同一个字段。"""
    app = create_app(dev_cors=False)
    app.state.ctx.conversation_store = ConversationStore(
        Path(__import__("tempfile").mkdtemp()) / "c.db")
    r = TestClient(app).get("/api/chat/narration",
                            params={"conversation_id": "conv-nothing-here"})
    assert r.json() == {"items": [], "latest_seq": 0, "degraded": False}


# ── 3. 真读到东西 ───────────────────────────────────────────────────────


def test_reads_back_what_narrate_wrote(wired):
    client, _store = wired
    _write("tip_pulse", params={"pulse_v": 10.0, "duration_s": 0.5, "count": 2})

    body = client.get("/api/chat/narration",
                      params={"conversation_id": CID}).json()
    assert body["degraded"] is False
    assert len(body["items"]) == 1
    item = body["items"][0]
    assert item["nk"] == "tip_pulse"
    assert item["tone"] == "warn"
    assert item["has_image"] is False
    # 句子里的数字必须是真的下发值 —— 这是整套设计要保证的那件事。
    assert "10 V" in item["text"] and "500 ms" in item["text"]
    assert item["facts"]["params.pulse_v"] == 10.0
    assert item["facts"]["params.duration_s"] == 0.5


def test_after_seq_is_incremental(wired):
    client, _store = wired
    _write("scan_start")
    first = client.get("/api/chat/narration",
                       params={"conversation_id": CID}).json()
    cursor = first["latest_seq"]
    assert cursor > 0

    again = client.get("/api/chat/narration",
                       params={"conversation_id": CID, "after_seq": cursor}).json()
    assert again["items"] == []
    assert again["latest_seq"] == cursor

    _write("scan_start")
    more = client.get("/api/chat/narration",
                      params={"conversation_id": CID, "after_seq": cursor}).json()
    assert len(more["items"]) == 1
    assert more["latest_seq"] > cursor


def test_cursor_steps_over_non_narration_rows(wired):
    """中间夹着群聊/状态行时，游标必须**跨过去**。

    只取旁白行的最大 seq 的话：一条旁白之后来了 50 条别的行，客户端的游标停在
    旁白那一条，于是每次轮询都把那 50 行重读一遍 —— 而且永远读不完。
    """
    client, store = wired
    _write("scan_start")
    for _ in range(5):
        store.append_message(CID, kind="status", text="别的行")
    body = client.get("/api/chat/narration",
                      params={"conversation_id": CID}).json()
    assert len(body["items"]) == 1
    assert body["latest_seq"] == 6, "游标停在旁白那一行了 —— 后面 5 行会被反复重读"


def test_other_conversations_are_not_returned(wired):
    client, _store = wired
    _write("scan_start")
    with turn_scope(conversation_id="conv-somebody-else", run_id=""):
        narration.narrate("scan_start")
    narration.flush(3.0)

    body = client.get("/api/chat/narration",
                      params={"conversation_id": "conv-somebody-else"}).json()
    assert len(body["items"]) == 1
    mine = client.get("/api/chat/narration",
                      params={"conversation_id": CID}).json()
    assert len(mine["items"]) == 1


# ── 4. 缩略图：原样读盘，不重渲染 ───────────────────────────────────────


def _png_bytes() -> bytes:
    """一张 1×1 的合法 PNG（不需要 matplotlib —— 这条路径上本来就没有它）。"""
    import base64
    return base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")


def test_the_image_endpoint_serves_the_exact_file_the_model_looked_at(wired, tmp_path):
    """**字节相同**。重渲染一张就是 #76/#78：判读配着另一帧的画面。"""
    client, _store = wired
    png = tmp_path / "scan-42_f4.png"
    png.write_bytes(_png_bytes())
    with turn_scope(conversation_id=CID, run_id="run-route"):
        narration.narrate("scan_milestone", frac=0.5, summary_zh="针尖状态良好。",
                          image={"src": str(png), "origin": "milestone_png"})
    assert narration.flush(3.0)

    seq = client.get("/api/chat/narration",
                     params={"conversation_id": CID}).json()["items"][0]["seq"]
    r = client.get(f"/api/chat/narration-image/{seq}",
                   params={"conversation_id": CID})
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert r.content == png.read_bytes()
    assert "immutable" in r.headers.get("cache-control", "")


def test_a_missing_file_404s_instead_of_serving_a_placeholder(wired, tmp_path):
    """图被清掉了（artifacts 是可清理目录）⇒ 404。

    占位图会被读成「这一帧本来就长这样」。前端 onError 说的是
    「这一帧的画面取不到了」—— 那是一句真话，而占位图不是。
    """
    client, _store = wired
    with turn_scope(conversation_id=CID, run_id="run-route"):
        narration.narrate("scan_milestone", frac=0.5,
                          image={"src": str(tmp_path / "gone.png"),
                                 "origin": "milestone_png"})
    assert narration.flush(3.0)
    seq = client.get("/api/chat/narration",
                     params={"conversation_id": CID}).json()["items"][0]["seq"]
    assert client.get(f"/api/chat/narration-image/{seq}",
                      params={"conversation_id": CID}).status_code == 404


def test_a_row_with_no_image_404s(wired):
    client, _store = wired
    _write("scan_start")
    seq = client.get("/api/chat/narration",
                     params={"conversation_id": CID}).json()["items"][0]["seq"]
    assert client.get(f"/api/chat/narration-image/{seq}",
                      params={"conversation_id": CID}).status_code == 404


def test_the_image_endpoint_only_serves_paths_we_wrote_ourselves(wired, tmp_path):
    """``origin`` 是**白名单**。

    ``meta.image.src`` 是一条写在 DB 行里的绝对路径，而这个端点会把那条路径的
    内容发给浏览器。所以放行的判据必须是「这条路径是我们自己写下的那一类」，
    不能是「它看起来像张图」。``sxm`` 那一档还没有生产方，现在就不开门。
    """
    client, store = wired
    png = tmp_path / "secret.png"
    png.write_bytes(_png_bytes())
    # 直接绕过 narrate()，模拟一条被改过 meta 的行。
    seq = store.append_message(
        CID, kind=narration.NARRATION_KIND, agent_id="", role="narration",
        text="伪造的一条",
        meta=json.dumps({"v": 1, "nk": "scan_milestone", "anchor": -1,
                         "image": {"src": str(png), "origin": "sxm"}}))
    assert client.get(f"/api/chat/narration-image/{seq}",
                      params={"conversation_id": CID}).status_code == 404


def test_the_image_endpoint_needs_the_right_conversation(wired, tmp_path):
    client, _store = wired
    png = tmp_path / "f.png"
    png.write_bytes(_png_bytes())
    with turn_scope(conversation_id=CID, run_id="run-route"):
        narration.narrate("scan_milestone", frac=0.5,
                          image={"src": str(png), "origin": "milestone_png"})
    assert narration.flush(3.0)
    seq = client.get("/api/chat/narration",
                     params={"conversation_id": CID}).json()["items"][0]["seq"]
    assert client.get(f"/api/chat/narration-image/{seq}",
                      params={"conversation_id": "conv-not-mine"}).status_code == 404


def test_the_image_path_resolves_to_this_module_too():
    app = create_app(dev_cors=False)
    matches = [r for r in app.routes
               if getattr(r, "path", "") == "/api/chat/narration-image/{seq}"]
    assert len(matches) == 1
    assert matches[0].endpoint is chat_narration.get_narration_image


def test_a_narration_row_never_reaches_the_agent(wired):
    """旁白**不在 message channel 上** —— 这是「agent 看不见」的结构证据。

    钉两件事，因为它们会各自独立地坏掉：
      · 行的 ``kind`` 不是 ``message`` ⇒ ``agent_activity()``（per-agent 群聊活动）
        查不到它；
      · 行的 ``agent_id`` 是空串 ⇒ 前端 ``transcriptCursorFor`` 的
        ``agent_id === 当前 agent`` 判据永远不成立，代理对话页不会因为一条旁白
        去重刷群聊转录（那是一条**假因果**）。
    """
    _client, store = wired
    _write("scan_start")
    rows = store.messages_since(CID, 0)
    assert [r["kind"] for r in rows] == [narration.NARRATION_KIND]
    assert rows[0]["agent_id"] == ""
    assert store.agent_activity("instrument_control") == []
    assert store.agent_activity("") == []
