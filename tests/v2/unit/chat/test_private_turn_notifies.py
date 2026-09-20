"""一个私聊回合写完之后，别的窗口能不能知道。

2026-08-06 要求：「智能体代理对话不会自动刷新」。

根因不在前端。群聊转录落库时 ``ConversationStore.append_message`` 会广播一个
游标，而**私聊的正文根本不经过那张表** —— 它落在 LangGraph 的 checkpointer 里。
于是私聊这条路上一个通知都没有，那一页只能靠「本浏览器自己发完一条」才刷新。

这份测试钉的是**接线本身**：``ConversationEngine._finish`` 每完成一个回合都要
广播一次。上面那份 ``api/test_transcript_multi_viewer_sync.py`` 钉的是广播的
**内容与形状**（不带正文、不借用群聊的 scope）；两边合起来才是一条完整的链路，
只有一边的话，「函数存在」和「有人调它」这两件事里总有一件没人守。

从仓库根运行::

    .venv-v2-py313/Scripts/python.exe -m pytest \
        tests/v2/unit/chat/test_private_turn_notifies.py -q
"""
from __future__ import annotations

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

from mast.chat.engine import ConversationEngine  # noqa: E402


class _Store:
    """只回答 ``_finish`` 会问的那两件事。"""

    def __init__(self, raises: bool = False):
        self.touched: list = []
        self._raises = raises

    def touch(self, conversation_id, *, preview=None, title=None):
        if self._raises:
            raise RuntimeError("db is locked")
        self.touched.append((conversation_id, preview, title))


@pytest.fixture()
def published(monkeypatch) -> list:
    seen: list = []
    monkeypatch.setattr("mast.chat.store.publish_private_turn_finished",
                        lambda cid, aid: seen.append((cid, aid)))
    return seen


def _engine(store) -> ConversationEngine:
    eng = ConversationEngine.__new__(ConversationEngine)
    eng._store = store
    eng._active_lock = __import__("threading").Lock()
    eng._active_threads = set()
    eng._record_sink = None
    return eng


def test_every_finished_turn_announces_itself(published):
    eng = _engine(_Store())
    eng._finish("conv-1", {"agent_id": "instrument_control", "title": "已有标题",
                           "last_message_preview": "前一轮"},
                "thread-1", "扫一张图",
                [{"role": "assistant", "content": "好的"}])
    assert published == [("conv-1", "instrument_control")]


def test_the_announcement_survives_a_failed_touch(published):
    """落库那一步失败了，通知照发。

    两件事的失败模式相反：touch 失败只是会话列表的排序戳没更新；不发通知则是
    别的窗口从此看不到这一轮。把通知写在 touch 的 try 里面（一个很自然的写法）
    会让前者静默吃掉后者。
    """
    eng = _engine(_Store(raises=True))
    eng._finish("conv-2", {"agent_id": "literature", "title": "t",
                           "last_message_preview": "p"},
                "thread-2", "查文献",
                [{"role": "assistant", "content": "找到 3 篇"}])
    assert published == [("conv-2", "literature")]


def test_a_broken_notifier_never_breaks_the_turn(monkeypatch):
    """通知抛了，回合照样算完成。

    这一轮对话是用户的东西；一条通知不值得赔上它。
    """
    def boom(_cid, _aid):
        raise RuntimeError("bus exploded")

    monkeypatch.setattr("mast.chat.store.publish_private_turn_finished", boom)
    store = _Store()
    eng = _engine(store)
    eng._finish("conv-3", {"agent_id": "ic", "title": "t",
                           "last_message_preview": "p"},
                "thread-3", "hi", [{"role": "assistant", "content": "ok"}])
    assert store.touched, "回合的正常记账被一条通知拖垮了"


def test_a_conversation_without_an_agent_id_still_announces(published):
    """``agent_id`` 缺失不是不发通知的理由。

    前端按 ``conversation_id`` 过滤，``agent_id`` 只是附带信息 —— 因为缺一个
    附带字段就整条不发，等于让一行历史遗留数据关掉这一页的刷新。
    """
    eng = _engine(_Store())
    eng._finish("conv-4", {"title": "t", "last_message_preview": "p"},
                "thread-4", "hi", [{"role": "assistant", "content": "ok"}])
    assert published == [("conv-4", "")]


# ── 出错的那一轮也要收尾（顺带查到）──────────────────────────
#
# 上面四条钉的是「_finish 一被调用就会广播」。这一条钉的是另一半：**出错的路径
# 有没有调它**。两者缺一不可 —— 广播函数完好无损、而某条路径压根不走它，症状
# 与 #31 一模一样，且上面每一条都是绿的。
#
# ``stream_turn``（打字聊天）的 except 分支一直调 _finish；``stream_events``
# （语音走的那条）原本直接 return。于是一轮出错的语音对话既不 touch（会话列表的
# updated_at 不动）也不广播，两个信号源同时缺席，而用户刚说的那句话已经进了
# checkpointer —— 别的窗口再也看不到它。


class _ExplodingGraph:
    def stream(self, *_a, **_kw):
        raise RuntimeError("模型连接断了")


def _voice_engine(finished: list) -> ConversationEngine:
    import threading

    eng = ConversationEngine.__new__(ConversationEngine)
    eng._store = _Store()
    eng._store.get = lambda cid: {"agent_id": "instrument_control",
                                  "thread_id": "thread-voice"}
    eng._active_lock = threading.Lock()
    eng._active_threads = set()
    eng._record_sink = None
    eng._tl = threading.local()
    eng._graph_for = lambda _aid: _ExplodingGraph()
    eng._config_for = lambda _aid, _g, _tid: {}
    eng._register_abort = lambda *_a: None
    eng._unregister_abort = lambda *_a: None
    eng._set_turn_context = lambda *_a: None
    eng._clear_turn_context = lambda *_a: None
    eng._new_run_id = lambda _tid: "run-1"
    eng._stale_interrupt_resume = lambda _g, _cfg: None
    eng._state_messages = lambda _g, _cfg: []
    eng._finish = lambda *a: finished.append(a)
    return eng


def test_a_voice_turn_that_errors_still_finishes():
    """语音回合出错 → 仍然 _finish（因而仍然 touch + 广播）。

    这里**不**断言广播本身 —— 上面四条已经钉死「_finish ⇒ 广播」。这一条只回答
    「那条路径走不走 _finish」，这是它原来唯一缺的东西。
    """
    finished: list = []
    eng = _voice_engine(finished)
    events = list(eng._stream_events_impl("conv-voice", "把针退回来"))

    assert any(e.get("type") == "error" for e in events), \
        f"出错事件没发出去：{events}"
    assert finished, "出错的语音回合没有收尾 —— 别的窗口永远看不到用户刚说的那句话"
    cid, conv, thread_id, user_text, _final = finished[0]
    assert cid == "conv-voice"
    assert thread_id == "thread-voice"
    assert user_text == "把针退回来"


def test_the_voice_error_path_does_not_wedge_the_conversation():
    """出错收尾不许把会话锁在「进行中」。

    ``_finish`` 在这份测试里是替身，所以真正的 discard 只能来自那条路径自己的
    清理。少了它，下一次开口会得到「该会话已有进行中的回合」，而这个状态没有
    任何界面能解除。
    """
    finished: list = []
    eng = _voice_engine(finished)
    list(eng._stream_events_impl("conv-voice", "hi"))
    assert eng._active_threads == set(), \
        f"会话被锁死在进行中：{eng._active_threads}"
