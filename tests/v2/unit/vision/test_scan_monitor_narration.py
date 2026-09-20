"""扫描里程碑 → 对话流里的旁白（带那一帧的缩略图）。

设计文档：``docs/v2/design/chat_narration_sidechannel.md`` 阶段 3

钉四件事：

1. **没有 sink 时，与今天逐字节相同。** 旁白是加进一条已经跑了很久的链路里的，
   「加了个开关但默认路径变了」是最贵的那种回归。
2. **图是原样引用的那张 PNG**，不是重渲染的。``image.src`` 必须等于
   ``_persist_frame_png`` 落下的那个 ``frame_path`` —— 那是模型**真正看过**的
   那一帧。重渲染就是 #76/#78：扫描途中每一次判读都配着上一张图。
3. **「扫完了」和「提前停下」是两句话。** 、以及「『停止』≠『达标』」
   四根因之一。判据不是时间到了，是扫描缓冲的 NaN 前沿（``_confirm_complete``）。
4. **裸线程读不到 ContextVar。** 这是本阶段唯一一个「不写测试就一定会被写错」的
   地方：在监视器线程里调无参 ``narrate()`` 会静默 no-op，看起来像「旁白坏了」。
   所以这里**证明**它是空的，并证明传进去的 sink 在同一根线程上确实能发出来。

替身纪律：判读结果用的是**真的** ``TipCoarseResult``（pydantic，字段有校验），
不是一个手捏的 dict —— 手捏的 dict 可以有真模型永远不会给的字段值。

    .venv-v2-py313/Scripts/python.exe -m pytest \
        tests/v2/unit/vision/test_scan_monitor_narration.py -q
"""
from __future__ import annotations

import sys
import threading
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

import json  # noqa: E402

# 源码级断言一律走它,不用 ``inspect.getsource``(2026-08-15)——
# 后者按 import 那一刻的行号切当前文件,别人同时在改就返回错位切片:
# ``in`` 那半给假红(吵、会被查),``not in`` 那半给**假绿**(不吵、没人会查)。
from tests.v2.srcref import source_of  # noqa: E402

import pytest  # noqa: E402

from mast.chat import narration  # noqa: E402
from mast.chat.store import ConversationStore  # noqa: E402
from mast.core.turn_context import turn_scope  # noqa: E402
from mast.vision.module import TipCoarseResult  # noqa: E402
from mast.vision.scan_monitor import ScanVisionMonitor  # noqa: E402

CID = "conv-scan-narration"


class _Buf:
    """BufferService 的最小替身：只回答监视器会问的那两件事。"""

    def __init__(self) -> None:
        self.events: list = []
        self._n = 0

    def next_seq(self) -> int:
        self._n += 1
        return self._n

    def emit_event(self, event) -> None:
        self.events.append(event)


@pytest.fixture()
def store(tmp_path):
    narration.reset_for_tests()
    st = ConversationStore(tmp_path / "chat" / "c.db")
    narration.set_store(st)
    yield st
    narration.flush(3.0)
    narration.reset_for_tests()


def _monitor(buf, sink=None) -> ScanVisionMonitor:
    return ScanVisionMonitor(
        MagicMock(), scan_id="scan-42", buffer=buf, narration_sink=sink,
        # 真的 vision 判读句子（buffer_summarizer.describe 那一套）走它自己的路；
        # 这里固定一句，让断言钉在「旁白复用了它」而不是钉在措辞上。
        translate=lambda kind, payload: "针尖状态良好。",
    )


def _lines(store: ConversationStore) -> list[dict]:
    assert narration.flush(3.0)
    return [r for r in store.messages_since(CID, 0)
            if r["kind"] == narration.NARRATION_KIND]


COARSE = TipCoarseResult(label="good", confidence=0.91)


# ── 1. 没有 sink = 今天的行为 ───────────────────────────────────────────


def test_without_a_sink_nothing_changes(store):
    buf = _Buf()
    mon = _monitor(buf, sink=None)
    with turn_scope(conversation_id=CID, run_id="r"):
        mon._emit_milestone_event(buf, 0.5, False, COARSE, None,
                                  frame_path="/tmp/x.png")
    assert len(buf.events) == 1, "buffer 事件照发 —— 旁白是加法，不是替换"
    assert _lines(store) == []


# ── 2. 有 sink：8 条里程碑，图是那一帧 ──────────────────────────────────


def test_a_milestone_narrates_with_the_frame_it_analysed(store):
    buf = _Buf()
    sink = narration.bind(CID)
    mon = _monitor(buf, sink=sink)
    mon._emit_milestone_event(buf, 0.5, False, COARSE, None,
                              frame_path="/tmp/scan-42_f4.png")
    rows = _lines(store)
    assert len(rows) == 1
    meta = json.loads(rows[0]["meta"])
    assert meta["nk"] == "scan_milestone"
    assert meta["image"] == {"src": "/tmp/scan-42_f4.png",
                             "origin": "milestone_png"}
    assert "50%" in rows[0]["text"]
    # 判读那句话是**复用**的，不是旁白自己又判了一遍。
    assert "针尖状态良好。" in rows[0]["text"]


def test_all_eight_milestones_come_through(store):
    buf = _Buf()
    mon = _monitor(buf, sink=narration.bind(CID))
    for i, frac in enumerate((0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875), 1):
        mon._emit_milestone_event(buf, frac, False, COARSE, None,
                                  frame_path=f"/tmp/f{i}.png")
    mon._emit_milestone_event(buf, 1.0, True, COARSE, None,
                              frame_path="/tmp/f8.png")
    rows = _lines(store)
    assert len(rows) == 8
    kinds = [json.loads(r["meta"])["nk"] for r in rows]
    assert kinds[:7] == ["scan_milestone"] * 7
    assert kinds[7] == "scan_done"


def test_a_milestone_without_a_frame_still_speaks(store):
    """图没落盘（全黑帧会被 _persist_frame_png 拒绝）时，句子照说，只是没有图。

    「没有图」不该让整条旁白消失 —— 那会让用户在扫描中途看到一段空白，
    而空白和「什么都没发生」长得一模一样。
    """
    buf = _Buf()
    mon = _monitor(buf, sink=narration.bind(CID))
    mon._emit_milestone_event(buf, 0.25, False, COARSE, None, frame_path=None)
    rows = _lines(store)
    assert len(rows) == 1
    assert "image" not in json.loads(rows[0]["meta"])


# ── 3. 「扫完了」≠「提前停下」 ─────────────────────────────────────────


def test_finished_and_stopped_early_are_two_different_sentences(store):
    buf = _Buf()
    mon = _monitor(buf, sink=narration.bind(CID))
    mon._emit_milestone_event(buf, 1.0, True, COARSE, None, frame_path=None)
    done = _lines(store)[-1]

    narration.reset_for_tests()
    st2 = ConversationStore(Path(store._db_path).parent / "c2.db")
    narration.set_store(st2)
    mon2 = _monitor(_Buf(), sink=narration.bind(CID))
    # _emit_incomplete 会去 grab 一帧；这里让它拿不到（MagicMock 池），
    # 那条路径本来就写着「grab 失败也要留下记录」。
    mon2._emit_incomplete(0.42)
    assert narration.flush(3.0)
    early = [r for r in st2.messages_since(CID, 0)
             if r["kind"] == narration.NARRATION_KIND][-1]

    assert json.loads(done["meta"])["nk"] == "scan_done"
    assert json.loads(early["meta"])["nk"] == "scan_stopped_early"
    assert done["text"] != early["text"]
    assert "扫完了" in done["text"]
    assert "提前停下" in early["text"] and "没有" in early["text"], (
        "「提前停下」必须明说它没有被记为完成 —— #143 的教训是"
        "「停止」被渲染成了「达标」")
    assert "42%" in early["text"]
    assert json.loads(done["meta"])["tone"] == "good"
    assert json.loads(early["meta"])["tone"] == "warn"


# ── 4. 裸线程读不到 ContextVar（这是 sink 必须被传进去的理由） ──────────


def test_a_bare_thread_cannot_see_the_conversation_but_a_bound_sink_can(store):
    """同一根裸线程上：无参 ``narrate()`` 静默 no-op，绑好的 sink 发得出来。

    这条断言的价值不在「sink 能用」，在**证明另一半是坏的** —— 如果哪天有人把
    ``ScanVisionMonitor`` 里的 ``sink.narrate(...)`` 改成模块级 ``narrate(...)``，
    旁白会一条不剩地消失，而**不会有任何报错**。
    """
    seen: dict = {}

    def body(sink):
        from mast.core.turn_context import current_turn

        seen["cid_in_thread"] = current_turn().get("conversation_id")
        narration.narrate("scan_start")          # 无参 —— 应该 no-op
        sink.narrate("scan_milestone", frac=0.75, summary_zh="针尖状态良好。")

    with turn_scope(conversation_id=CID, run_id="r"):
        sink = narration.bind()                  # 在**有** ContextVar 的线程上绑
        assert bool(sink) is True
        t = threading.Thread(target=body, args=(sink,))
        t.start()
        t.join(5)

    assert seen["cid_in_thread"] is None, (
        "裸线程居然读到了 conversation_id —— 那 bind() 这一整套就没必要了，"
        "但真机上（langgraph 之外的线程）它读不到")
    rows = _lines(store)
    assert len(rows) == 1
    assert json.loads(rows[0]["meta"])["nk"] == "scan_milestone"


def test_the_sink_is_bound_at_start_not_inside_the_thread():
    """``start_scan_vision_monitor`` 必须在**它自己这根线程**上绑。

    钉的是接线位置本身：绑定发生在起 daemon 之前。放到监视器线程里绑，
    上面那条测试证明的正是「绑出来的是个空的」。
    """
    import inspect

    from mast.vision import scan_monitor as sm

    src = source_of(sm.start_scan_vision_monitor)
    assert "_bind_narration_sink()" in src
    bind_src = source_of(sm._bind_narration_sink)
    assert "narration.bind()" in bind_src
    # 监视器**内部**不许自己去取会话 —— 那是裸线程，取到的永远是空的。
    mon_src = source_of(sm.ScanVisionMonitor)
    assert "class ScanVisionMonitor" in mon_src, (
        "取到的不是 ScanVisionMonitor 的源码 —— 下面那条 not in 会恒真")
    assert "current_turn" not in mon_src
