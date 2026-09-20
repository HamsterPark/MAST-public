"""判决席位 —— **「判不了」的三种来源里,只有一种是判决**。

M3-a。这一组测的全是那条区分:``decide_route`` 从不抛异常,任何失败都折进
``escape`` 路由 —— 对 composite 工作流是对的(降级到安全出口),对闸门**不够**:
一道 llm 闸门能发出 ``detour``,那是半夜把用户叫起来换样品、以小时计。
一次 provider 500 不该有这个权力。

一次网络都不打:``decide`` 与 ``log`` 全注入。
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import pytest

_MASTV2_ROOT = str(Path(__file__).resolve().parents[4] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)

from mast.conduct import llm_seat  # noqa: E402
from mast.conduct.llm_seat import SeatUnavailable, make_decide_route  # noqa: E402

NODE = {"id": "n", "responsibility": "判这一段还值不值得接着测",
        "routes": {"go": "继续", "hold": "停下来问人"}, "escape": "hold"}


class _Log:
    def __init__(self):
        self.rows: list[dict] = []

    def __call__(self, rec: dict) -> None:
        self.rows.append(dict(rec))


def _seat(decide, *, log=None, deadline_s=5.0, model_factory=None, context=None):
    return make_decide_route(
        decide=decide, log=log or _Log(), deadline_s=deadline_s,
        model_factory=model_factory or (lambda node: object()),
        context=context)


# ── 判决真的发生了 ───────────────────────────────────────────────────────

def test_a_real_decision_comes_back_untouched():
    """席位不改判决,只把它端出来。"""
    seat = _seat(lambda node, inputs, model=None: {
        "route": "go", "reason": "两个偏压都有分辨", "escaped": False,
        "parse_path": "structured", "model": "kimi-k3", "duration_ms": 812})
    out = seat(NODE, {"n_resolved": 2})
    assert out["route"] == "go"
    assert out["model"] == "kimi-k3"


def test_a_real_abstention_keeps_the_specs_escape_route():
    """``uncertain`` = 模型看了证据说自己判不了。**那是一次判决**,走 spec 声明
    的 escape 路由(spec 已禁止它映射到 pass)—— 不该被席位升级成异常。"""
    seat = _seat(lambda node, inputs, model=None: {
        "route": "hold", "reason": "证据不够", "escaped": True,
        "escape_reason": "uncertain", "parse_path": "structured"})
    out = seat(NODE, {})
    assert out["route"] == "hold" and out["escaped"] is True


# ── 判决**没有**发生的三种形状 ───────────────────────────────────────────

def test_an_off_enum_answer_is_not_a_judgement():
    """模型答了一个闭集外的名字 ⇒ 判决器坏了,不是「它选了 escape」。

    ``decide_route`` 会把它折进 escape 路由,而那条路由可能通向 ``detour``。
    """
    seat = _seat(lambda node, inputs, model=None: {
        "route": "hold", "escaped": True, "escape_reason": "off_enum"})
    with pytest.raises(SeatUnavailable) as e:
        seat(NODE, {})
    assert "off_enum" in str(e.value)


def test_a_provider_failure_is_not_a_judgement():
    seat = _seat(lambda node, inputs, model=None: {
        "route": "hold", "escaped": True,
        "escape_reason": "error: 502 Bad Gateway"})
    with pytest.raises(SeatUnavailable) as e:
        seat(NODE, {})
    assert "502" in str(e.value)


def test_no_api_key_means_no_seat_not_a_verdict():
    """建不出模型(一个 provider key 都没配)⇒ 席位不存在 ⇒ 判不了。"""
    def _boom(node):
        raise RuntimeError("no provider has an API key configured")

    seat = _seat(lambda node, inputs, model=None: {"route": "go"},
                 model_factory=_boom)
    with pytest.raises(SeatUnavailable) as e:
        seat(NODE, {})
    assert "API key" in str(e.value)


def test_a_stalled_provider_does_not_park_the_director_thread():
    """墙钟上限:一次卡住的 provider 连接不该把指挥线程停在一道闸上。

    急停与暂停都在**步边界**生效,而一道停等的闸门就是一个没有边界的地方。
    """
    started = threading.Event()

    def _hang(node, inputs, model=None):
        started.set()
        time.sleep(30.0)        # 测试里没人会等到它
        return {"route": "go"}

    seat = _seat(_hang, deadline_s=0.3)
    t0 = time.perf_counter()
    with pytest.raises(SeatUnavailable) as e:
        seat(NODE, {})
    elapsed = time.perf_counter() - t0
    assert started.is_set()
    assert elapsed < 5.0, "席位停等了 —— 墙钟上限没生效"
    assert "超过" in str(e.value)


def test_a_stalled_model_construction_is_bounded_too():
    """**建模型也在墙钟里面。**

    ``make_chat_model`` 会 import 一整条 langchain provider 链、读密钥文件 ——
    这些一样能停住调用它的那条线程,而调用它的正是指挥线程。一道只挡住了「判决」
    那一步的超时,挡不住卡在「建模型」那一步的失败:两步都能停住 tick,而当时
    只有第二步被围起来了。
    """
    def _hang(node):
        time.sleep(30.0)
        return object()

    t0 = time.perf_counter()
    with pytest.raises(SeatUnavailable) as e:
        _seat(lambda node, inputs, model=None: {"route": "go"},
              deadline_s=0.3, model_factory=_hang)(NODE, {})
    assert time.perf_counter() - t0 < 5.0
    assert "建判决模型" in str(e.value), "两句话要分得开:配置问题 vs provider 问题"


def test_a_decide_route_that_raises_is_carried_back_as_unavailable():
    """``decide_route`` 号称从不抛;万一抛了,也必须落成「判不了」而不是崩掉
    指挥线程。"""
    def _boom(node, inputs, model=None):
        raise ValueError("schema 建不出来")

    with pytest.raises(SeatUnavailable) as e:
        _seat(_boom)(NODE, {})
    assert "schema" in str(e.value)


def test_a_non_dict_answer_is_unavailable():
    with pytest.raises(SeatUnavailable):
        _seat(lambda node, inputs, model=None: "go")(NODE, {})


# ── 决策日志 ─────────────────────────────────────────────────────────────

def test_the_seat_writes_the_decision_log_because_decide_route_does_not():
    """``llm_node.decide_route`` **自己不写** JSONL —— 写的是调用方
    (``composite/interpreter._walk_llm``)。所以「接上就白送一份审计」是假的:
    席位不写,conduct 的每一次判决都不会留下任何一行。"""
    from mast.skills.composite import llm_node

    # 判据用 ``co_names``(该函数访问过的名字,取自**已加载的 code object**),
    # 不用 ``inspect.getsource`` —— 后者按 import 时记下的行号去切当前文件,
    # 别人在同一个文件上方插几行就切错位置,而这条是 ``not in``,错位的表现是
    # **假绿**:它会去检查一个完全不同的函数,当然找不到 log_decision。
    reads = llm_node.decide_route.__code__.co_names
    # **自检**:先证明这个判据抓得住「调用一个模块级函数」这种访问。少了它,
    # 下面那条 not-in 会在「co_names 根本抓不到这类调用」时同样通过。
    assert "describe_model" in reads, (
        f"判据失去区分力 —— decide_route 明明调 describe_model,而 co_names 里"
        f"没有它。换判法,别让这条绊线空转。它访问的名字:{sorted(reads)}")
    assert "log_decision" not in reads, (
        f"decide_route 开始自己写日志了 —— 那 llm_seat 这一份就成了重复记账,"
        f"去核对是不是该把席位这边的去掉。它访问的名字:{sorted(reads)}")

    log = _Log()
    seat = _seat(lambda node, inputs, model=None: {
        "route": "go", "reason": "ok", "escaped": False,
        "parse_path": "structured", "model": "kimi-k3"}, log=log)
    seat(NODE, {"n_resolved": 2})
    assert len(log.rows) == 1
    rec = log.rows[0]
    assert rec["mechanism"] == llm_seat.MECHANISM
    assert rec["route"] == "go" and rec["model"] == "kimi-k3"
    assert rec["inputs"] == {"n_resolved": 2}
    assert rec["responsibility"] == NODE["responsibility"]


def test_failures_are_logged_too():
    """只记成功的那些,日志就会显得判决器从不出错 —— 而那正是最该留下的一类
    记录(将来 router-graduation 分类器的负样本)。"""
    log = _Log()
    seat = _seat(lambda node, inputs, model=None: {
        "route": "hold", "escaped": True, "escape_reason": "error: 502"}, log=log)
    with pytest.raises(SeatUnavailable):
        seat(NODE, {})
    assert len(log.rows) == 1
    assert "502" in log.rows[0]["unavailable"]
    # 故障不许被折叠成一个具体的值:``decide_route`` 把失败折进了 ``hold``,
    # 而那不是模型选的。它留在 ``folded_route`` 上供追查,``route`` 是 None。
    assert log.rows[0]["route"] is None, "判决没发生,route 不该是一个名字"
    assert log.rows[0]["folded_route"] == "hold"


def test_a_broken_log_never_blocks_a_decision():
    def _boom(rec):
        raise OSError("磁盘满了")

    out = _seat(lambda node, inputs, model=None: {"route": "go"}, log=_boom)(
        NODE, {})
    assert out["route"] == "go"


def test_context_is_audit_only_and_never_reaches_the_prompt():
    """上下文只进日志。闸门问什么由 spec 的 ``responsibility`` 定死 ——
    运行时上下文改写不了那个问题。"""
    seen: list[dict] = []

    def _decide(node, inputs, model=None):
        seen.append({"node": dict(node), "inputs": dict(inputs)})
        return {"route": "go"}

    log = _Log()
    seat = _seat(_decide, log=log,
                 context=lambda: {"conduct_id": "c-1", "spec_id": "synthetic_sample_v1"})
    seat(NODE, {"n": 1})
    assert log.rows[0]["conduct_id"] == "c-1"
    # 送进模型的只有 node 与 inputs,一个字的 conduct 上下文都没有
    assert seen[0]["inputs"] == {"n": 1}
    assert "conduct_id" not in seen[0]["node"]


def test_a_broken_context_does_not_block_a_decision():
    def _boom():
        raise RuntimeError("store 读不到")

    out = _seat(lambda node, inputs, model=None: {"route": "go"},
                context=_boom)(NODE, {})
    assert out["route"] == "go"


# ── 模型带 request_timeout ──────────────────────────────────────────────

def test_the_default_model_is_built_with_a_request_timeout(monkeypatch):
    """两道超时,不是一道:``request_timeout`` 管单次 HTTP,墙钟管**我们**。

    只有墙钟时,一条卡住的连接会把那个 daemon 线程永久留在进程里;只有
    ``request_timeout`` 时,一次 SDK 层的重试链仍然能把 tick 停住。
    """
    seen: dict = {}

    def _fake(agent=None, **kw):
        seen.update(kw)
        seen["agent"] = agent
        return object()

    import mast.agents._shared.models as models_mod
    monkeypatch.setattr(models_mod, "make_chat_model", _fake)
    llm_seat._default_model_factory({})
    assert seen["agent"] == "orchestrator"
    assert seen["request_timeout"] == llm_seat.DEFAULT_REQUEST_TIMEOUT_S
    assert seen["request_timeout"] < llm_seat.DEFAULT_DEADLINE_S, (
        "单次 HTTP 上限该比墙钟小 —— 否则我们永远先超时,拿到的是「等够了」"
        "而不是 provider 那句具体的话")
