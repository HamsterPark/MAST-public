"""注入台账：抓包里那一大段 system，到底是谁塞的？

## 三条真正在防的事

1. **归属跨不过缓存中间件。** ``langchain_anthropic`` 的
   ``_tag_system_message`` 末尾是 ``return SystemMessage(content=new_content)``
   —— 它**重建了对象**，``additional_kwargs`` 一起没了。那个中间件在七张图里
   都排在全部注入器**内侧**，所以凡 ``isinstance(model, ChatAnthropic)``
   （Claude / MiniMax）的构建，挂在对象上的账走到模型回调前就消失。默认模型是
   Kimi（不挂它），于是本地测得好好的，切到 Claude 静默失效。
   ``test_block_map_survives_a_rebuilt_system_message`` 就是这一条的回归钉。
2. **token 被字符数冒充。** 折算出来的数字看起来和真的一模一样，而它在两处骗
   人：中英混排的 tokenizer 差得远，以及缓存命中根本不体现在字符里。取不到就
   是 ``None`` + ``tokens_source="unavailable"``。
3. **易变块把 system 弄脏。** 记忆召回 / 心愿单回程改挂最后一条 human 之后，
   system 必须**逐轮逐字节相同** —— 那是这次搬家的目的，不是副作用。

Run from repo root::

    .venv-v2-py313/Scripts/python.exe -m pytest \\
        tests/v2/unit/prompts/test_inject_ledger.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_REPO = Path(__file__).resolve().parents[4]
_MASTV2 = str(_REPO / "MASTv2")
if sys.path and sys.path[0] != _MASTV2:
    while _MASTV2 in sys.path:
        sys.path.remove(_MASTV2)
    sys.path.insert(0, _MASTV2)

from langchain_core.messages import (  # noqa: E402
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from mast.agents._shared import inject  # noqa: E402
from mast.prompts import capture, ledger  # noqa: E402


@pytest.fixture(autouse=True)
def _clean():
    ledger.get_ledger().clear()
    capture.get_ring().reset_for_tests()
    yield
    ledger.get_ledger().clear()
    capture.get_ring().reset_for_tests()


def _req(system="BASE", human="hi"):
    return SimpleNamespace(system_message=SystemMessage(content=system),
                           messages=[HumanMessage(content=human)])


# ── 1. 拼接与归属 ───────────────────────────────────────────────────────

def test_blocks_accumulate_with_correct_offsets():
    r = inject.append_system_block(_req(), "mw.a", "AAA")
    r = inject.append_system_block(r, "mw.b", "BB")
    text = r.system_message.content
    blocks = r.system_message.additional_kwargs["mast_blocks"]
    assert [b["id"] for b in blocks] == [inject.BASE_BLOCK_ID, "mw.a", "mw.b"]
    for b in blocks:
        assert text[b["start"]:b["end"]] == {"mw.a": "AAA", "mw.b": "BB"}.get(
            b["id"], "BASE")


def test_an_empty_block_leaves_no_trace():
    """「这一块这次没有内容」不是错误，也不该在账上留痕。"""
    r = inject.append_system_block(_req(), "mw.a", "")
    assert r.system_message.content == "BASE"
    assert "mast_blocks" not in r.system_message.additional_kwargs


def test_human_block_lands_on_the_last_human_and_leaves_system_alone():
    r = inject.append_human_block(_req(), "mw.live_state", "LIVE")
    assert r.system_message.content == "BASE"
    assert r.messages[-1].content == "hi\n\nLIVE"
    ids = [b["id"] for b in r.messages[-1].additional_kwargs["mast_blocks"]]
    assert ids == [inject.HUMAN_BASE_BLOCK_ID, "mw.live_state"]


def test_human_block_returns_none_when_there_is_no_human_turn():
    """返回 None 而不是悄悄退回 system —— 「往哪退」是调用方的判断。"""
    r = SimpleNamespace(system_message=SystemMessage(content="S"), messages=[])
    assert inject.append_human_block(r, "mw.a", "X") is None


def test_operator_text_is_not_labelled_as_an_injection():
    """用户自己说的话不是「注入」，账上要看得出区别。"""
    r = inject.append_human_block(_req(human="扫一张图"), "mw.a", "X")
    first = r.messages[-1].additional_kwargs["mast_blocks"][0]
    assert first["id"] == inject.HUMAN_BASE_BLOCK_ID != inject.BASE_BLOCK_ID


def test_multimodal_system_content_is_not_stringified_away():
    r = SimpleNamespace(
        system_message=SystemMessage(content=[{"type": "text", "text": "BASE"}]),
        messages=[])
    out = inject.append_system_block(r, "mw.a", "AAA")
    assert isinstance(out.system_message.content, list)
    assert out.system_message.content[-1] == {"type": "text", "text": "AAA"}


# ── 2. F2 的回归钉：缓存中间件重建对象之后还认得吗 ──────────────────────

def test_block_map_survives_a_rebuilt_system_message():
    """**这是本模块存在的理由。**

    对真的 ``AnthropicPromptCachingMiddleware._tag_system_message`` 跑一遍，
    然后确认归属还查得到。它重建 SystemMessage，additional_kwargs 一起没了 ——
    所以台账必须按**内容**寻址，不能挂在对象上。
    """
    r = inject.append_system_block(_req(), "mw.tip_context", "TIP")
    original = r.system_message

    from langchain_anthropic.middleware.prompt_caching import _tag_system_message

    rebuilt = _tag_system_message(original, {"type": "ephemeral"})

    assert rebuilt.additional_kwargs.get("mast_blocks") is None, (
        "上游不再重建对象了？那这条回归钉要重写 —— 但**别删**：它记的是"
        "「归属不能挂在对象上」这个结论。")
    flat = ledger.normalize_text(rebuilt.content)
    hit = ledger.lookup(flat)
    assert hit and [b.id for b in hit] == [inject.BASE_BLOCK_ID, "mw.tip_context"]


def test_capture_reads_the_block_map_off_the_ledger():
    r = inject.append_system_block(_req(), "mw.tip_context", "TIP")
    capture.record([r.system_message], source="instrument_control")
    snap = capture.get_ring().list()[0]
    assert snap.blocks_source == "ledger"
    assert [b["id"] for b in snap.messages[0].blocks] == [
        inject.BASE_BLOCK_ID, "mw.tip_context"]


def test_capture_says_none_rather_than_guessing():
    """查不到归属就说查不到 —— 「读不到」被折叠成一个具体的值是这个仓的老毛病。"""
    capture.record([SystemMessage(content="never injected through the helper")],
                   source="x")
    snap = capture.get_ring().list()[0]
    assert snap.blocks_source == "none"
    assert snap.messages[0].blocks is None


# ── 3. token 只认 provider ──────────────────────────────────────────────

def _llm_result(usage=None):
    """一个真的 LLMResult 形状。

    ``usage_metadata`` 是 pydantic 校验过的 —— 少 ``total_tokens`` 会 400，
    所以替身也得补齐；一个连真模型都造不出来的替身，测出来的是它自己。
    """
    if usage:
        full = dict(usage)
        full.setdefault("total_tokens",
                        (full.get("input_tokens") or 0) + (full.get("output_tokens") or 0))
        msg = AIMessage(content="ok", usage_metadata=full)
    else:
        msg = AIMessage(content="ok")
    return SimpleNamespace(generations=[[SimpleNamespace(message=msg)]], llm_output={})


def test_input_tokens_come_only_from_provider_usage():
    cb = capture.make_callback(source="ic", model_id="m", provider="p")
    cb.on_chat_model_start({}, [[HumanMessage(content="hi")]], run_id="r1")
    cb.on_llm_end(_llm_result({"input_tokens": 1234, "output_tokens": 56}),
                  run_id="r1")
    snap = capture.get_ring().list()[0]
    assert snap.input_tokens == 1234
    assert snap.output_tokens == 56
    assert snap.tokens_source == "provider"


def test_missing_usage_is_null_not_estimated():
    cb = capture.make_callback(source="ic", model_id="m", provider="p")
    cb.on_chat_model_start({}, [[HumanMessage(content="hi" * 5000)]], run_id="r2")
    cb.on_llm_end(_llm_result(None), run_id="r2")
    snap = capture.get_ring().list()[0]
    assert snap.input_tokens is None
    assert snap.tokens_source == "unavailable"
    assert snap.total_chars > 0, "字符数照记 —— 它是真的，只是不是 token"


def test_cache_token_details_are_recorded_when_present():
    """把易变块搬去 human 消息之后，要验收的就是这个数。"""
    cb = capture.make_callback(source="ic", model_id="m", provider="anthropic")
    cb.on_chat_model_start({}, [[HumanMessage(content="hi")]], run_id="r3")
    cb.on_llm_end(_llm_result({
        "input_tokens": 900, "output_tokens": 10,
        "input_token_details": {"cache_read": 850, "cache_creation": 0},
    }), run_id="r3")
    snap = capture.get_ring().list()[0]
    assert snap.cache_read_tokens == 850
    assert snap.cache_creation_tokens == 0


# ── 4. run_id 配对，不按 source 猜 ──────────────────────────────────────

def test_on_llm_end_matches_by_run_id_not_by_source():
    """并发下按 source 取「最新一条」会把 A 的回复贴到 B 的快照上，而且无声。"""
    cb = capture.make_callback(source="ic", model_id="m", provider="p")
    cb.on_chat_model_start({}, [[HumanMessage(content="first")]], run_id="A")
    cb.on_chat_model_start({}, [[HumanMessage(content="second")]], run_id="B")
    cb.on_llm_end(_llm_result({"input_tokens": 11, "output_tokens": 1}), run_id="A")

    by_seq = {s.run_id: s for s in capture.get_ring().list()}
    assert by_seq["A"].input_tokens == 11
    assert by_seq["B"].input_tokens is None, "回复贴到了后来的那一条上"


# ── 5. 工具面：从 provider 实收的那份量 ─────────────────────────────────

def test_tool_schema_bytes_are_measured_from_invocation_params():
    """**这一块不在消息里**，而它是 IC 一次请求里最大的一段。"""
    tools = [{"type": "function",
              "function": {"name": f"T{i}", "description": "x" * 200,
                           "parameters": {}}} for i in range(5)]
    cb = capture.make_callback(source="ic", model_id="m", provider="p")
    cb.on_chat_model_start({}, [[HumanMessage(content="hi")]],
                           run_id="r", invocation_params={"tools": tools})
    snap = capture.get_ring().list()[0]
    assert snap.tool_count == 5
    assert snap.tools_chars > 1000
    assert snap.tools_source == "invocation_params"
    assert snap.tools_top and snap.tools_top[0][0].startswith("T")


def test_no_tools_in_invocation_params_is_unavailable_not_zero():
    cb = capture.make_callback(source="ic", model_id="m", provider="p")
    cb.on_chat_model_start({}, [[HumanMessage(content="hi")]], run_id="r")
    snap = capture.get_ring().list()[0]
    assert snap.tools_source == "unavailable"


# ── 6. latest_by_source：忙的时候也答得出「这个 agent 最近收到了什么」──

def test_latest_by_source_is_not_evicted_by_the_ring():
    ring = capture.get_ring()
    capture.record([HumanMessage(content="mine")], source="paper_review")
    for i in range(capture.MAX_SNAPSHOTS + 5):
        capture.record([HumanMessage(content=f"other {i}")], source="instrument_control")
    assert ring.latest("paper_review") is not None, "环把它挤掉了"
    assert ring.latest("paper_review").messages[0].content == "mine"


def test_latest_by_source_ignores_middleware_sub_llm_calls():
    """compaction / tool_refine 的 summarizer 也经 make_chat_model(agent) 构建，
    source 同名。不筛的话，「这个 agent 最近收到了什么」会答成一条**摘要**请求。
    """
    ring = capture.get_ring()
    capture.record([HumanMessage(content="real turn")], source="data_processing",
                   metadata={"langgraph_node": "model"})
    capture.record([HumanMessage(content="summarise this")], source="data_processing",
                   metadata={"langgraph_node": "data_processing.before_model"})
    assert ring.latest("data_processing").messages[0].content == "real turn"


# ── 7. 落点搬家之后 system 真的不动了吗 ─────────────────────────────────

def test_system_message_is_byte_identical_across_turns_with_changing_memory():
    """这是把易变块搬去 human 的**目的**，不是副作用。"""
    from mast.agents._shared.memory_mw import MemoryRecallMiddleware

    calls = {"n": 0}

    class Cog:
        def memory_index(self, **kw):
            return "## MEMORY.md\n- idx"

        def recall(self, query, **kw):
            calls["n"] += 1
            return [SimpleNamespace(title=f"hit-{calls['n']}", text=f"body {calls['n']}",
                                    kind="note", path="n.md")]

    mw = MemoryRecallMiddleware(Cog(), namespace_provider=lambda: "E1")
    seen = []
    for q in ("样品信息", "扫描参数"):
        r = SimpleNamespace(system_message=SystemMessage(content="你是 IC"),
                            messages=[HumanMessage(content=q)])
        out = mw._apply(r)
        seen.append(out.system_message.content)
    assert seen[0] == seen[1] == "你是 IC", (
        "记忆召回又把 system 弄脏了 —— Anthropic 的 cache 断点在 system 末尾，"
        "system 一变，整段 system 加上全部历史每轮都 miss。")


# ── 8. F3 的行为级测试：覆写真的生效了吗 ────────────────────────────────

def test_upstream_header_override_is_actually_used(tmp_path, monkeypatch):
    """2026-08-24 之前这条覆写**从未生效过**。

    ``upstream_mw`` 调 ``resolve("mw.upstream_artifacts.header")`` 少传了必填的
    default，每次 TypeError 被 except 吞掉，永远回退常量。而当时唯一的闸门是
    「overridable 的 id 必须在源码里被引用」—— 字符串级 grep 看得见这个名字，
    看不见它被调错了。所以这里测的是**行为**：设了覆写，注出来的文本要变。
    """
    monkeypatch.setenv("MAST2_PROJECT_ROOT", str(tmp_path))
    from mast.prompts import overrides

    overrides.reset_cache() if hasattr(overrides, "reset_cache") else None
    from mast.agents._shared.upstream_mw import _resolve_header

    default = _resolve_header()
    assert default, "抬头默认值是空的？那这条测试什么也没测"

    overrides.set_override("mw.upstream_artifacts.header", "自定义抬头 XYZ")
    try:
        assert _resolve_header() == "自定义抬头 XYZ"
    finally:
        overrides.set_override("mw.upstream_artifacts.header", "")


def test_resolve_requires_a_default_everywhere_it_is_called():
    """AST 核实参个数 —— 少传 default 就是 F3 那一类死覆写。

    字符串级 grep 看不见这个错误：名字在源码里，调用却是坏的。
    """
    import ast

    bad: list[str] = []
    for path in sorted((_REPO / "MASTv2" / "mast").rglob("*.py")):
        if "prompts" in path.parts and path.name in ("registry.py", "overrides.py"):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):  # pragma: no cover
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            name = getattr(fn, "id", None) or getattr(fn, "attr", None)
            if name not in ("resolve", "resolve_prompt"):
                continue
            if not node.args or not isinstance(node.args[0], ast.Constant):
                continue
            if not str(node.args[0].value).count("."):
                continue        # 不像 prompt id，跳过（别的 resolve 也叫这名字）
            if len(node.args) + len(node.keywords) < 2:
                bad.append(f"{path.relative_to(_REPO)}:{node.lineno} "
                           f"resolve({node.args[0].value!r}) 少了 default")
    assert not bad, "这些 resolve() 调用会 TypeError（多半被 except 吞掉）：\n  " + \
                    "\n  ".join(bad)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
