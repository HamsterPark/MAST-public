"""A bounded ring of the last REAL model requests — "what did the agent get?".

WHY A CAPTURE AND NOT A DRY RUN
===============================
The obvious way to answer "what does the agent receive" is to re-assemble the
context on demand: build a fake request, run it through the middlewares, print
the result. We deliberately do not do that. A dry run cannot reproduce the live
hardware read, the conversation history, the tool that just failed, or the
middleware ordering of the graph that actually ran — so it produces text that
*resembles* the injection without *being* it. That is exactly the failure mode
this feature exists to prevent: someone debugging a real incident against
plausible invented text.

So instead: hook the one place every model call passes through — the LangChain
callback the model factory already attaches for billing — and keep the last few
message lists verbatim. The trade is that a snapshot may be stale (it is
timestamped, and the UI says so) and that nothing appears until a request has
actually run. Both are honest failure modes; a fabricated render is not.

WHAT THIS DOES NOT COVER
========================
The MESSAGE list only. Tool schemas — the ~250 instrument-control tool
definitions the provider is also handed — are bound to the model, not carried in
``on_chat_model_start``'s messages, so they are outside this record. The tool
list is inspectable elsewhere (Agents → 工具). Saying so matters: a reader who
assumed this was the entire payload would draw wrong conclusions about token
cost and about what the model could see.

BOUNDS
======
Memory only, never persisted, never leaves the process. ``MAX_SNAPSHOTS``
requests deep, each message truncated at ``_MAX_MESSAGE_CHARS`` and each
snapshot at ``_MAX_SNAPSHOT_CHARS`` — with the truncation marked and the
original length reported, so a truncated view is never mistaken for a short
prompt. ``get_ring().set_enabled(False)`` is the kill switch (also drops what is
already held).
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field

from mast.prompts import ledger as _ledger
from typing import Any

logger = logging.getLogger(__name__)

#: How many requests deep the ring goes.
#:
#: Was 8, raised to 40 on 2026-08-18 when the chat view started showing "what was
#: injected for THIS turn" inline. 8 is right for the admin inspector (a human
#: looking at the most recent call) and useless for the chat one: a single user
#: turn that runs a few tools issues one model call per tool round, so the turn
#: BEFORE the current one is usually already evicted — the toggle would say
#: "没有留底" on nearly every message.
#:
#: Cost is bounded by construction: ``_MAX_SNAPSHOT_CHARS`` per snapshot, so the
#: worst case is 40 × 200 kB ≈ 8 MB of process memory, never persisted. Real
#: snapshots run far under the cap.
#: 工具密集的回合会产生多次模型调用，较小的环可能在下一回合淘汰上一回合。
#: 200 条 × 200 kB 上限将最坏内存限定为约 40 MB。
#: `_Ring._latest_by_source` 另外保留每个 source 的最新记录，不受环淘汰影响。
MAX_SNAPSHOTS = 200
_MAX_MESSAGE_CHARS = 20_000
_MAX_SNAPSHOT_CHARS = 200_000

#: 工具 schema 的 top-N 明细记多少条。全量 394 个的名字+字节数约 12 kB，
#: 没必要每条快照都背着；top-N 足够回答「谁最大」。
_TOOLS_TOP_N = 12


@dataclass
class CapturedMessage:
    role: str
    content: str
    chars: int          # length BEFORE truncation
    truncated: bool = False
    #: 这条消息里各段的来源（``mast.prompts.ledger`` 的块清单，dict 形式）。
    #: None = 查不到归属 —— 与「没有注入」是两件事，见 ``blocks_source``。
    blocks: list[dict] | None = None


@dataclass
class CapturedRequest:
    """One real model call, as the provider received it."""

    ts: float
    source: str          # agent name ("instrument_control", "adhoc", …)
    model_id: str
    provider: str
    messages: list[CapturedMessage] = field(default_factory=list)
    total_chars: int = 0            # sum of ORIGINAL message lengths
    dropped_messages: int = 0       # messages omitted by the snapshot cap
    #: 块归属是从哪儿查到的：``ledger``（内容寻址，跨得过缓存中间件重建对象）/
    #: ``additional_kwargs``（链内快路）/ ``none``（查不到，如实说）。
    blocks_source: str = "none"

    #: provider **上报的**输入/输出 token。取不到就是 None —— 这里不做任何
    #: 字符数换算，因为换算出来的数字看起来和真的一模一样，而它会在两个地方
    #: 骗人：不同 tokenizer 的中英混排差得远，以及缓存命中根本不体现在字符里。
    input_tokens: int | None = None
    output_tokens: int | None = None
    #: Anthropic 的缓存明细（``usage_metadata["input_token_details"]``）。
    #: 兼容 OpenAI 口径的 provider 多半为 None。
    cache_read_tokens: int | None = None
    cache_creation_tokens: int | None = None
    #: ``provider`` = 上面几个数是 provider 报的；``unavailable`` = 没报。
    tokens_source: str = "unavailable"

    #: **工具 schema 的体量。这是本次请求里最大的一块，而它不在 messages 里。**
    #:
    #: 2026-08-24 之前这个模块的 docstring 写着「工具 schema 不在捕获范围」，
    #: 因为工具是 bind 在模型上的、不走消息。但 ``on_chat_model_start`` 的
    #: ``invocation_params`` 里带着 provider 实际收到的那份 schema
    #: （``ChatOpenAI.bind_tools`` → ``super().bind(tools=…)`` →
    #: ``_get_invocation_params`` 合并 bound kwargs → 回调）。实测 IC 是
    #: **394 个工具、354 k 字符**，约等于它静态提示词的 19 倍 —— 一份不含它的
    #: 「上下文构成」会把最大的那块画成不存在。
    tool_count: int | None = None
    tools_chars: int | None = None
    tools_top: list[tuple[str, int]] = field(default_factory=list)
    tools_source: str = "unavailable"

    #: LangGraph 执行上下文，用来把「主模型调用」与中间件内部的子模型调用
    #: （compaction / tool_refine 的 summarizer 也经 make_chat_model(agent) 构建，
    #: source 同名）分开，以及在群聊 fan-out 下分辨是哪条线程。
    node: str = ""
    checkpoint_ns: str = ""
    thread_id: str = ""
    #: 与 ``on_llm_end`` 配对用。按 source 猜会在并发下贴错人。
    run_id: str = ""

    #: Monotonic id, assigned on ``add``. **A stable handle; the index is not.**
    #:
    #: Index 0 means "newest", so every new model call renumbers every snapshot.
    #: A caller that lists, picks index 3, then fetches index 3 gets a DIFFERENT
    #: request if one call landed in between — and nothing about the result says
    #: so. That race is unobservable in the admin inspector (a human clicking)
    #: and constant in the chat view added 2026-08-18, where the agent is calling
    #: the model while the operator expands a turn. Look up by ``seq`` there.
    seq: int = 0


class _Ring:
    """Thread-safe fixed-size ring of :class:`CapturedRequest`."""

    #: 算「主模型调用」的 LangGraph 节点名。中间件内部的子模型调用跑在
    #: ``<agent>.before_model`` 这类节点里，它们的 source 与 agent 同名，
    #: 不筛就会让 ``latest("data_processing")`` 返回一条**摘要**请求。
    #: 空字符串 = 拿不到 langgraph 元数据（私聊直连、离线测试），按主调用算。
    MAIN_NODES = ("", "model", "agent", "supervisor")

    def __init__(self, maxlen: int = MAX_SNAPSHOTS):
        self._lock = threading.Lock()
        self._items: deque[CapturedRequest] = deque(maxlen=maxlen)
        self._enabled = True
        self._seen = 0
        #: source → 该 source 最近一次**主模型调用**。环挤不掉它。
        #:
        #: 没有它，「这个 agent 最近一次收到了什么」这个问题在忙的时候恰好
        #: 答不了：自主跑一轮约 23 次调用，几个 agent 并行时最早的那个先出局。
        self._latest_by_source: dict[str, CapturedRequest] = {}
        #: run_id → 快照，用于 on_llm_end 精确回贴（不按 source 猜）。
        self._by_run: dict[str, CapturedRequest] = {}

    def add(self, item: CapturedRequest) -> None:
        with self._lock:
            if not self._enabled:
                return
            self._seen += 1
            item.seq = self._seen
            self._items.append(item)
            if item.node in self.MAIN_NODES:
                self._latest_by_source[item.source] = item
            if item.run_id:
                self._by_run[item.run_id] = item
                # run_id 表跟着环走，别无限长。
                if len(self._by_run) > MAX_SNAPSHOTS * 2:
                    live = {i.run_id for i in self._items if i.run_id}
                    live |= {i.run_id for i in self._latest_by_source.values() if i.run_id}
                    self._by_run = {k: v for k, v in self._by_run.items() if k in live}

    def latest(self, source: str) -> CapturedRequest | None:
        """该 source 最近一次主模型调用。没有就是 None（不退而求其次）。"""
        with self._lock:
            return self._latest_by_source.get(source)

    def sources(self) -> list[str]:
        with self._lock:
            return sorted(self._latest_by_source)

    def list(self) -> list[CapturedRequest]:
        """Newest first."""
        with self._lock:
            return list(reversed(self._items))

    def get(self, index: int) -> CapturedRequest | None:
        items = self.list()
        return items[index] if 0 <= index < len(items) else None

    def get_by_seq(self, seq: int) -> CapturedRequest | None:
        """The snapshot with this ``seq``, or None if it has been evicted.

        Returning None for an evicted snapshot is the honest answer and the
        reason this exists: with index lookup the same call returns *a*
        snapshot — just not the one that was asked for.
        """
        with self._lock:
            for item in self._items:
                if item.seq == seq:
                    return item
        return None

    def clear(self) -> None:
        """清空快照。**不动 ``total_seen``** —— 它回答的是「这个进程发生过模型
        调用吗」，那个事实不因为清空展示缓冲而改变。四种空态里
        ``no_calls``（一次都没跑过）与 ``no_calls_for_agent``（跑过但不是它）
        正是靠这个区别分开的。"""
        with self._lock:
            self._items.clear()
            self._latest_by_source.clear()
            self._by_run.clear()

    def reset_for_tests(self) -> None:
        """连 ``total_seen`` 一起归零。**只给测试用。**

        存在的理由：``clear()`` 刻意保留 total_seen（见上），于是一个断言
        「本进程还没跑过模型调用」的测试会被同一个 session 里跑在它前面的任何
        测试污染 —— 而它污染出来的是**另一种合法答案**（no_calls_for_agent），
        看起来完全正常。"""
        with self._lock:
            self._items.clear()
            self._latest_by_source.clear()
            self._by_run.clear()
            self._seen = 0
            self._enabled = True

    def attach_response(self, source: str, message: CapturedMessage,
                        *, run_id: str = "", usage: dict | None = None) -> bool:
        """Append the model's OWN output to the snapshot that call came from.

        Everything else here records what went INTO a call. That is one turn too
        late for a tool call that came out wrong: the offending AIMessage only
        appears in a capture if there is a NEXT request, carrying it as history —
        and on 2026-08-03 the turn that emitted ``p_gain=3`` ended there, so the
        only record of what the model emitted was the tool result it produced.

        2026-08-24：配对改用 ``run_id``。以前按 source 取「最新的一条」，那在
        并发下会把 A 的回复贴到 B 的快照上，而且**不会有任何迹象** —— 群聊
        fan-out 与「一个 agent 同时开着私聊和群聊」都是常态。run_id 由
        langchain 在 ``on_chat_model_start`` / ``on_llm_end`` 两侧给同一个值。
        拿不到 run_id 时退回旧的按-source 猜法，并且**只在那时**是 best-effort。
        """
        with self._lock:
            target: CapturedRequest | None = None
            if run_id:
                target = self._by_run.get(run_id)
            if target is None:
                for item in reversed(self._items):
                    if item.source == source:
                        target = item
                        break
            if target is None:
                return False
            target.messages.append(message)
            if usage:
                target.input_tokens = usage.get("input_tokens")
                target.output_tokens = usage.get("output_tokens")
                target.cache_read_tokens = usage.get("cache_read")
                target.cache_creation_tokens = usage.get("cache_creation")
                target.tokens_source = usage.get("source") or "unavailable"
            return True

    @property
    def enabled(self) -> bool:
        with self._lock:
            return self._enabled

    def set_enabled(self, on: bool) -> None:
        with self._lock:
            self._enabled = bool(on)
            if not on:
                self._items.clear()
                self._latest_by_source.clear()
                self._by_run.clear()

    @property
    def total_seen(self) -> int:
        with self._lock:
            return self._seen


_RING = _Ring()


def get_ring() -> _Ring:
    return _RING


def _role_of(msg: Any) -> str:
    """LangChain message → a role name, without importing message classes."""
    t = getattr(msg, "type", None)
    if isinstance(t, str) and t:
        return t
    return type(msg).__name__.replace("Message", "").lower() or "unknown"


#: 扁平化用 ledger 的那一个实现。**必须是同一个函数**：注入侧按扁平化后的文本
#: 算哈希，抓包侧也按它算，两边差一个空格就永远查不到归属，而且不会报错。
_text_of = _ledger.normalize_text


#: Per-call cap on rendered tool-call arguments. Generous enough for any real
#: hardware call (they are a handful of scalars) and small enough that eight
#: snapshots of them cannot matter against the 200 kB snapshot budget.
_MAX_TOOLCALL_ARGS_CHARS = 1_000


def _render_tool_calls(calls: Any) -> str:
    """``[tool_calls] Name({"a": 1e-12}), Other({})`` — names AND arguments.

    Values are rendered with ``json.dumps`` so a float keeps its exponent
    (``repr`` would too, but json is what the provider actually exchanged, and
    the point of this record is to show what crossed the wire).
    """
    import json

    parts: list[str] = []
    for c in calls or []:
        try:
            name = str(c.get("name", "?"))
            args = c.get("args", None)
            if args is None:
                parts.append(name)
                continue
            rendered = json.dumps(args, ensure_ascii=False, default=str)
            if len(rendered) > _MAX_TOOLCALL_ARGS_CHARS:
                rendered = rendered[:_MAX_TOOLCALL_ARGS_CHARS] + "…(截断)"
            parts.append(f"{name}({rendered})")
        except Exception:  # noqa: BLE001 — fall back to the name alone
            try:
                parts.append(str(c.get("name", "?")))
            except Exception:  # noqa: BLE001
                parts.append("?")
    return "[tool_calls] " + ", ".join(parts)


def _blocks_of(msg: Any, raw: str) -> tuple[list[dict] | None, str]:
    """``(块清单, 来源)``。查不到就 ``(None, "none")`` —— 不猜、不编。"""
    hit = _ledger.lookup(raw)
    if hit:
        return [b.to_dict() for b in hit], "ledger"
    kw = getattr(msg, "additional_kwargs", None)
    if isinstance(kw, dict):
        raw_blocks = kw.get("mast_blocks")
        if isinstance(raw_blocks, list) and raw_blocks:
            return list(raw_blocks), "additional_kwargs"
    return None, "none"


def _measure_tools(invocation_params: Any) -> tuple[int, int, list, str]:
    """``(个数, schema 总字符, top-N, 来源)``，来自 provider **实收**的那份。"""
    try:
        params = invocation_params if isinstance(invocation_params, dict) else {}
        tools = params.get("tools")
        if not tools:
            return 0, 0, [], "unavailable"
        import json
        sizes: list[tuple[str, int]] = []
        total = 0
        for t in tools:
            try:
                blob = json.dumps(t, ensure_ascii=False, default=str)
            except Exception:  # noqa: BLE001
                blob = str(t)
            n = len(blob)
            total += n
            name = ""
            if isinstance(t, dict):
                fn = t.get("function")
                if isinstance(fn, dict):
                    name = str(fn.get("name") or "")
                name = name or str(t.get("name") or t.get("type") or "?")
            sizes.append((name or "?", n))
        sizes.sort(key=lambda kv: kv[1], reverse=True)
        return len(tools), total, sizes[:_TOOLS_TOP_N], "invocation_params"
    except Exception:  # noqa: BLE001
        return 0, 0, [], "unavailable"


def record(messages: Any, *, source: str = "", model_id: str = "",
           provider: str = "", run_id: str = "", metadata: Any = None,
           invocation_params: Any = None) -> None:
    """Store one request's message list. Never raises."""
    try:
        if not _RING.enabled:
            return
        meta = metadata if isinstance(metadata, dict) else {}
        snap = CapturedRequest(ts=time.time(), source=source or "unknown",
                               model_id=model_id, provider=provider,
                               run_id=str(run_id or ""),
                               node=str(meta.get("langgraph_node") or ""),
                               checkpoint_ns=str(meta.get("langgraph_checkpoint_ns") or ""),
                               thread_id=str(meta.get("thread_id") or ""))
        (snap.tool_count, snap.tools_chars, snap.tools_top,
         snap.tools_source) = _measure_tools(invocation_params)
        budget = _MAX_SNAPSHOT_CHARS
        for msg in (messages or []):
            raw = _text_of(getattr(msg, "content", msg))
            n = len(raw)
            snap.total_chars += n
            if budget <= 0:
                snap.dropped_messages += 1
                continue
            body, truncated = raw, False
            if n > _MAX_MESSAGE_CHARS:
                body, truncated = raw[:_MAX_MESSAGE_CHARS], True
            if len(body) > budget:
                body, truncated = body[:budget], True
            budget -= len(body)
            # Tool calls live outside .content on an AIMessage; show them or the
            # snapshot would claim an empty assistant turn where a tool was called.
            #
            # The ARGUMENTS are the point. Until 2026-08-03 this recorded only the
            # names — and then a tool call arrived with p_gain=3 where the model's
            # own reasoning said 3e-12, and the one artefact that would have shown
            # what it actually emitted held the string "[tool_calls] SetZCtrlGain".
            # The values had to be reconstructed backwards from the tool RESULT.
            # A capture that omits the arguments is silent in precisely the case
            # it exists for.
            calls = getattr(msg, "tool_calls", None)
            if calls:
                body = (body + "\n" + _render_tool_calls(calls)).strip()
            blocks, src = _blocks_of(msg, raw)
            if blocks is not None and snap.blocks_source == "none":
                snap.blocks_source = src
            snap.messages.append(
                CapturedMessage(role=_role_of(msg), content=body,
                                chars=n, truncated=truncated, blocks=blocks)
            )
        _RING.add(snap)
    except Exception:  # noqa: BLE001 — a capture must never break a model call
        logger.debug("prompt capture failed (swallowed)", exc_info=True)


def _extract_usage(response: Any) -> dict | None:
    """provider **上报的** token。取不到返回 None —— 绝不用字符数折算。

    主口径与计费侧 (:func:`mast.billing.capture._extract_usage`) 相同：先
    ``usage_metadata``，再退 ``llm_output["token_usage"|"usage"]``。这里多取一层
    ``input_token_details``（Anthropic 的缓存读/写明细），因为「system 到底有没有
    命中缓存」正是把易变块搬去 human 消息之后要验收的那个数。
    """
    try:
        in_tok = out_tok = None
        cache_read = cache_create = None
        for batch in (getattr(response, "generations", None) or []):
            for gen in (batch or []):
                msg = getattr(gen, "message", None)
                if msg is None:
                    continue
                um = getattr(msg, "usage_metadata", None)
                if isinstance(um, dict):
                    if um.get("input_tokens") is not None:
                        in_tok = int(um["input_tokens"])
                    if um.get("output_tokens") is not None:
                        out_tok = int(um["output_tokens"])
                    det = um.get("input_token_details")
                    if isinstance(det, dict):
                        if det.get("cache_read") is not None:
                            cache_read = int(det["cache_read"])
                        if det.get("cache_creation") is not None:
                            cache_create = int(det["cache_creation"])
        if in_tok is None and out_tok is None:
            lo = getattr(response, "llm_output", None) or {}
            tu = lo.get("token_usage") or lo.get("usage") or {}
            if isinstance(tu, dict) and tu:
                pv = tu.get("prompt_tokens", tu.get("input_tokens"))
                cv = tu.get("completion_tokens", tu.get("output_tokens"))
                in_tok = int(pv) if pv is not None else None
                out_tok = int(cv) if cv is not None else None
        if in_tok is None and out_tok is None:
            return None
        return {"input_tokens": in_tok, "output_tokens": out_tok,
                "cache_read": cache_read, "cache_creation": cache_create,
                "source": "provider"}
    except Exception:  # noqa: BLE001
        return None


def make_callback(*, source: str = "", model_id: str = "", provider: str = "") -> Any:
    """A LangChain callback handler that records every call it is attached to.

    Built lazily so importing this module never requires langchain_core.
    """
    from langchain_core.callbacks import BaseCallbackHandler

    class PromptCaptureCallback(BaseCallbackHandler):
        """Records the exact message list handed to the provider."""

        def __init__(self) -> None:
            super().__init__()
            self._source = source
            self._model_id = model_id
            self._provider = provider

        def on_chat_model_start(self, serialized: Any, messages: Any,
                                **kwargs: Any) -> None:
            try:
                # messages is list[list[BaseMessage]] — one inner list per generation.
                batch = messages[0] if messages else []
                record(batch, source=self._source, model_id=self._model_id,
                       provider=self._provider,
                       run_id=str(kwargs.get("run_id") or ""),
                       metadata=kwargs.get("metadata"),
                       invocation_params=kwargs.get("invocation_params"))
            except Exception:  # noqa: BLE001
                logger.debug("on_chat_model_start capture failed", exc_info=True)

        def on_llm_end(self, response: Any, **kwargs: Any) -> None:
            """Record what the model REPLIED, tool-call arguments included."""
            try:
                if not _RING.enabled:
                    return
                usage = _extract_usage(response)
                run_id = str(kwargs.get("run_id") or "")
                gens = getattr(response, "generations", None) or []
                msg = getattr(gens[0][0], "message", None) if gens and gens[0] else None
                if msg is None:
                    if usage:
                        _RING.attach_response(self._source or "unknown",
                                              CapturedMessage(role="ai:response",
                                                              content="", chars=0),
                                              run_id=run_id, usage=usage)
                    return
                raw = _text_of(getattr(msg, "content", ""))
                calls = getattr(msg, "tool_calls", None)
                if calls:
                    raw = (raw + "\n" + _render_tool_calls(calls)).strip()
                if not raw:
                    if usage:
                        _RING.attach_response(self._source or "unknown",
                                              CapturedMessage(role="ai:response",
                                                              content="", chars=0),
                                              run_id=run_id, usage=usage)
                    return
                n = len(raw)
                body, truncated = raw, False
                if n > _MAX_MESSAGE_CHARS:
                    body, truncated = raw[:_MAX_MESSAGE_CHARS], True
                _RING.attach_response(
                    self._source or "unknown",
                    CapturedMessage(role="ai:response", content=body,
                                    chars=n, truncated=truncated),
                    run_id=run_id, usage=usage,
                )
            except Exception:  # noqa: BLE001 — capture must never break a call
                logger.debug("on_llm_end capture failed", exc_info=True)

    return PromptCaptureCallback()


__all__ = [
    "MAX_SNAPSHOTS", "CapturedMessage", "CapturedRequest",
    "get_ring", "make_callback", "record",
]
