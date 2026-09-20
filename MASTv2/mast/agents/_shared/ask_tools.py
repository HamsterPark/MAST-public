"""Agent→operator STRUCTURED QUESTION — the blocking counterpart of request_tools.

Why this module exists (2026-08-01)
-----------------------------------
Until now an agent that hit a decision belonging to the *operator* had nothing to
call. ``experiment_design``'s prompt has been saying, in writing:

    If the research question is ambiguous, ask the operator ONE clarifying
    question then wait for a reply — do not guess.

**Nothing implemented "then wait for a reply."** The agent said its sentence,
had no tool to call, and the turn ended — usually tripping the fail-silent
termination guard. The one thing it could reach, ``request_user_action``, is
explicitly *not* a blocking gate: it posts to the 心愿单 board and the answer
arrives on some later turn.

So the system could pause for a DANGEROUS skill (the operator says yes/no to
something the agent already decided), but it could not pause to let the operator
*make* a decision. This module closes that gap using the same interrupt/resume
pipeline the approval gate already runs on — one new interrupt ``kind``, no new
transport.

Division of labour (say this in the tool docs, agents get it wrong otherwise)
----------------------------------------------------------------------------
* :func:`ask_user` — the answer CHANGES WHAT HAPPENS NEXT and the work cannot
  sensibly continue without it. Blocks the graph.
* ``request_user_action`` — the operator has to *do* something physical (upload a
  file, change a sample), or the ask does not block this step. Async board.

Replay idempotency
------------------
``interrupt()`` aborts the whole tool call and LangGraph REPLAYS the tool from
the top on resume (this is why ``skills/composite/interpreter.py`` caches its
human-node decisions before interrupting). This tool is safe under that rule by
construction: everything before ``interrupt()`` is pure payload validation with
no side effects, so a replay simply rebuilds the same payload and receives the
operator's answer.
"""
from __future__ import annotations

import logging

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

#: Options beyond this are dropped (and the drop is stated in the question, never
#: silently). A choice card the operator has to scroll is a choice they answer
#: badly; a model that wants 20 options is really asking an open question.
_MAX_OPTIONS = 8

#: What happens when nobody answers within the approval window (900 s).
#: ``continue`` = the tool returns "unanswered, use your stated fallback" and the
#: run carries on; ``halt`` = the run stops with the checkpoint intact so the
#: operator can resume it later. Anything else normalises to ``continue``.
_TIMEOUT_ACTIONS = ("continue", "halt")


def _normalize_options(options) -> list[dict]:
    """Coerce whatever the model sent into ``[{"label", "description"}]``.

    Deliberately permissive: weak models send ``["A", "B"]`` as often as the
    documented shape, and a schema-level rejection would fail the call *before*
    the tool body could return a correcting message. Anything unusable is
    dropped rather than raising.
    """
    out: list[dict] = []
    seen: set[str] = set()
    for raw in (options or []):
        if isinstance(raw, str):
            label, desc = raw.strip(), ""
        elif isinstance(raw, dict):
            label = str(raw.get("label") or raw.get("value") or "").strip()
            desc = str(raw.get("description") or raw.get("detail") or "").strip()
        else:
            continue
        if not label or label in seen:
            continue
        seen.add(label)
        out.append({"label": label, "description": desc})
    return out


def _build_payload(question: str, options, multi_select: bool,
                   allow_custom: bool, header: str,
                   timeout_action: str) -> dict | str:
    """Validate + normalise into the interrupt payload, or return an error string.

    A string return is handed straight back to the model as the tool result: the
    call is rejected *before* any interrupt happens, so the model can fix its
    arguments and retry without the operator ever seeing a malformed card.
    """
    q = str(question or "").strip()
    if not q:
        return ("ask_user: 需要一个具体问题（question 不能为空）。"
                "写清楚你要用户决定什么，并给出 2-4 个选项。")
    opts = _normalize_options(options)
    dropped = 0
    if len(opts) > _MAX_OPTIONS:
        dropped = len(opts) - _MAX_OPTIONS
        opts = opts[:_MAX_OPTIONS]
    if dropped:
        # SAY SO. A silently truncated list reads to the operator as "these are
        # all the choices", which is exactly the kind of quiet lie this repo
        # keeps paying for elsewhere.
        q = f"{q}\n（备选项过多，只显示前 {_MAX_OPTIONS} 个；另有 {dropped} 个未列出）"
    ta = str(timeout_action or "").strip().lower()
    if ta not in _TIMEOUT_ACTIONS:
        ta = "continue"
    return {
        "kind": "ask_user",
        "question": q,
        "header": str(header or "").strip()[:24],
        "options": opts,
        "multi_select": bool(multi_select) and bool(opts),
        # No options = an open question; a text box is then the ONLY way to
        # answer it, so the flag cannot be off whatever the caller passed.
        "allow_custom": bool(allow_custom) or not opts,
        "timeout_action": ta,
        "agent_id": "",
    }


def _format_answer(payload: dict, answer) -> str:
    """Render the operator's resume value as the tool's result text."""
    question = payload.get("question", "")
    if not isinstance(answer, dict):
        # Shouldn't happen (the API layer builds this value), but a resume that
        # arrives malformed must not silently look like a real answer.
        return (f"[用户回答] 问题：{question}\n"
                f"收到无法解析的回答（{answer!r}）。请把它当作『未获答复』处理："
                "按你提问时说明的保守默认继续，或用 request_user_action 留言。")
    if answer.get("timeout"):
        note = str(answer.get("note") or "").strip()
        return (f"[用户未应答] 问题：{question}\n"
                f"{note or '等待超时，没有收到回答。'}\n"
                "请按你提问时说明的保守默认继续，并在给用户的汇报里写明这一步是你"
                "自行决定的；若没有安全的默认做法，就用 request_user_action 留言后"
                "结束本回合，不要猜。")
    selected = [str(s) for s in (answer.get("selected") or []) if str(s).strip()]
    custom = str(answer.get("custom_text") or "").strip()
    note = str(answer.get("note") or "").strip()
    lines = [f"[用户回答] 问题：{question}"]
    if selected:
        lines.append("选择：" + "、".join(selected))
        lines.append(f"补充：{custom or '（无）'}")
    else:
        lines.append(f"回答：{custom or '（用户未给出内容）'}")
    if note:
        lines.append(f"用户备注：{note}")
    return "\n".join(lines)


@tool("ask_user")
def ask_user(question: str, options: list | None = None,
             multi_select: bool = False, allow_custom: bool = True,
             header: str = "", timeout_action: str = "continue") -> str:
    """向用户提一个**结构化问题并阻塞等待回答**（对方在界面上点选/填写后你才继续）。

    **什么时候用**：这个决定本来就属于人，不属于你——目标取舍（先追产率还是先追
    分辨率）、风险偏好（是否愿意冒针尖损伤换一张图）、资源牺牲（要不要放弃这块
    样品区域）、几个都成立的方案之间二选一。判据是：**换一个回答，你接下来做的事
    就不一样**。

    **什么时候不要用**：你自己查得到的（先调 list_documents / 环境工具 / 知识库）、
    属于专业判断而非偏好的、或者答案不影响你当前这一步的。需要用户**动手**做事
    （上传全文、换样品、开某个仪器），用 `request_user_action` 异步留言，不要用这个
    工具把整条流程堵在那里等人。

    **提问的规矩**：
      * 一次只问一个问题。要问两件事就分两次调用（先问最关键的那件）。
      * 给 2-4 个**具体**选项，每个写清楚代价和后果，别写"是/否"这种要对方猜含义的。
      * **必须在 question 里写明"如果没人回答，我会默认怎么做"** —— 超时会按你说的
        那个默认继续，写不出安全默认就说明这问题该用 timeout_action="halt"。

    Args:
        question: 问题正文。写清楚背景、你卡在哪、以及无人应答时你的保守默认。
        options:  选项列表，推荐 `[{"label": "选项名", "description": "代价/后果"}]`；
                  直接给字符串列表也接受。**留空 = 开放式提问**（只给文本框）。
                  最多 8 个，多了只显示前 8 个并在问题里说明。
        multi_select: True = 允许多选（比如"这几项里哪些要做"）。默认单选。
        allow_custom: 是否允许用户不选给出的选项、自己写一个答案。默认允许——
                  你列的选项很可能没覆盖对方真正想要的。
        header:   卡片上的短标题（≤12 字，比如"区域选择"）。可留空。
        timeout_action: 900 秒无人应答时怎么办。`"continue"`（默认）= 你会收到
                  "未应答"并**按你说明的保守默认自行继续**；`"halt"` = 本次运行就此
                  停下等人（进度已存档，用户回来可以继续）。**只有在没有任何安全
                  默认做法时才用 halt** —— 它会让整条流程停摆。

    Returns:
        用户的回答（选了什么 + 自定义补充 + 备注），或超时说明。
    """
    payload = _build_payload(question, options, multi_select, allow_custom,
                             header, timeout_action)
    if isinstance(payload, str):
        return payload                      # rejected before any interrupt

    # ★ v2 路径优先（2026-08-27）。
    #
    # 新循环在**自己的线程上**顺序执行工具，所以它把当前 ``RunContext`` 挂在一个
    # ContextVar 上，桥接进来的工具（这个就是）据此拿到提问通道。拿得到就**原地
    # 阻塞等答案**，没有重放。
    #
    # 不这么做的后果是实测出来的，而且安静得可怕：``interrupt()`` 在图运行时之外
    # 抛 ``KeyError`` → 下面那个 except 把它转成一句「这个入口没有接提问通道」→
    # 这一轮以 ``outcome="final"`` **正常收场**。agent 问了、用户永远看不到、
    # 现场看起来一切正常。
    #
    # 通道不在就**照旧往下走**（旧引擎、裸测试、GUI 手动路径都走那条），所以这段
    # 对旧路径是纯新增。
    try:
        from mast.agentruntime.context import current_context

        _ctx = current_context()
    except Exception:  # noqa: BLE001 — agentruntime 缺席时照旧走 interrupt
        _ctx = None
    if _ctx is not None and getattr(_ctx, "ask_human", None) is not None:
        answer = _ctx.ask({**payload, "_abort": getattr(_ctx, "abort", None)})
        if answer is None:
            return ("用户未在时限内回答，且本次提问声明了 halt。"
                    "请停下并说明你在等什么。")
        return _format_answer(payload, answer)

    try:
        from langgraph.errors import GraphInterrupt
        from langgraph.types import interrupt
    except ImportError:  # pragma: no cover — langgraph is pinned in v2
        return ("ask_user 当前不可用（langgraph 缺失）。"
                "请改用 request_user_action 留言，或按你的保守默认继续并说明。")
    try:
        answer = interrupt(payload)
    except GraphInterrupt:
        raise                               # control flow — NEVER swallow
    except Exception as exc:  # noqa: BLE001 — outside a graph runtime
        logger.debug("ask_user outside graph runtime: %s", exc)
        return (f"ask_user 当前不可用（{type(exc).__name__}）—— 这个入口没有接提问通道。"
                "请改用 request_user_action 留言，或按你的保守默认继续并说明理由。")
    return _format_answer(payload, answer)


#: Handed to EVERY agent (group chat + private chat). Asking the operator a
#: question is not a per-agent privilege — every one of them can hit a decision
#: that is not theirs to make.
ASK_USER_TOOLS: list = [ask_user]

__all__ = ["ask_user", "ASK_USER_TOOLS", "_build_payload", "_format_answer",
           "_normalize_options", "_MAX_OPTIONS", "_TIMEOUT_ACTIONS"]
