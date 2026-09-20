"""Agent→user request tool — an agent asks the operator to do something.

Posts a request to the shared :mod:`mast.wishlist` board (the 心愿单 tab), e.g.
the literature agent asking the user to upload a full-text PDF, or the instrument
agent asking the user to perform a hardware action. ASYNCHRONOUS + non-blocking:
the agent posts and continues / hands back; the user fulfils it in the 心愿单 tab
and the agent learns the outcome on a later turn (NOT a blocking HITL gate).

Shared module (``agents/_shared``) so it attaches to several agents without
crossing the agent boundary; mirrors ``memory_tools`` / ``cognition_tools``.
"""

from __future__ import annotations

import logging
import re
from typing import Callable

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# provider() -> {"agent_id": str, "experiment_id": str|None}
Provider = Callable[[], dict]

#: A DOI or an OpenAlex work_id in the message body. Narrow on purpose — see
#: :func:`_fulltext_redirect`.
_PAPER_ID_RE = re.compile(r"\b(10\.\d{4,9}/\S+|W\d{6,})", re.IGNORECASE)
#: Words that make an upload request specifically about a paper's full text.
_FULLTEXT_WORDS = ("全文", "原文", "pdf", "full text", "fulltext", "论文")


def _origin_conversation() -> str:
    """The conversation this tool call is running inside, or ``""``.

    Stored on the request so answering it can resume that conversation. Empty in
    a background run, which degrades to the previous behaviour (the answer is
    read back on whatever turn happens next).
    """
    try:
        from mast.core.turn_context import current_turn
        return str((current_turn() or {}).get("conversation_id") or "")
    except Exception:  # noqa: BLE001 — provenance is best-effort
        return ""


def _fulltext_redirect(message: str, kind: str) -> str:
    """A redirect string when this ask belongs on the fetch board, else ``""``.

    Both boards can hold "please give me this paper", but only one of them then
    files the paper against the experiment that asked, promotes it into the
    library, and resumes the conversation that was blocked on it. An agent that
    follows the generic route gets a worse outcome for the same request and no
    indication that it did — the failure is invisible precisely because the
    request itself succeeds.

    Kept deliberately narrow: it fires only when the ask is BOTH an upload AND
    carries a DOI / work_id AND mentions the full text. "请把上次那张图的 PDF 放到
    机器上" has no paper id and posts normally. A false positive costs one turn —
    the agent is told exactly which tool to call instead.
    """
    if (kind or "").strip().lower() != "upload":
        return ""
    text = message or ""
    m = _PAPER_ID_RE.search(text)
    if not m:
        return ""
    low = text.lower()
    if not any(w in low for w in _FULLTEXT_WORDS):
        return ""
    return (
        f"没有提交到心愿单 —— 这条应该走**取文请求板**：请改调 "
        f"request_fulltext(work_id=\"{m.group(1)}\", reason=\"为什么需要这篇的全文\")。"
        f"取文板会按 work_id 记账、冻结到当时的实验、上传后自动入库，并在满足时"
        f"**自动继续你的工作**；心愿单这条路以上全都没有。"
        f"（如果你要的其实不是论文全文，把 kind 改成 action 再发一次。）"
    )


def make_request_tools(provider: Provider) -> list:
    """Build [request_user_action] bound to *provider*."""

    def _ctx() -> dict:
        try:
            return provider() or {}
        except Exception as exc:  # pragma: no cover
            logger.debug("request provider failed: %s", exc)
            return {}

    @tool("request_user_action")
    def request_user_action(message: str, kind: str = "action") -> str:
        """向用户发起一条请求,记录在「心愿单」页等待用户处理(异步,不阻塞)。

        用于需要用户线下/线上配合的情形,例如:
          - 仪器:"请到机台手动 <操作>(更换样品 / 进针 / 对准等)"
          - 文件:"请把 <某文件> 放到机器上,并把路径填在心愿单的「路径」栏"

        **要论文全文请用 request_fulltext,不要用这个工具。** 取文请求板是专门为它做的:
        按 work_id 记账、冻结到当时的实验、上传后自动入库并**自动继续你的工作**;
        走心愿单则以上全都没有,你只会拿到一段自由文本。

        message: 给用户的清晰请求(建议以角色开头,如 "[仪器] …")。
        kind: action(需操作) | upload(需上传) | info(仅告知)。
        返回请求 id。用户在心愿单页处理后:①下一轮运行会把答复【自动注入】你的上下文;
        ②你也可用 check_request_reply(该 id) 主动轮询这一条是否已答复。不要阻塞等待。"""
        ctx = _ctx()
        redirect = _fulltext_redirect(message, kind)
        if redirect:
            return redirect
        try:
            from mast.wishlist import post_agent_request
            rec = post_agent_request(
                ctx.get("agent_id", "agent"), message, kind=kind,
                experiment_id=ctx.get("experiment_id"),
                origin_conversation_id=_origin_conversation(),
            )
            if rec.get("error"):
                return f"提交请求失败: {rec['error']}"
            return (f"已向用户发起请求({rec['id']}),在「心愿单」页等待处理。"
                    f"可用 check_request_reply(\"{rec['id']}\") 查询该请求是否已答复。")
        except Exception as exc:  # noqa: BLE001 — tool loop must not crash
            return f"提交请求失败: {exc}"

    @tool("report_upgrade_idea")
    def report_upgrade_idea(message: str) -> str:
        """把一个【能力缺口 / 升级建议】记到「心愿单」,供下次升级参考(异步,不阻塞;
        不是要用户立刻操作)。

        什么时候用:
          - 你需要一个【当前不存在的 skill / 命令】才能更好地完成任务;
          - 某条指令含义【模糊】,即使你已向用户澄清,这个歧义/需求仍值得固化为新能力;
          - 你发现某个 skill 不可靠、或某流程可改进。
        先用现有工具尽力完成当前任务、必要时先向用户澄清;这条只【记录改进想法】,不替代当前行动。
        message: 清晰描述缺口或建议(建议以角色开头,如 "[仪器] 需要一个 … 的 skill")。
        返回记录 id。"""
        ctx = _ctx()
        try:
            from mast.wishlist import post_agent_request
            rec = post_agent_request(
                ctx.get("agent_id", "agent"), message, kind="upgrade",
                experiment_id=ctx.get("experiment_id"),
            )
            if rec.get("error"):
                return f"记录升级建议失败: {rec['error']}"
            return f"已记录升级建议({rec['id']})到「心愿单」,供下次升级参考。"
        except Exception as exc:  # noqa: BLE001 — tool loop must not crash
            return f"记录升级建议失败: {exc}"

    @tool("check_my_requests")
    def check_my_requests() -> str:
        """查看用户是否已经答复了你之前用 request_user_action 发起的请求。

        什么时候用:
          - 你之前请用户做一件事(换样品 / 进针 / 提供文件路径 / 上传 PDF),
            现在想知道他做完没有、以及他给了什么答复;
          - 你需要一个【文件路径】而它不在你手上——用户会把它填在心愿单的
            「路径」栏里,这个工具会把它原样交给你。

        返回已答复但你尚未读过的请求(读过一次就不再重复返回)。没有新答复时明确告知。
        不要阻塞轮询——发起请求后先做别的,过一会儿再查。"""
        ctx = _ctx()
        try:
            from mast.wishlist import get_board

            board = get_board()
            agent_id = str(ctx.get("agent_id") or "")
            rows = board.resolved_requests_for(agent_id, undelivered_only=True)
            if not rows:
                return ("暂无新的用户答复。（发起过的请求若仍在「待处理」，"
                        "说明用户还没处理；不要重复发起同一请求。）")
            board.mark_delivered([r["id"] for r in rows])
            lines = []
            for r in rows:
                status = "已完成" if r.get("status") == "done" else "已忽略"
                line = f"• [{r['id']}] {status}：{r.get('message', '')}"
                if r.get("path"):
                    # THE point of #97: the path is a field, so it comes back
                    # verbatim — the agent does not have to parse it out of prose.
                    line += f"\n    → 用户提供的路径：{r['path']}"
                if r.get("note"):
                    line += f"\n    → 备注：{r['note']}"
                lines.append(line)
            return "用户已答复以下请求：\n" + "\n".join(lines)
        except Exception as exc:  # noqa: BLE001 — tool loop must not crash
            return f"查询请求答复失败: {exc}"

    @tool("check_request_reply")
    def check_request_reply(request_id: str) -> str:
        """轮询你之前用 request_user_action 发起的【某一条具体请求】是否已被用户答复。

        与 check_my_requests 的区别:
          - check_my_requests 一次性取回【所有】未读答复,读过即不再返回;
          - check_request_reply 针对【单条】请求,是【非破坏性】轮询——可反复查询
            同一条,不会消费答复(下一轮的自动注入仍会带上它)。

        request_id: request_user_action 返回的请求 id(如 "r-4")。
        返回该请求的状态(待处理/已完成/已忽略),以及用户给的【路径】/备注(若有)。
        用于:你请用户提供一个文件路径后,想确认"这条到底答复了没、路径是什么"。"""
        try:
            from mast.wishlist import get_board

            board = get_board()
            rec = board.get_request((request_id or "").strip())
            if not rec:
                return (f"未找到请求 {request_id}。请核对 id(如 r-4);"
                        "它可能从未创建或已被清理。")
            status = rec.get("status")
            if status == "pending":
                return (f"[{request_id}] 仍在【待处理】——用户尚未答复。"
                        "请勿重复发起同一请求,先做别的,过一会儿再查。")
            label = "已完成" if status == "done" else "已忽略"
            parts = [f"[{request_id}] 用户已{label}:{rec.get('message', '')}"]
            if rec.get("path"):
                parts.append(f"    → 用户提供的路径:{rec['path']}")
            if rec.get("note"):
                parts.append(f"    → 备注:{rec['note']}")
            return "\n".join(parts)
        except Exception as exc:  # noqa: BLE001 — tool loop must not crash
            return f"查询请求失败: {exc}"

    return [request_user_action, report_upgrade_idea,
            check_my_requests, check_request_reply]


__all__ = ["make_request_tools"]
