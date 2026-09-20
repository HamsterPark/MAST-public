"""agent 手里的技能市场工具 —— 搜得到全集，但改不了自己的工具面。

## 为什么给「搜索」

订阅列表把 agent 的装载面从几百个技能收窄到用户真正在用的那些（见
``mast.skills.subscription``）。这是好事 —— 工具表是模型每一回合都要读一遍的菜单。
但它带来一个新问题：**agent 从此看不见自己没有的能力**，于是遇到做不了的事只会说
「我没有这个工具」，而不是「本机有一个 CharacteriseCurrentNoise，要不要装上」。

``search_skill_market`` 就是补这个洞的：它查的是**市场全集**，不受订阅影响。

## 为什么**不给**「订阅」

``recommend_skill_subscription`` 只把想法写进 pending 队列，改订阅是人面上的动作。
理由不是「不信任模型」，是**自我扩权通道**这件事本身：一个能改自己工具面的 agent，
在任何自主度档位下都等于没有工具面约束。

这与 2026-08-20 conduct 那条「工具面全开、把关在服务端」**不矛盾**：conduct 工具的
每一次写都落进服务端裁决的意图队列（自主度 × validator 包络 × op 状态机），而订阅
没有对应的服务端裁决层 —— 给了就是直写。所以这里的做法是**工具层根本不给写工具**：
没有那个工具，就没有「换条路做」可换。

真要给的话，先回答：谁在夜里三点裁决「agent 想给自己加一个 DANGEROUS 技能」？
在那个裁决层存在之前，答案只能是人。

## 这一族**不**进工作流菜单

两个工具都不在任何 ``WORKFLOW_TOOL_EXPORTS`` 导出表里，所以
``register_workflow_tool_skills`` 结构性地够不到它们（与 handoff / buffer 同一类）。
一个 composite 步骤去「搜市场」是没有意义的，而一个 composite 步骤去「推荐订阅」
更糟：它会把一次工作流执行变成一串待人裁决的提示。
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

__all__ = ["make_market_tools", "MARKET_TOOL_NAMES"]

#: 这一族的工具名。与 :func:`make_market_tools` 的返回一一对应（有测试钉着 ——
#: 清单说有、图上没有是本仓踩过的形状）。
MARKET_TOOL_NAMES: tuple[str, ...] = (
    "search_skill_market",
    "recommend_skill_subscription",
)

#: 一次搜索最多返回几条。菜单化的东西一律要有上限：把 459 条塞回上下文，
#: 等于用一次搜索抹掉这一轮对话的其余部分。
_MAX_HITS = 25


def _j(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)


def _origin_conversation() -> str:
    """这次工具调用所在的会话（背景运行时为空）。读法照 request_tools。"""
    try:
        from mast.core.turn_context import current_turn
        return str((current_turn() or {}).get("conversation_id") or "")
    except Exception:  # noqa: BLE001 — provenance is best-effort
        return ""


def make_market_tools(agent_name: str = "") -> list:
    """建这一族工具。

    *agent_name* 只用来在推荐记录上署名 —— 与 ``make_conduct_tools`` 同一条纪律：
    署名答的是「谁提的」，**不用来决定给不给**。
    """
    who = str(agent_name or "unknown")

    @tool("search_skill_market")
    def search_skill_market(query: str = "", source: str = "", domain: str = "") -> str:
        """在**全部**技能里搜索（包括你现在手上没有的那些）。

        你的工具表只包含用户订阅了的技能。这个工具查的是市场全集 —— 当你遇到
        「这件事我没有工具做」时，先用它看看本机到底有没有这个能力。

        Args:
            query: 关键词（技能名/中文名/领域/标签，模糊匹配）。
            source: 可选来源过滤（builtin / composite / paper / user_composite / …）。
            domain: 可选领域过滤。

        返回每条技能的名字、中文名、领域、安全级，以及 ``subscribed``（你现在
        手上有没有它）。手上没有而又确实需要，用 recommend_skill_subscription。
        """
        try:
            from mast.skills import subscription as sub
            from mast.webui.builder_api import filter_index, get_catalog
        except Exception as exc:  # noqa: BLE001
            return f"技能市场暂时读不到（{exc}）"
        try:
            idx = filter_index(list(get_catalog().get("index") or []),
                               q=query or "", source=source or "", domain=domain or "")
        except Exception as exc:  # noqa: BLE001
            return f"技能市场暂时读不到（{exc}）"
        if not idx:
            return f"市场里没有匹配「{query}」的技能。"
        subs = sub.subscribed_names()
        hits = []
        for e in idx[:_MAX_HITS]:
            name = str(e.get("name") or "")
            hits.append({
                "name": name,
                "zh": e.get("zh") or "",
                "domain": e.get("domain") or "",
                "safety": e.get("safety") or "",
                "subscribed": (True if subs is None
                               else name in subs or name in sub.MANDATORY_SKILLS),
            })
        more = len(idx) - len(hits)
        out = _j(hits)
        if more > 0:
            # 截断了就说 —— 静默截断会被读成「市场里只有这些」。
            out += f"\n（还有 {more} 条未列出，把 query 写具体一些）"
        return out

    @tool("recommend_skill_subscription")
    def recommend_skill_subscription(skill_name: str, reason: str) -> str:
        """向用户**推荐**订阅一个技能（不会立刻生效）。

        用在你确认本机有这个能力、而它不在你手上的时候。这条推荐会出现在
        「技能 → 市场」页等用户确认；**只有他点了接受，它才会进你的工具表**。

        不要重复推荐同一个技能（重复提交会被合并成同一条），也不要等它 —— 提交
        之后继续做你手头能做的事，或者告诉用户你在等什么。

        Args:
            skill_name: 技能名（先用 search_skill_market 查准确的名字）。
            reason: 为什么需要它 —— 用户就看这一句来决定接不接受。
        """
        name = str(skill_name or "").strip()
        if not name:
            return "要推荐哪个技能？给一个技能名。"
        try:
            from mast.skills import subscription as sub
        except Exception as exc:  # noqa: BLE001
            return f"订阅列表暂时不可用（{exc}）"

        try:
            from mast.webui.builder_api import get_catalog
            known = {str(e.get("name") or "")
                     for e in (get_catalog().get("index") or [])}
        except Exception:  # noqa: BLE001
            known = set()
        if known and name not in known:
            return (f"市场里没有叫「{name}」的技能。先用 search_skill_market "
                    "查一下准确的名字。")
        if sub.is_subscribed(name):
            return f"「{name}」已经在你的工具表里了，不需要推荐 —— 直接用它。"

        try:
            rec = sub.add_recommendation(name, by_agent=who, reason=str(reason or ""),
                                         conversation_id=_origin_conversation())
        except Exception as exc:  # noqa: BLE001 — 工具环不能因此崩
            return f"提交推荐失败：{exc}"
        if not rec.get("ok"):
            return f"提交推荐失败：{rec.get('reason') or '未知原因'}"

        info = rec.get("recommendation") or {}
        rec_id = str(info.get("id") or "")
        _publish(rec_id, name, who)
        if rec.get("duplicate"):
            return (f"「{name}」已经有一条待确认的推荐（{rec_id}）—— 不用重复提交，"
                    "继续做你手头能做的事。")
        return (f"已提交推荐（{rec_id}）：等用户在「技能 → 市场」确认后，"
                f"「{name}」才会进你的工具表。别等它。")

    return [search_skill_market, recommend_skill_subscription]


def _publish(rec_id: str, skill: str, by_agent: str) -> None:
    """发一帧，让界面知道该去 refetch。

    **只做触发，不做状态源**：payload 里只有指针，界面收到后重新拉
    ``GET /api/skill-market/recommendations``。这条总线只重放最后 100 条，
    按帧累积列表的客户端迟早显示一份错的（同 CONDUCT_* 三帧那段注释）。
    """
    try:
        from mast.core.events import Event, EventBus, EventType

        EventBus.get().publish(Event(
            type=EventType.SKILL_RECOMMENDATION,
            data={"rec_id": rec_id, "skill": skill, "by_agent": by_agent}))
    except Exception as exc:  # noqa: BLE001 — 推送失败不该让推荐失败
        logger.debug("技能推荐事件发不出去：%s", exc)
