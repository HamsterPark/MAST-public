"""The inventory: every text MAST injects into an agent's context.

Two kinds of thing live in here, and the difference is the whole point:

``static``
    The injected text IS a module constant. What you read here is byte-for-byte
    what the agent gets, and an override replaces it outright.

``computed``
    The text is generated per call from state (hardware readings, the persisted
    rig profile, the conversation, the tool that just failed). There is no
    "the text" to edit — only a generator. Those entries carry
    ``availability`` telling you whether the current value can be shown:

    * ``live``          — rendered here from the CURRENT persisted state. Real,
                          just possibly older than the next request's value.
    * ``needs_hardware``— needs a Nanonis read. Not rendered.
    * ``needs_request`` — only exists inside a request (conversation, failing
                          tool, unfinished experiment). Not rendered.

``needs_hardware`` / ``needs_request`` entries return an EMPTY body plus a
reason. They are never filled with an illustrative sample. Someone debugging a
live incident against invented text is strictly worse off than someone who was
told the text is unavailable — that is the lesson the 2026-07-27 coordinate
incident already charged us for once. To see those blocks for real, read the
capture ring (:mod:`mast.prompts.capture`), which holds actual requests.

Adding an entry: append to ``_SPEC`` here, and — if it is overridable — make the
consuming site call :func:`resolve` instead of the bare constant. An entry that
is listed but not wired renders an override that silently does nothing, which is
the failure mode this whole feature exists to prevent.
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass, field
from typing import Callable

from mast.prompts import overrides as _ovr

logger = logging.getLogger(__name__)

# ── categories (UI grouping) ─────────────────────────────────────────────────
CAT_AGENT = "agent_system"      # a whole agent's static system prompt
CAT_ROUTING = "routing"         # supervisor / router prompts
CAT_MIDDLEWARE = "middleware"   # blocks middlewares append per model call
CAT_SUB_LLM = "sub_llm"         # prompts for helper LLM calls (refine, summarize)

# ── availability of the CURRENT text ─────────────────────────────────────────
AVAIL_STATIC = "static"                 # a constant; shown verbatim
AVAIL_LIVE = "live"                     # rendered now from persisted state
AVAIL_NEEDS_HARDWARE = "needs_hardware"  # needs a Nanonis read; not rendered
AVAIL_NEEDS_REQUEST = "needs_request"    # per-request only; not rendered

# ── 什么时候出现 ─────────────────────────────────────────────────────────
WHEN_ALWAYS = "always"          # 每一次模型调用
WHEN_WHEN_SET = "when_set"      # 只有配置/状态非空时（空则整块不注入）
WHEN_ON_EVENT = "on_event"      # 出了某件事才有（告警、空转、图像、召回命中）
WHEN_ON_MODE = "on_mode"        # 只在某个操作模式下
WHEN_BUILD_TIME = "build_time"  # 建图时拼进系统提示，之后逐轮不变

# ── 落在哪 ───────────────────────────────────────────────────────────────
POS_SYSTEM = "system"                  # system 消息末尾
POS_LAST_HUMAN = "last_human"          # 最后一条 human 消息末尾（逐轮易变的块）
POS_NEW_HUMAN = "new_human"            # 新插一条 human 消息
POS_STATE_MESSAGES = "state_messages"  # 改写消息列表本身（压缩）
POS_TOOL_RESULT = "tool_result"        # 改写 ToolMessage（精炼、安全门回执）
POS_TOOLS = "tools"                    # 工具面，不走消息

#: ``agents`` 里的这个值 = 全员。
ALL_AGENTS = "*"


@dataclass(frozen=True)
class PromptEntry:
    """One injectable text: where it comes from and whether it can be edited."""

    id: str
    label: str
    category: str
    availability: str
    source: str
    note: str
    overridable: bool = False
    agent: str = ""
    #: 哪些 agent 会收到它。``(ALL_AGENTS,)`` = 全员；空 = 看 ``agents_from``。
    #:
    #: **这里不是真源。** 定向注入的真源是中间件自己的模块级 ``AGENTS`` 常量
    #: （见 ``agents_from``）—— 让登记表决定挂不挂，等于把「清单」变成
    #: 「调度器」，而清单恰恰是 2026-08-24 发现归属标错了四条的地方。
    agents: tuple[str, ...] = ()
    #: ``"module:CONST"``，从中间件自己的常量派生。两边一个真源，不会漂。
    agents_from: str = ""
    when: str = WHEN_ALWAYS
    position: str = POS_SYSTEM
    #: 实现它的中间件类名（用于双向对账：栈里有的必须在册，在册的必须在栈里）。
    middleware: str = ""
    #: 哪几条建图路径上有它。工作流委托（``skills/composite/agent_node.py``）
    #: **不传 extra_middleware**，所以共享表那几条在那条路上确实不存在。
    paths: tuple[str, ...] = ("group", "standalone")
    #: 依赖哪个可选子系统才会挂上。空 = 无条件。
    #:
    #: 「全员」与「凡是有这个子系统的全员」是两件事。记忆召回登记成前者的时候，
    #: 矩阵会在一台没起 cognition 的机器上说谎 —— 而这正是矩阵最该说实话的场合。
    requires: str = ""
    #: Returns the code-default text. None for entries that cannot be rendered.
    loader: Callable[[], str] | None = field(default=None, compare=False, repr=False)
    #: Human-readable reason shown when the text cannot be rendered.
    unavailable_reason: str = ""
    #: Shown when the render SUCCEEDS but yields nothing. "Nothing is injected"
    #: and "we could not read it" are different facts and must not look alike.
    empty_note: str = ""


def _const(module: str, attr: str) -> Callable[[], str]:
    """Loader for a module-level string constant."""

    def _load() -> str:
        mod = importlib.import_module(module)
        return str(getattr(mod, attr))

    return _load


def _profile_block() -> str:
    from mast.core.instrument_profile import format_profile_block, get_profile

    return format_profile_block(get_profile())


def _prefs_block() -> str:
    from mast.agents._shared.experiment_prefs import format_prefs_block, get_prefs

    return format_prefs_block(get_prefs())


def _tool_index_block() -> str:
    """当前 IC 目录块。建过图才有 —— 没建过就是空，而空与「关掉了」都要说清楚。"""
    from mast.agents._shared import tool_packs as _tp
    from mast.prompts import builds as _b

    rec = _b.last_build("instrument_control")
    catalog = getattr(rec, "tool_catalog", None) if rec else None
    if catalog is None:
        return ""
    return _tp.render_index(catalog)


def _manual_index() -> str:
    from mast.knowledge.nanonis_manual import modules_index

    idx = modules_index()
    return ("\n\n# Nanonis 模块速查\n\n" + idx) if idx else ""


def _lit_unavailable_note() -> str:
    from mast.agents.literature.tools import tool_availability, unavailable_tools_note

    return unavailable_tools_note(tool_availability())


def _tip_block() -> str:
    from mast.core.instrument_profile import get_profile
    from mast.core.tip_state import format_tip_block, get_current_tip

    return format_tip_block(get_current_tip(), get_profile())


# ── the inventory ────────────────────────────────────────────────────────────
# Order = reading order in the UI: what the agent is, then who routes it, then
# what gets appended per call, then the helper-LLM prompts.
_SPEC: tuple[PromptEntry, ...] = (
    # ── agent static system prompts ──────────────────────────────────────────
    PromptEntry(
        id="agent.research_director.system",
        agents=("research_director",), when=WHEN_BUILD_TIME, position=POS_SYSTEM,
        label="科研策划 RD · 系统提示",
        category=CAT_AGENT, agent="research_director", availability=AVAIL_STATIC,
        source="MASTv2/mast/agents/research_director/prompts.py:SYSTEM_PROMPT",
        note=("决策链上半环：科学目标自己生成、自己迭代（Campaign 层 = 为什么做）。"
              "它不设计具体步骤、不填任何仪器数值 —— 覆写时别把这两条改掉，"
              "否则它会开始发明没有依据的偏压和 setpoint。图构建时读取。"),
        overridable=True,
        loader=_const("mast.agents.research_director.prompts", "SYSTEM_PROMPT"),
    ),
    PromptEntry(
        id="agent.literature.system",
        agents=("literature",), when=WHEN_BUILD_TIME, position=POS_SYSTEM,
        label="文献 LIT · 系统提示",
        category=CAT_AGENT, agent="literature", availability=AVAIL_STATIC,
        source="MASTv2/mast/agents/literature/prompts.py:SYSTEM_PROMPT",
        note="文献 agent 的角色定义。图构建时读取；改动在下一次编排器重建后生效。",
        overridable=True,
        loader=_const("mast.agents.literature.prompts", "SYSTEM_PROMPT"),
    ),
    PromptEntry(
        id="agent.experiment_design.system",
        agents=("experiment_design",), when=WHEN_BUILD_TIME, position=POS_SYSTEM,
        label="实验设计 XD · 系统提示",
        category=CAT_AGENT, agent="experiment_design", availability=AVAIL_STATIC,
        source="MASTv2/mast/agents/experiment_design/prompts.py:SYSTEM_PROMPT",
        note="实验设计 agent 的角色定义。图构建时读取；改动在下一次编排器重建后生效。",
        overridable=True,
        loader=_const("mast.agents.experiment_design.prompts", "SYSTEM_PROMPT"),
    ),
    PromptEntry(
        id="agent.instrument_control.system",
        agents=("instrument_control",), when=WHEN_BUILD_TIME, position=POS_SYSTEM,
        label="仪器控制 IC · 系统提示",
        category=CAT_AGENT, agent="instrument_control", availability=AVAIL_STATIC,
        source="MASTv2/mast/agents/instrument_control/prompts.py:SYSTEM_PROMPT",
        note=("驱动真实硬件的 agent —— 全系统改动风险最高的一段文本。"
              "图构建时读取，其后还会追加 Nanonis 模块速查（若本机已提取手册）。"),
        overridable=True,
        loader=_const("mast.agents.instrument_control.prompts", "SYSTEM_PROMPT"),
    ),
    PromptEntry(
        id="agent.data_processing.system",
        agents=("data_processing",), when=WHEN_BUILD_TIME, position=POS_SYSTEM,
        label="数据处理 DP · 系统提示",
        category=CAT_AGENT, agent="data_processing", availability=AVAIL_STATIC,
        source="MASTv2/mast/agents/data_processing/prompts.py:SYSTEM_PROMPT",
        note="数据处理 agent 的角色定义。图构建时读取。",
        overridable=True,
        loader=_const("mast.agents.data_processing.prompts", "SYSTEM_PROMPT"),
    ),
    PromptEntry(
        id="agent.paper_writing.system",
        agents=("paper_writing",), when=WHEN_BUILD_TIME, position=POS_SYSTEM,
        label="论文写作 PW · 系统提示",
        category=CAT_AGENT, agent="paper_writing", availability=AVAIL_STATIC,
        source="MASTv2/mast/agents/paper_writing/prompts.py:SYSTEM_PROMPT",
        note="论文写作 agent 的角色定义。图构建时读取。",
        overridable=True,
        loader=_const("mast.agents.paper_writing.prompts", "SYSTEM_PROMPT"),
    ),
    PromptEntry(
        id="agent.paper_review.system",
        agents=("paper_review",), when=WHEN_BUILD_TIME, position=POS_SYSTEM,
        label="论文审稿 PR · 系统提示",
        category=CAT_AGENT, agent="paper_review", availability=AVAIL_STATIC,
        source="MASTv2/mast/agents/paper_review/prompts.py:SYSTEM_PROMPT",
        note="论文审稿 agent 的角色定义。图构建时读取。",
        overridable=True,
        loader=_const("mast.agents.paper_review.prompts", "SYSTEM_PROMPT"),
    ),
    PromptEntry(
        id="agent.buffer_summarizer.system",
        agents=("buffer_summarizer",), when=WHEN_ALWAYS, position=POS_SYSTEM,
        label="缓冲区摘要 · 系统提示",
        category=CAT_AGENT, agent="buffer_summarizer", availability=AVAIL_STATIC,
        source="MASTv2/mast/agents/buffer_summarizer/prompts.py:SYSTEM_PROMPT",
        note="把仪器缓冲区的原始读数压成一段摘要的小 agent。每次调用时读取，立即生效。",
        overridable=True,
        loader=_const("mast.agents.buffer_summarizer.prompts", "SYSTEM_PROMPT"),
    ),
    PromptEntry(
        id="agent.buffer_summarizer.user_template",
        agents=("buffer_summarizer",), when=WHEN_ALWAYS, position=POS_NEW_HUMAN,
        label="缓冲区摘要 · 用户模板",
        category=CAT_AGENT, agent="buffer_summarizer", availability=AVAIL_STATIC,
        source="MASTv2/mast/agents/buffer_summarizer/prompts.py:USER_TEMPLATE",
        note=("包住缓冲区数据的用户消息模板。⚠️ 其中的 {占位符} 由代码 .format() 填充，"
              "覆写时必须原样保留，否则该次摘要会直接失败。"),
        overridable=True,
        loader=_const("mast.agents.buffer_summarizer.prompts", "USER_TEMPLATE"),
    ),
    # ── routing ─────────────────────────────────────────────────────────────
    PromptEntry(
        id="orchestrator.router.system",
        agents=("orchestrator",), when=WHEN_ALWAYS, position=POS_SYSTEM,
        label="编排器 · 路由系统提示",
        category=CAT_ROUTING, agent="orchestrator", availability=AVAIL_STATIC,
        source="MASTv2/mast/agents/orchestrator/graph.py:_ROUTER_PROMPT",
        note=("决定每一步派给哪些 agent 的提示词。每次路由调用时读取，立即生效。"
              "文本里的 JSON 示例是模型要照抄的输出格式，改坏会让路由整体失效。"),
        overridable=True,
        loader=_const("mast.agents.orchestrator.graph", "_ROUTER_PROMPT"),
    ),
    # ── per-call middleware injections ──────────────────────────────────────
    PromptEntry(
        id="mw.live_state",
        agents_from="mast.agents._shared.live_state_mw:AGENTS", when=WHEN_ALWAYS, position=POS_LAST_HUMAN, middleware="LiveStateMiddleware",
        label="实时仪器状态块",
        category=CAT_MIDDLEWARE, agent="instrument_control",
        availability=AVAIL_NEEDS_HARDWARE,
        source="MASTv2/mast/agents/_shared/live_state_mw.py:format_live_state_block",
        note=("每次模型调用前追加的实时偏压/电流/Z/扫描框读数，"
              "含「⚠️ MAGNITUDE CHECK」量级提醒。"
              "2026-07-27 坐标事故（1.2531 写成 1.2531e-6）就出在这一块的措辞里。"),
        overridable=False,
        unavailable_reason=(
            "需要实时硬件状态，无法离线渲染 —— 内容由 Nanonis 当次读数计算。"
            "要看真实注入内容，请展开下方「agent 实际收到了什么」的真实请求快照。"
        ),
    ),
    PromptEntry(
        id="mw.mode_belief.safe",
        agents_from="mast.agents._shared.mode_mw:AGENTS", when=WHEN_ON_MODE, position=POS_SYSTEM, middleware="ModeBeliefMiddleware",
        label="操作模式信念 · 安全模式 SAFE",
        category=CAT_MIDDLEWARE, availability=AVAIL_STATIC,
        source="MASTv2/mast/agents/_shared/mode_mw.py:_SAFE_BELIEF",
        note="自主度设为「安全」时追加的信念块（不修针、不电脉冲）。每次调用时读取，立即生效。",
        overridable=True,
        loader=_const("mast.agents._shared.mode_mw", "_SAFE_BELIEF"),
    ),
    PromptEntry(
        id="mw.mode_belief.semi",
        agents_from="mast.agents._shared.mode_mw:AGENTS", when=WHEN_ON_MODE, position=POS_SYSTEM, middleware="ModeBeliefMiddleware",
        label="操作模式信念 · 半自动 SEMI",
        category=CAT_MIDDLEWARE, availability=AVAIL_STATIC,
        source="MASTv2/mast/agents/_shared/mode_mw.py:_SEMI_BELIEF",
        note="自主度设为「半自动」时追加的信念块（⑰ 后：电脉冲直接执行并留痕，"
             "SEMI 与 AUTO 的差别只剩下压深度受限）。每次调用时读取，立即生效。",
        overridable=True,
        loader=_const("mast.agents._shared.mode_mw", "_SEMI_BELIEF"),
    ),
    PromptEntry(
        id="mw.instrument_profile",
        agents_from="mast.agents._shared.instrument_profile_mw:AGENTS", when=WHEN_ALWAYS, position=POS_SYSTEM, middleware="InstrumentProfileMiddleware",
        label="仪器配置块",
        category=CAT_MIDDLEWARE, availability=AVAIL_LIVE,
        source="MASTv2/mast/core/instrument_profile.py:format_profile_block",
        note=("本机 rig 的硬件事实 + 学到的 dI/dV 标定。下面是用**当前持久化配置**"
              "实时渲染的真实文本；配置改了这里就跟着变。"
              "内容由配置计算，不能直接改文本 —— 要改请改「设置 → 仪器配置」。"),
        overridable=False,
        loader=_profile_block,
        empty_note="本机尚未填写任何仪器配置，因此这一块当前**不会被注入**（空 = 真的没有内容）。",
    ),
    PromptEntry(
        id="mw.tip_context",
        agents_from="mast.agents._shared.tip_context_mw:AGENTS", when=WHEN_ALWAYS, position=POS_SYSTEM, middleware="TipContextMiddleware",
        label="当前针尖与信号链块",
        category=CAT_MIDDLEWARE, availability=AVAIL_LIVE,
        source="MASTv2/mast/core/tip_state.py:format_tip_block",
        note=("当前装在仪器里的针尖（材料/制备/形态/装入日期）＋偏压加在哪一侧＋"
              "前置放大器增益。下面是**实时渲染**的真实文本。"
              "针尖信息来自「针尖登记」（右栏针尖卡片或 register_tip 工具）；"
              "偏压极性与前放来自「设置 → 仪器配置 → 信号链」。"
              "这一块挂在共享中间件栈上，instrument_control / experiment_design / "
              "data_processing 三个 agent 的群聊与私聊都会收到 —— "
              "偏压极性对 data_processing 解释 dI/dV 谱是必需的。"),
        overridable=False,
        loader=_tip_block,
    ),
    PromptEntry(
        id="mw.experiment_prefs",
        agents_from="mast.agents._shared.experiment_prefs:AGENTS", when=WHEN_WHEN_SET, position=POS_SYSTEM, middleware="ExperimentPrefsMiddleware",
        label="实验默认偏好块",
        category=CAT_MIDDLEWARE, availability=AVAIL_LIVE,
        source="MASTv2/mast/agents/_shared/experiment_prefs.py:format_prefs_block",
        note=("操作者偏好的默认扫描/谱学参数（提示，不是安全上限）。下面是用"
              "**当前持久化偏好**实时渲染的真实文本。要改请改「设置 → 实验默认」。"),
        overridable=False,
        loader=_prefs_block,
        empty_note="尚未设置任何实验默认偏好，因此这一块当前**不会被注入**（空 = 真的没有内容）。",
    ),
    PromptEntry(
        id="mw.upstream_artifacts.header",
        when=WHEN_ON_EVENT, position=POS_SYSTEM, middleware="UpstreamArtifactMiddleware",
        label="上游产物块 · 抬头",
        category=CAT_MIDDLEWARE, availability=AVAIL_STATIC,
        source="MASTv2/mast/agents/_shared/artifact_channel.py:UPSTREAM_BLOCK_HEADER",
        note=("其他智能体已完成的产物（文献报告 / 实验方案 / 分析结果 / 草稿 / 评审）"
              "在每个智能体上下文里的抬头。抬头下面逐条列出的产物是**运行时真实数据**，"
              "不可编辑。每次调用时读取，改完立即生效。"),
        overridable=True,
        loader=_const("mast.agents._shared.artifact_channel", "UPSTREAM_BLOCK_HEADER"),
    ),
    PromptEntry(
        id="mw.upstream_artifacts.block",
        when=WHEN_ON_EVENT, position=POS_SYSTEM, middleware="UpstreamArtifactMiddleware",
        label="上游产物块 · 内容",
        category=CAT_MIDDLEWARE, availability=AVAIL_NEEDS_REQUEST,
        source="MASTv2/mast/agents/_shared/upstream_mw.py:UpstreamArtifactMiddleware",
        note=("逐条列出上游智能体的产物指针（doc_id / 版本 / 要点 / 怎么读全文）。"
              "谁能看到哪几类产物由 artifact_channel.CONSUMES 声明。"),
        overridable=False,
        unavailable_reason=(
            "内容来自本次运行的 MASTState 产物字段（哪个智能体刚存了什么报告/方案/"
            "分析），离线没有这些值，**绝不用示例产物填充** —— 拿一份不存在的报告去"
            "调试是比没有更坏的处境。请看真实请求快照。"
        ),
        empty_note="本轮上游没有任何产物，因此这一块**不会被注入**（空 = 真的没有产物）。",
    ),
    PromptEntry(
        id="mw.memory_recall",
        requires="cognition",
        when=WHEN_ON_EVENT, position=POS_LAST_HUMAN, middleware="MemoryRecallMiddleware",
        label="记忆召回块",
        category=CAT_MIDDLEWARE, availability=AVAIL_NEEDS_REQUEST,
        source="MASTv2/mast/agents/_shared/memory_mw.py:MemoryRecallMiddleware",
        note="按本轮用户提问检索出的历史记忆条目，逐轮不同。",
        overridable=False,
        unavailable_reason=(
            "只在一次真实请求内部存在（检索键 = 本轮用户提问 + 当前实验 ID），"
            "无法离线渲染。请看真实请求快照。"
        ),
    ),
    PromptEntry(
        id="mw.resume_context.experiment",
        agents=("orchestrator",), when=WHEN_ON_EVENT, position=POS_NEW_HUMAN, paths=("group",),
        label="会话恢复块 · 未完成实验",
        category=CAT_MIDDLEWARE, availability=AVAIL_NEEDS_REQUEST,
        source="MASTv2/mast/agents/_shared/resume_context.py:build_experiment_resume_block",
        note="让「继续」能接上最近一个未完成实验，而不是回「没有上下文」。",
        overridable=False,
        unavailable_reason=(
            "需要实验日志 + 存储的实时查询结果，无法离线渲染。请看真实请求快照。"
        ),
    ),
    PromptEntry(
        id="mw.request_readback",
        when=WHEN_ON_EVENT, position=POS_LAST_HUMAN, middleware="RequestReplyReadbackMiddleware",
        label="心愿单/请求回读块",
        category=CAT_MIDDLEWARE, availability=AVAIL_NEEDS_REQUEST,
        source="MASTv2/mast/agents/_shared/resume_context.py:build_request_reply_block",
        note="把用户请求的处理结果回读给 agent，避免它重复追问。",
        overridable=False,
        unavailable_reason="依赖本轮会话的请求记录，无法离线渲染。请看真实请求快照。",
    ),
    PromptEntry(
        id="mw.stall_guard.nudge",
        when=WHEN_ON_EVENT, position=POS_NEW_HUMAN, middleware="StallGuardMiddleware",
        label="空转保护 · 提醒/停机话术",
        category=CAT_MIDDLEWARE, availability=AVAIL_NEEDS_REQUEST,
        source="MASTv2/mast/agents/_shared/stall_guard_mw.py",
        note="同一个工具连续同错时插入的升级式提醒，最后强制结束该轮。",
        overridable=False,
        unavailable_reason=(
            "话术里嵌了本轮失败的工具名与次数（f-string），"
            "没有一段与运行无关的固定文本可展示。请看真实请求快照。"
        ),
    ),
    PromptEntry(
        id="mw.tool_pair_guard.synth",
        when=WHEN_ON_EVENT, position=POS_TOOL_RESULT, middleware="ToolPairGuardMiddleware",
        label="工具配对守卫 · 占位工具结果",
        category=CAT_MIDDLEWARE, availability=AVAIL_STATIC,
        source="MASTv2/mast/agents/_shared/tool_pair_guard_mw.py:_SYNTH_TOOL_RESULT",
        note=("交接后出现孤儿 tool result 时补的占位文本 —— 没有它 provider 直接 400。"
              "每次调用时读取，立即生效。"),
        overridable=True,
        loader=_const("mast.agents._shared.tool_pair_guard_mw", "_SYNTH_TOOL_RESULT"),
    ),
    PromptEntry(
        id="mw.prefill_guard.continue",
        when=WHEN_ON_EVENT, position=POS_NEW_HUMAN, middleware="ClaudePrefillGuardMiddleware",
        label="Claude prefill 守卫 · 续跑指令",
        category=CAT_MIDDLEWARE, availability=AVAIL_STATIC,
        source="MASTv2/mast/agents/_shared/prefill_guard_mw.py:_CONTINUE",
        note="Claude 不支持 assistant prefill 时替换进去的续跑指令。每次调用时读取，立即生效。",
        overridable=True,
        loader=_const("mast.agents._shared.prefill_guard_mw", "_CONTINUE"),
    ),
    PromptEntry(
        id="mw.safety_gate.block",
        agents=("instrument_control",), when=WHEN_ON_EVENT, position=POS_TOOL_RESULT, middleware="SafetyGateMiddleware",
        label="安全门 · 拦截回执",
        category=CAT_MIDDLEWARE, availability=AVAIL_NEEDS_REQUEST,
        source="MASTv2/mast/agents/_shared/safety_mw.py:SafetyGateMiddleware",
        note="技能被安全门拦下时回给 agent 的 ToolMessage（含被违反的具体限值）。",
        overridable=False,
        unavailable_reason=(
            "逐次由被拦的技能、参数与触发的限值组装，无固定文本。请看真实请求快照。"
        ),
    ),
    PromptEntry(
        id="mw.alert_delivery",
        label="电流/视觉告警 · 送达块",
        category=CAT_MIDDLEWARE,
        agents_from="mast.agents._shared.alert_delivery_mw:AGENTS",
        when=WHEN_ON_EVENT, position=POS_LAST_HUMAN,
        middleware="AlertDeliveryMiddleware",
        availability=AVAIL_NEEDS_REQUEST,
        source="MASTv2/mast/agents/_shared/alert_delivery_mw.py:AlertDeliveryMiddleware",
        note=("刚刚发生了什么 —— 电流监控告警与视觉事件的**送达**（2026-08-10）。"
              "挂最后一条 human 消息，所以它是模型看到的最新一段事实性上下文。"
              "为什么需要它：探测层全对而送达断了，一条 CRITICAL 报出来之后 "
              "agent 又扫了 13 分钟。"),
        overridable=False,
        unavailable_reason=(
            "逐次由本轮真实触发的告警组装（阈值、读数、时间），没有一段与运行"
            "无关的固定文本可展示。请看真实请求快照。"
        ),
        empty_note="本轮没有待送达的告警，因此这一块**不会被注入**。",
    ),
    PromptEntry(
        id="mw.skill_image.images",
        label="技能图像 · 图像证据",
        category=CAT_MIDDLEWARE,
        agents=("literature", "experiment_design", "instrument_control",
                "data_processing", "paper_writing", "paper_review"),
        when=WHEN_ON_EVENT, position=POS_NEW_HUMAN,
        middleware="SkillImageMiddleware",
        availability=AVAIL_NEEDS_REQUEST,
        source="MASTv2/mast/agents/_shared/vision_mw.py:SkillImageMiddleware",
        note=("技能渲染出来的 PNG 作为多模态消息送进上下文，让模型**真的看见**"
              "那一帧。research_director 没有这一块（它不跑技能）。"),
        overridable=False,
        unavailable_reason="内容是本轮技能产出的图像，离线没有。请看真实请求快照。",
        empty_note="本轮没有技能图像，因此这一块**不会被注入**。",
    ),
    PromptEntry(
        id="mw.skill_image.undelivered",
        label="技能图像 · 未送达声明",
        category=CAT_MIDDLEWARE,
        agents=("literature", "experiment_design", "instrument_control",
                "data_processing", "paper_writing", "paper_review"),
        when=WHEN_ON_EVENT, position=POS_NEW_HUMAN,
        middleware="SkillImageMiddleware",
        availability=AVAIL_NEEDS_REQUEST,
        source="MASTv2/mast/agents/_shared/vision_mw.py:SkillImageMiddleware",
        note=("图挂不上去时**说出来**的那一条（模型不支持视觉、或文件读不出来）。"
              "与上一条是两个条目：安静降级看起来和正常工作一模一样，而这一条"
              "存在的意义就是不让它长成那样。"),
        overridable=False,
        unavailable_reason="逐次由失败原因组装。请看真实请求快照。",
        empty_note="本轮没有送不出去的图像，因此这一块**不会被注入**。",
    ),
    PromptEntry(
        id="agent.instrument_control.manual_index",
        label="仪器控制 IC · Nanonis 模块速查",
        category=CAT_AGENT, agent="instrument_control",
        agents=("instrument_control",),
        when=WHEN_BUILD_TIME, position=POS_SYSTEM,
        availability=AVAIL_LIVE,
        source="MASTv2/mast/knowledge/nanonis_manual.py:modules_index",
        note=("去重后的 Nanonis 软件手册模块索引，建图时接在 IC 系统提示后面。"
              "**只放索引不放正文**：逐工具挂提示是 176 工具 × 8 条 ≈ 9k 冗余 "
              "token/轮，深度按需走 `nanonis_manual` 工具取。本机没提取手册时"
              "为空。"),
        overridable=False,
        loader=_manual_index,
        empty_note="本机尚未提取 Nanonis 手册，因此这一块当前**不会被注入**。",
    ),
    PromptEntry(
        id="agent.literature.unavailable_note",
        label="文献 LIT · 工具可用性说明",
        category=CAT_AGENT, agent="literature",
        agents=("literature",),
        when=WHEN_BUILD_TIME, position=POS_SYSTEM,
        availability=AVAIL_LIVE,
        source="MASTv2/mast/agents/literature/tools.py:unavailable_tools_note",
        note=("哪些检索工具这台机器上用不了，建图时接在 LIT 系统提示后面 —— "
              "不然它每次调用都要花一轮重新发现 web_search 是坏的。全部可用时"
              "为空。"),
        overridable=False,
        loader=_lit_unavailable_note,
        empty_note="本机文献工具全部可用，因此这一块当前**不会被注入**。",
    ),
    PromptEntry(
        id="mw.compaction.summary",
        label="历史压缩 · 摘要替换",
        category=CAT_MIDDLEWARE,
        when=WHEN_ON_EVENT, position=POS_STATE_MESSAGES,
        middleware="_MemorySinkSummarization",
        availability=AVAIL_NEEDS_REQUEST,
        source="MASTv2/mast/agents/_shared/compaction_mw.py",
        note=("上下文超过阈值时，把旧消息整段换成一段摘要（保留最近 20 条）。"
              "触发点按这个 agent **生效**的模型窗口算，不是代码默认那个。"
              "它改的是消息列表本身，不是往里加一块。"),
        overridable=False,
        unavailable_reason="摘要由本轮真实对话历史现生成，离线没有。请看真实请求快照。",
        empty_note="本轮没有触发压缩，因此消息列表**未被改写**。",
    ),
    PromptEntry(
        id="mw.tool_refine.rewrite",
        label="工具返回精炼 · 改写",
        category=CAT_MIDDLEWARE,
        when=WHEN_ON_EVENT, position=POS_TOOL_RESULT,
        middleware="ToolRefinementMiddleware",
        availability=AVAIL_NEEDS_REQUEST,
        source="MASTv2/mast/agents/_shared/tool_refine_mw.py:ToolRefinementMiddleware",
        note=("把较早的长工具返回就地压短（保最近一条不动）。**数值、坐标、单位"
              "原样保留**是它的护栏 —— 提示词见 `sub.tool_refine`。这一条记的是"
              "「改写这件事」，那一条记的是「拿什么提示词去改」。"),
        overridable=False,
        unavailable_reason="改写结果取决于本轮的工具返回，离线没有。请看真实请求快照。",
        empty_note="本轮没有够长的旧工具返回，因此**没有发生改写**。",
    ),
    PromptEntry(
        id="orchestrator.router.upstream_block",
        label="编排器 · 上游产物块",
        category=CAT_ROUTING, agent="orchestrator",
        agents=("orchestrator",),
        when=WHEN_ON_EVENT, position=POS_SYSTEM, paths=("group",),
        availability=AVAIL_NEEDS_REQUEST,
        source="MASTv2/mast/agents/orchestrator/graph.py:_render_upstream",
        note=("编排器拿**全部**产物 —— 「这一阶段是不是已经产出了」正是它要回答"
              "的问题。它一个工具都没有，所以渲染时刻意不带「怎么读全文」的"
              "工具提示。"),
        overridable=False,
        unavailable_reason="内容来自本次运行的产物字段，离线没有。请看真实请求快照。",
    ),
    PromptEntry(
        id="orchestrator.resume.pending_reply_hint",
        label="编排器 · 待读回答复提示",
        category=CAT_ROUTING, agent="orchestrator",
        agents=("orchestrator",),
        when=WHEN_ON_EVENT, position=POS_NEW_HUMAN, paths=("group",),
        availability=AVAIL_NEEDS_REQUEST,
        source="MASTv2/mast/api/routes/orchestrator.py:_resume_lead_messages",
        note=("用户答复了某个 agent 的请求时，给路由的一句**短提示**。全文交付"
              "由 `mw.request_readback` 负责 —— 这里刻意不标记已送达，否则会双"
              "投递。"),
        overridable=False,
        unavailable_reason="依赖本轮会话的请求记录，无法离线渲染。请看真实请求快照。",
    ),
    PromptEntry(
        id="workflow.no_handoff_note",
        label="工作流委托 · 无交接说明",
        category=CAT_ROUTING,
        when=WHEN_ON_EVENT, position=POS_NEW_HUMAN, paths=("workflow",),
        availability=AVAIL_NEEDS_REQUEST,
        source="MASTv2/mast/skills/composite/agent_node.py",
        note=("composite 里以工作流方式调 agent 时给它的说明（这条路上没有"
              "supervisor，交接工具不存在）。**这条路不经共享中间件栈** —— "
              "记忆召回 / 针尖块 / 心愿单回程在这里都没有，这是已知且被接受的"
              "缺口，IC 不在委托白名单里所以仪器路径无损。"),
        overridable=False,
        unavailable_reason="仅存在于一次委托调用内部。请看真实请求快照。",
    ),
    PromptEntry(
        id="mw.tool_index",
        label="工具目录（按需加载）",
        category=CAT_MIDDLEWARE,
        agents=("instrument_control",),
        when=WHEN_ALWAYS, position=POS_SYSTEM,
        middleware="ToolVisibilityMiddleware",
        availability=AVAIL_LIVE,
        source="MASTv2/mast/agents/_shared/tool_packs.py:render_index",
        note=("按需加载工具时贴在 system 末尾的目录：有哪些工具包、各管什么、"
              "怎么用 `search_tools` / `load_tool_pack` 把它们调出来。"
              "**为什么有它**：实测 IC 的工具 schema 是 354 021 字符 —— 静态"
              "系统提示词的 18.2 倍，而且中位数 545、最大的 20 个只占 24.4%，"
              "所以压缩描述治不了。收窄可见集之后每次调用只带核心 40 个"
              "（约 5.3 万字符，省 85%）。"
              "**这是目录不是门禁**：ToolNode 下面注册的仍是全部工具，"
              "SafetyGate / validator / 自主度策略一个字节没动。"),
        overridable=False,
        loader=_tool_index_block,
        empty_note="这个进程还没建过 IC 的图，或按需加载已关闭，因此没有目录可显示。",
    ),
    # ── helper-LLM prompts ──────────────────────────────────────────────────
    PromptEntry(
        id="sub.tool_refine",
        when=WHEN_ON_EVENT, position=POS_TOOL_RESULT, middleware="ToolRefinementMiddleware",
        label="工具返回精炼器 · 提示",
        category=CAT_SUB_LLM, availability=AVAIL_STATIC,
        source="MASTv2/mast/agents/_shared/tool_refine_mw.py:_REFINE_PROMPT",
        note=("压缩过长工具返回的小模型提示。⚠️ 其中「原样保留所有数值/坐标/单位」"
              "这条是防止精炼过程改写测量值的护栏 —— 删掉它等于允许改数。"
              "每次精炼时读取，立即生效。"),
        overridable=True,
        loader=_const("mast.agents._shared.tool_refine_mw", "_REFINE_PROMPT"),
    ),
)

_BY_ID: dict[str, PromptEntry] = {e.id: e for e in _SPEC}


def entries() -> tuple[PromptEntry, ...]:
    """The full inventory, in display order."""
    return _SPEC


def get_entry(prompt_id: str) -> PromptEntry | None:
    return _BY_ID.get(prompt_id)


def render_default(entry: PromptEntry) -> tuple[str, str]:
    """``(text, error)`` for *entry*'s code default.

    ``error`` non-empty means the text could not be produced; ``text`` is then
    empty. It is never backfilled with a sample.
    """
    if entry.loader is None:
        return "", entry.unavailable_reason
    try:
        return str(entry.loader() or ""), ""
    except Exception as exc:  # noqa: BLE001
        logger.info("prompt %s: default render failed: %s", entry.id, exc)
        return "", f"读取失败：{type(exc).__name__}: {exc}"


def agents_of(entry: PromptEntry) -> tuple[str, ...]:
    """这条注入会到达哪些 agent。``(ALL_AGENTS,)`` = 全员。

    优先从中间件自己的 ``AGENTS`` 常量派生（``agents_from``），这样「谁收到」
    只有一个真源；派生失败退回登记表里写死的那份，并记一条 debug —— 静静地
    答成「全员」会让这张矩阵在最需要它的时候说谎。
    """
    if entry.agents_from:
        mod_name, _, attr = entry.agents_from.partition(":")
        try:
            mod = importlib.import_module(mod_name)
            got = getattr(mod, attr or "AGENTS", None)
            if got:
                return tuple(got)
        except Exception as exc:  # noqa: BLE001
            logger.debug("agents_of(%s): %s 派生失败: %s", entry.id,
                         entry.agents_from, exc)
    if entry.agents:
        return tuple(entry.agents)
    return (ALL_AGENTS,)


def applies_to(entry: PromptEntry, agent: str) -> bool:
    got = agents_of(entry)
    return ALL_AGENTS in got or agent in got


def resolve(prompt_id: str, default: str) -> str:
    """Effective text for *prompt_id* — the override when set, else *default*.

    This is the function every consuming site should call in place of reading
    the bare constant. Fail-safe: any error yields *default*.
    """
    return _ovr.resolve(prompt_id, default)


__all__ = [
    "ALL_AGENTS",
    "AVAIL_LIVE", "AVAIL_NEEDS_HARDWARE", "AVAIL_NEEDS_REQUEST", "AVAIL_STATIC",
    "CAT_AGENT", "CAT_MIDDLEWARE", "CAT_ROUTING", "CAT_SUB_LLM",
    "POS_LAST_HUMAN", "POS_NEW_HUMAN", "POS_STATE_MESSAGES", "POS_SYSTEM",
    "POS_TOOL_RESULT", "POS_TOOLS",
    "WHEN_ALWAYS", "WHEN_BUILD_TIME", "WHEN_ON_EVENT", "WHEN_ON_MODE",
    "WHEN_WHEN_SET",
    "PromptEntry", "agents_of", "applies_to", "entries", "get_entry",
    "render_default", "resolve",
]
