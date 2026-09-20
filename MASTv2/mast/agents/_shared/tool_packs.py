"""工具包：把几百个工具分成「常驻的一小撮」＋「按需取的若干包」。

## 为什么需要它（2026-08-24 实测）

一次 instrument_control 的模型调用里，**工具 schema 是 354 021 字符 —— 静态
系统提示词（19 401）的 18.2 倍**。在这个数字出现之前，「上下文冗长」一直被当成
提示词问题在治；把 IC 提示词从 18 k 砍到 9 k，对单次请求体积的影响不到 3%。

而且这不是「几个大工具」的问题：392 个工具的 schema 中位数是 545 字符，
**最大的 20 个加起来只占 24.4%**。压缩描述治不了长尾。唯一有效的办法是让大多数
工具**不出现在这一次调用里**。

## 它不改变什么（这一条最重要）

2026-08-20 的裁决是「工具面全开 + 服务端包络裁决」，理由是：不给工具防的是模型
*想不到*去做，防不了它换条路做；真正拦住过事故的是**包络**不是**在场**。

这里改的是**可见性**，不是**可达性**：

* ``ToolNode`` 仍然注册全部工具 —— 模型只要报得出名字，调用照常执行。
* SafetyGate / validator 数值界 / 自主度策略 / op 状态机**一个字节都没动**。
* 任何工具都能通过 ``search_tools`` 找到、通过 ``load_tool_pack`` 调出来，
  一次对话内**只增不减**（这既是为了别让模型刚看见就丢，也是为了 prompt
  cache：可见集单调增长时，前缀才稳定）。

换句话说：这是**目录**，不是**门禁**。门禁在别处，而且没动。

## 分包依据

包按 ``SkillMetadata.tags`` 派生（标签体系已经很完整：spectroscopy 55、
scan 48、pll 45、z 44、tip 34……），不手抄名单 —— 手抄的名单会在加技能时静静
漂掉。``CORE_NAMES`` 是唯一一处显式名单，而它由
``tests/v2/unit/agents/test_tool_packs.py`` 钉住：**IC 系统提示词里点名的每一个
工具都必须在核心包里**，否则提示词是在叫模型调一个它看不见的东西。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

logger = logging.getLogger(__name__)

CORE = "core"


@dataclass(frozen=True)
class ToolPack:
    name: str
    label: str
    summary: str
    tags: tuple[str, ...] = ()
    #: 额外显式成员（标签盖不到的）。
    names: frozenset[str] = frozenset()


#: 包的定义。顺序 = 目录里的展示顺序。
PACKS: tuple[ToolPack, ...] = (
    ToolPack("scan", "扫描成像",
             "配置/启动/停止扫描、扫描框与角度、缓冲区与通道、图案化扫描、标记点",
             ("scan", "pattern", "marks", "image")),
    ToolPack("spectroscopy", "谱学",
             "STS / dI-dV / 偏压谱、扫频、锁相与解调器、谱学参数与采集",
             ("spectroscopy", "sts", "sweep", "lockin", "demodulator", "didv")),
    ToolPack("tip", "针尖处理",
             "修针（脉冲/扎针/成形）、针尖质量评估、原子级针尖制备与验证",
             ("tip", "atomic", "manipulation")),
    ToolPack("zctrl", "Z 反馈与进退针",
             "Z 控制器与增益、setpoint、进针/退针、Z 限位与安全针",
             ("z", "gain", "approach")),
    ToolPack("motion", "位移与漂移",
             "粗动马达、压电范围与标定、FolMe 精确移动、漂移跟踪与补偿",
             ("motor", "piezo", "folme", "range", "calibration", "drift")),
    ToolPack("signals", "信号与输出",
             "信号通道读写、用户输出、示波器、电流/偏压底层接口",
             ("signal", "output", "user_output", "oscilloscope", "current", "bias")),
    ToolPack("pll", "PLL / qPlus",
             "锁相环、共振频率与相位、振幅与激励 —— **STM 模式下基本用不到**",
             ("pll", "frequency", "phase", "oscillation", "amplitude", "excitation")),
    ToolPack("analysis", "在线分析",
             "图像质量与 FFT、晶格检测、平整区域、团簇形貌等在线判读",
             ("analysis", "fft", "lattice", "quality")),
    ToolPack("safety", "保护与急停",
             "Z 限位、安全针、急停、各类 stop",
             ("safety", "stop", "limit")),
    ToolPack("util", "杂项与脚本",
             "会话/文件、日志、TCP 与数据记录、通用工具",
             ("util", "script", "generic", "session", "file")),
    ToolPack("optics", "光学台 / TERS",
             "光学台位移与扫描、激光与干涉仪（本机可能没有这套硬件）",
             ("optics", "beam", "interferometer", "laser")),
    ToolPack("multiprobe", "多探针",
             "多探针专用接口（本机可能没有这套硬件）",
             ("multiprobe",)),
)

_PACK_BY_NAME = {p.name: p for p in PACKS}

#: 核心包：**永远可见**。
#:
#: 判据不是「重要」，是「IC 的系统提示词点名要它用的」＋「没有它就寸步难行的
#: 状态读数」。多放一个的代价是每轮几百字符；少放一个的代价是提示词在叫模型调
#: 一个它看不见的东西 —— 后者贵得多，所以这份名单宁可宽一点。
#:
#: ``test_tool_packs.py`` 从 IC 系统提示词里抓工具名反过来核这份名单。
CORE_NAMES: frozenset[str] = frozenset({
    # ── 最常做的那几件事，一步到位 ──────────────────────────────────
    "ScanAt",                       # 扫图的唯一入口，也是最贵的一个（4.6k）——
                                    # 但它的调用频率决定了它必须常驻
    "ApproachTip", "AutoApproach", "TryEngageController", "Withdraw",
    "SetBias", "SetSetpoint", "ApplyZCtrlPreset",
    "StopScan", "RelocateCoarseXY",
    # ConditionTip：视觉告警 CRITICAL 路径上的**即时**处置（StopScan →
    # ConditionTip）。这条路上多一次检索就是多让一个坏针尖再扫一会儿，
    # 所以它留在核心，而它的底层零件（TipShape / TipPulse）不留。
    "ConditionTip",
    # 调平：看到倾斜就要立刻反应，而且这两个加起来只有 701 字符。
    "GetPiezoTilt", "SetPiezoTilt",
    # ── 状态读数：没有它们连「现在是什么情况」都答不出 ──────────────
    "GetScanStatus", "GetBias", "GetCurrent", "GetSetpoint",
    "GetZPosition", "GetXYPosition", "GetZController", "GetScanFrame",
    "GetTipStatus", "GetTemperature", "ReadHardwareEvents",
    # ── 扫描地图 / 会话（非硬件，但每轮都可能用）────────────────────
    "show_plan_on_map", "clear_plan_on_map",
    # ── 缓冲区读（非阻塞，随时可用）────────────────────────────────
    "read_latest_tip_status", "get_scan_progress", "get_tip_history_since",
    # ── 手册检索 ────────────────────────────────────────────────────
    "nanonis_manual",
})

#: 提示词点名了、但**故意放在包里**的工具 —— 用之前要先取一次。
#:
#: ## 为什么单列这一类（2026-08-24 第二轮收紧）
#:
#: 第一版的核心判据是「凡提示词点名的都进核心」。那是个**派生规则**，不是
#: **成本感知的选择**，结果是核心 40 个、52 724 字符，平均每个 1 318 —— 是全体
#: 中位数（545）的 2.4 倍。因为核心被塞满了大件复合技能，而其中一多半按提示词
#: **自己的定义**就是罕用的：
#:
#:   PrepareNobleTip 5 670  「要求才跑」「几十分钟」
#:   ForgeAuTip      4 565  「可能一个多小时、走好几个站点」
#:   ConfigureScan   2 799  「**仅限**用户逐项点名底层参数、或调试仪器时」
#:
#: 算术是这样的：**砍核心省的钱按「每次调用」付，而多一次检索只付一次。**
#: 一轮里这些工具至多被用到一次，却在每一次模型调用里都要付一遍 schema 的钱。
#:
#: ## 代价与它的边界
#:
#: 代价是模型要多走一次 `load_tool_pack`。所以**提示词必须写清去哪取** ——
#: 「叫模型调一个它看不见的东西」是这套机制唯一真正的失败模式，而这份名单把
#: 它从「隐式漏掉」变成「显式声明 + 提示词配套」。
#: `test_tool_packs.py::test_deferred_tools_are_reachable_and_the_prompt_says_how`
#: 钉住配套那一半。
#:
#: **不放进来的判据**：这个工具处在一条「多等一轮就有实际代价」的路径上。
#: ConditionTip 因此留在核心（针尖坏了还在扫），ScanAt 因此留在核心（最高频）。
DEFERRED_NAMES: dict[str, str] = {
    # 修针的重活与底层零件 —— 全在 `tip` 包
    "PrepareNobleTip": "tip",
    "ForgeAuTip": "tip",
    "TipShapeWithReadback": "tip",
    "TipShape": "tip",
    "TipPulse": "tip",
    # 底层扫描参数：提示词自己说「仅限用户逐项点名、或调试仪器时」
    "ConfigureScan": "scan",
    "SetScanSpeed": "scan",
    "SetScanBuffer": "scan",
    "FullScan": "scan",
    "StartScan": "scan",
    # 成组采集与谱学
    "GridSTS": "spectroscopy",
    "ConfigureSTS": "spectroscopy",
    "AcquireSTS": "spectroscopy",
    "BatchRegionsScan": "scan",
    "SurveySurface": "scan",
    # 漂移：一轮里至多标定一次
    "DriftTrack": "motion",
    "SetDriftCompensation": "motion",
    # 压电标定（2026-08-25 由另一条工作线加进 IC 提示词并登记在此）。
    #
    # 它们靠标签本来就在 motion 包里 —— 登记在这里是为了让**提示词点名**与
    # **默认可见性**这两件事对上账：提示词那一节写着
    # `load_tool_pack("motion")`，`DEFERRED_NAMES` 就把它们钉死在 motion 里，
    # 那句话才不会因为哪个技能的 tags 一改就变成假话。
    #
    # 为什么不进核心：标定是**低频**动作（换制冷剂 / 换扫描器 / 重标定之后各跑
    # 一次），而多等一轮检索在这条路径上没有代价 —— 与 ConditionTip 那种
    # 「视觉告警 CRITICAL 后立刻要用」的路径不是一回事。
    "CheckPiezoRange": "motion",
    "CalibratePiezoFromLattice": "motion",
    "CalibratePiezoMultiAngle": "motion",
    "AcquireAngleSeriesForCalibration": "motion",
    "ReconcileSafetyEnvelope": "motion",
    # 出图判读：只在「怀疑双针尖 / 要看图」时用
    "AnalyzeScanImage": "analysis",
}

#: 提示词点名了、但**故意不进核心**的工具。
#:
#: 它们出现在 IC 系统提示词的「硬安全拦截」一段里，而那一段说的是「这些会被
#: 直接拒绝、不要重试、没有批准会到来」—— 提示词提到它们是为了**说明它们被
#: 禁止**，不是叫模型去用。所以「提示词点名的必须可见」那条判据对它们不成立。
#:
#: 藏起来是更安全的一侧，不是更危险的一侧：模型看不见就不会试；真的需要粗动
#: 时核心里有 RelocateCoarseXY；真要找到它们，`search_tools("粗动")` 取出
#: motion 包之后照样能调，然后照常被 SafetyGate 拒绝并拿到理由。
#:
#: 这份名单存在的意义是**让这个决定显式且可复核**——不写下来的话，闸门要么被
#: 放宽成摆设，要么逼着把危险工具塞进核心。
PROHIBITED_IN_PROMPT: frozenset[str] = frozenset({
    "MotorMove", "MotorMoveClosedLoop",      # 开环粗动 Z 逼近：撞针风险
    "EnableSafeTip", "SetZLimitsEnabled",    # 关掉硬件保护
    "SetBiasCalibration", "SetCurrentCalibration",  # 改标定 = 架空所有电压/电流界
    "SetMotorFreqAmp",                       # 粗动马达驱动电压/频率
})

#: 这些前缀的工具**永远**在核心包里：它们不是仪器技能，而是 agent 的骨架
#: （交接、提问、实验会话、conduct、记忆、后台任务、以及本模块的两个元工具）。
#: 把它们分包会让 agent 连「怎么找工具」都找不到。
CORE_PREFIXES: tuple[str, ...] = (
    "handoff_to_", "ask_user", "request_user_action", "check_my_requests",
    "start_experiment", "end_experiment", "start_sample", "end_sample",
    "rename_experiment", "rename_sample",
    "remember", "recall", "list_memories", "search_memories",
    "conduct_", "spawn_background_task",
    "search_tools", "load_tool_pack",
    # ── 「去查」那一族：知识库 / 预定义工作流 / 故障诊断 / 参考文献 ──────
    #
    # 这一族**整族进核心**，理由不是「它们重要」，是三条加起来：
    #
    # 1. **它们就是 v2 知识设计的实现方式。** 裁决是「知识走拉取式而不是注入
    #    式」—— agent 需要时自己去查，而不是把知识预先塞进系统提示词。把查询
    #    入口本身藏进包里，等于把那条裁决架空一半。
    # 2. **便宜。** 十个加起来 2 872 字符，八个不到 350 —— ScanAt 一个就 4 595。
    # 3. **它们是「先查再动」的入口。** 藏一步的代价不是多一轮，是模型压根想不
    #    起来去查，于是直接凭空猜参数 —— 那正是「移除发明数字的诱因」要防的。
    #
    # 2026-08-24 审计发现：这十个里**没有一个**被 IC 提示词提到过。工具建好了、
    # 挂上了、从来没被介绍过 —— 「一个模型从没被告知的工具等于不存在」。
    # 而 `get_map_analysis` 的 docstring 自己写着「这是判断『扫了哪 / 哪不能去 /
    # 下一步去哪 / 该不该换区』的唯一依据」，提示词却只说「地图告诉你没地方了就
    # 粗动换位」，从没说**怎么问地图**。提示词里的「# 动手之前先查」一节是这次
    # 补上的另一半。
    "query_knowledge", "get_workflow_advice", "get_skill_guidance",
    "get_literature_parameters", "get_fault_diagnosis", "get_noise_reference",
    "get_measurement_template", "search_deep_reference", "read_reference_section",
    "get_map_analysis", "get_next_scan_position", "get_coarse_map",
    "get_markers_near", "lookup_sample", "describe_skills",
    # ── 技能工坊（2026-08-25）───────────────────────────────────────────
    #
    # 整族进核心，理由与上一族**同构**：`skill_catalog` 是「官方技能优先」那条
    # 阶梯的强制第一步（造之前先查有没有现成的）。把它藏进包里就把阶梯倒过来了
    # —— 模型想不起来去查，于是直接开始写 spec，而这个仓的教训正是「同一个动作
    # 的 N 份实现往往只有一份是对的」。
    #
    # 便宜也成立：五个都是薄壳，参数是 str/int，没有一个技能 schema 那种量级。
    # `run_composite` 必须在核心还有一个结构性理由：它是「刚 save 出来的技能本轮
    # 还不在工具表里」的**唯一**桥，而它要用到的那一轮，恰恰是包还没取的那一轮。
    "skill_catalog", "draft_composite", "save_composite", "run_composite",
    "propose_python_skill",
)


@dataclass(frozen=True)
class Catalog:
    """一个 agent 的工具目录：谁在核心、谁在哪个包、各包多大。"""

    agent: str
    #: tool name → 它属于哪些包（可能多个；核心用 ``{CORE}``）。
    packs_by_tool: dict[str, frozenset[str]] = field(default_factory=dict)
    #: pack name → 成员
    tools_by_pack: dict[str, tuple[str, ...]] = field(default_factory=dict)
    #: tool name → 一行摘要（给 search_tools 用）
    summary_by_tool: dict[str, str] = field(default_factory=dict)
    #: pack name → schema 字符数（近似，用于目录里说明代价）
    chars_by_pack: dict[str, int] = field(default_factory=dict)
    core_chars: int = 0
    total_chars: int = 0

    @property
    def core(self) -> tuple[str, ...]:
        return self.tools_by_pack.get(CORE, ())

    def visible(self, loaded: Iterable[str]) -> frozenset[str]:
        """核心 ＋ 已加载的包 = 这一次调用能看见的工具名。"""
        names = set(self.tools_by_pack.get(CORE, ()))
        for p in loaded or ():
            names.update(self.tools_by_pack.get(p, ()))
        return frozenset(names)

    def known_packs(self) -> tuple[str, ...]:
        return tuple(p.name for p in PACKS if self.tools_by_pack.get(p.name))


def _tags_of(meta: Any) -> set[str]:
    return {str(t).lower() for t in (getattr(meta, "tags", None) or [])}


def classify(name: str, meta: Any) -> frozenset[str]:
    """这个工具属于哪些包。核心工具返回 ``{CORE}``（核心不再进别的包）。"""
    if name in CORE_NAMES or name.startswith(CORE_PREFIXES):
        return frozenset({CORE})
    tags = _tags_of(meta)
    hits = {p.name for p in PACKS if (tags & set(p.tags)) or name in p.names}
    # 明确 defer 的工具**钉在它被声明的那个包里**，不靠标签落 ——
    # 提示词写着「用之前先 load_tool_pack('tip')」，那它就必须真的在 tip 里。
    # 靠标签落的话，某个技能的 tags 改一次，提示词那句话就成了假话。
    declared = DEFERRED_NAMES.get(name)
    if declared:
        hits.add(declared)
    return frozenset(hits) if hits else frozenset({"util"})


def _summary_of(meta: Any, tool: Any) -> str:
    desc = (getattr(meta, "description", "") or
            getattr(tool, "description", "") or "")
    first = str(desc).strip().splitlines()[0] if desc else ""
    return first[:160]


def build_catalog(agent: str, tools: Any, registry: Any = None) -> Catalog:
    """建图时算一次。**永不抛** —— 目录坏了就退化成「全部可见」。"""
    from mast.prompts import tool_surface as _ts

    metas: dict[str, Any] = {}
    if registry is not None:
        try:
            metas = {m.name: m for m in registry.list_skills()}
        except Exception:  # noqa: BLE001
            metas = {}

    packs_by_tool: dict[str, frozenset[str]] = {}
    tools_by_pack: dict[str, list[str]] = {}
    summary_by_tool: dict[str, str] = {}
    chars_by_pack: dict[str, int] = {}
    core_chars = 0
    total_chars = 0

    for t in (tools or ()):
        name = str(getattr(t, "name", "") or "")
        if not name:
            continue
        meta = metas.get(name)
        try:
            _, chars = _ts._one(t)
        except Exception:  # noqa: BLE001
            chars = 0
        total_chars += chars
        where = classify(name, meta)
        packs_by_tool[name] = where
        summary_by_tool[name] = _summary_of(meta, t)
        for p in where:
            tools_by_pack.setdefault(p, []).append(name)
            chars_by_pack[p] = chars_by_pack.get(p, 0) + chars
        if CORE in where:
            core_chars += chars

    return Catalog(
        agent=agent,
        packs_by_tool=packs_by_tool,
        tools_by_pack={k: tuple(sorted(v)) for k, v in tools_by_pack.items()},
        summary_by_tool=summary_by_tool,
        chars_by_pack=chars_by_pack,
        core_chars=core_chars,
        total_chars=total_chars,
    )


# ── 目录块（进系统提示）──────────────────────────────────────────────────

INDEX_HEADER = "# 工具目录（你现在只看得见核心工具）"


def render_index(catalog: Catalog, loaded: Iterable[str] = ()) -> str:
    """给模型看的目录：有哪些包、各包管什么、怎么把它调出来。

    刻意**不列具体工具名** —— 列了就等于把 schema 换成一份更差的清单，省不下
    多少，还会诱导模型照着列表猜参数。要具体的就 ``search_tools``。
    """
    present = catalog.known_packs()
    if not present:
        return ""
    loaded_set = {p for p in (loaded or ()) if p in _PACK_BY_NAME}
    lines = [INDEX_HEADER, ""]
    lines.append(
        f"你手上有 {len(catalog.packs_by_tool)} 个工具，但**这一次只有核心的 "
        f"{len(catalog.core)} 个连同 schema 发给了你** —— 其余按需取。"
        "这不是权限限制：任何工具都取得到，取出来之后照常受安全包络约束。")
    lines.append("")
    lines.append("需要核心以外的能力时，**先取再用**：")
    lines.append("  · `search_tools(\"你想做的事\")` —— 按描述找，命中的包会**自动"
                 "调出来**，下一步就能直接调。找不到就如实说找不到。")
    lines.append("  · `load_tool_pack(\"包名\")` —— 已经知道要哪一类时直接取。")
    lines.append("一次对话里取过的包**不会再收回去**，不用反复取。")
    lines.append("")
    lines.append("可取的包：")
    for name in present:
        pack = _PACK_BY_NAME[name]
        n = len(catalog.tools_by_pack.get(name, ()))
        mark = "（已取）" if name in loaded_set else ""
        lines.append(f"  · `{name}` {pack.label}{mark} —— {pack.summary}（{n} 个）")
    return "\n".join(lines)


# ── 检索 ─────────────────────────────────────────────────────────────────
#
# **中文查询必须命中。** 第一版只做英文分词，实测 `search_tools("锁相放大器 调制
# 幅度")` 与 `search_tools("数据记录")` 都是零命中 —— 而工具名和 description 几乎
# 全是英文，用户和 agent 却都说中文。一个「查得到但只在你用英文问的时候」的检索
# 工具，症状是 agent 搜一次、失败、然后照着猜一个工具名，正好是这套机制要防的事。
#
# 修法两条：中英同义词表（下面这张），以及**包级兜底**（工具级零命中时，拿查询去
# 匹配包的中文标签与说明，至少把对的那一包捞出来）。

#: 中文（含常见英文别名）→ 工具名/描述里真正会出现的英文词。
#: 只收 STM 领域里**会被用来找工具**的词，不是词典。
SYNONYMS: dict[str, tuple[str, ...]] = {
    "锁相": ("lockin", "lock"), "锁相放大器": ("lockin", "lock"),
    # 「调制」在 STM 语境里几乎总是指锁相调制；别拉出 amplitude —— 那会把
    # SetMotorFreqAmp（粗动马达驱动幅度）排到锁相配置前面去。
    "调制": ("modulation", "modulate", "lockin"),
    "解调": ("demod", "demodulator"),
    "幅度": ("amplitude", "amp"), "振幅": ("amplitude", "amp"),
    "频率": ("freq", "frequency"), "共振": ("resonance", "resonant", "freq"),
    "相位": ("phase",), "扫频": ("sweep", "freq"),
    "谱": ("spectr", "sts", "spectrum"), "能谱": ("spectr", "sts"),
    "谱学": ("spectr", "sts"), "偏压谱": ("bias", "spectr", "sts"),
    "扫描": ("scan",), "扫图": ("scan",), "成像": ("scan", "image"),
    "图像": ("image", "scan"), "帧": ("frame",),
    "偏压": ("bias",), "电压": ("bias", "voltage"), "电流": ("current",),
    "设定点": ("setpoint",), "增益": ("gain",),
    "反馈": ("controller", "zctrl", "feedback"),
    "进针": ("approach", "engage"), "退针": ("withdraw", "retract"),
    "针尖": ("tip",), "扎针": ("tipshape", "shape", "poke", "tip"),
    "脉冲": ("pulse",), "修针": ("tip", "condition", "shape"),
    "马达": ("motor",), "粗动": ("motor", "coarse"), "移动": ("move", "folme"),
    "压电": ("piezo",), "标定": ("calibration", "calib"),
    "漂移": ("drift",), "倾斜": ("tilt",), "调平": ("tilt",),
    "温度": ("temperature", "temp"),
    "示波器": ("oscilloscope", "osc"), "信号": ("signal",),
    "输出": ("output",), "通道": ("channel", "signal"),
    "记录": ("log", "record"), "数据记录": ("datalog", "log"),
    "日志": ("log",), "历史": ("history",),
    "急停": ("stop", "estop"), "停止": ("stop",),
    "限位": ("limit",), "安全": ("safe", "safety"),
    "原子": ("atom", "atomic"), "晶格": ("lattice",),
    "缺陷": ("defect",), "台阶": ("step",),
    "光学": ("optic", "beam", "laser"), "激光": ("laser", "beam"),
    "多探针": ("multiprobe",),
    "网格": ("grid",), "位置": ("position", "xy"), "坐标": ("position", "xy"),
    "状态": ("status", "get"), "读": ("get", "read"), "设": ("set",),
}


def _tokens(text: str) -> list[str]:
    return [t for t in re.split(r"[^0-9A-Za-z一-鿿]+", (text or "").lower()) if t]


def _cjk_grams(text: str) -> set[str]:
    """CJK 串切成 1–3 字的片段 —— 中文没有空格，整词匹配等于不匹配。"""
    out: set[str] = set()
    for run in re.findall(r"[一-鿿]+", text or ""):
        for n in (2, 3):
            for i in range(len(run) - n + 1):
                out.add(run[i:i + n])
        if len(run) <= 3:
            out.add(run)
    return out


def _expand(query: str) -> tuple[set[str], set[str]]:
    """``(英文检索词, 中文片段)``。中文片段经同义词表折成英文词一并返回。"""
    latin = {t for t in _tokens(query) if t.isascii()}
    grams = _cjk_grams(query)
    for g in grams | {t for t in _tokens(query) if not t.isascii()}:
        for en in SYNONYMS.get(g, ()):
            latin.add(en)
    return latin, grams


def search(catalog: Catalog, query: str, *, limit: int = 12) -> list[dict[str, Any]]:
    """按名字与描述找工具。返回 ``[{name, packs, summary, score}]``，最相关在前。

    评分粗糙但可解释：名字精确/前缀命中权重最高，其次名字词命中，再次描述命中。
    刻意不用 embedding —— 这里要的是「找得到」而不是「排得漂亮」，而且一个离线
    可解释的判据比一个需要模型的判据更适合放在热路径上。
    """
    q = (query or "").strip()
    if not q:
        return []
    ql = q.lower()
    latin, grams = _expand(q)
    out: list[dict[str, Any]] = []
    for name, packs in catalog.packs_by_tool.items():
        nl = name.lower()
        summary = catalog.summary_by_tool.get(name, "")
        sl = summary.lower()
        score = 0.0
        if nl == ql:
            score += 100
        elif ql and (nl.startswith(ql) or ql in nl):
            score += 40
        # 驼峰切分会把 "LockIn" 拆成 Lock+In，于是 "lockin" 反而对不上；
        # 把整名小写也放进词表补上这一刀。
        ntok = set(_tokens(re.sub(r"(?<!^)(?=[A-Z])", " ", name))) | {nl}
        score += 12 * len(latin & ntok)
        for t in latin:
            if len(t) >= 3 and t in nl:
                score += 8
        stok = set(_tokens(summary))
        score += 4 * len(latin & stok)
        for t in latin:
            if len(t) >= 4 and t in sl:
                score += 2
        # 描述里偶尔有中文（composite 技能多半是中文写的）。
        score += 3 * len(grams & _cjk_grams(summary))
        if score <= 0:
            continue
        out.append({"name": name, "packs": sorted(packs), "summary": summary,
                    "score": score})
    out.sort(key=lambda d: (-d["score"], d["name"]))
    return out[:limit]


def search_packs(catalog: Catalog, query: str) -> list[str]:
    """工具级零命中时的兜底：拿查询去撞**包的中文标签与说明**。

    存在的理由：工具名与 description 几乎全是英文，而问问题的是中文。撞不到
    具体工具不等于没有这个能力 —— 至少要把对的那一包捞出来，让模型取了之后
    自己看 schema。仍然撞不到就返回空，由调用方如实说「没找到」。
    """
    q = (query or "").strip()
    if not q:
        return []
    latin, grams = _expand(q)
    scored: list[tuple[float, str]] = []
    for pack in PACKS:
        if not catalog.tools_by_pack.get(pack.name):
            continue
        blob = f"{pack.name} {pack.label} {pack.summary}"
        score = 0.0
        score += 6 * len(grams & _cjk_grams(blob))
        btok = set(_tokens(blob))
        score += 4 * len(latin & btok)
        score += 5 * len(latin & set(pack.tags))
        if score > 0:
            scored.append((score, pack.name))
    scored.sort(key=lambda kv: (-kv[0], kv[1]))
    return [n for _, n in scored[:3]]


def packs_of(hits: Iterable[dict[str, Any]]) -> list[str]:
    """命中里出现过的、可加载的包名（核心不算 —— 它本来就在）。"""
    seen: list[str] = []
    for h in hits or ():
        for p in h.get("packs") or ():
            if p != CORE and p in _PACK_BY_NAME and p not in seen:
                seen.append(p)
    return seen


def pack_exists(name: str) -> bool:
    return name in _PACK_BY_NAME


def pack_label(name: str) -> str:
    p = _PACK_BY_NAME.get(name)
    return p.label if p else name


__all__ = [
    "CORE", "CORE_NAMES", "CORE_PREFIXES", "DEFERRED_NAMES",
    "INDEX_HEADER", "PACKS",
    "PROHIBITED_IN_PROMPT", "SYNONYMS",
    "Catalog", "ToolPack", "build_catalog", "classify", "pack_exists",
    "pack_label", "packs_of", "render_index", "search", "search_packs",
]
