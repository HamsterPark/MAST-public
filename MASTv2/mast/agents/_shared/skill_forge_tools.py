"""技能工坊 —— agent 手里的「查 / 起草 / 保存 / 执行 / 提议」五件套。

## 它解决的问题

底层早就齐了:声明式 IR(``skills/composite/spec.py``)、通用解释器
(``interpreter.SpecComposite``)、版本库带 CAS(``version_store``)、热注册闭环
(``loader.register_spec``)、设计期校验内核(``webui/builder_api.validate_spec_payload``)。
缺的只有一件:**没有任何 agent 手里有创建/保存 composite 的工具** —— 那套东西
只挂在 GUI 的 builder 页上。于是仪器控制 agent 面对「这一串八步我今天要做三遍」
时,唯一的办法是把八步重打三遍,而它明明有能力把它们组合起来。

## 官方技能优先(2026-08-21 定案)

优先序是**三级阶梯,不跳级**:

1. **官方技能**(~254 个 builtin + 手写 composite)覆盖得了的,用官方的。它们的
   前置条件、回滚和参数包络是实机验证过的,新造的没有。
2. 官方**单个**技能覆盖不了的序列 / 分支 / 重试,才组合官方技能成新 composite。
3. 组合也表达不了(要全新的原子 TCP 能力),才 :func:`propose_python_skill` 提
   草稿 —— 落盘不启用,用户审过加白名单才生效。

这条优先序不只写在提示词里(说服模型是本仓已经失败过四次的做法,见
既有教训),还有三道**结构性**的:

* ``skill_catalog`` 每行带来源标签 —— 官方与自造在目录里一眼可分;
* ``_agent_track_lint`` 的**纯别名拒** —— 给官方技能套个壳的单步 spec 直接拒,
  并指回本体;
* ``draft_composite`` 的 ``hints`` —— 草稿的步骤集被某个现成 composite 覆盖时
  附一句「先看它够不够用」。**建议不阻拦**:照搬 DP 的 ``_py_hint``
  (``agents/data_processing/tools.py``),那里第一版设计成命中即拒、后来撤销了
  —— 是路由到更好的东西,不是拦路。

## 安全:组合不是绕过

这一族是**普通 @tool,没有 skill_metadata** —— 于是 ``safety_mw`` 与
``auto_approval_mw`` 看到 ``meta_obj is None`` 直接放行。这是**有意**的,治理不在
工具面而在执行层:

* 组合出来的技能每个子步都走 ``ExecutionContext.run``,那里有 abort 闸、sample
  gate、五条 Layer-0 硬闸(粗动 Z 进针 / 关保护 / 改标定 / 设粗动驱动 / 裸横向
  粗动)、参数包络(拒绝不夹紧)、以及 ⑰ 之后 DANGEROUS 的「执行 + 通知」;
* 安全级与能力标签由 ``interpreter._inherited_safety_level`` /
  ``_inherited_capabilities`` 从子步**继承**,声明只能收紧不能放松 —— 一个含
  ``TipPulse`` 的 spec 自称 ``auto`` 是没有用的。

**唯一需要在这里补的一层是操作模式闸(SAFE / SEMI)。** 它只活在
``agents/_shared/safety_mw._mode_block`` 里,而那道中间件对没有 skill_metadata 的
工具直接放行 —— 也就是说 :func:`run_composite` 会从它旁边绕过去。
``ExecutionContext.run`` 里**没有**模式闸。所以本模块自己判一次,判据取的是同一批
单源函数(``core.operating_mode.safe_mode_active`` + ``core.safety.is_tip_shaping``
/ ``is_electrical_pulse``),不另写一份。没有这一层,这个模块就是一条 SAFE 旁路。

## 不做的事

* **不执行任何 Python**。第 3 级只写文件,``allow_exec`` 一直是 False;真正的门
  是 ``skills/custom_loader`` 的 ``enabled.json`` 白名单(用户的显式动作)。
* **不自己调 registry.register**。保存一律经 ``loader.register_spec``,因为撞名
  拒绝和「引用了不存在的技能就不注册」两道守卫长在那里。
* **不新开执行收口**。:func:`run_composite` 现场 ``wrap_skill`` 再分发,走的是
  agent 路径本来那一条(RFC D7:三个收口已对齐,不开第四个)。
"""

from __future__ import annotations

import json
import logging
from typing import Annotated, Any

from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId

logger = logging.getLogger(__name__)

__all__ = [
    "make_skill_forge_tools",
    "save_composite_impl",
    "propose_python_skill_impl",
    "FORGE_TOOL_NAMES",
    "AGENT_NODE_TYPES",
    "AGENT_HW_LOOP_MAX_ITER",
    "SPEC_SYNTAX",
    "agent_track_lint",
    "official_overlap_hints",
]

#: 这一族的工具名。与 :func:`make_skill_forge_tools` 的返回一一对应,有测试钉着
#: —— 「清单说有、图上没有」是本仓踩过的形状。
FORGE_TOOL_NAMES: tuple[str, ...] = (
    "skill_catalog",
    "draft_composite",
    "save_composite",
    "run_composite",
    "propose_python_skill",
)

#: agent 轨允许的节点类型 —— 纯结构控制流。
#:
#: 禁掉的三种各有理由,都不是「不信任模型」:
#:   * ``human``  —— 走 LangGraph ``interrupt()``。夜里没人时它不是「等一下」,
#:     是死锁,而自主运行正是这个模块存在的场景。
#:   * ``llm``    —— 需要 persona 档案与模型配置(``composite/persona.py``),
#:     而且 route 模式的闭集设计是给编辑器用的;M2 再评估。
#:   * ``agent``  —— ``spec.py`` 本来就禁止委托 instrument_control。
#:
#: **只加在 agent 轨**:``CompositeSpec.validate`` 和 ``validate_spec_payload``
#: 一个字都不动,GUI 的 builder 页照旧能用全部 12 种节点。
AGENT_NODE_TYPES: frozenset[str] = frozenset({
    "step", "if", "loop", "set", "try", "break", "continue", "succeed", "fail",
})

#: 触硬件的循环必须显式声明的迭代上限。
#:
#: ``spec.DEFAULT_MAX_ITER`` 是 10000 —— 对一个每轮打脉冲的循环,那不是上限而是
#: 一夜。NL→spec 生成器的系统提示里本来就写着这条硬规则(「硬件 loop 必设
#: max_iter ≤ 100」),但**全仓没有任何代码在强制它**,只是写给模型看的一句话。
AGENT_HW_LOOP_MAX_ITER = 100

#: spec 的格式速查。**按需发**,不进任何 docstring。
#:
#: 核心包工具的 schema 每一次模型调用都在上下文里,而这段东西只在「真的要造一个
#: 技能、而且草稿还没写对」的时候有用 —— 放进 docstring 等于让每一轮都替那件
#: 偶尔发生的事付钱。跟着**失败回执**走,或者 ``draft_composite("?")`` 显式索取。
SPEC_SYNTAX = """CompositeSpec 格式:
{"name","description","safety_level":"auto|confirm|dangerous",
 "params":[{"name","type":"number|int|string|bool","default","required"}],
 "nodes":[...]}

节点(agent 轨只允许这几种):
- {"type":"step","id":"s1","skill":"ScanAt","params":{...}}
- {"type":"if","id":"c1","cond":"s1['fft_quality'] > 0.3","then":[...],"else":[...]}
- {"type":"loop","id":"l1","mode":"repeat","count":"3","max_iter":10,"var":"i","body":[...]}
  (mode 还可以是 foreach(配 iterable)或 while(配 cond))
- {"type":"try","id":"t1","body":[...],"finally":[...]}  ← 收尾退针写 finally
- {"type":"set","id":"v1","var":"n","value":"n + 1"}
- {"type":"succeed","id":"ok"} / {"type":"fail","id":"no","reason":"'…'"} / break / continue

表达式(三条最容易写错的):
1. 引用某一步的结果用**下标**:s1['fft_quality'](s1 = 那个 step 的 id,绑的就是
   它的结果字典);last 指最近一步。**属性访问 s1.data.x 是被禁的。**
2. loop 的 count/cond、set 的 value、if 的 cond 都是**表达式字符串** —— 写 "3" 不是 3。
3. 把工作流参数传给某一步用 {"$expr":"参数名"};字面量直接写。

会被拒绝的:只包一个 step 的「套壳」spec(直接调本体);llm/human/agent 节点;
触硬件的 loop 没写 max_iter(上限 100);本机关闭的技能;与已有技能撞名;
子技能的必填参数没给全(回执会逐步告诉你缺哪个)。"""

#: 判「这是官方技能吗」用的来源标签(``overlay.provenance.classify_origin``)。
#: ``user_composite`` 是 SpecComposite(builder 页或 agent 造的),``custom`` 是
#: 用户 .py,``overlay`` 是覆盖层 —— 三者都不算官方。
_OFFICIAL_ORIGINS = frozenset({"builtin", "composite", "paper"})

_ORIGIN_ZH = {
    "builtin": "官方原子",
    "composite": "官方组合",
    "paper": "官方论文",
    "user_composite": "组合(用户/agent 造)",
    "custom": "自定义 .py",
    "agent_tool": "Agent 工具",
    "overlay": "覆盖层",
    "other": "其他",
}


def _j(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)


# ─────────────────────────────────────────────────────────────────────────
# 只读小工具(全部永不抛 —— 判据坏掉的失败模式必须是「少说一句」而不是「工具炸了」)
# ─────────────────────────────────────────────────────────────────────────

def _disabled_names() -> frozenset[str]:
    """两张关闭名单的并集:没有这个硬件 ∪ 这条能力没被授予。

    ``build_instrument_skill_tools`` 就是按它把技能从工具表里摘掉的,「看不见 =
    用不了」。而 spec 的子步走 ``ExecutionContext.run``,那条路**不查这张名单**
    —— 于是把被关掉的高级能力(退出 Nanonis / 脚本文件 I/O / 粗动驱动)包进一个
    composite 就能洗白。这个函数是那道洗白防线的判据。
    """
    out: set[str] = set()
    for mod in ("mast.skills.hardware_modules", "mast.skills.advanced_capabilities"):
        try:
            m = __import__(mod, fromlist=["disabled_skill_names"])
            out |= set(m.disabled_skill_names() or ())
        except Exception as exc:  # noqa: BLE001 — 名单读不到不能让工具停摆
            logger.warning("技能工坊:%s 的关闭名单读不到(%s)——"
                           "本次不按它拒绝", mod, exc)
    return frozenset(out)


def _origin_of(registry, name: str) -> str:
    """一个已注册技能的来源标签;取不到回 ``"other"``(**不是**「官方」)。"""
    try:
        return _classify(registry.get(name))
    except Exception:  # noqa: BLE001
        return "other"


def _classify(cls) -> str:
    try:
        from mast.skills.overlay.provenance import classify_origin
        return classify_origin(cls)
    except Exception:  # noqa: BLE001 — 分类器不可用时不要冒充官方
        return "other"


def _seed_template_names() -> frozenset[str]:
    """内置模板种子的名字。

    ``load_spec_skills(seed=True)`` 每次启动都会把缺失的模板写回 store。agent
    若用了同名,删除之后下次启动它会**复活**成模板的内容(这类「composite 删除后
    内置模板重启复活」的现象)——那是一个安静得没人会去查的形状。
    """
    try:
        from mast.skills.composite.templates import builtin_templates
        return frozenset(s.name for s in builtin_templates())
    except Exception as exc:  # noqa: BLE001
        logger.debug("技能工坊:模板名单读不到(%s)", exc)
        return frozenset()


def _node_types(nodes: Any) -> list[str]:
    try:
        from mast.skills.composite.spec import walk_nodes
        return [str(n.get("type") or "") for n in walk_nodes(nodes)]
    except Exception as exc:  # noqa: BLE001
        logger.debug("技能工坊:遍历节点树失败(%s)", exc)
        return []


def _step_skills(nodes: Any) -> set[str]:
    try:
        from mast.skills.composite.spec import collect_step_skills
        return set(collect_step_skills(nodes))
    except Exception as exc:  # noqa: BLE001
        logger.debug("技能工坊:收集 step 技能失败(%s)", exc)
        return set()


# ─────────────────────────────────────────────────────────────────────────
# agent 轨附加 lint
# ─────────────────────────────────────────────────────────────────────────

def agent_track_lint(spec_dict: dict, registry, *, store=None) -> list[str]:
    """agent 轨专有的拒绝条款。返回问题清单(空 = 通过)。

    ``validate_spec_payload`` 那套(未知技能、参数名/边界、z-approach 硬错、
    悬空 $expr、声明级不得低于叶子 max)已经跑过了,这里**只加** GUI 轨不需要的
    那几条(见下面的分节)。刻意不把它们塞进 ``CompositeSpec.validate``:编辑器里
    人是在场的,``human`` 节点是功能不是死锁。

    (不写「共 N 条」——本仓的写死计数每次加一条都会漂,而漂了没人会发现。)
    """
    problems: list[str] = []
    nodes = spec_dict.get("nodes") or []
    types = _node_types(nodes)

    # ── 0. 一步都没有 ────────────────────────────────────────────────
    #
    # ``CompositeSpec.validate`` 对空 nodes 是满意的(结构上确实没毛病),于是
    # 一份什么都不做的 spec 能一路存进去、注册成技能、被调用、成功返回。
    # 「成功地什么都没做」是这个仓最不想要的那种结果。
    if not _step_skills(nodes):
        problems.append(
            "这份 spec 里一个 step 都没有 —— 它不会对仪器做任何事,"
            "存下来只会变成一个「调了就成功、但什么都没发生」的技能。"
            "(要看 spec 怎么写:draft_composite(\"?\"))")
        return problems

    # ── 1. 节点类型白名单 ────────────────────────────────────────────
    bad = sorted({t for t in types if t and t not in AGENT_NODE_TYPES})
    if bad:
        why = {
            "human": "human 节点走 interrupt(),夜里没人时它是死锁不是等待",
            "llm": "llm 节点要 persona 档案与模型配置,agent 轨暂不开放",
            "agent": "spec 层本就禁止把仪器动作委托给 agent 节点",
        }
        problems.append(
            "agent 轨不允许这些节点类型:" + "、".join(bad)
            + "。" + ";".join(why.get(t, "") for t in bad if t in why)
            + "。仪器动作请用 step 节点逐个技能写出来。")

    # ── 2. 纯别名:给官方技能套壳 ──────────────────────────────────────
    #
    # 「官方优先」在这一层是结构性的:一个只有单个 step、没有任何控制流的 spec
    # 不产生新能力,只产生一个新名字 —— 而多一个名字意味着下一次有人(或有模型)
    # 会去调那个壳而不是本体,于是本体的参数说明、单位提示、包络文案全都绕过去了。
    # 本仓的原话:同一个动作的 N 份实现往往只有一份是对的。
    #
    # 只在类型这一关过了之后才判:否则一个 [step, llm] 的草稿会同时收到「不许用
    # llm」和「这是套壳」两条,而后者是前者的副作用,读起来像两个独立的问题。
    structural = [t for t in types if t in ("if", "loop", "try", "set")]
    steps = [t for t in types if t == "step"]
    if not bad and len(steps) == 1 and not structural:
        only = ""
        for n in (nodes or []):
            if isinstance(n, dict) and n.get("type") == "step":
                only = str(n.get("skill") or "")
                break
        problems.append(
            f"这份 spec 只包了一个 step、没有任何分支/循环/重试 —— "
            f"那是给 {only or '一个官方技能'} 套壳,不是新能力。"
            f"直接调 {only or '它'} 本体。"
            "(要组合就至少有第二步,或者有 if/loop/try 让它比本体多做点什么。)")

    # ── 3. 触硬件的循环必须显式封顶 ──────────────────────────────────
    for n in _iter_nodes(nodes):
        if n.get("type") != "loop":
            continue
        if not _step_skills(n.get("body") or []):
            continue          # 纯计算循环,不打硬件
        mi = n.get("max_iter")
        nid = n.get("id") or "?"
        if mi is None:
            problems.append(
                f"循环 {nid!r} 的 body 里有 step(会打到硬件),必须显式写 "
                f"max_iter(≤ {AGENT_HW_LOOP_MAX_ITER})—— 默认值是 10000,"
                "对一个每轮动针尖的循环那不是上限,是一整夜。")
            continue
        try:
            mi_i = int(mi)
        except Exception:  # noqa: BLE001
            problems.append(f"循环 {nid!r} 的 max_iter={mi!r} 不是整数")
            continue
        if mi_i < 1 or mi_i > AGENT_HW_LOOP_MAX_ITER:
            problems.append(
                f"循环 {nid!r} 的 max_iter={mi_i} 超出 agent 轨允许范围 "
                f"[1, {AGENT_HW_LOOP_MAX_ITER}]")

    # ── 4. 被关掉的技能不许经 spec 洗白 ──────────────────────────────
    disabled = _disabled_names()
    used = _step_skills(nodes)
    blocked = sorted(used & disabled)
    if blocked:
        problems.append(
            "这些技能在本机是**关闭**的(没有对应硬件,或那条高级能力没有被授予):"
            + "、".join(blocked)
            + "。把它们包进 composite 也不会打开 —— 关闭是用户的决定,"
            "要用请他到【高级】页开。")

    # ── 5. 撞名 ─────────────────────────────────────────────────────
    name = str(spec_dict.get("name") or "")
    if name:
        problems.extend(_name_collision_problems(name, registry))

    return problems


def _iter_nodes(nodes: Any):
    try:
        from mast.skills.composite.spec import walk_nodes
        yield from walk_nodes(nodes)
    except Exception:  # noqa: BLE001
        return


def _name_collision_problems(name: str, registry) -> list[str]:
    """撞名预检 —— **必须跑在 store.save 之前**。

    ``register_spec`` 会拒绝与非 spec 技能同名,而保存是在注册之前发生的:先存后
    注册的顺序会造出一个「文件在盘上、永远注册不上」的幽灵,它唯一的痕迹是
    ``last_rejected_specs()`` 里的一行,而模型什么都看不到。
    """
    out: list[str] = []
    try:
        if not registry.has(name):
            pass
        else:
            from mast.skills.composite.interpreter import SpecComposite
            existing = registry.get(name)
            if not (isinstance(existing, type) and issubclass(existing, SpecComposite)):
                out.append(
                    f"名字 {name!r} 已经被一个**非组合**技能占用 "
                    f"({getattr(existing, '__module__', '?')}."
                    f"{getattr(existing, '__qualname__', '?')})——换个名字。"
                    "同名会让后续的删除动作把内置技能一起卸掉。")
    except Exception as exc:  # noqa: BLE001
        logger.debug("技能工坊:撞名预检失败(%s)", exc)
    if name in _seed_template_names():
        out.append(
            f"名字 {name!r} 与内置模板种子同名 —— 它会在下次启动时被模板内容覆盖"
            "(种子是 `if not exists: save`,而删除之后就 not exists 了)。换名。")
    return out


# ─────────────────────────────────────────────────────────────────────────
# 官方优先:建议式路由(不阻拦)
# ─────────────────────────────────────────────────────────────────────────

def official_overlap_hints(spec_dict: dict, registry, *, store=None) -> list[str]:
    """草稿要做的事已经有现成 composite 在做时,给一句建议。**不拒绝。**

    形态照搬 DP 的 ``_py_hint``:那里第一版设计成命中就拒绝执行,后来撤销了,
    注释写着「是路由到更好的输出,不是阻拦,所以放在结果后面、用建议的语气」。
    这里同理 —— 覆盖不等于等价(参数、判据、失败处置都可能不同),硬拒会把
    「我确实需要一个不一样的」也一起拒掉。

    判据用**步骤技能集合**的包含/重合度,数据来自版本库里已存的 spec(包含手写
    composite 的声明式孪生体和内置模板),因为手写 composite 的步骤序列没有别的
    地方是可读的。
    """
    hints: list[str] = []
    mine = _step_skills(spec_dict.get("nodes"))
    if len(mine) < 2:
        return hints
    myname = str(spec_dict.get("name") or "")
    try:
        st = store if store is not None else _default_store()
        if st is None:
            return hints
        for summary in st.list_specs():
            other = str(summary.get("name") or "")
            if not other or other == myname:
                continue
            try:
                theirs = _step_skills(st.load(other).nodes)
            except Exception:  # noqa: BLE001
                continue
            if not theirs:
                continue
            if mine <= theirs:
                hints.append(
                    f"现成的 {other!r} 已经用到了你这份草稿的全部步骤"
                    f"({'、'.join(sorted(mine))})—— 先看它够不够用,"
                    "或者以它为蓝本改(它的参数与失败处置是调过的)。")
                continue
            inter = mine & theirs
            union = mine | theirs
            if union and len(inter) / len(union) >= 0.6:
                hints.append(
                    f"{other!r} 与这份草稿高度重合(共用 {'、'.join(sorted(inter))})"
                    "—— 确认你要的不是它,再继续。")
    except Exception as exc:  # noqa: BLE001 — 建议拿不到绝不能挡住起草
        logger.debug("技能工坊:重合度建议不可用(%s)", exc)
    return hints[:3]


def _default_store():
    try:
        from mast.webui.composite_panel import composite_store
        return composite_store()
    except Exception:  # noqa: BLE001
        try:
            from mast.skills.composite.version_store import CompositeVersionStore
            return CompositeVersionStore()
        except Exception as exc:  # noqa: BLE001
            logger.warning("技能工坊:版本库打不开(%s)", exc)
            return None


# ─────────────────────────────────────────────────────────────────────────
# 校验(draft / save 共用同一条)
# ─────────────────────────────────────────────────────────────────────────

def _validate(spec_dict: dict, registry, *, store=None) -> dict:
    """设计期全量校验 = 官方内核 + agent 轨附加 + 缺失技能兜底。

    官方内核是 ``webui/builder_api.validate_spec_payload`` —— **复用而不是重写**:
    未知技能、钉住的版本不存在、未知/缺失参数、字面量超包络、z-approach 硬错、
    悬空 $expr、声明安全级不得低于叶子 max,这七条都在那里,而且 GUI 与 agent
    必须给出同一个答案。
    """
    report: dict
    try:
        from mast.webui.builder_api import validate_spec_payload
        report = validate_spec_payload(spec_dict, registry=registry)
    except TypeError:
        # 老签名(没有 registry 形参)—— 仍然可用,只是它读模块全局注册表。
        from mast.webui.builder_api import validate_spec_payload
        report = validate_spec_payload(spec_dict)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "problems": [f"校验内核不可用:{exc}"], "steps": []}

    problems = list(report.get("problems") or [])
    problems.extend(agent_track_lint(spec_dict, registry, store=store))

    # 注册前那道「引用了不存在的技能就不注册」的兜底 —— 与 loader 同一个函数,
    # 这样 draft 说 ok 而 save 时被 register_spec 拒掉的情况不会发生。
    try:
        from mast.skills.composite.loader import _missing_step_skills
        from mast.skills.composite.spec import CompositeSpec
        missing = _missing_step_skills(registry, CompositeSpec.from_dict(spec_dict))
        if missing:
            problems.append(
                "引用了注册表里不存在的技能:" + "、".join(missing)
                + "(注册会被拒:否则它会先把前面的硬件步打出去再崩)")
    except Exception as exc:  # noqa: BLE001
        logger.debug("技能工坊:缺失技能兜底检查失败(%s)", exc)

    steps = report.get("steps") or []
    hard = [p for p in problems if not str(p).startswith("警告：")]
    step_errs = [(s.get("id"), e) for s in steps for e in (s.get("errors") or [])]
    # 把逐步错误**也**汇总进 problems。不汇总的话 `problems` 会是空列表而
    # `ok=False` —— 一个「不行,但没说哪儿不行」的回执,模型下一步只能瞎猜。
    for sid, e in step_errs:
        problems.append(f"步骤 {sid!r}:{e}")
    ok = not hard and not step_errs
    return {"ok": ok, "problems": problems, "steps": steps}


# ─────────────────────────────────────────────────────────────────────────
# 工厂
# ─────────────────────────────────────────────────────────────────────────

def make_skill_forge_tools(
    registry,
    context_provider=None,
    *,
    recorder=None,
    post_hook=None,
    agent_name: str = "instrument_control",
    store=None,
) -> list:
    """建这一族工具。

    Parameters
    ----------
    registry:
        **活的** SkillRegistry。它同时是目录的来源、校验的判据和执行的取类处 ——
        三者必须是同一个对象,否则会出现「目录里有、执行时说没有」。
    context_provider:
        ``callable() -> ExecutionContext``。**直接透传 IC 的那一个**,不要在这里
        自己造:abort_event 的活性是靠调用方那个 provider 保证的(修复项 的教训是
        agent 路径构造 ExecutionContext 时不传 abort_event,于是 composite 内部
        轮询一个永远不会被 set 的 Event,E_STOP 对运行中的工作流不可达)。
        ``None`` ⇒ 不产出 :func:`run_composite`(没有硬件宿主的场合,比如把这族
        工具给一个不碰仪器的 agent)。
    store:
        版本库,``None`` ⇒ 进程单例。测试注入用。
    """
    who = f"agent:{agent_name or 'unknown'}"
    tools: list = []

    # ── 1. 查 ────────────────────────────────────────────────────────
    @tool("skill_catalog")
    def skill_catalog(query: str = "", detail: str = "") -> str:
        """查现有技能目录 —— **造任何东西之前的第一步**。

        `query` 是模糊词(技能名/中文描述/标签的子串,大小写不敏感),留空列出全部
        名字。`detail=<技能名>` 给那一个的完整参数卡:每个参数的类型、单位、
        取值范围、是否必填,以及前置条件与安全级。

        每一行都标了**来源**:官方原子 / 官方组合 = 实机验证过的,优先用;
        组合(用户/agent 造)= 之前造出来的。同一个动作有多份实现时,官方那份
        通常是对的那份。

        **什么时候该来查**:在你打算把几步串起来做第二遍之前;在计划的某一步你
        找不到对应技能的时候;在你想写一个新 composite 之前。查到现成的就直接用
        它 —— 那比造一个像它的东西便宜得多,也可靠得多。
        """
        try:
            metas = list(registry.list_skills())
        except Exception as exc:  # noqa: BLE001
            return _j({"ok": False, "reason": f"注册表读不到:{exc}"})

        if detail:
            for m in metas:
                if str(m.name) == detail:
                    return _j({"ok": True, "skill": _card(m, _origin_of(registry, m.name))})
            near = [str(m.name) for m in metas
                    if detail.lower() in str(m.name).lower()][:8]
            return _j({"ok": False, "reason": f"没有叫 {detail!r} 的技能",
                       "did_you_mean": near})

        disabled = _disabled_names()
        q = (query or "").strip().lower()
        rows: list[dict] = []
        for m in metas:
            name = str(m.name)
            if name in disabled:
                continue          # 工具表里也没有它,列出来只会诱导去调
            desc = str(getattr(m, "description", "") or "")
            tags = [str(t) for t in (getattr(m, "tags", None) or ())]
            if q and q not in " ".join([name.lower(), desc.lower(), " ".join(tags).lower()]):
                continue
            origin = _origin_of(registry, name)
            rows.append({
                "name": name,
                "origin": _ORIGIN_ZH.get(origin, origin),
                "official": origin in _OFFICIAL_ORIGINS,
                "safety": _sl(m),
                "params": [str(p.name) for p in (getattr(m, "parameters", None) or ())],
                "description": desc[:160],
            })
        rows.sort(key=lambda r: (not r["official"], r["name"]))
        return _j({
            "ok": True, "count": len(rows),
            "skills": rows[:400],
            "note": ("超过 400 条已截断,用 query 缩小范围" if len(rows) > 400 else ""),
            "reminder": "官方(官方原子/官方组合)覆盖得了的就别新造。",
        })

    tools.append(skill_catalog)

    # ── 2. 起草 ──────────────────────────────────────────────────────
    @tool("draft_composite")
    def draft_composite(spec_json: str) -> str:
        """校验一份组合技能草稿(**不落盘、不注册**),问题整批返回。

        **第一次用、或忘了 spec 怎么写:传 `spec_json="?"`**,它会返回完整格式说明
        (节点类型、表达式写法、会被拒绝的几种)。校验失败的回执里也带一份速查。

        **什么时候值得造一个**:同一串步骤这次任务里要重复 ≥2 次;计划的某一步没有
        任何单个技能对应;需要 try/finally 的鲁棒包装(「无论成败最后一定退针」)。
        只是给一个官方技能换个名字 —— 不值得,而且会被拒。

        返回 `{ok, problems, steps, hints}`。`problems` 一次给全,照着改完再调一次;
        `hints` 是建议不是错误(比如「已经有个现成的在做这件事」)。
        """
        if (spec_json or "").strip() in ("", "?", "？", "help"):
            return _j({"ok": False, "syntax": SPEC_SYNTAX,
                       "problems": ["(没给 spec —— 上面是格式说明)"]})
        try:
            d = json.loads(spec_json) if isinstance(spec_json, str) else spec_json
        except Exception as exc:  # noqa: BLE001
            return _j({"ok": False, "syntax": SPEC_SYNTAX,
                       "problems": [f"spec_json 不是合法 JSON:{exc}"]})
        if not isinstance(d, dict):
            return _j({"ok": False, "syntax": SPEC_SYNTAX,
                       "problems": ["spec_json 要是一个 JSON 对象"]})

        rep = _validate(d, registry, store=store)
        rep["hints"] = official_overlap_hints(d, registry, store=store)
        if rep["ok"]:
            rep["next"] = (f"校验通过。save_composite 保存并热注册,然后本轮用 "
                           f"run_composite({d.get('name')!r}, ...) 执行。")
        else:
            # 语法速查跟着**失败**走,不跟着 schema 走:放进 docstring 的话每一次
            # 模型调用都要付它的钱(核心包工具的 schema 每轮都在上下文里),而它
            # 只在真的要造技能、而且草稿还没写对的时候有用。
            rep["syntax"] = SPEC_SYNTAX
        return _j(rep)

    tools.append(draft_composite)

    # ── 3. 保存 ──────────────────────────────────────────────────────
    @tool("save_composite")
    def save_composite(spec_json: str, base_version: int = -1) -> str:
        """保存组合技能并**热注册**(立即可执行),写进版本库带完整历史。

        先用 draft_composite 改到 ok=true 再调这个。保存成功后:
        - 它成为一个真正的技能,和官方技能一样经过全部安全闸;
        - **本轮的工具表不会刷新**（那是建图时冻结的）——本轮用 `run_composite`
          调它;
        - **下一轮它会不会自己出现在工具表里,取决于这台机器的订阅列表**
          (2026-08-27 起): 用户没定制过订阅(出厂态)⇒ 会自己出现;定制过 ⇒
          它只进「技能→市场」等他点一下头。**两种情况下 `run_composite` 都能调它**
          ——订阅管的是工具表,不是能不能执行。所以别把「它没出现在工具表里」读成
          「保存失败了」;要让用户用得上它,在回答里请他去市场加一下。
        - 用户会在诊断台账看到一行。**你也要在回答里说一句你造了什么、为什么**。

        改自己之前存过的同名技能:传 `base_version` = 你上次拿到的版本号（乐观锁,
        防止覆盖掉别人同时做的修改）。用户做的技能不能改 —— 换个名字另存。
        """
        return _j(save_composite_impl(registry, who, spec_json, base_version,
                                      store=store))

    tools.append(save_composite)

    # ── 4. 执行 ──────────────────────────────────────────────────────
    if context_provider is not None:
        @tool("run_composite")
        def run_composite(
            name: str,
            params_json: str = "{}",
            tool_call_id: Annotated[str, InjectedToolCallId] = "",
        ) -> Any:
            """按名字执行一个组合技能 —— 给刚 save 出来、工具表里还没有的那个用。

            工具表是建图时冻结的,所以刚保存的技能**本轮**不会出现在你的工具列表
            里。这个工具是那座桥。**工具表里已经有的技能不要走这里** —— 直接调它
            本体,那样参数说明、单位提示和范围检查都在。

            `params_json` 是参数的 JSON 对象。执行走的是和普通技能完全一样的那条
            路:abort、样品闸、五条硬闸、参数包络、SAFE/SEMI 模式闸,一个都不少。
            """
            try:
                params = json.loads(params_json) if isinstance(params_json, str) else (params_json or {})
            except Exception as exc:  # noqa: BLE001
                return f"params_json 不是合法 JSON:{exc}"
            if not isinstance(params, dict):
                return "params_json 要是一个 JSON 对象"

            try:
                cls = registry.get(name)
            except Exception:
                return (f"注册表里没有 {name!r}。用 skill_catalog 查现有技能;"
                        "如果你刚 save 过,看那次调用是不是真的 ok=true。")

            # 只放行组合技能。普通技能已经在工具表里有自己的入口,按名放开只会
            # 多一条绕过 args_schema 描述的路。
            try:
                from mast.skills.composite._base import CompositeSkillGraph
                if not (isinstance(cls, type) and issubclass(cls, CompositeSkillGraph)):
                    return (f"{name!r} 不是组合技能。它在你的工具表里有自己的入口,"
                            "直接调它 —— 那里有完整的参数说明和范围提示。")
            except Exception as exc:  # noqa: BLE001
                return f"无法判定 {name!r} 的类型({exc})——拒绝执行"

            # 关闭名单:名字本身,以及它**引用到的**每一个技能。
            disabled = _disabled_names()
            if name in disabled:
                return f"{name!r} 在本机是关闭的(硬件不存在或该能力未授予)。"
            used = _spec_step_skills_of(cls)
            blocked = sorted(used & disabled)
            if blocked:
                _diag("safety_block", name,
                      f"组合技能引用了关闭的技能:{'、'.join(blocked)}")
                return (f"{name!r} 里用到了本机关闭的技能:{'、'.join(blocked)} —— "
                        "拒绝执行。关闭是用户的决定,包一层不会打开它。")

            meta = _meta_of(registry, cls)
            if meta is None:
                # 「读不到元数据」**不是**「没有限制」。下面的模式闸和 DANGEROUS
                # 判据全都读它,meta=None 时两者都会静静地放行 —— 那正是本仓
                # 一天之内记过四次的形状(读不到被折叠成一个具体的答案)。
                # 实践中 wrap_skill 也读 metadata,所以这条基本不可达;它存在
                # 是为了让**不可达**这件事写下来,而不是靠运气。
                _diag("safety_block", name, "组合技能的元数据读不到 —— 拒绝执行")
                return (f"读不到 {name!r} 的元数据,拒绝执行 —— 安全闸门要靠它判,"
                        "读不到就没法判。这是环境/注册表问题,不是参数问题。")

            # SAFE / SEMI 模式闸。这一层**只**活在 safety_mw 里,而那道中间件对
            # 没有 skill_metadata 的工具直接放行 —— 也就是说这条路会从它旁边绕
            # 过去,而 ExecutionContext.run 里没有模式闸。判据取的是同一批单源
            # 函数,不另写一份。
            refusal = _mode_refusal(name, meta, params)
            if refusal:
                _diag("mode_block", name, refusal)
                return refusal

            # composite 顶层的 DANGEROUS 通知。子步那一层由 ExecutionContext
            # 自己发(⑰),但顶层这一次本来是 AutoApprovalNoticeMiddleware 发的,
            # 而它同样看 skill_metadata —— 不补就会少一行台账。
            _notice(name, meta, params)

            try:
                from mast.agents._shared.skill_adapter import wrap_skill
                forged = wrap_skill(cls, context_provider,
                                    post_hook=post_hook, recorder=recorder)
            except Exception as exc:  # noqa: BLE001
                logger.warning("技能工坊:包装 %s 失败:%s", name, exc)
                return f"无法执行 {name!r}:{exc}"

            return _dispatch(forged, params, tool_call_id, name)

        tools.append(run_composite)
    else:
        logger.debug("技能工坊:没有 context_provider —— 不产出 run_composite")

    # ── 5. 提议新原子能力(人审轨) ───────────────────────────────────
    @tool("propose_python_skill")
    def propose_python_skill(name: str, code: str, rationale: str) -> str:
        """提议一个**新的原子技能**(Python 代码草稿)——组合表达不了的时候才用。

        用它之前先确认:官方技能没有、把官方技能组合起来也做不到(比如需要一个
        现在没人封装的 Nanonis TCP 调用)。

        **它不会立即生效。** 代码只写到磁盘上等用户审阅;要能用,他必须看过
        代码并把名字加进白名单。所以提完之后如实报告,然后**继续用现有手段完成
        当前任务** —— 不要等它,也不要假设它已经能用了。

        `code` 写一个 BaseSkill 子类(实现 metadata() 和 execute(context, params));
        `rationale` 写清楚为什么现有技能不够 —— 那是用户审阅时唯一的判据。
        """
        return _j(propose_python_skill_impl(registry, who, name, code, rationale))

    tools.append(propose_python_skill)
    return tools


# ─────────────────────────────────────────────────────────────────────────
# 保存 / 提议的实现本体 —— agent 工具与外部 agent 网关共用同一份裁决
# ─────────────────────────────────────────────────────────────────────────

#: 「自动化作者」的署名前缀。覆盖一份已存在的组合技能，只许在它也是自动化作者
#: 造的时候（进程内 agent 署 ``agent:<名>``，外部 agent 经网关署 ``ext:<名>``）；
#: 用户在 builder 页做的那份永远不改 —— 他手上那份是他调过的。
_AUTOMATED_AUTHOR_PREFIXES: tuple[str, ...] = ("agent:", "ext:")


def save_composite_impl(registry, who: str, spec: Any, base_version: Any = -1, *,
                        store=None, origin: str = "agent_forge") -> dict:
    """保存组合技能并热注册。返回回执 dict（工具把它序列化给模型）。

    ``who`` 是署名（``agent:<名>`` / ``ext:<名>``），``spec`` 可以是 JSON 字符串或
    dict。判据的顺序本身有讲究：撞名预检先于保存；已存在时先判作者、再判 CAS、
    最后才判「内容相同」（反过来的话，一个拿着陈旧 base_version 的调用只要内容碰巧
    一样就会收到 ok=True）。
    """
    try:
        d = json.loads(spec) if isinstance(spec, str) else spec
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": "bad_json", "problems": [str(exc)]}
    if not isinstance(d, dict) or not d.get("name"):
        return {"ok": False, "error": "bad_spec",
                "problems": ["spec 要是一个 JSON 对象,而且必须有 name"]}
    name = str(d["name"])

    st = store if store is not None else _default_store()
    if st is None:
        return {"ok": False, "error": "no_store",
                "problems": ["组合技能版本库打不开 —— 这是环境问题,不是 spec 问题"]}

    # 撞名预检**先于**保存。见 _name_collision_problems 的注释。
    rep = _validate(d, registry, store=st)
    if not rep["ok"]:
        rep["error"] = "invalid_spec"
        rep["hints"] = official_overlap_hints(d, registry, store=st)
        return rep

    # 已存在:只许改自动化作者造的,而且要带 CAS。
    prior_meta = {}
    try:
        prior_meta = st.load_meta(name) or {}
    except Exception:  # noqa: BLE001
        prior_meta = {}
    if st.exists(name):
        author = str(prior_meta.get("_author") or "")
        if not author.startswith(_AUTOMATED_AUTHOR_PREFIXES):
            return {
                "ok": False, "error": "not_yours",
                "problems": [
                    f"{name!r} 是用户(或 builder 页)做的,agent 不改它 —— "
                    "换个名字另存一份。他手上那份是他调过的。"],
            }
        stored = _stored_version(st, name)
        if base_version is None or int(base_version) < 0:
            return {
                "ok": False, "error": "base_version_required",
                "problems": [
                    f"{name!r} 已存在,要覆盖必须传 base_version(你上次拿到的"
                    "版本号)。不传等于「不管别人改了什么都盖掉」。"],
                "stored_version": stored,
            }
        # **CAS 先判,再判内容相同。** 反过来的话,一个拿着陈旧 base_version 的
        # 调用只要内容碰巧一样就会收到 ok=True —— 「你是最新的」这句话会在它
        # 明明不是最新的时候被说出口,而这正是 CAS 存在的理由。
        if stored is not None and int(base_version) != int(stored):
            return {
                "ok": False, "error": "version_conflict",
                "problems": [
                    f"{name!r} 在你上次读到之后变过了(库里是 v{stored},"
                    f"你的 base 是 v{base_version})——先看现在那一版,"
                    "把你的修改重新叠上去,或者换个名字另存。"],
                "stored_version": stored,
            }
        # 内容一模一样就别再存一版:每次 save 都会落一份不可变历史快照,
        # 一个反复微调的循环能把版本号刷到三位数,而那些版本之间没有差别。
        if _same_content(st, d, prior_meta):
            return {
                "ok": True, "name": name, "version": stored,
                "unchanged": True,
                "message": "内容与已存的这一版逐字节相同,没有新建版本。可以直接 run_composite。",
            }

    try:
        from mast.skills.composite.spec import CompositeSpec
        from mast.skills.composite.version_store import (
            VersionConflictError,
            VersionStoreError,
        )
        spec_obj = CompositeSpec.from_dict(d)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": "invalid_spec", "problems": [str(exc)]}

    extra = _extra_meta(who, d, registry, origin=origin)
    try:
        saved = st.save(spec_obj,
                        base_version=(int(base_version) if base_version is not None
                                      and int(base_version) >= 0 else None),
                        extra_meta=extra)
    except VersionConflictError as exc:
        return {"ok": False, "error": "version_conflict",
                "problems": [str(exc)],
                "stored_version": _stored_version(st, name)}
    except VersionStoreError as exc:
        return {"ok": False, "error": "invalid_spec", "problems": [str(exc)]}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": "save_failed", "problems": [str(exc)]}

    # 注册**必须**经 register_spec:撞名拒绝与缺失技能拒绝长在那里,
    # 而且它会把 registry 传给 make_spec_skill —— 安全级/能力标签的继承
    # 依赖那个参数,少传就等于让声明说了算。
    registered, reg_err = False, ""
    try:
        from mast.skills.composite.loader import register_spec
        register_spec(registry, saved)
        registered = True
    except Exception as exc:  # noqa: BLE001
        reg_err = str(exc)
        logger.warning("技能工坊:%s 存下来了但注册失败:%s", name, exc)

    refreshed = _refresh(f"agent 铸造技能 {name}")
    _diag("skill_forged", name,
          f"{who} 保存了组合技能 {name} v{saved.version}",
          version=saved.version, registered=registered,
          steps=sorted(_step_skills(saved.nodes)),
          safety_level=_effective_level(registry, name))

    return {
        "ok": True, "name": name, "version": saved.version,
        "hot_registered": registered,
        "registration_error": reg_err,
        "effective_safety_level": _effective_level(registry, name),
        "refresh": refreshed,
        "message": (
            f"{name} v{saved.version} 已保存并热注册。"
            "**本轮的工具表不会刷新**(建图时冻结),本轮用 "
            f"run_composite({name!r}, ...) 执行它;下一轮它会自己出现。"
            " 在你的回答里向用户说明你造了什么、为什么造。"),
    }


def propose_python_skill_impl(registry, who: str, name: str, code: str,
                              rationale: str) -> dict:
    """把一个新原子技能的 Python 草稿落盘待人审。**不注册、不 exec。**

    真正的门是 ``config/custom_skills/enabled.json`` 白名单那一步人审；这里的
    AST deny-list 是纵深防御不是沙箱，与 ``custom_loader`` 用的是同一个检查器。
    """
    code = code or ""
    try:
        from mast.llm.skill_author import (
            _CUSTOM_SKILLS_DIR,
            SkillAuthor,
            _validate_skill_name,
        )
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": "unavailable",
                "problems": [f"代码提议轨不可用:{exc}"]}

    try:
        _validate_skill_name(name)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": "bad_name", "problems": [str(exc)]}

    try:
        if registry.has(name):
            return {"ok": False, "error": "name_taken",
                    "problems": [f"{name!r} 已经是一个已注册的技能 —— "
                                 "先用 skill_catalog 看它是不是已经做了你要的事"]}
    except Exception:  # noqa: BLE001
        pass

    # AST deny-list —— 与 custom_loader 用的是同一个检查器(同样的绕法)。
    # 它是纵深防御不是沙箱;真正的门是 enabled.json 白名单那一步人审。
    try:
        violations = SkillAuthor._ast_safety_check(
            SkillAuthor.__new__(SkillAuthor), code)
    except Exception as exc:  # noqa: BLE001 — 检查器不可用时保守拒绝
        violations = [f"AST 检查器不可用:{exc}"]
    struct = _baseskill_shape_problems(code)
    if violations or struct:
        return {"ok": False, "error": "rejected",
                "problems": list(violations) + struct}

    try:
        d = _CUSTOM_SKILLS_DIR
        d.mkdir(parents=True, exist_ok=True)
        target = d / f"{name}.py"
        if target.resolve().parent != d.resolve():
            return {"ok": False, "error": "bad_path",
                    "problems": ["拒绝写到自定义技能目录之外"]}
        if target.exists():
            return {"ok": False, "error": "exists",
                    "problems": [f"{target.name} 已经在等审阅了 —— "
                                 "换个名字,或者让用户先处理那一份"]}
        header = (
            f'"""[agent 提议 · 未启用] {name}\n\n'
            f'提议者:{who}\n'
            f'理由:{(rationale or "(未填)").strip()}\n\n'
            f'这份代码是 agent 写的,**尚未经过任何人审阅**,也没有被加载。\n'
            f'要启用:审阅代码 → 把 "{name}" 加进 config/custom_skills/enabled.json\n'
            f'的 enabled 列表 → 重启。在此之前它只是一个文件。\n'
            f'"""\n\n'
        )
        target.write_text(header + code, encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": "write_failed", "problems": [str(exc)]}

    _diag("note", name,
          f"{who} 提议了一个新的原子技能(待人审,未启用):{(rationale or '')[:160]}",
          path=str(target))
    return {
        "ok": True, "name": name, "path": str(target), "enabled": False,
        "message": (
            "草稿已落盘,**没有启用、没有注册、没有执行**。用户审阅并把它加进 "
            "enabled.json 白名单、重启之后才可用。"
            "现在把这件事告诉用户(你提议了什么、为什么现有技能不够),"
            "然后用现有手段继续当前任务 —— 不要等它。"),
    }


# ─────────────────────────────────────────────────────────────────────────
# 执行侧的小零件
# ─────────────────────────────────────────────────────────────────────────

def _dispatch(forged, params: dict, tool_call_id: str, name: str):
    """把参数喂给包装好的技能工具。

    走 ``.invoke()`` 而不是 ``.func()``:``_schema_from_metadata`` 把参数包络写成
    pydantic 的 ``ge`` / ``le``,那是**真约束**(``admin/reload_wiring`` 的注释里
    专门写了这一点)。直调 ``.func`` 会从它旁边绕过去 —— 那正是这个模块最不该做
    的事。``handle_validation_error`` 也挂在工具上,超界的报错会被渲染成
    SafetyGate 那种人话而不是 pydantic 样板。
    """
    # 必须传**完整的 ToolCall 字典**:工具的 schema 里有 InjectedToolCallId,
    # LangChain 对这种工具拒绝「一个裸参数字典」的调用形式 —— 这也正是 ToolNode
    # 自己走的那条路。
    call = {"name": getattr(forged, "name", name), "args": dict(params),
            "id": tool_call_id or f"forge-{name}", "type": "tool_call"}
    try:
        return forged.invoke(call)
    except Exception as exc:  # noqa: BLE001
        # 这里**不做**「回退到 .func」——那会把刚说过是真约束的那一层悄悄绕开,
        # 而且失败会伪装成成功。如实报告。
        logger.warning("技能工坊:执行 %s 失败:%s", name, exc)
        return (f"[{name}] precondition_failed: 执行没能开始 —— {type(exc).__name__}: {exc}。"
                "先用 skill_catalog(detail=…) 核对参数名与取值范围,"
                "**不要**原样重发同一组参数。")


def _spec_step_skills_of(cls) -> set[str]:
    """一个组合技能类引用到的全部子技能名(只有声明式 spec 读得出来)。

    spec 挂在**实例**上(``make_spec_skill`` 的 ``_Bound.__init__`` 闭包进去的),
    类上没有,所以这里要现场实例化 —— SpecComposite 的构造是纯赋值,没有副作用。

    手写 composite 的步骤序列是 Python 代码,这里读不出 —— 返回空集,于是关闭
    名单那道检查对它不生效。这不是漏洞:手写 composite 是官方代码,它调什么在
    ``ExecutionContext.run`` 的五条硬闸和包络下照样受约束;而「洗白关闭名单」
    这个攻击面是 agent 现造 spec 才有的。
    """
    try:
        from mast.skills.composite.interpreter import SpecComposite
        if not (isinstance(cls, type) and issubclass(cls, SpecComposite)):
            return set()
        nodes = getattr(cls()._spec, "nodes", None)
    except Exception as exc:  # noqa: BLE001
        logger.debug("技能工坊:读不出 %s 的步骤集(%s)", cls, exc)
        return set()
    return _step_skills(nodes) if nodes else set()


def _meta_of(registry, cls):
    try:
        return registry._get_metadata(cls)
    except Exception:  # noqa: BLE001
        try:
            inst = cls()
            m = inst.metadata
            return m() if callable(m) else m
        except Exception:  # noqa: BLE001
            return None


def _mode_refusal(name: str, meta, params: dict) -> str:
    """SAFE / SEMI 下该不该拒。返回拒绝语,或空串。

    **判据不在这里** —— 2026-08-27 起统一在 :func:`mast.core.safety.mode_refusal`。
    在那之前这里是同一个判断的第二份实现(同一批谓词、另一套文案),而
    ``ExecutionContext.run`` 里一份都没有;三处各判各的,加一个新能力标签就只有
    改到的那一处会认。这里只剩「读模式 + 加前缀」。

    能力标签取自元数据:声明式 composite 的那份是 ``_inherited_capabilities``
    从子步并集来的,所以一个含 TipPulse 的 spec 自己带着 ``bias_pulse`` 标签,
    不需要在这里再走一遍树。
    """
    if meta is None:
        return ""
    try:
        from mast.core.operating_mode import current_operating_mode
        from mast.core.safety import mode_refusal
    except Exception as exc:  # noqa: BLE001
        logger.warning("技能工坊:模式闸判据不可用(%s)——放行", exc)
        return ""
    try:
        mode = current_operating_mode()
    except Exception:  # noqa: BLE001
        mode = None
    refusal = mode_refusal(name, meta, params or {}, mode)
    return f"[{name}] {refusal}" if refusal else ""


def _notice(name: str, meta, params: dict) -> None:
    """顶层 DANGEROUS 的那一行台账。判据单源,永不抛。"""
    try:
        from mast.core.auto_approval import notify, would_have_asked
        why = would_have_asked(meta, tool_name=name, args=params or {})
        if why:
            notify(name, f"{why} —— 组合技能经 run_composite 直接执行并通知,"
                         "不再等待人工批准")
    except Exception as exc:  # noqa: BLE001
        logger.debug("技能工坊:通知发不出(%s)", exc)


def _refresh(reason: str) -> str:
    """技能集合变了之后把三条链推一遍。返回一句可读的结果(**不假装成功**)。"""
    try:
        from mast.admin.reload_wiring import refresh_after_skill_change
        out = refresh_after_skill_change(reason)
        describe = getattr(out, "describe", None)
        return describe() if callable(describe) else str(out)
    except Exception as exc:  # noqa: BLE001
        logger.debug("技能工坊:刷新链不可用(%s)", exc)
        return f"刷新链不可用({exc})——技能已注册,但 UI 目录可能要等下次重启"


def _diag(kind: str, subject: str, reason: str, **fields) -> None:
    try:
        from mast.core.diagnostics import record
        record(kind, subject, reason, **fields)  # type: ignore[arg-type]
    except Exception as exc:  # noqa: BLE001
        logger.debug("技能工坊:诊断写不进去(%s)", exc)


# ─────────────────────────────────────────────────────────────────────────
# 保存侧的小零件
# ─────────────────────────────────────────────────────────────────────────

def _extra_meta(who: str, spec_dict: dict, registry, *,
                origin: str = "agent_forge") -> dict:
    """落盘的 ``_`` 前缀元数据。

    ``_author`` 是后续「这份是不是自动化作者造的」的唯一判据 —— 覆盖别人的作品要靠
    它拦。``_origin`` 记下是哪条路造的（进程内 agent 的技能工坊 / 外部 agent 网关）。
    """
    extra: dict = {}
    try:
        from mast.webui.builder_api import resolved_skills_snapshot, save_extra_meta
        extra = dict(save_extra_meta() or {})
        extra["_resolved_skills"] = resolved_skills_snapshot(spec_dict)
    except Exception as exc:  # noqa: BLE001
        logger.debug("技能工坊:sync-ready 记录不可用(%s)", exc)
    extra["_author"] = who
    extra["_origin"] = origin
    return extra


def _stored_version(store, name: str):
    try:
        return store.load(name).version
    except Exception:  # noqa: BLE001
        return None


def _same_content(store, spec_dict: dict, prior_meta: dict) -> bool:
    """新草稿与已存的那一版内容是否逐字节相同(版本号不算)。

    指纹用 ``version_store`` 自己写的 ``_content_sha256``,算法必须一致 ——
    它刻意把 version 排除在外,所以这里也要排除。
    """
    prior = str(prior_meta.get("_content_sha256") or "")
    if not prior:
        return False
    try:
        import hashlib

        from mast.skills.composite.spec import CompositeSpec
        h = CompositeSpec.from_dict(spec_dict).to_dict()
        h.pop("version", None)
        canon = json.dumps(h, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(canon.encode("utf-8")).hexdigest() == prior
    except Exception:  # noqa: BLE001
        return False


def _effective_level(registry, name: str) -> str:
    """注册之后**实际生效**的安全级 —— 不是 spec 里声明的那个。

    继承规则(``interpreter._inherited_safety_level``)会把它抬到子步的最高级,
    所以回给模型的必须是查出来的那个:声明 auto 而实际 dangerous 时,回声明值
    等于告诉它「你成功地把它降级了」。
    """
    try:
        return _sl(registry._get_metadata(registry.get(name)))
    except Exception:  # noqa: BLE001
        return "unknown"


def _sl(meta) -> str:
    v = getattr(meta, "safety_level", None)
    return str(getattr(v, "value", v) or "").lower() or "unknown"


def _card(meta, origin: str) -> dict:
    params = []
    for p in (getattr(meta, "parameters", None) or ()):
        params.append({
            "name": str(p.name),
            "type": str(getattr(p, "type", "")),
            "unit": getattr(p, "unit", None),
            "required": bool(getattr(p, "required", False)),
            "default": getattr(p, "default", None),
            "min": getattr(p, "min_value", None),
            "max": getattr(p, "max_value", None),
            "allowed": list(getattr(p, "allowed_values", None) or ()) or None,
            "description": str(getattr(p, "description", "") or "")[:200],
        })
    return {
        "name": str(meta.name),
        "origin": _ORIGIN_ZH.get(origin, origin),
        "official": origin in _OFFICIAL_ORIGINS,
        "description": str(getattr(meta, "description", "") or ""),
        "safety_level": _sl(meta),
        "capabilities": sorted(getattr(meta, "capabilities", None) or ()),
        "preconditions": list(getattr(meta, "preconditions", None) or ()),
        "estimated_duration_s": getattr(meta, "estimated_duration_s", None),
        "parameters": params,
    }


def _baseskill_shape_problems(code: str) -> list[str]:
    """草稿至少要长得像一个技能:恰好一个 BaseSkill 子类。

    AST deny-list 只回答「有没有危险的东西」,不回答「这是不是一个技能」。一份
    语法正确但没有技能类的文件会安安静静躺在待审目录里,等用户加白名单、重启,
    然后 loader 说「没找到技能」——那时候提议它的那一轮对话早就没了。
    """
    import ast
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [f"代码语法错误:{exc}"]
    classes = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.ClassDef)
        and any((isinstance(b, ast.Name) and b.id == "BaseSkill")
                or (isinstance(b, ast.Attribute) and b.attr == "BaseSkill")
                for b in n.bases)
    ]
    if not classes:
        return ["代码里没有 BaseSkill 的子类 —— 一个技能至少要有一个"]
    if len(classes) > 1:
        return [f"代码里有 {len(classes)} 个 BaseSkill 子类,一个文件放一个"]
    body = {n.name for n in ast.walk(classes[0]) if isinstance(n, ast.FunctionDef)}
    missing = [m for m in ("metadata", "execute") if m not in body]
    if missing:
        return [f"技能类缺少必须实现的方法:{'、'.join(missing)}"]
    return []
