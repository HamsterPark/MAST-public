"""重启恢复的**判断部分** —— 纯函数,不碰仪器、不碰数据库。

设计:``campaign_director_design.md`` §8(自检清单两档 + 预授权位语义表)、
§5(RECOVERY_PENDING 各行)、§0(「重启经显式恢复自检从**上一个已通过闸门**
续跑」)。执行部分(跑复验步、写事件、转状态)在 :mod:`mast.conduct.director`。

## 为什么把这几件事挑出来做成纯函数

它们全是**判断**,而且每一个判断错了都不会当场炸:回退到哪一步、温度窗从哪来、
哪些绑定在重启之后已经不能用了 —— 三件事的错误形态都是「照样跑下去,只是跑错」。
这类判断必须能不带硬件、不带 SQLite、单独喂进去一组数就断言出来。

## 可信 / 不可信的分界(这条是这个模块的地基)

* **可信**:状态机位置、闸门历史、预算账 —— 全部在 SQLite 里,崩溃前提交的;
* **一律不可信**:针尖、坐标、面板设置、温度 —— 全部是「世界现在是什么样」,
  而进程死过一次这件事对它们**什么都不保证**。

所以本模块只读第一类,并且把第二类的每一个「上次是这样」当成**必须重新去问**
的东西。:func:`resume_plan` 因此不是「从崩溃点接着跑」,而是「退到最近一次
**被闸门祝福过**的位置」——祝福是第一类,位置也是第一类。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from mast.conduct.spec import AT_ENTRY_GATE, ConductSpec, StageSpec, StepSpec

logger = logging.getLogger(__name__)

#: 会改变表面的能力。与 ``validator.SURFACE_CHANGING_CAPABILITIES`` **同一个
#: 名字、同一个来源** —— 这里 import 它而不是抄一份:一个判据两份字面量,
#: 改了一份另一份会安静地按旧定义继续拦(或者继续放行)。
try:                                        # pragma: no cover - import 形状
    from mast.conduct.validator import SURFACE_CHANGING_CAPABILITIES
except Exception:                           # pragma: no cover
    SURFACE_CHANGING_CAPABILITIES = frozenset(
        {"tip_shaping", "bias_pulse", "layer_removal"})


# ── A2:温度的「合理窗」从哪来 ────────────────────────────────────────────

@dataclass(frozen=True)
class TempWindow:
    """A2 要拿去对账的那个上限,以及**它是谁说的**。

    ``ceiling_k is None`` 有一个确切的含义:**这份 spec 没有声明过工作温度**,
    于是「值在合理窗内」这一问**没有被检查**。它不等于「检查通过」,也不等于
    「温度不对」—— 两者在面板上是两句不同的话,而报错报成第二种会让人去查一台
    好好的制冷机。
    """

    ceiling_k: "float | None" = None
    #: 人读来源(面板/事件里印这句)。没有窗时说清为什么没有。
    source: str = ""

    @property
    def declared(self) -> bool:
        return self.ceiling_k is not None


def _param(params: Mapping[str, Any], ref: str) -> "float | None":
    """``params.<name>`` → 数值。取不到回 ``None``,**不回一个占位数**。"""
    if not ref.startswith("params."):
        return None
    raw = params.get(ref[len("params."):])
    if raw is None or isinstance(raw, bool):
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def temperature_window(spec: ConductSpec,
                       params: "Mapping[str, Any] | None" = None) -> TempWindow:
    """这份 conduct 声明过的工作温度上限。

    两个来源,按优先级:

    1. ``spec.recovery.temp_ceiling_ref`` —— 模板作者**明说**「A2 拿这个数对账」;
    2. spec 自己声明的温度等待条件(``WaitSpec.condition``,``signal ==
       "temperature_k"`` 且 ``op`` 是 ``<=`` / ``<``)。

    **第二条为什么算数**:那些条件是「换完样品之后等温度**回到工作点**」的判据,
    数由用户通过 ``value_ref`` 填。它回答的正是 A2 要问的那个问题 ——「这块
    样品现在还在声明的工作点上吗」。这不是发明一个数,是**引用**一个已经被人
    填过、而且机器已经按它等过的数。

    **什么时候拒绝作答**:声明了**不止一个不同的**上限(比如不同阶段等不同温度)
    时,「哪一个是当前阶段的工作点」这个问题这里答不了 —— 于是不答,报没检查。
    猜一个(取最小?取当前阶段的?)会让 A2 稳定地在量一个不是目标的东西。
    """
    params = dict(params or {})
    ref = str(getattr(spec.recovery, "temp_ceiling_ref", "") or "")
    if ref:
        value = _param(params, ref)
        if value is None:
            return TempWindow(None, f"模板指名 {ref},而 conduct 参数里取不到它"
                                    f"(读不到 ≠ 通过)")
        return TempWindow(value, f"模板指名的 {ref} = {value:g} K")

    seen: "dict[float, str]" = {}
    for stage in spec.stages:
        for step in stage.all_steps:
            cond = getattr(step.wait, "condition", None) if step.wait else None
            if cond is None or cond.signal != "temperature_k":
                continue
            if cond.op not in ("<=", "<"):
                continue
            value = (_param(params, cond.value_ref) if cond.value_ref
                     else float(cond.value))
            if value is None:
                # 声明了 ref 却取不到 —— 「填了但没进去」那族。不拿占位值顶。
                continue
            seen.setdefault(float(value), f"{stage.stage_id}/{step.step_id} 的"
                                          f"等待条件({cond.op} {value:g} K)")
    if not seen:
        return TempWindow(None, "这份 spec 没有声明过工作温度上限 —— "
                                "「值在合理窗内」这一问没有被检查")
    if len(seen) > 1:
        return TempWindow(None, "这份 spec 声明了不止一个工作温度上限"
                                f"({sorted(seen)} K),哪一个属于当前阶段这里答不了"
                                " —— 不猜,报没检查")
    ceiling, why = next(iter(seen.items()))
    return TempWindow(ceiling, f"由 {why} 推得")


# ── 「从上一个已通过的闸门续跑」 ──────────────────────────────────────────

@dataclass(frozen=True)
class ResumePlan:
    """自检全过之后,**从哪儿接着跑**。

    ``blocked`` 非空 = 接不上,得等人。这不是失败也不是通过 —— 是「这台机器
    算不出一个自洽的续跑点」,而算不出来就说出来,别挑一个看起来能跑的。
    """

    stage_idx: int
    step_idx: int
    #: 真的往回退了(位置变了)。
    rewound: bool = False
    #: 非空 ⇒ 转 WAITING_OPERATOR,内容是那句给用户看的话。
    blocked: str = ""
    #: 人读说明,进 ``recovery_item`` 的 payload 与 ``status_reason``。
    why: str = ""
    #: 挡路的那些东西(测试与面板按它对账,不靠解析 ``why`` 的文案)。
    blockers: tuple = ()


def _gate_position(stage: StageSpec, gate_id: str, which: str) -> "int | None":
    """一道**判过 pass** 的闸门,放行之后位置在哪一步。

    * ``entry`` → 0(入口闸放行就是进第一步);
    * ``step``  → 那一步的下标 + 1(步级闸门长在步后面);
    * ``exit``  → ``None``:出口闸放行会推进到**下一个阶段**,而本函数只回答
      「本阶段内的位置」。跨阶段回退不在这里做(见 :func:`resume_plan`)。
    """
    if which == "entry":
        return 0
    if which == "exit":
        return None
    for i, step in enumerate(stage.all_steps):
        if step.gate is not None and step.gate.gate_id == gate_id:
            return i + 1
    return None


def _blocking_step(step: StepSpec, skill_meta: "Callable[[str], Any] | None"
                   ) -> str:
    """重跑这一步会不会付一笔**机器付不起的账**。非空 = 会,内容是为什么。

    两类,理由不同:

    * **等人步** —— 那个 ack 是一个人半夜起来换了样品。重跑它 = 把同一个人
      再叫一次去做一件已经做完的事,而机器这一侧一个字都说不清为什么。
      (ack 本身在审计流里,是可信的第一类事实 —— 既然记着,就不必再问。)
    * **会改变表面的动作 / DANGEROUS** —— 脉冲、扎针、除层。重跑它是**又打一发**。
      续跑的前提是「重放这一段不改变世界」,这类步一步都不满足。

    读不到技能元数据时按**挡路**处理:「查不到这个技能危不危险」不是
    「它不危险」。
    """
    if step.kind == "wait":
        return f"{step.step_id} 是等人步(那个 ack 是人半夜起来换的样品)"
    if not step.touches_hardware:
        return ""
    if skill_meta is None:
        return (f"{step.step_id}({step.skill}):没有技能元数据可查 —— "
                f"「查不到危不危险」不是「不危险」")
    try:
        meta = skill_meta(step.skill)
    except Exception as exc:                # noqa: BLE001
        return f"{step.step_id}({step.skill}):技能元数据读失败 {exc}"
    if meta is None:
        return (f"{step.step_id}({step.skill}):注册表里查不到它 —— "
                f"「查不到危不危险」不是「不危险」")
    caps = frozenset(getattr(meta, "capabilities", frozenset()) or ())
    level = str(getattr(getattr(meta, "safety_level", None), "value", "")).lower()
    offending = caps & SURFACE_CHANGING_CAPABILITIES
    if offending:
        return f"{step.step_id}({step.skill}) 声明了 {sorted(offending)}"
    if level == "dangerous":
        return f"{step.step_id}({step.skill}) 是 DANGEROUS 级"
    return ""


def _stale_bindings(stage: StageSpec, resume_idx: int,
                    spec: ConductSpec) -> list[str]:
    """从 ``resume_idx`` 往后,哪些绑定会吃到**重启之前**的产出。

    这一问是 A4 的直接后果,而且它今天在别处**没有人问**:``_collect_evidence``
    按 ``min_epoch`` 把跨代次证据挡在闸门外,而 ``_resolve_params`` 走的是另一条
    路 —— 它只问「上游产出过没有」,不问「哪一代产出的」。于是一次重启之后,
    一串重启前算出来的**坐标**可以原样喂进下一个硬件技能,而坐标正是「一律
    不可信」那一列里的第二个。

    这里不改 ``_resolve_params``(那会同时改掉绕道那条路的行为,是另一件事),
    而是在**选续跑点**的时候就把这种点判成不自洽:能重跑出来的算数,重跑不出来
    的就说「这里接不上」。
    """
    reruns = {st.step_id for st in stage.all_steps[max(resume_idx, 0):]}
    stale: list[str] = []
    for step in stage.all_steps[max(resume_idx, 0):]:
        for name, ref in step.bindings.items():
            if not ref.startswith("steps."):
                continue
            producer = _producer_of(ref, spec)
            if producer is None:
                # 绑到一个不存在的产出 —— 校验器的规则②该拦的,这里不重判。
                continue
            if producer not in reruns:
                stale.append(f"{step.step_id}.{name}←{ref}"
                             f"(产出方 {producer} 不会重跑)")
    return stale


def _producer_of(ref: str, spec: ConductSpec) -> "str | None":
    """``steps.<step_id>.<field>`` → ``step_id``。

    step_id 自己带点(``S4.01_points``),所以**不能** ``split('.')`` —— 按已知
    的 step_id 逐个前缀匹配。同一条纪律写在 §4 补全⑤里。
    """
    body = ref[len("steps."):]
    best: "str | None" = None
    for stage in spec.stages:
        for step in stage.all_steps:
            if body.startswith(step.step_id + "."):
                if best is None or len(step.step_id) > len(best):
                    best = step.step_id
    return best


def resume_plan(spec: ConductSpec, row: Mapping[str, Any],
                gate_events: Sequence[Mapping[str, Any]] = (),
                *, skill_meta: "Callable[[str], Any] | None" = None
                ) -> ResumePlan:
    """自检全过之后从哪儿接着跑(设计 §0 / §8 A4)。

    ## 规则

    1. **目标 = 本阶段最近一次判 ``pass`` 的闸门放行之后的那个位置**;本阶段
       一道都没判过 ⇒ 本阶段的**起点**(第 0 步)。起点算数是因为
       ``entry_actions`` 按定义就是「阶段入口重申设置(idempotent)——不信任
       跨阶段仪器状态」:那正是一个可以无条件重放的检查点。
    2. **只在本阶段内回退**(§8 逐字:「该阶段最近 epoch-safe 检查点」)。
       跨阶段回退要重放一整段测量,那是人的决定,不是清算的决定。
    3. **绝不往前跳**:目标比当前位置靠后 ⇒ 就地不动。
    4. 回退跨过的那一段里若有**等人步**或**会改变表面的动作** ⇒ 退不了,
       :attr:`ResumePlan.blocked` 非空 —— 停下来问人。
    5. 定下位置之后再问一遍**绑定自洽**:从这里往后有没有哪一步会吃到重启前的
       产出。有 ⇒ 同样退不了。

    ## 为什么不是「从崩溃点接着跑」

    A4 一律 bump 证据代次,于是重启前采的一切都不再参与闸门判定。就地续跑的
    结局是**跑到下一道闸门才发现证据全是上一代的**,那时闸门报「证据缺席 →
    转人」——听起来像判定出了问题,其实是这次续跑从一开始就接不上。把这件事
    提前到清算时说清楚,用户看到的才是真正的原因。
    """
    si = int(row.get("stage_idx") or 0)
    pi = int(row.get("step_idx") if row.get("step_idx") is not None else 0)
    if si >= len(spec.stages):
        return ResumePlan(si, pi, why="所有阶段都跑完了,没有要续的东西")
    stage = spec.stages[si]
    if pi == AT_ENTRY_GATE:
        return ResumePlan(si, pi, why=f"停在 {stage.stage_id} 的入口闸上,"
                                      f"这一段还什么都没跑,没有可回退的")

    target = 0
    target_why = f"{stage.stage_id} 还没有闸门判过 pass ⇒ 退到本阶段起点" \
                 f"(入口动作按定义可重放)"
    for ev in gate_events:
        payload = dict(ev.get("payload") or {})
        if str(payload.get("verdict") or "") != "pass":
            continue
        if str(ev.get("stage_id") or "") != stage.stage_id:
            continue
        pos = _gate_position(stage, str(payload.get("gate_id") or ""),
                             str(payload.get("which") or ""))
        if pos is None:
            continue
        target = pos
        target_why = (f"退到闸门 {payload.get('gate_id')} 放行之后的位置"
                      f"(本阶段最近一次 pass)")

    # 规则③:绝不往前跳。目标不比当前位置靠前 ⇒ 就地不动,但**照样要问第 5 条**
    # —— 「不用回退」不等于「接得上」:停在第 2 步、闸门刚好在第 2 步前放行过,
    # 而第 2 步绑着第 0 步的产出,那份产出照样是重启之前的。这条 return 早退过
    # 一版,它把「没得退」静默地当成了「没问题」。
    resume_idx = target if target < pi else pi
    rewound = target < pi
    if not rewound:
        target_why = f"最近一次闸门放行就在当前位置(第 {pi} 步),不用回退"
    else:
        span = stage.all_steps[target:pi]
        blockers = [b for b in (_blocking_step(st, skill_meta) for st in span) if b]
        if blockers:
            return ResumePlan(
                si, pi, blocked=(
                    f"要接着跑得先退回 {stage.stage_id} 的第 {target} 步"
                    f"({target_why}),而这中间有重放不得的动作:"
                    f"{'; '.join(blockers)}。**不自动重放,也不就地续跑** —— "
                    f"就地续跑会让下一道闸门拿到上一代证据。请人决定"),
                why=target_why, blockers=tuple(blockers))

    stale = _stale_bindings(stage, resume_idx, spec)
    if stale:
        return ResumePlan(
            si, pi, blocked=(
                f"从 {stage.stage_id} 第 {resume_idx} 步续跑,仍有绑定吃的是重启"
                f"之前的产出:{'; '.join(stale)} —— 重启一律作废旧代次证据,"
                f"而这些数(坐标最典型)重跑不出来。请人决定"),
            why=target_why, blockers=tuple(stale))

    if not rewound:
        return ResumePlan(si, pi, why=target_why)
    return ResumePlan(si, target, rewound=True,
                      why=f"{target_why};重放第 {target}..{pi - 1} 步"
                          f"(这一段没有等人步,也没有会改变表面的动作)")


__all__ = ["TempWindow", "temperature_window", "ResumePlan", "resume_plan",
           "SURFACE_CHANGING_CAPABILITIES"]
