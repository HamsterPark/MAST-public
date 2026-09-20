"""谱学采集的**条件表** —— 一条谱的全部数字住在这里,按**组名**取用。

设计文档:``docs/v2/design/`` 的「S4 STS」设计,D18/D19/§3.1(文件名按「S4 STS」
检索;通用层注释不写样品名前缀 —— ``stm_capability_vs_sample_layer.md`` 拍板④)。

形状照 :mod:`mast.core.special_tip_workflow`:frozen dataclass 出厂基线 +
``_BOUNDS`` 越界**拒绝不夹紧** + ``resolve_*(overrides)``。

## 为什么要有这张表(D19)

一条谱的条件是七八个数:稳定偏压、稳定设定点、扫描窗口两端、点数、调制幅度与
频率、整定时间。把它们做成技能的顶层参数,等于**在工具表里摆七八个空格子请人
填数** —— 而那个「人」经常是语言模型。本仓已经记过四次:**移除诱因,别在提示词
里说服模型**。所以:

* 技能表面只留一个 ``condition`` **组名**(以及一个可覆写的整定时间),
* 数字全部来自这张表或 conduct spec,
* **模型能做的只有「选哪张表」,不能填数** —— 与 ``ApplyLockInPreset``「按组名
  下发、零数值参数」是同一条纪律。

## 出厂表里为什么几乎是空的

四个字段没有出厂值,而且**刻意没有**:

    stab_bias_v / stab_setpoint_a / start_v / end_v

它们是**样品事实**,不是仪器事实。稳定偏压该取多少取决于这块样品的能隙、表面
态、针尖状态;编一组数出来,流程会照跑不误、每条谱都「成功」,而它们测的是一个
没人给过的假设。同一条纪律在 ``vision/scan_prep_thresholds`` 里写作「没有那个
样品的数据就造一个 profile 出来等于伪造标定」,在 ``vision/domain_reference``
里写作「没有参照系就永远不出 label」。

⇒ 出厂组 :data:`CONDITIONS` 里 ``"default"`` 的这四个字段是 ``None``,
:attr:`STSConditionSpec.calibrated` 因此是 False,消费方**必须拒绝下发**并说出
缺哪几个。这不是「还没做完」,这是「这个问题现在没有答案」。

其余字段(点数、调制频率、整定时间、Z offset)有出厂值,因为它们是**仪器与判据**
这一侧的事实:400 点与 ``ConfigureSTS`` 的 2..10000 同界,713 Hz 是本仓知识库里
的调制频率,1 s 整定是一个保守值。它们照样可以被覆写,照样过边界闸。

## 越界是**拒绝**,不是丢弃

``special_tip_workflow._resolve_into`` 对越界值的处置是**丢弃并写一条日志**。
那里成立:覆写来自流程作者写在代码里的一个 dict,越界是作者的笔误,退回出厂值是
安全的一边。

**这里不成立。** 这里的覆写来自一次**活的调用**(conduct spec / 用户 / 模型),
丢弃意味着调用方以为自己设的是 X、实际跑的是表里的 Y,而结果里没有任何字段会说
出这件事 —— 那正是本仓记过的「兜底值合理得让人看不出兜底发生了」。所以
:func:`resolve_condition` 把越界项**如实报出来**(``rejections``),由消费方拒绝
整次运行。夹紧更糟:夹紧连「你设过一个越界值」都不告诉任何人。

## 调制:数字来自表,但走的是显式传参那条路(D18)

``ApplyLockInPreset`` 按组名下发且**写后回读比对**,是更安全的机制 —— 但它给不了
「逐条谱不同幅度」,而条件序列整件事就是在不同调制幅度下各取一条(±1 V 综览要
10-20 mV,±100 mV 精细谱要 1-2 mV)。所以这里走 ``ConfigureLockIn`` 的**显式**
``amplitude_v`` / ``frequency_hz``。

⚠️ ``ConfigureLockIn`` 对**省略**的幅度/频率报 0.0,而且**不做写后回读**。
本表永远把两个数都显式给出(:meth:`lockin_config` 不会漏键),消费方应当在写完之后
自己调一次 ``GetLockInConfig`` 回读比对 —— 别把 修复项 当已修。

## MLS 是透传,不是第二套分段抽象

Nanonis 的 MLS 已经覆盖「一条曲线内多偏压段」(逐段起止/点数/整定/积分/lock-in
开关)。本表的 :attr:`STSConditionSpec.mls_segments` 只是把那七个数组**原样带着**
交给既有的 ``SetSTSMLSVals``,S4 不建第二套分段模型。MLS 不覆盖的是「多条独立谱、
各自不同的稳定条件」—— 那正是这张表存在的理由。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, fields, replace
from typing import Any

logger = logging.getLogger(__name__)


# ── 一条谱的条件 ──────────────────────────────────────────────────────────────

#: 四个**没有出厂值**的字段。它们是样品事实,代码不发明(见模块注释)。
#: 缺任何一个 ⇒ :attr:`STSConditionSpec.calibrated` 为 False。
SAMPLE_FACT_FIELDS: tuple[str, ...] = (
    "stab_bias_v", "stab_setpoint_a", "start_v", "end_v")


@dataclass(frozen=True)
class STSConditionSpec:
    """一条谱的采集条件。单位写在字段名里(``_v`` / ``_a`` / ``_hz`` / ``_s``)。

    单位写进名字是本仓流程表的纪律:一个裸的「0.5」在伏特和纳安上差九个数量级,
    而两边都是**看上去合理**的数。
    """

    #: 扫之前把针尖稳在这里。``None`` = **未标定**,不是 0 —— 0 V 是一个真实答案
    #: (恒流反馈下还是一个危险的答案:|V| 越小针尖被推得越近)。
    stab_bias_v: "float | None" = None
    stab_setpoint_a: "float | None" = None
    #: 扫描窗口两端。两个都给才算给 —— 只给一半的窗口有一半来自仪器上一次留下的
    #: 状态,而结果里没有任何字段会说出这件事。
    start_v: "float | None" = None
    end_v: "float | None" = None
    #: 每条谱的点数。与 ``ConfigureSTS`` 的 2..10000 同界。
    num_points: int = 400
    #: 这条谱要不要 dI/dV。
    mod_on: bool = True
    #: 调制幅度。**rms 还是峰值待真机确认** —— 与
    #: ``special_tip_workflow.mod_amp_v`` 同一个未决项,两处都要在同一次验收里答。
    mod_amp_v: float = 0.005
    mod_freq_hz: float = 713.0
    #: 到位后、开扫前的整定时间。
    settle_s: float = 1.0
    #: 谱学 Z offset,传给 ``ConfigureSTS``。
    #: ⚠️ 它**总是**会被写:``ConfigureSTS`` 省略时按 0 处理,所以「配了窗口」
    #: 顺带会清掉用户可能设过的保护性预退。0.0 是显式的 0,不是「没给」。
    z_offset_m: float = 0.0
    #: 非 ``None`` 时这一条走 MLS,内容原样透传给 ``SetSTSMLSVals``(七个等长数组)。
    #: 是 tuple 不是 list —— frozen dataclass 里放可变默认值是另一类事故。
    mls_segments: "tuple[dict, ...] | None" = None
    #: 进 .dat basename 与地图标签。
    label: str = ""

    # ── 完整性 ────────────────────────────────────────────────────────

    @property
    def missing_fields(self) -> tuple[str, ...]:
        """缺哪几个样品事实。**空 tuple 才能下发。**

        返回的是**名字**不是一个布尔:「没标定」要能说出缺的是哪一个,否则消费方
        只能报一句「未标定」,而用户得自己去猜该填什么。
        """
        return tuple(f for f in SAMPLE_FACT_FIELDS if getattr(self, f) is None)

    @property
    def calibrated(self) -> bool:
        """四个样品事实齐不齐。**任何一个是 None 都算未标定。**"""
        return not self.missing_fields

    def window_problem(self) -> str:
        """扫描窗口本身有没有毛病;没有返回 ``""``。

        零宽窗口不是一次扫描 —— ``ConfigureSTS`` 未必会拒,而一条起止相同的「谱」
        会正常落盘、正常被判据吃掉,只是它一个点都没扫过。
        """
        if self.start_v is None or self.end_v is None:
            return ""            # 缺字段由 missing_fields 负责,不在这里重复报
        if float(self.start_v) == float(self.end_v):
            return (f"扫描窗口起止相同({float(self.start_v):g} V)—— 零宽窗口不是"
                    f"一次扫描,它会正常落盘、正常被判据吃掉,只是一个点都没扫过。")
        return ""

    def problems(self) -> tuple[str, ...]:
        """下发之前所有说得出来的毛病(闭集之外的自由文本,给人看)。"""
        out: list[str] = []
        if self.missing_fields:
            out.append(
                f"条件组缺少样品事实 {list(self.missing_fields)} —— 它们没有出厂值,"
                f"而且刻意没有:稳定条件与扫描窗口取决于这块样品,编一组数出来"
                f"流程会照跑不误、每条谱都「成功」,测的却是一个没人给过的假设。"
                f"由 conduct spec 或用户显式给出。")
        wp = self.window_problem()
        if wp:
            out.append(wp)
        return tuple(out)

    # ── 翻译成子技能的参数 ────────────────────────────────────────────

    def bias_settle_params(self) -> dict[str, Any]:
        """交给 ``BiasSettleChange`` 的参数。

        **是 BiasSettleChange 不是 SetBias**:恒流反馈下偏压穿零会把针尖推进样品,
        而条件序列里相邻两条的稳定偏压完全可能异号(−1 V 综览 → +1 V 综览),
        穿零是必经的(D20)。
        """
        return {"bias_v": float(self.stab_bias_v), "settle_s": float(self.settle_s)}

    def setpoint_params(self) -> dict[str, Any]:
        """交给 ``SetSetpoint``(它自己做写后回读)。"""
        return {"setpoint_a": float(self.stab_setpoint_a)}

    def sweep_config(self) -> dict[str, Any]:
        """交给 ``ConfigureSTS`` 的参数。三个数一起给,``z_offset_m`` 显式给。"""
        return {"start_v": float(self.start_v), "end_v": float(self.end_v),
                "num_points": int(self.num_points),
                "z_offset_m": float(self.z_offset_m)}

    def lockin_config(self) -> dict[str, Any]:
        """交给 ``ConfigureLockIn`` 的参数。

        **幅度与频率永远都在**,哪怕 ``mod_on=False``:省略它们会让
        ``ConfigureLockIn`` 报 0.0,而一个报出来的 0.0 与「没动」在结果里
        长得一模一样。``phase_deg`` **一个字都不给** —— 本机的调制器根本没有相位
        字段,那次写入会被固件无条件拒绝(要改相位得走解调器侧)。
        """
        return {"mod_on": bool(self.mod_on),
                "amplitude_v": float(self.mod_amp_v),
                "frequency_hz": float(self.mod_freq_hz)}

    def mls_arrays(self) -> "dict[str, list] | None":
        """交给 ``SetSTSMLSVals`` 的七个等长数组;这一条不走 MLS 时 ``None``。

        **原样透传,不在这里做第二套校验**:等长与偏压边界的检查已经在
        ``SetSTSMLSVals`` 里(那是它的归属地),在这里再写一份就有了两个真源,
        而两份校验迟早会不一致。这里只负责把逐段的 dict 转置成逐字段的数组。
        """
        segs = self.mls_segments
        if not segs:
            return None
        keys = ("bias_start_v", "bias_end_v", "initial_settling_s", "settling_s",
                "integration_s", "steps", "lockin_run")
        return {k: [s.get(k) for s in segs] for k in keys}

    def basename_tag(self) -> str:
        """``label`` 里能进文件名的那部分;空 label 给 ``""``。

        不做「空就编一个」:一个自动生成的名字会让两次不同条件的运行长得一样。
        """
        return re.sub(r"[^0-9A-Za-z_.-]+", "_", str(self.label or "")).strip("_")

    def describe(self) -> str:
        """一行人话,进 summary 与逐条记录。未标定的字段说「未标定」,不说 0。"""
        def _v(x, unit, scale=1.0, fmt="{:g}"):
            return "未标定" if x is None else fmt.format(float(x) * scale) + unit

        bits = [
            f"稳定 {_v(self.stab_bias_v, ' V')} / "
            f"{_v(self.stab_setpoint_a, ' pA', 1e12, '{:.0f}')}",
            f"窗口 {_v(self.start_v, '')}..{_v(self.end_v, ' V')} × {self.num_points} 点",
        ]
        if self.mod_on:
            bits.append(f"调制 {self.mod_amp_v * 1e3:g} mV @ {self.mod_freq_hz:g} Hz")
        else:
            bits.append("调制关")
        if self.mls_segments:
            bits.append(f"MLS {len(self.mls_segments)} 段")
        if self.label:
            bits.append(f"「{self.label}」")
        return "，".join(bits)


# ── 出厂表 ────────────────────────────────────────────────────────────────────

#: 出厂条件组。**只有结构,没有样品事实**(见模块注释)。
#:
#: 只有一个组不是偷懒:多摆两个组名(「宽窗综览」「E_F 精细」)而四个样品事实全是
#: ``None``,那两个组彼此**一模一样** —— 那是布景,不是能力。用户答出稳定条件与
#: 窗口清单之后,新的组加在这里(或由 conduct spec 以覆写的形式给全)。
CONDITIONS: dict[str, STSConditionSpec] = {
    "default": STSConditionSpec(label="default"),
}

#: 默认组名。与 ``SpectroscopyAtPositions`` / ``LineSTSAcrossWall`` 的
#: ``condition`` 参数默认值是**同一个常量**,不是两处各写一遍的字面量。
DEFAULT_CONDITION = "default"


def list_conditions() -> tuple[str, ...]:
    """已知的组名(排序)。拒绝一个不存在的组名时要报得出这张单子。"""
    return tuple(sorted(CONDITIONS))


# ── 边界:越界**拒绝**,不夹紧,也不静默丢弃 ───────────────────────────────────

#: 字段边界。每一个数最后都会变成真实的硬件动作。
#:
#: 上下界的出处:``num_points`` 与 ``ConfigureSTS`` 的 2..10000 同界;``mod_amp_v``
#: 上限 1.0 与 ``ConfigureLockIn.amplitude_v`` 同界;``stab_setpoint_a`` 下限
#: 1 pA 是本仓其它流程表用的同一个值。**这些不是本模块发明的数**,是把别处已经
#: 存在的界抄过来 —— 两个真源的界迟早会不一致,而不一致的那一天没有人会发现。
_BOUNDS: dict[str, tuple[float, float]] = {
    "stab_bias_v": (-10.0, 10.0),
    "stab_setpoint_a": (1e-12, 1e-7),
    "start_v": (-10.0, 10.0),
    "end_v": (-10.0, 10.0),
    "num_points": (2, 10000),
    "mod_amp_v": (1e-5, 1.0),
    "mod_freq_hz": (1.0, 100000.0),
    "settle_s": (0.0, 60.0),
    #: 谱学 Z offset。±1 µm 远宽于任何合理的预退,但仍然挡得住「把纳米当米填」
    #: 那一类:1 nm 写成 1 会当场被拒,而不是把针尖抬走一米。
    "z_offset_m": (-1e-6, 1e-6),
}

_INT_FIELDS = frozenset({"num_points"})
_BOOL_FIELDS = frozenset({"mod_on"})
_STR_FIELDS = frozenset({"label"})
_PASSTHROUGH_FIELDS = frozenset({"mls_segments"})

#: 拒绝的原因码(闭集)。文案会改,码不该跟着改。
REJECT_UNKNOWN_FIELD = "unknown_field"
REJECT_NOT_A_NUMBER = "not_a_number"
REJECT_OUT_OF_BOUNDS = "out_of_bounds"
ALL_REJECT_CODES: tuple[str, ...] = (
    REJECT_UNKNOWN_FIELD, REJECT_NOT_A_NUMBER, REJECT_OUT_OF_BOUNDS)


@dataclass(frozen=True)
class ConditionResolution:
    """一次组名解析的结果。

    ``spec`` 只在**组名认不出来**时才是 ``None``;其余情况总有一份 spec,哪怕它
    未标定 —— 消费方要能把「缺哪几个字段」说给人听,而不是拿到一个 None 然后只能
    报「解析失败」。
    """

    name: str
    spec: "STSConditionSpec | None"
    #: 被拒的覆写项。**非空 ⇒ 消费方拒绝整次运行**(见模块注释)。
    rejections: tuple[dict, ...] = ()
    #: 组名本身的问题;``""`` 表示组名认得出来。
    problem: str = ""
    known_names: tuple[str, ...] = ()

    @property
    def usable(self) -> bool:
        """能不能拿去下发。三个条件全要:组名认得、没有被拒的覆写、样品事实齐。"""
        return (self.spec is not None and not self.problem
                and not self.rejections and not self.spec.problems())

    def refusal_detail(self) -> str:
        """不能用时的那段话。能用时返回 ``""``。"""
        if self.problem:
            return self.problem
        bits: list[str] = []
        for r in self.rejections:
            bits.append(str(r.get("message") or r.get("code") or ""))
        if self.spec is not None:
            bits.extend(self.spec.problems())
        return " ".join(b for b in bits if b)


def _reject(field: str, value: Any, code: str, message: str) -> dict:
    return {"field": field, "value": value, "code": code, "message": message}


def _coerce(field: str, value: Any) -> "tuple[Any, dict | None]":
    """把一个覆写值转成该字段的类型并过边界。``(值, 拒绝项)``,二者必有一个是 None。"""
    if field in _PASSTHROUGH_FIELDS:
        if value is None:
            return None, None
        try:
            return tuple(dict(s) for s in value), None
        except (TypeError, ValueError):
            return None, _reject(
                field, value, REJECT_NOT_A_NUMBER,
                f"{field} 不是一串逐段配置,忽略不了也用不了 —— MLS 段要的是"
                f"[{{bias_start_v, bias_end_v, ...}}, ...] 这个形状。")
    if field in _STR_FIELDS:
        return str(value), None
    if field in _BOOL_FIELDS:
        return bool(value), None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None, _reject(field, value, REJECT_NOT_A_NUMBER,
                             f"{field}={value!r} 不是一个数。")
    if num != num or num in (float("inf"), float("-inf")):
        return None, _reject(field, value, REJECT_NOT_A_NUMBER,
                             f"{field}={value!r} 不是一个有限的数。")
    lo, hi = _BOUNDS.get(field, (float("-inf"), float("inf")))
    if not (lo <= num <= hi):
        return None, _reject(
            field, num, REJECT_OUT_OF_BOUNDS,
            f"{field}={num:g} 超出 [{lo:g}, {hi:g}] —— **拒绝,不夹紧也不丢弃**:"
            f"夹紧会让调用方以为自己设的是 {num:g} 而实际跑的是别的数,"
            f"而丢弃连「你设过一个越界值」都不会告诉任何人。")
    return (int(round(num)) if field in _INT_FIELDS else num), None


def resolve_condition(name: "str | None" = None,
                      overrides: "dict[str, Any] | None" = None,
                      *, table: "dict[str, STSConditionSpec] | None" = None,
                      ) -> ConditionResolution:
    """组名(+ 覆写)→ 一条谱的条件。**纯函数:不碰硬件、不读文件、不抛。**

    ``overrides`` 的来路是 conduct spec 或用户,不是模型 —— 模型能做的只有选
    ``name``(D19)。``None`` 值一律当「没给」,与 ``special_tip_workflow`` 一致。

    组名认不出来 ⇒ ``problem`` 里带上**已知的组名单子**。「不知道有哪些」是这一类
    拒绝里最没用的一种回答。
    """
    tbl = CONDITIONS if table is None else table
    key = str(name or DEFAULT_CONDITION).strip()
    known = tuple(sorted(tbl))
    base = tbl.get(key)
    if base is None:
        return ConditionResolution(
            name=key, spec=None, known_names=known,
            problem=(f"没有名为 {key!r} 的谱学条件组。已知的组:"
                     f"{list(known) or '一个都没有'}。条件组是**流程表**里的一项,"
                     f"由 conduct spec 或用户定义 —— 认不出来的名字不会被"
                     f"就近匹配到一个相似的组,那会让一次跑错条件的运行看上去完全正常。"))

    if not overrides:
        return ConditionResolution(name=key, spec=base, known_names=known)

    known_fields = {f.name for f in fields(base)}
    clean: dict[str, Any] = {}
    rejects: list[dict] = []
    for field, value in overrides.items():
        if value is None:
            continue                      # 「没给」,不是「设成空」
        if field not in known_fields:
            rejects.append(_reject(
                field, value, REJECT_UNKNOWN_FIELD,
                f"条件组没有 {field!r} 这个字段 —— 认不出来的键**不会被安静地丢掉**:"
                f"「我明明设了它」和「它根本没收到」长得一模一样。"
                f"已知字段:{sorted(known_fields)}。"))
            continue
        coerced, bad = _coerce(field, value)
        if bad is not None:
            rejects.append(bad)
            continue
        clean[field] = coerced
    spec = replace(base, **clean) if clean else base
    if rejects:
        logger.warning("谱学条件组 %s:%d 项覆写被拒(不夹紧)", key, len(rejects))
    return ConditionResolution(name=key, spec=spec, known_names=known,
                               rejections=tuple(rejects))


# ── 条件序列 ──────────────────────────────────────────────────────────────────

def resolve_series(names: "list[str] | tuple[str, ...]",
                   overrides: "list[dict] | None" = None,
                   *, table: "dict[str, STSConditionSpec] | None" = None,
                   ) -> list[ConditionResolution]:
    """一串组名 → 一串条件。逐条独立解析,**一条坏不拖垮其余的解析**。

    拒绝的决定留给消费方:它可能想「跑得动的先跑、坏的报出来」,也可能想整批拒绝。
    这里只负责如实报告每一条的状态。
    """
    ov = list(overrides or ())
    return [resolve_condition(n, ov[i] if i < len(ov) else None, table=table)
            for i, n in enumerate(names or ())]


def crosses_zero(prev: STSConditionSpec, nxt: STSConditionSpec) -> bool:
    """相邻两条的稳定偏压异号 —— 也就是这一步会**穿零**(D20)。

    穿零本身不被禁止,它是宽偏压序列的必经之路;要紧的是**必须走
    ``BiasSettleChange``**:恒流反馈下偏压穿零会把针尖推进样品。这个谓词只用来在
    报告里把这一步指出来,让人事后能把一次针尖变化与它对上。
    """
    a, b = prev.stab_bias_v, nxt.stab_bias_v
    if a is None or b is None:
        return False                      # 读不到不是「不穿零」,但也不能说它穿了
    return (float(a) > 0 > float(b)) or (float(a) < 0 < float(b))


__all__ = [
    "ALL_REJECT_CODES",
    "CONDITIONS",
    "DEFAULT_CONDITION",
    "REJECT_NOT_A_NUMBER",
    "REJECT_OUT_OF_BOUNDS",
    "REJECT_UNKNOWN_FIELD",
    "SAMPLE_FACT_FIELDS",
    "ConditionResolution",
    "STSConditionSpec",
    "crosses_zero",
    "list_conditions",
    "resolve_condition",
    "resolve_series",
]
