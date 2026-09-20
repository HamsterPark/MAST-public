"""对账配置的安全包络与目标仪器的实际行程。

配置比硬件宽时不能描述真实边界；配置比硬件窄时可能浪费可用行程。
本模块只生成 findings，不写 SafetyLimits、不写覆盖文件，也不自动放宽限制。
启动检查与按需初始化检查共享 reconcile_envelope，避免比较逻辑分叉。

读取 Piezo_RangeGet 的全程、ZCtrl_LimitsGet 的软限值与其启用状态。
半程为全程的一半；只有已启用的软限才参与边界求交。读取失败须明确
报告 unreadable，不拿默认值充当实际行程。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)

#: 相对差小于这个比例就不报（读数抖动 / 单位末位差，不是配置错）。
_TOLERANCE = 0.02

WIDER = "wider"        # 配置允许的比硬件能做到的多 —— 这条上限没在限制任何东西
NARROWER = "narrower"  # 配置允许的比硬件少 —— 保守，但白扔掉行程

#: Z 行程的两种来源。**字符串是用户看得见的**（进 finding.source，也进 aux 快照），
#: 所以它们说的是「哪一条 Nanonis 读回答了这个问题」，不是一个内部代号。
Z_SOURCE_PIEZO_HALF = "Piezo_RangeGet/2"
Z_SOURCE_SOFT_LIMITS = "ZCtrl_LimitsGet (enabled)"


@dataclass(frozen=True)
class ZTravel:
    """Z 行程区间及其读取来源。
    
    未启用的软限值不参与边界计算。source 用于区分压电全程推导出的半程
    与已启用的软限，便于调用方解释余量所用的分母。
    """

    lo_m: float
    hi_m: float
    source: str

    @property
    def span_m(self) -> float:
        return self.hi_m - self.lo_m

    @property
    def half_bound_m(self) -> float:
        """离中心最近的那一端的幅度 —— 与 ``SafetyLimits.z_max_m`` 同口径。"""
        return min(abs(self.lo_m), abs(self.hi_m))


def resolve_z_travel(
    *,
    piezo_z_full_m: float | None,
    z_limits_m: "tuple[float, float] | list[float] | None" = None,
    z_limits_enabled: bool | None = None,
) -> "ZTravel | None":
    """Z 能走到哪 —— **全仓唯一的一份算法**。

    * ``Piezo_RangeGet`` 给的是**全程**，半程 = 全程 / 2，且半程是**从中心算起**
      。把全程当半程用，余量会算成两倍。
    * ``ZCtrl_LimitsGet`` 的那对数字**只有 ``ZCtrl_LimitsEnabledGet == 1`` 时才算数**。
      未启用时它是上次写进去的值，不拦任何东西。``None``（没问过 / 读不到）按
      **未启用**处理 —— 不知道它启没启用的时候拿它当边界，就是在猜。
    * 两者都有且软限已启用 ⇒ **取交集**（谁更紧听谁的，逐端判断）。

    读不到任何一条 ⇒ ``None``。调用方必须把 ``None`` 当「不知道行程」处理并让判据
    静默，**绝不拿一个猜的量程去判「快顶死了」**。

    这段逻辑此前只活在 :func:`reconcile_envelope` 里，而且只在开机对账时跑一次做
    报告；采样路径上另有一份**只读软限值**的算法。同一个物理量两个算法，其中一个
    是错的 —— 现在只有这一个。
    """
    lo = hi = None
    source = ""

    if isinstance(piezo_z_full_m, (int, float)) and not isinstance(piezo_z_full_m, bool):
        half = abs(float(piezo_z_full_m)) / 2.0
        if half > 0:
            lo, hi, source = -half, half, Z_SOURCE_PIEZO_HALF

    if z_limits_enabled is True and z_limits_m is not None:
        vals = [float(v) for v in z_limits_m
                if isinstance(v, (int, float)) and not isinstance(v, bool)]
        if len(vals) >= 2:
            s_lo, s_hi = min(vals), max(vals)
            if s_hi > s_lo:
                if lo is None:
                    lo, hi, source = s_lo, s_hi, Z_SOURCE_SOFT_LIMITS
                elif s_lo > lo or s_hi < hi:
                    lo, hi = max(lo, s_lo), min(hi, s_hi)
                    source = Z_SOURCE_SOFT_LIMITS

    if lo is None or hi is None or hi <= lo:
        return None
    return ZTravel(lo_m=float(lo), hi_m=float(hi), source=source)


@dataclass(frozen=True)
class EnvelopeFinding:
    """一条对不上的账。"""

    field_name: str      # SafetyLimits 上的字段名，用户要改的就是它
    configured: float    # 配置里的值
    measured: float      # 实测推出来的对应值
    direction: str       # WIDER / NARROWER
    ratio: float         # |configured| / |measured|，1.0 = 完全一致
    source: str          # 实测值是从哪条 Nanonis 读出来的
    note: str            # 人话

    def describe(self) -> str:
        arrow = "宽" if self.direction == WIDER else "窄"
        return (
            f"{self.field_name}: 配置 {self.configured:.4g} vs 实测 "
            f"{self.measured:.4g}（{self.source}）—— 配置{arrow} {self.ratio:.3g}×。"
            f"{self.note}"
        )


@dataclass(frozen=True)
class EnvelopeReconciliation:
    """开机对账的结果。**没有任何字段是 SafetyLimits。**"""

    findings: tuple[EnvelopeFinding, ...] = ()
    measured: dict[str, Any] = field(default_factory=dict)
    unreadable: tuple[str, ...] = ()
    z_limits_enabled: bool | None = None

    @property
    def ok(self) -> bool:
        """完全对得上，且该读的都读到了。"""
        return not self.findings and not self.unreadable

    @property
    def wider(self) -> tuple[EnvelopeFinding, ...]:
        """配置比硬件宽的那些 —— 这些是「以为有护栏，其实没有」。"""
        return tuple(f for f in self.findings if f.direction == WIDER)

    def summary(self) -> str:
        if not self.findings and not self.unreadable:
            return "安全包络与实测行程一致"
        parts: list[str] = []
        if self.findings:
            parts.append(f"{len(self.findings)} 项与实测不一致")
        if self.wider:
            parts.append(f"其中 {len(self.wider)} 项配置比硬件更宽（该上限没在拦任何东西）")
        if self.unreadable:
            parts.append(f"{len(self.unreadable)} 项读不到：{', '.join(self.unreadable)}")
        return "；".join(parts)


def _floats(value: Any, n: int) -> list[float] | None:
    """从 Nanonis 回包里取 n 个浮点数。取不够返回 None。

    回包形态是 ``(error_str, raw_bytes, [values...])``；str/bytes 不是数，bool 是
    int 的子类但不是测量值，所以两者都跳过。与
    ``skills/builtins/instrument_limits.py:_floats`` 同语义。
    """
    out: list[float] = []
    stack: list[Any] = [value]
    seen = 0
    while stack and seen < 128:
        seen += 1
        v = stack.pop(0)
        if isinstance(v, (list, tuple)):
            stack = list(v) + stack
            continue
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            out.append(float(v))
    return out[:n] if len(out) >= n else None


def _compare(
    field_name: str,
    configured: float,
    measured: float,
    source: str,
    note_wider: str,
    note_narrower: str,
) -> EnvelopeFinding | None:
    """按**幅度**比较一个包络边界。相等（容差内）返回 None。

    比较下界时也使用绝对值：``abs(configured) < abs(measured)`` 表示配置更窄。
    方向由幅度决定，不由符号决定。
    """
    if measured == 0:
        return None
    ratio = abs(configured) / abs(measured)
    if abs(ratio - 1.0) <= _TOLERANCE:
        return None
    wider = abs(configured) > abs(measured)
    return EnvelopeFinding(
        field_name=field_name,
        configured=float(configured),
        measured=float(measured),
        direction=WIDER if wider else NARROWER,
        ratio=float(ratio),
        source=source,
        note=note_wider if wider else note_narrower,
    )


def reconcile_envelope(
    safe_call: Callable[..., Any],
    limits: Any,
) -> EnvelopeReconciliation:
    """读一次硬件行程，与 *limits* 比对，返回 findings。**不改任何东西。**

    ``safe_call(verb)`` 是 ``ConnectionPool.safe_call`` 那一族：返回一个带
    ``.error`` / ``.return_value`` 的记录。任何一条读失败只让它自己进
    ``unreadable``，其余照常比对 —— 半份答案强过一句 "对账失败"。
    """
    measured: dict[str, Any] = {}
    unreadable: list[str] = []
    findings: list[EnvelopeFinding] = []

    def _read(key: str, verb_result: Any, n: int) -> list[float] | None:
        if verb_result is None or getattr(verb_result, "error", None):
            unreadable.append(key)
            return None
        vals = _floats(getattr(verb_result, "return_value", None), n)
        if vals is None:
            unreadable.append(key)
            return None
        measured[key] = vals
        return vals

    # 每个 verb 都必须是 safe_call("字面量") —— 全仓的 abort 策略检查、安全审计、
    # API 覆盖率普查都是靠 grep 这个形状找 Nanonis 调用的（readback.py:_read_many
    # 那段注释讲的就是这件事）。
    try:
        rec_range = safe_call("Piezo_RangeGet")
    except Exception as exc:  # noqa: BLE001 — 对账绝不能把启动打挂
        logger.warning("envelope reconcile: Piezo_RangeGet raised: %s", exc)
        rec_range = None
    try:
        rec_zlim = safe_call("ZCtrl_LimitsGet")
    except Exception as exc:  # noqa: BLE001
        logger.warning("envelope reconcile: ZCtrl_LimitsGet raised: %s", exc)
        rec_zlim = None
    try:
        rec_zen = safe_call("ZCtrl_LimitsEnabledGet")
    except Exception as exc:  # noqa: BLE001
        logger.warning("envelope reconcile: ZCtrl_LimitsEnabledGet raised: %s", exc)
        rec_zen = None

    piezo = _read("piezo_range_m", rec_range, 3)
    zlim = _read("z_limits_m", rec_zlim, 2)
    zen_vals = _read("z_limits_enabled", rec_zen, 1)
    z_enabled: bool | None = None if zen_vals is None else bool(zen_vals[0])
    if zen_vals is not None:
        measured["z_limits_enabled"] = z_enabled

    if piezo is not None:
        # Piezo_RangeGet 返回 X/Y/Z 的**全程**。半程 = 全程 / 2。
        # 这个"全程 vs 半程"正是 §1.2 要求逐条写清的三件事之一。
        xy_full = min(abs(piezo[0]), abs(piezo[1]))
        xy_half = xy_full / 2.0
        measured["xy_half_range_m"] = xy_half
        measured["xy_full_range_m"] = xy_full
        measured["z_half_range_m"] = abs(piezo[2]) / 2.0

        for f_name in ("xy_max_m", "xy_min_m"):
            cfg = getattr(limits, f_name, None)
            if isinstance(cfg, (int, float)):
                found = _compare(
                    f_name, float(cfg), xy_half, "Piezo_RangeGet/2",
                    note_wider=(
                        "包络允许的横向坐标超出压电行程 —— 这段「上限」不拦任何东西，"
                        "真正的边界是压电本身。"
                    ),
                    note_narrower=(
                        "包络比压电行程窄，样品上有一部分区域系统看不见（覆盖率与"
                        "换区建议都按这个假边界算）。"
                    ),
                )
                if found:
                    findings.append(found)

        cfg_scan = getattr(limits, "scan_size_max_m", None)
        if isinstance(cfg_scan, (int, float)):
            found = _compare(
                "scan_size_max_m", float(cfg_scan), xy_full, "Piezo_RangeGet",
                note_wider="扫描尺寸上限大于压电全程，扫不出来的框会在硬件侧被截断。",
                note_narrower="扫描尺寸上限小于压电全程，最大视野扫不到。",
            )
            if found:
                findings.append(found)

    # Z 的硬件边界：压电半程；若 Nanonis Z 限值**已启用**且更紧，取更紧的那个。
    # 未启用的限值不算数（§1.3：手册原话 "has no effect"）。
    # 判定本身住在 :func:`resolve_z_travel` —— 采样路径（monitoring/aux_channels）
    # 用的是同一个函数，两边不许各算各的。
    if zlim is not None:
        measured["z_limits_m"] = zlim
        if z_enabled is False:
            measured["z_limits_inert"] = True
    travel = resolve_z_travel(
        piezo_z_full_m=(abs(piezo[2]) if piezo is not None else None),
        z_limits_m=zlim, z_limits_enabled=z_enabled)
    z_bound = travel.half_bound_m if travel is not None else None
    z_source = travel.source if travel is not None else Z_SOURCE_PIEZO_HALF
    if travel is not None:
        measured["z_travel_m"] = [travel.lo_m, travel.hi_m]
        measured["z_travel_source"] = travel.source

    if isinstance(z_bound, (int, float)) and z_bound > 0:
        for f_name in ("z_max_m", "z_min_m"):
            cfg = getattr(limits, f_name, None)
            if not isinstance(cfg, (int, float)):
                continue
            if cfg == 0.0:
                # z_min_m = 0.0 是出厂默认，而 Z 的负向才是朝样品那一侧
                # （§1.2）。比例比不了（分子为 0），但这恰恰是最该说的一条。
                findings.append(EnvelopeFinding(
                    field_name=f_name, configured=0.0, measured=float(z_bound),
                    direction=NARROWER, ratio=0.0, source=z_source,
                    note=("配置把 Z 行程在 0 处切断，而这台机器的 Z 可以走到 "
                          f"∓{z_bound:.3g} m —— 0 不是一个有物理含义的边界，"
                          "隧穿侧那半个行程整个被挡在门外。"),
                ))
                continue
            found = _compare(
                f_name, float(cfg), float(z_bound), z_source,
                note_wider=("Z 包络超出实际行程 —— 这条上限不构成保护，"
                            "真正的撞针保护在 SafeTip / Z 控制器 / 进针逻辑里。"),
                note_narrower="Z 包络比实际行程窄，可用行程被白白砍掉一部分。",
            )
            if found:
                findings.append(found)

    return EnvelopeReconciliation(
        findings=tuple(findings),
        measured=measured,
        unreadable=tuple(unreadable),
        z_limits_enabled=z_enabled,
    )


def report_reconciliation(result: EnvelopeReconciliation) -> None:
    """把对账结果**大声**说出去：日志 + 诊断台账。绝不抛异常。

    诊断台账（``core.diagnostics``）是刻意选的落点：它已经有 API 端点
    （``routes/diagnostics.py``）和磁盘 JSONL，所以这条记录在进程死掉之后还在，
    不需要为它新开一个接口。
    """
    try:
        if result.ok:
            logger.info("安全包络对账：与实测行程一致")
            return
        level = logger.error if result.wider else logger.warning
        level("安全包络对账：%s", result.summary())
        for f in result.findings:
            level("  %s", f.describe())
        if result.z_limits_enabled is False:
            logger.warning(
                "  Nanonis Z 位置限值**未启用**（z_limits_enabled = 0）——"
                "回读到的限值此刻不起作用（KNOWN_ISSUES §1.3）"
            )
        try:
            from mast.core.diagnostics import record as diag_record

            diag_record(
                "note", subject="safety_envelope_reconcile",
                reason=result.summary(),
                findings=[f.describe() for f in result.findings],
                measured={k: v for k, v in result.measured.items()},
                unreadable=list(result.unreadable),
                z_limits_enabled=result.z_limits_enabled,
                # 对账只报告；这条 flag 是给读台账的人看的，别把它读成"已处理"。
                auto_applied=False,
            )
        except Exception:  # noqa: BLE001
            logger.debug("envelope reconcile: diagnostics record failed",
                         exc_info=True)
    except Exception:  # noqa: BLE001 — 报告失败绝不能变成启动失败
        logger.debug("envelope reconcile: reporting failed", exc_info=True)


def reconcile_at_boot(runtime: Any, *, background: bool = True) -> Any:
    """开机对账的接线口：从运行中的 runtime 取 pool + 生效限值，跑一次并报告。

    ``background=True``（默认）时在守护线程里跑 —— 三条 Nanonis 读取在链路正常时
    是毫秒级，但链路挂掉时每条都要等到 socket 超时，而这段时间不该记在启动上。
    返回线程对象（后台）或 ``EnvelopeReconciliation``（同步，测试用）。

    任何一步取不到就安静跳过：没有 pool 的离线进程不需要对账，也不该因此报错。
    """
    pool = getattr(runtime, "_pool", None)
    safe_call = getattr(pool, "safe_call", None) if pool is not None else None
    if not callable(safe_call):
        logger.debug("envelope reconcile: no connection pool — skipped")
        return None

    from mast.core.safety import _get_effective_limits
    from mast.config import SafetyLimits

    guard = getattr(runtime, "_safety", None)
    limits = getattr(guard, "_limits", None)
    if limits is None:
        # 没有活 guard 时现算一份合并值 —— 与 api/safety_view 同一套三档降级语义。
        cfg_safety = getattr(getattr(runtime, "config", None), "safety", None)
        limits = _get_effective_limits(cfg_safety or SafetyLimits())

    def _run() -> EnvelopeReconciliation:
        result = reconcile_envelope(safe_call, limits)
        report_reconciliation(result)
        return result

    if not background:
        return _run()

    import threading

    def _thread_body() -> None:
        try:
            _run()
        except Exception:  # noqa: BLE001
            logger.debug("envelope reconcile failed", exc_info=True)

    th = threading.Thread(
        target=_thread_body, name="safety-envelope-reconcile", daemon=True
    )
    th.start()
    return th


__all__ = [
    "EnvelopeFinding",
    "EnvelopeReconciliation",
    "NARROWER",
    "WIDER",
    "ZTravel",
    "Z_SOURCE_PIEZO_HALF",
    "Z_SOURCE_SOFT_LIMITS",
    "reconcile_at_boot",
    "reconcile_envelope",
    "report_reconciliation",
    "resolve_z_travel",
]
