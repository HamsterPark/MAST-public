"""Scan policy — 按尺度分层的扫描参数表(用户偏好 + 出厂默认)。

设计文档:``docs/v2/design/scan_intelligence_scripted_rfc.md``

**这个模块存在的理由**:在它之前,一次扫描的速度 / 像素 / PI 增益是 LLM 直接
写进 ``ConfigureScan`` 的数字 —— ``width_m`` / ``height_m`` 全 required 无默认,
``line_time_s`` 唯一的硬默认是 0.1 s,SafetyGate 只拒绝不修正,
``experiment_prefs`` 只是一段可以被无视的文本提示。也就是说「这个尺度下该扫多
快、取多少像素」这类领域知识,系统里根本没有存放的地方,只能靠模型每次现编。

本模块是那张表。它**只存机械参数**(像素 / 每线时间 / setpoint / PI),
**不存 bias** —— bias 决定电子态与成像对比,是物理意图参数,不是尺度的函数
(见 :mod:`mast.core.scan_resolver` 的优先级链)。

形状照抄 :mod:`mast.core.instrument_profile`(而不是
:mod:`mast.agents._shared.experiment_prefs`),因为这里需要的是**有出厂默认、
用户可覆写**的东西:experiment_prefs 刻意「空即无值」(不替用户发明扫描
尺寸),而一张参数表如果空着就没有存在意义。

档位是**可变长的**(1..8 档,2026-07-30 定案)。出厂给 4 档模板,数值化自
``mast.knowledge.experiment_design.WORKFLOW_SEQUENCES["progressive_zoom"]`` 的四级
协议 —— 那张表在知识库里躺了很久,是人类可读的区间字符串,从来没有任何代码
消费过。**出厂值只是起点,不是真值**:随仪器 / 样品变,UI 上明示「按本机校准」。

依赖:仅 stdlib。skills 层与 api 层都要读它(skills→core、api→core 都是既有
合法方向),所以不能有 langchain / agent 依赖。
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ── 档位字段规格 ──────────────────────────────────────────────────────────────
# key -> (中文标签, 单位, 强制类型, (lo, hi), 是否必需)
#
# 必需字段(pixels / line_time_s):自定义档位缺省时从「覆盖同一尺度的出厂档」
# 回填(见 _fill_from_factory)——这是字段级 fallback,不是整档丢弃。
# 可选字段(setpoint_a / p_gain / time_constant_s):**出厂一律 None**,
# None = 不下发该硬件写(空即 no-op)。
#   * setpoint 进表是因为用户确实有「大图小电流」的习惯,但没有普适安全默认;
#   * PI 增益进表是要求要的,但增益强依赖仪器 / 针尖态,出厂不敢给数 ——
#     本机调好后填进来才生效。
_TIER_FIELD_SPEC: dict[str, tuple[str, str, type, tuple[float, float], bool]] = {
    "pixels":          ("像素(线数)", "px", int, (16, 4096), True),
    "line_time_s":     ("每线时间", "s", float, (1e-4, 600.0), True),
    # 与 SetSetpoint 的 ParameterSpec 对齐(1 pA..100 nA)—— 表里存一个下游必然
    # 拒绝的值没有意义。
    "setpoint_a":      ("电流设定点", "A", float, (1e-12, 1e-7), False),
    "p_gain":          ("Z 反馈 P 增益", "", float, (0.0, 1e6), False),
    "time_constant_s": ("Z 反馈时间常数", "s", float, (0.0, 1e3), False),
}

#: 必需字段(缺省要回填),按 _TIER_FIELD_SPEC 推导,避免两处各写一份。
_REQUIRED_TIER_FIELDS: tuple[str, ...] = tuple(
    k for k, spec in _TIER_FIELD_SPEC.items() if spec[4]
)
#: 可选字段(缺省 = 不下发)。
_OPTIONAL_TIER_FIELDS: tuple[str, ...] = tuple(
    k for k, spec in _TIER_FIELD_SPEC.items() if not spec[4]
)

#: 档位数量上下限。上限 8 是防「一档一个尺寸」把表变成查找噩梦;
#: 下限 1 允许「我只扫一个尺度」的极简用法(那一档必须是兜底档)。
MIN_TIERS = 1
MAX_TIERS = 8

#: 档名长度上限(UI 显示用,不参与任何逻辑)。
_NAME_MAX = 32

#: 扫描边长的合法范围(与 ConfigureScan 的 width_m/height_m 一致)。
#: 档位边界超出这个范围没有意义 —— 硬件根本扫不了。
SIZE_MIN_M = 1e-10
SIZE_MAX_M = 1e-5

#: 查表时边界比较的相对容差。
#:
#: 没有它会出一个很难查的 bug:用户在界面上把边界设成 100 nm、又要求扫一张
#: 100 nm 的图,界面把 nm 转成米得到 ``100.0 * 1e-9 == 1.0000000000000001e-07``,
#: 比边界**大**一个最低位,于是静默落进下一档,用了错误的速度与像素。而用户
#: 看到的两个数字明明一模一样。
#:
#: 1e-9 的相对容差在 100 nm 上是 1e-16 m(0.1 飞米)—— 物理上毫无意义,但比
#: 双精度往返误差(~1e-16 相对)大七个数量级,足够吸收界面里 nm↔m↔µm 的来回换算。
_BOUND_REL_TOL = 1e-9

# 出厂扫描档位用于通用流程配置，不能视作目标仪器或样品的标定。
# 默认档服务区域搜索与迭代，slow 档用于更慢的成图；可按名称显式选择。
# 帧时由行数、线时间和方向数推导。改变像素数与线时间时应分别考虑总时长、
# 每像素驻留与反馈响应，不能用更多像素替代必要的反馈时间。
# 使用者需依据目标仪器与样品配置档位；方向性失配应先检查反馈和线时间。
_FACTORY_TIERS: tuple[dict[str, Any], ...] = (
    {
        "name": "slow",
        "upper_size_m": 2e-9,        # ≤2 nm(也可 intent="slow" 按名字选任意尺寸)
        "pixels": 512,
        "line_time_s": 1.75,         # 512×1.75×2 = 1792 s ≈ 29.9 min
        "setpoint_a": None,
        "p_gain": None,
        "time_constant_s": None,
    },
    # atomic_verify 是更细像素尺度的验收档，与 atomic 的快速观察用途分开。
    # 尺寸、像素数与线时间需联动：增加像素必须同时考虑每像素驻留，
    # 否则像素尺度门虽通过，反馈带宽仍可能不足以提供新增信息。
    # 当前参数是可配置的工作流默认值，不构成成像质量标定；变更会影响尺寸自动查表。
    # 测试应同时验证像素尺度、驻留比值、预计帧时与调用方的档位选择。
    {
        "name": "atomic_verify",
        "upper_size_m": 5e-9,        # ≤5 nm(512 px 下 nm/px = 0.00977,满权重余量 2×)
        "pixels": 512,
        "line_time_s": 0.30,         # 512×0.30×2 = 307 s ≈ 5.1 min;每像素 586 µs
        "setpoint_a": None,
        "p_gain": None,
        "time_constant_s": None,
    },
    {
        "name": "atomic",
        "upper_size_m": 1e-8,        # ≤10 nm
        "pixels": 256,
        "line_time_s": 1.2,          # 256×1.2×2 = 614 s ≈ 10.2 min
        "setpoint_a": None,
        "p_gain": None,
        "time_constant_s": None,
    },
    {
        "name": "highres",
        "upper_size_m": 1e-7,        # ≤100 nm(缝合 50-100nm)
        "pixels": 256,
        "line_time_s": 1.0,          # 256×1.0×2 = 512 s ≈ 8.5 min ← 最常用的那档
        "setpoint_a": None,
        "p_gain": None,
        "time_constant_s": None,
    },
    {
        "name": "roi",
        "upper_size_m": 5e-7,        # ≤500 nm(缝合 500nm-1µm 归下一档)
        "pixels": 256,
        "line_time_s": 0.8,          # 256×0.8×2 = 410 s ≈ 6.8 min
        "setpoint_a": None,
        "p_gain": None,
        "time_constant_s": None,
    },
    {
        "name": "survey",
        "upper_size_m": None,        # 兜底档,必须在末位
        "pixels": 256,
        "line_time_s": 0.5,          # 256×0.5×2 = 256 s ≈ 4.3 min
        "setpoint_a": None,
        "p_gain": None,
        "time_constant_s": None,
    },
)


def factory_tiers() -> list[dict[str, Any]]:
    """出厂档位表的深拷贝(调用方可以随便改不会污染模块常量)。"""
    return [dict(t) for t in _FACTORY_TIERS]


def tier_field_specs() -> list[dict[str, Any]]:
    """每档可编辑字段的规格(给设置界面渲染输入框 + 前端做同样的范围校验)。"""
    return [
        {
            "key": key,
            "label": label,
            "unit": unit,
            "type": kind.__name__,
            "min": lo,
            "max": hi,
            "required": required,
        }
        for key, (label, unit, kind, (lo, hi), required) in _TIER_FIELD_SPEC.items()
    ]


def _coerce_num(kind: type, value: Any) -> "int | float | None":
    """float()/int() 一个值,NaN / inf / 非数值 → None。"""
    try:
        val = kind(value)
    except (TypeError, ValueError):
        return None
    if isinstance(val, float) and (val != val or val in (float("inf"), float("-inf"))):
        return None
    return val


class PolicyRejected(ValueError):
    """档位表结构非法。

    刻意做成异常而不是「静默丢弃坏档」:半张表比没有表更危险 —— 用户以为
    自己设了 6 档,系统按 4 档跑,而这个差异在任何界面上都看不出来。
    这与 experiment_folder_persistence 的 ``INSERT OR REPLACE 清空未列列``
    是同一类教训:部分成功必须显式失败。
    """


def _validate_structure(tiers: list[dict[str, Any]]) -> None:
    """结构校验 —— 违反即抛 :class:`PolicyRejected`(fail-closed,绝不存半张表)。

    五条硬规则:
      1. 档数 ∈ [MIN_TIERS, MAX_TIERS];
      2. 恰好一个兜底档(upper_size_m 为 None);
      3. 兜底档必须在**末位**(否则它后面的档永远匹配不到,是静默死代码);
      4. 有界档的 upper_size_m 严格单调递增(否则区间重叠,查表结果取决于遍历
         顺序 —— 那不是表,是巧合);
      5. 边界落在硬件可扫范围内。
    """
    n = len(tiers)
    if n < MIN_TIERS or n > MAX_TIERS:
        raise PolicyRejected(
            f"档位数量必须在 {MIN_TIERS}..{MAX_TIERS} 之间(收到 {n} 档)"
        )

    open_idx = [i for i, t in enumerate(tiers) if t.get("upper_size_m") is None]
    if len(open_idx) != 1:
        raise PolicyRejected(
            f"必须恰好有 1 个兜底档(upper_size_m 留空),收到 {len(open_idx)} 个。"
            "没有兜底档时,超过最大边界的扫描尺寸查不到任何参数。"
        )
    if open_idx[0] != n - 1:
        raise PolicyRejected(
            f"兜底档必须放在最后一档(现在在第 {open_idx[0] + 1} 档)。"
            "它前面的档才有意义;它后面的档永远匹配不到。"
        )

    prev = None
    for i, tier in enumerate(tiers[:-1]):
        upper = tier.get("upper_size_m")
        if upper is None or upper <= 0:
            raise PolicyRejected(f"第 {i + 1} 档的边界无效: {upper!r}")
        if not (SIZE_MIN_M <= upper <= SIZE_MAX_M):
            raise PolicyRejected(
                f"第 {i + 1} 档边界 {upper:.3g} m 超出硬件可扫范围 "
                f"[{SIZE_MIN_M:.0e}, {SIZE_MAX_M:.0e}] m"
            )
        if prev is not None and upper <= prev:
            raise PolicyRejected(
                f"档位边界必须严格递增:第 {i + 1} 档 {upper:.3g} m "
                f"不大于前一档 {prev:.3g} m"
            )
        prev = upper


def _factory_tier_for_size(size_m: float) -> dict[str, Any]:
    """出厂表里覆盖该尺寸的档(必定命中,因为出厂表最后一档是兜底)。"""
    for tier in _FACTORY_TIERS:
        upper = tier["upper_size_m"]
        if upper is None or size_m <= float(upper) * (1.0 + _BOUND_REL_TOL):
            return tier
    return _FACTORY_TIERS[-1]        # pragma: no cover - 兜底档保证到不了这里


def _fill_from_factory(tier: dict[str, Any], index: int, total: int) -> dict[str, Any]:
    """必需字段缺省时,从「覆盖同一尺度的出厂档」回填(字段级 fallback)。

    自定义档数与出厂档数可以不同,所以不能按下标对齐 —— 按**尺度**对齐:拿这
    一档的代表尺寸(有界档取它的上边界,兜底档取出厂最大档的代表尺寸)去查出厂
    表。这样用户加一档 "300nm 专用" 却只填了像素时,每线时间会从出厂的 roi
    档来,而不是从「第 3 档」这种与物理无关的位置来。
    """
    upper = tier.get("upper_size_m")
    if upper is None:
        # 兜底档:用一个必定落进出厂最大档的代表尺寸。
        probe = SIZE_MAX_M
    else:
        probe = float(upper)
    fallback = _factory_tier_for_size(probe)
    out = dict(tier)
    filled: list[str] = []
    for key in _REQUIRED_TIER_FIELDS:
        if out.get(key) is None:
            out[key] = fallback[key]
            filled.append(key)
    # 记下哪些字段其实是出厂值 —— resolver 的 trace 要能说实话:用户定制了
    # 表不代表每个数字都是他填的。
    out["_factory_filled"] = filled
    return out


def sanitize(raw: Any) -> list[dict[str, Any]]:
    """把一个原始档位表强制成干净、有界、结构合法的列表。

    返回 ``[]`` 表示「用户没设过」(→ 使用出厂表);返回非空列表表示一张完整
    可用的自定义表。**结构非法时抛** :class:`PolicyRejected` —— 这是刻意的:
    写入路径要能把错误显示给用户,而不是静默退回出厂值让他以为自己设好了。

    数值层面是宽容的(clamp 越界、丢 NaN、丢未知键),结构层面是 fail-closed 的
    (档数 / 兜底档 / 单调性任一违反 → 整表拒绝)。
    """
    if raw is None:
        return []
    # {"tiers": [...]} 与裸列表都接受 —— 设置里存的是前者,测试里写后者更省事。
    if isinstance(raw, dict):
        raw = raw.get("tiers")
    if raw is None:
        return []
    if not isinstance(raw, (list, tuple)):
        raise PolicyRejected(f"档位表必须是列表(收到 {type(raw).__name__})")
    if not raw:
        return []

    cleaned: list[dict[str, Any]] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise PolicyRejected(f"第 {i + 1} 档不是对象(收到 {type(item).__name__})")
        tier: dict[str, Any] = {}

        name = item.get("name")
        tier["name"] = (
            str(name).strip()[:_NAME_MAX] if isinstance(name, str) and name.strip()
            else f"tier{i + 1}"
        )

        upper_raw = item.get("upper_size_m")
        if upper_raw is None or upper_raw == "":
            tier["upper_size_m"] = None
        else:
            upper = _coerce_num(float, upper_raw)
            if upper is None:
                raise PolicyRejected(f"第 {i + 1} 档的边界不是数值: {upper_raw!r}")
            tier["upper_size_m"] = float(upper)

        for key, (_label, _unit, kind, (lo, hi), _req) in _TIER_FIELD_SPEC.items():
            val_raw = item.get(key)
            if val_raw is None or val_raw == "":
                tier[key] = None
                continue
            val = _coerce_num(kind, val_raw)
            if val is None:
                tier[key] = None
                continue
            val = max(lo, min(hi, val))
            tier[key] = int(val) if kind is int else float(val)

        cleaned.append(tier)

    _validate_structure(cleaned)
    total = len(cleaned)
    return [_fill_from_factory(t, i, total) for i, t in enumerate(cleaned)]


# ── 进程级 holder(live-read) ────────────────────────────────────────────────
_lock = threading.RLock()
_tiers: list[dict[str, Any]] = []
_persist_sink: "Callable[[list[dict[str, Any]]], None] | None" = None


def set_persist_sink(fn: "Callable[[list[dict[str, Any]]], None] | None") -> None:
    """注入持久化回调(runtime 接线)。P0 没有 runtime 写回,留给 P2 的学习回写。"""
    global _persist_sink
    with _lock:
        _persist_sink = fn


def set_policy(raw: Any) -> list[dict[str, Any]]:
    """替换活动档位表(经 sanitize)。返回存进去的副本。

    结构非法时抛 :class:`PolicyRejected` 且**不改动现有表** —— 一次坏的 POST
    不能把用户已经调好的表打成半张。
    """
    clean = sanitize(raw)
    with _lock:
        _tiers.clear()
        _tiers.extend(clean)
        return [dict(t) for t in _tiers]


def get_policy() -> list[dict[str, Any]]:
    """当前**生效**的档位表(用户表为空时返回出厂表)。

    每档带 ``source`` 字段(``factory`` / ``operator``),供 resolver 的 trace
    与设置界面直接使用 —— 「这个每线时间是我设的还是出厂的」不该靠猜。
    """
    with _lock:
        custom = [dict(t) for t in _tiers]
    if custom:
        for tier in custom:
            tier["source"] = "operator"
        return custom
    out = factory_tiers()
    for tier in out:
        tier["source"] = "factory"
    return out


def get_stored_policy() -> list[dict[str, Any]]:
    """用户**实际存过**的表(没设过 → ``[]``)。设置界面回显用。"""
    with _lock:
        return [dict(t) for t in _tiers]


def is_customised() -> bool:
    """True 表示用户改过档位表(当前跑的不是出厂值)。"""
    with _lock:
        return bool(_tiers)


def get_tier_for_size(size_m: float) -> dict[str, Any]:
    """按扫描边长查档。

    区间语义:**闭上界**,从小到大取第一个 ``size_m <= upper_size_m`` 的档;
    最后的兜底档(``upper_size_m is None``)接住所有更大的尺寸。所以边界值
    落在**小的那一档**(出厂表里正好 100 nm 属 highres 而不是 roi)。

    这是台阶函数,**不插值**:用户的心智模型是档位(「大图我用 256」),
    插值会产出 384px 这种非常规像素数,而且 trace 里「来自 roi 档」远比
    「roi 与 highres 的对数插值 0.63」可用。
    """
    try:
        size = float(size_m)
    except (TypeError, ValueError):
        size = 0.0
    if size != size:                      # NaN
        size = 0.0
    tiers = get_policy()
    for tier in tiers:
        upper = tier.get("upper_size_m")
        if upper is None:
            return tier
        bound = float(upper)
        if size <= bound * (1.0 + _BOUND_REL_TOL):
            return tier
    return tiers[-1]                      # pragma: no cover - 兜底档保证到不了



def resolve_line_time(explicit, size_m, *, fallback: float = 0.1):
    """统一解析每线时间：显式参数优先，其次档位表，最后兜底。
    
    ConfigureScan、FullScan 与 PreScanCheck 共用此入口，避免不同路径悄悄采用
    不同线时间。返回 (秒, 来源)，来源必须进入调用记录，便于检查配置是否生效。"""
    if explicit is not None:
        try:
            v = float(explicit)
            if v > 0:
                return v, "explicit"
        except (TypeError, ValueError):
            pass
    try:
        tier = get_tier_for_size(float(size_m or 0.0))
        return float(tier["line_time_s"]), "tier:%s" % tier.get("name")
    except Exception:  # noqa: BLE001 — 查表失败绝不能让扫描起不来
        logger.debug("线时间档位表查询失败，退回 %.3f s", fallback, exc_info=True)
        return float(fallback), "fallback"

def get_tier_by_name(name: str) -> "dict[str, Any] | None":
    """按档名取档(``purpose`` 强制换档用)。找不到返回 None。"""
    if not isinstance(name, str) or not name.strip():
        return None
    want = name.strip().lower()
    for tier in get_policy():
        if str(tier.get("name", "")).strip().lower() == want:
            return tier
    return None


def tier_names() -> list[str]:
    """当前生效表的档名列表(给 ScanAt 的 purpose 枚举 / UI 下拉)。"""
    return [str(t.get("name", "")) for t in get_policy()]


def estimate_scan_seconds(pixels: Any, line_time_s: Any) -> float:
    """一帧的估计耗时(秒)= 线数 × 每线时间 × 2(正反扫)。

    给规划器排预算和给用户「这张图要扫多久」用。往返因子 2 是保守估计:
    Nanonis 的正反扫时间可以分别设,但本表只存一个 line_time,两个方向同速。
    """
    px = _coerce_num(int, pixels)
    lt = _coerce_num(float, line_time_s)
    if px is None or lt is None or px <= 0 or lt <= 0:
        return 0.0
    return float(px) * float(lt) * 2.0

# 等待预算在预计帧时上增加比例余量和固定开销。
# 这些是软件预算默认值，覆盖启动和存盘等额外等待，不是仪器实测标定。
_WAIT_HEADROOM_FRAC = 1.3
_WAIT_HEADROOM_S = 30.0


def wait_budget_s(estimated_scan_s: Any, *, floor_s: float) -> float:
    """统一推导扫描等待预算：max(floor_s, 预计帧时乘比例余量加固定开销)。
    
    ScanAt 与 PreScanCheck 共用此函数；预计帧时跟随当前分辨率与线时间。
    调用方可采用不同最低预算，不能另写与实际帧时无关的固定等待值。"""
    est = _coerce_num(float, estimated_scan_s) or 0.0
    return max(float(floor_s), est * _WAIT_HEADROOM_FRAC + _WAIT_HEADROOM_S)


def format_policy_block(tiers: "list[dict[str, Any]] | None" = None) -> str:
    """把档位表渲染成人类可读的表格(设置界面 / 上下文注入查看器用)。

    刻意**不**做成注入给 LLM 的 system 块:模型不需要知道这张表 —— 它连数字
    都不该填。这个函数是给人看的。
    """
    rows = tiers if tiers is not None else get_policy()
    if not rows:
        return ""
    lines = ["## 扫描参数档位表(按尺度)", ""]
    for tier in rows:
        upper = tier.get("upper_size_m")
        bound = "更大" if upper is None else f"≤ {float(upper) * 1e9:.4g} nm"
        src = {"factory": "出厂", "operator": "用户"}.get(
            str(tier.get("source", "")), ""
        )
        src_s = f" [{src}]" if src else ""
        px = tier.get("pixels")
        lt = tier.get("line_time_s")
        est = estimate_scan_seconds(px, lt)
        parts = [f"- **{tier.get('name')}**({bound}){src_s}: {px}px, {lt}s/线"]
        if est > 0:
            parts.append(f"≈{est / 60.0:.1f} min/帧")
        extras = []
        if tier.get("setpoint_a") is not None:
            sp = float(tier["setpoint_a"])
            extras.append(f"setpoint {sp:.3g} A (= {sp * 1e12:.4g} pA)")
        if tier.get("p_gain") is not None:
            extras.append(f"P={tier['p_gain']:.4g}")
        if tier.get("time_constant_s") is not None:
            extras.append(f"T={tier['time_constant_s']:.4g}s")
        if extras:
            parts.append("; ".join(extras))
        lines.append(", ".join(parts))
    return "\n".join(lines)


__all__ = [
    "MIN_TIERS",
    "MAX_TIERS",
    "SIZE_MIN_M",
    "SIZE_MAX_M",
    "PolicyRejected",
    "factory_tiers",
    "tier_field_specs",
    "sanitize",
    "set_policy",
    "get_policy",
    "get_stored_policy",
    "is_customised",
    "get_tier_for_size",
    "resolve_line_time",
    "get_tier_by_name",
    "tier_names",
    "estimate_scan_seconds",
    "wait_budget_s",
    "format_policy_block",
    "set_persist_sink",
]
