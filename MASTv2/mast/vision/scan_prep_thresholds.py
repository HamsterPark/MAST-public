"""扫描图预处理的命名阈值 profile，支持只读 JSON 配置与分布检查。

公开版仅带 ``generic-uncommissioned``：未标定的合成示例，不来自任何实验批次，
不声称适用于任何样品或仪器。行纯度、峰间距比、FFT 信噪比、周期带、扫描轴死区、
行相关和缺失比例使用演示参数；模型选择增益与色阶百分位是算法示例参数。
这些数值用于演示流程，不构成测量或仪器安全判据。起伏门的两个标定字段保持 None。

用于实际数据前应先运行 :mod:`mast.vision.scan_prep_commission` 检查分布，
再由使用者提供经过验证的 profile。``provenance`` 随技能返回与报告传递，
必须明确实际依据；不得把默认示例当作已标定数据。

自定义配置 ``<project_root>/config/scan_prep_profiles.json`` 的结构示例::

    {
      "custom-example": {
        "provenance": "由使用者填写验证依据；此处没有预置标定",
        "base": "generic-uncommissioned",
        "thresholds": {"step_sep": 6.0, "rowcorr_poor": 0.45}
      }
    }

省略 base 时使用内建示例。文件按 mtime 惰性重载；损坏配置记录警告并忽略。
本模块只读配置，分布检查工具也不会自动写入建议。
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

logger = logging.getLogger(__name__)

#: 内建默认 profile 的名字。所有「没指定 profile」的调用落到这里。
DEFAULT_PROFILE = "generic-uncommissioned"

CONFIG_FILENAME = "scan_prep_profiles.json"


@dataclass(frozen=True)
class ScanPrepThresholds:
    """一套扫描图预处理阈值的不可变快照。

    字段分两类,标定方式完全不同(见 :data:`CALIBRATABLE_FROM_DISTRIBUTION`):

    * **随样品漂移的**(``step_sep`` / ``rowcorr_poor`` / ``fine_periodic_snr`` /
      ``nan_annotate``):它们的合适取值取决于这块样品长什么样、这台机器有多吵,
      换体系必须重看分布。
    * **不随样品漂移的**(``line_gain`` / ``bow_gain`` / ``step_purity``):它们是关于
      *模型选择* 和 *几何* 的陈述。「逐行平场把残差压掉 30% 以上才值得用」与
      「真台阶横跨画面所以大多数行同时含两个高度」在任何样品上都成立。改这几个数
      要的是一个论证,不是一份分布。
    """

    # ── 模型选择(不随样品漂移) ──────────────────────────────────────────
    #: std(扣平面残差) / std(逐行一阶平场残差)。按构造 ≥1;>此值才逐行平场。
    line_gain: float = 1.30
    #: std(扣平面残差) / std(扣二阶曲面残差)。>此值说明面是弯的,基线改用二阶曲面。
    bow_gain: float = 1.15
    #: 两个高度能级中,「整行落在同一侧」的行占比。>此值 = 行向分层(针尖突变/z 漂移),
    #: **不是**台阶,不该被保护；该示例阈值只用于合成图演示。
    step_purity: float = 0.75

    # ── 表面形貌(随样品漂移) ────────────────────────────────────────────
    #: 高度直方图里算作「有能级」的峰数下限。
    step_peaks: float = 2.0
    #: 峰间距 / 像素级粗糙度。台阶密集的样品上这个值要重标。
    step_sep: float = 4.0

    # ── 精细周期结构(随样品/放大倍率漂移) ───────────────────────────────
    #: 受限带 FFT 峰对局部环背景的信噪比。**只用来决定色阶**,不产出「有没有晶格」
    #: 的结论 —— 那个结论来自 :func:`mast.vision.atomic_phase.assess_atomic_phase`。
    fine_periodic_snr: float = 10.0
    #: 可信周期带(纳米)。
    fine_period_min_nm: float = 0.10
    fine_period_max_nm: float = 2.00
    #: FFT 里两条扫描轴附近的死区半角(度)。扫描线噪声与逐行平场都在轴上留下强脊,
    #: 示例用 22.5°（八分之一半圆的角度）屏蔽轴附近；实际死区须按数据验证。
    axis_guard_deg: float = 22.5

    # ── 帧质量标注(随机器漂移) ──────────────────────────────────────────
    #: 相邻行相关的中位数低于此值 → 标注「噪声帧」。
    rowcorr_poor: float = 0.50
    #: NaN 像素占比高于此值 → 标注「扫描未完成」。
    nan_annotate: float = 0.01
    #: 坏行占比高于此值 → 标注。取自 :func:`mast.vision.scan_artifacts.detect_scan_artifacts`。
    bad_row_frac_annotate: float = 0.0
    #: 正反扫**不稳定度**上限(MAST 口径 = 1 − 允许横向位移的最大归一化互相关)。
    #: 注意这与 sxm_auto 的 ``fb_corr`` 方向相反:那边是「相关 > 0.5 算一致」,
    #: 这边是「不稳定度 < 0.5 算一致」,且这边补偿了压电迟滞的快轴偏移。
    fb_instability_max: float = 0.50

    # ── 批次一致性 ──────────────────────────────────────────────────────
    #: 同 (视野, 偏压) 组内至少这么多张才做多数票。
    group_min: float = 3.0

    # ── 色阶(百分位) ───────────────────────────────────────────────────
    clip_lattice_lo: float = 2.0
    clip_lattice_hi: float = 98.0
    clip_step_lo: float = 0.1
    clip_step_hi: float = 99.9
    clip_default_lo: float = 1.0
    clip_default_hi: float = 99.0

    # ── 起伏门(判据②;随样品漂移,**出厂就是「判不了」**) ─────────────────
    #
    # 这两个是**一组**,不是两个独立旋钮:一个 pm 阈值离开它标定时的视野就没有
    # 意义；改变采样尺度会改变可分辨的结构与滤波响应。因此
    # `judge_corrugation` 缺任何一个都返回 `undecidable`,**不换算、不外推**。
    #
    # ⚠️ 它们与既有字段的**越界处理不一样**:见 :data:`_NULLABLE_FIELDS` ——
    # 既有字段夹紧,这两个丢弃 + warning。既有字段的夹紧行为**没有跟着改**,
    # 那是另一个决定,要单独论证。
    #
    # ⚠️ pm 阈值依赖使用者声明的 z 标定状态；口径说明见
    # :data:`mast.vision.corrugation_gate.Z_CAL_NOTE`。z 重标之后这两个数作废,
    # 按比例缩放不算数 —— 分离度不一定跟着缩放存活。标定时把 z 标定状态写进
    # ``provenance``,让下一个人拿到这个数就看见它的口径。
    #: 起伏上限(pm)。``None`` = **判不了**,不是「没有上限所以都算正常」。
    corrugation_high_pm: float | None = None
    #: 上面那个阈值是在多大的视野上标的(nm)。``None`` = 没声明 ⇒ 判不了。
    corrugation_ref_scan_nm: float | None = None

    # ── 元信息(不是阈值,不参与标定) ─────────────────────────────────────
    #: 这组数是在什么数据上标出来的。必填,会跟着进每一份报告。
    provenance: str = '未标定（uncommissioned）：仅供合成示例和算法演示，未经任何样品或仪器验证。用于实际数据前须运行 python -m mast.vision.scan_prep_commission <folder>，检查本批分布、声明标定依据并提供自定义 profile；本示例不代表测量结论。'
    #: profile 名字,由 :func:`resolve` 填,便于结果里回溯。
    name: str = ""

    @property
    def clip_lattice(self) -> tuple[float, float]:
        return (float(self.clip_lattice_lo), float(self.clip_lattice_hi))

    @property
    def clip_step(self) -> tuple[float, float]:
        return (float(self.clip_step_lo), float(self.clip_step_hi))

    @property
    def clip_default(self) -> tuple[float, float]:
        return (float(self.clip_default_lo), float(self.clip_default_hi))

    def to_mapping(self) -> dict:
        return dict(asdict(self))

    def numeric_mapping(self) -> dict[str, float]:
        """只要**有值的**数值字段 —— 标定工具与 UI 用这个,不会撞上 provenance/name。

        可空字段(:data:`_NULLABLE_FIELDS`)没填时**整个键不出现**,而不是给一个
        0.0:「没标定」与「标成 0」是两件事,而下游拿到的是 ``f"{v:g}"`` 这种
        格式化 —— 一个 ``None`` 会当场炸,一个 0.0 会**看不出来**。
        """
        return {k: float(v) for k, v in asdict(self).items()
                if k not in _META_FIELDS and v is not None}

    @classmethod
    def from_mapping(cls, m: dict | None, *, base: "ScanPrepThresholds | None" = None
                     ) -> "ScanPrepThresholds":
        """从(部分)映射构造,对持久化内容宽容。

        未知键忽略,缺失键回落到 ``base``(默认内建默认值),数值键按
        :data:`FIELD_BOUNDS` 夹紧。``provenance`` / ``name`` 作为字符串原样取用。

        ⚠️ **两种越界处理并存,这是有意的**:

        * 既有数值字段 —— **夹紧**(历史行为,没跟着改);
        * :data:`_NULLABLE_FIELDS` —— **丢弃 + warning**,而且 ``None`` 保持
          ``None``。它们是判据阈值,夹紧会把一个越界的阈值静默改成边界值,
          而调用方以为自己设的是原值;对一个「``None`` = 判不了」的字段来说,
          那等于凭空造出一个从没标定过的判据。
        """
        start = base if base is not None else cls()
        if not m:
            return start
        known = {f.name for f in fields(cls)}
        clean: dict = {}
        for k, v in m.items():
            if k not in known:
                continue
            if k in _META_FIELDS:
                if isinstance(v, str) and v.strip():
                    clean[k] = v.strip()
                continue
            if k in _NULLABLE_FIELDS:
                if v is None:
                    # 显式 null = 「这个 profile 不声明这个口径」= 判不了。
                    # 保留它(而不是继承 base 的值),因为「判不了」是保守的一侧。
                    clean[k] = None
                    continue
                if isinstance(v, bool) or not isinstance(v, (int, float)):
                    logger.warning("scan_prep profile: %s=%r 不是数值,丢弃", k, v)
                    continue
                num = float(v)
                lo, hi = FIELD_BOUNDS.get(k, (float("-inf"), float("inf")))
                if not (lo <= num <= hi):
                    logger.warning(
                        "scan_prep profile: %s=%g 超出 [%g, %g],丢弃(**不夹紧**)"
                        " —— 判据阈值夹紧了就看不出兜底发生过", k, num, lo, hi)
                    continue
                clean[k] = num
                continue
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                continue
            lo, hi = FIELD_BOUNDS.get(k, (float("-inf"), float("inf")))
            clean[k] = float(min(hi, max(lo, float(v))))
        return replace(start, **clean)


#: 非数值字段 —— 标定/夹紧一律跳过它们。
_META_FIELDS: frozenset[str] = frozenset({"provenance", "name"})

#: 可空数值字段:``None`` = **判不了**(不是 0,也不是「没有限制」)。
#:
#: 与既有字段的两处不同,都写在 :meth:`ScanPrepThresholds.from_mapping` 里:
#: 越界**丢弃 + warning**(不夹紧),``None`` 保持 ``None``。
#: 出现这一族是因为「阈值还没标定」在本仓是一个**真实且常见**的状态 ——
#: 给它一个数就是伪造标定,而夹紧会让伪造看不出来。
_NULLABLE_FIELDS: frozenset[str] = frozenset({
    "corrugation_high_pm",
    "corrugation_ref_scan_nm",
})

#: 每个数值 knob 的合法区间。下界不是「最小可测值」而是「小于它这条判据就失效」:
#: ``line_gain``/``bow_gain`` 按构造 ≥1,设成 <1 等于「永远触发」。
FIELD_BOUNDS: dict[str, tuple[float, float]] = {
    "line_gain": (1.0, 10.0),
    "bow_gain": (1.0, 10.0),
    "step_purity": (0.0, 1.0),
    "step_peaks": (2.0, 16.0),
    "step_sep": (1.0, 100.0),
    "fine_periodic_snr": (1.0, 1000.0),
    "fine_period_min_nm": (0.05, 5.0),
    "fine_period_max_nm": (0.1, 50.0),
    "axis_guard_deg": (0.0, 44.0),
    "rowcorr_poor": (0.0, 1.0),
    "nan_annotate": (0.0, 1.0),
    "bad_row_frac_annotate": (0.0, 1.0),
    "fb_instability_max": (0.0, 1.0),
    "group_min": (2.0, 100.0),
    "clip_lattice_lo": (0.0, 49.0),
    "clip_lattice_hi": (51.0, 100.0),
    "clip_step_lo": (0.0, 49.0),
    "clip_step_hi": (51.0, 100.0),
    "clip_default_lo": (0.0, 49.0),
    "clip_default_hi": (51.0, 100.0),
    # 可空字段(_NULLABLE_FIELDS):这两行是**拒绝线**不是夹紧线 —— 越界丢弃。
    # 下界只是输入校验边界，不代表可测分辨率或已验证的物理阈值。
    "corrugation_high_pm": (0.1, 1e6),
    "corrugation_ref_scan_nm": (0.1, 1e5),
}

#: 中文标签 + 一句提示,给设置 UI / 标定报告用。
KNOB_LABELS: dict[str, tuple[str, str]] = {
    "line_gain": ("逐行平场增益阈", "扣平面残差 ÷ 逐行平场残差,超过才逐行平场"),
    "bow_gain": ("二阶曲面增益阈", "扣平面残差 ÷ 扣二阶残差,超过说明面是弯的"),
    "step_purity": ("行纯度阈", "超过即判定为行向分层(针尖突变),不是台阶"),
    "step_peaks": ("高度峰数下限", "直方图里有几个峰才算「有能级」"),
    "step_sep": ("峰间距/粗糙度阈", "台阶密集的样品上最需要重标的一个"),
    "fine_periodic_snr": ("精细周期 SNR 阈", "只决定色阶,不产出「有没有晶格」的结论"),
    "fine_period_min_nm": ("周期带下限(nm)", "小于此周期的谱峰当采样噪声"),
    "fine_period_max_nm": ("周期带上限(nm)", "大于此周期的当大尺度形貌"),
    "axis_guard_deg": ("轴向死区(度)", "FFT 里绕开两条扫描轴,防条纹伪影冒充周期结构"),
    "rowcorr_poor": ("行相关差阈", "相邻行相关中位数低于此值标注「噪声帧」"),
    "nan_annotate": ("NaN 占比阈", "超过即标注「扫描未完成」"),
    "bad_row_frac_annotate": ("坏行占比阈", "超过即标注受扰扫描线"),
    "fb_instability_max": ("正反扫不稳定度上限", "1−最大归一化互相关;已补偿压电迟滞偏移"),
    "group_min": ("批次多数票下限", "同视野同偏压至少几张才统一处理方式"),
    "clip_lattice_lo": ("色阶·有周期结构·下", "百分位"),
    "clip_lattice_hi": ("色阶·有周期结构·上", "百分位"),
    "clip_step_lo": ("色阶·有台阶·下", "百分位"),
    "clip_step_hi": ("色阶·有台阶·上", "百分位"),
    "clip_default_lo": ("色阶·默认·下", "百分位"),
    "clip_default_hi": ("色阶·默认·上", "百分位"),
    "corrugation_high_pm": (
        "起伏上限(pm)",
        "超过即判「起伏极大」。**空 = 判不了**,不是「没有上限」;"
        "它与下面的视野是一组,缺一个就判不了。"
        "注意低于弃权门(15 pm)的帧一律弃权,所以上限设得比它还低没有意义"),
    "corrugation_ref_scan_nm": (
        "起伏阈值的标定视野(nm)",
        "上面那个 pm 数是在多大的视野上标的。实际帧与它差超 5% ⇒ 判不了,"
        "**不换算、不外推**"),
}

#: 能从「跑一批图看分布」标定出来的 knob。其余的是关于模型选择/几何的陈述,
#: 不随样品的噪声水平漂移 —— 标定工具对它们只报双侧样本数,不给建议值。
#: 这个区分是 :mod:`mast.monitoring.commission` 里同一条纪律的翻版:那边把三条
#: CRITICAL 排除在标定之外,理由是「贴轨就是贴轨,与健康基线无关」。
#:
#: ⚠️ ``corrugation_high_pm`` 在这张表里,但 :mod:`mast.vision.scan_prep_commission`
#: 的 ``_METRICS`` 还没有对应的量 ⇒ 它**不会出现在标定报告里**。这是如实的:
#: 起伏门的选型(去趋势 × 统计量的 2×2,以及判据有效性四检验)还没跑过,
#: 没有口径就没有可标的分布。写在这里是为了「看不到它」不被读成「接线错了」。
#: 另外三个口径声明类的数(视野/去趋势/统计量)**不属于**这张表 —— 它们是关于
#: 模型选择的陈述,不是能从分布里标出来的数。
CALIBRATABLE_FROM_DISTRIBUTION: tuple[str, ...] = (
    "step_sep",
    "rowcorr_poor",
    "fine_periodic_snr",
    "fb_instability_max",
    "corrugation_high_pm",
)

#: 内建 profile 目录。**只有一个** —— 见模块注释。
PROFILES: dict[str, ScanPrepThresholds] = {
    DEFAULT_PROFILE: ScanPrepThresholds(
        name=DEFAULT_PROFILE,
        provenance='未标定（uncommissioned）：仅供合成示例和算法演示，未经任何样品或仪器验证。用于实际数据前须运行 python -m mast.vision.scan_prep_commission <folder>，检查本批分布、声明标定依据并提供自定义 profile；本示例不代表测量结论。',
    ),
}


# ── 外部 profile(JSON,只读) ───────────────────────────────────────────────

_lock = threading.Lock()
_active_name: str = DEFAULT_PROFILE
_file_cache: tuple[float, dict[str, ScanPrepThresholds]] | None = None


def config_path() -> Path:
    """自定义 profile 文件的位置(跟随 ``MAST2_PROJECT_ROOT`` / 冻结版路径)。"""
    from mast._runtime_paths import project_root

    return project_root() / "config" / CONFIG_FILENAME


def _load_file_profiles() -> dict[str, ScanPrepThresholds]:
    """读外部 profile,按 mtime 缓存。任何错误都退化成「没有外部 profile」。"""
    global _file_cache
    try:
        p = config_path()
        mtime = p.stat().st_mtime if p.exists() else 0.0
    except Exception:  # noqa: BLE001 — 配置文件不该让分析崩掉
        return {}
    cached = _file_cache
    if cached is not None and cached[0] == mtime:
        return cached[1]
    out: dict[str, ScanPrepThresholds] = {}
    if mtime:
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                for name, body in raw.items():
                    if not isinstance(body, dict):
                        continue
                    base_name = str(body.get("base") or DEFAULT_PROFILE)
                    base = PROFILES.get(base_name, PROFILES[DEFAULT_PROFILE])
                    th = ScanPrepThresholds.from_mapping(
                        body.get("thresholds") or {}, base=base)
                    prov = str(body.get("provenance") or "").strip()
                    out[str(name)] = replace(
                        th, name=str(name),
                        provenance=prov or f"外部 profile,未写 provenance(基于 {base_name})")
        except Exception as exc:  # noqa: BLE001
            logger.warning("scan_prep profile 文件读不动,忽略:%s: %s",
                           type(exc).__name__, exc)
            out = {}
    _file_cache = (mtime, out)
    return out


def available_profiles() -> dict[str, str]:
    """``{profile 名: provenance}`` —— 内建 + 外部。外部同名覆盖内建。"""
    out = {n: t.provenance for n, t in PROFILES.items()}
    out.update({n: t.provenance for n, t in _load_file_profiles().items()})
    return out


def resolve(profile: "str | ScanPrepThresholds | None" = None,
            *, overrides: dict | None = None) -> ScanPrepThresholds:
    """取一套阈值。

    ``profile`` 可以是名字、一个现成的 :class:`ScanPrepThresholds`,或 ``None``
    (用当前激活的 profile)。名字不认识时**回落到默认并在 provenance 里说清楚** ——
    静默用一套别的阈值算完再报一个数,是最难查的那种错。

    ``overrides`` 是单次调用的逐字段覆盖(技能参数走这条),不改动 profile 本身。
    """
    if isinstance(profile, ScanPrepThresholds):
        th = profile
    else:
        name = (profile or get_active_profile_name()).strip()
        merged = dict(PROFILES)
        merged.update(_load_file_profiles())
        th = merged.get(name)
        if th is None:
            fallback = merged.get(DEFAULT_PROFILE) or ScanPrepThresholds()
            th = replace(
                fallback, name=DEFAULT_PROFILE,
                provenance=f"(请求的 profile '{name}' 不存在,已回落到 "
                           f"{DEFAULT_PROFILE}) {fallback.provenance}")
    if overrides:
        th = ScanPrepThresholds.from_mapping(overrides, base=th)
    return th


def get_active_profile_name() -> str:
    return _active_name


def set_active_profile(name: "str | None") -> str:
    """切换当前样品体系的 profile。不认识的名字**不切换**,返回当前值。"""
    global _active_name
    want = (name or DEFAULT_PROFILE).strip()
    if want not in available_profiles():
        logger.warning("scan_prep profile '%s' 不存在,保持 '%s'", want, _active_name)
        return _active_name
    with _lock:
        _active_name = want
    return _active_name


def knob_catalog(profile: "str | None" = None) -> list[dict]:
    """把每个 knob 描述成数据(名字/标签/区间/默认/当前/能不能从分布标定)。

    形状与 :func:`mast.monitoring.thresholds.knob_catalog` 一致,所以将来接设置 UI
    时前端不必硬编码键名。本次**没有**接进 ``/api/settings`` —— 那条路上的文件
    (``webui/settings_store.py`` / ``api/routes/settings.py``)正被别的改动占用。
    """
    th = resolve(profile)
    active = th.numeric_mapping()
    defaults = PROFILES[DEFAULT_PROFILE].numeric_mapping()
    out: list[dict] = []
    for key in sorted(defaults):
        lo, hi = FIELD_BOUNDS.get(key, (0.0, 0.0))
        label, hint = KNOB_LABELS.get(key, (key, ""))
        out.append({
            "key": key,
            "label_zh": label,
            "hint_zh": hint,
            "min": float(lo),
            "max": float(hi),
            "default": float(defaults[key]),
            "value": float(active.get(key, defaults[key])),
            "calibratable_from_distribution": key in CALIBRATABLE_FROM_DISTRIBUTION,
            "nullable": False,
        })
    # 可空 knob 出厂就没有值,所以它们不在 numeric_mapping 里 —— 单独列出来,
    # 并把 ``None`` 如实带出去。UI 必须把它渲染成「未标定」而不是 0:
    # 一个显示成 0 的阈值会让人以为这道门开着(而它其实是关的)。
    for key in sorted(_NULLABLE_FIELDS):
        lo, hi = FIELD_BOUNDS.get(key, (0.0, 0.0))
        label, hint = KNOB_LABELS.get(key, (key, ""))
        val = getattr(th, key, None)
        out.append({
            "key": key,
            "label_zh": label,
            "hint_zh": hint,
            "min": float(lo),
            "max": float(hi),
            "default": None,
            "value": float(val) if val is not None else None,
            "calibratable_from_distribution": key in CALIBRATABLE_FROM_DISTRIBUTION,
            "nullable": True,
        })
    return out


__all__ = [
    "CALIBRATABLE_FROM_DISTRIBUTION",
    "CONFIG_FILENAME",
    "DEFAULT_PROFILE",
    "FIELD_BOUNDS",
    "KNOB_LABELS",
    "PROFILES",
    "ScanPrepThresholds",
    "available_profiles",
    "config_path",
    "get_active_profile_name",
    "knob_catalog",
    "resolve",
    "set_active_profile",
]
