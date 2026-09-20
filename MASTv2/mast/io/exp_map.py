"""Live experiment map — the spatial record of *where on the surface* every
operation happened, and where planned ones will happen.

This is the data + render layer behind the IC "扫描地图" tab. It is, by design,
**the experiment record's map**: markers are read straight from the
``map_markers`` table (``ExperimentStorage.get_markers``) so the picture and the
record can never drift apart. The live current scan frame + tip position are
overlaid from the cached ``HardwareState`` snapshot, and a planning overlay
draws the proposed route ahead like a car-navigation preview.

Pure / offline: no hardware, no agent state, no Nanonis client. ``classify_skill``
+ ``marker_from_skill`` turn a (skill_name, params, state) triple into a marker
for the recorder to persist; ``render_map_figure`` turns a marker list + live
frame + plan into a matplotlib Agg figure. Everything is unit-testable without a
GUI or an instrument.

Coordinate frame: Nanonis stage frame, METRES (displayed in nm). Scan footprints
are axis-aligned unless an angle is given (drawn rotated about their centre).
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from mast.core.types import HardwareState

logger = logging.getLogger(__name__)


# ── Marker model ────────────────────────────────────────────────────────────

# kind → (colour, matplotlib marker, Chinese label). "scan"/"frame"/"manual"
# footprints draw as rectangles; the rest draw as scatter points.
KIND_STYLE: dict[str, tuple[str, str, str]] = {
    "scan":        ("#3b82f6", "s", "扫图"),
    "sts":         ("#22c55e", "o", "STS 点谱"),
    "pulse":       ("#f59e0b", "*", "电脉冲"),
    "tip_shape":   ("#ef4444", "^", "修针尖"),
    "move":        ("#94a3b8", ".", "移动"),
    "manual":      ("#e879f9", "D", "手动操作"),
    "plan":        ("#38bdf8", "X", "计划"),
    "frame":       ("#06b6d4", "s", "当前扫描框"),
    "tip":         ("#f43f5e", "P", "针尖"),
    # 2026-07-30 — the three events that decide WHERE the tip may still go.
    # ``coarse_move`` also ends one coordinate generation and starts the next
    # (see ``map_markers.coord_epoch``).
    "coarse_move": ("#8b5cf6", "H", "粗动换区"),
    "approach":    ("#14b8a6", "v", "进针"),
    "crash":       ("#991b1b", "x", "撞针"),
}

# Positioned skill-name → marker kind. Substring match, lower-cased. Order
# matters: more specific patterns first. A skill that matches nothing here is
# NOT positioned and produces no marker.
#
# ``coarse_move`` and ``crash`` are deliberately ABSENT: their coordinate
# semantics differ from this table's generic precedence (a coarse move records
# where the boundary fell in the OLD frame; a crash records the frame centre of
# the scan that hit), so they are written by explicit branches in the recorder.
# Routing them through here would place them plausibly but wrongly.
_SKILL_KIND_RULES: list[tuple[tuple[str, ...], str]] = [
    (("prescancheck", "assessimage", "assessscan", "trackdrift"), ""),  # not a marker
    (("pulse",), "pulse"),
    # ``spectroscopytip`` MUST be here rather than relying on the sts rule below:
    # "MakeSpectroscopyTip" contains "spectr", so without this entry the sts rule
    # would claim it and every shallow plunge that recipe makes would be filed as
    # a spectroscopy POINT — no avoidance circle, and the next FindCleanSpot would
    # hand the tip straight back to a crater it just dug.
    (("tipshape", "conditiontip", "shapetip", "tip_shape", "fixtip",
      "tipprep", "tipcondition", "spectroscopytip", "resolutiontip"),
     "tip_shape"),
    (("biasspectr", "gridsts", "linests", "sts", "spectr", "didv",
      "ispectr", "zspectr"), "sts"),
    # Stop/status siblings FIRST: "StopAutoApproach" and
    # "GetAutoApproachStatus" both contain "autoapproach" as a substring, and
    # neither drives the tip into the surface — one halts a running approach,
    # the other just reads a flag. Substring matching means excluding them has
    # to happen before the rule below, not inside it.
    (("stopautoapproach", "autoapproachstatus"), ""),
    # Full skill names, NOT a bare "approach" substring: the bare word would
    # swallow any future skill that merely mentions approaching.
    (("approachtip", "autoapproach"), "approach"),
    (("scan",), "scan"),
    (("folme", "xypos", "moveto", "movetip", "gotoposition", "pointshoot"), "move"),
]


@dataclass
class MapMarker:
    """One spatial event on the experiment map (METRES, stage frame)."""
    kind: str
    x_m: float | None = None
    y_m: float | None = None
    w_m: float | None = None          # footprint width (scan / frame markers)
    h_m: float | None = None
    angle_deg: float = 0.0
    label: str = ""
    skill_name: str = ""
    status: str = "done"              # done | failed | active | planned
    source: str = "skill"             # skill | manual | plan | live | import
    timestamp: str = ""
    meta: dict = field(default_factory=dict)
    # Coordinate-system generation this marker's xy belongs to. None on markers
    # that were never persisted (live frame / tip / plan overlay) and on legacy
    # rows, where it reads as 0 — see ``map_markers.coord_epoch``.
    coord_epoch: int | None = None

    @property
    def has_xy(self) -> bool:
        return self.x_m is not None and self.y_m is not None

    @property
    def has_footprint(self) -> bool:
        return (self.has_xy and self.w_m is not None and self.h_m is not None
                and self.w_m > 0 and self.h_m > 0)

# 只读和纯分析动词不应产生表面事件标记；拟合、预测和评估不能冒充采集。
# 按动词统一排除，避免新分析技能被名称中的 spectr 等子串误归类。
#
# stop/configure/set/wait/save 不在此排除表中：StopScan 等动作可承担扫描落图的
# 记录边界。漏记表面动作会使后续选点复用已改变的位置，因此只排除确定无表面
# 事件的类别；其他情况仍由具体技能分类规则判断。
_READONLY_PREFIXES = ("get", "list",
                      "predict", "unmix", "fit", "plot", "analyze", "compute",
                      "estimate", "draw", "erase", "assess")

#: 逐个点名的只读/无关技能：动词前缀盖不住它们。
_NEVER_POSITIONED = (
    "selfcheck",          # *SelfCheck 一族：只读自检，什么都不做
    "scanbackground",     # 背景扣除的复制粘贴：图像运算，不动仪器
    "autocrop", "diffscans",                # 离线图像处理
    # 「li-sts-…」：ListSignalChannels 里含子串 "sts"，会被记成一个谱学测量点。
    # 它只是把信号通道列出来。（同形的 list_fetch_requests 由 snake_case 规则覆盖。）
    "listsignalchannels",
    # 光学台（TERS）是另一台仪器 —— 它的动作不该出现在 STM 的表面地图上。
    "opticalstage", "probescanner", "pumpprobe", "probebias", "delayline",
    "digitalline",
)


def _never_positions(raw: str, n: str) -> bool:
    """判断技能是否确定不产生 STM 表面事件。
    
    按只读/分析动词、SelfCheck、桥接分析工具命名和光学台类别排除，
    避免名称子串把读取或分析误认为采集。纯自检不能生成扫描或修针标记，
    否则会污染清洁位置选择和覆盖率；具体动词范围以 _READONLY_PREFIXES 为准。"""
    if any(n.startswith(p) for p in _READONLY_PREFIXES):
        return True
    if "_" in raw and raw == raw.lower():
        return True                      # snake_case = 桥接进来的 agent 工具
    return any(tok in n for tok in _NEVER_POSITIONED)


#: ``SkillCategory.ANALYSIS`` 的字符串值（``core/types.py``:「Data processing, no
#: hardware interaction」）。分析类技能按**定义**不碰仪器，所以它在表面地图上留一个
#: 标记**永远**是 bug —— 这是唯一一条不靠名字、不会被下一个新技能绕过的判据。
_ANALYSIS_CATEGORY = "analysis"


def _is_analysis(category: Any) -> bool:
    """这个 category 是不是 ANALYSIS。接受 ``SkillCategory`` 枚举或它的字符串值。

    比字符串而不是 ``isinstance(category, SkillCategory)``：本模块刻意不依赖
    ``core.types``（文件头的「Pure / offline」），而两种形态在调用方都真实存在
    （registry 交出枚举，spec / JSON 一路交出字符串）。
    """
    v = getattr(category, "value", category)
    return isinstance(v, str) and v.strip().lower() == _ANALYSIS_CATEGORY


def classify_skill(skill_name: str, category: Any = None) -> str | None:
    """Map a skill name to a positioned-marker kind, or None if not positioned.

    ``category`` 是注册表里那个技能的 ``SkillMetadata.category``（枚举或字符串），
    由调用方顺手查出来传进来；``ANALYSIS`` 一律不落标记。**不传也是合法的**，此时
    只按名字判 —— 名字规则是全部既有行为，类别只做减法，从不新增标记。这条结构性
    排除来自 S4 STS 设计 §1.4a/O7：名字规则靠子串匹配，实测抓到三个纯分析技能
    （批处理整个文件夹的 .sxm、读缓冲区判撞针、从保存的 .sxm 抽通道）各自往地图里
    画了一个不存在的扫描足迹，而它们连自己的 description 都写着「不碰硬件」。

    返回值只有两种：一个非空 kind，或 None。**不返回 ""** —— 表里的 ``""`` 条目
    （PreScanCheck 一族）在 ``return kind or None`` 那里就被折叠成 None 了。此前这
    段 docstring 说调用方可以靠 ``""`` 区分「skip」与「unknown」，那是假的：这个
    函数产生不出 ``""``，全部四个调用方（``marker_from_skill`` 的 ``if not kind``、
    ``runtime`` 的两处 ``is None`` 和一处 ``or "scan"``）也没有一个依赖这个区分。
    """
    if not skill_name:
        return None
    if _is_analysis(category):
        return None
    raw = skill_name.strip()
    n = raw.lower()
    if _never_positions(raw, n):
        return None
    for needles, kind in _SKILL_KIND_RULES:
        if any(tok in n for tok in needles):
            return kind or None
    return None


def can_be_positioned(skill_name: str, category: Any = None) -> bool:
    """这个技能**有没有可能**在表面上留下事件。

    与 :func:`classify_skill` 的区别只有一条,但那一条是承重的:``classify_skill``
    对「**不可能**留下事件」(只读动词、``*SelfCheck``、snake_case 的 agent 工具、
    光学台一族、ANALYSIS 类)和「名字规则**认不出**」都返回 ``None`` —— 两件完全
    不同的事被折叠成了同一个值。

    调用方(``runtime._record_map_marker`` 的逐点定位记录那条准入)需要分开它们:
    一个结果里**逐点报了坐标**的新 composite,名字里没有 "scan"/"sts" 只是名字规则
    追不上它,那不是「它不碰表面」的证据;而一个 ``analyze_*`` / ``plot_*`` 报出来
    的坐标是从数据里读出来的,不是针尖去过的地方 —— 后者进地图就是假足迹
    (2026-08 抓到过三个纯分析技能各画了一个不存在的扫描足迹)。

    判据本身**不在这里重写一遍**,原样复用 ``classify_skill`` 用的那两个:
    这条规则只有一份实现,否则两份迟早会漂。
    """
    if not skill_name:
        return False
    if _is_analysis(category):
        return False
    raw = skill_name.strip()
    return not _never_positions(raw, raw.lower())


def _coerce_float(v: Any) -> float | None:
    try:
        if v is None or isinstance(v, bool):
            return None
        f = float(v)
    except (TypeError, ValueError):
        return None
    # Reject NaN / ±inf at the source — a non-finite coordinate poisons the
    # whole extent math and makes matplotlib's set_xlim raise ().
    return f if math.isfinite(f) else None


def _param_length_m(params: dict, *keys: str) -> float | None:
    """Read a length param in metres. A bare ``*_nm`` key is treated as nm and
    converted; everything else is assumed already in metres (the v2 unit)."""
    if not params:
        return None
    for k in keys:
        if k in params:
            val = _coerce_float(params[k])
            if val is None:
                continue
            return val * 1e-9 if k.endswith("_nm") else val
    return None


def marker_from_skill(
    skill_name: str,
    params: dict | None,
    state: "HardwareState | None",
    *,
    status: str = "done",
    source: str = "skill",
    data: dict | None = None,
    category: Any = None,
) -> MapMarker | None:
    """Build a MapMarker for a positioned skill, or None.

    Position priority: explicit x/y params → the skill's OWN measured readback
    in its result ``data`` → live snapshot tip xy. Scan footprints take
    centre/size from the live scan frame (the skill just set it) and fall back
    to width/height params. Skill-agnostic: needs no per-skill knowledge beyond
    the kind classification, so all 130+ builtins are covered without
    enumerating their parameter schemas.

    The ``data`` slot exists because the operations that leave a PERMANENT MARK
    on the surface — tip shaping, bias pulses — carry no position parameter at
    all: they act wherever the tip already is. Falling back to the cached
    snapshot put those marks up to a second of drift away from the truth, and
    the whole avoidance model is built on their coordinates. A skill that reads
    its own position at the moment it fires reports it here (``x_m``/``y_m`` in
    metres); one that cannot read it omits the keys rather than guessing.

    ``category`` 原样转交 ``classify_skill``（ANALYSIS ⇒ 不落标记）。"""
    kind = classify_skill(skill_name, category)
    if not kind:
        return None

    params = params or {}
    x = _param_length_m(params, "x_m", "x", "x_nm")
    y = _param_length_m(params, "y_m", "y", "y_nm")
    if (x is None or y is None) and isinstance(data, dict):
        # All-or-nothing: a position is a PAIR. Taking x from the skill's
        # readback and y from the stale snapshot would synthesise a coordinate
        # where nothing ever happened — worse than either source alone, and
        # ``_pos_provenance`` could only label the result "unknown".
        dx = _coerce_float(data.get("x_m"))
        dy = _coerce_float(data.get("y_m"))
        if dx is not None and dy is not None:
            x, y = dx, dy
    if x is None and state is not None:
        x = getattr(state, "x_pos_m", None)
    if y is None and state is not None:
        y = getattr(state, "y_pos_m", None)

    w = h = None
    angle = 0.0
    if kind == "scan":
        # The scan just configured/ran the frame → take it from the live frame.
        if state is not None and getattr(state, "scan_center_x_m", None) is not None:
            x = state.scan_center_x_m
            y = state.scan_center_y_m
            w = state.scan_width_m
            h = state.scan_height_m
            angle = getattr(state, "scan_angle_deg", None) or 0.0
        # Param fallback (offline / state-less): explicit size params.
        if w is None:
            w = _param_length_m(params, "width_m", "width", "width_nm", "size_m", "range_m")
        if h is None:
            h = _param_length_m(params, "height_m", "height", "height_nm", "size_m", "range_m")

    if x is None or y is None:
        # Nothing to place — a move with no readable position is not a marker.
        return None

    return MapMarker(
        kind=kind, x_m=x, y_m=y, w_m=w, h_m=h, angle_deg=angle or 0.0,
        skill_name=skill_name, status=status, source=source,
        label=KIND_STYLE.get(kind, ("", "", skill_name))[2],
    )


def frame_marker_from_state(state: "HardwareState | None") -> MapMarker | None:
    """The live current scan frame as a 'frame' marker, or None if unknown."""
    if state is None:
        return None
    cx = getattr(state, "scan_center_x_m", None)
    cy = getattr(state, "scan_center_y_m", None)
    w = getattr(state, "scan_width_m", None)
    h = getattr(state, "scan_height_m", None)
    if cx is None or cy is None or not w or not h:
        return None
    return MapMarker(
        kind="frame", x_m=cx, y_m=cy, w_m=w, h_m=h,
        angle_deg=getattr(state, "scan_angle_deg", None) or 0.0,
        source="live", label="当前扫描框",
    )


def tip_xy_from_state(state: "HardwareState | None") -> tuple[float, float] | None:
    """The live tip xy (FolMe position), or None."""
    if state is None:
        return None
    x = getattr(state, "x_pos_m", None)
    y = getattr(state, "y_pos_m", None)
    if x is None or y is None:
        return None
    return (float(x), float(y))


# ── Manual-activity detection (state diff + spectrum files) ─────────────────
#
# Manual Nanonis operations cover the SAME space as the agent's skills (the
# operator can scan / run STS / change bias / approach-retract / move the tip by
# hand). We capture every manual op that is *reliably observable*:
#   • sustained state changes (bias / setpoint / Z-controller status / scan
#     frame / tip / scan start) → polled from the cached HardwareState
#   • spectroscopy (point/grid STS) → from the saved .dat/.3ds file (its header
#     carries the exact xy), because there is no pollable spectroscopy status
# Genuine blind spot (documented): one-shot transients with NO status method and
# sub-poll duration — manual *bias pulses* and *tip-shaping* done directly in the
# Nanonis GUI (the same ops done THROUGH MAST are captured as skill markers).

_ZSTATUS_LABEL = {
    "On": "手动启用 Z 反馈",
    "Off": "手动关闭 Z 反馈",
    "Hold": "手动暂停 Z 反馈",
    "SafeTip": "安全收针 (SafeTip)",
    "Withdrawing": "手动收针",
    "SwitchingOff": "手动关闭 Z 反馈",
}


def snapshot_track(state: "HardwareState | None") -> dict:
    """Extract the manual-activity tracked fields from a HardwareState into a
    plain dict (the watcher's baseline/prev unit). NOISY fields (current_a,
    z_pos_m) are deliberately excluded — they fluctuate even when idle."""
    if state is None:
        return {"bias": None, "setpoint": None, "zstatus": None,
                "scanning": None, "frame": None, "tip": None}
    # Coerce every NUMERIC field through _coerce_float so a non-numeric hardware
    # value (a mock sentinel in tests, or a bad TCP parse in production) becomes
    # None instead of poisoning the downstream `abs(a-b) <= tol` diff with a
    # TypeError ("'<=' not supported between X and float") — that used to be
    # raised + logged every watcher tick (2026-07-01 scan-map rework).
    cx = _coerce_float(getattr(state, "scan_center_x_m", None))
    cy = _coerce_float(getattr(state, "scan_center_y_m", None))
    w = _coerce_float(getattr(state, "scan_width_m", None))
    h = _coerce_float(getattr(state, "scan_height_m", None))
    ang = _coerce_float(getattr(state, "scan_angle_deg", None)) or 0.0
    frame = (cx, cy, w, h, ang) if (cx is not None and w) else None
    tx = _coerce_float(getattr(state, "x_pos_m", None))
    ty = _coerce_float(getattr(state, "y_pos_m", None))
    tip = (tx, ty) if (tx is not None and ty is not None) else None
    zstatus = getattr(state, "z_controller_status", None)
    if zstatus is None and getattr(state, "z_controller_on", None) is not None:
        zstatus = "On" if state.z_controller_on else "Off"
    scanning = getattr(state, "scan_running", None)
    return {"bias": _coerce_float(getattr(state, "bias_v", None)),
            "setpoint": _coerce_float(getattr(state, "setpoint_a", None)),
            "zstatus": zstatus if isinstance(zstatus, str) else None,
            "scanning": bool(scanning) if isinstance(scanning, (bool, int)) else None,
            "frame": frame, "tip": tip}


def _approx(a, b, tol) -> bool:
    if a is None or b is None:
        return (a is None) and (b is None)
    return abs(a - b) <= tol


def _frame_changed(a, b, tol_m) -> bool:
    if a is None or b is None:
        return (a is None) != (b is None)
    # Per-element: a None on exactly one side is a real change (don't coerce
    # None→0.0, which would mask a 0.0↔None transition — review).
    for i in range(4):
        if a[i] is None or b[i] is None:
            if (a[i] is None) != (b[i] is None):
                return True
            continue
        if abs(a[i] - b[i]) > tol_m:
            return True
    return abs((a[4] or 0.0) - (b[4] or 0.0)) > 0.5  # angle, degrees


def _tip_moved(a, b, tol_m) -> bool:
    if a is None or b is None:
        return (a is None) != (b is None)
    return abs(a[0] - b[0]) > tol_m or abs(a[1] - b[1]) > tol_m


def detect_manual_state_changes(
    baseline: dict, cur: dict, prev: dict, *,
    tip_xy: tuple[float, float] | None = None,
    tol_m: float = 0.5e-9, bias_tol: float = 1e-3, setpoint_tol: float = 1e-11,
    suppress_scan_start: bool = False,
) -> tuple[list[MapMarker], dict]:
    """Diff the tracked state and emit manual-operation markers.

    *baseline* = the last state we LOGGED against; *cur* = this tick; *prev* =
    last tick (used to debounce parameter sweeps: a bias/setpoint value is only
    logged once it has SETTLED, i.e. cur == prev, so a manual STS bias sweep
    doesn't spam one marker per swing). Returns (markers, new_baseline) where
    new_baseline advances only the fields that were emitted/handled.

    *suppress_scan_start*: when True, a scanning rising edge advances the baseline
    but emits NO "手动开始扫描" marker. The caller sets this when it KNOWS the scan
    is system-initiated (a scan-vision monitor is live), which the skill-recency
    window alone can miss — a composite scan does long prep before ``Scan_Action``,
    so ``scan_running`` can flip True long after the scan skill's gate stamp
    (an agent scan was mislabelled 手动开始扫描).

    Pure + skill-agnostic. The CALLER is responsible for the rest of the
    suppression policy (skip when a skill ran recently, when no experiment is
    active, etc.)."""
    markers: list[MapMarker] = []
    nb = dict(baseline)

    def _param_xy():
        return (tip_xy if tip_xy is not None else (None, None))

    # ── Scan engine owns the rastering tip while running ──
    # A scan does NOT change the scan FRAME (it rasters within it), so we keep
    # the pre-scan frame baseline untouched — that way a manual frame reconfigure
    # during/right-after the scan is still detected (review: do not blanket-adopt
    # the frame on scan start/stop). We DO sync the tip (the raster sweeps it, so
    # the post-scan tip position is the scan's, not a manual move).
    if cur.get("scanning"):
        if (not baseline.get("scanning") and cur.get("frame")
                and not suppress_scan_start):
            fr = cur["frame"]
            markers.append(MapMarker(
                kind="manual", x_m=fr[0], y_m=fr[1], w_m=fr[2], h_m=fr[3],
                angle_deg=fr[4] or 0.0, label="手动开始扫描", source="manual"))
        nb["scanning"] = True
        nb["tip"] = cur.get("tip")
        return markers, nb
    nb["scanning"] = False
    if baseline.get("scanning"):
        # Scan just stopped. Sync the tip (raster end). The frame is compared
        # against the PRE-scan baseline below: a scan leaves the frame unchanged,
        # so any diff is a manual reconfigure and IS logged.
        nb["tip"] = cur.get("tip")
        if _frame_changed(baseline.get("frame"), cur.get("frame"), tol_m) \
                and cur.get("frame"):
            fr = cur["frame"]
            markers.append(MapMarker(
                kind="manual", x_m=fr[0], y_m=fr[1], w_m=fr[2], h_m=fr[3],
                angle_deg=fr[4] or 0.0, label="手动改扫描框", source="manual"))
            nb["frame"] = cur.get("frame")
        return markers, nb

    # ── Scan frame reconfigure (manual), else tip move ──
    if _frame_changed(baseline.get("frame"), cur.get("frame"), tol_m) and cur.get("frame"):
        fr = cur["frame"]
        markers.append(MapMarker(
            kind="manual", x_m=fr[0], y_m=fr[1], w_m=fr[2], h_m=fr[3],
            angle_deg=fr[4] or 0.0, label="手动改扫描框", source="manual"))
        nb["frame"] = cur.get("frame")
    elif _tip_moved(baseline.get("tip"), cur.get("tip"), tol_m) and cur.get("tip"):
        tp = cur["tip"]
        markers.append(MapMarker(kind="manual", x_m=tp[0], y_m=tp[1],
                                 label="手动移动针尖", source="manual"))
        nb["tip"] = cur.get("tip")

    # ── Bias change (debounced: only when settled) ──
    if (not _approx(cur.get("bias"), baseline.get("bias"), bias_tol)
            and _approx(cur.get("bias"), prev.get("bias"), bias_tol)):
        v = cur.get("bias")
        px, py = _param_xy()
        markers.append(MapMarker(
            kind="manual", x_m=px, y_m=py,
            label=(f"手动改偏压 {v:.3f} V" if v is not None else "手动改偏压"),
            source="manual", meta={"bias_v": v}))
        nb["bias"] = cur.get("bias")

    # ── Setpoint change (debounced) ──
    if (not _approx(cur.get("setpoint"), baseline.get("setpoint"), setpoint_tol)
            and _approx(cur.get("setpoint"), prev.get("setpoint"), setpoint_tol)):
        s = cur.get("setpoint")
        px, py = _param_xy()
        markers.append(MapMarker(
            kind="manual", x_m=px, y_m=py,
            label=(f"手动改设定点 {s * 1e12:.1f} pA" if s is not None else "手动改设定点"),
            source="manual", meta={"setpoint_a": s}))
        nb["setpoint"] = cur.get("setpoint")

    # ── Z-controller status change (On/Off/Hold/Withdraw/SafeTip) ──
    if cur.get("zstatus") != baseline.get("zstatus") and cur.get("zstatus") is not None:
        px, py = _param_xy()
        lbl = _ZSTATUS_LABEL.get(cur["zstatus"], f"手动 Z 状态→{cur['zstatus']}")
        markers.append(MapMarker(kind="manual", x_m=px, y_m=py, label=lbl,
                                 source="manual", meta={"zstatus": cur["zstatus"]}))
        nb["zstatus"] = cur.get("zstatus")

    return markers, nb


def extract_dat_position(path) -> tuple[float, float] | None:
    """xy (metres) from a Nanonis .dat point-spectroscopy header, or None."""
    try:
        from mast.io.nanonis_files import read_dat
        h = (read_dat(str(path)) or {}).get("header", {}) or {}
    except Exception as exc:  # noqa: BLE001
        logger.debug("read_dat(%s) failed: %s", path, exc)
        return None

    def _get(*keys):
        for k in keys:
            if k in h:
                try:
                    f = float(h[k])
                    return f if math.isfinite(f) else None
                except (TypeError, ValueError):
                    continue
        return None

    x = _get("X (m)", "x (m)", "X(m)", "X")
    y = _get("Y (m)", "y (m)", "Y(m)", "Y")
    if x is None or y is None or not _plausible_xy(x, y):
        return None
    return (x, y)


def _plausible_xy(x: float, y: float) -> bool:
    """Guard against a mis-parsed header column landing a marker light-years away:
    a real Nanonis stage coordinate is within ±1 mm (the piezo+coarse range)."""
    return (math.isfinite(x) and math.isfinite(y)
            and abs(x) < 1e-3 and abs(y) < 1e-3)


def extract_3ds_bbox(path) -> tuple[float, float, float, float] | None:
    """(cx, cy, w, h) metres covering a .3ds grid's per-pixel xy, or None."""
    try:
        from mast.io.nanonis_files import read_3ds
        res = read_3ds(str(path)) or {}
    except Exception as exc:  # noqa: BLE001
        logger.debug("read_3ds(%s) failed: %s", path, exc)
        return None
    params = res.get("params", {}) or {}
    arr = params.get("param_array")
    if arr is None:
        return None
    names = list(params.get("fixed_param_names") or []) + \
        list(params.get("experiment_param_names") or [])
    try:
        ix = names.index("X (m)")
        iy = names.index("Y (m)")
    except ValueError:
        return None
    try:
        xs = np.asarray(arr)[..., ix].ravel()
        ys = np.asarray(arr)[..., iy].ravel()
        xs = xs[np.isfinite(xs)]
        ys = ys[np.isfinite(ys)]
        if xs.size == 0 or ys.size == 0:
            return None
        xmin, xmax = float(xs.min()), float(xs.max())
        ymin, ymax = float(ys.min()), float(ys.max())
    except (IndexError, ValueError):
        return None
    cx, cy = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
    if not _plausible_xy(cx, cy):   # mis-parsed param column → reject
        return None
    return (cx, cy, max(xmax - xmin, 1e-12), max(ymax - ymin, 1e-12))


def manual_marker_from_spectrum_file(path) -> MapMarker | None:
    """Build a manual spectroscopy marker from a saved .dat / .3ds file.

    The saved file is the reliable record of a manual STS/grid (there is no
    pollable spectroscopy status), and its header carries the exact stage xy."""
    pl = str(path).lower()
    if pl.endswith(".3ds"):
        bbox = extract_3ds_bbox(path)
        if bbox is None:
            return None
        return MapMarker(kind="manual", x_m=bbox[0], y_m=bbox[1],
                         w_m=bbox[2], h_m=bbox[3], label="手动跑网格谱",
                         source="manual", meta={"file": str(path)})
    if pl.endswith(".dat"):
        pos = extract_dat_position(path)
        if pos is None:
            return None
        return MapMarker(kind="manual", x_m=pos[0], y_m=pos[1],
                         label="手动跑点谱 (STS)", source="manual",
                         meta={"file": str(path)})
    return None


def marker_from_row(row: dict) -> MapMarker:
    """Re-hydrate a MapMarker from an ``ExperimentStorage.get_markers`` row."""
    return MapMarker(
        kind=row.get("kind") or "move",
        x_m=row.get("x_m"), y_m=row.get("y_m"),
        w_m=row.get("w_m"), h_m=row.get("h_m"),
        angle_deg=row.get("angle_deg") or 0.0,
        label=row.get("label") or "",
        skill_name=row.get("skill_name") or "",
        status=row.get("status") or "done",
        source=row.get("source") or "skill",
        timestamp=row.get("timestamp") or "",
        meta=row.get("meta") if isinstance(row.get("meta"), dict) else {},
        coord_epoch=epoch_of_row(row),
    )


def epoch_of_row(row: dict) -> int:
    """A marker row's coordinate generation. NULL — a legacy row written before
    the column existed — reads as generation 0, because a database with no
    ``coarse_move`` rows has only ever had one coordinate system."""
    v = _coerce_float((row or {}).get("coord_epoch"))
    return int(v) if v is not None else 0


def markers_from_rows(rows: list[dict]) -> list[MapMarker]:
    return [marker_from_row(r) for r in (rows or [])]


# ── Extent ──────────────────────────────────────────────────────────────────

def _footprint_corners(m: MapMarker) -> list[tuple[float, float]]:
    if not m.has_footprint:
        return [(m.x_m, m.y_m)] if m.has_xy else []
    hw, hh = m.w_m / 2.0, m.h_m / 2.0
    return [(m.x_m - hw, m.y_m - hh), (m.x_m + hw, m.y_m + hh)]


def compute_extent(
    markers: list[MapMarker],
    *,
    frame: MapMarker | None = None,
    tip: tuple[float, float] | None = None,
    plan: list[MapMarker] | None = None,
    pad_frac: float = 0.12,
) -> tuple[float, float, float, float] | None:
    """Bounding box (x_min, x_max, y_min, y_max) in METRES over every element,
    padded. Returns None if there is nothing to show."""
    xs: list[float] = []
    ys: list[float] = []
    for m in list(markers) + list(plan or []):
        for (px, py) in _footprint_corners(m):
            if px is not None and py is not None:
                xs.append(px)
                ys.append(py)
    if frame is not None:
        for (px, py) in _footprint_corners(frame):
            xs.append(px)
            ys.append(py)
    if tip is not None:
        xs.append(tip[0])
        ys.append(tip[1])
    # Drop any non-finite coordinate (inf/nan from an error state or a bad TCP
    # parse) — one would make min/max/span NaN and crash set_xlim ().
    xs = [x for x in xs if x is not None and math.isfinite(x)]
    ys = [y for y in ys if y is not None and math.isfinite(y)]
    if not xs or not ys:
        return None
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    span_x = x_max - x_min
    span_y = y_max - y_min
    # Degenerate (single point / zero span) → give it a small FIXED window so the
    # marker isn't a dot in an infinite plane. Use a fixed ~100 nm (a typical STM
    # frame) rather than the coordinate magnitude — a lone marker at 500 µm must
    # not zoom out to a 500 µm window ().
    if span_x <= 0 and span_y <= 0:
        pad = 5e-8  # → 100 nm window
        return (x_min - pad, x_max + pad, y_min - pad, y_max + pad)
    span = max(span_x, span_y, 1e-12)
    pad = span * pad_frac
    # Keep the box square-ish so equal aspect doesn't crush one axis.
    cx, cy = (x_min + x_max) / 2.0, (y_min + y_max) / 2.0
    half = span / 2.0 + pad
    return (cx - half, cx + half, cy - half, cy + half)


# ── Render ──────────────────────────────────────────────────────────────────

_CJK_FP: Any = None  # FontProperties | False (resolved once)


def _cjk_fontprops():
    """A FontProperties for an available CJK font, applied ONLY to the map's
    Chinese text (title / legend / empty-state), or None if none is installed.

    Surgical on purpose: we do NOT mutate global ``rcParams['font.sans-serif']``.
    That would switch the default font for every other matplotlib figure in the
    process (scan_preview, mosaic, tip-shape) — and a CJK font lacks glyphs those
    English figures use (e.g. the ⟨⟩ angle brackets in scan_preview), so they'd
    regress to tofu. A per-artist FontProperties is baked into the Text at
    creation, so it survives the frontend's deferred draw. Cached + fail-safe."""
    global _CJK_FP
    if _CJK_FP is not None:
        return _CJK_FP or None
    _CJK_FP = False
    try:
        from matplotlib import font_manager as fm
        avail = {f.name for f in fm.fontManager.ttflist}
        for cand in ("Microsoft YaHei", "SimHei", "Microsoft JhengHei",
                     "Noto Sans CJK SC", "Source Han Sans SC", "Arial Unicode MS",
                     "DengXian", "SimSun"):
            if cand in avail:
                _CJK_FP = fm.FontProperties(family=cand)
                break
    except Exception:  # noqa: BLE001 — never break rendering over fonts
        pass
    return _CJK_FP or None


def _add_footprint(ax, m: MapMarker, *, edgecolor, facecolor, lw, ls, alpha,
                   zorder, label=None):
    from matplotlib.patches import Rectangle

    w_nm = m.w_m * 1e9
    h_nm = m.h_m * 1e9
    x0 = m.x_m * 1e9 - w_nm / 2.0
    y0 = m.y_m * 1e9 - h_nm / 2.0
    kw = dict(linewidth=lw, edgecolor=edgecolor, facecolor=facecolor,
              alpha=alpha, zorder=zorder, linestyle=ls, label=label)
    if abs(m.angle_deg) > 0.5:
        try:
            rect = Rectangle((x0, y0), w_nm, h_nm, angle=m.angle_deg,
                             rotation_point="center", **kw)
        except (TypeError, ValueError):  # old matplotlib: no rotation_point
            rect = Rectangle((x0, y0), w_nm, h_nm, **kw)
    else:
        rect = Rectangle((x0, y0), w_nm, h_nm, **kw)
    ax.add_patch(rect)


def render_map_figure(
    markers: list[MapMarker],
    *,
    frame: MapMarker | None = None,
    tip: tuple[float, float] | None = None,
    plan: list[MapMarker] | None = None,
    sample_label: str = "",
    title: str = "",
) -> "Figure":
    """matplotlib Figure of the experiment map (axes in nm, Agg, no pyplot).

    Draws, back-to-front: scan footprints (filled), point operations (scatter by
    kind), the planned route (dashed polyline + ghost markers, car-navigation
    style), the live current scan frame (bright outline), and the live tip
    (crosshair). An empty map renders an honest waiting-state instead of a blank
    canvas."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fp = _cjk_fontprops()
    _fp_kw = {"fontproperties": fp} if fp is not None else {}
    markers = list(markers or [])
    plan = list(plan or [])

    fig = Figure(figsize=(7.4, 7.0), dpi=110)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ax.set_facecolor("#0b1017")
    fig.patch.set_facecolor("#0b1017")
    for spine in ax.spines.values():
        spine.set_color("#334155")
    ax.tick_params(colors="#94a3b8", labelsize=8)
    ax.xaxis.label.set_color("#cbd5e1")
    ax.yaxis.label.set_color("#cbd5e1")

    extent = compute_extent(markers, frame=frame, tip=tip, plan=plan)
    if extent is None:
        ax.text(0.5, 0.5,
                "等待扫描范围…\n\n开始扫描或读取仪器状态后，\n本图将显示扫描框与所有带位置的操作。",
                ha="center", va="center", color="#64748b", fontsize=11,
                transform=ax.transAxes, **_fp_kw)
        ax.set_axis_off()
        return fig

    x_min, x_max, y_min, y_max = extent
    ax.set_xlim(x_min * 1e9, x_max * 1e9)
    ax.set_ylim(y_min * 1e9, y_max * 1e9)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (nm)")
    ax.set_ylabel("Y (nm)")
    ax.grid(True, color="#1e293b", linewidth=0.6, zorder=0)

    # 1) scan footprints (history) — filled, low alpha, behind points.
    for m in markers:
        if m.kind == "scan" and m.has_footprint:
            failed = m.status == "failed"
            _add_footprint(
                ax, m,
                edgecolor=("#64748b" if failed else "#3b82f6"),
                facecolor=("#1e293b" if failed else "#1d4ed8"),
                lw=1.0, ls=":" if failed else "-", alpha=0.22, zorder=2)

    # 2) planned route — dashed polyline through plan markers in order, with
    #    ghost markers. The first pending step is highlighted as "下一步".
    if plan:
        pts = [(m.x_m * 1e9, m.y_m * 1e9) for m in plan if m.has_xy]
        if len(pts) >= 2:
            ax.plot([p[0] for p in pts], [p[1] for p in pts],
                    color="#38bdf8", lw=1.4, ls="--", alpha=0.7, zorder=4,
                    label="计划路线")
        for i, m in enumerate(plan):
            if not m.has_xy:
                continue
            color, marker, _ = KIND_STYLE.get(m.kind, KIND_STYLE["plan"])
            is_next = i == 0
            if m.has_footprint:
                _add_footprint(ax, m, edgecolor="#38bdf8", facecolor="none",
                               lw=1.4, ls="--", alpha=0.85, zorder=5)
            ax.scatter([m.x_m * 1e9], [m.y_m * 1e9],
                       s=(150 if is_next else 90), c="none",
                       edgecolors="#38bdf8", linewidths=1.6,
                       marker=marker, zorder=6, alpha=0.95)
            tag = f"{i + 1}"
            ax.annotate(tag, (m.x_m * 1e9, m.y_m * 1e9),
                        color="#7dd3fc", fontsize=8, ha="center", va="center",
                        zorder=7)

    # 3) point operations (history) — scatter, grouped per kind for one legend
    #    entry each.
    seen_legend: set[str] = set()
    for kind in ("sts", "pulse", "tip_shape", "move", "manual"):
        color, marker, zh = KIND_STYLE[kind]
        pts_x = [m.x_m * 1e9 for m in markers if m.kind == kind and m.has_xy]
        pts_y = [m.y_m * 1e9 for m in markers if m.kind == kind and m.has_xy]
        if not pts_x:
            continue
        # Manual frame-changes also draw a faint dashed rect footprint.
        if kind == "manual":
            for m in markers:
                if m.kind == "manual" and m.has_footprint:
                    _add_footprint(ax, m, edgecolor="#e879f9", facecolor="none",
                                   lw=0.9, ls="--", alpha=0.5, zorder=3)
        ax.scatter(pts_x, pts_y, s=(60 if kind != "move" else 26),
                   c=color, marker=marker, edgecolors="#0b1017",
                   linewidths=0.5, zorder=8, alpha=0.92,
                   label=(zh if kind not in seen_legend else None))
        seen_legend.add(kind)

    # 4) live current scan frame — bright cyan outline, no fill ("你在这").
    if frame is not None and frame.has_footprint:
        _add_footprint(ax, frame, edgecolor="#06b6d4", facecolor="none",
                       lw=2.0, ls="-", alpha=1.0, zorder=9, label="当前扫描框")

    # 5) live tip — crosshair + dot.
    if tip is not None:
        tx, ty = tip[0] * 1e9, tip[1] * 1e9
        ax.axhline(ty, color="#f43f5e", lw=0.6, alpha=0.35, zorder=9)
        ax.axvline(tx, color="#f43f5e", lw=0.6, alpha=0.35, zorder=9)
        ax.scatter([tx], [ty], s=70, c="#f43f5e", marker="P",
                   edgecolors="white", linewidths=0.6, zorder=11, label="针尖")

    n_ops = sum(1 for m in markers if m.kind != "frame")
    ttl = title or (f"扫描地图 · {sample_label}" if sample_label else "扫描地图")
    ax.set_title(f"{ttl}   ({n_ops} 个操作)", color="#e2e8f0", fontsize=11,
                 **_fp_kw)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        leg_kw = {"fontsize": 7.5}
        if fp is not None:
            _leg_prop = fp.copy()
            _leg_prop.set_size(7.5)  # prop overrides fontsize, so bake size in
            leg_kw = {"prop": _leg_prop}
        leg = ax.legend(loc="upper right", framealpha=0.82,
                        facecolor="#111827", edgecolor="#334155", **leg_kw)
        for txt in leg.get_texts():
            txt.set_color("#cbd5e1")

    fig.tight_layout()
    return fig


def save_map_png(fig: "Figure", *, label: str = "", now=None) -> str:
    """Write a rendered map figure to artifacts/maps/ and return the PNG path."""
    from datetime import datetime

    from mast._runtime_paths import project_root

    now = now or datetime.now()
    stamp = now.strftime("%Y%m%dT%H%M%S")
    safe = "".join(c for c in (label or "") if c.isalnum() or c in "-_") or "map"
    d = project_root() / "artifacts" / "maps"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"scanmap_{safe}_{stamp}.png"
    fig.savefig(str(path), dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    return str(path)


def summarize_markers(markers: list[MapMarker]) -> str:
    """One-line Markdown summary of marker counts per kind (for status text)."""
    counts: dict[str, int] = {}
    for m in markers:
        if m.kind == "frame":
            continue
        counts[m.kind] = counts.get(m.kind, 0) + 1
    if not counts:
        return "尚无带位置的操作记录。"
    parts = []
    for kind, n in counts.items():
        zh = KIND_STYLE.get(kind, (None, None, kind))[2]
        parts.append(f"{zh} ×{n}")
    return "已记录 " + "，".join(parts) + "。"


__all__ = [
    "MapMarker", "KIND_STYLE", "classify_skill", "can_be_positioned",
    "marker_from_skill",
    "frame_marker_from_state", "tip_xy_from_state", "marker_from_row",
    "markers_from_rows", "compute_extent", "render_map_figure",
    "save_map_png", "summarize_markers",
    "snapshot_track", "detect_manual_state_changes",
    "extract_dat_position", "extract_3ds_bbox", "manual_marker_from_spectrum_file",
]
