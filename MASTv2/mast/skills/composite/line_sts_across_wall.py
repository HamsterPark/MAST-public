"""LineSTSAcrossWall：计算跨畴界线谱坐标并调用逐点取谱引擎。

几何计算由 mast.core.sts_line_plan 承担。引用对象需匹配 id、epoch 与已确认的坐标基准，
信息不确定时拒绝执行。条纹方向与法向分别处理。
预计耗时使用可配置速度；未配置的名义值只供计划估算，不能作为仪器读回。
采集仍通过技能注册表执行，以保留安全检查、人工确认和进度上报。
"""

from __future__ import annotations

import json
import logging
from typing import Any

from mast.agents._shared.skill_adapter import wrap_skill
from mast.core.sts_line_plan import (
    CROSS_SITE_HINT,
    LinePlanRefused,
    estimate_line_duration,
    format_duration_note,
    looks_like_cross_site,
    plan_from_geometry,
    resolve_line_geometry,
    resolve_line_spec,
)
from mast.core.sts_workflow import DEFAULT_CONDITION, resolve_condition
from mast.core.types import (
    ParameterSpec,
    SafetyLevel,
    SkillCategory,
    SkillMetadata,
    SkillResult,
)
from mast.skills.composite._base import CompositeSkillGraph
from mast.skills.composite.graph_executor import CompositeProgress, CompositeStep

logger = logging.getLogger(__name__)

#: 逐点取谱引擎的技能名。**按名字引用,不 import 它的类**(见模块注释)。
POINT_ENGINE_SKILL = "SpectroscopyAtPositions"

#: 本壳自己的拒绝码(几何层的另有一套,见 ``sts_line_plan.REFUSAL_CODES``)。
#: 拒绝的**理由**要能被程序分辨 —— 文案会改,码不该跟着改。
SHELL_REFUSAL_CODES: tuple[str, ...] = (
    "no_input",                  # 既没有 id 也没有显式几何
    "markers_unavailable",       # 读不到地图记录 ——「读不到」不是「一致」
    "coord_epoch_mismatch",
    # 2026-08-25 补登记：:352 与 :415 两处都在 `_refuse` 它，而这张表里一直没有。
    # 漏登记的后果不是报错，是**按这张表分派的下游会漏掉这一路** —— 而它的下一步
    # （先做半步长粗动把站点插进去）与其它拒绝码完全不同。
    "cross_site_no_metres",      # 只有粗动步数、没有米坐标
    "search_not_found",
    "bracket_not_found",
    "bracket_ambiguous",         # 同一 role 多条且无法定序 —— 不猜
    "bracket_verdicts_agree",    # 两端判定相同 ⇒ 中间没有界
    "verdict_undetermined",
    "mixed_frame_not_found",
    "mixed_frame_ambiguous",
    "reference_version_unknown",
    "reference_not_found",
    "reference_not_confirmed",
    "reference_version_conflict",
)

#: 原样转发给逐点引擎的采谱条件参数名。**名字必须与引擎的参数表逐字一致** ——
#: 引擎不认得的键会被安静地丢掉,于是「我明明设了稳定偏压」和「它根本没收到」
#: 长得一模一样。
#:
#: 2026-08-15:条件表(``core.sts_workflow.STSConditionSpec``)落地,这一串里的
#: 六个显式数值(``stab_bias_v`` / ``stab_setpoint_a`` / ``sweep_start_v`` /
#: ``sweep_end_v`` / ``num_points`` / ``lockin_preset``)**换成了一个组名**
#: ``condition``。两边同时换的理由不是整洁:一条线谱是几十个点、一晚上,而那六个
#: 数每一个都会变成真实的硬件动作 —— 把它们摆成工具表里的空格子,等于请调用方
#: (经常是语言模型)填数。**移除诱因,别在提示词里说服模型**(本仓已记四次)。
#:
#: ⚠️ 这个 tuple 是**结构闸门**:多一个引擎不认识的名字,后果不是报错,是那一项
#: 静默失效。对应的测试断言它逐项都在引擎的参数表里。
_ENGINE_PASSTHROUGH: tuple[str, ...] = (
    "condition", "settle_s",
    "spectral_family", "assess",
    "stop_after_consecutive_discard", "per_point_timeout_s", "run_tag",
)

#: marker meta 里表示「这一帧里两个畴都有」的判定值(上游 §3.1 的闭集成员)。
_VERDICT_MIXED = "mixed"
#: 「判不了」。**永远不折叠成一个 label** —— 拒绝跑,不去赌。
_VERDICT_UNDETERMINED = "undetermined"


def _meta(marker) -> dict:
    m = getattr(marker, "meta", None)
    return m if isinstance(m, dict) else {}


def _latest(markers: list) -> "tuple[Any, bool]":
    """一组同 role 的标记里取**最后一次**的那条,以及「定得出序吗」。

    二分每走一步就把 lo 或 hi 换掉,所以当前 bracket = 各自最新的那一条。定不出
    序(多条且时间戳缺失/重复)时返回 ``ordered=False`` —— 由调用方拒绝,而不是
    在这里随手挑一个。挑错一端会让整条线偏到畴内,而且不会有任何东西报警。
    """
    if len(markers) == 1:
        return markers[0], True
    stamps = [str(getattr(m, "timestamp", "") or "") for m in markers]
    if any(not s for s in stamps) or len(set(stamps)) != len(stamps):
        return None, False
    return max(markers, key=lambda m: str(getattr(m, "timestamp", ""))), True


class LineSTSAcrossWall(CompositeSkillGraph):
    """沿 bracket 轴跨过畴界布一条线谱,近墙密、远墙疏,按 ``|s|`` 升序采。"""

    #: 这四样在 :meth:`run_composite` 里算好后才有值。给类级默认是为了让
    #: ``aggregate`` 在任何被提前调用的路径上也不会因 AttributeError 崩掉 ——
    #: 崩在汇总里会把一次真实的失败原因换成一句无关的 traceback。
    _plan_out = None
    _budget_out: dict = {}
    _evidence_out: dict = {}
    _positions_payload: list = []

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="LineSTSAcrossWall",
            version="1.0.0",
            category=SkillCategory.COMPOSITE,
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "沿一条线**跨过**一道畴界采一串 STS 谱 —— 这道畴界是此前一次畴界"
                "搜索用 bracket 夹出来的。用 id 指向那次搜索 —— `domain_search_id` "
                "加 `bracket_id` —— 它自己会从实验地图里把 bracket 的两个端点读出来;"
                "**不要**递给它猜出来的坐标。点位近墙密、远墙疏,而且按由近及远的顺序"
                "采,让 piezo drift 损坏的是最不重要的(远处那些)点。"
                "精细采样窗**永远不会窄于**这次搜索自身的位置不确定度 —— 窗更窄的话,"
                "有可能把每一个密集点都摆在畴界的**同一侧**,于是跑出一整晚好看却什么"
                "都回答不了的谱。"
                "下列情况一律拒绝:坐标代次变了、bracket 的判定是 undetermined、"
                "没有人确认过畴参照系、或者这道畴界只定位到了**粗动步数**"
                "(不存在米坐标 —— 必须先把它挪进单个站点之内)。置 `dry_run` 可以"
                "只拿到点位表与时间预估,不碰针尖。"
            ),
            parameters=[
                ParameterSpec(
                    name="domain_search_id", type="str",
                    description=("几何存在这次畴界搜索的标记里 —— 给它的 id。"
                                 "这是首选输入形态:说一个 id,别说坐标。"),
                    required=False, default=""),
                ParameterSpec(
                    name="bracket_id", type="str",
                    description=("那次搜索下面的哪一个 bracket(它的 lo/hi 一对"
                                 "确定了原点与轴向)。**只有**在目标是一帧判定为 "
                                 "'mixed' 的帧时才可以省略。"),
                    required=False, default=""),
                ParameterSpec(
                    name="expected_coord_epoch", type="int",
                    description=("这批坐标属于哪一代坐标系。对不上、或者地图读"
                                 "不到,都拒绝 —— 横向粗动之后,同样的 (x,y) 指的"
                                 "是另一片表面。"),
                    required=True, min_value=0),
                ParameterSpec(
                    name="origin_x_m", type="float", unit="m",
                    description=("显式给出的线中心 X。只给用户 / conduct spec "
                                 "用;用 id 的时候留空。"),
                    required=False),
                ParameterSpec(
                    name="origin_y_m", type="float", unit="m",
                    description="显式给出的线中心 Y。", required=False),
                ParameterSpec(
                    name="axis_deg", type="float", unit="deg",
                    description=("显式给出的轴方位角,与 x/y 在**同一个**参照系里"
                                 "(+x = 0 deg)。目标是一帧 'mixed' 而又没有 "
                                 "bracket 时必填。这是**搜索**轴,不是畴界法向,"
                                 "而且绝不拿扫描角来代替它。"),
                    required=False, min_value=-360.0, max_value=360.0),
                ParameterSpec(
                    name="uncertainty_m", type="float", unit="m",
                    description=("畴界的位置沿轴向有多大的不确定度。用显式几何时"
                                 "**必填** —— 省掉它会把密集窗静默地缩回流程表里"
                                 "的缺省值。用 id 时它由 bracket 的跨度得出。"),
                    required=False, min_value=0.0),
                ParameterSpec(
                    name="fine_spacing_nm", type="float", unit="nm",
                    description=("密集窗**之内**的点间距。取自流程表 / conduct "
                                 "spec —— 语言模型不许自己编这个数。"),
                    required=False, min_value=0.01, max_value=100.0),
                ParameterSpec(
                    name="coarse_spacing_nm", type="float", unit="nm",
                    description="密集窗**之外**的点间距。",
                    required=False, min_value=0.01, max_value=1000.0),
                ParameterSpec(
                    name="fine_half_width_nm", type="float", unit="nm",
                    description=("密集半窗的**下限**。实际用的值是 max(这个数, "
                                 "搜索自身的位置不确定度)。"),
                    required=False, min_value=0.0, max_value=1000.0),
                ParameterSpec(
                    name="line_half_length_nm", type="float", unit="nm",
                    description="整条线的半长。",
                    required=False, min_value=0.01, max_value=5000.0),
                ParameterSpec(
                    name="max_points", type="int",
                    description="这条线的点数预算。",
                    required=False, min_value=1, max_value=400),
                ParameterSpec(
                    name="require_confirmed_reference", type="bool",
                    description=("除非有人确认过那些判定所依据的那一版畴参照系,"
                                 "否则拒绝执行。默认 true —— 一道没人确认过的"
                                 "「畴界」可能白白花掉一整晚。"),
                    required=False, default=True),
                # ── 采谱条件:一个**组名**,本壳一个数都不发明也不转发 ──
                # 2026-08-15 由六个显式数值参数换成组名(S4 STS 设计 D19)。
                # 一条线谱是几十个点、一晚上,那六个数每一个都会变成真实的硬件
                # 动作 —— 把它们摆成工具表里的空格子,等于请调用方(经常是语言
                # 模型)填数。数字住在 ``core.sts_workflow`` 的条件表里。
                ParameterSpec(
                    name="condition", type="str",
                    description=(
                        "线上每一个点是在哪一个采谱条件**组**下采的 —— 给**组名**,"
                        "**不是数字**。稳定 bias/setpoint、扫描窗口、点数与调制"
                        "全都住在那个组里(core.sts_workflow)。名字不认识、"
                        "覆写值越界、或者组里样品相关的字段还没标定 —— 这三种"
                        "逐点引擎都会**拒绝**,并指名是哪里不对。"),
                    required=False, default=DEFAULT_CONDITION),
                ParameterSpec(
                    name="settle_s", type="float", unit="s",
                    description=("**只**覆写那个组的整定时间,单位秒。它是一个"
                                 "时间旋钮,不是一个物理设定值,而且它还要喂给"
                                 "时间预估。省略就用组自己的值。"),
                    required=False, min_value=0.0, max_value=60.0),
                ParameterSpec(
                    name="spectral_family", type="str",
                    description=("哪些谱的子判据可以起闸。留空 = unknown,"
                                 "而 unknown **不是** metallic。"),
                    required=False, default="",
                    allowed_values=["", "metallic", "gapped", "unknown"]),
                ParameterSpec(
                    name="assess", type="bool",
                    description="逐点跑一遍谱质量闸。",
                    required=False, default=True),
                ParameterSpec(
                    name="stop_after_consecutive_discard", type="int",
                    description=("连续这么多次判成 'discard' 就停。0 = 永不停。"
                                 "'unrated' 不计入。"),
                    required=False, min_value=0, max_value=64),
                ParameterSpec(
                    name="per_point_timeout_s", type="float", unit="s",
                    description="每个点的时间上限,每采完一个点检查一次。",
                    required=False, min_value=1.0, max_value=7200.0),
                ParameterSpec(
                    name="run_tag", type="str",
                    description="逐点 .dat 文件基名的前缀。",
                    required=False, default=""),
                ParameterSpec(
                    name="per_point_acquire_s", type="float", unit="s",
                    description=("每条谱的**实测**秒数,只用于时间预估。留空表示"
                                 "**未知**,预估会如实说明,而不是悄悄按 0 算。"),
                    required=False, min_value=0.0, max_value=3600.0),
                ParameterSpec(
                    name="dry_run", type="bool",
                    description=("只算点位与时间预估,一条谱都不采。在把一整晚"
                                 "押给这条线之前先用它。"),
                    required=False, default=False),
            ],
            preconditions=["z_controller_on"],
            estimated_duration_s=3600.0,
            # 唯一一个把另一个 composite 当子步骤的谱学壳 —— 比 L3 高一层。
            composition_level=4,
            tags=["spectroscopy", "sts", "line", "domain", "composite"],
        )

    # ── 拒绝 ──────────────────────────────────────────────────────────────

    def _refuse(self, code: str, message: str, **extra) -> SkillResult:
        """一次开跑前的拒绝。**带码带下一步**,不是一句「失败」。"""
        return SkillResult(
            skill_name=self._skill_name(), success=False, error=message,
            data={"refusal_code": code, "acquired": False, "points": [], **extra},
            summary=message,
        )

    # ── 几何解析 ──────────────────────────────────────────────────────────

    def _geometry_from_params(self, params: dict):
        """显式几何(用户 / conduct spec)。给全了才算数,给一半 ⇒ ``None``。"""
        ox, oy = params.get("origin_x_m"), params.get("origin_y_m")
        if ox is None or oy is None:
            return None
        return resolve_line_geometry(
            "explicit", origin_xy=(ox, oy), axis_deg=params.get("axis_deg"),
            uncertainty_m=params.get("uncertainty_m"),
            source_note="显式几何(用户 / conduct spec)")

    def _geometry_from_map(self, params: dict) -> "tuple[Any, dict] | SkillResult":
        """按 ``(domain_search_id, bracket_id)`` 从地图标记里取几何。

        返回 ``(LineGeometry, 证据 dict)``,或者一个已经成形的拒绝 SkillResult。
        """
        from mast.core.map_scope import load_markers

        search_id = str(params.get("domain_search_id") or "").strip()
        bracket_id = str(params.get("bracket_id") or "").strip()

        markers, epoch, available = load_markers()
        if not available:
            # 读不到记录 ≠ 代次一致。这条闸的全部意义就是不让「不知道」放行。
            return self._refuse(
                "markers_unavailable",
                "读不到实验地图记录(没有活动实验或存储不可用),因此**无法确认**"
                "这批坐标属于哪一代坐标系。「读不到」不是「一致」⇒ 不跑。"
                "请先打开实验记录,或用显式几何参数并自行确认坐标代次。")

        expected = params.get("expected_coord_epoch")
        if expected is not None and int(expected) != int(epoch):
            return self._refuse(
                "coord_epoch_mismatch",
                f"这批坐标属于第 {int(expected)} 代坐标系,而当前是第 {int(epoch)} 代"
                "(中间发生过横向粗动)。同样的 (x, y) 现在指的是另一片表面 ⇒ 拒绝。"
                "跨代次换算在本仓有意不存在:请重新定位畴界。",
                expected_coord_epoch=int(expected), current_coord_epoch=int(epoch))

        mine = [m for m in markers if str(_meta(m).get("domain_search_id") or "") == search_id]
        if not mine:
            return self._refuse(
                "search_not_found",
                f"当前坐标代次里没有 domain_search_id={search_id!r} 的采样标记。"
                "请确认 id,或者先跑一次畴界搜索。")

        if bracket_id:
            return self._bracket_geometry(mine, search_id, bracket_id)
        return self._mixed_geometry(mine, search_id, params)

    def _bracket_geometry(self, mine: list, search_id: str, bracket_id: str):
        same = [m for m in mine if str(_meta(m).get("bracket_id") or "") == bracket_id]
        if not same:
            return self._refuse(
                "bracket_not_found",
                f"搜索 {search_id!r} 下没有 bracket_id={bracket_id!r} 的标记。")

        cross = [m for m in same if looks_like_cross_site(
            _meta(m), x_m=getattr(m, "x_m", None), y_m=getattr(m, "y_m", None))]
        if cross:
            return self._refuse(
                "cross_site_no_metres",
                "这条畴界只定位到粗动步数(站点之间 ±N 步),没有米坐标,线谱无从"
                f"布点。{CROSS_SITE_HINT}:在两站之间做一次半步长粗动把新站点插"
                "进去,再在单个站点内重新定位。不去就近编一个坐标。")

        ends: dict[str, Any] = {}
        for role in ("lo", "hi"):
            cand = [m for m in same if str(_meta(m).get("role") or "") == role]
            if not cand:
                return self._refuse(
                    "bracket_not_found",
                    f"bracket {bracket_id!r} 缺 {role} 端的标记 —— 二分没跑完,"
                    "或者标记没落库。")
            picked, ordered = _latest(cand)
            if not ordered:
                return self._refuse(
                    "bracket_ambiguous",
                    f"bracket {bracket_id!r} 的 {role} 端有 {len(cand)} 条标记,"
                    "而时间戳定不出先后 ⇒ 不知道哪一条是当前端点,**不猜**。"
                    "请指定一个更细的 bracket_id。")
            ends[role] = picked

        for role, m in ends.items():
            v = str(_meta(m).get("verdict") or "")
            if not v or v == _VERDICT_UNDETERMINED:
                return self._refuse(
                    "verdict_undetermined",
                    f"bracket {bracket_id!r} 的 {role} 端判定是 "
                    f"{v or '(空)'} —— 「判不了」不等于「在畴 X 里」。"
                    "先把这一端重扫/重判,再跑线谱。",
                    verdict_reason=_meta(m).get("verdict_reason"))
        if str(_meta(ends["lo"]).get("verdict")) == str(_meta(ends["hi"]).get("verdict")):
            return self._refuse(
                "bracket_verdicts_agree",
                f"bracket {bracket_id!r} 两端的判定相同"
                f"({_meta(ends['lo']).get('verdict')!r}),中间并没有确立一道畴界。"
                "在这条线上跑一晚只会得到一族一模一样的谱。")

        lo, hi = ends["lo"], ends["hi"]
        geom = resolve_line_geometry(
            "bracket", lo_xy=(lo.x_m, lo.y_m), hi_xy=(hi.x_m, hi.y_m),
            source_note=f"bracket {bracket_id} @ search {search_id}")
        return geom, self._evidence(search_id, bracket_id, [lo, hi], form="bracket")

    def _mixed_geometry(self, mine: list, search_id: str, params: dict):
        mixed = [m for m in mine
                 if str(_meta(m).get("verdict") or "") == _VERDICT_MIXED]
        if not mixed:
            return self._refuse(
                "mixed_frame_not_found",
                f"搜索 {search_id!r} 下没有判成 {_VERDICT_MIXED!r} 的帧,而 "
                "bracket_id 也没给。请给 bracket_id(一对 lo/hi)。")
        if len(mixed) > 1:
            where = ", ".join(f"({m.x_m}, {m.y_m})" for m in mixed[:5])
            return self._refuse(
                "mixed_frame_ambiguous",
                f"搜索 {search_id!r} 下有 {len(mixed)} 帧判成 {_VERDICT_MIXED!r}"
                f"({where}…) —— 不猜用哪一帧。请给 bracket_id,或用显式几何指定"
                "线的中心。")
        frame = mixed[0]
        if looks_like_cross_site(_meta(frame), x_m=getattr(frame, "x_m", None),
                                 y_m=getattr(frame, "y_m", None)):
            return self._refuse(
                "cross_site_no_metres",
                f"这一条结果没有米坐标,线谱无从布点。{CROSS_SITE_HINT}。")

        # 轴向:优先用同一次搜索里的 bracket;没有就要求显式给角度。
        # **绝不拿 scan_angle_deg 当轴角** —— 它读不到时是 null 而不是 0,而且
        # 帧角与搜索方向本来就是两件事。
        lo_xy = hi_xy = None
        lo = [m for m in mine if str(_meta(m).get("role") or "") == "lo"]
        hi = [m for m in mine if str(_meta(m).get("role") or "") == "hi"]
        if len(lo) == 1 and len(hi) == 1:
            lo_xy = (lo[0].x_m, lo[0].y_m)
            hi_xy = (hi[0].x_m, hi[0].y_m)
        geom = resolve_line_geometry(
            "mixed_frame", center_xy=(frame.x_m, frame.y_m),
            frame_size_m=getattr(frame, "w_m", None),
            lo_xy=lo_xy, hi_xy=hi_xy,
            axis_deg=params.get("axis_deg"),
            uncertainty_m=params.get("uncertainty_m"),
            source_note=f"mixed 帧 @ search {search_id}")
        used = [frame] + ([lo[0], hi[0]] if lo_xy else [])
        return geom, self._evidence(search_id, "", used, form="mixed_frame")

    @staticmethod
    def _evidence(search_id: str, bracket_id: str, markers: list,
                  *, form: str) -> dict:
        """这批谱是按哪一版参照系、哪几条标记认定的畴界取的 —— 写进产物。

        参照系换版之后旧结论要能重判,靠的就是这几个字段。``scan_angle_deg`` 原样
        透传:**null 表示读不到,不是 0**,而且本壳从不拿它做任何角度换算。
        """
        versions = {(_meta(m).get("reference_version") or None) for m in markers}
        angles = [_meta(m).get("scan_angle_deg") for m in markers]
        return {
            "domain_search_id": search_id,
            "bracket_id": bracket_id,
            "input_form": form,
            "reference_versions": sorted(str(v) for v in versions if v is not None),
            "reference_version_missing": any(v is None for v in versions),
            "scan_angle_deg": angles[0] if angles else None,
            "marker_verdicts": [_meta(m).get("verdict") for m in markers],
            "marker_coord_epochs": [getattr(m, "coord_epoch", None) for m in markers],
        }

    # ── 参照系确认闸 ──────────────────────────────────────────────────────

    def _reference_gate(self, evidence: dict, require: bool):
        """``DomainReference.confirmed_by`` 为空 ⇒ 拒绝(§3.4 的人工确认闸)。

        关掉这道闸是允许的,但**要留痕**:返回的 warnings 里会写明这一次是绕过的。
        """
        versions = list(evidence.get("reference_versions") or ())
        if len(versions) > 1:
            return self._refuse(
                "reference_version_conflict",
                f"这个 bracket 的两端是按不同版本的参照系判的({versions}) —— "
                "两端不可比,不能当成一道畴界。请用同一版参照系重判。")
        if not require:
            evidence["reference_confirmed_by"] = None
            evidence["reference_confirmation_bypassed"] = True
            return None
        if evidence.get("reference_version_missing") or not versions:
            return self._refuse(
                "reference_version_unknown",
                "标记里没有记下是按哪一版参照系判的,因此**无法确认**这道畴界经过"
                "人工确认。「读不到」不是「确认过」⇒ 不跑一晚上。")
        from mast.vision.domain_reference import load_reference

        ref = load_reference(version=versions[0])
        if ref is None:
            return self._refuse(
                "reference_not_found",
                f"找不到版本为 {versions[0]!r} 的畴参照系(判定就是按它做的)。"
                "参照系文件缺失/损坏时一律当作没有参照系。")
        if not str(getattr(ref, "confirmed_by", "") or "").strip():
            return self._refuse(
                "reference_not_confirmed",
                f"参照系 {versions[0]} 还没有人确认过(confirmed_by 是空的)。"
                "一条线谱是一晚上,跑在没人看过的「畴界」上是最贵的浪费 ⇒ 先让人"
                "看一眼那两簇帧并确认,或显式把 require_confirmed_reference 关掉。")
        evidence["reference_confirmed_by"] = ref.confirmed_by
        evidence["reference_confirmation_bypassed"] = False
        return None

    # ── 计划 ──────────────────────────────────────────────────────────────

    def plan(self, params: dict) -> list[CompositeStep]:
        """一个子步骤:把算好的坐标交给逐点取谱引擎。"""
        sub: dict[str, Any] = {
            "positions": json.dumps(self._positions_payload, ensure_ascii=False),
            "expected_coord_epoch": int(params.get("expected_coord_epoch") or 0),
        }
        # 只透传**用户真的给了**的项。引擎那边「省略 = 不改仪器上现有的设置」,
        # 所以给一个「本模块自己的默认」不是保守,是替人改了他配好的条件 ——
        # 静态默认遮蔽条件默认,本仓吃过这个亏。
        for key in _ENGINE_PASSTHROUGH:
            v = params.get(key)
            if v not in (None, ""):
                sub[key] = v
        return [CompositeStep(
            step_id="acquire_line",
            skill_name=POINT_ENGINE_SKILL,
            params=sub,
            optional=False,
            checkpoint_after=True,
            tags=("sts", "line"),
        )]

    # ── 汇总 ──────────────────────────────────────────────────────────────

    def aggregate(self, sub_results: dict, progress: CompositeProgress) -> dict:
        plan = self._plan_out
        budget = self._budget_out
        if plan is None:      # 没走到布点就被中止 —— 如实说,不去编一份摘要
            return {"acquired": False, "points": [],
                    "note": "没有走到布点这一步"}
        out: dict[str, Any] = dict(plan.summary_dict())
        out.update(self._evidence_out)
        out["estimated_duration_s"] = budget.get("total_s")
        out["estimated_duration_note"] = format_duration_note(budget)
        out["time_budget"] = budget
        out["points"] = self._merge_points(sub_results.get("acquire_line"))
        engine = getattr(sub_results.get("acquire_line"), "data", None) or {}
        for key in ("n_keep", "n_flagged", "n_discard", "n_unrated",
                    "n_move_failed", "n_suspect", "dat_paths",
                    "stopped_early", "stopped_reason", "coord_epoch"):
            if key in engine:
                out[key] = engine[key]
        out["point_count"] = len(out["points"])
        out["acquired"] = bool(engine)
        return out

    def _merge_points(self, sub_result) -> list[dict]:
        """几何(轴向位置 / 采集顺序 / 疏密档)与引擎的逐点结果按位置对齐。

        引擎少回几条时**不去 zip**:少的那些如实标 ``no_result``,而不是让第 k 条
        结果安到第 k 个坐标上 —— 那是「借最新文件伪造历史」的同一族错误。
        """
        planned = self._positions_payload
        rows = []
        data = getattr(sub_result, "data", None) or {}
        got = data.get("points") if isinstance(data.get("points"), list) else []
        for i, p in enumerate(planned):
            row = dict(p)
            row["success"] = False          # 显式写,缺省会被记录层当成 True
            row["status"] = "no_result"
            if i < len(got) and isinstance(got[i], dict):
                got_i = dict(got[i])
                # 坐标以**几何**为准:引擎回的是它自己收到的那份,两者不符说明
                # 中间有人改过,那件事要看得见而不是被覆盖掉。
                row["engine_x_m"] = got_i.get("x_m")
                row["engine_y_m"] = got_i.get("y_m")
                for k in ("success", "status", "error", "path", "verdict"):
                    if k in got_i:
                        row[k] = got_i[k]
            rows.append(row)
        return rows

    # ── 主流程 ────────────────────────────────────────────────────────────

    def run_composite(self, context, params: dict) -> SkillResult:
        spec = resolve_line_spec({
            k: params.get(k) for k in
            ("fine_spacing_nm", "coarse_spacing_nm", "fine_half_width_nm",
             "line_half_length_nm", "max_points")})

        try:
            geom = self._geometry_from_params(params)
            evidence: dict[str, Any] = {}
            if geom is None:
                if not str(params.get("domain_search_id") or "").strip():
                    return self._refuse(
                        "no_input",
                        "既没有 domain_search_id(优先形态:说一个 id,不说四个"
                        "浮点数),也没有完整的显式几何(origin_x_m + origin_y_m + "
                        "axis_deg + uncertainty_m)。")
                got = self._geometry_from_map(params)
                if isinstance(got, SkillResult):
                    return got
                geom, evidence = got
            else:
                evidence = {"input_form": "explicit",
                            "domain_search_id": "", "bracket_id": "",
                            "reference_versions": [],
                            "reference_version_missing": True,
                            "scan_angle_deg": None}
            plan = plan_from_geometry(geom, spec)
        except LinePlanRefused as exc:
            return self._refuse(exc.code, exc.message)

        if evidence.get("input_form") != "explicit":
            refused = self._reference_gate(
                evidence, bool(params.get("require_confirmed_reference", True)))
            if refused is not None:
                return refused

        # 整定时间要从**条件组**里取,不是 ``params.get("settle_s") or 0.0``:
        # 组名换成条件表之后,``settle_s`` 通常是省略的 —— 拿 0 去算,几十个点的
        # 线谱预计时长会短掉几分钟到几十分钟,而 D24 要这个数**开跑前**就诚实。
        # 解析用的是引擎同一个纯函数(``resolve_condition``),不是第二份实现;
        # 组名坏掉时由引擎拒绝,这里只是拿不到数就退回 0 并说明。
        cond = resolve_condition(
            params.get("condition"),
            {"settle_s": params.get("settle_s")} if params.get("settle_s") is not None
            else None)
        settle_s = (float(cond.spec.settle_s) if cond.spec is not None
                    else float(params.get("settle_s") or 0.0))
        budget = estimate_line_duration(
            plan.points,
            settle_s_per_point=settle_s,
            acquire_s_per_point=params.get("per_point_acquire_s"))

        self._plan_out = plan
        self._budget_out = budget
        self._evidence_out = evidence
        # 交给引擎的位置表 **按采集顺序**(D23:|s| 升序),引擎按列表顺序走。
        self._positions_payload = plan.positions()

        head = (f"线谱 {plan.point_count} 点,沿 bracket 轴"
                f"(方位 {plan.axis_angle_deg:.1f}°,**不是畴界法向**),"
                f"精细窗 ±{plan.fine_half_width_m * 1e9:.1f} nm"
                f"(来源: {plan.fine_half_width_source})。")
        note = format_duration_note(budget)

        if params.get("dry_run"):
            data = dict(plan.summary_dict())
            data.update(evidence)
            data.update({
                "acquired": False,
                "dry_run": True,
                # **不叫 `points`**:记录层只认 ``regions``/``points``,而且缺
                # ``success`` 字段时按 True 处理 —— 一次 dry_run 若把点位表放进
                # ``points``,实验地图上就会多出 N 个「已完成」的谱学足迹,而针尖
                # 一步都没动过。假标记比没有标记坏得多。
                "planned_points": self._positions_payload,
                "time_budget": budget,
                "estimated_duration_s": budget.get("total_s"),
                "estimated_duration_note": note,
            })
            return SkillResult(
                skill_name=self._skill_name(), success=True, data=data,
                summary=head + note + "(dry_run:没有采集)")

        result = self._graph_execute(context, params)
        result.summary = head + note
        return result


def make_tool(context_provider):
    return wrap_skill(LineSTSAcrossWall, context_provider)


__all__ = ["LineSTSAcrossWall", "POINT_ENGINE_SKILL", "SHELL_REFUSAL_CODES",
           "make_tool"]
