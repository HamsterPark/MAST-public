"""``CrossPointTipCheck`` —— 判据③的执行体:换 N 个位置各测一帧,再交给闭集聚合。

S1 修针循环设计 D4 / D5 / §3.5 的 ``S1.00`` 行。判据本体是纯函数
:func:`mast.conduct.cross_check.aggregate_cross_points`,**这一层一条裁决规则都
不写** —— 它只负责「把点采出来」,和 :class:`AssessFrameCorrugation` 与
``judge_corrugation`` 的分工完全同款。

## 它回答的是单帧答不了的那半句

对坏针的完整说法是「表面起伏极大**且**换位置扫依旧」。前半句是单帧观察,
**它分不开「针尖团簇」与「表面台阶簇」**;分得开的是换个位置再测一次。所以本技能
的产出只有闭集三值 ``bad_tip`` / ``surface_feature`` / ``undecidable``,
而且 ``bad_tip`` 需要 **judged 全票 + judged ≥ 2**。

## 为什么默认 N = 3

**3 不是标定出来的**,也不是从哪个分布上取的分位点 —— 它是「**最小能产生分歧的
奇数**」。N=2 时「1 坏 1 好」与「1 坏 1 判不了」在直觉上很不同,而聚合规则给它们
两个不同的待遇(前者 ``surface_feature``、后者 ``undecidable``),没有第三票可依;
N=3 让这两种情形都还有一票可查。再大只是线性地花时间(每点一整帧,分钟量级),
买不到新的定性区分。**这个数待用户拍板,不要拿它去标定什么。**

## 选点:两条后置筛,都是踩出来的

``FindCleanSpot(purpose="tip_shape")`` 给的是「不与任何避让圆相交」的点,那还不够:

1. **两两距离 ≥ 帧宽 × :data:`MIN_SEPARATION_FRAMES`** —— 否则两帧重叠 = 同一片
   表面测了两次,不是独立证据。``nearest_clean_from`` 的 ``exclude`` 判据是
   ``2 × r_spot``,``tip_shape`` 档只有 60 nm,而复测帧本身可能就有 100 nm ⇒
   光靠它不够。1.5 倍 = 两帧边缘之间还留半帧余量(1.0 倍才刚好不重叠)。
2. **``crash_count(x, y) == 0``** —— 在一个撞过针的坑上复测「仍然差」,那个差
   可能是坑,不是针。这是「**在自己刚炸出来的坑上判针尖**」那条教训的直接落地。

⚠️ ``tip_crash_tracker`` 是**进程内**记忆:无持久化、无代次,进程重启后它是空的。
所以「crash_count 全是 0」有两种读法 —— 「这些点没撞过」与「这个进程还什么都不
知道」。返回值里的 ``crash_memory`` 把该进程记着多少个格子如实说出来,不让第二种
读法伪装成第一种。(持久且带代次的那份历史是地图标记,接它是另一件事。)

## 只读:不记撞针、不写地图标记、不产生避让圆

复测是**观察**,不是动作。尤其 ``surface_feature`` 的点**不许上避让地图**:
避让圆是给「这里被我们弄脏了」用的,把「这里长得不一样」标成避让,会让后面找畴
界的搜索躲开自己的目标。测试里有一条只读性钉子(调用白名单 + 源码级断言)。

## 视野钉死,而且和阈值成对

起伏 RMS 随视野变,而把它归一化需要一个「RMS ∝ √W」的模型 —— **那是发明**。
所以复测视野是**协议钉死的常量**而不是判据要处理的变量:``scan_nm`` 不给就取
profile 里那对阈值自己声明的 ``corrugation_ref_scan_nm``,两个都没有就直接说
判不了。**显式阈值必须配显式视野**:拿一个显式阈值去配 profile 声明的视野,
量的就不是同一件事了(同 ``resolve_threshold_pair`` 的「要么都给,要么都不给」)。

## 「跑不起来」也是一个判决,不是一次失败

参数缺、候选不够、一个点都没测到 —— 全部返回 ``success=True`` +
``cross_verdict="undecidable"`` + 一句说清下一步的 ``reason``,**不返回失败**。
理由:下游闸门按 ``cross_verdict`` 分三路,而「步骤失败」会走 evidence_missing
那条更粗的路,把「判不了」和「技能崩了」折叠成同一件事。
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Iterator

from mast.core.types import (
    ParameterSpec,
    SafetyLevel,
    SkillCategory,
    SkillMetadata,
    SkillResult,
)
from mast.skills.composite._base import CompositeSkillGraph
from mast.skills.composite.graph_executor import (
    CompositeProgress,
    CompositeStep,
    GraphExecutor,
)

logger = logging.getLogger(__name__)

_NAME = "CrossPointTipCheck"

#: 默认复测点数 —— 「最小能产生分歧的奇数」,见模块注释。**不是标定值。**
DEFAULT_N_POINTS = 3

#: 两点最小间距 = 帧宽 × 这个倍数。1.0 倍才刚好不重叠,1.5 留半帧余量。
#: 刻意**不做成参数**:它不是一个可调的品味,是「两帧不许是同一片表面」这条
#: 几何要求;做成旋钮只会请人把它调到重叠。
MIN_SEPARATION_FRAMES = 1.5

#: 一次向 ``FindCleanSpot`` 要多少个候选。两条后置筛会淘汰掉一部分,只要一个
#: 最近点的话,一次「太近」就得整点作废。
CANDIDATE_COUNT = 32

#: 候选被淘汰的原因(闭集)。每一条都要能回答「那下一步该做什么」。
REJECT_NO_COORDS = "no_coords"          # 候选里没有可用坐标 —— 上游的问题
REJECT_TOO_CLOSE = "too_close"          # 与已用点重叠 ⇒ 不是独立证据
REJECT_CRASH_HISTORY = "crash_history"  # 这个格子撞过针 ⇒ 判的可能是坑
REJECT_REASONS: tuple[str, ...] = (
    REJECT_NO_COORDS, REJECT_TOO_CLOSE, REJECT_CRASH_HISTORY)

#: 本技能允许调用的子技能(白名单)。**新增一个名字就是一次只读性的重新论证** ——
#: 复测只观察,不留痕迹。测试对它逐条断言。
ALLOWED_SUBSKILLS: tuple[str, ...] = (
    "FindCleanSpot", "MoveToXY", "PreScanCheck", "SaveScan",
    "AssessFrameCorrugation")


def _num(value: Any) -> "float | None":
    """能转成有限 float 就转,否则 ``None``(「读不到」不是 0)。"""
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out and out not in (float("inf"), float("-inf")) else None


def pick_candidate(candidates: "list | tuple", *,
                   taken: "list[tuple[float, float]]",
                   min_sep_m: float,
                   crash_count: "Callable[[float, float], int]",
                   ) -> "tuple[dict | None, list[dict]]":
    """从候选里挑第一个同时过两条后置筛的点。**纯函数**(撞针记忆由调用方注入)。

    返回 ``(chosen | None, rejected)``。``rejected`` 逐条记下被淘汰的坐标和
    :data:`REJECT_REASONS` 里的原因 —— 一次「没找到点」必须说得出**为什么**,
    否则「这片表面用完了」和「筛得太严」在报告里长得一模一样。

    候选按距离由近及远,所以第一个过筛的就是最近的那个:一次大跨度移动会重新激起
    压电蠕变,而这一轮要走 N 次。
    """
    rejected: list[dict] = []
    for cand in candidates or ():
        if not isinstance(cand, dict):
            rejected.append({"why": REJECT_NO_COORDS, "candidate": repr(cand)})
            continue
        x = _num(cand.get("x_m"))
        y = _num(cand.get("y_m"))
        if x is None or y is None:
            rejected.append({"why": REJECT_NO_COORDS, "candidate": repr(cand)})
            continue
        near = _nearest_taken(x, y, taken)
        if near is not None and near[0] < float(min_sep_m):
            rejected.append({
                "why": REJECT_TOO_CLOSE, "x_m": x, "y_m": y,
                "distance_m": near[0], "min_separation_m": float(min_sep_m),
                "conflicts_with_index": near[1]})
            continue
        try:
            hits = int(crash_count(x, y))
        except Exception as exc:  # noqa: BLE001 —— 查不到撞针记忆不该拦住采点
            logger.debug("撞针记忆查询失败(按「没记录」处理): %s", exc)
            hits = 0
        if hits > 0:
            rejected.append({"why": REJECT_CRASH_HISTORY, "x_m": x, "y_m": y,
                             "crash_count": hits})
            continue
        return ({"x_m": x, "y_m": y,
                 "distance_m": _num(cand.get("distance_m"))}, rejected)
    return None, rejected


def _nearest_taken(x: float, y: float,
                   taken: "list[tuple[float, float]]",
                   ) -> "tuple[float, int] | None":
    """离 ``(x, y)`` 最近的已用点:``(距离 m, 序号)``;一个都没有时 ``None``。"""
    best: "tuple[float, int] | None" = None
    for i, (tx, ty) in enumerate(taken or ()):
        d = ((x - tx) ** 2 + (y - ty) ** 2) ** 0.5
        if best is None or d < best[0]:
            best = (d, i)
    return best


class CrossPointTipCheck(CompositeSkillGraph):
    """换 N 个位置各测一帧,聚合成闭集三值。只读。"""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name=_NAME,
            version="1.0.0",
            category=SkillCategory.COMPOSITE,
            # 会移动针尖并扫描 N 帧(分钟量级 × N),所以要人点头 —— 但它
            # **不改变**任何仪器设定,也不在表面上留下任何痕迹。
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "判定一根被怀疑的坏针尖,问题在**针尖**还是在**表面**:做法是在 N 个"
                "彼此拉开距离的干净落点上各复测一帧,再把逐点证据聚合成**恰好三个**"
                "答案:'bad_tip'、'surface_feature' 或 'undecidable'。单帧"
                "**分不开**这两者 —— 一簇台阶给出的起伏和一根坏针尖一样大 —— "
                "而这正是本技能存在的全部理由。'bad_tip' 要求在至少 2 个判得动的点"
                "上取得**全票一致**;判不了的那个点既不投这边也不投那边。"
                "**只读**:它会移动、会成像,但不记撞针、不写地图标记、也不产生避让圆,"
                "所以在这里被看出「不一样」的点,绝不会因此变成后面的搜索要绕开的地方。"
                "候选落点还要再过两道筛:任何两帧都不许重叠,而且这个点此前不许撞过针 "
                "—— 在我们自己弄出来的坑上判针尖,对针尖什么都说明不了。凑不出足够多"
                "彼此拉开的候选时,答案就是 'undecidable',并附上淘汰了几个、为什么;"
                "它绝不凑数。复测视野是**钉死的**(绝不跨尺度换算),因为起伏上限只在"
                "它被标定的那个视野上才有意义。"
            ),
            parameters=[
                ParameterSpec(
                    name="n_points", type="int",
                    description=(
                        "要复测几个落点。默认 3 —— 这是**还能产生分歧的最小奇数**,"
                        "**不是**一个标定值。小于 2 时根本不存在「另一个位置」,"
                        "结论就退化成单帧。"),
                    required=False, default=DEFAULT_N_POINTS,
                    min_value=2, max_value=9),
                ParameterSpec(
                    name="scan_nm", type="float",
                    description=(
                        "复测视野,单位**纳米**(就写一个普通数字,例如 100 表示 "
                        "100 nm)。留空则取样品 profile 里那条起伏上限**标定时所用"
                        "的**视野 —— 阈值与它的视野是**一对**,绝不跨尺度换算。"
                        "它同时也定下落点之间的最小间距(1.5 个帧宽)。"),
                    required=False, min_value=1.0, max_value=1e4),
                ParameterSpec(
                    name="corrugation_threshold_pm", type="float",
                    description=(
                        "起伏上限,单位**皮米**(就写一个普通数字,例如 40 表示 "
                        "40 pm)。留空则从样品 profile 里取 —— 这里**刻意没有 "
                        "default**:没传的阈值必须保持「没传」,查 profile 那一行"
                        "才到得了。你要是传了这个数,就**必须(MUST)**同时把 "
                        "scan_nm 也传上;"
                        "一个显式阈值配上 profile 声明的视野,量的就是另一件事了。"),
                    # 从 schema 入口验证省略参数，防止默认值注入绕过按尺度选择工作点。
                    required=False, min_value=0.1, max_value=1e6),
                ParameterSpec(
                    name="profile", type="str",
                    description=("样品阈值 profile 的名字。"
                                 "留空就用当前生效的那一个。"),
                    required=False, default=""),
                ParameterSpec(
                    name="dry_run", type="bool",
                    description=(
                        "只彩排选点:报告**将会**用哪些落点(以及淘汰了哪些),"
                        "不移动针尖,也不扫任何一帧。这种情况下判定恒为 "
                        "'undecidable' —— 彩排不产生证据。"),
                    required=False, default=False),
            ],
            # D9:本设计不新增任何 precondition。这里用的两条是子技能已有的
            # (``MoveToXY`` / ``PreScanCheck`` 都要 ``z_controller_on``),
            # 判「上一步产出的证据满不满足条件」是闸门的活,不是 precondition 的活。
            preconditions=["z_controller_on"],
            estimated_duration_s=900.0,
            composition_level=4,
            tags=["tip", "surface", "corrugation", "cross-check", "verdict",
                  "scan", "read"],
        )

    # ── 计划 ──────────────────────────────────────────────────────────────

    def plan_dynamic(self, params: dict,
                     executor: GraphExecutor) -> Iterator[CompositeStep]:
        profile = str(params.get("profile") or "").strip()
        dry_run = bool(params.get("dry_run", False))
        thr_pm = _num(params.get("corrugation_threshold_pm"))
        scan_nm_given = _num(params.get("scan_nm"))

        try:
            n_points = int(params.get("n_points") or DEFAULT_N_POINTS)
        except (TypeError, ValueError):
            n_points = 0
        if n_points < 2:
            self._refuse(
                f"n_points={params.get('n_points')!r} 少于 2 —— 一个点不构成"
                f"「换位置复测」,那时结论就退回成单帧,而单帧分不开针尖团簇与"
                f"表面台阶簇。要更严可以调大,不能调小。")
            return

        # ── 视野与阈值:成对,不许拼 ────────────────────────────────────────
        scan_nm, scan_src = self._resolve_scan_nm(scan_nm_given, profile)
        if scan_nm is None:
            self._refuse(
                "复测视野定不下来:既没有显式 scan_nm,样品 profile 里也没有"
                "corrugation_ref_scan_nm。起伏阈值离开它标定时的视野就没有意义,"
                "而跨视野换算在本仓有意不存在 —— 请显式给一个 scan_nm,或者先把"
                "profile 的那对(阈值 + 参照视野)标出来。")
            return
        if thr_pm is not None and scan_nm_given is None:
            self._refuse(
                f"给了显式阈值 {thr_pm} pm 却没给 scan_nm。一个显式阈值配上 "
                f"profile 声明的视野({scan_nm} nm),量的就不是同一件事了 —— "
                f"两个数要么都显式给,要么都走 profile。")
            return

        width_m = float(scan_nm) * 1e-9
        min_sep_m = width_m * MIN_SEPARATION_FRAMES
        self._setup = {
            "n_requested": n_points, "scan_nm": float(scan_nm),
            "scan_nm_source": scan_src, "min_separation_m": min_sep_m,
            "threshold_pm": thr_pm,
            "threshold_source": "explicit" if thr_pm is not None else "profile",
            "profile": profile or None, "dry_run": dry_run,
        }
        executor.set_partial("setup", dict(self._setup))

        # ── N 个点,逐个采 ──────────────────────────────────────────────────
        points: list = []
        taken: list[tuple[float, float]] = []
        rejected: list[dict] = []
        planned: list[dict] = []
        stopped_early = ""
        for index in range(n_points):
            spot, rej = yield from self._find_spot(
                executor, index, taken=taken, min_sep_m=min_sep_m)
            rejected.extend(rej)
            if spot is None:
                # 「候选不足」是一个结论,不是一次凑数的机会。停在这里。
                stopped_early = "no_candidate"
                break
            planned.append(dict(spot))
            taken.append((spot["x_m"], spot["y_m"]))
            if dry_run:
                continue
            point = yield from self._measure_point(
                executor, index, spot=spot, width_m=width_m,
                profile=profile, thr_pm=thr_pm, scan_nm=float(scan_nm))
            points.append(point)
            if len(points) > 1 and point.coord_epoch != points[0].coord_epoch:
                # 代次变了 ⇒ 中途发生过粗动,这批已经作废。**别再扫下去** ——
                # 后面每一帧都是几分钟,而它们已经不可能拼成一次「同一片区域的
                # 换位置复测」。作废由聚合器下(它会点名是哪一点),不在这里下。
                stopped_early = "coord_epoch_changed"
                break

        self._finish(points, planned=planned, rejected=rejected,
                     stopped_early=stopped_early, dry_run=dry_run,
                     n_points=n_points)

    # ── 采点 ──────────────────────────────────────────────────────────────

    def _find_spot(self, executor: GraphExecutor, index: int, *,
                   taken: "list[tuple[float, float]]", min_sep_m: float):
        """一个过筛的落点。``(spot | None, rejected)``。"""
        step_id = f"p{index + 1}_find"
        yield CompositeStep(
            step_id=step_id, skill_name="FindCleanSpot",
            params={
                # D5:**不给 FindCleanSpot 加第三个 purpose**。它的 purpose 取值
                # 映射到「这里被我们弄脏了」的损伤词表,而「复测」不是一种损伤。
                "purpose": "tip_shape",
                "exclude_spots": ";".join(f"{x},{y}" for x, y in taken),
                "count": CANDIDATE_COUNT,
            },
            optional=True, checkpoint_after=False, tags=("find", "read"))

        result = executor.sub_results.get(step_id)
        if result is None or not getattr(result, "success", False):
            why = getattr(result, "error", "") or "FindCleanSpot 没有返回结果"
            return None, [{"why": REJECT_NO_COORDS, "detail": str(why)}]

        data = dict(getattr(result, "data", None) or {})
        # 「读不到实验记录」与「表面干净」在几何上不可区分 —— 一路传到报告。
        self._map_known = self._map_known and bool(data.get("map_known", True))
        candidates = data.get("candidates")
        if not isinstance(candidates, (list, tuple)) or not candidates:
            # 只回了一个点也算候选:别让上游的返回形状决定我们能不能挑。
            if _num(data.get("x_m")) is not None:
                candidates = [{"x_m": data.get("x_m"), "y_m": data.get("y_m"),
                               "distance_m": data.get("distance_m")}]
            else:
                candidates = []
        spot, rejected = pick_candidate(
            candidates, taken=taken, min_sep_m=min_sep_m,
            crash_count=self._crash_count)
        return spot, rejected

    @staticmethod
    def _crash_count(x_m: float, y_m: float) -> int:
        from mast.core.tip_crash_tracker import get_tip_crash_tracker

        return int(get_tip_crash_tracker().crash_count(x_m, y_m))

    # ── 一个点的证据 ──────────────────────────────────────────────────────

    def _measure_point(self, executor: GraphExecutor, index: int, *,
                       spot: dict, width_m: float, profile: str,
                       thr_pm: "float | None", scan_nm: float):
        """移动 → 预检 → 存盘 → 起伏,收成一条 ``PointVerdict``。

        每一步都是 ``optional`` 的:一个点测不成是**这个点判不了**,不是整批失败。
        「判不了」既不投坏票也不投好票 —— 它有自己的名字,而这正是聚合规则要的。
        """
        from mast.conduct.cross_check import PointVerdict

        label = f"P{index + 1}"
        x_m, y_m = float(spot["x_m"]), float(spot["y_m"])
        abstain: list[str] = []

        move_id = f"p{index + 1}_move"
        yield CompositeStep(
            step_id=move_id, skill_name="MoveToXY",
            # coord_epoch 刻意不传:这对坐标是几秒前对着**当前**地图解出来的,
            # 而 FindCleanSpot 回传的那个代次是从截断窗口推导出来的,不是权威值,
            # 拿它去盖一道会拒绝的章,只会制造假的「陈旧」。真正的代次一致性由
            # 每点各记一次的权威查询 + 聚合器负责。
            params={"x_m": x_m, "y_m": y_m, "wait": True},
            optional=True, checkpoint_after=False, tags=("move", "write"))
        if executor.sub_results.get(move_id) is None:
            # 没走到 ⇒ **绝不在原地扫**:那是把同一片表面测第二次,冒充独立证据。
            abstain.append("移动失败,这个点没有去成")
            return PointVerdict(
                x_m=x_m, y_m=y_m, coord_epoch=self._read_epoch(),
                crash_count=0, abstain_reason=";".join(abstain),
                map_known=self._map_known, label=label)

        prescan_id = f"p{index + 1}_prescan"
        yield CompositeStep(
            step_id=prescan_id, skill_name="PreScanCheck",
            # 只钉视野(协议常量),线数与线时一律**不传** —— 传了就等于本技能
            # 替用户改了分辨率与针尖横向速度,而它自称只是来看一眼的。
            params={"center_x_m": x_m, "center_y_m": y_m, "width_m": width_m},
            optional=True, checkpoint_after=True, tags=("prescan", "scan"))
        prescan = executor.sub_results.get(prescan_id)
        pre_data = dict(getattr(prescan, "data", None) or {})
        tip_ready = pre_data.get("tip_ready")
        if not isinstance(tip_ready, bool):
            tip_ready = None            # 三态:``None`` 不折叠成坏,也不折叠成好
        if prescan is None:
            abstain.append("预检没跑起来")
        elif tip_ready is None:
            abstain.append(str(pre_data.get("abstain_reason")
                               or pre_data.get("unusable_reason")
                               or "预检判不了"))

        save_id = f"p{index + 1}_save"
        yield CompositeStep(
            step_id=save_id, skill_name="SaveScan", params={},
            optional=True, checkpoint_after=False, tags=("save", "write"))
        saved = dict(getattr(executor.sub_results.get(save_id), "data", None) or {})
        # 等待超时视为没有可用文件，不能继续声称已经取得本次采集。
        scan_path = None if saved.get("timed_out") else saved.get("saved_path")
        if saved.get("timed_out"):
            abstain.append("存盘超时,不拿「最新文件」冒充这一帧")

        corr: dict = {}
        if scan_path:
            corr_id = f"p{index + 1}_corrugation"
            yield CompositeStep(
                step_id=corr_id, skill_name="AssessFrameCorrugation",
                params=self._corrugation_params(str(scan_path), profile,
                                                thr_pm, scan_nm),
                optional=True, checkpoint_after=True,
                tags=("corrugation", "read"))
            corr = dict(getattr(executor.sub_results.get(corr_id),
                                "data", None) or {})
            if not corr:
                abstain.append("起伏判据没跑起来")
            else:
                # 存下来的**是不是我们刚扫的这一帧** —— 用几何对一次账。视野对不上
                # 时判据②本来也会说「尺度不匹配」,但那句话指向的是「阈值标在别的
                # 视野上」;这里要说的是另一件事:**这个文件可能根本不是这一帧**。
                # 两个原因指向两个完全不同的下一步,不许折叠成一句。
                got = _num(corr.get("width_nm"))
                if got is not None and abs(got - scan_nm) > 0.05 * scan_nm:
                    abstain.append(
                        f"存下来的帧宽 {got:.1f} nm ≠ 这次要扫的 {scan_nm:.1f} nm"
                        f" —— 这个文件很可能不是刚才那一帧,它的起伏结论作废")
                    # 丢掉,不是打个折:一个关于**别的帧**的结论,再准也回答不了
                    # 「这个点上起伏多大」。判据①不受影响 —— 它有自己的取帧路径。
                    corr = {}
        else:
            # 实时缓冲未必与保存文件同源，不能默认二者可比。
            abstain.append("这一帧没存下来,起伏判不了")

        verdict = str(corr.get("verdict") or "undecidable")
        # 代次在这个点的**证据采完之后**记:它要证的是「量这一帧的整段时间里
        # 没发生过粗动」,而不是「排计划的那一刻是第几代」。走权威查询,
        # **不用** FindCleanSpot 回传的那个(它从截断窗口推导,会漏报代次)。
        return PointVerdict(
            x_m=x_m, y_m=y_m,
            coord_epoch=self._read_epoch(),
            frame_usable=(bool(corr["frame_usable"])
                          if isinstance(corr.get("frame_usable"), bool) else None),
            similarity=_num(pre_data.get("similarity")),
            tip_ready=tip_ready,
            corrugation_verdict=verdict,
            corrugation_value_pm=_num(corr.get("value_pm")),
            corrugation_detrend=str(corr.get("detrend") or ""),
            corrugation_statistic=str(corr.get("statistic") or ""),
            # 选点时已筛为 0;记下来是为了事后能查「这个结论是不是在坑上判的」。
            crash_count=self._crash_count(x_m, y_m),
            # 判据④(谱正反扫迟滞)的前置筛本技能不开 ⇒ ``skipped``。
            # 它**不是** ``not_fired`` —— 「没看见」不许给「好」投票。
            spectrum_hysteresis="skipped",
            frame_skipped_by_prescreen=False,
            abstain_reason=";".join(a for a in abstain if a),
            map_known=self._map_known,
            label=label)

    @staticmethod
    def _corrugation_params(scan_path: str, profile: str,
                            thr_pm: "float | None", scan_nm: float) -> dict:
        """交给判据②的参数。阈值和它的参照视野**同源**:要么都显式,要么都不传。

        不传时 ``AssessFrameCorrugation`` 去查 profile 的那一对 —— 而**那一行只有
        在参数没有 default 时才到得了**,所以这里也绝不能顺手填一个数。
        """
        out: dict[str, Any] = {"scan_path": scan_path}
        if profile:
            out["profile"] = profile
        if thr_pm is not None:
            out["threshold_pm"] = float(thr_pm)
            out["ref_scan_nm"] = float(scan_nm)
        return out

    @staticmethod
    def _read_epoch() -> "int | None":
        """当前坐标代次的**权威**查询。读不到返回 ``None`` —— 不是 0。"""
        from mast.core.coord_epoch import read_current_epoch

        return read_current_epoch()

    def _resolve_scan_nm(self, given: "float | None",
                         profile: str) -> "tuple[float | None, str]":
        """复测视野:显式优先,否则取 profile 里那对阈值声明的参照视野。"""
        if given is not None:
            return float(given), "explicit"
        try:
            from mast.vision.scan_prep_thresholds import resolve

            ref = _num(getattr(resolve(profile or None),
                               "corrugation_ref_scan_nm", None))
        except Exception as exc:  # noqa: BLE001 —— 读不到 profile ⇒ 定不下来
            logger.debug("样品 profile 读不到(视野定不下来): %s", exc)
            return None, "unavailable"
        if ref is None:
            return None, "unset"
        return ref, f"profile:{profile or 'active'}"

    # ── 收口 ──────────────────────────────────────────────────────────────

    def _refuse(self, reason: str) -> None:
        """协议根本没跑起来 ⇒ ``undecidable`` + 一句说清下一步的话。"""
        self._out = {
            "verdict": "undecidable", "cross_verdict": "undecidable",
            "n_points": 0, "n_judged": 0, "n_bad": 0, "n_good": 0,
            "coord_epoch": None, "map_known": self._map_known,
            "reason": reason, "points": [],
            "planned_points": [], "rejected_candidates": [],
            "insufficient_candidates": False, "stopped_early": "not_started",
            "crash_memory": self._crash_memory(),
        }
        self._out.update(self._setup)

    def _finish(self, points: list, *, planned: list, rejected: list,
                stopped_early: str, dry_run: bool, n_points: int) -> None:
        """聚合 + 记账。裁决规则**只有一份**,在聚合器里。"""
        from mast.conduct.cross_check import aggregate_cross_points

        res = aggregate_cross_points(points)
        out = res.as_dict()
        short = len(points) < n_points

        reason = res.reason
        if dry_run:
            reason = (f"彩排:选出了 {len(planned)}/{n_points} 个落点,"
                      f"没有移动针尖、没有扫一帧,所以没有证据 —— 判不了。")
            out["verdict"] = "undecidable"
        elif short:
            # ⚠️ 这是本技能**唯一**一处盖掉聚合器结论的地方,而且只往
            # 「更保守」的方向盖:采不满 N 个点意味着这一片表面已经用完了,
            # 而在一片用完的表面上判「换了位置仍然差」,差的可能正是那些用完它
            # 的动作 —— 不是针。判不了会走到人那里,而 bad_tip 会启动一整套
            # 退针 / 换样品 / 修针(小时量级)。两个方向的代价不对称。
            # 每一个点的原始证据仍在 ``points`` 里,一条都没丢。
            head = (f"候选不足:要 {n_points} 个互不重叠的落点,只凑出 "
                    f"{len(points)} 个(淘汰 {len(rejected)} 个候选)。"
                    f"**不凑数** —— 这一片表面可能已经用完,换个区域再判,"
                    f"或者先确认筛选半径。")
            reason = f"{head} 已测到的证据:{res.reason}"
            out["verdict"] = "undecidable"
        elif stopped_early == "coord_epoch_changed":
            reason = (f"中途坐标代次变了(发生过粗动),剩下的点没有再扫 —— "
                      f"{res.reason}")

        out["reason"] = reason
        # 闸门 selector 用的名字 —— 与 ``cross_check_analysis`` 那一路同名,
        # 免得同一个结论在两条路上叫两个名字。
        out["cross_verdict"] = out["verdict"]
        out["insufficient_candidates"] = bool(short and not dry_run)
        out["stopped_early"] = stopped_early
        out["planned_points"] = list(planned)
        out["rejected_candidates"] = list(rejected)
        out["rejected_by_reason"] = {
            why: sum(1 for r in rejected if r.get("why") == why)
            for why in REJECT_REASONS}
        out["crash_memory"] = self._crash_memory()
        out.update(self._setup)
        self._out = out

    @staticmethod
    def _crash_memory() -> dict:
        """撞针记忆的**出身**,不是它的结论。

        ``crash_count == 0`` 有两种读法 ——「这些点没撞过」与「这个进程还什么都
        不知道」。追踪器是进程内的、无持久化、无代次,粗动之后整体清空,所以第二
        种读法很常见。把它记着多少个格子如实说出来,读的人才分得开。
        """
        try:
            from mast.core.tip_crash_tracker import get_tip_crash_tracker

            snap = dict(get_tip_crash_tracker().snapshot())
        except Exception as exc:  # noqa: BLE001
            logger.debug("撞针记忆快照读不到: %s", exc)
            snap = {}
        snap["scope"] = "process_local"
        snap["note"] = ("撞针记忆是进程内的(无持久化、无坐标代次,粗动后清空)。"
                        "tracked_regions=0 意味着这个进程还没见过任何撞针,"
                        "**不等于**这些点历史上没撞过。")
        return snap

    # ── 框架接口 ──────────────────────────────────────────────────────────

    def aggregate(self, sub_results: dict, progress: CompositeProgress) -> dict:
        out = dict(self._out or {})
        if not out:
            # 计划一步都没走完(中止 / 异常)—— 「没跑到裁决」不是「没问题」。
            out = {
                "verdict": "undecidable", "cross_verdict": "undecidable",
                "n_points": 0, "n_judged": 0, "n_bad": 0, "n_good": 0,
                "coord_epoch": None, "map_known": self._map_known,
                "reason": "没有走到裁决 —— 判不了。", "points": [],
            }
        return out

    def _validate_products(self, data: dict) -> tuple[bool, str]:
        """本技能的产物是一个**判断**,不是一张图 —— 跳过通用产物闸。

        通用闸会拿顶层的路径键再判一次帧(死平 / 全 NaN ⇒ degraded),而
        「这一帧太平」正是判据②已经用 ``low`` 说出来的话,而且它在这里是**证据**
        (弃权),不是缺陷。让通用闸再判一遍,等于给同一帧两个来源不同的结论,
        然后用后一个盖掉前一个。
        """
        return True, ""

    def run_composite(self, context, params: dict) -> SkillResult:
        self._out = {}
        self._setup = {}
        # 任一点读不到地图 ⇒ 顶层为假,一路传进报告。
        self._map_known = True
        result = self._graph_execute(context, params)
        if not result.success:
            return result
        data = dict(result.data or {})
        return SkillResult(
            skill_name=self._skill_name(), success=True, data=data,
            summary=_summarize(data),
            nanonis_calls=list(result.nanonis_calls))


def _summarize(data: dict) -> str:
    """给用户/模型读的一句话。**「判不了」与「判出来是针」是两句不同的话。**"""
    head = {
        "bad_tip": "换了位置仍然差 ⇒ 指向针尖",
        "surface_feature": "换个位置就不一样了 ⇒ 指向表面,不是针",
        "undecidable": "判不了",
    }.get(str(data.get("verdict")), str(data.get("verdict")))
    out = f"{head} —— {data.get('reason', '')}"
    if data.get("dry_run"):
        out = "【彩排】" + out
    if data.get("map_known") is False:
        out += " ⚠️ 读不到实验记录,无法确认这些点是否干净(不知道 ≠ 干净)。"
    return out


def make_tool(context_provider):
    from mast.agents._shared.skill_adapter import wrap_skill

    return wrap_skill(CrossPointTipCheck, context_provider)


__all__ = ["ALLOWED_SUBSKILLS", "CANDIDATE_COUNT", "DEFAULT_N_POINTS",
           "MIN_SEPARATION_FRAMES", "REJECT_CRASH_HISTORY", "REJECT_NO_COORDS",
           "REJECT_REASONS", "REJECT_TOO_CLOSE", "CrossPointTipCheck",
           "pick_candidate"]
