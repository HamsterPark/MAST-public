"""SearchDomainBoundary —— 在表面上找畴界。两个模式:``survey`` 铺网格、``bisect`` 二分。

S3 畴搜索设计 D7/D8/D9/D9b/D10/D13、§3.3、§3.4。

## 两个模式是两件事,不是同一件事的粗细两档

``mode="survey"`` 铺一张粗网格,每个格点扫一张原子帧、抽一次畴指纹,把结果如实报
出来,并做一次**无监督聚类**(k=2)让人看。它**不**判「这是 A 相」——那需要一个人
确认过的参照系(``vision.domain_reference``),而参照系正是这一步的**产物**,不是
它的前提。

⇒ 没有参照系时每一个点都是 ``undetermined(no_reference)``,**这是正确行为,
不是失败**(设计 D3/R6)。代码不能自己发明「这是 A 相」——同一条纪律在
``vision/scan_prep_thresholds`` 里写作「没有那个样品的数据就造一个 profile 出来
等于伪造标定」。验收 R6 的通过判据逐字就是「全部 ``no_reference`` + M 个指纹」,
而**出现任何 label 就是停机信号**。

``mode="bisect"`` 反过来:它**要求**参照系已经标定好——两个端点必须各自判出一个
**互不相同的 label**,否则「在两者之间找分界」这句话没有定义。没有参照系时每个点
都是 ``undetermined``,于是永远凑不出一对端点;这里把它做成一条**显式拒绝**而不是
让它表现成「这一对不成立」,因为两句话指向的下一步完全不同(去建参照系 vs 换一对点)。

跨站点普查(``allow_coarse_move=True``)两个模式都**显式拒绝**而不是悄悄降级:
悄悄降级会让调用方以为畴界被定位了,而那是这条流程里最贵的一种误解。跨站点畴界的
诚实形态见 :func:`cross_site_downgrade` —— **步 + 不确定度,永远不是米坐标**。

## 三件必须写在这里、否则下一个人一定会重新踩的事

**一、粗网格必须显式解除中心区。** ``map_scope.analysis_config()`` 在有 XY 粗动的
仪器上带 ``center_zone_side_m = 500 nm``(它与 ``pulse_r_m = 500 nm`` 是**一对**:
一发脉冲盖满中心区 ⇒ 被迫换区)。畴界间距是微米量级,直接用那份配置,整张网格会被
压进 500 nm 见方的一小块——比一条畴界的间距还小的概率很大,于是「扫了九个点全是
同一个畴」变成一句必然的废话。所以 :func:`release_center_zone` 显式
``center_zone_side_m=None`` 并把理由写进结果(设计陷阱 2)。
**只解除中心区,绝不顺手调小 ``pulse_r_m``** ——那两个 500 nm 是刻意配成一对的,
改一个不改另一个,「一发就被迫换地方」那条设计就散了(陷阱 3)。

**二、下发扫描之前先算 nm/px。** 常见的 50 nm / 256 px = 0.195 nm/px 直接落在
``scale_gate="off"``,判据只会说「判不了」。扫完一张判不了的图再说,白花的不只是
一帧的时间——流程还会把「判不了」误读成「这里没有畴」。预检走
``atomic_phase.plan_scale``(阈值 0.02/0.05 的归属地,不在这里复制第二份),
不过就**拒绝并说清楚要改什么**,一帧都不发(陷阱 10)。

**三、帧角读不到就是读不到,不按 0° 处理。** 指纹的角度必须归到样品系
(``k_angle_sample_deg = k_angle_frame_deg + scan_angle_deg``);把 ``None`` 折叠成
``0.0`` 会让**同一个畴在两种帧角下报成两个畴**,而且零报错。所以逐帧走 ``ScanAt``
(它有 ``angle_deg``,而 ``ExecuteScanPlan`` 的 frame schema 不透传角度),角度由
``AssessDomainPhase`` 从 ``.sxm`` header 直读,读不到 ⇒ 该点
``undetermined(unknown_frame_angle)``(设计 D4/D12b、陷阱 1)。

## 二分的四条硬约束(设计 D9/D9b)

**一、终止条件的物理下限是帧宽,不是压电分辨率。** 见 :func:`locate_tolerance` 的
文档——这一条写在那里,因为下一个想「再调细一点」的人会先打开那个函数。

**二、二分落点不走 ``pick_next_position``。** 那条路的 ``reuse_overlap_frac`` 会把
中点丢掉:中点与两个端点帧的重叠几乎必然 ≥30%。二分要的正是「回到已经扫过的地方
中间再扫一帧」,而 ``pick_next_position`` 的整个用途是「找**没**扫过的地方」——
两种语义共用一个函数,结果是二分永远排不出下一个点,而且**零报错**(它只会说
「这片表面在当前策略下没有位置了」)。所以二分自己算坐标,只查压电范围与避让圈
(:func:`probe_position_problem`,设计陷阱 4)。

**三、「不可判」的下一个坐标由算法给,不由模型给。** 三级:同点重采一帧 → 沿
``lo→hi`` 的**垂直**方向偏移一个帧宽(左右交替,:func:`offset_probe`) → 预算用尽就
诚实 ``ABANDONED(undetermined)``。全程没有一个数字来自语言模型——同一条纪律在
``io/map_analysis`` 的模块头里写作「never by asking a language model to look at a
picture and guess」。垂直偏移不是随便挑的方向:畴界搜索的信息全在 ``lo→hi`` 这个
轴上,垂直移动**不改变沿轴位置**,所以偏移点判出来的 label 照样能拿来收窄区间。

**四、二分中途发生粗动 ⇒ 当前 bracket 立即作废,不是暂停。** ``p_lo``/``p_hi`` 的
米坐标在粗动之后指向的是**另一片表面**,它们之间已经不存在「那条畴界」。作废
(``INVALIDATED_BY_EPOCH``)与暂停的差别不是措辞:暂停意味着「回头还能接着用这两个
坐标」,而那正是要防的事。跨代次坐标换算在本仓**有意不存在**(开环步长随温度差五倍),
所以唯一的处置是重新规划。

## 聚类需要一个代码给不出的数

距离在**晶格对称性的商空间**里算,而 ``symmetry_deg``(六角 60°、矩形 90°)是
**样品事实**,不是仪器事实,更不是可以由代码猜的东西(设计 D5)。所以:

* 有参照系 ⇒ 用参照系里的 ``symmetry_deg``;
* 没有参照系但用户显式传了 ``symmetry_deg`` ⇒ 用它,并记下来源;
* 两个都没有 ⇒ **指纹照报,聚类不做**,理由是 ``no_symmetry_deg`` 并给出下一步。
  这不是失败,是「这个问题现在没有定义」——猜一个 60 出来,聚类会照常出结果、
  照常看起来合理,而它量的是一个没人给过的假设。

同理,没有参照系时比对权重 ``w_angle``/``w_period`` **未标定**,聚类用等权 1:1 并在
报告里说明:这两个数的量纲由实测分布定,只有参照系确定之后才有意义。

## 「分母比分子重要」

分离度 = 簇间距离 / 同簇最大距离。每簇只有一帧时**分母没有样本**,这时比值
**报 ``None`` 并说原因**,不报一个无穷大或者一个凑出来的数(设计 D6:一个判据在
正例上有信号不算数,要证明它在反例上没有)。

## 状态的真源在 map_markers,不在 sidecar

每个采样点在 ``data["points"]`` 里如实报告一条记录(坐标 + 足迹 + ``meta``),
由 runtime 的 recorder 落库成 ``kind="scan"`` 的 marker——**不新增 marker kind**
(新 kind 会撞上双端 KIND 镜像那个结构性缺口,失败模式是静默变灰),也**不在技能里
直接写库**(分层原则)。⚠️ recorder 是 fire-and-forget:字段名拼错不报错,只会
表现为「``meta.fingerprint`` 一直是 None」——所以对应的验证测试是**造 payload 走
真的记录路径看库里落了什么**,不是 grep 落点(陷阱 16)。
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field, replace
from typing import Any, Iterator

from mast.agents._shared.skill_adapter import wrap_skill
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


# ── 闭集 ──────────────────────────────────────────────────────────────────────

MODE_SURVEY = "survey"
MODE_BISECT = "bisect"
ALL_MODES: tuple[str, ...] = (MODE_SURVEY, MODE_BISECT)

#: 采样点的状态。**与指纹的 verdict 是两回事**:这里回答「这个点采到了吗」,
#: verdict 回答「这一帧是哪个畴」。合成一个字段会让「没扫成」和「扫成了但判不了」
#: 长得一模一样,而两者的下一步完全不同。
POINT_ASSESSED = "assessed"          # 扫到了、判据也跑了(verdict 另说)
POINT_PLANNED = "planned"            # dry_run:只排了计划,没发扫描
POINT_SCAN_FAILED = "scan_failed"    # ScanAt 失败(帧没扫成)
POINT_NO_FRAME = "no_frame"          # 扫完了但找不到落盘的 .sxm
POINT_STALE_FRAME = "stale_frame"    # 找到的是上一个点的文件 —— 不许拿它当本点
POINT_ASSESS_FAILED = "assess_failed"  # 判据壳本身没跑起来
POINT_EPOCH_STALE = "epoch_stale"    # 粗动了,这个坐标已经指向另一片表面
POINT_SKIPPED = "skipped"            # 前面的失败让整轮提前收尾
POINT_UNREACHABLE = "unreachable"    # 落点超压电范围 / 压在避让圈上 —— 针尖没去
ALL_POINT_STATUS: tuple[str, ...] = (
    POINT_ASSESSED, POINT_PLANNED, POINT_SCAN_FAILED, POINT_NO_FRAME,
    POINT_STALE_FRAME, POINT_ASSESS_FAILED, POINT_EPOCH_STALE, POINT_SKIPPED,
    POINT_UNREACHABLE,
)

#: **针尖真的去过**的那些状态 —— 只有它们进 ``data["points"]``,也就是只有它们
#: 会变成地图上的足迹。
#:
#: 这条区分是承重的:``planned`` / ``skipped`` / ``epoch_stale`` 三种点针尖一次都
#: 没去过,把它们记进地图就是往实验记录里写没发生过的事。同一个坑本仓刚补过一轮
#: (三个纯分析技能各自画了一个不存在的扫描足迹,而它们的 description 都写着
#: 「不碰硬件」)。完整的账在 ``data["point_log"]`` 里,一条不少。
VISITED_STATUS: tuple[str, ...] = (
    POINT_ASSESSED, POINT_SCAN_FAILED, POINT_NO_FRAME, POINT_STALE_FRAME,
    POINT_ASSESS_FAILED,
)

#: 拒绝码(闭集)。拒绝 = 「这件事没做成」⇒ 技能失败;而「判不了」是正常产物,
#: 技能照样成功。两者混用会让 composite 的必需步骤把一次正常的普查整条中止。
REFUSE_MODE_UNKNOWN = "mode_unknown"
REFUSE_COARSE_NOT_YET = "cross_site_not_implemented"
REFUSE_CROSS_SITE_BISECT = "cross_site_cannot_be_bisected"
REFUSE_SCALE = "atomic_scale_unreachable"
REFUSE_NO_POSITIONS = "no_grid_positions"
REFUSE_EPOCH_STALE = "coord_epoch_stale"
REFUSE_NO_BRACKET = "no_bracket_given"
REFUSE_BRACKET_NOT_FOUND = "bracket_not_found"
REFUSE_BRACKET_EPOCH_SPLIT = "bracket_endpoints_span_epochs"
REFUSE_BRACKET_NOT_A_PAIR = "bracket_endpoints_are_not_a_pair"
REFUSE_NO_REFERENCE = "bisect_needs_a_reference"
ALL_REFUSALS: tuple[str, ...] = (
    REFUSE_MODE_UNKNOWN, REFUSE_COARSE_NOT_YET, REFUSE_CROSS_SITE_BISECT,
    REFUSE_SCALE, REFUSE_NO_POSITIONS, REFUSE_EPOCH_STALE,
    REFUSE_NO_BRACKET, REFUSE_BRACKET_NOT_FOUND, REFUSE_BRACKET_EPOCH_SPLIT,
    REFUSE_BRACKET_NOT_A_PAIR, REFUSE_NO_REFERENCE,
)

#: 聚类做不了的原因(闭集)。每一条都要给出**能做的一件事**。
CLUSTER_NO_SYMMETRY = "no_symmetry_deg"
CLUSTER_TOO_FEW = "too_few_comparable_fingerprints"
CLUSTER_OK = ""

#: 采样点在一次搜索里的角色(闭集,落进 marker 的 ``meta.role``,设计 §3.4)。
#: 普查的点全是 ``seed``(不属于任何 bracket);二分的点是后面四个。
ROLE_SEED = "seed"
ROLE_LO = "lo"
ROLE_HI = "hi"
ROLE_MID = "mid"
ROLE_OFFSET = "offset"
ALL_ROLES: tuple[str, ...] = (ROLE_SEED, ROLE_LO, ROLE_HI, ROLE_MID, ROLE_OFFSET)

#: 二分状态机的状态(闭集,设计 D9)。**三个终态互不折叠**:收敛了、放弃了、
#: 被粗动作废了 —— 下一步分别是「拿去做线谱」「换一对点」「重新规划坐标」。
STATE_SEEDING = "SEEDING"
STATE_BRACKETED = "BRACKETED"
STATE_BISECTING = "BISECTING"
STATE_CONVERGED = "CONVERGED"
STATE_ABANDONED = "ABANDONED"
STATE_INVALIDATED = "INVALIDATED_BY_EPOCH"
ALL_STATES: tuple[str, ...] = (
    STATE_SEEDING, STATE_BRACKETED, STATE_BISECTING,
    STATE_CONVERGED, STATE_ABANDONED, STATE_INVALIDATED,
)
TERMINAL_STATES: tuple[str, ...] = (
    STATE_CONVERGED, STATE_ABANDONED, STATE_INVALIDATED)

#: 终态的细分理由(闭集)。``ABANDONED`` 有三种成因,合成一个字段会让「预算用尽」
#: 与「出现了第三个畴」长得一样,而后者是**证据矛盾**,该去问人不该去换点。
END_TOLERANCE = "locate_tolerance_reached"
END_MIXED = "mixed_frame"
END_UNDETERMINED = "undetermined_budget_spent"
#: 「一帧都没采成」与「采到了但判不了」是两件事:前者去查针尖/反馈,后者去看
#: verdict_reason(尺度门、帧角、原子相)。合成一个码会把人送错方向。
END_NOT_SAMPLED = "probes_could_not_be_taken"
END_MAX_ITERATIONS = "max_iterations_spent"
END_THIRD_LABEL = "third_label_contradiction"
END_NOT_A_PAIR = "endpoints_are_not_a_pair"
END_EPOCH = "coarse_move_invalidated_the_metres"

#: 二分的默认预算。
#:
#: ``max_iterations``:压电范围 3 µm、原子帧 5 nm ⇒ log₂(3000/5) ≈ 9.2,12 次够把
#: 整个视野二分到帧宽,再多是在原地打转(区间已经小于一个帧宽,两帧看的是同一片
#: 表面,指纹必然相同 —— 见 :func:`locate_tolerance`)。
#: ``offset_budget``:**按 bracket 计**,不是按中点计 —— 预算用尽时整个 bracket
#: 就 ``ABANDONED`` 了,所以「每个中点各给两次」和「一共给两次」在行为上同一件事,
#: 选按 bracket 计是因为它是**能被读出来的那个数**(marker 里数得出来)。
DEFAULT_MAX_ITERATIONS = 12
DEFAULT_OFFSET_BUDGET = 2

#: 连续多少帧扫不出来就收尾。两帧连着扫不成,后面几十帧多半也扫不成,而每一帧
#: 在原子档上都是几分钟的机时。刻意不设成 1:单帧失败(一次超时、一次保存竞态)
#: 在真机上是常事,为它放弃整轮普查太贵。
MAX_CONSECUTIVE_SCAN_FAILURES = 2

#: 找落盘 .sxm 时允许的最大文件年龄 = 这一帧的估计耗时 + 这个余量(秒)。
#: 上界必须有:没有它,「最近一个 .sxm」可以是半小时前另一个流程留下的文件,
#: 而那正是本仓记过的「借最新文件伪造历史」。
FRAME_LOOKUP_HEADROOM_S = 120.0


# ── 采样网格:复用既有规划器,只改两件事 ──────────────────────────────────────

def default_grid_pitch_m(cfg) -> float:
    """粗网格间距的兜底值 = 可用半径 / 4(设计 D8 第 0 级)。

    传进来的 ``cfg`` **必须是已经解除中心区的那一份**,否则算出来的是 500 nm 见方
    那一小块的四分之一(125 nm),比畴界间距小两三个数量级。调用顺序由
    :func:`release_center_zone` 保证:先解除,再拿 ``effective_half_range_m`` 算。

    目标不是覆盖整片表面,是**碰到两个不同的指纹**——所以间距按可用区尺度定,
    不按帧尺度定。
    """
    try:
        half = float(cfg.effective_half_range_m)
    except (TypeError, ValueError, AttributeError):  # pragma: no cover — 防御
        return 0.0
    return half / 4.0 if half > 0 else 0.0


def release_center_zone(base, *, frame_size_m: float,
                        grid_pitch_m: "float | None" = None) -> tuple[Any, str]:
    """粗网格用的 ``AnalysisConfig`` + 「为什么解除中心区」那句话。

    返回 ``(cfg, reason)``。三处改动,每一处都有理由:

    * ``center_zone_side_m=None`` —— 有 XY 粗动的仪器上它是 500 nm,会把整张网格
      压进比畴界间距还小的一块(设计陷阱 2)。
    * ``frame_size_m`` —— 网格铺的是**原子帧**,不是巡览帧;候选点间距是帧尺寸的
      倍数,拿巡览帧算出来的路线放到原子帧上是错的。
    * ``point_spacing_factor`` / ``ring_width_factor`` —— 一起改成 ``间距 / 帧宽``。
      只拉开环上的点距而不拉开环间距,只是把「挨着排」从一个方向换到另一个方向
      (``map_scope`` 自己的注释)。

    **``pulse_r_m`` 一个字都不动。** 它与中心区是刻意配成的一对(一发脉冲盖满中心区
    ⇒ 被迫换区);顺手调小它会把那条设计拆散,而且方向是**更不保守**的那一边
    (设计陷阱 3)。避让圈只会让网格少几个点,那是对的代价。
    """
    frame = float(frame_size_m)
    if not (frame > 0):
        raise ValueError(f"帧尺寸必须为正,收到 {frame_size_m!r}")

    # 先只解除中心区(间距还没定),这样 effective_half_range_m 已经是真实可用区,
    # 兜底间距才算得对 —— 顺序在这里是承重的。
    released = replace(base, center_zone_side_m=None, frame_size_m=frame)
    pitch = float(grid_pitch_m) if grid_pitch_m else default_grid_pitch_m(released)
    if not (pitch > 0):
        pitch = frame
    spacing = pitch / frame

    cfg = replace(released, point_spacing_factor=spacing, ring_width_factor=spacing)
    had_zone = getattr(base, "center_zone_side_m", None)
    if had_zone:
        why = (
            f"粗网格解除了中心区限制(原本 {float(had_zone) * 1e9:.0f} nm 见方):"
            f"那个数是为「一发脉冲盖满中心区 ⇒ 被迫换区」配的,而畴界间距是微米量级,"
            f"照用会把整张网格压进比一条畴界还小的一块地方。整个压电范围都要用上。"
            f"脉冲避让半径**没有动**——它与中心区是一对,只解一边、不碰另一边。")
    else:
        why = ("这台仪器本来就没有中心区限制(没有 XY 粗动的仪器不设),"
               "网格直接用满整个压电范围。")
    why += (f" 网格间距 {pitch * 1e9:.0f} nm = 帧宽 {frame * 1e9:.1f} nm 的 "
            f"{spacing:.1f} 倍。")
    return cfg, why


# ── 聚类:最简 k=2,给人看的那一步 ──────────────────────────────────────────────

@dataclass(frozen=True)
class ClusterReport:
    """一次 k=2 聚类的产物。**簇不是畴**——要不要当成两个畴由人确认(设计 D10)。"""

    available: bool
    reason: str = CLUSTER_OK
    next_step: str = ""
    n_comparable: int = 0
    symmetry_deg: "float | None" = None
    symmetry_source: str = ""
    w_angle: float = 1.0
    w_period: float = 1.0
    weights_source: str = ""
    #: ``[{"name","seed_index","members","intra_max","representative_frame",
    #:    "fingerprint"}, ...]``
    clusters: list[dict] = field(default_factory=list)
    inter_min: "float | None" = None
    seed_distance: "float | None" = None
    intra_max: "float | None" = None
    separation_ratio: "float | None" = None
    ratio_reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "reason": self.reason,
            "next_step": self.next_step,
            "n_comparable": self.n_comparable,
            "symmetry_deg": self.symmetry_deg,
            "symmetry_source": self.symmetry_source,
            "w_angle": self.w_angle,
            "w_period": self.w_period,
            "weights_source": self.weights_source,
            "clusters": list(self.clusters),
            "inter_min": self.inter_min,
            "seed_distance": self.seed_distance,
            "intra_max": self.intra_max,
            "separation_ratio": self.separation_ratio,
            "ratio_reason": self.ratio_reason,
        }


def cluster_two(items: "list[dict]", *, symmetry_deg: "float | None",
                w_angle: float = 1.0, w_period: float = 1.0,
                symmetry_source: str = "", weights_source: str = "",
                ) -> ClusterReport:
    """把 M 个指纹分成两簇。纯函数:不碰硬件、不读配置、不抛异常。

    ``items``: ``[{"index", "fingerprint": [[角度,周期,功率],...], "frame": 路径}, ...]``,
    只放**可比**的指纹(不可比的连距离都算不出来,混进来只会污染分母)。

    做法逐字照设计 D10 第 2 步:**取距离最远的一对当种子,其余归最近**。这是
    「无监督聚类」不是「分类」——两个簇的名字是 ``cluster_1``/``cluster_2``,
    刻意不叫 A/B:没有人确认过之前,代码里出现一个畴的名字就已经越界了。

    ⚠️ **分母比分子重要。** 每簇只有一帧时同簇最大距离**没有样本**,分离度报
    ``None`` 并写明原因,不报 ``inf``,也不拿 0 当分母凑一个大得好看的数。
    """
    n = len(items or ())
    if symmetry_deg is None or not (float(symmetry_deg) > 0):
        return ClusterReport(
            available=False, reason=CLUSTER_NO_SYMMETRY, n_comparable=n,
            symmetry_source=symmetry_source or "none",
            next_step=("聚类要在晶格对称性的商空间里算距离,而 symmetry_deg"
                       "(六角 60、矩形 90)是**样品事实**,代码不猜。"
                       "把它作为参数传进来,或者先建一份参照系。"
                       "指纹已经照报,拿去人工比对不受影响。"))
    if n < 2:
        return ClusterReport(
            available=False, reason=CLUSTER_TOO_FEW, n_comparable=n,
            symmetry_deg=float(symmetry_deg),
            symmetry_source=symmetry_source or "parameter",
            next_step=("可比的指纹少于 2 个,分不出两簇。先看每个点的 verdict_reason:"
                       "scale_gate 要缩视野/加像素,no_atomic_phase 要换点或修针,"
                       "unknown_frame_angle 要补角度来源。"))

    from mast.vision.domain_phase import peak_distance

    sym = float(symmetry_deg)
    kw = dict(symmetry_deg=sym, w_angle=float(w_angle), w_period=float(w_period))
    peaks = [tuple(tuple(float(v) for v in p) for p in it["fingerprint"])
             for it in items]

    dist: dict[tuple[int, int], float] = {}
    for i in range(n):
        for j in range(i + 1, n):
            d = peak_distance(peaks[i], peaks[j], **kw)
            if d is not None:
                dist[(i, j)] = float(d)

    def _d(i: int, j: int) -> "float | None":
        if i == j:
            return 0.0
        return dist.get((i, j) if i < j else (j, i))

    if not dist:
        return ClusterReport(
            available=False, reason=CLUSTER_TOO_FEW, n_comparable=n,
            symmetry_deg=sym, symmetry_source=symmetry_source or "parameter",
            next_step="任何一对指纹之间都算不出距离(峰表是空的)——先查指纹提取。")

    (a, b), seed_d = max(dist.items(), key=lambda kv: kv[1])
    members: list[list[int]] = [[a], [b]]
    for i in range(n):
        if i in (a, b):
            continue
        da, db = _d(i, a), _d(i, b)
        if da is None and db is None:
            continue
        if db is None or (da is not None and da <= db):
            members[0].append(i)
        else:
            members[1].append(i)

    def _intra(group: list[int]) -> "float | None":
        vals = [v for x in range(len(group)) for y in range(x + 1, len(group))
                if (v := _d(group[x], group[y])) is not None]
        return max(vals) if vals else None

    intra = [_intra(g) for g in members]
    cross = [v for i in members[0] for j in members[1]
             if (v := _d(i, j)) is not None]
    inter_min = min(cross) if cross else None

    clusters: list[dict] = []
    for k, group in enumerate(members):
        seed = group[0]
        clusters.append({
            "name": f"cluster_{k + 1}",
            "seed_index": items[seed].get("index"),
            "members": [items[i].get("index") for i in sorted(group)],
            "size": len(group),
            "intra_max": intra[k],
            "representative_frame": items[seed].get("frame"),
            "fingerprint": [list(p) for p in peaks[seed]],
        })

    known_intra = [v for v in intra if v is not None]
    intra_max = max(known_intra) if known_intra else None
    ratio: "float | None" = None
    ratio_reason = ""
    if inter_min is None:
        ratio_reason = "两簇之间一对距离都算不出来 —— 分子缺席。"
    elif intra_max is None:
        # D6 的原话:分母比分子重要。两个单点簇之间的距离再大也说明不了分离度,
        # 因为「同一个畴的两帧能差多少」这个数根本没有测过。
        ratio_reason = ("每簇只有一帧,同簇最大距离**没有样本** —— 分离度的分母"
                        "缺席,比值无从谈起。同点连扫几帧(或多铺几个点)才有分母。")
    elif intra_max <= 0:
        ratio_reason = "同簇最大距离是 0(同簇只有重复的同一份指纹),不拿 0 当分母。"
    else:
        ratio = float(inter_min) / float(intra_max)

    return ClusterReport(
        available=True, reason=CLUSTER_OK, n_comparable=n, symmetry_deg=sym,
        symmetry_source=symmetry_source or "parameter",
        w_angle=float(w_angle), w_period=float(w_period),
        weights_source=weights_source or "uncalibrated_equal_weights",
        clusters=clusters, inter_min=inter_min, seed_distance=float(seed_d),
        intra_max=intra_max, separation_ratio=ratio, ratio_reason=ratio_reason,
        next_step=("看两簇的代表帧:它们是两个畴 / 是同一个畴的漂移 / 其中一簇是"
                   "坏针?确认之后才谈得上固化成参照系 —— 这一步只能由人来。"))


# ── 尺度预检 ──────────────────────────────────────────────────────────────────

def scale_refusal(size_m: float, pixels: int) -> "dict[str, Any] | None":
    """下发扫描**之前**的尺度拒绝;过得了满权重档返回 ``None``。

    判定归 ``atomic_phase.plan_scale``(0.02 / 0.05 两个阈值的归属地),这里只负责
    措辞与两条**算得出来的**替代路。**绝不静默缩帧**:帧宽是调用方的意图,偷偷改掉
    等于换了被测对象。

    ⚠️ 加像素那条路必须**同比加线时**:提像素不提线时会让 ``nm/px`` 变好看而每像素
    驻留砍半——尺度门是被骗过去的,不是真的过了。
    """
    from mast.vision.atomic_phase import (
        SCALE_FULL_NMPP,
        min_pixels_for_scale,
        plan_scale,
    )

    nmpp, scale, problem = plan_scale(size_m, pixels)
    if scale == "full":
        return None

    alternatives: list[str] = []
    need_px = min_pixels_for_scale(size_m)
    if need_px:
        alternatives.append(
            f"这个视野({(size_m or 0.0) * 1e9:g} nm)要 {need_px} px 以上"
            f"(线时必须同比加大,守住每像素驻留)")
    try:
        px = int(pixels)
    except (TypeError, ValueError):
        px = 0
    if px > 0:
        alternatives.append(
            f"保持 {px} px 的话,视野要小于 {px * SCALE_FULL_NMPP:g} nm")
    return {
        "code": REFUSE_SCALE,
        "detail": (problem or "这组帧参数进不了原子判据的满权重档")
                  + " 一帧都没有发出去:扫一张判不了的图,花的不只是机时——"
                    "流程会把「判不了」读成「这里没有畴」。",
        "alternatives": alternatives,
        "nm_per_px": nmpp,
        "scale": scale,
        # 原样回传 —— 证明这是一次拒绝,不是一次悄悄的改写。
        "frame_size_m": size_m,
        "pixels": pixels,
    }


# ── 二分:几何(纯函数,零 IO) ─────────────────────────────────────────────────

def locate_tolerance(frame_size_m: float,
                     operator_tolerance_m: "float | None" = None,
                     ) -> tuple[float, bool, str]:
    """二分的终止容差 = ``max(frame_size_m, operator_tolerance_m)``,外加它被夹过没有。

    **这一条的物理下限是帧宽,不是压电分辨率。** 两个中心相距小于一个帧宽的帧看的
    是**同一片表面**,指纹必然相同 —— 二分走到那里就自然停住了,再往下的每一步都是
    在把一张图和它自己比。压电分辨率(sub-pm)在这里根本不是约束,它比帧宽小三四个
    数量级,拿它当终止条件只会让状态机空转到 ``max_iterations`` 为止,而且每一次空转
    都真的去扫了一帧。

    要再往下只有一条路:**缩小帧**。而缩小帧会改 nm/px、改 ``scale_gate`` 的档位、
    改 ``angular_concentration`` 的环像素数 —— **那是换一套判据,不是调一个数**。
    所以这里不接受比帧宽还小的容差,只会把它夹上去并说明白;想要更细的定位,先决定
    用什么帧去看,再回来。**这段话写在这里,是因为下一个想「再调细一点」的人会先
    打开这个函数**(设计 D9)。

    返回 ``(tolerance_m, clamped, why)``。``clamped`` 为 True 表示用户给的数被帧宽
    顶上去了 —— 这件事必须出现在报告里,否则下一个人只会看到「收敛了」,以为自己要
    的那个精度真的达到了。
    """
    frame = float(frame_size_m)
    if not (frame > 0):
        raise ValueError(f"帧尺寸必须为正,收到 {frame_size_m!r}")
    try:
        want = float(operator_tolerance_m) if operator_tolerance_m else 0.0
    except (TypeError, ValueError):
        want = 0.0
    if want <= 0:
        return frame, False, (
            f"定位容差没有指定,取帧宽 {frame * 1e9:.1f} nm —— 这是物理下限:"
            f"两个中心相距小于一个帧宽的帧看的是同一片表面,指纹必然相同。")
    if want < frame:
        return frame, True, (
            f"定位容差 {want * 1e12:.3g} pm 比帧宽 {frame * 1e9:.1f} nm 还小,"
            f"已夹到帧宽。**这不是压电分不出来**(压电还能细三四个数量级),是"
            f"两个相距不到一个帧宽的帧看的就是同一片表面 —— 指纹必然相同,"
            f"再分下去每一帧都是在和自己比。要更细的定位只能换更小的帧,"
            f"而那会改 nm/px、改 scale_gate 档位、改环像素数:是换一套判据,"
            f"不是调一个数。")
    return want, False, (
        f"定位容差 {want * 1e9:.1f} nm(用户给的),大于帧宽 "
        f"{frame * 1e9:.1f} nm,按用户的数收敛。")


def bracket_gap_m(lo: "tuple[float, float]", hi: "tuple[float, float]") -> float:
    """两个端点的直线距离(米)。二分的进度就是这个数在减半。"""
    return math.hypot(float(hi[0]) - float(lo[0]), float(hi[1]) - float(lo[1]))


def midpoint(lo: "tuple[float, float]",
             hi: "tuple[float, float]") -> tuple[float, float]:
    """``mid = (p_lo + p_hi) / 2``(设计 D9)。"""
    return ((float(lo[0]) + float(hi[0])) / 2.0,
            (float(lo[1]) + float(hi[1])) / 2.0)


def offset_probe(lo: "tuple[float, float]", hi: "tuple[float, float]", *,
                 frame_size_m: float, attempt: int) -> tuple[float, float]:
    """第 ``attempt`` 次「判不了」偏移点(从 1 起数)。**确定性:同样的输入永远同一个点。**

    沿 ``lo→hi`` 的**垂直**方向偏移整数个帧宽,左右交替:第 1 次 +1 帧宽、第 2 次
    −1 帧宽、第 3 次 +2 帧宽、第 4 次 −2 帧宽……沿轴位置**始终是中点**。

    选垂直方向不是为了好看:畴界搜索的全部信息都在 ``lo→hi`` 这个轴上,垂直移动
    **不改变「这个点落在 lo 与 hi 之间的哪一处」**,所以偏移点判出来的 label 照样能
    拿来收窄区间 —— 沿轴偏移就不行,那等于换了一个中点,二分的不变式当场就散了。

    左右交替而不是一直往一边:畴界往中点的哪一侧让开是未知的,交替两侧的期望代价
    更低,而且**不需要任何关于畴界走向的假设**。

    这个函数存在本身是一条纪律:「判不了,下一个点扫哪儿」是一个**几何问题**,答案
    由 lo、hi、帧宽三个数唯一确定 —— 不该去问语言模型,也不该由它给一个坐标。
    """
    mx, my = midpoint(lo, hi)
    dx = float(hi[0]) - float(lo[0])
    dy = float(hi[1]) - float(lo[1])
    norm = math.hypot(dx, dy)
    if norm <= 0:
        # 两端重合:没有「沿轴」方向可言。退化成沿 +x 排开 —— 仍然是确定性的,
        # 而且这种 bracket 早就在收敛判据那里停住了,走不到这里。
        ux, uy = 0.0, 1.0
    else:
        ux, uy = -dy / norm, dx / norm      # lo→hi 逆时针转 90°
    k = max(1, int(attempt))
    magnitude = float((k + 1) // 2) * float(frame_size_m)
    side = 1.0 if (k % 2) else -1.0
    return (mx + side * magnitude * ux, my + side * magnitude * uy)


def probe_position_problem(x: float, y: float, *, cfg,
                           circles: "list | tuple" = ()) -> "str | None":
    """二分落点能不能扫。**只查压电范围与避让圈;刻意不查重叠**(设计陷阱 4)。

    ⚠️ 二分落点**不走** ``pick_next_position``。那条路的 ``reuse_overlap_frac`` 会把
    中点丢掉:中点与两个端点帧的重叠几乎必然 ≥30%。二分要的正是「回到已经扫过的两
    个地方中间再扫一帧」,而 ``pick_next_position`` 的整个用途是「找**没**扫过的地
    方」—— 两种语义共用一个函数,后果是二分永远排不出下一个点,而且**零报错**:它只
    会说「这片表面在当前策略下没有位置了」,听起来像表面用完了。

    「已扫 ≠ 不可用」在这一层也成立:``coverage_stats`` 的 ``usable_frac`` 本来就不
    减已扫面积,细化点不消耗表面预算。

    压电范围用的是 ``piezo_half_range_m``(硬边界),不是 ``effective_half_range_m``
    ——后者含 6% 的边缘余量,那是**排路线**时的客气,不是硬件限制;一个直接给出来的
    坐标要过的是硬边界。
    """
    half = max(float(cfg.frame_size_m) / 2.0, 0.0)
    limit = float(cfg.piezo_half_range_m)
    if abs(float(x)) + half > limit or abs(float(y)) + half > limit:
        return (f"落点 ({float(x) * 1e9:.0f}, {float(y) * 1e9:.0f}) nm 的帧会伸出压电"
                f"范围(半程 {limit * 1e9:.0f} nm,帧半宽 {half * 1e9:.1f} nm)。")
    # 与既有几何共用同一份实现 —— 这三行在本仓已经有一份,再写一遍就是第二个真源。
    from mast.io.map_analysis import _frame_hits_circle

    for c in circles or ():
        if _frame_hits_circle(float(x), float(y), half, c):
            return (f"落点 ({float(x) * 1e9:.0f}, {float(y) * 1e9:.0f}) nm 的帧压在"
                    f"避让区上({c.label or c.kind},半径 {c.radius_m * 1e9:.0f} nm)。")
    return None


# ── 二分:代次(设计 D9b) ────────────────────────────────────────────────────

def check_bracket_epochs(lo_epoch: "int | None", hi_epoch: "int | None",
                         current: "int | None") -> "str | None":
    """两个端点能不能一起被二分。返回问题描述;``None`` = 可以。

    D9b 第 1 条:**bracket 的两个端点必须同代次才允许二分**。两个端点如果分属粗动
    前后,它们之间根本不存在一条连续的表面 —— 「在两者之间取中点」这句话没有定义,
    而算出来的那个中点是一个看上去完全正常的米坐标。

    ``None`` 代次(读不到 / 旧格式行)**不当陈旧**,但也不当「对得上」:两端都没有章
    时放行并由调用方去说「这一轮没有代次保护」,一端有一端没有就是拒绝 —— 那是真的
    分不清,而分不清时二分是不安全的一边。
    """
    if lo_epoch is None and hi_epoch is None:
        return None
    if lo_epoch is None or hi_epoch is None:
        return ("两个端点里只有一个盖了坐标代次章"
                f"(lo={lo_epoch!r}, hi={hi_epoch!r})—— 分不清它们是不是同一片表面上"
                "的两个点。分不清的时候不许二分:算出来的中点会是一个看上去完全"
                "正常的米坐标。")
    if int(lo_epoch) != int(hi_epoch):
        return (f"两个端点属于不同的坐标代次(lo 第 {int(lo_epoch)} 代、"
                f"hi 第 {int(hi_epoch)} 代)—— 中间发生过粗动,它们指的是**两片不同的"
                f"表面**,「在两者之间」这句话没有定义。**不做跨代次换算**:"
                f"粗动步进是开环的,换算出来的坐标看上去和真坐标一样,而它是编的。")
    if current is not None and int(lo_epoch) != int(current):
        return (f"这一对端点属于第 {int(lo_epoch)} 代坐标系,当前已是第 {int(current)} "
                f"代 —— 粗动之后这两个米坐标指向另一片表面。请按当前代次重新找一对"
                f"端点。**不做跨代次换算**。")
    return None


def cross_site_downgrade(site_lo: int, site_hi: int, *,
                         uncertainty_steps: "int | None" = None,
                         label_lo: str = "", label_hi: str = "") -> dict:
    """跨站点畴界的**诚实输出形态**:步 + 不确定度,**永远不是米坐标**(设计 D9b 第 3 条)。

    站点 k 全是一个相、站点 k+1 全是另一个相,这时畴界确实**存在于两站之间**——但
    「之间」是用**粗动步数**量的,不是用米量的。压电是米/单代次的,粗动是步/跨代次
    的;开环步长随驱动幅度、负载、温度漂移,同样 100 步在 300 K 能比 4 K 远五倍。
    ``xy_motor_step_m`` 只是个标注,本仓没有任何东西拿它计算 —— 把它乘上步数报成
    「畴界在 (x, y)」就是凭空造一个坐标出来,而它看上去和真坐标一模一样。

    所以这个函数**不返回任何米制字段**,一个都没有。要拿这份报告去驱动下一步,唯一
    可执行的动作是:在两站之间做一次**半步长**粗动把新站点插进去,再在新站点内铺粗
    网格。⚠️ 顺序铺,不能跳着来 —— 粗动记账对间距的要求随 ``gap_moves`` 增长,
    相邻插入放行,回到很早的站点附近会被拒。
    """
    lo, hi = int(site_lo), int(site_hi)
    return {
        "kind": "cross_site_observation",
        "between_sites": [lo, hi],
        "uncertainty_steps": (None if uncertainty_steps is None
                              else int(uncertainty_steps)),
        "label_at_lower_site": label_lo or None,
        "label_at_upper_site": label_hi or None,
        "localised": False,
        "why_not_metres": (
            "压电是米/单代次的,粗动是步/跨代次的。开环步长随驱动幅度、负载、温度"
            "漂移(同样 100 步在 300 K 能比 4 K 远五倍),所以跨站点的位置只能用步"
            "来说。把步数乘一个标称步长报成米坐标,得到的是一个看上去和真坐标一模"
            "一样的编造值 —— 本仓刻意没有跨代次坐标换算。"),
        "next_step": (
            f"在站点 {lo} 与 {hi} 之间做一次**半步长**粗动,把新站点插进去,再在新"
            f"站点内铺粗网格(mode='survey')。顺序铺,不能跳着来:粗动记账对间距的"
            f"要求随 gap_moves 增长,相邻插入放行,回到很早的站点附近会被拒。"),
    }


# ── 二分:确定性状态机(设计 D9) ─────────────────────────────────────────────

@dataclass(frozen=True)
class BisectProbe:
    """状态机要求采的下一个点。``role`` 直接落进 marker 的 ``meta.role``。"""

    role: str
    x_m: float
    y_m: float
    iteration: int
    gap_m: float
    why: str = ""


@dataclass
class BisectMachine:
    """一个 bracket 的二分状态机。**纯逻辑:不碰硬件、不读库、不叫模型。**

    驱动方式是两个方法轮流调:``plan_probe()`` 给下一个要采的点(``None`` = 到终态
    了),采完把判定交给 ``record()``。中途发现粗动就调 ``invalidate()``。

    **不存游标。** 这个对象是一次运行内的工作副本,真源是 marker
    (:func:`rebuild_brackets` 从 marker 把它重建回来)—— composite 的 sidecar 按
    ``(name, run_id)`` 分键,重启就是新 run_id,靠 sidecar 续跑等于没有续跑。

    ``lo_label`` / ``hi_label`` 是两端各自判出来的**畴名**,必须互不相同:如果两端
    是同一个相,「在两者之间找分界」这句话没有定义。
    """

    bracket_id: str
    lo: tuple[float, float]
    hi: tuple[float, float]
    lo_label: str
    hi_label: str
    tolerance_m: float
    frame_size_m: float
    max_iterations: int = DEFAULT_MAX_ITERATIONS
    offset_budget: int = DEFAULT_OFFSET_BUDGET
    state: str = STATE_BRACKETED
    end_reason: str = ""
    iterations: int = 0
    offsets_used: int = 0
    #: 连续判不了的次数。0 = 干净的中点,1 = 该同点重采(第一级),≥2 = 该走偏移点。
    undetermined_streak: int = 0
    boundary: "tuple[float, float] | None" = None
    boundary_uncertainty_m: "float | None" = None
    note: str = ""
    seed_lo: "tuple[float, float] | None" = None
    seed_hi: "tuple[float, float] | None" = None
    history: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.lo = (float(self.lo[0]), float(self.lo[1]))
        self.hi = (float(self.hi[0]), float(self.hi[1]))
        if self.seed_lo is None:
            self.seed_lo = self.lo
        if self.seed_hi is None:
            self.seed_hi = self.hi

    @property
    def gap_m(self) -> float:
        return bracket_gap_m(self.lo, self.hi)

    @property
    def finished(self) -> bool:
        return self.state in TERMINAL_STATES

    def plan_probe(self) -> "BisectProbe | None":
        """下一个要采的点;``None`` 表示到终态了(``state`` 说明是哪一个)。

        终止判断集中在这里,只有这一处 —— ``record()`` 刻意不判「该不该停」,否则
        「什么时候收敛」会有两个真源,而它们迟早会不一致。
        """
        if self.finished:
            return None
        gap = self.gap_m
        if gap <= self.tolerance_m:
            self._finish(STATE_CONVERGED, END_TOLERANCE)
            return None
        if self.iterations >= int(self.max_iterations):
            self._finish(STATE_ABANDONED, END_MAX_ITERATIONS)
            return None
        self.state = STATE_BISECTING
        it = self.iterations + 1
        if self.undetermined_streak <= 1:
            mx, my = midpoint(self.lo, self.hi)
            why = ("中点" if not self.undetermined_streak else
                   "同一个中点**重采一帧**(判不了的第一级:一帧的噪声不该判死一个位置)")
            return BisectProbe(role=ROLE_MID, x_m=mx, y_m=my, iteration=it,
                               gap_m=gap, why=why)
        if self.offsets_used >= int(self.offset_budget):
            self._finish(STATE_ABANDONED, END_UNDETERMINED)
            return None
        k = self.offsets_used + 1
        ox, oy = offset_probe(self.lo, self.hi,
                              frame_size_m=self.frame_size_m, attempt=k)
        return BisectProbe(
            role=ROLE_OFFSET, x_m=ox, y_m=oy, iteration=it, gap_m=gap,
            why=(f"判不了的第二级:沿 lo→hi 的**垂直**方向偏移第 {k} 档"
                 f"(左右交替,每档一个帧宽)。垂直偏移不改变沿轴位置,"
                 f"所以它判出来的 label 照样能收窄区间。"))

    def record(self, probe: BisectProbe, *, verdict: str,
               label: "str | None" = None) -> None:
        """把一次判定并进状态。**收窄一律用当时的中点**,不是 probe 的坐标。

        偏移点的坐标是中点垂直移开一个帧宽的地方 —— 它带的信息是「沿轴的这一处属于
        哪个相」,所以要收窄的是中点那一刀。拿偏移点的坐标去当新端点,区间会歪掉,
        而且歪得看不出来。
        """
        v = str(verdict or "")
        lab = (str(label) if label else "") or None
        mid = midpoint(self.lo, self.hi)
        entry = {
            "role": probe.role, "x_m": probe.x_m, "y_m": probe.y_m,
            "iteration": probe.iteration, "gap_m": probe.gap_m,
            "verdict": v, "label": lab, "narrowed": None,
        }
        self.history.append(entry)

        if v == "mixed":
            # 两个畴的峰**都在这一帧里** ⇒ 畴界就在这一帧的范围内,二分到此为止。
            # 这是找畴界时最有价值的一种判定,比继续二分收敛得更快也更直接。
            self.boundary = (float(probe.x_m), float(probe.y_m))
            self.boundary_uncertainty_m = float(self.frame_size_m) / 2.0
            entry["narrowed"] = "converged_on_a_mixed_frame"
            self._finish(STATE_CONVERGED, END_MIXED)
            return
        if lab and lab == self.lo_label:
            self.lo = mid
            self.iterations += 1
            self.undetermined_streak = 0
            entry["narrowed"] = "lo"
            return
        if lab and lab == self.hi_label:
            self.hi = mid
            self.iterations += 1
            self.undetermined_streak = 0
            entry["narrowed"] = "hi"
            return
        if lab:
            # 第三个 label:既不是 lo 那一相也不是 hi 那一相。这是**证据矛盾**,
            # 不是「判不了」—— 把它归进最近的一簇会二分出一条根本不存在的边界。
            # 无人值守时走保守分支(放弃这个 bracket),不猜、也不半夜叫人。
            self.note = (f"中点判成了第三个相 {lab!r},而这个 bracket 的两端是 "
                         f"{self.lo_label!r} 与 {self.hi_label!r} —— 这是证据矛盾,"
                         f"不是判不了。可能是出现了第三个畴,也可能是参照系不全。"
                         f"该由人看这一帧,不该由二分自己归簇。")
            entry["narrowed"] = "contradiction"
            self._finish(STATE_ABANDONED, END_THIRD_LABEL)
            return
        # 判不了:走三级处理,预算够不够由下一次 plan_probe() 判。
        self.undetermined_streak += 1
        if probe.role == ROLE_OFFSET:
            self.offsets_used += 1

    def skip(self, probe: BisectProbe, *, reason: str) -> None:
        """这个落点**采不了**(超压电范围 / 压在避让圈上)—— 针尖根本没去。

        不是「判不了」:那是采到了但读不出相。但下一步是同一条(换偏移点),所以它
        同样吃偏移预算。直接把连续计数顶到 2:同一个采不了的坐标重采一帧没有意义,
        那是在为一个**几何**问题花机时。
        """
        self.history.append({
            "role": probe.role, "x_m": probe.x_m, "y_m": probe.y_m,
            "iteration": probe.iteration, "gap_m": probe.gap_m,
            "verdict": "", "label": None, "narrowed": "unreachable",
            "reason": reason,
        })
        self.undetermined_streak = max(self.undetermined_streak, 1) + 1
        if probe.role == ROLE_OFFSET:
            self.offsets_used += 1

    def invalidate(self, message: str = "") -> None:
        """粗动了 ⇒ 这个 bracket **立刻作废**,不是暂停(设计 D9b 第 2 条)。

        作废与暂停的差别不是措辞:暂停意味着「回头还能接着用这两个坐标」,而
        ``p_lo``/``p_hi`` 在粗动之后指向的是**另一片表面** —— 它们之间已经不存在
        那条畴界了。已经收敛出来的边界坐标也一并清掉:它是用旧坐标系说的话。
        """
        self.state = STATE_INVALIDATED
        self.end_reason = END_EPOCH
        self.boundary = None
        self.boundary_uncertainty_m = None
        if message:
            self.note = message

    def abandon(self, reason: str, message: str = "") -> None:
        """由外部条件放弃(端点凑不成一对、连着扫不出图)。"""
        self._finish(STATE_ABANDONED, reason)
        if message:
            self.note = message

    def _finish(self, state: str, reason: str) -> None:
        self.state = state
        self.end_reason = reason
        if state == STATE_CONVERGED and self.boundary is None:
            self.boundary = midpoint(self.lo, self.hi)
            self.boundary_uncertainty_m = self.gap_m / 2.0

    def as_dict(self) -> dict[str, Any]:
        """给报告用。``boundary_*`` 只在 ``CONVERGED`` 时有值 —— 其余终态**没有**
        一个「大概在这儿」的坐标可给,给了就是编。"""
        return {
            "bracket_id": self.bracket_id,
            "state": self.state,
            "end_reason": self.end_reason,
            "note": self.note,
            "lo_label": self.lo_label,
            "hi_label": self.hi_label,
            "seed_lo_m": list(self.seed_lo or ()),
            "seed_hi_m": list(self.seed_hi or ()),
            "lo_m": list(self.lo),
            "hi_m": list(self.hi),
            "gap_m": self.gap_m,
            "tolerance_m": float(self.tolerance_m),
            "frame_size_m": float(self.frame_size_m),
            "iterations": int(self.iterations),
            "max_iterations": int(self.max_iterations),
            "offsets_used": int(self.offsets_used),
            "offset_budget": int(self.offset_budget),
            "boundary_m": (list(self.boundary) if self.boundary is not None
                           else None),
            "boundary_uncertainty_m": self.boundary_uncertainty_m,
            "history": list(self.history),
        }


# ── 二分:状态从 marker 重建(设计 D9 末条) ────────────────────────────────────

def _marker_view(m) -> "tuple[float, float, dict, int | None] | None":
    """把一条 marker 读成 ``(x, y, meta, coord_epoch)``;读不出形状就 ``None``。

    两种形态都要吃:``map_scope.load_markers()`` 给的是 ``MapMarker`` 对象,
    ``storage.get_markers()`` 给的是库里那一行的 dict —— 重建状态这件事在两条路上
    都要成立,而它们之间没有一个共同基类可以指望。
    """
    if isinstance(m, dict):
        x, y = m.get("x_m"), m.get("y_m")
        meta, epoch = m.get("meta"), m.get("coord_epoch")
    else:
        x, y = getattr(m, "x_m", None), getattr(m, "y_m", None)
        meta, epoch = getattr(m, "meta", None), getattr(m, "coord_epoch", None)
    if x is None or y is None or not isinstance(meta, dict):
        return None
    try:
        return float(x), float(y), meta, (None if epoch is None else int(epoch))
    except (TypeError, ValueError):
        return None


#: 重建时认为「记录的中点」与「重算的中点」对得上的容差(米)。1 pm 远在物理意义
#: 之下、远在 float64 噪声之上 —— 对不上说明**中点规则被改过**,那是要说出来的事。
_REPLAY_MATCH_M = 1e-12


def rebuild_brackets(markers, *, current_epoch: "int | None" = None) -> dict:
    """从 marker 重建每个 bracket 的状态。**这是二分状态的唯一真源**(设计 D9 末条)。

    为什么不存游标:composite 的 sidecar 按 ``(name, run_id)`` 分键,而重启就是一个
    新 run_id —— 靠 sidecar 续跑等于没有续跑。marker 是重启活得下来的那一份记录。

    ⚠️ **代次一律用权威计数,绝不从这一批行里数**(设计陷阱 6):``get_markers`` 只回
    最新 ``limit`` 行,而且截掉的是**最老**的 —— 一次搜索早期的采样点正好是最老的
    那些。所以 ``current_epoch`` 由调用方从 ``coord_epoch.read_current_epoch()``
    拿,不在这里数。

    传进来的 marker **应当含所有代次**:只有看得见两端各自的代次,才分得清「端点跨
    代次」和「端点根本没记上」——两者的下一步完全不同(重新找一对 vs 去查记录链)。

    返回 ``{bracket_id: {...}}``。``problems`` 非空 = 这个 bracket **不能**直接拿去
    二分,里面写着为什么。
    """
    groups: dict[str, list] = {}
    for m in markers or ():
        view = _marker_view(m)
        if view is None:
            continue
        x, y, meta, epoch = view
        bid = meta.get("bracket_id")
        if not bid:
            continue                      # 普查的种子点不属于任何 bracket
        groups.setdefault(str(bid), []).append((x, y, meta, epoch))
    return {bid: _replay_bracket(bid, rows, current_epoch)
            for bid, rows in groups.items()}


def _determinate_label(meta: dict) -> "str | None":
    """这条 marker 判出了一个畴名没有。``mixed`` / ``undetermined`` 都**不是**畴名。"""
    v = str(meta.get("verdict") or "")
    if not v or v in ("undetermined", "mixed"):
        return None
    return v


def _replay_bracket(bracket_id: str, rows: list, current_epoch: "int | None") -> dict:
    """按记录的时间顺序重放一个 bracket。纯函数:只吃 rows,不查任何东西。

    重放而不是「读一个存下来的区间」:存下来的区间会和记录不一致(而且不一致时没有
    任何东西会喊),重放则是**由记录本身定义**当前区间 —— 记录错了,重放的结果就是
    错的,这是对的:那说明记录链断了,该去修那个,不是在这里兜底。
    """
    ends: dict[str, tuple] = {}
    probes: list[tuple] = []
    for x, y, meta, epoch in rows:
        role = str(meta.get("role") or "")
        if role in (ROLE_LO, ROLE_HI):
            ends.setdefault(role, (x, y, meta, epoch))   # 端点是种子,第一条为准
        elif role in (ROLE_MID, ROLE_OFFSET):
            probes.append((x, y, meta, role))

    out: dict[str, Any] = {
        "bracket_id": bracket_id,
        "n_markers": len(rows),
        "n_probes": len(probes),
        "problems": [],
        "replay_notes": [],
        "state": STATE_SEEDING,
        "end_reason": "",
    }
    if ROLE_LO not in ends or ROLE_HI not in ends:
        missing = [r for r in (ROLE_LO, ROLE_HI) if r not in ends]
        out["problems"].append(
            f"这个 bracket 的记录里缺端点 {missing} —— 只有两端都在,「在两者之间」"
            f"才有定义。先跑一次 survey 拿到一对指纹不同的点。")
        return out

    lo_x, lo_y, lo_meta, lo_epoch = ends[ROLE_LO]
    hi_x, hi_y, hi_meta, hi_epoch = ends[ROLE_HI]
    out.update({
        "seed_lo_m": [lo_x, lo_y], "seed_hi_m": [hi_x, hi_y],
        "lo_epoch": lo_epoch, "hi_epoch": hi_epoch,
        "current_epoch": current_epoch,
        "lo_label": _determinate_label(lo_meta),
        "hi_label": _determinate_label(hi_meta),
        "reference_version": lo_meta.get("reference_version"),
        "domain_search_id": lo_meta.get("domain_search_id"),
    })

    epoch_problem = check_bracket_epochs(lo_epoch, hi_epoch, current_epoch)
    out["epoch_problem"] = epoch_problem or ""
    if epoch_problem:
        out["problems"].append(epoch_problem)
        out["state"] = STATE_INVALIDATED
        out["end_reason"] = END_EPOCH
        return out

    lo_label, hi_label = out["lo_label"], out["hi_label"]
    if not lo_label or not hi_label:
        out["problems"].append(
            f"两端里有判不出畴名的(lo={lo_label!r}, hi={hi_label!r})—— 二分要求两端"
            f"各自判出一个相。没有标定过的参照系时每个点都是 undetermined,"
            f"那时先要做的是建参照系,不是二分。")
        out["end_reason"] = END_NOT_A_PAIR
        return out
    if lo_label == hi_label:
        out["problems"].append(
            f"两端判的是同一个相 {lo_label!r} —— 它们之间没有畴界可找。"
            f"换一对指纹不同的点。")
        out["end_reason"] = END_NOT_A_PAIR
        return out

    lo, hi = (lo_x, lo_y), (hi_x, hi_y)
    iterations = offsets_used = undetermined_streak = 0
    state, end_reason = STATE_BRACKETED, ""
    boundary: "tuple[float, float] | None" = None
    for x, y, meta, role in probes:
        if state in TERMINAL_STATES:
            out["replay_notes"].append(
                "记录里在终态之后还有采样点 —— 多半是同一个 bracket_id 被复用了。")
            break
        mid = midpoint(lo, hi)
        if role == ROLE_MID and (abs(x - mid[0]) > _REPLAY_MATCH_M
                                 or abs(y - mid[1]) > _REPLAY_MATCH_M):
            out["replay_notes"].append(
                f"记录里的中点 ({x * 1e9:.3f}, {y * 1e9:.3f}) nm 与按记录重算的中点 "
                f"({mid[0] * 1e9:.3f}, {mid[1] * 1e9:.3f}) nm 对不上 —— 中点规则被改过,"
                f"或者这条记录不属于这个 bracket。重放按**重算的**中点继续。")
        state = STATE_BISECTING
        verdict = str(meta.get("verdict") or "")
        label = _determinate_label(meta)
        if verdict == "mixed":
            boundary = (x, y)
            state, end_reason = STATE_CONVERGED, END_MIXED
            continue
        if label == lo_label:
            lo, iterations, undetermined_streak = mid, iterations + 1, 0
        elif label == hi_label:
            hi, iterations, undetermined_streak = mid, iterations + 1, 0
        elif label:
            state, end_reason = STATE_ABANDONED, END_THIRD_LABEL
        else:
            undetermined_streak += 1
            if role == ROLE_OFFSET:
                offsets_used += 1

    out.update({
        "lo_m": [lo[0], lo[1]], "hi_m": [hi[0], hi[1]],
        "gap_m": bracket_gap_m(lo, hi),
        "iterations": iterations,
        "offsets_used": offsets_used,
        "undetermined_streak": undetermined_streak,
        "state": state,
        "end_reason": end_reason,
        "boundary_m": (list(boundary) if boundary is not None else None),
    })
    return out


# ── 技能 ──────────────────────────────────────────────────────────────────────

class SearchDomainBoundary(CompositeSkillGraph):
    """铺粗网格找畴界:逐点扫原子帧、抽指纹、聚类给人看。本轮只有 survey。"""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="SearchDomainBoundary",
            version="1.0.0",
            category=SkillCategory.COMPOSITE,
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "在多畴 / 孪晶 / 多相表面上找**畴界**。两个模式。"
                "mode='survey' 在压电范围上铺一张**粗网格**，每个格点扫一张"
                "原子分辨帧，各抽一次方向-周期指纹，再报一次**无监督**的"
                "两簇聚类给人看。它**不**给畴命名：说出「这是 A 相」需要一份"
                "用户确认过的参照系，而建起那份参照系正是普查的产物。"
                "没有参照系时每个点回来都是 undetermined/no_reference "
                "**外加**它的指纹 —— 那是正确的首轮结果，不是失败。"
                "mode='bisect' 在两个指纹已经判成两个**不同**畴名的点"
                "**之间**定位畴界，做法是反复扫中点。所以它**要求**参照系"
                "已经标定，没有就拒绝。"
                "它停在**一个帧宽**上，不是停在压电分辨率上：两个中心相距"
                "不到一个帧宽的帧，看的是同一片表面。"
                "二分途中发生一次粗动，那个 bracket **当场作废**"
                "（它的米坐标此刻指向的是另一片表面）—— 它从不「暂停再续」。"
                "allow_coarse_move=True（跨粗动站点搜索）**没有实现**，"
                "而且是**显式拒绝**，不是悄悄降级；跨站点的畴界只能诚实地"
                "报成「在站点 k 与 k+1 之间，+/-N 步」，永远不报成一个米坐标。"
                "比原子尺度门更粗的帧，在**发出任何一次扫描之前**就被拒绝；"
                "扫描角读不出来的帧一律报 undetermined —— 绝不按 0 度处理。"
            ),
            parameters=[
                ParameterSpec(
                    name="mode", type="str", required=True,
                    allowed_values=list(ALL_MODES),
                    description=(
                        "'survey' 铺粗网格，报指纹 + 聚类 —— **先做这个**，"
                        "参照系也是这一步建起来的。'bisect' 在两个已知分属"
                        "两个**不同**畴名的点之间定位畴界；它需要一份标定好的"
                        "参照系，外加一个 bracket_id 或者一对显式的 lo/hi。")),
                ParameterSpec(
                    name="frame_size_m", type="float", unit="m", required=False,
                    min_value=1e-10, max_value=1e-7,
                    description=(
                        "每张原子帧的边长，单位是**米**（SI），**不是纳米**："
                        "5 nm -> 5n。留空则取工作流的默认值（5 nm）。"
                        "每一帧都必须过得了原子尺度门（< 0.02 nm/px，"
                        "**严格小于** —— atomic_phase.py:266 是 "
                        "`nm_per_px < SCALE_FULL_NMPP`，恰好 0.02 会落到 "
                        "reduced 而被拒），"
                        "这一条在扫任何东西之前就查。")),
                ParameterSpec(
                    name="pixels", type="int", unit="px", required=False,
                    min_value=16, max_value=4096,
                    description=(
                        "每行的像素数。留空就由用户那张按尺度分档的策略表"
                        "决定 —— **只有**用户点了名的时候才传它。不管最后"
                        "定成多少，都会在第一次扫描之前过一遍尺度检查。")),
                ParameterSpec(
                    name="grid_pitch_m", type="float", unit="m", required=False,
                    min_value=1e-9, max_value=1e-4,
                    description=(
                        "网格点之间的间距，单位是**米**（SI）。留空则取可用"
                        "压电半程的四分之一。目标不是覆盖，而是**碰到两个"
                        "不同的指纹**，所以这个数按可达区域的尺度定，"
                        "不按帧的尺度定。")),
                ParameterSpec(
                    name="max_points", type="int", required=False, default=9,
                    min_value=2, max_value=36,
                    description=(
                        "预算：最多扫几个网格点。每一张原子帧都要花掉几分钟，"
                        "所以 36 个点就是好几个小时的机时 —— 这个数是**预算**，"
                        "不是目标。少于 2 个就聚不了类。")),
                ParameterSpec(
                    name="symmetry_deg", type="float", unit="deg", required=False,
                    min_value=1.0, max_value=360.0,
                    description=(
                        "**这个样品**的晶格对称性（六角 60，矩形 90）。"
                        "指纹之间要能比对，全靠它 —— 它是**样品事实**，"
                        "绝不去猜。不填、又没有载入参照系时，指纹照样报出来，"
                        "但聚类会跳过，理由写作 'no_symmetry_deg'。")),
                ParameterSpec(
                    name="reference_version", type="str", required=False,
                    default="",
                    description=(
                        "拿来判读的畴参照系版本（例如 'v001'）。留空 = 取最新"
                        "的那一份，或者干脆一份都没有。指定的版本不存在时由"
                        "加载器拒绝，绝不悄悄换成另一份。")),
                ParameterSpec(
                    name="sample", type="str", required=False, default="",
                    description=(
                        "样品名。只有好几个样品的参照系同时存在时才需要"
                        "（那种情况下它会拒绝，而不是去猜）。")),
                ParameterSpec(
                    name="bracket_id", type="str", required=False, default="",
                    description=(
                        "mode='bisect' 时：要接着做的那个 bracket 的 id。"
                        "它的状态（两个端点、到目前为止取过的每一个中点、"
                        "以及它们各自的判定）是**从地图 marker 重建出来的**，"
                        "所以一次二分扛得住重启。给这个，**或者**给一对"
                        "显式的 lo/hi。")),
                ParameterSpec(
                    name="lo_x_m", type="float", unit="m", required=False,
                    min_value=-1e-4, max_value=1e-4,
                    description=(
                        "mode='bisect' 时：**第一个**端点的 x，单位是**米**"
                        "（SI），**不是纳米**：300 nm -> 300n。要开一个新的 "
                        "bracket 就把 lo_x_m/lo_y_m/hi_x_m/hi_y_m 四个都传上；"
                        "两个端点会先各扫一张，而且必须判回两个**不同**的畴。")),
                ParameterSpec(
                    name="lo_y_m", type="float", unit="m", required=False,
                    min_value=-1e-4, max_value=1e-4,
                    description=(
                        "mode='bisect' 时：第一个端点的 y，单位是**米**"
                        "（SI），**不是纳米**。")),
                ParameterSpec(
                    name="hi_x_m", type="float", unit="m", required=False,
                    min_value=-1e-4, max_value=1e-4,
                    description=(
                        "mode='bisect' 时：**第二个**端点的 x，单位是**米**"
                        "（SI），**不是纳米**。")),
                ParameterSpec(
                    name="hi_y_m", type="float", unit="m", required=False,
                    min_value=-1e-4, max_value=1e-4,
                    description=(
                        "mode='bisect' 时：**第二个**端点的 y，单位是**米**"
                        "（SI），**不是纳米**。")),
                ParameterSpec(
                    name="locate_tolerance_m", type="float", unit="m",
                    required=False, min_value=1e-12, max_value=1e-6,
                    description=(
                        "mode='bisect' 时：两个端点近到这个程度就停，单位是"
                        "**米**（SI）。**会被向上夹到一个帧宽**：两个中心相距"
                        "不到一个帧宽的帧看的是同一片表面，所以它们的指纹"
                        "必然相同。这里的限制**不是**压电分辨率 —— 想再细"
                        "只能换**更小的帧**，而那会改 nm/px、改尺度门的档位、"
                        "改环像素数，也就是换一套判据，不是换一个数。")),
                ParameterSpec(
                    name="offset_budget", type="int", required=False,
                    default=DEFAULT_OFFSET_BUDGET, min_value=0, max_value=6,
                    description=(
                        "mode='bisect' 时：一个中点判不了的时候，**每个 "
                        "bracket** 最多花几张偏移帧。偏移点由几何算出来"
                        "（垂直于 lo->hi，每档一个帧宽，左右交替）—— 一个坐标"
                        "都不去问模型要。预算花完就如实地把这个 bracket 收成 "
                        "ABANDONED(undetermined)。")),
                ParameterSpec(
                    name="max_iterations", type="int", required=False,
                    default=DEFAULT_MAX_ITERATIONS, min_value=1, max_value=40,
                    description=(
                        "mode='bisect' 时：成功对折次数的上限。每折一次要花掉"
                        "一张原子帧（几分钟）。12 次能把 3 um 的压电范围收到 "
                        "5 nm 的一帧；再往下 bracket 本身已经比一个帧宽还窄了。")),
                ParameterSpec(
                    name="allow_coarse_move", type="bool", required=False,
                    default=False,
                    description=(
                        "允许这次搜索跨**粗动站点**。**没有实现**：传 true "
                        "会被拒绝并给出理由，绝不悄悄忽略。粗动会让此前收集到"
                        "的每一个米坐标失效，所以那是**另一次**搜索，不是"
                        "一次更大的搜索 —— 而且跨站点的畴界只能报成"
                        "「在站点 k 与 k+1 之间，+/-N 步」。")),
                ParameterSpec(
                    name="dry_run", type="bool", required=False, default=False,
                    description=(
                        "把整台状态机跑一遍 —— 配置、网格或 bracket、尺度预检、"
                        "代次检查、参照系载入、容差夹紧、第一个落点 —— 但"
                        "**一次扫描都不发**。给没有硬件的验收用。")),
            ],
            preconditions=["z_controller_on"],
            estimated_duration_s=1800.0,
            composition_level=3,
            tags=["domain", "boundary", "survey", "grid", "lattice", "composite"],
        )

    # ── 准备:全部的判断都在这里,一帧都还没发 ────────────────────────────

    def _prepare(self, context, params: dict) -> dict[str, Any]:
        """把「这一轮到底能不能做、怎么做」算清楚。**不发任何命令。**

        返回一个 dict;``refusal`` 非空表示这一轮不做,理由已经写好。
        """
        out: dict[str, Any] = {"refusal": None, "warnings": []}

        mode = str(params.get("mode") or "").strip().lower()
        if mode not in ALL_MODES:
            out["refusal"] = {
                "code": REFUSE_MODE_UNKNOWN,
                "detail": f"mode 只认 {list(ALL_MODES)},收到 {params.get('mode')!r}。",
                "alternatives": [
                    f"mode='{MODE_SURVEY}':铺粗网格拿指纹 + 聚类(先做这个)",
                    f"mode='{MODE_BISECT}':在一对已知指纹不同的点之间收敛到畴界"
                    f"(要求参照系已标定)",
                ],
            }
            return out
        out["mode"] = mode
        if bool(params.get("allow_coarse_move", False)):
            out["refusal"] = self._cross_site_refusal(mode)
            return out

        # ── 帧参数:先定尺寸,再问策略层要像素,最后过尺度门 ──
        frame = params.get("frame_size_m")
        frame_source = "parameter"
        if not frame:
            frame, frame_source = self._default_frame_m()
        frame = float(frame)
        out["frame_size_m"] = frame
        out["frame_size_source"] = frame_source

        pixels, est_s, pixels_source = self._resolve_pixels(params, frame)
        out["pixels"] = pixels
        out["pixels_source"] = pixels_source
        out["estimated_scan_s"] = est_s

        refusal = scale_refusal(frame, pixels)
        if refusal is not None:
            out["refusal"] = refusal
            return out
        from mast.vision.atomic_phase import plan_scale
        nmpp, scale, _ = plan_scale(frame, pixels)
        out["nm_per_px"] = nmpp
        out["scale"] = scale

        # ── 坐标代次:盖一次章,每个点之前复核一次 ──
        from mast.core.coord_epoch import read_current_epoch

        out["coord_epoch"] = read_current_epoch()
        if out["coord_epoch"] is None:
            out["warnings"].append(
                "当前坐标代次查不到 —— 这一轮没有代次保护。"
                "「查不到」不等于陈旧,也不等于当前。")

        # ── 参照系(survey 可以没有;bisect 没有就做不了,见 _prepare_bisect) ──
        out.update(self._load_reference(params))

        if out["mode"] == MODE_BISECT:
            self._prepare_bisect(context, params, out)
        else:
            self._prepare_survey(context, params, out)
        return out

    def _prepare_survey(self, context, params: dict, out: dict) -> None:
        """普查这一路:解除中心区 → 铺网格。就地改 ``out``。"""
        from mast.core.map_scope import analysis_config, load_markers

        frame = out["frame_size_m"]
        base_cfg = analysis_config(getattr(context, "state", None),
                                   frame_size_m=frame)
        cfg, why = release_center_zone(
            base_cfg, frame_size_m=frame,
            grid_pitch_m=params.get("grid_pitch_m") or None)
        out["cfg"] = cfg
        out["center_zone_reason"] = why
        out["grid_pitch_m"] = cfg.frame_size_m * cfg.point_spacing_factor
        out["pulse_r_m"] = getattr(cfg, "pulse_r_m", None)

        markers, epoch_of_markers, available = load_markers()
        out["map_history_available"] = bool(available)
        if not available:
            # 空表和「这片表面确实干净」在几何上分不开 —— 说出来,别让下游把
            # 「读不到历史」读成「没有坑」。
            out["warnings"].append(
                "读不到这片表面的历史记录(没有活动实验 / 存储不可用)—— "
                "网格避开的只有「已知」的坑,而现在一个坑都不知道。")
        if out.get("coord_epoch") is not None and epoch_of_markers is not None \
                and available and int(epoch_of_markers) != int(out["coord_epoch"]):
            out["warnings"].append(
                f"地图记录读到的代次是 {epoch_of_markers},权威代次是 "
                f"{out['coord_epoch']} —— 代次一律以权威计数为准,不从截断的记录窗口里数。")

        from mast.io.map_analysis import pick_next_positions

        want = int(params.get("max_points") or 9)
        picked = pick_next_positions(markers, cfg, want)
        out["positions"] = [(float(p.x_m), float(p.y_m)) for p in picked]
        out["position_reasons"] = [str(p.reason) for p in picked]
        out["points_requested"] = want
        if len(picked) < want:
            # 少给的那几个不许悄悄吞掉。
            out["warnings"].append(
                f"网格只排得下 {len(picked)} 个点(要了 {want} 个):"
                f"可用区被避让圈或压电范围限住了。间距调小或换个站点。")
        if not picked:
            out["refusal"] = {
                "code": REFUSE_NO_POSITIONS,
                "detail": ("解除中心区之后,整片压电范围里一个合规的网格点都排不出来。"
                           "多半是避让圈把可用区吃光了。"),
                "alternatives": [
                    f"把 grid_pitch_m 调小(当前 {out['grid_pitch_m'] * 1e9:.0f} nm)",
                    "换一个站点(粗动)之后重来",
                ],
            }

    def _prepare_bisect(self, context, params: dict, out: dict) -> None:
        """二分这一路:参照系 → 容差 → bracket → 落点闸。就地改 ``out``。

        **这条路一次都不调 ``pick_next_position*``**(设计陷阱 4)。它不铺网格:
        二分的每一个落点由 lo、hi 和帧宽算出来,是几何,不是路线规划。
        """
        if not out.get("reference_calibrated"):
            # 没有参照系时每个点都是 undetermined ⇒ 永远凑不出一对端点。让它表现成
            # 「这一对不成立」会把人送去换点,而真正该做的是先建参照系 —— 两句话
            # 指向的下一步完全不同,所以这里明说。
            out["refusal"] = {
                "code": REFUSE_NO_REFERENCE,
                "detail": (
                    "二分要求**已经标定好的畴参照系**:两个端点必须各自判出一个畴名,"
                    "而且这两个名字不一样,「在两者之间找分界」这句话才有定义。"
                    "现在没有(或参照系未标定 —— match.* 全 0 表示未标定,不是零容差),"
                    "于是每一帧都会是 undetermined(no_reference),二分永远开不了头。"),
                "alternatives": [
                    f"mode='{MODE_SURVEY}':先铺网格拿指纹与聚类,人确认两簇是两个畴,"
                    f"再固化成参照系(设计 D10)",
                    "已经有参照系文件的话,检查 reference_version / sample 是否指对了",
                ],
            }
            return

        frame = out["frame_size_m"]
        tol, clamped, why = locate_tolerance(frame, params.get("locate_tolerance_m"))
        out["locate_tolerance_m"] = tol
        out["locate_tolerance_clamped"] = bool(clamped)
        out["locate_tolerance_reason"] = why
        if clamped:
            out["warnings"].append(why)
        out["offset_budget"] = int(params.get("offset_budget")
                                   if params.get("offset_budget") is not None
                                   else DEFAULT_OFFSET_BUDGET)
        out["max_iterations"] = int(params.get("max_iterations")
                                    or DEFAULT_MAX_ITERATIONS)

        bracket, refusal = self._resolve_bracket(params, out)
        if refusal is not None:
            out["refusal"] = refusal
            return
        out["bracket"] = bracket

        # 落点闸:压电范围 + 避让圈。**不查重叠** —— 中点几乎必然与两端点帧重叠,
        # 那正是二分要的(设计陷阱 4)。用未解除中心区的原始配置就够:落点闸看的是
        # piezo_half_range_m(硬边界),中心区改的是 effective_half_range_m。
        from mast.core.map_scope import analysis_config, load_markers
        from mast.io.map_analysis import build_avoid_circles

        cfg = analysis_config(getattr(context, "state", None), frame_size_m=frame)
        markers, _epoch_of_rows, available = load_markers()
        out["cfg"] = cfg
        out["map_history_available"] = bool(available)
        out["avoid_circles"] = build_avoid_circles(markers, cfg)
        if not available:
            out["warnings"].append(
                "读不到这片表面的历史记录(没有活动实验 / 存储不可用)—— "
                "二分落点避开的只有「已知」的坑,而现在一个坑都不知道。")

    def _resolve_bracket(self, params: dict, out: dict
                         ) -> "tuple[dict | None, dict | None]":
        """定出这一轮要二分的那一对端点。返回 ``(bracket, refusal)``。

        两条来路,**语义不同**:

        * **显式 lo/hi 坐标** —— 一对刚给出来的坐标,属于当前代次(用户是现在说的
          这句话)。两端还没判过,所以状态是 ``SEEDING``:先各扫一帧,判出来是不是
          一对,再进 ``BRACKETED``。
        * **``bracket_id``** —— 从 marker 重建(设计 D9 末条)。这条路上两端各自带着
          **自己的** ``coord_epoch``,所以跨代次是看得见的,也必须当场拒(D9b 第 1 条)。
          刻意取**全部代次**的 marker:只看当前代次的话,一个跨代次的 bracket 会表现
          成「少了一个端点」,而那两句话指向的下一步完全不同。
        """
        lo_x, lo_y = params.get("lo_x_m"), params.get("lo_y_m")
        hi_x, hi_y = params.get("hi_x_m"), params.get("hi_y_m")
        given = [v for v in (lo_x, lo_y, hi_x, hi_y) if v is not None]
        bid = str(params.get("bracket_id") or "").strip()

        if given and len(given) < 4:
            return None, {
                "code": REFUSE_NO_BRACKET,
                "detail": ("端点坐标只给了一半 —— lo_x_m / lo_y_m / hi_x_m / hi_y_m "
                           "四个要么都给,要么都不给。补齐一半的坐标就是在编一个点。"),
                "alternatives": ["四个坐标都给(米制 SI)", "或者只给 bracket_id"],
            }
        if len(given) == 4:
            lo, hi = (float(lo_x), float(lo_y)), (float(hi_x), float(hi_y))
            gap = bracket_gap_m(lo, hi)
            if gap <= float(out["locate_tolerance_m"]):
                return None, {
                    "code": REFUSE_BRACKET_NOT_A_PAIR,
                    "detail": (
                        f"两个端点只相距 {gap * 1e9:.2f} nm,已经不超过定位容差 "
                        f"{float(out['locate_tolerance_m']) * 1e9:.2f} nm —— 它们看的"
                        f"是同一片表面,指纹必然相同,中间没有可二分的东西。"),
                    "alternatives": ["换一对离得更远的点(先用 survey 找)"],
                }
            return {
                "bracket_id": bid or "",     # 空 = 这一轮现开一个,id 在计划时生成
                "source": "parameters",
                "state": STATE_SEEDING,
                "lo": lo, "hi": hi, "seed_lo": lo, "seed_hi": hi,
                "lo_label": "", "hi_label": "",
                "iterations": 0, "offsets_used": 0,
                "gap_m": gap,
                "note": ("端点坐标由调用方直接给出 —— 它们按**当前代次**理解"
                         "(用户是现在说的这句话)。两端各扫一帧确认是一对之后"
                         "才开始二分。"),
            }, None

        if not bid:
            return None, {
                "code": REFUSE_NO_BRACKET,
                "detail": ("mode='bisect' 要么给 bracket_id(从 marker 重建已有的那一对),"
                           "要么给四个端点坐标 lo_x_m/lo_y_m/hi_x_m/hi_y_m。"
                           "两个都没有的话,「在两者之间」里的「两者」是谁没有定义。"),
                "alternatives": [
                    f"先跑 mode='{MODE_SURVEY}' 找一对指纹不同的点",
                    "已经有一对了就把它们的坐标传进来",
                ],
            }

        from mast.core.map_scope import load_markers

        markers, _epoch_of_rows, available = load_markers(all_epochs=True)
        if not available:
            return None, {
                "code": REFUSE_BRACKET_NOT_FOUND,
                "detail": (f"读不到实验记录(没有活动实验 / 存储不可用),"
                           f"重建不了 bracket {bid!r} —— 而二分的状态真源就是记录。"
                           f"「读不到」不是「没有」。"),
                "alternatives": ["确认有活动实验之后重来",
                                 "或者直接给四个端点坐标,不走重建"],
            }
        built = rebuild_brackets(markers, current_epoch=out.get("coord_epoch"))
        out["known_bracket_ids"] = sorted(built)
        bk = built.get(bid)
        if bk is None:
            return None, {
                "code": REFUSE_BRACKET_NOT_FOUND,
                "detail": (f"记录里没有 bracket_id={bid!r} 的采样点。"
                           f"已知的 bracket:{sorted(built) or '一个都没有'}。"),
                "alternatives": [f"用上面列出的某个 id",
                                 f"先跑 mode='{MODE_SURVEY}' 找一对点"],
            }
        if bk.get("epoch_problem"):
            # D9b 第 1 条。**不夹紧、不换算** —— 换算出来的坐标看上去和真坐标一样。
            return None, {
                "code": REFUSE_BRACKET_EPOCH_SPLIT,
                "detail": bk["epoch_problem"],
                "alternatives": [f"按当前代次重跑 mode='{MODE_SURVEY}' 找一对新端点"],
            }
        if bk.get("problems"):
            return None, {
                "code": REFUSE_BRACKET_NOT_A_PAIR,
                "detail": "；".join(bk["problems"]),
                "alternatives": [f"先跑 mode='{MODE_SURVEY}' 找一对指纹不同的点"],
            }
        return {
            "bracket_id": bid,
            "source": "markers",
            "state": bk.get("state") or STATE_BRACKETED,
            "lo": tuple(bk["lo_m"]), "hi": tuple(bk["hi_m"]),
            "seed_lo": tuple(bk["seed_lo_m"]), "seed_hi": tuple(bk["seed_hi_m"]),
            "lo_label": bk["lo_label"], "hi_label": bk["hi_label"],
            "iterations": int(bk.get("iterations") or 0),
            "offsets_used": int(bk.get("offsets_used") or 0),
            "gap_m": float(bk.get("gap_m") or 0.0),
            "rebuilt": bk,
            "note": (f"状态从 {bk['n_markers']} 条 marker 重建("
                     f"{bk['n_probes']} 个已采的中点/偏移点)—— 不存游标:"
                     f"composite 的 sidecar 按 (name, run_id) 分键,重启就是新 run_id,"
                     f"靠它续跑等于没有续跑。"),
        }, None

    @staticmethod
    def _cross_site_refusal(mode: str) -> dict[str, Any]:
        """``allow_coarse_move=True`` 的拒绝。两个模式共用,措辞按模式分。

        跨站点畴界**不做二分**(设计 D9b 第 3 条):粗动一次,两端的米坐标就指向两片
        不同的表面,「取中点」没有定义。诚实的降级形态由 :func:`cross_site_downgrade`
        给出 —— **步 + 不确定度,一个米制字段都没有**。
        """
        form = cross_site_downgrade(0, 1)
        if mode == MODE_BISECT:
            return {
                "code": REFUSE_CROSS_SITE_BISECT,
                "detail": (
                    "跨站点的畴界**不能二分**。一次横向粗动之后,bracket 两端的米坐标"
                    "指向的是两片不同的表面,「取中点」这个动作没有定义 —— 算出来的"
                    "中点会是一个看上去完全正常、实际上凭空造出来的坐标。跨站点只能"
                    "报成「站点 k 与 k+1 之间 ±N 步」,那是一套独立的输出形态,"
                    "不是这个模式的产物。"),
                "alternatives": [
                    "allow_coarse_move=False:在**单个站点内**二分(压电范围内)",
                    form["next_step"],
                ],
                "cross_site_form": form,
            }
        return {
            "code": REFUSE_COARSE_NOT_YET,
            "detail": ("跨站点普查(allow_coarse_move=True)**这一版还没有**,排在下一轮。"
                       "一次横向粗动之后,已经采到的每一个米坐标都指向另一片表面 —— "
                       "跨站点的结果只能报成「站点 k 与 k+1 之间 ±N 步」,"
                       "不能伪装成米坐标,那是一套独立的输出形态。"),
            "alternatives": ["allow_coarse_move=False:先把当前站点的压电范围铺完",
                             form["next_step"]],
            "cross_site_form": form,
        }

    @staticmethod
    def _default_frame_m() -> tuple[float, str]:
        """默认原子帧 —— 取既有工作流的评估帧,不在这里发明第二个数。"""
        try:
            from mast.core.special_tip_workflow import SpecialTipRecipe

            return float(SpecialTipRecipe().eval_frame_nm) * 1e-9, "workflow_default"
        except Exception as exc:  # noqa: BLE001 — 拿不到就用一个说得出出处的数
            logger.debug("取评估帧默认值失败: %s", exc)
            return 5e-9, "fallback_5nm"

    @staticmethod
    def _resolve_pixels(params: dict, frame_m: float) -> tuple[int, float, str]:
        """这一帧**实际会用**的像素数与估计耗时。

        问的是 ``ScanAt`` 自己那一层(``core.scan_resolver``),不是另算一遍:
        预检必须检**将要下发的那个数**,否则闸门看的是一份和现实无关的参数。
        """
        explicit = params.get("pixels")
        try:
            from mast.core.scan_resolver import PURPOSE_AUTO, ScanIntent, resolve_scan

            intent = ScanIntent(
                center_x_m=0.0, center_y_m=0.0, size_m=float(frame_m),
                purpose=PURPOSE_AUTO,
                explicit={"pixels": int(explicit)} if explicit else {})
            resolved = resolve_scan(intent)
            px = int(resolved.set_scan_buffer["pixels"])
            est = float(getattr(resolved, "estimated_scan_s", 0.0) or 0.0)
            return px, est, ("parameter" if explicit else "policy_table")
        except Exception as exc:  # noqa: BLE001 — 解析不了就用显式值/保守值
            logger.debug("扫描参数解析失败,退到显式值: %s", exc)
            return (int(explicit) if explicit else 512), 0.0, (
                "parameter" if explicit else "fallback_512")

    @staticmethod
    def _load_reference(params: dict) -> dict[str, Any]:
        """参照系(可以没有)。读不动 = 没有,不抛。"""
        version = str(params.get("reference_version") or "").strip()
        sample = str(params.get("sample") or "").strip()
        try:
            from mast.vision.domain_reference import load_reference

            ref = load_reference(version or None, sample=sample or None)
        except Exception as exc:  # noqa: BLE001
            logger.warning("畴参照系加载失败,按「没有参照系」处理: %s", exc)
            ref = None
        out: dict[str, Any] = {
            "reference": ref,
            "reference_requested": version or None,
            "reference_version": getattr(ref, "version", None) if ref else None,
            "reference_calibrated": bool(getattr(ref, "calibrated", False)) if ref
                                    else False,
        }
        sym = params.get("symmetry_deg")
        if sym:
            out["symmetry_deg"] = float(sym)
            out["symmetry_source"] = "parameter"
        elif ref is not None and getattr(ref, "symmetry_deg", 0):
            out["symmetry_deg"] = float(ref.symmetry_deg)
            out["symmetry_source"] = f"reference:{ref.version}"
        else:
            out["symmetry_deg"] = None
            out["symmetry_source"] = "none"
        if ref is not None and getattr(ref, "calibrated", False):
            out["w_angle"] = float(ref.w_angle)
            out["w_period"] = float(ref.w_period)
            out["weights_source"] = f"reference:{ref.version}"
        else:
            out["w_angle"] = 1.0
            out["w_period"] = 1.0
            out["weights_source"] = "uncalibrated_equal_weights"
        return out

    # ── 计划 ──────────────────────────────────────────────────────────────

    def plan_dynamic(self, params: dict,
                     executor: GraphExecutor) -> Iterator[CompositeStep]:
        prep = self._prepare(getattr(self, "_context", None), params)
        self._prep = prep
        self._points: list[dict] = []
        set_partial = executor.set_partial
        for key in ("mode", "frame_size_m", "frame_size_source", "pixels",
                    "pixels_source", "nm_per_px", "scale", "grid_pitch_m",
                    "center_zone_reason", "coord_epoch", "map_history_available",
                    "reference_version", "reference_requested",
                    "reference_calibrated", "symmetry_deg", "symmetry_source",
                    "points_requested", "estimated_scan_s",
                    "locate_tolerance_m", "locate_tolerance_clamped",
                    "locate_tolerance_reason", "offset_budget", "max_iterations",
                    "known_bracket_ids"):
            if key in prep:
                set_partial(key, prep[key])
        set_partial("warnings", list(prep.get("warnings") or ()))

        if prep.get("refusal"):
            self._refusal = dict(prep["refusal"])
            set_partial("refusal", self._refusal)
            return

        self._refusal = None
        if prep["mode"] == MODE_BISECT:
            yield from self._plan_bisect(params, executor, prep)
        else:
            yield from self._plan_survey(params, executor, prep)

    # ── 普查 ──────────────────────────────────────────────────────────────

    def _plan_survey(self, params: dict, executor: GraphExecutor,
                     prep: dict) -> Iterator[CompositeStep]:
        set_partial = executor.set_partial
        positions = list(prep["positions"])
        set_partial("planned_positions",
                    [{"index": i + 1, "x_m": x, "y_m": y}
                     for i, (x, y) in enumerate(positions)])
        set_partial("estimated_total_s",
                    float(prep.get("estimated_scan_s") or 0.0) * len(positions))

        search_id = self._search_id(executor)
        set_partial("domain_search_id", search_id)

        if bool(params.get("dry_run", False)):
            # 完整状态机跑完,一条扫描都不发。
            for i, (x, y) in enumerate(positions):
                self._points.append(self._point_record(
                    i + 1, x, y, prep, search_id, status=POINT_PLANNED,
                    note="dry_run:没有发出扫描"))
            set_partial("dry_run", True)
            return

        seen_frames: set[str] = set()
        consecutive_failures = 0
        stopped = ""
        for i, (x, y) in enumerate(positions):
            idx = i + 1
            if stopped:
                self._points.append(self._point_record(
                    idx, x, y, prep, search_id, status=POINT_SKIPPED, note=stopped))
                continue

            verdict = self._epoch_ok(prep)
            if verdict is not None:
                stopped = verdict
                self._points.append(self._point_record(
                    idx, x, y, prep, search_id, status=POINT_EPOCH_STALE,
                    note=verdict))
                continue

            status, path, assessed, note = yield from self._sample_frame(
                executor, str(idx), x, y, prep, params, seen_frames,
                tags=("scan", "grid"))
            if status == POINT_SCAN_FAILED:
                consecutive_failures += 1
                if consecutive_failures >= MAX_CONSECUTIVE_SCAN_FAILURES:
                    stopped = (f"连续 {consecutive_failures} 帧没扫成,提前收尾 —— "
                               f"后面的点多半也扫不成,而每一帧都在花机时。"
                               f"先查针尖/反馈,再重跑。")
            else:
                consecutive_failures = 0
            self._points.append(self._point_record(
                idx, x, y, prep, search_id, status=status, frame=path,
                note=note, assessed=assessed))

        if stopped:
            set_partial("stopped_early", stopped)

    # ── 采一个点:扫一帧 → 找落盘文件 → 跑判据 ──────────────────────────────

    def _sample_frame(self, executor: GraphExecutor, tag: str, x: float, y: float,
                      prep: dict, params: dict, seen_frames: "set[str]", *,
                      tags: tuple = ("scan",)
                      ) -> "Iterator[CompositeStep]":
        """在 ``(x, y)`` 采一个点。``yield from`` 的返回值是
        ``(status, frame_path, assessed, note)``。

        两个模式共用**同一份**实现。这三步(扫 → 找文件 → 判)每一步的失败都有自己的
        含义 —— 「没扫成」「扫了但没落盘」「落的是上一帧」「判据壳没跑起来」的下一步
        各不相同,合成一个 ``success`` 会把它们抹平。抄第二份的话,迟早只有一份是对的
        (本仓记过:同一个动作 N 份实现,往往只有带事故注释的那份对)。

        ⚠️ 走 ``ScanAt`` 逐帧而不是 ``ExecuteScanPlan``:后者的 frame schema 不透传
        ``angle_deg``,而帧角是样品系归一的必要输入;二分更是**下一个位置取决于上一帧
        的判定**,本来就排不成一个预先的 plan(设计 D12b)。
        """
        yield CompositeStep(
            step_id=f"scan_{tag}", skill_name="ScanAt",
            params=self._scan_params(x, y, prep),
            optional=True, checkpoint_after=True, tags=tuple(tags))
        scan_res = executor.sub_results.get(f"scan_{tag}")
        if scan_res is None or not getattr(scan_res, "success", False):
            err = getattr(scan_res, "error", None) or "ScanAt 没有返回结果"
            return POINT_SCAN_FAILED, "", None, str(err)[:300]

        yield CompositeStep(
            step_id=f"frame_{tag}", skill_name="GetLatestScanFile",
            params={"max_age_s": int(
                float(prep.get("estimated_scan_s") or 0.0)
                + FRAME_LOOKUP_HEADROOM_S)},
            optional=True, checkpoint_after=False, tags=("read",))
        frame_res = executor.sub_results.get(f"frame_{tag}")
        path = str((getattr(frame_res, "data", None) or {}).get("path") or "")
        if not path:
            return POINT_NO_FRAME, "", None, (
                "扫描报成功,但找不到刚落盘的 .sxm —— 判据只吃已保存的帧,"
                "这个点没有证据。")
        if self._frame_is_reused(path, seen_frames):
            # 「借最新文件伪造历史」:上一个点的文件被当成这一个点的证据,
            # 指纹会一模一样,而那个「一样」是假的。
            return POINT_STALE_FRAME, path, None, (
                "找到的 .sxm 是前一个采样点已经用过的那一份 —— 这一帧没有落盘。"
                "拿它当本点的证据会造出一个假的「指纹相同」。")
        seen_frames.add(path)

        yield CompositeStep(
            step_id=f"assess_{tag}", skill_name="AssessDomainPhase",
            params=self._assess_params(path, params),
            optional=True, checkpoint_after=False, tags=("verdict", "read"))
        assess_res = executor.sub_results.get(f"assess_{tag}")
        if assess_res is None or not getattr(assess_res, "success", False):
            err = getattr(assess_res, "error", None) or "判据没有返回结果"
            return POINT_ASSESS_FAILED, path, None, str(err)[:300]
        return (POINT_ASSESSED, path,
                dict(getattr(assess_res, "data", None) or {}), "")

    # ── 二分 ──────────────────────────────────────────────────────────────

    def _plan_bisect(self, params: dict, executor: GraphExecutor,
                     prep: dict) -> Iterator[CompositeStep]:
        """二分状态机的驱动(设计 D9)。**一次都不调 ``pick_next_position*``。**

        每个落点由 :class:`BisectMachine` 算出来,准入只查压电范围与避让圈
        (:func:`probe_position_problem`)——中点与两个端点帧的重叠几乎必然 ≥30%,
        走 ``pick_next_position`` 会被 ``reuse_overlap_frac`` 静默丢掉(设计陷阱 4)。

        **代次在每一个落点之前复核一次。** 陈旧 ⇒ 整个 bracket 立刻
        ``INVALIDATED_BY_EPOCH`` 并停 —— 不是暂停(D9b 第 2 条)。
        """
        set_partial = executor.set_partial
        bracket = prep["bracket"]
        search_id = self._search_id(executor)
        bracket_id = bracket["bracket_id"] or f"{search_id}-b1"
        set_partial("domain_search_id", search_id)
        set_partial("bracket_id", bracket_id)
        set_partial("bracket_source", bracket["source"])
        set_partial("bracket_note", bracket.get("note") or "")
        if bracket.get("rebuilt"):
            set_partial("rebuilt_from_markers", dict(bracket["rebuilt"]))

        machine = BisectMachine(
            bracket_id=bracket_id,
            lo=bracket["lo"], hi=bracket["hi"],
            lo_label=bracket["lo_label"], hi_label=bracket["hi_label"],
            tolerance_m=float(prep["locate_tolerance_m"]),
            frame_size_m=float(prep["frame_size_m"]),
            max_iterations=int(prep["max_iterations"]),
            offset_budget=int(prep["offset_budget"]),
            state=(STATE_SEEDING if bracket["state"] == STATE_SEEDING
                   else STATE_BRACKETED),
            iterations=int(bracket.get("iterations") or 0),
            offsets_used=int(bracket.get("offsets_used") or 0),
            seed_lo=bracket["seed_lo"], seed_hi=bracket["seed_hi"],
        )
        self._machine = machine
        seen_frames: set[str] = set()
        self._point_no = 0

        if bool(params.get("dry_run", False)):
            self._bisect_dry_run(machine, prep, search_id, set_partial)
            return

        # ── SEEDING:显式坐标的两端还没判过,先各扫一帧 ──
        if machine.state == STATE_SEEDING:
            labels: dict[str, str] = {}
            for role, (px, py) in ((ROLE_LO, machine.lo), (ROLE_HI, machine.hi)):
                stale = self._epoch_ok(prep)
                if stale is not None:
                    machine.invalidate(stale)
                    self._record_probe(prep, search_id, bracket_id, role, px, py,
                                       status=POINT_EPOCH_STALE, note=stale)
                    break
                status, path, assessed, note = yield from self._sample_frame(
                    executor, f"{role}", px, py, prep, params, seen_frames,
                    tags=("scan", "bisect"))
                self._record_probe(prep, search_id, bracket_id, role, px, py,
                                   status=status, frame=path, note=note,
                                   assessed=assessed)
                if status == POINT_ASSESSED:
                    labels[role] = str((assessed or {}).get("label") or "")
            if machine.state != STATE_INVALIDATED:
                lo_lab, hi_lab = labels.get(ROLE_LO, ""), labels.get(ROLE_HI, "")
                if not lo_lab or not hi_lab or lo_lab == hi_lab:
                    machine.abandon(END_NOT_A_PAIR, message=(
                        f"两端判出来的是 lo={lo_lab or '判不了'}、"
                        f"hi={hi_lab or '判不了'} —— 二分要求两端各判出一个畴名,"
                        f"而且两个名字不一样。同一个相之间没有畴界可找;判不了的话"
                        f"先看那一帧的 verdict_reason(尺度门要缩视野/加像素,"
                        f"帧角读不到要补角度来源)。"))
                else:
                    machine.lo_label, machine.hi_label = lo_lab, hi_lab
                    machine.state = STATE_BRACKETED

        # ── BISECTING:中点 → 判定 → 收窄 ──
        consecutive_failures = 0
        while True:
            probe = machine.plan_probe()
            if probe is None:
                break
            stale = self._epoch_ok(prep)
            if stale is not None:
                # D9b 第 2 条:**立刻作废,不是暂停**。p_lo / p_hi 的米坐标已经指向
                # 另一片表面,它们之间已经不存在那条畴界。
                machine.invalidate(stale)
                self._record_probe(prep, search_id, bracket_id, probe.role,
                                   probe.x_m, probe.y_m,
                                   status=POINT_EPOCH_STALE, note=stale)
                break
            problem = probe_position_problem(
                probe.x_m, probe.y_m, cfg=prep["cfg"],
                circles=prep.get("avoid_circles") or ())
            if problem:
                machine.skip(probe, reason=problem)
                self._record_probe(prep, search_id, bracket_id, probe.role,
                                   probe.x_m, probe.y_m,
                                   status=POINT_UNREACHABLE, note=problem)
                continue
            tag = f"{probe.role}{len(machine.history) + 1}"
            status, path, assessed, note = yield from self._sample_frame(
                executor, tag, probe.x_m, probe.y_m, prep, params, seen_frames,
                tags=("scan", "bisect"))
            self._record_probe(prep, search_id, bracket_id, probe.role,
                               probe.x_m, probe.y_m, status=status, frame=path,
                               note=note, assessed=assessed)
            if status != POINT_ASSESSED:
                # 采不到判定:与「判不了」走同一条降级路(换偏移点),但记的状态与终态
                # 理由都不同 —— 「没采成」去查针尖/反馈,「采到了但判不了」去看
                # verdict_reason。合成一句会把人送错方向。
                consecutive_failures += 1
                machine.skip(probe, reason=note or status)
                if consecutive_failures >= MAX_CONSECUTIVE_SCAN_FAILURES:
                    machine.abandon(END_NOT_SAMPLED, message=(
                        f"连续 {consecutive_failures} 个落点没采成 —— 后面多半也采不成,"
                        f"而每一帧都在花机时。先查针尖/反馈,再重跑。"))
                    break
                continue
            consecutive_failures = 0
            machine.record(probe, verdict=str((assessed or {}).get("verdict") or ""),
                           label=(assessed or {}).get("label"))

        set_partial("bisect", machine.as_dict())

    def _bisect_dry_run(self, machine: BisectMachine, prep: dict,
                        search_id: str, set_partial) -> None:
        """排练:整条状态机的准备都算完,**一帧都不发**。

        排练能算出来的只有**第一个**落点和一个迭代次数的上界 —— 后面每一步都取决于
        前一帧的判定,那是硬件才有的信息。报一条假的完整路径出来会让人以为这次排练
        验过了整条路。
        """
        probe = machine.plan_probe()
        planned: list[dict] = []
        if probe is not None:
            problem = probe_position_problem(
                probe.x_m, probe.y_m, cfg=prep["cfg"],
                circles=prep.get("avoid_circles") or ())
            planned.append({"role": probe.role, "x_m": probe.x_m,
                            "y_m": probe.y_m, "why": probe.why,
                            "position_problem": problem or ""})
            self._record_probe(prep, search_id, machine.bracket_id, probe.role,
                               probe.x_m, probe.y_m, status=POINT_PLANNED,
                               note="dry_run:没有发出扫描")
        gap = machine.gap_m
        tol = float(machine.tolerance_m)
        bound = (0 if gap <= tol else
                 int(math.ceil(math.log2(gap / tol))) if gap > 0 else 0)
        set_partial("dry_run", True)
        set_partial("planned_probes", planned)
        set_partial("iterations_upper_bound", min(bound, machine.max_iterations))
        set_partial("bisect", machine.as_dict())

    def _record_probe(self, prep: dict, search_id: str, bracket_id: str,
                      role: str, x: float, y: float, *, status: str,
                      frame: str = "", note: str = "",
                      assessed: "dict | None" = None) -> None:
        """把一个二分采样点记进 ``self._points``(marker 的落库素材)。

        编号从 1 连着数 —— 落库那一层按 ``index`` 分区记账,而 marker 的
        ``meta.role`` / ``meta.bracket_id`` 才是重建状态要读的两个键(设计 §3.4)。
        """
        self._point_no = int(getattr(self, "_point_no", 0)) + 1
        self._points.append(self._point_record(
            self._point_no, x, y, prep, search_id, status=status, frame=frame,
            note=note, assessed=assessed, role=role, bracket_id=bracket_id))

    # ── 每个点的记录 ──────────────────────────────────────────────────────

    def _point_record(self, index: int, x: float, y: float, prep: dict,
                      search_id: str, *, status: str, frame: str = "",
                      note: str = "",
                      assessed: "dict | None" = None,
                      role: str = ROLE_SEED,
                      bracket_id: "str | None" = None) -> dict[str, Any]:
        """一个采样点的完整记录 —— 同时是 marker 的落库素材(设计 §3.4)。

        ``meta`` 里那几个键就是设计里那张表。**``scan_angle_deg`` 为 None 表示
        「读不到」,不是 0**;而落库那一层会把 None 的键整个丢掉,所以另配一个
        ``scan_angle_known`` 布尔:没有它,「读不到角度」与「这条 marker 是旧格式」
        在库里长得一模一样。

        ``role`` 与 ``bracket_id`` 是二分状态**唯一**的落脚点:重启之后
        :func:`rebuild_brackets` 只靠这两个键 + ``verdict`` 就能把区间重放出来
        (设计 D9 末条 —— 不存游标)。普查的点是 ``seed`` / ``bracket_id=None``。
        """
        a = assessed or {}
        angle = a.get("scan_angle_deg")
        fingerprint = [list(t) for t in (a.get("fingerprint") or ())]
        verdict = str(a.get("verdict") or "") or None
        reason = str(a.get("verdict_reason") or "") or None
        if status != POINT_ASSESSED:
            # 没有判据结果的点不许携带一个像模像样的 verdict —— 「没测」和
            # 「测出来判不了」必须是两句话。
            verdict, reason, fingerprint, angle = None, None, [], None

        meta: dict[str, Any] = {
            "domain_search_id": search_id,
            "bracket_id": bracket_id or None,   # 普查的点不属于任何 bracket
            "role": role,
            "point_index": index,
            "point_status": status,
            "fingerprint": fingerprint,
            "scan_angle_deg": angle,
            "scan_angle_known": angle is not None,
            "verdict": verdict,
            "verdict_reason": reason,
            "reference_version": prep.get("reference_version"),
            "coord_epoch": prep.get("coord_epoch"),
        }
        rec: dict[str, Any] = {
            "index": index,
            # 显式声明 marker kind。**不新增 kind** —— 采样帧本来就是扫描图,
            # 而新 kind 会撞上双端 KIND 镜像那个结构性缺口(后端 KIND_STYLE ↔
            # 前端配色/图例),失败模式是地图上静默变灰。不声明的话 recorder 会
            # 按技能名去猜,而名字规则认不出这个 composite。
            "kind": "scan",
            "center_x_m": float(x),
            "center_y_m": float(y),
            "width_m": float(prep["frame_size_m"]),
            "height_m": float(prep["frame_size_m"]),
            "success": status in (POINT_ASSESSED, POINT_PLANNED),
            "status": status,
            "error": note or None,
            "sxm_path": frame or None,
            "label": (f"畴普查 #{index}" if role == ROLE_SEED
                      else f"畴二分 #{index} ({role})"),
            "meta": meta,
            # ↓ 报告用的展开(marker 只吃 meta;这几个键让结果本身读得懂)
            "verdict": verdict,
            "verdict_reason": reason,
            "next_step": a.get("next_step") or "",
            "fingerprint": fingerprint,
            "scan_angle_deg": angle,
            "n_peaks": a.get("n_peaks"),
            "nm_per_px": a.get("nm_per_px"),
            "scale": a.get("scale"),
            "label_out": a.get("label"),
        }
        return rec

    @staticmethod
    def _scan_params(x: float, y: float, prep: dict) -> dict[str, Any]:
        """交给 ``ScanAt`` 的参数。

        走 ``ScanAt`` 而不是 ``ExecuteScanPlan``:后者的 frame schema 不透传
        ``angle_deg``,而帧角是样品系归一的必要输入(设计 D12b)。

        ``pixels`` 只在**调用方显式给过**时传下去;不传就是走策略层那条默认路 ——
        这里绝不把自己刚从策略层查出来的那个数再原样传回去,那会让它在
        trace 里被标成「用户显式」,而没有人说过它。
        """
        out: dict[str, Any] = {
            "center_x_m": float(x),
            "center_y_m": float(y),
            "size_m": float(prep["frame_size_m"]),
        }
        if prep.get("pixels_source") == "parameter":
            out["pixels"] = int(prep["pixels"])
        return out

    @staticmethod
    def _assess_params(path: str, params: dict) -> dict[str, Any]:
        return {
            "scan_path": path,
            "channel": "Z",
            "reference_version": str(params.get("reference_version") or ""),
            "sample": str(params.get("sample") or ""),
        }

    @staticmethod
    def _frame_is_reused(path: str, seen: "set[str]") -> bool:
        """这个 ``.sxm`` 是不是前面某个采样点已经用过的那一份。

        单独一个方法而不是一行 ``in``:这是一道**判据闸**,要能被单独变异掉验红。
        它防的是本仓记过的「借最新文件伪造历史」—— 一帧没落盘时
        ``GetLatestScanFile`` 会诚实地把上一帧交出来,而两个点拿到同一张图,
        指纹当然一模一样,普查于是得出「这一片全是同一个畴」。
        """
        return str(path) in seen

    @staticmethod
    def _epoch_ok(prep: dict) -> "str | None":
        """粗动了没有。陈旧 ⇒ 返回一句话(停);其余 ⇒ None(继续)。

        ``UNVERIFIABLE`` / ``UNSTAMPED`` **不当陈旧**:那会让「读不到记录」变成
        一道解不开的闸。它们已经在 ``warnings`` 里说过「这一轮没有代次保护」。
        """
        stamped = prep.get("coord_epoch")
        if stamped is None:
            return None
        from mast.core.coord_epoch import verify

        what = ("这个 bracket 的两个端点坐标"
                if prep.get("mode") == MODE_BISECT else "这一轮普查的网格坐标")
        v = verify(stamped, what=what)
        return v.message if v.stale else None

    def _search_id(self, executor: GraphExecutor) -> str:
        """一次搜索的 id。落进每条 marker 的 meta,事后靠它把这一轮的点收拢起来。

        前缀分模式:一次普查的点和一次二分的点在库里是两批不同用途的记录,前缀让人
        (和事后的查询)一眼分得开。``bracket_id`` 才是二分状态的键 —— 这个 id 只管
        「这些点是同一次运行采的」。
        """
        ctx = getattr(self, "_context", None)
        run_id = str(getattr(ctx, "run_id", "") or "")
        if not run_id:
            run_id = str(getattr(executor.progress, "started_at", "") or "local")
        mode = (getattr(self, "_prep", None) or {}).get("mode") or MODE_SURVEY
        return f"domain-{mode}-{run_id}"

    # ── 汇总 ──────────────────────────────────────────────────────────────

    def aggregate(self, sub_results: dict, progress: CompositeProgress) -> dict:
        data = dict(progress.partial_data)
        prep = getattr(self, "_prep", None) or {}
        log = list(getattr(self, "_points", None) or ())
        refusal = getattr(self, "_refusal", None)
        data["refusal"] = refusal
        # 完整的账在 point_log 里(一条不少);地图只吃针尖真的去过的那些。
        data["point_log"] = log
        data["points"] = [p for p in log if p["status"] in VISITED_STATUS]

        assessed = [p for p in log if p["status"] == POINT_ASSESSED]
        comparable = [
            {"index": p["index"], "fingerprint": p["fingerprint"],
             "frame": p.get("sxm_path")}
            for p in assessed if p.get("fingerprint")
        ]
        if refusal or data.get("mode") == MODE_BISECT:
            # 拒绝了就别再报一个第二来源的「聚类没做」理由 —— 那会把人送去查
            # symmetry_deg,而真正发生的事是这一轮压根没开始。
            # 二分同理:它的产物是一个区间,不是一次聚类;报一个「聚类没做」出来
            # 只会让人去找一个这个模式根本不需要的东西。
            data["clustering"] = {
                "available": False, "reason": "not_run",
                "next_step": ("这一轮被拒绝了,先看 refusal。" if refusal else
                              "二分模式不做聚类 —— 它的产物是 bisect 里那个区间。"),
                "clusters": [], "n_comparable": 0}
        else:
            data["clustering"] = cluster_two(
                comparable,
                symmetry_deg=prep.get("symmetry_deg"),
                w_angle=float(prep.get("w_angle", 1.0)),
                w_period=float(prep.get("w_period", 1.0)),
                symmetry_source=str(prep.get("symmetry_source") or ""),
                weights_source=str(prep.get("weights_source") or ""),
            ).as_dict()

        verdicts: dict[str, int] = {}
        reasons: dict[str, int] = {}
        for p in assessed:
            key = str(p.get("verdict") or "?")
            verdicts[key] = verdicts.get(key, 0) + 1
            if p.get("verdict_reason"):
                reasons[p["verdict_reason"]] = reasons.get(p["verdict_reason"], 0) + 1
        data["verdict_counts"] = verdicts
        data["undetermined_reasons"] = reasons
        data["points_assessed"] = len(assessed)
        data["points_visited"] = len(data["points"])
        data["points_planned"] = len(log)
        data["labels_emitted"] = sorted(
            {str(p["label_out"]) for p in assessed if p.get("label_out")})

        angles = sorted({round(float(p["scan_angle_deg"]), 3)
                         for p in assessed if p.get("scan_angle_deg") is not None})
        data["scan_angles_deg"] = angles
        warns = list(data.get("warnings") or ())
        if len(angles) > 1:
            warns.append(
                f"这一轮的帧角不止一个({angles})—— 样品系归一会把它们抹平,"
                f"但如果 xy 压电是各向异性的,同一个畴在不同帧角下会报成两个畴。"
                f"先在**同一帧角**下比对。")
        n_unknown_angle = sum(1 for p in assessed
                              if p.get("verdict_reason") == "unknown_frame_angle")
        if n_unknown_angle:
            warns.append(
                f"{n_unknown_angle} 个点的帧角读不到 —— 这些点判不了,"
                f"**没有按 0° 处理**(那会让同一个畴在两种帧角下报成两个畴)。")
        data["warnings"] = warns
        return data

    def _validate_products(self, data: dict) -> tuple[bool, str]:
        """本技能的产物是一批**读数**,不是一张图 —— 跳过通用产物闸。

        通用闸会挑 ``scan_path`` 一族的键去把那一帧再判一次(死平/全 NaN ⇒ degraded)。
        这里每个点的帧已经各自被判过,而且「这一帧死平」正是 ``no_atomic_phase``
        已经说出来的话:让通用闸再判一遍,等于对同一批帧给出第二个来源不同的结论,
        然后用它盖掉第一个。
        """
        return True, ""

    def _decide_outcome(self, all_good: bool, progress: CompositeProgress,
                        data: dict) -> tuple[bool, str]:
        """拒绝 ⇒ 失败;「判不了」⇒ 成功。

        两者混用是这条流程里最容易犯的错:全部 ``undetermined(no_reference)``
        是**正确的第一轮产物**(设计 D3/R6),把它记成失败会让上层去修一个
        根本没坏的东西;而 mode/尺度/跨站点那几条是「这件事没做成」,
        必须以失败的面目出现。

        二分这一边多一条**不对称**,值得写清楚:

        * ``ABANDONED``(判不了用光预算 / 端点不成一对 / 出现第三个相)⇒ **成功**。
          验收 R8 的通过判据逐字就是「收敛到 locate_tolerance_m,或诚实
          ``ABANDONED``」—— 它是一个信息量足够的答案,上层照着它换一对点就行。
        * ``INVALIDATED_BY_EPOCH`` ⇒ **失败**。这一条不是「答案是没找到」,而是
          **这一轮的坐标全部作废**:粗动之后 p_lo/p_hi 指向另一片表面,已采的
          那几帧属于哪一片都说不清。报成功会让上层拿着一个空产物往下走 ——
          本仓记过的「假成功」正是这个形状。
        """
        refusal = getattr(self, "_refusal", None)
        if refusal:
            return False, f"[{refusal['code']}] {refusal['detail']}"
        bisect = data.get("bisect") or {}
        if bisect.get("state") == STATE_INVALIDATED:
            return False, (
                f"[{REFUSE_EPOCH_STALE}] 二分中途发生了粗动,bracket "
                f"{bisect.get('bracket_id')!r} 立刻作废({STATE_INVALIDATED})—— "
                f"p_lo / p_hi 的米坐标现在指向另一片表面,它们之间已经不存在那条"
                f"畴界。**不是暂停**:这两个坐标回头也不能再用。"
                f"{bisect.get('note') or ''} 请按当前代次重新找一对端点。"
                f"**不做跨代次换算**:粗动步进是开环的,换算出来的坐标看上去和真"
                f"坐标一样,而它是编的。")
        return super()._decide_outcome(all_good, progress, data)

    # ── 执行 ──────────────────────────────────────────────────────────────

    def run_composite(self, context, params: dict) -> SkillResult:
        self._context = context
        self._prep = {}
        self._points = []
        self._refusal = None
        self._machine = None
        self._point_no = 0
        result = self._graph_execute(context, params)
        data = dict(result.data or {})
        if not result.success:
            return SkillResult(
                skill_name=self._skill_name(), success=False,
                error=result.error, data=data,
                nanonis_calls=list(result.nanonis_calls))
        return SkillResult(
            skill_name=self._skill_name(), success=True, data=data,
            summary=self._summary(data),
            nanonis_calls=list(result.nanonis_calls))

    @staticmethod
    def _summary(data: dict) -> str:
        if data.get("mode") == MODE_BISECT:
            return SearchDomainBoundary._bisect_summary(data)
        lines: list[str] = []
        n = data.get("points_planned") or 0
        ok = data.get("points_assessed") or 0
        if data.get("dry_run"):
            lines.append(f"排练(dry_run):排了 {n} 个网格点,**一帧都没有发出去**,"
                         f"地图上也不留任何足迹。")
        else:
            lines.append(f"畴普查:{n} 个网格点,{ok} 个拿到了判据结果"
                         f"(针尖去过 {data.get('points_visited') or 0} 个)。")
        pitch = data.get("grid_pitch_m")
        frame = data.get("frame_size_m")
        if pitch and frame:
            lines.append(
                f"网格间距 {float(pitch) * 1e9:.0f} nm,帧 {float(frame) * 1e9:.1f} nm / "
                f"{data.get('pixels')} px = {float(data.get('nm_per_px') or 0):.4f} nm/px"
                f"（{data.get('scale')} 档）")
        if data.get("center_zone_reason"):
            lines.append(str(data["center_zone_reason"]))

        counts = data.get("verdict_counts") or {}
        if counts:
            lines.append("判定：" + "、".join(f"{k} ×{v}" for k, v in sorted(counts.items())))
        reasons = data.get("undetermined_reasons") or {}
        if reasons:
            lines.append("判不了的细分：" + "、".join(
                f"{k} ×{v}" for k, v in sorted(reasons.items())))
        if not data.get("reference_calibrated"):
            lines.append(
                "没有标定过的畴参照系 —— 每个点都是 undetermined(no_reference)。"
                "**这是正确的第一轮产物,不是失败**:指纹已经拿到了,"
                "下一步是人看两簇的代表帧、确认它们是不是两个畴,再固化成参照系。")

        cl = data.get("clustering") or {}
        if cl.get("available"):
            bits = []
            for c in cl.get("clusters") or ():
                intra = c.get("intra_max")
                bits.append(f"{c['name']}: {c['size']} 帧"
                            + (f",簇内最大距离 {intra:.4f}" if intra is not None
                               else ",簇内距离无样本"))
            lines.append("聚类(k=2,**簇不是畴的名字**)：" + "；".join(bits))
            if cl.get("inter_min") is not None:
                lines.append(f"簇间最小距离 {cl['inter_min']:.4f}")
            if cl.get("separation_ratio") is not None:
                lines.append(f"分离度 = 簇间/簇内 = {cl['separation_ratio']:.2f}")
            elif cl.get("ratio_reason"):
                lines.append(f"分离度算不出来：{cl['ratio_reason']}")
            if cl.get("next_step"):
                lines.append(f"下一步：{cl['next_step']}")
        elif cl.get("reason"):
            lines.append(f"聚类没做（{cl['reason']}）：{cl.get('next_step', '')}")

        if data.get("stopped_early"):
            lines.append(f"提前收尾：{data['stopped_early']}")
        for w in (data.get("warnings") or ()):
            lines.append(f"⚠️ {w}")
        return "\n".join(lines)

    @staticmethod
    def _bisect_summary(data: dict) -> str:
        """二分的人话。**三个终态各说各的下一步** —— 合成一句「没找到」等于没说。"""
        b = data.get("bisect") or {}
        lines: list[str] = []
        frame = float(data.get("frame_size_m") or 0.0)
        tol = float(b.get("tolerance_m") or data.get("locate_tolerance_m") or 0.0)
        if data.get("dry_run"):
            bound = data.get("iterations_upper_bound")
            lines.append(
                f"排练(dry_run):bracket {b.get('bracket_id')} 的状态机跑完了,"
                f"**一帧都没有发出去**,地图上也不留任何足迹。"
                + (f" 按当前区间 {float(b.get('gap_m') or 0) * 1e9:.0f} nm 与容差 "
                   f"{tol * 1e9:.1f} nm,最多再要 {bound} 次二分。"
                   if bound is not None else ""))
            for p in (data.get("planned_probes") or ()):
                lines.append(
                    f"第一个落点({p['role']}):({p['x_m'] * 1e9:.1f}, "
                    f"{p['y_m'] * 1e9:.1f}) nm —— {p['why']}"
                    + (f" ⚠️ 但它{p['position_problem']}"
                       if p.get("position_problem") else ""))
            lines.append(
                "排练只算得出**第一个**落点:后面每一步都取决于前一帧的判定,"
                "那是硬件才有的信息。报一条完整路径出来会让人以为整条路已经验过了。")
        else:
            state = b.get("state")
            lines.append(
                f"二分 bracket {b.get('bracket_id')}（{b.get('lo_label')} ↔ "
                f"{b.get('hi_label')}）：{state}／{b.get('end_reason') or '—'}，"
                f"{b.get('iterations')} 次收窄、{b.get('offsets_used')} 个偏移点，"
                f"针尖去过 {data.get('points_visited') or 0} 个位置。")
            if state == STATE_CONVERGED and b.get("boundary_m"):
                bx, by = b["boundary_m"]
                unc = float(b.get("boundary_uncertainty_m") or 0.0)
                lines.append(
                    f"畴界在 ({bx * 1e9:.1f}, {by * 1e9:.1f}) nm，±{unc * 1e9:.1f} nm"
                    + ("（这一帧里两个畴的峰**都在** —— 畴界就落在这一帧内，"
                       "比继续二分更直接）" if b.get("end_reason") == END_MIXED
                       else f"（区间已收到 {float(b.get('gap_m') or 0) * 1e9:.1f} nm"
                            f"，容差 {tol * 1e9:.1f} nm）"))
            elif state == STATE_ABANDONED:
                lines.append(
                    f"**没有定位到畴界** —— 诚实地停在 ABANDONED，"
                    f"区间还剩 {float(b.get('gap_m') or 0) * 1e9:.0f} nm。"
                    f"不给一个「大概在这儿」的坐标：那会是编的。")
            elif state == STATE_INVALIDATED:
                lines.append(
                    "**这个 bracket 已作废**：二分中途发生了粗动，两端的米坐标现在"
                    "指向另一片表面。不是暂停 —— 这两个坐标回头也不能再用。")
            if b.get("note"):
                lines.append(str(b["note"]))
        if frame and tol:
            lines.append(
                f"终止容差 {tol * 1e9:.1f} nm，帧宽 {frame * 1e9:.1f} nm。"
                + ("**容差被帧宽顶上去了** —— " if data.get("locate_tolerance_clamped")
                   else "")
                + "下限是帧宽不是压电分辨率：两个中心相距不到一个帧宽的帧看的是同一片"
                  "表面，指纹必然相同。要更细只能换更小的帧，而那是换一套判据。")
        if data.get("bracket_note"):
            lines.append(str(data["bracket_note"]))
        for w in (data.get("warnings") or ()):
            lines.append(f"⚠️ {w}")
        return "\n".join(lines)


def make_tool(context_provider):
    return wrap_skill(SearchDomainBoundary, context_provider)


__all__ = [
    "ALL_MODES",
    "ALL_POINT_STATUS",
    "ALL_REFUSALS",
    "ALL_ROLES",
    "ALL_STATES",
    "CLUSTER_NO_SYMMETRY",
    "CLUSTER_TOO_FEW",
    "DEFAULT_MAX_ITERATIONS",
    "DEFAULT_OFFSET_BUDGET",
    "END_EPOCH",
    "END_MAX_ITERATIONS",
    "END_MIXED",
    "END_NOT_A_PAIR",
    "END_NOT_SAMPLED",
    "END_THIRD_LABEL",
    "END_TOLERANCE",
    "END_UNDETERMINED",
    "MAX_CONSECUTIVE_SCAN_FAILURES",
    "MODE_BISECT",
    "MODE_SURVEY",
    "POINT_ASSESSED",
    "POINT_ASSESS_FAILED",
    "POINT_EPOCH_STALE",
    "POINT_NO_FRAME",
    "POINT_PLANNED",
    "POINT_SCAN_FAILED",
    "POINT_SKIPPED",
    "POINT_STALE_FRAME",
    "POINT_UNREACHABLE",
    "REFUSE_BRACKET_EPOCH_SPLIT",
    "REFUSE_BRACKET_NOT_A_PAIR",
    "REFUSE_BRACKET_NOT_FOUND",
    "REFUSE_COARSE_NOT_YET",
    "REFUSE_CROSS_SITE_BISECT",
    "REFUSE_EPOCH_STALE",
    "REFUSE_MODE_UNKNOWN",
    "REFUSE_NO_BRACKET",
    "REFUSE_NO_POSITIONS",
    "REFUSE_NO_REFERENCE",
    "REFUSE_SCALE",
    "ROLE_HI",
    "ROLE_LO",
    "ROLE_MID",
    "ROLE_OFFSET",
    "ROLE_SEED",
    "STATE_ABANDONED",
    "STATE_BISECTING",
    "STATE_BRACKETED",
    "STATE_CONVERGED",
    "STATE_INVALIDATED",
    "STATE_SEEDING",
    "TERMINAL_STATES",
    "VISITED_STATUS",
    "BisectMachine",
    "BisectProbe",
    "ClusterReport",
    "SearchDomainBoundary",
    "bracket_gap_m",
    "check_bracket_epochs",
    "cluster_two",
    "cross_site_downgrade",
    "default_grid_pitch_m",
    "locate_tolerance",
    "midpoint",
    "offset_probe",
    "probe_position_problem",
    "rebuild_brackets",
    "release_center_zone",
    "scale_refusal",
]
