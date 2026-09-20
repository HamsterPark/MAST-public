"""The coarse map: where on the SAMPLE we have been, at stage scale.

There are two maps, and they answer different questions.

``map_analysis`` works at **piezo scale** — a ±1.5 µm square in metres, one
generation at a time — and answers "what happened inside this patch". The
generation boundary is the whole point: after a lateral coarse move those metre
coordinates address a different piece of surface, so the old markers are excluded.

This module works at **stage scale**. It answers directly the question of how to
avoid returning to a spot already worked.

Not as a rule — as a map. Each generation is a **site**: somewhere on the sample
the tip has actually worked. Plot the sites and the answer to "where should we go
next" is geometry, visible to a person and computable by the planner, instead of
a policy sentence that has to be remembered.

WHY STEPS, NOT METRES
=====================
A coarse stepper is open-loop. Step size drifts with drive amplitude, with load,
and above all with temperature — the same 100 steps can travel five times further
at 300 K than at 4 K. ``xy_motor_step_m`` exists as an optional calibration and
the repo already refuses to compute with it ("nothing computes with it, and no
marker is ever re-projected from it"). This module keeps that promise: the
odometer is in **steps**, and metres appear only as an annotation for humans.

What replaces the missing precision is an explicit, GROWING uncertainty: a site is
drawn as a blob, not a point, and the blob gets bigger the further you have
travelled. That is what "有大概标记" means here — the approximation is in the
picture, not hidden behind a number that looks exact.

WHERE THE DATA COMES FROM
=========================
Nowhere new. The odometer is the prefix sum of the ``coarse_move`` rows already in
``map_markers`` — the same event stream ``coord_epoch`` is derived from, read a
second way. No table, no column, no second copy of "where are we" to drift out of
sync with the first. Site *k* is the position after *k* moves; it is also
generation *k*, so a site and an epoch are the same object seen from two scales.

HONEST DEGRADATION
==================
A ``coarse_move`` row may carry no direction or no step count — the manual
backfill path allows both to be blank, because knowing THAT a move happened is
worth recording even when the details are lost. From such a row onward the
absolute position is unknown, and this module says so (``position_known=False``)
rather than carrying on with a number that is now fiction. The planner then
restricts itself to continuing in the current direction, which stays meaningful
under a relative-only odometer, and refuses to plan a return, which does not.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Sequence

__all__ = [
    "CoarseMapConfig",
    "CoarseSite",
    "RelocationPlan",
    "CoarseMap",
    "AXIS_OF",
    "SIGN_OF",
    "derive_sites",
    "plan_relocation",
    "build_coarse_map",
]

#: Lateral direction → (axis, sign). Z is absent on purpose: approaching or
#: retracting does not change WHERE on the surface the tip is.
AXIS_OF: dict[str, str] = {"x+": "x", "x-": "x", "y+": "y", "y-": "y"}
SIGN_OF: dict[str, int] = {"x+": 1, "x-": -1, "y+": 1, "y-": -1}

#: Kinds that mean the surface at that spot was consumed, for the per-site
#: summary. Mirrors ``map_analysis.DAMAGE_KINDS`` but is only ever counted here.
_DAMAGE_KINDS = ("tip_shape", "pulse", "crash", "approach")

#: The row that ENDS a site rather than happening at one. Named once so the
#: per-site tally and the odometer cannot disagree about what a boundary is.
_BOUNDARY_KIND = "coarse_move"


@dataclass(frozen=True)
class CoarseMapConfig:
    """Everything the derivation and the planner are allowed to assume.

    One object, no private module constants — the same rule ``AnalysisConfig``
    follows, so a test can vary a threshold without patching internals."""

    #: Minimum separation between two sites, in steps. Must be large enough that
    #: the new region falls entirely outside the piezo range, or "relocating"
    #: only slides the old region into view.
    site_spacing_steps: int = 200
    #: Per-axis travel budget. The stage has finite travel; past the end it just
    #: slips (harmless) but position knowledge is gone.
    #: **该上限从 5000 调大到了 10000**。这是**行程预算**,不是安全限位:走到头只会空滑
    #: (无害),丢掉的是位置认知。把它调大等于说「这台台子的行程比 5000 步长」。
    axis_step_budget: int = 10_000
    #: Relative scatter of **one individual step** (not of one move command).
    #: Open-loop, so this is the whole error model: after N steps of travel the
    #: blob radius is ``frac * sqrt(N)`` steps. See :meth:`uncertainty_steps`.
    step_uncertainty_frac: float = 0.3
    #: Optional step→metre calibration. ANNOTATION ONLY (see module docstring).
    step_m: float | None = None
    #: Default size of a relocation when the caller does not ask for one.
    #:
    #: ⚠️ 它等于 ``site_spacing_steps``,而按构造**一次刚好等于间距的移动永远
    #: 满足不了自己的要求**(要的是 间距 + 不确定量)。所以规划器每次都会先放大
    #: 一档(``move * 1.5 + 1``,200 → 301),这个 200 实际从来没被真正用过。
    #: 2026-08-18 只是把它写下来,没有改 —— 改它等于改真机上每次换区实走的距离。
    default_move_steps: int = 200
    # 2026-08-18 删掉了 ``max_candidates: int = 4096``。
    #
    # **没有任何代码读过它。** 规划器只考察四个方向(``for direction in
    # ("x+", "x-", "y+", "y-")``),候选数恒等于 4;那个 4096 是从
    # ``map_analysis.AnalysisConfig`` 抄过来的名字,而那边是真的在用。
    #
    # 留着它的代价不是浪费一行:它是一句**关于这个规划器的假话** —— 谁想「让它多
    # 看几个落点」都会先去调这个数,而调了什么都不会发生,还查不出为什么。
    # (本仓 `producer_wired_consumer_absent` 的同一形状。)

    def uncertainty_steps(self, total_steps: float) -> float:
        """走过 ``total_steps`` 步之后的模糊半径,单位是**步**。

        ``frac * sqrt(N)``:单步步长围绕均值散布,N 步的和是随机游走,误差按平方
        根累积。这正是参数名说的那件事 —— ``step_uncertainty_frac`` /
        ``xy_step_uncertainty_frac`` / 设置页的「单步位移不确定度」,三处都写着
        **单步**。

        ## 2026-08-18:原来的式子把误差算大了一个数量级

        原实现是 ``frac * mean_step * sqrt(n_moves)``,即**每一次粗动命令**整体
        误差 30%,命令之间平方根累积。而名字、profile 描述、UI 标签和这段
        docstring 说的都是「单步」。差多少:

        ====================  ==========  ==========  ======
        走法                  原来        现在        倍数
        ====================  ==========  ==========  ======
        1 次 × 200 步          60.0 步     4.2 步     14×
        10 次 × 200 步        189.7 步    13.4 步     14×
        1 次 × 1000 步        300.0 步     9.5 步     32×
        ====================  ==========  ==========  ======

        后果不只是图上画得糊:这个数直接进 :func:`_blocked_by` 的
        ``need = site_spacing_steps + rel_unc``。走过 20 次之后,新落点被要求离
        旧站点 468 步而不是 200 步 —— 在 10000 步的行程预算里,这等于把可用站点
        数砍掉一多半,然后报「没有可去的新站点了」。「粗动大地图现在的默认误差大的离谱。」

        原式还有一个一眼可见的毛病:**同一段路,拆成的命令越多误差越小**。
        200 步一次是 60,拆成 4 次 50 步是 30,拆成 16 次是 15 —— 而走过的路一样长。
        现在这个式子只看总步数,与怎么切命令无关(``test_chunking_a_journey``)。

        ## 这个模型**不**包含什么

        只有**随机散布**。系统性的步长标定误差(温度/驱动幅度/负载 —— 模块开头
        那句「同样 100 步在 300 K 能走出 4 K 的五倍」)**不在此列**,而且不需要在:
        这个数唯一参与的判决是 :func:`_blocked_by` 里的**相对**间距,而系统性误差
        是两个坐标共有的,按比例同时缩放,相对几何不变(那一段的注释已经写了这条
        道理,这里只是说清它对哪一项成立)。

        ⚠️ 反过来的推论要记住:**地图上那个 blob 不是「我在样品上的绝对位置误差」**。
        步长标定偏 20% 的话,走 2000 步之后绝对位置能差 400 步,而 blob 只画 ±13。
        它回答的是「这块地方我去过没有」,不是「我在哪」。
        """
        n = abs(float(total_steps or 0.0))
        if n <= 0.0:
            return 0.0
        return float(self.step_uncertainty_frac) * math.sqrt(n)


@dataclass(frozen=True)
class CoarseSite:
    """One generation, i.e. one patch of sample the tip has worked on."""

    index: int                       # == coord_epoch
    x_steps: int = 0
    y_steps: int = 0
    position_known: bool = True
    uncertainty_steps: float = 0.0
    temperature_k: float | None = None
    first_ts: str = ""
    last_ts: str = ""
    is_current: bool = False
    summary: dict[str, Any] = field(default_factory=dict)

    def distance_steps(self, other: "CoarseSite") -> float:
        return math.hypot(self.x_steps - other.x_steps, self.y_steps - other.y_steps)

    def as_dict(self, step_m: float | None = None) -> dict[str, Any]:
        out: dict[str, Any] = {
            "index": self.index, "x_steps": self.x_steps, "y_steps": self.y_steps,
            "position_known": self.position_known,
            "uncertainty_steps": round(self.uncertainty_steps, 1),
            "is_current": self.is_current, "summary": dict(self.summary),
        }
        if self.temperature_k is not None:
            out["temperature_k"] = self.temperature_k
        if self.first_ts:
            out["first_ts"] = self.first_ts
        if self.last_ts:
            out["last_ts"] = self.last_ts
        if step_m:
            # Annotation only, and labelled as such at every layer that shows it.
            out["approx_x_um"] = round(self.x_steps * step_m * 1e6, 3)
            out["approx_y_um"] = round(self.y_steps * step_m * 1e6, 3)
        return out


@dataclass(frozen=True)
class RelocationPlan:
    """Where to go next, and why — or, when ``None`` is returned, why not."""

    axis: str                        # "x" | "y"
    direction: str                   # "x+" | "x-" | "y+" | "y-"
    steps: int
    lands_at: tuple[int, int]
    clearance_steps: float           # margin to the nearest visited blob
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {"axis": self.axis, "direction": self.direction, "steps": self.steps,
                "lands_at_steps": list(self.lands_at),
                "clearance_steps": round(self.clearance_steps, 1),
                "reason": self.reason}


@dataclass(frozen=True)
class CoarseMap:
    sites: list[CoarseSite]
    current: CoarseSite
    suggestion: RelocationPlan | None
    note: str
    budget_used: dict[str, int]
    config: CoarseMapConfig

    def as_dict(self) -> dict[str, Any]:
        return {
            "sites": [s.as_dict(self.config.step_m) for s in self.sites],
            "current_index": self.current.index,
            "position_known": self.current.position_known,
            "suggestion": self.suggestion.as_dict() if self.suggestion else None,
            "note": self.note,
            "budget_used_steps": dict(self.budget_used),
            "axis_step_budget": self.config.axis_step_budget,
            "site_spacing_steps": self.config.site_spacing_steps,
        }


# ── Derivation ───────────────────────────────────────────────────────────────

def _meta(row: Any) -> dict:
    m = (row or {}).get("meta")
    if isinstance(m, dict):
        return m
    if isinstance(m, str) and m.strip():
        try:
            parsed = json.loads(m)
            return parsed if isinstance(parsed, dict) else {}
        except (ValueError, TypeError):
            return {}
    return {}


def _int_or_none(v: Any) -> int | None:
    if isinstance(v, bool) or v is None:
        return None
    try:
        n = int(v)
    except (TypeError, ValueError):
        return None
    return n


def derive_sites(rows: Sequence[dict], cfg: CoarseMapConfig) -> list[CoarseSite]:
    """Sites from the ``coarse_move`` event stream, oldest first.

    Site 0 always exists — it is where the tip started, before any move. Site *k*
    is the odometer after the *k*-th move, and equals generation *k*.

    Each site's ``summary`` counts what happened while the tip was THERE, taken
    from the rows carrying that ``coord_epoch``. That is what makes the map worth
    looking at: a blob labelled "3 scans, 1 crash" is a place you know something
    about, and one labelled "0 scans" is a place you left without trying.

    THE SUMMARY MUST ADD UP (2026-08-10)
    ====================================
    ``scans`` + ``sts`` + ``damage`` used to be the whole tally, and every row
    that was none of those was **silently dropped**. A real ``ForgeAuTip`` run
    added 26 markers to the map and this summary did not move by one — the site
    still read "30 扫图 / 2 破坏", exactly as before the run. Nothing was wrong
    with the arithmetic; the ledger simply had no column for what those rows were
    (they had arrived as ``kind="manual"``, see the note on ``other`` below), so
    "26 things happened here" and "nothing happened here" rendered identically.

    ``other`` closes that: it counts every positioned row the three named columns
    did not claim, so a reader can always tell "this site is quiet" from "this
    site is busy with things this tally cannot name". A count that silently omits
    what it cannot classify is a count that lies by rounding to zero."""
    rows = list(rows or [])

    # Per-generation activity, keyed by the stamped epoch (NULL ≡ 0, the legacy
    # convention: a database with no coarse_move rows only ever had one frame).
    per_epoch: dict[int, dict[str, Any]] = {}
    seen_moves = 0
    for row in rows:
        stored = _int_or_none((row or {}).get("coord_epoch"))
        epoch = stored if stored is not None else seen_moves
        bucket = per_epoch.setdefault(
            epoch, {"scans": 0, "sts": 0, "damage": 0, "other": 0,
                    "first_ts": "", "last_ts": ""})
        kind = str((row or {}).get("kind") or "")
        if kind == "scan":
            bucket["scans"] += 1
        elif kind == "sts":
            bucket["sts"] += 1
        if kind in _DAMAGE_KINDS:
            bucket["damage"] += 1
        elif kind not in ("scan", "sts", _BOUNDARY_KIND):
            # Everything the three named columns did not claim. ``coarse_move``
            # is excluded because it is the BOUNDARY between sites, not work done
            # at one — counting a departure as activity would make every site
            # look one event busier than it was.
            #
            # In practice this column is mostly ``manual``: MAST's own composite
            # sub-steps do not reach the map recorder (it is wired at the agent-
            # tool boundary only), so a long composite's bias / setpoint / frame /
            # Z-status changes are picked up by the manual-activity watcher and
            # filed as 手动操作. Those rows are REAL events at this site; they are
            # just mis-attributed. Until the recorder is wired for sub-steps they
            # belong here rather than nowhere.
            bucket["other"] += 1
        ts = str((row or {}).get("timestamp") or "")
        if ts:
            if not bucket["first_ts"]:
                bucket["first_ts"] = ts
            bucket["last_ts"] = ts
        if kind == "coarse_move":
            seen_moves += 1

    moves = [r for r in rows if str((r or {}).get("kind") or "") == "coarse_move"]

    sites: list[CoarseSite] = []
    x = y = 0
    known = True
    total_steps = 0.0

    def _make(index: int, temp: float | None) -> CoarseSite:
        info = per_epoch.get(index, {})
        return CoarseSite(
            index=index, x_steps=x, y_steps=y, position_known=known,
            uncertainty_steps=cfg.uncertainty_steps(total_steps),
            temperature_k=temp,
            first_ts=str(info.get("first_ts", "")),
            last_ts=str(info.get("last_ts", "")),
            summary={"scans": int(info.get("scans", 0)),
                     "sts": int(info.get("sts", 0)),
                     "damage": int(info.get("damage", 0)),
                     "other": int(info.get("other", 0))},
        )

    sites.append(_make(0, None))

    for i, row in enumerate(moves, start=1):
        meta = _meta(row)
        direction = str(meta.get("direction") or "").strip().lower()
        steps = _int_or_none(meta.get("steps"))
        temp = meta.get("temperature_k")
        temp = float(temp) if isinstance(temp, (int, float)) and not isinstance(temp, bool) else None

        if direction in AXIS_OF and steps is not None and steps > 0:
            if AXIS_OF[direction] == "x":
                x += SIGN_OF[direction] * steps
            else:
                y += SIGN_OF[direction] * steps
            total_steps += abs(steps)
        else:
            # A move happened; how far and which way is lost. From here on the
            # ABSOLUTE odometer is fiction. Keep counting sites (the generation
            # is still real) but stop claiming to know where they are.
            known = False

        sites.append(_make(i, temp))

    if sites:
        sites[-1] = replace(sites[-1], is_current=True)
    return sites


def budget_used(sites: Sequence[CoarseSite]) -> dict[str, int]:
    """Absolute travel from the origin on each axis, in steps."""
    cur = sites[-1] if sites else CoarseSite(index=0)
    return {"x": abs(int(cur.x_steps)), "y": abs(int(cur.y_steps))}


# ── Planning ─────────────────────────────────────────────────────────────────

def _blocked_by(candidate: tuple[int, int], sites: Sequence[CoarseSite],
                cfg: CoarseMapConfig,
                move_steps: float | None = None) -> tuple[bool, float]:
    """Does the candidate overlap a visited site's blob? ``(blocked, clearance)``.

    Required separation is the configured spacing plus the **relative**
    uncertainty between the candidate and that particular site — not the site's
    uncertainty from the origin.

    That distinction matters and is easy to get wrong. Both positions are derived
    from the same odometer, so the error they share cancels: what is uncertain
    between two sites is only the travel BETWEEN them. Using the absolute
    (from-origin) uncertainty instead makes the requirement grow without bound as
    the run goes on, so after a handful of relocations every direction is
    "blocked" and the map declares a perfectly good sample exhausted.

    Errors add in quadrature over the moves separating the two, since individual
    step sizes scatter about a mean rather than all erring the same way."""
    cx, cy = candidate
    n_cand = (sites[-1].index + 1) if sites else 0
    worst = math.inf
    blocked = False
    for s in sites:
        if not s.position_known:
            continue
        d = math.hypot(cx - s.x_steps, cy - s.y_steps)
        gap_moves = max(1, n_cand - int(s.index))
        span = float(move_steps or cfg.default_move_steps) * gap_moves
        # 只看**这两个站点之间走了多少步**;拆成几次命令无关(见 uncertainty_steps)。
        rel_unc = cfg.uncertainty_steps(span)
        need = cfg.site_spacing_steps + rel_unc
        worst = min(worst, d - need)
        if d < need:
            blocked = True
    return blocked, (0.0 if worst is math.inf else worst)


def plan_relocation(sites: Sequence[CoarseSite], cfg: CoarseMapConfig,
                    *, steps: int | None = None) -> tuple[RelocationPlan | None, str]:
    """Where to move next. ``(plan, note)``; ``plan is None`` means "nowhere good".

    Candidates are the four lateral moves of a fixed size from the current site.
    Each is rejected for a stated reason — lands on a visited blob, or runs past
    the axis travel budget — and among the survivors **the one closest to the
    origin wins**, i.e. the one that keeps total travel smallest.

    ## 2026-08-18:为什么不是「clearance 最大的赢」

    已知问题:粗动大地图的推荐方向总是把人引向 +X。

    原来的排序是 ``sort(key=-clearance)``。而 ``clearance`` 是**离所有已访问站点
    的最小余量** —— 最大化它,字面意思就是「尽量远离去过的地方」。于是:

    * 四个方向完全对称时(第一次移动)它们**并列**,稳定排序保持插入顺序
      ``("x+", "x-", "y+", "y-")`` ⇒ 永远选 x+;
    * 对称一破,**向外那个 clearance 必然更大**(离旧站点更远)⇒ 继续向外。

    两者叠起来就是「一路 +X 跑到行程尽头」:前 10 次换区就跑出 3010 步。
    而这个 docstring 当时写着「ties broken by preferring an axis with more budget
    left」—— **那条并列规则从来没有实现过**(排序只有一个键)。照它实现还更糟:
    实测只能用出 329 个站点(见下表)。

    ## clearance 是**约束**,不是目标

    它的职责是「别落在用过的地方上」,而那件事上面的 ``blocked`` 已经做完了。
    一个已经合格的落点,余量再大也不多买到任何东西:210 步外的新鲜表面和 600 步外
    的一样新鲜,而近的那个**走得少**。走得少直接便宜三件事:少花行程预算、
    少积累位置不确定度(它按 ``√总步数`` 长,见 :meth:`CoarseMapConfig.
    uncertainty_steps`)、以及**留在样品上那块已知是好的区域附近**。

    所以现在:``blocked`` 当约束,**离原点最近**当目标。走出来的是螺旋
    (``x+ y+ x- x- y- y- x+ x+ x+ y+``),不是直线。

    ==============================  ========  ==================
    策略                            用出站点   前 10 次跑到多远
    ==============================  ========  ==================
    clearance 最大(原来)              1189       3010 步
    「剩余预算多的轴」(原 docstring)     329       —
    **离原点最近(现在)**            **4001+**   **602 步**
    ==============================  ========  ==================

    (同一套参数:``move=301``、单轴预算 10000。4001 是我的模拟上限,不是它的极限。)

    ``clearance`` 仍然算、仍然报(``RelocationPlan.clearance_steps`` 和那句
    「距最近的已访问站点还余 N 步」)—— 它是给人看的余量,只是不再当排序目标。

    Under an unknown odometer only "keep going the way we were" is offered: a
    relative move along one axis is still monotone even when the absolute
    position is lost, whereas a return trip needs a number we no longer have."""
    if not sites:
        return None, "还没有任何粗动记录。"
    cur = sites[-1]
    move = int(steps or cfg.default_move_steps)
    if move <= 0:
        return None, "移动步数必须为正。"

    if not cur.position_known:
        last_dir = _last_known_direction(sites)
        if last_dir is None:
            return None, (
                "粗动里程表已失效(有一次粗动没有记下方向/步数),而且找不到最近一次"
                "已知方向 —— 无法在不知道自己在哪的情况下规划落点。"
                "请用户补记一次粗动方向,或直接手动移动后用 record_coarse_move 补记。")
        axis = AXIS_OF[last_dir]
        return RelocationPlan(
            axis=axis, direction=last_dir, steps=move,
            lands_at=(cur.x_steps, cur.y_steps), clearance_steps=0.0,
            reason=(f"粗动里程表已失效(某次粗动缺方向/步数),因此**只允许沿上一次的方向"
                    f"{last_dir} 继续前进** {move} 步,不允许回头 —— "
                    "相对位移在单轴上仍然单调,而回到某个具体位置需要绝对坐标,"
                    "那个数字现在是假的。")), "里程表不确定:只给同向前进。"

    # An open-loop stage needs MARGIN, not just nominal spacing: the destination
    # has to clear the old site by the spacing PLUS the travel uncertainty. So a
    # move of exactly `site_spacing_steps` can never satisfy its own requirement.
    # Rather than declaring the sample exhausted over that arithmetic, grow the
    # move until it fits and SAY SO — "you need 260 steps, not 200" is the answer
    # the caller actually wanted.
    grown = False
    attempts = 0
    while attempts < 8:
        attempts += 1
        candidates: list[tuple[float, RelocationPlan]] = []
        rejected: list[str] = []
        budget_hit = False
        for direction in ("x+", "x-", "y+", "y-"):
            axis = AXIS_OF[direction]
            sign = SIGN_OF[direction]
            nx = cur.x_steps + (sign * move if axis == "x" else 0)
            ny = cur.y_steps + (sign * move if axis == "y" else 0)
            travelled = abs(nx) if axis == "x" else abs(ny)
            if travelled > cfg.axis_step_budget:
                budget_hit = True
                rejected.append(f"{direction}:会走到 {travelled} 步,超出单轴行程预算 "
                                f"{cfg.axis_step_budget} 步")
                continue
            blocked, clearance = _blocked_by((nx, ny), sites, cfg, move)
            if blocked:
                rejected.append(f"{direction}:落点会压在已经去过的站点上"
                                f"(需要 {cfg.site_spacing_steps} 步间距 + 这段路程的"
                                f"不确定量)")
                continue
            grow_note = (f"(已从 {cfg.default_move_steps} 步自动加大到 {move} 步 —— "
                         "开环粗动需要留出行程不确定量的余量)" if grown else "")
            candidates.append((clearance, RelocationPlan(
                axis=axis, direction=direction, steps=move, lands_at=(nx, ny),
                clearance_steps=clearance,
                reason=(f"沿 {direction} 走 {move} 步到 ({nx}, {ny}) 步位置{grow_note},"
                        f"距最近的已访问站点还余 {clearance:.0f} 步。"
                        f"当前站点 #{cur.index} 已用 "
                        f"{cur.summary.get('scans', 0)} 次扫图 / "
                        f"{cur.summary.get('damage', 0)} 次破坏"
                        # 未归类的那些也要露面 —— 否则一个刚跑完修针、记了几十条
                        # 事件的站点会显示成「0 扫图 / 0 破坏」,读起来像没动过。
                        + (f" / 另有 {cur.summary.get('other', 0)} 条未归类事件"
                           if cur.summary.get("other") else "")
                        + "。"))))

        if candidates:
            # 目标 = **少走**(离原点最近);``clearance`` 已经在 ``blocked`` 那里
            # 当过约束了,这里只拿它打破并列。理由见本函数 docstring。
            #
            # 用平方距离:整数、精确,不引入浮点并列。
            candidates.sort(key=lambda t: (t[1].lands_at[0] ** 2
                                           + t[1].lands_at[1] ** 2, -t[0]))
            note = f"已排除 {len(rejected)} 个方向。" if rejected else ""
            return candidates[0][1], note

        if budget_hit and all("行程预算" in r for r in rejected):
            # Growing the move cannot help when the wall is the travel budget.
            break
        if steps is not None:
            # The caller named a size; do not silently substitute another.
            break
        move = int(move * 1.5) + 1
        grown = True

    return None, ("四个方向都不能走:" + ";".join(rejected) +
                  "。样品这一带已经用完了 —— 需要换样品,"
                  "或由用户放宽单轴行程预算/站点间距。")


def _last_known_direction(sites: Sequence[CoarseSite]) -> str | None:
    """The direction of the most recent move whose direction we DO know.

    Derived from consecutive site positions rather than re-reading the rows, so
    it stays consistent with whatever ``derive_sites`` decided."""
    for i in range(len(sites) - 1, 0, -1):
        a, b = sites[i - 1], sites[i]
        dx, dy = b.x_steps - a.x_steps, b.y_steps - a.y_steps
        if dx and not dy:
            return "x+" if dx > 0 else "x-"
        if dy and not dx:
            return "y+" if dy > 0 else "y-"
    return None


def build_coarse_map(rows: Sequence[dict], cfg: CoarseMapConfig,
                     *, steps: int | None = None) -> CoarseMap:
    """The one entry point: rows in, map + suggestion out."""
    sites = derive_sites(rows, cfg)
    plan, note = plan_relocation(sites, cfg, steps=steps)
    return CoarseMap(sites=sites, current=sites[-1], suggestion=plan, note=note,
                     budget_used=budget_used(sites), config=cfg)
