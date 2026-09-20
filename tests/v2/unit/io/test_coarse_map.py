"""The stage-scale map: where on the SAMPLE we have been, and where to go next.

The operator's requirement was not a rule but a picture — *「移动来移动去，我们要
记得不能回原来的地方对不对？」* — so "don't go back" is enforced as geometry on a
map rather than as a sentence somebody has to remember.

Two properties matter more than any individual number here:

* **Steps, not metres.** A coarse stepper is open-loop and its step size drifts
  with drive amplitude, load and (strongly) temperature. Converting steps to
  metres and then doing geometry would dress an estimate up as a coordinate.
* **The uncertainty is visible and RELATIVE.** A site is a blob, not a point.
  The blob used in a comparison is the uncertainty of the travel BETWEEN the two
  positions, not each one's error from the origin — those share a common history
  that cancels, and using the absolute figure makes the requirement grow without
  bound until a perfectly good sample is declared exhausted.
"""
from __future__ import annotations

import math

import pytest

from mast.io import coarse_map as cm


def _scan(epoch: int, ts: str = "t") -> dict:
    return {"kind": "scan", "coord_epoch": epoch, "timestamp": ts}


def _move(direction: str | None, steps: int | None, epoch: int = 0,
          temp: float | None = None) -> dict:
    meta: dict = {}
    if direction is not None:
        meta["direction"] = direction
    if steps is not None:
        meta["steps"] = steps
    if temp is not None:
        meta["temperature_k"] = temp
    return {"kind": "coarse_move", "coord_epoch": epoch, "meta": meta,
            "timestamp": "t"}


def _walk(*legs: tuple[str, int]) -> list[dict]:
    """Rows for a run that scanned, moved, scanned, moved... along ``legs``."""
    rows: list[dict] = []
    epoch = 0
    for direction, steps in legs:
        rows.append(_scan(epoch))
        rows.append(_move(direction, steps, epoch=epoch))
        epoch += 1
    rows.append(_scan(epoch))
    return rows


# ── Odometer ────────────────────────────────────────────────────────────────

def test_a_fresh_sample_has_exactly_one_site():
    sites = cm.derive_sites([_scan(0)], cm.CoarseMapConfig())
    assert len(sites) == 1
    assert (sites[0].x_steps, sites[0].y_steps) == (0, 0)
    assert sites[0].is_current and sites[0].position_known


def test_the_odometer_is_the_prefix_sum_of_the_move_events():
    """No new table, no new column: the same event stream ``coord_epoch`` is
    derived from, read a second way. A separate store of "where are we" would be
    a second source of truth to drift out of sync with the first."""
    sites = cm.derive_sites(_walk(("x+", 200), ("y+", 300), ("x-", 50)),
                            cm.CoarseMapConfig())
    assert [(s.x_steps, s.y_steps) for s in sites] == [
        (0, 0), (200, 0), (200, 300), (150, 300)]
    assert sites[-1].is_current


def test_site_index_is_the_coordinate_generation():
    sites = cm.derive_sites(_walk(("x+", 200), ("x+", 200)), cm.CoarseMapConfig())
    assert [s.index for s in sites] == [0, 1, 2]


def test_each_site_summarises_what_happened_there():
    """A blob labelled "3 scans, 1 crash" is a place you know something about;
    one labelled "0 scans" is a place you left without trying.

    ``other`` joined the tally on 2026-08-10: rows that none of the three named
    columns claim used to be dropped silently, so a site that had just absorbed
    26 events read exactly like one nobody had visited. The whole dict is still
    asserted (not just the keys that were interesting today) — that equality is
    what makes ADDING a column a deliberate act with a test to update, rather
    than something that slips in unnoticed. See
    ``test_map_ledger_does_not_round_to_zero.py`` for the behaviour itself.
    """
    rows = [_scan(0), _scan(0), {"kind": "crash", "coord_epoch": 0, "timestamp": "t"},
            _move("x+", 300, epoch=0), _scan(1)]
    sites = cm.derive_sites(rows, cm.CoarseMapConfig())
    assert sites[0].summary == {"scans": 2, "sts": 0, "damage": 1, "other": 0}
    assert sites[1].summary["scans"] == 1


def test_temperature_rides_along_with_the_move():
    """The same step count travels several times further at 300 K than at 4 K,
    so an entry without a temperature cannot be compared with the next one."""
    rows = [_scan(0), _move("x+", 200, epoch=0, temp=4.2), _scan(1)]
    sites = cm.derive_sites(rows, cm.CoarseMapConfig())
    assert sites[1].temperature_k == pytest.approx(4.2)


def test_legacy_null_epoch_reads_as_generation_zero():
    rows = [{"kind": "scan", "timestamp": "t"},
            {"kind": "coarse_move", "meta": {"direction": "x+", "steps": 100},
             "timestamp": "t"},
            {"kind": "scan", "timestamp": "t"}]
    sites = cm.derive_sites(rows, cm.CoarseMapConfig())
    assert len(sites) == 2 and sites[1].x_steps == 100


def test_meta_stored_as_json_text_is_parsed():
    rows = [{"kind": "coarse_move", "coord_epoch": 0,
             "meta": '{"direction": "y-", "steps": 40}', "timestamp": "t"}]
    sites = cm.derive_sites(rows, cm.CoarseMapConfig())
    assert (sites[1].x_steps, sites[1].y_steps) == (0, -40)


# ── Uncertainty ─────────────────────────────────────────────────────────────

def test_uncertainty_starts_at_zero_and_grows():
    sites = cm.derive_sites(_walk(("x+", 200), ("x+", 200), ("x+", 200)),
                            cm.CoarseMapConfig())
    unc = [s.uncertainty_steps for s in sites]
    assert unc[0] == 0.0
    assert unc == sorted(unc), "uncertainty must be monotone non-decreasing"
    assert unc[-1] > 0


def test_uncertainty_grows_as_sqrt_not_linearly():
    """Individual step sizes scatter about a mean; they do not all err the same
    way. Linear accumulation would draw blobs so large after a few moves that
    the map would stop saying anything.

    ⚠️ 2026-08-18:这条测试的**原版分辨不出两个模型**。它比的是
    ``(1 移动, 200 步)`` 与 ``(4 移动, 800 步)`` —— 移动数和步数同比例变,
    于是「按次算」和「按步算」两个公式都给出 ×2,两种都绿。
    真正的判据在下面 ``test_chunking_a_journey_does_not_change_its_uncertainty``:
    **固定总路程,只改怎么切命令**。
    """
    cfg = cm.CoarseMapConfig(step_uncertainty_frac=0.3)
    one = cfg.uncertainty_steps(200)
    four = cfg.uncertainty_steps(800)
    assert four == pytest.approx(one * math.sqrt(4))


def test_chunking_a_journey_does_not_change_its_uncertainty():
    """**同一段路,拆成几次命令走,模糊半径必须一样。**

    这一条是 2026-08-18 那个 bug 的判据。原实现是
    ``frac * mean_step * sqrt(n_moves)`` —— 每一次**命令**整体误差 30%,
    于是把 200 步拆成 16 次 12.5 步,误差从 60 步掉到 15 步。走过的路一样长,
    只因为多按了几次按钮就「更准了」,这在物理上说不通。

    误差的来源是**单步**的散布(参数名、profile 描述、设置页标签三处都写着
    「单步」),而 N 个独立步的和只跟 N 有关,与谁把它们分成几组无关。
    """
    cfg = cm.CoarseMapConfig(step_uncertainty_frac=0.3)
    # 总步数取 400:下面每个 chunks 都整除它。**整除是这条测试的前提** ——
    # 第一版用 200/16 取整成 192,于是断言红了,而红的是我的算术不是被测的公式。
    total = 400
    ref = cfg.uncertainty_steps(total)
    for chunks in (1, 2, 4, 8, 16):
        per = total // chunks
        assert per * chunks == total, f"{chunks} 不整除 {total} —— 测试自己的算术错了"
        walk = _walk(*(("x+", per),) * chunks)
        sites = cm.derive_sites(walk, cfg)
        assert sites[-1].uncertainty_steps == pytest.approx(ref), (
            f"拆成 {chunks} 次走同样的 {total} 步,模糊半径变了")


def test_the_blob_is_not_absurd_after_one_ordinary_move():
    """一次出厂默认的换区(200 步)之后,模糊半径必须**远小于站点间距**。

    「粗动大地图现在的默认误差大的离谱。」当时一次 200 步
    的换区就画出 ±60 步的 blob —— 而站点间距只有 200 步,也就是说刚走一次,
    模糊半径已经吃掉三成间距;走过十次是 ±190,几乎盖满整个间距。

    这条不钉具体数字(它随 ``step_uncertainty_frac`` 走),钉的是那个**比例**:
    一次常规换区之后,模糊半径不该超过站点间距的十分之一。
    """
    cfg = cm.CoarseMapConfig()
    unc = cfg.uncertainty_steps(cfg.default_move_steps)
    assert unc < cfg.site_spacing_steps / 10.0, (
        f"走一次 {cfg.default_move_steps} 步就模糊 ±{unc:.0f} 步,"
        f"而站点间距只有 {cfg.site_spacing_steps} 步")


def test_the_required_separation_stays_flat_as_the_run_goes_on():
    """走得越久,**新落点被要求离旧站点多远**不该越滚越大。

    ``_blocked_by`` 的 ``need = site_spacing_steps + rel_unc``。旧模型下这个数
    从 260 一路涨到 468(20 次换区之后),在 10000 步的行程预算里等于把可用站点
    砍掉一多半,然后报「没有可去的新站点了」—— 而模块里那段注释早就写过
    这个失效形状(「makes the requirement grow without bound」),
    只是当时只修了「用相对而不是绝对」,没修公式本身。
    """
    cfg = cm.CoarseMapConfig()
    need = [cfg.site_spacing_steps + cfg.uncertainty_steps(cfg.default_move_steps * n)
            for n in (1, 5, 10, 20)]
    growth = need[-1] / need[0]
    assert growth < 1.1, f"要求的间距 20 次之内涨了 {growth:.2f} 倍:{need}"


# ── Honest degradation when a move lacks details ────────────────────────────

def test_a_move_without_direction_kills_the_absolute_position():
    """The manual backfill path allows both fields to be blank, because knowing
    THAT a move happened is worth recording even when the details are lost. From
    that row on the absolute odometer is fiction and must say so."""
    rows = _walk(("x+", 200)) + [_move(None, None, epoch=1), _scan(2)]
    sites = cm.derive_sites(rows, cm.CoarseMapConfig())
    assert sites[1].position_known is True
    assert sites[2].position_known is False
    assert len(sites) == 3, "the generation is still real even when the vector is not"


def test_an_unknown_odometer_allows_only_continuing_the_same_way():
    """A relative move along one axis stays monotone with a broken odometer.
    A return trip does not — it needs the absolute position we no longer have."""
    rows = _walk(("x+", 300)) + [_move(None, None, epoch=1), _scan(2)]
    sites = cm.derive_sites(rows, cm.CoarseMapConfig())
    plan, note = cm.plan_relocation(sites, cm.CoarseMapConfig())
    assert plan is not None
    assert plan.direction == "x+", "must continue the last KNOWN direction"
    assert "不允许回头" in plan.reason
    assert "不确定" in note


def test_an_unknown_odometer_with_no_prior_direction_refuses():
    rows = [_scan(0), _move(None, None, epoch=0), _scan(1)]
    sites = cm.derive_sites(rows, cm.CoarseMapConfig())
    plan, note = cm.plan_relocation(sites, cm.CoarseMapConfig())
    assert plan is None
    assert "无法在不知道自己在哪的情况下规划落点" in note


# ── Planning ────────────────────────────────────────────────────────────────

def test_the_planner_never_lands_on_a_visited_site():
    cfg = cm.CoarseMapConfig()
    sites = cm.derive_sites(_walk(("x+", 300)), cfg)
    plan, _ = cm.plan_relocation(sites, cfg)
    assert plan is not None
    blocked, _ = cm._blocked_by(plan.lands_at, sites, cfg, plan.steps)
    assert not blocked


def test_a_long_run_of_relocations_keeps_finding_somewhere_to_go():
    """The regression that motivated the relative-uncertainty model.

    With the ABSOLUTE (from-origin) uncertainty in the requirement, the needed
    separation grew every move until all four directions were "blocked" and a
    perfectly good sample was declared exhausted after a handful of relocations."""
    cfg = cm.CoarseMapConfig(axis_step_budget=100_000)
    legs = [("x+", 400)] * 12
    sites = cm.derive_sites(_walk(*legs), cfg)
    plan, note = cm.plan_relocation(sites, cfg)
    assert plan is not None, f"declared exhausted after 12 moves: {note}"


def test_the_planner_grows_the_move_rather_than_giving_up():
    """A move of exactly ``site_spacing_steps`` can never satisfy its own
    requirement, because the requirement is spacing PLUS travel uncertainty.
    Answering "you need 260 steps, not 200" beats answering "nowhere to go"."""
    cfg = cm.CoarseMapConfig(site_spacing_steps=200, default_move_steps=200)
    sites = cm.derive_sites(_walk(("x+", 300)), cfg)
    plan, _ = cm.plan_relocation(sites, cfg)
    assert plan is not None
    assert plan.steps > cfg.default_move_steps
    assert "自动加大" in plan.reason


def test_an_explicitly_requested_size_is_never_silently_substituted():
    cfg = cm.CoarseMapConfig(site_spacing_steps=200, default_move_steps=200)
    sites = cm.derive_sites(_walk(("x+", 300)), cfg)
    plan, _ = cm.plan_relocation(sites, cfg, steps=205)
    assert plan is None or plan.steps == 205


def test_the_travel_budget_is_a_wall_growing_cannot_climb():
    """Growing the move helps against a proximity block; against the end of the
    stage's travel it makes things worse. The two rejections must not be
    conflated, or the planner loops enlarging a move it can never take."""
    cfg = cm.CoarseMapConfig(axis_step_budget=50, site_spacing_steps=100,
                             default_move_steps=100)
    sites = cm.derive_sites([_scan(0)], cfg)
    plan, note = cm.plan_relocation(sites, cfg)
    assert plan is None
    assert "行程预算" in note
    assert "需要换样品" in note, "the caller must be told what the real remedy is"


def test_a_backwards_move_is_fine_when_it_lands_somewhere_new():
    """"Don't go back" means "don't re-use a patch", not "never reverse a sign".

    A sample is two-dimensional; after stepping x+ then y+, the region at
    (small x, large y) has never been visited and is a perfectly good target."""
    cfg = cm.CoarseMapConfig(site_spacing_steps=100, default_move_steps=150,
                             axis_step_budget=10_000)
    sites = cm.derive_sites(_walk(("x+", 300), ("y+", 300)), cfg)
    plan, _ = cm.plan_relocation(sites, cfg)
    assert plan is not None
    blocked, _ = cm._blocked_by(plan.lands_at, sites, cfg, plan.steps)
    assert not blocked, "whatever it chose, it must not be a visited patch"


def test_the_reason_is_a_complete_chinese_sentence_with_the_numbers_in_it():
    """agent and operator read the SAME sentence — so when they disagree about
    what to do next, they are at least disagreeing about the same words."""
    cfg = cm.CoarseMapConfig()
    sites = cm.derive_sites(_walk(("x+", 400)), cfg)
    plan, _ = cm.plan_relocation(sites, cfg)
    assert plan is not None
    assert "步" in plan.reason and str(plan.steps) in plan.reason
    assert "距最近的已访问站点" in plan.reason


# ── Metres are an annotation, never an input ────────────────────────────────

def test_metres_appear_only_as_an_annotation():
    cfg = cm.CoarseMapConfig(step_m=1e-7)
    sites = cm.derive_sites(_walk(("x+", 200)), cfg)
    d = sites[1].as_dict(cfg.step_m)
    assert d["x_steps"] == 200
    assert d["approx_x_um"] == pytest.approx(20.0)


def test_planning_is_identical_with_and_without_a_step_calibration():
    """Nothing may navigate by ``xy_motor_step_m``. It is an open-loop estimate
    whose value drifts with temperature; the repo's existing rule is that
    nothing computes with it, and this keeps that promise."""
    rows = _walk(("x+", 300), ("y+", 300))
    a, _ = cm.plan_relocation(cm.derive_sites(rows, cm.CoarseMapConfig(step_m=None)),
                              cm.CoarseMapConfig(step_m=None))
    b, _ = cm.plan_relocation(cm.derive_sites(rows, cm.CoarseMapConfig(step_m=5e-7)),
                              cm.CoarseMapConfig(step_m=5e-7))
    assert a is not None and b is not None
    assert (a.direction, a.steps, a.lands_at) == (b.direction, b.steps, b.lands_at)


# ── Top level ───────────────────────────────────────────────────────────────

def test_build_coarse_map_reports_budget_and_serialises():
    cfg = cm.CoarseMapConfig()
    m = cm.build_coarse_map(_walk(("x+", 400), ("y-", 400)), cfg)
    assert m.budget_used == {"x": 400, "y": 400}
    d = m.as_dict()
    assert d["current_index"] == 2
    assert len(d["sites"]) == 3
    assert d["axis_step_budget"] == cfg.axis_step_budget


def test_an_empty_row_set_still_yields_the_origin_site():
    """No markers means a fresh sample sitting at site 0 — NOT "no information".
    (The "cannot read the record" case is signalled by the provider returning
    None, and is handled by the caller, not here.)"""
    m = cm.build_coarse_map([], cm.CoarseMapConfig())
    assert len(m.sites) == 1 and m.current.index == 0
    assert m.suggestion is not None, "a fresh sample can always be relocated on"


# ── 推荐方向:不许只会往一个方向跑(2026-08-18) ──────────────────────────


def _autopilot(cfg: "cm.CoarseMapConfig", n: int):
    """连着让规划器自己走 n 次,返回 (方向序列, 落点序列)。"""
    import json as _json

    rows: list[dict] = []
    dirs: list[str] = []
    pts: list[tuple[int, int]] = [(0, 0)]
    for k in range(n):
        sites = cm.derive_sites(rows, cfg)
        plan, note = cm.plan_relocation(sites, cfg)
        assert plan is not None, f"第 {k + 1} 次就规划不出来了:{note}"
        dirs.append(plan.direction)
        pts.append(plan.lands_at)
        rows.append({"kind": "coarse_move", "coord_epoch": k,
                     "meta": _json.dumps({"direction": plan.direction,
                                          "steps": plan.steps}),
                     "timestamp": "t"})
    return dirs, pts


def test_the_recommendation_is_not_always_the_same_direction():
    """**这一条就是那个 bug。**

    「粗动大地图的推荐方向为什么只会让人往 +X 疯狂跑?」

    原因不是「缺一条并列规则」,是**目标函数本身**:排序键是 ``-clearance``,
    而 clearance 是「离所有已访问站点的最小余量」—— 最大化它字面意思就是
    「尽量远离去过的地方」。四方向对称时并列(稳定排序 ⇒ 取元组里第一个 = x+),
    对称一破则向外那个必然更大 ⇒ 一路 +X 到行程尽头。
    """
    dirs, _pts = _autopilot(cm.CoarseMapConfig(), 12)
    assert len(set(dirs)) > 1, f"12 次换区全是同一个方向:{dirs}"
    # 更强:头四次里就该出现两个不同的轴,而不是走到墙才拐弯。
    assert len({cm.AXIS_OF[d] for d in dirs[:4]}) == 2, (
        f"前四次没换过轴:{dirs[:4]}")


def test_it_stays_near_home_instead_of_running_to_the_edge():
    """**近处的新鲜表面和远处的一样新鲜,而近的走得少。**

    走得少直接便宜三件事:少花行程预算、少积累位置不确定度(按 √总步数 长)、
    以及留在样品上那块已知是好的区域附近。

    原来前 10 次换区就跑出 3010 步(= 单轴预算的三成);现在 602 步。
    这条钉的是那个**量级**:十次换区之后离原点的距离不该超过「一直朝一个方向走」
    的一半。
    """
    cfg = cm.CoarseMapConfig()
    _dirs, pts = _autopilot(cfg, 10)
    far = max(math.hypot(x, y) for x, y in pts)
    straight = 10 * cfg.default_move_steps   # 一路朝一个方向走会跑到的距离
    assert far < straight / 2, (
        f"十次换区跑出 {far:.0f} 步,而一路直走也才 {straight} 步 —— 它在直线狂奔")


def test_clearance_is_a_constraint_not_an_objective():
    """已经合格的落点,余量再大也不多买到任何东西 —— 所以**近的赢**。

    ``clearance`` 的职责是「别落在用过的地方上」,而那件事 ``blocked`` 已经做完了。
    再拿它当排序目标,等于说「离去过的地方越远越好」,那正是「疯狂往 +X 跑」的
    发动机。

    这一条钉的是**策略本身**:规划器给出的落点,必须是四个合格候选里离原点最近的
    那一个。(它读起来接近实现的复述 —— 那是有意的:这是一个**决定**,
    而决定要有一处写下来的地方,否则下次有人把排序键改回去不会有任何东西反对。)
    """
    cfg = cm.CoarseMapConfig()
    _dirs, _pts = _autopilot(cfg, 2)     # 走两步,让局面不再四向对称
    rows = [{"kind": "coarse_move", "coord_epoch": i,
             "meta": f'{{"direction": "{d}", "steps": {cfg.default_move_steps}}}',
             "timestamp": "t"}
            for i, d in enumerate(_dirs)]
    sites = cm.derive_sites(rows, cfg)
    plan, _note = cm.plan_relocation(sites, cfg)
    assert plan is not None
    cur = sites[-1]
    reachable = []
    for d in ("x+", "x-", "y+", "y-"):
        ax, sg = cm.AXIS_OF[d], cm.SIGN_OF[d]
        nx = cur.x_steps + (sg * plan.steps if ax == "x" else 0)
        ny = cur.y_steps + (sg * plan.steps if ax == "y" else 0)
        if not cm._blocked_by((nx, ny), sites, cfg, plan.steps)[0]:
            reachable.append((math.hypot(nx, ny), d))
    assert len(reachable) > 1, "构造失败:只剩一个方向可走,这条测试什么都没测"
    nearest = min(r for r, _ in reachable)
    assert math.hypot(*plan.lands_at) == pytest.approx(nearest), (
        f"选了 {plan.direction} 落在 {math.hypot(*plan.lands_at):.0f} 步处,"
        f"而最近的合格落点只有 {nearest:.0f} 步:{sorted(reachable)}")


def test_the_walk_is_a_spiral():
    """走出来应该是螺旋 —— 一圈一圈往外,而不是一条直线。

    判据不看具体形状(那会把实现钉死),看**它有没有回到过负半轴**:
    直线狂奔永远只在一个象限,螺旋会绕过原点四周。
    """
    _dirs, pts = _autopilot(cm.CoarseMapConfig(), 9)
    assert any(x < 0 for x, _ in pts), f"x 从来没到过负半轴:{pts}"
    assert any(y < 0 for _, y in pts), f"y 从来没到过负半轴:{pts}"
