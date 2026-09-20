"""Built-in declarative composite templates.

These are starting-point :class:`CompositeSpec` blueprints that exercise the
full IR — conditionals, ``while``/``repeat`` loops, nested loops, result-bound
conditions — so a user can clone one ("以自身为蓝本复制成类似的技能") and adapt
it instead of authoring a spec from scratch.

**Every step here must name a skill that actually exists, with that skill's real
parameter names, and read result keys that skill actually returns.**

(): all five templates were written against *plausible*
skill names — ``AssessTipQuality``, ``LogNote``, ``BiasSpectroscopy`` — which
have never existed in v2's 418-skill registry. Nothing was deleted or renamed;
the specs were simply authored against an imagined API and never checked. The
loader's unknown-skill guard therefore refused all five at every startup, and
the two automatic tip-repair loops (``ConditionTipUntilSharp``,
``ScanThenConditionTip``) were absent for the whole session in which the tip
degraded five times. Because a refused spec just doesn't appear in the UI, no
one could tell "broken" from "never designed".

They were broken past the skill names too: ``MoveToXY`` takes ``x_m``/``y_m``
(the specs passed ``x``/``y``), ``FullScan`` requires a centre and an extent
(the specs passed ``{}``), and ``AssessImageQuality`` returns ``fft_quality``,
not ``quality``. So even after the loader accepted them they would have failed
mid-run. All three classes of error are now pinned by
``tests/v2/unit/skills/test_composite_templates_real_skills.py``, which resolves
every step against a live registry.

:func:`seed_templates` writes any missing template into a version store so a
fresh install ships with examples to clone.
"""

from __future__ import annotations

import logging

from mast.skills.composite.spec import CompositeSpec, ParamSpec

logger = logging.getLogger(__name__)


def _condition_tip_until_sharp() -> CompositeSpec:
    """while-loop with a result-bound exit condition + accumulators."""
    return CompositeSpec(
        name="ConditionTipUntilSharp",
        description="反复脉冲修复针尖,每次重扫并评估图像质量,直到合格或达到最大次数。",
        safety_level="confirm",
        params=[
            ParamSpec("pulse_v", "number", 3.0, "修复脉冲电压 (V)", True),
            ParamSpec("center_x_m", "number", 0.0, "检验扫描中心 X (m)", True),
            ParamSpec("center_y_m", "number", 0.0, "检验扫描中心 Y (m)", True),
            ParamSpec("check_scan_m", "number", 2e-8, "检验扫描边长 (m,默认 20 nm)", False),
            ParamSpec("quality_threshold", "number", 0.3,
                      "合格 FFT 质量阈值 (AssessImageQuality.fft_quality)", False),
            ParamSpec("max_attempts", "int", 5, "最大尝试次数", False),
        ],
        nodes=[
            {"type": "set", "id": "init_attempts", "var": "attempts", "value": "0"},
            {"type": "set", "id": "init_sharp", "var": "sharp", "value": "False"},
            {"type": "loop", "id": "repair_loop", "mode": "while",
             "cond": "not sharp and attempts < max_attempts", "var": "iter",
             "max_iter": 100, "body": [
                {"type": "step", "id": "pulse", "skill": "BiasPulse",
                 "params": {"bias_v": {"$expr": "pulse_v"}, "width_s": 0.1}},
                # Tip quality is not directly measurable — scan, then judge the
                # image. (There is no AssessTipQuality skill; that name is what
                # made this whole template unregisterable.)
                {"type": "step", "id": "recheck", "skill": "FullScan", "params": {
                    "center_x_m": {"$expr": "center_x_m"},
                    "center_y_m": {"$expr": "center_y_m"},
                    "width_m": {"$expr": "check_scan_m"},
                    "height_m": {"$expr": "check_scan_m"}}},
                {"type": "step", "id": "q", "skill": "AssessImageQuality",
                 "params": {}},
                {"type": "set", "id": "upd_sharp", "var": "sharp",
                 "value": "q['fft_quality'] >= quality_threshold"},
                {"type": "set", "id": "upd_attempts", "var": "attempts",
                 "value": "attempts + 1"},
             ]},
            {"type": "if", "id": "verdict", "cond": "sharp", "then": [
                {"type": "succeed", "id": "accept",
                 "reason": "'tip conditioned after ' + str(attempts) + ' pulse(s)'"},
            ], "else": [
                {"type": "fail", "id": "give_up",
                 "reason": "'tip still blunt after ' + str(attempts) + ' attempt(s)'"},
            ]},
        ],
        tags=["tip", "conditioning", "template"],
        notes=("Template: pulse → re-scan → FFT quality, looping until sharp. "
               "Clone and retune pulse_v / quality_threshold for your tip."),
    )


def _grid_sts_with_precheck() -> CompositeSpec:
    """Nested repeat loops + a per-point conditional skip."""
    return CompositeSpec(
        name="GridSTSWithPrecheck",
        description="在 nx×ny 网格上逐点移动并做 STS;每点先质量预检,不合格则跳过。",
        safety_level="confirm",
        params=[
            ParamSpec("nx", "int", 3, "X 方向点数", True),
            ParamSpec("ny", "int", 3, "Y 方向点数", True),
            ParamSpec("x0", "number", 0.0, "起点 X (m)", False),
            ParamSpec("y0", "number", 0.0, "起点 Y (m)", False),
            ParamSpec("spacing_m", "number", 1e-9, "点间距 (m)", False),
            ParamSpec("min_quality", "number", 0.3,
                      "网格开跑前的最低 FFT 质量 (AssessImageQuality.fft_quality)", False),
            ParamSpec("max_radius_m", "number", 5e-9,
                      "只在距起点该半径内取谱 (m) —— 矩形网格上开圆形窗", False),
        ],
        nodes=[
            # Gate the whole grid on the CURRENT image once. Per-point tip
            # assessment isn't a thing (no such skill), and re-judging the same
            # image at every point would be theatre.
            {"type": "step", "id": "pre", "skill": "AssessImageQuality", "params": {}},
            {"type": "if", "id": "surface_ok",
             "cond": "pre['fft_quality'] < min_quality", "then": [
                {"type": "fail", "id": "bad_surface",
                 "reason": "'image quality below min_quality — grid not started'"},
             ]},
            {"type": "set", "id": "init_skipped", "var": "skipped", "value": "0"},
            {"type": "loop", "id": "row", "mode": "repeat", "count": "nx", "var": "ix",
             "body": [
                {"type": "loop", "id": "col", "mode": "repeat", "count": "ny", "var": "iy",
                 "body": [
                    # Per-point conditional on a REAL quantity: skip grid nodes
                    # outside a circular window.
                    {"type": "if", "id": "in_window",
                     "cond": ("(ix * spacing_m) ** 2 + (iy * spacing_m) ** 2 "
                              "<= max_radius_m ** 2"), "then": [
                        {"type": "step", "id": "move", "skill": "MoveToXY", "params": {
                            "x_m": {"$expr": "x0 + ix * spacing_m"},
                            "y_m": {"$expr": "y0 + iy * spacing_m"}}},
                        {"type": "step", "id": "sts", "skill": "AcquireSTS",
                         "params": {}},
                     ], "else": [
                        {"type": "set", "id": "skip", "var": "skipped",
                         "value": "skipped + 1"},
                     ]},
                 ]},
             ]},
        ],
        tags=["sts", "grid", "template"],
        notes=("Template: nested loops + a per-point conditional (circular window "
               "on a rectangular grid), gated on one up-front quality check."),
    )


def _scan_assess_rescan() -> CompositeSpec:
    """质量门控:扫描→评估→不合格则重扫,直到合格或达上限(while + 累加器)。"""
    return CompositeSpec(
        name="ScanAssessRescan",
        description="扫描成像后评估图像质量,不合格则重扫,直到合格或达到最大重扫次数。",
        safety_level="confirm",
        params=[
            ParamSpec("center_x_m", "number", 0.0, "扫描中心 X (m)", True),
            ParamSpec("center_y_m", "number", 0.0, "扫描中心 Y (m)", True),
            ParamSpec("scan_m", "number", 1e-7, "扫描边长 (m,默认 100 nm)", False),
            ParamSpec("quality_threshold", "number", 0.3,
                      "合格 FFT 质量阈值 (AssessImageQuality.fft_quality)", False),
            ParamSpec("max_rescans", "int", 3, "最大重扫次数", False),
        ],
        nodes=[
            {"type": "set", "id": "init_good", "var": "good", "value": "False"},
            {"type": "set", "id": "init_n", "var": "n", "value": "0"},
            {"type": "loop", "id": "rescan_loop", "mode": "while",
             "cond": "not good and n < max_rescans", "var": "iter",
             "max_iter": 50, "body": [
                {"type": "step", "id": "scan", "skill": "FullScan", "params": {
                    "center_x_m": {"$expr": "center_x_m"},
                    "center_y_m": {"$expr": "center_y_m"},
                    "width_m": {"$expr": "scan_m"},
                    "height_m": {"$expr": "scan_m"}}},
                {"type": "step", "id": "assess", "skill": "AssessImageQuality",
                 "params": {}},
                {"type": "set", "id": "upd_good", "var": "good",
                 "value": "assess['fft_quality'] >= quality_threshold"},
                {"type": "set", "id": "upd_n", "var": "n", "value": "n + 1"},
             ]},
            {"type": "if", "id": "verdict", "cond": "good", "then": [
                {"type": "succeed", "id": "accept",
                 "reason": "'scan accepted after ' + str(n) + ' pass(es)'"},
            ], "else": [
                {"type": "fail", "id": "give_up",
                 "reason": "'scan still poor after ' + str(n) + ' rescan(s)'"},
            ]},
        ],
        tags=["scan", "quality", "template"],
        notes=("Template: scan → FFT quality → rescan until good. Clone and "
               "retune quality_threshold / max_rescans."),
    )


def _line_profile_sts() -> CompositeSpec:
    """沿一条线等间距取 n 个点做 STS(repeat 循环 + 表达式坐标)。"""
    return CompositeSpec(
        name="LineProfileSTS",
        description="从起点沿固定方向等间距移动 n 个点,每点做一次 STS 谱采集(线扫谱)。",
        safety_level="confirm",
        params=[
            ParamSpec("n_points", "int", 5, "采点数", True),
            ParamSpec("x0", "number", 0.0, "起点 X (m)", False),
            ParamSpec("y0", "number", 0.0, "起点 Y (m)", False),
            ParamSpec("dx", "number", 1e-9, "每步 X 增量 (m)", False),
            ParamSpec("dy", "number", 0.0, "每步 Y 增量 (m)", False),
        ],
        nodes=[
            {"type": "loop", "id": "line", "mode": "repeat", "count": "n_points",
             "var": "i", "body": [
                {"type": "step", "id": "move", "skill": "MoveToXY", "params": {
                    "x_m": {"$expr": "x0 + i * dx"},
                    "y_m": {"$expr": "y0 + i * dy"}}},
                {"type": "step", "id": "sts", "skill": "AcquireSTS", "params": {}},
             ]},
        ],
        tags=["sts", "line", "template"],
        notes=("Template: equally-spaced MoveToXY + AcquireSTS along a line. "
               "Clone and change dx/dy for a different direction."),
    )


def _scan_then_condition_tip() -> CompositeSpec:
    """巡扫成像 → 质量评估 → 质量不足则脉冲修针(扫描-质量-条件分支)。"""
    return CompositeSpec(
        name="ScanThenConditionTip",
        description="扫描成像并评估质量;若质量低于阈值则脉冲修复针尖,否则记录通过。",
        safety_level="confirm",
        params=[
            ParamSpec("center_x_m", "number", 0.0, "扫描中心 X (m)", True),
            ParamSpec("center_y_m", "number", 0.0, "扫描中心 Y (m)", True),
            ParamSpec("scan_m", "number", 1e-7, "扫描边长 (m,默认 100 nm)", False),
            ParamSpec("quality_threshold", "number", 0.3,
                      "FFT 质量阈值 (AssessImageQuality.fft_quality)", False),
            ParamSpec("pulse_v", "number", 3.0, "修针脉冲电压 (V)", True),
        ],
        nodes=[
            {"type": "step", "id": "scan", "skill": "FullScan", "params": {
                "center_x_m": {"$expr": "center_x_m"},
                "center_y_m": {"$expr": "center_y_m"},
                "width_m": {"$expr": "scan_m"},
                "height_m": {"$expr": "scan_m"}}},
            {"type": "step", "id": "assess", "skill": "AssessImageQuality",
             "params": {}},
            {"type": "if", "id": "needs_fix",
             "cond": "assess['fft_quality'] < quality_threshold", "then": [
                {"type": "step", "id": "pulse", "skill": "BiasPulse",
                 "params": {"bias_v": {"$expr": "pulse_v"}, "width_s": 0.1}},
                {"type": "succeed", "id": "note_fix",
                 "reason": "'tip pulsed due to low image quality'"},
            ], "else": [
                {"type": "succeed", "id": "note_ok",
                 "reason": "'image quality acceptable — no pulse needed'"},
            ]},
        ],
        tags=["scan", "tip", "conditioning", "template"],
        notes=("Template: scan → FFT quality → pulse the tip only when needed. "
               "Clone and retune quality_threshold / pulse_v."),
    )


def builtin_templates() -> list[CompositeSpec]:
    return [
        _condition_tip_until_sharp(),
        _grid_sts_with_precheck(),
        _scan_assess_rescan(),
        _line_profile_sts(),
        _scan_then_condition_tip(),
    ]


def _references_unknown_skill(registry, spec) -> bool:
    """True iff *spec* names at least one step skill the registry doesn't have."""
    from mast.skills.composite.loader import _collect_step_skills

    try:
        for sk in _collect_step_skills(getattr(spec, "nodes", None)):
            if not registry.has(sk):
                return True
    except Exception:  # pragma: no cover - defensive
        return False
    return False


def seed_templates(store, registry=None) -> list[str]:
    """Save any missing built-in template into *store*. Returns names written.

    Also REPAIRS a previously-seeded template that is currently unregisterable.
    Seeding was ``if not exists: save``, so the broken specs shipped before
    2026-07-27 () would survive every upgrade — the machine that lost its
    tip-repair loops would still be missing them after installing the fix. When
    *registry* is supplied we re-seed a stored template that (a) still carries
    the ``template`` tag and (b) references a skill that does not exist, i.e.
    has no working behaviour to preserve, while the builtin replacing it fully
    resolves. Nothing is destroyed: ``store.save`` archives the prior version,
    so an operator whose own edit is overwritten can restore it from history.
    """
    seeded: list[str] = []
    for spec in builtin_templates():
        try:
            if not store.exists(spec.name):
                store.save(spec)
                seeded.append(spec.name)
                continue
            if registry is None:
                continue
            stored = store.load(spec.name)
            if "template" not in (getattr(stored, "tags", None) or ()):
                continue  # renamed/retagged by the operator — leave it alone
            if (_references_unknown_skill(registry, stored)
                    and not _references_unknown_skill(registry, spec)):
                store.save(spec)
                seeded.append(spec.name)
                logger.warning(
                    "composite template %s referenced skills that do not exist; "
                    "replaced with the current builtin (prior version archived)",
                    spec.name)
        except Exception as exc:  # pragma: no cover - best-effort seeding
            logger.warning("seed_templates: %s failed: %s", spec.name, exc)
    if seeded:
        logger.info("composite templates seeded: %s", ", ".join(seeded))
    return seeded


__all__ = ["builtin_templates", "seed_templates"]
