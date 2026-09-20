"""特异化针尖锻造的开工前体检 —— 无硬件也能跑完大半。

``TipConditioningSelfCheck`` 体检的是「把针尖修到基础态」那条流程;这一份体检的是
它之后的两条:做 STS 的金属性针尖、原子分辨针尖。分开是因为
``REQUIRED_SKILLS`` 有一条**元测试**钉着它必须与流程实际调用的技能精确对齐 ——
两个流程的技能集混在一起，缺谁都说不清。

体检的重点与修针那份不同,因为这两条流程的失败方式不同:

* 修针失败通常是**硬件**没准备好(Tip Shaper 模块没开、针尖没登记);
* 锻造失败更多是**判据无从建立**:不知道衬底就定不出表面态的期望位置;帧参数
  选得太粗就判不出原子相。这两件事都能在开工前算出来,而它们各自会浪费掉十几次
  扎针和半小时的扫描。

所以这里额外查两项别处不查的:**衬底能不能定出判据**,以及**评估帧的像素尺度**。

判据本身也在这里干跑一遍(合成台阶谱、合成晶格、纯噪声),这样「判据模块坏了」会
在开工前就暴露,而不是等到真机上拿到一条谱之后。
"""
from __future__ import annotations

import logging
from typing import Any

from mast.core.registry import registered_skill_names
from mast.core.types import (
    ParameterSpec,
    SafetyLevel,
    SkillCategory,
    SkillMetadata,
    SkillResult,
)
from mast.skills.base import BaseSkill

logger = logging.getLogger(__name__)

#: 两条锻造流程真正会调到的技能。
#:
#: 判据是「被 ``builtins.__init__`` / ``composite.__init__`` import 过」而不是
#: 「写进 __all__」—— 冻结构建里前者才决定技能存不存在(见
#: ``builtins/__init__.py`` 的冻结兜底注释)。
REQUIRED_SKILLS: tuple[str, ...] = (
    # 本层新增
    "MakeSpectroscopyTip", "MakeAtomicResolutionTip",
    "AssessShockleyOnset", "AssessAtomicPhase", "BiasWiggle",
    # 被当作子步骤调用的既有技能
    "PokeConditionTip", "TipShapeWithReadback", "AssessClusterRoundness",
    "FindCleanSpot", "FindFlatRegion", "MoveToXY", "ScanAt", "SaveScan",
    "GetLatestScanFile", "ConfigureScan", "StartScan", "StopScan",
    "SetBias", "SetSetpoint", "ZControllerOnOff",
    "ConfigureLockIn", "ConfigureSTS", "AcquireSTS",
)


class TipForgeSelfCheck(BaseSkill):
    """开工前体检：两条特异化针尖锻造流程现在跑得起来吗。"""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="TipForgeSelfCheck",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description=(
                "特异化针尖配方（MakeSpectroscopyTip / MakeAtomicResolutionTip）的预检：技能在这个 "
                "build 里在不在、衬底能不能解出 Shockley onset、评估帧到底分不分得开晶格、"
                "判据模块能不能工作。只读；除一次 Tip Shaper 探测外无需硬件。结论看 data.ready ，"
                "不是 success。"
            ),
            parameters=[
                ParameterSpec(
                    name="substrate", type="str",
                    description=("用来对照检查的衬底。留空则用已登记的样品。"),
                    required=False, default=""),
            ],
            estimated_duration_s=4.0,
            composition_level=1,
            tags=["tip", "forge", "selfcheck", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        calls: list = []
        checks: list[dict[str, Any]] = []
        blockers: list[str] = []
        warnings: list[str] = []

        def add(name: str, ok: "bool | None", detail: str, *, blocking=False):
            checks.append({"check": name, "ok": ok, "detail": detail})
            if ok is False:
                (blockers if blocking else warnings).append(f"{name}: {detail}")
            elif ok is None:
                warnings.append(f"{name}: {detail}")

        # ── 1. 技能是否真的在这个构建里 ────────────────────────────────
        registry = getattr(context, "_registry", None)
        installed = registered_skill_names(registry)
        missing = [n for n in REQUIRED_SKILLS if n not in installed]
        add("技能齐全", not missing,
            f"全部就位（本机注册 {len(installed)} 个）" if not missing
            else f"缺 {len(missing)} 个: {', '.join(missing)}",
            blocking=True)

        # ── 2. 衬底 → 判据期望值 ───────────────────────────────────────
        # 这是锻造流程最容易白跑的一环:不知道衬底就定不出表面态该在哪,而配方会
        # 扎十几次才发现判据无从建立。
        substrate_name = ""
        try:
            from mast.core.sample_facts import resolve_substrate

            facts = resolve_substrate(params.get("substrate") or None)
            substrate_name = facts.material or ""
            if not facts.available:
                add("衬底可解析", False,
                    f"{facts.reason}（配方 1 会直接拒绝执行；配方 2 只是少一项"
                    f"晶格常数核对，仍可跑）", blocking=False)
            elif facts.surface_state_onset_v is None:
                add("衬底可解析", True,
                    f"{facts.material}（来源：{facts.source}）")
                add("有肖克利表面态", False,
                    f"{facts.material} 没有表面态 —— 配方 1（STS 金属性针尖）"
                    f"在这块衬底上不适用，用台阶锐度判据代替。")
            else:
                add("衬底可解析", True,
                    f"{facts.material}（来源：{facts.source}）")
                add("有肖克利表面态", True,
                    f"onset 期望 {facts.surface_state_onset_v * 1000:.0f} mV"
                    + (f"，原子行间距 {facts.row_spacing_nm:.3f} nm"
                       if facts.row_spacing_nm else ""))
        except Exception as exc:  # noqa: BLE001
            add("衬底可解析", None, f"查不了（{exc}）")

        # ── 3. 评估帧的像素尺度 ────────────────────────────────────────
        # 判不出原子相的帧不值得扫。这一项在开工前就能算,不需要硬件。
        try:
            from mast.core.special_tip_workflow import ATOMIC_TIP

            problem = ATOMIC_TIP.scale_problem()
            nmpp = ATOMIC_TIP.eval_pixel_size_nm()
            add("评估帧能分辨原子", not problem,
                (f"{ATOMIC_TIP.eval_frame_nm:g} nm / {ATOMIC_TIP.eval_pixels} px "
                 f"= {nmpp:.4f} nm/px（满权重档）") if not problem else problem,
                blocking=bool(problem))
        except Exception as exc:  # noqa: BLE001
            add("评估帧能分辨原子", None, f"查不了（{exc}）")

        # ── 4. 判据模块干跑(合成数据,不碰硬件) ─────────────────────────
        # 「判据模块坏了」应该在开工前暴露,而不是等真机上拿到一条谱之后。
        try:
            import numpy as np

            from mast.vision.spectroscopy import (
                assess_shockley_onset,
                broadening_floor_v,
            )

            v = np.linspace(-0.8, 0.3, 400)
            w = broadening_floor_v(4.2, 0.005) / 4.394
            step = 1.0 + 1.0 / (1.0 + np.exp(-(v + 0.49) / w))
            good = assess_shockley_onset(v, step, expected_onset_v=-0.49)
            noise = np.random.default_rng(0).normal(0.0, 1.0, v.size)
            bad = assess_shockley_onset(v, noise, expected_onset_v=-0.49)
            ok = bool(good.passed) and not bad.passed
            add("表面态判据自测", ok,
                "合成台阶通过、纯噪声被拒" if ok
                else f"异常：台阶 passed={good.passed}、噪声 passed={bad.passed}",
                blocking=True)
        except Exception as exc:  # noqa: BLE001
            add("表面态判据自测", None, f"跑不了（{exc}）")

        try:
            import math

            import numpy as np

            from mast.vision.atomic_phase import assess_atomic_phase

            n, nmpp2 = 128, 5.0 / 256
            yy, xx = np.mgrid[0:n, 0:n].astype(float)
            k = 2 * math.pi / 0.2494
            lat = sum(np.cos(k * (xx * nmpp2 * math.cos(math.radians(d))
                                  + yy * nmpp2 * math.sin(math.radians(d))))
                      for d in (0.0, 60.0, 120.0)) / 3.0 * 10e-12
            g = assess_atomic_phase(lat, nm_per_px=nmpp2)
            b = assess_atomic_phase(
                np.random.default_rng(0).normal(0, 2e-12, (n, n)),
                nm_per_px=nmpp2)
            ok = bool(g.passed) and not b.passed
            add("原子相判据自测", ok,
                "合成晶格通过、纯噪声被拒" if ok
                else f"异常：晶格 passed={g.passed}（{g.reasons}）、"
                     f"噪声 passed={b.passed}",
                blocking=True)
        except Exception as exc:  # noqa: BLE001
            add("原子相判据自测", None, f"跑不了（{exc}）")

        # ── 5. 扎针深度在针尖包络内 ────────────────────────────────────
        try:
            from mast.core.special_tip_workflow import SPECTROSCOPY_TIP
            from mast.skills.builtins._tip_policy import apply_tip_policy

            depth_m = -abs(SPECTROSCOPY_TIP.poke_depth_nm) * 1e-9
            _, plan = apply_tip_policy({"poke_deep_depth_m": depth_m},
                                       ("poke_deep_depth_m",))
            allowed = plan is None or bool(getattr(plan, "ok", True))
            add(f"{SPECTROSCOPY_TIP.poke_depth_nm:g} nm 浅扎在包络内", allowed,
                "允许" if allowed
                else "；".join(getattr(plan, "refusals", []) or ["被包络拒绝"]),
                blocking=True)
        except Exception as exc:  # noqa: BLE001
            add("扎针包络", None, f"查不了（{exc}）")

        # ── 6. 偏压扰动会不会被电流监控读成异常 ────────────────────────
        # 扰动期间电流本来就会跳。监控那半边靠技能名子串豁免,名字改了就会每一发
        # 都被读成 CRITICAL 并把配方自己 halt 掉。
        try:
            from mast.core.tip_intent import is_tip_work
            from mast.monitoring.service import SUPPRESS_SKILL_PATTERNS

            names = ("MakeAtomicResolutionTip", "MakeSpectroscopyTip", "BiasWiggle")
            uncovered = [n for n in names
                         if not any(p in n.lower() for p in SUPPRESS_SKILL_PATTERNS)]
            add("电流监控已豁免锻造流程", not uncovered,
                "三个技能名都被豁免表覆盖" if not uncovered
                else f"{uncovered} 不在 SUPPRESS_SKILL_PATTERNS 里 —— "
                     f"扰动会被读成电流异常并中止配方")
            # ⑰-C1(2026-08-09):后果那句原文是「扎完针扫图会触发 CRITICAL 并把配方
            # 自己中止」—— 视觉 halt 整条割掉之后那已经不会发生了,留着就是给用户
            # 一句假话。这张表**没有作废**,它换了消费者:现在管的是「修针期间电流的
            # **瞬变**类不中止流程」(⑰-C2)和通知措辞。后果因此改成真实的那个。
            unmarked = [n for n in names if not is_tip_work(n)]
            add("修针令牌已覆盖锻造流程", not unmarked,
                "三个技能名都在 TIP_WORK_PATTERNS 里" if not unmarked
                else f"{unmarked} 不在 tip_intent.TIP_WORK_PATTERNS 里 —— "
                     f"脉冲/扎针打出的电流瞬变不会被认成本职动作，"
                     f"会当成物理越界把配方自己中止")
        except Exception as exc:  # noqa: BLE001
            add("监控豁免", None, f"查不了（{exc}）")

        # ── 7. 地图标记会不会记成损伤 ──────────────────────────────────
        try:
            from mast.io.exp_map import _SKILL_KIND_RULES

            def _kind(name: str) -> str:
                low = name.lower()
                for needles, kind in _SKILL_KIND_RULES:
                    if any(nd in low for nd in needles):
                        return kind
                return ""

            wrong = {n: _kind(n) for n in
                     ("MakeSpectroscopyTip", "MakeAtomicResolutionTip")
                     if _kind(n) != "tip_shape"}
            add("扎针落点记为损伤", not wrong,
                "两个配方都归入 tip_shape（会生成避让圆）" if not wrong
                else f"归类错误 {wrong} —— 扎出的坑不会产生避让圆，"
                     f"下一次 FindCleanSpot 会把它当干净表面")
        except Exception as exc:  # noqa: BLE001
            add("地图标记归类", None, f"查不了（{exc}）")

        # ── 8. Tip Shaper 模块(唯一一次硬件读) ─────────────────────────
        rec = context.safe_call("TipShaper_PropsGet")
        calls.append(rec)
        shaper_ok = not getattr(rec, "error", "")
        add("Tip Shaper 模块", shaper_ok,
            "在跑" if shaper_ok
            else f"读不到: {getattr(rec, 'error', '')}（Nanonis 里把 Tip Shaper 打开）",
            blocking=True)

        # ── 9. 运行模式 ────────────────────────────────────────────────
        try:
            from mast.core.operating_mode import current_operating_mode
            from mast.core.types import OperatingMode

            mode = current_operating_mode()
            if mode == OperatingMode.SAFE:
                add("操作模式", False,
                    "SAFE —— 改针尖的操作被整类硬拦，两条流程都跑不了",
                    blocking=True)
            elif mode == OperatingMode.SEMI:
                add("操作模式", True,
                    "SEMI —— 偏压扰动会按电学修针逐次等人确认")
            else:
                add("操作模式", True, "AUTO —— 一次批准跑完整条流程")
        except Exception:  # noqa: BLE001
            pass          # 读不到模式不是问题，门控自己会说话

        ready = not blockers
        lines = [("✅ 可以开工" if ready else "❌ 还不能开工")]
        for c in checks:
            mark = {True: "✅", False: "❌", None: "⚠"}[c["ok"]]
            lines.append(f"{mark} {c['check']}：{c['detail']}")
        return SkillResult(
            skill_name="TipForgeSelfCheck",
            success=True,          # 体检本身成功了；结论在 data.ready 里
            data={"ready": ready, "checks": checks, "blockers": blockers,
                  "warnings": warnings, "missing_skills": missing,
                  "substrate": substrate_name},
            nanonis_calls=calls,
            summary="\n".join(lines),
        )


def make_tool(context_provider):
    from mast.agents._shared.skill_adapter import wrap_skill

    return wrap_skill(TipForgeSelfCheck, context_provider)


__all__ = ["TipForgeSelfCheck", "REQUIRED_SKILLS"]
