"""修针流程开工前只读自检，确认当前环境具备所需依赖。

检查冻结构建的技能注册、所需仪器模块、针尖档案与活动实验状态，
以便在动作前发现缺失条件。注册依赖实际 eager import，不能仅依据 __all__。"""
from __future__ import annotations

import logging
from typing import Any

from mast.core.registry import registered_skill_names
from mast.core.types import (
    SafetyLevel,
    SkillCategory,
    SkillMetadata,
    SkillResult,
)
from mast.skills.base import BaseSkill

logger = logging.getLogger(__name__)

#: 修针流程依赖的全部技能。少一个就有一条链是断的 —— 而且断得很安静。
REQUIRED_SKILLS: tuple[str, ...] = (
    # 本层新增
    "PrepareNobleTip", "PulseConditionTip", "PokeConditionTip",
    "BiasPulseWithReadback", "FindCleanSpot", "AssessTipSharpness",
    # 被它们当作子步骤调用的既有技能
    # ``GetBias``:扎针前要把当前偏压读下来存着,好在扎完放回去(qPlus 上扎针要先
    # 降到 20 mV —— 用户范本 2026-08-10 C2)。读不到就不改偏压,所以这一条是
    # 那条物理修正**能不能生效**的前提,不是可有可无的诊断。
    # ``CaptureSignalBuffer``:扎针前后采振幅,用来发现音叉起跳(用户
    # 2026-08-10「怎么发现起跳?」)。采不到只是没有诊断,不挡扎针。
    "TipShapeWithReadback", "MoveToXY", "GetBias", "SetBias", "SetSetpoint",
    "CaptureSignalBuffer",
    "ZControllerOnOff", "ScanAt", "SaveScan", "GetLatestScanFile",
    "PreScanCheck", "FindFlatRegion", "AutoTilt", "AnalyzeFrameTilt",
    "AssessClusterRoundness",
)


class TipConditioningSelfCheck(BaseSkill):
    """开工前的只读体检:技能在不在、模块开没开、针尖登没登记、地图读不读得到。"""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="TipConditioningSelfCheck",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description=(
                "针尖整形工作流的只读预检：它要用的技能在这个 build 里是不是真的都注册了、Tip Shaper "
                "模块是不是在跑、针尖有没有登记（未登记的针尖会把脉冲压到保守默认值）、"
                "有没有实验可供读取扫描地图、以及 Z 噪声本底是不是小到让 50 pm 的台阶判据还有意义。"
                "什么都不碰。"
            ),
            parameters=[],
            estimated_duration_s=3.0,
            composition_level=1,
            tags=["tip", "conditioning", "selfcheck", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        from mast.core.noble_tip_workflow import NOBLE_METAL_BASELINE as wf

        calls: list = []
        checks: list[dict[str, Any]] = []
        blockers: list[str] = []
        warnings: list[str] = []

        def add(name: str, ok: bool | None, detail: str, *, blocking=False):
            checks.append({"check": name, "ok": ok, "detail": detail})
            if ok is False:
                (blockers if blocking else warnings).append(f"{name}: {detail}")
            elif ok is None:
                warnings.append(f"{name}: {detail}")

        # 从 core.registry 的统一实现取得技能名称，查询活动注册表，避免用元数据对象的字符串表示误判缺失。
        registry = getattr(context, "_registry", None)
        installed = registered_skill_names(registry)
        missing = [n for n in REQUIRED_SKILLS if n not in installed]
        add("技能齐全", not missing,
            f"全部就位（本机注册 {len(installed)} 个）" if not missing
            else f"缺 {len(missing)} 个: {', '.join(missing)}",
            blocking=True)

        # ── 2. Tip Shaper 模块 ─────────────────────────────────────────
        rec = context.safe_call("TipShaper_PropsGet")
        calls.append(rec)
        shaper_ok = not getattr(rec, "error", "")
        add("Tip Shaper 模块", shaper_ok,
            "在跑" if shaper_ok else f"读不到: {getattr(rec, 'error', '')}"
                                    "（Nanonis 里把 Tip Shaper 模块打开）",
            blocking=True)

        # ── 3. 针尖登记与脉冲包络 ──────────────────────────────────────
        tip_name = ""
        try:
            from mast.core.tip_conditioning_resolver import resolve_conditioning
            from mast.core.tip_state import current_tip_facts
            facts = current_tip_facts()
            tip_name = (facts or {}).get("name") or ""
            plan = resolve_conditioning(("pulse_v",), {"pulse_v": wf.pulse_v})
            allowed = bool(getattr(plan, "ok", False))
            if facts is None:
                add("针尖已登记", False,
                    f"未登记 —— 安全包络退到保守通用档，流程默认的 "
                    f"{wf.pulse_v:g} V 大修脉冲会被拒。先 register_tip。",
                    blocking=True)
            else:
                # a qPlus sensor's registered constants are facts no Nanonis readback
                # carries; a force inversion needs k and this line is where an operator
                # (or an agent) can read what was registered (2026-09-13, STM-Bench P5)
                extras = []
                for key, label, unit in (("qplus_k_n_per_m", "k", " N/m"), ("qplus_q", "Q", ""),
                                         ("qplus_f0_hz", "f0", " Hz")):
                    val = facts.get(key)
                    if isinstance(val, (int, float)) and not isinstance(val, bool):
                        extras.append(f"{label}={val:g}{unit}")
                add("针尖已登记", True,
                    f"{tip_name or '未命名'}（{facts.get('material') or '?'}/"
                    f"{facts.get('form') or '?'}）"
                    + (f"，登记常数 {'、'.join(extras)}" if extras else ""))
                add(f"{wf.pulse_v:g} V 脉冲在包络内", allowed,
                    "允许" if allowed
                    else "；".join(getattr(plan, "refusals", []) or ["被包络拒绝"]),
                    blocking=True)
        except Exception as exc:  # noqa: BLE001
            add("针尖包络", None, f"查不了（{exc}）")

        # ── 4. 扫描地图 ────────────────────────────────────────────────
        try:
            from mast.core.map_scope import analysis_config, load_markers
            markers, epoch, known = load_markers()
            cfg = analysis_config(getattr(context, "state", None))
            add("扫描地图可读", known,
                f"代次 {epoch}，{len(markers)} 个标记" if known
                else "读不到（没有活动实验？）—— 换点时无法确认落点是否干净",
                blocking=True)
            add("避让半径", True,
                f"脉冲 {cfg.pulse_r_m * 1e9:.0f} nm / "
                f"扎针 {cfg.tip_shape_r_m * 1e9:.0f} nm")
        except Exception as exc:  # noqa: BLE001
            add("扫描地图可读", None, f"查不了（{exc}）")

        # ── 5. Z 噪声底 vs 临界步进 ────────────────────────────────────
        # 50 pm 一级的临界浅扎，落在噪声里就是随机游走。
        try:
            from mast.core import instrument_profile as ip
            floor = ip.get_config("z_noise_floor_m", None)
            step_m = wf.critical_step_pm * 1e-12
            if floor is None:
                add("Z 噪声底", None,
                    f"未标定（每次从数据现估）。临界浅扎每级 "
                    f"{wf.critical_step_pm:.0f} pm，真机上先量一下噪声底再跑。")
            else:
                ratio = step_m / float(floor) if float(floor) > 0 else float("inf")
                add("临界步进可分辨", ratio >= 3.0,
                    f"步进 {wf.critical_step_pm:.0f} pm / 噪声底 "
                    f"{float(floor) * 1e12:.0f} pm = {ratio:.1f}×"
                    + ("" if ratio >= 3.0 else " —— 太接近噪声，临界搜索会变成随机游走"))
        except Exception as exc:  # noqa: BLE001
            add("Z 噪声底", None, f"查不了（{exc}）")

        # ── 6. 运行模式 ────────────────────────────────────────────────
        try:
            from mast.core.operating_mode import current_operating_mode
            from mast.core.types import OperatingMode
            mode = current_operating_mode()
            if mode == OperatingMode.SAFE:
                add("操作模式", False,
                    "SAFE —— 修针类操作被整类硬拦，流程跑不了", blocking=True)
            elif mode == OperatingMode.SEMI:
                add("操作模式", True,
                    "SEMI —— 每一发电脉冲都会单独等人确认（这是该模式的本意）")
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
            skill_name="TipConditioningSelfCheck",
            success=True,          # 体检本身成功了；结论在 data.ready 里
            data={"ready": ready, "checks": checks, "blockers": blockers,
                  "warnings": warnings, "missing_skills": missing,
                  "tip_name": tip_name},
            nanonis_calls=calls,
            summary="\n".join(lines),
        )


def make_tool(context_provider):
    from mast.agents._shared.skill_adapter import wrap_skill
    return wrap_skill(TipConditioningSelfCheck, context_provider)
