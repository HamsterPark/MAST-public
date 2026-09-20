"""ScanIntelSelfCheck —— 扫图智能脚本层的真机自检(只读)。

设计文档:``docs/v2/design/scan_intelligence_scripted_rfc.md``
验收手册:``docs/v2/design/scan_intelligence_commissioning.md``

打包上真机之后,有几件事在开发机上**无法得知**,而它们决定了一整层的默认值对不对:

  * 7 个新技能在**冻结版**里是不是真的都在(walk_packages 在冻结环境里枚举不到
    任何东西,靠的是包 __init__ 的 eager import —— 少一个就是整条链断掉,而且
    不报错);
  * 档位表加载的是用户的还是出厂的;
  * ``instrument_profile`` 里那几个仪器级常数(Z 量程 / 针尖速度上限 / 倾斜上限)
    是不是还是出厂猜测;
  * **倾斜响应矩阵有没有标定** —— 没有它 AutoTilt 一律跳过;
  * 硬件当前的分辨率、倾斜、扫描框各是多少。

这个技能把上面这些收成**一次调用**,免得靠人一条条戳。它是只读的,不动针尖、
不改任何设置。

唯一的例外是 ``probe_buffer_semantics``(默认关):它回答「``Scan_BufferSet(ch,0,0)``
到底是保持还是重置分辨率」这个尚未证实的问题。打开时发出的那一条调用,与
``ConfigureScan`` 每次执行时**本来就会发**的完全相同 —— 所以它不引入新的风险,
只是把一个一直在发生、却从没人回读过的动作显式地测一次。
"""

from __future__ import annotations

import logging
from typing import Any

from mast.core.types import (
    ParameterSpec,
    SafetyLevel,
    SkillCategory,
    SkillMetadata,
    SkillResult,
)
from mast.io.nanonis_files import channel_ids_from_buffer, scalar_int
from mast.skills.base import BaseSkill

logger = logging.getLogger(__name__)

#: 这一层依赖的全部技能。少一个就有一条链是断的 —— 而且断得很安静。
REQUIRED_SKILLS: tuple[str, ...] = (
    "ScanAt", "SetScanBuffer", "TiltProbeCircle", "TiltCalibrate",
    "AutoTilt", "BiasSettleChange", "ExecuteScanPlan",
    # 被上面这些当作子步骤调用的既有技能
    "ConfigureScan", "SetScanSpeed", "StartScan", "WaitScanComplete",
    "SetBias", "SetBiasRamp", "SetSetpoint", "SetZCtrlGain",
    "GetScanBuffer", "SetPiezoTilt", "GetPiezoTilt", "SaveScan",
)

#: 这些仪器级常数如果还是出厂默认,说明没人按本机填过。
_PROFILE_KEYS: tuple[tuple[str, Any, str], ...] = (
    ("z_range_m", 1.5e-6, "Z 压电总量程 —— 调平触发判据的分母"),
    ("v_tip_max_m_s", 2e-6, "针尖横向速度上限 —— resolver 的组合约束"),
    ("tilt_limit_deg", 5.0, "压电倾斜绝对上限"),
    ("z_noise_floor_m", None, "Z 噪声底(留空 = 每次从数据估)"),
)


def _vals(record) -> "list | None":
    parsed = getattr(record, "return_value", None)
    if not isinstance(parsed, (list, tuple)) or len(parsed) <= 2:
        return None
    vals = parsed[2]
    return list(vals) if isinstance(vals, (list, tuple)) else None


class ScanIntelSelfCheck(BaseSkill):
    """Read-only self-check of the scripted scan-intelligence layer."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="ScanIntelSelfCheck",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description=(
                "对脚本化扫描层做一次只读自检：它的哪些技能在**这一次构建**里真的注册上了、逐尺度策略表用的是用户那份还是出厂那份、"
                "机器常数与压电倾斜标定填没填、以及仪器当前报出来的分辨率／倾斜／扫描帧是什么。部署到一台新机器之后先跑它 —— 它一次调用就回答了本来要做十几项手工检查才问得清的事。"
                "它什么都不移动，也什么都不改。"
            ),
            parameters=[
                ParameterSpec(
                    name="probe_buffer_semantics",
                    type="bool",
                    description=(
                        "顺带判定 Scan_BufferSet(channels, 0, 0) 到底是**保留**还是**重置**分辨率 —— 整个这一层都依赖着这个假设，"
                        "而它从来没有被验证过。它发出的那次调用，与 ConfigureScan 每一次调用都会发的那一次逐字节相同，"
                        "所以不增加任何新风险；它只是把结果读回来。事后会把原来的分辨率恢复。默认**关闭**。"
                    ),
                    required=False,
                    default=False,
                ),
            ],
            estimated_duration_s=3.0,
            composition_level=2,
            tags=["scan", "diagnostics", "commissioning", "read"],
        )

    # ── 各分项 ───────────────────────────────────────────────────────────────

    @staticmethod
    def _check_registry(registry=None) -> dict[str, Any]:
        """读取实际注册的技能名称，并在冻结构建中验证技能完整性。
        
        使用 core.registry.registered_skill_names 的统一实现；有活动 registry 时优先
        查询它，因为 ctx.run 也使用该表，否则才建立新 registry 并 discover。
        开发环境的包遍历与冻结构建的 eager import 不同，需覆盖实际打包入口。"""
        try:
            from mast.core.registry import registered_skill_names
            installed = registered_skill_names(registry)
            missing = [n for n in REQUIRED_SKILLS if n not in installed]
            return {
                "total_registered": len(installed),
                "missing": missing,
                "ok": not missing,
                "note": ("" if not missing else
                         "这些技能在本版里不存在 —— 多半是打包时包 __init__ 没有 "
                         "import 到它们所在的模块(冻结环境不做 walk_packages)"),
            }
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}

    @staticmethod
    def _check_policy() -> dict[str, Any]:
        try:
            from mast.core import scan_policy, scan_resolver
            tiers = scan_policy.get_policy()
            preview = {}
            for nm in (5.0, 50.0, 200.0, 1000.0):
                try:
                    p = scan_resolver.preview(nm * 1e-9)
                    preview[f"{nm:g}nm"] = {
                        "tier": p["tier_name"], "pixels": p["pixels"],
                        "line_time_s": p["line_time_s"],
                        "est_min": round(p["estimated_scan_s"] / 60.0, 1),
                    }
                except Exception as exc:  # noqa: BLE001
                    preview[f"{nm:g}nm"] = {"error": str(exc)}
            return {
                "customised": scan_policy.is_customised(),
                "n_tiers": len(tiers),
                "tiers": [{"name": t.get("name"),
                           "upper_size_nm": (None if t.get("upper_size_m") is None
                                             else t["upper_size_m"] * 1e9),
                           "pixels": t.get("pixels"),
                           "line_time_s": t.get("line_time_s"),
                           "source": t.get("source")} for t in tiers],
                "preview": preview,
                "note": ("" if scan_policy.is_customised() else
                         "当前用的是**出厂参考值**(取自渐进缩放协议的四级建议)。"
                         "这些数字随仪器和样品变 —— 请按本机实际情况校准。"),
            }
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}

    @staticmethod
    def _check_profile() -> dict[str, Any]:
        try:
            from mast.core import instrument_profile as ip
            out: dict[str, Any] = {"values": {}, "still_factory": []}
            for key, factory, why in _PROFILE_KEYS:
                val = ip.get_config(key, None)
                out["values"][key] = {"value": val, "why": why}
                if val == factory:
                    out["still_factory"].append(key)
            calib = ip.get_tilt_calibration()
            out["tilt_calibration"] = calib
            out["tilt_calibrated"] = calib is not None
            if calib is None:
                out["tilt_note"] = (
                    "**没有倾斜响应标定 → AutoTilt 一律跳过。** Piezo_TiltSet 的"
                    "轴对应与符号取决于本机接线,猜错方向会把倾斜往反方向加倍。"
                    "在一块平坦区上跑一次 TiltCalibrate。")
            return out
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}

    def _check_hardware(self, context, calls) -> dict[str, Any]:
        out: dict[str, Any] = {}

        rec = context.safe_call("Scan_BufferGet")
        calls.append(rec)
        v = _vals(rec) if not rec.error else None
        if v and len(v) >= 4:
            out["scan_buffer"] = {"pixels": int(v[2]), "lines": int(v[3]),
                                  "n_channels": int(v[0])}
        else:
            out["scan_buffer"] = {"error": rec.error or "回包无法解析"}

        rec = context.safe_call("Scan_FrameGet")
        calls.append(rec)
        v = _vals(rec) if not rec.error else None
        if v and len(v) >= 5:
            out["scan_frame"] = {
                "center_x_m": float(v[0]), "center_y_m": float(v[1]),
                "width_m": float(v[2]), "height_m": float(v[3]),
                "angle_deg": float(v[4])}
        else:
            out["scan_frame"] = {"error": rec.error or "回包无法解析"}

        rec = context.safe_call("Piezo_TiltGet")
        calls.append(rec)
        v = _vals(rec) if not rec.error else None
        if v and len(v) >= 2:
            out["piezo_tilt"] = {"tilt_x_deg": float(v[0]),
                                 "tilt_y_deg": float(v[1])}
        else:
            out["piezo_tilt"] = {"error": rec.error or "回包无法解析"}

        return out

    def _probe_buffer_semantics(self, context, calls) -> dict[str, Any]:
        """回答:``Scan_BufferSet(ch, 0, 0)`` 是保持还是重置分辨率?

        整层设计都建立在「0/0 = 保持」这个假设上(ConfigureScan 每次都这么发),
        但从来没人回读过。这里读→发 0/0→回读→复原。
        """
        rec = context.safe_call("Scan_BufferGet")
        calls.append(rec)
        before = _vals(rec) if not rec.error else None
        if not before or len(before) < 4:
            return {"ok": False,
                    "error": "读不到当前缓冲配置,无法探测(未发出任何写入)"}

        # 还原扫描缓冲设置前必须得到整数通道列表；统一解析单元素元组，禁止把原始元组写回仪器。
        channels = channel_ids_from_buffer(before)
        if not channels:
            return {"ok": False,
                    "error": "读不到当前采集通道,无法原样写回(未发出任何写入)"}
        px0, ln0 = scalar_int(before[2]), scalar_int(before[3])
        if px0 is None or ln0 is None:
            return {"ok": False,
                    "error": "读不到当前分辨率,探测会无法复原(未发出任何写入)"}

        rec = context.safe_call("Scan_BufferSet", channels, 0, 0)
        calls.append(rec)
        if rec.error:
            return {"ok": False, "error": f"Scan_BufferSet(0,0) 被拒: {rec.error}"}

        rec = context.safe_call("Scan_BufferGet")
        calls.append(rec)
        after = _vals(rec) if not rec.error else None
        if not after or len(after) < 4:
            return {"ok": False, "error": "回读失败 —— 分辨率状态未知,请手工确认"}
        px1, ln1 = scalar_int(after[2]), scalar_int(after[3])
        if px1 is None or ln1 is None:
            return {"ok": False, "error": "回读失败 —— 分辨率状态未知,请手工确认"}

        kept = (px1 == px0 and ln1 == ln0)
        restored = True
        if not kept:
            # 语义是「重置」→ 把用户原来的分辨率还回去
            rec = context.safe_call("Scan_BufferSet", channels, px0, ln0)
            calls.append(rec)
            restored = not rec.error

        return {
            "ok": True,
            "before": [px0, ln0],
            "after_zero_zero": [px1, ln1],
            "semantics": "keep" if kept else "reset",
            "restored": restored,
            "note": ("0/0 = 保持现值 —— 与设计假设一致,ConfigureScan 的既有行为安全。"
                     if kept else
                     "**0/0 会重置分辨率** —— 与设计假设相反。ScanAt 把 SetScanBuffer "
                     "排在 ConfigureScan 之后,所以 ScanAt 路径仍然正确;但**直接调用 "
                     "ConfigureScan 会清掉分辨率设置**,这一点必须写进文档并复查所有"
                     "直接调用点。"),
        }

    # ── 执行 ─────────────────────────────────────────────────────────────────

    def execute(self, context, params: dict) -> SkillResult:
        calls: list = []
        data: dict[str, Any] = {
            "registry": self._check_registry(getattr(context, "_registry", None)),
            "scan_policy": self._check_policy(),
            "instrument_profile": self._check_profile(),
        }
        try:
            data["hardware"] = self._check_hardware(context, calls)
        except Exception as exc:  # noqa: BLE001
            data["hardware"] = {"error": str(exc)}

        if params.get("probe_buffer_semantics"):
            try:
                data["buffer_semantics"] = self._probe_buffer_semantics(
                    context, calls)
            except Exception as exc:  # noqa: BLE001
                data["buffer_semantics"] = {"ok": False, "error": str(exc)}

        # 待办清单:把「还没做的事」显式列出来,而不是让人从上面一堆字段里推。
        todo: list[str] = []
        reg = data["registry"]
        if not reg.get("ok"):
            todo.append(f"技能缺失:{reg.get('missing')} —— 打包注册有问题,先修这个")
        prof = data["instrument_profile"]
        if not prof.get("tilt_calibrated"):
            todo.append("在平坦区跑一次 TiltCalibrate(否则 AutoTilt 一律跳过)")
        if prof.get("still_factory"):
            todo.append(f"这些仪器常数还是出厂猜测,按本机填:{prof['still_factory']}")
        if not data["scan_policy"].get("customised"):
            todo.append("档位表还是出厂参考值,按本机习惯校准扫描速度与分辨率")
        if "buffer_semantics" not in data:
            todo.append("用 probe_buffer_semantics=true 再跑一次,确认 "
                        "Scan_BufferSet(0,0) 的语义")

        data["todo"] = todo
        data["ready"] = not todo

        summary = (f"技能 {reg.get('total_registered', '?')} 个"
                   f"({'齐' if reg.get('ok') else '缺 ' + str(len(reg.get('missing') or []))})"
                   f";档位表{'已定制' if data['scan_policy'].get('customised') else '=出厂'}"
                   f";倾斜标定{'有' if prof.get('tilt_calibrated') else '**无**'}"
                   f";待办 {len(todo)} 项")

        return SkillResult(skill_name="ScanIntelSelfCheck", success=True,
                           data=data, summary=summary, nanonis_calls=calls)
