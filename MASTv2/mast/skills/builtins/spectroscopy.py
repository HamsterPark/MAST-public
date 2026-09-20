"""Spectroscopy skills: STS (Bias Spectroscopy) and Z Spectroscopy.

vendored from v1 mast/skills/builtins/spectroscopy.py 2026-04-23. Zero behavioural changes.
38 skills: AcquireSTS, ConfigureSTS, ConfigureZSpectr, AcquireZSpectr,
           ConfigureSTSTiming, StopSTS, StopZSpectr, ConfigureSTSChannels,
           ConfigureZSpectrTiming, GetSTSChannels, SetSTSChannels, GetSTSLimits,
           SetSTSAdvancedProps, GetSTSTiming, GetSTSAltZCtrl,
           GetZSpectrChannels, SetZSpectrChannels, GetZSpectrRange, SetZSpectrRange,
           GetZSpectrRetract, SetZSpectrRetract,
           GetSTSDigSync, GetSTSTTLSync, GetSTSPulseSeqSync, GetSTSZOffRevert,
           GetSTSMLSLockinPerSeg, SetSTSMLSMode, SetSTSMLSVals,
           SetSTSSafeCond1, GetSTSSafeCond1, SetSTSSafeCond2,
           SetZSpectrAdvProps, GetZSpectrDigSync, GetZSpectrPulseSeqSync,
           GetZSpectrRetract2nd, SetZSpectrRetractDelay, GetZSpectrTTLSync,
           GetZSpectrTiming.
"""

from __future__ import annotations

import logging

from mast.core.types import (
    ParameterSpec,
    SafetyLevel,
    SkillCategory,
    SkillMetadata,
    SkillResult,
)
from mast.io.nanonis_files import channel_ids_from_buffer
from mast.skills.base import BaseSkill


def _reshape_spectrum(variables):
    """将 BiasSpectr/ZSpectr.Start 的 Variables 解析成通道到完整谱线的映射。
    
    二维布局为行=通道、列=扫掠点；转置会混合不同通道。
    返回 (channel_map, num_points, reason)，成功时 reason 为空。
    解码失败必须带原因，与一条有效但零点数的谱明确区分，不能静默缺少字段。"""
    if not (isinstance(variables, (list, tuple)) and len(variables) >= 6):
        n = len(variables) if isinstance(variables, (list, tuple)) else "非序列"
        return {}, 0, f"Variables 不足 6 项(实际 {n}),不是一个谱数据块"
    ch_names = variables[2] if isinstance(variables[2], list) else []
    data_2d = variables[5]
    try:
        rows = int(variables[3])  # rows = channels
        cols = int(variables[4])  # cols = sweep points
    except (TypeError, ValueError):
        return {}, 0, (f"行列数不是整数(rows={variables[3]!r}, "
                       f"cols={variables[4]!r})")
    if hasattr(data_2d, "tolist"):
        data_2d = data_2d.tolist()
    if not (isinstance(data_2d, list) and rows > 0 and cols > 0):
        return {}, 0, (f"数据块不是列表或行列数非正(rows={rows}, cols={cols}, "
                       f"data={type(data_2d).__name__})")
    import numpy as np
    try:
        arr = np.array(data_2d, dtype=np.float64).reshape(rows, cols)
    except (ValueError, TypeError) as exc:
        return {}, 0, f"{rows}×{cols} 装不下这段数据: {exc}"
    channel_map: dict = {}
    for i, name in enumerate(ch_names[:rows]):
        channel_map[name] = arr[i, :].tolist()
    if not channel_map:
        return {}, 0, f"数据解开了({rows}×{cols}),但一个通道名都没有"
    return channel_map, cols, ""


def _match_channel(channel_map: dict, needles: tuple[str, ...]):
    """First channel whose (lower-cased) name contains any needle, else None."""
    for name, trace in channel_map.items():
        low = str(name).lower()
        if any(n in low for n in needles):
            return trace
    return None


def _coerce_int_list(value) -> list[int]:
    """Coerce a list-valued parameter to ``list[int]``.

    The agent tool-schema builder (``agents/_shared/skill_adapter._TYPE_MAP``)
    only understands the scalar ParameterSpec types int/float/str/bool, so a
    ``list[int]`` spec is silently presented to the LLM as a *string*. Declaring
    these params as ``str`` (comma-separated) and coercing here makes the skill
    work over BOTH paths: agent (arrives as "0, 1, 2") and direct/programmatic
    (arrives as a real ``[0, 1, 2]`` list).
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [int(float(x.strip())) for x in value.replace(";", ",").split(",") if x.strip()]
    if isinstance(value, (list, tuple)):
        return [int(x) for x in value]
    return [int(value)]


def _coerce_float_list(value) -> list[float]:
    """Coerce a list-valued parameter to ``list[float]`` (see _coerce_int_list)."""
    if value is None:
        return []
    if isinstance(value, str):
        return [float(x.strip()) for x in value.replace(";", ",").split(",") if x.strip()]
    if isinstance(value, (list, tuple)):
        return [float(x) for x in value]
    return [float(value)]


logger = logging.getLogger(__name__)


def _attach_saved_dat(context, data: dict) -> None:
    """Put the .dat Nanonis just autosaved into ``data["path"]``.

    WHY: ConfigureSTS turns autosave on, so every
    acquisition DOES land a .dat on disk — but the skill never reported where,
    so ``data`` carried no ``path``. The tool adapter only harvests
    ``path``/``file_path``/``sxm_path`` into ``scan_paths``, so the handoff to
    data_processing had no real file to point at. All it could offer was the
    tool-return sidecar (a summary-overflow dump), which data_processing then
    fed to ``load_scan`` — and that crashed.

    Nanonis picks the directory and the index itself, so the only way to name
    the file is to look for the newest .dat right after the sweep. Best-effort
    by design: no .dat (autosave off) simply means no ``path`` — never a guess,
    because a wrong path here points the analysis agent at someone else's data.
    """
    try:
        from mast.skills.builtins.scan_extra import (
            _candidate_save_dirs,
            find_latest_saved,
        )

        latest = find_latest_saved(_candidate_save_dirs(context), "*.dat",
                                   max_age_s=120)
        if latest is None:
            return
        data["path"] = str(latest)
        try:
            from mast.core.scan_registry import record_scan_path
            record_scan_path(latest)
        except Exception:  # noqa: BLE001 — registry is best-effort
            pass
    except Exception as exc:  # noqa: BLE001 — never fail an acquisition over this
        logger.debug("AcquireSTS: could not resolve autosaved .dat: %s", exc)

# 扫掠接收预算包含采集比例余量和存盘等固定开销。
# 预算不足会导致迟到回包破坏后续流同步；预算更长则延后故障返回。
_SWEEP_BUDGET_FACTOR = 1.35
_SWEEP_BUDGET_PAD_S = 45.0
#: 读不到设定时的回退。**「读不到」不是「很快」**——退回一个小数字正是
#: 这个 bug 的形状，所以这里退到上限那一侧。
_SWEEP_BUDGET_FALLBACK_S = 600.0


def _sweep_duration_s(context, calls: list) -> tuple[float, dict]:
    """从仪器**自己的设定**算一条谱要跑多久（秒），外加算给谁看的明细。

    不写死数字：用户随时会改点数 / 积分时间 / sweep 数，写死的预算下一次
    就又不够。这两次读是 ~50 ms 的代价，换的是这个技能对任何配置都成立。

        时长 ≈ 点数 × (settling + integration) × (往返 ? 2 : 1) × sweep 数
               + sweep 数 × (initial_settling + end_settling + z_control + z_avg)
    """
    npts = nsw = None
    bwd = True
    rec = context.safe_call("BiasSpectr_PropsGet")
    calls.append(rec)
    v = rec.return_value
    var = v[2] if isinstance(v, (list, tuple)) and len(v) > 2 else None
    # BiasSpectr.PropsGet Variables =
    #   [Save all, Number of sweeps, Backward sweep, Number of points, ...]
    if isinstance(var, (list, tuple)) and len(var) >= 4:
        try:
            nsw = max(int(var[1]), 1)
            bwd = bool(int(var[2]))
            npts = max(int(var[3]), 1)
        except (TypeError, ValueError):
            npts = nsw = None

    t: dict = {}
    rec2 = context.safe_call("BiasSpectr_TimingGet")
    calls.append(rec2)
    v2 = rec2.return_value
    var2 = v2[2] if isinstance(v2, (list, tuple)) and len(v2) > 2 else None
    keys = ["z_averaging_time_s", "z_offset_m", "initial_settling_time_s",
            "max_slew_rate_v_per_s", "settling_time_s", "integration_time_s",
            "end_settling_time_s", "z_control_time_s"]
    if isinstance(var2, (list, tuple)):
        for i, k in enumerate(keys):
            if i < len(var2):
                try:
                    t[k] = float(var2[i])
                except (TypeError, ValueError):
                    pass

    if npts is None or nsw is None or "integration_time_s" not in t:
        return _SWEEP_BUDGET_FALLBACK_S, {
            "why": "读不到扫掠设定，退回保守上限",
            "npts": npts, "nsweeps": nsw, "timing_keys": sorted(t),
        }
    per_pt = t.get("settling_time_s", 0.0) + t["integration_time_s"]
    per_sweep = (t.get("initial_settling_time_s", 0.0)
                 + t.get("end_settling_time_s", 0.0)
                 + t.get("z_control_time_s", 0.0)
                 + t.get("z_averaging_time_s", 0.0))
    core = npts * per_pt * (2 if bwd else 1) * nsw + per_sweep * nsw
    return core, {"npts": npts, "nsweeps": nsw, "backward": bwd,
                  "per_point_s": round(per_pt, 4), "core_s": round(core, 1)}


class AcquireSTS(BaseSkill):
    """Acquire a single STS spectrum at the current position."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="AcquireSTS",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "在当前针尖位置采一条 STS 谱。使用当前的 lock-in 与 bias 扫描设置。传 save_basename "
                "可以控制保存下来的 .dat 文件名（为了可追溯 —— 例如按网格点命名）。"
            ),
            parameters=[
                ParameterSpec(
                    name="save_basename",
                    type="str",
                    description=("保存下来的谱文件的基名（留空 = 沿用模块当前的基名）。"
                                 "它让网格/批量测量能把每一个 .dat 与它的坐标对应起来。"),
                    required=False,
                    default="",
                ),
            ],
            preconditions=["z_controller_on"],
            estimated_duration_s=30.0,
            composition_level=1,
            tags=["spectroscopy", "sts", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        calls = []
        rec_open = context.safe_call("BiasSpectr_Open")
        calls.append(rec_open)

        # NOTE: AcquireSTS deliberately does NOT call BiasSpectr_PropsSet here.
        # PropsSet's Z-offset arg has no "no change" sentinel and BiasSpectr_
        # PropsGet does not even return it, so the old hardcoded
        # PropsSet(1,0,0,0,0.0,1,2) silently ZEROED the operator's configured
        # protective pre-STS Z retract on every acquire.
        # Save/autosave/Z-offset are configured via ConfigureSTS; Acquire only
        # starts the sweep and (Get data=1) returns the spectrum regardless.
        save_basename = str(params.get("save_basename", "") or "")

        # BiasSpectr_Start 阻塞直至整条扫掠结束，recv 预算必须覆盖实际采集时长。
        # 过早超时会使迟到回包污染下一次调用，也可能令控制器状态停在采集中间。
        # 根据仪器当前点数、积分时间和 sweep 数推导预算，不写死某次采集的时长。
        core_s, budget_detail = _sweep_duration_s(context, calls)
        recv_budget_s = core_s * _SWEEP_BUDGET_FACTOR + _SWEEP_BUDGET_PAD_S
        record = context.safe_call("BiasSpectr_Start", 1, save_basename,
                                   recv_timeout_s=recv_budget_s)
        calls.append(record)
        if record.error:
            return SkillResult(
                skill_name="AcquireSTS",
                success=False,
                error=record.error,
                nanonis_calls=calls,
            )
        # ``acquisition_complete`` 说的是**这次扫掠调用跑完了**(``record.error``
        # 上面已经挡过),不是「谱拿到了」。这两件事在这个技能里是分开的 ——
        # 见下面 success 的取法。
        data: dict = {"acquisition_complete": True}
        # 记账：这条谱按仪器设定算出来要多久、我们把 recv 超时抬到了多少。
        # 出问题时这是第一现场 —— 抬得不够会表现成「连接莫名其妙废了」。
        data["sweep_estimate_s"] = round(core_s, 1)
        data["recv_timeout_s"] = round(recv_budget_s, 1)
        data["sweep_settings"] = budget_detail
        if save_basename:
            data["save_basename"] = save_basename
        parsed = record.return_value
        if isinstance(parsed, (list, tuple)) and len(parsed) > 2:
            channel_map, num_points, why = _reshape_spectrum(parsed[2])
        else:
            channel_map, num_points, why = {}, 0, (
                f"return_value 不是 (err, raw, Variables) 三元组: "
                f"{type(parsed).__name__}")
        # **总是**落这个键。只在失败时出现的标志,会让按它分支的调用方在成功
        # 路径上读到 KeyError —— 那是把一个诚实标志变成一颗新的雷。
        data["spectrum_parsed"] = bool(channel_map)
        if channel_map:
            data["channel_names"] = list(channel_map.keys())
            data["num_points"] = num_points
            data.update(channel_map)   # one trace per channel name
            # Convenience aliases resolved BY CHANNEL NAME (not by a fixed
            # column index, which is what the transposed parse got wrong).
            volt = _match_channel(channel_map, ("bias", "volt"))
            if volt is not None:
                data["voltage"] = volt
            cur = _match_channel(channel_map, ("current",))
            if cur is not None:
                data["current"] = cur
        else:
            data["spectrum_unparsed_reason"] = why
            logger.warning("AcquireSTS: 返回的数据块解不开 —— %s", why)
        _attach_saved_dat(context, data)

        # ── success 的取法(2026-08-15 普查 A3;判据写在这里,别照搬别处)──
        # ConfigureSTS 打开 autosave ⇒ 每次采集都有一份 .dat 落盘。所以
        # 「内联块解不开」时数据**可能仍然在盘上**;那时报失败会把 agent 推去
        # 重扫同一个点 —— 多一次针尖停留、多一次针尖变化的机会,而好数据本来
        # 就在。反过来,内联解不开**且**盘上也没找到,这次采集就没有在任何地方
        # 留下东西,说成成功就是把一次故障答成了数据。
        #
        # ⚠️ 这条判断的弱处写在这儿,好让下一个人能推翻它:``_attach_saved_dat``
        # 找的是「最近 120 s 内最新的 .dat」,是个启发式,它**可能漏**(慢盘、
        # 目录不在候选表里)。所以 error 文本必须告诉人去盘上自己看一眼。
        saved = str(data.get("path") or "")
        if not channel_map and not saved:
            return SkillResult(
                skill_name="AcquireSTS",
                success=False,
                error=(f"扫掠跑完了,但返回的数据块解不开({why}),"
                       f"而且没有找到自动保存的 .dat —— 这次采集没有在任何地方"
                       f"留下可用的谱。若 autosave 是开着的,请到 Nanonis 的保存"
                       f"目录自己确认一次(这里只看最近 120 s 内最新的 .dat)。"),
                data=data,
                nanonis_calls=calls,
            )
        return SkillResult(
            skill_name="AcquireSTS",
            success=True,
            data=data,
            nanonis_calls=calls,
        )


class ConfigureSTS(BaseSkill):
    """Configure STS sweep parameters."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="ConfigureSTS",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description="配置 STS 的 bias 扫描参数。",
            parameters=[
                ParameterSpec(
                    name="start_v",
                    type="float",
                    description="扫描起始电压，单位伏特",
                    unit="V",
                    required=True,
                    min_value=-10.0,
                    max_value=10.0,
                ),
                ParameterSpec(
                    name="end_v",
                    type="float",
                    description="扫描终止电压，单位伏特",
                    unit="V",
                    required=True,
                    min_value=-10.0,
                    max_value=10.0,
                ),
                ParameterSpec(
                    name="num_points",
                    type="int",
                    description="扫描的点数",
                    required=True,
                    min_value=2,
                    max_value=10000,
                ),
                ParameterSpec(
                    name="z_offset_m",
                    type="float",
                    description="谱学期间的 Z 偏移，单位米",
                    unit="m",
                    required=False,
                    default=0.0,
                ),
            ],
            estimated_duration_s=1.0,
            composition_level=1,
            tags=["spectroscopy", "sts", "configure", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        start_v = params["start_v"]
        end_v = params["end_v"]
        num_points = params["num_points"]
        z_offset_m = params.get("z_offset_m", 0.0)
        calls = []

        rec_open = context.safe_call("BiasSpectr_Open")
        calls.append(rec_open)
        if rec_open.error:
            return SkillResult(
                skill_name="ConfigureSTS",
                success=False,
                error=rec_open.error,
                nanonis_calls=calls,
            )

        rec_limits = context.safe_call("BiasSpectr_LimitsSet", start_v, end_v)
        calls.append(rec_limits)
        if rec_limits.error:
            return SkillResult(
                skill_name="ConfigureSTS",
                success=False,
                error=rec_limits.error,
                nanonis_calls=calls,
            )

        rec_props = context.safe_call(
            "BiasSpectr_PropsSet", 1, 1, 1, num_points, z_offset_m, 1, 2,
        )
        calls.append(rec_props)
        if rec_props.error:
            return SkillResult(
                skill_name="ConfigureSTS",
                success=False,
                error=rec_props.error,
                nanonis_calls=calls,
            )

        # Advanced props: pin the two behaviours the STS
        # chain used to inherit from whatever the operator last left in the
        # Nanonis GUI — a dangerous ambiguity. AdvPropsSet(reset_bias,
        # z_ctrl_hold, record_final_z, lockin_run), each 0=no change/1=On/2=Off:
        #   • Z-Controller Hold = On → feedback is SUSPENDED during the bias
        #     sweep (constant-height STS). Without this, feedback stays active
        #     and as the sweep crosses 0 V it drives the tip into the surface.
        #   • Reset Bias = On → bias returns to the imaging value after the sweep
        #     (don't leave the junction parked at the sweep end voltage).
        rec_adv = context.safe_call("BiasSpectr_AdvPropsSet", 1, 1, 0, 0)
        calls.append(rec_adv)
        # Non-fatal: some controllers/firmware may reject AdvPropsSet; the sweep
        # can still run. Surface it in data rather than failing the config.
        adv_ok = not getattr(rec_adv, "error", "")

        return SkillResult(
            skill_name="ConfigureSTS",
            success=True,
            data={
                "start_v": start_v,
                "end_v": end_v,
                "num_points": num_points,
                "z_offset_m": z_offset_m,
                "z_controller_hold": True if adv_ok else None,
                "reset_bias": True if adv_ok else None,
            },
            nanonis_calls=calls,
        )


class ConfigureZSpectr(BaseSkill):
    """Configure Z spectroscopy parameters."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="ConfigureZSpectr",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description="配置 Z 谱学：Z 偏移、扫描距离与点数。",
            parameters=[
                ParameterSpec(
                    name="z_offset_m",
                    type="float",
                    description="Z 偏移，单位米",
                    unit="m",
                    required=True,
                ),
                ParameterSpec(
                    name="z_sweep_distance_m",
                    type="float",
                    description="Z 扫描距离，单位米",
                    unit="m",
                    required=True,
                    min_value=0.0,
                ),
                ParameterSpec(
                    name="num_points",
                    type="int",
                    description="扫描的点数",
                    required=True,
                    min_value=2,
                    max_value=10000,
                ),
                ParameterSpec(
                    name="backward_sweep",
                    type="bool",
                    description="启用反扫",
                    required=False,
                    default=True,
                ),
            ],
            estimated_duration_s=1.0,
            composition_level=1,
            tags=["spectroscopy", "z", "configure", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        z_offset = params["z_offset_m"]
        z_distance = params["z_sweep_distance_m"]
        num_points = params["num_points"]
        backward = params.get("backward_sweep", True)
        calls = []

        rec_open = context.safe_call("ZSpectr_Open")
        calls.append(rec_open)
        if rec_open.error:
            return SkillResult(
                skill_name="ConfigureZSpectr",
                success=False,
                error=rec_open.error,
                nanonis_calls=calls,
            )

        rec_range = context.safe_call("ZSpectr_RangeSet", z_offset, z_distance)
        calls.append(rec_range)
        if rec_range.error:
            return SkillResult(
                skill_name="ConfigureZSpectr",
                success=False,
                error=rec_range.error,
                nanonis_calls=calls,
            )

        rec_props = context.safe_call(
            "ZSpectr_PropsSet",
            int(backward), num_points, 1, 1, 2, 1,
        )
        calls.append(rec_props)
        if rec_props.error:
            return SkillResult(
                skill_name="ConfigureZSpectr",
                success=False,
                error=rec_props.error,
                nanonis_calls=calls,
            )

        return SkillResult(
            skill_name="ConfigureZSpectr",
            success=True,
            data={
                "z_offset_m": z_offset,
                "z_sweep_distance_m": z_distance,
                "num_points": num_points,
                "backward_sweep": backward,
            },
            nanonis_calls=calls,
        )


class AcquireZSpectr(BaseSkill):
    """Acquire a Z spectrum at the current position."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="AcquireZSpectr",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description="在当前针尖位置采一条 Z 谱。",
            preconditions=["z_controller_on"],
            estimated_duration_s=30.0,
            composition_level=1,
            tags=["spectroscopy", "z", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        calls = []

        rec_open = context.safe_call("ZSpectr_Open")
        calls.append(rec_open)

        rec_props = context.safe_call("ZSpectr_PropsSet", 0, 0, 0, 1, 2, 1)
        calls.append(rec_props)

        record = context.safe_call("ZSpectr_Start", 1, "")
        calls.append(record)
        if record.error:
            return SkillResult(
                skill_name="AcquireZSpectr",
                success=False,
                error=record.error,
                nanonis_calls=calls,
            )

        data: dict = {"acquisition_complete": True}
        parsed = record.return_value
        # return_value is (error_string, raw_bytes, Variables). For ZSpectr.Start
        # the Variables (ResponseTypes ["i","i","*+c","i","i","2f","i","*f"]) are:
        #   [0] channel-names byte size (int)   [1] number of channels (int)
        #   [2] channel names (1D str array)    [3] data rows (int)
        #   [4] data columns (int)              [5] data (2D float32) <- the spectrum
        #   [6] number of parameters (int)      [7] parameters (1D float32)
        # The OLD code read variables[0]/[1] (header ints) as the z/current
        # arrays — i.e. it returned the channel-count, never the spectrum.
        if isinstance(parsed, (list, tuple)) and len(parsed) > 2:
            channel_map, num_points, why = _reshape_spectrum(parsed[2])
        else:
            channel_map, num_points, why = {}, 0, (
                f"return_value 不是 (err, raw, Variables) 三元组: "
                f"{type(parsed).__name__}")
        data["spectrum_parsed"] = bool(channel_map)
        if channel_map:
            data["channel_names"] = list(channel_map.keys())
            data["num_points"] = num_points
            data.update(channel_map)
            # Resolve z / current / dIdV BY CHANNEL NAME (the old fixed-column
            # parse was transposed and mislabelled the traces).
            z = _match_channel(channel_map, ("z (", "z(", "z_m", "z pos"))
            if z is None:
                # fall back: a bare-"z"-named channel
                for name, trace in channel_map.items():
                    if str(name).strip().lower().startswith("z"):
                        z = trace
                        break
            if z is not None:
                data["z"] = z
            cur = _match_channel(channel_map, ("current",))
            if cur is not None:
                data["current"] = cur
            did = _match_channel(channel_map, ("lix", "did", "di/dv", "di_dv"))
            if did is not None:
                data["dIdV"] = did
        else:
            # ⚠️ **这里的结论和 AcquireSTS 不一样,而不一样是有依据的**:
            # 这个技能不调 ``_attach_saved_dat``,``ZSpectr_Start(1, "")`` 也不
            # 给 basename —— **内联这一份就是唯一的一份**。解不开 = 什么都没
            # 拿到,没有第二处可以去找。所以这里一律报失败,而 STS 那边只在
            # 「盘上也没有」时才报。(2026-08-15 普查 A3)
            data["spectrum_unparsed_reason"] = why
            logger.warning("AcquireZSpectr: 返回的数据块解不开 —— %s", why)
            return SkillResult(
                skill_name="AcquireZSpectr",
                success=False,
                error=(f"扫掠跑完了,但返回的数据块解不开({why})。"
                       f"这个技能不落盘,内联返回是唯一的一份 —— "
                       f"这次采集没有留下任何可用的谱。"),
                data=data,
                nanonis_calls=calls,
            )
        return SkillResult(
            skill_name="AcquireZSpectr",
            success=True,
            data=data,
            nanonis_calls=calls,
        )


class ConfigureSTSTiming(BaseSkill):
    """Configure STS timing parameters."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="ConfigureSTSTiming",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description="配置 Bias Spectroscopy 的时序参数。",
            parameters=[
                ParameterSpec(
                    name="z_avg_time_s", type="float",
                    description="Z 平均时间", unit="s",
                    required=True, min_value=0.0,
                ),
                ParameterSpec(
                    name="z_offset_m", type="float",
                    description="Z 偏移", unit="m",
                    required=False, default=0.0,
                ),
                ParameterSpec(
                    name="init_settling_s", type="float",
                    description="初始建立时间", unit="s",
                    required=True, min_value=0.0,
                ),
                ParameterSpec(
                    name="max_slew_rate_v_s", type="float",
                    description="最大压摆率", unit="V/s",
                    required=True, min_value=0.0,
                ),
                ParameterSpec(
                    name="settling_s", type="float",
                    description="每点的建立时间", unit="s",
                    required=True, min_value=0.0,
                ),
                ParameterSpec(
                    name="integration_s", type="float",
                    description="每点的积分时间", unit="s",
                    required=True, min_value=0.0,
                ),
                ParameterSpec(
                    name="end_settling_s", type="float",
                    description="结束时的建立时间", unit="s",
                    required=False, default=0.0, min_value=0.0,
                ),
                ParameterSpec(
                    name="z_ctrl_time_s", type="float",
                    description="Z 控制时间", unit="s",
                    required=False, default=0.0, min_value=0.0,
                ),
            ],
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["spectroscopy", "sts", "timing", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        record = context.safe_call(
            "BiasSpectr_TimingSet",
            params["z_avg_time_s"],
            params.get("z_offset_m", 0.0),
            params["init_settling_s"],
            params["max_slew_rate_v_s"],
            params["settling_s"],
            params["integration_s"],
            params.get("end_settling_s", 0.0),
            params.get("z_ctrl_time_s", 0.0),
        )
        if record.error:
            return SkillResult(
                skill_name="ConfigureSTSTiming",
                success=False,
                error=record.error,
                nanonis_calls=[record],
            )
        return SkillResult(
            skill_name="ConfigureSTSTiming",
            success=True,
            data={k: params.get(k, 0.0) for k in [
                "z_avg_time_s", "z_offset_m", "init_settling_s",
                "max_slew_rate_v_s", "settling_s", "integration_s",
                "end_settling_s", "z_ctrl_time_s",
            ]},
            nanonis_calls=[record],
        )


class StopSTS(BaseSkill):
    """Stop a running Bias Spectroscopy measurement."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="StopSTS",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.AUTO,
            description="停止当前的 Bias Spectroscopy 测量。",
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["spectroscopy", "sts", "stop"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        record = context.safe_call("BiasSpectr_Stop")
        if record.error:
            return SkillResult(
                skill_name="StopSTS", success=False,
                error=record.error, nanonis_calls=[record],
            )
        return SkillResult(
            skill_name="StopSTS", success=True,
            data={"stopped": True}, nanonis_calls=[record],
        )


class StopZSpectr(BaseSkill):
    """Stop a running Z Spectroscopy measurement."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="StopZSpectr",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.AUTO,
            description="停止当前的 Z Spectroscopy 测量。",
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["spectroscopy", "z", "stop"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        record = context.safe_call("ZSpectr_Stop")
        if record.error:
            return SkillResult(
                skill_name="StopZSpectr", success=False,
                error=record.error, nanonis_calls=[record],
            )
        return SkillResult(
            skill_name="StopZSpectr", success=True,
            data={"stopped": True}, nanonis_calls=[record],
        )


class ConfigureSTSChannels(BaseSkill):
    """Configure STS recorded channels."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="ConfigureSTSChannels",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description="设置 Bias Spectroscopy 期间记录哪些通道。",
            parameters=[
                ParameterSpec(
                    name="channel_indexes",
                    type="str",
                    description="逗号分隔的通道索引（0-23）",
                    required=True,
                ),
            ],
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["spectroscopy", "sts", "channels", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        idx_str = params["channel_indexes"]
        indexes = [int(x.strip()) for x in idx_str.split(",") if x.strip()]
        record = context.safe_call("BiasSpectr_ChsSet", indexes)
        if record.error:
            return SkillResult(
                skill_name="ConfigureSTSChannels", success=False,
                error=record.error, nanonis_calls=[record],
            )
        return SkillResult(
            skill_name="ConfigureSTSChannels", success=True,
            data={"channel_indexes": indexes}, nanonis_calls=[record],
        )


class ConfigureZSpectrTiming(BaseSkill):
    """Configure Z spectroscopy timing parameters."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="ConfigureZSpectrTiming",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description="配置 Z Spectroscopy 的时序参数。",
            parameters=[
                ParameterSpec(
                    name="z_avg_time_s", type="float",
                    description="Z 平均时间", unit="s",
                    required=True, min_value=0.0,
                ),
                ParameterSpec(
                    name="init_settling_s", type="float",
                    description="初始建立时间", unit="s",
                    required=True, min_value=0.0,
                ),
                ParameterSpec(
                    name="max_slew_rate_v_s", type="float",
                    description="最大压摆率", unit="V/s",
                    required=True, min_value=0.0,
                ),
                ParameterSpec(
                    name="settling_s", type="float",
                    description="每点的建立时间", unit="s",
                    required=True, min_value=0.0,
                ),
                ParameterSpec(
                    name="integration_s", type="float",
                    description="每点的积分时间", unit="s",
                    required=True, min_value=0.0,
                ),
                ParameterSpec(
                    name="end_settling_s", type="float",
                    description="结束时的建立时间", unit="s",
                    required=False, default=0.0, min_value=0.0,
                ),
                ParameterSpec(
                    name="z_ctrl_time_s", type="float",
                    description="Z 控制时间", unit="s",
                    required=False, default=0.0, min_value=0.0,
                ),
            ],
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["spectroscopy", "z", "timing", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        record = context.safe_call(
            "ZSpectr_TimingSet",
            params["z_avg_time_s"],
            params["init_settling_s"],
            params["max_slew_rate_v_s"],
            params["settling_s"],
            params["integration_s"],
            params.get("end_settling_s", 0.0),
            params.get("z_ctrl_time_s", 0.0),
        )
        if record.error:
            return SkillResult(
                skill_name="ConfigureZSpectrTiming",
                success=False,
                error=record.error,
                nanonis_calls=[record],
            )
        return SkillResult(
            skill_name="ConfigureZSpectrTiming",
            success=True,
            data={k: params.get(k, 0.0) for k in [
                "z_avg_time_s", "init_settling_s", "max_slew_rate_v_s",
                "settling_s", "integration_s", "end_settling_s", "z_ctrl_time_s",
            ]},
            nanonis_calls=[record],
        )


class GetSTSChannels(BaseSkill):
    """Get the list of recorded channels for Bias Spectroscopy."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetSTSChannels",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description="取 Bias Spectroscopy 记录的通道列表。",
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["spectroscopy", "sts", "channels", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        record = context.safe_call("BiasSpectr_ChsGet")
        if record.error:
            return SkillResult(
                skill_name="GetSTSChannels", success=False,
                error=record.error, nanonis_calls=[record],
            )
        # Decode the response envelope before interpreting its channel fields.
        # Numeric channel indexes and string names use different parser paths;
        # preserve the distinction instead of applying numeric tuple handling
        # indiscriminately to strings. core.nanonis_patch normalizes the parser,
        # while channel_ids_from_buffer also accepts supported test-double forms.
        parsed = record.return_value
        channel_indexes: list = []
        channel_names: list = []
        variables = parsed[2] if isinstance(parsed, (list, tuple)) and len(parsed) > 2 else None
        if isinstance(variables, (list, tuple)):
            if len(variables) > 1 and isinstance(variables[1], (list, tuple)):
                channel_indexes = channel_ids_from_buffer(variables)
            if len(variables) > 4 and isinstance(variables[4], (list, tuple)):
                channel_names = list(variables[4])
        return SkillResult(
            skill_name="GetSTSChannels", success=True,
            data={"channel_indexes": channel_indexes, "channel_names": channel_names},
            nanonis_calls=[record],
        )


class SetSTSChannels(BaseSkill):
    """Set the recorded channels for Bias Spectroscopy."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="SetSTSChannels",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description="设置 Bias Spectroscopy 记录的通道。",
            parameters=[
                ParameterSpec(
                    name="channel_indexes",
                    # NOTE: declared as str (comma-separated) because the agent
                    # schema builder has no list type — execute() coerces to a
                    # real list. See _coerce_int_list.
                    type="str",
                    description="通道索引（0-127），逗号分隔，例如 '0, 1, 2'",
                    required=True,
                ),
            ],
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["spectroscopy", "sts", "channels", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        channel_indexes = _coerce_int_list(params["channel_indexes"])
        # BiasSpectr_ChsSet(Channel_indexes: list) takes ONE arg — the library's
        # "+*i" wire format sends the array length itself, so passing len() too
        # is a TypeError on real hardware. (ConfigureSTSChannels above is the
        # correct pattern.) — spectroscopy channel-config fix 2026-06-02.
        record = context.safe_call("BiasSpectr_ChsSet", channel_indexes)
        if record.error:
            return SkillResult(
                skill_name="SetSTSChannels", success=False,
                error=record.error, nanonis_calls=[record],
            )
        return SkillResult(
            skill_name="SetSTSChannels", success=True,
            data={"channel_indexes": channel_indexes},
            nanonis_calls=[record],
        )


class GetSTSLimits(BaseSkill):
    """Get the bias voltage range for Bias Spectroscopy."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetSTSLimits",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description="取 Bias Spectroscopy 的 bias 电压范围。",
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["spectroscopy", "sts", "limits", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        record = context.safe_call("BiasSpectr_LimitsGet")
        if record.error:
            return SkillResult(
                skill_name="GetSTSLimits", success=False,
                error=record.error, nanonis_calls=[record],
            )
        # return_value is (error_string, raw_bytes, Variables). BiasSpectr.LimitsGet
        # ResponseTypes ["f","f"] -> Variables [start_v, end_v]. OLD code read
        # parsed[0]/[1] (the empty error string + raw bytes) -> float("") ValueError.
        parsed = record.return_value
        start_v = 0.0
        end_v = 0.0
        variables = parsed[2] if isinstance(parsed, (list, tuple)) and len(parsed) > 2 else None
        if isinstance(variables, (list, tuple)) and len(variables) >= 2:
            start_v = float(variables[0])
            end_v = float(variables[1])
        return SkillResult(
            skill_name="GetSTSLimits", success=True,
            data={"start_v": start_v, "end_v": end_v},
            nanonis_calls=[record],
        )


class SetSTSAdvancedProps(BaseSkill):
    """Set Bias Spectroscopy advanced properties."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="SetSTSAdvancedProps",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description="设置 Bias Spectroscopy 的高级属性。",
            parameters=[
                ParameterSpec(
                    name="reset_bias", type="int",
                    description="扫描后复位 bias（0=不改，1=On，2=Off）",
                    required=True, min_value=0, max_value=2,
                ),
                ParameterSpec(
                    name="z_controller_hold", type="int",
                    description="扫描期间保持 Z 控制器（0=不改，1=On，2=Off）",
                    required=True, min_value=0, max_value=2,
                ),
                ParameterSpec(
                    name="record_final_z", type="int",
                    description="记录最终的 Z（0=不改，1=On，2=Off）",
                    required=True, min_value=0, max_value=2,
                ),
                ParameterSpec(
                    name="lockin_run", type="int",
                    description="扫描期间运行 Lock-In（0=不改，1=On，2=Off）",
                    required=True, min_value=0, max_value=2,
                ),
            ],
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["spectroscopy", "sts", "advanced", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        record = context.safe_call(
            "BiasSpectr_AdvPropsSet",
            params["reset_bias"],
            params["z_controller_hold"],
            params["record_final_z"],
            params["lockin_run"],
        )
        if record.error:
            return SkillResult(
                skill_name="SetSTSAdvancedProps", success=False,
                error=record.error, nanonis_calls=[record],
            )
        return SkillResult(
            skill_name="SetSTSAdvancedProps", success=True,
            data={
                "reset_bias": params["reset_bias"],
                "z_controller_hold": params["z_controller_hold"],
                "record_final_z": params["record_final_z"],
                "lockin_run": params["lockin_run"],
            },
            nanonis_calls=[record],
        )


class GetSTSTiming(BaseSkill):
    """Get Bias Spectroscopy timing parameters."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetSTSTiming",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description="取 Bias Spectroscopy 的时序参数。",
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["spectroscopy", "sts", "timing", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        record = context.safe_call("BiasSpectr_TimingGet")
        if record.error:
            return SkillResult(
                skill_name="GetSTSTiming", success=False,
                error=record.error, nanonis_calls=[record],
            )
        # return_value is (error_string, raw_bytes, Variables). BiasSpectr.TimingGet
        # ResponseTypes are 8 floats -> read them from Variables (parsed[2]), not
        # from parsed[i] (which is error string / raw bytes / then out of range).
        parsed = record.return_value
        data: dict = {}
        keys = [
            "z_averaging_time_s", "z_offset_m", "initial_settling_time_s",
            "max_slew_rate_v_per_s", "settling_time_s", "integration_time_s",
            "end_settling_time_s", "z_control_time_s",
        ]
        variables = parsed[2] if isinstance(parsed, (list, tuple)) and len(parsed) > 2 else None
        if isinstance(variables, (list, tuple)):
            for i, key in enumerate(keys):
                if i < len(variables):
                    data[key] = float(variables[i])
        return SkillResult(
            skill_name="GetSTSTiming", success=True,
            data=data, nanonis_calls=[record],
        )


class GetSTSAltZCtrl(BaseSkill):
    """Get Bias Spectroscopy alternative Z controller settings."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetSTSAltZCtrl",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description="取 Bias Spectroscopy 的备用 Z 控制器设置。",
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["spectroscopy", "sts", "zctrl", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        record = context.safe_call("BiasSpectr_AltZCtrlGet")
        if record.error:
            return SkillResult(
                skill_name="GetSTSAltZCtrl", success=False,
                error=record.error, nanonis_calls=[record],
            )
        # return_value is (error_string, raw_bytes, Variables). BiasSpectr.AltZCtrlGet
        # ResponseTypes ["H","f","f"] -> Variables [enabled, setpoint, settling_time_s].
        parsed = record.return_value
        enabled = False
        setpoint = 0.0
        settling_time_s = 0.0
        variables = parsed[2] if isinstance(parsed, (list, tuple)) and len(parsed) > 2 else None
        if isinstance(variables, (list, tuple)) and len(variables) >= 3:
            enabled = bool(variables[0])
            setpoint = float(variables[1])
            settling_time_s = float(variables[2])
        return SkillResult(
            skill_name="GetSTSAltZCtrl", success=True,
            data={
                "enabled": enabled,
                "setpoint": setpoint,
                "settling_time_s": settling_time_s,
            },
            nanonis_calls=[record],
        )


class GetZSpectrChannels(BaseSkill):
    """Get the list of recorded channels for Z Spectroscopy."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetZSpectrChannels",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description="取 Z Spectroscopy 记录的通道列表。",
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["spectroscopy", "z", "channels", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        record = context.safe_call("ZSpectr_ChsGet")
        if record.error:
            return SkillResult(
                skill_name="GetZSpectrChannels", success=False,
                error=record.error, nanonis_calls=[record],
            )
        # return_value is (error_string, raw_bytes, Variables). For ZSpectr.ChsGet
        # (ResponseTypes ["i","*i","i","i","*+c"]) Variables[1]=channel indexes
        # (1D int array), Variables[4]=channel names. OLD code read parsed[0]/[1]
        # = the error string / raw bytes, never the data. Index unwrapping is the
        # byte-for-byte twin of GetSTSChannels above — see the note there.
        parsed = record.return_value
        channel_indexes: list = []
        channel_names: list = []
        variables = parsed[2] if isinstance(parsed, (list, tuple)) and len(parsed) > 2 else None
        if isinstance(variables, (list, tuple)):
            if len(variables) > 1 and isinstance(variables[1], (list, tuple)):
                channel_indexes = channel_ids_from_buffer(variables)
            if len(variables) > 4 and isinstance(variables[4], (list, tuple)):
                channel_names = list(variables[4])
        return SkillResult(
            skill_name="GetZSpectrChannels", success=True,
            data={"channel_indexes": channel_indexes, "channel_names": channel_names},
            nanonis_calls=[record],
        )


class SetZSpectrChannels(BaseSkill):
    """Set the recorded channels for Z Spectroscopy."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="SetZSpectrChannels",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description="设置 Z Spectroscopy 记录的通道。",
            parameters=[
                ParameterSpec(
                    name="channel_indexes",
                    # declared str (comma-separated); execute() coerces — agent
                    # schema has no list type. See _coerce_int_list.
                    type="str",
                    description="通道索引（0-127），逗号分隔，例如 '0, 1, 2'",
                    required=True,
                ),
            ],
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["spectroscopy", "z", "channels", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        channel_indexes = _coerce_int_list(params["channel_indexes"])
        # ZSpectr_ChsSet(Channel_indexes) takes ONE arg (the "+*i" wire format
        # sends the length); passing len() too is a TypeError on hardware.
        record = context.safe_call("ZSpectr_ChsSet", channel_indexes)
        if record.error:
            return SkillResult(
                skill_name="SetZSpectrChannels", success=False,
                error=record.error, nanonis_calls=[record],
            )
        return SkillResult(
            skill_name="SetZSpectrChannels", success=True,
            data={"channel_indexes": channel_indexes},
            nanonis_calls=[record],
        )


class GetZSpectrRange(BaseSkill):
    """Get Z Spectroscopy range settings."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetZSpectrRange",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description="取 Z Spectroscopy 的量程设置。",
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["spectroscopy", "z", "range", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        record = context.safe_call("ZSpectr_RangeGet")
        if record.error:
            return SkillResult(
                skill_name="GetZSpectrRange", success=False,
                error=record.error, nanonis_calls=[record],
            )
        # return_value is (error_string, raw_bytes, Variables). ZSpectr.RangeGet
        # ResponseTypes ["f","f"] -> Variables [z_offset_m, z_sweep_distance_m].
        parsed = record.return_value
        z_offset_m = 0.0
        z_sweep_distance_m = 0.0
        variables = parsed[2] if isinstance(parsed, (list, tuple)) and len(parsed) > 2 else None
        if isinstance(variables, (list, tuple)) and len(variables) >= 2:
            z_offset_m = float(variables[0])
            z_sweep_distance_m = float(variables[1])
        return SkillResult(
            skill_name="GetZSpectrRange", success=True,
            data={"z_offset_m": z_offset_m, "z_sweep_distance_m": z_sweep_distance_m},
            nanonis_calls=[record],
        )


class SetZSpectrRange(BaseSkill):
    """Set Z Spectroscopy range."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="SetZSpectrRange",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description="设置 Z Spectroscopy 的量程。",
            parameters=[
                ParameterSpec(
                    name="z_offset_m", type="float",
                    description="Z 偏移，单位米", unit="m",
                    required=True,
                ),
                ParameterSpec(
                    name="z_sweep_distance_m", type="float",
                    description="Z 扫描距离，单位米", unit="m",
                    required=True, min_value=0.0,
                ),
            ],
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["spectroscopy", "z", "range", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        record = context.safe_call(
            "ZSpectr_RangeSet",
            params["z_offset_m"],
            params["z_sweep_distance_m"],
        )
        if record.error:
            return SkillResult(
                skill_name="SetZSpectrRange", success=False,
                error=record.error, nanonis_calls=[record],
            )
        return SkillResult(
            skill_name="SetZSpectrRange", success=True,
            data={
                "z_offset_m": params["z_offset_m"],
                "z_sweep_distance_m": params["z_sweep_distance_m"],
            },
            nanonis_calls=[record],
        )


class GetZSpectrRetract(BaseSkill):
    """Get Z Spectroscopy auto-retract configuration."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetZSpectrRetract",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description="取 Z Spectroscopy 的自动退针配置。",
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["spectroscopy", "z", "retract", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        record = context.safe_call("ZSpectr_RetractGet")
        if record.error:
            return SkillResult(
                skill_name="GetZSpectrRetract", success=False,
                error=record.error, nanonis_calls=[record],
            )
        # return_value is (error_string, raw_bytes, Variables). ZSpectr.RetractGet
        # ResponseTypes ["H","f","i","H"] -> Variables [enable, threshold,
        # signal_index, comparison].
        parsed = record.return_value
        enabled = False
        threshold = 0.0
        signal_index = 0
        comparison = ">"
        variables = parsed[2] if isinstance(parsed, (list, tuple)) and len(parsed) > 2 else None
        if isinstance(variables, (list, tuple)) and len(variables) >= 4:
            enabled = bool(variables[0])
            threshold = float(variables[1])
            signal_index = int(variables[2])
            comparison = ">" if int(variables[3]) == 0 else "<"
        return SkillResult(
            skill_name="GetZSpectrRetract", success=True,
            data={
                "enabled": enabled,
                "threshold": threshold,
                "signal_index": signal_index,
                "comparison": comparison,
            },
            nanonis_calls=[record],
        )


class SetZSpectrRetract(BaseSkill):
    """Set Z Spectroscopy auto-retract conditions."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="SetZSpectrRetract",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description="设置 Z Spectroscopy 的自动退针条件。",
            parameters=[
                ParameterSpec(
                    name="enabled", type="int",
                    description="启用退针（0=不改，1=On，2=Off）",
                    required=True, min_value=0, max_value=2,
                ),
                ParameterSpec(
                    name="threshold", type="float",
                    description="退针的阈值",
                    required=True,
                ),
                ParameterSpec(
                    name="signal_index", type="int",
                    description="阈值所用的信号索引（-1=不改）",
                    required=True,
                ),
                ParameterSpec(
                    name="comparison", type="int",
                    description="比较运算符（0=>，1=<，2=不改）",
                    required=True, min_value=0, max_value=2,
                ),
            ],
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["spectroscopy", "z", "retract", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        record = context.safe_call(
            "ZSpectr_RetractSet",
            params["enabled"],
            params["threshold"],
            params["signal_index"],
            params["comparison"],
        )
        if record.error:
            return SkillResult(
                skill_name="SetZSpectrRetract", success=False,
                error=record.error, nanonis_calls=[record],
            )
        return SkillResult(
            skill_name="SetZSpectrRetract", success=True,
            data={
                "enabled": params["enabled"],
                "threshold": params["threshold"],
                "signal_index": params["signal_index"],
                "comparison": params["comparison"],
            },
            nanonis_calls=[record],
        )


class GetSTSDigSync(BaseSkill):
    """Get Bias Spectroscopy digital synchronization setting."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetSTSDigSync",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description="取 Bias Spectroscopy 的数字同步模式（Off/TTL/PulseSeq）。",
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["spectroscopy", "sts", "digsync", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        record = context.safe_call("BiasSpectr_DigSyncGet")
        if record.error:
            return SkillResult(
                skill_name="GetSTSDigSync", success=False,
                error=record.error, nanonis_calls=[record],
            )
        # return_value is (error_string, raw_bytes, Variables). DigSyncGet
        # ResponseTypes ["H"] -> Variables[0] = dig_sync.
        parsed = record.return_value
        dig_sync = 0
        variables = parsed[2] if isinstance(parsed, (list, tuple)) and len(parsed) > 2 else None
        if isinstance(variables, (list, tuple)) and len(variables) > 0:
            dig_sync = int(variables[0])
        _DIGSYNC_MAP = {0: "Off", 1: "TTL Sync", 2: "Pulse Sequence"}
        return SkillResult(
            skill_name="GetSTSDigSync", success=True,
            data={"dig_sync": dig_sync, "dig_sync_label": _DIGSYNC_MAP.get(dig_sync, str(dig_sync))},
            nanonis_calls=[record],
        )


class GetSTSTTLSync(BaseSkill):
    """Get Bias Spectroscopy TTL synchronization configuration."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetSTSTTLSync",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description="取 Bias Spectroscopy 的 TTL 同步配置。",
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["spectroscopy", "sts", "ttlsync", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        record = context.safe_call("BiasSpectr_TTLSyncGet")
        if record.error:
            return SkillResult(
                skill_name="GetSTSTTLSync", success=False,
                error=record.error, nanonis_calls=[record],
            )
        # return_value is (error_string, raw_bytes, Variables). TTLSyncGet
        # ResponseTypes ["H","H","f","f"] -> Variables [ttl_line, ttl_polarity,
        # time_to_on_s, on_duration_s].
        parsed = record.return_value
        ttl_line = 0
        ttl_polarity = 0
        time_to_on_s = 0.0
        on_duration_s = 0.0
        variables = parsed[2] if isinstance(parsed, (list, tuple)) and len(parsed) > 2 else None
        if isinstance(variables, (list, tuple)) and len(variables) >= 4:
            ttl_line = int(variables[0])
            ttl_polarity = int(variables[1])
            time_to_on_s = float(variables[2])
            on_duration_s = float(variables[3])
        return SkillResult(
            skill_name="GetSTSTTLSync", success=True,
            data={
                "ttl_line": ttl_line,
                "ttl_polarity": ttl_polarity,
                "time_to_on_s": time_to_on_s,
                "on_duration_s": on_duration_s,
            },
            nanonis_calls=[record],
        )


class GetSTSPulseSeqSync(BaseSkill):
    """Get Bias Spectroscopy pulse sequence synchronization configuration."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetSTSPulseSeqSync",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description="取 Bias Spectroscopy 的脉冲序列同步配置。",
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["spectroscopy", "sts", "pulseseq", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        record = context.safe_call("BiasSpectr_PulseSeqSyncGet")
        if record.error:
            return SkillResult(
                skill_name="GetSTSPulseSeqSync", success=False,
                error=record.error, nanonis_calls=[record],
            )
        # return_value is (error_string, raw_bytes, Variables). PulseSeqSyncGet
        # ResponseTypes ["H","I"] -> Variables [pulse_seq_nr, nr_periods].
        parsed = record.return_value
        pulse_seq_nr = 0
        nr_periods = 0
        variables = parsed[2] if isinstance(parsed, (list, tuple)) and len(parsed) > 2 else None
        if isinstance(variables, (list, tuple)) and len(variables) >= 2:
            pulse_seq_nr = int(variables[0])
            nr_periods = int(variables[1])
        return SkillResult(
            skill_name="GetSTSPulseSeqSync", success=True,
            data={"pulse_seq_nr": pulse_seq_nr, "nr_periods": nr_periods},
            nanonis_calls=[record],
        )


class GetSTSZOffRevert(BaseSkill):
    """Get Bias Spectroscopy Z Offset Revert flag."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetSTSZOffRevert",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description="取 Bias Spectroscopy 的 Z Offset Revert 标志。",
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["spectroscopy", "sts", "zoffrevert", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        record = context.safe_call("BiasSpectr_ZOffRevertGet")
        if record.error:
            return SkillResult(
                skill_name="GetSTSZOffRevert", success=False,
                error=record.error, nanonis_calls=[record],
            )
        # return_value is (error_string, raw_bytes, Variables). ZOffRevertGet
        # ResponseTypes ["H"] -> Variables[0] = z_off_revert.
        parsed = record.return_value
        z_off_revert = 0
        variables = parsed[2] if isinstance(parsed, (list, tuple)) and len(parsed) > 2 else None
        if isinstance(variables, (list, tuple)) and len(variables) > 0:
            z_off_revert = int(variables[0])
        return SkillResult(
            skill_name="GetSTSZOffRevert", success=True,
            data={"z_off_revert": bool(z_off_revert)},
            nanonis_calls=[record],
        )


class GetSTSMLSLockinPerSeg(BaseSkill):
    """Get Bias Spectroscopy MLS Lock-In per segment flag."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetSTSMLSLockinPerSeg",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description="取 MLS 模式下的 Lock-In per Segment 标志。",
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["spectroscopy", "sts", "mls", "lockin", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        record = context.safe_call("BiasSpectr_MLSLockinPerSegGet")
        if record.error:
            return SkillResult(
                skill_name="GetSTSMLSLockinPerSeg", success=False,
                error=record.error, nanonis_calls=[record],
            )
        # return_value is (error_string, raw_bytes, Variables). MLSLockinPerSegGet
        # ResponseTypes ["I"] -> Variables[0] = lockin_per_segment.
        parsed = record.return_value
        lockin_per_seg = 0
        variables = parsed[2] if isinstance(parsed, (list, tuple)) and len(parsed) > 2 else None
        if isinstance(variables, (list, tuple)) and len(variables) > 0:
            lockin_per_seg = int(variables[0])
        return SkillResult(
            skill_name="GetSTSMLSLockinPerSeg", success=True,
            data={"lockin_per_segment": bool(lockin_per_seg)},
            nanonis_calls=[record],
        )


class SetSTSMLSMode(BaseSkill):
    """Set Bias Spectroscopy sweep mode (Linear / MLS)."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="SetSTSMLSMode",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description="设置 Bias Spectroscopy 的扫描模式：Linear 或 MLS。",
            parameters=[
                ParameterSpec(
                    name="mode",
                    type="str",
                    description="'Linear' 或 'MLS'",
                    required=True,
                    allowed_values=["Linear", "MLS"],
                ),
            ],
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["spectroscopy", "sts", "mls", "mode", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        mode = params["mode"]
        # BiasSpectr_MLSModeSet(Sweep_mode: str) takes ONE arg — the "+*c" wire
        # format sends the string length itself, so passing len(mode) too is a
        # TypeError on real hardware. (Same pattern as the *ChsSet fixes.)
        record = context.safe_call("BiasSpectr_MLSModeSet", mode)
        if record.error:
            return SkillResult(
                skill_name="SetSTSMLSMode", success=False,
                error=record.error, nanonis_calls=[record],
            )
        return SkillResult(
            skill_name="SetSTSMLSMode", success=True,
            data={"mode": mode},
            nanonis_calls=[record],
        )


class SetSTSMLSVals(BaseSkill):
    """Set Bias Spectroscopy MLS segment configuration."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="SetSTSMLSVals",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description="设置 Bias Spectroscopy MLS 模式的多线段配置。",
            # NOTE: all per-segment arrays are declared as str (comma-separated)
            # because the agent schema builder has no list type — execute()
            # coerces each via _coerce_float_list / _coerce_int_list. Programmatic
            # callers may still pass real lists.
            parameters=[
                ParameterSpec(
                    name="bias_start_v",
                    type="str",
                    description="每段的起始 bias（V），逗号分隔，例如 '-1.0, 0.5'",
                    required=True,
                ),
                ParameterSpec(
                    name="bias_end_v",
                    type="str",
                    description="每段的终止 bias（V），逗号分隔",
                    required=True,
                ),
                ParameterSpec(
                    name="initial_settling_s",
                    type="str",
                    description="每段的初始建立时间（s），逗号分隔",
                    required=True,
                ),
                ParameterSpec(
                    name="settling_s",
                    type="str",
                    description="每段的建立时间（s），逗号分隔",
                    required=True,
                ),
                ParameterSpec(
                    name="integration_s",
                    type="str",
                    description="每段的积分时间（s），逗号分隔",
                    required=True,
                ),
                ParameterSpec(
                    name="steps",
                    type="str",
                    description="每段的步数，逗号分隔的整数",
                    required=True,
                ),
                ParameterSpec(
                    name="lockin_run",
                    type="str",
                    description="每段的 Lock-In 运行标志（0=Off，1=On），逗号分隔",
                    required=True,
                ),
            ],
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["spectroscopy", "sts", "mls", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        bias_start = _coerce_float_list(params["bias_start_v"])
        bias_end = _coerce_float_list(params["bias_end_v"])
        init_settling = _coerce_float_list(params["initial_settling_s"])
        settling = _coerce_float_list(params["settling_s"])
        integration = _coerce_float_list(params["integration_s"])
        steps = _coerce_int_list(params["steps"])
        lockin_run = _coerce_int_list(params["lockin_run"])
        num_segments = len(bias_start)

        # Equal-length validation: all 7 per-segment arrays
        # MUST have the same length. BiasSpectr_MLSValsSet sends num_segments then
        # each array; if they disagree the wire body contradicts the declared
        # segment count → misframed TCP / garbage segment config on the hardware.
        lengths = {
            "bias_start_v": len(bias_start), "bias_end_v": len(bias_end),
            "initial_settling_s": len(init_settling), "settling_s": len(settling),
            "integration_s": len(integration), "steps": len(steps),
            "lockin_run": len(lockin_run),
        }
        if num_segments == 0:
            return SkillResult(
                skill_name="SetSTSMLSVals", success=False,
                error="no MLS segments provided (bias_start_v is empty)")
        mismatched = {k: n for k, n in lengths.items() if n != num_segments}
        if mismatched:
            return SkillResult(
                skill_name="SetSTSMLSVals", success=False,
                error=(f"MLS per-segment arrays must all have {num_segments} "
                       f"elements; got lengths {lengths}"))

        # Bias-bound validation: these arrays are str-typed to
        # fit the agent tool schema, so the automatic ParameterSpec + global
        # bias-limit checks (which only run on NUMERIC params) are BYPASSED. An
        # LLM-hallucinated segment start/end could otherwise reach the junction
        # far outside ±10 V. Enforce the effective global bias bound here.
        try:
            from mast.config import SafetyLimits
            from mast.core.safety import _get_effective_limits
            lim = _get_effective_limits(SafetyLimits())
            lo, hi = float(lim.bias_min_v), float(lim.bias_max_v)
        except Exception:  # pragma: no cover - fall back to spec default
            lo, hi = -10.0, 10.0
        for pname, arr in (("bias_start_v", bias_start), ("bias_end_v", bias_end)):
            for i, v in enumerate(arr):
                if v < lo or v > hi:
                    return SkillResult(
                        skill_name="SetSTSMLSVals", success=False,
                        error=(f"MLS {pname}[{i}] = {v} V is outside the global "
                               f"bias safety bound [{lo}, {hi}] V"))

        record = context.safe_call(
            "BiasSpectr_MLSValsSet",
            num_segments, bias_start, bias_end,
            init_settling, settling, integration,
            steps, lockin_run,
        )
        if record.error:
            return SkillResult(
                skill_name="SetSTSMLSVals", success=False,
                error=record.error, nanonis_calls=[record],
            )
        return SkillResult(
            skill_name="SetSTSMLSVals", success=True,
            data={"num_segments": num_segments},
            nanonis_calls=[record],
        )


class SetSTSSafeCond1(BaseSkill):
    """Set Bias Spectroscopy 1st safe condition."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="SetSTSSafeCond1",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "[DEPRECATED —— Nanonis 里 Bias Spectroscopy 没有 safe-condition API；它总是失败。"
                "请改用 Z 谱学的 SetZSpectrRetract。] 设置 Bias Spectroscopy 的第一条 safe "
                "condition。"
            ),
            parameters=[
                ParameterSpec(
                    name="condition", type="int",
                    description="动作：0=不改，1=Off，2=Reverse，3=Stop",
                    required=True, min_value=0, max_value=3,
                ),
                ParameterSpec(
                    name="threshold", type="float",
                    description="阈值（NaN=不改）",
                    required=True,
                ),
                ParameterSpec(
                    name="signal_index", type="int",
                    description="信号索引（0-127，-1=不改）",
                    required=True,
                ),
                ParameterSpec(
                    name="comparison", type="int",
                    description="0=高于，1=低于，2=不改",
                    required=True, min_value=0, max_value=2,
                ),
            ],
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["spectroscopy", "sts", "safecond", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        # The Nanonis Bias Spectroscopy module has NO safe-condition / auto-retract
        # feature in nanonis_spm (no BiasSpectr_SafeCond* method exists; grep
        # NanonisClass.py confirms BiasSpectr_* ends at MLSValsGet). The auto-retract
        # safe-condition feature exists only for *Z* Spectroscopy
        # (SetZSpectrRetract / SetZSpectrRetractSecond). The old code called the
        # non-existent BiasSpectr_SafeCond1Set, which only ever returned a cryptic
        # "Method 'BiasSpectr_SafeCond1Set' not found on Nanonis instance" — and
        # only on real hardware. Fail fast with a clear, deterministic message.
        return SkillResult(
            skill_name="SetSTSSafeCond1",
            success=False,
            error=(
                "Bias Spectroscopy has no safe-condition / auto-retract feature in "
                "the Nanonis API. This control only exists for Z Spectroscopy — use "
                "SetZSpectrRetract (main condition) or the 2nd-condition Z retract "
                "skills instead."
            ),
            nanonis_calls=[],
        )


class GetSTSSafeCond1(BaseSkill):
    """Get Bias Spectroscopy 1st safe condition."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetSTSSafeCond1",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description=(
                "[DEPRECATED —— Nanonis 里 Bias Spectroscopy 没有 safe-condition API；它总是失败。"
                "请改用 Z 谱学的 GetZSpectrRetract。] 取 Bias Spectroscopy 的第一条 safe condition。"
            ),
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["spectroscopy", "sts", "safecond", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        # No BiasSpectr_SafeCond1Get method exists in nanonis_spm — the
        # safe-condition / auto-retract feature is Z-Spectroscopy-only
        # (GetZSpectrRetract). The old call to BiasSpectr_SafeCond1Get returned
        # only "Method ... not found" on real hardware. Fail fast and clearly.
        return SkillResult(
            skill_name="GetSTSSafeCond1",
            success=False,
            error=(
                "Bias Spectroscopy has no safe-condition / auto-retract feature in "
                "the Nanonis API. This control only exists for Z Spectroscopy — use "
                "GetZSpectrRetract / GetZSpectrRetract2nd instead."
            ),
            nanonis_calls=[],
        )


class SetSTSSafeCond2(BaseSkill):
    """Set Bias Spectroscopy 2nd safe condition."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="SetSTSSafeCond2",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "[DEPRECATED —— Nanonis 里 Bias Spectroscopy 没有 safe-condition API；它总是失败。"
                "请改用 Z 谱学的第二条 retract（GetZSpectrRetract2nd）。] 设置 Bias Spectroscopy "
                "的第二条 safe condition。"
            ),
            parameters=[
                ParameterSpec(
                    name="condition", type="int",
                    description="逻辑：-1=不改，0=Off，1=OR，2=AND，3=THEN",
                    required=True, min_value=-1, max_value=3,
                ),
                ParameterSpec(
                    name="threshold", type="float",
                    description="阈值（NaN=不改）",
                    required=True,
                ),
                ParameterSpec(
                    name="signal_index", type="int",
                    description="信号索引（0-127，-1=不改）",
                    required=True,
                ),
                ParameterSpec(
                    name="comparison", type="int",
                    description="0=高于，1=低于，2=不改",
                    required=True, min_value=0, max_value=2,
                ),
            ],
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["spectroscopy", "sts", "safecond", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        # No BiasSpectr_SafeCond2Set method exists in nanonis_spm — the
        # 2nd-condition auto-retract logic (OR/AND/THEN) is Z-Spectroscopy-only
        # (ZSpectr_RetractSecondSet). The old call to BiasSpectr_SafeCond2Set
        # returned only "Method ... not found" on real hardware. Fail fast.
        return SkillResult(
            skill_name="SetSTSSafeCond2",
            success=False,
            error=(
                "Bias Spectroscopy has no 2nd safe-condition / auto-retract feature "
                "in the Nanonis API. This control only exists for Z Spectroscopy — "
                "the 2nd-condition (OR/AND/THEN) retract logic is configured via the "
                "Z Spectroscopy auto-retract skills."
            ),
            nanonis_calls=[],
        )


class SetZSpectrAdvProps(BaseSkill):
    """Set Z Spectroscopy advanced properties."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="SetZSpectrAdvProps",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description="设置 Z Spectroscopy 的高级属性。",
            parameters=[
                ParameterSpec(
                    name="time_between_sweeps_s", type="float",
                    description="正扫与反扫之间的时间（s）",
                    unit="s", required=True, min_value=0.0,
                ),
                ParameterSpec(
                    name="record_final_z", type="int",
                    description="记录最终的 Z（0=不改，1=On，2=Off）",
                    required=True, min_value=0, max_value=2,
                ),
                ParameterSpec(
                    name="lockin_run", type="int",
                    description="运行 Lock-In（0=不改，1=On，2=Off）",
                    required=True, min_value=0, max_value=2,
                ),
                ParameterSpec(
                    name="reset_z", type="int",
                    description="扫描后复位 Z（0=不改，1=On，2=Off）",
                    required=True, min_value=0, max_value=2,
                ),
            ],
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["spectroscopy", "z", "advanced", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        record = context.safe_call(
            "ZSpectr_AdvPropsSet",
            params["time_between_sweeps_s"],
            params["record_final_z"],
            params["lockin_run"],
            params["reset_z"],
        )
        if record.error:
            return SkillResult(
                skill_name="SetZSpectrAdvProps", success=False,
                error=record.error, nanonis_calls=[record],
            )
        return SkillResult(
            skill_name="SetZSpectrAdvProps", success=True,
            data={
                "time_between_sweeps_s": params["time_between_sweeps_s"],
                "record_final_z": params["record_final_z"],
                "lockin_run": params["lockin_run"],
                "reset_z": params["reset_z"],
            },
            nanonis_calls=[record],
        )


class GetZSpectrDigSync(BaseSkill):
    """Get Z Spectroscopy digital synchronization setting."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetZSpectrDigSync",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description="取 Z Spectroscopy 的数字同步模式（Off/TTL/PulseSeq）。",
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["spectroscopy", "z", "digsync", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        record = context.safe_call("ZSpectr_DigSyncGet")
        if record.error:
            return SkillResult(
                skill_name="GetZSpectrDigSync", success=False,
                error=record.error, nanonis_calls=[record],
            )
        # return_value is (error_string, raw_bytes, Variables). ZSpectr.DigSyncGet
        # ResponseTypes ["H"] -> Variables[0] = dig_sync.
        parsed = record.return_value
        dig_sync = 0
        variables = parsed[2] if isinstance(parsed, (list, tuple)) and len(parsed) > 2 else None
        if isinstance(variables, (list, tuple)) and len(variables) > 0:
            dig_sync = int(variables[0])
        _DIGSYNC_MAP = {0: "Off", 1: "TTL Sync", 2: "Pulse Sequence"}
        return SkillResult(
            skill_name="GetZSpectrDigSync", success=True,
            data={"dig_sync": dig_sync, "dig_sync_label": _DIGSYNC_MAP.get(dig_sync, str(dig_sync))},
            nanonis_calls=[record],
        )


class GetZSpectrPulseSeqSync(BaseSkill):
    """Get Z Spectroscopy pulse sequence synchronization configuration."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetZSpectrPulseSeqSync",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description="取 Z Spectroscopy 的脉冲序列同步配置。",
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["spectroscopy", "z", "pulseseq", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        record = context.safe_call("ZSpectr_PulseSeqSyncGet")
        if record.error:
            return SkillResult(
                skill_name="GetZSpectrPulseSeqSync", success=False,
                error=record.error, nanonis_calls=[record],
            )
        # return_value is (error_string, raw_bytes, Variables). PulseSeqSyncGet
        # ResponseTypes ["H","I"] -> Variables [pulse_seq_nr, nr_periods].
        parsed = record.return_value
        pulse_seq_nr = 0
        nr_periods = 0
        variables = parsed[2] if isinstance(parsed, (list, tuple)) and len(parsed) > 2 else None
        if isinstance(variables, (list, tuple)) and len(variables) >= 2:
            pulse_seq_nr = int(variables[0])
            nr_periods = int(variables[1])
        return SkillResult(
            skill_name="GetZSpectrPulseSeqSync", success=True,
            data={"pulse_seq_nr": pulse_seq_nr, "nr_periods": nr_periods},
            nanonis_calls=[record],
        )


class GetZSpectrRetract2nd(BaseSkill):
    """Get Z Spectroscopy 2nd auto-retract condition."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetZSpectrRetract2nd",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description="取 Z Spectroscopy 的第二条自动退针条件。",
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["spectroscopy", "z", "retract", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        # Real method is ZSpectr_RetractSecondGet (the old "ZSpectr_Retract2ndGet"
        # does not exist -> "Method not found" on hardware). return_value is
        # (error_string, raw_bytes, Variables); ResponseTypes ["i","f","i","H"]
        # -> Variables [second_condition, threshold, signal_index, comparison].
        record = context.safe_call("ZSpectr_RetractSecondGet")
        if record.error:
            return SkillResult(
                skill_name="GetZSpectrRetract2nd", success=False,
                error=record.error, nanonis_calls=[record],
            )
        parsed = record.return_value
        condition = 0
        threshold = 0.0
        signal_index = 0
        comparison = 0
        variables = parsed[2] if isinstance(parsed, (list, tuple)) and len(parsed) > 2 else None
        if isinstance(variables, (list, tuple)) and len(variables) >= 4:
            condition = int(variables[0])
            threshold = float(variables[1])
            signal_index = int(variables[2])
            comparison = int(variables[3])
        _COND_MAP = {0: "-No-", 1: "OR", 2: "AND", 3: "THEN"}
        return SkillResult(
            skill_name="GetZSpectrRetract2nd", success=True,
            data={
                "condition": condition,
                "condition_label": _COND_MAP.get(condition, str(condition)),
                "threshold": threshold,
                "signal_index": signal_index,
                "comparison": ">" if comparison == 0 else "<",
            },
            nanonis_calls=[record],
        )


class SetZSpectrRetractDelay(BaseSkill):
    """Set Z Spectroscopy retract delay."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="SetZSpectrRetractDelay",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description="设置 Z Spectroscopy 中正扫与反扫之间的退针延迟（s）。",
            parameters=[
                ParameterSpec(
                    name="retract_delay_s", type="float",
                    description="退针延迟，单位秒",
                    unit="s", required=True, min_value=0.0,
                ),
            ],
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["spectroscopy", "z", "retract", "delay", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        record = context.safe_call(
            "ZSpectr_RetractDelaySet", params["retract_delay_s"],
        )
        if record.error:
            return SkillResult(
                skill_name="SetZSpectrRetractDelay", success=False,
                error=record.error, nanonis_calls=[record],
            )
        return SkillResult(
            skill_name="SetZSpectrRetractDelay", success=True,
            data={"retract_delay_s": params["retract_delay_s"]},
            nanonis_calls=[record],
        )


class GetZSpectrTTLSync(BaseSkill):
    """Get Z Spectroscopy TTL synchronization configuration."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetZSpectrTTLSync",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description="取 Z Spectroscopy 的 TTL 同步配置。",
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["spectroscopy", "z", "ttlsync", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        record = context.safe_call("ZSpectr_TTLSyncGet")
        if record.error:
            return SkillResult(
                skill_name="GetZSpectrTTLSync", success=False,
                error=record.error, nanonis_calls=[record],
            )
        # return_value is (error_string, raw_bytes, Variables). ZSpectr.TTLSyncGet
        # ResponseTypes ["H","H","f","f"] -> Variables [ttl_line, ttl_polarity,
        # time_to_on_s, on_duration_s].
        parsed = record.return_value
        ttl_line = 0
        ttl_polarity = 0
        time_to_on_s = 0.0
        on_duration_s = 0.0
        variables = parsed[2] if isinstance(parsed, (list, tuple)) and len(parsed) > 2 else None
        if isinstance(variables, (list, tuple)) and len(variables) >= 4:
            ttl_line = int(variables[0])
            ttl_polarity = int(variables[1])
            time_to_on_s = float(variables[2])
            on_duration_s = float(variables[3])
        return SkillResult(
            skill_name="GetZSpectrTTLSync", success=True,
            data={
                "ttl_line": ttl_line,
                "ttl_polarity": ttl_polarity,
                "time_to_on_s": time_to_on_s,
                "on_duration_s": on_duration_s,
            },
            nanonis_calls=[record],
        )


class GetZSpectrTiming(BaseSkill):
    """Get Z Spectroscopy timing parameters."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetZSpectrTiming",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description="取 Z Spectroscopy 的时序参数。",
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["spectroscopy", "z", "timing", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        record = context.safe_call("ZSpectr_TimingGet")
        if record.error:
            return SkillResult(
                skill_name="GetZSpectrTiming", success=False,
                error=record.error, nanonis_calls=[record],
            )
        # return_value is (error_string, raw_bytes, Variables). ZSpectr.TimingGet
        # ResponseTypes are 7 floats -> read them from Variables (parsed[2]).
        parsed = record.return_value
        data: dict = {}
        keys = [
            "z_averaging_time_s", "initial_settling_time_s",
            "max_slew_rate_v_per_s", "settling_time_s",
            "integration_time_s", "end_settling_time_s",
            "z_control_time_s",
        ]
        variables = parsed[2] if isinstance(parsed, (list, tuple)) and len(parsed) > 2 else None
        if isinstance(variables, (list, tuple)):
            for i, key in enumerate(keys):
                if i < len(variables):
                    data[key] = float(variables[i])
        return SkillResult(
            skill_name="GetZSpectrTiming", success=True,
            data=data, nanonis_calls=[record],
        )
