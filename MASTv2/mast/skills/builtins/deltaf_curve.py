"""``AcquireDeltaFCurve`` —— 采一条 Δf(z) 力谱曲线,并留下可分析的 .dat。

``AcquireZSpectr`` 写死了空的存盘名,所以它不落文件;力谱的整个分析半段都要读 ``.dat``,
于是这一条自己走 ``safe_call``,照 ``AcquireSTS`` 的样子传 ``save_basename``。

采之前先把 PLL 的状态读回来:输出没开就直接失败并说清楚,而不是采一条全是噪声的曲线。
f0、振幅设定点、静止 Δf 都记进结果,反演技能要用。

通道留空时按供应商信号名寻找频移、电流和振幅；不能写死目标仪器的通道索引。
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
from mast.skills.base import BaseSkill

logger = logging.getLogger(__name__)

_NAME = "AcquireDeltaFCurve"
_DF_HINTS = ("freq. shift", "freq shift", "frequency shift")
_AMP_HINTS = ("amplitude",)
_OSC_HINTS = ("oc ", "ocd", "osc", "pll", "excitation")


class AcquireDeltaFCurve(BaseSkill):
    """在当前位置采一条 Δf(z),存成 .dat。"""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name=_NAME,
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "在针尖当前位置采一条频移随高度变化的曲线 Δf(z),同时记电流与振幅,并存成 "
                ".dat 供反演使用。扫描方向朝表面,正的 z_sweep_distance_m 表示往里走多远。"
                "采之前会读回 PLL 状态:输出没打开就直接失败,不采一条噪声。"
                "力谱必须配一条干净表面上的同扫程曲线做背景,分两次调用本技能取。"
            ),
            parameters=[
                ParameterSpec(
                    name="save_basename", type="str",
                    description=("存盘名前缀,例如 'Fe_atom'。强烈建议给:后面的反演按"
                                 "文件路径工作,不给就只能靠时间猜是哪一条。"),
                    required=False, default=""),
                ParameterSpec(
                    name="z_sweep_distance_m", type="float", unit="m",
                    description="往表面走多远,例如 '500p'(SI 前缀必须写)。",
                    required=True, min_value=1e-11, max_value=1e-8),
                ParameterSpec(
                    name="z_offset_m", type="float", unit="m",
                    description=("起点相对反馈高度的偏移,正值是先抬高,例如 '300p'。"
                                 "Δf 的极小值常常在反馈高度之上,所以通常要先抬。"
                                 "留空表示从反馈高度直接起扫。"),
                    required=False, min_value=-5e-9, max_value=5e-9),
                ParameterSpec(name="num_points", type="int",
                              description="曲线点数。", required=False, default=200,
                              min_value=8, max_value=4096),
                ParameterSpec(name="backward_sweep", type="bool",
                              description="是否也采回程(用来看迟滞)。",
                              required=False, default=True),
                ParameterSpec(
                    name="channel_indexes", type="str",
                    description=("要记的信号索引,逗号分隔。留空则按名字自动找频移、电流、"
                                 "振幅三路。"),
                    required=False, default=""),
                ParameterSpec(name="modulator_index", type="int",
                              description="PLL 调制器序号(单传感器机器就是 1)。",
                              required=False, default=1, min_value=1, max_value=8),
                ParameterSpec(name="require_pll_on", type="bool",
                              description="PLL 输出没开时直接失败(默认如此)。",
                              required=False, default=True),
            ],
            preconditions=["z_controller_on"],
            estimated_duration_s=60.0,
            composition_level=1,
            tags=["spectroscopy", "z", "pll", "frequency", "qplus", "afm", "force", "write"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        calls = []
        mod = int(params.get("modulator_index") or 1)

        def call(name, *args):
            if name == "PLL_OutOnOffGet":
                rec = context.safe_call("PLL_OutOnOffGet", *args)
            elif name == "PLL_CenterFreqGet":
                rec = context.safe_call("PLL_CenterFreqGet", *args)
            elif name == "PLL_FreqShiftGet":
                rec = context.safe_call("PLL_FreqShiftGet", *args)
            elif name == "PLL_AmpCtrlSetpntGet":
                rec = context.safe_call("PLL_AmpCtrlSetpntGet", *args)
            elif name == "Signals_NamesGet":
                rec = context.safe_call("Signals_NamesGet", *args)
            elif name == "ZSpectr_Open":
                rec = context.safe_call("ZSpectr_Open", *args)
            elif name == "ZSpectr_ChsSet":
                rec = context.safe_call("ZSpectr_ChsSet", *args)
            elif name == "ZSpectr_RangeSet":
                rec = context.safe_call("ZSpectr_RangeSet", *args)
            elif name == "ZSpectr_PropsSet":
                rec = context.safe_call("ZSpectr_PropsSet", *args)
            elif name == "ZSpectr_Start":
                rec = context.safe_call("ZSpectr_Start", *args)
            else:
                raise ValueError(f"Unsupported delta-f acquisition command: {name}")
            calls.append(rec)
            return rec

        rec = call("PLL_OutOnOffGet", mod)
        if rec.error:
            return SkillResult(skill_name=_NAME, success=False, nanonis_calls=calls,
                               error=f"读不到 PLL 输出状态: {rec.error}")
        on = self._first_number(rec.return_value)
        if params.get("require_pll_on", True) and not on:
            return SkillResult(skill_name=_NAME, success=False, nanonis_calls=calls,
                               error=("PLL 输出是关的,采不到力谱。先 PLLOnOff 打开输出、"
                                      "设好振幅,等振荡起振(约 3 倍 Q/πf0)再来。"))
        f0 = self._first_number(call("PLL_CenterFreqGet", mod).return_value)
        df_rest = self._first_number(call("PLL_FreqShiftGet", mod).return_value)
        amp_set = self._first_number(call("PLL_AmpCtrlSetpntGet", mod).return_value)

        chans, chan_err = self._channels(params, call)
        if chans is None:
            return SkillResult(skill_name=_NAME, success=False, nanonis_calls=calls,
                               error=chan_err)

        call("ZSpectr_Open")
        call("ZSpectr_ChsSet", chans)
        z_off = params.get("z_offset_m")
        call("ZSpectr_RangeSet", float(z_off or 0.0), float(params["z_sweep_distance_m"]))
        n = int(params.get("num_points") or 200)
        bwd = 1 if params.get("backward_sweep", True) else 2
        call("ZSpectr_PropsSet", bwd, n, 1, 1, 2, 1)
        basename = str(params.get("save_basename") or "")
        watermark = self._watermark(context)
        rec = call("ZSpectr_Start", 1, basename)
        if rec.error:
            return SkillResult(skill_name=_NAME, success=False, nanonis_calls=calls,
                               error=f"Δf(z) 采集失败: {rec.error}")

        data = {"acquisition_complete": True, "num_points": n,
                "z_sweep_distance_m": float(params["z_sweep_distance_m"]),
                "z_offset_m": float(z_off or 0.0),
                "f0_hz": f0, "df_rest_hz": df_rest, "amplitude_setpoint_m": amp_set,
                "sign_convention": "z_rel 从 0 起、向表面为负",
                "save_basename": basename or None}
        parsed = self._parse(rec.return_value, data)
        data["spectrum_parsed"] = parsed
        path = self._attach_dat(context, basename, watermark)
        if path:
            data["path"] = path
        else:
            data.setdefault("warnings", []).append("dat_attribution_failed")
        if not parsed and not path:
            return SkillResult(skill_name=_NAME, success=False, nanonis_calls=calls, data=data,
                               error="曲线块解不开,也没找到对应的 .dat")
        summary = (f"Δf(z) {n} 点,扫程 {float(params['z_sweep_distance_m']) * 1e12:.0f} pm"
                   + (f",最小 {data['df_min_hz']:.2f} Hz" if data.get("df_min_hz") is not None else "")
                   + (f",存为 {path}" if path else ",未找到存盘文件"))
        return SkillResult(skill_name=_NAME, success=True, data=data, summary=summary,
                           nanonis_calls=calls)

    # ── helpers ──
    @staticmethod
    def _first_number(value):
        """The first number anywhere in a reply, however deeply it is nested.

        A parsed Nanonis reply is ``(error, raw_bytes, [values...])`` — the numbers live one
        level down. Scanning only the top level finds nothing but an empty error string and a
        bytes blob, and returns None for every reading; the caller then reports the PLL as off
        while it is running.
        """
        if isinstance(value, bool):
            return float(value)
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, (list, tuple)):
            for v in value:
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    return float(v)
                if isinstance(v, (list, tuple)):
                    for w in v:
                        if isinstance(w, (int, float)) and not isinstance(w, bool):
                            return float(w)
        return None

    def _channels(self, params: dict, call) -> tuple[list[int] | None, str]:
        raw = str(params.get("channel_indexes") or "").strip()
        if raw:
            try:
                return [int(v) for v in raw.replace(";", ",").split(",") if v.strip()], ""
            except ValueError:
                return None, f"channel_indexes 读不成整数列表: {raw!r}"
        rec = call("Signals_NamesGet")
        names = None if rec.error else self._string_list(rec.return_value)
        if not names:
            return None, "读不到信号表,请显式给 channel_indexes(频移、电流、振幅)"
        out: list[int] = []
        df_i = self._find(names, _DF_HINTS)
        cur_i = self._find(names, ("current",))
        amp_i = self._find_amplitude(names)
        if df_i is None:
            return None, f"信号表里没有频移通道(有的是 {names[:24]}…)"
        out.append(df_i)
        for i in (cur_i, amp_i):
            if i is not None and i not in out:
                out.append(i)
        return out, ""

    @classmethod
    def _string_list(cls, value) -> list[str] | None:
        """The list of names inside a parsed reply, however deeply it is nested.

        ``Signals_NamesGet`` comes back as ``(error, raw, [size, count, [names...]])``, so the
        names are two levels down. Looking only one level finds a list whose first element is
        an integer and gives up — and the caller then says the signal table is unreadable on
        an instrument that answered it perfectly."""
        if isinstance(value, (str, bytes)):
            return None
        if isinstance(value, (list, tuple)):
            if value and all(isinstance(v, str) for v in value):
                return [str(v) for v in value]
            for item in value:
                got = cls._string_list(item)
                if got:
                    return got
        return None

    @staticmethod
    def _find(names: list[str], hints) -> int | None:
        for i, n in enumerate(names):
            low = str(n).lower()
            if any(h in low for h in hints):
                return i
        return None

    @staticmethod
    def _find_amplitude(names: list[str]) -> int | None:
        for i, n in enumerate(names):
            low = str(n).lower()
            if any(h in low for h in _AMP_HINTS) and any(o in low for o in _OSC_HINTS):
                return i
        return None

    @staticmethod
    def _parse(value, data: dict) -> bool:
        import numpy as np

        try:
            from mast.skills.builtins.spectroscopy import _reshape_spectrum

            # a parsed reply is (error, raw_bytes, [variables...]) and the spectrum block is
            # the variables list — handing the whole triple over gives "3 items, not a
            # spectrum", which is true of the envelope and says nothing about the sweep
            block = value[2] if (isinstance(value, (list, tuple)) and len(value) > 2) else value
            # (channel_map, num_points, reason) — the reason is the whole point of the third
            # element: an unparsable block and a genuinely empty sweep look identical without it
            channels, n_points, reason = _reshape_spectrum(block)
        except Exception as exc:  # noqa: BLE001
            data["parse_error"] = repr(exc)
            return False
        if not channels:
            data["parse_error"] = reason or "empty channel map"
            return False
        data["channel_names"] = list(channels)
        data["num_points"] = int(n_points)
        for name, arr in channels.items():
            low = str(name).lower()
            a = np.asarray(arr, dtype=float)
            if "[bwd]" in low:
                continue
            if "z rel" in low:
                data["z_rel"] = a.tolist()
            elif any(h in low for h in _DF_HINTS):
                data["freq_shift_hz"] = a.tolist()
                if a.size:
                    data["df_min_hz"] = float(np.nanmin(a))
                    if "z_rel" in data:
                        data["z_at_df_min_m"] = float(np.asarray(data["z_rel"])[int(np.nanargmin(a))])
            elif "current" in low:
                data["current_a"] = a.tolist()
            elif any(h in low for h in _AMP_HINTS):
                data["amplitude_m"] = a.tolist()
        return "z_rel" in data or "freq_shift_hz" in data

    @staticmethod
    def _watermark(context) -> float:
        from mast.skills.builtins.scan_extra import _candidate_save_dirs

        newest = 0.0
        for root in _candidate_save_dirs(context):
            try:
                for p in root.rglob("*.dat"):
                    newest = max(newest, p.stat().st_mtime)
            except OSError:
                continue
        return newest

    @staticmethod
    def _attach_dat(context, basename: str, watermark: float) -> str | None:
        """The newest .dat that carries our basename and post-dates the acquisition.

        Three layers, the same ones ``SpectroscopyAtPositions`` uses: a file was found at all,
        its name carries this basename, and it is newer than the mark taken before the sweep.
        A file that fails any of them is not this measurement, and inheriting the previous
        point's file is the one mistake that quietly ruins a whole run."""
        from mast.skills.builtins.scan_extra import _candidate_save_dirs, find_latest_saved

        latest = find_latest_saved(_candidate_save_dirs(context), "*.dat", max_age_s=120)
        if latest is None:
            return None
        try:
            if latest.stat().st_mtime < watermark - 1e-6:
                return None
        except OSError:
            return None
        if basename and basename not in latest.name:
            return None
        try:
            from mast.core.scan_registry import record_scan_path

            record_scan_path(latest)
        except Exception:  # noqa: BLE001 — best effort
            pass
        return str(latest)
