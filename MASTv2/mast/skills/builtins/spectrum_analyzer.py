"""Spectrum analyser — how you find out WHY the images are noisy.

The Nanonis Spectrum Analyzer (``SpectrumAnlzr_*``, 18 verbs) takes any signal and
shows you its noise spectrum. It is the instrument you reach for when a scan looks
wrong and you need to know whether it is the building, the mains, the preamp or the
tip:

  * a peak at 50 Hz and its harmonics → mains pickup, a ground loop
  * a broad hump at a few hundred Hz → the building / the isolation table
  * a rise toward DC → drift, thermal
  * a flat floor that is simply too high → the preamp, the gain, the cabling

MAST could read the spectrum and could not configure the analyser — the FFT window,
the averaging, the AC coupling and the cursors were all unwrapped. Which meant the
agent could look at a spectrum it had no way to make trustworthy: no averaging (so
every peak is noise), a rectangular window (so every peak is smeared), DC coupled
(so the interesting part is buried under the offset).

``GetSpectrumAnalyzerData`` returns the band RMS and the DC value together with the
spectrum, because "how much noise, in this band, in real units" is the number you
actually want — not a picture.
"""

from __future__ import annotations

from mast.core.types import (
    ParameterSpec,
    SafetyLevel,
    SkillCategory,
    SkillMetadata,
    SkillResult,
)
from mast.skills.base import BaseSkill

# SpectrumAnlzr_CursorPosSet(instance, Cursor_type, X1_Hz, X2_Hz): cursor_type
# 0 = the pair that bound the RMS band.
_CURSOR_BAND = 0


def _fail(name: str, error: str, calls: list) -> SkillResult:
    return SkillResult(skill_name=name, success=False, error=error, nanonis_calls=calls)


def _rv(record):
    """用 io.nanonis_files.decode_reply 提取回包内容，排除错误与原始字节信封。
    
    原始 bytes 不应进入需要 JSON 序列化的结果；所有调用方共用同一解码入口。"""
    from mast.io.nanonis_files import decode_reply

    rv = getattr(record, "return_value", None)
    return None if rv is None else decode_reply(rv)


class ConfigureSpectrumAnalyzer(BaseSkill):
    """Window, averaging, coupling — the three things that decide whether a spectrum means anything."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="ConfigureSpectrumAnalyzer",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.AUTO,
            description=(
                "配置频谱分析仪：FFT 窗、平均、以及 AC 耦合。只读 —— 分析仪只是把信号数字化，"
                "它不驱动任何东西。\n"
                "\n"
                "这三项决定了一条谱值不值得看：\n"
                "• **averaging** —— count=1 时每个 bin 都只是一次带噪的单点采样，每一个「峰」"
                "都是巧合。相信一个峰之前，先平均 10–50×。\n"
                "• **fft_window** —— 矩形窗（0）会把每一个音调抹到相邻的 bin 上。"
                "看噪声时该用的默认值是 Hann（通常是 1）。\n"
                "• **ac_coupling** —— DC 耦合下，一个大的偏置会主导整条谱，"
                "有意思的那几个数量级会被压在它底下。\n"
                "\n"
                "然后调用 GetSpectrumAnalyzerData。"
            ),
            parameters=[
                ParameterSpec(name="instance", type="int",
                              description="用哪一个分析仪实例（从 1 开始，Nanonis 编号）",
                              required=False, default=1, min_value=1, max_value=8),
                ParameterSpec(name="fft_window", type="int",
                              description="FFT 窗索引（0 = 矩形窗；通常该用 Hann）",
                              required=False, default=1, min_value=0, max_value=8),
                ParameterSpec(name="averaging_count", type="int",
                              description="平均多少条谱（1 = 不平均 —— 每个峰都是噪声）",
                              required=False, default=20, min_value=1, max_value=10_000),
                ParameterSpec(name="averaging_mode", type="int",
                              description="平均模式索引（0 = 不平均，1 = 线性，2 = 指数…）",
                              required=False, default=1, min_value=0, max_value=4),
                ParameterSpec(name="weighting_mode", type="int",
                              description="加权模式索引",
                              required=False, default=0, min_value=0, max_value=4),
                ParameterSpec(name="ac_coupling", type="bool",
                              description="对输入做 AC 耦合（剥掉 DC 偏置）",
                              required=False, default=True),
            ],
            estimated_duration_s=1.5,
            composition_level=0,
            tags=["spectrum", "noise", "fft", "diagnostics"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        name = "ConfigureSpectrumAnalyzer"
        inst = int(params.get("instance", 1) or 1)
        calls: list = []

        rec = context.safe_call("SpectrumAnlzr_FFTWindowSet", inst,
                                int(params.get("fft_window", 1) or 1))
        calls.append(rec)
        if rec.error:
            return _fail(name, f"SpectrumAnlzr_FFTWindowSet failed: {rec.error}", calls)

        rec = context.safe_call("SpectrumAnlzr_AveragSet", inst,
                                int(params.get("averaging_mode", 1) or 1),
                                int(params.get("weighting_mode", 0) or 0),
                                int(params.get("averaging_count", 20) or 20))
        calls.append(rec)
        if rec.error:
            return _fail(name, f"SpectrumAnlzr_AveragSet failed: {rec.error}", calls)

        rec = context.safe_call("SpectrumAnlzr_ACCouplingSet", inst,
                                1 if bool(params.get("ac_coupling", True)) else 0)
        calls.append(rec)
        if rec.error:
            return _fail(name, f"SpectrumAnlzr_ACCouplingSet failed: {rec.error}", calls)

        return SkillResult(
            skill_name=name, success=True, nanonis_calls=calls,
            data={"instance": inst,
                  "averaging_count": int(params.get("averaging_count", 20) or 20),
                  "ac_coupling": bool(params.get("ac_coupling", True))},
            summary=(f"频谱分析仪 #{inst} 已配置："
                     f"平均 {params.get('averaging_count', 20)} 次，"
                     f"{'AC' if params.get('ac_coupling', True) else 'DC'} 耦合"),
        )


class SetSpectrumAnalyzerBand(BaseSkill):
    """The frequency band the RMS is measured over."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="SetSpectrumAnalyzerBand",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.AUTO,
            description=(
                "设置频谱分析仪上报 band RMS 所用的频段（那一对游标），单位是 HERTZ。\n"
                "\n"
                "band RMS 是回答「到底有多少噪声」的那一个数 —— 例如电流信号上 1 Hz 到 1 kHz，"
                "告诉你反馈环实际要与之共处的噪声。先设频段，再读 GetSpectrumAnalyzerData.band_rms。"
            ),
            parameters=[
                ParameterSpec(name="f_low_hz", type="float",
                              description="频段下边界，单位 HERTZ",
                              unit="Hz", required=True, min_value=0.0, max_value=1e7),
                ParameterSpec(name="f_high_hz", type="float",
                              description="频段上边界，单位 HERTZ",
                              unit="Hz", required=True, min_value=0.0, max_value=1e7),
                ParameterSpec(name="instance", type="int",
                              description="用哪一个分析仪实例（从 1 开始）",
                              required=False, default=1, min_value=1, max_value=8),
            ],
            estimated_duration_s=1.0,
            composition_level=0,
            tags=["spectrum", "noise", "diagnostics"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        name = "SetSpectrumAnalyzerBand"
        lo = float(params["f_low_hz"])
        hi = float(params["f_high_hz"])
        if lo >= hi:
            return _fail(name, f"f_low_hz ({lo:g}) 必须小于 f_high_hz ({hi:g})", [])
        inst = int(params.get("instance", 1) or 1)
        rec = context.safe_call("SpectrumAnlzr_CursorPosSet", inst, _CURSOR_BAND, lo, hi)
        if rec.error:
            return _fail(name, f"SpectrumAnlzr_CursorPosSet failed: {rec.error}", [rec])
        return SkillResult(skill_name=name, success=True, nanonis_calls=[rec],
                           data={"instance": inst, "f_low_hz": lo, "f_high_hz": hi},
                           summary=f"频谱 RMS 频带 = [{lo:g}, {hi:g}] Hz")


class GetSpectrumAnalyzerData(BaseSkill):
    """The spectrum, the band RMS, and the DC value."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="GetSpectrumAnalyzerData",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description=(
                "读频谱分析仪：谱本身、BAND RMS（由 SetSpectrumAnalyzerBand 设定的频段内的噪声 —— "
                "值得拿出来说的就是这一个数）、DC 值，以及当前设置，好让你判断这条谱可不可信。\n"
                "\n"
                "诊断读法：50 Hz 处的峰及其谐波是市电串入 / 地环路；"
                "几百赫兹处宽缓的鼓包是楼体或隔振台；朝 DC 方向抬起是漂移；单纯就是太高的本底是前放、"
                "增益或走线。\n"
                "\n"
                "相信任何一个峰之前，先看 `averaging` —— count=1 时每个 bin "
                "都只是一次带噪的单点采样。"
            ),
            parameters=[
                ParameterSpec(name="instance", type="int",
                              description="用哪一个分析仪实例（从 1 开始）",
                              required=False, default=1, min_value=1, max_value=8),
            ],
            estimated_duration_s=2.0,
            composition_level=0,
            tags=["spectrum", "noise", "read", "diagnostics"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        inst = int(params.get("instance", 1) or 1)
        calls: list = []
        data: dict = {"instance": inst}
        # Literal verbs: every safety tool in this repo (abort-policy check,
        # security audit, API-coverage census) finds Nanonis calls by grepping
        # safe_call("…"). A verb behind a variable is invisible to all of them.
        for key, thunk in (
            ("band_rms", lambda: context.safe_call("SpectrumAnlzr_BandRMSGet", inst)),
            ("dc", lambda: context.safe_call("SpectrumAnlzr_DCGet", inst)),
            ("band", lambda: context.safe_call("SpectrumAnlzr_CursorPosGet", inst, _CURSOR_BAND)),
            ("averaging", lambda: context.safe_call("SpectrumAnlzr_AveragGet", inst)),
            ("fft_window", lambda: context.safe_call("SpectrumAnlzr_FFTWindowGet", inst)),
            ("ac_coupling", lambda: context.safe_call("SpectrumAnlzr_ACCouplingGet", inst)),
        ):
            rec = thunk()
            calls.append(rec)
            data[key] = None if rec.error else _rv(rec)
        return SkillResult(skill_name="GetSpectrumAnalyzerData", success=True,
                           nanonis_calls=calls, data=data)
