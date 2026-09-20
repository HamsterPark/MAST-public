"""ScanAt 门面 + SetScanBuffer(策略层的执行侧)。

用假的 ExecutionContext 断言**下发了哪些子调用、按什么顺序**。重点:
  * 策略层说「不下发」的东西,子调用序列里就不该出现(空即 no-op);
  * 分辨率必须排在 ConfigureScan 之后(它内部会写 Scan_BufferSet(ch,0,0));
  * 超时必须如实失败并停扫(硬件「停止」不等于「达标」)。
"""

from __future__ import annotations

import pytest

from mast.core import scan_policy
from mast.core.types import NanonisCallRecord, SkillResult
from mast.skills.builtins.scan_buffer import SetScanBuffer
from mast.skills.composite.scan_at import ScanAt


@pytest.fixture(autouse=True)
def _clean_policy():
    scan_policy.set_policy(None)
    yield
    scan_policy.set_policy(None)


class FakeContext:
    """记录 run()/safe_call() 的假上下文。

    ``sub_results`` 按技能名给出要返回的结果;没列的技能一律成功返回空 data。
    """

    def __init__(self, sub_results: dict | None = None,
                 call_returns: dict | None = None):
        self.runs: list[tuple[str, dict]] = []
        self.calls: list[tuple[str, tuple]] = []
        self._sub_results = sub_results or {}
        self._call_returns = call_returns or {}

    def run(self, skill_name: str, params: dict, version=None) -> SkillResult:
        self.runs.append((skill_name, dict(params)))
        if skill_name in self._sub_results:
            return self._sub_results[skill_name]
        return SkillResult(skill_name=skill_name, success=True, data={})

    def safe_call(self, method: str, *args, **kwargs) -> NanonisCallRecord:
        self.calls.append((method, args))
        ret = self._call_returns.get(method)
        if isinstance(ret, Exception):
            return NanonisCallRecord(method=method, args=args, error=str(ret))
        return NanonisCallRecord(method=method, args=args, return_value=ret)

    def check_abort(self) -> bool:
        return False


def _run_scan_at(context, **params):
    params.setdefault("center_x_m", 0.0)
    params.setdefault("center_y_m", 0.0)
    params.setdefault("size_m", 1e-7)
    skill = ScanAt()
    return skill.run_composite(context, params)


def _skill_order(ctx) -> list[str]:
    return [name for name, _ in ctx.runs]


# ── 子调用序列 ────────────────────────────────────────────────────────────────

def test_minimal_intent_produces_the_full_scan_sequence():
    """只说「扫哪、多大」,一次调用就把配置/分辨率/启动/等待全做完。"""
    ctx = FakeContext()
    _run_scan_at(ctx)
    assert _skill_order(ctx) == [
        "ConfigureScan", "SetScanBuffer", "StartScan", "WaitScanComplete",
    ]


def test_no_setpoint_bias_or_gain_calls_when_policy_says_keep_current():
    """空即 no-op:出厂表没配 setpoint/PI,就一个相关子调用都不许出现。

    「不下发」不等于「下发 0」—— 后者会把 setpoint 设成 0 安培。
    """
    ctx = FakeContext()
    _run_scan_at(ctx)
    order = _skill_order(ctx)
    assert "SetSetpoint" not in order
    assert "SetBias" not in order
    assert "SetZCtrlGain" not in order


def test_resolution_is_set_after_configure_scan():
    """配置之后显式设置分辨率，避免零值的协议歧义。"""
    ctx = FakeContext()
    _run_scan_at(ctx)
    order = _skill_order(ctx)
    assert order.index("SetScanBuffer") > order.index("ConfigureScan")


def test_explicit_bias_produces_a_setbias_step_before_configure():
    ctx = FakeContext()
    _run_scan_at(ctx, bias_v=-1.2)
    order = _skill_order(ctx)
    assert "SetBias" in order
    assert order.index("SetBias") < order.index("ConfigureScan")
    bias_params = dict(ctx.runs[order.index("SetBias")][1])
    assert bias_params == {"bias_v": -1.2}


def test_tier_setpoint_and_gains_become_steps():
    scan_policy.set_policy([
        {"name": "only", "upper_size_m": None, "pixels": 256, "line_time_s": 1.0,
         "setpoint_a": 3e-11, "p_gain": 1e-11, "time_constant_s": 5e-4},
    ])
    ctx = FakeContext()
    _run_scan_at(ctx)
    order = _skill_order(ctx)
    assert "SetSetpoint" in order and "SetZCtrlGain" in order
    gain_params = dict(ctx.runs[order.index("SetZCtrlGain")][1])
    assert gain_params["p_gain"] == 1e-11
    assert gain_params["time_constant_s"] == 5e-4
    assert gain_params["i_gain"] == pytest.approx(1e-11 / 5e-4)


def test_resolved_parameters_reach_configure_scan():
    ctx = FakeContext()
    _run_scan_at(ctx, size_m=1e-7)
    cfg = dict(ctx.runs[_skill_order(ctx).index("ConfigureScan")][1])
    assert cfg["width_m"] == pytest.approx(1e-7)
    assert cfg["height_m"] == pytest.approx(1e-7)
    assert cfg["line_time_s"] == 1.0          # highres 档
    assert cfg["set_scan_speed"] is True
    buf = dict(ctx.runs[_skill_order(ctx).index("SetScanBuffer")][1])
    assert buf == {"pixels": 256, "lines": 256}


def test_angle_is_omitted_when_not_specified():
    """不给角度 = 保持硬件现值;传 0 会让每次 recenter 都把画面转回 0°。"""
    ctx = FakeContext()
    _run_scan_at(ctx)
    cfg = dict(ctx.runs[_skill_order(ctx).index("ConfigureScan")][1])
    assert "angle_deg" not in cfg


def test_explicit_angle_is_passed_through():
    ctx = FakeContext()
    _run_scan_at(ctx, angle_deg=30.0)
    cfg = dict(ctx.runs[_skill_order(ctx).index("ConfigureScan")][1])
    assert cfg["angle_deg"] == 30.0


# ── 结果与 trace ─────────────────────────────────────────────────────────────

def test_result_carries_the_parameter_trace():
    """用户要能看到「这次用了什么参数、每个数字哪来的」。"""
    ctx = FakeContext()
    res = _run_scan_at(ctx, size_m=1e-7)
    assert res.success
    assert res.data["tier_name"] == "highres"
    assert res.data["param_trace"]["line_time_s"]["source"] == "tier-factory"
    assert res.data["param_trace"]["pixels"]["value"] == 256
    assert isinstance(res.data["param_summary"], list)


def test_explicit_values_are_traced_as_operator_specified():
    """转述与发明在结构上无法区分,所以每个显式值都要留下可审计的痕迹。"""
    ctx = FakeContext()
    res = _run_scan_at(ctx, line_time_s=0.05)
    assert res.data["param_trace"]["line_time_s"]["source"] == "explicit"


def test_invalid_size_fails_without_touching_hardware():
    ctx = FakeContext()
    res = _run_scan_at(ctx, size_m=0.0)
    assert not res.success
    assert "size_m" in res.error
    assert ctx.runs == []           # 一个子技能都没跑


# ── 超时:硬件「停止」不等于「达标」 ──────────────────────────────────────────

def test_timeout_fails_the_scan_even_though_every_step_succeeded():
    ctx = FakeContext(sub_results={
        "WaitScanComplete": SkillResult(
            skill_name="WaitScanComplete", success=True,
            data={"timed_out": True}),
    })
    res = _run_scan_at(ctx)
    assert not res.success
    assert "没有完成" in res.error


def test_timeout_stops_the_scan():
    """留着一个还在跑的扫描,会让之后每个要求 scan_not_running 的动作莫名失败。"""
    ctx = FakeContext(sub_results={
        "WaitScanComplete": SkillResult(
            skill_name="WaitScanComplete", success=True,
            data={"timed_out": True}),
    })
    _run_scan_at(ctx)
    assert ("Scan_Action", (1, 0)) in ctx.calls


def test_no_stop_call_on_a_normal_completion():
    ctx = FakeContext()
    _run_scan_at(ctx)
    assert not any(m == "Scan_Action" for m, _ in ctx.calls)


# ── 中途停止（v6.1.3，KNOWN_ISSUES §2.24）────────────────────────────────
#
# 这几条钉的是「**有人在读 `stopped_early`**」,不是「这个字段存在」。
# `WaitScanComplete` 从 v6.1.2 起就会如实报它了,而 `ScanAt` —— 用户扫图的
# 主路径 —— 一直没读,于是在主路径上按 Stop 打断一帧照样报成功。
# 断言全部落在 **ScanAt 的 outcome** 上,一条都不去问 data 里有没有那个键。


def _stopped_early_ctx(**over):
    data = {"timed_out": False, "stopped_early": True, "outcome": "stopped_early",
            "lines_done": 123, "lines_total": 512, "lines_verified": True}
    data.update(over)
    return FakeContext(sub_results={
        "WaitScanComplete": SkillResult(
            skill_name="WaitScanComplete", success=True, data=data),
    })


def test_a_scan_stopped_part_way_fails_even_though_every_step_succeeded():
    """主路径上的假成功:每一步都 success,而帧只扫了 24%。"""
    res = _run_scan_at(_stopped_early_ctx())
    assert not res.success
    assert "中途停止" in res.error


def test_the_error_says_how_far_it_got():
    """「扫到哪了」决定用户下一步做什么 —— 停在 3/512 和停在 500/512
    是两件不同的事。"""
    res = _run_scan_at(_stopped_early_ctx())
    assert "123/512" in res.error


def test_a_stop_and_a_timeout_do_not_read_the_same():
    """两者要做的事相反:超时要调大 timeout,中途停止要去查是谁停的。
    合成一句会把人送去调一个根本没问题的参数。

    断言两条**互不相同**,而不是「中途停止那句里不出现『超时』」——
    它里面**故意**出现了(「**不是**超时,调大 timeout 不解决问题」),
    那正是这条消息最有用的一句话。"""
    stopped = _run_scan_at(_stopped_early_ctx()).error
    timed_out = _run_scan_at(FakeContext(sub_results={
        "WaitScanComplete": SkillResult(
            skill_name="WaitScanComplete", success=True,
            data={"timed_out": True}),
    })).error

    assert stopped != timed_out
    assert "中途停止" in stopped and "中途停止" not in timed_out
    assert "没有完成" in timed_out and "没有完成" not in stopped
    # 而且它主动把人从错误的修法上劝开。
    assert "不是" in stopped and "timeout" in stopped


def test_a_stopped_scan_is_not_stopped_again():
    """走到这里说明 Scan_StatusGet 已经读到 0,扫描本来就停了。
    再发一次 Scan_Action 是对着已停的扫描做一次无意义的硬件写 ——
    「失败了就顺手停一下」看着对称,但它把一个只读的结论变成了写操作。"""
    ctx = _stopped_early_ctx()
    _run_scan_at(ctx)
    assert not any(m == "Scan_Action" for m, _ in ctx.calls)


def test_a_complete_frame_still_succeeds():
    """对照组。少了它,「中途停止会失败」也可能只是因为 ScanAt 全都失败。"""
    ctx = FakeContext(sub_results={
        "WaitScanComplete": SkillResult(
            skill_name="WaitScanComplete", success=True,
            data={"timed_out": False, "stopped_early": False,
                  "outcome": "completed", "lines_done": 512,
                  "lines_total": 512, "lines_verified": True}),
    })
    assert _run_scan_at(ctx).success


def test_an_implementation_that_does_not_report_it_keeps_the_old_behaviour():
    """失败开放:缺字段读作「这个实现不报中途停止」,退回旧行为 ——
    而不是把「少一个可选字段」变成一次扫描失败。"""
    ctx = FakeContext(sub_results={
        "WaitScanComplete": SkillResult(
            skill_name="WaitScanComplete", success=True,
            data={"timed_out": False}),      # v6.1.1 形态
    })
    assert _run_scan_at(ctx).success


def test_wait_timeout_is_derived_from_the_resolved_geometry():
    """写死的短超时会把慢扫 / 高分辨率的图在中途截断。"""
    ctx = FakeContext()
    _run_scan_at(ctx, size_m=1e-7)          # highres: 256 线 × 1 s × 2 = 512 s
    wait = dict(ctx.runs[_skill_order(ctx).index("WaitScanComplete")][1])
    assert wait["timeout_ms"] > 512 * 1000


def test_explicit_wait_timeout_is_a_floor_not_a_cap():
    """显式等待时长是下限：较小值由几何预算抬高，较大值保持优先。"""
    ctx = FakeContext()
    _run_scan_at(ctx, wait_timeout_s=60.0)
    wait = dict(ctx.runs[_skill_order(ctx).index("WaitScanComplete")][1])
    assert wait["timeout_ms"] >= 300_000, (
        "60 s 是个比下限（300 s）还小的显式值 —— 它不该压过几何推导，"
        "实际拿到 %s ms" % wait["timeout_ms"])

    ctx2 = FakeContext()
    _run_scan_at(ctx2, wait_timeout_s=9_999.0)
    wait2 = dict(ctx2.runs[_skill_order(ctx2).index("WaitScanComplete")][1])
    assert wait2["timeout_ms"] == 9_999_000, (
        "显式值大于几何值时它必须胜出 —— ForgeAuTip 的 1300 s 不许被缩短")


# ── purpose ──────────────────────────────────────────────────────────────────

def test_purpose_survey_on_a_small_frame_uses_coarse_parameters():
    """「在这 50 nm 快扫一眼」—— 尺寸是精扫的,参数要粗扫的。"""
    ctx = FakeContext()
    res = _run_scan_at(ctx, size_m=5e-8, purpose="survey")
    assert res.data["tier_name"] == "survey"
    buf = dict(ctx.runs[_skill_order(ctx).index("SetScanBuffer")][1])
    assert buf["pixels"] == 256
    assert any("强制换档" in w for w in res.data["policy_warnings"])


# ── 元数据契约 ────────────────────────────────────────────────────────────────

def test_optional_overrides_have_no_defaults():
    """有默认值就等于「模型不填也会有个数字」—— 那正是要消除的东西。"""
    meta = ScanAt().metadata()
    by_name = {p.name: p for p in meta.parameters}
    for key in ("bias_v", "setpoint_a", "line_time_s", "pixels", "angle_deg"):
        assert by_name[key].required is False
        assert by_name[key].default is None, f"{key} 不该有默认值"


def test_override_descriptions_tell_the_model_to_omit():
    """显式参数覆盖应保持声明语义。"""
    meta = ScanAt().metadata()
    by_name = {p.name: p for p in meta.parameters}
    for key in ("bias_v", "setpoint_a", "line_time_s", "pixels", "angle_deg"):
        d = by_name[key].description
        assert "用户" in d and ("说过" in d or "点名" in d or "指定" in d), (
            f"{key} 没说清「只有现场提过这个值才传」：{d}")
        assert "留空" in d or "别传" in d or "不要传" in d, (
            f"{key} 没说清「否则就留空」：{d}")


def test_pi_gains_are_not_exposed_as_scan_at_parameters():
    """「用 P=1e-11 扫」不是人话 —— 要调就去档位表或直接用 SetZCtrlGain。"""
    names = {p.name for p in ScanAt().metadata().parameters}
    assert "p_gain" not in names and "i_gain" not in names


def test_required_parameters_are_exactly_the_intent():
    required = {p.name for p in ScanAt().metadata().parameters if p.required}
    assert required == {"center_x_m", "center_y_m", "size_m"}

# SetScanBuffer 的通道列表兼容单元素元组和裸整数。
# 每条断言分别覆盖两种回包形态，防止只适配模拟器或只适配仪器协议。
@pytest.fixture(params=["real_rig_one_tuple", "stub_bare_int"])
def wrap_ch(request):
    return (lambda c: (c,)) if request.param == "real_rig_one_tuple" else (lambda c: c)


def _buffer_get(channels, pixels, lines, wrap=lambda c: c):
    return ["", b"", [len(channels), [wrap(c) for c in channels], pixels, lines]]


def test_set_scan_buffer_preserves_the_current_channels(wrap_ch):
    """传空通道列表会把采集通道清空 —— 扫出来的图一个通道都没有。

    写回去的必须是**裸 int**:元组原样送进 Scan_BufferSet 是另一个方向的 bug。
    """
    ctx = FakeContext(call_returns={
        "Scan_BufferGet": _buffer_get([0, 14], 256, 256, wrap=wrap_ch),
    })
    res = SetScanBuffer().execute(ctx, {"pixels": 512})
    assert res.success
    set_calls = [c for c in ctx.calls if c[0] == "Scan_BufferSet"]
    assert set_calls == [("Scan_BufferSet", ([0, 14], 512, 512))]


def test_set_scan_buffer_defaults_lines_to_pixels(wrap_ch):
    ctx = FakeContext(call_returns={
        "Scan_BufferGet": _buffer_get([0], 256, 256, wrap=wrap_ch)})
    SetScanBuffer().execute(ctx, {"pixels": 1024})
    assert ("Scan_BufferSet", ([0], 1024, 1024)) in ctx.calls


def test_set_scan_buffer_honours_explicit_lines(wrap_ch):
    ctx = FakeContext(call_returns={
        "Scan_BufferGet": _buffer_get([0], 256, 256, wrap=wrap_ch)})
    SetScanBuffer().execute(ctx, {"pixels": 512, "lines": 128})
    assert ("Scan_BufferSet", ([0], 512, 128)) in ctx.calls


def test_set_scan_buffer_refuses_when_channels_cannot_be_read():
    """猜一个通道列表写回去会破坏采集配置 —— 宁可失败。"""
    ctx = FakeContext(call_returns={"Scan_BufferGet": "garbage"})
    res = SetScanBuffer().execute(ctx, {"pixels": 512})
    assert not res.success
    assert "无法解析" in res.error
    assert not any(m == "Scan_BufferSet" for m, _ in ctx.calls)


def test_set_scan_buffer_refuses_when_no_channel_selected(wrap_ch):
    ctx = FakeContext(call_returns={
        "Scan_BufferGet": _buffer_get([], 256, 256, wrap=wrap_ch)})
    res = SetScanBuffer().execute(ctx, {"pixels": 512})
    assert not res.success
    assert "采集通道" in res.error


def test_set_scan_buffer_reports_unverified_readback(wrap_ch):
    """「硬件没报错」不等于「值真的变了」。"""
    calls = {"Scan_BufferGet": _buffer_get([0], 256, 256, wrap=wrap_ch)}
    ctx = FakeContext(call_returns=calls)
    res = SetScanBuffer().execute(ctx, {"pixels": 512})
    # 回读仍是 256(桩不会变),所以 verified 必须是 False 且明说
    assert res.success
    assert res.data["verified"] is False
    assert "回读未能确认" in res.data["warning"]


def test_set_scan_buffer_verified_when_readback_matches(wrap_ch):
    class VaryingContext(FakeContext):
        def __init__(self):
            super().__init__()
            self._n = 0

        def safe_call(self, method, *args, **kwargs):
            self.calls.append((method, args))
            if method == "Scan_BufferGet":
                self._n += 1
                px = 256 if self._n == 1 else 512
                return NanonisCallRecord(
                    method=method, args=args,
                    return_value=_buffer_get([0], px, px, wrap=wrap_ch))
            return NanonisCallRecord(method=method, args=args)

    ctx = VaryingContext()
    res = SetScanBuffer().execute(ctx, {"pixels": 512})
    assert res.data["verified"] is True
    assert "warning" not in res.data
    assert res.data["previous_pixels"] == 256


def test_set_scan_buffer_survives_the_exact_real_rig_reply():
    """按协议结构独立构造数据包。"""
    ctx = FakeContext(call_returns={
        "Scan_BufferGet": ["", b"", [2, [(0,), (30,)], 512, 512]]})
    res = SetScanBuffer().execute(ctx, {"pixels": 256})
    assert res.success
    assert ("Scan_BufferSet", ([0, 30], 256, 256)) in ctx.calls


def test_set_scan_buffer_writes_even_when_previous_resolution_is_unreadable():
    """通道读到了、分辨率读不出来 → 照写(通道保住了),但 summary 不许印
    ``Nonex None``。把可写的情况拦下来才是更糟的 bug。"""
    ctx = FakeContext(call_returns={
        "Scan_BufferGet": ["", b"", [1, [(0,)], "?", None]]})
    res = SetScanBuffer().execute(ctx, {"pixels": 512})
    assert res.success
    assert ("Scan_BufferSet", ([0], 512, 512)) in ctx.calls
    assert res.data["previous_pixels"] is None
    assert "None" not in res.summary and "未知" in res.summary
