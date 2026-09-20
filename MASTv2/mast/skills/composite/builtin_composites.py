"""Built-in composites reconstructed as DECLARATIVE specs (P5, 2026-06-26).

The hand-written ``CompositeSkillGraph`` composites (FullScan, ConditionTip,
ShapeTipOnSurface, …) were the proof that the declarative IR was not expressive
enough to hold them. P5 closed that gap:

  * control primitives — ``try``/``finally``, ``break``/``continue``,
    ``succeed``/``fail``, ``success_when`` (see spec.py / interpreter.py);
  * G1 governed wrappers — ``GrabScanFrameData`` / ``CheckScanForCrash`` wrap
    the only out-of-graph raw ``Scan_FrameDataGrab`` + numpy crash math, so the
    declarative layer never needs a raw-TCP / eval node (the「数据而非代码」rail).

This module reconstructs the built-in workflows as ``CompositeSpec`` trees that
reference only REAL registered skills. They are seeded into the version store so
they are openable / forkable / editable in the builder (the original point of
the exercise). The hand-written Python versions are kept as the runtime
reference; these declarative twins coexist for authoring.

Every spec here has a behaviour test in
``tests/v2/skills/composite/test_builtin_composites.py`` asserting it drives the
same sub-skill sequence + verdict as its Python counterpart.
"""
from __future__ import annotations

from mast.skills.composite.spec import CompositeSpec, ParamSpec


def _wait_finished(var: str = "wait") -> str:
    """表达式:``var`` 这一步的扫描**真的扫完了** —— 没失败、没超时、没被中途停下。

    ``WaitScanComplete`` 在**每一种**扫描结束方式上都报 success,所以「这一步没报错」
    远不等于「这一帧扫完了」(KNOWN_ISSUES §2.24)。下面每个模板里凡是要拿这一帧
    去做下一步判断的地方,都过这道闸。

    ⚠️ 三个子句都用 ``in`` 守住,不写裸下标:``safe_eval`` 的下标缺键抛 ExprError,
    **整条复合当场中止**,而报出来的是「扫描技能崩了」,跟真正的原因(少一个可选
    字段)长得毫无关系。任何不报这些字段的 ``WaitScanComplete``(旧版 / 替身 / 桩)
    都会踩到 —— 加守卫之前这个坑当场咬过一次。
    缺字段读作「这个实现不报这件事」→ 退回旧行为。
    """
    return (f"'_failed' not in {var}"
            f" and not ('timed_out' in {var} and {var}['timed_out'])"
            f" and not ('stopped_early' in {var} and {var}['stopped_early'])")


def _full_scan() -> CompositeSpec:
    """ConfigureScan → SetScanSpeed → StartScan → WaitScanComplete → crash check.

    Fails on scan timeout or a detected tip crash (CheckScanForCrash, G1).
    """
    return CompositeSpec(
        name="FullScan",
        description="一步式扫描：配置帧+速度→启动→等结束→撞针检测；超时或撞针则失败。",
        safety_level="confirm",
        params=[
            ParamSpec("center_x_m", "number", 0.0, "扫描中心 X (m)", True),
            ParamSpec("center_y_m", "number", 0.0, "扫描中心 Y (m)", True),
            ParamSpec("width_m", "number", 1e-7, "宽 (m)", True),
            ParamSpec("height_m", "number", 1e-7, "高 (m)", True),
            # ⚠️ 0.1 与 Python 真身**不一致,而且现在改不了**(2026-08-25 查清)。
            #
            # 真身 full_scan.py:114 刻意写 `default=None`,注释说:写了 0.1,
            # 「没传」与「显式传 0.1」就永远分不开,分尺度档位表也就永远轮不到。
            #
            # 照着改成 None,**12 条测试当场报红**:
            #   TypeError: unsupported operand type(s) for /: float and NoneType
            #   (spec.py:116,来自下面的 {"$expr": "width_m / line_time_s"})
            # 孪生体的表达式引擎在**编排时**就要拿它做除法,而它**没有**真身
            # 那套「留空则查档位表」的解析 —— 同 wait_timeout_s 那条注释说的
            # 「静态 spec 读不到实时行数」是同一件事。
            #
            # ⇒ 这不是笔误,是**两条执行路径的能力差**。0.1 对 1 µm 的巡查图和
            #   5 nm 的原子分辨图给同一个速度(差一两个数量级),所以走 builder
            #   这条路的扫描仍然拿不到档位表的速度。
            #
            # **要对齐需要先做什么**:让 spec 的表达式层能解析「留空 → 查
            #   scan_policy.resolve_line_time(尺寸)」。那是功能改动,得有人拍板。
            ParamSpec("line_time_s", "number", 0.1, "每行时间 (s)", False),
            ParamSpec("channels", "string", "Z,Current", "采集通道", False),
            # Authoring default floored at the runtime FullScan floor (300 s). The
            # old 60 s timed out every normal scan; the declarative twin is a
            # static spec so it can't read the live line count the way the Python
            # runtime does — 300 s is the safe minimum for a builder-run.
            ParamSpec("wait_timeout_s", "number", 300.0, "等扫完超时 (s)", False),
        ],
        nodes=[
            {"type": "step", "id": "configure", "skill": "ConfigureScan", "params": {
                "center_x_m": {"$expr": "center_x_m"}, "center_y_m": {"$expr": "center_y_m"},
                "width_m": {"$expr": "width_m"}, "height_m": {"$expr": "height_m"},
                "channels": {"$expr": "channels"}, "set_scan_speed": False}},
            {"type": "step", "id": "set_speed", "skill": "SetScanSpeed", "params": {
                "fwd_speed": {"$expr": "width_m / line_time_s"},
                "bwd_speed": {"$expr": "width_m / line_time_s"},
                "fwd_line_time": {"$expr": "line_time_s"},
                "bwd_line_time": {"$expr": "line_time_s"}, "keep_const": 0}},
            {"type": "step", "id": "start", "skill": "StartScan", "params": {}},
            {"type": "step", "id": "wait", "skill": "WaitScanComplete",
             "params": {"timeout_ms": {"$expr": "wait_timeout_s * 1000"}}},
            # ⚠️ 两条 cond 都用 `in` 守住,不写裸下标 —— safe_eval 的下标缺键抛
            # ExprError,**整条复合当场中止**,而报出来的是「扫描技能崩了」,
            # 跟真正的原因(少一个可选字段)长得毫无关系。任何不报该字段的
            # WaitScanComplete(旧版 / 替身 / 桩)都会踩到。
            #
            # 缺字段读作「这个实现不报这件事」→ 退回旧行为。这与判据在
            # **信息缺失**时的取向一致 —— `_measure_lines` 的每一条早退
            # (读不到缓冲区 / 读不到帧 / 帧行数与配置不符 / 抛异常)都返回
            # `stopped_early=False`,只有唯一一条完整验证过的路径才置 True。
            # ⚠️ 仅限「信息缺失」这一面:判据对**错误的证据**并不安全
            # (机器若不按 NaN 填空行,会假报中途停止)——那一面靠 §2.24 的
            # 回滚判据兜,不靠这个守卫。
            {"type": "if", "id": "timed_out",
             "cond": "'timed_out' in wait and wait['timed_out']",
             "then": [{"type": "fail", "reason": "'Scan timed out'"}], "else": []},
            # 中途停止是**第三种**结局,与超时分开报:超时要调 timeout,
            # 中途停止要去查是谁停的(用户 Stop / Nanonis 自停 / 安全停机)。
            # 排在撞针检测之前 —— 那个检测读帧数据,在一片 NaN 上它量的不是针尖。
            {"type": "if", "id": "stopped_early",
             "cond": "'stopped_early' in wait and wait['stopped_early']",
             "then": [{"type": "fail",
                       "reason": "'Scan stopped early — frame incomplete "
                                 "(' + str(wait['lines_done']) + '/' "
                                 "+ str(wait['lines_total']) + ' lines)'"}],
             "else": []},
            {"type": "step", "id": "crash", "skill": "CheckScanForCrash",
             "params": {"channels": "0,14"}},
            {"type": "if", "id": "crashed", "cond": "crash['crash_indicator']",
             "then": [{"type": "fail",
                       "reason": "'CRASH_DETECTED: ' + crash['status']"}], "else": []},
        ],
        outputs=[{"name": "crash_check", "expr": "crash['status']"}],
        tags=["scan", "builtin"],
        notes="Declarative twin of full_scan.py:FullScan (P5).",
    )


def _tip_pulse() -> CompositeSpec:
    """Snapshot bias, then apply N hardware-timed bias pulses for conditioning."""
    return CompositeSpec(
        name="TipPulse",
        description="偏压脉冲修针：先快照当前偏压，再施加 count 次硬件定时脉冲。",
        safety_level="confirm",
        params=[
            ParamSpec("pulse_v", "number", 3.0, "脉冲电压 (V)", True),
            ParamSpec("duration_s", "number", 0.1, "脉冲宽度 (s)", False),
            ParamSpec("count", "int", 1, "脉冲次数", False),
        ],
        nodes=[
            {"type": "step", "id": "snapshot_bias", "skill": "GetBias", "params": {}},
            {"type": "loop", "id": "pulses", "mode": "repeat",
             "count": "count", "var": "i", "body": [
                {"type": "step", "id": "pulse", "skill": "BiasPulse", "params": {
                    "width_s": {"$expr": "duration_s"}, "bias_v": {"$expr": "pulse_v"},
                    "z_hold": 0, "absolute": True}},
             ]},
        ],
        outputs=[{"name": "original_bias_v", "expr": "snapshot_bias['bias_v']"}],
        tags=["tip", "conditioning", "builtin"],
        notes="Declarative twin of tip_pulse.py:TipPulse (P5).",
    )


def _grid_sts() -> CompositeSpec:
    """NxN grid STS with partial-success semantics (ok iff ≥1 point succeeded)."""
    return CompositeSpec(
        name="GridSTS",
        description="N×N 栅格 STS：逐点移动+采谱；部分成功（≥1 点成功即算成功）。",
        safety_level="confirm",
        params=[
            ParamSpec("center_x_m", "number", 0.0, "中心 X (m)", True),
            ParamSpec("center_y_m", "number", 0.0, "中心 Y (m)", True),
            ParamSpec("nx", "int", 3, "X 点数", False),
            ParamSpec("ny", "int", 3, "Y 点数", False),
            ParamSpec("spacing_m", "number", 1e-9, "点间距 (m)", True),
            ParamSpec("start_v", "number", -2.0, "STS 起始 (V)", False),
            ParamSpec("end_v", "number", 2.0, "STS 终止 (V)", False),
            ParamSpec("num_points", "int", 40, "STS 点数", False),
        ],
        success_when="succeeded >= 1",
        fail_message="'no STS point succeeded'",
        nodes=[
            {"type": "set", "id": "init", "var": "succeeded", "value": "0"},
            {"type": "step", "id": "configure", "skill": "ConfigureSTS", "params": {
                "start_v": {"$expr": "start_v"}, "end_v": {"$expr": "end_v"},
                "num_points": {"$expr": "num_points"}}},
            {"type": "loop", "id": "rows", "mode": "repeat", "count": "ny", "var": "iy", "body": [
                {"type": "loop", "id": "cols", "mode": "repeat", "count": "nx", "var": "ix", "body": [
                    {"type": "step", "id": "move", "skill": "MoveToXY", "optional": True, "params": {
                        "x_m": {"$expr": "center_x_m + (ix - (nx - 1) / 2) * spacing_m"},
                        "y_m": {"$expr": "center_y_m + (iy - (ny - 1) / 2) * spacing_m"},
                        "wait": True}},
                    {"type": "step", "id": "sts", "skill": "AcquireSTS", "optional": True, "params": {}},
                    {"type": "if", "id": "ok", "cond": "'_failed' not in sts",
                     "then": [{"type": "set", "id": "inc", "var": "succeeded",
                               "value": "succeeded + 1"}], "else": []},
                ]},
            ]},
        ],
        outputs=[{"name": "succeeded", "expr": "succeeded"}],
        tags=["sts", "grid", "builtin"],
        notes="Declarative twin of grid_sts.py:GridSTS (P5).",
    )


def _condition_tip() -> CompositeSpec:
    """Closed-loop tip conditioning: pulse → scan → assess → repeat until sharp."""
    return CompositeSpec(
        name="ConditionTip",
        description="修针闭环：脉冲→扫描→FFT 评质→重复，直到达标或用尽次数。",
        safety_level="confirm",
        params=[
            ParamSpec("pulse_v", "number", 3.0, "脉冲电压 (V)", True),
            ParamSpec("max_attempts", "int", 5, "最大轮次", False),
            ParamSpec("target_quality", "number", 0.3, "目标质量", False),
            ParamSpec("center_x_m", "number", 0.0, "扫描中心 X (m)", False),
            ParamSpec("center_y_m", "number", 0.0, "扫描中心 Y (m)", False),
            ParamSpec("scan_width_m", "number", 10e-9, "测试扫描宽 (m)", False),
        ],
        success_when="target_reached",
        fail_message="'target quality not reached after ' + str(attempt) + ' attempts'",
        nodes=[
            {"type": "set", "id": "i0", "var": "attempt", "value": "0"},
            {"type": "set", "id": "q0", "var": "quality", "value": "0.0"},
            {"type": "set", "id": "t0", "var": "target_reached", "value": "False"},
            {"type": "loop", "id": "loop", "mode": "while",
             "cond": "not target_reached and attempt < max_attempts", "var": "it",
             "max_iter": 50, "body": [
                {"type": "step", "id": "pulse", "skill": "TipPulse", "params": {
                    "pulse_v": {"$expr": "pulse_v"}, "duration_s": 0.1, "count": 1}},
                {"type": "step", "id": "configure", "skill": "ConfigureScan", "params": {
                    "center_x_m": {"$expr": "center_x_m"}, "center_y_m": {"$expr": "center_y_m"},
                    "width_m": {"$expr": "scan_width_m"}, "height_m": {"$expr": "scan_width_m"}}},
                {"type": "step", "id": "start", "skill": "StartScan", "params": {}},
                {"type": "step", "id": "wait", "skill": "WaitScanComplete",
                 "params": {"timeout_ms": 60000}},
                {"type": "step", "id": "assess", "skill": "AssessImageQuality",
                 "optional": True, "params": {}},
                # 没扫完的帧不许产出一个质量分:那个分数会被下一行拿去和
                # target_quality 比,一次被中途停下的扫描于是可能**提前结束修针循环**
                # (假成功),或者白烧一轮。沿用本行既有的写法 —— assess 失败时记 0.0,
                # 帧没扫完时同样记 0.0,让这一轮如实算作没达标。
                {"type": "set", "id": "uq", "var": "quality",
                 "value": ("assess['fft_quality'] if ('_failed' not in assess and "
                           + _wait_finished() + ") else 0.0")},
                {"type": "set", "id": "ut", "var": "target_reached",
                 "value": "quality >= target_quality"},
                {"type": "set", "id": "ua", "var": "attempt", "value": "attempt + 1"},
             ]},
        ],
        outputs=[{"name": "attempts", "expr": "attempt"},
                 {"name": "final_quality", "expr": "quality"}],
        tags=["tip", "conditioning", "builtin"],
        notes="Declarative twin of condition_tip.py:ConditionTip (P5).",
    )


def _demo_scan_and_sts() -> CompositeSpec:
    """DEMO: scan + save + multi-point STS, skipping STS where the move failed."""
    return CompositeSpec(
        name="DemoScanAndSTS",
        description="演示：扫描+存盘+多点 STS；某点移动失败则跳过该点采谱（_failed 标记）。",
        safety_level="auto",
        params=[
            ParamSpec("center_x_m", "number", 0.0, "中心 X (m)", False),
            ParamSpec("center_y_m", "number", 0.0, "中心 Y (m)", False),
            ParamSpec("scan_size_m", "number", 50e-9, "扫描尺寸 (m)", False),
            ParamSpec("sts_count", "int", 5, "STS 点数", False),
            ParamSpec("sts_start_v", "number", -1.0, "STS 起始 (V)", False),
            ParamSpec("sts_end_v", "number", 1.0, "STS 终止 (V)", False),
            ParamSpec("sts_num_points", "int", 100, "STS 点数", False),
            ParamSpec("scan_timeout_s", "number", 180.0, "扫描超时 (s)", False),
        ],
        nodes=[
            {"type": "step", "id": "configure_scan", "skill": "ConfigureScan", "params": {
                "center_x_m": {"$expr": "center_x_m"}, "center_y_m": {"$expr": "center_y_m"},
                "width_m": {"$expr": "scan_size_m"}, "height_m": {"$expr": "scan_size_m"}}},
            {"type": "step", "id": "start", "skill": "StartScan", "params": {}},
            {"type": "step", "id": "wait", "skill": "WaitScanComplete",
             "params": {"timeout_ms": {"$expr": "scan_timeout_s * 1000"}}},
            {"type": "step", "id": "save", "skill": "SaveScan", "optional": True, "params": {}},
            {"type": "step", "id": "configure_sts", "skill": "ConfigureSTS", "params": {
                "start_v": {"$expr": "sts_start_v"}, "end_v": {"$expr": "sts_end_v"},
                "num_points": {"$expr": "sts_num_points"}}},
            {"type": "loop", "id": "pts", "mode": "repeat", "count": "sts_count", "var": "i", "body": [
                {"type": "step", "id": "move", "skill": "MoveToXY", "optional": True, "params": {
                    "x_m": {"$expr": "center_x_m + (i - (sts_count - 1) / 2) * scan_size_m / sts_count"},
                    "y_m": {"$expr": "center_y_m"}, "wait": True}},
                {"type": "if", "id": "moved", "cond": "'_failed' not in move",
                 "then": [{"type": "step", "id": "sts", "skill": "AcquireSTS",
                           "optional": True, "params": {}}], "else": []},
            ]},
        ],
        # 与 Python 孪生同一条策略:**如实记录,不中止 STS**。
        # STS 点由几何生成、不取自图像,所以截断的帧不使任何一条谱失效;
        # 而这个技能的存在理由就是「演示时保证跑完的序列」。
        # 但帧的真相必须出得来 —— 否则一次没扫完的扫描在结果里和扫完的一模一样。
        outputs=[{"name": "scan_completed", "expr": _wait_finished()}],
        tags=["demo", "scan", "sts", "builtin"],
        notes="Declarative twin of demo_scan_and_sts.py:DemoScanAndSTS (P5).",
    )


def _shape_tip_on_surface() -> CompositeSpec:
    """Crown-jewel: wide scan → find flat → progressive plunge (break on contact)
    → cluster scan → roundness (break on round) → retry; ALWAYS restore feedback
    (try/finally)."""
    return CompositeSpec(
        name="ShapeTipOnSurface",
        description=("样品上整针：宽扫→选平点→渐进扎针(接触即停)→扫坑→评圆度(够圆即收)"
                     "→不行换点重试；无论成败都恢复 Z 反馈(try/finally)。"),
        safety_level="confirm",
        params=[
            ParamSpec("wide_scan_width_m", "number", 1e-7, "宽扫宽 (m)", False),
            ParamSpec("cluster_window_m", "number", 5e-9, "坑区窗 (m)", False),
            ParamSpec("shallow_depth_m", "number", -5e-10, "最浅扎入 (m)", False),
            ParamSpec("deep_depth_m", "number", -3e-9, "最深扎入 (m)", False),
            ParamSpec("n_depth_steps", "int", 5, "深度步数", False),
            ParamSpec("contact_threshold_a", "number", 5e-8, "接触电流阈 (A)", False),
            ParamSpec("min_axis_ratio", "number", 0.75,
                      "圆度阈(等效轴比;旧 round_threshold=0.65 已作废,不可换算)", False),
            ParamSpec("max_attempts", "int", 3, "最大尝试", False),
        ],
        success_when="accepted",
        fail_message="'no round cluster after ' + str(attempt) + ' attempts'",
        nodes=[
            {"type": "set", "id": "a0", "var": "accepted", "value": "False"},
            {"type": "set", "id": "at0", "var": "attempt", "value": "0"},
            {"type": "try",
             "body": [
                {"type": "loop", "id": "attempts", "mode": "while",
                 "cond": "not accepted and attempt < max_attempts", "var": "att",
                 "max_iter": 10, "body": [
                    {"type": "set", "id": "ua", "var": "attempt", "value": "attempt + 1"},
                    # wide scan to find a clean spot
                    {"type": "step", "id": "wide_configure", "skill": "ConfigureScan", "params": {
                        "center_x_m": 0.0, "center_y_m": 0.0,
                        "width_m": {"$expr": "wide_scan_width_m"},
                        "height_m": {"$expr": "wide_scan_width_m"}}},
                    {"type": "step", "id": "wide_start", "skill": "StartScan", "params": {}},
                    {"type": "step", "id": "wide_wait", "skill": "WaitScanComplete",
                     "params": {"timeout_ms": 300000}},
                    # ⚠️ 这一支**故意是 fail,而且故意中止整个重试循环**。
                    # 宽扫是用来选一个安全落点的:`FindFlatRegion` 在一片没扫完的
                    # 帧上照样会返回一个「平坦区」,而下一步是**把针扎进去**。
                    # 拿一张不完整的图去选扎针点,是这条链上代价最高的错误 ——
                    # 宁可整条停下让人来看,也不要换个点接着扎。
                    # (try/finally 仍会恢复 Z 反馈,所以中止是安全的。)
                    {"type": "if", "id": "wide_incomplete",
                     "cond": "not (" + _wait_finished("wide_wait") + ")",
                     "then": [{"type": "fail",
                               "reason": "'wide scan did not finish — refusing to "
                                         "pick a plunge site from an incomplete frame'"}],
                     "else": []},
                    {"type": "step", "id": "wide_save", "skill": "SaveScan", "optional": True, "params": {}},
                    {"type": "step", "id": "wide", "skill": "GetLatestScanFile", "params": {}},
                    {"type": "step", "id": "flat", "skill": "FindFlatRegion", "params": {
                        "scan_path": {"$expr": "wide['path']"},
                        "min_separation_m": {"$expr": "cluster_window_m"}}},
                    # progressive plunge: break out the moment we make contact
                    {"type": "set", "id": "c0", "var": "contact", "value": "False"},
                    {"type": "loop", "id": "plunge", "mode": "repeat",
                     "count": "n_depth_steps", "var": "i", "body": [
                        {"type": "set", "id": "depth", "var": "d",
                         "value": "shallow_depth_m + (deep_depth_m - shallow_depth_m) * i / max(1, n_depth_steps - 1)"},
                        {"type": "step", "id": "shape", "skill": "TipShape", "optional": True, "params": {
                            "tip_lift_m": {"$expr": "d"}, "lift_height_m": {"$expr": "0 - d"},
                            "change_bias": False, "restore_feedback": True}},
                        {"type": "step", "id": "mon", "skill": "MonitorCurrent", "optional": True, "params": {
                            "duration_s": 0.3, "poll_hz": 100,
                            "contact_threshold_a": {"$expr": "contact_threshold_a"},
                            "min_contact_samples": 3}},
                        {"type": "if", "id": "hit",
                         "cond": "'_failed' not in mon and mon['contact_detected']",
                         "then": [{"type": "set", "id": "sc", "var": "contact", "value": "True"},
                                  {"type": "break"}], "else": []},
                     ]},
                    {"type": "if", "id": "nocontact", "cond": "not contact",
                     "then": [{"type": "continue"}], "else": []},
                    # cluster scan + roundness assessment
                    {"type": "step", "id": "cluster_configure", "skill": "ConfigureScan", "params": {
                        "center_x_m": {"$expr": "flat['center_x_m']"},
                        "center_y_m": {"$expr": "flat['center_y_m']"},
                        "width_m": {"$expr": "cluster_window_m"},
                        "height_m": {"$expr": "cluster_window_m"}}},
                    {"type": "step", "id": "cluster_start", "skill": "StartScan", "optional": True, "params": {}},
                    {"type": "step", "id": "cluster_wait", "skill": "WaitScanComplete",
                     "optional": True, "params": {"timeout_ms": 120000}},
                    {"type": "step", "id": "clpath", "skill": "GetLatestScanFile", "optional": True, "params": {}},
                    # 显式选择 bright 极性，避免背景在自动极性分割中被误选。
                    {"type": "step", "id": "round", "skill": "AssessClusterRoundness", "optional": True, "params": {
                        "scan_path": {"$expr": "clpath['path']"},
                        "min_axis_ratio": {"$expr": "min_axis_ratio"},
                        "polarity": "bright"}},
                    # 坑扫没扫完就不许判「够圆了」—— 那会让一次被中途停下的扫描
                    # 直接 break 出重试循环,把没验证过的针尖当成整好了。
                    # 这一支不 fail:坑扫本来就是 optional,判不了就换个点再来,
                    # 与本循环既有的策略一致(宽扫那一支不同,理由见上)。
                    {"type": "if", "id": "isround",
                     "cond": ("'_failed' not in round and round['is_round'] and "
                              + _wait_finished("cluster_wait")),
                     "then": [{"type": "set", "id": "acc", "var": "accepted", "value": "True"},
                              {"type": "break"}], "else": []},
                 ]},
             ],
             "finally": [
                {"type": "step", "id": "restore_fb", "skill": "ZControllerOnOff",
                 "optional": True, "params": {"enable": True}},
             ]},
        ],
        outputs=[{"name": "accepted", "expr": "accepted"},
                 {"name": "attempts", "expr": "attempt"}],
        tags=["tip", "shaping", "builtin"],
        notes="Declarative twin of shape_tip_on_surface.py:ShapeTipOnSurface (P5).",
    )


def _survey_surface() -> CompositeSpec:
    """grid_n × grid_n tile survey; partial-success; tracks the best-quality tile."""
    return CompositeSpec(
        name="SurveySurface_TileScan",
        description="把方形区域切成 grid_n×grid_n 瓦片逐块巡扫建全局概览；部分成功；可评质并记最佳瓦。",
        safety_level="confirm",
        params=[
            ParamSpec("center_x_m", "number", 0.0, "中心 X (m)", False),
            ParamSpec("center_y_m", "number", 0.0, "中心 Y (m)", False),
            ParamSpec("grid_n", "int", 3, "每边瓦片数 (N×N)", False),
            ParamSpec("tile_size_m", "number", 50e-9, "瓦片边长 (m)", False),
            # ⚠️ 0.1 与 Python 真身**不一致,而且现在改不了**(2026-08-25 查清)。
            #
            # 真身 full_scan.py:114 刻意写 `default=None`,注释说:写了 0.1,
            # 「没传」与「显式传 0.1」就永远分不开,分尺度档位表也就永远轮不到。
            #
            # 照着改成 None,**12 条测试当场报红**:
            #   TypeError: unsupported operand type(s) for /: float and NoneType
            #   (spec.py:116,来自下面的 {"$expr": "width_m / line_time_s"})
            # 孪生体的表达式引擎在**编排时**就要拿它做除法,而它**没有**真身
            # 那套「留空则查档位表」的解析 —— 同 wait_timeout_s 那条注释说的
            # 「静态 spec 读不到实时行数」是同一件事。
            #
            # ⇒ 这不是笔误,是**两条执行路径的能力差**。0.1 对 1 µm 的巡查图和
            #   5 nm 的原子分辨图给同一个速度(差一两个数量级),所以走 builder
            #   这条路的扫描仍然拿不到档位表的速度。
            #
            # **要对齐需要先做什么**:让 spec 的表达式层能解析「留空 → 查
            #   scan_policy.resolve_line_time(尺寸)」。那是功能改动,得有人拍板。
            ParamSpec("line_time_s", "number", 0.1, "每行时间 (s)", False),
            ParamSpec("channels", "string", "Z,Current", "采集通道", False),
            ParamSpec("wait_timeout_s", "number", 180.0, "每瓦等扫超时 (s)", False),
            ParamSpec("assess_quality", "bool", True, "是否逐瓦评质", False),
        ],
        success_when="scanned >= 1",
        fail_message="'no tile scanned'",
        nodes=[
            {"type": "set", "id": "s0", "var": "scanned", "value": "0"},
            {"type": "set", "id": "bq", "var": "best_q", "value": "-1.0"},
            {"type": "set", "id": "bt", "var": "best_tile", "value": "-1"},
            {"type": "set", "id": "ti", "var": "tile_idx", "value": "-1"},
            {"type": "loop", "id": "rows", "mode": "repeat", "count": "grid_n", "var": "row", "body": [
                {"type": "loop", "id": "cols", "mode": "repeat", "count": "grid_n", "var": "col", "body": [
                    {"type": "set", "id": "inc_idx", "var": "tile_idx", "value": "tile_idx + 1"},
                    {"type": "step", "id": "configure", "skill": "ConfigureScan", "optional": True, "params": {
                        "center_x_m": {"$expr": "center_x_m + (col - (grid_n - 1) / 2) * tile_size_m"},
                        "center_y_m": {"$expr": "center_y_m + (row - (grid_n - 1) / 2) * tile_size_m"},
                        "width_m": {"$expr": "tile_size_m"}, "height_m": {"$expr": "tile_size_m"},
                        "channels": {"$expr": "channels"}}},
                    {"type": "step", "id": "speed", "skill": "SetScanSpeed", "optional": True, "params": {
                        "fwd_speed": {"$expr": "tile_size_m / line_time_s"},
                        "bwd_speed": {"$expr": "tile_size_m / line_time_s"},
                        "fwd_line_time": {"$expr": "line_time_s"},
                        "bwd_line_time": {"$expr": "line_time_s"}, "keep_const": 0}},
                    {"type": "step", "id": "start", "skill": "StartScan", "optional": True, "params": {}},
                    {"type": "step", "id": "wait", "skill": "WaitScanComplete", "optional": True,
                     "params": {"timeout_ms": {"$expr": "wait_timeout_s * 1000"}}},
                    # 与 Python 孪生同一条策略:**只判这一格没扫成,整批继续**。
                    # 巡查的价值就在于一格出问题不影响其余;而且这里刻意**不 fail** ——
                    # spec 里的 fail 会中止整个循环,正好与孪生的「只判这一格」相反。
                    {"type": "if", "id": "tile_ok", "cond": _wait_finished(), "then": [
                        {"type": "set", "id": "inc", "var": "scanned", "value": "scanned + 1"},
                        {"type": "if", "id": "do_assess", "cond": "assess_quality", "then": [
                            {"type": "step", "id": "assess", "skill": "AssessImageQuality", "optional": True, "params": {}},
                            {"type": "if", "id": "better",
                             "cond": "'_failed' not in assess and assess['fft_quality'] > best_q", "then": [
                                {"type": "set", "id": "ub", "var": "best_q", "value": "assess['fft_quality']"},
                                {"type": "set", "id": "ubt", "var": "best_tile", "value": "tile_idx"},
                             ], "else": []},
                        ], "else": []},
                    ], "else": []},
                ]},
            ]},
        ],
        outputs=[{"name": "scanned", "expr": "scanned"},
                 {"name": "recommended_tile", "expr": "best_tile"},
                 {"name": "best_quality", "expr": "best_q"}],
        tags=["scan", "survey", "builtin"],
        notes="Declarative twin of survey_surface.py:SurveySurface_TileScan (P5).",
    )


def _batch_regions() -> CompositeSpec:
    """ParseRegions → foreach region: configure(angle)/speed/start/wait/(save); partial-success."""
    return CompositeSpec(
        name="BatchRegionsScan",
        description="按 JSON 区域清单逐个扫（每区可不同角度），部分成功。",
        safety_level="confirm",
        params=[
            ParamSpec("regions", "string", "[]", "区域清单 JSON 数组", True),
            ParamSpec("channels", "string", "Z,Current", "采集通道", False),
            # ⚠️ 0.1 与 Python 真身**不一致,而且现在改不了**(2026-08-25 查清)。
            #
            # 真身 full_scan.py:114 刻意写 `default=None`,注释说:写了 0.1,
            # 「没传」与「显式传 0.1」就永远分不开,分尺度档位表也就永远轮不到。
            #
            # 照着改成 None,**12 条测试当场报红**:
            #   TypeError: unsupported operand type(s) for /: float and NoneType
            #   (spec.py:116,来自下面的 {"$expr": "width_m / line_time_s"})
            # 孪生体的表达式引擎在**编排时**就要拿它做除法,而它**没有**真身
            # 那套「留空则查档位表」的解析 —— 同 wait_timeout_s 那条注释说的
            # 「静态 spec 读不到实时行数」是同一件事。
            #
            # ⇒ 这不是笔误,是**两条执行路径的能力差**。0.1 对 1 µm 的巡查图和
            #   5 nm 的原子分辨图给同一个速度(差一两个数量级),所以走 builder
            #   这条路的扫描仍然拿不到档位表的速度。
            #
            # **要对齐需要先做什么**:让 spec 的表达式层能解析「留空 → 查
            #   scan_policy.resolve_line_time(尺寸)」。那是功能改动,得有人拍板。
            ParamSpec("line_time_s", "number", 0.1, "每行时间 (s)", False),
            ParamSpec("wait_timeout_s", "number", 180.0, "每区等扫超时 (s)", False),
            ParamSpec("save_each", "bool", True, "每区存盘", False),
        ],
        success_when="scanned >= 1",
        fail_message="'no region scanned'",
        nodes=[
            {"type": "step", "id": "parse", "skill": "ParseRegions",
             "params": {"regions": {"$expr": "regions"}}},
            {"type": "set", "id": "s0", "var": "scanned", "value": "0"},
            {"type": "loop", "id": "regs", "mode": "foreach", "var": "r",
             "iterable": "parse['regions']", "body": [
                {"type": "step", "id": "configure", "skill": "ConfigureScan", "optional": True, "params": {
                    "center_x_m": {"$expr": "r['center_x_m']"}, "center_y_m": {"$expr": "r['center_y_m']"},
                    "width_m": {"$expr": "r['width_m']"}, "height_m": {"$expr": "r['height_m']"},
                    "angle_deg": {"$expr": "r['angle_deg']"}, "channels": {"$expr": "channels"}}},
                {"type": "step", "id": "speed", "skill": "SetScanSpeed", "optional": True, "params": {
                    "fwd_speed": {"$expr": "max(r['width_m'], r['height_m']) / line_time_s"},
                    "bwd_speed": {"$expr": "max(r['width_m'], r['height_m']) / line_time_s"},
                    "fwd_line_time": {"$expr": "line_time_s"},
                    "bwd_line_time": {"$expr": "line_time_s"}, "keep_const": 0}},
                {"type": "step", "id": "start", "skill": "StartScan", "optional": True, "params": {}},
                {"type": "step", "id": "wait", "skill": "WaitScanComplete", "optional": True,
                 "params": {"timeout_ms": {"$expr": "wait_timeout_s * 1000"}}},
                # 同 SurveySurface:只判这一区没扫成,整批继续,不 fail。
                {"type": "if", "id": "reg_ok", "cond": _wait_finished(),
                 "then": [{"type": "set", "id": "inc", "var": "scanned", "value": "scanned + 1"}], "else": []},
                {"type": "if", "id": "do_save", "cond": "save_each",
                 "then": [{"type": "step", "id": "save", "skill": "SaveScan", "optional": True, "params": {}}],
                 "else": []},
             ]},
        ],
        outputs=[{"name": "scanned", "expr": "scanned"},
                 {"name": "region_count", "expr": "parse['count']"}],
        tags=["scan", "batch", "builtin"],
        notes="Declarative twin of batch_regions_scan.py:BatchRegionsScan (P5).",
    )


def _track_drift() -> CompositeSpec:
    """SetBias → reference FullScan → ComputeDriftVector → conditional frame shift.

    First call (no ref_image_path) just stores the reference and succeeds."""
    return CompositeSpec(
        name="TrackDrift_ReferenceScan",
        description="按参考图跟踪样品漂移并补偿扫描框；首次（无参考路径）则把本次当参考。",
        safety_level="confirm",
        params=[
            ParamSpec("ref_x_m", "number", 0.0, "参考点 X (m)", True),
            ParamSpec("ref_y_m", "number", 0.0, "参考点 Y (m)", True),
            ParamSpec("ref_width_m", "number", 20e-9, "参考扫描宽 (m)", False),
            ParamSpec("bias_v", "number", -0.5, "偏压 (V)", False),
            ParamSpec("ref_image_path", "string", "", "参考图 .npy 路径（空=存为参考）", False),
        ],
        nodes=[
            {"type": "step", "id": "set_bias", "skill": "SetBias", "optional": True,
             "params": {"bias_v": {"$expr": "bias_v"}}},
            {"type": "step", "id": "ref_scan", "skill": "FullScan", "params": {
                "center_x_m": {"$expr": "ref_x_m"}, "center_y_m": {"$expr": "ref_y_m"},
                "width_m": {"$expr": "ref_width_m"}, "height_m": {"$expr": "ref_width_m"},
                "line_time_s": 0.1}},
            {"type": "if", "id": "first", "cond": "ref_image_path == ''",
             "then": [{"type": "succeed", "reason": "'reference stored (no prior reference)'"}], "else": []},
            {"type": "step", "id": "drift", "skill": "ComputeDriftVector", "params": {
                "ref_path": {"$expr": "ref_image_path"}, "scan_width_m": {"$expr": "ref_width_m"}}},
            {"type": "if", "id": "shifted",
             "cond": "abs(drift['drift_x_m']) > 1e-12 or abs(drift['drift_y_m']) > 1e-12",
             "then": [
                {"type": "step", "id": "compensate", "skill": "ConfigureScan", "optional": True, "params": {
                    "center_x_m": {"$expr": "ref_x_m + drift['drift_x_m']"},
                    "center_y_m": {"$expr": "ref_y_m + drift['drift_y_m']"},
                    "width_m": {"$expr": "ref_width_m"}, "height_m": {"$expr": "ref_width_m"}}},
             ], "else": []},
        ],
        outputs=[{"name": "drift_x_m", "expr": "drift['drift_x_m']"},
                 {"name": "drift_y_m", "expr": "drift['drift_y_m']"}],
        tags=["scan", "drift", "builtin"],
        notes="Declarative twin of drift_track.py:TrackDrift_ReferenceScan (P5).",
    )


def _prescan_check() -> CompositeSpec:
    """Pre-scan and live-buffer acquisition without a tip-quality verdict.

    Buffer contents may not belong to the current frame and cannot be compared
    directly with saved .sxm data. Keep both verdict fields inconclusive.
    """
    return CompositeSpec(
        name="PreScanCheck",
        description=(
            "预扫描 + 抓实时缓冲区的正反扫。**它给不出针尖判决** —— 缓冲区里的"
            "数据与 .sxm 不可比、且未必是本帧,所以 tip_ready/similarity 恒为 None"
            "(判不了),另报 inconclusive_reason 说明为什么。要针尖判决请用手写的"
            "PreScanCheck(它先读存盘的 .sxm)。"),
        safety_level="confirm",
        params=[
            ParamSpec("center_x_m", "number", 0.0, "中心 X (m)", True),
            ParamSpec("center_y_m", "number", 0.0, "中心 Y (m)", True),
            ParamSpec("width_m", "number", 1e-7, "宽 (m，与完整扫描同宽)", True),
            # 保留旧参数仅为调用签名兼容，不把旧余弦阈值传给新相关性判据。
            # 没有显式 correlation_threshold 时保持无法判定，不生成合格结论。
            ParamSpec("quality_threshold", "number", 0.8,
                      "【已不再接线】旧余弦阈,保留仅为兼容签名", False),
            # 声明式流程无法查询仪器档位表，必须由调用者显式提供每线时间。
            # 不设置默认值，避免把未提供的参数解释成已经确认的扫描设置。
            ParamSpec("line_time_s", "number", None,
                      "每线时间(s)，必须显式提供。请根据当前仪器配置与扫描范围设置；"
                      "声明式流程无法查询档位表，不提供默认值。", True),
        ],
        nodes=[
            # 缺失或为零的每线时间必须在任何硬件调用前拒绝。
            # safe_eval 不支持 is；此处的布尔检查同时拒绝 None 和零。
            {"type": "if", "id": "need_line_time", "cond": "not line_time_s", "then": [
                {"type": "fail", "reason":
                    "'PreScanCheck(声明式)必须显式给 line_time_s。"
                    "请根据当前仪器配置与扫描范围提供每线时间；此流程无法查询档位表。'"},
            ], "else": []},
            {"type": "step", "id": "configure", "skill": "ConfigureScan", "params": {
                "center_x_m": {"$expr": "center_x_m"}, "center_y_m": {"$expr": "center_y_m"},
                "width_m": {"$expr": "width_m"},
                # 使用方形扫描区域。帧时由行数与每线时间决定，无需压缩高度。
                "height_m": {"$expr": "width_m"}, "set_scan_speed": False}},
            # 扫描速度和每线时间由同一个输入派生，保持正反向设置一致。
            {"type": "step", "id": "speed", "skill": "SetScanSpeed", "optional": True, "params": {
                "fwd_speed": {"$expr": "width_m / line_time_s"},
                "bwd_speed": {"$expr": "width_m / line_time_s"},
                "fwd_line_time": {"$expr": "line_time_s"},
                "bwd_line_time": {"$expr": "line_time_s"}, "keep_const": 0}},
            {"type": "step", "id": "start", "skill": "StartScan", "params": {}},
            {"type": "step", "id": "wait", "skill": "WaitScanComplete", "params": {"timeout_ms": 15000}},
            # 两条 cond 都带 `in` 守卫,理由与取舍范围见 FullScan 那处的注释。
            {"type": "if", "id": "timed_out",
             "cond": "'timed_out' in wait and wait['timed_out']", "then": [
                {"type": "step", "id": "stop", "skill": "StopScan", "optional": True, "params": {}},
                {"type": "fail", "reason": "'Pre-scan timed out'"},
            ], "else": []},
            # ⚠️ 这一支**没有** StopScan,与上面那支刻意不同:走到这里说明
            # Scan_StatusGet 已经读到 0,扫描本来就停了。再发一次是对着已停的
            # 扫描做一次无意义的硬件写。对称好看,但不对。
            {"type": "if", "id": "stopped_early",
             "cond": "'stopped_early' in wait and wait['stopped_early']",
             "then": [{"type": "fail",
                       "reason": "'Pre-scan stopped early — the line was never "
                                 "finished, so it measures nothing'"}],
             "else": []},
            # None means no comparable measurement; False would claim a failed
            # tip-quality assessment and could incorrectly suggest a repair.
            {"type": "set", "id": "tr0", "var": "tip_ready", "value": "None"},
            {"type": "set", "id": "sim0", "var": "similarity", "value": "None"},
            {"type": "set", "id": "why0", "var": "inconclusive_reason",
             "value": "'预扫描还没走到取数那一步'"},
            {"type": "step", "id": "fwd", "skill": "GrabScanFrameData", "optional": True,
             "params": {"channel_index": 0, "direction": 1}},
            {"type": "step", "id": "bwd", "skill": "GrabScanFrameData", "optional": True,
             "params": {"channel_index": 0, "direction": 0}},
            # 实时缓冲区未必对应当前帧，且不能与存盘文件的指标直接比较。
            # 不调用 CheckLineQuality；两个判定字段保持 None 并解释原因。
            # 保留可编辑模板供 builder 使用，行为与手写流程的缓冲回落一致。
            {"type": "if", "id": "have_lines",
             "cond": "'_failed' not in fwd and '_failed' not in bwd", "then": [
                {"type": "set", "id": "why_buf", "var": "inconclusive_reason",
                 "value": "'判不了：缓冲路径相似度与 .sxm 路径不可比,且缓冲内容未必是本帧。"
                          "这是弃权,不是「针尖不合格」:下一步是去 .sxm 里取这一帧 / "
                          "换个地方重新量,不是修针。'"},
             ], "else": [
                {"type": "set", "id": "why_read", "var": "inconclusive_reason",
                 "value": "'实时缓冲区取不回正反扫数据(GrabScanFrameData 失败)"
                          "—— 通信/模块问题,不是针尖问题。'"},
             ]},
        ],
        outputs=[{"name": "tip_ready", "expr": "tip_ready"},
                 {"name": "similarity", "expr": "similarity"},
                 # 判不了的时候**必须说得出为什么** —— 一个静默的 None 与
                 # 「测了,针尖没问题」在回包里长得一样。
                 {"name": "inconclusive_reason", "expr": "inconclusive_reason"}],
        tags=["scan", "prescan", "tip", "builtin"],
        notes="Declarative twin of prescan_check.py:PreScanCheck (P5).",
    )


def reconstructed_composites() -> list[CompositeSpec]:
    """The built-in workflows reconstructed as declarative specs (P5)."""
    return [
        _full_scan(),
        _tip_pulse(),
        _grid_sts(),
        _condition_tip(),
        _demo_scan_and_sts(),
        _shape_tip_on_surface(),
        _survey_surface(),
        _batch_regions(),
        _track_drift(),
        _prescan_check(),
    ]


#: 出厂内容指纹的写法 —— 与 ``version_store.save`` 里那一份**逐字同一个配方**
#: (``to_dict()`` 去掉 version 之后按键排序的规范 JSON 的 sha256)。
#: 两份配方一旦分家,守卫就会把每一条都判成「被改过」而永不重播,**而且不报错**。
KEY_FACTORY_MARKER = "_factory_sha256"


def factory_content_hash(spec: "CompositeSpec") -> str:
    """这份 spec 的内容指纹(不含版本号)。

    版本号被排除是有意的:同样的内容在不同机器上可能是 v1 也可能是 v7
    (谁先 seed、之间存过几次)。守卫问的是「内容变没变」,不是「存过几次」。
    """
    import hashlib
    import json as _json

    d = spec.to_dict()
    d.pop("version", None)
    return hashlib.sha256(
        _json.dumps(d, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


#: **迁移用**:标记出现之前发出去的那些出厂内容的指纹。
#:
#: ## 为什么需要这张表(以及它什么时候可以删)
#:
#: 守卫要回答的是「库里这份是我们发的原样,还是用户改过的?」。2026-08-14 起
#: 每次 seed/重播都会随手写下 ``_factory_sha256`` 标记,之后这个问题**自证**:
#: 标记 == 当前内容指纹 ⇒ 没人动过。但**已经装在机器上的那些条目没有标记** ——
#: 而它们恰恰是这次要修的对象(细条几何 + 0.1 s/线的刮针速度)。
#: 所以这里冻结历史各代出厂内容的指纹,只为覆盖「标记之前」那一段。
#:
#: ⇒ **等所有在役安装都经过一次带标记的发布之后,这张表就可以整个删掉。**
#: 它是一次性的迁移垫片,不是永久的维护税。
#:
#: 重新生成(改了某个孪生体、需要把它的上一代指纹加进来时)::
#:
#:     git show <上一个提交>:MASTv2/mast/skills/composite/builtin_composites.py > /tmp/old.py
#:     # 导入 old.py,对目标 spec 调 factory_content_hash()
#:
#: 下面这 6 个是 ``PreScanCheck`` 在 git 历史里出现过的**全部**出厂代
#: (2026-08-14 遍历该文件的 10 次提交算出,去重后 6 代)。不在表里的指纹一律
#: 按「被改过」处理 —— **保守方向**:漏修一台机器,总好过盖掉用户的编辑。
_LEGACY_FACTORY_HASHES: dict[str, frozenset[str]] = {
    "PreScanCheck": frozenset({
        "6eaebcf237a1a44900700bf762803250109a955757e7efd1a21bedf2defd3b8b",
        "5e4bce3151b42e6634313f62a7298213fa4449abb06296619dfeb9ad48ff3982",
        "b2bb77d35a2464a7f97d1f19bbd0f6f85deb142ba7e6d63c09bc4dfa77d12dba",
        "ca113c4ec04162abfc1521a16b8f087a76d77a210a9142f93145cff3728894bf",
        "1b50c9b6c06a9d0b40079f01f109cfcb96022c5864eba96d345bda47306c2e1f",
        "a92a7219c19ecfda011cd0954fc536cacae0f0312e465579c4ea7ef492ab65af",
    }),
}


def _stored_content_hash(store, name: str) -> "str | None":
    """库里那一份的内容指纹。``None`` = **读不出来**(不是「不一样」)。

    重算而不是直接信文件里的 ``_content_sha256``:那个键是后加的,更老的库里
    根本没有。走一遍 ``from_dict → to_dict`` 与出厂那份同口径,才比得了。
    """
    try:
        return factory_content_hash(store.load(name))
    except Exception:  # noqa: BLE001 — 读不出来就当身世不明,下面会「不碰」
        return None


def seed_builtin_composites(store) -> list[str]:
    """把出厂孪生体写进 *store*;**没被用户改过的旧条目会被重播成新版**。

    返回**写进去了的名字**(新 seed 的 + 重播更新的)。三条路,一条都不许静默:

    ====================  ==========================================
    库里那一份            动作
    ====================  ==========================================
    不存在                写入(首次 seed)
    == 当前出厂内容       什么都不做(已经是最新)
    == 某一代出厂内容     **重播**:替换成新版 + INFO 一行
    其它(用户改过)     **不碰** + INFO 一行说明为什么没动它
    ====================  ==========================================

    ## 为什么原来的 if-not-exists 是个洞

    仅在条目不存在时写入，会使已经生成的模板永远收不到更新。
    因此需要识别未被用户编辑的出厂模板，并把它们替换为当前版本。

    ## 为什么不能无条件覆盖

    库里这些条目**是可编辑的** —— builder 的整个用途就是让用户改/fork 它们。
    无条件重播会把人家的编辑冲掉,而且是静默的。所以先问「这份还是我们发的原样
    吗」,只有答案是「是」才替换。指纹对不上就当作被改过 —— **保守方向**:
    漏修一台,好过毁掉一份人写的东西。

    重播本身也不是破坏性的:``store.save`` 会把旧版归档进 history,用户随时
    ``restore``。日志里那一行会把版本号说出来。

    Best-effort:某一条炸了只记一条 warning,绝不让整个 seeding 失败(它跑在
    ``composite_store()`` 的首次初始化路径上,抛出去会连带整个面板打不开)。
    """
    import logging
    logger = logging.getLogger(__name__)
    written: list[str] = []
    for spec in reconstructed_composites():
        try:
            if not store.exists(spec.name):
                store.save(spec, extra_meta={
                    KEY_FACTORY_MARKER: factory_content_hash(spec)})
                written.append(spec.name)
                continue

            current = factory_content_hash(spec)
            stored = _stored_content_hash(store, spec.name)
            if stored == current:
                continue                      # 已经是最新的出厂内容

            # 身世判定。标记那一路自证(标记 == 库里内容 ⇒ 我们写完没人动过);
            # 标记不存在的老条目退回冻结的历史指纹表。
            # ⚠️ `load_meta` 是 2026-08-14 新加的:鸭子类型的替身 store 可能没有,
            # 拿不到就当**没有标记**(而不是当成「没改过」)。
            meta = {}
            if callable(getattr(store, "load_meta", None)):
                try:
                    meta = store.load_meta(spec.name) or {}
                except Exception:  # noqa: BLE001
                    meta = {}
            marker = meta.get(KEY_FACTORY_MARKER)
            untouched = (
                (marker is not None and stored is not None and marker == stored)
                or (stored is not None
                    and stored in _LEGACY_FACTORY_HASHES.get(spec.name, frozenset()))
            )

            if untouched:
                saved = store.save(spec, extra_meta={
                    KEY_FACTORY_MARKER: current})
                written.append(spec.name)
                logger.info(
                    "builtin composite %s: 库里那份是未经改动的出厂副本,已重播为"
                    " v%s(旧版已归档,可 restore)", spec.name,
                    getattr(saved, "version", "?"))
            else:
                logger.info(
                    "builtin composite %s: 库里那份与任何一代出厂内容都对不上"
                    "(指纹 %s)—— 当作**用户改过**,不动它。"
                    "要拿新版请在 builder 里另存/克隆后对照。",
                    spec.name, (stored or "读不出来")[:12])
        except Exception as exc:  # pragma: no cover - best-effort seeding
            logger.warning("seed_builtin_composites: %s failed: %s", spec.name, exc)
    if written:
        logger.info("builtin composites written: %s", ", ".join(written))
    return written


__all__ = ["reconstructed_composites", "seed_builtin_composites",
           "factory_content_hash", "KEY_FACTORY_MARKER"]
