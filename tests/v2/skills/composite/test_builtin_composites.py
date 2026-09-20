"""P3: the reconstructed built-in composites behave like their Python twins.

Each test drives the declarative spec against a FakeCtx and asserts the
sub-skill call sequence + verdict — proving the extended IR (try/finally,
break/continue, succeed/fail, success_when, the `_failed` marker, the G1
CheckScanForCrash skill) faithfully expresses the hand-written composites.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest tests/v2/skills/composite/test_builtin_composites.py -x -v
"""
from __future__ import annotations

# ── path setup BEFORE any mast.* imports ──
import sys
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[4] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _name in list(sys.modules):
    if _name == "mast" or _name.startswith("mast."):
        _f = getattr(sys.modules[_name], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_name]

import pytest

from mast.skills.composite.builtin_composites import reconstructed_composites


class _Res:
    def __init__(self, success=True, data=None, error=""):
        self.success = success
        self.data = data or {}
        self.error = error
        self.nanonis_calls = []


class FakeCtx:
    def __init__(self, results=None):
        self.calls = []
        self._results = results or {}

    def run(self, skill_name, params):
        self.calls.append((skill_name, dict(params)))
        r = self._results.get(skill_name)
        if callable(r):
            return r(params)
        return r or _Res(success=True, data={})


def _run(name, params, ctx):
    from mast.skills.composite.interpreter import SpecComposite
    spec = next(s for s in reconstructed_composites() if s.name == name)
    return SpecComposite(spec).execute(ctx, params)


def _skills(ctx):
    return [c[0] for c in ctx.calls]


# ── all specs are structurally valid ────────────────────────────────────
def test_all_reconstructed_specs_validate():
    for spec in reconstructed_composites():
        assert spec.validate() == [], f"{spec.name}: {spec.validate()}"


# ── FullScan ────────────────────────────────────────────────────────────
class TestFullScan:
    def test_clean_scan(self):
        ctx = FakeCtx({
            "WaitScanComplete": _Res(data={"timed_out": False}),
            "CheckScanForCrash": _Res(data={"crash_indicator": False, "status": "ok"}),
        })
        res = _run("FullScan", {}, ctx)
        assert res.success
        assert _skills(ctx) == ["ConfigureScan", "SetScanSpeed", "StartScan",
                                "WaitScanComplete", "CheckScanForCrash"]

    def test_crash_fails(self):
        ctx = FakeCtx({
            "WaitScanComplete": _Res(data={"timed_out": False}),
            "CheckScanForCrash": _Res(data={"crash_indicator": True, "status": "crash"}),
        })
        res = _run("FullScan", {}, ctx)
        assert res.success is False
        assert res.error == "CRASH_DETECTED: crash"

    def test_timeout_fails_before_crash_check(self):
        ctx = FakeCtx({"WaitScanComplete": _Res(data={"timed_out": True})})
        res = _run("FullScan", {}, ctx)
        assert res.success is False
        assert res.error == "Scan timed out"
        assert "CheckScanForCrash" not in _skills(ctx)

    # ── stopped part-way (v6.1.3, KNOWN_ISSUES §2.24) ──────────────────

    def test_stopped_early_fails_before_crash_check(self):
        """Ordering is load-bearing: CheckScanForCrash reads the frame, and on a
        field of NaN it is not measuring a tip. The crash check must never be
        the thing that adjudicates a truncated scan."""
        ctx = FakeCtx({"WaitScanComplete": _Res(data={
            "timed_out": False, "stopped_early": True,
            "lines_done": 123, "lines_total": 512})})
        res = _run("FullScan", {}, ctx)
        assert res.success is False
        assert "stopped early" in res.error
        assert "123/512" in res.error
        assert "CheckScanForCrash" not in _skills(ctx)

    def test_a_wait_result_without_the_field_still_runs(self):
        """FAIL OPEN, and this is why the condition is guarded with `in` rather
        than a bare subscript: safe_eval raises ExprError on a missing key, which
        aborts the WHOLE composite. Any WaitScanComplete that does not report the
        field — an older build, a stub, a hand-written twin — would turn "one
        optional field is absent" into "the scan skill crashed".

        This test failed before the guard went in, on the pre-existing doubles."""
        ctx = FakeCtx({
            "WaitScanComplete": _Res(data={"timed_out": False}),   # v6.1.1 shape
            "CheckScanForCrash": _Res(data={"crash_indicator": False,
                                            "status": "ok"}),
        })
        res = _run("FullScan", {}, ctx)
        assert res.success
        assert "CheckScanForCrash" in _skills(ctx)

    def test_a_wait_result_with_NEITHER_field_still_runs(self):
        """The `timed_out` subscript next door was the same live mine, unexploded
        only because every double in the tree happens to supply that key. A wait
        implementation reporting neither flag must degrade to the pre-flag
        behaviour, not blow up the composite with an error that names nothing
        related to the actual cause."""
        ctx = FakeCtx({
            "WaitScanComplete": _Res(data={"polls": 1}),      # reports neither
            "CheckScanForCrash": _Res(data={"crash_indicator": False,
                                            "status": "ok"}),
        })
        res = _run("FullScan", {}, ctx)
        assert res.success
        assert "CheckScanForCrash" in _skills(ctx)


# ── TipPulse ────────────────────────────────────────────────────────────
def test_tip_pulse_count():
    ctx = FakeCtx({"GetBias": _Res(data={"bias_v": 0.5})})
    res = _run("TipPulse", {"pulse_v": 4.0, "count": 2}, ctx)
    assert res.success
    assert _skills(ctx) == ["GetBias", "BiasPulse", "BiasPulse"]
    assert res.data.get("outputs", {}).get("original_bias_v") == 0.5


# ── GridSTS ─────────────────────────────────────────────────────────────
class TestGridSTS:
    def test_partial_success(self):
        ctx = FakeCtx()  # all steps succeed
        res = _run("GridSTS", {"nx": 2, "ny": 1, "spacing_m": 1e-9}, ctx)
        assert res.success
        assert _skills(ctx) == ["ConfigureSTS", "MoveToXY", "AcquireSTS",
                                "MoveToXY", "AcquireSTS"]
        assert res.data["outputs"]["succeeded"] == 2

    def test_all_fail(self):
        ctx = FakeCtx({"AcquireSTS": _Res(success=False, error="no tunneling")})
        res = _run("GridSTS", {"nx": 2, "ny": 1, "spacing_m": 1e-9}, ctx)
        assert res.success is False
        assert res.error == "no STS point succeeded"


# ── ConditionTip ────────────────────────────────────────────────────────
class TestConditionTip:
    def test_reaches_target_first_attempt(self):
        ctx = FakeCtx({"AssessImageQuality": _Res(data={"fft_quality": 0.9})})
        res = _run("ConditionTip", {"target_quality": 0.3, "max_attempts": 5}, ctx)
        assert res.success
        assert _skills(ctx) == ["TipPulse", "ConfigureScan", "StartScan",
                                "WaitScanComplete", "AssessImageQuality"]
        assert res.data["outputs"]["attempts"] == 1

    def test_fails_after_max_attempts(self):
        ctx = FakeCtx({"AssessImageQuality": _Res(data={"fft_quality": 0.1})})
        res = _run("ConditionTip", {"target_quality": 0.3, "max_attempts": 2}, ctx)
        assert res.success is False
        assert "after 2 attempts" in res.error
        # 2 attempts × 5 steps each
        assert _skills(ctx).count("TipPulse") == 2
        assert _skills(ctx).count("AssessImageQuality") == 2


# ── DemoScanAndSTS — the `_failed` marker ───────────────────────────────
class TestDemoScanAndSTS:
    def test_sts_runs_when_move_ok(self):
        ctx = FakeCtx()
        _run("DemoScanAndSTS", {"sts_count": 1}, ctx)
        assert _skills(ctx) == ["ConfigureScan", "StartScan", "WaitScanComplete",
                                "SaveScan", "ConfigureSTS", "MoveToXY", "AcquireSTS"]

    def test_sts_skipped_when_move_fails(self):
        ctx = FakeCtx({"MoveToXY": _Res(success=False, error="out of range")})
        _run("DemoScanAndSTS", {"sts_count": 1}, ctx)
        assert "AcquireSTS" not in _skills(ctx)
        assert _skills(ctx)[-1] == "MoveToXY"


# ── ShapeTipOnSurface — try/finally + break + continue (crown jewel) ─────
class TestShapeTipOnSurface:
    def _ctx(self, contact, is_round):
        return FakeCtx({
            "GetLatestScanFile": _Res(data={"path": "x.sxm"}),
            "FindFlatRegion": _Res(data={"center_x_m": 0.0, "center_y_m": 0.0}),
            "MonitorCurrent": _Res(data={"contact_detected": contact}),
            "AssessClusterRoundness": _Res(data={"is_round": is_round}),
        })

    def test_accept_on_contact_and_round(self):
        ctx = self._ctx(contact=True, is_round=True)
        res = _run("ShapeTipOnSurface", {"max_attempts": 2, "n_depth_steps": 3}, ctx)
        assert res.success
        assert res.data["outputs"]["accepted"] is True
        # contact on first plunge → only one TipShape/MonitorCurrent pair
        assert _skills(ctx).count("TipShape") == 1
        # finally ALWAYS restores Z feedback
        assert "ZControllerOnOff" in _skills(ctx)

    def test_finally_runs_even_on_failure(self):
        # never makes contact → retries, then fails — but finally still runs.
        ctx = self._ctx(contact=False, is_round=False)
        res = _run("ShapeTipOnSurface", {"max_attempts": 1, "n_depth_steps": 1}, ctx)
        assert res.success is False
        assert "no round cluster" in res.error
        assert _skills(ctx)[-1] == "ZControllerOnOff"  # finally ran last


# ── SurveySurface_TileScan — grid loop + recommended-tile accumulator ────
class TestSurveySurface:
    def test_grid_partial_success(self):
        ctx = FakeCtx({
            "WaitScanComplete": _Res(data={"timed_out": False}),
            "AssessImageQuality": _Res(data={"fft_quality": 0.7}),
        })
        res = _run("SurveySurface_TileScan", {"grid_n": 2, "assess_quality": True}, ctx)
        assert res.success
        assert _skills(ctx).count("ConfigureScan") == 4  # 2×2 tiles
        assert res.data["outputs"]["scanned"] == 4
        assert res.data["outputs"]["recommended_tile"] == 0  # all equal → first


# ── BatchRegionsScan — ParseRegions + foreach ───────────────────────────
class TestBatchRegions:
    def test_foreach_regions(self):
        ctx = FakeCtx({
            "ParseRegions": _Res(data={"regions": [
                {"center_x_m": 0.0, "center_y_m": 0.0, "width_m": 5e-8, "height_m": 5e-8,
                 "angle_deg": 0.0, "label": "R1"},
                {"center_x_m": 1e-7, "center_y_m": 0.0, "width_m": 5e-8, "height_m": 5e-8,
                 "angle_deg": 0.0, "label": "R2"}], "count": 2}),
            "WaitScanComplete": _Res(data={"timed_out": False}),
        })
        res = _run("BatchRegionsScan", {"regions": "[]", "save_each": False}, ctx)
        assert res.success
        assert _skills(ctx).count("ConfigureScan") == 2
        assert res.data["outputs"]["scanned"] == 2
        assert "SaveScan" not in _skills(ctx)  # save_each=False


# ── TrackDrift_ReferenceScan — succeed-early + conditional compensate ────
class TestTrackDrift:
    def test_first_call_stores_reference(self):
        ctx = FakeCtx()
        res = _run("TrackDrift_ReferenceScan",
                   {"ref_x_m": 0.0, "ref_y_m": 0.0, "ref_image_path": ""}, ctx)
        assert res.success
        # `succeed` fires before the drift computation
        assert _skills(ctx) == ["SetBias", "FullScan"]
        assert "ComputeDriftVector" not in _skills(ctx)

    def test_compensates_on_drift(self):
        ctx = FakeCtx({"ComputeDriftVector": _Res(data={"drift_x_m": 2e-9, "drift_y_m": 0.0})})
        res = _run("TrackDrift_ReferenceScan",
                   {"ref_x_m": 0.0, "ref_y_m": 0.0, "ref_image_path": "ref.npy"}, ctx)
        assert res.success
        assert "ComputeDriftVector" in _skills(ctx)
        # the compensating ConfigureScan ran (drift > 1e-12)
        assert "ConfigureScan" in _skills(ctx)


# ── PreScanCheck — 扫 + 抓缓冲区,**不出判决**─────────────────────

# 显式提供每线时间，使测试实际走到被测分支，而非提前停在参数校验处。
# 下列数值仅为 mock 流程输入，不代表仪器配置或安全标定。
_PRESCAN_PARAMS = {"width_m": 1e-7, "line_time_s": 1.0}


class TestPreScanCheck:
    def test_it_never_claims_a_tip_verdict_off_the_live_buffer(self):
        """Live-buffer data cannot establish frame identity or a tip verdict.

        Both outputs must remain None, the reason must be explicit, and the
        workflow must not feed buffer data into CheckLineQuality.
        """
        ctx = FakeCtx({
            "WaitScanComplete": _Res(data={"timed_out": False}),
            "GrabScanFrameData": _Res(data={"frame_path": "x.npy"}),
        })
        res = _run("PreScanCheck", _PRESCAN_PARAMS,ctx)
        assert res.success
        out = res.data["outputs"]
        assert out["tip_ready"] is None, out
        assert out["similarity"] is None, out
        # 判不了必须说得出为什么:一个静默的 None 与「测了,没问题」长得一样。
        assert "判不了" in (out.get("inconclusive_reason") or ""), out
        # 而且**不许**再把缓冲数据喂给判据 —— 那是数从哪来的那一步。
        assert "CheckLineQuality" not in _skills(ctx)

    def test_a_failed_buffer_grab_says_so(self):
        """两种判不了要分得开:读不到 vs 读到了但这条路不可信。

        下一步不同(一个查通信/模块,一个去 .sxm 取这一帧),而它们在回包里
        原本会折叠成同一个 `tip_ready=None`。
        """
        ok = FakeCtx({
            "WaitScanComplete": _Res(data={"timed_out": False}),
            "GrabScanFrameData": _Res(data={"frame_path": "x.npy"}),
        })
        bad = FakeCtx({
            "WaitScanComplete": _Res(data={"timed_out": False}),
            "GrabScanFrameData": _Res(success=False, error="tcp reset"),
        })
        why_ok = _run("PreScanCheck", _PRESCAN_PARAMS,ok).data[
            "outputs"]["inconclusive_reason"]
        why_bad = _run("PreScanCheck", _PRESCAN_PARAMS,bad).data[
            "outputs"]["inconclusive_reason"]
        assert why_ok != why_bad, (
            f"两种判不了说了同一句话({why_ok!r})—— 下一步不同,不能共用一句")
        assert "缓冲区" in why_bad and "不是针尖问题" in why_bad

    def test_timeout_fails(self):
        ctx = FakeCtx({"WaitScanComplete": _Res(data={"timed_out": True})})
        res = _run("PreScanCheck", _PRESCAN_PARAMS,ctx)
        assert res.success is False
        assert "timed out" in res.error

    # ── stopped part-way (v6.1.3, KNOWN_ISSUES §2.24) ──────────────────

    def test_stopped_early_fails_without_claiming_a_tip_verdict(self):
        """A line that never finished measures nothing — and the worst outcome
        is not the failure, it is a tip_ready verdict read off a line that was
        never acquired."""
        ctx = FakeCtx({"WaitScanComplete": _Res(data={
            "timed_out": False, "stopped_early": True,
            "lines_done": 3, "lines_total": 512})})
        res = _run("PreScanCheck", _PRESCAN_PARAMS,ctx)
        assert res.success is False
        assert "stopped early" in res.error
        # 2026-08-14:原来盯的是 `CheckLineQuality`,而 修复项 把那一步整个删了
        # ⇒ 那条断言从此**永远成立**,一道不会开火的门。改盯还在的那一步。
        #
        # ⚠️ 同一个坑当天差点又踩一次:任务 #11 在最前面加了一道「必须给
        # line_time_s」的闸,而这些用例原本不传它 —— 整条会停在闸上,于是
        # 「没调 X」再次变成永真。所以先断言**它真的扫到了 wait 那一步**,
        # 再断言后面那一步没被调。否定式断言必须自带一条肯定式的证据。
        assert "WaitScanComplete" in _skills(ctx), (
            "整条没跑到等待就结束了 —— 那「没调 GrabScanFrameData」不证明任何事")
        assert "GrabScanFrameData" not in _skills(ctx)

    def test_stopped_early_does_not_stop_the_scan_again(self):
        """The timeout branch above runs StopScan; this one deliberately does
        not. Reaching it means Scan_StatusGet already read 0, so the scan is
        stopped — re-issuing it is a hardware write on a read-only conclusion,
        and the two branches differing is the point, not an oversight."""
        ctx = FakeCtx({"WaitScanComplete": _Res(data={
            "timed_out": False, "stopped_early": True,
            "lines_done": 3, "lines_total": 512})})
        _run("PreScanCheck", _PRESCAN_PARAMS,ctx)
        # 同上:先证明扫描真的起来了,「没有再停一次」才有意义。
        assert "StartScan" in _skills(ctx), (
            "扫描根本没起来 —— 那「没再发 StopScan」是废话")
        assert "StopScan" not in _skills(ctx)

    def test_a_wait_result_without_the_field_still_runs(self):
        """Fail open — same reasoning as the FullScan twin above."""
        ctx = FakeCtx({
            "WaitScanComplete": _Res(data={"timed_out": False}),   # v6.1.1 shape
            "GrabScanFrameData": _Res(data={"frame_path": "x.npy"}),
        })
        res = _run("PreScanCheck", _PRESCAN_PARAMS,ctx)
        assert res.success

    def test_a_wait_result_with_NEITHER_field_still_runs(self):
        """Same as the FullScan twin: the `timed_out` subscript was equally
        unguarded here."""
        ctx = FakeCtx({
            "WaitScanComplete": _Res(data={"polls": 1}),      # reports neither
            "GrabScanFrameData": _Res(data={"frame_path": "x.npy"}),
        })
        res = _run("PreScanCheck", _PRESCAN_PARAMS,ctx)
        assert res.success


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))


# ── 六个模板都要消费「中途停止」（v6.1.3, KNOWN_ISSUES §2.24 关闭条件）────
#
# 这些 spec 在生产里不执行（声明式孪生不遮蔽同名 builtin），但它们是用户在技能
# 构建器里派生新复合时的**模板** —— 照着派生出来的东西会继承缺口。
#
# ⚠️ 每条断言都落在**这个模板自己的行为**上（成功/失败、扫成了几格、输出里写了
# 什么），一条都不去问 data 里有没有那个键。「字段存在」不是「有人在读」。


#: 这些 spec 还会调别的技能；给一份足够完整的替身，免得测试挂在与本次无关的
#: 缺字段上（缺字段会抛 ExprError，那是另一个话题，已在别处钉住）。
_SPEC_DEFAULTS = {
    "AssessImageQuality": lambda p: _Res(data={"fft_quality": 0.9, "label": "ok",
                                               "snr_db": 10.0}),
    "GetScanFrame": lambda p: _Res(data={"angle_deg": 0.0, "center_x_m": 0.0,
                                         "center_y_m": 0.0, "width_m": 1e-7,
                                         "height_m": 1e-7}),
    "GetLatestScanFile": lambda p: _Res(data={"path": "x.sxm"}),
    "FindFlatRegion": lambda p: _Res(data={"center_x_m": 0.0, "center_y_m": 0.0}),
    "AssessClusterRoundness": lambda p: _Res(data={"is_round": True}),
    "CheckScanForCrash": lambda p: _Res(data={"crash_indicator": False,
                                              "status": "ok"}),
    "ParseRegions": lambda p: _Res(data={"count": 1, "regions": [
        {"center_x_m": 0.0, "center_y_m": 0.0, "width_m": 1e-7, "height_m": 1e-7,
         "angle_deg": 0.0, "label": "R1"}]}),
}


def _spec_ctx(**results):
    merged = dict(_SPEC_DEFAULTS)
    merged.update(results)
    return FakeCtx(merged)


def _stopped_early(**over):
    d = {"timed_out": False, "stopped_early": True,
         "lines_done": 77, "lines_total": 512}
    d.update(over)
    return _Res(data=d)


def _completed(**over):
    d = {"timed_out": False, "stopped_early": False,
         "lines_done": 512, "lines_total": 512}
    d.update(over)
    return _Res(data=d)


def test_condition_tip_does_not_score_quality_off_an_unfinished_frame():
    """没扫完的帧不许产出质量分:那个分数会被拿去和 target_quality 比,
    一次被中途停下的扫描于是可能**提前结束修针循环**（假成功）。"""
    ctx = _spec_ctx(WaitScanComplete=_stopped_early(),
                    # 一个「够好」的评分:帧没扫完却仍被采信的话,循环会当场收工。
                    AssessImageQuality=_Res(data={"fft_quality": 0.99}))
    _run("ConditionTip", {"target_quality": 0.8, "max_attempts": 2}, ctx)
    # 帧从未扫完 ⇒ quality 恒为 0.0 ⇒ 绝不会因为「达标」而提前退出:
    # 两轮都得跑满，TipPulse 应当被调用两次。
    assert len([c for c in ctx.calls if c[0] == "TipPulse"]) == 2


def test_condition_tip_still_finishes_early_on_a_complete_good_frame():
    """对照组。少了它，上一条也可能只是因为 ConditionTip 永远跑满。"""
    ctx = _spec_ctx(WaitScanComplete=_completed(),
                    AssessImageQuality=_Res(data={"fft_quality": 0.99}))
    _run("ConditionTip", {"target_quality": 0.8, "max_attempts": 2}, ctx)
    assert len([c for c in ctx.calls if c[0] == "TipPulse"]) == 1


def test_shape_tip_refuses_to_pick_a_plunge_site_from_an_unfinished_wide_scan():
    """这条链上代价最高的错误:`FindFlatRegion` 在没扫完的帧上照样会返回一个
    「平坦区」，而下一步是**把针扎进去**。宁可整条停下，也不要换个点接着扎。"""
    ctx = _spec_ctx(WaitScanComplete=_stopped_early())
    res = _run("ShapeTipOnSurface", {"max_attempts": 3}, ctx)
    assert res.success is False
    assert "wide scan did not finish" in res.error
    assert "FindFlatRegion" not in _skills(ctx), "拿不完整的帧选了扎针点"


def test_survey_does_not_count_a_truncated_tile_as_scanned():
    """与 Python 孪生同策略:只判这一格没扫成，整批继续 —— 所以**不是** fail，
    而是这一格不计数。"""
    ctx = _spec_ctx(WaitScanComplete=_stopped_early())
    res = _run("SurveySurface_TileScan", {"grid_n": 2, "tile_size_m": 50e-9}, ctx)
    assert res.data["outputs"]["scanned"] == 0
    # 整批仍然跑完了所有格子 —— 没有被 fail 中止（success_when="scanned>=1"
    # 让整体判失败，那是既有语义；关键是四格都跑过了）。
    assert len([c for c in ctx.calls if c[0] == "ConfigureScan"]) == 4


def test_survey_counts_complete_tiles():
    ctx = _spec_ctx(WaitScanComplete=_completed())
    res = _run("SurveySurface_TileScan", {"grid_n": 2, "tile_size_m": 50e-9}, ctx)
    assert res.data["outputs"]["scanned"] == 4


def test_demo_reports_that_the_frame_did_not_finish_without_aborting_sts():
    """孪生策略:如实记录、不中止 STS（STS 点是几何生成的，不取自图像）。
    但帧的真相必须出得来，否则没扫完和扫完在结果里一模一样。"""
    ctx = _spec_ctx(WaitScanComplete=_stopped_early())
    res = _run("DemoScanAndSTS", {"sts_count": 2}, ctx)
    assert res.success
    assert res.data["outputs"]["scan_completed"] is False
    assert len([c for c in ctx.calls if c[0] == "AcquireSTS"]) == 2, "STS 被误伤"


def test_demo_reports_a_complete_frame_as_complete():
    ctx = _spec_ctx(WaitScanComplete=_completed())
    res = _run("DemoScanAndSTS", {"sts_count": 2}, ctx)
    assert res.data["outputs"]["scan_completed"] is True


@pytest.mark.parametrize("name,params", [
    ("ConditionTip", {"target_quality": 0.8, "max_attempts": 1}),
    ("SurveySurface_TileScan", {"grid_n": 1, "tile_size_m": 100e-9}),
    ("DemoScanAndSTS", {"sts_count": 1}),
    ("BatchRegionsScan", {"regions": "[]"}),
])
def test_a_wait_result_reporting_neither_flag_still_runs(name, params):
    """失败开放,六处一致:任何不报这些字段的 WaitScanComplete（旧版/替身/桩）
    都必须退回旧行为，而不是把「少一个可选字段」变成「扫描技能崩了」。
    加守卫之前这个坑当场咬过一次。"""
    ctx = _spec_ctx(WaitScanComplete=_Res(data={"polls": 1}))
    res = _run(name, params, ctx)
    assert res is not None and "ExprError" not in (res.error or "")
