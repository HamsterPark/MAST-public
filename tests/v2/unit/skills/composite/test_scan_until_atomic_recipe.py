# -*- coding: utf-8 -*-
"""验证换区、合适像素尺度和当前线速度共同传入配方；证据不足不应直接触发换区。"""
from __future__ import annotations

import pytest

from mast.skills.composite import scan_until_atomic as SUA
from mast.vision.atomic_phase import SCALE_FULL_NMPP


# ── 成帧：压进 full 档 ────────────────────────────────────────────────────

def test_a_transition_scale_frame_is_shrunk_into_the_full_gate():
    """8 nm / 256 px = 0.0313 nm/px 落在过渡带 ⇒ 必须缩。"""
    new = SUA._shrink_into_full_gate(8e-9, 256)
    assert new is not None
    assert (new * 1e9) / 256 < SCALE_FULL_NMPP, "缩完还没进 full 档"


def test_a_frame_already_in_the_full_gate_is_left_alone():
    """已经在档里就别动用户的视场 —— 少改一次是一次。"""
    assert SUA._shrink_into_full_gate(5e-9, 256) is None


def test_the_boundary_is_not_taken_from_a_copied_constant():
    """闸门边界必须来自 ``atomic_phase``，不许在这里抄一个 0.02。

    抄一份就是两处各漂各的：判据改了边界而成帧不知道，于是这个技能会**稳定地**
    交出一批刚好判不了的帧，而每一步看起来都对。
    """
    import inspect

    src = inspect.getsource(SUA._shrink_into_full_gate)
    assert "SCALE_FULL_NMPP" in src
    # 只看**代码**，不看 docstring —— 文档里当然会提到那个数。
    code = src.split('"""')[0] + '"""'.join(src.split('"""')[2:])
    assert "0.02" not in code, "把闸门边界抄进代码里了"


def test_the_shrink_leaves_margin_for_float_round_trips():
    """压线不算进档：闸门是**严格小于**，而 nm↔m 往返会把 0.02 变成 0.0200…4。"""
    for px in (128, 256, 512, 1024):
        new = SUA._shrink_into_full_gate(50e-9, px)
        assert new is not None
        assert (new * 1e9) / px < SCALE_FULL_NMPP


def test_an_absurdly_small_frame_is_refused_rather_than_produced():
    """缩到比晶格常数还小的视场上没有可判的东西 —— 宁可不缩。"""
    assert SUA._shrink_into_full_gate(1e-9, 4) is None


# ── 读仪器：读不到就说读不到，不猜 ────────────────────────────────────────

class _Ctx:
    def __init__(self, data=None, boom=False):
        self._d = data or {}
        self._boom = boom

    def run(self, name, params):
        if self._boom:
            raise RuntimeError("仪器没接")
        return type("R", (), {"data": self._d.get(name)})()


def test_pixels_and_speed_are_read_from_the_instrument():
    ctx = _Ctx({"GetScanBuffer": {"pixels": 512},
                "GetScanSpeed": {"fwd_speed_m_s": 6.51e-9}})
    assert SUA._read_pixels(ctx) == 512
    assert SUA._read_linear_speed(ctx) == pytest.approx(6.51e-9)


@pytest.mark.parametrize("ctx", [
    _Ctx({}),                                   # 键不在
    _Ctx({"GetScanBuffer": {"pixels": 0}}),     # 0 不是像素数
    _Ctx(boom=True),                            # 仪器抛
    None,                                       # 根本没有 context
])
def test_unreadable_instrument_state_yields_none_not_a_guess(ctx):
    """**不猜 256，也不猜档位表里的速度。** 读不到就退回原有行为。"""
    assert SUA._read_pixels(ctx) is None
    assert SUA._read_linear_speed(ctx) is None


# ── 换地方：只有「真的判成没有」才算证据 ──────────────────────────────────

def _plan(params, ctx_data, history_seed=None):
    """跑 plan_dynamic，把它 yield 的步骤收集起来（不执行）。"""
    from mast.skills.composite.graph_executor import CompositeProgress

    class _Ex:
        def __init__(self):
            self.progress = CompositeProgress("ScanUntilAtomicResolution")
            self.progress.partial_data.update(history_seed or {})
            self.sub_results = {}
            self.context = _Ctx(ctx_data)

        def set_partial(self, k, v):
            self.progress.partial_data[k] = v

        def set_total_steps(self, n):
            pass

    ex = _Ex()
    steps = []
    gen = SUA.ScanUntilAtomicResolution().plan_dynamic(dict(params), ex)
    for st in gen:
        steps.append(st)
        if len(steps) > 6:          # 只看头几步；再往下要 sub_results
            break
    return steps, ex


def test_the_plan_records_why_this_frame_is_that_size():
    """「为什么这帧是 4.9 nm 不是 8 nm」必须能事后回答。"""
    steps, ex = _plan(
        {"max_attempts": 1, "center_x_m": 0.0, "center_y_m": 0.0,
         "scan_size_m": None},
        {"GetScanFrame": {"center_x_m": 0.0, "center_y_m": 0.0, "width_m": 8e-9},
         "GetScanBuffer": {"pixels": 256},
         "GetScanSpeed": {"fwd_speed_m_s": 6.51e-9}})
    plan = ex.progress.partial_data.get("frame_plan") or []
    assert any("full 档" in p for p in plan), plan
    assert any("线速度" in p for p in plan), plan
    assert ex.progress.partial_data["planned_size_m"] < 8e-9


def test_an_explicit_size_is_never_overridden():
    """调用方明说了视场就别改 —— 这个技能不该悄悄换掉用户看的地方。"""
    steps, ex = _plan(
        {"max_attempts": 1, "scan_size_m": 8e-9,
         "center_x_m": 0.0, "center_y_m": 0.0},
        {"GetScanBuffer": {"pixels": 256},
         "GetScanSpeed": {"fwd_speed_m_s": 6.51e-9}})
    assert ex.progress.partial_data["planned_size_m"] == pytest.approx(8e-9)


def test_an_explicit_line_time_wins_over_the_current_speed():
    steps, ex = _plan(
        {"max_attempts": 1, "scan_size_m": 5e-9, "line_time_s": 2.0,
         "center_x_m": 0.0, "center_y_m": 0.0},
        {"GetScanBuffer": {"pixels": 256},
         "GetScanSpeed": {"fwd_speed_m_s": 6.51e-9}})
    # 计划里**记下实际用的值**（报告要能回答「这次用了什么」），
    # 但它不该是从仪器速度算来的 —— 那条路没走。
    assert ex.progress.partial_data.get("planned_line_time_s") == pytest.approx(2.0)
    assert not any("线速度" in p
                   for p in (ex.progress.partial_data.get("frame_plan") or [])),         "显式给了每线时间，却还是去问了仪器速度"
    assert steps[0].params.get("line_time_s") == pytest.approx(2.0)


def test_relocation_is_off_by_default():
    """默认保持「原地多扫几帧」的原意 —— 那是要求的做法。"""
    p = {x.name: x.default
         for x in SUA.ScanUntilAtomicResolution().metadata().parameters}
    assert p["relocate_after_attempts"] == 0
    assert p["relocate_step_m"] == pytest.approx(3e-8)
    assert p["keep_current_speed"] is True
    assert p["prefer_full_scale_gate"] is True


def test_the_advice_points_at_relocation_when_it_was_never_tried():
    """一轮下来没换过地方 ⇒ 建议里要提这条，别直接把人推去修针。"""
    from mast.skills.composite.graph_executor import CompositeProgress

    prog = CompositeProgress("ScanUntilAtomicResolution")
    prog.partial_data.update({
        "attempts_done": 4, "relocations": 0,
        "history": [{"attempt": i, "verdict": "absent"} for i in range(1, 5)]})
    out = SUA.ScanUntilAtomicResolution().aggregate({}, prog)
    assert "relocate_after_attempts" in out["advice"]
    assert out["relocations"] == 0


def test_the_advice_does_not_nag_when_relocation_already_happened():
    from mast.skills.composite.graph_executor import CompositeProgress

    prog = CompositeProgress("ScanUntilAtomicResolution")
    prog.partial_data.update({
        "attempts_done": 4, "relocations": 2,
        "history": [{"attempt": i, "verdict": "absent"} for i in range(1, 5)]})
    out = SUA.ScanUntilAtomicResolution().aggregate({}, prog)
    assert "relocate_after_attempts" not in out["advice"]


def test_undetermined_frames_never_trigger_a_relocation():
    """**判不了不是「没有」。**

    拿残帧去触发换地方，等于让一次采集中断把针尖赶到别处 —— 而那正是
    2026-08-20 那晚发生的事（``WaitScanComplete`` 秒回陈旧结论，四帧全被打断，
    每一帧都是 coverage 23% 的残帧）。这条钉住：只数 ``absent``。
    """
    import inspect

    src = inspect.getsource(SUA.ScanUntilAtomicResolution.plan_dynamic)
    assert "if reloc_after:" in src, "换地方那一段不见了"
    seg = src.split("if reloc_after:")[1].split("logger.info")[0]
    assert 'get("verdict") == "absent"' in seg, (
        "换地方的计数没有限定 absent —— 判不了会把它触发")


# ── 契约：失败必须**指路** ────────────────────────────────────────────────

def _agg(**partial):
    from mast.skills.composite.graph_executor import CompositeProgress

    prog = CompositeProgress("ScanUntilAtomicResolution")
    prog.partial_data.update(partial)
    return SUA.ScanUntilAtomicResolution().aggregate({}, prog)


@pytest.mark.parametrize("shape,partial", [
    ("一帧都没扫（读不到扫描框）",
     {"attempts_done": 0, "relocations": 0, "history": [],
      "aborted_reason": "no_scan_frame"}),
    ("全是残帧（判不了）",
     {"attempts_done": 3, "relocations": 0,
      "history": [{"attempt": i, "verdict": "undetermined"} for i in range(1, 4)]}),
    ("扫完了都没有，没换过地方",
     {"attempts_done": 4, "relocations": 0,
      "history": [{"attempt": i, "verdict": "absent"} for i in range(1, 5)]}),
    ("扫完了都没有，换过地方",
     {"attempts_done": 6, "relocations": 2,
      "history": [{"attempt": i, "verdict": "absent"} for i in range(1, 7)]}),
    ("有没有混着（部分判不了）",
     {"attempts_done": 4, "relocations": 1,
      "history": [{"attempt": 1, "verdict": "absent"},
                  {"attempt": 2, "verdict": "undetermined"},
                  {"attempt": 3, "verdict": "absent"},
                  {"attempt": 4, "verdict": "undetermined"}]}),
])
def test_every_failed_shape_hands_back_a_next_step(shape, partial):
    """所有 found=False 分支必须返回 advice，使调用方能选择后续步骤。"""
    out = _agg(**partial)
    assert out["found"] is False, "这个形状本来就该是失败"
    assert out.get("advice"), "%s：found=false 却没有 advice —— agent 无路可走" % shape
    assert len(str(out["advice"])) > 20, "%s：advice 短到说不出下一步" % shape


def test_a_successful_hunt_needs_no_advice():
    """成功时**不**该有 advice —— 有的话，读到它的 agent 会以为还得接着修。

    这一条同时是上一条的反证：若 ``advice`` 是无条件塞进去的，
    上面那五个形状全过也证明不了什么。
    """
    out = _agg(attempts_done=1, relocations=0, found_at=1,
               found_path="x.sxm",
               history=[{"attempt": 1, "verdict": "atomic", "concentration": 300.0}])
    assert out["found"] is True
    assert "advice" not in out, "成功了还指路 —— agent 会当成没做完"
