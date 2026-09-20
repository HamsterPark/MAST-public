"""换位置必须体现在旁白中。

同点脉冲预算、寻找落点和移动是不同步骤；旁白应显示移动结果与未移动的理由，
让调用者不必读取内部轨迹即可区分重复脉冲与换位置。"""
from __future__ import annotations

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

import pytest  # noqa: E402

from mast.chat.narration_templates import (  # noqa: E402
    BEGIN_KIND_FOR_SKILL,
    RESULT_KIND_FOR_SKILL,
    TEMPLATES,
    dig,
)


def _r(kind: str, d: dict) -> str:
    return TEMPLATES[kind].render(d)


# ── 接线 ────────────────────────────────────────────────────────────────────

def test_the_two_invisible_steps_are_now_wired():
    """「问地图」和「挪窝」各自有一条旁白 —— 这两步此前一个字都不发。"""
    assert BEGIN_KIND_FOR_SKILL["FindCleanSpot"] == "find_spot"
    assert BEGIN_KIND_FOR_SKILL["MoveToXY"] == "move_xy"
    assert RESULT_KIND_FOR_SKILL["FindCleanSpot"] == "find_spot_result"
    assert RESULT_KIND_FOR_SKILL["BiasPulseWithReadback"] == "pulse_readback"
    for k in ("find_spot", "find_spot_result", "move_xy", "pulse_readback"):
        assert k in TEMPLATES, f"接了 {k} 却没有模板 —— 会永远走不到"


def test_skills_without_a_pinned_vocabulary_are_still_silent():
    """没钉住返回字段的技能**仍然不发** —— 那不是遗漏,是同一条纪律。

    宁可不说,也不说一句读错字段的话:后者永远走 fallback,而且看起来完全正常。
    """
    for skill in ("AssessTipQuality", "AssessTipSharpness", "ScanAt"):
        assert skill not in RESULT_KIND_FOR_SKILL, (
            f"{skill} 被接上了结论旁白 —— 先把它的返回字段钉死(单测 + 产物声明)")


# ── 说得出口 ────────────────────────────────────────────────────────────────

def test_asking_the_map_says_what_it_is_asking_for():
    assert "脉冲" in _r("find_spot", {"params": {"purpose": "pulse"}})
    assert "扎针" in _r("find_spot", {"params": {"purpose": "tip_shape"}})


def test_the_answer_carries_the_numbers_you_would_debug_with():
    s = _r("find_spot_result", {"result": {
        "x_m": 959e-9, "y_m": 1030e-9, "distance_m": 1.41e-6,
        "candidates": [1, 2, 3, 4], "effective_half_range_m": 250e-9,
        "spot_radius_m": 500e-9, "map_known": True, "markers_seen": 12,
        "crash_memory_points": 0, "crash_memory_unlocated": 0, "recentred": False}})
    assert "959" in s and "1030" in s, "落点坐标没说"
    assert "1410 nm" in s, "距离没说 —— 那正是「它到底挪没挪」的答案"
    assert "候选 4 个" in s
    assert "±250 nm" in s, "可用区没说 —— 那是「为什么找不到地方」的答案"
    assert "12 个标记" in s


def test_a_zero_distance_says_so_instead_of_pretending_to_move():
    """零距离不应下发移动指令，旁白应明确说明未移动，避免把选择位置误报为执行移动。"""
    s = _r("find_spot_result", {"result": {
        "x_m": 0.0, "y_m": 0.0, "distance_m": 0.0, "candidates": [1],
        "map_known": True, "markers_seen": 0}})
    assert "脚下" in s and "不用移动" in s


def test_moving_says_where_to():
    s = _r("move_xy", {"params": {"x_m": -1.0407e-6, "y_m": 29.7e-9}})
    assert "-1041" in s and "30" in s


# ── 三态 ────────────────────────────────────────────────────────────────────

def test_an_unreadable_map_is_never_rendered_as_a_clean_one():
    """``map_known=False`` 是**读不到实验记录**,不是「这儿很干净」。

    两者在几何上完全一样 —— 只有这个布尔分得开。在读不到历史的情况下往表面打
    10 V 脉冲,和在确认干净的地方打,是两件不同的事。
    """
    s = _r("find_spot_result", {"result": {
        "x_m": 0.0, "y_m": 0.0, "distance_m": 0.0, "candidates": [1],
        "map_known": False}})
    assert "读不到实验记录" in s
    assert "个标记" not in s, f"读不到地图却报出了标记数:{s}"

    # 第三态:连「读没读到」都没记下来 —— 也要说,不许默认成读到了。
    s3 = _r("find_spot_result", {"result": {
        "x_m": 0.0, "y_m": 0.0, "distance_m": 0.0, "candidates": [1]}})
    assert "没记下来" in s3


def test_an_uncoordinated_crash_is_announced_because_the_circle_cannot_be_drawn():
    """撞了但读不到坐标 ⇒ 避让圈画不出来 ⇒ 这个落点**无法保证**不在坑上。"""
    s = _r("find_spot_result", {"result": {
        "x_m": 0.0, "y_m": 0.0, "distance_m": 0.0, "candidates": [1],
        "map_known": True, "markers_seen": 3, "crash_memory_unlocated": 2}})
    assert "读不到坐标" in s and "圈画不出来" in s


def test_recentring_is_announced_because_the_spot_may_be_a_micron_away():
    """悄悄把「从针尖处找」换成「从区中心找」,调用方会以为落点就在手边。"""
    s = _r("find_spot_result", {"result": {
        "x_m": 0.0, "y_m": 0.0, "distance_m": 1.4e-6, "candidates": [1],
        "map_known": True, "markers_seen": 3, "recentred": True}})
    assert "区中心重搜" in s


def test_the_pulse_verdict_keeps_its_three_states_apart():
    """``insufficient_data``(读不到)绝不许写成 ``none``(确实没跳)。

    两句话指向完全不同的下一步:查信号链 vs 再打一发。
    ``_tip_phases`` 里 ``unreadable`` 与 ``no_effect`` 分开记账,正是这个道理 ——
    20 发全是 no_effect 说的是「脉冲打不动这根针」,20 发全是 unreadable
    说的是「Z 读回来有问题」。混在一起就都问不出来了。
    """
    hit = _r("pulse_readback", {"result": {"step": {"direction": "up", "delta_m": 22e-9}}})
    assert "改变了针尖" in hit and "22" in hit

    miss = _r("pulse_readback", {"result": {"step": {"direction": "none", "delta_m": 3e-11}}})
    assert "未改变针尖" in miss
    assert "读不出来" not in miss

    dunno = _r("pulse_readback",
               {"result": {"step": {"direction": "insufficient_data"}}})
    assert "读不出来" in dunno
    assert "未改变针尖" not in dunno, f"把「判不了」说成了「未改变针尖」:{dunno}"
    assert "信号链" in dunno, "没说出下一步该查什么"


# ── parity:模板读的字段,技能真的会给 ──────────────────────────────────────

def test_every_path_the_template_reads_exists_in_what_the_skill_returns():
    """**跑一次真的 FindCleanSpot**,逐条核对模板要读的路径都在返回里。

    这条是这一组里最重要的:一个写错的路径不会报错,只会让那句话永远走
    fallback —— 语法完全正确、永远不带数字,而没有人会注意到。
    ``requires`` 少一条就整句降级,``records`` 少一条则事后对账缺一个数。

    不查字面量、不查文档:直接问技能要一份返回,拿模板声明的路径去 ``dig``。
    """
    from mast.skills.builtins.clean_spot import FindCleanSpot

    class Ctx:
        """针尖停在原点,其余调用一律成功(与 test_crash_point... 同款)。"""

        state = None
        _registry = None

        def safe_call(self, method, *args, role="main"):
            class R:
                error = ""
                return_value = ("", b"", [0.0, 0.0, 0.0])
            return R()

    res = FindCleanSpot().execute(Ctx(), {"purpose": "pulse", "count": 4})
    assert res.success, f"技能没跑成,这条 parity 就没意义了:{res.error}"

    d = {"result": res.data}
    tpl = TEMPLATES["find_spot_result"]
    missing = [p for p in tpl.requires if dig(d, p) is None]
    assert not missing, (
        f"模板的必需路径在技能返回里不存在:{missing} —— 这句话会永远走 fallback")

    # records 缺一条只是对账少一个数,但同样是「读错字段」,一并逮。
    absent = [p for p in tpl.records
              if p.split(".", 1)[1] not in (res.data or {})]
    assert not absent, f"模板记录的路径技能不返回:{absent}"


def test_the_pulse_template_reads_the_key_the_skill_actually_writes():
    """判定挂在 ``data["step"]`` 下,不是顶层的 ``direction``。

    ``bias_pulse_readback.py`` 里 ``data["step"] = verdict``,而 ``verdict``
    才有 ``direction`` / ``delta_m``。写成 ``result.direction`` 会永远读不到。
    """
    import inspect

    from mast.skills.builtins import bias_pulse_readback as B

    src = inspect.getsource(B)
    assert '"step": verdict' in src or '"step": verdict,' in src, (
        "技能不再把判定放在 data['step'] 下 —— 模板的路径要跟着改")
    for p in TEMPLATES["pulse_readback"].requires:
        assert p.startswith("result.step."), (
            f"{p} 没走 data['step'] —— 那一层是判定的真正位置")


# ── 调平(2026-08-17) ────────────────────────────────────────────────────────

def test_the_levelling_step_says_whether_it_happened():
    """调平旁白必须区分已执行、已跳过和未确定。
    测试载荷取自 AutoTilt 的实际 report 字段 outcome 与 before/after.z_span_m，
    不构造生产方从未返回的字段来让模板测试通过。"""
    done = _r("auto_tilt_result", {"result": {
        "outcome": "applied",
        "before": {"z_span_m": 1.8e-9}, "after": {"z_span_m": 0.3e-9}}})
    assert "调平**已执行**" in done
    # 落差要看得出量级差别 —— 取整会把 0.3 nm 报成 "0",读起来像「完全平了」。
    assert "0 nm" not in done, f"把 0.3 nm 报成了 0:{done}"

    skipped = _r("auto_tilt_result", {"result": {
        "outcome": "skipped", "reason": "calibration_missing",
        "detail": "本机还没做过倾斜标定"}})
    assert "调平**未执行**" in skipped
    assert "调平**已执行**" not in skipped, f"skip 被写成了完成:{skipped}"
    assert "倾斜标定" in skipped, "没说为什么未执行"

    # 第三态:连做没做都没记下来 —— 不许默认成做了。
    dunno = _r("auto_tilt_result", {"result": {}})
    assert "没记下来" in dunno
    # ⚠️ 断言**完整的判决前缀**,不是「已执行」「未执行」这两个片段 ——
    # 后者会在「做没做没记下来」里假命中(第一版就是这么写的)。
    # (2026-08-18 术语化:原文是「做了 / 没做」,要求尽量用术语。
    #  这条测试钉的始终是**三态分得开**,不是某几个字。)
    assert "调平**已执行**" not in dunno
    assert "调平**未执行**" not in dunno


def test_levelling_is_wired_at_both_ends():
    assert BEGIN_KIND_FOR_SKILL["AutoTilt"] == "auto_tilt"
    assert RESULT_KIND_FOR_SKILL["AutoTilt"] == "auto_tilt_result"
    assert BEGIN_KIND_FOR_SKILL["FindFlatRegion"] == "flat_region"


# ── 这道闸门是一整天的假话换来的(2026-08-17) ────────────────────────────────

def test_no_wired_template_silently_falls_back_on_a_realistic_payload():
    """每个接了线的结论模板,喂**真实形状的载荷**必须真的说出话来。

    ── 出处

    问题场景:扎针后扫 10nm 看到团簇,但旁白只说「分析了团簇」,分析结果本身没有保存、没有给出结论。

    查下来 ``cluster_roundness`` **每一次都在走 fallback** —— 连 ``is_round=True``
    的时候也是。根因在 ``_satisfied``:它为了防「``True`` 被念成打一发 1 V」而
    **排除所有 bool**,而 ``is_round`` 恰恰是个真正的布尔判读。把它写进
    ``requires`` 等于让这条模板永远说那句「没记下来」。

    **它整整一天没被发现**,因为那句话语法完全正确、看起来完全正常 ——
    正是这个仓库里最贵的那种失效。已有的闸门都查不到它:
      · ``requires`` 里的路径是**真存在**的(所以 parity 测试绿);
      · 句子里没有写死的数字(所以字面量闸门绿);
      · 模板注册了、接线了(所以接线测试绿)。
    **没有一道闸门去问「它到底说不说得出话」。** 这条就是那道。
    """
    from mast.chat.narration_templates import render

    # 每个接了线的技能,给一份**它真的会返回的**载荷。
    payloads = {
        "cluster_roundness": {"result": {
            "is_round": True, "equivalent_axis_ratio": 0.86,
            "area_px": 420, "n_components": 1, "multi_tip": False}},
        "find_spot_result": {"result": {
            "x_m": 4e-7, "y_m": -4e-7, "distance_m": 5.6e-7,
            "candidates": [1, 2], "effective_half_range_m": 6e-7,
            "spot_radius_m": 2e-7, "map_known": True, "markers_seen": 3}},
        "pulse_readback": {"result": {
            "step": {"direction": "up", "delta_m": 6e-9}}},
        # 夹具键名遵循 AutoTilt.report 的实际返回结构。
        # 只用设计中存在、生产中不存在的键会让接线测试失去判别力。
        "auto_tilt_result": {"result": {
            "outcome": "applied", "reason": "",
            "before": {"z_span_m": 1.8e-9, "measured_slope_deg": 0.021},
            "after": {"z_span_m": 3e-10}, "iterations": 2}},
    }
    wired = set(RESULT_KIND_FOR_SKILL.values())
    assert wired <= set(payloads), (
        f"接了线却没给真实载荷的模板:{sorted(wired - set(payloads))} —— "
        "补一份它真会返回的 data,否则这道闸门看不见它")

    for kind, data in payloads.items():
        r = render(kind, data)
        assert r is not None, f"{kind}: render 返回 None"
        assert r.degraded is False, (
            f"{kind} 在一份真实载荷上**走了 fallback** —— 它永远不会说出话来。"
            f"实际说的是:{r.text}")
        assert r.text != TEMPLATES[kind].fallback, (
            f"{kind} 说的就是 fallback 那句:{r.text}")
