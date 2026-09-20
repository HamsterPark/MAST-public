"""修针流程的三个新 builtin：就近选点、台阶锐利度、开工前自检。"""
from __future__ import annotations

import sys
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[5] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _name in list(sys.modules):
    if _name == "mast" or _name.startswith("mast."):
        _f = getattr(sys.modules[_name], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_name]

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from mast.skills.builtins.clean_spot import FindCleanSpot, parse_spots  # noqa: E402
from mast.skills.builtins.tip_conditioning_selfcheck import (  # noqa: E402
    REQUIRED_SKILLS,
    TipConditioningSelfCheck,
)
from mast.skills.builtins.tip_sharpness import AssessTipSharpness  # noqa: E402

# 源码级断言走它,不用 ``inspect.getsource``(2026-08-15):后者按 import 那一刻
# 的行号切当前文件,别人同时在改就返回错位切片 —— ``in`` 那半给假红,
# ``not in`` 那半给**假绿**。整模块 getsource 是安全档,不在此列。
from tests.v2.srcref import source_of  # noqa: E402


class Ctx:
    """记录技能调用的测试上下文；带 registry 的分支和无 registry 的分支分别覆盖。"""

    def __init__(self, tip=(0.0, 0.0), *, folme_error="", registry=None):
        self.tip = tip
        self.folme_error = folme_error
        self.calls: list[str] = []
        self._registry = registry
        self.state = None

    def safe_call(self, method, *args, role="main"):
        self.calls.append(method)
        outer = self

        class R:
            error = outer.folme_error if method == "FolMe_XYPosGet" else ""
            return_value = ("", b"", list(outer.tip) if method == "FolMe_XYPosGet"
                            else [0.0])

        return R()

    def check_abort(self):
        return False


# ── FindCleanSpot ───────────────────────────────────────────────────────────

def test_parse_spots_round_trips_and_skips_junk():
    assert parse_spots("1e-9,2e-9;3e-9,-4e-9") == [(1e-9, 2e-9), (3e-9, -4e-9)]
    assert parse_spots("") == []
    assert parse_spots("bad;1,2;also,bad,here") == [(1.0, 2.0)]


def test_finds_a_spot_near_the_tip():
    res = FindCleanSpot().execute(Ctx(tip=(1e-8, -2e-8)), {"purpose": "pulse"})
    assert res.success, res.error
    assert res.data["origin_source"] == "live_tip"
    assert res.data["distance_m"] >= 0.0
    assert res.data["candidates"]


def test_purpose_picks_the_matching_avoidance_radius():
    """脉冲溅得远（150 nm），扎针只留一个小簇（30 nm）—— 半径取自地图配置。"""
    pulse = FindCleanSpot().execute(Ctx(), {"purpose": "pulse"})
    poke = FindCleanSpot().execute(Ctx(), {"purpose": "tip_shape"})
    assert pulse.data["spot_radius_m"] > poke.data["spot_radius_m"]


def test_unknown_map_is_flagged_not_silently_treated_as_clean():
    """读不到记录 ≠ 表面干净 —— 这个区别必须传出去。"""
    res = FindCleanSpot().execute(Ctx(), {})
    assert res.data["map_known"] is False
    assert "无法确认" in res.data["reason"]


def test_no_origin_is_an_error_not_a_guess():
    """读不到针尖位置又没给起点 —— 没有起点可搜，不能随便挑一个。"""
    res = FindCleanSpot().execute(Ctx(folme_error="TCP down"), {})
    assert res.success is False and "origin" in res.error


def test_explicit_origin_skips_the_hardware_read():
    ctx = Ctx()
    res = FindCleanSpot().execute(ctx, {"from_x_m": 5e-8, "from_y_m": 0.0})
    assert res.data["origin_source"] == "explicit"
    assert "FolMe_XYPosGet" not in ctx.calls


def test_is_read_only():
    from mast.core.types import SkillCategory
    m = FindCleanSpot().metadata()
    assert m.category == SkillCategory.READ
    assert not m.capabilities, "只读选点不该带任何危险能力标签"


# ── AssessTipSharpness ──────────────────────────────────────────────────────

def _write_sxm(tmp_path, img, *, width_m=1e-7, name="frame"):
    """最小可读 .sxm：头 + 一个 Z 通道。"""
    ny, nx = img.shape
    header = (
        ":NANONIS_VERSION:\n2\n"
        ":SCANIT_TYPE:\nFLOAT            MSBFIRST\n"
        ":REC_DATE:\n01.08.2026\n"
        ":REC_TIME:\n12:00:00\n"
        ":SCAN_PIXELS:\n"
        f"{nx} {ny}\n"
        ":SCAN_RANGE:\n"
        f"{width_m:.6E} {width_m:.6E}\n"
        ":SCAN_OFFSET:\n0.0E+0 0.0E+0\n"
        ":SCAN_ANGLE:\n0.000E+0\n"
        ":SCAN_DIR:\nup\n"
        ":DATA_INFO:\n"
        "\tChannel\tName\tUnit\tDirection\tCalibration\tOffset\n"
        "\t14\tZ\tm\tboth\t1.000E+0\t0.000E+0\n"
        ":SCANIT_END:\n\n"
    )
    p = tmp_path / f"{name}.sxm"
    with open(p, "wb") as f:
        f.write(header.encode("utf-8"))
        f.write(b"\x1a\x04")
        data = np.asarray(img, dtype=">f4")
        f.write(data.tobytes())          # forward
        f.write(data[:, ::-1].tobytes())  # backward
    return str(p)


def test_a_flat_frame_reports_no_step_rather_than_a_number(tmp_path):
    """平坦区上的「边缘宽度」是在量噪声 —— 编一个数字出来比不给更糟。"""
    rng = np.random.default_rng(0)
    img = rng.normal(0.0, 1e-12, (64, 64))
    res = AssessTipSharpness().execute(None, {"scan_path": _write_sxm(tmp_path, img)})
    assert res.success, res.error
    assert res.data["verdict"] == "no_step"
    assert res.data["edge_resolution_nm"] is None
    assert res.data["has_step"] is False


def _terraced(n=128, *, n_steps=2, height_m=2.4e-10, edge_px=1.0,
              noise_frac=0.02, seed=1):
    """台阶形状由数学函数生成。"""
    from scipy import ndimage as ndi
    rng = np.random.default_rng(seed)
    img = np.zeros((n, n))
    width = n // (n_steps + 1)
    for k in range(1, n_steps + 1):
        img[:, k * width:] += height_m
    img = ndi.gaussian_filter(img, (0.0, edge_px))     # 有限的边缘宽度
    return img + rng.normal(0.0, height_m * noise_frac, img.shape)


def test_a_stepped_surface_is_measured(tmp_path):
    """阶梯表面上应当量得出边缘宽度。"""
    path = _write_sxm(tmp_path, _terraced())
    res = AssessTipSharpness().execute(None, {"scan_path": path})
    assert res.success, res.error
    assert res.data["has_step"] is True, res.data
    assert res.data["edge_resolution_nm"] > 0


def test_a_blunter_tip_smears_the_edge_wider(tmp_path):
    """钝针尖把台阶抹开 —— 边缘宽度就是区分它和尖针尖的那个量。"""
    sharp = AssessTipSharpness().execute(None, {
        "scan_path": _write_sxm(tmp_path, _terraced(edge_px=0.6), name="sharp")})
    blunt = AssessTipSharpness().execute(None, {
        "scan_path": _write_sxm(tmp_path, _terraced(edge_px=2.0), name="blunt")})
    assert sharp.data["edge_resolution_nm"] < blunt.data["edge_resolution_nm"]


def test_the_criterion_is_robust_to_noise(tmp_path):
    """噪声由固定随机种子独立生成。"""
    widths = []
    for i, nf in enumerate((0.005, 0.02, 0.05)):
        res = AssessTipSharpness().execute(None, {
            "scan_path": _write_sxm(tmp_path, _terraced(noise_frac=nf),
                                    name=f"n{i}")})
        assert res.data["has_step"] is True, nf
        widths.append(res.data["edge_resolution_nm"])
    assert max(widths) / min(widths) < 1.2, widths


def test_dense_steps_defeat_the_edge_criterion(tmp_path):
    """台阶排得密时判据会说「判不了」——这是它的真实工作区间，不是 bug。

    边缘判据认的是相干的梯度离群点。台阶越密，边缘像素占比越高，「离群」就越不
    成立。记在这里，是因为读结果的人必须知道 no_step 有两种含义：这块地方没台阶，
    **或者针尖钝到把台阶抹平了**。绝不能把它当作「针尖没问题」。
    """
    res = AssessTipSharpness().execute(None, {
        "scan_path": _write_sxm(tmp_path, _terraced(n_steps=4, edge_px=2.0))})
    assert res.data["verdict"] == "no_step"
    # 但起伏还在 —— 这正是「有台阶但判不出边缘」与「真的平」的区别。
    assert res.data["corrugation_rms_m"] > 1e-12


def test_verdict_needs_an_explicit_threshold(tmp_path):
    """不给阈值就只报数 —— 「够不够尖」取决于像素大小和表面，不该由这一层替人定。"""
    path = _write_sxm(tmp_path, _terraced())
    assert AssessTipSharpness().execute(None, {"scan_path": path})        .data["verdict"] == "measured"
    v = AssessTipSharpness().execute(
        None, {"scan_path": path, "sharp_edge_nm": 50.0}).data["verdict"]
    assert v in ("sharp", "blunt")


def test_missing_file_fails_cleanly():
    res = AssessTipSharpness().execute(None, {"scan_path": "/nope/none.sxm"})
    assert res.success is False and "不存在" in res.error


# ── TipConditioningSelfCheck ────────────────────────────────────────────────

def test_selfcheck_is_read_only_and_never_fails():
    """体检本身总是成功；结论在 data.ready 里 —— 把「查出问题」报成技能失败，
    会让调用方分不清「体检没做成」和「体检发现了问题」。"""
    res = TipConditioningSelfCheck().execute(Ctx(), {})
    assert res.success is True
    assert isinstance(res.data["ready"], bool)
    assert res.data["checks"]


def test_selfcheck_finds_every_required_skill_in_this_build():
    """判据是「模块进了 sys.modules」，不是「符号挂在包命名空间上」——
    __init__ 里有一整段是按模块 import 的，用 hasattr 查会把它们全判成缺失。"""
    import mast.skills.builtins  # noqa: F401  — 触发注册
    import mast.skills.composite  # noqa: F401
    res = TipConditioningSelfCheck().execute(Ctx(), {})
    assert res.data["missing_skills"] == []


def test_required_skills_covers_what_the_workflow_actually_calls():
    """清单要和流程实际调用的技能对得上，否则自检说「齐了」而流程还是断的。

    ⚠️ 走 **AST**，不是正则扫文本（2026-08-17 改）。

    原来是 ``re.findall(r'skill_name="([A-Za-z_]+)"', src)`` —— 它连注释和
    docstring 一起扫。当天在 ``_tip_phases`` 里写了一句解释性注释、里面带着
    ``skill_name="X"`` 这个形状，这道闸门当场报「流程调用了技能 X」。

    而且只认 ``CompositeStep(skill_name=...)``：``skill_name`` 这个关键字在别处
    还有第二个含义 —— ``storage.log_marker(skill_name=...)`` 记的是**标记的出处**，
    不是一次技能调用。把出处混进技能清单，会让自检去找一个不存在的依赖。

    (本仓为「grep 命中 ≠ 源码里有」付过学费：``artifact_claims`` 里
     ``tip_halt_source`` 只出现在注释中，被判 BOGUS。这里是同一件事的反面 ——
     注释造成的**假阳性**。)
    """
    import ast

    called: set[str] = set()
    for f in ("MASTv2/mast/skills/composite/_tip_phases.py",
              "MASTv2/mast/skills/composite/prepare_noble_tip.py"):
        src = (Path(_MASTV2_ROOT).parent / f).read_text(encoding="utf-8")
        for node in ast.walk(ast.parse(src)):
            if not isinstance(node, ast.Call):
                continue
            if getattr(node.func, "id", "") != "CompositeStep":
                continue
            for kw in node.keywords:
                if kw.arg == "skill_name" and isinstance(kw.value, ast.Constant) \
                        and isinstance(kw.value.value, str):
                    called.add(kw.value.value)
    # 闸门自检：扫不到任何东西的闸门永远是绿的。
    assert len(called) > 10, (
        f"AST 只扫到 {len(called)} 个 CompositeStep(skill_name=...) —— "
        f"构造写法变了？这道闸门已经什么都没在查")
    assert called <= set(REQUIRED_SKILLS), sorted(called - set(REQUIRED_SKILLS))


def test_unregistered_tip_is_a_blocker(monkeypatch):
    """未登记针尖时包络只给到保守档，10 V 大修一发也放不出去 —— 必须拦在开工前。"""
    import mast.core.tip_state as tip_state
    monkeypatch.setattr(tip_state, "get_current_tip", lambda: None)
    res = TipConditioningSelfCheck().execute(Ctx(), {})
    assert res.data["ready"] is False
    assert any("登记" in b for b in res.data["blockers"])


# 用真实 registry 对象验证自检，避免把 SkillMetadata 的字符串表示误作技能名。

@pytest.fixture(scope="module")
def live_registry():
    from mast.core.registry import SkillRegistry
    reg = SkillRegistry()
    reg.discover("mast.skills.builtins", "mast.skills.composite")
    return reg


def test_selfcheck_with_a_live_registry_finds_everything(live_registry):
    """线上形状：context 带着 ExecutionContext 真有的那个 _registry。"""
    assert live_registry.has("GetLatestScanFile"), "fixture 坏了，不是被测代码坏了"
    res = TipConditioningSelfCheck().execute(Ctx(registry=live_registry), {})
    assert res.data["missing_skills"] == []
    assert not any("技能齐全" in b for b in res.data["blockers"])


def test_forge_selfcheck_with_a_live_registry_finds_everything(live_registry):
    """两份自检现在共用 core.registry 的同一份实现，就一起钉住。"""
    from mast.skills.builtins.tip_forge_selfcheck import TipForgeSelfCheck
    res = TipForgeSelfCheck().execute(Ctx(registry=live_registry), {})
    assert res.data["missing_skills"] == []


def test_selfcheck_still_reports_a_genuinely_missing_skill(live_registry,
                                                           monkeypatch):
    """灵敏度。少了这一条，一个恒报「齐了」的自检也能通过上面两条。

    真的把技能从 registry 里摘掉 —— 不是改名单 —— 因为要测的正是「注册表里没有
    这个技能」这件事本身能不能被看见。"""
    from mast.core.registry import SkillRegistry
    reg = SkillRegistry()
    reg.discover("mast.skills.builtins", "mast.skills.composite")
    assert reg.unregister("GetLatestScanFile") is True

    res = TipConditioningSelfCheck().execute(Ctx(registry=reg), {})
    assert res.data["missing_skills"] == ["GetLatestScanFile"]
    assert res.data["ready"] is False
    assert any("技能齐全" in b for b in res.data["blockers"])


def test_the_two_selfchecks_ask_one_implementation(live_registry):
    """自检与运行使用同一个注册表。"""
    import inspect

    from mast.core.registry import registered_skill_names
    from mast.skills.builtins import (
        scan_intel_selfcheck,
        tip_conditioning_selfcheck,
        tip_forge_selfcheck,
    )
    for mod in (tip_conditioning_selfcheck, tip_forge_selfcheck):
        assert mod.registered_skill_names is registered_skill_names
    assert "registered_skill_names" in source_of(
        scan_intel_selfcheck.ScanIntelSelfCheck._check_registry)

    # 注册表计数应与可调用技能集合一致。
    names = registered_skill_names(live_registry)
    reg_report = scan_intel_selfcheck.ScanIntelSelfCheck._check_registry(
        live_registry)
    assert reg_report["total_registered"] == len(names)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
