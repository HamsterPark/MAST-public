"""治疗段的三条引擎语义（2026-08-15 补的设计缺口）。

M2-S1 实施时撞出来的:设计假定「修针段只在绕道时跑、段内可以有闸门、绕道进来
先只读诊断」,而引擎三条都表达不出来 —— 不是笔误,是当时没想到的形状。

1. **`entered_only_by_detour`**:正常流程一步都不进它。在这个字段之前只有两个
   位置可选,而两个都荒谬:排最前 ⇒ 每次 conduct 开工先做一整轮换样品修针;
   排最后 ⇒ 实验做完之后再修一次针。
2. **`StepSpec.gate`**:段内闸门。以前只有首尾两个闸位,拆成两个 stage 又会被
   「绕道活跃 + 任意阶段跑完 ⇒ 立刻返回」截断,第二半永远不跑。
3. **绕道进来先只读诊断**:判据从「第 0 步必须是退针技能」改成「退针之前不得
   有会改变表面的动作」,理由与反例见 `test_validator.py`。

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest tests/v2/unit/conduct/test_director_detour_stage.py -x -v
"""
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest  # noqa: E402

from mast.conduct.spec import DetourPolicy  # noqa: E402

from _harness import (  # noqa: E402
    FakeExecutor,
    build,
    outcome,
    retract_step,
    rule_gate,
    spec,
    stage,
    step,
)


def _repair_stage(**kw):
    """治疗段:只读复测 → 确认式退针 → 修。

    复测排在退针**前面**是刻意的,而且是新判据放行的那一种:它只读,不改变表面。
    """
    kw.setdefault("entered_only_by_detour", True)
    return stage("R", steps=(
        step("R.00_recheck", skill="CrossPointTipCheck", produces=("verdict",)),
        retract_step("R.01_retract"),
        step("R.02_forge", skill="ForgeAuTip", produces=("outcome",)),
    ), **kw)


def _spec_with_repair(fail_verdict="detour", **stage_kw):
    # 触发源**只声明 gate_verdict**。恢复自检那条触发源今天还没有生产方
    # (没有任何代码发得出它),而这里的测试一条都驱动不了它 —— 声明一个驱动
    # 不了的触发源,等于让这个文件看起来在测一条它没测的路。
    # 那条路今天的保障在**结构层**:校验器的 detour_writes_before_retract
    # 拦住「刚崩过的针继续往表面上开」,见 test_validator.py。
    return spec(
        [_repair_stage(**stage_kw),
         stage("M", steps=(step("M.01", produces=("n",)),),
               exit_gate=rule_gate("g", selector="M.01",
                                   fail_verdict=fail_verdict))],
        detour=DetourPolicy(target_stage="R",
                            triggers=frozenset({"gate_verdict"})))


# ══════════════════════════════════════════════════════════════════════
# 1. 正常流程一步都不进治疗段
# ══════════════════════════════════════════════════════════════════════

def test_the_normal_flow_never_enters_the_repair_stage(tmp_path):
    """**这条是 `entered_only_by_detour` 存在的全部理由。**

    治疗段必须待在 ``stages`` 里(绕道要按 stage_index 找得到它),而它排在最前。
    没有这个字段,每一次 conduct 开工都会先做一整轮换样品修针 —— 两次人工换
    样品、几小时等待,在一根好针上。
    """
    ex = FakeExecutor({"ScanAt": [outcome(ok=True, data={"n": 5})]})
    rig = build(tmp_path, _spec_with_repair(), executor=ex)
    for _ in range(8):
        rig.tick()
    called = {c[0] for c in ex.calls}
    assert "CrossPointTipCheck" not in called, f"正常流程跑进了治疗段:{called}"
    assert "ForgeAuTip" not in called
    assert rig.row()["status"] in ("running", "completed")


def test_a_conduct_does_not_start_inside_the_repair_stage(tmp_path):
    """起点也要跳过它 —— ``adopted`` 是硬编码 stage_idx=0 的那一处。"""
    rig = build(tmp_path, _spec_with_repair(),
                executor=FakeExecutor({"ScanAt": [outcome(ok=True, data={"n": 5})]}))
    rig.tick()
    assert rig.row()["stage_idx"] == 1, "conduct 从治疗段开工了"


# ══════════════════════════════════════════════════════════════════════
# 2. 绕道进得来,而且第一步是只读诊断
# ══════════════════════════════════════════════════════════════════════

def _drive_into_detour(tmp_path, ex, **kw):
    rig = build(tmp_path, _spec_with_repair(**kw), executor=ex)
    for _ in range(10):
        rig.tick()
        if rig.row()["detour"]:
            break
    return rig


def test_a_detour_still_reaches_the_repair_stage(tmp_path):
    ex = FakeExecutor({"ScanAt": [outcome(ok=True, data={"n": 0})]})
    rig = _drive_into_detour(tmp_path, ex)
    row = rig.row()
    assert row["detour"], "闸门判坏了却没进绕道"
    assert row["stage_idx"] == 0 and row["step_idx"] == 0
    assert row["evidence_epoch"] == 1, "进 detour 没有 bump 证据代次"


def test_the_detour_runs_the_read_only_recheck_before_the_retract(tmp_path):
    """**「怀疑」与「处置」分开计价** —— 先花 15 分钟只读复查确认真是针坏了,
    再付换样品修针的钱。这一步以前被旧判据结构性地挡在外面。"""
    ex = FakeExecutor({"ScanAt": [outcome(ok=True, data={"n": 0})]})
    rig = _drive_into_detour(tmp_path, ex)
    for _ in range(4):
        rig.tick()
    names = [c[0] for c in ex.calls]
    assert "CrossPointTipCheck" in names, f"绕道没跑复测:{names}"
    assert names.index("CrossPointTipCheck") < names.index("SafeRetract"), (
        f"复测排到退针后面去了 —— 退了针就没法复测:{names}")


def test_the_repair_stage_is_safe_for_a_tip_that_may_have_just_crashed(tmp_path):
    """绕道的触发源**不止**「闸门判坏」这一条 —— 恢复自检判针坏也会进来,
    而那根针可能刚崩过。所以「退针之前只许只读」这条不是给正常路径定的礼节,
    是给最坏那条路定的下限。

    ⚠️ 这里**不驱动**那条触发源:它今天还没有生产方(没有任何代码发得出它),
    驱动一条发不出来的路等于让测试看起来覆盖了它没覆盖的东西。它今天的保障
    在结构层 —— 校验器的 ``detour_writes_before_retract``(见 test_validator.py
    的 TipShape/BiasPulse 两个负例)。这条测试只钉一件事:**这一段的排布本身
    经得起那条最坏路径**,与它是被哪个触发源带进来的无关。
    """
    from mast.conduct import validator as V

    s = _spec_with_repair()
    skills = {"CrossPointTipCheck": _ReadOnly(), "SafeRetract": _ReadOnly(),
              "ForgeAuTip": _Shaper(), "ScanAt": _ReadOnly()}
    findings = [f.code for f in V.validate_spec(s, skills=skills).findings]
    assert "detour_writes_before_retract" not in findings, findings
    # 而且把顺序颠倒(修针排到退针前面)必须被拦下来 —— 否则上面那条是空的。
    bad = spec([stage("R", steps=(step("R.00_forge", skill="ForgeAuTip"),
                                  retract_step("R.01"))),
                stage("M", steps=(step("M.01", produces=("n",)),))],
               detour=DetourPolicy(target_stage="R",
                                   triggers=frozenset({"gate_verdict"})))
    assert "detour_writes_before_retract" in [
        f.code for f in V.validate_spec(bad, skills=skills).findings]


class _ReadOnly:
    class _L:
        value = "auto"
    safety_level = _L()
    capabilities = frozenset()
    parameters = ()


class _Shaper:
    class _L:
        value = "confirm"
    safety_level = _L()
    capabilities = frozenset({"tip_shaping"})
    parameters = ()


def test_the_detour_returns_only_after_the_repair_stage_itself_finishes(tmp_path):
    """以前是「绕道活跃 + **任意**阶段跑完 ⇒ 立刻返回」。那让治疗段一旦拆成
    两个 stage,第二个永远不跑,而且没有任何地方会说话。"""
    ex = FakeExecutor({"ScanAt": [outcome(ok=True, data={"n": 0})]})
    rig = _drive_into_detour(tmp_path, ex)
    assert rig.row()["detour"]["return_stage_idx"] == 1
    for _ in range(12):
        rig.tick()
        if not rig.row()["detour"]:
            break
    assert rig.row()["stage_idx"] == 1, "回来之后没有落在原来那个阶段上"


# ══════════════════════════════════════════════════════════════════════
# 3. 段内闸门
# ══════════════════════════════════════════════════════════════════════

def test_a_step_level_gate_stops_the_stage_right_after_that_step(tmp_path):
    """修完没成就该停,不该继续走完「换回样品 → 等人 → 进针 → 复验」再说不行 ——
    那是白烧一次用户往返。"""
    gate = rule_gate("forge_gate", field_name="outcome", op="==", value="ready",
                     selector="R.02_forge", fail_verdict="wait_operator")
    repair = stage("R", steps=(
        step("R.00_recheck", skill="CrossPointTipCheck", produces=("verdict",)),
        retract_step("R.01_retract"),
        step("R.02_forge", skill="ForgeAuTip", produces=("outcome",), gate=gate),
        step("R.03_after", skill="ApproachTip"),
    ), entered_only_by_detour=True)
    s = spec([repair, stage("M", steps=(step("M.01", produces=("n",)),),
                            exit_gate=rule_gate("g", selector="M.01",
                                                fail_verdict="detour"))],
             detour=DetourPolicy(target_stage="R"))
    ex = FakeExecutor({"ScanAt": [outcome(ok=True, data={"n": 0})],
                       "ForgeAuTip": [outcome(ok=True,
                                              data={"outcome": "sites_exhausted"})]})
    rig = build(tmp_path, s, executor=ex)
    for _ in range(14):
        rig.tick()
        if rig.row()["status"] == "waiting_operator":
            break
    assert rig.row()["status"] == "waiting_operator", "闸门没拦住"
    assert "ApproachTip" not in {c[0] for c in ex.calls}, (
        "forge 没成还继续往下走了 —— 白烧一次换回样品的往返")


def test_a_passing_step_gate_lets_the_stage_carry_on(tmp_path):
    gate = rule_gate("forge_gate", field_name="outcome", op="==", value="ready",
                     selector="R.02_forge", fail_verdict="wait_operator")
    repair = stage("R", steps=(
        retract_step("R.01_retract"),
        step("R.02_forge", skill="ForgeAuTip", produces=("outcome",), gate=gate),
        step("R.03_after", skill="ApproachTip"),
    ), entered_only_by_detour=True)
    s = spec([repair, stage("M", steps=(step("M.01", produces=("n",)),),
                            exit_gate=rule_gate("g", selector="M.01",
                                                fail_verdict="detour"))],
             detour=DetourPolicy(target_stage="R"))
    ex = FakeExecutor({"ScanAt": [outcome(ok=True, data={"n": 0})],
                       "ForgeAuTip": [outcome(ok=True, data={"outcome": "ready"})]})
    rig = build(tmp_path, s, executor=ex)
    for _ in range(14):
        rig.tick()
    assert "ApproachTip" in {c[0] for c in ex.calls}, "闸门判过了却没放行"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
