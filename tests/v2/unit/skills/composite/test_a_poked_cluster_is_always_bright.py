"""评估本次扎出的凸起时必须显式使用 bright 极性，避免 auto 将背景当作目标。"""
from __future__ import annotations

import ast
import pathlib

import pytest

#: 评「刚扎出来的簇」的调用点。加第四个就要加进这张表 —— 而下面那条
#: 结构闸门会先红,提醒你这件事。
POKE_SITES = (
    "mast/skills/composite/_tip_phases.py",
    "mast/skills/composite/shape_tip_on_surface.py",
    "mast/skills/composite/builtin_composites.py",
)
ROOT = pathlib.Path(__file__).resolve().parents[5] / "MASTv2"


def _sources():
    for rel in POKE_SITES:
        p = ROOT / rel
        assert p.exists(), f"调用点文件不在了:{rel}"
        yield rel, p.read_text(encoding="utf-8")


@pytest.mark.parametrize("rel", POKE_SITES)
def test_every_poked_cluster_call_declares_bright(rel):
    src = (ROOT / rel).read_text(encoding="utf-8")
    n = src.count("AssessClusterRoundness")
    assert n >= 1, f"{rel} 里没有这个调用了 —— 表该更新"
    # 每一处调用附近都要出现 polarity=bright。用 quote 两种写法都认
    # (dict 字面量 "polarity": "bright" 与声明式 JSON 同形)。
    got = src.count('"polarity": "bright"') + src.count("'polarity': 'bright'")
    assert got >= 1, f"{rel} 评刚扎的簇却没声明 polarity=bright"


def test_no_fourth_caller_slipped_in_without_declaring():
    """全仓扫一遍:凡是把 min_axis_ratio 传给 AssessClusterRoundness 的地方,
    都必须同时声明 polarity。

    `min_axis_ratio` 是「我在判这个簇够不够圆」的标志 —— 分析用途的调用
    (data_processing 的工具、别人手动查一张图)不会带它。
    """
    missing = []
    for p in (ROOT / "mast").rglob("*.py"):
        src = p.read_text(encoding="utf-8", errors="replace")
        if "AssessClusterRoundness" not in src or "min_axis_ratio" not in src:
            continue
        if p.name in ("cluster_roundness.py", "cluster_extract.py"):
            continue                      # 技能自己和它的说明文档
        if "noble_tip_workflow" in p.name or "special_tip_workflow" in p.name:
            continue                      # 只是在文档里引用这个名字
        if '"polarity"' not in src and "'polarity'" not in src:
            missing.append(str(p.relative_to(ROOT)))
    assert not missing, (
        "这些地方在判刚扎的簇却没声明 polarity(会吃 auto,可能选中背景):\n  "
        + "\n  ".join(missing))


def test_auto_is_still_the_skill_default_for_everyone_else():
    """技能本身的默认**不改** —— 别的调用方(找坑、分析一张图)确实需要 auto。

    改默认会把这个修复变成一次全局行为改动,而证据只覆盖「刚扎出来的簇」
    这一种用途。修在调用点,不修在默认值。
    """
    from mast.skills.builtins.cluster_roundness import AssessClusterRoundness

    spec = next(p for p in AssessClusterRoundness().metadata().parameters
                if p.name == "polarity")
    assert spec.default == "auto"
    assert "bright" in (spec.description or ""), "说明里得写清 bright 是给凸起的"


def test_generated_cluster_assessment_keeps_the_poked_target_and_bright_polarity():
    """用合成步骤回包验证本次落点与明极性一路传到判读步骤；不执行任何技能。"""
    from types import SimpleNamespace

    from mast.core.noble_tip_workflow import NobleTipWorkflow
    from mast.skills.composite import _tip_phases

    prefix = "synthetic_poke"
    scan_path = "synthetic_cluster.sxm"
    executor = SimpleNamespace(sub_results={})
    wf = NobleTipWorkflow()
    steps = _tip_phases._cluster_look(
        executor, wf, step_prefix=prefix, x=-23e-9, y=41e-9)

    scan = next(steps)
    assert scan.skill_name == "ScanAt"
    assert scan.params["center_x_m"] == -23e-9
    assert scan.params["center_y_m"] == 41e-9
    executor.sub_results[scan.step_id] = SimpleNamespace(data={})

    save = next(steps)
    assert save.skill_name == "SaveScan"
    executor.sub_results[save.step_id] = SimpleNamespace(data={"path": scan_path})
    assert next(steps).skill_name == "GetLatestScanFile"

    assessment = next(steps)
    assert assessment.skill_name == "AssessClusterRoundness"
    assert assessment.params["scan_path"] == scan_path
    assert assessment.params["polarity"] == "bright"
    assert assessment.params["select"] == "center"
    assert assessment.params["min_axis_ratio"] == wf.min_axis_ratio
    steps.close()


def test_failed_cluster_scan_never_produces_an_assessment():
    """合成扫描失败时不得拿未知帧评估针尖。"""
    from types import SimpleNamespace

    from mast.core.noble_tip_workflow import NobleTipWorkflow
    from mast.skills.composite import _tip_phases

    steps = _tip_phases._cluster_look(
        SimpleNamespace(sub_results={}), NobleTipWorkflow(),
        step_prefix="synthetic_failure", x=0.0, y=0.0)
    assert next(steps).skill_name == "ScanAt"
    with pytest.raises(StopIteration) as stopped:
        next(steps)
    assert stopped.value.value["ok"] is False
    assert stopped.value.value["reason"]


def test_it_parses():
    """三个文件改完还得是合法 Python(声明式那份是嵌在字面量里的)。"""
    for rel, src in _sources():
        ast.parse(src)
