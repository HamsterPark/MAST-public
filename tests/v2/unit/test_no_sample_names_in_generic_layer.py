"""通用层样品名隔离：固定白名单之外禁止出现特定样品标识符，合成注入验证检测能力。"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

MAST_ROOT = Path(__file__).resolve().parents[3] / "MASTv2" / "mast"

# 自定义边界而不是 \b：下划线要算分隔符，否则 `WOI_SYMMETRY_DEG` 这种
# 最该抓的样品名常量会漏网（\b 把 _ 当词内字符——本测试的自检当场抓过这一漏）。
SAMPLE_TOKEN = re.compile(
    r"(?<![A-Za-z0-9])(wo2i2|woi2?|au111)(?![A-Za-z0-9])", re.IGNORECASE
)

# 永远允许的目录前缀：实验模板层（conduct 落地前可不存在）+ 知识库
# （策展 paper 级信息，样品指涉是它的本职——knowledge_no_feedback_loop 纪律）
ALLOWED_DIR_PREFIXES = ("conduct/templates/", "knowledge/")

# 固定白名单仅允许公开通用名称与示例；新增条目必须说明理由。
# 扫描仍检查白名单外的代码，也检查已失效的豁免。
ALLOWLIST: dict[str, str] = {
    'core/sample_facts.py': '材料名规范化的回归示例',
    'wishlist/store.py': '清理逻辑的示例名称',
    'documents/__init__.py': '示例文件名、路径或对象标识',
    'agents/data_processing/tools.py': '示例文件名、路径或对象标识',
    'skills/composite/forge_au_tip.py': '公开通用的 Au(111) 修针技能名称与标签',
    'skills/builtins/herringbone_assess.py': 'Au(111) 已知表面重构的检测器名称与标签',
    'vision/double_tip.py': '已知 Au(111) 台阶物理常量，允许用户覆盖',
    'agents/_shared/artifacts.py': '示例文件名、路径或对象标识',
    'agents/paper_review/tools.py': '示例文件名、路径或对象标识',
    'agents/paper_writing/tools.py': '示例文件名、路径或对象标识',
    'api/routes/agents_topology.py': '示例文件名、路径或对象标识',
    'api/routes/documents.py': '示例文件名、路径或对象标识',
    'api/schemas_literature.py': '示例文件名、路径或对象标识',
    'api/schemas_literature_cognition.py': '示例文件名、路径或对象标识',
    'core/scan_registry.py': '示例文件名、路径或对象标识',
    'logging/v2/filestore.py': '示例文件名、路径或对象标识',
    'core/tip_intent.py': '通用流程设计文档名',
    'skills/composite/__init__.py': '通用流程设计文档名',
}


def _scan(root: Path) -> dict[str, list[tuple[int, str]]]:
    """返回 {相对路径: [(行号, 命中行)]}，只扫 .py。"""
    hits: dict[str, list[tuple[int, str]]] = {}
    for py in sorted(root.rglob("*.py")):
        rel = py.relative_to(root).as_posix()
        try:
            text = py.read_text(encoding="utf-8")
        except UnicodeDecodeError:  # pragma: no cover — 仓内不该有，防御
            continue
        found = [
            (i, line.strip()[:120])
            for i, line in enumerate(text.split("\n"), 1)
            if SAMPLE_TOKEN.search(line)
        ]
        if found:
            hits[rel] = found
    return hits


def _violations(hits: dict[str, list[tuple[int, str]]]) -> dict[str, list[tuple[int, str]]]:
    out = {}
    for rel, found in hits.items():
        if rel in ALLOWLIST:
            continue
        if any(rel.startswith(p) for p in ALLOWED_DIR_PREFIXES):
            continue
        out[rel] = found
    return out


def test_generic_layer_has_no_sample_names():
    assert MAST_ROOT.is_dir(), f"mast 根不存在: {MAST_ROOT}"
    violations = _violations(_scan(MAST_ROOT))
    msg = "\n".join(
        f"  {rel}:{ln}: {line}" for rel, found in sorted(violations.items()) for ln, line in found
    )
    assert not violations, (
        "通用层出现样品名（MAST 是 STM 系统不是 WOI 系统——"
        "见 docs/v2/design/stm_capability_vs_sample_layer.md §1）。\n"
        "样品数据请放 profile/DomainReference/conduct 模板层；"
        "若确属 provenance/事故记录，把文件加进本测试 ALLOWLIST 并写出理由：\n" + msg
    )


def test_allowlist_entries_still_exist_and_still_hit():
    """白名单不许腐烂：文件没了或已无命中 ⇒ 从名单里删掉（否则名单会静默扩权）。"""
    hits = _scan(MAST_ROOT)
    stale = [rel for rel in ALLOWLIST if rel not in hits]
    assert not stale, f"白名单条目已无命中，请删除: {stale}"


def test_scanner_catches_a_planted_violation(tmp_path):
    """验证器的验证（变异等价物）：种一个违例，扫描器必须抓到。

    不去动真实文件——在临时树里种，断言同一套 _scan/_violations 逻辑能抓。
    """
    bad = tmp_path / "vision" / "new_detector.py"
    bad.parent.mkdir(parents=True)
    bad.write_text("WOI_SYMMETRY_DEG = 60.0  # planted\n", encoding="utf-8")
    ok = tmp_path / "conduct" / "templates" / "wo2i2.py"
    ok.parent.mkdir(parents=True)
    ok.write_text("SPEC_ID = 'wo2i2_v1'\n", encoding="utf-8")

    violations = _violations(_scan(tmp_path))
    assert "vision/new_detector.py" in violations, "扫描器没抓到种下的违例——检查器坏了"
    assert "conduct/templates/wo2i2.py" not in violations, "模板层被误伤"


def test_token_word_boundary_no_false_positive():
    """`woi` 的词边界不许把普通词（avoid/awoken 等）误伤。"""
    for word in ("avoid", "awoken", "twoImages", "showing"):
        assert not SAMPLE_TOKEN.search(word), f"误伤: {word}"
    for tok in ("WOI", "wo2i2", "WOI2", "Au111", "au111",
                "WOI_SYMMETRY_DEG", "expected_wo2i2_a_nm", "_au111_onset"):
        assert SAMPLE_TOKEN.search(tok), f"漏抓: {tok}"
