"""contrib/skills/ 下的每一个投稿都必须过合规判据，而且它们自带的测试必须真的跑过。

``pyproject.toml`` 的 testpaths 只有 ``tests/v2``，``contrib/`` 里的测试不会被顺带收集 ——
不在这里用子进程跑一遍，投稿自带的测试就只是摆设。

判据本体：``MASTv2/mast/skills/compliance.py``；命令行：``scripts/skill_check.py``。

从仓库根跑::

    .venv-v2-py313/Scripts/python.exe -m pytest tests/v2/unit/skills/test_contrib_compliance.py -q
"""
from __future__ import annotations

# ── path setup BEFORE any mast.* imports（同 tests/v2/unit/test_wrap_skill_minimal.py）──
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

import json  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402

import pytest  # noqa: E402

from mast.skills.compliance import check_contrib_dir  # noqa: E402

REPO = Path(__file__).resolve().parents[4]
CONTRIB = REPO / "contrib" / "skills"


def _skill_dirs() -> list[Path]:
    if not CONTRIB.is_dir():
        return []
    return sorted(d for d in CONTRIB.iterdir() if d.is_dir() and not d.name.startswith((".", "_")))


SKILL_DIRS = _skill_dirs()


def test_the_sweep_finds_contributions_of_both_kinds() -> None:
    """先证明遍历扫到了东西 —— 一个恒空的遍历会让下面每条断言都白过。"""
    assert len(SKILL_DIRS) >= 2, f"contrib/skills/ 下只找到 {[d.name for d in SKILL_DIRS]}"
    kinds = {json.loads((d / "manifest.json").read_text(encoding="utf-8")).get("kind")
             for d in SKILL_DIRS if (d / "manifest.json").is_file()}
    assert {"python", "spec"} <= kinds, f"两个范例（Python 技能 + 组合 spec）都该在，实际 kind={kinds}"


@pytest.mark.parametrize("skill_dir", SKILL_DIRS, ids=[d.name for d in SKILL_DIRS])
def test_every_contribution_passes_the_checker(skill_dir: Path) -> None:
    rep = check_contrib_dir(skill_dir)
    assert rep.ok, "\n" + rep.render(verbose=True)
    # 零 FAIL 必须是「查过了」，不是「没查」：manifest 三项 + 本体判据都在 checks_run 里
    assert {"M01", "M02", "M03", "S01", "S04", "S05", "V03"} <= set(rep.checks_run), rep.checks_run
    assert rep.footprint in ("pure-analysis", "hardware-read-only", "hardware-write")


def test_test_file_names_are_unique_across_contrib() -> None:
    """pytest 在没有 __init__.py 的目录里按文件名导入测试模块：两个 test_skill.py 会互相顶掉。"""
    seen: dict[str, str] = {}
    dupes = []
    for d in SKILL_DIRS:
        for t in d.glob("test_*.py"):
            if t.name in seen:
                dupes.append(f"{t.name}: {seen[t.name]} / {d.name}")
            seen[t.name] = d.name
    assert seen, "一个 contrib 测试文件都没找到"
    assert not dupes, "contrib 里的测试文件名必须全局唯一：\n  " + "\n  ".join(dupes)


def test_skill_names_are_unique_across_contrib() -> None:
    names = [d.name for d in SKILL_DIRS]
    assert len(names) == len({n.lower() for n in names}), f"大小写不敏感地撞名：{names}"


def test_contribution_tests_actually_run_and_pass() -> None:
    """用子进程跑 ``pytest contrib``：它们不在 testpaths 里，不这样跑就永远不会被执行。"""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (_MASTV2_ROOT, env.get("PYTHONPATH", "")) if p)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    n_files = sum(1 for d in SKILL_DIRS for _ in d.glob("test_*.py"))
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", str(REPO / "contrib"), "-q", "-p", "no:cacheprovider"],
        cwd=str(REPO), env=env, capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=900,
    )
    tail = (proc.stdout + proc.stderr)[-3000:]
    assert proc.returncode == 0, f"contrib 自带测试没过（退出码 {proc.returncode}）：\n{tail}"
    m = re.search(r"(\d+) passed", proc.stdout)
    passed = int(m.group(1)) if m else 0
    # 防空转：至少每个测试文件贡献一条通过（一个都没收集到时 pytest 退出码是 5，但别只靠它）
    assert passed >= max(n_files, 2), f"只跑过 {passed} 条（{n_files} 个测试文件）：\n{tail}"
    assert not re.search(r"\d+ (failed|error)", proc.stdout), tail
