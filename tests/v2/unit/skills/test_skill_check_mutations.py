"""合规判据的变异验证：每一个代号都要证明自己**会红**，而且只在该红的时候红。

好样本取 contrib 里的两个范例（``EstimateScanDuration`` 的 skill.py、``ReadJunctionState``
的 spec.json）。每一行变异断言三件事：

1. **文本确实变了** —— 锚点在好样本里找得到，变异后的文本与原文不同（否则「变红」可能
   来自别处，「没变红」也可能只是变异没落地）；
2. **期望的代号变红**；
3. **其余代号不变** —— 好样本零 FAIL，所以变异后的 FAIL 代号集合必须**恰好**等于期望集合。
   有两行天然牵连两个代号（动词藏进变量 ⇒ 足迹也看不透；SkillResult 关键字写错 ⇒
   冒烟执行也会抛），期望集合里明写。

最后用子进程真跑一次命令行：好目录退出 0，坏拷贝退出 1。

从仓库根跑::

    .venv-v2-py313/Scripts/python.exe -m pytest tests/v2/unit/skills/test_skill_check_mutations.py -q
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

import copy  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from typing import Callable  # noqa: E402

import pytest  # noqa: E402

from mast.core.types import ParameterSpec, SafetyLevel, SkillCategory, SkillMetadata, SkillResult  # noqa: E402
from mast.skills.base import BaseSkill  # noqa: E402
from mast.skills.compliance import (  # noqa: E402
    CODES,
    FAIL,
    WARN,
    _nanonis_methods,
    check_contrib_dir,
    check_python_source,
    check_spec,
    default_registry,
)

REPO = Path(__file__).resolve().parents[4]
PY_DIR = REPO / "contrib" / "skills" / "EstimateScanDuration"
SPEC_DIR = REPO / "contrib" / "skills" / "ReadJunctionState"
GOOD_SRC = (PY_DIR / "skill.py").read_text(encoding="utf-8")
GOOD_SPEC = json.loads((SPEC_DIR / "spec.json").read_text(encoding="utf-8"))
GOOD_NAME = "EstimateScanDuration.py"

#: execute() 的第一行 —— 往执行体里插语句用的锚点
_EXEC = "        try:\n            width = parse_quantity("


def _in_execute(*lines: str) -> tuple[str, str]:
    return _EXEC, "".join(f"        {ln}\n" for ln in lines) + _EXEC


def _fails(rep) -> set[str]:
    return {f.code for f in rep.findings if f.level == FAIL}


@dataclass
class PyMut:
    id: str
    edits: list[tuple[str, str]]
    expect: set[str]
    filename: str = GOOD_NAME
    needs_nanonis: bool = False


PY_MUTATIONS: list[PyMut] = [
    PyMut("S01-no-safety-level", [("            safety_level=SafetyLevel.AUTO,\n", "")], {"S01"}),
    PyMut("S02-two-skill-classes",
          [('    return f"{minutes} min {secs:02d} s"\n',
            '    return f"{minutes} min {secs:02d} s"\n\n\nclass SecondSkill(BaseSkill):\n'
            "    def metadata(self):\n        return None\n\n"
            "    def execute(self, context, params):\n        return None\n")],
          {"S02"}),
    PyMut("S03-forbidden-import",
          [("from mast.skills.base import BaseSkill\n", "from mast.skills.base import BaseSkill\nimport os\n")],
          {"S03"}),
    PyMut("S04-name-drifts-from-class-and-file",
          [('_NAME = "EstimateScanDuration"', '_NAME = "EstimateFrameTime"')], {"S04"}),
    PyMut("S05-takes-a-builtin-name",
          [('_NAME = "EstimateScanDuration"', '_NAME = "GetBias"'),
           ("class EstimateScanDuration(BaseSkill):", "class GetBias(BaseSkill):")],
          {"S05"}, filename="GetBias.py"),
    PyMut("P01-dimensioned-without-max", [("                    max_value=1e-5,\n", "")], {"P01"}),
    PyMut("P02-teaches-exponent", [("如 '50n'、'1u'", "如 5e-8")], {"P02"}),
    PyMut("P03-unknown-precondition", [("preconditions=[],", 'preconditions=["tip_is_sharp"],')], {"P03"}),
    PyMut("P04-pulse-without-capability",
          [("category=SkillCategory.ANALYSIS,", "category=SkillCategory.WRITE,"),
           _in_execute('context.safe_call("Bias_Pulse", 0.1, 0.05, 1, 0, 0)')],
          {"P04"}),
    PyMut("V01-verb-in-a-variable",
          [_in_execute('verb = "Bias_Get"', "context.safe_call(verb)")],
          {"V01", "V03"}),                         # 动词看不见 ⇒ 足迹也无从分类
    PyMut("V02-verb-not-in-library",
          [("category=SkillCategory.ANALYSIS,", "category=SkillCategory.READ,"),
           _in_execute('context.safe_call("Bias_Gett")')],
          {"V02"}, needs_nanonis=True),
    PyMut("V03-analysis-reads-hardware", [_in_execute('context.safe_call("Bias_Get")')], {"V03"}),
    PyMut("V03-context-escapes", [_in_execute("id(context)")], {"V03"}),
    PyMut("V03-imports-the-vendor-library",       # 绕开执行上下文、直接拿厂商库
          [("from mast.skills.base import BaseSkill\n", "from mast.skills.base import BaseSkill\nimport nanonis_spm\n")],
          {"V03"}),
    PyMut("R01-bad-skillresult-kwarg", [("            summary=(\n", "            message=(\n")],
          {"R01", "X01"}),                         # 调用那一刻抛 TypeError ⇒ 冒烟也红
    PyMut("R02-stringified-envelope",
          [('"frame_time_s": frame_s,', '"raw": str(params),\n                "frame_time_s": frame_s,')],
          {"R02"}),
    PyMut("X01-execute-raises", [_in_execute('raise RuntimeError("冒烟应当抓到这个异常")')], {"X01"}),
]


def _apply(src: str, edits: list[tuple[str, str]]) -> str:
    out = src
    for old, new in edits:
        assert out.count(old) == 1, f"锚点在样本里出现 {out.count(old)} 次（要求恰好 1 次）：{old!r}"
        out = out.replace(old, new)
    assert out != src, "变异没有改动任何文本"
    return out


@pytest.fixture(scope="module")
def registry():
    reg, env = default_registry()
    assert reg is not None, [f.message for f in env]
    return reg


# ── 好样本：零 FAIL，而且是「全都查过了」的零 ─────────────────────────────────

def test_good_python_sample_is_clean(registry) -> None:
    rep = check_python_source(GOOD_SRC, filename=GOOD_NAME, run_smoke=True)
    assert rep.ok, rep.render(verbose=True)
    ran = set(rep.checks_run)
    expected = {"S01", "S02", "S03", "S04", "S05", "P01", "P02", "P03", "P04",
                "V01", "V02", "V03", "V04", "R01", "R02", "X01", "E01"}
    assert expected <= ran, f"好样本上没评估的判据：{sorted(expected - ran)}"
    assert rep.footprint == "pure-analysis"
    assert rep.skill_name == "EstimateScanDuration"


def test_good_spec_sample_is_clean(registry) -> None:
    rep = check_spec(copy.deepcopy(GOOD_SPEC), registry=registry)
    assert rep.ok, rep.render(verbose=True)
    assert {"S01", "S04", "S05", "C01", "V03", "E01"} <= set(rep.checks_run)
    assert rep.footprint == "hardware-read-only"


# ── Python 技能的变异 ──────────────────────────────────────────────────────

@pytest.mark.parametrize("mut", PY_MUTATIONS, ids=[m.id for m in PY_MUTATIONS])
def test_python_mutation_turns_exactly_the_expected_codes_red(mut: PyMut, registry) -> None:
    if mut.needs_nanonis and _nanonis_methods() is None:
        pytest.skip("nanonis_spm 未安装：命令名存在性（V02）无从核对 —— 这一行没有验证，不是通过")
    mutated = _apply(GOOD_SRC, mut.edits)
    rep = check_python_source(mutated, filename=mut.filename, run_smoke=True)
    assert _fails(rep) == mut.expect, (
        f"期望恰好 {sorted(mut.expect)} 变红，实际 {sorted(_fails(rep))}\n" + rep.render(verbose=True))


def test_write_without_readback_only_warns(registry) -> None:
    """V04 是启发式：只提醒，不拦。"""
    mutated = _apply(GOOD_SRC, [("category=SkillCategory.ANALYSIS,", "category=SkillCategory.WRITE,"),
                                _in_execute('context.safe_call("Bias_Set", 0.1)')])
    rep = check_python_source(mutated, filename=GOOD_NAME, run_smoke=True)
    assert _fails(rep) == set(), rep.render(verbose=True)
    assert "V04" in {f.code for f in rep.findings if f.level == WARN}, rep.render(verbose=True)
    assert rep.footprint == "hardware-write"
    assert rep.verbs == ["Bias_Set"]


class _EmptyRegistry:
    def has(self, name: str) -> bool:
        return False

    def get(self, name: str, version: str | None = None):
        raise KeyError(name)

    def list_skills(self) -> list:
        return []


def test_an_empty_registry_is_an_environment_failure_not_a_pass() -> None:
    """注册表是空的 ⇒ 撞名无从判断。这必须是 FAIL（E01），不能是「没撞名」。"""
    rep = check_python_source(GOOD_SRC, filename=GOOD_NAME, registry=_EmptyRegistry())
    assert _fails(rep) == {"E01"}, rep.render(verbose=True)
    assert any(s.startswith("S05") for s in rep.skipped), rep.skipped


# ── 组合 spec 的变异 ─────────────────────────────────────────────────────────

class _CustomProbe(BaseSkill):
    """一个「自定义来源」的只读技能 —— 模块名落在 mast.skills.custom 之下。"""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(name="ContribProbeRead", version="1.0.0", category=SkillCategory.READ,
                             safety_level=SafetyLevel.AUTO, description="测试用", parameters=[])

    def execute(self, context, params: dict) -> SkillResult:
        return SkillResult(skill_name="ContribProbeRead", success=True)


_CustomProbe.__module__ = "mast.skills.custom.contrib_probe"


class _WithCustomSkill:
    """在真注册表之上多挂一个自定义来源的技能（不改动进程共享的那份注册表）。"""

    def __init__(self, base) -> None:
        self._base = base
        self._extra = {"ContribProbeRead": _CustomProbe}

    def has(self, name: str) -> bool:
        return name in self._extra or self._base.has(name)

    def get(self, name: str, version: str | None = None):
        if name in self._extra:
            return self._extra[name]
        return self._base.get(name, version)

    def _get_metadata(self, cls):
        if cls in self._extra.values():
            return cls().metadata()
        return self._base._get_metadata(cls)

    def list_skills(self):
        return list(self._base.list_skills()) + [_CustomProbe().metadata()]


def _spec_edit(fn: Callable[[dict], None]) -> dict:
    s = copy.deepcopy(GOOD_SPEC)
    fn(s)
    assert s != GOOD_SPEC, "变异没有改动 spec"
    return s


@dataclass
class SpecMut:
    id: str
    edit: Callable[[dict], None]
    expect: set[str]
    custom_registry: bool = False


def _rename_step(skill: str, to: str) -> Callable[[dict], None]:
    def fn(s: dict) -> None:
        hits = [n for n in s["nodes"] if n.get("skill") == skill]
        assert len(hits) == 1, f"spec 里 {skill} 出现 {len(hits)} 次"
        hits[0]["skill"] = to
    return fn


SPEC_MUTATIONS: list[SpecMut] = [
    SpecMut("S01-spec-no-safety-level", lambda s: s.pop("safety_level"), {"S01"}),
    SpecMut("S04-spec-bad-name", lambda s: s.__setitem__("name", "Read-Junction"), {"S04"}),
    SpecMut("S05-spec-takes-a-builtin-name", lambda s: s.__setitem__("name", "GetBias"), {"S05"}),
    SpecMut("C01-unknown-step", _rename_step("GetCurrent", "GetCurrentTypo"), {"C01"}),
    SpecMut("C01-pure-alias", lambda s: s.__setitem__("nodes", s["nodes"][:1]), {"C01"}),
    SpecMut("C01-non-official-step", _rename_step("GetCurrent", "ContribProbeRead"), {"C01"},
            custom_registry=True),
]


@pytest.mark.parametrize("mut", SPEC_MUTATIONS, ids=[m.id for m in SPEC_MUTATIONS])
def test_spec_mutation_turns_exactly_the_expected_codes_red(mut: SpecMut, registry) -> None:
    reg = _WithCustomSkill(registry) if mut.custom_registry else registry
    if mut.custom_registry:
        # 先证明好样本在这份加了料的注册表上照样干净 —— 红只能来自变异本身
        assert check_spec(copy.deepcopy(GOOD_SPEC), registry=reg).ok
    rep = check_spec(_spec_edit(mut.edit), registry=reg)
    assert _fails(rep) == mut.expect, (
        f"期望恰好 {sorted(mut.expect)} 变红，实际 {sorted(_fails(rep))}\n" + rep.render(verbose=True))


# ── contrib 目录（manifest）的变异：在临时拷贝上改 ───────────────────────────

def _edit_manifest(fn: Callable[[dict], None]) -> Callable[[Path], None]:
    def op(d: Path) -> None:
        p = d / "manifest.json"
        m = json.loads(p.read_text(encoding="utf-8"))
        before = copy.deepcopy(m)
        fn(m)
        assert m != before, "manifest 变异没有改动内容"
        p.write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
    return op


def _edit_file(name: str, old: str, new: str) -> Callable[[Path], None]:
    def op(d: Path) -> None:
        p = d / name
        text = p.read_text(encoding="utf-8")
        p.write_text(_apply(text, [(old, new)]), encoding="utf-8")
    return op


def _add_file(name: str, content: str) -> Callable[[Path], None]:
    def op(d: Path) -> None:
        assert not (d / name).exists()
        (d / name).write_text(content, encoding="utf-8")
    return op


@dataclass
class DirMut:
    id: str
    source: Path
    op: Callable[[Path], None]
    expect: set[str] = field(default_factory=set)


DIR_MUTATIONS: list[DirMut] = [
    DirMut("M01-no-license", PY_DIR, _edit_manifest(lambda m: m.pop("license")), {"M01"}),
    DirMut("M01-unknown-verification-level", PY_DIR,
           _edit_manifest(lambda m: m.__setitem__("verification", "maintainer-verified")), {"M01"}),
    DirMut("M02-manifest-name-drifts", PY_DIR, _edit_manifest(lambda m: m.__setitem__("name", "Other")), {"M02"}),
    DirMut("M02-version-drifts", PY_DIR, _edit_manifest(lambda m: m.__setitem__("version", "1.0.1")), {"M02"}),
    DirMut("M02-extra-module-would-not-be-installed", PY_DIR, _add_file("helpers.py", "X = 1\n"), {"M02"}),
    DirMut("M02-spec-safety-drifts-from-effective", SPEC_DIR,
           _edit_manifest(lambda m: m.__setitem__("safety_level", "confirm")), {"M02"}),
    DirMut("M03-paper-port-doi", PY_DIR,
           _edit_file("skill.py", '_NAME = "EstimateScanDuration"\n',
                      '_NAME = "EstimateScanDuration"\n# 方法见 doi:10.1103/PhysRevLett.49.57\n'),
           {"M03"}),
    DirMut("M03-policy-not-attested", PY_DIR,
           _edit_manifest(lambda m: m["policy"].__setitem__("original_work", False)), {"M03"}),
    DirMut("V03-manifest-footprint-disagrees", PY_DIR,
           _edit_manifest(lambda m: m.__setitem__("footprint", "hardware-write")), {"V03"}),
    DirMut("V03-spec-manifest-footprint-disagrees", SPEC_DIR,
           _edit_manifest(lambda m: m.__setitem__("footprint", "pure-analysis")), {"V03"}),
]


def _copy(src: Path, tmp_path: Path) -> Path:
    dst = tmp_path / src.name
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
    return dst


@pytest.mark.parametrize("source", [PY_DIR, SPEC_DIR], ids=["python", "spec"])
def test_good_contrib_dirs_are_clean(source: Path, tmp_path: Path, registry) -> None:
    rep = check_contrib_dir(_copy(source, tmp_path))
    assert rep.ok, rep.render(verbose=True)


@pytest.mark.parametrize("mut", DIR_MUTATIONS, ids=[m.id for m in DIR_MUTATIONS])
def test_contrib_dir_mutation_turns_exactly_the_expected_codes_red(mut: DirMut, tmp_path: Path, registry) -> None:
    d = _copy(mut.source, tmp_path)
    mut.op(d)
    rep = check_contrib_dir(d)
    assert _fails(rep) == mut.expect, (
        f"期望恰好 {sorted(mut.expect)} 变红，实际 {sorted(_fails(rep))}\n" + rep.render(verbose=True))


def test_every_code_has_a_mutation_that_turns_it_red() -> None:
    """代号表里的每一条都要有变异证明它会红（V04 只提醒、E01 是环境，各有专门的一条）。"""
    covered = set().union(*(m.expect for m in PY_MUTATIONS), *(m.expect for m in SPEC_MUTATIONS),
                          *(m.expect for m in DIR_MUTATIONS))
    covered |= {"V04", "E01"}
    assert covered == set(CODES), f"没有变异覆盖的代号：{sorted(set(CODES) - covered)}"


# ── 命令行：真跑一次 ─────────────────────────────────────────────────────────

def _cli(*args: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run([sys.executable, str(REPO / "scripts" / "skill_check.py"), *args],
                          cwd=str(REPO), env=env, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=600)


def test_cli_passes_a_good_contribution() -> None:
    proc = _cli("contrib/skills/EstimateScanDuration")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "[PASS] contrib/skills/EstimateScanDuration" in proc.stdout, proc.stdout


def test_cli_fails_a_bad_copy_and_says_why(tmp_path: Path) -> None:
    d = _copy(PY_DIR, tmp_path)
    _edit_file("skill.py", "from mast.skills.base import BaseSkill\n",
               "from mast.skills.base import BaseSkill\nimport os\n")(d)
    proc = _cli(str(d), "--json")
    assert proc.returncode == 1, proc.stdout + proc.stderr
    reports = json.loads(proc.stdout)
    assert len(reports) == 1 and reports[0]["ok"] is False
    assert "S03" in {f["code"] for f in reports[0]["findings"] if f["level"] == "FAIL"}


def test_cli_fails_on_a_path_that_does_not_exist(tmp_path: Path) -> None:
    proc = _cli(str(tmp_path / "NoSuchSkill"))
    assert proc.returncode == 1, proc.stdout + proc.stderr
