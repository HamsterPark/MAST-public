"""导入纪律 —— 每条守着一个「一旦破了就静默」的边界。

大部分用 AST 而不是跑一遍：它们要证明的是**代码里写没写**，不是**这次跑到没跑到**。
一条只在某个分支里出现的 ``import mast.instruments``，行为测试可能永远碰不到，
而它一样是一条通往仪器的路。

但 AST 有个盲区：它只看得见**写在这个包里**的 import，看不见「A 干净、但 A
import 的 B 不干净」。所以最后一条走干净子进程查传递依赖 —— 两种检查各补对方
的洞，缺一个都会留下一整类看不见的路径。
"""

from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

import pytest

PYEXEC = Path(__file__).resolve().parents[4] / "MASTv2" / "mast" / "pyexec"

#: 拷进会话、跑在子进程里的文件。子进程里**根本没有 mast 包**，它们 import 一下
#: 就是一次 ModuleNotFoundError —— 而那会表现成「分析环境坏了」。
CHILD_SIDE = ("child_audit.py", "child_sitecustomize.py", "runtime_helper.py")

HARDWARE_PREFIXES = ("mast.instruments", "mast.core.connection",
                     "mast.core.executor", "mast.core.safety", "nanonis_spm")


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            out.add(node.module)
            out.update(f"{node.module}.{a.name}" for a in node.names)
    return out


@pytest.mark.parametrize("name", CHILD_SIDE)
def test_child_side_files_import_no_mast(name):
    """跑在子进程里的文件不许 import 任何 ``mast``。

    它们住在源码树里（可 lint、可 review、可 diff），但每次运行是被**拷贝**进
    会话目录执行的 —— 那个解释器里没有 MAST。
    """
    bad = sorted(m for m in _imports(PYEXEC / name)
                 if m == "mast" or m.startswith("mast."))
    assert not bad, f"{name} import 了 {bad} —— 子进程里那些不存在"


@pytest.mark.parametrize("name", CHILD_SIDE)
def test_child_side_files_use_only_stdlib_and_numpy(name):
    """第三方依赖只许 numpy。

    这三个文件在**任何**分析运行时上都必须能跑起来 —— 包括一个只装了必需三件套
    的最小运行时。多依赖一个库，就多一个「在这台机器上装不起来」的可能。
    """
    allowed = set(sys.stdlib_module_names) | {"numpy", "_mast_audit", "mastdata"}
    bad = sorted(m for m in _imports(PYEXEC / name)
                 if m.split(".")[0] not in allowed)
    assert not bad, f"{name} 依赖了 {bad}"


def test_staging_uses_only_the_canonical_parser():
    """staging 只许经 ``mast.io.nanonis_files`` 读字节。

    2026-07-20 删掉 ``data/formats.py`` 的纪律：全系统只留一套字节 parser。
    这里一旦冒出第二个 reader（``nanonis_spm``、``pySPM``、``struct.unpack``
    自己拆），两套实现就会各自漂移，而漂移出来的差异没有任何测试在看。
    """
    imports = _imports(PYEXEC / "staging.py")
    parsers = {m for m in imports
               if m.startswith(("mast.io.", "mast.data.")) or m in ("nanonis_spm",)}
    unexpected = parsers - {
        "mast.io.nanonis_files",
        "mast.io.nanonis_files.read_sxm", "mast.io.nanonis_files.read_dat",
        "mast.io.nanonis_files.read_txt", "mast.io.nanonis_files.sxm_frame_meta",
        "mast.io.nanonis_files._CHANNEL_UNIT_HINT",
    }
    assert not unexpected, f"staging.py 引入了第二套 parser：{sorted(unexpected)}"


def test_pyexec_never_imports_hardware():
    """整个 ``mast/pyexec/`` 包不许碰硬件。

    ⚠️ 这个包**不在** agent_boundary 钩子（不随仓） 的射程内（那个 hook 只
    管 ``MASTv2/mast/agents/**``）。所以它的边界是这条测试，没有别的东西在看。
    """
    offenders: dict[str, list[str]] = {}
    for f in sorted(PYEXEC.glob("*.py")):
        bad = sorted(m for m in _imports(f)
                     if any(m.startswith(p) for p in HARDWARE_PREFIXES))
        if bad:
            offenders[f.name] = bad
    assert not offenders, f"pyexec 里出现了硬件 import：{offenders}"


def test_importing_pyexec_pulls_in_no_hardware_modules():
    """``import mast.pyexec`` 的**传递依赖**里不该有硬件模块。

    必须在一个**干净的子进程**里问。第一版直接看父进程的 ``sys.modules``，红了 ——
    但那不是 pyexec 拖进来的，是同一个 session 里跑过的别的测试留下的。
    那个断言问的是「此刻内存里有没有」，而我要问的是「import 它会不会带进来」，
    两件事只是恰好经常一致。

    传递依赖必须单独测：AST 那几条只看得见**写在这个包里**的 import，
    一条「A 干净、但 A import 的 B 不干净」的链它一个字都看不到。
    """
    import json
    import subprocess

    probe = (
        "import sys, json;"
        "import mast.pyexec;"
        "bad=[m for m in sys.modules if m.startswith("
        "('mast.instruments','mast.core.connection','mast.core.executor',"
        "'mast.core.safety'))];"
        "print('RESULT'+json.dumps(sorted(bad)))"
    )
    root = str(Path(__file__).resolve().parents[4] / "MASTv2")
    proc = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True, text=True, timeout=180,
        env={**os.environ, "PYTHONPATH": root},
        creationflags=(0x08000000 if sys.platform == "win32" else 0))
    assert "RESULT" in proc.stdout, (
        f"探测进程没跑起来（exit {proc.returncode}）：{proc.stderr[-600:]}")
    bad = json.loads(proc.stdout[proc.stdout.index("RESULT") + 6:].strip())
    assert not bad, f"import mast.pyexec 拖进了 {bad}"
