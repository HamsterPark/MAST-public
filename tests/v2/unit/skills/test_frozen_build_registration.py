"""在独立解释器中模拟冻结注册，避免开发发现过程污染 sys.modules 后令断言恒真。"""

from __future__ import annotations

import importlib
import json
import os
import pkgutil
import subprocess
import sys

import pytest

from mast.core.registry import SkillRegistry

SKILL_PACKAGES = ("mast.skills.builtins", "mast.skills.composite")

#: 在子进程里跑的冻结模拟:只 import 包本身,再扫 sys.modules —— 这正是
#: ``SkillRegistry.discover`` 在冻结环境里实际走的那条路(walk_packages 返回空,
#: 兜底逻辑从 sys.modules 里捡)。
_FROZEN_PROBE = """
import importlib, json, sys
from mast.core.registry import SkillRegistry

PKGS = {pkgs!r}
for pkg in PKGS:
    importlib.import_module(pkg)          # 只 import 包本身,不递归
reg = SkillRegistry()
for name, mod in list(sys.modules.items()):
    if mod is None:
        continue
    if any(name.startswith(p + ".") for p in PKGS):
        reg._scan_module(mod)
print(json.dumps({{n: reg.get(n).__module__ for n in reg._skills}}))
"""


def _dev_registry() -> SkillRegistry:
    """开发环境:walk_packages 全量递归发现。"""
    reg = SkillRegistry()
    reg.discover(*SKILL_PACKAGES)
    return reg


@pytest.fixture(scope="module")
def frozen_skills() -> dict[str, str]:
    """``{技能名: 模块}``,来自一个干净解释器里的冻结模拟。

    module 级 fixture:一次子进程,五条测试共用。
    """
    env = dict(os.environ)
    root = str(__import__("pathlib").Path(__file__).resolve().parents[4] / "MASTv2")
    env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-c", _FROZEN_PROBE.format(pkgs=SKILL_PACKAGES)],
        capture_output=True, text=True, env=env, timeout=300,
    )
    assert proc.returncode == 0, (
        f"冻结模拟子进程失败:\n{proc.stdout}\n{proc.stderr}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_no_skill_disappears_in_the_frozen_build(frozen_skills):
    """冻结注册表应与声明的技能集合一致。"""
    dev = set(_dev_registry()._skills)
    missing = sorted(dev - set(frozen_skills))
    assert not missing, (
        f"{len(missing)} 个技能在打包版里会消失(它们所在的模块没有被 "
        f"skills 包的 __init__ import 到):\n  " + "\n  ".join(missing[:40])
        + ("\n  …" if len(missing) > 40 else "")
    )


def test_every_skill_module_is_reachable_from_the_package_init(frozen_skills):
    """按模块给出更可读的诊断:哪个文件忘了在 __init__ 里 import。"""
    dev = _dev_registry()
    orphan_modules = sorted({
        dev.get(name).__module__
        for name in set(dev._skills) - set(frozen_skills)
    })
    assert not orphan_modules, (
        "这些 skill 模块没有被包的 __init__ import,打包后整块消失:\n  "
        + "\n  ".join(orphan_modules)
    )


def test_the_frozen_probe_can_actually_fail(frozen_skills):
    """守卫的守卫:证明这条模拟**分得出**在与不在。

    2026-08-05 之前它分不出 —— 同进程里跑,dev 发现已经把每个模块塞进 sys.modules,
    于是「冻结环境里有没有」这个问题恒答「有」。一条只会通过的断言,和没有断言
    长得一模一样。

    这里用一个**保证不存在**的名字反向确认:若它也「在」,说明模拟又退化成了
    「照抄开发环境」。
    """
    assert "ThisSkillDoesNotExist_ForgeGuardProbe" not in frozen_skills
    # 而且模拟确实产出了东西 —— 空集合会让上面两条断言同样恒真。
    assert len(frozen_skills) > 300, f"冻结模拟只看到 {len(frozen_skills)} 个技能"


def test_the_scan_intelligence_skills_survive_the_frozen_build(frozen_skills):
    """关键技能必须纳入注册和依赖闭包。"""
    for name in ("ScanAt", "SetScanBuffer", "TiltProbeCircle",
                 "TiltCalibrate", "AutoTilt", "BiasSettleChange",
                 "ExecuteScanPlan"):
        assert name in frozen_skills, f"{name} 在冻结环境里不存在"


def test_the_v62_skills_survive_the_frozen_build(frozen_skills):
    """核对指定的关键技能是否随包可用。"""
    for name in ("ForgeAuTip", "ReadHardwareEvents"):
        assert name in frozen_skills, f"{name} 在冻结环境里不存在"


@pytest.mark.parametrize("package", SKILL_PACKAGES)
def test_no_skill_module_fails_to_import(package):
    """任何一个 skill 模块 import 失败,在冻结环境里都是整块消失。

    开发时 walk_packages 只把导入失败记成一条 warning 就跳过了,所以这种问题在
    别的测试里往往看不见。
    """
    pkg = importlib.import_module(package)
    failures = []
    for info in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
        try:
            importlib.import_module(info.name)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{info.name}: {exc}")
    assert not failures, "以下 skill 模块 import 失败:\n" + "\n".join(failures)


@pytest.mark.parametrize("package", SKILL_PACKAGES)
def test_package_exports_resolve(package):
    """``__all__`` 里列的名字必须真的存在(打错字同样是静默失效)。"""
    mod = importlib.import_module(package)
    for name in getattr(mod, "__all__", ()) or ():
        assert hasattr(mod, name), f"{package}.__all__ 里的 {name!r} 无法解析"
