"""解释器发现、B1 的诚实报告，以及 mastkit 名单。

这份测试里最重要的一条是 ``test_isolation_report_matches_reality``。

写第一版实现时我漏掉了 B1 的测试（计划里有，落地的六条里没有），结果是：
**dev 回退路径下 ``import nanonis_spm`` 是成功的**，而这件事本来会一直没人发现
—— 因为所有测试都绿。原因不难懂：dev 回退用的是 ``sys.executable``，那就是 MAST
自己的 venv，里面当然装着 nanonis_spm。

所以这里钉的不是「B1 永远成立」（那在 dev 上是假话），而是
**「``rt.isolated`` 必须与子进程里的实际情况一致」** —— 报告诚实是能在任何运行时
上验证的，而「以为它碰不到仪器」正是这整个功能敢存在的那个理由。
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

import pytest

from mast.pyexec import kit_manifest as km
from mast.pyexec.runtime import (
    LEAK_MODULES,
    REQUIRED_PACKAGES,
    ProbeReport,
    find_runtime,
    probe_runtime,
    reset_runtime_cache,
)

CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0
_RT = find_runtime()


# ── 解释器发现 ───────────────────────────────────────────────────────
@pytest.fixture
def no_system_python(monkeypatch):
    """屏蔽**所有**系统候选来源。

    第一版只 monkeypatch 了 ``shutil.which``，结果测试失败 —— 因为注册表探测
    绕过了它，在这台机器上真找到了一个装着科学栈的 Python 3.13。那次失败本身是
    好消息（说明注册表那条路真的通），但也说明「屏蔽外部世界」不能只堵一个口。
    """
    monkeypatch.setattr("mast.pyexec.runtime._system_candidates", lambda: [])
    monkeypatch.setattr(sys, "frozen", True, raising=False)   # 同时关掉 dev 分支


def test_find_runtime_never_raises(monkeypatch, no_system_python):
    """``tailscale.find_cli`` 的契约：找不到就 None，绝不抛。

    一个探测函数抛异常，会在调用点长出一堆 try/except，而其中一处忘了写就是
    整条功能挂掉。
    """
    monkeypatch.setenv("MAST_PYRUNTIME_DIR", "/definitely/not/here")
    reset_runtime_cache()
    try:
        rep = probe_runtime(refresh=True)
        assert isinstance(rep, ProbeReport)
        assert rep.runtime is None
    finally:
        reset_runtime_cache()


def test_not_found_says_what_it_looked_at(monkeypatch, tmp_path, no_system_python):
    """「没找到」必须是一句能照着做的话，不是一个空值。"""
    fake = tmp_path / "python.exe"
    fake.write_text("not a real interpreter", encoding="utf-8")
    monkeypatch.setenv("MAST_PYRUNTIME_DIR", str(tmp_path))
    reset_runtime_cache()
    try:
        rep = probe_runtime(refresh=True)
        assert rep.runtime is None
        why = rep.why_not()
        assert str(fake) in why, "报告里必须点名查过的候选"
        assert "Python 3.13" in why, "必须告诉用户该装什么"
    finally:
        reset_runtime_cache()


def test_a_frozen_build_never_falls_back_to_sys_executable(monkeypatch):
    """打包版里绝不能把 ``sys.executable`` 当分析解释器 —— 那是 MAST.exe。

    这条同时守着 B1 的一半：dev 回退是唯一一条**已知不隔离**的路径，它必须只在
    开发机上出现。
    """
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr("mast.pyexec.runtime._system_candidates", lambda: [])
    monkeypatch.setenv("MAST_PYRUNTIME_DIR", "/definitely/not/here")
    reset_runtime_cache()
    try:
        assert probe_runtime(refresh=True).runtime is None, (
            "frozen 时不该找到任何运行时 —— 找到了说明 dev 分支漏进了打包版")
    finally:
        reset_runtime_cache()


@pytest.mark.skipif(_RT is None, reason="本机没有可用的分析运行时")
def test_a_found_runtime_has_the_required_stack():
    assert all(p in _RT.packages for p in REQUIRED_PACKAGES)
    assert _RT.version.startswith("3.")


# ── B1 的诚实报告（这一组是本文件的重点）──────────────────────────
@pytest.mark.skipif(_RT is None, reason="本机没有可用的分析运行时")
def test_isolation_report_matches_reality():
    """``rt.isolated`` 必须与子进程里**真的能不能 import** 一致。

    两个方向都要错不得：
      · 报 isolated 但其实能 import → 「我们以为它碰不到仪器」，最危险的那种错；
      · 报不 isolated 但其实不能 → 无谓的告警，会让人学会忽略这条警告。

    注意断言的是**子进程**的 import 结果。父进程里 ``nanonis_spm`` 是
    ``tests/conftest.py:16-19`` 装的 MagicMock，拿它判什么都得不出真话。
    """
    actually_reachable = []
    for mod in LEAK_MODULES:
        proc = subprocess.run(
            [str(_RT.exe), "-I", "-c",
             f"import {mod}" ],
            capture_output=True, text=True, timeout=60,
            creationflags=CREATE_NO_WINDOW)
        if proc.returncode == 0:
            actually_reachable.append(mod)

    assert sorted(_RT.leaks) == sorted(actually_reachable), (
        f"报告说能 import {sorted(_RT.leaks)}，实际是 {sorted(actually_reachable)} "
        "—— 报告和事实对不上，这条报告就没有价值了"
    )
    assert _RT.isolated == (not actually_reachable)


@pytest.mark.skipif(_RT is None or _RT.kind != "bundled",
                    reason="只有随包发的运行时才承诺 B1 硬成立（dev/system 回退不承诺）")
def test_the_bundled_runtime_is_actually_isolated():
    """随包运行时**必须**是隔离的 —— 这是 B1 唯一硬成立的地方。

    它一红说明打包时把 mast 或 nanonis_spm 装进分析运行时了，那就等于开了一条
    没有闸门的硬件通道。
    """
    assert _RT.isolated, f"随包运行时可 import {_RT.leaks}"


@pytest.mark.skipif(_RT is None, reason="本机没有可用的分析运行时")
def test_a_leaky_runtime_says_so_in_its_summary():
    """不隔离就要在 summary 里说 —— 这行字会进日志，也会给用户看。"""
    if _RT.isolated:
        pytest.skip("本机运行时是隔离的，没有可检查的告警文案")
    assert "不成立" in _RT.summary


# ── mastkit 名单 ────────────────────────────────────────────────────
def test_kit_manifest_is_closed():
    """名单必须是**模块级依赖的传递闭包**。

    第一版漏了三个模块，因为闭包分析跳过了相对导入（``from .processors import``）
    —— 症状是子进程里 ``import mast.data.loaders`` 抛
    ``ModuleNotFoundError: mast.data.processors``。修的不是名单，是分析器；
    这条测试每次重算，名单再漏就红。
    """
    mods, _deferred, _third = km.closure()
    missing = sorted(mods - set(km.KIT_MODULES))
    assert not missing, f"这些在闭包里却不在名单：{missing}"


def test_kit_closure_pulls_in_no_hardware():
    """名单闭包里出现硬件 = **从后门拆掉 B1**。

    一个看起来无害的分析模块，只要它 import 了 ``mast.core.connection``，
    子进程就拿到了一条通往仪器的路 —— 而且是从「我们主动拷进去的东西」里来的。
    """
    mods, _deferred, _third = km.closure()
    bad = sorted(m for m in mods
                 if any(m.startswith(f) for f in km.FORBIDDEN_PREFIXES))
    assert not bad, f"名单闭包里有硬件模块：{bad}"


def test_kit_third_party_deps_are_all_in_the_shipped_stack():
    """名单只能依赖分析运行时里真有的库。

    依赖一个没装的库，表现是「这个 import 怎么会失败」—— 而 agent 会以为是自己
    写错了。
    """
    _mods, _deferred, third = km.closure()
    shipped = {"numpy", "scipy", "matplotlib", "pandas", "skimage", "sklearn",
               "PIL", "mast", "mastkit"}
    ext = sorted(t for t in third
                 if t not in sys.stdlib_module_names and t not in shipped)
    assert not ext, f"名单依赖了运行时里没有的库：{ext}"


def test_deferred_deps_are_reported_not_hidden():
    """函数内的惰性 mast 依赖要被**列出来**，因为那是「调用时才失败」的功能。

    ``data/quality.py`` 在函数体里 import ``mast.vision.atomic_phase``（不在名单，
    它的闭包会拖进整个 VIGIL）。这是可接受的降级 —— 但必须写进给 agent 的说明，
    否则它会在一个用不了的函数上反复重试。
    """
    _mods, deferred, _third = km.closure()
    assert isinstance(deferred, set)
    # 不断言具体内容（名单会变），断言这个通道存在且能报出东西来
    assert all(d.startswith("mast") for d in deferred)


def test_install_then_verify_is_byte_identical(tmp_path):
    """拷进去的每一份都必须与源文件**逐字节**相同。

    这是「同一份代码的部署副本」和「第二套实现」之间唯一的区别所在
    （2026-07-20 删 ``data/formats.py`` 那条纪律）。所以判据是 sha256，不是
    行为比对 —— 行为比对只覆盖被测到的那条路径。
    """
    digests = km.install(tmp_path)
    assert len(digests) == len(km.KIT_MODULES)
    assert km.verify(tmp_path) == []

    # 改掉一个字节 → verify 必须发现
    victim = next(iter(digests))
    p = tmp_path / victim
    p.write_bytes(p.read_bytes() + b"\n# drifted\n")
    assert victim in km.verify(tmp_path)


@pytest.mark.skipif(_RT is None, reason="本机没有可用的分析运行时")
def test_the_installed_kit_imports_in_a_child(tmp_path):
    """装完之后，子进程里真的 import 得动 —— 而硬件模块真的 import 不动。

    ``mast.core.connection`` 在这里抛 ImportError **不是被拦下来的**，是那个文件
    根本没被拷进去。B1 由「文件不存在」保证，不由任何运行时逻辑保证。
    """
    km.install(tmp_path)
    # 真实会话里 _provision 会把它拷进来。这里也要拷，否则随包运行时上的
    # sitecustomize 会 fail-closed（rc=97）——**那是对的行为**，只是这个测试
    # 想验的是别的东西。
    from mast.pyexec import _srcfiles
    shutil.copy2(_srcfiles.pyexec_file("child_audit.py"), tmp_path / "_mast_audit.py")
    script = tmp_path / "t.py"
    script.write_text('''
from mast.io.nanonis_files import read_sxm, read_dat, read_3ds, sxm_frame_meta
from mast.io import map_analysis, exp_map, product_validity
from mast.data import load_image_2d
from mast.data.processors import plane_subtract
from mastkit import read_sxm as rk
assert rk is read_sxm, "mastkit 必须指向同一个对象，不是第二份"
print("KIT OK")
for m in ("mast.core.connection", "mast.instruments", "mast.core.executor",
          "mast.agents", "mast.vision.module"):
    try:
        __import__(m)
        print("REACHABLE", m)
    except ImportError:
        pass
''', encoding="utf-8")
    env = {k: os.environ[k] for k in
           ("SystemRoot", "windir", "PATH", "TEMP", "TMP", "COMSPEC", "PATHEXT")
           if k in os.environ}
    # 两条路都要给，理由见 execute.build_env 那段注释：随包运行时带 ._pth ⇒
    # 默认 isolated ⇒ **忽略 PYTHONPATH**。只给 PYTHONPATH 时这个测试在 dev venv
    # 上绿、在真正要发出去的那个运行时上红 —— 而红的方式是「mastkit 找不到」。
    env.update({"PYTHONPATH": str(tmp_path),
                "MAST_PYEXEC_SESSION": str(tmp_path),
                "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"})
    proc = subprocess.run([str(_RT.exe), "-s", "-B", "-X", "utf8", str(script)],
                          cwd=tmp_path, env=env, capture_output=True, text=True,
                          timeout=120, creationflags=CREATE_NO_WINDOW)
    assert proc.returncode == 0, proc.stderr[-1200:]
    assert "KIT OK" in proc.stdout
    assert "REACHABLE" not in proc.stdout, (
        f"硬件/重模块从 mastkit 这条路够到了：{proc.stdout}")


# ---------------------------------------------------------------------------
# 会话目录要靠两条独立的路送进子进程
# ---------------------------------------------------------------------------

def test_env_carries_the_session_by_two_independent_routes(tmp_path):
    """``PYTHONPATH`` 和 ``MAST_PYEXEC_SESSION`` **都**要有。

    这条不需要任何运行时就能跑，这正是它的价值：

    随包运行时带 ``python313._pth``，而 ``._pth`` 的存在让解释器默认 isolated ——
    isolated 忽略 ``PYTHONPATH``。于是会话目录不在 ``sys.path`` 上，
    ``sitecustomize`` 和 ``_mast_audit`` 都找不到，**审计钩子静默不装**，
    而分析进程看起来完全正常。

    开发环境走的是非 isolated 的 venv，所以只留 ``PYTHONPATH`` 的话，
    整套 pyexec 测试在 dev 上全绿 —— 洞只存在于要发给用户的那个运行时上。
    （2026-08-20 实测：接上随包运行时之后，pyexec 一次红了 15 条。）

    删掉任一条这里都会红，而且不依赖本机有没有 isolated 解释器。
    """
    from mast.pyexec import execute as X
    from mast.pyexec.session import PySession

    sess = PySession(root=tmp_path / "s", sid="envprobe")
    env = X.build_env(sess, scan_dirs=(), blocked_ports=())
    assert env.get("PYTHONPATH") == str(sess.root), "PYTHONPATH 这条路断了"
    assert env.get("MAST_PYEXEC_SESSION") == str(sess.root), (
        "MAST_PYEXEC_SESSION 这条路断了 —— isolated 运行时上钩子会静默不装")


@pytest.mark.skipif(_RT is None or not getattr(_RT, "isolated", False),
                    reason="本机没有 isolated 的分析运行时（随包运行时才是）")
def test_the_audit_hook_installs_on_an_isolated_runtime(tmp_path):
    """在**真的 isolated** 的运行时上，钩子必须装得上。

    上面那条测的是「env 里有没有」，这条测的是「那条链子通不通」——
    env → sitecustomize → sys.path → _mast_audit.install()。两条都要有：
    env 对了而运行时里没装 sitecustomize，链子照样断。
    """
    from mast.pyexec import _srcfiles

    shutil.copy2(_srcfiles.pyexec_file("child_audit.py"), tmp_path / "_mast_audit.py")
    probe = tmp_path / "probe.py"
    probe.write_text(
        "import json\n"
        "import _mast_audit as A\n"
        "print('HOOK' + json.dumps({'installed': A._INSTALLED}))\n",
        encoding="utf-8")
    env = {k: os.environ[k] for k in
           ("SystemRoot", "windir", "PATH", "TEMP", "TMP", "COMSPEC", "PATHEXT")
           if k in os.environ}
    env["MAST_PYEXEC_SESSION"] = str(tmp_path)
    env.pop("PYTHONPATH", None)          # 故意不给 —— 就是要验另一条路
    proc = subprocess.run([str(_RT.exe), "-s", "-B", str(probe)],
                          cwd=tmp_path, env=env, capture_output=True, text=True,
                          timeout=120, creationflags=CREATE_NO_WINDOW)
    assert proc.returncode == 0, (
        f"rc={proc.returncode}（97 = 钩子装不上，fail-closed）\n{proc.stderr[-1200:]}")
    assert '"installed": true' in proc.stdout.lower().replace(" ", " "), proc.stdout
