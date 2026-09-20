"""子进程的三条底线 —— 以及同样重要的：**它做得了该做的事**。

这份测试刻意分成两半，篇幅差不多：

* **底线组** —— 碰不到仪器、毁不掉原始测量数据、停得下来。
* **能力组** —— 读得了数据、写得了派生结果、六库能 import、并行能用、
  traceback 完整回得来。

只测前一半会造出一个「安全但发挥不出水平」的环境，而且没人会发现 —— 因为所有
测试都是绿的。第二组存在的意义就是让那种退化变红。

一条实测出来的架构结论（值得留在这里）
====================================
第一版用「引导器 exec 用户脚本」，撞出两个问题：``multiprocessing`` 的 spawn
子进程按 ``__main__.<fn>`` 找不到函数（PicklingError），而且**那些子进程根本
不装钩子**。改成 ``sitecustomize`` 之后两个问题一起消失 —— 并行能用了，而且
每个 spawn 子进程都会 import 它，保护面反而更完整。
``test_spawned_children_are_protected_too`` 钉的就是这一条。
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from mast.pyexec.runtime import find_runtime

PYEXEC_SRC = Path(__file__).resolve().parents[4] / "MASTv2" / "mast" / "pyexec"
CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0

_RT = find_runtime()
pytestmark = pytest.mark.skipif(
    _RT is None,
    reason="没有可用的分析 Python 运行时 —— 隔离与能力都无从验证（这条 skip 是"
           "刻意吵闹的：静默跳过一整套隔离测试比没有测试更糟）")


@pytest.fixture
def lab(tmp_path):
    """一个会话目录 + 一份「原始测量数据」，全都在 tmp_path 里。

    ``tests/v2/conftest.py`` 记着本仓被「测试写进用户真实数据」咬过五次，
    而这些测试**按设计**就要试着写测量文件 —— 所以一个字节都不许落在 tmp 之外。
    """
    session = tmp_path / "session"
    (session / "code").mkdir(parents=True)
    data = tmp_path / "data"
    data.mkdir()

    sxm = data / "Au111_001.sxm"
    sxm.write_bytes(b":NANONIS_VERSION:\n2\n:SCANIT_END:\n" + b"\x00" * 512)
    (session / "own.dat").write_text("mine", encoding="utf-8")
    subdir = data / "subdir"
    subdir.mkdir()
    (subdir / "b.sxm").write_bytes(b"x" * 64)

    shutil.copy2(PYEXEC_SRC / "child_audit.py", session / "_mast_audit.py")
    shutil.copy2(PYEXEC_SRC / "child_sitecustomize.py", session / "sitecustomize.py")

    class Lab:
        def __init__(self):
            self.session = session
            self.data = data
            self.sxm = sxm
            self.subdir = subdir
            self._n = 0

        def run(self, code: str, timeout: float = 180.0):
            self._n += 1
            f = session / "code" / f"step{self._n:02d}.py"
            f.write_text(code, encoding="utf-8")
            keep = ("SystemRoot", "windir", "COMSPEC", "PATHEXT",
                    "NUMBER_OF_PROCESSORS", "TEMP", "TMP", "PATH")
            env = {k: os.environ[k] for k in keep if k in os.environ}
            env.update({
                "PYTHONPATH": str(session),
                "MAST_PYEXEC_SESSION": str(session),
                "MAST_PYEXEC_BLOCKED_PORTS": "6501,6502,6503,6504",
                "PYTHONUTF8": "1",
                "PYTHONIOENCODING": "utf-8",
                "MPLBACKEND": "Agg",
                "MPLCONFIGDIR": str(tmp_path / "mplcache"),
                "OMP_NUM_THREADS": "2",
            })
            # -s 禁 user site。**不用 -I** —— 它会连 PYTHONPATH 一起忽略，
            # sitecustomize 就找不到了，钩子静默不装。
            proc = subprocess.run(
                [str(_RT.exe), "-s", "-B", "-X", "utf8", str(f)],
                cwd=session, env=env, capture_output=True, text=True,
                timeout=timeout, creationflags=CREATE_NO_WINDOW)
            return proc.returncode, (proc.stdout or ""), (proc.stderr or "")

    return Lab()


# ── 钩子装没装上，本身要可观察 ──────────────────────────────────────
def test_the_hook_is_actually_installed(lab):
    """「装上了」必须是能读到的事实，不是一个假设。

    钩子没装上时，下面每一条「被拒」测试都会红 —— 但那时人会先去怀疑判据。
    先有这一条，排查顺序才对。
    """
    rc, out, err = lab.run("import _mast_audit; print(_mast_audit.describe())")
    assert rc == 0, err
    assert '"installed": true' in out


# ── 底线 B2：毁不掉原始测量数据 ─────────────────────────────────────
def test_overwriting_an_existing_measurement_file_is_refused(lab):
    rc, _out, err = lab.run(f'open(r"{lab.sxm}", "wb").write(b"ruined")')
    assert rc != 0 and "MastDataProtection" in err
    assert lab.sxm.stat().st_size > 500, "文件被改了 —— 拦截没生效"


@pytest.mark.parametrize("what,code", [
    ("remove", 'import os; os.remove(r"{p}")'),
    ("truncate", 'import os; os.truncate(r"{p}", 0)'),
    ("rename", 'import os; os.rename(r"{p}", r"{p}.moved")'),
])
def test_destroying_an_existing_measurement_file_is_refused(lab, what, code):
    """删 / 截断 / 改名 —— 三条都要拦。

    ``os.truncate`` 是实测审计事件时才发现的：把一个 .sxm 截到 0 字节和删掉它
    一样彻底，而它不走 ``open`` 也不走 ``os.remove``。读文档想不到，跑一遍就
    看见了。
    """
    rc, _out, err = lab.run(code.format(p=lab.sxm))
    assert rc != 0 and "MastDataProtection" in err
    assert lab.sxm.exists() and lab.sxm.stat().st_size > 500


def test_rmtree_over_a_dir_holding_measurements_is_refused(lab):
    """rmtree 要看整棵树 —— 只看根目录名是看不出里面有什么的。"""
    rc, _out, err = lab.run(f'import shutil; shutil.rmtree(r"{lab.subdir}")')
    assert rc != 0 and "MastDataProtection" in err
    assert (lab.subdir / "b.sxm").exists()


def test_spawned_children_are_protected_too(lab):
    """并行子进程也受保护 —— 第一版方案在这里是裸奔的。

    这是最容易批量出事的地方（一个 ``pool.map`` 里的误写会同时打中很多文件），
    而 exec-引导器那条路上 spawn 出来的子进程压根不经过钩子。
    """
    rc, out, err = lab.run(f'''
import multiprocessing as mp
def try_ruin(_):
    try:
        open(r"{lab.sxm}", "wb").write(b"ruined from a child")
        return "WROTE"
    except PermissionError as e:
        return "BLOCKED:" + type(e).__name__
if __name__ == "__main__":
    with mp.Pool(2) as pool:
        print(pool.map(try_ruin, range(2)))
''')
    assert rc == 0, err
    assert "WROTE" not in out, "spawn 子进程绕过了保护"
    assert "BLOCKED:MastDataProtection" in out
    assert lab.sxm.stat().st_size > 500


# ── 底线 B1（的顺手一层）：连不上仪器端口 ───────────────────────────
def test_connecting_to_a_nanonis_port_is_refused(lab):
    """断言的是**钩子的**异常，不是 ConnectionRefusedError。

    后者在没开 Nanonis 的机器上本来就会出现 —— 拿它当通过，等于什么都没测。
    """
    rc, out, _err = lab.run('''
import socket
s = socket.socket(); s.settimeout(0.3)
try:
    s.connect(("127.0.0.1", 6501)); print("CONNECTED")
except PermissionError as e:
    print("BLOCKED:", type(e).__name__)
except Exception as e:
    print("OTHER:", type(e).__name__)
''')
    assert rc == 0
    assert "BLOCKED: MastInstrumentPortBlocked" in out


# ══════════════════════════════════════════════════════════════════════
# 能力组 —— 和上面同等重要
# ══════════════════════════════════════════════════════════════════════
def test_the_whole_science_stack_imports_and_plots_without_a_single_false_positive(lab):
    """完整 import 六库 + 画图 + 存盘，**零钩子误报**。

    这是拦截面最容易出事的地方：完整 import 会触发两千多次 ``open`` 和若干次
    ``os.remove``，全是库自己的临时文件。放它们过去的是「**测量后缀**」这一条
    （那些临时文件叫 ``h5f7jo5j`` 这种名字，根本没有后缀），**不是**「已存在」
    ——变异实测确认：去掉「已存在」时这条测试仍然绿，红的是
    ``test_creating_a_new_measurement_file_is_allowed``。两个子句各管一件事，
    说反了下一个人就会删错那一个。

    误伤的表现是「运行时坏了」，最难诊断的那一类，所以这条必须一直在。
    """
    rc, out, err = lab.run(f'''
import os, matplotlib
matplotlib.use("Agg")
import numpy, scipy, scipy.optimize, scipy.signal, scipy.ndimage
import pandas, skimage.filters, skimage.measure, sklearn.cluster
import matplotlib.pyplot as plt
f = plt.figure(); plt.plot([1,2,3],[1,4,9])
f.savefig(os.path.join(r"{lab.session}", "fig.png")); plt.close(f)
numpy.save(os.path.join(r"{lab.session}", "x.npy"), numpy.arange(5))
print("STACK OK")
''')
    assert rc == 0, err
    assert "STACK OK" in out
    assert "MastDataProtection" not in err, "库自己的文件操作被误伤了"


def test_writing_a_derived_file_next_to_the_source_succeeds(lab):
    """B2 的对偶。不测这一半，就只证明了「什么都写不了」。

    往源文件旁边写派生结果是有真实先例的正当做法
    （``skills/paper/data_processing.py:114-121`` 就在这么干）。
    """
    rc, out, err = lab.run(f'''
import numpy
numpy.save(r"{lab.data / "Au111_001_leveled.npy"}", numpy.zeros((4, 4)))
print("DERIVED OK")
''')
    assert rc == 0, err
    assert "DERIVED OK" in out
    assert (lab.data / "Au111_001_leveled.npy").exists()


def test_creating_a_new_measurement_file_is_allowed(lab):
    """新建一个同后缀的文件完全正当 —— 受保护的是「已经存在的」那些。"""
    rc, out, err = lab.run(
        f'open(r"{lab.data / "brand_new.sxm"}", "wb").write(b"n"); print("NEW OK")')
    assert rc == 0, err
    assert "NEW OK" in out


def test_files_inside_the_session_are_freely_writable(lab):
    """会话内的 .dat 是脚本自己的产物，随便改。"""
    rc, out, err = lab.run(
        f'open(r"{lab.session / "own.dat"}", "w").write("ok"); print("SESS OK")')
    assert rc == 0, err
    assert "SESS OK" in out


def test_non_instrument_network_is_not_blocked(lab):
    """只拦仪器端口，不是禁网。钉住「只拦」这个决定。"""
    rc, out, _err = lab.run('''
import socket
s = socket.socket(); s.settimeout(0.3)
try:
    s.connect(("127.0.0.1", 59999)); print("connected")
except PermissionError as e:
    print("WRONGLY BLOCKED:", e)
except Exception as e:
    print("refused-but-not-blocked", type(e).__name__)
''')
    assert rc == 0
    assert "WRONGLY BLOCKED" not in out


def test_multiprocessing_actually_works(lab):
    """并行可用 —— 这是「放开子进程」这个决定的凭据。

    把 ``ActiveProcessLimit`` 改回 1、或把审计钩子改成拦 ``subprocess``，
    这条就红。
    """
    rc, out, err = lab.run('''
import multiprocessing as mp
import numpy as np
def work(i):
    return float(np.arange(1000 * (i + 1)).sum())
if __name__ == "__main__":
    with mp.Pool(4) as pool:
        vals = pool.map(work, range(4))
    print("PARALLEL OK", len(vals))
''')
    assert rc == 0, err
    assert "PARALLEL OK 4" in out


def test_a_failure_comes_back_as_a_full_traceback(lab):
    """完整 traceback（行号 + 函数名）—— 模型靠它自修。

    这正是进程内那套 ``run_numpy_snippet`` 做不到的事：它的错误是一行摘要
    （``tools.py:1061``），模型只能猜。
    """
    rc, _out, err = lab.run(
        "import numpy\ndef f():\n    return numpy.zeros(3)[9]\nf()\n")
    assert rc != 0
    assert "IndexError" in err
    assert "in f" in err and "line 3" in err


def test_a_long_running_script_can_be_killed_with_its_children(lab, tmp_path):
    """B3：跑飞了要停得下来，连同它开出去的进程。"""
    from mast.pyexec.jobobject import JobBox

    f = lab.session / "code" / "spin.py"
    f.write_text('''
import multiprocessing as mp, time
def spin():
    while True:
        pass
if __name__ == "__main__":
    ps = [mp.Process(target=spin) for _ in range(2)]
    for p in ps: p.start()
    print("KIDS " + ",".join(str(p.pid) for p in ps), flush=True)
    while True: time.sleep(0.1)
''', encoding="utf-8")
    env = {k: os.environ[k] for k in ("SystemRoot", "windir", "PATH", "TEMP", "TMP")
           if k in os.environ}
    env["PYTHONPATH"] = str(lab.session)
    box = JobBox(memory_bytes=1024 ** 3, active_process_limit=8)
    proc = subprocess.Popen([str(_RT.exe), "-s", "-B", str(f)], cwd=lab.session,
                            env=env, stdout=subprocess.PIPE, text=True,
                            creationflags=CREATE_NO_WINDOW)
    try:
        box.assign(proc.pid)
        line = proc.stdout.readline().strip()
        kids = [int(x) for x in line[len("KIDS "):].split(",")] if line.startswith("KIDS") else []
        t0 = time.time()
        box.kill(proc.pid)
        proc.wait(timeout=10)
        time.sleep(0.5)
        assert time.time() - t0 < 10
        if kids:
            import psutil
            assert not [k for k in kids if psutil.pid_exists(k)], "子进程活下来了"
    finally:
        box.close()
        if proc.poll() is None:
            proc.kill()
