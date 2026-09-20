"""Job Object：可停止性 —— 这是「放开子进程」的前提，不是它的对立面。

分析脚本可以自由 ``multiprocessing`` 并行（200 张图并行处理是真需求），**因为**
整棵树关在一个 job 里、而且句柄一关 OS 就收干净。没有这一层就只能禁子进程。

同时这里修掉一个真实的退步：``agents/data_processing/tools.py`` 的
``_maybe_set_memory_rlimit()`` 在 Windows 上直接 return，所以现役 numpy 沙箱在
真机上**根本没有内存上限**，而墙钟超时杀不掉一个 CPython 线程。第三条测试就是
在证明这件事真的变了 —— 不是「代码看起来对」。
"""

from __future__ import annotations

import subprocess
import sys
import time

import pytest

from mast.pyexec.jobobject import IS_WINDOWS, JobBox

psutil = pytest.importorskip("psutil")

pytestmark = pytest.mark.skipif(
    not IS_WINDOWS, reason="Job Object 是 Windows 机制；非 Windows 上 JobBox 是 no-op")

CREATE_NO_WINDOW = 0x08000000

_SPAWNER = """
import multiprocessing, time
def spin():
    while True:
        pass
if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    ps = [multiprocessing.Process(target=spin) for _ in range(3)]
    for p in ps:
        p.start()
    print("KIDS " + ",".join(str(p.pid) for p in ps), flush=True)
    while True:
        time.sleep(0.1)
"""


def _popen(code: str, **kw):
    return subprocess.Popen(
        [sys.executable, "-c", code], creationflags=CREATE_NO_WINDOW, **kw)


def test_jobbox_is_available_on_this_machine():
    """建不起来就只剩「超时杀主进程」，那是降级 —— 至少要吵一声。"""
    box = JobBox(memory_bytes=256 * 1024 ** 2)
    try:
        assert box.supported, f"Job Object 建立失败：{box.reason}"
    finally:
        box.close()


def test_kill_takes_the_whole_process_tree():
    """父进程 + 3 个 spawn 出来的子进程，一起死。

    这条是「可以放开 multiprocessing」的凭据。它一红，就说明并行开出去的进程
    会在超时后留在机器上 —— 而这台机器同时在采数据。
    """
    box = JobBox(memory_bytes=1024 ** 3, active_process_limit=8)
    proc = _popen(_SPAWNER, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert box.assign(proc.pid), "assign 失败，这次测试什么也没证明"
        line = proc.stdout.readline().strip()
        assert line.startswith("KIDS"), f"子进程没起来：{line!r}"
        kids = [int(x) for x in line[len("KIDS "):].split(",")]
        assert all(psutil.pid_exists(k) for k in kids), "子进程本来就没活着"

        box.kill(proc.pid)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pytest.fail("父进程没在 5 秒内退出")
        time.sleep(0.5)
        survivors = [q for q in ([proc.pid] + kids) if psutil.pid_exists(q)]
        assert not survivors, f"这些进程活下来了：{survivors}"
    finally:
        box.close()
        if proc.poll() is None:
            proc.kill()


def test_memory_limit_actually_refuses_the_allocation():
    """256 MiB 上限下分配 1 GiB 必须失败。

    反例（这就是这条测试存在的理由）：进程内那套用的是 POSIX ``resource``
    rlimit，在 Windows 上 ``_maybe_set_memory_rlimit`` **直接 return** —— 上限
    只存在于注释里。出进程如果不配 Job Object，这个退步会原样保留，而「我们出
    进程是为了资源控制」就只有一半是真的。
    """
    box = JobBox(memory_bytes=256 * 1024 ** 2)
    proc = _popen("b = bytearray(1024**3); print('ALLOCATED')",
                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        box.assign(proc.pid)
        out, err = proc.communicate(timeout=60)
        assert "ALLOCATED" not in (out or ""), (
            "1 GiB 在 256 MiB 的上限下分配成功了 —— 上限没生效")
        assert proc.returncode != 0
        assert "MemoryError" in (err or ""), f"死因不是内存：{(err or '')[-200:]}"
    finally:
        box.close()
        if proc.poll() is None:
            proc.kill()


def test_closing_the_handle_reaps_survivors():
    """KILL_ON_JOB_CLOSE：父进程崩了，OS 也会收干净。

    这是任何基于 psutil 的收割器都保证不了的一条 —— 收割器自己死了就不收割了。
    """
    box = JobBox(memory_bytes=512 * 1024 ** 2)
    proc = _popen("import time\nwhile True: time.sleep(0.1)")
    try:
        box.assign(proc.pid)
        assert psutil.pid_exists(proc.pid)
        box.close()                      # 不显式 kill，只关句柄
        time.sleep(1.0)
        assert not psutil.pid_exists(proc.pid), (
            "关掉 job 句柄之后子进程还活着 —— KILL_ON_JOB_CLOSE 没起作用")
    finally:
        if proc.poll() is None:
            proc.kill()


def test_a_failed_jobbox_degrades_instead_of_raising(monkeypatch):
    """建不起来要降级 + 说原因，不能把整个分析挡死。

    资源上限是纵深防御的一层；它自己坏了不该变成「不能用 Python 了」。
    """
    import mast.pyexec.jobobject as jb

    monkeypatch.setattr(jb._k32, "CreateJobObjectW", lambda *a: 0)
    box = JobBox(memory_bytes=1024 ** 2)
    assert box.supported is False
    assert box.reason, "降级了却没说原因 —— 那就没人知道上限没生效"
    box.assign(12345)     # 必须是安全的 no-op
    box.kill(12345)
    box.close()
