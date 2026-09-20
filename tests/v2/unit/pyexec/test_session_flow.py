"""会话 → staging → 执行 → 产物回收，整条链。

这里盯三件容易「看起来成了、其实没有」的事：

1. **图必须落进 ``figures_dir()``**。留在会话 ``out/`` 里的图，对
   ``list_figures()`` 完全不存在 —— 下游写报告的 agent 一张也看不到。
   「画出来了」和「交付了」之间隔着这一次拷贝。
2. **``result.json`` 逐字节回传**。这是数值不经过模型 token 流的那条路。本仓记着
   一次 ``3e-12`` 被念成「3 米」的事故 —— 重建出来的数总是看起来合理。
3. **子进程环境里没有 ``_MEI*`` / ``PYTHONHOME``**。前者会让子进程
   ``ERROR_BAD_EXE_FORMAT`` 直接死；后者会把它引向 MAST 的 ``_internal``，
   而那里 ``nanonis_spm/__init__.py`` **是真实存在的文件**。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from mast.pyexec import (
    build_env, get_session, harvest, images_for_toolmessage, run, snapshot, stage,
)
from mast.pyexec.runtime import find_runtime

_RT = find_runtime()
needs_runtime = pytest.mark.skipif(
    _RT is None, reason="没有可用的分析运行时 —— 整条链无从验证")

#: 一份真的 .sxm。CI / 别的机器上可能没有 —— 那时跳过用它的那几条，
#: 但**不跳过**整条链（.npy 那条覆盖 session/run/harvest 的全部）。
REAL_SXM = Path("<offload-dir>/stm-datasets/repos/ML-STM/example data"
                "/STM_WTip_WSe2-SL445_022.sxm")
needs_real_sxm = pytest.mark.skipif(
    not REAL_SXM.is_file(),
    reason=f"真 .sxm 样本不在本机（{REAL_SXM}）—— 跳过依赖真格式的那几条")


@pytest.fixture(autouse=True)
def _isolated_dirs(tmp_path, monkeypatch):
    """会话根和图库都指向 tmp_path。

    ``tests/v2/conftest.py:78-110`` 记着本仓被「测试写进用户真实数据」咬过五次，
    而 harvest **按设计**就往 ``figures_dir()`` 里写 —— 不重定向就会真的往那儿写。
    """
    monkeypatch.setenv("MAST_DP_SESSIONS_DIR", str(tmp_path / "dp-sessions"))
    monkeypatch.setenv("MAST_FIGURES_DIR", str(tmp_path / "figures"))
    yield


def _run_script(session, code: str, *, timeout_s: float = 120.0):
    before = snapshot(session)
    name = session.next_step_name()
    (session.code_dir / name).write_text(code, encoding="utf-8")
    res = run(session, f"code/{name}", runtime=_RT, timeout_s=timeout_s)
    return res, harvest(session, before)


# ── 会话本身 ────────────────────────────────────────────────────────
def test_session_is_provisioned_completely():
    """钩子、helper、mastkit 一个都不能少。

    幂等地重装是刻意的：一个被手工删掉的 ``sitecustomize.py`` 会让**保护静默
    消失**，那是最不能靠「应该还在吧」的东西。
    """
    s = get_session(experiment_id="exp1", thread_id="t1")
    for sub in ("code", "logs", "out", "inputs", "tmp"):
        assert (s.root / sub).is_dir(), sub
    for f in ("sitecustomize.py", "_mast_audit.py", "mastdata.py",
              "mastkit.py", "mast/io/nanonis_files.py"):
        assert (s.root / f).is_file(), f

    (s.root / "sitecustomize.py").unlink()          # 手工破坏
    s2 = get_session(experiment_id="exp1", thread_id="t1")
    assert (s2.root / "sitecustomize.py").is_file(), "重进会话没有把钩子补回来"


def test_session_id_is_stable_per_thread():
    a = get_session(experiment_id="exp1", thread_id="t1")
    b = get_session(experiment_id="exp1", thread_id="t1")
    c = get_session(experiment_id="exp1", thread_id="t2")
    assert a.sid == b.sid
    assert a.sid != c.sid


def test_reset_archives_instead_of_deleting():
    """``reset=True`` 归档不删除 —— 用户的中间结果是数据，不是垃圾。"""
    s = get_session(experiment_id="exp1", thread_id="t1")
    (s.out_dir / "precious.txt").write_text("hours of work", encoding="utf-8")

    s2 = get_session(experiment_id="exp1", thread_id="t1", reset=True)
    assert not (s2.out_dir / "precious.txt").exists(), "新会话应该是干净的"

    archived = list((s2.root.parent / "_archive").rglob("precious.txt"))
    assert archived, "旧会话被删掉了 —— 应该是归档"
    assert archived[0].read_text(encoding="utf-8") == "hours of work"


# ── 子进程环境 ──────────────────────────────────────────────────────
def test_child_env_leaks_nothing_dangerous(monkeypatch):
    """env 是**从零构造**的，不是 ``os.environ.copy()``。

    断言的是 ``build_env`` 造出来的那个 dict 本身 —— 比「跑一遍看行为」更早、
    更准，而且一眼看得出哪一项没挡住。
    """
    monkeypatch.setenv("_MEIPASS2", "C:/some/frozen/dir")
    monkeypatch.setenv("_PYI_APPLICATION_HOME_DIR", "C:/whatever")
    monkeypatch.setenv("PYTHONHOME", "C:/mast/_internal")
    monkeypatch.setenv("MAST_LAN_TOKEN", "secret")
    monkeypatch.setenv("MAST2_PROJECT_ROOT", "D:/MAST-data")

    s = get_session(experiment_id="exp1", thread_id="t1")
    env = build_env(s)

    for k in env:
        assert not k.startswith("_MEI"), f"{k} 会让子进程 ERROR_BAD_EXE_FORMAT"
        assert not k.startswith("_PYI"), k
        assert not k.startswith("MAST_LAN"), f"{k} 是凭据，分析进程没有理由知道"
    assert "PYTHONHOME" not in env, (
        "PYTHONHOME 会把子进程引向 MAST 的 _internal —— 而那里 "
        "nanonis_spm/__init__.py 是真实存在的文件")
    assert "MAST2_PROJECT_ROOT" not in env
    # PYTHONPATH 只许有会话目录（sitecustomize 靠它才装得上）
    assert env["PYTHONPATH"] == str(s.root)
    assert env["MPLBACKEND"] == "Agg"
    assert int(env["OMP_NUM_THREADS"]) >= 2


# ── 整条链（不依赖真 .sxm）───────────────────────────────────────────
@needs_runtime
def test_full_flow_with_an_npy_input():
    s = get_session(experiment_id="exp1", thread_id="t1")
    src = s.root.parent.parent / "raw.npy"
    np.save(src, np.arange(64, dtype=np.float64).reshape(8, 8) * 1e-12)

    staged = stage(s, path=str(src), name="probe")
    assert staged.kind == "array"
    assert (s.inputs_dir / "probe.npz").is_file()

    res, h = _run_script(s, '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mastdata

arrays, meta = mastdata.load("probe")
a = arrays["data"]
fig = plt.figure(); plt.imshow(a); mastdata.savefig(fig, "probe.png"); plt.close(fig)
mastdata.save_result(mean_m=float(a.mean()), n=int(a.size))
print("OK", a.shape)
''')
    assert res.ok, res.stderr[-800:]
    assert "OK (8, 8)" in res.stdout
    assert "probe.png" in h.new_files and "result.json" in h.new_files
    assert h.figures, "图没进图库"
    assert Path(h.figures[0]).is_file()


@needs_runtime
def test_figures_reach_the_shared_pool_that_list_figures_reads():
    """图必须能被 ``list_figures()`` 找到 —— 那是 paper_writing 唯一的读路径。"""
    from mast.agents._shared.figure_tools import list_figures

    s = get_session(experiment_id="exp1", thread_id="t1")
    res, h = _run_script(s, '''
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mastdata
fig = plt.figure(); plt.plot([1, 2, 3])
mastdata.savefig(fig, "trend.png"); plt.close(fig)
''')
    assert res.ok, res.stderr[-800:]
    assert h.figures
    listed = str(list_figures.invoke({}) if hasattr(list_figures, "invoke")
                 else list_figures())
    assert "trend" in listed, f"list_figures 看不见这张图：{listed[:300]}"


@needs_runtime
def test_result_json_survives_byte_for_byte():
    """``3.2e-12`` 这种数必须原样回来 —— 模型是**读**到它，不是重建它。"""
    s = get_session(experiment_id="exp1", thread_id="t1")
    res, h = _run_script(s, '''
import mastdata
mastdata.save_result(tunnel_current_A=3.2e-12, gap_eV=1.37e-3,
                     label="Au(111) herringbone")
''')
    assert res.ok, res.stderr[-800:]
    assert "3.2e-12" in h.result_json, f"数值形态变了：{h.result_json}"
    on_disk = (s.out_dir / "result.json").read_text(encoding="utf-8")
    assert json.loads(h.result_json) == json.loads(on_disk)


@needs_runtime
def test_images_for_toolmessage_respects_the_shared_cap():
    """上限从 ``vision_mw`` import，不重打字面量。"""
    from mast.agents._shared.vision_mw import MAX_IMAGES_PER_REQUEST

    s = get_session(experiment_id="exp1", thread_id="t1")
    res, h = _run_script(s, '''
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mastdata
for i in range(5):
    fig = plt.figure(); plt.plot([i, i + 1])
    mastdata.savefig(fig, f"f{i}.png"); plt.close(fig)
''')
    assert res.ok, res.stderr[-800:]
    assert len(h.figures) == 5
    assert len(images_for_toolmessage(h)) == MAX_IMAGES_PER_REQUEST


@needs_runtime
def test_a_timeout_is_reported_with_partial_output():
    """超时要说清楚，而且**还留着已经打出来的东西**。

    这是「用文件不用 PIPE」买到的：``communicate()`` 那条路上超时就什么都没有了，
    而部分输出往往正好指出它卡在哪一步。
    """
    s = get_session(experiment_id="exp1", thread_id="t1")
    res, _h = _run_script(s, '''
import sys, time
print("phase-1 done", flush=True)
while True:
    time.sleep(0.05)
''', timeout_s=3)
    assert res.timed_out
    assert res.returncode is None
    assert "3" in res.killed_reason
    assert "phase-1 done" in res.stdout, "超时后连已有输出都没了"


@needs_runtime
def test_a_traceback_comes_back_whole():
    s = get_session(experiment_id="exp1", thread_id="t1")
    res, _h = _run_script(s, "import numpy\ndef f():\n    return numpy.zeros(3)[9]\nf()\n")
    assert not res.ok
    assert "IndexError" in res.stderr and "in f" in res.stderr


# ── 真 .sxm ─────────────────────────────────────────────────────────
@needs_runtime
@needs_real_sxm
def test_staging_a_real_sxm_carries_the_physical_scale():
    """物理尺度由主进程算好写进 manifest —— 模型一个数字都不用打。

    ⚠️ 这里有个实测才发现的坑：``sxm_frame_meta`` 返回的
    ``scan_range`` / ``scan_offset`` 是**字符串**（只有 ``scan_pixels`` 是列表）。
    第一版只按 list 判断，``nm_per_px`` 静默变成 None，脚本就得自己去 split 那个
    字符串 —— 正好是我们想替它省掉的那一步。
    """
    s = get_session(experiment_id="exp1", thread_id="t1")
    stage(s, path=str(REAL_SXM), name="real")
    e = json.loads((s.inputs_dir / "manifest.json").read_text(encoding="utf-8"))["inputs"][0]

    assert e["nm_per_px"], "物理尺度没算出来"
    assert e["nm_per_px"][0] == pytest.approx(5.0 / 512, rel=1e-3)
    assert isinstance(e["bias_V"], float)
    assert e["parser"] == "mast.io.nanonis_files.read_sxm"
    assert isinstance(e["frame"]["scan_range"], list), (
        "scan_range 还是字符串 —— 脚本会拿到一串文本而不是数")
    assert any(k.startswith("z_") for k in e["arrays"])
    assert e["arrays"][next(k for k in e["arrays"] if k.startswith("z_"))]["unit"] == "m"


@needs_runtime
@needs_real_sxm
def test_a_real_analysis_runs_end_to_end():
    """真数据 + scipy + 出图 + 数值回传，一整条。"""
    s = get_session(experiment_id="exp1", thread_id="t1")
    stage(s, path=str(REAL_SXM), name="scan")
    res, h = _run_script(s, '''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mastdata

arrays, meta = mastdata.load("scan")
key = [k for k in arrays if k.startswith("z_")][0]
z = arrays[key]
yy, xx = np.mgrid[0:z.shape[0], 0:z.shape[1]]
A = np.c_[xx.ravel(), yy.ravel(), np.ones(z.size)]
coef, *_ = np.linalg.lstsq(A, z.ravel(), rcond=None)
flat = z - (A @ coef).reshape(z.shape)

fig = plt.figure(); plt.imshow(flat * 1e12, cmap="afmhot")
mastdata.savefig(fig, "flat.png"); plt.close(fig)
mastdata.save_result(rms_pm=float(np.std(flat) * 1e12),
                     nm_per_px=meta["nm_per_px"][0])
print("ANALYSIS OK", z.shape)
''')
    assert res.ok, res.stderr[-1000:]
    assert "ANALYSIS OK (512, 512)" in res.stdout
    assert h.figures and h.result_json
    assert json.loads(h.result_json)["rms_pm"] > 0
