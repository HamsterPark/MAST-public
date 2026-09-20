"""扫描发现不能把持续更新的环境 CSV 边车当成扫描数据。

即使边车的修改时间较新，也不得占满扫描结果窗口；测试使用临时目录构造输入。
"""

from __future__ import annotations

import os
import time

from mast.webui.scan_preview import _collect_scans, get_latest_scans


def _touch(path, mtime: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x", encoding="utf-8")
    os.utime(path, (mtime, mtime))


def test_env_sidecar_csv_is_not_scan_data(tmp_path):
    """``<exp>/env/*.csv`` 永不出现在扫描发现结果里。"""
    now = time.time()
    exp = tmp_path / "2026-08-02__实验__abcd1234"
    # 合成环境边车的 mtime 晚于扫描文件。
    _touch(exp / "env" / "tunnel_current_2026-08-03.csv", now)
    _touch(exp / "env" / "vacuum_2026-08-03.csv", now)
    # 样品级切片走的是另一条路径，但直接父目录同样叫 env。
    _touch(exp / "samples" / "s1" / "env" / "SPM_COM3.csv", now)
    # 真正的扫描：更老。
    _touch(exp / "scans" / "synthetic_0001.sxm", now - 3600)

    found = [p.name for p in _collect_scans(str(tmp_path))]
    assert found == ["synthetic_0001.sxm"], found


def test_scans_survive_a_full_window_of_env_csv(tmp_path):
    """合成大量新边车，验证它们不会挤占扫描窗口。"""
    now = time.time()
    exp = tmp_path / "exp"
    for i in range(20):
        _touch(exp / "env" / f"sensor{i}_2026-08-03.csv", now)
    for i in range(15):
        _touch(exp / "scans" / f"synthetic_{i:04d}.sxm", now - 3600 - i)

    top12 = get_latest_scans(12, str(tmp_path))
    assert len(top12) == 12
    assert all(p.suffix == ".sxm" for p in top12), [p.name for p in top12]


def test_csv_outside_the_env_sidecar_is_still_discovered(tmp_path):
    """一般数值文本仍然是「数据」标签页认得的东西 —— 排除的是 env/ 这个目录，
    不是 .csv 这个扩展名。"""
    now = time.time()
    _touch(tmp_path / "sessions" / "sts_curve.csv", now)
    found = [p.name for p in _collect_scans(str(tmp_path))]
    assert found == ["sts_curve.csv"], found


def test_a_search_dir_the_caller_named_env_is_still_honoured(tmp_path):
    """边车规则只作用于**我们自己走进去的**目录。调用方直接指名一个叫 env 的
    目录（`?dir=` 覆盖、或站点就这么命名），那是他的意思，不该被静默清空。"""
    now = time.time()
    d = tmp_path / "env"
    _touch(d / "topo_0001.sxm", now)
    found = [p.name for p in _collect_scans(str(d))]
    assert found == ["topo_0001.sxm"], found


# ── 同一个形状的第二个落点（2026-08-05） ───────────────────────────


def test_the_experimental_monitor_csv_is_not_scan_data(tmp_path):
    """``<exp>/env/signals/monitor_*.csv`` 与 ``experiments/monitors/``。

    #53 修的是 ``env/``，而这两个写入方的**直接父目录**叫 signals / monitors，
    规则按直接父目录匹配，所以一个都没挡住。
    ``core.runtime.start_experimental_monitor`` 每 5 秒追加一行，于是
    「电流的 csv」重新霸占了整个窗口。
    """
    now = time.time()
    exp = tmp_path / "exp"
    _touch(exp / "env" / "signals" / "monitor_current_1785900000.csv", now)
    _touch(exp / "env" / "signals" / "monitor_bias_1785900001.csv", now)
    _touch(tmp_path / "monitors" / "monitor_current_1785900002.csv", now)
    _touch(exp / "scans" / "synthetic_0001.sxm", now - 3600)

    found = [p.name for p in _collect_scans(str(tmp_path))]
    assert found == ["synthetic_0001.sxm"], found


def test_scans_in_the_managed_sample_layout_are_reachable_at_all(tmp_path):
    """**这才是「看不到 sxm 了」的另一半：它们根本没被列出来过。**

    MAST 自管布局把扫描放在
    ``<exp>/samples/<sample>/raw/sxm/<file>``（``experiment_paths`` 称之为
    该处最深路径），从 experiments_dir 数下去是第 5 层；而遥测 CSV 在
    ``<exp>/env/signals/`` 只有第 3 层。旧的 max_depth=3 因此**看得见噪声、
    看不见信号** —— 正是用户描述的那个倒置。
    """
    now = time.time()
    exp = tmp_path / "2026-08-05__实验__abcd1234"
    smp = exp / "samples" / "0001__样品__ab12"
    _touch(smp / "raw" / "sxm" / "topo_0001.sxm", now - 60)
    _touch(smp / "raw" / "nanonis" / "insitu_0002.sxm", now - 30)
    _touch(smp / "raw" / "dat" / "sts_0001.dat", now - 90)

    names = sorted(p.name for p in _collect_scans(str(tmp_path)))
    assert names == ["insitu_0002.sxm", "sts_0001.dat", "topo_0001.sxm"], names


def test_the_depth_limit_is_the_thing_that_was_wrong(tmp_path):
    """反证：把深度调回 3，同一棵树就再也看不到扫描。

    没有这一条，上面那条测试并不能证明**深度**是原因 —— 它可能只是碰巧过了。
    """
    now = time.time()
    smp = tmp_path / "exp" / "samples" / "s1"
    _touch(smp / "raw" / "sxm" / "topo_0001.sxm", now)
    assert _collect_scans(str(tmp_path), max_depth=3) == []
    assert [p.name for p in _collect_scans(str(tmp_path))] == ["topo_0001.sxm"]
