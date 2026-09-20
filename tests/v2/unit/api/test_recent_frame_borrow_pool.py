"""近期扫描借图池应先按扩展名筛选，再按数量截断。

使用较新的非 SXM 文件覆盖排序窗口，验证它们不会挤掉仍然可用的 SXM 帧。
这一性质适用于各种非目标文件类型。"""

from __future__ import annotations

import os
import time

import pytest

from mast.webui.scan_preview import _collect_scans, get_latest_scans


def _touch(path, mtime: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x", encoding="utf-8")
    os.utime(path, (mtime, mtime))


def _rig_like(tmp_path, *, n_dat: int = 40, n_sxm: int = 26):
    """一个正在做点谱的会话：.dat 全都比 .sxm 新。"""
    now = time.time()
    exp = tmp_path / "exp"
    for i in range(n_dat):
        _touch(exp / "sts" / f"Bias-Spectroscopy{i:03d}.dat", now - i)
    for i in range(n_sxm):
        _touch(exp / "scans" / f"synthetic_{i:04d}.sxm", now - 100000 - i)
    return tmp_path


def test_exts_filter_applies_before_the_limit(tmp_path):
    """要 24 个 .sxm 就得到 24 个 .sxm —— 哪怕最新的 40 个文件全是别的类型。"""
    _rig_like(tmp_path)
    got = get_latest_scans(24, str(tmp_path), exts=(".sxm",))
    assert len(got) == 24, [p.name for p in got]
    assert all(p.suffix == ".sxm" for p in got)


def test_the_old_filter_after_slice_would_have_returned_nothing(tmp_path):
    """钉住这个缺陷本身：旧写法在同样的目录上一张也拿不到。

    这不是在测已删掉的代码，而是证明上面那条断言不是白给的 —— 如果哪天
    ``exts`` 参数被绕过、退回成「先取再筛」，这里的 0 就是那时的返回值。
    """
    _rig_like(tmp_path)
    old_style = [p for p in get_latest_scans(24, str(tmp_path)) if p.suffix == ".sxm"]
    assert old_style == [], "前提已变：.dat 不再占满窗口，这条对照就失去意义了"


def test_recent_scan_paths_survives_a_spectroscopy_session(tmp_path, monkeypatch):
    """端到端走 ``_recent_scan_paths``：借图池装满，SCAN_COMPLETE 有图可借。"""
    from mast.api.routes import vision as vision_routes

    _rig_like(tmp_path)
    monkeypatch.setattr(vision_routes, "_scan_search_dirs", lambda _app: [str(tmp_path)])

    paths = vision_routes._recent_scan_paths(object(), 24)
    assert len(paths) == 24
    assert all(p.lower().endswith(".sxm") for p in paths)
    # 借图是按「下一个次新」逐个取的，顺序必须是 mtime 倒序。
    assert paths == sorted(paths, key=lambda p: os.stat(p).st_mtime, reverse=True)


def test_env_sidecar_exclusion_still_holds_with_exts(tmp_path, monkeypatch):
    """新增的 ``exts`` 不得绕开 env/ 边车规则（两条防线要能叠加）。"""
    from mast.api.routes import vision as vision_routes

    now = time.time()
    exp = tmp_path / "exp"
    # 边车里也放一个 .sxm —— 扩展名对得上，但它住在 env/ 里。
    _touch(exp / "env" / "trap.sxm", now)
    _touch(exp / "scans" / "real_0001.sxm", now - 3600)
    monkeypatch.setattr(vision_routes, "_scan_search_dirs", lambda _app: [str(tmp_path)])

    paths = vision_routes._recent_scan_paths(object(), 24)
    assert [os.path.basename(p) for p in paths] == ["real_0001.sxm"]


def test_no_search_dirs_is_an_empty_pool_not_an_error(tmp_path, monkeypatch):
    """没有可搜的目录时安静地返回空 —— 借不到图是一种正常状态，不是 500。"""
    from mast.api.routes import vision as vision_routes

    monkeypatch.setattr(vision_routes, "_scan_search_dirs", lambda _app: [])
    assert vision_routes._recent_scan_paths(object(), 24) == []


@pytest.mark.parametrize("exts", [None, ()])
def test_default_behaviour_is_unchanged_for_every_other_caller(tmp_path, exts):
    """``exts`` 缺省 / 空元组 = 老语义（全部已识别类型）。

    ``get_latest_scans`` 还有四个调用方（数据条、get_latest_scan_info、
    assess_quality、quickask），它们要的就是「任意可渲染的数据文件」。
    """
    now = time.time()
    _touch(tmp_path / "d" / "a.dat", now)
    _touch(tmp_path / "d" / "b.sxm", now - 10)
    got = get_latest_scans(10, str(tmp_path), exts=exts)
    assert [p.name for p in got] == ["a.dat", "b.sxm"]


def test_collect_scans_exts_is_case_insensitive(tmp_path):
    """大写扩展名照收 —— Nanonis 在不同版本里写过 .SXM。"""
    now = time.time()
    _touch(tmp_path / "d" / "UPPER.SXM", now)
    got = _collect_scans(str(tmp_path), exts=(".SXM",))
    assert [p.name for p in got] == ["UPPER.SXM"]


# ── 两个调用方都必须把扩展名交给「搜索」 ────────────────────────────────────
#
# 这个缺陷在 vision.py 里有**两份**（近期帧借图池 + 扫描地图底图），因为过滤是在
# 调用方手抄的 —— 抄写就是漂移的种子。下面两条各钉一个调用点：断言它们把
# ``exts`` 传下去，而不是自己在结果上筛。任何一处被改回手抄式过滤都会红。


def _spy_get_latest_scans(monkeypatch):
    """把 scan_preview.get_latest_scans 换成记录器（两处都是函数内 import，
    所以改模块属性就能拦到）。返回空列表，让调用方走各自的空路径。"""
    calls: list[dict] = []

    def _recorder(n, *search_dirs, max_depth=3, exts=None):
        calls.append({"n": n, "dirs": list(search_dirs), "exts": exts})
        return []

    import mast.webui.scan_preview as sp

    monkeypatch.setattr(sp, "get_latest_scans", _recorder)
    return calls


def test_recent_scan_paths_asks_the_search_for_sxm(tmp_path, monkeypatch):
    """近期帧借图池。"""
    from mast.api.routes import vision as vision_routes

    calls = _spy_get_latest_scans(monkeypatch)
    monkeypatch.setattr(vision_routes, "_scan_search_dirs", lambda _app: [str(tmp_path)])

    vision_routes._recent_scan_paths(object(), 24)
    assert len(calls) == 1
    assert calls[0]["exts"] == (".sxm",), "扩展名必须交给搜索，不能在结果上筛"
    assert calls[0]["n"] == 24


def test_scan_map_underlay_asks_the_search_for_sxm(tmp_path, monkeypatch):
    """扫描地图底图（同一形状的第二份）。

    这一处比借图池更隐蔽：底图饿死时地图不会报错，只是从「真实表面马赛克」
    退化成一堆空outline，而没有任何地方说为什么。
    """
    from mast.api.routes import vision as vision_routes

    calls = _spy_get_latest_scans(monkeypatch)

    class _Cfg:
        experiments_dir = str(tmp_path)

    class _App:
        config = _Cfg()

    out = vision_routes._recent_scan_images(_App(), limit=12)
    assert out == []  # 记录器返回空 → 底图为空，但调用已经被拦到
    assert len(calls) == 1
    assert calls[0]["exts"] == (".sxm",), "扩展名必须交给搜索，不能在结果上筛"
