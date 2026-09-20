"""扫描地图底图 —— 取向与叠放顺序。

已知问题:扫图的 up 和 down 只有一个方向会自动显示在扫描地图上。

两个独立缺陷凑出这句话，两组测试各钉一个：

* **取向**：Nanonis 按采集顺序写行。``down`` 第一条线是画面上边，``up`` 第一条线是
  画面**下**边。底图缩略图从来没读过 ``:SCAN_DIR:``，一律 ``origin="lower"`` ——
  于是同一片表面的两个方向在地图上是上下镜像的。bouncy 开着时相邻帧方向交替。
* **叠放顺序**：底图列表按 mtime **降序**交给画布，而画布后画的盖前画的（后端注释
  自己就是这么说的：「live frame goes LAST so it paints over…」）。同一个 footprint
  被重复成像时，旧帧会遮住新帧。测试通过构造不同时间戳验证正确叠放顺序。

第三条是结构闸门：这个翻转的规则只允许有**一份**。#94 的直接成因就是它有三种私人
拼写（``sxm_oriented_frames`` / ``mosaic`` / 而缩略图那条路一份都没有）。

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest tests/v2/unit/io/test_scan_map_underlay.py -q
"""
from __future__ import annotations

import ast
import base64
import io as _io
import os
import struct
import sys
import tokenize
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

import numpy as np
import pytest

from mast.io.mosaic import load_scan_for_mosaic
from mast.io.nanonis_files import read_sxm, rows_top_first, sxm_oriented_frames
from mast.webui.scan_preview import render_scan_thumbnail

PX = 24


def _write_sxm(path, *, fwd, scan_dir="down", offset=(0.0, 0.0), range_m=5e-8):
    """最小但真实的 Nanonis ``.sxm``（大端 float32，头部按真文件格式）。"""
    ny, nx = fwd.shape
    header = (
        ":NANONIS_VERSION:\n2\n"
        ":SCANIT_TYPE:\n\t FLOAT            MSBFIRST\n"
        ":REC_DATE:\n 09.08.2026\n"
        ":REC_TIME:\n12:00:00\n"
        ":BIAS:\n\t1.000000E+0\n"
        f":SCAN_PIXELS:\n{nx:>10d}{ny:>10d}\n"
        f":SCAN_OFFSET:\n{offset[0]:>19.6E}{offset[1]:>19.6E}\n"
        f":SCAN_RANGE:\n{range_m:>19.6E}{range_m:>19.6E}\n"
        ":SCAN_ANGLE:\n\t0.000E+0\n"
        f":SCAN_DIR:\n{scan_dir}\n"
        ":Z-CONTROLLER>SETPOINT:\n20.0000E-12\n"
        ":DATA_INFO:\n"
        "\tChannel\tName\tUnit\tDirection\tCalibration\tOffset\n"
        "\t14\tZ\tm\tfwd\t9.000E-9\t0.000E+0\n"
        ":SCANIT_END:\n\n"
    )
    blob = struct.pack(">%df" % fwd.size,
                       *fwd.astype(np.float32).ravel().tolist())
    Path(path).write_bytes(header.encode("utf-8") + b"\x1a\x04" + blob)
    return Path(path)


def _surface(n=PX):
    """一片**上下不对称**的表面：最上面几行是亮台阶，其余是暗底。

    行 0 = 画面的**上**边（高 y）。上下对称的图案在这里等于没有判据 —— 翻转前后
    长得一模一样，测试会全绿而缺陷原样留着。
    """
    a = np.full((n, n), 1e-11, dtype=np.float64)
    a[: n // 6, :] = 9e-11          # 顶部亮条
    a += np.linspace(0.0, 4e-12, n)[None, :]   # 沿 x 的弱斜坡，左右也不对称
    return a


def _png_gray(data_uri):
    """data-URI PNG → 灰度 2D ndarray（行 0 = 图片顶端）。"""
    assert data_uri and "," in data_uri, "缩略图没渲染出来"
    import matplotlib.image as mpimg
    raw = base64.b64decode(data_uri.split(",", 1)[1])
    img = mpimg.imread(_io.BytesIO(raw), format="png")
    return img[..., :3].mean(axis=-1) if img.ndim == 3 else img


# ── 一、翻转规则本身 ─────────────────────────────────────────────────────────

def test_rows_top_first_flips_up_only():
    a = _surface(8)
    assert np.array_equal(rows_top_first(a, "down"), a)
    assert np.array_equal(rows_top_first(a, "up"), np.flipud(a))
    assert np.array_equal(rows_top_first(a, "UP  "), np.flipud(a)), "大小写/空白"


def test_rows_top_first_leaves_unreadable_direction_alone():
    """读不到方向时**不动它** —— 恒等是唯一不宣称我们不知道的事情的操作。

    ``mosaic`` 从前按 ``!= "down"`` 判，也就是**方向读不出来的文件照翻不误**。两条
    规则在任何真 Nanonis 文件上结论相同（它们都写 up 或 down），只在证据缺席的地方
    分道扬镳，而那正是该给安静答案的地方。这条钉的就是那个分歧。
    """
    a = _surface(8)
    for unknown in (None, "", "   ", "sideways"):
        assert np.array_equal(rows_top_first(a, unknown), a), (
            f"scan_dir={unknown!r} 读不出来，却还是翻了一下")


def test_three_readers_agree_on_which_row_is_the_top(tmp_path):
    """同一片表面，两个扫描方向，三条读取路径 —— 必须得到同一个「上边」。"""
    surface = _surface()
    d = _write_sxm(tmp_path / "d.sxm", fwd=surface, scan_dir="down")
    # up 扫描先采下边，所以硬件写下去的块是倒过来的
    u = _write_sxm(tmp_path / "u.sxm", fwd=np.flipud(surface), scan_dir="up")

    for reader_name, of_down, of_up in (
        ("sxm_oriented_frames",
         sxm_oriented_frames(read_sxm(str(d)), "Z")["forward"],
         sxm_oriented_frames(read_sxm(str(u)), "Z")["forward"]),
        ("load_scan_for_mosaic",
         load_scan_for_mosaic(str(d))["data"],
         load_scan_for_mosaic(str(u))["data"]),
    ):
        assert np.allclose(of_down, surface, atol=1e-15), reader_name
        assert np.allclose(of_up, surface, atol=1e-15), (
            f"{reader_name}: up 帧没有被翻回来")


def test_map_underlay_renders_both_directions_the_same_way(tmp_path):
    """#94 的正题：地图底图上，一片表面的 up 帧与 down 帧必须长得一样，
    **而且亮条在上面**。

    只断言「两者相同」是不够的 —— 两张一致地上下颠倒的图同样会通过，而地图把它们
    按真实 footprint 摆下去，上下颠倒就是把表面画反了。所以第二条断言钉的是绝对
    方向，不是一致性。
    """
    surface = _surface()
    d = _write_sxm(tmp_path / "d.sxm", fwd=surface, scan_dir="down")
    u = _write_sxm(tmp_path / "u.sxm", fwd=np.flipud(surface), scan_dir="up")

    img_d = _png_gray(render_scan_thumbnail(str(d), size=120))
    img_u = _png_gray(render_scan_thumbnail(str(u), size=120))

    assert img_d.shape == img_u.shape
    assert np.allclose(img_d, img_u, atol=0.02), (
        "同一片表面的两个扫描方向在地图底图上不一样 —— 相邻两帧上下镜像")

    n = img_d.shape[0]
    top = float(img_d[: n // 6, :].mean())
    bottom = float(img_d[-(n // 6):, :].mean())
    assert top > bottom, (
        "亮条画到下面去了 —— 底图整体上下颠倒（两个方向一致地画反也会通过上一条）")


# ── 二、叠放顺序 ─────────────────────────────────────────────────────────────

class _StubApp:
    """`_recent_scan_images` 摸到的全部 app 表面。"""

    def __init__(self, session_dir):
        self._session_dir = str(session_dir)
        self.config = type("C", (), {"experiments_dir": None})()

    def _resolve_session_dir(self):
        return self._session_dir


@pytest.fixture()
def isolated_root(tmp_path, monkeypatch):
    """底图搜索路径里有 ``project_root()/working-sessions`` —— 不隔离的话，这些断言
    读的是**开发机上真实的扫描文件**，本地绿、别处红，而且断言的对象根本不是造出来
    的那几帧（第一版当场撞上：拿回来 12 个 001_00xx）。"""
    monkeypatch.setenv("MAST2_PROJECT_ROOT", str(tmp_path / "_root"))
    return tmp_path


def _touch(path, when):
    os.utime(path, (when, when))


def test_underlay_is_oldest_first_so_the_newest_frame_lands_on_top(isolated_root):
    """画布后画的盖前画的，合成不同时间戳验证列表按旧到新排序。"""
    from mast.api.routes.vision import _recent_scan_images

    surface = _surface()
    names = ["a", "b", "c"]
    for i, nm in enumerate(names):
        p = _write_sxm(isolated_root / f"{nm}.sxm", fwd=surface,
                       scan_dir="down" if i % 2 == 0 else "up")
        _touch(p, 1_700_000_000 + i * 60)     # a 最旧, c 最新

    out = _recent_scan_images(_StubApp(isolated_root), limit=12)
    got = [Path(s.path).stem for s in out]
    assert got == names, f"底图不是旧→新：{got}"


def test_underlay_dedupes_the_same_file_spelled_two_ways(isolated_root):
    """同一合成文件的两种路径拼写必须被去重，避免重复传输和绘制。"""
    from mast.api.routes.vision import _recent_scan_images

    p = _write_sxm(isolated_root / "one.sxm", fwd=_surface())
    weird = str(p).replace("\\", "/")          # 同一个文件，另一种拼法
    if weird == str(p):                        # POSIX：换成 ./ 前缀
        weird = str(p.parent / "." / p.name)

    out = _recent_scan_images(_StubApp(isolated_root), limit=12,
                              extra_paths=[weird])
    assert len(out) == 1, f"同一个文件进了 {len(out)} 次"


def test_underlay_still_includes_an_imported_scan_from_outside_the_dirs(isolated_root):
    """去重不许顺手废掉「导入的扫描无论在哪、多旧都显示」这条。"""
    from mast.api.routes.vision import _recent_scan_images

    inside, outside = isolated_root / "sess", isolated_root / "elsewhere"
    inside.mkdir()
    outside.mkdir()
    a = _write_sxm(inside / "a.sxm", fwd=_surface())
    b = _write_sxm(outside / "b.sxm", fwd=_surface(), offset=(1e-7, 0.0))
    _touch(a, 1_700_000_000)
    _touch(b, 1_700_000_100)

    out = _recent_scan_images(_StubApp(inside), limit=12, extra_paths=[str(b)])
    assert [Path(s.path).stem for s in out] == ["a", "b"]


# ── 三、结构闸门：这条规则只允许有一份 ───────────────────────────────────────

_MAST_SRC = Path(_MASTV2_ROOT) / "mast"
#: 允许把 ``flipud`` 和 ``scan_dir`` 写在一起的唯一文件 —— 规则住在这里。
_FLIP_RULE_OWNER = "io/nanonis_files.py"


def _blank_comments_and_docstrings(src: str) -> str:
    """注释与 docstring 涂成空格（保留行列，其余字符串**保留**）。

    先剥再扫，是因为讲清楚一个错误写法最好的办法就是把它原样写进注释 —— 不剥的话，
    「写下教训」这件事本身会把闸门判红，而下一个人多半会去删那段注释。
    其它字符串字面量不能剥：``header.get("scan_dir")`` 里的正是关键标识符。
    """
    lines = src.splitlines(keepends=True)
    spans: list[tuple[int, int, int, int]] = []   # (l0, c0, l1, c1), 1-based line
    try:
        tree = ast.parse(src)
    except SyntaxError:                       # pragma: no cover
        return src
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not isinstance(body, list) or not body:
            continue
        first = body[0]
        if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)):
            spans.append((first.lineno, first.col_offset,
                          first.end_lineno, first.end_col_offset))
    try:
        for tok in tokenize.generate_tokens(_io.StringIO(src).readline):
            if tok.type == tokenize.COMMENT:
                spans.append((tok.start[0], tok.start[1],
                              tok.end[0], tok.end[1]))
    except (tokenize.TokenError, IndentationError):   # pragma: no cover
        pass
    for l0, c0, l1, c1 in spans:
        for ln in range(l0, l1 + 1):
            line = lines[ln - 1]
            a = c0 if ln == l0 else 0
            b = c1 if ln == l1 else len(line.rstrip("\r\n"))
            keep_nl = line[len(line.rstrip("\r\n")):]
            body_txt = line.rstrip("\r\n")
            lines[ln - 1] = (body_txt[:a] + " " * (b - a) + body_txt[b:]) + keep_nl
    return "".join(lines)


def test_the_scan_dir_flip_has_exactly_one_home():
    """``flipud`` 与 ``scan_dir`` 只许在一个文件里同时出现。

    #94 的直接成因是这个翻转有三种私人拼写，而第三条路（地图底图缩略图）一份都没有。
    第四种拼法出现时，这条会红。
    """
    scanned = 0
    owners: list[str] = []
    callers: list[str] = []
    for py in sorted(_MAST_SRC.rglob("*.py")):
        try:
            src = py.read_text(encoding="utf-8")
        except OSError:                        # pragma: no cover
            continue
        scanned += 1
        code = _blank_comments_and_docstrings(src)
        rel = py.relative_to(_MAST_SRC.parent).as_posix().removeprefix("mast/")
        if "rows_top_first" in code:
            callers.append(rel)
        # ``flipud(`` —— 带括号的**调用**，不是名字。第一版按裸名字匹配，当场把
        # `agents/data_processing/tools.py` 的 numpy 白名单（`"flipud",` 只是列表
        # 里的一个字符串）判成了第四处私人拼写。
        if "flipud(" in code and "scan_dir" in code:
            owners.append(rel)

    # 自检：一条匹配不到任何东西的闸门永远是绿的，和「确实没问题」一模一样。
    assert scanned >= 100, f"只扫到 {scanned} 个 .py —— 闸门没扫到东西"
    assert _FLIP_RULE_OWNER in callers, (
        f"{_FLIP_RULE_OWNER} 里找不到 rows_top_first —— 规则被改名或搬走了，"
        "闸门检查的是一个不存在的名字")
    assert len(callers) >= 3, (
        f"只有 {callers} 调用 rows_top_first —— 三条读取路径应该都在（"
        "nanonis_files / mosaic / webui.scan_preview）")

    assert owners == [_FLIP_RULE_OWNER], (
        f"scan_dir 的翻转规则出现在 {owners} —— 只允许 {_FLIP_RULE_OWNER} 一份。"
        "多一份私人拼写就是 #94 的成因。")


@pytest.mark.parametrize("consumer", [
    "io/mosaic.py",
    "webui/scan_preview.py",
])
def test_every_sxm_display_path_goes_through_the_rule(consumer):
    """两条把 .sxm 画出来的路径都必须调用那条规则（而不是自己再写一遍）。"""
    code = _blank_comments_and_docstrings(
        (_MAST_SRC / consumer).read_text(encoding="utf-8"))
    assert "rows_top_first" in code, f"{consumer} 没走单一真源"
