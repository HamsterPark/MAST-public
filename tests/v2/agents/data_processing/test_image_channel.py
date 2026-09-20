"""出图工具必须把图挂上图像通道 —— 否则这个 agent 看不见自己的基线交付。

2026-08-19 之前的事实：``SkillImageMiddleware`` 挂在 data_processing 上，但
``IMAGES_KEY`` 全仓只有一处写点（``_shared/skill_adapter.py:1024-1026``），而
**DP 的工具一个都不走 skill adapter** —— 它们返回裸字符串。中间件在那儿等着，
一张图都没收到过。

这不只是「模型少看了点东西」：出图是这个 agent 的基线交付，而它无法判断自己刚交
付的东西对不对（色标压死了？画错通道了？）。那类问题只有看一眼才知道。

这份测试同时钉住**行为改变面被收窄**这个设计：``tool_call_id`` 为空时原样返回
字符串，所以几十个直调单测一条都没坏。
"""

from __future__ import annotations

import numpy as np
import pytest

from mast.agents._shared.vision_mw import IMAGES_KEY, MAX_IMAGES_PER_REQUEST
from mast.agents.data_processing import tools as dp


@pytest.fixture(autouse=True)
def _figures_to_tmp(tmp_path, monkeypatch):
    """图库指向 tmp。本仓被「测试写进用户真实数据」咬过五次。"""
    monkeypatch.setenv("MAST_FIGURES_DIR", str(tmp_path / "figures"))
    yield


@pytest.fixture
def scan_npy(tmp_path):
    p = tmp_path / "probe_scan.npy"
    yy, xx = np.mgrid[0:32, 0:32]
    np.save(p, (1e-10 * (xx * 0.3 + yy * 0.1)).astype(np.float64))
    return str(p)


def _images_of(ret) -> list[str]:
    """从工具返回里取出挂着的图像路径（没挂就返回空）。"""
    upd = getattr(ret, "update", None)
    if not isinstance(upd, dict):
        return []
    for msg in upd.get("messages", []):
        kw = getattr(msg, "additional_kwargs", None) or {}
        if IMAGES_KEY in kw:
            return list(kw[IMAGES_KEY])
    return []


def test_plot_scan_attaches_the_figure_it_just_drew(scan_npy):
    """这是整条回路的核心断言：画完的图必须到得了模型眼前。"""
    ret = dp.plot_scan.func(path=scan_npy, label="probe",
                            tool_call_id="call-1")
    assert "Saved figure" in str(ret), str(ret)
    imgs = _images_of(ret)
    assert imgs, "图没挂上 —— 模型看不见自己画的图"
    assert imgs[0].lower().endswith(".png")
    from pathlib import Path
    assert Path(imgs[0]).is_file()


def test_without_a_tool_call_id_the_return_is_still_a_plain_string(scan_npy):
    """行为改变面被刻意收窄到「只有跑在 agent 里时」。

    几十个直调单测对返回文本做断言（``.startswith`` / 正则 / ``.splitlines()``）。
    ``tool_call_id`` 只有框架会注入，直调时是空的 —— 于是那些测试一条都没坏。
    """
    ret = dp.plot_scan.func(path=scan_npy, label="probe")
    assert isinstance(ret, str)
    assert not _images_of(ret)


def test_a_failure_return_carries_no_images(tmp_path):
    """报错路径不能挂图 —— 挂一个不存在的路径，模型会收到空通道并困惑。"""
    ret = dp.plot_scan.func(path=str(tmp_path / "nope.npy"),
                            tool_call_id="call-1")
    assert "failed" in str(ret).lower()
    assert not _images_of(ret)


def test_only_paths_that_really_exist_are_attached(tmp_path):
    """判据是「文件真的在盘上」，不是「看起来像路径」。

    这条也解释了为什么不去记每个工具的字段名：那份对应关系没人维护，而
    「文件存在」这条判据不会过期。
    """
    real = tmp_path / "real.png"
    real.write_bytes(b"fake-png-bytes" * 4)   # 判据只看后缀 + 存在，内容无关
    fake = tmp_path / "ghost.png"

    got = dp._figures_in(f"Saved figure: {real}")
    assert got == [str(real)]

    assert dp._figures_in(f"Saved figure: {fake}") == [], "不存在的路径被挂上了"
    # JSON 载荷里的图像路径也要识别。
    # 用 json.dumps 造，免得手写转义 —— Windows 路径里的反斜杠在 JSON 里要双写。
    import json
    assert dp._figures_in("ok: " + json.dumps({"output_path": str(real)})) == [str(real)]
    assert dp._figures_in('processing ok: {"npy": "C:/x/out.npy"}') == [], (
        ".npy 数组文件不应进入图像通道")


def test_attached_images_respect_the_shared_cap(tmp_path):
    """上限从 ``vision_mw`` import，不重打字面量。"""
    lines = []
    for i in range(MAX_IMAGES_PER_REQUEST + 3):
        f = tmp_path / f"f{i}.png"
        f.write_bytes(b"fake-png-bytes")
        lines.append(f"Saved figure: {f}")
    ret = dp._with_figures("\n".join(lines), "call-1", "probe")
    assert len(_images_of(ret)) == MAX_IMAGES_PER_REQUEST


def test_every_figure_tool_routes_through_the_helper():
    """结构闸门：新增出图工具时别忘了接。

    「每个工具各自记得」这种事人肉查不齐 —— 本仓的教训是先写结构闸门。
    """
    import ast
    from pathlib import Path

    src = Path(dp.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    # Static public figure-tool contract; do not derive it from the source being checked.
    expected = {"plot_scan", "plot_spectrum", "mosaic_scans", "py_run"}
    assert expected <= {tool.name for tool in dp.AGENT_TOOLS}
    routed = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name not in expected:
            continue
        body = ast.dump(node)
        if "_with_figures" in body or "IMAGES_KEY" in body:
            routed.add(node.name)
    assert routed == expected, f"这些出图工具没接图像通道：{sorted(expected - routed)}"
