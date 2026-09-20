"""拷进会话目录改名 ``mastdata.py`` —— 分析脚本的薄 helper。

**零 ``mast`` import。** 它跑在一个没有 MAST 的解释器里（那正是 B1 的内容），
只用标准库和 numpy。这一条由 ``test_child_helpers_import_no_mast`` 用 AST 强制。

它做四件事，每件都是「省掉一次容易出错的手工步骤」：

* :func:`load` —— 读 ``py_stage_data`` 放进来的 npz，**连同物理尺度一起**。
  ``nm_per_px`` / ``bias_V`` 是主进程用 ``sxm_frame_meta`` 算好写进 manifest 的，
  脚本直接读就行 —— 不用重推，模型也不用把这些数字打一遍
  （一个 ``3.2e-12`` 被复述成 ``3.2`` 就是一次静默的三个数量级）。
* :func:`scan_dirs` —— **主动告诉**脚本数据在哪。不设读白名单，但 agent 不知道
  路径就等于没有权限。
* :func:`savefig` —— 存到 ``out/``，会被自动收进图库，下游写报告的 agent 找得到，
  而且模型下一轮**看得见自己画的图**。
* :func:`save_result` —— 数值结论写进 ``out/result.json``，原样回传。
  这是让数字**不经过模型 token 流**的那条路。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent

__all__ = ["load", "manifest", "names", "out", "savefig", "save_result",
           "scan_dirs", "session_dir"]


def session_dir() -> Path:
    """本次会话的根目录。"""
    return _HERE


def manifest() -> dict:
    """``inputs/manifest.json`` —— 每份输入的来源、数组、单位、物理尺度。"""
    p = _HERE / "inputs" / "manifest.json"
    if not p.is_file():
        return {"schema": 1, "inputs": []}
    return json.loads(p.read_text(encoding="utf-8"))


def names() -> list[str]:
    """已经放进来的输入名。"""
    return [str(e.get("name")) for e in manifest().get("inputs", []) if e.get("name")]


def _entry(name: str) -> dict:
    for e in manifest().get("inputs", []):
        if e.get("name") == name:
            return e
    have = ", ".join(names()) or "（还没有）"
    raise KeyError(
        f"输入 {name!r} 不在本会话里。现有：{have}。\n"
        f"用 py_stage_data 放一份进来，或者直接用 mastkit 读原文件："
        f"scan_dirs() 会告诉你数据在哪。")


def load(name: str) -> tuple[dict[str, "np.ndarray"], dict]:
    """``(arrays, meta)``。

    ``arrays`` 是 ``{通道名: ndarray}``；``meta`` 至少含 ``nm_per_px`` /
    ``bias_V`` / ``frame`` / ``source_path`` / ``arrays``（每个通道的单位）。

    单位在 ``meta["arrays"][通道]["unit"]`` 里（``"m"`` / ``"A"`` …）—— 高度是
    **米**不是纳米，电流是**安培**不是皮安。画图前记得换算，别把 SI 值当 nm 用。
    """
    e = _entry(name)
    npz = np.load(_HERE / e["file"], allow_pickle=False)
    arrays = {k: npz[k] for k in npz.files}
    meta = {k: v for k, v in e.items() if k != "file"}
    return arrays, meta


def scan_dirs() -> list[str]:
    """已知的测量数据目录（只读用）。

    这里**不设读白名单** —— 脚本能读它有权限的任何路径。这个函数只是把「数据在
    哪」主动说出来，省掉一轮猜路径。
    """
    raw = os.environ.get("MAST_PYEXEC_SCAN_DIRS", "")
    return [p for p in raw.split(os.pathsep) if p.strip()]


def out(filename: str) -> str:
    """``out/`` 下的绝对路径（目录会自动建）。

    ``out/`` 是**推荐**落点：这里的新文件会被自动收走（图进图库、result.json
    原样回传）。往别处写也允许，只是要自己在 result.json 里报路径。
    """
    d = _HERE / "out"
    d.mkdir(parents=True, exist_ok=True)
    return str(d / filename)


def savefig(fig, name: str, *, dpi: int = 150) -> str:
    """存一张图到 ``out/``，返回路径。

    存进来的图会被拷进图库（``list_figures`` 只读那一个目录），而且**你下一轮会
    看见它** —— 可以据此判断这张图对不对，再决定要不要重画。
    """
    path = out(name if name.lower().endswith((".png", ".jpg", ".jpeg", ".svg", ".pdf"))
               else name + ".png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


def save_result(**metrics) -> str:
    """把数值结论写进 ``out/result.json``（原子替换）。

    **这是让数字不经过你的 token 流的那条路。** 这个文件会被逐字节回传，并且可以
    直接喂给 ``record_analysis(metrics_path=...)`` —— 从 ``curve_fit`` 到最终记录
    的整条路径上，没有任何一步需要你把数字重打一遍。

    值可以是数、字符串、列表、字典 —— 任何 JSON 能表达的东西。
    """
    path = Path(out("result.json"))
    existing: dict = {}
    if path.is_file():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            existing = {}
    if not isinstance(existing, dict):
        existing = {}
    existing.update(metrics)
    tmp = path.with_suffix(".json.part")
    tmp.write_text(json.dumps(existing, ensure_ascii=False, indent=1, default=str),
                   encoding="utf-8")
    os.replace(tmp, path)
    return str(path)
