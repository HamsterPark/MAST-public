"""原始数据出口：列文件、取原始字节、取服务端统一朝向后的帧。

## 为什么要有它

HTTP 上原来只有渲染后的 PNG（≤1024 px）。拿 PNG 算判据会得到错的数 —— 外部 agent
于是只能另开 scp 通道拉 ``.sxm``，再自己解析；而 ``.sxm`` 有两条很容易做反的约定
（反向扫描的数据沿 x 镜像存储；``scan_dir=up`` 的第一行是图的底边），每一个自己写
解析器的人都要各踩一遍。``/data/frame`` 在服务端用仓里唯一的那份实现
（``io.nanonis_files.read_sxm`` + ``sxm_oriented_frames``）把朝向一次处理对。

## 路径

**只许读允许根目录之内的文件**：数据搜索目录（与 Data 页同一份）+ 实验根目录。
先 ``resolve()`` 再判归属（``..``、符号链接、UNC 路径都在这一步被挡住），再判扩展名
白名单。**绝不转发客户端给的目录** —— ``records_export._scan_search_dirs`` 的 ``extra``
参数是排他覆盖，转发它等于任意目录读。
"""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Query, Request
from fastapi.responses import FileResponse, Response

from mast.api.ext.common import ExtError, ctx_of, runtime_of

logger = logging.getLogger(__name__)

router = APIRouter(tags=["data"])

#: 允许下载的扩展名：测量数据与它们的图。**不含** .db / .json / .env / .py —— 允许根里
#: 也放着记录库、配置与清单，那些不是「数据」。
ALLOWED_EXTS: frozenset[str] = frozenset({
    ".sxm", ".dat", ".3ds", ".txt", ".csv", ".png", ".npy", ".npz",
})


def allowed_roots(request: Request) -> list[Path]:
    roots: list[str] = []
    try:
        from mast.api.routes.records_export import _scan_search_dirs

        roots += _scan_search_dirs(ctx_of(request), None)      # 永远不传客户端的目录
    except Exception as exc:  # noqa: BLE001
        logger.debug("ext data: 搜索目录不可用(%s)", exc)
    try:
        from mast.core.experiment_paths import experiment_root

        roots.append(str(experiment_root()))
    except Exception:  # noqa: BLE001
        pass
    out: list[Path] = []
    for r in roots:
        try:
            p = Path(r).expanduser().resolve()
        except Exception:  # noqa: BLE001
            continue
        if p not in out:
            out.append(p)
    return out


def confine(request: Request, raw: str) -> Path:
    """把客户端给的路径解析成允许根之内、白名单扩展名的现存文件；否则抛 ExtError。"""
    text = str(raw or "").strip()
    if not text:
        raise ExtError(422, "bad_path", "path 不能为空")
    if text.startswith(("\\\\", "//")):
        raise ExtError(403, "path_not_allowed", "不接受网络路径")
    try:
        p = Path(text).expanduser().resolve(strict=True)
    except (OSError, RuntimeError):
        raise ExtError(404, "not_found", f"文件不存在：{text}") from None
    if not p.is_file():
        raise ExtError(404, "not_found", f"不是文件：{text}")
    if p.suffix.lower() not in ALLOWED_EXTS:
        raise ExtError(403, "extension_not_allowed",
                       f"只允许下载这些类型：{sorted(ALLOWED_EXTS)}")
    roots = allowed_roots(request)
    if not any(p.is_relative_to(r) for r in roots):
        raise ExtError(403, "path_not_allowed",
                       "只允许读数据搜索目录与实验根目录之内的文件",
                       allowed_roots=[str(r) for r in roots])
    return p


@router.get("/data/files")
def list_files(request: Request, n: int = Query(20, ge=1, le=200),
               ext: str | None = Query(None, description="逗号分隔，如 'sxm,dat'"),
               offset: int = Query(0, ge=0)):
    """最近的数据文件（mtime 新的在前，同一份数据的多个拷贝折叠成一条）。"""
    from mast.api.routes.records_export import latest_scans

    res = latest_scans(request, n=n, dir=None, ext=ext, offset=offset)
    body = res.model_dump() if hasattr(res, "model_dump") else dict(res)
    return {"files": body.get("scans") or [], "count": body.get("count", 0),
            "has_more": body.get("has_more", False), "degraded": body.get("degraded", False),
            "counts_by_ext": body.get("counts_by_ext") or {}}


@router.get("/data/file")
def get_file(request: Request, path: str = Query(..., description="文件的绝对路径")):
    """原始字节（``application/octet-stream``）。"""
    p = confine(request, path)
    return FileResponse(str(p), media_type="application/octet-stream", filename=p.name)


def _ascii_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=True, default=str, separators=(",", ":"))


@router.get("/data/frame")
def get_frame(request: Request, path: str = Query(..., description=".sxm 文件的绝对路径"),
              channel: str = Query("Z", description="通道名，如 Z / Current")):
    """一个 ``.sxm`` 通道的帧，服务端统一朝向后打成 ``.npz``：

    * ``forward`` / ``backward``：第 0 行是图的**顶边**（``scan_dir=up`` 已翻转），
      backward 已**去镜像**（与 forward 同一几何朝向）；只有 backward 的通道，那一块
      作为 ``forward`` 给出，``served_block`` 如实写明；
    * ``meta_json``：宽高（nm）、nm/px、偏压、设定点、扫描方向、单位、记录时间、
      可用通道；同一份元数据也放在响应头 ``X-MAST-Frame-Meta``（ASCII JSON）。

    读：``numpy.load(io.BytesIO(body), allow_pickle=False)``。
    """
    p = confine(request, path)
    if p.suffix.lower() != ".sxm":
        raise ExtError(422, "not_sxm", "/data/frame 只接受 .sxm；其他文件用 /data/file 取原始字节")
    import numpy as np

    from mast.io.nanonis_files import read_sxm, sxm_oriented_frames

    try:
        scan = read_sxm(str(p))
    except Exception as exc:  # noqa: BLE001
        raise ExtError(422, "unreadable", f"读不了这个 .sxm：{type(exc).__name__}: {exc}") from None
    channels = sorted((scan or {}).get("channels") or {})
    raw = ((scan or {}).get("channels") or {}).get(channel)
    if raw is None:
        lowered = {str(k).lower(): k for k in channels}
        key = lowered.get(str(channel).lower())
        raw = ((scan or {}).get("channels") or {}).get(key) if key else None
    if raw is None:
        raise ExtError(404, "unknown_channel", f"这个文件没有通道 {channel!r}", channels=channels)
    served = "forward" if raw.get("forward") is not None else "backward (un-mirrored, served as forward)"
    fr = sxm_oriented_frames(scan, channel)
    fwd, bwd = fr.get("forward"), fr.get("backward")
    if fwd is None:
        raise ExtError(422, "empty_channel", f"通道 {channel!r} 没有数据")
    meta = {k: fr.get(k) for k in ("channel", "nm_per_px", "width_nm", "height_nm", "bias_v",
                                   "setpoint_a", "scan_dir", "unit", "rec_time")}
    meta.update({"served_block": served, "has_backward": bwd is not None,
                 "shape": list(np.asarray(fwd).shape), "channels": channels,
                 "orientation": "row 0 = top edge; backward un-mirrored",
                 "file": p.name})
    buf = io.BytesIO()
    arrays = {"forward": np.asarray(fwd), "meta_json": np.array(json.dumps(meta, default=str))}
    if bwd is not None:
        arrays["backward"] = np.asarray(bwd)
    np.savez_compressed(buf, **arrays)
    from urllib.parse import quote

    name = f"{p.stem}_{channel}.npz"
    ascii_name = name.encode("ascii", "ignore").decode("ascii").replace('"', "") or "frame.npz"
    disposition = f'attachment; filename="{ascii_name}"'
    if ascii_name != name:              # 非 ASCII 文件名：RFC 5987 形式另给一份
        disposition += f"; filename*=utf-8''{quote(name)}"
    return Response(content=buf.getvalue(), media_type="application/octet-stream",
                    headers={"Content-Disposition": disposition,
                             "X-MAST-Frame-Meta": _ascii_json(meta)})


__all__ = ["ALLOWED_EXTS", "allowed_roots", "confine", "router"]
