"""文档 API —— 五类文档的浏览、编辑、版本、差异、认领、导出。

设计文档：``docs/v2/design/document_and_library_management.md`` §3.10
数据层：``mast.documents``（``store()`` 是唯一写入口）

这个模块换了什么
----------------

旧实现把全局 ``data/{drafts,reviews}/*.md`` 当作全部文档世界，身份是**文件名**
（``draft:<title>_v003``），于是：

* 版本族由 LLM 自由填的 title 决定 —— 换个措辞就分叉成两条历史，两个无关实验
  撞名就误合并成一条；
* 拿着一个 ``Au111_report_v003.md`` 反查不出它属于哪个实验、哪次对话；
* PUT 走 ``next_version_path``（glob-then-write，无锁无 ``O_EXCL``），两个并发
  写入者会算出同一个 ``_v003``，**后写者静默覆盖前者** —— 正是「永不覆盖」承诺
  的破口（陷阱 ①）；
* diff 只能比相邻两版，「v1 到 v7 一共改了什么」问不出来。

现在身份是 ``doc_id``（ULID），落点是实验文件夹里的一个目录，版本分配只经
``store``（per-doc 锁 + ``open('x')`` claim + ``.part`` + ``os.replace``）。
**本模块不得再出现 ``next_version_path``。**

三条房规
--------

1. **永不 500。** 客户端错误回结构化 4xx（404 找不到 / 409 版本冲突 / 400 空正文），
   我们自己坏了回 ``degraded=True`` + ``detail``，绝不把 traceback 甩给前端。
2. **全部 handler 是 ``def`` 不是 ``async def``。** 里面是目录扫描、读文件、渲染
   HTML —— 同步 I/O。写成 ``def`` FastAPI 会丢线程池，事件循环不被阻塞；写成
   ``async def`` 再做同步 I/O 就是把整个服务卡住（house rule：UI 绝不冻结）。
3. **没有终态。** 不存在 finalize / archive / 关闭文档这种端点。文档永远可以
   再来一版（INCREMENTAL-ONLY）。

路由注册顺序
------------

``/documents/{doc_id}/versions`` 等子资源**必须**排在 ``/documents/{doc_id}``
之前，且 ``{doc_id}`` **不能**用 ``:path`` 转换器 —— ``:path`` 的 ``.*`` 会把
``abc/versions`` 整个吞成 doc_id（同款事故：``/agents/group-transcript`` 曾被
``/agents/{agent_id}`` 遮蔽）。
"""

from __future__ import annotations

import difflib
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

from fastapi import APIRouter, Query, Request, Response
from fastapi.responses import PlainTextResponse

from mast.api.schemas_documents import (
    DocumentClaimRequest,
    DocumentDetail,
    DocumentDiffResponse,
    DocumentExportResponse,
    DocumentPatchRequest,
    DocumentSaveResponse,
    DocumentsResponse,
    DocumentSummary,
    DocumentVersionsResponse,
    DocumentWriteRequest,
    KindOption,
    VersionInfo,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["documents"])

_VERDICT_RE = re.compile(r"<!--\s*verdict:\s*(\w+)\s*-->", re.IGNORECASE)
#: 一份手稿是文本，但仍要挡住病态大文件把整个响应撑爆。
_MAX_BODY_CHARS = 400_000
#: verdict 只可能在首行附近 —— 读个头就够，不必把整篇拉进内存。
_VERDICT_HEAD_BYTES = 2_048
_MAX_LIMIT = 1_000


# ── 接线 ──────────────────────────────────────────────────────────────

def _storage(request: Request):
    """live 的 ``ExperimentStorage``；拿不到就按规范路径自己开一个。

    回落**不是猜**：``mast.documents.paths.storage(None)`` 用的是
    ``data_paths.experiment_db_path()`` —— 与 store 写索引、与 runtime 用的是同一个
    真源（``MAST_EXPERIMENT_DB`` env > ``<project_root>/experiments/mast_experiments.db``）。

    以前这里 ctx 拿不到就直接返回 ``None``，于是半接线的进程（纯 API 测试、
    离线脚本）里 ``experiment_name`` 一列**恒为空字符串** —— 而「这份文档属于哪个
    实验」正是关联文档那一栏唯一的信息量。
    """
    try:
        st = request.app.state.ctx.experiment_storage
        if st is not None:
            return st
    except Exception:  # noqa: BLE001 — ctx 未接线
        pass
    try:
        from mast.documents.paths import storage as _doc_storage
        return _doc_storage(None)
    except Exception as exc:  # noqa: BLE001 — 库文件不可达：少一列，不报错
        logger.debug("documents: fallback storage unavailable: %r", exc)
        return None


def _docstore(request: Request):
    """DocumentStore，尽量绑在 live 的 ExperimentStorage 上。

    绑 live 实例只是省掉每次写 DB 时重建连接；``None`` 时 store 自己按
    ``MAST_EXPERIMENT_DB`` 新建一个，读路径完全不依赖 DB（权威是文件夹）。
    """
    from mast.documents import DocumentStore

    return DocumentStore(_storage(request))


# ── 投影 ──────────────────────────────────────────────────────────────

def _kind_options() -> list[KindOption]:
    from mast.documents import KIND_LABELS, KINDS

    return [KindOption(value=k, label=KIND_LABELS.get(k, k)) for k in KINDS]


def _epoch(iso: str) -> float | None:
    """ISO8601 → epoch 秒。**只给旧字段 ``modified_at`` 用。**

    新代码请直接用 ISO 字符串：epoch 秒会丢时区语义，而本项目已经因为「把本地
    时间当 UTC」漏认过一整批历史文件。
    """
    s = (iso or "").strip()
    if not s:
        return None
    try:
        return datetime.fromisoformat(s).timestamp()
    except ValueError:
        return None


def _read_verdict(path: Path | None) -> str | None:
    """从正文头部取 ``<!-- verdict: X -->``。

    这一行是 ``save_review`` 写的第一行，前端的 ACCEPT/REVISE/REJECT 徽章解析它。
    """
    if path is None:
        return None
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            head = fh.read(_VERDICT_HEAD_BYTES)
    except OSError:
        return None
    m = _VERDICT_RE.search(head)
    return m.group(1).upper() if m else None


def _version_infos(entry) -> list[VersionInfo]:
    return [
        VersionInfo(
            version=vm.v, words=vm.words, sha256=vm.sha256,
            created_at=vm.created_at, created_by=vm.created_by, note=vm.note,
            conversation_id=vm.conversation_id, run_id=vm.run_id,
        )
        for vm in entry.versions
    ]


def _summary(entry, *, relation: str = "", exp_names: dict[str, str] | None = None,
             ) -> DocumentSummary:
    from mast.documents import KIND_LABELS

    meta = entry.meta
    latest = entry.versions[-1] if entry.versions else None
    path = entry.version_path()
    eid = meta.experiment_id or ""
    return DocumentSummary(
        doc_id=meta.doc_id,
        kind=meta.kind,
        kind_label=KIND_LABELS.get(meta.kind, meta.kind),
        title=meta.title,
        version=entry.latest_version,
        versions_count=len(entry.versions),
        experiment_id=meta.experiment_id,
        experiment_name=(exp_names or {}).get(eid, ""),
        sample_id=meta.sample_id,
        relation=relation,
        root_kind=meta.root_kind,
        verdict=_read_verdict(path) if meta.kind == "review" else None,
        words=latest.words if latest else 0,
        created_at=meta.created_at,
        updated_at=meta.updated_at,
        created_by=meta.created_by,
        conversation_id=meta.conversation_id,
        run_id=meta.run_id,
        target_doc_id=meta.target_doc_id,
        target_version=meta.target_version,
        path=str(path) if path else str(entry.dir),
        dir_path=str(entry.dir),
        modified_at=_epoch(meta.updated_at or meta.created_at),
    )


def _experiment_names(storage, entries) -> dict[str, str]:
    """``experiment_id → name``，**按去重后的 id 取**（不是每行一次查询）。

    上限是机器上的实验数，与文档数无关 —— 一个实验有 20 份文档也只查一次。
    """
    if storage is None:
        return {}
    ids: list[str] = []
    for e in entries:
        eid = e.meta.experiment_id
        if eid and eid not in ids:
            ids.append(eid)
    out: dict[str, str] = {}
    for eid in ids:
        try:
            row = storage.get_experiment(eid)
        except Exception as exc:  # noqa: BLE001 — 名字缺失只是少一列
            logger.debug("experiment name lookup failed (%s): %r", eid, exc)
            continue
        if row:
            out[eid] = str(row.get("name") or "")
    return out


def _documents_root(storage) -> str:
    """当前作用域下新文档会落到哪里 —— 空列表时给用户的一句交代。"""
    try:
        from mast.documents import paths as dpaths

        eid, _sample = dpaths.current_scope(storage)
        home, _root_kind = dpaths.doc_home("experiment_report", eid, create=False,
                                           st=storage)
        return str(home)
    except Exception as exc:  # noqa: BLE001
        logger.debug("documents root resolve failed: %r", exc)
        return ""


# ── 解析 ──────────────────────────────────────────────────────────────

def _resolve(s, doc_id: str):
    """doc_id → DocEntry。

    先精确匹配 doc_id，再走 ``resolve_ref``（它额外接受 ``current`` / ``latest``、
    旧的 ``<kind>:<stem>``、裸 stem、标题近似）。**精确匹配永远优先** —— 标题匹配
    是便利，不是身份。
    """
    did = (doc_id or "").strip()
    if not did:
        return None
    entry = s.get(did)
    if entry is not None:
        return entry
    return s.resolve_ref(did)


def _not_found(doc_id: str) -> str:
    return (f"找不到文档 {doc_id!r}。它可能已被搬移或从未存在 —— "
            f"用 GET /api/documents 列一遍现有文档。")


# ══════════════════════════════════════════════════════════════════════
# GET /api/documents
# ══════════════════════════════════════════════════════════════════════

@router.get("/documents", response_model=DocumentsResponse)
def list_documents(
    request: Request,
    experiment_id: str = Query("", description="按实验过滤（主归属 + 关联）；空 = 全局"),
    kind: str = Query("", description="按 kind 过滤；空 = 全部"),
    include_unfiled: bool = Query(True, description="含无主文档（_unfiled，待认领）"),
    include_related: bool = Query(True, description="含「关联到该实验」的文档，不只主归属"),
    include_discarded: bool = Query(
        False, description="含已废弃文档（_discarded；未删除，可 restore 恢复）"),
    limit: int = Query(200, ge=1, le=_MAX_LIMIT),
) -> DocumentsResponse:
    """列文档，最近更新在前。

    **空列表是正常状态**（还没有任何文档），不是 degraded。每行带 ``relation``：
    传了 ``experiment_id`` 时 ``primary`` = 主归属该实验、``related`` = 声明关联；
    没传时 ``primary`` / ``unfiled`` 表示它自己有没有归属。
    """
    storage = _storage(request)
    try:
        s = _docstore(request)
        entries = s.list(experiment_id=experiment_id or None, kind=kind or None,
                         include_unfiled=include_unfiled,
                         include_related=include_related,
                         include_discarded=include_discarded)
    except Exception as exc:  # noqa: BLE001 — degrade，永不 500
        logger.warning("documents list failed: %r", exc)
        return DocumentsResponse(degraded=True, detail=str(exc),
                                 kinds=_kind_options())

    entries = entries[:limit]
    try:
        names = _experiment_names(storage, entries)
        rows = [_summary(e, relation=s.relation_of(e, experiment_id or None),
                         exp_names=names)
                for e in entries]
    except Exception as exc:  # noqa: BLE001
        logger.warning("documents projection failed: %r", exc)
        return DocumentsResponse(degraded=True, detail=str(exc),
                                 kinds=_kind_options())

    root = _documents_root(storage)
    return DocumentsResponse(
        documents=rows, total=len(rows), count=len(rows),
        kinds=_kind_options(), documents_root=root,
        drafts_dir=root, reviews_dir=root, degraded=False,
    )


# ══════════════════════════════════════════════════════════════════════
# 子资源（字面后缀）—— 必须先注册，见模块 docstring
# ══════════════════════════════════════════════════════════════════════

@router.get("/documents/{doc_id}/versions", response_model=DocumentVersionsResponse,
            responses={404: {"model": DocumentVersionsResponse}})
def get_document_versions(doc_id: str, request: Request,
                          response: Response) -> DocumentVersionsResponse:
    """一个文档的全部版本，v 升序。

    版本集合 = **目录扫描 ∪ versions.jsonl**：jsonl 有就用它的元数据，只在目录里
    出现的（claim 成功但登记前崩了）按内容非空纳入。所以崩溃过的文档在这里也是
    完整的（设计陷阱 ⑬）。
    """
    try:
        s = _docstore(request)
        entry = _resolve(s, doc_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("versions read failed (%s): %r", doc_id, exc)
        return DocumentVersionsResponse(doc_id=doc_id, degraded=True, detail=str(exc))
    if entry is None:
        response.status_code = 404
        return DocumentVersionsResponse(doc_id=doc_id, detail=_not_found(doc_id))
    infos = _version_infos(entry)
    return DocumentVersionsResponse(
        doc_id=entry.doc_id, versions=infos, latest_version=entry.latest_version,
        total=len(infos),
    )


@router.get("/documents/{doc_id}/diff", response_model=DocumentDiffResponse,
            responses={404: {"model": DocumentDiffResponse}})
def get_document_diff(
    doc_id: str,
    request: Request,
    response: Response,
    from_v: int = Query(0, ge=0, description="起始版本；0/省略 = 最新的上一版"),
    to_v: int = Query(0, ge=0, description="目标版本；0/省略 = 最新版"),
) -> DocumentDiffResponse:
    """**任意两版**的 unified diff。

    旧实现只能比相邻两版。这里 ``from_v`` / ``to_v`` 任取，所以「v1 到 v7 一共
    改了什么」终于能问 —— 那是用户真正想问的问题。

    单版本文档回 ``changed=False`` + 空 diff，**不会**把整篇报成新增（旧
    artifacts_edit 因为拿空串当 original，每次 diff 都撒这个谎）。
    """
    try:
        s = _docstore(request)
        entry = _resolve(s, doc_id)
        if entry is None:
            response.status_code = 404
            return DocumentDiffResponse(doc_id=doc_id, detail=_not_found(doc_id))

        avail = [vm.v for vm in entry.versions]
        if not avail:
            return DocumentDiffResponse(doc_id=entry.doc_id, versions_available=[],
                                        detail="该文档还没有任何版本。")
        to_version = int(to_v) or avail[-1]
        if to_version not in avail:
            response.status_code = 404
            return DocumentDiffResponse(
                doc_id=entry.doc_id, versions_available=avail,
                detail=f"没有版本 v{to_version}（现有：{avail}）")
        if from_v:
            from_version = int(from_v)
            if from_version not in avail:
                response.status_code = 404
                return DocumentDiffResponse(
                    doc_id=entry.doc_id, to_version=to_version,
                    versions_available=avail,
                    detail=f"没有版本 v{from_version}（现有：{avail}）")
        else:
            earlier = [v for v in avail if v < to_version]
            if not earlier:
                return DocumentDiffResponse(
                    doc_id=entry.doc_id, from_version=0, to_version=to_version,
                    to_words=_words_of(entry, to_version), changed=False,
                    versions_available=avail,
                    detail="只有一个版本，没有可对比的先前版本。")
            from_version = earlier[-1]

        a = entry.read_text(from_version) or ""
        b = entry.read_text(to_version) or ""
        changed = a != b
        diff_text = ""
        if changed:
            diff_text = "".join(difflib.unified_diff(
                a.splitlines(keepends=True), b.splitlines(keepends=True),
                fromfile=f"v{from_version:03d}.md", tofile=f"v{to_version:03d}.md"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("diff failed (%s): %r", doc_id, exc)
        return DocumentDiffResponse(doc_id=doc_id, degraded=True, detail=str(exc))

    return DocumentDiffResponse(
        doc_id=entry.doc_id, from_version=from_version, to_version=to_version,
        from_words=_words_of(entry, from_version), to_words=_words_of(entry, to_version),
        diff=diff_text, changed=changed, versions_available=avail,
    )


def _words_of(entry, v: int) -> int:
    vm = next((x for x in entry.versions if x.v == v), None)
    return vm.words if vm else 0


@router.post("/documents/{doc_id}/claim", response_model=DocumentSaveResponse,
             responses={404: {"model": DocumentSaveResponse},
                        400: {"model": DocumentSaveResponse}})
def claim_document(doc_id: str, body: DocumentClaimRequest, request: Request,
                   response: Response) -> DocumentSaveResponse:
    """把文档认领到一个实验 —— **物理搬家**（主归属决定落点）。

    搬完还会把正文引用到的 ``_assets/`` 图复制进目标实验的图池：markdown 里写的是
    ``../_assets/x.png``，跨实验搬家后指向的是新实验的图池，图不跟过去就是一堆
    碎图标（设计陷阱 ⑫）。
    """
    try:
        s = _docstore(request)
        entry = _resolve(s, doc_id)
        if entry is None:
            response.status_code = 404
            return DocumentSaveResponse(doc_id=doc_id, detail=_not_found(doc_id))
        res = s.claim(entry.doc_id, body.experiment_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("claim failed (%s): %r", doc_id, exc)
        return DocumentSaveResponse(doc_id=doc_id, degraded=True, detail=str(exc))
    if not res.ok:
        response.status_code = 400
        return DocumentSaveResponse(doc_id=entry.doc_id, detail=res.error)
    return _save_response(s, res,
                          detail=f"已认领到实验 {res.experiment_id}，文档目录已搬到 {res.path}")


@router.post("/documents/{doc_id}/discard", response_model=DocumentSaveResponse,
             responses={404: {"model": DocumentSaveResponse},
                        400: {"model": DocumentSaveResponse}})
def discard_document(doc_id: str, request: Request, response: Response,
                     reason: str = Query("", description="为什么废弃（留痕用）"),
                     ) -> DocumentSaveResponse:
    """废弃一份文档 —— **搬进 ``_discarded`` 区，一个字节都不删。**

    没有「删除」这个动作，理由和 ``_quarantine`` 一样：绝不丢字节，宁可事后认领。
    一份看着是垃圾的报告，可能是用户唯一还留着的那一版。

    为什么需要它：``save()`` 在 doc_id 找不到时刻意另立新文档（丢内容不可接受），
    所以增殖是被允许的失败模式 —— 而允许增殖的前提是事后能清理。用
    ``POST .../restore`` 撤销。
    """
    try:
        s = _docstore(request)
        entry = _resolve(s, doc_id)
        if entry is None:
            response.status_code = 404
            return DocumentSaveResponse(doc_id=doc_id, detail=_not_found(doc_id))
        res = s.discard(entry.doc_id, reason=reason)
    except Exception as exc:  # noqa: BLE001
        logger.warning("discard failed (%s): %r", doc_id, exc)
        return DocumentSaveResponse(doc_id=doc_id, degraded=True, detail=str(exc))
    if not res.ok:
        response.status_code = 400
        return DocumentSaveResponse(doc_id=entry.doc_id, detail=res.error)
    return _save_response(s, res, detail=(
        f"已废弃（未删除，字节仍在 {res.path}）。要撤销请调 "
        f"POST /api/documents/{res.doc_id}/restore。"))


@router.post("/documents/{doc_id}/restore", response_model=DocumentSaveResponse,
             responses={404: {"model": DocumentSaveResponse},
                        400: {"model": DocumentSaveResponse}})
def restore_document(doc_id: str, request: Request, response: Response,
                     experiment_id: str = Query(
                         "", description="直接归到这个实验；留空则回未归属区待认领"),
                     ) -> DocumentSaveResponse:
    """把废弃的文档搬回来。

    给了 ``experiment_id`` 就直接归到那个实验；没给就回 ``_unfiled`` 待认领 ——
    **不猜**它原来属于谁（猜错的归属比没有归属更糟）。
    """
    try:
        s = _docstore(request)
        entry = _resolve(s, doc_id)
        if entry is None:
            response.status_code = 404
            return DocumentSaveResponse(doc_id=doc_id, detail=_not_found(doc_id))
        res = s.restore(entry.doc_id, experiment_id=experiment_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("restore failed (%s): %r", doc_id, exc)
        return DocumentSaveResponse(doc_id=doc_id, degraded=True, detail=str(exc))
    if not res.ok:
        response.status_code = 400
        return DocumentSaveResponse(doc_id=entry.doc_id, detail=res.error)
    return _save_response(s, res, detail=f"已恢复到 {res.path}")


@router.get("/documents/{doc_id}/export", response_model=None,
            responses={404: {"model": DocumentExportResponse},
                       400: {"model": DocumentExportResponse},
                       200: {"description": "文件内容（markdown / HTML / docx），"
                                            "带 Content-Disposition 附件头"}})
def export_document(
    doc_id: str,
    request: Request,
    response: Response,
    format: str = Query("md", description="md | html | docx"),
    version: int = Query(0, ge=0, description="0 = 最新版"),
):
    """导出一个版本：``md`` 正文原文 / ``html`` 自包含单文件 / ``docx`` Word 文档。

    三种格式各有各的用途，**不是重复功能**：

    * ``md`` —— 工作稿。可在编辑器里改、有版本、能比 diff。
    * ``html`` —— **给人看**的交付件。图是 base64 内嵌的，所以离开实验文件夹也
      永不断图（md 里的 ``../_assets/x.png`` 一走就全断）。
    * ``docx`` —— **给人改**的交付件。这是 HTML 覆盖不到的那一半：导师/合作者要用
      Word 的**修订和批注**，或者期刊要求 ``.docx`` 投稿。全部挂 Word 内置样式
      （``Heading 1``/``Normal``/``Caption``/``Table Grid`` …），所以对方改一次样式
      就能重排全文、套期刊模板只是一次样式替换。

    ``html`` 与 ``docx`` 都**留档**到 ``<exp>/exports/``（带时间戳，多份共存）。旧实现是
    ``data/reports/<stem>.html`` 直接覆盖，上一份交付件就这么没了。
    """
    fmt = (format or "md").strip().lower()
    if fmt in ("markdown", "text", "txt"):
        fmt = "md"
    try:
        s = _docstore(request)
        entry = _resolve(s, doc_id)
        if entry is None:
            response.status_code = 404
            return DocumentExportResponse(doc_id=doc_id, format=fmt,
                                          detail=_not_found(doc_id))
        v = int(version) or entry.latest_version
        src = entry.version_path(v)
        text = entry.read_text(v)
        if src is None or text is None:
            response.status_code = 404
            return DocumentExportResponse(
                doc_id=entry.doc_id, format=fmt, version=v,
                detail=f"版本 v{v} 的文件读不到（现有：{[x.v for x in entry.versions]}）")
    except Exception as exc:  # noqa: BLE001
        logger.warning("export resolve failed (%s): %r", doc_id, exc)
        return DocumentExportResponse(doc_id=doc_id, format=fmt, degraded=True,
                                      detail=str(exc))

    base = _export_stem(entry, v)
    if fmt == "md":
        return PlainTextResponse(
            content=text, media_type="text/markdown; charset=utf-8",
            headers=_attachment(f"{base}.md"),
        )
    if fmt in ("word", "doc"):
        fmt = "docx"
    if fmt == "docx":
        return _export_docx(s, entry, v, src, text, base, response)
    if fmt != "html":
        response.status_code = 400
        return DocumentExportResponse(
            doc_id=entry.doc_id, format=fmt, version=v,
            detail=f"不支持的格式 {fmt!r}（可用：md | html | docx）")

    try:
        from mast.agents._shared.report_html import render_html

        # base_dir **必须**是版本文件所在目录 —— 相对链接 ../_assets/x.png 就是
        # 按它解析的。传实验目录或 cwd 会让每张图都 missing（静默交付无图报告）。
        html, inlined, missing = render_html(
            text, base_dir=src.parent,
            title=entry.meta.title or "实验文档",
            subtitle=f"v{v:03d} · {entry.meta.updated_at or entry.meta.created_at}")
    except Exception as exc:  # noqa: BLE001
        logger.warning("html render failed (%s): %r", doc_id, exc)
        return DocumentExportResponse(doc_id=entry.doc_id, format="html", version=v,
                                      degraded=True, detail=f"HTML 渲染失败：{exc}")

    archived = ""
    try:
        out = s.export_path(entry, ".html")
        if v != entry.latest_version:
            out = out.with_name(out.name.replace(
                f"_v{entry.latest_version:03d}_", f"_v{v:03d}_"))
        out.write_text(html, encoding="utf-8")
        archived = str(out)
    except Exception as exc:  # noqa: BLE001 — 留档失败不该让下载失败
        logger.warning("export archive write failed (%s): %r", doc_id, exc)

    headers = _attachment(f"{base}.html")
    headers["X-Export-Path"] = quote(archived)
    headers["X-Images-Inlined"] = str(inlined)
    headers["X-Images-Missing"] = str(missing)
    return Response(content=html, media_type="text/html; charset=utf-8",
                    headers=headers)


_DOCX_MIME = ("application/vnd.openxmlformats-officedocument"
              ".wordprocessingml.document")


def _export_docx(s, entry, v: int, src, text: str, base: str, response: Response):
    """渲染并回传 ``.docx``，同时留档到 ``<exp>/exports/``。

    与 HTML 分支逐条对齐：``base_dir`` 必须是**版本文件所在目录**（相对链接
    ``../_assets/x.png`` 就是按它解析的），缺图数经 ``X-Images-Missing`` 如实回报，
    留档失败不让下载失败。
    """
    try:
        from mast.agents._shared.report_docx import render_docx

        blob, embedded, missing = render_docx(
            text, base_dir=src.parent,
            title=entry.meta.title or "实验文档",
            subtitle=f"v{v:03d} · {entry.meta.updated_at or entry.meta.created_at}")
    except ImportError as exc:
        # python-docx 缺失（未装 / 打包漏了数据文件）—— 如实说，别让它长得像
        # 「渲染失败」。装它的命令直接给出来。
        logger.warning("docx render unavailable: %r", exc)
        response.status_code = 400
        return DocumentExportResponse(
            doc_id=entry.doc_id, format="docx", version=v, degraded=True,
            detail=("Word 导出不可用：python-docx 未安装。"
                    "装它：pip install python-docx（已在 requirements-v2.txt 里）。"
                    "md 与 html 两种格式不受影响。"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("docx render failed (%s): %r", entry.doc_id, exc)
        response.status_code = 400
        return DocumentExportResponse(doc_id=entry.doc_id, format="docx", version=v,
                                      degraded=True, detail=f"Word 渲染失败：{exc}")

    archived = ""
    try:
        out = s.export_path(entry, ".docx")
        if v != entry.latest_version:
            out = out.with_name(out.name.replace(
                f"_v{entry.latest_version:03d}_", f"_v{v:03d}_"))
        out.write_bytes(blob)
        archived = str(out)
    except Exception as exc:  # noqa: BLE001 — 留档失败不该让下载失败
        logger.warning("docx archive write failed (%s): %r", entry.doc_id, exc)

    headers = _attachment(f"{base}.docx")
    headers["X-Export-Path"] = quote(archived)
    headers["X-Images-Inlined"] = str(embedded)
    headers["X-Images-Missing"] = str(missing)
    return Response(content=blob, media_type=_DOCX_MIME, headers=headers)


def _export_stem(entry, v: int) -> str:
    """下载文件名的主干：``<标题>_vNNN``。

    旧 ArtifactEditor 硬编码 ``${id}.txt``，下载下来是一堆 ``draft_xxx.txt``，
    过一周谁也认不出哪份是哪份。
    """
    from mast.core.experiment_paths import slug

    return f"{slug(entry.meta.title or '文档', 'document', max_chars=48)}_v{v:03d}"


def _attachment(filename: str) -> dict[str, str]:
    """``Content-Disposition``，中文文件名走 RFC 5987。

    Starlette 的 header 值必须 latin-1 可编码，所以 ``filename=`` 只能放 ASCII
    兜底名；真正的中文名放 ``filename*=UTF-8''…``（现代浏览器都用它）。
    """
    ascii_name = re.sub(r"[^A-Za-z0-9._-]+", "_", filename).strip("_") or "document"
    return {"Content-Disposition":
            f"attachment; filename=\"{ascii_name}\"; "
            f"filename*=UTF-8''{quote(filename)}"}


# ══════════════════════════════════════════════════════════════════════
# 单文档：详情 / 编辑 / 头部
# ══════════════════════════════════════════════════════════════════════

@router.get("/documents/{doc_id}", response_model=DocumentDetail,
            responses={404: {"model": DocumentDetail}})
def get_document(doc_id: str, request: Request, response: Response,
                 version: int = Query(0, ge=0, description="0 = 最新版")) -> DocumentDetail:
    """一个文档的正文 + 全部版本元数据。

    ``doc_id`` 也接受 ``current`` / ``latest``（= 最近更新的那一份）与旧的
    ``<kind>:<stem>``，方便从对话里记下的名字直接打开。
    """
    try:
        s = _docstore(request)
        entry = _resolve(s, doc_id)
        if entry is None:
            response.status_code = 404
            return DocumentDetail(doc_id=doc_id, detail=_not_found(doc_id))
        storage = _storage(request)
        names = _experiment_names(storage, [entry])
        base = _summary(entry, relation=s.relation_of(entry, None), exp_names=names)

        v = int(version) or None
        text = entry.read_text(v)
        if text is None:
            detail = ("该文档还没有任何版本。"
                      if not entry.versions else
                      f"版本 v{version} 不存在（现有：{[x.v for x in entry.versions]}）")
            return DocumentDetail(**base.model_dump(), versions=_version_infos(entry),
                                  version_requested=v, detail=detail)
        truncated = len(text) > _MAX_BODY_CHARS
        if truncated:
            text = text[:_MAX_BODY_CHARS] + "\n\n…[截断]"
    except Exception as exc:  # noqa: BLE001
        logger.warning("document read failed (%s): %r", doc_id, exc)
        return DocumentDetail(doc_id=doc_id, degraded=True, detail=str(exc))

    return DocumentDetail(
        **base.model_dump(), content=text, body=text,
        version_requested=v, versions=_version_infos(entry), truncated=truncated,
    )


@router.put("/documents/{doc_id}", response_model=DocumentSaveResponse,
            responses={404: {"model": DocumentSaveResponse},
                       409: {"model": DocumentSaveResponse},
                       400: {"model": DocumentSaveResponse}})
def put_document(doc_id: str, body: DocumentWriteRequest, request: Request,
                 response: Response) -> DocumentSaveResponse:
    """用户编辑 —— **存为新版本**（``created_by="operator"``）。

    这是让用户成为参与者而不是旁观者的地方（）。旧的对象编辑器把
    修改 POST 进一个内存 dict，声称「orchestrator 下一个 super-step 会读」——
    从来没有任何 agent 读过它。写进文件就真的闭环了：``load_draft("current")``
    读的就是这一版。

    ``base_version`` 是乐观并发校验：填上你打开编辑框时看到的版本号，如果这期间
    智能体又存了一版，你会拿到 409 而不是用旧文本盖掉它（版本不会丢，但「基于
    过期内容的编辑覆盖了新内容」是内容损失）。
    """
    text = body.text().strip()
    try:
        s = _docstore(request)
        entry = _resolve(s, doc_id)
        if entry is None:
            response.status_code = 404
            return DocumentSaveResponse(doc_id=doc_id, detail=_not_found(doc_id))
        if not text:
            response.status_code = 400
            return DocumentSaveResponse(
                doc_id=entry.doc_id, latest_version=entry.latest_version,
                detail="拒绝保存空文档 —— 清空正文不是一次编辑。")
        if body.base_version is not None and int(body.base_version) != entry.latest_version:
            response.status_code = 409
            return DocumentSaveResponse(
                doc_id=entry.doc_id, conflict=True,
                latest_version=entry.latest_version,
                versions_count=len(entry.versions),
                detail=(f"该文档已有更新的版本（你基于 v{int(body.base_version):03d} 编辑，"
                        f"当前最新是 v{entry.latest_version:03d}）。"
                        f"请重新载入最新版本后再改 —— 没有任何版本被覆盖。"))
        res = s.save(text=text, doc_id=entry.doc_id, created_by="operator",
                     note=body.note or "用户编辑")
    except Exception as exc:  # noqa: BLE001
        logger.warning("document write failed (%s): %r", doc_id, exc)
        return DocumentSaveResponse(doc_id=doc_id, degraded=True, detail=str(exc))
    if not res.ok:
        response.status_code = 400
        return DocumentSaveResponse(doc_id=doc_id, detail=res.error)
    logger.info("operator edit saved → %s", res.path)
    return _save_response(
        s, res,
        detail=f"已保存为新版本 v{res.version:03d}（智能体下次读取该文档时会看到你的修改）")


@router.patch("/documents/{doc_id}", response_model=DocumentSaveResponse,
              responses={404: {"model": DocumentSaveResponse},
                         400: {"model": DocumentSaveResponse}})
def patch_document(doc_id: str, body: DocumentPatchRequest, request: Request,
                   response: Response) -> DocumentSaveResponse:
    """改可变头部：标题 / kind / 样品 / 关联实验。

    **不发新版本，不动目录名。** 目录名创建时冻结（``__<id8>`` 保证名字过时也能
    被 id 找回）；改标题只改 ``doc.json`` + DB，旧标题进 ``title_history`` 留痕。

    改主归属请用 ``POST .../claim``（那个要搬目录 + 复制图，不是改一个字段）。
    """
    try:
        s = _docstore(request)
        entry = _resolve(s, doc_id)
        if entry is None:
            response.status_code = 404
            return DocumentSaveResponse(doc_id=doc_id, detail=_not_found(doc_id))
        res = s.patch(
            entry.doc_id, title=body.title, kind=body.kind,
            sample_id=body.sample_id,
            related_experiment_ids=body.related_experiment_ids,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("document patch failed (%s): %r", doc_id, exc)
        return DocumentSaveResponse(doc_id=doc_id, degraded=True, detail=str(exc))
    if not res.ok:
        response.status_code = 400
        return DocumentSaveResponse(doc_id=doc_id, detail=res.error)
    return _save_response(s, res, detail="已更新")


def _save_response(s, res: Any, *, detail: str) -> DocumentSaveResponse:
    from mast.documents import KIND_LABELS

    entry = s.get(res.doc_id)
    return DocumentSaveResponse(
        ok=True, doc_id=res.doc_id, version=res.version,
        latest_version=entry.latest_version if entry else res.version,
        versions_count=len(entry.versions) if entry else 0,
        path=res.path, kind=res.kind, kind_label=KIND_LABELS.get(res.kind, res.kind),
        title=res.title, experiment_id=res.experiment_id, root_kind=res.root_kind,
        words=res.words, created_new=res.created_new,
        doc_id_unknown=res.doc_id_unknown, detail=detail, degraded=False,
    )
