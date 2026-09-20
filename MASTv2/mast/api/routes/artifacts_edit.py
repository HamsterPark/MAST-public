"""**兼容层** —— 旧对象编辑器的五个端点，内部已改走文档 store。

    新代码请用 ``/api/documents``（``routes/documents.py``）。

    这里的五个 URL（``POST/DELETE /api/artifacts/{id}/edit``、``GET .../history``、
    ``GET .../diff``、``GET .../export``）与响应形状**逐字保持不变**，只为让前端
    ``ArtifactEditor.tsx`` 在切到新端点之前继续工作。前端切换后**整个文件连同
    ``schemas_artifacts_edit.py`` 一起删**（设计 §5.3 P2）。

新端点的对应关系与升级点：

===============================  ==========================================
本模块（旧）                     ``/api/documents``（新）
===============================  ==========================================
``POST .../edit``                ``PUT /api/documents/{doc_id}``
                                 —— 多了 ``base_version`` 乐观并发校验
``DELETE .../edit``（revert）    ``PUT`` 上一版内容（语义相同，客户端两步）
``GET .../history``             ``GET .../versions`` —— 回**元数据**
                                 （sha256/词数/作者/note），不把每一版正文
                                 全塞进一个响应
``GET .../diff``                ``GET .../diff?from_v=&to_v=`` —— **任意两版**，
                                 不只相邻两版
``GET .../export``              ``GET .../export?format=md|html`` —— HTML 自包含
                                 （图 base64 内嵌），文件名带标题+版本
===============================  ==========================================

为什么内部必须换掉
------------------

旧实现把身份放在**文件名**上（``draft:<title>_v003``），版本分配走
``data_paths.next_version_path``（glob-then-write，无锁无 ``O_EXCL``）。两个并发
写入者会算出同一个 ``_v003``，**后写者静默覆盖前者** —— 这正是「永不覆盖」承诺
的破口（设计陷阱 ①）。现在版本分配只经 ``store``（per-doc 锁 + ``open('x')``
claim + ``.part`` + ``os.replace``），本模块**不得**再出现 ``next_version_path``。

id 解析：先当 ``doc_id`` 精确查，再走 ``resolve_ref``（它认旧的
``<kind>:<stem>``、裸 stem 与标题近似，靠 ``doc.json.legacy_stem`` 别名）。解析
不到就回原来那种 degraded 响应 —— **不 500，不猜**（猜一份别人的文档去覆盖是
比报错糟得多的失败）。

两条语义照旧
------------

* **保存 = 新版本，永不覆盖。** 智能体的版本和用户的修改都留在盘上。
* **回退 = 把上一版内容再存成一个新版本**，不删任何文件。undo 不能是全 app 里
  唯一会丢数据的按钮。
"""

from __future__ import annotations

import difflib
import logging
import re
from urllib.parse import quote

from fastapi import APIRouter, Query, Request
from fastapi.responses import PlainTextResponse

from mast.api.schemas_artifacts_edit import (
    ArtifactDiffResponse,
    ArtifactEditRequest,
    ArtifactEditResponse,
    ArtifactExportResponse,
    ArtifactHistoryEntry,
    ArtifactHistoryResponse,
    ArtifactRevertResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["artifacts_edit"])

_MAX_BODY_CHARS = 400_000


# ── 解析 ─────────────────────────────────────────────────────────────

def _docstore(request: Request):
    from mast.documents import DocumentStore

    try:
        storage = request.app.state.ctx.experiment_storage
    except Exception:  # noqa: BLE001
        storage = None
    return DocumentStore(storage)


def _not_editable(artifact_id: str) -> str | None:
    """非文件型产物的诚实拒绝，并指出它真正存在哪里。

    返回 None 表示这个 id 不是一个已知的产物**类别**（那它可能是个合法的单文件 id）。
    """
    try:
        from mast.agents._shared.artifacts import ARTIFACT_BY_ID

        art = ARTIFACT_BY_ID.get(artifact_id)
    except Exception:  # noqa: BLE001
        return None
    if art is None or art.editable:
        return None
    return (
        f"「{art.label}」不是可手工编辑的文本产物（存储于 {art.store}）。"
        "请通过对应的智能体或界面修改它。"
    )


def _resolve(s, artifact_id: str):
    """``artifact_id`` → DocEntry，或 None。

    刻意**不**接受 ``""`` / ``current`` / ``latest``：``resolve_ref`` 会把它们解析成
    「最近更新的那一份」，而这个兼容层的每个调用都是**写或读某一份具体文档**，
    落到别人的文档上是不可接受的。想要「当前那一份」请用新端点。
    """
    aid = (artifact_id or "").strip()
    if not aid or aid in ("current", "latest"):
        return None
    # 路径穿越护栏：id 永远是一个裸名字，不是路径。
    if any(ch in aid for ch in ("/", "\\", "..")):
        return None
    stem = aid.split(":", 1)[1] if ":" in aid else aid
    if not stem:
        return None
    entry = s.get(aid)
    if entry is not None:
        return entry
    return s.resolve_ref(aid)


_BAD_ID = "artifact_id 必须是 '<draft|review>:<文件名>' 或一个 doc_id（见 GET /api/artifacts）"


def _read(entry, v: int | None = None) -> str:
    return (entry.read_text(v) or "")[:_MAX_BODY_CHARS]


def _mtime(entry, v: int | None = None) -> float | None:
    p = entry.version_path(v)
    if p is None:
        return None
    try:
        return p.stat().st_mtime
    except OSError:
        return None


def _filename(entry, v: int) -> str:
    """``<标题>_vNNN.md`` —— 与旧的 ``<stem>_vNNN.md`` 形状一致。

    磁盘上的真名是 ``vNNN.md``（身份在目录名里），但那个名字下载下来认不出是谁，
    所以这里合成一个带标题的展示名。
    """
    from mast.core.experiment_paths import slug

    return f"{slug(entry.meta.title or '文档', 'document', max_chars=48)}_v{v:03d}.md"


# ── POST /api/artifacts/{artifact_id}/edit ───────────────────────────────────
@router.post("/artifacts/{artifact_id:path}/edit", response_model=ArtifactEditResponse)
def save_artifact_edit(
    artifact_id: str, body: ArtifactEditRequest, request: Request
) -> ArtifactEditResponse:
    """保存用户编辑 —— 存为文档的**新版本**。

    新端点：``PUT /api/documents/{doc_id}``（多一个 ``base_version`` 乐观并发校验，
    避免用基于过期内容的编辑盖掉智能体刚存的那一版）。
    """
    refusal = _not_editable(artifact_id)
    if refusal:
        return ArtifactEditResponse(
            artifact_id=artifact_id, detail=refusal, degraded=True)
    text = (body.body or "").strip()
    try:
        s = _docstore(request)
        entry = _resolve(s, artifact_id)
        if entry is None:
            return ArtifactEditResponse(
                artifact_id=artifact_id, detail=_BAD_ID, degraded=True)
        if not text:
            return ArtifactEditResponse(
                artifact_id=artifact_id, detail="拒绝保存空文档", degraded=True)
        res = s.save(text=text, doc_id=entry.doc_id, created_by="operator",
                     note="用户编辑（兼容端点 /api/artifacts/…/edit）")
        if not res.ok:
            return ArtifactEditResponse(
                artifact_id=artifact_id, detail=res.error, degraded=True)
        entry = s.get(res.doc_id) or entry
    except Exception as exc:  # noqa: BLE001 — degrade，永不 500
        logger.warning("artifacts_edit: save failed (%s): %r", artifact_id, exc)
        return ArtifactEditResponse(
            artifact_id=artifact_id, detail=str(exc), degraded=True)
    logger.info("artifacts_edit: operator saved %s → %s v%d (%d chars)",
                artifact_id, res.doc_id, res.version, len(text))
    return ArtifactEditResponse(
        ok=True, artifact_id=artifact_id, body=text, t=_mtime(entry),
        history_len=max(0, len(entry.versions) - 1),
        detail=f"已保存为新版本 v{res.version:03d}（智能体下次读取该文档时会看到你的修改）",
        degraded=False,
    )


# ── DELETE /api/artifacts/{artifact_id}/edit ─────────────────────────────────
@router.delete("/artifacts/{artifact_id:path}/edit",
               response_model=ArtifactRevertResponse)
def revert_artifact_edit(artifact_id: str, request: Request) -> ArtifactRevertResponse:
    """撤销上一次编辑 —— 把**上一版内容重新存成一个新版本**。

    不是删除。版本就是历史，销毁一版等于扔掉智能体（或用户）真做过的工作，而
    「回退」绝不能是全 app 里唯一会丢数据的按钮。重新提交旧正文让
    ``load_*("current")`` 重新看到它 —— 这才是这里 undo 的含义。
    """
    refusal = _not_editable(artifact_id)
    if refusal:
        return ArtifactRevertResponse(
            artifact_id=artifact_id, detail=refusal, degraded=True)
    try:
        s = _docstore(request)
        entry = _resolve(s, artifact_id)
        if entry is None:
            return ArtifactRevertResponse(
                artifact_id=artifact_id, detail=_BAD_ID, degraded=True)
        versions = entry.versions
        if len(versions) < 2:
            return ArtifactRevertResponse(
                artifact_id=artifact_id, reverted=False,
                history_len=max(0, len(versions) - 1),
                detail="只有一个版本，没有可回退的历史。", degraded=False)
        prev_v = versions[-2].v
        prev_text = _read(entry, prev_v)
        if not prev_text.strip():
            return ArtifactRevertResponse(
                artifact_id=artifact_id, reverted=False,
                history_len=max(0, len(versions) - 1),
                detail=f"上一版 v{prev_v:03d} 读不到内容，不做回退（不拿空内容盖掉现版本）。",
                degraded=True)
        res = s.save(text=prev_text, doc_id=entry.doc_id, created_by="operator",
                     note=f"回退到 v{prev_v:03d}（兼容端点）")
        if not res.ok:
            return ArtifactRevertResponse(
                artifact_id=artifact_id, detail=res.error, degraded=True)
        entry = s.get(res.doc_id) or entry
    except Exception as exc:  # noqa: BLE001
        logger.warning("artifacts_edit: revert failed (%s): %r", artifact_id, exc)
        return ArtifactRevertResponse(
            artifact_id=artifact_id, detail=str(exc), degraded=True)
    logger.info("artifacts_edit: reverted %s → v%d (from v%d)",
                artifact_id, res.version, prev_v)
    return ArtifactRevertResponse(
        ok=True, artifact_id=artifact_id, reverted=True,
        history_len=max(0, len(entry.versions) - 1),
        detail=f"已回退 v{prev_v:03d} 的内容，另存为 v{res.version:03d}（历史未被删除）",
        degraded=False,
    )


# ── GET /api/artifacts/{artifact_id}/history ─────────────────────────────────
@router.get("/artifacts/{artifact_id:path}/history",
            response_model=ArtifactHistoryResponse)
def get_artifact_history(artifact_id: str, request: Request) -> ArtifactHistoryResponse:
    """该文档的每一个版本（旧 → 新），来自磁盘。

    新端点：``GET /api/documents/{doc_id}/versions`` —— 回**元数据**
    （sha256 / 词数 / 作者 / note / 对话 id），而不是把每一版的完整正文都塞进
    一个响应（一份 30 页手稿 × 7 版就是几 MB 的 JSON）。
    """
    refusal = _not_editable(artifact_id)
    if refusal:
        return ArtifactHistoryResponse(
            artifact_id=artifact_id, detail=refusal, degraded=True)
    try:
        s = _docstore(request)
        entry = _resolve(s, artifact_id)
        if entry is None:
            return ArtifactHistoryResponse(artifact_id=artifact_id, detail=_BAD_ID,
                                           degraded=True)
        versions = entry.versions
        entries = [ArtifactHistoryEntry(body=_read(entry, vm.v), t=_mtime(entry, vm.v))
                   for vm in versions[:-1]]
        current = (ArtifactHistoryEntry(body=_read(entry, versions[-1].v),
                                        t=_mtime(entry, versions[-1].v))
                   if versions else None)
    except Exception as exc:  # noqa: BLE001
        logger.warning("artifacts_edit: history read failed (%s): %r", artifact_id, exc)
        return ArtifactHistoryResponse(artifact_id=artifact_id, detail=str(exc),
                                       degraded=True)
    return ArtifactHistoryResponse(
        artifact_id=artifact_id, entries=entries, count=len(entries),
        current=current, degraded=False,
    )


# ── GET /api/artifacts/{artifact_id}/diff ────────────────────────────────────
@router.get("/artifacts/{artifact_id:path}/diff", response_model=ArtifactDiffResponse)
def get_artifact_diff(artifact_id: str, request: Request) -> ArtifactDiffResponse:
    """最新两版的 unified diff。

    新端点：``GET /api/documents/{doc_id}/diff?from_v=&to_v=`` —— **任意两版**。
    「v1 到 v7 一共改了什么」在这个旧端点上问不出来。
    """
    refusal = _not_editable(artifact_id)
    if refusal:
        return ArtifactDiffResponse(
            artifact_id=artifact_id, detail=refusal, degraded=True)
    try:
        s = _docstore(request)
        entry = _resolve(s, artifact_id)
        if entry is None:
            return ArtifactDiffResponse(artifact_id=artifact_id, detail=_BAD_ID,
                                        degraded=True)
        versions = entry.versions
        if not versions:
            return ArtifactDiffResponse(
                artifact_id=artifact_id, has_original=False, has_edit=False,
                changed=False, diff="", detail="该文档尚无任何版本", degraded=False)
        current = _read(entry, versions[-1].v)
        has_original = len(versions) >= 2
        # 只有一版时没有可比的东西 —— 拿空串当 original 会把整篇报成新增，那正是
        # 旧实现每次 diff 都在撒的谎（它的 original 侧永远是空的，因为从没有人写过
        # ``task["artifacts"]["produced"]``）。
        base = _read(entry, versions[-2].v) if has_original else ""
        changed = has_original and base != current
        diff_text = ""
        if changed:
            diff_text = "".join(difflib.unified_diff(
                base.splitlines(keepends=True), current.splitlines(keepends=True),
                fromfile=_filename(entry, versions[-2].v),
                tofile=_filename(entry, versions[-1].v)))
    except Exception as exc:  # noqa: BLE001
        logger.warning("artifacts_edit: diff failed (%s): %r", artifact_id, exc)
        return ArtifactDiffResponse(
            artifact_id=artifact_id, detail=str(exc), degraded=True)
    return ArtifactDiffResponse(
        artifact_id=artifact_id, has_edit=len(versions) >= 2,
        has_original=has_original, changed=changed, diff=diff_text, degraded=False,
    )


# ── GET /api/artifacts/{artifact_id}/export ──────────────────────────────────
# response_model=None: 刻意异质 —— text/markdown 下载走 PlainTextResponse，
# format=json 与降级路径走 ArtifactExportResponse。
@router.get("/artifacts/{artifact_id:path}/export", response_model=None)
def export_artifact(
    artifact_id: str,
    request: Request,
    format: str = Query(default="text", description="text | json"),
):
    """导出当前正文（最新版本）。

    新端点：``GET /api/documents/{doc_id}/export?format=md|html`` —— 多一个自包含
    HTML（图 base64 内嵌，交付件永不断图），且下载文件名带标题+版本而不是
    ``<id>.txt``。
    """
    fmt = (format or "text").strip().lower()
    refusal = _not_editable(artifact_id)
    if refusal:
        return ArtifactExportResponse(
            artifact_id=artifact_id, filename=f"{artifact_id}.txt",
            detail=refusal, degraded=True)
    try:
        s = _docstore(request)
        entry = _resolve(s, artifact_id)
        if entry is None:
            return ArtifactExportResponse(
                artifact_id=artifact_id, filename=f"{artifact_id}.txt",
                detail=_BAD_ID, degraded=True)
        if not entry.versions:
            return ArtifactExportResponse(
                artifact_id=artifact_id, filename=f"{artifact_id}.txt",
                detail=f"该文档尚无任何版本：{entry.dir}", degraded=True)
        v = entry.versions[-1].v
        body = _read(entry, v)
        name = _filename(entry, v)
    except Exception as exc:  # noqa: BLE001
        logger.warning("artifacts_edit: export failed (%s): %r", artifact_id, exc)
        return ArtifactExportResponse(
            artifact_id=artifact_id, filename=f"{artifact_id}.txt",
            detail=str(exc), degraded=True)

    if fmt == "json":
        return ArtifactExportResponse(
            ok=True, artifact_id=artifact_id, body=body, filename=name,
            source="file", degraded=False)
    return PlainTextResponse(
        content=body, media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": _disposition(name)},
    )


def _disposition(name: str) -> str:
    """中文文件名走 RFC 5987 —— Starlette 的 header 值必须 latin-1 可编码。"""
    ascii_name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "document.md"
    return f"attachment; filename=\"{ascii_name}\"; filename*=UTF-8''{quote(name)}"


__all__ = ["router"]
