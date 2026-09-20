"""文档存储 —— 版本写入协议、读侧自愈、DB 索引。

设计文档：``docs/v2/design/document_and_library_management.md`` §3.2 / §3.3

INCREMENTAL-ONLY（硬不变式，与实验文件夹同一条）
------------------------------------------------

**任意时刻拔电源，磁盘上已有的内容必须自洽、可用、可继续。** 落盘只有两种形态：

* 原子追加 —— ``versions.jsonl`` 单行 JSON + ``\\n`` + flush
* 原子替换 —— ``.part`` + ``os.replace``（``doc.json``、进度视图）

**禁止**在本模块编写 ``on_experiment_end`` / ``on_archive`` / ``finalize_*``
形式的收尾函数。文档没有终态，永远可以再来一版。

为什么版本分配要 claim（陷阱 ①）
--------------------------------

旧实现 ``data_paths.next_version_path()`` 是 glob-then-write：算出下一个空号，
返回路径，调用方随后 ``write_text``。中间没有锁、没有 ``O_EXCL``、没有原子创建，
而并发写入者至少四个（save_draft / save_review / PUT documents / POST artifacts
edit）。两个同时保存 → 算出同一个 ``_v003`` → **后写者静默覆盖前者**。这正是
「永不覆盖」承诺的破口。

这里的顺序是：进程内 per-doc 锁 → 扫描取 ``max(v)+1`` → ``open(..., 'x')``
**claim**（``FileExistsError`` 就 +1 重试，这是跨进程背带）→ ``.part`` 写内容 →
``os.replace`` → jsonl 追加 → ``doc.json`` 替换 → DB best-effort。

为什么版本序不看 mtime（陷阱 ③）
--------------------------------

``data_paths.latest_versions`` 的 docstring 记着实测：**同机背靠背写盘 70.5%
落在同一 mtime**，并列时 stable sort 落回文件名序，导致 ``load_draft("current")``
返回用户编辑**之前**的版本，五次一现。这里版本序恒为 ``versions.jsonl`` 里的
整数 ``v``，mtime 只在恢复孤儿文件时用作创建时间的近似值。

读侧自愈规则（陷阱 ⑬）
----------------------

**版本集合 = 目录扫描 ∪ jsonl。** jsonl 有该版本就用它的元数据；只在目录里出现
的（claim 成功但 jsonl 追加前崩了）按内容非空纳入，空文件跳过（claim 之后立刻
崩，没有内容）。半行 jsonl 直接丢弃 —— 与 ``filestore.read_manifest`` 同款容错。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import threading
from dataclasses import dataclass, field
from pathlib import Path

from mast.documents import paths as dpaths
from mast.documents.model import (
    DOC_JSON,
    KIND_CODES,
    SLUG_MAX_DOC,
    VERSION_FILE_FMT,
    VERSIONS_JSONL,
    DocMeta,
    SaveResult,
    VersionMeta,
    doc_dir_name,
    doc_id8,
    normalize_kind,
    now_iso,
)
from mast.logging.v2.ulid import ulid_now

logger = logging.getLogger(__name__)

_VERSION_FILE_RE = re.compile(r"^v(\d{3,})$")

#: per-doc 写锁。跨进程靠 ``open(..., 'x')`` claim，这里挡的是同进程内的并发
#: （API 线程池 + agent 工具 + 定时任务同时写一个文档）。
_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()

#: ``doc_id → 文档目录`` 的进程内缓存。未命中就全量重扫（有界：每实验两个子目录）
#: 并重建。DB 是索引、文件夹是记录，所以缓存失效永远只是慢一点，不会答错。
_dir_cache: dict[str, Path] = {}
_cache_guard = threading.Lock()

_CJK_RE = re.compile(r"[㐀-䶿一-鿿豈-﫿]")


def _lock_for(doc_id: str) -> threading.Lock:
    with _locks_guard:
        lk = _locks.get(doc_id)
        if lk is None:
            lk = _locks[doc_id] = threading.Lock()
        return lk


def _publish_artifact_saved(**data) -> None:
    """广播「有一份产物落盘了」。**发布失败绝不影响保存。**

    为什么要有这个(2026-07-30):在这之前,产物落盘**不发任何事件** ——
    ``EventType`` 里没有 artifact 成员,``save()`` 也不 publish。于是「等资料到位
    再唤醒某个 agent」这件事在系统里没有触发源可用,只能靠轮询。

    三条边界,一条都不能松:

    * **best-effort**:文件已经在磁盘上了,那才是权威。事件只是通知。
    * **懒导入**:``mast.documents`` 不该为了发通知而依赖 ``mast.core``。
    * **单例可能不存在**:没有 runtime 的场合(测试、离线脚本)就静静跳过。

    EventBus 是**同步扇出、在发布线程上执行订阅回调、异常被吞** —— 也就是说
    订阅方在这条调用栈里跑。所以订阅回调必须**只入队、不做判断**,任何 LLM 询问
    都要挪到别的线程去(见 ``core/wake_scheduler.py``)。
    """
    try:
        from mast.core.events import EventBus

        EventBus.get().publish_artifact_saved(**data)
    except Exception as exc:  # noqa: BLE001 — 通知从不阻塞保存
        logger.debug("artifact_saved publish skipped: %s", exc)


def count_words(text: str) -> int:
    """词数。CJK 按字计，其余按空白切分 —— 中文报告用 ``len(text.split())``
    会算出接近行数的荒谬值（UI 里显示「12 词」的三千字报告）。"""
    cjk = len(_CJK_RE.findall(text))
    ascii_words = len([w for w in _CJK_RE.sub(" ", text).split() if w])
    return cjk + ascii_words


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_atomic(path: Path, text: str) -> None:
    """``.part`` + fsync + ``os.replace``。抛 OSError 由调用方处理。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    part = Path(str(path) + ".part")
    with part.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(part, path)


def _append_jsonl(path: Path, obj: dict) -> None:
    """单行 JSON + flush。崩溃最多丢最后一行，读侧丢弃不可解析的行。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
        fh.flush()


#: 文档目录里最长的那个文件名（进度视图的临时文件），用来算路径预算。
_LONGEST_LEAF = "progress.md.part"


def _path_budget_ok(home: Path, dir_name: str) -> bool:
    """检查最长叶子 progress.md.part 是否在 Windows 路径预算之内。

    复用 experiment_paths 的 260 字符口径。主动计算叶子路径，不只等待
    mkdir 抛错：系统允许创建的路径未必能被资源管理器等工具正常操作。
    """
    from mast.core.experiment_paths import _MAX_PATH
    try:
        return len(str(home / dir_name / _LONGEST_LEAF)) <= _MAX_PATH
    except Exception:  # noqa: BLE001 — 预算算不出来就不拦（交给 mkdir 判）
        return True


def write_doc_json(doc_dir: Path, meta: "DocMeta") -> bool:
    """写 ``doc.json``，**带 fsync**。永不抛，返回是否写成。

    刻意不用 ``manifest.write_json_atomic``：那个写入器是 ``tmp.write_text`` +
    ``os.replace``，**没有 fsync**。对 ``experiment.json`` 够用（丢了实验还在，
    reindex 能从别处补），但 ``doc.json`` 是文档存在性的**唯一**声明：它一旦变成
    0 字节，``load_entry`` 返回 None，文档从所有列表里消失、``get(doc_id)`` 找不到，
    **连 reindex 都救不回来** —— 而 ``vNNN.md`` 里的正文一个字节都没少。

    读侧的自愈规则（陷阱 ⑬）只覆盖 versions.jsonl 缺行，对这一条没有兜底，所以这里
    既要 fsync，:func:`load_entry` 也要能在 doc.json 读不出时重建一个最小 meta。
    """
    try:
        doc_dir.mkdir(parents=True, exist_ok=True)
        path = doc_dir / DOC_JSON
        part = Path(str(path) + ".part")
        with part.open("w", encoding="utf-8", newline="\n") as fh:
            json.dump(meta.to_json(), fh, ensure_ascii=False, indent=2, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(part, path)
        return True
    except OSError as exc:
        logger.warning("doc.json write failed (%s): %r", doc_dir, exc)
        return False


def read_versions(doc_dir: Path) -> list[VersionMeta]:
    """版本列表，按 ``v`` 升序。实现读侧自愈规则（见模块 docstring）。"""
    rows: dict[int, VersionMeta] = {}
    jsonl = doc_dir / VERSIONS_JSONL
    if jsonl.is_file():
        try:
            for line in jsonl.read_text(encoding="utf-8", errors="replace").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue  # 半行（崩在 flush 中间）—— 丢弃，不是错误
                if isinstance(obj, dict):
                    vm = VersionMeta.from_obj(obj)
                    if vm is not None:
                        rows[vm.v] = vm
        except OSError as exc:
            logger.warning("versions.jsonl read failed (%s): %r", doc_dir, exc)
    # 目录扫描：补 jsonl 里没有的版本（claim 成功但登记前崩）。
    try:
        for p in sorted(doc_dir.iterdir()):
            if p.suffix.lower() != ".md":
                continue
            m = _VERSION_FILE_RE.match(p.stem)
            if not m:
                continue
            v = int(m.group(1))
            if v in rows:
                continue
            try:
                if p.stat().st_size == 0:
                    continue  # claim 之后立刻崩：没有内容，不算一个版本
                text = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            rows[v] = VersionMeta(
                v=v, file=p.name, sha256=sha256_text(text), words=count_words(text),
                created_at=_mtime_iso(p), created_by="unknown",
                note="从目录扫描恢复（versions.jsonl 缺这一行）",
            )
    except OSError:
        pass
    return [rows[k] for k in sorted(rows)]


def _mtime_iso(p: Path) -> str:
    from datetime import datetime, timezone
    try:
        return datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).isoformat(timespec="seconds")
    except OSError:
        return ""


@dataclass
class DocEntry:
    """一个文档的完整视图：sidecar 头部 + 目录 + 版本表。"""

    meta: DocMeta
    dir: Path
    versions: list[VersionMeta] = field(default_factory=list)

    @property
    def doc_id(self) -> str:
        return self.meta.doc_id

    @property
    def latest_version(self) -> int:
        return self.versions[-1].v if self.versions else 0

    def version_path(self, v: int | None = None) -> Path | None:
        if not self.versions:
            return None
        if v is None:
            vm = self.versions[-1]
        else:
            vm = next((x for x in self.versions if x.v == int(v)), None)
            if vm is None:
                return None
        p = self.dir / (vm.file or VERSION_FILE_FMT.format(v=vm.v))
        return p if p.is_file() else None

    def read_text(self, v: int | None = None) -> str | None:
        p = self.version_path(v)
        if p is None:
            return None
        try:
            return p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None


def _root_kind_of(doc_dir: Path) -> str:
    """文档目录所在的区 —— **位置即状态**（没有 ``discarded`` 这种字段）。"""
    parts = doc_dir.parts
    if dpaths.DISCARDED_DIR in parts:
        return "discarded"
    if dpaths.UNFILED_DIR in parts:
        return "unfiled"
    return "experiment"


def load_entry(doc_dir: Path, root_kind: str = "experiment") -> DocEntry | None:
    """从一个文档目录读出 DocEntry。``doc.json`` 缺失/损坏返回 ``None``。"""
    from mast.logging.v2.manifest import read_json
    obj = read_json(doc_dir / DOC_JSON)
    meta = DocMeta.from_json(obj) if obj else None
    versions = read_versions(doc_dir)
    if meta is None:
        # ★ doc.json 读不出来（0 字节 / 半个 JSON —— 掉电后 replace 的典型残留），
        # 但只要还有一个版本文件，这份文档的**正文就完好无损**。直接返回 None 等于
        # 让它从所有列表里消失、连 reindex 都救不回来，而磁盘上明明还在 ——
        # 那就违背了「任意时刻拔电源，已有内容必须自洽、可用」。
        #
        # 目录名 ``<码>__<日期>__<slug>__<id8>`` 自带 kind 与 id8，但 id8 只有 8 位、
        # 不是完整 doc_id，所以**不能**拿它当身份。此处从 versions.jsonl 里取真正的
        # doc_id（每行都不带 doc_id，但文档目录唯一，所以用目录名做临时身份并明确
        # 标注 degraded），让用户至少能在界面上看到并救出内容。
        if not versions:
            return None
        meta = _recover_meta(doc_dir, versions)
        logger.warning("doc.json 不可读，已从目录名 + versions.jsonl 重建最小元数据：%s",
                       doc_dir)
    if not meta.dir_name:
        meta.dir_name = doc_dir.name
    meta.root_kind = root_kind or meta.root_kind
    if versions:
        meta.latest_version = versions[-1].v
    return DocEntry(meta=meta, dir=doc_dir, versions=versions)


def _recover_meta(doc_dir: Path, versions: list[VersionMeta]) -> DocMeta:
    """``doc.json`` 丢了时的最小元数据。**标题前缀写明它是恢复出来的。**

    身份用 ``recovered:<目录名>``：目录名里的 ``__<id8>`` 只有 8 位、且实测会撞
    （见 ``model.py`` 的说明），拿它冒充 doc_id 会让两份文档共享身份。用带前缀的
    目录名做临时 id 既唯一又一眼看得出是残局，用户救出内容后可以重新保存。
    """
    parts = doc_dir.name.split("__")
    code = parts[0] if parts else ""
    kind = next((k for k, c in KIND_CODES.items() if c == code), "experiment_report")
    title = "__".join(parts[2:-1]) if len(parts) >= 4 else doc_dir.name
    first, last = versions[0], versions[-1]
    return DocMeta(
        doc_id=f"recovered:{doc_dir.name}",
        kind=kind,
        title=f"[元数据损坏，正文完好] {title.replace('_', ' ')}",
        created_at=first.created_at,
        created_by=first.created_by or "unknown",
        updated_at=last.created_at,
        latest_version=last.v,
        dir_name=doc_dir.name,
    )


def _scan_all(*, include_discarded: bool = False,
              root: Path | None = None) -> list[DocEntry]:
    out: list[DocEntry] = []
    for home, root_kind in dpaths.search_roots(include_discarded=include_discarded,
                                               root=root):
        try:
            children = sorted(home.iterdir())
        except OSError:
            continue
        for child in children:
            # ``_`` 前缀的一律不是文档目录：``_assets``（图池）、``_discarded``
            # （实验内废弃区，由 search_roots 单独作为一个扫描根给出）。
            if not child.is_dir() or child.name.startswith("_"):
                continue
            entry = load_entry(child, root_kind)
            if entry is not None:
                out.append(entry)
    with _cache_guard:
        for e in out:
            _dir_cache[e.doc_id] = e.dir
    return out


class DocumentStore:
    """文档的读写入口。

    **文件夹是记录，DB 是可重建索引。** 所有读路径都从文件夹取事实（DB 只用来
    加速跨实验列举和供 records / 导出联表），所有写路径都先落文件夹、再 best-effort
    写 DB。DB 写失败只告警 —— 磁盘上已经是权威了。
    """

    def __init__(self, storage=None):
        self._st = storage

    # ── 定位 ────────────────────────────────────────────────────────

    def get(self, doc_id: str) -> DocEntry | None:
        """按 doc_id 取文档。缓存未命中就全量重扫并重建缓存。"""
        did = str(doc_id or "").strip()
        if not did:
            return None
        with _cache_guard:
            cached = _dir_cache.get(did)
        if cached is not None and (cached / DOC_JSON).is_file():
            entry = load_entry(cached, _root_kind_of(cached))
            if entry is not None and entry.doc_id == did:
                return entry
        # 单份取用时**连废弃区一起扫**：拿着 id 来找的人已经知道自己要什么
        # （UI 的「已废弃」列表、restore 操作），不该因为它被废弃而找不到。
        for entry in _scan_all(include_discarded=True):
            if entry.doc_id == did:
                return entry
        return None

    def list(self, *, experiment_id: str | None = None, kind: str | None = None,
             include_unfiled: bool = True, include_related: bool = True,
             include_discarded: bool = False) -> list[DocEntry]:
        """列文档，最近更新在前。

        ``experiment_id`` 给定时，主归属该实验的文档标 ``primary``，声明关联该
        实验的标 ``related``（``include_related=False`` 只要主归属）。

        废弃的文档**默认不出现**（``include_discarded=True`` 才列）—— 它们没被删，
        只是搬进了 ``_discarded`` 区，随时能 restore 回来。
        """
        eid = str(experiment_id or "").strip()
        want_kind = normalize_kind(kind) if kind else None
        out: list[DocEntry] = []
        for e in _scan_all(include_discarded=include_discarded):
            if e.meta.root_kind == "unfiled" and not include_unfiled:
                continue
            if e.meta.root_kind == "discarded" and not include_discarded:
                continue
            if want_kind and e.meta.kind != want_kind:
                continue
            if eid:
                primary = e.meta.experiment_id == eid
                related = include_related and eid in (e.meta.related_experiment_ids or [])
                if not (primary or related):
                    continue
            out.append(e)
        # 次键 doc_id 不是装饰：``updated_at`` 是 ``timespec="seconds"``，背靠背保存
        # 两份文档**必然**并列，而 stable sort 并列时回落到扫描顺序（目录名序），于是
        # ``latest()`` 返回**最老**的那一份 —— `load_draft("current")` 会把审稿人送到
        # 另一份手稿上，`export_report_html("current")` 导出错的交付件。
        #
        # 这正是陷阱③（mtime 并列，「五次一现」）在**文档级**的重演：版本级早就靠
        # ``versions.jsonl`` 的整数 v 修好了，文档级一直漏着。doc_id 是 ULID，字典序
        # 即创建序 —— 不碰时钟、不碰 mtime，也不需要更高精度的时间戳。
        out.sort(key=lambda e: (e.meta.updated_at or e.meta.created_at or "", e.doc_id),
                 reverse=True)
        return out

    def relation_of(self, entry: DocEntry, experiment_id: str | None) -> str:
        eid = str(experiment_id or "").strip()
        if not eid:
            return "primary" if entry.meta.experiment_id else "unfiled"
        if entry.meta.experiment_id == eid:
            return "primary"
        if eid in (entry.meta.related_experiment_ids or []):
            return "related"
        return "other"

    def latest(self, *, experiment_id: str | None = None,
               kinds: tuple[str, ...] | None = None) -> DocEntry | None:
        """「当前那一份」—— 指定实验（或全局）下最近更新的、kind 命中的文档。"""
        cands = self.list(experiment_id=experiment_id, include_unfiled=True)
        if kinds:
            want = {normalize_kind(k) for k in kinds}
            cands = [c for c in cands if c.meta.kind in want]
        cands = [c for c in cands if c.versions]
        return cands[0] if cands else None

    def resolve_ref(self, ref: str, *, experiment_id: str | None = None,
                    kinds: tuple[str, ...] | None = None) -> DocEntry | None:
        """把一个宽松的引用解析成文档。

        接受：doc_id、``"current"`` / ``"latest"`` / 空串（= 最近一份）、旧的
        ``<kind>:<stem>`` 或裸 ``<stem>``（legacy_stem 别名，供 artifacts_edit 与
        旧对话里记下的名字继续工作）、以及标题的近似匹配。

        **顺序是刻意的**：doc_id 精确匹配永远优先，标题匹配放最后 —— 标题匹配是
        便利，不是身份（陷阱 ②）。
        """
        r = str(ref or "").strip()
        if r in ("", "current", "latest", "current_draft"):
            return self.latest(experiment_id=experiment_id, kinds=kinds)
        entry = self.get(r)
        if entry is not None:
            return entry
        stem = r.split(":", 1)[1] if ":" in r else r
        all_docs = _scan_all()
        for e in all_docs:
            if e.meta.legacy_stem and e.meta.legacy_stem == stem:
                return e
        # 旧命名 <stem>_vNNN → 去掉版本后缀再比一次
        bare = re.sub(r"_v\d+$", "", stem)
        for e in all_docs:
            if e.meta.legacy_stem and re.sub(r"_v\d+$", "", e.meta.legacy_stem) == bare:
                return e
        low = bare.replace("_", "").casefold()
        if low:
            for e in all_docs:
                if e.meta.title.replace(" ", "").replace("_", "").casefold() == low:
                    return e
        return None

    # ── 写 ──────────────────────────────────────────────────────────

    def save(self, *, text: str, kind: str = "experiment_report", title: str = "",
             doc_id: str = "", experiment_id: str | None = None,
             sample_id: str | None = None, created_by: str = "",
             note: str = "", target_doc_id: str | None = None,
             target_version: int | None = None,
             related_experiment_ids: list[str] | None = None,
             legacy_stem: str | None = None,
             conversation_id: str | None = None,
             run_id: str | None = None,
             use_turn_context: bool = True) -> SaveResult:
        """保存一版内容。``doc_id`` 空 = 新文档，非空 = 该文档的新版本。

        **doc_id 给了但找不到时另立新文档并明说**（``doc_id_unknown=True``）。
        理由：报错等于把 LLM 已经写好的整篇内容扔了，而「多了一个文档」是可以
        事后合并的记账问题，「两个实验的报告混进一条版本历史」不可逆（陷阱 ⑪）。
        """
        body = (text or "")
        if not body.strip():
            return SaveResult(ok=False, error="内容为空，拒绝保存。")

        kind = normalize_kind(kind)
        conv_id, rid = conversation_id, run_id
        if use_turn_context and (conv_id is None or rid is None):
            try:
                from mast.core.turn_context import current_turn
                turn = current_turn()
                conv_id = conv_id if conv_id is not None else turn.get("conversation_id")
                rid = rid if rid is not None else turn.get("run_id")
            except Exception:  # noqa: BLE001 — provenance 是 best-effort，绝不阻塞保存
                pass

        requested = str(doc_id or "").strip()
        entry = self.get(requested) if requested else None
        if entry is not None:
            return self._add_version(entry, body, created_by=created_by, note=note,
                                     conversation_id=conv_id, run_id=rid)

        if experiment_id is None:
            experiment_id, scope_sample = dpaths.current_scope(self._st)
            if sample_id is None:
                sample_id = scope_sample
        res = self._create(
            text=body, kind=kind, title=title, experiment_id=experiment_id,
            sample_id=sample_id, created_by=created_by, note=note,
            target_doc_id=target_doc_id, target_version=target_version,
            related_experiment_ids=related_experiment_ids, legacy_stem=legacy_stem,
            conversation_id=conv_id, run_id=rid,
        )
        if res.ok and requested:
            res.doc_id_unknown = True
        return res

    def _create(self, *, text: str, kind: str, title: str,
                experiment_id: str | None, sample_id: str | None,
                created_by: str, note: str,
                target_doc_id: str | None = None, target_version: int | None = None,
                related_experiment_ids: list[str] | None = None,
                legacy_stem: str | None = None,
                conversation_id: str | None = None,
                run_id: str | None = None) -> SaveResult:
        did = ulid_now()
        created = now_iso()
        try:
            home, root_kind = dpaths.doc_home(kind, experiment_id, create=True, st=self._st)
        except OSError as exc:
            return SaveResult(ok=False, error=f"创建文档目录失败：{type(exc).__name__}: {exc}")
        if root_kind == "unfiled":
            experiment_id = None
            sample_id = None

        dir_name = doc_dir_name(kind, title or "未命名", did, created)
        doc_dir = home / dir_name
        doc_dir = self._mkdir_within_budget(home, kind, title, did, created)
        if doc_dir is None:
            return SaveResult(
                ok=False,
                error=("创建文档目录失败（路径超出系统上限，连最短的名字也放不下）。"
                       f"实验根 {home} 太深 —— 请把 MAST_EXPERIMENT_ROOT 指到更浅的路径。"))

        meta = DocMeta(
            doc_id=did, kind=kind, title=(title or "未命名").strip(),
            experiment_id=experiment_id, sample_id=sample_id,
            related_experiment_ids=[str(x) for x in (related_experiment_ids or []) if x],
            created_at=created, created_by=created_by or "unknown",
            conversation_id=conversation_id, run_id=run_id,
            target_doc_id=target_doc_id, target_version=target_version,
            latest_version=0, updated_at=created, dir_name=doc_dir.name,
            root_kind=root_kind, legacy_stem=legacy_stem,
        )
        write_doc_json(doc_dir, meta)
        entry = DocEntry(meta=meta, dir=doc_dir, versions=[])
        with _cache_guard:
            _dir_cache[did] = doc_dir
        return self._add_version(entry, text, created_by=created_by, note=note,
                                 conversation_id=conversation_id, run_id=run_id,
                                 created_new=True)

    def _mkdir_within_budget(self, home: Path, kind: str, title: str, doc_id: str,
                             created: str) -> Path | None:
        """建文档目录，**名字太长就逐级收缩 slug**（陷阱 ⑱ 承诺的那件事）。

        两道机制，缺一不可：

        1. **主动比长度**（``_path_budget_ok``）。第一版只做了下面那道「失败就重试」，
           而复审实测：本机 ``mkdir`` 一路建到 **416 字符**都不报错（``LongPathsEnabled``
           查不到，Python/NTFS 就是接受了），所以阶梯**一次都没执行过** —— 那条
           「深根下内容不丢」的测试是空过的（内容确实没丢，但原因是 mkdir 成功）。
           更要紧的是：⑱ 关心的从来不是「mkdir 有没有成功」，而是**互操作性** ——
           416 字符的路径 Python 读得回来，资源管理器和任何不支持长路径的工具打不开。
        2. **失败就重试**（保留）。上限跨平台、跨文件系统不一样，长路径开关还能被改，
           所以主动预算算过了也仍然可能失败；那时继续砍短，而不是让 ``_create``
           返回失败 = 把 LLM 写好的整篇报告扔掉。

        最后一档连标题都不要，只留 ``<码>__<日期>__<id8>``。
        """
        for max_chars in (SLUG_MAX_DOC, 16, 8, 0):
            dir_name = doc_dir_name(kind, title or "未命名", doc_id, created,
                                    max_chars=max_chars) if max_chars else \
                f"{KIND_CODES.get(normalize_kind(kind), 'doc')}__{(created or '')[:10]}__{doc_id8(doc_id)}"
            if max_chars and not _path_budget_ok(home, dir_name):
                logger.debug("文档目录名超出路径预算，收缩 slug（%s 字符）", max_chars)
                continue
            for candidate in (home / dir_name, home / f"{dir_name}__2"):
                try:
                    candidate.mkdir(parents=True, exist_ok=False)
                    return candidate
                except FileExistsError:
                    continue
                except OSError as exc:
                    logger.debug("文档目录名过长或不可用（%s 字符预算）：%r", max_chars, exc)
                    break
        return None

    def _add_version(self, entry: DocEntry, text: str, *, created_by: str = "",
                     note: str = "", conversation_id: str | None = None,
                     run_id: str | None = None,
                     created_new: bool = False) -> SaveResult:
        meta = entry.meta
        with _lock_for(meta.doc_id):
            versions = read_versions(entry.dir)
            v = (versions[-1].v if versions else 0) + 1
            path: Path | None = None
            # claim 循环：``open('x')`` 是跨进程的原子占位。另一个进程/线程抢到
            # 这个号，我们就往后挪 —— 绝不覆盖。
            for _ in range(64):
                cand = entry.dir / VERSION_FILE_FMT.format(v=v)
                try:
                    with cand.open("x", encoding="utf-8"):
                        pass
                    path = cand
                    break
                except FileExistsError:
                    v += 1
                except OSError as exc:
                    return SaveResult(ok=False, doc_id=meta.doc_id,
                                      error=f"占位失败：{type(exc).__name__}: {exc}")
            if path is None:
                return SaveResult(ok=False, doc_id=meta.doc_id,
                                  error="连续 64 个版本号都被占用，放弃（这不该发生）。")
            try:
                _write_atomic(path, text)
            except OSError as exc:
                return SaveResult(ok=False, doc_id=meta.doc_id,
                                  error=f"写入失败：{type(exc).__name__}: {exc}")

            vm = VersionMeta(
                v=v, file=path.name, sha256=sha256_text(text), words=count_words(text),
                created_at=now_iso(), created_by=created_by or "unknown",
                conversation_id=conversation_id, run_id=run_id, note=note,
            )
            try:
                _append_jsonl(entry.dir / VERSIONS_JSONL, json.loads(vm.to_line()))
            except (OSError, json.JSONDecodeError) as exc:
                # 版本文件已经落盘了 —— 读侧的目录扫描会把它兜住（自愈规则）。
                logger.warning("versions.jsonl append failed (%s): %r", entry.dir, exc)

            # 回写头部前**重读一次磁盘**：``entry.meta`` 可能是几秒前读的，而这段
            # 时间里用户可能刚改了标题、加了关联实验。拿内存里的旧副本整体覆盖
            # 就是一次静默的 lost update（版本文件不受影响，但改名会凭空回滚）。
            from mast.logging.v2.manifest import read_json
            fresh = DocMeta.from_json(read_json(entry.dir / DOC_JSON) or {})
            if fresh is not None and fresh.doc_id == meta.doc_id:
                fresh.dir_name = fresh.dir_name or entry.dir.name
                fresh.root_kind = meta.root_kind
                meta = fresh
                entry.meta = meta
            meta.latest_version = v
            meta.updated_at = vm.created_at
            if conversation_id and not meta.conversation_id:
                meta.conversation_id = conversation_id
            if run_id and not meta.run_id:
                meta.run_id = run_id
            write_doc_json(entry.dir, meta)
            entry.versions = versions + [vm]

        self._index(entry, vm)
        logger.info("documents: %s %s v%d → %s", meta.kind, meta.doc_id, v, path)
        # 产物落盘事件（2026-07-30）。在这里，而不是在 ``save()``：这是新建文档和
        # 新增版本**唯一共同**的成功点，挂在 save() 上会漏掉 ``_add_version`` 那条路。
        # 纯 best-effort —— 发布失败绝不影响保存(文件已经在磁盘上了,那才是权威)。
        _publish_artifact_saved(kind=meta.kind, doc_id=meta.doc_id, version=v,
                                title=meta.title, path=str(path),
                                experiment_id=meta.experiment_id or "")
        return SaveResult(
            ok=True, doc_id=meta.doc_id, version=v, path=str(path), kind=meta.kind,
            title=meta.title, experiment_id=meta.experiment_id,
            root_kind=meta.root_kind, created_new=created_new, words=vm.words,
        )

    def patch(self, doc_id: str, *, title: str | None = None, kind: str | None = None,
              sample_id: str | None = None,
              related_experiment_ids: list[str] | None = None,
              target_doc_id: str | None = None,
              target_version: int | None = None) -> SaveResult:
        """改可变头部。**不动目录名** —— 目录名是创建时冻结的（陷阱：改目录等于改身份）。"""
        entry = self.get(doc_id)
        if entry is None:
            return SaveResult(ok=False, error=f"找不到文档 {doc_id!r}。")
        meta = entry.meta
        with _lock_for(meta.doc_id):
            if title is not None and title.strip() and title.strip() != meta.title:
                meta.title_history.append({"title": meta.title, "changed_at": now_iso()})
                meta.title = title.strip()
            if kind is not None:
                meta.kind = normalize_kind(kind, meta.kind)
            if sample_id is not None:
                meta.sample_id = sample_id or None
            if related_experiment_ids is not None:
                seen: list[str] = []
                for x in related_experiment_ids:
                    s = str(x or "").strip()
                    if s and s != meta.experiment_id and s not in seen:
                        seen.append(s)
                meta.related_experiment_ids = seen
            if target_doc_id is not None:
                meta.target_doc_id = target_doc_id or None
            if target_version is not None:
                meta.target_version = int(target_version) or None
            meta.updated_at = now_iso()
            write_doc_json(entry.dir, meta)
        self._index(entry, None)
        return SaveResult(ok=True, doc_id=meta.doc_id, version=entry.latest_version,
                          kind=meta.kind, title=meta.title,
                          experiment_id=meta.experiment_id, root_kind=meta.root_kind,
                          path=str(entry.dir))

    def claim(self, doc_id: str, experiment_id: str) -> SaveResult:
        """认领 / 改主归属 —— **物理搬家**（主归属决定落点，见设计决策 2）。

        搬完还要把文档引用到的 ``_assets/`` 图复制进目标实验的图池（陷阱 ⑫）：
        markdown 里的链接是 ``../_assets/x.png``，跨实验搬家后指向的是新实验的
        图池，图不跟过去就是一堆碎图标。
        """
        entry = self.get(doc_id)
        if entry is None:
            return SaveResult(ok=False, error=f"找不到文档 {doc_id!r}。")
        eid = str(experiment_id or "").strip()
        if not eid:
            return SaveResult(ok=False, error="claim 需要一个实验 id。")
        if entry.meta.experiment_id == eid and entry.meta.root_kind == "experiment":
            return SaveResult(ok=True, doc_id=entry.doc_id, kind=entry.meta.kind,
                              title=entry.meta.title, experiment_id=eid,
                              root_kind="experiment", path=str(entry.dir),
                              version=entry.latest_version)
        target_home, root_kind = dpaths.doc_home(entry.meta.kind, eid, create=True, st=self._st)
        if root_kind != "experiment":
            return SaveResult(ok=False, error=f"实验 {eid} 不存在或目录无法创建。")
        dest = target_home / entry.dir.name
        if dest.exists():
            return SaveResult(ok=False, error=f"目标位置已存在同名目录：{dest}")

        src_assets = dpaths.assets_dir_for(entry.dir.parent, entry.meta.root_kind, create=False)
        with _lock_for(entry.doc_id):
            try:
                shutil.move(str(entry.dir), str(dest))
            except OSError as exc:
                return SaveResult(ok=False, error=f"搬移失败：{type(exc).__name__}: {exc}")
            copied = _copy_referenced_assets(dest, src_assets,
                                             dpaths.assets_dir_for(target_home, "experiment", create=True))
            meta = entry.meta
            meta.experiment_id = eid
            meta.root_kind = "experiment"
            meta.updated_at = now_iso()
            if eid in (meta.related_experiment_ids or []):
                meta.related_experiment_ids = [x for x in meta.related_experiment_ids if x != eid]
            write_doc_json(dest, meta)
            entry.dir = dest
            try:
                _append_jsonl(dest / "moves.jsonl", {
                    "op": "claim", "at": meta.updated_at, "experiment_id": eid,
                    "assets_copied": copied})
            except OSError as exc:
                logger.warning("moves.jsonl append failed (%s): %r", dest, exc)
        with _cache_guard:
            _dir_cache[entry.doc_id] = dest
        self._index(entry, None)
        logger.info("documents: claimed %s → experiment %s (%d assets copied)",
                    entry.doc_id, eid, copied)
        return SaveResult(ok=True, doc_id=entry.doc_id, kind=entry.meta.kind,
                          title=entry.meta.title, experiment_id=eid,
                          root_kind="experiment", path=str(dest),
                          version=entry.latest_version)

    def discard(self, doc_id: str, *, reason: str = "") -> SaveResult:
        """废弃一份文档 —— **搬进 ``_discarded`` 区，一个字节都不删。**

        为什么不是删除：``_quarantine`` 定下的规矩是「绝不丢字节，宁可事后认领」。
        一份看着是垃圾的报告，可能是用户唯一还留着的那一版。

        为什么需要它：``save()`` 在 doc_id 找不到时刻意另立新文档（内容损失不可
        接受），所以增殖是被允许的失败模式 —— 允许增殖的前提是**事后能清理**。
        没有这条路的话，一个忘传 doc_id 的 agent 就能在报告列表里永久留下一串重复。

        ``restore()`` 或 ``claim()`` 可以把它搬回来。
        """
        entry = self.get(doc_id)
        if entry is None:
            return SaveResult(ok=False, error=f"找不到文档 {doc_id!r}。")
        if entry.meta.root_kind == "discarded":
            return SaveResult(ok=True, doc_id=entry.doc_id, kind=entry.meta.kind,
                              title=entry.meta.title, root_kind="discarded",
                              path=str(entry.dir), version=entry.latest_version)
        try:
            # 有归属的废弃到**它自己实验文件夹里**的 reports/_discarded/ —— 因为
            # discard 保留 experiment_id，那份文档逻辑上还属于这个实验，搬出去就
            # 破了「一个实验的所有数据都在一个文件夹里」。
            dest_home = dpaths.discarded_home_for(entry.meta.experiment_id,
                                                  create=True, st=self._st)
        except OSError as exc:
            return SaveResult(ok=False, error=f"无法创建废弃区：{exc}")
        return self._move_zone(entry, dest_home, "discarded",
                               {"op": "discard", "reason": reason,
                                "from_experiment_id": entry.meta.experiment_id})

    def restore(self, doc_id: str, *, experiment_id: str = "") -> SaveResult:
        """把废弃的文档搬回来。

        落点的优先级：显式给的 ``experiment_id`` > 文档**自己记着的**主归属 >
        ``_unfiled`` 待认领。

        中间那一档不是「猜」：``discard`` 刻意**不清 experiment_id**（区由路径决定，
        不需要靠清空归属来表达「已废弃」），所以那是文档自己带着的事实。把东西丢进
        未归属区让用户再找一遍原主，才是白扔掉已知信息。
        """
        entry = self.get(doc_id)
        if entry is None:
            return SaveResult(ok=False, error=f"找不到文档 {doc_id!r}。")
        eid = str(experiment_id or "").strip() or str(entry.meta.experiment_id or "").strip()
        if eid:
            res = self.claim(doc_id, eid)
            if res.ok:
                self.append_event(doc_id, "moves.jsonl",
                                  {"op": "restore", "at": now_iso(), "experiment_id": eid})
                return res
            # 原实验行没了（被删/库被换过），``claim`` 会失败。**继续往 _unfiled 兜底**，
            # 否则文档永久卡在废弃区：常规列表不扫那里，restore 又走不通，用户只能
            # 先勾「含已废弃」才看得见、再手动指定另一个存活实验才能捞出来。
            # docstring 承诺的三级优先里，最后一级不能因为第二级失败就消失。
            logger.warning("restore: 目标实验 %s 不可用（%s）—— 退回 _unfiled 待认领",
                           eid, res.error)
        if entry.meta.root_kind != "discarded":
            return SaveResult(ok=True, doc_id=entry.doc_id, kind=entry.meta.kind,
                              title=entry.meta.title, root_kind=entry.meta.root_kind,
                              path=str(entry.dir), version=entry.latest_version)
        try:
            dest_home = dpaths.unfiled_docs_dir(create=True)
        except OSError as exc:
            return SaveResult(ok=False, error=f"无法创建未归属区：{exc}")
        return self._move_zone(entry, dest_home, "unfiled", {"op": "restore"})

    def _move_zone(self, entry: DocEntry, dest_home: Path, root_kind: str,
                   event: dict) -> SaveResult:
        """在区之间搬一个文档目录。``moves.jsonl`` 留痕（只追加）。"""
        dest = dest_home / entry.dir.name
        if dest.exists():
            return SaveResult(ok=False, error=f"目标位置已存在同名目录：{dest}")
        with _lock_for(entry.doc_id):
            try:
                shutil.move(str(entry.dir), str(dest))
            except OSError as exc:
                return SaveResult(ok=False, error=f"搬移失败：{type(exc).__name__}: {exc}")
            meta = entry.meta
            meta.root_kind = root_kind
            # 搬进 ``_discarded`` **保留**主归属：区由路径表达，不必靠清空归属来说
            # 「已废弃」。留着它，``restore`` 才知道该放回哪个实验，而且
            # ``list(experiment_id=X, include_discarded=True)`` 能如实回答「这个实验
            # 还有一份被废弃的报告」。只有明确落到 ``_unfiled`` 才真的没有主归属。
            if root_kind == "unfiled":
                meta.experiment_id = None
                meta.sample_id = None
            meta.updated_at = now_iso()
            write_doc_json(dest, meta)
            entry.dir = dest
            try:
                _append_jsonl(dest / "moves.jsonl", {"at": meta.updated_at, **event})
            except OSError as exc:
                logger.warning("moves.jsonl append failed (%s): %r", dest, exc)
        with _cache_guard:
            _dir_cache[entry.doc_id] = dest
        self._index(entry, None)
        logger.info("documents: %s → %s (%s)", entry.doc_id, root_kind, dest)
        return SaveResult(ok=True, doc_id=entry.doc_id, kind=entry.meta.kind,
                          title=entry.meta.title, root_kind=root_kind,
                          path=str(dest), version=entry.latest_version)

    # ── 边车：事件日志与视图（计划进度用） ──────────────────────────

    def append_event(self, doc_id: str, filename: str, obj: dict) -> bool:
        """往文档目录里的一个 jsonl 追加一条事件。永不抛。"""
        entry = self.get(doc_id)
        if entry is None:
            return False
        try:
            _append_jsonl(entry.dir / filename, obj)
            return True
        except OSError as exc:
            logger.warning("append_event(%s/%s) failed: %r", doc_id, filename, exc)
            return False

    def read_events(self, doc_id: str, filename: str) -> list[dict]:
        entry = self.get(doc_id)
        if entry is None:
            return []
        p = entry.dir / filename
        if not p.is_file():
            return []
        out: list[dict] = []
        try:
            for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    out.append(obj)
        except OSError:
            return out
        return out

    def write_view(self, doc_id: str, filename: str, text: str) -> bool:
        """原子替换一个**视图**文件（不是版本）。

        视图可以被反复重写 —— 它是事件日志的渲染结果，丢了随时能重生成。版本
        文件永不覆盖，两者的语义必须分开（设计 §3.7：定义修订才是版本，进度是
        事件 + 视图）。
        """
        entry = self.get(doc_id)
        if entry is None:
            return False
        try:
            _write_atomic(entry.dir / filename, text)
            return True
        except OSError as exc:
            logger.warning("write_view(%s/%s) failed: %r", doc_id, filename, exc)
            return False

    # ── 导出落点 ────────────────────────────────────────────────────

    def export_path(self, entry: DocEntry, suffix: str = ".html",
                    *, stamp: str = "") -> Path:
        """``<exp>/exports/<slug>__<id8>_vNNN_<ts><suffix>``。

        带时间戳、多份共存、随时可重跑 —— 与 ``export_record_db`` / ``export_rocrate``
        同一形态（「按需动作，不是收尾步骤」）。旧实现是 ``data/reports/<stem>.html``
        直接覆盖，上一份交付件就这么没了。
        """
        from datetime import datetime
        ts = stamp or datetime.now().strftime("%Y%m%dT%H%M%S")
        if entry.meta.root_kind == "experiment" and entry.meta.experiment_id:
            base = dpaths.exp_dir_for(entry.meta.experiment_id, create=True, st=self._st)
            out_dir = (base / "exports") if base is not None else (entry.dir / "exports")
        else:
            out_dir = dpaths.unfiled_docs_dir(create=True) / "exports"
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = entry.dir.name.split("__")
        name = "__".join(stem[2:]) if len(stem) >= 3 else entry.dir.name
        base_name = f"{name}_v{entry.latest_version:03d}_{ts}"
        cand = out_dir / f"{base_name}{suffix}"
        # 时间戳只到秒 —— 同一秒里导两次会撞同一个名字，第二次就把第一份盖掉了。
        # 加序号后缀（与 filestore._resolve_dest 的 ``__001`` 同一手法）：导出件
        # 是可再生的，但「多份共存、随时可重跑」这句话得真的成立。
        n = 2
        while cand.exists() and n < 1000:
            cand = out_dir / f"{base_name}-{n}{suffix}"
            n += 1
        return cand

    # ── DB 索引（best-effort） ──────────────────────────────────────

    def _index(self, entry: DocEntry, vm: VersionMeta | None) -> None:
        """写 DB 索引。失败只告警 —— 文件夹已经是权威，reindex 随时能补上。"""
        try:
            st = dpaths.storage(self._st)
            st.upsert_document(entry.meta.to_json(), entry.meta.related_experiment_ids)
            if vm is not None:
                st.insert_document_version(entry.doc_id, {
                    "version": vm.v,
                    "rel_path": f"{entry.dir.name}/{vm.file}",
                    "sha256": vm.sha256, "words": vm.words,
                    "created_at": vm.created_at, "created_by": vm.created_by,
                    "conversation_id": vm.conversation_id, "run_id": vm.run_id,
                    "note": vm.note,
                })
        except Exception as exc:  # noqa: BLE001
            logger.warning("document index write failed (%s): %r", entry.doc_id, exc)


_ASSET_REF_RE = re.compile(r"_assets/([^\s\)\"'>]+)")


def _copy_referenced_assets(doc_dir: Path, src_assets: Path, dst_assets: Path) -> int:
    """把文档正文引用到的图从源图池复制到目标图池。返回复制张数。"""
    if not src_assets.is_dir() or src_assets.resolve() == dst_assets.resolve():
        return 0
    names: set[str] = set()
    try:
        for p in doc_dir.iterdir():
            if p.suffix.lower() != ".md":
                continue
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for m in _ASSET_REF_RE.finditer(text):
                names.add(m.group(1).split("/")[-1])
    except OSError:
        return 0
    copied = 0
    for name in sorted(names):
        src = src_assets / name
        dst = dst_assets / name
        if not src.is_file() or dst.exists():
            continue
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied += 1
        except OSError as exc:
            logger.warning("asset copy failed (%s): %r", src, exc)
    return copied


# ── 模块级便捷入口（裸 @tool 函数用） ──────────────────────────────

_default: DocumentStore | None = None


def store() -> DocumentStore:
    """进程内默认 store。裸 ``@tool`` 函数没有 ctx 可注入，走这里。"""
    global _default
    if _default is None:
        _default = DocumentStore()
    return _default


def reset_caches() -> None:
    """清掉进程内缓存。测试在切换 ``MAST_EXPERIMENT_ROOT`` 后必须调。"""
    global _default
    with _cache_guard:
        _dir_cache.clear()
    _default = None
