"""从实验文件夹重建 DB 索引 —— 让「自包含」可证明，而不只是一句口号。

设计文档：``docs/v2/design/experiment_folder_persistence.md`` §6 / §15(P2)

整套设计的地基是一句判断：**DB 是索引（可重建），文件夹是记录（权威字节 + JSON
边车）。** 这个模块是那句话的可执行证明 —— 只要它能跑通，"DB 没了可以重建、
文件夹没了才是真丢"就是事实而非愿望。

三件事：

* :func:`scan_folders` —— 只读盘点整个 experiment_root，把每个实验文件夹的
  manifest 读出来。不碰 DB。
* :func:`rebuild_from_folders` —— 用盘点结果补齐 v1 的实验/样品行和 v2 的
  ``file_locations``。**只补不改**：已经存在的行一律不动，因为 DB 里可能有
  文件夹里没有的东西（动作、审批、轨迹）。
* :func:`verify_all` —— 逐实验做 fixity 校验（``scan_files.fixity_ok`` 这个列
  至今从未被真正验证过）。

为什么是"只补不改"
------------------

文件夹里有的是**字节和它的来历**；DB 里还有动作时间线、审批、轨迹、观测。
重建能恢复前者，不能凭空造出后者。所以一次重建之后 DB 至少不比之前差 —— 这是
安全的操作，可以在任何时候跑。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def scan_folders(root: Path | None = None) -> list[dict]:
    """只读盘点 experiment_root 下的所有实验文件夹。不碰 DB。"""
    from mast.core.experiment_paths import experiment_root
    from mast.logging.v2.filestore import read_manifest
    from mast.logging.v2.manifest import read_json

    base = Path(root) if root is not None else experiment_root()
    out: list[dict] = []
    if not base.is_dir():
        return out

    for d in sorted(base.iterdir()):
        if not d.is_dir() or d.name.startswith("_"):
            continue
        meta = read_json(d / "experiment.json")
        if not meta:
            continue
        samples = []
        sdir = d / "samples"
        if sdir.is_dir():
            for sp in sorted(sdir.iterdir()):
                if not sp.is_dir():
                    continue
                smeta = read_json(sp / "sample.json")
                rows = read_manifest(sp / "raw" / "_manifest.jsonl")
                files = [r for r in rows if not r.get("duplicate_of")]
                samples.append({
                    "dir_name": sp.name, "meta": smeta,
                    "files": files,
                    "bytes": sum(int(r.get("size") or 0) for r in files),
                })
        out.append({
            "dir_name": d.name, "path": str(d), "meta": meta, "samples": samples,
            "chats": _count_files(d / "chats", ".jsonl")
                     + sum(_count_files(Path(s["path"]) / "chats", ".jsonl")
                           if s.get("path") else 0 for s in []),
            "env_files": _count_files(d / "env", ".csv"),
            "documents": _count_doc_dirs(d),
            "library_members": _count_lines(d / "library" / "members.jsonl"),
        })
    return out


def _count_doc_dirs(exp_dir: Path) -> int:
    """该实验下的文档数（``reports/`` 与 ``plans/`` 里带 ``doc.json`` 的目录）。"""
    n = 0
    for sub in ("reports", "plans"):
        d = exp_dir / sub
        if not d.is_dir():
            continue
        try:
            for child in d.iterdir():
                if child.is_dir() and (child / "doc.json").is_file():
                    n += 1
        except OSError:
            continue
    return n


def _count_lines(path: Path) -> int:
    try:
        return sum(1 for line in path.read_text(encoding="utf-8",
                                                errors="replace").splitlines()
                   if line.strip())
    except OSError:
        return 0


def _count_files(d: Path, suffix: str) -> int:
    try:
        return sum(1 for p in d.iterdir() if p.is_file() and p.suffix == suffix)
    except OSError:
        return 0


def rebuild_from_folders(*, storage: Any, repos: Any = None,
                         root: Path | None = None, dry_run: bool = False) -> dict:
    """从文件夹补齐 DB 索引。**只补不改**，可以随时跑。

    返回 ``{experiments_added, samples_added, locations_added, skipped, errors}``。
    """
    res = {"experiments_added": 0, "samples_added": 0, "locations_added": 0,
           "documents_added": 0, "document_versions_added": 0,
           "libraries_rebuilt": 0, "skipped": 0, "errors": [],
           "dry_run": bool(dry_run)}

    for folder in scan_folders(root):
        meta = folder["meta"]
        eid = str(meta.get("id") or "")
        if not eid:
            res["errors"].append(f"{folder['dir_name']}: experiment.json 没有 id")
            continue
        try:
            existing = storage.get_experiment(eid)
        except Exception as exc:  # noqa: BLE001
            res["errors"].append(f"{folder['dir_name']}: {exc}")
            continue

        if existing is None:
            if not dry_run:
                _insert_experiment(storage, eid, meta, folder["dir_name"])
            res["experiments_added"] += 1
        else:
            res["skipped"] += 1
            if not dry_run and not existing.get("dir_name"):
                try:
                    storage.set_experiment_dir_name(eid, folder["dir_name"])
                except Exception:  # noqa: BLE001
                    pass

        for s in folder["samples"]:
            smeta = s.get("meta") or {}
            sid = str(smeta.get("id") or "")
            if not sid:
                continue
            try:
                if storage.get_sample(sid) is None:
                    if not dry_run:
                        _insert_sample(storage, sid, eid, smeta, s["dir_name"])
                    res["samples_added"] += 1
                elif not dry_run:
                    storage.set_sample_dir_name(sid, s["dir_name"],
                                                int(smeta.get("index") or 1))
            except Exception as exc:  # noqa: BLE001
                res["errors"].append(f"{s['dir_name']}: {exc}")

            if repos is None or dry_run:
                continue
            for row in s["files"]:
                try:
                    repos.file_locations.record(
                        sha256=str(row.get("sha256") or ""),
                        experiment_id=eid, sample_id=sid,
                        rel_path=str(row.get("rel_path") or ""),
                        source=str(row.get("source") or "import"),
                        origin_path=row.get("origin_path"),
                        size_bytes=int(row.get("size") or 0),
                        status="ok")
                    res["locations_added"] += 1
                except Exception:  # noqa: BLE001
                    # file_locations 的 experiment_id 外键指向 **v2** 的实验行；
                    # 重建 v1 索引时那一行未必存在，失败是预期内的，不算错误。
                    pass

    # 文档与文献库索引。放在实验/样品循环之后是刻意的：文档行引用实验 id，
    # 实验行得先在（虽然 documents 表没有外键约束 —— 它是索引不是账本）。
    _rebuild_documents(storage, res, root=root, dry_run=dry_run)
    _rebuild_libraries(res, root=root, dry_run=dry_run)
    return res


def _rebuild_documents(storage: Any, res: dict, *, root: Path | None = None,
                       dry_run: bool = False) -> None:
    """从 ``doc.json`` + ``versions.jsonl`` 重建 documents / document_versions。

    「只补不改」：已有的行不动（``INSERT OR REPLACE`` 的是**索引**，一行就是
    sidecar 当前状态的投影，没有历史可丢；版本行用 ``INSERT OR IGNORE``）。

    这是「文件夹是记录、DB 是可重建索引」这句话对文档子系统的兑现 —— 集成测试
    会删掉整个 DB 再跑这里，然后断言文档全部复原。
    """
    try:
        from mast.documents.store import _scan_all, reset_caches
    except Exception as exc:  # noqa: BLE001
        res["errors"].append(f"documents 模块不可用：{exc!r}")
        return
    if root is not None:
        # 显式传 root 时清缓存，否则 doc_id → 目录 的进程内缓存会给出旧根的结果。
        # （``root`` 从前是被**吞掉**的：这里只清缓存，随后 ``_scan_all`` 自己从
        # experiment_paths 解析根 —— 于是 rebuild_from_folders(root=A) 会从 A 重建
        # 实验/样品、却从全局配置根重建文档。签名读起来像支持，实际不支持。）
        reset_caches()
    try:
        # **连废弃区一起索引。** 常规列表默认不扫 ``_discarded``（废弃的不该出现在
        # 报告列表里），但重建索引要的是「把文件夹里有的东西如实登记进 DB」——
        # 漏掉它们，一台新机器上 reindex 之后 DB 就少了一部分磁盘上确实存在的文档，
        # 「DB 是可重建索引」这句话对废弃文档就不成立了。读路径不受影响（走文件夹），
        # 这里补的是索引的完整性。
        entries = _scan_all(include_discarded=True, root=root)
    except Exception as exc:  # noqa: BLE001
        res["errors"].append(f"文档扫描失败：{exc!r}")
        return
    for e in entries:
        if dry_run:
            res["documents_added"] += 1
            res["document_versions_added"] += len(e.versions)
            continue
        try:
            storage.upsert_document(e.meta.to_json(), e.meta.related_experiment_ids)
            res["documents_added"] += 1
            for vm in e.versions:
                storage.insert_document_version(e.doc_id, {
                    "version": vm.v,
                    "rel_path": f"{e.dir.name}/{vm.file}",
                    "sha256": vm.sha256, "words": vm.words,
                    "created_at": vm.created_at, "created_by": vm.created_by,
                    "conversation_id": vm.conversation_id, "run_id": vm.run_id,
                    "note": vm.note,
                })
                res["document_versions_added"] += 1
        except Exception as exc:  # noqa: BLE001
            res["errors"].append(f"{e.dir.name}: {exc}")


def _rebuild_libraries(res: dict, *, root: Path | None = None,
                       dry_run: bool = False) -> None:
    """从各实验的 ``library/members.jsonl`` 重建 registry 里的实验库条目。

    书目的权威在实验文件夹（复制实验文件夹就带走了书目）；registry.json 是全局
    索引 + 缓存。冲突时**文件夹赢** —— 绝不反向用 registry 重写文件夹。
    """
    if dry_run:
        return
    try:
        from mast.knowledge.experiment_library import rebuild_registry_from_folders
    except Exception:  # noqa: BLE001 — 文献库子系统可选，缺了不该让重建失败
        return
    try:
        res["libraries_rebuilt"] = int(rebuild_registry_from_folders(root=root) or 0)
    except Exception as exc:  # noqa: BLE001
        res["errors"].append(f"文献库重建失败：{exc!r}")


def _insert_experiment(storage: Any, eid: str, meta: dict, dir_name: str) -> None:
    with storage._connect() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO experiments "
            "(id, name, goal_text, start_time, status, notes, dir_name, last_active_at) "
            "VALUES (?, ?, ?, ?, 'running', ?, ?, ?)",
            (eid, meta.get("title") or "(重建)", meta.get("goal") or "",
             meta.get("created_at") or "", "由 reindex 从实验文件夹重建",
             dir_name, meta.get("last_active_at")))


def _insert_sample(storage: Any, sid: str, eid: str, smeta: dict, dir_name: str) -> None:
    with storage._connect() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO samples "
            "(id, experiment_id, name, description, start_time, status, "
            " sample_type, sample_subtype, dir_name, sample_index, last_active_at) "
            "VALUES (?, ?, ?, ?, ?, 'active', ?, ?, ?, ?, ?)",
            (sid, eid, smeta.get("name") or "(重建)", smeta.get("description") or "",
             smeta.get("created_at") or "", smeta.get("sample_type") or "",
             smeta.get("sample_subtype") or "", dir_name,
             int(smeta.get("index") or 1), smeta.get("last_active_at")))


def verify_all(root: Path | None = None, *, deep: bool = False) -> dict:
    """逐实验做完整性校验。``deep=True`` 重算 sha256。

    ``scan_files.fixity_ok`` 这一列自建库以来从未被真正验证过 —— 这里是第一个
    真的去读字节的地方。
    """
    from mast.logging.v2.filestore import ExperimentFileStore

    out = {"experiments": 0, "files": 0, "missing": [], "size_mismatch": [],
           "sha_mismatch": []}
    for folder in scan_folders(root):
        out["experiments"] += 1
        try:
            r = ExperimentFileStore(Path(folder["path"])).verify_folder(deep=deep)
        except Exception as exc:  # noqa: BLE001
            out["missing"].append(f"{folder['dir_name']}: {exc}")
            continue
        out["files"] += r["files"]
        for k in ("missing", "size_mismatch", "sha_mismatch"):
            out[k].extend(f"{folder['dir_name']}/{x}" for x in r[k])
    return out


def export_rocrate(exp_dir: Path, *, repos: Any, v2_experiment_id: str) -> Path | None:
    """把 RO-Crate 导出到实验文件夹自己的 ``exports/`` 下。

    ``rocrate.export`` 本身早就存在（而且是本仓库唯一"按实验归档测量文件"的既有
    实现），只是从前没有一个自然的落点。现在有了：实验文件夹。同样是**按需**
    动作，可以有多份、随时重跑。
    """
    from datetime import datetime, timezone
    try:
        from mast.logging.v2.rocrate import export as _export
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        out = Path(exp_dir) / "exports" / f"ro-crate_{ts}"
        return _export(repos, v2_experiment_id, out, copy_files=True)
    except Exception as exc:  # noqa: BLE001
        logger.warning("rocrate export failed: %r", exc)
        return None


def export_record_db(exp_dir: Path, *, storage: Any) -> Path | None:
    """把本实验相关的 DB 行抽成一个独立的 ``exports/record_<ts>.sqlite``。

    **按需动作，不是收尾步骤** —— 实验永不归档，这个文件随时可以重新生成，也
    可以有很多份。给需要结构化查询、又不想装 MAST 的场景用。
    """
    import sqlite3
    from datetime import datetime, timezone

    from mast.logging.v2.manifest import read_json

    meta = read_json(Path(exp_dir) / "experiment.json")
    eid = str(meta.get("id") or "")
    if not eid:
        return None
    out_dir = Path(exp_dir) / "exports"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    dest = out_dir / f"record_{ts}.sqlite"

    try:
        src = sqlite3.connect(str(getattr(storage, "_db_path")))
        src.row_factory = sqlite3.Row
        dst = sqlite3.connect(str(dest))
        try:
            for table, where in (("experiments", "id = ?"),
                                 ("samples", "experiment_id = ?"),
                                 ("actions", "experiment_id = ?"),
                                 ("conversation_log", "experiment_id = ?"),
                                 ("map_markers", "experiment_id = ?"),
                                 ("feedback", "experiment_id = ?"),
                                 ("environment_log", "experiment_id = ?"),
                                 # 文档索引（正文本身就在实验文件夹里，随导出目录
                                 # 一起走；这里带的是索引与版本表，让导出的 db
                                 # 能独立回答「这个实验有哪些报告/计划」）。
                                 ("documents", "experiment_id = ?"),
                                 ("document_versions",
                                  "doc_id IN (SELECT doc_id FROM documents "
                                  "WHERE experiment_id = ?)"),
                                 ("document_links", "experiment_id = ?")):
                try:
                    cols = [r[1] for r in src.execute(f"PRAGMA table_info({table})")]
                    if not cols:
                        continue
                    ddl = src.execute(
                        "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
                        (table,)).fetchone()
                    if ddl and ddl[0]:
                        dst.execute(ddl[0])
                    rows = src.execute(
                        f"SELECT * FROM {table} WHERE {where}", (eid,)).fetchall()
                    if rows:
                        ph = ",".join("?" for _ in cols)
                        dst.executemany(
                            f"INSERT INTO {table} VALUES ({ph})",
                            [tuple(r[c] for c in cols) for r in rows])
                except sqlite3.Error as exc:
                    logger.debug("export_record_db: %s skipped (%s)", table, exc)
            dst.commit()
        finally:
            src.close()
            dst.close()
        return dest
    except Exception as exc:  # noqa: BLE001
        logger.warning("export_record_db failed: %r", exc)
        return None
