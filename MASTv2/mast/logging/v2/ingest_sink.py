"""永不阻塞的数据落盘队列 —— 把文件复制赶出 skill 的返回路径。

设计文档：``docs/v2/design/experiment_folder_persistence.md`` §7.1

形状照抄 ``trace_sink.py::QueuedTraceSink``（同一个仓库里已验证过的范式），
包括 :meth:`_note_failure` 的"首次 LOUD、之后 ≤1 次/分钟"告警节流 —— 那条纪律
是 2026-07-27 forensics 的产物：一个静默吞掉 100% 失败率的 sink 会让日志看起来
只是"空"，而不是"坏了"。

为什么是单 worker
-----------------

一个 200 MB 的 .3ds 复制不该有四路并发把磁盘打死。FIFO 单线程也让同一个文件的
重试顺序可预期。

★ scope 必须在 submit() 时快照
------------------------------

``exp_dir`` / ``sample_dir_name`` 由调用方在 :meth:`submit` 时同步传入，**绝不**
在 worker 线程里再去查"当前样品"。否则一次快速换样品会把上一个样品的扫描
归到新样品名下 —— 这种错归属事后无法分辨，因为文件已经躺在错的目录里了。
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mast.logging.v2.filestore import ExperimentFileStore, IngestResult, Zone

logger = logging.getLogger(__name__)

_FAIL_QUIET_S = 60.0
#: 文件"还在写"时的退避重试节奏。3 次之后放弃并记 failed。
_RETRY_DELAYS = (2.0, 8.0, 30.0)


@dataclass
class _Job:
    path: Path
    exp_dir: Path
    sample_dir_name: str
    experiment_id: str
    sample_id: str | None
    action_id: str | None
    source: str
    skill: str = ""
    copy_mode: str = "copy"
    zone: Zone | None = None
    active_sample_dir: Path | None = None
    stray_sample_dir_name: str = ""
    attempts: int = 0
    not_before: float = 0.0


class NullIngestSink:
    """关闭数据归档时的空实现。调用方永远不用判空。"""

    def submit(self, *a: Any, **kw: Any) -> None:
        pass

    def stats(self) -> dict:
        return {"enabled": False}

    def flush(self, timeout: float | None = None) -> bool:
        return True

    def close(self) -> None:
        pass


class QueuedIngestSink:
    """把文件收进实验文件夹，全部在后台单线程上做。

    :meth:`submit` 只做一次 ``put_nowait``，队列满就丢弃并计数 —— **绝不阻塞
    skill 的返回**。扫描流程的正确性不能依赖磁盘有空间。
    """

    def __init__(self, repos: Any = None, *, maxsize: int = 2048,
                 on_result=None) -> None:
        self._repos = repos
        self._on_result = on_result
        self._q: queue.Queue = queue.Queue(maxsize=maxsize)
        self._stop = object()
        self._deferred: list[_Job] = []
        self._stores: dict[str, ExperimentFileStore] = {}
        self._fail_log: dict[str, list] = {}
        self._lock = threading.Lock()

        self.queued = 0
        self.done = 0
        self.duplicate = 0
        self.skipped = 0
        self.failed = 0
        self.dropped = 0
        self.bytes_copied = 0
        self.last_error: str | None = None

        self._worker = threading.Thread(
            target=self._run, name="mast-data-ingest", daemon=True)
        self._worker.start()

    # ── 提交 ──────────────────────────────────────────────────────────

    def submit(
        self, paths, *,
        exp_dir: str | Path,
        sample_dir_name: str,
        experiment_id: str = "",
        sample_id: str | None = None,
        action_id: str | None = None,
        source: str = "skill",
        skill: str = "",
        copy_mode: str = "copy",
        zone: Zone | None = None,
        active_sample_dir: Path | None = None,
        stray_sample_dir_name: str = "",
    ) -> None:
        """把一批路径排进落盘队列。永不阻塞、永不抛。"""
        if not paths:
            return
        for p in paths:
            try:
                job = _Job(
                    path=Path(p), exp_dir=Path(exp_dir),
                    sample_dir_name=sample_dir_name,
                    experiment_id=experiment_id, sample_id=sample_id,
                    action_id=action_id, source=source, skill=skill,
                    copy_mode=copy_mode, zone=zone,
                    active_sample_dir=active_sample_dir,
                    stray_sample_dir_name=stray_sample_dir_name,
                )
                self._q.put_nowait(job)
                self.queued += 1
            except queue.Full:
                self.dropped += 1
                self._note_failure("overflow", None,
                                   "ingest queue full — dropping %s" % p)
            except Exception as exc:  # noqa: BLE001 — 提交路径绝不抛
                self.dropped += 1
                self._note_failure("submit", exc, "ingest submit failed")

    # ── worker ────────────────────────────────────────────────────────

    def _store_for(self, exp_dir: Path) -> ExperimentFileStore:
        key = str(exp_dir)
        st = self._stores.get(key)
        if st is None:
            st = ExperimentFileStore(exp_dir)
            self._stores[key] = st
        return st

    def _run(self) -> None:
        while True:
            timeout = self._deferred_wait()
            try:
                item = self._q.get(timeout=timeout)
            except queue.Empty:
                self._promote_deferred()
                continue
            try:
                if item is self._stop:
                    return
                self._handle(item)
            except Exception as exc:  # noqa: BLE001 — worker 绝不能死
                self.failed += 1
                self.last_error = f"{type(exc).__name__}: {exc}"
                self._note_failure(f"worker:{type(exc).__name__}", exc, "ingest worker failed")
            finally:
                self._q.task_done()

    def _deferred_wait(self) -> float:
        """下一个待重试任务还要等多久（没有就等 1 秒去轮询队列）。"""
        with self._lock:
            if not self._deferred:
                return 1.0
        now = time.monotonic()
        with self._lock:
            nxt = min(j.not_before for j in self._deferred)
        return max(0.05, min(1.0, nxt - now))

    def _promote_deferred(self) -> None:
        now = time.monotonic()
        with self._lock:
            ready = [j for j in self._deferred if j.not_before <= now]
            self._deferred = [j for j in self._deferred if j.not_before > now]
        for job in ready:
            try:
                self._handle(job)
            except Exception as exc:  # noqa: BLE001
                self.failed += 1
                self._note_failure("retry", exc, "ingest retry failed")

    def _handle(self, job: _Job) -> None:
        store = self._store_for(job.exp_dir)
        res = store.ingest(
            job.path,
            sample_dir_name=job.sample_dir_name,
            source=job.source,
            action_id=job.action_id,
            skill=job.skill,
            copy_mode=job.copy_mode,
            zone=job.zone,
            active_sample_dir=job.active_sample_dir,
        )

        if res.disposition == "retry":
            job.attempts += 1
            if job.attempts <= len(_RETRY_DELAYS):
                job.not_before = time.monotonic() + _RETRY_DELAYS[job.attempts - 1]
                with self._lock:
                    self._deferred.append(job)
                return
            self.failed += 1
            self.last_error = f"give up after {job.attempts} retries: {res.reason}"
            self._note_failure("retry_exhausted", None,
                               "ingest gave up on %s (%s)" % (job.path, res.reason))
            return

        if res.disposition == "new":
            self.done += 1
            self.bytes_copied += res.size_bytes
        elif res.disposition == "duplicate":
            self.duplicate += 1
        elif res.disposition == "inplace":
            self.done += 1
        elif res.disposition == "skipped":
            self.skipped += 1
        else:  # failed
            self.failed += 1
            self.last_error = res.reason
            self._note_failure("ingest:" + (res.reason or "unknown"), None,
                               "ingest failed for %s (%s)" % (job.path, res.reason))

        # 错归属：文件写在别的样品的原位目录里，但归属真源是活跃样品。
        # 副本已经落到活跃样品了，这里在【源所在样品】的台账补一行，
        # 让它不会静默地包含一个不属于它的文件。
        if res.zone is Zone.INPLACE_FOREIGN and job.stray_sample_dir_name:
            store.note_stray(job.stray_sample_dir_name, {
                "sha256": res.sha256, "size": res.size_bytes,
                "rel_path": _rel(job.path, job.exp_dir),
                "belongs_to_sample": job.sample_id or "",
                "belongs_to_rel_path": res.rel_path,
                "note": "Nanonis session path 未随样品切换而更新",
            })

        self._record_location(job, res)

        if self._on_result is not None:
            try:
                self._on_result(job, res)
            except Exception as exc:  # noqa: BLE001
                self._note_failure("on_result", exc, "ingest callback failed")

    def _record_location(self, job: _Job, res: IngestResult) -> None:
        """把"文件在哪"写进 ``file_locations``。

        **复制失败也要登记**（``root_kind="origin"`` + ``status="copy_failed"``）：
        记录必须活下来，哪怕副本没成 —— 那样最坏也只是退回到"只登记不搬运"的
        今日行为，零回归。
        """
        repo = getattr(self._repos, "file_locations", None)
        if repo is None or not res.sha256:
            return
        try:
            if res.disposition == "failed":
                repo.record(
                    sha256=res.sha256, experiment_id=job.experiment_id,
                    sample_id=job.sample_id, rel_path="", root_kind="origin",
                    origin_path=str(job.path), source=job.source,
                    action_id=job.action_id, size_bytes=res.size_bytes,
                    status="copy_failed",
                )
            elif res.ok and res.rel_path:
                repo.record(
                    sha256=res.sha256, experiment_id=job.experiment_id,
                    sample_id=job.sample_id, rel_path=res.rel_path,
                    root_kind="experiment_folder", origin_path=str(job.path),
                    source=("inplace" if res.disposition == "inplace" else job.source),
                    action_id=job.action_id, size_bytes=res.size_bytes,
                    status="ok",
                )
        except Exception as exc:  # noqa: BLE001 — DB 出问题不影响文件已经落好
            self._note_failure("file_locations", exc, "file_locations record failed")

    # ── 诊断 ──────────────────────────────────────────────────────────

    def _note_failure(self, key: str, exc: BaseException | None, msg: str) -> None:
        """首次 LOUD，之后至多每分钟一次（带累计次数）。

        照抄 trace_sink._note_failure 的纪律：吞异常是为了不让扫描崩，
        **从来不是为了安静**。
        """
        st = self._fail_log.setdefault(key, [0, 0.0])
        st[0] += 1
        now = time.monotonic()
        if st[0] == 1:
            st[1] = now
            logger.warning("%s — 实验数据归档正在丢数据%s", msg,
                           "" if exc is None else ": %r" % (exc,),
                           exc_info=exc is not None)
        elif now - st[1] >= _FAIL_QUIET_S:
            st[1] = now
            logger.warning("%s — 仍在失败（累计 %d 次）%s", msg, st[0],
                           "" if exc is None else ": %r" % (exc,))

    def stats(self) -> dict:
        with self._lock:
            deferred = len(self._deferred)
        return {
            "enabled": True,
            "queued": self.queued, "pending": self._q.qsize(), "retrying": deferred,
            "done": self.done, "duplicate": self.duplicate, "skipped": self.skipped,
            "failed": self.failed, "dropped": self.dropped,
            "bytes": self.bytes_copied, "last_error": self.last_error,
        }

    def flush(self, timeout: float | None = 10.0) -> bool:
        """等队列排空（仅测试/关机用）。返回是否排空。"""
        deadline = time.monotonic() + (timeout if timeout is not None else 1e9)
        while time.monotonic() < deadline:
            with self._lock:
                deferred = len(self._deferred)
            if self._q.unfinished_tasks == 0 and deferred == 0:
                return True
            time.sleep(0.05)
        return False

    def close(self) -> None:
        try:
            self._q.put_nowait(self._stop)
        except queue.Full:
            pass
        self._worker.join(timeout=5.0)


def _rel(p: Path, base: Path) -> str:
    try:
        return Path(p).relative_to(base).as_posix()
    except ValueError:
        return Path(p).as_posix()
