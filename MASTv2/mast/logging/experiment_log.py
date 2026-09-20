"""High-level interface for structured experiment logging.

P (Ported) from v1 mast/logging/experiment_log.py. Module-level active log
singleton (get_active_log / set_active_log) lets skills (StartScan) build
meaningful sxm filenames without coupling to GUI state.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import numpy as np

from mast.core.types import ActionRecord
from mast.logging.scope import ScopeChange, notify as _notify_scope
from mast.logging.storage import ExperimentStorage

logger = logging.getLogger(__name__)


# Module-level pointer to the active ExperimentLog singleton, set by
# whoever creates it (mast.webui.app). Lets low-level skills (e.g. StartScan)
# read the active experiment / sample name to build a meaningful sxm
# basename instead of leaving Nanonis to fall back to "unnamed####".
_ACTIVE_LOG: "ExperimentLog | None" = None


def get_active_log() -> "ExperimentLog | None":
    """Return the currently registered ExperimentLog, or None.

    Skills must tolerate None — e.g. when running headless / in tests.
    """
    return _ACTIVE_LOG


def set_active_log(log: "ExperimentLog | None") -> None:
    """Register (or unregister) the active ExperimentLog singleton."""
    global _ACTIVE_LOG
    _ACTIVE_LOG = log


def _clear_scan_map_plan() -> None:
    """Clear the scan-map planning overlay on an experiment/sample transition so
    a plan drawn for the previous scope doesn't ghost onto the new one (review
    #7/#9: 换样品 → fresh canvas). Best-effort — the overlay is a GUI-side
    convenience; logging must not hard-depend on it."""
    try:
        from mast.io.plan_overlay import get_plan_overlay
        get_plan_overlay().clear()
    except Exception:  # noqa: BLE001
        pass


class ExperimentLog:
    """High-level interface for structured experiment logging.

    Wraps ExperimentStorage with session-aware convenience methods.
    Tracks current experiment and current sample.
    """

    def __init__(self, storage: ExperimentStorage):
        self._storage = storage
        # 作用域是**一个 tuple**，不是两个属性。两个属性分别赋值会让并发读者
        # 看到半更新的「新实验 + 旧样品」对 —— 而 chat / 群聊 / 后台 run 三条
        # 线程都在读它。在此之前这里完全没有锁。
        self._scope: tuple[str | None, str | None] = (None, None)
        self._lock = threading.RLock()
        # Whether the LAST start_experiment / start_sample attached to an
        # existing open row (idempotent reuse) rather than creating a new one.
        # The agent-facing tool reads these to tell the model "reused, not
        # duplicated" — the field saw one NiI2 study spawn 6 experiments across
        # session restarts because every start minted a fresh row.
        self._last_start_reused: bool = False
        self._last_sample_reused: bool = False

    # ── 作用域指针 ────────────────────────────────────────────────
    #
    # 读取端保持 ``current_experiment_id`` / ``current_sample_id`` 这两个属性名：
    # 全仓库有十几处在读它们（imaging 的 sxm 文件名、vision 导入、对话打标、
    # MapActivityWatcher 的基线重置……）。它们现在从 ``_scope`` 派生。

    @property
    def current_experiment_id(self) -> str | None:
        return self._scope[0]

    @property
    def current_sample_id(self) -> str | None:
        return self._scope[1]

    # 旧代码（含测试）直接给 ``_current_experiment_id`` 赋值。保留这两个可写
    # 别名，让它们落到同一个 tuple 上，避免出现两份不一致的真相。
    @property
    def _current_experiment_id(self) -> str | None:
        return self._scope[0]

    @_current_experiment_id.setter
    def _current_experiment_id(self, value: str | None) -> None:
        with self._lock:
            self._scope = (value, self._scope[1])

    @property
    def _current_sample_id(self) -> str | None:
        return self._scope[1]

    @_current_sample_id.setter
    def _current_sample_id(self, value: str | None) -> None:
        with self._lock:
            self._scope = (self._scope[0], value)

    def _set_scope(self, experiment_id: str | None, sample_id: str | None,
                   *, source: str = "", note: str = "", persist: bool = True) -> None:
        """原子地移动指针（内存 + DB 一起）。"""
        with self._lock:
            self._scope = (experiment_id, sample_id)
        if persist:
            try:
                self._storage.set_active_scope(experiment_id, sample_id,
                                               updated_by=source, note=note)
            except Exception as exc:  # noqa: BLE001 — 指针持久化失败不该让切换崩
                logger.warning("active_scope persist failed: %r", exc)

    # ── Experiment lifecycle ──────────────────────────────────────

    def _find_reusable_experiment(self, name: str) -> str | None:
        """同名实验的 id（当前的优先），没有则 None。

        2026-07-28：**去掉了 ``status == 'running'`` 这个条件**。实验没有终态，
        一条三个月前的同名实验和一条今天的一样可以被继续做 —— 场景是
        「有的实验可能过了十年重启」。带 status 条件的旧行为会在旧实验被标成
        completed/superseded 之后凭空造出一条重复实验。
        """
        target = (name or "").strip().casefold()
        if not target:
            return None
        cur_id = self._scope[0]
        if cur_id is not None:
            cur = self._storage.get_experiment(cur_id)
            if cur and (cur.get("name") or "").strip().casefold() == target:
                return cur_id
        found = self._storage.find_experiment_by_name(name)
        return found["id"] if found else None

    # ── 切换原语（2026-07-28） ────────────────────────────────────
    #
    # 切换 = 只改指针。绝不写 status，绝不写 end_time，绝不碰任何实验/样品行
    # 的内容。参见 docs/v2/design/experiment_folder_persistence.md §10。

    def _snapshot(self) -> ScopeChange:
        eid, sid = self._scope
        exp = self._storage.get_experiment(eid) if eid else None
        smp = self._storage.get_sample(sid) if sid else None
        return ScopeChange(
            ok=True, experiment_id=eid, sample_id=sid,
            experiment_name=(exp or {}).get("name", "") or "",
            sample_name=(smp or {}).get("name", "") or "",
        )

    def switch_experiment(self, experiment_id: str, *, sample_id: str | None = None,
                          source: str = "gui", reason: str = "") -> ScopeChange:
        """把当前实验切到 *experiment_id*。**永不抛异常。**

        样品的落点：显式指定的（必须属于该实验）→ 该实验上次用过的样品 → None。
        落在"上次在这个实验里用的那块样品"是回到旧项目时最符合直觉的行为。
        """
        exp = self._storage.get_experiment(experiment_id)
        if not exp:
            return ScopeChange(
                ok=False, block_code="unknown_experiment",
                error="实验不存在（记录库可能被替换过）。请在列表里选一个现有实验。")

        if sample_id:
            smp = self._storage.get_sample(sample_id)
            if not smp:
                return ScopeChange(ok=False, block_code="unknown_sample",
                                   error="样品不存在。")
            if smp.get("experiment_id") != experiment_id:
                return ScopeChange(
                    ok=False, block_code="sample_mismatch",
                    error="该样品不属于这个实验。")
            target_sample = sample_id
        else:
            last = self._storage.get_last_active_sample(experiment_id)
            target_sample = last["id"] if last else None

        old = self._snapshot()
        if self._scope == (experiment_id, target_sample):
            # 幂等空转：不清 overlay、不发事件、不写 DB。双击和模型重试免费。
            return ScopeChange(
                ok=True, changed=False, experiment_id=experiment_id,
                sample_id=target_sample, experiment_name=exp.get("name", "") or "",
                sample_name=old.sample_name)

        self._set_scope(experiment_id, target_sample, source=source, note=reason)
        _clear_scan_map_plan()   # 新 scope → 新画布
        new = self._snapshot()
        _notify_scope(old, new)
        return ScopeChange(
            ok=True, changed=True, experiment_id=experiment_id,
            sample_id=target_sample, experiment_name=new.experiment_name,
            sample_name=new.sample_name)

    def switch_sample(self, sample_id: str, *, source: str = "gui",
                      reason: str = "") -> ScopeChange:
        """切换当前样品。跨实验切样品会一并把实验也切过去。"""
        smp = self._storage.get_sample(sample_id)
        if not smp:
            return ScopeChange(ok=False, block_code="unknown_sample",
                               error="样品不存在。")
        eid = smp.get("experiment_id")
        if eid and eid != self._scope[0]:
            return self.switch_experiment(eid, sample_id=sample_id,
                                          source=source, reason=reason)
        if self._scope[1] == sample_id:
            return ScopeChange(ok=True, changed=False, experiment_id=self._scope[0],
                               sample_id=sample_id,
                               sample_name=smp.get("name", "") or "")
        old = self._snapshot()
        self._set_scope(self._scope[0], sample_id, source=source, note=reason)
        _clear_scan_map_plan()   # 换样品 → 新画布
        new = self._snapshot()
        _notify_scope(old, new)
        return ScopeChange(ok=True, changed=True, experiment_id=new.experiment_id,
                           sample_id=sample_id, experiment_name=new.experiment_name,
                           sample_name=new.sample_name)

    def clear_sample(self, *, source: str = "gui", reason: str = "") -> ScopeChange:
        """取消选中样品（**物理出样时用**）。

        这是「结束样品」唯一还剩的真实用例：样品被从腔体里取下来了、还没装新的，
        用户不希望下一次扫描被误归到它名下。注意它**不写样品行的任何字段** ——
        样品仍然可以随时被切回来继续用。
        """
        if self._scope[1] is None:
            return ScopeChange(ok=True, changed=False, experiment_id=self._scope[0])
        old = self._snapshot()
        self._set_scope(self._scope[0], None, source=source, note=reason)
        _clear_scan_map_plan()
        new = self._snapshot()
        _notify_scope(old, new)
        return ScopeChange(ok=True, changed=True, experiment_id=new.experiment_id,
                           sample_id=None, experiment_name=new.experiment_name)

    def restore_scope(self) -> ScopeChange:
        """启动时从持久指针恢复作用域。**永不抛。**

        校验链：实验行还在吗 → 样品行还在、且属于该实验吗。任何一环断了就
        降级并**写回校正后的指针**（自愈：下次启动就干净了）。

        指针为空（升级后首启）→ 播种：取 ``status='running'`` 里最新的一条
        （**全仓库最后一处读 status**，纯粹当升级提示用），没有就取最新建的。
        对用户而言升级是无感的：他上次在哪就还在哪。
        """
        try:
            row = self._storage.get_active_scope()
        except Exception as exc:  # noqa: BLE001
            logger.warning("active_scope read failed: %r", exc)
            row = None

        eid = (row or {}).get("experiment_id")
        sid = (row or {}).get("sample_id")
        seeded = False

        if not eid:
            eid, sid = self._seed_scope()
            seeded = eid is not None

        if not eid:
            self._scope = (None, None)
            return ScopeChange(ok=True, changed=False)

        exp = self._storage.get_experiment(eid)
        if not exp:
            logger.warning("active_scope points at a missing experiment %s — reseeding", eid)
            eid, sid = self._seed_scope()
            if not eid:
                self._set_scope(None, None, source="restore", note="dangling pointer")
                return ScopeChange(
                    ok=True, changed=True,
                    error="上次的实验记录已找不到（记录库可能被替换），请选择或新建一个实验。")
            exp = self._storage.get_experiment(eid)
            seeded = True

        if sid:
            smp = self._storage.get_sample(sid)
            if not smp or smp.get("experiment_id") != eid:
                # 样品丢了或不属于这个实验 → 只降样品，实验保留。
                # 门控随即生效并要求用户选样品 —— 这是正确行为。
                logger.warning("active_scope sample %s invalid for experiment %s — dropping",
                               sid, eid)
                sid = None

        self._set_scope(eid, sid, source="restore",
                        note="seeded" if seeded else "restored")
        return ScopeChange(
            ok=True, changed=True, experiment_id=eid, sample_id=sid,
            experiment_name=(exp or {}).get("name", "") or "",
            sample_name=((self._storage.get_sample(sid) or {}).get("name", "") if sid else ""))

    def _seed_scope(self) -> tuple[str | None, str | None]:
        """升级路径：从旧库猜一次当前作用域。**全仓库最后一处读 status。**"""
        try:
            rows = self._storage.list_experiments(limit=50)
        except Exception:  # noqa: BLE001
            return None, None
        pick = next((r for r in rows if (r.get("status") or "") == "running"), None)
        if pick is None:
            pick = rows[0] if rows else None
        if pick is None:
            return None, None
        eid = pick["id"]
        last = self._storage.get_last_active_sample(eid)
        return eid, (last["id"] if last else None)

    def resume_experiment(self, experiment_id: str) -> bool:
        """Attach to an EXISTING experiment (and its active sample) WITHOUT
        creating a new row. Returns True if attached, False if the id is unknown.

        The resume primitive: on a session restart / re-dispatch, point the log
        at the run already in progress so the agent continues it instead of
        minting a duplicate. (The runtime's startup auto-restore does the same
        inline; this is the reusable, tested API for that behaviour.)

        2026-07-28：现在它就是 :meth:`switch_experiment` 的一层薄壳 —— resume
        本来就是"切换"的雏形，只是从前没有被 HTTP 层暴露出来。"""
        return self.switch_experiment(experiment_id, source="restore").ok

    def start_experiment(self, name: str, goal: str = "", *,
                         reuse_open: bool = False) -> str:
        """Start a new experiment session. Returns experiment_id.

        Ends any currently-running experiment (and its active sample) FIRST, so
        starting a new one doesn't leave the previous experiment stuck in the
        'running' state forever — every prior experiment was orphaned as active. Mirrors start_sample's end-prior behaviour.

        ``reuse_open`` (opt-in, default False → unchanged create-always for every
        existing caller): when True, a start whose *name* matches an existing
        experiment ATTACHES to it (restoring its last-used sample) instead of
        creating a duplicate. The agent path passes True; an explicit
        'new experiment' user action leaves it False. Sets
        ``self._last_start_reused`` so the caller can report which happened.

        2026-07-28：**不再 supersede 上一个实验**。切走一个实验不是结束它 ——
        实验永久存在，随时可以切回来继续做。旧行为把「哪个是当前」编码进
        ``status``，那正是陈旧 active 行和前端两处推断打架的根因。
        """
        self._last_start_reused = False
        if reuse_open:
            reuse_id = self._find_reusable_experiment(name)
            if reuse_id is not None:
                # Continue the existing run — 不 supersede、不新建，保留扫描地图
                # （同一 scope）。恢复它上次用过的样品。
                last = self._storage.get_last_active_sample(reuse_id)
                self._set_scope(reuse_id, last["id"] if last else None,
                                source="agent", note="reuse_open")
                self._last_start_reused = True
                return reuse_id
        old = self._snapshot()
        new_id = self._storage.create_experiment(name, goal)
        self._set_scope(new_id, None, source="gui", note="new experiment")
        _clear_scan_map_plan()  # new scope → fresh map canvas
        _notify_scope(old, self._snapshot())
        return new_id

    def end_experiment(self, status: str = "completed") -> None:
        """DEPRECATED (2026-07-28)：实验没有「结束」这个动作。

        现在它只做一件事：**如果这个实验正是当前作用域，就把指针清掉**。它不再
        写 ``end_time``/``status`` —— 用户明确否决了归档：「没必要做归档。有的
        实验可能过了十年重启。何必归档呢？如果说要给一个实验写总结，写报告，
        并不必以归档为前提。」

        保留这个方法只为不破坏零散调用点；agent 工具层已经把它移除了（在
        "什么都不会结束"的模型下留一个叫 end_experiment 的工具是主动的危害 ——
        模型会调用它并相信自己干了什么）。
        """
        if self._scope[0] is None:
            raise RuntimeError("No active experiment to end.")
        old = self._snapshot()
        self._set_scope(None, None, source="gui", note=f"end_experiment({status})")
        _notify_scope(old, self._snapshot())

    # ── Sample lifecycle ──────────────────────────────────────────

    def start_sample(
        self,
        name: str,
        description: str = "",
        sample_type: str = "",
        sample_subtype: str = "",
        *,
        reuse_active: bool = False,
    ) -> str:
        """Start a new sample under the current experiment (ends the prior one).

        ``reuse_active`` (opt-in, default False → unchanged for every existing
        caller): when True, a start whose *name* matches an EXISTING sample of
        this experiment attaches to it instead of creating a duplicate. The agent
        path passes True. Sets ``self._last_sample_reused``.

        2026-07-28：**不再 end 上一个样品**。换样品不是结束上一个 —— STM 样品
        会被反复换回来用，旧行为让"切回上周那块样品"变得不可能。
        """
        self._last_sample_reused = False
        eid = self._scope[0]
        if eid is None:
            raise RuntimeError("No active experiment — start one before adding a sample.")
        if reuse_active:
            # 先看当前样品，再看整个实验下的同名样品（去掉了 status=='active'
            # 条件：被旧代码 end 成 completed 的样品同样应该被复用）。
            cur_id = self._scope[1]
            if cur_id is not None:
                cur = self._storage.get_sample(cur_id)
                if (cur and (cur.get("name") or "").strip().casefold()
                        == (name or "").strip().casefold()):
                    self._last_sample_reused = True
                    return cur_id
            found = self._storage.find_sample_by_name(eid, name)
            if found:
                self._set_scope(eid, found["id"], source="agent", note="reuse_active")
                self._last_sample_reused = True
                return found["id"]
        old = self._snapshot()
        sid = self._storage.create_sample(
            eid, name, description,
            sample_type=sample_type, sample_subtype=sample_subtype,
        )
        self._set_scope(eid, sid, source="gui", note="new sample")
        _clear_scan_map_plan()  # 换样品 → fresh map canvas
        _notify_scope(old, self._snapshot())
        return sid

    def end_sample(self, status: str = "completed") -> None:
        """DEPRECATED (2026-07-28)：样品没有「结束」这个动作。

        等价于 :meth:`clear_sample` —— 只把指针里的样品清掉（物理出样场景），
        不写样品行的任何字段。样品随时可以被切回来继续用。
        """
        if self._scope[1] is None:
            raise RuntimeError("No active sample to end.")
        self.clear_sample(source="gui", reason=f"end_sample({status})")

    # ── Rename (MAST records names — NOT Nanonis fields) ───────────

    def rename_experiment(self, name: str, experiment_id: str | None = None) -> bool:
        """Rename an experiment (defaults to the current one)."""
        eid = experiment_id or self._current_experiment_id
        if eid is None:
            raise RuntimeError("No experiment_id provided and no active experiment.")
        return self._storage.rename_experiment(eid, name)

    def rename_sample(self, name: str, sample_id: str | None = None) -> bool:
        """Rename a sample (defaults to the current one)."""
        sid = sample_id or self._current_sample_id
        if sid is None:
            raise RuntimeError("No sample_id provided and no active sample.")
        return self._storage.rename_sample(sid, name)

    # ── Action logging ────────────────────────────────────────────

    def log_skill_execution(self, record: ActionRecord) -> None:
        """把一条动作记进当前作用域。

        **仅在空时填充**（2026-07-28，原为无条件覆盖）：如果调用方在动作**开始
        时**就已经把作用域写进了 record，这里不再覆盖它。这样一次长扫描永远记在
        它**开始时**的样品名下，而不是它结束时碰巧是哪个样品 —— 用户在扫描
        途中切换样品不会再让这条记录归错样品。
        """
        eid, sid = self._scope
        if not record.experiment_id and eid:
            record.experiment_id = eid
        if not record.sample_id and sid:
            record.sample_id = sid
        self._storage.log_action(record)

    def get_history(self, experiment_id: str | None = None) -> list[ActionRecord]:
        eid = experiment_id or self._current_experiment_id
        if eid is None:
            raise RuntimeError(
                "No experiment_id provided and no active experiment."
            )
        return self._storage.get_actions(eid)

    # ── Training data collection ─────────────────────────────

    def record_training_sample(
        self, image: np.ndarray, label: str, skill_name: str,
    ) -> None:
        try:
            db_path = getattr(self._storage, '_db_path', None) or getattr(self._storage, 'db_path', '')
            base_dir = Path(str(db_path)).parent if db_path else Path(".")
            training_dir = base_dir / "training_data" / label
            training_dir.mkdir(parents=True, exist_ok=True)

            import uuid
            filename = f"{skill_name}_{uuid.uuid4().hex[:8]}.npy"
            np.save(str(training_dir / filename), image)
            logger.debug("Training sample saved: %s/%s", label, filename)
        except Exception as exc:
            logger.debug("Failed to save training sample: %s", exc)
