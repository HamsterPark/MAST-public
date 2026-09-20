"""Cross-session resume context (2026-07 analysis ⑥/⑦).

The supervisor used to answer 「No prior context or ongoing task found」 to "继续"
on a fresh thread — while a half-finished NiI2/Au(111) experiment sat `running` in
the record — and an operator's 心愿单 answer sat unread because the blocked agent
never polled for it. These pin the pure helpers that close both gaps:

  * is_resume_intent — recognises "继续"/"resume" so the route knows to inject.
  * build_experiment_resume_block — reconstructs the unfinished experiment.
  * build_request_reply_block — surfaces answered-but-undelivered requests.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from mast.agents._shared.resume_context import (
    build_experiment_resume_block,
    build_request_reply_block,
    is_resume_intent,
)
from mast.core.types import ActionRecord
from mast.logging.experiment_log import ExperimentLog
from mast.logging.storage import ExperimentStorage
from mast.wishlist import WishlistBoard


# ── is_resume_intent ─────────────────────────────────────────────────────────
class TestResumeIntent:
    def test_bare_continue_words_are_resume(self):
        for t in ("继续", "接着做", "继续之前的 NiI2 实验", "resume",
                  "continue where we left off", "keep going", "接下来"):
            assert is_resume_intent(t) is True, t

    def test_a_fresh_specific_task_is_not_resume(self):
        for t in ("扫描 Au(111) 5nm", "开始新实验 MoS2", "分析这张图的缺陷密度",
                  "start a Si(111)-7x7 experiment", "", "   "):
            assert is_resume_intent(t) is False, t

    def test_never_raises_on_odd_input(self):
        assert is_resume_intent(None) is False  # type: ignore[arg-type]


# ── build_experiment_resume_block ────────────────────────────────────────────
class TestExperimentResumeBlock:
    def _seed(self, tmp_path):
        st = ExperimentStorage(str(tmp_path / "exp.db"))
        log = ExperimentLog(st)
        log.start_experiment("NiI2 quality study", "characterize NiI2 monolayer")
        log.start_sample("NiI2 film A", sample_type="magnetic_spm")
        for skill in ("AutoApproach", "StartScan"):
            log.log_skill_execution(ActionRecord(skill_name=skill))
        return st, log

    def test_it_describes_the_unfinished_experiment(self, tmp_path):
        st, log = self._seed(tmp_path)
        block = build_experiment_resume_block(log, st)
        assert block is not None
        # experiment + sample + goal + recent actions are all surfaced
        assert "NiI2 quality study" in block
        assert "NiI2 film A" in block
        assert "characterize NiI2 monolayer" in block
        assert "StartScan" in block and "AutoApproach" in block
        assert "恢复上下文" in block
        # short experiment id prefix present
        assert log.current_experiment_id[:8] in block

    def test_none_when_no_unfinished_experiment(self, tmp_path):
        st = ExperimentStorage(str(tmp_path / "exp.db"))
        log = ExperimentLog(st)
        assert build_experiment_resume_block(log, st) is None

    def test_cleared_scope_is_not_resumed(self, tmp_path):
        """Stepping away from an experiment means "继续" has nothing to resume.

        REVISED 2026-07-28: this used to be ``test_ended_experiment_is_not_resumed``
        and relied on ``end_experiment`` marking the row 'completed'. Experiments
        no longer have a terminal state; what ``end_experiment`` does now is clear
        the scope pointer, and THAT is what must suppress the resume block.
        """
        st, log = self._seed(tmp_path)
        log.end_experiment()          # 现在 = 清指针，不写终态
        assert build_experiment_resume_block(ExperimentLog(st), st) is None

    def test_falls_back_to_persisted_pointer_on_disk(self, tmp_path):
        """A brand-new log in a new session resumes via the persisted pointer.

        Previously this leaned on scanning for a ``status == 'running'`` row.
        The pointer is now explicit, so a fresh session picks up exactly the
        experiment the operator was on — not merely the newest unfinished-looking
        row.
        """
        st, log = self._seed(tmp_path)
        eid = log.current_experiment_id
        block = build_experiment_resume_block(ExperimentLog(st), st)
        assert block is not None and eid[:8] in block

    def test_active_plan_phase_is_surfaced(self, tmp_path):
        st, log = self._seed(tmp_path)

        class _Phase:
            def __init__(self, name):
                self.name = name

        class _Plan:
            name = "overnight NiI2"
            current_phase_idx = 1
            phases = [_Phase("approach"), _Phase("scan"), _Phase("sts")]
            status = "running"

        class _PS:
            def get_active(self):
                return _Plan()

        block = build_experiment_resume_block(log, st, _PS())
        assert "overnight NiI2" in block
        assert "2/3" in block and "scan" in block

    def test_best_effort_never_raises(self, tmp_path):
        # a plan store that explodes must not sink the whole block
        class _Boom:
            def get_active(self):
                raise RuntimeError("boom")

        st, log = self._seed(tmp_path)
        block = build_experiment_resume_block(log, st, _Boom())
        assert block is not None and "NiI2 quality study" in block


# ── build_request_reply_block ────────────────────────────────────────────────
class TestRequestReplyBlock:
    def _board(self, tmp_path):
        return WishlistBoard(board_dir=str(tmp_path / "wl"))

    def test_answered_request_is_surfaced_with_path(self, tmp_path):
        b = self._board(tmp_path)
        rec = b.post_agent_request("data_processing", "请提供 Au111 那张图的路径", kind="info")
        rid = rec["id"]
        b.resolve_agent_request(rid, "done", path=r"D:\Data\Au111_001.sxm", note="就是这张")

        block, ids = build_request_reply_block(b)
        assert block is not None
        assert rid in block and "已完成" in block
        assert r"D:\Data\Au111_001.sxm" in block
        assert "就是这张" in block
        assert ids == [rid]

    def test_nothing_undelivered_returns_none(self, tmp_path):
        b = self._board(tmp_path)
        assert build_request_reply_block(b) == (None, [])

    def test_delivered_answers_are_not_re_surfaced(self, tmp_path):
        b = self._board(tmp_path)
        rid = b.post_agent_request("dp", "路径?", kind="info")["id"]
        b.resolve_agent_request(rid, "done", path="D:\\x.sxm")
        _block, ids = build_request_reply_block(b)
        b.mark_delivered(ids)                      # caller hands over once
        assert build_request_reply_block(b) == (None, [])

    def test_a_dismissed_answer_is_still_surfaced(self, tmp_path):
        b = self._board(tmp_path)
        rid = b.post_agent_request("lit", "上传 PDF", kind="upload")["id"]
        b.resolve_agent_request(rid, "dismissed", note="这台机器没有")
        block, ids = build_request_reply_block(b)
        assert block is not None and "已忽略" in block and "这台机器没有" in block

    def test_ancient_answer_is_filtered_by_the_time_window(self, tmp_path):
        b = self._board(tmp_path)
        rid = b.post_agent_request("dp", "路径?", kind="info")["id"]
        b.resolve_agent_request(rid, "done", path="D:\\old.sxm")
        # backdate the answer well beyond the default 72h window
        old = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat(timespec="seconds")
        b._requests[0]["resolved_at"] = old
        assert build_request_reply_block(b, within_hours=72) == (None, [])
        # but with no window it is still surfaced (fail-open)
        block, _ids = build_request_reply_block(b, within_hours=None)
        assert block is not None and "D:\\old.sxm" in block


if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
