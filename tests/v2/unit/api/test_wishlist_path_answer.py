"""The operator can hand the agent a PATH, and the agent can read it back .

    心愿单备注中提供的路径 agent 读不到；心愿单里也没有让用户显式提供路径的字段。

Both halves were broken, and the second one is the worse of the two:

  * **No field.** The agent's single most common ask is *"where is the file?"* and
    the only place to answer was a free-text note. A path typed into prose is not a
    channel — the agent had to guess it out of a sentence.
  * **No way back.** There was NO agent tool that could read the wishlist. The only
    delivery was an interjection relay that fires solely if a run happens to be
    STREAMING at the moment the operator clicks resolve — and the agent asked
    precisely BECAUSE it was blocked and had stopped. So the normal case was: the
    operator answers, the answer lands on the board, and no agent ever sees it.
    Meanwhile ``request_user_action``'s own docstring promised
    「你可在后续轮次得知结果」 — a promise the system could not keep.

So: ``path`` is a first-class field, the answer is persisted with an undelivered
flag, and ``check_my_requests`` hands it back on the agent's next turn — whenever
that is, with or without a live run.
"""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mast.api.context import AppContext
from mast.api.routes.wishlist import router
from mast.wishlist import get_board, reset_default_board


@pytest.fixture()
def client(tmp_path, monkeypatch) -> TestClient:
    monkeypatch.setenv("MAST2_PROJECT_ROOT", str(tmp_path))
    reset_default_board()
    app = FastAPI()
    app.state.ctx = AppContext()
    app.include_router(router, prefix="/api")
    yield TestClient(app)
    reset_default_board()


def _ask(agent_id: str = "data_processing") -> str:
    from mast.wishlist import post_agent_request

    rec = post_agent_request(
        agent_id, "[数据处理] 请提供 Au111 那张图的完整路径", kind="info")
    return rec["id"]


# ════════════════════════════════════════════════════════════════════════
# The loop that has to close: operator answers → the AGENT reads it
# ════════════════════════════════════════════════════════════════════════

class TestTheAnswerReachesTheAgent:
    def test_a_path_answered_with_no_run_live_is_read_back_next_turn(self, client):
        """THE case: the agent asked because it was blocked, so it is NOT running
        when the operator answers. The old relay could only deliver into a live
        stream; with no run, the answer sat on the board forever."""
        from mast.agents._shared.request_tools import make_request_tools

        rid = _ask("data_processing")
        r = client.post(f"/api/wishlist/requests/{rid}/resolve", json={
            "action": "done",
            "path": r"D:\Data\fsSTM\Au111_mica_001.sxm",
            "note": "就是这张",
        }).json()
        assert r["ok"] is True
        # no live run — and the operator is told the truth about what happens next
        assert "check_my_requests" in r["message"]

        tools = {t.name: t for t in make_request_tools(
            lambda: {"agent_id": "data_processing"})}
        out = tools["check_my_requests"].invoke({})

        assert r"D:\Data\fsSTM\Au111_mica_001.sxm" in out, (
            "the agent cannot read the path the operator gave it — which is the "
            "entire complaint")
        assert "就是这张" in out

    def test_an_answer_is_handed_over_once_not_forever(self, client):
        from mast.agents._shared.request_tools import make_request_tools

        rid = _ask()
        client.post(f"/api/wishlist/requests/{rid}/resolve",
                    json={"action": "done", "path": "D:\\x.sxm"})
        tools = {t.name: t for t in make_request_tools(
            lambda: {"agent_id": "data_processing"})}

        assert "D:\\x.sxm" in tools["check_my_requests"].invoke({})
        again = tools["check_my_requests"].invoke({})
        assert "D:\\x.sxm" not in again, "the same answer is served on every turn"
        assert "暂无新的用户答复" in again

    def test_an_agent_only_sees_its_own_answers(self, client):
        from mast.agents._shared.request_tools import make_request_tools

        rid = _ask("literature")
        client.post(f"/api/wishlist/requests/{rid}/resolve",
                    json={"action": "done", "path": "D:\\paper.pdf"})

        dp = {t.name: t for t in make_request_tools(
            lambda: {"agent_id": "data_processing"})}
        assert "paper.pdf" not in dp["check_my_requests"].invoke({})

        lit = {t.name: t for t in make_request_tools(
            lambda: {"agent_id": "literature"})}
        assert "paper.pdf" in lit["check_my_requests"].invoke({})

    def test_nothing_answered_yet_says_so_and_says_not_to_re_ask(self, client):
        from mast.agents._shared.request_tools import make_request_tools

        _ask()
        tools = {t.name: t for t in make_request_tools(
            lambda: {"agent_id": "data_processing"})}
        out = tools["check_my_requests"].invoke({})
        assert "暂无新的用户答复" in out
        assert "不要重复发起同一请求" in out, (
            "an agent told only 'nothing yet' will just ask again — which is a spin")


# ════════════════════════════════════════════════════════════════════════
# The field itself
# ════════════════════════════════════════════════════════════════════════

class TestThePathIsAField:
    def test_it_is_persisted_and_surfaced_on_the_row(self, client):
        rid = _ask()
        client.post(f"/api/wishlist/requests/{rid}/resolve",
                    json={"action": "done", "path": "D:\\Data\\a.sxm", "note": "n"})
        board = client.get("/api/wishlist").json()
        row = next(r for r in board["requests"] if r["id"] == rid)
        assert row["path"] == "D:\\Data\\a.sxm", (
            "the path is not on the record — it was only ever prose")
        assert row["note"] == "n"

    def test_the_tool_exists_on_every_agent_that_can_ask(self):
        """``request_user_action`` promises 「你可在后续轮次得知结果」. A promise with
        no tool behind it is a lie the agent repeats to the operator."""
        from mast.agents._shared.request_tools import make_request_tools

        names = {t.name for t in make_request_tools(lambda: {"agent_id": "x"})}
        assert {"request_user_action", "report_upgrade_idea",
                "check_my_requests"} <= names

    def test_a_dismissed_request_is_still_readable(self, client):
        """"已忽略" is an answer too — the agent must stop waiting on it."""
        from mast.agents._shared.request_tools import make_request_tools

        rid = _ask()
        client.post(f"/api/wishlist/requests/{rid}/resolve",
                    json={"action": "dismissed", "note": "这台机器上没有这个文件"})
        tools = {t.name: t for t in make_request_tools(
            lambda: {"agent_id": "data_processing"})}
        out = tools["check_my_requests"].invoke({})
        assert "已忽略" in out and "没有这个文件" in out
