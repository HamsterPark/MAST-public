"""Draft/review persistence — rewritten
2026-07-29 for the document store.

The full-flow run on 2026-07-10 finished with ACCEPT but no manuscript file
anywhere: paper_writing had no save tool (the draft lived only in the
conversation) and paper_review looked for drafts under a wrong, install-dir
derived path (C:\\MAST\\data\\drafts). The original fix put both under a shared
``data/drafts`` / ``data/reviews``, versioned by the LLM-supplied title.

That fix had two holes of its own, which this file now pins shut:

  * **the title was the family key** — rewording it forked one manuscript into two
    histories, and two unrelated experiments choosing the same title merged into
    one. Identity is now ``doc_id``; the title is a display name.
  * **a review was named after the draft's file stem**, which contained the
    draft's version number, so every new draft version started a new review
    family. A review now records ``target_doc_id`` explicitly.

Still pinned from the original: nothing is ever overwritten, and what save_draft
writes is what load_draft reads back.
"""
from __future__ import annotations

import pytest

from tests.v2.toolcall import tool_call, tool_text
from mast.agents._shared.data_paths import (
    drafts_dir,
    next_version_path,
    reviews_dir,
)
from mast.agents.paper_review.tools import load_draft, save_review
from mast.agents.paper_writing.tools import load_review, save_draft


@pytest.fixture()
def draft_env(tmp_path, monkeypatch, documents_root):
    """Documents land in a disposable experiment root; the legacy dirs are
    redirected too so the read-only fallback cannot see the operator's files.

    ``documents_root`` is not optional decoration — the experiment folder is
    resolved from ``MAST_EXPERIMENT_ROOT``, which neither ``MAST2_PROJECT_ROOT``
    nor ``MAST_DRAFTS_DIR`` reaches. Without it these tests write real documents
    into the operator's data root (11 junk directories, first run).
    """
    d = tmp_path / "drafts"
    r = tmp_path / "reviews"
    monkeypatch.setenv("MAST_DRAFTS_DIR", str(d))
    monkeypatch.setenv("MAST_REVIEWS_DIR", str(r))
    return d, r


def _doc_id(save_output) -> str:
    """The doc_id out of a save_draft/save_review return.

    Parsed rather than hard-coded because the whole point of the return text is
    that the MODEL can find the id in it — if the shape drifts so that a regex
    over "doc_id = X" stops matching, the agent has stopped being told.

    ``str()`` because since 2026-07-29 these tools return an
    ``ArtifactToolReturn`` (a ``Command`` subclass that also writes the doc
    pointer into MASTState) rather than a bare string. ``str(x)`` / ``"y" in x``
    still see the same human summary; only a real regex needs the cast."""
    import re
    text = str(save_output)
    m = re.search(r"doc_id = (\S+)", text)
    assert m, f"save output does not name a doc_id the model can reuse:\n{text}"
    return m.group(1)


def _all_versions(documents_root):
    """Every ``vNNN.md`` under the disposable root, as ``<doc-dir>/<file>``."""
    return sorted(f"{p.parent.name}/{p.name}"
                  for p in documents_root.rglob("v[0-9][0-9][0-9].md"))


# ════════════════════════════════════════════════════════════════════════
# The legacy path resolver — still the answer for migration/compat reads
# ════════════════════════════════════════════════════════════════════════

class TestDataPaths:
    def test_env_override_wins(self, draft_env):
        d, r = draft_env
        assert drafts_dir() == d
        assert reviews_dir() == r

    def test_project_root_fallback(self, monkeypatch, tmp_path):
        monkeypatch.delenv("MAST_DRAFTS_DIR", raising=False)
        monkeypatch.setenv("MAST2_PROJECT_ROOT", str(tmp_path))
        assert drafts_dir() == tmp_path / "data" / "drafts"

    def test_next_version_increments(self, tmp_path):
        p1 = next_version_path(tmp_path, "report", ".md")
        assert p1.name == "report_v001.md"
        p1.write_text("x", encoding="utf-8")
        p2 = next_version_path(tmp_path, "report", ".md")
        assert p2.name == "report_v002.md"

    def test_next_version_strips_caller_suffix(self, tmp_path):
        p = next_version_path(tmp_path, "report_v2", ".md")
        assert p.name == "report_v001.md"  # not report_v2_v001.md

    def test_slug_keeps_cjk(self, tmp_path):
        p = next_version_path(tmp_path, "Au111 实验报告", ".md")
        assert "实验报告" in p.name

    def test_it_is_documented_as_off_limits_to_new_code(self):
        """It is a check-then-act race (glob for the next number, write later, no
        lock and no O_EXCL in between) with at least four concurrent writers. It
        stays for migration, so the docstring is the only thing stopping the next
        caller."""
        doc = next_version_path.__doc__ or ""
        assert "MUST NOT CALL THIS" in doc
        assert "mast.documents.store" in doc


# ════════════════════════════════════════════════════════════════════════
# save_draft — a document in the experiment folder, versioned by doc_id
# ════════════════════════════════════════════════════════════════════════

class TestSaveDraft:
    def test_same_doc_id_is_a_new_version(self, draft_env, documents_root):
        out1 = save_draft.invoke(tool_call(save_draft, {"title": "Au111 report",
                                  "markdown_text": "# T\nbody"}))
        did = _doc_id(out1)
        assert "v1" in out1

        out2 = save_draft.invoke(tool_call(save_draft, {"title": "Au111 report", "doc_id": did,
                                  "markdown_text": "# T\nbody rev2"}))
        assert _doc_id(out2) == did, "passing the doc_id back forked the family"
        assert "v2" in out2

        # One document directory, two version files — never an overwrite.
        files = _all_versions(documents_root)
        assert len(files) == 2
        assert len({f.split("/")[0] for f in files}) == 1
        assert sorted(f.split("/")[1] for f in files) == ["v001.md", "v002.md"]

    def test_no_doc_id_means_a_NEW_document_even_with_the_same_title(
            self, draft_env, documents_root):
        """The chosen failure mode. Proliferation is recoverable (the operator can
        merge or delete); merging two experiments' reports into one version history
        is not. So identity is never inferred from the title."""
        a = _doc_id(save_draft.invoke(tool_call(save_draft, {"title": "同一个名字",
                                       "markdown_text": "第一份报告"})))
        b = _doc_id(save_draft.invoke(tool_call(save_draft, {"title": "同一个名字",
                                       "markdown_text": "另一份完全无关的报告"})))
        assert a != b
        dirs = {f.split("/")[0] for f in _all_versions(documents_root)}
        assert len(dirs) == 2, "same title collapsed two documents into one"

    def test_the_return_text_tells_the_model_to_pass_doc_id_back(self, draft_env):
        """The tool's return value is the only place this rule reaches the model
        mid-run (trap ⑪: forgetting the doc_id proliferates documents)."""
        out = save_draft.invoke(tool_call(save_draft, {"title": "T", "markdown_text": "body"}))
        assert "doc_id" in out
        assert "传回" in out and "save_draft" in out

    def test_an_unknown_doc_id_saves_anyway_and_says_so(self, draft_env):
        """Refusing would throw away text that exists nowhere else. Saving it
        silently would hide that the version history did not continue."""
        out = save_draft.invoke(tool_call(save_draft, {"title": "T", "doc_id": "01NOSUCHDOCIDATALL0000000",
                                 "markdown_text": "内容不能丢"}))
        assert _doc_id(out)                      # it was saved
        assert "找不到" in out and "新文档" in out  # and the break is declared

    def test_lands_in_the_active_experiments_folder(self, draft_env, documents_root):
        """The whole point of the move: a saved report is inside the experiment it
        belongs to, so the folder is self-contained and reverse-lookup works."""
        from mast.agents._shared.data_paths import experiment_db_path
        from mast.logging.storage import ExperimentStorage

        st = ExperimentStorage(experiment_db_path())
        eid = st.create_experiment("Au(111) 形貌", "goal")
        st.set_active_scope(eid, None)

        out = save_draft.invoke(tool_call(save_draft, {"title": "报告", "markdown_text": "正文"}))
        path = next(p for p in documents_root.rglob("v001.md")
                    if "_unfiled" not in p.parts)
        assert path.parent.parent.name == "reports"
        assert eid[:8].lower() in path.parent.parent.parent.name.lower()
        assert "_unfiled" not in out

    def test_with_no_active_experiment_it_falls_open_to_unfiled(
            self, draft_env, documents_root):
        """Content loss is not a bookkeeping problem: the text only exists in the
        conversation, so refusing to save destroys it. Store it unfiled and say so."""
        out = save_draft.invoke(tool_call(save_draft, {"title": "无主报告", "markdown_text": "正文"}))
        assert (documents_root / "_unfiled" / "documents").is_dir()
        assert "未归属" in out and "_unfiled" in out
        assert "认领" in out, "told the operator it is unfiled but not what to do"

    def test_kind_defaults_to_experiment_report(self, draft_env, documents_root):
        """prompts.py: 「默认产出是一份内部实验报告，不是投稿手稿」. On disk the two
        used to be indistinguishable — same directory, same naming, differing only
        in how the LLM worded the title."""
        save_draft.invoke(tool_call(save_draft, {"title": "T", "markdown_text": "body"}))
        d = next(p for p in documents_root.rglob("doc.json"))
        assert d.parent.name.startswith("rpt__")

    def test_paper_draft_kind_is_distinguishable_on_disk(self, draft_env,
                                                         documents_root):
        save_draft.invoke(tool_call(save_draft, {"title": "T", "markdown_text": "body",
                           "kind": "paper_draft"}))
        d = next(p for p in documents_root.rglob("doc.json"))
        assert d.parent.name.startswith("draft__")

    def test_a_kind_this_agent_may_not_write_is_coerced_not_obeyed(
            self, draft_env, documents_root):
        """A writer that can mint any kind makes kind meaningless — 'review'
        belongs to paper_review, and a review written by the author is not one."""
        out = save_draft.invoke(tool_call(save_draft, {"title": "T", "markdown_text": "body",
                                 "kind": "review"}))
        assert _doc_id(out)
        d = next(p for p in documents_root.rglob("doc.json"))
        assert d.parent.name.startswith("rpt__")

    def test_rejects_empty(self, draft_env):
        # A rejection returns a plain string (nothing was saved, so there is no
        # pointer to publish) — LangChain wraps that in a ToolMessage. A success
        # returns a Command. tool_text() reads either.
        out = save_draft.invoke(tool_call(save_draft, {"title": "x", "markdown_text": "   "}))
        assert "empty" in tool_text(out)

    def test_rejects_scaffold_markers(self, draft_env):
        out = save_draft.invoke(tool_call(save_draft,
            {"title": "x", "markdown_text": "# T\n[FILL: intro]\n"}
        ))
        text = tool_text(out)
        assert "[FILL:" in text and "已保存" not in text

    def test_a_rejected_save_publishes_no_pointer(self, draft_env):
        """A refusal must not leave a DocRef behind: a pointer to a document that
        was never written would show the next agent an artifact that does not
        exist — strictly worse than showing it nothing."""
        out = save_draft.invoke(tool_call(save_draft, {"title": "x", "markdown_text": "  "}))
        assert getattr(out, "update", None) in (None, {}), \
            "a refused save still wrote to the artifact channel"

    def test_load_draft_roundtrip(self, draft_env):
        out = save_draft.invoke(tool_call(save_draft, {"title": "roundtrip",
                                 "markdown_text": "# Title\nHello world"}))
        did = _doc_id(out)
        loaded = load_draft.invoke({"doc_id": did})
        assert "Hello world" in loaded
        assert did in loaded, "the reviewer cannot quote back an id it was not given"

    def test_load_draft_current_finds_the_newest(self, draft_env):
        did = _doc_id(save_draft.invoke(tool_call(save_draft, {"title": "r", "markdown_text": "第一版"})))
        save_draft.invoke(tool_call(save_draft, {"title": "r", "doc_id": did, "markdown_text": "第二版"}))
        loaded = load_draft.invoke({"doc_id": "current"})
        assert "第二版" in loaded and "第一版" not in loaded

    def test_current_is_the_newest_DOCUMENT_not_just_the_newest_version(
            self, draft_env):
        """Two separate documents saved back to back: "current" must be the second.

        The operator-visible symptom of getting this wrong is the one already on
        record for the version level — 「load_draft('current') 拿到编辑之前的版本，
        五次一现」 — and it came back at the DOCUMENT level: doc.json's updated_at is
        stamped to the second, so two saves in the same second tie, and a stable
        sort then falls back to directory order and returns the OLDER document. The
        reviewer would review a different manuscript than the one just written.

        The store sorts with doc_id (a ULID, so lexicographic order is creation
        order) as the tiebreak. This test is at the TOOL seam on purpose: the
        store's own test pins list() ordering, while this one pins the sentence the
        operator would actually say. Do not weaken it to "either document is fine".

        The titles are ASCII-ordered ("A …" before "B …") deliberately. On a tie the
        old code fell through to ascending DIRECTORY NAME, so the bug only shows
        when the older document's directory sorts first — with the tiebreak removed
        this test was verified to fail, and with CJK-collated titles it might
        coincidentally have passed and caught nothing."""
        save_draft.invoke(tool_call(save_draft, {"title": "A 先写的", "markdown_text": "第一份手稿的正文"}))
        save_draft.invoke(tool_call(save_draft, {"title": "B 后写的", "markdown_text": "第二份手稿的正文"}))

        from mast.documents import store
        stamps = {e.meta.updated_at for e in store().list(kind="experiment_report")}
        loaded = load_draft.invoke({"doc_id": "current"})
        assert "第二份手稿的正文" in loaded, (
            "load_draft('current') returned the older document"
            + ("  (the two updated_at values TIED, which is the failing case)"
               if len(stamps) == 1 else ""))
        assert "第一份手稿的正文" not in loaded


# ════════════════════════════════════════════════════════════════════════
# save_review — linked to what it reviewed, not to a filename
# ════════════════════════════════════════════════════════════════════════

class TestSaveReview:
    def test_saves_and_links_to_the_target(self, draft_env, documents_root):
        target = _doc_id(save_draft.invoke(tool_call(save_draft, {"title": "Au111 report",
                                            "markdown_text": "# T\nbody"})))
        out = save_review.invoke(tool_call(save_review, {
            "target_doc_id": target,
            "verdict": "accept",
            "report_markdown": "## Overall Verdict: ACCEPT\nnone",
        }))
        rid = _doc_id(out)
        assert "判决 ACCEPT" in out and target in out

        from mast.documents import store
        entry = store().get(rid)
        assert entry is not None
        assert entry.meta.kind == "review"
        assert entry.meta.target_doc_id == target
        assert entry.meta.target_version == 1
        assert entry.dir.name.startswith("rev__")
        # The verdict comment is a load-bearing format — the frontend and the
        # documents API parse the verdict out of that first line.
        assert entry.read_text().startswith("<!-- verdict: ACCEPT -->")

    def test_rejects_bad_verdict(self, draft_env):
        out = save_review.invoke(tool_call(save_review, {
            "target_doc_id": "x", "verdict": "MAYBE", "report_markdown": "y",
        }))
        assert "verdict must be" in tool_text(out)

    def test_rejects_empty_report(self, draft_env):
        out = save_review.invoke(tool_call(save_review, {
            "target_doc_id": "x", "verdict": "ACCEPT", "report_markdown": "  ",
        }))
        assert "empty" in tool_text(out)

    def test_every_round_is_kept(self, draft_env):
        """Two rounds on one manuscript are two review documents sharing a
        target_doc_id. Before, the draft's version number was baked into the review
        FILENAME, so round 2 on v002 started a family unrelated to round 1 on v001."""
        target = _doc_id(save_draft.invoke(tool_call(save_draft, {"title": "d", "markdown_text": "body"})))
        for verdict in ("REVISE", "ACCEPT"):
            save_review.invoke(tool_call(save_review, {
                "target_doc_id": target, "verdict": verdict,
                "report_markdown": f"## Overall Verdict: {verdict}",
            }))
        from mast.documents import store
        reviews = [e for e in store().list(kind="review")
                   if e.meta.target_doc_id == target]
        assert len(reviews) == 2

    def test_an_unresolvable_target_still_saves_but_warns(self, draft_env):
        """Same rule as save_draft: the report exists nowhere else, so a broken
        link must not cost the text. It must cost a warning."""
        out = save_review.invoke(tool_call(save_review, {
            "target_doc_id": "Au111_report_v002",     # an old-style file stem
            "verdict": "REVISE", "report_markdown": "## Overall Verdict: REVISE",
        }))
        assert _doc_id(out)
        assert "解析不到" in out and "没有和任何手稿建立关联" in out

    def test_target_accepts_current(self, draft_env):
        save_draft.invoke(tool_call(save_draft, {"title": "d", "markdown_text": "body"}))
        out = save_review.invoke(tool_call(save_review, {
            "target_doc_id": "current", "verdict": "ACCEPT",
            "report_markdown": "## Overall Verdict: ACCEPT",
        }))
        assert "评审对象" in out and "解析不到" not in out


# ════════════════════════════════════════════════════════════════════════
# load_review — the revision loop, closed by doc_id rather than by prefix
# ════════════════════════════════════════════════════════════════════════

class TestLoadReview:
    def test_a_reports_doc_id_returns_the_review_OF_that_report(self, draft_env):
        """The bug this replaces: load_review matched review filename PREFIXES and
        sorted by mtime, so it could hand back the review of a different manuscript
        — or an older round of this one."""
        a = _doc_id(save_draft.invoke(tool_call(save_draft, {"title": "报告 A", "markdown_text": "A 的正文"})))
        b = _doc_id(save_draft.invoke(tool_call(save_draft, {"title": "报告 B", "markdown_text": "B 的正文"})))
        save_review.invoke(tool_call(save_review, {"target_doc_id": a, "verdict": "ACCEPT",
                            "report_markdown": "## Overall Verdict: ACCEPT\nA 的评审"}))
        save_review.invoke(tool_call(save_review, {"target_doc_id": b, "verdict": "REVISE",
                            "report_markdown": "## Overall Verdict: REVISE\nB 的评审"}))

        got = load_review.invoke({"doc_id": a})
        assert "A 的评审" in got and "B 的评审" not in got
        assert a in got

    def test_the_newest_round_wins(self, draft_env):
        target = _doc_id(save_draft.invoke(tool_call(save_draft, {"title": "r", "markdown_text": "正文"})))
        save_review.invoke(tool_call(save_review, {"target_doc_id": target, "verdict": "REVISE",
                            "report_markdown": "## Overall Verdict: REVISE\n第一轮"}))
        save_review.invoke(tool_call(save_review, {"target_doc_id": target, "verdict": "ACCEPT",
                            "report_markdown": "## Overall Verdict: ACCEPT\n第二轮"}))
        got = load_review.invoke({"doc_id": target})
        assert "第二轮" in got and "第一轮" not in got

    def test_a_report_with_no_review_yet_is_honest(self, draft_env):
        target = _doc_id(save_draft.invoke(tool_call(save_draft, {"title": "没审过", "markdown_text": "正文"})))
        got = load_review.invoke({"doc_id": target})
        assert "还没有" in got and "没审过" in got

    def test_a_reviews_own_doc_id_also_works(self, draft_env):
        target = _doc_id(save_draft.invoke(tool_call(save_draft, {"title": "r", "markdown_text": "正文"})))
        rid = _doc_id(save_review.invoke(tool_call(save_review, {
            "target_doc_id": target, "verdict": "ACCEPT",
            "report_markdown": "## Overall Verdict: ACCEPT\n意见正文"})))
        got = load_review.invoke({"doc_id": rid})
        assert "意见正文" in got and rid in got

    def test_latest_with_nothing_saved_is_honest(self, draft_env):
        got = load_review.invoke({"doc_id": "latest"})
        assert "一份评审都没有" in got

    def test_it_tells_the_writer_to_reuse_the_reports_doc_id(self, draft_env):
        target = _doc_id(save_draft.invoke(tool_call(save_draft, {"title": "r", "markdown_text": "正文"})))
        save_review.invoke(tool_call(save_review, {"target_doc_id": target, "verdict": "REVISE",
                            "report_markdown": "## Overall Verdict: REVISE\n1. 改这里"}))
        got = load_review.invoke({"doc_id": target})
        assert "doc_id" in got and "新版本" in got


# ═════════════════════════════════════════════════════════════════════
# Word 导出（请求：「做，用 word 内置。」）
# ═════════════════════════════════════════════════════════════════════

class TestExportReportDocx:
    """``export_report_docx`` —— 交付「能被别人**改**」的那一半。

    HTML 交付件解决的是「双击就开、图不断」；它解决不了修订/批注、期刊模板、
    投稿。所以两个工具并存，docstring 里写清什么时候用哪个。

    这个类存在的另一个理由是方法上的：上一版把工具加进了 ``AGENT_TOOLS`` 却
    从没执行过它 —— 文件里有一处 f-string 里的真实换行，模块**根本导不进来**，
    而只测渲染器和 HTTP 端点的话完全看不见。注册 ≠ 跑过。
    """

    def test_export_produces_a_docx_in_the_experiments_exports_dir(
            self, draft_env, documents_root):
        from mast.agents.paper_writing.tools import export_report_docx

        did = _doc_id(save_draft.invoke(tool_call(save_draft, 
            {"title": "Au(111) 阶段报告",
             "markdown_text": "# 阶段报告\n\n偏压 **-0.5 V**。\n"})))
        out = export_report_docx.invoke({"doc_id": did})

        assert "已导出 Word 文档" in out and "内置样式" in out
        files = list(documents_root.rglob("*.docx"))
        assert len(files) == 1
        p = files[0]
        assert p.parent.name == "exports"
        assert "_v001_" in p.name, "文件名必须带版本号，否则两份改稿分不出先后"
        assert p.read_bytes()[:2] == b"PK"

    def test_every_export_is_kept(self, draft_env, documents_root):
        """留档带时间戳共存 —— 上一份已经发出去的交付件不能被下一次导出抹掉。"""
        from mast.agents.paper_writing.tools import export_report_docx

        did = _doc_id(save_draft.invoke(tool_call(save_draft, {"title": "报告", "markdown_text": "v1"})))
        export_report_docx.invoke({"doc_id": did})
        save_draft.invoke(tool_call(save_draft, {"doc_id": did, "title": "报告", "markdown_text": "v2 正文"}))
        export_report_docx.invoke({"doc_id": did})
        names = sorted(p.name for p in documents_root.rglob("*.docx"))
        assert len(names) == 2 and "_v001_" in names[0] and "_v002_" in names[1]

    def test_unembeddable_figures_are_counted_in_the_return_value(
            self, draft_env, documents_root):
        """★ 缺图必须报数给**模型**看到。

        智能体只看得见返回字符串。一份图全丢的报告如果返回「已导出」，它就会
        当成交付完成继续往下走 —— 这正是「看着完整、实则少了数据」的产生方式。
        """
        from mast.agents.paper_writing.tools import export_report_docx

        did = _doc_id(save_draft.invoke(tool_call(save_draft, 
            {"title": "报告",
             "markdown_text": "# R\n\n![形貌](../_assets/nope.png)\n"})))
        out = export_report_docx.invoke({"doc_id": did})
        assert "1 张图片未能嵌入" in out
        assert "reports/_assets/" in out, "要告诉模型图该放哪，否则它无从修"

    def test_nothing_to_export_is_honest_and_actionable(self, draft_env,
                                                        documents_root):
        from mast.agents.paper_writing.tools import export_report_docx

        out = export_report_docx.invoke({"doc_id": "current"})
        assert "找不到可导出的报告/草稿" in out and "save_draft" in out
        assert not list(documents_root.rglob("*.docx"))

    def test_it_is_registered_so_the_agent_can_actually_call_it(self):
        from mast.agents.paper_writing.tools import AGENT_TOOLS

        assert "export_report_docx" in {t.name for t in AGENT_TOOLS}
