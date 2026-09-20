"""文档存储的不变式。

设计文档：docs/v2/design/document_and_library_management.md

这里钉住的是**用户诉求**（「一个实验不一定对应一篇论文」「版本管理要能用」）
和三条最容易回归的机制：并发版本分配不覆盖、读侧能从残局自愈、无归属不丢内容。
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from mast.documents import reset_caches, store
from mast.documents.model import DOC_JSON as DOC_JSON_NAME, normalize_kind
from mast.documents.store import DocumentStore, count_words, read_versions
from mast.logging.storage import ExperimentStorage

# 源码级断言一律走它,不用 ``inspect.getsource``(2026-08-15)——
# 后者按 import 那一刻的行号切当前文件,别人同时在改就返回错位切片:
# ``in`` 那半给假红(吵、会被查),``not in`` 那半给**假绿**(不吵、没人会查)。
from tests.v2.srcref import source_of


@pytest.fixture()
def env(tmp_path, monkeypatch):
    """独立的实验根 + 独立的 v1 库，并清掉进程内缓存。

    ``reset_caches()`` 不是可选的：store 用 ``doc_id → 目录`` 的进程内缓存加速定位，
    换了实验根还留着上一个测试的缓存就会指向已经不存在的目录。
    """
    root = tmp_path / "MAST-Data" / "experiments"
    monkeypatch.setenv("MAST_EXPERIMENT_ROOT", str(root))
    monkeypatch.setenv("MAST_EXPERIMENT_DB", str(tmp_path / "db" / "exp.db"))
    root.mkdir(parents=True, exist_ok=True)
    reset_caches()
    yield root
    reset_caches()


@pytest.fixture()
def st(env):
    from mast.agents._shared.data_paths import experiment_db_path
    return ExperimentStorage(experiment_db_path())


@pytest.fixture()
def eid(st):
    e = st.create_experiment("Au(111) 形貌与 STS", "看清重构")
    st.set_active_scope(e, None, updated_by="test")
    return e


def test_save_lands_in_experiment_folder(env, eid):
    s = store()
    res = s.save(text="# 报告\n\n初稿。", kind="experiment_report",
                 title="Au111 形貌", created_by="agent:paper_writing")
    assert res.ok, res.error
    assert res.version == 1
    assert res.created_new
    assert res.experiment_id == eid
    assert res.root_kind == "experiment"
    p = Path(res.path)
    assert p.name == "v001.md"
    assert p.parent.parent.name == "reports"
    # sidecar 三件套
    assert (p.parent / "doc.json").is_file()
    assert (p.parent / "versions.jsonl").is_file()


def test_same_doc_id_appends_version_and_never_overwrites(env, eid):
    s = store()
    a = s.save(text="第一版", kind="experiment_report", title="报告")
    b = s.save(text="第二版", doc_id=a.doc_id)
    assert (b.version, b.created_new) == (2, False)
    entry = s.get(a.doc_id)
    assert [v.v for v in entry.versions] == [1, 2]
    assert entry.read_text(1) == "第一版"        # 旧版本原样在
    assert entry.read_text() == "第二版"


def test_unknown_doc_id_creates_new_doc_instead_of_losing_content(env, eid):
    """doc_id 找不到时**内容照存**，另立新文档并标记。

    报错等于把 LLM 已经写好的整篇内容扔了；「多一个文档」可以事后 claim 合并，
    「两个实验的报告混进一条版本历史」不可逆。
    """
    s = store()
    res = s.save(text="不能丢的内容", doc_id="NOSUCHDOC0000000000000000", title="孤儿")
    assert res.ok and res.created_new and res.doc_id_unknown
    assert s.get(res.doc_id).read_text() == "不能丢的内容"


def test_title_never_merges_two_documents(env, eid):
    """同一个标题保存两次 = 两个独立文档（不是一条版本链）。

    旧实现拿 LLM 自由填的 title 当版本族键：换措辞就分叉、撞名就误合并。身份只
    由 doc_id 决定，标题只是显示名。
    """
    s = store()
    a = s.save(text="A 的内容", title="月度总结", kind="experiment_report")
    b = s.save(text="B 的内容", title="月度总结", kind="experiment_report")
    assert a.doc_id != b.doc_id
    assert s.get(a.doc_id).read_text() == "A 的内容"
    assert s.get(b.doc_id).read_text() == "B 的内容"


def test_concurrent_versions_never_collide(env, eid):
    """N 线程同时保存同一文档：版本号无重复、每版内容都在（直击 TOCTOU）。

    旧的 ``next_version_path`` 是 glob-then-write，无锁无 O_EXCL：两个写入者算出
    同一个 ``_v003``，后写者静默覆盖前者 —— 这正是「永不覆盖」承诺的破口。
    """
    s = store()
    base = s.save(text="起点", title="并发", kind="paper_draft")
    n = 16
    out: list = [None] * n

    def worker(i: int) -> None:
        out[i] = s.save(text=f"并发内容 {i}", doc_id=base.doc_id, created_by=f"t{i}")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    oks = [r for r in out if r is not None and r.ok]
    assert len(oks) == n, [r.error for r in out if r and not r.ok]
    versions = [r.version for r in oks]
    assert len(set(versions)) == n, f"版本号重复：{sorted(versions)}"

    entry = s.get(base.doc_id)
    assert [v.v for v in entry.versions] == list(range(1, n + 2))
    bodies = {entry.read_text(v.v) for v in entry.versions}
    assert len(bodies) == n + 1, "有版本被覆盖了"


def test_read_side_heals_orphans_and_broken_jsonl(env, eid):
    """版本集合 = 目录扫描 ∪ jsonl，且空占位与半行都不算版本。"""
    s = store()
    res = s.save(text="v1", title="自愈", kind="experiment_report")
    d = s.get(res.doc_id).dir

    (d / "v007.md").write_text("孤儿：claim 成功但 jsonl 追加前崩了", encoding="utf-8")
    (d / "v008.md").write_text("", encoding="utf-8")   # claim 之后立刻崩：没有内容
    with (d / "versions.jsonl").open("a", encoding="utf-8") as fh:
        fh.write('{"v": 9, "file": "v00')               # 半行

    nums = [v.v for v in read_versions(d)]
    assert 7 in nums, "孤儿版本文件必须被目录扫描兜住"
    assert 8 not in nums, "空占位文件不是一个版本"
    assert 9 not in nums, "半行 jsonl 必须被丢弃"


def test_existing_placeholder_is_never_overwritten(env, eid):
    """撞上已存在的版本文件就往后挪 —— 哪怕它是空的。

    空文件可能是另一个进程刚 claim 完还没写内容，覆盖它就是数据损坏。"""
    s = store()
    res = s.save(text="v1", title="占位", kind="experiment_report")
    d = s.get(res.doc_id).dir
    (d / "v002.md").write_text("", encoding="utf-8")

    nxt = s.save(text="接着写", doc_id=res.doc_id)
    assert nxt.version == 3
    assert (d / "v002.md").read_text(encoding="utf-8") == ""


def test_no_active_experiment_falls_back_to_unfiled(env, st):
    """没有活跃实验时**照存**，落 ``_unfiled``。

    拒存的话，LLM 已生成的整篇内容只活在对话流里 —— 那是内容损失，不是记账损失。
    """
    st.set_active_scope(None, None, updated_by="test")
    reset_caches()
    res = store().save(text="没选实验时写的综述", title="无主", kind="literature_report")
    assert res.ok and res.root_kind == "unfiled"
    assert "_unfiled" in res.path
    assert res.experiment_id is None


def test_claim_moves_document_and_copies_referenced_assets(env, st, eid):
    """认领 = 物理搬家（主归属决定落点），被引用的图跟着走。"""
    from mast.documents.paths import assets_dir_for, doc_home, unfiled_docs_dir

    st.set_active_scope(None, None, updated_by="test")
    reset_caches()
    s = store()
    res = s.save(text="![图](../_assets/scan.png)\n\n正文", title="带图", kind="experiment_report")
    assert res.root_kind == "unfiled"
    src_assets = unfiled_docs_dir(create=True) / "_assets"
    src_assets.mkdir(parents=True, exist_ok=True)
    (src_assets / "scan.png").write_bytes(b"PNGDATA")

    out = s.claim(res.doc_id, eid)
    assert out.ok, out.error
    assert out.root_kind == "experiment" and "_unfiled" not in out.path
    entry = s.get(res.doc_id)
    assert entry.meta.experiment_id == eid
    assert entry.read_text().startswith("![图]")
    dst_assets = assets_dir_for(*doc_home("experiment_report", eid), create=False)
    assert (dst_assets / "scan.png").read_bytes() == b"PNGDATA", "图没跟过来就是一堆碎图标"


def test_one_experiment_holds_many_documents_of_each_kind(env, eid):
    """一个实验对应 0..N 篇论文 / N 份报告 / N 个计划 —— 一对多是常态。"""
    s = store()
    made = []
    for kind, title in (("experiment_report", "阶段总结一"),
                        ("experiment_report", "阶段总结二"),
                        ("paper_draft", "PRL 投稿"),
                        ("paper_draft", "Nature 投稿"),
                        ("literature_report", "文献综述"),
                        ("experiment_plan", "过夜计划")):
        r = s.save(text=f"{title} 的正文", kind=kind, title=title)
        assert r.ok, r.error
        made.append(r)
    listed = s.list(experiment_id=eid)
    assert len(listed) == len(made)
    assert len({e.doc_id for e in listed}) == len(made), "文档之间不该互相覆盖"
    assert len(s.list(experiment_id=eid, kind="paper_draft")) == 2


def test_related_experiments_are_visible_from_the_other_side(env, st, eid):
    """一篇论文可以综合多个实验的数据：主归属决定落点，关联实验也能看到它。"""
    other = st.create_experiment("NbSe2 CDW", "")
    s = store()
    res = s.save(text="跨实验论文", title="综合论文", kind="paper_draft")
    s.patch(res.doc_id, related_experiment_ids=[other])

    ids = {e.doc_id for e in s.list(experiment_id=other)}
    assert res.doc_id in ids
    entry = s.get(res.doc_id)
    assert s.relation_of(entry, other) == "related"
    assert s.relation_of(entry, eid) == "primary"
    assert res.doc_id not in {e.doc_id for e in s.list(experiment_id=other,
                                                      include_related=False)}


def test_patch_keeps_identity_and_records_title_history(env, eid):
    s = store()
    res = s.save(text="正文", title="旧名字", kind="experiment_report")
    d0 = s.get(res.doc_id).dir
    out = s.patch(res.doc_id, title="新名字")
    assert out.ok
    entry = s.get(res.doc_id)
    assert entry.meta.title == "新名字"
    assert entry.dir == d0, "改名不动目录 —— 目录名创建时冻结"
    assert entry.meta.title_history and entry.meta.title_history[0]["title"] == "旧名字"


def test_doc_json_has_no_terminal_state(env, eid):
    """文档和实验一样**没有终态**。

    有人后来加回 status / finalized_at / archived，这条测试要立刻炸。
    """
    s = store()
    res = s.save(text="正文", title="无终态", kind="experiment_report")
    meta = json.loads((s.get(res.doc_id).dir / "doc.json").read_text(encoding="utf-8"))
    for key in ("status", "finalized_at", "archived", "ended_at", "closed_at"):
        assert key not in meta


def test_review_points_at_its_target(env, eid):
    """评审靠 target_doc_id 关联被评文档，不再靠文件名前缀猜。"""
    s = store()
    draft = s.save(text="手稿正文", title="手稿", kind="paper_draft")
    rev = s.save(text="<!-- verdict: REVISE -->\n\n1. 缺误差棒", kind="review",
                 title="手稿 评审", target_doc_id=draft.doc_id, target_version=1)
    entry = s.get(rev.doc_id)
    assert entry.meta.target_doc_id == draft.doc_id
    assert entry.meta.target_version == 1


def test_db_index_mirrors_the_folder(env, st, eid):
    s = store()
    res = s.save(text="正文", title="索引", kind="experiment_report")
    s.save(text="第二版", doc_id=res.doc_id)
    row = st.get_document(res.doc_id)
    assert row and row["latest_version"] == 2 and row["experiment_id"] == eid
    assert len(st.get_document_versions(res.doc_id)) == 2
    assert st.count_documents(eid) >= 1


def test_resolve_ref_accepts_current_and_legacy_stem(env, eid):
    s = store()
    old = s.save(text="旧命名迁进来的", title="Au111 report", kind="experiment_report",
                 legacy_stem="Au111_report")
    assert s.resolve_ref(old.doc_id).doc_id == old.doc_id
    assert s.resolve_ref("Au111_report").doc_id == old.doc_id
    assert s.resolve_ref("draft:Au111_report").doc_id == old.doc_id
    assert s.resolve_ref("Au111_report_v002").doc_id == old.doc_id
    cur = s.resolve_ref("current", experiment_id=eid, kinds=("experiment_report",))
    assert cur is not None and cur.meta.kind == "experiment_report"


def test_kind_normalisation_tolerates_model_freestyle(env):
    assert normalize_kind("Paper-Draft") == "paper_draft"
    assert normalize_kind("literature review") == "literature_report"
    assert normalize_kind("完全不认识的东西") == "experiment_report"
    assert normalize_kind("", fallback="review") == "review"


def test_word_count_handles_cjk(env):
    """中文报告用 ``len(text.split())`` 会算出接近行数的荒谬值。"""
    assert count_words("这是一份中文报告") == 8
    assert count_words("hello world") == 2
    assert count_words("混合 text 内容") == 5


def test_latest_wins_when_two_documents_share_a_timestamp(env, eid, monkeypatch):
    """★ 同一秒保存两份 → ``latest()`` / ``resolve_ref("current")`` 必须给后存的那份。

    ``updated_at`` 是 ``timespec="seconds"``，背靠背保存**必然**并列；并列时 stable
    sort 回落到扫描顺序（目录名序），于是返回**最老**的那一份。这是陷阱③（mtime
    并列「五次一现」）在文档级的重演，而且不是理论风险 —— 连着两次调用就必现。

    后果是具体的：``load_draft("current")`` 把审稿人送到另一份手稿上，
    ``export_report_html("current")`` 把另一份报告当交付件导出，
    ``save_review(target_doc_id="current")`` 把评审挂到错的手稿上。
    """
    # 并列是这条测试的**前提**，所以显式构造，不靠「两次保存恰好落在同一秒」。
    #
    # 原来靠墙钟:两次背靠背 save 通常同秒,但**跨秒边界时前提就不成立**,第一个
    # 断言当场失败(实测约 1/10:'…T12:58:42+00:00' == '…T12:58:43+00:00')。
    # 那不是被测代码回归,是测试自己的前提没达成 —— 而一条约 10% 概率变红的测试
    # 会污染每一次全量闸门。原注释已经写明该怎么办(「请改成显式构造并列」),
    # 这里照做:把时钟钉住,让并列必然发生。
    # 钉在**使用处**而不是定义处:``store.py`` 在 import 时就把 ``now_iso`` 绑成了
    # 自己模块里的名字,补 ``model.now_iso`` 对它没有任何影响(第一版就是这么写的,
    # 失败率从 ~1/10 降到 ~1/15 —— 一个「看着修好了」的半修)。
    # 用 import_module 取模块:``mast.documents.__init__`` 里有一个同名的 ``store``
    # **函数**,``import mast.documents.store as _store`` 拿到的是那个函数,
    # setattr 会报 'function' object has no attribute 'now_iso'。
    import importlib

    _store = importlib.import_module("mast.documents.store")
    frozen = _store.now_iso()
    monkeypatch.setattr(_store, "now_iso", lambda: frozen)
    s = store()
    a = s.save(text="先存的 A", kind="experiment_report", title="A")
    b = s.save(text="后存的 B", kind="experiment_report", title="B")
    ea, eb = s.get(a.doc_id), s.get(b.doc_id)
    assert ea.meta.updated_at == eb.meta.updated_at, (
        "并列没构造出来 —— 时间戳不再取自 model.now_iso() 了?"
        "请把这里改成钉住新的时间来源,而不是删掉这条测试")

    assert s.latest(experiment_id=eid).doc_id == b.doc_id
    assert s.resolve_ref("current", experiment_id=eid).doc_id == b.doc_id
    assert [e.doc_id for e in s.list(experiment_id=eid)][0] == b.doc_id


def test_discard_hides_the_document_without_deleting_a_byte(env, eid):
    """「废弃」= 搬进 ``_discarded`` 区。**位置即状态**，没有 discarded 字段。

    为什么不是删除：`_quarantine` 定下的规矩是「绝不丢字节，宁可事后认领」。
    为什么需要它：``save()`` 在 doc_id 找不到时刻意另立新文档，允许增殖的前提是
    事后能清理。
    """
    import json as _json

    s = store()
    keep = s.save(text="留着的", kind="experiment_report", title="留着")
    junk = s.save(text="重复产生的垃圾", kind="experiment_report", title="垃圾")

    out = s.discard(junk.doc_id, reason="agent 忘传 doc_id 造成的重复")
    assert out.ok, out.error
    assert out.root_kind == "discarded" and "_discarded" in out.path

    listed = {e.doc_id for e in s.list(experiment_id=eid)}
    assert listed == {keep.doc_id}, "废弃的不该出现在常规列表里"
    assert junk.doc_id in {e.doc_id for e in s.list(include_discarded=True)}

    # 字节还在，正文照样读得出来
    entry = s.get(junk.doc_id)
    assert entry is not None and entry.read_text() == "重复产生的垃圾"
    # 位置即状态：doc.json 里没有任何终态字段
    meta = _json.loads((entry.dir / DOC_JSON_NAME).read_text(encoding="utf-8"))
    for key in ("discarded", "discarded_at", "status", "deleted"):
        assert key not in meta
    # 搬动留痕（只追加）
    moves = [_json.loads(ln) for ln in
             (entry.dir / "moves.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert moves and moves[-1]["op"] == "discard"
    assert moves[-1]["reason"].startswith("agent 忘传")


def test_discard_keeps_the_owner_so_restore_puts_it_back(env, st, eid):
    """废弃**不清**主归属 —— 区由路径表达，归属是文档自己带着的事实。

    留着它才能:restore 直接放回原实验(而不是丢进未归属区让用户再找一遍),而且
    ``list(include_discarded=True)`` 能如实回答「这个实验还有一份被废弃的报告」。
    """
    s = store()
    junk = s.save(text="其实有用", kind="experiment_report", title="误废")
    s.discard(junk.doc_id)

    discarded = s.get(junk.doc_id)
    assert discarded.meta.root_kind == "discarded"
    assert discarded.meta.experiment_id == eid, "废弃不该抹掉归属"
    assert junk.doc_id in {e.doc_id for e in
                           s.list(experiment_id=eid, include_discarded=True)}

    out = s.restore(junk.doc_id)           # 不给 id → 用文档自己记着的主归属
    assert out.ok and out.root_kind == "experiment"
    entry = s.get(junk.doc_id)
    assert entry.meta.experiment_id == eid
    assert entry.read_text() == "其实有用"
    assert junk.doc_id in {e.doc_id for e in s.list(experiment_id=eid)}


def test_restore_of_a_never_filed_document_goes_to_unfiled(env, st):
    """从来没有归属过的文档,恢复后回未归属区 —— **不猜**它该属于谁。"""
    st.set_active_scope(None, None, updated_by="test")
    reset_caches()
    s = store()
    d = s.save(text="无主的", kind="literature_report", title="无主恢复")
    assert d.root_kind == "unfiled"
    s.discard(d.doc_id)
    out = s.restore(d.doc_id)
    assert out.ok and out.root_kind == "unfiled"
    assert s.get(d.doc_id).meta.experiment_id is None


def test_discard_is_idempotent(env, eid):
    s = store()
    d = s.save(text="x", kind="experiment_report", title="幂等")
    first = s.discard(d.doc_id)
    second = s.discard(d.doc_id)
    assert first.ok and second.ok
    assert second.root_kind == "discarded"
    assert len(s.list(include_discarded=True)) == 1


def test_asset_links_resolve_from_both_reports_and_plans(env, eid):
    """图池按**实验**分（一个池），所以相对前缀要按文档所在子目录算。

    ``reports/<doc>/vNNN.md`` → ``../_assets/``
    ``plans/<doc>/vNNN.md``   → ``../../reports/_assets/``

    这两条如果算错，markdown 渲染器会去错的地方找图 —— 旧实现写死
    ``../figures/`` 指向全局目录，搬一次机器就全断。
    """
    from mast.agents._shared.report_html import render_html
    from mast.documents.paths import assets_dir_for, assets_rel_prefix, doc_home

    s = store()
    pool = assets_dir_for(*doc_home("experiment_report", eid), create=True)
    (pool / "scan.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"Q" * 32)

    for kind in ("experiment_report", "experiment_plan"):
        link = assets_rel_prefix(kind) + "scan.png"
        res = s.save(text=f"# 带图\n\n![形貌]({link})\n", kind=kind, title=f"{kind} 带图")
        assert res.ok, res.error
        version_file = s.get(res.doc_id).version_path()
        _doc, inlined, missing = render_html(
            s.get(res.doc_id).read_text() or "", base_dir=version_file.parent,
            title="t")
        assert (inlined, missing) == (1, 0), f"{kind} 的图没解析到：{link}"


def test_export_paths_coexist_within_the_same_second(env, eid):
    """导出件「多份共存、随时可重跑」—— 时间戳只到秒，同秒两次不能互相盖掉。"""
    s = store()
    res = s.save(text="正文", title="导出", kind="experiment_report")
    entry = s.get(res.doc_id)
    a = s.export_path(entry, ".html", stamp="20260729T120000")
    a.parent.mkdir(parents=True, exist_ok=True)
    a.write_text("first", encoding="utf-8")
    b = s.export_path(entry, ".html", stamp="20260729T120000")
    assert b != a
    b.write_text("second", encoding="utf-8")
    assert a.read_text(encoding="utf-8") == "first"


def test_new_version_does_not_roll_back_a_concurrent_rename(env, eid):
    """存新版本时要重读 doc.json，否则一次改名会被内存里的旧副本静默回滚。"""
    a = DocumentStore()
    res = a.save(text="v1", title="原名", kind="experiment_report")
    stale = a.get(res.doc_id)          # 此刻读到的 header

    DocumentStore().patch(res.doc_id, title="现场改定的新名")
    a._add_version(stale, "v2", created_by="agent")   # 用旧 header 存新版本

    entry = a.get(res.doc_id)
    assert entry.meta.title == "现场改定的新名"
    assert entry.latest_version == 2


def test_isolated_store_instances_share_the_folder(env, eid):
    """两个 store 实例看同一份磁盘 —— 权威是文件夹，不是进程内状态。"""
    a = DocumentStore()
    res = a.save(text="由 A 写入", title="共享", kind="experiment_report")
    b = DocumentStore()
    assert b.get(res.doc_id).read_text() == "由 A 写入"


# ── 架构审查 2026-07-29 抓到的两条(H2 / M8) ─────────────────────────────

def test_a_broken_doc_json_does_not_make_the_document_vanish(env, eid):
    """★ H2:``doc.json`` 被截成 0 字节，文档不能凭空消失 —— 正文完好无损。

    掉电后 ``write_text`` + ``os.replace`` 的典型残留就是这个。原实现下
    ``load_entry`` 返回 None → 文档从所有列表消失、``get(doc_id)`` 找不到、
    **连 reindex 都救不回来**，而 ``vNNN.md`` 里的正文一个字节都没少。
    那就违背了「任意时刻拔电源，已有内容必须自洽、可用」。
    """
    s = store()
    res = s.save(text="# 报告\n真实内容", kind="experiment_report", title="会坏的")
    d = s.get(res.doc_id).dir

    (d / DOC_JSON_NAME).write_text("", encoding="utf-8")     # 0 字节

    listed = s.list(include_discarded=True)
    assert len(listed) == 1, "文档消失了 —— 而它的正文还在磁盘上"
    entry = listed[0]
    assert entry.read_text() == "# 报告\n真实内容", "正文必须还能读出来"
    assert [v.v for v in entry.versions] == [1]
    assert "元数据损坏" in entry.meta.title, "标题要明说这是残局，别装作正常"
    assert entry.meta.kind == "experiment_report", "kind 能从目录名前缀恢复"
    # 身份用带前缀的临时 id：目录名里的 id8 只有 8 位且实测会撞，不能冒充 doc_id
    assert entry.doc_id.startswith("recovered:")


def test_doc_json_is_written_with_fsync(env, eid):
    """``doc.json`` 必须走带 fsync 的写入器，不是 manifest 那个没 fsync 的。"""
    import inspect
    from importlib import import_module

    # 不能写 ``from mast.documents import store`` —— 包的 __init__ 把名字 ``store``
    # 重导出成了那个**函数**，会把子模块遮蔽掉（仓库里 test_report_html_export.py
    # 也记过这个坑）。
    store_mod = import_module("mast.documents.store")
    src = source_of(store_mod.write_doc_json)
    assert "fsync" in src
    # 且 store 里不再有任何 doc.json 走 write_json_atomic 的写入点
    whole = inspect.getsource(store_mod)
    assert "write_json_atomic(entry.dir / DOC_JSON" not in whole
    assert "write_json_atomic(doc_dir / DOC_JSON" not in whole


def test_restore_falls_back_to_unfiled_when_the_experiment_is_gone(env, st, eid):
    """★ M8:原实验行没了，``restore`` 仍然要能把文档捞出来。

    docstring 承诺三级优先「显式 > 自己记着的 > `_unfiled` 待认领」。第二级失败
    （实验被删）时不能直接返回失败 —— 那会让文档永久卡在废弃区：常规列表不扫那里，
    restore 又走不通。
    """
    s = store()
    res = s.save(text="有用的内容", kind="experiment_report", title="孤儿救援")
    s.discard(res.doc_id, reason="先废弃")

    with st._connect() as conn:            # 实验行消失（换库/清理/手工删）
        conn.execute("DELETE FROM experiments WHERE id = ?", (eid,))

    out = s.restore(res.doc_id)
    assert out.ok, f"restore 失败了：{out.error}"
    assert out.root_kind == "unfiled"
    entry = s.get(res.doc_id)
    assert entry.read_text() == "有用的内容"
    assert res.doc_id in {e.doc_id for e in s.list()}, "恢复后必须在常规列表里可见"


def test_a_deep_root_actually_shrinks_the_directory_name(env, st, monkeypatch, tmp_path):
    """★ M7：根路径很深时**目录名真的被砍短**，而且内容不丢。

    陷阱 ⑱ 承诺「创建时检查预算，超了收缩 slug」。第一版只做了「``mkdir`` 失败就
    重试」，而复审实测本机 ``mkdir`` 一路建到 **416 字符都不报错** —— 阶梯从未执行，
    这条测试也就退化成只验「没报错」了（内容确实没丢，但原因是 mkdir 成功）。
    所以断言必须钉在**收缩发生了**上。

    ⑱ 关心的本来也不是「mkdir 有没有成功」，而是互操作性：超长路径 Python 读得
    回来，资源管理器打不开。
    """
    from mast.documents.model import doc_dir_name

    deep = tmp_path / ("d" * 70) / ("e" * 70) / "experiments"
    deep.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("MAST_EXPERIMENT_ROOT", str(deep))
    reset_caches()
    e2 = st.create_experiment("很深的根" * 6, "")
    st.set_active_scope(e2, None, updated_by="test")
    long_title = "标题也很长" * 8
    res = store().save(text="内容不能丢", kind="experiment_report", title=long_title)
    assert res.ok, f"深根下保存失败 = 内容损失：{res.error}"
    entry = store().get(res.doc_id)
    assert entry.read_text() == "内容不能丢"

    full_name = doc_dir_name("experiment_report", long_title, res.doc_id,
                             entry.meta.created_at)
    assert len(entry.dir.name) < len(full_name), (
        f"目录名没被砍短（{entry.dir.name!r}）—— 收缩没生效，"
        "这条测试就退化成只验「没报错」了")
    # 砍到了最短那一档（只有 <码>__<日期>__<id8>，连标题都不要了）
    assert entry.dir.name.count("__") == 2 and "标题" not in entry.dir.name

    # 注意这里**不能**断言全路径 ≤ 260：这个根自己就有 ~300 字符，再短的目录名也
    # 救不回来。那时的正确行为是**照样保存**（内容优先于整洁）——预算检查只作用于
    # 「还能砍」的那几档，最短那一档无条件尝试。预算算术本身由
    # test_path_budget_predicate 直接钉住。
    assert entry.read_text() == "内容不能丢"


def test_path_budget_predicate(env):
    """预算谓词本身：算到**最长的那个叶子**，而不是目录。

    目录建得下、里面的 ``progress.md.part`` 却建不下，是同一个问题的更晚一步。
    """
    from mast.documents.store import _path_budget_ok

    shallow = Path("C:/D/experiments/2026-07-29__e__abcdef01/reports")
    assert _path_budget_ok(shallow, "rpt__2026-07-29__短名字__01kyq0x1")
    assert not _path_budget_ok(Path("C:/" + "x" * 240), "rpt__2026-07-29__x__01kyq0x1")

    # 边界：total = len(home) + 1 + len(dir) + 1 + len(leaf)
    name, leaf = "rpt__d__i", "progress.md.part"
    fixed = 1 + len(name) + 1 + len(leaf)
    just_ok = Path("C:/" + "y" * (260 - fixed - 3))          # total == 260
    just_over = Path("C:/" + "y" * (261 - fixed - 3))        # total == 261
    assert len(str(just_ok / name / leaf)) == 260
    assert len(str(just_over / name / leaf)) == 261
    assert _path_budget_ok(just_ok, name)
    assert not _path_budget_ok(just_over, name)


def test_shrink_ladder_survives_a_hostile_mkdir(env, eid, monkeypatch):
    """阶梯的最后一档：主动预算算过了，``mkdir`` 仍然拒绝长名字。

    本机 ``mkdir`` 太宽容（建到 416 字符都不报错），所以用 monkeypatch 造一个
    「名字超过 40 字符就拒绝」的文件系统，把「失败就砍短」这条路径真正跑一遍 ——
    否则它在 CI 上永远绿、也永远没被执行过。
    """
    real_mkdir = Path.mkdir

    def picky_mkdir(self, *a, **kw):
        if len(self.name) > 40:
            raise OSError(206, "name too long (simulated)")
        return real_mkdir(self, *a, **kw)

    monkeypatch.setattr(Path, "mkdir", picky_mkdir)
    res = store().save(text="内容还是不能丢", kind="experiment_report",
                       title="一个非常非常长的标题" * 5)
    monkeypatch.undo()
    assert res.ok, f"阶梯没兜住：{res.error}"
    assert len(Path(res.path).parent.name) <= 40
    assert store().get(res.doc_id).read_text() == "内容还是不能丢"
