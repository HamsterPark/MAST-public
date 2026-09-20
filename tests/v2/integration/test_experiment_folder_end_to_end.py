"""实验文件夹端到端：一个实验的所有数据都在一个文件夹里。

设计文档：docs/v2/design/experiment_folder_persistence.md

这里验证的是**用户诉求本身**，不是某个函数的行为：
「一个实验是持久的，其所有数据都保存在一个文件夹内」
「即使用户在 Nanonis 里设定了独立文件夹，我们的实验文件夹内也应该有个副本」
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from mast.core import experiment_paths as ep
from mast.logging.experiment_log import ExperimentLog
from mast.logging.storage import ExperimentStorage
from mast.logging.v2 import manifest as mf
from mast.logging.v2.filestore import (
    ExperimentFileStore,
    Zone,
    classify,
    read_manifest,
    sweep_partials,
)
from mast.logging.v2.ingest_sink import QueuedIngestSink


@pytest.fixture()
def root(tmp_path, monkeypatch):
    r = tmp_path / "MAST-Data" / "experiments"
    monkeypatch.setenv("MAST_EXPERIMENT_ROOT", str(r))
    r.mkdir(parents=True, exist_ok=True)
    return r


@pytest.fixture()
def nanonis_dir(tmp_path):
    """模拟用户在 Nanonis 里自己设的保存目录（在实验根之外）。"""
    d = tmp_path / "SomeWhere" / "NanonisSession"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _aged_file(d: Path, name: str, payload: bytes) -> Path:
    """写一个文件并把 mtime 拨老，让它过静置门。"""
    p = d / name
    p.write_bytes(payload)
    old = time.time() - 60
    os.utime(p, (old, old))
    return p


def _scaffold(root, tmp_path, exp_title="NiI2质量表征", smp_name="film-A"):
    st = ExperimentStorage(str(tmp_path / "exp.db"))
    log = ExperimentLog(st)
    eid = log.start_experiment(exp_title, "验证文件夹自包含")
    sid = log.start_sample(smp_name, sample_type="2D")
    exp = st.get_experiment(eid)
    dir_name = ep.experiment_dir_name(eid, exp["name"], str(exp["start_time"]))
    st.set_experiment_dir_name(eid, dir_name)
    exp_dir = ep.experiment_dir(dir_name, create=True)
    sdir_name = ep.sample_dir_name(1, smp_name, sid)
    st.set_sample_dir_name(sid, sdir_name, 1)
    ep.sample_dir(exp_dir, sdir_name, create=True)
    mf.write_experiment_manifest(exp_dir, experiment_id=eid, title=exp["name"],
                                 goal=exp.get("goal_text") or "",
                                 created_at=str(exp.get("start_time") or ""),
                                 dir_name=dir_name,
                                 samples=[{"index": 1, "name": smp_name,
                                           "dir_name": sdir_name, "id": sid}])
    smp_row = st.get_sample(sid) or {}
    mf.write_sample_manifest(ep.sample_dir(exp_dir, sdir_name),
                             sample_id=sid, name=smp_name, experiment_id=eid,
                             sample_type=smp_row.get("sample_type") or "",
                             sample_subtype=smp_row.get("sample_subtype") or "",
                             description=smp_row.get("description") or "",
                             created_at=str(smp_row.get("start_time") or ""),
                             dir_name=sdir_name, index=1)
    mf.write_readme(exp_dir)
    return st, log, eid, sid, exp_dir, sdir_name


# ── 核心诉求 ──────────────────────────────────────────────────────────

def test_nanonis_file_saved_elsewhere_lands_in_the_experiment_folder(
        root, nanonis_dir, tmp_path):
    """★ 判据诉求：Nanonis 存在别处，实验文件夹里也要有副本。"""
    st, log, eid, sid, exp_dir, sdir = _scaffold(root, tmp_path)
    src = _aged_file(nanonis_dir, "Au111_mica_001.sxm", b"SXMDATA" * 900)

    store = ExperimentFileStore(exp_dir)
    res = store.ingest(src, sample_dir_name=sdir, source="skill",
                       action_id="act-1", skill="SaveScan")

    assert res.disposition == "new"
    copy = exp_dir / res.rel_path
    assert copy.is_file()
    assert copy.read_bytes() == src.read_bytes()
    # 原文件原封不动 —— 我们复制，从不搬走用户的数据
    assert src.is_file()

    rows = read_manifest(exp_dir / "samples" / sdir / "raw" / "_manifest.jsonl")
    assert len(rows) == 1
    assert rows[0]["action_id"] == "act-1"
    assert rows[0]["source"] == "skill"
    assert rows[0]["origin_path"] == str(src)
    assert len(rows[0]["sha256"]) == 64


def test_folder_is_self_contained_and_readable_without_mast(
        root, nanonis_dir, tmp_path):
    """整个文件夹拷走之后，靠 json + md 就能读懂它是什么。"""
    st, log, eid, sid, exp_dir, sdir = _scaffold(root, tmp_path)
    ExperimentFileStore(exp_dir).ingest(
        _aged_file(nanonis_dir, "s1.sxm", b"X" * 4096),
        sample_dir_name=sdir, source="skill")

    meta = json.loads((exp_dir / "experiment.json").read_text(encoding="utf-8"))
    assert meta["title"] == "NiI2质量表征"
    assert meta["samples"][0]["name"] == "film-A"

    readme = (exp_dir / "README.md").read_text(encoding="utf-8")
    assert "NiI2质量表征" in readme
    assert "没有「结束」或「归档」状态" in readme     # 语义写在文件夹里

    assert (exp_dir / "samples" / sdir / "raw" / "sxm" / "s1.sxm").is_file()
    assert (exp_dir / "chats").is_dir()             # 实验级对话有地方放
    assert (exp_dir / "env").is_dir()


def test_experiment_json_has_no_terminal_state(root, tmp_path):
    """★ 防回归：实验没有终态。

    「没必要做归档。有的实验可能过了十年重启。」——如果有人后来加回
    ended_at / status，这条测试要立刻炸。
    """
    st, log, eid, sid, exp_dir, sdir = _scaffold(root, tmp_path)
    meta = json.loads((exp_dir / "experiment.json").read_text(encoding="utf-8"))
    assert "ended_at" not in meta
    assert "status" not in meta
    assert "created_at" in meta and "last_active_at" in meta


# ── 防自我复制 ────────────────────────────────────────────────────────

def test_copy_is_never_re_ingested(root, nanonis_dir, tmp_path):
    """★ 递归闸门：副本喂回去必须被无条件忽略。

    如果这条挂了，watcher 会无限重新收编自己的产物。
    """
    st, log, eid, sid, exp_dir, sdir = _scaffold(root, tmp_path)
    store = ExperimentFileStore(exp_dir)
    res = store.ingest(_aged_file(nanonis_dir, "a.sxm", b"Z" * 2048),
                       sample_dir_name=sdir, source="skill")
    copy = exp_dir / res.rel_path

    for _ in range(3):
        again = store.ingest(copy, sample_dir_name=sdir, source="manual")
        assert again.disposition == "skipped"
        assert again.zone is Zone.MANAGED

    rows = read_manifest(exp_dir / "samples" / sdir / "raw" / "_manifest.jsonl")
    assert len(rows) == 1, "自管区的文件被重复收编了"
    assert len(list((exp_dir / "samples" / sdir / "raw" / "sxm").iterdir())) == 1


def test_inplace_file_is_still_ingested(root, tmp_path):
    """★ 与上一条对偶：原位模式下 Nanonis 写进来的文件必须被收进记录。

    天真的「排除整个 experiment_root」会让这条挂掉，而且是静默的 ——
    症状只是 scan_files 永远为 0。
    """
    st, log, eid, sid, exp_dir, sdir = _scaffold(root, tmp_path)
    sample_path = ep.sample_dir(exp_dir, sdir)
    inplace = ep.nanonis_inplace_dir(sample_path, create=True)
    src = _aged_file(inplace, "live_001.sxm", b"Q" * 3000)

    assert classify(src, active_sample_dir=sample_path) is Zone.INPLACE_ACTIVE

    res = ExperimentFileStore(exp_dir).ingest(
        src, sample_dir_name=sdir, source="inplace",
        active_sample_dir=sample_path)
    assert res.disposition == "inplace"
    assert res.sha256 and res.dest == src        # 原位：登记但不搬字节


def test_stray_file_from_previous_sample_is_attributed_to_active_sample(
        root, tmp_path):
    """★ 换样品后 Nanonis 还在往旧目录写 —— 归属真源是活跃样品，不是路径。"""
    st, log, eid, sid, exp_dir, sdir = _scaffold(root, tmp_path)
    sid2 = log.start_sample("film-B")
    sdir2 = ep.sample_dir_name(2, "film-B", sid2)
    st.set_sample_dir_name(sid2, sdir2, 2)
    s2_path = ep.sample_dir(exp_dir, sdir2, create=True)

    # Nanonis 仍在往 S01 的原位目录写
    old_inplace = ep.nanonis_inplace_dir(ep.sample_dir(exp_dir, sdir), create=True)
    stray = _aged_file(old_inplace, "wrong_place.sxm", b"W" * 2500)

    assert classify(stray, active_sample_dir=s2_path) is Zone.INPLACE_FOREIGN

    store = ExperimentFileStore(exp_dir)
    res = store.ingest(stray, sample_dir_name=sdir2, source="manual",
                       active_sample_dir=s2_path)
    # 副本落在【活跃样品】S02 下，而不是它在磁盘上待的那个 S01
    assert res.disposition == "new"
    assert sdir2 in res.rel_path


# ── 崩溃一致性 ────────────────────────────────────────────────────────

def test_no_half_files_ever_appear(root, nanonis_dir, tmp_path):
    """正在写的文件不会被收，也不会在 raw/ 里留下半个文件。"""
    st, log, eid, sid, exp_dir, sdir = _scaffold(root, tmp_path)
    fresh = nanonis_dir / "being_written.sxm"
    fresh.write_bytes(b"partial")          # mtime = now → 未过静置门

    res = ExperimentFileStore(exp_dir).ingest(fresh, sample_dir_name=sdir,
                                              source="manual")
    assert res.disposition == "retry"
    assert not list((exp_dir / "samples" / sdir / "raw" / "sxm").iterdir())


def test_manifest_survives_a_torn_last_line(root, nanonis_dir, tmp_path):
    """崩溃留下的半行 JSON 被丢弃，其余记录照常可读。"""
    st, log, eid, sid, exp_dir, sdir = _scaffold(root, tmp_path)
    store = ExperimentFileStore(exp_dir)
    store.ingest(_aged_file(nanonis_dir, "a.sxm", b"A" * 1500),
                 sample_dir_name=sdir, source="skill")

    manifest = exp_dir / "samples" / sdir / "raw" / "_manifest.jsonl"
    with open(manifest, "a", encoding="utf-8") as f:
        f.write('{"sha256": "truncated…')     # 断电写了一半

    rows = read_manifest(manifest)
    assert len(rows) == 1                      # 好行还在，坏行被丢弃
    assert rows[0]["source"] == "skill"


def test_sha_index_rebuilds_from_manifest(root, nanonis_dir, tmp_path):
    """索引可以丢：manifest 是权威，索引从它重建。"""
    st, log, eid, sid, exp_dir, sdir = _scaffold(root, tmp_path)
    store = ExperimentFileStore(exp_dir)
    res = store.ingest(_aged_file(nanonis_dir, "a.sxm", b"B" * 1800),
                       sample_dir_name=sdir, source="skill")

    (exp_dir / ".mast" / "sha256.idx").unlink()
    fresh_store = ExperimentFileStore(exp_dir)
    assert fresh_store.has_sha(res.sha256)


def test_partials_are_swept(root, tmp_path):
    st, log, eid, sid, exp_dir, sdir = _scaffold(root, tmp_path)
    junk = exp_dir / "samples" / sdir / "raw" / "sxm" / "x.sxm.part-123-abc"
    junk.parent.mkdir(parents=True, exist_ok=True)
    junk.write_bytes(b"half")
    assert sweep_partials(exp_dir) == 1
    assert not junk.exists()


# ── 去重 ──────────────────────────────────────────────────────────────

def test_same_bytes_are_not_copied_twice(root, nanonis_dir, tmp_path):
    st, log, eid, sid, exp_dir, sdir = _scaffold(root, tmp_path)
    src = _aged_file(nanonis_dir, "dup.sxm", b"D" * 2222)
    store = ExperimentFileStore(exp_dir)

    first = store.ingest(src, sample_dir_name=sdir, source="skill", action_id="a1")
    second = store.ingest(src, sample_dir_name=sdir, source="manual")

    assert first.disposition == "new"
    assert second.disposition == "duplicate"
    assert len(list((exp_dir / "samples" / sdir / "raw" / "sxm").iterdir())) == 1
    # 第二次目击仍然入账（source 不同），但没有第二份字节
    rows = read_manifest(exp_dir / "samples" / sdir / "raw" / "_manifest.jsonl")
    assert len(rows) == 2
    assert rows[1].get("duplicate_of")


# ── 队列 ──────────────────────────────────────────────────────────────

def test_sink_copies_without_blocking_the_caller(root, nanonis_dir, tmp_path):
    st, log, eid, sid, exp_dir, sdir = _scaffold(root, tmp_path)
    files = [_aged_file(nanonis_dir, f"f{i}.sxm", bytes([i]) * 4096) for i in range(5)]

    sink = QueuedIngestSink(None)
    try:
        sink.submit(files, exp_dir=exp_dir, sample_dir_name=sdir,
                    experiment_id=eid, sample_id=sid, action_id="a1",
                    source="skill", skill="SaveScan")
        assert sink.flush(timeout=15.0), "ingest queue did not drain"
    finally:
        sink.close()

    got = sorted(p.name for p in (exp_dir / "samples" / sdir / "raw" / "sxm").iterdir())
    assert got == [f"f{i}.sxm" for i in range(5)]
    assert sink.stats()["done"] == 5


# ── 对话导出 ──────────────────────────────────────────────────────────

def _conv_setup(root, tmp_path):
    from mast.chat.store import ConversationStore
    st, log, eid, sid, exp_dir, sdir = _scaffold(root, tmp_path)
    return st, log, eid, sid, exp_dir, sdir, ConversationStore.from_storage(st)


def test_chat_export_splits_experiment_and_sample_level(root, tmp_path):
    """没选样品时开的对话归实验级，选了样品后开的归样品级。"""
    from mast.chat.export import export_conversations
    st, log, eid, sid, exp_dir, sdir, cs = _conv_setup(root, tmp_path)

    c_exp = cs.create("_supervisor", kind="group", title="方案讨论",
                      thread_id="t1", experiment_id=eid, sample_id=None)
    cs.append_message(c_exp["conversation_id"], agent_id="_supervisor",
                      role="user", text="先做什么", kind="message")
    c_smp = cs.create("instrument_control", kind="private", title="扫描优化",
                      thread_id="t2", experiment_id=eid, sample_id=sid)
    cs.append_message(c_smp["conversation_id"], agent_id="instrument_control",
                      role="user", text="setpoint 50pA", kind="message")

    res = export_conversations(exp_dir, cs, experiment_id=eid,
                               sample_dir_resolver=lambda s: sdir if s == sid else None)
    assert res["conversations"] == 2 and res["entries"] == 2
    assert any(p.suffix == ".jsonl" for p in (exp_dir / "chats").iterdir())
    assert any(p.suffix == ".md"
               for p in (exp_dir / "samples" / sdir / "chats").iterdir())


def test_chat_export_is_incremental_and_idempotent(root, tmp_path):
    from mast.chat.export import export_conversations
    st, log, eid, sid, exp_dir, sdir, cs = _conv_setup(root, tmp_path)
    c = cs.create("_supervisor", kind="group", title="t", thread_id="t1",
                  experiment_id=eid, sample_id=None)
    cs.append_message(c["conversation_id"], agent_id="a", role="user",
                      text="one", kind="message")

    assert export_conversations(exp_dir, cs, experiment_id=eid)["entries"] == 1
    assert export_conversations(exp_dir, cs, experiment_id=eid)["entries"] == 0

    cs.append_message(c["conversation_id"], agent_id="a", role="assistant",
                      text="two", kind="message")
    assert export_conversations(exp_dir, cs, experiment_id=eid)["entries"] == 1

    jl = next(p for p in (exp_dir / "chats").iterdir() if p.suffix == ".jsonl")
    assert len(jl.read_text(encoding="utf-8").strip().splitlines()) == 2


def test_export_keeps_history_the_db_has_already_trimmed(root, tmp_path):
    """★ 这是导出真正的价值，不只是复制一份。

    ConversationStore 超过 _MAX_TRANSCRIPT_ROWS 会裁掉最旧的转录行。导出按 seq
    增量只追加，所以只要导出跑过，那段历史就永久留在实验文件夹里 —— 哪怕 DB
    里已经没有了。一个跑了三个月的群聊，它的开头只在这里。
    """
    import mast.chat.store as store_mod
    from mast.chat.export import export_conversations
    st, log, eid, sid, exp_dir, sdir, cs = _conv_setup(root, tmp_path)
    c = cs.create("_supervisor", kind="group", title="长群聊", thread_id="t1",
                  experiment_id=eid, sample_id=None)
    cid = c["conversation_id"]

    monkey_cap = 6
    orig = store_mod._MAX_TRANSCRIPT_ROWS
    store_mod._MAX_TRANSCRIPT_ROWS = monkey_cap
    try:
        for i in range(4):
            cs.append_message(cid, agent_id="a", role="user",
                              text=f"早期消息{i}", kind="message")
        # 先导出一次 —— 这一步是关键：它把早期消息落进了文件
        assert export_conversations(exp_dir, cs, experiment_id=eid)["entries"] == 4

        # 继续聊，把 DB 顶过上限，最旧的几条被裁掉
        for i in range(6):
            cs.append_message(cid, agent_id="a", role="assistant",
                              text=f"后期消息{i}", kind="message")
        export_conversations(exp_dir, cs, experiment_id=eid)
    finally:
        store_mod._MAX_TRANSCRIPT_ROWS = orig

    in_db = cs.messages_for(cid, limit=1000)
    assert len(in_db) <= monkey_cap, "前提没成立：DB 并没有裁剪"
    assert not any("早期消息0" in (m.get("text") or "") for m in in_db), \
        "前提没成立：最旧的消息还在 DB 里"

    jl = next(p for p in (exp_dir / "chats").iterdir() if p.suffix == ".jsonl")
    text = jl.read_text(encoding="utf-8")
    assert "早期消息0" in text, "★ 被 DB 裁掉的历史没有被文件夹保住"
    assert "后期消息5" in text


def test_chat_export_skips_conversations_with_no_experiment(root, tmp_path):
    """连活跃实验都没有时开的对话无处可落 —— 跳过，但 DB 里不动。"""
    from mast.chat.export import export_conversations
    st, log, eid, sid, exp_dir, sdir, cs = _conv_setup(root, tmp_path)
    cs.create("_supervisor", kind="group", title="孤儿", thread_id="t9",
              experiment_id=None, sample_id=None)
    res = export_conversations(exp_dir, cs, experiment_id=eid)
    assert res["entries"] == 0
    assert cs.list(limit=50), "对话必须仍在 DB 里"


# ── 环境 CSV 双写 ─────────────────────────────────────────────────────

def test_env_csv_writes_experiment_and_sample_levels(root, tmp_path):
    """温度是连续量（实验级按天轮转），但分析样品时要"这段时间的温度"（样品级）。

    双写而不是"样品结束时切一段出来" —— 根本没有"结束"这个时机。
    """
    from mast.environment.csv_sink import EnvironmentCsvSink
    st, log, eid, sid, exp_dir, sdir = _scaffold(root, tmp_path)
    state = {"sd": sdir}
    sink = EnvironmentCsvSink(lambda: (exp_dir, state["sd"], eid, sid))
    try:
        for i in range(3):
            sink.write("LakeShore335_A", 4.2 + i * 0.01, "K", "ok")
    finally:
        sink.close()

    day_files = list((exp_dir / "env").glob("LakeShore335_A_*.csv"))
    assert len(day_files) == 1
    lines = day_files[0].read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("timestamp_iso,epoch_s,sensor")
    assert len(lines) == 4                       # 表头 + 3
    assert eid in lines[1] and sid in lines[1]   # scope 写在行里

    smp_csv = exp_dir / "samples" / sdir / "env" / "LakeShore335_A.csv"
    assert smp_csv.is_file()
    assert len(smp_csv.read_text(encoding="utf-8").splitlines()) == 4


def test_env_csv_rotates_handles_on_sample_switch(root, tmp_path):
    from mast.environment.csv_sink import EnvironmentCsvSink
    st, log, eid, sid, exp_dir, sdir = _scaffold(root, tmp_path)
    sid2 = log.start_sample("film-B")
    sdir2 = ep.sample_dir_name(2, "film-B", sid2)
    ep.sample_dir(exp_dir, sdir2, create=True)

    state = {"sd": sdir}
    sink = EnvironmentCsvSink(lambda: (exp_dir, state["sd"], eid, sid))
    try:
        sink.write("T", 4.2, "K", "ok")
        state["sd"] = sdir2
        sink.rotate()
        sink.write("T", 77.4, "K", "ok")
    finally:
        sink.close()

    # 按 value 列断言，**不要**对整份 CSV 做子串搜索（2026-08-03 修）。
    # 每行是 timestamp_iso,epoch_s,sensor,value,unit,status,experiment_id,sample_id，
    # 前两列都是时间戳，而 "4.2" 会在里面随机出现：
    #     ...T09:14:04.205+00:00,1785662044.206,...
    #        ^^^^^^ 04.205 含 "4.2"      ^^^^^ 44.206 也含 "4.2"
    # 于是 `assert "4.2" not in b` 按墙钟随机假红（约百分之几的概率）。实测撞到过
    # 一次、紧接着重跑两次都过 —— 这种测试会周期性地把「全量零回归」这道验收门
    # 骗过去，而那道门是打包发布的判据。
    def _values(path):
        rows = path.read_text(encoding="utf-8").strip().splitlines()[1:]  # 跳表头
        return [r.split(",")[3] for r in rows if r.strip()]

    a = _values(exp_dir / "samples" / sdir / "env" / "T.csv")
    b = _values(exp_dir / "samples" / sdir2 / "env" / "T.csv")
    assert a == ["4.2"], f"轮转前那份应只含 4.2，实际 {a}"
    assert b == ["77.4"], f"轮转后那份应只含 77.4，实际 {b}"


def test_env_csv_never_raises_when_disk_fails(root, tmp_path, monkeypatch):
    """写失败必须自我禁用而不是把监控循环带崩 —— 它的首要职责是告警。"""
    from mast.environment import csv_sink as mod
    st, log, eid, sid, exp_dir, sdir = _scaffold(root, tmp_path)
    sink = mod.EnvironmentCsvSink(lambda: (exp_dir, sdir, eid, sid))

    def boom(*a, **kw):
        raise OSError("disk full")

    monkeypatch.setattr(mod.EnvironmentCsvSink, "_append", boom)
    sink.write("T", 1.0, "K", "ok")        # 不抛
    assert sink.stats()["failed"] == 1
    assert sink.stats()["disabled"] is True


# ── 迁移 ──────────────────────────────────────────────────────────────

def test_migration_plans_then_applies_then_rolls_back(root, tmp_path, monkeypatch):
    """Stage 0/1/2 + 回滚。原目录一个字节都不改。"""
    import time as _t

    from mast.core.types import ActionRecord
    from mast.logging.v2 import migrate_folders as mig

    monkeypatch.setenv("MAST2_PROJECT_ROOT", str(tmp_path))
    st = ExperimentStorage(str(tmp_path / "exp.db"))
    log = ExperimentLog(st)
    eid = log.start_experiment("NiI2质量表征")
    sid = log.start_sample("film-A")
    st.log_action(ActionRecord(skill_name="StartScan", experiment_id=eid, sample_id=sid))
    st.create_experiment("动作路由测试-实验")      # 空壳，像库里那几十条

    legacy = tmp_path / "working-sessions" / "20260728"
    legacy.mkdir(parents=True)
    keep = legacy / "during.sxm"
    keep.write_bytes(b"K" * 2048)
    now = _t.time()
    os.utime(keep, (now - 5, now - 5))

    p = mig.plan(storage=st, scan_dirs=[str(legacy)])
    assert p["counts"]["skipped_empty"] == 1, "空壳实验应被跳过"
    assert p["counts"]["with_folder"] == 1
    assert p["counts"]["files_claimed"] == 1

    res = mig.apply(p, storage=st, stage=2)
    assert res["created"] == 1 and res["files"] == 1
    exp_dir = ep.experiment_dir(p["experiments"][0]["dir_name"]
                                if p["experiments"][0]["dir_name"]
                                else p["experiments"][1]["dir_name"])
    assert list(exp_dir.rglob("during.sxm")), "历史文件没被认领"

    # 幂等
    res2 = mig.apply(p, storage=st, stage=2)
    assert res2["files"] == 0 and res2["duplicates"] == 1

    # 回滚：只删我们建的，原文件纹丝不动
    rb = mig.rollback(p)
    assert rb["removed"] == 1
    assert keep.is_file(), "迁移绝不能动原目录"


# ── P2：自包含可证明 ──────────────────────────────────────────────────

def test_db_can_be_rebuilt_from_folders_alone(root, nanonis_dir, tmp_path):
    """★ 「DB 是索引（可重建），文件夹是记录」—— 这条测试是那句话的证明。

    给一个全新的空库（等价于 DB 丢失），只靠实验文件夹重建出实验和样品。
    """
    from mast.logging.v2 import reindex

    st, log, eid, sid, exp_dir, sdir = _scaffold(root, tmp_path)
    ExperimentFileStore(exp_dir).ingest(
        _aged_file(nanonis_dir, "d.sxm", b"D" * 4096),
        sample_dir_name=sdir, source="skill")

    fresh = ExperimentStorage(str(tmp_path / "rebuilt.db"))
    assert fresh.list_experiments() == []

    res = reindex.rebuild_from_folders(storage=fresh)
    assert res["experiments_added"] == 1
    assert res["samples_added"] == 1

    got = fresh.get_experiment(eid)
    assert got and got["name"] == "NiI2质量表征"
    assert got["goal_text"] == "验证文件夹自包含"
    smp = fresh.get_samples(eid)
    assert len(smp) == 1 and smp[0]["name"] == "film-A" and smp[0]["sample_type"] == "2D"

    # 只补不改：重跑一次什么都不加
    again = reindex.rebuild_from_folders(storage=fresh)
    assert again["experiments_added"] == 0 and again["skipped"] == 1


def test_deep_fixity_detects_corruption(root, nanonis_dir, tmp_path):
    """★ scan_files.fixity_ok 自建库以来从未被真的验证过 —— 这里第一次读字节。"""
    from mast.logging.v2 import reindex

    st, log, eid, sid, exp_dir, sdir = _scaffold(root, tmp_path)
    res = ExperimentFileStore(exp_dir).ingest(
        _aged_file(nanonis_dir, "x.sxm", b"GOOD" * 512),
        sample_dir_name=sdir, source="skill")

    assert reindex.verify_all(deep=True)["sha_mismatch"] == []

    # 悄悄改掉副本的内容（保持长度，size 检查抓不到）
    copy = exp_dir / res.rel_path
    copy.write_bytes(b"BAAD" * 512)
    out = reindex.verify_all(deep=True)
    assert out["sha_mismatch"], "深度校验没能发现内容被篡改"


def test_export_record_db_is_independently_readable(root, tmp_path):
    """按需导出的 sqlite 不需要 MAST 就能打开。"""
    import sqlite3

    from mast.logging.v2 import reindex

    st, log, eid, sid, exp_dir, sdir = _scaffold(root, tmp_path)
    p = reindex.export_record_db(exp_dir, storage=st)
    assert p and p.is_file()
    assert p.parent.name == "exports"

    conn = sqlite3.connect(str(p))
    try:
        names = [r[0] for r in conn.execute("SELECT name FROM experiments")]
    finally:
        conn.close()
    assert names == ["NiI2质量表征"]


# ── 文档：报告 / 计划 / 论文草稿也住在实验文件夹里 ─────────────────────


def test_documents_live_in_the_experiment_folder(root, tmp_path, monkeypatch):
    """★ 「一个实验的所有数据都在一个文件夹里」现在也包括它产出的文档。

    一个实验可以有多份报告、多篇论文草稿、多个计划 —— 一对多是常态，它们互不
    覆盖，每一份自己带完整版本历史。
    """
    from mast.documents import reset_caches, store

    st, log, eid, sid, exp_dir, sdir = _scaffold(root, tmp_path)
    monkeypatch.setenv("MAST_EXPERIMENT_DB", str(getattr(st, "_db_path")))
    st.set_active_scope(eid, sid, updated_by="test")
    reset_caches()
    s = store()

    made = {}
    for kind, title in (("experiment_report", "阶段总结"),
                        ("paper_draft", "PRL 投稿"),
                        ("literature_report", "NiI2 文献综述")):
        r = s.save(text=f"# {title}\n\n正文。", kind=kind, title=title)
        assert r.ok, r.error
        assert r.experiment_id == eid
        made[kind] = r

    # 同一份报告改一版 → v002，v001 原样在
    again = s.save(text="# 阶段总结\n\n第二版。", doc_id=made["experiment_report"].doc_id)
    assert again.version == 2

    reports = exp_dir / "reports"
    assert reports.is_dir()
    doc_dirs = [d for d in reports.iterdir() if d.is_dir() and (d / "doc.json").is_file()]
    assert len(doc_dirs) == 3, "三份文档应当是三个独立目录"

    entry = s.get(made["experiment_report"].doc_id)
    assert entry.read_text(1).endswith("正文。")
    assert entry.read_text().endswith("第二版。")
    assert (entry.dir / "versions.jsonl").is_file()


def test_documents_survive_a_lost_db(root, tmp_path, monkeypatch):
    """★ 文档的「文件夹是记录，DB 是索引」—— 空库重建后文档全部复原。"""
    from mast.documents import reset_caches, store
    from mast.logging.v2 import reindex

    st, log, eid, sid, exp_dir, sdir = _scaffold(root, tmp_path)
    monkeypatch.setenv("MAST_EXPERIMENT_DB", str(getattr(st, "_db_path")))
    st.set_active_scope(eid, sid, updated_by="test")
    reset_caches()
    s = store()
    a = s.save(text="报告 v1", kind="experiment_report", title="总结")
    s.save(text="报告 v2", doc_id=a.doc_id)
    b = s.save(text="草稿", kind="paper_draft", title="投稿")
    before = {a.doc_id: 2, b.doc_id: 1}

    fresh = ExperimentStorage(str(tmp_path / "rebuilt_docs.db"))
    assert fresh.list_documents() == []

    res = reindex.rebuild_from_folders(storage=fresh)
    assert res["documents_added"] == 2
    assert res["document_versions_added"] == 3

    rows = fresh.list_documents()
    assert {r["doc_id"]: r["latest_version"] for r in rows} == before
    # 正文本来就在文件夹里，从没依赖过 DB
    assert store().get(a.doc_id).read_text() == "报告 v2"


def test_export_record_db_carries_the_document_index(root, tmp_path, monkeypatch):
    """按需导出的 sqlite 要能独立回答「这个实验有哪些报告」。

    正文本来就在实验文件夹里、随目录一起走；这里带的是索引与版本表。
    """
    import sqlite3

    from mast.documents import reset_caches, store
    from mast.logging.v2 import reindex

    st, log, eid, sid, exp_dir, sdir = _scaffold(root, tmp_path)
    monkeypatch.setenv("MAST_EXPERIMENT_DB", str(getattr(st, "_db_path")))
    st.set_active_scope(eid, sid, updated_by="test")
    reset_caches()
    a = store().save(text="报告 v1", kind="experiment_report", title="阶段总结")
    store().save(text="报告 v2", doc_id=a.doc_id)

    p = reindex.export_record_db(exp_dir, storage=st)
    assert p and p.is_file()
    conn = sqlite3.connect(str(p))
    try:
        docs = conn.execute("SELECT doc_id, kind, latest_version FROM documents").fetchall()
        vers = conn.execute("SELECT version FROM document_versions").fetchall()
    finally:
        conn.close()
    assert docs == [(a.doc_id, "experiment_report", 2)]
    assert sorted(v[0] for v in vers) == [1, 2]


# ── 文献库：一实验一专属库，书目住在实验文件夹里 ───────────────────────


@pytest.fixture()
def libs(tmp_path, monkeypatch):
    """把文献库注册表指到临时目录 —— 绝不碰用户真实的 registry.json。"""
    d = tmp_path / "literature_libs"
    monkeypatch.setenv("MAST_LITERATURE_LIBS_DIR", str(d))
    d.mkdir(parents=True, exist_ok=True)
    return d


def test_each_experiment_gets_its_own_library(root, tmp_path, libs, monkeypatch):
    """★ 两个实验并行加文献，各落各的 members.jsonl，互不污染。

    旧模型只有一个全机 ``active_library_id``：切实验不切库，两个实验并行会互相踩
    同一个活动库。
    """
    from mast.documents import reset_caches
    from mast.knowledge import experiment_library as xl

    st, log, eid_a, sid, exp_dir_a, sdir = _scaffold(root, tmp_path, exp_title="实验A")
    monkeypatch.setenv("MAST_EXPERIMENT_DB", str(getattr(st, "_db_path")))
    reset_caches()
    eid_b = st.create_experiment("实验B", "")
    from mast.documents.paths import exp_dir_for
    exp_dir_b = exp_dir_for(eid_b, create=True)

    lib_a = xl.ensure_experiment_library(eid_a)
    lib_b = xl.ensure_experiment_library(eid_b)
    assert lib_a and lib_b and lib_a != lib_b
    assert xl.ensure_experiment_library(eid_a) == lib_a, "懒创建必须幂等"

    xl.add_members(eid_a, ["W111", "W222"], reason="Au(111) 重构")
    xl.add_members(eid_b, ["W333"], reason="CDW")

    got_a = {m["work_id"] for m in xl.current_members(eid_a)}
    got_b = {m["work_id"] for m in xl.current_members(eid_b)}
    assert "W333" not in got_a and "W111" not in got_b, "两个实验的书目串了"
    assert (exp_dir_a / "library" / "members.jsonl").is_file()
    assert (exp_dir_b / "library" / "members.jsonl").is_file()


def test_library_is_carried_by_the_folder(root, tmp_path, libs, monkeypatch):
    """★ 「复制实验文件夹 = 带走书目」—— registry 丢了也能从文件夹重建。"""
    from mast.documents import reset_caches
    from mast.knowledge import experiment_library as xl

    st, log, eid, sid, exp_dir, sdir = _scaffold(root, tmp_path)
    monkeypatch.setenv("MAST_EXPERIMENT_DB", str(getattr(st, "_db_path")))
    reset_caches()
    xl.ensure_experiment_library(eid)
    xl.add_members(eid, ["W111", "W222"], reason="参考")
    xl.remove_members(eid, ["W111"])
    before = {m["work_id"] for m in xl.current_members(eid)}
    assert before == {"W222"}, "删除是事件，折叠后应当只剩 W222"

    # registry.json 是索引 + 缓存 —— 删掉它不该让书目消失
    reg_file = libs / "registry.json"
    if reg_file.exists():
        reg_file.unlink()

    n = xl.rebuild_registry_from_folders()
    assert n >= 1
    assert {m["work_id"] for m in xl.folder_members(eid)} == {"W222"}


def test_bibliography_is_readable_without_mast(root, tmp_path, libs, monkeypatch):
    """★ ``library/refs.md`` —— 没装 MAST 的人也读得出这个实验参考了哪些文献。

    「自包含」的兑现标准是「拷到另一台电脑，靠 ``experiment.json`` + ``README.md``
    就能读懂」。只有 ``members.jsonl`` 的话，那台电脑上的人得**自己在脑子里折叠一遍
    事件日志** —— 那不叫读得懂。refs.md 是视图（全量重渲染 + 原子替换），不是版本。
    """
    from mast.documents import reset_caches
    from mast.knowledge import experiment_library as xl

    st, log, eid, sid, exp_dir, sdir = _scaffold(root, tmp_path)
    monkeypatch.setenv("MAST_EXPERIMENT_DB", str(getattr(st, "_db_path")))
    reset_caches()
    xl.add_members(eid, ["W111"], reason="Au(111) 重构参考")
    xl.add_members(eid, ["W222"], reason="STS 对照")
    xl.set_fulltext(eid, "W111", "ingested", "papers/W111_au111")

    refs = exp_dir / "library" / "refs.md"
    assert refs.is_file(), "refs.md 没写出来 —— README 里对 library/ 的承诺就只兑现了一半"
    text = refs.read_text(encoding="utf-8")
    assert "W111" in text and "W222" in text
    assert "Au(111) 重构参考" in text and "STS 对照" in text
    assert "papers/W111_au111" in text          # 全文在本机的哪里
    assert "指针" in text                        # 别让人以为文件夹里应该有 PDF

    # 视图会跟着成员集重写：删掉一条，它就不在里面了
    xl.remove_members(eid, ["W222"])
    text2 = refs.read_text(encoding="utf-8")
    assert "W111" in text2 and "W222" not in text2


def test_experiment_folder_survives_a_move_to_another_machine(root, tmp_path, libs,
                                                              monkeypatch):
    """★★ 设计里最强的那句承诺的实证:**只拷实验文件夹**到一台新机器。

    新根、空库、没有 registry —— 然后:
      1. 正文完全不依赖 DB 就能读（文件夹是记录）
      2. reindex 把实验行、文档索引、版本行、文献库全部重建出来
      3. README + doc.json + versions.jsonl 让没装 MAST 的人也读得懂

    这三条同时成立，「一个实验的所有数据都在一个文件夹里」才不是一句口号。
    """
    import shutil

    from mast.documents import reset_caches, store
    from mast.knowledge import experiment_library as xl
    from mast.knowledge import libraries as libmod
    from mast.logging.v2 import reindex

    # ── 机器 A ──────────────────────────────────────────────────────
    st, log, eid, sid, exp_dir, sdir = _scaffold(root, tmp_path)
    monkeypatch.setenv("MAST_EXPERIMENT_DB", str(getattr(st, "_db_path")))
    st.set_active_scope(eid, sid, updated_by="A")
    reset_caches()
    s = store()
    rep = s.save(text="# 总结\n\n第一版。", kind="experiment_report", title="阶段总结")
    s.save(text="# 总结\n\n第二版。", doc_id=rep.doc_id)
    draft = s.save(text="# 投稿\n\n手稿。", kind="paper_draft", title="PRL 投稿")
    # 一份被**废弃**的文档同样要跟着文件夹走并被重新索引：废弃是「搬进 _discarded」，
    # 不是删除，所以它照样属于「文件夹里确实存在的东西」。
    junk = s.save(text="# 重复\n\n垃圾。", kind="experiment_report", title="重复报告")
    s.discard(junk.doc_id, reason="忘传 doc_id 造成的重复")
    xl.ensure_experiment_library(eid)
    xl.add_members(eid, ["W2741809807", "W1998"], reason="Au(111) 重构参考")
    xl.set_fulltext(eid, "W1998", "ingested", "papers/W1998")
    expected = {rep.doc_id: 2, draft.doc_id: 1}
    dir_name = exp_dir.name

    # ── 搬机器：只拷文件夹树，DB / registry 一概不带 ─────────────────
    new_root = tmp_path / "machine_B" / "experiments"
    new_root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(exp_dir, new_root / dir_name)
    monkeypatch.setenv("MAST_EXPERIMENT_ROOT", str(new_root))
    monkeypatch.setenv("MAST_EXPERIMENT_DB", str(tmp_path / "machine_B" / "fresh.db"))
    monkeypatch.setenv("MAST_LITERATURE_LIBS_DIR", str(tmp_path / "machine_B" / "libs"))
    reset_caches()
    libmod.reset_default_registry()

    st_b = ExperimentStorage(str(tmp_path / "machine_B" / "fresh.db"))
    assert st_b.list_experiments() == [] and st_b.list_documents() == []

    # 1) 正文不依赖 DB
    s_b = store()
    assert {d.doc_id: d.latest_version for d in s_b.list()} == expected
    assert s_b.get(rep.doc_id).read_text(1).endswith("第一版。")
    assert s_b.get(rep.doc_id).read_text().endswith("第二版。")

    # 2) reindex 重建索引
    res = reindex.rebuild_from_folders(storage=st_b)
    assert res["experiments_added"] == 1
    assert res["documents_added"] == 3          # 含那份被废弃的
    assert res["document_versions_added"] == 4
    assert res["libraries_rebuilt"] >= 1
    assert not res["errors"]
    rows = {r["doc_id"]: r["latest_version"] for r in st_b.list_documents()}
    assert {k: v for k, v in rows.items() if k in expected} == expected
    assert junk.doc_id in rows, "废弃的文档也该被重新索引 —— 它没被删，只是搬了区"
    # 但它不该出现在常规列表里
    assert junk.doc_id not in {e.doc_id for e in s_b.list()}
    assert junk.doc_id in {e.doc_id for e in s_b.list(include_discarded=True)}
    assert s_b.get(junk.doc_id).read_text() == "# 重复\n\n垃圾。"
    assert {m["work_id"] for m in xl.folder_members(eid)} == {"W2741809807", "W1998"}
    assert xl.experiment_library_id(eid) in {
        r["library_id"] for r in libmod._reg(None).list_libraries()}

    # 3) 人读得懂
    readme = (new_root / dir_name / "README.md").read_text(encoding="utf-8")
    assert "reports/" in readme and "library/" in readme
    meta = json.loads((s_b.get(rep.doc_id).dir / "doc.json").read_text(encoding="utf-8"))
    assert meta["title"] == "阶段总结" and meta["experiment_id"] == eid
    lines = (s_b.get(rep.doc_id).dir / "versions.jsonl").read_text(
        encoding="utf-8").splitlines()
    assert len(lines) == 2 and all("sha256" in ln for ln in lines)


def test_readme_documents_the_document_layout(root, tmp_path):
    """README 是给人看的那一页纸 —— 它承诺的结构必须与磁盘一致。

    旧版 README 写着「reports/ 报告（永不覆盖，_v001 _v002 …）」，而当时根本没有
    任何代码往那个目录写东西。承诺和现实不一致比没有承诺更糟。
    """
    st, log, eid, sid, exp_dir, sdir = _scaffold(root, tmp_path)
    readme = (exp_dir / "README.md").read_text(encoding="utf-8")
    assert "versions.jsonl" in readme
    assert "plans/" in readme
    assert "library/" in readme
    assert "没有「结束」或「归档」状态" in readme
