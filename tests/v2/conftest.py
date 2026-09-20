r"""Pytest config for tests/v2 — make `import mast.*` resolve to MASTv2/mast/ not v1.

The repo has both:
  <repo>\mast\          (v1, legacy)
  <repo>\MASTv2\mast\   (v2, target)

When pytest auto-roots from pyproject.toml at repo root, sys.path gets the
project root, which means `import mast` resolves to v1. We inject MASTv2/ at
the front of sys.path so v2 wins.

This conftest.py runs at the start of pytest collection for any test under
tests/v2/. It does NOT affect tests/unit/ or tests/integration/ (those are
v1 tests run under the .venv Python 3.14 venv).
"""
from __future__ import annotations

import sys
from pathlib import Path

_MASTV2_ROOT = Path(__file__).resolve().parent.parent.parent / "MASTv2"

if str(_MASTV2_ROOT) not in sys.path:
    sys.path.insert(0, str(_MASTV2_ROOT))

# Defensive: if any v1 mast.* modules were imported before this conftest ran,
# purge them so subsequent `import mast.*` re-resolves through MASTv2/.
for _mod_name in list(sys.modules):
    if _mod_name == "mast" or _mod_name.startswith("mast."):
        _mod_path = getattr(sys.modules[_mod_name], "__file__", "") or ""
        if "MASTv2" not in _mod_path.replace("\\", "/"):
            del sys.modules[_mod_name]

# NOTE: even with this conftest, individual test files still need a top-of-file
# sys.path manipulation block because pytest's rootdir-driven sys.path setup
# (which adds D:\...\MAST first) overrides conftest's insert(0, MASTv2) at the
# point of test module import. See tests/v2/unit/test_wrap_skill_minimal.py
# for the canonical block to copy into new test files.

import os
import shutil

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_REAL_COMPOSITES = _REPO_ROOT / "config" / "composite_skills"
_REAL_WISHLIST = _REPO_ROOT / "MASTv2" / "artifacts" / "wishlist" / "wishlist.json"
_REAL_SUBSCRIPTION = _REPO_ROOT / "config" / "skill_subscription.json"

# Resolved ONCE, here, before any test can monkeypatch the env: this must be the
# OPERATOR's root, not whatever the test under inspection redirected to. Same
# reason _REAL_COMPOSITES / _REAL_WISHLIST are module constants.
try:
    from mast.core.experiment_paths import experiment_root as _experiment_root

    _REAL_EXPERIMENT_ROOT: Path | None = _experiment_root(create=False)
except Exception:  # pragma: no cover — no guard is better than a broken conftest
    _REAL_EXPERIMENT_ROOT = None


def _composite_store_fingerprint() -> tuple:
    """(relpath, size, mtime_ns) for every JSON under the operator's real skill
    store. Cheap on purpose — two scandirs, no file opens — because this runs
    around EVERY test."""
    out: list[tuple] = []
    for d in (_REAL_COMPOSITES, _REAL_COMPOSITES / "_history"):
        if not d.is_dir():
            continue
        try:
            for e in os.scandir(d):
                if e.is_file() and e.name.endswith(".json"):
                    st = e.stat()
                    out.append((f"{d.name}/{e.name}", st.st_size, st.st_mtime_ns))
        except OSError:  # pragma: no cover — a vanished dir is not a test failure
            continue
    return tuple(sorted(out))


@pytest.fixture(autouse=True)
def _real_composite_store_is_read_only():
    """THE SUITE MUST NEVER WRITE TO THE OPERATOR'S REAL SKILL DEFINITIONS.

    A full test run used to bump ``BatchRegionsScan`` from v46 to v47 and drop
    two junk files into ``_history/`` — ``test_skills_ext.py`` called the REAL
    ``composite_store()`` and its clone/restore round-trips wrote straight into
    ``config/composite_skills/``. It surfaced only because ``git checkout``
    refused to switch branches over the dirty file.

    The mess is the small part. The danger is that a test which fails or raises
    mid-write can leave a real composite the operator depends on in a corrupt or
    silently-mutated state — and a version bump is exactly the kind of change
    nobody notices.

    ``test_builder.py`` / ``test_version_store.py`` / ``test_composite_panel_*``
    all isolate correctly; ``test_skills_ext.py`` simply never got the fixture.
    Isolation is the fix (see :func:`seeded_composite_store`); THIS is the guard
    that stops the next test file from forgetting. It names the offending test.
    """
    before = _composite_store_fingerprint()
    yield
    after = _composite_store_fingerprint()
    if before == after:
        return
    added = sorted(n for n, _s, _m in set(after) - set(before))
    removed = sorted(n for n, _s, _m in set(before) - set(after))
    changed = sorted(
        n for n, _s, _m in set(after) - set(before)
        if n in {x for x, _, _ in before}
    )
    pytest.fail(
        "this test wrote to the operator's REAL composite skill store "
        f"({_REAL_COMPOSITES}).\n"
        f"  changed: {changed or '—'}\n"
        f"  added:   {[n for n in added if n not in changed] or '—'}\n"
        f"  removed: {removed or '—'}\n"
        "Use the `seeded_composite_store` fixture (tests/v2/conftest.py): it "
        "gives you a tmp copy of the real specs, so clone/restore round-trips "
        "still run against real data without touching it."
    )


@pytest.fixture(autouse=True)
def _real_subscription_is_read_only():
    """THE SUITE MUST NEVER WRITE TO THE OPERATOR'S REAL SUBSCRIPTION LIST.

    这是同一个形状的第六次（composite store / wishlist / literature registry /
    trace dir / …）。这一次的代价比前几次更直接：订阅列表决定 agent 手上有哪些
    工具，一个测试把它写成「只订阅了三个技能」，用户下次开机会发现显微镜大部分
    动作都不见了 —— 而界面上一切正常，因为那**就是**一份合法的订阅列表。

    ``subscription.store_path()`` 是惰性的、走 ``project_root()``，所以
    ``monkeypatch.setenv("MAST2_PROJECT_ROOT", tmp_path)`` +
    ``subscription.reset_default_store()`` 能真的重定向（见 subscription_store
    fixture）。这条守卫拦的是下一个忘了用它的人。

    文件**从不存在被凭空建出来**也要报 —— None → 有 = 变化。
    """
    def _fp():
        try:
            st = _REAL_SUBSCRIPTION.stat()
            return (st.st_size, st.st_mtime_ns)
        except OSError:
            return None

    before = _fp()
    yield
    after = _fp()
    if before != after:
        pytest.fail(
            f"this test wrote to the operator's REAL subscription list "
            f"({_REAL_SUBSCRIPTION}).\n"
            "Use the `subscription_store` fixture (tests/v2/conftest.py): it sets "
            "MAST2_PROJECT_ROOT to tmp_path and calls "
            "mast.skills.subscription.reset_default_store() so the file resolves "
            "somewhere disposable. 这份文件决定 agent 手上有哪些工具。"
        )


@pytest.fixture
def subscription_store(tmp_path, monkeypatch):
    """一个一次性的订阅 store（``<tmp>/config/skill_subscription.json``）。

    yield 出去的是那个 Path。进出都 ``reset_default_store()`` —— holder 是进程级的，
    不重置的话上一个测试的定制会漏给下一个（而它们全都 autouse 地共享同一个进程）。
    """
    from mast.skills import subscription as sub

    monkeypatch.setenv("MAST2_PROJECT_ROOT", str(tmp_path))
    sub.reset_default_store()
    try:
        yield tmp_path / "config" / sub.STORE_FILENAME
    finally:
        sub.reset_default_store()


@pytest.fixture(autouse=True)
def _real_wishlist_is_read_only():
    """THE SUITE MUST NEVER WRITE TO THE OPERATOR'S REAL WISHLIST.

    Same failure as the composite store, discovered the same way — by looking. The
    board's directory was a module-level constant built from a private
    ``parents[3]`` walk, so it ignored ``MAST2_PROJECT_ROOT`` entirely: every test
    that believed it had redirected the board with ``monkeypatch.setenv`` had
    redirected NOTHING and was writing into the operator's real wishlist. Twenty-
    five junk 「请提供 Au111 那张图的完整路径」 rows piled up in there before anyone
    noticed — mixed in among the agent's genuine upgrade requests.

    ``store._default_dir()`` is now lazy and goes through ``project_root()``, so
    the env override works. This guard is what stops the next private path-walk
    from quietly undoing that.
    """
    def _fp():
        try:
            st = _REAL_WISHLIST.stat()
            return (st.st_size, st.st_mtime_ns)
        except OSError:
            return None

    before = _fp()
    yield
    after = _fp()
    if before != after:
        pytest.fail(
            f"this test wrote to the operator's REAL wishlist ({_REAL_WISHLIST}).\n"
            "Set MAST2_PROJECT_ROOT to tmp_path (and call "
            "mast.wishlist.reset_default_board()) so the board resolves somewhere "
            "disposable — the agent's genuine 升级建议 live in that file."
        )


_REAL_LIB_REGISTRY = (_REPO_ROOT / "MASTv2" / "artifacts" / "literature_libs"
                      / "registry.json")


@pytest.fixture(autouse=True)
def _real_library_registry_is_read_only():
    """THE SUITE MUST NEVER WRITE THE OPERATOR'S REAL LIBRARY REGISTRY.

    ``_isolate_literature_data`` redirects the three literature paths for every
    test; **this** is what catches the next code path that resolves the registry
    some other way (a private repo-walk, a module-level constant captured at
    import time, a singleton built during collection before fixtures run).

    Why it earns its own guard: ``registry.json`` holds the operator's curated
    reading lists — ``work_id`` pointer sets with the reason each paper was kept.
    Nothing regenerates them. And the damage is invisible: the file still parses,
    the UI still lists libraries, there are just fewer papers in one of them than
    the operator put there. On 2026-07-29 this file was rewritten by an unisolated
    suite run (``saved_at`` moved to 11:30 that morning) and one library's member
    count could no longer be confirmed against any other record.

    Same shape as :func:`_real_wishlist_is_read_only`, same reason.
    """
    def _fp():
        try:
            st = _REAL_LIB_REGISTRY.stat()
            return (st.st_size, st.st_mtime_ns)
        except OSError:
            return None

    before = _fp()
    yield
    after = _fp()
    if before != after:
        pytest.fail(
            "this test wrote the operator's REAL literature registry "
            f"({_REAL_LIB_REGISTRY}).\n"
            "Those are hand-curated reading lists and nothing regenerates them. "
            "Redirect with MAST_LITERATURE_LIBS_DIR (the autouse "
            "_isolate_literature_data fixture already does — if you are seeing "
            "this, some code path is NOT going through knowledge/paths.py) and "
            "call mast.knowledge.libraries.reset_default_registry()."
        )


_REAL_TRACE_DIR = _REPO_ROOT / "experiments" / "traces"


@pytest.fixture(autouse=True)
def _isolate_readback_traces(monkeypatch, tmp_path_factory):
    """扎针 / 电脉冲采下来的**原始 z+电流曲线**必须落进 tmp,不落进用户的数据根。

    ``TipShapeWithReadback`` / ``BiasPulseWithReadback`` 从 2026-08-11 起把整条
    采集曲线写成 ``experiments/traces/*.json``(在那之前它采完就消失了)。这两个
    技能被**十七个**测试文件够得着 —— 扎针的四个 composite、修针工作流、
    operating_mode、HITL 推导…… 指望每个文件各自记得重定向就是「每页各自记得」,
    人肉找不齐;所以在这里统一设一次。

    **为什么这不只是「多几个垃圾文件」**:那个目录的用途是把「扎针深度 → 曲线
    → 团簇尺寸」串起来。合成曲线混进去之后,它们和真实的扎针记录**长得一模一样**
    —— 事后没有任何字段能把 2 kHz 的替身数列和真机数列分开。这与「借最新文件伪造
    历史」是同一种伤害。

    夹具同时是闸门:退出时核对真实目录**一个条目都没多**。重定向了 env A 而代码
    读 env B(或走一段私有的 ``parents[N]``)是本仓这条老伤反复复发的确切形状,
    而重定向本身不会告诉你它没生效 —— 得有人去看。
    """
    d = tmp_path_factory.getbasetemp() / "readback_traces"
    d.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("MAST_TRACES_DIR", str(d))

    def _fp():
        try:
            return tuple(sorted(e.name for e in os.scandir(_REAL_TRACE_DIR)))
        except OSError:
            return None      # 目录还不存在 —— 「凭空建出来」本身就要报

    before = _fp()
    yield
    after = _fp()
    if before != after:
        pytest.fail(
            "this test wrote readback traces into the operator's REAL data root "
            f"({_REAL_TRACE_DIR}).\n"
            f"  added: {sorted(set(after or ()) - set(before or ())) or '—'}\n"
            "MAST_TRACES_DIR is set for every test by this fixture, so if you are "
            "seeing this, some code path is NOT going through "
            "skills.builtins._readback_stream.trace_dir()."
        )


def _real_unfiled_docs_fingerprint() -> tuple:
    """What the suite must not change inside the operator's REAL experiment root.

    Three areas, because three writers reach into experiment folders:

    * ``_unfiled/documents/`` — documents saved with no active experiment;
    * ``<experiment>/reports/`` and ``<experiment>/plans/`` — documents that DID
      resolve an experiment (reports,論文 drafts, reviews, plan definitions);
    * ``<experiment>/library/`` — the experiment-scoped literature bibliography
      (``members.jsonl`` is append-only, so its **size** is the tell).

    The reports/plans half was added after the first version of this guard missed
    five junk documents (``Au111 report``, ``roundtrip``, ``first``, ``second``)
    that landed in a REAL experiment's ``reports/``: the guard only watched
    ``_unfiled``, and a test whose ``active_scope`` still pointed at the
    operator's live experiment therefore wrote straight into it, invisibly. The
    ``_unfiled`` path is what you hit with no active scope; **a leaked active
    scope is the more dangerous case**, because the junk then shows up in the
    operator's Records UI as real reports of a real experiment.

    Cheap enough to run around EVERY test: a handful of scandirs, no file opens.
    Names alone suffice for documents (a document directory is only ever created,
    never renamed); for the library the size catches appended events.
    """
    if _REAL_EXPERIMENT_ROOT is None:
        return ()
    out: list[str] = []
    # 两个 root 级的文档区：无归属（_unfiled）与已废弃（_discarded）。加了新的写入
    # 落点就必须同步扩这里的指纹范围，否则守卫给出的是**虚假的安全感** —— 这正是
    # 第一版漏掉 <exp>/reports/ 的教训。
    for zone in ("_unfiled", "_discarded"):
        try:
            out.extend(f"{zone}/documents/{e.name}"
                       for e in os.scandir(_REAL_EXPERIMENT_ROOT / zone / "documents"))
        except OSError:  # not there = nothing to protect yet, which is the good state
            pass
    try:
        for exp in os.scandir(_REAL_EXPERIMENT_ROOT):
            if not exp.is_dir() or exp.name.startswith("_"):
                continue
            for sub in ("reports", "plans"):
                try:
                    for c in os.scandir(Path(exp.path) / sub):
                        if c.is_dir():
                            out.append(f"{exp.name}/{sub}/{c.name}")
                except OSError:
                    continue
            # 只记一层目录名对这三个是不够的：它们的**内容**才是写入落点，而一层
            # 目录名一旦出现过就再也不变（第一次 discard 可见、之后每次隐形；往
            # _assets 里加图永远不改变指纹）。2026-07-29 架构审查在这里抓到三处
            # 同型盲区，三处都是当轮新增的写入落点。
            #   exports/        —— store.export_path() 写这里（HTML 交付件留档）
            #   reports/_assets/—— embed_figure 写这里（图池）
            #   reports/_discarded/ —— 有归属文档的废弃区（每废弃一份多一个子目录）
            #
            # 而且**必须带 size**，理由和紧接着的 library/ 一样：名字集合不是这三个
            # 目录的 tell。第一版只记名字，于是「同名文件再漏一次」永久隐形 ——
            # ``embed_figure`` 撞上同名同内容会走 reused 分支不再写盘，指纹前后完全
            # 相同，守卫全程沉默（复审实测：泄漏测试 1 passed，而它确实写到了真实根）。
            # 同名不同内容的覆写、exports/ 里同名 HTML 被重新导出，也都是这一类。
            for rel in ("exports", "reports/_assets", "reports/_discarded"):
                try:
                    for f in os.scandir(Path(exp.path) / rel):
                        try:
                            out.append(f"{exp.name}/{rel}/{f.name}:{f.stat().st_size}")
                        except OSError:
                            out.append(f"{exp.name}/{rel}/{f.name}:?")
                except OSError:
                    continue
            try:
                for f in os.scandir(Path(exp.path) / "library"):
                    out.append(f"{exp.name}/library/{f.name}:{f.stat().st_size}")
            except OSError:
                continue
    except OSError:
        pass
    return tuple(sorted(out))


@pytest.fixture(autouse=True)
def _scan_registry_starts_empty():
    """The scan registry is PROCESS-WIDE MEMORY — no env var can redirect it.

    ``core.scan_registry`` keeps ``_recent_files`` / ``_records`` / ``_id_order``
    / ``_session_dirs`` as module-level lists (bounded to 50). ``/api/artifacts``
    enumerates them through ``agents._shared.artifacts.list_existing()``, so a
    scan registered by ANY earlier test shows up as a "produced artifact" in a
    later one.

    Found 2026-08-15 by bisection: ``tests/v2/unit/skills/`` +
    ``test_agents_topology.py`` reproduced two failures that both files pass in
    isolation. Demonstrated directly — with every env var redirected,
    ``list_existing()`` still went 0 → 1 from a single in-memory append.

    **The guard below cannot see this**: it fingerprints the real experiment
    root on DISK, and this pollution never touches disk. Two different escape
    routes need two different guards.

    ``scan_registry.clear()`` already existed, and its docstring already said
    "Test helper." — it had simply never been wired to anything. That is the
    repo's most-recorded shape: a producer in place, no consumer.
    """
    from mast.core import scan_registry

    scan_registry.clear()
    yield
    scan_registry.clear()


@pytest.fixture(autouse=True)
def _real_experiment_root_is_read_only():
    """THE SUITE MUST NEVER WRITE INTO THE OPERATOR'S REAL DATA ROOT.

    Covers both writers into experiment folders: documents (``_unfiled/documents``)
    and the experiment-scoped literature bibliography (``<experiment>/library/``).
    The library half was added on 2026-07-29 after a suite run appended 17 test
    events (``W100``, ``W42``, ``local:fixed``) to a real experiment's
    ``members.jsonl`` — a unit test of ``knowledge.libraries`` had no active-scope
    isolation, so ``resolve_effective_library()`` read the operator's live
    ``active_scope`` and lazily created that experiment's library for real.
    Redirect BOTH env vars in any test that touches libraries or ingest::

        monkeypatch.setenv("MAST_EXPERIMENT_ROOT", str(tmp_path / "experiments"))
        monkeypatch.setenv("MAST_EXPERIMENT_DB", str(tmp_path / "exp.db"))

    Third instance of the same failure the two guards above cover, and it landed
    the day ``save_draft`` started going through ``mast.documents.store``
    (2026-07-29): the experiment root is resolved from ``MAST_EXPERIMENT_ROOT``
    (default ``D:\\MAST-Data\\experiments``) — it does NOT follow
    ``MAST2_PROJECT_ROOT``, so every test that redirected ``MAST_DRAFTS_DIR`` /
    ``MAST2_PROJECT_ROOT`` and believed itself isolated wrote real files into
    ``<real root>/_unfiled/documents/``. One subset run left 11 junk document
    directories, 24 version files among them, in the operator's data.

    Exactly the wishlist trap: a store whose path comes from a DIFFERENT env var
    than the one the test set. Redirect with::

        monkeypatch.setenv("MAST_EXPERIMENT_ROOT", str(tmp_path / "experiments"))
        mast.documents.reset_caches()      # the doc_id → dir cache is process-wide

    ``reset_caches()`` matters as much as the env var: without it the store keeps
    resolving doc_ids through directories under the previous root.
    """
    before = _real_unfiled_docs_fingerprint()
    yield
    after = _real_unfiled_docs_fingerprint()
    if before == after:
        return
    added = sorted(set(after) - set(before))
    pytest.fail(
        "this test wrote into the operator's REAL experiment root "
        f"({_REAL_EXPERIMENT_ROOT}).\n"
        f"  added/changed: {added or '—'}\n"
        f"  removed: {sorted(set(before) - set(after)) or '—'}\n"
        'Set MAST_EXPERIMENT_ROOT to a tmp dir and call '
        "mast.documents.reset_caches() — MAST2_PROJECT_ROOT does NOT redirect it."
    )


@pytest.fixture()
def documents_root(tmp_path, monkeypatch):
    """A disposable experiment root with a clean document store. Yields the root.

    Use this in ANY test that reaches code which may save a document — save_draft,
    save_review, save_literature_report, the documents API, ``list_existing()``.
    Redirecting ``MAST2_PROJECT_ROOT`` or ``MAST_DRAFTS_DIR`` is NOT enough: the
    experiment folder comes from ``MAST_EXPERIMENT_ROOT`` (see
    :func:`_real_experiment_root_is_read_only`, the guard that fails the test if
    you forget).

    ``reset_caches()`` runs on both sides: the store's ``doc_id → directory`` map
    is process-wide, so a previous test's entries would otherwise still resolve —
    and would resolve to paths under a tmp_path that no longer exists.
    """
    import mast.documents as _documents

    root = tmp_path / "experiments"
    root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("MAST_EXPERIMENT_ROOT", str(root))
    monkeypatch.setenv("MAST_EXPERIMENT_DB", str(root / "mast_experiments.db"))
    _documents.reset_caches()
    yield root
    _documents.reset_caches()


@pytest.fixture()
def seeded_composite_store(tmp_path, monkeypatch):
    """A composite version store rooted in tmp, SEEDED with a copy of the real
    specs + their ``_history``.

    Seeding matters: the clone/restore round-trip tests ``pytest.skip`` when the
    store is empty, so handing them a bare tmp dir would not isolate them — it
    would silently DELETE them. They must keep exercising real specs with real
    version history; they just must not do it in the operator's directory.
    """
    import mast.webui.composite_panel as cp
    from mast.skills.composite.version_store import CompositeVersionStore

    root = tmp_path / "composite_skills"
    if _REAL_COMPOSITES.is_dir():
        shutil.copytree(_REAL_COMPOSITES, root)
    else:  # pragma: no cover — a checkout with no skills is still testable
        root.mkdir(parents=True)

    store = CompositeVersionStore(root=root)
    monkeypatch.setattr(cp, "_store", store, raising=False)
    return store


@pytest.fixture(autouse=True)
def _isolate_composite_sidecar(tmp_path, monkeypatch):
    """P2-G: GraphExecutor writes step-progress sidecars under
    project_root()/experiments/composite_progress/ — running ANY composite in
    a test would pollute the real repo AND leak resume state across tests
    (a stale sidecar makes the next test silently resume-skip its steps).
    Redirect every test's sidecars into its own tmp dir."""
    try:
        import re as _re

        from mast.skills.composite import graph_executor as _ge
    except Exception:  # pragma: no cover — collection of non-mast tests
        yield
        return
    d = tmp_path / "_sidecars"
    d.mkdir(exist_ok=True)

    def _tmp_sidecar(name, run_id=""):
        # MUST mirror the real _sidecar_path(composite_name, run_id) signature.
        # It took only `name` until 2026-07-27, so every GraphExecutor
        # construction in the suite raised TypeError, was swallowed by the
        # "sidecar must never block a run" except, and left the executor with no
        # sidecar at all — i.e. the resume path was NEVER under test. The bug
        # that shipped because of it (a total_steps == 0 sidecar resuming into
        # fake 0.37 s scans) is pinned in
        # tests/v2/unit/skills/test_sidecar_unmeasurable.py.
        safe = _re.sub(r"[^\w一-鿿-]", "_", str(name))[:80] or "composite"
        rid = _re.sub(r"[^\w-]", "_", str(run_id))[:40]
        return d / (f"{safe}__{rid}.json" if rid else f"{safe}.json")

    monkeypatch.setattr(_ge, "_sidecar_path", _tmp_sidecar)
    yield


@pytest.fixture(autouse=True)
def _isolate_literature_data(tmp_path, monkeypatch):
    """Keep the whole test suite off the operator's real literature data.

    Three machine-level assets live outside any experiment folder and are shared
    by every literature code path: the big index (``artifacts/literature_index/``
    — 205 MB of vectors + 50k abstracts, *not* reproducible from anything in the
    repo), the library registry (``artifacts/literature_libs/registry.json``), and
    the full-text store (``data/papers/``). Any test that touches
    ``libraries`` / ``ingest`` / ``fetch_board`` without isolating first writes
    straight into them.

    Why this is autouse rather than per-test opt-in: on 2026-07-29 a test file
    isolated itself with ``monkeypatch.setattr(lib_mod, "_DEFAULT_LIBS_DIR", ...,
    raising=False)``. The constant was removed when the five private repo-walks in
    ``knowledge/*`` converged onto ``knowledge/paths.py``, the patch silently
    became a no-op, and one suite run added four fixture libraries to the real
    registry and flipped the operator's active library. Opt-in isolation fails
    open; this fails closed. A test that wants a specific directory still just
    sets the same env var in its own fixture (explicitly requested fixtures run
    after autouse ones, so it wins).

    The singleton reset on both sides is required: the registry and board cache a
    process-wide instance that captured its directory at construction time, so
    without a reset a test inherits whichever directory ran first.

    ``MAST_PAPER_CORPUS`` is set for the same fail-closed reason. The literature
    agent's PDF tools read their own env var, NOT ``MAST_PAPERS_DIR`` — and since
    ``_corpus_dirs()`` was fixed to also scan the legacy ``<repo>/data/papers``
    and ``<repo>/papers`` locations when that var is unset, leaving it unset here
    would point the suite straight at the operator's real PDFs. Setting it to the
    same empty tmp dir keeps every corpus scan inside the sandbox. (This is the
    same "redirect env A while the store reads env B" shape that caused the four
    previous pollution incidents.)
    """
    d = tmp_path / "_literature"
    monkeypatch.setenv("MAST_LITERATURE_INDEX_DIR", str(d / "literature_index"))
    monkeypatch.setenv("MAST_LITERATURE_LIBS_DIR", str(d / "literature_libs"))
    monkeypatch.setenv("MAST_PAPERS_DIR", str(d / "papers"))
    monkeypatch.setenv("MAST_PAPER_CORPUS", str(d / "papers"))

    def _reset() -> None:
        for mod, fn in (("mast.knowledge.libraries", "reset_default_registry"),
                        ("mast.knowledge.fetch_board", "reset_default_board")):
            try:
                import importlib
                getattr(importlib.import_module(mod), fn)()
            except Exception:  # pragma: no cover — non-mast test collection
                pass

    _reset()
    yield d
    _reset()


@pytest.fixture(autouse=True)
def _reset_tip_crash_tracker():
    """The tip-crash tracker is a PROCESS-LEVEL singleton (mast.core.
    tip_crash_tracker). Without a reset, crashes recorded by one test would
    leak into the next — a later scan at the same coords would be spuriously
    'crash-blocked'. Wipe it around every test."""
    try:
        from mast.core.tip_crash_tracker import reset_tip_crash_tracker
        reset_tip_crash_tracker()
    except Exception:  # pragma: no cover — non-mast test collection
        pass
    yield
    try:
        from mast.core.tip_crash_tracker import reset_tip_crash_tracker
        reset_tip_crash_tracker()
    except Exception:  # pragma: no cover
        pass


@pytest.fixture(autouse=True)
def _env_history_store_is_isolated(tmp_path_factory):
    """THE SUITE MUST NEVER WRITE TO THE OPERATOR'S REAL ENVIRONMENT HISTORY.

    ``mast.envhistory.store.get_store()`` is a process-level singleton that
    resolves ``project_root()/experiments/env_history/env_history.sqlite`` on
    first use. Anything that constructs a recorder, a sink, or simply calls the
    API handlers would otherwise open — and write to — the real archive.

    That archive is PERMANENT by design: statistics buckets and noise spectra
    are never swept. Test rows landing in it would not age out; they would sit
    in the operator's temperature history forever, and they would look exactly
    like real measurements.

    Fail-closed: every test gets its own tmp store whether it asked for one or
    not, and the singleton is cleared afterwards so the next test cannot inherit
    a closed connection. Tests that want their own store still call
    ``set_store_for_test`` and win — this only guarantees the DEFAULT is never
    the real file.
    """
    try:
        from mast.envhistory.store import EnvHistoryStore, set_store_for_test
    except Exception:  # pragma: no cover — non-mast test collection
        yield
        return
    d = tmp_path_factory.mktemp("envhistory")
    store = EnvHistoryStore(d / "env_history.sqlite")
    set_store_for_test(store)
    try:
        yield store
    finally:
        try:
            store.close()
        except Exception:  # pragma: no cover
            pass
        set_store_for_test(None)


@pytest.fixture(autouse=True)
def _env_history_recorder_is_reset():
    """The recorder is a process-level singleton too, and the current monitor's
    per-segment hook looks it up by that singleton. A recorder left behind by
    one test would keep receiving segments from the next one — and, worse, keep
    a reference to that test's tmp store after it was torn down."""
    try:
        from mast.envhistory.recorder import set_recorder
    except Exception:  # pragma: no cover
        yield
        return
    set_recorder(None)
    yield
    set_recorder(None)


@pytest.fixture(autouse=True)
def _env_history_thresholds_are_default():
    """Knobs are a process-level live-read holder. A test that flips
    ``eh_z_enabled`` on must not leave it on for the rest of the run — that one
    is the only knob in the recorder that issues TCP."""
    try:
        from mast.envhistory.thresholds import set_env_history_thresholds
    except Exception:  # pragma: no cover
        yield
        return
    set_env_history_thresholds(None)
    yield
    set_env_history_thresholds(None)


@pytest.fixture(autouse=True)
def _runtime_settings_are_isolated(tmp_path_factory):
    """THE SUITE MUST NOT READ THE OPERATOR'S REAL ``ui_settings.json``.

    2026-08-10: seven call sites in ``core/`` read settings through
    ``SettingsStore()`` — no argument, which raised TypeError inside a broad
    ``except`` and returned the fallback every single time. They now go through
    ``settings_store_for_runtime()``, which resolves the PUBLISHED store (the one
    ``POST /api/settings`` writes through) or, failing that, builds one against
    ``<project_root()>/config``.

    That fix has a side effect worth closing on the spot: those reads used to be
    unconditionally the defaults, and now they are whatever is on the developer's
    disk. ``_tip_policy`` / ``tip_conditioning_selfcheck`` call
    ``resolve_conditioning`` without injecting overrides, so an operator who sets
    ``tip_conditioning_overrides`` on this machine would quietly change what those
    skill tests assert. Nothing WRITES from those paths, so this is not the
    familiar "the suite scribbled on real data" trap — it is its mirror image:
    the suite reading a file that varies per machine.

    Publishing a disposable store reproduces the pre-fix behaviour EXACTLY (an
    empty store answers every key with its default), so this changes no existing
    test — it only stops the machine's own settings from leaking in. A test that
    wants the resolution path itself calls ``reset_process_store()``.
    """
    try:
        from mast.webui.settings_store import (
            SettingsStore,
            reset_process_store,
            set_process_store,
        )
    except Exception:  # pragma: no cover — no guard beats a broken conftest
        yield
        return
    d = tmp_path_factory.mktemp("ui_settings")
    set_process_store(SettingsStore(d))
    yield
    reset_process_store()


@pytest.fixture(autouse=True)
def _scan_policy_tiers_are_restored():
    """扫描档位表是**进程级全局**,谁改了谁得放回去 —— 这里替所有人放。

    ``mast.core.scan_policy._tiers`` 是模块级列表(用户自定义档位表),
    ``set_policy()`` 直接改它。测试里有若干处 ``set_policy(_TIERS)`` 之后
    **从不还原**,于是同一个 xdist worker 里后面的每一条测试都看到别人的表。

    2026-08-13 抓到:并发跑时
    ``test_a_smaller_frame_alone_does_not_make_the_scan_shorter`` 变红,
    单独跑绿 —— 50 nm 和 20 nm 在出厂表里同档(帧时相同,这正是它要钉的),
    在被污染的两档表里落进了不同档。

    同一个 flake 还有**反方向**的一半:
    ``test_preview_endpoint_answers_what_parameters_a_given_size_would_use``
    期望出厂表的 roi/256px/0.8s,它只在跑在污染者**之前**时才绿。

    修法放在这里而不是那两条测试上:「记得还原」是每个作者都要记住一次的事,
    而漏掉一次的症状是**别人的测试随机变红** —— 排查成本落在无辜的那一方。
    参见项目记忆 tests_polluting_real_user_data(同一形状,已第六次)。
    """
    from mast.core import scan_policy as _sp

    with _sp._lock:
        before = [dict(t) for t in _sp._tiers]
    yield
    with _sp._lock:
        _sp._tiers[:] = [dict(t) for t in before]


@pytest.fixture(autouse=True)
def _gallery_state_is_isolated(tmp_path, monkeypatch):
    """数据图库的状态目录在每条测试里都指向 tmp —— **阻断型**，不是事后指纹。

    图库（``mast.gallery``）的状态目录默认是 ``<experiment_root()>/_gallery``，也就是
    操作员真实的 ``D:\\MAST-Data\\experiments`` 下面：配置、缓存、几百 MB 缩略图、以及
    **操作员的全部数据标记**（``marks.json``）。上面那道
    ``_real_experiment_root_is_read_only`` 是 teardown 时比指纹 —— 它只能在写已经发生之后
    指认，而且只盯文档与文献库两处。本仓测试污染真实数据已经六次，第六次的教训写得很清楚：
    **只有指认型守卫时，第一次事故一定会发生。**

    所以这里在每条测试开始前就把 ``MAST_GALLERY_DIR`` 指到 tmp（``paths.state_dir()`` 每次
    调用都重新读 env），任何一条测试 —— 包括装配真实 app、逐个请求所有 GET 的
    ``test_boot_smoke`` —— 都够不着真实状态目录。目录本身**不预先创建**：图库的读路径
    承诺不 mkdir，测试可以直接断言它不存在。

    与第五次事故（「测试复位把真实路径重新变成可达的」）的区别：图库不缓存路径，
    后台构建在启动时解析一次并显式传下去；``mast.gallery.service._reset_for_tests()``
    只清内存状态。"""
    monkeypatch.setenv("MAST_GALLERY_DIR", str(tmp_path / "gallery"))
    yield
