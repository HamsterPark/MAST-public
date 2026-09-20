"""Regression: CoreRuntime._meta_tool_context() must NOT raise.

The operator's installed app reported "实验记录不可用" for EVERY experiment/sample
meta-tool while storage was demonstrably fine (the DB existed and was being
written). Root cause: ``_meta_tool_context`` referenced ``self._get_chat_navigator``
— a Gradio-era method that was never ported to the TS/API runtime — so building the
provider dict raised ``AttributeError``. ``meta_tools._ctx()`` swallows any provider
exception into an EMPTY dict, so ``ctx.get("experiment_log")`` came back None and
every meta-tool returned "实验记录不可用".

This unit test exercises the exact provider wiring (no frozen build, no sim, no
hardware) — the test class that was missing and let the bug ship. It fails with
AttributeError on the old code and passes once the reference is getattr-guarded.

(2026-07-30: ``get_navigator`` is gone — the surface-navigation tools now derive
everything from the recorded scan-map markers via ``storage`` +
``map_analysis_cfg``. The regression this file guards is unchanged and is not
about that key: building the provider dict must never raise, because
``meta_tools._ctx()`` swallows the exception and every meta-tool then reports the
records store as unavailable.)
"""
from __future__ import annotations

from mast.config import MASTConfig
from mast.core.runtime import CoreRuntime


def _bare_runtime() -> CoreRuntime:
    # Skip __init__/setup() — _meta_tool_context only needs `config` + getattr
    # fallbacks for everything else. This isolates the provider-dict construction.
    rt = CoreRuntime.__new__(CoreRuntime)
    rt.config = MASTConfig()
    return rt


def test_meta_tool_context_does_not_raise_on_unported_navigator():
    rt = _bare_runtime()
    ctx = rt._meta_tool_context()  # must NOT raise AttributeError
    assert isinstance(ctx, dict)
    # The keys the meta-tools read must all be present (a raised provider would
    # have been swallowed into {} → "实验记录不可用").
    for key in ("experiment_log", "plan_store", "storage", "map_analysis_cfg",
                "experiments_dir", "session_path", "session_dir"):
        assert key in ctx, f"meta-tool context missing key {key!r}"


def test_map_analysis_config_builds_without_hardware_or_profile():
    """The scan-map tools must still answer on a bare runtime.

    ``build_map_analysis_config`` reads the instrument profile, the piezo safety
    limits and the live scan frame — none of which exist here. It has to fall
    back rather than raise, or the provider dict fails to build and takes every
    unrelated meta-tool down with it (the bug this file exists for)."""
    rt = _bare_runtime()
    cfg = rt.build_map_analysis_config()
    assert cfg.piezo_half_range_m == MASTConfig().safety.xy_max_m
    assert cfg.strategy in ("center_first", "perimeter_inward")
    assert cfg.frame_size_m > 0


def test_experiment_log_reaches_meta_tools_when_storage_present(monkeypatch, tmp_path):
    # With a wired _experiment_log, the provider must surface it (NOT None) so
    # start_experiment doesn't report "实验记录不可用".
    rt = _bare_runtime()
    sentinel = object()
    rt._experiment_log = sentinel
    ctx = rt._meta_tool_context()
    assert ctx["experiment_log"] is sentinel

    # And the meta-tool itself: a populated provider → start_experiment proceeds
    # past the "实验记录不可用" guard (calls into the log).
    from mast.agents._shared.meta_tools import make_meta_tools

    class _Log:
        # **kwargs so the fake tolerates the idempotency kwarg the meta-tool now
        # passes (reuse_open=True) — and any future one — without breaking.
        def start_experiment(self, name, goal="", **kwargs):
            return "exp-123"

    tools = make_meta_tools(lambda: {"experiment_log": _Log()})
    start = next(t for t in tools if t.name == "start_experiment")
    out = start.invoke({"name": "t", "goal": "g"})
    assert "实验记录不可用" not in out
    assert "exp-123" in out


# ── session-dir resolution on the runtime (2026-07-01) ──────────────────────
# `_session_path`/`_session_dir` were NEVER assigned (only read → always None) so
# scans Nanonis saved outside working-sessions were undiscoverable. The runtime now
# resolves the live save dir via Util_SessionPathGet (cached ~15 s) and mirrors it
# onto those attrs, so the meta-tool context + Records route + vision thumbnails
# find real .sxm. (Parallel inline copy of scan_extra._session_dir_from_caller —
# core must not import skills; test both so the two can't silently drift.)

class _FakePool:
    """Minimal pool: counts Util_SessionPathGet calls, returns a canned path."""

    def __init__(self, path: str):
        self._path = path
        self.calls = 0

    def safe_call(self, method, *args, role="main"):
        from mast.core.types import NanonisCallRecord

        self.calls += 1
        if method == "Util_SessionPathGet":
            return NanonisCallRecord(method=method, args=args,
                                     return_value=("", b"", [0, self._path]), error="")
        return NanonisCallRecord(method=method, args=args, error="unmocked")


def test_resolve_session_dir_none_without_pool():
    rt = _bare_runtime()
    assert rt._resolve_session_dir() is None  # no _pool → degrades, never raises


def test_resolve_session_dir_reads_pool_and_mirrors_attrs(tmp_path):
    rt = _bare_runtime()
    rt._pool = _FakePool(str(tmp_path))
    assert rt._resolve_session_dir() == str(tmp_path)
    # mirrored onto the attrs the meta-tool context + records route read
    assert rt._session_path == str(tmp_path)
    assert rt._session_dir == str(tmp_path)


def test_resolve_session_dir_caches_within_ttl(tmp_path):
    rt = _bare_runtime()
    pool = _FakePool(str(tmp_path))
    rt._pool = pool
    rt._resolve_session_dir()
    rt._resolve_session_dir()  # within 15 s → cache hit, no 2nd Nanonis round-trip
    assert pool.calls == 1


def test_meta_tool_context_surfaces_live_session_dir(tmp_path):
    rt = _bare_runtime()
    rt._pool = _FakePool(str(tmp_path))
    ctx = rt._meta_tool_context()
    assert ctx["session_path"] == str(tmp_path)
    assert ctx["session_dir"] == str(tmp_path)
