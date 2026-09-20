"""外部 agent 网关测试的夹具。共用零件在 ``_ext_world.py``（两边 import 同一个模块）。"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from _ext_world import GATES, PROBES, RAN, RT, Pool, State  # noqa: F401

from mast.core.registry import SkillRegistry


# ─────────────────────────────────────────────────────────────────────
# 夹具
# ─────────────────────────────────────────────────────────────────────

@pytest.fixture()
def world(tmp_path, monkeypatch, documents_root):
    import mast.llm.skill_author as sa
    import mast.webui.composite_panel as cp
    from mast.agents._shared.cognition import CognitionContext
    from mast.api.ext import create_ext_app
    from mast.api.ext.jobs import JobManager
    from mast.core.operating_mode import bind_mode_source
    from mast.logging.experiment_log import ExperimentLog, set_active_log
    from mast.logging.storage import ExperimentStorage
    from mast.logging.v2.live import open_live_v2
    from mast.skills.composite.version_store import CompositeVersionStore
    from mast.wishlist import reset_default_board

    RAN.clear()
    GATES.clear()
    monkeypatch.setenv("MAST_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MAST2_PROJECT_ROOT", str(tmp_path / "proj"))
    (tmp_path / "proj" / "working-sessions").mkdir(parents=True)
    reset_default_board()
    monkeypatch.setattr(sa, "_CUSTOM_SKILLS_DIR", tmp_path / "custom_skills")
    monkeypatch.setattr(cp, "_store", CompositeVersionStore(root=tmp_path / "composites"),
                        raising=False)
    monkeypatch.setattr(CognitionContext, "_ensure_vec", lambda self: None)

    repos, v2eid = open_live_v2()
    st = ExperimentStorage(str(tmp_path / "exp.db"))
    log = ExperimentLog(st)
    set_active_log(log)
    eid = log.start_experiment("外部网关测试")
    sid = log.start_sample("样品A", "测试样品")
    cog = CognitionContext(str(tmp_path / "exp.db"), author="agent")
    rt = RT(st, log, (repos, v2eid), cog)
    reg = SkillRegistry()
    for cls in PROBES:
        reg.register(cls)
    pool, state = Pool(), State()
    ctx = SimpleNamespace(connection_pool=pool, state=state, skill_registry=reg,
                          experiment_storage=st, live_app=rt, app=None, runtime=rt,
                          cognition=cog, memory_store=cog.store, config=None,
                          environment_monitor=None)
    jm = JobManager(tmp_path / "journal")
    app = create_ext_app(ctx, job_manager=jm)
    client = TestClient(app)
    w = SimpleNamespace(tmp=tmp_path, ctx=ctx, rt=rt, st=st, log=log, eid=eid, sid=sid,
                        repos=repos, v2eid=v2eid, cog=cog, reg=reg, pool=pool, state=state,
                        jm=jm, app=app, client=client)
    yield w
    for g in GATES.values():
        g.set()
    jm.shutdown(timeout_s=5)
    set_active_log(None)
    bind_mode_source(None)
    reset_default_board()

