"""Skill 覆盖层的 HTTP 入口。

这份测试盯三件事，第一件最容易被「简化」掉：

1. **路径不能是 ``/skills/overlay``。** ``skills_ext.py:121`` 的
   ``GET /skills/{name}`` 会把它当成 ``name="overlay"`` 捕获，返回 **200 加一张
   不存在的技能卡片** —— 不报错、不 404，UI 拿到的是一个语法上完全合法的错答案。
2. **绝不 500。** 房规：router import + API 独立启动；缺东西就返回
   ``degraded=True`` 的有效响应。
3. **「排队了」不能显示成「生效了」。** 响应里 ``ok`` 之外必须有
   ``agent_path_pending`` 和 ``fingerprint_matches``，而后者为 ``None`` 表示
   **判断不了**，不是 False。
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("MAST2_PROJECT_ROOT", str(tmp_path))
    from mast.api.app import create_app

    return TestClient(create_app())


# ── 路由遮蔽（这条钉住那个决定）─────────────────────────────────────
def test_the_overlay_path_is_not_shadowed_by_skills_name(client):
    """``/api/skills/overlay`` **会**被 ``/skills/{name}`` 吃掉 —— 所以我们不用它。

    这条不是在测我们的路由，是在证明**那个坑真的存在**：一旦有人觉得
    ``/skill-overlay`` 难看、改回 ``/skills/overlay``，UI 就会静默拿到一张
    ``name="overlay"`` 的空技能卡片。
    """
    shadowed = client.get("/api/skills/overlay")
    assert shadowed.status_code == 200
    body = shadowed.json()
    assert body.get("name") == "overlay", (
        "这里本该被 /skills/{name} 捕获 —— 如果不再是，说明路由布局变了，"
        "可以重新考虑用 /skills/overlay，但要先确认没有别的通配捕获它")

    ours = client.get("/api/skill-overlay")
    assert ours.status_code == 200
    assert "overlay_dir" in ours.json(), "我们的端点没注册上"


def test_all_four_endpoints_are_registered(client):
    paths = {r.path for r in client.app.routes if hasattr(r, "path")}
    for p in ("/api/skill-overlay", "/api/skill-overlay/reload",
              "/api/skill-overlay/entry", "/api/skill-overlay/restore-all"):
        assert p in paths, f"{p} 没注册"


# ── 绝不 500 ────────────────────────────────────────────────────────
def test_status_works_without_a_live_runtime(client):
    r = client.get("/api/skill-overlay")
    assert r.status_code == 200
    d = r.json()
    assert d["degraded"] is False       # 读清单不需要 runtime
    assert d["entries"] == [] and d["untracked"] == []


def test_reload_without_a_runtime_degrades_instead_of_500(client):
    r = client.post("/api/skill-overlay/reload", json={"pin": "", "reason": "t"})
    assert r.status_code == 200
    d = r.json()
    assert d["degraded"] is True and d["ok"] is False
    assert "运行时" in d["reason"], "没说清为什么不行"


def test_restore_without_a_registry_degrades(client):
    r = client.post("/api/skill-overlay/restore-all", json={"pin": ""})
    assert r.status_code == 200
    assert r.json()["degraded"] is True


# ── 「判断不了」不是 False ──────────────────────────────────────────
def test_fingerprint_matches_is_none_when_no_tools_were_ever_built(client):
    """进程刚起、还没建过工具表 ⇒ **None**。

    报 False 会让 UI 显示「工具表未跟上」，而真相是「还没有工具表」——
    两件事的处理完全不同。这条纪律和 ``reload_wiring`` 那句
    「不知道就报 None，别报 False」是同一条。
    """
    d = client.get("/api/skill-overlay").json()
    assert d["fingerprint_matches"] is None


# ── 清单写入 ────────────────────────────────────────────────────────
def test_entry_write_derives_the_overlaid_module(client):
    r = client.post("/api/skill-overlay/entry",
                    json={"path": "builtins/bias.py", "enabled": True, "pin": ""})
    assert r.status_code == 200
    d = r.json()
    assert d["ok"] is True
    assert d["entry"]["overlay_of"] == "mast.skills.builtins.bias"
    assert d["entry"]["exists"] is False, "盘上没有这个文件，不能报成有"


@pytest.mark.parametrize("bad", ["../evil.py", "/abs/x.py", "x.txt",
                                 "builtins/2bad/x.py"])
def test_entry_refuses_bad_paths(client, bad):
    d = client.post("/api/skill-overlay/entry",
                    json={"path": bad, "enabled": True, "pin": ""}).json()
    assert d["ok"] is False and d["reason"]


def test_entry_write_shows_up_in_status(client):
    client.post("/api/skill-overlay/entry",
                json={"path": "builtins/bias.py", "enabled": True, "pin": ""})
    d = client.get("/api/skill-overlay").json()
    assert [e["path"] for e in d["entries"]] == ["builtins/bias.py"]
    assert d["entries"][0]["enabled"] is True
    assert d["entries"][0]["applied"] is False, (
        "写了清单 ≠ 生效了 —— 那需要一次重载")


def test_an_unreadable_manifest_does_not_get_overwritten(client, tmp_path):
    """清单读不出来时**拒绝写入** —— 否则这一次写会把已有条目全抹掉。

    「读不到」不能当成「空的」：前者意味着有人配了但我们没看懂，后者意味着还没人
    用过。当成空的处理，就会把用户显式启用的东西静默地全部停掉。
    """
    from mast.skills.overlay import paths as P

    d = P.overlay_dir(create=True)
    (d / P.MANIFEST_NAME).write_text("{ not json at all", encoding="utf-8")
    r = client.post("/api/skill-overlay/entry",
                    json={"path": "builtins/bias.py", "enabled": True, "pin": ""})
    assert r.json()["ok"] is False
    assert "读不出来" in r.json()["reason"]
    assert "{ not json" in (d / P.MANIFEST_NAME).read_text(encoding="utf-8"), (
        "坏清单被覆盖了 —— 用户原来的配置就这么没了")


# ── PIN ─────────────────────────────────────────────────────────────
def test_pin_gates_mutations_when_one_is_set(client, tmp_path, monkeypatch):
    """设了 PIN 就要验；**没设就放行**。

    后者是刻意的：这是台实验台仪器，把用户锁在自己的显微镜外面，比 PIN 想防的
    那件事更糟（``admin_pin.py`` 的威胁模型 —— 它防的是一只人手）。
    """
    from mast.api import admin_pin

    ok, _ = admin_pin.set_pin("2468")
    assert ok

    bad = client.post("/api/skill-overlay/entry",
                      json={"path": "builtins/bias.py", "enabled": True,
                            "pin": "0000"}).json()
    assert bad["ok"] is False and bad["reason"]

    good = client.post("/api/skill-overlay/entry",
                       json={"path": "builtins/bias.py", "enabled": True,
                             "pin": "2468"}).json()
    assert good["ok"] is True


def test_an_unreadable_pin_check_refuses_rather_than_letting_through(
        client, monkeypatch):
    """门读不出来 ⇒ **拒绝**，不是放行。

    「守卫不可用」当成「守卫通过」，正好把守卫变成摆设 —— 而且是在它最需要工作的
    那一刻（系统有别的地方坏了）。
    """
    def _boom():
        raise RuntimeError("pin store unreadable")

    monkeypatch.setattr("mast.api.admin_pin.pin_is_set", _boom)
    d = client.post("/api/skill-overlay/entry",
                    json={"path": "builtins/bias.py", "enabled": True,
                          "pin": "x"}).json()
    assert d["ok"] is False
    assert "拒绝" in d["reason"]


# ── 有 runtime 的那条路 ─────────────────────────────────────────────
def test_reload_with_a_runtime_reports_what_actually_happened(client, tmp_path):
    """真的走一遍重载，并断言响应**如实**转述了三条链的结果。

    上面那些测的都是 degraded 分支。只测降级的话，一个「永远 degraded」的实现
    也能全绿 —— 而那正是这个端点最没用的形态。
    """
    from mast.core.registry import SkillRegistry
    from mast.skills.overlay import loader

    loader.reset_for_tests()
    reg = SkillRegistry()

    class _FakeRuntime:
        _registry = reg

        def reload_overlay_skills(self, *, reason=""):
            rep = loader.reload_skills(reg, reason=reason)
            return {"status": rep.status, "summary": rep.describe(),
                    "applied": [r.rel for r in rep.applied],
                    "failed": [], "restored": [], "baseline_drift": [],
                    "refresh": {"described": "手动路径=已刷新；私聊图=下一回合重建",
                                "agent_path_pending": False}}

        def pending_agent_rebuild(self):
            return None

    client.app.state.ctx.live_app = _FakeRuntime()
    try:
        d = client.post("/api/skill-overlay/reload",
                        json={"pin": "", "reason": "test"}).json()
        assert d["degraded"] is False
        assert d["ok"] is True
        assert d["agent_path_pending"] is False
        assert "已刷新" in d["refresh"], "三条链的结果没转述出来"
    finally:
        client.app.state.ctx.live_app = None
        loader.reset_for_tests()


def test_a_queued_reload_is_never_reported_as_done(client):
    """任务运行中排队时，``agent_path_pending`` **必须**是 True。

    这是整个端点最重要的一条：``ok`` 可能是 True（没有失败的条目），但东西
    **一点都没换**。UI 按 ``ok`` 显示的话，用户会以为改动已经生效，然后基于
    那个假设去做下一件事。
    """
    class _BusyRuntime:
        _registry = None

        def reload_overlay_skills(self, *, reason=""):
            return {"status": "queued", "summary": "任务运行中，已排队",
                    "applied": [], "failed": [], "restored": [],
                    "baseline_drift": [], "refresh": {"queued": True}}

        def pending_agent_rebuild(self):
            return {"reason": "overlay reload"}

    client.app.state.ctx.live_app = _BusyRuntime()
    try:
        d = client.post("/api/skill-overlay/reload", json={"pin": ""}).json()
        assert d["status"] == "queued"
        assert d["agent_path_pending"] is True, (
            "排队被报成了「已跟上」—— 用户会以为改动生效了")
        assert "排队" in d["refresh"]
    finally:
        client.app.state.ctx.live_app = None
