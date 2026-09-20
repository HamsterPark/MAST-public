"""``/api/conducts/*`` —— 用户入口的契约。

设计 §7。这里钉的六件事,每一件都对应一个「合起来就出事」的区分:

1. **状态词表只有一份**:schema 的 Literal 与 ``store.STATUSES`` 逐字相等
   (「双端镜像」教训第三次 —— 一开始就配 parity 测试);
2. **引擎关着 ≠ 没有 conduct**:读端点 degraded + 一句为什么,写端点 503;
3. **超包络拒绝,不夹紧**:400 + 逐字段回显,而且**什么都没建**;
4. **「没检查」不许长得像「检查通过」**:注册表读不到 ⇒ approve 走不通;
5. **面板是一个端点**:一次响应 = 一个时刻,含 timeline / heartbeat / folder;
6. **abort 不等 tick**:立刻置 per-run Event,同时入队。

外加一条结构闸:**approve 永远不是 agent 工具**。

全程 ``tmp_path``:store 落 tmp 的库,实验文件夹用注入 resolver,**绝不碰真实
experiments**。
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

_MASTV2_ROOT = str(Path(__file__).resolve().parents[4] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)

from mast.api.context import AppContext  # noqa: E402
from mast.api.routes import conducts as conduct_routes  # noqa: E402
from mast.conduct import service as svc_mod  # noqa: E402
from mast.conduct.settings import get_conduct_knobs, set_conduct_knobs  # noqa: E402
from mast.conduct.store import OPS, STATUSES  # noqa: E402

SMOKE_PARAMS = {"target_temperature_k": 5.0,
                "temperature_stale_after_s": 600.0,
                "probe_positions_m": "1e-8,2e-8; -3e-8,0",
                "probe_points_max": 4}


class _Rt:
    def __init__(self, db_path, registry=None):
        self.config = type("C", (), {"db_path": db_path})()
        self._orch_abort = threading.Event()
        self._pool = object()
        self._state = object()
        self._registry = registry

    def emergency_latch_state(self):
        return {"latched": False, "abort_set": False, "why": ""}

    def latest_temperature(self, channel=None):
        from mast.core.temperature import NO_SOURCE, TempReading
        return TempReading(channel=str(channel or ""), reason=NO_SOURCE)


class _Bus:
    def publish_conduct_status(self, conduct_id, **kw):
        pass

    def publish_conduct_gate(self, conduct_id, **kw):
        pass

    def publish_conduct_alert(self, conduct_id, **kw):
        pass


def _real_registry():
    """真的注册表 —— approve 的规则③要它,而且要它是**真的**。

    传 ``skills=None`` 的话 ``validate_spec`` 会把规则③记进 checks_skipped,
    approvable 就是 False;那正是第 4 条要钉的东西,所以这里必须给一份真的。
    """
    from mast.core.registry import SkillRegistry

    reg = SkillRegistry()
    reg.discover("mast.skills.builtins", "mast.skills.composite")
    return reg


@pytest.fixture(autouse=True)
def _isolated():
    before = get_conduct_knobs()
    svc_mod.set_service_for_test(None)
    yield
    svc_mod.stop_service(drop=True)
    svc_mod.set_service_for_test(None)
    set_conduct_knobs(before)


def _client(ctx=None) -> TestClient:
    app = FastAPI()
    app.state.ctx = ctx or AppContext()
    app.include_router(conduct_routes.router, prefix="/api")
    return TestClient(app)


def _wire(tmp_path, *, registry=None):
    """开着的引擎 + 一个只往 tmp_path 写的 service。返回 (client, svc)。"""
    set_conduct_knobs({"cd_enabled": 1.0})
    db = tmp_path / "db" / "camp.db"
    svc = svc_mod.ConductService(_Rt(db, registry), db_path=db,
                                  folder_resolver=lambda _e: tmp_path / "exp",
                                  bus=_Bus())
    svc_mod.set_service_for_test(svc)
    ctx = AppContext()
    if registry is not None:
        ctx.wire(skill_registry=registry)
    return _client(ctx), svc


# ── 1. 词表 parity ────────────────────────────────────────────────

def test_status_and_op_vocabularies_match_the_store_exactly():
    """schema 的闭集 = store 的闭集。**逐字**,不是「差不多」。

    状态枚举同时住在后端 schema、store 和(下一轮的)前端面板里;三份手抄的
    名单,总有一份会漏。这条测试是那道闸的第一半。
    """
    import typing

    from mast.api.schemas_conduct import ConductStatus, OpName

    assert set(typing.get_args(ConductStatus)) == set(STATUSES)
    assert set(typing.get_args(OpName)) == set(OPS)


def test_approve_route_is_never_an_agent_tool():
    """结构闸:这个模块里不许有 ``@tool``,也不许 import agent 工具层。

    批准一份要跑三天、要动针的流程是**人**的动作。做成工具就等于把「要不要
    做这个实验」交给一次采样。
    """
    import re

    src = Path(conduct_routes.__file__).read_text(encoding="utf-8")
    # 装饰器**位置**匹配,不是子串匹配 —— 模块 docstring 里就写着「不许有 @tool」,
    # 一个天真的 `"@tool" not in src` 会被自己的说明文字绊倒。
    decorated = re.findall(r"(?m)^\s*@(?:tool|.*\.tool)\b", src)
    assert decorated == [], f"这个模块里出现了工具装饰器: {decorated}"
    assert "langchain_core.tools" not in src
    assert "mast.agents" not in src


# ── 2. 引擎关着 ───────────────────────────────────────────────────

def test_engine_off_reads_degraded_not_empty():
    set_conduct_knobs({"cd_enabled": 0.0})
    r = _client().get("/api/conducts")
    assert r.status_code == 200
    body = r.json()
    assert body["degraded"] is True and body["reason"]
    assert body["engine_enabled"] is False
    assert body["conducts"] == []


def test_engine_off_refuses_writes_with_503():
    set_conduct_knobs({"cd_enabled": 0.0})
    r = _client().post("/api/conducts", json={"spec_id": "_smoke_v1",
                                              "experiment_id": "e1"})
    assert r.status_code == 503
    assert r.json()["degraded"] is True


def test_engine_off_still_renders_the_panel_endpoint(tmp_path):
    """读端点的降级方向与写端点相反:**200 + degraded**,不是 5xx。

    用户点开一条旧链接时页面要渲染出「引擎未启用」——「任何点击最坏只能
    无反应/提示」是硬规则。写端点该 503 的照样 503:那是拒绝,拒绝要响。
    """
    set_conduct_knobs({"cd_enabled": 0.0})
    r = _client().get("/api/conducts/whatever")
    assert r.status_code == 200
    body = r.json()
    assert body["degraded"] is True and body["reason"]
    assert body["engine_enabled"] is False


# ── 3. 建:超包络拒绝,不夹紧 ─────────────────────────────────────

def test_out_of_envelope_param_is_refused_and_nothing_is_created(tmp_path):
    client, svc = _wire(tmp_path)
    bad = dict(SMOKE_PARAMS, target_temperature_k=9999.0)   # 上限 400 K
    r = client.post("/api/conducts", json={"spec_id": "_smoke_v1",
                                           "experiment_id": "e1",
                                           "params": bad})
    assert r.status_code == 400
    body = r.json()
    assert body["ok"] is False
    echo = {e["name"]: e for e in body["params_echo"]}
    assert echo["target_temperature_k"]["ok"] is False
    assert "拒绝,不夹紧" in echo["target_temperature_k"]["error"]
    assert echo["probe_points_max"]["ok"] is True, "别的字段不该被连坐"
    assert svc.store.list_conducts() == [], "拒了却还是建了一份"


def test_a_parameter_the_template_never_declared_is_echoed_back(tmp_path):
    """悄悄被忽略的参数,填的人会一直以为它生效了。"""
    client, _svc = _wire(tmp_path)
    r = client.post("/api/conducts", json={
        "spec_id": "_smoke_v1", "experiment_id": "e1",
        "params": dict(SMOKE_PARAMS, target_temp_k=5.0)})   # 名字拼错
    assert r.status_code == 400
    names = {e["name"] for e in r.json()["params_echo"]}
    assert "target_temp_k" in names


def test_create_then_second_create_is_409(tmp_path):
    client, _svc = _wire(tmp_path)
    body = {"spec_id": "_smoke_v1", "experiment_id": "e1",
            "params": SMOKE_PARAMS}
    first = client.post("/api/conducts", json=body)
    assert first.status_code == 201 and first.json()["status"] == "draft"
    assert [s["stage_id"] for s in first.json()["stages_summary"]] == ["SMOKE"]
    second = client.post("/api/conducts", json=body)
    assert second.status_code == 409, "单活跃不变式没拦住第二份"
    assert "单活跃" in second.json()["errors"][0]


def test_unknown_template_is_400_and_lists_what_exists(tmp_path):
    client, _svc = _wire(tmp_path)
    r = client.post("/api/conducts", json={"spec_id": "nope_v9",
                                           "experiment_id": "e1"})
    assert r.status_code == 400
    assert "_smoke_v1" in r.json()["errors"][0]


# ── 4. 批:「没检查」不许长得像「检查通过」 ──────────────────────

def test_approve_without_a_registry_is_refused_as_incomplete(tmp_path):
    """注册表读不到 ⇒ 规则③没跑 ⇒ **批不下去**。

    只看 ``ok`` 的话,一次「什么都没检查」会长得和「检查全过」一模一样。
    """
    client, svc = _wire(tmp_path)                  # registry=None
    client.post("/api/conducts", json={"spec_id": "_smoke_v1",
                                       "experiment_id": "e1",
                                       "params": SMOKE_PARAMS})
    cid = svc.store.active()["conduct_id"]
    r = client.post(f"/api/conducts/{cid}/approve", json={"approved_by": "用户"})
    assert r.status_code == 409
    body = r.json()
    assert body["validation_ok"] is True          # 没发现错误
    assert body["validation_complete"] is False   # 但**没检查全**
    assert body["checks_skipped"], "跳过了哪些检查要说出来"
    assert svc.store.get(cid)["status"] == "draft"


def test_approve_renders_a_human_snapshot_and_freezes_the_params(tmp_path):
    client, svc = _wire(tmp_path, registry=_real_registry())
    client.post("/api/conducts", json={"spec_id": "_smoke_v1",
                                       "experiment_id": "e1",
                                       "params": SMOKE_PARAMS})
    cid = svc.store.active()["conduct_id"]
    r = client.post(f"/api/conducts/{cid}/approve", json={"approved_by": "用户"})
    assert r.status_code == 200, r.json()
    body = r.json()
    assert body["ok"] and body["status"] == "approved"
    assert body["spec_doc_path"] == "spec_v001.md"
    assert body["validation_ok"] and body["validation_complete"]

    doc = tmp_path / "exp" / "conduct" / cid / "spec_v001.md"
    assert doc.is_file()
    text = doc.read_text(encoding="utf-8")
    assert "5.0" in text, "冻结的参数没进人读快照"
    # 绑定过的阈值印的是**引用**,不是模板里那个 400 的占位值。
    assert "params.target_temperature_k" in text
    assert "400" not in text.split("## SMOKE")[-1].split("条件")[-1][:80]

    row = svc.store.get(cid)
    assert row["approved_by"] == "用户" and row["approved_at"]


def test_approving_twice_is_409(tmp_path):
    client, svc = _wire(tmp_path, registry=_real_registry())
    client.post("/api/conducts", json={"spec_id": "_smoke_v1",
                                       "experiment_id": "e1",
                                       "params": SMOKE_PARAMS})
    cid = svc.store.active()["conduct_id"]
    client.post(f"/api/conducts/{cid}/approve", json={"approved_by": "a"})
    again = client.post(f"/api/conducts/{cid}/approve", json={"approved_by": "b"})
    assert again.status_code == 409


def test_templates_endpoint_says_which_ones_are_approvable_here(tmp_path):
    """下拉框里不该有「看起来能跑、批的时候才炸」的模板。

    所以这个端点当场把校验器跑一遍:``approvable`` 回答的是「**这台机器上**
    批不批得下去」,不是「结构立不立得住」。
    """
    client, _svc = _wire(tmp_path, registry=_real_registry())
    rows = {t["spec_id"]: t
            for t in client.get("/api/conducts/templates").json()["templates"]}
    assert rows["_smoke_v1"]["approvable"] is True
    assert rows["_smoke_v1"]["spec_version"] >= 1
    assert rows["_smoke_v1"]["stages"] == ["SMOKE"]


def test_templates_without_a_registry_are_not_approvable(tmp_path):
    """注册表读不到 ⇒ 规则③没跑 ⇒ ``approvable=False`` + 说清跳过了什么。

    这是「没检查」与「检查通过」在下拉框这一层的分界。
    """
    client, _svc = _wire(tmp_path)                 # registry=None
    rows = {t["spec_id"]: t
            for t in client.get("/api/conducts/templates").json()["templates"]}
    for row in rows.values():
        assert row["approvable"] is False
        assert row["checks_skipped"]


def test_templates_carry_what_the_form_has_to_render(tmp_path):
    """下拉选中一个模板之后,表单要渲染的东西全在这一回里。

    M1-e 之前这个端点只回名字与批不批得下去,于是「新建一份 conduct」在界面上
    没有任何入口 —— 只剩 curl。**做不到的动作和不存在的动作,在屏幕上长得一模
    一样**(与 approve 只接受 UI 来源那条是同一个形状)。
    """
    from mast.conduct.templates import get_template

    client, _svc = _wire(tmp_path, registry=_real_registry())
    rows = {t["spec_id"]: t
            for t in client.get("/api/conducts/templates").json()["templates"]}
    got = {p["name"]: p for p in rows["_smoke_v1"]["params_schema"]}
    want = {p.name: p for p in get_template("_smoke_v1").params_schema}
    assert set(got) == set(want), "表单渲染的格子与模板声明的参数对不上"
    for name, p in want.items():
        assert got[name]["type"] == p.type
        assert got[name]["unit"] == (p.unit or "")
        # 包络必须过河:超包络**拒绝不夹紧**,而表单要在人按下建立**之前**
        # 就把允许区间摆出来(1.5 nA 填成 1.5 安培那次,缺的正是这句话)。
        assert got[name]["min_value"] == p.min_value
        assert got[name]["max_value"] == p.max_value
        assert got[name]["help"] == (p.help or "")


def test_the_form_contract_cannot_carry_a_prefilled_value(tmp_path):
    """``ParamSpecRow`` **没有** ``default`` 这个字段 —— 拿不到就不会预填。

    模板的 ``ParamSpec.default`` 全是 ``None`` 且刻意如此:填不出来就说明这次
    实验的工作点还没定,那正是该停下来的时候。这条钉的是**契约本身**,不是
    前端的自律:渲染件根本收不到那个数,「别预填」就不再依赖谁记得住。
    (``sts_condition`` 是最刺眼的一例 —— 出厂只有一个 ``default`` 组,
    而它刻意是未标定的。)
    """
    from mast.api.schemas_conduct import ParamSpecRow

    assert "default" not in ParamSpecRow.model_fields, (
        "表单契约长出了 default —— 表单会预填一个没人定过的工作点")

    client, _svc = _wire(tmp_path, registry=_real_registry())
    for row in client.get("/api/conducts/templates").json()["templates"]:
        for p in row["params_schema"]:
            assert "default" not in p
            # 必填与否是**派生**的,与 check_params 判必填用同一个条件
            # (``p.default is None``),不是另写一遍。
            assert p["required"] is True


@pytest.mark.parametrize("literal,marker", [("templates", "templates"),
                                            ("config", "knobs")])
def test_literal_paths_are_not_captured_as_a_conduct_id(tmp_path, literal, marker):
    """路由遮蔽:``/conduct/templates`` 与 ``/conduct/config`` 不能被
    ``/conduct/{id}`` 吃掉。

    本仓踩过两次,两次都是 200 + 错的 handler —— 页面不报错,只是永远空着。
    """
    client, _svc = _wire(tmp_path)
    body = client.get(f"/api/conducts/{literal}").json()
    assert marker in body and "conduct_id" not in body


def test_config_separates_the_switch_from_the_thread(tmp_path):
    """「开关开着」与「线程活着」是两件事。合成一个数,「设了但没生效」就看不见了。"""
    client, svc = _wire(tmp_path)          # 开关开着,但这里没起线程
    body = client.get("/api/conducts/config").json()
    assert body["enabled"] is True
    assert body["director_running"] is False
    # 目录与 settings 那一侧对账，不在这里抄一份名单：抄一份的话，加旋钮时
    # 两处会各改各的，而「设置页少了一个开关」不会有任何东西报错。
    from mast.conduct.settings import EDITABLE_KEYS

    keys = {k["key"] for k in body["knobs"]}
    assert keys == set(EDITABLE_KEYS)
    assert {"cd_enabled", "cd_autonomy"} <= keys, (
        "开关与自主度是这一层最要紧的两个旋钮，少哪个都会让用户配不出想要的行为")
    # 字段名与别的旋钮目录逐字一致 —— 前端那份共用渲染件认的是这几个名字。
    one = body["knobs"][0]
    assert {"label_zh", "hint_zh", "is_bool", "step", "default", "value"} <= set(one)

    # ── 三档的档位名从这里出去（2026-08-27） ──────────────────────
    # `cd_autonomy` 是 0/1/2 三档，不是连续量。档位名的**唯一真源**是
    # `autonomy.AUTONOMY_LEVELS` + `describe()`；前端拿到什么就显示什么。
    # 在那边再写一份中文档位名，两处岔开时不会有任何东西报错 —— 界面上写着
    # 「半自主」，存进去的却是别的档。
    from mast.conduct.autonomy import AUTONOMY_LEVELS, describe

    by_key = {k["key"]: k for k in body["knobs"]}
    choices = by_key["cd_autonomy"]["choices"]
    assert [c["value"] for c in choices] == [float(i)
                                             for i in range(len(AUTONOMY_LEVELS))]
    assert [c["label_zh"] for c in choices] == [describe(l) for l in AUTONOMY_LEVELS]
    # 连续量不许带档位 —— 带了的话前端会把一个范围渲染成几个按钮。
    for key in ("cd_enabled", "cd_stall_grace_s", "cd_ignition_delay_s"):
        assert by_key[key]["choices"] == [], key


# ── 5. 面板:一个端点,一个时刻 ──────────────────────────────────

def test_detail_is_the_whole_panel_in_one_response(tmp_path):
    client, svc = _wire(tmp_path, registry=_real_registry())
    client.post("/api/conducts", json={"spec_id": "_smoke_v1",
                                       "experiment_id": "e1",
                                       "params": SMOKE_PARAMS})
    cid = svc.store.active()["conduct_id"]
    client.post(f"/api/conducts/{cid}/approve", json={"approved_by": "用户"})

    body = client.get(f"/api/conducts/{cid}").json()
    assert body["ok"] and body["status"] == "approved"
    assert body["title"] and body["spec_id"] == "_smoke_v1"
    assert body["timeline"] and body["timeline"][0]["stage_id"] == "SMOKE"
    assert body["timeline"][0]["steps_total"] >= 4
    assert body["heartbeat"]["threshold_s"] > 0
    assert body["folder"]["spec_doc"] == "spec_v001.md"
    assert body["folder"]["progress_lines"] >= 2
    assert body["params"]["target_temperature_k"] == 5.0


def test_a_long_step_is_not_reported_as_a_stall(tmp_path):
    """在步里的时候心跳**本来就不更新** —— 那是设计,不是故障。

    合成一支判据,一次正常的两小时扫描会天天报警;报警报久了就没人看,
    那时真的停滞发生了也一样没人看。
    """
    client, svc = _wire(tmp_path)
    cid = _draft(client, svc)
    now = svc.store.now_epoch()
    svc.store.record(cid, "adopted", changes={"status": "approved"})
    svc.store.record(cid, "adopted", changes={"status": "running"})
    svc.store.touch_heartbeat(cid)
    # 心跳写在 20 分钟前,而这一步刚开始 1 分钟。
    svc.store.record(cid, "step_started", run_id="r1",
                     changes={"active_run_id": "r1"},
                     payload={"at": now - 60.0})
    import sqlite3
    with sqlite3.connect(str(tmp_path / "db" / "camp.db")) as conn:
        conn.execute("UPDATE conducts SET heartbeat_at=? WHERE conduct_id=?",
                     (now - 1200.0, cid))

    hb = client.get(f"/api/conducts/{cid}").json()["heartbeat"]
    assert hb["in_step"] is True
    assert hb["age_s"] > 1000, "心跳确实很旧 —— 这条测试要的就是这个前提"
    assert hb["stalled"] is False, "把「正常长步」报成了停滞"
    assert hb["step_elapsed_s"] < 120


def test_the_step_start_time_comes_from_this_run_not_an_old_one(tmp_path):
    """跑过很多步之后,「当前步什么时候开始的」必须还是**当前**那一步。

    审计流是升序 + LIMIT:「取前 N 条 step_started 再 reverse」在第 N 步之后
    问到的是很早那一步的时刻 —— 而停滞判据拿它一减,一个刚开始的步立刻被判成
    停滞了几小时。
    """
    client, svc = _wire(tmp_path)
    cid = _draft(client, svc)
    svc.store.record(cid, "adopted", changes={"status": "approved"})
    svc.store.record(cid, "adopted", changes={"status": "running"})
    now = svc.store.now_epoch()
    for i in range(60):                       # 远超任何「取前 50 条」的窗口
        svc.store.record(cid, "step_started", run_id=f"old-{i}",
                         payload={"at": now - 100000.0})
    svc.store.record(cid, "step_started", run_id="current",
                     changes={"active_run_id": "current"},
                     payload={"at": now - 30.0})
    body = client.get(f"/api/conducts/{cid}").json()
    assert body["step"]["started_at"] == pytest.approx(now - 30.0), (
        "拿到的是一个很早的 step_started —— 当前步会被判成跑了一整天")
    assert body["heartbeat"]["step_elapsed_s"] < 120


def test_no_running_step_means_no_start_time_rather_than_an_old_one(tmp_path):
    """不在步里 ⇒ 「当前步的开始时刻」本来就不存在,留空而不是找个旧的顶上。"""
    client, svc = _wire(tmp_path)
    cid = _draft(client, svc)
    svc.store.record(cid, "step_started", run_id="finished",
                     payload={"at": svc.store.now_epoch() - 5000.0})
    assert client.get(f"/api/conducts/{cid}").json()["step"]["started_at"] is None


def test_a_driven_conduct_with_no_heartbeat_at_all_is_flagged(tmp_path):
    """处在驱动态却一次心跳都没有 ⇒ **报出来**。

    「没有读数」在这里不是「没问题」:多半是引擎关着而库里留着一行 RUNNING。
    """
    client, svc = _wire(tmp_path)
    cid = _draft(client, svc)
    svc.store.record(cid, "adopted", changes={"status": "approved"})
    svc.store.record(cid, "adopted", changes={"status": "running"})
    hb = client.get(f"/api/conducts/{cid}").json()["heartbeat"]
    assert hb["at"] is None and hb["stalled"] is True
    assert "从来没碰过它" in hb["reason"]


def test_a_paused_conduct_is_not_a_stall(tmp_path):
    """人按了暂停 ⇒ 心跳与停滞无关,别拿告警去烦他。"""
    client, svc = _wire(tmp_path)
    cid = _draft(client, svc)
    svc.store.record(cid, "status_change",
                     changes={"status": "paused", "status_reason": "人按了暂停"})
    hb = client.get(f"/api/conducts/{cid}").json()["heartbeat"]
    assert hb["stalled"] is False and hb["reason"]


def test_budget_says_unreadable_rather_than_zero(tmp_path):
    """空着的预算条会被读成「没花钱」。读不到就要说读不到。

    ⚠️ 探针注入:不注入就会去读 ``experiments/usage_ledger.sqlite`` ——
    **用户真实的账本**。只读不写,不是污染,但会让这条测试的结果取决于那台
    机器上那个文件里有什么。
    """
    client, svc = _wire(tmp_path)
    svc._cost_probe = lambda _cid: None          # 读不到
    client.post("/api/conducts", json={"spec_id": "_smoke_v1",
                                       "experiment_id": "e1",
                                       "params": SMOKE_PARAMS})
    cid = svc.store.active()["conduct_id"]
    b = client.get(f"/api/conducts/{cid}").json()
    assert b["budget"]["spent_usd"] is None
    assert b["budget"]["reason"]
    assert "USD 花销" in b["not_available"]


def test_unknown_conduct_is_404_not_an_empty_panel(tmp_path):
    client, _svc = _wire(tmp_path)
    r = client.get("/api/conducts/deadbeef")
    assert r.status_code == 404 and r.json()["ok"] is False


def test_a_missing_template_is_reported_not_rendered_as_an_empty_conduct(tmp_path):
    """模板不在本机 ⇒ 面板说「算不出来」,而不是显示一份 0 阶段的 conduct。"""
    client, svc = _wire(tmp_path)
    cid = svc.store.create(experiment_id="e1", spec_id="ghost_v1", spec_version=1)
    b = client.get(f"/api/conducts/{cid}").json()
    assert b["timeline"] == []
    assert any("ghost_v1" in s for s in b["not_available"])


# ── 6. 意图 ───────────────────────────────────────────────────────

def _draft(client, svc):
    client.post("/api/conducts", json={"spec_id": "_smoke_v1",
                                       "experiment_id": "e1",
                                       "params": SMOKE_PARAMS})
    return svc.store.active()["conduct_id"]


def test_abort_signals_the_running_step_and_queues_the_intent(tmp_path):
    client, svc = _wire(tmp_path)
    cid = _draft(client, svc)
    ev = svc.aborts.register("run-1")
    svc.store.record(cid, "step_started", run_id="run-1",
                     changes={"active_run_id": "run-1"})
    r = client.post(f"/api/conducts/{cid}/abort",
                    json={"by": "用户", "reason": "针不对了"})
    assert r.status_code == 200
    body = r.json()
    assert body["queued"] and body["abort_signalled"] is True
    assert ev.is_set(), "Director 卡在长步里时,abort 必须不经 tick 就生效"
    assert [o["op"] for o in svc.store.pending_ops(cid)] == ["abort"]


def test_abort_without_a_reason_is_refused_by_the_schema(tmp_path):
    client, svc = _wire(tmp_path)
    cid = _draft(client, svc)
    r = client.post(f"/api/conducts/{cid}/abort", json={"by": "用户"})
    assert r.status_code == 422, "没有理由的中止,事后没人答得上为什么"


def test_pause_only_queues_and_never_touches_state(tmp_path):
    """状态的单写者是 Director。API 只写意图。"""
    client, svc = _wire(tmp_path)
    cid = _draft(client, svc)
    before = svc.store.get(cid)["status"]
    r = client.post(f"/api/conducts/{cid}/pause", json={"by": "用户"})
    assert r.status_code == 200 and r.json()["queued"]
    assert svc.store.get(cid)["status"] == before
    assert [o["op"] for o in svc.store.pending_ops(cid)] == ["pause"]


def test_ack_against_an_old_wait_point_is_409(tmp_path):
    """对旧等待点的确认要当场认出来。

    否则用户按了按钮、什么都没发生,还得去翻审计流才知道按错了地方。
    """
    client, svc = _wire(tmp_path)
    cid = _draft(client, svc)
    svc.store.record(cid, "wait_entered", changes={
        "status": "waiting_operator", "status_reason": "等人",
        "active_wait": {"wait_id": "w2", "kind": "operator",
                        "ack_required": True, "message": "请来一下"}})
    r = client.post(f"/api/conducts/{cid}/ack",
                    json={"wait_id": "w1", "by": "用户"})
    assert r.status_code == 409
    assert "w1" in r.json()["errors"][0]
    assert svc.store.pending_ops(cid) == []


def test_ack_echoes_which_gate_is_still_missing(tmp_path):
    client, svc = _wire(tmp_path)
    cid = _draft(client, svc)
    svc.store.record(cid, "wait_entered", changes={
        "status": "waiting_operator", "status_reason": "等人",
        "active_wait": {"wait_id": "w1", "kind": "operator",
                        "ack_required": True, "message": "请来一下"}})
    r = client.post(f"/api/conducts/{cid}/ack",
                    json={"wait_id": "w1", "by": "用户"})
    assert r.status_code == 200
    # ack 下一 tick 才生效,所以这一刻人的确认仍然「还缺」—— 如实回。
    assert r.json()["wait_echo"] == ["人的确认"]


def test_waive_requires_a_reason(tmp_path):
    """waive 不是默默放行:面板会持续显示这个标记,所以理由必填。"""
    client, svc = _wire(tmp_path)
    cid = _draft(client, svc)
    r = client.post(f"/api/conducts/{cid}/waive-condition",
                    json={"wait_id": "w1", "by": "用户"})
    assert r.status_code == 422


def test_ops_on_an_unknown_conduct_are_404(tmp_path):
    client, _svc = _wire(tmp_path)
    r = client.post("/api/conducts/nope/pause", json={"by": "用户"})
    assert r.status_code == 404


# ── 7. 判决留痕到得了人眼前(M3-a)──────────────────────────────────

def test_the_panel_says_which_model_answered_a_gate(tmp_path):
    """一条 LLM 判决与一条 rule 判定在闸门史上不能长得一样。

    面板是用户唯一会看的地方,而这两者事后要做的核对完全不同:一个查判据,
    一个查那次判决本身(哪个模型答的、怎么解析出来的、这一段的唤醒预算还剩
    几次)。``model`` 记的是**真的答了这一题的那个**,不是配置里写的 ——
    静默回退到另一家 provider 在本仓发生过,事后从配置根本看不出来。
    """
    client, svc = _wire(tmp_path)
    cid = _draft(client, svc)
    svc.store.record(cid, "gate_evaluated", stage_id="S", payload={
        "gate_id": "lg", "which": "exit", "kind": "llm", "verdict": "pass",
        "route": "go", "reason": "两个偏压都有分辨", "escaped": False,
        "llm": {"model": "kimi-k3", "parse_path": "structured",
                "duration_ms": 812, "wakes_used": 1, "wakes_max": 1,
                "responsibility": "判这一段还值不值得接着测"}})
    row = client.get(f"/api/conducts/{cid}").json()["gates_history"][-1]
    assert row["kind"] == "llm"
    assert row["llm_model"] == "kimi-k3"
    assert row["llm_parse_path"] == "structured"
    assert row["llm_wakes_used"] == 1 and row["llm_wakes_max"] == 1
    assert row["llm_unavailable"] == ""


def test_a_gate_that_could_not_be_judged_says_so_on_the_panel(tmp_path):
    """判决**没发生**(超时/建不出模型/返回值不在闭集里)必须在面板上说出来。

    折进一个空白格,它就与「判了,是 pass」之外的任何东西都分不开。
    """
    client, svc = _wire(tmp_path)
    cid = _draft(client, svc)
    svc.store.record(cid, "gate_evaluated", stage_id="S", payload={
        "gate_id": "lg", "kind": "llm", "verdict": "wait_operator", "route": "",
        "reason": "llm 判决调用失败 ⇒ 判不了: 超过 60 s 还没回来", "escaped": True,
        "llm": {"unavailable": "LLM 判决超过 60 s 还没回来 —— 判不了"}})
    row = client.get(f"/api/conducts/{cid}").json()["gates_history"][-1]
    assert row["kind"] == "llm" and row["escaped"] is True
    assert "60 s" in row["llm_unavailable"]
    assert row["llm_model"] == ""          # 没答上来就没有模型名,不编一个
    assert row["llm_wakes_used"] is None   # 读不到 ≠ 0


def test_a_rule_gate_row_is_not_dressed_up_as_an_llm_one(tmp_path):
    client, svc = _wire(tmp_path)
    cid = _draft(client, svc)
    svc.store.record(cid, "gate_evaluated", stage_id="S", payload={
        "gate_id": "rg", "kind": "rule", "verdict": "pass", "route": "pass",
        "reason": "rule=true", "rule_state": "true"})
    row = client.get(f"/api/conducts/{cid}").json()["gates_history"][-1]
    assert row["kind"] == "rule"
    assert row["llm_model"] == "" and row["llm_wakes_max"] is None


# ── 8. 预算条必须承认自己拦不住(M3-d)────────────────────────────────

def test_the_panel_says_the_usd_cap_is_not_enforceable(tmp_path):
    """**装得像在拦的闸比没有闸更危险** —— 看的人会据此放心。

    面板上一个 `$0.00 / $20.00` 的进度条读起来就是「花了一点点,离上限还远」,
    而真相是没有任何东西会因为它停下来:账本按 provider 原生币种实测,上限是
    USD,合并需要汇率而本仓不自造汇率。
    """
    from mast.conduct.journal import USD_MAX_NOT_ENFORCEABLE

    from mast.conduct.adapters import ConductCost

    client, svc = _wire(tmp_path)
    cid = _draft(client, svc)
    # 真实配置下的样子:判决模型按原生币种计 CNY。
    svc._cost_probe = lambda _cid: ConductCost(
        {"CNY": 0.03}, count=2, all_priced=True, reason="")
    b = client.get(f"/api/conducts/{cid}").json()["budget"]
    assert b["enforceable"] is False
    assert b["not_enforceable_why"] == USD_MAX_NOT_ENFORCEABLE
    assert b["measured_by_currency"] == {"CNY": 0.03}
    assert b["spent_usd"] is None, "CNY 花销被当成了一个 USD 数"


def test_a_conduct_that_spent_nothing_yet_does_not_claim_the_cap_works(tmp_path):
    """⚠️ **「拦不拦得住」不看这一刻花了多少。**

    第一版把 ``enforceable`` 写成「这一刻读到了一个 USD 数」—— 于是一份还没花过
    钱的 conduct(读数 0.0)会在面板上宣称这条上限**有效**,而它下一次判决就
    可能花出一笔 CNY,同一条上限又变回无效。上限的可执行性是**结构性质**。
    """
    from mast.conduct.adapters import ConductCost

    client, svc = _wire(tmp_path)
    cid = _draft(client, svc)
    svc._cost_probe = lambda _cid: ConductCost({}, count=0, all_priced=True,
                                                reason="还没有任何一条调用")
    b = client.get(f"/api/conducts/{cid}").json()["budget"]
    assert b["enforceable"] is False, "零花销被当成了「上限有效」"


def test_an_all_usd_conduct_is_the_one_case_the_cap_can_be_enforced(tmp_path):
    """判据自我纠正:哪天口径真的统一成 USD,这里不用改就会变 True。"""
    from mast.conduct.adapters import ConductCost

    client, svc = _wire(tmp_path)
    cid = _draft(client, svc)
    svc._cost_probe = lambda _cid: ConductCost({"USD": 1.25}, count=3,
                                                all_priced=True, reason="")
    b = client.get(f"/api/conducts/{cid}").json()["budget"]
    assert b["enforceable"] is True and b["not_enforceable_why"] == ""
    assert b["spent_usd"] == 1.25


def test_a_per_currency_reading_reaches_the_panel_without_a_total(tmp_path):
    """逐币种读数照实上面板,**没有合计**。"""
    from mast.conduct.adapters import ConductCost

    client, svc = _wire(tmp_path)
    cid = _draft(client, svc)
    svc._cost_probe = lambda _cid: ConductCost(
        {"CNY": 3.5}, count=2, all_priced=True, reason="")
    b = client.get(f"/api/conducts/{cid}").json()["budget"]
    assert b["measured_by_currency"] == {"CNY": 3.5}
    assert b["spent_usd"] is None          # 没有把 CNY 折成 USD
    assert b["enforceable"] is False


# ── 7. plan → conduct:干跑端点、编译建、approve 实测冻结 ────────────
#
# 断链在这里接上:方案(人/agent 起草)→ 编译器 → 建草稿 → approve。
# 三处各钉一条「合起来就出事」的区分:
#   · 干跑**什么都不建**,而且「编不动」是 200 不是 4xx —— 它是一个答上来的答案;
#   · plan 库没接上是 503 + degraded —— 那是**读不到**,不是「这份方案没问题」;
#   · approve 那一刻**实测**冻结 coord_epoch,读不到就**拒批**(读不到 ≠ 0)。

WO_PARAMS = {
    "setpoint_a": 5e-11, "bias_series_v": "-1.0,-0.5,0.5,1.0",
    "frame_size_m": 5e-9, "sts_center_x_m": 0.0, "sts_center_y_m": 0.0,
    "ledger_dir": "ledger", "sts_positions_m": "1e-8,2e-8; -3e-8,0",
    "sts_points_n": 8, "sts_condition": "default", "coord_epoch": 0,
    "conditioning_sample": "修针样品", "measurement_sample": "测量样品",
    "s1_wait_temperature_k": 5.0, "s1_wait_temp_hold_s": 600.0,
    "s1_wait_temp_stale_after_s": 900.0,
}


def _plans(tmp_path):
    """只往 ``tmp_path`` 写的 PlanStore(**绝不碰真实实验库**)。"""
    from mast.planning.plan_store import PlanStore

    return PlanStore(tmp_path / "plans.db", plans_dir=tmp_path / "plans")


def _wire_plans(client, store):
    """把 plan 库挂到 ctx 上 —— 路由读的是 ``ctx.live_app._plan_store``。"""
    client.app.state.ctx.live_app = type("_Live", (), {"_plan_store": store})()
    return store


def _approved_plan(store, *, plan_id="p1", block=None):
    import json as _json

    from mast.planning.plan_store import ExperimentPlan, PlanStatus

    store.save(ExperimentPlan(
        plan_id=plan_id, experiment_id="", name="方案", goal="接线",
        status=PlanStatus.APPROVED,
        notes=_json.dumps({"conduct": block}, ensure_ascii=False)))
    return plan_id


def test_compile_dry_run_without_a_plan_store_is_503_not_an_empty_ok(tmp_path):
    """plan 库没接上 = **读不到**,不是「这份方案没问题但空空如也」。"""
    client, _svc = _wire(tmp_path, registry=_real_registry())
    r = client.post("/api/conducts/compile-from-plan/p1")
    assert r.status_code == 503
    body = r.json()
    assert body["degraded"] is True and "plan 库" in body["reason"]
    assert body["ok"] is False


def test_compile_dry_run_reports_each_unfilled_slot_and_builds_nothing(tmp_path):
    """编不动照样是 200 —— 它是一个**答上来了的答案**。而且什么都没建。"""
    client, svc = _wire(tmp_path, registry=_real_registry())
    plans = _wire_plans(client, _plans(tmp_path))
    _approved_plan(plans, block={"template": "_smoke_v1", "params": {}})

    r = client.post("/api/conducts/compile-from-plan/p1")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert [e["code"] for e in body["errors"]] == ["SLOT_UNFILLED"] * 4
    assert sorted(e["slot_name"] for e in body["errors"]) == sorted(SMOKE_PARAMS)
    # 编不动时一屏错误码也要有落点 —— 所以照样说得出是哪份模板、要填什么。
    assert body["spec_id"] == "_smoke_v1" and body["title"]
    assert body["stages_summary"], "表单还要渲染阶段概览"
    assert svc.store.list_conducts() == [], "干跑建了东西"


def test_compile_dry_run_passes_and_echoes_the_params(tmp_path):
    client, _svc = _wire(tmp_path, registry=_real_registry())
    plans = _wire_plans(client, _plans(tmp_path))
    _approved_plan(plans, block={"template": "_smoke_v1",
                                 "params": dict(SMOKE_PARAMS)})
    body = client.post("/api/conducts/compile-from-plan/p1").json()
    assert body["ok"] is True and body["errors"] == []
    assert body["params"]["probe_points_max"] == 4
    echo = {e["name"]: e for e in body["params_echo"]}
    assert all(e["ok"] for e in echo.values())


def test_creating_from_a_plan_freezes_the_compiled_params(tmp_path):
    """``from_plan_id`` + 空 params ⇒ 走编译;``created_by`` 记的是编译器。

    事后要分得清这份 conduct 是**谁装配的** —— 一个人手填的和一个从方案编出来
    的,出问题时要去查的地方不一样。
    """
    client, svc = _wire(tmp_path, registry=_real_registry())
    plans = _wire_plans(client, _plans(tmp_path))
    _approved_plan(plans, block={"template": "_smoke_v1",
                                 "params": dict(SMOKE_PARAMS)})

    r = client.post("/api/conducts", json={"experiment_id": "e1",
                                           "from_plan_id": "p1"})
    assert r.status_code == 201, r.json()
    cid = r.json()["conduct_id"]
    row = svc.store.get(cid)
    assert row["spec_id"] == "_smoke_v1"
    assert row["params"] == SMOKE_PARAMS
    created = svc.store.events(cid, kind="created")[0]["payload"]
    assert created["created_by"] == "compiler:plan=p1"


def test_creating_from_an_uncompilable_plan_is_4xx_with_closed_codes(tmp_path):
    """编译失败回 4xx **带闭集码**,不是 500,也不是一段红字。

    界面要按码分支(``SLOT_UNFILLED`` → 去方案页填这个槽)。拍扁成
    ``errors: list[str]`` 的话,人只看见一段红字,而每一码要做的事不同。
    """
    client, svc = _wire(tmp_path, registry=_real_registry())
    plans = _wire_plans(client, _plans(tmp_path))
    _approved_plan(plans, block={"template": "_smoke_v1", "params": {}})

    r = client.post("/api/conducts", json={"experiment_id": "e1",
                                           "from_plan_id": "p1"})
    assert r.status_code == 400
    body = r.json()
    assert {e["code"] for e in body["compile_errors"]} == {"SLOT_UNFILLED"}
    assert svc.store.list_conducts() == []


def test_creating_from_a_missing_plan_is_404(tmp_path):
    client, _svc = _wire(tmp_path, registry=_real_registry())
    _wire_plans(client, _plans(tmp_path))
    r = client.post("/api/conducts", json={"experiment_id": "e1",
                                           "from_plan_id": "nope"})
    assert r.status_code == 404
    assert [e["code"] for e in r.json()["compile_errors"]] == ["PLAN_NOT_FOUND"]


def test_two_sources_for_the_template_are_refused_not_silently_picked(tmp_path):
    """``spec_id`` 与编出来的模板不一致 ⇒ 400。**明拒优于静默挑一个。**"""
    client, _svc = _wire(tmp_path, registry=_real_registry())
    plans = _wire_plans(client, _plans(tmp_path))
    _approved_plan(plans, block={"template": "_smoke_v1",
                                 "params": dict(SMOKE_PARAMS)})
    r = client.post("/api/conducts", json={"experiment_id": "e1",
                                           "spec_id": "synthetic_sample_v1",
                                           "from_plan_id": "p1"})
    assert r.status_code == 400
    assert "不一致" in r.json()["errors"][0]


def test_a_create_without_spec_id_or_plan_says_so(tmp_path):
    """``spec_id`` 由必填改成选填之后,空的那条路要**说得出话**,不是 500。"""
    client, _svc = _wire(tmp_path, registry=_real_registry())
    r = client.post("/api/conducts", json={"experiment_id": "e1"})
    assert r.status_code == 400
    assert "spec_id 必填" in r.json()["errors"][0]


# ── approve 实测冻结 coord_epoch(治 D-5 残留)────────────────────────

def _wo_draft(client, svc):
    r = client.post("/api/conducts", json={"spec_id": "synthetic_sample_v1",
                                           "experiment_id": "e1",
                                           "params": dict(WO_PARAMS)})
    assert r.status_code == 201, r.json()
    return r.json()["conduct_id"]


def test_a_template_without_the_slot_is_untouched_by_the_freeze(tmp_path,
                                                                monkeypatch):
    """没声明 ``coord_epoch`` 的模板照旧 approve —— 即使代次根本读不到。

    这条是回归闸:冻结一旦无条件跑,**每一台没有活动实验记录的机器**上的每一次
    approve 都会 409,而那与这个槽毫无关系。
    """
    import mast.core.coord_epoch as ce

    monkeypatch.setattr(ce, "read_current_epoch", lambda: None)
    client, svc = _wire(tmp_path, registry=_real_registry())
    cid = _draft(client, svc)
    r = client.post(f"/api/conducts/{cid}/approve", json={"approved_by": "用户"})
    assert r.status_code == 200, r.json()
    assert r.json()["frozen_params"] == {}
    assert svc.store.get(cid)["status"] == "approved"


# ── 撤销窗从 approve 响应一路走到面板（2026-08-27） ──────────────────

def test_a_human_approval_is_not_deferred(tmp_path):
    """人批准在任何一档都立刻点火 —— 人已经在场，撤销窗防的正是「不在场」。"""
    client, svc = _wire(tmp_path, registry=_real_registry())
    client.post("/api/conducts", json={"spec_id": "_smoke_v1",
                                       "experiment_id": "e1",
                                       "params": SMOKE_PARAMS})
    cid = svc.store.active()["conduct_id"]
    body = client.post(f"/api/conducts/{cid}/approve",
                       json={"approved_by": "用户"}).json()
    assert body["deferred"] is False and body["ignition_delay_s"] == 0.0
    # 面板上也不该出现倒计时。
    assert client.get(f"/api/conducts/{cid}").json()["ignition"] is None


def test_an_agent_approval_under_supervised_is_deferred_and_visible(tmp_path):
    """响应里此前只有 ``ok=True``：「批了就跑」与「批了、十分钟后跑、这段时间
    能撤回」在调用方眼里长得一模一样 —— 而后者正是这一档存在的理由。"""
    from mast.conduct.settings import set_conduct_knobs

    client, svc = _wire(tmp_path, registry=_real_registry())
    set_conduct_knobs({"cd_enabled": 1.0, "cd_autonomy": 1.0})   # supervised
    client.post("/api/conducts", json={"spec_id": "_smoke_v1",
                                       "experiment_id": "e1",
                                       "params": SMOKE_PARAMS})
    cid = svc.store.active()["conduct_id"]
    body = client.post(f"/api/conducts/{cid}/approve",
                       json={"approved_by": "agent:experiment_design"}).json()
    assert body["ok"] is True
    assert body["deferred"] is True
    assert body["ignition_delay_s"] > 0

    # 事件 payload 带着**绝对时刻** —— 存剩余秒数的话，窗口中间重启会从头再数。
    ev = [e for e in svc.store.events(cid, kind="approved", limit=10)]
    assert ev and float(ev[-1]["payload"]["ignite_at"]) > svc.store.now_epoch()

    ig = client.get(f"/api/conducts/{cid}").json()["ignition"]
    assert ig and ig["by"] == "agent:experiment_design"
    assert 0 < ig["remaining_s"] <= ig["delay_s"]
