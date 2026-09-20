"""技能目录的 safety_level 必须来自注册元数据，并与实际工具接口一致。"""
from __future__ import annotations

# ── path bootstrap ───────────────────────────────────────────────────────────
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.modules.setdefault("nanonis_spm", MagicMock())


def _find_mastv2_root() -> str:
    p = Path(__file__).resolve()
    while p.parent != p:
        candidate = p / "MASTv2"
        if candidate.is_dir():
            return str(candidate)
        p = p.parent
    raise RuntimeError("MASTv2 dir not found")


_MASTV2_ROOT = _find_mastv2_root()
if sys.path and sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _n in list(sys.modules):
    if _n == "mast" or _n.startswith("mast."):
        _f = getattr(sys.modules[_n], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_n]

from unittest.mock import patch  # noqa: E402

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from mast.api.routes.skills import router  # noqa: E402

#: builder_api 真实产出的形状 —— 键是 `safety`，值是**小写**。
_CATALOG = {
    "index": [
        {"name": "QuitNanonis", "safety": "dangerous", "domain": "系统"},
        {"name": "LoadNanonisScript", "safety": "dangerous"},
        {"name": "SetSetpoint", "safety": "confirm"},
        {"name": "GetBias", "safety": "auto"},
        {"name": "MysteryTool"},                       # 连键都没有
    ],
    "cards": {},
}


def _client():
    app = FastAPI()
    app.include_router(router, prefix="/api")
    app.state.ctx = MagicMock()
    app.state.ctx.skill_registry = MagicMock()
    return TestClient(app)


def _levels():
    with patch("mast.webui.builder_api.get_catalog", return_value=_CATALOG):
        r = _client().get("/api/skills/catalog")
    assert r.status_code == 200, r.text
    body = r.json()
    assert not body.get("degraded"), "端点降级了，下面每条断言都会变成空转"
    return {row["name"]: row["safety_level"] for row in body["index"]}


# ════════════════════════════════════════════════════════════════════════════
def test_a_dangerous_skill_is_not_reported_as_auto() -> None:
    """**核心回归。** 这正是真机上看到的：DANGEROUS 显示成 AUTO。"""
    lv = _levels()
    assert lv["QuitNanonis"] == "DANGEROUS", (
        f"QuitNanonis 报成了 {lv['QuitNanonis']!r} —— 用户会以为它无害")
    assert lv["LoadNanonisScript"] == "DANGEROUS"


def test_every_level_round_trips() -> None:
    lv = _levels()
    assert lv["SetSetpoint"] == "CONFIRM"
    assert lv["GetBias"] == "AUTO"


def test_an_unknown_level_is_not_rendered_as_harmless() -> None:
    """没有安全级别信息时，**绝不能**显示成 AUTO。

    看错方向的代价不对称：把危险的显示成安全的会让人放心去点；把不知道的显示成
    「未知」只是多问一句。
    """
    assert _levels()["MysteryTool"] != "AUTO"


def test_the_fixture_matches_what_builder_api_actually_emits() -> None:
    """自检：本文件假设 builder_api 写的键叫 `safety`、值是小写。

    假设错了的话上面三条会一起变成「测了个不存在的形状」—— 而那正是这个 bug 的
    成因（两边对键名的假设不一致，谁都没验证过对方）。
    """
    src = (Path(_MASTV2_ROOT) / "mast" / "webui" / "builder_api.py").read_text(
        encoding="utf-8")
    assert '"safety": str(sl).lower()' in src, (
        "builder_api 写的键/大小写变了 —— 本文件的 fixture 和路由的读法都要跟着改")
