"""发布凭据与客户端凭据是两把钥匙。

**为什么要分**

客户端那把（``update_client_token.env`` > ``_defaults.DEFAULT_TOKEN``）由打包
脚本写进 ``mast/update/_defaults.py``，随字节码进 ``MAST.exe`` 的 PYZ ——
反编译 .pyc 就能读出来。也就是说 **它等价于公开值：谁拿到安装包谁就有它**。

在分离之前，同一把钥匙同时守着「拉更新」和 ``POST /skills/pack/publish``
（把技能包直接放进 packs 目录、供每一台机器下载）。Ed25519 签名链挡住了
「发布任意代码」那一半，但「谁能往服务器写」这一半只由这个公开值把着。

**这组测试钉的是什么**

1. 客户端 token **发布不了**（403）—— 分离真的生效了；
2. 发布 token **发布得了**（200）—— 分离没有把功能一起锁死；
3. 客户端 token 照样能读 —— 没有误伤客户端；
4. 发布 token 也能读 —— 发布机手上只有一把也够用；
5. **没配发布凭据时发布是 503，不是放行** —— 这条最重要，见下。

第 5 条单独说：回落是这道门唯一会失效的方式。「没配发布 token 就用客户端
token」听起来像是向后兼容，实际是把判据取消了 —— 门还在，每台机器又都能发布。
本仓在别处反复吃过这个形状（守卫看着在、其实没有）。
"""
from __future__ import annotations

import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from mast.update import signing  # noqa: E402
from mast.update import skillpack as SP  # noqa: E402
from mast.update.skillpack import K, PackFile, PackManifest  # noqa: E402

SKILL_SRC = "from mast.skills.base import BaseSkill\n"


@pytest.fixture
def keys(monkeypatch):
    """一把只在本测试里存在的签名密钥，并让它成为「出厂烘焙」的那把。"""
    priv, pub = signing.generate_keypair()
    monkeypatch.setattr(signing, "get_release_public_key", lambda: pub)
    return priv, pub


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    """覆盖层目录指到 tmp —— 测试写进真实用户数据目录，本仓被咬过五次。"""
    monkeypatch.setenv("MAST2_PROJECT_ROOT", str(tmp_path))
    yield


def _pack(tmp_path: Path, priv: str, *, pack_id="probe", version="1.0.0") -> Path:
    files = {"builtins/pack_probe.py": SKILL_SRC.encode()}
    man = PackManifest(
        pack_id=pack_id, version=version,
        created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        author="test", min_app_version="", max_app_version="",
        files=[PackFile(path=r, sha256=SP.sha256_bytes(b), size=len(b))
               for r, b in files.items()])
    out = tmp_path / SP.pack_filename(pack_id, version)
    with zipfile.ZipFile(out, "w") as z:
        z.writestr(SP.MANIFEST_NAME, json.dumps(man.to_dict(), ensure_ascii=False))
        z.writestr(SP.SIG_NAME, json.dumps(
            {K.SIGNATURE: signing.sign_manifest(man.to_dict(), priv)}))
        for r, b in files.items():
            z.writestr(SP.FILES_PREFIX + r, b)
    return out


def _srv(tmp_path, *, client_tok: str | None = None, publish_tok: str | None = None):
    """起一个 app，按需配两把钥匙。返回 (client, client_tok, publish_tok)。"""
    from fastapi.testclient import TestClient

    from mast.update import server as SRV

    ct = client_tok if client_tok is not None else SRV.generate_token()
    SRV.write_token(tmp_path, ct)
    pt = publish_tok
    if pt is not None:
        SRV.write_publish_token(tmp_path, pt)
    return TestClient(SRV.build_app(tmp_path)), ct, pt


def _bearer(tok: str) -> dict:
    return {"Authorization": "Bearer " + tok}


# ── 1 + 2：分离生效，且没把功能锁死 ────────────────────────────────────

def test_client_token_cannot_publish(tmp_path, keys):
    """**这条是整组的重点**：拿着随包分发的那把，发布要被拒。"""
    from mast.update import server as SRV

    priv, _ = keys
    c, ct, pt = _srv(tmp_path, publish_tok=SRV.generate_token())
    pack = _pack(tmp_path, priv)

    r = c.post("/skills/pack/publish", content=pack.read_bytes(),
               headers=_bearer(ct))
    assert r.status_code == 403, (
        f"客户端 token 竟然发布成功了（{r.status_code}）—— 分离没有生效。"
        "那把钥匙随安装包分发，等于每台机器都能发布。")
    assert "PUBLISH" in str(r.json().get("detail", "")).upper()


def test_publish_token_can_publish(tmp_path, keys):
    """分离不能把功能一起锁死 —— 正确的钥匙必须真的发得出去。"""
    from mast.update import server as SRV

    priv, _ = keys
    pt = SRV.generate_token()
    c, _ct, _ = _srv(tmp_path, publish_tok=pt)
    pack = _pack(tmp_path, priv)

    r = c.post("/skills/pack/publish", content=pack.read_bytes(),
               headers=_bearer(pt))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True and body["pack_id"] == "probe"
    # 真的落进 packs 索引了 —— 不只是返回了个 200。
    idx = c.get("/skills/pack/index", headers=_bearer(pt))
    assert idx.status_code == 200
    assert any(e["pack_id"] == "probe" for e in idx.json()["packs"])


# ── 3 + 4：读侧没有误伤 ────────────────────────────────────────────────

def test_client_token_still_reads(tmp_path):
    """客户端那几件事一件都不能少。"""
    from mast.update import server as SRV

    c, ct, _ = _srv(tmp_path, publish_tok=SRV.generate_token())
    for ep in ("/skills/pack/index", "/skills/index", "/subscriptions/index"):
        assert c.get(ep, headers=_bearer(ct)).status_code == 200, ep


def test_publish_token_also_reads(tmp_path):
    """发布机手上可能只有发布 token，它也得读得动。"""
    from mast.update import server as SRV

    pt = SRV.generate_token()
    c, _ct, _ = _srv(tmp_path, publish_tok=pt)
    assert c.get("/skills/pack/index", headers=_bearer(pt)).status_code == 200


# ── 5：没配发布凭据时**不回落** ────────────────────────────────────────

def test_unconfigured_publish_token_refuses_instead_of_falling_back(tmp_path, keys):
    """没配发布凭据 ⇒ 503 并说清楚怎么配。**绝不能**退回客户端 token。

    这是这道门唯一会失效的方式：一旦回落，每台装了 MAST 的机器又都能发布，
    而代码看起来仍然「有一道发布权限检查」。
    """
    priv, _ = keys
    c, ct, _ = _srv(tmp_path)          # 只配客户端 token
    pack = _pack(tmp_path, priv)

    r = c.post("/skills/pack/publish", content=pack.read_bytes(),
               headers=_bearer(ct))
    assert r.status_code == 503, (
        f"没配发布凭据时返回了 {r.status_code} —— 如果是 200，说明它回落到了"
        "客户端 token，这道门等于不存在。")
    detail = str(r.json().get("detail", ""))
    assert "publish-token" in detail, "报错要说清楚怎么配，否则只是挡住人"

    # 而且**读**不受影响：没配发布凭据不该把客户端也一起挡了。
    assert c.get("/skills/pack/index", headers=_bearer(ct)).status_code == 200


def test_no_token_at_all_is_still_401(tmp_path):
    """裸请求仍然是 401，不是 503 —— 别把「没带钥匙」说成「服务器没配好」。"""
    from mast.update import server as SRV

    c, _ct, _ = _srv(tmp_path, publish_tok=SRV.generate_token())
    assert c.post("/skills/pack/publish", content=b"x").status_code == 401
    assert c.get("/skills/pack/index").status_code == 401


# ── 凭据来源：发布 token 不该有任何「出厂默认」 ─────────────────────────

def test_publish_token_has_no_baked_in_default(tmp_path):
    """``_read_publish_token`` 没配就是空 —— 它**不能**像客户端那把一样有
    出厂默认值，否则它会跟着 ``_defaults.py`` 一起进 PYZ，分离就白做了。"""
    from mast.update import server as SRV

    assert SRV._read_publish_token(tmp_path) == ""
    # 对照：客户端那把在没有文件时会去问 _defaults（可能为空，但路径存在）。
    import inspect
    src = inspect.getsource(SRV._read_publish_token)
    assert "get_default_token" not in src, (
        "发布凭据回落到了客户端的默认值 —— 那个值随安装包分发。")
