# -*- coding: utf-8 -*-
"""凭据守卫必须认得**裸的**关键词名 —— 否则它只在名字恰好有前缀时生效。

═══════════════════════════════════════════════════════════════════════════
一道只在特定命名下才生效的守卫，等于没有
═══════════════════════════════════════════════════════════════════════════

2026-08-21 按验证清单 修复项 ③ 做「故意破坏一次」：在 ``MASTv2/mast/`` 下放一个

    TOKEN = "zZq7Z1mPtqQ0aBcDeFgHiJkLmNoPqRsTuVwXyZ0123456789ab"

的 .py，按清单**构建必须 SystemExit 并点名它**。实际结果是构建**照常走完**，
守卫一声没吭，那个文件被原样收进了 ``mast/_src/``。

根因：正则的名字部分写成 ``[A-Z][A-Z0-9_]*(?:TOKEN|SECRET|…)``，那个开头的
``[A-Z]`` 要求关键词**前面至少有一个字符**。于是：

* ``DEFAULT_TOKEN`` / ``MY_SECRET`` / ``X_API_KEY`` —— 命中
* ``TOKEN`` / ``SECRET`` / ``PASSWORD`` / ``API_KEY`` / ``PRIVATE_KEY`` /
  ``CREDENTIAL`` / ``TOKEN_V2`` / ``SECRET_KEY`` —— **全漏**（13 个现实命名里漏 10 个）

这道守卫是**唯一**挡住「把推送服务器明文 token 交给一个会写代码的 agent」的东西
（DP 的分析子进程对 ``mast/_src/`` 有只读视图），所以它的召回率不是风格问题。

**要推翻这条**：给出一个真实场景，其中裸的 ``TOKEN = "<长串>"`` 不该被当成凭据。
"""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

_SRC = (Path(__file__).resolve().parents[4]
        / "MASTv2" / "scripts" / "bundle_sources.py")

#: 一个够长、不含占位词的假 token（守卫要求 ≥12 字符）。
_FAKE = "zZq7Z1mPtqQ0aBcDeFgHiJkLmNoPqRsTuVwXyZ0123456789ab"


def _mod():
    spec = importlib.util.spec_from_file_location("bundle_sources_probe", _SRC)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


#: 真机 修复项 ③ 里漏掉的那一批，逐个钉住。
@pytest.mark.parametrize("name", [
    "TOKEN", "SECRET", "PASSWORD", "PASSWD", "APIKEY", "API_KEY",
    "PRIVATE_KEY", "CREDENTIAL", "TOKEN_V2", "SECRET_KEY",
])
def test_a_bare_keyword_name_is_caught(name):
    """裸关键词 —— 这十个此前全漏。"""
    assert _mod()._CRED_ASSIGN.search('%s = "%s"' % (name, _FAKE)), (
        "%s = <50 字符> 没被认出来 —— 守卫只在名字有前缀时生效" % name)


@pytest.mark.parametrize("name", ["DEFAULT_TOKEN", "MY_SECRET", "X_API_KEY"])
def test_prefixed_names_still_caught(name):
    """带前缀的那一批本来就命中 —— 修召回不许把它们弄丢。"""
    assert _mod()._CRED_ASSIGN.search('%s = "%s"' % (name, _FAKE))


# ── 召回提高了，但误报不许跟着涨 ──────────────────────────────────────────

def test_a_function_call_is_not_a_credential():
    """``token = get_token()`` 是代码，不是凭据（模块自述的原话）。"""
    assert not _mod()._CRED_ASSIGN.search("TOKEN = get_token()")


def test_a_short_value_is_not_a_credential():
    """守卫只认 ≥12 字符的字面量。"""
    assert not _mod()._CRED_ASSIGN.search('TOKEN = "short"')


def test_an_empty_default_does_not_swallow_a_later_string():
    """``DEFAULT_TOKEN = ""`` 不许拿第二个引号当开头一路吃到下一个字符串。

    这是既有注释里记着的一次真误报（``mast/update/defaults.py`` 实测），
    由「引号里不许跨行」那个子句挡住 —— 提高召回不许把它撞坏。
    """
    src = 'DEFAULT_TOKEN = ""\n\nOTHER = "a long harmless string here"\n'
    m = _mod()._CRED_ASSIGN.search(src)
    assert m is None or "\n" not in m.group(2)


def test_a_lowercase_name_is_left_alone():
    """约定是大写常量。小写名放行，否则整个仓库的局部变量都会中招。"""
    assert not _mod()._CRED_ASSIGN.search('token = "%s"' % _FAKE)


def test_the_public_key_stays_allowlisted():
    """Ed25519 **公钥**公开是它的用途 —— 提高召回不许把它变成阻断项。"""
    assert "DEFAULT_SIGNING_PUBKEY" in _mod().CREDENTIAL_ALLOWLIST


@pytest.mark.parametrize("val", [
    "CHANGEME_BEFORE_USE", "<your-token-here>", "REPLACE_ME_xxxxxxxxxxxx",
    "EXAMPLE_TOKEN_VALUE_1", "PLACEHOLDER_abcdefgh",
])
def test_placeholders_are_still_treated_as_placeholders(val):
    """占位值不是凭据。一道会误报的守卫，人会学着忽略它。"""
    assert _mod()._PLACEHOLDER.search(val)


# ── 结构：别再写回那个可选前缀 ────────────────────────────────────────────

def test_the_name_prefix_stays_optional():
    """把前缀写回 ``[A-Z][A-Z0-9_]*`` 就是把那 10 个名字重新放走。

    这条查的是**正则本身的形状**，因为上面每一条 parametrize 都可能被人以
    「太啰嗦」为由删掉，而这一条说清了为什么不能删。
    """
    src = _SRC.read_text(encoding="utf-8")
    assert "([A-Z0-9_]*(?:TOKEN" in src, "名字前缀又变成必需的了"
    assert "([A-Z][A-Z0-9_]*(?:TOKEN" not in src


def test_the_regex_is_anchored_to_a_line_start():
    """只认行首赋值 —— 否则字符串里、注释里提到 TOKEN 都会中招。"""
    assert _mod()._CRED_ASSIGN.flags & re.MULTILINE
    assert _mod()._CRED_ASSIGN.pattern.startswith(r"^\s*")


# ── 「位置」不是「秘密」（加宽召回之后立刻需要的那一条）──────────────────

@pytest.mark.parametrize("value", [
    "update_client_token.env", "update_server_token.env",
    "config/secrets.json", r"C:\keys\id.pem", "/etc/mast/token.txt",
    "settings.yaml", "creds.ini", "store.sqlite3",
])
def test_a_value_that_names_a_location_is_not_a_credential(value):
    """``TOKEN_FILE = "update_client_token.env"`` 指的是**放凭据的地方**。

    把名字前缀改成可选之后，全仓立刻冒出两处这样的误报
    （``update/client.py`` 与 ``update/server.py``，值都以 ``.env`` 结尾）。
    而模块自述里那句话是认真的：**一道会误报的守卫，人会学着忽略它，
    那时它挡不住任何东西。**
    """
    assert _mod()._LOCATOR_VALUE.search(value), value


@pytest.mark.parametrize("value", [
    "zZq7Z1mPtqQ0aBcDeFgHiJkLmNoPqRsTuVwXyZ0123456789ab",
    "f4IPgvqjA2D2zaWTyKls0CnUFdxYD4Ps1ozkMvStxfudMQab",
    "sk-xxxxxxxxxxxxxxxxxxxxxxxx",
])
def test_a_real_looking_token_is_not_mistaken_for_a_location(value):
    """base64/hex 形状的凭据本体不含分隔符、也不以那些后缀结尾。"""
    assert not _mod()._LOCATOR_VALUE.search(value), value


def test_the_two_known_false_positives_are_gone_on_the_real_tree():
    """真树上跑一遍：``TOKEN_FILE`` 那两处不许再出现在命中里。

    这条查的是**真文件**，不是构造的字符串 —— 那两处是真实存在的写法，
    而这道守卫将来每一次加宽召回都会重新撞上它们。
    """
    root = Path(__file__).resolve().parents[4] / "MASTv2" / "mast" / "update"
    m = _mod()
    for f in ("client.py", "server.py"):
        hits = m.scan_credentials(root / f)
        assert not hits, "%s 又被当成凭据了：%s" % (f, hits)


def test_a_token_file_holding_a_real_token_is_still_caught():
    """名字叫 ``TOKEN_FILE`` 不构成豁免 —— 判据在**值**上，不在名字上。"""
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "probe.py"
        p.write_text('TOKEN_FILE = "%s"\n' % _FAKE, encoding="utf-8")
        assert _mod().scan_credentials(p), "值是真 token 却因为名字被放过了"
