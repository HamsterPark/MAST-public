"""量程远小于 1 的参数，描述里不得再教指数写法（2026-08-04）。

有量纲参数自 2026-08-04 起走**字符串通道**（模型发数字时尾数与指数会被拆开，
真实工具调用通道上实测 0/12 对 12/12）。其中「整个量程远小于 1」的那些参数
（``needs_strict_prefix``）**强制要求 SI 前缀** —— 对它们 ``"1e-7"`` 不是不推荐，
是 ``parse_si`` **直接拒绝**。

改造完成的当天，71 个这样的参数里仍有 **38 个**在描述里给模型看 ``1e-7`` / ``5e-8``
这类示例。它构成一个闭环：模型照着写 → 被拒 → 拒绝语（当时也在教指数写法）让它
再写一次 → 再被拒。永远出不去，而每一步看起来都在「按提示做」。

本测试钉住两件事：

1. strict 参数的描述里**不出现指数写法**；
2. 描述里至少有一个**真的能通过 strict 解析**的示例 —— 光删掉错的不够，
   模型需要看到对的。

⚠️ 这里**不检查**「描述有没有把对的形式说成错的」。那是语义错误，测试抓不住 ——
而它真的发生过：批量转换脚本把我手写的「**不接受** '3e-12' 这种科学计数法」里的
``3e-12`` 一并转成了 ``3p``，于是那句话变成「**不接受** '3p'」，把唯一正确的写法
标成了被拒的。教训是程序性的：**不要拿无脑转换器去改手写的安全文案**，尤其是那种
故意提及错误形式以作警告的句子。

从仓库根跑::

    .venv-v2-py313/Scripts/python.exe -m pytest \
        tests/v2/unit/skills/test_strict_param_descriptions.py -q
"""
from __future__ import annotations

# ── path bootstrap ───────────────────────────────────────────────────────────
import re
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

import pytest  # noqa: E402

from mast.core.registry import SkillRegistry  # noqa: E402
from mast.core.si_quantity import (  # noqa: E402
    SIParseError,
    needs_strict_prefix,
    parse_quantity,
)

SKILL_PACKAGES = ("mast.skills.builtins", "mast.skills.composite")

#: 指数写法：1e-7 / 5e-8 / 100e-9 / 1.5e-6 / 1e-09
_EXPONENT = re.compile(r"(?<![\w.])\d+(?:\.\d+)?[eE]-\d+(?![\w])")
#: 候选示例：数字 + SI 前缀（用于「有没有给出对的写法」）
_PREFIXED = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?\s?[afpnuµmkMG](?![\w])")


def _strict_params():
    """[(skill, param, description)] —— 全部强制前缀的有量纲参数。"""
    reg = SkillRegistry()
    reg.discover(*SKILL_PACKAGES)
    out = []
    for name, versions in reg._skills.items():
        cls = list(versions.values())[-1]
        try:
            meta = cls().metadata()
        except Exception:  # noqa: BLE001 — 实例化失败由别的测试管
            continue
        for spec in getattr(meta, "parameters", []) or []:
            if not getattr(spec, "unit", None):
                continue
            if not needs_strict_prefix(min_value=spec.min_value,
                                       max_value=spec.max_value):
                continue
            out.append((name, spec.name, spec.description or ""))
    return out


# ════════════════════════════════════════════════════════════════════════════
def test_the_scan_actually_finds_strict_params() -> None:
    """先证明扫描有效 —— 一个恒空的扫描会让下面两条断言都白过。

    这不是形式主义：本文件初稿的枚举脚本把 registry 的值当成了类（其实是
    ``{版本: 类}`` 的 dict），实例化全部抛异常、被 ``except: continue`` 吞掉，
    于是「0 个问题」——而真实数字是 38 个。
    """
    params = _strict_params()
    assert len(params) > 40, f"只找到 {len(params)} 个强制前缀参数，扫描逻辑可能坏了"
    names = {f"{s}.{p}" for s, p, _ in params}
    assert "SetSetpoint.setpoint_a" in names
    assert "SetZCtrlGain.p_gain" in names


def test_no_strict_param_teaches_exponent_form() -> None:
    """描述里出现 ``1e-7``，就是在教一种这个参数**会拒绝**的写法。"""
    bad = [(s, p, _EXPONENT.findall(d)) for s, p, d in _strict_params()
           if _EXPONENT.search(d)]
    assert not bad, (
        f"{len(bad)} 个强制前缀参数的描述里还在给模型看指数写法 —— "
        f"而它们会拒绝这种写法：\n  "
        + "\n  ".join(f"{s}.{p}: {hits}" for s, p, hits in bad[:25])
    )


def test_the_model_facing_schema_always_shows_a_form_that_actually_parses() -> None:
    """光删掉错的不够 —— 模型需要看到一个**真的能过**的例子。

    检查的是 **adapter 产出的那份描述**，不是技能里写的原文：``skill_adapter``
    会给每个有量纲参数追加格式说明与量程，所以原文里有没有例子并不重要，
    重要的是模型最终读到的那份里有。（初版对着原文断言，要求 40 多个只有一句话
    的参数各自带例子 —— 那是在要求一件系统**已经在别处提供**的事。）

    候选示例逐个喂给 ``parse_quantity(strict=True)`` 验证，不靠正则判断
    「看起来对」—— 判断格式对不对的唯一权威是解析器本身。
    """
    from mast.agents._shared.skill_adapter import _schema_from_metadata

    reg = SkillRegistry()
    reg.discover(*SKILL_PACKAGES)
    missing = []
    for name, versions in reg._skills.items():
        cls = list(versions.values())[-1]
        try:
            meta = cls().metadata()
            model = _schema_from_metadata(meta)
        except Exception:  # noqa: BLE001
            continue
        for spec in getattr(meta, "parameters", []) or []:
            if not getattr(spec, "unit", None):
                continue
            if not needs_strict_prefix(min_value=spec.min_value,
                                       max_value=spec.max_value):
                continue
            field = model.model_fields.get(spec.name)
            shown = (getattr(field, "description", "") or "") if field else ""
            ok = False
            for cand in _PREFIXED.findall(shown):
                try:
                    parse_quantity(cand.replace(" ", ""), strict=True,
                                   what=spec.name)
                    ok = True
                    break
                except SIParseError:
                    continue
            if not ok:
                missing.append(f"{name}.{spec.name}")
    assert not missing, (
        f"{len(missing)} 个强制前缀参数,模型看到的描述里没有任何一个能通过 strict "
        f"解析的示例：\n  " + "\n  ".join(missing[:25])
    )


def test_the_exponent_pattern_does_not_fire_on_prefixed_examples() -> None:
    """反向自检：判据本身不能把正确写法误判成错的。"""
    for good in ("'3p'", "100n", "1.5u", "写成 '50p' 即可", "-300p"):
        assert not _EXPONENT.search(good), good
    for bad in ("1e-7", "5e-8", "100e-9", "1.5e-6", "1e-09"):
        assert _EXPONENT.search(bad), bad
