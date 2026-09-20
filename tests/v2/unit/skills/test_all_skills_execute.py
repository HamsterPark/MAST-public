"""Every skill must actually RUN. Registration is not execution.

This file exists because of a bug that got all the way to a commit: 58 new skills
were written with ``SkillResult(..., message=...)``. ``SkillResult`` has no
``message`` field — it is ``summary``. Every one of those skills would have raised
``TypeError`` the first time an agent called it.

Nothing caught it. The registry only calls ``metadata()``, so discovery was clean.
The gate tests checked names, safety levels and tool-list membership — all of which
live in ``metadata()`` too. 4148 tests passed. The skills were, every one of them,
dead on the first call.

So: walk the whole builtins package, instantiate every skill, feed it a context that
answers every safe_call plausibly, and CALL IT. This does not check that a skill does
the right thing — the per-skill tests do that. It checks the thing no per-skill test
was ever going to check, because you only write a per-skill test for skills you are
thinking about.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
import time

import pytest

import mast.skills.builtins as builtins_pkg
from mast.core.types import SkillResult
from mast.skills.base import BaseSkill

# 源码级断言走它,不用 ``inspect.getsource``(2026-08-15):后者按 import 那一刻
# 的行号切当前文件,别人同时在改就返回错位切片 —— ``in`` 那半给假红,
# ``not in`` 那半给**假绿**。整模块 getsource 是安全档,不在此列。
from tests.v2.srcref import source_of


class _Rec:
    """A Nanonis record. return_value is (header, error, body) — body at index 2."""

    def __init__(self, body=(0.0,), error=None):
        self.return_value = [0, 0, list(body)]
        self.error = error


class PlausibleCtx:
    """Answers every safe_call with a well-shaped, zero-ish reply.

    Deliberately NOT a MagicMock: a MagicMock returns a Mock for `.error`, which is
    truthy, so every skill would take its error path and the happy path would never
    execute — the exact code we are trying to smoke out.
    """

    def __init__(self):
        self._t0 = time.monotonic()
        self.calls: list[str] = []

    def safe_call(self, verb, *args, **kw):
        self.calls.append(verb)
        # A body long enough that any [0]/[1]/[2] indexing lands somewhere.
        return _Rec(body=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))

    def check_abort(self):
        """探针上下文：**跑得太久就叫停**。

        这条测试问的是「调下去会不会炸」，不是「跑完要多久」。而技能的默认
        参数里合法地存在长耗时 —— ``CharacteriseQuietDrift`` 出厂就是驻停
        **30 分钟**。让它真睡，整条测试线就挂在那儿。

        恒回 ``True`` 也不对：那样每个技能都走中止分支，这条测试就不再是
        「主路径跑得通」了。所以按**时间**分档：前两秒照常（绝大多数技能的
        主路径在这之内跑完），超过两秒才报中止。

        顺带钉住一条真规矩：**凡是会等的技能，都必须及时响应中止**。
        2026-08-22 之前 ``CharacteriseQuietDrift`` 调的
        ``self.abortable_sleep`` 根本不存在（它继承 builtin 基类，而那个方法
        当时只在 composite 基类上）—— 秒崩，连中止都轮不到。
        """
        return (time.monotonic() - self._t0) > 2.0

    def run(self, skill_name, params=None):
        """Composite skills drive other skills through ctx.run. Give them a plausible
        success back rather than skipping the whole composite family — a composite
        that raises is exactly as dead as a leaf skill that raises."""
        self.calls.append(f"run:{skill_name}")
        return SkillResult(skill_name=skill_name, success=True, data={})


def _concrete_skills():
    out = []
    for mod_info in pkgutil.iter_modules(builtins_pkg.__path__):
        mod = importlib.import_module(f"mast.skills.builtins.{mod_info.name}")
        for _, obj in inspect.getmembers(mod, inspect.isclass):
            if (issubclass(obj, BaseSkill) and obj is not BaseSkill
                    and obj.__module__ == mod.__name__
                    and not inspect.isabstract(obj)):
                out.append((f"{mod_info.name}.{obj.__name__}", obj))
    return sorted(out)


SKILLS = _concrete_skills()

# Skills whose execute() legitimately needs more than a Nanonis connection —
# a real file, a live registry, an experiment on disk. Their own tests cover them.
# Keep this list SHORT and justified: every entry is a skill this smoke cannot see.
_NEEDS_MORE_THAN_A_CONNECTION = {
    # composite/meta skills drive other skills through a real ExecutionContext
    "composite_runner",
    # these read/write real artifacts on disk
    "mosaic", "literature", "paper",
}


def _skip_reason(mod_name: str) -> str | None:
    for frag in _NEEDS_MORE_THAN_A_CONNECTION:
        if frag in mod_name:
            return f"{mod_name} 需要真实文件/注册表环境，由其专属测试覆盖"
    return None


def test_there_are_skills_to_test():
    assert len(SKILLS) > 200, f"只发现 {len(SKILLS)} 个 skill 类——发现逻辑坏了"


@pytest.mark.parametrize("label,cls", SKILLS, ids=[s[0] for s in SKILLS])
def test_skill_executes_without_exploding(label, cls):
    """Call execute() with its own declared defaults. It may FAIL — that is a normal,
    typed outcome — but it may not RAISE, and it must return a SkillResult.

    A skill that raises reaches the agent as an unhandled exception rather than a
    SkillResult, which means no error path, no diagnostics record, no HITL, no
    recovery — just a dead turn.
    """
    reason = _skip_reason(label.split(".")[0])
    if reason:
        pytest.skip(reason)

    skill = cls()
    meta = skill.metadata()

    # Build params from the declared spec: required ones get a plausible value of the
    # declared type, optional ones their default. This is also a check on the spec —
    # a required parameter with no usable type is a spec MAST cannot fill either.
    params: dict = {}
    for p in meta.parameters or []:
        if not p.required:
            if p.default is not None:
                params[p.name] = p.default
            continue
        if p.allowed_values:
            params[p.name] = p.allowed_values[0]
        elif p.type == "int":
            params[p.name] = int(p.min_value if p.min_value is not None else 0)
        elif p.type == "float":
            lo = p.min_value if p.min_value is not None else 0.0
            hi = p.max_value if p.max_value is not None else 1.0
            params[p.name] = float(lo) if lo > 0 else float(min(1e-9, hi))
        elif p.type == "bool":
            params[p.name] = False
        else:
            params[p.name] = "0"

    ctx = PlausibleCtx()
    try:
        result = skill.execute(ctx, params)
    except Exception as exc:  # noqa: BLE001 — that is precisely what we are looking for
        raise AssertionError(
            f"{label}.execute() 抛异常而不是返回 SkillResult：{type(exc).__name__}: {exc}\n"
            f"  参数：{params}\n"
            "  抛异常的 skill 到不了 agent 的错误处理路径——没有 SkillResult、没有诊断记录、"
            "没有 HITL、没有恢复，只有一个死掉的回合。"
        ) from exc

    assert isinstance(result, SkillResult), (
        f"{label}.execute() 返回了 {type(result).__name__}，不是 SkillResult"
    )
    assert result.skill_name, f"{label} 返回的 SkillResult 没有 skill_name"


@pytest.mark.parametrize("label,cls", SKILLS, ids=[s[0] for s in SKILLS])
def test_skill_result_fields_are_real(label, cls):
    """The specific trap: SkillResult has `summary`, not `message`. A typo'd kwarg is
    a TypeError at call time and invisible at registration time."""
    import dataclasses
    valid = {f.name for f in dataclasses.fields(SkillResult)}
    src = source_of(cls)
    import re
    for kw in re.findall(r"SkillResult\(\s*((?:[^()]|\([^()]*\))*)\)", src, re.S):
        for key in re.findall(r"(?:^|,)\s*([a-z_]+)\s*=", kw):
            assert key in valid, (
                f"{label} 给 SkillResult 传了不存在的字段 `{key}=`（合法字段：{sorted(valid)}）。"
                "注册时发现不了——registry 只调 metadata()。"
            )
