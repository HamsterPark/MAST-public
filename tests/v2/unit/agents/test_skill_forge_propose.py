"""第三级:提议一个新的原子技能。**落盘,不启用。**

这一级的形态是:组合轨零代码、注册即用;而 Python 代码轨
生成的裸代码**不经过参数包络所在的那一层**,所以它只写文件,由人审过、加进
``enabled.json`` 白名单、重启之后才生效。

这一组最重要的不是「它写了文件」,是**它没有做的三件事**:没有碰 enabled.json、
没有注册进 registry、没有 exec。少一条,这一级就从「提议」变成了「开一条没有闸门
的硬件通道」。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[4]
_MASTV2_ROOT = str(_REPO / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest  # noqa: E402
from _forge_fixtures import StubCtx, call, fresh_registry, tools  # noqa: E402

GOOD = '''
from mast.skills.base import BaseSkill
from mast.core.types import SkillMetadata, SkillResult


class ReadWidgetTemperature(BaseSkill):
    def metadata(self):
        return SkillMetadata(name="ReadWidgetTemperature", description="d")

    def execute(self, context, params):
        rec = context.safe_call("Widget_TempGet")
        return SkillResult(skill_name="ReadWidgetTemperature", success=True,
                           data={"raw": rec.return_value})
'''


@pytest.fixture
def kit(tmp_path, monkeypatch):
    import mast.llm.skill_author as sa
    d = tmp_path / "custom_skills"
    monkeypatch.setattr(sa, "_CUSTOM_SKILLS_DIR", d)
    reg = fresh_registry()
    return d, reg, tools(reg=reg, ctx=StubCtx())


def _propose(ts, name, code, why="现有技能都读不到这个通道"):
    return call(ts["propose_python_skill"], name=name, code=code, rationale=why)


# ── 它做的那件事 ────────────────────────────────────────────────────

def test_a_good_draft_lands_on_disk(kit):
    d, _reg, ts = kit
    out = _propose(ts, "ReadWidgetTemperature", GOOD)
    assert out["ok"] is True, out
    assert out["enabled"] is False
    assert (d / "ReadWidgetTemperature.py").exists()


def test_the_rationale_is_written_into_the_file_header(kit):
    """审阅这份代码的人手上只有这一句话可以判断「为什么现有技能不够」。"""
    d, _reg, ts = kit
    _propose(ts, "ReadWidgetTemperature", GOOD, why="现有技能读不到 Widget 温度通道")
    text = (d / "ReadWidgetTemperature.py").read_text(encoding="utf-8")
    assert "现有技能读不到 Widget 温度通道" in text
    assert "未启用" in text
    assert "enabled.json" in text, "文件里要写清楚怎么启用,否则审阅者得去翻文档"


# ── 它**没有**做的三件事 ────────────────────────────────────────────

def test_proposing_does_not_touch_the_allowlist(kit):
    """``enabled.json`` 是用户的显式动作 —— 那就是这一级的门。"""
    d, _reg, ts = kit
    _propose(ts, "ReadWidgetTemperature", GOOD)
    assert not (d / "enabled.json").exists()


def test_proposing_does_not_register_the_skill(kit):
    d, reg, ts = kit
    _propose(ts, "ReadWidgetTemperature", GOOD)
    assert not reg.has("ReadWidgetTemperature")


def test_proposing_does_not_execute_the_code(kit):
    """写文件不等于运行它 —— 用一个「跑起来就会留痕」的载荷来证明。"""
    d, _reg, ts = kit
    marker = d.parent / "SHOULD_NOT_EXIST.txt"
    code = GOOD + f'\n_p = r"{marker}"\nopen(_p, "w").write("ran")\n'
    out = _propose(ts, "SideEffecty", code)
    # open(mode="w") 本来就在 deny-list 上,所以这里应当直接被拒;
    # 无论拒不拒,那个文件都绝不能出现。
    assert not marker.exists(), "提议阶段把代码跑起来了"
    assert out["ok"] is False


# ── AST deny-list(与 custom_loader 同一个检查器) ────────────────

@pytest.mark.parametrize("bad,needle", [
    ("import os\n" + GOOD, "os"),
    ("import subprocess\n" + GOOD, "subprocess"),
    (GOOD + "\nx = eval('1+1')\n", "eval"),
    (GOOD + "\ny = ().__class__.__bases__\n", "__class__"),
])
def test_dangerous_code_is_refused(kit, bad, needle):
    d, _reg, ts = kit
    out = _propose(ts, "Nasty", bad)
    assert out["ok"] is False
    assert needle in "\n".join(out["problems"]), out["problems"]
    assert not (d / "Nasty.py").exists(), "拒绝了却还是写了文件"


def test_syntactically_broken_code_is_refused(kit):
    d, _reg, ts = kit
    out = _propose(ts, "Broken", "def f(:\n  pass")
    assert out["ok"] is False
    assert not (d / "Broken.py").exists()


# ── 结构检查:它得长得像一个技能 ────────────────────────────────

def test_code_without_a_baseskill_subclass_is_refused(kit):
    """一份语法正确但没有技能类的文件会安安静静躺到审阅、加白名单、重启之后,
    才在 loader 那里说「没找到技能」—— 那时提议它的那一轮早就没了。"""
    d, _reg, ts = kit
    out = _propose(ts, "NotASkill", "VALUE = 42\n")
    assert out["ok"] is False
    assert "BaseSkill" in "\n".join(out["problems"])
    assert not (d / "NotASkill.py").exists()


def test_a_skill_class_missing_execute_is_refused(kit):
    code = (
        "from mast.skills.base import BaseSkill\n"
        "class Half(BaseSkill):\n"
        "    def metadata(self):\n        return None\n"
    )
    d, _reg, ts = kit
    out = _propose(ts, "Half", code)
    assert out["ok"] is False
    assert "execute" in "\n".join(out["problems"])


# ── 名字 ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", ["../evil", "..\\evil", "__init__", "a.b"])
def test_path_traversal_names_are_refused(kit, name):
    _d, _reg, ts = kit
    out = _propose(ts, name, GOOD)
    assert out["ok"] is False
    assert out["error"] == "bad_name"


def test_a_name_already_in_the_registry_is_refused(kit):
    """先去看看那个技能是不是已经做了你要的事 —— 官方优先在这一级同样成立。"""
    _d, _reg, ts = kit
    out = _propose(ts, "SetBias", GOOD)
    assert out["ok"] is False
    assert out["error"] == "name_taken"


def test_a_draft_already_awaiting_review_is_not_overwritten(kit):
    """用户可能已经在看那一份了 —— 覆盖掉等于把他读到一半的东西换掉。"""
    d, _reg, ts = kit
    assert _propose(ts, "ReadWidgetTemperature", GOOD)["ok"] is True
    out = _propose(ts, "ReadWidgetTemperature", GOOD)
    assert out["ok"] is False
    assert out["error"] == "exists"


# ── 回执要把「它还不能用」说清楚 ──────────────────────────────────

def test_the_reply_says_it_is_not_usable_yet_and_to_carry_on(kit):
    """回执含糊的话,模型会以为它能用了,然后去调一个不存在的技能。"""
    _d, _reg, ts = kit
    out = _propose(ts, "ReadWidgetTemperature", GOOD)
    msg = out["message"]
    assert "没有启用" in msg and "没有注册" in msg
    assert "继续" in msg, "要明说「用现有手段把当前任务做完」,不然它会停下来等"
    assert json.dumps(out, ensure_ascii=False)
