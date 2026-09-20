"""`scripts/check_openapi_delta.py` —— 并发写 openapi 的守门人，它自己也要被守着。

**一个没人跑的检查器和一个通过的检查器，从外面看一模一样。**
本仓刚为这句话付过学费：`_check_spec_preimport.py` 引用了一个早就删掉的模块，
烂了很久没人发现，正因为没有任何东西跑它。所以这个文件存在的首要目的不是覆盖率，
是**让它保持可运行**。

守的是两件事：

* 有删除 → 退出码非 0（那是「把别人的端点吞掉了」的信号）；
* 只有新增 → 退出码 0（那是正常干活，不该报错）。
"""
from __future__ import annotations

# ── path setup BEFORE any mast.* imports (see tests/v2/conftest.py) ──
import sys
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[3] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)

import importlib.util
import json

import pytest

_SCRIPT = Path(_MASTV2_ROOT) / "scripts" / "check_openapi_delta.py"


def _load_script():
    """按路径 import —— `scripts/` 不是一个包。

    这一步本身就是价值所在：脚本一旦语法错、或者 import 了已删除的东西，
    这里立刻红。`_check_spec_preimport.py` 烂掉的正是这一层。
    """
    spec = importlib.util.spec_from_file_location("check_openapi_delta", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


C = _load_script()


def _doc(paths: dict | None = None, schemas: dict | None = None) -> dict:
    return {"openapi": "3.1.0",
            "paths": paths if paths is not None else {"/a": {"get": {}}},
            "components": {"schemas": schemas if schemas is not None
                           else {"S": {"properties": {"x": {}}}}}}


def test_the_script_is_importable_and_exposes_its_contract():
    for name in ("compare", "removals", "render", "main"):
        assert callable(getattr(C, name)), name


def test_identical_documents_report_nothing():
    d = C.compare(_doc(), _doc())
    assert C.removals(d) == []
    assert d["paths_added"] == [] and d["schemas_added"] == []


def test_pure_additions_are_not_an_error():
    """新增是正常干活。只有**删除**才是「吞掉了别人的东西」。"""
    base = _doc()
    head = _doc(paths={"/a": {"get": {}}, "/b": {"get": {}}},
                schemas={"S": {"properties": {"x": {}}}, "T": {}})
    d = C.compare(base, head)
    assert d["paths_added"] == ["/b"]
    assert d["schemas_added"] == ["T"]
    assert C.removals(d) == []


@pytest.mark.parametrize("head,what", [
    (_doc(paths={}), "路径"),
    (_doc(schemas={}), "schema"),
    (_doc(schemas={"S": {"properties": {}}}), "字段"),
])
def test_every_kind_of_removal_is_caught(head, what):
    """路径 / schema / 字段，三种都要抓 —— 削掉别人 schema 上的一个字段
    和删掉整条路径是同一件事，只是更难看出来。"""
    d = C.compare(_doc(), head)
    bad = C.removals(d)
    assert bad, f"{what}被删却没报出来"
    assert any(what in b for b in bad), bad


def test_exit_code_is_the_gate(tmp_path):
    """退出码才是闸门 —— 报告好不好看无所谓，CI 看的是它。"""
    base = tmp_path / "base.json"
    base.write_text(json.dumps(_doc()), encoding="utf-8")

    same = tmp_path / "same.json"
    same.write_text(json.dumps(_doc()), encoding="utf-8")
    assert C.main(["--base", str(base), "--head", str(same)]) == 0

    added = tmp_path / "added.json"
    added.write_text(json.dumps(_doc(
        paths={"/a": {"get": {}}, "/new": {}})), encoding="utf-8")
    assert C.main(["--base", str(base), "--head", str(added)]) == 0

    swallowed = tmp_path / "swallowed.json"
    swallowed.write_text(json.dumps(_doc(paths={})), encoding="utf-8")
    assert C.main(["--base", str(base), "--head", str(swallowed)]) == 1


def test_a_missing_ref_fails_loudly_not_silently(tmp_path):
    """读不到就要喊。**静默地把「读不到」当成「没差异」是这个工具最坏的失败模式** ——
    它会在你最需要它说话的时候放行。"""
    base = tmp_path / "base.json"
    base.write_text(json.dumps(_doc()), encoding="utf-8")
    with pytest.raises(SystemExit):
        C.main(["--base", "no/such/ref-or-file", "--head", str(base)])


def test_report_says_what_to_do_about_a_removal():
    d = C.compare(_doc(), _doc(paths={}))
    text = C.render(d, "HEAD", "工作树")
    assert "有东西不见了" in text
    assert "git log -S" in text, "报告要给出下一步，不能只报告坏消息"


def test_report_clears_you_when_there_are_no_removals():
    text = C.render(C.compare(_doc(), _doc()), "HEAD", "工作树")
    assert "零删除" in text


def test_it_actually_runs_against_this_repo():
    """对着仓库真跑一次。

    合成文档过了不代表对真文件也过 —— 这个工具的全部意义就是对着那份
    三万行的真 JSON 用。工作树相对 HEAD 不该有删除（有的话就是真出事了，
    那正是它该红的时候）。
    """
    rc = C.main([])
    assert rc == 0, "工作树的 openapi.json 相对 HEAD 有删除 —— 去看上面的报告"
