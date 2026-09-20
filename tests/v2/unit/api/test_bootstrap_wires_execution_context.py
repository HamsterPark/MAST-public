"""``build_live_context()`` 必须挂全 ``_execution_context()`` 要的那几个单例。

2026-08-02 实机抓到：``ctx.state`` **全仓从来没有任何地方赋过值**。
``signals._execution_context()`` 要 pool + state + registry 三者齐全才肯建
``ExecutionContext``，于是它恒为 None，两个端点永远退化：

- ``/api/experimental/signals`` —— 返回**硬编码的默认通道表**（16 条，名字都很像真
  的：Current (A) / Bias (V) / Z (m) / LI Demod 1 X (A)…），``timebases=[]``，
  而 ``detail`` 写的是「nanonis not wired」。当时 Nanonis 连得好好的：电流监控正在
  同一个连接池上轮询，``/api/monitoring/status`` 里有 6 个从仪器读回来的真时基。
  **一个看着像真数据的降级默认值 + 一句方向指错的诊断**，这是本仓库反复吃亏的
  那一类。
- ``/api/coarse-map/selfcheck`` —— 恒报「内核未就绪(pool/state/registry 未接线)」。
  它偏偏就是为真机验收写的粗动调试自检。

为什么用静态契约检查而不是真跑一遍 ``build_live_context()``：那个函数会构造整个
``CoreRuntime``（视觉后端、六个 agent、checkpointer、连接池），单元测试里跑不动。
而这条不变式本来就是**名字对不对得上**的问题，AST 判得又准又快。
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

_API = Path(__file__).resolve().parents[4] / "MASTv2" / "mast" / "api"
_BOOTSTRAP = _API / "bootstrap.py"
_SIGNALS = _API / "routes" / "signals.py"

#: ``_execution_context()`` 读的属性，按「或」分组 —— 每组至少要有一个被挂上。
#: 组内是 ``getattr(ctx, a, None) or getattr(ctx, b, None)`` 的关系。
_REQUIRED_GROUPS = (
    ("connection_pool",),
    ("state", "instrument_state"),
    ("skill_registry", "registry"),
    # ``routes/scope.py::_runtime()`` 找 ``app.state.runtime`` 或 ``ctx.runtime``，
    # 两个此前全仓都没人赋过值 —— `/api/scope/folder-health` 于是永远报
    # folder_path=null / exists=false / samples=0 / ingest.enabled=false，
    # 哪怕实验文件夹在磁盘上结构完整。`/api/scope/nanonis-dir` 读写两端同样退化。
    # 2026-08-02 实机第三次撞上同一个根因，才有了这份守卫。
    ("runtime",),
)

#: 同样被 ``_execution_context`` 读、但**不参与 None 判定**的：它们是把进程级
#: 急停事件串进 ExecutionContext 的尽力而为路径（缺了只是这一次采集中止不了，
#: 不是整个上下文建不起来）。列在这里而不是直接忽略，是为了让「新增了一个真正的
#: 必需依赖」仍然会把下面那条测试撞红 —— 加进 REQUIRED 还是 OPTIONAL 必须是一次
#: 有意识的选择。
_OPTIONAL_READS = frozenset({"live_app", "app", "_orch_abort"})


def _names_helper_reads(path: Path, func: str) -> set[str]:
    """``func`` 里所有 ``getattr(…, "名字")`` 读到的属性名。"""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    fn = next((n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name == func), None)
    assert fn is not None, f"{path.name}::{func} 不见了 —— 契约变了，本测试要跟着改"
    out: set[str] = set()
    for node in ast.walk(fn):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "getattr" and len(node.args) >= 2
                and isinstance(node.args[1], ast.Constant)
                and isinstance(node.args[1].value, str)):
            out.add(node.args[1].value)
    return out


def _names_execution_context_reads() -> set[str]:
    return _names_helper_reads(_SIGNALS, "_execution_context")


def _names_bootstrap_wires() -> set[str]:
    """bootstrap 挂到 ctx 上的属性名：``ctx.x = …`` 与 ``ctx.wire(x=…)`` 都算。"""
    tree = ast.parse(_BOOTSTRAP.read_text(encoding="utf-8"))
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if (isinstance(tgt, ast.Attribute) and isinstance(tgt.value, ast.Name)
                        and tgt.value.id == "ctx"):
                    out.add(tgt.attr)
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "wire"):
            out.update(kw.arg for kw in node.keywords if kw.arg)
    return out


def test_bootstrap_wires_everything_execution_context_needs():
    wired = _names_bootstrap_wires()
    missing = [g for g in _REQUIRED_GROUPS if not (set(g) & wired)]
    assert not missing, (
        "build_live_context() 没挂这些单例，_execution_context() 会恒返回 None，"
        "凡是走它的端点永远退化（而且退化得像正常数据）：\n"
        + "\n".join("  " + " 或 ".join(g) for g in missing)
        + f"\n\nbootstrap 目前挂了：{sorted(wired)}"
    )


def test_the_required_group_list_still_covers_what_the_code_reads():
    """有人往 ``_execution_context`` 里加了新依赖时，逼上面那条测试跟着更新。

    否则会退回原状：新依赖没人挂，端点静默退化，而测试一片绿。
    """
    declared = {n for g in _REQUIRED_GROUPS for n in g} | set(_OPTIONAL_READS)
    actual = _names_execution_context_reads()
    unaccounted = sorted(actual - declared)
    assert not unaccounted, (
        f"_execution_context() 现在还读了 {unaccounted}，但 _REQUIRED_GROUPS 里没有。"
        "把它加进去（并确认 bootstrap 真的挂了它），否则这条依赖没有任何守卫。"
    )


@pytest.mark.parametrize("group", _REQUIRED_GROUPS, ids=lambda g: g[0])
def test_each_singleton_is_wired_from_the_runtime(group):
    """光有 ``ctx.x = …`` 还不够，值得确认它取自 runtime 而不是恒 None 的占位。"""
    src = _BOOTSTRAP.read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if (isinstance(tgt, ast.Attribute) and isinstance(tgt.value, ast.Name)
                        and tgt.value.id == "ctx" and tgt.attr in group):
                    seg = ast.get_source_segment(src, node.value) or ""
                    assert "rt" in seg, (
                        f"ctx.{tgt.attr} 没有从 runtime 取值：{seg!r}")
                    return
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "wire"):
            for kw in node.keywords:
                if kw.arg in group:
                    seg = ast.get_source_segment(src, kw.value) or ""
                    assert "rt" in seg, (
                        f"wire({kw.arg}=…) 没有从 runtime 取值：{seg!r}")
                    return
    pytest.fail(f"{' 或 '.join(group)} 在 bootstrap 里根本没被赋值")


def test_scope_runtime_helper_names_are_wired():
    """``routes/scope.py::_runtime()`` 读的名字也必须有人挂。

    它找 ``app.state.runtime`` / ``ctx.runtime``，而 bootstrap 一直只挂
    ``live_app`` 与 ``app`` —— 同一个 CoreRuntime 的三个别名，少挂一个就有三个端点
    静默变空（folder-health 报「文件夹不存在」而磁盘上结构完整，是最误导的一个）。
    """
    scope = _API / "routes" / "scope.py"
    reads = _names_helper_reads(scope, "_runtime")
    wired = _names_bootstrap_wires()
    missing = sorted(reads - wired)
    assert not missing, (
        f"scope._runtime() 读 {sorted(reads)}，但 bootstrap 只挂了这些里的 "
        f"{sorted(reads & wired)}；缺 {missing} → _runtime() 恒返回 None，"
        "/api/scope/folder-health 与 /api/scope/nanonis-dir 永远退化。"
    )
