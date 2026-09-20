"""扫描注册表是**进程级内存**,它不能从一条测试漏进下一条。

2026-08-15 的实事:全量跑里 ``test_agents_topology.py`` 的两条 artifacts 测试红,
而那个文件单跑全绿。二分下来是 ``tests/v2/unit/skills/`` 里某条测试注册过扫描,
而 ``/api/artifacts`` 通过 ``agents._shared.artifacts.list_existing()`` 把它当成
「已产出的产物」数了进去。

**为什么既有的守卫拦不住**:``conftest._real_experiment_root_is_read_only`` 对真实
实验根**在磁盘上**取指纹,而这条污染**从不落盘** —— `core.scan_registry` 的
``_recent_files`` / ``_records`` 是模块级列表。两条不同的逃逸路径要两道不同的守卫。

修法是 ``conftest._scan_registry_starts_empty``(autouse)。这个文件钉的是**它真的
在生效** —— 下面两条测试**顺序相关**:第一条注册一张扫描,第二条断言它不在了。
把 autouse fixture 摘掉,第二条立刻红。

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest \
        tests/v2/unit/api/test_scan_registry_does_not_leak_between_tests.py -x -v
"""
from __future__ import annotations

import sys
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[4] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)

import pytest  # noqa: E402

from mast.core import scan_registry  # noqa: E402

#: 两条测试之间传递的那个文件名。**不是 fixture** —— 它要跨测试活着,
#: 那正是这里要证明它**活不下来**的东西。
_LEAK_NAME = "leaked_scan_from_the_previous_test.sxm"


def test_a_registers_a_scan(tmp_path):
    """第一条:注册一张扫描。这条自己必须看得见它,否则第二条就没在证明什么。"""
    f = tmp_path / _LEAK_NAME
    f.write_bytes(b"not really a scan")
    scan_registry.record_scan_path(str(f))
    assert any(_LEAK_NAME in p for p in scan_registry.recent_scan_paths()), (
        "注册完自己都看不见 —— 那下一条测试的「看不见」什么都没证明")


def test_b_does_not_see_the_previous_scan():
    """第二条:上一条注册的那张,在这里必须已经不在了。

    ⚠️ **这两条是顺序相关的**(pytest 按文件内顺序跑)。这是刻意的:
    「一条测试不污染下一条」这件事,只能用两条真的一前一后的测试来证。
    """
    leaked = [p for p in scan_registry.recent_scan_paths() if _LEAK_NAME in p]
    assert not leaked, (
        f"上一条测试注册的扫描漏进来了:{leaked}\n"
        f"⇒ conftest 的 _scan_registry_starts_empty 没生效。它拦的是"
        f"「/api/artifacts 把别的测试的扫描数成本次产出」——"
        f"症状是 test_agents_topology 只在全量跑时红、单跑绿。")


def test_the_artifacts_listing_is_the_consumer_that_makes_this_matter(tmp_path):
    """说清**为什么**这条泄漏要紧:它直接改变 ``/api/artifacts`` 的计数。

    这条不是重复上面两条 —— 上面证的是「泄漏被挡住了」,这条证的是
    「不挡的话,后果落在哪」。少了它,读者会以为这只是内存卫生问题。
    """
    from mast.agents._shared.artifacts import list_existing

    before = len(list_existing())
    f = tmp_path / "counted_by_the_artifacts_endpoint.sxm"
    f.write_bytes(b"x")
    scan_registry.record_scan_path(str(f))
    after = len(list_existing())
    assert after == before + 1, (
        "注册一张扫描没有改变 artifacts 计数 —— 那说明这条泄漏路径已经变了,"
        "上面两条测试可能正在守一个不存在的东西,去重新确认 list_existing 的来源")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
