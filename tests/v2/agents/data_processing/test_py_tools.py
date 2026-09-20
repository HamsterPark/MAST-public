"""``py_run`` / ``py_stage_data`` 的 tool 层行为。

底下那条链（会话/执行/回收）由 ``tests/v2/unit/pyexec/`` 覆盖。这里只盯 tool 层
自己的几件事，每一件都对应一个「本来会静默」的形状：

* 没有运行时时 **说清楚查了哪些地方**，而不是一句「不可用」；
* 提示是**附加的**，不是拦截 —— 第一版设计成命中就拒绝执行，撤销了；
* 连续失败要停手 —— StallGuard 未必认得出「同一个错」，因为 traceback 里带行号
  和临时路径，而那正是它做归一化要解决的问题；
* prompt 里 ``py_run`` 的**位置**是防滥用机制的一部分，位置变了要红。
"""

from __future__ import annotations

import pytest

from mast.agents.data_processing import tools as dp


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("MAST_DP_SESSIONS_DIR", str(tmp_path / "dp-sessions"))
    monkeypatch.setenv("MAST_FIGURES_DIR", str(tmp_path / "figures"))
    dp._py_failures.clear()
    yield
    dp._py_failures.clear()


# ── 降级要说人话 ────────────────────────────────────────────────────
def test_without_a_runtime_it_names_what_was_probed(monkeypatch):
    """「不可用」必须附带一句能照着做的话。

    一个只说「py_run 不可用」的返回，会让模型反复重试同一个调用 —— 它没有任何
    信息可以据此改变行为。
    """
    from mast.pyexec.runtime import ProbeReport

    monkeypatch.setattr(
        "mast.pyexec.runtime.probe_runtime",
        lambda **_kw: ProbeReport(runtime=None,
                                  tried=(("C:/nope/python.exe", "文件不存在"),)))
    out = str(dp.py_run.func(code="print(1)", tool_call_id="c1"))
    assert "不可用" in out
    assert "C:/nope/python.exe" in out, "没说查过哪里"
    assert "Python 3.13" in out, "没说该装什么"


# ── 提示是附加的，不是拦截 ──────────────────────────────────────────
def test_a_hint_is_appended_not_enforced():
    """命中提示模式的代码**照常执行**，提示挂在结果后面。

    第一版设计成「命中就返回重定向、不执行」。撤销的理由：一个更复杂的分析只要
    恰好用了 imshow+savefig 就被打回，那一轮白费 —— 而这条的真正价值不是省一次
    执行，是 plot_scan 的图**客观更好**（真 nm 坐标轴、带单位色标）。
    是路由到更好的输出，不是阻拦。
    """
    hint = dp._py_hint("import matplotlib.pyplot as plt\nplt.imshow(z)\nplt.savefig('a.png')")
    assert "plot_scan" in hint
    assert "不受影响" in hint, "提示必须说明本次结果没被影响"


def test_hints_are_precise_not_greedy():
    """精度优先于召回：误判一次浪费一句话，漏判一次没什么损失。"""
    assert dp._py_hint("import numpy as np\nprint(np.mean(a))") == ""
    assert dp._py_hint("x = fft2(a)  # 顺手") != ""          # 只做 FFT → 有提示
    assert "fit_sts_peaks" in dp._py_hint("peaks = find_peaks(y)")


# ── 尺寸守卫 ────────────────────────────────────────────────────────
def test_oversize_code_is_refused_with_the_reason():
    """这是防呆，不是管束长度 —— 所以门槛设在 max_tokens 打不到的地方。"""
    out = str(dp.py_run.func(code="x = 1\n" * 20000, tool_call_id="c1"))
    assert "拆成几步" in out
    assert "工作目录" in out, "拒绝时要说明为什么拆得动（会话保留中间结果）"


def test_empty_code_is_refused():
    assert "空" in str(dp.py_run.func(code="   ", tool_call_id="c1"))


# ── 连续失败要停手 ──────────────────────────────────────────────────
def test_repeated_failures_stop_the_loop(monkeypatch):
    """同一会话连续失败到上限就拒绝，并让模型报告现状。

    ``StallGuardMiddleware`` 靠归一化错误签名判重试，而 traceback 里带**行号和
    临时路径** —— 那恰恰是归一化要解决的那类噪声，它未必认得出「同一个错」。
    这个计数器是兜底：不让模型在一个改不好的脚本上把整轮的工具预算烧光。
    """
    from mast.pyexec import get_session

    sid = get_session(experiment_id="", thread_id="dp-default").sid
    dp._py_failures[sid] = dp._PY_MAX_CONSECUTIVE_FAILURES

    out = str(dp.py_run.func(code="print(1)", tool_call_id="c1"))
    assert "先停下来" in out
    assert "报告" in out
    assert dp._py_failures[sid] == 0, "拒绝之后要清零，否则这个会话永远打不开"


# ── py_stage_data ───────────────────────────────────────────────────
def test_stage_without_a_target_says_so():
    assert "scan_id" in str(dp.py_stage_data.func())


def test_stage_a_missing_file_is_a_readable_error(tmp_path):
    out = str(dp.py_stage_data.func(path=str(tmp_path / "ghost.sxm")))
    assert "找不到" in out


def test_stage_reports_arrays_and_scale(tmp_path):
    """返回里要有数组名、形状、单位 —— 脚本靠这些知道拿到了什么。"""
    import numpy as np

    src = tmp_path / "probe.npy"
    np.save(src, np.zeros((4, 5)))
    out = str(dp.py_stage_data.func(path=str(src), name="probe"))
    assert "mastdata.load" in out, "没告诉脚本怎么取"
    assert "(4, 5)" in out


# ── prompt 的位置是机制的一部分 ─────────────────────────────────────
def test_py_run_is_introduced_after_the_question_to_tool_table():
    """位置本身就是防滥用的机制，比多写几句劝告有效。

    模型先读到「问题 → 工具」映射表并在那里找到匹配；``py_run`` 被框成
    「表答不了的那些」放在它**后面**。把它提到 ``# Available tools`` 里去，
    就是把「最后的选择」变成「并列的选择」。
    """
    from mast.agents.data_processing.prompts import SYSTEM_PROMPT

    table = SYSTEM_PROMPT.find('"能隙 / 超导 / Delta 多大"')
    intro = SYSTEM_PROMPT.find("# 表里没有的问题")
    # 2026-08-24：`# Available tools` 那一节删了（它是 schema 的第二份副本）。
    # 现在工具在「问题 → 工具」路由表里介绍，锚点换成那张表的标题。
    tools_block = SYSTEM_PROMPT.find("## 问题 → 工具")

    assert table > 0 and intro > 0 and tools_block > 0
    assert tools_block < table, "路由表的标题应当在表内容之前"
    assert intro > table, "py_run 的介绍跑到映射表前面去了"
    assert "py_run" not in SYSTEM_PROMPT[tools_block:table], (
        "py_run 出现在路由表的开头 —— 那会让它变成一个并列选项，"
        "而它必须是「表答不了的那些」才用的最后选择")


def test_the_prompt_says_numbers_go_through_result_json():
    """这条是数值不经过 token 流那条路的**入口**：模型得知道有这条路。"""
    from mast.agents.data_processing.prompts import SYSTEM_PROMPT

    assert "save_result" in SYSTEM_PROMPT
    assert "metrics_path" in SYSTEM_PROMPT


# ── 代码本身的鲁棒性 ────────────────────────────────────────────────
def test_awkward_code_survives_the_trip_to_the_child():
    """反斜杠、中文、三引号、f-string 嵌套 —— 一路完整到达并执行。

    这段只覆盖 **tool 参数 → 文件 → 子进程** 这一程。真正的风险在更上游：
    provider 把 tool call 的 arguments 序列化成 JSON 时，Windows 路径里的
    反斜杠和 CJK 都要转义，而截断发生在 arguments 中间会产生**非法 JSON** ——
    表现为 provider 报错或解析失败，一个困惑的失败而不是干净的失败。
    那一段只有真跑一次 provider 才验得了；
    这里先把我们自己这一程钉死。
    """
    from mast.pyexec import find_runtime

    if find_runtime() is None:
        pytest.skip("本机没有分析运行时")

    # raw 三引号：里面的反斜杠是**字面**反斜杠，正是要传过去的东西。
    # （不用普通字符串 —— `\U` 会被当成 unicode 转义，测试文件自己都编译不过。）
    code = r'''# 中文注释：路径与转义
import os
p = r"C:\Users\test\data\scan_001.sxm"
q = "D:\\raw\\a.dat"
doc = """三引号
跨行 with "quotes" and 'single'
"""
name = "针尖"
print(f"{name}: {os.path.basename(p)} | {len(doc)} | {q[:3]}")
print("TAB\tAND")
'''
    out = str(dp.py_run.func(code=code, tool_call_id="c1", timeout_s=60))
    assert "针尖: scan_001.sxm" in out, out[-800:]
    assert "D:\\" in out, "反斜杠没活着到子进程"
    assert "TAB\tAND" in out, "转义序列在子进程里没被正确解释"
