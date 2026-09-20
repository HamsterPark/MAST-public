"""``AtomicBiasSeries`` —— 逐偏压序列的执行体（账在 core/s2_bias_ledger）。

S2 设计 D4 / D6 / §5「集成（假硬件）」。账本自身的规则由
``tests/v2/unit/core/test_s2_bias_ledger.py`` 钉；这里盯的是**执行这一层**
最容易悄悄坏掉的六件事：

1. **每一次尝试都落账**，中止时前面的账完整（针尖事件那条路尤其）；
2. **每帧都重设偏压** —— `execute_scan_plan` 那份实现曾经只在第一次尝试设，
   中间插一次会改偏压的针尖复核，重扫就用错偏压扫（设计陷阱 18）；
3. **anchor 帧无条件收尾**，否则「整段扫的是同一片原子」没有证据；
4. **判据只吃已保存的 .sxm** —— 拿不到路径是「判不了」，不是「没有原子」；
5. **高偏压槽位没声明时要出声**，不许沉默地表现得像检查过了；
6. **下发前的尺度预检**：判不出来的帧参数一帧都不扫。

⚠️ 假上下文用**显式哨兵**区分「没配置这个技能」与「配置成失败」。两者都写成
`None` 会让一个本该失败的步骤悄悄变成成功。

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest tests/v2/unit/skills/composite/test_atomic_bias_series.py -x -v
"""
from __future__ import annotations

import sys
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[5] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _name in list(sys.modules):
    if _name == "mast" or _name.startswith("mast."):
        _f = getattr(sys.modules[_name], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_name]

import pytest  # noqa: E402

from mast.core import s2_bias_ledger as L  # noqa: E402
from mast.core.types import SkillResult  # noqa: E402
from mast.skills.composite.atomic_bias_series import (  # noqa: E402
    SLOT_UNDECLARED,
    TIER_NAME,
    AtomicBiasSeries,
)

FAIL = object()          # 「配置成失败」的哨兵 —— 与「没配置」不是一回事
NM = 1e-9

#: `_params` 里那个 `expected_coord_epoch`。测试要跑到帧上去,就得先让代次对得上。
CURRENT_EPOCH = 7


@pytest.fixture(autouse=True)
def _coord_epoch_matches(monkeypatch):
    """让当前代次可读且**与 `_params` 一致**(2026-08-15 起必需)。

    在这之前 `expected_coord_epoch` 只被写进账本,没有任何东西会因为它停下来,
    所以测试不给这个前提也照跑。现在那道闸是真的:读不到当前代次 ⇒ UNVERIFIABLE
    ⇒ 一帧都不扫。于是**每一条要跑到帧上去的测试,都必须先声明这个前提** ——
    这不是给测试打补丁,是那道闸从此真的在拦。

    三条拒绝分支自己在 `_coord_epoch_gate` 那一节里各显式覆盖一次。
    """
    from mast.core import coord_epoch as ce

    monkeypatch.setattr(ce, "read_current_epoch", lambda: CURRENT_EPOCH)


class FakeCtx:
    """按技能名派活。**没配置的技能当场报错**，不静默成功。"""

    def __init__(self, script: dict):
        self.script = dict(script)
        self.calls: list[tuple[str, dict]] = []
        self.unscripted: list[str] = []
        self._n: dict[str, int] = {}
        self.run_id = "test-atomic-bias-series"

    def run(self, skill_name, params, version=None):
        self.calls.append((skill_name, dict(params)))
        if skill_name not in self.script:
            self.unscripted.append(skill_name)
            raise AssertionError(
                f"假上下文没配置 {skill_name!r} —— 「没配置」不许静默变成"
                f"「跑成功了」。已配置: {sorted(self.script)}")
        n = self._n.get(skill_name, 0)
        self._n[skill_name] = n + 1
        fn = self.script[skill_name]
        out = fn(params, n) if callable(fn) else fn
        if out is FAIL:
            return SkillResult(skill_name=skill_name, success=False,
                               error="scripted failure")
        return SkillResult(skill_name=skill_name, success=True, data=dict(out))

    def check_abort(self):
        return False

    def check_halt(self):
        return ""


def _scan(tmp_path, n, **extra):
    """一帧扫描的返回：造一个真的 .sxm 文件，好让路径是绝对的且存在。"""
    p = tmp_path / f"frame{n:03d}.sxm"
    p.write_bytes(b"fake")
    return {"saved_path": str(p), "scan_id": f"s{n:03d}",
            "size_m": 5 * NM, "pixels": 512, "line_time_s": 0.3, **extra}


def _verdict(v, admitted=True, **extra):
    return {"verdict": v, "remedy": (None if v != L.VERDICT_UNDECIDABLE
                                     else "rescan_frame"),
            "frame_admission": {"passed": admitted},
            "angular_concentration": 44.0, "fft_sharpness": 12.0, "snr": 9.0,
            "nm_per_px": 0.00977, "scale": "full",
            "period_fast_axis_nm": 0.31, "profile_name": "test-profile",
            "profile_provenance": "合成语料，测试用", **extra}


def _params(tmp_path, **over):
    p = {
        "bias_values": "-0.4,-0.2,0.2",
        "center_x_m": 1e-8, "center_y_m": 2e-8,
        "size_m": 5 * NM, "pixels": 512, "line_time_s": 0.3,
        "ledger_dir": str(tmp_path),
        "evidence_epoch": 3, "expected_coord_epoch": 7,
        "conduct_id": "c1", "stage_id": "S2",
    }
    p.update(over)
    return p


def _all_good(tmp_path, verdict=L.VERDICT_RESOLVED):
    return {
        "BiasSettleChange": lambda p, n: {"bias_v": p["bias_v"]},
        "ScanAt": lambda p, n: _scan(tmp_path, n),
        "SaveScan": lambda p, n: {},
        "VerifyAtomicResolution": lambda p, n: _verdict(verdict),
    }


def _per_bias_script(tmp_path, verdicts: dict[float, list[str]]):
    """按**偏压**给裁决，不按调用序号。

    序号很脆:判据只在扫描成功时被调，一次重试就把后面的全错开一位，于是测试
    看起来在测「-0.2 上是 absent」，实际测的是「第 3 次调用是 absent」。
    这里让假上下文记住最近一次设的偏压，判据据此查表 —— 与被测流程用的是
    同一把钥匙（规范化偏压键）。
    """
    state = {"bias": None, "used": {}}

    def _bias(p, n):
        state["bias"] = float(p["bias_v"])
        return {"bias_v": p["bias_v"]}

    def _verify(p, n):
        key = L.bias_key(state["bias"])
        seq = {L.bias_key(k): v for k, v in verdicts.items()}.get(key, [])
        i = state["used"].get(key, 0)
        state["used"][key] = i + 1
        v = seq[i] if i < len(seq) else (seq[-1] if seq else L.VERDICT_RESOLVED)
        return _verdict(v)

    return {
        "BiasSettleChange": _bias,
        "ScanAt": lambda p, n: _scan(tmp_path, n),
        "SaveScan": lambda p, n: {},
        "VerifyAtomicResolution": _verify,
    }


def _run(tmp_path, script, **over):
    skill = AtomicBiasSeries()
    ctx = FakeCtx(script)
    res = skill.run_composite(ctx, _params(tmp_path, **over))
    read = L.read_ledger(L.ledger_path(tmp_path), current_evidence_epoch=3)
    return res, ctx, read


# ══════════════════════════════════════════════════════════════════════
# 1. 短序列：账里每偏压一条 + 一条 anchor
# ══════════════════════════════════════════════════════════════════════

def test_a_short_series_writes_one_row_per_bias_plus_an_anchor(tmp_path):
    res, ctx, read = _run(tmp_path, _all_good(tmp_path))
    assert res.success is True, res.error
    assert len(read.rows) == 4, f"3 个偏压 + 1 个 anchor 应当是 4 条:{read.rows}"
    roles = [r["role"] for r in read.rows]
    assert roles.count(L.ROLE_ANCHOR) == 1
    keys = {r["bias_key"] for r in read.rows}
    assert keys == {L.bias_key(v) for v in (-0.4, -0.2, 0.2)}, (
        "anchor 用的是起始偏压，所以去重后仍然是 3 个键")
    assert res.data["series_exit"] == L.EXIT_COMPLETE
    assert res.data["n_resolved"] == 3


def test_the_series_runs_monotonically(tmp_path):
    """偏压来回跳会反复激起结的回滞 —— 单调走一遍。"""
    _res, ctx, _read = _run(tmp_path, _all_good(tmp_path),
                            bias_values="0.2,-0.4,-0.2")
    sent = [c[1]["bias_v"] for c in ctx.calls if c[0] == "BiasSettleChange"]
    series = sent[:3]
    assert series == sorted(series) or series == sorted(series, reverse=True), (
        f"偏压不是单调走的:{sent}")


def test_every_frame_sets_its_bias_again(tmp_path):
    """设计陷阱 18：`execute_scan_plan` 那份实现只在第一次尝试设偏压，
    中间插一次会改偏压的针尖复核，重扫就用错偏压扫。这里每帧都设。

    ⚠️ **必须走重试路径才测得到这件事**。第一版这条测试跑的是全成功脚本，
    于是一个偏压只有一次尝试，「只在第一次设」和「每次都设」表现完全一样 ——
    变异验证当场证明它拦不住任何东西。现在用 undecidable 逼出第二次尝试。
    """
    script = _per_bias_script(tmp_path, {-0.4: [L.VERDICT_UNDECIDABLE,
                                                L.VERDICT_RESOLVED]})
    _res, ctx, read = _run(tmp_path, script, bias_values="-0.4",
                           max_attempts_per_bias=3,
                           max_undecidable_retries_per_bias=2)
    n_bias = sum(1 for c in ctx.calls if c[0] == "BiasSettleChange")
    n_scan = sum(1 for c in ctx.calls if c[0] == "ScanAt")
    assert n_bias == n_scan, f"设偏压 {n_bias} 次 / 扫 {n_scan} 帧 —— 不是一一对应"
    assert [r["outcome"] for r in read.rows].count(L.OUTCOME_BLOCKED) == 0, (
        "出现了 blocked 记录，但脚本里没有任何东西拒绝过 —— "
        "多半是某次尝试跳过了设偏压那一步，被 `_ok` 当成「被拒」了")
    assert n_scan >= 3, (
        f"只扫了 {n_scan} 帧:要么没走到重试路径（那这条测试什么也没测），"
        f"要么重试在中途断了。两种都要看一眼。")


def test_every_scan_asks_for_the_atomic_tier(tmp_path):
    _res, ctx, _read = _run(tmp_path, _all_good(tmp_path))
    purposes = {c[1].get("purpose") for c in ctx.calls if c[0] == "ScanAt"}
    assert purposes == {TIER_NAME}


def test_the_whole_series_stays_on_one_spot(tmp_path):
    """换点等于换了被测对象。"""
    _res, ctx, _read = _run(tmp_path, _all_good(tmp_path))
    centers = {(c[1]["center_x_m"], c[1]["center_y_m"])
               for c in ctx.calls if c[0] == "ScanAt"}
    assert len(centers) == 1, f"序列跑到了不同的点上:{centers}"


def test_the_series_never_moves_site_on_its_own(tmp_path):
    """**钉住被否方案:本流程不实现 `max_site_moves_per_bias`。**

    设计 D4 的预算表里有这一项（建议默认 1），而同一份设计的漂移一节写着
    「科学要求是同一片原子在所有偏压下成像 —— 换点等于换了被测对象」。
    两条放在一起是矛盾的：让流程自己换点，等于让它在无人值守时**悄悄把
    整条序列的可比性作废**，而账里每一行看起来都正常。

    判据侧也支持这个方向：`VerifyAtomicResolution` 的 `move_site` 是给
    **调用方**的 remedy 建议，不是给执行体的指令。换点是「重排一条新序列」
    的决定（新的起始锚点、新的 evidence_epoch），归上层。

    所以这里连续 undecidable + move_site 也**不动中心**：把预算耗完，报
    `bias_undecided`（证据不足），把选择交出去。要推翻这个判断，需要先回答：
    换点之后，这条序列里换点前后的帧凭什么还能放在一张图上比较。
    """
    script = _per_bias_script(tmp_path, {-0.4: [L.VERDICT_UNDECIDABLE]})
    script["VerifyAtomicResolution"] = lambda p, n: _verdict(
        L.VERDICT_UNDECIDABLE, admitted=True)
    res, ctx, read = _run(tmp_path, script, bias_values="-0.4",
                          max_attempts_per_bias=3,
                          max_undecidable_retries_per_bias=2)
    centers = {(c[1]["center_x_m"], c[1]["center_y_m"])
               for c in ctx.calls if c[0] == "ScanAt"}
    assert len(centers) == 1, (
        f"反复判不了之后流程自己换了点:{centers} —— "
        f"换点前后的帧不再是同一片原子，而账里看不出来")
    assert res.data["n_undecided"] == 1, "证据不足要如实报，不许靠换点凑一个结论"
    assert "max_site_moves_per_bias" not in {
        p.name for p in AtomicBiasSeries().metadata().parameters}, (
        "声明了这个旋钮就等于承诺会用它 —— 要么真做要么别挂在参数表上")
    del read


def test_rows_carry_both_epochs_and_absolute_paths(tmp_path):
    _res, _ctx, read = _run(tmp_path, _all_good(tmp_path))
    for r in read.rows:
        assert r["evidence_epoch"] == 3 and r["coord_epoch"] == 7
        assert Path(r["frame_path"]).is_absolute()
        assert r["tier_name"] == TIER_NAME
        assert r["conduct_id"] == "c1" and r["stage_id"] == "S2"


# ══════════════════════════════════════════════════════════════════════
# 2. 部分失败：五栏分列
# ══════════════════════════════════════════════════════════════════════

def test_a_partial_series_lists_undecided_and_absent_separately(tmp_path):
    """2 resolved / 1 absent_confirmed / 2 undecided ⇒ series_partial，
    而且两栏**分列**。合并它们就是把「证据不足」说成「这里没有原子」。"""
    script = _per_bias_script(tmp_path, {
        -0.4: [L.VERDICT_RESOLVED],
        -0.2: [L.VERDICT_RESOLVED],
        0.0: [L.VERDICT_ABSENT],
        0.2: [L.VERDICT_UNDECIDABLE],
        0.4: [L.VERDICT_UNDECIDABLE],
    })
    res, _ctx, read = _run(tmp_path, script,
                           bias_values="-0.4,-0.2,0,0.2,0.4",
                           max_undecidable_retries_per_bias=1)
    d = res.data
    assert d["n_resolved"] == 2
    assert d["n_absent_confirmed"] == 1
    assert d["n_undecided"] == 2
    assert d["series_exit"] == L.EXIT_PARTIAL
    assert "证据不足" in res.summary and "确认没有对比度" in res.summary
    assert d["n_undecided"] + d["n_absent_confirmed"] != d["n_absent_confirmed"], (
        "两栏被合并了")


def test_an_undecidable_verdict_is_retried_but_an_absent_one_is_not(tmp_path):
    """判据说了「没有」就是说了 —— 重试改变不了一个已经成立的裁决。
    而「判不了」值得再试一次，且这笔预算与失败预算分开计。"""
    script = _per_bias_script(tmp_path, {-0.4: [L.VERDICT_ABSENT]})
    _res, ctx, _read = _run(tmp_path, script, bias_values="-0.4",
                            max_attempts_per_bias=3)
    absent_scans = sum(1 for c in ctx.calls if c[0] == "ScanAt")
    assert absent_scans == 2, f"absent 被重试了:{absent_scans} 帧（1 主 + 1 anchor）"

    tmp2 = tmp_path / "b"
    tmp2.mkdir()
    script2 = _per_bias_script(tmp2, {-0.4: [L.VERDICT_UNDECIDABLE]})
    skill = AtomicBiasSeries()
    ctx2 = FakeCtx(script2)
    skill.run_composite(ctx2, _params(tmp2, bias_values="-0.4",
                                      max_attempts_per_bias=3,
                                      max_undecidable_retries_per_bias=2))
    rows = L.read_ledger(L.ledger_path(tmp2), current_evidence_epoch=3).rows
    series_rows = [r for r in rows if r["role"] == L.ROLE_SERIES]
    assert len(series_rows) >= 2, f"undecidable 一次都没重试:{series_rows}"


def test_a_refused_bias_is_blocked_not_absent(tmp_path):
    """安全门拒绝是**拒绝**，不是「这里没有原子」。"""
    script = dict(_all_good(tmp_path))
    script["BiasSettleChange"] = lambda p, n: (
        FAIL if abs(p["bias_v"] - 0.2) < 1e-9 else {"bias_v": p["bias_v"]})
    res, _ctx, read = _run(tmp_path, script)
    assert res.data["n_blocked"] == 1
    assert res.data["n_absent_confirmed"] == 0
    blocked = [r for r in read.rows if r["outcome"] == L.OUTCOME_BLOCKED]
    assert len(blocked) == 1 and blocked[0]["verdict"] is None, (
        "被拒的那条不该带 verdict —— 判据根本没开口")


# ══════════════════════════════════════════════════════════════════════
# 3. 针尖事件：账停在中止点之前且完整
# ══════════════════════════════════════════════════════════════════════

def test_a_tip_event_aborts_the_series_and_keeps_the_ledger(tmp_path):
    script = dict(_all_good(tmp_path))
    script["ScanAt"] = lambda p, n: (
        _scan(tmp_path, n, tip_change_critical=True) if n == 1
        else _scan(tmp_path, n))
    res, _ctx, read = _run(tmp_path, script)
    assert res.data["series_exit"] == L.EXIT_ABORTED_TIP
    assert res.success is False, "针尖事件中止不是一次成功的序列"
    assert len(read.rows) == 2, f"中止点之前的账要完整:{read.rows}"
    assert read.rows[0]["verdict"] == L.VERDICT_RESOLVED
    assert read.rows[1]["outcome"] == L.OUTCOME_TIP_ABORT
    assert res.data["n_tip_aborted"] == 1


def test_no_anchor_is_taken_after_a_tip_event(tmp_path):
    """中止那一刻针尖状态不明，再扫一张既不安全也不构成证据。"""
    script = dict(_all_good(tmp_path))
    script["ScanAt"] = lambda p, n: _scan(tmp_path, n, tip_change_critical=True)
    _res, _ctx, read = _run(tmp_path, script)
    assert not [r for r in read.rows if r["role"] == L.ROLE_ANCHOR]


# ══════════════════════════════════════════════════════════════════════
# 4. 判据只吃已保存的 .sxm
# ══════════════════════════════════════════════════════════════════════

def test_a_frame_with_no_saved_path_is_undecidable_not_absent(tmp_path):
    """缓冲同一性已被真机证伪 —— 拿不到已保存路径就是判不了。"""
    script = dict(_all_good(tmp_path))
    script["ScanAt"] = lambda p, n: {"scan_id": "x"}      # 没有 saved_path
    res, ctx, read = _run(tmp_path, script, bias_values="-0.4",
                          max_attempts_per_bias=1)
    assert "VerifyAtomicResolution" not in [c[0] for c in ctx.calls], (
        "没有已保存的帧却还是把它送进了判据")
    assert res.data["n_undecided"] == 1 and res.data["n_absent_confirmed"] == 0
    assert all(r["verdict"] == L.VERDICT_UNDECIDABLE for r in read.rows)


def test_the_verdict_only_ever_receives_a_path(tmp_path):
    _res, ctx, _read = _run(tmp_path, _all_good(tmp_path))
    for name, params in ctx.calls:
        if name == "VerifyAtomicResolution":
            assert set(params) == {"scan_path"}, (
                f"判据收到了帧路径以外的东西:{params}")
            assert str(params["scan_path"]).endswith(".sxm")


# ══════════════════════════════════════════════════════════════════════
# 5. 高偏压槽位
# ══════════════════════════════════════════════════════════════════════

def test_an_undeclared_elevated_slot_says_so_instead_of_staying_silent(tmp_path):
    """沉默会被读成「查过了，没有超标的偏压」。"""
    res, ctx, read = _run(tmp_path, _all_good(tmp_path))
    assert res.data["elevated_slot_active"] is False
    assert "PreScanCheck" not in [c[0] for c in ctx.calls]
    whats = {u["what"] for u in res.data["unreadable"]}
    assert "elevated_bias_abs_v" in whats, (
        "槽位没生效这件事没有出现在任何一栏里")
    assert all(r["note"] == SLOT_UNDECLARED for r in read.rows
               if r["role"] == L.ROLE_SERIES)


def test_a_declared_slot_forces_a_tip_check_after_the_high_bias_frame(tmp_path):
    script = dict(_all_good(tmp_path))
    script["PreScanCheck"] = lambda p, n: {"quality": 0.9}
    res, ctx, _read = _run(tmp_path, script, elevated_bias_abs_v=0.3)
    assert res.data["elevated_slot_active"] is True
    checks = [c for c in ctx.calls if c[0] == "PreScanCheck"]
    assert len(checks) == 2, ("|-0.4| > 0.3 的那一帧（主序 + anchor）之后各要复核"
                              f"一次，实际 {len(checks)}")
    order = [c[0] for c in ctx.calls]
    assert order.index("PreScanCheck") > order.index("VerifyAtomicResolution"), (
        "针尖复核必须在**帧边界**消费，不是运行中打断")


def test_a_failed_tip_check_does_not_rescan_that_bias(tmp_path):
    """复核不过 ⇒ 走 S1 绕道。重扫等于在刚炸出来的坑上再判一次。

    ⚠️ 这个偏压的**终态仍然是 resolved** —— 那一帧确实拿到了裁决，而且是在针尖
    退化之前拿到的。把它改判成 tip_aborted 会与设计表第一条（「任一次
    atomic_resolved ⇒ bias_resolved」）打架。让它重跑的机制**不是终态，是代次**：
    绕道回来 bump `evidence_epoch`，这一行就变 stale，那个偏压自然要重做。
    两套机制各管各的，别在这里再造第三套。
    """
    script = dict(_all_good(tmp_path))
    script["PreScanCheck"] = FAIL
    res, ctx, read = _run(tmp_path, script, bias_values="-0.4",
                          elevated_bias_abs_v=0.3, max_attempts_per_bias=3)
    scans = sum(1 for c in ctx.calls if c[0] == "ScanAt")
    assert scans == 1, f"复核没过之后又重扫了那个偏压:{scans} 帧"
    assert res.data["series_exit"] == L.EXIT_ABORTED_TIP
    assert read.rows[-1]["outcome"] == L.OUTCOME_TIP_ABORT
    assert "S1" in read.rows[-1]["note"]
    # 代次一 bump，这一段证据整体作废 —— 那才是「重做这个偏压」的入口。
    bumped = L.read_ledger(L.ledger_path(tmp_path), current_evidence_epoch=4)
    assert all(r["stale"] is True for r in bumped.rows)


# ══════════════════════════════════════════════════════════════════════
# 6. 下发前的尺度预检 + 输入拒绝
# ══════════════════════════════════════════════════════════════════════

def test_an_unresolvable_scale_is_refused_before_any_frame(tmp_path):
    """扫完一张判不了的图再说，白花一帧的时间 —— 而且流程会把「判不了」
    误读成「还没弄出原子相」接着去扰动针尖。"""
    ctx = FakeCtx({})
    res = AtomicBiasSeries().run_composite(
        ctx, _params(tmp_path, size_m=2e-8, pixels=256))
    assert res.success is False
    assert ctx.calls == [], "已经知道判不出来了，却还是下发了帧"
    assert "判不出原子相" in res.error
    assert res.data["series_exit"] == L.EXIT_BLOCKED


def test_a_missing_ledger_dir_is_refused(tmp_path):
    """账没有落点，这条序列跑完也没有证据 —— 那就别跑。"""
    ctx = FakeCtx({})
    res = AtomicBiasSeries().run_composite(ctx, _params(tmp_path, ledger_dir=""))
    assert res.success is False and ctx.calls == []
    assert "ledger_dir" in res.error


def test_an_unexpandable_series_is_refused_not_invented(tmp_path):
    ctx = FakeCtx({})
    res = AtomicBiasSeries().run_composite(
        ctx, _params(tmp_path, bias_values="", bias_start=-0.4, bias_stop=0.4))
    assert res.success is False and ctx.calls == []
    assert "展不成一个序列" in res.error


def test_duplicate_biases_collapse_by_normalised_key(tmp_path):
    """-0.3 与 -0.30000000000000004 是同一个偏压，不是两个。"""
    res, _ctx, read = _run(tmp_path, _all_good(tmp_path),
                           bias_values="-0.3,-0.30000000000000004")
    assert res.data["n_biases"] == 1
    assert len({r["bias_key"] for r in read.rows}) == 1


# ══════════════════════════════════════════════════════════════════════
# 7. anchor 是无条件的
# ══════════════════════════════════════════════════════════════════════

def test_a_closing_anchor_is_always_taken(tmp_path):
    _res, _ctx, read = _run(tmp_path, _all_good(tmp_path), bias_values="-0.4")
    anchors = [r for r in read.rows if r["role"] == L.ROLE_ANCHOR]
    assert len(anchors) == 1, "收尾锚点是无条件的"
    assert anchors[0]["bias_key"] == L.bias_key(-0.4)


def test_intermediate_anchors_appear_at_the_requested_cadence(tmp_path):
    _res, _ctx, read = _run(tmp_path, _all_good(tmp_path),
                            bias_values="-0.4,-0.2,0,0.2,0.4",
                            anchor_every_n=2)
    anchors = [r for r in read.rows if r["role"] == L.ROLE_ANCHOR]
    assert len(anchors) == 3, f"每 2 个偏压一次 + 收尾一次:{len(anchors)}"
    assert {a["bias_key"] for a in anchors} == {L.bias_key(-0.4)}


def test_the_summary_says_when_there_is_no_anchor_evidence(tmp_path):
    """没有锚点时「扫的是同一片原子」是个无从证明的断言 —— summary 要说。"""
    skill = AtomicBiasSeries()
    data = {"series_exit": L.EXIT_COMPLETE, "n_biases": 1, "n_resolved": 1,
            "anchor_consistency": {"n_anchors": 0}, "unreadable": []}

    class _P:
        aborted = False

    class _Ex:
        progress = _P()
    res = skill._finalize(_Ex(), dict(data))
    assert "没有任何 anchor 帧" in res.summary


def test_the_anchor_cannot_corroborate_an_all_absent_series(tmp_path):
    """⚠️ **这条钉的是一个「做不到」,而它推翻了 S2 设计里的一条通过条件。**

    设计给「这些偏压上确实没有原子对比度」这条阴性结论的通过条件是
    「全部 absent_confirmed **且至少一张 anchor 帧 resolved**」。

    但 anchor 取的是**序列起始偏压**(``anchor_bias = ordered[0]``)。如果每一个
    偏压都没有对比度,起始偏压也没有 —— anchor 帧自己就判不出原子分辨。于是那个
    条件在它**唯一适用的场景里永远为假**。2026-08-15 有人照设计把它写进 campaign
    的 S2 出口闸门,写到一半被这条事实拦下来,规则撤回。

    根因:anchor 证的是「扫的是同一片、没漂移」,证不了「针尖当时有能力分辨原子」。
    后者需要一张**已知该有对比度的偏压**上的参照帧 —— 而那个偏压正是 S2 要去发现
    的东西。

    **这条测试变红 = 有人改了 anchor 的取法**(比如改成在用户声明的参照偏压上取)。
    那时上面那条通过条件才可能重新成立,而不是把断言改掉了事。
    """
    res, _ctx, _read = _run(tmp_path, _all_good(tmp_path, L.VERDICT_ABSENT))
    assert res.data["n_resolved"] == 0
    assert res.data["n_absent_confirmed"] == 3
    assert res.data["anchor_consistency"]["n_anchors"] == 1, "anchor 是拍了的"
    assert res.data["anchor_verdict"] == "unproven_anchor_unresolved", (
        "全 absent 的序列里 anchor 居然佐证成立了 —— 去看 anchor 现在取在哪个偏压上")


def test_the_anchor_does_corroborate_when_the_series_resolves(tmp_path):
    """上一条的对照:序列真的判出原子分辨时,anchor 佐证是成立的。

    少了这一半,上一条就只证明了「这个字段总是 unproven」,而不是
    「它在那个特定场景里做不到」——**一个恒为某值的判据,和一个判不出的判据,
    在只看一侧时长得一模一样**。
    """
    res, _ctx, _read = _run(tmp_path, _all_good(tmp_path, L.VERDICT_RESOLVED))
    assert res.data["anchor_verdict"] == "corroborated"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))


# ── 坐标代次闸(2026-08-15:从「只写账本」变成真的拒绝)──────────────────
#
# 在这之前它是一道**装得像在拦**的闸:参数描述逐字写着「A coarse move
# invalidates them」,而这个值全程只被写进账本一列 —— 全文件没有
# `from mast.core.coord_epoch`,五个子步骤也都不查。一个名字、一句描述、一列账,
# **没有任何东西会因为它停下来**。


def _called(ctx) -> list:
    """假上下文实际被派到的技能名。"""
    return [c[0] for c in ctx.calls]


def _epoch_run(tmp_path, monkeypatch, current, **over):
    """按给定的「当前代次」跑一次(`None` = 读不到)。"""
    from mast.core import coord_epoch as ce

    monkeypatch.setattr(ce, "read_current_epoch", lambda: current)
    skill = AtomicBiasSeries()
    ctx = FakeCtx(_all_good(tmp_path))
    return skill.run_composite(ctx, _params(tmp_path, **over)), ctx


def test_a_coarse_move_since_the_centres_were_planned_scans_nothing(
        tmp_path, monkeypatch):
    """**一帧都不扫。** 粗动之后同一个 (x, y) 是另一片表面,而这一步要拿着它
    连扫几个小时。"""
    res, ctx = _epoch_run(tmp_path, monkeypatch, CURRENT_EPOCH + 1)
    assert res.success is False
    assert res.data.get("refusal_code") == "coord_epoch_stale"
    assert res.data.get("requested_coord_epoch") == CURRENT_EPOCH
    assert res.data.get("current_coord_epoch") == CURRENT_EPOCH + 1
    assert "ScanAt" not in _called(ctx), "拒绝之后还是扫了"
    assert "BiasSettleChange" not in _called(ctx)


def test_an_unreadable_generation_is_refused_not_waved_through(
        tmp_path, monkeypatch):
    """**「查不到」不能当成「对得上」。**

    通用模块的默认策略是放行加告警 —— 那是给「省略代次是常态」的调用方定的。
    这里不是:中心坐标是用户在 approve 时填的,可能是几天前。
    加严在**这里**,模块本身一个字不改(与 `SpectroscopyAtPositions` 同一条)。
    """
    res, ctx = _epoch_run(tmp_path, monkeypatch, None)
    assert res.success is False
    assert res.data.get("refusal_code") == "coord_epoch_unverifiable"
    assert "更严" in res.error
    assert "ScanAt" not in _called(ctx)


def test_omitting_the_generation_is_refused_too(tmp_path, monkeypatch):
    """不给章 ⇒ 拒绝。**以前不给是允许的,而且悄悄地什么都不守。**"""
    res, ctx = _epoch_run(tmp_path, monkeypatch, CURRENT_EPOCH,
                          expected_coord_epoch=None)
    assert res.success is False
    assert res.data.get("refusal_code") == "coord_epoch_unstamped"
    assert "ScanAt" not in _called(ctx)


def test_a_matching_generation_runs_the_whole_series(tmp_path, monkeypatch):
    """对照:章对得上就照跑 —— 闸门只在该拦的时候拦。"""
    res, ctx = _epoch_run(tmp_path, monkeypatch, CURRENT_EPOCH)
    assert res.success is True
    assert "ScanAt" in _called(ctx)


def test_the_refusal_never_converts_or_clamps_the_generation(
        tmp_path, monkeypatch):
    """陈旧坐标的处置只有一个:**拒绝,让人重新规划**。既不夹紧也不换算 ——
    换算出来的坐标看上去和真坐标一模一样,而它是编的。"""
    res, _ctx = _epoch_run(tmp_path, monkeypatch, CURRENT_EPOCH + 3)
    assert "重新规划" in res.error
    assert "不做跨代次换算" in res.error


def test_the_gate_is_not_just_a_ledger_column(tmp_path, monkeypatch):
    """⚠️ 钉住那个**回归形状**:把 verify 拿掉、只留下写账本那一行。

    那正是 2026-08-15 之前的样子 —— 账本里有一列 `coord_epoch`,读的人以为
    它被查过。**一列账不是一道闸。**
    """
    import inspect

    from mast.skills.composite import atomic_bias_series as mod

    src = inspect.getsource(mod.AtomicBiasSeries._epoch_gate)
    assert "coord_epoch" in src and "verify" in src, (
        "代次闸不再调 verify 了 —— 它又变回了一列账")
    # 而且它真的被 run_composite 调着(不是一个没人叫的方法)。
    assert "_epoch_gate" in inspect.getsource(mod.AtomicBiasSeries.run_composite)
