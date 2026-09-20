"""SpectroscopyAtPositions —— S4 STS 设计 §5.3 的全套钉子。

这个 composite 里最容易**静默**出错的一条是 .dat 归属：`AcquireSTS` 找文件的办法
是「最近 120 秒内最新的那个」，批量循环里只要有一个点没落盘，下一个点就会拿到
上一个点的文件，并把它当成自己的数据 —— 没有任何一条下游判据会报警，因为两个
文件长得一模一样。所以本文件里最长的那两条测试都在钉这件事：一条钉「文件名对不
上就不要」，一条钉「同名但是采集之前就存在的旧文件也不要」。两条防线互相独立，
各自单独失效都必须被抓住。

其余各条对应设计里的：D11(移动失败不采 / suspect 只给位置不可信) ·
D12(归属三层) · D13(代次闸门两个拒绝分支) · D14(预算不是判决) ·
D15(部分成功要说出缺口) · D16(收尾关调制，中止时不关)。

跑法：
    .venv-v2-py313/Scripts/python.exe -m pytest \
        tests/v2/unit/skills/composite/test_spectroscopy_at_positions.py -q
"""
from __future__ import annotations

# ── path bootstrap ──
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

import json
import os
import time
from dataclasses import dataclass, field

import pytest

from mast.core.types import NanonisCallRecord, SkillResult
from mast.skills.composite import spectroscopy_at_positions as sap
from mast.skills.composite.spectroscopy_at_positions import SpectroscopyAtPositions

#: 测试位置的 x 间隔。索引由 x 反解出来（``x = (i+1) * _DX``），这样假硬件不必
#: 数调用次数就知道现在在第几个点 —— 数调用次数在有跳过的用例里会错位。
_DX = 1e-8


def _positions(n: int) -> str:
    return json.dumps([{"x_m": (i + 1) * _DX, "y_m": 0.0, "label": f"P{i + 1}"}
                       for i in range(n)])


def _idx_from_x(x: float) -> int:
    return int(round(float(x) / _DX)) - 1


_DAT_TEMPLATE = (
    "Experiment\tbias spectroscopy\n"
    "X (m)\t{x:.6E}\n"
    "Y (m)\t{y:.6E}\n"
    "Z (m)\t-1.219330E-9\n"
    "[DATA]\n"
    "Bias calc (V)\tCurrent (A)\tCurrent [bwd] (A)\n"
    "-1.0\t1.0E-12\t1.0E-12\n"
    "0.0\t0.0\t0.0\n"
    "1.0\t-1.0E-12\t-1.0E-12\n"
)


@dataclass
class FakeCtx:
    """假硬件 + 假文件系统。每个行为都由一张按**点序号**索引的脚本表驱动。"""

    dat_dir: Path
    #: idx → "ok" | "fail" | "no_arrival"
    move_plan: dict = field(default_factory=dict)
    #: idx → "ok" | "no_save"(返回上一个点的文件) | "stale"(返回同名旧文件)
    #:      | "fail" | "no_path"(成功但没有 path)
    acquire_plan: dict = field(default_factory=dict)
    #: idx → AssessSpectrum 的 verdict
    verdicts: dict = field(default_factory=dict)
    #: idx → (dx, dy) 写进 .dat 头的 xy 相对命令 xy 的偏移(米)
    dat_offset: dict = field(default_factory=dict)
    #: idx → True 表示这一点的 .dat 头不写 xy(位置核不了)
    dat_no_xy: dict = field(default_factory=dict)
    #: 跑到第几次 run() 之后 check_abort 开始返回 True(-1 = 永不)
    abort_after_runs: int = -1
    #: AcquireSTS 里的假耗时(秒)，用来撑爆每点预算
    acquire_sleep_s: float = 0.0
    assess_fails: set = field(default_factory=set)

    run_log: list = field(default_factory=list)
    safe_calls: list = field(default_factory=list)
    written: list = field(default_factory=list)
    _last_xy: tuple = (0.0, 0.0)
    _tick: int = 0
    _base: float = field(default_factory=lambda: time.time() - 3600.0)

    # -- 文件 --------------------------------------------------------

    def _write_dat(self, basename: str, idx: int) -> Path:
        """写一个 .dat，mtime 逐个递增(与真机上一个接一个落盘同形)。"""
        x, y = self._last_xy
        dx, dy = self.dat_offset.get(idx, (0.0, 0.0))
        body = _DAT_TEMPLATE.format(x=x + dx, y=y + dy)
        if self.dat_no_xy.get(idx):
            body = "\n".join(ln for ln in body.split("\n")
                             if not ln.startswith(("X (m)", "Y (m)")))
        p = self.dat_dir / f"{basename}00001.dat"
        p.write_text(body, encoding="utf-8")
        self._tick += 1
        stamp = self._base + self._tick
        os.utime(p, (stamp, stamp))
        self.written.append(p)
        return p

    # -- ExecutionContext 表面 ---------------------------------------

    def run(self, skill_name: str, params: dict) -> SkillResult:
        self.run_log.append((skill_name, dict(params)))
        if skill_name == "MoveToXY":
            idx = _idx_from_x(params["x_m"])
            self._last_xy = (float(params["x_m"]), float(params["y_m"]))
            mode = self.move_plan.get(idx, "ok")
            if mode == "fail":
                return SkillResult(skill_name=skill_name, success=False,
                                   error="压电到不了那里")
            arrived = None if mode == "no_arrival" else True
            return SkillResult(skill_name=skill_name, success=True,
                               data={"x_m": params["x_m"], "y_m": params["y_m"],
                                     "arrived": arrived})
        if skill_name == "AcquireSTS":
            basename = params.get("save_basename") or ""
            idx = int(basename.rsplit("_p", 1)[1]) if "_p" in basename else 0
            if self.acquire_sleep_s:
                time.sleep(self.acquire_sleep_s)
            mode = self.acquire_plan.get(idx, "ok")
            if mode == "fail":
                return SkillResult(skill_name=skill_name, success=False,
                                   error="扫描没开起来")
            if mode == "no_path":
                return SkillResult(skill_name=skill_name, success=True,
                                   data={"acquisition_complete": True})
            if mode == "no_save":
                # 这一点**没有落盘**。真实的 `_attach_saved_dat` 会去找「最近
                # 120 秒里最新的 .dat」，那就是上一个点的文件 —— 一模一样地
                # 交回来。这正是要被抓住的那颗定时炸弹。
                stale = str(self.written[-1]) if self.written else None
                data = {"acquisition_complete": True}
                if stale:
                    data["path"] = stale
                return SkillResult(skill_name=skill_name, success=True, data=data)
            if mode == "foreign":
                # 采集期间**别人**往同一个目录写了一个更新的 .dat(用户在
                # Nanonis 界面上手存了一条谱)。它比采前水位新 ⇒ mtime 那一层
                # 放行；只有文件名那一层认得出它不是本点的。
                p = self._write_dat(f"manual_save{self._tick}", idx)
                return SkillResult(skill_name=skill_name, success=True,
                                   data={"acquisition_complete": True,
                                         "path": str(p)})
            if mode == "stale":
                # 文件名对得上(同一个 run_tag 重跑)，但它在这一点开采之前就
                # 已经躺在盘上了 —— 由测试**在开跑前**造出来。文件名那一层对它
                # 完全无能为力，只有 mtime 那一层抓得住。
                p = self.dat_dir / f"{basename}00001.dat"
                assert p.exists(), "stale 模式要求测试先造出那个旧文件"
                return SkillResult(skill_name=skill_name, success=True,
                                   data={"acquisition_complete": True,
                                         "path": str(p)})
            p = self._write_dat(basename, idx)
            return SkillResult(skill_name=skill_name, success=True,
                               data={"acquisition_complete": True, "path": str(p)})
        if skill_name == "AssessSpectrum":
            idx = self._assess_idx(params.get("dat_path", ""))
            if idx in self.assess_fails:
                return SkillResult(skill_name=skill_name, success=False,
                                   error="读不动")
            verdict = self.verdicts.get(idx, "keep")
            return SkillResult(skill_name=skill_name, success=True,
                               data={"verdict": verdict,
                                     "gated_criteria": ["saturation"],
                                     "ungated_criteria": ["spectrum_snr"],
                                     "reasons": []})
        return SkillResult(skill_name=skill_name, success=True, data={})

    @staticmethod
    def _assess_idx(path: str) -> int:
        stem = Path(str(path)).name
        return int(stem.rsplit("_p", 1)[1][:3]) if "_p" in stem else 0

    def safe_call(self, method: str, *args) -> NanonisCallRecord:
        self.safe_calls.append((method, args))
        # 读调制状态：回包形状照 `_read_mod_on` 的解析(rv[2][0])。
        return NanonisCallRecord(method=method, args=args,
                                 return_value=(0, 0, [0]), error="")

    def check_abort(self) -> bool:
        return (self.abort_after_runs >= 0
                and len(self.run_log) >= self.abort_after_runs)

    def emit_progress(self, progress) -> None:
        pass

    def checkpoint_flush(self) -> None:
        pass


@pytest.fixture
def ctx(tmp_path, monkeypatch):
    """假上下文 + 把「候选保存目录」指到 tmp。"""
    dat_dir = tmp_path / "dats"
    dat_dir.mkdir()
    from mast.skills.builtins import scan_extra

    monkeypatch.setattr(scan_extra, "_candidate_save_dirs",
                        lambda context: [dat_dir])
    return FakeCtx(dat_dir=dat_dir)


@pytest.fixture(autouse=True)
def _epoch_ok(monkeypatch):
    """默认让代次对得上；单独测闸门的用例自己再改。"""
    from mast.core import coord_epoch

    monkeypatch.setattr(coord_epoch, "read_current_epoch", lambda: 7)
    return 7


#: 一组**标定过的**谱学条件。出厂的 ``default`` 组刻意是未标定的(稳定条件与扫描
#: 窗口是样品事实，代码不发明)，所以每一条用例都得自己声明它在什么条件下取谱 ——
#: 这正是 D19 想要的形状：数字有出处，而出处不是这一层。
TEST_CONDITION = "tst"
COND_VALUES = dict(stab_bias_v=-0.7, stab_setpoint_a=100e-12,
                   start_v=-1.0, end_v=1.0, num_points=256,
                   mod_amp_v=0.005, mod_freq_hz=713.0, settle_s=0.5)

#: 第一个点之前一定会跑的条件建立步骤数：BiasSettleChange / SetSetpoint /
#: ConfigureLockIn / GetLockInConfig / ConfigureSTS。用它算偏移量，不写死数字 ——
#: 见 ``test_a_point_interrupted_mid_flight_is_not_drawn_as_done`` 里的理由。
_SETUP_STEPS = 5


@pytest.fixture(autouse=True)
def _condition_table(monkeypatch):
    """把一组标定过的条件挂进真表 —— 走的是**真的**组名解析，不是替身。

    用 ``monkeypatch.setitem`` 而不是整表替换：整表替换会顺手把「组名认不出来」
    那条路也换掉，而那正是要被测的一条。
    """
    from mast.core import sts_workflow as W

    monkeypatch.setitem(W.CONDITIONS, TEST_CONDITION,
                        W.STSConditionSpec(label=TEST_CONDITION, **COND_VALUES))
    return TEST_CONDITION


def _run(ctx, n=3, **kw) -> SkillResult:
    params = {"positions": _positions(n), "expected_coord_epoch": 7,
              "run_tag": "t1", "assess": True, "condition": TEST_CONDITION}
    params.update(kw)
    return SpectroscopyAtPositions().execute(ctx, params)


def _pt(res: SkillResult, i: int) -> dict:
    return res.data["points"][i]


# ── 输入校验 ─────────────────────────────────────────────────────────


def test_more_than_64_points_is_refused(ctx):
    res = _run(ctx, n=65)
    assert res.success is False
    assert "65" in res.error and "64" in res.error


def test_empty_and_malformed_positions_are_refused(ctx):
    assert SpectroscopyAtPositions().execute(
        ctx, {"positions": "[]", "expected_coord_epoch": 7}).success is False
    bad = SpectroscopyAtPositions().execute(
        ctx, {"positions": '[{"x_m": 1e-8}]', "expected_coord_epoch": 7})
    assert bad.success is False and "第 0 个点" in bad.error


def test_out_of_range_position_names_the_offender(ctx):
    # 把 100 nm 写成了 100 —— 单位滑落，整批拒绝并说是第几个。
    raw = json.dumps([{"x_m": 1e-8, "y_m": 0.0}, {"x_m": 100.0, "y_m": 0.0}])
    res = SpectroscopyAtPositions().execute(
        ctx, {"positions": raw, "expected_coord_epoch": 7})
    assert res.success is False and "第 1 个点" in res.error


def test_an_unknown_condition_group_is_refused_with_the_known_ones(ctx):
    """组名认不出来 ⇒ 拒绝，并报出**已知的组名单子**。

    「不知道有哪些」是这一类拒绝里最没用的一种回答；而就近匹配到一个相似的组
    更糟 —— 那会让一次跑错条件的运行看上去完全正常。
    """
    res = _run(ctx, n=1, condition="no_such_group")
    assert res.success is False
    assert "no_such_group" in res.error and TEST_CONDITION in res.error
    assert ctx.run_log == [], "拒绝了还发命令"


def test_an_uncalibrated_condition_group_is_refused_naming_the_gaps(ctx):
    """出厂的 ``default`` 组**刻意**没有稳定条件与扫描窗口 —— 拒绝并点名缺哪几个。

    这不是「还没做完」，是「这个问题现在没有答案」：编一组稳定条件出来，整批谱会
    正常落盘、正常被判据吃掉，而它们测的是一个没人给过的假设。同一条纪律在
    ``scan_prep_thresholds``（造 profile = 伪造标定）与 ``domain_reference``
    （没参照系就不出 label）里各写过一遍。
    """
    res = _run(ctx, n=1, condition="default")
    assert res.success is False
    for missing in ("stab_bias_v", "stab_setpoint_a", "start_v", "end_v"):
        assert missing in res.error, f"没点名缺的是 {missing}"
    assert ctx.run_log == []


def test_an_out_of_bounds_override_is_refused_not_clamped(ctx):
    """越界的覆写**拒绝，不夹紧也不丢弃**。

    夹紧会让调用方以为自己设的是 X 而实际跑的是别的数；丢弃连「你设过一个越界值」
    都不告诉任何人。两种兜底都合理得让人看不出兜底发生了。
    """
    res = _run(ctx, n=1, settle_s=999.0)      # _BOUNDS 上限 60 s
    assert res.success is False
    assert "settle_s" in res.error and "不夹紧" in res.error
    assert ctx.run_log == []


# ── D13 坐标代次闸门：两个拒绝分支 ───────────────────────────────────


def test_epoch_mismatch_refuses_and_does_not_touch_the_instrument(ctx, monkeypatch):
    from mast.core import coord_epoch

    monkeypatch.setattr(coord_epoch, "read_current_epoch", lambda: 9)
    res = _run(ctx, n=3, expected_coord_epoch=7)
    assert res.success is False
    assert res.data["refusal_code"] == coord_epoch.STALE
    assert ctx.run_log == []            # 一条命令都没发出去


def test_epoch_unreadable_also_refuses(ctx, monkeypatch):
    """「读不到」不是「一致」。

    注意这比 :mod:`mast.core.coord_epoch` 的默认策略严：那个模块对
    ``UNVERIFIABLE`` 是放行加告警。加严在**调用方**，模块本身不动 ——
    这条测试就是那句加严的执行体。
    """
    from mast.core import coord_epoch

    monkeypatch.setattr(coord_epoch, "read_current_epoch", lambda: None)
    res = _run(ctx, n=3)
    assert res.success is False
    assert res.data["refusal_code"] == coord_epoch.UNVERIFIABLE
    assert ctx.run_log == []


def test_epoch_module_default_would_have_let_it_through(monkeypatch):
    """钉住上一条的**前提**：模块自己并不认为「读不到」是拒绝。

    没有这一条，上一条测试通过时你分不出「调用方加严生效了」和「模块本来就
    拦」——而后者是假的，谁把加严删掉都不会有测试变红。
    """
    from mast.core import coord_epoch

    monkeypatch.setattr(coord_epoch, "read_current_epoch", lambda: None)
    v = coord_epoch.verify(7, what="x")
    assert v.state == coord_epoch.UNVERIFIABLE
    assert v.stale is False and v.verified is False


def test_epoch_match_runs_and_records_the_epoch(ctx):
    res = _run(ctx, n=2)
    assert res.success is True
    assert res.data["coord_epoch"] == 7


# ── D11 移动失败就不采 / suspect 只给位置不可信 ──────────────────────


def test_move_failure_skips_the_acquisition_entirely(ctx):
    ctx.move_plan = {1: "fail"}
    res = _run(ctx, n=3)
    assert res.success is True
    assert _pt(res, 1)["status"] == "move_failed"
    assert _pt(res, 1)["success"] is False
    assert _pt(res, 1)["path"] is None
    # 关键：那个点**根本没有** AcquireSTS —— 静态 plan 做不到这一点。
    basenames = [p.get("save_basename") for s, p in ctx.run_log if s == "AcquireSTS"]
    assert basenames == ["t1_p000", "t1_p002"]
    assert res.data["n_move_failed"] == 1


def test_arrived_none_is_not_treated_as_arrived(ctx):
    """``arrived`` 是三态。把 None 当 True 就是又一次「读不到被当成答了」。"""
    ctx.move_plan = {0: "no_arrival"}
    res = _run(ctx, n=2)
    assert _pt(res, 0)["status"] == "suspect"
    assert _pt(res, 0)["success"] is False
    assert "arrived" in (_pt(res, 0)["error"] or "")
    # 但它**采了** —— suspect 的含义就是「采了但位置不可信」。
    assert _pt(res, 0)["path"]
    assert _pt(res, 1)["status"] == "ok"


def test_position_mismatch_marks_suspect(ctx):
    """.dat 头 xy 与命令 xy 差 5 nm、容差 2 nm ⇒ suspect。"""
    ctx.dat_offset = {1: (5e-9, 0.0)}
    res = _run(ctx, n=3, position_tol_nm=2.0)
    assert _pt(res, 1)["status"] == "suspect"
    assert _pt(res, 1)["position_check"] == "mismatch"
    assert _pt(res, 0)["position_check"] == "match"
    assert res.data["n_suspect"] == 1
    # 它的文件不进 dat_paths（success 是 False）。
    assert _pt(res, 1)["path"] not in res.data["dat_paths"]


def test_position_within_tolerance_is_not_suspect(ctx):
    ctx.dat_offset = {1: (1e-9, 0.0)}
    res = _run(ctx, n=2, position_tol_nm=2.0)
    assert _pt(res, 1)["status"] == "ok"


def test_unreadable_dat_position_is_neither_match_nor_mismatch(ctx):
    """位置读不到 ⇒ ``unverified``，而且必须**说出来**。

    读不到既不该被当成对得上(那会让一条错位的谱以 ok 落地)，也不该被当成
    对不上(那会把一批好数据全打成 suspect)。
    """
    ctx.dat_no_xy = {1: True}
    res = _run(ctx, n=2)
    assert _pt(res, 1)["position_check"] == "unverified"
    assert _pt(res, 1)["status"] == "ok"
    assert res.data["n_position_unverified"] == 1
    assert "没有被核对过" in (res.summary or "")


# ── D12 .dat 归属三层 ────────────────────────────────────────────────


def test_missing_save_makes_that_point_unrated_and_never_leaks_to_the_next(ctx):
    """本设计里最容易静默出错的一条。

    第 2 个点没落盘 ⇒ 真实的 `_attach_saved_dat` 会把**第 1 个点的文件**当成它的
    数据交回来。这里断言三件事：那个点自己是 unrated 且没有 path；第 3 个点拿到
    的是自己的文件而不是第 1 个点的；`dat_paths` 里一个文件只出现一次。
    """
    ctx.acquire_plan = {1: "no_save"}
    res = _run(ctx, n=3)

    p1 = _pt(res, 1)
    assert p1["path"] is None
    assert p1["verdict"] == "unrated"
    assert p1["reason"] == "dat_attribution_failed"
    assert p1["success"] is False
    assert p1["status"] == "acquire_failed"

    # 第 3 个点没有继承第 1 个点的文件。
    p0_path, p2_path = _pt(res, 0)["path"], _pt(res, 2)["path"]
    assert p0_path and p2_path and p0_path != p2_path
    assert "t1_p000" in Path(p0_path).name
    assert "t1_p002" in Path(p2_path).name
    assert res.data["dat_paths"] == [p0_path, p2_path]
    assert len(set(res.data["dat_paths"])) == 2
    assert res.data["n_dat_attribution_failed"] == 1
    # 归属失败是「判不了」，不是「不合格」。
    assert res.data["n_discard"] == 0 and res.data["n_unrated"] == 1


def test_a_file_that_predates_the_acquisition_is_rejected_even_with_the_right_name(ctx):
    """第二层防线单独成立：同名但**采集之前就存在**的旧文件必须被拒。

    场景是真的：用户给了同一个 ``run_tag`` 重跑一遍，上一轮的 ``t1_p000…dat``
    还躺在保存目录里。文件名那一层对它完全无能为力(名字一模一样)，只有
    「mtime 严格高于采前水位」抓得住 —— 这就是两层必须都在的理由。
    """
    old = ctx.dat_dir / "t1_p00000001.dat"
    old.write_text(_DAT_TEMPLATE.format(x=1e-8, y=0.0), encoding="utf-8")
    stamp = time.time() - 86400.0
    os.utime(old, (stamp, stamp))

    ctx.acquire_plan = {0: "stale"}
    res = _run(ctx, n=2)
    assert _pt(res, 0)["path"] is None
    assert _pt(res, 0)["reason"] == "dat_attribution_failed"
    assert "mtime" in (_pt(res, 0)["error"] or "")
    assert _pt(res, 1)["status"] == "ok"


def test_a_newer_file_with_someone_elses_name_is_rejected(ctx):
    """第一层防线单独成立：比水位**新**、但名字不是本点的文件必须被拒。

    上一条钉的是 mtime 层；这一条钉的是文件名层。两条必须各自单独可证 ——
    否则删掉其中一层不会有任何测试变红，而报告会把「另一层顺手接住了」读成
    「两层都在起作用」(这条测试就是变异验证逼出来的)。
    """
    ctx.acquire_plan = {1: "foreign"}
    res = _run(ctx, n=3)
    p1 = _pt(res, 1)
    assert p1["path"] is None
    assert p1["reason"] == "dat_attribution_failed"
    assert "basename" in (p1["error"] or "")
    # 那个外来文件绝不能出现在产物里。
    assert not any("manual_save" in p for p in res.data["dat_paths"])
    assert _pt(res, 2)["status"] == "ok"


def test_acquire_without_any_path_is_unrated_not_success(ctx):
    """`AcquireSTS` 的 success=True 只说明 Start 没报错(设计 §1.1)。"""
    ctx.acquire_plan = {0: "no_path"}
    res = _run(ctx, n=2)
    assert _pt(res, 0)["success"] is False
    assert _pt(res, 0)["verdict"] == "unrated"
    assert _pt(res, 0)["reason"] == "dat_attribution_failed"


def test_every_point_gets_its_own_basename(ctx):
    _run(ctx, n=4)
    names = [p["save_basename"] for s, p in ctx.run_log if s == "AcquireSTS"]
    assert names == ["t1_p000", "t1_p001", "t1_p002", "t1_p003"]
    assert len(set(names)) == 4


def test_watermark_is_taken_before_each_acquisition(ctx, monkeypatch):
    """水位必须在采集**之前**取：取晚了，本点自己落的盘会把它顶上去。"""
    seen: list[str] = []
    real = sap._dat_mtime_watermark
    monkeypatch.setattr(sap, "_dat_mtime_watermark",
                        lambda c: (seen.append("watermark"), real(c))[1])
    orig_run = ctx.run

    def spy(name, params):
        if name == "AcquireSTS":
            seen.append("acquire")
        return orig_run(name, params)

    ctx.run = spy  # type: ignore[method-assign]
    _run(ctx, n=2)
    assert seen == ["watermark", "acquire", "watermark", "acquire"]


def test_unenumerable_save_dirs_make_points_unrated_not_ok(ctx, monkeypatch):
    """水位查不了 ⇒ 归属证不了 ⇒ unrated。「查不了」不是「通过」。"""
    monkeypatch.setattr(sap, "_dat_mtime_watermark", lambda c: None)
    res = _run(ctx, n=2)
    assert res.success is False          # 0 个点成功 = 硬失败
    assert all(p["verdict"] == "unrated" for p in res.data["points"])
    assert res.data["n_dat_attribution_failed"] == 2


# ── D14 预算不是判决 ─────────────────────────────────────────────────


def test_three_consecutive_discards_stop_the_batch(ctx):
    ctx.verdicts = {0: "discard", 1: "discard", 2: "discard"}
    res = _run(ctx, n=6, stop_after_consecutive_discard=3)
    assert res.data["stopped_early"] is True
    assert "discard" in res.data["stopped_reason"]
    assert [p["status"] for p in res.data["points"]][3:] == ["skipped"] * 3
    # 早停不做补救 —— 没有任何修针 / 换点动作。
    assert not any(s.startswith(("TipPulse", "Condition", "Shape", "Forge"))
                   for s, _ in ctx.run_log)
    assert "不做任何补救" in (res.summary or "")


def test_three_consecutive_unrated_do_not_stop(ctx):
    """「读不到」不是「不合格」—— unrated 不计入 consecutive_discard。"""
    ctx.verdicts = {0: "unrated", 1: "unrated", 2: "unrated"}
    res = _run(ctx, n=5, stop_after_consecutive_discard=3)
    assert res.data["stopped_early"] is False
    assert res.data["n_unrated"] == 3
    assert all(p["order"] is not None for p in res.data["points"])


def test_a_keep_between_discards_resets_the_streak(ctx):
    ctx.verdicts = {0: "discard", 1: "discard", 2: "keep", 3: "discard",
                    4: "discard"}
    res = _run(ctx, n=5, stop_after_consecutive_discard=3)
    assert res.data["stopped_early"] is False


def test_stop_after_zero_disables_the_early_stop(ctx):
    ctx.verdicts = {i: "discard" for i in range(5)}
    res = _run(ctx, n=5, stop_after_consecutive_discard=0)
    assert res.data["stopped_early"] is False
    assert res.data["n_discard"] == 5


def test_per_point_budget_stops_before_the_next_point(ctx):
    res = _run(ctx, n=4, per_point_timeout_s=1e-6)
    assert res.data["stopped_early"] is True
    assert "预算" in res.data["stopped_reason"]
    assert res.data["n_attempted"] == 1


def test_a_generous_per_point_budget_does_not_stop(ctx):
    res = _run(ctx, n=3, per_point_timeout_s=3600.0)
    assert res.data["stopped_early"] is False
    assert res.data["n_attempted"] == 3


def test_dedup_skips_points_that_are_too_close(ctx):
    raw = json.dumps([{"x_m": 1e-8, "y_m": 0.0},
                      {"x_m": 1e-8 + 3e-10, "y_m": 0.0},
                      {"x_m": 2e-8, "y_m": 0.0}])
    res = SpectroscopyAtPositions().execute(
        ctx, {"positions": raw, "expected_coord_epoch": 7, "run_tag": "t1",
              "condition": TEST_CONDITION, "min_point_separation_nm": 1.0})
    assert _pt(res, 1)["status"] == "skipped"
    assert _pt(res, 1)["order"] is None
    assert res.data["n_attempted"] == 2


def test_dedup_is_off_by_default(ctx):
    """「已测过」不等于「已损坏」：复测浪费时间，不伤表面 ⇒ 默认不去重。"""
    raw = json.dumps([{"x_m": 1e-8, "y_m": 0.0},
                      {"x_m": 1e-8 + 3e-10, "y_m": 0.0}])
    res = SpectroscopyAtPositions().execute(
        ctx, {"positions": raw, "expected_coord_epoch": 7, "run_tag": "t1",
              "condition": TEST_CONDITION})
    assert res.data["n_attempted"] == 2


# ── D15 部分成功 ─────────────────────────────────────────────────────


def test_zero_successful_points_is_a_hard_failure(ctx):
    ctx.move_plan = {0: "fail", 1: "fail", 2: "fail"}
    res = _run(ctx, n=3)
    assert res.success is False
    assert "0/3" in res.error
    assert res.data["dat_paths"] == []


def test_partial_success_says_the_gap_in_the_summary(ctx):
    """`fail_count` 躺在 data 里没人看 —— 缺口必须进 summary。"""
    ctx.move_plan = {1: "fail"}
    ctx.acquire_plan = {2: "fail"}
    res = _run(ctx, n=4)
    assert res.success is True
    assert res.data["success_count"] == 2
    assert "2/4" in (res.summary or "")
    assert "移动失败 1" in res.summary and "采集/归属失败 1" in res.summary


def test_full_success_summary_has_no_gap_wording(ctx):
    res = _run(ctx, n=3)
    assert res.success is True
    assert res.summary.startswith("3/3")
    assert "部分完成" not in res.summary


def test_summary_warns_when_nothing_actually_gated(ctx):
    """`gated_criteria` 为空 ⇒ n_keep 只是「采到了」，不是「合格」。"""
    ctx.verdicts = {0: "keep"}
    res = _run(ctx, n=1, assess=False)
    assert res.data["gated_criteria"] == []
    assert "不等于「合格」" in res.summary
    assert res.data["n_unassessed"] == 1


def test_assess_failure_is_unrated_not_discard(ctx):
    ctx.assess_fails = {1}
    res = _run(ctx, n=3)
    assert _pt(res, 1)["verdict"] == "unrated"
    assert _pt(res, 1)["reason"] == "assess_failed"
    assert res.data["n_discard"] == 0


# ── 记录层契约：points 键名 + success 显式性 ─────────────────────────


def test_points_key_and_explicit_success_reach_the_recorder(ctx):
    """把 aggregate 喂给真的 ``_marker_subrecords``。

    键名叫别的(比如 spectra)会静默退回单标记路径 —— 一整批谱在地图上只剩一个点；
    缺 ``success`` 字段会被当成 True —— 一个没跑成的点以 done 落在地图上。
    """
    from mast.core.runtime import _marker_subrecords

    ctx.move_plan = {1: "fail"}
    res = _run(ctx, n=3)
    recs = _marker_subrecords(res.data)
    assert len(recs) == 3
    assert recs[1]["success"] is False       # 显式 False，不是缺省
    assert recs[0]["success"] is True
    assert [r["x_m"] for r in recs] == pytest.approx([1e-8, 2e-8, 3e-8])
    assert recs[0]["artifact_path"] == _pt(res, 0)["path"]
    assert recs[1]["artifact_path"] is None


def test_every_point_record_carries_success_explicitly(ctx):
    ctx.move_plan = {0: "fail"}
    ctx.acquire_plan = {1: "fail"}
    res = _run(ctx, n=4, stop_after_consecutive_discard=0)
    for rec in res.data["points"]:
        assert "success" in rec and isinstance(rec["success"], bool)
        assert rec["status"] in sap.POINT_STATUSES


def test_a_point_interrupted_mid_flight_is_not_drawn_as_done(ctx):
    """中止在一个点**做到一半**时，那个点的记录必须已经带着显式 success=False。

    这是「缺省 success 视为 True」真正会咬人的地方：跑完的点都会被收口函数写上
    success，**只有做到一半的那个**是靠记录初始化时就写死 False 才没有以 done
    落在地图上。上一条测试抓不到它(它的每个点都收口了)——这条测试是变异验证
    逼出来的第二条。
    """
    from mast.core.runtime import _marker_subrecords

    # 条件建立的五步在前(bias / setpoint / lockin / 回读 / STS)，再一步是 p000
    # 的 move —— 中止落在它跑完、acquire 还没开始的那一刻。**由常量算出来**：
    # 写死一个 6 会在条件建立多一步的那天变成「中止在 setup 里」，而那条测试
    # 会照样绿(它断言的是 aborted=True)。
    ctx.abort_after_runs = _SETUP_STEPS + 1
    res = _run(ctx, n=3)
    assert res.data["aborted"] is True
    in_flight = _pt(res, 0)
    assert in_flight["order"] == 1          # 它确实被开工了
    assert in_flight["path"] is None        # 但没采到东西
    assert in_flight["success"] is False
    recs = _marker_subrecords(res.data)
    assert all(r["success"] is False for r in recs)


# ── D16 条件建立走正门 + 收尾 ────────────────────────────────────────


def test_stabilisation_bias_goes_through_bias_settle_not_setbias(ctx):
    """恒流反馈下偏压穿零会把针尖推进样品 —— 这条路只有一条正门。"""
    _run(ctx, n=1)
    names = [s for s, _ in ctx.run_log]
    assert "BiasSettleChange" in names
    assert "SetBias" not in names
    assert names.index("BiasSettleChange") < names.index("MoveToXY")
    settle = next(p for s, p in ctx.run_log if s == "BiasSettleChange")
    assert settle["bias_v"] == COND_VALUES["stab_bias_v"], (
        "稳定偏压不是条件组里那个数 —— 它是从哪来的？")
    assert settle["settle_s"] == COND_VALUES["settle_s"]


def test_the_condition_is_always_established_before_the_first_point(ctx):
    """条件**每一次**都建立，没有「什么都不给就什么都不改」这条路(D19)。

    旧的形状是「给了数才设」，于是「原先以为它沿用了仪器上的设置」和「它确实沿用了」
    长得一模一样 —— 而一批在未知条件下取的谱，没有任何下游判据会报警。现在条件是
    必填的一个**组名**，五个建立步骤一个都不能少。
    """
    _run(ctx, n=1)
    names = [s for s, _ in ctx.run_log]
    for step in ("BiasSettleChange", "SetSetpoint", "ConfigureLockIn",
                 "GetLockInConfig", "ConfigureSTS"):
        assert step in names, f"条件建立缺了 {step}"
        assert names.index(step) < names.index("MoveToXY"), (
            f"{step} 排在第一个点之后 —— 那个点是在旧条件下采的")


def test_modulation_is_configured_explicitly_and_read_back(ctx):
    """调制走 ``ConfigureLockIn`` 的**显式**传参 + 一次回读(D18 / 陷阱 11-12)。

    ``ApplyLockInPreset`` 自带写后回读、更安全，但它给不了「逐条谱不同幅度」，
    而条件表整件事就是在不同调制幅度下各取一条。显式传参同时绕开 修复项
    （省略的幅度/频率会被报成 0.0）；回读那一半靠紧随其后的 ``GetLockInConfig``
    补上 —— 没有它，「调制已配好」只是这条流程在把自己的请求复述给自己听。
    """
    _run(ctx, n=1)
    cfg = next(p for s, p in ctx.run_log if s == "ConfigureLockIn")
    assert cfg["amplitude_v"] == COND_VALUES["mod_amp_v"]
    assert cfg["frequency_hz"] == COND_VALUES["mod_freq_hz"]
    assert cfg["mod_on"] is True
    assert "phase_deg" not in cfg, (
        "本机的调制器根本没有相位字段，那次写入会被固件无条件拒绝")
    names = [s for s, _ in ctx.run_log]
    assert names.index("GetLockInConfig") == names.index("ConfigureLockIn") + 1


def test_a_lockin_readback_that_contradicts_stops_before_any_point(ctx):
    """回读**否定**了调制设置 ⇒ 一帧都不采。

    整批 dI/dV 会落在一个未知的调制幅度上，而每一条谱看上去都完全正常 ——
    这是几小时之后才发现的一种错误。⚠️ 与「读不到」必须分开：读不到只是警告。
    """
    orig = ctx.run

    def _wrong_readback(skill_name, params):
        if skill_name == "GetLockInConfig":
            ctx.run_log.append((skill_name, dict(params)))
            return SkillResult(skill_name=skill_name, success=True,
                               data={"mod_on": True, "amplitude": 0.05,
                                     "frequency_hz": 713.0})
        return orig(skill_name, params)

    ctx.run = _wrong_readback
    res = _run(ctx, n=2)
    assert not any(s == "AcquireSTS" for s, _ in ctx.run_log), "对不上还采了"
    assert res.data["stopped_early"] is True
    assert "amplitude" in res.data["stopped_reason"]


def test_an_unreadable_lockin_readback_only_warns(ctx):
    """「读不到」不是「不一致」—— 那会让一次读失败废掉整批已经排好的谱。

    但它也不是「对得上」：这一轮没有这层保护，必须出声说出来。
    """
    res = _run(ctx, n=1)          # 假上下文对 GetLockInConfig 回空 data
    assert res.success is True
    assert any(s == "AcquireSTS" for s, _ in ctx.run_log)
    assert any("读不到" in w for w in (res.data.get("warnings") or [])), (
        res.data.get("warnings"))


def test_the_sweep_window_comes_from_the_group_with_an_explicit_z_offset(ctx):
    """窗口三个数一起来自条件组，``z_offset_m`` **显式**给。

    它总是会被写：``ConfigureSTS`` 省略时按 0 处理，所以「配了窗口」顺带会清掉
    用户可能设过的保护性预退。显式的 0 与「没给」在结果里必须分得开。
    """
    _run(ctx, n=1)
    cfg = next(p for s, p in ctx.run_log if s == "ConfigureSTS")
    assert cfg == {"start_v": COND_VALUES["start_v"],
                   "end_v": COND_VALUES["end_v"],
                   "num_points": COND_VALUES["num_points"],
                   "z_offset_m": 0.0}


def test_the_condition_is_recorded_as_a_name_and_as_numbers(ctx):
    """组名与展开后的数字**两个都记**。

    只记组名：流程表改版之后，同一个名字指向另一组数，旧结果就没法回答「这批谱
    是在什么条件下取的」。只记数字：对不上是哪一组，也就没法复现。
    """
    res = _run(ctx, n=1)
    assert res.data["condition"] == TEST_CONDITION
    assert res.data["condition_values"]["stab_bias_v"] == COND_VALUES["stab_bias_v"]
    assert "稳定" in res.data["condition_summary"]


def test_the_settle_override_reaches_the_instrument(ctx):
    """``settle_s`` 是唯一保留的覆写口 —— 它是时间旋钮，不是物理设定点。"""
    _run(ctx, n=1, settle_s=3.0)
    settle = next(p for s, p in ctx.run_log if s == "BiasSettleChange")
    assert settle["settle_s"] == 3.0


def test_a_failed_condition_step_aborts_instead_of_acquiring(ctx, monkeypatch):
    orig = ctx.run

    def fail_setpoint(name, params):
        if name == "SetSetpoint":
            ctx.run_log.append((name, dict(params)))
            return SkillResult(skill_name=name, success=False, error="写不进去")
        return orig(name, params)

    ctx.run = fail_setpoint  # type: ignore[method-assign]
    res = _run(ctx, n=2, stab_setpoint_a=1e-10)
    assert res.success is False
    assert not any(s == "AcquireSTS" for s, _ in ctx.run_log)


def test_modulation_is_closed_at_the_end(ctx):
    """没有任何人会替含 "sts"/"spectr" 的流程关调制 —— 责任在终端消费者。"""
    res = _run(ctx, n=2)
    assert res.data["modulation_closed_by"] == "SpectroscopyAtPositions"
    assert ("LockIn_ModOnOffSet", (1, 0)) in ctx.safe_calls
    # 关调制的三次调用要出现在 nanonis_calls 里，免得出现一句在调用记录里
    # 找不到对应命令的「我关掉了」。
    assert any(c.method == "LockIn_ModOnOffSet" for c in res.nanonis_calls)


def test_modulation_is_not_closed_when_aborted(ctx):
    """中止不是流程结束(``_preflight`` 的分界)。"""
    ctx.abort_after_runs = 2
    res = _run(ctx, n=4)
    assert res.success is False
    assert res.data["aborted"] is True
    assert not any(m == "LockIn_ModOnOffSet" for m, _ in ctx.safe_calls)
    assert "modulation_closed_by" not in res.data


def test_refusal_before_the_run_does_not_touch_the_lockin(ctx, monkeypatch):
    from mast.core import coord_epoch

    monkeypatch.setattr(coord_epoch, "read_current_epoch", lambda: 99)
    _run(ctx, n=2)
    assert ctx.safe_calls == []


# ── sidecar ──────────────────────────────────────────────────────────


def _sidecar(ctx) -> Path:
    from mast.skills.composite import graph_executor as ge

    return ge._sidecar_path("SpectroscopyAtPositions",
                            str(getattr(ctx, "run_id", "") or ""))


def test_finished_run_clears_its_sidecar(ctx):
    """跑到头的运行绝不能给下一次同规模运行留 sidecar(2026-07-10 #95)。"""
    _run(ctx, n=2)
    assert not _sidecar(ctx).exists()


def test_aborted_run_keeps_its_sidecar_for_resume(ctx):
    ctx.abort_after_runs = 2
    _run(ctx, n=4)
    assert _sidecar(ctx).exists()


# ── 其它 ─────────────────────────────────────────────────────────────


def test_dat_paths_only_carry_successful_attributed_points(ctx):
    ctx.move_plan = {1: "fail"}
    ctx.acquire_plan = {2: "no_save"}
    ctx.dat_offset = {3: (9e-9, 0.0)}
    res = _run(ctx, n=5, position_tol_nm=2.0)
    good = [p["path"] for p in res.data["points"] if p["success"]]
    assert res.data["dat_paths"] == good
    assert len(res.data["dat_paths"]) == 2      # 只有第 1 和第 5 个点
    assert all(pth for pth in res.data["dat_paths"])


def test_assess_is_skipped_when_turned_off(ctx):
    res = _run(ctx, n=2, assess=False)
    assert not any(s == "AssessSpectrum" for s, _ in ctx.run_log)
    assert all(p["verdict"] is None for p in res.data["points"])
    assert res.data["success_count"] == 2


def test_spectral_family_is_forwarded_verbatim(ctx):
    _run(ctx, n=1, spectral_family="gapped")
    assess = next(p for s, p in ctx.run_log if s == "AssessSpectrum")
    assert assess["spectral_family"] == "gapped"


def test_counters_add_up(ctx):
    ctx.move_plan = {0: "fail"}
    ctx.acquire_plan = {1: "fail"}
    ctx.dat_offset = {2: (9e-9, 0.0)}
    res = _run(ctx, n=4, position_tol_nm=2.0)
    d = res.data
    assert d["n_planned"] == 4 == len(d["points"])
    assert (d["n_move_failed"] + d["n_acquire_failed"] + d["n_suspect"]
            + d["n_skipped"] + d["success_count"]) == 4
