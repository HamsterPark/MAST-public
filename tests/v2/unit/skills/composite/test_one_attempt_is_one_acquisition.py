# -*- coding: utf-8 -*-
"""max_attempts=N 必须触发 N 次独立采集；模拟器以缓冲代次区分采集，文件数量不作证据。"""
from __future__ import annotations

import pytest

from mast.core.types import SkillResult
from mast.skills.composite import scan_until_atomic as SUA
from mast.skills.composite.scan_until_atomic import (
    JUDGED_VERDICTS,
    ScanUntilAtomicResolution,
)


@pytest.fixture(autouse=True)
def _isolated_project_root(tmp_path, monkeypatch):
    """sidecar 落在 tmp 上 —— 绝不写用户真实的 experiments/。"""
    monkeypatch.setenv("MAST2_PROJECT_ROOT", str(tmp_path))


# ── 假仪器:一个缓冲代次 + 一个「存盘总给新文件名」的存盘 ────────────────


class _Rig:
    """替身区分采集代次与文件名。"""

    run_id = "rig-run-1"

    def __init__(self, *, scan_ok=lambda n: True, save_path=None,
                 latest_path=None, verdicts=None):
        self.calls: list[str] = []
        self._scan_ok = scan_ok
        self._save_path = save_path      # None ⇒ 正常发新文件名
        self._latest_path = latest_path  # GetLatestScanFile 的回答
        self._verdicts = verdicts or {}
        self.generation = 0              # 缓冲里躺着第几次采集的数据
        self.n_scan_started = 0
        self.n_files = 0
        #: 文件名 → 它承载的采集代次。判据据此给读数。
        self.files: dict[str, int] = {}

    def check_abort(self):
        return False

    def run(self, skill, params=None, version=None):
        self.calls.append(skill)
        if skill == "GetScanFrame":
            return SkillResult(skill_name=skill, success=True, data={
                "center_x_m": 0.0, "center_y_m": 0.0, "width_m": 5e-9})
        if skill == "GetScanBuffer":
            return SkillResult(skill_name=skill, success=True,
                               data={"pixels": 256})
        if skill == "GetScanSpeed":
            return SkillResult(skill_name=skill, success=True,
                               data={"fwd_speed_m_s": 6.51e-9})
        if skill == "FullScan":
            self.n_scan_started += 1
            if not self._scan_ok(self.n_scan_started):
                return SkillResult(
                    skill_name=skill, success=False,
                    error="scan timed out at 12/256 lines")
            self.generation += 1          # ← **只有这里**产生新数据
            return SkillResult(skill_name=skill, success=True, data={
                "wait_outcome": "completed", "scan_lines_done": 256,
                "scan_lines_total": 256, "scan_lines_verified": True})
        if skill == "SaveScan":
            if self._save_path == "none":
                return SkillResult(skill_name=skill, success=True, data={})
            self.n_files += 1
            name = "F%04d.sxm" % (454 + self.n_files)
            self.files[name] = self.generation
            return SkillResult(skill_name=skill, success=True,
                               data={"saved_path": name})
        if skill == "GetLatestScanFile":
            p = self._latest_path or (
                sorted(self.files)[-1] if self.files else "")
            return SkillResult(skill_name=skill, success=True, data={"path": p})
        if skill == "AssessAtomicResolution":
            gen = self.files.get(str(params.get("scan_path")), 0)
            v = self._verdicts.get(gen, "absent")
            # 读数**只由代次决定** —— 同一代次判两次 ⇒ 逐位相同。
            return SkillResult(skill_name=skill, success=True, data={
                "verdict": v, "angular_concentration": 90.0 + 5.5 * gen,
                "coverage": 1.0})
        return SkillResult(skill_name=skill, success=True, data={})


def _run(rig, **params):
    p = {"max_attempts": 3, "relocate_after_attempts": 0}
    p.update(params)
    return ScanUntilAtomicResolution().execute(rig, p)


def _judged(res):
    return [h for h in res.data["history"] if h.get("verdict") in JUDGED_VERDICTS]


# ── ① 健康的一轮:三次 attempt = 三次互不相同的采集 ─────────────────────


def test_three_attempts_are_three_distinct_acquisitions():
    """反证的另一半:一切正常时**三次就是三次**,新加的账不许把好路径卡住。"""
    rig = _Rig()
    res = _run(rig)
    assert res.success, res.error
    assert rig.n_scan_started == 3
    assert rig.generation == 3, "三次 attempt 只产生了 %d 次采集" % rig.generation
    assert res.data["scans_ok"] == 3
    assert res.data["scans_failed"] == 0
    assert res.data["repeat_frames"] == 0
    assert res.data["frames_judged"] == 3

    # **判过的每一帧来自不同的采集** —— 这是「N 次 attempt = N 次真扫描」
    # 唯一说得清的写法。数文件、数调用次数都证不到它。
    gens = [rig.files[h["path"]] for h in _judged(res)]
    assert sorted(gens) == [1, 2, 3], gens


# ── ② 事故本身:扫失败了,不许存盘、不许判 ────────────────────────────


def test_a_failed_scan_is_not_saved_and_not_judged():
    """扫描失败时不得保存旧缓冲或据此生成新的判据样本。"""
    rig = _Rig(scan_ok=lambda n: n == 1)
    res = _run(rig)

    assert rig.n_scan_started == 3, "三次 attempt 该发三次扫描"
    assert rig.generation == 1, "只有第一次真的采到了"
    # 只存了一次盘 —— 后两次失败的 attempt **一个文件都没造**
    assert rig.n_files == 1, (
        "扫失败了还去存盘 —— 存出来的是上一帧的缓冲,%d 个文件" % rig.n_files)
    assert rig.calls.count("AssessAtomicResolution") == 1, (
        "同一帧被判了 %d 次" % rig.calls.count("AssessAtomicResolution"))

    assert res.data["scans_ok"] == 1
    assert res.data["scans_failed"] == 2
    assert res.data["frames_judged"] == 1, (
        "报告说采到并判过 %d 帧,实际只采到 1 帧" % res.data["frames_judged"])
    rows = [h for h in res.data["history"] if h["verdict"] == "no_scan"]
    assert len(rows) == 2, res.data["history"]
    assert all("timed out" in str(h.get("error")) for h in rows), (
        "没把「为什么没采到」记下来:%s" % rows)


def test_a_failed_scan_is_not_evidence_about_the_tip():
    """``no_scan`` **不许**混进 ``absent`` / ``undetermined``。

    上层(``achieve_atomic`` 的两道分诊、本技能的换地方计数)只认那三个词。
    把「没采到」算成「没有原子分辨」,就是拿一次扫描故障当成针尖的证据 ——
    下一步是去磨一根可能本来就好的针。
    """
    rig = _Rig(scan_ok=lambda n: n == 1)
    res = _run(rig)
    bad = [h for h in res.data["history"]
           if h["verdict"] == "no_scan" and h["verdict"] in JUDGED_VERDICTS]
    assert not bad
    assert res.data["n_undetermined"] == 0, "没采到被当成了「判不了」"
    assert [h["verdict"] for h in _judged(res)] == ["absent"]


def test_a_run_that_acquired_nothing_says_so_instead_of_blaming_the_tip():
    """一帧都没采到 ⇒ advice 必须说「这里没有关于针尖的证据」。

    上一版这种跑给出的是「扫了 3 帧仍无原子分辨……再下一档才是修针」——
    一句把扫描故障翻译成针尖诊断的话。
    """
    rig = _Rig(scan_ok=lambda n: False)
    res = _run(rig)
    assert res.data["frames_judged"] == 0
    adv = res.data["advice"]
    assert "一帧都没有真的采到" in adv, adv
    assert "没有任何关于针尖的证据" in adv, adv
    assert "别据此去修针" in adv, adv


# ── ③ 同一帧不许判第二次 ─────────────────────────────────────────────


def test_the_latest_file_fallback_never_re_judges_the_same_frame():
    """存盘没给路径 ⇒ 兜底去拿「最新的一张」—— 那多半是**上一次存的那张**。

    路径重复是免费而确定的证据:同一个文件不可能是两次采集。
    ``save_every_frame=False`` 时这条兜底是**唯一**的取图路径,于是三次 attempt
    全会拿到同一个文件名 —— 以前它们就变成了三条 ``absent``。
    """
    rig = _Rig(latest_path="F0455.sxm")
    rig.files["F0455.sxm"] = 9            # 盘上已经有一张(上一档留下的)
    res = _run(rig, save_every_frame=False)

    verdicts = [h["verdict"] for h in res.data["history"]]
    assert verdicts.count("no_new_frame") == 2, verdicts
    assert res.data["frames_judged"] == 1, (
        "同一张图被算成了 %d 帧" % res.data["frames_judged"])
    assert rig.calls.count("AssessAtomicResolution") == 1, (
        "同一张图送了判据 %d 次" % rig.calls.count("AssessAtomicResolution"))
    assert res.data["repeat_frames"] == 2
    assert "没给出新样本" in res.data["advice"], res.data["advice"]


def test_with_no_file_at_all_the_attempt_is_not_a_verdict():
    """兜底也拿不到文件时,那一次 attempt 是 ``no_file`` —— 不是「没有原子分辨」。"""
    rig = _Rig(latest_path="")
    res = _run(rig, save_every_frame=False, max_attempts=2)
    assert [h["verdict"] for h in res.data["history"]] == ["no_file", "no_file"]
    assert res.data["frames_judged"] == 0
    assert res.data["n_undetermined"] == 0


def test_bit_identical_readings_are_flagged_not_counted_as_a_second_sample():
    """逐位相同的重复读数保留原判决，但不重复计作独立采样，并在报告中标记。"""
    # FullScan 报成功但**不产生新数据**(缓冲代次不动)—— 正是「一步不跑地
    # 报成功」在下游看到的样子。
    class _Stuck(_Rig):
        def run(self, skill, params=None, version=None):
            if skill == "FullScan":
                self.calls.append(skill)
                self.n_scan_started += 1
                return SkillResult(skill_name=skill, success=True, data={
                    "wait_outcome": "completed", "scan_lines_done": 256})
            return super().run(skill, params, version)

    rig = _Stuck()
    rig.generation = 7                      # 缓冲里躺着上一帧
    res = _run(rig)
    assert rig.n_files == 3, "三次都存了盘(真机就是这样)"
    flagged = [h for h in res.data["history"]
               if h.get("same_readings_as_previous")]
    assert len(flagged) == 2, res.data["history"]
    assert res.data["frames_judged"] == 1, (
        "三个文件、同一块数据,却算成了 %d 帧" % res.data["frames_judged"])
    assert "没给出新样本" in res.data["advice"], res.data["advice"]


def test_an_unreadable_concentration_is_never_read_as_identical():
    """判据读数没到手(None)时**什么都不推断** —— 「读不到」不是「相同」。"""

    class _NoConc(_Rig):
        def run(self, skill, params=None, version=None):
            r = super().run(skill, params, version)
            if skill == "AssessAtomicResolution":
                return SkillResult(skill_name=skill, success=True,
                                   data={"verdict": "absent", "coverage": 1.0})
            return r

    rig = _NoConc()
    res = _run(rig)
    assert res.data["repeat_frames"] == 0, (
        "把两个「读不到」当成了「读数相同」:%s" % res.data["history"])
    assert res.data["frames_judged"] == 3


# ── ③b 判据没跑成 ≠ 没有原子分辨 ─────────────────────────────────────


def test_a_failed_assessment_is_not_folded_into_absent():
    """判定这一步也是 ``optional=True``,而 ``… else "absent"`` 会把一个空回包
    折叠成一条 ``absent`` —— 「读不到」被念成「量到了,没有」。

    而 ``absent`` 是**关于针尖的证据**:上层据此升级去修针。同一个形状本仓
    一天里出现过五次,而它就在扫描那个 bug 往下三行。
    """
    class _NoJudge(_Rig):
        def run(self, skill, params=None, version=None):
            if skill == "AssessAtomicResolution":
                self.calls.append(skill)
                return SkillResult(skill_name=skill, success=False,
                                   error="cannot read sxm: truncated header")
            return super().run(skill, params, version)

    rig = _NoJudge()
    res = _run(rig, max_attempts=2)
    verdicts = [h["verdict"] for h in res.data["history"]]
    assert verdicts == ["no_verdict", "no_verdict"], verdicts
    assert "absent" not in verdicts, "判据没跑成被念成了「没有原子分辨」"
    assert res.data["assess_failed"] == 2
    assert res.data["frames_judged"] == 0
    assert res.data["scans_ok"] == 2, "扫描是好的,别把它也算成坏的"
    adv = res.data["advice"]
    assert "判据一次都没跑成" in adv, adv
    assert "没有任何关于针尖的证据" in adv, adv
    assert "truncated header" in str(res.data["history"][0]["error"])


# ── ④ 找到了就停,而且账要对 ──────────────────────────────────────────


def test_a_hit_stops_the_loop_and_the_ledger_still_adds_up():
    rig = _Rig(verdicts={2: "atomic"})
    res = _run(rig)
    assert res.data["found"] is True
    assert res.data["found_at_attempt"] == 2
    assert rig.n_scan_started == 2, "拿到了还接着扫"
    assert res.data["frames_judged"] == 2
    assert "advice" not in res.data


# ── ⑤ 「全是判不了」那道分支的分母 ────────────────────────────────────


def test_the_all_undetermined_branch_counts_judged_rows_not_history_rows():
    """分母是**判过的行数**,不是 ``len(history)``。

    history 里还有 ``relocated`` / ``no_scan`` / ``no_new_frame`` 这些记账行。
    拿 ``len(history)`` 当分母时,只要发生过一次换地方,这条判定就**永远不成立**
    —— 一道看着在防护、其实从没触发过的分支。而它要说的那句话
    (「这不是没有原子分辨,是每一帧都没扫完」)恰恰是最要紧的那一句。
    """
    from mast.skills.composite.graph_executor import CompositeProgress

    prog = CompositeProgress("ScanUntilAtomicResolution")
    prog.partial_data.update({
        "attempts_done": 3, "relocations": 1, "scans_ok": 3,
        "history": [
            {"attempt": 1, "verdict": "undetermined"},
            {"attempt": 2, "verdict": "undetermined"},
            {"attempt": 2, "verdict": "relocated"},      # ← 记账行,不是一帧
            {"attempt": 3, "verdict": "undetermined"},
        ]})
    out = ScanUntilAtomicResolution().aggregate({}, prog)
    assert "全部**判不了**" in out["advice"], out["advice"]
    assert "没扫完" in out["advice"]


# ── ⑥ 单一真源 ───────────────────────────────────────────────────────


def test_the_judged_verdicts_are_read_from_one_place():
    """``achieve_atomic`` 的分诊必须 **import** 这三个词,不许抄第二份。

    抄一份的后果:这边加了 ``no_scan`` 这样的记账行,那边不知道,于是「没采到」
    被那道分诊当成一帧真判定 —— 而那道分诊的全部职责就是别把非证据当证据。
    """
    import inspect

    from mast.skills.composite import achieve_atomic as A

    src = inspect.getsource(A.AchieveAtomicResolution.plan_dynamic)
    assert "JUDGED_VERDICTS" in src, "分诊没用共享常量"
    assert '"atomic", "absent", "undetermined"' not in src, (
        "又抄了一份判定词表 —— 两份迟早只有一份对")
    assert JUDGED_VERDICTS == frozenset({"atomic", "absent", "undetermined"})
