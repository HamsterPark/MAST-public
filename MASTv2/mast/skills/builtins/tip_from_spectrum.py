"""从一条谱判**针尖** —— `assess_iz` / `assess_iv` 的技能外壳。

判据本体是 :func:`mast.vision.spectroscopy.assess_iz` 与
:func:`~mast.vision.spectroscopy.assess_iv` 两个纯函数（零 IO、只吃数组），
它们在这个仓库里存在了很久，**但一直没有任何 skill 壳**：只能从 Python 里直接
调，agent 调不到、composite 调不到、conduct 的闸门更调不到。于是 S4（STS）那道
「谱质量闸」在设计文档里写着，在运行时是一句空话——`AcquireSTS` 只要 Start 不
报错就 success。这个文件补的就是那一层壳。

## 它与 `AssessSpectrum` 是两件事，别合并

============  =====================================================
`AssessSpectrum`  **这条数据留不留**（饱和、信噪、正反扫迟滞…）
本文件            **这根针行不行**（I(z) 是不是单指数、I(V) 有没有跳变）
============  =====================================================

一条 `discard` 的谱可能只是窗口开错了、相位反了、表面不是那个面——针尖完全
没问题。反过来，一条数据质量挑不出毛病的谱，也可能是一根双针尖测出来的。
两件事分开放，是为了不让下游把「这条谱不好」读成「这根针不行」，然后去反复
修一根其实没问题的针（本仓在「在自己刚炸出来的坑上判针尖」上栽过六版）。

## 未标定的阈值只能说「判不了」

I(z) 的表观势垒有物理量纲（清洁金属针尖 4–5 eV），所以它**有**一个可以写下
依据的默认区间；I(V) 的 smoothness / symmetry 是无量纲的形状分，**没有**。
凡是没标定的，这里只报数、给 `unrated`，**不出坏裁决**——一个未标定的阈值
没资格否决人，这条在本仓是写下来的纪律。

## 只接 `.dat` 路径，不接内存数组

与 `AssessSpectrum` 同三条理由：agent 的参数适配器只认标量（一条 400 点谱塞成
逗号串会正面撞上「LLM 丢指数」那类事故）；数组不进 checkpoint；`.dat` 头里带
真实 xy，那是逐点位置核对的唯一依据。

**没有 .dat ⇒ `unrated`，不是「针坏了」。** 「谱没落盘」是流程问题。
"""

from __future__ import annotations

import logging

from mast.core.types import (
    ParameterSpec,
    SafetyLevel,
    SkillCategory,
    SkillMetadata,
    SkillResult,
)
from mast.skills.base import BaseSkill
from mast.skills.builtins.spectrum_assess import (
    _header_kind,
    _pick_directional,
    _BIAS_PATTERNS,
    _CURRENT_PATTERNS,
    _Z_PATTERNS,
)

logger = logging.getLogger(__name__)

#: 针尖裁决的闭集。**四态**，与谱质量那一层同构：
#: 「判不了」永远不折进「不行」。
TIP_VERDICTS: tuple[str, ...] = ("tip_ok", "tip_suspect", "tip_bad", "unrated")

#: 清洁金属针尖的表观势垒区间（eV）。这是**有依据的**默认：真空隧道结的
#: κ ≈ 0.5123·√φ，金属功函数 4–5 eV，实测落在这一带之外通常意味着针尖脏了、
#: 钝了，或者根本不在隧道区。宽到 3.0–8.0 是刻意的——这道闸要拦的是「明显
#: 不在隧道区」，不是给势垒做测量。
DEFAULT_BARRIER_EV_MIN = 3.0
DEFAULT_BARRIER_EV_MAX = 8.0


def _to_verdict(flags: list[str], gated: list[str]) -> str:
    """把逐条判据的结论收成一个裁决。

    没有任何一条判据真的参与过（`gated` 空）⇒ `unrated`。这一条比什么都重要：
    「所有阈值都没填」和「每条都通过了」在数值上都是「零个 flag」，而它们是
    完全相反的两句话。
    """
    if not gated:
        return "unrated"
    if not flags:
        return "tip_ok"
    return "tip_bad" if len(flags) >= 2 else "tip_suspect"


class AssessTipFromSpectrum(BaseSkill):
    """从一条存盘的 .dat 谱反推针尖状态（只读文件，不碰硬件）。"""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="AssessTipFromSpectrum",
            version="1.0.0",
            category=SkillCategory.ANALYSIS,
            safety_level=SafetyLevel.AUTO,
            description=(
                "从一条已保存的谱（.dat）判断针尖。只读，不碰硬件。这不是数据质量闸门 —— 那是 "
                "AssessSpectrum，它回答的是「这条曲线值不值得留」。这里问的是「这根针尖好不好」："
                "干净针尖的 I(z) 曲线是单一指数，其衰减给出 ~4-5 eV 的表观势垒；稳定针尖的 I(V) 光滑、"
                "无尖峰、大致反对称。verdict 是一个封闭的四态集合：tip_ok / tip_suspect / tip_bad / "
                "unrated —— 「判不了」绝不（NEVER）被折叠进 'tip_bad'。留空的阈值是 UNCALIBRATED，"
                "对那条判据只能给出 'unrated'，绝不给出坏结论；gated_criteria "
                "说明实际是哪几条做的判决。I(z) 的势垒窗有物理依据，因此带默认值；I(V) "
                "的形状分是无量纲的，出厂为 NONE。success=true 只表示这个文件读得出来；判断在 "
                "data.verdict 里。"
            ),
            parameters=[
                ParameterSpec(name="dat_path", type="str",
                              description="已保存的 .dat 谱文件路径。",
                              required=True),
                ParameterSpec(
                    name="kind", type="str",
                    description=("auto / iv / iz。'auto' 从 DATA 本身判断（哪一列在被扫）；"
                                 "文件头只用来交叉核对，因为文件头字段说的是软件最后被设成了什么。"),
                    required=False, default="auto",
                    allowed_values=["auto", "iv", "iz"]),
                ParameterSpec(
                    name="min_fit_r2", type="float",
                    description=("仅 I(z)。判定「一条干净的单指数」所需的对数-线性拟合质量下限。默认 "
                                 "UNCALIBRATED —— 留空则这条判据只报一个数。"),
                    required=False, min_value=0.0, max_value=1.0),
                ParameterSpec(
                    name="barrier_ev_min", type="float",
                    description=("仅 I(z)。表观势垒窗的下沿。默认 "
                                 f"{DEFAULT_BARRIER_EV_MIN} eV —— 这个默认值是 "
                                 "PHYSICAL 的（干净金属功函数 4-5 eV），不是标定出来的。"
                                 "设为 0 可关掉这条判据。"),
                    required=False, default=DEFAULT_BARRIER_EV_MIN,
                    min_value=0.0, max_value=50.0),
                ParameterSpec(
                    name="barrier_ev_max", type="float",
                    description=("仅 I(z)。表观势垒窗的上沿。默认 "
                                 f"{DEFAULT_BARRIER_EV_MAX} eV。"
                                 "设为 0 可关掉这条判据。"),
                    required=False, default=DEFAULT_BARRIER_EV_MAX,
                    min_value=0.0, max_value=50.0),
                ParameterSpec(
                    name="max_jumps", type="int",
                    description=("允许出现多少次电流突跳（I(z) 看 n_jumps / I(V) 看 n_spikes）。"
                                 "在斜坡中途跳变的针尖是不稳定的。默认 UNCALIBRATED。"),
                    required=False, min_value=0, max_value=10000),
                ParameterSpec(
                    name="min_smoothness", type="float",
                    description=("仅 I(V)。无量纲形状分，取值在 [0,1]。UNCALIBRATED —— "
                                 "这个量没有公开发表的数值，本台设备上也没测过，所以它是刻意留空的。"),
                    required=False, min_value=0.0, max_value=1.0),
                ParameterSpec(
                    name="min_symmetry", type="float",
                    description=("仅 I(V)。无量纲反对称性分，取值在 [0,1]。UNCALIBRATED —— "
                                 "出厂留空。NOTE 有能隙或不对称的衬底会让一根 GOOD 针尖看起来不对称，"
                                 "所以在不知道衬底的情况下拿它否决，会把好针尖判成坏的。"),
                    required=False, min_value=0.0, max_value=1.0),
            ],
            tags=("analysis", "tip", "spectroscopy", "readonly"),
        )

    def execute(self, context, params: dict) -> SkillResult:
        from pathlib import Path

        name = "AssessTipFromSpectrum"
        path = str(params["dat_path"])
        if not Path(path).exists():
            return SkillResult(skill_name=name, success=False,
                               error=f"文件不存在: {path}")

        try:
            import numpy as np

            from mast.io.nanonis_files import read_dat
            from mast.vision.spectroscopy import assess_iv, assess_iz
        except ImportError as exc:
            return SkillResult(skill_name=name, success=False,
                               error=f"缺依赖: {exc}")

        try:
            dat = read_dat(path)
        except Exception as exc:  # noqa: BLE001
            return SkillResult(skill_name=name, success=False,
                               error=f".dat 读取失败: {exc}")

        columns = dat.get("columns") or {}
        header = dat.get("header") or {}
        if not columns:
            return SkillResult(skill_name=name, success=False,
                               error=".dat 里没有 [DATA] 段")

        names = list(columns.keys())
        cur_name = _pick_directional(names, _CURRENT_PATTERNS, backward=False)
        if cur_name is None:
            # 文件读得动、但没有电流列 ⇒ **如实回答「判不了」**，不是「没做成」。
            return SkillResult(
                skill_name=name, success=True,
                data={"verdict": "unrated", "reasons": ["no_current_column"],
                      "gated_criteria": [], "dat_path": path})

        z_name = _pick_directional(names, _Z_PATTERNS, backward=False)
        v_name = _pick_directional(names, _BIAS_PATTERNS, backward=False)

        want = str(params.get("kind") or "auto").lower()
        header_kind, header_raw = _header_kind(header)
        if want == "auto":
            # 数据说了算：谁在被扫，谁就是自变量。头只作交叉检验 —— 字段标签
            # 会说谎（它是仪器软件上一次设置留下的）。
            kind = "iz" if z_name else ("iv" if v_name else "")
        else:
            kind = want
        if not kind:
            return SkillResult(
                skill_name=name, success=True,
                data={"verdict": "unrated", "reasons": ["no_sweep_column"],
                      "gated_criteria": [], "dat_path": path,
                      "header_says": header_raw})

        I = np.asarray(columns[cur_name], dtype=float)
        flags: list[str] = []
        gated: list[str] = []
        metrics: dict = {}
        notes: list[str] = []
        if header_kind and kind and header_kind != kind:
            # 不拦，只说：两个来源不一致本身是一条值得看见的信息。
            notes.append(f"header_says={header_raw!r}，而数据看起来是 {kind}")

        max_jumps = params.get("max_jumps")

        if kind == "iz":
            if z_name is None:
                return SkillResult(
                    skill_name=name, success=True,
                    data={"verdict": "unrated", "reasons": ["no_z_column"],
                          "gated_criteria": [], "dat_path": path})
            z_m = np.asarray(columns[z_name], dtype=float)
            res = assess_iz(z_m * 1e9, I)     # 纯函数吃 nm
            metrics = {"is_clean_exponential": bool(res.is_clean_exponential),
                       "fit_r2": float(res.fit_r2),
                       "barrier_ev": res.barrier_ev,
                       "decay_per_nm": res.decay_per_nm,
                       "n_jumps": int(res.n_jumps)}

            # 「是不是一条干净的单指数」由判据本体自己回答（它内含拟合质量与
            # 跳变数）。这一条**默认参与判决**：一条不是单指数的 I(z) 本身就是
            # 针尖信号（脏了、不稳、或者根本不在隧道区），不需要额外标定。
            gated.append("clean_exponential")
            if not res.is_clean_exponential:
                flags.append("not_clean_exponential")

            min_r2 = params.get("min_fit_r2")
            if min_r2 is not None:
                gated.append("fit_r2")
                if float(res.fit_r2) < float(min_r2):
                    flags.append("fit_r2_below_min")

            b_min = params.get("barrier_ev_min", DEFAULT_BARRIER_EV_MIN)
            b_max = params.get("barrier_ev_max", DEFAULT_BARRIER_EV_MAX)
            if b_min and b_max and float(b_max) > float(b_min):
                if res.barrier_ev is None:
                    # 拟合不出势垒 ⇒ 这一条判不了，**不是**「势垒不对」。
                    notes.append("barrier_ev 拟合不出来 —— 这一条没参与判决")
                elif not res.is_clean_exponential:
                    # ⚠️ 这一支是一条真机纪律：**在一条不是指数的曲线上，
                    # 「指数拟合的斜率」没有物理意义**。纯噪声照样能 polyfit 出
                    # 一个斜率，于是也照样能换算出一个「势垒」——那个数不是读数，
                    # 是拟合的副产物。拿它去判针尖，就是在自己刚造出来的数上做
                    # 判决。曲线的问题已经由上面那条 clean_exponential 说了。
                    notes.append("拟合站不住（不是单指数），势垒值没有物理意义 —— "
                                 "这一条没参与判决")
                else:
                    gated.append("barrier_ev")
                    if not (float(b_min) <= float(res.barrier_ev) <= float(b_max)):
                        flags.append("barrier_outside_window")

            if max_jumps is not None:
                gated.append("n_jumps")
                if int(res.n_jumps) > int(max_jumps):
                    flags.append("too_many_jumps")

        else:  # iv
            if v_name is None:
                return SkillResult(
                    skill_name=name, success=True,
                    data={"verdict": "unrated", "reasons": ["no_bias_column"],
                          "gated_criteria": [], "dat_path": path})
            V = np.asarray(columns[v_name], dtype=float)
            res = assess_iv(V, I)
            metrics = {"is_stable": bool(res.is_stable),
                       "smoothness": float(res.smoothness),
                       "symmetry": float(res.symmetry),
                       "n_spikes": int(res.n_spikes),
                       "gap_ev": res.gap_ev}

            if max_jumps is not None:
                gated.append("n_spikes")
                if int(res.n_spikes) > int(max_jumps):
                    flags.append("too_many_spikes")

            min_sm = params.get("min_smoothness")
            if min_sm is not None:
                gated.append("smoothness")
                if float(res.smoothness) < float(min_sm):
                    flags.append("smoothness_below_min")

            min_sym = params.get("min_symmetry")
            if min_sym is not None:
                gated.append("symmetry")
                if float(res.symmetry) < float(min_sym):
                    flags.append("symmetry_below_min")
                notes.append("symmetry 参与了判决 —— 注意带隙/不对称衬底会让"
                             "一根好针看起来不对称")

        verdict = _to_verdict(flags, gated)
        return SkillResult(
            skill_name=name, success=True,
            data={
                "verdict": verdict,
                "kind": kind,
                "reasons": flags,
                "gated_criteria": gated,
                "ungated_criteria": [c for c in
                                     (("clean_exponential", "fit_r2",
                                       "barrier_ev", "n_jumps")
                                      if kind == "iz"
                                      else ("n_spikes", "smoothness", "symmetry"))
                                     if c not in gated],
                "metrics": metrics,
                "notes": notes,
                "dat_path": path,
            })
