"""Apply Z-controller parameters BY NAME — the agent never types the numbers.

The agent says ``ApplyZCtrlPreset('approach')``. Code looks the values up in the
place the operator maintains them, checks them, writes them, reads them back and
compares — in Python. There is no numeric argument anywhere in this path, so
there is nothing for a lost exponent to land in.

This exists because of 2026-08-03: asked for ``p_gain=3e-12``, an agent emitted
``3``. See ``mast.core.zctrl_presets`` for the resolution rules and
``docs/v2/fixes/2026-08-03-tool-call-number-corruption.md`` for the incident.

3 skills: ApplyZCtrlPreset (CONFIRM), CreateZCtrlPreset (DANGEROUS),
ListZCtrlPresets (AUTO).
"""

from __future__ import annotations

from mast.core.si_quantity import format_si
from mast.core.types import (
    ParameterSpec,
    SafetyLevel,
    SkillCategory,
    SkillMetadata,
    SkillResult,
)
from mast.core.zctrl_presets import (
    PresetRejected,
    available_names,
    get_presets,
    resolve,
    upsert_preset,
)
from mast.skills.base import BaseSkill


class ApplyZCtrlPreset(BaseSkill):
    """Write a named set of Z-controller parameters to the instrument."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="ApplyZCtrlPreset",
            version="1.0.0",
            category=SkillCategory.WRITE,
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "按**参数组名**把 Z 控制器参数(P/I 增益 + 设定点)写进硬件。\n\n"
                "**这是设置 Z 参数的首选方式**,优先于 SetZCtrlGain / SetSetpoint:"
                "具体数值由代码从用户维护的存储里取出并写入,你只需要说用哪一组,"
                "不需要(也不应该)自己写出任何数字。\n\n"
                "常用组名:\n"
                "- `approach` —— 进针参数(来自仪器档案,用户填写)\n"
                "- `scan` —— 扫图参数(按当前扫描帧尺寸自动选档,与 ScanAt 同源)\n"
                "- 也可以直接用扫描档名(如 `atomic`)或自定义组名\n\n"
                "用 ListZCtrlPresets 查看当前可用的全部组名与数值。"
                "写入后会自动回读比对,不一致会报失败。"
            ),
            parameters=[
                ParameterSpec(
                    name="preset",
                    type="str",
                    description=(
                        "参数组名,如 'approach' / 'scan' / 档名 / 自定义组名。"
                        "不确定有哪些就先调 ListZCtrlPresets。"
                    ),
                    required=True,
                ),
            ],
            estimated_duration_s=3.0,
            composition_level=1,
            tags=["z", "gain", "setpoint", "preset", "write", "readback"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        name = params["preset"]
        try:
            resolved = resolve(name, context=context)
        except PresetRejected as exc:
            # The name is validated here rather than as a schema enum: the legal
            # set changes at runtime (the operator can add a tier or a group),
            # while a tool schema is frozen when the agent is built. An error
            # that lists what IS available beats an enum that goes stale.
            return SkillResult(
                skill_name="ApplyZCtrlPreset", success=False, error=str(exc),
            )

        calls: list = []
        applied: dict = {}

        # Sub-steps rather than direct safe_call: this is the same path ScanAt
        # uses, so the leaf skills' own bounds, SafetyGate checks and — since
        # 2026-08-03 — their write-then-read-back verification all apply. It also
        # keeps the (P, T, I) argument order known in exactly one place.
        gain_res = context.run("SetZCtrlGain", resolved.gain_params())
        calls.extend(getattr(gain_res, "nanonis_calls", []) or [])
        if not gain_res.success:
            return SkillResult(
                skill_name="ApplyZCtrlPreset",
                success=False,
                error=(
                    f"参数组 '{resolved.name}' 的增益写入失败: {gain_res.error}"
                ),
                data={"preset": resolved.name,
                      "trace": resolved.trace_lines()},
                nanonis_calls=calls,
            )
        applied["gains"] = resolved.gain_params()

        if resolved.setpoint_a is not None:
            sp_res = context.run("SetSetpoint", {"setpoint_a": resolved.setpoint_a})
            calls.extend(getattr(sp_res, "nanonis_calls", []) or [])
            if not sp_res.success:
                return SkillResult(
                    skill_name="ApplyZCtrlPreset",
                    success=False,
                    error=(
                        f"参数组 '{resolved.name}' 的增益已写入,但设定点写入失败: "
                        f"{sp_res.error}"
                    ),
                    data={"preset": resolved.name, "applied": applied,
                          "trace": resolved.trace_lines()},
                    nanonis_calls=calls,
                )
            applied["setpoint_a"] = resolved.setpoint_a

        # Every leaf write above verified itself by reading back and comparing in
        # Python. Re-reading here would only add round-trips and a second place
        # for the comparison rule to drift.
        summary = "\n".join(resolved.trace_lines())
        notes = "\n".join(resolved.notes)
        return SkillResult(
            skill_name="ApplyZCtrlPreset",
            success=True,
            data={
                "preset": resolved.name,
                "applied": applied,
                "trace": resolved.trace_lines(),
                "sources": resolved.sources,
                "notes": resolved.notes,
                "message": (
                    f"已按参数组 '{resolved.name}' 设置 Z 控制器,写入后回读已确认:\n"
                    + summary + (f"\n{notes}" if notes else "")
                ),
            },
            nanonis_calls=calls,
        )


class CreateZCtrlPreset(BaseSkill):
    """Create or replace a custom Z-parameter group."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="CreateZCtrlPreset",
            version="1.0.0",
            category=SkillCategory.WRITE,
            # DANGEROUS: this writes a value the system will keep trusting. A
            # wrong number here is not one bad frame, it is every future
            # ApplyZCtrlPreset of this group — so it goes through the human
            # approval card, where the operator sees the ARGUMENTS AS PARSED
            # (hitl_bridge mirrors them verbatim) and can edit or reject before
            # anything is stored. That card is the "建完他被要求检查" step.
            safety_level=SafetyLevel.DANGEROUS,
            description=(
                "新建(或覆盖)一个自定义 Z 参数组,之后可以用 ApplyZCtrlPreset 按名应用。"
                "\n\n"
                "**数值必须写成带 SI 前缀的字符串**,例如 p_gain='3p'、i_gain='180n'、"
                "setpoint_a='150p' —— 与 Nanonis 面板上的写法一致。"
                "**前缀不可省略**:裸数字(如 '3')会被直接拒绝。原因是量级一旦丢失,"
                "裸数字仍然是一个合法的数,错一万亿倍也没人发现;而前缀掉了就解析失败,"
                "你会立刻收到一次明确的拒绝。\n\n"
                "不要用本技能去改进针参数或扫图档位表 —— 那两套由用户在设置界面维护。"
                "本技能只管自定义组。"
            ),
            parameters=[
                ParameterSpec(
                    name="name",
                    type="str",
                    description=(
                        "参数组名。不能与保留名(approach / scan)或扫描档名重复。"
                    ),
                    required=True,
                ),
                ParameterSpec(
                    name="p_gain",
                    type="str",
                    description=(
                        "Z 比例增益,带 SI 前缀的字符串,单位米。例如 '3p' = 3p m。"
                        "典型 1p ~ 10p。**必须带前缀**。"
                    ),
                    unit="m",
                    required=True,
                ),
                ParameterSpec(
                    name="i_gain",
                    type="str",
                    description=(
                        "Z 积分增益,带 SI 前缀的字符串,单位米每秒。"
                        "例如 '180n' = 180n m/s。典型 10n ~ 1u。**必须带前缀**。"
                    ),
                    unit="m/s",
                    required=True,
                ),
                ParameterSpec(
                    name="setpoint_a",
                    type="str",
                    description=(
                        "可选。电流设定点,带 SI 前缀的字符串,单位安培。"
                        "例如 '150p' = 150p A。留空表示这组不改设定点。"
                    ),
                    unit="A",
                    required=False,
                ),
                ParameterSpec(
                    name="note",
                    type="str",
                    description="可选。这组参数的用途说明,给用户看。",
                    required=False,
                ),
                ParameterSpec(
                    name="overwrite",
                    type="bool",
                    description="同名组已存在时是否替换。默认 False(存在则拒绝)。",
                    required=False,
                    default=False,
                ),
            ],
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["z", "gain", "preset", "config"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        try:
            stored = upsert_preset(
                {
                    "name": params.get("name"),
                    "p_gain": params.get("p_gain"),
                    "i_gain": params.get("i_gain"),
                    "setpoint_a": params.get("setpoint_a"),
                    "note": params.get("note"),
                },
                overwrite=bool(params.get("overwrite", False)),
            )
        except PresetRejected as exc:
            return SkillResult(
                skill_name="CreateZCtrlPreset", success=False, error=str(exc),
            )

        # Echo back what STORAGE now holds, re-resolved — not the arguments that
        # came in. If anything were altered on the way through, this is where it
        # shows, and the operator reads a fact rather than a repetition of the
        # request.
        readback = resolve(stored["name"])
        return SkillResult(
            skill_name="CreateZCtrlPreset",
            success=True,
            data={
                "preset": stored["name"],
                "stored": stored,
                "trace": readback.trace_lines(),
                "message": (
                    f"参数组 '{stored['name']}' 已保存。存储里现在的值是:\n"
                    + "\n".join(readback.trace_lines())
                    + f"\n\n用 ApplyZCtrlPreset('{stored['name']}') 应用它。"
                ),
            },
        )


class ListZCtrlPresets(BaseSkill):
    """List every Z-parameter group that can be applied right now."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="ListZCtrlPresets",
            version="1.0.0",
            category=SkillCategory.READ,
            safety_level=SafetyLevel.AUTO,
            description=(
                "列出当前可用的全部 Z 参数组名及其数值与来源。"
                "在调用 ApplyZCtrlPreset 之前不确定有哪些组时使用。"
            ),
            parameters=[],
            estimated_duration_s=0.5,
            composition_level=0,
            tags=["z", "gain", "preset", "read"],
        )

    def execute(self, context, params: dict) -> SkillResult:
        rows: list[dict] = []
        for name in available_names():
            entry: dict = {"name": name}
            try:
                # No context: the 'scan' alias needs the current frame size and
                # would otherwise refuse. Listing is not the place to do hardware
                # I/O — say what it resolves against instead.
                res = resolve(name)
                entry.update({
                    "p_gain": format_si(res.p_gain) + "m",
                    "i_gain": format_si(res.i_gain) + "m/s",
                    "time_constant_s": format_si(res.time_constant_s) + "s",
                    "setpoint_a": (format_si(res.setpoint_a) + "A"
                                   if res.setpoint_a is not None else None),
                    "sources": res.sources,
                    "usable": True,
                })
            except PresetRejected as exc:
                entry.update({"usable": False, "why": str(exc)})
            rows.append(entry)
        custom = {p["name"] for p in get_presets()}
        for row in rows:
            row["kind"] = (
                "reserved" if row["name"] in ("approach", "scan")
                else "custom" if row["name"] in custom
                else "scan-tier"
            )
        return SkillResult(
            skill_name="ListZCtrlPresets",
            success=True,
            data={
                "presets": rows,
                "note": (
                    "'scan' 按当前扫描帧尺寸选档,所以这里不显示它的具体数值 —— "
                    "应用时才确定。"
                ),
            },
        )
