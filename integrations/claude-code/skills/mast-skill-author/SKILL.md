---
name: mast-skill-author
description: "Add a capability to MAST the right way - reuse an official skill if one fits, otherwise build a composite skill from existing skills (mast_composite_draft, mast_composite_save), and only as the last resort propose a Python skill for human review (mast_propose_skill); then contribute it to the open-source repository's contrib/ area."
when_to_use: "Use when MAST has no single skill for what you need, when the same sequence of steps repeats, when a step needs try/finally robustness (always withdraw at the end), or when the user wants to share a skill upstream. Trigger words - new skill, composite skill, combine skills, write a skill, contribute a skill, skill_check, 新技能, 组合技能, 造技能, 投稿, 贡献技能."
---

# Authoring MAST skills

Extend MAST by reusing its skill contracts, execution checks and composition tools.
Choose the first option below that expresses the required behavior. Source-only authoring and
tests need no instrument session; MCP draft/save calls require a configured MAST service.

## 1. An official skill already does it

Search by action first: `mast_find_skills` with the verb or Nanonis command ("withdraw",
"z controller", "bias spectroscopy"), then read the card with `mast_skill_card`. Official
skills expose their parameters, safety metadata and hardware footprint through this contract;
check the feature's documented verification status. A thin rename is not a new capability,
and MAST rejects it.

## 2. A composite skill (the normal case)

A composite is a JSON graph of existing skills. It runs under the same safety gates as official
skills and inherits their safety levels, so it cannot open anything its steps cannot.

1. `mast_composite_draft(spec="?")` returns the spec syntax: node types, expressions, what gets
   rejected.
2. Write the spec and call `mast_composite_draft(spec={...})` until the answer is valid. All
   problems come back at once; `hints` point at official skills that already overlap.
3. `mast_composite_save(spec={...})` saves it, keeps its version history and hot-registers it.
   Run it with `mast_run(skill="<name>", params=...)`.
4. To change a composite you saved earlier, pass `base_version` = the version you last got (an
   optimistic lock). Composites made by a person are never overwritten: save yours under a new
   name.

Worth building when: the same steps repeat within a task, a plan step has no single skill, or a
sequence needs an explicit cleanup path. A `try/finally` path can attempt cleanup;
abort state, communication failures and physical conditions can still prevent an action from completing.

## 3. A Python skill proposal (last resort)

Only when neither official skills nor a composite can express it, for example a Nanonis command
that no skill wraps yet. `mast_propose_skill(name, code, rationale)`:

- `code` defines a `BaseSkill` subclass implementing `metadata()` and
  `execute(context, params)`; declare the safety level honestly.
- `rationale` explains why existing skills and composites are not enough. It is what the human
  reviewer judges.
- The file is written for review and is **not** registered or run. Only a person can enable it.
  Report that you proposed it, then finish the task with existing skills.

## 4. Contributing upstream

Skills that proved themselves can go to the open-source repository's community tier,
`contrib/skills/<Name>/`, as `contrib/README.md` there describes:

- one directory per skill: `manifest.json`, exactly one of `spec.json` (a composite, preferred)
  or `skill.py` (Python), and at least one `test_<name>.py`;
- a contributed spec may use official skills only: steps that are themselves community or
  custom skills are refused when MAST registers it;
- `python scripts/skill_check.py contrib/skills/<Name>` must exit 0 before you open a pull
  request, and a `SKIPPED` check is not a pass;
- community skills are never loaded automatically; a maintainer moves one into the official
  set only after verifying it on a real instrument. The rules for contributions are in
  `CONTRIBUTING.md` at the repository root.

## Reference

- Authoring skills: [English](../mast-operator/references/en/05-authoring-skills.md) · [中文](../mast-operator/references/zh/05-authoring-skills.md)
- Contributing skills: [English](../mast-operator/references/en/06-contributing-skills.md) · [中文](../mast-operator/references/zh/06-contributing-skills.md)
