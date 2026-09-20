---
name: mast-operator
description: "Operate a MAST-controlled scanning tunneling microscope (STM, Nanonis) through the mast_* MCP tools - read the briefing, set the experiment and sample, find skills by the action you need, run them as jobs and poll, fetch raw data, keep notes, ask the operator, and hand over."
when_to_use: "Use for any request that touches the STM or MAST - scanning and imaging, approach and withdraw, bias and setpoint, tip conditioning, spectroscopy (STS, dI/dV, grids), drift, fetching frames or spectra, experiment records, notes, handover. Trigger words - STM, scanning tunneling microscope, Nanonis, MAST, scan, tip, bias, setpoint, spectroscopy, dI/dV, approach, withdraw, 扫描隧道显微镜, 针尖, 扫描, 谱, 偏压, 设定点, 进针, 退针, 实验记录, 交接."
---

# Operating MAST from Claude Code

MAST connects agent workflows to a real scanning tunneling microscope through Nanonis.
This plugin's `mast` MCP server exposes the 6.5.0 external API through `mast_*` tools.
Use this workflow when the user requests instrument operation on a configured MAST deployment.
For repository review or skill development without a device, inspect source and tests without
starting a hardware session. The new API is software-tested and awaits hardware validation.

Hardware-writing actions have physical consequences: incorrect parameters can damage the tip
or sample. Read-only and analysis tools have distinct footprints shown on their skill cards.

## Workflow

1. **Briefing first.** `mast_briefing` returns the whole state in one call: mode and abort flag,
   who holds the instrument, scope and recording, resume notes, tip, instrument profile, live
   readings, the operator's parameter preferences, safety envelope, recent actions and files,
   alarms, notes, your jobs and requests. `mast_status` is the cheap re-check.
2. **Scope.** `mast_scope` shows or sets the active experiment and sample. Set the intended
   scope before experimental work (`experiment_name`, `goal`, `sample_name`).
   Experiment-linked recording requires an active experiment; inspect each job's `recorded`
   fields to determine which records were written.
3. **Find skills by action.** Call `mast_find_skills` with the verb or Nanonis command you need
   ("withdraw", "set bias", "z controller", "scan frame", "bias spectroscopy"). Many skills are
   named after the problem they solve, so guessing names fails. Prefer hits marked `official`.
4. **Read the card.** `mast_skill_card` before a skill's first use: parameters with units and
   limits, `si_params`, preconditions, safety level, footprint, whether it takes the instrument
   lock or needs a sample, typical duration, and whether it is switched off on this machine.
5. **Run as a job, then poll.** `mast_run(skill, params, request_id, wait_s)` submits and waits
   up to `wait_s` (at most 50 s). If the answer says `running`, continue with
   `mast_job(job_id, wait_s=30)` until it ends. Never submit a running action again. Pass a
   `request_id` whenever you might retry after a timeout. For retained jobs, the same caller,
   ID and payload resolve to the original job; a changed payload conflicts. Check job state
   before retrying. Submission deduplication does not establish exactly-once hardware execution.
6. **Results and data.** The job view carries `result` (summary, error, data) and `recorded`.
   `mast_list_data` lists recent files; `mast_fetch(path, mode)` saves one to disk and returns
   the local path: `raw` keeps the original bytes, `frame` writes an `.npz` with `forward` and
   `backward` arrays (orientation already normalized) plus `meta_json`.
7. **Notes.** `mast_note_write` puts findings, calibrations and caveats into MAST's memory,
   where MAST's own agents recall them; `mast_note_search` finds earlier ones.
8. **Ask, do not assume.** `mast_ask_operator` for anything that needs a person. It does not
   block; read the answer later with `mast_operator_reply`.
9. **Handover.** When you are done, leave the instrument idle (no job running) and call
   `mast_handover(summary, next_steps)`.

## Rules

- Read the briefing at the start, and again after a restart, a `lost_on_restart` job or any
  surprise.
- `refused_busy` means someone else holds the instrument: your action did not run and was not
  queued. Do not retry at once; check `mast_status` and wait until the holder is done.
- `lost_on_restart` means MAST restarted while the job was in flight and never replays it. Its
  outcome is unknown: read the briefing before deciding to submit again.
- `mast_cancel` is cooperative. It acts between skills, between sub-steps and before write
  commands, never inside one blocking Nanonis call. To request a stop, run a stop skill such
  as `StopScan`, which bypasses the instrument ownership lock, and check instrument feedback.
  For real danger use `mast_emergency_stop`; afterwards the abort flag refuses writes until the operator
  clears it.
- Never approve human-in-the-loop requests on the operator's behalf.
- SAFE is the handover mode. The operator sets it before handing you the instrument; it refuses
  autonomous tip shaping and pulses, including external jobs and composite substeps; these tools cannot change it.
  If a task needs more, ask.
- Stay inside the operator's parameter preferences and the safety envelope from the briefing.
  A refusal from a safety gate is an answer, not an obstacle to route around.
- A failed skill is not an API error, and a degraded answer (a subsystem not wired) is not an
  empty one. Read the job state, the result and `degraded` before concluding anything.
- Use `mast_fetch(mode="frame")` for scan images: MAST applies the orientation conventions
  (backward mirror, scan-direction flip) once and correctly.
- Interrupting a tool call does not stop the job on MAST; look it up with `mast_jobs`.

## Parameters and units

Numbers are SI base units: volts, amperes, metres, seconds. Parameters listed under `si_params`
in the skill card also accept prefixed strings: `"5n"` is 5e-9, `"200p"` is 2e-10; the card also
says where the prefix is required. Check `min`, `max` and the allowed values in the card before
the first run.

## Building new skills

Use an official skill when one fits; otherwise build a composite from existing skills, and
propose Python code only as the last resort. The `mast-skill-author` skill walks through it.

## When something goes wrong

| What you see | Meaning | What to do |
|---|---|---|
| `refused_busy` | Another agent or the operator holds the instrument | Wait; check `mast_status`; resubmit after they finish |
| `failed`, `refused_by: needs_human_node` | The composite has a step for a person | Ask the operator to run it in MAST |
| `failed`, refused by a sample gate | The skill produces data and needs an active sample | `mast_scope(sample_name=...)` |
| `failed` with a parameter error | A value could not be read or is out of range | Re-read the card; use SI numbers or `si_params` strings |
| Abort flag set (status or job) | E-STOP or a stop is latched; writes are refused | Tell the operator; only they clear it |
| `recorded.v1: false` | The legacy experiment record was not written | Check scope and the other `recorded` fields before the next action |
| `lost_on_restart` / `crashed` | MAST restarted, or the job died inside MAST | `mast_briefing` first, then decide |
| Cannot connect | MAST not running, wrong port, or http vs https | Start MAST; LAN mode is https |
| HTML page instead of JSON | That MAST has no external agent API | Update MAST or fix `mast_url` |
| Login required or rejected | LAN mode uses HTTP Basic | Set `mast_user` / `mast_password` |
| Remote address refused | Only this machine or a VPN is allowed | Turn on `allow_remote`, over a VPN only |
| Too many jobs (429) | The concurrent job limit is reached | Wait for running jobs (`mast_jobs`) |
| `skill_disabled` | That hardware or capability is off on this machine | Pick another skill or ask the operator |

## Reference

Full guide (also served as MCP resources `mast://guide/en/...` and `mast://guide/zh/...`):

- Overview: [English](references/en/README.md) · [中文](references/zh/README.md)
- Quickstart: [English](references/en/01-quickstart.md) · [中文](references/zh/01-quickstart.md)
- Concepts: [English](references/en/02-concepts.md) · [中文](references/zh/02-concepts.md)
- Operating rules: [English](references/en/03-operating-rules.md) · [中文](references/zh/03-operating-rules.md)
- Records and context: [English](references/en/04-records-and-context.md) · [中文](references/zh/04-records-and-context.md)
- Authoring skills: [English](references/en/05-authoring-skills.md) · [中文](references/zh/05-authoring-skills.md)
- Contributing skills: [English](references/en/06-contributing-skills.md) · [中文](references/zh/06-contributing-skills.md)
- API reference: [English](references/en/07-api-reference.md) · [中文](references/zh/07-api-reference.md)
- Troubleshooting: [English](references/en/08-troubleshooting.md) · [中文](references/zh/08-troubleshooting.md)
