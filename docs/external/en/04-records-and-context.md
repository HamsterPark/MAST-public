# Records and context

MAST keeps a permanent experiment record, independent of whatever your own harness remembers
between sessions. This file explains what lands in it when you act, what does not, and where to
find the context MAST already holds before you start.

## What gets recorded, and where

When experiment scope and recording services are available, completed or refused skill calls
are written under your signature so later readers can distinguish them from other callers.
Check the returned receipt for which writes succeeded:

| Record | Field | Value |
|---|---|---|
| v1 `actions` table | `context` | `ext:<actor>` (or `ext:<actor>/<session>` when you sent a session name) |
| Action record | `agent_id` | `ext:<actor>` |
| Action record | `thread_id` | `ext:<actor>` (or `ext:<actor>/<session>`) |
| Data files the action produced | — | The recording path registers available artifacts and attempts to copy them into the current experiment's folder. Check the recorded paths and any errors. |

Each job's own view carries a `recorded` field with the same facts:

```json
{"v1": true, "v2_action_id": "...", "experiment_id": "...", "sample_id": "..."}
```

## Scope and recording status

An action without an active experiment cannot be filed in that experiment's action record.
The job journal is separate, and notes can still be stored in the global namespace.
`recorded.v1: false` means the v1 write did not succeed: missing scope, unavailable storage or a
write error are possible causes. Inspect `experiment_id`, `sample_id`, `v2_action_id` and any
`error`; an empty receipt is not confirmation of recording. The briefing's `recording` section
describes the current scope and wiring, while each job's receipt reports its actual outcome.
Set the intended sample and experiment before work that needs an experiment record.

## The briefing's sections

`GET /briefing` is sectioned; ask for all of it, or name the sections you want:

| Section | Contents |
|---|---|
| `status` | Mode, abort flag, instrument lock, connection, live readings, job counts, as text. |
| `scope` | The current experiment and sample, and what happens if there is neither. |
| `resume` | How far this experiment has come — the same resume block MAST's own agents read. |
| `tip` | The registered facts about the current tip. |
| `instrument` | The instrument profile and any learned calibrations. |
| `live` | Live readings, from the cached snapshot. |
| `prefs` | The operator's default-parameter preferences. |
| `safety` | The safety envelope in force, and any latched stops with how to release them. |
| `recent_actions` | Recent actions in this experiment, with who did each one. |
| `recent_files` | Recent data files. |
| `alarms` | The environment monitor's overall status and its most recent alarms. |
| `notes` | Recent notes in this experiment, from any author. |
| `recording` | Your signature and whether experiment recording is configured for the current scope. |
| `jobs` | Your own recent jobs. |
| `operator_requests` | Your requests to the operator, and any answers. |

See [the API reference](07-api-reference.md) for the exact shape of each section's answer.

## Notes and MAST's memory

A note you write lands in the same memory store MAST's own internal agents read from, and they
recall it automatically in later turns — you do not need to tell anyone it exists. The reverse is
also true: you can search notes MAST's own agents wrote, which is often the fastest way to learn
what an earlier session already found out.

## Asking the operator

A question or request to the operator is answered **asynchronously**. Do not wait on it: keep
working on whatever does not depend on the answer, and check back later (or read the
`operator_requests` briefing section) rather than blocking your whole session on a reply that may
take a while to arrive.

## Handover

A handover report is written into MAST's document library, where the operator can find it on the
reports page — it is not a private log only you can see. Write one whenever you finish a session,
even a short one: the jobs you ran, the actions recorded under your name, and your own notes are
gathered into it automatically, alongside the summary and next steps you provide.

## After a restart

A job that was still running when MAST's process restarted is marked `lost_on_restart` in the
record and is never replayed — nobody, including MAST itself, knows how far it actually got on the
instrument. Read the briefing again before deciding what to do next; do not resubmit blind.
