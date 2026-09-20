# Operating rules

The twelve rules below are also returned in the JSON `rules` array of `GET /guide`, for an agent that never
reads this file at all; their `id` is worth keeping in mind, since a refusal from the API sometimes
names one directly. The sections after them are operational rules that do not fit in that short
list, each with the reason behind it — see [Concepts](02-concepts.md) for the mechanisms they refer
to.

## The twelve rules

| ID | Rule | Why |
|---|---|---|
| `briefing-first` | Read the briefing first | `GET /briefing` carries what MAST's own agents see every turn: tip, instrument profile, live readings, operator preferences, the resume block, recent actions by anyone, and your open requests. |
| `scope-before-data` | Set the scope before producing data | Choose or create the experiment and sample first. Data-producing skills need sample scope; experiment action recording needs an active experiment and working storage. Check each job's `recorded` receipt. |
| `find-by-action` | Find skills by what they do | Search using the action you need, not a guessed name; skills are named after the problem they solve. Read the skill card before running: units, bounds, preconditions, measured duration on this instrument. |
| `jobs-not-timeouts` | Run skills as jobs | Submit and poll. A client-side timeout is not a failure: the skill keeps running and keeps the instrument. Retry a lost submission with the same `request_id`, never a new one. |
| `busy-means-wait` | Busy means someone else is driving | `refused_busy` means another driver holds the instrument. Retrying immediately will not make it finish sooner: wait for it, or ask the operator. |
| `how-to-stop` | How to stop | Cancel takes effect at the job's next cooperative check. Request physical stopping with a stop skill (exempt from the instrument lock), or use E-STOP for danger; check the response and instrument state. Clearing an emergency latch is the operator's call. |
| `three-states` | ok, success and degraded are different facts | A failed skill is not an endpoint error, and a degraded answer (a subsystem not wired) is not an empty answer. Read all three before concluding anything. |
| `operating-mode` | The operating mode is the operator's | SAFE refuses tip processing on autonomous execution paths, including external jobs and composite substeps. Do not route around it or change it; ask the operator if you need tip processing. |
| `no-proxy-approval` | Never decide for the operator | Do not approve or resolve human-in-the-loop prompts on the operator's behalf. Ask and wait for the answer. |
| `after-restart` | After a restart, look before you act | Jobs running when the service restarted are marked `lost_on_restart` and are never replayed. Read the briefing first to learn what state the instrument is in. |
| `raw-data` | Use the server's frames | Fetch scan images through the data endpoint: the server applies the orientation conventions (backward-direction mirror, scan-direction flip) once and correctly. |
| `leave-a-trail` | Leave a trail | Write conclusions as notes; MAST's own agents recall them. End with a handover, and leave the instrument idle: no running jobs, scan stopped. |

## Stopping, precisely

Stop any running scan, and wait until it has actually stopped, before starting a coarse motion —
the two must never overlap. Request a physical stop with a skill such as `StopScan`,
which the instrument lock never blocks, or call the emergency stop for a full E-STOP; then cancel
the job itself so its own record reflects that you asked it to stop. The stop skill or the
emergency call attempts the hardware stop; inspect its result and confirm the instrument state.
A cancel by itself only takes effect at the
job's next cooperative check, and does nothing to a call already blocked inside a single Nanonis
command.

A stop or retract skill is also exempt from the concurrent-job limit, so submitting one is never
refused with `429` just because other jobs are running. But once the emergency latch itself is
set, every job — a stop skill included — is refused before it starts. The latch is set before the
hardware stop/retract attempts, so check the E-STOP reply's `errors` and `retracted` rather than
assuming success. If stopping is unconfirmed, contact the operator immediately for the
instrument's local stop procedure. The operator verifies recovery before releasing the latch.

## Long-running tasks in an agent harness

When your own harness spawns a task that will drive the instrument for a while, start the
interpreter or executable directly rather than through a shell wrapper script: stopping the wrapper
does not stop the grandchild process that is actually holding the instrument, and a wrapper sitting
in a wait loop can go on to its next command regardless of what you intended to stop. Do not launch
it with `nohup` either, for the same underlying reason — you need the ability to actually stop what
you started, not just to stop watching it.

## After an interruption

If a long task is interrupted — your harness is killed, the connection drops, the process driving
it is stopped from outside — do not assume the instrument is where you left it. An error handler
may have withdrawn the tip, or the last command may have left it mid-motion. A client disconnect
does not cancel a server job: look up its original `job_id` first. Jobs left unfinished by a
service restart are marked `lost_on_restart` and are not replayed. Read the briefing, including
each section's `ok`, the `degraded` list and `live.stale`; its readings are cached. If current
instrument state remains uncertain, have the operator verify it before continuing.

## After a bias change

Wait for piezo creep to settle before imaging or running spectroscopy after you change the bias.
The settle time is specific to this instrument: ask the operator, or check the operator
preferences section of the briefing. Do not write a number of your own into a skill call — a
plausible-looking constant that is wrong for this instrument is worse than asking.

## Human-in-the-loop points

A composite skill can contain a step that only a person may resolve. On the external-agent path
nobody is listening for that pause, so the call fails outright — `refused_by: needs_human_node` —
rather than actually waiting: a pause with nobody responsible for noticing it is not a wait, it is
a stall. If the underlying task genuinely needs a person, run it from MAST's own interface instead,
or ask the operator and continue with whatever else you can do while you wait for an answer. Either
way, never approve, dismiss or otherwise resolve a human-in-the-loop prompt for the operator —
including one that belongs to MAST's own internal agents. That decision belongs to a person, not to
you.

## SAFE mode is not a fence against you

SAFE blocks autonomous tip processing, including external jobs and composite substeps. It is the
operator's setting and cannot be changed through this API. Authentication covers the whole
service, so the same credentials may also reach other application endpoints; SAFE is not a
separate access-control boundary. Follow the operator's mode and task scope rather than using
another endpoint to change them.
