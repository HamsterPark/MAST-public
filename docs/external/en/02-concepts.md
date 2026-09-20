# Concepts

The ideas below explain what the endpoints in [the API reference](07-api-reference.md) are
actually doing. Read this once; the [Operating rules](03-operating-rules.md) that follow will make
more sense with these names in hand.

## Skills and skill metadata

A **skill** is one named, registered action: a read (get the bias), a write (set a scan frame), a
composite of several skills, or a pure analysis with no hardware involved. Every skill carries
metadata that the skill card (`GET /skills/{name}`) exposes in full:

| Field | What it tells you |
|---|---|
| `category` | `read`, `write`, `composite` or `analysis` — its shape, not its risk |
| `safety_level` | `auto`, `confirm` or `dangerous` — see [Safety levels](#safety-levels) below |
| `parameters` | Each parameter's type, unit, required flag, default, and `min`/`max` bounds or an allowed-values list |
| `preconditions` | Instrument-state conditions checked before it runs (for example, that a scan is not already running) |
| `capabilities` | Tags such as electrical-pulse or tip-shaping, read by the operating-mode gate — see below |

A dimensioned parameter's bounds are in SI base units (volts, amperes, metres, seconds); some also
accept an SI-prefixed string such as `"5n"` instead of a bare float — the skill card's `si_params`
says which, and whether the prefix is required. Do not guess a skill's name from what it does:
search for it by the action, as [Operating rules](03-operating-rules.md) explains.

## Composite skills

A **composite skill** is a graph of existing skills expressed as a JSON `CompositeSpec` — data, not
code. It runs under exactly the same gates as any other skill. Its effective safety level is the
**stricter** of what the spec itself declares and the strictest of its steps, and its capability
tags are the **union** of its steps' tags — so a declaration can only tighten what its steps
already are, never loosen it: a spec declared `dangerous` stays `dangerous` even if every step is
`auto`; a spec declared `auto` that contains one `confirm` step runs as `confirm`; and a spec
containing a tip-processing step carries that capability tag, so SAFE refuses the whole composite
exactly as it would refuse that one step alone. [Authoring skills](05-authoring-skills.md) covers
writing one.

## The execution choke point

An external job first passes submission checks, SI-string parsing and the sample-scope check,
then calls `ExecutionContext.run`. That method also runs composite substeps and applies the
checks below. Internal agent wrappers and the manual executor have distinct entry paths and
approval policies; this list describes the external job path.

1. **Abort check** — refused outright if the abort flag is set.
2. **Sample gate** — data-producing skills need an active sample; substeps inherit scope
   admission when the outer call has already passed it (see below).
3. **Operating-mode gate** — SAFE or SEMI may refuse tip-processing calls (see below).
4. **Five hard gates** — physical protections that are refused unconditionally on this path, with
   no approval to wait for: open-loop coarse Z approach toward the sample; disabling a hardware
   protection (SafeTip, Z soft limits); rewriting the global bias/current calibration; setting the
   coarse motor's drive voltage or frequency; and a raw lateral coarse move that bypasses the
   guarded relocation skill. Nothing you do changes these five; they are not addressed to you.
5. **Global safety envelope** — parameters covered by the configured safety limits are checked
   using their metadata and the administrator's bounds, alongside the skill's own validation.
6. **Parameter validation** — the skill's own `validate_params`.
7. **Preconditions** — when a state snapshot is available, check the skill's declared conditions.
   An unmet condition triggers a hardware refresh and recheck; a failed refresh preserves the
   refusal. Missing state is not proof that a physical precondition holds.
8. **Instrument arbitration** — the instrument lock (see below).

The skill's execution starts after the applicable checks. Precondition refresh may itself read
the instrument. Execution refusals appear in the job result; submission errors such as an unknown
skill or disabled capability also use HTTP error codes. Read [Troubleshooting](08-troubleshooting.md)
for both forms. These checks depend on configuration and available state; inspect degraded
readings and missing subsystems before relying on them.

## Safety levels

`safety_level` is informational on this path, not a built-in brake — read it as "how much care this
deserves", not "MAST will pause for a human here":

| Level | What happens on the external-agent path |
|---|---|
| `auto` | No separate approval prompt; execution remains subject to the other checks. Common for reads and small, repeatable writes. |
| `confirm` | No separate approval prompt on this path. Scan parameters, spectroscopy, tip conditioning and guarded lateral coarse motion are typically this level. |
| `dangerous` | The level alone does not block execution. MAST attempts to record the action and notify the operator; the other execution checks still apply. |

The five hard gates reject their restricted actions regardless of `safety_level`; mode, parameter,
scope and other checks can also refuse a call. A skill's level does not authorize an action on the
operator's behalf. Follow the agreed task and instrument procedure.

## Operating modes

The **operating mode** is the operator's global setting, reported by `GET /status` as `mode` (or
`unknown` when nothing has bound one):

| Mode | What it allows |
|---|---|
| `safe` | No tip processing at all: every electrical pulse and mechanical tip-shaping call is refused, including one buried inside a composite. Scans, spectroscopy and all other measurement continue normally. |
| `semi` | Shallow, purely mechanical tip shaping is allowed; a plunge that is too deep is refused. An electrical pulse runs and is logged — it is not silently blocked here. |
| `auto` | Everything allowed (subject to every other gate above). |

You cannot change the mode from the external-agent API, and should not try to route around it: if a
task needs more than the current mode allows, ask the operator (see
[Operating rules](03-operating-rules.md)).

## Sample gate

Skills that produce data (scans, spectroscopy) require an active sample; skills that only read
state or that are safety remedies (a stop, a retract, a withdraw) never need one. When a skill's
classification is ambiguous, this gate **fails open** — on purpose: an unlabelled recording is a
bookkeeping loss, while refusing a safety remedy in the wrong moment is not an acceptable trade.
Set a sample with `POST /scope` before running anything that produces data.

## Instrument lock

Within one MAST process, instrument-changing skills share a lock taken per skill call. It is
re-entrant for composite substeps; read/analysis skills and recognized stop/retract remedies
bypass it. Acquisition waits up to five seconds by default for brief overlaps, then returns
`refused_busy` with the current holder. There is no persistent queue behind a long scan. Wait
and check back instead of repeatedly submitting. The lock does not coordinate a second MAST
process or independent controller software.

## Abort and cancel

Two different stop mechanisms exist at two different scopes. The process-wide **abort flag** — the
emergency latch — is set by E-STOP (the operator's own button, or your own `POST /estop`), an
environment alarm, or a hardware safety signal; the operator simply stopping one of MAST's own
agent runs does not set it. While the latch is set, every skill call is refused before it starts —
reads and stop skills included. The latch is set before hardware stop/retract attempts, so it
does not prove they succeeded. Inspect the E-STOP response's `errors` and `retracted` fields and
have the operator verify instrument state; only the operator should release the latch. Status
and briefing still answer from cached state rather than running a skill. `GET /status` reports it as
`abort: {set, emergency, why}`.

A per-job **cancel** (`POST /jobs/{job_id}/cancel`) is a narrower, cooperative mechanism scoped to
one job: the skill stops at its next check, which means a skill blocked inside a single Nanonis
call only stops once that call returns. To request physical stopping without waiting for that
check, use the appropriate stop skill (exempt from the instrument lock), or E-STOP for immediate
danger. Inspect the response and confirm the instrument stopped; cancellation is not that confirmation.

## Jobs and their states

Submitting a skill through `POST /jobs` returns immediately with a `job_id`; the work happens in
its own thread and you poll for the result:

| State | Meaning |
|---|---|
| `queued` / `running` | Not finished yet — keep polling. |
| `succeeded` | The skill itself reported success. |
| `failed` | The skill failed or was refused. Read `result.error`; `refused_by` identifies selected refusal categories and can be null. |
| `refused_busy` | The instrument lock was held by someone else; nothing ran and nothing was queued. |
| `cancelled` | Cancellation was requested and the job did not report success. Verify the resulting instrument state separately. |
| `crashed` | The job's own thread failed unexpectedly; treat the outcome as unknown. |
| `lost_on_restart` | MAST restarted while the job was in flight. It is never replayed, and nobody knows how far it got on the instrument. |

A client-side timeout while waiting is not one of these states: the job keeps running on MAST
regardless of whether you are still watching it, which is the whole reason jobs exist instead of a
single long HTTP call. [Operating rules](03-operating-rules.md) covers how to act on each state.
