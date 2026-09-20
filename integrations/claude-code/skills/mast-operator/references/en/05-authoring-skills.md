# Authoring skills

Work down a short ladder and stop at the first rung that does the job. If an official skill already
does what you need, use it — search first, as [Operating rules](03-operating-rules.md) says. If a
sequence of official skills would do it, build a **composite skill**: a JSON graph, not code, that
runs under exactly the gates in [Concepts](02-concepts.md). Only when composing existing skills
genuinely cannot express what you need — typically a Nanonis command nothing wraps yet — propose a
new **Python skill** for a person to review. This file covers both tracks.

## Composite skills

A composite is data: a JSON `CompositeSpec` describing a graph of existing skills, validated and
executed by one shared interpreter. Its effective safety level is the stricter of what it declares
and its strictest step, and its capability tags are the union of its steps' tags (see
[Concepts](02-concepts.md)) — so composing cannot open a door none of its steps would open alone.

### Draft, then save

`POST /composites/draft` validates a spec without saving or registering anything, and returns every
problem at once, along with hints when your draft's steps are already covered by an existing
official composite — worth reading before you build the same thing twice. Send `spec: "?"` to get
the format description directly from MAST instead of guessing at it.

`POST /composites` saves a validated spec and hot-registers it under your signature (`ext:<actor>`):
from that point on it runs like any other skill, through `POST /jobs`. Saving over your own earlier
version needs `base_version` — the version number you last read back — as an optimistic lock;
saving identical content again does not create a pointless new version. A composite made by a
person, rather than by an agent, is never overwritten this way: save yours under a different name
instead.

### The spec format

A spec has a `name`, a `safety_level`, a list of `params`, and a list of `nodes`. On this track the
allowed node types are purely structural: `step`, `if`, `loop`, `try`, `set`, `succeed`, `fail`,
`break` and `continue` — human, LLM and agent-delegation nodes are not available here. A `try` node
with a `finally` body expresses cleanup for ordinary completion or failure. A hard interruption
(`GeneratorExit`) skips that body, and process termination cannot guarantee it runs. Even when
cleanup executes, verify its hardware result; a `finally` block is not proof that the tip withdrew.

Three rules make the expressions readable: a step's result is referenced by **subscript**, using
its own `id` (`s1['bias_v']`, where `s1` is the id of a `step` node; `last` means the most recently
run step) — attribute access such as `s1.data.x` is not accepted. Every `cond`, `count` and `value`
is an **expression string**, not a literal — write `"3"`, not `3`. A workflow parameter is passed
into a step with `{"$expr": "param_name"}`; a literal value is written directly. A minimal spec with
no branching, referencing three read-only steps by their outputs, looks like this:

```json
{
  "name": "ExampleReadSequence",
  "safety_level": "auto",
  "params": [],
  "nodes": [
    {"type": "step", "id": "s1", "skill": "GetBias", "params": {}},
    {"type": "step", "id": "s2", "skill": "GetSetpoint", "params": {}},
    {"type": "step", "id": "s3", "skill": "GetCurrent", "params": {}}
  ],
  "outputs": [
    {"name": "bias_v", "expr": "s1['bias_v']"},
    {"name": "setpoint_a", "expr": "s2['setpoint_a']"},
    {"name": "current_a", "expr": "s3['current_a']"}
  ]
}
```

### What gets refused

| Situation | Refused because |
|---|---|
| A spec with no `step` node at all | It would do nothing to the instrument while still reporting success — not useful. |
| A spec with exactly one step and no branching, loop or retry | It renames the one skill it wraps rather than adding anything; call that skill directly. |
| A `human`, `llm` or `agent` node | Only the structural node types listed above are open on this track. |
| A loop whose body touches hardware, with no `max_iter` | The default cap is large enough to run all night unattended; a hardware-touching loop here must declare one of at most 100. |
| A step naming a skill switched off on this installation | Wrapping a disabled skill in a composite does not turn it back on — that is the operator's decision. |
| A name already used by a registered skill, or by a built-in template | Reusing it would silently replace that skill or be overwritten by it again the next time MAST starts. |

### Composites may reference official skills only

At startup MAST registers skills in a fixed order: built-ins, then stored composite specs, then
agent tools, then custom skills, then overlays. A spec you save is validated and registered as a
stored spec — before custom skills or agent tools exist yet — so a step naming either of those is
refused, at every startup, with only a log line to show for it. Build only from built-in atomic
skills or other official composites.

## Python skill proposals

`POST /skills/proposals` is the last resort: source code for a brand-new atomic skill, written to
this installation's custom-skill folder **for a person to review** — nothing about it is registered
or executed. Report what you proposed and why, then keep working with what already exists; do not
wait on it.

Making a proposal live is entirely the operator's decision: they read the code, add its name to the
enabled list, and restart MAST. That list keys a proposal by its **file name's stem**, not by the
class name written inside it, so name it to match or the enable step will look for a file that is
not there. Only the first `BaseSkill` subclass MAST finds in the file is ever registered — a second
class in the same file is silently ignored — so keep one skill per file.

### The deny-list

A static deny-list runs twice: once when you propose the code, and again when the operator's MAST
loads it after enabling, so a file that is later hand-edited to fail the second check is simply
skipped, with only a log line. It blocks importing modules such as `os`, `pathlib` or `socket`;
calling `eval`, `exec`, `getattr` and a handful of others; and any dunder name or attribute access
at all, to close the usual sandbox-escape tricks. That last rule is blunt enough to catch an
ordinary pattern along with the escapes it targets: `super().__init__()` is itself a forbidden
dunder attribute access, so write your skill without relying on it.

### The compliance report

Every proposal's reply carries a compliance report — the same checks a community contribution goes
through (safety level declared, the deny-list, parameter units and bounds, whether the skill's
footprint can be classified, and more) — so you can read exactly what to fix without waiting for a
person to point it out. The same checks run from a checkout's command line against a file on disk,
if you want to iterate before proposing: see [Contributing skills](06-contributing-skills.md).
