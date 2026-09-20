# Community skills (`contrib/`)

[中文](README.zh.md)

Community skills are the public extension point for MAST's experimental capabilities, including data
analysis, instrument operations and composed workflows. This guide covers contribution formats,
checks and local installation. Human-written and agent-assisted skills follow the same standards
and remain in the community tier until maintainer hardware verification.

## Two tiers

| Tier | Where | Requirements for contributions |
|---|---|---|
| **Community** | `contrib/skills/<Name>/` | Passes `scripts/skill_check.py` and CI. **Not verified by the maintainers on a real instrument.** Nothing here is loaded automatically: every user installs and enables each skill explicitly. |
| **Official** | `MASTv2/mast/skills/builtins/`, `config/composite_skills/` | A community skill must be verified on hardware by the maintainers before graduation into this tree. |

A community skill **graduates** after a maintainer has verified it on a real instrument: it moves out of
`contrib/` into the official tree, its **major version is bumped** (it is now held to a different
standard), and [`GRADUATED.md`](GRADUATED.md) keeps the credit.

This graduation requirement does not establish hardware validation for every existing builtin or
composite skill. Assess each feature against its documented validation status.

**Prefer CompositeSpec JSON to compose existing official skills.** It reuses their execution logic,
inherits safety levels and capability tags from its steps (a declaration can only make them stricter),
and uses the same composite validator as the workflow editor and agents.
Submit Python when the required capability cannot be expressed with existing skills.

## Directory layout

```
contrib/skills/<Name>/
├── manifest.json
├── skill.py | spec.json
├── test_<name>.py
└── README.md
```

- `manifest.json` — required, see below.
- `skill.py` (a Python skill) **or** `spec.json` (a CompositeSpec) — exactly one of the two.
- `test_<name>.py` — at least one test; the file name must be unique across `contrib/`.
- `README.md` — optional.

`<Name>` is the skill name: a plain identifier (`^[A-Za-z][A-Za-z0-9_]*$`), identical to the class name
and to `metadata().name` (Python) or to `"name"` (spec). Only `skill.py` is copied when the skill is
installed, so it cannot import helper modules from its directory or from `contrib`.

Two worked examples:

- [`EstimateScanDuration`](skills/EstimateScanDuration/) — a pure-analysis Python skill.
- [`ReadJunctionState`](skills/ReadJunctionState/) — a read-only CompositeSpec of four official steps.

## `manifest.json`

| Key | Required | Value |
|---|---|---|
| `schema` | yes | `1` |
| `name` | yes | the skill name (= directory name) |
| `kind` | yes | `"python"` or `"spec"` |
| `version` | yes | `"X.Y.Z"`; for Python it must equal `metadata().version` |
| `summary` | yes | one line, English |
| `summary_zh` | no | one line, Chinese |
| `safety_level` | yes | `"auto"`, `"confirm"` or `"dangerous"`; for a spec, the *effective* level (inherited from its steps) |
| `footprint` | yes | `"pure-analysis"`, `"hardware-read-only"` or `"hardware-write"`; must match what the checker derives from the code |
| `authors` | yes | `[{"name": "...", "github": "..."}]`, at least one entry with a `name` |
| `license` | yes | `"MIT"` |
| `verification` | yes | `"unit-tested"` (minimum) or `"contributor-hardware"` |
| `hardware_notes` | if `contributor-hardware` | which instrument / controller version, what was run, what was observed |
| `tests` | yes | the test files in this directory |
| `policy` | yes | `{"original_work": true, "no_machine_specific_defaults": true, "accepts_inbound_license": true}` |

## Checking a contribution

Run from the repository root with Python 3.13 and `MASTv2/requirements-ci.txt` installed in the
chosen environment. Here, `python` means that environment's interpreter.
See [the review and development guide](../AGENTS.md#validation-without-hardware) for Windows and Unix
setup. The commands below use Unix shell syntax.

```bash
export PYTHONPATH="$PWD/MASTv2"
python scripts/skill_check.py contrib/skills/<Name>
python scripts/skill_check.py --all-contrib
python scripts/skill_check.py --all-contrib --json
python -m pytest contrib -q
```

The four commands check one contribution, check everything under `contrib/skills/`, print the same as
machine-readable JSON, and run the contributions' own tests. Exit code `0` means every check passed, `1` means at least one `FAIL`. A `SKIPPED` line means a check
**did not run** (for example, Nanonis command names cannot be checked without the `nanonis_spm` package
installed) — skipped is not passed.

| Code | Check |
|---|---|
| S01 | `safety_level` is written out explicitly (the default would silently be CONFIRM) |
| S02 | exactly one `BaseSkill` subclass, with `metadata()` and `execute()`; `metadata()` builds `SkillMetadata(...)` directly |
| S03 | the AST deny-list — the same checker the custom-skill loader runs again at load time |
| S04 | the name is a plain identifier and install file name = class name = `metadata().name` |
| S05 | the name is not taken by a registered skill or a built-in template |
| P01 | every dimensioned parameter (one with a `unit`) has `min_value` and `max_value`; the report names the global safety-envelope row it falls under |
| P02 | a parameter whose whole range is far from 1 must be written with an SI prefix (`'50n'`), so its description must not teach exponent notation (`5e-8`) |
| P03 | every precondition is in the known vocabulary |
| P04 | a skill that pulses the bias or shapes the tip declares the matching capability tag (`bias_pulse` / `tip_shaping`) — the SAFE/SEMI operating-mode gate only sees tags |
| V01 | every `safe_call` verb is a string literal |
| V02 | every verb exists in `nanonis_spm` (or in MAST's patch layer) |
| V03 | the hardware footprint can be classified, and agrees with `category` and with the manifest |
| V04 | a write is read back (heuristic, warning only) |
| R01 | every `SkillResult(...)` keyword is a real field |
| R02 | the raw three-part Nanonis reply is never handed back as data |
| X01 | smoke run: `execute()` with a plausible fake instrument returns a `SkillResult` (runs only after S03 passed) |
| C01 | a spec passes the full design-time validation and every step is an official skill |
| M01 | `manifest.json` structure |
| M02 | the manifest agrees with the code and the directory |
| M03 | policy: original work, no machine-specific defaults, inbound license accepted; no ports of published papers |
| E01 | the checker's own environment (registry, dependencies) |

For V03, the instrument is reachable only through the execution context `execute()` receives. A skill
that hands that context to code the checker cannot see, stores it, or imports a way around it (the
vendor library, MAST's connection or runtime layers, a serial or HTTP client) cannot be classified, and
fails.

The checker reads this machine's optional-hardware toggles and administrator overrides; CI runs in the
factory configuration, where every optional hardware module is off. A spec whose steps belong to an
optional module is therefore rejected in this tier.

## Installing a community skill on your own MAST

Nothing in `contrib/` is loaded until you install and enable it. Read the code first: enabling a Python
skill runs it with the full privileges of the MAST process — the deny-list is a guard against mistakes,
not a sandbox.

### A Python skill

1. Copy `contrib/skills/<Name>/skill.py` to `config/custom_skills/<Name>.py` (under the MAST data root).
2. Add the name to `config/custom_skills/enabled.json`: `{"enabled": ["<Name>"]}`.
3. Restart MAST and check that `<Name>` appears in the skill list.

Four ways this fails **silently** — the loader only writes a log line, and the skill is simply not there:

- **Copying is not enabling.** A file that is not listed in `enabled.json` is never executed.
- **The enable key is the file name.** `enabled.json` names the file stem; the skill registers under
  `metadata().name`. Name the file exactly `<Name>.py`.
- **One skill per file.** Only the first `BaseSkill` subclass in the file is registered.
- **The deny-list runs again at load time.** A file that fails it is skipped.

### A CompositeSpec

Import it in the workflow editor, or post it to a running MAST:

```bash
curl -X POST http://127.0.0.1:7862/api/composites \
     -H "Content-Type: application/json" \
     -d "{\"spec\": $(cat contrib/skills/<Name>/spec.json)}"
```

(7862 is the launcher's default port; a development run of `python -m mast` uses its configured port.)
**A contributed spec may reference official skills only.** At
startup MAST registers skills in this order: built-ins → stored specs → agent tools → custom skills →
overlays. When a spec is registered, custom skills and agent tools do not exist yet, so a spec that
references one is refused — at every startup, with only a log line to show for it.

## Contributing

See [`CONTRIBUTING.md`](../CONTRIBUTING.md) for the rules (inbound license, DCO sign-off, what is not
accepted), and the guides [`docs/external/en/05-authoring-skills.md`](../docs/external/en/05-authoring-skills.md)
and [`docs/external/en/06-contributing-skills.md`](../docs/external/en/06-contributing-skills.md) for the
technical details.
