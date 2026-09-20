# Contributing to MAST

[中文](CONTRIBUTING.zh.md)

**Help extend MAST with reusable experimental skills.** A contribution can connect a new analysis
method, instrument operation or multi-step workflow to the existing parameter checks, execution
constraints and operator interface. Human-written and agent-assisted skills follow the same review
and verification process.

This is MAST's public source edition. Code contributions enter through `contrib/skills/` and progress
from community checks to maintainer hardware verification before joining the official tree.
The rest of the source is maintained through exports from the development repository; use issues
for related bugs and improvement proposals. Report vulnerability details privately through
[`SECURITY.md`](SECURITY.md).

## Contribution scope

- A new skill under `contrib/skills/<Name>/`: prefer **CompositeSpec JSON** to reuse official skills;
  write a **Python skill** when the required execution logic cannot be expressed by composition.
- Fixes to an existing contribution under `contrib/`.

## Scope and constraints

- **Ports of published work.** Skills that re-implement a method from a paper, or code ported from someone
  else's repository. This snapshot deliberately ships without them; the checker rejects citations, DOIs and
  arXiv identifiers (code `M03`).
- **Defaults that describe one instrument.** Calibration values, piezo ranges, controller gains, bias or
  current working points measured on your machine. Parameters get instrument-independent defaults (or
  none) and explicit `min_value` / `max_value`.
- **Changes outside `contrib/`.** Everything else is regenerated from the private repository and would be
  overwritten by the next export. Open an issue instead.

## Two tiers

| Tier | Where | Requirements for contributions |
|---|---|---|
| Community | `contrib/skills/<Name>/` | Passes the checker and CI; users install and enable it explicitly. It has not been verified on hardware by the maintainers. |
| Official | `MASTv2/mast/skills/builtins/`, `config/composite_skills/` | A community skill must be verified on hardware by the maintainers before graduation into this tree. |

This table describes the contribution pipeline, not the validation status of every existing builtin or
composite skill. See [README.md](README.md) for the published validation boundaries and
[the snapshot notes](docs/OPEN_SOURCE_NOTES.md) for this snapshot's recorded checks.

Layout, the manifest format, the meaning of every check code and how to install a community skill locally
are in [`contrib/README.md`](contrib/README.md).

## Checklist for a pull request

Use a Python 3.13 environment. For contribution-only checks, install `MASTv2/requirements-ci.txt`.
Here, `python` means the interpreter in that environment; run commands from the repository root.
See [AGENTS.md](AGENTS.md#validation-without-hardware) for setup and broader validation routes.

- [ ] Only files under `contrib/skills/<Name>/` change.
- [ ] `manifest.json`, then `skill.py` **or** `spec.json`, and at least one test (`test_<name>.py`, a file
      name not used anywhere else in `contrib/`).
- [ ] `python scripts/skill_check.py contrib/skills/<Name>` exits `0`.
- [ ] `python -m pytest contrib -q` passes.
- [ ] The manifest's `policy` attestations are true.
- [ ] Every commit is signed off (see below).

These are local submission checks. CI runs `python scripts/skill_check.py --all-contrib`, followed by
contribution compliance tests, checker mutation tests and documentation guards. Its current commands
are in [`.github/workflows/ci.yml`](.github/workflows/ci.yml).

## Verification levels

| Level | Set by | Meaning |
|---|---|---|
| `unit-tested` | contributor | Passes the checker and its own tests; no hardware verification is declared. This is the minimum. |
| `contributor-hardware` | contributor | Also run on the contributor's own instrument; `hardware_notes` says which instrument and controller version, what was run and what was seen. |
| graduated | maintainers | Verified by the maintainers on their instrument and moved into the official tree. |

## Graduation

1. A maintainer runs the skill on a real instrument.
2. It moves into `MASTv2/mast/skills/builtins/` (Python) or `config/composite_skills/` (spec); the
   `contrib/` copy is removed. The code may be adapted to internal conventions on the way.
3. Its **major version is bumped** — from now on it is held to the official standard.
4. [`contrib/GRADUATED.md`](contrib/GRADUATED.md) records the skill and its authors. The credit stays.

## Developer Certificate of Origin

Every commit must carry a `Signed-off-by: Your Name <email>` line (`git commit -s`), certifying the
[Developer Certificate of Origin 1.1](https://developercertificate.org/): you wrote the contribution, or
otherwise have the right to submit it under the license below.

## License of contributions

- **Inbound = outbound:** contributions are accepted under the MIT License ([`LICENSE`](LICENSE)).
- **A contribution may also be included in closed-source or commercial distributions of MAST.** The MIT
  License permits this, and by submitting you agree to it. If you do not want that, please do not submit.

## Technical details

- [`docs/external/en/05-authoring-skills.md`](docs/external/en/05-authoring-skills.md) — writing a skill:
  parameters and units, safety levels, capability tags, footprint.
- [`docs/external/en/06-contributing-skills.md`](docs/external/en/06-contributing-skills.md) — the
  contribution workflow step by step.
- [`contrib/README.md`](contrib/README.md) — layout, manifest, check codes, local installation.
