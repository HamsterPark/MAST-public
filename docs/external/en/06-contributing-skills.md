# Contributing skills

A skill you build for your own session can also become part of MAST for everyone else, through the
community contribution tree, `contrib/skills/`. This file walks through that workflow; for the spec
format and the proposal mechanics themselves, see [Authoring skills](05-authoring-skills.md).

## The two tiers

| Tier | Verified on a real instrument? | Where it lives |
|---|---|---|
| Community | No — it passes the automated checks and its own tests; every user installs and enables it explicitly | `contrib/skills/<Name>/` |
| Official | Yes, by a maintainer | the built-in skill tree |

A community contribution is never auto-discovered and never shipped enabled: nothing in the
community tier runs on anyone's instrument until that person deliberately turns it on, the same way
a proposed Python skill only runs after the operator reviews and enables it.

## What a contribution looks like

Each contribution is one directory, `contrib/skills/<Name>/`, holding a `manifest.json` (its
name, safety level, footprint, authors, license and verification level), exactly one of `skill.py`
or `spec.json`, and at least one test file. Prefer a spec over Python source whenever the job can
be expressed as a `CompositeSpec`: it is data, not code, no new execution path is added, and its
safety level and capability tags are inherited from the official skills it steps through rather
than asserted by hand — see [Authoring skills](05-authoring-skills.md) for the format itself. Write
Python only when no combination of official skills can do what you need.

## Checking your work

The same compliance checks a proposal's reply carries run from the command line against a checkout,
so you can iterate locally before opening a pull request:

```bash
export PYTHONPATH=$PWD/MASTv2
python scripts/skill_check.py contrib/skills/<Name>
python -m pytest contrib -q
```

Exit code `0` means every check passed; `1` means at least one failed. A check the tool could not
run at all (for example, because an optional dependency is missing) is reported separately, and it
does not count as passing — install what is missing and run it again rather than treating a skipped
check as a green light.

## What is not accepted

| Not accepted | Why |
|---|---|
| A port of a published paper's method, or code copied from someone else's repository | This tree is for original work; the checks also reject citations, DOIs and arXiv identifiers found in the source. |
| A default that describes one particular instrument (a calibration, a piezo range, a controller gain, a measured working point) | Defaults must be instrument-independent, or left unset with explicit bounds instead. |
| Changes outside `contrib/skills/<Name>/` | Everything else is maintained elsewhere and would not survive an update — raise those as an issue instead. |

## Licensing and sign-off

Contributions are accepted under the MIT license, the same license as the rest of the project, and
each commit needs a sign-off (`git commit -s`) certifying the
[Developer Certificate of Origin](https://developercertificate.org/): that you wrote the
contribution, or otherwise have the right to submit it under that license. Your contribution may
also be included in closed-source or commercial distributions built from this project — the MIT
license permits that, and submitting means you agree to it. If that is not acceptable to you,
please do not submit.

## Graduation

A community skill graduates once a maintainer has run it on a real instrument and verified it
there. It then moves out of `contrib/` into the official tree, its major version is bumped — from
that point it is held to the official standard, not the community one — and it is recorded in
`contrib/GRADUATED.md` so the credit for it stays with whoever wrote it.
