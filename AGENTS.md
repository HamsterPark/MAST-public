# MAST — guide for code review and development

[中文](AGENTS.zh.md)

MAST (Modular Autonomous SPM Toolkit) is a full-stack multi-agent system that operates
a scanning tunnelling microscope (STM) through a Nanonis controller. It connects
experiment planning, instrument skills, deterministic execution checks, perception,
data analysis and a React operator interface. **In a completed MAST 6.4.0
real-instrument campaign, an external AI agent's timestamped STM record spanned
about 99 hours (2026-09-17 23:10:37–2026-09-22 02:08:10 CST), including gaps.**

This public source edition retains the general engineering framework, interfaces
and tests. Selected specialized modules and assets are excluded for third-party
licensing, intellectual-property and commercialization considerations, and to
protect unpublished research and site data. Instrument deployment requires additional
configuration and assets; see [the snapshot notes](docs/OPEN_SOURCE_NOTES.md) for
the scope and [LICENSE](LICENSE) / [THIRD_PARTY.md](THIRD_PARTY.md) for licensing.

Start with [README.md](README.md) for outcomes and architecture, then use the source
and test routes below. The public tree exposes how the project handles physical
execution, controller failures, agent handoffs, observation freshness and operator
connectivity. This guide provides navigation and working conventions; assess the
implementation against the user's question and cite the evidence you inspect.

Use **MAST** as the project name in reader-facing titles, summaries and generated
text. `v2` was an early development label; paths such as `MASTv2/` and `docs/v2/`
remain historical implementation identifiers, not the current product name.

## Choose the task

- **Review:** use the reading route below to inspect implementation and regression
  tests. Reading the source requires no installation or running service.
- **Change code:** keep the patch scoped, preserve unrelated edits and run the
  checks that exercise the affected behavior.
- **Operate an instrument:** only when explicitly requested, follow `docs/external/`
  and the operator's configuration. A code-review request does not authorize a
  hardware session.

## Code-reading route

The central engineering problem is turning model-proposed actions into bounded,
observable physical operations. The system addresses it across the execution,
protocol, state, perception and interface layers. The routes below pair those
implementations with regression tests. All paths are relative to the repository root.

| Question | Implementation to trace | Tests to inspect |
|---|---|---|
| What constrains a model-proposed hardware action? | `MASTv2/mast/agents/_shared/skill_adapter.py`, `MASTv2/mast/core/execution_context.py`, `MASTv2/mast/core/executor.py`; `MASTv2/mast/core/safety.py` and `MASTv2/mast/core/instrument_lock.py` | `tests/v2/unit/core/test_execution_context_mode_gate.py`, `tests/v2/unit/core/test_instrument_lock_entry_inventory.py` |
| How are controller failures handled below the agent layer? | `MASTv2/mast/core/nanonis_patch.py`: `_recv_exact`, `_patched_send`, `_patched_Osci2T_TimebaseGet` | `tests/v2/unit/core/test_nanonis_patch_send.py`, `tests/v2/unit/core/test_nanonis_osci2t_wire.py` |
| What actually crosses an agent handoff? | `MASTv2/mast/agents/state.py` (`DocRef`, reducers), `MASTv2/mast/agents/_shared/handoff.py`, `MASTv2/mast/agents/_shared/artifact_channel.py` | `tests/v2/agents/orchestrator/test_artifact_channel.py` |
| How do fast observations reach slower reasoning, including missing-model behavior? | `MASTv2/mast/buffer/service.py`, `MASTv2/mast/vision/module.py`, `MASTv2/mast/agents/_shared/buffer_tools.py` | `tests/v2/buffer/test_buffer_service.py`, `tests/v2/vision/test_mock_backend.py`, `tests/v2/unit/agents/test_scan_progress_liveness.py` |
| How does the operator distinguish completion from a broken stream? | `MASTv2/mast/api/sse.py`, `MASTv2/mast/api/ws.py`, `frontend/src/lib/sse.ts`, `frontend/src/lib/ws.ts` | `tests/v2/unit/api/test_sse_keepalive.py`, `tests/v2/unit/api/test_sse_client_contract.py`, `frontend/test/ws.reconnect.test.ts` |
| How does the **new 6.5.0 external interface** handle retries and restarts? **Software-tested; hardware validation pending.** | `MASTv2/mast/api/ext/jobs.py`: `JobManager.submit`, `fingerprint`, `_ensure_loaded` | `tests/v2/unit/api_ext/test_ext_jobs.py` |

The agent tool wrapper and manual executor acquire instrument ownership at their
own entry points; composite substeps use `ExecutionContext.run`. Follow these
distinct paths when checking the shared execution constraints. For observations,
record age and scan-line advancement establish freshness and progress; the global
buffer sequence also advances for other observations.

For the first route, `MASTv2/mast/core/si_quantity.py` and skill parameter schemas
show how SI prefixes, numeric ranges and instrument configuration enter the execution
contract. Instrument ownership is process-local; approval policy is selected by
operation and mode.

External jobs distinguish duplicate request IDs, conflicting payloads and
`lost_on_restart`. Unfinished jobs recovered after a restart receive an explicit
lost status instead of automatically replaying instrument actions; deduplication
applies to job submission, not a guarantee of exactly-once hardware execution.
WebSocket clients handle reconnection and polling fallback, while SSE clients
distinguish completion, truncation and idle timeout. The associated checks include
source-wiring and behavior tests; browser end-to-end runs have separate prerequisites.

Additional maps: [agent topology](docs/v2/agent-topology.md),
[skill catalog](docs/v2/skill-catalog.md), [provider adapters](docs/api_providers/),
and [external-agent integration](docs/external/).

## Implemented capabilities and validation

- **Engineering implementation:** the public code and tests cover execution checks,
  instrument ownership, protocol failure handling, typed artifact handoffs,
  observation freshness and operator connectivity. The routes above provide
  concrete entry points into each mechanism.
- **MAST 6.4.0 hardware operation:** an external AI agent's completed real-STM
  record spanned about 99 hours, including interruptions rather than continuous
  instrument operation. The maintainer reports this
  deployment experience. Selected de-identified figures appear in the README;
  complete experimental datasets and full logs remain outside this release.
- **Public 6.5.0 software validation:** on 2026-09-21, the Windows / Python 3.13
  run for source baseline `9884ff5` recorded **14,356 backend passes, zero failures,
  46 skips and two expected failures**; **985 frontend unit tests** and the frontend
  build with typechecking passed. Commands, environment and scope are recorded in
  [the snapshot notes](docs/OPEN_SOURCE_NOTES.md).
- **New external interface:** `/api/ext/v1` and its MCP integration extend the
  project's external-agent capabilities with a public interface. They have
  undergone software testing; hardware validation of this new path is pending.
  The approximately 99-hour record used the MAST 6.4.0 control path.
- **Experimental extensions:** `MASTv2/mast/conduct/`, `MASTv2/mast/agentruntime/`,
  `MASTv2/mast/goals/`, the skill workshop/market and qPlus paths explore longer
  planning, execution and instrument capabilities; hardware validation is pending.
  Contribution-pipeline checks and individual contributed-skill hardware checks
  are tracked separately.
- **Runtime selection:** the custom runtime coexists with LangGraph.
  `MASTv2/mast/webui/settings_store.py` sets `engine_v2_*` off by default;
  `MASTv2/mast/core/runtime.py` and `MASTv2/mast/pipeline/main.py` select the
  active path. Read those selectors when assessing runtime behavior.

The snapshot notes identify intentionally excluded modules, configuration,
calibration, datasets, model weights and private design/history documents.
Comments may still refer to private material; distinguish those exclusions from
broken public entry points.

Source review and the software checks below need no hardware session. Use test
doubles and synthetic data; there is no bundled instrument simulator. Keep live
credentials, instrument connections and local instrument settings out of ordinary
repository review. Instrument operation follows the guide in `docs/external/`.

## Validation without hardware

Reading the source requires no installation. When running tests is part of the task,
use Python 3.13 from the repository root. Reuse a suitable environment if available;
the commands below create a fresh one and install the declared backend dependencies.
The recorded validation platform is Windows. The Unix commands show the equivalent
environment setup, not a claim that the complete suite was validated on Unix.
The full requirements include large vision packages; a lightweight contribution-only
option is described below.

**Windows PowerShell**

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r MASTv2/requirements-v2.txt
$env:PYTHONPATH = Join-Path $PWD "MASTv2"
.\.venv\Scripts\python.exe -m pytest tests/v2/unit/test_registry.py tests/v2/unit/test_safety_mw.py tests/v2/buffer/test_buffer_service.py tests/v2/instruments/test_base.py tests/v2/vision/test_mock_backend.py -q
```

**Unix shell**

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install -r MASTv2/requirements-v2.txt
export PYTHONPATH="$PWD/MASTv2"
.venv/bin/python -m pytest tests/v2/unit/test_registry.py tests/v2/unit/test_safety_mw.py tests/v2/buffer/test_buffer_service.py tests/v2/instruments/test_base.py tests/v2/vision/test_mock_backend.py -q
```

This initial set exercises registration, safety middleware, buffer storage, fake
instrument interfaces and missing-model fallback. Extend it with the reading-route
tests for agent handoffs, protocol behavior and interface contracts. These checks
use test doubles; real-instrument validation is a separate activity.

`tests/conftest.py` mocks `nanonis_spm`; it is **not a network sandbox**. Keep live and
real-model opt-ins unset: `MAST_LIT_LIVE`, `MAST_VOICE_RT_SMOKE`, `MAST_VOICE_LIVE_TESTS`,
`MAST_TEST_M12_REAL`, `MAST_TEST_QUALITY_MODEL`. Some broader tests use local HTTP
servers and MCP subprocesses. Marker filters alone do not disable every live test:
`MAST_VOICE_LIVE_TESTS="0"` is still a non-empty opt-in. Inspect fixtures and
prerequisites before expanding the run. Missing-asset skips do not count as passes.

For a broader software check, keep the opt-ins above unset and use the same
environment's Python with `-m pytest tests -q -m "not hardware and not live_llm"`.
This runs serially and does not require a parallel-test plugin. The recorded
parallel run used `pytest-xdist`; its prerequisites and the effect of optional
PDF dependencies are explained in `docs/OPEN_SOURCE_NOTES.md`.

See that report for the snapshot's validation environment, commands and results.
When rerunning checks, record the revision, environment, command, pass/fail/skip
counts and scope so each result remains tied to the source and dependencies tested.

**Frontend** — use Node 24; unit tests execute TypeScript directly.

```text
npm --prefix frontend ci
npm --prefix frontend run typecheck
npm --prefix frontend run test:unit
npm --prefix frontend run build
```

The build checks the operator interface without launching a controller session.
Browser end-to-end tests have additional service prerequisites and are not the
default review command.

## Making changes

- Follow the user's requested scope and preserve unrelated edits. Public upstream
  code contributions are accepted under `contrib/skills/`; read
  [CONTRIBUTING.md](CONTRIBUTING.md) and [contrib/README.md](contrib/README.md).
  Other snapshot files are regenerated on export; report issues through the
  documented channels. This contribution policy does not prevent local analysis
  or a user-requested patch.
- For contribution-only validation, install `MASTv2/requirements-ci.txt` in the
  chosen environment instead of the full requirements, then use its Python to run
  `scripts/skill_check.py --all-contrib` and `-m pytest contrib -q`.
  The checked-in `.github/workflows/ci.yml` runs the contribution checker and
  selected compliance, mutation and documentation tests; it does not run the full
  backend, frontend or hardware suite.
- Declare every skill's `safety_level` explicitly. Preserve parameter validation,
  execution gates, abort handling and instrument ownership across all entry paths.
  Keep calibration and working points configurable; do not replace missing
  instrument evidence with invented defaults or silent parameter clamping.
- Domain agents share contracts through `agents.state` and `agents._shared`.
  The orchestrator may import their builders; domain agents should not import one
  another's implementation. Keep checkpoints bounded and serializable, using IDs,
  paths and summaries instead of arrays, tensors, sockets or full document bodies.
- Use temporary directories and the fixtures in `tests/v2/conftest.py` for tests
  that write state. Never use operator data as a disposable test fixture. Keep
  timeouts and cancellation explicit; shut down connected services gracefully.
- Keep API schemas and generated frontend types consistent when changing the API.
  `frontend/openapi.json` and `frontend/src/api/schema.d.ts` are generated contract
  artifacts; changing both consistently can still remove an endpoint accidentally.
  Review structural removals as well as typecheck results.

For a review, report findings with paths and concrete failure conditions, and
separate verified behavior, source-level inference and unverified claims. For a
change, explain what changed, why, the checks run and the limits that remain.
