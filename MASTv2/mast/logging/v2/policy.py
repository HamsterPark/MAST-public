"""Approval policy registration for the v2 logging system.

The trigger ``trg_action_requires_approval`` (declared in ``schema.py``)
rejects, at the DB layer, any ``actions`` insert whose ``action_type``
matches an active ``requires_approval`` row in the ``policies`` table
unless an approval row already exists. This module manages that table.

The trigger is a *generic, operator-extensible* gate: it fires on whatever
``action_type`` strings are registered. It is NOT the hardware-safety
boundary. Under the 2026-06-11 safety re-scoping the only action that can
physically damage the instrument — open-loop coarse Z stepping toward the
sample — is gated by ``mast.core.safety.is_coarse_sample_approach`` at the
executor/agent layer (executor forces human approval; the agent
``SafetyGateMiddleware`` blocks it fail-closed).

CORRECTION (2026-07-28): this file used to add "No builtin skill is
``SafetyLevel.DANGEROUS`` any more, so ``_derive_hitl_map()`` returns ``{}``".
That is not true — auto-discovery over builtins+composite yields NINE DANGEROUS
skills (LockNanonisUI, QuitNanonis, LoadNanonisScript, LoadMultiPassConfig,
MoveProbeXY, SetLaserOnOff, SetPiControllerOnOff, StartRfGenerator,
RunRfFrequencySweep) and the agent-layer HITL middleware IS mounted in
production. The same false claim had also been copied into
``agents/instrument_control/graph.py`` and the dispatch walkthrough.

What remains true is this module's own scope: there is **no default DANGEROUS
seed** for the DB policy table. That is a deliberate choice about *this* gate,
not a statement about the agent-layer one — the v1 seed strings never matched a
real ``action_type`` (see the historical note below), so seeding them produced
dead rows rather than protection. ``DEFAULT_DANGEROUS_ACTION_TYPES`` is
therefore empty and ``seed_default_policies`` is a no-op by default; an operator
who wants DB-level approval registers the action types they actually want.

Historical note (why this is empty rather than a v1 list): the previous seed
enumerated v1 SafetyLevel.DANGEROUS strings (``tip_pulse``, ``condition_tip``,
``auto_approach``, ``atom_manipulation``, …). Those NEVER matched a real
``action_type``: at runtime the ``action_type`` column is populated from the
registered skill name (PascalCase — ``BiasPulse``, ``MotorMove``, … — see
``logging/v2/migrate.py``, ``shadow.py`` and ``gui/tip_shape_records.py``),
so the trigger never fired on any seeded row. That made the seed
"claims-to-gate-but-no-ops" dead config; it has been removed rather than
re-spelled, because the new model intentionally has no builtin DANGEROUS
skills to gate.

Operators/admins may still register a bespoke policy at runtime via
``register_policies`` (e.g. to gate a specific custom ``action_type``); pass
the **exact** ``action_type`` string that will be written to the ``actions``
table (i.e. the skill name), not a v1 alias.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable

from mast.logging.v2.storage import ExperimentStoreV2
from mast.logging.v2.ulid import ulid_now

# No builtin skill auto-requires DB-level approval under the 2026-06-11 safety
# model (hardware safety is enforced by Nanonis fail-safes + the executor/agent
# is_coarse_sample_approach gate, not by this table). Kept as an explicit empty
# tuple so seed_default_policies has a single, documented source of truth and
# operators can see the gate is intentionally unseeded rather than forgotten.
DEFAULT_DANGEROUS_ACTION_TYPES: tuple[str, ...] = ()

POLICY_VERSION = "v1.0.0"
POLICY_REASON = (
    "Action involves tip safety, large biases, or persistent state changes "
    "that cannot be safely auto-approved. Requires human or signed policy approval."
)


def seed_default_policies(store: ExperimentStoreV2) -> int:
    """Insert the default DANGEROUS policy rows if not already present.

    Under the current safety model ``DEFAULT_DANGEROUS_ACTION_TYPES`` is empty,
    so this seeds nothing and returns 0 — it is retained as the call-site for
    GUI/server startup (``gui/redesign_server.py``) and so the gate has a named
    seeding hook should a future build reintroduce a default-gated action type.
    Custom per-deployment policies should be added via ``register_policies``.

    Returns the number of rows inserted (0 by default).
    """
    return register_policies(store, DEFAULT_DANGEROUS_ACTION_TYPES, requires_approval=True)


def register_policies(
    store: ExperimentStoreV2,
    action_types: Iterable[str],
    *,
    requires_approval: bool = True,
    required_kind: str = "any_human",
    version: str = POLICY_VERSION,
    reason: str = POLICY_REASON,
) -> int:
    inserted = 0
    with store.connect() as conn:
        for at in action_types:
            row = conn.execute(
                "SELECT id FROM policies WHERE action_type = ? AND version = ?",
                (at, version),
            ).fetchone()
            if row is not None:
                continue
            conn.execute(
                "INSERT INTO policies "
                "(id, action_type, requires_approval, required_kind, reason, version, "
                "active, created_at) VALUES (?, ?, ?, ?, ?, ?, 1, ?)",
                (
                    ulid_now(),
                    at,
                    1 if requires_approval else 0,
                    required_kind,
                    reason,
                    version,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            inserted += 1
    return inserted


def list_active_policies(store: ExperimentStoreV2) -> list[dict]:
    with store.connect() as conn:
        return [
            dict(r)
            for r in conn.execute(
                "SELECT * FROM policies WHERE active = 1 ORDER BY action_type, version DESC"
            ).fetchall()
        ]


def is_dangerous(store: ExperimentStoreV2, action_type: str) -> bool:
    with store.connect() as conn:
        row = conn.execute(
            "SELECT 1 FROM policies WHERE active = 1 AND requires_approval = 1 "
            "AND action_type = ? LIMIT 1",
            (action_type,),
        ).fetchone()
        return row is not None
