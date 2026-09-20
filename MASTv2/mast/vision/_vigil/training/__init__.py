"""Vendored VIGIL training-package surface — inference-only subset (Phase 9 / v2.5).

MAST does NOT train these heads. The only thing vendored here is
:mod:`multihead`, trimmed to the **forward-construction** surface the M12 /
VIGIL-v2.5 inference path needs:

  * ``build_active_heads``   — construct the 6-head ``nn.ModuleDict`` (q/c/t/n/s/k)
  * ``resolve_tap_layers``   — layer-tap name → block indices (``last`` → ``None``)
  * ``HeadLossCfg``          — config dataclass carried through inference
  * ``q_head_input`` / ``_feats_for_t`` / ``_patch_tokens`` — feature routing
  * ``_Q_SOFT_FORMS``        — the soft-ordinal Q forms set

The training-time loss/metric aggregation (``compute_active_losses`` /
``compute_active_metrics`` and the legacy b/o/v/d heads, ``supervision``,
``eval.metrics``, ``v25_prereq``) is intentionally **out of closure** — it is
not needed for inference and pulls in modules MAST does not vendor.

Unlike upstream's ``__init__`` (which re-exports gradnorm/losses/scheduler/UW),
this package re-exports nothing at import time: ``import …training`` must stay
dependency-light (peft/timm are deferred). Import from ``…training.multihead``.
"""
from __future__ import annotations
