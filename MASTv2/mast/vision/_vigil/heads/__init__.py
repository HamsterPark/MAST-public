"""Vendored VIGIL task heads (inference-only).

v2.2 surface: Head Q (sharpness regression), Head B (tip state), Head C-L1
(4-class segmentation), ScaleEmbedding.

v2.5 surface (vendored for the 6-head ssl_sf09c1 model): soft-ordinal Head Q
(+ ``q_readout`` / ``QBinning``) and the four tip heads T / N / S / K.

Real semantics of the v2.5 tip heads (column names are historically mis-named —
see each head's module docstring):
  * Head Q  — quality, soft-ordinal continuous score ~[0,100], ↑ = good/sharp.
  * Head C  — 4-class segmentation: terrace / step / defect / contamination.
  * Head T  — tip instability P, ↑ = bad (switching ∨ perturbation).
  * Head N  — multi-apex P(apex≥2), ↑ = bad. (``n_noise`` column ≠ noise!)
  * Head S  — apex geometry: axis_ratio ∈ (0,1] + asym_logit.
  * Head K  — tip contamination P, ↑ = bad. (``k_kink`` column ≠ kink!)
"""
from __future__ import annotations

from .head_c_l1_common import head_c_supervision_mask
from .head_c_l1_dino import HeadCLevel1DINO
from .head_k import HeadK, head_k_loss, head_k_metrics
from .head_n import HeadN, head_n_loss, head_n_metrics
from .head_q_sharpness import HeadQSharpness, ScaleEmbedding
from .head_q_soft_ordinal import (
    HeadQSoftOrdinal,
    QBinning,
    head_q_soft_ordinal_loss,
    head_q_soft_ordinal_metrics,
    q_readout,
)
from .head_s import HeadS, head_s_loss, head_s_metrics
from .head_t import HeadT, head_t_loss, head_t_metrics

__all__ = [
    "ScaleEmbedding",
    "HeadQSharpness",
    "HeadCLevel1DINO",
    "head_c_supervision_mask",
    # v2.5 soft-ordinal Q
    "HeadQSoftOrdinal",
    "QBinning",
    "q_readout",
    "head_q_soft_ordinal_loss",
    "head_q_soft_ordinal_metrics",
    # v2.5 tip heads
    "HeadT",
    "head_t_loss",
    "head_t_metrics",
    "HeadN",
    "head_n_loss",
    "head_n_metrics",
    "HeadS",
    "head_s_loss",
    "head_s_metrics",
    "HeadK",
    "head_k_loss",
    "head_k_metrics",
]
