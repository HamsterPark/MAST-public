"""Regression: compaction trigger scales with the agent's ACTUAL model window
(2026-07-07 fix). A small-window model must compact before it overflows, not at
kimi's hardcoded 240k trigger."""
from mast.agents._shared.compaction_mw import compaction_trigger_tokens


def test_trigger_scales_with_model_window():
    kimi = compaction_trigger_tokens("kimi-k2.6")            # 240k window
    deepseek = compaction_trigger_tokens("deepseek-v4-pro")  # 120k window
    assert deepseek < kimi
    # must fire BELOW deepseek's own 120k window (the overflow bug)
    assert deepseek < 120000


def test_unknown_model_conservative_default():
    t = compaction_trigger_tokens("some-unknown-model-x")
    assert t >= 8000            # never degenerate
    assert t < 120000           # below the conservative default window
