"""Regression for F3 (2026-06-08, hardened after adversarial review): MiniMax
sometimes leaks <think>/<mm:think> chain-of-thought into a visible TEXT block.
_strip_think_tags must remove ONLY complete paired blocks, must NOT re-leak the
whole-block case, and must NOT corrupt legitimate text containing a literal tag.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "MASTv2"))

from mast.llm.client import _strip_think_tags  # noqa: E402


# ── paired blocks are removed, real answer kept ──
def test_paired_think_block_removed_answer_kept():
    assert _strip_think_tags("<think>reasoning</think>Answer is 1.2 V") == "Answer is 1.2 V"


def test_mixed_open_close_dialect_removed():
    out = _strip_think_tags("<think>the user wants bias</mm:think>Bias = 1.2 V")
    assert out == "Bias = 1.2 V"
    assert "think" not in out.lower()


def test_mm_think_paired_removed():
    assert _strip_think_tags("<mm:think>cot</mm:think>Final") == "Final"


def test_whole_block_leak_returns_empty_not_raw():
    # The whole text is a leaked block → return "" (caller drops empty text).
    # MUST NOT fall back to the raw text (that re-leaks the CoT it just stripped).
    out = _strip_think_tags("<think>everything was reasoning</think>")
    assert out == ""
    assert "reasoning" not in out


# ── must NOT corrupt legitimate text (adversarial-review repros) ──
def test_literal_think_token_in_prose_untouched():
    s = "Use <think> tag in your HTML template."
    assert _strip_think_tags(s) == s  # no paired block → untouched


def test_custom_thinker_element_untouched():
    s = "Hello <thinker>abc</thinker> world"
    assert _strip_think_tags(s) == s  # \b after 'think' → <thinker> not matched


def test_orphan_close_tag_left_untouched():
    # No complete paired block → leave it (orphan tags are rare/ambiguous; better
    # to keep a stray tag than risk eating real answer text).
    s = "partial cot</mm:think>Real answer"
    assert _strip_think_tags(s) == s


def test_clean_text_is_noop():
    s = "The tunneling current decays exponentially with distance."
    assert _strip_think_tags(s) == s


def test_legit_think_prose_untouched():
    s = "I think the bias should be 1 V for this sample."
    assert _strip_think_tags(s) == s


def test_empty_and_none_safe():
    assert _strip_think_tags("") == ""
    assert _strip_think_tags(None) is None
