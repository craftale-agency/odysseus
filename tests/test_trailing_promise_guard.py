"""Trailing-promise continuation guard (2026-09-02 "wave sponsor" class).

A finished 0-tool answer that ends with a colon promising content
("here's my suggestion:") was cut by a spurious EOS — the model delivered
the intro and nothing after it. The detector must fire on promise-phrase
colons and stay silent on legitimate colon/punctuation endings.
"""

from src.agent_loop import _trailing_promise_needed


def test_wave_sponsor_case_fires():
    text = (
        "Based on reading wave's README: it's a social market for "
        "natural-language on-chain strategies on 1inch SwapVM, with The Graph "
        "as the data layer. You've already applied for The Graph and 1inch "
        "Aqua. For the third sponsor in the Continuity track, here's my suggestion:"
    )
    assert _trailing_promise_needed(text, retry_used=False) is True


def test_candidates_intro_fires():
    assert _trailing_promise_needed(
        "I reviewed the stack and considered fit with the project's "
        "architecture. After narrowing, the candidates are:",
        retry_used=False,
    ) is True


def test_such_as_intro_fires():
    assert _trailing_promise_needed(
        "The project already uses several sponsor technologies, such as:",
        retry_used=False,
    ) is True


def test_bare_list_colon_does_not_fire():
    # A label-style colon ending is not a promise phrase.
    assert _trailing_promise_needed(
        "wave — social market for on-chain strategies. Built for ETHGlobal. Tags:",
        retry_used=False,
    ) is False


def test_normal_sentence_does_not_fire():
    text = (
        "The third sponsor should be ENS — the project already uses it as "
        "the identity layer, making it the strongest continuity story."
    )
    assert _trailing_promise_needed(text, retry_used=False) is False


def test_short_text_does_not_fire():
    assert _trailing_promise_needed("here's my suggestion:", retry_used=False) is False


def test_retry_used_blocks():
    text = (
        "Long substantive answer that builds up context and then says: "
        "for example:" + " " * 60
    )
    # Even a promise tail is ignored once the turn already used its retry.
    assert _trailing_promise_needed(text, retry_used=True) is False


def test_colon_must_be_terminal():
    text = (
        "Here's my suggestion: ENS is the natural third sponsor because the "
        "project uses it as the identity layer already."
    )
    # Promise phrase present but NOT terminal — complete answer, no fire.
    assert _trailing_promise_needed(text, retry_used=False) is False
