"""Mid-word cliff continuation guard (2026-08-28 "Flavi" EOS incident).

qwen3.5:9b sampled a spurious EOS at token 126 mid-word ("Flavi") after 565
chars; Ollama, the proxy and the app all reported success (natural stop,
truncated=0). One incident in 129 stored long answers — rare, but invisible
to every transport-level fix. The guard is the generic healer: when a
naturally-finished 0-tool round of ordinary answer length ends MID-WORD,
feed the partial back as an assistant prefix and run ONE continuation round
("Continue EXACTLY where you stopped — do not repeat, do not restart").

Gate is strictly the mid-word tail (last char /[A-Za-z0-9]$/): punctuation-
tail gating alone would false-flag ~18/129 legit stored answers (quotes,
parens and colons are legitimate endings). Cap 1 per turn, independent of
the trailing_intent_retry cap.
"""

import asyncio
import json

import src.agent_loop as al


# ── pure function: trigger conditions ────────────────────────────────────

def _padded(prefix, tail):
    body = " ".join(["filler word"] * ((200 - len(prefix)) // 12 + 1))
    text = f"{prefix} {body} {tail}".strip()
    assert len(text) >= 200, len(text)
    return text


_INCIDENT_TAIL = "and then we should talk to Flavi"


def test_triggers_on_incident_shape_mid_word():
    text = _padded("Here is the plan summary.", _INCIDENT_TAIL)
    assert al._cliff_continue_needed(
        round_text=text,
        tool_calls_this_turn=0,
        retry_used=False,
    )


def test_blocked_on_sentence_punctuation_tail():
    for tail in ("talk to Flavio.", "is ready!", "sure?", "here it is:"):
        text = _padded("Summary of the incident.", tail)
        assert not al._cliff_continue_needed(
            round_text=text, tool_calls_this_turn=0, retry_used=False,
        ), tail


def test_blocked_on_code_fence_tail():
    # A round ending with a closed code block is a delivered answer.
    text = (
        "Here is the fixed script with all the changes applied to the parser. "
        + "explanation " * 12
        + "\n```python\nprint('done')\n```"
    )
    assert len(text) >= 200
    assert not al._cliff_continue_needed(
        round_text=text, tool_calls_this_turn=0, retry_used=False,
    )


def test_blocked_on_closing_quote_and_bracket_tails():
    # The ~18/129 legit endings the diagnosis warned about: quotes, parens,
    # brackets. None of these is mid-word.
    for tail in ('called "reliable")', "see step 3]", "per RFC 2351}"):
        text = _padded("The context for this decision.", tail)
        assert not al._cliff_continue_needed(
            round_text=text, tool_calls_this_turn=0, retry_used=False,
        ), tail


def test_blocked_on_short_text():
    assert not al._cliff_continue_needed(
        round_text="Sure, I will talk to Flavi",
        tool_calls_this_turn=0,
        retry_used=False,
    )


def test_blocked_on_overlong_text():
    text = _padded("Essay.", "and then we talk to Flavi") + " " + "padding " * 300
    assert len(text) > 2000
    assert not al._cliff_continue_needed(
        round_text=text, tool_calls_this_turn=0, retry_used=False,
    )


def test_blocked_after_tool_calls_in_turn():
    text = _padded("Post-tool summary of everything.", _INCIDENT_TAIL)
    assert not al._cliff_continue_needed(
        round_text=text,
        tool_calls_this_turn=2,
        retry_used=False,
    )


def test_blocked_when_retry_already_used():
    text = _padded("Here is the plan summary.", _INCIDENT_TAIL)
    assert not al._cliff_continue_needed(
        round_text=text,
        tool_calls_this_turn=0,
        retry_used=True,
    )


def test_blocked_on_empty_text():
    assert not al._cliff_continue_needed(
        round_text="   ", tool_calls_this_turn=0, retry_used=False,
    )


def test_accented_latin_tail_triggers():
    # Italian users: a cut right before an accented letter is still mid-word.
    text = _padded("Ecco la sintesi della situazione attuale.", "la storia della città")
    assert al._cliff_continue_needed(
        round_text=text, tool_calls_this_turn=0, retry_used=False,
    )


def test_digit_tail_triggers_and_is_bounded_by_merge_policy():
    # Digits stay in the gate per spec; the caller's shape-aware merge bounds
    # a false positive (a complete "…total: 148") to an appended paragraph.
    text = _padded("The migration report and remaining steps.", "running total so far is 148")
    assert al._cliff_continue_needed(
        round_text=text, tool_calls_this_turn=0, retry_used=False,
    )


# ── replay-style integration through the real loop ───────────────────────

def _collect(gen):
    async def _run():
        return [c async for c in gen]
    return asyncio.run(_run())


def _types(chunks):
    out = []
    for c in chunks:
        if c.startswith("data: ") and not c.startswith("data: [DONE]"):
            try:
                out.append(json.loads(c[6:]))
            except Exception:
                pass
    return out


def _patch_common(monkeypatch):
    monkeypatch.setattr(al, "get_setting", lambda key, default=None: default, raising=False)
    monkeypatch.setattr(al, "get_mcp_manager", lambda: None, raising=False)
    monkeypatch.setattr(al, "estimate_tokens", lambda *a, **k: 10, raising=False)

    async def _fake_exec(block, *a, **k):
        return (block.tool_type, {"output": "ok", "exit_code": 0})
    monkeypatch.setattr(al, "execute_tool_block", _fake_exec, raising=False)


def _run_loop(monkeypatch, round_texts, max_rounds=6, model="qwen3.5:9b"):
    """Round items are consumed in order; the last one repeats if needed.

    A str item is streamed as a text delta; a dict item is streamed as a
    native tool_calls chunk.
    """
    state = {"n": 0}
    calls = []

    async def _fake_stream(_candidates, messages, **kwargs):
        i = min(state["n"], len(round_texts) - 1)
        state["n"] += 1
        calls.append({"messages": [dict(m) for m in messages], "kwargs": kwargs})
        item = round_texts[i]
        if isinstance(item, dict):
            yield f'data: {json.dumps({"type": "tool_calls", "calls": item["calls"]})}\n\n'
        else:
            yield f'data: {json.dumps({"delta": item})}\n\n'
        yield "data: [DONE]\n\n"

    monkeypatch.setattr(al, "stream_llm_with_fallback", _fake_stream, raising=False)

    gen = al.stream_agent_loop(
        "http://x/v1", model,
        [{"role": "user", "content": "summarize the incident and next steps"}],
        max_rounds=max_rounds,
        relevant_tools={"bash", "todowrite"},
    )
    return _types(_collect(gen)), calls


_CUT_TEXT = _padded("Here is the summary of the incident and the recovery plan.", _INCIDENT_TAIL)
_CONTINUATION = "o Moretti tomorrow morning, so the agenda is covered."


def test_cut_round_fires_one_continuation_and_merges(monkeypatch):
    """The incident replayed: mid-word cut -> guard fires -> continuation
    completes the word and the turn; history keeps ONE seamless entry."""
    _patch_common(monkeypatch)
    events, calls = _run_loop(
        monkeypatch,
        [_CUT_TEXT, _CONTINUATION],
    )
    retries = [e for e in events if e.get("type") == "cliff_continue_retry"]
    assert len(retries) == 1, events
    assert retries[0]["round"] == 1
    # The continuation request fed the partial back as an assistant prefix
    # plus the continue-exactly instruction.
    second = calls[1]["messages"]
    assert second[-2]["role"] == "assistant"
    assert second[-2]["content"].endswith("Flavi")
    assert second[-1]["role"] == "system"
    assert "Continue EXACTLY where you stopped" in second[-1]["content"]
    # Metrics carry a SINGLE merged round text with the word completed.
    metrics = [e for e in events if e.get("type") == "metrics"][-1]["data"]
    assert len(metrics["round_texts"]) == 1, metrics["round_texts"]
    assert "talk to Flavio Moretti tomorrow" in metrics["round_texts"][0]
    assert not any(e.get("type") == "rounds_exhausted" for e in events)


def test_guard_caps_at_one_retry(monkeypatch):
    """If the continuation is cut again, the turn ends — the guard cannot
    loop."""
    _patch_common(monkeypatch)
    events, _ = _run_loop(monkeypatch, [_CUT_TEXT, _CUT_TEXT, _CUT_TEXT], max_rounds=8)
    retries = [e for e in events if e.get("type") == "cliff_continue_retry"]
    assert len(retries) == 1, events


def test_completed_tail_never_fires(monkeypatch):
    _patch_common(monkeypatch)
    done = _padded("Here is the full summary of the incident response.", "and the matter is closed.")
    events, _ = _run_loop(monkeypatch, [done])
    assert not any(e.get("type") == "cliff_continue_retry" for e in events), events


def test_code_fence_tail_never_fires(monkeypatch):
    _patch_common(monkeypatch)
    fenced = (
        "Here is the fixed script with all the changes applied to the parser. "
        + "explanation " * 12
        + "\n```python\nprint('done')\n```"
    )
    events, _ = _run_loop(monkeypatch, [fenced])
    assert not any(e.get("type") == "cliff_continue_retry" for e in events), events


def test_post_tool_round_never_fires(monkeypatch):
    """A mid-word-ish summary AFTER a real tool call is not a cliff cut."""
    _patch_common(monkeypatch)
    recovered = {"calls": [{
        "name": "update_plan",
        "arguments": json.dumps({"plan": "- [ ] plan updated"}),
    }]}
    events, _ = _run_loop(monkeypatch, [recovered, _CUT_TEXT])
    assert any(e.get("type") == "tool_start" for e in events), events
    assert not any(e.get("type") == "cliff_continue_retry" for e in events), events


def test_composes_with_trailing_intent_retry(monkeypatch):
    """Both guards can fire in the same turn — exactly once each, in any
    order: round 1 trailing-intent narration, round 2 mid-word cut, round 3
    done."""
    _patch_common(monkeypatch)
    events, _ = _run_loop(
        monkeypatch,
        ["Let me create them:", _CUT_TEXT, "Done — recovered and completed."],
        max_rounds=6,
    )
    trailing = [e for e in events if e.get("type") == "trailing_intent_retry"]
    cliff = [e for e in events if e.get("type") == "cliff_continue_retry"]
    assert len(trailing) == 1, events
    assert len(cliff) == 1, events
    assert not any(e.get("type") == "rounds_exhausted" for e in events)


def test_capitalized_continuation_merges_as_new_paragraph(monkeypatch):
    """False-positive bound (measured 7/103): when the continuation starts
    with a capital letter it is a fresh sentence, NOT a mid-word resume —
    join as a paragraph instead of splicing into the (possibly complete)
    answer."""
    _patch_common(monkeypatch)
    events, _ = _run_loop(
        monkeypatch,
        [_CUT_TEXT, "That said, the recovery plan above already covers it."],
    )
    assert len([e for e in events if e.get("type") == "cliff_continue_retry"]) == 1
    metrics = [e for e in events if e.get("type") == "metrics"][-1]["data"]
    merged = metrics["round_texts"][0]
    assert len(metrics["round_texts"]) == 1
    assert merged.endswith("Flavi\n\nThat said, the recovery plan above already covers it.")


def test_lowercase_continuation_splices_seamlessly(monkeypatch):
    _patch_common(monkeypatch)
    events, _ = _run_loop(monkeypatch, [_CUT_TEXT, "o Moretti will join tomorrow."])
    metrics = [e for e in events if e.get("type") == "metrics"][-1]["data"]
    # cleaned_round is stripped before the merge, so a lowercase resume
    # lands at the exact cut character: "Flavi" + "o Moretti…" -> "Flavio".
    assert "talk to Flavio Moretti will join tomorrow" in metrics["round_texts"][0]


def test_no_fire_on_last_allowed_round(monkeypatch):
    """A cliff retry on the final round would promise a continuation that
    can never run — the guard stays silent and the turn ends as a normal
    completion (clean break, no rounds_exhausted either)."""
    _patch_common(monkeypatch)
    events, _ = _run_loop(monkeypatch, [_CUT_TEXT], max_rounds=1)
    assert not any(e.get("type") == "cliff_continue_retry" for e in events), events
    assert not any(e.get("type") == "rounds_exhausted" for e in events), events


def test_deadline_cut_round_never_fires(monkeypatch):
    """A round whose stream was killed by the wall-clock deadline is NOT a
    natural stop — a continuation cannot heal a transport cut. The clock
    trips between the text delta and [DONE], so the round still carries the
    cut-shaped text when the guard is evaluated."""
    import time as _time

    _patch_common(monkeypatch)
    real_time = _time.time
    shared = {"trip": False, "calls": 0}

    def _trip_clock():
        if shared["trip"]:
            return 9.0e9
        return real_time()

    async def _fake_stream(_candidates, messages, **kwargs):
        shared["calls"] += 1
        yield f'data: {json.dumps({"delta": _CUT_TEXT})}\n\n'
        shared["trip"] = True
        yield "data: [DONE]\n\n"

    monkeypatch.setattr(al.time, "time", _trip_clock)
    monkeypatch.setattr(al, "stream_llm_with_fallback", _fake_stream, raising=False)
    try:
        gen = al.stream_agent_loop(
            "http://x/v1", "qwen3.5:9b",
            [{"role": "user", "content": "summarize the incident"}],
            max_rounds=3,
            relevant_tools={"bash", "todowrite"},
        )
        events = _types(_collect(gen))
    finally:
        monkeypatch.setattr(al.time, "time", real_time)
    # No continuation round ran and no retry was promised.
    assert shared["calls"] == 1, shared
    assert not any(e.get("type") == "cliff_continue_retry" for e in events), events
