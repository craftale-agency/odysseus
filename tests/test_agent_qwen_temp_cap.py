"""Two-tier Qwen temperature cap (2026-08-28 "Flavi" EOS incident).

qwen3.5:9b running at the preset-default temperature 1.0 sampled a spurious
EOS at token 126, mid-word ("Flavi"), with every layer reporting success.
The existing safety cap `_ody_qwen_temperature_cap` (0.2) only matched the
`odysseus-qwen3*` finetunes, so plain qwen3/qwen3.5 base models ran at the
caller's full temperature. Qwen3 vendor guidance is 0.6-0.7.

`_qwen_temperature_cap` widens the net WITHOUT touching finetune behavior:
  - odysseus-qwen3*  -> hard cap 0.2 (unchanged — finetune destabilizes above)
  - plain qwen3*/qwen3.5* -> cap 0.7 (lower requests pass through)
  - everything else -> passthrough
"""

import asyncio
import json

import src.agent_loop as al


# ── pure function: per-model cap values ──────────────────────────────────

def test_finetune_cap_unchanged_at_02():
    assert al._qwen_temperature_cap("odysseus-qwen3:8b", 1.0) == 0.2
    assert al._qwen_temperature_cap("ODYSSEUS-QWEN3:8B", 0.9) == 0.2
    # min() semantics: an already-low request is not raised.
    assert al._qwen_temperature_cap("odysseus-qwen3:8b", 0.1) == 0.1
    # None materializes at the cap (mirrors the finetune helper).
    assert al._qwen_temperature_cap("odysseus-qwen3:8b", None) == 0.2


def test_plain_qwen_capped_at_07():
    for model in ("qwen3.5:9b", "qwen3:14b", "qwen3-coder-30b", "Qwen3.5:9b"):
        assert al._qwen_temperature_cap(model, 1.0) == 0.7, model
        assert al._qwen_temperature_cap(model, 0.99) == 0.7, model


def test_plain_qwen_lower_requests_pass_through():
    assert al._qwen_temperature_cap("qwen3.5:9b", 0.5) == 0.5
    assert al._qwen_temperature_cap("qwen3:14b", 0.7) == 0.7
    assert al._qwen_temperature_cap("qwen3.5:9b", 0.0) == 0.0


def test_plain_qwen_none_materializes_at_cap():
    assert al._qwen_temperature_cap("qwen3.5:9b", None) == 0.7


def test_plain_qwen_garbage_temperature_falls_back_to_cap():
    assert al._qwen_temperature_cap("qwen3.5:9b", "banana") == 0.7
    assert al._qwen_temperature_cap("odysseus-qwen3:8b", "banana") == 0.2


def test_non_qwen_models_pass_through_untouched():
    for model in ("gemma4:12b", "gpt-4o", "deepseek-v4-flash", "thinker14b:latest"):
        assert al._qwen_temperature_cap(model, 1.0) == 1.0, model
        assert al._qwen_temperature_cap(model, None) is None, model


def test_finetune_prefix_is_not_plain_qwen():
    # The finetune branch must keep its stricter cap; "odysseus-qwen3…" does
    # not start with "qwen3", and older qwen families are untouched.
    assert not al._is_plain_qwen3_model("odysseus-qwen3:8b")
    assert al._is_plain_qwen3_model("qwen3.5:9b")
    assert al._is_plain_qwen3_model("qwen3:14b")
    assert not al._is_plain_qwen3_model("qwen2.5:7b")
    assert not al._is_plain_qwen3_model("")


# ── integration: the loop sends the capped temperature ───────────────────

def _collect(gen):
    async def _run():
        return [c async for c in gen]
    return asyncio.run(_run())


def _run_loop(monkeypatch, model, temperature=1.0):
    seen = {}

    async def _fake_stream(_candidates, messages, **kwargs):
        seen["temperature"] = kwargs.get("temperature")
        yield f'data: {json.dumps({"delta": "Done."})}\n\n'
        yield "data: [DONE]\n\n"

    monkeypatch.setattr(al, "stream_llm_with_fallback", _fake_stream, raising=False)
    monkeypatch.setattr(al, "get_setting", lambda key, default=None: default, raising=False)
    monkeypatch.setattr(al, "get_mcp_manager", lambda: None, raising=False)
    monkeypatch.setattr(al, "estimate_tokens", lambda *a, **k: 10, raising=False)

    chunks = _collect(al.stream_agent_loop(
        "http://x/v1", model,
        [{"role": "user", "content": "hi"}],
        max_rounds=2,
        temperature=temperature,
    ))
    assert any(c.startswith("data: [DONE]") for c in chunks)
    return seen["temperature"]


def test_loop_caps_plain_qwen35_to_07(monkeypatch):
    assert _run_loop(monkeypatch, "qwen3.5:9b") == 0.7


def test_loop_keeps_finetune_cap_at_02(monkeypatch):
    assert _run_loop(monkeypatch, "odysseus-qwen3:8b") == 0.2


def test_loop_passthrough_for_non_qwen(monkeypatch):
    assert _run_loop(monkeypatch, "gemma4:12b") == 1.0


def _run_loop_with_fallbacks(monkeypatch, primary_model, fallback_model):
    """Exercise the per-candidate request factory (authoritative in
    production: llm_core merges its kwargs over the loop defaults)."""
    captured = {}

    async def _fake_stream(candidates, messages, **kwargs):
        factory = kwargs["candidate_request_factory"]
        for i, (url, model, headers) in enumerate(candidates):
            req = factory(i, url, model, headers)
            captured[model] = req["kwargs"]["temperature"]
        yield f'data: {json.dumps({"delta": "Done."})}\n\n'
        yield "data: [DONE]\n\n"

    monkeypatch.setattr(al, "stream_llm_with_fallback", _fake_stream, raising=False)
    monkeypatch.setattr(al, "get_setting", lambda key, default=None: default, raising=False)
    monkeypatch.setattr(al, "get_mcp_manager", lambda: None, raising=False)
    monkeypatch.setattr(al, "estimate_tokens", lambda *a, **k: 10, raising=False)

    chunks = _collect(al.stream_agent_loop(
        "http://x/v1", primary_model,
        [{"role": "user", "content": "hi"}],
        max_rounds=1,
        temperature=1.0,
        fallbacks=[
            ("http://x/v1", primary_model, {}),
            ("http://x/v1", fallback_model, {}),
        ],
    ))
    assert any(c.startswith("data: [DONE]") for c in chunks)
    return captured


def test_factory_caps_plain_qwen_fallback_at_07(monkeypatch):
    captured = _run_loop_with_fallbacks(monkeypatch, "gemma4:12b", "qwen3.5:9b")
    assert captured["gemma4:12b"] == 1.0, captured
    assert captured["qwen3.5:9b"] == 0.7, captured


def test_factory_keeps_finetune_fallback_at_02(monkeypatch):
    captured = _run_loop_with_fallbacks(monkeypatch, "gemma4:12b", "odysseus-qwen3:8b")
    assert captured["odysseus-qwen3:8b"] == 0.2, captured


def test_factory_caps_plain_qwen_under_finetune_primary(monkeypatch):
    captured = _run_loop_with_fallbacks(monkeypatch, "odysseus-qwen3:8b", "qwen3:14b")
    assert captured["odysseus-qwen3:8b"] == 0.2, captured
    assert captured["qwen3:14b"] == 0.7, captured
