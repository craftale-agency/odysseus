"""Recovery for models emitting concatenated JSON objects as tool arguments.

Observed live (2026-08-24, qwen3.5:9b): parallel tool calls arrive as
`{}{"q": "", "limit": 50}` — json.loads rejects it and the call used to be
silently dropped, surfacing as empty agent rounds / "no substantive output".

Imports follow the repo's circular-import workaround (see
test_ask_user_tool.py): src.agent_tools first, function-local for the rest.
"""

import json

from src.agent_tools import ToolBlock  # noqa: E402,F401  (import first to avoid circular)


def test_concatenated_empty_then_real_object_recovers():
    from src.tool_schemas import function_call_to_tool_block

    block = function_call_to_tool_block("ask_user", '{}{"q": "", "limit": 50}')
    assert block is not None
    assert json.loads(block.content) == {"q": "", "limit": 50}


def test_concatenated_multiple_keeps_last_non_empty():
    from src.tool_schemas import function_call_to_tool_block

    block = function_call_to_tool_block("ask_user", '{"a": 1}{"b": 2}')
    assert block is not None
    # The real args follow the stale prefix; keep the last non-empty object.
    assert json.loads(block.content) == {"b": 2}


def test_all_empty_objects_is_valid_empty_args():
    from src.tool_schemas import function_call_to_tool_block

    # Seen live (2026-08-24): mcp__…__project-all called with '{}{}' — the
    # RIGHT tool, no arguments. Empty args are valid; a tool that requires
    # arguments rejects them loudly at its own validation layer.
    import json as _json

    block = function_call_to_tool_block("ask_user", "{}{}")
    assert block is not None and _json.loads(block.content) == {}


def test_trailing_garbage_is_not_recovered():
    from src.tool_schemas import function_call_to_tool_block

    assert function_call_to_tool_block("ask_user", '{"a": 1} not json') is None


def test_valid_single_object_untouched():
    from src.tool_schemas import function_call_to_tool_block

    block = function_call_to_tool_block("ask_user", '{"q": "x"}')
    assert block is not None and json.loads(block.content) == {"q": "x"}


def test_empty_string_still_empty_args():
    from src.tool_schemas import function_call_to_tool_block

    block = function_call_to_tool_block("ask_user", "")
    assert block is not None and json.loads(block.content) == {}
