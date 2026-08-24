"""Continuation turns inherit recently-used session tools (the 'it just stops' fix).

Short continuations ('yes', 'continue') carry no keywords for tool-RAG, so
the turn armed the model with ~3 generic tools — the model narrated intent,
had nothing to call, and the turn ended. The fix merges tools from the
conversation's recent tool_events on low-signal turns.
"""

from src.agent_tools import ToolBlock  # noqa: E402,F401  (import first to avoid circular)


def test_extracts_recent_tool_names():
    from src.agent_loop import _recent_session_tool_names

    messages = [
        {"role": "user", "content": "scan services"},
        {"role": "assistant", "content": "ok", "metadata": {"tool_events": [
            {"round": 1, "tool": "mcp__c44bc921__project-all", "command": "", "output": ""},
        ]}},
        {"role": "user", "content": "yes please proceed"},
    ]
    assert _recent_session_tool_names(messages) == {"mcp__c44bc921__project-all"}


def test_multiple_messages_and_events():
    from src.agent_loop import _recent_session_tool_names

    messages = [
        {"role": "assistant", "metadata": {"tool_events": [{"tool": "bash"}, {"tool": "python"}]}},
        {"role": "assistant", "metadata": {"tool_events": [{"tool": "bash"}]}},
    ]
    assert _recent_session_tool_names(messages) == {"bash", "python"}


def test_no_metadata_is_safe():
    from src.agent_loop import _recent_session_tool_names

    assert _recent_session_tool_names([{"role": "user", "content": "hi"}]) == set()
    assert _recent_session_tool_names([]) == set()
    assert _recent_session_tool_names(None) == set()
