"""GitHub MCP integration fixes: embedded-resource extraction + arg coercion.

Replays the exact live failure seen 2026-08-27/28 against the official GitHub
MCP server (github-mcp-server v1.10.1 behind the github-readonly nginx gateway):

  tools/call get_file_contents {owner:"ppezzull", repo:"wave", path:"README.md",
  ref:"main"} → result.content = [
    {"type":"text","text":"successfully downloaded text file (SHA: ecb524ff…)"},
    {"type":"resource","resource":{"uri":"repo://…","mimeType":"text/plain;
     charset=utf-8","text":"<17880-char README>"}}]

The old renderer only surfaced content[0]; the model was blind to the file.
Also covers the coercion fix for `parameter minimal_output is not of type
bool, is string` on search_repositories.
"""

import asyncio
import base64

import pytest

mcp_types = pytest.importorskip("mcp.types")

import src.mcp_manager as mm
from src.mcp_manager import (
    McpManager,
    _coerce_args_to_schema,
    _coerce_scalar,
    _MCP_RESOURCE_TEXT_MAX,
)

README_TEXT = "# wave\n\n> A social market for natural-language on-chain strategies.\n" + "line\n" * 200
SHA = "ecb524ff5fd6e63417431b2f43c7ce9445e12b34"
URI = "repo://ppezzull/wave/sha/c38fa4b47c1449e4cc2f7281f97ff8e9d7f5d9be/contents/README.md"
MIME = "text/plain; charset=utf-8"


def _live_result():
    """Replay of the real gateway response for the failing call."""
    summary = mcp_types.TextContent(type="text", text=f"successfully downloaded text file (SHA: {SHA})")
    resource = mcp_types.EmbeddedResource(
        type="resource",
        resource=mcp_types.TextResourceContents(uri=URI, mimeType=MIME, text=README_TEXT),
    )
    return mcp_types.CallToolResult(content=[summary, resource])


class _Session:
    def __init__(self, result):
        self._result = result
        self.captured = None

    async def call_tool(self, name, arguments):
        self.captured = (name, arguments)
        return self._result


def _run(coro):
    return asyncio.run(coro)


# --- Bug A: embedded-resource extraction -------------------------------------

def test_do_call_surfaces_embedded_resource_text():
    """The exact failing GitHub call: model must receive the README text."""
    result = _run(McpManager()._do_call(_Session(_live_result()), "get_file_contents", {}))

    assert result["exit_code"] == 0
    assert "successfully downloaded text file" in result["stdout"]
    assert "# wave" in result["stdout"]
    assert "on-chain strategies" in result["stdout"]
    assert "[resource: README.md" in result["stdout"]


def test_do_call_truncates_oversized_resource_and_logs(monkeypatch, caplog):
    big = "x" * (_MCP_RESOURCE_TEXT_MAX + 5000)
    resource = mcp_types.EmbeddedResource(
        type="resource",
        resource=mcp_types.TextResourceContents(uri=URI, mimeType=MIME, text=big),
    )
    result_holder = mcp_types.CallToolResult(content=[resource])

    with caplog.at_level("INFO", logger="src.mcp_manager"):
        result = _run(McpManager()._do_call(_Session(result_holder), "get_file_contents", {}))

    assert "…truncated: showing first" in result["stdout"]
    assert f"of {len(big)} characters" in result["stdout"]
    assert len(result["stdout"]) < _MCP_RESOURCE_TEXT_MAX + 1000
    assert any("truncated" in r.message for r in caplog.records)


def test_do_call_base64_blob_resource_decoded_to_text():
    payload = base64.b64encode("def main():\n    return 42\n".encode()).decode()
    resource = mcp_types.EmbeddedResource(
        type="resource",
        resource=mcp_types.BlobResourceContents(
            type="blob", uri="repo://o/r/sha/x/contents/main.py",
            mimeType="text/x-python", blob=payload,
        ),
    )
    result = _run(McpManager()._do_call(
        _Session(mcp_types.CallToolResult(content=[resource])), "get_file_contents", {}))

    assert "def main():" in result["stdout"]
    assert "[resource: main.py" in result["stdout"]


def test_do_call_binary_blob_gets_placeholder_not_crash():
    payload = base64.b64encode(b"\xff\xfe\x00\x01binary").decode()
    resource = mcp_types.EmbeddedResource(
        type="resource",
        resource=mcp_types.BlobResourceContents(
            type="blob", uri="repo://o/r/sha/x/logo.png",
            mimeType="image/png", blob=payload,
        ),
    )
    result = _run(McpManager()._do_call(
        _Session(mcp_types.CallToolResult(content=[resource])), "get_file_contents", {}))

    assert result["exit_code"] == 0
    assert "[binary resource: logo.png" in result["stdout"]


def test_do_call_legacy_json_file_payload_unwrapped():
    payload = base64.b64encode("# legacy shape\n".encode()).decode()
    text = mcp_types.TextContent(
        type="text",
        text='{"type":"file","name":"README.md","encoding":"base64",'
             f'"content":"{payload}","size":16,"sha":"abc"}}',
    )
    result = _run(McpManager()._do_call(
        _Session(mcp_types.CallToolResult(content=[text])), "get_file_contents", {}))

    assert "# legacy shape" in result["stdout"]
    assert "[file: README.md]" in result["stdout"]


def test_do_call_json_search_results_pass_through_untouched():
    raw = '{"total_count": 1, "items": [{"name": "wave", "content": "match"}]}'
    text = mcp_types.TextContent(type="text", text=raw)
    result = _run(McpManager()._do_call(
        _Session(mcp_types.CallToolResult(content=[text])), "search_code", {}))

    assert result["stdout"] == raw  # not a file object → unchanged


# --- Bug B: schema-driven argument coercion ----------------------------------

SEARCH_REPOS_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string"},
        "minimal_output": {"type": "boolean"},
        "page": {"type": "number"},
        "perPage": {"type": "integer"},
        "order": {"type": "string", "enum": ["asc", "desc"]},
    },
    "required": ["query"],
}


def test_coerce_scalar_table():
    assert _coerce_scalar("False", "boolean") is False
    assert _coerce_scalar("false", "boolean") is False
    assert _coerce_scalar("true", "boolean") is True
    assert _coerce_scalar("TRUE", "boolean") is True
    assert _coerce_scalar(True, "boolean") is True
    assert _coerce_scalar("maybe", "boolean") == "maybe"      # unparseable → untouched
    assert _coerce_scalar("true", "string") == "true"          # schema says string → untouched
    assert _coerce_scalar("3.5", "number") == 3.5
    assert _coerce_scalar(3.5, "number") == 3.5
    assert _coerce_scalar("5", "integer") == 5
    assert _coerce_scalar("5.0", "integer") == 5
    assert _coerce_scalar("abc", "integer") == "abc"
    assert _coerce_scalar(None, "boolean") is None


def test_coerce_args_nested_object_and_array():
    schema = {
        "type": "object",
        "properties": {
            "filters": {
                "type": "object",
                "properties": {"archived": {"type": "boolean"}},
            },
            "pages": {"type": "array", "items": {"type": "number"}},
            "label": {"type": "string"},
        },
    }
    out = _coerce_args_to_schema(
        {"filters": {"archived": "True"}, "pages": ["1", "2"], "label": "x"},
        schema, "mcp__srv__t",
    )
    assert out == {"filters": {"archived": True}, "pages": [1.0, 2.0], "label": "x"}


def test_coerce_args_ignores_missing_schema():
    out = _coerce_args_to_schema({"minimal_output": "False"}, {}, "mcp__srv__t")
    assert out == {"minimal_output": "False"}


def test_call_tool_coerces_arguments_before_dispatch():
    """The exact failing call: search_repositories with minimal_output "False"."""
    mgr = McpManager()
    session = _Session(mcp_types.CallToolResult(
        content=[mcp_types.TextContent(type="text", text='{"items":[]}')]))
    mgr._sessions["935ee6e8"] = session
    mgr._tools["935ee6e8"] = [{
        "name": "search_repositories",
        "description": "Search repositories",
        "input_schema": SEARCH_REPOS_SCHEMA,
    }]
    mgr._connections["935ee6e8"] = {"status": "connected", "name": "github-readonly"}

    result = _run(mgr.call_tool(
        "mcp__935ee6e8__search_repositories",
        {"query": "wave org:ppezzull", "minimal_output": "False", "page": "2"},
    ))

    assert result["exit_code"] == 0
    name, args = session.captured
    assert name == "search_repositories"
    assert args["minimal_output"] is False      # real bool dispatched
    assert args["page"] == 2
    assert args["query"] == "wave org:ppezzull"  # strings untouched


# --- Fix C: per-server usage note in the prompt -------------------------------

def _mgr_with_github_server():
    mgr = McpManager()
    mgr._tools["935ee6e8"] = [{
        "name": "get_file_contents",
        "description": "Get file or directory contents",
        "input_schema": {"type": "object", "properties": {
            "owner": {"type": "string"}, "repo": {"type": "string"},
            "path": {"type": "string"}, "ref": {"type": "string"}}},
    }]
    mgr._connections["935ee6e8"] = {"status": "connected", "name": "github-readonly"}
    return mgr


def test_prompt_includes_github_usage_note():
    desc = _mgr_with_github_server().get_tool_descriptions_for_prompt()
    assert "Usage notes:" in desc
    assert "its output IS the file text" in desc
    assert "search_repositories is ONLY for discovering" in desc
    assert 'never "true"' in desc


def test_prompt_usage_note_absent_for_other_servers():
    mgr = _mgr_with_github_server()
    mgr._connections["935ee6e8"] = {"status": "connected", "name": "email-primary"}
    desc = mgr.get_tool_descriptions_for_prompt()
    assert "Usage notes:" not in desc
