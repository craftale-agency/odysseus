"""Terminus clamp phrasings + post-external-context read-only email tools.

2026-09-10 E2E root cause (nebula seat, 2 live runs): "read all mails
forwarded from PIETRO.PEZZULLO05@stu-mail.citytech.cuny.edu in
ppezz.dev@gmail.com" retrieved the email tools correctly, then the
local-computer clamp REPLACED the toolset with Terminus — the named-machine
regex read the email LOCAL-PART as a hostname. Every "from X@..." task lost
its email tools and the model improvised for 11 rounds.

Companion fix (baked from the seat's verified in-container change):
READ_PRIVATE removed from POST_EXTERNAL_BLOCKED_EFFECTS so read-only email
tools keep working after external untrusted context, while every
write/execute/egress/side-effect class stays gated.
"""

import pytest

from src.agent_loop import (
    _classify_agent_request,
    _looks_like_local_computer_request,
    _should_clamp_to_terminus,
)
from src.tool_capabilities import (
    ToolEffect,
    ToolRunSecurityContext,
    capabilities_for_tool,
)


def _classify(text):
    return _classify_agent_request([{"role": "user", "content": text}], text)


def _clamp(text, intent=None, **kwargs):
    return _should_clamp_to_terminus(
        kwargs.pop("workspace", None), text,
        intent if intent is not None else _classify(text),
        kwargs.pop("active_document_relevant", False),
        kwargs.pop("active_email", False),
    )


# ---------------------------------------------------------------------------
# The live E2E phrasings: never clamp for email-from constructions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text",
    [
        # The exact live failing phrasing (2 E2E runs lost their email tools).
        "read all mails forwarded from PIETRO.PEZZULLO05@stu-mail.citytech.cuny.edu in ppezz.dev@gmail.com",
        "forwarded from PIETRO.PEZZULLO05@stu-mail.citytech.cuny.edu",
        "from bob@gmail.com",
        "from X@Y",
        "mails from a.b@c.d",
        # Dot-bearing local-parts exercise the backtracking variants: the
        # token regex can settle on "PIETRO" mid-local-part, which is still
        # followed by ".PEZZULLO05@" — every prefix must be rejected.
        "from pietro.pezzullo05@stu-mail.citytech.cuny.edu",
        "from u.s.e.r@host.example",
        "on server@example.com",
    ],
)
def test_email_local_part_never_matches_local_computer(text):
    assert _looks_like_local_computer_request(text) is False, text


def test_person_sender_without_at_never_clamps_via_email_intent():
    """"mails from pietro" has no @ (the regex alone would match "pietro" as
    a machine name) — the intent-domain guard must suppress the clamp."""
    intent = _classify("list mails from pietro")
    assert intent["domains"] == {"email"}
    assert _looks_like_local_computer_request("list mails from pietro") is True
    assert _clamp("list mails from pietro") is False


def test_live_e2e_phrasing_full_pipeline():
    text = ("read all mails forwarded from PIETRO.PEZZULLO05@stu-mail.citytech.cuny.edu"
            " in ppezz.dev@gmail.com")
    intent = _classify(text)
    assert intent["domains"] == {"email"}  # retrieval kept the email lane
    assert _clamp(text) is False  # and the clamp no longer replaces it


def test_mixed_machine_plus_email_request_keeps_email_tools():
    """"read the log on nebula and email me the result" genuinely names a
    machine — but the email intent means the replacing clamp would strip
    the tools half the task needs. No clamp; retrieval keeps both lanes."""
    text = "read the log on nebula and email me the result"
    assert _looks_like_local_computer_request(text) is True
    assert _clamp(text) is False


# ---------------------------------------------------------------------------
# Genuine local-computer requests must KEEP clamping
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text",
    [
        "open the logs on my computer",
        "list files on this machine",
        "run the backup on nebula",
        "check the gpu temperature from pi4.local",
        "files on odysseus-main",
        "read local machine files",
        "on the host system, clean the docker cache",
    ],
)
def test_genuine_local_computer_phrasings_still_clamp(text):
    assert _looks_like_local_computer_request(text) is True, text
    assert _clamp(text) is True, text


def test_non_local_phrasings_do_not_clamp():
    for text in (
        "show me pictures from the gallery",
        "from the gallery, pick the best three",
        "what is on my calendar",
        # NOTE: "from today"/"from the office" style bare tokens DO match the
        # named-machine alternative and always have (pre-existing, no email
        # involved) — out of scope for this fix, kept clamping.
    ):
        assert _clamp(text) is False, text


# ---------------------------------------------------------------------------
# Adjacent guards must not regress
# ---------------------------------------------------------------------------

def test_open_document_and_email_draft_targets_suppress_clamp():
    text = "run the build on nebula"
    assert _clamp(text) is True
    assert _clamp(text, active_document_relevant=True) is False
    assert _clamp(text, active_email=True) is False


def test_workspace_coding_branch_still_clamps():
    intent = _classify("fix the failing test")
    assert _clamp(
        "fix the failing test",
        intent=intent,
        workspace="/repos/odysseus",
    ) is True
    # ...and still never for an email intent, even with a workspace bound.
    email_intent = _classify("read mails from pietro")
    assert _clamp("read mails from pietro", intent=email_intent, workspace="/x") is False


# ---------------------------------------------------------------------------
# BUG B: read-only email tools pass post-external-context; the rest stays gated
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "tool_name",
    ["list_emails", "read_email", "search_emails", "list_email_accounts"],
)
def test_read_only_email_tools_pass_after_external_context(tool_name):
    context = ToolRunSecurityContext(external_untrusted_context_seen=True)
    assert capabilities_for_tool(tool_name).effects == {ToolEffect.READ_PRIVATE}
    assert context.decision_for(tool_name).allowed is True


@pytest.mark.parametrize(
    "tool_name",
    [
        "send_email",
        "delete_email",
        "create_document",
        "bash",
        "write_file",
        "manage_settings",
    ],
)
def test_write_side_effect_tools_still_blocked_after_external_context(tool_name):
    context = ToolRunSecurityContext(external_untrusted_context_seen=True)
    assert context.decision_for(tool_name).allowed is False, tool_name
