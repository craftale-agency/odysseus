"""Scope enforcement on raw /api/email routes (probe 2026-11-09 follow-up).

A scope-chat bearer token must NOT reach email data; email:read must pass.
Cookie-session callers bypass the guard (existing auth unchanged).
"""
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from routes.email_routes import (
    EMAIL_READ_SCOPES,
    _require_email_scope,
)


def _request(*, api_token=True, scopes=None):
    return SimpleNamespace(
        state=SimpleNamespace(
            api_token=api_token,
            api_token_scopes=scopes if scopes is not None else ["chat"],
        ),
    )


def test_chat_only_token_is_rejected_on_email_routes():
    guard = _require_email_scope(EMAIL_READ_SCOPES)
    with pytest.raises(HTTPException) as exc:
        guard(_request(scopes=["chat"]))
    assert exc.value.status_code == 403
    assert "email:read" in exc.value.detail


def test_email_read_token_passes():
    guard = _require_email_scope(EMAIL_READ_SCOPES)
    guard(_request(scopes=["chat", "email:read"]))  # no raise


def test_session_caller_untouched():
    guard = _require_email_scope(EMAIL_READ_SCOPES)
    guard(_request(api_token=False))  # no raise
