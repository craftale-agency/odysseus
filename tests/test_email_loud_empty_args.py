"""read_email: loud empty-args guard before any mailbox contact.

2026-09-10 live incident (qwen3.5:9b): the model emitted read_email with
empty args {} twice. The old path resolved the DEFAULT account, opened IMAP
for it, and surfaced that account's own error ("Basic authentication is
disabled" on a basic-auth-disabled mailbox) instead of "you gave me no
uid" — a failure the model could not act on; the loop-breaker keyed both
calls as identical repetition and the session degraded to plain chat. Same
family as the calendar action-less spiral (0c29caf7).
"""
import sqlite3

import pytest

pytest.importorskip("mcp")

import mcp_servers.email_server as es


@pytest.fixture(autouse=True)
def _clear_mcp_email_owner_env(monkeypatch):
    for key in es._OWNER_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    es._ACCOUNT_CACHE.clear()
    yield
    es._ACCOUNT_CACHE.clear()


def _init_accounts_db(path, rows):
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE email_accounts (
            id TEXT PRIMARY KEY,
            owner TEXT,
            name TEXT NOT NULL,
            is_default INTEGER NOT NULL DEFAULT 0,
            enabled INTEGER NOT NULL DEFAULT 1,
            imap_host TEXT,
            imap_port INTEGER,
            imap_user TEXT,
            imap_password TEXT,
            imap_starttls INTEGER,
            smtp_host TEXT,
            smtp_port INTEGER,
            smtp_security TEXT,
            smtp_user TEXT,
            smtp_password TEXT,
            from_address TEXT,
            created_at TEXT
        )
        """
    )
    conn.executemany(
        """
        INSERT INTO email_accounts
        (id, owner, name, is_default, enabled, imap_host, imap_port, imap_user,
         imap_password, imap_starttls, smtp_host, smtp_port, smtp_security,
         smtp_user, smtp_password, from_address, created_at)
        VALUES (?, ?, ?, ?, 1, 'imap.example.com', 993, ?, '', 1,
                'smtp.example.com', 465, 'ssl', ?, '', ?, ?)
        """,
        rows,
    )
    conn.commit()
    conn.close()


_ALICE_ROWS = [
    ("acct-a", "alice", "Alice Mail", 1, "alice@example.com", "alice@example.com", "alice@example.com", "2026-01-01"),
]
_ALICE_TWO_ROWS = [
    ("acct-a", "alice", "Alice Mail", 1, "alice@example.com", "alice@example.com", "alice@example.com", "2026-01-01"),
    ("acct-c", "alice", "CUNY Mail", 0, "cuny@example.edu", "cuny@example.edu", "cuny@example.edu", "2026-01-02"),
]


def _forbid_mailbox(monkeypatch):
    """Any touch of the mailbox layer on an arg-less call is a regression."""
    def _boom(*args, **kwargs):
        raise AssertionError("read_email must not contact a mailbox on empty args")
    monkeypatch.setattr(es, "_imap_connect", _boom)
    monkeypatch.setattr(es, "_read_email", _boom)
    monkeypatch.setattr(es, "_read_email_across_accounts", _boom)


async def _read(payload):
    return await es.call_tool("read_email", {**payload, "_odysseus_owner": "alice"})


# ---------------------------------------------------------------------------
# Guard: empty args are loud and side-effect free
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_empty_args_single_account_is_loud_without_contacting_imap(tmp_path, monkeypatch):
    db_path = tmp_path / "app.db"
    _init_accounts_db(db_path, _ALICE_ROWS)
    monkeypatch.setattr(es, "APP_DB", str(db_path))
    _forbid_mailbox(monkeypatch)

    out = await _read({})
    text = out[0].text

    assert text.startswith("Error:")
    assert "requires an explicit 'uid'" in text
    assert "'message_id'" in text
    assert "no mailbox was contacted" in text
    assert "list_emails" in text


@pytest.mark.asyncio
async def test_empty_args_lists_account_names_when_several_exist(tmp_path, monkeypatch):
    db_path = tmp_path / "app.db"
    _init_accounts_db(db_path, _ALICE_TWO_ROWS)
    monkeypatch.setattr(es, "APP_DB", str(db_path))
    _forbid_mailbox(monkeypatch)

    out = await _read({})
    text = out[0].text

    assert "'account'" in text
    assert "Alice Mail (default)" in text
    assert "CUNY Mail" in text


@pytest.mark.asyncio
async def test_account_without_uid_names_list_emails_not_an_auth_error(tmp_path, monkeypatch):
    # The live incident shape: the model named an account but no uid, and the
    # old default surfaced that account's "Basic authentication is disabled"
    # instead of pointing back at list_emails.
    db_path = tmp_path / "app.db"
    _init_accounts_db(db_path, _ALICE_TWO_ROWS)
    monkeypatch.setattr(es, "APP_DB", str(db_path))
    _forbid_mailbox(monkeypatch)

    out = await _read({"account": "CUNY Mail"})
    text = out[0].text

    assert text.startswith("Error:")
    assert "list_emails" in text
    assert "Basic authentication" not in text


@pytest.mark.asyncio
async def test_whitespace_uid_is_treated_as_missing(tmp_path, monkeypatch):
    db_path = tmp_path / "app.db"
    _init_accounts_db(db_path, _ALICE_ROWS)
    monkeypatch.setattr(es, "APP_DB", str(db_path))
    _forbid_mailbox(monkeypatch)

    out = await _read({"uid": "   "})
    text = out[0].text

    assert text.startswith("Error:")
    assert "requires an explicit 'uid'" in text


# ---------------------------------------------------------------------------
# Regression: valid calls pass the guard unchanged
# ---------------------------------------------------------------------------

def _canned_read(uid="42"):
    return {
        "uid": uid,
        "account": "Alice Mail",
        "account_email": "alice@example.com",
        "account_id": "acct-a",
        "message_id": "<42@example.com>",
        "subject": "Quarterly report",
        "from": "Alice",
        "from_address": "alice@example.com",
        "date": "Fri, 11 Sep 2026 10:00:00 +0000",
        "body": "Hello from the regression test.",
        "attachments": [],
    }


@pytest.mark.asyncio
async def test_uid_still_reaches_read_email(tmp_path, monkeypatch):
    db_path = tmp_path / "app.db"
    _init_accounts_db(db_path, _ALICE_ROWS)
    monkeypatch.setattr(es, "APP_DB", str(db_path))
    seen = {}

    def _fake_read(**kwargs):
        seen.update(kwargs)
        return _canned_read(kwargs.get("uid") or "42")

    monkeypatch.setattr(es, "_read_email", _fake_read)

    out = await _read({"uid": "42"})
    text = out[0].text

    assert seen.get("uid") == "42"
    assert seen.get("account") is None
    assert "Quarterly report" in text


@pytest.mark.asyncio
async def test_int_uid_is_normalized_and_passes(tmp_path, monkeypatch):
    db_path = tmp_path / "app.db"
    _init_accounts_db(db_path, _ALICE_ROWS)
    monkeypatch.setattr(es, "APP_DB", str(db_path))
    seen = {}

    def _fake_read(**kwargs):
        seen.update(kwargs)
        return _canned_read(str(kwargs.get("uid")))

    monkeypatch.setattr(es, "_read_email", _fake_read)

    out = await _read({"uid": 7})

    assert seen.get("uid") == "7"
    assert "Quarterly report" in out[0].text


@pytest.mark.asyncio
async def test_message_id_only_still_reaches_read_email(tmp_path, monkeypatch):
    db_path = tmp_path / "app.db"
    _init_accounts_db(db_path, _ALICE_ROWS)
    monkeypatch.setattr(es, "APP_DB", str(db_path))
    seen = {}

    def _fake_read(**kwargs):
        seen.update(kwargs)
        return _canned_read()

    monkeypatch.setattr(es, "_read_email", _fake_read)

    out = await _read({"message_id": "<42@example.com>"})
    text = out[0].text

    assert seen.get("message_id") == "<42@example.com>"
    assert "Quarterly report" in text


@pytest.mark.asyncio
async def test_uid_with_two_accounts_still_fans_out(tmp_path, monkeypatch):
    db_path = tmp_path / "app.db"
    _init_accounts_db(db_path, _ALICE_TWO_ROWS)
    monkeypatch.setattr(es, "APP_DB", str(db_path))
    seen = {}

    def _fake_fanout(**kwargs):
        seen.update(kwargs)
        return _canned_read()

    monkeypatch.setattr(es, "_read_email_across_accounts", _fake_fanout)

    out = await _read({"uid": "42"})

    assert seen.get("uid") == "42"
    assert "Quarterly report" in out[0].text


# ---------------------------------------------------------------------------
# Internal caller: _read_email validates before resolving/connecting
# ---------------------------------------------------------------------------

def test_internal_read_email_validates_before_any_connection(monkeypatch):
    def _boom(*args, **kwargs):
        raise AssertionError("_read_email must validate args before loading config or connecting")

    monkeypatch.setattr(es, "_load_config", _boom)
    monkeypatch.setattr(es, "_imap_connect", _boom)

    res = es._read_email(uid=None, message_id=None, folder="INBOX", account=None)

    assert res == {"error": "No UID or Message-ID provided"}


def test_internal_read_email_empty_strings_validate_before_any_connection(monkeypatch):
    def _boom(*args, **kwargs):
        raise AssertionError("_read_email must validate args before loading config or connecting")

    monkeypatch.setattr(es, "_load_config", _boom)
    monkeypatch.setattr(es, "_imap_connect", _boom)

    res = es._read_email(uid="  ", message_id="", folder="INBOX", account="CUNY Mail")

    assert res == {"error": "No UID or Message-ID provided"}
