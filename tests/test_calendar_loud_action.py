"""manage_calendar: loud missing-action + create_event duplicate guard.

2026-09-10 live incident (qwen3.5:9b): mid-session create payloads lost the
`action` key and the silent `or "list_events"` default answered every one
with "No events between ..." — the model concluded creation was silently
failing and burned ~15 rounds on format roulette. The retry storm then
produced duplicate events whose cleanup turned destructive.
"""
import json
import sys
import uuid

import pytest

from tests.helpers.import_state import clear_fake_database_modules
from tests.helpers.sqlite_db import make_temp_sqlite

clear_fake_database_modules()

import core.database as cdb
from core.database import CalendarEvent

_TS, _ENGINE, _TMPDB = make_temp_sqlite(cdb.Base.metadata)


@pytest.fixture(autouse=True)
def _bind_temp_db(monkeypatch):
    monkeypatch.setitem(sys.modules, "core.database", cdb)
    parent = sys.modules.get("core")
    if parent is not None:
        monkeypatch.setattr(parent, "database", cdb, raising=False)
    monkeypatch.setattr(cdb, "SessionLocal", _TS)
    yield


async def _call(payload: dict, owner: str):
    from src.tool_implementations import do_manage_calendar

    return await do_manage_calendar(json.dumps(payload), owner=owner)


def _events(owner: str):
    db = _TS()
    try:
        rows = db.query(CalendarEvent).filter(CalendarEvent.summary.like("%" + owner[:6] + "%")).all()
        return rows
    finally:
        db.close()


# ---------------------------------------------------------------------------
# FIX 1: missing action — loud for create shapes, silent list kept otherwise
# ---------------------------------------------------------------------------

async def test_missing_action_with_create_shape_is_loud():
    owner = "act" + uuid.uuid4().hex[:6]
    res = await _call({"dtstart": "2026-09-20T10:00:00", "summary": "Dentista"}, owner)
    assert res.get("exit_code") == 1
    error = res.get("error", "")
    assert "requires an explicit 'action'" in error
    assert "create" in error  # names the detected intent
    assert '"action": "create_event"' in error
    assert "list_events" in error and "delete_event" in error  # valid list
    assert _events(owner) == []  # nothing was created


async def test_missing_action_summary_only_is_loud():
    owner = "act" + uuid.uuid4().hex[:6]
    res = await _call({"summary": "Only a title"}, owner)
    assert res.get("exit_code") == 1
    assert "looks like a create" in res.get("error", "")


async def test_missing_action_description_only_is_loud():
    owner = "act" + uuid.uuid4().hex[:6]
    res = await _call({"description": "call the studio"}, owner)
    assert res.get("exit_code") == 1


@pytest.mark.parametrize("payload", [
    {"start": "2026-09-13", "end": "2026-09-27"},   # the live working shape
    {"start_time": "2026-09-13", "end_time": "2026-09-20"},
    {},                                            # no args at all
])
async def test_missing_action_list_shapes_keep_silent_default(payload):
    owner = "act" + uuid.uuid4().hex[:6]
    res = await _call(payload, owner)
    assert res.get("exit_code") == 0
    assert "events" in res  # a list response, not an error
    assert res.get("response", "").startswith("No events")


async def test_whitespace_action_same_as_missing_create_shape():
    owner = "act" + uuid.uuid4().hex[:6]
    res = await _call({"action": "   ", "summary": "Trim", "dtstart": "2026-09-20T09:00:00"}, owner)
    assert res.get("exit_code") == 1
    assert "looks like a create" in res.get("error", "")


async def test_unknown_action_error_unchanged():
    owner = "act" + uuid.uuid4().hex[:6]
    res = await _call({"action": "teleport_event", "uid": "x"}, owner)
    assert res.get("exit_code") == 1
    assert "Unknown action: teleport_event" in res.get("error", "")


async def test_create_with_explicit_action_still_works():
    owner = "act" + uuid.uuid4().hex[:6]
    res = await _call(
        {"action": "create_event", "summary": owner + " Standup",
         "dtstart": "2026-09-20T10:00:00"},
        owner,
    )
    assert res.get("exit_code") == 0
    assert "Created event" in res.get("response", "")


# ---------------------------------------------------------------------------
# FIX 2: duplicate guard on create_event
# ---------------------------------------------------------------------------

def _create_payload(owner, summary, start, end=None):
    payload = {
        "action": "create_event",
        "summary": summary,
        "dtstart": start,
    }
    if end:
        payload["dtend"] = end
    return payload


async def _seed(owner, summary="Dentista", start="2026-09-20T10:00:00", end="2026-09-20T11:00:00"):
    res = await _call(_create_payload(owner, summary, start, end), owner)
    assert res.get("exit_code") == 0
    return res


async def test_same_summary_same_time_blocked():
    owner = "dup" + uuid.uuid4().hex[:6]
    await _seed(owner, summary=owner + " Dentista")
    res = await _call(_create_payload(owner, owner + " Dentista",
                                      "2026-09-20T10:00:00", "2026-09-20T11:00:00"), owner)
    assert res.get("exit_code") == 0
    assert res.get("duplicate") is True
    response = res.get("response", "")
    assert "A similar event already exists" in response
    assert "allow_duplicate=true" in response
    assert len(_events(owner)) == 1  # nothing new created


async def test_same_summary_overlapping_time_blocked():
    """Shifted-by-minutes duplicates (the exact-match hole) must be caught
    by the overlap comparison."""
    owner = "dup" + uuid.uuid4().hex[:6]
    await _seed(owner, summary=owner + " Gym", start="2026-09-20T06:00:00", end="2026-09-20T07:00:00")
    res = await _call(_create_payload(owner, owner + " Gym",
                                      "2026-09-20T06:30:00", "2026-09-20T07:30:00"), owner)
    assert res.get("duplicate") is True
    assert len(_events(owner)) == 1


async def test_case_insensitive_summary_blocked():
    owner = "dup" + uuid.uuid4().hex[:6]
    await _seed(owner, summary=owner + " Morning Gym")
    res = await _call(_create_payload(owner, owner + " MORNING GYM",
                                      "2026-09-20T10:00:00", "2026-09-20T11:00:00"), owner)
    assert res.get("duplicate") is True
    assert len(_events(owner)) == 1


async def test_allow_duplicate_true_creates_anyway():
    owner = "dup" + uuid.uuid4().hex[:6]
    await _seed(owner, summary=owner + " Gym")
    res = await _call(
        {**_create_payload(owner, owner + " Gym", "2026-09-20T10:00:00", "2026-09-20T11:00:00"),
         "allow_duplicate": True},
        owner,
    )
    assert res.get("exit_code") == 0
    assert res.get("duplicate") is None
    assert len(_events(owner)) == 2


async def test_different_summary_same_time_created():
    owner = "dup" + uuid.uuid4().hex[:6]
    await _seed(owner, summary=owner + " Gym")
    res = await _call(_create_payload(owner, owner + " Physio",
                                      "2026-09-20T10:00:00", "2026-09-20T11:00:00"), owner)
    assert res.get("exit_code") == 0
    assert res.get("duplicate") is None
    assert len(_events(owner)) == 2


async def test_same_summary_non_overlapping_time_created():
    owner = "dup" + uuid.uuid4().hex[:6]
    await _seed(owner, summary=owner + " Gym", start="2026-09-20T06:00:00", end="2026-09-20T07:00:00")
    res = await _call(_create_payload(owner, owner + " Gym",
                                      "2026-09-20T18:00:00", "2026-09-20T19:00:00"), owner)
    assert res.get("exit_code") == 0
    assert len(_events(owner)) == 2


async def test_cancelled_duplicate_does_not_block():
    from routes.calendar_routes import _record_caldav_delete_tombstone  # noqa: F401
    owner = "dup" + uuid.uuid4().hex[:6]
    first = await _seed(owner, summary=owner + " Gym")
    db = _TS()
    try:
        row = db.query(CalendarEvent).filter(CalendarEvent.uid == first["uid"]).first()
        row.status = "cancelled"
        db.commit()
    finally:
        db.close()
    res = await _call(_create_payload(owner, owner + " Gym",
                                      "2026-09-20T10:00:00", "2026-09-20T11:00:00"), owner)
    assert res.get("duplicate") is None
    assert len(_events(owner)) == 2


# ---------------------------------------------------------------------------
# Batch normalization path unaffected
# ---------------------------------------------------------------------------

async def test_batch_events_two_distinct_events_still_created():
    owner = "bat" + uuid.uuid4().hex[:6]
    payload = {
        "events": [
            {"summary": owner + " Gym", "start": "2026-09-21T06:00:00", "end": "2026-09-21T07:00:00"},
            {"summary": owner + " Gym", "start": "2026-09-22T06:00:00", "end": "2026-09-22T07:00:00"},
        ]
    }
    res = await _call(payload, owner)
    assert res.get("exit_code") == 0
    assert "Created 2 event(s)" in res.get("response", "")
    assert len(_events(owner)) == 2


async def test_batch_events_true_duplicate_reports_blocked():
    """A batch containing the same event twice reports the duplicate guard
    message instead of silently creating two copies."""
    owner = "bat" + uuid.uuid4().hex[:6]
    payload = {
        "events": [
            {"summary": owner + " Gym", "start": "2026-09-21T06:00:00", "end": "2026-09-21T07:00:00"},
            {"summary": owner + " Gym", "start": "2026-09-21T06:00:00", "end": "2026-09-21T07:00:00"},
        ]
    }
    res = await _call(payload, owner)
    assert len(_events(owner)) == 1
    response = res.get("response", "")
    assert "A similar event already exists" in response
    assert "allow_duplicate=true" in response
