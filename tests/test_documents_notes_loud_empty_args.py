"""manage_documents / manage_notes: loud empty-args guards on siblings.

Same family as the 2026-09-10 live incident (read_email with empty args
defaulting to the wrong account) and the calendar action-less spiral
(0c29caf7): silent defaults on empty/partial arguments route the call
somewhere the model never asked for. Here:

- manage_documents: an action-less payload that names a document is
  ambiguous between read and delete (was silently listed); delete without
  a document_id soft-deleted the ACTIVE or MOST RECENT document instead.
- manage_notes toggle_item: a missing index silently toggled item 0 of a
  multi-item checklist — a misdirected mutation.
"""
import json
import sys
import uuid

import pytest

from tests.helpers.import_state import clear_fake_database_modules
from tests.helpers.sqlite_db import make_temp_sqlite

clear_fake_database_modules()

import core.database as cdb
from core.database import Document, Note

_TS, _ENGINE, _TMPDB = make_temp_sqlite(cdb.Base.metadata)


@pytest.fixture(autouse=True)
def _bind_temp_db(monkeypatch):
    monkeypatch.setitem(sys.modules, "core.database", cdb)
    parent = sys.modules.get("core")
    if parent is not None:
        monkeypatch.setattr(parent, "database", cdb, raising=False)
    monkeypatch.setattr(cdb, "SessionLocal", _TS)
    yield


def _mk_doc(owner="alice", title="Q3 Plan", content="Plan body", active=True) -> str:
    doc_id = "doc-" + uuid.uuid4().hex[:10]
    db = _TS()
    try:
        db.add(Document(
            id=doc_id,
            owner=owner,
            title=title,
            language="markdown",
            current_content=content,
            version_count=1,
            is_active=active,
        ))
        db.commit()
    finally:
        db.close()
    return doc_id


def _doc(doc_id) -> Document:
    db = _TS()
    try:
        return db.query(Document).filter(Document.id == doc_id).first()
    finally:
        db.close()


def _mk_note(items, title="Shopping", owner="alice") -> str:
    note_id = str(uuid.uuid4())
    db = _TS()
    try:
        db.add(Note(
            id=note_id,
            owner=owner,
            title=title,
            items=json.dumps(items),
            note_type="checklist",
        ))
        db.commit()
    finally:
        db.close()
    return note_id


def _note(note_id) -> Note:
    db = _TS()
    try:
        return db.query(Note).filter(Note.id == note_id).first()
    finally:
        db.close()


async def _manage_documents(payload: dict, ctx=None):
    from src.agent_tools.document_tools import ManageDocumentTool
    return await ManageDocumentTool().execute(json.dumps(payload), ctx or {"owner": "alice"})


async def _manage_notes(payload: dict, owner="alice"):
    from src.tool_implementations import do_manage_notes
    return await do_manage_notes(json.dumps(payload), owner=owner)


# ---------------------------------------------------------------------------
# manage_documents: action-less payload that names a document
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_document_id_without_action_is_loud():
    doc_id = _mk_doc()
    res = await _manage_documents({"document_id": doc_id})

    assert res.get("exit_code") == 1
    error = res.get("error", "")
    assert "requires an explicit 'action'" in error
    assert '"action": "read"' in error
    assert '"action": "delete"' in error
    assert _doc(doc_id).is_active is True  # nothing happened to the doc


@pytest.mark.asyncio
async def test_id_alias_without_action_is_loud():
    doc_id = _mk_doc()
    res = await _manage_documents({"id": doc_id})

    assert res.get("exit_code") == 1
    assert "requires an explicit 'action'" in res.get("error", "")


@pytest.mark.asyncio
async def test_null_action_with_document_id_is_loud():
    doc_id = _mk_doc()
    res = await _manage_documents({"action": None, "document_id": doc_id})

    assert res.get("exit_code") == 1
    assert "requires an explicit 'action'" in res.get("error", "")


@pytest.mark.asyncio
async def test_no_args_still_defaults_to_list():
    _mk_doc()
    res = await _manage_documents({})

    assert res.get("exit_code") == 0
    assert "Q3 Plan" in res.get("response", "")


@pytest.mark.asyncio
async def test_list_filters_without_action_still_list():
    _mk_doc(title="Q3 Plan")
    _mk_doc(title="Groceries")
    res = await _manage_documents({"search": "Q3"})

    assert res.get("exit_code") == 0
    assert "Q3 Plan" in res.get("response", "")
    assert "Groceries" not in res.get("response", "")


# ---------------------------------------------------------------------------
# manage_documents: delete requires an explicit document_id
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_delete_without_id_is_loud_and_deletes_nothing():
    # The old fallback soft-deleted the ACTIVE document, then the most
    # recently updated one — a destructive default on an empty argument.
    kept_a = _mk_doc(title="Kept A")
    kept_b = _mk_doc(title="Kept B")
    res = await _manage_documents({"action": "delete"})

    assert res.get("exit_code") == 1
    error = res.get("error", "")
    assert "requires an explicit document_id" in error
    assert "no document was deleted" in error
    assert _doc(kept_a).is_active is True
    assert _doc(kept_b).is_active is True


@pytest.mark.asyncio
async def test_delete_with_id_still_works():
    doc_id = _mk_doc(title="Doomed")
    res = await _manage_documents({"action": "delete", "document_id": doc_id})

    assert res.get("exit_code") == 0
    assert "Deleted document 'Doomed'" in res.get("response", "")
    assert _doc(doc_id).is_active is False


@pytest.mark.asyncio
async def test_delete_with_unknown_id_reports_not_found():
    res = await _manage_documents({"action": "delete", "document_id": "doc-nope"})

    assert res.get("exit_code") == 1
    assert "not found" in res.get("error", "")


@pytest.mark.asyncio
async def test_read_with_id_unchanged():
    doc_id = _mk_doc(title="Readable", content="Read me")
    res = await _manage_documents({"action": "read", "document_id": doc_id})

    assert res.get("exit_code") == 0
    assert "Read me" in res.get("response", "")


# ---------------------------------------------------------------------------
# manage_notes toggle_item: no index on a multi-item checklist
# ---------------------------------------------------------------------------

def _items(note):
    return json.loads(_note(note).items)


@pytest.mark.asyncio
async def test_toggle_item_without_index_on_multi_item_is_loud():
    note_id = _mk_note([
        {"text": "milk", "done": False},
        {"text": "eggs", "done": False},
        {"text": "bread", "done": False},
    ])
    res = await _manage_notes({"action": "toggle_item", "id": note_id})

    assert res.get("exit_code") == 1
    error = res.get("error", "")
    assert "requires an explicit 'index'" in error
    assert "3 items (0-2)" in error
    assert "no item was toggled" in error
    assert all(not item["done"] for item in _items(note_id))


@pytest.mark.asyncio
async def test_toggle_item_single_item_defaults_to_index_zero():
    note_id = _mk_note([{"text": "only", "done": False}])
    res = await _manage_notes({"action": "toggle_item", "id": note_id})

    assert res.get("exit_code") == 0
    assert _items(note_id)[0]["done"] is True


@pytest.mark.asyncio
async def test_toggle_item_explicit_index_still_works():
    note_id = _mk_note([
        {"text": "milk", "done": False},
        {"text": "eggs", "done": False},
        {"text": "bread", "done": False},
    ])
    res = await _manage_notes({"action": "toggle_item", "id": note_id, "index": 2})

    assert res.get("exit_code") == 0
    items = _items(note_id)
    assert items[0]["done"] is False
    assert items[1]["done"] is False
    assert items[2]["done"] is True
