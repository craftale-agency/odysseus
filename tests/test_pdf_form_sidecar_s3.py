"""Field-sidecar storage under backend=s3 (2nd-round review follow-ups).

Object-stored PDFs keep their field schema in a COMPANION OBJECT
(<object-key>.fields.json) in the same bucket — never next to the
materialized /tmp copy (deleted on route exit, so the old file-based
sidecar was orphaned and every later export/render degraded), and never
as an uploads.json row. Local rows keep the historical file behavior.

Also pins the import_pdf route: extraction failures degrade to the
plain-PDF document instead of escaping as a raw 500, and the s3 happy
path writes the companion through the backend.
"""
import io
import json
import os
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import storage_backend
from src.pdf_form_doc import (
    SIDECAR_SUFFIX,
    load_field_sidecar,
    save_field_sidecar,
    sidecar_path,
)
from src.storage_backend import S3Backend

BUCKET = "odysseus-test"


class FakeS3Backend(S3Backend):
    """In-memory object store: real URI/key semantics, no network, no boto3."""

    def __init__(self, bucket=BUCKET):
        super().__init__(bucket=bucket)
        self.objects: dict[str, bytes] = {}
        self.content_types: dict[str, str] = {}
        self.get_bytes_calls: list[str] = []

    def put(self, key, fileobj, content_type):
        self.objects[key] = fileobj.read()
        self.content_types[key] = content_type

    def get_bytes(self, key):
        self.get_bytes_calls.append(key)
        if key not in self.objects:
            raise FileNotFoundError(key)
        return self.objects[key]

    def exists(self, key):
        return key in self.objects

    def delete(self, key):
        self.objects.pop(key, None)


@pytest.fixture(autouse=True)
def _clean_backend(monkeypatch):
    storage_backend.reset_storage_backend()
    monkeypatch.setattr(storage_backend, "_URI_READ_BACKEND", None)
    yield
    storage_backend.reset_storage_backend()
    monkeypatch.setattr(storage_backend, "_URI_READ_BACKEND", None)


def _use_fake(monkeypatch, fake: FakeS3Backend):
    monkeypatch.setattr(storage_backend, "_s3_backend_for_uri", lambda: fake)


FIELDS = [{
    "name": "full_name", "type": "text", "label": "Full name",
    "value": "", "options": [], "page": 1, "rect": [10, 10, 90, 24],
    "required": False,
}]


# ---------------------------------------------------------------------------
# Companion-object semantics (s3 rows)
# ---------------------------------------------------------------------------

def test_s3_row_saves_companion_object_not_local_file(monkeypatch, tmp_path):
    fake = FakeS3Backend()
    _use_fake(monkeypatch, fake)
    pdf_uri = f"s3://{BUCKET}/2026/09/09/" + "a" * 32 + ".pdf"

    location = save_field_sidecar(pdf_uri, FIELDS)

    companion_key = "2026/09/09/" + "a" * 32 + ".pdf" + SIDECAR_SUFFIX
    assert location == f"s3://{BUCKET}/{companion_key}"
    assert json.loads(fake.objects[companion_key]) == FIELDS
    assert fake.content_types[companion_key] == "application/json"
    # Nothing was written anywhere on the local filesystem.
    assert list(tmp_path.iterdir()) == []


def test_s3_row_load_reads_companion(monkeypatch):
    fake = FakeS3Backend()
    _use_fake(monkeypatch, fake)
    pdf_uri = f"s3://{BUCKET}/2026/09/09/" + "b" * 32 + ".pdf"
    companion = "2026/09/09/" + "b" * 32 + ".pdf" + SIDECAR_SUFFIX
    fake.objects[companion] = json.dumps(FIELDS).encode("utf-8")

    assert load_field_sidecar(pdf_uri) == FIELDS
    assert fake.get_bytes_calls == [companion]


def test_s3_row_roundtrip_save_then_load(monkeypatch):
    fake = FakeS3Backend()
    _use_fake(monkeypatch, fake)
    pdf_uri = f"s3://{BUCKET}/2026/09/09/" + "c" * 32 + ".pdf"
    save_field_sidecar(pdf_uri, FIELDS)
    assert load_field_sidecar(pdf_uri) == FIELDS


@pytest.mark.parametrize("payload", [b"not json at all", b'{"dict": "not a list"}'])
def test_s3_row_load_corrupt_companion_degrades_to_none(monkeypatch, payload):
    fake = FakeS3Backend()
    _use_fake(monkeypatch, fake)
    pdf_uri = f"s3://{BUCKET}/2026/09/09/" + "d" * 32 + ".pdf"
    fake.objects["2026/09/09/" + "d" * 32 + ".pdf" + SIDECAR_SUFFIX] = payload
    assert load_field_sidecar(pdf_uri) is None


def test_s3_row_load_missing_companion_is_none_without_put(monkeypatch):
    fake = FakeS3Backend()
    _use_fake(monkeypatch, fake)
    pdf_uri = f"s3://{BUCKET}/2026/09/09/" + "e" * 32 + ".pdf"
    assert load_field_sidecar(pdf_uri) is None
    # The read was attempted exactly once (the companion companion key).
    assert fake.get_bytes_calls == ["2026/09/09/" + "e" * 32 + ".pdf" + SIDECAR_SUFFIX]


def test_foreign_bucket_row_degrades_without_byte_fetch(monkeypatch):
    fake = FakeS3Backend()
    _use_fake(monkeypatch, fake)
    foreign = "s3://other-bucket/2026/09/09/" + "f" * 32 + ".pdf"

    assert save_field_sidecar(foreign, FIELDS) == sidecar_path(foreign)
    assert load_field_sidecar(foreign) is None
    # No companion put, no byte fetch: the foreign row is rejected before
    # any storage traffic.
    assert fake.objects == {}
    assert not fake.get_bytes_calls


def test_backend_failure_never_raises_into_caller(monkeypatch):
    fake = FakeS3Backend()

    def _boom(key, fileobj, content_type):
        raise RuntimeError("simulated S3 outage")

    fake.put = _boom
    _use_fake(monkeypatch, fake)
    pdf_uri = f"s3://{BUCKET}/2026/09/09/" + "1" * 32 + ".pdf"
    # Both directions swallow: callers regenerate/degrade as before.
    save_field_sidecar(pdf_uri, FIELDS)
    assert load_field_sidecar(pdf_uri) is None


def test_companion_keys_are_cacheable_by_the_read_cache(tmp_path):
    """The read cache refuses slash-less keys; companions inherit the
    date-sharded prefix, so they flow through put/get_bytes like any
    object (and get cached on Odypezzus)."""
    from src.storage_cache import CachedS3Storage

    wrapper = CachedS3Storage(
        FakeS3Backend(), cache_dir=str(tmp_path / "s3-cache"), max_bytes=1024
    )
    companion = "2026/09/09/" + "2" * 32 + ".pdf" + SIDECAR_SUFFIX
    assert wrapper._cacheable_key(companion) is True
    wrapper.put(companion, io.BytesIO(b"{}"), "application/json")
    assert wrapper.get_bytes(companion) == b"{}"


# ---------------------------------------------------------------------------
# Local rows keep the historical file behavior
# ---------------------------------------------------------------------------

def test_local_row_sidecar_file_unchanged(tmp_path):
    pdf = tmp_path / "2026" / "09" / "09" / ("3" * 32 + ".pdf")
    pdf.parent.mkdir(parents=True)
    pdf.write_bytes(b"%PDF-1.4 local")

    location = save_field_sidecar(str(pdf), FIELDS)
    assert location == str(pdf) + SIDECAR_SUFFIX
    assert json.loads(Path(location).read_text(encoding="utf-8")) == FIELDS
    assert load_field_sidecar(str(pdf)) == FIELDS


def test_local_row_missing_sidecar_is_none(tmp_path):
    pdf = tmp_path / "gone.pdf"
    assert load_field_sidecar(str(pdf)) is None


# ---------------------------------------------------------------------------
# import_pdf route: extraction failure degrades, s3 happy path companions
# ---------------------------------------------------------------------------

class _StubUploadHandler:
    """Just enough handler for the import route: save_upload stores the
    object in the fake backend (so the route's materialization step can
    download it back) and resolve_upload owner-resolves the s3 row."""

    def __init__(self, fake: FakeS3Backend):
        self.fake = fake
        self.saved: list = []

    def _key(self, upload_id) -> str:
        return f"2026/09/09/{upload_id}"

    def save_upload(self, u, client_ip, owner=None):
        self.saved.append(u)
        key = self._key(u.upload_id)
        self.fake.objects[key] = u.content
        self.fake.content_types[key] = "application/pdf"
        return {
            "id": u.upload_id,
            "path": self.fake.uri_for_key(key),
            "mime": "application/pdf",
            "size": len(u.content),
            "name": u.filename,
            "original_name": u.filename,
            "hash": "h" * 8,
            "uploaded_at": "2026-09-09T00:00:00",
        }

    def resolve_upload(self, upload_id, owner=None, auth_manager=None):
        return {
            "id": upload_id,
            "path": self.fake.uri_for_key(self._key(upload_id)),
            "mime": "application/pdf",
            "owner": owner,
        }


class _FakeUpload:
    def __init__(self, content: bytes, filename="form.pdf", upload_id=None):
        self.file = io.BytesIO(content)
        self.filename = filename
        self.content = content
        self.upload_id = upload_id or (uuid.uuid4().hex + ".pdf")


def _req():
    return SimpleNamespace(
        state=SimpleNamespace(current_user="tester"),
        app=SimpleNamespace(state=SimpleNamespace(auth_manager=None)),
        client=SimpleNamespace(host="127.0.0.1"),
    )


def _import_endpoint(upload_handler):
    import routes.document_routes as droutes

    router = droutes.setup_document_routes(MagicMock(), upload_handler)
    for r in router.routes:
        if getattr(r, "path", None) == "/api/documents/import-pdf":
            return r.endpoint
    raise RuntimeError("import-pdf endpoint not found")


@pytest.fixture
def doc_db(monkeypatch):
    import tempfile

    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import NullPool

    import core.database as cdb
    import routes.document_routes  # noqa: F401 — monkeypatch needs it loaded
    import src.database as sdb

    tmpdb = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmpdb.close()
    engine = create_engine(
        f"sqlite:///{tmpdb.name}",
        connect_args={"check_same_thread": False},
        poolclass=NullPool,
    )
    cdb.Base.metadata.create_all(engine)
    ts = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    monkeypatch.setattr("routes.document_routes.SessionLocal", ts)  # imported above
    monkeypatch.setattr("src.database.SessionLocal", ts)
    yield ts
    engine.dispose()
    try:
        os.unlink(tmpdb.name)
    except OSError:
        pass


def _patch_pdf_internals(monkeypatch, *, is_form=True, fields=None, extract_raises=False):
    # Import the patch targets first: monkeypatch string paths resolve via
    # getattr and never import submodules, so patching
    # src.agent_tools.document_tools in a fresh process would AttributeError.
    import src.agent_tools.document_tools as document_tools
    import src.document_processor as document_processor
    import src.pdf_forms as pdf_forms

    monkeypatch.setattr(pdf_forms, "has_form_fields", lambda path: is_form)
    if extract_raises:
        def _explode(path):
            raise RuntimeError("fitz exploded on a weird widget tree")
        monkeypatch.setattr(pdf_forms, "extract_fields", _explode)
    else:
        monkeypatch.setattr(pdf_forms, "extract_fields", lambda path: fields or FIELDS)
    monkeypatch.setattr(
        document_processor, "_process_pdf", lambda p, owner=None: "pdf body text"
    )
    monkeypatch.setattr(
        document_tools, "set_active_document", lambda *a, **k: None
    )


async def test_import_pdf_s3_form_writes_companion_sidecar(monkeypatch, doc_db):
    fake = FakeS3Backend()
    _use_fake(monkeypatch, fake)
    _patch_pdf_internals(monkeypatch)

    upload = _FakeUpload(b"%PDF-1.4 form bytes")
    handler = _StubUploadHandler(fake)
    # The materialized copy the route works on is a temp file whose sidecar
    # would historically be orphaned.
    result = await _import_endpoint(handler)(_req(), upload, None)

    assert result["title"] == "form"
    companion = "2026/09/09/" + upload.upload_id + SIDECAR_SUFFIX
    assert json.loads(fake.objects[companion]) == FIELDS
    # The created doc is the FORM flavour (front-matter marker), so the
    # companion will actually be consulted on export.
    db = doc_db()
    try:
        from core.database import Document

        doc = db.query(Document).filter(Document.id == result["id"]).first()
        assert 'pdf_form_source upload_id="' + upload.upload_id in doc.current_content
    finally:
        db.close()


async def test_import_pdf_extraction_failure_degrades_to_plain_doc(monkeypatch, doc_db):
    """(a) follow-up: an unexpected extraction error must not escape the
    route — the import completes as a plain PDF document."""
    fake = FakeS3Backend()
    _use_fake(monkeypatch, fake)
    _patch_pdf_internals(monkeypatch, is_form=True, extract_raises=True)

    upload = _FakeUpload(b"%PDF-1.4 hostile form")
    handler = _StubUploadHandler(fake)
    result = await _import_endpoint(handler)(_req(), upload, None)

    assert result["title"] == "form"
    db = doc_db()
    try:
        from core.database import Document

        doc = db.query(Document).filter(Document.id == result["id"]).first()
        # Plain-PDF marker (not the form one): no sidecar is needed.
        assert 'pdf_source upload_id="' + upload.upload_id in doc.current_content
        assert "pdf_form_source" not in doc.current_content
    finally:
        db.close()
    # No companion was stored for the failed extraction.
    assert not any(k.endswith(SIDECAR_SUFFIX) for k in fake.objects)
