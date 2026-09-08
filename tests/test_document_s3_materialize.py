"""W1 regression guard: the document PDF routes must keep working for
object-stored (s3://) upload rows via _materialized_upload, which downloads
the object to a temp file (suffix preserved) and removes it after the route
finishes. Local rows keep their resolved path with zero new I/O."""
import asyncio
import json
import sys
import tempfile
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import core.database as cdb
from core.database import Document
from src import storage_backend
from src.storage_backend import S3Backend
from src.upload_handler import UploadHandler

BUCKET = "odysseus-test"
PDF_BYTES = b"%PDF-1.4 fake pdf body for materialize tests"


class _FakeS3Backend(S3Backend):
    def __init__(self, bucket=BUCKET):
        super().__init__(bucket=bucket)
        self.objects: dict[str, bytes] = {}

    def put(self, key, fileobj, content_type):
        self.objects[key] = fileobj.read()

    def get_bytes(self, key):
        if key not in self.objects:
            raise FileNotFoundError(key)
        return self.objects[key]

    def get_stream(self, key):
        return iter([self.get_bytes(key)])

    def exists(self, key):
        return key in self.objects

    def delete(self, key):
        self.objects.pop(key, None)

    def stat(self, key):
        if key not in self.objects:
            return None
        return {"size": len(self.objects[key]), "content_type": None, "last_modified": None}

    def list_keys(self):
        return iter(sorted(self.objects))


@pytest.fixture(autouse=True)
def _clean_backend(monkeypatch):
    storage_backend.reset_storage_backend()
    monkeypatch.delenv(storage_backend.ENV_STORAGE_BACKEND, raising=False)
    yield
    storage_backend.reset_storage_backend()


def _make_handler(tmp_path: Path) -> UploadHandler:
    base = tmp_path / "base"
    upload = tmp_path / "uploads"
    base.mkdir()
    upload.mkdir()
    return UploadHandler(str(base), str(upload))


def _seed_s3_row(handler: UploadHandler, fake: _FakeS3Backend, file_id: str,
                 data: bytes, owner: str = "tester") -> str:
    key = f"2026/09/08/{file_id}"
    fake.objects[key] = data
    uri = fake.uri_for_key(key)
    db = Path(handler.upload_dir) / "uploads.json"
    index = json.loads(db.read_text()) if db.exists() else {}
    index[f"{owner}:hash-{file_id[:6]}"] = {
        "id": file_id,
        "path": uri,
        "mime": "application/pdf",
        "size": len(data),
        "name": file_id,
        "original_name": "source.pdf",
        "hash": f"hash-{file_id[:6]}",
        "checksum_sha256": f"hash-{file_id[:6]}",
        "owner": owner,
        "uploaded_at": "2026-09-08T00:00:00",
        "created_at": "2026-09-08T00:00:00",
        "last_accessed": "2026-09-08T00:00:00",
    }
    db.write_text(json.dumps(index), encoding="utf-8")
    return uri


def _seed_local_row(handler: UploadHandler, file_id: str, data: bytes,
                    owner: str = "tester") -> str:
    local_file = Path(handler.upload_dir) / "2026" / "09" / "08" / file_id
    local_file.parent.mkdir(parents=True)
    local_file.write_bytes(data)
    db = Path(handler.upload_dir) / "uploads.json"
    index = json.loads(db.read_text()) if db.exists() else {}
    index[f"{owner}:hash-local"] = {
        "id": file_id,
        "path": str(local_file),
        "mime": "application/pdf",
        "size": len(data),
        "name": file_id,
        "original_name": "source.pdf",
        "hash": "hash-local",
        "checksum_sha256": "hash-local",
        "owner": owner,
        "uploaded_at": "2026-09-08T00:00:00",
        "created_at": "2026-09-08T00:00:00",
        "last_accessed": "2026-09-08T00:00:00",
    }
    db.write_text(json.dumps(index), encoding="utf-8")
    return str(local_file)


def _request(user="tester"):
    return SimpleNamespace(
        state=SimpleNamespace(current_user=user),
        app=SimpleNamespace(state=SimpleNamespace(auth_manager=None)),
    )


# ---------------------------------------------------------------------------
# _materialized_upload context manager
# ---------------------------------------------------------------------------

def test_materialized_upload_yields_local_path_unchanged(tmp_path, monkeypatch):
    from routes.document_routes import _materialized_upload

    handler = _make_handler(tmp_path)
    file_id = "a" * 32 + ".pdf"
    local_path = _seed_local_row(handler, file_id, PDF_BYTES)

    with _materialized_upload(handler, _request(), file_id, "tester") as path:
        assert path == local_path
    # Local rows are never touched or removed.
    assert Path(local_path).exists()


def test_materialized_upload_downloads_s3_row_to_temp_and_cleans_up(tmp_path, monkeypatch):
    from routes.document_routes import _materialized_upload

    handler = _make_handler(tmp_path)
    fake = _FakeS3Backend()
    monkeypatch.setattr(storage_backend, "get_storage_backend", lambda: fake)
    file_id = "b" * 32 + ".pdf"
    uri = _seed_s3_row(handler, fake, file_id, PDF_BYTES)

    with _materialized_upload(handler, _request(), file_id, "tester") as path:
        assert path is not None
        assert path != uri
        assert not path.startswith("s3://")
        # Suffix preserved so fitz/pypdf format sniffing keeps working.
        assert path.endswith(".pdf")
        # The helpers receive the real object bytes.
        assert Path(path).read_bytes() == PDF_BYTES
        tmp_inside = path
    # Temp file removed after the block, even on normal exit.
    assert not Path(tmp_inside).exists()


def test_materialized_upload_cleans_up_temp_on_exception(tmp_path, monkeypatch):
    from routes.document_routes import _materialized_upload

    handler = _make_handler(tmp_path)
    fake = _FakeS3Backend()
    monkeypatch.setattr(storage_backend, "get_storage_backend", lambda: fake)
    file_id = "c" * 32 + ".pdf"
    _seed_s3_row(handler, fake, file_id, PDF_BYTES)

    seen: dict[str, str] = {}
    with pytest.raises(RuntimeError, match="boom"):
        with _materialized_upload(handler, _request(), file_id, "tester") as path:
            seen["path"] = path
            raise RuntimeError("boom")
    assert not Path(seen["path"]).exists()


def test_materialized_upload_yields_none_for_unresolvable_upload(tmp_path):
    from routes.document_routes import _materialized_upload

    handler = _make_handler(tmp_path)
    with _materialized_upload(handler, _request(), "d" * 32 + ".pdf", "tester") as path:
        assert path is None


def test_materialized_upload_rejects_cross_owner_s3_row(tmp_path, monkeypatch):
    from routes.document_routes import _materialized_upload

    handler = _make_handler(tmp_path)
    fake = _FakeS3Backend()
    monkeypatch.setattr(storage_backend, "get_storage_backend", lambda: fake)
    file_id = "e" * 32 + ".pdf"
    _seed_s3_row(handler, fake, file_id, PDF_BYTES, owner="someone-else")

    with _materialized_upload(handler, _request(user="tester"), file_id, "tester") as path:
        assert path is None


# ---------------------------------------------------------------------------
# Route-level: POST /api/document/{doc_id}/extract-pdf-text with an s3 row
# ---------------------------------------------------------------------------

@pytest.fixture
def test_db(monkeypatch):
    tmpdb = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmpdb.close()
    engine = create_engine(
        f"sqlite:///{tmpdb.name}",
        connect_args={"check_same_thread": False},
        poolclass=NullPool,
    )
    cdb.Base.metadata.create_all(engine)
    ts = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    import routes.document_routes as droutes
    monkeypatch.setattr(droutes, "SessionLocal", ts)
    try:
        yield ts
    finally:
        engine.dispose()
        try:
            os.unlink(tmpdb.name)
        except OSError:
            pass


import os  # noqa: E402  (used by the fixture above)


def _endpoint(router, method: str, path: str):
    for r in router.routes:
        if getattr(r, "path", None) == path and method in getattr(r, "methods", set()):
            return r.endpoint
    raise RuntimeError(f"{method} {path} not found")


def _make_pdf_doc(db_session, upload_id: str) -> str:
    content = (
        f'<!-- pdf_form_source upload_id="{upload_id}" fields="3" -->\n'
        "# Original Title\n\n- Field 1: value1\n"
    )
    db = db_session()
    try:
        doc = Document(
            id=str(uuid.uuid4()),
            session_id=None,
            title="t",
            language="markdown",
            current_content=content,
            version_count=1,
            is_active=True,
            owner="tester",
        )
        db.add(doc)
        db.commit()
        return doc.id
    finally:
        db.close()


def test_extract_pdf_text_materializes_s3_source(tmp_path, monkeypatch, test_db):
    import routes.document_routes as droutes
    import src.document_processor as dproc

    handler = _make_handler(tmp_path)
    fake = _FakeS3Backend()
    monkeypatch.setattr(storage_backend, "get_storage_backend", lambda: fake)
    file_id = "f" * 32 + ".pdf"
    _seed_s3_row(handler, fake, file_id, PDF_BYTES)

    captured: dict[str, object] = {}

    def _fake_process_pdf(path, owner=None):
        captured["path"] = path
        captured["content"] = Path(path).read_bytes()
        return "\n\n[PDF content]:\n\nre-extracted body text"

    monkeypatch.setattr(dproc, "_process_pdf", _fake_process_pdf)

    router = droutes.setup_document_routes(MagicMock(), handler)
    extract = _endpoint(router, "POST", "/api/document/{doc_id}/extract-pdf-text")
    doc_id = _make_pdf_doc(test_db, file_id)

    result = asyncio.run(extract(_request(), doc_id))

    assert result["extracted"] is True
    # The route fed a real local file holding the object bytes to the
    # extractor — not the s3:// URI.
    assert not str(captured["path"]).startswith("s3://")
    assert captured["content"] == PDF_BYTES
    # Materialized temp file is gone once the route finished.
    assert not Path(str(captured["path"])).exists()

    db = test_db()
    try:
        doc = db.query(Document).filter(Document.id == doc_id).first()
        assert "re-extracted body text" in (doc.current_content or "")
    finally:
        db.close()
