"""routes/upload_routes.py download_file x S3 backend: owner gating, byte
streaming via the backend, Content-Disposition/filename semantics, and the
local-ephemeral thumbnail cache generated from backend bytes on first hit."""
import asyncio
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.responses import FileResponse, StreamingResponse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import storage_backend
from src.storage_backend import S3Backend
from src.upload_handler import UploadHandler

BUCKET = "odysseus-test"


class _FakeS3Backend(S3Backend):
    def __init__(self, bucket=BUCKET):
        super().__init__(bucket=bucket)
        self.objects: dict[str, bytes] = {}
        self.get_bytes_calls: list[str] = []

    def put(self, key, fileobj, content_type):
        self.objects[key] = fileobj.read()

    def get_bytes(self, key):
        self.get_bytes_calls.append(key)
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


class _AuthManager:
    is_configured = True

    def __init__(self, admins=()):
        self._admins = set(admins)

    def is_admin(self, user):
        return user in self._admins


class _Request:
    def __init__(self, user=None, auth_manager=None, body=None):
        self.state = SimpleNamespace(current_user=user)
        self.app = SimpleNamespace(state=SimpleNamespace(auth_manager=auth_manager))
        self.client = SimpleNamespace(host="127.0.0.1")
        self._body = body

    async def json(self):
        return self._body


def _png_bytes() -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (48, 24), (20, 140, 220)).save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture(autouse=True)
def _clean_backend(monkeypatch):
    storage_backend.reset_storage_backend()
    monkeypatch.delenv(storage_backend.ENV_STORAGE_BACKEND, raising=False)
    yield
    storage_backend.reset_storage_backend()


def _make_s3_store(tmp_path, monkeypatch):
    import fastapi.dependencies.utils as dependency_utils
    from routes.upload_routes import router, setup_upload_routes

    monkeypatch.setattr(dependency_utils, "ensure_multipart_is_installed", lambda: None)

    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir(parents=True)
    handler = UploadHandler(str(tmp_path), str(upload_dir))

    fake = _FakeS3Backend()
    monkeypatch.setattr(storage_backend, "get_storage_backend", lambda: fake)

    before = len(router.routes)
    setup_upload_routes(handler)
    endpoints = {route.endpoint.__name__: route.endpoint for route in router.routes[before:]}
    return handler, fake, endpoints, upload_dir


def _add_s3_row(handler, fake, *, file_id, data, owner, name="pic.png", mime="image/png"):
    key = f"2026/09/08/{file_id}"
    fake.objects[key] = data
    row = {
        "id": file_id,
        "path": fake.uri_for_key(key),
        "mime": mime,
        "size": len(data),
        "name": name,
        "original_name": name,
        "hash": f"hash-{file_id[:6]}",
        "checksum_sha256": f"hash-{file_id[:6]}",
        "owner": owner,
        "uploaded_at": "2026-09-08T00:00:00",
        "created_at": "2026-09-08T00:00:00",
        "last_accessed": "2026-09-08T00:00:00",
    }
    db = Path(handler.upload_dir) / "uploads.json"
    index = json.loads(db.read_text()) if db.exists() else {}
    index[f"{owner}:{row['hash']}"] = row
    db.write_text(json.dumps(index), encoding="utf-8")
    return row


async def _drain(response):
    chunks = []
    async for chunk in response.body_iterator:
        chunks.append(chunk if isinstance(chunk, bytes) else chunk.encode())
    return b"".join(chunks)


# ---------------------------------------------------------------------------
# Streaming + headers
# ---------------------------------------------------------------------------

def test_download_file_streams_s3_row_bytes(tmp_path, monkeypatch):
    handler, fake, endpoints, _dir = _make_s3_store(tmp_path, monkeypatch)
    png = _png_bytes()
    alice_id = "a" * 32 + ".png"
    _add_s3_row(handler, fake, file_id=alice_id, data=png, owner="alice", name="holiday-pic.png")

    response = asyncio.run(
        endpoints["download_file"](_Request(user="alice", auth_manager=_AuthManager()), alice_id)
    )

    assert isinstance(response, StreamingResponse)
    assert response.media_type == "image/png"
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["Content-Disposition"] == 'attachment; filename="holiday-pic.png"'
    body = asyncio.run(_drain(response))
    assert body == png


def test_download_file_s3_ascii_safe_filename_disposition(tmp_path, monkeypatch):
    handler, fake, endpoints, _dir = _make_s3_store(tmp_path, monkeypatch)
    file_id = "b" * 32 + ".txt"
    _add_s3_row(
        handler, fake, file_id=file_id, data=b"data", owner="alice",
        name="naïve—name.txt", mime="text/plain",
    )
    response = asyncio.run(
        endpoints["download_file"](_Request(user="alice", auth_manager=_AuthManager()), file_id)
    )
    # Non-ascii filenames use the RFC 5987 form, matching FileResponse.
    assert response.headers["Content-Disposition"].startswith("attachment; filename*=utf-8''")


def test_download_file_s3_owner_check_404_for_other_user(tmp_path, monkeypatch):
    handler, fake, endpoints, _dir = _make_s3_store(tmp_path, monkeypatch)
    bob_id = "c" * 32 + ".png"
    _add_s3_row(handler, fake, file_id=bob_id, data=_png_bytes(), owner="bob")

    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            endpoints["download_file"](_Request(user="alice", auth_manager=_AuthManager()), bob_id)
        )
    assert exc.value.status_code == 404


def test_download_file_s3_denies_anonymous_when_auth_configured(tmp_path, monkeypatch):
    handler, fake, endpoints, _dir = _make_s3_store(tmp_path, monkeypatch)
    alice_id = "d" * 32 + ".png"
    _add_s3_row(handler, fake, file_id=alice_id, data=_png_bytes(), owner="alice")

    with pytest.raises(HTTPException) as exc:
        asyncio.run(endpoints["download_file"](_Request(auth_manager=_AuthManager()), alice_id))
    assert exc.value.status_code == 403


def test_download_file_s3_admin_cross_owner_allowed(tmp_path, monkeypatch):
    handler, fake, endpoints, _dir = _make_s3_store(tmp_path, monkeypatch)
    bob_id = "e" * 32 + ".png"
    _add_s3_row(handler, fake, file_id=bob_id, data=_png_bytes(), owner="bob")

    response = asyncio.run(
        endpoints["download_file"](
            _Request(user="admin", auth_manager=_AuthManager(admins={"admin"})), bob_id
        )
    )
    assert isinstance(response, StreamingResponse)


def test_download_file_s3_missing_object_404(tmp_path, monkeypatch):
    handler, fake, endpoints, _dir = _make_s3_store(tmp_path, monkeypatch)
    alice_id = "1a" * 16 + ".png"
    _add_s3_row(handler, fake, file_id=alice_id, data=_png_bytes(), owner="alice")
    fake.objects.clear()

    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            endpoints["download_file"](_Request(user="alice", auth_manager=_AuthManager()), alice_id)
        )
    assert exc.value.status_code == 404


def test_download_file_s3_foreign_bucket_row_404(tmp_path, monkeypatch):
    """A row pointing at a bucket other than the configured one must 404,
    not stream bytes from (or even probe) an unmanaged bucket."""
    handler, fake, endpoints, _dir = _make_s3_store(tmp_path, monkeypatch)
    alice_id = "2e" * 16 + ".png"
    row = _add_s3_row(handler, fake, file_id=alice_id, data=_png_bytes(), owner="alice")
    # Rewrite the row's path to a foreign bucket (same key shape).
    row["path"] = "s3://someone-else/2026/09/08/" + alice_id
    db = Path(handler.upload_dir) / "uploads.json"
    index = json.loads(db.read_text(encoding="utf-8"))
    index[f"alice:hash-{alice_id[:6]}"]["path"] = row["path"]
    db.write_text(json.dumps(index), encoding="utf-8")

    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            endpoints["download_file"](_Request(user="alice", auth_manager=_AuthManager()), alice_id)
        )
    assert exc.value.status_code == 404


# ---------------------------------------------------------------------------
# Thumbnail cache (local-ephemeral, built from backend bytes)
# ---------------------------------------------------------------------------

def test_download_file_s3_thumb_cached_locally_from_backend_bytes(tmp_path, monkeypatch):
    handler, fake, endpoints, upload_dir = _make_s3_store(tmp_path, monkeypatch)
    alice_id = "2b" * 16 + ".png"
    _add_s3_row(handler, fake, file_id=alice_id, data=_png_bytes(), owner="alice")

    response = asyncio.run(
        endpoints["download_file"](
            _Request(user="alice", auth_manager=_AuthManager()), alice_id, thumb=1
        )
    )

    assert isinstance(response, FileResponse)
    assert response.media_type == "image/jpeg"
    assert response.path.endswith(alice_id + ".jpg")
    thumb_path = Path(upload_dir) / ".thumbs" / (alice_id + ".jpg")
    assert thumb_path.is_file()
    # Generated from backend bytes, not from any local upload file.
    key = f"2026/09/08/{alice_id}"
    assert fake.get_bytes_calls, "thumbnail must be generated from backend.get_bytes"
    assert key in fake.get_bytes_calls
    assert not (Path(upload_dir) / "2026").exists(), "no local copy of the object may appear"

    # Second hit is served from the cache without re-reading the object.
    calls_before = len(fake.get_bytes_calls)
    again = asyncio.run(
        endpoints["download_file"](
            _Request(user="alice", auth_manager=_AuthManager()), alice_id, thumb=1
        )
    )
    assert isinstance(again, FileResponse)
    assert len(fake.get_bytes_calls) == calls_before


def test_download_file_s3_thumb_owner_gate_before_generation(tmp_path, monkeypatch):
    handler, fake, endpoints, upload_dir = _make_s3_store(tmp_path, monkeypatch)
    bob_id = "3c" * 16 + ".png"
    _add_s3_row(handler, fake, file_id=bob_id, data=_png_bytes(), owner="bob")

    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            endpoints["download_file"](
                _Request(user="alice", auth_manager=_AuthManager()), bob_id, thumb=1
            )
        )
    assert exc.value.status_code == 404
    assert fake.get_bytes_calls == [], "owner gate must run before fetching bytes"
    assert not (Path(upload_dir) / ".thumbs").exists() or \
        not list((Path(upload_dir) / ".thumbs").iterdir())


# ---------------------------------------------------------------------------
# Local rows keep their historical FileResponse behaviour
# ---------------------------------------------------------------------------

def test_download_file_local_row_still_uses_file_response(tmp_path, monkeypatch):
    handler, fake, endpoints, upload_dir = _make_s3_store(tmp_path, monkeypatch)
    local_id = "4d" * 16 + ".png"
    local_file = Path(upload_dir) / "2026" / "09" / "08" / local_id
    local_file.parent.mkdir(parents=True)
    local_file.write_bytes(_png_bytes())
    db = Path(upload_dir) / "uploads.json"
    index = json.loads(db.read_text()) if db.exists() else {}
    index["alice:hlocal"] = {
        "id": local_id, "path": str(local_file), "mime": "image/png",
        "size": local_file.stat().st_size, "name": "local.png",
        "original_name": "local.png", "hash": "hlocal",
        "checksum_sha256": "hlocal", "owner": "alice",
        "uploaded_at": "2026-09-08T00:00:00",
        "created_at": "2026-09-08T00:00:00",
        "last_accessed": "2026-09-08T00:00:00",
    }
    db.write_text(json.dumps(index), encoding="utf-8")

    response = asyncio.run(
        endpoints["download_file"](_Request(user="alice", auth_manager=_AuthManager()), local_id)
    )
    assert isinstance(response, FileResponse)
    assert response.path.endswith(local_id)
    assert response.media_type == "image/png"
    assert fake.objects == {}
