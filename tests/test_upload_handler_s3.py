"""UploadHandler x S3 backend: save -> dedupe -> resolve roundtrip with a fake
object store (no real MinIO, no moto — a simple in-memory S3Backend subclass),
uploads.json "path" value shapes, and local/s3 row coexistence (rollback safety).
"""
import hashlib
import io
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import storage_backend
from src.storage_backend import S3Backend, is_s3_uri
from src.upload_handler import UploadHandler

BUCKET = "odysseus-test"
S3_PATH_RE = re.compile(rf"^s3://{BUCKET}/\d{{4}}/\d{{2}}/\d{{2}}/[0-9a-f]{{32}}(?:\.[A-Za-z0-9]+)?$")


class FakeS3Backend(S3Backend):
    """In-memory object store: real URI/key semantics, no network, no boto3."""

    def __init__(self, bucket=BUCKET):
        super().__init__(bucket=bucket)
        self.objects: dict[str, bytes] = {}
        self.content_types: dict[str, str] = {}
        self.puts: list[str] = []

    def put(self, key, fileobj, content_type):
        self.objects[key] = fileobj.read()
        self.content_types[key] = content_type
        self.puts.append(key)

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
        return {
            "size": len(self.objects[key]),
            "content_type": self.content_types.get(key),
            "last_modified": None,
        }

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
    return UploadHandler(base_dir=str(base), upload_dir=str(upload))


def _use_fake_backend(monkeypatch, fake: FakeS3Backend):
    monkeypatch.setattr(storage_backend, "get_storage_backend", lambda: fake)


def _fake_upload(content: bytes, filename: str = "photo.png"):
    return type("U", (), {"filename": filename, "file": io.BytesIO(content)})()


def _index(handler: UploadHandler) -> dict:
    db = Path(handler.upload_dir) / "uploads.json"
    if not db.exists():
        return {}
    return json.loads(db.read_text(encoding="utf-8"))


def _png_bytes() -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (32, 16), (10, 120, 200)).save(buf, "PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# save_upload through the s3 backend
# ---------------------------------------------------------------------------

def test_save_upload_local_default_path_shape_regression_lock(tmp_path, monkeypatch):
    """Default env (no ODYSSEUS_STORAGE_BACKEND): save_upload must keep
    writing the historical local layout — absolute path under
    upload_dir/YYYY/MM/DD/<id> — with no s3 URI in sight."""
    assert storage_backend.get_storage_backend().is_s3 is False

    handler = _make_handler(tmp_path)
    content = b"plain local world"
    meta = handler.save_upload(_fake_upload(content, "note.txt"), "127.0.0.1", owner="alice")

    expected_dir = Path(handler.upload_dir)
    # Exact historical shape: <upload_dir>/<YYYY>/<MM>/<DD>/<id>
    rel = Path(meta["path"]).relative_to(expected_dir)
    assert len(rel.parts) == 4
    assert re.fullmatch(r"\d{4}", rel.parts[0])
    assert re.fullmatch(r"\d{2}", rel.parts[1])
    assert re.fullmatch(r"\d{2}", rel.parts[2])
    assert rel.parts[3] == meta["id"]
    assert not is_s3_uri(meta["path"])
    assert Path(meta["path"]).read_bytes() == content


def test_save_upload_writes_s3_uri_path_with_date_sharded_key(tmp_path, monkeypatch):
    handler = _make_handler(tmp_path)
    fake = FakeS3Backend()
    _use_fake_backend(monkeypatch, fake)

    meta = handler.save_upload(_fake_upload(b"hello s3"), "127.0.0.1", owner="alice")

    assert S3_PATH_RE.match(meta["path"]), meta["path"]
    key = fake.key_for_uri(meta["path"])
    assert key is not None
    # The object exists and holds exactly the uploaded bytes.
    assert fake.exists(key)
    assert fake.get_bytes(key) == b"hello s3"
    # The uploads.json row carries the same URI form.
    rows = _index(handler)
    (row,) = rows.values()
    assert row["path"] == meta["path"]
    assert row["id"] == meta["id"]
    # No file was written into the local date-sharded tree.
    assert not any(p.match("????/??/??") for p in Path(handler.upload_dir).rglob("*") if p.is_file())


def test_save_upload_image_dimensions_read_via_backend(tmp_path, monkeypatch):
    handler = _make_handler(tmp_path)
    fake = FakeS3Backend()
    _use_fake_backend(monkeypatch, fake)

    meta = handler.save_upload(_fake_upload(_png_bytes()), "127.0.0.1", owner="alice")

    assert meta["width"] == 32
    assert meta["height"] == 16


def test_save_upload_dedupe_roundtrip_same_owner(tmp_path, monkeypatch):
    handler = _make_handler(tmp_path)
    fake = FakeS3Backend()
    _use_fake_backend(monkeypatch, fake)
    content = b"duplicate me"

    first = handler.save_upload(_fake_upload(content, "a.txt"), "127.0.0.1", owner="alice")
    second = handler.save_upload(_fake_upload(content, "renamed.txt"), "127.0.0.1", owner="alice")

    assert second["is_duplicate"] is True
    assert second["id"] == first["id"]
    assert second["path"] == first["path"]
    assert len(fake.puts) == 1, "dedupe must not write a second object"


def test_save_upload_no_dedupe_across_owners(tmp_path, monkeypatch):
    handler = _make_handler(tmp_path)
    fake = FakeS3Backend()
    _use_fake_backend(monkeypatch, fake)
    content = b"shared bytes"

    a = handler.save_upload(_fake_upload(content), "127.0.0.1", owner="alice")
    b = handler.save_upload(_fake_upload(content), "127.0.0.1", owner="bob")

    assert a["id"] != b["id"]
    assert len(fake.objects) == 2


# ---------------------------------------------------------------------------
# reserve/resolve for s3 rows
# ---------------------------------------------------------------------------

def test_resolve_upload_roundtrip_for_s3_row(tmp_path, monkeypatch):
    handler = _make_handler(tmp_path)
    fake = FakeS3Backend()
    _use_fake_backend(monkeypatch, fake)

    meta = handler.save_upload(_fake_upload(b"resolve me"), "127.0.0.1", owner="alice")
    resolved = handler.resolve_upload(meta["id"], owner="alice")

    assert resolved is not None
    assert resolved["path"] == meta["path"]
    assert resolved["name"] == meta["name"]
    assert resolved["owner"] == "alice"


def test_resolve_upload_rejects_wrong_owner_for_s3_row(tmp_path, monkeypatch):
    handler = _make_handler(tmp_path)
    _use_fake_backend(monkeypatch, FakeS3Backend())

    meta = handler.save_upload(_fake_upload(b"alice only"), "127.0.0.1", owner="alice")
    assert handler.resolve_upload(meta["id"], owner="bob") is None


def test_resolve_upload_s3_row_with_missing_object_fails_closed(tmp_path, monkeypatch):
    handler = _make_handler(tmp_path)
    fake = FakeS3Backend()
    _use_fake_backend(monkeypatch, fake)

    meta = handler.save_upload(_fake_upload(b"doomed"), "127.0.0.1", owner="alice")
    fake.delete(fake.key_for_uri(meta["path"]))

    # No stale-path repair for s3 rows: the object is gone, resolution fails.
    assert handler.resolve_upload(meta["id"], owner="alice") is None


def test_reserve_upload_s3_row_in_foreign_bucket_is_rejected(tmp_path, monkeypatch):
    handler = _make_handler(tmp_path)
    _use_fake_backend(monkeypatch, FakeS3Backend())
    file_id = "a" * 32 + ".png"
    _seed_index_row(handler, {
        "id": file_id,
        "path": f"s3://someone-else/2026/01/01/{file_id}",
        "mime": "image/png",
        "size": 1,
        "name": file_id,
        "hash": "h",
        "checksum_sha256": "h",
        "owner": "alice",
        "uploaded_at": "2026-01-01T00:00:00",
        "created_at": "2026-01-01T00:00:00",
        "last_accessed": "2026-01-01T00:00:00",
    })
    assert handler.reserve_upload(file_id, owner="alice") is None


# ---------------------------------------------------------------------------
# Coexistence: local rows and s3 rows in one index
# ---------------------------------------------------------------------------

def _seed_index_row(handler: UploadHandler, row: dict) -> None:
    db = Path(handler.upload_dir) / "uploads.json"
    index = json.loads(db.read_text()) if db.exists() else {}
    index[f"{row.get('owner')}:{row['hash']}"] = row
    db.write_text(json.dumps(index), encoding="utf-8")


def test_local_row_and_s3_row_coexist_in_one_index(tmp_path, monkeypatch):
    handler = _make_handler(tmp_path)
    fake = FakeS3Backend()
    _use_fake_backend(monkeypatch, fake)

    # A pre-existing local row with real bytes on disk.
    local_id = "b" * 32 + ".txt"
    local_file = Path(handler.upload_dir) / "2026" / "01" / "02" / local_id
    local_file.parent.mkdir(parents=True)
    local_file.write_bytes(b"legacy local bytes")
    _seed_index_row(handler, {
        "id": local_id,
        "path": str(local_file),
        "mime": "text/plain",
        "size": 18,
        "name": local_id,
        "hash": hashlib.sha256(b"legacy local bytes").hexdigest(),
        "checksum_sha256": hashlib.sha256(b"legacy local bytes").hexdigest(),
        "original_name": "legacy.txt",
        "owner": "alice",
        "uploaded_at": "2026-01-02T00:00:00",
        "created_at": "2026-01-02T00:00:00",
        "last_accessed": "2026-01-02T00:00:00",
    })

    # A fresh s3 row via save_upload.
    s3_meta = handler.save_upload(_fake_upload(b"new object bytes"), "127.0.0.1", owner="alice")

    index = _index(handler)
    assert len(index) == 2
    assert {str(local_file), s3_meta["path"]} == {row["path"] for row in index.values()}

    # Both rows resolve under the s3 backend.
    assert handler.resolve_upload(local_id, owner="alice")["path"] == str(local_file)
    assert handler.resolve_upload(s3_meta["id"], owner="alice")["path"] == s3_meta["path"]
    # Both rows are readable through the storage read API.
    assert storage_backend.read_attachment_bytes({"path": str(local_file)}) == b"legacy local bytes"
    assert storage_backend.read_attachment_bytes({"path": s3_meta["path"]}) == b"new object bytes"


def test_local_backend_never_prunes_s3_rows_as_stale(tmp_path, monkeypatch):
    """Rollback safety: after flipping the env back to local, re-uploading the
    same content must NOT sweep the s3 row out of uploads.json as a 'stale
    duplicate' — its bytes may still live in the object store."""
    handler = _make_handler(tmp_path)
    # Active backend: local (the default, unpatched factory).
    assert not storage_backend.get_storage_backend().is_s3

    content = b"byte-identical"
    file_hash = hashlib.sha256(content).hexdigest()
    s3_id = "c" * 32 + ".bin"
    _seed_index_row(handler, {
        "id": s3_id,
        "path": f"s3://{BUCKET}/2026/01/01/{s3_id}",
        "mime": "application/octet-stream",
        "size": len(content),
        "name": s3_id,
        "hash": file_hash,
        "checksum_sha256": file_hash,
        "original_name": "from-s3-era.bin",
        "owner": "alice",
        "uploaded_at": "2026-01-01T00:00:00",
        "created_at": "2026-01-01T00:00:00",
        "last_accessed": "2026-01-01T00:00:00",
    })

    meta = handler.save_upload(_fake_upload(content, "again.bin"), "127.0.0.1", owner="alice")
    assert meta.get("is_duplicate") is not True

    index = _index(handler)
    paths = {row["path"] for row in index.values()}
    assert f"s3://{BUCKET}/2026/01/01/{s3_id}" in paths, "s3 row must survive the local-backend dedupe sweep"
    assert os.path.isfile(meta["path"])


def test_cleanup_s3_enumerates_and_deletes_unreferenced_old_objects(tmp_path, monkeypatch):
    handler = _make_handler(tmp_path)
    fake = FakeS3Backend()
    _use_fake_backend(monkeypatch, fake)
    handler.cleanup_days = 30

    # An old, unreferenced object (date shard far in the past).
    old_id = "d" * 32 + ".txt"
    old_key = "2020/01/01/" + old_id
    fake.objects[old_key] = b"old unreferenced"
    fake.content_types[old_key] = "text/plain"
    _seed_index_row(handler, {
        "id": old_id,
        "path": fake.uri_for_key(old_key),
        "mime": "text/plain",
        "size": 17,
        "name": old_id,
        "hash": "hash-old",
        "checksum_sha256": "hash-old",
        "original_name": "old.txt",
        "owner": "alice",
        "uploaded_at": "2020-01-01T00:00:00",
        "created_at": "2020-01-01T00:00:00",
        "last_accessed": "2020-01-01T00:00:00",
    })
    # A recent object that must survive.
    fresh_meta = handler.save_upload(_fake_upload(b"fresh"), "127.0.0.1", owner="alice")

    cleaned = handler.cleanup_old_uploads(
        referenced_upload_ids=set(),
        referenced_upload_hashes=set(),
    )

    assert cleaned == 1
    assert not fake.exists(old_key)
    assert fake.exists(fake.key_for_uri(fresh_meta["path"]))
    assert fresh_meta["id"] in {row["id"] for row in _index(handler).values()}
    assert old_id not in {row["id"] for row in _index(handler).values()}


def test_cleanup_s3_fails_closed_without_reference_snapshot(tmp_path, monkeypatch):
    handler = _make_handler(tmp_path)
    fake = FakeS3Backend()
    _use_fake_backend(monkeypatch, fake)
    old_id = "e" * 32 + ".txt"
    old_key = "2020/01/01/" + old_id
    fake.objects[old_key] = b"keep me"

    assert handler.cleanup_old_uploads() == 0
    assert fake.exists(old_key)


def test_cleanup_s3_spares_referenced_objects(tmp_path, monkeypatch):
    handler = _make_handler(tmp_path)
    fake = FakeS3Backend()
    _use_fake_backend(monkeypatch, fake)

    old_id = "f" * 32 + ".txt"
    old_key = "2020/01/01/" + old_id
    fake.objects[old_key] = b"referenced"
    _seed_index_row(handler, {
        "id": old_id,
        "path": fake.uri_for_key(old_key),
        "mime": "text/plain",
        "size": 11,
        "name": old_id,
        "hash": "hash-ref",
        "checksum_sha256": "hash-ref",
        "original_name": "ref.txt",
        "owner": "alice",
        "uploaded_at": "2020-01-01T00:00:00",
        "created_at": "2020-01-01T00:00:00",
        "last_accessed": "2020-01-01T00:00:00",
    })

    cleaned = handler.cleanup_old_uploads(
        referenced_upload_ids={old_id},
        referenced_upload_hashes=set(),
    )
    assert cleaned == 0
    assert fake.exists(old_key)
