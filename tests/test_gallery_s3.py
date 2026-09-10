"""Gallery object-storage lane (ODYSSEUS_GALLERY_STORAGE) — src/gallery_storage.py.

Repo style: pytest + fakes, no moto / no real MinIO. Pins the dispatch
matrix (local default vs s3), uploads.json-style value semantics on the
GalleryImage.filename column, coexistence + rollback safety, the one-shot
migration (idempotent, resumable, per-file-safe), foreign-bucket refusal,
and read-cache flow-through.
"""
import json
import os
import re
import sys
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import HTTPException

from src import gallery_storage
from src.gallery_storage import (
    ENV_GALLERY_STORAGE,
    gallery_delete_stored,
    gallery_local_path,
    gallery_new_filename,
    gallery_read_bytes,
    gallery_stored_from_url_name,
    gallery_store_image,
    gallery_url_name,
    gallery_write_stored,
    migrate_gallery_assets_to_backend,
    validate_gallery_storage_at_boot,
)
from src.storage_backend import S3Backend

BUCKET = "odysseus-test"
S3_URI_KEY_RE = re.compile(
    rf"^s3://{BUCKET}/gallery/\d{{4}}/\d{{2}}/\d{{2}}/[0-9a-f]{{32}}\.[a-z0-9]+$"
)

ALL_S3_ENV = {
    "ODYSSEUS_S3_ENDPOINT": "https://minio.example.com",
    "ODYSSEUS_S3_ACCESS_KEY": "AKIAEXAMPLE",
    "ODYSSEUS_S3_SECRET_KEY": "secretexample",
    "ODYSSEUS_S3_BUCKET": "odysseus-uploads",
}


class FakeS3Backend(S3Backend):
    """In-memory object store: real URI/key semantics, no network, no boto3."""

    def __init__(self, bucket=BUCKET):
        super().__init__(bucket=bucket)
        self.objects: dict[str, bytes] = {}
        self.content_types: dict[str, str] = {}
        self.get_calls: list[str] = []
        self.put_calls: list[str] = []
        self.fail_puts: set[str] = set()

    def put(self, key, fileobj, content_type):
        if key in self.fail_puts:
            raise RuntimeError(f"simulated put failure for {key}")
        self.objects[key] = fileobj.read()
        self.content_types[key] = content_type
        self.put_calls.append(key)

    def get_bytes(self, key):
        self.get_calls.append(key)
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
        return {"size": len(self.objects[key]), "content_type": self.content_types.get(key)}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch, tmp_path):
    monkeypatch.delenv(ENV_GALLERY_STORAGE, raising=False)
    for name in ALL_S3_ENV:
        monkeypatch.delenv(name, raising=False)
    # Keep every test's local writes inside its own tmp tree. Patch the
    # SAME seam the routes own (GALLERY_IMAGE_DIR) — gallery_storage
    # resolves its local dir through it, so the two can never disagree.
    import routes.gallery.gallery_routes as _gr
    monkeypatch.setattr(_gr, "GALLERY_IMAGE_DIR", tmp_path / "generated_images")
    yield


def _use_s3(monkeypatch, fake: FakeS3Backend):
    monkeypatch.setenv(ENV_GALLERY_STORAGE, "s3")
    monkeypatch.setattr(gallery_storage, "_gallery_backend", lambda: fake)


def _seed_local_asset(tmp_path: Path, content: bytes = b"png-bytes", ext: str = "png") -> str:
    """Write a legacy local gallery file and return its bare filename."""
    name = f"{uuid.uuid4().hex[:12]}.{ext}"
    directory = tmp_path / "generated_images"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_bytes(content)
    return name


# ---------------------------------------------------------------------------
# Dispatch matrix: default local, opt-in s3, bad value
# ---------------------------------------------------------------------------

def test_store_default_writes_local_file_unchanged(tmp_path):
    stored, url_name = gallery_store_image(b"local-bytes", "png")
    assert stored == url_name  # bare filename both ways
    assert re.fullmatch(r"[0-9a-f]{32}\.png", stored)
    assert (tmp_path / "generated_images" / stored).read_bytes() == b"local-bytes"


def test_store_explicit_local_env_behaves_identically(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV_GALLERY_STORAGE, "local")
    stored, url_name = gallery_store_image(b"still-local", "jpg")
    assert stored == url_name
    assert (tmp_path / "generated_images" / stored).read_bytes() == b"still-local"


def test_store_s3_writes_object_and_uri(monkeypatch, tmp_path):
    fake = FakeS3Backend()
    _use_s3(monkeypatch, fake)

    stored, url_name = gallery_store_image(b"object-bytes", "png")

    assert S3_URI_KEY_RE.fullmatch(stored)
    assert url_name == stored.rsplit("/", 1)[-1]
    key = fake.key_for_uri(stored)
    assert fake.objects[key] == b"object-bytes"
    assert fake.content_types[key] == "image/png"
    # Nothing touched local disk.
    assert not (tmp_path / "generated_images").exists()


def test_invalid_env_value_raises(monkeypatch):
    monkeypatch.setenv(ENV_GALLERY_STORAGE, "minio")
    with pytest.raises(RuntimeError, match=ENV_GALLERY_STORAGE):
        gallery_store_image(b"x", "png")


def test_read_write_delete_dispatch_both_shapes(monkeypatch, tmp_path):
    fake = FakeS3Backend()
    _use_s3(monkeypatch, fake)

    s3_stored, _ = gallery_store_image(b"object-bytes", "png")
    local_name = _seed_local_asset(tmp_path, b"local-bytes")

    # Coexistence: both shapes readable through one API, one env state.
    assert gallery_read_bytes(s3_stored) == b"object-bytes"
    assert gallery_read_bytes(local_name) == b"local-bytes"
    # URL names stay bare for both.
    assert gallery_url_name(s3_stored) == s3_stored.rsplit("/", 1)[-1]
    assert gallery_url_name(local_name) == local_name

    # Replace/rotate path: overwrite the object in place.
    gallery_write_stored(s3_stored, b"rotated")
    assert fake.objects[fake.key_for_uri(s3_stored)] == b"rotated"

    # Delete removes the right thing per shape.
    gallery_delete_stored(s3_stored)
    assert not fake.exists(fake.key_for_uri(s3_stored))
    gallery_delete_stored(local_name)
    assert not (tmp_path / "generated_images" / local_name).exists()


def test_foreign_bucket_row_rejected_without_fetch(monkeypatch):
    fake = FakeS3Backend()
    _use_s3(monkeypatch, fake)
    foreign = f"s3://other-bucket/gallery/2026/09/09/{'a' * 32}.png"
    with pytest.raises(HTTPException) as exc:
        gallery_read_bytes(foreign)
    assert exc.value.status_code == 400
    assert fake.get_calls == []  # refused before any storage traffic


def test_unsafe_local_name_rejected():
    # Malformed names raise 400 (same charset as the old resolver)...
    for bad in ("../escape.png", "not hex.png", "sub/dir.png", "..", "."):
        with pytest.raises(HTTPException):
            gallery_local_path(bad)
    # ...while empty/None/s3 values are simply "not local" (None).
    for not_local in ("", None, f"s3://{BUCKET}/gallery/2026/09/09/x.png"):
        assert gallery_local_path(not_local) is None


def test_new_filenames_fit_serve_pattern():
    from src.generated_images import GENERATED_IMAGE_RE

    for ext in ("png", "jpg", "mp4"):
        assert GENERATED_IMAGE_RE.fullmatch(gallery_new_filename(ext))


# ---------------------------------------------------------------------------
# Boot validation
# ---------------------------------------------------------------------------

def test_boot_validation_noop_for_local(monkeypatch):
    validate_gallery_storage_at_boot()  # unset
    monkeypatch.setenv(ENV_GALLERY_STORAGE, "local")
    validate_gallery_storage_at_boot()


def test_boot_validation_s3_requires_s3_env(monkeypatch):
    monkeypatch.setenv(ENV_GALLERY_STORAGE, "s3")
    with pytest.raises(RuntimeError, match=ENV_GALLERY_STORAGE):
        validate_gallery_storage_at_boot()
    for name, value in ALL_S3_ENV.items():
        monkeypatch.setenv(name, value)
    validate_gallery_storage_at_boot()  # complete config passes


def test_boot_validation_rejects_bad_value(monkeypatch):
    monkeypatch.setenv(ENV_GALLERY_STORAGE, "wasb")
    with pytest.raises(RuntimeError, match=ENV_GALLERY_STORAGE):
        validate_gallery_storage_at_boot()


# ---------------------------------------------------------------------------
# Serve dispatch: URL basename -> s3 row suffix lookup
# ---------------------------------------------------------------------------

@pytest.fixture
def gallery_db(monkeypatch):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import NullPool

    import core.database as cdb

    tmpdb = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmpdb.close()
    engine = create_engine(
        f"sqlite:///{tmpdb.name}",
        connect_args={"check_same_thread": False},
        poolclass=NullPool,
    )
    cdb.Base.metadata.create_all(engine)
    ts = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    monkeypatch.setattr("core.database.SessionLocal", ts)
    monkeypatch.setattr("src.database.SessionLocal", ts)
    yield ts
    engine.dispose()
    try:
        os.unlink(tmpdb.name)
    except OSError:
        pass


def _add_gallery_row(db_session, filename: str, owner="alice", active=True) -> str:
    from core.database import GalleryImage

    db = db_session()
    try:
        image_id = str(uuid.uuid4())
        db.add(GalleryImage(
            id=image_id, filename=filename, prompt="p", model="imported",
            owner=owner, is_active=active,
        ))
        db.commit()
        return image_id
    finally:
        db.close()


def test_url_name_resolves_s3_row_by_suffix(gallery_db):
    fake = FakeS3Backend()
    stored, url_name = fake.uri_for_key(f"gallery/2026/09/09/{'b' * 32}.png"), f"{'b' * 32}.png"
    fake.objects[fake.key_for_uri(stored)] = b"x"
    _add_gallery_row(gallery_db, stored)
    _add_gallery_row(gallery_db, f"{'c' * 12}.png")  # local row: must not match

    db = gallery_db()
    try:
        assert gallery_stored_from_url_name(url_name, db) == stored
        # Unknown basename / inactive rows / local rows resolve to None.
        assert gallery_stored_from_url_name(f"{'d' * 12}.png", db) is None
    finally:
        db.close()


def test_url_name_ignores_inactive_rows(gallery_db):
    stored = f"s3://{BUCKET}/gallery/2026/09/09/{'e' * 32}.png"
    _add_gallery_row(gallery_db, stored, active=False)
    db = gallery_db()
    try:
        assert gallery_stored_from_url_name(f"{'e' * 32}.png", db) is None
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------

def _seed_row_with_file(gallery_db, tmp_path, content=b"legacy-photo") -> tuple[str, Path]:
    name = _seed_local_asset(tmp_path, content)
    _add_gallery_row(gallery_db, name)
    return name, tmp_path / "generated_images" / name


def test_migration_moves_files_rows_and_frees_bytes(monkeypatch, tmp_path, gallery_db):
    fake = FakeS3Backend()
    _use_s3(monkeypatch, fake)
    seeded = [
        _seed_row_with_file(gallery_db, tmp_path, b"photo-A"),
        _seed_row_with_file(gallery_db, tmp_path, b"photo-B"),
    ]
    names = [name for name, _ in seeded]

    summary = migrate_gallery_assets_to_backend()

    assert summary["moved"] == 2
    assert summary["skipped"] == 0
    assert summary["bytes_freed"] == len(b"photo-A") + len(b"photo-B")
    assert summary["errors"] == []
    db = gallery_db()
    try:
        from core.database import GalleryImage

        for name, content in zip(names, [b"photo-A", b"photo-B"]):
            row = db.query(GalleryImage).filter(GalleryImage.filename.endswith("/" + name)).first()
            assert row is not None
            key = fake.key_for_uri(row.filename)
            assert re.fullmatch(rf"gallery/\d{{4}}/\d{{2}}/\d{{2}}/{name}", key)
            assert fake.objects[key] == content
            # URL stability: basename survives the migration.
            assert gallery_url_name(row.filename) == name
            assert not (tmp_path / "generated_images" / name).exists()
    finally:
        db.close()


def test_migration_partial_failure_is_resumable(monkeypatch, tmp_path, gallery_db):
    fake = FakeS3Backend()
    _use_s3(monkeypatch, fake)
    ok_name, ok_path = _seed_row_with_file(gallery_db, tmp_path, b"good")
    bad_name, bad_path = _seed_row_with_file(gallery_db, tmp_path, b"bad")

    # Fail exactly the bad asset: the migration key embeds the original
    # filename, so select on it.
    original_put = fake.put

    def _selective_put(key, fileobj, content_type):
        if bad_name in key:
            raise RuntimeError("simulated put failure")
        return original_put(key, fileobj, content_type)

    fake.put = _selective_put

    summary = migrate_gallery_assets_to_backend()

    assert summary["moved"] == 1
    assert len(summary["errors"]) == 1
    assert summary["errors"][0]["filename"] == bad_name
    # Per-file safety: failed put -> file kept AND row unchanged (retryable).
    assert bad_path.exists()
    db = gallery_db()
    try:
        from core.database import GalleryImage

        bad_row = db.query(GalleryImage).filter(GalleryImage.filename == bad_name).first()
        assert bad_row is not None  # still the bare local name
        assert ok_path is not None and not ok_path.exists()  # good one moved
    finally:
        db.close()

    # Resume: repair the failure and re-run — only the failed asset moves.
    fake.put = original_put
    second = migrate_gallery_assets_to_backend()
    assert second["moved"] == 1
    assert second["errors"] == []
    assert any(k.endswith("/" + bad_name) for k in fake.objects)
    assert fake.objects[[k for k in fake.objects if k.endswith("/" + bad_name)][0]] == b"bad"
    assert not bad_path.exists()


def test_migration_idempotent_and_dry_run(monkeypatch, tmp_path, gallery_db):
    fake = FakeS3Backend()
    _use_s3(monkeypatch, fake)
    _seed_row_with_file(gallery_db, tmp_path, b"one")
    _seed_row_with_file(gallery_db, tmp_path, b"two")

    dry = migrate_gallery_assets_to_backend(dry_run=True)
    assert dry["dry_run"] is True
    assert dry["moved"] == 2
    assert dry["bytes_freed"] == 6
    assert fake.objects == {}  # nothing uploaded
    assert len(list((tmp_path / "generated_images").iterdir())) == 2  # nothing unlinked

    migrate_gallery_assets_to_backend()
    again = migrate_gallery_assets_to_backend()
    assert again["moved"] == 0
    assert again["skipped"] == 2  # already s3:// rows


def test_migration_requires_env(monkeypatch, tmp_path, gallery_db):
    _seed_row_with_file(gallery_db, tmp_path, b"stray")
    summary = migrate_gallery_assets_to_backend()
    assert "error" in summary and ENV_GALLERY_STORAGE in summary["error"]
    assert (tmp_path / "generated_images").exists()


def test_migration_skips_missing_files_and_unsafe_names(monkeypatch, tmp_path, gallery_db):
    fake = FakeS3Backend()
    _use_s3(monkeypatch, fake)
    # Row whose file is gone (already cleaned / partial state).
    _add_gallery_row(gallery_db, f"{'f' * 12}.png")
    summary = migrate_gallery_assets_to_backend()
    assert summary["moved"] == 0
    assert summary["skipped"] == 1
    assert summary["errors"] == []


# ---------------------------------------------------------------------------
# Read-cache flow-through (companion of the CachedS3Storage tests)
# ---------------------------------------------------------------------------

def test_gallery_objects_flow_through_the_read_cache(monkeypatch, tmp_path):
    from src.storage_cache import CachedS3Storage

    fake = FakeS3Backend()
    wrapper = CachedS3Storage(
        fake, cache_dir=str(tmp_path / "s3-cache"), max_bytes=1024 * 1024
    )
    _use_s3(monkeypatch, fake)
    monkeypatch.setattr(gallery_storage, "_gallery_backend", lambda: wrapper)

    stored, _ = gallery_store_image(b"cached-gallery-bytes", "png")
    key = wrapper.key_for_uri(stored)
    assert key.startswith("gallery/")  # cacheable: slash-containing key
    assert wrapper._cacheable_key(key) is True

    # Write-through: the store() already cached the object, so even the
    # FIRST read is a local hit — zero S3 GETs.
    assert gallery_read_bytes(stored) == b"cached-gallery-bytes"
    assert fake.get_calls.count(key) == 0
    cache_file = tmp_path / "s3-cache" / key.replace("/", "__")
    assert cache_file.exists()

    # Cold cache (e.g. after eviction): one S3 GET repopulates, then hits.
    os.unlink(cache_file)
    assert gallery_read_bytes(stored) == b"cached-gallery-bytes"
    assert fake.get_calls.count(key) == 1
    assert gallery_read_bytes(stored) == b"cached-gallery-bytes"
    assert fake.get_calls.count(key) == 1  # served from the repopulated cache


# ---------------------------------------------------------------------------
# Review round 1: W1-W6 regressions
# ---------------------------------------------------------------------------

def _route_endpoint(path: str, method: str = "POST"):
    import routes.gallery.gallery_routes as gallery_routes

    router = gallery_routes.setup_gallery_routes()
    for r in router.routes:
        if getattr(r, "path", None) == path and method in getattr(r, "methods", set()):
            return gallery_routes, r.endpoint
    raise RuntimeError(f"{method} {path} not found")


def test_w1_album_cover_urls_use_bare_basename():
    """Source pin (closure handlers, AST convention of the album tests):
    both cover paths must go through gallery_url_name — a raw s3:// URI in
    cover_url would be unresolvable."""
    source = Path("routes/gallery/gallery_routes.py").read_text(encoding="utf-8")
    assert 'cover_url = f"/api/generated-image/{gallery_url_name(cover.filename)}"' in source
    assert 'cover_url = f"/api/generated-image/{gallery_url_name(first.filename)}"' in source
    assert 'f"/api/generated-image/{cover.filename}"' not in source
    assert 'f"/api/generated-image/{first.filename}"' not in source


async def test_w2_ai_tag_reads_s3_rows_via_dispatch(tmp_path, monkeypatch, gallery_db):
    """ai-tag must read object-stored rows through gallery_read_bytes; the
    old local resolver 400'd on URIs before the vision call."""
    fake = FakeS3Backend()
    _use_s3(monkeypatch, fake)
    stored, _ = gallery_store_image(b"vision-target-bytes", "png")
    image_id = _add_gallery_row(gallery_db, stored)
    gallery_routes, endpoint = _route_endpoint("/api/gallery/{image_id}/ai-tag")
    monkeypatch.setattr(gallery_routes, "SessionLocal", gallery_db)
    monkeypatch.setattr(gallery_routes, "get_current_user", lambda r: "alice")
    monkeypatch.setattr(
        "src.document_processor._load_vl_settings",
        lambda: {"vision_enabled": True, "vision_model": ""},
    )
    def _no_model(*a, **kw):
        raise ValueError("no vision model")
    monkeypatch.setattr("src.document_processor._resolve_vl_model", _no_model)

    from fastapi import Request

    result = await endpoint(Request(scope={"type": "http"}), image_id)

    # The bytes were read fine (no 400/404); the route reached the vision
    # configuration stage and returned its operator-facing error.
    assert "No vision model configured" in result.get("error", "")
    assert fake.get_calls == [fake.key_for_uri(stored)]


def test_w3_suffix_lookup_rejects_like_wildcard_names(gallery_db):
    """W3: the serve-dispatch lookup must not turn URL garbage into a LIKE
    pattern ('%25' -> '/%' would match the first active row of ANY owner,
    leaking foreign bytes + an existence oracle)."""
    stored, url_name = f"s3://{BUCKET}/gallery/2026/09/09/{'a' * 32}.png", f"{'a' * 32}.png"
    _add_gallery_row(gallery_db, stored)

    db = gallery_db()
    try:
        # Injection attempts resolve to nothing...
        for attack in ("%25", "%252e%252e", "100%.png", "_", "..%2fpng", "a%25"):
            assert gallery_stored_from_url_name(attack, db) is None, attack
        # ...while the legitimate basename keeps resolving.
        assert gallery_stored_from_url_name(url_name, db) == stored
    finally:
        db.close()


async def test_w4_chat_scrub_matches_migrated_rows(tmp_path, monkeypatch, gallery_db):
    """Post-delete chat-history scrub must match the BARE name for migrated
    rows (chat events embed URL names; the URI contains it, never the
    reverse) — otherwise old tool-event bubbles survive the delete."""
    import json as _json

    from core.database import ChatMessage, Session as DbSession

    fake = FakeS3Backend()
    _use_s3(monkeypatch, fake)
    stored, url_name = f"s3://{BUCKET}/gallery/2026/09/09/{'b' * 32}.png", f"{'b' * 32}.png"
    image_id = _add_gallery_row(gallery_db, stored)

    db = gallery_db()
    try:
        db.add(DbSession(id="sess-1", name="s", endpoint_url="", model=""))
        db.add(ChatMessage(
            id="m-user", session_id="sess-1", role="user",
            content="make me a picture",
        ))
        db.commit()
        db.flush()
        db.add(ChatMessage(
            id="m-assistant", session_id="sess-1", role="assistant",
            content="Generated image",
            meta_data=_json.dumps({"tool_events": [{
                "image_id": image_id,
                "image_url": f"/api/generated-image/{url_name}",
            }]}),
        ))
        db.commit()
    finally:
        db.close()

    gallery_routes, endpoint = _route_endpoint("/api/gallery/{image_id}", method="DELETE")
    monkeypatch.setattr(gallery_routes, "SessionLocal", gallery_db)
    monkeypatch.setattr(gallery_routes, "get_current_user", lambda r: "alice")

    from fastapi import Request

    result = await endpoint(Request(scope={"type": "http"}), image_id)
    assert result["status"] == "deleted"

    db = gallery_db()
    try:
        assert db.query(ChatMessage).filter(ChatMessage.id == "m-assistant").first() is None
        assert db.query(ChatMessage).filter(ChatMessage.id == "m-user").first() is None
    finally:
        db.close()


async def test_w5_admin_wipe_deletes_objects_and_real_dir(tmp_path, monkeypatch, gallery_db):
    """The gallery wipe must collect s3 URIs BEFORE dropping rows and delete
    the objects, plus rmtree the REAL asset dir (GENERATED_IMAGES_DIR seam)."""
    fake = FakeS3Backend()
    _use_s3(monkeypatch, fake)
    stored, _ = gallery_store_image(b"object-bytes", "png")
    _add_gallery_row(gallery_db, stored)  # the row the wipe must harvest
    local_name = _seed_local_asset(tmp_path, b"local-bytes")

    from core.database import GalleryAlbum

    db = gallery_db()
    try:
        db.add(GalleryAlbum(id="album-1", name="a"))
        db.commit()
    finally:
        db.close()

    import routes.admin_wipe_routes as admin_wipe

    monkeypatch.setattr(admin_wipe, "SessionLocal", gallery_db)
    monkeypatch.setattr(admin_wipe, "require_admin", lambda r: None)

    from fastapi import Request

    router = admin_wipe.setup_admin_wipe_routes(session_manager=None)
    handler = next(r for r in router.routes if r.path == "/api/admin/wipe/{kind}").endpoint
    result = handler(kind="gallery", request=Request(scope={"type": "http"}))

    assert result["status"] == "deleted"
    assert not fake.exists(fake.key_for_uri(stored))  # object gone with the rows
    assert not (tmp_path / "generated_images" / local_name).exists()  # real dir wiped
    db = gallery_db()
    try:
        from core.database import GalleryImage

        assert db.query(GalleryImage).count() == 0
    finally:
        db.close()


async def test_w6_upload_row_failure_compensates_object(tmp_path, monkeypatch, gallery_db):
    """Bytes reach storage BEFORE the row insert; if the commit fails the
    object must be deleted again (row-less objects 404 forever and nothing
    sweeps them)."""
    fake = FakeS3Backend()
    _use_s3(monkeypatch, fake)

    gallery_routes, endpoint = _route_endpoint("/api/gallery/upload")
    monkeypatch.setattr(gallery_routes, "SessionLocal", gallery_db)
    monkeypatch.setattr(gallery_routes, "get_current_user", lambda r: "alice")

    class _CommitFails(gallery_db().__class__):
        def commit(self):
            raise RuntimeError("simulated DB commit failure")

    _fail_factory = _CommitFails  # noqa: F841 — clarity
    monkeypatch.setattr(gallery_routes, "SessionLocal", _CommitFails)

    from fastapi import FastAPI, Request
    from fastapi.testclient import TestClient

    app = FastAPI()
    app.include_router(gallery_routes.setup_gallery_routes())
    client = TestClient(app, raise_server_exceptions=False)
    response = client.post(
        "/api/gallery/upload",
        files={"file": ("photo.png", b"compensated-bytes", "image/png")},
    )

    assert response.status_code == 500
    assert fake.objects == {}  # the stored object was removed again
    # s3 mode never touches local disk in the first place.
    assert not (tmp_path / "generated_images").exists()
