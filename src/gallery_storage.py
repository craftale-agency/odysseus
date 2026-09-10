"""
gallery_storage.py

Object-storage lane for GALLERY assets, opt-in via ODYSSEUS_GALLERY_STORAGE.

WHY: the gallery is the disk-hungriest lane (bulk photo imports), and on
hosts where the data volume is thin (nebula/Odypezzus) the assets belong
in MinIO, not on local disk. The main instance (crafthost) also runs
ODYSSEUS_STORAGE_BACKEND=s3 but has plenty of disk — so this is a SEPARATE
switch, default local: unset, gallery behavior is byte-identical to today.

    ODYSSEUS_GALLERY_STORAGE   "local" (default) or "s3"

    s3 requires the four ODYSSEUS_S3_* env vars (same contract as the
    uploads backend; validate_gallery_storage_at_boot fails fast naming
    this var). Reuses the ACTIVE storage backend singleton when it is
    already s3 (one client, and the CachedS3Storage read cache applies to
    gallery objects for free); with ODYSSEUS_STORAGE_BACKEND=local a
    dedicated lazily-built S3Backend serves gallery traffic only.

How the gallery tracks assets (exploration finding): DB rows
(gallery_images, unique `filename` column) + files in GENERATED_IMAGES_DIR
(data/generated_images). GALLERY_DIR/GALLERY_UPLOADS_DIR are legacy dead
constants (only the admin wipe touches them). The URL surface is
``/api/generated-image/<bare-filename>`` — and those URLs are ALSO baked
permanently into chat history (ChatMessage.meta_data tool_events), so the
URL scheme must never change. Value semantics therefore mirror uploads.json:

* ``filename`` stores ``s3://<bucket>/gallery/YYYY/MM/DD/<uuid32><ext>``
  for new/migrated assets when enabled — bare local filenames keep
  working unchanged (coexistence + rollback-safe: flip the env back and
  old rows still resolve against local files).
* URLs keep using the BARE basename (gallery_url_name) for both shapes;
  the serve route dispatches: local file first (fast path, zero DB hit
  for every legacy/local asset), then a suffix lookup of the s3 row.

The ``gallery/`` key prefix keeps the lane visually separate in the
bucket, and the date-sharded keys stay slash-containing so the read
cache's _cacheable_key accepts them.

Out of scope this round (known follow-up): non-gallery GENERATED_IMAGES_DIR
mechanics (PIL temp flows). Every writer that creates a GalleryImage row
routes through gallery_store_image, so row-backed assets are fully covered.
"""

import io
import logging
import re
import mimetypes
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from fastapi import HTTPException

from src.constants import GENERATED_IMAGES_DIR
from src.storage_backend import (
    REQUIRED_S3_ENV,
    S3Backend,
    is_s3_uri,
    parse_s3_uri,
)

logger = logging.getLogger(__name__)

ENV_GALLERY_STORAGE = "ODYSSEUS_GALLERY_STORAGE"
GALLERY_KEY_PREFIX = "gallery"

_GALLERY_VIDEO_EXTS = {"mp4", "mov", "webm", "mkv", "m4v"}
_GALLERY_IMAGE_EXTS = {"png", "jpg", "jpeg", "webp", "gif"}


def gallery_storage_enabled() -> bool:
    """True when the gallery lane is armed for object storage.

    Read at call time like every other storage env. An unrecognized value
    raises (a typo must not silently mean "local").
    """
    raw = os.getenv(ENV_GALLERY_STORAGE, "local").strip().lower()
    if not raw or raw == "local":
        return False
    if raw == "s3":
        return True
    raise RuntimeError(
        f"{ENV_GALLERY_STORAGE} must be 'local' or 's3' (got {raw!r})"
    )


def validate_gallery_storage_at_boot() -> None:
    """Fail fast at startup when ODYSSEUS_GALLERY_STORAGE=s3 is misconfigured.

    No-op unless the env selects s3. s3 requires every mandatory
    ODYSSEUS_S3_* var (same contract as the uploads backend — there is
    deliberately no silent local fallback) and an importable boto3.
    """
    try:
        enabled = gallery_storage_enabled()
    except RuntimeError as e:
        raise RuntimeError(f"{e}. Refusing to start with a broken gallery "
                           "storage selection.") from None
    if not enabled:
        return
    missing = [env for env in REQUIRED_S3_ENV if not os.getenv(env, "").strip()]
    if missing:
        raise RuntimeError(
            f"{ENV_GALLERY_STORAGE}=s3 requires "
            + ", ".join(missing)
            + " to be set (non-empty). Refusing to start with a half-configured"
            " gallery object store."
        )
    try:
        import boto3  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            f"{ENV_GALLERY_STORAGE}=s3 requires the optional dependency "
            "boto3, which is not installed. Install it with: "
            "pip install -r requirements-optional.txt"
        ) from e


def _gallery_backend() -> S3Backend:
    """Backend for gallery object traffic.

    Reuses the active singleton when it is s3 (shared client + read cache);
    otherwise builds a dedicated S3Backend from the S3 env (boto3 client
    stays unbuilt until first use).
    """
    from src import storage_backend

    active = storage_backend.get_storage_backend()
    if isinstance(active, S3Backend):
        return active
    return storage_backend._s3_backend_for_uri()


def gallery_new_filename(ext: str) -> str:
    """Bare filename for a new asset: uuid32 + extension (fits the
    GENERATED_IMAGE_RE serve pattern [8-64 hex] and matches the uploads
    key style)."""
    ext = ext.lstrip(".").lower() or "png"
    return f"{uuid.uuid4().hex}.{ext}"


def gallery_object_key(filename: str) -> str:
    """Date-sharded object key for a gallery asset (gallery/YYYY/MM/DD/name)."""
    now = datetime.now(timezone.utc)
    return f"{GALLERY_KEY_PREFIX}/{now.strftime('%Y/%m/%d')}/{filename}"


def gallery_mime_for(name: str) -> str:
    ext = Path(str(name or "")).suffix.lstrip(".").lower()
    mime = {
        "png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
        "webp": "image/webp", "gif": "image/gif",
        "mp4": "video/mp4", "mov": "video/quicktime", "webm": "video/webm",
        "mkv": "video/x-matroska", "m4v": "video/mp4",
    }.get(ext)
    return mime or mimetypes.guess_type(str(name))[0] or "application/octet-stream"


def gallery_url_name(stored: Any) -> str:
    """The filename component used in /api/generated-image/ URLs.

    Bare local filenames pass through unchanged; s3 URIs contribute their
    basename (URLs stay identical across a migration, so chat-history
    image_urls and browser caches keep working).
    """
    if is_s3_uri(stored):
        return str(stored).rsplit("/", 1)[-1]
    return str(stored or "")


# Same charset as the historical gallery resolver (_sanitize_gallery_filename):
# one safe path component, no separators, no traversal. Deliberately looser
# than the serve route's GENERATED_IMAGE_RE so legacy stored names (e.g.
# short test fixtures) keep working through delete/replace/rotate/zip.
_GALLERY_LOCAL_NAME_RE = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


def gallery_local_path(stored: Any) -> Optional[Path]:
    """Confined local path for a LOCAL stored filename; None for s3 rows.

    Same containment contract as the old routes-gallery resolver (charset
    check + realpath/commonpath under GENERATED_IMAGES_DIR); never 404s on
    a missing file (existence is the caller's concern) — a malformed or
    unsafe name raises 400.
    """
    if is_s3_uri(stored) or not isinstance(stored, str) or not stored:
        return None
    if stored in (".", "..") or not _GALLERY_LOCAL_NAME_RE.fullmatch(stored):
        raise HTTPException(400, "Unsafe gallery filename")
    root = _local_gallery_dir().resolve()
    path = (root / stored).resolve()
    try:
        if os.path.commonpath([str(root), str(path)]) != str(root):
            raise ValueError
    except Exception:
        raise HTTPException(400, "Unsafe gallery filename")
    return path


def _local_gallery_dir() -> Path:
    """The gallery's local image dir, resolved through the owning module.

    routes.gallery.gallery_routes.GALLERY_IMAGE_DIR is the historical seam
    (tests patch it; it is derived from GENERATED_IMAGES_DIR at import).
    Resolving through it — lazily, no import cycle — keeps the routes
    resolver and the storage dispatch on ONE root instead of two
    independently-imported copies of the same constant.
    """
    try:
        from routes.gallery.gallery_routes import GALLERY_IMAGE_DIR
        return Path(GALLERY_IMAGE_DIR)
    except Exception:
        return Path(GENERATED_IMAGES_DIR)


def _put_object(backend: S3Backend, key: str, content: bytes, name: str) -> None:
    backend.put(key, io.BytesIO(content), gallery_mime_for(name))


def gallery_store_image(content: bytes, ext: str) -> Tuple[str, str]:
    """Persist one new gallery asset. Returns (stored_value, url_name).

    Enabled  -> s3://<bucket>/gallery/YYYY/MM/DD/<uuid32><ext> + bare name.
    Disabled -> the historical behavior: file in GENERATED_IMAGES_DIR, the
    filename is both the stored value and the URL name.
    Writers creating GalleryImage rows MUST route through this so the
    `filename` column keeps its value semantics.
    """
    filename = gallery_new_filename(ext)
    if gallery_storage_enabled():
        backend = _gallery_backend()
        key = gallery_object_key(filename)
        _put_object(backend, key, content, filename)
        return backend.uri_for_key(key), filename
    img_dir = _local_gallery_dir()
    img_dir.mkdir(parents=True, exist_ok=True)
    (img_dir / filename).write_bytes(content)
    return filename, filename


def gallery_write_stored(stored: Any, content: bytes) -> None:
    """Overwrite the bytes behind an EXISTING asset (replace/rotate).

    Dispatches on the stored value: s3 rows put the same key again (the
    read cache refreshes with it); local rows rewrite the confined file.
    """
    if is_s3_uri(stored):
        parsed = parse_s3_uri(stored)
        if parsed is None:
            raise HTTPException(400, "Unsafe gallery filename")
        _, key = parsed
        _put_object(_gallery_backend(), key, content, key)
        return
    path = gallery_local_path(stored)
    if path is None:
        raise HTTPException(400, "Unsafe gallery filename")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def gallery_read_bytes(stored: Any) -> bytes:
    """Read the bytes behind an asset (either storage shape)."""
    if is_s3_uri(stored):
        parsed = parse_s3_uri(stored)
        if parsed is None:
            raise HTTPException(400, "Unsafe gallery filename")
        backend = _gallery_backend()
        _, key = parsed
        if backend.key_for_uri(stored) is None:
            # Foreign-bucket row (hand-edited DB / bucket flip): refuse
            # before any storage traffic.
            raise HTTPException(400, "Gallery object is not in the managed bucket")
        return backend.get_bytes(key)
    path = gallery_local_path(stored)
    if path is None or not path.exists():
        raise HTTPException(404, "Image file not found")
    return path.read_bytes()


def gallery_delete_stored(stored: Any) -> None:
    """Remove the bytes behind an asset (either storage shape). Local misses
    are not an error (soft-delete already committed)."""
    if is_s3_uri(stored):
        parsed = parse_s3_uri(stored)
        if parsed is None:
            return
        backend = _gallery_backend()
        key = backend.key_for_uri(stored)
        if key is not None:
            backend.delete(key)
        return
    try:
        path = gallery_local_path(stored)
        if path is not None and path.exists():
            path.unlink()
    except OSError as e:
        logger.warning("Could not remove gallery image file for %r: %s", stored, e)


def gallery_stored_from_url_name(name: str, db) -> Optional[str]:
    """Resolve a URL basename to the s3 row's stored URI (serve dispatch).

    Only s3 rows match: their stored value ends with '/<name>' while local
    rows store the bare name (and were already served from disk before
    this lookup). First active match wins; None when no such row.

    SECURITY: *name* arrives straight off the URL path (the serve route
    only rejects it AFTER this lookup when nothing resolves). endswith()
    compiles to a SQL LIKE with '%'-wildcards, so a name like '%25' would
    become the pattern '/%' and match the first active row of ANY owner —
    serving foreign bytes and confirming existence. Two independent
    guards: the name must be a single safe path component (no %, _, . or
    any other LIKE metachar survives the gallery charset), and the query
    escapes what the charset let through anyway.
    """
    from core.database import GalleryImage
    from sqlalchemy import and_

    if not isinstance(name, str) or not _GALLERY_LOCAL_NAME_RE.fullmatch(name):
        return None
    try:
        row = (
            db.query(GalleryImage)
            .filter(
                and_(
                    GalleryImage.filename.endswith(
                        "/" + name, autoescape=True
                    ),
                    GalleryImage.is_active == True,  # noqa: E712
                )
            )
            .first()
        )
    except Exception as e:
        logger.warning("Gallery s3 row lookup failed for %r: %s", name, e)
        return None
    stored = getattr(row, "filename", None)
    return stored if is_s3_uri(stored) else None


# ---------------------------------------------------------------------------
# One-shot migration: local gallery assets -> object storage
# ---------------------------------------------------------------------------

def migrate_gallery_assets_to_backend(
    *, dry_run: bool = False, limit: Optional[int] = None
) -> Dict[str, Any]:
    """Move EXISTING local gallery assets into the object backend.

    Per-file safe order: put + verify -> re-stat the local file -> update
    the row's filename to the s3:// URI -> unlink the local file. A failed
    put leaves file AND row untouched, so the run is idempotent and
    resumable (re-run skips rows already migrated and retries the
    failures). URLs never change: the object key reuses the original
    filename, so gallery_url_name of the new value equals the old one.
    The re-stat guards against a replace/rotate racing the migration: if
    the file changed (size/mtime) since it was read, that file is aborted
    — row and file stay local, and the next run re-reads the new bytes.
    On an unlink failure the row is already migrated: the STALE local
    file then shadows the object on every read (local-first dispatch), so
    run the migration in a quiet window and delete stragglers manually if
    the summary reports unlink failures.

    The summary also reports orphan local files (bytes in the gallery dir
    with NO row behind them) — the rows-only migration cannot move them;
    the operator decides. `limit` consumes moved+skipped budget, so
    small-limit re-runs can no-op once the leading rows are migrated.

    Runnable via the admin endpoint AND via docker exec (import this
    module and call) — no auth in the function, the HTTP route gates it.

    Requires ODYSSEUS_GALLERY_STORAGE=s3 (guards against uploading to the
    wrong bucket from a mis-wired shell).
    """
    from core.database import GalleryImage, SessionLocal

    summary: Dict[str, Any] = {
        "moved": 0,
        "skipped": 0,
        "bytes_freed": 0,
        "errors": [],
        "dry_run": bool(dry_run),
    }
    if not gallery_storage_enabled():
        summary["error"] = (
            f"{ENV_GALLERY_STORAGE} must be 's3' to migrate gallery assets"
            " (refusing to upload to an unconfigured backend)"
        )
        return summary

    backend = _gallery_backend()
    db = SessionLocal()
    try:
        query = db.query(GalleryImage).filter(GalleryImage.is_active == True)  # noqa: E712
        rows = query.all()
        for img in rows:
            if limit is not None and summary["moved"] + summary["skipped"] >= limit:
                break
            stored = img.filename
            if is_s3_uri(stored):
                summary["skipped"] += 1
                continue
            try:
                path = gallery_local_path(stored)
            except HTTPException as e:
                summary["errors"].append(
                    {"id": img.id, "filename": stored, "error": f"unsafe name: {e.detail}"}
                )
                continue
            if path is None or not path.exists():
                summary["skipped"] += 1
                continue
            try:
                stat_before = path.stat()
                content = path.read_bytes()
                key = _migration_key(path, stored)
                if dry_run:
                    summary["moved"] += 1
                    summary["bytes_freed"] += len(content)
                    continue
                _put_object(backend, key, content, stored)
                info = backend.stat(key)
                if not isinstance(info, dict) or info.get("size") != len(content):
                    raise RuntimeError(
                        f"post-put verification failed (size {info.get('size') if isinstance(info, dict) else None} != {len(content)})"
                    )
                # S1: a replace/rotate racing the migration would leave the
                # object holding stale bytes and the unlink would drop the
                # new ones — abort the file (row stays local; next run
                # re-reads the current bytes).
                stat_after = path.stat()
                if (stat_after.st_mtime_ns, stat_after.st_size) != (
                    stat_before.st_mtime_ns, stat_before.st_size
                ):
                    summary["errors"].append({
                        "id": img.id, "filename": stored,
                        "error": "file changed during migration (size/mtime mismatch); not migrated, retry on next run",
                    })
                    continue
            except Exception as e:
                # File NOT unlinked, row untouched: safe to re-run.
                summary["errors"].append({"id": img.id, "filename": stored, "error": str(e)})
                continue
            img.filename = backend.uri_for_key(key)
            db.commit()
            summary["moved"] += 1
            try:
                path.unlink()
                summary["bytes_freed"] += len(content)
            except OSError as e:
                # S2: freed-bytes only counted on a real unlink.
                summary["errors"].append(
                    {"id": img.id, "filename": stored, "error": f"row migrated but local unlink failed (stale local file shadows the object until removed): {e}"}
                )
        db.commit()
    except Exception as e:
        db.rollback()
        summary["errors"].append({"error": f"migration aborted: {e}"})
    finally:
        db.close()

    # S3: report orphan local files — bytes in the gallery dir with NO row
    # behind them. The rows-only migration above cannot move them (and must
    # not: no row means no URL/ownership to preserve), but the operator
    # needs to know what is still eating the disk.
    summary["orphan_files"] = 0
    summary["orphan_bytes"] = 0
    try:
        from core.database import GalleryImage, SessionLocal

        _db = SessionLocal()
        try:
            _row_names = {
                gallery_url_name(row[0])
                for row in _db.query(GalleryImage.filename).all()
                if row[0]
            }
        finally:
            _db.close()
        _dir = _local_gallery_dir()
        if _dir.is_dir():
            for _entry in _dir.iterdir():
                if not _entry.is_file() or _entry.name.startswith("."):
                    continue
                if _entry.name in _row_names:
                    continue  # a row still backs this file (local or failed)
                summary["orphan_files"] += 1
                try:
                    summary["orphan_bytes"] += _entry.stat().st_size
                except OSError:
                    pass
    except Exception as e:
        logger.warning("Gallery migration orphan scan failed: %s", e)
    return summary


def _migration_key(path: Path, stored: str) -> str:
    """Object key for a migrated asset: gallery/<file-mtime date>/<name>.

    The original filename is kept so URLs (and chat-history references)
    survive unchanged; the date shard uses the file's mtime so repeated
    runs target the same key.
    """
    name = stored.rsplit("/", 1)[-1]
    try:
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    except Exception:
        mtime = datetime.now(timezone.utc)
    return f"{GALLERY_KEY_PREFIX}/{mtime.strftime('%Y/%m/%d')}/{name}"
