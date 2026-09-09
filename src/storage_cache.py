"""
storage_cache.py

Write-through LRU read cache wrapping S3Backend.

WHY: when Odysseus runs on a remote host and S3 (MinIO) sits behind a thin
upstream (home connection), uploads are cheap (download direction on the
MinIO side) but every GET re-pulls the bytes over the WAN. The upload path
already has the bytes in hand, so it tees them into a local cache file;
reads of fresh/hot objects then run at local-disk speed and only cold or
evicted objects cross the WAN.

Opt-in envs (read at call time like every other storage env, see
src/storage_backend.py):

    ODYSSEUS_S3_CACHE_DIR        cache root; empty/absent = cache DISABLED
                                (get_storage_backend returns the plain
                                S3Backend — zero behavior change)
    ODYSSEUS_S3_CACHE_MAX_BYTES  eviction cap in bytes, default 2 GiB

Placement contract — the env IS the contract, nothing is hardcoded, but:

* The dir must be writable and (in Docker) live under the persistent data
  volume, e.g. ``/app/data/s3-cache`` for the stock compose (the
  odysseus-data named volume) so cached bytes survive container restarts.
* The dir must NOT be inside UPLOAD_DIR (DATA_DIR/uploads): the local
  upload-cleanup walker enumerates that tree and cache files must never be
  mistaken for upload bytes. A sibling like ``<DATA_DIR>/s3-cache`` is the
  recommended shape.

Key <-> file mapping: flat filenames, every ``/`` in the object key becomes
``__`` (``2026/09/09/<uuid>.png`` -> ``2026__09__09__<uuid>.png``). Chosen
over nested mirror dirs because eviction/removal then never needs directory
pruning and an index rebuild is a single flat walk. Upload object keys are
date-sharded UUID names and never contain ``__``, so the mapping cannot
collide in practice; a key literally containing ``__`` would share a cache
file with its ``/``-spelled twin (worst case: an extra cache miss).

Index: ``cache-index.json`` in the cache dir, ``{key: {size, atime}}`` plus
an optional ``content_type`` row field so stat() served from cache stays
faithful to what S3 reports (put knows it exactly; read-populate guesses
from the key extension). Persisted with the same atomic-write pattern as
UploadHandler._atomic_write_json (temp file in the same directory + fsync +
os.replace); that helper is an instance method there, so it is replicated
here rather than imported. A corrupt or missing index is rebuilt by walking
the dir (atime falls back to file mtime); index rows whose file is gone are
dropped on load.

Failure semantics (the two invariants callers rely on):

* S3 put failure NEVER leaves a cache-only orphan — the temp copy is
  removed and the error propagates (S3 remains the source of truth).
* Cache write failure NEVER fails the upload — it is logged and the S3 put
  still receives the complete byte stream (seek(0) restart when possible,
  otherwise the already-spilled prefix is chained with the unread
  remainder).

Locking: one threading.RLock guards index state and cache-file placement.
S3 network calls always run OUTSIDE the lock; only local file ops and the
small index JSON are inside it.
"""

import json
import logging
import mimetypes
import os
import tempfile
import threading
import time
from typing import Any, Dict, Iterator, Optional

from src.storage_backend import (
    DEFAULT_S3_CACHE_MAX_BYTES,
    ENV_S3_CACHE_MAX_BYTES,
    S3Backend,
)

logger = logging.getLogger(__name__)

INDEX_FILENAME = "cache-index.json"
# Spill/read chunk size: same magnitude as the backend streaming loops.
CHUNK_BYTES = 65536


def resolve_cache_max_bytes(raw: Optional[str] = None) -> int:
    """Parse ODYSSEUS_S3_CACHE_MAX_BYTES (>= 1) with the 2 GiB default.

    Shared by CachedS3Storage and validate_storage_backend_at_boot so both
    fail with the same message on garbage input.
    """
    if raw is None:
        raw = os.getenv(ENV_S3_CACHE_MAX_BYTES, "")
    raw = (raw or "").strip()
    if not raw:
        return DEFAULT_S3_CACHE_MAX_BYTES
    try:
        value = int(raw)
    except ValueError:
        raise RuntimeError(
            f"{ENV_S3_CACHE_MAX_BYTES} must be an integer number of bytes "
            f"(got {raw!r})"
        ) from None
    if value < 1:
        raise RuntimeError(
            f"{ENV_S3_CACHE_MAX_BYTES} must be >= 1 byte (got {value})"
        )
    return value


class _ChainedReader:
    """Fileobj serving the already-spilled prefix from *path* followed by
    the unread remainder of *rest* — used when a cache write failed midway
    so the S3 upload still receives the complete stream. Implements plain
    read() semantics (a single read() drains across the boundary), which is
    what both boto3's uploader and naive readers expect."""

    def __init__(self, path: str, rest):
        self._rest = rest
        self._file = open(path, "rb")

    def read(self, n: int = -1):
        if n is None or n < 0:
            prefix = self._file.read() or b""
            self._file.close()
            return prefix + self._rest.read()
        parts = []
        need = n
        while need > 0:
            if not self._file.closed:
                chunk = self._file.read(need)
                if chunk:
                    parts.append(chunk)
                    need -= len(chunk)
                    continue
                self._file.close()
            chunk = self._rest.read(need)
            if not chunk:
                break
            parts.append(chunk)
            need -= len(chunk)
        return b"".join(parts)


class CachedS3Storage(S3Backend):
    """S3Backend + write-through LRU local read cache.

    Subclasses S3Backend so every S3-specific accessor stays inherited and
    isinstance(active, S3Backend) checks (e.g. _s3_backend_for_uri) keep
    matching this wrapper — read_attachment_bytes and friends get cached
    reads for free. Byte-path methods are overridden and route S3 traffic
    to the wrapped ``inner`` instance (one shared boto3 client, one
    configuration snapshot); pure S3 metadata calls delegate too so the
    wrapper never diverges from the backend it fronts.
    """

    def __init__(
        self,
        inner: S3Backend,
        cache_dir: str,
        max_bytes: Optional[int] = None,
    ):
        super().__init__(
            endpoint_url=inner.endpoint_url,
            access_key=inner.access_key,
            secret_key=inner.secret_key,
            bucket=inner.bucket,
            region=inner.region,
            path_style=inner.path_style,
        )
        self.inner = inner
        self.cache_dir = cache_dir
        self.max_bytes = (
            max_bytes if max_bytes is not None
            else resolve_cache_max_bytes()
        )
        self._lock = threading.RLock()
        self._index: Optional[Dict[str, Dict[str, Any]]] = None

    # -- S3 surface delegation (S3 stays the source of truth) -------------

    @property
    def is_s3(self) -> bool:
        return True

    def client(self):
        return self.inner.client()

    def uri_for_key(self, key: str) -> str:
        return self.inner.uri_for_key(key)

    def key_for_uri(self, uri: str):
        return self.inner.key_for_uri(uri)

    def owns_uri(self, uri: str) -> bool:
        return self.inner.owns_uri(uri)

    def list_keys(self) -> Iterator[str]:
        """Enumerate the bucket, never the cache (S3 is the source of truth)."""
        yield from self.inner.list_keys()

    # -- cache plumbing ---------------------------------------------------

    def _cache_path(self, key: str) -> str:
        return os.path.join(self.cache_dir, key.replace("/", "__"))

    def _index_path(self) -> str:
        return os.path.join(self.cache_dir, INDEX_FILENAME)

    def _atomic_write_index(self, index: Dict[str, Dict[str, Any]]) -> None:
        """UploadHandler._atomic_write_json pattern, replicated (the original
        is an instance method there and not importable): temp file in the
        same directory, fsync, atomic os.replace."""
        fd, tmp = tempfile.mkstemp(
            prefix=".cache-index-", suffix=".tmp", dir=self.cache_dir
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(index, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self._index_path())
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def _cached_file_size(self, key: str) -> Optional[int]:
        try:
            return os.path.getsize(self._cache_path(key))
        except OSError:
            return None

    def _load_index_locked(self) -> Dict[str, Dict[str, Any]]:
        """Best-effort index load, memoized. Corrupt/missing index -> rebuild
        by walking the dir (atime falls back to file mtime). Rows whose
        cache file has vanished (crash between file delete and index write,
        manual cleanup) are dropped."""
        if self._index is not None:
            return self._index
        try:
            with open(self._index_path(), "r", encoding="utf-8") as f:
                loaded = json.load(f)
        except (OSError, ValueError) as e:
            logger.warning(
                "S3 cache index unreadable (%s); rebuilding from %s",
                e, self.cache_dir,
            )
            self._index = self._rebuild_index_locked()
            return self._index
        if not isinstance(loaded, dict):
            self._index = self._rebuild_index_locked()
            return self._index
        # A readable index is trusted (an empty one means "everything was
        # evicted"); only rows without a file behind them are dropped.
        self._index = {
            key: row
            for key, row in loaded.items()
            if isinstance(key, str)
            and isinstance(row, dict)
            and self._cached_file_size(key) is not None
        }
        return self._index

    def _rebuild_index_locked(self) -> Dict[str, Dict[str, Any]]:
        """Rebuild the index from the directory contents."""
        rebuilt: Dict[str, Dict[str, Any]] = {}
        try:
            names = os.listdir(self.cache_dir)
        except OSError:
            return rebuilt
        for name in names:
            if name == INDEX_FILENAME or name.startswith(".cache-"):
                continue
            path = os.path.join(self.cache_dir, name)
            try:
                st = os.stat(path)
            except OSError:
                continue
            if not os.path.isfile(path):
                continue
            key = name.replace("__", "/")
            rebuilt[key] = {"size": st.st_size, "atime": st.st_mtime}
        return rebuilt

    def _persist_index_locked(self) -> None:
        try:
            self._atomic_write_index(self._index or {})
        except OSError as e:
            logger.warning("S3 cache: index write failed: %s", e)

    def _commit_cache_file(
        self,
        key: str,
        tmp_path: str,
        content_type: Optional[str],
    ) -> None:
        """Move a finished temp file into place, index it, evict if needed.

        Objects larger than the cap are never indexed (the classic LRU
        pathology: one giant entry evicting everything else); the temp is
        removed instead. Best-effort: a failure here costs a cache entry,
        never the upload/read that produced it.
        """
        try:
            size = os.path.getsize(tmp_path)
            if size > self.max_bytes:
                logger.info(
                    "S3 cache: skipping %r (%d bytes > cap %d)",
                    key, size, self.max_bytes,
                )
                os.unlink(tmp_path)
                return
            os.makedirs(self.cache_dir, exist_ok=True)
            os.replace(tmp_path, self._cache_path(key))
        except OSError as e:
            logger.warning(
                "S3 cache: cannot place cache file for %r: %s", key, e
            )
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            return
        with self._lock:
            index = self._load_index_locked()
            row = {"size": size, "atime": time.time()}
            if content_type:
                row["content_type"] = content_type
            index[key] = row
            self._evict_locked(index)
            self._persist_index_locked()

    def _evict_locked(self, index: Dict[str, Dict[str, Any]]) -> None:
        """While total indexed size exceeds the cap, drop the least recently
        used entries (file first, then row). Unlink failures still drop the
        row: a file we cannot delete must not pin the index forever."""
        total = sum(int(row.get("size") or 0) for row in index.values())
        while total > self.max_bytes and index:
            key = min(index, key=lambda k: index[k].get("atime") or 0.0)
            total -= int(index[key].get("size") or 0)
            index.pop(key)
            try:
                os.unlink(self._cache_path(key))
            except OSError as e:
                logger.warning(
                    "S3 cache: eviction unlink failed for %r: %s", key, e
                )

    def _touch_locked(self, key: str) -> None:
        """Refresh LRU order for a hit and persist it (order survives
        restarts). Self-heals a missing row for an existing file."""
        index = self._load_index_locked()
        row = index.get(key)
        if row is None:
            size = self._cached_file_size(key)
            if size is None:
                return
            row = index[key] = {"size": size}
        row["atime"] = time.time()
        self._persist_index_locked()

    def _store_bytes(self, key: str, data: bytes, content_type: Optional[str]) -> None:
        """Read-through populate path: spill fetched bytes into the cache.
        Best-effort; never raises into the read that triggered it."""
        if len(data) > self.max_bytes:
            logger.info(
                "S3 cache: skipping %r (%d bytes > cap %d)",
                key, len(data), self.max_bytes,
            )
            return
        tmp: Optional[str] = None
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            fd, tmp = tempfile.mkstemp(
                prefix=".cache-populate-", suffix=".tmp", dir=self.cache_dir
            )
            with os.fdopen(fd, "wb") as f:
                f.write(data)
        except OSError as e:
            logger.warning(
                "S3 cache: populate write failed for %r: %s", key, e
            )
            if tmp is not None:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
            return
        self._commit_cache_file(key, tmp, content_type)

    # -- StorageBackend API ------------------------------------------------

    def _spill_to_cache(self, fd: int, fileobj) -> int:
        """Stream fileobj into the cache temp file *fd* in bounded-memory
        chunks; returns the number of bytes written. Raises OSError on cache
        write trouble (disk full, IO error) — the caller decides how to keep
        the upload alive."""
        written = 0
        with os.fdopen(fd, "wb") as out:
            while True:
                chunk = fileobj.read(CHUNK_BYTES)
                if not chunk:
                    break
                out.write(chunk)
                written += len(chunk)
        return written

    def put(self, key: str, fileobj, content_type: str) -> None:
        """Tee the upload: stream the source once into a cache temp file,
        then upload the spilled bytes to S3. S3 failure -> temp removed and
        the error re-raised (no cache-only orphan; the cleanup happens in
        the finally below). Cache failure -> warning only, S3 still gets
        the complete stream."""
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            fd, tmp = tempfile.mkstemp(
                prefix=".cache-put-", suffix=".tmp", dir=self.cache_dir
            )
        except OSError as e:
            # Nothing consumed from fileobj yet: straight passthrough.
            logger.warning(
                "S3 cache: cannot open cache temp for %r, uploading without "
                "cache: %s", key, e,
            )
            self.inner.put(key, fileobj, content_type)
            return

        spilled = 0
        cache_ok = True
        try:
            spilled = self._spill_to_cache(fd, fileobj)
        except OSError as e:
            # Disk trouble mid-stream. fileobj sits right after everything
            # written to tmp, so full content == spilled prefix + remainder.
            cache_ok = False
            logger.warning(
                "S3 cache: write failed mid-upload for %r (%d bytes spilled),"
                " continuing upload without cache: %s", key, spilled, e,
            )

        try:
            if cache_ok:
                with open(tmp, "rb") as body:
                    self.inner.put(key, body, content_type)
                # S3 put failed -> the finally below unlinks tmp: no orphan.
                self._commit_cache_file(key, tmp, content_type)
                tmp = None  # committed (or cap-skipped); nothing to clean
            else:
                try:
                    fileobj.seek(0)
                except (AttributeError, OSError):
                    # Unseekable source: the spilled prefix plus the unread
                    # remainder is still the complete content.
                    self.inner.put(
                        key, _ChainedReader(tmp, fileobj), content_type
                    )
                else:
                    self.inner.put(key, fileobj, content_type)
        finally:
            if tmp is not None:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass

    def get_bytes(self, key: str) -> bytes:
        with self._lock:
            hit = self._cached_file_size(key) is not None
            if hit:
                self._touch_locked(key)
        if hit:
            try:
                with open(self._cache_path(key), "rb") as f:
                    return f.read()
            except OSError as e:
                logger.warning(
                    "S3 cache: hit read failed for %r, falling back to S3: %s",
                    key, e,
                )
        data = self.inner.get_bytes(key)
        if len(data) <= self.max_bytes:
            self._store_bytes(
                key, data, mimetypes.guess_type(key)[0]
            )
        return data

    def _serve_local_chunks(self, key: str):
        """Chunk iterator over the cache file (LocalBackend.get_stream shape:
        the file handle closes when the consumer finishes or disconnects)."""
        f = open(self._cache_path(key), "rb")

        def _chunks():
            try:
                while True:
                    chunk = f.read(CHUNK_BYTES)
                    if not chunk:
                        break
                    yield chunk
            finally:
                f.close()

        return _chunks()

    def get_stream(self, key: str):
        with self._lock:
            hit = self._cached_file_size(key) is not None
            if hit:
                self._touch_locked(key)
        if hit:
            return self._serve_local_chunks(key)
        # Cold read: decide cacheability from the S3 size (one HEAD), then
        # populate first so this AND subsequent reads are local-speed.
        # Oversize/unknown-size objects stream straight from S3.
        try:
            info = self.inner.stat(key)
        except Exception as e:
            logger.warning("S3 cache: stat failed for %r: %s", key, e)
            info = None
        size = int((info or {}).get("size") or 0)
        if info is not None and 0 < size <= self.max_bytes:
            self._populate_stream(key, info)
            with self._lock:
                hit = self._cached_file_size(key) is not None
                if hit:
                    self._touch_locked(key)
            if hit:
                return self._serve_local_chunks(key)
        return self.inner.get_stream(key)

    def _populate_stream(self, key: str, info: Optional[Dict[str, Any]]) -> None:
        try:
            self._store_bytes(
                key,
                self.inner.get_bytes(key),
                (info or {}).get("content_type"),
            )
        except Exception as e:
            logger.warning("S3 cache: populate fetch failed for %r: %s", key, e)

    def exists(self, key: str) -> bool:
        if self._cached_file_size(key) is not None:
            return True
        return self.inner.exists(key)

    def delete(self, key: str) -> None:
        # Cache copy first (failure = warning only), then the authoritative
        # S3 delete whose failures propagate exactly like plain S3Backend.
        path = self._cache_path(key)
        removed = False
        try:
            os.unlink(path)
            removed = True
        except FileNotFoundError:
            removed = True
        except OSError as e:
            logger.warning("S3 cache: cache delete failed for %r: %s", key, e)
        if removed:
            with self._lock:
                if self._load_index_locked().pop(key, None) is not None:
                    self._persist_index_locked()
        self.inner.delete(key)

    def stat(self, key: str) -> Optional[Dict[str, Any]]:
        size = self._cached_file_size(key)
        if size is None:
            return self.inner.stat(key)
        with self._lock:
            row = dict(self._load_index_locked().get(key) or {})
        content_type = row.get("content_type") or mimetypes.guess_type(key)[0]
        try:
            last_modified = os.path.getmtime(self._cache_path(key))
        except OSError:
            last_modified = None
        return {
            "size": size,
            "content_type": content_type,
            "last_modified": last_modified,
        }
