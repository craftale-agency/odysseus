"""Tests for src.storage_cache — CachedS3Storage write-through LRU read cache.

Repo style: pytest + fakes, no moto and no real MinIO. The fake inner
backend is the same in-memory S3Backend subclass pattern used by
tests/test_upload_handler_s3.py, extended with call counters so cache
hit/miss behaviour is observable without touching the network.
"""
import io
import json
import os
from pathlib import Path

import pytest

from src import storage_backend
from src.storage_backend import (
    DEFAULT_S3_CACHE_MAX_BYTES,
    ENV_S3_BUCKET,
    ENV_S3_CACHE_DIR,
    ENV_S3_CACHE_MAX_BYTES,
    ENV_S3_ENDPOINT,
    ENV_S3_ACCESS_KEY,
    ENV_S3_SECRET_KEY,
    S3Backend,
    get_storage_backend,
    reset_storage_backend,
    validate_storage_backend_at_boot,
)
from src.storage_cache import CachedS3Storage

BUCKET = "odysseus-test"

ALL_S3_ENV = {
    storage_backend.ENV_STORAGE_BACKEND: "s3",
    ENV_S3_ENDPOINT: "https://minio.example.com",
    ENV_S3_ACCESS_KEY: "AKIAEXAMPLE",
    ENV_S3_SECRET_KEY: "secretexample",
    ENV_S3_BUCKET: "odysseus-uploads",
    storage_backend.ENV_S3_REGION: "us-east-1",
    storage_backend.ENV_S3_PATH_STYLE: "true",
}


class FakeS3Backend(S3Backend):
    """In-memory object store: real URI/key semantics, no network, no boto3."""

    def __init__(self, bucket=BUCKET):
        super().__init__(bucket=bucket)
        self.objects: dict[str, bytes] = {}
        self.content_types: dict[str, str] = {}
        self.get_calls: list[str] = []
        self.put_calls: list[tuple] = []

    def put(self, key, fileobj, content_type):
        self.objects[key] = fileobj.read()
        self.content_types[key] = content_type
        self.put_calls.append((key, content_type))

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
        return {
            "size": len(self.objects[key]),
            "content_type": self.content_types.get(key),
            "last_modified": None,
        }

    def list_keys(self):
        return iter(sorted(self.objects))


class _Unseekable:
    """Minimal non-seekable byte source (read-only stream semantics)."""

    def __init__(self, data: bytes):
        self._buf = io.BytesIO(data)

    def read(self, n=-1):
        return self._buf.read(n)


@pytest.fixture(autouse=True)
def _clean_env_and_singleton(monkeypatch):
    reset_storage_backend()
    monkeypatch.setattr(storage_backend, "_URI_READ_BACKEND", None)
    for name in list(ALL_S3_ENV) + [ENV_S3_CACHE_DIR, ENV_S3_CACHE_MAX_BYTES]:
        monkeypatch.delenv(name, raising=False)
    yield
    reset_storage_backend()
    monkeypatch.setattr(storage_backend, "_URI_READ_BACKEND", None)


def _make_cached(tmp_path: Path, max_bytes=None, fake=None):
    cache_dir = tmp_path / "s3-cache"
    wrapper = CachedS3Storage(
        fake or FakeS3Backend(),
        cache_dir=str(cache_dir),
        **({"max_bytes": max_bytes} if max_bytes is not None else {}),
    )
    return wrapper


def _cache_files(wrapper: CachedS3Storage) -> set[str]:
    try:
        return set(os.listdir(wrapper.cache_dir))
    except OSError:
        return set()


def _index_on_disk(wrapper: CachedS3Storage) -> dict:
    path = Path(wrapper.cache_dir) / "cache-index.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Opt-in / passthrough (disabled by default)
# ---------------------------------------------------------------------------

def test_cache_disabled_by_default_is_plain_s3_backend(monkeypatch, tmp_path):
    for name, value in ALL_S3_ENV.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(storage_backend, "S3Backend", FakeS3Backend)

    backend = get_storage_backend()
    assert isinstance(backend, S3Backend)
    assert not isinstance(backend, CachedS3Storage)
    assert backend.is_s3 is True

    # Full byte roundtrip without any cache artifact anywhere on disk.
    key = "2026/09/09/" + "a" * 32 + ".png"
    backend.put(key, io.BytesIO(b"data"), "image/png")
    assert backend.get_bytes(key) == b"data"
    matches = [
        p for p in tmp_path.rglob("*")
        if p.name == "cache-index.json" or "cache" in p.name
    ]
    assert matches == []


def test_factory_env_read_at_call_time_for_cache(monkeypatch, tmp_path):
    for name, value in ALL_S3_ENV.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(storage_backend, "S3Backend", FakeS3Backend)

    plain = get_storage_backend()
    assert isinstance(plain, FakeS3Backend)

    monkeypatch.setenv(ENV_S3_CACHE_DIR, str(tmp_path / "s3-cache"))
    reset_storage_backend()
    wrapped = get_storage_backend()
    assert isinstance(wrapped, CachedS3Storage)
    assert isinstance(wrapped.inner, FakeS3Backend)


def test_factory_wrapped_backend_keeps_s3_identity(monkeypatch, tmp_path):
    for name, value in ALL_S3_ENV.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv(ENV_S3_CACHE_DIR, str(tmp_path / "s3-cache"))

    backend = get_storage_backend()
    assert isinstance(backend, CachedS3Storage)
    # Subclass contract: every existing isinstance/is_s3 check keeps working.
    assert isinstance(backend, S3Backend)
    assert backend.is_s3 is True
    assert backend.bucket == "odysseus-uploads"
    assert backend.max_bytes == DEFAULT_S3_CACHE_MAX_BYTES
    # The s3:// row read path must go through the cache, not a second client.
    assert storage_backend._s3_backend_for_uri() is backend
    uri = backend.uri_for_key("2026/09/09/" + "b" * 32 + ".png")
    assert uri == "s3://odysseus-uploads/2026/09/09/" + "b" * 32 + ".png"
    assert backend.key_for_uri(uri) == "2026/09/09/" + "b" * 32 + ".png"
    assert backend.owns_uri(uri)


def test_factory_reads_max_bytes_env(monkeypatch, tmp_path):
    for name, value in ALL_S3_ENV.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv(ENV_S3_CACHE_DIR, str(tmp_path / "s3-cache"))
    monkeypatch.setenv(ENV_S3_CACHE_MAX_BYTES, "4096")
    backend = get_storage_backend()
    assert isinstance(backend, CachedS3Storage)
    assert backend.max_bytes == 4096


# ---------------------------------------------------------------------------
# put: tee to both stores
# ---------------------------------------------------------------------------

def test_put_writes_s3_and_cache(tmp_path):
    fake = FakeS3Backend()
    wrapper = _make_cached(tmp_path, fake=fake)
    key = "2026/09/09/" + "a" * 32 + ".bin"

    wrapper.put(key, io.BytesIO(b"payload-bytes"), "application/octet-stream")

    assert fake.objects[key] == b"payload-bytes"
    assert fake.content_types[key] == "application/octet-stream"
    assert wrapper.exists(key)
    # Cache file present under the documented flat __ mapping.
    expected = tmp_path / "s3-cache" / key.replace("/", "__")
    assert expected.read_bytes() == b"payload-bytes"
    assert key in _index_on_disk(wrapper)
    assert _index_on_disk(wrapper)[key]["content_type"] == "application/octet-stream"


def test_put_large_payload_streams_without_full_buffer(tmp_path):
    fake = FakeS3Backend()
    wrapper = _make_cached(tmp_path, fake=fake)
    key = "2026/09/09/" + "c" * 32 + ".bin"
    data = os.urandom(300_000)  # several CHUNK_BYTES spans

    wrapper.put(key, io.BytesIO(data), "application/octet-stream")

    assert fake.objects[key] == data
    assert (Path(wrapper.cache_dir) / key.replace("/", "__")).read_bytes() == data


def test_s3_put_failure_cleans_cache_orphan(tmp_path):
    fake = FakeS3Backend()

    def _boom(key, fileobj, content_type):
        raise RuntimeError("simulated S3 outage")

    fake.put = _boom
    wrapper = _make_cached(tmp_path, fake=fake)
    key = "2026/09/09/" + "d" * 32 + ".bin"

    with pytest.raises(RuntimeError, match="S3 outage"):
        wrapper.put(key, io.BytesIO(b"orphan-candidate"), "text/plain")

    assert _cache_files(wrapper) == set()  # no file, no index, no leftover tmp


def test_cache_open_failure_upload_still_succeeds(monkeypatch, tmp_path):
    fake = FakeS3Backend()
    wrapper = _make_cached(tmp_path, fake=fake)

    def _no_temp(*args, **kwargs):
        raise OSError("disk quota exceeded")

    monkeypatch.setattr("src.storage_cache.tempfile.mkstemp", _no_temp)
    key = "2026/09/09/" + "e" * 32 + ".bin"
    source = io.BytesIO(b"must-reach-s3")

    wrapper.put(key, source, "text/plain")  # must NOT raise

    assert fake.objects[key] == b"must-reach-s3"
    assert _cache_files(wrapper) == set()


def test_cache_midstream_failure_upload_still_succeeds_unseekable(monkeypatch, tmp_path):
    """Disk dies halfway through the spill with an unseekable source: the
    already-spilled prefix must be chained onto the unread remainder so S3
    still receives the complete stream."""
    fake = FakeS3Backend()
    wrapper = _make_cached(tmp_path, fake=fake)
    data = b"0123456789abcdef"
    key = "2026/09/09/" + "f" * 32 + ".bin"

    def _partial_spill(fd, fileobj):
        prefix = fileobj.read(8)  # consume half the stream like a real spill
        os.write(fd, prefix)
        os.close(fd)
        raise OSError("disk full mid-write")

    monkeypatch.setattr(wrapper, "_spill_to_cache", _partial_spill)

    wrapper.put(key, _Unseekable(data), "text/plain")  # must NOT raise

    assert fake.objects[key] == data  # prefix + remainder reassembled
    assert _cache_files(wrapper) == set()


def test_cache_midstream_failure_upload_still_succeeds_seekable(monkeypatch, tmp_path):
    """Same failure with a seekable source: seek(0) restart, direct upload."""
    fake = FakeS3Backend()
    wrapper = _make_cached(tmp_path, fake=fake)
    data = b"0123456789abcdef"
    key = "2026/09/09/" + "1" * 32 + ".bin"

    def _partial_spill(fd, fileobj):
        fileobj.read(8)
        os.close(fd)
        raise OSError("disk full mid-write")

    monkeypatch.setattr(wrapper, "_spill_to_cache", _partial_spill)

    source = io.BytesIO(data)
    wrapper.put(key, source, "text/plain")

    assert fake.objects[key] == data
    assert _cache_files(wrapper) == set()


# ---------------------------------------------------------------------------
# get: hit / miss / populate
# ---------------------------------------------------------------------------

def test_get_miss_populates_cache_then_hits(tmp_path):
    fake = FakeS3Backend()
    wrapper = _make_cached(tmp_path, fake=fake)
    key = "2026/09/09/" + "2" * 32 + ".bin"
    fake.objects[key] = b"cold-object"
    fake.content_types[key] = "application/x-cold"

    assert wrapper.get_bytes(key) == b"cold-object"
    assert len(fake.get_calls) == 1
    assert (Path(wrapper.cache_dir) / key.replace("/", "__")).read_bytes() == b"cold-object"

    # Second read is served locally: no additional S3 GET.
    assert wrapper.get_bytes(key) == b"cold-object"
    assert len(fake.get_calls) == 1

    # get_stream also hits locally once populated.
    assert b"".join(wrapper.get_stream(key)) == b"cold-object"
    assert len(fake.get_calls) == 1


def test_get_stream_cold_populates_then_serves_local(tmp_path):
    fake = FakeS3Backend()
    wrapper = _make_cached(tmp_path, fake=fake)
    key = "2026/09/09/" + "3" * 32 + ".bin"
    fake.objects[key] = b"stream-me"

    assert b"".join(wrapper.get_stream(key)) == b"stream-me"
    # One HEAD (stat) + one GET for the populate; the served stream is local.
    assert len(fake.get_calls) == 1
    assert (Path(wrapper.cache_dir) / key.replace("/", "__")).read_bytes() == b"stream-me"
    assert b"".join(wrapper.get_stream(key)) == b"stream-me"
    assert len(fake.get_calls) == 1


def test_get_missing_object_raises_from_s3(tmp_path):
    fake = FakeS3Backend()
    wrapper = _make_cached(tmp_path, fake=fake)
    with pytest.raises(FileNotFoundError):
        wrapper.get_bytes("2026/09/09/" + "4" * 32 + ".bin")
    assert _cache_files(wrapper) == set()


def test_stat_prefers_cache_and_keeps_content_type(tmp_path):
    fake = FakeS3Backend()
    wrapper = _make_cached(tmp_path, fake=fake)
    key = "2026/09/09/" + "5" * 32 + ".png"
    wrapper.put(key, io.BytesIO(b"pngbytes"), "image/png")
    fake.objects.pop(key)  # S3 copy vanishes: stat must still answer locally

    info = wrapper.stat(key)
    assert info is not None
    assert info["size"] == len(b"pngbytes")
    assert info["content_type"] == "image/png"
    assert wrapper.exists(key) is True


def test_stat_falls_back_to_s3_when_not_cached(tmp_path):
    fake = FakeS3Backend()
    wrapper = _make_cached(tmp_path, fake=fake)
    key = "2026/09/09/" + "6" * 32 + ".bin"
    fake.objects[key] = b"not-cached"
    assert wrapper.stat(key) == fake.stat(key)


# ---------------------------------------------------------------------------
# delete: both stores
# ---------------------------------------------------------------------------

def test_delete_removes_s3_and_cache(tmp_path):
    fake = FakeS3Backend()
    wrapper = _make_cached(tmp_path, fake=fake)
    key = "2026/09/09/" + "7" * 32 + ".bin"
    wrapper.put(key, io.BytesIO(b"doomed"), "text/plain")

    wrapper.delete(key)

    assert key not in fake.objects
    assert not (Path(wrapper.cache_dir) / key.replace("/", "__")).exists()
    assert key not in _index_on_disk(wrapper)


def test_delete_cache_failure_is_warning_only(tmp_path):
    fake = FakeS3Backend()
    wrapper = _make_cached(tmp_path, fake=fake)
    key = "2026/09/09/" + "8" * 32 + ".bin"
    wrapper.put(key, io.BytesIO(b"locked-in"), "text/plain")

    os.chmod(wrapper.cache_dir, 0o500)  # unlink now fails with PermissionError
    try:
        wrapper.delete(key)  # must NOT raise
    finally:
        os.chmod(wrapper.cache_dir, 0o700)
    assert key not in fake.objects  # S3 delete still happened


# ---------------------------------------------------------------------------
# Eviction / LRU
# ---------------------------------------------------------------------------

def test_lru_eviction_drops_oldest_keeps_newest(tmp_path):
    fake = FakeS3Backend()
    wrapper = _make_cached(tmp_path, max_bytes=30, fake=fake)

    def _put(name: str) -> str:
        key = f"2026/09/09/{name * 32}.bin"
        wrapper.put(key, io.BytesIO(b"x" * 10), "text/plain")
        return key

    oldest, middle = _put("a"), _put("b")
    newest = _put("c")  # 30 bytes stored == cap: nothing evicted yet
    assert all(k in _index_on_disk(wrapper) for k in (oldest, middle, newest))

    _put("d")  # 40 bytes > 30: oldest (a) must go
    index = _index_on_disk(wrapper)
    assert oldest not in index
    assert middle in index and newest in index
    assert not (Path(wrapper.cache_dir) / oldest.replace("/", "__")).exists()
    assert fake.objects[oldest] == b"x" * 10  # S3 untouched by eviction


def test_atime_update_changes_eviction_order(tmp_path):
    fake = FakeS3Backend()
    wrapper = _make_cached(tmp_path, max_bytes=20, fake=fake)
    first = "2026/09/09/" + "a" * 32 + ".bin"
    second = "2026/09/09/" + "b" * 32 + ".bin"
    wrapper.put(first, io.BytesIO(b"x" * 10), "text/plain")
    wrapper.put(second, io.BytesIO(b"y" * 10), "text/plain")

    # Touch the OLDER entry so the newer one becomes the eviction victim.
    wrapper.get_bytes(first)
    third = "2026/09/09/" + "c" * 32 + ".bin"
    wrapper.put(third, io.BytesIO(b"z" * 10), "text/plain")

    index = _index_on_disk(wrapper)
    assert second not in index  # older atime after the touch on `first`
    assert first in index and third in index


def test_oversize_object_skipped_on_put_and_read(tmp_path):
    fake = FakeS3Backend()
    wrapper = _make_cached(tmp_path, max_bytes=10, fake=fake)
    key = "2026/09/09/" + "9" * 32 + ".bin"
    big = b"z" * 40

    wrapper.put(key, io.BytesIO(big), "text/plain")
    assert fake.objects[key] == big
    assert _cache_files(wrapper) == set()  # cap-skipped, no orphan tmp

    # A cold read of an oversize object is served but never cached.
    fake2_key = "2026/09/09/" + "0" * 32 + ".bin"
    fake.objects[fake2_key] = big
    assert wrapper.get_bytes(fake2_key) == big
    assert (Path(wrapper.cache_dir) / fake2_key.replace("/", "__")).exists() is False
    assert b"".join(wrapper.get_stream(fake2_key)) == big
    assert (Path(wrapper.cache_dir) / fake2_key.replace("/", "__")).exists() is False


def test_eviction_respects_touched_order_across_instances(tmp_path):
    """atime persistence: eviction order survives a restart (index reloaded)."""
    fake = FakeS3Backend()
    wrapper = _make_cached(tmp_path, max_bytes=20, fake=fake)
    first = "2026/09/09/" + "a" * 32 + ".bin"
    second = "2026/09/09/" + "b" * 32 + ".bin"
    wrapper.put(first, io.BytesIO(b"x" * 10), "text/plain")
    wrapper.put(second, io.BytesIO(b"y" * 10), "text/plain")
    wrapper.get_bytes(first)  # first is now the hot entry

    reborn = CachedS3Storage(fake, cache_dir=wrapper.cache_dir, max_bytes=20)
    third = "2026/09/09/" + "c" * 32 + ".bin"
    reborn.put(third, io.BytesIO(b"z" * 10), "text/plain")

    index = _index_on_disk(reborn)
    assert second not in index
    assert first in index and third in index


# ---------------------------------------------------------------------------
# Index resilience
# ---------------------------------------------------------------------------

def test_index_rebuild_after_corruption(tmp_path):
    fake = FakeS3Backend()
    wrapper = _make_cached(tmp_path, max_bytes=20, fake=fake)
    keys = []
    for name in ("a", "b"):
        key = f"2026/09/09/{name * 32}.bin"
        wrapper.put(key, io.BytesIO(b"x" * 10), "text/plain")
        keys.append(key)
    assert len(_index_on_disk(wrapper)) == 2

    (Path(wrapper.cache_dir) / "cache-index.json").write_text("{not json", encoding="utf-8")

    reborn = CachedS3Storage(fake, cache_dir=wrapper.cache_dir, max_bytes=20)
    # Hits still work (files are the fast path) and the index self-heals.
    assert reborn.get_bytes(keys[0]) == b"x" * 10
    assert reborn.get_bytes(keys[1]) == b"x" * 10
    assert len(fake.get_calls) == 0

    # Eviction still functions on the rebuilt index (mtimes as atimes).
    third = "2026/09/09/" + "c" * 32 + ".bin"
    reborn.put(third, io.BytesIO(b"z" * 10), "text/plain")
    index = _index_on_disk(reborn)
    assert keys[0] not in index  # oldest file mtime evicted first
    assert keys[1] in index and third in index


def test_index_missing_rebuilds_from_dir(tmp_path):
    fake = FakeS3Backend()
    wrapper = _make_cached(tmp_path, fake=fake)
    key = "2026/09/09/" + "a" * 32 + ".bin"
    wrapper.put(key, io.BytesIO(b"kept"), "text/plain")
    os.remove(Path(wrapper.cache_dir) / "cache-index.json")

    reborn = CachedS3Storage(fake, cache_dir=wrapper.cache_dir)
    assert reborn.get_bytes(key) == b"kept"
    assert len(fake.get_calls) == 0  # served from the rebuilt cache
    assert key in _index_on_disk(reborn)


def test_stale_index_entry_without_file_is_dropped(tmp_path):
    fake = FakeS3Backend()
    wrapper = _make_cached(tmp_path, fake=fake)
    wrapper.put("2026/09/09/" + "a" * 32 + ".bin", io.BytesIO(b"real"), "text/plain")

    ghost = "2026/09/09/" + "g" * 32 + ".bin"
    index = _index_on_disk(wrapper)
    index[ghost] = {"size": 123, "atime": 1.0}
    (Path(wrapper.cache_dir) / "cache-index.json").write_text(
        json.dumps(index), encoding="utf-8"
    )

    reborn = CachedS3Storage(fake, cache_dir=wrapper.cache_dir)
    assert ghost not in reborn._load_index_locked()


def test_cache_dir_is_created_lazily(tmp_path):
    fake = FakeS3Backend()
    cache_dir = tmp_path / "nested" / "s3-cache"
    wrapper = CachedS3Storage(fake, cache_dir=str(cache_dir), max_bytes=100)
    assert not cache_dir.exists()  # nothing touches disk until first write
    key = "2026/09/09/" + "a" * 32 + ".bin"
    wrapper.put(key, io.BytesIO(b"lazy"), "text/plain")
    assert (cache_dir / key.replace("/", "__")).read_bytes() == b"lazy"


# ---------------------------------------------------------------------------
# resolve_cache_max_bytes / boot validation
# ---------------------------------------------------------------------------

def test_resolve_cache_max_bytes_defaults_and_parses(monkeypatch):
    from src.storage_cache import resolve_cache_max_bytes

    assert resolve_cache_max_bytes("") == DEFAULT_S3_CACHE_MAX_BYTES
    assert resolve_cache_max_bytes("  ") == DEFAULT_S3_CACHE_MAX_BYTES
    monkeypatch.setenv(ENV_S3_CACHE_MAX_BYTES, "1024")
    assert resolve_cache_max_bytes() == 1024
    assert resolve_cache_max_bytes("2048") == 2048


@pytest.mark.parametrize("bad", ["nope", "10kb", "0", "-5", "1.5"])
def test_resolve_cache_max_bytes_rejects_garbage(bad):
    from src.storage_cache import resolve_cache_max_bytes

    with pytest.raises(RuntimeError, match=ENV_S3_CACHE_MAX_BYTES):
        resolve_cache_max_bytes(bad)


def _set_s3_env(monkeypatch, **extra):
    for name, value in ALL_S3_ENV.items():
        monkeypatch.setenv(name, value)
    for name, value in extra.items():
        monkeypatch.setenv(name, value)


def test_boot_validation_ok_with_cache_dir(monkeypatch, tmp_path):
    cache_dir = tmp_path / "s3-cache"
    _set_s3_env(monkeypatch, **{ENV_S3_CACHE_DIR: str(cache_dir)})
    validate_storage_backend_at_boot()
    assert cache_dir.is_dir()  # created + write-probed


def test_boot_validation_unwritable_cache_dir_fails_fast(monkeypatch, tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file where a dir should be", encoding="utf-8")
    _set_s3_env(
        monkeypatch,
        **{ENV_S3_CACHE_DIR: str(blocker / "s3-cache")},
    )
    with pytest.raises(RuntimeError, match=ENV_S3_CACHE_DIR):
        validate_storage_backend_at_boot()


def test_boot_validation_rejects_bad_max_bytes(monkeypatch, tmp_path):
    _set_s3_env(
        monkeypatch,
        **{
            ENV_S3_CACHE_DIR: str(tmp_path / "s3-cache"),
            ENV_S3_CACHE_MAX_BYTES: "two-gig",
        },
    )
    with pytest.raises(RuntimeError, match=ENV_S3_CACHE_MAX_BYTES):
        validate_storage_backend_at_boot()


def test_boot_validation_cache_dir_ignored_for_local_backend(monkeypatch, tmp_path):
    # Cache env set but backend=local: inert, must not fail boot.
    monkeypatch.setenv(storage_backend.ENV_STORAGE_BACKEND, "local")
    monkeypatch.setenv(ENV_S3_CACHE_DIR, str(tmp_path / "unused"))
    assert validate_storage_backend_at_boot() is None
