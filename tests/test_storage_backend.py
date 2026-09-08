"""Tests for src.storage_backend — env parsing, factory, boot fail-fast, URI conversion."""
import importlib
import sys

import pytest

from src import storage_backend
from src.storage_backend import (
    ENV_S3_BUCKET,
    ENV_S3_ENDPOINT,
    ENV_S3_ACCESS_KEY,
    ENV_S3_SECRET_KEY,
    LocalBackend,
    S3Backend,
    get_storage_backend,
    is_s3_uri,
    parse_s3_uri,
    read_attachment_bytes,
    reset_storage_backend,
    validate_storage_backend_at_boot,
)


ALL_S3_ENV = {
    ENV_S3_ENDPOINT: "https://minio.example.com",
    ENV_S3_ACCESS_KEY: "AKIAEXAMPLE",
    ENV_S3_SECRET_KEY: "secretexample",
    ENV_S3_BUCKET: "odysseus-uploads",
    storage_backend.ENV_S3_REGION: "us-east-1",
    storage_backend.ENV_S3_PATH_STYLE: "true",
    storage_backend.ENV_STORAGE_BACKEND: "s3",
}


class _Boto3Blocker:
    """Meta-path finder that makes `boto3` unimportable while installed."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] == "boto3":
            raise ModuleNotFoundError("No module named 'boto3' (blocked by test)")
        return None


@pytest.fixture(autouse=True)
def _clean_singleton(monkeypatch):
    reset_storage_backend()
    monkeypatch.setattr(storage_backend, "_URI_READ_BACKEND", None)
    # Default to the local backend for every test unless explicitly overridden.
    for name in ALL_S3_ENV:
        monkeypatch.delenv(name, raising=False)
    yield
    reset_storage_backend()
    monkeypatch.setattr(storage_backend, "_URI_READ_BACKEND", None)


# ---------------------------------------------------------------------------
# URI <-> key conversion
# ---------------------------------------------------------------------------

def test_is_s3_uri():
    assert is_s3_uri("s3://bucket/key")
    assert is_s3_uri("s3://b/2026/09/08/abc.png")
    assert not is_s3_uri("S3://bucket/key")  # case-sensitive scheme
    assert not is_s3_uri("/local/path")
    assert not is_s3_uri("")
    assert not is_s3_uri(None)
    assert not is_s3_uri(123)


def test_parse_s3_uri():
    assert parse_s3_uri("s3://bucket/a/b.png") == ("bucket", "a/b.png")
    assert parse_s3_uri("s3://bucket/key") == ("bucket", "key")
    # Malformed forms return None rather than raising.
    assert parse_s3_uri("s3://bucket") is None          # no key
    assert parse_s3_uri("s3:///key") is None            # no bucket
    assert parse_s3_uri("s3://") is None
    assert parse_s3_uri("s3://bucket/") is None         # empty key
    assert parse_s3_uri("s3://bad bucket/k") is None    # whitespace in bucket
    assert parse_s3_uri("/plain/path") is None
    assert parse_s3_uri("") is None
    assert parse_s3_uri(None) is None


def test_s3_uri_key_roundtrip(monkeypatch):
    backend = S3Backend(bucket="buck")
    uri = backend.uri_for_key("2026/09/08/" + "a" * 32 + ".png")
    assert uri == "s3://buck/2026/09/08/" + "a" * 32 + ".png"
    assert backend.key_for_uri(uri) == "2026/09/08/" + "a" * 32 + ".png"
    # A URI in a different bucket is not ours.
    assert backend.key_for_uri("s3://other/key") is None
    assert not backend.owns_uri("s3://other/key")
    assert backend.owns_uri(uri)


# ---------------------------------------------------------------------------
# Env parsing / factory
# ---------------------------------------------------------------------------

def test_factory_defaults_to_local(monkeypatch):
    backend = get_storage_backend()
    assert isinstance(backend, LocalBackend)
    assert not backend.is_s3


def test_factory_reads_env_at_call_time(monkeypatch, tmp_path):
    # First call resolves local and caches the singleton (chroma_client style).
    first = get_storage_backend()
    assert isinstance(first, LocalBackend)

    # Flipping the env after the singleton exists keeps returning it...
    monkeypatch.setenv(storage_backend.ENV_STORAGE_BACKEND, "s3")
    for name, value in ALL_S3_ENV.items():
        monkeypatch.setenv(name, value)
    assert get_storage_backend() is first

    # ...but after reset, the same call re-reads the env and builds S3.
    reset_storage_backend()
    second = get_storage_backend()
    assert isinstance(second, S3Backend)
    assert second.is_s3
    assert second.bucket == "odysseus-uploads"
    assert second.endpoint_url == "https://minio.example.com"
    assert second.region == "us-east-1"
    assert second.path_style is True

    # Import-time safety: the module must not have captured the env earlier.
    import src.upload_handler as uh
    assert "storage_backend" in dir(uh)  # import works with any env


def test_factory_rejects_unknown_backend_name(monkeypatch):
    monkeypatch.setenv(storage_backend.ENV_STORAGE_BACKEND, "gcs")
    with pytest.raises(RuntimeError, match="ODYSSEUS_STORAGE_BACKEND"):
        get_storage_backend()


def test_factory_blank_env_means_local(monkeypatch):
    monkeypatch.setenv(storage_backend.ENV_STORAGE_BACKEND, "   ")
    assert isinstance(get_storage_backend(), LocalBackend)


def test_s3_defaults_path_style_true_region_fallback(monkeypatch):
    for name, value in ALL_S3_ENV.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv(storage_backend.ENV_S3_REGION, raising=False)
    monkeypatch.delenv(storage_backend.ENV_S3_PATH_STYLE, raising=False)
    backend = S3Backend()
    assert backend.region == "us-east-1"
    assert backend.path_style is True
    monkeypatch.setenv(storage_backend.ENV_S3_PATH_STYLE, "false")
    assert S3Backend().path_style is False


def test_s3_path_style_falsy_value_parsing(monkeypatch):
    for name, value in ALL_S3_ENV.items():
        monkeypatch.setenv(name, value)
    for falsy in ("0", " off ", "No", "FALSE"):
        monkeypatch.setenv(storage_backend.ENV_S3_PATH_STYLE, falsy)
        assert S3Backend().path_style is False, falsy
    for truthy in ("1", "true", " yes ", "ON"):
        monkeypatch.setenv(storage_backend.ENV_S3_PATH_STYLE, truthy)
        assert S3Backend().path_style is True, truthy


# ---------------------------------------------------------------------------
# Boot fail-fast matrix
# ---------------------------------------------------------------------------

def test_boot_validation_noop_for_local(monkeypatch):
    validate_storage_backend_at_boot()  # unset env
    monkeypatch.setenv(storage_backend.ENV_STORAGE_BACKEND, "local")
    validate_storage_backend_at_boot()


def test_boot_validation_missing_each_env_var(monkeypatch):
    for missing in (ENV_S3_ENDPOINT, ENV_S3_ACCESS_KEY, ENV_S3_SECRET_KEY, ENV_S3_BUCKET):
        for name, value in ALL_S3_ENV.items():
            monkeypatch.setenv(name, value)
        monkeypatch.delenv(missing, raising=False)
        with pytest.raises(RuntimeError) as exc:
            validate_storage_backend_at_boot()
        assert missing in str(exc.value)
        assert "s3" in str(exc.value).lower()


def test_boot_validation_empty_env_var_counts_as_missing(monkeypatch):
    for name, value in ALL_S3_ENV.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv(ENV_S3_BUCKET, "   ")
    with pytest.raises(RuntimeError, match=ENV_S3_BUCKET):
        validate_storage_backend_at_boot()


def test_boot_validation_missing_boto3(monkeypatch):
    for name, value in ALL_S3_ENV.items():
        monkeypatch.setenv(name, value)
    blocker = _Boto3Blocker()
    monkeypatch.setattr(sys, "meta_path", [blocker] + list(sys.meta_path))
    monkeypatch.delitem(sys.modules, "boto3", raising=False)
    try:
        with pytest.raises(RuntimeError) as exc:
            validate_storage_backend_at_boot()
        assert "boto3" in str(exc.value)
    finally:
        # No monkeypatch restore for sys.modules["boto3"] deletion if the
        # import succeeded elsewhere; re-import to heal the process.
        try:
            importlib.import_module("boto3")
        except ImportError:
            pass


def test_boot_validation_never_falls_back_to_local(monkeypatch):
    for name, value in ALL_S3_ENV.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv(ENV_S3_SECRET_KEY, raising=False)
    with pytest.raises(RuntimeError, match="fall"):
        validate_storage_backend_at_boot()
    # And the factory must not hand out a local backend in that state either.
    reset_storage_backend()
    with pytest.raises(RuntimeError, match=ENV_S3_SECRET_KEY):
        get_storage_backend()


def test_boot_validation_unknown_backend_name_raises(monkeypatch):
    monkeypatch.setenv(storage_backend.ENV_STORAGE_BACKEND, "ftp")
    with pytest.raises(RuntimeError, match="ODYSSEUS_STORAGE_BACKEND"):
        validate_storage_backend_at_boot()


def test_boot_validation_rejects_endpoint_without_scheme(monkeypatch):
    for name, value in ALL_S3_ENV.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv(ENV_S3_ENDPOINT, "minio.example.com")
    with pytest.raises(RuntimeError, match=ENV_S3_ENDPOINT):
        validate_storage_backend_at_boot()


def test_boot_validation_rejects_invalid_bucket_name(monkeypatch):
    for name, value in ALL_S3_ENV.items():
        monkeypatch.setenv(name, value)
    for bad in ("Upper_Case", "ab", "-leading-hyphen", "trailing.hyphen.",
                "spaces in name", "a" * 64):
        monkeypatch.setenv(ENV_S3_BUCKET, bad)
        with pytest.raises(RuntimeError, match=ENV_S3_BUCKET):
            validate_storage_backend_at_boot()
    # Sanity: a conventional name still passes.
    monkeypatch.setenv(ENV_S3_BUCKET, "odysseus-uploads")
    assert validate_storage_backend_at_boot() is None


def test_boot_validation_accepts_configured_s3(monkeypatch):
    # No network is touched: validation imports boto3 and checks env only.
    for name, value in ALL_S3_ENV.items():
        monkeypatch.setenv(name, value)
    assert validate_storage_backend_at_boot() is None


# ---------------------------------------------------------------------------
# read_attachment_bytes dispatch
# ---------------------------------------------------------------------------

def test_read_attachment_bytes_local_path(tmp_path):
    target = tmp_path / "file.txt"
    target.write_bytes(b"local-bytes")
    assert read_attachment_bytes({"path": str(target)}) == b"local-bytes"


def test_read_attachment_bytes_dispatches_on_s3_prefix(monkeypatch):
    calls = {}

    class _FakeS3(S3Backend):
        def __init__(self):
            super().__init__(bucket="buck")
            self._client = object()  # never used by overridden methods

        def get_bytes(self, key):
            calls["key"] = key
            return b"s3-bytes"

    # Active backend is local; the s3:// row still reads through S3.
    assert isinstance(get_storage_backend(), LocalBackend)
    monkeypatch.setattr(storage_backend, "_s3_backend_for_uri", lambda: _FakeS3())
    assert read_attachment_bytes({"path": "s3://buck/2026/01/02/" + "a" * 32}) == b"s3-bytes"
    assert calls["key"] == "2026/01/02/" + "a" * 32


def test_read_attachment_bytes_rejects_foreign_bucket(monkeypatch):
    class _FakeS3(S3Backend):
        def __init__(self):
            super().__init__(bucket="buck")

    monkeypatch.setattr(storage_backend, "_s3_backend_for_uri", lambda: _FakeS3())
    with pytest.raises(ValueError, match="bucket"):
        read_attachment_bytes({"path": "s3://other/key"})


def test_read_attachment_bytes_input_validation():
    with pytest.raises(TypeError):
        read_attachment_bytes("s3://bucket/key")
    with pytest.raises(ValueError):
        read_attachment_bytes({"path": None})
