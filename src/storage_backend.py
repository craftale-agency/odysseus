"""
storage_backend.py

Pluggable storage backend for chat upload bytes: local filesystem (the
historical behaviour) or S3-compatible object storage (MinIO, Garage, AWS).

Selection is environment-driven and read at *call* time (not import time):

    ODYSSEUS_STORAGE_BACKEND   "local" (default) or "s3"

S3 mode additionally reads (see validate_storage_backend_at_boot for which
are mandatory — there is deliberately NO silent fallback to local):

    ODYSSEUS_S3_ENDPOINT       https://minio.example.com  (required)
    ODYSSEUS_S3_ACCESS_KEY     access key                 (required)
    ODYSSEUS_S3_SECRET_KEY     secret key                 (required)
    ODYSSEUS_S3_BUCKET         bucket name                (required)
    ODYSSEUS_S3_REGION         region   (default us-east-1)
    ODYSSEUS_S3_PATH_STYLE     "true"/"1" (default) -> path-style addressing,
                               required by most self-hosted S3 servers

uploads.json keeps storing a single "path" value per row; for S3 rows that
value is the URI form ``s3://<bucket>/<object key>`` while local rows keep
absolute filesystem paths, so both row shapes coexist in one index and a
backend flip never invalidates the other half of the index. Object keys keep
the date-sharded ``YYYY/MM/DD/{uuid32}{ext}`` shape used by save_upload.
"""

import os
import re
import abc
import logging
import mimetypes
from typing import Any, Dict, Iterator, Optional, Tuple

logger = logging.getLogger(__name__)

ENV_STORAGE_BACKEND = "ODYSSEUS_STORAGE_BACKEND"
ENV_S3_ENDPOINT = "ODYSSEUS_S3_ENDPOINT"
ENV_S3_ACCESS_KEY = "ODYSSEUS_S3_ACCESS_KEY"
ENV_S3_SECRET_KEY = "ODYSSEUS_S3_SECRET_KEY"
ENV_S3_BUCKET = "ODYSSEUS_S3_BUCKET"
ENV_S3_REGION = "ODYSSEUS_S3_REGION"
ENV_S3_PATH_STYLE = "ODYSSEUS_S3_PATH_STYLE"

# Missing any of these with backend=s3 aborts startup instead of degrading.
REQUIRED_S3_ENV: Tuple[str, ...] = (
    ENV_S3_ENDPOINT,
    ENV_S3_ACCESS_KEY,
    ENV_S3_SECRET_KEY,
    ENV_S3_BUCKET,
)

S3_URI_PREFIX = "s3://"

_BACKEND = None
# Separate lazily-built S3 backend used purely to read s3:// rows when the
# active backend is local (mixed index during/after a backend flip). The
# boto3 client inside stays unbuilt until an actual read happens.
_URI_READ_BACKEND = None


# ---------------------------------------------------------------------------
# URI helpers
# ---------------------------------------------------------------------------

def is_s3_uri(value: Any) -> bool:
    """Return True when *value* is a string in s3://bucket/key form."""
    return isinstance(value, str) and value.startswith(S3_URI_PREFIX)


def parse_s3_uri(uri: str) -> Optional[Tuple[str, str]]:
    """Split ``s3://bucket/key`` into ``(bucket, key)``.

    Returns None for anything that is not a well-formed, non-empty
    bucket/key pair (no scheme, empty bucket, empty key, whitespace).
    """
    if not is_s3_uri(uri):
        return None
    rest = uri[len(S3_URI_PREFIX):]
    bucket, _, key = rest.partition("/")
    bucket = (bucket or "").strip()
    key = (key or "").strip("/")
    if not bucket or not key or " " in bucket:
        return None
    return bucket, key


class StorageBackend(abc.ABC):
    """Minimal byte-store contract used by the upload pipeline.

    Keys are opaque to callers except that LocalBackend keys are absolute
    filesystem paths under the uploads dir and S3Backend keys are raw object
    keys (never s3://-prefixed — the URI form lives only in uploads.json).
    """

    @property
    def is_s3(self) -> bool:
        return False

    @abc.abstractmethod
    def put(self, key: str, fileobj, content_type: str) -> None:
        """Store the full contents of *fileobj* under *key*."""

    @abc.abstractmethod
    def get_bytes(self, key: str) -> bytes:
        """Return the full contents stored under *key*."""

    @abc.abstractmethod
    def get_stream(self, key: str):
        """Return a file-like/chunk iterator suitable for StreamingResponse."""

    @abc.abstractmethod
    def exists(self, key: str) -> bool:
        """Return True when *key* holds an object."""

    @abc.abstractmethod
    def delete(self, key: str) -> None:
        """Remove *key* (idempotent: a missing key is not an error)."""

    @abc.abstractmethod
    def stat(self, key: str) -> Optional[Dict[str, Any]]:
        """Return {"size": int, "content_type": str, ...} or None if missing."""


# ---------------------------------------------------------------------------
# Local filesystem backend (the historical behaviour)
# ---------------------------------------------------------------------------

class LocalBackend(StorageBackend):
    """Thin wrapper preserving today's filesystem semantics.

    Keys are absolute paths. Writes are chunked exactly like the previous
    inline loop in save_upload, and every operation enforces the same
    containment check style as UploadHandler._inside_upload_dir
    (realpath + normcase + commonpath) so a crafted key cannot escape the
    uploads directory through the backend.
    """

    def __init__(self, root_dir: str):
        self.root_dir = root_dir

    def _assert_inside(self, key: str) -> None:
        if not self._inside_root(key):
            raise ValueError(f"path escapes storage root: {key!r}")

    def _inside_root(self, key: str) -> bool:
        base = os.path.normcase(os.path.realpath(self.root_dir))
        candidate = os.path.normcase(os.path.realpath(key))
        try:
            return os.path.commonpath([base, candidate]) == base
        except Exception:
            return False

    def put(self, key: str, fileobj, content_type: str) -> None:
        self._assert_inside(key)
        directory = os.path.dirname(key)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(key, "wb") as f:
            while chunk := fileobj.read(8192):
                f.write(chunk)

    def get_bytes(self, key: str) -> bytes:
        self._assert_inside(key)
        with open(key, "rb") as f:
            return f.read()

    def get_stream(self, key: str):
        self._assert_inside(key)
        f = open(key, "rb")

        def _chunks():
            try:
                while True:
                    chunk = f.read(65536)
                    if not chunk:
                        break
                    yield chunk
            finally:
                f.close()

        return _chunks()

    def exists(self, key: str) -> bool:
        self._assert_inside(key)
        return os.path.isfile(key)

    def delete(self, key: str) -> None:
        self._assert_inside(key)
        try:
            os.remove(key)
        except FileNotFoundError:
            pass

    def stat(self, key: str) -> Optional[Dict[str, Any]]:
        self._assert_inside(key)
        try:
            st = os.stat(key)
        except OSError:
            return None
        return {
            "size": st.st_size,
            "content_type": mimetypes.guess_type(key)[0],
            "last_modified": st.st_mtime,
        }


# ---------------------------------------------------------------------------
# S3 backend
# ---------------------------------------------------------------------------

class S3Backend(StorageBackend):
    """S3-compatible object backend (boto3, lazy client).

    The boto3 client is created on first use so importing this module — and
    running the app with the default local backend — never requires boto3.
    Configuration is snapshotted from the environment at construction.
    """

    def __init__(
        self,
        endpoint_url: Optional[str] = None,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        bucket: Optional[str] = None,
        region: Optional[str] = None,
        path_style: Optional[bool] = None,
    ):
        self.endpoint_url = (endpoint_url if endpoint_url is not None
                             else os.getenv(ENV_S3_ENDPOINT, "")).strip() or None
        self.access_key = (access_key if access_key is not None
                           else os.getenv(ENV_S3_ACCESS_KEY, "")).strip()
        self.secret_key = (secret_key if secret_key is not None
                           else os.getenv(ENV_S3_SECRET_KEY, "")).strip()
        self.bucket = (bucket if bucket is not None
                       else os.getenv(ENV_S3_BUCKET, "")).strip()
        self.region = (region if region is not None
                       else os.getenv(ENV_S3_REGION, "")).strip() or "us-east-1"
        if path_style is None:
            raw = os.getenv(ENV_S3_PATH_STYLE, "true").strip().lower()
            path_style = raw not in ("0", "false", "no", "off")
        self.path_style = bool(path_style)
        self._client = None

    @property
    def is_s3(self) -> bool:
        return True

    def client(self):
        """Return the lazily-created boto3 client (thread-safe enough for the
        single-process upload pipeline; races at worst build two clients)."""
        if self._client is None:
            try:
                import boto3
                from botocore.config import Config
            except ImportError as e:
                raise RuntimeError(
                    "ODYSSEUS_STORAGE_BACKEND=s3 requires the optional "
                    "dependency boto3. Install it with: "
                    "pip install -r requirements-optional.txt"
                ) from e
            self._client = boto3.client(
                "s3",
                endpoint_url=self.endpoint_url,
                aws_access_key_id=self.access_key or None,
                aws_secret_access_key=self.secret_key or None,
                region_name=self.region,
                config=Config(
                    signature_version="s3v4",
                    retries={"max_attempts": 5, "mode": "standard"},
                    connect_timeout=10,
                    read_timeout=60,
                    s3={"addressing_style": "path" if self.path_style else "auto"},
                ),
            )
        return self._client

    # -- URI <-> key conversion ------------------------------------------

    def uri_for_key(self, key: str) -> str:
        """Object key -> the s3://bucket/key URI stored in uploads.json."""
        return f"{S3_URI_PREFIX}{self.bucket}/{(key or '').lstrip('/')}"

    def key_for_uri(self, uri: str) -> Optional[str]:
        """uploads.json URI -> raw object key, or None when not our bucket."""
        parsed = parse_s3_uri(uri)
        if parsed is None or parsed[0] != self.bucket:
            return None
        return parsed[1]

    def owns_uri(self, uri: str) -> bool:
        return self.key_for_uri(uri) is not None

    # -- StorageBackend API ----------------------------------------------

    def put(self, key: str, fileobj, content_type: str) -> None:
        self.client().upload_fileobj(
            fileobj,
            self.bucket,
            key,
            ExtraArgs={"ContentType": content_type} if content_type else None,
        )

    def get_bytes(self, key: str) -> bytes:
        obj = self.client().get_object(Bucket=self.bucket, Key=key)
        return obj["Body"].read()

    def get_stream(self, key: str):
        obj = self.client().get_object(Bucket=self.bucket, Key=key)
        body = obj["Body"]

        def _chunks():
            # Close the StreamingBody on ANY exit (including client
            # disconnects mid-download) so the pooled connection is
            # released immediately instead of waiting out the read
            # timeout.
            try:
                yield from body.iter_chunks(chunk_size=65536)
            finally:
                body.close()

        return _chunks()

    def exists(self, key: str) -> bool:
        from botocore.exceptions import ClientError
        try:
            self.client().head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError as e:
            code = str(e.response.get("Error", {}).get("Code", ""))
            status = e.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            if code in ("404", "NoSuchKey", "NotFound") or status == 404:
                return False
            raise

    def delete(self, key: str) -> None:
        self.client().delete_object(Bucket=self.bucket, Key=key)

    def stat(self, key: str) -> Optional[Dict[str, Any]]:
        from botocore.exceptions import ClientError
        try:
            head = self.client().head_object(Bucket=self.bucket, Key=key)
        except ClientError as e:
            code = str(e.response.get("Error", {}).get("Code", ""))
            status = e.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            if code in ("404", "NoSuchKey", "NotFound") or status == 404:
                return None
            raise
        return {
            "size": head.get("ContentLength"),
            "content_type": head.get("ContentType"),
            "last_modified": head.get("LastModified"),
        }

    def list_keys(self) -> Iterator[str]:
        """Yield every object key in the bucket (ListObjectsV2, paginated)."""
        paginator = self.client().get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket):
            for item in page.get("Contents", []) or []:
                key = item.get("Key")
                if key:
                    yield key


# ---------------------------------------------------------------------------
# Factory / singleton (mirrors src/chroma_client.py)
# ---------------------------------------------------------------------------

def _requested_backend_name() -> str:
    raw = os.getenv(ENV_STORAGE_BACKEND, "").strip().lower()
    if not raw or raw == "local":
        return "local"
    if raw == "s3":
        return "s3"
    raise RuntimeError(
        f"{ENV_STORAGE_BACKEND} must be 'local' or 's3' (got {raw!r})"
    )


def _missing_s3_env() -> list:
    """Required-but-unset S3 env var names (empty string counts as unset)."""
    return [env for env in REQUIRED_S3_ENV if not os.getenv(env, "").strip()]


def get_storage_backend() -> StorageBackend:
    """Get or create the singleton storage backend.

    The environment is read when the singleton is first requested (call
    time), never at import time — mirroring get_chroma_client(). A
    backend=s3 selection with missing required env raises instead of
    silently degrading to local.
    """
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND
    name = _requested_backend_name()
    if name == "s3":
        missing = _missing_s3_env()
        if missing:
            raise RuntimeError(
                "ODYSSEUS_STORAGE_BACKEND=s3 requires "
                + ", ".join(missing)
                + " to be set (non-empty). Refusing to fall back to local storage."
            )
        _BACKEND = S3Backend()
        logger.info(
            "Storage backend: s3 (bucket=%s endpoint=%s path_style=%s)",
            _BACKEND.bucket, _BACKEND.endpoint_url, _BACKEND.path_style,
        )
    else:
        from src.constants import UPLOAD_DIR
        _BACKEND = LocalBackend(UPLOAD_DIR)
    return _BACKEND


def reset_storage_backend() -> None:
    """Reset the singleton (e.g. after env change or in tests)."""
    global _BACKEND
    _BACKEND = None


def _s3_backend_for_uri() -> S3Backend:
    """Backend used to read s3:// rows when the active backend is local.

    A mixed index (local + s3 rows) must stay readable in both directions,
    so reading an s3:// path never depends on ODYSSEUS_STORAGE_BACKEND.
    """
    global _URI_READ_BACKEND
    active = get_storage_backend()
    if isinstance(active, S3Backend):
        return active
    if _URI_READ_BACKEND is None:
        _URI_READ_BACKEND = S3Backend()
    return _URI_READ_BACKEND


def read_attachment_bytes(info: Dict[str, Any]) -> bytes:
    """Read the bytes behind one uploads.json row.

    This is THE read API for downstream consumers: it dispatches on the
    ``s3://`` prefix of the row's "path" (object read) versus a local
    filesystem path (plain open). Raises on unreadable/missing data.
    """
    if not isinstance(info, dict):
        raise TypeError(f"read_attachment_bytes expects an uploads.json row dict, got {type(info)!r}")
    path = info.get("path")
    if not path:
        raise ValueError("uploads.json row has no path")
    if is_s3_uri(path):
        backend = _s3_backend_for_uri()
        key = backend.key_for_uri(path)
        if key is None:
            raise ValueError(f"s3 upload row is not in bucket {backend.bucket!r}: {path!r}")
        return backend.get_bytes(key)
    with open(path, "rb") as f:
        return f.read()


def validate_storage_backend_at_boot() -> None:
    """Fail fast at startup when backend=s3 is misconfigured.

    No-op for the default local backend. For s3, every required env var
    must be set AND boto3 must be importable, otherwise RuntimeError names
    exactly what is missing. There is NEVER a silent fallback to local —
    a half-configured S3 deployment must be visible in the container logs.
    """
    name = _requested_backend_name()
    if name != "s3":
        return
    missing = _missing_s3_env()
    if missing:
        raise RuntimeError(
            "ODYSSEUS_STORAGE_BACKEND=s3 requires "
            + ", ".join(missing)
            + " to be set (non-empty). Refusing to fall back to local storage."
        )
    endpoint = os.getenv(ENV_S3_ENDPOINT, "").strip()
    if not (endpoint.startswith("http://") or endpoint.startswith("https://")):
        raise RuntimeError(
            f"{ENV_S3_ENDPOINT} must start with http:// or https:// "
            f"(got {endpoint!r}). Refusing to fall back to local storage."
        )
    bucket = os.getenv(ENV_S3_BUCKET, "").strip()
    if not re.fullmatch(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]", bucket):
        raise RuntimeError(
            f"{ENV_S3_BUCKET} is not a valid S3 bucket name "
            f"(3-63 chars, lowercase alphanumerics, dots and hyphens; got {bucket!r}). "
            "Refusing to fall back to local storage."
        )
    try:
        import boto3  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "ODYSSEUS_STORAGE_BACKEND=s3 requires the optional dependency "
            "boto3, which is not installed. Install it with: "
            "pip install -r requirements-optional.txt "
            "(missing dependency: boto3)"
        ) from e
