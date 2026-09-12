"""Pin the BOOT WIRING of the storage/gallery validators — not the functions
(their behavior has own tests) but that initialize_managers actually CALLS
them, storage first. Born from the 2026-09-12 upstream sync: the
app_initializer conflict resolution silently dropped both calls; every
function-level test stayed green while the production fail-fast contracts
(S3 env presence, cache cap parse, gallery cache-dir placement) were dead
code."""
import pytest


def test_calls_both_validators_before_anything_else(monkeypatch, tmp_path):
    import src.app_initializer as ai
    order = []
    monkeypatch.setattr("src.storage_backend.validate_storage_backend_at_boot",
                        lambda: order.append("storage"))
    monkeypatch.setattr("src.gallery_storage.validate_gallery_storage_at_boot",
                        lambda: order.append("gallery"))
    monkeypatch.setattr(ai, "create_directories",
                        lambda: order.append("dirs"))
    monkeypatch.setattr(ai, "MemoryManager", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("stop-after-wiring")))
    monkeypatch.setattr(ai, "SkillsManager", lambda *a, **k: None)
    monkeypatch.setattr(ai, "SessionManager", lambda *a, **k: None)
    with pytest.raises(RuntimeError, match="stop-after-wiring"):
        ai.initialize_managers(base_dir=str(tmp_path))
    assert order == ["storage", "gallery", "dirs"]


def test_gallery_validator_failure_refuses_boot(monkeypatch, tmp_path):
    import src.app_initializer as ai
    monkeypatch.setattr("src.storage_backend.validate_storage_backend_at_boot",
                        lambda: None)
    def boom():
        raise RuntimeError("gallery cache dir misconfigured")
    monkeypatch.setattr("src.gallery_storage.validate_gallery_storage_at_boot", boom)
    with pytest.raises(RuntimeError, match="gallery cache dir misconfigured"):
        ai.initialize_managers(base_dir=str(tmp_path))


def test_storage_validator_failure_refuses_boot(monkeypatch, tmp_path):
    import src.app_initializer as ai
    def boom():
        raise RuntimeError("requires the optional dependency boto3")
    monkeypatch.setattr("src.storage_backend.validate_storage_backend_at_boot", boom)
    with pytest.raises(RuntimeError, match="boto3"):
        ai.initialize_managers(base_dir=str(tmp_path))
