"""Smoke tests that don't require a live openBIS instance."""

from __future__ import annotations

import re

import pytest

from openbis_mcp_server.openbis_client import OpenbisClient, OpenbisConfigError

# Regex matching the expected S3 key format:
# {YYYY-MM-DDTHH-MM-SS.ffffff}_{dataset_type}_{username}_{filename}
_S3_KEY_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}\.\d+_[^_]+_.+_.+$"
)


def test_missing_url_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENBIS_URL", raising=False)
    monkeypatch.delenv("OPENBIS_TOKEN", raising=False)
    monkeypatch.delenv("OPENBIS_USERNAME", raising=False)
    monkeypatch.delenv("OPENBIS_PASSWORD", raising=False)
    with pytest.raises(OpenbisConfigError, match="OPENBIS_URL"):
        OpenbisClient()


def test_missing_credentials_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENBIS_URL", "https://example.invalid")
    monkeypatch.delenv("OPENBIS_TOKEN", raising=False)
    monkeypatch.delenv("OPENBIS_USERNAME", raising=False)
    monkeypatch.delenv("OPENBIS_PASSWORD", raising=False)
    with pytest.raises(OpenbisConfigError, match="No credentials"):
        OpenbisClient()


def test_token_takes_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENBIS_URL", "https://example.invalid")
    monkeypatch.setenv("OPENBIS_TOKEN", "tok")
    # Should not raise; lazy connect is not triggered.
    client = OpenbisClient()
    assert client.url == "https://example.invalid"


def test_server_module_imports() -> None:
    """Importing the server module must not require a live connection."""
    import openbis_mcp_server.server as server

    assert server.mcp is not None


# ---------------------------------------------------------------------------
# S3 key naming convention tests (no network required)
# ---------------------------------------------------------------------------


def _make_client(monkeypatch: pytest.MonkeyPatch, username: str = "testuser") -> OpenbisClient:
    monkeypatch.setenv("OPENBIS_URL", "https://example.invalid")
    monkeypatch.setenv("OPENBIS_USERNAME", username)
    monkeypatch.setenv("OPENBIS_PASSWORD", "pw")
    return OpenbisClient()


def test_s3_key_format(monkeypatch: pytest.MonkeyPatch) -> None:
    """The S3 key must match {timestamp}_{dataset_type}_{username}_{filename}."""
    client = _make_client(monkeypatch, username="alice")
    key = client._make_s3_key("/some/path/my_data.h5", "RAW_DATA")
    assert _S3_KEY_RE.match(key), f"Unexpected S3 key format: {key!r}"


def test_s3_key_contains_dataset_type(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _make_client(monkeypatch)
    key = client._make_s3_key("/data/result.csv", "SIMULATION_OUTPUT")
    assert "SIMULATION_OUTPUT" in key


def test_s3_key_contains_username(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _make_client(monkeypatch, username="bob")
    key = client._make_s3_key("/data/result.csv", "RAW_DATA")
    assert "bob" in key


def test_s3_key_contains_original_filename(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _make_client(monkeypatch)
    key = client._make_s3_key("/some/nested/dir/experiment.tar.gz", "RAW_DATA")
    assert key.endswith("experiment.tar.gz")


def test_s3_key_differs_from_original_filename(monkeypatch: pytest.MonkeyPatch) -> None:
    """The S3 key must not equal the bare original filename."""
    client = _make_client(monkeypatch)
    original = "my_results.zip"
    key = client._make_s3_key(f"/uploads/{original}", "RAW_DATA")
    assert key != original


def test_s3_key_unique_across_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two consecutive calls must produce different keys (timestamp differs)."""
    import time

    client = _make_client(monkeypatch)
    key1 = client._make_s3_key("/data/file.txt", "RAW_DATA")
    time.sleep(0.001)  # ensure timestamps differ
    key2 = client._make_s3_key("/data/file.txt", "RAW_DATA")
    assert key1 != key2


def test_s3_key_unknown_username_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """When no username is configured the key should use 'unknown'."""
    monkeypatch.setenv("OPENBIS_URL", "https://example.invalid")
    monkeypatch.setenv("OPENBIS_TOKEN", "tok")
    monkeypatch.delenv("OPENBIS_USERNAME", raising=False)
    client = OpenbisClient()
    key = client._make_s3_key("/data/file.txt", "RAW_DATA")
    assert "unknown" in key


def test_upload_to_s3_missing_credentials(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pytest.TempPathFactory
) -> None:
    """upload_to_s3 raises OpenbisConfigError when S3 env vars are absent."""
    client = _make_client(monkeypatch)
    monkeypatch.delenv("S3_ACCESS_KEY", raising=False)
    monkeypatch.delenv("S3_ACCESS_SECRET", raising=False)
    monkeypatch.delenv("S3_BUCKET", raising=False)
    # Re-create client so it picks up the cleared env vars
    client = _make_client(monkeypatch)
    test_file = tmp_path / "dummy.txt"  # type: ignore[operator]
    test_file.write_text("hello")
    with pytest.raises(OpenbisConfigError, match="S3"):
        client.upload_to_s3(str(test_file), "RAW_DATA")


def test_upload_to_s3_file_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    """upload_to_s3 raises FileNotFoundError for a non-existent file."""
    monkeypatch.setenv("OPENBIS_URL", "https://example.invalid")
    monkeypatch.setenv("OPENBIS_USERNAME", "user")
    monkeypatch.setenv("OPENBIS_PASSWORD", "pw")
    monkeypatch.setenv("S3_ACCESS_KEY", "key")
    monkeypatch.setenv("S3_ACCESS_SECRET", "secret")
    monkeypatch.setenv("S3_BUCKET", "mybucket")
    client = OpenbisClient()
    with pytest.raises(FileNotFoundError):
        client.upload_to_s3("/nonexistent/path/file.txt", "RAW_DATA")

