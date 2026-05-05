"""Smoke tests that don't require a live openBIS instance."""

from __future__ import annotations

import pathlib
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


def test_token_file_env(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """OPENBIS_TOKEN_FILE should be read when OPENBIS_TOKEN is empty."""
    token_path = tmp_path / "tok"
    token_path.write_text("secret-token-from-file\n")
    monkeypatch.setenv("OPENBIS_URL", "https://example.invalid")
    monkeypatch.delenv("OPENBIS_TOKEN", raising=False)
    monkeypatch.setenv("OPENBIS_TOKEN_FILE", str(token_path))
    client = OpenbisClient()
    assert client._token == "secret-token-from-file"


def test_write_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Write tools must refuse to run when OPENBIS_MCP_ALLOW_WRITE is unset."""
    monkeypatch.delenv("OPENBIS_MCP_ALLOW_WRITE", raising=False)
    from openbis_mcp_server.server import _require_write, _write_enabled

    assert _write_enabled() is False
    with pytest.raises(RuntimeError, match="OPENBIS_MCP_ALLOW_WRITE"):
        _require_write()


def test_write_enabled_when_flag_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENBIS_MCP_ALLOW_WRITE", "1")
    from openbis_mcp_server.server import _require_write, _write_enabled

    assert _write_enabled() is True
    _require_write()  # must not raise


# ---------------------------------------------------------------------------
# S3 config tests
# ---------------------------------------------------------------------------


def test_s3_config_not_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    """get_s3_config returns None when no S3 env vars are set."""
    for var in [
        "S3_ACCESS_KEY",
        "S3_ACCESS_SECRET",
        "S3_BUCKET",
        "S3_DMS_CODE",
        "S3_CONFIG_FILE",
    ]:
        monkeypatch.delenv(var, raising=False)

    from openbis_mcp_server.s3_support import get_s3_config

    assert get_s3_config() is None


def test_s3_config_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """get_s3_config returns an S3Config when all required env vars are present."""
    monkeypatch.delenv("S3_CONFIG_FILE", raising=False)
    monkeypatch.setenv("S3_ACCESS_KEY", "AKID123")
    monkeypatch.setenv("S3_ACCESS_SECRET", "secret456")
    monkeypatch.setenv("S3_BUCKET", "my-bucket")
    monkeypatch.setenv("S3_DMS_CODE", "MY_DMS")
    monkeypatch.setenv("S3_REGION", "us-east-1")

    from openbis_mcp_server.s3_support import get_s3_config

    cfg = get_s3_config()
    assert cfg is not None
    assert cfg.access_key == "AKID123"
    assert cfg.access_secret == "secret456"
    assert cfg.bucket == "my-bucket"
    assert cfg.dms_code == "MY_DMS"
    assert cfg.region == "us-east-1"
    assert cfg.endpoint_url is None


def test_s3_config_endpoint_url_with_port(monkeypatch: pytest.MonkeyPatch) -> None:
    """Endpoint URL and port are merged into a single URL."""
    monkeypatch.delenv("S3_CONFIG_FILE", raising=False)
    monkeypatch.setenv("S3_ACCESS_KEY", "key")
    monkeypatch.setenv("S3_ACCESS_SECRET", "sec")
    monkeypatch.setenv("S3_BUCKET", "bucket")
    monkeypatch.setenv("S3_DMS_CODE", "DMS1")
    monkeypatch.setenv("S3_ENDPOINT_URL", "https://s3.example.com")
    monkeypatch.setenv("S3_ENDPOINT_PORT", "9000")

    from openbis_mcp_server.s3_support import get_s3_config

    cfg = get_s3_config()
    assert cfg is not None
    assert cfg.endpoint_url == "https://s3.example.com:9000"


def test_s3_config_from_file(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """load_s3_config_from_file parses an INI config file correctly."""
    cfg_file = tmp_path / "s3.ini"
    cfg_file.write_text(
        "[s3]\n"
        "s3_region = eu-central-1\n"
        "s3_endpoint_url = https://s3.custom.com\n"
        "s3_access_key = file_key\n"
        "s3_access_secret = file_secret\n"
        "s3_bucket = file-bucket\n"
        "\n"
        "[openbis]\n"
        "dms_code = FILE_DMS\n"
    )

    from openbis_mcp_server.s3_support import load_s3_config_from_file

    cfg = load_s3_config_from_file(str(cfg_file))
    assert cfg.access_key == "file_key"
    assert cfg.bucket == "file-bucket"
    assert cfg.dms_code == "FILE_DMS"
    assert cfg.endpoint_url == "https://s3.custom.com"


def test_s3_config_file_missing_dms_code(tmp_path: pathlib.Path) -> None:
    """A config file without dms_code raises S3ConfigError."""
    cfg_file = tmp_path / "bad.ini"
    cfg_file.write_text("[s3]\ns3_access_key = k\ns3_access_secret = s\ns3_bucket = b\n")

    from openbis_mcp_server.s3_support import S3ConfigError, load_s3_config_from_file

    with pytest.raises(S3ConfigError, match="dms_code"):
        load_s3_config_from_file(str(cfg_file))


def test_s3_config_file_not_found() -> None:
    """load_s3_config_from_file raises S3ConfigError for a missing file."""
    from openbis_mcp_server.s3_support import S3ConfigError, load_s3_config_from_file

    with pytest.raises(S3ConfigError, match="not found"):
        load_s3_config_from_file("/nonexistent/path/s3.ini")


def test_s3_status_tool_not_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    """get_s3_status returns configured=False when S3 is not set up."""
    for var in ["S3_ACCESS_KEY", "S3_ACCESS_SECRET", "S3_BUCKET", "S3_DMS_CODE", "S3_CONFIG_FILE"]:
        monkeypatch.delenv(var, raising=False)

    from openbis_mcp_server.server import get_s3_status

    result = get_s3_status()
    assert result["configured"] is False


def test_s3_status_tool_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    """get_s3_status returns configured=True with bucket info when S3 is set up."""
    monkeypatch.delenv("S3_CONFIG_FILE", raising=False)
    monkeypatch.setenv("S3_ACCESS_KEY", "k")
    monkeypatch.setenv("S3_ACCESS_SECRET", "s")
    monkeypatch.setenv("S3_BUCKET", "test-bucket")
    monkeypatch.setenv("S3_DMS_CODE", "TEST_DMS")

    from openbis_mcp_server.server import get_s3_status

    result = get_s3_status()
    assert result["configured"] is True
    assert result["bucket"] == "test-bucket"
    assert result["dms_code"] == "TEST_DMS"


def test_s3_get_file_metadata(tmp_path: pathlib.Path) -> None:
    """get_file_metadata returns the expected structure for a real file."""
    test_file = tmp_path / "sample.dat"
    test_file.write_bytes(b"hello world")

    from openbis_mcp_server.s3_support import get_file_metadata

    meta = get_file_metadata(
        str(test_file), "https://s3.example.com/sample.dat", compute_crc32=True
    )
    assert len(meta) == 1
    entry = meta[0]
    assert entry["fileLength"] == 11
    assert entry["directory"] is False
    assert entry["path"] == "https://s3.example.com/sample.dat"
    assert "checksum" in entry


# ---------------------------------------------------------------------------
# S3 key naming convention tests (no network required)
# ---------------------------------------------------------------------------


def test_s3_key_format() -> None:
    """The S3 key must match {timestamp}_{dataset_type}_{username}_{filename}."""
    from openbis_mcp_server.s3_support import make_s3_key

    key = make_s3_key("/some/path/my_data.h5", "RAW_DATA", "alice")
    assert _S3_KEY_RE.match(key), f"Unexpected S3 key format: {key!r}"


def test_s3_key_contains_dataset_type() -> None:
    from openbis_mcp_server.s3_support import make_s3_key

    key = make_s3_key("/data/result.csv", "SIMULATION_OUTPUT", "user")
    assert "SIMULATION_OUTPUT" in key


def test_s3_key_contains_username() -> None:
    from openbis_mcp_server.s3_support import make_s3_key

    key = make_s3_key("/data/result.csv", "RAW_DATA", "bob")
    assert "bob" in key


def test_s3_key_contains_original_filename() -> None:
    from openbis_mcp_server.s3_support import make_s3_key

    key = make_s3_key("/some/nested/dir/experiment.tar.gz", "RAW_DATA", "user")
    assert key.endswith("experiment.tar.gz")


def test_s3_key_differs_from_original_filename() -> None:
    """The S3 key must not equal the bare original filename."""
    from openbis_mcp_server.s3_support import make_s3_key

    original = "my_results.zip"
    key = make_s3_key(f"/uploads/{original}", "RAW_DATA", "user")
    assert key != original


def test_s3_key_unique_across_calls() -> None:
    """Two consecutive calls must produce different keys (timestamp differs)."""
    import time

    from openbis_mcp_server.s3_support import make_s3_key

    key1 = make_s3_key("/data/file.txt", "RAW_DATA", "user")
    time.sleep(0.001)  # ensure timestamps differ
    key2 = make_s3_key("/data/file.txt", "RAW_DATA", "user")
    assert key1 != key2


def test_s3_key_unknown_username_fallback() -> None:
    """When no username is provided the key should use 'unknown'."""
    from openbis_mcp_server.s3_support import make_s3_key

    key = make_s3_key("/data/file.txt", "RAW_DATA")
    assert "unknown" in key


def test_client_username_property_with_username(monkeypatch: pytest.MonkeyPatch) -> None:
    """OpenbisClient.username returns the configured username."""
    monkeypatch.setenv("OPENBIS_URL", "https://example.invalid")
    monkeypatch.setenv("OPENBIS_USERNAME", "alice")
    monkeypatch.setenv("OPENBIS_PASSWORD", "pw")
    client = OpenbisClient()
    assert client.username == "alice"


def test_client_username_property_token_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    """OpenbisClient.username returns 'unknown' when only a token is configured."""
    monkeypatch.setenv("OPENBIS_URL", "https://example.invalid")
    monkeypatch.setenv("OPENBIS_TOKEN", "tok")
    monkeypatch.delenv("OPENBIS_USERNAME", raising=False)
    client = OpenbisClient()
    assert client.username == "unknown"


