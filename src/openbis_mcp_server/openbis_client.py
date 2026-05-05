"""Thin wrapper around pyBIS that handles connection setup from environment variables.

The wrapper is intentionally minimal: it owns one ``Openbis`` instance and lazily
logs in on first use. Higher-level logic (search, create, upload, ...) lives in
``server.py`` so this file stays focused on connection management.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from threading import Lock
from typing import Any


class OpenbisConfigError(RuntimeError):
    """Raised when required configuration is missing or inconsistent."""


class OpenbisClient:
    """Lazy, thread-safe wrapper around a single pyBIS ``Openbis`` connection."""

    def __init__(
        self,
        url: str | None = None,
        token: str | None = None,
        username: str | None = None,
        password: str | None = None,
        verify_certificates: bool | None = None,
    ) -> None:
        self._url = url or os.environ.get("OPENBIS_URL")
        self._token = token or os.environ.get("OPENBIS_TOKEN")
        self._username = username or os.environ.get("OPENBIS_USERNAME")
        self._password = password or os.environ.get("OPENBIS_PASSWORD")

        verify_env = os.environ.get("OPENBIS_VERIFY_CERTIFICATES", "true")
        if verify_certificates is None:
            self._verify_certificates = verify_env.strip().lower() not in {
                "false",
                "0",
                "no",
            }
        else:
            self._verify_certificates = verify_certificates

        if not self._url:
            raise OpenbisConfigError(
                "OPENBIS_URL is not set. Provide it via env var or constructor."
            )
        if not self._token and not (self._username and self._password):
            raise OpenbisConfigError(
                "No credentials found. Set OPENBIS_TOKEN, or both "
                "OPENBIS_USERNAME and OPENBIS_PASSWORD."
            )

        # S3 configuration (optional — only required for upload_to_s3)
        self._s3_access_key = os.environ.get("S3_ACCESS_KEY")
        self._s3_access_secret = os.environ.get("S3_ACCESS_SECRET")
        self._s3_bucket = os.environ.get("S3_BUCKET")
        self._s3_region = os.environ.get("S3_REGION")
        self._s3_endpoint_url = os.environ.get("S3_ENDPOINT_URL")
        self._s3_endpoint_port = os.environ.get("S3_ENDPOINT_PORT")

        self._openbis: Any | None = None
        self._lock = Lock()

    @property
    def url(self) -> str:
        assert self._url is not None  # checked in __init__
        return self._url

    def connect(self) -> Any:
        """Return a logged-in pyBIS ``Openbis`` instance, creating it on first call."""
        if self._openbis is not None:
            return self._openbis

        with self._lock:
            if self._openbis is not None:
                return self._openbis

            # Imported lazily so module import doesn't require pybis to be installed.
            from pybis import Openbis  # type: ignore[import-not-found]

            ob = Openbis(self._url, verify_certificates=self._verify_certificates)
            if self._token:
                ob.set_token(self._token)
            else:
                ob.login(self._username, self._password, save_token=False)

            self._openbis = ob
            return ob

    def close(self) -> None:
        """Log out and drop the connection."""
        with self._lock:
            if self._openbis is not None:
                try:
                    self._openbis.logout()
                finally:
                    self._openbis = None

    # ------------------------------------------------------------------
    # S3 helpers
    # ------------------------------------------------------------------

    def _make_s3_key(self, file_path: str, dataset_type: str) -> str:
        """Return a collision-avoiding S3 object key for *file_path*.

        The key follows the convention used in pyiron_rdm/pybis_aixtended:

            ``{timestamp}_{dataset_type}_{username}_{original_filename}``

        where *timestamp* is UTC in the format ``YYYY-MM-DDTHH-MM-SS.ffffff``.
        This prefix makes every upload unique even when the same file is
        uploaded multiple times or by different users.
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S.%f")
        username = self._username or "unknown"
        basename = os.path.basename(file_path)
        return f"{timestamp}_{dataset_type}_{username}_{basename}"

    def _build_s3_client(self) -> Any:
        """Build and return a boto3 S3 client from the configured environment variables."""
        import boto3  # type: ignore[import-not-found]
        import boto3.session  # type: ignore[import-not-found]

        endpoint_url: str | None = None
        if self._s3_endpoint_url:
            endpoint_url = self._s3_endpoint_url
            if self._s3_endpoint_port:
                endpoint_url = f"{endpoint_url}:{self._s3_endpoint_port}"

        config = boto3.session.Config(
            signature_version="s3v4",
            connect_timeout=5,
            read_timeout=10,
        )
        return boto3.client(
            service_name="s3",
            endpoint_url=endpoint_url,
            region_name=self._s3_region,
            aws_access_key_id=self._s3_access_key,
            aws_secret_access_key=self._s3_access_secret,
            config=config,
        )

    def upload_to_s3(self, file_path: str, dataset_type: str) -> str:
        """Upload *file_path* to S3 and return the S3 object key.

        The object key on S3 is different from the original filename to avoid
        collisions.  It follows the convention::

            {timestamp}_{dataset_type}_{username}_{original_filename}

        Requires the ``S3_ACCESS_KEY``, ``S3_ACCESS_SECRET``, and
        ``S3_BUCKET`` environment variables to be set.

        Args:
            file_path: Local path to the file to upload.
            dataset_type: openBIS dataset type code (e.g. ``"RAW_DATA"``).

        Returns:
            The S3 object key under which the file was stored.

        Raises:
            OpenbisConfigError: If S3 credentials or bucket are not configured.
            FileNotFoundError: If *file_path* does not exist.
        """
        if not self._s3_access_key or not self._s3_access_secret:
            raise OpenbisConfigError(
                "S3 credentials not configured. Set S3_ACCESS_KEY and S3_ACCESS_SECRET."
            )
        if not self._s3_bucket:
            raise OpenbisConfigError("S3 bucket not configured. Set S3_BUCKET.")
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        s3_key = self._make_s3_key(file_path, dataset_type)
        s3_client = self._build_s3_client()
        s3_client.upload_file(Filename=file_path, Bucket=self._s3_bucket, Key=s3_key)
        return s3_key
