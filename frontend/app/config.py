"""Application configuration loaded from environment variables.

All settings are read from the environment (or a ``.env`` file) via
Pydantic Settings.  Override any value by exporting the corresponding
``ATTNRES_``-prefixed environment variable before starting the server,
or by editing the ``.env`` file next to this module.

Keys in ``.env`` **must** include the ``ATTNRES_`` prefix, exactly as they
appear when set in the shell::

    # .env
    ATTNRES_CHECKPOINTS_DIR=/models
    ATTNRES_AUTH_USERNAME=admin
    ATTNRES_AUTH_PASSWORD=secret
    ATTNRES_SECRET_KEY=<openssl rand -hex 32>

The ``.env`` file is resolved relative to this file (``app/config.py``),
so it is found correctly regardless of the working directory from which the
server is started.

Example shell usage::

    export ATTNRES_AUTH_PASSWORD=secret
    uvicorn app.main:app --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve .env relative to this file so it is always found, regardless of
# the working directory from which the server or tests are launched.
_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


class Settings(BaseSettings):
    """Top-level application settings.

    Attributes:
        app_title: Human-readable application name shown in the browser tab.
        app_version: API version string exposed at ``/api/version``.
        host: Interface address for the Uvicorn server.
        port: Port number for the Uvicorn server.
        workers: Number of Uvicorn worker processes.  Set to 1 when using
            GPU inference to avoid loading the model multiple times.
        log_level: Uvicorn log level (``"debug"``, ``"info"``, ``"warning"``).
        checkpoints_dir: Directory that will be scanned for ``*.pt`` checkpoint
            files.  Each file becomes a selectable model in the UI.
        attnres_src_dir: Path to the root of the main ``attnres`` package so
            that ``models``, ``utils``, and ``dataset`` modules can be imported
            even when the frontend is run from its own directory.
        device: PyTorch device string.  ``"auto"`` selects CUDA if available,
            then MPS, then CPU.
        max_new_tokens_limit: Hard ceiling on the ``max_new_tokens`` slider
            to prevent runaway generation requests.
        auth_username: HTTP Basic-Auth username.
        auth_password: HTTP Basic-Auth password stored in plain text.  For
            production use, provide a bcrypt hash via ``auth_password_hash``
            instead and leave this field empty.
        auth_password_hash: bcrypt hash of the admin password.  When set,
            this takes precedence over ``auth_password``.  Generate with::

                python -c "from passlib.hash import bcrypt; print(bcrypt.hash('mypassword'))"
        auth_realm: WWW-Authenticate realm string sent in 401 responses.
        secret_key: Secret used to sign session tokens.  Change this in
            production.  Generate with ``openssl rand -hex 32``.
        session_expire_minutes: Session cookie lifetime in minutes.
    """

    model_config = SettingsConfigDict(
        env_prefix="ATTNRES_",
        # Absolute path so .env is found regardless of cwd.
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        # Fall back gracefully when .env does not exist (e.g. in CI).
        env_file_override=False,
        extra="ignore",
    )

    # ── Server ────────────────────────────────────────────────────────────────
    app_title: str = "AttnRes Chat"
    app_version: str = "0.1.0"
    host: str = "0.0.0.0"
    port: int = 8080
    workers: int = 1
    log_level: str = "info"

    # ── Model paths ───────────────────────────────────────────────────────────
    checkpoints_dir: Path = Field(
        default=Path("../checkpoints"),
        description="Directory scanned for *.pt checkpoint files.",
    )
    attnres_src_dir: Path = Field(
        default=Path(".."),
        description="Root of the attnres source tree (parent of models/, utils/, …).",
    )

    # ── Inference ─────────────────────────────────────────────────────────────
    device: str = "auto"
    max_new_tokens_limit: int = 256

    # ── HTTP Basic Auth ───────────────────────────────────────────────────────
    auth_username: str = "admin"
    auth_password: str = "admin123"
    auth_password_hash: str = ""
    auth_realm: str = "AttnRes Chat"
    secret_key: str = "CHANGE-THIS-IN-PRODUCTION-use-openssl-rand-hex-32"
    session_expire_minutes: int = 480  # 8 hours

    @field_validator("checkpoints_dir", "attnres_src_dir", mode="before")
    @classmethod
    def _expand_path(cls, v: object) -> Path:
        """Expand ``~`` and resolve relative paths.

        Args:
            v: Raw path value from the environment or default.

        Returns:
            Expanded :class:`~pathlib.Path` (``~`` resolved; relative paths
            left relative so they are resolved from the server's cwd at
            runtime, which is usually the ``frontend/`` directory).
        """
        return Path(str(v)).expanduser()


# Module-level singleton — import this everywhere instead of re-instantiating.
settings = Settings()
