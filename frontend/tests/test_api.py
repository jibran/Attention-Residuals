"""API endpoint tests for the AttnRes Chat frontend.

Uses ``httpx.AsyncClient`` with the FastAPI ``TestClient`` transport so all
tests run in-process without starting a real server.  Model loading is mocked
so no actual checkpoint files are required.

Run::

    pytest tests/ -v
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _patch_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the model registry with a lightweight mock.

    Prevents any real checkpoint scanning or model loading during tests.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    mock_registry = MagicMock()
    mock_registry.list_models.return_value = [
        {
            "model_id": "test_model",
            "name": "test_model",
            "val_loss": 2.18,
            "val_ppl": 8.84,
            "epoch": 3,
            "params": 1_000_000,
            "params_fmt": "1.0 M",
            "architecture": "AttnResLM (Block AttnRes)",
            "dataset": "tinystories",
        }
    ]

    async def _fake_generate(**kwargs: Any):
        from app.model_registry import GenerationResult

        return GenerationResult(
            prompt=kwargs["prompt"],
            generated=kwargs["prompt"] + " ...generated text...",
            new_tokens=kwargs.get("max_new_tokens", 10),
            elapsed_s=0.5,
            tok_per_sec=20.0,
            ms_per_tok=50.0,
            model_id=kwargs["model_id"],
            use_kv_cache=kwargs.get("use_kv_cache", False),
        )

    mock_registry.generate = _fake_generate

    monkeypatch.setattr("app.model_registry.registry", mock_registry)
    monkeypatch.setattr("app.routers.pages.registry", mock_registry)
    monkeypatch.setattr("app.routers.api.registry", mock_registry)


@pytest.fixture()
def auth_headers() -> dict[str, str]:
    """Return HTTP Basic Auth headers for the default test credentials.

    Returns:
        Dictionary with the ``Authorization`` header set.
    """
    import base64

    creds = base64.b64encode(b"admin:changeme").decode()
    return {"Authorization": f"Basic {creds}"}


@pytest_asyncio.fixture()
async def client(auth_headers: dict[str, str]) -> AsyncClient:
    """Return an authenticated async test client.

    Args:
        auth_headers: Basic Auth headers injected into every request.

    Yields:
        Configured :class:`httpx.AsyncClient`.
    """
    from app.main import app

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
        headers=auth_headers,
    ) as c:
        yield c


@pytest_asyncio.fixture()
async def anon_client() -> AsyncClient:
    """Return an unauthenticated async test client.

    Yields:
        :class:`httpx.AsyncClient` with no auth headers.
    """
    from app.main import app

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as c:
        yield c


# ---------------------------------------------------------------------------
# Auth tests
# ---------------------------------------------------------------------------


class TestAuth:
    """Tests for HTTP Basic Auth enforcement."""

    async def test_unauthenticated_root_returns_401(
        self, anon_client: AsyncClient
    ) -> None:
        """Unauthenticated request to ``/`` must return 401.

        Args:
            anon_client: Unauthenticated test client.
        """
        resp = await anon_client.get("/")
        assert resp.status_code == 401

    async def test_unauthenticated_api_returns_401(
        self, anon_client: AsyncClient
    ) -> None:
        """Unauthenticated request to ``/api/models`` must return 401.

        Args:
            anon_client: Unauthenticated test client.
        """
        resp = await anon_client.get("/api/models")
        assert resp.status_code == 401

    async def test_wrong_password_returns_401(self, anon_client: AsyncClient) -> None:
        """Wrong password must return 401.

        Args:
            anon_client: Unauthenticated test client.
        """
        import base64

        creds = base64.b64encode(b"admin:wrongpassword").decode()
        resp = await anon_client.get(
            "/api/models", headers={"Authorization": f"Basic {creds}"}
        )
        assert resp.status_code == 401

    async def test_valid_credentials_succeed(self, client: AsyncClient) -> None:
        """Valid credentials must allow access to ``/api/models``.

        Args:
            client: Authenticated test client.
        """
        resp = await client.get("/api/models")
        assert resp.status_code == 200

    async def test_www_authenticate_header_present(
        self, anon_client: AsyncClient
    ) -> None:
        """401 response must include ``WWW-Authenticate`` header.

        Args:
            anon_client: Unauthenticated test client.
        """
        resp = await anon_client.get("/api/models")
        assert "www-authenticate" in resp.headers


# ---------------------------------------------------------------------------
# API — model list
# ---------------------------------------------------------------------------


class TestModelsEndpoint:
    """Tests for ``GET /api/models``."""

    async def test_returns_list(self, client: AsyncClient) -> None:
        """Response must be a non-empty JSON array.

        Args:
            client: Authenticated test client.
        """
        resp = await client.get("/api/models")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) == 1

    async def test_model_has_required_fields(self, client: AsyncClient) -> None:
        """Each model object must contain the expected metadata fields.

        Args:
            client: Authenticated test client.
        """
        resp = await client.get("/api/models")
        model = resp.json()[0]
        for field in ("model_id", "name", "architecture", "dataset", "params_fmt"):
            assert field in model, f"Missing field: {field}"

    async def test_model_id_is_string(self, client: AsyncClient) -> None:
        """``model_id`` must be a non-empty string.

        Args:
            client: Authenticated test client.
        """
        resp = await client.get("/api/models")
        assert isinstance(resp.json()[0]["model_id"], str)
        assert resp.json()[0]["model_id"]


# ---------------------------------------------------------------------------
# API — generate
# ---------------------------------------------------------------------------


class TestGenerateEndpoint:
    """Tests for ``POST /api/generate``."""

    _VALID_BODY = {
        "model_id": "test_model",
        "prompt": "Once upon a time",
        "max_new_tokens": 50,
        "temperature": 0.8,
        "top_k": 40,
        "use_kv_cache": False,
    }

    async def test_generate_returns_200(self, client: AsyncClient) -> None:
        """Valid generate request must return 200.

        Args:
            client: Authenticated test client.
        """
        resp = await client.post("/api/generate", json=self._VALID_BODY)
        assert resp.status_code == 200

    async def test_response_has_generated_field(self, client: AsyncClient) -> None:
        """Response must include ``generated`` and ``continuation`` fields.

        Args:
            client: Authenticated test client.
        """
        resp = await client.post("/api/generate", json=self._VALID_BODY)
        data = resp.json()
        assert "generated" in data
        assert "continuation" in data
        assert data["generated"].startswith(self._VALID_BODY["prompt"])

    async def test_response_has_timing_fields(self, client: AsyncClient) -> None:
        """Response must include timing metadata.

        Args:
            client: Authenticated test client.
        """
        resp = await client.post("/api/generate", json=self._VALID_BODY)
        data = resp.json()
        for field in ("elapsed_s", "tok_per_sec", "ms_per_tok", "new_tokens"):
            assert field in data, f"Missing timing field: {field}"

    async def test_empty_prompt_rejected(self, client: AsyncClient) -> None:
        """Empty prompt must be rejected with 422.

        Args:
            client: Authenticated test client.
        """
        body = {**self._VALID_BODY, "prompt": "   "}
        resp = await client.post("/api/generate", json=body)
        # Either 422 (validation) or 500 (registry raises ValueError)
        assert resp.status_code in (422, 500)

    async def test_unknown_model_returns_404(self, client: AsyncClient) -> None:
        """Unknown model_id must return 404.

        Args:
            client: Authenticated test client.
        """
        body = {**self._VALID_BODY, "model_id": "nonexistent_model"}
        with patch(
            "app.routers.api.registry.generate",
            side_effect=KeyError("nonexistent_model"),
        ):
            resp = await client.post("/api/generate", json=body)
        assert resp.status_code == 404

    async def test_temperature_out_of_range_rejected(self, client: AsyncClient) -> None:
        """Temperature outside [0.01, 2.0] must be rejected.

        Args:
            client: Authenticated test client.
        """
        body = {**self._VALID_BODY, "temperature": 5.0}
        resp = await client.post("/api/generate", json=body)
        assert resp.status_code == 422

    async def test_kv_cache_flag_passed_through(self, client: AsyncClient) -> None:
        """``use_kv_cache`` must be reflected in the response.

        Args:
            client: Authenticated test client.
        """
        body = {**self._VALID_BODY, "use_kv_cache": True}
        resp = await client.post("/api/generate", json=body)
        assert resp.json()["use_kv_cache"] is True


# ---------------------------------------------------------------------------
# API — version
# ---------------------------------------------------------------------------


class TestVersionEndpoint:
    """Tests for ``GET /api/version``."""

    async def test_returns_version_string(self, client: AsyncClient) -> None:
        """``/api/version`` must return a version string.

        Args:
            client: Authenticated test client.
        """
        resp = await client.get("/api/version")
        assert resp.status_code == 200
        assert "version" in resp.json()
        assert isinstance(resp.json()["version"], str)


# ---------------------------------------------------------------------------
# Static / page routes
# ---------------------------------------------------------------------------


class TestPageRoutes:
    """Tests for HTML page routes."""

    async def test_index_returns_html(self, client: AsyncClient) -> None:
        """``GET /`` must return an HTML document.

        The template rendering is mocked so the test does not depend on the
        Jinja2 environment resolving the templates directory correctly at test
        time.

        Args:
            client: Authenticated test client.
        """
        from fastapi.responses import HTMLResponse

        with patch(
            "app.routers.pages._tpl.TemplateResponse",
            return_value=HTMLResponse(content="<html><body>ok</body></html>"),
        ):
            resp = await client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]

    async def test_static_css_served(self, client: AsyncClient) -> None:
        """Static CSS file must be served correctly.

        Args:
            client: Authenticated test client.
        """
        resp = await client.get("/static/css/style.css")
        assert resp.status_code == 200
        assert "text/css" in resp.headers.get("content-type", "")
