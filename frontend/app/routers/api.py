"""REST API router.

Endpoints
---------
``GET  /api/models``
    List all discovered checkpoint files with metadata.

``POST /api/generate``
    Generate a text continuation.

``GET  /api/version``
    Return the application version string.

All endpoints require authentication (injected via the ``require_auth``
dependency).
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from app.auth import require_auth
from app.config import settings
from app.model_registry import GenerationResult, registry

router = APIRouter(prefix="/api", tags=["api"])

AuthDep = Annotated[str, Depends(require_auth)]


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class GenerateRequest(BaseModel):
    """Request body for ``POST /api/generate``.

    Attributes:
        model_id: Checkpoint identifier as returned by ``GET /api/models``.
        prompt: Input text to continue.
        max_new_tokens: Number of tokens to generate (1–1024).
        temperature: Sampling temperature (0.01–2.0).
        top_k: Top-k logit filter; 0 disables it.
        use_kv_cache: Whether to enable the KV cache for faster decoding.
    """

    model_id: str
    prompt: str = Field(min_length=1, max_length=8192)
    max_new_tokens: int = Field(default=200, ge=1, le=1024)
    temperature: float = Field(default=0.8, ge=0.01, le=2.0)
    top_k: int = Field(default=40, ge=0, le=200)
    use_kv_cache: bool = False

    @field_validator("prompt")
    @classmethod
    def _prompt_not_blank(cls, v: str) -> str:
        """Reject prompts that contain only whitespace.

        Args:
            v: Raw prompt string from the request body.

        Returns:
            The original string when it contains at least one non-whitespace
            character.

        Raises:
            :exc:`ValueError`: When the stripped prompt is empty.
        """
        if not v.strip():
            raise ValueError("Prompt must not be blank.")
        return v

    @field_validator("max_new_tokens")
    @classmethod
    def _cap_tokens(cls, v: int) -> int:
        """Clamp ``max_new_tokens`` to the server-side hard limit.

        Args:
            v: Requested token count.

        Returns:
            Value clamped to ``settings.max_new_tokens_limit``.
        """
        return min(v, settings.max_new_tokens_limit)


class GenerateResponse(BaseModel):
    """Response body for ``POST /api/generate``.

    Attributes:
        prompt: The original prompt text.
        generated: Full generated text (prompt + continuation).
        continuation: The newly generated portion only.
        new_tokens: Number of generated tokens.
        elapsed_s: Wall-clock seconds for the generation call.
        tok_per_sec: Tokens per second throughput.
        ms_per_tok: Mean milliseconds per generated token.
        model_id: Identifier of the model that produced this result.
        use_kv_cache: Whether KV caching was active.
    """

    prompt: str
    generated: str
    continuation: str
    new_tokens: int
    elapsed_s: float
    tok_per_sec: float
    ms_per_tok: float
    model_id: str
    use_kv_cache: bool


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/models", response_model=list[dict[str, Any]])
async def list_models(user: AuthDep) -> list[dict[str, Any]]:
    """List all discovered checkpoint files with metadata.

    Args:
        user: Authenticated username (injected by dependency).

    Returns:
        List of model-info dictionaries.
    """
    return registry.list_models()


@router.post("/generate", response_model=GenerateResponse)
async def generate(
    body: GenerateRequest,
    user: AuthDep,
) -> GenerateResponse:
    """Generate a text continuation for the given prompt.

    Args:
        body: Generation parameters including model selection and sampling
            settings.
        user: Authenticated username (injected by dependency).

    Returns:
        :class:`GenerateResponse` with the generated text and timing stats.

    Raises:
        :class:`~fastapi.HTTPException` 404: If ``model_id`` is not found.
        :class:`~fastapi.HTTPException` 422: If the request body is invalid.
        :class:`~fastapi.HTTPException` 500: If generation fails unexpectedly.
    """
    try:
        result: GenerationResult = await registry.generate(
            model_id=body.model_id,
            prompt=body.prompt,
            max_new_tokens=body.max_new_tokens,
            temperature=body.temperature,
            top_k=body.top_k,
            use_kv_cache=body.use_kv_cache,
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model not found: {exc}",
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Generation failed: {exc}",
        ) from exc

    continuation = result.generated[len(result.prompt) :]

    return GenerateResponse(
        prompt=result.prompt,
        generated=result.generated,
        continuation=continuation,
        new_tokens=result.new_tokens,
        elapsed_s=result.elapsed_s,
        tok_per_sec=result.tok_per_sec,
        ms_per_tok=result.ms_per_tok,
        model_id=result.model_id,
        use_kv_cache=result.use_kv_cache,
    )


@router.get("/version")
async def version(user: AuthDep) -> dict[str, str]:
    """Return the running application version.

    Args:
        user: Authenticated username (injected by dependency).

    Returns:
        Dictionary with ``"version"`` key.
    """
    return {"version": settings.app_version}
