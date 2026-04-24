"""HTML page router.

Serves the single-page chat interface and handles the auth login/logout flow.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates

from app.auth import clear_session_cookie, require_auth
from app.config import settings
from app.model_registry import registry

router = APIRouter(tags=["pages"])

# Resolve the templates directory relative to this file's location:
# pages.py lives at app/routers/pages.py → parent.parent is app/ → app/templates/
_tpl = Jinja2Templates(
    directory=str(Path(__file__).resolve().parent.parent / "templates")
)

AuthDep = Annotated[str, Depends(require_auth)]


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


@router.get("/", response_class=HTMLResponse)
async def index(request: Request, user: AuthDep) -> HTMLResponse:
    """Render the main chat interface.

    Args:
        request: The incoming HTTP request.
        user: Authenticated username (injected by dependency).

    Returns:
        HTML response with the rendered chat page.
    """
    models = registry.list_models()
    return _tpl.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "user": user,
            "models": models,
            "app_title": settings.app_title,
            "max_tokens_limit": settings.max_new_tokens_limit,
        },
    )


@router.get("/logout")
async def logout(request: Request) -> Response:
    """Clear the session cookie and redirect to the root.

    Args:
        request: The incoming HTTP request.

    Returns:
        Redirect to ``/`` with the session cookie cleared.
    """
    response = RedirectResponse(url="/", status_code=status.HTTP_302_FOUND)
    clear_session_cookie(response)
    return response
