"""HTTP Basic Authentication and session-cookie management.

Authentication flow
-------------------
1. Every request passes through :func:`require_auth`.
2. If a valid ``session`` cookie is present the request proceeds.
3. If the cookie is missing or expired the client receives a
   ``401 Unauthorized`` response with a ``WWW-Authenticate: Basic`` header,
   which causes browsers to display the native login dialog.
4. On a successful ``POST /auth/login`` the server issues a signed session
   cookie (HMAC-SHA256 via :mod:`python-jose`) and redirects to the
   originally requested URL.
5. ``POST /auth/logout`` clears the cookie.

The password may be stored as a plain string (``ATTNRES_AUTH_PASSWORD``) or as
a bcrypt hash (``ATTNRES_AUTH_PASSWORD_HASH``).  The hash takes precedence when
both are set.

Example environment variables::

    ATTNRES_AUTH_USERNAME=admin
    ATTNRES_AUTH_PASSWORD_HASH=$2b$12$…   # bcrypt hash
    ATTNRES_SECRET_KEY=<openssl rand -hex 32>
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import Cookie, Depends, HTTPException, Request, status
from fastapi.responses import Response
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext

from app.config import settings

_pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")
_security = HTTPBasic(auto_error=False)

_ALGORITHM = "HS256"
_COOKIE_NAME = "session"


# ---------------------------------------------------------------------------
# Password verification
# ---------------------------------------------------------------------------


def _verify_password(plain: str) -> bool:
    """Verify ``plain`` against the configured credential.

    Checks the bcrypt hash when ``auth_password_hash`` is set, otherwise
    falls back to a plain-text comparison.

    Args:
        plain: The password supplied by the user.

    Returns:
        ``True`` if the password matches.
    """
    if settings.auth_password_hash:
        return _pwd_ctx.verify(plain, settings.auth_password_hash)
    return plain == settings.auth_password


def _check_credentials(username: str, password: str) -> bool:
    """Return ``True`` when both ``username`` and ``password`` are valid.

    Args:
        username: Supplied username.
        password: Supplied password.

    Returns:
        ``True`` on a successful match.
    """
    return username == settings.auth_username and _verify_password(password)


# ---------------------------------------------------------------------------
# JWT session tokens
# ---------------------------------------------------------------------------


def _create_session_token(username: str) -> str:
    """Issue a short-lived JWT that serves as the session cookie payload.

    Args:
        username: Authenticated username to embed in the token.

    Returns:
        Signed JWT string.
    """
    expire = datetime.now(timezone.utc) + timedelta(
        minutes=settings.session_expire_minutes
    )
    payload = {"sub": username, "exp": expire}
    return jwt.encode(payload, settings.secret_key, algorithm=_ALGORITHM)


def _decode_session_token(token: str) -> str | None:
    """Decode and validate a session JWT.

    Args:
        token: JWT string from the session cookie.

    Returns:
        The ``sub`` (username) claim on success, or ``None`` if the token is
        invalid or expired.
    """
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[_ALGORITHM])
        return payload.get("sub")
    except JWTError:
        return None


# ---------------------------------------------------------------------------
# FastAPI dependencies
# ---------------------------------------------------------------------------


def _unauthorized(request: Request) -> Response:
    """Build a ``401`` response that triggers the browser's Basic Auth dialog.

    Args:
        request: The current request (used to pass ``next`` in the redirect).

    Returns:
        :class:`~fastapi.responses.Response` with status 401 and the
        ``WWW-Authenticate`` header set.
    """
    return Response(
        content="Authentication required",
        status_code=status.HTTP_401_UNAUTHORIZED,
        headers={
            "WWW-Authenticate": f'Basic realm="{settings.auth_realm}"',
        },
    )


async def require_auth(
    request: Request,
    session: str | None = Cookie(default=None),
    credentials: HTTPBasicCredentials | None = Depends(_security),
) -> str:
    """FastAPI dependency that enforces authentication on every request.

    Checks (in order):

    1. A valid ``session`` JWT cookie.
    2. HTTP Basic Auth credentials in the ``Authorization`` header.

    If neither is present or valid, returns a 401 response.

    Args:
        request: The incoming HTTP request.
        session: Optional session cookie value.
        credentials: Optional HTTP Basic Auth credentials.

    Returns:
        The authenticated username string.

    Raises:
        :class:`~fastapi.HTTPException`: With status 401 when unauthenticated.
    """
    # 1. Check session cookie
    if session:
        username = _decode_session_token(session)
        if username:
            return username

    # 2. Check Basic Auth header (browser native dialog or API clients)
    if credentials:
        if _check_credentials(credentials.username, credentials.password):
            return credentials.username

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required",
        headers={"WWW-Authenticate": f'Basic realm="{settings.auth_realm}"'},
    )


def set_session_cookie(response: Response, username: str) -> None:
    """Write a signed session cookie onto ``response``.

    Args:
        response: The response object to modify in-place.
        username: Authenticated username to embed in the token.
    """
    token = _create_session_token(username)
    response.set_cookie(
        key=_COOKIE_NAME,
        value=token,
        httponly=True,
        secure=False,  # set True behind TLS/HTTPS
        samesite="lax",
        max_age=settings.session_expire_minutes * 60,
    )


def clear_session_cookie(response: Response) -> None:
    """Remove the session cookie from ``response``.

    Args:
        response: The response object to modify in-place.
    """
    response.delete_cookie(_COOKIE_NAME)
