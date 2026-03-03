"""
Authentication middleware for the Unified-M API.

Behavior:
  - If ``API_AUTH_TOKEN`` is set, all ``/api/`` routes require bearer auth.
  - Public routes (``/health``, ``/docs``, ``/redoc``) are always open.
  - Connector-management routes are always protected by default, even when
    ``API_AUTH_TOKEN`` is unset, because they can access secret material.

To explicitly allow unauthenticated connector routes in local development,
set ``ALLOW_INSECURE_CONNECTOR_ROUTES=true``.
"""

from __future__ import annotations

import os
from collections.abc import Callable

from fastapi import Request
from fastapi.responses import JSONResponse
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware

# Routes that never require auth
_PUBLIC_PATHS = frozenset({
    "/health",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/",
})

_SENSITIVE_PATH_PREFIXES = (
    "/api/v1/connectors",
)


def _is_truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


class BearerAuthMiddleware(BaseHTTPMiddleware):
    """
    Optional bearer-token auth middleware.

    Reads ``API_AUTH_TOKEN`` from environment.  If empty or unset,
    auth is disabled and all requests pass through.
    """

    def __init__(self, app: Callable, token: str | None = None):
        super().__init__(app)
        self._token = token or os.getenv("API_AUTH_TOKEN", "")
        self._allow_insecure_connector_routes = _is_truthy(
            os.getenv("ALLOW_INSECURE_CONNECTOR_ROUTES")
        )
        if self._token:
            logger.info("API auth enabled (bearer token)")
        else:
            logger.info("API auth disabled (no API_AUTH_TOKEN set)")
        if self._allow_insecure_connector_routes:
            logger.warning("Insecure mode: connector routes allowed without auth")

    async def dispatch(self, request: Request, call_next: Callable):
        path = request.url.path

        # Always allow public paths
        if path in _PUBLIC_PATHS:
            return await call_next(request)

        # Sensitive routes remain protected by default even if API_AUTH_TOKEN
        # is unset, unless explicitly overridden for local development.
        if path.startswith(_SENSITIVE_PATH_PREFIXES):
            if not self._token and not self._allow_insecure_connector_routes:
                return JSONResponse(
                    status_code=401,
                    content={
                        "detail": (
                            "Connector routes require authentication. "
                            "Set API_AUTH_TOKEN and send Authorization: Bearer <token>."
                        )
                    },
                )

        # Skip auth if no token configured
        if not self._token:
            return await call_next(request)

        # Allow non-API paths (e.g. static files)
        if not path.startswith("/api"):
            return await call_next(request)

        # Check bearer token
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            provided_token = auth_header[7:]
            if provided_token == self._token:
                return await call_next(request)

        return JSONResponse(
            status_code=401,
            content={"detail": "Invalid or missing authentication token"},
        )
