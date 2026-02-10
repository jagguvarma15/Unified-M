"""
Authentication middleware for the Unified-M API.

Provides optional bearer-token authentication.  If ``API_AUTH_TOKEN``
is set, all ``/api/`` routes require ``Authorization: Bearer <token>``.
Public routes (``/health``, ``/docs``, ``/redoc``) are always open.

For local development, leave ``API_AUTH_TOKEN`` unset to disable auth.
"""

from __future__ import annotations

import os
from typing import Callable

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


class BearerAuthMiddleware(BaseHTTPMiddleware):
    """
    Optional bearer-token auth middleware.

    Reads ``API_AUTH_TOKEN`` from environment.  If empty or unset,
    auth is disabled and all requests pass through.
    """

    def __init__(self, app: Callable, token: str | None = None):
        super().__init__(app)
        self._token = token or os.getenv("API_AUTH_TOKEN", "")
        if self._token:
            logger.info("API auth enabled (bearer token)")
        else:
            logger.info("API auth disabled (no API_AUTH_TOKEN set)")

    async def dispatch(self, request: Request, call_next: Callable):
        # Skip auth if no token configured
        if not self._token:
            return await call_next(request)

        # Always allow public paths
        if request.url.path in _PUBLIC_PATHS:
            return await call_next(request)

        # Allow non-API paths (e.g. static files)
        if not request.url.path.startswith("/api"):
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
