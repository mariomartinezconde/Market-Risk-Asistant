"""
FastAPI middleware:
  - Request/response structured logging.
  - Optional API-key authentication.
  - CORS.
"""
from __future__ import annotations

import time
import uuid

from fastapi import Request, HTTPException, status
from fastapi.responses import Response

from app.logger import get_logger
from config import get_settings

logger = get_logger(__name__)


async def logging_middleware(request: Request, call_next) -> Response:
    request_id = str(uuid.uuid4())
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = (time.perf_counter() - start) * 1000
    logger.info(
        "http.request",
        request_id=request_id,
        method=request.method,
        path=request.url.path,
        status=response.status_code,
        duration_ms=round(elapsed, 2),
    )
    response.headers["X-Request-ID"] = request_id
    return response


async def api_key_middleware(request: Request, call_next) -> Response:
    """
    Optional API-key gate.
    Only active when API_KEY is set in environment.
    Health endpoint is always public.
    """
    settings = get_settings()
    if settings.API_KEY and request.url.path not in ("/health", "/"):
        provided = request.headers.get(settings.API_KEY_HEADER, "")
        if provided != settings.API_KEY:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing API key.",
            )
    return await call_next(request)
