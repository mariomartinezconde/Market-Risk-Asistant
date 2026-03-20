"""Request logging and optional API-key middleware."""
from __future__ import annotations

import time
import uuid

from fastapi import Request, HTTPException, status
from fastapi.responses import Response

from app.core.logging import get_logger

logger = get_logger(__name__)


async def logging_middleware(request: Request, call_next) -> Response:
    rid = str(uuid.uuid4())[:8]
    t0 = time.perf_counter()
    response = await call_next(request)
    ms = round((time.perf_counter() - t0) * 1000, 1)
    logger.info("http", rid=rid, method=request.method, path=request.url.path,
                status=response.status_code, ms=ms)
    response.headers["X-Request-ID"] = rid
    return response
