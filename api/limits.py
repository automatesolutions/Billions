"""
Per-IP rate limits. Behind Railway/Render, uvicorn runs with --proxy-headers so
the client address is the real visitor, not the proxy.
"""

from slowapi import Limiter
from slowapi.util import get_remote_address

from api.config import settings

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[settings.RATE_LIMIT_DEFAULT],
    enabled=settings.RATE_LIMIT_ENABLED,
)
