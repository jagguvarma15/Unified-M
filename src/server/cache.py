"""
Cache layer for Unified-M serving API.

Provides a unified cache interface with two backends:
  1. Redis (production: shared across workers, persistent)
  2. In-memory LRU (development: no external deps needed)

The API server automatically uses Redis if REDIS_URL is set,
otherwise falls back to in-memory.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any

from loguru import logger


class CacheBackend(ABC):
    """Abstract cache interface."""

    @abstractmethod
    def get(self, key: str) -> Any | None:
        ...

    @abstractmethod
    def set(self, key: str, value: Any, ttl: int = 300) -> None:
        ...

    @abstractmethod
    def delete(self, key: str) -> None:
        ...

    @abstractmethod
    def clear(self) -> None:
        ...

    @abstractmethod
    def stats(self) -> dict[str, Any]:
        ...


class InMemoryCache(CacheBackend):
    """
    Simple in-memory LRU cache.

    Thread-safe enough for single-worker development use.
    For multi-worker production, use Redis.
    """

    def __init__(self, max_size: int = 256):
        self._max_size = max_size
        self._cache: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Any | None:
        if key in self._cache:
            value, expiry = self._cache[key]
            if expiry > time.time():
                self._hits += 1
                self._cache.move_to_end(key)
                return value
            else:
                del self._cache[key]
        self._misses += 1
        return None

    def set(self, key: str, value: Any, ttl: int = 300) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = (value, time.time() + ttl)
        if len(self._cache) > self._max_size:
            self._cache.popitem(last=False)

    def delete(self, key: str) -> None:
        self._cache.pop(key, None)

    def clear(self) -> None:
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    def stats(self) -> dict[str, Any]:
        total = self._hits + self._misses
        return {
            "backend": "in_memory",
            "entries": len(self._cache),
            "max_size": self._max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / total, 4) if total > 0 else 0,
        }


class RedisCache(CacheBackend):
    """
    Redis-backed cache for production serving.

    Requires: pip install redis
    Configured via REDIS_URL env var.
    """

    def __init__(self, url: str | None = None, prefix: str = "umm:"):
        try:
            import redis
        except ImportError:
            raise ImportError(
                "redis package not installed. "
                "Run: pip install redis"
            )

        self._url = url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self._prefix = prefix
        self._client = redis.from_url(self._url, decode_responses=True)
        self._hits = 0
        self._misses = 0

        # Test connection
        try:
            self._client.ping()
            logger.info(f"Redis cache connected: {self._url}")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
            raise

    def _key(self, key: str) -> str:
        return f"{self._prefix}{key}"

    def get(self, key: str) -> Any | None:
        raw = self._client.get(self._key(key))
        if raw is not None:
            self._hits += 1
            return json.loads(raw)
        self._misses += 1
        return None

    def set(self, key: str, value: Any, ttl: int = 300) -> None:
        self._client.setex(
            self._key(key),
            ttl,
            json.dumps(value, default=str),
        )

    def delete(self, key: str) -> None:
        self._client.delete(self._key(key))

    def clear(self) -> None:
        keys = self._client.keys(f"{self._prefix}*")
        if keys:
            self._client.delete(*keys)
        self._hits = 0
        self._misses = 0

    def stats(self) -> dict[str, Any]:
        info = self._client.info("memory")
        total = self._hits + self._misses
        return {
            "backend": "redis",
            "url": self._url,
            "used_memory": info.get("used_memory_human", "?"),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / total, 4) if total > 0 else 0,
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_cache_instance: CacheBackend | None = None


def get_cache() -> CacheBackend:
    """
    Get or create the global cache instance.

    Uses Redis if REDIS_URL is set and reachable, otherwise in-memory.
    """
    global _cache_instance
    if _cache_instance is not None:
        return _cache_instance

    redis_url = os.getenv("REDIS_URL")
    if redis_url:
        try:
            _cache_instance = RedisCache(url=redis_url)
            return _cache_instance
        except Exception as e:
            logger.warning(f"Redis unavailable ({e}), falling back to in-memory cache")

    _cache_instance = InMemoryCache(max_size=256)
    logger.info("Using in-memory cache (set REDIS_URL for Redis)")
    return _cache_instance


def cache_key(*parts: str) -> str:
    """Build a deterministic cache key from parts."""
    raw = ":".join(str(p) for p in parts)
    return hashlib.md5(raw.encode()).hexdigest()
