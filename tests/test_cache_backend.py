"""Regression tests for cache backend behavior."""

from __future__ import annotations

from server.cache import RedisCache


class _FakeRedis:
    def __init__(self, keys: list[str]):
        self._keys = keys
        self.deleted_calls: list[tuple[str, ...]] = []

    def scan_iter(self, match: str, count: int = 1000):
        del match, count
        for k in self._keys:
            yield k

    def delete(self, *keys: str):
        self.deleted_calls.append(tuple(keys))
        return len(keys)

    def keys(self, pattern: str):  # pragma: no cover - should never be called
        raise AssertionError("Redis KEYS must not be used in clear()")


def test_redis_clear_uses_scan_and_batched_delete():
    cache = RedisCache.__new__(RedisCache)
    cache._prefix = "umm:"
    cache._client = _FakeRedis([f"umm:k{i}" for i in range(1203)])
    cache._hits = 11
    cache._misses = 7

    cache.clear()

    # 1203 keys with batch_size=500 -> 3 delete calls
    assert len(cache._client.deleted_calls) == 3
    assert len(cache._client.deleted_calls[0]) == 500
    assert len(cache._client.deleted_calls[1]) == 500
    assert len(cache._client.deleted_calls[2]) == 203
    assert cache._hits == 0
    assert cache._misses == 0
