//! In-memory LRU cache with TTL for fast get/set from Python.
//!
//! Used by the serving API when Redis is not available; provides
//! lower overhead and predictable memory use vs a Python OrderedDict.

use std::num::NonZeroUsize;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use lru::LruCache;
use pyo3::prelude::*;

struct Inner {
    cache: LruCache<String, (Vec<u8>, u64)>,
    max_size: usize,
    hits: u64,
    misses: u64,
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// LRU cache with TTL, exposed to Python for the in-memory cache backend.
#[pyclass]
pub struct PyLruCache {
    inner: Mutex<Inner>,
}

#[pymethods]
impl PyLruCache {
    #[new]
    #[pyo3(signature = (max_size=256))]
    fn new(max_size: usize) -> PyResult<Self> {
        let cap = NonZeroUsize::new(max_size.max(1)).unwrap();
        Ok(Self {
            inner: Mutex::new(Inner {
                cache: LruCache::new(cap),
                max_size,
                hits: 0,
                misses: 0,
            }),
        })
    }

    /// Get value by key. Returns None if missing or expired.
    fn get(&self, key: &str) -> Option<Vec<u8>> {
        let mut g = self.inner.lock().unwrap();
        let now = now_secs();
        let expired = g.cache.get_mut(key).map(|(value, expiry)| (*expiry <= now, value.clone()));
        match expired {
            Some((false, v)) => {
                g.hits += 1;
                Some(v)
            }
            Some((true, _)) => {
                g.cache.pop(key);
                g.misses += 1;
                None
            }
            None => {
                g.misses += 1;
                None
            }
        }
    }

    /// Set key to value with optional TTL in seconds (default 300).
    fn set(&self, key: &str, value: &[u8], ttl_secs: u64) -> PyResult<()> {
        let ttl = if ttl_secs > 0 { ttl_secs } else { 300 };
        let expiry = now_secs().saturating_add(ttl);
        let mut g = self.inner.lock().unwrap();
        g.cache.put(key.to_string(), (value.to_vec(), expiry));
        Ok(())
    }

    /// Remove key from cache.
    fn delete(&self, key: &str) -> PyResult<()> {
        let mut g = self.inner.lock().unwrap();
        g.cache.pop(key);
        Ok(())
    }

    /// Remove all entries and reset hit/miss stats.
    fn clear(&self) -> PyResult<()> {
        let mut g = self.inner.lock().unwrap();
        g.cache.clear();
        g.hits = 0;
        g.misses = 0;
        Ok(())
    }

    /// Return (entries, max_size, hits, misses) for stats.
    fn stats(&self) -> (usize, usize, u64, u64) {
        let g = self.inner.lock().unwrap();
        (g.cache.len(), g.max_size, g.hits, g.misses)
    }
}
