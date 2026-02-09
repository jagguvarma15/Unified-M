"""
Model registry -- discovers and instantiates MMM backends.

Backends register themselves by name.  The pipeline asks for a backend
by string (e.g. ``"builtin"``, ``"pymc"``, ``"meridian"``) and the
registry returns a ready-to-use ``BaseMMM`` instance.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from core.base_model import BaseMMM
from core.exceptions import ModelRegistryError


# ---------------------------------------------------------------------------
# Registry storage
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, type[BaseMMM]] = {}


def register_backend(name: str, cls: type[BaseMMM]) -> None:
    """
    Register a model backend class under *name*.

    Called at import time by each adapter module.
    """
    _REGISTRY[name.lower()] = cls
    logger.debug(f"Registered model backend: {name}")


def list_backends() -> list[str]:
    """Return the names of all registered (and importable) backends."""
    _auto_discover()
    return sorted(_REGISTRY.keys())


def get_model(name: str = "builtin", **kwargs: Any) -> BaseMMM:
    """
    Instantiate a model backend by name.

    Args:
        name:    One of ``list_backends()``.
        **kwargs: Passed to the backend constructor.

    Returns:
        A ready-to-use ``BaseMMM`` instance.

    Raises:
        ModelRegistryError: If the backend is unknown or cannot be imported.
    """
    _auto_discover()
    key = name.lower()

    if key not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys())) or "(none)"
        raise ModelRegistryError(
            f"Unknown model backend '{name}'. Available: {available}",
            backend=name,
        )

    cls = _REGISTRY[key]
    logger.info(f"Creating model backend: {cls.__name__}")
    return cls(**kwargs)


# ---------------------------------------------------------------------------
# Auto-discovery of built-in adapters
# ---------------------------------------------------------------------------

_discovered = False


def _auto_discover() -> None:
    """Import adapter modules so they self-register."""
    global _discovered
    if _discovered:
        return
    _discovered = True

    # Built-in (always available)
    try:
        import models.builtin  # noqa: F401
    except Exception as exc:
        logger.warning(f"Could not load builtin model: {exc}")

    # PyMC-Marketing adapter (optional)
    try:
        import models.pymc_adapter  # noqa: F401
    except ImportError:
        logger.debug("PyMC-Marketing adapter not available (pymc-marketing not installed)")
    except Exception as exc:
        logger.warning(f"Could not load PyMC adapter: {exc}")

    # Future: Meridian, Robyn, etc.
