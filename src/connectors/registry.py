"""
Persistent connector registry.

Stores connection configurations as JSON files under ``data/connectors/``.
Secrets are encrypted at rest using Fernet symmetric encryption when
``CONNECTOR_SECRET_KEY`` is set in the environment.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from loguru import logger


def _get_fernet():
    """Return a Fernet instance if the secret key env var is set."""
    key = os.getenv("CONNECTOR_SECRET_KEY", "")
    if not key:
        return None
    try:
        from cryptography.fernet import Fernet
        return Fernet(key.encode() if isinstance(key, str) else key)
    except Exception:
        return None


def _encrypt(value: str) -> str:
    f = _get_fernet()
    if f is None:
        return value
    return f.encrypt(value.encode()).decode()


def _decrypt(value: str) -> str:
    f = _get_fernet()
    if f is None:
        return value
    try:
        return f.decrypt(value.encode()).decode()
    except Exception:
        return value


_SENSITIVE_KEYS = frozenset({
    "password", "secret", "token", "key", "account_key",
    "aws_secret_access_key", "sas_token",
})


def _is_sensitive_key(key: str) -> bool:
    key_lower = key.lower()
    if key_lower in _SENSITIVE_KEYS:
        return True
    return any(part in key_lower for part in ("password", "secret", "token", "key"))


def _mask_secret(value: str) -> str:
    if not value:
        return "***"
    if len(value) <= 4:
        return "*" * len(value)
    return f"{value[:2]}{'*' * (len(value) - 4)}{value[-2:]}"


def _mask_config(cfg: dict[str, Any]) -> dict[str, Any]:
    masked: dict[str, Any] = {}
    for k, v in cfg.items():
        if _is_sensitive_key(k):
            if isinstance(v, str):
                masked[k] = _mask_secret(v)
            elif v is None:
                masked[k] = None
            else:
                masked[k] = "***"
        else:
            masked[k] = v
    return masked


def _encrypt_config(cfg: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in cfg.items():
        if isinstance(v, str) and k.lower() in _SENSITIVE_KEYS:
            out[k] = _encrypt(v)
        else:
            out[k] = v
    return out


def _decrypt_config(cfg: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in cfg.items():
        if isinstance(v, str) and k.lower() in _SENSITIVE_KEYS:
            out[k] = _decrypt(v)
        else:
            out[k] = v
    return out


class ConnectorStore:
    """
    File-backed CRUD store for saved connection configurations.

    Each connection is a JSON file: ``<base_dir>/<id>.json``.
    """

    def __init__(self, base_dir: Path | str):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def list(self) -> list[dict[str, Any]]:
        results = []
        for p in sorted(self.base_dir.glob("*.json")):
            try:
                data = json.loads(p.read_text())
                data.pop("config", None)
                results.append(data)
            except Exception:
                continue
        results.sort(key=lambda c: c.get("created_at", ""), reverse=True)
        return results

    def get(self, connector_id: str, include_secrets: bool = False) -> dict[str, Any] | None:
        path = self.base_dir / f"{connector_id}.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        decrypted = _decrypt_config(data.get("config", {}))
        data["config"] = decrypted if include_secrets else _mask_config(decrypted)
        return data

    def create(
        self,
        name: str,
        connector_type: str,
        subtype: str,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        connector_id = uuid4().hex[:12]
        record = {
            "id": connector_id,
            "name": name,
            "type": connector_type,
            "subtype": subtype,
            "config": _encrypt_config(config),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "last_tested": None,
            "status": "untested",
        }
        path = self.base_dir / f"{connector_id}.json"
        path.write_text(json.dumps(record, indent=2))
        logger.info(f"Created connector '{name}' ({connector_type}/{subtype}) id={connector_id}")
        safe = {**record, "config": {}}
        return safe

    def update(
        self,
        connector_id: str,
        name: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        existing = self._load_raw(connector_id)
        if existing is None:
            return None
        if name is not None:
            existing["name"] = name
        if config is not None:
            existing["config"] = _encrypt_config(config)
        self._save_raw(connector_id, existing)
        safe = {**existing, "config": {}}
        return safe

    def delete(self, connector_id: str) -> bool:
        path = self.base_dir / f"{connector_id}.json"
        if path.exists():
            path.unlink()
            return True
        return False

    def set_test_result(self, connector_id: str, success: bool) -> None:
        existing = self._load_raw(connector_id)
        if existing is None:
            return
        existing["last_tested"] = datetime.now(timezone.utc).isoformat()
        existing["status"] = "connected" if success else "failed"
        self._save_raw(connector_id, existing)

    def _load_raw(self, connector_id: str) -> dict[str, Any] | None:
        path = self.base_dir / f"{connector_id}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def _save_raw(self, connector_id: str, data: dict[str, Any]) -> None:
        path = self.base_dir / f"{connector_id}.json"
        path.write_text(json.dumps(data, indent=2))
