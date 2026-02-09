"""
Configuration management for Unified-M.

Centralised configuration with YAML loading, environment variable
overrides, and sensible defaults.  The config drives every layer:
storage paths, model hyperparameters, reconciliation weights,
API settings, and artifact versioning.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Section configs
# ---------------------------------------------------------------------------

class StorageConfig(BaseModel):
    """Filesystem paths for data and artifacts."""

    raw_path: Path = Field(default=Path("data/raw"))
    processed_path: Path = Field(default=Path("data/processed"))
    outputs_path: Path = Field(default=Path("data/outputs"))
    models_path: Path = Field(default=Path("models"))
    runs_path: Path = Field(default=Path("runs"))


class ModelConfig(BaseModel):
    """Default model hyperparameters."""

    backend: str = Field(default="builtin", description="Model backend: builtin, pymc, meridian")
    adstock_max_lag: int = Field(default=8)
    saturation_type: str = Field(default="hill")
    n_samples: int = Field(default=1000, description="MCMC samples (Bayesian backends only)")
    n_chains: int = Field(default=4)
    target_accept: float = Field(default=0.9)


class ReconciliationConfig(BaseModel):
    """Fusion weights and settings."""

    mmm_weight: float = Field(default=0.5)
    incrementality_weight: float = Field(default=0.3)
    attribution_weight: float = Field(default=0.2)
    fusion_method: str = Field(default="weighted_average")
    confidence_threshold: float = Field(default=0.8)


class OptimizationConfig(BaseModel):
    """Budget optimiser settings."""

    method: str = Field(default="SLSQP")
    max_iterations: int = Field(default=1000)
    tolerance: float = Field(default=1e-8)
    min_channel_budget_pct: float = Field(default=0.0)
    max_channel_budget_pct: float = Field(default=1.0)


class ServerConfig(BaseModel):
    """API and UI server settings."""

    api_host: str = Field(default="127.0.0.1")
    api_port: int = Field(default=8000)
    ui_port: int = Field(default=5173)
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------

class UnifiedMConfig(BaseModel):
    """Root configuration for Unified-M."""

    project_name: str = Field(default="Unified-M")
    environment: str = Field(default="development")

    storage: StorageConfig = Field(default_factory=StorageConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    reconciliation: ReconciliationConfig = Field(default_factory=ReconciliationConfig)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)

    @classmethod
    def from_yaml(cls, path: Path | str) -> "UnifiedMConfig":
        """Load config from a YAML file."""
        with open(Path(path)) as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)

    def to_yaml(self, path: Path | str) -> None:
        """Write config to a YAML file."""
        with open(Path(path), "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)

    def to_flat_dict(self) -> dict[str, Any]:
        """Return a flat dict for use as a config snapshot in artifacts."""
        return self.model_dump()

    def ensure_directories(self) -> None:
        """Create all required directories."""
        for p in [
            self.storage.raw_path,
            self.storage.processed_path,
            self.storage.outputs_path,
            self.storage.models_path,
            self.storage.runs_path,
        ]:
            p.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------

_config: UnifiedMConfig | None = None


def get_config() -> UnifiedMConfig:
    """Return the global config instance (creates default if needed)."""
    global _config
    if _config is None:
        _config = UnifiedMConfig()
    return _config


def set_config(config: UnifiedMConfig) -> None:
    """Override the global config instance."""
    global _config
    _config = config


def load_config(path: Path | str | None = None) -> UnifiedMConfig:
    """
    Load config from file, falling back to standard locations, then defaults.
    """
    global _config

    if path is not None:
        _config = UnifiedMConfig.from_yaml(path)
    else:
        for candidate in [Path("config.yaml"), Path("config/config.yaml")]:
            if candidate.exists():
                _config = UnifiedMConfig.from_yaml(candidate)
                break
        else:
            _config = UnifiedMConfig()

    return _config
