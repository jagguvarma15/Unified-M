"""
Configuration management for Unified-M.

Centralized configuration with sensible defaults and environment variable overrides.
"""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class StorageConfig(BaseModel):
    """Storage layer configuration."""
    
    raw_path: Path = Field(default=Path("data/raw"))
    validated_path: Path = Field(default=Path("data/validated"))
    transformed_path: Path = Field(default=Path("data/transformed"))
    outputs_path: Path = Field(default=Path("data/outputs"))
    models_path: Path = Field(default=Path("models"))


class MMMConfig(BaseModel):
    """MMM model configuration."""
    
    # Adstock parameters
    adstock_max_lag: int = Field(default=8, description="Maximum lag for adstock transformation")
    
    # Saturation parameters
    saturation_type: str = Field(default="logistic", description="Type of saturation curve")
    
    # Sampling parameters
    n_samples: int = Field(default=1000, description="Number of posterior samples")
    n_chains: int = Field(default=4, description="Number of MCMC chains")
    target_accept: float = Field(default=0.9, description="Target acceptance rate")
    
    # Priors (can be overridden per channel)
    default_adstock_alpha_prior: tuple[float, float] = Field(
        default=(0.0, 1.0), description="Beta prior for adstock decay"
    )
    default_saturation_lambda_prior: tuple[float, float] = Field(
        default=(0.5, 1.0), description="Gamma prior for saturation parameter"
    )


class ReconciliationConfig(BaseModel):
    """Reconciliation layer configuration."""
    
    # Weighting scheme for fusion
    mmm_weight: float = Field(default=0.5, description="Weight for MMM estimates")
    incrementality_weight: float = Field(default=0.3, description="Weight for incrementality tests")
    attribution_weight: float = Field(default=0.2, description="Weight for attribution signals")
    
    # Calibration settings
    use_hierarchical_calibration: bool = Field(default=False)
    confidence_threshold: float = Field(default=0.8)


class OptimizationConfig(BaseModel):
    """Budget optimization configuration."""
    
    method: str = Field(default="SLSQP", description="scipy.optimize method")
    max_iterations: int = Field(default=1000)
    tolerance: float = Field(default=1e-8)
    
    # Constraints
    min_channel_budget_pct: float = Field(default=0.0, description="Minimum % of total budget per channel")
    max_channel_budget_pct: float = Field(default=1.0, description="Maximum % of total budget per channel")


class APIConfig(BaseModel):
    """API configuration."""
    
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    reload: bool = Field(default=False)


class UIConfig(BaseModel):
    """Streamlit UI configuration."""
    
    port: int = Field(default=8501)
    theme: str = Field(default="dark")


class UnifiedMConfig(BaseModel):
    """Root configuration for Unified-M."""
    
    project_name: str = Field(default="Unified-M")
    environment: str = Field(default="development")
    
    storage: StorageConfig = Field(default_factory=StorageConfig)
    mmm: MMMConfig = Field(default_factory=MMMConfig)
    reconciliation: ReconciliationConfig = Field(default_factory=ReconciliationConfig)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    ui: UIConfig = Field(default_factory=UIConfig)
    
    @classmethod
    def from_yaml(cls, path: Path) -> "UnifiedMConfig":
        """Load configuration from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    def to_yaml(self, path: Path) -> None:
        """Save configuration to YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)
    
    def ensure_directories(self) -> None:
        """Create all required directories."""
        for path in [
            self.storage.raw_path,
            self.storage.validated_path,
            self.storage.transformed_path,
            self.storage.outputs_path,
            self.storage.models_path,
        ]:
            path.mkdir(parents=True, exist_ok=True)


# Global config instance (can be overridden)
_config: UnifiedMConfig | None = None


def get_config() -> UnifiedMConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = UnifiedMConfig()
    return _config


def set_config(config: UnifiedMConfig) -> None:
    """Set the global configuration instance."""
    global _config
    _config = config


def load_config(path: Path | str | None = None) -> UnifiedMConfig:
    """Load configuration from file or use defaults."""
    global _config
    
    if path is not None:
        _config = UnifiedMConfig.from_yaml(Path(path))
    else:
        # Check for config file in standard locations
        for config_path in [Path("config.yaml"), Path("config/config.yaml")]:
            if config_path.exists():
                _config = UnifiedMConfig.from_yaml(config_path)
                break
        else:
            _config = UnifiedMConfig()
    
    return _config

