"""
Core framework module for Unified-M.

Provides the canonical data contracts, abstract model interface,
artifact versioning, and exception types that form the foundation
of the entire framework.
"""

from core.artifacts import ArtifactStore
from core.base_model import BaseMMM
from core.contracts import (
    AttributionInput,
    ChannelResult,
    ControlInput,
    IncrementalityTestInput,
    MediaSpendInput,
    MMMDataset,
    ModelMetrics,
    OutcomeInput,
    RunManifest,
)
from core.exceptions import (
    ArtifactError,
    ConnectorError,
    DataValidationError,
    ModelNotFittedError,
    PipelineError,
    UnifiedMError,
)

__all__ = [
    "MediaSpendInput",
    "OutcomeInput",
    "ControlInput",
    "IncrementalityTestInput",
    "AttributionInput",
    "MMMDataset",
    "ChannelResult",
    "ModelMetrics",
    "RunManifest",
    "BaseMMM",
    "ArtifactStore",
    "UnifiedMError",
    "DataValidationError",
    "ModelNotFittedError",
    "ConnectorError",
    "ArtifactError",
    "PipelineError",
]
