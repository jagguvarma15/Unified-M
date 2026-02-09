"""
Core framework module for Unified-M.

Provides the canonical data contracts, abstract model interface,
artifact versioning, and exception types that form the foundation
of the entire framework.
"""

from core.contracts import (
    MediaSpendInput,
    OutcomeInput,
    ControlInput,
    IncrementalityTestInput,
    AttributionInput,
    MMMDataset,
    ChannelResult,
    ModelMetrics,
    RunManifest,
)
from core.base_model import BaseMMM
from core.artifacts import ArtifactStore
from core.exceptions import (
    UnifiedMError,
    DataValidationError,
    ModelNotFittedError,
    ConnectorError,
    ArtifactError,
    PipelineError,
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
