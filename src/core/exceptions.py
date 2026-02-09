"""
Custom exception types for Unified-M.

Every exception carries a machine-readable code so callers can
handle specific failure modes programmatically.
"""


class UnifiedMError(Exception):
    """Base exception for all Unified-M errors."""

    def __init__(self, message: str, code: str = "UNIFIED_M_ERROR"):
        self.code = code
        super().__init__(message)


class DataValidationError(UnifiedMError):
    """Raised when input data fails schema validation."""

    def __init__(self, message: str, field: str | None = None):
        self.field = field
        super().__init__(message, code="DATA_VALIDATION_ERROR")


class ModelNotFittedError(UnifiedMError):
    """Raised when attempting to use a model before fitting."""

    def __init__(self, model_name: str = ""):
        msg = f"Model '{model_name}' has not been fitted. Call fit() first."
        super().__init__(msg, code="MODEL_NOT_FITTED")


class ConnectorError(UnifiedMError):
    """Raised when a data connector fails to load or write."""

    def __init__(self, message: str, source: str = ""):
        self.source = source
        super().__init__(message, code="CONNECTOR_ERROR")


class ArtifactError(UnifiedMError):
    """Raised when artifact storage operations fail."""

    def __init__(self, message: str, run_id: str = ""):
        self.run_id = run_id
        super().__init__(message, code="ARTIFACT_ERROR")


class PipelineError(UnifiedMError):
    """Raised when a pipeline step fails."""

    def __init__(self, message: str, step: str = ""):
        self.step = step
        super().__init__(message, code="PIPELINE_ERROR")


class ModelRegistryError(UnifiedMError):
    """Raised when a model backend is not available or misconfigured."""

    def __init__(self, message: str, backend: str = ""):
        self.backend = backend
        super().__init__(message, code="MODEL_REGISTRY_ERROR")
