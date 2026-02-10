"""
Cloud storage connectors for Unified-M.

Supports AWS S3, Azure Blob Storage, and Google Cloud Storage.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any
from pathlib import Path
import pandas as pd
from loguru import logger

from core.exceptions import ConnectorError


class CloudStorageConnector(ABC):
    """Base class for cloud storage connectors."""

    @abstractmethod
    def load(self, path: str, **kwargs: Any) -> pd.DataFrame:
        """Load file from cloud storage."""
        ...

    @abstractmethod
    def test_connection(self) -> bool:
        """Test if connection is valid."""
        ...


class S3Connector(CloudStorageConnector):
    """AWS S3 connector."""

    def __init__(self, bucket: str, aws_access_key_id: str | None = None,
                 aws_secret_access_key: str | None = None,
                 region_name: str = "us-east-1", **kwargs: Any):
        try:
            import boto3
        except ImportError:
            raise ConnectorError(
                "boto3 is not installed. Run: pip install boto3"
            )
        
        self.bucket = bucket
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.region_name = region_name
        self.kwargs = kwargs
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import boto3
            self._client = boto3.client(
                's3',
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key,
                region_name=self.region_name,
                **self.kwargs
            )
        return self._client

    def load(self, path: str, **kwargs: Any) -> pd.DataFrame:
        """Load file from S3 bucket."""
        import io
        
        logger.info(f"Loading {path} from S3 bucket {self.bucket}")
        
        try:
            obj = self.client.get_object(Bucket=self.bucket, Key=path)
            data = obj['Body'].read()
            
            # Determine file type
            path_lower = path.lower()
            if path_lower.endswith('.parquet'):
                return pd.read_parquet(io.BytesIO(data), **kwargs)
            elif path_lower.endswith('.csv'):
                return pd.read_csv(io.BytesIO(data), **kwargs)
            elif path_lower.endswith(('.xlsx', '.xls')):
                return pd.read_excel(io.BytesIO(data), **kwargs)
            else:
                raise ConnectorError(f"Unsupported file type: {path}")
        except Exception as e:
            raise ConnectorError(f"Failed to load from S3: {e}")

    def test_connection(self) -> bool:
        """Test S3 connection by listing bucket."""
        try:
            self.client.head_bucket(Bucket=self.bucket)
            return True
        except Exception as e:
            logger.error(f"S3 connection test failed: {e}")
            return False


class AzureBlobConnector(CloudStorageConnector):
    """Azure Blob Storage connector."""

    def __init__(self, account_name: str, container_name: str,
                 account_key: str | None = None, sas_token: str | None = None,
                 **kwargs: Any):
        try:
            from azure.storage.blob import BlobServiceClient
        except ImportError:
            raise ConnectorError(
                "azure-storage-blob is not installed. Run: pip install azure-storage-blob"
            )
        
        self.account_name = account_name
        self.container_name = container_name
        self.account_key = account_key
        self.sas_token = sas_token
        self.kwargs = kwargs
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from azure.storage.blob import BlobServiceClient
            
            if self.account_key:
                conn_str = (
                    f"DefaultEndpointsProtocol=https;"
                    f"AccountName={self.account_name};"
                    f"AccountKey={self.account_key};"
                    f"EndpointSuffix=core.windows.net"
                )
                self._client = BlobServiceClient.from_connection_string(conn_str)
            elif self.sas_token:
                account_url = f"https://{self.account_name}.blob.core.windows.net"
                self._client = BlobServiceClient(
                    account_url=account_url,
                    credential=self.sas_token
                )
            else:
                raise ConnectorError("Either account_key or sas_token must be provided")
        
        return self._client

    def load(self, path: str, **kwargs: Any) -> pd.DataFrame:
        """Load file from Azure Blob Storage."""
        import io
        
        logger.info(f"Loading {path} from Azure container {self.container_name}")
        
        try:
            blob_client = self.client.get_blob_client(
                container=self.container_name,
                blob=path
            )
            data = blob_client.download_blob().readall()
            
            # Determine file type
            path_lower = path.lower()
            if path_lower.endswith('.parquet'):
                return pd.read_parquet(io.BytesIO(data), **kwargs)
            elif path_lower.endswith('.csv'):
                return pd.read_csv(io.BytesIO(data), **kwargs)
            elif path_lower.endswith(('.xlsx', '.xls')):
                return pd.read_excel(io.BytesIO(data), **kwargs)
            else:
                raise ConnectorError(f"Unsupported file type: {path}")
        except Exception as e:
            raise ConnectorError(f"Failed to load from Azure Blob: {e}")

    def test_connection(self) -> bool:
        """Test Azure Blob connection."""
        try:
            container_client = self.client.get_container_client(self.container_name)
            container_client.get_container_properties()
            return True
        except Exception as e:
            logger.error(f"Azure Blob connection test failed: {e}")
            return False


def create_cloud_connector(
    cloud_type: str,
    **kwargs: Any
) -> CloudStorageConnector:
    """Factory function to create appropriate cloud connector."""
    cloud_type_lower = cloud_type.lower()
    
    if cloud_type_lower in ["s3", "aws", "aws s3"]:
        return S3Connector(**kwargs)
    elif cloud_type_lower in ["azure", "azure blob", "azureblob"]:
        return AzureBlobConnector(**kwargs)
    else:
        raise ConnectorError(f"Unsupported cloud storage type: {cloud_type}")
