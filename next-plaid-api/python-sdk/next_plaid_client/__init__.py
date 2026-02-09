"""
Next Plaid Client - A Python client library for the Next Plaid ColBERT Search API.
"""

from .client import NextPlaidClient
from .async_client import AsyncNextPlaidClient
from .exceptions import (
    NextPlaidError,
    IndexNotFoundError,
    IndexExistsError,
    ValidationError,
    RateLimitError,
    ModelNotLoadedError,
)
from .models import (
    IndexConfig,
    IndexInfo,
    HealthResponse,
    SearchParams,
    SearchResult,
    QueryResult,
    Document,
    MetadataResponse,
    RerankResult,
    RerankResponse,
)

__version__ = "1.0.3"
__all__ = [
    "NextPlaidClient",
    "AsyncNextPlaidClient",
    "NextPlaidError",
    "IndexNotFoundError",
    "IndexExistsError",
    "ValidationError",
    "RateLimitError",
    "ModelNotLoadedError",
    "IndexConfig",
    "IndexInfo",
    "HealthResponse",
    "SearchParams",
    "SearchResult",
    "QueryResult",
    "Document",
    "MetadataResponse",
    "RerankResult",
    "RerankResponse",
]
