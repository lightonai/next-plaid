"""
Base client with shared logic for sync and async implementations.
"""

import json
from typing import Optional, List, Dict, Any, Union
from urllib.parse import urljoin

from .exceptions import (
    NextPlaidError,
    IndexNotFoundError,
    IndexExistsError,
    ValidationError,
    RateLimitError,
    ModelNotLoadedError,
    ServerError,
)
from .models import Document, SearchParams


class BaseNextPlaidClient:
    """
    Base class with shared logic for Next Plaid API clients.

    This class handles:
    - URL construction
    - Request payload preparation
    - Response parsing and error handling

    Subclasses must implement:
    - _request() method (sync or async)
    - close() method
    - Context manager methods
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        timeout: float = 30.0,
        headers: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize the base client.

        Args:
            base_url: Base URL of the Next Plaid API server.
            timeout: Request timeout in seconds.
            headers: Additional headers to include in all requests.
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._default_headers = headers or {}

    def _build_url(self, endpoint: str) -> str:
        """Build full URL from endpoint."""
        return urljoin(self.base_url + "/", endpoint.lstrip("/"))

    def _handle_response(
        self,
        status_code: int,
        content: bytes,
        text: str,
    ) -> Any:
        """
        Handle API response and raise appropriate exceptions.

        Args:
            status_code: HTTP status code
            content: Raw response content
            text: Response text for error messages

        Returns:
            Parsed JSON data or status message

        Raises:
            Appropriate NextPlaidError subclass for error responses
        """
        if status_code == 202:
            # Accepted - background processing
            return text.strip('"')

        if status_code in (200, 201):
            if content:
                return json.loads(content)
            return None

        # Handle errors
        try:
            error_data = json.loads(content)
            code = error_data.get("code", "UNKNOWN_ERROR")
            message = error_data.get("message", "Unknown error")
            details = error_data.get("details")
        except (ValueError, KeyError, json.JSONDecodeError):
            code = "UNKNOWN_ERROR"
            message = text or f"HTTP {status_code}"
            details = None

        if status_code == 404:
            if "INDEX_NOT_FOUND" in code or "not found" in message.lower():
                raise IndexNotFoundError(
                    message, code=code, details=details, status_code=404
                )
            raise NextPlaidError(message, code=code, details=details, status_code=404)

        if status_code == 409:
            raise IndexExistsError(message, code=code, details=details, status_code=409)

        if status_code == 400:
            raise ValidationError(message, code=code, details=details, status_code=400)

        if status_code == 429:
            raise RateLimitError(message, code=code, details=details, status_code=429)

        if status_code == 503:
            if "MODEL" in code.upper() or "model" in message.lower():
                raise ModelNotLoadedError(
                    message, code=code, details=details, status_code=503
                )
            raise ServerError(message, code=code, details=details, status_code=503)

        if status_code >= 500:
            raise ServerError(
                message, code=code, details=details, status_code=status_code
            )

        raise NextPlaidError(
            message, code=code, details=details, status_code=status_code
        )

    def _prepare_documents_payload(
        self,
        documents: List[Union[Document, Dict[str, List[List[float]]]]],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Prepare payload for document operations."""
        docs = [d.to_dict() if isinstance(d, Document) else d for d in documents]
        payload: Dict[str, Any] = {"documents": docs}
        if metadata:
            payload["metadata"] = metadata
        return payload

    def _prepare_search_payload(
        self,
        queries: List[Union[Dict[str, List[List[float]]], List[List[float]]]],
        params: Optional[SearchParams] = None,
        subset: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """Prepare payload for search operations."""
        query_dicts = []
        for q in queries:
            if isinstance(q, dict) and "embeddings" in q:
                query_dicts.append(q)
            else:
                query_dicts.append({"embeddings": q})

        payload: Dict[str, Any] = {"queries": query_dicts}
        if params:
            payload["params"] = params.to_dict()
        if subset:
            payload["subset"] = subset
        return payload

    def _prepare_filtered_search_payload(
        self,
        queries: List[Union[Dict[str, List[List[float]]], List[List[float]]]],
        filter_condition: str,
        filter_parameters: Optional[List[Any]] = None,
        params: Optional[SearchParams] = None,
    ) -> Dict[str, Any]:
        """Prepare payload for filtered search operations."""
        query_dicts = []
        for q in queries:
            if isinstance(q, dict) and "embeddings" in q:
                query_dicts.append(q)
            else:
                query_dicts.append({"embeddings": q})

        payload: Dict[str, Any] = {
            "queries": query_dicts,
            "filter_condition": filter_condition,
        }
        if filter_parameters:
            payload["filter_parameters"] = filter_parameters
        if params:
            payload["params"] = params.to_dict()
        return payload
