"""
Synchronous client for the Next Plaid API.
"""

import httpx
from typing import Optional, List, Dict, Any, Union

from ._base import BaseNextPlaidClient
from .exceptions import ConnectionError as NextPlaidConnectionError
from .models import (
    IndexConfig,
    IndexInfo,
    HealthResponse,
    SearchParams,
    SearchResult,
    Document,
    MetadataResponse,
    MetadataCheckResponse,
    EncodeResponse,
)


class NextPlaidClient(BaseNextPlaidClient):
    """
    Synchronous client for the Next Plaid ColBERT Search API.

    Example usage:
        client = NextPlaidClient("http://localhost:8080")
        health = client.health()
        print(f"Server status: {health.status}")

        # Create an index
        client.create_index("my_index", IndexConfig(nbits=4))

        # Search
        results = client.search("my_index", queries=[...])

        # Or use as context manager
        with NextPlaidClient("http://localhost:8080") as client:
            indices = client.list_indices()
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        timeout: float = 30.0,
        headers: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize the Next Plaid client.

        Args:
            base_url: Base URL of the Next Plaid API server.
            timeout: Request timeout in seconds.
            headers: Additional headers to include in all requests.
        """
        super().__init__(base_url, timeout, headers)
        self._client = httpx.Client(
            headers=self._default_headers,
            timeout=httpx.Timeout(timeout),
        )

    def _request(
        self,
        method: str,
        endpoint: str,
        json: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Make a synchronous HTTP request to the API."""
        url = self._build_url(endpoint)

        try:
            response = self._client.request(
                method=method,
                url=url,
                json=json,
                params=params,
            )
        except httpx.ConnectError as e:
            raise NextPlaidConnectionError(
                f"Failed to connect to {url}: {e}", code="CONNECTION_ERROR"
            )
        except httpx.TimeoutException as e:
            raise NextPlaidConnectionError(
                f"Request timed out: {e}", code="TIMEOUT_ERROR"
            )

        return self._handle_response(
            response.status_code,
            response.content,
            response.text,
        )

    # ==================== Health & Monitoring ====================

    def health(self) -> HealthResponse:
        """
        Check server health and get status information.

        Returns:
            HealthResponse with server status and index information.
        """
        data = self._request("GET", "/health")
        return HealthResponse.from_dict(data)

    # ==================== Index Management ====================

    def list_indices(self) -> List[str]:
        """
        List all available indices.

        Returns:
            List of index names.
        """
        return self._request("GET", "/indices")

    def get_index(self, name: str) -> IndexInfo:
        """
        Get detailed information about a specific index.

        Args:
            name: Name of the index.

        Returns:
            IndexInfo with detailed index information.

        Raises:
            IndexNotFoundError: If the index does not exist.
        """
        data = self._request("GET", f"/indices/{name}")
        return IndexInfo.from_dict(data)

    def create_index(
        self, name: str, config: Optional[IndexConfig] = None
    ) -> Dict[str, Any]:
        """
        Create a new index with the specified configuration.

        Args:
            name: Name for the new index.
            config: Index configuration (optional, uses defaults if not provided).

        Returns:
            Dict with index creation response.

        Raises:
            IndexExistsError: If an index with this name already exists.
            ValidationError: If the configuration is invalid.
        """
        payload: Dict[str, Any] = {"name": name}
        if config:
            payload["config"] = config.to_dict()
        return self._request("POST", "/indices", json=payload)

    def delete_index(self, name: str) -> Dict[str, Any]:
        """
        Delete an index and all its data.

        Args:
            name: Name of the index to delete.

        Returns:
            Dict confirming deletion.

        Raises:
            IndexNotFoundError: If the index does not exist.
        """
        return self._request("DELETE", f"/indices/{name}")

    def update_index_config(
        self, name: str, max_documents: Optional[int]
    ) -> Dict[str, Any]:
        """
        Update index configuration (e.g., max_documents limit).

        Args:
            name: Name of the index.
            max_documents: Maximum number of documents (None to remove limit).

        Returns:
            Dict with updated configuration.

        Raises:
            IndexNotFoundError: If the index does not exist.
        """
        return self._request(
            "PUT", f"/indices/{name}/config", json={"max_documents": max_documents}
        )

    # ==================== Document Management ====================

    def add_documents(
        self,
        index_name: str,
        documents: List[Union[Document, Dict[str, List[List[float]]]]],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Add documents to an index (async, returns immediately).

        Args:
            index_name: Name of the index.
            documents: List of documents with embeddings.
            metadata: Optional list of metadata dicts for each document.

        Returns:
            Status message confirming the update was queued.

        Raises:
            IndexNotFoundError: If the index does not exist.
            ValidationError: If the documents are invalid.
        """
        payload = self._prepare_documents_payload(documents, metadata)
        return self._request("POST", f"/indices/{index_name}/documents", json=payload)

    def update_documents(
        self,
        index_name: str,
        documents: List[Union[Document, Dict[str, List[List[float]]]]],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Update index by adding documents (batched, async).

        Args:
            index_name: Name of the index.
            documents: List of documents with embeddings.
            metadata: Optional list of metadata dicts for each document.

        Returns:
            Status message confirming the update was queued.

        Raises:
            IndexNotFoundError: If the index does not exist.
        """
        payload = self._prepare_documents_payload(documents, metadata)
        return self._request("POST", f"/indices/{index_name}/update", json=payload)

    def update_documents_with_encoding(
        self,
        index_name: str,
        documents: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Update index with document texts (requires model to be loaded).

        The server will encode the texts and add them to the index.

        Args:
            index_name: Name of the index.
            documents: List of document texts to encode and add.
            metadata: Optional list of metadata dicts for each document.

        Returns:
            Status message confirming the update was queued.

        Raises:
            IndexNotFoundError: If the index does not exist.
            ModelNotLoadedError: If no model is loaded on the server.
        """
        payload: Dict[str, Any] = {"documents": documents}
        if metadata:
            payload["metadata"] = metadata
        return self._request(
            "POST", f"/indices/{index_name}/update_with_encoding", json=payload
        )

    def delete_documents(
        self,
        index_name: str,
        condition: str,
        parameters: Optional[List[Any]] = None,
    ) -> str:
        """
        Delete documents matching a metadata filter condition.

        This is an asynchronous operation - it returns immediately with a status
        message and processes deletion in the background.

        Args:
            index_name: Name of the index.
            condition: SQL WHERE condition for selecting documents to delete
                      (e.g., "category = ? AND year < ?").
            parameters: Parameters for the condition placeholders.

        Returns:
            Status message indicating the delete request was queued.

        Raises:
            IndexNotFoundError: If the index does not exist.
            MetadataNotFoundError: If the index has no metadata.
            ValidationError: If the condition is invalid.
        """
        payload: Dict[str, Any] = {"condition": condition}
        if parameters:
            payload["parameters"] = parameters
        return self._request(
            "DELETE",
            f"/indices/{index_name}/documents",
            json=payload,
        )

    # ==================== Search Operations ====================

    def search(
        self,
        index_name: str,
        queries: List[Union[Dict[str, List[List[float]]], List[List[float]]]],
        params: Optional[SearchParams] = None,
        subset: Optional[List[int]] = None,
    ) -> SearchResult:
        """
        Search an index with query embeddings.

        Args:
            index_name: Name of the index to search.
            queries: List of query embeddings (each is a list of token embeddings).
            params: Search parameters (optional).
            subset: Optional list of document IDs to search within.

        Returns:
            SearchResult with results for each query.

        Raises:
            IndexNotFoundError: If the index does not exist.
        """
        payload = self._prepare_search_payload(queries, params, subset)
        data = self._request("POST", f"/indices/{index_name}/search", json=payload)
        return SearchResult.from_dict(data)

    def search_filtered(
        self,
        index_name: str,
        queries: List[Union[Dict[str, List[List[float]]], List[List[float]]]],
        filter_condition: str,
        filter_parameters: Optional[List[Any]] = None,
        params: Optional[SearchParams] = None,
    ) -> SearchResult:
        """
        Search an index with metadata filtering.

        Args:
            index_name: Name of the index to search.
            queries: List of query embeddings.
            filter_condition: SQL WHERE condition for filtering.
            filter_parameters: Parameters for the filter condition.
            params: Search parameters (optional).

        Returns:
            SearchResult with filtered results.

        Raises:
            IndexNotFoundError: If the index does not exist.
        """
        payload = self._prepare_filtered_search_payload(
            queries, filter_condition, filter_parameters, params
        )
        data = self._request(
            "POST", f"/indices/{index_name}/search/filtered", json=payload
        )
        return SearchResult.from_dict(data)

    def search_with_encoding(
        self,
        index_name: str,
        queries: List[str],
        params: Optional[SearchParams] = None,
        subset: Optional[List[int]] = None,
    ) -> SearchResult:
        """
        Search an index using text queries (requires model to be loaded).

        Args:
            index_name: Name of the index to search.
            queries: List of text queries to encode and search.
            params: Search parameters (optional).
            subset: Optional list of document IDs to search within.

        Returns:
            SearchResult with results for each query.

        Raises:
            IndexNotFoundError: If the index does not exist.
            ModelNotLoadedError: If no model is loaded on the server.
        """
        payload: Dict[str, Any] = {"queries": queries}
        if params:
            payload["params"] = params.to_dict()
        if subset:
            payload["subset"] = subset

        data = self._request(
            "POST", f"/indices/{index_name}/search_with_encoding", json=payload
        )
        return SearchResult.from_dict(data)

    def search_filtered_with_encoding(
        self,
        index_name: str,
        queries: List[str],
        filter_condition: str,
        filter_parameters: Optional[List[Any]] = None,
        params: Optional[SearchParams] = None,
    ) -> SearchResult:
        """
        Search with text queries and metadata filtering (requires model).

        Args:
            index_name: Name of the index to search.
            queries: List of text queries.
            filter_condition: SQL WHERE condition for filtering.
            filter_parameters: Parameters for the filter condition.
            params: Search parameters (optional).

        Returns:
            SearchResult with filtered results.

        Raises:
            IndexNotFoundError: If the index does not exist.
            ModelNotLoadedError: If no model is loaded on the server.
        """
        payload: Dict[str, Any] = {
            "queries": queries,
            "filter_condition": filter_condition,
        }
        if filter_parameters:
            payload["filter_parameters"] = filter_parameters
        if params:
            payload["params"] = params.to_dict()

        data = self._request(
            "POST", f"/indices/{index_name}/search/filtered_with_encoding", json=payload
        )
        return SearchResult.from_dict(data)

    # ==================== Metadata Management ====================

    def get_metadata(self, index_name: str) -> MetadataResponse:
        """
        Get all metadata entries for an index.

        Args:
            index_name: Name of the index.

        Returns:
            MetadataResponse with all metadata entries.

        Raises:
            IndexNotFoundError: If the index does not exist.
        """
        data = self._request("GET", f"/indices/{index_name}/metadata")
        return MetadataResponse.from_dict(data)

    def add_metadata(
        self, index_name: str, metadata: List[Dict[str, Any]]
    ) -> Dict[str, int]:
        """
        Add or update metadata entries.

        Args:
            index_name: Name of the index.
            metadata: List of metadata dicts to add.

        Returns:
            Dict with count of added entries.

        Raises:
            IndexNotFoundError: If the index does not exist.
        """
        return self._request(
            "POST", f"/indices/{index_name}/metadata", json={"metadata": metadata}
        )

    def get_metadata_count(self, index_name: str) -> Dict[str, Any]:
        """
        Get count of metadata entries.

        Args:
            index_name: Name of the index.

        Returns:
            Dict with count and has_metadata flag.

        Raises:
            IndexNotFoundError: If the index does not exist.
        """
        return self._request("GET", f"/indices/{index_name}/metadata/count")

    def check_metadata(
        self, index_name: str, document_ids: List[int]
    ) -> MetadataCheckResponse:
        """
        Check which documents exist in the metadata database.

        Args:
            index_name: Name of the index.
            document_ids: List of document IDs to check.

        Returns:
            MetadataCheckResponse with existing and missing IDs.

        Raises:
            IndexNotFoundError: If the index does not exist.
        """
        data = self._request(
            "POST",
            f"/indices/{index_name}/metadata/check",
            json={"document_ids": document_ids},
        )
        return MetadataCheckResponse.from_dict(data)

    def query_metadata(
        self,
        index_name: str,
        condition: str,
        parameters: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        """
        Query metadata using SQL WHERE conditions.

        Args:
            index_name: Name of the index.
            condition: SQL WHERE condition (e.g., "category = ? AND score > ?").
            parameters: Parameters for the condition placeholders.

        Returns:
            Dict with document_ids and count.

        Raises:
            IndexNotFoundError: If the index does not exist.
        """
        payload: Dict[str, Any] = {"condition": condition}
        if parameters:
            payload["parameters"] = parameters
        return self._request(
            "POST", f"/indices/{index_name}/metadata/query", json=payload
        )

    def get_metadata_by_ids(
        self,
        index_name: str,
        document_ids: Optional[List[int]] = None,
        condition: Optional[str] = None,
        parameters: Optional[List[Any]] = None,
        limit: Optional[int] = None,
    ) -> MetadataResponse:
        """
        Get metadata by document IDs or SQL condition.

        Args:
            index_name: Name of the index.
            document_ids: Optional list of document IDs.
            condition: Optional SQL WHERE condition.
            parameters: Parameters for the condition.
            limit: Maximum number of results.

        Returns:
            MetadataResponse with matching metadata.

        Raises:
            IndexNotFoundError: If the index does not exist.
        """
        payload: Dict[str, Any] = {}
        if document_ids:
            payload["document_ids"] = document_ids
        if condition:
            payload["condition"] = condition
        if parameters:
            payload["parameters"] = parameters
        if limit:
            payload["limit"] = limit

        data = self._request(
            "POST", f"/indices/{index_name}/metadata/get", json=payload
        )
        return MetadataResponse.from_dict(data)

    # ==================== Text Encoding ====================

    def encode(
        self,
        texts: List[str],
        input_type: str = "document",
    ) -> EncodeResponse:
        """
        Encode texts into ColBERT embeddings (requires model to be loaded).

        Args:
            texts: List of texts to encode.
            input_type: Either "document" or "query".

        Returns:
            EncodeResponse with embeddings for each text.

        Raises:
            ModelNotLoadedError: If no model is loaded on the server.
            ValidationError: If input_type is invalid.
        """
        data = self._request(
            "POST",
            "/encode",
            json={"texts": texts, "input_type": input_type},
        )
        return EncodeResponse.from_dict(data)

    # ==================== Utility Methods ====================

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self) -> "NextPlaidClient":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
