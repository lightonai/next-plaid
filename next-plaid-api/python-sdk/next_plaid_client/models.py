"""
Data models for the NextPlaid SDK.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class IndexConfig:
    """Configuration for creating a new index."""

    nbits: int = 4
    batch_size: int = 50000
    seed: Optional[int] = None
    start_from_scratch: int = 999
    max_documents: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "nbits": self.nbits,
            "batch_size": self.batch_size,
            "start_from_scratch": self.start_from_scratch,
        }
        if self.seed is not None:
            result["seed"] = self.seed
        if self.max_documents is not None:
            result["max_documents"] = self.max_documents
        return result


@dataclass
class IndexInfo:
    """Information about an index."""

    name: str
    num_documents: int
    num_embeddings: int
    num_partitions: int
    avg_doclen: float
    dimension: int
    has_metadata: bool
    metadata_count: Optional[int] = None
    max_documents: Optional[int] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IndexInfo":
        return cls(
            name=data["name"],
            num_documents=data["num_documents"],
            num_embeddings=data["num_embeddings"],
            num_partitions=data["num_partitions"],
            avg_doclen=data["avg_doclen"],
            dimension=data["dimension"],
            has_metadata=data["has_metadata"],
            metadata_count=data.get("metadata_count"),
            max_documents=data.get("max_documents"),
        )


@dataclass
class IndexSummary:
    """Summary information about an index (from health endpoint)."""

    name: str
    num_documents: int
    num_embeddings: int
    num_partitions: int
    dimension: int
    nbits: int
    avg_doclen: float
    has_metadata: bool
    max_documents: Optional[int] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IndexSummary":
        return cls(
            name=data["name"],
            num_documents=data["num_documents"],
            num_embeddings=data["num_embeddings"],
            num_partitions=data["num_partitions"],
            dimension=data["dimension"],
            nbits=data["nbits"],
            avg_doclen=data["avg_doclen"],
            has_metadata=data["has_metadata"],
            max_documents=data.get("max_documents"),
        )


@dataclass
class HealthResponse:
    """Response from the health endpoint."""

    status: str
    version: str
    loaded_indices: int
    index_dir: str
    memory_usage_bytes: int
    indices: List[IndexSummary]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HealthResponse":
        return cls(
            status=data["status"],
            version=data["version"],
            loaded_indices=data["loaded_indices"],
            index_dir=data["index_dir"],
            memory_usage_bytes=data["memory_usage_bytes"],
            indices=[IndexSummary.from_dict(idx) for idx in data.get("indices", [])],
        )


@dataclass
class SearchParams:
    """Parameters for search operations.

    Attributes:
        top_k: Number of results to return per query (default: 10)
        n_ivf_probe: Number of IVF cells to probe (default: 8)
        n_full_scores: Number of documents for exact re-ranking (default: 4096)
        centroid_score_threshold: Centroid score threshold for centroid pruning (default: None = disabled).
            Centroids with max score below this threshold are filtered out.
            Set to a float value (e.g., 0.4) to enable pruning for faster but potentially less accurate search.
    """

    top_k: int = 10
    n_ivf_probe: int = 8
    n_full_scores: int = 4096
    centroid_score_threshold: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "top_k": self.top_k,
            "n_ivf_probe": self.n_ivf_probe,
            "n_full_scores": self.n_full_scores,
        }
        # Include threshold - None means disable pruning, float value enables it
        if self.centroid_score_threshold is not None:
            result["centroid_score_threshold"] = self.centroid_score_threshold
        else:
            result["centroid_score_threshold"] = None
        return result


@dataclass
class QueryResult:
    """Result for a single query in a search response."""

    query_id: int
    document_ids: List[int]
    scores: List[float]
    metadata: Optional[List[Optional[Dict[str, Any]]]] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QueryResult":
        return cls(
            query_id=data["query_id"],
            document_ids=data["document_ids"],
            scores=data["scores"],
            metadata=data.get("metadata"),
        )


@dataclass
class SearchResult:
    """Response from a search operation."""

    results: List[QueryResult]
    num_queries: int

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SearchResult":
        return cls(
            results=[QueryResult.from_dict(r) for r in data["results"]],
            num_queries=data["num_queries"],
        )


@dataclass
class Document:
    """A document with embeddings for indexing."""

    embeddings: List[List[float]]

    def to_dict(self) -> Dict[str, Any]:
        return {"embeddings": self.embeddings}


@dataclass
class MetadataResponse:
    """Response from metadata operations."""

    metadata: List[Dict[str, Any]]
    count: int

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MetadataResponse":
        return cls(
            metadata=data["metadata"],
            count=data["count"],
        )


@dataclass
class MetadataCheckResponse:
    """Response from metadata check operation."""

    existing_ids: List[int]
    missing_ids: List[int]
    existing_count: int
    missing_count: int

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MetadataCheckResponse":
        return cls(
            existing_ids=data["existing_ids"],
            missing_ids=data["missing_ids"],
            existing_count=data["existing_count"],
            missing_count=data["missing_count"],
        )


@dataclass
class EncodeResponse:
    """Response from the encode endpoint."""

    embeddings: List[List[List[float]]]
    num_texts: int

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EncodeResponse":
        return cls(
            embeddings=data["embeddings"],
            num_texts=data["num_texts"],
        )


@dataclass
class DeleteDocumentsResponse:
    """Response from document deletion."""

    deleted: int
    remaining: int

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DeleteDocumentsResponse":
        return cls(
            deleted=data["deleted"],
            remaining=data["remaining"],
        )


@dataclass
class RerankResult:
    """A single result from the rerank operation."""

    index: int
    score: float

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RerankResult":
        return cls(
            index=data["index"],
            score=data["score"],
        )


@dataclass
class RerankResponse:
    """Response from a rerank operation."""

    results: List[RerankResult]
    num_documents: int

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RerankResponse":
        return cls(
            results=[RerankResult.from_dict(r) for r in data["results"]],
            num_documents=data["num_documents"],
        )
