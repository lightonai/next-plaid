"""
Custom exceptions for the Next Plaid Client SDK.
"""

from typing import Optional, Any


class NextPlaidError(Exception):
    """Base exception for all Next Plaid Client SDK errors."""

    def __init__(
        self,
        message: str,
        code: Optional[str] = None,
        details: Optional[Any] = None,
        status_code: Optional[int] = None,
    ):
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = details
        self.status_code = status_code

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(message={self.message!r}, code={self.code!r})"


class IndexNotFoundError(NextPlaidError):
    """Raised when the requested index does not exist."""

    pass


class IndexExistsError(NextPlaidError):
    """Raised when trying to create an index that already exists."""

    pass


class ValidationError(NextPlaidError):
    """Raised when request validation fails."""

    pass


class RateLimitError(NextPlaidError):
    """Raised when rate limit is exceeded."""

    pass


class ModelNotLoadedError(NextPlaidError):
    """Raised when encoding is requested but no model is loaded."""

    pass


class ConnectionError(NextPlaidError):
    """Raised when connection to the server fails."""

    pass


class ServerError(NextPlaidError):
    """Raised when the server returns an internal error."""

    pass
