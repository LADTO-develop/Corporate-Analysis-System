"""HTTP protocols used by external evidence providers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol


class HttpResponse(Protocol):
    """Small response protocol used to keep collectors testable."""

    @property
    def content(self) -> bytes:
        """Return raw response bytes when an endpoint is not JSON."""

    def raise_for_status(self) -> None:
        """Raise when the HTTP response indicates failure."""

    def json(self) -> object:
        """Return the decoded JSON body."""


class HttpClient(Protocol):
    """Small HTTP client protocol compatible with requests.Session."""

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, object] | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> HttpResponse:
        """Send an HTTP GET request."""

    def post(
        self,
        url: str,
        *,
        json: Mapping[str, object] | None = None,
        timeout: float | None = None,
    ) -> HttpResponse:
        """Send an HTTP POST request."""
