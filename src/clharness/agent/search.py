"""Web search for the Scout, backed by the Exa API.

The Scout model never searches by itself and never sees the key. Code issues the search, turns
each result into a `Source` (untrusted text), and the Scout reads those sources as quoted data.

The request follows Exa's recommended shape for a search that is one step of a larger task:
the query, an `objective` that says what the search is for, `type="auto"` and bare
`highlights`. No category, domain or date filters: source preference goes in the query and the
objective (Exa's guidance: hard filters drop good pages).

The key is read from `EXA_API_KEY` (put it in `.env`; `clharness.env` loads it). It is never
logged or stored.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Sequence
from typing import Any, Protocol

from .contracts import Source, utc_iso
from .roles import Clock, SystemClock

MAX_SNIPPET_CHARS = 1500  # per source, after joining highlights; keeps the Scout prompt small


class SearchError(Exception):
    """Search could not run (no key, no package, network or API error)."""


class SearchClient(Protocol):
    def search(self, query: str, objective: str) -> tuple[Source, ...]:
        """Sources for one query. Raises SearchError on any failure."""
        ...


def source_id(query: str, url: str) -> str:
    return "src-" + hashlib.sha256(f"{query}\n{url}".encode()).hexdigest()[:12]


def _field(item: Any, *names: str) -> Any:
    for name in names:
        value = item.get(name) if isinstance(item, dict) else getattr(item, name, None)
        if value is not None:
            return value
    return None


class ExaSearch:
    """`exa.search(query, objective=..., type="auto", contents={"highlights": True})`."""

    def __init__(self, *, api_key_env: str = "EXA_API_KEY", clock: Clock | None = None,
                 client: Any = None) -> None:
        self._api_key_env = api_key_env
        self._clock = clock or SystemClock()
        self._client = client

    def _sdk(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from exa_py import Exa
        except ImportError as exc:
            raise SearchError("the `exa-py` package is not installed: "
                              "uv sync --extra exa (or pip install exa-py)") from exc
        key = os.environ.get(self._api_key_env)
        if not key:
            raise SearchError(f"environment variable {self._api_key_env} is not set")
        self._client = Exa(api_key=key)
        return self._client

    def search(self, query: str, objective: str) -> tuple[Source, ...]:
        try:
            response = self._sdk().search(query, objective=objective, type="auto",
                                          contents={"highlights": True})
        except SearchError:
            raise
        except Exception as exc:  # any SDK or network failure is a SearchError
            raise SearchError(f"{type(exc).__name__}: {exc}") from exc
        now = utc_iso(self._clock.now())
        sources: list[Source] = []
        for item in _field(response, "results") or ():
            url = _field(item, "url")
            if not url:
                continue
            highlights = _field(item, "highlights") or ()
            snippet = " … ".join(str(h).strip() for h in highlights)[:MAX_SNIPPET_CHARS]
            sources.append(Source(
                id=source_id(query, url), query=query, url=url,
                title=str(_field(item, "title") or "")[:300], snippet=snippet,
                published=_field(item, "published_date", "publishedDate"), retrieved_at=now,
            ))
        return tuple(sources)


class FakeSearch:
    """Returns scripted sources per query; records every (query, objective) it was asked."""

    def __init__(self, results: dict[str, Sequence[Source]] | None = None,
                 fail: bool = False) -> None:
        self.results = dict(results or {})
        self.fail = fail
        self.calls: list[tuple[str, str]] = []

    def search(self, query: str, objective: str) -> tuple[Source, ...]:
        self.calls.append((query, objective))
        if self.fail:
            raise SearchError("fake search failure")
        return tuple(self.results.get(query, ()))
