"""Shared fixtures for the integrations test suite."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import pytest

from llmdiff import Corpus, evaluate

CORPUS_RECORDS: list[dict[str, Any]] = [
    {
        "id": "c1",
        "prompt": "hi",
        "baseline_output": json.dumps({"a": 1, "b": 2}),
        "candidate_output": json.dumps({"a": 1}),
        "metadata": {"model": "m1"},
    },
    {
        "id": "c2",
        "prompt": "yo",
        "baseline_output": "hello world",
        "candidate_output": "hello there world",
        "metadata": {},
    },
]


class MockResponse:
    """A minimal httpx-like response."""

    def __init__(
        self,
        status_code: int = 200,
        json_data: Any = None,
        text: str | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self._json_data = json_data
        if text is not None:
            self.text = text
        elif json_data is not None:
            self.text = json.dumps(json_data)
        else:
            self.text = ""
        self.headers = dict(headers or {})

    def json(self) -> Any:
        if self._json_data is not None:
            return self._json_data
        return json.loads(self.text)


class MockTransport:
    """Records requests and returns queued responses."""

    def __init__(self, responses: list[Any] | None = None) -> None:
        self.responses: list[Any] = list(responses or [MockResponse(200, {"ok": True})])
        self.requests: list[dict[str, Any]] = []

    def _next(self, method: str, url: str, kwargs: Mapping[str, Any]) -> MockResponse:
        self.requests.append({"method": method, "url": url, **dict(kwargs)})
        if not self.responses:
            return MockResponse(200, {"ok": True})
        item = self.responses[0] if len(self.responses) == 1 else self.responses.pop(0)
        if callable(item):
            return item(method, url, kwargs)
        return item

    def post(self, url: str, *, headers: Any = None, json: Any = None, data: Any = None, content: Any = None, params: Any = None, timeout: float = 10.0) -> MockResponse:
        return self._next("POST", url, {"headers": dict(headers or {}), "json": json, "data": data, "content": content, "params": params, "timeout": timeout})

    def get(self, url: str, *, headers: Any = None, params: Any = None, timeout: float = 10.0) -> MockResponse:
        return self._next("GET", url, {"headers": dict(headers or {}), "params": params, "timeout": timeout})

    def request(self, method: str, url: str, *, headers: Any = None, json: Any = None, data: Any = None, content: Any = None, params: Any = None, timeout: float = 10.0) -> MockResponse:
        return self._next(method.upper(), url, {"headers": dict(headers or {}), "json": json, "data": data, "content": content, "params": params, "timeout": timeout})

    @property
    def last(self) -> dict[str, Any]:
        return self.requests[-1]

    @property
    def last_json(self) -> Any:
        return self.requests[-1].get("json")

    @property
    def last_headers(self) -> dict[str, str]:
        return self.requests[-1].get("headers", {})


@pytest.fixture
def transport() -> MockTransport:
    return MockTransport()


@pytest.fixture
def make_result() -> Any:
    """Build a real DiffResult from the fixed corpus."""

    def _make(records: list[dict[str, Any]] | None = None) -> Any:
        corpus = Corpus.from_records(records if records is not None else CORPUS_RECORDS)
        return evaluate(corpus)

    return _make
