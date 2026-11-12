"""The Wikipedia tool, with the network call replaced by a stand-in"""

from src.node import reactnode
from src.node.reactnode import wikipedia_search


class FakeResponse:
    def __init__(self, status_code=200, payload=None, headers=None):
        self.status_code = status_code
        self._payload = payload or {}
        self.headers = headers or {}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def fake_get(response, calls=None):
    def get(url, **kwargs):
        if calls is not None:
            calls.append(kwargs)
        return response

    return get


def test_returns_summaries_in_search_order_with_one_request(monkeypatch):
    pages = {
        "7": {"title": "Second", "index": 2, "extract": "Second summary."},
        "3": {"title": "First", "index": 1, "extract": "First summary."},
        "9": {"title": "Empty", "index": 3, "extract": ""},
    }
    calls = []
    monkeypatch.setattr(
        reactnode.requests,
        "get",
        fake_get(FakeResponse(payload={"query": {"pages": pages}}), calls),
    )

    result = wikipedia_search("anything")

    assert result == (
        "Page: First\nSummary: First summary.\n\nPage: Second\nSummary: Second summary."
    )
    assert len(calls) == 1
    assert calls[0]["headers"]["User-Agent"]


def test_no_results(monkeypatch):
    monkeypatch.setattr(reactnode.requests, "get", fake_get(FakeResponse(payload={})))
    assert wikipedia_search("zzqx") == "No Wikipedia page found."


def test_rate_limit_is_reported_not_raised(monkeypatch):
    response = FakeResponse(status_code=429, headers={"Retry-After": "30"})
    monkeypatch.setattr(reactnode.requests, "get", fake_get(response))
    assert "30 seconds" in wikipedia_search("anything")


def test_network_error_is_reported_not_raised(monkeypatch):
    def broken(url, **kwargs):
        raise ConnectionError("no network")

    monkeypatch.setattr(reactnode.requests, "get", broken)
    assert wikipedia_search("anything").startswith("Wikipedia search failed")
