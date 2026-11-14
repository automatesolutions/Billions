"""
Polygon bars, the shared rate limiter, and the cache. Every upstream call is faked; no network.
"""

from datetime import date, datetime, timezone

import pytest
import requests

from api.config import settings
from api.services import alpaca_data, polygon
from api.services.throttle import RateLimited, RateLimiter, TTLCache

# 2026-10-15 is a Thursday. 13:30 UTC is 9:30 New York, 20:00 UTC is 16:00.
OPEN_MS = int(datetime(2026, 10, 15, 13, 30, tzinfo=timezone.utc).timestamp() * 1000)
MINUTE_MS = 60_000


def bar(t_ms, close):
    return {"t": t_ms, "o": close, "h": close, "l": close, "c": close, "v": 100}


@pytest.fixture(autouse=True)
def fresh_cache(monkeypatch):
    polygon.clear_cache()
    alpaca_data.clear_cache()
    monkeypatch.setattr(settings, "POLYGON_API_KEY", "test-key")
    yield
    polygon.clear_cache()


class FakeClock:
    def __init__(self):
        self.now = 1000.0
        self.slept = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.now += seconds


# ---------------------------------------------------------------- rate limiter and cache


def test_rate_limiter_allows_limit_then_waits_for_the_oldest_call():
    clock = FakeClock()
    limiter = RateLimiter(5, 60, clock=clock, sleep=clock.sleep)
    for _ in range(5):
        assert limiter.acquire() == 0
    clock.now += 10
    assert limiter.wait_time() == pytest.approx(50)
    assert limiter.acquire() == pytest.approx(50)  # the sixth call waits until the first leaves the window
    assert clock.slept == [pytest.approx(50)]


def test_rate_limiter_gives_up_past_max_wait():
    clock = FakeClock()
    limiter = RateLimiter(1, 60, clock=clock, sleep=clock.sleep)
    limiter.acquire()
    with pytest.raises(RateLimited) as exc:
        limiter.acquire(max_wait=5)
    assert exc.value.retry_after == pytest.approx(60)
    assert clock.slept == []


def test_rate_limiter_estimates_scan_time():
    clock = FakeClock()
    limiter = RateLimiter(5, 60, clock=clock, sleep=clock.sleep)
    assert limiter.seconds_for(5) == 0  # all fit in the free slots
    assert limiter.seconds_for(6) == pytest.approx(60)  # the sixth waits a full window after the first five
    assert limiter.seconds_for(25) == pytest.approx(4 * 60)


def test_rate_limiter_estimate_after_use():
    clock = FakeClock()
    limiter = RateLimiter(5, 60, clock=clock, sleep=clock.sleep)
    for _ in range(5):
        limiter.acquire()
    clock.now += 20
    assert limiter.seconds_for(5) == pytest.approx(40)  # next batch when the first call leaves the window
    assert limiter.seconds_for(11) == pytest.approx(40 + 2 * 60)


def test_ttl_cache_expires_and_evicts():
    clock = FakeClock()
    cache = TTLCache(max_size=2, clock=clock)
    cache.set("a", 1, ttl=10)
    cache.set("b", 2, ttl=100)
    assert cache.get("a") == 1
    clock.now += 11
    assert cache.get("a") is None
    cache.set("c", 3, ttl=100)
    cache.set("d", 4, ttl=100)  # evicts the least recently used ("b")
    assert cache.get("b") is None and cache.get("d") == 4


# ---------------------------------------------------------------- bars


def test_get_bars_builds_frame_and_follows_pages():
    calls = []
    pages = {
        0: {"status": "OK", "results": [bar(OPEN_MS, 10.0)], "next_url": f"{settings.POLYGON_URL}/v2/aggs/next?cursor=abc"},
        1: {"status": "OK", "results": [bar(OPEN_MS + MINUTE_MS, 10.5)]},
    }

    def fetch(url, params, max_wait):
        calls.append((url, dict(params)))
        return pages[len(calls) - 1]

    df = polygon.get_bars("aapl", 1, "minute", date(2026, 10, 15), date(2026, 10, 15), fetch=fetch)
    assert list(df["close"]) == [10.0, 10.5]
    assert str(df.index.tz) == "UTC"
    assert calls[0][0].endswith("/v2/aggs/ticker/AAPL/range/1/minute/2026-10-15/2026-10-15")
    assert calls[0][1]["adjusted"] == "true" and calls[1][1] == {}  # next_url carries its own query


def test_get_bars_is_cached():
    calls = []

    def fetch(url, params, max_wait):
        calls.append(url)
        return {"status": "OK", "results": [bar(OPEN_MS, 10.0)]}

    for _ in range(3):
        polygon.get_bars("AAPL", 1, "minute", date(2026, 10, 15), date(2026, 10, 15), fetch=fetch)
    assert len(calls) == 1
    assert polygon.is_cached("AAPL", 1, "minute", date(2026, 10, 15), date(2026, 10, 15))


def test_next_url_on_another_host_is_not_followed():
    calls = []

    def fetch(url, params, max_wait):
        calls.append(url)
        return {"status": "OK", "results": [bar(OPEN_MS, 10.0)], "next_url": "https://evil.example.com/steal"}

    polygon.get_bars("AAPL", 1, "minute", date(2026, 10, 15), date(2026, 10, 15), fetch=fetch)
    assert len(calls) == 1


def test_empty_results_give_an_empty_frame():
    df = polygon.get_bars("AAPL", 1, "day", date(2026, 10, 1), date(2026, 10, 2), fetch=lambda *a: {"status": "OK"})
    assert df.empty and list(df.columns) == polygon.COLUMNS


def test_regular_session_drops_pre_and_post_market():
    rows = [bar(OPEN_MS - MINUTE_MS, 1), bar(OPEN_MS, 2), bar(OPEN_MS + 389 * MINUTE_MS, 3), bar(OPEN_MS + 390 * MINUTE_MS, 4)]
    df = polygon.get_intraday_bars(
        "AAPL", date(2026, 10, 15), date(2026, 10, 15), fetch=lambda *a: {"status": "OK", "results": rows}
    )
    assert list(df["close"]) == [2, 3]  # 9:29 and 16:00 are out


@pytest.mark.parametrize("ticker", ["", "$$$", "TOO-LONG-TICKER"])
def test_bad_ticker_is_a_polygon_error(ticker):
    with pytest.raises(polygon.PolygonError):
        polygon.get_bars(ticker, 1, "day", date(2026, 10, 1), date(2026, 10, 2), fetch=lambda *a: {})


def test_not_configured(monkeypatch):
    monkeypatch.setattr(settings, "POLYGON_API_KEY", "")
    with pytest.raises(polygon.PolygonError, match="POLYGON_API_KEY"):
        polygon.get_bars("AAPL", 1, "day", date(2026, 10, 1), date(2026, 10, 2))


def test_error_status_in_body():
    with pytest.raises(polygon.PolygonError):
        polygon.get_bars("AAPL", 1, "day", date(2026, 10, 1), date(2026, 10, 2), fetch=lambda *a: {"status": "ERROR"})


class FakeResponse:
    def __init__(self, status, body=None, bad_json=False):
        self.status_code = status
        self._body = body or {}
        self._bad = bad_json

    def json(self):
        if self._bad:
            raise ValueError("not json")
        return self._body


@pytest.mark.parametrize(
    "response, text",
    [
        (FakeResponse(429), "rate limit"),
        (FakeResponse(401), "POLYGON_API_KEY"),
        (FakeResponse(500), "500"),
        (FakeResponse(200, bad_json=True), "valid JSON"),
    ],
)
def test_fetch_json_maps_http_errors(monkeypatch, response, text):
    monkeypatch.setattr(polygon.limiter, "acquire", lambda max_wait=None: 0)
    monkeypatch.setattr(polygon.requests, "get", lambda *a, **k: response)
    with pytest.raises(polygon.PolygonError, match=text):
        polygon.fetch_json("https://api.polygon.io/x", {})


def test_fetch_json_network_error_hides_details(monkeypatch):
    def boom(*a, **k):
        raise requests.ConnectionError("secret host detail")

    monkeypatch.setattr(polygon.limiter, "acquire", lambda max_wait=None: 0)
    monkeypatch.setattr(polygon.requests, "get", boom)
    with pytest.raises(polygon.PolygonError) as exc:
        polygon.fetch_json("https://api.polygon.io/x", {})
    assert "secret" not in str(exc.value)


def test_fetch_json_sends_key_in_header_not_url(monkeypatch):
    seen = {}

    def get(url, params=None, headers=None, timeout=None):
        seen.update(url=url, params=params, headers=headers)
        return FakeResponse(200, {"status": "OK"})

    monkeypatch.setattr(polygon.limiter, "acquire", lambda max_wait=None: 0)
    monkeypatch.setattr(polygon.requests, "get", get)
    polygon.fetch_json("https://api.polygon.io/x", {"a": 1})
    assert seen["headers"]["Authorization"] == "Bearer test-key"
    assert "test-key" not in seen["url"] and "apiKey" not in seen["params"]


def test_fetch_json_rate_limited_becomes_polygon_error(monkeypatch):
    def full(max_wait=None):
        raise RateLimited(42)

    monkeypatch.setattr(polygon.limiter, "acquire", full)
    with pytest.raises(polygon.PolygonError) as exc:
        polygon.fetch_json("https://api.polygon.io/x", {}, max_wait=1)
    assert exc.value.retry_after == 42


def test_estimate_requests():
    # 5 trading days of minute bars fit in one page; 60 trading days need two.
    assert polygon.estimate_requests(date(2026, 10, 12), date(2026, 10, 16)) == 1
    assert polygon.estimate_requests(date(2026, 7, 1), date(2026, 10, 1)) == 2
    assert polygon.estimate_requests(date(2026, 7, 1), date(2026, 10, 1), "day") == 1


# ---------------------------------------------------------------- Alpaca historical bars


def test_alpaca_bars_follow_page_tokens():
    calls = []
    pages = [
        {"bars": [{"t": "2026-10-15T13:30:00Z", "o": 1, "h": 1, "l": 1, "c": 1, "v": 5}], "next_page_token": "p2"},
        {"bars": [{"t": "2026-10-15T13:31:00Z", "o": 2, "h": 2, "l": 2, "c": 2, "v": 5}], "next_page_token": None},
    ]

    def fetch(url, params, max_wait):
        calls.append(dict(params))
        return pages[len(calls) - 1]

    df = alpaca_data.get_intraday_bars("AMD", date(2026, 10, 15), date(2026, 10, 15), fetch=fetch)
    assert list(df["close"]) == [1, 2]
    assert "page_token" not in calls[0] and calls[1]["page_token"] == "p2"
    assert calls[0]["timeframe"] == "1Min" and calls[0]["feed"] == settings.ALPACA_FEED
