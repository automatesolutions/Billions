"""
Alpaca paper broker: the paper-URL guard, request shapes, and error handling. HTTP is faked; no network.
"""

import json

import pytest
import requests

from api.config import settings
from api.services import paper_broker as pb

ORDER_ID = "61e69015-8549-4bfd-b9c3-01e75843f47d"


class FakeResponse:
    def __init__(self, status=200, body=None):
        self.status_code = status
        self._body = body
        self.content = b"" if body is None else json.dumps(body).encode()

    def json(self):
        if self._body is None:
            raise ValueError("empty")
        return self._body


class FakeSession:
    def __init__(self, response=None):
        self.calls = []
        self.response = response or FakeResponse(200, {})

    def request(self, method, url, params=None, json=None, headers=None, timeout=None):
        self.calls.append({"method": method, "url": url, "params": params, "json": json, "headers": headers})
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def broker(response=None):
    session = FakeSession(response)
    return pb.PaperBroker(base_url=pb.PAPER_URL, key="k", secret="s", session=session), session


# ---------------------------------------------------------------- the guard


@pytest.mark.parametrize(
    "url", ["https://paper-api.alpaca.markets", "https://paper-api.alpaca.markets/", " https://PAPER-API.alpaca.markets "]
)
def test_paper_url_is_accepted(url):
    assert pb.assert_paper_url(url) == pb.PAPER_URL


@pytest.mark.parametrize(
    "url",
    [
        "https://api.alpaca.markets",  # live trading
        "http://paper-api.alpaca.markets",  # not TLS
        "https://paper-api.alpaca.markets.evil.com",  # look-alike host
        "https://paper-api.alpaca.markets@api.alpaca.markets",  # userinfo trick: the real host is live
        "https://evil.com/?h=paper-api.alpaca.markets",
        "https://paper-api.alpaca.markets:8443",
        "https://paper-api.alpaca.markets/v2",
        "https://paper-api.alpaca.markets:notaport",
        "",
    ],
)
def test_anything_else_is_refused(url):
    with pytest.raises(pb.LiveTradingRefused):
        pb.assert_paper_url(url)


def test_broker_refuses_live_url_from_settings(monkeypatch):
    monkeypatch.setattr(settings, "ALPACA_TRADING_URL", "https://api.alpaca.markets")
    with pytest.raises(pb.LiveTradingRefused):
        pb.PaperBroker(key="k", secret="s", session=FakeSession())


def test_broker_needs_keys():
    with pytest.raises(pb.BrokerNotConfigured):
        pb.PaperBroker(base_url=pb.PAPER_URL, key="", secret="", session=FakeSession())


def test_guard_runs_on_every_request():
    b, session = broker()
    b.base_url = "https://api.alpaca.markets"  # tampered after construction
    with pytest.raises(pb.LiveTradingRefused):
        b.get_account()
    assert session.calls == []


# ---------------------------------------------------------------- requests


def test_reads_go_to_the_paper_host_with_keys():
    b, session = broker(FakeResponse(200, {"cash": "1000"}))
    assert b.get_account() == {"cash": "1000"}
    call = session.calls[0]
    assert call["method"] == "GET" and call["url"] == "https://paper-api.alpaca.markets/v2/account"
    assert call["headers"] == {"APCA-API-KEY-ID": "k", "APCA-API-SECRET-KEY": "s"}


def test_get_orders_params():
    b, session = broker(FakeResponse(200, []))
    assert b.get_orders(limit=1000) == []
    assert session.calls[0]["params"] == {"status": "all", "limit": 500, "direction": "desc"}


def test_market_order_payload():
    b, session = broker(FakeResponse(200, {"id": ORDER_ID}))
    b.place_market_order("amd", "buy", 3, client_order_id="bot-1")
    call = session.calls[0]
    assert call["method"] == "POST" and call["url"].endswith("/v2/orders")
    assert call["json"] == {
        "symbol": "AMD",
        "qty": "3",
        "side": "buy",
        "type": "market",
        "time_in_force": "day",
        "client_order_id": "bot-1",
    }


def test_limit_order_payload():
    b, session = broker(FakeResponse(200, {"id": ORDER_ID}))
    b.place_limit_order("AAPL", "sell", 2, 187.456)
    assert session.calls[0]["json"]["type"] == "limit"
    assert session.calls[0]["json"]["limit_price"] == "187.46"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"symbol": "AAPL", "side": "short", "qty": 1},
        {"symbol": "AAPL", "side": "buy", "qty": 0},
        {"symbol": "AAPL", "side": "buy", "qty": -2},
        {"symbol": "$$$", "side": "buy", "qty": 1},
    ],
)
def test_bad_orders_never_reach_alpaca(kwargs):
    b, session = broker()
    with pytest.raises(ValueError):
        b.place_market_order(**kwargs)
    assert session.calls == []


def test_limit_price_must_be_positive():
    b, session = broker()
    with pytest.raises(ValueError):
        b.place_limit_order("AAPL", "buy", 1, 0)


def test_cancel_and_close():
    b, session = broker(FakeResponse(204))
    assert b.cancel_order(ORDER_ID) is None
    session.response = FakeResponse(207, [{"id": "a"}, {"id": "b"}])
    assert b.cancel_all_orders() == 2
    assert b.close_all_positions() == 2
    session.response = FakeResponse(200, {"id": ORDER_ID})
    b.close_position("nvda")
    urls = [(c["method"], c["url"].replace(pb.PAPER_URL, "")) for c in session.calls]
    assert urls == [
        ("DELETE", f"/v2/orders/{ORDER_ID}"),
        ("DELETE", "/v2/orders"),
        ("DELETE", "/v2/positions"),
        ("DELETE", "/v2/positions/NVDA"),
    ]
    assert session.calls[2]["params"] == {"cancel_orders": "true"}


def test_order_id_is_validated():
    b, session = broker()
    with pytest.raises(ValueError):
        b.cancel_order("../account")
    assert session.calls == []


def test_http_error_carries_alpaca_message():
    b, _ = broker(FakeResponse(403, {"code": 40310000, "message": "insufficient buying power"}))
    with pytest.raises(pb.BrokerError) as exc:
        b.place_market_order("AAPL", "buy", 1)
    assert exc.value.status_code == 403
    assert "insufficient buying power" in str(exc.value)


def test_network_error_is_a_broker_error():
    b, _ = broker(requests.ConnectionError("secret detail"))
    with pytest.raises(pb.BrokerError) as exc:
        b.get_positions()
    assert "secret" not in str(exc.value)


def test_summaries_convert_numbers():
    assert pb.account_summary({"cash": "100.5", "equity": "200", "status": "ACTIVE"})["cash"] == 100.5
    assert pb.position_summary({"symbol": "AMD", "qty": "3", "market_value": "450.3"})["qty"] == 3.0
    assert pb.order_summary({"id": ORDER_ID, "order_type": "market", "filled_avg_price": None})["type"] == "market"
