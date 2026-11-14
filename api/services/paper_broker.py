"""
Alpaca paper trading: account, positions, orders, and order entry.

Paper only. assert_paper_url() refuses any trading URL except Alpaca's paper host,
and it runs both when the broker is built and before every request, so a
misconfigured .env can't send an order to a live account.
"""

import logging
import re
from typing import Dict, List, Optional
from urllib.parse import urlsplit

import requests

from api.config import settings
from api.services.prices import normalize_ticker

logger = logging.getLogger(__name__)

PAPER_HOST = "paper-api.alpaca.markets"
PAPER_URL = f"https://{PAPER_HOST}"
REQUEST_TIMEOUT_SECONDS = 10
ORDER_ID = re.compile(r"^[0-9a-fA-F-]{36}$")
SIDES = ("buy", "sell")
TIME_IN_FORCE = ("day", "gtc", "ioc", "fok")

REFUSED = (
    f"ALPACA_TRADING_URL must be {PAPER_URL}. The bot only trades on Alpaca's paper account, "
    "so it won't start with any other URL."
)
NOT_CONFIGURED = "The bot needs Alpaca paper keys. Set ALPACA_API_KEY and ALPACA_SECRET_KEY in .env."


class LiveTradingRefused(RuntimeError):
    """The trading URL isn't Alpaca's paper endpoint."""


class BrokerNotConfigured(RuntimeError):
    pass


class BrokerError(RuntimeError):
    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


def assert_paper_url(url: str) -> str:
    """Return the canonical paper URL, or raise LiveTradingRefused for anything else."""
    try:
        parts = urlsplit((url or "").strip())
        port = parts.port
    except ValueError:
        raise LiveTradingRefused(REFUSED) from None
    if (
        parts.scheme != "https"
        or parts.hostname != PAPER_HOST
        or parts.username is not None
        or parts.password is not None
        or port not in (None, 443)
        or parts.path not in ("", "/")
        or parts.query
        or parts.fragment
    ):
        raise LiveTradingRefused(REFUSED)
    return PAPER_URL


def configured() -> bool:
    return bool(settings.ALPACA_API_KEY and settings.ALPACA_SECRET_KEY)


def _symbol(value: str) -> str:
    symbol = normalize_ticker(value)
    if symbol is None:
        raise ValueError(f"'{value}' isn't a valid ticker.")
    return symbol


def _order_id(value: str) -> str:
    if not ORDER_ID.match(value or ""):
        raise ValueError("That isn't a valid order id.")
    return value


def _qty(value) -> str:
    qty = float(value)
    if not qty > 0:
        raise ValueError("Quantity must be more than zero.")
    return f"{qty:g}"


class PaperBroker:
    def __init__(
        self,
        base_url: Optional[str] = None,
        key: Optional[str] = None,
        secret: Optional[str] = None,
        session=None,
        timeout: float = REQUEST_TIMEOUT_SECONDS,
    ):
        # The guard comes first: with a live URL nothing else is even read.
        self.base_url = assert_paper_url(settings.ALPACA_TRADING_URL if base_url is None else base_url)
        key = settings.ALPACA_API_KEY if key is None else key
        secret = settings.ALPACA_SECRET_KEY if secret is None else secret
        if not (key and secret):
            raise BrokerNotConfigured(NOT_CONFIGURED)
        self._headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
        self._session = session or requests.Session()
        self._timeout = timeout

    # ------------------------------------------------------------ transport

    def _request(self, method: str, path: str, params: Optional[Dict] = None, json: Optional[Dict] = None):
        if not path.startswith("/v2/"):
            raise ValueError("Broker paths must start with /v2/.")
        url = assert_paper_url(self.base_url) + path  # checked again on every request
        try:
            response = self._session.request(
                method, url, params=params, json=json, headers=self._headers, timeout=self._timeout
            )
        except requests.RequestException as exc:
            logger.warning("Alpaca paper request failed: %s", type(exc).__name__)
            raise BrokerError("Alpaca's paper API didn't respond. Try again in a minute.") from None
        if response.status_code >= 400:
            raise BrokerError(
                f"Alpaca refused the request ({response.status_code}): {_message(response)}", response.status_code
            )
        if response.status_code == 204 or not response.content:
            return None
        try:
            return response.json()
        except ValueError:
            raise BrokerError("Alpaca sent a response that isn't valid JSON.") from None

    # ------------------------------------------------------------ reads

    def get_account(self) -> Dict:
        return self._request("GET", "/v2/account")

    def get_positions(self) -> List[Dict]:
        return self._request("GET", "/v2/positions") or []

    def get_orders(self, status: str = "all", limit: int = 50) -> List[Dict]:
        if status not in ("open", "closed", "all"):
            raise ValueError("status must be open, closed, or all.")
        params = {"status": status, "limit": max(1, min(int(limit), 500)), "direction": "desc"}
        return self._request("GET", "/v2/orders", params=params) or []

    def get_order(self, order_id: str) -> Dict:
        return self._request("GET", f"/v2/orders/{_order_id(order_id)}")

    # ------------------------------------------------------------ orders

    def place_market_order(
        self, symbol: str, side: str, qty, time_in_force: str = "day", client_order_id: Optional[str] = None
    ) -> Dict:
        return self._submit(symbol, side, qty, "market", time_in_force, None, client_order_id)

    def place_limit_order(
        self,
        symbol: str,
        side: str,
        qty,
        limit_price: float,
        time_in_force: str = "day",
        client_order_id: Optional[str] = None,
    ) -> Dict:
        if not float(limit_price) > 0:
            raise ValueError("Limit price must be more than zero.")
        return self._submit(symbol, side, qty, "limit", time_in_force, float(limit_price), client_order_id)

    def _submit(self, symbol, side, qty, order_type, time_in_force, limit_price, client_order_id) -> Dict:
        if side not in SIDES:
            raise ValueError("side must be buy or sell.")
        if time_in_force not in TIME_IN_FORCE:
            raise ValueError(f"time_in_force must be one of {', '.join(TIME_IN_FORCE)}.")
        payload = {
            "symbol": _symbol(symbol),
            "qty": _qty(qty),
            "side": side,
            "type": order_type,
            "time_in_force": time_in_force,
        }
        if limit_price is not None:
            payload["limit_price"] = f"{limit_price:.2f}" if limit_price >= 1 else f"{limit_price:.4f}"
        if client_order_id:
            payload["client_order_id"] = str(client_order_id)[:48]
        logger.info("Paper order: %s %s %s (%s)", side, payload["qty"], payload["symbol"], order_type)
        return self._request("POST", "/v2/orders", json=payload)

    def cancel_order(self, order_id: str) -> None:
        self._request("DELETE", f"/v2/orders/{_order_id(order_id)}")

    def cancel_all_orders(self) -> int:
        """Cancel every open order. Returns how many Alpaca reported."""
        result = self._request("DELETE", "/v2/orders")
        return len(result) if isinstance(result, list) else 0

    def close_position(self, symbol: str) -> Dict:
        return self._request("DELETE", f"/v2/positions/{_symbol(symbol)}")

    def close_all_positions(self, cancel_orders: bool = True) -> int:
        result = self._request("DELETE", "/v2/positions", params={"cancel_orders": str(cancel_orders).lower()})
        return len(result) if isinstance(result, list) else 0


def _message(response) -> str:
    try:
        body = response.json()
        text = body.get("message") if isinstance(body, dict) else None
    except ValueError:
        text = None
    return (text or "no details")[:200]


# ---------------------------------------------------------------- shapes for the API


def _num(value) -> Optional[float]:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def account_summary(raw: Dict) -> Dict:
    return {
        "cash": _num(raw.get("cash")),
        "equity": _num(raw.get("equity")),
        "last_equity": _num(raw.get("last_equity")),
        "buying_power": _num(raw.get("buying_power")),
        "status": raw.get("status"),
        "currency": raw.get("currency", "USD"),
    }


def position_summary(raw: Dict) -> Dict:
    return {
        "symbol": raw.get("symbol"),
        "qty": _num(raw.get("qty")),
        "side": raw.get("side"),
        "avg_entry_price": _num(raw.get("avg_entry_price")),
        "current_price": _num(raw.get("current_price")),
        "market_value": _num(raw.get("market_value")),
        "unrealized_pl": _num(raw.get("unrealized_pl")),
        "unrealized_plpc": _num(raw.get("unrealized_plpc")),
    }


def order_summary(raw: Dict) -> Dict:
    return {
        "id": raw.get("id"),
        "symbol": raw.get("symbol"),
        "side": raw.get("side"),
        "type": raw.get("type") or raw.get("order_type"),
        "qty": _num(raw.get("qty")),
        "filled_qty": _num(raw.get("filled_qty")),
        "filled_avg_price": _num(raw.get("filled_avg_price")),
        "limit_price": _num(raw.get("limit_price")),
        "status": raw.get("status"),
        "submitted_at": raw.get("submitted_at"),
        "filled_at": raw.get("filled_at"),
    }
