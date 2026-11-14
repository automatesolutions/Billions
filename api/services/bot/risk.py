"""
Risk limits every order passes through, whatever the strategy.

- Max dollars per position: entries are sized to stay under it (whole shares).
- Max orders per day: counted in New York days, seeded from the database at start
  so restarting the bot doesn't reset the count.
- Kill switch: once on, nothing more is sent. The runner also cancels open orders.

Exits are allowed past the daily order limit, so a position is never stuck open
because the bot ran out of orders. The kill switch blocks exits too: after a
kill, you decide what to close.
"""

import math
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, Optional

from pydantic import BaseModel, Field

from api.services.market_calendar import NEW_YORK


class RiskLimits(BaseModel):
    max_position_dollars: float = Field(1000.0, gt=0, le=1_000_000)
    max_orders_per_day: int = Field(20, ge=1, le=1000)


@dataclass
class RiskCheck:
    allowed: bool
    reason: str
    qty: int = 0


class RiskManager:
    def __init__(self, limits: RiskLimits, orders_today: int = 0, clock: Optional[Callable[[], datetime]] = None):
        self.limits = limits
        self._clock = clock or (lambda: datetime.now(NEW_YORK))
        self._day = self._clock().astimezone(NEW_YORK).date()
        self.orders_today = orders_today
        self.killed = False
        self.kill_reason: Optional[str] = None
        self._lock = threading.Lock()

    def _roll_day(self) -> None:
        today = self._clock().astimezone(NEW_YORK).date()
        if today != self._day:
            self._day, self.orders_today = today, 0

    def size_entry(self, price: float, existing_value: float = 0.0) -> RiskCheck:
        with self._lock:
            self._roll_day()
            if self.killed:
                return RiskCheck(False, f"Kill switch is on ({self.kill_reason}).")
            if self.orders_today >= self.limits.max_orders_per_day:
                return RiskCheck(
                    False, f"Daily order limit reached ({self.orders_today} of {self.limits.max_orders_per_day})."
                )
            if not price or price <= 0:
                return RiskCheck(False, "No valid price to size the order.")
            limit = self.limits.max_position_dollars
            room = limit - abs(existing_value)
            qty = math.floor(room / price) if room > 0 else 0
            if qty < 1:
                return RiskCheck(
                    False,
                    f"The ${limit:,.0f} position limit leaves room for less than one share at ${price:,.2f}"
                    + (f" (${abs(existing_value):,.0f} already held)." if existing_value else "."),
                )
            return RiskCheck(True, f"Sized to {qty} shares (${qty * price:,.2f}) under the ${limit:,.0f} limit.", qty)

    def allow_exit(self) -> RiskCheck:
        with self._lock:
            self._roll_day()
            if self.killed:
                return RiskCheck(False, f"Kill switch is on ({self.kill_reason}). Close positions yourself.")
            return RiskCheck(True, "Exits are always allowed while the bot runs.")

    def record_order(self) -> None:
        with self._lock:
            self._roll_day()
            self.orders_today += 1

    def kill(self, reason: str) -> None:
        with self._lock:
            self.killed = True
            self.kill_reason = reason

    def status(self) -> Dict:
        with self._lock:
            self._roll_day()
            return {
                "max_position_dollars": self.limits.max_position_dollars,
                "max_orders_per_day": self.limits.max_orders_per_day,
                "orders_today": self.orders_today,
                "killed": self.killed,
                "kill_reason": self.kill_reason,
            }
