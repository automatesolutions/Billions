"""
US equity market calendar (NYSE/NASDAQ regular session).

Covers full-day holidays and 1pm early closes. Pure functions, no network.
"""

from datetime import date, datetime, time, timedelta
from typing import Dict, Optional
from zoneinfo import ZoneInfo

from dateutil.easter import easter

NEW_YORK = ZoneInfo("America/New_York")
OPEN_TIME = time(9, 30)
CLOSE_TIME = time(16, 0)
EARLY_CLOSE_TIME = time(13, 0)


def _observed(day: date) -> date:
    """Saturday holidays move to Friday, Sunday holidays to Monday."""
    if day.weekday() == 5:
        return day - timedelta(days=1)
    if day.weekday() == 6:
        return day + timedelta(days=1)
    return day


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    first = date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + timedelta(days=offset + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    last = date(year, month + 1, 1) - timedelta(days=1) if month < 12 else date(year, 12, 31)
    return last - timedelta(days=(last.weekday() - weekday) % 7)


def holidays(year: int) -> set:
    days = {
        _nth_weekday(year, 1, 0, 3),  # Martin Luther King Jr. Day
        _nth_weekday(year, 2, 0, 3),  # Washington's Birthday
        easter(year) - timedelta(days=2),  # Good Friday
        _last_weekday(year, 5, 0),  # Memorial Day
        _observed(date(year, 7, 4)),  # Independence Day
        _nth_weekday(year, 9, 0, 1),  # Labor Day
        _nth_weekday(year, 11, 3, 4),  # Thanksgiving
        _observed(date(year, 12, 25)),  # Christmas
    }
    new_year = date(year, 1, 1)
    if new_year.weekday() != 5:  # NYSE does not observe a Saturday New Year on the prior Friday
        days.add(_observed(new_year))
    if year >= 2022:
        days.add(_observed(date(year, 6, 19)))  # Juneteenth
    return days


def early_closes(year: int) -> set:
    days = {_nth_weekday(year, 11, 3, 4) + timedelta(days=1)}  # day after Thanksgiving
    for candidate in (date(year, 7, 3), date(year, 12, 24)):
        if candidate.weekday() < 5 and candidate not in holidays(year):
            days.add(candidate)
    return days


def is_trading_day(day: date) -> bool:
    return day.weekday() < 5 and day not in holidays(day.year)


def session_close(day: date) -> datetime:
    close = EARLY_CLOSE_TIME if day in early_closes(day.year) else CLOSE_TIME
    return datetime.combine(day, close, tzinfo=NEW_YORK)


def session_open(day: date) -> datetime:
    return datetime.combine(day, OPEN_TIME, tzinfo=NEW_YORK)


def previous_trading_day(day: date) -> date:
    day -= timedelta(days=1)
    while not is_trading_day(day):
        day -= timedelta(days=1)
    return day


def next_trading_day(day: date) -> date:
    day += timedelta(days=1)
    while not is_trading_day(day):
        day += timedelta(days=1)
    return day


def last_session_close(now: datetime) -> datetime:
    """The most recent regular-session close at or before `now`."""
    local = now.astimezone(NEW_YORK)
    day = local.date()
    if is_trading_day(day) and local >= session_close(day):
        return session_close(day)
    return session_close(previous_trading_day(day))


def market_state(now: Optional[datetime] = None) -> Dict:
    """Return {'is_open', 'state', 'next_open', 'next_close', 'last_close'} for `now` (UTC or aware)."""
    now = now or datetime.now(tz=NEW_YORK)
    local = now.astimezone(NEW_YORK)
    today = local.date()

    if is_trading_day(today) and session_open(today) <= local < session_close(today):
        return {
            "is_open": True,
            "state": "open",
            "next_open": None,
            "next_close": session_close(today).isoformat(),
            "last_close": session_close(previous_trading_day(today)).isoformat(),
        }

    if not is_trading_day(today):
        state = "weekend" if today.weekday() >= 5 else "holiday"
    elif local < session_open(today):
        state = "pre_market"
    else:
        state = "after_hours"

    next_open_day = today if (is_trading_day(today) and local < session_open(today)) else next_trading_day(today)
    return {
        "is_open": False,
        "state": state,
        "next_open": session_open(next_open_day).isoformat(),
        "next_close": None,
        "last_close": last_session_close(local).isoformat(),
    }
