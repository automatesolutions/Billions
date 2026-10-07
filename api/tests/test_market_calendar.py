from datetime import date, datetime

import pytest

from api.services import market_calendar as mc


@pytest.mark.parametrize(
    "day",
    [
        date(2026, 1, 1),  # New Year's Day
        date(2026, 1, 19),  # MLK Day
        date(2026, 2, 16),  # Presidents' Day
        date(2026, 4, 3),  # Good Friday
        date(2026, 5, 25),  # Memorial Day
        date(2026, 6, 19),  # Juneteenth
        date(2026, 7, 3),  # Independence Day observed (Jul 4 is a Saturday)
        date(2026, 9, 7),  # Labor Day
        date(2026, 11, 26),  # Thanksgiving
        date(2026, 12, 25),  # Christmas
    ],
)
def test_2026_holidays(day):
    assert not mc.is_trading_day(day)


def test_saturday_new_year_is_not_observed_on_friday():
    # Jan 1 2022 was a Saturday; NYSE stayed open on Fri Dec 31 2021.
    assert mc.is_trading_day(date(2021, 12, 31))


def test_early_closes_2026():
    assert date(2026, 11, 27) in mc.early_closes(2026)  # day after Thanksgiving
    assert date(2026, 12, 24) in mc.early_closes(2026)
    assert date(2026, 7, 3) not in mc.early_closes(2026)  # a full holiday that year
    assert mc.session_close(date(2026, 11, 27)).hour == 13


def at(y, m, d, hh, mm):
    return datetime(y, m, d, hh, mm, tzinfo=mc.NEW_YORK)


def test_market_state_open_and_closed():
    assert mc.market_state(at(2026, 10, 7, 10, 0))["state"] == "open"
    assert mc.market_state(at(2026, 10, 7, 8, 0))["state"] == "pre_market"
    assert mc.market_state(at(2026, 10, 7, 16, 30))["state"] == "after_hours"
    assert mc.market_state(at(2026, 10, 10, 12, 0))["state"] == "weekend"
    assert mc.market_state(at(2026, 11, 26, 12, 0))["state"] == "holiday"


def test_next_open_skips_weekend():
    state = mc.market_state(at(2026, 10, 9, 17, 0))  # Friday evening
    assert state["next_open"].startswith("2026-10-12T09:30")


def test_last_session_close():
    assert mc.last_session_close(at(2026, 10, 12, 9, 0)).date() == date(2026, 10, 9)  # Monday pre-market
    assert mc.last_session_close(at(2026, 10, 7, 16, 5)).date() == date(2026, 10, 7)
