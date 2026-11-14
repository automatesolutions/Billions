"""
Lead-lag math on synthetic data with a planted lead. The engine must find who leads, by how much,
and how often the follower follows, and must not be fooled by noise or by a pattern that fades.
"""

from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from api.services import lead_lag as ll
from api.services import market_calendar as mc

SIGMA = 0.002  # 0.2% per minute


def trading_days(n, start=date(2026, 6, 1)):
    days, day = [], start
    while len(days) < n:
        if mc.is_trading_day(day):
            days.append(day)
        day += timedelta(days=1)
    return days


def minute_index(days):
    stamps = []
    for day in days:
        open_ = datetime.combine(day, mc.OPEN_TIME, tzinfo=mc.NEW_YORK)
        stamps.extend(open_ + timedelta(minutes=m) for m in range(390))
    return pd.DatetimeIndex(stamps).tz_convert("UTC")


def planted(n_days=30, lag=2, noise=0.0005, follow_days=None, seed=7):
    """Leader is a random walk; follower copies its return `lag` minutes later (only on the first `follow_days` days)."""
    rng = np.random.default_rng(seed)
    days = trading_days(n_days)
    idx = minute_index(days)
    lr = rng.normal(0, SIGMA, len(idx))
    fr = rng.normal(0, noise, len(idx))
    per_day = 390
    for d in range(n_days):
        s = d * per_day
        if follow_days is None or d < follow_days:
            fr[s + lag : s + per_day] += lr[s : s + per_day - lag]
        else:
            fr[s : s + per_day] = rng.normal(0, SIGMA, per_day)  # relationship gone
    leader = pd.Series(100 * np.exp(np.cumsum(lr)), index=idx)
    follower = pd.Series(50 * np.exp(np.cumsum(fr)), index=idx)
    return leader, follower


PARAMS = ll.EngineParams(threshold_pct=0.3, lookback_seconds=60, max_wait_seconds=300)


def test_finds_planted_leader_lag_and_accuracy():
    leader, follower = planted()
    r = ll.analyze_pair("LEAD", leader, "FOLL", follower, PARAMS)
    assert r["status"] == "ok"
    assert (r["leader"], r["follower"]) == ("LEAD", "FOLL")
    assert r["bar_seconds"] == 60 and r["resolution"] == "minute"
    assert r["lag_seconds"] == 120  # the planted 2-minute lag
    # Not 100%: the follower's 5-minute window also copies four other random leader minutes (about 0.4% combined),
    # which sometimes outweigh the 0.3%+ event move. Theory puts accuracy near 80% to 85%.
    assert 0.75 < r["accuracy"] < 0.92 and r["events"] >= 300
    assert r["out_of_sample"]["accuracy"] > 0.75 and r["out_of_sample"]["direction_held"] is True
    assert r["p_value"] < 1e-6
    assert r["confidence"] == "strong"
    assert r["avg_follower_move_pct"] > 0.2
    assert r["avg_lag_seconds"] == pytest.approx(120, abs=40)
    peak = max(r["details"]["lag_histogram"], key=lambda h: h["count"])
    assert peak["lag_seconds"] == 120


def test_argument_order_does_not_matter():
    leader, follower = planted()
    r = ll.analyze_pair("FOLL", follower, "LEAD", leader, PARAMS)
    assert (r["leader"], r["follower"]) == ("LEAD", "FOLL") and r["lag_seconds"] == 120


def test_xcorr_peaks_at_the_planted_lag():
    leader, follower = planted(lag=3)
    r = ll.analyze_pair("LEAD", leader, "FOLL", follower, PARAMS)
    profile = {p["lag_seconds"]: p["corr"] for p in r["details"]["xcorr"]}
    assert max(profile, key=lambda k: profile[k] or -1) == 180
    assert profile[180] > 0.9 and abs(profile[-180]) < 0.1


def test_event_path_shows_follower_catching_up():
    leader, follower = planted()
    path = {p["offset_seconds"]: p for p in ll.analyze_pair("LEAD", leader, "FOLL", follower, PARAMS)["details"]["event_path"]}
    assert path[0]["leader_pct"] == 0 and path[0]["follower_pct"] == 0
    assert path[-60]["leader_pct"] < -0.3  # the leader rose into the event (direction-adjusted)
    assert path[300]["follower_pct"] > 0.3  # and the follower caught up afterwards


def test_independent_walks_score_about_a_coin_flip():
    rng = np.random.default_rng(3)
    idx = minute_index(trading_days(30))
    a = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, SIGMA, len(idx)))), index=idx)
    b = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, SIGMA, len(idx)))), index=idx)
    r = ll.analyze_pair("A", a, "B", b, PARAMS)
    assert 0.4 < r["accuracy"] < 0.6
    assert r["confidence"] in ("weak", "not_enough_data")


def test_fading_relationship_is_caught_out_of_sample():
    # Lead holds on the first 21 of 30 days, then disappears. Hindsight would look great; the recent test doesn't.
    leader, follower = planted(follow_days=21)
    r = ll.analyze_pair("LEAD", leader, "FOLL", follower, PARAMS)
    assert r["in_sample"]["accuracy"] > 0.75
    assert r["out_of_sample"]["accuracy"] < 0.65
    assert r["windows"]["5"]["accuracy"] < 0.65 < r["windows"]["60"]["accuracy"]
    assert r["confidence"] != "strong"


def test_window_sample_counts_grow_with_the_window():
    r = ll.analyze_pair("LEAD", *planted()[:1], "FOLL", planted()[1], PARAMS)
    w = r["windows"]
    assert w["5"]["events"] < w["20"]["events"] <= w["60"]["events"] == r["events"]
    assert r["in_sample"]["events"] + r["out_of_sample"]["events"] == r["events"]
    assert r["in_sample"]["days"] == 21 and r["out_of_sample"]["days"] == 9


def test_not_enough_data():
    leader, follower = planted(n_days=2)
    r = ll.analyze_pair("LEAD", leader, "FOLL", follower, PARAMS)
    assert r["status"] == "not_enough_data" and r["confidence"] == "not_enough_data"
    assert "Only 2 trading days" in r["message"]


def test_events_do_not_overlap_or_cross_days():
    leader, follower = planted(n_days=5)
    df = ll.align(leader, follower)
    events, _ = ll.find_events(df, "a", "b", lookback=1, wait=5, threshold_pct=0.3)
    by_day = {}
    for e in events:
        by_day.setdefault(e["day"], []).append(e["time"])
    for times in by_day.values():
        gaps = np.diff([t.value for t in times]) / 60e9
        assert (gaps >= 5).all()


def test_confidence_labels():
    many = {"accuracy": 0.7, "events": 200}
    win = {
        "5": {"accuracy": 0.68, "events": 20},
        "20": {"accuracy": 0.7, "events": 80},
        "60": {"accuracy": 0.7, "events": 200},
    }
    assert ll.confidence_label(many, many, {"accuracy": 0.7, "events": 50}, win, 1e-6, True) == "strong"
    assert ll.confidence_label(many, many, {"accuracy": 0.7, "events": 50}, win, 1e-6, False) == "weak"
    assert ll.confidence_label(many, many, {"accuracy": 0.57, "events": 50}, win, 0.01, True) == "moderate"
    assert (
        ll.confidence_label({"accuracy": 0.9, "events": 20}, many, {"accuracy": 0.9, "events": 5}, win, 0.01, True)
        == "not_enough_data"
    )


def test_bar_size_is_inferred_for_second_bars():
    idx = pd.date_range("2026-06-01 13:30", periods=100, freq="1s", tz="UTC")
    assert ll.infer_bar_seconds(idx) == 1
    assert ll.bars_for(5, 1) == 5 and ll.bars_for(5, 60) == 1


def test_lag_grid_is_symmetric_and_capped():
    grid = ll.lag_grid(300)
    assert grid[0] == -300 and grid[-1] == 300 and 0 in grid
    assert len(grid) <= 2 * ll.MAX_LAG_POINTS + 1


def test_noise_reports_no_lag():
    rng = np.random.default_rng(11)
    idx = minute_index(trading_days(10))
    a = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, SIGMA, len(idx)))), index=idx)
    b = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, SIGMA, len(idx)))), index=idx)
    assert ll.analyze_pair("A", a, "B", b, PARAMS)["lag_seconds"] is None
