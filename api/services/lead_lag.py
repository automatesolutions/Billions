"""
Lead-lag research engine: does one instrument move first, and how reliably does the other follow?

Pure functions over two price series (no network), so the math can be tested on
synthetic data with a known, planted lead.

1. Cross-correlation: corr(leader return at t, follower return at t + k) for each
   lag k. If correlations at positive lags beat those at negative lags, A leads B.
   Returns never cross the overnight gap (each day is handled on its own).
2. Events: every time the leader moves at least `threshold_pct` over the lookback.
   A hit means the follower's move over the next `max_wait` has the same sign.
   Accuracy = hits / events. A coin flip scores about 50%. Events never overlap,
   so one big move counts once.
3. Out-of-sample: which ticker leads is decided on the older days only; accuracy is
   then measured on the newest days with that same orientation. A pattern that only
   holds in hindsight shows up as a gap between the two numbers.
4. Confidence combines accuracy, sample size, significance, and stability across
   the last 5, 20, and 60 trading days.
"""

import math
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from api.services.market_calendar import NEW_YORK

MIN_EVENTS = 30
MIN_OOS_EVENTS = 10
MIN_DAYS = 4
MIN_ROWS = 200
WINDOWS = (5, 20, 60)
OOS_FRACTION = 0.3
MAX_LAG_POINTS = 40  # cross-correlation points on each side
FFILL_LIMIT = 3  # bars a missing price may be carried forward within a day


@dataclass(frozen=True)
class EngineParams:
    threshold_pct: float = 0.3
    lookback_seconds: float = 60
    max_wait_seconds: float = 300


def bars_for(seconds: float, bar_seconds: int) -> int:
    return max(1, int(round(seconds / bar_seconds)))


def infer_bar_seconds(index: pd.DatetimeIndex) -> int:
    if len(index) < 3:
        return 60
    # total_seconds(), not asi8: pandas may store times in micro- or nanoseconds.
    diffs = pd.Series(index).diff().dt.total_seconds().to_numpy()[1:]
    diffs = diffs[(diffs > 0) & (diffs < 3600)]
    return max(1, int(round(float(np.median(diffs))))) if len(diffs) else 60


def align(a: pd.Series, b: pd.Series) -> pd.DataFrame:
    """Both closes on one time index, with a `day` column (New York date). Short gaps are carried forward within a day."""
    df = pd.concat({"a": a.astype(float), "b": b.astype(float)}, axis=1).sort_index()
    index = df.index if df.index.tz is not None else df.index.tz_localize("UTC")
    df["day"] = index.tz_convert(NEW_YORK).date
    df[["a", "b"]] = df.groupby("day")[["a", "b"]].ffill(limit=FFILL_LIMIT)
    df = df.dropna(subset=["a", "b"])
    return df[(df["a"] > 0) & (df["b"] > 0)]


def lag_grid(max_lag: int) -> List[int]:
    if max_lag <= MAX_LAG_POINTS:
        positive = list(range(1, max_lag + 1))
    else:
        positive = sorted({int(round(x)) for x in np.geomspace(1, max_lag, MAX_LAG_POINTS)})
    return [-k for k in reversed(positive)] + [0] + positive


def cross_correlation(df: pd.DataFrame, lags: List[int]) -> Dict[int, Optional[float]]:
    """{k: corr(a return at t, b return at t + k)}. Positive k with high correlation means a leads b."""
    if df.empty:
        return {k: None for k in lags}
    ra = np.log(df["a"]).groupby(df["day"]).diff()
    rb = np.log(df["b"]).groupby(df["day"]).diff()
    grouped = rb.groupby(df["day"])
    out = {}
    for k in lags:
        corr = ra.corr(grouped.shift(-k))
        out[k] = None if corr is None or not np.isfinite(corr) else float(corr)
    return out


def orientation(xcorr: Dict[int, Optional[float]]) -> Tuple[bool, float]:
    """(a leads, strength). Strength is mean correlation at positive lags minus mean at negative lags."""
    pos = [c for k, c in xcorr.items() if k > 0 and c is not None]
    neg = [c for k, c in xcorr.items() if k < 0 and c is not None]
    if not pos or not neg:
        return True, 0.0
    strength = float(np.mean(pos) - np.mean(neg))
    return strength >= 0, strength


def oriented(xcorr: Dict[int, Optional[float]], a_leads: bool) -> Dict[int, Optional[float]]:
    """Re-key so positive lags always mean "leader first, follower later"."""
    return dict(sorted(((k if a_leads else -k), c) for k, c in xcorr.items()))


def find_events(df: pd.DataFrame, leader: str, follower: str, lookback: int, wait: int, threshold_pct: float):
    """Non-overlapping leader moves and what the follower did next. Returns (events, average paths)."""
    events = []
    width = lookback + wait + 1
    sum_leader = np.zeros(width)
    sum_follower = np.zeros(width)
    for day, group in df.groupby("day", sort=True):
        pl = group[leader].to_numpy()
        pf = group[follower].to_numpy()
        n = len(pl)
        if n < width:
            continue
        move = np.full(n, np.nan)
        move[lookback:] = pl[lookback:] / pl[:-lookback] - 1
        candidates = np.where(np.abs(move) * 100 >= threshold_pct)[0]
        next_ok = 0
        for i in candidates:
            if i < next_ok or i + wait >= n:
                continue
            sign = 1.0 if move[i] > 0 else -1.0
            path = (pf[i + 1 : i + wait + 1] / pf[i] - 1) * sign * 100
            final = float(path[-1])
            hit = final > 0
            lag = int(np.argmax(path >= final / 2)) + 1 if hit else None  # bars until half the move was in
            events.append(
                {
                    "day": day,
                    "time": group.index[i],
                    "leader_move_pct": float(move[i] * 100),
                    "follower_move_pct": final,
                    "hit": hit,
                    "lag_bars": lag,
                }
            )
            sum_leader += (pl[i - lookback : i + wait + 1] / pl[i] - 1) * sign * 100
            sum_follower += (pf[i - lookback : i + wait + 1] / pf[i] - 1) * sign * 100
            next_ok = i + wait
    count = max(len(events), 1)
    return events, (sum_leader / count, sum_follower / count)


def accuracy(events: List[Dict]) -> Dict:
    n = len(events)
    hits = sum(e["hit"] for e in events)
    return {"accuracy": hits / n if n else None, "events": n, "hits": hits}


def confidence_label(
    full: Dict, is_: Dict, oos: Dict, windows: Dict, p_value: Optional[float], direction_held: Optional[bool]
) -> str:
    if full["events"] < MIN_EVENTS or oos["events"] < MIN_OOS_EVENTS:
        return "not_enough_data"
    accs = [w["accuracy"] for w in windows.values() if w["events"] >= 10 and w["accuracy"] is not None]
    stable = len(accs) >= 2 and max(accs) - min(accs) <= 0.10
    a, ia, oa = full["accuracy"], is_["accuracy"] or 0, oos["accuracy"]
    if direction_held is False:
        return "weak"
    if a >= 0.60 and ia >= 0.60 and oa >= 0.60 and full["events"] >= 100 and (p_value or 1) < 0.01 and stable:
        return "strong"
    if a >= 0.55 and oa >= 0.55 and (p_value or 1) < 0.05:
        return "moderate"
    return "weak"


def _round(value, digits=4):
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return None
    return round(float(value), digits)


def analyze_pair(
    a_name: str,
    a: pd.Series,
    b_name: str,
    b: pd.Series,
    params: EngineParams = EngineParams(),
    bar_seconds: Optional[int] = None,
    oos_fraction: float = OOS_FRACTION,
) -> Dict:
    """Full analysis of one pair. Whichever way round you pass them, the result names the leader."""
    df = align(a, b)
    bar_s = bar_seconds or infer_bar_seconds(df.index)
    days = sorted(set(df["day"]))
    pair_key = "|".join(sorted([a_name, b_name]))
    base = {
        "pair_key": pair_key,
        "leader": a_name,
        "follower": b_name,
        "bar_seconds": bar_s,
        "resolution": "second" if bar_s < 60 else "minute",
        "days": len(days),
        "params": asdict(params),
    }
    if len(days) < MIN_DAYS or len(df) < MIN_ROWS:
        return {
            **base,
            "status": "not_enough_data",
            "message": (
                f"Only {len(days)} trading days and {len(df)} shared bars. "
                f"Need at least {MIN_DAYS} days and {MIN_ROWS} bars."
            ),
            "confidence": "not_enough_data",
            "events": 0,
            "hits": 0,
            "accuracy": None,
            "lag_seconds": None,
            "lead_strength": None,
            "p_value": None,
            "in_sample": {"accuracy": None, "events": 0, "days": 0},
            "out_of_sample": {"accuracy": None, "events": 0, "days": 0, "direction_held": None},
            "avg_follower_move_pct": None,
            "avg_lag_seconds": None,
            "windows": {str(w): {"accuracy": None, "events": 0} for w in WINDOWS},
        }

    n_oos = max(1, int(round(len(days) * oos_fraction)))
    is_days, oos_days = set(days[:-n_oos]), set(days[-n_oos:])
    wait = bars_for(params.max_wait_seconds, bar_s)
    lookback = bars_for(params.lookback_seconds, bar_s)
    lags = lag_grid(wait)

    # Orientation comes from the older days only, so the out-of-sample test is honest.
    a_leads, strength = orientation(cross_correlation(df[df["day"].isin(is_days)], lags))
    oos_a_leads, _ = orientation(cross_correlation(df[df["day"].isin(oos_days)], lags))
    leader, follower = (a_name, b_name) if a_leads else (b_name, a_name)
    lcol, fcol = ("a", "b") if a_leads else ("b", "a")

    profile = oriented(cross_correlation(df, lags), a_leads)
    positive = {k: c for k, c in profile.items() if k > 0 and c is not None}
    # Only call it a lag if the peak clears the rough 95% noise band for a correlation, 2 / sqrt(N).
    noise_band = 2 / math.sqrt(max(len(df) - len(days), 1))
    best = max(positive, key=positive.get) if positive and max(positive.values()) > noise_band else None

    events, (leader_path, follower_path) = find_events(df, lcol, fcol, lookback, wait, params.threshold_pct)
    full = accuracy(events)
    is_ = accuracy([e for e in events if e["day"] in is_days])
    oos = accuracy([e for e in events if e["day"] in oos_days])
    p_value = binomtest(full["hits"], full["events"], 0.5, alternative="greater").pvalue if full["events"] else None
    windows = {}
    for w in WINDOWS:
        recent = set(days[-w:])
        stats = accuracy([e for e in events if e["day"] in recent])
        windows[str(w)] = {"accuracy": _round(stats["accuracy"]), "events": stats["events"]}

    lags_hit = [e["lag_bars"] for e in events if e["lag_bars"] is not None]
    histogram = pd.Series(lags_hit).value_counts().sort_index() if lags_hit else pd.Series(dtype=int)
    daily = []
    for day in days:
        stats = accuracy([e for e in events if e["day"] == day])
        daily.append({"day": day.isoformat(), "accuracy": _round(stats["accuracy"]), "events": stats["events"]})

    direction_held = oos_a_leads == a_leads
    return {
        **base,
        "status": "ok",
        "leader": leader,
        "follower": follower,
        "lag_seconds": best * bar_s if best else None,
        "lead_strength": _round(abs(strength)),
        "accuracy": _round(full["accuracy"]),
        "events": full["events"],
        "hits": full["hits"],
        "p_value": _round(p_value, 6),
        "in_sample": {"accuracy": _round(is_["accuracy"]), "events": is_["events"], "days": len(is_days)},
        "out_of_sample": {
            "accuracy": _round(oos["accuracy"]),
            "events": oos["events"],
            "days": len(oos_days),
            "direction_held": direction_held,
        },
        "avg_follower_move_pct": _round(np.mean([e["follower_move_pct"] for e in events])) if events else None,
        "avg_lag_seconds": _round(np.mean(lags_hit) * bar_s, 1) if lags_hit else None,
        "windows": windows,
        "confidence": confidence_label(full, is_, oos, windows, p_value, direction_held),
        "details": {
            "xcorr": [{"lag_seconds": k * bar_s, "corr": _round(c)} for k, c in profile.items()],
            "event_path": [
                {"offset_seconds": (i - lookback) * bar_s, "leader_pct": _round(lp), "follower_pct": _round(fp)}
                for i, (lp, fp) in enumerate(zip(leader_path, follower_path))
            ],
            "lag_histogram": [{"lag_seconds": int(k) * bar_s, "count": int(v)} for k, v in histogram.items()],
            "daily": daily,
        },
    }
