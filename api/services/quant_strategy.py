"""
Quant strategy at a chosen bar interval, following docs/quant_trading_comprehensive_guide.pdf.

Pure functions: closes in, numbers out. No network, no database.

Pipeline (one bar = 1 minute, 1 hour, 4 hours or 1 day):
1. Log returns r_t = ln(P_t / P_{t-1}) and leakage-safe features (quant_analysis.build_features).
2. Three base models (AR(1), XGBoost, online Passive-Aggressive) and a stacked meta-learner
   with non-negative weights, fit on the oldest 75% and tested on the newest 25%.
3. Validation: expanding and rolling walk-forward, and a random up/down baseline.
4. Sizing (guide section 4, smooth non-linear): position = tanh(y_hat / (0.1 * bar volatility)),
   a fraction of the user's maximum position, from -1 (full short) to +1 (full long).
5. Costs: changing the position by d units costs d * (round-trip cost / 2).
6. Action gates: the action is "flat" unless the tested edge survives randomness and costs.

This describes a model's output. It places no orders.
"""

import math
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from api.services import quant_analysis as qa
from api.services.bars import Interval

MIN_BARS = qa.MIN_RETURNS
TARGET_FOLDS = 12  # walk-forward folds per scheme, whatever the interval
MIN_LIVE_SIGNALS = 20  # below this the live record is "too early to tell"


# --------------------------------------------------------------------------- building blocks


def session_starts(index: pd.DatetimeIndex) -> np.ndarray:
    """True where a bar is the first of its session (its return spans the overnight gap)."""
    dates = pd.Series(index.date, index=index)
    return (dates != dates.shift(1)).values


def feature_frame(returns: pd.Series, iv: Interval) -> pd.DataFrame:
    return qa.build_features(returns, iv.slot_minutes, iv.slots)


def usable_rows(returns: pd.Series, iv: Interval) -> pd.DataFrame:
    """Feature rows with complete features. Flat-overnight intervals drop targets that span the gap."""
    data = feature_frame(returns, iv)
    if iv.flat_overnight:
        data = data[~session_starts(pd.DatetimeIndex(returns.index))]
    return data.dropna()


def xgb_features(iv: Interval) -> List[str]:
    return qa.slot_features(iv.slots) if iv.slot_minutes else list(qa.XGB_FEATURES)


def size_position(forecast: np.ndarray, bar_volatility: float) -> np.ndarray:
    """Guide's smooth sizing: tanh(y_hat / (0.1 * sigma)), bounded in (-1, 1)."""
    scale = qa.STRENGTH_SCALE * bar_volatility
    if not scale or not np.isfinite(scale):
        return np.zeros_like(np.asarray(forecast, float))
    return np.tanh(np.asarray(forecast, float) / scale)


def net_returns(position: np.ndarray, actual: np.ndarray, cost_bps: float, breaks: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Per-bar return of holding `position` over each bar, after costs.
    Moving the position by d costs d * cost/2 (half a round trip per side).
    `breaks[i]` True means bar i starts a new session for a flat-overnight strategy:
    the previous position is closed at the prior session's end and reopened here.
    """
    pos, y = np.asarray(position, float), np.asarray(actual, float)
    half = cost_bps / 2 / 1e4
    prev = np.concatenate([[0.0], pos[:-1]])
    cost = np.abs(pos - prev)
    if breaks is not None and len(pos):
        b = np.array(breaks, dtype=bool)  # copy: the caller's array stays untouched
        b[0] = False
        cost = np.where(b, np.abs(prev) + np.abs(pos), cost)  # close yesterday's, open today's
    return pos * y - cost * half


def _stamp(ts: pd.Timestamp, iv: Interval) -> str:
    return pd.Timestamp(ts).isoformat() if iv.intraday else pd.Timestamp(ts).date().isoformat()


def _row(name: str, pred: np.ndarray, y: np.ndarray, iv: Interval) -> Dict:
    edge = qa.edge_metrics(qa.signal_returns(pred, y), iv.periods_per_year)
    return {
        "model": name,
        "hit_rate": qa.hit_rate(pred, y),
        "expected_value": edge["expected_value"],
        "sharpe": edge["sharpe"],
        "total_log_return": edge["total_log_return"],
        "n": edge["n"],
    }


# --------------------------------------------------------------------------- heavy part (cached per bar)


def fit_and_test(close: pd.Series, iv: Interval, next_start: pd.Timestamp) -> Dict:
    """
    Fit, test and validate the stack, then forecast the next bar.
    Independent of costs, so it can be cached until the next bar closes.
    Raises ValueError when there is not enough history.
    """
    close = close.dropna().astype(float)
    returns = qa.log_returns(close)
    data = usable_rows(returns, iv)
    if len(data) < MIN_BARS:
        raise ValueError(f"Needs at least {MIN_BARS} usable {iv.label} bars; found {len(data)}.")
    features = xgb_features(iv)

    def stack_predict(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
        return qa.predict_stack_oos(qa.fit_stack(train, xgb_features=features), test)["stacked"]

    train, test = qa.time_split(data)
    y_test = test["target"].values
    stack = qa.fit_stack(train, xgb_features=features)
    oos = qa.predict_stack_oos(stack, test)
    bar_volatility = float(train["target"].std(ddof=1))  # training period only: no look-ahead in sizing

    step = max(qa.WALK_FORWARD_STEP, math.ceil(len(data) * 0.5 / TARGET_FOLDS))
    walk = [
        qa.walk_forward(data, scheme, stack_predict, step=step, periods_per_year=iv.periods_per_year, intraday=iv.intraday)
        for scheme in ("expanding", "rolling")
    ]
    strategy = qa.signal_returns(oos["stacked"], y_test)
    baseline = qa.random_baseline(y_test, float(strategy.sum()))
    ar = stack.ar1.params

    # Next bar: refit on everything, then forecast with each model and the stack.
    full = qa.fit_stack(data, xgb_features=features)
    extended = pd.concat([returns, pd.Series([np.nan], index=pd.DatetimeIndex([next_start]))])
    row = feature_frame(extended, iv).iloc[[-1]].drop(columns="target")
    base = {"ar1": float(full.ar1.predict(row)[0]), "xgboost": float(full.xgb.predict(row)[0]), "online": float(full.online.predict(row)[0])}
    forecast = float(qa.meta_predict(full.meta, np.array([[base["ar1"], base["xgboost"], base["online"]]]))[0])
    full_volatility = float(data["target"].std(ddof=1))

    return {
        "interval": iv.key,
        "last_bar": _stamp(close.index[-1], iv),
        "last_bar_ts": pd.Timestamp(close.index[-1]),
        "next_bar": _stamp(next_start, iv),
        "price": float(close.iloc[-1]),
        "bars": int(len(data)),
        "test_index": test.index,
        "y_test": y_test,
        "oos_stacked": oos["stacked"],
        "bar_volatility_train": bar_volatility,
        "breaks": session_starts(pd.DatetimeIndex(test.index)) if iv.flat_overnight else None,
        "models": [_row(name, oos[name], y_test, iv) for name in (*qa.BASE_MODELS, "stacked")],
        "gross": qa.edge_metrics(strategy, iv.periods_per_year),
        "hit_rate": qa.hit_rate(oos["stacked"], y_test),
        "walk_forward": walk,
        "random_baseline": {**baseline, "beats_random": baseline["percentile"] >= qa.SIGNIFICANCE_PERCENTILE},
        "regime": qa.regime_label(ar["w"], ar["t_stat"]),
        "split": {
            "train_bars": len(train),
            "test_bars": len(test),
            "train_start": _stamp(train.index[0], iv),
            "test_start": _stamp(test.index[0], iv),
            "test_end": _stamp(test.index[-1], iv),
            "shuffled": False,
        },
        "next": {
            "features": {k: float(row[k].iloc[0]) for k in qa.ONLINE_FEATURES + ["dir_1", "dirma_5", "dirma_20"]},
            "slot": _slot(row, iv),
            "base": base,
            "meta_weights": dict(zip(qa.BASE_MODELS, (float(w) for w in full.meta["weights"]))),
            "meta_bias": float(full.meta["bias"]),
            "ar1": {"w": float(full.ar1.params["w"]), "b": float(full.ar1.params["b"])},
            "forecast": forecast,
            "bar_volatility": full_volatility,
        },
    }


def _slot(row: pd.DataFrame, iv: Interval) -> Optional[int]:
    if not iv.slot_minutes:
        return None
    hits = [k for k in range(iv.slots) if row[f"slot_{k}"].iloc[0] == 1.0]
    return hits[0] if hits else None


# --------------------------------------------------------------------------- light part (per request)


def backtest_net(fit: Dict, cost_bps: float, iv: Interval) -> Dict:
    """What following the sized position would have earned on test bars, after costs."""
    position = size_position(fit["oos_stacked"], fit["bar_volatility_train"])
    net = net_returns(position, fit["y_test"], cost_bps, fit["breaks"])
    equity, drawdown = qa.equity_and_drawdown(net)
    hold, _ = qa.equity_and_drawdown(fit["y_test"])
    edge = qa.edge_metrics(net, iv.periods_per_year)
    turnover = np.abs(np.diff(np.concatenate([[0.0], position])))
    return {
        **edge,
        "cost_bps": cost_bps,
        "avg_turnover": float(turnover.mean()) if len(turnover) else 0.0,
        "equity": equity.tolist(),
        "hold_equity": hold.tolist(),
        "drawdown": drawdown.tolist(),
        "max_drawdown": float(drawdown.min()) if len(drawdown) else 0.0,
        "dates": [_stamp(t, iv) for t in fit["test_index"]],
        "expected_move_bps": float(np.mean(np.abs(fit["oos_stacked"])) * 1e4),
    }


def gates(fit: Dict, net: Dict, live_verdict: str, strength: float, iv: Interval) -> List[Dict]:
    """Checks the action must pass. Any failure makes the action "flat"."""
    walk_ok = all((w["expected_value"] or 0) > 0 for w in fit["walk_forward"])
    baseline = fit["random_baseline"]
    session = []
    if iv.flat_overnight:
        same = fit["next_bar"][:10] == fit["last_bar"][:10]
        session = [
            {
                "id": "same_session",
                "label": "Next bar is in this session",
                "passed": same,
                "detail": "The next bar is in the same session."
                if same
                else "The session has ended. This strategy is flat overnight and was not tested across the gap.",
            }
        ]
    return session + [
        {
            "id": "beats_random",
            "label": "Beats random on test bars",
            "passed": bool(baseline["beats_random"]),
            "detail": f"Beats {baseline['percentile']:.0f}% of random up/down strategies (needs 95%).",
        },
        {
            "id": "walk_forward",
            "label": "Holds up in walk-forward",
            "passed": bool(walk_ok),
            "detail": "Expected value per bar is positive in both expanding and rolling walk-forward."
            if walk_ok
            else "Expected value per bar is not positive in both walk-forward schemes.",
        },
        {
            "id": "after_costs",
            "label": "Profitable after costs",
            "passed": bool((net["expected_value"] or 0) > 0),
            "detail": f"Sized strategy earned {net['expected_value'] * 1e4:+.2f} bps per bar on test bars after a "
            f"{net['cost_bps']:.1f} bps round-trip cost.",
        },
        {
            "id": "live_record",
            "label": "Live record not failing",
            "passed": live_verdict != "underperforming",
            "detail": "The live hit rate is clearly below the backtest."
            if live_verdict == "underperforming"
            else "No sign yet that the live signals do worse than the backtest.",
        },
        {
            "id": "strength",
            "label": "Signal strong enough",
            "passed": abs(strength) >= qa.NEUTRAL_BAND,
            "detail": f"Strength {strength:+.2f} (needs at least ±{qa.NEUTRAL_BAND:.2f}).",
        },
    ]


def action(fit: Dict, net: Dict, live_verdict: str, iv: Interval) -> Dict:
    """The current action: direction, size and the gates behind it."""
    nxt = fit["next"]
    strength = float(size_position(np.array([nxt["forecast"]]), nxt["bar_volatility"])[0])
    checks = gates(fit, net, live_verdict, strength, iv)
    model_direction = "neutral" if abs(strength) < qa.NEUTRAL_BAND else ("long" if strength > 0 else "short")
    passed = all(c["passed"] for c in checks)
    return {
        "action": model_direction if passed and model_direction != "neutral" else "flat",
        "model_direction": model_direction,
        "position": strength if passed else 0.0,
        "strength": strength,
        "forecast_log_return": nxt["forecast"],
        "forecast_bps": nxt["forecast"] * 1e4,
        "expected_price": fit["price"] * float(np.exp(nxt["forecast"])),
        "for_bar": fit["next_bar"],
        "gates": checks,
    }


# --------------------------------------------------------------------------- live record


def live_summary(signals: Sequence[Dict], fit: Dict, iv: Interval) -> Dict:
    """
    Score the recorded signals and compare them with the backtest.

    `signals` are dicts with forecast, position, cost_bps, realized (log return or None),
    bar_time and target_time, oldest first. Two records:
    - model record: follow sign(forecast) on every scored signal, before costs (comparable to the backtest);
    - action record: hold the recorded position (flat when gated), after the recorded cost.
    """
    scored = [s for s in signals if s["realized"] is not None]
    n = len(scored)
    forecasts = np.array([s["forecast"] for s in scored], float)
    realized = np.array([s["realized"] for s in scored], float)
    model = qa.signal_returns(forecasts, realized) if n else np.array([])

    # Action record. A position is carried only between back-to-back signals.
    action_returns = []
    prev_pos, prev_target = 0.0, None
    for s in scored:
        pos, half = float(s["position"]), float(s["cost_bps"]) / 2 / 1e4
        carried = prev_pos if prev_target is not None and prev_target == s["bar_time"] else 0.0
        closing = abs(prev_pos) * half if carried == 0.0 else 0.0  # close an orphaned position
        action_returns.append(pos * s["realized"] - abs(pos - carried) * half - closing)
        prev_pos, prev_target = pos, s["target_time"]
    action_returns = np.array(action_returns, float)
    equity, _ = qa.equity_and_drawdown(action_returns)

    bt_hit = fit["hit_rate"]
    bt_ev = fit["gross"]["expected_value"]
    bt_std = float(np.std(qa.signal_returns(fit["oos_stacked"], fit["y_test"]), ddof=1))
    live_hit = qa.hit_rate(forecasts, realized) if n else None
    live_ev = float(model.mean()) if n else None

    if n < MIN_LIVE_SIGNALS:
        verdict, z = "too_early", None
    else:
        se = math.sqrt(bt_hit * (1 - bt_hit) / n) if bt_hit and 0 < bt_hit < 1 else None
        z = (live_hit - bt_hit) / se if se and live_hit is not None else None
        if z is not None and z <= -2:
            verdict = "underperforming"
        elif z is not None and z >= 2:
            verdict = "outperforming"
        else:
            verdict = "in_line"

    live_baseline = qa.random_baseline(realized, float(model.sum())) if n >= MIN_LIVE_SIGNALS else None
    ev_band = 2 * bt_std / math.sqrt(n) if n else None
    return {
        "verdict": verdict,
        "min_signals": MIN_LIVE_SIGNALS,
        "recorded": len(signals),
        "scored": n,
        "pending": sum(1 for s in signals if s["realized"] is None and not s.get("void")),
        "model": {
            "hit_rate": live_hit,
            "expected_value": live_ev,
            "total_log_return": float(model.sum()) if n else None,
            "hit_rate_z": z,
        },
        "action": {
            "expected_value": float(action_returns.mean()) if n else None,
            "total_log_return": float(action_returns.sum()) if n else None,
            "equity": equity.tolist(),
            "dates": [_stamp(pd.Timestamp(s["bar_time"]), iv) for s in scored],
            "trades": int(sum(1 for s in scored if s["position"] != 0)),
        },
        "backtest": {
            "hit_rate": bt_hit,
            "expected_value": bt_ev,
            "hit_rate_band": [  # a hit rate lives in [0, 1]
                max(0.0, bt_hit - 2 * math.sqrt(bt_hit * (1 - bt_hit) / n)),
                min(1.0, bt_hit + 2 * math.sqrt(bt_hit * (1 - bt_hit) / n)),
            ]
            if n and bt_hit and 0 < bt_hit < 1
            else None,
            "expected_value_band": [bt_ev - ev_band, bt_ev + ev_band] if ev_band is not None and bt_ev is not None else None,
        },
        "random_baseline": live_baseline,
    }


# --------------------------------------------------------------------------- order wording

INSTRUCTIONS = {
    # (side before, side after): (code, headline, plain sentence)
    (0, 1): ("buy", "Buy", "Open a long position."),
    (0, -1): ("sell_short", "Sell short", "Open a short position."),
    (0, 0): ("hold_flat", "Hold", "Stay out: no position."),
    (1, 1): ("hold_long", "Hold", "Keep the long position."),
    (-1, -1): ("hold_short", "Hold", "Keep the short position."),
    (1, 0): ("sell", "Sell", "Close the long position."),
    (-1, 0): ("buy_to_cover", "Buy to cover", "Close the short position."),
    (1, -1): ("reverse_to_short", "Sell + short", "Close the long and open a short."),
    (-1, 1): ("reverse_to_long", "Cover + buy", "Close the short and open a long."),
}


def _side(position: float) -> int:
    return 0 if abs(position) < 1e-9 else (1 if position > 0 else -1)


def instruction(previous: float, position: float) -> Dict:
    """
    The order that moves the guide's position from `previous` (held over the bar that just closed)
    to `position` (for the next bar): buy, sell, sell short, buy to cover, or hold.
    """
    code, headline, sentence = INSTRUCTIONS[(_side(previous), _side(position))]
    change = position - previous
    if code in ("hold_long", "hold_short") and abs(change) >= 0.05:
        verb = "add to" if abs(position) > abs(previous) else "trim"
        sentence = f"{sentence[:-1]}, and {verb} it from {abs(previous):.0%} to {abs(position):.0%} of your max."
    return {"code": code, "headline": headline, "sentence": sentence, "previous_position": previous, "change": change}
