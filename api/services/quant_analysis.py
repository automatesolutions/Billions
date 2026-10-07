"""
Per-stock quant analysis, following docs/reference/quant-trading-handbook.pdf.

Pure functions: prices in, numbers out. No network, no database, no clock.
Everything here describes data. Nothing produces orders, sizes or leverage.

Conventions
- r_t = ln(P_t / P_{t-1}) (log returns, time-additive).
- A feature row indexed t is used to forecast r_t, and may only use returns up to t-1.
- "Out-of-sample" (OOS) means predicted by a model that never saw that day in training.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from xgboost import XGBRegressor

TRADING_DAYS = 252
TRAIN_FRACTION = 0.75
LAGS = (1, 2, 3, 5)
WINDOWS = (5, 20)
MIN_RETURNS = 250  # about one year of daily data
MIN_TEST_OBS = 100  # below this the test sample is flagged as small
DEFAULT_ROUND_TRIP_COST_BPS = 10.0
RANDOM_BASELINE_DRAWS = 1000
WALK_FORWARD_STEP = 21  # one trading month per fold
NEUTRAL_BAND = 0.15  # |strength| below this is "neutral"
SIGNIFICANCE_PERCENTILE = 95.0

AR_FEATURES = ["lag_1"]
XGB_FEATURES = ["wd_0", "wd_1", "wd_2", "wd_3", "wd_4", "dir_1", "dirma_5", "dirma_20"]
ONLINE_FEATURES = ["lag_1", "lag_2", "lag_3", "lag_5", "ma_5", "ma_20"]
BASE_MODELS = ("ar1", "xgboost", "online")


# --------------------------------------------------------------------------- returns and features


def log_returns(close: pd.Series) -> pd.Series:
    """r_t = ln(P_t / P_{t-1}). The first day has no return and is dropped."""
    close = close.astype(float)
    return np.log(close / close.shift(1)).dropna()


def build_features(returns: pd.Series) -> pd.DataFrame:
    """
    One row per day t with the target r_t and features known before day t.

    Leakage guard: every feature is computed from `returns.shift(k)` with k >= 1,
    and rolling windows are applied *after* shift(1). The weekday is a property of
    the calendar date itself, known in advance.
    """
    r = returns.astype(float)
    direction = np.sign(r)
    prev = r.shift(1)
    out = pd.DataFrame(index=r.index)
    out["target"] = r
    for k in LAGS:
        out[f"lag_{k}"] = r.shift(k)
    out["dir_1"] = direction.shift(1)
    for n in WINDOWS:
        out[f"ma_{n}"] = prev.rolling(n).mean()
        out[f"dirma_{n}"] = direction.shift(1).rolling(n).mean()
    weekday = pd.Series(pd.DatetimeIndex(r.index).weekday, index=r.index)
    for d in range(5):
        out[f"wd_{d}"] = (weekday == d).astype(float)
    return out


def next_day_features(returns: pd.Series, next_date: pd.Timestamp) -> pd.DataFrame:
    """Feature row for the next trading day (its target is unknown)."""
    extended = pd.concat([returns, pd.Series([np.nan], index=pd.DatetimeIndex([next_date]))])
    return build_features(extended).iloc[[-1]].drop(columns="target")


# --------------------------------------------------------------------------- edge and risk metrics


def edge_metrics(strategy_returns: Sequence[float]) -> Dict:
    """
    Win rate, average win/loss, expected value and annualised Sharpe of a series of per-period returns.
    EV = P(win) * avg win - P(loss) * avg loss. Days with exactly zero return count as neither.
    """
    s = np.asarray(strategy_returns, dtype=float)
    s = s[np.isfinite(s)]
    n = len(s)
    wins, losses = s[s > 0], s[s < 0]
    p_win = len(wins) / n if n else np.nan
    p_loss = len(losses) / n if n else np.nan
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(-losses.mean()) if len(losses) else 0.0
    std = s.std(ddof=1) if n > 1 else np.nan
    return {
        "n": n,
        "win_rate": p_win,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "expected_value": p_win * avg_win - p_loss * avg_loss if n else np.nan,
        "sharpe": float(s.mean() / std * np.sqrt(TRADING_DAYS)) if n > 1 and std > 0 else np.nan,
        "total_log_return": float(s.sum()) if n else np.nan,
    }


def equity_and_drawdown(log_returns_: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    """Growth of 1 (compounded via exp of summed log returns) and drawdown from the running peak."""
    lr = np.nan_to_num(np.asarray(log_returns_, dtype=float))
    equity = np.exp(np.cumsum(lr))
    peak = np.maximum.accumulate(np.concatenate([[1.0], equity]))[1:]
    return equity, equity / peak - 1.0


def max_drawdown(log_returns_: Sequence[float]) -> float:
    _, dd = equity_and_drawdown(log_returns_)
    return float(dd.min()) if len(dd) else 0.0


MIN_RISK_RETURNS = 20


def risk_context(returns: pd.Series) -> Dict:
    """Plain risk numbers for holding the stock over the whole sample."""
    r = returns.dropna()
    worst_idx = r.idxmin()
    return {
        "annual_volatility": float(r.std(ddof=1) * np.sqrt(TRADING_DAYS)),
        "max_drawdown": max_drawdown(r.values),
        "worst_day": {"date": pd.Timestamp(worst_idx).date().isoformat(), "return": float(np.expm1(r.min()))},
        "sharpe": float(r.mean() / r.std(ddof=1) * np.sqrt(TRADING_DAYS)),
        "total_return": float(np.expm1(r.sum())),
        "days": int(len(r)),
    }


def hit_rate(pred: Sequence[float], actual: Sequence[float]) -> float:
    """Share of days where the forecast sign matches the realised sign (zero days skipped)."""
    p, a = np.sign(np.asarray(pred, float)), np.sign(np.asarray(actual, float))
    mask = (p != 0) & (a != 0)
    return float((p[mask] == a[mask]).mean()) if mask.any() else np.nan


def signal_returns(pred: Sequence[float], actual: Sequence[float]) -> np.ndarray:
    """Per-day return of following the forecast sign: sign(y_hat_t) * r_t."""
    return np.sign(np.asarray(pred, float)) * np.asarray(actual, float)


# --------------------------------------------------------------------------- AR(1)


def fit_ar1(x: np.ndarray, y: np.ndarray) -> Dict:
    """OLS fit of y = w x + b, with the standard error and t-statistic of w."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    x_mean, y_mean = x.mean(), y.mean()
    sxx = ((x - x_mean) ** 2).sum()
    w = float(((x - x_mean) * (y - y_mean)).sum() / sxx) if sxx > 0 else 0.0
    b = float(y_mean - w * x_mean)
    resid = y - (w * x + b)
    dof = max(len(x) - 2, 1)
    se = float(np.sqrt((resid**2).sum() / dof / sxx)) if sxx > 0 else np.nan
    t = w / se if se and np.isfinite(se) and se > 0 else np.nan
    return {"w": w, "b": b, "se": se, "t_stat": float(t), "n": int(len(x))}


def regime_label(w: float, t_stat: float) -> Dict:
    regime = "mean_reversion" if w < 0 else "momentum"
    significant = bool(np.isfinite(t_stat) and abs(t_stat) >= 2.0)
    return {"regime": regime, "w": w, "t_stat": t_stat, "significant": significant}


# --------------------------------------------------------------------------- models


class AR1Model:
    def fit(self, X: pd.DataFrame, y: np.ndarray):
        self.params = fit_ar1(X[AR_FEATURES[0]].values, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.params["w"] * X[AR_FEATURES[0]].values + self.params["b"]


class XGBModel:
    def fit(self, X: pd.DataFrame, y: np.ndarray):
        self.model = XGBRegressor(
            max_depth=3, n_estimators=50, learning_rate=0.12, subsample=1.0, random_state=0, n_jobs=1, verbosity=0
        )
        self.model.fit(X[XGB_FEATURES].values, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X[XGB_FEATURES].values).astype(float)


def pa1_step(w: np.ndarray, b: float, x: np.ndarray, y: float, aggressiveness: float) -> Tuple[np.ndarray, float]:
    """
    One Passive-Aggressive (PA-I) regression update with epsilon = 0 (Crammer et al., 2006):
    tau = min(C, |error| / ||x||^2); w += tau * sign(error) * x; b += tau * sign(error).
    Matches scikit-learn's SGDRegressor(loss="epsilon_insensitive", penalty=None, learning_rate="pa1")
    exactly (see test_quant_analysis.py), without its per-row Python overhead.
    """
    error = y - (x @ w + b)
    norm = x @ x
    if norm > 0 and error != 0:
        tau = min(aggressiveness, abs(error) / norm)
        w = w + tau * np.sign(error) * x
        b = b + tau * np.sign(error)
    return w, b


class OnlineModel:
    """
    Passive-Aggressive regression, updated one day at a time.

    scikit-learn deprecated PassiveAggressiveRegressor in 1.8 in favour of
    SGDRegressor(learning_rate="pa1"); `pa1_step` is that same update.
    epsilon = 0 because daily returns (~0.01) are far below sklearn's default 0.1.
    Features are standardised with training-period statistics only.
    """

    def __init__(self, aggressiveness: float = 0.1):
        self.aggressiveness = aggressiveness

    def _scale(self, X: pd.DataFrame) -> np.ndarray:
        return (X[ONLINE_FEATURES].values - self.mean) / self.std

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        """One pass in time order. Stores the forecast made *before* each update (out-of-sample by construction)."""
        values = X[ONLINE_FEATURES].values
        self.mean, self.std = values.mean(axis=0), values.std(axis=0)
        self.std[self.std == 0] = 1.0
        self.w, self.b = np.zeros(len(ONLINE_FEATURES)), 0.0
        self.prequential = self._run(self._scale(X), np.asarray(y, float))
        return self

    def _run(self, Z: np.ndarray, y: np.ndarray) -> np.ndarray:
        preds = np.empty(len(y))
        for i in range(len(y)):
            preds[i] = Z[i] @ self.w + self.b
            self.w, self.b = pa1_step(self.w, self.b, Z[i], y[i], self.aggressiveness)
        return preds

    def predict_online(self, X: pd.DataFrame, y: np.ndarray) -> np.ndarray:
        """Forecast each day, then learn from it. Never sees r_t before forecasting r_t."""
        return self._run(self._scale(X), np.asarray(y, float))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self._scale(X) @ self.w + self.b).astype(float)


def fit_meta(base_preds: np.ndarray, y: np.ndarray) -> Dict:
    """
    Constrained linear meta-learner: y ≈ b + Σ w_i p_i with every w_i >= 0 and a free bias b.
    Solved as non-negative least squares on centred data, then b = mean(y) - mean(p)·w.
    """
    P, y = np.asarray(base_preds, float), np.asarray(y, float)
    p_mean, y_mean = P.mean(axis=0), y.mean()
    weights, _ = nnls(P - p_mean, y - y_mean)
    return {"weights": weights, "bias": float(y_mean - p_mean @ weights)}


def meta_predict(meta: Dict, base_preds: np.ndarray) -> np.ndarray:
    return np.asarray(base_preds, float) @ meta["weights"] + meta["bias"]


@dataclass
class StackedFit:
    ar1: AR1Model
    xgb: XGBModel
    online: OnlineModel
    meta: Dict
    oof_base: np.ndarray  # out-of-fold base predictions used to fit the meta-learner
    oof_y: np.ndarray


def fit_stack(train: pd.DataFrame, n_blocks: int = 3) -> StackedFit:
    """
    Fit the three base models and the meta-learner on `train` without leakage:
    the meta-learner is fit on out-of-fold base forecasts (each block predicted by
    models trained only on earlier blocks).
    """
    y = train["target"].values
    n = len(train)
    edges = np.linspace(0, n, n_blocks + 1).astype(int)

    online = OnlineModel().fit(train, y)  # its prequential forecasts are already out-of-sample
    oof_rows, oof_y = [], []
    for k in range(1, n_blocks):
        past, block = train.iloc[: edges[k]], train.iloc[edges[k] : edges[k + 1]]
        ar = AR1Model().fit(past, past["target"].values)
        xg = XGBModel().fit(past, past["target"].values)
        oof_rows.append(np.column_stack([ar.predict(block), xg.predict(block), online.prequential[edges[k] : edges[k + 1]]]))
        oof_y.append(block["target"].values)
    oof_base, oof_target = np.vstack(oof_rows), np.concatenate(oof_y)

    return StackedFit(
        ar1=AR1Model().fit(train, y),
        xgb=XGBModel().fit(train, y),
        online=online,
        meta=fit_meta(oof_base, oof_target),
        oof_base=oof_base,
        oof_y=oof_target,
    )


def predict_stack_oos(fit: StackedFit, test: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Out-of-sample forecasts on `test` for every base model and the stack. The online model keeps learning."""
    base = np.column_stack(
        [fit.ar1.predict(test), fit.xgb.predict(test), fit.online.predict_online(test, test["target"].values)]
    )
    return {"ar1": base[:, 0], "xgboost": base[:, 1], "online": base[:, 2], "stacked": meta_predict(fit.meta, base)}


# --------------------------------------------------------------------------- validation


def time_split(data: pd.DataFrame, train_fraction: float = TRAIN_FRACTION) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Oldest `train_fraction` for training, newest rest for testing. Never shuffled."""
    cut = int(len(data) * train_fraction)
    return data.iloc[:cut], data.iloc[cut:]


def walk_forward(
    data: pd.DataFrame,
    scheme: str,
    fit_predict: Callable[[pd.DataFrame, pd.DataFrame], np.ndarray],
    start_fraction: float = 0.5,
    step: int = WALK_FORWARD_STEP,
) -> Dict:
    """
    Expanding: train on [0, s), test on [s, s+step). Rolling: train on the last `window` days before s.
    The window length for rolling equals the first expanding window.
    """
    if scheme not in ("expanding", "rolling"):
        raise ValueError(scheme)
    n = len(data)
    start = int(n * start_fraction)
    windows = [(0 if scheme == "expanding" else s - start, s, min(s + step, n)) for s in range(start, n, step)]

    def run(window):
        lo, s, hi = window
        train, test = data.iloc[lo:s], data.iloc[s:hi]
        return train, test, fit_predict(train, test)

    results = [run(w) for w in windows]

    folds, preds, actual = [], [], []
    for train, test, p in results:
        y = test["target"].values
        folds.append(
            {
                "start": test.index[0].date().isoformat(),
                "train_days": len(train),
                "test_days": len(test),
                "hit_rate": hit_rate(p, y),
            }
        )
        preds.append(p)
        actual.append(y)
    p_all, y_all = np.concatenate(preds), np.concatenate(actual)
    edge = edge_metrics(signal_returns(p_all, y_all))
    return {
        "scheme": scheme,
        "folds": folds,
        "fold_count": len(folds),
        "n": int(len(y_all)),
        "hit_rate": hit_rate(p_all, y_all),
        "sharpe": edge["sharpe"],
        "expected_value": edge["expected_value"],
    }


def random_baseline(actual: Sequence[float], model_total: float, draws: int = RANDOM_BASELINE_DRAWS, seed: int = 0) -> Dict:
    """
    Compare the model's total test return with strategies that go up or down uniformly at random each day.
    Percentile = share of random strategies the model beats. p-value = share that do at least as well.
    """
    y = np.asarray(actual, float)
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(draws, len(y)))
    totals = signs @ y
    return {
        "draws": draws,
        "random_mean": float(totals.mean()),
        "random_p95": float(np.percentile(totals, 95)),
        "model_total": float(model_total),
        "excess_vs_random_mean": float(model_total - totals.mean()),
        "percentile": float((totals < model_total).mean() * 100),
        "p_value": float((totals >= model_total).mean()),
    }


# --------------------------------------------------------------------------- signal


STRENGTH_SCALE = 0.1  # a forecast worth 1/10 of a typical day's move scores tanh(1) = 0.76


def signal_strength(forecast: float, daily_volatility: float) -> float:
    """Bounded score in (-1, 1): tanh(y_hat / (0.1 * daily volatility))."""
    forecast_scale = STRENGTH_SCALE * daily_volatility
    if not np.isfinite(forecast) or not forecast_scale or not np.isfinite(forecast_scale):
        return 0.0
    return float(np.tanh(forecast / forecast_scale))


def describe_strength(strength: float) -> Dict:
    magnitude = abs(strength)
    if magnitude < NEUTRAL_BAND:
        direction = "neutral"
    else:
        direction = "up" if strength > 0 else "down"
    label = "weak" if magnitude < 0.33 else "moderate" if magnitude < 0.66 else "strong"
    return {"strength": strength, "direction": direction, "label": label}


# --------------------------------------------------------------------------- full analysis


def _model_row(name: str, pred: np.ndarray, y: np.ndarray) -> Dict:
    edge = edge_metrics(signal_returns(pred, y))
    return {
        "model": name,
        "hit_rate": hit_rate(pred, y),
        "expected_value": edge["expected_value"],
        "sharpe": edge["sharpe"],
        "total_log_return": edge["total_log_return"],
        "n": edge["n"],
    }


def _stack_fit_predict(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    return predict_stack_oos(fit_stack(train), test)["stacked"]


def analyze(close: pd.Series, next_date: Optional[pd.Timestamp] = None, cost_bps: float = DEFAULT_ROUND_TRIP_COST_BPS) -> Dict:
    """
    Full analysis of one stock from its daily closes (date-indexed, oldest first).
    Raises ValueError when there is not enough history.
    """
    returns = log_returns(close.dropna())
    data = build_features(returns).dropna()
    if len(data) < MIN_RETURNS:
        raise ValueError(f"Needs at least {MIN_RETURNS} trading days of usable history; found {len(data)}.")

    train, test = time_split(data)
    y_test = test["target"].values

    # Models: fit on train, forecast the test period.
    stack = fit_stack(train)
    oos = predict_stack_oos(stack, test)
    models = [_model_row(name, oos[name], y_test) for name in (*BASE_MODELS, "stacked")]

    # In-sample (labelled as such): the stack scored on the out-of-fold part of the training data it was fit on.
    in_sample_pred = meta_predict(stack.meta, stack.oof_base)
    in_sample = {
        "hit_rate": hit_rate(in_sample_pred, stack.oof_y),
        **{
            k: v
            for k, v in edge_metrics(signal_returns(in_sample_pred, stack.oof_y)).items()
            if k in ("sharpe", "expected_value", "n")
        },
    }

    # Edge of following the stacked forecast on test days.
    strategy = signal_returns(oos["stacked"], y_test)
    edge = edge_metrics(strategy)
    equity, drawdown = equity_and_drawdown(strategy)
    hold_equity, _ = equity_and_drawdown(y_test)

    # AR(1) regime from training data (in-sample estimate of w).
    ar = stack.ar1.params
    regime = regime_label(ar["w"], ar["t_stat"])

    # Validation.
    walk = [walk_forward(data, scheme, _stack_fit_predict) for scheme in ("expanding", "rolling")]
    baseline = random_baseline(y_test, float(strategy.sum()))

    # Latest signal: refit on all data, forecast the next trading day.
    full = fit_stack(data)
    daily_volatility = float(returns.std(ddof=1))
    errors = y_test - oos["stacked"]
    next_forecast, next_index = None, None
    if next_date is not None:
        row = next_day_features(returns, pd.Timestamp(next_date))
        base = np.column_stack([full.ar1.predict(row), full.xgb.predict(row), full.online.predict(row)])
        next_forecast = float(meta_predict(full.meta, base)[0])
        next_index = pd.Timestamp(next_date).date().isoformat()
    latest = next_forecast if next_forecast is not None else float(oos["stacked"][-1])
    strength = describe_strength(signal_strength(latest, daily_volatility))

    # Cost check (handbook, market taking): does the forecast move per day cover a round trip?
    # Ex-ante: average size of the stacked forecast on test days. Ex-post: realised EV per day.
    expected_move_bps = float(np.mean(np.abs(oos["stacked"])) * 1e4)
    latest_move_bps = abs(latest) * 1e4
    ev_bps = float(edge["expected_value"] * 1e4)

    # Caveats in plain words.
    beats_random = baseline["percentile"] >= SIGNIFICANCE_PERCENTILE
    caveats = []
    if len(test) < MIN_TEST_OBS:
        caveats.append(f"Small sample: only {len(test)} test days.")
    if not beats_random:
        caveats.append("No significant edge: the stacked model does not beat 95% of random up/down strategies on test days.")
    if not regime["significant"]:
        caveats.append("The AR(1) weight is not statistically different from zero, so the regime label is weak.")

    last_close = float(close.dropna().iloc[-1])
    return {
        "as_of": pd.Timestamp(close.dropna().index[-1]).date().isoformat(),
        "price": last_close,
        "history_days": int(len(returns)),
        "horizon_days": 1,
        "signal": {
            **strength,
            "daily_volatility": daily_volatility,
            "forecast_log_return": next_forecast,
            "forecast_for": next_index,
            "regime": regime,
        },
        "forecast": {
            "for_date": next_index,
            "expected_price": last_close * float(np.exp(next_forecast)) if next_forecast is not None else None,
            "low": last_close * float(np.exp((next_forecast or 0) - errors.std(ddof=1))),
            "high": last_close * float(np.exp((next_forecast or 0) + errors.std(ddof=1))),
            "band": "±1 standard deviation of test-period forecast errors",
        },
        "edge": {
            **edge,
            "sample": "out_of_sample",
            "equity": equity.tolist(),
            "hold_equity": hold_equity.tolist(),
            "drawdown": drawdown.tolist(),
            "max_drawdown": float(drawdown.min()) if len(drawdown) else 0.0,
            "dates": [d.date().isoformat() for d in test.index],
        },
        "in_sample": {**in_sample, "sample": "in_sample"},
        "risk": risk_context(returns),
        "models": models,
        "meta_weights": dict(zip(BASE_MODELS, (float(w) for w in stack.meta["weights"]))),
        "meta_bias": float(stack.meta["bias"]),
        "validation": {
            "split": {
                "train_days": len(train),
                "test_days": len(test),
                "train_start": train.index[0].date().isoformat(),
                "test_start": test.index[0].date().isoformat(),
                "test_end": test.index[-1].date().isoformat(),
                "shuffled": False,
            },
            "walk_forward": walk,
            "random_baseline": {**baseline, "beats_random": beats_random},
        },
        "cost_check": {
            "round_trip_cost_bps": cost_bps,
            "expected_move_bps": expected_move_bps,
            "latest_move_bps": latest_move_bps,
            "expected_value_bps": ev_bps,
            "covers_cost": expected_move_bps > cost_bps,
        },
        "microstructure": {
            "available": False,
            "reason": "Order-book imbalance and mid-price need level-2 quotes (bids and asks at each price). "
            "Yahoo Finance provides daily bars only, so these measures are not shown.",
        },
        "caveats": caveats,
    }


def history_window(close: pd.Series, days: int = TRADING_DAYS) -> List[Dict]:
    """Last `days` closes for the price chart."""
    tail = close.dropna().tail(days)
    return [{"date": pd.Timestamp(d).date().isoformat(), "close": float(v)} for d, v in tail.items()]
