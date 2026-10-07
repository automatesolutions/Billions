"""
Unit tests for api/services/quant_analysis.py: known answers, leakage guard and validation mechanics.
"""

import numpy as np
import pandas as pd
import pytest

from api.services import quant_analysis as qa


def _dates(n, start="2023-01-02"):
    return pd.bdate_range(start, periods=n)


def _ar1_returns(phi, n=600, sigma=0.01, seed=0):
    rng = np.random.default_rng(seed)
    r = np.zeros(n)
    for t in range(1, n):
        r[t] = phi * r[t - 1] + rng.normal(0, sigma)
    return pd.Series(r, index=_dates(n))


def _prices_from_returns(r: pd.Series, p0=100.0) -> pd.Series:
    close = p0 * np.exp(r.cumsum())
    first = pd.Series([p0], index=[r.index[0] - pd.offsets.BDay(1)])
    return pd.concat([first, close])


# ------------------------------------------------------------------ returns


def test_log_returns_known_answer():
    close = pd.Series([100.0, 110.0, 99.0], index=_dates(3))
    r = qa.log_returns(close)
    assert list(r.index) == list(close.index[1:])
    assert r.iloc[0] == pytest.approx(np.log(1.1))
    assert r.iloc[1] == pytest.approx(np.log(0.9))
    assert r.sum() == pytest.approx(np.log(99 / 100))  # time-additive


# ------------------------------------------------------------------ leakage guard


def test_no_feature_at_t_uses_data_after_t_minus_1():
    """
    For every day t, the feature row built from the full series must equal the row built
    from a series that stops at t-1 (with r_t unknown). If any feature used r_t or later,
    the two would differ.
    """
    r = _ar1_returns(0.2, n=120, seed=3)
    full = qa.build_features(r).drop(columns="target")
    for t in range(1, len(r)):
        truncated = pd.concat([r.iloc[:t], pd.Series([np.nan], index=[r.index[t]])])
        row_without_future = qa.build_features(truncated).drop(columns="target").iloc[-1]
        pd.testing.assert_series_equal(full.iloc[t], row_without_future, check_names=False)


def test_changing_the_future_does_not_change_past_features():
    r = _ar1_returns(0.0, n=200, seed=4)
    t = 150
    shocked = r.copy()
    shocked.iloc[t:] = 0.5  # absurd values from t onward
    a = qa.build_features(r).drop(columns="target").iloc[: t + 1]
    b = qa.build_features(shocked).drop(columns="target").iloc[: t + 1]
    pd.testing.assert_frame_equal(a, b)


def test_rolling_features_are_shifted_before_rolling():
    r = pd.Series(np.arange(1, 31, dtype=float) / 1000, index=_dates(30))
    f = qa.build_features(r)
    # ma_5 at row t is the mean of r[t-5..t-1]
    assert f["ma_5"].iloc[10] == pytest.approx(r.iloc[5:10].mean())
    assert f["lag_1"].iloc[10] == r.iloc[9]
    assert f["dir_1"].iloc[10] == 1.0


# ------------------------------------------------------------------ edge metrics (known answers)


def test_expected_value_known_answer():
    s = [0.02, -0.01, 0.03, -0.02]
    m = qa.edge_metrics(s)
    assert m["win_rate"] == pytest.approx(0.5)
    assert m["avg_win"] == pytest.approx(0.025)
    assert m["avg_loss"] == pytest.approx(0.015)
    # EV = 0.5 * 0.025 - 0.5 * 0.015 = 0.005, which also equals the mean
    assert m["expected_value"] == pytest.approx(0.005)
    assert m["expected_value"] == pytest.approx(np.mean(s))


def test_expected_value_low_win_rate_can_be_positive():
    # Win 1 in 4, but the win is large: EV = 0.25*0.10 - 0.75*0.02 = +0.01
    m = qa.edge_metrics([0.10, -0.02, -0.02, -0.02])
    assert m["win_rate"] == pytest.approx(0.25)
    assert m["expected_value"] == pytest.approx(0.01)


def test_sharpe_known_answer():
    s = np.array([0.01, 0.02, -0.01, 0.00, 0.03])
    expected = s.mean() / s.std(ddof=1) * np.sqrt(252)
    assert qa.edge_metrics(s)["sharpe"] == pytest.approx(expected)
    assert qa.edge_metrics([0.01, 0.01, 0.01])["sharpe"] is np.nan or np.isnan(qa.edge_metrics([0.01] * 3)["sharpe"])


def test_equity_and_drawdown_known_answer():
    equity, dd = qa.equity_and_drawdown(np.log([1.1, 0.5, 2.0]))
    np.testing.assert_allclose(equity, [1.1, 0.55, 1.1])
    np.testing.assert_allclose(dd, [0.0, -0.5, 0.0])
    assert qa.max_drawdown(np.log([0.8])) == pytest.approx(-0.2)


def test_hit_rate_skips_zero_days():
    assert qa.hit_rate([1, -1, 1, 0], [0.01, 0.02, 0.0, 0.03]) == pytest.approx(0.5)


# ------------------------------------------------------------------ AR(1)


@pytest.mark.parametrize("phi,regime", [(-0.35, "mean_reversion"), (0.35, "momentum")])
def test_ar1_recovers_weight_and_regime(phi, regime):
    r = _ar1_returns(phi, n=2000, seed=1)
    fit = qa.fit_ar1(r.shift(1).dropna().values, r.iloc[1:].values)
    assert fit["w"] == pytest.approx(phi, abs=0.05)
    label = qa.regime_label(fit["w"], fit["t_stat"])
    assert label["regime"] == regime and label["significant"]


def test_ar1_on_noise_is_not_significant():
    r = _ar1_returns(0.0, n=500, seed=7)
    fit = qa.fit_ar1(r.shift(1).dropna().values, r.iloc[1:].values)
    assert not qa.regime_label(fit["w"], fit["t_stat"])["significant"]


# ------------------------------------------------------------------ models


def test_pa1_step_matches_sklearn_sgd_pa1():
    from sklearn.linear_model import SGDRegressor

    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 6))
    y = 0.01 * rng.normal(size=300) + 0.002 * X[:, 0]
    sk = SGDRegressor(loss="epsilon_insensitive", penalty=None, learning_rate="pa1", eta0=0.1, epsilon=0.0, random_state=0)
    w, b = np.zeros(6), 0.0
    for i in range(300):
        sk.partial_fit(X[i : i + 1], y[i : i + 1])
        w, b = qa.pa1_step(w, b, X[i], y[i], 0.1)
    np.testing.assert_allclose(w, sk.coef_, atol=1e-12)
    assert b == pytest.approx(sk.intercept_[0], abs=1e-12)


def test_meta_weights_are_non_negative_with_free_bias():
    rng = np.random.default_rng(2)
    good = rng.normal(size=500)
    bad = -good + rng.normal(scale=0.1, size=500)  # anti-correlated base model
    y = 0.003 + 0.8 * good + rng.normal(scale=0.1, size=500)
    meta = qa.fit_meta(np.column_stack([good, bad]), y)
    assert (meta["weights"] >= 0).all()
    assert meta["weights"][1] == pytest.approx(0.0, abs=1e-9)
    assert meta["weights"][0] == pytest.approx(0.8, abs=0.05)
    assert meta["bias"] == pytest.approx(0.003, abs=0.02)


def test_stack_meta_is_fit_on_out_of_fold_predictions_only():
    data = qa.build_features(_ar1_returns(0.3, n=400, seed=5)).dropna()
    fit = qa.fit_stack(data)
    first_block_end = len(data) // 3
    # The meta-learner never sees forecasts for the first block (no earlier data to train on).
    assert len(fit.oof_y) == len(data) - first_block_end
    np.testing.assert_array_equal(fit.oof_y, data["target"].values[first_block_end:])


# ------------------------------------------------------------------ validation


def test_time_split_is_ordered_and_not_shuffled():
    data = qa.build_features(_ar1_returns(0.0, n=400)).dropna()
    train, test = qa.time_split(data)
    assert len(train) == int(len(data) * 0.75)
    assert train.index.max() < test.index.min()
    assert train.index.is_monotonic_increasing and test.index.is_monotonic_increasing


@pytest.mark.parametrize("scheme", ["expanding", "rolling"])
def test_walk_forward_never_trains_on_test_days(scheme):
    data = qa.build_features(_ar1_returns(0.0, n=400)).dropna()
    seen = []

    def spy(train, test):
        assert train.index.max() < test.index.min()
        seen.append((train.index.min(), len(train)))
        return np.zeros(len(test))

    result = qa.walk_forward(data, scheme, spy)
    assert result["n"] == len(data) - int(len(data) * 0.5)
    if scheme == "expanding":
        assert all(start == data.index[0] for start, _ in seen)
        assert [n for _, n in seen] == sorted(n for _, n in seen)
    else:
        assert len({n for _, n in seen}) == 1  # fixed-size window


def test_random_baseline_extremes():
    y = _ar1_returns(0.0, n=200, seed=9).values
    perfect = qa.random_baseline(y, float(np.abs(y).sum()))
    assert perfect["percentile"] == pytest.approx(100.0) and perfect["p_value"] == 0.0
    worst = qa.random_baseline(y, float(-np.abs(y).sum()))
    assert worst["percentile"] == 0.0


# ------------------------------------------------------------------ signal


def test_signal_strength_is_bounded_and_labelled():
    assert qa.signal_strength(0.0, 0.02) == 0.0
    assert -1 < qa.signal_strength(-0.003, 0.02) < 0
    assert qa.signal_strength(0.002, 0.02) == pytest.approx(np.tanh(1.0))
    assert qa.describe_strength(0.05)["direction"] == "neutral"
    assert qa.describe_strength(0.5) == {"strength": 0.5, "direction": "up", "label": "moderate"}
    assert qa.describe_strength(-0.9)["label"] == "strong"


# ------------------------------------------------------------------ full analysis


def test_analyze_requires_enough_history():
    with pytest.raises(ValueError):
        qa.analyze(_prices_from_returns(_ar1_returns(0.0, n=100)))


def test_analyze_end_to_end_on_momentum_series():
    close = _prices_from_returns(_ar1_returns(0.4, n=520, seed=11))
    result = qa.analyze(close, next_date=close.index[-1] + pd.offsets.BDay(1))

    assert [m["model"] for m in result["models"]] == ["ar1", "xgboost", "online", "stacked"]
    assert result["edge"]["sample"] == "out_of_sample" and result["in_sample"]["sample"] == "in_sample"
    split = result["validation"]["split"]
    assert split["shuffled"] is False and split["test_days"] == result["edge"]["n"]
    assert {w["scheme"] for w in result["validation"]["walk_forward"]} == {"expanding", "rolling"}
    assert all(w >= 0 for w in result["meta_weights"].values())
    assert result["microstructure"]["available"] is False
    assert result["signal"]["regime"]["regime"] == "momentum"
    assert -1 < result["signal"]["strength"] < 1
    # A strong AR(1) process is predictable: the stack should beat most random strategies.
    assert result["validation"]["random_baseline"]["percentile"] > 90
    assert len(result["edge"]["equity"]) == len(result["edge"]["dates"]) == split["test_days"]
