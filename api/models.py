from datetime import datetime, timezone

from sqlalchemy import JSON, TIMESTAMP, Boolean, Column, Date, DateTime, Float, Integer, String, Text, UniqueConstraint

from api.database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class PerfMetric(Base):
    """One row per (strategy, symbol): the returns and z-scores behind the outlier scatter."""

    __tablename__ = "performance_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    strategy = Column(String(16), index=True)  # scalp, swing, longterm
    symbol = Column(String(10), index=True)
    metric_x = Column(Float)  # % return over the longer window
    metric_y = Column(Float)  # % return over the shorter window
    z_x = Column(Float)
    z_y = Column(Float)
    is_outlier = Column(Boolean)
    price_date = Column(Date)  # date of the last close used
    inserted = Column(TIMESTAMP(timezone=True), default=_utcnow)


class QuantSignal(Base):
    """
    One signal per (ticker, interval, bar): recorded when the bar closes, scored when the next bar closes.
    This is the live record behind "Is it working?". Times are New York wall-clock bar starts.
    """

    __tablename__ = "quant_signals"
    __table_args__ = (UniqueConstraint("ticker", "interval", "bar_time", name="uq_quant_signal_bar"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(10), index=True)
    interval = Column(String(4), index=True)
    bar_time = Column(DateTime)  # the bar whose close the signal was made at
    target_time = Column(DateTime)  # the bar the signal forecasts
    close = Column(Float)
    forecast = Column(Float)  # stacked forecast of the next bar's log return
    strength = Column(Float)
    action = Column(String(8))  # long, short, flat
    position = Column(Float)  # -1..1 share of max position; 0 when gated
    cost_bps = Column(Float)  # round-trip cost assumed when the signal was made
    realized = Column(Float, nullable=True)  # log return of the target bar
    void = Column(Boolean, default=False)  # the target bar never arrived (data gap)
    created_at = Column(TIMESTAMP(timezone=True), default=_utcnow)


class QuantWatch(Base):
    """A (ticker, interval) pair the scheduler keeps recording signals for, even with no page open."""

    __tablename__ = "quant_watch"
    __table_args__ = (UniqueConstraint("ticker", "interval", name="uq_quant_watch"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(10))
    interval = Column(String(4))
    created_at = Column(TIMESTAMP(timezone=True), default=_utcnow)


def utcnow_naive() -> datetime:
    """UTC without tzinfo: SQLite drops time zones, so bot and research tables store naive UTC throughout."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class BotDecision(Base):
    """Every decision the paper bot makes, with its reason: orders placed, blocked, skipped, and holds."""

    __tablename__ = "bot_decisions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=utcnow_naive, index=True)
    strategy = Column(String(24))
    symbol = Column(String(10), nullable=True)
    action = Column(String(12))  # buy, sell, hold, skip, start, stop, kill, error, info
    status = Column(String(12), index=True)  # placed, blocked, skipped, held, failed, info
    reason = Column(Text)
    price = Column(Float, nullable=True)
    qty = Column(Float, nullable=True)
    order_id = Column(String(64), nullable=True)
    details = Column(JSON, nullable=True)


class BotTrade(Base):
    """One lead-lag round trip, with the timestamps needed to see whether the edge survives latency."""

    __tablename__ = "bot_trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=utcnow_naive, index=True)
    leader = Column(String(10))
    follower = Column(String(10), index=True)
    side = Column(String(4))  # buy (long) or sell (short)
    qty = Column(Float)
    status = Column(String(12))  # pending, open, closing, closed, failed
    leader_move_pct = Column(Float)
    leader_move_started_at = Column(DateTime)
    leader_move_at = Column(DateTime)  # exchange time of the leader trade that crossed the threshold
    leader_price = Column(Float)
    follower_price_at_signal = Column(Float)
    follower_move_pct = Column(Float)
    signal_at = Column(DateTime)
    entry_reason = Column(Text)
    entry_order_id = Column(String(64), nullable=True)
    entry_submitted_at = Column(DateTime, nullable=True)
    entry_filled_at = Column(DateTime, nullable=True)
    entry_price = Column(Float, nullable=True)
    exit_reason = Column(Text, nullable=True)
    exit_signal_at = Column(DateTime, nullable=True)
    exit_order_id = Column(String(64), nullable=True)
    exit_filled_at = Column(DateTime, nullable=True)
    exit_price = Column(Float, nullable=True)
    pnl_pct = Column(Float, nullable=True)


class LeadLagResult(Base):
    """One research result per pair per day. `payload` holds the full analysis; the columns are for sorting."""

    __tablename__ = "lead_lag_results"
    __table_args__ = (UniqueConstraint("run_date", "pair_key", name="uq_lead_lag_result_day"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_date = Column(Date, index=True)
    pair_key = Column(String(24), index=True)  # the two tickers, sorted, joined with "|"
    leader = Column(String(10))
    follower = Column(String(10))
    trigger = Column(String(8))  # daily or manual
    source = Column(String(8))
    confidence = Column(String(16))
    accuracy = Column(Float, nullable=True)
    events = Column(Integer)
    oos_accuracy = Column(Float, nullable=True)
    oos_events = Column(Integer)
    lag_seconds = Column(Float, nullable=True)
    avg_follower_move_pct = Column(Float, nullable=True)
    payload = Column(JSON)
    created_at = Column(DateTime, default=utcnow_naive)


class LeadLagWatch(Base):
    """A pair you asked the daily research job to rescan."""

    __tablename__ = "lead_lag_watch"
    __table_args__ = (UniqueConstraint("pair_key", name="uq_lead_lag_watch"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    pair_key = Column(String(24))
    ticker_a = Column(String(10))
    ticker_b = Column(String(10))
    created_at = Column(DateTime, default=utcnow_naive)
