from datetime import datetime, timezone

from sqlalchemy import TIMESTAMP, Boolean, Column, Date, DateTime, Float, Integer, String, UniqueConstraint

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
