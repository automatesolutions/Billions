from datetime import datetime, timezone

from sqlalchemy import TIMESTAMP, Boolean, Column, Date, Float, Integer, String

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
