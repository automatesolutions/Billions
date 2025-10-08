from sqlalchemy import Column, BigInteger, String, Numeric, Boolean, TIMESTAMP, Integer
from datetime import datetime

from db.core import Base

class PerfMetric(Base):
    """Table holding performance metrics for outlier visualisation."""

    __tablename__ = "performance_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    strategy = Column(String(16), index=True)  # scalp, swing, longterm, crypto
    symbol = Column(String(10), index=True)
    metric_x = Column(Numeric)
    metric_y = Column(Numeric)
    z_x = Column(Numeric)
    z_y = Column(Numeric)
    is_outlier = Column(Boolean)
    inserted = Column(TIMESTAMP(timezone=True), default=datetime.utcnow)
