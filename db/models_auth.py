"""
Authentication and User Management Models
"""

from sqlalchemy import Column, String, Boolean, TIMESTAMP, Integer, Text, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime

from db.core import Base


class User(Base):
    """User model for authentication"""
    
    __tablename__ = "users"
    
    id = Column(String(255), primary_key=True)  # Google OAuth ID
    email = Column(String(255), unique=True, nullable=False, index=True)
    name = Column(String(255))
    image = Column(String(512))
    email_verified = Column(TIMESTAMP(timezone=True))
    created_at = Column(TIMESTAMP(timezone=True), default=datetime.utcnow)
    updated_at = Column(TIMESTAMP(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # User role and status
    role = Column(String(20), default="free")  # free, premium, admin
    is_active = Column(Boolean, default=True)
    
    # Relationships
    preferences = relationship("UserPreference", back_populates="user", uselist=False)
    watchlists = relationship("Watchlist", back_populates="user")
    alerts = relationship("Alert", back_populates="user")


class UserPreference(Base):
    """User preferences and settings"""
    
    __tablename__ = "user_preferences"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(255), ForeignKey("users.id"), unique=True, nullable=False)
    
    # Display preferences
    theme = Column(String(20), default="dark")  # dark, light, system
    language = Column(String(10), default="en")
    
    # Notification preferences
    email_notifications = Column(Boolean, default=True)
    price_alerts = Column(Boolean, default=True)
    outlier_alerts = Column(Boolean, default=True)
    
    # Trading preferences
    default_strategy = Column(String(20), default="swing")  # scalp, swing, longterm
    risk_tolerance = Column(String(20), default="medium")  # low, medium, high
    
    created_at = Column(TIMESTAMP(timezone=True), default=datetime.utcnow)
    updated_at = Column(TIMESTAMP(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship
    user = relationship("User", back_populates="preferences")


class Watchlist(Base):
    """User watchlists for tracking stocks"""
    
    __tablename__ = "watchlists"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(255), ForeignKey("users.id"), nullable=False)
    symbol = Column(String(10), nullable=False)
    name = Column(String(100))
    notes = Column(Text)
    added_at = Column(TIMESTAMP(timezone=True), default=datetime.utcnow)
    
    # Relationship
    user = relationship("User", back_populates="watchlists")
    
    # Ensure unique user-symbol combination
    __table_args__ = (
        {"sqlite_autoincrement": True},
    )


class Alert(Base):
    """Price and event alerts"""
    
    __tablename__ = "alerts"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(255), ForeignKey("users.id"), nullable=False)
    symbol = Column(String(10), nullable=False)
    
    # Alert configuration
    alert_type = Column(String(20), nullable=False)  # price_above, price_below, outlier_detected
    target_value = Column(String(50))  # Price threshold or condition
    
    # Alert status
    is_active = Column(Boolean, default=True)
    triggered_at = Column(TIMESTAMP(timezone=True))
    
    # Metadata
    created_at = Column(TIMESTAMP(timezone=True), default=datetime.utcnow)
    updated_at = Column(TIMESTAMP(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship
    user = relationship("User", back_populates="alerts")

