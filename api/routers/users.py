"""
User management endpoints
"""

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from typing import Optional
from pydantic import BaseModel, EmailStr
from api.database import get_db
from db.models_auth import User, UserPreference, Watchlist, Alert
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/users", tags=["Users"])


# Pydantic models for request/response
class UserCreate(BaseModel):
    id: str
    email: EmailStr
    name: Optional[str] = None
    image: Optional[str] = None


class UserResponse(BaseModel):
    id: str
    email: str
    name: Optional[str]
    image: Optional[str]
    role: str
    is_active: bool
    
    class Config:
        from_attributes = True


class PreferenceUpdate(BaseModel):
    theme: Optional[str] = None
    language: Optional[str] = None
    email_notifications: Optional[bool] = None
    price_alerts: Optional[bool] = None
    outlier_alerts: Optional[bool] = None
    default_strategy: Optional[str] = None
    risk_tolerance: Optional[str] = None


@router.post("/", response_model=UserResponse)
async def create_or_update_user(
    user_data: UserCreate,
    db: Session = Depends(get_db)
):
    """
    Create a new user or update existing user (OAuth callback)
    """
    try:
        # Check if user exists
        user = db.query(User).filter(User.id == user_data.id).first()
        
        if user:
            # Update existing user
            user.name = user_data.name
            user.image = user_data.image
        else:
            # Create new user
            user = User(
                id=user_data.id,
                email=user_data.email,
                name=user_data.name,
                image=user_data.image,
            )
            db.add(user)
            
            # Create default preferences
            preferences = UserPreference(user_id=user.id)
            db.add(preferences)
        
        db.commit()
        db.refresh(user)
        
        return user
        
    except Exception as e:
        logger.error(f"Error creating/updating user: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: str,
    db: Session = Depends(get_db)
):
    """Get user by ID"""
    user = db.query(User).filter(User.id == user_id).first()
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return user


@router.get("/{user_id}/preferences")
async def get_user_preferences(
    user_id: str,
    db: Session = Depends(get_db)
):
    """Get user preferences"""
    preferences = db.query(UserPreference).filter(
        UserPreference.user_id == user_id
    ).first()
    
    if not preferences:
        raise HTTPException(status_code=404, detail="Preferences not found")
    
    return {
        "theme": preferences.theme,
        "language": preferences.language,
        "email_notifications": preferences.email_notifications,
        "price_alerts": preferences.price_alerts,
        "outlier_alerts": preferences.outlier_alerts,
        "default_strategy": preferences.default_strategy,
        "risk_tolerance": preferences.risk_tolerance,
    }


@router.put("/{user_id}/preferences")
async def update_user_preferences(
    user_id: str,
    updates: PreferenceUpdate,
    db: Session = Depends(get_db)
):
    """Update user preferences"""
    preferences = db.query(UserPreference).filter(
        UserPreference.user_id == user_id
    ).first()
    
    if not preferences:
        raise HTTPException(status_code=404, detail="Preferences not found")
    
    # Update only provided fields
    update_data = updates.dict(exclude_unset=True)
    for key, value in update_data.items():
        setattr(preferences, key, value)
    
    try:
        db.commit()
        db.refresh(preferences)
        return {"message": "Preferences updated successfully"}
    except Exception as e:
        logger.error(f"Error updating preferences: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{user_id}/watchlist")
async def get_watchlist(
    user_id: str,
    db: Session = Depends(get_db)
):
    """Get user's watchlist"""
    watchlist = db.query(Watchlist).filter(
        Watchlist.user_id == user_id
    ).all()
    
    return [
        {
            "id": item.id,
            "symbol": item.symbol,
            "name": item.name,
            "notes": item.notes,
            "added_at": item.added_at.isoformat() if item.added_at else None,
        }
        for item in watchlist
    ]


@router.post("/{user_id}/watchlist")
async def add_to_watchlist(
    user_id: str,
    symbol: str,
    name: Optional[str] = None,
    notes: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Add symbol to watchlist"""
    # Check if already exists
    existing = db.query(Watchlist).filter(
        Watchlist.user_id == user_id,
        Watchlist.symbol == symbol
    ).first()
    
    if existing:
        raise HTTPException(status_code=400, detail="Symbol already in watchlist")
    
    try:
        item = Watchlist(
            user_id=user_id,
            symbol=symbol.upper(),
            name=name,
            notes=notes
        )
        db.add(item)
        db.commit()
        db.refresh(item)
        
        return {"message": "Added to watchlist", "id": item.id}
    except Exception as e:
        logger.error(f"Error adding to watchlist: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{user_id}/watchlist/{item_id}")
async def remove_from_watchlist(
    user_id: str,
    item_id: int,
    db: Session = Depends(get_db)
):
    """Remove symbol from watchlist"""
    item = db.query(Watchlist).filter(
        Watchlist.id == item_id,
        Watchlist.user_id == user_id
    ).first()
    
    if not item:
        raise HTTPException(status_code=404, detail="Watchlist item not found")
    
    try:
        db.delete(item)
        db.commit()
        return {"message": "Removed from watchlist"}
    except Exception as e:
        logger.error(f"Error removing from watchlist: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

