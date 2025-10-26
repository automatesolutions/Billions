"""
NASDAQ News API endpoints
Specialized endpoints for first-edge NASDAQ market news
"""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Dict
import logging
import asyncio
from datetime import datetime, timezone
import os

# Import the NASDAQ news service
from ..services.nasdaq_news_service import NASDAQNewsService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/nasdaq-news", tags=["NASDAQ News"])


@router.get("/latest")
async def get_latest_nasdaq_news(
    limit: int = Query(default=10, ge=1, le=50, description="Number of news items")
):
    """
    Get the latest first-edge NASDAQ news and market information
    
    - **limit**: Number of news items (1-50)
    
    Features:
    - Real-time NASDAQ news from multiple sources
    - First-edge market information
    - AI-powered sentiment analysis
    - Urgency scoring for immediate impact
    - Categorized by type (earnings, IPO, merger, etc.)
    """
    try:
        logger.info(f"Fetching {limit} latest NASDAQ news items")
        
        # Initialize NASDAQ news service
        nasdaq_service = NASDAQNewsService()
        
        # Fetch latest NASDAQ news
        news_items = await nasdaq_service.fetch_nasdaq_news(limit)
        
        # If no news found, provide informative message
        if not news_items:
            logger.warning("No NASDAQ news found")
            return {
                "news_count": 0,
                "news_items": [],
                "message": "No recent NASDAQ news found. This could be due to: market hours, API rate limits, or limited news coverage during off-hours.",
                "last_updated": datetime.now(timezone.utc).isoformat()
            }
        
        # Analyze news with AI
        analyzed_news = await nasdaq_service.analyze_nasdaq_news(news_items)
        
        # Convert to API response format
        processed_news = []
        for item in analyzed_news:
            processed_item = {
                "title": item.title,
                "content": item.content,
                "source": item.source,
                "url": item.url,
                "published_at": item.published_at.isoformat(),
                "category": item.category,
                "impact_level": item.impact_level,
                "tickers_mentioned": item.tickers_mentioned,
                "sentiment_score": item.sentiment_score,
                "urgency_score": item.urgency_score,
                "time_ago": _get_time_ago(item.published_at)
            }
            processed_news.append(processed_item)
        
        # Calculate overall metrics
        overall_metrics = _calculate_nasdaq_metrics(analyzed_news)
        
        logger.info(f"Successfully processed {len(processed_news)} NASDAQ news items")
        
        return {
            "news_count": len(processed_news),
            "news_items": processed_news,
            "overall_metrics": overall_metrics,
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "sources_checked": [
                "NASDAQ RSS Feeds",
                "NewsAPI",
                "Alpha Vantage",
                "Polygon.io",
                "IEX Cloud"
            ]
        }
    
    except Exception as e:
        logger.error(f"Error fetching NASDAQ news: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch NASDAQ news: {str(e)}")


@router.get("/urgent")
async def get_urgent_nasdaq_news(
    limit: int = Query(default=5, ge=1, le=20, description="Number of urgent news items")
):
    """
    Get urgent/high-impact NASDAQ news only
    
    - **limit**: Number of urgent news items (1-20)
    
    Returns only news items with high urgency scores and impact levels
    """
    try:
        logger.info(f"Fetching {limit} urgent NASDAQ news items")
        
        # Initialize NASDAQ news service
        nasdaq_service = NASDAQNewsService()
        
        # Fetch more news to filter for urgent items
        all_news = await nasdaq_service.fetch_nasdaq_news(limit * 3)
        
        # Filter for urgent/high-impact news
        urgent_news = [
            item for item in all_news 
            if item.urgency_score >= 5.0 or item.impact_level == 'high'
        ]
        
        # Sort by urgency and take top items
        urgent_news = sorted(urgent_news, key=lambda x: x.urgency_score, reverse=True)
        urgent_news = urgent_news[:limit]
        
        # If no urgent news found, return regular news
        if not urgent_news:
            urgent_news = all_news[:limit]
        
        # Analyze news with AI
        analyzed_news = await nasdaq_service.analyze_nasdaq_news(urgent_news)
        
        # Convert to API response format
        processed_news = []
        for item in analyzed_news:
            processed_item = {
                "title": item.title,
                "content": item.content,
                "source": item.source,
                "url": item.url,
                "published_at": item.published_at.isoformat(),
                "category": item.category,
                "impact_level": item.impact_level,
                "tickers_mentioned": item.tickers_mentioned,
                "sentiment_score": item.sentiment_score,
                "urgency_score": item.urgency_score,
                "time_ago": _get_time_ago(item.published_at),
                "is_urgent": item.urgency_score >= 5.0
            }
            processed_news.append(processed_item)
        
        logger.info(f"Successfully processed {len(processed_news)} urgent NASDAQ news items")
        
        return {
            "news_count": len(processed_news),
            "news_items": processed_news,
            "urgent_count": len([item for item in processed_news if item["is_urgent"]]),
            "last_updated": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error fetching urgent NASDAQ news: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch urgent NASDAQ news: {str(e)}")


@router.get("/category/{category}")
async def get_nasdaq_news_by_category(
    category: str,
    limit: int = Query(default=10, ge=1, le=30, description="Number of news items")
):
    """
    Get NASDAQ news filtered by category
    
    - **category**: News category (earnings, ipo, merger, regulation, technology, market_data)
    - **limit**: Number of news items (1-30)
    
    Available categories:
    - earnings: Quarterly results, guidance updates
    - ipo: Initial public offerings
    - merger: Mergers and acquisitions
    - regulation: SEC filings, regulatory news
    - technology: Tech sector news
    - market_data: General market information
    """
    valid_categories = ['earnings', 'ipo', 'merger', 'regulation', 'technology', 'market_data']
    
    if category not in valid_categories:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid category. Must be one of: {', '.join(valid_categories)}"
        )
    
    try:
        logger.info(f"Fetching {limit} NASDAQ news items for category: {category}")
        
        # Initialize NASDAQ news service
        nasdaq_service = NASDAQNewsService()
        
        # Fetch news and filter by category
        all_news = await nasdaq_service.fetch_nasdaq_news(limit * 2)
        category_news = [item for item in all_news if item.category == category]
        category_news = category_news[:limit]
        
        # If no category-specific news found, return general news
        if not category_news:
            category_news = all_news[:limit]
        
        # Analyze news with AI
        analyzed_news = await nasdaq_service.analyze_nasdaq_news(category_news)
        
        # Convert to API response format
        processed_news = []
        for item in analyzed_news:
            processed_item = {
                "title": item.title,
                "content": item.content,
                "source": item.source,
                "url": item.url,
                "published_at": item.published_at.isoformat(),
                "category": item.category,
                "impact_level": item.impact_level,
                "tickers_mentioned": item.tickers_mentioned,
                "sentiment_score": item.sentiment_score,
                "urgency_score": item.urgency_score,
                "time_ago": _get_time_ago(item.published_at)
            }
            processed_news.append(processed_item)
        
        logger.info(f"Successfully processed {len(processed_news)} {category} NASDAQ news items")
        
        return {
            "category": category,
            "news_count": len(processed_news),
            "news_items": processed_news,
            "last_updated": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error fetching NASDAQ news for category {category}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch NASDAQ news for category {category}: {str(e)}")


def _get_time_ago(published_at: datetime) -> str:
    """Calculate time ago string"""
    now = datetime.now(timezone.utc)
    diff = now - published_at
    
    if diff.days > 0:
        return f"{diff.days} day{'s' if diff.days != 1 else ''} ago"
    elif diff.seconds > 3600:
        hours = diff.seconds // 3600
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    elif diff.seconds > 60:
        minutes = diff.seconds // 60
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
    else:
        return "Just now"


def _calculate_nasdaq_metrics(news_items: List) -> Dict:
    """Calculate overall NASDAQ news metrics"""
    if not news_items:
        return {
            "overall_sentiment": {"score": 0.0, "label": "neutral"},
            "average_urgency": 0.0,
            "high_impact_count": 0,
            "categories": {},
            "top_tickers": []
        }
    
    # Calculate average sentiment
    avg_sentiment = sum(item.sentiment_score for item in news_items) / len(news_items)
    sentiment_label = 'positive' if avg_sentiment > 0.1 else 'negative' if avg_sentiment < -0.1 else 'neutral'
    
    # Calculate average urgency
    avg_urgency = sum(item.urgency_score for item in news_items) / len(news_items)
    
    # Count high impact news
    high_impact_count = len([item for item in news_items if item.impact_level == 'high'])
    
    # Count by category
    categories = {}
    for item in news_items:
        categories[item.category] = categories.get(item.category, 0) + 1
    
    # Get top mentioned tickers
    all_tickers = []
    for item in news_items:
        all_tickers.extend(item.tickers_mentioned)
    
    ticker_counts = {}
    for ticker in all_tickers:
        ticker_counts[ticker] = ticker_counts.get(ticker, 0) + 1
    
    top_tickers = sorted(ticker_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    
    return {
        "overall_sentiment": {
            "score": round(avg_sentiment, 3),
            "label": sentiment_label
        },
        "average_urgency": round(avg_urgency, 2),
        "high_impact_count": high_impact_count,
        "categories": categories,
        "top_tickers": [{"ticker": ticker, "mentions": count} for ticker, count in top_tickers]
    }
