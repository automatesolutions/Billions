"""
Enhanced News & Sentiment Analysis endpoints with Real News Sources and AI Analysis
"""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Dict
import logging
import asyncio
from datetime import datetime
import os

# Import the enhanced news service
from ..services.enhanced_news_service import EnhancedNewsService

logger = logging.getLogger(__name__)

# Legacy functions removed - now using EnhancedNewsService
router = APIRouter(prefix="/news", tags=["News & Sentiment"])


@router.get("/{ticker}")
async def get_news(
    ticker: str,
    limit: int = Query(default=10, ge=1, le=50, description="Number of articles")
):
    """
    Get REAL news articles for a ticker with AI-powered sentiment, hype, and risk analysis
    
    - **ticker**: Stock symbol
    - **limit**: Number of articles (1-50)
    
    Features:
    - Real news from multiple sources (NewsAPI, RSS feeds, Alpha Vantage)
    - AI-powered sentiment analysis (OpenAI GPT or local analysis)
    - Advanced HYPE detection
    - Caveat Emptor risk analysis
    """
    ticker = ticker.upper()
    
    try:
        logger.info(f"Fetching REAL news for {ticker} with AI analysis")
        
        # Initialize enhanced news service
        news_service = EnhancedNewsService()
        
        # Fetch real news from multiple sources
        articles = await news_service.fetch_real_news(ticker, limit)
        
        # If no real news found, provide informative message instead of fake data
        if not articles:
            logger.warning(f"No real news found for {ticker}")
            return {
                "ticker": ticker,
                "news_count": 0,
                "articles": [],
                "message": f"No recent news found for {ticker}. This could be due to: limited news coverage, API rate limits, or the ticker not being actively covered by financial news sources.",
                "overall_sentiment": {
                    "polarity": 0.0,
                    "label": "neutral"
                },
                "hype_analysis": {
                    "overall_status": "NO DATA",
                    "hype_articles_count": 0,
                    "average_hype_score": 0.0,
                    "total_hype_score": 0
                },
                "caveat_emptor": {
                    "overall_status": "NO DATA",
                    "risky_articles_count": 0,
                    "average_risk_score": 0.0,
                    "total_risk_score": 0
                }
            }
        
        # Analyze articles with AI
        analyzed_articles = await news_service.analyze_with_ai(articles)
        
        # Calculate overall metrics
        overall_metrics = news_service.calculate_overall_metrics(analyzed_articles)
        
        # Convert to API response format
        processed_news = []
        for article in analyzed_articles:
            processed_news.append({
                "title": article.title,
                "publisher": article.source,
                "link": article.url,
                "published_at": article.published_at.isoformat(),
                "sentiment": {
                    "polarity": round(article.sentiment_score, 3),
                    "subjectivity": 0.5,  # Could be enhanced
                    "label": article.ai_analysis.get('sentiment_label', 'neutral')
                },
                "hype_analysis": {
                    "is_hype": article.hype_score >= 5,
                    "hype_score": round(article.hype_score, 2),
                    "indicators": article.ai_analysis.get('hype_indicators', [])
                },
                "caveat_emptor": {
                    "is_risky": article.risk_score >= 5,
                    "risk_score": round(article.risk_score, 2),
                    "warnings": article.ai_analysis.get('risk_indicators', [])
                },
                "ai_summary": article.ai_analysis.get('summary', '')
            })
        
        logger.info(f"Successfully processed {len(processed_news)} real news articles for {ticker}")
        
        return {
            "ticker": ticker,
            "news_count": len(processed_news),
            "articles": processed_news,
            "overall_sentiment": overall_metrics['sentiment'],
            "hype_analysis": {
                "overall_status": overall_metrics['hype']['status'],
                "hype_articles_count": overall_metrics['hype']['count'],
                "average_hype_score": overall_metrics['hype']['score'],
                "total_hype_score": overall_metrics['hype']['total_score']
            },
            "caveat_emptor": {
                "overall_status": overall_metrics['risk']['status'],
                "risky_articles_count": overall_metrics['risk']['count'],
                "average_risk_score": overall_metrics['risk']['score'],
                "total_risk_score": overall_metrics['risk']['total_score']
            },
            "data_source": "Real News APIs + AI Analysis",
            "last_updated": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error fetching real news for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching news: {str(e)}")

