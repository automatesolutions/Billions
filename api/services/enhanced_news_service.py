"""
Enhanced News Service with Real News Sources and AI Analysis
"""

import os
import logging
import requests
import feedparser
from typing import List, Dict, Optional
from datetime import datetime, timedelta, timezone
import json
import re
from dataclasses import dataclass
import asyncio
import aiohttp
from .advanced_hype_detector import EnhancedNewsAnalyzer

logger = logging.getLogger(__name__)

@dataclass
class NewsArticle:
    title: str
    content: str
    source: str
    url: str
    published_at: datetime
    sentiment_score: float
    hype_score: float
    risk_score: float
    ai_analysis: Dict

class EnhancedNewsService:
    """Enhanced news service with multiple sources and AI analysis"""
    
    def __init__(self):
        self.newsapi_key = os.getenv('NEWS_API_KEY')
        self.alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        self.openai_key = os.getenv('OPENAI_API_KEY')
        self.anthropic_key = os.getenv('ANTHROPIC_API_KEY')
        
        # Initialize advanced analyzer
        self.analyzer = EnhancedNewsAnalyzer()
        
        # RSS feeds for financial news
        self.rss_feeds = [
            'https://feeds.finance.yahoo.com/rss/2.0/headline',
            'https://feeds.marketwatch.com/marketwatch/marketpulse/',
            'https://feeds.bloomberg.com/markets/news.rss',
            'https://feeds.reuters.com/news/wealth',
            'https://feeds.cnn.com/rss/money_latest.rss',
            'https://feeds.nasdaq.com/rss/headlines',
            'https://feeds.fool.com/fool/headlines'
        ]
        
        # NewsAPI sources for financial news
        self.newsapi_sources = [
            'bloomberg', 'reuters', 'financial-times', 'wall-street-journal',
            'marketwatch', 'cnbc', 'yahoo-finance', 'benzinga', 'seeking-alpha'
        ]
    
    async def fetch_real_news(self, ticker: str, limit: int = 10) -> List[NewsArticle]:
        """Fetch real news from multiple sources"""
        articles = []
        
        # Fetch from multiple sources concurrently
        tasks = [
            self._fetch_newsapi_news(ticker, limit),
            self._fetch_rss_news(ticker, limit),
            self._fetch_alpha_vantage_news(ticker, limit),
            self._fetch_web_search_news(ticker, limit)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine and deduplicate articles
        for result in results:
            if isinstance(result, list):
                articles.extend(result)
        
        # Remove duplicates and sort by date
        articles = self._deduplicate_articles(articles)
        articles = sorted(articles, key=lambda x: x.published_at, reverse=True)
        
        return articles[:limit]
    
    async def _fetch_newsapi_news(self, ticker: str, limit: int) -> List[NewsArticle]:
        """Fetch news from NewsAPI"""
        if not self.newsapi_key:
            return []
        
        try:
            url = "https://newsapi.org/v2/everything"
            params = {
                'q': f'"{ticker}" OR "{ticker} stock" OR "{ticker} earnings"',
                'sources': ','.join(self.newsapi_sources),
                'language': 'en',
                'sortBy': 'publishedAt',
                'pageSize': min(limit, 100),
                'apiKey': self.newsapi_key
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        articles = []
                        
                        for item in data.get('articles', []):
                            article = NewsArticle(
                                title=item.get('title', ''),
                                content=item.get('description', ''),
                                source=item.get('source', {}).get('name', 'NewsAPI'),
                                url=item.get('url', ''),
                                published_at=self._ensure_timezone_aware(
                                    datetime.fromisoformat(
                                        item.get('publishedAt', '').replace('Z', '+00:00')
                                    )
                                ),
                                sentiment_score=0.0,
                                hype_score=0.0,
                                risk_score=0.0,
                                ai_analysis={}
                            )
                            articles.append(article)
                        
                        logger.info(f"Fetched {len(articles)} articles from NewsAPI for {ticker}")
                        return articles
                    else:
                        logger.warning(f"NewsAPI request failed: {response.status}")
                        return []
        
        except Exception as e:
            logger.error(f"Error fetching NewsAPI news: {e}")
            return []
    
    async def _fetch_rss_news(self, ticker: str, limit: int) -> List[NewsArticle]:
        """Fetch news from RSS feeds"""
        articles = []
        
        try:
            for feed_url in self.rss_feeds:
                try:
                    feed = feedparser.parse(feed_url)
                    
                    for entry in feed.entries[:5]:  # Limit per feed
                        # Check if article mentions the ticker
                        content = f"{entry.get('title', '')} {entry.get('summary', '')}"
                        if ticker.lower() in content.lower():
                            article = NewsArticle(
                                title=entry.get('title', ''),
                                content=entry.get('summary', ''),
                                source=feed.feed.get('title', 'RSS Feed'),
                                url=entry.get('link', ''),
                                published_at=self._parse_rss_date(entry.get('published', '')),
                                sentiment_score=0.0,
                                hype_score=0.0,
                                risk_score=0.0,
                                ai_analysis={}
                            )
                            articles.append(article)
                
                except Exception as e:
                    logger.warning(f"Error parsing RSS feed {feed_url}: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Error fetching RSS news: {e}")
        
        return articles[:limit]
    
    async def _fetch_alpha_vantage_news(self, ticker: str, limit: int) -> List[NewsArticle]:
        """Fetch news from Alpha Vantage"""
        if not self.alpha_vantage_key:
            return []
        
        try:
            url = "https://www.alphavantage.co/query"
            params = {
                'function': 'NEWS_SENTIMENT',
                'tickers': ticker,
                'limit': min(limit, 50),
                'apikey': self.alpha_vantage_key
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        articles = []
                        
                        for item in data.get('feed', []):
                            article = NewsArticle(
                                title=item.get('title', ''),
                                content=item.get('summary', ''),
                                source=item.get('source', 'Alpha Vantage'),
                                url=item.get('url', ''),
                                published_at=self._ensure_timezone_aware(
                                    datetime.fromisoformat(
                                        item.get('time_published', '').replace('Z', '+00:00')
                                    )
                                ),
                                sentiment_score=float(item.get('overall_sentiment_score', 0)),
                                hype_score=0.0,
                                risk_score=0.0,
                                ai_analysis={}
                            )
                            articles.append(article)
                        
                        logger.info(f"Fetched {len(articles)} articles from Alpha Vantage for {ticker}")
                        return articles
                    else:
                        logger.warning(f"Alpha Vantage request failed: {response.status}")
                        return []
        
        except Exception as e:
            logger.error(f"Error fetching Alpha Vantage news: {e}")
            return []
    
    async def _fetch_web_search_news(self, ticker: str, limit: int) -> List[NewsArticle]:
        """Fetch news using web search (as fallback)"""
        try:
            # This would integrate with a web search API like SerpAPI or Google Custom Search
            # For now, return empty list
            return []
        
        except Exception as e:
            logger.error(f"Error fetching web search news: {e}")
            return []
    
    def _parse_rss_date(self, date_str: str) -> datetime:
        """Parse RSS date string"""
        try:
            # Try different date formats
            formats = [
                '%a, %d %b %Y %H:%M:%S %z',
                '%a, %d %b %Y %H:%M:%S %Z',
                '%Y-%m-%d %H:%M:%S',
                '%Y-%m-%dT%H:%M:%S%z'
            ]
            
            for fmt in formats:
                try:
                    parsed = datetime.strptime(date_str, fmt)
                    # If no timezone info, make it UTC-aware
                    if parsed.tzinfo is None:
                        return parsed.replace(tzinfo=timezone.utc)
                    return parsed
                except ValueError:
                    continue
            
            # Fallback to current time
            return datetime.now(timezone.utc)
        
        except Exception:
            return datetime.now(timezone.utc)
    
    def _ensure_timezone_aware(self, dt: datetime) -> datetime:
        """Ensure datetime is timezone-aware"""
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt
    
    def _deduplicate_articles(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        """Remove duplicate articles based on title similarity"""
        unique_articles = []
        seen_titles = set()
        
        for article in articles:
            # Simple deduplication based on title
            title_key = article.title.lower().strip()
            if title_key not in seen_titles and len(title_key) > 10:
                seen_titles.add(title_key)
                unique_articles.append(article)
        
        return unique_articles
    
    async def analyze_with_ai(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        """Analyze articles using AI for sentiment, hype, and risk"""
        if not articles:
            return articles
        
        logger.info(f"Analyzing {len(articles)} articles with AI")
        
        # Use OpenAI if available, otherwise use local analysis
        if self.openai_key:
            logger.info("Using OpenAI for analysis")
            try:
                return await self._analyze_with_openai(articles)
            except Exception as e:
                logger.error(f"OpenAI analysis failed: {e}")
                logger.info("Falling back to local analysis")
                return await self._analyze_with_local_ai(articles)
        else:
            logger.info("Using local analysis (no OpenAI key)")
            return await self._analyze_with_local_ai(articles)
    
    async def _analyze_with_openai(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        """Analyze articles using OpenAI GPT"""
        try:
            for article in articles:
                prompt = f"""
                Analyze this financial news article for sentiment, hype, and risk:

                Title: {article.title}
                Content: {article.content}

                Provide analysis in JSON format:
                {{
                    "sentiment_score": -1.0 to 1.0 (negative to positive),
                    "hype_score": 0-10 (0=no hype, 10=extreme hype),
                    "risk_score": 0-10 (0=low risk, 10=high risk),
                    "sentiment_label": "positive/negative/neutral",
                    "hype_indicators": ["list", "of", "hype", "words"],
                    "risk_indicators": ["list", "of", "risk", "words"],
                    "summary": "Brief analysis summary"
                }}
                """
                
                headers = {
                    'Authorization': f'Bearer {self.openai_key}',
                    'Content-Type': 'application/json'
                }
                
                data = {
                    'model': 'gpt-3.5-turbo',
                    'messages': [{'role': 'user', 'content': prompt}],
                    'max_tokens': 500,
                    'temperature': 0.3
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        'https://api.openai.com/v1/chat/completions',
                        headers=headers,
                        json=data
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            analysis_text = result['choices'][0]['message']['content']
                            
                            # Parse JSON response
                            try:
                                analysis = json.loads(analysis_text)
                                article.sentiment_score = analysis.get('sentiment_score', 0.0)
                                article.hype_score = analysis.get('hype_score', 0.0)
                                article.risk_score = analysis.get('risk_score', 0.0)
                                article.ai_analysis = analysis
                            except json.JSONDecodeError:
                                logger.warning(f"Failed to parse OpenAI response for article: {article.title}")
                        else:
                            logger.warning(f"OpenAI API request failed: {response.status}")
                            # If OpenAI fails, fall back to local analysis
                            logger.info("Falling back to local analysis due to OpenAI failure")
                            return await self._analyze_with_local_ai(articles)
        
        except Exception as e:
            logger.error(f"Error analyzing with OpenAI: {e}")
            # Fallback to local analysis if OpenAI fails
            return await self._analyze_with_local_ai(articles)
        
        return articles
    
    async def _analyze_with_local_ai(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        """Analyze articles using advanced local AI/ML models"""
        from textblob import TextBlob
        
        logger.info(f"Starting local AI analysis for {len(articles)} articles")
        
        try:
            for i, article in enumerate(articles):
                logger.info(f"Analyzing article {i+1}: {article.title[:50]}...")
                
                # Basic sentiment analysis
                blob = TextBlob(f"{article.title} {article.content}")
                article.sentiment_score = blob.sentiment.polarity
                
                # Advanced hype and risk detection
                analysis = self.analyzer.analyze_article(article.title, article.content)
                
                article.hype_score = analysis['hype']['score']
                article.risk_score = analysis['risk']['score']
                
                logger.info(f"Article {i+1} analysis: Hype={article.hype_score:.1f}, Risk={article.risk_score:.1f}")
                
                # Create comprehensive analysis summary
                article.ai_analysis = {
                    'sentiment_label': 'positive' if article.sentiment_score > 0.1 else 'negative' if article.sentiment_score < -0.1 else 'neutral',
                    'hype_indicators': analysis['hype']['indicators'],
                    'risk_indicators': analysis['risk']['indicators'],
                    'hype_level': analysis['hype']['level'],
                    'risk_level': analysis['risk']['level'],
                    'hype_explanation': analysis['hype']['explanation'],
                    'risk_explanation': analysis['risk']['explanation'],
                    'hype_confidence': analysis['hype']['confidence'],
                    'risk_confidence': analysis['risk']['confidence'],
                    'summary': f"Sentiment: {article.sentiment_score:.2f}, Hype: {analysis['hype']['level']} ({article.hype_score:.1f}), Risk: {analysis['risk']['level']} ({article.risk_score:.1f})"
                }
                
                logger.info(f"Article {i+1} final analysis: {article.ai_analysis['summary']}")
        
        except Exception as e:
            logger.error(f"Error analyzing with advanced local AI: {e}")
            import traceback
            traceback.print_exc()
        
        logger.info("Local AI analysis completed")
        return articles
    
# Old detection methods removed - now using AdvancedHypeDetector
    
    def calculate_overall_metrics(self, articles: List[NewsArticle]) -> Dict:
        """Calculate overall sentiment, hype, and risk metrics"""
        if not articles:
            return {
                'sentiment': {'score': 0.0, 'label': 'neutral'},
                'hype': {'status': 'LOW HYPE', 'score': 0.0, 'count': 0},
                'risk': {'status': 'LOW RISK', 'score': 0.0, 'count': 0}
            }
        
        # Calculate averages
        avg_sentiment = sum(a.sentiment_score for a in articles) / len(articles)
        avg_hype = sum(a.hype_score for a in articles) / len(articles)
        avg_risk = sum(a.risk_score for a in articles) / len(articles)
        
        # Count high hype/risk articles
        hype_count = sum(1 for a in articles if a.hype_score >= 5)
        risk_count = sum(1 for a in articles if a.risk_score >= 5)
        
        # Determine status labels
        sentiment_label = 'positive' if avg_sentiment > 0.1 else 'negative' if avg_sentiment < -0.1 else 'neutral'
        
        if hype_count >= 2 or avg_hype >= 5:
            hype_status = 'HIGH HYPE'
        elif hype_count >= 1 or avg_hype >= 2:
            hype_status = 'MODERATE HYPE'
        else:
            hype_status = 'LOW HYPE'
        
        if risk_count >= 2 or avg_risk >= 5:
            risk_status = 'HIGH RISK'
        elif risk_count >= 1 or avg_risk >= 2:
            risk_status = 'MODERATE RISK'
        else:
            risk_status = 'LOW RISK'
        
        return {
            'sentiment': {
                'score': round(avg_sentiment, 3),
                'label': sentiment_label
            },
            'hype': {
                'status': hype_status,
                'score': round(avg_hype, 2),
                'count': hype_count,
                'total_score': sum(a.hype_score for a in articles)
            },
            'risk': {
                'status': risk_status,
                'score': round(avg_risk, 2),
                'count': risk_count,
                'total_score': sum(a.risk_score for a in articles)
            }
        }
