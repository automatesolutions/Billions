"""
NASDAQ First-Edge News Service
Specialized service for real-time NASDAQ market news and information
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
from .enhanced_news_service import NewsArticle, EnhancedNewsService

logger = logging.getLogger(__name__)

@dataclass
class NASDAQNewsItem:
    title: str
    content: str
    source: str
    url: str
    published_at: datetime
    category: str  # 'market_data', 'earnings', 'ipo', 'merger', 'regulation', 'technology'
    impact_level: str  # 'high', 'medium', 'low'
    tickers_mentioned: List[str]
    sentiment_score: float
    urgency_score: float  # 0-10, how urgent/immediate this news is

class NASDAQNewsService:
    """Specialized service for first-edge NASDAQ news and market information"""
    
    def __init__(self):
        self.newsapi_key = os.getenv('NEWS_API_KEY')
        self.alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        self.polygon_key = os.getenv('POLYGON_API_KEY')
        self.iex_key = os.getenv('IEX_CLOUD_API_KEY')
        
        # Initialize base news service for AI analysis
        self.base_service = EnhancedNewsService()
        
        # NASDAQ-specific RSS feeds for first-edge information
        self.nasdaq_rss_feeds = [
            'https://feeds.nasdaq.com/rss/headlines',
            'https://feeds.nasdaq.com/rss/marketnews',
            'https://feeds.nasdaq.com/rss/earnings',
            'https://feeds.nasdaq.com/rss/ipos',
            'https://feeds.nasdaq.com/rss/mergers',
            'https://feeds.nasdaq.com/rss/regulatory',
            'https://feeds.finance.yahoo.com/rss/2.0/headline',
            'https://feeds.marketwatch.com/marketwatch/marketpulse/',
            'https://feeds.bloomberg.com/markets/news.rss',
            'https://feeds.reuters.com/news/wealth',
            'https://feeds.cnn.com/rss/money_latest.rss',
            'https://feeds.fool.com/fool/headlines',
            'https://feeds.benzinga.com/benzinga',
            'https://feeds.seekingalpha.com/news',
            'https://feeds.financialtimes.com/us',
            'https://feeds.wsj.com/public/rss/2.0/headlines.xml'
        ]
        
        # NASDAQ-specific keywords for filtering
        self.nasdaq_keywords = [
            'nasdaq', 'nasdaq composite', 'nasdaq 100', 'qqq', 'ndx',
            'technology stocks', 'growth stocks', 'tech earnings',
            'ipo', 'initial public offering', 'merger', 'acquisition',
            'earnings', 'quarterly results', 'guidance', 'outlook',
            'federal reserve', 'interest rates', 'inflation', 'gdp',
            'market volatility', 'trading halt', 'circuit breaker',
            'sec', 'sec filing', 'regulatory', 'compliance'
        ]
        
        # High-impact keywords that indicate urgent news
        self.urgency_keywords = [
            'breaking', 'urgent', 'immediate', 'halt', 'suspended',
            'emergency', 'crisis', 'surge', 'plunge', 'crash',
            'merger', 'acquisition', 'ipo', 'earnings surprise',
            'guidance change', 'sec investigation', 'lawsuit',
            'bankruptcy', 'restructuring', 'layoffs', 'recall'
        ]
    
    async def fetch_nasdaq_news(self, limit: int = 10) -> List[NASDAQNewsItem]:
        """Fetch first-edge NASDAQ news from multiple sources"""
        logger.info(f"Fetching {limit} NASDAQ news items from first-edge sources")
        
        news_items = []
        
        # Fetch from multiple sources concurrently
        tasks = [
            self._fetch_nasdaq_rss_news(limit),
            self._fetch_newsapi_nasdaq(limit),
            self._fetch_alpha_vantage_nasdaq(limit),
            self._fetch_polygon_news(limit),
            self._fetch_iex_news(limit)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine and deduplicate news items
        for result in results:
            if isinstance(result, list):
                news_items.extend(result)
        
        # Remove duplicates and sort by urgency and date
        news_items = self._deduplicate_news(news_items)
        news_items = sorted(news_items, key=lambda x: (x.urgency_score, x.published_at), reverse=True)
        
        logger.info(f"Retrieved {len(news_items)} unique NASDAQ news items")
        return news_items[:limit]
    
    async def _fetch_nasdaq_rss_news(self, limit: int) -> List[NASDAQNewsItem]:
        """Fetch news from NASDAQ-specific RSS feeds"""
        news_items = []
        
        try:
            for feed_url in self.nasdaq_rss_feeds:
                try:
                    feed = feedparser.parse(feed_url)
                    
                    for entry in feed.entries[:3]:  # Limit per feed
                        # Check if article is NASDAQ-relevant
                        content = f"{entry.get('title', '')} {entry.get('summary', '')}"
                        
                        if self._is_nasdaq_relevant(content):
                            news_item = NASDAQNewsItem(
                                title=entry.get('title', ''),
                                content=entry.get('summary', ''),
                                source=feed.feed.get('title', 'RSS Feed'),
                                url=entry.get('link', ''),
                                published_at=self._parse_rss_date(entry.get('published', '')),
                                category=self._categorize_news(content),
                                impact_level=self._assess_impact(content),
                                tickers_mentioned=self._extract_tickers(content),
                                sentiment_score=0.0,
                                urgency_score=self._calculate_urgency(content)
                            )
                            news_items.append(news_item)
                
                except Exception as e:
                    logger.warning(f"Error parsing RSS feed {feed_url}: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Error fetching NASDAQ RSS news: {e}")
        
        return news_items[:limit]
    
    async def _fetch_newsapi_nasdaq(self, limit: int) -> List[NASDAQNewsItem]:
        """Fetch NASDAQ news from NewsAPI"""
        if not self.newsapi_key:
            return []
        
        try:
            url = "https://newsapi.org/v2/everything"
            params = {
                'q': 'NASDAQ OR "nasdaq composite" OR "nasdaq 100" OR "technology stocks" OR "tech earnings"',
                'sources': 'bloomberg,reuters,financial-times,wall-street-journal,marketwatch,cnbc,yahoo-finance,benzinga,seeking-alpha',
                'language': 'en',
                'sortBy': 'publishedAt',
                'pageSize': min(limit, 100),
                'apiKey': self.newsapi_key
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        news_items = []
                        
                        for item in data.get('articles', []):
                            content = f"{item.get('title', '')} {item.get('description', '')}"
                            
                            if self._is_nasdaq_relevant(content):
                                news_item = NASDAQNewsItem(
                                    title=item.get('title', ''),
                                    content=item.get('description', ''),
                                    source=item.get('source', {}).get('name', 'NewsAPI'),
                                    url=item.get('url', ''),
                                    published_at=self._ensure_timezone_aware(
                                        datetime.fromisoformat(
                                            item.get('publishedAt', '').replace('Z', '+00:00')
                                        ) if item.get('publishedAt') else datetime.now(timezone.utc)
                                    ),
                                    category=self._categorize_news(content),
                                    impact_level=self._assess_impact(content),
                                    tickers_mentioned=self._extract_tickers(content),
                                    sentiment_score=0.0,
                                    urgency_score=self._calculate_urgency(content)
                                )
                                news_items.append(news_item)
                        
                        logger.info(f"Fetched {len(news_items)} NASDAQ articles from NewsAPI")
                        return news_items
                    else:
                        logger.warning(f"NewsAPI request failed: {response.status}")
                        return []
        
        except Exception as e:
            logger.error(f"Error fetching NewsAPI NASDAQ news: {e}")
            return []
    
    async def _fetch_alpha_vantage_nasdaq(self, limit: int) -> List[NASDAQNewsItem]:
        """Fetch NASDAQ news from Alpha Vantage"""
        if not self.alpha_vantage_key:
            return []
        
        try:
            # Fetch news for major NASDAQ indices and tech stocks
            nasdaq_tickers = ['QQQ', 'NDX', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA']
            news_items = []
            
            for ticker in nasdaq_tickers[:3]:  # Limit to avoid rate limits
                url = "https://www.alphavantage.co/query"
                params = {
                    'function': 'NEWS_SENTIMENT',
                    'tickers': ticker,
                    'limit': 5,
                    'apikey': self.alpha_vantage_key
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            for item in data.get('feed', []):
                                content = f"{item.get('title', '')} {item.get('summary', '')}"
                                
                                if self._is_nasdaq_relevant(content):
                                    news_item = NASDAQNewsItem(
                                        title=item.get('title', ''),
                                        content=item.get('summary', ''),
                                        source=item.get('source', 'Alpha Vantage'),
                                        url=item.get('url', ''),
                                        published_at=self._parse_alpha_vantage_date(
                                            item.get('time_published', '')
                                        ),
                                        category=self._categorize_news(content),
                                        impact_level=self._assess_impact(content),
                                        tickers_mentioned=self._extract_tickers(content),
                                        sentiment_score=float(item.get('overall_sentiment_score', 0)),
                                        urgency_score=self._calculate_urgency(content)
                                    )
                                    news_items.append(news_item)
            
            logger.info(f"Fetched {len(news_items)} NASDAQ articles from Alpha Vantage")
            return news_items
        
        except Exception as e:
            logger.error(f"Error fetching Alpha Vantage NASDAQ news: {e}")
            return []
    
    async def _fetch_polygon_news(self, limit: int) -> List[NASDAQNewsItem]:
        """Fetch news from Polygon.io (if API key available)"""
        if not self.polygon_key:
            return []
        
        try:
            url = "https://api.polygon.io/v2/reference/news"
            params = {
                'ticker': 'QQQ',  # NASDAQ 100 ETF
                'limit': limit,
                'apikey': self.polygon_key
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        news_items = []
                        
                        for item in data.get('results', []):
                            content = f"{item.get('title', '')} {item.get('description', '')}"
                            
                            if self._is_nasdaq_relevant(content):
                                news_item = NASDAQNewsItem(
                                    title=item.get('title', ''),
                                    content=item.get('description', ''),
                                    source=item.get('publisher', 'Polygon'),
                                    url=item.get('article_url', ''),
                                    published_at=datetime.fromtimestamp(
                                        item.get('published_utc', 0), tz=timezone.utc
                                    ),
                                    category=self._categorize_news(content),
                                    impact_level=self._assess_impact(content),
                                    tickers_mentioned=self._extract_tickers(content),
                                    sentiment_score=0.0,
                                    urgency_score=self._calculate_urgency(content)
                                )
                                news_items.append(news_item)
                        
                        logger.info(f"Fetched {len(news_items)} NASDAQ articles from Polygon")
                        return news_items
                    else:
                        logger.warning(f"Polygon request failed: {response.status}")
                        return []
        
        except Exception as e:
            logger.error(f"Error fetching Polygon NASDAQ news: {e}")
            return []
    
    async def _fetch_iex_news(self, limit: int) -> List[NASDAQNewsItem]:
        """Fetch news from IEX Cloud (if API key available)"""
        if not self.iex_key:
            return []
        
        try:
            url = "https://cloud.iexapis.com/stable/news"
            params = {
                'symbols': 'QQQ,AAPL,MSFT,GOOGL,AMZN,TSLA,META,NVDA',
                'limit': limit,
                'token': self.iex_key
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        news_items = []
                        
                        for item in data:
                            content = f"{item.get('headline', '')} {item.get('summary', '')}"
                            
                            if self._is_nasdaq_relevant(content):
                                news_item = NASDAQNewsItem(
                                    title=item.get('headline', ''),
                                    content=item.get('summary', ''),
                                    source=item.get('source', 'IEX Cloud'),
                                    url=item.get('url', ''),
                                    published_at=datetime.fromtimestamp(
                                        item.get('datetime', 0) / 1000, tz=timezone.utc
                                    ),
                                    category=self._categorize_news(content),
                                    impact_level=self._assess_impact(content),
                                    tickers_mentioned=self._extract_tickers(content),
                                    sentiment_score=0.0,
                                    urgency_score=self._calculate_urgency(content)
                                )
                                news_items.append(news_item)
                        
                        logger.info(f"Fetched {len(news_items)} NASDAQ articles from IEX Cloud")
                        return news_items
                    else:
                        logger.warning(f"IEX Cloud request failed: {response.status}")
                        return []
        
        except Exception as e:
            logger.error(f"Error fetching IEX Cloud NASDAQ news: {e}")
            return []
    
    def _is_nasdaq_relevant(self, content: str) -> bool:
        """Check if content is relevant to NASDAQ"""
        content_lower = content.lower()
        
        # Check for NASDAQ-specific keywords
        for keyword in self.nasdaq_keywords:
            if keyword in content_lower:
                return True
        
        # Check for tech company mentions
        tech_companies = ['apple', 'microsoft', 'google', 'amazon', 'tesla', 'meta', 'nvidia', 'netflix', 'adobe']
        for company in tech_companies:
            if company in content_lower:
                return True
        
        return False
    
    def _categorize_news(self, content: str) -> str:
        """Categorize news based on content"""
        content_lower = content.lower()
        
        if any(word in content_lower for word in ['earnings', 'quarterly', 'revenue', 'profit']):
            return 'earnings'
        elif any(word in content_lower for word in ['ipo', 'initial public offering', 'going public']):
            return 'ipo'
        elif any(word in content_lower for word in ['merger', 'acquisition', 'buyout', 'takeover']):
            return 'merger'
        elif any(word in content_lower for word in ['sec', 'regulatory', 'investigation', 'lawsuit']):
            return 'regulation'
        elif any(word in content_lower for word in ['technology', 'tech', 'innovation', 'ai', 'artificial intelligence']):
            return 'technology'
        else:
            return 'market_data'
    
    def _assess_impact(self, content: str) -> str:
        """Assess the potential market impact"""
        content_lower = content.lower()
        
        high_impact_keywords = ['breaking', 'surge', 'plunge', 'crash', 'merger', 'acquisition', 'earnings surprise', 'guidance change']
        medium_impact_keywords = ['earnings', 'ipo', 'partnership', 'product launch', 'expansion']
        
        if any(word in content_lower for word in high_impact_keywords):
            return 'high'
        elif any(word in content_lower for word in medium_impact_keywords):
            return 'medium'
        else:
            return 'low'
    
    def _extract_tickers(self, content: str) -> List[str]:
        """Extract stock tickers mentioned in content"""
        # Simple regex to find potential tickers (3-5 uppercase letters)
        ticker_pattern = r'\b[A-Z]{3,5}\b'
        potential_tickers = re.findall(ticker_pattern, content)
        
        # Filter out common words that aren't tickers
        common_words = {'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HER', 'WAS', 'ONE', 'OUR', 'HAD', 'BUT', 'WILL', 'NEW', 'NOW', 'MAY', 'GET', 'SEE', 'USE', 'WAY', 'MAY', 'SAY', 'SHE', 'EACH', 'WHICH', 'THEIR', 'TIME', 'WILL', 'ABOUT', 'IF', 'UP', 'OUT', 'MANY', 'THEN', 'THEM', 'THESE', 'SO', 'SOME', 'HER', 'WOULD', 'MAKE', 'LIKE', 'INTO', 'HIM', 'TIME', 'HAS', 'TWO', 'MORE', 'GO', 'NO', 'MY', 'FIRST', 'BEEN', 'CALL', 'WHO', 'ITS', 'NOW', 'FIND', 'LONG', 'DOWN', 'DAY', 'DID', 'GET', 'HAS', 'HAD', 'HIM', 'HIS', 'HOW', 'ITS', 'JUST', 'KNOW', 'LIKE', 'MAKE', 'MANY', 'MORE', 'MOST', 'NEW', 'NOW', 'ONLY', 'OTHER', 'OUR', 'OUT', 'OVER', 'SAID', 'SAME', 'SEE', 'SHE', 'SHOULD', 'SOME', 'STILL', 'SUCH', 'TAKE', 'THAN', 'THAT', 'THEM', 'THEN', 'THERE', 'THESE', 'THEY', 'THIS', 'TIME', 'VERY', 'WAS', 'WAY', 'WELL', 'WERE', 'WHAT', 'WHEN', 'WHERE', 'WHICH', 'WHILE', 'WHO', 'WILL', 'WITH', 'WOULD', 'YOUR'}
        
        tickers = [ticker for ticker in potential_tickers if ticker not in common_words]
        return tickers[:5]  # Limit to 5 tickers
    
    def _calculate_urgency(self, content: str) -> float:
        """Calculate urgency score based on content"""
        content_lower = content.lower()
        urgency_score = 0.0
        
        # Check for urgency keywords
        for keyword in self.urgency_keywords:
            if keyword in content_lower:
                urgency_score += 2.0
        
        # Check for time-sensitive words
        time_words = ['today', 'now', 'immediate', 'urgent', 'breaking', 'live', 'just', 'recent']
        for word in time_words:
            if word in content_lower:
                urgency_score += 1.0
        
        # Check for market-moving words
        market_words = ['surge', 'plunge', 'crash', 'rally', 'halt', 'suspended', 'emergency']
        for word in market_words:
            if word in content_lower:
                urgency_score += 1.5
        
        return min(urgency_score, 10.0)  # Cap at 10
    
    def _ensure_timezone_aware(self, dt: datetime) -> datetime:
        """Ensure datetime is timezone-aware (UTC)"""
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt
    
    def _parse_alpha_vantage_date(self, date_str: str) -> datetime:
        """Parse Alpha Vantage date string"""
        if not date_str:
            return datetime.now(timezone.utc)
        
        try:
            # Alpha Vantage format: "20240101T120000" or "20240101T120000+00:00"
            # Try ISO format first
            date_str_clean = date_str.replace('Z', '+00:00')
            try:
                dt = datetime.fromisoformat(date_str_clean)
                return self._ensure_timezone_aware(dt)
            except (ValueError, AttributeError):
                pass
            
            # Try parsing as YYYYMMDDTHHMMSS format
            try:
                if 'T' in date_str:
                    date_part, time_part = date_str.split('T')
                    if len(date_part) == 8 and len(time_part) >= 6:
                        dt = datetime.strptime(date_str[:15], '%Y%m%dT%H%M%S')
                        return self._ensure_timezone_aware(dt)
            except (ValueError, AttributeError):
                pass
        except Exception:
            pass
        
        return datetime.now(timezone.utc)
    
    def _parse_rss_date(self, date_str: str) -> datetime:
        """Parse RSS date string"""
        try:
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
                    return self._ensure_timezone_aware(parsed)
                except ValueError:
                    continue
            
            # Return timezone-aware datetime.now()
            return datetime.now(timezone.utc)
        
        except Exception:
            return datetime.now(timezone.utc)
    
    def _deduplicate_news(self, news_items: List[NASDAQNewsItem]) -> List[NASDAQNewsItem]:
        """Remove duplicate news items"""
        unique_items = []
        seen_titles = set()
        
        for item in news_items:
            title_key = item.title.lower().strip()
            if title_key not in seen_titles and len(title_key) > 10:
                seen_titles.add(title_key)
                unique_items.append(item)
        
        return unique_items
    
    async def analyze_nasdaq_news(self, news_items: List[NASDAQNewsItem]) -> List[NASDAQNewsItem]:
        """Analyze NASDAQ news items with AI"""
        if not news_items:
            return news_items
        
        logger.info(f"Analyzing {len(news_items)} NASDAQ news items with AI")
        
        # Convert to base NewsArticle format for analysis
        articles = []
        for item in news_items:
            article = NewsArticle(
                title=item.title,
                content=item.content,
                source=item.source,
                url=item.url,
                published_at=item.published_at,
                sentiment_score=item.sentiment_score,
                hype_score=0.0,
                risk_score=0.0,
                ai_analysis={}
            )
            articles.append(article)
        
        # Use base service for AI analysis
        analyzed_articles = await self.base_service.analyze_with_ai(articles)
        
        # Update NASDAQ news items with analysis results
        for i, analyzed_article in enumerate(analyzed_articles):
            if i < len(news_items):
                news_items[i].sentiment_score = analyzed_article.sentiment_score
        
        return news_items
