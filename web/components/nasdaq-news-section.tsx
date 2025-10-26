'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ExternalLink, TrendingUp, TrendingDown, AlertTriangle, Clock, Zap } from "lucide-react";
import { api } from "@/lib/api";

interface NASDAQNewsItem {
  title: string;
  content: string;
  source: string;
  url: string;
  published_at: string;
  category: string;
  impact_level: string;
  tickers_mentioned: string[];
  sentiment_score: number;
  urgency_score: number;
  time_ago: string;
  is_urgent?: boolean;
}

interface NASDAQNewsResponse {
  news_count: number;
  news_items: NASDAQNewsItem[];
  overall_metrics: {
    overall_sentiment: {
      score: number;
      label: string;
    };
    average_urgency: number;
    high_impact_count: number;
    categories: Record<string, number>;
    top_tickers: Array<{ ticker: string; mentions: number }>;
  };
  last_updated: string;
}

export function NASDAQNewsSection() {
  const [newsData, setNewsData] = useState<NASDAQNewsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showUrgentOnly, setShowUrgentOnly] = useState(false);

  const fetchNASDAQNews = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const data = showUrgentOnly 
        ? await api.getUrgentNASDAQNews(10)
        : await api.getNASDAQNews(10);
      setNewsData(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch NASDAQ news');
      console.error('Error fetching NASDAQ news:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchNASDAQNews();
    
    // Refresh every 5 minutes
    const interval = setInterval(fetchNASDAQNews, 5 * 60 * 1000);
    return () => clearInterval(interval);
  }, [showUrgentOnly]);

  const getCategoryIcon = (category: string) => {
    switch (category) {
      case 'earnings': return '📊';
      case 'ipo': return '🚀';
      case 'merger': return '🤝';
      case 'regulation': return '⚖️';
      case 'technology': return '💻';
      default: return '📈';
    }
  };

  const getImpactColor = (impact: string) => {
    switch (impact) {
      case 'high': return 'bg-red-100 text-red-800 border-red-200';
      case 'medium': return 'bg-yellow-100 text-yellow-800 border-yellow-200';
      case 'low': return 'bg-green-100 text-green-800 border-green-200';
      default: return 'bg-gray-100 text-gray-800 border-gray-200';
    }
  };

  const getSentimentColor = (score: number) => {
    if (score > 0.1) return 'text-green-600';
    if (score < -0.1) return 'text-red-600';
    return 'text-gray-600';
  };

  const getUrgencyIcon = (score: number) => {
    if (score >= 7) return <AlertTriangle className="h-4 w-4 text-red-500" />;
    if (score >= 4) return <Zap className="h-4 w-4 text-yellow-500" />;
    return <Clock className="h-4 w-4 text-gray-500" />;
  };

  if (loading && !newsData) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="text-sm font-medium flex items-center gap-2">
            📈 NASDAQ First-Edge News
            <Badge variant="outline" className="text-xs">Loading...</Badge>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {[1, 2, 3].map((i) => (
              <div key={i} className="animate-pulse">
                <div className="h-4 bg-gray-200 rounded w-3/4 mb-2"></div>
                <div className="h-3 bg-gray-200 rounded w-1/2"></div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="text-sm font-medium flex items-center gap-2">
            📈 NASDAQ First-Edge News
            <Badge variant="destructive" className="text-xs">Error</Badge>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-4">
            <p className="text-sm text-muted-foreground mb-3">{error}</p>
            <Button onClick={fetchNASDAQNews} variant="outline" size="sm">
              Retry
            </Button>
          </div>
        </CardContent>
      </Card>
    );
  }

  const newsItems = newsData?.news_items || [];
  const metrics = newsData?.overall_metrics;

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm font-medium flex items-center gap-2">
            📈 NASDAQ First-Edge News
            <Badge variant="outline" className="text-xs">
              {newsData?.news_count || 0} items
            </Badge>
            {metrics && metrics.high_impact_count > 0 && (
              <Badge variant="destructive" className="text-xs">
                {metrics.high_impact_count} High Impact
              </Badge>
            )}
          </CardTitle>
          <div className="flex gap-2">
            <Button
              variant={showUrgentOnly ? "default" : "outline"}
              size="sm"
              onClick={() => setShowUrgentOnly(!showUrgentOnly)}
              className="text-xs"
            >
              {showUrgentOnly ? "All News" : "Urgent Only"}
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={fetchNASDAQNews}
              className="text-xs"
            >
              Refresh
            </Button>
          </div>
        </div>
        {metrics && (
          <CardDescription className="flex items-center gap-4 text-xs">
            <span className={`flex items-center gap-1 ${getSentimentColor(metrics.overall_sentiment.score)}`}>
              {metrics.overall_sentiment.score > 0 ? <TrendingUp className="h-3 w-3" /> : <TrendingDown className="h-3 w-3" />}
              {metrics.overall_sentiment.label} ({metrics.overall_sentiment.score.toFixed(2)})
            </span>
            <span>Avg Urgency: {metrics.average_urgency.toFixed(1)}</span>
            {metrics.top_tickers.length > 0 && (
              <span>Top: {metrics.top_tickers.slice(0, 3).map(t => t.ticker).join(', ')}</span>
            )}
          </CardDescription>
        )}
      </CardHeader>
      <CardContent>
        {newsItems.length === 0 ? (
          <div className="text-center py-4">
            <p className="text-sm text-muted-foreground">
              {showUrgentOnly 
                ? "No urgent NASDAQ news at the moment" 
                : "No recent NASDAQ news available"
              }
            </p>
            <p className="text-xs text-muted-foreground mt-1">
              Last updated: {newsData?.last_updated ? new Date(newsData.last_updated).toLocaleTimeString() : 'Unknown'}
            </p>
          </div>
        ) : (
          <div className="space-y-3">
            {newsItems.map((item, index) => (
              <div key={index} className="border rounded-lg p-3 hover:bg-muted/20 transition-colors">
                <div className="flex items-start justify-between gap-2 mb-2">
                  <div className="flex items-center gap-2 flex-wrap">
                    <span className="text-lg">{getCategoryIcon(item.category)}</span>
                    <Badge variant="outline" className="text-xs">
                      {item.category}
                    </Badge>
                    <Badge 
                      variant="outline" 
                      className={`text-xs ${getImpactColor(item.impact_level)}`}
                    >
                      {item.impact_level} impact
                    </Badge>
                    {item.is_urgent && (
                      <Badge variant="destructive" className="text-xs">
                        URGENT
                      </Badge>
                    )}
                  </div>
                  <div className="flex items-center gap-1 text-xs text-muted-foreground">
                    {getUrgencyIcon(item.urgency_score)}
                    <span>{item.urgency_score.toFixed(1)}</span>
                  </div>
                </div>
                
                <h4 className="font-medium text-sm mb-1 line-clamp-2">
                  {item.title}
                </h4>
                
                <p className="text-xs text-muted-foreground mb-2 line-clamp-2">
                  {item.content}
                </p>
                
                <div className="flex items-center justify-between text-xs">
                  <div className="flex items-center gap-2">
                    <span className="text-muted-foreground">{item.source}</span>
                    <span className="text-muted-foreground">•</span>
                    <span className="text-muted-foreground">{item.time_ago}</span>
                    {item.tickers_mentioned.length > 0 && (
                      <>
                        <span className="text-muted-foreground">•</span>
                        <div className="flex gap-1">
                          {item.tickers_mentioned.slice(0, 3).map((ticker, i) => (
                            <Badge key={i} variant="secondary" className="text-xs px-1 py-0">
                              {ticker}
                            </Badge>
                          ))}
                        </div>
                      </>
                    )}
                  </div>
                  
                  <div className="flex items-center gap-2">
                    <span className={`text-xs ${getSentimentColor(item.sentiment_score)}`}>
                      {item.sentiment_score > 0 ? '+' : ''}{item.sentiment_score.toFixed(2)}
                    </span>
                    {item.url && (
                      <Button
                        variant="ghost"
                        size="sm"
                        className="h-6 w-6 p-0"
                        onClick={() => window.open(item.url, '_blank')}
                      >
                        <ExternalLink className="h-3 w-3" />
                      </Button>
                    )}
                  </div>
                </div>
              </div>
            ))}
            
            <div className="text-center pt-2">
              <p className="text-xs text-muted-foreground">
                Last updated: {newsData?.last_updated ? new Date(newsData.last_updated).toLocaleTimeString() : 'Unknown'}
              </p>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
