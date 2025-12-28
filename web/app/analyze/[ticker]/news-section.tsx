'use client';

import { useEffect, useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { HypeWarningCard } from "@/components/hype-warning-card";
import { api } from '@/lib/api';

interface NewsSectionProps {
  ticker: string;
}

export function NewsSection({ ticker }: NewsSectionProps) {
  const [news, setNews] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchNews = async () => {
      try {
        const data = await api.getNews(ticker, 5);
        setNews(data);
      } catch (error) {
        console.error('Failed to fetch news:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchNews();
  }, [ticker]);

  return (
    <Card className="bg-gray-800 border-gray-700">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="text-white">Latest News</CardTitle>
            <CardDescription className="text-gray-400">Recent headlines with sentiment analysis</CardDescription>
          </div>
          {news && (
            <Badge variant={
              news.overall_sentiment.label === 'positive' ? 'default' :
              news.overall_sentiment.label === 'negative' ? 'destructive' :
              'secondary'
            }>
              {news.overall_sentiment.label} sentiment
            </Badge>
          )}
        </div>
      </CardHeader>
      <CardContent>
        {loading ? (
          <div className="space-y-4">
            {[1, 2, 3].map((i) => (
              <div key={i} className="space-y-2">
                <Skeleton className="h-5 w-3/4 bg-gray-700" />
                <Skeleton className="h-4 w-1/2 bg-gray-700" />
              </div>
            ))}
          </div>
        ) : news && news.articles.length > 0 ? (
          <div className="space-y-4">
            {news.articles.map((article: any, index: number) => (
              <div key={index} className="pb-4 border-b border-gray-700 last:border-0">
                <div className="flex items-start justify-between gap-2">
                  <a
                    href={article.link}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="hover:underline flex-1"
                  >
                    <h4 className="font-semibold text-sm text-white">{article.title}</h4>
                  </a>
                  <Badge
                    variant={
                      article.sentiment.label === 'positive' ? 'default' :
                      article.sentiment.label === 'negative' ? 'destructive' :
                      'outline'
                    }
                    className="text-xs"
                  >
                    {article.sentiment.label}
                  </Badge>
                </div>
                <p className="text-xs text-gray-500 mt-1">
                  {article.publisher} • {article.published_at ? new Date(article.published_at).toLocaleDateString() : ''}
                </p>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-sm text-gray-400">No news available for {ticker}</p>
        )}
      </CardContent>
      
      {/* HYPE and CAVEAT EMPTOR Analysis */}
      {news && news.hype_analysis && news.caveat_emptor && (
        <div className="p-6 pt-0 border-t border-gray-700">
          <HypeWarningCard 
            hypeAnalysis={news.hype_analysis}
            caveatEmptor={news.caveat_emptor}
          />
        </div>
      )}
    </Card>
  );
}

