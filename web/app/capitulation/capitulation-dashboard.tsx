'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { 
  TrendingDown, 
  AlertTriangle, 
  BarChart3, 
  RefreshCw,
  Target,
  Activity,
  Zap,
  Eye
} from "lucide-react";
import { api } from "@/lib/api";

interface CapitulationStock {
  symbol: string;
  current_price: number;
  market_cap: number;
  avg_volume: number;
  current_volume: number;
  sector: string;
  industry: string;
  is_capitulation: boolean;
  capitulation_score: number;
  confidence: number;
  signals: string[];
  signal_count: number;
  signal_types: number;
  risk_level: string;
  indicators: {
    rsi: number;
    volume_ratio_20: number;
    price_change: number;
    price_change_3d: number;
    price_change_5d: number;
    distance_sma20: number;
    distance_sma50: number;
    volatility: number;
  };
  timestamp: string;
}

interface CapitulationSummary {
  total_stocks_analyzed: number;
  capitulation_stocks: CapitulationStock[];
  capitulation_count: number;
  capitulation_rate: number;
  errors: number;
  market_summary: MarketSummary;
  timestamp: string;
  analysis_type: string;
}

interface MarketSummary {
  vix: number;
  vix_change: number;
  spy_change: number;
  qqq_change: number;
  market_condition: string;
  market_trend: string;
  timestamp: string;
}

export function CapitulationDashboard() {
  const [summary, setSummary] = useState<CapitulationSummary | null>(null);
  const [marketSummary, setMarketSummary] = useState<MarketSummary | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [lastUpdated, setLastUpdated] = useState<string>('');

  const fetchCapitulationData = async () => {
    setIsLoading(true);
    try {
      // Fetch capitulation screening data from enhanced API
      const screenData = await api.screenCapitulation(50);
      setSummary(screenData);
      setLastUpdated(new Date().toLocaleTimeString());

      // Use market summary from screening data
      if (screenData.market_summary) {
        setMarketSummary(screenData.market_summary);
      }
    } catch (error) {
      console.error('Error fetching capitulation data:', error);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchCapitulationData();
  }, []);

  const getSignalBadge = (signal: string) => {
    const signalConfig = {
      // Volume Signals
      'volume_spike_20': { label: 'Volume Spike (20d)', color: 'bg-red-500' },
      'volume_elevated_20': { label: 'Volume Elevated (20d)', color: 'bg-orange-500' },
      'volume_spike_50': { label: 'Volume Spike (50d)', color: 'bg-red-600' },
      
      // RSI Signals
      'rsi_extreme_oversold': { label: 'RSI Extreme Oversold', color: 'bg-red-700' },
      'rsi_oversold': { label: 'RSI Oversold', color: 'bg-red-500' },
      'rsi_near_oversold': { label: 'RSI Near Oversold', color: 'bg-orange-500' },
      'rsi_weak': { label: 'RSI Weak', color: 'bg-yellow-500' },
      
      // Momentum Signals
      'macd_bearish': { label: 'MACD Bearish', color: 'bg-purple-500' },
      'stoch_oversold': { label: 'Stochastic Oversold', color: 'bg-purple-600' },
      'williams_oversold': { label: 'Williams Oversold', color: 'bg-purple-700' },
      
      // Price Action Signals
      'extreme_down_day': { label: 'Extreme Down Day', color: 'bg-red-800' },
      'large_down_day': { label: 'Large Down Day', color: 'bg-red-600' },
      'moderate_down_day': { label: 'Moderate Down Day', color: 'bg-orange-600' },
      'small_down_day': { label: 'Small Down Day', color: 'bg-yellow-600' },
      
      // Multi-day Signals
      'extreme_3d_decline': { label: 'Extreme 3D Decline', color: 'bg-red-800' },
      'large_3d_decline': { label: 'Large 3D Decline', color: 'bg-red-600' },
      'moderate_3d_decline': { label: 'Moderate 3D Decline', color: 'bg-orange-600' },
      'extreme_5d_decline': { label: 'Extreme 5D Decline', color: 'bg-red-800' },
      'large_5d_decline': { label: 'Large 5D Decline', color: 'bg-red-600' },
      'moderate_5d_decline': { label: 'Moderate 5D Decline', color: 'bg-orange-600' },
      
      // Trend Signals
      'far_below_sma20': { label: 'Far Below SMA20', color: 'bg-blue-600' },
      'below_sma20': { label: 'Below SMA20', color: 'bg-blue-500' },
      'near_sma20': { label: 'Near SMA20', color: 'bg-blue-400' },
      'far_below_sma50': { label: 'Far Below SMA50', color: 'bg-indigo-600' },
      'below_sma50': { label: 'Below SMA50', color: 'bg-indigo-500' },
      'far_below_sma200': { label: 'Far Below SMA200', color: 'bg-violet-600' },
      'below_sma200': { label: 'Below SMA200', color: 'bg-violet-500' },
      
      // Volatility Signals
      'high_volatility': { label: 'High Volatility', color: 'bg-pink-500' },
      'elevated_volatility': { label: 'Elevated Volatility', color: 'bg-pink-400' },
      
      // Pattern Signals
      'hammer_pattern': { label: 'Hammer Pattern', color: 'bg-green-500' },
      'long_lower_tail': { label: 'Long Lower Tail', color: 'bg-green-400' },
      'doji_pattern': { label: 'Doji Pattern', color: 'bg-gray-500' },
      'gap_down': { label: 'Gap Down', color: 'bg-red-500' },
      'lower_lows_pattern': { label: 'Lower Lows Pattern', color: 'bg-red-400' },
      
      // Legacy signals (for backward compatibility)
      'volume_spike': { label: 'Volume Spike', color: 'bg-red-500' },
      'large_down_candle': { label: 'Large Down Candle', color: 'bg-red-600' },
      'moderate_down_candle': { label: 'Moderate Down Candle', color: 'bg-orange-600' },
      'long_tail': { label: 'Long Tail', color: 'bg-blue-500' }
    };

    const config = signalConfig[signal as keyof typeof signalConfig] || { label: signal, color: 'bg-gray-500' };
    
    return (
      <Badge className={`${config.color} text-white`} key={signal}>
        {config.label}
      </Badge>
    );
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'text-red-600';
    if (confidence >= 0.6) return 'text-orange-600';
    if (confidence >= 0.4) return 'text-yellow-600';
    return 'text-gray-600';
  };

  const formatMarketCap = (marketCap: number) => {
    if (marketCap >= 1e12) return `$${(marketCap / 1e12).toFixed(1)}T`;
    if (marketCap >= 1e9) return `$${(marketCap / 1e9).toFixed(1)}B`;
    if (marketCap >= 1e6) return `$${(marketCap / 1e6).toFixed(1)}M`;
    return `$${marketCap.toFixed(0)}`;
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <AlertTriangle className="h-5 w-5 text-red-500" />
          Capitulation Detection
        </CardTitle>
        <CardDescription>
          Real-time screening of NASDAQ stocks for capitulation signals using live market data
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-6">
          {/* Market Summary */}
          {marketSummary && (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <Card className="p-4">
                <div className="flex items-center gap-2 mb-2">
                  <Activity className="h-4 w-4 text-red-600" />
                  <span className="text-sm font-medium">VIX</span>
                </div>
                <div className="text-2xl font-bold">{marketSummary.vix.toFixed(1)}</div>
                <div className={`text-sm ${marketSummary.vix_change >= 0 ? 'text-red-600' : 'text-green-600'}`}>
                  {marketSummary.vix_change >= 0 ? '+' : ''}{marketSummary.vix_change.toFixed(1)}%
                </div>
              </Card>

              <Card className="p-4">
                <div className="flex items-center gap-2 mb-2">
                  <Target className="h-4 w-4 text-orange-600" />
                  <span className="text-sm font-medium">Market Trend</span>
                </div>
                <div className="text-lg font-bold">{marketSummary.market_trend}</div>
                <div className="text-sm text-muted-foreground">
                  SPY: {marketSummary.spy_change >= 0 ? '+' : ''}{marketSummary.spy_change.toFixed(2)}% | 
                  QQQ: {marketSummary.qqq_change >= 0 ? '+' : ''}{marketSummary.qqq_change.toFixed(2)}%
                </div>
              </Card>

              <Card className="p-4">
                <div className="flex items-center gap-2 mb-2">
                  <BarChart3 className="h-4 w-4 text-purple-600" />
                  <span className="text-sm font-medium">Capitulation Rate</span>
                </div>
                <div className="text-2xl font-bold">
                  {summary ? summary.capitulation_rate.toFixed(1) : '0.0'}%
                </div>
                <div className="text-sm text-muted-foreground">
                  {summary ? `${summary.capitulation_count}/${summary.total_stocks_analyzed}` : '0/0'} stocks
                </div>
              </Card>
            </div>
          )}

          {/* Controls */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <div className={`w-3 h-3 rounded-full ${isLoading ? 'bg-yellow-500 animate-pulse' : 'bg-green-500'}`}></div>
              <span className="text-sm text-muted-foreground">
                {isLoading ? 'Scanning...' : `Last updated: ${lastUpdated}`}
              </span>
            </div>
            <Button 
              onClick={fetchCapitulationData}
              disabled={isLoading}
              size="sm"
              variant="outline"
            >
              <RefreshCw className={`h-4 w-4 mr-2 ${isLoading ? 'animate-spin' : ''}`} />
              Refresh
            </Button>
          </div>

          {/* Capitulation Stocks Table */}
          {summary && summary.capitulation_stocks.length > 0 ? (
            <div className="space-y-4">
              <h3 className="text-lg font-semibold flex items-center gap-2">
                <Zap className="h-5 w-5 text-red-500" />
                Stocks in Capitulation ({summary.capitulation_stocks.length})
              </h3>
              
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Symbol</TableHead>
                    <TableHead>Sector</TableHead>
                    <TableHead>Price</TableHead>
                    <TableHead>Change</TableHead>
                    <TableHead>Volume Ratio</TableHead>
                    <TableHead>RSI</TableHead>
                    <TableHead>Score</TableHead>
                    <TableHead>Signals</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {summary.capitulation_stocks.map((stock) => (
                    <TableRow key={stock.symbol}>
                      <TableCell>
                        <Badge variant="outline" className="font-mono">
                          {stock.symbol}
                        </Badge>
                      </TableCell>
                      <TableCell>
                        <div>
                          <div className="font-medium">{stock.sector}</div>
                          <div className="text-sm text-muted-foreground">
                            {stock.industry}
                          </div>
                        </div>
                      </TableCell>
                      <TableCell>
                        <div className="font-medium">
                          ${stock.current_price.toFixed(2)}
                        </div>
                        <div className="text-sm text-muted-foreground">
                          {formatMarketCap(stock.market_cap)}
                        </div>
                      </TableCell>
                      <TableCell className={stock.indicators.price_change >= 0 ? 'text-green-600' : 'text-red-600'}>
                        {stock.indicators.price_change >= 0 ? '+' : ''}{stock.indicators.price_change.toFixed(2)}%
                      </TableCell>
                      <TableCell>
                        <Badge variant={stock.indicators.volume_ratio_20 >= 2.5 ? 'destructive' : 'secondary'}>
                          {stock.indicators.volume_ratio_20.toFixed(1)}x
                        </Badge>
                      </TableCell>
                      <TableCell>
                        <Badge variant={stock.indicators.rsi <= 30 ? 'destructive' : stock.indicators.rsi <= 35 ? 'secondary' : 'outline'}>
                          {stock.indicators.rsi.toFixed(1)}
                        </Badge>
                      </TableCell>
                      <TableCell>
                        <div className="flex items-center gap-2">
                          <span className={`font-bold ${getConfidenceColor(stock.confidence)}`}>
                            {stock.capitulation_score}/20
                          </span>
                          <div className="w-16 bg-gray-200 rounded-full h-2">
                            <div 
                              className={`h-2 rounded-full ${getConfidenceColor(stock.confidence).replace('text-', 'bg-')}`}
                              style={{ width: `${stock.confidence * 100}%` }}
                            ></div>
                          </div>
                        </div>
                      </TableCell>
                      <TableCell>
                        <div className="flex flex-wrap gap-1">
                          {stock.signals.slice(0, 3).map(getSignalBadge)}
                          {stock.signals.length > 3 && (
                            <Badge variant="outline" className="text-xs">
                              +{stock.signals.length - 3}
                            </Badge>
                          )}
                        </div>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          ) : (
            <div className="text-center py-8">
              <Eye className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
              <h3 className="text-lg font-semibold mb-2">No Capitulation Signals Detected</h3>
              <p className="text-muted-foreground">
                {summary ? 'No stocks are currently showing capitulation signals.' : 'Loading capitulation data...'}
              </p>
            </div>
          )}

          {/* Legend */}
          <Card className="p-4">
            <h4 className="font-semibold mb-3">Capitulation Indicators</h4>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
              <div>
                <h5 className="font-medium mb-2">Enhanced Technical Signals:</h5>
                <ul className="space-y-1 text-muted-foreground">
                  <li>• <strong>Volume Spike:</strong> 2.5x+ average volume (lowered threshold)</li>
                  <li>• <strong>RSI Oversold:</strong> Below 30 (extreme below 25)</li>
                  <li>• <strong>MACD Bearish:</strong> Negative momentum</li>
                  <li>• <strong>Price Declines:</strong> 1.5%+ daily, 5%+ 3-day, 7%+ 5-day</li>
                  <li>• <strong>Trend Breaks:</strong> Below moving averages</li>
                  <li>• <strong>Patterns:</strong> Hammer, Doji, Gap Down</li>
                </ul>
              </div>
              <div>
                <h5 className="font-medium mb-2">Enhanced Market Indicators:</h5>
                <ul className="space-y-1 text-muted-foreground">
                  <li>• <strong>VIX:</strong> Volatility index (fear gauge)</li>
                  <li>• <strong>SPY/QQQ:</strong> Market trend context</li>
                  <li>• <strong>Score:</strong> 3+ indicates capitulation (lowered from 5)</li>
                  <li>• <strong>Confidence:</strong> Signal strength (0-1)</li>
                  <li>• <strong>Risk Level:</strong> Low/Moderate/High/Extreme</li>
                  <li>• <strong>Coverage:</strong> Extended NASDAQ stock list</li>
                </ul>
              </div>
            </div>
          </Card>
        </div>
      </CardContent>
    </Card>
  );
}
