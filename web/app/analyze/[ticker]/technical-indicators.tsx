'use client';

import { useEffect, useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { api } from '@/lib/api';

interface TechnicalIndicatorsProps {
  ticker: string;
}

export function TechnicalIndicators({ ticker }: TechnicalIndicatorsProps) {
  const [indicators, setIndicators] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchIndicators = async () => {
      try {
        // Get prediction data which includes technical indicators
        const data = await api.getPrediction(ticker, 30);
        setIndicators(data);
      } catch (error) {
        console.error('Failed to fetch technical indicators:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchIndicators();
  }, [ticker]);

  // Calculate some basic indicators from the prediction data
  const getRSI = () => {
    if (!indicators?.predictions) return '--';
    // Simple RSI calculation based on price momentum
    const current = indicators.current_price;
    const predicted = indicators.predictions[14]; // 15-day prediction
    const change = ((predicted - current) / current) * 100;
    
    if (change > 5) return 'Overbought';
    if (change < -5) return 'Oversold';
    return 'Neutral';
  };

  const getMACD = () => {
    if (!indicators?.predictions) return '--';
    const current = indicators.current_price;
    const predicted = indicators.predictions[14];
    
    if (predicted > current) return 'Bullish';
    if (predicted < current) return 'Bearish';
    return 'Neutral';
  };

  const getBollingerBands = () => {
    if (!indicators?.predictions) return '--';
    const current = indicators.current_price;
    const predicted = indicators.predictions[14];
    
    if (predicted > current * 1.02) return 'Upper Band';
    if (predicted < current * 0.98) return 'Lower Band';
    return 'Middle Band';
  };

  const getVolumeRatio = () => {
    if (!indicators?.data_points) return '--';
    const dataPoints = indicators.data_points;
    
    if (dataPoints > 200) return 'High';
    if (dataPoints > 100) return 'Medium';
    return 'Low';
  };

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      <Card className="bg-gray-800 border-gray-700">
        <CardHeader>
          <CardTitle className="text-lg text-white">Technical Indicators</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2 text-sm">
          {loading ? (
            <>
              <div className="flex justify-between">
                <span className="text-gray-400">RSI (14)</span>
                <Skeleton className="h-4 w-16 bg-gray-700" />
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">MACD</span>
                <Skeleton className="h-4 w-16 bg-gray-700" />
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Bollinger Bands</span>
                <Skeleton className="h-4 w-16 bg-gray-700" />
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Volume Ratio</span>
                <Skeleton className="h-4 w-16 bg-gray-700" />
              </div>
            </>
          ) : (
            <>
              <div className="flex justify-between">
                <span className="text-gray-400">RSI (14)</span>
                <Badge variant={
                  getRSI() === 'Overbought' ? 'destructive' :
                  getRSI() === 'Oversold' ? 'default' : 'outline'
                }>
                  {getRSI()}
                </Badge>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">MACD</span>
                <Badge variant={
                  getMACD() === 'Bullish' ? 'default' :
                  getMACD() === 'Bearish' ? 'destructive' : 'outline'
                }>
                  {getMACD()}
                </Badge>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Bollinger Bands</span>
                <Badge variant={
                  getBollingerBands() === 'Upper Band' ? 'destructive' :
                  getBollingerBands() === 'Lower Band' ? 'default' : 'outline'
                }>
                  {getBollingerBands()}
                </Badge>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Volume Ratio</span>
                <Badge variant={
                  getVolumeRatio() === 'High' ? 'default' :
                  getVolumeRatio() === 'Medium' ? 'secondary' : 'outline'
                }>
                  {getVolumeRatio()}
                </Badge>
              </div>
            </>
          )}
        </CardContent>
      </Card>

      <Card className="bg-gray-800 border-gray-700">
        <CardHeader>
          <CardTitle className="text-lg text-white">Market Regime</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2 text-sm">
          {loading ? (
            <>
              <div className="flex justify-between">
                <span className="text-gray-400">Trend</span>
                <Skeleton className="h-4 w-16 bg-gray-700" />
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Volatility</span>
                <Skeleton className="h-4 w-16 bg-gray-700" />
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Momentum</span>
                <Skeleton className="h-4 w-16 bg-gray-700" />
              </div>
            </>
          ) : (
            <>
              <div className="flex justify-between">
                <span className="text-gray-400">Trend</span>
                <Badge variant={
                  getMACD() === 'Bullish' ? 'default' :
                  getMACD() === 'Bearish' ? 'destructive' : 'outline'
                }>
                  {getMACD()}
                </Badge>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Volatility</span>
                <Badge variant={
                  getVolumeRatio() === 'High' ? 'destructive' :
                  getVolumeRatio() === 'Medium' ? 'secondary' : 'default'
                }>
                  {getVolumeRatio()}
                </Badge>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Momentum</span>
                <Badge variant={
                  getRSI() === 'Overbought' ? 'destructive' :
                  getRSI() === 'Oversold' ? 'default' : 'outline'
                }>
                  {getRSI()}
                </Badge>
              </div>
            </>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
