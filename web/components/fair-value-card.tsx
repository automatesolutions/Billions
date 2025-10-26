"use client";

import { useValuation } from '@/hooks/use-valuation';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { TrendingUp, TrendingDown, Minus, Calculator, DollarSign, BarChart3 } from 'lucide-react';

interface FairValueCardProps {
  ticker: string;
}

export function FairValueCard({ ticker }: FairValueCardProps) {
  const { data, loading, error } = useValuation(ticker);

  if (loading) {
    return (
      <Card className="bg-gray-800 border-gray-700">
        <CardHeader className="pb-3">
          <CardTitle className="text-lg flex items-center gap-2 text-white">
            <Calculator className="h-5 w-5" />
            Fair Value Analysis
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="animate-pulse space-y-3">
            <div className="h-4 bg-gray-700 rounded w-3/4"></div>
            <div className="h-4 bg-gray-700 rounded w-1/2"></div>
            <div className="h-4 bg-gray-700 rounded w-2/3"></div>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (error || !data) {
    return (
      <Card className="bg-gray-800 border-gray-700">
        <CardHeader className="pb-3">
          <CardTitle className="text-lg flex items-center gap-2 text-white">
            <Calculator className="h-5 w-5" />
            Fair Value Analysis
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center text-gray-400 py-4">
            <Calculator className="h-8 w-8 mx-auto mb-2 opacity-50" />
            <p>Unable to calculate fair value</p>
            <p className="text-sm">{error}</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  const getValuationIcon = () => {
    switch (data.valuation_status) {
      case 'Undervalued':
        return <TrendingUp className="h-4 w-4 text-green-400" />;
      case 'Overvalued':
        return <TrendingDown className="h-4 w-4 text-red-400" />;
      default:
        return <Minus className="h-4 w-4 text-yellow-400" />;
    }
  };

  const getValuationColor = () => {
    switch (data.valuation_color) {
      case 'green':
        return 'bg-green-900/30 text-green-400 border-green-500/30';
      case 'red':
        return 'bg-red-900/30 text-red-400 border-red-500/30';
      case 'yellow':
        return 'bg-yellow-900/30 text-yellow-400 border-yellow-500/30';
      default:
        return 'bg-gray-700/30 text-gray-300 border-gray-600/30';
    }
  };

  const priceDifference = data.current_price - data.fair_value;
  const priceDifferencePercent = (priceDifference / data.fair_value) * 100;

  return (
    <Card className="bg-gray-800 border-gray-700">
      <CardHeader className="pb-3">
        <CardTitle className="text-lg flex items-center gap-2 text-white">
          <Calculator className="h-5 w-5" />
          Black-Scholes Fair Value
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Current vs Fair Value */}
        <div className="grid grid-cols-2 gap-4">
          <div className="text-center p-3 bg-gray-700/50 rounded-lg border border-gray-600">
            <div className="flex items-center justify-center gap-1 mb-1">
              <DollarSign className="h-4 w-4 text-gray-400" />
              <span className="text-sm text-gray-400">Current</span>
            </div>
            <div className="text-xl font-bold text-white">${data.current_price.toFixed(2)}</div>
          </div>
          <div className="text-center p-3 bg-gray-700/50 rounded-lg border border-blue-500/30">
            <div className="flex items-center justify-center gap-1 mb-1">
              <BarChart3 className="h-4 w-4 text-blue-400" />
              <span className="text-sm text-gray-400">Fair Value</span>
            </div>
            <div className="text-xl font-bold text-blue-400">${data.fair_value.toFixed(2)}</div>
          </div>
        </div>

        {/* Valuation Status */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            {getValuationIcon()}
            <span className="font-medium text-white">Valuation Status:</span>
          </div>
          <Badge className={getValuationColor()}>
            {data.valuation_status}
          </Badge>
        </div>

        {/* Price Difference */}
        <div className="p-3 bg-gray-700/50 rounded-lg border border-gray-600">
          <div className="flex items-center justify-between">
            <span className="text-sm text-gray-400">Price vs Fair Value:</span>
            <div className="text-right">
              <div className={`font-medium ${priceDifference >= 0 ? 'text-red-400' : 'text-green-400'}`}>
                {priceDifference >= 0 ? '+' : ''}${priceDifference.toFixed(2)}
              </div>
              <div className={`text-sm ${priceDifference >= 0 ? 'text-red-400' : 'text-green-400'}`}>
                ({priceDifferencePercent >= 0 ? '+' : ''}{priceDifferencePercent.toFixed(1)}%)
              </div>
            </div>
          </div>
        </div>

        {/* Key Metrics */}
        <div className="grid grid-cols-2 gap-3 text-sm">
          <div className="flex justify-between">
            <span className="text-gray-400">Valuation Ratio:</span>
            <span className="font-medium text-white">{data.valuation_ratio.toFixed(3)}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-400">Volatility:</span>
            <span className="font-medium text-white">{(data.volatility * 100).toFixed(1)}%</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-400">Risk-Free Rate:</span>
            <span className="font-medium text-white">{(data.risk_free_rate * 100).toFixed(2)}%</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-400">Beta:</span>
            <span className="font-medium text-white">{data.beta?.toFixed(2) || 'N/A'}</span>
          </div>
        </div>

        {/* Analysis Date */}
        <div className="text-xs text-gray-500 text-center pt-2 border-t border-gray-700">
          Analysis Date: {new Date(data.analysis_date).toLocaleDateString()}
        </div>

        {/* Explanation */}
        <div className="p-3 bg-blue-900/20 rounded-lg border border-blue-800/30">
          <h4 className="font-semibold text-blue-400 mb-2 text-sm flex items-center gap-2">
            <BarChart3 className="h-4 w-4" />
            Black-Scholes-Merton Model
          </h4>
          <div className="text-xs text-blue-300 space-y-1">
            <p>
              <strong className="text-blue-200">Fair Value:</strong> Calculated using option pricing theory, considering volatility, 
              risk-free rate, and time to expiry.
            </p>
            <p>
              <strong className="text-blue-200">Valuation Status:</strong> 
              {data.valuation_status === 'Undervalued' && ' Stock may be trading below its theoretical fair value.'}
              {data.valuation_status === 'Overvalued' && ' Stock may be trading above its theoretical fair value.'}
              {data.valuation_status === 'Fairly Valued' && ' Stock appears to be trading near its fair value.'}
            </p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
