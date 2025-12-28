'use client';

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { useTickerInfo } from "@/hooks/use-ticker-info";
import { usePrediction } from "@/hooks/use-prediction";
import { CandlestickPredictionChart } from "@/components/charts/candlestick-prediction-chart";

interface ClientAnalyzePageProps {
  ticker: string;
}

export function ClientAnalyzePage({ ticker }: ClientAnalyzePageProps) {
  const { data: info, loading: infoLoading } = useTickerInfo(ticker);
  const { data: prediction, loading: predLoading } = usePrediction(ticker, 30);

  return (
    <>
      {/* Stock Info */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle>{ticker}</CardTitle>
            {infoLoading ? (
              <Badge variant="secondary">Loading...</Badge>
            ) : info ? (
              <Badge variant="default">Live Data</Badge>
            ) : (
              <Badge variant="destructive">Error</Badge>
            )}
          </div>
          <CardDescription>
            {info?.name || 'Stock information'}
          </CardDescription>
        </CardHeader>
        <CardContent>
          {infoLoading ? (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {[1, 2, 3, 4].map((i) => (
                <div key={i}>
                  <Skeleton className="h-4 w-20 mb-2" />
                  <Skeleton className="h-8 w-24" />
                </div>
              ))}
            </div>
          ) : info ? (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
              <div>
                <p className="text-muted-foreground">Current Price</p>
                <p className="text-2xl font-bold">
                  ${info.current_price?.toFixed(2) || '--'}
                </p>
              </div>
              <div>
                <p className="text-muted-foreground">Market Cap</p>
                <p className="text-xl font-semibold">
                  ${(info.market_cap / 1e9).toFixed(1)}B
                </p>
              </div>
              <div>
                <p className="text-muted-foreground">Volume</p>
                <p className="text-xl font-semibold">
                  {(info.volume / 1e6).toFixed(1)}M
                </p>
              </div>
              <div>
                <p className="text-muted-foreground">Sector</p>
                <p className="text-xl font-semibold">{info.sector}</p>
              </div>
            </div>
          ) : (
            <p className="text-destructive">Failed to load stock info</p>
          )}
        </CardContent>
      </Card>

      {/* 30-Day Forecast */}
      <Card>
        <CardHeader>
          <CardTitle>30-Day ML Forecast</CardTitle>
          <CardDescription>
            LSTM-based prediction with confidence intervals
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {predLoading ? (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {[1, 2, 3].map((i) => (
                <div key={i} className="p-4 border rounded-lg">
                  <Skeleton className="h-4 w-24 mb-2" />
                  <Skeleton className="h-8 w-20" />
                </div>
              ))}
            </div>
          ) : prediction ? (
            <>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="p-4 border rounded-lg">
                  <p className="text-sm text-muted-foreground">Current Price</p>
                  <p className="text-2xl font-bold">
                    ${prediction.current_price.toFixed(2)}
                  </p>
                </div>
                <div className="p-4 border rounded-lg">
                  <p className="text-sm text-muted-foreground">30-Day Target</p>
                  <p className="text-2xl font-bold text-green-600">
                    ${prediction.predictions[29]?.toFixed(2) || '--'}
                  </p>
                  <p className="text-xs text-muted-foreground">
                    {((prediction.predictions[29] - prediction.current_price) / prediction.current_price * 100).toFixed(1)}% expected
                  </p>
                </div>
              </div>

              <div className="text-xs text-muted-foreground">
                Last updated: {new Date(prediction.last_updated).toLocaleDateString()}
              </div>

              {/* Candlestick Prediction Chart */}
        <CandlestickPredictionChart
          ticker={ticker}
          currentPrice={prediction.current_price}
          predictions={prediction.predictions}
          confidenceUpper={prediction.confidence_upper}
          confidenceLower={prediction.confidence_lower}
        />
            </>
          ) : (
            <div className="text-sm text-muted-foreground">
              Model not loaded or prediction failed. Train LSTM model first.
            </div>
          )}
        </CardContent>
      </Card>
    </>
  );
}

