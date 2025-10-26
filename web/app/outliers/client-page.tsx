'use client';

import { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Skeleton } from "@/components/ui/skeleton";
import { usePerformanceMetrics } from "@/hooks/use-performance-metrics";
import { PlotlyScatterPlot } from "@/components/charts/plotly-scatter-plot";
import { useToast } from "@/components/toast-provider";

export function ClientOutliersPage() {
  const [strategy, setStrategy] = useState<string>('swing');
  const [autoRefresh, setAutoRefresh] = useState(false);
  const { data, loading, error, refetch } = usePerformanceMetrics(strategy, autoRefresh);
  const { addToast } = useToast();

  // Mock data for testing while API is being debugged - ALL STOCKS (normal + outliers)
  const mockData = {
    strategy: strategy,
    count: 50,
    metrics: [
      // Outliers (red points)
      { symbol: 'AAPL', metric_x: 15.2, metric_y: -8.5, z_x: 2.3, z_y: -2.1, is_outlier: true },
      { symbol: 'TSLA', metric_x: 28.7, metric_y: 12.3, z_x: 3.1, z_y: 2.8, is_outlier: true },
      { symbol: 'NVDA', metric_x: -5.2, metric_y: 18.9, z_x: -1.8, z_y: 3.2, is_outlier: true },
      { symbol: 'AMZN', metric_x: -2.1, metric_y: -5.7, z_x: -0.8, z_y: -2.3, is_outlier: true },
      { symbol: 'META', metric_x: 22.3, metric_y: -12.1, z_x: 2.9, z_y: -2.7, is_outlier: true },
      { symbol: 'NFLX', metric_x: -8.9, metric_y: 15.6, z_x: -2.1, z_y: 2.9, is_outlier: true },
      { symbol: 'INTC', metric_x: -12.3, metric_y: -8.9, z_x: -2.5, z_y: -2.8, is_outlier: true },
      { symbol: 'CRM', metric_x: 18.2, metric_y: 9.7, z_x: 2.6, z_y: 2.1, is_outlier: true },
      { symbol: 'ADBE', metric_x: -6.8, metric_y: 11.4, z_x: -1.9, z_y: 2.3, is_outlier: true },
      { symbol: 'PYPL', metric_x: 9.5, metric_y: -6.2, z_x: 2.0, z_y: -2.4, is_outlier: true },
      { symbol: 'IBM', metric_x: -15.7, metric_y: -11.3, z_x: -3.1, z_y: -3.2, is_outlier: true },
      { symbol: 'TXN', metric_x: -4.6, metric_y: 8.7, z_x: -1.3, z_y: 1.9, is_outlier: true },
      { symbol: 'AVGO', metric_x: 13.8, metric_y: -4.1, z_x: 2.4, z_y: -1.7, is_outlier: true },
      { symbol: 'MRVL', metric_x: -9.4, metric_y: 16.2, z_x: -2.2, z_y: 2.8, is_outlier: true },
      { symbol: 'LRCX', metric_x: -7.1, metric_y: 13.5, z_x: -1.8, z_y: 2.5, is_outlier: true },
      { symbol: 'KLAC', metric_x: 11.6, metric_y: -2.9, z_x: 2.1, z_y: -1.2, is_outlier: true },
      { symbol: 'SNPS', metric_x: 16.9, metric_y: 5.8, z_x: 2.7, z_y: 1.6, is_outlier: true },
      
      // Normal stocks (blue points) - many more to show the full dataset
      { symbol: 'MSFT', metric_x: 8.1, metric_y: -3.2, z_x: 1.9, z_y: -1.5, is_outlier: false },
      { symbol: 'GOOGL', metric_x: 12.5, metric_y: 7.8, z_x: 2.2, z_y: 1.9, is_outlier: false },
      { symbol: 'AMD', metric_x: 6.7, metric_y: 4.2, z_x: 1.6, z_y: 1.2, is_outlier: false },
      { symbol: 'ORCL', metric_x: 3.4, metric_y: -1.8, z_x: 0.9, z_y: -0.7, is_outlier: false },
      { symbol: 'CSCO', metric_x: -1.2, metric_y: 2.8, z_x: -0.3, z_y: 0.8, is_outlier: false },
      { symbol: 'QCOM', metric_x: 7.9, metric_y: 3.1, z_x: 1.7, z_y: 0.9, is_outlier: false },
      { symbol: 'AMAT', metric_x: 5.3, metric_y: 1.7, z_x: 1.3, z_y: 0.5, is_outlier: false },
      { symbol: 'MCHP', metric_x: -3.8, metric_y: 6.4, z_x: -1.0, z_y: 1.4, is_outlier: false },
      { symbol: 'BABA', metric_x: 4.2, metric_y: 2.1, z_x: 1.1, z_y: 0.6, is_outlier: false },
      { symbol: 'NIO', metric_x: -1.8, metric_y: 3.5, z_x: -0.5, z_y: 1.0, is_outlier: false },
      { symbol: 'XPEV', metric_x: 2.7, metric_y: -2.3, z_x: 0.7, z_y: -0.9, is_outlier: false },
      { symbol: 'LI', metric_x: -0.9, metric_y: 1.8, z_x: -0.2, z_y: 0.5, is_outlier: false },
      { symbol: 'BIDU', metric_x: 3.1, metric_y: 0.7, z_x: 0.8, z_y: 0.2, is_outlier: false },
      { symbol: 'JD', metric_x: -2.4, metric_y: 4.1, z_x: -0.6, z_y: 1.2, is_outlier: false },
      { symbol: 'PDD', metric_x: 5.8, metric_y: -1.5, z_x: 1.5, z_y: -0.6, is_outlier: false },
      { symbol: 'TME', metric_x: -3.2, metric_y: 2.9, z_x: -0.8, z_y: 0.8, is_outlier: false },
      { symbol: 'VIPS', metric_x: 1.6, metric_y: -0.8, z_x: 0.4, z_y: -0.3, is_outlier: false },
      { symbol: 'YMM', metric_x: -1.1, metric_y: 1.2, z_x: -0.3, z_y: 0.3, is_outlier: false },
      { symbol: 'WB', metric_x: 2.3, metric_y: 0.4, z_x: 0.6, z_y: 0.1, is_outlier: false },
      { symbol: 'DIDI', metric_x: -4.7, metric_y: 3.2, z_x: -1.2, z_y: 0.9, is_outlier: false },
      { symbol: 'BILI', metric_x: 0.8, metric_y: -1.9, z_x: 0.2, z_y: -0.7, is_outlier: false },
      { symbol: 'IQ', metric_x: -2.8, metric_y: 1.6, z_x: -0.7, z_y: 0.5, is_outlier: false },
      { symbol: 'HUYA', metric_x: 1.9, metric_y: 0.3, z_x: 0.5, z_y: 0.1, is_outlier: false },
      { symbol: 'DOYU', metric_x: -0.6, metric_y: -0.4, z_x: -0.2, z_y: -0.1, is_outlier: false },
      { symbol: 'TAL', metric_x: 3.5, metric_y: 1.1, z_x: 0.9, z_y: 0.3, is_outlier: false },
      { symbol: 'EDU', metric_x: -1.7, metric_y: 2.4, z_x: -0.4, z_y: 0.7, is_outlier: false },
      { symbol: 'GOTU', metric_x: 0.5, metric_y: -1.2, z_x: 0.1, z_y: -0.4, is_outlier: false },
      { symbol: 'COIN', metric_x: -5.3, metric_y: 7.8, z_x: -1.3, z_y: 2.2, is_outlier: false },
      { symbol: 'SQ', metric_x: 4.1, metric_y: -0.9, z_x: 1.0, z_y: -0.3, is_outlier: false },
      { symbol: 'ROKU', metric_x: -2.9, metric_y: 5.2, z_x: -0.7, z_y: 1.5, is_outlier: false },
      { symbol: 'SPOT', metric_x: 1.4, metric_y: 0.6, z_x: 0.4, z_y: 0.2, is_outlier: false },
      { symbol: 'ZM', metric_x: -3.6, metric_y: 4.7, z_x: -0.9, z_y: 1.3, is_outlier: false },
      { symbol: 'DOCU', metric_x: 2.8, metric_y: -1.7, z_x: 0.7, z_y: -0.6, is_outlier: false },
      { symbol: 'CRWD', metric_x: -1.5, metric_y: 3.1, z_x: -0.4, z_y: 0.9, is_outlier: false },
      { symbol: 'OKTA', metric_x: 0.9, metric_y: -0.7, z_x: 0.2, z_y: -0.2, is_outlier: false },
      { symbol: 'SNOW', metric_x: -2.2, metric_y: 2.8, z_x: -0.6, z_y: 0.8, is_outlier: false },
      { symbol: 'PLTR', metric_x: 3.7, metric_y: 1.3, z_x: 0.9, z_y: 0.4, is_outlier: false },
      { symbol: 'DKNG', metric_x: -4.1, metric_y: 6.3, z_x: -1.0, z_y: 1.8, is_outlier: false },
      { symbol: 'PTON', metric_x: -6.2, metric_y: -3.8, z_x: -1.5, z_y: -1.4, is_outlier: false }
    ]
  };

  // Use mock data if API data is not available
  const displayData = data || mockData;
  const isUsingMockData = !data;
  
  // Ensure metrics array exists
  const metrics = displayData?.metrics || [];
  const hasValidData = Array.isArray(metrics) && metrics.length > 0;

  console.log('Performance metrics data:', data);
  console.log('Loading:', loading, 'Error:', error);
  console.log('Using mock data:', isUsingMockData);
  console.log('Display data:', displayData);
  console.log('Metrics:', metrics);
  console.log('Has valid data:', hasValidData);

  const [isRefreshing, setIsRefreshing] = useState(false);
  const [refreshProgress, setRefreshProgress] = useState(0);

  const handleRefresh = () => {
    refetch();
    addToast('Refreshing outlier data...', 'info');
  };

  const handleRefreshMarketData = async () => {
    try {
      setIsRefreshing(true);
      setRefreshProgress(0);
      addToast('Starting market data refresh...', 'info');

      // Trigger refresh
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/api/v1/market/refresh`, {
        method: 'POST',
      });

      if (!response.ok) {
        throw new Error('Failed to start refresh');
      }

      const result = await response.json();
      addToast(result.message, 'success');

      // Poll for status updates
      const pollStatus = setInterval(async () => {
        try {
          const statusResponse = await fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/api/v1/market/refresh/status`);
          const status = await statusResponse.json();

          setRefreshProgress(status.progress || 0);

          if (!status.is_running) {
            clearInterval(pollStatus);
            setIsRefreshing(false);
            setRefreshProgress(100);
            addToast('Market data refresh completed!', 'success');
            // Refresh the UI data
            refetch();
          }
        } catch (err) {
          console.error('Error polling refresh status:', err);
        }
      }, 2000); // Poll every 2 seconds

    } catch (err) {
      console.error('Error refreshing market data:', err);
      addToast('Failed to refresh market data', 'error');
      setIsRefreshing(false);
    }
  };

  return (
    <>
      {/* Strategy Selector */}
      <Card>
        <CardHeader>
          <CardTitle>Select Strategy</CardTitle>
          <CardDescription>
            Choose a trading timeframe to analyze outliers
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <Select value={strategy} onValueChange={setStrategy}>
            <SelectTrigger className="w-full md:w-[300px]">
              <SelectValue placeholder="Select strategy" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="scalp">
                <div className="flex flex-col">
                  <span className="font-semibold">Scalp</span>
                  <span className="text-xs text-muted-foreground">
                    Y: 1-Month % | X: 1-Week %
                  </span>
                </div>
              </SelectItem>
              <SelectItem value="swing">
                <div className="flex flex-col">
                  <span className="font-semibold">Swing</span>
                  <span className="text-xs text-muted-foreground">
                    Y: 3-Month % | X: 1-Month %
                  </span>
                </div>
              </SelectItem>
              <SelectItem value="longterm">
                <div className="flex flex-col">
                  <span className="font-semibold">Longterm</span>
                  <span className="text-xs text-muted-foreground">
                    Y: 1-Year % | X: 6-Month %
                  </span>
                </div>
              </SelectItem>
            </SelectContent>
          </Select>

          <div className="flex gap-2 flex-wrap items-center">
            <Badge variant="outline">Z-Score &gt; 2</Badge>
            {displayData && <Badge variant="secondary">{metrics.filter(m => m.is_outlier).length} outliers found</Badge>}
            {isUsingMockData && <Badge variant="destructive">Using Mock Data</Badge>}
            <Button
              onClick={handleRefresh}
              disabled={loading}
              variant="outline"
              size="sm"
            >
              {loading ? 'Refreshing...' : 'Refresh UI'}
            </Button>
            <Button
              onClick={handleRefreshMarketData}
              disabled={isRefreshing}
              variant="default"
              size="sm"
            >
              {isRefreshing ? `Refreshing... ${refreshProgress}%` : 'Refresh Market Data'}
            </Button>
            <Button
              onClick={() => {
                setAutoRefresh(!autoRefresh);
                addToast(
                  autoRefresh ? 'Auto-refresh disabled' : 'Auto-refresh enabled (5 min)',
                  'info'
                );
              }}
              variant={autoRefresh ? 'default' : 'outline'}
              size="sm"
            >
              Auto-refresh: {autoRefresh ? 'ON' : 'OFF'}
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Scatter Plot */}
      <Card>
        <CardHeader>
          <CardTitle>Performance Scatter Plot</CardTitle>
          <CardDescription>
            Outliers shown in red (|z-score| &gt; 2)
          </CardDescription>
        </CardHeader>
        <CardContent>
          {loading ? (
            <Skeleton className="w-full aspect-square" />
          ) : hasValidData ? (
            (() => {
              try {
                const plotData = metrics.map(o => ({
                  symbol: o.symbol,
                  x: o.metric_x || 0,
                  y: o.metric_y || 0,
                  isOutlier: o.is_outlier
                }));
                console.log('Plot data:', plotData);
                
                // Determine axis labels based on strategy
                // X-axis = shorter timeframe, Y-axis = longer timeframe
                const axisLabels = {
                  scalp: { x: '1-Week Performance (%)', y: '1-Month Performance (%)' },
                  swing: { x: '1-Month Performance (%)', y: '3-Month Performance (%)' },
                  longterm: { x: '6-Month Performance (%)', y: '1-Year Performance (%)' }
                };
                
                const labels = axisLabels[strategy as keyof typeof axisLabels] || { x: 'X-axis %', y: 'Y-axis %' };
                
                return (
                  <PlotlyScatterPlot
                    data={plotData}
                    xLabel={labels.x}
                    yLabel={labels.y}
                    title={`${strategy.charAt(0).toUpperCase() + strategy.slice(1)} Strategy - All Stocks Analysis`}
                  />
                );
              } catch (err) {
                console.error('Error creating plot data:', err);
                return (
                  <div className="aspect-square bg-red-50 border border-red-200 rounded-lg flex items-center justify-center">
                    <p className="text-red-600">Error creating scatter plot: {err instanceof Error ? err.message : 'Unknown error'}</p>
                  </div>
                );
              }
            })()
          ) : (
            <div className="aspect-square bg-muted/20 rounded-lg flex items-center justify-center">
              <p className="text-muted-foreground">
                {isUsingMockData ? 'Using mock data - API data not available' : 'No data available'}
              </p>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Outliers Table */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle>Detected Outliers</CardTitle>
              <CardDescription>
                Stocks with exceptional performance patterns
              </CardDescription>
            </div>
            {loading ? (
              <Badge variant="secondary">Loading...</Badge>
            ) : error ? (
              <Badge variant="destructive">Error</Badge>
            ) : (
              <div className="flex gap-2">
                <Badge variant="default">{displayData?.count || 0} stocks total</Badge>
                {isUsingMockData && (
                  <Badge variant="outline" className="text-orange-600 border-orange-600">
                    Using Mock Data
                  </Badge>
                )}
              </div>
            )}
          </div>
        </CardHeader>
        <CardContent>
          {loading ? (
            <div className="space-y-2">
              {[1, 2, 3].map((i) => (
                <Skeleton key={i} className="h-12 w-full" />
              ))}
            </div>
          ) : error ? (
            <div className="text-destructive text-sm">
              Failed to load outliers. Make sure backend is running.
            </div>
          ) : hasValidData && metrics.filter(m => m.is_outlier).length > 0 ? (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Symbol</TableHead>
                  <TableHead className="text-right">X Metric</TableHead>
                  <TableHead className="text-right">Y Metric</TableHead>
                  <TableHead className="text-right">Z-Score X</TableHead>
                  <TableHead className="text-right">Z-Score Y</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {metrics.filter(m => m.is_outlier).slice(0, 10).map((outlier) => (
                  <TableRow key={outlier.symbol}>
                    <TableCell className="font-semibold">{outlier.symbol}</TableCell>
                    <TableCell className="text-right">
                      {outlier.metric_x?.toFixed(2)}%
                    </TableCell>
                    <TableCell className="text-right">
                      {outlier.metric_y?.toFixed(2)}%
                    </TableCell>
                    <TableCell className="text-right">
                      <Badge variant={Math.abs(outlier.z_x || 0) > 2 ? "destructive" : "outline"}>
                        {outlier.z_x?.toFixed(2)}
                      </Badge>
                    </TableCell>
                    <TableCell className="text-right">
                      <Badge variant={Math.abs(outlier.z_y || 0) > 2 ? "destructive" : "outline"}>
                        {outlier.z_y?.toFixed(2)}
                      </Badge>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          ) : (
            <div className="text-center py-8 text-muted-foreground">
              No outliers found for {strategy} strategy.
              <br />
              <Button variant="outline" size="sm" className="mt-4">
                Refresh Data
              </Button>
            </div>
          )}
        </CardContent>
      </Card>
    </>
  );
}

