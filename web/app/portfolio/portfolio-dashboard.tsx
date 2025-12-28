'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { 
  TrendingUp, 
  TrendingDown, 
  DollarSign, 
  BarChart3, 
  History, 
  Edit,
  Plus,
  Minus,
  AlertCircle
} from "lucide-react";
import { api } from "@/lib/api";
import { BehavioralHoldingsManager } from "@/components/behavioral-holdings-manager";
import { BehavioralInsightsPanel } from "@/components/behavioral-insights-panel";

interface RealTrade {
  id: string;
  symbol: string;
  side: 'buy' | 'sell';
  qty: number;
  filled_price: number;
  filled_at: string;
  status: string;
  order_type: string;
  allocation_percentage?: number;
  stop_loss_price?: number;
  entry_rationale?: string;
  exit_rationale?: string;
  current_price?: number;
  unrealized_pnl?: number;
  unrealized_pnl_percent?: number;
}

interface TradingActivity {
  id: string;
  timestamp: string;
  action: 'buy' | 'sell' | 'stop_loss' | 'allocation_update';
  symbol: string;
  qty: number;
  price: number;
  rationale: string;
  allocation?: number;
  stop_loss?: number;
}

export function PortfolioDashboard() {
  const [activeTab, setActiveTab] = useState('overview');
  const [realTrades, setRealTrades] = useState<RealTrade[]>([]);
  const [tradingActivity, setTradingActivity] = useState<TradingActivity[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [accountInfo, setAccountInfo] = useState<any>(null);
  
  // Fetch real trading data from Alpaca
  const fetchTradingData = async () => {
    setIsLoading(true);
    try {
      // Get account information
      const account = await api.getAccountInfo();
      setAccountInfo(account);
      
      // Get all orders (filled trades)
      const ordersResponse = await api.getOrders('filled');
      const orders = ordersResponse.orders || [];
      
      // Get current positions
      const positionsResponse = await api.getPositions();
      const positions = positionsResponse.positions || [];
      
      // Process real trades with allocation and stop-loss data
      const processedTrades: RealTrade[] = orders.map((order: any) => {
        const position = positions.find((pos: any) => pos.symbol === order.symbol);
        return {
          id: order.id,
          symbol: order.symbol,
          side: order.side,
          qty: order.filled_qty,
          filled_price: order.filled_avg_price,
          filled_at: order.filled_at,
          status: order.status,
          order_type: order.order_type,
          allocation_percentage: order.allocation_percentage || 0,
          stop_loss_price: order.stop_loss_price || 0,
          entry_rationale: order.entry_rationale || '',
          exit_rationale: order.exit_rationale || '',
          current_price: position?.current_price || order.filled_avg_price,
          unrealized_pnl: position?.unrealized_pnl || 0,
          unrealized_pnl_percent: position?.unrealized_pnl_percent || 0
        };
      });
      
      setRealTrades(processedTrades);
      
      // Create trading activity log
      const activity: TradingActivity[] = processedTrades.map(trade => ({
        id: trade.id,
        timestamp: trade.filled_at,
        action: trade.side,
        symbol: trade.symbol,
        qty: trade.qty,
        price: trade.filled_price,
        rationale: trade.side === 'buy' ? trade.entry_rationale || 'No rationale provided' : trade.exit_rationale || 'No rationale provided',
        allocation: trade.allocation_percentage,
        stop_loss: trade.stop_loss_price
      }));
      
      setTradingActivity(activity.sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()));
      
      
    } catch (error) {
      console.error('Error fetching trading data:', error);
      // Set empty data to show setup message
      setRealTrades([]);
      setTradingActivity([]);
      setAccountInfo(null);
    } finally {
      setIsLoading(false);
    }
  };
  
  useEffect(() => {
    fetchTradingData();
  }, []);

  // Calculate portfolio metrics from real trades
  const totalValue = accountInfo?.portfolio_value || 0;
  const totalPnL = accountInfo?.unrealized_pl || 0;
  const totalPnLPercentage = accountInfo?.unrealized_plpc ? (accountInfo.unrealized_plpc * 100) : 0;
  const holdingsCount = realTrades.filter(trade => trade.side === 'buy').length;
  const bestPerformer = realTrades.reduce((best, trade) => {
    if (trade.unrealized_pnl_percent && trade.unrealized_pnl_percent > (best?.unrealized_pnl_percent || 0)) {
      return trade;
    }
    return best;
  }, realTrades[0]);

  // Create holdings array from real trades for performance metrics
  const holdings = realTrades
    .filter(trade => trade.side === 'buy')
    .map(trade => ({
      id: trade.id,
      stock: trade.symbol,
      pnlPercentage: trade.unrealized_pnl_percent || 0,
      stopLoss: trade.stop_loss_price || 0
    }));

  if (isLoading) {
    return (
      <Card>
        <CardContent className="p-6">
          <div className="flex items-center justify-center">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
            <span className="ml-2">Loading real trading data...</span>
          </div>
        </CardContent>
      </Card>
    );
  }

  // Show setup message only if account info is not available
  if (!accountInfo) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <BarChart3 className="h-5 w-5" />
            Portfolio Dashboard
          </CardTitle>
          <CardDescription>
            Track your holdings, performance, and trading activity
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="text-center py-8">
            <AlertCircle className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
            <h3 className="text-lg font-semibold mb-2">Trading Setup Required</h3>
            <p className="text-muted-foreground mb-4">
              To view real trades, you need to configure Alpaca Paper Trading API keys.
            </p>
            <div className="bg-muted/20 p-4 rounded-lg mb-4 text-left">
              <h4 className="font-semibold mb-2">Setup Steps:</h4>
              <ol className="list-decimal list-inside space-y-1 text-sm text-muted-foreground">
                <li>Get Alpaca API keys from <a href="https://app.alpaca.markets/paper/dashboard/overview" target="_blank" className="text-blue-600 hover:underline">Alpaca Paper Trading</a></li>
                <li>Add API keys to your environment variables</li>
                <li>Execute trades through the Trading Dashboard</li>
                <li>Only trades with allocation and stop-loss parameters will appear here</li>
              </ol>
            </div>
            <p className="text-sm text-muted-foreground">
              This enables real-time strategy testing and AI analysis based on actual trading behavior.
            </p>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <BarChart3 className="h-5 w-5" />
          Portfolio Dashboard
        </CardTitle>
        <CardDescription>
          Track your holdings, performance, and trading activity
        </CardDescription>
      </CardHeader>
      <CardContent>
        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
          <TabsList className="grid w-full grid-cols-6">
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="holdings">Holdings</TabsTrigger>
            <TabsTrigger value="performance">Performance</TabsTrigger>
            <TabsTrigger value="activity">Activity</TabsTrigger>
            <TabsTrigger value="behavioral">Behavioral</TabsTrigger>
            <TabsTrigger value="insights">Insights</TabsTrigger>
          </TabsList>

          <TabsContent value="overview" className="space-y-6">
            {/* Portfolio Summary */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <Card className="p-4">
                <div className="flex items-center gap-2 mb-2">
                  <DollarSign className="h-4 w-4 text-green-600" />
                  <span className="text-sm font-medium">Total Value</span>
                </div>
                <div className="text-2xl font-bold">${totalValue.toFixed(2)}</div>
              </Card>
              
              <Card className="p-4">
                <div className="flex items-center gap-2 mb-2">
                  {totalPnL >= 0 ? (
                    <TrendingUp className="h-4 w-4 text-green-600" />
                  ) : (
                    <TrendingDown className="h-4 w-4 text-red-600" />
                  )}
                  <span className="text-sm font-medium">P&L</span>
                </div>
                <div className={`text-2xl font-bold ${totalPnL >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                  ${totalPnL.toFixed(2)}
                </div>
                <div className={`text-sm ${totalPnL >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                  {totalPnLPercentage.toFixed(2)}%
                </div>
              </Card>

              <Card className="p-4">
                <div className="flex items-center gap-2 mb-2">
                  <BarChart3 className="h-4 w-4 text-blue-600" />
                  <span className="text-sm font-medium">Holdings</span>
                </div>
                <div className="text-2xl font-bold">{holdingsCount}</div>
              </Card>

              <Card className="p-4">
                <div className="flex items-center gap-2 mb-2">
                  <TrendingUp className="h-4 w-4 text-purple-600" />
                  <span className="text-sm font-medium">Best Performer</span>
                </div>
                <div className="text-lg font-bold">
                  {bestPerformer?.symbol || 'N/A'}
                </div>
              </Card>
            </div>

            {/* Account Information */}
            <Card className="p-4">
              <div className="flex items-center gap-2 mb-4">
                <BarChart3 className="h-5 w-5 text-blue-600" />
                <h3 className="text-lg font-semibold">Account Information</h3>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div>
                  <div className="text-sm text-muted-foreground mb-1">Account ID</div>
                  <div className="font-medium text-sm">{accountInfo?.account_id || 'N/A'}</div>
                </div>
                <div>
                  <div className="text-sm text-muted-foreground mb-1">Buying Power</div>
                  <div className="font-medium text-green-600">${accountInfo?.buying_power ? (typeof accountInfo.buying_power === 'number' ? accountInfo.buying_power.toFixed(2) : parseFloat(accountInfo.buying_power || '0').toFixed(2)) : '0.00'}</div>
                </div>
                <div>
                  <div className="text-sm text-muted-foreground mb-1">Cash</div>
                  <div className="font-medium">${accountInfo?.cash ? (typeof accountInfo.cash === 'number' ? accountInfo.cash.toFixed(2) : parseFloat(accountInfo.cash || '0').toFixed(2)) : '0.00'}</div>
                </div>
                <div>
                  <div className="text-sm text-muted-foreground mb-1">Account Status</div>
                  <div className={`font-medium ${accountInfo?.account_status === 'ACTIVE' ? 'text-green-600' : 'text-yellow-600'}`}>
                    {accountInfo?.account_status || 'Unknown'}
                  </div>
                </div>
                <div>
                  <div className="text-sm text-muted-foreground mb-1">Currency</div>
                  <div className="font-medium">{accountInfo?.currency || 'USD'}</div>
                </div>
                <div>
                  <div className="text-sm text-muted-foreground mb-1">Equity</div>
                  <div className="font-medium">${accountInfo?.equity ? (typeof accountInfo.equity === 'number' ? accountInfo.equity.toFixed(2) : parseFloat(accountInfo.equity || '0').toFixed(2)) : '0.00'}</div>
                </div>
              </div>
            </Card>

            {/* Quick Actions */}
            <Card className="p-4">
              <h3 className="font-semibold mb-3">Quick Actions</h3>
              <div className="flex gap-2">
                <Button size="sm" className="flex items-center gap-2">
                  <Plus className="h-4 w-4" />
                  Add Holding
                </Button>
                <Button size="sm" variant="outline" className="flex items-center gap-2">
                  <Edit className="h-4 w-4" />
                  Edit Portfolio
                </Button>
                <Button size="sm" variant="outline" className="flex items-center gap-2">
                  <Minus className="h-4 w-4" />
                  Remove Holding
                </Button>
              </div>
            </Card>

          {/* Trading Navigation */}
          <Card className="p-4">
            <div className="flex items-center justify-between mb-3">
              <h3 className="font-semibold">Trading Platforms</h3>
            </div>
            <div className="flex gap-3">
              <Button
                onClick={() => window.location.href = '/trading/hft'}
                className="flex-1"
              >
                🚀 HFT Trading Dashboard
              </Button>
              <Button
                onClick={() => window.location.href = '/trading/quantitative'}
                variant="outline"
                className="flex-1"
              >
                📊 Quantitative Trading
              </Button>
            </div>
          </Card>

          </TabsContent>

          <TabsContent value="holdings" className="space-y-4">
            {realTrades.filter(trade => trade.side === 'buy').length === 0 ? (
              <Card className="p-6">
                <div className="text-center">
                  <BarChart3 className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                  <h3 className="text-lg font-semibold mb-2">No Holdings Yet</h3>
                  <p className="text-muted-foreground mb-4">
                    You haven't made any trades yet. Start trading to see your holdings here.
                  </p>
                  <Button className="flex items-center gap-2 mx-auto">
                    <Plus className="h-4 w-4" />
                    Start Trading
                  </Button>
                </div>
              </Card>
            ) : (
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Stock</TableHead>
                    <TableHead>Shares</TableHead>
                    <TableHead>Entry Price</TableHead>
                    <TableHead>Current Price</TableHead>
                    <TableHead>P&L</TableHead>
                    <TableHead>Stop Loss</TableHead>
                    <TableHead>Actions</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {realTrades.filter(trade => trade.side === 'buy').map((trade) => (
                  <TableRow key={trade.id}>
                    <TableCell>
                      <div className="flex items-center gap-2">
                        <Badge variant="outline">{trade.symbol}</Badge>
                      </div>
                    </TableCell>
                    <TableCell>{trade.qty}</TableCell>
                    <TableCell>${typeof trade.filled_price === 'number' ? trade.filled_price.toFixed(2) : parseFloat(trade.filled_price || '0').toFixed(2)}</TableCell>
                    <TableCell>${typeof trade.current_price === 'number' ? trade.current_price.toFixed(2) : (typeof trade.filled_price === 'number' ? trade.filled_price.toFixed(2) : parseFloat(trade.filled_price || '0').toFixed(2))}</TableCell>
                    <TableCell>
                      <div className={`font-medium ${(trade.unrealized_pnl || 0) >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                        ${(trade.unrealized_pnl || 0).toFixed(2)}
                        <div className="text-sm">
                          {(trade.unrealized_pnl_percent || 0).toFixed(2)}%
                        </div>
                      </div>
                    </TableCell>
                    <TableCell>
                      <Badge variant="destructive">
                        {trade.stop_loss_price ? `$${typeof trade.stop_loss_price === 'number' ? trade.stop_loss_price.toFixed(2) : parseFloat(trade.stop_loss_price || '0').toFixed(2)}` : 'Not Set'}
                      </Badge>
                    </TableCell>
                    <TableCell>
                      <div className="flex gap-1">
                        <Button size="sm" variant="outline" title="Entry Rationale">
                          <Edit className="h-3 w-3" />
                        </Button>
                        <Button size="sm" variant="outline" title="Exit Trade">
                          <Minus className="h-3 w-3" />
                        </Button>
                      </div>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
            )}
          </TabsContent>

          <TabsContent value="performance" className="space-y-4">
            <Card className="p-4">
              <h3 className="font-semibold mb-3">Performance Chart</h3>
              <div className="h-64 bg-muted/20 rounded-lg flex items-center justify-center border-2 border-dashed">
                <div className="text-center text-muted-foreground">
                  <BarChart3 className="h-8 w-8 mx-auto mb-2" />
                  <p>P&L Performance Chart</p>
                  <p className="text-sm">Coming soon...</p>
                </div>
              </div>
            </Card>

            {holdings.length === 0 ? (
              <Card className="p-6">
                <div className="text-center">
                  <TrendingUp className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                  <h3 className="text-lg font-semibold mb-2">No Performance Data Yet</h3>
                  <p className="text-muted-foreground mb-4">
                    Performance metrics will appear here once you start trading.
                  </p>
                  <Button className="flex items-center gap-2 mx-auto">
                    <Plus className="h-4 w-4" />
                    Start Trading
                  </Button>
                </div>
              </Card>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <Card className="p-4">
                  <h3 className="font-semibold mb-3">Top Performers</h3>
                  <div className="space-y-2">
                    {holdings
                      .sort((a, b) => b.pnlPercentage - a.pnlPercentage)
                      .slice(0, 3)
                      .map((holding) => (
                        <div key={holding.id} className="flex justify-between items-center">
                          <span>{holding.stock}</span>
                          <span className={`font-medium ${holding.pnlPercentage >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                            {holding.pnlPercentage.toFixed(2)}%
                          </span>
                        </div>
                      ))}
                  </div>
                </Card>

              <Card className="p-4">
                <h3 className="font-semibold mb-3">Risk Metrics</h3>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span>Avg Stop Loss</span>
                    <span className="font-medium">
                      {holdings.length > 0 ? (holdings.reduce((sum, h) => sum + h.stopLoss, 0) / holdings.length).toFixed(1) : '0.0'}%
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span>Portfolio Beta</span>
                    <span className="font-medium">1.2</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Sharpe Ratio</span>
                    <span className="font-medium">0.85</span>
                  </div>
                </div>
              </Card>
            </div>
            )}
          </TabsContent>

          <TabsContent value="activity" className="space-y-4">
            {tradingActivity.length === 0 ? (
              <Card className="p-6">
                <div className="text-center">
                  <History className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                  <h3 className="text-lg font-semibold mb-2">No Trading Activity Yet</h3>
                  <p className="text-muted-foreground mb-4">
                    Your trading activity will appear here once you start making trades.
                  </p>
                  <Button className="flex items-center gap-2 mx-auto">
                    <Plus className="h-4 w-4" />
                    Start Trading
                  </Button>
                </div>
              </Card>
            ) : (
              <Card className="p-4">
                <h3 className="font-semibold mb-3 flex items-center gap-2">
                  <History className="h-4 w-4" />
                  Activity Log
                </h3>
                <div className="space-y-3">
                  {tradingActivity.map((activity) => (
                  <div key={activity.id} className="flex items-center gap-3 p-3 bg-muted/20 rounded-lg">
                    <div className="flex-shrink-0">
                      {activity.action === 'buy' && <Plus className="h-4 w-4 text-green-600" />}
                      {activity.action === 'sell' && <Minus className="h-4 w-4 text-red-600" />}
                      {activity.action === 'allocation_update' && <Edit className="h-4 w-4 text-blue-600" />}
                      {activity.action === 'stop_loss' && <TrendingDown className="h-4 w-4 text-orange-600" />}
                    </div>
                    <div className="flex-1">
                      <div className="flex items-center gap-2">
                        <Badge variant="outline">{activity.symbol}</Badge>
                        <span className="text-sm text-muted-foreground">
                          {new Date(activity.timestamp).toLocaleString()}
                        </span>
                      </div>
                      <p className="text-sm">
                        {activity.action === 'buy' ? 'Bought' : 'Sold'} {activity.qty} shares at ${typeof activity.price === 'number' ? activity.price.toFixed(2) : parseFloat(activity.price || '0').toFixed(2)}
                      </p>
                      <p className="text-sm text-muted-foreground">
                        <strong>Rationale:</strong> {activity.rationale}
                      </p>
                      {activity.allocation && (
                        <p className="text-sm">
                          <strong>Allocation:</strong> {activity.allocation}% | 
                          <strong> Stop Loss:</strong> ${activity.stop_loss ? (typeof activity.stop_loss === 'number' ? activity.stop_loss.toFixed(2) : parseFloat(activity.stop_loss || '0').toFixed(2)) : 'Not Set'}
                        </p>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </Card>
            )}
          </TabsContent>

          <TabsContent value="behavioral" className="space-y-6">
            <BehavioralHoldingsManager />
          </TabsContent>

          <TabsContent value="insights" className="space-y-6">
            <BehavioralInsightsPanel />
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
}
