'use client';

import { useState, useEffect } from 'react';
import { api } from '@/lib/api';
import { useOrderBook } from '@/hooks/use-orderbook';
import { useToast } from '@/components/toast-provider';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Badge } from '@/components/ui/badge';

export default function HftTradingPage() {
  // Toast notifications
  const { addToast } = useToast();
  
  // Order Management State
  const [hftSymbol, setHftSymbol] = useState('AAPL');
  const [hftSide, setHftSide] = useState<'buy' | 'sell'>('buy');
  const [hftOrderType, setHftOrderType] = useState<'market' | 'limit' | 'twap' | 'vwap'>('limit');
  const [hftQuantity, setHftQuantity] = useState(1);
  const [hftPrice, setHftPrice] = useState(150.00);
  const [hftTimeInForce, setHftTimeInForce] = useState<'day' | 'gtc' | 'fok' | 'ioc' | 'opg' | 'cls'>('day');
  const [orderSubmitting, setOrderSubmitting] = useState(false);

  // Performance Metrics State
  const [performanceMetrics, setPerformanceMetrics] = useState<any>(null);
  const [hftStatus, setHftStatus] = useState<any>(null);

  // Algorithm and Display State
  const [algorithm, setAlgorithm] = useState('Momentum');
  const [showOverlay, setShowOverlay] = useState(false);

  // Auto-sync symbol from Order Management to Order Book
  const { orderBook, isConnected, isMockData: orderBookIsMock } = useOrderBook(hftSymbol, !!hftSymbol);

  useEffect(() => {
    fetchHftStatus();
    fetchPerformanceMetrics();
  }, []);

  const fetchHftStatus = async () => {
    try {
      const status = await api.hftStatus();
      setHftStatus(status);
    } catch (error: any) {
      console.error('Failed to fetch HFT status:', error);
      addToast(`⚠️ Failed to fetch HFT status: ${error?.message || 'Unknown error'}`, 'error');
    }
  };

  const fetchPerformanceMetrics = async () => {
    try {
      const metrics = await api.hftPerformance();
      setPerformanceMetrics(metrics);
    } catch (error: any) {
      console.error('Failed to fetch performance metrics:', error);
      addToast(`⚠️ Failed to fetch performance metrics: ${error?.message || 'Unknown error'}`, 'error');
    }
  };

  const handleStartHft = async () => {
    try {
      addToast('Starting HFT Engine...', 'info');
      await api.hftStart();
      await fetchHftStatus();
      addToast('✅ HFT Engine started successfully!', 'success');
    } catch (error: any) {
      console.error('Failed to start HFT engine:', error);
      addToast(`❌ Failed to start HFT engine: ${error?.message || 'Unknown error'}`, 'error');
    }
  };

  const handleStopHft = async () => {
    try {
      addToast('Stopping HFT Engine...', 'info');
      await api.hftStop();
      await fetchHftStatus();
      addToast('✅ HFT Engine stopped successfully!', 'success');
    } catch (error: any) {
      console.error('Failed to stop HFT engine:', error);
      addToast(`❌ Failed to stop HFT engine: ${error?.message || 'Unknown error'}`, 'error');
    }
  };

  const handleClearOrders = async () => {
    try {
      addToast('Clearing all open orders...', 'info');
      const response = await api.hftClearAllOrders();
      const cancelledCount = response?.data?.cancelled_count || 0;
      
      if (cancelledCount > 0) {
        addToast(`✅ Successfully cancelled ${cancelledCount} orders!`, 'success');
        // Refresh status and metrics
        await fetchHftStatus();
        await fetchPerformanceMetrics();
      } else {
        addToast('ℹ️ No open orders to cancel', 'info');
      }
    } catch (error: any) {
      console.error('Failed to clear orders:', error);
      addToast(`❌ Failed to clear orders: ${error?.message || 'Unknown error'}`, 'error');
    }
  };

  const handleSubmitOrder = async () => {
    // Prevent double-clicks
    if (orderSubmitting) {
      addToast('Order already being processed, please wait...', 'info');
      return;
    }
    
    if (!hftSymbol || !hftQuantity) {
      addToast('Please enter symbol and quantity', 'error');
      return;
    }
    
    setOrderSubmitting(true);
    
    // Show loading toast
    addToast(`Submitting ${hftSide.toUpperCase()} ${hftOrderType.toUpperCase()} order for ${hftSymbol}...`, 'info');
    
    try {
      const orderData: any = {
        order_type: hftOrderType,
        symbol: hftSymbol.toUpperCase(),
        side: hftSide,
        quantity: hftQuantity,
      };

      if (hftOrderType === 'limit') {
        orderData.price = hftPrice;
        orderData.time_in_force = hftTimeInForce;
      } else if (hftOrderType === 'twap') {
        orderData.duration_minutes = 30;
        orderData.interval_seconds = 60;
      } else if (hftOrderType === 'vwap') {
        orderData.volume_weight = 0.5;
      }

      const response = await api.hftSubmitOrder(orderData);
      
      // Validate the response properly
      const orderId = response?.data?.order_id;
      
      if (!orderId || orderId === 'Unknown' || orderId === '') {
        // Order was rejected by Alpaca
        addToast(
          `❌ Order rejected by Alpaca: ${response?.data?.message || 'Invalid order ID returned'}`, 
          'error'
        );
      } else {
        // Valid order ID - order was accepted
        addToast(
          `✅ Order accepted by Alpaca! Order ID: ${orderId.substring(0, 8)}...`, 
          'success'
        );
        
        // Refresh status and metrics
        await fetchHftStatus();
        await fetchPerformanceMetrics();
      }
      
    } catch (error: any) {
      console.error('Failed to submit order:', error);
      
      // Error toast with details
      const errorMessage = error?.message || 'Unknown error occurred';
      addToast(
        `❌ Order failed: ${errorMessage}`, 
        'error'
      );
    } finally {
      setOrderSubmitting(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#0b0f14] text-[#cde7ff]">
      <div className="container mx-auto p-6 space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-[#00eaff]">HFT Trading Dashboard</h1>
            <p className="text-[#7fb7ff] mt-1">High-Frequency Trading with Advanced Order Types</p>
          </div>
          <div className="flex gap-3">
            <Button
              onClick={() => window.location.href = '/portfolio'}
              className="bg-[#1a2332] border-2 border-[#00eaff] text-[#00eaff] hover:bg-[#00eaff] hover:text-[#0b0f14] font-semibold transition-all duration-200 px-4"
            >
              🏠 Portfolio Dashboard
            </Button>
            <Button
              onClick={handleStartHft}
              disabled={hftStatus?.is_running}
              className="bg-green-600 text-white hover:bg-green-700"
            >
              Start Engine
            </Button>
            <Button
              onClick={handleStopHft}
              disabled={!hftStatus?.is_running}
              className="bg-red-600 text-white hover:bg-red-700"
            >
              Stop Engine
            </Button>
            <Button
              onClick={handleClearOrders}
              className="bg-orange-600 text-white hover:bg-orange-700"
            >
              Clear All Orders
            </Button>
          </div>
        </div>

        {/* Performance Metrics */}
        <div className="grid grid-cols-4 gap-4">
          <Card className="p-4 bg-[#0b0f14] border-[#16324a] text-[#cde7ff]">
            <div className="text-[#7fb7ff] text-sm">Total Trades</div>
            <div className="text-2xl font-bold text-[#00eaff]">
              {performanceMetrics?.total_trades || 0}
            </div>
          </Card>
          <Card className="p-4 bg-[#0b0f14] border-[#16324a] text-[#cde7ff]">
            <div className="text-[#7fb7ff] text-sm">Win Rate</div>
            <div className="text-2xl font-bold text-green-400">
              {performanceMetrics?.win_rate ? `${(performanceMetrics.win_rate * 100).toFixed(1)}%` : '0%'}
            </div>
          </Card>
          <Card className="p-4 bg-[#0b0f14] border-[#16324a] text-[#cde7ff]">
            <div className="text-[#7fb7ff] text-sm">Total P&L</div>
            <div className={`text-2xl font-bold ${(performanceMetrics?.total_pnl || 0) >= 0 ? 'text-green-400' : 'text-red-400'}`}>
              ${performanceMetrics?.total_pnl?.toFixed(2) || '0.00'}
            </div>
          </Card>
          <Card className="p-4 bg-[#0b0f14] border-[#16324a] text-[#cde7ff]">
            <div className="text-[#7fb7ff] text-sm">Avg Latency</div>
            <div className="text-2xl font-bold text-[#00eaff]">
              {performanceMetrics?.avg_latency ? `${performanceMetrics.avg_latency.toFixed(1)}ms` : '0ms'}
            </div>
          </Card>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Order Management */}
          <Card className="p-6 bg-[#0b0f14] border-[#16324a] text-[#cde7ff]">
            <h3 className="text-xl font-semibold text-[#00eaff] mb-4">Order Management</h3>
            
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm text-[#7fb7ff] mb-2">Symbol</label>
                  <Input
                    value={hftSymbol}
                    onChange={(e) => setHftSymbol(e.target.value.toUpperCase())}
                    placeholder="e.g., AAPL"
                    className="bg-[#0e1420] border-[#16324a] text-[#cde7ff]"
                  />
                </div>
                <div>
                  <label className="block text-sm text-[#7fb7ff] mb-2">Side</label>
                  <Select value={hftSide} onValueChange={(value: 'buy' | 'sell') => setHftSide(value)}>
                    <SelectTrigger className="bg-[#0e1420] border-[#16324a] text-[#cde7ff]">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent className="bg-[#0e1420] border-[#16324a]">
                      <SelectItem value="buy" className="text-[#cde7ff] hover:bg-[#16324a]">Buy</SelectItem>
                      <SelectItem value="sell" className="text-[#cde7ff] hover:bg-[#16324a]">Sell</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm text-[#7fb7ff] mb-2">Order Type</label>
                  <Select value={hftOrderType} onValueChange={(value: any) => setHftOrderType(value)}>
                    <SelectTrigger className="bg-[#0e1420] border-[#16324a] text-[#cde7ff]">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent className="bg-[#0e1420] border-[#16324a]">
                      <SelectItem value="market" className="text-[#cde7ff] hover:bg-[#16324a]">Market</SelectItem>
                      <SelectItem value="limit" className="text-[#cde7ff] hover:bg-[#16324a]">Limit</SelectItem>
                      <SelectItem value="twap" className="text-[#cde7ff] hover:bg-[#16324a]">TWAP</SelectItem>
                      <SelectItem value="vwap" className="text-[#cde7ff] hover:bg-[#16324a]">VWAP</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div>
                  <label className="block text-sm text-[#7fb7ff] mb-2">Quantity</label>
                  <Input
                    type="number"
                    value={hftQuantity}
                    onChange={(e) => setHftQuantity(Number(e.target.value))}
                    className="bg-[#0e1420] border-[#16324a] text-[#cde7ff]"
                  />
                </div>
              </div>

              {hftOrderType === 'limit' && (
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm text-[#7fb7ff] mb-2">Limit Price</label>
                    <div className="relative">
                      <span className="absolute left-3 top-1/2 transform -translate-y-1/2 text-[#7fb7ff]">$</span>
                      <Input
                        type="number"
                        step="0.01"
                        value={hftPrice}
                        onChange={(e) => setHftPrice(Number(e.target.value))}
                        className="bg-[#0e1420] border-[#16324a] text-[#cde7ff] pl-8"
                      />
                    </div>
                  </div>
                  <div>
                    <label className="block text-sm text-[#7fb7ff] mb-2">Time in Force</label>
                    <Select value={hftTimeInForce} onValueChange={(value: any) => setHftTimeInForce(value)}>
                      <SelectTrigger className="bg-[#0e1420] border-[#16324a] text-[#cde7ff]">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent className="bg-[#0e1420] border-[#16324a]">
                        <SelectItem value="day" className="text-[#cde7ff] hover:bg-[#16324a]">DAY</SelectItem>
                        <SelectItem value="gtc" className="text-[#cde7ff] hover:bg-[#16324a]">GTC</SelectItem>
                        <SelectItem value="fok" className="text-[#cde7ff] hover:bg-[#16324a]">FOK</SelectItem>
                        <SelectItem value="ioc" className="text-[#cde7ff] hover:bg-[#16324a]">IOC</SelectItem>
                        <SelectItem value="opg" className="text-[#cde7ff] hover:bg-[#16324a]">OPG</SelectItem>
                        <SelectItem value="cls" className="text-[#cde7ff] hover:bg-[#16324a]">CLS</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>
              )}

              <Button
                onClick={handleSubmitOrder}
                disabled={orderSubmitting || !hftSymbol || !hftQuantity}
                className="w-full bg-[#00eaff] text-[#0b0f14] hover:bg-[#00d4e6] font-semibold py-3"
              >
                {orderSubmitting ? 'Submitting...' : `Submit ${hftSide.toUpperCase()} ${hftOrderType.toUpperCase()} Order`}
              </Button>
            </div>
          </Card>

          {/* Order Book */}
          <Card className="p-6 bg-[#0b0f14] border-[#16324a] text-[#cde7ff]">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-3">
                <h3 className="text-xl font-semibold text-[#00eaff]">Order Book</h3>
                <div className="flex items-center gap-2">
                  <div className={`h-2 w-2 rounded-full ${orderBookIsMock ? 'bg-yellow-400 animate-pulse' : 'bg-green-400'}`}></div>
                  <span className="text-xs text-[#7fb7ff]">
                    {orderBookIsMock ? 'Mock Data' : 'Live Data'}
                  </span>
                </div>
                {hftSymbol && (
                  <div className="text-sm text-[#00eaff] font-mono">
                    {hftSymbol}
                  </div>
                )}
              </div>
              <div className="flex items-center gap-3">
                <Select value={algorithm} onValueChange={setAlgorithm}>
                  <SelectTrigger className="w-[160px] bg-[#0e1420] border-[#16324a] text-[#cde7ff]">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent className="bg-[#0e1420] border-[#16324a]">
                    <SelectItem value="Momentum" className="text-[#cde7ff] hover:bg-[#16324a]">Momentum</SelectItem>
                    <SelectItem value="Mean Reversion" className="text-[#cde7ff] hover:bg-[#16324a]">Mean Reversion</SelectItem>
                    <SelectItem value="VWAP" className="text-[#cde7ff] hover:bg-[#16324a]">VWAP</SelectItem>
                    <SelectItem value="TWAP" className="text-[#cde7ff] hover:bg-[#16324a]">TWAP</SelectItem>
                    <SelectItem value="Arbitrage" className="text-[#cde7ff] hover:bg-[#16324a]">Arbitrage</SelectItem>
                  </SelectContent>
                </Select>
                <Button
                  onClick={() => setShowOverlay(true)}
                  disabled={!hftSymbol}
                  className="bg-[#00eaff] text-[#0b0f14] hover:bg-[#00d4e6]"
                >
                  Full Dashboard
                </Button>
              </div>
            </div>

            {hftSymbol ? (
              <div className="space-y-4">
                {/* Current Price Display */}
                {orderBook && (
                  <div className="grid grid-cols-3 gap-4 text-center mb-4">
                    <div className="bg-[#0e1420] border border-[#16324a] rounded p-3">
                      <div className="text-xs text-[#7fb7ff] mb-1">HIGHEST BID</div>
                      <div className="text-green-400 font-mono text-lg font-bold">
                        ${orderBook.bids[0]?.price.toFixed(2) || '0.00'}
                      </div>
                    </div>
                    <div className="bg-[#0e1420] border border-[#16324a] rounded p-3">
                      <div className="text-xs text-[#7fb7ff] mb-1">MID PRICE</div>
                      <div className="text-[#cde7ff] font-mono text-lg font-bold">
                        ${((orderBook.bids[0]?.price || 0) + (orderBook.asks[0]?.price || 0) / 2).toFixed(2)}
                      </div>
                    </div>
                    <div className="bg-[#0e1420] border border-[#16324a] rounded p-3">
                      <div className="text-xs text-[#7fb7ff] mb-1">LOWEST ASK</div>
                      <div className="text-red-400 font-mono text-lg font-bold">
                        ${orderBook.asks[0]?.price.toFixed(2) || '0.00'}
                      </div>
                    </div>
                  </div>
                )}

                 {/* Order Book Table */}
                 {orderBook ? (
                   <div className="grid grid-cols-2 gap-4">
                     {/* Bids */}
                     <div>
                       <div className="text-center text-sm font-semibold text-green-400 mb-2">BIDS</div>
                       <div className="text-xs text-[#7fb7ff] mb-2 px-2 flex justify-between">
                         <span>Price</span>
                         <span>Size</span>
                         <span>Total</span>
                       </div>
                       <div className="space-y-1">
                         {orderBook.bids.slice(0, 8).map((bid, index) => {
                           const maxSize = Math.max(...orderBook.bids.slice(0, 8).map(b => b.size));
                           const barWidth = (bid.size / maxSize) * 100;
                           return (
                             <div key={index} className="relative p-2 bg-[#0e1420]/50 rounded hover:bg-[#0e1420] transition-colors overflow-hidden">
                               {/* Horizontal bar background */}
                               <div className="absolute left-0 top-0 bottom-0 bg-gradient-to-r from-green-400/30 to-green-400/10 rounded-l transition-all duration-300" style={{width: `${barWidth}%`}}></div>
                               <div className="relative flex justify-between items-center z-10">
                                 <div className="text-green-400 font-mono text-sm">${bid.price.toFixed(2)}</div>
                                 <div className="text-[#cde7ff] font-mono text-sm">{bid.size.toFixed(0)}</div>
                                 <div className="text-[#7fb7ff] font-mono text-xs">{bid.total?.toFixed(0) || '0'}</div>
                               </div>
                             </div>
                           );
                         })}
                       </div>
                     </div>

                     {/* Asks */}
                     <div>
                       <div className="text-center text-sm font-semibold text-red-400 mb-2">ASKS</div>
                       <div className="text-xs text-[#7fb7ff] mb-2 px-2 flex justify-between">
                         <span>Price</span>
                         <span>Size</span>
                         <span>Total</span>
                       </div>
                       <div className="space-y-1">
                         {orderBook.asks.slice(0, 8).map((ask, index) => {
                           const maxSize = Math.max(...orderBook.asks.slice(0, 8).map(a => a.size));
                           const barWidth = (ask.size / maxSize) * 100;
                           return (
                             <div key={index} className="relative p-2 bg-[#0e1420]/50 rounded hover:bg-[#0e1420] transition-colors overflow-hidden">
                               {/* Horizontal bar background */}
                               <div className="absolute right-0 top-0 bottom-0 bg-gradient-to-l from-red-400/30 to-red-400/10 rounded-r transition-all duration-300" style={{width: `${barWidth}%`}}></div>
                               <div className="relative flex justify-between items-center z-10">
                                 <div className="text-red-400 font-mono text-sm">${ask.price.toFixed(2)}</div>
                                 <div className="text-[#cde7ff] font-mono text-sm">{ask.size.toFixed(0)}</div>
                                 <div className="text-[#7fb7ff] font-mono text-xs">{ask.total?.toFixed(0) || '0'}</div>
                               </div>
                             </div>
                           );
                         })}
                       </div>
                     </div>
                   </div>
                ) : (
                  <div className="text-center text-[#7fb7ff] py-8">
                    <div className="text-4xl mb-2">📊</div>
                    <p>Loading order book...</p>
                  </div>
                )}
              </div>
            ) : (
              <div className="text-center text-[#7fb7ff] py-8">
                <div className="text-4xl mb-2">📊</div>
                <p>Enter a symbol in Order Management to see the order book</p>
                <p className="text-xs text-[#7fb7ff]/70 mt-2">
                  {orderBookIsMock ? 'Note: Using mock data due to API key issues' : 'Real-time market data available'}
                </p>
              </div>
            )}
          </Card>
        </div>

        {/* Full Screen Overlay */}
        {showOverlay && (
          <div className="fixed inset-0 z-50 bg-[#0b0f14]/95 text-[#cde7ff]" style={{backdropFilter: 'blur(2px)'}}>
            <div className="h-full flex flex-col">
              <div className="flex items-center justify-between p-6 border-b border-[#16324a]">
                <h2 className="text-2xl font-bold text-[#00eaff]">Full HFT Dashboard</h2>
                <Button
                  onClick={() => setShowOverlay(false)}
                  className="bg-red-600 text-white hover:bg-red-700"
                >
                  Close
                </Button>
              </div>
              <div className="flex-1 p-6 overflow-auto">
                <div className="text-center text-[#7fb7ff] py-20">
                  <div className="text-6xl mb-4">🚀</div>
                  <h3 className="text-2xl font-bold text-[#00eaff] mb-2">Full Dashboard Coming Soon</h3>
                  <p>Advanced order book visualization, real-time charts, and more trading tools will be available here.</p>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
