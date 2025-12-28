'use client';

import { useState, useEffect } from 'react';
import { api } from '@/lib/api';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Badge } from '@/components/ui/badge';
import { toast } from 'sonner';

export default function QuantitativeTradingPage() {
  const [selectedStrategy, setSelectedStrategy] = useState('momentum');
  const [selectedSymbols, setSelectedSymbols] = useState<string[]>([]);
  const [newSymbol, setNewSymbol] = useState('');
  const [strategyParams, setStrategyParams] = useState({
    lookbackPeriod: 20,
    threshold: 0.02,
    maxPosition: 1000,
    stopLoss: 0.05,
    takeProfit: 0.10
  });
  const [backtestResults, setBacktestResults] = useState<any>(null);
  const [isRunning, setIsRunning] = useState(false);

  const strategies = [
    { value: 'momentum', label: 'Momentum Strategy', description: 'Trades based on price momentum and trend following' },
    { value: 'mean_reversion', label: 'Mean Reversion', description: 'Trades when prices deviate from their average' },
    { value: 'pairs_trading', label: 'Pairs Trading', description: 'Trades correlated pairs when they diverge' },
    { value: 'arbitrage', label: 'Statistical Arbitrage', description: 'Exploits price discrepancies between related assets' },
    { value: 'market_making', label: 'Market Making', description: 'Provides liquidity and captures bid-ask spreads' }
  ];

  const addSymbol = () => {
    if (newSymbol && !selectedSymbols.includes(newSymbol.toUpperCase())) {
      setSelectedSymbols([...selectedSymbols, newSymbol.toUpperCase()]);
      setNewSymbol('');
    }
  };

  const removeSymbol = (symbol: string) => {
    setSelectedSymbols(selectedSymbols.filter(s => s !== symbol));
  };

  const runBacktest = async () => {
    if (selectedSymbols.length === 0) {
      toast.error('Please select at least one symbol');
      return;
    }

    setIsRunning(true);
    try {
      // Simulate backtest - replace with actual API call
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      const mockResults = {
        totalReturn: Math.random() * 50 - 10, // -10% to 40%
        sharpeRatio: Math.random() * 3,
        maxDrawdown: Math.random() * 20,
        winRate: Math.random() * 40 + 40, // 40% to 80%
        totalTrades: Math.floor(Math.random() * 100) + 10,
        avgTradeReturn: Math.random() * 2 - 0.5
      };
      
      setBacktestResults(mockResults);
      toast.success('Backtest completed successfully');
    } catch (error) {
      toast.error('Backtest failed');
    } finally {
      setIsRunning(false);
    }
  };

  return (
    <div className="container mx-auto p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-white">Quantitative Trading</h1>
          <p className="text-muted-foreground mt-2">Algorithmic strategies and systematic trading approaches</p>
        </div>
        <Badge variant="outline" className="text-lg px-4 py-2">
          Strategy Lab
        </Badge>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Strategy Selection */}
        <Card className="p-6">
          <h3 className="text-xl font-semibold mb-4">Strategy Configuration</h3>
          
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-2">Trading Strategy</label>
              <Select value={selectedStrategy} onValueChange={setSelectedStrategy}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {strategies.map(strategy => (
                    <SelectItem key={strategy.value} value={strategy.value}>
                      {strategy.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <p className="text-sm text-muted-foreground mt-1">
                {strategies.find(s => s.value === selectedStrategy)?.description}
              </p>
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">Symbols</label>
              <div className="flex gap-2 mb-2">
                <Input
                  placeholder="AAPL"
                  value={newSymbol}
                  onChange={(e) => setNewSymbol(e.target.value.toUpperCase())}
                  onKeyDown={(e) => e.key === 'Enter' && addSymbol()}
                />
                <Button onClick={addSymbol} variant="outline">Add</Button>
              </div>
              <div className="flex flex-wrap gap-2">
                {selectedSymbols.map(symbol => (
                  <Badge key={symbol} variant="secondary" className="cursor-pointer" onClick={() => removeSymbol(symbol)}>
                    {symbol} ×
                  </Badge>
                ))}
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium mb-2">Lookback Period</label>
                <Input
                  type="number"
                  value={strategyParams.lookbackPeriod}
                  onChange={(e) => setStrategyParams({...strategyParams, lookbackPeriod: parseInt(e.target.value)})}
                  min={5}
                  max={100}
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-2">Threshold (%)</label>
                <Input
                  type="number"
                  step="0.01"
                  value={strategyParams.threshold}
                  onChange={(e) => setStrategyParams({...strategyParams, threshold: parseFloat(e.target.value)})}
                  min={0.01}
                  max={1}
                />
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium mb-2">Max Position</label>
                <Input
                  type="number"
                  value={strategyParams.maxPosition}
                  onChange={(e) => setStrategyParams({...strategyParams, maxPosition: parseInt(e.target.value)})}
                  min={1}
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-2">Stop Loss (%)</label>
                <Input
                  type="number"
                  step="0.01"
                  value={strategyParams.stopLoss}
                  onChange={(e) => setStrategyParams({...strategyParams, stopLoss: parseFloat(e.target.value)})}
                  min={0.01}
                  max={1}
                />
              </div>
            </div>

            <Button onClick={runBacktest} disabled={isRunning} className="w-full">
              {isRunning ? 'Running Backtest...' : 'Run Backtest'}
            </Button>
          </div>
        </Card>

        {/* Results */}
        <Card className="p-6">
          <h3 className="text-xl font-semibold mb-4">Backtest Results</h3>
          
          {backtestResults ? (
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div className="text-center p-3 bg-gradient-to-r from-green-900/20 to-green-800/20 border border-green-500/30 rounded">
                  <div className="text-green-400 text-sm font-medium">Total Return</div>
                  <div className={`text-2xl font-bold ${backtestResults.totalReturn >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                    {backtestResults.totalReturn.toFixed(2)}%
                  </div>
                </div>
                <div className="text-center p-3 bg-gradient-to-r from-blue-900/20 to-blue-800/20 border border-blue-500/30 rounded">
                  <div className="text-blue-400 text-sm font-medium">Sharpe Ratio</div>
                  <div className="text-2xl font-bold text-white">{backtestResults.sharpeRatio.toFixed(2)}</div>
                </div>
              </div>
              
              <div className="grid grid-cols-2 gap-4">
                <div className="text-center p-3 bg-gradient-to-r from-red-900/20 to-red-800/20 border border-red-500/30 rounded">
                  <div className="text-red-400 text-sm font-medium">Max Drawdown</div>
                  <div className="text-2xl font-bold text-white">{backtestResults.maxDrawdown.toFixed(2)}%</div>
                </div>
                <div className="text-center p-3 bg-gradient-to-r from-purple-900/20 to-purple-800/20 border border-purple-500/30 rounded">
                  <div className="text-purple-400 text-sm font-medium">Win Rate</div>
                  <div className="text-2xl font-bold text-white">{backtestResults.winRate.toFixed(1)}%</div>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="text-center p-3 bg-gradient-to-r from-orange-900/20 to-orange-800/20 border border-orange-500/30 rounded">
                  <div className="text-orange-400 text-sm font-medium">Total Trades</div>
                  <div className="text-2xl font-bold text-white">{backtestResults.totalTrades}</div>
                </div>
                <div className="text-center p-3 bg-gradient-to-r from-cyan-900/20 to-cyan-800/20 border border-cyan-500/30 rounded">
                  <div className="text-cyan-400 text-sm font-medium">Avg Trade Return</div>
                  <div className={`text-2xl font-bold ${backtestResults.avgTradeReturn >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                    {backtestResults.avgTradeReturn.toFixed(2)}%
                  </div>
                </div>
              </div>
            </div>
          ) : (
            <div className="text-center text-muted-foreground py-8">
              <div className="text-4xl mb-2">📊</div>
              <p>Run a backtest to see performance metrics</p>
            </div>
          )}
        </Card>
      </div>

      {/* Strategy Library */}
      <Card className="p-6">
        <h3 className="text-xl font-semibold mb-4">Strategy Library</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {strategies.map(strategy => (
            <div key={strategy.value} className="p-4 border rounded-lg hover:bg-muted/50 cursor-pointer transition-colors">
              <h4 className="font-semibold mb-2">{strategy.label}</h4>
              <p className="text-sm text-muted-foreground mb-3">{strategy.description}</p>
              <div className="flex gap-2">
                <Button size="sm" variant="outline" onClick={() => setSelectedStrategy(strategy.value)}>
                  Select
                </Button>
                <Button size="sm" variant="ghost">
                  Details
                </Button>
              </div>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}
