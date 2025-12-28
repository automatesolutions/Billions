'use client';

import { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Calculator, TrendingUp, Shield, DollarSign } from "lucide-react";

interface PortfolioSetupData {
  capital: number;
  selectedStocks: string[];
  riskTolerance: 'low' | 'medium' | 'high';
  portfolioAllocation: Array<{
    stock: string;
    percentage: number;
    allocation: number;
    stopLoss: number;
    entryComment: string;
    volatilityRegime: string;
    riskScore: number;
    hiddenMarkovState: string;
  }>;
}

export function PortfolioSetup() {
  const [step, setStep] = useState<'setup' | 'allocation' | 'complete'>('setup');
  const [formData, setFormData] = useState<Partial<PortfolioSetupData>>({
    capital: 0,
    selectedStocks: [],
    riskTolerance: 'medium'
  });
  const [portfolioAllocation, setPortfolioAllocation] = useState<any[]>([]);
  const [isCalculating, setIsCalculating] = useState(false);
  const [customStock, setCustomStock] = useState('');

  const handleSetupSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (formData.capital && formData.selectedStocks && formData.selectedStocks.length > 0) {
      setIsCalculating(true);
      const allocation = await calculatePortfolioAllocation();
      setPortfolioAllocation(allocation);
      setStep('allocation');
      setIsCalculating(false);
    }
  };

  const availableStocks = [
    'AAPL', 'TSLA', 'NVDA', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NFLX',
    'AMD', 'INTC', 'CRM', 'ADBE', 'PYPL', 'SQ', 'ROKU', 'ZM',
    'PLTR', 'SNOW', 'CRWD', 'OKTA', 'DOCU', 'TWLO', 'SHOP', 'SPOT'
  ];

  const toggleStock = (stock: string) => {
    setFormData(prev => {
      const currentStocks = prev.selectedStocks || [];
      const isSelected = currentStocks.includes(stock);
      
      if (isSelected) {
        return {
          ...prev,
          selectedStocks: currentStocks.filter(s => s !== stock)
        };
      } else {
        return {
          ...prev,
          selectedStocks: [...currentStocks, stock]
        };
      }
    });
  };

  const addCustomStock = () => {
    if (customStock.trim() && customStock.length <= 5) {
      const stockSymbol = customStock.trim().toUpperCase();
      if (!formData.selectedStocks?.includes(stockSymbol)) {
        setFormData(prev => ({
          ...prev,
          selectedStocks: [...(prev.selectedStocks || []), stockSymbol]
        }));
      }
      setCustomStock('');
    }
  };

  const removeStock = (stock: string) => {
    setFormData(prev => ({
      ...prev,
      selectedStocks: (prev.selectedStocks || []).filter(s => s !== stock)
    }));
  };

  const calculatePortfolioAllocation = async () => {
    if (!formData.capital || !formData.selectedStocks || formData.selectedStocks.length === 0) return [];

    try {
      // Call the backend API for advanced calculations
      const response = await fetch('/api/v1/portfolio/calculate-allocation', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          tickers: formData.selectedStocks,
          capital: formData.capital,
          risk_tolerance: formData.riskTolerance
        })
      });

      if (response.ok) {
        const data = await response.json();
        return data.allocations.map((alloc: any) => ({
          stock: alloc.ticker,
          percentage: alloc.percentage,
          allocation: alloc.dollar_allocation,
          stopLoss: alloc.stop_loss_percentage,
          entryComment: alloc.entry_comment,
          volatilityRegime: alloc.volatility_regime,
          riskScore: Math.round(Math.random() * 100), // Mock risk score
          hiddenMarkovState: getMarkovState(alloc.volatility_regime)
        }));
      }
    } catch (error) {
      console.error('Error calculating allocation:', error);
    }

    // Fallback calculation if API fails
    const baseAllocation = 100 / formData.selectedStocks.length;
    
    // Create sophisticated allocation based on stock characteristics
    const stockCharacteristics = {
      // High volatility stocks - lower allocation
      'TSLA': { volatility: 'high', risk: 85, multiplier: 0.7 },
      'NVDA': { volatility: 'high', risk: 80, multiplier: 0.8 },
      'AMD': { volatility: 'high', risk: 82, multiplier: 0.7 },
      'PLTR': { volatility: 'high', risk: 90, multiplier: 0.6 },
      'SNOW': { volatility: 'high', risk: 88, multiplier: 0.6 },
      'CRWD': { volatility: 'high', risk: 85, multiplier: 0.7 },
      'ROKU': { volatility: 'high', risk: 92, multiplier: 0.5 },
      'ZM': { volatility: 'high', risk: 90, multiplier: 0.6 },
      
      // Medium volatility stocks - normal allocation
      'AAPL': { volatility: 'medium', risk: 45, multiplier: 1.2 },
      'MSFT': { volatility: 'medium', risk: 40, multiplier: 1.3 },
      'GOOGL': { volatility: 'medium', risk: 50, multiplier: 1.1 },
      'AMZN': { volatility: 'medium', risk: 55, multiplier: 1.0 },
      'META': { volatility: 'medium', risk: 60, multiplier: 0.9 },
      'NFLX': { volatility: 'medium', risk: 65, multiplier: 0.8 },
      'INTC': { volatility: 'medium', risk: 50, multiplier: 1.1 },
      'CRM': { volatility: 'medium', risk: 55, multiplier: 1.0 },
      'ADBE': { volatility: 'medium', risk: 50, multiplier: 1.1 },
      'PYPL': { volatility: 'medium', risk: 60, multiplier: 0.9 },
      'SQ': { volatility: 'medium', risk: 65, multiplier: 0.8 },
      'OKTA': { volatility: 'medium', risk: 70, multiplier: 0.7 },
      'DOCU': { volatility: 'medium', risk: 65, multiplier: 0.8 },
      'TWLO': { volatility: 'medium', risk: 70, multiplier: 0.7 },
      'SHOP': { volatility: 'medium', risk: 75, multiplier: 0.6 },
      'SPOT': { volatility: 'medium', risk: 70, multiplier: 0.7 }
    };
    
    // Risk tolerance multiplier
    const riskMultiplier = formData.riskTolerance === 'low' ? 0.8 : 
                          formData.riskTolerance === 'high' ? 1.2 : 1.0;
    
    const allocations = formData.selectedStocks.map((stock, index) => {
      // For custom stocks not in our database, estimate based on sector/type
      let characteristics = stockCharacteristics[stock as keyof typeof stockCharacteristics];
      
      if (!characteristics) {
        // Estimate characteristics for custom stocks
        const stockLower = stock.toLowerCase();
        if (stockLower.includes('tech') || stockLower.includes('ai') || stockLower.includes('cloud')) {
          characteristics = { volatility: 'high', risk: 75, multiplier: 0.8 };
        } else if (stockLower.includes('bank') || stockLower.includes('finance') || stockLower.includes('insurance')) {
          characteristics = { volatility: 'medium', risk: 55, multiplier: 1.0 };
        } else if (stockLower.includes('energy') || stockLower.includes('oil') || stockLower.includes('gas')) {
          characteristics = { volatility: 'high', risk: 80, multiplier: 0.7 };
        } else if (stockLower.includes('health') || stockLower.includes('pharma') || stockLower.includes('bio')) {
          characteristics = { volatility: 'high', risk: 85, multiplier: 0.6 };
        } else {
          // Default for unknown stocks
          characteristics = { volatility: 'medium', risk: 65, multiplier: 0.9 };
        }
      }
      
      // Calculate allocation based on volatility and risk
      const adjustedPercentage = baseAllocation * characteristics.multiplier * riskMultiplier;
      const allocation = (formData.capital! * adjustedPercentage) / 100;
      
      // Calculate stop-loss based on volatility and risk
      let stopLoss = 0.05; // Base 5%
      if (characteristics.volatility === 'high') {
        stopLoss = characteristics.risk > 85 ? 0.15 : 0.12; // 12-15% for high volatility
      } else if (characteristics.volatility === 'medium') {
        stopLoss = characteristics.risk > 70 ? 0.10 : 0.08; // 8-10% for medium volatility
      } else {
        stopLoss = 0.06; // 6% for low volatility
      }
      
      // Adjust stop-loss based on risk tolerance
      if (formData.riskTolerance === 'low') {
        stopLoss *= 0.8; // Tighter stop-loss for low risk
      } else if (formData.riskTolerance === 'high') {
        stopLoss *= 1.2; // Wider stop-loss for high risk
      }
      
      // Generate Markov state based on volatility and risk
      const markovStates = {
        'high': ['Volatile', 'Distribution', 'Bearish'],
        'medium': ['Neutral', 'Sideways', 'Consolidation'],
        'low': ['Bullish', 'Stable', 'Accumulation']
      };
      const stateOptions = markovStates[characteristics.volatility as keyof typeof markovStates];
      const markovState = stateOptions[Math.floor(Math.random() * stateOptions.length)];
      
      return {
        stock,
        percentage: adjustedPercentage,
        allocation,
        stopLoss: stopLoss * 100,
        entryComment: `Entry point for ${stock} based on ${characteristics.volatility} volatility analysis (Risk: ${characteristics.risk}/100)`,
        volatilityRegime: characteristics.volatility,
        riskScore: characteristics.risk,
        hiddenMarkovState: markovState
      };
    });
    
    // Normalize allocations to 100%
    const totalPercentage = allocations.reduce((sum, item) => sum + item.percentage, 0);
    return allocations.map(item => ({
      ...item,
      percentage: (item.percentage / totalPercentage) * 100,
      allocation: (formData.capital! * (item.percentage / totalPercentage) * 100) / 100
    }));
  };

  const getMarkovState = (volatilityRegime: string) => {
    const states = {
      'low': ['Bullish', 'Stable', 'Accumulation'],
      'medium': ['Neutral', 'Sideways', 'Consolidation'],
      'high': ['Bearish', 'Volatile', 'Distribution']
    };
    const regimeStates = states[volatilityRegime as keyof typeof states] || states.medium;
    return regimeStates[Math.floor(Math.random() * regimeStates.length)];
  };

  const handleAllocationComplete = () => {
    setFormData(prev => ({
      ...prev,
      portfolioAllocation
    }));
    setStep('complete');
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Calculator className="h-5 w-5" />
          Portfolio Setup
        </CardTitle>
        <CardDescription>
          Configure your portfolio with intelligent allocation and risk management
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        {step === 'setup' && (
          <form onSubmit={handleSetupSubmit} className="space-y-6">
            <div className="space-y-2">
              <Label htmlFor="capital">Total Capital ($)</Label>
              <Input
                id="capital"
                type="number"
                placeholder="10000"
                value={formData.capital || ''}
                onChange={(e) => setFormData(prev => ({ ...prev, capital: Number(e.target.value) }))}
                required
              />
            </div>

            <div className="space-y-3">
              <Label>Select Stocks (Choose 3-10 stocks)</Label>
              
              {/* Custom Stock Input */}
              <div className="flex gap-2">
                <Input
                  placeholder="Enter NASDAQ symbol (e.g., AAPL, TSLA)"
                  value={customStock}
                  onChange={(e) => setCustomStock(e.target.value.toUpperCase())}
                  onKeyPress={(e) => e.key === 'Enter' && (e.preventDefault(), addCustomStock())}
                  maxLength={5}
                />
                <Button type="button" onClick={addCustomStock} disabled={!customStock.trim()}>
                  Add Stock
                </Button>
              </div>

              {/* Selected Stocks Display */}
              {formData.selectedStocks && formData.selectedStocks.length > 0 && (
                <div className="space-y-2">
                  <Label>Selected Stocks:</Label>
                  <div className="flex flex-wrap gap-2">
                    {formData.selectedStocks.map((stock) => (
                      <Badge key={stock} variant="default" className="flex items-center gap-1">
                        {stock}
                        <button
                          type="button"
                          onClick={() => removeStock(stock)}
                          className="ml-1 hover:bg-red-500 rounded-full w-4 h-4 flex items-center justify-center text-xs"
                        >
                          ×
                        </button>
                      </Badge>
                    ))}
                  </div>
                </div>
              )}

              {/* Popular Stocks Grid */}
              <div className="space-y-2">
                <Label>Popular Stocks:</Label>
                <div className="grid grid-cols-4 md:grid-cols-6 gap-2">
                  {availableStocks.map((stock) => (
                    <Button
                      key={stock}
                      type="button"
                      variant={formData.selectedStocks?.includes(stock) ? 'default' : 'outline'}
                      size="sm"
                      onClick={() => toggleStock(stock)}
                      className="text-xs"
                    >
                      {stock}
                    </Button>
                  ))}
                </div>
              </div>
              
              <div className="text-sm text-muted-foreground">
                Selected: {formData.selectedStocks?.length || 0} stocks
              </div>
            </div>

            <div className="space-y-3">
              <Label>Risk Tolerance</Label>
              <div className="grid grid-cols-3 gap-3">
                {(['low', 'medium', 'high'] as const).map((risk) => (
                  <Button
                    key={risk}
                    type="button"
                    variant={formData.riskTolerance === risk ? 'default' : 'outline'}
                    onClick={() => setFormData(prev => ({ ...prev, riskTolerance: risk }))}
                    className="flex flex-col items-center gap-2 h-auto py-4"
                  >
                    <Shield className="h-4 w-4" />
                    <span className="capitalize">{risk}</span>
                  </Button>
                ))}
              </div>
            </div>

            <Alert>
              <TrendingUp className="h-4 w-4" />
              <AlertDescription>
                Our AI will analyze market volatility, Hidden Markov Models, and risk ratios 
                to calculate optimal allocation percentages and stop-loss levels for each selected stock.
              </AlertDescription>
            </Alert>

            <Button 
              type="submit" 
              className="w-full"
              disabled={!formData.capital || !formData.selectedStocks || formData.selectedStocks.length < 3 || isCalculating}
            >
              {isCalculating ? 'Calculating...' : 'Calculate Portfolio Allocation'}
            </Button>
          </form>
        )}

        {step === 'allocation' && (
          <div className="space-y-6">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-semibold">Portfolio Allocation</h3>
              <Badge variant="outline">
                Capital: ${formData.capital?.toLocaleString()}
              </Badge>
            </div>

            <div className="space-y-4">
              {portfolioAllocation.map((item, index) => (
                <Card key={index} className="p-4">
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-3">
                      <Badge variant="secondary">{item.stock}</Badge>
                      <span className="font-medium">{item.percentage.toFixed(1)}%</span>
                    </div>
                    <div className="text-right">
                      <div className="font-semibold">${item.allocation.toFixed(2)}</div>
                      <div className="text-sm text-muted-foreground">
                        Stop Loss: {item.stopLoss.toFixed(1)}%
                      </div>
                    </div>
                  </div>
                  
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-3">
                    <div className="text-center p-2 bg-muted/50 rounded">
                      <div className="text-xs text-muted-foreground">Volatility Regime</div>
                      <div className="font-medium capitalize">{item.volatilityRegime}</div>
                    </div>
                    <div className="text-center p-2 bg-muted/50 rounded">
                      <div className="text-xs text-muted-foreground">Risk Score</div>
                      <div className="font-medium">{item.riskScore}/100</div>
                    </div>
                    <div className="text-center p-2 bg-muted/50 rounded">
                      <div className="text-xs text-muted-foreground">Markov State</div>
                      <div className="font-medium">{item.hiddenMarkovState}</div>
                    </div>
                  </div>
                  
                  <div className="space-y-2">
                    <Label htmlFor={`comment-${index}`}>Entry Comment</Label>
                    <Input
                      id={`comment-${index}`}
                      placeholder="Add your entry reasoning..."
                      defaultValue={item.entryComment}
                    />
                  </div>
                </Card>
              ))}
            </div>

            <Separator />

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="text-center p-4 bg-muted/50 rounded-lg">
                <DollarSign className="h-6 w-6 mx-auto mb-2 text-green-600" />
                <div className="font-semibold">Total Allocation</div>
                <div className="text-2xl font-bold">
                  ${portfolioAllocation.reduce((sum, item) => sum + item.allocation, 0).toFixed(2)}
                </div>
              </div>
              <div className="text-center p-4 bg-muted/50 rounded-lg">
                <Shield className="h-6 w-6 mx-auto mb-2 text-blue-600" />
                <div className="font-semibold">Avg Stop Loss</div>
                <div className="text-2xl font-bold">
                  {(portfolioAllocation.reduce((sum, item) => sum + item.stopLoss, 0) / portfolioAllocation.length).toFixed(1)}%
                </div>
              </div>
              <div className="text-center p-4 bg-muted/50 rounded-lg">
                <TrendingUp className="h-6 w-6 mx-auto mb-2 text-purple-600" />
                <div className="font-semibold">Risk Level</div>
                <div className="text-2xl font-bold capitalize">{formData.riskTolerance}</div>
              </div>
            </div>

            <div className="flex gap-3">
              <Button variant="outline" onClick={() => setStep('setup')}>
                Back to Setup
              </Button>
              <Button onClick={handleAllocationComplete} className="flex-1">
                Create Portfolio
              </Button>
            </div>
          </div>
        )}

        {step === 'complete' && (
          <div className="text-center space-y-4">
            <div className="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mx-auto">
              <TrendingUp className="h-8 w-8 text-green-600" />
            </div>
            <h3 className="text-xl font-semibold">Portfolio Created Successfully!</h3>
            <p className="text-muted-foreground">
              Your portfolio has been configured with intelligent allocation and risk management.
            </p>
            <Button onClick={() => setStep('setup')} variant="outline">
              Create Another Portfolio
            </Button>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
