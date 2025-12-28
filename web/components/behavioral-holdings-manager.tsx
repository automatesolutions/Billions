'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
// import { textarea } from "@/components/ui/textarea";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { 
  Edit, 
  Plus, 
  Minus, 
  MessageSquare, 
  TrendingUp, 
  TrendingDown,
  Target,
  AlertTriangle,
  CheckCircle,
  Clock,
  DollarSign,
  BarChart3,
  Brain,
  FileText
} from "lucide-react";
import { api } from "@/lib/api";

interface Holding {
  id: string;
  symbol: string;
  qty: number;
  current_price: number;
  unrealized_pnl: number;
  unrealized_pnl_percent: number;
  entry_rationale?: string;
  exit_rationale?: string;
  annotations?: TradeAnnotation[];
}

interface TradeAnnotation {
  id: string;
  trade_id: string;
  action_type: 'entry' | 'addition' | 'partial_exit' | 'full_exit' | 'stop_loss' | 'take_profit';
  rationale: string;
  market_conditions?: string;
  technical_indicators?: string[];
  fundamental_factors?: string[];
  risk_assessment?: string;
  confidence_level: number;
  expected_hold_time?: string;
  target_price?: number;
  stop_loss_price?: number;
  position_size_reasoning?: string;
  created_at: string;
  updated_at?: string;
}

interface ExitDecision {
  position_id: string;
  symbol: string;
  exit_type: 'partial' | 'full' | 'stop_loss' | 'take_profit';
  exit_percentage: number;
  exit_quantity: number;
  exit_price: number;
  exit_reason: string;
  market_context?: string;
  technical_reason?: string;
  fundamental_reason?: string;
  emotional_factors?: string;
  lessons_learned?: string;
  would_reenter?: boolean;
  reentry_conditions?: string;
}

interface AdditionDecision {
  position_id: string;
  symbol: string;
  addition_quantity: number;
  addition_price: number;
  addition_reason: string;
  market_opportunity?: string;
  technical_setup?: string;
  fundamental_catalyst?: string;
  risk_reward_ratio?: number;
  position_sizing_logic?: string;
}

export function BehavioralHoldingsManager() {
  const [holdings, setHoldings] = useState<Holding[]>([]);
  const [selectedHolding, setSelectedHolding] = useState<Holding | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  
  // Dialog states
  const [annotationDialogOpen, setAnnotationDialogOpen] = useState(false);
  const [exitDialogOpen, setExitDialogOpen] = useState(false);
  const [additionDialogOpen, setAdditionDialogOpen] = useState(false);
  
  // Form states
  const [annotationForm, setAnnotationForm] = useState<Partial<TradeAnnotation>>({
    action_type: 'entry',
    confidence_level: 5,
    rationale: '',
    market_conditions: '',
    technical_indicators: [],
    fundamental_factors: [],
    risk_assessment: 'medium'
  });
  
  const [exitForm, setExitForm] = useState<Partial<ExitDecision>>({
    exit_type: 'partial',
    exit_percentage: 0.25,
    exit_reason: '',
    market_context: '',
    technical_reason: '',
    fundamental_reason: '',
    emotional_factors: '',
    lessons_learned: '',
    would_reenter: false
  });
  
  const [additionForm, setAdditionForm] = useState<Partial<AdditionDecision>>({
    addition_reason: '',
    market_opportunity: '',
    technical_setup: '',
    fundamental_catalyst: '',
    position_sizing_logic: ''
  });

  const fetchHoldings = async () => {
    try {
      setLoading(true);
      setError(null);
      
      // Get positions from Alpaca
      const positionsResponse = await api.getPositions();
      const positions = positionsResponse.positions || [];
      
      // Get behavioral context for each position
      const holdingsWithContext = await Promise.all(
        positions.map(async (position: any) => {
          try {
            const context = await api.getHoldingContext(position.symbol);
            return {
              id: position.asset_id,
              symbol: position.symbol,
              qty: position.qty,
              current_price: position.current_price,
              unrealized_pnl: position.unrealized_pl,
              unrealized_pnl_percent: position.unrealized_plpc * 100,
              annotations: context.rationales || []
            };
          } catch (err) {
            return {
              id: position.asset_id,
              symbol: position.symbol,
              qty: position.qty,
              current_price: position.current_price,
              unrealized_pnl: position.unrealized_pl,
              unrealized_pnl_percent: position.unrealized_plpc * 100,
              annotations: []
            };
          }
        })
      );
      
      setHoldings(holdingsWithContext);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch holdings');
      console.error('Error fetching holdings:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchHoldings();
  }, []);

  const handleAddAnnotation = async () => {
    if (!selectedHolding || !annotationForm.rationale) return;
    
    try {
      const rationale = {
        trade_id: selectedHolding.id,
        action_type: annotationForm.action_type,
        rationale: annotationForm.rationale,
        market_conditions: annotationForm.market_conditions,
        technical_indicators: annotationForm.technical_indicators,
        fundamental_factors: annotationForm.fundamental_factors,
        risk_assessment: annotationForm.risk_assessment,
        confidence_level: annotationForm.confidence_level,
        expected_hold_time: annotationForm.expected_hold_time,
        target_price: annotationForm.target_price,
        stop_loss_price: annotationForm.stop_loss_price,
        position_size_reasoning: annotationForm.position_size_reasoning
      };
      
      await api.addTradeRationale(rationale);
      setAnnotationDialogOpen(false);
      setAnnotationForm({
        action_type: 'entry',
        confidence_level: 5,
        rationale: '',
        market_conditions: '',
        technical_indicators: [],
        fundamental_factors: [],
        risk_assessment: 'medium'
      });
      await fetchHoldings();
    } catch (err) {
      console.error('Error adding annotation:', err);
    }
  };

  const handleExecuteExit = async () => {
    if (!selectedHolding || !exitForm.exit_reason) return;
    
    try {
      const exitDecision = {
        position_id: selectedHolding.id,
        symbol: selectedHolding.symbol,
        exit_type: exitForm.exit_type,
        exit_percentage: exitForm.exit_percentage,
        exit_quantity: Math.floor(selectedHolding.qty * (exitForm.exit_percentage || 0.25)),
        exit_price: selectedHolding.current_price,
        exit_reason: exitForm.exit_reason,
        market_context: exitForm.market_context,
        technical_reason: exitForm.technical_reason,
        fundamental_reason: exitForm.fundamental_reason,
        emotional_factors: exitForm.emotional_factors,
        lessons_learned: exitForm.lessons_learned,
        would_reenter: exitForm.would_reenter,
        reentry_conditions: exitForm.reentry_conditions
      };
      
      await api.executeExitDecision(exitDecision);
      setExitDialogOpen(false);
      setExitForm({
        exit_type: 'partial',
        exit_percentage: 0.25,
        exit_reason: '',
        market_context: '',
        technical_reason: '',
        fundamental_reason: '',
        emotional_factors: '',
        lessons_learned: '',
        would_reenter: false
      });
      await fetchHoldings();
    } catch (err) {
      console.error('Error executing exit:', err);
    }
  };

  const handleExecuteAddition = async () => {
    if (!selectedHolding || !additionForm.addition_reason) return;
    
    try {
      const additionDecision = {
        position_id: selectedHolding.id,
        symbol: selectedHolding.symbol,
        addition_quantity: additionForm.addition_quantity || 1,
        addition_price: selectedHolding.current_price,
        addition_reason: additionForm.addition_reason,
        market_opportunity: additionForm.market_opportunity,
        technical_setup: additionForm.technical_setup,
        fundamental_catalyst: additionForm.fundamental_catalyst,
        risk_reward_ratio: additionForm.risk_reward_ratio,
        position_sizing_logic: additionForm.position_sizing_logic
      };
      
      await api.executeAdditionDecision(additionDecision);
      setAdditionDialogOpen(false);
      setAdditionForm({
        addition_reason: '',
        market_opportunity: '',
        technical_setup: '',
        fundamental_catalyst: '',
        position_sizing_logic: ''
      });
      await fetchHoldings();
    } catch (err) {
      console.error('Error executing addition:', err);
    }
  };

  if (loading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Brain className="h-5 w-5" />
            Behavioral Holdings Manager
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {[1, 2, 3].map((i) => (
              <div key={i} className="animate-pulse">
                <div className="h-4 bg-gray-200 rounded w-3/4 mb-2"></div>
                <div className="h-3 bg-gray-200 rounded w-1/2"></div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Brain className="h-5 w-5" />
            Behavioral Holdings Manager
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-4">
            <p className="text-sm text-muted-foreground mb-3">{error}</p>
            <Button onClick={fetchHoldings} variant="outline" size="sm">
              Retry
            </Button>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      {/* Holdings List */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Brain className="h-5 w-5" />
            Behavioral Holdings Manager
            <Badge variant="outline">{holdings.length} Holdings</Badge>
          </CardTitle>
          <CardDescription>
            Manage your positions with behavioral context and decision tracking
          </CardDescription>
        </CardHeader>
        <CardContent>
          {holdings.length === 0 ? (
            <div className="text-center py-8">
              <BarChart3 className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
              <h3 className="text-lg font-semibold mb-2">No Holdings Found</h3>
              <p className="text-muted-foreground mb-4">
                You don't have any current positions to manage.
              </p>
            </div>
          ) : (
            <div className="space-y-4">
              {holdings.map((holding, index) => (
                <Card key={`${holding.symbol}-${holding.id || index}`} className="border-l-4 border-l-blue-500">
                  <CardContent className="p-4">
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center gap-3">
                        <div>
                          <h3 className="font-semibold text-lg">{holding.symbol}</h3>
                          <p className="text-sm text-muted-foreground">
                            {holding.qty} shares @ ${holding.current_price.toFixed(2)}
                          </p>
                        </div>
                        <Badge 
                          variant={holding.unrealized_pnl >= 0 ? "default" : "destructive"}
                          className="ml-2"
                        >
                          {holding.unrealized_pnl >= 0 ? '+' : ''}${holding.unrealized_pnl.toFixed(2)} 
                          ({holding.unrealized_pnl_percent >= 0 ? '+' : ''}{holding.unrealized_pnl_percent.toFixed(2)}%)
                        </Badge>
                      </div>
                      
                      <div className="flex gap-2">
                        <Dialog open={annotationDialogOpen} onOpenChange={setAnnotationDialogOpen}>
                          <DialogTrigger asChild>
                            <Button 
                              variant="outline" 
                              size="sm"
                              onClick={() => setSelectedHolding(holding)}
                            >
                              <MessageSquare className="h-4 w-4 mr-1" />
                              Annotate
                            </Button>
                          </DialogTrigger>
                        </Dialog>
                        
                        <Dialog open={additionDialogOpen} onOpenChange={setAdditionDialogOpen}>
                          <DialogTrigger asChild>
                            <Button 
                              variant="outline" 
                              size="sm"
                              onClick={() => setSelectedHolding(holding)}
                            >
                              <Plus className="h-4 w-4 mr-1" />
                              Add
                            </Button>
                          </DialogTrigger>
                        </Dialog>
                        
                        <Dialog open={exitDialogOpen} onOpenChange={setExitDialogOpen}>
                          <DialogTrigger asChild>
                            <Button 
                              variant="outline" 
                              size="sm"
                              onClick={() => setSelectedHolding(holding)}
                            >
                              <Minus className="h-4 w-4 mr-1" />
                              Exit
                            </Button>
                          </DialogTrigger>
                        </Dialog>
                      </div>
                    </div>
                    
                    {/* Annotations Summary */}
                    {holding.annotations && holding.annotations.length > 0 && (
                      <div className="mt-3">
                        <div className="flex items-center gap-2 mb-2">
                          <FileText className="h-4 w-4 text-muted-foreground" />
                          <span className="text-sm font-medium">Recent Annotations</span>
                          <Badge variant="secondary" className="text-xs">
                            {holding.annotations.length}
                          </Badge>
                        </div>
                        <div className="space-y-2">
                          {holding.annotations.slice(0, 2).map((annotation, annotationIndex) => (
                            <div key={annotation.id || `annotation-${annotationIndex}`} className="bg-muted/50 rounded p-2">
                              <div className="flex items-center gap-2 mb-1">
                                <Badge variant="outline" className="text-xs">
                                  {annotation.action_type}
                                </Badge>
                                <span className="text-xs text-muted-foreground">
                                  Confidence: {annotation.confidence_level}/10
                                </span>
                              </div>
                              <p className="text-sm">{annotation.rationale}</p>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </CardContent>
                </Card>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Annotation Dialog */}
      <Dialog open={annotationDialogOpen} onOpenChange={setAnnotationDialogOpen}>
        <DialogContent className="max-w-2xl">
          <DialogHeader>
            <DialogTitle>Add Trade Annotation</DialogTitle>
            <DialogDescription>
              Document your decision-making process for {selectedHolding?.symbol}
            </DialogDescription>
          </DialogHeader>
          
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <Label htmlFor="action_type">Action Type</Label>
                <Select 
                  value={annotationForm.action_type} 
                  onValueChange={(value) => setAnnotationForm(prev => ({ ...prev, action_type: value as any }))}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="entry">Entry</SelectItem>
                    <SelectItem value="addition">Addition</SelectItem>
                    <SelectItem value="partial_exit">Partial Exit</SelectItem>
                    <SelectItem value="full_exit">Full Exit</SelectItem>
                    <SelectItem value="stop_loss">Stop Loss</SelectItem>
                    <SelectItem value="take_profit">Take Profit</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              
              <div>
                <Label htmlFor="confidence_level">Confidence Level (1-10)</Label>
                <Input
                  id="confidence_level"
                  type="number"
                  min="1"
                  max="10"
                  value={annotationForm.confidence_level}
                  onChange={(e) => setAnnotationForm(prev => ({ ...prev, confidence_level: parseInt(e.target.value) }))}
                />
              </div>
            </div>
            
            <div>
              <Label htmlFor="rationale">Rationale *</Label>
              <textarea
                id="rationale"
                placeholder="Explain your decision-making process..."
                value={annotationForm.rationale}
                onChange={(e: React.ChangeEvent<HTMLTextAreaElement>) => setAnnotationForm(prev => ({ ...prev, rationale: e.target.value }))}
                rows={4}
                className="flex min-h-[80px] w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
              />
            </div>
            
            <div className="grid grid-cols-2 gap-4">
              <div>
                <Label htmlFor="market_conditions">Market Conditions</Label>
                <Select 
                  value={annotationForm.market_conditions} 
                  onValueChange={(value) => setAnnotationForm(prev => ({ ...prev, market_conditions: value }))}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Select market conditions" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="bull">Bull Market</SelectItem>
                    <SelectItem value="bear">Bear Market</SelectItem>
                    <SelectItem value="sideways">Sideways</SelectItem>
                    <SelectItem value="volatile">Volatile</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              
              <div>
                <Label htmlFor="risk_assessment">Risk Assessment</Label>
                <Select 
                  value={annotationForm.risk_assessment} 
                  onValueChange={(value) => setAnnotationForm(prev => ({ ...prev, risk_assessment: value }))}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Select risk level" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="low">Low Risk</SelectItem>
                    <SelectItem value="medium">Medium Risk</SelectItem>
                    <SelectItem value="high">High Risk</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>
            
            <div className="grid grid-cols-2 gap-4">
              <div>
                <Label htmlFor="target_price">Target Price</Label>
                <Input
                  id="target_price"
                  type="number"
                  step="0.01"
                  placeholder="Optional target price"
                  value={annotationForm.target_price || ''}
                  onChange={(e) => setAnnotationForm(prev => ({ ...prev, target_price: parseFloat(e.target.value) }))}
                />
              </div>
              
              <div>
                <Label htmlFor="stop_loss_price">Stop Loss Price</Label>
                <Input
                  id="stop_loss_price"
                  type="number"
                  step="0.01"
                  placeholder="Optional stop loss"
                  value={annotationForm.stop_loss_price || ''}
                  onChange={(e) => setAnnotationForm(prev => ({ ...prev, stop_loss_price: parseFloat(e.target.value) }))}
                />
              </div>
            </div>
          </div>
          
          <DialogFooter>
            <Button variant="outline" onClick={() => setAnnotationDialogOpen(false)}>
              Cancel
            </Button>
            <Button onClick={handleAddAnnotation} disabled={!annotationForm.rationale}>
              Add Annotation
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Exit Decision Dialog */}
      <Dialog open={exitDialogOpen} onOpenChange={setExitDialogOpen}>
        <DialogContent className="max-w-2xl">
          <DialogHeader>
            <DialogTitle>Execute Exit Decision</DialogTitle>
            <DialogDescription>
              Log your exit decision for {selectedHolding?.symbol}
            </DialogDescription>
          </DialogHeader>
          
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <Label htmlFor="exit_type">Exit Type</Label>
                <Select 
                  value={exitForm.exit_type} 
                  onValueChange={(value) => setExitForm(prev => ({ ...prev, exit_type: value as any }))}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="partial">Partial Exit</SelectItem>
                    <SelectItem value="full">Full Exit</SelectItem>
                    <SelectItem value="stop_loss">Stop Loss</SelectItem>
                    <SelectItem value="take_profit">Take Profit</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              
              <div>
                <Label htmlFor="exit_percentage">Exit Percentage</Label>
                <Input
                  id="exit_percentage"
                  type="number"
                  min="0.01"
                  max="1"
                  step="0.01"
                  value={exitForm.exit_percentage}
                  onChange={(e) => setExitForm(prev => ({ ...prev, exit_percentage: parseFloat(e.target.value) }))}
                />
              </div>
            </div>
            
            <div>
              <Label htmlFor="exit_reason">Exit Reason *</Label>
              <textarea
                id="exit_reason"
                placeholder="Why are you exiting this position?"
                value={exitForm.exit_reason}
                onChange={(e: React.ChangeEvent<HTMLTextAreaElement>) => setExitForm(prev => ({ ...prev, exit_reason: e.target.value }))}
                rows={3}
                className="flex min-h-[80px] w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
              />
            </div>
            
            <div className="grid grid-cols-2 gap-4">
              <div>
                <Label htmlFor="technical_reason">Technical Reason</Label>
                <textarea
                  id="technical_reason"
                  placeholder="Technical analysis factors..."
                  value={exitForm.technical_reason}
                  onChange={(e: React.ChangeEvent<HTMLTextAreaElement>) => setExitForm(prev => ({ ...prev, technical_reason: e.target.value }))}
                  rows={2}
                  className="flex min-h-[80px] w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
                />
              </div>
              
              <div>
                <Label htmlFor="fundamental_reason">Fundamental Reason</Label>
                <textarea
                  id="fundamental_reason"
                  placeholder="Fundamental analysis factors..."
                  value={exitForm.fundamental_reason}
                  onChange={(e: React.ChangeEvent<HTMLTextAreaElement>) => setExitForm(prev => ({ ...prev, fundamental_reason: e.target.value }))}
                  className="flex min-h-[80px] w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
                />
              </div>
            </div>
            
            <div>
              <Label htmlFor="lessons_learned">Lessons Learned</Label>
              <textarea
                id="lessons_learned"
                placeholder="What did you learn from this trade?"
                value={exitForm.lessons_learned}
                onChange={(e: React.ChangeEvent<HTMLTextAreaElement>) => setExitForm(prev => ({ ...prev, lessons_learned: e.target.value }))}
                rows={2}
              />
            </div>
          </div>
          
          <DialogFooter>
            <Button variant="outline" onClick={() => setExitDialogOpen(false)}>
              Cancel
            </Button>
            <Button onClick={handleExecuteExit} disabled={!exitForm.exit_reason}>
              Log Exit Decision
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Addition Decision Dialog */}
      <Dialog open={additionDialogOpen} onOpenChange={setAdditionDialogOpen}>
        <DialogContent className="max-w-2xl">
          <DialogHeader>
            <DialogTitle>Add to Position</DialogTitle>
            <DialogDescription>
              Log your decision to add to {selectedHolding?.symbol}
            </DialogDescription>
          </DialogHeader>
          
          <div className="space-y-4">
            <div>
              <Label htmlFor="addition_quantity">Quantity to Add</Label>
              <Input
                id="addition_quantity"
                type="number"
                min="1"
                value={additionForm.addition_quantity || ''}
                onChange={(e) => setAdditionForm(prev => ({ ...prev, addition_quantity: parseInt(e.target.value) }))}
              />
            </div>
            
            <div>
              <Label htmlFor="addition_reason">Addition Reason *</Label>
              <textarea
                id="addition_reason"
                placeholder="Why are you adding to this position?"
                value={additionForm.addition_reason}
                onChange={(e: React.ChangeEvent<HTMLTextAreaElement>) => setAdditionForm(prev => ({ ...prev, addition_reason: e.target.value }))}
                  rows={3}
                  className="flex min-h-[80px] w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
                />
            </div>
            
            <div className="grid grid-cols-2 gap-4">
              <div>
                <Label htmlFor="market_opportunity">Market Opportunity</Label>
                <textarea
                  id="market_opportunity"
                  placeholder="What market opportunity did you identify?"
                  value={additionForm.market_opportunity}
                  onChange={(e: React.ChangeEvent<HTMLTextAreaElement>) => setAdditionForm(prev => ({ ...prev, market_opportunity: e.target.value }))}
                  className="flex min-h-[80px] w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
                />
              </div>
              
              <div>
                <Label htmlFor="technical_setup">Technical Setup</Label>
                <textarea
                  id="technical_setup"
                  placeholder="Technical setup for addition..."
                  value={additionForm.technical_setup}
                  onChange={(e: React.ChangeEvent<HTMLTextAreaElement>) => setAdditionForm(prev => ({ ...prev, technical_setup: e.target.value }))}
                  className="flex min-h-[80px] w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
                />
              </div>
            </div>
            
            <div>
              <Label htmlFor="position_sizing_logic">Position Sizing Logic</Label>
              <textarea
                id="position_sizing_logic"
                placeholder="How did you determine the position size?"
                value={additionForm.position_sizing_logic}
                onChange={(e: React.ChangeEvent<HTMLTextAreaElement>) => setAdditionForm(prev => ({ ...prev, position_sizing_logic: e.target.value }))}
                rows={2}
              />
            </div>
          </div>
          
          <DialogFooter>
            <Button variant="outline" onClick={() => setAdditionDialogOpen(false)}>
              Cancel
            </Button>
            <Button onClick={handleExecuteAddition} disabled={!additionForm.addition_reason}>
              Log Addition Decision
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
