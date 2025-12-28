'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { 
  Brain, 
  TrendingUp, 
  TrendingDown, 
  Target, 
  AlertTriangle,
  CheckCircle,
  Clock,
  BarChart3,
  Lightbulb,
  BookOpen,
  PieChart,
  Activity
} from "lucide-react";
import { api } from "@/lib/api";

interface BehavioralInsight {
  id: string;
  user_id: string;
  insight_type: 'pattern' | 'recommendation' | 'warning' | 'success';
  title: string;
  description: string;
  confidence_score: number;
  supporting_data: Record<string, any>;
  actionable_recommendations: string[];
  created_at: string;
}

interface PerformanceAnalysis {
  total_trades: number;
  completed_trades: number;
  average_return_percentage: number;
  win_rate: number;
  average_hold_duration_days: number;
  performances: any[];
}

export function BehavioralInsightsPanel() {
  const [insights, setInsights] = useState<BehavioralInsight[]>([]);
  const [performanceAnalysis, setPerformanceAnalysis] = useState<PerformanceAnalysis | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState('insights');

  const fetchBehavioralData = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const [insightsResponse, performanceResponse] = await Promise.all([
        api.getBehavioralInsights(20),
        api.getTradePerformanceAnalysis()
      ]);
      
      setInsights(insightsResponse.insights || []);
      setPerformanceAnalysis(performanceResponse);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch behavioral data');
      console.error('Error fetching behavioral data:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchBehavioralData();
  }, []);

  const getInsightIcon = (type: string) => {
    switch (type) {
      case 'pattern': return <BarChart3 className="h-5 w-5" />;
      case 'recommendation': return <Lightbulb className="h-5 w-5" />;
      case 'warning': return <AlertTriangle className="h-5 w-5" />;
      case 'success': return <CheckCircle className="h-5 w-5" />;
      default: return <Brain className="h-5 w-5" />;
    }
  };

  const getInsightColor = (type: string) => {
    switch (type) {
      case 'pattern': return 'text-blue-600 bg-blue-100';
      case 'recommendation': return 'text-green-600 bg-green-100';
      case 'warning': return 'text-orange-600 bg-orange-100';
      case 'success': return 'text-emerald-600 bg-emerald-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const getConfidenceColor = (score: number) => {
    if (score >= 0.8) return 'text-green-600';
    if (score >= 0.6) return 'text-yellow-600';
    return 'text-red-600';
  };

  if (loading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Brain className="h-5 w-5" />
            Behavioral Insights
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
            Behavioral Insights
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-4">
            <p className="text-sm text-muted-foreground mb-3">{error}</p>
            <Button onClick={fetchBehavioralData} variant="outline" size="sm">
              Retry
            </Button>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Brain className="h-5 w-5" />
          Behavioral Insights & Analysis
          <Badge variant="outline">{insights.length} Insights</Badge>
        </CardTitle>
        <CardDescription>
          AI-powered analysis of your trading patterns and decision-making
        </CardDescription>
      </CardHeader>
      <CardContent>
        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="insights">Insights</TabsTrigger>
            <TabsTrigger value="performance">Performance</TabsTrigger>
          </TabsList>
          
          <TabsContent value="insights" className="space-y-4">
            {insights.length === 0 ? (
              <div className="text-center py-8">
                <Brain className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
                <h3 className="text-lg font-semibold mb-2">No Insights Yet</h3>
                <p className="text-muted-foreground mb-4">
                  Start annotating your trades to generate behavioral insights.
                </p>
                <Button onClick={fetchBehavioralData} variant="outline">
                  Refresh
                </Button>
              </div>
            ) : (
              <div className="space-y-4">
                {insights.map((insight) => (
                  <Card key={insight.id} className="border-l-4 border-l-blue-500">
                    <CardContent className="p-4">
                      <div className="flex items-start justify-between mb-3">
                        <div className="flex items-center gap-2">
                          <div className={`p-2 rounded-full ${getInsightColor(insight.insight_type)}`}>
                            {getInsightIcon(insight.insight_type)}
                          </div>
                          <div>
                            <h3 className="font-semibold">{insight.title}</h3>
                            <div className="flex items-center gap-2">
                              <Badge variant="outline" className="text-xs">
                                {insight.insight_type}
                              </Badge>
                              <span className={`text-xs font-medium ${getConfidenceColor(insight.confidence_score)}`}>
                                Confidence: {Math.round(insight.confidence_score * 100)}%
                              </span>
                            </div>
                          </div>
                        </div>
                        <span className="text-xs text-muted-foreground">
                          {new Date(insight.created_at).toLocaleDateString()}
                        </span>
                      </div>
                      
                      <p className="text-sm text-muted-foreground mb-3">
                        {insight.description}
                      </p>
                      
                      {insight.actionable_recommendations && insight.actionable_recommendations.length > 0 && (
                        <div className="mb-3">
                          <h4 className="text-sm font-medium mb-2 flex items-center gap-1">
                            <Target className="h-4 w-4" />
                            Recommendations
                          </h4>
                          <ul className="space-y-1">
                            {insight.actionable_recommendations.map((rec, index) => (
                              <li key={index} className="text-sm flex items-start gap-2">
                                <span className="text-muted-foreground">•</span>
                                <span>{rec}</span>
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}
                      
                      {Object.keys(insight.supporting_data).length > 0 && (
                        <div className="bg-muted/50 rounded p-2">
                          <h4 className="text-xs font-medium mb-1">Supporting Data</h4>
                          <div className="text-xs text-muted-foreground">
                            {Object.entries(insight.supporting_data).map(([key, value]) => (
                              <div key={key} className="flex justify-between">
                                <span className="capitalize">{key.replace('_', ' ')}:</span>
                                <span>{String(value)}</span>
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
          </TabsContent>
          
          <TabsContent value="performance" className="space-y-4">
            {performanceAnalysis ? (
              <div className="space-y-6">
                {/* Performance Metrics */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <Card>
                    <CardContent className="p-4">
                      <div className="flex items-center gap-2 mb-2">
                        <Activity className="h-4 w-4 text-blue-600" />
                        <span className="text-sm font-medium">Total Trades</span>
                      </div>
                      <div className="text-2xl font-bold">{performanceAnalysis.total_trades}</div>
                    </CardContent>
                  </Card>
                  
                  <Card>
                    <CardContent className="p-4">
                      <div className="flex items-center gap-2 mb-2">
                        <CheckCircle className="h-4 w-4 text-green-600" />
                        <span className="text-sm font-medium">Win Rate</span>
                      </div>
                      <div className="text-2xl font-bold">{performanceAnalysis.win_rate}%</div>
                    </CardContent>
                  </Card>
                  
                  <Card>
                    <CardContent className="p-4">
                      <div className="flex items-center gap-2 mb-2">
                        <TrendingUp className="h-4 w-4 text-emerald-600" />
                        <span className="text-sm font-medium">Avg Return</span>
                      </div>
                      <div className="text-2xl font-bold">
                        {performanceAnalysis.average_return_percentage >= 0 ? '+' : ''}
                        {performanceAnalysis.average_return_percentage}%
                      </div>
                    </CardContent>
                  </Card>
                  
                  <Card>
                    <CardContent className="p-4">
                      <div className="flex items-center gap-2 mb-2">
                        <Clock className="h-4 w-4 text-purple-600" />
                        <span className="text-sm font-medium">Avg Hold</span>
                      </div>
                      <div className="text-2xl font-bold">{performanceAnalysis.average_hold_duration_days}d</div>
                    </CardContent>
                  </Card>
                </div>
                
                {/* Recent Performance */}
                {performanceAnalysis?.performances && performanceAnalysis.performances.length > 0 && (
                  <Card>
                    <CardHeader>
                      <CardTitle className="flex items-center gap-2">
                        <PieChart className="h-5 w-5" />
                        Recent Trade Performance
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-3">
                        {performanceAnalysis.performances.map((perf, index) => (
                          <div key={index} className="flex items-center justify-between p-3 bg-muted/20 rounded-lg">
                            <div className="flex items-center gap-3">
                              <Badge variant="outline">{perf.symbol}</Badge>
                              <span className="text-sm text-muted-foreground">
                                {perf.hold_duration_days}d hold
                              </span>
                            </div>
                            <div className="flex items-center gap-2">
                              <span className={`text-sm font-medium ${
                                (perf.return_percentage || 0) >= 0 ? 'text-green-600' : 'text-red-600'
                              }`}>
                                {perf.return_percentage >= 0 ? '+' : ''}{perf.return_percentage?.toFixed(2)}%
                              </span>
                              <span className="text-sm text-muted-foreground">
                                ${perf.total_return?.toFixed(2)}
                              </span>
                            </div>
                          </div>
                        ))}
                      </div>
                    </CardContent>
                  </Card>
                )}
                
                {/* Behavioral Recommendations */}
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <BookOpen className="h-5 w-5" />
                      Behavioral Recommendations
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      <div className="p-3 bg-blue-50 rounded-lg">
                        <h4 className="font-medium text-blue-900 mb-1">Improve Decision Consistency</h4>
                        <p className="text-sm text-blue-700">
                          Track your rationale quality scores and aim for consistency in your decision-making process.
                        </p>
                      </div>
                      
                      <div className="p-3 bg-green-50 rounded-lg">
                        <h4 className="font-medium text-green-900 mb-1">Optimize Hold Times</h4>
                        <p className="text-sm text-green-700">
                          Analyze your average hold duration and consider if it aligns with your strategy.
                        </p>
                      </div>
                      
                      <div className="p-3 bg-orange-50 rounded-lg">
                        <h4 className="font-medium text-orange-900 mb-1">Risk Management</h4>
                        <p className="text-sm text-orange-700">
                          Review your risk assessment patterns and ensure they're consistent with your risk tolerance.
                        </p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </div>
            ) : (
              <div className="text-center py-8">
                <BarChart3 className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
                <h3 className="text-lg font-semibold mb-2">No Performance Data</h3>
                <p className="text-muted-foreground mb-4">
                  Complete some trades to see performance analysis.
                </p>
              </div>
            )}
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
}
