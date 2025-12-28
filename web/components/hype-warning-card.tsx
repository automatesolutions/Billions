"use client";

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { AlertTriangle, TrendingUp, AlertCircle } from 'lucide-react';

interface HypeWarningCardProps {
  hypeAnalysis: {
    overall_status: string;
    hype_articles_count: number;
    average_hype_score: number;
    total_hype_score: number;
  };
  caveatEmptor: {
    overall_status: string;
    risky_articles_count: number;
    average_risk_score: number;
    total_risk_score: number;
  };
}

export function HypeWarningCard({ hypeAnalysis, caveatEmptor }: HypeWarningCardProps) {
  const getHypeColor = (status: string) => {
    switch (status) {
      case 'HIGH HYPE':
        return 'bg-red-900/30 text-red-400 border-red-500/30';
      case 'MODERATE HYPE':
        return 'bg-yellow-900/30 text-yellow-400 border-yellow-500/30';
      default:
        return 'bg-green-900/30 text-green-400 border-green-500/30';
    }
  };

  const getRiskColor = (status: string) => {
    switch (status) {
      case 'HIGH RISK':
        return 'bg-red-900/30 text-red-400 border-red-500/30';
      case 'MODERATE RISK':
        return 'bg-orange-900/30 text-orange-400 border-orange-500/30';
      default:
        return 'bg-green-900/30 text-green-400 border-green-500/30';
    }
  };

  const getHypeIcon = (status: string) => {
    switch (status) {
      case 'HIGH HYPE':
        return <TrendingUp className="h-4 w-4 text-red-400" />;
      case 'MODERATE HYPE':
        return <TrendingUp className="h-4 w-4 text-yellow-400" />;
      default:
        return <TrendingUp className="h-4 w-4 text-green-400" />;
    }
  };

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      {/* HYPE Analysis */}
      <Card className="border-l-4 border-l-orange-500 bg-gray-800 border-gray-700">
        <CardHeader className="pb-3">
          <CardTitle className="text-lg flex items-center gap-2 text-white">
            {getHypeIcon(hypeAnalysis.overall_status)}
            HYPE Analysis
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="flex items-center justify-between">
            <span className="font-medium text-gray-300">Hype Level:</span>
            <Badge className={getHypeColor(hypeAnalysis.overall_status)}>
              {hypeAnalysis.overall_status}
            </Badge>
          </div>
          
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div className="text-center p-2 bg-gray-700/50 rounded border border-gray-600">
              <div className="font-semibold text-white">{hypeAnalysis.hype_articles_count}</div>
              <div className="text-gray-400">Hype Articles</div>
            </div>
            <div className="text-center p-2 bg-gray-700/50 rounded border border-gray-600">
              <div className="font-semibold text-white">{hypeAnalysis.average_hype_score}</div>
              <div className="text-gray-400">Avg Score</div>
            </div>
          </div>

          {hypeAnalysis.overall_status === 'HIGH HYPE' && (
            <div className="p-3 bg-red-900/20 border border-red-500/30 rounded-lg">
              <div className="flex items-center gap-2 mb-2">
                <AlertTriangle className="h-4 w-4 text-red-400" />
                <span className="font-semibold text-red-400">High Hype Detected!</span>
              </div>
              <p className="text-sm text-red-300">
                Multiple articles contain sensational language, pump terminology, or unrealistic claims. 
                Exercise caution before making investment decisions.
              </p>
            </div>
          )}

          {hypeAnalysis.overall_status === 'MODERATE HYPE' && (
            <div className="p-3 bg-yellow-900/20 border border-yellow-500/30 rounded-lg">
              <div className="flex items-center gap-2 mb-2">
                <AlertTriangle className="h-4 w-4 text-yellow-400" />
                <span className="font-semibold text-yellow-400">Moderate Hype Detected</span>
              </div>
              <p className="text-sm text-yellow-300">
                Some articles contain promotional language. Review carefully before investing.
              </p>
            </div>
          )}
        </CardContent>
      </Card>

      {/* CAVEAT EMPTOR */}
      <Card className="border-l-4 border-l-red-500 bg-gray-800 border-gray-700">
        <CardHeader className="pb-3">
          <CardTitle className="text-lg flex items-center gap-2 text-white">
            <AlertCircle className="h-4 w-4 text-red-400" />
            Caveat Emptor
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="flex items-center justify-between">
            <span className="font-medium text-gray-300">Risk Level:</span>
            <Badge className={getRiskColor(caveatEmptor.overall_status)}>
              {caveatEmptor.overall_status}
            </Badge>
          </div>
          
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div className="text-center p-2 bg-gray-700/50 rounded border border-gray-600">
              <div className="font-semibold text-white">{caveatEmptor.risky_articles_count}</div>
              <div className="text-gray-400">Risky Articles</div>
            </div>
            <div className="text-center p-2 bg-gray-700/50 rounded border border-gray-600">
              <div className="font-semibold text-white">{caveatEmptor.average_risk_score}</div>
              <div className="text-gray-400">Avg Risk</div>
            </div>
          </div>

          {caveatEmptor.overall_status === 'HIGH RISK' && (
            <div className="p-3 bg-red-900/20 border border-red-500/30 rounded-lg">
              <div className="flex items-center gap-2 mb-2">
                <AlertCircle className="h-4 w-4 text-red-400" />
                <span className="font-semibold text-red-400">HIGH RISK WARNING!</span>
              </div>
              <p className="text-sm text-red-300">
                Multiple risk indicators detected including financial concerns, 
                regulatory issues, or market volatility. Proceed with extreme caution.
              </p>
            </div>
          )}

          {caveatEmptor.overall_status === 'MODERATE RISK' && (
            <div className="p-3 bg-orange-900/20 border border-orange-500/30 rounded-lg">
              <div className="flex items-center gap-2 mb-2">
                <AlertCircle className="h-4 w-4 text-orange-400" />
                <span className="font-semibold text-orange-400">Moderate Risk</span>
              </div>
              <p className="text-sm text-orange-300">
                Some risk indicators present. Conduct thorough due diligence before investing.
              </p>
            </div>
          )}

          <div className="text-xs text-gray-500 pt-2 border-t border-gray-700">
            <strong>Caveat Emptor:</strong> "Let the buyer beware" - Always do your own research 
            and consult financial advisors before making investment decisions.
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
