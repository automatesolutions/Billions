import { useState, useEffect } from 'react';
import { api } from '@/lib/api';

interface ValuationData {
  ticker: string;
  current_price: number;
  fair_value: number;
  valuation_status: string;
  valuation_color: string;
  valuation_ratio: number;
  volatility: number;
  risk_free_rate: number;
  analysis_date: string;
  price_change_1d?: number;
  price_change_30d?: number;
  price_change_1y?: number;
  beta?: number;
  sharpe_ratio?: number;
}

export function useValuation(ticker: string, daysBack: number = 252) {
  const [data, setData] = useState<ValuationData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchValuation = async () => {
      try {
        setLoading(true);
        setError(null);
        console.log('Fetching valuation for:', ticker, 'days:', daysBack);
        const result = await api.getFairValue(ticker, daysBack);
        console.log('Valuation result:', result);
        setData(result);
      } catch (err) {
        console.error('Valuation error:', err);
        setError(err instanceof Error ? err.message : 'Failed to fetch valuation');
      } finally {
        setLoading(false);
      }
    };

    if (ticker) {
      fetchValuation();
    }
  }, [ticker, daysBack]);

  return { data, loading, error };
}
