'use client';

import { useState, useEffect } from 'react';
import { api } from '@/lib/api';

export function usePrediction(ticker: string, days: number = 30) {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchPrediction = async () => {
      try {
        setLoading(true);
        setError(null);
        console.log('Fetching prediction for:', ticker, 'days:', days);
        const result = await api.getPrediction(ticker, days);
        console.log('Prediction result:', result);
        setData(result);
      } catch (err) {
        console.error('Prediction error:', err);
        setError(err instanceof Error ? err.message : 'Failed to fetch prediction');
      } finally {
        setLoading(false);
      }
    };

    if (ticker) {
      fetchPrediction();
    }
  }, [ticker, days]);

  return { data, loading, error, refetch: () => {} };
}

