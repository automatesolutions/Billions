'use client';

import { useState, useEffect } from 'react';
import { api } from '@/lib/api';

export function useTickerInfo(ticker: string) {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchInfo = async () => {
      try {
        setLoading(true);
        setError(null);
        const result = await api.getTickerInfo(ticker);
        setData(result);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch ticker info');
      } finally {
        setLoading(false);
      }
    };

    if (ticker) {
      fetchInfo();
    }
  }, [ticker]);

  return { data, loading, error };
}

