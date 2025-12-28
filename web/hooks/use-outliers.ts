'use client';

import { useState, useEffect, useCallback } from 'react';
import { api } from '@/lib/api';
import type { OutliersResponse } from '@/types';
import { useAutoRefresh } from './use-auto-refresh';

export function useOutliers(strategy: string, autoRefresh: boolean = false) {
  const [data, setData] = useState<OutliersResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchOutliers = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      console.log('Fetching outliers for strategy:', strategy);
      const result = await api.getOutliers(strategy);
      console.log('Outliers result:', result);
      setData(result);
    } catch (err) {
      console.error('Outliers fetch error:', err);
      setError(err instanceof Error ? err.message : 'Failed to fetch outliers');
    } finally {
      setLoading(false);
    }
  }, [strategy]);

  useEffect(() => {
    if (strategy) {
      fetchOutliers();
    }
  }, [strategy, fetchOutliers]);

  // Auto-refresh every 5 minutes if enabled
  useAutoRefresh(fetchOutliers, 300000, autoRefresh);

  return { data, loading, error, refetch: fetchOutliers };
}

