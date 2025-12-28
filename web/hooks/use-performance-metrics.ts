'use client';

import { useState, useEffect, useCallback } from 'react';
import { api } from '@/lib/api';
import type { PerformanceMetricsResponse } from '@/types';
import { useAutoRefresh } from './use-auto-refresh';

export function usePerformanceMetrics(strategy: string, autoRefresh: boolean = false) {
  const [data, setData] = useState<PerformanceMetricsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchPerformanceMetrics = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      console.log('Fetching performance metrics for strategy:', strategy);
      const result = await api.getPerformanceMetrics(strategy);
      console.log('Performance metrics result:', result);
      setData(result);
    } catch (err) {
      console.error('Performance metrics fetch error:', err);
      setError(err instanceof Error ? err.message : 'Failed to fetch performance metrics');
    } finally {
      setLoading(false);
    }
  }, [strategy]);

  useEffect(() => {
    if (strategy) {
      fetchPerformanceMetrics();
    }
  }, [strategy, fetchPerformanceMetrics]);

  // Auto-refresh every 5 minutes if enabled
  useAutoRefresh(fetchPerformanceMetrics, 300000, autoRefresh);

  return { data, loading, error, refetch: fetchPerformanceMetrics };
}
