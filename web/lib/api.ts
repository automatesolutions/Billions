/**
 * Read-only client for the BILLIONS API.
 */

import type { PerformanceMetricsResponse } from '@/types';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

async function get<T>(endpoint: string): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${endpoint}`);
  if (!response.ok) {
    let message = `HTTP ${response.status}`;
    try {
      const body = await response.json();
      message = body.detail || message;
    } catch {
      // Body was not JSON; keep the status message.
    }
    throw new Error(message);
  }
  return response.json();
}

export const api = {
  getPerformanceMetrics: (strategy: string) =>
    get<PerformanceMetricsResponse>(`/api/v1/outliers/${encodeURIComponent(strategy)}`),
};
