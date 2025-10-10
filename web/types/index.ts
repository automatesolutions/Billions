/**
 * Shared TypeScript types for BILLIONS
 */

export interface PerfMetric {
  symbol: string;
  metric_x: number | null;
  metric_y: number | null;
  z_x: number | null;
  z_y: number | null;
  is_outlier: boolean;
  inserted?: string;
}

export interface OutliersResponse {
  strategy: string;
  count: number;
  outliers: PerfMetric[];
}

export interface PerformanceMetricsResponse {
  strategy: string;
  count: number;
  metrics: PerfMetric[];
}

export type Strategy = 'scalp' | 'swing' | 'longterm';

export interface HealthCheck {
  status: string;
  service: string;
  version: string;
}

