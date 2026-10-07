/** Shapes returned by the BILLIONS API. */

export const STRATEGIES = ['scalp', 'swing', 'longterm'] as const;
export type Strategy = (typeof STRATEGIES)[number];

export type Direction = 'up' | 'down' | 'mixed' | 'neutral';

export interface MarketState {
  is_open: boolean;
  state: 'open' | 'pre_market' | 'after_hours' | 'weekend' | 'holiday';
  next_open: string | null;
  next_close: string | null;
  last_close: string;
}

export interface RefreshStatus {
  is_running: boolean;
  last_success: string | null;
  last_attempt: string | null;
  last_error: { at: string; message: string } | null;
}

export interface ScatterPoint {
  symbol: string;
  x: number;
  y: number;
  z_x: number;
  z_y: number;
  is_outlier: boolean;
}

export interface RankedOutlier {
  rank: number;
  symbol: string;
  score: number;
  direction: Exclude<Direction, 'neutral'>;
  reason: string;
  x: number;
  y: number;
  z_x: number;
  z_y: number;
}

export interface OutliersResponse {
  strategy: Strategy;
  x_label: string;
  y_label: string;
  x_days: number;
  y_days: number;
  min_dollar_volume: number;
  z_threshold: number;
  as_of: string | null;
  computed_at: string | null;
  market: MarketState;
  refresh: RefreshStatus;
  universe_count: number;
  outlier_count: number;
  outliers: RankedOutlier[];
  points: ScatterPoint[];
}
