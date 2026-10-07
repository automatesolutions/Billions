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

// ---------------------------------------------------------------- analysis

export interface PricePoint {
  date: string;
  close: number;
}

export interface RiskContext {
  annual_volatility: number;
  max_drawdown: number;
  worst_day: { date: string; return: number };
  sharpe: number | null;
  total_return: number;
  days: number;
}

export interface ModelRow {
  model: 'ar1' | 'xgboost' | 'online' | 'stacked';
  hit_rate: number | null;
  expected_value: number | null;
  sharpe: number | null;
  total_log_return: number | null;
  n: number;
}

export interface WalkForward {
  scheme: 'expanding' | 'rolling';
  folds: { start: string; train_days: number; test_days: number; hit_rate: number | null }[];
  fold_count: number;
  n: number;
  hit_rate: number | null;
  sharpe: number | null;
  expected_value: number | null;
}

interface AnalysisBase {
  ticker: string;
  price: number;
  as_of: string;
  session_close: string;
  history: PricePoint[];
  default_cost_bps: number;
  generated_at: string;
}

export interface AnalysisInsufficient extends AnalysisBase {
  status: 'insufficient_history';
  message: string;
  needed_days: number;
  history_days: number;
  risk: RiskContext | null;
}

export interface AnalysisOk extends AnalysisBase {
  status: 'ok';
  history_days: number;
  horizon_days: number;
  signal: {
    strength: number;
    direction: 'up' | 'down' | 'neutral';
    label: 'weak' | 'moderate' | 'strong';
    daily_volatility: number;
    forecast_log_return: number | null;
    forecast_for: string | null;
    regime: { regime: 'mean_reversion' | 'momentum'; w: number; t_stat: number | null; significant: boolean };
  };
  forecast: { for_date: string | null; expected_price: number | null; low: number; high: number; band: string };
  edge: {
    n: number;
    win_rate: number | null;
    avg_win: number;
    avg_loss: number;
    expected_value: number | null;
    sharpe: number | null;
    total_log_return: number | null;
    sample: 'out_of_sample';
    equity: number[];
    hold_equity: number[];
    drawdown: number[];
    max_drawdown: number;
    dates: string[];
  };
  in_sample: { hit_rate: number | null; sharpe: number | null; expected_value: number | null; n: number; sample: 'in_sample' };
  risk: RiskContext;
  models: ModelRow[];
  meta_weights: Record<'ar1' | 'xgboost' | 'online', number>;
  meta_bias: number;
  validation: {
    split: { train_days: number; test_days: number; train_start: string; test_start: string; test_end: string; shuffled: false };
    walk_forward: WalkForward[];
    random_baseline: {
      draws: number;
      random_mean: number;
      random_p95: number;
      model_total: number;
      excess_vs_random_mean: number;
      percentile: number;
      p_value: number;
      beats_random: boolean;
    };
  };
  cost_check: {
    round_trip_cost_bps: number;
    expected_move_bps: number;
    latest_move_bps: number;
    expected_value_bps: number;
    covers_cost: boolean;
  };
  microstructure: { available: false; reason: string };
  caveats: string[];
}

export type AnalysisResponse = AnalysisOk | AnalysisInsufficient;
