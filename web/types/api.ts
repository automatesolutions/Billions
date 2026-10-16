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
    random_baseline: RandomBaseline & { beats_random: boolean };
  };
  cost_check: {
    round_trip_cost_bps: number;
    expected_move_bps: number;
    latest_move_bps: number;
    expected_value_bps: number;
    covers_cost: boolean;
  };
  caveats: string[];
}

export type AnalysisResponse = AnalysisOk | AnalysisInsufficient;

/** Live best bid and ask (top of book) for one ticker. */
export interface QuoteOk {
  ticker: string;
  available: true;
  source: string;
  feed: string;
  market_open: boolean;
  bid: number;
  ask: number;
  bid_size: number;
  ask_size: number;
  mid: number;
  spread: number;
  spread_bps: number;
  /** (bid size − ask size) / (bid size + ask size), from −1 to +1. */
  imbalance: number | null;
  /** Size-weighted mid-price. */
  microprice: number | null;
  quoted_at: string | null;
  age_seconds: number | null;
}

export interface QuoteUnavailable {
  ticker: string;
  available: false;
  reason: string;
  source?: string;
  feed?: string;
  market_open?: boolean;
}

export type QuoteResponse = QuoteOk | QuoteUnavailable;

// ---------------------------------------------------------------- quant page

export interface RandomBaseline {
  draws: number;
  random_mean: number;
  random_p95: number;
  model_total: number;
  excess_vs_random_mean: number;
  percentile: number;
  p_value: number;
}

export const QUANT_INTERVALS = ['1m', '1h', '4h', '1d'] as const;
export type QuantInterval = (typeof QUANT_INTERVALS)[number];

export interface QuantGate {
  id: 'same_session' | 'beats_random' | 'walk_forward' | 'after_costs' | 'live_record' | 'strength';
  label: string;
  passed: boolean;
  detail: string;
}

export type QuantOrder =
  | 'buy'
  | 'sell_short'
  | 'hold_flat'
  | 'hold_long'
  | 'hold_short'
  | 'sell'
  | 'buy_to_cover'
  | 'reverse_to_short'
  | 'reverse_to_long';

export interface QuantAction {
  action: 'long' | 'short' | 'flat';
  model_direction: 'long' | 'short' | 'neutral';
  /** Share of the maximum position, −1 (full short) to +1 (full long). 0 when a gate fails. */
  position: number;
  strength: number;
  forecast_log_return: number;
  forecast_bps: number;
  expected_price: number;
  for_bar: string;
  gates: QuantGate[];
  /** The order that moves the guide from the position it held over the last bar to `position`. */
  instruction: { code: QuantOrder; headline: string; sentence: string; previous_position: number; change: number };
}

export interface QuantWalk {
  scheme: 'expanding' | 'rolling';
  fold_count: number;
  n: number;
  hit_rate: number | null;
  sharpe: number | null;
  expected_value: number | null;
}

export interface QuantSignalRow {
  bar_time: string;
  target_time: string;
  action: 'long' | 'short' | 'flat';
  position: number;
  instruction: QuantOrder;
  forecast_bps: number;
  realized_bps: number | null;
  correct: boolean | null;
  status: 'open' | 'scored' | 'void';
}

export type QuantVerdict = 'too_early' | 'in_line' | 'underperforming' | 'outperforming';

export interface QuantBase {
  ticker: string;
  interval: QuantInterval;
  interval_label: string;
  price: number;
  last_bar: string;
  next_bar: string;
  next_update_at: string;
  /** When the last bar became final: the start of the wait for the next one. */
  last_update_at: string;
  market: MarketState;
  watching: boolean;
  generated_at: string;
}

export interface QuantInsufficient extends QuantBase {
  status: 'insufficient_history';
  message: string;
  needed_bars: number;
}

export interface QuantOk extends QuantBase {
  status: 'ok';
  bars: number;
  cost: { bps: number; source: 'live_spread' | 'default' };
  action: QuantAction;
  explain: {
    features: Record<'lag_1' | 'lag_2' | 'lag_3' | 'lag_5' | 'ma_5' | 'ma_20' | 'dir_1' | 'dirma_5' | 'dirma_20', number>;
    slot: number | null;
    base: { ar1: number; xgboost: number; online: number };
    meta_weights: { ar1: number; xgboost: number; online: number };
    meta_bias: number;
    ar1: { w: number; b: number };
    forecast: number;
    bar_volatility: number;
    strength_scale: number;
    neutral_band: number;
    slot_minutes: number | null;
    flat_overnight: boolean;
  };
  backtest: {
    split: { train_bars: number; test_bars: number; train_start: string; test_start: string; test_end: string; shuffled: false };
    models: ModelRow[];
    hit_rate: number;
    gross: { n: number; win_rate: number; expected_value: number | null; sharpe: number | null; total_log_return: number | null };
    net: {
      n: number;
      win_rate: number;
      avg_win: number;
      avg_loss: number;
      expected_value: number | null;
      sharpe: number | null;
      total_log_return: number | null;
      cost_bps: number;
      avg_turnover: number;
      equity: number[];
      hold_equity: number[];
      drawdown: number[];
      max_drawdown: number;
      dates: string[];
      expected_move_bps: number;
    };
    walk_forward: QuantWalk[];
    random_baseline: RandomBaseline & { beats_random: boolean };
    regime: { regime: 'momentum' | 'mean_reversion'; w: number; t_stat: number; significant: boolean };
    periods_per_year: number;
  };
  live: {
    verdict: QuantVerdict;
    min_signals: number;
    recorded: number;
    scored: number;
    pending: number;
    model: { hit_rate: number | null; expected_value: number | null; total_log_return: number | null; hit_rate_z: number | null };
    action: { expected_value: number | null; total_log_return: number | null; equity: number[]; dates: string[]; trades: number };
    backtest: { hit_rate: number; expected_value: number | null; hit_rate_band: [number, number] | null; expected_value_band: [number, number] | null };
    random_baseline: RandomBaseline | null;
    recent: QuantSignalRow[];
  };
  caveats: string[];
}

export type QuantResponse = QuantOk | QuantInsufficient;
