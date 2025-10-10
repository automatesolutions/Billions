/**
 * ML Prediction Types
 */

export interface Prediction {
  ticker: string;
  current_price: number;
  predictions: number[];
  confidence_upper: number[];
  confidence_lower: number[];
  prediction_days: number;
  model_features: number;
  data_points: number;
  last_updated: string;
}

export interface StockInfo {
  symbol: string;
  name: string;
  sector: string;
  industry: string;
  market_cap: number;
  current_price: number;
  volume: number;
  avg_volume: number;
  pe_ratio?: number;
  dividend_yield?: number;
  "52_week_high"?: number;
  "52_week_low"?: number;
}

export interface TickerSearchResult {
  symbol: string;
  name: string;
}

export interface SearchResponse {
  query: string;
  results: TickerSearchResult[];
}

export interface StrategyInfo {
  strategy: string;
  x_period: string;
  y_period: string;
  lookback_x_days: number;
  lookback_y_days: number;
  min_market_cap: number;
}

export interface StrategiesResponse {
  strategies: StrategyInfo[];
}

export interface RefreshResponse {
  message: string;
  status: string;
  note?: string;
}

