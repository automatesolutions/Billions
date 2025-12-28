/**
 * API client for BILLIONS backend
 */

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

console.log('API_BASE_URL:', API_BASE_URL);
console.log('NEXT_PUBLIC_API_URL:', process.env.NEXT_PUBLIC_API_URL);

export interface ApiError {
  detail: string;
}

export class ApiClient {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;
    
    console.log('API Request:', url);
    
    try {
      const response = await fetch(url, {
        ...options,
        headers: {
          'Content-Type': 'application/json',
          ...options.headers,
        },
      });

      console.log('API Response status:', response.status);
      console.log('API Response ok:', response.ok);

      if (!response.ok) {
        let errorMessage = `HTTP ${response.status}: ${response.statusText}`;
        try {
          const error: ApiError = await response.json();
          errorMessage = error.detail || errorMessage;
        } catch {
          // If we can't parse the error response, use the status text
        }
        console.error('API Error:', errorMessage);
        throw new Error(errorMessage);
      }

      const data = await response.json();
      console.log('API Response data:', data);
      return data;
    } catch (error) {
      console.error('Network error:', error);
      if (error instanceof TypeError && error.message.includes('fetch')) {
        throw new Error(`Network error: Unable to connect to ${url}. Please check if the backend server is running.`);
      }
      throw error;
    }
  }

  async get<T>(endpoint: string): Promise<T> {
    return this.request<T>(endpoint, { method: 'GET' });
  }

  async post<T>(endpoint: string, data: unknown): Promise<T> {
    return this.request<T>(endpoint, {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async put<T>(endpoint: string, data: unknown): Promise<T> {
    return this.request<T>(endpoint, {
      method: 'PUT',
      body: JSON.stringify(data),
    });
  }

  async delete<T>(endpoint: string): Promise<T> {
    return this.request<T>(endpoint, { method: 'DELETE' });
  }

  // Health check
  async healthCheck() {
    return this.get<{ status: string; service: string; version: string }>('/health');
  }

  // Ping
  async ping() {
    return this.get<{ message: string }>('/api/v1/ping');
  }

  // ML Predictions
  async getPrediction(ticker: string, days: number = 30) {
    return this.get<any>(`/api/v1/predictions/${ticker}?days=${days}`);
  }

  async getTickerInfo(ticker: string) {
    return this.get<any>(`/api/v1/predictions/info/${ticker}`);
  }

  async searchTickers(query: string, limit: number = 10) {
    return this.get<any>(`/api/v1/predictions/search?q=${query}&limit=${limit}`);
  }

  // Outliers
  async getOutliers(strategy: string) {
    return this.get<any>(`/api/v1/market/outliers/${strategy}`);
  }

  // Performance Metrics (ALL stocks - normal + outliers)
  async getPerformanceMetrics(strategy: string) {
    return this.get<any>(`/api/v1/market/performance/${strategy}`);
  }

  async getStrategies() {
    return this.get<any>('/api/v1/outliers/strategies');
  }

  async getStrategyInfo(strategy: string) {
    return this.get<any>(`/api/v1/outliers/${strategy}/info`);
  }

  async refreshOutliers(strategy: string) {
    return this.post<any>(`/api/v1/outliers/${strategy}/refresh`, {});
  }

  // User Management
  async createUser(userData: any) {
    return this.post<any>('/api/v1/users/', userData);
  }

  async getUser(userId: string) {
    return this.get<any>(`/api/v1/users/${userId}`);
  }

  async getUserPreferences(userId: string) {
    return this.get<any>(`/api/v1/users/${userId}/preferences`);
  }

  async updateUserPreferences(userId: string, preferences: any) {
    return this.put<any>(`/api/v1/users/${userId}/preferences`, preferences);
  }

  async getWatchlist(userId: string) {
    return this.get<any>(`/api/v1/users/${userId}/watchlist`);
  }

  async addToWatchlist(userId: string, symbol: string, name?: string, notes?: string) {
    const params = new URLSearchParams({ symbol });
    if (name) params.append('name', name);
    if (notes) params.append('notes', notes);
    return this.post<any>(`/api/v1/users/${userId}/watchlist?${params.toString()}`, {});
  }

  async removeFromWatchlist(userId: string, itemId: number) {
    return this.delete<any>(`/api/v1/users/${userId}/watchlist/${itemId}`);
  }

  // News & Sentiment
  async getNews(ticker: string, limit: number = 10) {
    return this.get<any>(`/api/v1/news/${ticker}?limit=${limit}`);
  }

  // Valuation & Fair Value
  async getValuation(ticker: string, daysBack: number = 252) {
    return this.get<any>(`/api/v1/valuation/${ticker}?days_back=${daysBack}`);
  }

  async getFairValue(ticker: string, daysBack: number = 252) {
    return this.get<any>(`/api/v1/valuation/${ticker}/fair-value?days_back=${daysBack}`);
  }

  // Portfolio Management
  async analyzeVolatility(ticker: string, days: number = 252) {
    return this.post<any>(`/api/v1/portfolio/analyze-volatility/${ticker}`, { days });
  }

  async calculateAllocation(tickers: string[], capital: number, riskTolerance: string = 'medium') {
    return this.post<any>('/api/v1/portfolio/calculate-allocation', {
      tickers,
      capital,
      risk_tolerance: riskTolerance
    });
  }

  async calculatePortfolioMetrics(holdings: any[]) {
    return this.post<any>('/api/v1/portfolio/calculate-metrics', holdings);
  }

  async getRiskAnalysis(ticker: string) {
    return this.get<any>(`/api/v1/portfolio/risk-analysis/${ticker}`);
  }

  // Trading & Real-time Data
  async getTradingStatus() {
    return this.get<any>('/api/v1/trading/status');
  }

  async getAccountInfo() {
    return this.get<any>('/api/v1/trading/account');
  }

  async getPositions() {
    return this.get<any>('/api/v1/trading/positions');
  }

  async getOrders(status: string = 'all') {
    return this.get<any>(`/api/v1/trading/orders?status=${status}`);
  }

  async getRealTimeQuote(symbol: string) {
    return this.post<any>(`/api/v1/trading/quote/${symbol}`);
  }

  async getOrderbook(symbol: string) {
    return this.post<any>(`/api/v1/trading/orderbook/${symbol}`);
  }

  async getMarketData(symbols: string[]) {
    return this.post<any>('/api/v1/trading/market-data', symbols);
  }

  async executeTrade(symbol: string, qty: number, side: string, orderType: string = 'market') {
    return this.post<any>('/api/v1/trading/execute', {
      symbol,
      qty,
      side,
      order_type: orderType
    });
  }

  async getPortfolio() {
    return this.get<any>('/api/v1/trading/portfolio');
  }

  async getMarketStatus() {
    return this.get<any>('/api/v1/trading/market-status');
  }

  async syncPortfolio() {
    return this.post<any>('/api/v1/trading/sync-portfolio');
  }

  async getBulkQuotes(symbols: string[]) {
    return this.post<any>('/api/v1/trading/bulk-quotes', symbols);
  }

  // Capitulation Detection
  async screenCapitulation(limit: number = 20) {
    return this.get<any>(`/api/v1/capitulation/screen?limit=${limit}`);
  }

  async getCapitulationSummary() {
    return this.get<any>('/api/v1/capitulation/summary');
  }

  async analyzeStockCapitulation(symbol: string) {
    return this.get<any>(`/api/v1/capitulation/analyze/${symbol}`);
  }

  async getCapitulationIndicators() {
    return this.get<any>('/api/v1/capitulation/indicators');
  }

  async getCapitulationStats() {
    return this.get<any>('/api/v1/capitulation/stats');
  }

  // HFT Endpoints
  async hftStatus() {
    return this.get<any>('/api/v1/hft/status');
  }

  async hftPerformance() {
    return this.get<any>('/api/v1/hft/performance');
  }

  async hftStart() {
    return this.post<any>('/api/v1/hft/start', {});
  }

  async hftStop() {
    return this.post<any>('/api/v1/hft/stop', {});
  }

  async hftSubmitOrder(payload: {
    order_type: 'market' | 'limit' | 'twap' | 'vwap';
    symbol: string;
    side: 'buy' | 'sell';
    quantity: number;
    price?: number;
    duration_minutes?: number;
    interval_seconds?: number;
    volume_weight?: number;
    time_in_force?: 'day' | 'gtc' | 'fok' | 'ioc' | 'opg' | 'cls';
  }) {
    return this.post<any>('/api/v1/hft/orders', payload);
  }

  async hftOrderTypes() {
    return this.get<any>('/api/v1/hft/order-types');
  }

  async hftSymbols() {
    return this.get<any>('/api/v1/hft/symbols');
  }

  async hftHealth() {
    return this.get<any>('/api/v1/hft/health');
  }

  async hftClearAllOrders() {
    return this.delete<any>('/api/v1/hft/orders/all');
  }

  // NASDAQ News methods
  async getNASDAQNews(limit: number = 10) {
    return this.get<any>(`/api/v1/nasdaq-news/latest?limit=${limit}`);
  }

  async getUrgentNASDAQNews(limit: number = 5) {
    return this.get<any>(`/api/v1/nasdaq-news/urgent?limit=${limit}`);
  }

  async getNASDAQNewsByCategory(category: string, limit: number = 10) {
    return this.get<any>(`/api/v1/nasdaq-news/category/${category}?limit=${limit}`);
  }

  // Behavioral Trading methods
  async addTradeRationale(rationale: any) {
    return this.post<any>('/api/v1/behavioral/rationale', rationale);
  }

  async getTradeRationales(tradeId: string) {
    return this.get<any>(`/api/v1/behavioral/rationale/${tradeId}`);
  }

  async addPositionAnnotation(annotation: any) {
    return this.post<any>('/api/v1/behavioral/position-annotation', annotation);
  }

  async getPositionAnnotations(symbol?: string) {
    const url = symbol 
      ? `/api/v1/behavioral/position-annotations?symbol=${symbol}`
      : '/api/v1/behavioral/position-annotations';
    return this.get<any>(url);
  }

  async executeExitDecision(exitDecision: any) {
    return this.post<any>('/api/v1/behavioral/exit-decision', exitDecision);
  }

  async executeAdditionDecision(additionDecision: any) {
    return this.post<any>('/api/v1/behavioral/addition-decision', additionDecision);
  }

  async getBehavioralInsights(limit: number = 10) {
    return this.get<any>(`/api/v1/behavioral/insights?limit=${limit}`);
  }

  async getTradePerformanceAnalysis(symbol?: string) {
    const url = symbol 
      ? `/api/v1/behavioral/performance-analysis?symbol=${symbol}`
      : '/api/v1/behavioral/performance-analysis';
    return this.get<any>(url);
  }

  async getHoldingContext(symbol: string) {
    return this.get<any>(`/api/v1/behavioral/holdings/${symbol}/context`);
  }

  async updateTradeRationale(rationaleId: string, updates: any) {
    return this.put<any>(`/api/v1/behavioral/rationale/${rationaleId}`, updates);
  }

  async deleteTradeRationale(rationaleId: string) {
    return this.delete<any>(`/api/v1/behavioral/rationale/${rationaleId}`);
  }
}

export const api = new ApiClient();

