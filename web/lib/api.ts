/**
 * API client for BILLIONS backend
 */

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

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
    
    const response = await fetch(url, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
    });

    if (!response.ok) {
      const error: ApiError = await response.json();
      throw new Error(error.detail || 'An error occurred');
    }

    return response.json();
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
}

export const api = new ApiClient();

