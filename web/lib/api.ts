/**
 * Read-only client for the BILLIONS API. Works on the server and in the browser.
 */

import type { AnalysisResponse, OutliersResponse, QuantInterval, QuantResponse, QuoteResponse, Strategy } from '@/types/api';

/** Server-side code may use a private URL (API_URL); the browser uses the public one. */
export const API_BASE_URL =
  (typeof window === 'undefined' ? process.env.API_URL : undefined) ||
  process.env.NEXT_PUBLIC_API_URL ||
  'http://localhost:8000';

export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
  ) {
    super(message);
  }
}

async function get<T>(path: string, init?: RequestInit & { next?: { revalidate?: number } }): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, { ...init, signal: init?.signal ?? AbortSignal.timeout(15000) });
  } catch {
    throw new ApiError('The data service is not reachable.', 0);
  }
  if (!response.ok) {
    let message = `The data service returned an error (${response.status}).`;
    try {
      const body = await response.json();
      if (typeof body.detail === 'string') message = body.detail;
    } catch {
      // Body was not JSON; keep the generic message.
    }
    throw new ApiError(message, response.status);
  }
  return response.json() as Promise<T>;
}

export function getOutliers(strategy: Strategy, init?: Parameters<typeof get>[1]) {
  return get<OutliersResponse>(`/api/v1/outliers/${strategy}`, init);
}

export function getAnalysis(ticker: string, init?: Parameters<typeof get>[1]) {
  return get<AnalysisResponse>(`/api/v1/analysis/${encodeURIComponent(ticker)}`, init);
}

export function getQuote(ticker: string, init?: Parameters<typeof get>[1]) {
  return get<QuoteResponse>(`/api/v1/quotes/${encodeURIComponent(ticker)}`, { cache: 'no-store', ...init });
}

export function getQuant(ticker: string, interval: QuantInterval, init?: Parameters<typeof get>[1]) {
  return get<QuantResponse>(`/api/v1/quant/${encodeURIComponent(ticker)}?interval=${interval}`, { cache: 'no-store', ...init });
}

export function getWatching(init?: Parameters<typeof get>[1]) {
  return get<{ watching: { ticker: string; interval: QuantInterval }[]; max: number }>('/api/v1/quant/watching', { cache: 'no-store', ...init });
}

export function setWatch(ticker: string, interval: QuantInterval, on: boolean) {
  return get<{ watching: boolean }>(`/api/v1/quant/${encodeURIComponent(ticker)}/watch?interval=${interval}`, { method: on ? 'POST' : 'DELETE' });
}
