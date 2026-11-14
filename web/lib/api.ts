/**
 * Client for the BILLIONS API. Works on the server and in the browser.
 * Everything is read-only except the paper bot and lead-lag research calls at the end.
 */

import type {
  AnalysisResponse,
  BotInfo,
  BotStartRequest,
  BotStatus,
  LeadLagPair,
  LeadLagResult,
  LeadLagSource,
  OutliersResponse,
  QuantInterval,
  QuantResponse,
  QuoteResponse,
  ScanEstimate,
  ScanStatus,
  Strategy,
} from '@/types/api';

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
      // Validation errors (422) arrive as a list; the first one says what to fix.
      else if (Array.isArray(body.detail) && typeof body.detail[0]?.msg === 'string')
        message = body.detail[0].msg.replace(/^Value error, /, '');
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

// ---------------------------------------------------------------- paper bot and lead-lag research

/** JSON request, with the bot token when you've entered one. */
function send<T>(path: string, method: 'GET' | 'POST' | 'DELETE', body?: unknown, token?: string) {
  const headers: Record<string, string> = {};
  if (body !== undefined) headers['Content-Type'] = 'application/json';
  if (token) headers['X-Bot-Token'] = token;
  return get<T>(path, { method, headers, cache: 'no-store', body: body === undefined ? undefined : JSON.stringify(body) });
}

export const getBotInfo = () => send<BotInfo>('/api/v1/bot/info', 'GET');
export const getBotStatus = (token?: string) => send<BotStatus>('/api/v1/bot/status', 'GET', undefined, token);
export const startBot = (config: BotStartRequest, token?: string) => send<{ running: boolean }>('/api/v1/bot/start', 'POST', config, token);
export const stopBot = (token?: string) => send<{ running: boolean }>('/api/v1/bot/stop', 'POST', {}, token);
export const killBot = (closePositions: boolean, token?: string) =>
  send<{ running: boolean; message: string }>('/api/v1/bot/kill', 'POST', { close_positions: closePositions }, token);

export interface AnalyzeRequest {
  a: string;
  b: string;
  threshold_pct: number;
  lookback_seconds: number;
  max_wait_seconds: number;
  days: number;
  source?: LeadLagSource;
}

export const analyzePair = (body: AnalyzeRequest) => send<LeadLagResult>('/api/v1/lead-lag/analyze', 'POST', body);
export const getLeadLagResults = () =>
  send<{ run_date: string | null; results: LeadLagResult[]; scan: ScanStatus | null }>('/api/v1/lead-lag/results', 'GET');
export const getPairHistory = (a: string, b: string) =>
  send<{ pair_key: string; history: LeadLagResult[]; latest: LeadLagResult | null }>(
    `/api/v1/lead-lag/pairs/${encodeURIComponent(a)}/${encodeURIComponent(b)}`,
    'GET',
  );
export const estimateScan = (body: { watchlist: boolean; candidates: boolean }) =>
  send<ScanEstimate>('/api/v1/lead-lag/scan/estimate', 'POST', body);
export const startScan = (body: { watchlist: boolean; candidates: boolean }) => send<ScanStatus>('/api/v1/lead-lag/scan', 'POST', body);
export const getScan = () => send<{ scan: ScanStatus | null }>('/api/v1/lead-lag/scan', 'GET');
export const getWatchlist = () => send<{ pairs: LeadLagPair[]; max: number }>('/api/v1/lead-lag/watchlist', 'GET');
export const addWatch = (a: string, b: string) => send<{ pairs: LeadLagPair[] }>('/api/v1/lead-lag/watchlist', 'POST', { a, b });
export const removeWatch = (a: string, b: string) =>
  send<{ pairs: LeadLagPair[] }>(`/api/v1/lead-lag/watchlist/${encodeURIComponent(a)}/${encodeURIComponent(b)}`, 'DELETE');
export const getCandidates = () => send<{ pairs: LeadLagPair[] }>('/api/v1/lead-lag/candidates', 'GET');
