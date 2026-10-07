/**
 * Read-only client for the BILLIONS API. Works on the server and in the browser.
 */

import type { OutliersResponse, Strategy } from '@/types/api';

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
