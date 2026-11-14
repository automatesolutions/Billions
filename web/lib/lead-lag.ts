import type { LeadLagConfidence, LeadLagResult } from '@/types/api';

/** "68% (412 events)". The count always travels with the percentage. */
export function accuracyText(accuracy: number | null | undefined, events: number): string {
  if (accuracy === null || accuracy === undefined || !Number.isFinite(accuracy) || events === 0) return `No events (${events})`;
  return `${Math.round(accuracy * 100)}% (${events.toLocaleString('en-US')} ${events === 1 ? 'event' : 'events'})`;
}

export const CONFIDENCE: Record<LeadLagConfidence, { label: string; rank: number; dot: 'up' | 'muted' | 'down' }> = {
  strong: { label: 'Strong', rank: 3, dot: 'up' },
  moderate: { label: 'Moderate', rank: 2, dot: 'up' },
  weak: { label: 'Weak', rank: 1, dot: 'muted' },
  not_enough_data: { label: 'Not enough data', rank: 0, dot: 'down' },
};

export function lagText(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined) return 'No clear lag';
  if (seconds < 60) return `${Math.round(seconds)} s`;
  const minutes = seconds / 60;
  return `${Number.isInteger(minutes) ? minutes : minutes.toFixed(1)} min`;
}

/** Bot page link with this pair's lead-lag settings filled in. */
export function botHref(r: Pick<LeadLagResult, 'leader' | 'follower' | 'params'>): string {
  const q = new URLSearchParams({
    strategy: 'lead_lag',
    leader: r.leader,
    followers: r.follower,
    threshold: String(r.params.threshold_pct),
    lookback: String(r.params.lookback_seconds),
    max_wait: String(r.params.max_wait_seconds),
  });
  return `/bot?${q.toString()}`;
}

export type SortKey = 'pair' | 'lag' | 'accuracy' | 'oos' | 'move' | 'confidence' | 'updated';

export function sortValue(r: LeadLagResult, key: SortKey): number | string {
  switch (key) {
    case 'pair':
      return `${r.leader}/${r.follower}`;
    case 'lag':
      return r.lag_seconds ?? Number.POSITIVE_INFINITY;
    case 'accuracy':
      return r.accuracy ?? -1;
    case 'oos':
      return r.out_of_sample.accuracy ?? -1;
    case 'move':
      return r.avg_follower_move_pct ?? -1;
    case 'confidence':
      return CONFIDENCE[r.confidence].rank * 10 + (r.out_of_sample.accuracy ?? 0);
    case 'updated':
      return r.run_date ?? '';
  }
}

export function sortResults(rows: LeadLagResult[], key: SortKey, descending: boolean): LeadLagResult[] {
  return [...rows].sort((a, b) => {
    const va = sortValue(a, key);
    const vb = sortValue(b, key);
    const order = va < vb ? -1 : va > vb ? 1 : 0;
    return descending ? -order : order;
  });
}
