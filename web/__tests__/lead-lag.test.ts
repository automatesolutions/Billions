import { describe, expect, it } from 'vitest';
import { accuracyText, botHref, lagText, sortResults } from '@/lib/lead-lag';
import type { LeadLagResult } from '@/types/api';

const row = (leader: string, accuracy: number | null, oos: number | null, confidence: LeadLagResult['confidence']) =>
  ({ leader, follower: 'X', accuracy, out_of_sample: { accuracy: oos }, confidence, lag_seconds: 60 }) as unknown as LeadLagResult;

describe('lead-lag helpers', () => {
  it('always shows the sample count next to accuracy', () => {
    expect(accuracyText(0.6789, 412)).toBe('68% (412 events)');
    expect(accuracyText(1, 1)).toBe('100% (1 event)');
    expect(accuracyText(null, 0)).toBe('No events (0)');
    expect(accuracyText(0.5, 1250)).toBe('50% (1,250 events)');
  });

  it('formats lags', () => {
    expect(lagText(5)).toBe('5 s');
    expect(lagText(120)).toBe('2 min');
    expect(lagText(90)).toBe('1.5 min');
    expect(lagText(null)).toBe('No clear lag');
  });

  it('builds a Bot page link with lead-lag settings', () => {
    const href = botHref({ leader: 'NVDA', follower: 'AMD', params: { threshold_pct: 0.3, lookback_seconds: 60, max_wait_seconds: 300 } });
    expect(href).toBe('/bot?strategy=lead_lag&leader=NVDA&followers=AMD&threshold=0.3&lookback=60&max_wait=300');
  });

  it('sorts by confidence, then out-of-sample accuracy', () => {
    const rows = [row('A', 0.7, 0.6, 'moderate'), row('B', 0.8, 0.75, 'strong'), row('C', 0.9, 0.5, 'weak'), row('D', 0.72, 0.7, 'moderate')];
    expect(sortResults(rows, 'confidence', true).map((r) => r.leader)).toEqual(['B', 'D', 'A', 'C']);
    expect(sortResults(rows, 'accuracy', false).map((r) => r.leader)).toEqual(['A', 'D', 'B', 'C']);
  });
});
