import { describe, expect, it } from 'vitest';
import { barLabel } from '@/components/charts/crossover-chart';

describe('barLabel', () => {
  it('shows minute bars as New York clock time', () => {
    // 15:41 UTC in November is 10:41 AM in New York (EST, UTC-5).
    expect(barLabel('2026-11-13T15:41:00+00:00', 'minute')).toBe('10:41 AM');
  });

  it('shows daily bars as the New York date', () => {
    // Polygon stamps daily bars at midnight New York time, 05:00 UTC in November.
    expect(barLabel('2026-11-12T05:00:00+00:00', 'day')).toBe('Nov 12');
  });
});