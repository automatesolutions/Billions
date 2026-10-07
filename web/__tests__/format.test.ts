import { describe, expect, it } from 'vitest';
import { dateET, pct, plain, signed } from '@/lib/format';

describe('format', () => {
  it('signs numbers with a real minus', () => {
    expect(pct(12.345)).toBe('+12.3%');
    expect(pct(-3.1)).toBe('\u22123.1%');
    expect(signed(0, 1)).toBe('0.0');
    expect(plain(-1.5, 1)).toBe('\u22121.5');
  });

  it('formats plain dates without shifting the day', () => {
    expect(dateET('2026-10-06')).toBe('Oct 6, 2026');
  });
});
