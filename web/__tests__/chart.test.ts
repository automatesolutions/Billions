import { describe, expect, it } from 'vitest';
import { linear, nearestIndex, niceTicks, pathFrom } from '@/lib/chart';

describe('chart helpers', () => {
  it('maps a domain onto a range', () => {
    const s = linear([0, 10], [0, 100]);
    expect(s(5)).toBe(50);
    expect(linear([0, 10], [100, 0])(10)).toBe(0);
  });

  it('makes round ticks', () => {
    expect(niceTicks(0, 100, 5)).toEqual([0, 20, 40, 60, 80, 100]);
    expect(niceTicks(0.93, 1.27, 4)).toEqual([1, 1.1, 1.2]);
    expect(niceTicks(-0.34, 0, 4)).toEqual([-0.3, -0.2, -0.1, 0]);
  });

  it('finds the nearest index', () => {
    expect(nearestIndex([0, 10, 20, 30], 14)).toBe(1);
    expect(nearestIndex([0, 10, 20, 30], 26)).toBe(3);
  });

  it('builds an SVG path', () => {
    expect(pathFrom([[0, 0], [1.25, 2]])).toBe('M0.0,0.0L1.3,2.0');
  });
});
