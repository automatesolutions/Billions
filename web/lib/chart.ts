/** Small helpers for hand-written SVG charts. */

export function linear(domain: [number, number], range: [number, number]) {
  const [d0, d1] = domain;
  const [r0, r1] = range;
  const span = d1 - d0 || 1;
  return (v: number) => r0 + ((v - d0) / span) * (r1 - r0);
}

/** About `count` round tick values covering [min, max] (1, 2, 2.5 or 5 times a power of ten). */
export function niceTicks(min: number, max: number, count = 5): number[] {
  if (!Number.isFinite(min) || !Number.isFinite(max)) return [];
  if (min === max) return [min];
  const raw = (max - min) / Math.max(count, 1);
  const power = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * power).find((s) => s >= raw) ?? raw;
  const decimals = Math.max(0, -Math.floor(Math.log10(step)) + 1);
  const first = Math.ceil(min / step - 1e-9);
  const last = Math.floor(max / step + 1e-9);
  const ticks: number[] = [];
  for (let i = first; i <= last; i++) ticks.push(Number((i * step).toFixed(decimals)) + 0); // + 0 turns -0 into 0
  return ticks;
}

/** Index of the item nearest to `x` in a sorted array of positions. */
export function nearestIndex(positions: number[], x: number): number {
  let lo = 0;
  let hi = positions.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (positions[mid] < x) lo = mid;
    else hi = mid;
  }
  return Math.abs(positions[lo] - x) <= Math.abs(positions[hi] - x) ? lo : hi;
}

export function pathFrom(points: [number, number][]): string {
  return points.map(([x, y], i) => `${i ? 'L' : 'M'}${x.toFixed(1)},${y.toFixed(1)}`).join('');
}
