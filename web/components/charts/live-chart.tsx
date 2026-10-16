'use client';

import { useMemo, useRef } from 'react';
import { linear, pathFrom } from '@/lib/chart';
import { useElementWidth } from '@/hooks/use-element-width';

const HEIGHT = 120;
const PAD = 8;

/** Growth of $1 from the live action record, with the break-even line at 1. */
export function LiveChart({ equity, label }: { equity: number[]; label: string }) {
  const ref = useRef<HTMLDivElement>(null);
  const width = useElementWidth(ref);

  const g = useMemo(() => {
    if (equity.length < 2 || width <= 0) return null;
    const lo = Math.min(...equity, 1);
    const hi = Math.max(...equity, 1);
    const pad = (hi - lo) * 0.1 || 0.001;
    const sx = linear([0, equity.length - 1], [PAD, width - PAD]);
    const sy = linear([lo - pad, hi + pad], [HEIGHT - PAD, PAD]);
    return { line: pathFrom(equity.map((v, i) => [sx(i), sy(v)])), one: sy(1), up: (equity.at(-1) ?? 1) >= 1 };
  }, [equity, width]);

  return (
    <div ref={ref} className="w-full" style={{ height: HEIGHT }}>
      {g && (
        <svg width={width} height={HEIGHT} role="img" aria-label={label} className="block">
          <line x1={PAD} x2={width - PAD} y1={g.one} y2={g.one} className="stroke-hairline" strokeDasharray="4 4" />
          <path d={g.line} className={g.up ? 'fill-none stroke-up' : 'fill-none stroke-down'} strokeWidth={2} strokeLinejoin="round" />
        </svg>
      )}
    </div>
  );
}
