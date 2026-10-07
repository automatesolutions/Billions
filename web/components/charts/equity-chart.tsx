'use client';

import { useMemo, useRef, useState } from 'react';
import { linear, nearestIndex, niceTicks, pathFrom } from '@/lib/chart';
import { dateET, pct } from '@/lib/format';
import { useElementWidth } from '@/hooks/use-element-width';

const MARGIN = { top: 12, right: 16, bottom: 28, left: 56 };
const DD_HEIGHT = 96;

/**
 * Out-of-sample growth of $1: following the stacked signal vs holding the stock,
 * with the signal's drawdown in a separate panel below (one y-axis per panel).
 */
export function EquityChart({ dates, equity, hold, drawdown }: { dates: string[]; equity: number[]; hold: number[]; drawdown: number[] }) {
  const ref = useRef<HTMLDivElement>(null);
  const width = useElementWidth(ref);
  const [hover, setHover] = useState<number | null>(null);
  const mainH = width < 480 ? 200 : 260;
  const innerW = Math.max(0, width - MARGIN.left - MARGIN.right);
  const innerH = mainH - MARGIN.top - MARGIN.bottom;
  const ddInner = DD_HEIGHT - MARGIN.top - MARGIN.bottom + 12;

  const g = useMemo(() => {
    if (!equity.length || innerW <= 0) return null;
    const lo = Math.min(...equity, ...hold, 1);
    const hi = Math.max(...equity, ...hold, 1);
    const pad = (hi - lo) * 0.08 || 0.05;
    const sx = linear([0, equity.length - 1], [0, innerW]);
    const sy = linear([lo - pad, hi + pad], [innerH, 0]);
    const ddLo = Math.min(...drawdown, -0.01);
    const sdd = linear([ddLo, 0], [ddInner, 0]);
    return {
      sx,
      sy,
      sdd,
      xs: equity.map((_, i) => sx(i)),
      yTicks: niceTicks(lo - pad, hi + pad, 4),
      ddTicks: niceTicks(ddLo, 0, 2),
      ddLo,
    };
  }, [equity, hold, drawdown, innerW, innerH, ddInner]);

  const onMove = (e: React.PointerEvent<SVGSVGElement>) => {
    if (!g) return;
    const x = e.clientX - e.currentTarget.getBoundingClientRect().left - MARGIN.left;
    setHover(x >= 0 && x <= innerW ? nearestIndex(g.xs, x) : null);
  };

  return (
    <figure className="m-0">
      <div ref={ref} className="relative w-full" style={{ height: mainH + DD_HEIGHT }}>
        {g && width > 0 && (
          <>
            <svg
              width={width}
              height={mainH}
              role="img"
              aria-label={`Growth of one dollar over ${equity.length} test days: following the signal ended at ${equity.at(-1)?.toFixed(2)}, holding the stock ended at ${hold.at(-1)?.toFixed(2)}.`}
              onPointerMove={onMove}
              onPointerLeave={() => setHover(null)}
              className="block touch-pan-y"
            >
              <g transform={`translate(${MARGIN.left},${MARGIN.top})`}>
                {g.yTicks.map((t) => (
                  <g key={t} transform={`translate(0,${g.sy(t)})`}>
                    <line x2={innerW} className={t === 1 ? 'stroke-muted' : 'stroke-hairline'} />
                    <text x={-8} dy="0.32em" textAnchor="end" className="fill-body text-caption tabular">
                      ${t.toFixed(2)}
                    </text>
                  </g>
                ))}
                <path d={pathFrom(hold.map((v, i) => [g.sx(i), g.sy(v)]))} className="fill-none stroke-body" strokeWidth={2} strokeLinejoin="round" />
                <path d={pathFrom(equity.map((v, i) => [g.sx(i), g.sy(v)]))} className="fill-none stroke-ink" strokeWidth={2} strokeLinejoin="round" />
                <text x={0} y={innerH + 20} className="fill-body text-caption">
                  {dateET(dates[0])}
                </text>
                <text x={innerW} y={innerH + 20} textAnchor="end" className="fill-body text-caption">
                  {dateET(dates[dates.length - 1])}
                </text>
                {hover !== null && (
                  <>
                    <line x1={g.sx(hover)} x2={g.sx(hover)} y2={innerH} className="stroke-muted" />
                    <circle cx={g.sx(hover)} cy={g.sy(equity[hover])} r={4} className="fill-ink stroke-canvas" strokeWidth={2} />
                    <circle cx={g.sx(hover)} cy={g.sy(hold[hover])} r={4} className="fill-body stroke-canvas" strokeWidth={2} />
                  </>
                )}
              </g>
            </svg>

            <svg width={width} height={DD_HEIGHT} role="img" aria-label={`Drawdown of the signal. Deepest: ${pct(g.ddLo * 100)}.`} className="block">
              <g transform={`translate(${MARGIN.left},${MARGIN.top})`}>
                {g.ddTicks.map((t) => (
                  <g key={t} transform={`translate(0,${g.sdd(t)})`}>
                    <line x2={innerW} className="stroke-hairline" />
                    <text x={-8} dy="0.32em" textAnchor="end" className="fill-body text-caption tabular">
                      {t === 0 ? '0%' : pct(t * 100, 0)}
                    </text>
                  </g>
                ))}
                <path
                  d={`${pathFrom(drawdown.map((v, i) => [g.sx(i), g.sdd(v)]))}L${g.sx(drawdown.length - 1)},0L0,0Z`}
                  className="fill-down"
                  opacity={0.12}
                />
                <path d={pathFrom(drawdown.map((v, i) => [g.sx(i), g.sdd(v)]))} className="fill-none stroke-down" strokeWidth={2} />
                {hover !== null && <line x1={g.sx(hover)} x2={g.sx(hover)} y2={ddInner} className="stroke-muted" />}
              </g>
            </svg>
          </>
        )}

        {hover !== null && g && (
          <div
            className="pointer-events-none absolute top-0 border border-hairline bg-canvas px-xs py-xxs text-body-sm"
            style={{ left: Math.min(MARGIN.left + g.sx(hover) + 12, width - 200) }}
          >
            <p className="text-body">{dateET(dates[hover])}</p>
            <p className="tabular text-ink">Signal ${equity[hover].toFixed(3)}</p>
            <p className="tabular text-body">Holding ${hold[hover].toFixed(3)}</p>
            <p className="tabular text-body">Drawdown {pct(drawdown[hover] * 100)}</p>
          </div>
        )}
      </div>
      <figcaption className="mt-xs flex flex-wrap gap-x-sm gap-y-xxs text-caption text-body">
        <span className="inline-flex items-center gap-xxs">
          <span aria-hidden className="inline-block h-px w-xs bg-ink" /> Following the stacked signal
        </span>
        <span className="inline-flex items-center gap-xxs">
          <span aria-hidden className="inline-block h-px w-xs bg-body" /> Holding the stock
        </span>
        <span className="inline-flex items-center gap-xxs">
          <span aria-hidden className="inline-block h-px w-xs bg-down" /> Signal drawdown (lower panel)
        </span>
        <span>Test period only, before costs.</span>
      </figcaption>
    </figure>
  );
}
