'use client';

import { useMemo, useRef, useState } from 'react';
import type { PricePoint } from '@/types/api';
import { linear, nearestIndex, niceTicks, pathFrom } from '@/lib/chart';
import { dateET, money } from '@/lib/format';
import { useElementWidth } from '@/hooks/use-element-width';

const MARGIN = { top: 16, right: 16, bottom: 32, left: 64 };
const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

export interface ForecastMark {
  date: string;
  expected: number;
  low: number;
  high: number;
}

/** Daily closes (last year) plus an optional next-day forecast with its range. One axis: price in USD. */
export function PriceChart({ history, forecast, ticker }: { history: PricePoint[]; forecast?: ForecastMark | null; ticker: string }) {
  const ref = useRef<HTMLDivElement>(null);
  const width = useElementWidth(ref);
  const [hover, setHover] = useState<number | null>(null);
  const height = width < 480 ? 240 : 320;
  const innerW = Math.max(0, width - MARGIN.left - MARGIN.right - (forecast ? 32 : 0));
  const innerH = height - MARGIN.top - MARGIN.bottom;

  const g = useMemo(() => {
    if (!history.length || innerW <= 0) return null;
    const values = history.map((p) => p.close);
    const lo = Math.min(...values, forecast?.low ?? Infinity);
    const hi = Math.max(...values, forecast?.high ?? -Infinity);
    const pad = (hi - lo) * 0.06 || 1;
    const sx = linear([0, history.length - 1], [0, innerW]);
    const sy = linear([lo - pad, hi + pad], [innerH, 0]);
    const xs = history.map((_, i) => sx(i));
    const monthTicks = history
      .map((p, i) => ({ i, month: Number(p.date.slice(5, 7)), prev: i ? Number(history[i - 1].date.slice(5, 7)) : -1 }))
      .filter((m) => m.month !== m.prev && m.i > 0)
      .filter((_, k, all) => (innerW < 480 ? k % 3 === 0 : all.length <= 13 || k % 2 === 0));
    return { sx, sy, xs, yTicks: niceTicks(lo - pad, hi + pad, 5), monthTicks };
  }, [history, forecast, innerW, innerH]);

  const hovered = hover !== null ? history[hover] : null;

  return (
    <figure className="m-0">
      <div ref={ref} className="relative w-full" style={{ height }}>
        {g && width > 0 && (
          <svg
            width={width}
            height={height}
            role="img"
            aria-label={`${ticker} daily closing price over the last ${history.length} trading days${forecast ? `, with a forecast range for ${dateET(forecast.date)}` : ''}.`}
            onPointerMove={(e) => {
              const x = e.clientX - e.currentTarget.getBoundingClientRect().left - MARGIN.left;
              setHover(x >= 0 && x <= innerW ? nearestIndex(g.xs, x) : null);
            }}
            onPointerLeave={() => setHover(null)}
            className="block touch-pan-y"
          >
            <g transform={`translate(${MARGIN.left},${MARGIN.top})`}>
              {g.yTicks.map((t) => (
                <g key={t} transform={`translate(0,${g.sy(t)})`}>
                  <line x2={innerW + (forecast ? 32 : 0)} className="stroke-hairline" />
                  <text x={-8} dy="0.32em" textAnchor="end" className="fill-body text-caption tabular">
                    {money(t)}
                  </text>
                </g>
              ))}
              {g.monthTicks.map((m) => (
                <text key={m.i} x={g.sx(m.i)} y={innerH + 20} textAnchor="middle" className="fill-body text-caption">
                  {MONTHS[m.month - 1]}
                </text>
              ))}

              <path
                d={pathFrom(history.map((p, i) => [g.sx(i), g.sy(p.close)]))}
                className="fill-none stroke-ink"
                strokeWidth={2}
                strokeLinejoin="round"
                strokeLinecap="round"
              />

              {forecast && (
                <g transform={`translate(${innerW + 24},0)`}>
                  <line y1={g.sy(forecast.high)} y2={g.sy(forecast.low)} className="stroke-body" strokeWidth={2} />
                  <line x1={-5} x2={5} y1={g.sy(forecast.high)} y2={g.sy(forecast.high)} className="stroke-body" strokeWidth={2} />
                  <line x1={-5} x2={5} y1={g.sy(forecast.low)} y2={g.sy(forecast.low)} className="stroke-body" strokeWidth={2} />
                  <circle cy={g.sy(forecast.expected)} r={5} className="fill-primary stroke-canvas" strokeWidth={2} />
                </g>
              )}

              {hovered && hover !== null && (
                <g>
                  <line x1={g.sx(hover)} x2={g.sx(hover)} y2={innerH} className="stroke-muted" />
                  <circle cx={g.sx(hover)} cy={g.sy(hovered.close)} r={4} className="fill-ink stroke-canvas" strokeWidth={2} />
                </g>
              )}
            </g>
          </svg>
        )}
        {hovered && hover !== null && g && (
          <div
            className="pointer-events-none absolute top-0 border border-hairline bg-canvas px-xs py-xxs text-body-sm"
            style={{ left: Math.min(MARGIN.left + g.sx(hover) + 12, width - 150) }}
          >
            <p className="text-body">{dateET(hovered.date)}</p>
            <p className="tabular text-title-sm text-ink">{money(hovered.close)}</p>
          </div>
        )}
      </div>
      {forecast && (
        <figcaption className="mt-xs flex flex-wrap gap-x-sm gap-y-xxs text-caption text-body">
          <span className="inline-flex items-center gap-xxs">
            <span aria-hidden className="inline-block h-px w-xs bg-ink" /> Daily close
          </span>
          <span className="inline-flex items-center gap-xxs">
            <svg width="10" height="10" aria-hidden>
              <circle cx="5" cy="5" r="4" className="fill-primary" />
            </svg>
            Forecast for {dateET(forecast.date)}: {money(forecast.expected)} (range {money(forecast.low)} to {money(forecast.high)})
          </span>
        </figcaption>
      )}
    </figure>
  );
}
