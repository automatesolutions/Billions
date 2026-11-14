'use client';

import { useMemo, useRef, useState } from 'react';
import type { BotChart } from '@/types/api';
import { linear, nearestIndex, niceTicks, pathFrom } from '@/lib/chart';
import { money } from '@/lib/format';
import { useElementWidth } from '@/hooks/use-element-width';

const MARGIN = { top: 16, right: 16, bottom: 32, left: 64 };
const ET = 'America/New_York';

/** A bar's time in New York: "10:41 AM" for minute bars, "Nov 12" for daily bars. */
export function barLabel(iso: string, timespan: BotChart['timespan']): string {
  const opts: Intl.DateTimeFormatOptions =
    timespan === 'minute' ? { hour: 'numeric', minute: '2-digit', timeZone: ET } : { month: 'short', day: 'numeric', timeZone: ET };
  return new Intl.DateTimeFormat('en-US', opts).format(new Date(iso));
}

/**
 * Price with the short and long averages. The band between the averages is green while the
 * short one is above and red while it's below, so every color change is a cross.
 */
export function CrossoverChart({ chart }: { chart: BotChart }) {
  const ref = useRef<HTMLDivElement>(null);
  const width = useElementWidth(ref);
  const [hover, setHover] = useState<number | null>(null);
  const height = width < 480 ? 260 : 320;
  const innerW = Math.max(0, width - MARGIN.left - MARGIN.right);
  const innerH = height - MARGIN.top - MARGIN.bottom;
  const { bars } = chart;

  const g = useMemo(() => {
    if (bars.length < 2 || innerW <= 0) return null;
    const values = bars.flatMap((b) => [b.close, b.short, b.long]).filter((v): v is number => v != null);
    const lo = Math.min(...values);
    const hi = Math.max(...values);
    const pad = (hi - lo) * 0.08 || 1;
    const sx = linear([0, bars.length - 1], [0, innerW]);
    const sy = linear([lo - pad, hi + pad], [innerH, 0]);
    const index = new Map(bars.map((b, i) => [b.t, i]));
    const line = (key: 'close' | 'short' | 'long') =>
      pathFrom(bars.flatMap((b, i) => (b[key] == null ? [] : [[sx(i), sy(b[key] as number)] as [number, number]])));
    const band = bars.slice(1).flatMap((b, k) => {
      const a = bars[k];
      if (a.short == null || a.long == null || b.short == null || b.long == null) return [];
      const above = a.short + b.short > a.long + b.long;
      const pts = [
        [sx(k), sy(a.short)],
        [sx(k + 1), sy(b.short)],
        [sx(k + 1), sy(b.long)],
        [sx(k), sy(a.long)],
      ];
      return [{ key: b.t, above, points: pts.map((p) => p.map((v) => v.toFixed(1)).join(',')).join(' ') }];
    });
    const step = Math.max(1, Math.ceil(bars.length / (innerW < 480 ? 4 : 7)));
    const xTicks = bars.map((b, i) => ({ i, t: b.t })).filter(({ i }) => i % step === 0);
    return {
      sx,
      sy,
      xs: bars.map((_, i) => sx(i)),
      index,
      band,
      close: line('close'),
      short: line('short'),
      long: line('long'),
      yTicks: niceTicks(lo - pad, hi + pad, 5),
      xTicks,
    };
  }, [bars, innerW, innerH]);

  const hovered = hover !== null ? bars[hover] : null;
  const label = `${chart.symbol} ${chart.timespan === 'minute' ? 'minute' : 'daily'} closes with the ${chart.short_window}-bar and ${chart.long_window}-bar averages, ${chart.crosses.length} crosses, and ${chart.orders.length} bot orders.`;

  return (
    <figure className="m-0">
      <div ref={ref} className="relative w-full" style={{ height }}>
        {g && width > 0 && (
          <svg
            width={width}
            height={height}
            role="img"
            aria-label={label}
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
                  <line x2={innerW} className="stroke-hairline" />
                  <text x={-8} dy="0.32em" textAnchor="end" className="fill-body text-caption tabular">
                    {money(t)}
                  </text>
                </g>
              ))}
              {g.xTicks.map(({ i, t }) => (
                <text key={t} x={g.sx(i)} y={innerH + 20} textAnchor="middle" className="fill-body text-caption">
                  {barLabel(t, chart.timespan)}
                </text>
              ))}

              {g.band.map((q) => (
                <polygon key={q.key} points={q.points} className={q.above ? 'fill-up' : 'fill-down'} fillOpacity={0.18} />
              ))}

              <path d={g.close} className="fill-none stroke-muted" strokeWidth={1.5} strokeLinejoin="round" />
              <path d={g.long} className="fill-none stroke-info" strokeWidth={2} strokeLinejoin="round" />
              <path d={g.short} className="fill-none stroke-primary" strokeWidth={2} strokeLinejoin="round" />

              {chart.crosses.map((c) => {
                const i = g.index.get(c.t);
                const bar = i === undefined ? null : bars[i];
                if (i === undefined || bar?.short == null) return null;
                return (
                  <g key={c.t}>
                    <line x1={g.sx(i)} x2={g.sx(i)} y2={innerH} className="stroke-body" strokeDasharray="3 4" strokeOpacity={0.6} />
                    <circle cx={g.sx(i)} cy={g.sy(bar.short)} r={5} className={c.direction === 'up' ? 'fill-up stroke-canvas' : 'fill-down stroke-canvas'} strokeWidth={2}>
                      <title>{`${c.direction === 'up' ? 'Crossed above' : 'Crossed below'} at ${barLabel(c.t, chart.timespan)}`}</title>
                    </circle>
                  </g>
                );
              })}

              {chart.orders.map((o) => {
                const i = o.t ? g.index.get(o.t) : undefined;
                const price = o.price ?? (i !== undefined ? bars[i].close : null);
                if (i === undefined || price == null) return null;
                const x = g.sx(i);
                const y = g.sy(price);
                const buy = o.side === 'buy';
                const tip = buy ? `${x},${y + 8} ${x - 7},${y + 20} ${x + 7},${y + 20}` : `${x},${y - 8} ${x - 7},${y - 20} ${x + 7},${y - 20}`;
                return (
                  <g key={`${o.at}-${o.side}`}>
                    <polygon points={tip} className={buy ? 'fill-up' : 'fill-down'}>
                      <title>{`${buy ? 'Bought' : 'Sold'} ${o.qty ?? ''} at ${money(price)}`}</title>
                    </polygon>
                    <text x={x} y={buy ? y + 34 : y - 26} textAnchor="middle" className={buy ? 'fill-up text-caption' : 'fill-down text-caption'}>
                      {buy ? 'Buy' : 'Sell'} {o.qty ?? ''}
                    </text>
                  </g>
                );
              })}

              {hovered && hover !== null && <line x1={g.sx(hover)} x2={g.sx(hover)} y2={innerH} className="stroke-muted" />}
            </g>
          </svg>
        )}
        {hovered && hover !== null && g && (
          <div
            className="pointer-events-none absolute top-0 border border-hairline bg-canvas px-xs py-xxs text-body-sm"
            style={{ left: Math.max(0, Math.min(MARGIN.left + g.sx(hover) + 12, width - 190)) }}
          >
            <p className="text-body">{barLabel(hovered.t, chart.timespan)} ET</p>
            <p className="tabular text-ink">Price {hovered.close == null ? '—' : money(hovered.close)}</p>
            <p className="tabular text-primary">Short {hovered.short == null ? '—' : money(hovered.short)}</p>
            <p className="tabular text-info">Long {hovered.long == null ? '—' : money(hovered.long)}</p>
          </div>
        )}
      </div>
      <figcaption className="mt-xs flex flex-wrap gap-x-sm gap-y-xxs text-caption text-body">
        <span className="inline-flex items-center gap-xxs">
          <span aria-hidden className="inline-block h-px w-xs bg-muted" /> Price
        </span>
        <span className="inline-flex items-center gap-xxs">
          <span aria-hidden className="inline-block h-[2px] w-xs bg-primary" /> Short average ({chart.short_window} bars)
        </span>
        <span className="inline-flex items-center gap-xxs">
          <span aria-hidden className="inline-block h-[2px] w-xs bg-info" /> Long average ({chart.long_window} bars)
        </span>
        <span className="inline-flex items-center gap-xxs">
          <span aria-hidden className="inline-block size-xxs rounded-full bg-up" /> Crossed above
        </span>
        <span className="inline-flex items-center gap-xxs">
          <span aria-hidden className="inline-block size-xxs rounded-full bg-down" /> Crossed below
        </span>
        <span>Triangles are the bot&apos;s orders.</span>
      </figcaption>
    </figure>
  );
}
