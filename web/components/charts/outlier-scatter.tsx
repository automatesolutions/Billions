'use client';

import { useMemo, useRef, useState } from 'react';
import { useRouter } from 'next/navigation';
import type { RankedOutlier, ScatterPoint } from '@/types/api';
import { pct, plain } from '@/lib/format';
import { useElementWidth } from '@/hooks/use-element-width';

/**
 * Scatter of every stock in the strategy: long-window return (x) vs short-window return (y).
 * Axes use asinh compression so a +600% move and a +3% move both stay readable;
 * tick labels show the real percentages.
 */

const SOFTNESS = 10; // % where the scale turns from linear to logarithmic
const squash = (v: number) => Math.asinh(v / SOFTNESS);
const CANDIDATE_TICKS = [-90, -75, -50, -25, -10, 0, 10, 25, 50, 100, 200, 500, 1000, 2000];
const MARGIN = { top: 16, right: 16, bottom: 48, left: 64 };
const LABEL_COUNT = 5;
const HIT_RADIUS = 24;

function meanStd(values: number[]) {
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  const std = Math.sqrt(values.reduce((a, b) => a + (b - mean) ** 2, 0) / values.length);
  return { mean, std };
}

/** Ticks inside [min, max] whose screen positions are at least `gap` px apart. Zero always stays. */
function spacedTicks(min: number, max: number, scale: (v: number) => number, gap: number) {
  const inRange = CANDIDATE_TICKS.filter((t) => t >= min && t <= max);
  const kept: number[] = inRange.includes(0) ? [0] : [];
  // Walk outward from zero so the ticks nearest the center win.
  const order = [...inRange.filter((t) => t > 0), ...inRange.filter((t) => t < 0).reverse()];
  for (const t of order) {
    if (kept.every((k) => Math.abs(scale(k) - scale(t)) >= gap)) kept.push(t);
  }
  return kept.sort((a, b) => a - b);
}

const LABEL_CHAR_PX = 7;
const LABEL_HEIGHT_PX = 14;

/** Greedy label placement: higher-ranked labels first, skip any that would overlap. */
function placeLabels(items: { symbol: string; x: number; y: number }[], innerW: number) {
  const placed: { symbol: string; x: number; y: number; right: boolean; box: [number, number, number, number] }[] = [];
  for (const item of items) {
    const width = item.symbol.length * LABEL_CHAR_PX;
    const right = item.x + 8 + width < innerW;
    const x0 = right ? item.x + 8 : item.x - 8 - width;
    const box: [number, number, number, number] = [x0, item.y - LABEL_HEIGHT_PX / 2, x0 + width, item.y + LABEL_HEIGHT_PX / 2];
    const overlaps = placed.some((p) => box[0] < p.box[2] && box[2] > p.box[0] && box[1] < p.box[3] && box[3] > p.box[1]);
    if (!overlaps) placed.push({ ...item, right, box });
  }
  return placed;
}

export function OutlierScatter({
  points,
  outliers,
  xLabel,
  yLabel,
  zThreshold,
}: {
  points: ScatterPoint[];
  outliers: RankedOutlier[];
  xLabel: string;
  yLabel: string;
  zThreshold: number;
}) {
  const router = useRouter();
  const containerRef = useRef<HTMLDivElement>(null);
  const width = useElementWidth(containerRef);
  const [hover, setHover] = useState<ScatterPoint | null>(null);

  const height = Math.round(Math.min(560, Math.max(320, width * 0.62)));
  const innerW = Math.max(0, width - MARGIN.left - MARGIN.right);
  const innerH = height - MARGIN.top - MARGIN.bottom;

  const geometry = useMemo(() => {
    if (!points.length) return null;
    const xs = points.map((p) => p.x);
    const ys = points.map((p) => p.y);
    const pad = (lo: number, hi: number) => [squash(lo) - 0.15, squash(hi) + 0.15] as const;
    const [x0, x1] = pad(Math.min(...xs, 0), Math.max(...xs, 0));
    const [y0, y1] = pad(Math.min(...ys, 0), Math.max(...ys, 0));
    const sx = (v: number) => ((squash(v) - x0) / (x1 - x0)) * innerW;
    const sy = (v: number) => innerH - ((squash(v) - y0) / (y1 - y0)) * innerH;
    const mx = meanStd(xs);
    const my = meanStd(ys);
    const band = {
      x: sx(mx.mean - zThreshold * mx.std),
      y: sy(my.mean + zThreshold * my.std),
      w: sx(mx.mean + zThreshold * mx.std) - sx(mx.mean - zThreshold * mx.std),
      h: sy(my.mean - zThreshold * my.std) - sy(my.mean + zThreshold * my.std),
    };
    return {
      sx,
      sy,
      band,
      xTicks: spacedTicks(Math.min(...xs), Math.max(...xs), sx, 48),
      yTicks: spacedTicks(Math.min(...ys), Math.max(...ys), sy, 24),
    };
  }, [points, innerW, innerH, zThreshold]);


  if (!geometry || width === 0) {
    return (
      <figure className="m-0">
        <div ref={containerRef} className="relative w-full" style={{ height }} />
      </figure>
    );
  }
  const { sx, sy, band, xTicks, yTicks } = geometry;
  const background = points.filter((p) => !p.is_outlier);
  const highlighted = points.filter((p) => p.is_outlier);

  const nearest = (clientX: number, clientY: number, svg: SVGSVGElement) => {
    const rect = svg.getBoundingClientRect();
    const px = clientX - rect.left - MARGIN.left;
    const py = clientY - rect.top - MARGIN.top;
    let best: ScatterPoint | null = null;
    let bestD = HIT_RADIUS ** 2;
    for (const p of points) {
      const d = (sx(p.x) - px) ** 2 + (sy(p.y) - py) ** 2;
      if (d < bestD || (d === bestD && p.is_outlier)) {
        best = p;
        bestD = d;
      }
    }
    return best;
  };

  const tipLeft = hover ? Math.min(MARGIN.left + sx(hover.x) + 12, width - 180) : 0;
  const tipTop = hover ? Math.max(MARGIN.top + sy(hover.y) - 72, 0) : 0;

  return (
    <figure className="m-0">
      <div ref={containerRef} className="relative w-full">
        <svg
          width={width}
          height={height}
          role="img"
          aria-label={`Scatter of ${points.length} stocks: ${xLabel} return against ${yLabel} return. ${highlighted.length} outliers are highlighted. The table below lists them.`}
          className="block touch-manipulation"
          onPointerMove={(e) => setHover(nearest(e.clientX, e.clientY, e.currentTarget))}
          onPointerLeave={() => setHover(null)}
          onClick={(e) => {
            const p = nearest(e.clientX, e.clientY, e.currentTarget);
            if (p) router.push(`/analysis/${p.symbol}`);
          }}
          style={{ cursor: hover ? 'pointer' : 'default' }}
        >
          <g transform={`translate(${MARGIN.left},${MARGIN.top})`}>
            {/* grid */}
            {xTicks.map((t) => (
              <g key={`x${t}`} transform={`translate(${sx(t)},0)`}>
                <line y2={innerH} className={t === 0 ? 'stroke-muted' : 'stroke-hairline'} strokeWidth={1} />
                <text y={innerH + 20} textAnchor="middle" className="fill-body text-caption tabular">
                  {t === 0 ? '0%' : pct(t, 0)}
                </text>
              </g>
            ))}
            {yTicks.map((t) => (
              <g key={`y${t}`} transform={`translate(0,${sy(t)})`}>
                <line x2={innerW} className={t === 0 ? 'stroke-muted' : 'stroke-hairline'} strokeWidth={1} />
                <text x={-8} dy="0.32em" textAnchor="end" className="fill-body text-caption tabular">
                  {t === 0 ? '0%' : pct(t, 0)}
                </text>
              </g>
            ))}

            {/* the normal zone: within ±threshold σ on both axes */}
            <rect x={band.x} y={band.y} width={band.w} height={band.h} className="fill-elevated" opacity={0.35} />

            {background.map((p) => (
              <circle key={p.symbol} cx={sx(p.x)} cy={sy(p.y)} r={2.5} className="fill-muted" />
            ))}
            {highlighted.map((p) => (
              <circle key={p.symbol} cx={sx(p.x)} cy={sy(p.y)} r={4.5} className="fill-primary stroke-canvas" strokeWidth={2} />
            ))}
            {placeLabels(
              outliers.slice(0, LABEL_COUNT).map((o) => ({ symbol: o.symbol, x: sx(o.x), y: sy(o.y) })),
              innerW,
            ).map((l) => (
              <text
                key={`l${l.symbol}`}
                x={l.x + (l.right ? 8 : -8)}
                y={l.y}
                dy="0.32em"
                textAnchor={l.right ? 'start' : 'end'}
                className="fill-ink text-caption"
                style={{ paintOrder: 'stroke', stroke: 'var(--color-canvas)', strokeWidth: 3 }}
              >
                {l.symbol}
              </text>
            ))}
            {hover && (
              <circle cx={sx(hover.x)} cy={sy(hover.y)} r={7} className="fill-transparent stroke-ink" strokeWidth={1.5} />
            )}

            <text x={innerW / 2} y={innerH + 40} textAnchor="middle" className="fill-body text-caption">
              {xLabel} return
            </text>
            <text transform={`translate(${-54},${innerH / 2}) rotate(-90)`} textAnchor="middle" className="fill-body text-caption">
              {yLabel} return
            </text>
          </g>
        </svg>

        {hover && (
          <div
            className="pointer-events-none absolute border border-hairline bg-canvas px-xs py-xxs text-body-sm"
            style={{ left: tipLeft, top: tipTop, minWidth: 168 }}
          >
            <p className="text-title-sm text-ink">{hover.symbol}</p>
            <p className="tabular text-body">
              {xLabel}: <span className="text-ink">{pct(hover.x)}</span> ({plain(hover.z_x, 1)}σ)
            </p>
            <p className="tabular text-body">
              {yLabel}: <span className="text-ink">{pct(hover.y)}</span> ({plain(hover.z_y, 1)}σ)
            </p>
          </div>
        )}
      </div>

      <figcaption className="mt-xs flex flex-wrap items-center gap-x-sm gap-y-xxs text-caption text-body">
        <span className="inline-flex items-center gap-xxs">
          <svg width="10" height="10" aria-hidden>
            <circle cx="5" cy="5" r="4.5" className="fill-primary" />
          </svg>
          Outlier (|z| &gt; {zThreshold})
        </span>
        <span className="inline-flex items-center gap-xxs">
          <svg width="10" height="10" aria-hidden>
            <circle cx="5" cy="5" r="2.5" className="fill-muted" />
          </svg>
          Other stocks
        </span>
        <span className="inline-flex items-center gap-xxs">
          <span aria-hidden className="inline-block size-xs bg-elevated opacity-35" />
          Within ±{zThreshold}σ on both axes
        </span>
        <span>Axes are compressed so large moves fit. Tap a dot to open its analysis.</span>
      </figcaption>
    </figure>
  );
}
