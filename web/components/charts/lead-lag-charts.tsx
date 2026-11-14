'use client';

import { useRef } from 'react';
import { linear, niceTicks, pathFrom } from '@/lib/chart';
import { useElementWidth } from '@/hooks/use-element-width';

const M = { top: 12, right: 16, bottom: 32, left: 52 };
const HEIGHT = 220;

function Axes({
  width,
  yTicks,
  y,
  yFormat,
  xLabels,
}: {
  width: number;
  yTicks: number[];
  y: (v: number) => number;
  yFormat: (v: number) => string;
  xLabels: { x: number; text: string }[];
}) {
  return (
    <g className="text-caption" fill="currentColor">
      {yTicks.map((t) => (
        <g key={t}>
          <line x1={M.left} x2={width - M.right} y1={y(t)} y2={y(t)} className="stroke-hairline" strokeWidth={1} />
          <text x={M.left - 8} y={y(t)} dy="0.32em" textAnchor="end" className="fill-body">
            {yFormat(t)}
          </text>
        </g>
      ))}
      {xLabels.map((l) => (
        <text key={`${l.x}-${l.text}`} x={l.x} y={HEIGHT - 10} textAnchor="middle" className="fill-body">
          {l.text}
        </text>
      ))}
    </g>
  );
}

function secondsLabel(s: number) {
  return Math.abs(s) >= 60 ? `${s / 60}m` : `${s}s`;
}

/** Average leader and follower move around signal events (direction-adjusted, percent from the event). */
export function EventPathChart({ points }: { points: { offset_seconds: number; leader_pct: number; follower_pct: number }[] }) {
  const ref = useRef<HTMLDivElement>(null);
  const width = useElementWidth(ref);
  if (!points.length) return null;
  const xs = points.map((p) => p.offset_seconds);
  const values = points.flatMap((p) => [p.leader_pct, p.follower_pct]);
  const yTicks = niceTicks(Math.min(0, ...values), Math.max(0, ...values), 4);
  const x = linear([xs[0], xs[xs.length - 1]], [M.left, Math.max(M.left + 1, width - M.right)]);
  const y = linear([yTicks[0], yTicks[yTicks.length - 1]], [HEIGHT - M.bottom, M.top]);
  const step = Math.max(1, Math.ceil(points.length / 6));
  const labels = points.filter((_, i) => i % step === 0).map((p) => ({ x: x(p.offset_seconds), text: secondsLabel(p.offset_seconds) }));
  return (
    <figure ref={ref} className="flex flex-col gap-xxs">
      {width > 0 && (
        <svg width={width} height={HEIGHT} role="img" aria-label="Average leader and follower move around signal events">
          <Axes width={width} yTicks={yTicks} y={y} yFormat={(v) => `${v}%`} xLabels={labels} />
          <line x1={x(0)} x2={x(0)} y1={M.top} y2={HEIGHT - M.bottom} className="stroke-body" strokeDasharray="3 3" />
          <path d={pathFrom(points.map((p) => [x(p.offset_seconds), y(p.leader_pct)]))} fill="none" className="stroke-ink" strokeWidth={2} />
          <path d={pathFrom(points.map((p) => [x(p.offset_seconds), y(p.follower_pct)]))} fill="none" className="stroke-primary" strokeWidth={2} />
        </svg>
      )}
      <figcaption className="flex flex-wrap gap-sm text-caption">
        <span className="flex items-center gap-xxs">
          <span aria-hidden className="h-[2px] w-xs bg-ink" /> Leader
        </span>
        <span className="flex items-center gap-xxs">
          <span aria-hidden className="h-[2px] w-xs bg-primary" /> Follower
        </span>
        <span>Time 0 is the leader&apos;s signal. Moves are in the leader&apos;s direction.</span>
      </figcaption>
    </figure>
  );
}

/** Bars with a label under each, values 0..max. */
export function BarChart({
  bars,
  format,
  label,
  reference,
}: {
  bars: { key: string; label: string; value: number | null; note?: string }[];
  format: (v: number) => string;
  label: string;
  reference?: { value: number; text: string };
}) {
  const ref = useRef<HTMLDivElement>(null);
  const width = useElementWidth(ref);
  const values = bars.map((b) => b.value ?? 0);
  const yTicks = niceTicks(0, Math.max(reference?.value ?? 0, ...values, 1e-9), 4);
  const top = yTicks[yTicks.length - 1] || 1;
  const y = linear([0, top], [HEIGHT - M.bottom, M.top]);
  const inner = Math.max(0, width - M.left - M.right);
  const slot = bars.length ? inner / bars.length : 0;
  const step = Math.max(1, Math.ceil(bars.length / 8));
  return (
    <div ref={ref}>
      {width > 0 && bars.length > 0 && (
        <svg width={width} height={HEIGHT} role="img" aria-label={label}>
          <Axes
            width={width}
            yTicks={yTicks}
            y={y}
            yFormat={format}
            xLabels={bars.filter((_, i) => i % step === 0).map((b) => ({ x: M.left + bars.indexOf(b) * slot + slot / 2, text: b.label }))}
          />
          {bars.map((b, i) =>
            b.value == null ? null : (
              <rect
                key={b.key}
                x={M.left + i * slot + slot * 0.15}
                width={Math.max(1, slot * 0.7)}
                y={y(b.value)}
                height={Math.max(0, y(0) - y(b.value))}
                className="fill-primary"
              >
                <title>{`${b.label}: ${format(b.value)}${b.note ? ` (${b.note})` : ''}`}</title>
              </rect>
            ),
          )}
          {reference && (
            <g>
              <line x1={M.left} x2={width - M.right} y1={y(reference.value)} y2={y(reference.value)} className="stroke-info" strokeDasharray="4 4" />
              <text x={width - M.right} y={y(reference.value) - 4} textAnchor="end" className="fill-info text-caption">
                {reference.text}
              </text>
            </g>
          )}
        </svg>
      )}
    </div>
  );
}
