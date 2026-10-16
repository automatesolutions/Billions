'use client';

import { useEffect, useState } from 'react';
import type { QuantOk, QuantSignalRow } from '@/types/api';
import { barTime } from '@/lib/format';
import { INTERVAL_COPY } from '@/lib/intervals';
import { cn } from '@/lib/utils';

const TAPE_LENGTH = 30;

function useNow() {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(id);
  }, []);
  return now;
}

function clock(ms: number) {
  const s = Math.max(0, Math.round(ms / 1000));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = String(s % 60).padStart(2, '0');
  return h ? `${h}:${String(m).padStart(2, '0')}:${sec}` : `${m}:${sec}`;
}

function squareClass(row: QuantSignalRow) {
  if (row.status === 'open') return 'tape-wait border border-body bg-transparent';
  if (row.status === 'void' || row.correct === null) return 'bg-muted';
  return row.correct ? 'bg-up' : 'bg-down';
}

function squareLabel(row: QuantSignalRow) {
  const at = barTime(row.bar_time);
  if (row.status === 'open') return `${at}: waiting for the next bar`;
  if (row.status === 'void') return `${at}: no data`;
  if (row.correct === null) return `${at}: flat bar`;
  return `${at}: direction ${row.correct ? 'right' : 'wrong'}`;
}

/**
 * Live recording strip shown while a pair is tracked: a REC light, a bar clock that fills
 * until the next bar closes, and a tape with one square per recorded signal.
 * `engaged` plays a one-time sweep when tracking was just switched on.
 */
export function TrackingTape({ data, engaged }: { data: QuantOk; engaged: boolean }) {
  const now = useNow();
  const start = new Date(data.last_update_at).getTime();
  const due = new Date(data.next_update_at).getTime();
  const progress = Math.min(1, Math.max(0, (now - start) / Math.max(1, due - start)));
  const paused = !data.market.is_open && data.interval !== '1d';
  const scoring = now >= due;
  const bar = INTERVAL_COPY[data.interval].bar;

  const tape = data.live.recent.slice(0, TAPE_LENGTH).reverse(); // oldest first
  const right = tape.filter((r) => r.correct === true).length;
  const wrong = tape.filter((r) => r.correct === false).length;
  const waiting = tape.filter((r) => r.status === 'open').length;

  return (
    <section
      aria-labelledby="tape-title"
      className="relative flex flex-col gap-xs overflow-hidden border border-hairline bg-black px-xs py-sm sm:px-sm"
    >
      {engaged && (
        <>
          <span aria-hidden className="engage absolute inset-x-0 top-0 h-[2px] origin-left bg-primary" />
          <span aria-hidden className="engage pointer-events-none absolute inset-0 origin-left bg-primary/10" />
        </>
      )}

      <div className="flex flex-wrap items-center justify-between gap-xs">
        <h3 id="tape-title" className="flex items-center gap-xxs text-caption-upper uppercase text-ink">
          <span aria-hidden className={cn('size-xxs rounded-full', paused ? 'bg-body' : 'rec-dot bg-primary')} />
          {paused ? 'Paused' : 'Rec'}
          <span className="text-body normal-case">
            · Tracking {data.ticker} every {bar}
          </span>
        </h3>
        <p className="tabular text-title-sm text-ink" role="timer" aria-live="off">
          {paused ? 'Market closed' : scoring ? 'Scoring…' : `Bar closes in ${clock(due - now)}`}
        </p>
      </div>

      <div
        role="progressbar"
        aria-label={`Time until the next ${bar} closes`}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Math.round(progress * 100)}
        className="relative h-xxs w-full overflow-hidden bg-elevated"
      >
        <span
          className={cn('absolute inset-y-0 left-0 bg-primary transition-[width] duration-1000 ease-linear', !paused && 'track-sweep')}
          style={{ width: `${progress * 100}%` }}
        />
        {!paused && (
          <span
            aria-hidden
            className="absolute inset-y-0 w-[3px] bg-ink shadow-[0_0_10px_2px_var(--color-primary)] transition-[left] duration-1000 ease-linear"
            style={{ left: `calc(${progress * 100}% - 3px)` }}
          />
        )}
      </div>
      <p className="flex justify-between text-caption text-body">
        <span>Last bar {barTime(data.last_bar)}</span>
        <span>Next {barTime(data.next_bar)}</span>
      </p>

      <ol aria-label={`Last ${tape.length} signals: ${right} right, ${wrong} wrong, ${waiting} waiting`} className="flex flex-wrap gap-xxxs pt-xxs">
        {tape.map((row, i) => (
          <li
            key={row.bar_time}
            title={squareLabel(row)}
            aria-label={squareLabel(row)}
            className={cn('tape-in size-xs', squareClass(row))}
            style={{ animationDelay: engaged ? `${300 + i * 35}ms` : '0ms' }}
          />
        ))}
      </ol>
      <p className="flex flex-wrap gap-x-sm gap-y-xxxs text-caption text-body">
        <span className="flex items-center gap-xxxs">
          <span aria-hidden className="size-xxs bg-up" /> Right
        </span>
        <span className="flex items-center gap-xxxs">
          <span aria-hidden className="size-xxs bg-down" /> Wrong
        </span>
        <span className="flex items-center gap-xxxs">
          <span aria-hidden className="size-xxs border border-body" /> Waiting
        </span>
        <span className="flex items-center gap-xxxs">
          <span aria-hidden className="size-xxs bg-muted" /> Flat or no data
        </span>
        <span>
          One square per {bar}, newest on the right. {data.live.recorded} recorded, {data.live.scored} scored.
        </span>
      </p>
    </section>
  );
}
