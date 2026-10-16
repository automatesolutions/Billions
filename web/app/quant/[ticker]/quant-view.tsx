'use client';

import { useCallback, useEffect, useId, useRef, useState } from 'react';
import dynamic from 'next/dynamic';
import Link from 'next/link';
import { IconAlertTriangle, IconCircleCheck, IconCircleX, IconClock, IconInfoCircle } from '@tabler/icons-react';
import type { QuantGate, QuantInterval, QuantOk, QuantOrder, QuantResponse, QuantVerdict } from '@/types/api';
import { QUANT_INTERVALS } from '@/types/api';
import { ApiError, getQuant, setWatch } from '@/lib/api';
import { barTime, bps, money, percent, ratio } from '@/lib/format';
import { INTERVAL_COPY } from '@/lib/intervals';
import { cn } from '@/lib/utils';
import { useReveal } from '@/components/motion';
import { Metric } from '@/components/analysis/metric';
import { LiveChart } from '@/components/charts/live-chart';
import { TrackingTape } from '@/components/quant/tracking-tape';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { PageHead } from '@/components/ui/page-head';
import { Panel } from '@/components/ui/panel';
import { Skeleton } from '@/components/ui/skeleton';

const EquityChart = dynamic(() => import('@/components/charts/equity-chart').then((m) => m.EquityChart), {
  ssr: false,
  loading: () => <Skeleton className="h-chart w-full" />,
});

const MIN_REFRESH_MS = 15_000;
const MAX_REFRESH_MS = 10 * 60_000;

// ---------------------------------------------------------------- data + refresh

/** Re-fetches just after the next bar is due, while the tab is visible. */
function useQuant(initial: QuantResponse) {
  const [data, setData] = useState(initial);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => setData(initial), [initial]);

  const load = useCallback(() => {
    getQuant(initial.ticker, initial.interval)
      .then((next) => {
        setData(next);
        setError(null);
      })
      .catch((e: Error) => setError(e.message));
  }, [initial.ticker, initial.interval]);

  useEffect(() => {
    const due = new Date(data.next_update_at).getTime() - Date.now() + 3000;
    const wait = Math.min(MAX_REFRESH_MS, Math.max(MIN_REFRESH_MS, due));
    const id = setTimeout(() => {
      if (document.visibilityState === 'visible') load();
      else document.addEventListener('visibilitychange', load, { once: true });
    }, wait);
    return () => {
      clearTimeout(id);
      document.removeEventListener('visibilitychange', load);
    };
  }, [data, load]);

  return { data, error, reload: load };
}

export function QuantView({ initial }: { initial: QuantResponse }) {
  const rootRef = useRef<HTMLDivElement>(null);
  useReveal(rootRef);
  const { data, error, reload } = useQuant(initial);

  return (
    <div ref={rootRef} className="flex flex-col gap-md">
      <Header data={data} />
      {error && (
        <p role="alert" className="flex items-center gap-xxs text-body-md text-down">
          <IconAlertTriangle aria-hidden size={18} stroke={1.5} /> Could not refresh: {error} Showing the last result.
        </p>
      )}
      {data.status === 'ok' ? (
        <>
          <ActionPanel data={data} />
          <WorkingPanel data={data} onChange={reload} />
          <BacktestPanel data={data} />
          <HowPanel data={data} />
          <Limits caveats={data.caveats} />
        </>
      ) : (
        <div role="status" className="flex items-start gap-xs border-l-2 border-primary py-xxs pl-xs">
          <IconInfoCircle aria-hidden size={20} stroke={1.5} className="mt-xxxs shrink-0 text-ink" />
          <div>
            <p className="text-title-sm text-ink">Not enough {data.interval_label.toLowerCase()} history</p>
            <p className="text-body-md">
              {data.message} Try a longer interval: there are more usable bars per stock at 1 hour than at 1 minute.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------- header

function useNow(everyMs = 1000) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), everyMs);
    return () => clearInterval(id);
  }, [everyMs]);
  return now;
}

function countdown(ms: number) {
  if (ms <= 0) return 'any moment';
  const s = Math.round(ms / 1000);
  if (s < 60) return `in ${s}s`;
  const m = Math.round(s / 60);
  if (m < 120) return `in ${m} min`;
  const h = Math.round(m / 60);
  return h < 48 ? `in ${h} h` : `in ${Math.round(h / 24)} days`;
}

function Header({ data }: { data: QuantResponse }) {
  const now = useNow();
  return (
    <>
      <PageHead eyebrow="Quant strategy" title={data.ticker}>
        <p className="tabular text-number-lg text-ink lg:text-number-xl lg:font-medium">{money(data.price)}</p>
        <p className="text-body-sm">
          Last {data.interval_label.toLowerCase()} bar: <span className="text-ink">{barTime(data.last_bar)}</span> · Next update{' '}
          <span className="text-ink" suppressHydrationWarning>
            {countdown(new Date(data.next_update_at).getTime() - now)}
          </span>
          {!data.market.is_open && ' · Market closed'}
        </p>
      </PageHead>
      <IntervalSwitcher ticker={data.ticker} current={data.interval} />
    </>
  );
}

function IntervalSwitcher({ ticker, current }: { ticker: string; current: QuantInterval }) {
  return (
    <nav aria-label="Bar interval" className="grid grid-cols-4 border border-hairline">
      {QUANT_INTERVALS.map((iv) => {
        const active = iv === current;
        return (
          <Link
            key={iv}
            href={`/quant/${ticker}?interval=${iv}`}
            aria-current={active ? 'page' : undefined}
            scroll={false}
            className={cn(
              'flex min-h-lg flex-col justify-center border-l border-hairline px-xxs py-xxs transition-colors duration-300 first:border-l-0 sm:px-sm',
              active ? 'bg-livery text-on-primary' : 'text-body hover:bg-black hover:text-ink',
            )}
          >
            <span className={cn('whitespace-nowrap text-nav uppercase sm:text-button', active ? 'text-on-primary' : 'text-ink')}>
              {INTERVAL_COPY[iv].name}
            </span>
            <span className={cn('hidden text-caption sm:block', active && 'text-on-primary')}>{INTERVAL_COPY[iv].note}</span>
          </Link>
        );
      })}
    </nav>
  );
}

// ---------------------------------------------------------------- action

const ACTION_WORD = { long: 'Long', short: 'Short', flat: 'Flat' } as const;

const ORDER_WORD: Record<QuantOrder, string> = {
  buy: 'Buy',
  sell_short: 'Sell short',
  hold_flat: 'Hold',
  hold_long: 'Hold',
  hold_short: 'Hold',
  sell: 'Sell',
  buy_to_cover: 'Buy to cover',
  reverse_to_short: 'Sell + short',
  reverse_to_long: 'Cover + buy',
};

/** Which side of the book the order is on: buying, selling, or no order. */
const ORDER_SIDE: Record<QuantOrder, 'buy' | 'sell' | 'hold'> = {
  buy: 'buy',
  reverse_to_long: 'buy',
  buy_to_cover: 'buy',
  sell: 'sell',
  sell_short: 'sell',
  reverse_to_short: 'sell',
  hold_flat: 'hold',
  hold_long: 'hold',
  hold_short: 'hold',
};

const SIDE_TEXT = { buy: 'text-up', sell: 'text-down', hold: 'text-ink' } as const;

function targetText(position: number) {
  if (position === 0) return 'Flat';
  return `${position > 0 ? 'Long' : 'Short'} ${percent(Math.abs(position), 0)}`;
}

function ActionPanel({ data }: { data: QuantOk }) {
  const a = data.action;
  const order = a.instruction;
  const side = ORDER_SIDE[order.code];
  const failed = a.gates.filter((g) => !g.passed);
  const bar = INTERVAL_COPY[data.interval].bar;
  const why =
    a.action !== 'flat'
      ? `Every check passed. The model expects ${bps(a.forecast_log_return)} over the next ${bar}, so the guide sizes the position at ${percent(Math.abs(a.position), 0)} of your maximum (tanh sizing).`
      : a.model_direction === 'neutral'
        ? `The model has no clear lean for the next ${bar} (${bps(a.forecast_log_return)} expected).`
        : `The model leans ${a.model_direction === 'long' ? 'buy' : 'sell'} (${bps(a.forecast_log_return)} expected over the next ${bar}), but ${failed.length} ${failed.length === 1 ? 'check fails' : 'checks fail'}, so the guide holds no position until every check passes.`;
  return (
    <Panel id="action" label={`Next bar · starts ${barTime(a.for_bar)}`} title="What to do on the next bar">
      <div className="grid gap-md lg:grid-cols-3">
        <div data-reveal className="flex flex-col gap-sm lg:col-span-2">
          <div className="flex flex-col gap-xxs">
            <p className={cn('text-display-lg uppercase sm:text-display-xl', SIDE_TEXT[side])}>{ORDER_WORD[order.code]}</p>
            <p className="text-title-md text-ink">{order.sentence}</p>
          </div>
          <OrderScale side={side} strength={a.strength} band={data.explain.neutral_band} />
          <p className="max-w-prose text-body-md">{why}</p>
          <dl className="grid grid-cols-2 gap-sm sm:grid-cols-4">
            <Metric label="Held now" value={targetText(order.previous_position)} note="Share of max, last bar" />
            <Metric label="Next bar" value={targetText(a.position)} note="Share of max, target" />
            <Metric label="Forecast" value={bps(a.forecast_log_return)} note={`Next ${bar}, stacked model`} />
            <Metric label="Expected price" value={money(a.expected_price)} note={`From ${money(data.price)}`} />
          </dl>
          <PositionCalculator position={a.position} price={data.price} />
          <p className="text-body-sm">
            The order assumes you followed the guide on the last bar. If you hold something else, move to the
            &ldquo;Next bar&rdquo; target instead.
          </p>
        </div>
        <div data-reveal className="flex flex-col gap-xs border-t border-hairline pt-xs lg:border-l lg:border-t-0 lg:pl-md lg:pt-0">
          <h3 className="text-caption-upper uppercase text-body">Checks before acting</h3>
          <GateList gates={a.gates} />
        </div>
      </div>
    </Panel>
  );
}

/** Sell | Hold | Buy, with the active order lit and a tick where the model's raw lean sits (−1 to +1). */
function OrderScale({ side, strength, band }: { side: 'buy' | 'sell' | 'hold'; strength: number; band: number }) {
  const lean = ((Math.max(-1, Math.min(1, strength)) + 1) / 2) * 100;
  const hold = band * 100; // the neutral band (±band) as a share of the −1..+1 range, so the tick lands in the right segment
  const segments = [
    { id: 'sell', label: 'Sell', width: 50 - hold / 2, on: 'bg-down text-canvas' },
    { id: 'hold', label: 'Hold', width: hold, on: 'bg-ink text-canvas' },
    { id: 'buy', label: 'Buy', width: 50 - hold / 2, on: 'bg-up text-canvas' },
  ] as const;
  return (
    <div className="flex max-w-prose flex-col gap-xxs">
      <div
        role="img"
        aria-label={`Order side: ${side}. Model lean ${ratio(strength)} on a scale from −1 (sell) to +1 (buy).`}
        className="relative flex h-lg w-full border border-hairline"
      >
        {segments.map((seg) => (
          <span
            key={seg.id}
            className={cn(
              'flex items-center justify-center border-l border-hairline text-button uppercase transition-colors duration-500 first:border-l-0',
              side === seg.id ? seg.on : 'text-body',
            )}
            style={{ width: `${seg.width}%` }}
          >
            {seg.label}
          </span>
        ))}
        <span aria-hidden className="absolute -bottom-xxs h-xs w-[3px] -translate-x-1/2 bg-primary" style={{ left: `${lean}%` }} />
      </div>
      <p className="flex justify-between text-caption text-body" aria-hidden>
        <span>−1</span>
        <span className="text-ink">Red tick: model lean {ratio(strength)}</span>
        <span>+1</span>
      </p>
    </div>
  );
}

function GateList({ gates }: { gates: QuantGate[] }) {
  return (
    <ul className="flex flex-col gap-xs">
      {gates.map((g) => (
        <li key={g.id} className="flex items-start gap-xxs">
          {g.passed ? (
            <IconCircleCheck aria-label="Passed" size={20} stroke={1.5} className="mt-px shrink-0 text-up" />
          ) : (
            <IconCircleX aria-label="Failed" size={20} stroke={1.5} className="mt-px shrink-0 text-down" />
          )}
          <span>
            <span className="block text-title-sm text-ink">{g.label}</span>
            <span className="block text-body-sm">{g.detail}</span>
          </span>
        </li>
      ))}
    </ul>
  );
}

function PositionCalculator({ position, price }: { position: number; price: number }) {
  const [max, setMax] = useState('');
  const id = useId();
  const value = Number(max);
  const dollars = Number.isFinite(value) && value > 0 ? value * position : null;
  return (
    <div className="flex flex-wrap items-end gap-xs">
      <div className="flex flex-col gap-xxxs">
        <label htmlFor={id} className="text-caption-upper uppercase text-body">
          Your max position ($)
        </label>
        <input
          id={id}
          type="number"
          inputMode="decimal"
          min={0}
          step={100}
          placeholder="10000"
          value={max}
          onChange={(e) => setMax(e.target.value)}
          className="h-lg w-xxl rounded-sm border border-hairline bg-canvas px-xs text-body-md text-ink tabular"
        />
      </div>
      <p className="pb-xxs text-body-md text-ink" role="status">
        {dollars === null
          ? 'Enter an amount to see the size in dollars and shares.'
          : dollars === 0
            ? 'Size: $0. Stay flat.'
            : `${dollars > 0 ? 'Long' : 'Short'} ${money(Math.abs(dollars))}, about ${Math.floor(Math.abs(dollars) / price).toLocaleString('en-US')} shares.`}
      </p>
    </div>
  );
}

// ---------------------------------------------------------------- is it working?

const VERDICT: Record<QuantVerdict, { title: string; tone: 'up' | 'down' | 'muted' }> = {
  too_early: { title: 'Too early to tell', tone: 'muted' },
  in_line: { title: 'Matching the backtest', tone: 'up' },
  underperforming: { title: 'Not working', tone: 'down' },
  outperforming: { title: 'Ahead of the backtest', tone: 'up' },
};

function verdictText(data: QuantOk) {
  const { live } = data;
  const band = live.backtest.hit_rate_band;
  const range = band ? `${percent(band[0])} to ${percent(band[1])}` : '';
  const hit = percent(live.model.hit_rate);
  switch (live.verdict) {
    case 'too_early':
      return `${live.scored} of ${live.min_signals} signals scored. Each closed ${INTERVAL_COPY[data.interval].bar} adds one. Until then, a few lucky or unlucky bars would decide the result.`;
    case 'in_line':
      return data.backtest.random_baseline.beats_random
        ? `The live hit rate of ${hit} is inside the range the backtest predicts for ${live.scored} signals (${range}).`
        : `The live hit rate of ${hit} matches the backtest (${range}), but the backtest itself does not beat random. Matching a coin flip is not an edge.`;
    case 'underperforming':
      return `The live hit rate of ${hit} is below the range the backtest predicts for ${live.scored} signals (${range}). The edge may have faded, so the guide stays flat.`;
    case 'outperforming':
      return `The live hit rate of ${hit} is above the range the backtest predicts (${range}). Keep watching: a short run of luck looks the same.`;
  }
}

function WorkingPanel({ data, onChange }: { data: QuantOk; onChange: () => void }) {
  const { live } = data;
  const noEdge = live.verdict === 'in_line' && !data.backtest.random_baseline.beats_random;
  const v = noEdge ? { ...VERDICT.in_line, tone: 'muted' as const } : VERDICT[live.verdict];
  const Icon = v.tone === 'up' ? IconCircleCheck : v.tone === 'down' ? IconAlertTriangle : IconClock;
  const progress = Math.min(1, live.scored / live.min_signals);
  // Optimistic: the strip appears the moment the button is pressed; the server confirms on reload.
  const [watching, setWatching] = useState(data.watching);
  const [engaged, setEngaged] = useState(false);
  useEffect(() => setWatching(data.watching), [data.watching]);
  const change = (on: boolean) => {
    setWatching(on);
    setEngaged(on);
  };
  return (
    <Panel
      id="working"
      label={`Live record · ${live.scored} scored · ${live.pending} waiting`}
      title="Is it working?"
      action={<TrackButton data={data} watching={watching} onToggle={change} onChange={onChange} />}
    >
      <div className="flex flex-col gap-md">
        {watching && <TrackingTape data={data} engaged={engaged} />}
        <div data-reveal className="flex items-start gap-xs">
          <Icon
            aria-hidden
            size={24}
            stroke={1.5}
            className={cn('mt-px shrink-0', v.tone === 'up' ? 'text-up' : v.tone === 'down' ? 'text-down' : 'text-body')}
          />
          <div className="flex flex-col gap-xxs">
            <p className="text-title-md text-ink">{v.title}</p>
            <p className="max-w-prose text-body-md">{verdictText(data)}</p>
            {live.verdict === 'too_early' && (
              <div
                role="progressbar"
                aria-label="Signals scored"
                aria-valuemin={0}
                aria-valuemax={live.min_signals}
                aria-valuenow={live.scored}
                className="mt-xxs h-xxs w-full max-w-prose bg-elevated"
              >
                <span className="block h-full bg-ink" style={{ width: `${progress * 100}%` }} />
              </div>
            )}
          </div>
        </div>

        <LiveVsBacktest data={data} />

        {live.action.equity.length >= 2 && (
          <section aria-labelledby="live-eq-title" data-reveal>
            <h3 id="live-eq-title" className="text-title-sm text-ink">
              Following the guide live: {bps(live.action.total_log_return ?? NaN)} total
            </h3>
            <p className="text-body-sm">
              Growth of $1 holding each recorded position for one bar, after the recorded cost. Flat bars earn nothing.
            </p>
            <div className="mt-xs">
              <LiveChart
                equity={live.action.equity}
                label={`Live growth of one dollar over ${live.action.equity.length} signals, ending at ${live.action.equity.at(-1)?.toFixed(4)}.`}
              />
            </div>
          </section>
        )}

        <SignalTable data={data} />
      </div>
    </Panel>
  );
}

function LiveVsBacktest({ data }: { data: QuantOk }) {
  const { live, backtest } = data;
  const hitBand = live.backtest.hit_rate_band;
  const evBand = live.backtest.expected_value_band;
  const rows: { label: string; live: string; test: string; note: string }[] = [
    {
      label: 'Hit rate',
      live: percent(live.model.hit_rate),
      test: percent(live.backtest.hit_rate),
      note: hitBand ? `Expected range now: ${percent(hitBand[0])} to ${percent(hitBand[1])}` : 'Direction right, 50% is a coin flip',
    },
    {
      label: 'EV per bar, before costs',
      live: bps(live.model.expected_value ?? NaN, 2),
      test: bps(live.backtest.expected_value ?? NaN, 2),
      note: evBand ? `Expected range now: ${bps(evBand[0], 2)} to ${bps(evBand[1], 2)}` : 'Following the forecast direction',
    },
    {
      label: 'EV per bar, guide after costs',
      live: bps(live.action.expected_value ?? NaN, 2),
      test: bps(backtest.net.expected_value ?? NaN, 2),
      note: `Sized positions, ${data.cost.bps.toFixed(1)} bps round trip`,
    },
    {
      label: 'Beats random',
      live: live.random_baseline ? `${live.random_baseline.percentile.toFixed(0)}%` : '—',
      test: `${backtest.random_baseline.percentile.toFixed(0)}%`,
      note: 'Share of random up/down strategies beaten. 95% is an edge',
    },
  ];
  return (
    <div data-reveal className="-mx-xs overflow-x-auto px-xs">
      <table className="w-full min-w-table border-collapse text-left">
        <caption className="sr-only">Live signals compared with the out-of-sample backtest.</caption>
        <thead>
          <tr className="border-b border-hairline text-caption-upper uppercase text-body">
            <th scope="col" className="py-xs pr-xs font-semibold">Measure</th>
            <th scope="col" className="py-xs pr-xs text-right font-semibold">Live ({live.scored})</th>
            <th scope="col" className="py-xs pr-xs text-right font-semibold">Backtest ({backtest.split.test_bars})</th>
          </tr>
        </thead>
        <tbody className="tabular">
          {rows.map((r) => (
            <tr key={r.label} className="border-b border-hairline">
              <th scope="row" className="py-xs pr-xs text-left font-normal">
                <span className="block text-title-sm text-ink">{r.label}</span>
                <span className="block text-caption text-body">{r.note}</span>
              </th>
              <td className="py-xs pr-xs text-right text-ink">{r.live}</td>
              <td className="py-xs pr-xs text-right">{r.test}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function SignalTable({ data }: { data: QuantOk }) {
  const rows = data.live.recent;
  if (!rows.length) return null;
  return (
    <section aria-labelledby="signals-title" data-reveal>
      <h3 id="signals-title" className="text-title-sm text-ink">
        Recent signals
      </h3>
      <div className="-mx-xs mt-xs overflow-x-auto px-xs">
        <table className="w-full min-w-table border-collapse text-left">
          <thead>
            <tr className="border-b border-hairline text-caption-upper uppercase text-body">
              <th scope="col" className="py-xxs pr-xs font-semibold">Made at close of</th>
              <th scope="col" className="py-xxs pr-xs font-semibold">Order · held</th>
              <th scope="col" className="py-xxs pr-xs text-right font-semibold">Forecast</th>
              <th scope="col" className="py-xxs pr-xs text-right font-semibold">Actual</th>
              <th scope="col" className="py-xxs text-right font-semibold">Direction</th>
            </tr>
          </thead>
          <tbody className="tabular">
            {rows.map((r) => (
              <tr key={r.bar_time} className="border-b border-hairline">
                <td className="py-xxs pr-xs text-ink">{barTime(r.bar_time)}</td>
                <td className="py-xxs pr-xs">
                  <span className={SIDE_TEXT[ORDER_SIDE[r.instruction]]}>{ORDER_WORD[r.instruction]}</span>
                  <span className="text-body">
                    {' '}
                    · {ACTION_WORD[r.action]}
                    {r.action !== 'flat' && ` ${percent(Math.abs(r.position), 0)}`}
                  </span>
                </td>
                <td className="py-xxs pr-xs text-right">{bps(r.forecast_bps / 1e4, 1)}</td>
                <td className="py-xxs pr-xs text-right">{r.realized_bps === null ? '—' : bps(r.realized_bps / 1e4, 1)}</td>
                <td className="py-xxs text-right">
                  {r.status === 'open' ? (
                    <span className="text-body">Waiting</span>
                  ) : r.status === 'void' ? (
                    <span className="text-body">No data</span>
                  ) : r.correct === null ? (
                    <span className="text-body">Flat bar</span>
                  ) : r.correct ? (
                    <span className="text-up">Right</span>
                  ) : (
                    <span className="text-down">Wrong</span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="mt-xs text-body-sm">
        Each signal is saved when its bar closes and scored when the next bar closes. &ldquo;Direction&rdquo; checks the
        model&apos;s forecast, even when the guide said flat, so the record grows on every bar.
      </p>
    </section>
  );
}

function TrackButton({
  data,
  watching,
  onToggle,
  onChange,
}: {
  data: QuantOk;
  watching: boolean;
  onToggle: (on: boolean) => void;
  onChange: () => void;
}) {
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const toggle = () => {
    const on = !watching;
    setBusy(true);
    setMessage(null);
    onToggle(on);
    setWatch(data.ticker, data.interval, on)
      .then(onChange)
      .catch((e) => {
        onToggle(!on); // roll back the optimistic switch
        setMessage(e instanceof ApiError ? e.message : 'Could not change tracking.');
      })
      .finally(() => setBusy(false));
  };
  return (
    <div className="flex flex-col items-start gap-xxxs sm:items-end">
      <Button variant={watching ? 'outline' : 'primary'} onClick={toggle} disabled={busy} aria-pressed={watching}>
        {watching && <span aria-hidden className="rec-dot size-xxs rounded-full bg-primary" />}
        {watching ? 'Tracking · Stop' : 'Track every bar'}
      </Button>
      <p className="max-w-search text-caption text-body sm:text-right" role="status">
        {message ??
          (watching
            ? `Recording every ${INTERVAL_COPY[data.interval].bar}, even with this page closed.`
            : 'Now recording only while this page is open.')}
      </p>
    </div>
  );
}

// ---------------------------------------------------------------- backtest

function BacktestPanel({ data }: { data: QuantOk }) {
  const { backtest: bt } = data;
  const rb = bt.random_baseline;
  const fmt = (iso: string) => barTime(iso);
  return (
    <Panel id="backtest" label={`Out-of-sample · ${bt.split.test_bars} test bars`} title="Backtest">
      <div className="flex flex-col gap-md">
        <dl className="grid grid-cols-2 gap-sm lg:grid-cols-5">
          <Metric label="Hit rate" value={percent(bt.hit_rate)} note="Direction right" />
          <Metric label="EV after costs" value={bps(bt.net.expected_value ?? NaN, 2)} note={`Per bar, ${bt.net.cost_bps.toFixed(1)} bps round trip`} />
          <Metric label="Sharpe" value={ratio(bt.net.sharpe)} note="After costs, annualized" />
          <Metric label="Max drawdown" value={percent(bt.net.max_drawdown)} note="Of the guide" />
          <Metric label="Beats random" value={`${rb.percentile.toFixed(0)}%`} note={rb.beats_random ? 'Edge: above 95%' : 'Below the 95% line'} />
        </dl>
        <EquityChart
          dates={bt.net.dates}
          equity={bt.net.equity}
          hold={bt.net.hold_equity}
          drawdown={bt.net.drawdown}
          formatDate={fmt}
          unit="test bars"
          signalLabel="Following the guide (sized, after costs)"
        />
        <p className="text-body-sm">
          Train: {fmt(bt.split.train_start)} to before {fmt(bt.split.test_start)} ({bt.split.train_bars.toLocaleString('en-US')} bars).
          Test: {fmt(bt.split.test_start)} to {fmt(bt.split.test_end)}. Never shuffled. The guide&apos;s position moves{' '}
          {percent(bt.net.avg_turnover, 0)} of the maximum per bar on average, and every move pays half a round trip.
        </p>

        <div className="grid gap-md lg:grid-cols-2">
          <section aria-labelledby="q-wf-title" data-reveal>
            <h3 id="q-wf-title" className="text-title-sm text-ink">Walk-forward</h3>
            <p className="text-body-md">Re-fit in steps through the second half of the data, tested only on the bars that follow.</p>
            <table className="mt-xs w-full border-collapse text-left">
              <thead>
                <tr className="border-b border-hairline text-caption-upper uppercase text-body">
                  <th scope="col" className="py-xxs pr-xs font-semibold">Window</th>
                  <th scope="col" className="py-xxs pr-xs text-right font-semibold">Refits</th>
                  <th scope="col" className="py-xxs pr-xs text-right font-semibold">Hit rate</th>
                  <th scope="col" className="py-xxs text-right font-semibold">EV / bar</th>
                </tr>
              </thead>
              <tbody className="tabular">
                {bt.walk_forward.map((w) => (
                  <tr key={w.scheme} className="border-b border-hairline">
                    <th scope="row" className="py-xxs pr-xs font-normal text-ink">
                      {w.scheme === 'expanding' ? 'Expanding' : 'Rolling'}
                    </th>
                    <td className="py-xxs pr-xs text-right">{w.fold_count}</td>
                    <td className="py-xxs pr-xs text-right">{percent(w.hit_rate)}</td>
                    <td className="py-xxs text-right">{bps(w.expected_value ?? NaN, 2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>
          <section aria-labelledby="q-models-title" data-reveal>
            <h3 id="q-models-title" className="text-title-sm text-ink">Models on test bars</h3>
            <p className="text-body-md">Before costs, following each model&apos;s direction.</p>
            <table className="mt-xs w-full border-collapse text-left">
              <thead>
                <tr className="border-b border-hairline text-caption-upper uppercase text-body">
                  <th scope="col" className="py-xxs pr-xs font-semibold">Model</th>
                  <th scope="col" className="py-xxs pr-xs text-right font-semibold">Hit rate</th>
                  <th scope="col" className="py-xxs text-right font-semibold">EV / bar</th>
                </tr>
              </thead>
              <tbody className="tabular">
                {bt.models.map((m) => (
                  <tr key={m.model} className={cn('border-b border-hairline', m.model === 'stacked' && 'text-ink')}>
                    <th scope="row" className="py-xxs pr-xs font-normal text-ink">{MODEL_LABEL[m.model]}</th>
                    <td className="py-xxs pr-xs text-right">{percent(m.hit_rate)}</td>
                    <td className="py-xxs text-right">{bps(m.expected_value ?? NaN, 2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>
        </div>
      </div>
    </Panel>
  );
}

const MODEL_LABEL = { ar1: 'AR(1)', xgboost: 'XGBoost', online: 'Passive-Aggressive', stacked: 'Stacked' } as const;

// ---------------------------------------------------------------- how it is built

const unsignedBps = (decimal: number, digits = 2) => `${(decimal * 1e4).toFixed(digits)} bps`;

const WEEKDAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'];

function slotText(data: QuantOk) {
  const e = data.explain;
  if (e.slot_minutes && e.slot !== null) {
    const start = 9 * 60 + 30 + e.slot * e.slot_minutes;
    const end = Math.min(16 * 60, start + e.slot_minutes);
    const t = (m: number) => `${Math.floor(m / 60) > 12 ? Math.floor(m / 60) - 12 : Math.floor(m / 60)}:${String(m % 60).padStart(2, '0')}`;
    return `time-of-session slot ${e.slot + 1} (${t(start)} to ${t(end)} ET)`;
  }
  const day = new Date(`${data.action.for_bar.slice(0, 10)}T12:00:00Z`).getUTCDay();
  return `weekday (${WEEKDAYS[day - 1] ?? 'weekend'})`;
}

function Step({ n, title, children }: { n: number; title: string; children: React.ReactNode }) {
  return (
    <li data-reveal className="grid grid-cols-[auto_1fr] gap-xs border-b border-hairline pb-sm last:border-b-0">
      <span aria-hidden className="tabular text-title-md text-primary">
        {n}
      </span>
      <div className="flex min-w-0 flex-col gap-xxs">
        <h3 className="text-title-sm text-ink">{title}</h3>
        {children}
      </div>
    </li>
  );
}

function Formula({ children }: { children: React.ReactNode }) {
  return <p className="tabular overflow-x-auto whitespace-nowrap bg-elevated px-xs py-xxs text-body-md text-ink">{children}</p>;
}

function HowPanel({ data }: { data: QuantOk }) {
  const e = data.explain;
  const f = e.features;
  const w = e.meta_weights;
  const b = e.base;
  const bar = INTERVAL_COPY[data.interval].bar;
  const sigma = e.bar_volatility;
  return (
    <Panel id="how" label={`Worked example · next ${bar}`} title="How the signal is built">
      <ol className="flex flex-col gap-sm">
        <Step n={1} title="Log returns and features">
          <p className="text-body-md">
            Each bar&apos;s return is r = ln(P now ÷ P before). Features use only bars that have already closed (every
            feature is shifted by one bar), so the model never sees the bar it predicts.
          </p>
          <Formula>
            last returns: {bps(f.lag_1)}, {bps(f.lag_2)}, {bps(f.lag_3)} · 5-bar ago {bps(f.lag_5)} · average of last 5 {bps(f.ma_5, 2)}, last 20{' '}
            {bps(f.ma_20, 2)}
          </Formula>
          <p className="text-body-sm">
            Direction features: last bar {f.dir_1 > 0 ? 'up' : f.dir_1 < 0 ? 'down' : 'flat'}, share of up minus down bars{' '}
            {ratio(f.dirma_5)} (last 5) and {ratio(f.dirma_20)} (last 20). Calendar feature: {slotText(data)}.
          </p>
        </Step>
        <Step n={2} title="AR(1): yesterday's move, scaled">
          <Formula>
            ŷ = w × r(last) + b = {ratio(e.ar1.w, 4)} × {bps(f.lag_1)} + {bps(e.ar1.b, 2)} = {bps(b.ar1, 2)}
          </Formula>
          <p className="text-body-sm">
            w {e.ar1.w < 0 ? 'below 0 means moves have tended to reverse (mean reversion)' : 'above 0 means moves have tended to continue (momentum)'}.
          </p>
        </Step>
        <Step n={3} title="XGBoost: patterns by time and direction">
          <Formula>ŷ = {bps(b.xgboost, 2)}</Formula>
          <p className="text-body-sm">
            50 shallow trees (depth 3, so it cannot memorize) on the calendar slot and the direction features.
          </p>
        </Step>
        <Step n={4} title="Passive-Aggressive: learns after every bar">
          <Formula>ŷ = {bps(b.online, 2)}</Formula>
          <p className="text-body-sm">
            A linear model on the six return features that updates its weights after each bar&apos;s error, so it adapts when
            the market changes.
          </p>
        </Step>
        <Step n={5} title="Stack: a weighted blend">
          <Formula>
            ŷ = {ratio(w.ar1, 3)} × {bps(b.ar1, 2)} + {ratio(w.xgboost, 3)} × {bps(b.xgboost, 2)} + {ratio(w.online, 3)} ×{' '}
            {bps(b.online, 2)} + {bps(e.meta_bias, 2)} = <strong>{bps(e.forecast, 2)}</strong>
          </Formula>
          <p className="text-body-sm">
            Weights can&apos;t go below 0 and were fit on forecasts the base models made for bars they had not seen.
          </p>
        </Step>
        <Step n={6} title="Size: tanh of the forecast">
          <Formula>
            size = tanh(ŷ ÷ ({e.strength_scale} × σ)) = tanh({bps(e.forecast, 2)} ÷ {unsignedBps(e.strength_scale * sigma)}) ={' '}
            <strong>{ratio(data.action.strength)}</strong>
          </Formula>
          <p className="text-body-sm">
            σ = {unsignedBps(sigma, 1)} is a typical {bar}&apos;s move. Size is a share of your maximum: +1 is fully long,
            −1 fully short. Below ±{e.neutral_band} the model has no clear lean.
          </p>
        </Step>
        <Step n={7} title="Checks: act only on a tested edge">
          <p className="text-body-md">
            The size is used only if every check in &ldquo;What to do on the next bar&rdquo; passes. Otherwise the guide is
            flat: {data.action.action === 'flat' ? 'that is the case now.' : `all pass now, so the guide is ${data.action.action}.`}
          </p>
        </Step>
      </ol>
      <details className="mt-sm border-t border-hairline pt-sm">
        <summary className="cursor-pointer text-button uppercase text-ink">See the code</summary>
        <pre className="mt-xs overflow-x-auto bg-black p-xs text-caption leading-relaxed text-ink">
          <code>{CODE}</code>
        </pre>
        <p className="mt-xxs text-body-sm">From api/services/quant_analysis.py and quant_strategy.py, shortened.</p>
      </details>
    </Panel>
  );
}

const CODE = `# Features: only past bars (shift >= 1), rolling windows after the shift
out[f"lag_{k}"] = r.shift(k)                     # k = 1, 2, 3, 5
out[f"ma_{n}"]  = r.shift(1).rolling(n).mean()   # n = 5, 20

# AR(1): ordinary least squares on yesterday's return
w = cov(x, y) / var(x);  b = mean(y) - w * mean(x)

# XGBoost: shallow trees to avoid overfitting
XGBRegressor(max_depth=3, n_estimators=50, learning_rate=0.12)

# Passive-Aggressive (PA-I): update after every bar
error = y - (x @ w + b)
tau = min(C, abs(error) / (x @ x))
w += tau * sign(error) * x;  b += tau * sign(error)

# Stack: non-negative weights on out-of-fold forecasts
weights, _ = nnls(P - P.mean(0), y - y.mean())
forecast = base_preds @ weights + bias

# Size (guide, smooth non-linear) and costs
position = tanh(forecast / (0.1 * bar_volatility))
net = position * r - abs(position - prev_position) * cost / 2

# Validation: 75/25 time split, walk-forward, 1000 random strategies
beats_random = percentile_vs_random >= 95`;

// ---------------------------------------------------------------- limits

function Limits({ caveats }: { caveats: string[] }) {
  return (
    <section aria-labelledby="q-limits" className="flex flex-col gap-xs border-t border-hairline pt-md">
      <h2 id="q-limits" className="text-title-md text-ink">
        What this does not tell you
      </h2>
      <ul className="flex list-disc flex-col gap-xxs pl-sm text-body-md">
        {caveats.map((c) => (
          <li key={c}>{c}</li>
        ))}
        <li>A backtest is one past path. Live results can be worse, and the record above is the honest test.</li>
        <li>Costs here are the quoted spread only. Slippage, fees and taxes are not included.</li>
        <li>
          <Badge>Information only</Badge> This is a model&apos;s output, not advice. BILLIONS does not place trades.
        </li>
      </ul>
    </section>
  );
}
