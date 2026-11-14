'use client';

import { useCallback, useEffect, useMemo, useState, type FormEvent, type ReactNode } from 'react';
import Link from 'next/link';
import { IconArrowRight, IconArrowsSort, IconPlus, IconSearch, IconX } from '@tabler/icons-react';
import {
  addWatch,
  analyzePair,
  ApiError,
  estimateScan,
  getCandidates,
  getLeadLagResults,
  getPairHistory,
  getScan,
  getWatchlist,
  removeWatch,
  startScan,
} from '@/lib/api';
import { dateET, pct } from '@/lib/format';
import { accuracyText, botHref, CONFIDENCE, lagText, sortResults, type SortKey } from '@/lib/lead-lag';
import { useAutoRefresh } from '@/hooks/use-auto-refresh';
import { cn } from '@/lib/utils';
import type { LeadLagConfidence, LeadLagPair, LeadLagResult, ScanEstimate, ScanStatus } from '@/types/api';
import { Panel } from '@/components/ui/panel';
import { Badge } from '@/components/ui/badge';
import { Button, buttonClass } from '@/components/ui/button';
import { StateMessage } from '@/components/ui/state';
import { Field, Stat } from '@/components/ui/field';
import { BarChart, EventPathChart } from '@/components/charts/lead-lag-charts';

const errorText = (e: unknown) => (e instanceof ApiError ? e.message : 'Something went wrong. Try again.');
const percent0 = (v: number) => `${Math.round(v * 100)}%`;

function minutes(seconds: number): string {
  if (seconds < 60) return 'under a minute';
  const m = Math.round(seconds / 60);
  return m === 1 ? 'about 1 minute' : `about ${m} minutes`;
}

export function LeadLagView() {
  const [selected, setSelected] = useState<LeadLagResult | null>(null);
  const [history, setHistory] = useState<LeadLagResult[]>([]);

  const open = useCallback((r: LeadLagResult) => {
    setSelected(r);
    setHistory([]);
    getPairHistory(r.leader, r.follower)
      .then((h) => {
        setHistory(h.history);
        if (!r.details && h.latest) setSelected(h.latest);
      })
      .catch(() => setHistory([]));
  }, []);

  return (
    <div className="flex flex-col gap-md">
      <p className="max-w-prose border-l-2 border-info pl-xs text-body-md">
        Accuracy here is measured on past prices. It doesn&apos;t guarantee future results: relationships between instruments change,
        and a few seconds of edge can disappear once you pay the spread, slippage, and latency.
      </p>
      <AnalyzePanel onResult={open} />
      <ResultsPanel onOpen={open} selectedKey={selected?.pair_key} />
      {selected && <PairDetail result={selected} history={history} onClose={() => setSelected(null)} />}
      <div className="grid gap-md lg:grid-cols-2">
        <ScanPanel />
        <WatchlistPanel />
      </div>
    </div>
  );
}

// ---------------------------------------------------------------- analyze any pair

function AnalyzePanel({ onResult }: { onResult: (r: LeadLagResult) => void }) {
  const [form, setForm] = useState({ a: '', b: '', threshold: '0.3', lookback: '60', maxWait: '300', days: '60' });
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const set = (key: keyof typeof form) => (value: string) => setForm((f) => ({ ...f, [key]: value }));

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const result = await analyzePair({
        a: form.a.trim().toUpperCase(),
        b: form.b.trim().toUpperCase(),
        threshold_pct: Number(form.threshold),
        lookback_seconds: Number(form.lookback),
        max_wait_seconds: Number(form.maxWait),
        days: Number(form.days),
      });
      onResult(result);
    } catch (err) {
      setError(errorText(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <Panel id="analyze" label="Any two stocks or ETFs" title="Analyze a pair">
      <form onSubmit={submit} className="flex flex-col gap-sm" aria-label="Analyze a pair">
        <div className="grid grid-cols-2 gap-xs sm:grid-cols-[1fr_1fr_repeat(4,minmax(0,0.8fr))]">
          <Field label="First" value={form.a} onChange={set('a')} placeholder="NVDA" />
          <Field label="Second" value={form.b} onChange={set('b')} placeholder="AMD" />
          <Field label="Threshold" suffix="%" type="number" step={0.05} value={form.threshold} onChange={set('threshold')} />
          <Field label="Lookback" suffix="s" type="number" step={1} value={form.lookback} onChange={set('lookback')} />
          <Field label="Max wait" suffix="s" type="number" step={1} value={form.maxWait} onChange={set('maxWait')} />
          <Field label="History" suffix="days" type="number" min={5} max={250} value={form.days} onChange={set('days')} />
        </div>
        <p className="max-w-prose text-caption text-body">
          With minute bars, the lookback and max wait round to whole minutes. Second bars need a paid Polygon plan. On the free
          plan this takes up to a minute, because Polygon allows 5 requests a minute.
        </p>
        {error && (
          <p role="alert" className="text-body-md text-down">
            {error}
          </p>
        )}
        <button
          type="submit"
          disabled={busy || !form.a.trim() || !form.b.trim()}
          className="sheen inline-flex min-h-lg items-center justify-center gap-xxs self-start bg-primary px-md text-button uppercase text-on-primary hover:bg-livery disabled:cursor-not-allowed disabled:opacity-50"
        >
          <IconSearch aria-hidden size={18} stroke={1.75} />
          {busy ? 'Analyzing…' : 'Analyze'}
        </button>
      </form>
    </Panel>
  );
}

// ---------------------------------------------------------------- ranked results

const COLUMNS: { key: SortKey; label: string; align?: 'right' }[] = [
  { key: 'pair', label: 'Leader → follower' },
  { key: 'lag', label: 'Lag', align: 'right' },
  { key: 'accuracy', label: 'Accuracy', align: 'right' },
  { key: 'oos', label: 'Out of sample', align: 'right' },
  { key: 'move', label: 'Avg follower move', align: 'right' },
  { key: 'confidence', label: 'Confidence' },
  { key: 'updated', label: 'Updated' },
];

function ResultsPanel({ onOpen, selectedKey }: { onOpen: (r: LeadLagResult) => void; selectedKey?: string }) {
  const [data, setData] = useState<{ run_date: string | null; results: LeadLagResult[] } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [sort, setSort] = useState<{ key: SortKey; desc: boolean }>({ key: 'confidence', desc: true });
  const [query, setQuery] = useState('');
  const [minConfidence, setMinConfidence] = useState<LeadLagConfidence | 'all'>('all');

  const load = useCallback(() => {
    getLeadLagResults()
      .then((d) => {
        setData(d);
        setError(null);
      })
      .catch((e) => setError(errorText(e)));
  }, []);
  useEffect(load, [load]);
  useAutoRefresh(load, 30000, true, true);

  const rows = useMemo(() => {
    const q = query.trim().toUpperCase();
    const floor = minConfidence === 'all' ? -1 : CONFIDENCE[minConfidence].rank;
    const filtered = (data?.results ?? []).filter(
      (r) => (!q || r.leader.includes(q) || r.follower.includes(q)) && CONFIDENCE[r.confidence].rank >= floor,
    );
    return sortResults(filtered, sort.key, sort.desc);
  }, [data, query, minConfidence, sort]);

  const header = (c: (typeof COLUMNS)[number]) => {
    const active = sort.key === c.key;
    return (
      <th key={c.key} scope="col" aria-sort={active ? (sort.desc ? 'descending' : 'ascending') : 'none'} className={cn('py-xs pr-xs font-semibold', c.align === 'right' && 'text-right')}>
        <button
          type="button"
          onClick={() => setSort((s) => ({ key: c.key, desc: s.key === c.key ? !s.desc : true }))}
          className={cn('inline-flex items-center gap-xxxs uppercase hover:text-ink', active && 'text-ink')}
        >
          {c.label}
          <IconArrowsSort aria-hidden size={12} stroke={1.75} />
        </button>
      </th>
    );
  };

  return (
    <Panel
      id="results"
      label={data?.run_date ? `Latest research: ${dateET(data.run_date)}` : 'Daily research'}
      title="Best pairs today"
      action={
        <div className="flex flex-wrap items-end gap-xs">
          <Field label="Filter" value={query} onChange={setQuery} placeholder="Ticker" className="w-[140px]" />
          <label className="flex flex-col gap-xxxs text-caption-upper uppercase text-body">
            Confidence
            <select
              value={minConfidence}
              onChange={(e) => setMinConfidence(e.target.value as LeadLagConfidence | 'all')}
              className="h-lg rounded-none border border-hairline bg-canvas px-xs text-body-md normal-case tracking-normal text-ink"
            >
              <option value="all">All</option>
              <option value="strong">Strong only</option>
              <option value="moderate">Moderate or better</option>
              <option value="weak">Weak or better</option>
            </select>
          </label>
        </div>
      }
    >
      {error ? (
        <StateMessage kind="error" title="Results aren't available">
          {error}
        </StateMessage>
      ) : !data ? (
        <p className="text-body-md">Loading…</p>
      ) : data.results.length === 0 ? (
        <StateMessage kind="empty" title="No research yet">
          Run a scan below, or wait for the daily research after the market closes.
        </StateMessage>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full min-w-[760px] border-collapse text-left">
            <caption className="sr-only">Pairs ranked by confidence. Select a pair to see its details.</caption>
            <thead>
              <tr className="border-b border-hairline text-caption-upper uppercase text-body">{COLUMNS.map(header)}</tr>
            </thead>
            <tbody className="text-body-md">
              {rows.map((r) => (
                <tr
                  key={r.pair_key}
                  className={cn('group border-b border-hairline transition-colors duration-300 hover:bg-elevated', selectedKey === r.pair_key && 'bg-elevated')}
                >
                  <td className="py-xs pr-xs">
                    <button type="button" onClick={() => onOpen(r)} className="text-title-sm text-ink underline-offset-4 hover:underline">
                      {r.leader} → {r.follower}
                    </button>
                  </td>
                  <td className="py-xs pr-xs text-right tabular">{lagText(r.lag_seconds)}</td>
                  <td className="py-xs pr-xs text-right tabular text-ink">{accuracyText(r.accuracy, r.events)}</td>
                  <td className="py-xs pr-xs text-right tabular">{accuracyText(r.out_of_sample.accuracy, r.out_of_sample.events)}</td>
                  <td className="py-xs pr-xs text-right tabular">{r.avg_follower_move_pct == null ? '—' : pct(r.avg_follower_move_pct, 2)}</td>
                  <td className="py-xs pr-xs">
                    <Badge dot={CONFIDENCE[r.confidence].dot}>{CONFIDENCE[r.confidence].label}</Badge>
                  </td>
                  <td className="py-xs pr-xs">{r.run_date ? dateET(r.run_date) : '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {rows.length === 0 && <p className="py-sm text-body-md">No pairs match these filters.</p>}
        </div>
      )}
    </Panel>
  );
}

// ---------------------------------------------------------------- pair detail

function PairDetail({ result: r, history, onClose }: { result: LeadLagResult; history: LeadLagResult[]; onClose: () => void }) {
  const d = r.details;
  const overTime =
    history.length >= 3
      ? history.map((h) => ({ key: h.run_date ?? '', label: h.run_date ? dateET(h.run_date).replace(/, \d{4}$/, '') : '', value: h.accuracy, note: `${h.events} events` }))
      : (d?.daily ?? []).map((day) => ({ key: day.day, label: dateET(day.day).replace(/, \d{4}$/, ''), value: day.events ? day.accuracy : null, note: `${day.events} events` }));

  return (
    <Panel
      id="detail"
      label={`${r.resolution === 'second' ? 'Second' : 'Minute'} bars, ${r.days} trading days, ${r.source ?? ''}`}
      title={`${r.leader} → ${r.follower}`}
      action={
        <div className="flex flex-wrap gap-xs">
          <Link href={botHref(r)} className={buttonClass('primary')}>
            Use on the Bot page
            <IconArrowRight aria-hidden size={18} stroke={1.75} />
          </Link>
          <Button variant="text" onClick={onClose} aria-label="Close the pair details">
            <IconX aria-hidden size={18} stroke={1.75} />
          </Button>
        </div>
      }
    >
      {r.status !== 'ok' ? (
        <StateMessage kind="empty" title="Not enough data for this pair">
          {r.message}
        </StateMessage>
      ) : (
        <div className="flex flex-col gap-md">
          <div className="grid grid-cols-2 gap-sm sm:grid-cols-3 lg:grid-cols-6">
            <Stat label="Leader" value={r.leader} />
            <Stat label="Lag" value={lagText(r.lag_seconds)} />
            <Stat label="Accuracy" value={accuracyText(r.accuracy, r.events)} />
            <Stat label="Out of sample" value={accuracyText(r.out_of_sample.accuracy, r.out_of_sample.events)} />
            <Stat label="Avg follower move" value={r.avg_follower_move_pct == null ? '—' : pct(r.avg_follower_move_pct, 2)} />
            <div className="flex flex-col gap-xxxs">
              <span className="text-caption-upper uppercase text-body">Confidence</span>
              <Badge dot={CONFIDENCE[r.confidence].dot} className="self-start">
                {CONFIDENCE[r.confidence].label}
              </Badge>
            </div>
          </div>

          <Explain r={r} />

          <div className="grid gap-md lg:grid-cols-2">
            <ChartBlock title="Moves around signal events">{d && <EventPathChart points={d.event_path} />}</ChartBlock>
            <ChartBlock title={history.length >= 3 ? 'Accuracy by research date' : 'Accuracy by trading day'}>
              <BarChart bars={overTime} format={percent0} label="Accuracy over time" reference={{ value: 0.5, text: 'Coin flip 50%' }} />
            </ChartBlock>
            <ChartBlock title="How long the follower took">
              {d && d.lag_histogram.length > 0 ? (
                <BarChart
                  bars={d.lag_histogram.map((h) => ({ key: String(h.lag_seconds), label: lagText(h.lag_seconds), value: h.count }))}
                  format={(v) => String(Math.round(v))}
                  label="Lag distribution: events by time until the follower made half its move"
                />
              ) : (
                <p className="text-body-md">No follow-through events to measure.</p>
              )}
            </ChartBlock>
            <ChartBlock title="Accuracy over the last 5, 20, and 60 trading days">
              <ul className="flex flex-col gap-xs">
                {(['5', '20', '60'] as const).map((w) => (
                  <li key={w} className="flex items-center justify-between border-b border-hairline pb-xxs text-body-md">
                    <span>Last {w} days</span>
                    <span className="tabular text-ink">{accuracyText(r.windows[w].accuracy, r.windows[w].events)}</span>
                  </li>
                ))}
              </ul>
              <p className="mt-xs text-body-sm">If the shorter windows score lower than the longer ones, the relationship may be fading.</p>
            </ChartBlock>
          </div>
        </div>
      )}
    </Panel>
  );
}

function ChartBlock({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="flex flex-col gap-xs">
      <h3 className="text-caption-upper uppercase text-body">{title}</h3>
      {children}
    </section>
  );
}

function Explain({ r }: { r: LeadLagResult }) {
  const p = r.params;
  return (
    <div className="flex max-w-prose flex-col gap-xxs text-body-md">
      <p>
        {r.leader} moved {p.threshold_pct}% or more within {lagText(p.lookback_seconds)} {r.events.toLocaleString('en-US')} times. In{' '}
        {r.accuracy == null ? 'none' : `${Math.round(r.accuracy * 100)}%`} of them, {r.follower} moved the same way over the next{' '}
        {lagText(p.max_wait_seconds)}. A coin flip would score about 50%.
      </p>
      <p>
        Which one leads was decided on the first {r.in_sample.days} days ({accuracyText(r.in_sample.accuracy, r.in_sample.events)}), then tested
        on the latest {r.out_of_sample.days} days ({accuracyText(r.out_of_sample.accuracy, r.out_of_sample.events)}).
        {r.out_of_sample.direction_held === false && ' On the recent days the other ticker led, so treat this pair with care.'}
        {r.p_value != null && ` The chance of this accuracy from luck alone is ${r.p_value < 0.0001 ? 'under 0.01%' : `${(r.p_value * 100).toFixed(2)}%`}.`}
      </p>
    </div>
  );
}

// ---------------------------------------------------------------- scans

function ScanPanel() {
  const [include, setInclude] = useState({ watchlist: true, candidates: true });
  const [estimate, setEstimate] = useState<ScanEstimate | null>(null);
  const [scan, setScan] = useState<ScanStatus | null>(null);
  const [error, setError] = useState<string | null>(null);

  const poll = useCallback(() => {
    getScan()
      .then((s) => setScan(s.scan))
      .catch(() => undefined);
  }, []);
  useEffect(poll, [poll]);
  useAutoRefresh(poll, 5000, scan?.state === 'running', true);

  useEffect(() => {
    setEstimate(null);
    estimateScan(include)
      .then(setEstimate)
      .catch((e) => setError(errorText(e)));
  }, [include]);

  const start = () => {
    setError(null);
    startScan(include)
      .then(setScan)
      .catch((e) => setError(errorText(e)));
  };

  const running = scan?.state === 'running';
  return (
    <Panel id="scan" label="Uses your data plan's rate limit" title="Scan now">
      <div className="flex flex-col gap-sm">
        <div className="flex flex-col gap-xxs text-body-md text-ink">
          <label className="flex items-center gap-xs">
            <input type="checkbox" checked={include.watchlist} onChange={(e) => setInclude((i) => ({ ...i, watchlist: e.target.checked }))} className="size-xs accent-primary" />
            Your watchlist
          </label>
          <label className="flex items-center gap-xs">
            <input type="checkbox" checked={include.candidates} onChange={(e) => setInclude((i) => ({ ...i, candidates: e.target.checked }))} className="size-xs accent-primary" />
            Suggested candidates (sector peers, ETFs and top holdings, index ETFs)
          </label>
        </div>
        {estimate && (
          <p className="text-body-md">
            {estimate.pairs} pairs, {estimate.tickers} tickers ({estimate.cached_tickers} already cached), {estimate.requests} requests to{' '}
            {estimate.source === 'polygon' ? 'Polygon' : 'Alpaca'}. This takes <span className="text-ink">{minutes(estimate.seconds)}</span> on{' '}
            {estimate.resolution} bars.
          </p>
        )}
        {error && (
          <p role="alert" className="text-body-md text-down">
            {error}
          </p>
        )}
        {scan && (
          <p role="status" className="text-body-md">
            {running ? 'Scanning' : scan.state === 'done' ? 'Last scan finished' : 'Last scan failed'}: {scan.done_pairs} of {scan.total_pairs} pairs
            {scan.errors.length > 0 && `, ${scan.errors.length} skipped (${scan.errors.slice(0, 2).join('; ')})`}.
          </p>
        )}
        <Button variant="outline" disabled={running || !estimate || estimate.pairs === 0} onClick={start} className="self-start">
          {running ? 'Scan running' : 'Start the scan'}
        </Button>
        <p className="text-caption text-body">
          The daily research runs this same scan after each trading day&apos;s close when LEAD_LAG_SCHEDULER is on.
        </p>
      </div>
    </Panel>
  );
}

// ---------------------------------------------------------------- watchlist

function WatchlistPanel() {
  const [pairs, setPairs] = useState<LeadLagPair[]>([]);
  const [max, setMax] = useState(0);
  const [candidates, setCandidates] = useState<LeadLagPair[]>([]);
  const [form, setForm] = useState({ a: '', b: '' });
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getWatchlist()
      .then((w) => {
        setPairs(w.pairs);
        setMax(w.max);
      })
      .catch((e) => setError(errorText(e)));
    getCandidates()
      .then((c) => setCandidates(c.pairs))
      .catch(() => undefined);
  }, []);

  const change = (p: Promise<{ pairs: LeadLagPair[] }>) =>
    p
      .then((w) => {
        setPairs(w.pairs);
        setError(null);
      })
      .catch((e) => setError(errorText(e)));

  const watched = new Set(pairs.map((p) => [p.a, p.b].sort().join('|')));
  return (
    <Panel id="watchlist" label={`${pairs.length} of ${max}`} title="Watchlist">
      <form
        className="flex items-end gap-xs"
        onSubmit={(e) => {
          e.preventDefault();
          change(addWatch(form.a.trim().toUpperCase(), form.b.trim().toUpperCase())).then(() => setForm({ a: '', b: '' }));
        }}
      >
        <Field label="First" value={form.a} onChange={(a) => setForm((f) => ({ ...f, a }))} placeholder="SPY" />
        <Field label="Second" value={form.b} onChange={(b) => setForm((f) => ({ ...f, b }))} placeholder="QQQ" />
        <Button variant="outline" type="submit" aria-label="Add the pair to your watchlist">
          <IconPlus aria-hidden size={18} stroke={1.75} />
        </Button>
      </form>
      {error && (
        <p role="alert" className="mt-xs text-body-md text-down">
          {error}
        </p>
      )}
      <ul className="mt-sm flex flex-wrap gap-xs">
        {pairs.map((p) => (
          <li key={`${p.a}-${p.b}`} className="inline-flex items-center gap-xxs border border-hairline pl-xs text-button uppercase text-ink">
            {p.a} / {p.b}
            <button type="button" onClick={() => change(removeWatch(p.a, p.b))} aria-label={`Remove ${p.a} and ${p.b}`} className="flex size-lg items-center justify-center text-body hover:text-ink">
              <IconX aria-hidden size={16} stroke={1.75} />
            </button>
          </li>
        ))}
        {pairs.length === 0 && <li className="text-body-md">No pairs yet. Add one, or pick a suggestion below.</li>}
      </ul>
      {candidates.length > 0 && (
        <details className="mt-sm">
          <summary className="cursor-pointer text-caption-upper uppercase text-body hover:text-ink">Suggestions ({candidates.length})</summary>
          <ul className="mt-xs flex flex-col">
            {candidates.map((c) => (
              <li key={`${c.a}-${c.b}`} className="flex items-center justify-between gap-xs border-b border-hairline py-xxs text-body-sm">
                <span>
                  <span className="text-ink">
                    {c.a} / {c.b}
                  </span>{' '}
                  {c.why}
                </span>
                {!watched.has([c.a, c.b].sort().join('|')) && (
                  <button type="button" onClick={() => change(addWatch(c.a, c.b))} className="text-button uppercase text-ink hover:underline">
                    Add
                  </button>
                )}
              </li>
            ))}
          </ul>
        </details>
      )}
    </Panel>
  );
}
