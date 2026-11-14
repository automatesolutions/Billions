'use client';

import { useCallback, useEffect, useRef, useState, type FormEvent } from 'react';
import { useSearchParams } from 'next/navigation';
import { IconPlayerPlay, IconPlayerStop, IconAlertOctagon } from '@tabler/icons-react';
import { ApiError, getBotInfo, getBotStatus, killBot, startBot, stopBot } from '@/lib/api';
import { duration, money, pct, timeSecondsET } from '@/lib/format';
import { useAutoRefresh } from '@/hooks/use-auto-refresh';
import { cn } from '@/lib/utils';
import type { BotChart, BotDecision, BotInfo, BotStartRequest, BotStatus, BotStrategyName, BotTrade } from '@/types/api';
import { Panel } from '@/components/ui/panel';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { StateMessage } from '@/components/ui/state';
import { Field, Segmented, Stat } from '@/components/ui/field';
import { CrossoverChart, barLabel } from '@/components/charts/crossover-chart';

const TOKEN_KEY = 'billions-bot-token';
const REFRESH_MS = 5000;

const tickers = (text: string) =>
  text
    .split(/[\s,]+/)
    .map((t) => t.trim().toUpperCase())
    .filter(Boolean);

const num = (text: string, fallback: number) => {
  const n = Number(text);
  return Number.isFinite(n) ? n : fallback;
};

function useForm() {
  const params = useSearchParams();
  const from = (key: string, fallback: string) => params.get(key) ?? fallback;
  return useState({
    strategy: (params.get('strategy') === 'lead_lag' ? 'lead_lag' : 'ma_crossover') as BotStrategyName,
    tickers: from('tickers', 'AAPL, MSFT'),
    short: '10',
    long: '30',
    timespan: 'day' as 'day' | 'minute',
    poll: '60',
    leader: from('leader', 'NVDA'),
    followers: from('followers', 'AMD'),
    threshold: from('threshold', '0.3'),
    lookback: from('lookback', '5'),
    maxWait: from('max_wait', '30'),
    followerMax: '0.1',
    takeProfit: '0.3',
    stopLoss: '0.2',
    cooldown: '60',
    allowShort: false,
    source: 'alpaca' as 'alpaca' | 'polygon',
    maxDollars: '1000',
    maxOrders: '20',
  });
}

type Form = ReturnType<typeof useForm>[0];

function toRequest(f: Form): BotStartRequest {
  const risk = { max_position_dollars: num(f.maxDollars, 1000), max_orders_per_day: num(f.maxOrders, 20) };
  if (f.strategy === 'ma_crossover') {
    return {
      strategy: 'ma_crossover',
      ma_crossover: { tickers: tickers(f.tickers), short_window: num(f.short, 10), long_window: num(f.long, 30), timespan: f.timespan, poll_seconds: num(f.poll, 60) },
      risk,
    };
  }
  return {
    strategy: 'lead_lag',
    lead_lag: {
      leader: f.leader.trim().toUpperCase(),
      followers: tickers(f.followers),
      threshold_pct: num(f.threshold, 0.3),
      lookback_seconds: num(f.lookback, 5),
      max_wait_seconds: num(f.maxWait, 30),
      follower_max_move_pct: num(f.followerMax, 0.1),
      take_profit_pct: num(f.takeProfit, 0.3),
      stop_loss_pct: num(f.stopLoss, 0.2),
      cooldown_seconds: num(f.cooldown, 60),
      stale_seconds: 60,
      allow_short: f.allowShort,
      source: f.source,
    },
    risk,
  };
}

function message(err: unknown): string {
  if (err instanceof ApiError) return err.message;
  return 'Something went wrong. Try again.';
}

export function BotView() {
  const [info, setInfo] = useState<BotInfo | null>(null);
  const [status, setStatus] = useState<BotStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [token, setToken] = useState('');
  const [busy, setBusy] = useState(false);
  const [form, setForm] = useForm();
  const set = <K extends keyof Form>(key: K) => (value: Form[K]) => setForm((f) => ({ ...f, [key]: value }));

  useEffect(() => {
    setToken(sessionStorage.getItem(TOKEN_KEY) ?? '');
    getBotInfo()
      .then(setInfo)
      .catch((e) => setError(message(e)));
  }, []);

  const refresh = useCallback(() => {
    if (!info?.enabled) return;
    getBotStatus(token || undefined)
      .then((s) => {
        setStatus(s);
        setError(null);
      })
      .catch((e) => setError(message(e)));
  }, [info?.enabled, token]);

  useEffect(refresh, [refresh]);
  useAutoRefresh(refresh, REFRESH_MS, Boolean(info?.enabled), true);

  const act = async (fn: () => Promise<{ message?: string }>, done: string) => {
    setBusy(true);
    setNotice(null);
    try {
      const result = await fn();
      setNotice(result.message ?? done);
      setError(null);
    } catch (e) {
      setError(message(e));
    } finally {
      setBusy(false);
      refresh();
    }
  };

  const saveToken = (value: string) => {
    setToken(value);
    sessionStorage.setItem(TOKEN_KEY, value);
  };

  if (!info) {
    return error ? (
      <StateMessage kind="error" title="The bot service isn't reachable">
        {error}
      </StateMessage>
    ) : (
      <p className="text-body-md">Loading the bot…</p>
    );
  }

  if (!info.enabled) {
    return (
      <StateMessage kind="empty" title="The paper bot is off">
        <p>
          Set <code className="text-ink">BOT_ENABLED=true</code> in the API&apos;s <code className="text-ink">.env</code>, add your Alpaca paper
          keys, and restart the API. If anyone else can reach your API, also set <code className="text-ink">BOT_ADMIN_TOKEN</code>.
        </p>
      </StateMessage>
    );
  }

  const running = Boolean(status?.running);

  return (
    <div className="flex flex-col gap-md">
      {!info.paper_url_ok && (
        <StateMessage kind="error" title="The trading URL isn't the paper URL">
          ALPACA_TRADING_URL must be {info.paper_url}. The bot refuses to start with any other URL.
        </StateMessage>
      )}
      {info.token_required && (
        <div className="max-w-prose">
          <Field label="Bot token" type="password" value={token} onChange={saveToken} hint="The BOT_ADMIN_TOKEN value. Kept in this tab only." />
        </div>
      )}
      {error && (
        <p role="alert" className="border-l-2 border-down pl-xs text-body-md text-down">
          {error}
        </p>
      )}
      {notice && (
        <p role="status" className="border-l-2 border-up pl-xs text-body-md text-ink">
          {notice}
        </p>
      )}

      <StatusBar status={status} />

      <div className="grid gap-md lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)]">
        <Panel id="settings" label={running ? 'Running. Stop to change settings.' : 'Choose and start'} title="Strategy">
          <SettingsForm
            form={form}
            set={set}
            disabled={running || busy}
            onSubmit={(e) => {
              e.preventDefault();
              act(() => startBot(toRequest(form), token || undefined).then(() => ({})), 'The bot started.');
            }}
          />
          <div className="mt-sm flex flex-wrap items-center gap-xs border-t border-hairline pt-sm">
            <Button variant="outline" disabled={!running || busy} onClick={() => act(() => stopBot(token || undefined).then(() => ({})), 'The bot stopped.')}>
              <IconPlayerStop aria-hidden size={18} stroke={1.75} />
              Stop
            </Button>
            <KillSwitch disabled={busy} onKill={(close) => act(() => killBot(close, token || undefined), 'Kill switch done.')} />
          </div>
        </Panel>

        <div className="flex flex-col gap-md">
          <AccountPanel status={status} />
          <PositionsPanel status={status} />
        </div>
      </div>

      {status?.strategy === 'ma_crossover' && <ChartPanel charts={status.chart ?? []} running={running} />}
      <TradesPanel trades={status?.trades ?? []} />
      <OrdersPanel status={status} />
      <DecisionLog decisions={status?.decisions ?? []} />

      {status?.caveats?.length ? (
        <Panel id="caveats" title="Before you trust the results">
          <ul className="flex list-disc flex-col gap-xxs pl-sm text-body-md">
            {status.caveats.map((c) => (
              <li key={c}>{c}</li>
            ))}
          </ul>
        </Panel>
      ) : null}
    </div>
  );
}

function StatusBar({ status }: { status: BotStatus | null }) {
  if (!status) return null;
  const r = status.risk;
  return (
    <div className="flex flex-wrap items-center gap-xs">
      <Badge dot={status.running ? 'up' : 'muted'}>{status.running ? `Running: ${status.strategy_label}` : 'Stopped'}</Badge>
      <Badge dot={status.market.is_open ? 'up' : 'muted'}>{status.market.is_open ? 'Market open' : 'Market closed'}</Badge>
      <Badge>
        Orders today {r.orders_today} of {r.max_orders_per_day}
      </Badge>
      {r.killed && <Badge dot="down">Kill switch on</Badge>}
      {status.stream && (
        <Badge dot={status.stream.connected ? 'up' : 'down'}>
          {status.stream.source} stream {status.stream.connected ? 'connected' : 'disconnected'}
        </Badge>
      )}
      {!status.running && status.stopped_reason && <span className="text-body-sm">{status.stopped_reason}</span>}
    </div>
  );
}

function SettingsForm({
  form,
  set,
  disabled,
  onSubmit,
}: {
  form: Form;
  set: <K extends keyof Form>(key: K) => (value: Form[K]) => void;
  disabled: boolean;
  onSubmit: (e: FormEvent) => void;
}) {
  return (
    <form onSubmit={onSubmit} className="flex flex-col gap-sm" aria-label="Bot settings">
      <fieldset disabled={disabled} className="flex flex-col gap-sm disabled:opacity-60">
        <Segmented
          legend="Strategy"
          value={form.strategy}
          onChange={set('strategy')}
          options={[
            { value: 'ma_crossover', label: 'MA crossover', note: 'Polygon bars' },
            { value: 'lead_lag', label: 'Lead-lag', note: 'Live trade stream' },
          ]}
        />

        {form.strategy === 'ma_crossover' ? (
          <>
            <Field label="Tickers" value={form.tickers} onChange={set('tickers')} hint="Up to 10, separated by commas." />
            <div className="grid grid-cols-2 gap-xs">
              <Field label="Short window" suffix="bars" type="number" min={2} value={form.short} onChange={set('short')} />
              <Field label="Long window" suffix="bars" type="number" min={3} value={form.long} onChange={set('long')} />
            </div>
            <div className="grid grid-cols-2 gap-xs">
              <Segmented
                legend="Bars"
                value={form.timespan}
                onChange={set('timespan')}
                options={[
                  { value: 'day', label: 'Daily' },
                  { value: 'minute', label: 'Minute' },
                ]}
              />
              <Field label="Check every" suffix="seconds" type="number" min={15} value={form.poll} onChange={set('poll')} />
            </div>
          </>
        ) : (
          <>
            <div className="grid grid-cols-2 gap-xs">
              <Field label="Leader" value={form.leader} onChange={set('leader')} hint="Moves first." />
              <Field label="Followers" value={form.followers} onChange={set('followers')} hint="Up to 5, separated by commas." />
            </div>
            <div className="grid grid-cols-3 gap-xs">
              <Field label="Threshold" suffix="%" type="number" step={0.05} value={form.threshold} onChange={set('threshold')} />
              <Field label="Lookback" suffix="s" type="number" step={1} value={form.lookback} onChange={set('lookback')} />
              <Field label="Max wait" suffix="s" type="number" step={1} value={form.maxWait} onChange={set('maxWait')} />
            </div>
            <div className="grid grid-cols-3 gap-xs">
              <Field label="Take profit" suffix="%" type="number" step={0.05} value={form.takeProfit} onChange={set('takeProfit')} />
              <Field label="Stop loss" suffix="%" type="number" step={0.05} value={form.stopLoss} onChange={set('stopLoss')} />
              <Field label="Cooldown" suffix="s" type="number" step={5} value={form.cooldown} onChange={set('cooldown')} />
            </div>
            <Field
              label="Follower still"
              suffix="% or less"
              type="number"
              step={0.05}
              value={form.followerMax}
              onChange={set('followerMax')}
              hint="The follower counts as not moved yet while its move over the lookback stays under this."
            />
            <Segmented
              legend="Price stream"
              value={form.source}
              onChange={set('source')}
              options={[
                { value: 'alpaca', label: 'Alpaca IEX', note: 'Free, real time' },
                { value: 'polygon', label: 'Polygon', note: 'Real-time plan only' },
              ]}
            />
            <label className="flex items-center gap-xs text-body-md text-ink">
              <input type="checkbox" checked={form.allowShort} onChange={(e) => set('allowShort')(e.target.checked)} className="size-xs accent-primary" />
              Short the follower on a leader down move (otherwise the bot skips it)
            </label>
          </>
        )}

        <div className="grid grid-cols-2 gap-xs border-t border-hairline pt-sm">
          <Field label="Max per position" suffix="$" type="number" min={1} value={form.maxDollars} onChange={set('maxDollars')} />
          <Field label="Max orders a day" type="number" min={1} value={form.maxOrders} onChange={set('maxOrders')} />
        </div>
        <button
          type="submit"
          className="sheen inline-flex min-h-lg items-center justify-center gap-xxs self-start bg-primary px-md text-button uppercase text-on-primary hover:bg-livery disabled:cursor-not-allowed"
        >
          <IconPlayerPlay aria-hidden size={18} stroke={1.75} />
          Start the bot
        </button>
      </fieldset>
    </form>
  );
}

/** Two presses: the first arms it for 5 seconds. No browser dialog. */
function KillSwitch({ disabled, onKill }: { disabled: boolean; onKill: (closePositions: boolean) => void }) {
  const [armed, setArmed] = useState(false);
  const [close, setClose] = useState(false);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(() => () => {
    if (timer.current) clearTimeout(timer.current);
  }, []);
  return (
    <div className="flex flex-wrap items-center gap-xs">
      <button
        type="button"
        disabled={disabled}
        onClick={() => {
          if (!armed) {
            setArmed(true);
            timer.current = setTimeout(() => setArmed(false), 5000);
            return;
          }
          setArmed(false);
          onKill(close);
        }}
        className={cn(
          'inline-flex min-h-lg items-center gap-xxs border px-md text-button uppercase transition-colors duration-300 disabled:opacity-50',
          armed ? 'border-down bg-down text-canvas' : 'border-down text-down hover:bg-down hover:text-canvas',
        )}
      >
        <IconAlertOctagon aria-hidden size={18} stroke={1.75} />
        {armed ? 'Press again to confirm' : 'Kill switch'}
      </button>
      <label className="flex items-center gap-xxs text-body-sm">
        <input type="checkbox" checked={close} onChange={(e) => setClose(e.target.checked)} className="size-xs accent-primary" />
        Also close every position
      </label>
    </div>
  );
}

function AccountPanel({ status }: { status: BotStatus | null }) {
  const a = status?.account;
  const change = a?.equity != null && a?.last_equity ? a.equity - a.last_equity : null;
  return (
    <Panel id="account" label="Alpaca paper" title="Account">
      {status?.broker_error ? (
        <p className="text-body-md text-down">{status.broker_error}</p>
      ) : !a ? (
        <p className="text-body-md">Loading…</p>
      ) : (
        <div className="grid grid-cols-2 gap-sm sm:grid-cols-4">
          <Stat label="Equity" value={money(a.equity ?? NaN)} />
          <Stat label="Cash" value={money(a.cash ?? NaN)} />
          <Stat label="Buying power" value={money(a.buying_power ?? NaN)} />
          <Stat label="Today" value={change == null ? '—' : money(change)} tone={change == null ? undefined : change >= 0 ? 'up' : 'down'} />
        </div>
      )}
    </Panel>
  );
}

function Table({ head, children, caption }: { head: string[]; children: React.ReactNode; caption: string }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[640px] border-collapse text-left">
        <caption className="sr-only">{caption}</caption>
        <thead>
          <tr className="border-b border-hairline text-caption-upper uppercase text-body">
            {head.map((h) => (
              <th key={h} scope="col" className="py-xs pr-xs font-semibold">
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="text-body-md">{children}</tbody>
      </table>
    </div>
  );
}

const td = 'border-b border-hairline py-xs pr-xs align-top';

function PositionsPanel({ status }: { status: BotStatus | null }) {
  const rows = status?.positions ?? [];
  return (
    <Panel id="positions" title="Positions">
      {rows.length === 0 ? (
        <p className="text-body-md">No open positions.</p>
      ) : (
        <Table caption="Open paper positions" head={['Ticker', 'Shares', 'Entry', 'Now', 'Value', 'P/L']}>
          {rows.map((p) => (
            <tr key={p.symbol}>
              <td className={cn(td, 'text-ink')}>{p.symbol}</td>
              <td className={cn(td, 'tabular')}>{p.qty}</td>
              <td className={cn(td, 'tabular')}>{money(p.avg_entry_price ?? NaN)}</td>
              <td className={cn(td, 'tabular')}>{money(p.current_price ?? NaN)}</td>
              <td className={cn(td, 'tabular')}>{money(p.market_value ?? NaN)}</td>
              <td className={cn(td, 'tabular', (p.unrealized_pl ?? 0) >= 0 ? 'text-up' : 'text-down')}>
                {money(p.unrealized_pl ?? NaN)} ({pct((p.unrealized_plpc ?? NaN) * 100, 2)})
              </td>
            </tr>
          ))}
        </Table>
      )}
    </Panel>
  );
}

function OrdersPanel({ status }: { status: BotStatus | null }) {
  const rows = status?.orders ?? [];
  return (
    <Panel id="orders" title="Recent orders">
      {rows.length === 0 ? (
        <p className="text-body-md">No orders yet.</p>
      ) : (
        <Table caption="Recent paper orders" head={['Submitted', 'Ticker', 'Side', 'Shares', 'Fill price', 'Status']}>
          {rows.map((o) => (
            <tr key={o.id}>
              <td className={cn(td, 'tabular')}>{o.submitted_at ? timeSecondsET(o.submitted_at) : '—'}</td>
              <td className={cn(td, 'text-ink')}>{o.symbol}</td>
              <td className={cn(td, o.side === 'buy' ? 'text-up' : 'text-down')}>{o.side}</td>
              <td className={cn(td, 'tabular')}>
                {o.filled_qty ?? 0} of {o.qty}
              </td>
              <td className={cn(td, 'tabular')}>{o.filled_avg_price ? money(o.filled_avg_price) : '—'}</td>
              <td className={td}>{o.status}</td>
            </tr>
          ))}
        </Table>
      )}
    </Panel>
  );
}

/** Where each average is now, in words: "The short average is $0.75 (0.32%) below the long one." */
function gapSentence(chart: BotChart): string {
  const last = chart.bars[chart.bars.length - 1];
  if (!last || chart.gap == null || last.long == null) {
    return `Not enough bars yet. The long average needs ${chart.long_window} bars.`;
  }
  const side = chart.gap > 0 ? 'above' : chart.gap < 0 ? 'below' : 'level with';
  const size = `${money(Math.abs(chart.gap))} (${Math.abs((chart.gap / last.long) * 100).toFixed(2)}%)`;
  const next = chart.gap > 0 ? 'The next cross would be a sell, if you hold shares.' : 'The next cross would be a buy.';
  return `The short average is ${chart.gap === 0 ? '' : `${size} `}${side} the long one. ${next}`;
}

function ChartPanel({ charts, running }: { charts: BotChart[]; running: boolean }) {
  const [picked, setPicked] = useState<string | null>(null);
  const chart = charts.find((c) => c.symbol === picked) ?? charts[0];
  return (
    <Panel id="chart" label="Moving-average crossover" title="Price and averages">
      <p className="mb-sm max-w-prose text-body-md">
        The bot buys when the red line (short average) crosses above the blue one (long average) and sells when it crosses back
        below. The shading between them turns green or red at each cross.
      </p>
      {!chart ? (
        <p className="text-body-md">{running ? 'The chart appears after the bot’s first check.' : 'Start the bot to see its chart.'}</p>
      ) : (
        <div className="flex flex-col gap-sm">
          {charts.length > 1 && (
            <Segmented legend="Ticker" value={chart.symbol} onChange={setPicked} options={charts.map((c) => ({ value: c.symbol, label: c.symbol }))} />
          )}
          <p className="text-body-md text-ink">
            {chart.symbol}: {gapSentence(chart)}
          </p>
          <CrossoverChart chart={chart} />
          {chart.bars.length > 0 && (
            <p className="text-body-sm">
              Last completed bar: {barLabel(chart.bars[chart.bars.length - 1].t, chart.timespan)} ET.
              {chart.timespan === 'minute' && ' On Polygon’s free plan this is about 15 minutes behind the market.'}
            </p>
          )}
        </div>
      )}
    </Panel>
  );
}

function TradesPanel({ trades }: { trades: BotTrade[] }) {
  if (trades.length === 0) return null;
  return (
    <Panel id="trades" label="Leader move, signal, order, and fill" title="Lead-lag trades">
      <p className="mb-xs max-w-prose text-body-sm">
        Slippage compares the fill with the follower&apos;s price when the signal fired. A few seconds of edge disappears fast, so
        watch the leader-to-fill time and slippage against your take-profit setting.
      </p>
      <Table
        caption="Lead-lag trades with timings"
        head={['Signal', 'Follower', 'Leader move', 'Price at signal', 'Fill', 'Leader to signal', 'Signal to fill', 'Leader to fill', 'Slippage', 'Result', 'Why']}
      >
        {trades.map((t) => (
          <tr key={t.id}>
            <td className={cn(td, 'tabular')}>{t.signal_at ? timeSecondsET(t.signal_at) : '—'}</td>
            <td className={cn(td, 'text-ink')}>
              {t.side === 'buy' ? 'Long' : 'Short'} {t.follower}
              <span className="block text-caption text-body">{t.status}</span>
            </td>
            <td className={cn(td, 'tabular')}>
              {t.leader} {pct(t.leader_move_pct, 2)}
            </td>
            <td className={cn(td, 'tabular')}>{money(t.follower_price_at_signal)}</td>
            <td className={cn(td, 'tabular')}>{t.entry_price ? money(t.entry_price) : '—'}</td>
            <td className={cn(td, 'tabular')}>{duration(t.timings.leader_to_signal_ms)}</td>
            <td className={cn(td, 'tabular')}>
              {duration(t.timings.signal_to_submit_ms != null && t.timings.submit_to_fill_ms != null ? t.timings.signal_to_submit_ms + t.timings.submit_to_fill_ms : null)}
            </td>
            <td className={cn(td, 'tabular')}>{duration(t.timings.leader_to_fill_ms)}</td>
            <td className={cn(td, 'tabular', (t.timings.slippage_pct ?? 0) > 0 ? 'text-down' : 'text-body')}>
              {t.timings.slippage_pct == null ? '—' : pct(t.timings.slippage_pct, 3)}
            </td>
            <td className={cn(td, 'tabular', t.pnl_pct == null ? '' : t.pnl_pct >= 0 ? 'text-up' : 'text-down')}>
              {t.pnl_pct == null ? '—' : pct(t.pnl_pct, 2)}
            </td>
            <td className={cn(td, 'min-w-[260px] text-body-sm')}>
              <span className="block">In: {t.entry_reason}</span>
              {t.exit_reason && <span className="block text-ink">Out: {t.exit_reason}</span>}
            </td>
          </tr>
        ))}
      </Table>
    </Panel>
  );
}

const STATUS_TONE: Record<BotDecision['status'], string> = {
  placed: 'text-up',
  blocked: 'text-down',
  failed: 'text-down',
  skipped: 'text-body',
  held: 'text-body',
  info: 'text-info',
};

function DecisionLog({ decisions }: { decisions: BotDecision[] }) {
  return (
    <Panel id="decisions" label="Newest first" title="Decision log">
      {decisions.length === 0 ? (
        <p className="text-body-md">No decisions yet. Start the bot to see what it does and why.</p>
      ) : (
        <ol className="flex flex-col">
          {decisions.map((d) => (
            <li key={d.id} className="grid grid-cols-[88px_72px_56px_minmax(0,1fr)] gap-xs border-b border-hairline py-xxs text-body-sm sm:grid-cols-[112px_88px_64px_minmax(0,1fr)]">
              <span className="tabular">{timeSecondsET(d.at)}</span>
              <span className={cn('uppercase text-caption-upper', STATUS_TONE[d.status])}>{d.status}</span>
              <span className="text-ink">{d.symbol ?? ''}</span>
              <span className="text-body-md text-ink">{d.reason}</span>
            </li>
          ))}
        </ol>
      )}
    </Panel>
  );
}
