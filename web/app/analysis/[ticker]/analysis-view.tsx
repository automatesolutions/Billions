'use client';

import { useId, useRef, useState, type ReactNode } from 'react';
import dynamic from 'next/dynamic';
import { IconAlertTriangle, IconCircleCheck, IconInfoCircle } from '@tabler/icons-react';
import type { AnalysisInsufficient, AnalysisOk, AnalysisResponse, ModelRow, RiskContext } from '@/types/api';
import { bps, dateET, money, percent, ratio } from '@/lib/format';
import { cn } from '@/lib/utils';
import { CountUp, useReveal } from '@/components/motion';
import { Direction } from '@/components/direction';
import { Metric } from '@/components/analysis/metric';
import { StrengthGauge } from '@/components/analysis/strength-gauge';
import { Badge } from '@/components/ui/badge';
import { Panel } from '@/components/ui/panel';
import { Skeleton } from '@/components/ui/skeleton';

const PriceChart = dynamic(() => import('@/components/charts/price-chart').then((m) => m.PriceChart), {
  ssr: false,
  loading: () => <Skeleton className="h-chart w-full" />,
});
const EquityChart = dynamic(() => import('@/components/charts/equity-chart').then((m) => m.EquityChart), {
  ssr: false,
  loading: () => <Skeleton className="h-chart w-full" />,
});

const MODEL_NAMES: Record<ModelRow['model'], { name: string; note: string }> = {
  ar1: { name: 'AR(1)', note: 'Linear, yesterday’s return' },
  xgboost: { name: 'XGBoost', note: 'Trees, weekday + recent direction' },
  online: { name: 'Passive-Aggressive', note: 'Online, updates daily' },
  stacked: { name: 'Stacked', note: 'Weighted blend of the three' },
};

const LEAN: Record<'up' | 'down' | 'neutral', string> = {
  up: 'The models lean up',
  down: 'The models lean down',
  neutral: 'No clear lean',
};

const price = (n: number) => money(n);

export function AnalysisView({ data }: { data: AnalysisResponse }) {
  const rootRef = useRef<HTMLDivElement>(null);
  useReveal(rootRef);

  return (
    <div ref={rootRef} className="flex flex-col gap-md">
      <Header data={data} />
      {data.status === 'ok' ? <FullAnalysis data={data} /> : <Insufficient data={data} />}
    </div>
  );
}

// ---------------------------------------------------------------- header

function Header({ data }: { data: AnalysisResponse }) {
  return (
    <header className="flex flex-col gap-xs">
      <p className="text-caption-upper uppercase text-body">Stock analysis</p>
      <div className="flex flex-wrap items-end gap-x-md gap-y-xs">
        <h1 className="text-display-lg sm:text-display-xl">{data.ticker}</h1>
        <p className="pb-xxs text-number-lg text-ink">
          <CountUp value={data.price} format={price} />
        </p>
      </div>
      <p className="text-body-sm">
        Close on <time dateTime={data.session_close} className="text-ink">{dateET(data.as_of)}</time> · Daily prices from Yahoo
        Finance · {data.history_days.toLocaleString('en-US')} trading days
      </p>
    </header>
  );
}

// ---------------------------------------------------------------- short history

function Insufficient({ data }: { data: AnalysisInsufficient }) {
  return (
    <>
      <div role="status" className="flex items-start gap-xs border-l-2 border-primary py-xxs pl-xs">
        <IconInfoCircle aria-hidden size={20} stroke={1.5} className="mt-xxxs shrink-0 text-ink" />
        <div>
          <p className="text-title-sm text-ink">Not enough history for the models</p>
          <p className="text-body-md">
            {data.ticker} has {data.history_days} trading days of prices. The models need at least {data.needed_days}, about
            one year, to train and test fairly. Price and risk are shown below.
          </p>
        </div>
      </div>
      <Panel id="price" label={data.history.length < 252 ? `All ${data.history.length} trading days` : 'Last year'} title="Price">
        <PriceChart history={data.history} ticker={data.ticker} />
      </Panel>
      {data.risk && <RiskPanel risk={data.risk} />}
      <Limits caveats={[]} hasCostCheck={false} />
    </>
  );
}

// ---------------------------------------------------------------- full analysis

function FullAnalysis({ data }: { data: AnalysisOk }) {
  const { signal, validation } = data;
  const baseline = validation.random_baseline;
  return (
    <>
      <SignalSummary data={data} />

      <Panel id="price" label={`Last year + next session`} title="Price and forecast">
        <PriceChart
          history={data.history}
          ticker={data.ticker}
          forecast={
            data.forecast.for_date && data.forecast.expected_price
              ? { date: data.forecast.for_date, expected: data.forecast.expected_price, low: data.forecast.low, high: data.forecast.high }
              : null
          }
        />
        <p className="mt-xs text-body-sm">
          The range is ±1 standard deviation of the model&apos;s errors on test days. It is not a guarantee.
        </p>
      </Panel>

      <Panel id="edge" label={`Out-of-sample · ${data.edge.n} test days`} title="Edge of the signal">
        <dl className="grid grid-cols-2 gap-sm lg:grid-cols-5">
          <Metric label="Expected value" value={bps(data.edge.expected_value ?? NaN)} note="Per day, before costs" />
          <Metric label="Win rate" value={percent(data.edge.win_rate)} note={`${percent(data.edge.avg_win, 2)} avg win`} />
          <Metric label="Avg loss" value={percent(data.edge.avg_loss, 2)} note="On losing days" />
          <Metric label="Sharpe" value={ratio(data.edge.sharpe)} note="Annualized" />
          <Metric label="Max drawdown" value={percent(data.edge.max_drawdown)} note="Of the signal" />
        </dl>
        <div className="mt-md">
          <EquityChart dates={data.edge.dates} equity={data.edge.equity} hold={data.edge.hold_equity} drawdown={data.edge.drawdown} />
        </div>
        <p className="mt-xs text-body-sm">
          &ldquo;Following the signal&rdquo; means taking the sign of each day&apos;s stacked forecast and measuring the
          next day&apos;s return. It is a test of the forecast, not a plan.
        </p>
      </Panel>

      <RiskPanel risk={data.risk} />

      <Panel id="models" label={`Out-of-sample · ${data.edge.n} test days`} title="Model comparison">
        <ModelTable data={data} />
      </Panel>

      <Panel id="validation" label="How the models were tested" title="Validation">
        <Validation data={data} />
      </Panel>

      <div className="grid gap-md lg:grid-cols-2">
        <CostCheck data={data} />
        <Panel id="microstructure" label="Order book" title="Microstructure">
          <p className="text-title-sm text-ink">Not available with current data source</p>
          <p className="mt-xxs text-body-md">{data.microstructure.reason}</p>
        </Panel>
      </div>

      <Limits caveats={data.caveats} beatsRandom={baseline.beats_random} regimeSignificant={signal.regime.significant} />
    </>
  );
}

function SignalSummary({ data }: { data: AnalysisOk }) {
  const { signal, validation } = data;
  const baseline = validation.random_baseline;
  const regime = signal.regime;
  return (
    <Panel id="signal" label={signal.forecast_for ? `Next session · ${dateET(signal.forecast_for)}` : 'Latest'} title="Signal">
      <div className="grid gap-md lg:grid-cols-3">
        <div data-reveal className="flex flex-col gap-xs lg:col-span-2">
          <div className="flex flex-wrap items-center gap-xs">
            <Direction value={signal.direction} className="text-display-md" />
            <span className="text-title-sm text-body">
              {LEAN[signal.direction]} · {signal.label} ({ratio(signal.strength)})
            </span>
          </div>
          <StrengthGauge strength={signal.strength} direction={signal.direction} muted={!baseline.beats_random} />
          <EdgeVerdict beats={baseline.beats_random} percentile={baseline.percentile} pValue={baseline.p_value} />
        </div>
        <dl data-reveal className="flex flex-col gap-xxxs border-t border-hairline pt-xs lg:border-l lg:border-t-0 lg:pl-md lg:pt-0">
          <dt className="text-caption-upper uppercase text-body">Regime</dt>
          <dd className="text-display-md text-ink">{regime.regime === 'momentum' ? 'Momentum' : 'Mean reversion'}</dd>
          <dd className="text-body-md">
            {regime.regime === 'momentum' ? 'Moves have tended to continue.' : 'Moves have tended to reverse.'} AR(1) weight{' '}
            <span className="tabular text-ink">{ratio(regime.w, 3)}</span>
            {regime.significant ? ', statistically significant.' : ', not statistically different from zero.'}
          </dd>
          <dd className="text-caption">In-sample estimate on training days.</dd>
        </dl>
      </div>
    </Panel>
  );
}

function EdgeVerdict({ beats, percentile, pValue }: { beats: boolean; percentile: number; pValue: number }) {
  const Icon = beats ? IconCircleCheck : IconAlertTriangle;
  return (
    <p className="flex items-start gap-xxs text-body-md text-ink">
      <Icon aria-hidden size={20} stroke={1.5} className={cn('mt-px shrink-0', beats ? 'text-up' : 'text-down')} />
      <span>
        {beats ? (
          <>
            <strong className="font-semibold">Edge beats random.</strong> The stacked model did better than{' '}
            {percentile.toFixed(0)}% of random up/down strategies on test days (p = {pValue.toFixed(2)}).
          </>
        ) : (
          <>
            <strong className="font-semibold">No significant edge.</strong> The stacked model did better than only{' '}
            {percentile.toFixed(0)}% of random up/down strategies on test days (p = {pValue.toFixed(2)}). Treat the lean above
            as noise.
          </>
        )}
      </span>
    </p>
  );
}

function RiskPanel({ risk }: { risk: RiskContext }) {
  return (
    <Panel id="risk" label={`Holding the stock · ${risk.days} trading days`} title="Risk">
      <dl className="grid grid-cols-2 gap-sm lg:grid-cols-4">
        <Metric
          label="Volatility"
          value={percent(risk.annual_volatility)}
          note={`A typical year moves about ±${percent(risk.annual_volatility, 0)}`}
        />
        <Metric label="Max drawdown" value={percent(risk.max_drawdown)} note="Largest fall from a peak" />
        <Metric label="Worst day" value={percent(risk.worst_day.return)} note={dateET(risk.worst_day.date)} />
        <Metric label="Sharpe" value={ratio(risk.sharpe)} note="Return per unit of risk, annualized" />
      </dl>
    </Panel>
  );
}

function ModelTable({ data }: { data: AnalysisOk }) {
  const weights = data.meta_weights;
  return (
    <div className="-mx-xs overflow-x-auto px-xs">
      <table className="w-full min-w-table border-collapse text-left">
        <caption className="sr-only">Each model&apos;s out-of-sample results on the test days, and its weight in the stacked model.</caption>
        <thead>
          <tr className="border-b border-hairline text-caption-upper uppercase text-body">
            <th scope="col" className="py-xs pr-xs font-semibold">Model</th>
            <th scope="col" className="py-xs pr-xs text-right font-semibold">Hit rate</th>
            <th scope="col" className="py-xs pr-xs text-right font-semibold">EV / day</th>
            <th scope="col" className="py-xs pr-xs text-right font-semibold">Sharpe</th>
            <th scope="col" className="py-xs text-right font-semibold">Weight</th>
          </tr>
        </thead>
        <tbody className="tabular">
          {data.models.map((m) => {
            const stacked = m.model === 'stacked';
            return (
              <tr key={m.model} data-reveal className={cn('border-b border-hairline', stacked && 'text-ink')}>
                <th scope="row" className="py-xs pr-xs text-left font-normal">
                  <span className="block text-title-sm text-ink">{MODEL_NAMES[m.model].name}</span>
                  <span className="block text-caption text-body">{MODEL_NAMES[m.model].note}</span>
                </th>
                <td className="py-xs pr-xs text-right">{percent(m.hit_rate)}</td>
                <td className="py-xs pr-xs text-right">{bps(m.expected_value ?? NaN)}</td>
                <td className="py-xs pr-xs text-right">{ratio(m.sharpe)}</td>
                <td className="py-xs text-right">{stacked ? `bias ${bps(data.meta_bias)}` : ratio(weights[m.model as keyof typeof weights], 3)}</td>
              </tr>
            );
          })}
          <tr className="text-body">
            <th scope="row" className="py-xs pr-xs text-left font-normal">
              <span className="flex items-center gap-xxs text-title-sm">
                Stacked <Badge>In-sample</Badge>
              </span>
              <span className="block text-caption">Training days the blend was fit on. Expect this to look better.</span>
            </th>
            <td className="py-xs pr-xs text-right">{percent(data.in_sample.hit_rate)}</td>
            <td className="py-xs pr-xs text-right">{bps(data.in_sample.expected_value ?? NaN)}</td>
            <td className="py-xs pr-xs text-right">{ratio(data.in_sample.sharpe)}</td>
            <td className="py-xs text-right">—</td>
          </tr>
        </tbody>
      </table>
      <p className="mt-xs text-body-sm">
        Hit rate is the share of days the forecast got the direction right. 50% is a coin flip. Stack weights are never
        negative; a weight of 0 means the blend ignores that model.
      </p>
    </div>
  );
}

function Validation({ data }: { data: AnalysisOk }) {
  const { split, walk_forward: walk, random_baseline: rb } = data.validation;
  const trainShare = (split.train_days / (split.train_days + split.test_days)) * 100;
  return (
    <div className="flex flex-col gap-md">
      <section aria-labelledby="split-title" data-reveal>
        <h3 id="split-title" className="text-title-sm text-ink">Time-ordered split</h3>
        <div className="mt-xs flex h-xs w-full" aria-hidden>
          <span className="h-full bg-elevated" style={{ width: `${trainShare}%` }} />
          <span className="h-full bg-ink" style={{ width: `${100 - trainShare}%` }} />
        </div>
        <p className="mt-xxs text-body-md">
          Train: {dateET(split.train_start)} to before {dateET(split.test_start)} ({split.train_days} days). Test:{' '}
          {dateET(split.test_start)} to {dateET(split.test_end)} ({split.test_days} days). Never shuffled.
        </p>
      </section>

      <section aria-labelledby="wf-title" data-reveal>
        <h3 id="wf-title" className="text-title-sm text-ink">Walk-forward</h3>
        <p className="text-body-md">The stack is re-fit every month and tested only on the month that follows.</p>
        <table className="mt-xs w-full border-collapse text-left">
          <thead>
            <tr className="border-b border-hairline text-caption-upper uppercase text-body">
              <th scope="col" className="py-xxs pr-xs font-semibold">Window</th>
              <th scope="col" className="py-xxs pr-xs text-right font-semibold">Refits</th>
              <th scope="col" className="py-xxs pr-xs text-right font-semibold">Hit rate</th>
              <th scope="col" className="py-xxs text-right font-semibold">Sharpe</th>
            </tr>
          </thead>
          <tbody className="tabular">
            {walk.map((w) => (
              <tr key={w.scheme} className="border-b border-hairline">
                <th scope="row" className="py-xxs pr-xs font-normal text-ink">
                  {w.scheme === 'expanding' ? 'Expanding (all past days)' : 'Rolling (fixed length)'}
                </th>
                <td className="py-xxs pr-xs text-right">{w.fold_count}</td>
                <td className="py-xxs pr-xs text-right">{percent(w.hit_rate)}</td>
                <td className="py-xxs text-right">{ratio(w.sharpe)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>

      <section aria-labelledby="rb-title" data-reveal>
        <h3 id="rb-title" className="text-title-sm text-ink">Against random</h3>
        <p className="text-body-md">
          Better than {rb.percentile.toFixed(0)}% of {rb.draws.toLocaleString('en-US')} strategies that pick up or down at
          random each test day. An edge needs 95%.
        </p>
        <div className="relative mt-xs h-xxs w-full bg-elevated" aria-hidden>
          <span className="absolute inset-y-0 w-px bg-ink" style={{ left: '95%' }} />
          <span className={cn('absolute inset-y-0 left-0', rb.beats_random ? 'bg-up' : 'bg-body')} style={{ width: `${rb.percentile}%` }} />
        </div>
        <p className="mt-xxs flex justify-between text-caption text-body" aria-hidden>
          <span>0%</span>
          <span>95% line</span>
        </p>
      </section>
    </div>
  );
}

function CostCheck({ data }: { data: AnalysisOk }) {
  const [cost, setCost] = useState(data.default_cost_bps);
  const inputId = useId();
  const cc = data.cost_check;
  const covers = cc.expected_move_bps > cost;
  return (
    <Panel id="cost" label={`Horizon · ${data.horizon_days} trading day`} title="Cost check">
      <div className="flex flex-col gap-xs">
        <label htmlFor={inputId} className="text-caption-upper uppercase text-body">
          Round-trip cost (bps)
        </label>
        <input
          id={inputId}
          type="number"
          inputMode="decimal"
          min={0}
          max={200}
          step={1}
          value={Number.isFinite(cost) ? cost : ''}
          onChange={(e) => setCost(Math.max(0, Number(e.target.value)))}
          className="h-lg w-xxl rounded-sm border border-hairline bg-canvas px-xs text-body-md text-ink tabular"
        />
        <dl className="grid grid-cols-2 gap-xs pt-xs">
          <Metric label="Avg forecast move" value={`${cc.expected_move_bps.toFixed(1)} bps`} note="Per day, test period" />
          <Metric label="Next session" value={`${cc.latest_move_bps.toFixed(1)} bps`} note="Size of the latest forecast" />
        </dl>
        <p className="flex items-start gap-xxs text-body-md text-ink" role="status">
          {covers ? (
            <IconCircleCheck aria-hidden size={20} stroke={1.5} className="mt-px shrink-0 text-up" />
          ) : (
            <IconAlertTriangle aria-hidden size={20} stroke={1.5} className="mt-px shrink-0 text-down" />
          )}
          {covers
            ? `The average forecast move covers a ${cost} bps round trip.`
            : `The average forecast move is smaller than a ${cost} bps round trip, so costs would use up any edge.`}
        </p>
        <p className="text-body-sm">
          1 bps = 0.01%. Realized EV on test days was {bps(cc.expected_value_bps / 1e4)} a day before costs.
        </p>
      </div>
    </Panel>
  );
}

function Limits({
  caveats,
  beatsRandom,
  regimeSignificant,
  hasCostCheck = true,
}: {
  caveats: string[];
  beatsRandom?: boolean;
  regimeSignificant?: boolean;
  hasCostCheck?: boolean;
}) {
  const fixed: ReactNode[] = [
    'What to do. This page describes past data; it is not a recommendation.',
    'Whether a pattern will last. Relationships in prices often stop working without warning.',
    'News, earnings dates or company events. The models see daily prices only.',
    hasCostCheck ? 'Costs, taxes and slippage, except the rough cost check above.' : 'Costs, taxes and slippage.',
    'Intraday moves or order-book pressure. There is no level-2 data.',
  ];
  return (
    <section aria-labelledby="limits-title" className="border border-hairline p-sm">
      <h2 id="limits-title" className="text-title-md text-ink">
        What this does not tell you
      </h2>
      {caveats.length > 0 && (
        <ul className="mt-xs flex flex-col gap-xxs">
          {caveats.map((c) => (
            <li key={c} className="flex items-start gap-xxs text-body-md text-ink">
              <IconAlertTriangle aria-hidden size={18} stroke={1.5} className="mt-px shrink-0 text-down" />
              {c}
            </li>
          ))}
        </ul>
      )}
      <ul className="mt-xs flex list-disc flex-col gap-xxs pl-sm text-body-md">
        {fixed.map((f, i) => (
          <li key={i}>{f}</li>
        ))}
      </ul>
      {beatsRandom === false && regimeSignificant === false && (
        <p className="mt-xs text-body-sm">For this stock, both checks say the same thing: no reliable pattern was found.</p>
      )}
    </section>
  );
}
