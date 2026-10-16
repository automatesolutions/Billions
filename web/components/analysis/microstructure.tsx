'use client';

import { useCallback, useEffect, useState } from 'react';
import type { QuoteOk, QuoteResponse } from '@/types/api';
import { getQuote } from '@/lib/api';
import { dateTimeET, money, plain } from '@/lib/format';
import { useAutoRefresh } from '@/hooks/use-auto-refresh';
import { Metric } from '@/components/analysis/metric';
import { Panel } from '@/components/ui/panel';
import { Skeleton } from '@/components/ui/skeleton';

const REFRESH_MS = 15000;
const STALE_SECONDS = 120;

type QuoteState = { status: 'loading' } | { status: 'error'; message: string } | { status: 'done'; quote: QuoteResponse };

/** Live top-of-book quote. Refreshes every 15s while the market is open and the tab is visible. */
export function useQuote(ticker: string): QuoteState {
  const [state, setState] = useState<QuoteState>({ status: 'loading' });

  const load = useCallback(() => {
    getQuote(ticker)
      .then((quote) => setState({ status: 'done', quote }))
      .catch((error: Error) => setState((prev) => (prev.status === 'done' ? prev : { status: 'error', message: error.message })));
  }, [ticker]);

  useEffect(load, [load]);
  const live = state.status === 'done' && state.quote.available && state.quote.market_open;
  useAutoRefresh(load, REFRESH_MS, live, true);
  return state;
}

export function Microstructure({ state }: { state: QuoteState }) {
  const quote = state.status === 'done' ? state.quote : null;
  const label = quote?.available ? `Top of book · ${quote.feed.toUpperCase()} via ${quote.source}` : 'Order book';
  return (
    <Panel id="microstructure" label={label} title="Microstructure">
      {state.status === 'loading' && <Skeleton className="h-xxl w-full" />}
      {state.status === 'error' && <Unavailable reason={state.message} />}
      {quote && (quote.available ? <QuoteView quote={quote} /> : <Unavailable reason={quote.reason} />)}
    </Panel>
  );
}

function Unavailable({ reason }: { reason: string }) {
  return (
    <>
      <p className="text-title-sm text-ink">Not available with current data source</p>
      <p className="mt-xxs text-body-md">{reason}</p>
    </>
  );
}

function QuoteView({ quote }: { quote: QuoteOk }) {
  const stale = !quote.market_open || (quote.age_seconds ?? Infinity) > STALE_SECONDS;
  const imbalance = quote.imbalance;
  return (
    <div className="flex flex-col gap-sm">
      <dl className="grid grid-cols-2 gap-sm">
        <Metric label="Bid" value={money(quote.bid)} note={`${quote.bid_size.toLocaleString('en-US')} size`} />
        <Metric label="Ask" value={money(quote.ask)} note={`${quote.ask_size.toLocaleString('en-US')} size`} />
        <Metric label="Mid-price" value={money(quote.mid)} note={quote.microprice ? `${money(quote.microprice)} size-weighted` : undefined} />
        <Metric label="Spread" value={`${quote.spread_bps.toFixed(1)} bps`} note={`${money(quote.spread)} wide`} />
      </dl>

      {imbalance !== null && <ImbalanceBar imbalance={imbalance} />}

      <p className="text-body-sm" role="status">
        {stale ? 'Last quote' : 'Quote'}
        {quote.quoted_at ? ` at ${dateTimeET(quote.quoted_at)}` : ''}
        {quote.market_open ? '' : ' · market closed, so this may be old'}. Quotes come from the {quote.feed.toUpperCase()} exchange
        only, not the whole market, so the real spread is usually tighter.
      </p>
    </div>
  );
}

function ImbalanceBar({ imbalance }: { imbalance: number }) {
  const bidShare = ((imbalance + 1) / 2) * 100;
  const lean = Math.abs(imbalance) < 0.2 ? 'Balanced' : imbalance > 0 ? 'More size on the bid' : 'More size on the ask';
  return (
    <div className="flex flex-col gap-xxs">
      <div className="flex items-baseline justify-between">
        <p className="text-caption-upper uppercase text-body">Order-book imbalance</p>
        <p className="tabular text-title-sm text-ink">{plain(imbalance)}</p>
      </div>
      <div
        role="meter"
        aria-label="Order-book imbalance"
        aria-valuemin={-1}
        aria-valuemax={1}
        aria-valuenow={imbalance}
        aria-valuetext={`${plain(imbalance)}, ${lean.toLowerCase()}`}
        className="flex h-xxs w-full overflow-hidden bg-elevated"
      >
        <span className="h-full bg-up" style={{ width: `${bidShare}%` }} />
        <span className="h-full bg-down" style={{ width: `${100 - bidShare}%` }} />
      </div>
      <p className="text-body-sm">
        {lean}. Imbalance is (bid size − ask size) ÷ (bid size + ask size), at the best prices only. It changes in
        milliseconds and is not a forecast.
      </p>
    </div>
  );
}
