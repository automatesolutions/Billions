'use client';

import { useEffect, useId, useState, type FormEvent } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { IconArrowRight } from '@tabler/icons-react';
import { getWatching } from '@/lib/api';
import { INTERVAL_COPY } from '@/lib/intervals';
import { cn } from '@/lib/utils';
import { QUANT_INTERVALS, type QuantInterval } from '@/types/api';
import { Panel } from '@/components/ui/panel';

const TICKER = /^[A-Za-z][A-Za-z0-9.-]{0,9}$/;

export function QuantPicker() {
  const [ticker, setTicker] = useState('');
  const [interval, setChosen] = useState<QuantInterval>('1h');
  const [invalid, setInvalid] = useState(false);
  const router = useRouter();
  const id = useId();

  const submit = (e: FormEvent) => {
    e.preventDefault();
    const value = ticker.trim();
    if (!TICKER.test(value)) {
      setInvalid(true);
      return;
    }
    router.push(`/quant/${value.toUpperCase()}?interval=${interval}`);
  };

  return (
    <form onSubmit={submit} className="flex max-w-prose flex-col gap-sm" aria-label="Choose a stock and interval">
      <div className="flex flex-col gap-xxxs">
        <label htmlFor={id} className="text-caption-upper uppercase text-body">
          Stock
        </label>
        <input
          id={id}
          value={ticker}
          onChange={(e) => {
            setTicker(e.target.value);
            setInvalid(false);
          }}
          placeholder="Ticker, e.g. AAPL"
          autoComplete="off"
          autoCapitalize="characters"
          spellCheck={false}
          aria-invalid={invalid}
          aria-describedby={invalid ? `${id}-error` : undefined}
          className="h-xl w-full rounded-none border border-hairline bg-canvas px-xs text-title-sm text-ink placeholder:text-body hover:border-muted focus:border-ink"
        />
        {invalid && (
          <p id={`${id}-error`} role="alert" className="text-caption text-down">
            Use letters, numbers, dots or dashes.
          </p>
        )}
      </div>
      <fieldset className="flex flex-col gap-xxxs">
        <legend className="mb-xxxs text-caption-upper uppercase text-body">Signal every</legend>
        <div className="grid grid-cols-2 border border-hairline sm:grid-cols-4">
          {QUANT_INTERVALS.map((iv) => (
            <label
              key={iv}
              className={cn(
                'flex min-h-lg cursor-pointer flex-col justify-center border-hairline px-xs py-xxs transition-colors duration-300 [&:not(:first-child)]:border-l',
                interval === iv ? 'bg-livery text-on-primary' : 'text-body hover:bg-black hover:text-ink',
              )}
            >
              <input type="radio" name="interval" value={iv} checked={interval === iv} onChange={() => setChosen(iv)} className="sr-only" />
              <span className={cn('text-button uppercase', interval === iv ? 'text-on-primary' : 'text-ink')}>{INTERVAL_COPY[iv].name}</span>
              <span className={cn('text-caption', interval === iv && 'text-on-primary')}>{INTERVAL_COPY[iv].note}</span>
            </label>
          ))}
        </div>
      </fieldset>
      <button
        type="submit"
        className="sheen inline-flex min-h-lg items-center justify-center gap-xxs self-start bg-primary px-md text-button uppercase text-on-primary hover:bg-livery"
      >
        Show the strategy
        <IconArrowRight aria-hidden size={18} stroke={1.75} />
      </button>
    </form>
  );
}

export function Tracked() {
  const [state, setState] = useState<{ watching: { ticker: string; interval: QuantInterval }[]; max: number } | null>(null);
  useEffect(() => {
    getWatching()
      .then(setState)
      .catch(() => setState(null));
  }, []);
  if (!state || !state.watching.length) return null;
  return (
    <Panel id="tracked" label={`${state.watching.length} of ${state.max}`} title="Being tracked">
      <ul className="flex flex-wrap gap-xs">
        {state.watching.map((w) => (
          <li key={`${w.ticker}-${w.interval}`}>
            <Link
              href={`/quant/${w.ticker}?interval=${w.interval}`}
              className="inline-flex min-h-lg items-center gap-xxs border border-hairline px-xs text-button uppercase text-ink hover:border-ink"
            >
              {w.ticker} <span className="text-body">· {INTERVAL_COPY[w.interval].name}</span>
            </Link>
          </li>
        ))}
      </ul>
      <p className="mt-xs text-body-sm">These record a signal on every bar, even when no one has the page open.</p>
    </Panel>
  );
}
