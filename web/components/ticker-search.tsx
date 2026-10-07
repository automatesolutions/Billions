'use client';

import { useId, useState, type FormEvent } from 'react';
import { useRouter } from 'next/navigation';
import { IconArrowRight, IconSearch } from '@tabler/icons-react';
import { cn } from '@/lib/utils';

const TICKER = /^[A-Za-z][A-Za-z0-9.-]{0,9}$/;

/**
 * Ticker lookup. `compact` sits in the header; `hero` is the large field on the home page,
 * with a visible Analyze button so the action is obvious.
 */
export function TickerSearch({ variant = 'compact' }: { variant?: 'compact' | 'hero' }) {
  const [ticker, setTicker] = useState('');
  const [invalid, setInvalid] = useState(false);
  const router = useRouter();
  const id = useId();
  const hero = variant === 'hero';

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    const value = ticker.trim();
    if (!value) return;
    if (!TICKER.test(value)) {
      setInvalid(true);
      return;
    }
    setInvalid(false);
    setTicker('');
    router.push(`/analysis/${value.toUpperCase()}`);
  };

  return (
    <form role="search" onSubmit={handleSubmit} className={cn('relative flex items-stretch', hero && 'w-full max-w-prose')}>
      <label htmlFor={id} className="sr-only">
        Ticker
      </label>
      <IconSearch
        aria-hidden
        size={hero ? 22 : 18}
        stroke={1.5}
        className={cn('pointer-events-none absolute top-1/2 -translate-y-1/2 text-body', hero ? 'left-sm' : 'left-xs')}
      />
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
        className={cn(
          'w-full border border-hairline bg-canvas text-ink transition-colors duration-300 placeholder:text-body',
          'hover:border-muted focus:border-ink',
          hero ? 'h-xl rounded-none pl-xl pr-xs text-title-sm' : 'h-lg rounded-sm pl-lg pr-xs text-body-md sm:w-search',
        )}
      />
      <button
        type="submit"
        className={cn(
          hero
            ? 'sheen inline-flex shrink-0 items-center gap-xxs bg-primary px-md text-button uppercase text-on-primary hover:bg-livery'
            : 'sr-only',
        )}
      >
        Analyze
        {hero && <IconArrowRight aria-hidden size={18} stroke={1.75} />}
      </button>
      {invalid && (
        <p id={`${id}-error`} role="alert" className="absolute top-full mt-xxxs text-caption text-down">
          Use letters, numbers, dots or dashes.
        </p>
      )}
    </form>
  );
}
