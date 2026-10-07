'use client';

import { useState, type FormEvent } from 'react';
import { useRouter } from 'next/navigation';
import { IconSearch } from '@tabler/icons-react';

const TICKER = /^[A-Za-z][A-Za-z0-9.-]{0,9}$/;

export function TickerSearch() {
  const [ticker, setTicker] = useState('');
  const [invalid, setInvalid] = useState(false);
  const router = useRouter();

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
    <form role="search" onSubmit={handleSubmit} className="relative flex items-center">
      <label htmlFor="ticker-search" className="sr-only">
        Ticker
      </label>
      <IconSearch aria-hidden size={18} stroke={1.5} className="pointer-events-none absolute left-xs text-body" />
      <input
        id="ticker-search"
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
        aria-describedby={invalid ? 'ticker-search-error' : undefined}
        className="h-lg w-full rounded-sm border border-hairline bg-canvas pl-lg pr-xs text-body-md text-ink placeholder:text-body sm:w-search"
      />
      <button type="submit" className="sr-only">
        Analyze
      </button>
      {invalid && (
        <p id="ticker-search-error" role="alert" className="absolute top-full mt-xxxs text-caption text-down">
          Use letters, numbers, dots or dashes.
        </p>
      )}
    </form>
  );
}
