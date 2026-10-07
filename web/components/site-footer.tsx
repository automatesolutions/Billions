import Link from 'next/link';

export const DISCLAIMER = 'Information only. Not financial advice. This tool does not place trades.';

export function SiteFooter() {
  return (
    <footer className="mt-xl border-t border-hairline">
      <div className="mx-auto flex max-w-content flex-col gap-xs px-xs py-md text-body-sm sm:px-md">
        <p className="text-ink">{DISCLAIMER}</p>
        <p>
          Prices from Yahoo Finance. They can be delayed or wrong. Read the{' '}
          <Link href="/methodology" className="text-ink underline underline-offset-4">
            methodology
          </Link>{' '}
          for how the numbers are made and where they stop being useful.
        </p>
      </div>
    </footer>
  );
}
