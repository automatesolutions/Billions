import Link from 'next/link';
import { NavLinks } from './nav-links';
import { TickerSearch } from './ticker-search';

export function BrandMark() {
  return (
    <span aria-hidden className="inline-flex size-md items-center justify-center bg-primary text-title-sm text-on-primary">
      B
    </span>
  );
}

export function SiteHeader() {
  return (
    <header className="border-b border-hairline">
      <a
        href="#main"
        className="sr-only focus:not-sr-only focus:absolute focus:left-xs focus:top-xs focus:z-10 focus:bg-canvas focus:p-xs focus:text-ink"
      >
        Skip to content
      </a>
      <div className="mx-auto flex max-w-content flex-wrap items-center gap-x-sm gap-y-xs px-xs py-xs sm:px-md">
        <Link href="/outliers/swing" className="flex min-h-lg items-center gap-xxs text-nav uppercase text-ink sm:gap-xs">
          <BrandMark />
          BILLIONS
        </Link>
        <NavLinks />
        <div className="order-last w-full sm:order-none sm:ml-auto sm:w-auto">
          <TickerSearch />
        </div>
      </div>
    </header>
  );
}
