import Link from 'next/link';
import { Logo } from './logo';
import { NavLinks } from './nav-links';
import { TickerSearch } from './ticker-search';

export function SiteHeader() {
  return (
    <header className="glass sticky top-0 z-20 border-b border-hairline">
      <a
        href="#main"
        className="sr-only focus:not-sr-only focus:absolute focus:left-xs focus:top-xs focus:z-30 focus:bg-canvas focus:p-xs focus:text-ink"
      >
        Skip to content
      </a>
      <div className="mx-auto flex min-h-nav max-w-content flex-wrap items-center gap-x-sm gap-y-xxs px-xs py-xxs sm:px-md">
        <Link href="/" aria-label="BILLIONS home" className="flex min-h-lg items-center">
          <Logo />
        </Link>
        <NavLinks />
        <div className="order-last w-full pb-xxs sm:order-none sm:ml-auto sm:w-auto sm:pb-0">
          <TickerSearch />
        </div>
      </div>
    </header>
  );
}
