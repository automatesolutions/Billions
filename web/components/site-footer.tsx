import Link from 'next/link';
import { STRATEGIES } from '@/types/api';
import { STRATEGY_COPY } from '@/lib/strategies';
import { Logo } from './logo';

export const DISCLAIMER = 'Information only. Not financial advice. This tool does not place trades.';

const COLUMNS = [
  {
    title: 'Outliers',
    links: STRATEGIES.map((s) => ({ href: `/outliers/${s}`, label: `${STRATEGY_COPY[s].name} · ${STRATEGY_COPY[s].horizon.toLowerCase()}` })),
  },
  {
    title: 'Methodology',
    links: [
      { href: '/methodology#outliers', label: 'How outliers are found' },
      { href: '/methodology#analysis', label: 'How a stock is analyzed' },
      { href: '/methodology#validation', label: 'How the models are checked' },
      { href: '/methodology#limits', label: 'Limits of the data' },
    ],
  },
];

export function SiteFooter() {
  return (
    <footer className="mt-xxl border-t border-hairline">
      <div className="mx-auto grid max-w-content gap-lg px-xs py-xl sm:px-md md:grid-cols-4">
        <div className="flex flex-col gap-xs md:col-span-2">
          <Logo />
          <p className="max-w-prose text-body-sm">
            Stocks moving unusually far from the pack, with a tested quant analysis of each. Prices from Yahoo Finance. They can
            be delayed or wrong.
          </p>
        </div>
        {COLUMNS.map((col) => (
          <nav key={col.title} aria-label={`Footer ${col.title.toLowerCase()}`} className="flex flex-col gap-xxs">
            <p className="text-caption-upper uppercase text-ink">{col.title}</p>
            <ul className="flex flex-col">
              {col.links.map((l) => (
                <li key={l.href}>
                  <Link
                    href={l.href}
                    className="flex min-h-lg items-center text-body-sm text-body transition-colors duration-300 hover:text-ink md:min-h-md"
                  >
                    {l.label}
                  </Link>
                </li>
              ))}
            </ul>
          </nav>
        ))}
      </div>
      <div className="border-t border-hairline">
        <div className="mx-auto flex max-w-content flex-col gap-xxs px-xs py-sm text-body-sm sm:flex-row sm:items-center sm:justify-between sm:px-md">
          <p className="text-ink">{DISCLAIMER}</p>
          <p>© {new Date().getFullYear()} BILLIONS</p>
        </div>
      </div>
    </footer>
  );
}
