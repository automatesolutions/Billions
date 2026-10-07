import type { Metadata } from 'next';
import type { ReactNode } from 'react';
import Link from 'next/link';
import { IconArrowRight, IconChevronRight } from '@tabler/icons-react';
import { getOutliers } from '@/lib/api';
import { STRATEGY_COPY } from '@/lib/strategies';
import { compactDollars, plain } from '@/lib/format';
import { STRATEGIES, type OutliersResponse } from '@/types/api';
import { AsOf } from '@/components/as-of';
import { Direction } from '@/components/direction';
import { PackField } from '@/components/home/pack-field';
import { TickerSearch } from '@/components/ticker-search';
import { ButtonLink } from '@/components/ui/button';
import { cn } from '@/lib/utils';

export const revalidate = 60;

export const metadata: Metadata = {
  title: { absolute: 'BILLIONS · Stocks moving far from the pack' },
  alternates: { canonical: '/' },
};

const integer = (n: number) => Math.round(n).toLocaleString('en-US');

export default async function Home() {
  let swing: OutliersResponse | null = null;
  try {
    swing = await getOutliers('swing', { next: { revalidate } });
  } catch {
    // The live board is left out when the API is unreachable; the rest of the page still works.
  }
  const live = swing?.as_of ? swing : null;

  return (
    <div className="-mt-md flex flex-col">
      <Hero />
      {live && <LiveBoard data={live} />}
      <Strategies />
      <Livery />
      <Lookup />
    </div>
  );
}

// ---------------------------------------------------------------- hero

function Hero() {
  return (
    <section aria-labelledby="hero-title" className="full-bleed relative isolate flex min-h-hero items-end overflow-hidden bg-black">
      <div aria-hidden className="absolute inset-0 -z-10">
        <div className="bg-grid absolute inset-0 [mask-image:radial-gradient(ellipse_at_70%_45%,black,transparent_75%)]" />
        <div className="streak bg-livery absolute -top-xl right-1/4 h-[140%] w-xl opacity-40 blur-3xl" />
        <div className="streak bg-livery absolute -top-xl right-[8%] h-[140%] w-xxs opacity-60 blur-sm [animation-delay:-4s]" />
        <PackField className="absolute inset-0 size-full" />
        <div className="absolute inset-0 bg-linear-to-r from-canvas via-canvas/70 to-transparent" />
        <div className="absolute inset-x-0 bottom-0 h-1/2 bg-linear-to-t from-canvas to-transparent" />
      </div>

      <div className="mx-auto w-full max-w-content px-xs pb-xl pt-xxl sm:px-md sm:pb-xxl">
        <div className="flex max-w-[760px] flex-col gap-sm">
          <p className="flex items-center gap-xs text-caption-upper uppercase text-ink">
            <span className="stripe" />
            Outlier scanner · NASDAQ
          </p>
          <h1 id="hero-title" className="text-display-lg sm:text-display-xl lg:text-display-mega">
            Stocks moving far from the pack.
          </h1>
          <p className="max-w-prose text-title-sm text-body">
            BILLIONS scans the 1,000 most traded NASDAQ stocks every 30 minutes, ranks the moves that sit furthest from the
            group, then tests each one for a real edge.
          </p>
          <div className="flex flex-wrap gap-xs pt-xs">
            <ButtonLink href="/outliers/swing" variant="primary">
              See today&apos;s outliers
              <IconArrowRight aria-hidden size={18} stroke={1.75} />
            </ButtonLink>
            <ButtonLink href="/methodology" variant="outline">
              How it works
            </ButtonLink>
          </div>
        </div>
      </div>
    </section>
  );
}

// ---------------------------------------------------------------- live board

function LiveBoard({ data }: { data: OutliersResponse }) {
  const top = data.outliers.slice(0, 5);
  return (
    <section aria-labelledby="live-title" className="flex flex-col gap-md pt-xl">
      <SectionHead eyebrow="Right now · Swing" title="The classification" id="live-title">
        <AsOf computedAt={data.computed_at} market={data.market} />
      </SectionHead>

      <dl className="grid grid-cols-2 border-y border-hairline sm:grid-cols-3">
        <Spec label="Outliers now" value={integer(data.outlier_count)} />
        <Spec label="Stocks scanned" value={integer(data.universe_count)} className="border-l border-hairline" />
        <Spec
          label="Liquidity floor a day"
          value={compactDollars(data.min_dollar_volume)}
          className="col-span-2 border-t border-hairline sm:col-span-1 sm:border-l sm:border-t-0"
        />
      </dl>

      {top.length > 0 && (
        <ol aria-label="Top five swing outliers" className="flex flex-col">
          {top.map((o) => (
            <li key={o.symbol} className="border-b border-hairline">
              <Link
                href={`/analysis/${o.symbol}?from=swing`}
                className="group relative grid grid-cols-[var(--spacing-xl)_1fr_auto] items-center gap-x-sm gap-y-xxxs py-xs transition-colors duration-300 hover:bg-elevated/40 sm:grid-cols-[var(--spacing-xxl)_var(--spacing-super)_1fr_auto_auto]"
              >
                <span aria-hidden className="absolute inset-y-0 left-0 w-[2px] origin-top scale-y-0 bg-primary transition-transform duration-500 ease-out group-hover:scale-y-100" />
                <span className="pl-xs text-number-lg tabular text-primary sm:text-display-xl">
                  <span className="sr-only">Rank </span>
                  {o.rank}
                </span>
                <span className="text-display-md text-ink">{o.symbol}</span>
                <span className="col-start-2 text-body-md sm:col-start-auto">{o.reason}</span>
                <span className="col-start-3 row-start-1 flex flex-col items-end gap-xxxs sm:col-start-auto sm:row-start-auto">
                  <span className="text-caption-upper uppercase text-body">Score</span>
                  <span className="tabular text-title-md text-ink">{plain(o.score, 1)}</span>
                </span>
                <span className="hidden items-center gap-xs sm:flex">
                  <Direction value={o.direction} />
                  <IconChevronRight
                    aria-hidden
                    size={20}
                    stroke={1.5}
                    className="text-body transition-transform duration-300 group-hover:translate-x-xxxs group-hover:text-ink"
                  />
                </span>
              </Link>
            </li>
          ))}
        </ol>
      )}

      <div>
        <ButtonLink href="/outliers/swing" variant="outline">
          All {integer(data.outlier_count)} swing outliers
        </ButtonLink>
      </div>
    </section>
  );
}

function Spec({ label, value, className }: { label: string; value: string; className?: string }) {
  return (
    <div className={cn('flex flex-col-reverse justify-end gap-xxxs px-xs py-sm sm:px-sm', className)}>
      <dt className="text-caption-upper uppercase text-body">{label}</dt>
      <dd className="text-number-lg tabular text-ink sm:text-number-xl sm:font-medium">{value}</dd>
    </div>
  );
}

// ---------------------------------------------------------------- strategies

function Strategies() {
  return (
    <section aria-labelledby="strategies-title" className="flex flex-col gap-md pt-xxl">
      <SectionHead eyebrow="Three horizons" title="Pick how far you look back" id="strategies-title" />
      <ul className="grid gap-px border border-hairline bg-hairline md:grid-cols-3">
        {STRATEGIES.map((s, i) => {
          const c = STRATEGY_COPY[s];
          return (
            <li key={s} className="bg-canvas">
              <Link
                href={`/outliers/${s}`}
                className="group relative flex h-full min-h-[320px] flex-col justify-between gap-lg overflow-hidden p-sm transition-colors duration-500 hover:bg-black sm:p-md"
              >
                <span aria-hidden className="absolute inset-x-0 top-0 h-[2px] origin-left scale-x-0 bg-primary transition-transform duration-700 ease-out group-hover:scale-x-100" />
                <span aria-hidden className="text-number-xl tabular text-elevated transition-colors duration-500 group-hover:text-primary">
                  {String(i + 1).padStart(2, '0')}
                </span>
                <span className="flex flex-col gap-xxs">
                  <span className="text-caption-upper uppercase text-body">{c.horizon}</span>
                  <span className="text-display-lg text-ink">{c.name}</span>
                  <span className="text-body-md">{c.summary}</span>
                  <span className="mt-xs flex items-center justify-between border-t border-hairline pt-xs text-nav uppercase text-ink">
                    {c.short} · {c.long}
                    <IconArrowRight aria-hidden size={18} stroke={1.5} className="transition-transform duration-300 group-hover:translate-x-xxs" />
                  </span>
                </span>
              </Link>
            </li>
          );
        })}
      </ul>
    </section>
  );
}

// ---------------------------------------------------------------- livery band

const PROOF = [
  { k: '75 / 25', v: 'Time-ordered train and test split. Never shuffled.' },
  { k: '1,000', v: 'Random strategies every model has to beat.' },
  { k: 't − 1', v: 'Every feature uses data up to the day before. No look-ahead.' },
];

function Livery() {
  return (
    <section aria-labelledby="livery-title" className="full-bleed bg-livery mt-xxl">
      <div className="mx-auto grid max-w-content gap-lg px-xs py-xxl sm:px-md lg:grid-cols-[1fr_1.2fr]">
        <div className="flex flex-col gap-sm">
          <p className="text-caption-upper uppercase text-on-primary">Tested, not promised</p>
          <h2 id="livery-title" className="text-display-lg text-on-primary sm:text-display-xl">
            Every number is checked on days the models never saw.
          </h2>
          <div>
            <ButtonLink href="/methodology#validation" variant="outline" className="border-on-primary text-on-primary">
              Read the methodology
            </ButtonLink>
          </div>
        </div>
        <dl className="grid content-end gap-sm sm:grid-cols-3 lg:grid-cols-1">
          {PROOF.map((p) => (
            <div key={p.k} className="flex flex-col gap-xxxs border-t border-on-primary/40 pt-xs">
              <dt className="text-number-lg tabular text-on-primary">{p.k}</dt>
              <dd className="text-body-md text-on-primary">{p.v}</dd>
            </div>
          ))}
        </dl>
      </div>
    </section>
  );
}

// ---------------------------------------------------------------- lookup

function Lookup() {
  return (
    <section aria-labelledby="lookup-title" className="flex flex-col items-start gap-sm pt-xxl">
      <SectionHead eyebrow="Any ticker" title="Analyze one stock" id="lookup-title" />
      <p className="max-w-prose text-title-sm text-body">
        Signal, edge, risk and a cost check for any NASDAQ ticker, from about two years of daily prices.
      </p>
      <TickerSearch variant="hero" />
    </section>
  );
}

// ---------------------------------------------------------------- shared

function SectionHead({ eyebrow, title, id, children }: { eyebrow: string; title: string; id: string; children?: ReactNode }) {
  return (
    <header className="flex flex-wrap items-end justify-between gap-xs">
      <div className="flex flex-col gap-xs">
        <p className="flex items-center gap-xs text-caption-upper uppercase text-body">
          <span className="stripe" />
          {eyebrow}
        </p>
        <h2 id={id} className="text-display-lg">
          {title}
        </h2>
      </div>
      {children}
    </header>
  );
}
