import type { Metadata } from 'next';
import Link from 'next/link';
import { notFound } from 'next/navigation';
import { Suspense } from 'react';
import { IconArrowLeft } from '@tabler/icons-react';
import { pageOpenGraph } from '@/lib/metadata';
import { ApiError, getQuant } from '@/lib/api';
import { QUANT_INTERVALS, type QuantInterval } from '@/types/api';
import { StateMessage } from '@/components/ui/state';
import { Skeleton } from '@/components/ui/skeleton';
import { buttonClass } from '@/components/ui/button';
import { QuantView } from './quant-view';

const TICKER = /^[A-Z][A-Z0-9.-]{0,9}$/;

type Params = { params: Promise<{ ticker: string }>; searchParams: Promise<{ interval?: string }> };

const toInterval = (value?: string): QuantInterval =>
  (QUANT_INTERVALS as readonly string[]).includes(value ?? '') ? (value as QuantInterval) : '1h';

export async function generateMetadata({ params }: Params): Promise<Metadata> {
  const ticker = (await params).ticker.toUpperCase();
  const title = `${ticker} quant`;
  const description = `${ticker}: what the stacked model says now at your chosen interval, how it was built, and whether its live signals are working.`;
  return { title, description, alternates: { canonical: `/quant/${ticker}` }, openGraph: pageOpenGraph(title, description, `/quant/${ticker}`) };
}

export default async function QuantPage({ params, searchParams }: Params) {
  const ticker = decodeURIComponent((await params).ticker).toUpperCase();
  if (!TICKER.test(ticker)) notFound();
  const interval = toInterval((await searchParams).interval);

  return (
    <div className="flex flex-col gap-md">
      <Link href="/quant" className="inline-flex min-h-lg items-center gap-xxs self-start text-nav uppercase text-body hover:text-ink">
        <IconArrowLeft aria-hidden size={18} stroke={1.75} />
        Choose another stock
      </Link>
      <Suspense key={interval} fallback={<QuantSkeleton ticker={ticker} />}>
        <QuantContent ticker={ticker} interval={interval} />
      </Suspense>
    </div>
  );
}

async function QuantContent({ ticker, interval }: { ticker: string; interval: QuantInterval }) {
  try {
    const data = await getQuant(ticker, interval, { signal: AbortSignal.timeout(60000) });
    return <QuantView initial={data} />;
  } catch (e) {
    const error = e instanceof ApiError ? e : new ApiError('Something went wrong.', 500);
    return (
      <StateMessage
        kind="error"
        title={error.status === 404 ? `No prices for ${ticker}` : `Could not load ${ticker}`}
        action={
          <Link href="/quant" className={buttonClass('outline')}>
            Choose another stock
          </Link>
        }
      >
        {error.message}
      </StateMessage>
    );
  }
}

function QuantSkeleton({ ticker }: { ticker: string }) {
  return (
    <div className="flex flex-col gap-md" aria-busy="true">
      <h1 className="text-display-lg sm:text-display-xl">{ticker}</h1>
      <p className="text-body-md">Fitting the models and checking the live record. This takes a few seconds.</p>
      <Skeleton className="h-xl w-full" />
      <Skeleton className="h-chart w-full" />
      <Skeleton className="h-chart w-full" />
    </div>
  );
}
