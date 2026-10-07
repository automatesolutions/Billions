import type { Metadata } from 'next';
import Link from 'next/link';
import { notFound } from 'next/navigation';
import { Suspense } from 'react';
import { IconArrowLeft } from '@tabler/icons-react';
import { ApiError, getAnalysis } from '@/lib/api';
import { STRATEGY_COPY } from '@/lib/strategies';
import { STRATEGIES, type Strategy } from '@/types/api';
import { AnalysisView } from './analysis-view';
import { AnalysisError } from './analysis-error';
import { AnalysisSkeleton } from './analysis-skeleton';

const TICKER = /^[A-Z][A-Z0-9.-]{0,9}$/;

type Params = { params: Promise<{ ticker: string }>; searchParams: Promise<{ from?: string }> };

export async function generateMetadata({ params }: Params): Promise<Metadata> {
  const ticker = (await params).ticker.toUpperCase();
  const title = `${ticker} analysis`;
  const description = `${ticker}: signal, edge, risk and out-of-sample model checks from daily prices. Information only.`;
  return {
    title,
    description,
    alternates: { canonical: `/analysis/${ticker}` },
    openGraph: { title, description, url: `/analysis/${ticker}` },
  };
}

export default async function AnalysisPage({ params, searchParams }: Params) {
  const ticker = decodeURIComponent((await params).ticker).toUpperCase();
  if (!TICKER.test(ticker)) notFound();
  const from = (await searchParams).from;
  const strategy = (STRATEGIES as readonly string[]).includes(from ?? '') ? (from as Strategy) : null;

  return (
    <div className="flex flex-col gap-md">
      <Link
        href={strategy ? `/outliers/${strategy}` : '/outliers/swing'}
        className="inline-flex min-h-lg items-center gap-xxs self-start text-nav uppercase text-body hover:text-ink"
      >
        <IconArrowLeft aria-hidden size={18} stroke={1.75} />
        {strategy ? `Back to ${STRATEGY_COPY[strategy].name.toLowerCase()} outliers` : 'Back to outliers'}
      </Link>
      <Suspense fallback={<AnalysisSkeleton ticker={ticker} />}>
        <AnalysisContent ticker={ticker} />
      </Suspense>
    </div>
  );
}

async function AnalysisContent({ ticker }: { ticker: string }) {
  try {
    const data = await getAnalysis(ticker, { next: { revalidate: 300 }, signal: AbortSignal.timeout(60000) });
    return <AnalysisView data={data} />;
  } catch (e) {
    const error = e instanceof ApiError ? e : new ApiError('Something went wrong.', 500);
    return <AnalysisError ticker={ticker} message={error.message} status={error.status} />;
  }
}
