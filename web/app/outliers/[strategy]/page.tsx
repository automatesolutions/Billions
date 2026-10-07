import type { Metadata } from 'next';
import { pageOpenGraph } from '@/lib/metadata';
import { notFound } from 'next/navigation';
import { getOutliers } from '@/lib/api';
import { STRATEGY_COPY } from '@/lib/strategies';
import { STRATEGIES, type OutliersResponse, type Strategy } from '@/types/api';
import { OutliersView } from './outliers-view';

export const revalidate = 60;

export function generateStaticParams() {
  return STRATEGIES.map((strategy) => ({ strategy }));
}

const isStrategy = (value: string): value is Strategy => (STRATEGIES as readonly string[]).includes(value);

export async function generateMetadata({ params }: { params: Promise<{ strategy: string }> }): Promise<Metadata> {
  const { strategy } = await params;
  if (!isStrategy(strategy)) return {};
  const copy = STRATEGY_COPY[strategy];
  const title = `${copy.name} outliers`;
  const description = `${copy.summary} Ranked by how far each move sits from the group. Information only.`;
  return {
    title,
    description,
    alternates: { canonical: `/outliers/${strategy}` },
    openGraph: pageOpenGraph(title, description, `/outliers/${strategy}`),
  };
}

export default async function OutliersPage({ params }: { params: Promise<{ strategy: string }> }) {
  const { strategy } = await params;
  if (!isStrategy(strategy)) notFound();

  let initial: OutliersResponse | null = null;
  try {
    initial = await getOutliers(strategy, { next: { revalidate } });
  } catch {
    // The client view retries and shows an error state if the API stays unreachable.
  }

  return <OutliersView strategy={strategy} initial={initial} />;
}
