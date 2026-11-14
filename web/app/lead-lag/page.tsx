import { Suspense } from 'react';
import type { Metadata } from 'next';
import { pageOpenGraph } from '@/lib/metadata';
import { PageHead } from '@/components/ui/page-head';
import { LeadLagView } from './lead-lag-view';

const title = 'Lead-lag analysis';
const description =
  'Find pairs where one instrument tends to move first. See who leads, by how long, how often the follower follows, and whether it holds up on recent data.';

export const metadata: Metadata = {
  title,
  description,
  alternates: { canonical: '/lead-lag' },
  openGraph: pageOpenGraph(title, description, '/lead-lag'),
};

export default function LeadLagPage() {
  return (
    <div className="flex flex-col gap-lg">
      <PageHead eyebrow="Research" title="Who moves first?">
        <p className="max-w-prose text-body-md">
          Pick two instruments to see whether one tends to move before the other, by how long, and how often the follower
          moves the same way. Each pair is tested on older data, then checked on the most recent days, so you can tell a lasting
          relationship from one that only looks good in hindsight.
        </p>
      </PageHead>
      <Suspense>
        <LeadLagView />
      </Suspense>
    </div>
  );
}
