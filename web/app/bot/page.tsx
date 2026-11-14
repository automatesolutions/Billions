import { Suspense } from 'react';
import type { Metadata } from 'next';
import { pageOpenGraph } from '@/lib/metadata';
import { PageHead } from '@/components/ui/page-head';
import { BotView } from './bot-view';

const title = 'Paper bot';
const description =
  'Run a moving-average or lead-lag strategy on an Alpaca paper account. See every decision, order, fill, and delay.';

export const metadata: Metadata = { title, description, alternates: { canonical: '/bot' }, openGraph: pageOpenGraph(title, description, '/bot') };

export default function BotPage() {
  return (
    <div className="flex flex-col gap-lg">
      <PageHead eyebrow="Paper trading only" title="Paper bot">
        <p className="max-w-prose text-body-md">
          Pick a strategy, its settings, and its risk limits, then start the bot. It trades your Alpaca paper account only,
          and only while the market is open. Every decision is logged with its reason, so you can see why it did what it did.
        </p>
      </PageHead>
      <Suspense>
        <BotView />
      </Suspense>
    </div>
  );
}
