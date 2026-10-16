import type { Metadata } from 'next';
import { pageOpenGraph } from '@/lib/metadata';
import { PageHead } from '@/components/ui/page-head';
import { QuantPicker, Tracked } from './quant-picker';

const title = 'Quant';
const description =
  'Pick a stock and an interval. See what the stacked model says for the next bar, how it was built, and whether its live signals are working.';

export const metadata: Metadata = { title, description, alternates: { canonical: '/quant' }, openGraph: pageOpenGraph(title, description, '/quant') };

export default function QuantHome() {
  return (
    <div className="flex flex-col gap-lg">
      <PageHead eyebrow="Quant strategy" title="Is the strategy working?">
        <p className="max-w-prose text-body-md">
          Pick a stock and how often you want a signal: every minute, hour, 4 hours or day. You get the action for the next
          bar, the checks behind it, the worked calculation, and a live record that scores every signal against what happened.
        </p>
      </PageHead>
      <QuantPicker />
      <Tracked />
    </div>
  );
}
