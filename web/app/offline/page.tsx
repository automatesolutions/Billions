import type { Metadata } from 'next';

export const metadata: Metadata = { title: 'Offline', robots: { index: false } };

export default function OfflinePage() {
  return (
    <div className="flex flex-col items-start gap-sm py-xl">
      <h1 className="text-display-lg">You&apos;re offline</h1>
      <p className="max-w-prose text-title-sm text-body">
        BILLIONS needs a connection to load market data. Reconnect, then reload this page.
      </p>
    </div>
  );
}
