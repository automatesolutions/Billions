'use client';

import { useRouter } from 'next/navigation';
import { useTransition } from 'react';
import { Button, ButtonLink } from '@/components/ui/button';
import { StateMessage } from '@/components/ui/state';

export function AnalysisError({ ticker, message, status }: { ticker: string; message: string; status: number }) {
  const router = useRouter();
  const [pending, startTransition] = useTransition();
  const notFound = status === 404;

  return (
    <div className="flex flex-col gap-xs">
      <h1 className="text-display-lg sm:text-display-xl">{ticker}</h1>
      <StateMessage
        kind={notFound ? 'empty' : 'error'}
        title={notFound ? `No prices for ${ticker}` : `Can't analyze ${ticker} right now`}
        action={
          notFound ? (
            <ButtonLink href="/outliers/swing">Back to outliers</ButtonLink>
          ) : (
            <Button variant="primary" disabled={pending} onClick={() => startTransition(() => router.refresh())}>
              {pending ? 'Trying' : 'Try again'}
            </Button>
          )
        }
      >
        {notFound ? 'Check the ticker. BILLIONS covers US-listed stocks with daily prices on Yahoo Finance.' : message}
      </StateMessage>
    </div>
  );
}
