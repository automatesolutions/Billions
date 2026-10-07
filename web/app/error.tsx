'use client';

import { Button, ButtonLink } from '@/components/ui/button';

export default function ErrorPage({ reset }: { error: Error; reset: () => void }) {
  return (
    <div role="alert" className="flex flex-col items-start gap-sm py-xl">
      <h1 className="text-display-lg">Something went wrong</h1>
      <p className="max-w-prose text-title-sm text-body">This page didn&apos;t load. Try again, or go back to the outliers.</p>
      <div className="flex flex-wrap gap-xs">
        <Button variant="primary" onClick={reset}>
          Try again
        </Button>
        <ButtonLink href="/outliers/swing">Back to outliers</ButtonLink>
      </div>
    </div>
  );
}
