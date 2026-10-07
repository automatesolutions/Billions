import type { Metadata } from 'next';
import { ButtonLink } from '@/components/ui/button';

export const metadata: Metadata = { title: 'Page not found' };

export default function NotFound() {
  return (
    <div className="flex flex-col items-start gap-sm py-xl">
      <p className="text-caption-upper uppercase text-body">404</p>
      <h1 className="text-display-lg">Page not found</h1>
      <p className="max-w-prose text-title-sm text-body">This address doesn&apos;t match a page. Check the link, or go back to the outliers.</p>
      <ButtonLink href="/outliers/swing" variant="primary">
        Back to outliers
      </ButtonLink>
    </div>
  );
}
