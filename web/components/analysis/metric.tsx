import type { ReactNode } from 'react';
import { cn } from '@/lib/utils';

/** Ferrari spec-cell: big number, uppercase caption, optional plain-language note. */
export function Metric({ label, value, note, className }: { label: string; value: ReactNode; note?: ReactNode; className?: string }) {
  return (
    <div data-reveal className={cn('flex flex-col gap-xxxs', className)}>
      <dt className="text-caption-upper uppercase text-body">{label}</dt>
      <dd className="tabular text-display-md text-ink">{value}</dd>
      {note && <dd className="text-body-sm text-body">{note}</dd>}
    </div>
  );
}
