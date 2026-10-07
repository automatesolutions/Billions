import type { ReactNode } from 'react';
import { cn } from '@/lib/utils';

/** Ferrari badge-pill: the only rounded shape in the system. */
export function Badge({ children, className, dot }: { children: ReactNode; className?: string; dot?: 'up' | 'down' | 'muted' }) {
  return (
    <span
      className={cn(
        'inline-flex items-center gap-xxs rounded-full bg-elevated px-xs py-xxxs text-caption-upper uppercase text-ink',
        className,
      )}
    >
      {dot && (
        <span
          aria-hidden
          className={cn('size-xxs rounded-full', dot === 'up' && 'bg-up', dot === 'down' && 'bg-down', dot === 'muted' && 'bg-body')}
        />
      )}
      {children}
    </span>
  );
}
