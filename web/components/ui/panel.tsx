import type { ReactNode } from 'react';
import { cn } from '@/lib/utils';

/** Canvas surface with a hairline border (DESIGN.md: cards are hairline, not filled). */
export function Panel({
  title,
  label,
  action,
  children,
  className,
  id,
}: {
  title?: string;
  label?: string;
  action?: ReactNode;
  children: ReactNode;
  className?: string;
  id?: string;
}) {
  return (
    <section id={id} aria-labelledby={title && id ? `${id}-title` : undefined} className={cn('border border-hairline', className)}>
      {(title || action) && (
        <header className="flex flex-wrap items-end justify-between gap-xs border-b border-hairline px-xs py-xs sm:px-sm">
          <div>
            {label && <p className="text-caption-upper uppercase text-body">{label}</p>}
            {title && (
              <h2 id={id ? `${id}-title` : undefined} className="text-title-md text-ink">
                {title}
              </h2>
            )}
          </div>
          {action}
        </header>
      )}
      <div className="px-xs py-sm sm:px-sm">{children}</div>
    </section>
  );
}
