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
    <section id={id} aria-labelledby={title && id ? `${id}-title` : undefined} className={cn('group/panel relative border border-hairline transition-colors duration-500 hover:border-steel', className)}>
      {/* Racing stripe on the top edge: draws in when the panel is in use. */}
      <span
        aria-hidden
        className="absolute -top-px left-0 h-[2px] w-xl origin-left scale-x-50 bg-primary transition-transform duration-700 ease-out group-hover/panel:scale-x-100"
      />
      {(title || action) && (
        <header className="flex flex-wrap items-end justify-between gap-xs border-b border-hairline px-xs py-xs sm:px-sm">
          <div className="flex flex-col gap-xxxs">
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
