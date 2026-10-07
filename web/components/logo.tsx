import { cn } from '@/lib/utils';

/**
 * The BILLIONS mark: three rising bars and one point that has broken away from the pack.
 * It is the product in one picture. Rosso Corsa plate, sharp corners (DESIGN.md: the Cavallino slot).
 */
export function LogoMark({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 32 32" aria-hidden className={cn('size-md shrink-0', className)}>
      <rect width="32" height="32" className="fill-primary" />
      <rect x="6" y="18" width="4" height="8" className="fill-on-primary" opacity="0.55" />
      <rect x="12" y="14" width="4" height="12" className="fill-on-primary" opacity="0.78" />
      <rect x="18" y="10" width="4" height="16" className="fill-on-primary" />
      <rect x="23" y="5" width="4" height="4" className="fill-on-primary" />
    </svg>
  );
}

/** Mark plus wordmark. Wide tracking, medium weight: confident, never loud. */
export function Logo({ className, size = 'md' }: { className?: string; size?: 'md' | 'lg' }) {
  return (
    <span className={cn('inline-flex items-center gap-xs text-ink', className)}>
      <LogoMark className={size === 'lg' ? 'size-lg' : undefined} />
      <span
        className={cn(
          'font-semibold uppercase tracking-brand',
          size === 'lg' ? 'text-display-md' : 'text-title-sm',
        )}
      >
        Billions
      </span>
    </span>
  );
}
