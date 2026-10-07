import Link from 'next/link';
import { STRATEGIES, type Strategy } from '@/types/api';
import { STRATEGY_COPY } from '@/lib/strategies';
import { cn } from '@/lib/utils';

export function StrategySwitcher({ current }: { current: Strategy }) {
  return (
    <nav aria-label="Strategy" className="grid grid-cols-3 border border-hairline">
      {STRATEGIES.map((s) => {
        const active = s === current;
        const copy = STRATEGY_COPY[s];
        return (
          <Link
            key={s}
            href={`/outliers/${s}`}
            aria-current={active ? 'page' : undefined}
            scroll={false}
            className={cn(
              'flex min-h-lg flex-col justify-center border-l border-hairline px-xxs py-xxs first:border-l-0 sm:px-sm',
              active ? 'bg-primary text-on-primary' : 'text-body hover:text-ink',
            )}
          >
            <span className={cn('whitespace-nowrap text-nav uppercase sm:text-button', active ? 'text-on-primary' : 'text-ink')}>{copy.name}</span>
            <span className="hidden text-caption sm:block">
              {copy.short} · {copy.long}
            </span>
          </Link>
        );
      })}
    </nav>
  );
}
