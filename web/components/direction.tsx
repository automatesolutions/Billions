import { IconArrowDownRight, IconArrowsDiff, IconArrowUpRight, IconMinus } from '@tabler/icons-react';
import type { Direction as DirectionValue } from '@/types/api';
import { cn } from '@/lib/utils';

const META: Record<DirectionValue, { label: string; Icon: typeof IconMinus; color: string }> = {
  up: { label: 'Up', Icon: IconArrowUpRight, color: 'text-up' },
  down: { label: 'Down', Icon: IconArrowDownRight, color: 'text-down' },
  mixed: { label: 'Mixed', Icon: IconArrowsDiff, color: 'text-ink' },
  neutral: { label: 'Neutral', Icon: IconMinus, color: 'text-body' },
};

/**
 * Direction is always icon + word + color, never color alone.
 * `muted` drops the color (e.g. when the edge behind the direction is not significant).
 */
export function Direction({ value, className, muted = false }: { value: DirectionValue; className?: string; muted?: boolean }) {
  const { label, Icon, color } = META[value];
  return (
    <span className={cn('inline-flex items-center gap-xxxs text-title-sm', muted ? 'text-ink' : color, className)}>
      <Icon aria-hidden size={18} stroke={1.75} />
      {label}
    </span>
  );
}
