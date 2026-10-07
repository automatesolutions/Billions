import { IconArrowDownRight, IconArrowsDiff, IconArrowUpRight, IconMinus } from '@tabler/icons-react';
import type { Direction as DirectionValue } from '@/types/api';
import { cn } from '@/lib/utils';

const META: Record<DirectionValue, { label: string; Icon: typeof IconMinus; color: string }> = {
  up: { label: 'Up', Icon: IconArrowUpRight, color: 'text-up' },
  down: { label: 'Down', Icon: IconArrowDownRight, color: 'text-down' },
  mixed: { label: 'Mixed', Icon: IconArrowsDiff, color: 'text-ink' },
  neutral: { label: 'Neutral', Icon: IconMinus, color: 'text-body' },
};

/** Direction is always icon + word + color, never color alone. */
export function Direction({ value, className }: { value: DirectionValue; className?: string }) {
  const { label, Icon, color } = META[value];
  return (
    <span className={cn('inline-flex items-center gap-xxxs text-title-sm', color, className)}>
      <Icon aria-hidden size={18} stroke={1.75} />
      {label}
    </span>
  );
}
