import { cn } from '@/lib/utils';
import { plain } from '@/lib/format';

/**
 * Bounded signal strength from −1 (down) to +1 (up). The fill grows from the center.
 * Text carries the meaning; the bar is a visual aid.
 */
export function StrengthGauge({
  strength,
  direction,
  muted = false,
}: {
  strength: number;
  direction: 'up' | 'down' | 'neutral';
  /** Gray fill when the edge is not significant, so color doesn't overstate the signal. */
  muted?: boolean;
}) {
  const clamped = Math.max(-1, Math.min(1, strength));
  const half = Math.abs(clamped) * 50;
  return (
    <div>
      <div
        role="meter"
        aria-label="Signal strength"
        aria-valuemin={-1}
        aria-valuemax={1}
        aria-valuenow={Number(clamped.toFixed(2))}
        aria-valuetext={`${plain(clamped, 2)} on a scale from minus 1 (down) to plus 1 (up)`}
        className="relative h-xxs w-full bg-elevated"
      >
        <span aria-hidden className="absolute inset-y-0 left-1/2 w-px bg-body" />
        <span
          aria-hidden
          className={cn(
            'absolute inset-y-0',
            muted || direction === 'neutral' ? 'bg-body' : direction === 'up' ? 'bg-up' : 'bg-down',
          )}
          style={clamped >= 0 ? { left: '50%', width: `${half}%` } : { right: '50%', width: `${half}%` }}
        />
      </div>
      <div className="mt-xxs flex justify-between text-caption text-body" aria-hidden>
        <span>Strong down</span>
        <span>Neutral</span>
        <span>Strong up</span>
      </div>
    </div>
  );
}
