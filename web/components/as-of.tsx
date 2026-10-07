import type { MarketState, RefreshStatus } from '@/types/api';
import { Badge } from '@/components/ui/badge';
import { dateTimeET, weekdayTimeET } from '@/lib/format';

const CLOSED_REASON: Record<MarketState['state'], string> = {
  open: 'Market open',
  pre_market: 'Market closed · pre-market',
  after_hours: 'Market closed · after hours',
  weekend: 'Market closed · weekend',
  holiday: 'Market closed · holiday',
};

/** "Data as of" line plus market state. Times are New York time. */
export function AsOf({
  computedAt,
  market,
  refresh,
}: {
  computedAt: string | null;
  market: MarketState;
  refresh?: RefreshStatus;
}) {
  return (
    <div className="flex flex-wrap items-center gap-xs text-body-sm">
      <Badge dot={market.is_open ? 'up' : 'muted'}>{CLOSED_REASON[market.state]}</Badge>
      {computedAt && (
        <p>
          Data as of <time dateTime={computedAt} className="text-ink">{dateTimeET(computedAt)}</time>
          {!market.is_open && market.next_open && <> · Next open {weekdayTimeET(market.next_open)}</>}
        </p>
      )}
      {refresh?.is_running && <Badge>Updating</Badge>}
      {refresh?.last_error && !refresh.is_running && (
        <p className="text-down">The last update failed. These numbers may be out of date.</p>
      )}
    </div>
  );
}
