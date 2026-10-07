'use client';

import { useCallback, useEffect, useRef, useState, type ReactNode } from 'react';
import dynamic from 'next/dynamic';
import { IconRefresh } from '@tabler/icons-react';
import { getOutliers } from '@/lib/api';
import type { OutliersResponse, Strategy } from '@/types/api';
import { STRATEGY_COPY } from '@/lib/strategies';
import { compactDollars } from '@/lib/format';
import { useAutoRefresh } from '@/hooks/use-auto-refresh';
import { AsOf } from '@/components/as-of';
import { CountUp, useReveal } from '@/components/motion';
import { OutlierTable } from '@/components/outlier-table';
import { StrategySwitcher } from '@/components/strategy-switcher';
import { Button } from '@/components/ui/button';
import { PageHead } from '@/components/ui/page-head';
import { Panel } from '@/components/ui/panel';
import { Skeleton } from '@/components/ui/skeleton';
import { StateMessage } from '@/components/ui/state';

const OutlierScatter = dynamic(() => import('@/components/charts/outlier-scatter').then((m) => m.OutlierScatter), {
  ssr: false,
  loading: () => <Skeleton className="h-chart w-full" />,
});

const REFRESH_MS = 5 * 60 * 1000;
const integer = (n: number) => Math.round(n).toLocaleString('en-US');

export function OutliersView({ strategy, initial }: { strategy: Strategy; initial: OutliersResponse | null }) {
  const [data, setData] = useState<OutliersResponse | null>(initial);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(!initial);
  const rootRef = useRef<HTMLDivElement>(null);
  const copy = STRATEGY_COPY[strategy];

  const load = useCallback(async () => {
    setLoading(true);
    try {
      setData(await getOutliers(strategy, { cache: 'no-store' }));
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Something went wrong.');
    } finally {
      setLoading(false);
    }
  }, [strategy]);

  useEffect(() => {
    if (!initial) load();
  }, [initial, load]);

  useAutoRefresh(load, REFRESH_MS, true, true);
  useReveal(rootRef, !!data);

  const hasData = !!data?.as_of;

  return (
    <div ref={rootRef} className="flex flex-col gap-md">
      <PageHead eyebrow={`Outliers · ${copy.horizon}`} title={`${copy.name} outliers`}>
        <p className="max-w-prose text-title-sm text-body">{copy.summary}</p>
      </PageHead>

      <StrategySwitcher current={strategy} />

      <div className="flex flex-wrap items-center justify-between gap-xs">
        {data ? <AsOf computedAt={data.computed_at} market={data.market} refresh={data.refresh} /> : <span />}
        <Button onClick={load} disabled={loading}>
          <IconRefresh aria-hidden size={18} stroke={1.75} className={loading ? 'animate-spin' : undefined} />
          {loading ? 'Loading' : 'Refresh'}
        </Button>
      </div>

      {error && !data && (
        <StateMessage
          kind="error"
          title="Can't load outliers"
          action={
            <Button variant="primary" onClick={load}>
              Try again
            </Button>
          }
        >
          {error} Check your connection, then try again.
        </StateMessage>
      )}
      {error && data && (
        <p role="status" className="text-body-sm text-down">
          Refresh failed. Showing the last data loaded.
        </p>
      )}

      {!data && !error && <LoadingState />}

      {data && !hasData && (
        <StateMessage kind="empty" title="No data yet">
          The first scan covers about 4,000 NASDAQ stocks and takes a few minutes after the service starts. This page checks
          again every 5 minutes.
        </StateMessage>
      )}

      {data && hasData && (
        <>
          <dl className="grid grid-cols-2 gap-xs border-y border-hairline py-sm sm:grid-cols-3">
            <Stat label="Outliers">
              <CountUp value={data.outlier_count} format={integer} />
            </Stat>
            <Stat label="Stocks scanned">
              <CountUp value={data.universe_count} format={integer} />
            </Stat>
            <Stat label="Liquidity floor" className="col-span-2 sm:col-span-1">
              {compactDollars(data.min_dollar_volume)}
              <span className="text-title-sm text-body"> a day</span>
            </Stat>
          </dl>

          <Panel id="scatter" label="Every stock scanned" title={`${copy.long} vs ${copy.short} return`}>
            <OutlierScatter
              points={data.points}
              outliers={data.outliers}
              xLabel={copy.long}
              yLabel={copy.short}
              zThreshold={data.z_threshold}
            />
          </Panel>

          <Panel id="ranked" label={`${data.outlier_count} stocks`} title="Ranked outliers">
            {data.outliers.length ? (
              <OutlierTable outliers={data.outliers} strategy={strategy} />
            ) : (
              <StateMessage kind="empty" title="No outliers right now">
                No stock is more than {data.z_threshold} standard deviations from the group on either window. Try another
                strategy.
              </StateMessage>
            )}
          </Panel>
        </>
      )}
    </div>
  );
}

function Stat({ label, children, className }: { label: string; children: ReactNode; className?: string }) {
  return (
    <div data-reveal className={className}>
      <dt className="text-caption-upper uppercase text-body">{label}</dt>
      <dd className="tabular text-number-lg text-ink lg:text-number-xl lg:font-medium">{children}</dd>
    </div>
  );
}

function LoadingState() {
  return (
    <div aria-busy="true" aria-label="Loading outliers" className="flex flex-col gap-md">
      <Skeleton className="h-xxl w-full" />
      <Skeleton className="h-chart w-full" />
      {[0, 1, 2, 3].map((i) => (
        <Skeleton key={i} className="h-lg w-full" />
      ))}
    </div>
  );
}
