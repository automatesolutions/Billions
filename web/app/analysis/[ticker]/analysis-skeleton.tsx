import { Skeleton } from '@/components/ui/skeleton';

export function AnalysisSkeleton({ ticker }: { ticker: string }) {
  return (
    <div aria-busy="true" className="flex flex-col gap-md">
      <div className="flex flex-col gap-xs">
        <p className="text-caption-upper uppercase text-body">Stock analysis</p>
        <h1 className="text-display-lg sm:text-display-xl">{ticker}</h1>
        <p role="status" className="text-title-sm text-body">
          Testing the models on two years of prices. The first look at a stock takes a few seconds.
        </p>
      </div>
      <Skeleton className="h-xxl w-full" />
      <Skeleton className="h-chart w-full" />
      <div className="grid grid-cols-2 gap-xs sm:grid-cols-4">
        {[0, 1, 2, 3].map((i) => (
          <Skeleton key={i} className="h-xl" />
        ))}
      </div>
    </div>
  );
}
