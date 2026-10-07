'use client';

import { useState } from 'react';
import Link from 'next/link';
import { Button } from "@/components/ui/button";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { usePerformanceMetrics } from "@/hooks/use-performance-metrics";

const STRATEGIES = ['scalp', 'swing', 'longterm'] as const;

export function ClientOutliersPage() {
  const [strategy, setStrategy] = useState<string>('swing');
  const { data, loading, error } = usePerformanceMetrics(strategy, true);
  const outliers = (data?.metrics ?? []).filter((m) => m.is_outlier);

  return (
    <>
      <div className="flex gap-2">
        {STRATEGIES.map((s) => (
          <Button key={s} variant={s === strategy ? 'default' : 'outline'} size="sm" onClick={() => setStrategy(s)}>
            {s}
          </Button>
        ))}
      </div>

      {loading ? (
        <p>Loading…</p>
      ) : error ? (
        <p className="text-destructive">Could not load outliers: {error}</p>
      ) : outliers.length === 0 ? (
        <p className="text-muted-foreground">No outliers for this strategy yet.</p>
      ) : (
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Symbol</TableHead>
              <TableHead className="text-right">Long window %</TableHead>
              <TableHead className="text-right">Short window %</TableHead>
              <TableHead className="text-right">z (long)</TableHead>
              <TableHead className="text-right">z (short)</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {outliers.map((o) => (
              <TableRow key={o.symbol}>
                <TableCell>
                  <Link href={`/analysis/${o.symbol}`}>{o.symbol}</Link>
                </TableCell>
                <TableCell className="text-right">{o.metric_x?.toFixed(2)}</TableCell>
                <TableCell className="text-right">{o.metric_y?.toFixed(2)}</TableCell>
                <TableCell className="text-right">{o.z_x?.toFixed(2)}</TableCell>
                <TableCell className="text-right">{o.z_y?.toFixed(2)}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      )}
    </>
  );
}
