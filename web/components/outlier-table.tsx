import Link from 'next/link';
import { IconChevronRight } from '@tabler/icons-react';
import type { RankedOutlier, Strategy } from '@/types/api';
import { Direction } from '@/components/direction';
import { plain } from '@/lib/format';

const href = (o: RankedOutlier, strategy: Strategy) => `/analysis/${o.symbol}?from=${strategy}`;

/** Ranked outliers. A table on wide screens, a stacked list on phones. Each row opens the analysis. */
export function OutlierTable({ outliers, strategy }: { outliers: RankedOutlier[]; strategy: Strategy }) {
  return (
    <>
      <table className="hidden w-full border-collapse text-left md:table">
        <caption className="sr-only">Outliers ranked by score. Select a ticker to open its analysis.</caption>
        <thead>
          <tr className="border-b border-hairline text-caption-upper uppercase text-body">
            <th scope="col" className="font-semibold w-lg py-xs pr-xs">
              #
            </th>
            <th scope="col" className="font-semibold py-xs pr-xs">
              Ticker
            </th>
            <th scope="col" className="font-semibold py-xs pr-xs text-right">
              Score
            </th>
            <th scope="col" className="font-semibold py-xs pr-xs">
              Direction
            </th>
            <th scope="col" className="font-semibold py-xs pr-xs">
              Why it stands out
            </th>
            <th scope="col" className="font-semibold w-md py-xs">
              <span className="sr-only">Open</span>
            </th>
          </tr>
        </thead>
        <tbody>
          {outliers.map((o) => (
            <tr key={o.symbol} data-reveal className="relative border-b border-hairline hover:bg-elevated focus-within:bg-elevated">
              <td className="py-xs pr-xs tabular text-body">{o.rank}</td>
              <td className="py-xs pr-xs">
                <Link
                  href={href(o, strategy)}
                  className="text-title-sm text-ink after:absolute after:inset-0 after:content-['']"
                >
                  {o.symbol}
                </Link>
              </td>
              <td className="py-xs pr-xs text-right tabular text-title-sm text-ink">{plain(o.score, 1)}</td>
              <td className="py-xs pr-xs">
                <Direction value={o.direction} />
              </td>
              <td className="py-xs pr-xs text-body-md">{o.reason}</td>
              <td className="py-xs text-body">
                <IconChevronRight aria-hidden size={18} stroke={1.5} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      <ol className="md:hidden">
        {outliers.map((o) => (
          <li key={o.symbol} data-reveal className="border-b border-hairline">
            <Link href={href(o, strategy)} className="flex items-start gap-xs py-xs">
              <span className="w-sm shrink-0 pt-xxxs tabular text-body-sm text-body">{o.rank}</span>
              <span className="flex min-w-0 flex-1 flex-col gap-xxxs">
                <span className="flex items-center justify-between gap-xs">
                  <span className="text-title-sm text-ink">{o.symbol}</span>
                  <span className="tabular text-body-sm text-body">
                    Score <span className="text-ink">{plain(o.score, 1)}</span>
                  </span>
                </span>
                <Direction value={o.direction} className="text-body-md" />
                <span className="text-body-md">{o.reason}</span>
              </span>
              <IconChevronRight aria-hidden size={18} stroke={1.5} className="mt-xxxs shrink-0 text-body" />
            </Link>
          </li>
        ))}
      </ol>
    </>
  );
}
