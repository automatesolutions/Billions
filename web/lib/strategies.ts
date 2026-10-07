import type { Strategy } from '@/types/api';

/** Copy for each strategy. Windows match api/services/outlier_engine.py. */
export const STRATEGY_COPY: Record<Strategy, { name: string; horizon: string; long: string; short: string; summary: string }> = {
  scalp: {
    name: 'Scalp',
    horizon: 'Days to weeks',
    long: '1 month',
    short: '1 week',
    summary: 'Stocks with an unusual move over the last week or month.',
  },
  swing: {
    name: 'Swing',
    horizon: 'Weeks to months',
    long: '3 months',
    short: '1 month',
    summary: 'Stocks with an unusual move over the last month or quarter.',
  },
  longterm: {
    name: 'Long-term',
    horizon: 'Months to a year',
    long: '1 year',
    short: '6 months',
    summary: 'Stocks with an unusual move over the last six months or year.',
  },
};
