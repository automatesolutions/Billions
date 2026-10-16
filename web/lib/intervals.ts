import type { QuantInterval } from '@/types/api';

export const INTERVAL_COPY: Record<QuantInterval, { name: string; bar: string; note: string }> = {
  '1m': { name: '1 min', bar: 'minute', note: 'Last 7 sessions' },
  '1h': { name: '1 hour', bar: 'hour', note: 'Last 2 years' },
  '4h': { name: '4 hours', bar: '4-hour bar', note: 'Last 2 years' },
  '1d': { name: 'Daily', bar: 'day', note: 'Last 2 years' },
};
