/** Number and time formatting. All market times are shown in New York time (ET). */

const MINUS = '\u2212';

export function signed(value: number, digits = 1, suffix = ''): string {
  if (!Number.isFinite(value)) return '—';
  const abs = Math.abs(value).toFixed(digits);
  if (Number(abs) === 0) return `0${digits ? '.' + '0'.repeat(digits) : ''}${suffix}`;
  return `${value > 0 ? '+' : MINUS}${abs}${suffix}`;
}

export const pct = (value: number, digits = 1) => signed(value, digits, '%');

export function plain(value: number, digits = 2): string {
  if (!Number.isFinite(value)) return '—';
  return value < 0 ? `${MINUS}${Math.abs(value).toFixed(digits)}` : value.toFixed(digits);
}

export function money(value: number): string {
  if (!Number.isFinite(value)) return '—';
  return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 2 }).format(value);
}

export function compactDollars(value: number): string {
  return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', notation: 'compact' }).format(value);
}

const ET = 'America/New_York';

export function dateET(iso: string): string {
  // Plain dates (YYYY-MM-DD) are calendar days, not instants: format them without a time zone shift.
  const date = /^\d{4}-\d{2}-\d{2}$/.test(iso) ? new Date(`${iso}T12:00:00Z`) : new Date(iso);
  return new Intl.DateTimeFormat('en-US', { month: 'short', day: 'numeric', year: 'numeric', timeZone: ET }).format(date);
}

export function dateTimeET(iso: string): string {
  const text = new Intl.DateTimeFormat('en-US', {
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
    timeZone: ET,
  }).format(new Date(iso));
  return `${text} ET`;
}

export function weekdayTimeET(iso: string): string {
  const text = new Intl.DateTimeFormat('en-US', { weekday: 'short', hour: 'numeric', minute: '2-digit', timeZone: ET }).format(
    new Date(iso),
  );
  return `${text} ET`;
}

/** A log return or decimal as basis points, signed: 0.0019 -> "+19.0 bps". */
export const bps = (decimal: number, digits = 1) => signed(decimal * 1e4, digits, ' bps');

/** A decimal as a percent: 0.125 -> "12.5%" (unsigned) */
export function percent(decimal: number | null | undefined, digits = 1): string {
  if (decimal === null || decimal === undefined || !Number.isFinite(decimal)) return '—';
  const v = decimal * 100;
  return v < 0 ? `${MINUS}${Math.abs(v).toFixed(digits)}%` : `${v.toFixed(digits)}%`;
}

export function ratio(value: number | null | undefined, digits = 2): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return '—';
  return plain(value, digits);
}
