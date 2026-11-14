'use client';

import { useId, type ReactNode } from 'react';
import { cn } from '@/lib/utils';

const inputClass =
  'h-lg w-full rounded-none border border-hairline bg-canvas px-xs text-body-md text-ink placeholder:text-body hover:border-muted focus:border-ink';

/** Labeled text or number input with an optional hint, in the Quant picker's style. */
export function Field({
  label,
  value,
  onChange,
  type = 'text',
  hint,
  step,
  min,
  max,
  placeholder,
  className,
  suffix,
}: {
  label: string;
  value: string | number;
  onChange: (value: string) => void;
  type?: 'text' | 'number' | 'password';
  hint?: ReactNode;
  step?: number;
  min?: number;
  max?: number;
  placeholder?: string;
  className?: string;
  suffix?: string;
}) {
  const id = useId();
  return (
    <div className={cn('flex flex-col gap-xxxs', className)}>
      <label htmlFor={id} className="text-caption-upper uppercase text-body">
        {label}
        {suffix && <span className="normal-case tracking-normal"> ({suffix})</span>}
      </label>
      <input
        id={id}
        type={type}
        value={value}
        step={step}
        min={min}
        max={max}
        placeholder={placeholder}
        onChange={(e) => onChange(e.target.value)}
        autoComplete="off"
        spellCheck={false}
        aria-describedby={hint ? `${id}-hint` : undefined}
        className={cn(inputClass, 'tabular')}
      />
      {hint && (
        <p id={`${id}-hint`} className="text-caption text-body">
          {hint}
        </p>
      )}
    </div>
  );
}

/** Segmented single choice, like the Quant interval picker. */
export function Segmented<T extends string>({
  legend,
  value,
  options,
  onChange,
}: {
  legend: string;
  value: T;
  options: { value: T; label: string; note?: string }[];
  onChange: (value: T) => void;
}) {
  const name = useId();
  return (
    <fieldset className="flex flex-col gap-xxxs">
      <legend className="mb-xxxs text-caption-upper uppercase text-body">{legend}</legend>
      <div className={cn('grid border border-hairline', options.length > 2 ? 'grid-cols-2 sm:grid-cols-4' : 'grid-cols-2')}>
        {options.map((o) => (
          <label
            key={o.value}
            className={cn(
              'flex min-h-lg cursor-pointer flex-col justify-center border-hairline px-xs py-xxs transition-colors duration-300 [&:not(:first-child)]:border-l',
              value === o.value ? 'bg-livery text-on-primary' : 'text-body hover:bg-black hover:text-ink',
            )}
          >
            <input type="radio" name={name} value={o.value} checked={value === o.value} onChange={() => onChange(o.value)} className="sr-only" />
            <span className={cn('text-button uppercase', value === o.value ? 'text-on-primary' : 'text-ink')}>{o.label}</span>
            {o.note && <span className={cn('text-caption', value === o.value && 'text-on-primary')}>{o.note}</span>}
          </label>
        ))}
      </div>
    </fieldset>
  );
}

/** Label/value pair for summary rows. */
export function Stat({ label, value, tone }: { label: string; value: ReactNode; tone?: 'up' | 'down' }) {
  return (
    <div className="flex flex-col gap-xxxs">
      <span className="text-caption-upper uppercase text-body">{label}</span>
      <span className={cn('tabular text-title-md', tone === 'up' ? 'text-up' : tone === 'down' ? 'text-down' : 'text-ink')}>{value}</span>
    </div>
  );
}
