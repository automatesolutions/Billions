import Link from 'next/link';
import type { ComponentProps } from 'react';
import { cn } from '@/lib/utils';

type Variant = 'primary' | 'outline' | 'text';

const base =
  'inline-flex min-h-lg items-center justify-center gap-xxs rounded-none text-button uppercase select-none ' +
  'transition-colors duration-300 ease-out disabled:cursor-not-allowed disabled:opacity-50';

const variants: Record<Variant, string> = {
  // Rosso Corsa, with the brand gradient and a light sweep on hover (DESIGN.md "Decorative Depth").
  primary: 'sheen bg-primary px-md text-on-primary hover:bg-livery active:bg-primary-active active:bg-none',
  // White outline fills white on hover, like Ferrari's secondary CTA.
  outline: 'border border-ink px-md text-ink hover:bg-ink hover:text-canvas active:bg-body',
  text: 'px-xxs text-ink underline-offset-4 hover:underline',
};

export function buttonClass(variant: Variant = 'outline', className?: string) {
  return cn(base, variants[variant], className);
}

export function Button({ variant, className, ...props }: ComponentProps<'button'> & { variant?: Variant }) {
  return <button type="button" className={buttonClass(variant, className)} {...props} />;
}

export function ButtonLink({ variant, className, ...props }: ComponentProps<typeof Link> & { variant?: Variant }) {
  return <Link className={buttonClass(variant, className)} {...props} />;
}
