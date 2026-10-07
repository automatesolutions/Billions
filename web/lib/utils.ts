import { clsx, type ClassValue } from 'clsx';

/** Joins class names. No merging: our token names are custom, so tailwind-merge would guess wrong. */
export function cn(...inputs: ClassValue[]) {
  return clsx(inputs);
}
