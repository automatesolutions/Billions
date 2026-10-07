/**
 * The few token values needed outside CSS (meta theme-color, web manifest).
 * They mirror app/globals.css; __tests__/tokens.test.ts fails if they drift.
 */
export const TOKENS = {
  canvas: '#181818',
  primary: '#da291c',
} as const;
