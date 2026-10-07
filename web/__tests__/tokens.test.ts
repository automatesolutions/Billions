import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { TOKENS } from '@/lib/tokens';

const css = readFileSync(join(__dirname, '..', 'app', 'globals.css'), 'utf8');

describe('design tokens', () => {
  it('lib/tokens.ts mirrors globals.css', () => {
    expect(css).toContain(`--color-canvas: ${TOKENS.canvas};`);
    expect(css).toContain(`--color-primary: ${TOKENS.primary};`);
  });
});
