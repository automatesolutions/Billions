import { describe, it, expect } from 'vitest';

describe('Example Test Suite', () => {
  it('should pass a basic test', () => {
    expect(1 + 1).toBe(2);
  });

  it('should test string equality', () => {
    expect('BILLIONS').toBe('BILLIONS');
  });

  it('should test array includes', () => {
    const strategies = ['scalp', 'swing', 'longterm'];
    expect(strategies).toContain('swing');
  });
});

