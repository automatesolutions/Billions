import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook } from '@testing-library/react';
import { useAutoRefresh } from '@/hooks/use-auto-refresh';

describe('useAutoRefresh', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('calls callback at specified interval', () => {
    const callback = vi.fn();
    renderHook(() => useAutoRefresh(callback, 1000, true));

    expect(callback).not.toHaveBeenCalled();

    vi.advanceTimersByTime(1000);
    expect(callback).toHaveBeenCalledTimes(1);

    vi.advanceTimersByTime(1000);
    expect(callback).toHaveBeenCalledTimes(2);
  });

  it('does not call callback when disabled', () => {
    const callback = vi.fn();
    renderHook(() => useAutoRefresh(callback, 1000, false));

    vi.advanceTimersByTime(5000);
    expect(callback).not.toHaveBeenCalled();
  });

  it('cleans up interval on unmount', () => {
    const callback = vi.fn();
    const { unmount } = renderHook(() => useAutoRefresh(callback, 1000, true));

    vi.advanceTimersByTime(1000);
    expect(callback).toHaveBeenCalledTimes(1);

    unmount();
    vi.advanceTimersByTime(2000);
    // Should still be 1, not 3
    expect(callback).toHaveBeenCalledTimes(1);
  });
});

