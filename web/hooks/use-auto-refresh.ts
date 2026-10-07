'use client';

import { useEffect, useRef } from 'react';

/**
 * Calls `callback` every `interval` ms while `enabled`.
 * With `pauseWhenHidden`, it skips ticks while the tab is hidden and runs once when it comes back.
 */
export function useAutoRefresh(callback: () => void, interval = 60000, enabled = true, pauseWhenHidden = false) {
  const saved = useRef(callback);
  const missed = useRef(false);

  useEffect(() => {
    saved.current = callback;
  }, [callback]);

  useEffect(() => {
    if (!enabled) return;
    const id = setInterval(() => {
      if (pauseWhenHidden && document.visibilityState === 'hidden') {
        missed.current = true;
        return;
      }
      saved.current();
    }, interval);

    const onVisible = () => {
      if (document.visibilityState === 'visible' && missed.current) {
        missed.current = false;
        saved.current();
      }
    };
    if (pauseWhenHidden) document.addEventListener('visibilitychange', onVisible);

    return () => {
      clearInterval(id);
      document.removeEventListener('visibilitychange', onVisible);
    };
  }, [interval, enabled, pauseWhenHidden]);
}
