'use client';

import { useEffect } from 'react';

/** Registers /sw.js in production so the app can show an offline page. */
export function ServiceWorker() {
  useEffect(() => {
    if (process.env.NODE_ENV === 'production' && 'serviceWorker' in navigator) {
      navigator.serviceWorker.register('/sw.js').catch(() => {
        // Offline support is optional; the app works without it.
      });
    }
  }, []);
  return null;
}
