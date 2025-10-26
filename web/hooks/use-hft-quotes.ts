'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { api } from '@/lib/api';

export interface Quote {
  symbol: string;
  bid?: number;
  ask?: number;
  last?: number;
  time?: number;
  isMockData?: boolean;
}

export function useHftQuotes(selectedSymbols: string[], intervalMs: number = 1000) {
  const [quotes, setQuotes] = useState<Record<string, Quote>>({});
  const timerRef = useRef<NodeJS.Timeout | null>(null);

  const fetchQuotes = useCallback(async () => {
    if (!selectedSymbols.length) return;
    try {
      const data = await api.getBulkQuotes(selectedSymbols);
      // Expecting array; normalize into record
      const next: Record<string, Quote> = { ...quotes };
      (data || []).forEach((q: any) => {
        const symbol = q.symbol || q.ticker || '';
        if (!symbol) return;
        next[symbol] = {
          symbol,
          bid: q.bid ?? q.best_bid ?? q.b ?? q.quote?.bid ?? undefined,
          ask: q.ask ?? q.best_ask ?? q.a ?? q.quote?.ask ?? undefined,
          last: q.last_price ?? q.p ?? q.last ?? undefined,
          time: q.timestamp ?? q.t ?? Date.now(),
          isMockData: q.status === 'mock_data'
        };
      });
      setQuotes(next);
    } catch (err) {
      // If API fails, use mock data for demonstration
      console.warn('HFT quotes API error, using mock data:', err);
      const mockData: Record<string, Quote> = { ...quotes };
      selectedSymbols.forEach(symbol => {
        if (!mockData[symbol]) {
          // Generate realistic mock prices
          const basePrice = 100 + (symbol.charCodeAt(0) % 26) * 10;
          const variation = (Math.random() - 0.5) * 2; // ±1 dollar variation
          const currentPrice = basePrice + variation;
          const spread = 0.01 + Math.random() * 0.05; // 1-6 cent spread
          
          mockData[symbol] = {
            symbol,
            bid: currentPrice - spread / 2,
            ask: currentPrice + spread / 2,
            last: currentPrice,
            time: Date.now(),
            isMockData: true
          };
        }
      });
      setQuotes(mockData);
    }
  }, [selectedSymbols, quotes]);

  useEffect(() => {
    // Immediate fetch
    fetchQuotes();
    // Start poller
    if (timerRef.current) clearInterval(timerRef.current as unknown as number);
    timerRef.current = setInterval(fetchQuotes, intervalMs);
    return () => {
      if (timerRef.current) clearInterval(timerRef.current as unknown as number);
    };
  }, [fetchQuotes, intervalMs]);

  return quotes;
}


