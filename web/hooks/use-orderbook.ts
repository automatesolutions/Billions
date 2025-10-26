'use client';

import { useCallback, useEffect, useRef, useState } from 'react';

export interface OrderBookLevel {
  price: number;
  size: number;
  total?: number;
}

export interface OrderBook {
  symbol: string;
  bids: OrderBookLevel[];
  asks: OrderBookLevel[];
  lastUpdate: number;
  isMockData?: boolean;
}

export function useOrderBook(symbol: string, isActive: boolean = true) {
  const [orderBook, setOrderBook] = useState<OrderBook | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const lastSymbolRef = useRef<string>('');

  const generateMockOrderBook = useCallback((sym: string): OrderBook => {
    const basePrice = 100 + (sym.charCodeAt(0) % 26) * 10;
    const variation = (Math.random() - 0.5) * 2;
    const midPrice = basePrice + variation;
    
    const bids: OrderBookLevel[] = [];
    const asks: OrderBookLevel[] = [];
    
    // Generate 10 levels each for bids and asks with more realistic distribution
    for (let i = 0; i < 10; i++) {
      const bidPrice = midPrice - (i + 1) * 0.01;
      const askPrice = midPrice + (i + 1) * 0.01;
      
      // Create more realistic size distribution (larger sizes near the spread)
      const bidSizeMultiplier = Math.max(0.1, 1 - (i * 0.1));
      const askSizeMultiplier = Math.max(0.1, 1 - (i * 0.1));
      
      const bidSize = (Math.random() * 200 + 50) * bidSizeMultiplier;
      const askSize = (Math.random() * 200 + 50) * askSizeMultiplier;
      
      bids.push({
        price: parseFloat(bidPrice.toFixed(2)),
        size: parseFloat(bidSize.toFixed(2))
      });
      
      asks.push({
        price: parseFloat(askPrice.toFixed(2)),
        size: parseFloat(askSize.toFixed(2))
      });
    }
    
    // Calculate cumulative totals
    let bidTotal = 0;
    let askTotal = 0;
    
    bids.forEach(level => {
      bidTotal += level.size;
      level.total = parseFloat(bidTotal.toFixed(2));
    });
    
    asks.forEach(level => {
      askTotal += level.size;
      level.total = parseFloat(askTotal.toFixed(2));
    });
    
    return {
      symbol: sym,
      bids: bids.sort((a, b) => b.price - a.price), // Highest bid first
      asks: asks.sort((a, b) => a.price - b.price), // Lowest ask first
      lastUpdate: Date.now(),
      isMockData: true
    };
  }, []);

  const connectWebSocket = useCallback(() => {
    if (!symbol || !isActive) return;
    
    // For now, use mock data since Polygon WebSocket requires authentication
    // In production, this would connect to: wss://socket.polygon.io/stocks
    console.log(`Connecting to order book for ${symbol}`);
    setIsConnected(true);
    
    // Generate initial mock order book
    const mockOrderBook = generateMockOrderBook(symbol);
    setOrderBook(mockOrderBook);
    
    // Update order book every second with realistic changes
    const updateInterval = setInterval(() => {
      if (lastSymbolRef.current !== symbol) {
        clearInterval(updateInterval);
        return;
      }
      
      const currentOrderBook = generateMockOrderBook(symbol);
      setOrderBook(currentOrderBook);
    }, 1000);
    
    // Cleanup interval when component unmounts or symbol changes
    return () => clearInterval(updateInterval);
  }, [symbol, isActive, generateMockOrderBook]);

  const disconnectWebSocket = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    setIsConnected(false);
    setOrderBook(null);
  }, []);

  useEffect(() => {
    if (symbol && isActive) {
      lastSymbolRef.current = symbol;
      disconnectWebSocket();
      const cleanup = connectWebSocket();
      return cleanup;
    } else {
      disconnectWebSocket();
    }
  }, [symbol, isActive, connectWebSocket, disconnectWebSocket]);

  useEffect(() => {
    return () => {
      disconnectWebSocket();
    };
  }, [disconnectWebSocket]);

  return {
    orderBook,
    isConnected,
    isMockData: orderBook?.isMockData || false
  };
}
