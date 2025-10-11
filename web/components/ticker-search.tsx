'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';

export function TickerSearch() {
  const [ticker, setTicker] = useState('');
  const router = useRouter();

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    if (ticker.trim()) {
      router.push(`/analyze/${ticker.toUpperCase()}`);
    }
  };

  return (
    <form onSubmit={handleSearch} className="flex gap-2">
      <Input
        type="text"
        placeholder="Enter ticker (e.g., TSLA, AAPL)..."
        value={ticker}
        onChange={(e) => setTicker(e.target.value)}
        className="max-w-xs"
      />
      <Button type="submit">Analyze</Button>
    </form>
  );
}

