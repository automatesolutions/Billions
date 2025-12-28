'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";

export function AnalyzeStockSearch() {
  const [ticker, setTicker] = useState('');
  const router = useRouter();

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    if (ticker.trim()) {
      router.push(`/analyze/${ticker.toUpperCase()}`);
    }
  };

  return (
    <Card className="bg-background border-0 shadow-none">
      <CardHeader>
        <CardTitle className="text-lg text-white">📊 Analyze Stock</CardTitle>
        <CardDescription className="text-gray-500">
          Search and analyze with ML predictions
        </CardDescription>
      </CardHeader>
      <CardContent>
        <form onSubmit={handleSearch} className="flex gap-2">
          <Input
            type="text"
            placeholder="Enter ticker (e.g., TSLA, AAPL)..."
            value={ticker}
            onChange={(e) => setTicker(e.target.value)}
            className="flex-1 bg-gray-900 border-gray-700 text-white placeholder:text-gray-500"
          />
          <Button type="submit" className="bg-yellow-500 hover:bg-yellow-600 text-black">
            Analyze
          </Button>
        </form>
        <p className="text-sm text-gray-500 mt-2">
          ML predictions, technical analysis
        </p>
      </CardContent>
    </Card>
  );
}

