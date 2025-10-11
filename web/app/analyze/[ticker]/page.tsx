import { auth } from "@/auth";
import { redirect } from "next/navigation";
import Image from "next/image";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";

interface PageProps {
  params: Promise<{ ticker: string }>;
}

export default async function AnalyzePage({ params }: PageProps) {
  const session = await auth();
  const resolvedParams = await params;

  if (!session?.user) {
    redirect("/login");
  }

  const ticker = resolvedParams.ticker.toUpperCase();

  return (
    <div className="min-h-screen bg-background p-8">
      <div className="max-w-7xl mx-auto space-y-8">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Image
              src="/logo.png"
              alt="BILLIONS Logo"
              width={60}
              height={60}
            />
            <div>
              <h1 className="text-3xl font-bold">{ticker} Analysis</h1>
              <p className="text-muted-foreground">
                Technical analysis and ML predictions
              </p>
            </div>
          </div>
          <Button variant="outline">Add to Watchlist</Button>
        </div>

        {/* Stock Info */}
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle>{ticker}</CardTitle>
              <Badge variant="secondary">Loading...</Badge>
            </div>
            <CardDescription>
              Stock information from /api/v1/predictions/info/{ticker}
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
              <div>
                <p className="text-muted-foreground">Current Price</p>
                <p className="text-2xl font-bold">$---.--</p>
              </div>
              <div>
                <p className="text-muted-foreground">Market Cap</p>
                <p className="text-xl font-semibold">---B</p>
              </div>
              <div>
                <p className="text-muted-foreground">Volume</p>
                <p className="text-xl font-semibold">---M</p>
              </div>
              <div>
                <p className="text-muted-foreground">Sector</p>
                <p className="text-xl font-semibold">---</p>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Price Chart Placeholder */}
        <Card>
          <CardHeader>
            <CardTitle>Price Chart & 30-Day Prediction</CardTitle>
            <CardDescription>
              Historical prices with ML forecast overlay
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="aspect-[16/9] bg-muted/20 rounded-lg flex items-center justify-center border-2 border-dashed">
              <div className="text-center text-muted-foreground">
                <p className="text-lg font-semibold mb-2">Candlestick Chart</p>
                <p className="text-sm">With 30-day prediction overlay</p>
                <p className="text-xs mt-2">Coming in Phase 5.4 - Data Visualization</p>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* 30-Day Forecast */}
        <Card>
          <CardHeader>
            <CardTitle>30-Day ML Forecast</CardTitle>
            <CardDescription>
              LSTM-based prediction from /api/v1/predictions/{ticker}
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="p-4 border rounded-lg">
                <p className="text-sm text-muted-foreground">Current Price</p>
                <p className="text-2xl font-bold">$---.--</p>
              </div>
              <div className="p-4 border rounded-lg">
                <p className="text-sm text-muted-foreground">30-Day Target</p>
                <p className="text-2xl font-bold text-green-600">$---.--</p>
                <p className="text-xs text-muted-foreground">+--% expected</p>
              </div>
              <div className="p-4 border rounded-lg">
                <p className="text-sm text-muted-foreground">Confidence</p>
                <p className="text-2xl font-bold">--%</p>
                <p className="text-xs text-muted-foreground">Model accuracy</p>
              </div>
            </div>

            <div className="text-sm text-muted-foreground">
              Prediction data will be fetched from API using TanStack Query
            </div>
          </CardContent>
        </Card>

        {/* Technical Indicators */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Technical Indicators</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-muted-foreground">RSI (14)</span>
                <span className="font-semibold">--</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">MACD</span>
                <span className="font-semibold">--</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Bollinger Bands</span>
                <span className="font-semibold">--</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Volume Ratio</span>
                <span className="font-semibold">--</span>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Market Regime</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Trend</span>
                <Badge variant="outline">--</Badge>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Volatility</span>
                <Badge variant="outline">--</Badge>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Momentum</span>
                <Badge variant="outline">--</Badge>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}

