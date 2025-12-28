import { auth } from "@/auth";
import { redirect } from "next/navigation";
import Image from "next/image";
import Link from "next/link";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ClientAnalyzePage } from "./client-page";
import { NewsSection } from "./news-section";
import { TechnicalIndicators } from "./technical-indicators";
import { FairValueCard } from "@/components/fair-value-card";

interface PageProps {
  params: Promise<{ ticker: string }>;
}

export default async function AnalyzePage({ params }: PageProps) {
  const session = await auth();
  const resolvedParams = await params;

  // Allow demo access without authentication
  // if (!session?.user) {
  //   redirect("/login");
  // }

  const ticker = resolvedParams.ticker.toUpperCase();

  return (
    <div className="min-h-screen bg-background p-8">
      <div className="max-w-7xl mx-auto space-y-8">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Image
              src="/logo1.png"
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
          <div className="flex gap-2">
            <Link href="/dashboard">
              <Button variant="outline" size="sm">Back to Dashboard</Button>
            </Link>
            <Button variant="outline" size="sm">Add to Watchlist</Button>
          </div>
        </div>

        {/* Client-side data fetching components */}
        <ClientAnalyzePage ticker={ticker} />

        {/* News & Sentiment */}
        <NewsSection ticker={ticker} />

        {/* Technical Indicators */}
        <TechnicalIndicators ticker={ticker} />

        {/* Fair Value Analysis */}
        <FairValueCard ticker={ticker} />
      </div>
    </div>
  );
}

