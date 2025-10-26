import Link from "next/link";
import Image from "next/image";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { TickerSearch } from "@/components/ticker-search";

export default function DemoPage() {
  return (
    <div className="min-h-screen bg-background">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div className="flex items-center gap-4">
            <Image
              src="/logo1.png"
              alt="BILLIONS"
              width={40}
              height={40}
              className="rounded"
            />
            <div>
              <h1 className="text-2xl font-bold">BILLIONS</h1>
              <p className="text-muted-foreground">Demo Dashboard</p>
            </div>
          </div>
          <Link href="/">
            <Button variant="outline">Back to Home</Button>
          </Link>
        </div>

        {/* Welcome Card */}
        <Card className="mb-6">
          <CardHeader>
            <CardTitle>Welcome to BILLIONS Demo</CardTitle>
            <CardDescription>
              Explore the ML-powered stock analysis features
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <p className="text-sm text-muted-foreground">
              This is a demo version where you can explore the interface and features. 
              For full functionality including Google OAuth login, set up authentication.
            </p>
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <Link href="/analyze/TSLA">
                <Card className="hover:bg-accent transition-colors cursor-pointer">
                  <CardHeader className="pb-3">
                    <CardTitle className="text-lg">📈 Analyze Stock</CardTitle>
                    <CardDescription>Try analyzing TSLA</CardDescription>
                  </CardHeader>
                </Card>
              </Link>
              
              <Link href="/outliers">
                <Card className="hover:bg-accent transition-colors cursor-pointer">
                  <CardHeader className="pb-3">
                    <CardTitle className="text-lg">🎯 Outlier Detection</CardTitle>
                    <CardDescription>View market outliers</CardDescription>
                  </CardHeader>
                </Card>
              </Link>
              
              <Link href="/portfolio">
                <Card className="hover:bg-accent transition-colors cursor-pointer">
                  <CardHeader className="pb-3">
                    <CardTitle className="text-lg">💼 Portfolio</CardTitle>
                    <CardDescription>Portfolio management</CardDescription>
                  </CardHeader>
                </Card>
              </Link>
            </div>
          </CardContent>
        </Card>

        {/* Ticker Search */}
        <Card>
          <CardHeader>
            <CardTitle>Quick Stock Search</CardTitle>
            <CardDescription>
              Search for any stock ticker to analyze
            </CardDescription>
          </CardHeader>
          <CardContent>
            <TickerSearch />
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
