import { auth } from "@/auth";
import { redirect } from "next/navigation";
import Link from "next/link";
import Image from "next/image";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { signOut } from "@/auth";
import { AnalyzeStockSearch } from "@/components/analyze-stock-search";
import { NASDAQNewsSection } from "@/components/nasdaq-news-section";

export default async function DashboardPage() {
  const session = await auth();

  // For demo purposes, allow access without authentication
  // if (!session?.user) {
  //   redirect("/login");
  // }

  async function handleSignOut() {
    "use server";
    await signOut({ redirectTo: "/" });
  }

  return (
    <div className="min-h-screen bg-background p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div className="flex items-center gap-4">
            <Image
              src="/logo1.png"
              alt="BILLIONS Logo"
              width={60}
              height={60}
            />
            <div>
              <h1 className="text-3xl font-bold">Dashboard</h1>
              <p className="text-muted-foreground">
                {session?.user ? `Welcome back, ${session.user.name}!` : "Demo Dashboard - No login required"}
              </p>
            </div>
          </div>
          
          <div className="flex items-center gap-4">
            {/* Account Information - Compact with Avatar */}
            <Card className="bg-background border-0 shadow-none">
              <CardContent className="p-2.5">
                <div className="flex items-center gap-2.5">
                  <div className="flex items-center justify-center w-8 h-8 rounded-full bg-yellow-500 text-white font-bold text-sm">
                    {session?.user?.name?.charAt(0).toUpperCase() || "D"}
                  </div>
                  <div>
                    <div className="font-semibold text-white text-xs">{session?.user?.name || "Demo User"}</div>
                    <div className="text-xs text-gray-400 text-[10px]">{session?.user?.email || "demo@billions.app"}</div>
                  </div>
                </div>
              </CardContent>
            </Card>

            {session?.user ? (
              <form action={handleSignOut}>
                <Button variant="outline" type="submit">
                  Sign Out
                </Button>
              </form>
            ) : (
              <Link href="/">
                <Button variant="outline">
                  Back to Home
                </Button>
              </Link>
            )}
          </div>
        </div>

        <div className="flex gap-8">
          {/* Sidebar */}
          <div className="w-64 space-y-4 flex-shrink-0">
            {/* Quant Trade */}
            <Link href="/trading/hft">
              <Card className="hover:bg-gray-900 transition-colors cursor-pointer bg-background border-0 shadow-none">
                <CardHeader>
                  <CardTitle className="text-lg text-white">💼 Quant Trade</CardTitle>
                  <CardDescription className="text-gray-500">
                    Track holdings and performance
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-gray-500">
                    Real-time trading with Polygon.io
                  </p>
                </CardContent>
              </Card>
            </Link>

            {/* Outlier Detection */}
            <Link href="/outliers">
              <Card className="hover:bg-gray-900 transition-colors cursor-pointer bg-background border-0 shadow-none">
                <CardHeader>
                  <CardTitle className="text-lg text-white">🎯 Outlier Detection</CardTitle>
                  <CardDescription className="text-gray-500">
                    Find exceptional performance patterns
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-gray-500">
                    3 strategies: Scalp, Swing, Longterm
                  </p>
                </CardContent>
              </Card>
            </Link>

            {/* Capitulation Detection */}
            <Link href="/capitulation">
              <Card className="hover:bg-gray-900 transition-colors cursor-pointer bg-background border-0 shadow-none">
                <CardHeader>
                  <CardTitle className="text-lg text-white">⚠️ Capitulation Detection</CardTitle>
                  <CardDescription className="text-gray-500">
                    Screen all NASDAQ stocks for capitulation signals
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-gray-500">
                    Volume spikes, RSI oversold, MACD bearish
                  </p>
                </CardContent>
              </Card>
            </Link>
          </div>

          {/* Main Content */}
          <div className="flex-1 space-y-8">
            {/* Analyze Stock */}
            <AnalyzeStockSearch />

            {/* NASDAQ First-Edge News */}
            <NASDAQNewsSection />
          </div>
        </div>
      </div>
    </div>
  );
}

