import { auth } from "@/auth";
import { redirect } from "next/navigation";
import Link from "next/link";
import Image from "next/image";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { PortfolioSetup } from "./portfolio-setup";
import { PortfolioDashboard } from "./portfolio-dashboard";

export default async function PortfolioPage() {
  const session = await auth();

  // Allow demo access without authentication
  // if (!session?.user) {
  //   redirect("/login");
  // }

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
              <h1 className="text-3xl font-bold">Portfolio</h1>
              <p className="text-muted-foreground">
                Track your holdings and performance
              </p>
            </div>
          </div>
          <Link href="/dashboard">
            <Button variant="outline" size="sm">Back to Dashboard</Button>
          </Link>
        </div>

        {/* Portfolio Setup & Dashboard */}
        <PortfolioSetup />
        <PortfolioDashboard />
        {/* Legacy TradingDashboard removed in favor of integrated HFT controls */}
      </div>
    </div>
  );
}

