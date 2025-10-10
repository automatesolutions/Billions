import { auth } from "@/auth";
import { redirect } from "next/navigation";
import Image from "next/image";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { signOut } from "@/auth";

export default async function DashboardPage() {
  const session = await auth();

  if (!session?.user) {
    redirect("/login");
  }

  async function handleSignOut() {
    "use server";
    await signOut({ redirectTo: "/" });
  }

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
              <h1 className="text-3xl font-bold">Dashboard</h1>
              <p className="text-muted-foreground">
                Welcome back, {session.user.name}!
              </p>
            </div>
          </div>
          
          <form action={handleSignOut}>
            <Button variant="outline" type="submit">
              Sign Out
            </Button>
          </form>
        </div>

        {/* User Info Card */}
        <Card>
          <CardHeader>
            <CardTitle>Account Information</CardTitle>
            <CardDescription>Your profile details</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center gap-4">
              {session.user.image && (
                <Image
                  src={session.user.image}
                  alt={session.user.name || "User"}
                  width={64}
                  height={64}
                  className="rounded-full"
                />
              )}
              <div>
                <div className="font-semibold text-lg">{session.user.name}</div>
                <div className="text-sm text-muted-foreground">{session.user.email}</div>
                <Badge variant="secondary" className="mt-2">Free Tier</Badge>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Quick Stats */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <Card>
            <CardHeader>
              <CardTitle className="text-sm font-medium">Watchlist</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">0</div>
              <p className="text-xs text-muted-foreground">Stocks tracked</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-sm font-medium">Predictions</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">0</div>
              <p className="text-xs text-muted-foreground">Generated today</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-sm font-medium">Alerts</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">0</div>
              <p className="text-xs text-muted-foreground">Active alerts</p>
            </CardContent>
          </Card>
        </div>

        {/* Coming Soon */}
        <Card>
          <CardHeader>
            <CardTitle>Features Coming Soon</CardTitle>
            <CardDescription>Under development</CardDescription>
          </CardHeader>
          <CardContent className="space-y-2">
            <div className="flex items-center gap-2">
              <Badge variant="outline">Phase 4</Badge>
              <span className="text-sm">30-Day ML Predictions</span>
            </div>
            <div className="flex items-center gap-2">
              <Badge variant="outline">Phase 4</Badge>
              <span className="text-sm">Outlier Detection (Scalp, Swing, Longterm)</span>
            </div>
            <div className="flex items-center gap-2">
              <Badge variant="outline">Phase 5</Badge>
              <span className="text-sm">Interactive Charts & Analysis</span>
            </div>
            <div className="flex items-center gap-2">
              <Badge variant="outline">Phase 5</Badge>
              <span className="text-sm">Portfolio Tracking</span>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

