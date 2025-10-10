import Image from "next/image";
import Link from "next/link";
import { auth } from "@/auth";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

export default async function Home() {
  const session = await auth();

  return (
    <div className="min-h-screen bg-background p-8">
      <div className="max-w-4xl mx-auto space-y-8">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Image
              src="/logo.png"
              alt="BILLIONS Logo"
              width={80}
              height={80}
              priority
            />
            <div>
              <h1 className="text-4xl font-bold tracking-tight">BILLIONS</h1>
              <p className="text-muted-foreground">
                Stock Market Forecasting & Outlier Detection
              </p>
            </div>
          </div>
          
          {session ? (
            <Link href="/dashboard">
              <Button>Go to Dashboard</Button>
            </Link>
          ) : (
            <Link href="/login">
              <Button>Sign In</Button>
            </Link>
          )}
        </div>

        {/* Welcome Card */}
        <Card>
          <CardHeader>
            <CardTitle>Welcome to BILLIONS</CardTitle>
            <CardDescription>
              {session 
                ? `Welcome back, ${session.user.name}!`
                : "Sign in to access all features"
              }
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {session ? (
              <div>
                <p className="text-sm text-muted-foreground mb-4">
                  You're signed in and ready to explore market insights, predictions, and outlier detection.
                </p>
                <Link href="/dashboard">
                  <Button className="w-full">
                    Go to Dashboard
                  </Button>
                </Link>
              </div>
            ) : (
              <div>
                <p className="text-sm text-muted-foreground mb-4">
                  Sign in with Google to access your personalized dashboard, save watchlists, and get predictions.
                </p>
                <Link href="/login">
                  <Button className="w-full">
                    Sign In with Google
                  </Button>
                </Link>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Quick Links */}
        <Card>
          <CardHeader>
            <CardTitle>Quick Links</CardTitle>
            <CardDescription>
              Development resources and documentation
            </CardDescription>
          </CardHeader>
          <CardContent className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <a
              href="http://localhost:8000/docs"
              target="_blank"
              rel="noopener noreferrer"
              className="p-4 border rounded-lg hover:bg-accent transition-colors"
            >
              <h3 className="font-semibold mb-1">API Documentation</h3>
              <p className="text-sm text-muted-foreground">
                OpenAPI/Swagger interactive docs
              </p>
            </a>
            
            <a
              href="http://localhost:8000/health"
              target="_blank"
              rel="noopener noreferrer"
              className="p-4 border rounded-lg hover:bg-accent transition-colors"
            >
              <h3 className="font-semibold mb-1">Health Check</h3>
              <p className="text-sm text-muted-foreground">
                Backend health status endpoint
              </p>
            </a>
          </CardContent>
        </Card>

        {/* Progress Info */}
        <Card>
          <CardHeader>
            <CardTitle>Development Progress 🎉</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2 text-sm">
            <p>✅ Phase 0: Foundation & Analysis</p>
            <p>✅ Phase 1: Infrastructure Setup</p>
            <p>✅ Phase 2: Testing Infrastructure (19 tests, 76% coverage)</p>
            <p>✅ Phase 3: Authentication & User Management</p>
            <div className="pt-4 text-muted-foreground">
              <p className="font-semibold mb-2">Coming Next:</p>
              <ul className="list-disc list-inside space-y-1">
                <li>Phase 4: ML Backend APIs (predictions, outliers)</li>
                <li>Phase 5: Frontend UI & Charts</li>
                <li>Phase 6: Deployment & Monitoring</li>
                <li>Phase 7: Data Migration</li>
                <li>Phase 8: Launch</li>
              </ul>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
