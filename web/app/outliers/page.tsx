import { auth } from "@/auth";
import { redirect } from "next/navigation";
import Image from "next/image";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

export default async function OutliersPage() {
  const session = await auth();

  if (!session?.user) {
    redirect("/login");
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
              <h1 className="text-3xl font-bold">Outlier Detection</h1>
              <p className="text-muted-foreground">
                Identify exceptional stock performance patterns
              </p>
            </div>
          </div>
        </div>

        {/* Strategy Selector */}
        <Card>
          <CardHeader>
            <CardTitle>Select Strategy</CardTitle>
            <CardDescription>
              Choose a trading timeframe to analyze outliers
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <Select defaultValue="swing">
              <SelectTrigger className="w-full md:w-[300px]">
                <SelectValue placeholder="Select strategy" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="scalp">
                  <div className="flex flex-col">
                    <span className="font-semibold">Scalp (1-Week)</span>
                    <span className="text-xs text-muted-foreground">
                      1-week vs 1-month performance
                    </span>
                  </div>
                </SelectItem>
                <SelectItem value="swing">
                  <div className="flex flex-col">
                    <span className="font-semibold">Swing (3-Month)</span>
                    <span className="text-xs text-muted-foreground">
                      3-month vs 1-month performance
                    </span>
                  </div>
                </SelectItem>
                <SelectItem value="longterm">
                  <div className="flex flex-col">
                    <span className="font-semibold">Longterm (1-Year)</span>
                    <span className="text-xs text-muted-foreground">
                      1-year vs 6-month performance
                    </span>
                  </div>
                </SelectItem>
              </SelectContent>
            </Select>

            <div className="flex gap-2">
              <Badge variant="outline">High Market Cap Filter</Badge>
              <Badge variant="outline">Z-Score &gt; 2</Badge>
            </div>
          </CardContent>
        </Card>

        {/* Placeholder for Scatter Plot */}
        <Card>
          <CardHeader>
            <CardTitle>Performance Scatter Plot</CardTitle>
            <CardDescription>
              Outliers shown in red (|z-score| &gt; 2)
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="aspect-video bg-muted/20 rounded-lg flex items-center justify-center border-2 border-dashed">
              <div className="text-center text-muted-foreground">
                <p className="text-lg font-semibold mb-2">Scatter Plot</p>
                <p className="text-sm">Chart component coming in next iteration</p>
                <p className="text-xs mt-2">Phase 5.4 - Data Visualization</p>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Outliers List */}
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardTitle>Detected Outliers</CardTitle>
                <CardDescription>
                  Stocks with exceptional performance patterns
                </CardDescription>
              </div>
              <Badge variant="secondary">Loading from API...</Badge>
            </div>
          </CardHeader>
          <CardContent>
            <div className="text-sm text-muted-foreground text-center py-8">
              Connected to: /api/v1/market/outliers/swing
              <br />
              Data fetching will be implemented with TanStack Query
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

