import { auth } from "@/auth";
import { redirect } from "next/navigation";
import Link from "next/link";
import Image from "next/image";
import { Button } from "@/components/ui/button";
import { ClientOutliersPage } from "./client-page";

export default async function OutliersPage() {
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
              <h1 className="text-3xl font-bold">Outlier Detection</h1>
              <p className="text-muted-foreground">
                Identify exceptional stock performance patterns
              </p>
            </div>
          </div>
          <Link href="/dashboard">
            <Button variant="outline" size="sm">Back to Dashboard</Button>
          </Link>
        </div>

        {/* Client-side components with data fetching */}
        <ClientOutliersPage />
      </div>
    </div>
  );
}

