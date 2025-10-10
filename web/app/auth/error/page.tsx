import Link from "next/link";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";

export default function AuthErrorPage({
  searchParams,
}: {
  searchParams: { error?: string };
}) {
  const error = searchParams.error;

  return (
    <div className="min-h-screen flex items-center justify-center bg-background p-4">
      <Card className="w-full max-w-md">
        <CardHeader>
          <CardTitle className="text-2xl text-destructive">Authentication Error</CardTitle>
          <CardDescription>
            {error === "Configuration" && "There is a problem with the server configuration."}
            {error === "AccessDenied" && "Access was denied. Please try again."}
            {error === "Verification" && "The verification token has expired or has already been used."}
            {!error && "An unknown error occurred during authentication."}
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <p className="text-sm text-muted-foreground">
            If this problem persists, please contact support.
          </p>
          <Link href="/login">
            <Button className="w-full">
              Back to Login
            </Button>
          </Link>
        </CardContent>
      </Card>
    </div>
  );
}

