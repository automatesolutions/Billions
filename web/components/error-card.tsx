import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";

interface ErrorCardProps {
  title?: string;
  description?: string;
  error: string;
  onRetry?: () => void;
}

export function ErrorCard({ 
  title = "Error",
  description = "Something went wrong",
  error,
  onRetry 
}: ErrorCardProps) {
  return (
    <Card className="border-destructive">
      <CardHeader>
        <CardTitle className="text-destructive">{title}</CardTitle>
        <CardDescription>{description}</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <p className="text-sm text-muted-foreground">{error}</p>
        {onRetry && (
          <Button onClick={onRetry} variant="outline" size="sm">
            Try Again
          </Button>
        )}
      </CardContent>
    </Card>
  );
}

