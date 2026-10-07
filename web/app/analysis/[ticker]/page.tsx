interface PageProps {
  params: Promise<{ ticker: string }>;
}

export default async function AnalysisPage({ params }: PageProps) {
  const { ticker } = await params;

  return (
    <main className="min-h-screen bg-background p-8">
      <div className="max-w-7xl mx-auto space-y-4">
        <h1 className="text-3xl font-bold">{ticker.toUpperCase()}</h1>
        <p className="text-muted-foreground">Analysis is being rebuilt.</p>
      </div>
    </main>
  );
}
