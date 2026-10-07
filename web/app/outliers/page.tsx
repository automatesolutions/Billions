import { ClientOutliersPage } from "./client-page";

export default function OutliersPage() {
  return (
    <main className="min-h-screen bg-background p-8">
      <div className="max-w-7xl mx-auto space-y-8">
        <h1 className="text-3xl font-bold">Outliers</h1>
        <ClientOutliersPage />
      </div>
    </main>
  );
}
