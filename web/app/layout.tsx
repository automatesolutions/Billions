import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "BILLIONS",
  description: "Outlier stocks and per-stock quant analysis. Information only.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className="dark">
      <body className="antialiased">
        {children}
        <footer className="p-8 text-sm text-muted-foreground">
          Information only. Not financial advice. This tool does not place trades.
        </footer>
      </body>
    </html>
  );
}
