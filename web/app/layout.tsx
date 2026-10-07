import type { Metadata, Viewport } from 'next';
import { generalSans } from './fonts';
import { SiteHeader } from '@/components/site-header';
import { SiteFooter } from '@/components/site-footer';
import { ServiceWorker } from '@/components/service-worker';
import { TOKENS } from '@/lib/tokens';
import './globals.css';

const SITE_URL = process.env.NEXT_PUBLIC_SITE_URL || 'http://localhost:3000';
const DESCRIPTION = 'Stocks moving unusually far from the pack, with a tested quant analysis of each. Information only.';

export const metadata: Metadata = {
  metadataBase: new URL(SITE_URL),
  title: { default: 'BILLIONS', template: '%s · BILLIONS' },
  description: DESCRIPTION,
  applicationName: 'BILLIONS',
  openGraph: { type: 'website', siteName: 'BILLIONS', title: 'BILLIONS', description: DESCRIPTION, locale: 'en_US' },
  twitter: { card: 'summary_large_image', title: 'BILLIONS', description: DESCRIPTION },
  appleWebApp: { capable: true, title: 'BILLIONS', statusBarStyle: 'black-translucent' },
  formatDetection: { telephone: false },
};

export const viewport: Viewport = {
  themeColor: TOKENS.canvas,
  colorScheme: 'dark',
  width: 'device-width',
  initialScale: 1,
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={generalSans.variable} suppressHydrationWarning>
      <body className="flex min-h-dvh flex-col" suppressHydrationWarning>
        <SiteHeader />
        <main id="main" className="mx-auto w-full max-w-content flex-1 px-xs pt-md sm:px-md">
          {children}
        </main>
        <SiteFooter />
        <ServiceWorker />
      </body>
    </html>
  );
}
