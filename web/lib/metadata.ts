import type { Metadata } from 'next';

/**
 * Open Graph for a page. A page-level `openGraph` replaces the root one instead of merging,
 * so the shared preview image is added here every time.
 */
export function pageOpenGraph(title: string, description: string, url: string): Metadata['openGraph'] {
  return {
    type: 'website',
    siteName: 'BILLIONS',
    title,
    description,
    url,
    images: [{ url: '/opengraph-image.png', width: 1200, height: 630, alt: 'BILLIONS: outlier stocks, measured and explained' }],
  };
}
