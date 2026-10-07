import type { MetadataRoute } from 'next';
import { TOKENS } from '@/lib/tokens';

export default function manifest(): MetadataRoute.Manifest {
  return {
    name: 'BILLIONS',
    short_name: 'BILLIONS',
    description: 'Outlier stocks, measured and explained. Information only.',
    start_url: '/outliers/swing',
    scope: '/',
    display: 'standalone',
    background_color: TOKENS.canvas,
    theme_color: TOKENS.canvas,
    categories: ['finance'],
    icons: [
      { src: '/icons/icon-192.png', sizes: '192x192', type: 'image/png', purpose: 'any' },
      { src: '/icons/icon-512.png', sizes: '512x512', type: 'image/png', purpose: 'any' },
      { src: '/icons/icon-maskable-512.png', sizes: '512x512', type: 'image/png', purpose: 'maskable' },
    ],
  };
}
