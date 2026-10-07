# Lighthouse results

Lighthouse 12.8.2, 2026-10-07. Production build (`next build && next start`) on localhost, serving recorded real data through the E2E mock API. Mobile runs use Lighthouse's default throttling (simulated slow 4G, 4x CPU slowdown); each mobile page was run three times.

| Run | Performance | Accessibility | Best practices | SEO | FCP | LCP | TBT | CLS |
|---|---|---|---|---|---|---|---|---|
| analysis-AAPL-desktop-1 | 100 | 100 | 100 | 100 | 0.3 s | 0.6 s | 0 ms | 0 |
| analysis-AAPL-mobile-1 | 95 | 100 | 100 | 100 | 0.9 s | 2.9 s | 60 ms | 0.001 |
| analysis-AAPL-mobile-2 | 95 | 100 | 100 | 100 | 1.2 s | 2.7 s | 110 ms | 0.001 |
| analysis-AAPL-mobile-3 | 96 | 100 | 100 | 100 | 1.2 s | 2.3 s | 120 ms | 0.084 |
| outliers-swing-desktop-1 | 100 | 100 | 100 | 100 | 0.3 s | 0.6 s | 0 ms | 0.005 |
| outliers-swing-mobile-1 | 95 | 100 | 100 | 100 | 1.4 s | 2.5 s | 140 ms | 0.006 |
| outliers-swing-mobile-2 | 97 | 100 | 100 | 100 | 1.4 s | 2.4 s | 80 ms | 0.006 |
| outliers-swing-mobile-3 | 98 | 100 | 100 | 100 | 1.1 s | 2.2 s | 100 ms | 0.005 |

Re-run: `npx lighthouse http://localhost:3000/outliers/swing --only-categories=performance,accessibility,best-practices,seo`.
