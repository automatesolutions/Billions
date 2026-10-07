import { defineConfig, devices } from '@playwright/test';

const WEB_PORT = 3100;
const API_PORT = 8010;
const api = `http://localhost:${API_PORT}`;

export default defineConfig({
  testDir: './e2e',
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  workers: process.env.CI ? 1 : 2,
  expect: { timeout: 15_000 },
  reporter: process.env.CI ? [['github'], ['html', { open: 'never' }]] : 'list',
  use: {
    baseURL: `http://localhost:${WEB_PORT}`,
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
  },
  projects: [
    { name: 'desktop', use: { ...devices['Desktop Chrome'] } },
    { name: 'mobile', use: { ...devices['Pixel 7'] } },
  ],
  webServer: [
    { command: 'node e2e/mock-api.mjs', url: `${api}/health`, reuseExistingServer: !process.env.CI, env: { MOCK_API_PORT: String(API_PORT) } },
    {
      command: `pnpm build && pnpm start -p ${WEB_PORT}`,
      url: `http://localhost:${WEB_PORT}/methodology`,
      timeout: 240_000,
      reuseExistingServer: !process.env.CI,
      env: { API_URL: api, NEXT_PUBLIC_API_URL: api, NEXT_PUBLIC_SITE_URL: `http://localhost:${WEB_PORT}` },
    },
  ],
});
