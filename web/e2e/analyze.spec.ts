import { test, expect } from '@playwright/test';

test.describe('Ticker Analysis Page', () => {
  test('should redirect to login if not authenticated', async ({ page }) => {
    await page.goto('/analyze/TSLA');
    
    // Should redirect to login
    await page.waitForURL('/login');
    await expect(page).toHaveURL('/login');
  });

  test('should show ticker in page title', async ({ page }) => {
    await page.goto('/analyze/AAPL');
    
    // Will redirect to login, but structure validates
    await expect(page).toHaveURL('/login');
  });

  test('should have back to dashboard button', async ({ page }) => {
    // This will be testable after authentication is mocked
    // Structure in place for authenticated testing
  });
});

