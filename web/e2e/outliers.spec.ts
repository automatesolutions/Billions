import { test, expect } from '@playwright/test';

test.describe('Outliers Page', () => {
  test('should redirect to login if not authenticated', async ({ page }) => {
    await page.goto('/outliers');
    
    await page.waitForURL('/login');
    await expect(page).toHaveURL('/login');
  });

  test('should protect outliers route', async ({ page }) => {
    await page.goto('/outliers');
    
    // Should not allow access without auth
    await expect(page).toHaveURL('/login');
  });
});

