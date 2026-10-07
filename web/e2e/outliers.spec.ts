import { test, expect } from '@playwright/test';

test('outliers page loads without login', async ({ page }) => {
  await page.goto('/outliers');
  await expect(page.getByRole('heading', { name: 'Outliers' })).toBeVisible();
});
