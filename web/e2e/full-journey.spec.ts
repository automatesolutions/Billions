import { test, expect } from '@playwright/test';

test.describe('Complete User Journey', () => {
  test('user can navigate through all features', async ({ page }) => {
    // 1. Start at homepage
    await page.goto('/');
    await expect(page.getByText('BILLIONS')).toBeVisible();
    
    // 2. Try to access dashboard (should redirect to login)
    await page.goto('/dashboard');
    await expect(page).toHaveURL('/login');
    
    // 3. See login page
    await expect(page.getByRole('button', { name: /Sign in with Google/i })).toBeVisible();
    
    // 4. Try to access outliers (should redirect to login)
    await page.goto('/outliers');
    await expect(page).toHaveURL('/login');
    
    // 5. Try ticker analysis (should redirect to login)
    await page.goto('/analyze/TSLA');
    await expect(page).toHaveURL('/login');
  });

  test('all protected routes require authentication', async ({ page }) => {
    const protectedRoutes = ['/dashboard', '/outliers', '/analyze/AAPL', '/portfolio'];
    
    for (const route of protectedRoutes) {
      await page.goto(route);
      await expect(page).toHaveURL('/login');
    }
  });

  test('public routes are accessible', async ({ page }) => {
    // Homepage
    await page.goto('/');
    await expect(page).toHaveURL('/');
    
    // Login page
    await page.goto('/login');
    await expect(page).toHaveURL('/login');
  });
});

