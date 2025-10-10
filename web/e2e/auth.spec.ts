import { test, expect } from '@playwright/test';

test.describe('Authentication Flow', () => {
  test('should show login page for unauthenticated users', async ({ page }) => {
    await page.goto('/dashboard');
    
    // Should redirect to login
    await expect(page).toHaveURL('/login');
    
    // Should show login elements
    await expect(page.getByRole('heading', { name: 'BILLIONS' })).toBeVisible();
    await expect(page.getByRole('button', { name: /Sign in with Google/i })).toBeVisible();
  });

  test('should display login page correctly', async ({ page }) => {
    await page.goto('/login');
    
    // Check for main elements
    await expect(page.getByText('BILLIONS')).toBeVisible();
    await expect(page.getByText(/Stock Market Forecasting/i)).toBeVisible();
    await expect(page.getByAltText('BILLIONS Logo')).toBeVisible();
    
    // Check for Google sign in button
    const signInButton = page.getByRole('button', { name: /Sign in with Google/i });
    await expect(signInButton).toBeVisible();
    await expect(signInButton).toBeEnabled();
  });

  test('should show terms and conditions', async ({ page }) => {
    await page.goto('/login');
    
    await expect(page.getByText(/By signing in, you agree to our/i)).toBeVisible();
    await expect(page.getByText(/Terms of Service/i)).toBeVisible();
  });

  test('should protect dashboard route', async ({ page }) => {
    await page.goto('/dashboard');
    
    // Should redirect to login page
    await page.waitForURL('/login');
    await expect(page).toHaveURL('/login');
  });

  test('should protect analyze route', async ({ page }) => {
    await page.goto('/analyze/TSLA');
    
    // Should redirect to login page
    await page.waitForURL('/login');
    await expect(page).toHaveURL('/login');
  });

  test('should protect outliers route', async ({ page }) => {
    await page.goto('/outliers');
    
    // Should redirect to login page
    await page.waitForURL('/login');
    await expect(page).toHaveURL('/login');
  });

  test('should allow access to homepage without auth', async ({ page }) => {
    await page.goto('/');
    
    // Should stay on homepage
    await expect(page).toHaveURL('/');
    await expect(page.getByText('BILLIONS')).toBeVisible();
  });
});

