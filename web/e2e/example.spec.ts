import { test, expect } from '@playwright/test';

test.describe('BILLIONS Homepage', () => {
  test('should load the homepage', async ({ page }) => {
    await page.goto('/');
    
    // Check for BILLIONS heading
    await expect(page.getByRole('heading', { name: /BILLIONS/i })).toBeVisible();
    
    // Check for logo
    await expect(page.getByAltText(/BILLIONS Logo/i)).toBeVisible();
  });

  test('should display system status card', async ({ page }) => {
    await page.goto('/');
    
    // Check for system status section
    await expect(page.getByText(/System Status/i)).toBeVisible();
    await expect(page.getByText(/API Status/i)).toBeVisible();
  });

  test('should have accessible links', async ({ page }) => {
    await page.goto('/');
    
    // Check for API documentation link
    const apiDocsLink = page.getByRole('link', { name: /API Documentation/i });
    await expect(apiDocsLink).toBeVisible();
    await expect(apiDocsLink).toHaveAttribute('href', 'http://localhost:8000/docs');
  });
});

