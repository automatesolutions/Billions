import { expect, test } from '@playwright/test';

const DISCLAIMER = 'Information only. Not financial advice. This tool does not place trades.';

test('home shows the live board and leads to the outliers', async ({ page }) => {
  await page.goto('/');
  await expect(page.getByRole('heading', { level: 1, name: 'Stocks moving far from the pack.' })).toBeVisible();
  await expect(page.getByRole('list', { name: 'Top five swing outliers' }).getByRole('link').first()).toContainText('USDE');
  await expect(page.getByText(DISCLAIMER)).toBeVisible();
  await page.getByRole('link', { name: /See today.s outliers/ }).click();
  await expect(page).toHaveURL(/\/outliers\/swing$/);
});

test('outliers list opens a stock analysis and back again', async ({ page, isMobile }) => {
  await page.goto('/outliers/swing');
  await expect(page).toHaveTitle('Swing outliers · BILLIONS');
  await expect(page.getByRole('heading', { level: 1, name: 'Swing outliers' })).toBeVisible();
  await expect(page.getByText('Data as of')).toBeVisible();
  await expect(page.getByText(DISCLAIMER)).toBeVisible();
  await expect(page.locator('head meta[property="og:image"]')).toHaveCount(1);
  await expect(page.locator('head meta[name="description"]')).toHaveCount(1);

  // The first ranked outlier links to its analysis.
  const ranked = page.locator('#ranked');
  const first = isMobile ? ranked.locator('ol a').first() : ranked.getByRole('table').getByRole('link').first();
  await expect(first).toContainText('USDE');
  await first.click();

  await expect(page).toHaveURL(/\/analysis\/USDE\?from=swing$/);
  await expect(page).toHaveTitle('USDE analysis · BILLIONS');
  await expect(page.getByRole('heading', { level: 1, name: 'USDE' })).toBeVisible();
  await expect(page.locator('head meta[property="og:image"]')).toHaveCount(1);
  for (const section of ['Signal', 'Price and forecast', 'Edge of the signal', 'Risk', 'Model comparison', 'Validation', 'Cost check', 'Microstructure']) {
    await expect(page.getByRole('heading', { level: 2, name: section, exact: true })).toBeVisible();
  }
  await expect(page.getByRole('meter', { name: 'Order-book imbalance' })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Use live spread' })).toBeVisible();
  await expect(page.getByText(/No significant edge|Edge beats random/).first()).toBeVisible();
  await expect(page.getByRole('heading', { name: 'What this does not tell you' })).toBeVisible();
  await expect(page.getByText(DISCLAIMER)).toBeVisible();

  // Editing the cost updates the verdict.
  const cost = page.getByLabel('Round-trip cost (bps)');
  await cost.fill('500');
  await expect(page.getByText('smaller than a 500 bps round trip')).toBeVisible();

  // Browser back returns to the outliers.
  await page.goBack();
  await expect(page).toHaveURL(/\/outliers\/swing$/);
});

test('strategy switcher changes the URL', async ({ page }) => {
  await page.goto('/outliers/swing');
  await page.getByRole('navigation', { name: 'Strategy' }).getByRole('link', { name: /Scalp/ }).click();
  await expect(page).toHaveURL(/\/outliers\/scalp$/);
  await expect(page.getByRole('heading', { level: 1, name: 'Scalp outliers' })).toBeVisible();
  await page.goBack();
  await expect(page).toHaveURL(/\/outliers\/swing$/);
});

test('unknown ticker shows a clear message', async ({ page }) => {
  await page.goto('/analysis/NOPE');
  await expect(page.getByText('No prices for NOPE')).toBeVisible();
});

test('quant page: pick a stock and interval, see the action and the live record', async ({ page, isMobile }) => {
  const ticker = isMobile ? 'MSFT' : 'NVDA'; // separate tracking state per project on the shared mock
  await page.goto('/quant');
  await expect(page).toHaveTitle('Quant · BILLIONS');
  await expect(page.getByRole('heading', { level: 1, name: 'Is the strategy working?' })).toBeVisible();
  await expect(page.getByRole('link', { name: /AAPL/ }).first()).toBeVisible(); // tracked list
  await page.getByRole('textbox', { name: 'Stock' }).fill(ticker.toLowerCase());
  await page.getByText('4 hours', { exact: true }).click();
  await page.getByRole('button', { name: 'Show the strategy' }).click();

  await expect(page).toHaveURL(new RegExp(`/quant/${ticker}\\?interval=4h$`));
  await expect(page.getByRole('heading', { level: 1, name: ticker })).toBeVisible();
  for (const section of ['What to do on the next bar', 'Is it working?', 'Backtest', 'How the signal is built']) {
    await expect(page.getByRole('heading', { level: 2, name: section, exact: true })).toBeVisible();
  }
  await expect(page.getByRole('navigation', { name: 'Bar interval' }).getByRole('link', { name: /1 hour/ })).toBeVisible();
  await expect(page.getByText('Checks before acting')).toBeVisible();
  await expect(page.getByRole('img', { name: /^Order side: (buy|sell|hold)\./ })).toBeVisible();
  await expect(page.locator('#action p').filter({ hasText: /^(Buy|Sell|Hold|Sell short|Buy to cover|Sell \+ short|Cover \+ buy)$/ })).toBeVisible(); // the headline, not the scale
  await page.getByRole('button', { name: 'Track every bar' }).click();
  await expect(page.getByRole('button', { name: 'Tracking · Stop' })).toHaveAttribute('aria-pressed', 'true');
  await expect(page.getByRole('heading', { name: new RegExp(`Tracking ${ticker} every 4-hour bar`) })).toBeVisible();
  await expect(page.getByRole('progressbar', { name: /Time until the next/ })).toBeVisible();
  await expect(page.getByRole('timer')).toContainText('Bar closes in');
  await page.getByRole('button', { name: 'Tracking · Stop' }).click();
  await expect(page.getByRole('button', { name: 'Track every bar' })).toBeVisible();
  await expect(page.getByText(DISCLAIMER)).toBeVisible();
});
