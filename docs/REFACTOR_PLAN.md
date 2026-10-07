# BILLIONS refactor plan: outliers only

Branch: `refactor/outliers-only`
Status: Phases 0–4 done. Decisions: all recommendations accepted (Q1–Q8). Q7: no history rewrite was done, so the old Alpaca keys still need revoking.

Goal: a public, read-only web app with three pages:

| Route | Purpose |
|---|---|
| `/` → redirects to `/outliers/swing` | Home |
| `/outliers/[strategy]` (`scalp`, `swing`, `longterm`) | Scatter plus ranked outlier table |
| `/analysis/[ticker]` | Per-stock quant analysis (Phase 3) |
| `/methodology` | Plain-language metrics and data limits |

No accounts. Nothing places, routes or simulates a trade.

## How this audit was done

I read the code, not the README. These are the main ways the README is wrong:

- The README says "89 tests passing, 85% coverage". The repo has 5 backend test files and 5 Vitest files. Most frontend tests cover auth and a test page.
- It describes LSTM forecasts with confidence bands. The bands come from a heuristic (see bug B9), not a statistical interval.
- It says the outlier page shows live data. In practice it silently shows **hard-coded mock data** whenever the API returns nothing, and the engine almost never finds outliers (see B4, B5 and the B1 retraction).

Sources read: `README.md`, `PLAN.md`, `SYSTEM_ARCHITECTURE_FLOWCHART.md`, every file in `api/`, `db/`, `funda/*.py`, `outlier/*.py`, `web/app`, `web/components`, `web/hooks`, `web/lib`, all config, deploy and CI files, `docs/reference/quant-trading-handbook.pdf`, `docs/reference/design-tricks-guide.pdf`, and `design-md/ferrari/` (`DESIGN.md` + `README.md`) from VoltAgent/awesome-design-md.

> Note: the PDFs were not in the repo. I copied them into `docs/reference/` as `quant-trading-handbook.pdf` and `design-tricks-guide.pdf`.

---

## 1. Keep / delete / rewrite

Legend: **KEEP** = reuse as is, apart from cleanup. **REWRITE** = keep the idea, replace the code. **DELETE** = remove in Phase 1. **ASK** = I think it should go, but you didn't list it, so I need a yes from you.

### 1.1 Frontend pages (`web/app`)

| Path | Decision | Notes |
|---|---|---|
| `page.tsx` (home, video + "Log In") | DELETE → redirect | Replace with a redirect to `/outliers/swing`. |
| `outliers/page.tsx` | REWRITE → `outliers/[strategy]/page.tsx` | Server component. Pre-fetches the list so each strategy has a shareable URL. |
| `outliers/client-page.tsx` | REWRITE | Remove the mock data, `console.log` calls and the two-button refresh. Fix the swapped axis labels (B2). |
| `analyze/[ticker]/page.tsx` | REWRITE → `analysis/[ticker]/page.tsx` | New layout from Phase 3. Remove "Add to Watchlist". |
| `analyze/[ticker]/client-page.tsx` | REWRITE | LSTM forecast replaced by the quant analysis. |
| `analyze/[ticker]/technical-indicators.tsx` | DELETE | Its "RSI" and "MACD" values are derived from the LSTM forecast, so they are not RSI or MACD (B8). |
| `analyze/[ticker]/news-section.tsx` | **ASK** | News and hype detection. It is not in the Phase 3 layout. |
| `api/auth/[...nextauth]/route.ts` | DELETE | Auth. |
| `auth/error/page.tsx` | DELETE | Auth. |
| `login/page.tsx` | DELETE | Auth. |
| `dashboard/page.tsx` | DELETE | Links to HFT, capitulation and portfolio. |
| `portfolio/*` (3 files) | DELETE | Portfolio. |
| `trading/hft/page.tsx`, `page-new.tsx` | DELETE | Trading. |
| `trading/quantitative/page.tsx` | DELETE | Trading. |
| `capitulation/*` (2 files) | DELETE | On your default delete list. |
| `demo/page.tsx` | DELETE | Ticker-search demo page. |
| `test-api/page.tsx` | DELETE | Debug page. |
| `layout.tsx` | REWRITE | Fonts, metadata, OG tags, manifest, footer disclaimer. Remove `SessionProvider`. |
| `providers.tsx` | REWRITE | Remove NextAuth. Keep a small query provider if needed. |
| `globals.css` | REWRITE | Ferrari tokens through Tailwind v4 `@theme`. |
| `favicon.ico` | REWRITE | New icon set (favicon, apple-touch, PWA 192/512, maskable). |
| *(new)* `methodology/page.tsx` | NEW | |
| *(new)* `manifest.ts`, `offline` fallback, `opengraph-image`, `not-found.tsx`, `error.tsx` | NEW | |

### 1.2 Frontend components, hooks, lib, types

| Path | Decision | Notes |
|---|---|---|
| `components/charts/plotly-scatter-plot.tsx` | REWRITE | Plotly.js is about 3.5 MB and would make a 90+ Lighthouse score very hard. Replace it with a hand-written SVG scatter. |
| `components/charts/scatter-plot.tsx` (Recharts) | DELETE | Replaced by the SVG scatter. |
| `components/charts/candlestick-prediction-chart.tsx` | DELETE | LSTM chart. A new SVG price and forecast chart replaces it. |
| `components/charts/plotly-candlestick-chart.tsx`, `prediction-chart.tsx`, `simple-line-chart.tsx` | DELETE | Unused, or LSTM only. |
| `components/ui/*` (shadcn: button, card, badge, table, tabs, skeleton, select, …) | KEEP + restyle | Restyle to the Ferrari tokens. Remove the ones that end up unused (dialog, dropdown, textarea, label, input if unused). |
| `components/error-card.tsx`, `loading-card.tsx` | REWRITE | Currently unused. Turn them into proper empty, loading and error states. |
| `components/toast-provider.tsx` | DELETE | Not needed after the refresh flow is simplified. |
| `components/nav-menu.tsx` | REWRITE | Currently unused. The new top nav has Outliers and Methodology. |
| `components/fair-value-card.tsx` | DELETE | Valuation (Black-Scholes). |
| `components/hype-warning-card.tsx`, `nasdaq-news-section.tsx` | **ASK** | Belong with news. |
| `components/behavioral-*.tsx` | DELETE | Behavioral/portfolio. |
| `components/analyze-stock-search.tsx`, `ticker-search.tsx` | **ASK** | A ticker search box would let users open `/analysis/ANY`. Useful, but not in your spec. I'd keep a simple one in the nav. |
| `hooks/use-performance-metrics.ts`, `use-auto-refresh.ts` | KEEP, then REWRITE | Remove the logs. Add a "data as of" time and pause polling when the tab is hidden. |
| `hooks/use-outliers.ts` | DELETE | Unused duplicate. |
| `hooks/use-ticker-info.ts` | REWRITE | Analysis header. |
| `hooks/use-prediction.ts`, `use-valuation.ts`, `use-orderbook.ts`, `use-hft-quotes.ts` | DELETE | |
| `lib/api.ts` | REWRITE | Down from about 60 methods to about 5. Remove the `console.log` of every response. Remove `executeTrade` and `hftSubmitOrder` (order-sending client code). |
| `types/next-auth.d.ts` | DELETE | |
| `types/index.ts`, `types/predictions.ts` | REWRITE | |
| `auth.ts`, `middleware.ts` | DELETE | NextAuth. The middleware only exists for auth. |
| `next.config.ts` | REWRITE | Remove the googleusercontent image host. Add security headers. |

### 1.3 Frontend tests and config

| Path | Decision | Notes |
|---|---|---|
| `__tests__/auth.test.tsx` | DELETE | |
| `__tests__/charts.test.tsx` | REWRITE | Test the new SVG charts. |
| `__tests__/ticker-search.test.tsx` | Depends on ASK | |
| `__tests__/use-auto-refresh.test.ts` | KEEP | |
| `__tests__/example.test.tsx` | DELETE | Placeholder. |
| `e2e/auth.spec.ts`, `dashboard.spec.ts`, `example.spec.ts`, `full-journey.spec.ts`, `analyze.spec.ts` | DELETE | |
| `e2e/outliers.spec.ts` | REWRITE | Currently asserts a redirect to `/login`. Becomes the outliers → analysis E2E test, with the API mocked. |
| `playwright.config.ts` | REWRITE | Chromium plus one mobile project in CI. |
| `package-lock.json` | DELETE | Both npm and pnpm lockfiles are committed. Keep pnpm only, since CI and Vercel use pnpm. |
| `package.json` | REWRITE | Remove `next-auth`, `plotly.js`, `react-plotly.js`, `recharts`, `sonner`, `lucide-react`, `date-fns` (if unused). Add `@tabler/icons-react` and `gsap`. Keep `@tanstack/react-query` (already installed but unused) or drop it. |
| `Dockerfile.dev`, `README.md` | REWRITE | |

### 1.4 Backend routers (`api/routers`)

| Router | Decision | Notes |
|---|---|---|
| `market.py` (`/market/outliers`, `/market/performance`, `/market/refresh`) | REWRITE | This is the real outlier read path. It moves into `outliers.py` as `GET /api/v1/outliers/{strategy}`, with validation, `as_of`, a market-state field, cache headers, and rank, direction and reason fields. |
| `outliers.py` (`/strategies`, `/{s}/info`, `POST /{s}/refresh`) | REWRITE | Merged with the above. The public `POST refresh` becomes rate-limited, or moves to a scheduler (see Q5). |
| `historical.py` | REWRITE | Prices are needed by analysis. They go into the analysis service. Its router is mounted with **no prefix** (`/{ticker}/historical` at the root under `/api/v1`), which catches paths it shouldn't. |
| `predictions.py` (`/predictions/{t}`, `/info/{t}`, `/search`) | **ASK** (LSTM part); REWRITE `/info` | `/search` is unreachable: `/info/{ticker}` and `/{ticker}` are declared first, so `/predictions/search` matches `/{ticker}` with ticker="SEARCH" (B6). |
| `news.py`, `nasdaq_news.py` | **ASK** | Not in the spec. They depend on NewsAPI, OpenAI, Anthropic, RSS, TextBlob and aiohttp. |
| `trading.py` | DELETE | Alpaca orders. |
| `hft.py` | DELETE | Order submission. |
| `portfolio.py` | DELETE | |
| `users.py` | DELETE | |
| `behavioral.py` | DELETE | |
| `capitulation.py` | DELETE | |
| `valuation.py` + inline `/valuation/{t}/fair-value` in `main.py` | DELETE | |
| `/api/v1/test-hype` in `main.py` | DELETE | Debug endpoint. |
| *(new)* `analysis.py` → `GET /api/v1/analysis/{ticker}` | NEW | Phase 3. |

### 1.5 Backend services, models, DB

| Path | Decision | Notes |
|---|---|---|
| `services/outlier_detection.py` | REWRITE | Thin wrapper. It will own caching, scheduling, rank, direction and reason. |
| `services/market_data.py` | KEEP, then REWRITE | Cache logic is reusable. Move the cache out of the source tree (`funda/cache`), add input validation, and remove the fake `search_tickers` list. |
| `services/predictions.py`, `markov_predictor.py` | **ASK** | LSTM and Markov. They pull in `torch` (and `tensorflow` in requirements). |
| `services/enhanced_news_service.py`, `nasdaq_news_service.py`, `advanced_hype_detector.py` | **ASK** | Belong with news. |
| `services/trading_service.py` | DELETE | |
| `services/behavioral_service.py` | DELETE | |
| `services/capitulation_detector.py`, `enhanced_capitulation_detector.py` | DELETE | |
| `services/black_scholes.py` | DELETE | Valuation. |
| *(new)* `services/quant_analysis.py` | NEW | Phase 3. Pure functions. |
| `models/behavioral_models.py` + tracked `__pycache__/*.pyc` | DELETE | |
| `db/models.py` (`PerfMetric`) | KEEP | Add `as_of` (price date) and `computed_at` columns. |
| `db/models_auth.py` (User, UserPreference, Watchlist, Alert) | DELETE | |
| `db/core.py` | REWRITE | Ignores `settings.DATABASE_URL` (B10). Uses `echo=True`, which logs every SQL statement in production. |
| `api/database.py` | REWRITE | Drop the auth imports and the `sys.path` hack. |
| `api/config.py` | REWRITE | Remove Alpaca, HFT, JWT, OpenAI, Anthropic, Polygon and FRED settings. Read `CORS_ORIGINS` from env. Default `DEBUG=False`. |
| `api/main.py` | REWRITE | Mount only outliers, analysis and health. Add the rate limiter and an error handler. |

### 1.6 Backend tests

| Path | Decision |
|---|---|
| `tests/conftest.py` | KEEP |
| `tests/test_main.py` | KEEP + update |
| `tests/test_market.py`, `test_outliers.py` | REWRITE (new endpoint shape) |
| `tests/test_predictions.py` | Depends on ASK |
| `tests/test_users.py` | DELETE |
| *(new)* `tests/test_quant_analysis.py`, `tests/test_analysis_endpoint.py` | NEW |
| root `test_api_endpoints.py`, `test_syntax.py` | DELETE (ad-hoc scripts) |

### 1.7 Engines and scripts

| Path | Decision | Notes |
|---|---|---|
| `funda/outlier_engine.py` | KEEP, then REWRITE | This is the real engine. It has bugs B3, B4 and B5, which must be fixed. |
| `funda/refresh_outliers.py` | KEEP + simplify | Thread plus status dict. Fine for a single worker. |
| `funda/SPS.py` (3,655-line Dash app) | **ASK** → DELETE | The old desktop-style dashboard. Superseded by the web app. |
| `funda/enhanced_features.py`, `train_lstm_model.py`, `fine_tuning_strategy.py`, `model_diagnostics.py`, `funda/model/*.pt` | Depends on the LSTM answer | |
| `funda/data/*.csv`, `funda/data/*.xlsx`, `funda/cache/*.csv`, `funda/billions.db` | DELETE | Generated data committed by accident. `*.db` is in `.gitignore`, but this file was committed anyway. |
| `funda/assets/*` (DePixel and Minecraft fonts, 2 MP4 videos, logos) | DELETE | Not referenced by the web app. |
| `outlier/Outlier_Nasdaq_{Scalp,Swing,Longterm}.py` | DELETE | Superseded by `outlier_engine.py`. They hard-code a Windows path to another project's `.env` and **print the API key** to stdout. |
| `outlier/cache/*`, `outlier/data/*` | DELETE | Stale CSVs from Sep 2025. Not read by any code. |
| `historical/data/*` (xlsx, keras, PPO zip) | **ASK** → DELETE | Unrelated research artefacts. |
| `alpaca_websocket_hft_manager.py`, `hft_quick_start.py`, `hft_trading_manager.py`, `hft_trading_manager_simple.py`, `hft_engine/` | DELETE | |
| `populate-test-data.py`, `run-populate.bat`, `test-refresh.bat` | DELETE | Not used by the app. |
| `create-env.bat` | DELETE + **ROTATE KEYS** | **Contains a committed Alpaca key pair** as fallback defaults (see S1). |
| `start-backend.{bat,sh}`, `start-frontend.{bat,sh}` | DELETE | Replaced by one `pnpm dev` at the root. |

### 1.8 Config, deploy, CI, docs

| Path | Decision | Notes |
|---|---|---|
| `requirements.txt` (root) | REWRITE → single source | Root and `api/` both have requirements files and they disagree (the root one lists Dash and tensorflow). Keep one, `api/requirements.txt`. The root file will `-r` it. |
| `api/requirements.txt` | REWRITE | Remove tensorflow, TA-Lib, openai, anthropic, aiohttp, feedparser, textblob, bs4, python-jose, passlib, email-validator, fredapi, alembic, openpyxl, lxml. Remove torch if the LSTM goes. Add `xgboost`, `slowapi` (rate limiting), and `pandas-market-calendars` or a small built-in NYSE calendar. |
| `api/requirements-dev.txt` | KEEP + trim | |
| `docker-compose.yml` | REWRITE | |
| `api/Dockerfile.dev` | REWRITE | Add a production Dockerfile for Railway and Render. |
| `vercel.json` | REWRITE | Remove `jonel/webapp` and set `rootDirectory` handling. |
| `railway.json`, `render.yaml`, `Procfile`, `runtime.txt` | REWRITE | `render.yaml` declares a Postgres `databases:` entry that nothing uses. |
| `.github/workflows/test.yml`, `lint.yml`, `deploy.yml` | REWRITE | |
| `.github/workflows/python-app.yml` | DELETE | A duplicate legacy workflow on Python 3.10 that installs the root requirements (Dash and tensorflow). |
| `.gitignore` | REWRITE | **It ignores `test_*.py`**, so every new backend test file would be silently left out of git (B11). |
| `.pre-commit-config.yaml`, `.flake8`, `pyproject.toml`, `pytest.ini` | KEEP + tidy | `pytest.ini` and `pyproject.toml` both configure pytest, and `pytest.ini` wins. Keep one. |
| `README.md`, `CHANGELOG.md` | REWRITE | |
| `PLAN.md`, `SYSTEM_ARCHITECTURE_FLOWCHART.md`, `SYSTEM_ARCHITECTURE_FLOWCHART.html`, `BILLIONS_SYSTEM_DRAWIO_FIXED.xml`, `API_TESTING_RESULTS.md`, `SCREENSHOTS.md`, `api_docs.html`, `QUICKSTART.md`, `CREATE_ENV_FILE.md`, `FAQ.md` | DELETE | Stale. They describe deleted features. README and `/methodology` replace them. |
| `CONTRIBUTING.md`, `SECURITY.md` | REWRITE (short) | |
| `LICENSE` | KEEP | |
| *(new)* `.env.example` (root) and `web/.env.example` | NEW | Neither exists today. |
| *(new)* root `package.json` with `pnpm dev` running API and web together | NEW | |

---

## 2. Reused as is vs rewritten

**Reused (logic kept, code cleaned):**
- `STRATEGIES` table and the z-score outlier rule (`|z| > 2` on either axis) in `funda/outlier_engine.py`.
- `PerfMetric` table.
- The refresh-in-background pattern (`funda/refresh_outliers.py`).
- The cache-to-disk idea in `MarketDataService`.
- `useAutoRefresh` hook (5-minute interval).
- shadcn UI primitives, restyled.

**Rewritten:**
- Outlier fetch and ranking: fixes B3, B4 and B5, and adds rank, direction and a one-line reason.
- The outliers page: no mock data, an SVG scatter, a table and per-strategy URLs.
- The whole analysis page and endpoint (Phase 3 quant method, new `quant_analysis.py`).
- Layout, theming, metadata, PWA, deploy configs, CI, docs.

**New:** `quant_analysis.py`, the `/analysis/{ticker}` endpoint, `/methodology`, the design tokens, the PWA manifest and icons.

---

## 3. Bugs and half-finished work in the paths we keep

| # | Where | Problem |
|---|---|---|
| ~~B1~~ | `outlier_engine._fetch_batch` | **Retracted.** I claimed Yahoo reads `period="Nd"` as calendar days. A live check with yfinance 1.7 returned exactly N trading rows (`68d` → 68 rows), so the old engine got enough data. The empty table came from B4/B5 instead: no key was found, so it fell back to 16 tickers, and almost nothing passes \|z\| > 2 in a group that small. |
| **B2** | `outliers/client-page.tsx` | Axis labels are swapped against the engine. The engine stores `metric_x` = the *longer* window (scalp: 1m) and `metric_y` = the *shorter* one (1w). The UI labels X as 1-week and Y as 1-month. The strategy dropdown text is also inconsistent. |
| **B3** | `outlier_engine._calc_pct` | `ser.iloc[-lookback]` is `lookback-1` bars back. That is an off-by-one: a "21-day" return is really 20 days. |
| **B4** | `outlier_engine` | With no `ALPHA_VANTAGE_API_KEY`, the universe falls back to 16 mega-caps. A z-score across 16 names rarely passes 2, so in practice there are no outliers. With a key, it calls `yf.Ticker(t).info` once per NASDAQ ticker (around 3,000+ calls), with sleeps. That takes hours and Yahoo will rate-limit it. See Q4. |
| **B5** | `outlier_engine` | It reads `.env` from `outlier/.env`, not the project `.env`. |
| **B6** | `predictions.py` | Route order makes `/predictions/search` unreachable. |
| **B7** | `outliers/client-page.tsx` | It silently shows hard-coded mock data, with invented z-scores for real tickers, as if it were real. A small badge is the only hint. This must go: never show invented data. |
| **B8** | `technical-indicators.tsx` | The "RSI" and "MACD" values come from the LSTM's forecast, not from price history. They are mislabelled. |
| **B9** | `predictions.py` | The LSTM confidence bands are heuristic, not statistical intervals. There is no out-of-sample validation anywhere. |
| **B10** | `db/core.py` | Hard-codes `sqlite:///<repo>/billions.db` and ignores `DATABASE_URL`. `echo=True` logs every SQL statement. |
| **B11** | `.gitignore` | `test_*.py` is ignored. New backend tests would not be committed. |
| **B12** | `lib/api.ts`, hooks | `console.log` on every request and response, in production too. |
| **B13** | `main.py` | `DEBUG=True` by default. CORS origins are hard-coded to localhost, with `allow_credentials=True` and wildcard methods and headers. |
| **B14** | `outliers.py` | A public `POST /outliers/{s}/refresh` with no rate limit starts hours of Yahoo scraping. Anyone could trigger it. |
| **B15** | `historical.py` | The router has no prefix, so `/api/v1/{ticker}/historical` sits at the API root. No ticker validation. Raw exception text goes back to the client. |
| **B16** | Repo | `api/models/__pycache__/*.pyc` and `funda/billions.db` are committed despite `.gitignore`. |
| **B17** | `outliers/page.tsx`, `analyze/page.tsx` | Each calls `await auth()` and throws the result away. They both link "Back to Dashboard", a page we are deleting. |

### Security

| # | Problem |
|---|---|
| **S1** | `create-env.bat` contains an **Alpaca API key ID and secret** as hard-coded fallbacks. They look like paper-trading keys (`PK…`). Deleting the file in Phase 1 does **not** remove them from git history. **You should revoke them in the Alpaca dashboard now.** Rewriting git history is your call (Q7). |
| **S2** | `auth.ts` falls back to the secret `"development-secret-change-in-production"`. This goes away with auth. |
| **S3** | `config.py` has the default `SECRET_KEY="your-secret-key-change-in-production"`. This goes away with JWT. |

### Order-path inventory (to prove it's gone after Phase 1)

Code that can send or simulate orders today: `api/routers/trading.py`, `api/routers/hft.py`, `api/services/trading_service.py`, `hft_*.py`, `alpaca_websocket_hft_manager.py`, `hft_engine/` (C++ `order_executor.h`), `web/lib/api.ts` (`executeTrade`, `hftSubmitOrder`, `hftClearAllOrders`), `web/app/trading/**`. Phase 1 deletes all of these, then runs `grep -rniE "execute|order|alpaca|broker|buy|sell"` and puts the output in the report.

---

## 4. Data source (what's possible, honestly)

- **Prices:** Yahoo Finance through `yfinance`. It's unofficial, has no SLA, gets rate-limited, and intraday quotes can be delayed. Daily OHLCV is reliable enough for this tool.
- **Level 2 / order book:** **not available** from yfinance (and not from Alpha Vantage's free tier). The handbook's order-book imbalance and mid-price cards will show "Not available with current data source", with an explanation. Nothing will be faked.
- **Bid/ask:** `Ticker.info` sometimes has a top-of-book `bid` and `ask`, but they are often 0 or stale outside market hours. I won't compute mid-price from them.
- **Forecast horizon:** daily bars, so the horizon is **1 trading day**. The handbook says taking strategies need horizons long enough that expected return beats round-trip cost. The cost check compares the 1-day forecast with an editable cost constant (default **10 bps round trip**; tell me if you want a different default).
- **Market hours:** I'll use the NYSE calendar for open, closed and holiday states, plus "data as of <last close>".

---

## 5. Design system notes (from `design-md/ferrari/DESIGN.md`)

- Tokens: canvas `#181818`, elevated `#303030`, Rosso Corsa `#da291c` (used sparingly), 8px spacing ladder (`xxxs` 4 … `super` 128), radius 0 on buttons and cards, pill shape only on badges, display weight 500, uppercase tracked button and nav labels.
- **Font:** FerrariSans is proprietary. DESIGN.md suggests Inter. Your rule is "closest free Fontshare match with a commercial licence". My pick is **General Sans** (Fontshare, ITF Free Font License, commercial use OK). It is a neutral grotesk with a 500 weight that matches FerrariSans' restrained display style. **Switzer** is the backup. I'll self-host the woff2 files and note the substitution in `web/DESIGN.md`. Numbers use tabular figures.
- **WCAG AA conflicts in the source palette.** These need small, documented adjustments:
  - `muted #666666` on `#181818` ≈ 3.1:1. That fails AA for body text, so it's only for non-text and large text. Small secondary text uses `body #969696` (≈ 5.9:1).
  - `primary #da291c` as *text* on `#181818` ≈ 3.2:1, which fails for small text. Red stays a fill (white on red ≈ 5.5:1 passes) and an accent, never small red text.
  - `semantic-success #03904a` as text on canvas ≈ 4.2:1, which fails for small text.
  - **Direction colours (up/down):** Ferrari has no up/down pair. I'll add two tokens, a lightened success green and a lightened warning red, each checked at ≥ 4.5:1 on canvas. Direction will also be shown with an icon and a word, never colour alone. These are the only additions to the palette, and they'll be listed in `web/DESIGN.md`.
- The cinematic hero photography doesn't apply to a data tool. Instead, the "spec-cell" and "race-position" number styles drive the big metrics: the outlier score and signal strength.
- Icons: Tabler (`@tabler/icons-react`), outline style only. Motion: GSAP for list reveals and number count-ups, turned off under `prefers-reduced-motion`.
- The `impeccable` skill isn't installed here. I'll do the same audit by hand: hierarchy, spacing, type, accessibility, and empty, loading and error states.

---

## 6. Phase 3 notes (quant method)

- Data: about 2 years of daily closes, so roughly 500 log returns. The test split is 25%, about 125 observations. I'll flag "small sample" below 100 test observations.
- Models: AR(1) closed-form OLS; `XGBRegressor(max_depth=3)` on weekday one-hot plus rolling-direction features; online learner; stacked meta-learner with non-negative weights and a free bias (`scipy.optimize.nnls` on centred data plus an intercept). The stack is fit on out-of-fold base predictions to avoid leakage.
- **`PassiveAggressiveRegressor` is deprecated in recent scikit-learn** (1.8 points to `SGDRegressor` with a passive-aggressive learning rate). I'll check the installed version. If it's deprecated, I'll use the documented `SGDRegressor` equivalent and label it "Passive-Aggressive (online)" in the UI.
- Validation: a time-ordered 75/25 split; expanding and rolling walk-forward; a Monte Carlo uniform-random-direction baseline (for example 1,000 draws), reported as a percentile and p-value of the model's out-of-sample return.
- Signal strength = `tanh(ŷ / σ_ŷ)`. Raw `tanh` of a daily log return around 0.001 is always about 0, so the gauge would never move. I'll scale by the forecast's own out-of-sample spread and document it on `/methodology`.
- Output is information only: no sizes, leverage, entries, exits or buy/sell wording. The handbook's sizing formulas become the bounded strength gauge only.

---

## 7. Questions for you (blocking Phase 1)

| # | Question | My recommendation |
|---|---|---|
| **Q1** | **News and hype detection** (`news.py`, `nasdaq_news.py`, 3 services, `news-section.tsx`, `hype-warning-card.tsx`, `nasdaq-news-section.tsx`). It isn't in your keep list or your delete list. | **Delete.** It's not in the Phase 3 layout, it needs paid API keys, and some of it calls LLMs. |
| **Q2** | **LSTM 30-day forecast and Markov predictor** (`predictions.py`, `markov_predictor.py`, `funda/model/*.pt`, the training scripts, `torch`). | **Delete.** Phase 3 replaces it with validated models. `torch` alone makes the Railway/Render image about 2 GB and slow to start. |
| **Q3** | `funda/SPS.py` (old Dash app) and `historical/data/*` (research files). | **Delete.** |
| **Q4** | **Outlier universe.** Today it's either Alpha Vantage (needs a key; with `.info` screening, hours per refresh) or 16 hard-coded mega-caps. | Use the free **NASDAQ Trader symbol file** (no key). Screen liquidity and market cap from a bulk `yf.download` of volume plus close (minutes, not hours). Cap the universe at about the top 1,000 by dollar volume. Keep Alpha Vantage as an optional source. |
| **Q5** | **Refresh model.** Who recomputes outliers? | The backend recomputes on a schedule (every 30 minutes in market hours, once after the close) and on startup if the cache is stale. The public "refresh" button only refetches the cache and never starts a scrape. The frontend polls every 5 minutes. |
| **Q6** | Ticker search box in the nav (opens `/analysis/{ticker}` for any symbol). | **Keep a minimal one.** Tell me if you'd rather analysis be reachable only from outliers. |
| **Q7** | Committed Alpaca keys (S1). | Revoke them now. A history rewrite (`git filter-repo`) is optional, and it needs a force-push, which I won't do without your go-ahead. |
| **Q8** | Default round-trip cost for the cost check. | 10 bps, shown and editable in the UI. |

---

## 8. Changes made after the audit

- **Liquidity filter instead of market cap.** The old engine fetched `Ticker.info` once per symbol to read market cap (thousands of slow calls). The new engine screens on **median daily dollar volume over 20 sessions**, using the same bulk download. Floors: scalp $25M, swing $15M, longterm $50M. It also skips stocks under $3 and drops symbols with no close in the last 3 sessions. The universe is capped at the 1,000 most liquid names.
- **Returns use exact trading-day windows:** `P[-1] / P[-1-N] - 1`. This fixes B3.
- **One download feeds all three strategies.** A full refresh is about 4,000 symbols in batches of 400.
- **Refresh is scheduled, not public.** It runs at startup when there is no data, every 30 minutes while the market is open, and once 20 minutes after each close. `POST /refresh` is gone (B14).
- **Ranking:** score = √(z_x² + z_y²). Direction comes from the axes past the threshold, giving up, down or mixed. The reason is one plain sentence.
