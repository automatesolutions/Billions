# BILLIONS

BILLIONS finds US stocks that are moving unusually far from the rest of the market. For each one, it shows a tested quant analysis: signal, edge, risk, and whether any of it beats chance.

It's a market data web app with an optional paper trading bot. **Information only. Not financial advice. This tool never places real trades.** The bot trades an Alpaca paper account only, and it refuses to start with any other trading URL.

![Swing outliers](docs/screenshots/outliers-swing.png)

| Quant | Lead-lag research |
|---|---|
| ![Quant strategy picker](docs/screenshots/quant-picker.png) | ![Lead-lag research](docs/screenshots/lead-lag.png) |

![Paper bot](docs/screenshots/bot.png)

| Outliers (phone) | Stock analysis (phone) |
|---|---|
| ![Outliers on a phone](docs/screenshots/outliers-mobile.png) | ![Stock analysis on a phone](docs/screenshots/analysis-mobile.png) |

Full analysis page: [docs/screenshots/analysis-desktop-full.png](docs/screenshots/analysis-desktop-full.png). Earlier desktop captures: [outliers](docs/screenshots/outliers-desktop.png), [analysis](docs/screenshots/analysis-desktop.png).

## Pages

| URL | What it shows |
|---|---|
| `/` | Home page: a hero, the live board (top five swing outliers and scan stats), the three strategies, and a ticker lookup |
| `/outliers/scalp`, `/outliers/swing`, `/outliers/longterm` | Every liquid NASDAQ stock on a scatter of two return windows, plus a ranked list of the ones more than 2 standard deviations from the group |
| `/analysis/{TICKER}` | Signal, price and next-session forecast, out-of-sample edge, risk, model comparison, validation, cost check, and the limits of all of it |
| `/quant`, `/quant/{TICKER}?interval=1m\|1h\|4h\|1d` | Pick a stock and a bar interval. Shows the action for the next bar (long, short or flat, with a tanh-sized position and the checks behind it), the live record of every past signal scored against what happened, the backtest after costs, and the worked calculation |
| `/lead-lag` | Lead-lag research: analyze any two stocks or ETFs, a ranked table of the day's best pairs, pair details, your watchlist, and scans |
| `/bot` | The paper bot: pick a strategy and risk limits, start, stop, or kill it, and watch the account, positions, orders, lead-lag timings, and decision log |
| `/methodology` | How every number is made, in plain words |

The header on every page has a ticker search, so you can open any NASDAQ stock's analysis from anywhere.

Each page has its own URL, title and link preview, and works with the browser's back and forward buttons. The app can be installed to a phone's home screen (web app manifest and icons), and it shows an offline page without a connection.

## Features

### Home

- Hero with the brand line and links into today's outliers and the methodology.
- **Live board:** top five swing outliers, scan stats (universe size, how many cleared the liquidity floor, how many are outliers), and the data-as-of time with NYSE market state.
- Cards for the three strategies (Scalp, Swing, Long-term) with their return windows.
- Ticker lookup that opens `/analysis/{TICKER}`.

### Outlier scanner

Scans about 4,000 NASDAQ common stocks (free NASDAQ Trader list), keeps names above a dollar-volume floor and at least $3, then ranks the **1,000 most traded**. For each strategy it computes two trailing returns, turns each into a z-score against the group, and flags a stock when either z-score is beyond ±2. Score = √(z₁² + z₂²). Direction is up, down, or mixed.

| Strategy | Short window | Long window | Liquidity floor | Horizon |
|---|---|---|---|---|
| Scalp | 1 week (5 days) | 1 month (21 days) | $25M / day | Days to weeks |
| Swing | 1 month (21 days) | 3 months (63 days) | $15M / day | Weeks to months |
| Long-term | 6 months (126 days) | 1 year (252 days) | $50M / day | Months to a year |

Each outliers page shows an SVG scatter of every scanned stock, a ranked list with score, direction and a one-line reason, data-as-of time, and market state. The page refreshes every 5 minutes. The scan itself runs at API startup, every 30 minutes while the market is open, and once after the close.

### Stock analysis (`/analysis/{TICKER}`)

Per-ticker read of about two years of daily closes (quant handbook in `docs/reference/`):

- **Signal:** direction and a strength score from −1 to +1 (tanh of forecast ÷ 0.1 × typical daily move; below 0.15 counts as neutral).
- **Price chart** and next-session forecast from the stacked model.
- **Edge:** out-of-sample expected value and Sharpe; in-sample numbers are labeled as such.
- **Risk:** drawdown and context for a typical day.
- **Models:** AR(1) regime (mean reversion vs momentum), XGBoost (depth 3), online Passive-Aggressive, and a stacked blend with non-negative weights.
- **Validation:** 75/25 time-ordered split, expanding and rolling walk-forward, comparison with 1,000 random up/down strategies.
- **Cost check:** editable round-trip cost (default 10 bps); uses the live Alpaca spread when keys are set.
- **Microstructure panel:** live best bid/ask from Alpaca IEX (mid, size-weighted mid, spread, imbalance), refreshing every 15 s while the market is open. Without keys the panel says not available; nothing is estimated.
- No look-ahead: every feature for day *t* uses data up to *t−1* only (unit-tested). Output is information only: no sizes, orders, or buy/sell instructions. Thin history (under about a year) gets price and risk only.

### Quant (`/quant`, `/quant/{TICKER}`)

Same stacked pipeline at a bar interval you choose — **1m, 1h, 4h, or 1d** — following [`docs/quant_trading_comprehensive_guide.pdf`](docs/quant_trading_comprehensive_guide.pdf):

- Intraday bars use time-of-session instead of weekday. The 1-minute strategy is flat overnight.
- **Position** = tanh(forecast ÷ (0.1 × bar volatility)), as a share of your maximum. Every change pays half a round trip (live Alpaca spread or 10 bps).
- **Action** is flat unless every check passes: beats 95% of random strategies, walk-forward EV positive, profitable after costs, live record not failing, signal strong enough.
- **Is it working?** Every signal is saved when its bar closes and scored when the next one closes. Live hit rate and EV are compared with the backtest's predicted range. Below the range → "Not working" and the action stays flat. Before 20 scored signals it says it is too early to tell.
- **Track every bar** keeps recording in the background with no page open (up to `QUANT_MAX_WATCHES` pairs).
- Worked calculation, backtest-after-costs chart, and a live record tape. It places no orders; sizes and actions are model output for information.

### Lead-lag research (`/lead-lag`)

Research any two stocks or ETFs: who leads, by how long, accuracy with sample counts, 5/20/60-day windows, out-of-sample check, and a confidence label. Best-pairs table, pair detail charts, watchlist, suggested candidates (sector peers, ETF holdings, index ETF stand-ins for futures), timed scans, and a daily post-close rescan. **Use on the Bot page** opens `/bot` with that pair's settings. Full walkthrough below.

### Paper bot (`/bot`)

Optional Alpaca **paper** trading only (moving-average crossover or lead-lag). Risk limits, kill switch, account/positions/orders, lead-lag timing and slippage log. Off unless `BOT_ENABLED=true`. Full walkthrough below.

### Methodology, PWA, and hardening

- **`/methodology`:** plain-language explanation of every number, with the limits of the data.
- **PWA:** installable on a phone home screen (manifest + icons); offline page and service worker when there is no connection.
- Per-page titles, canonical URLs, and Open Graph previews; 404 and error pages.
- Per-IP rate limits on the API; JSON error responses; Ferrari-based design system in [`web/DESIGN.md`](web/DESIGN.md) (General Sans, Tabler icons, GSAP with reduced-motion support, WCAG AA contrast).

## How it works

**Outliers.** Universe → liquidity screen → top 1,000 by median dollar volume → two return windows per strategy → z-scores → flag beyond ±2σ. Background scan at startup, every `REFRESH_INTERVAL_MINUTES` while the market is open, and once after each close. Pages poll every 5 minutes.

**Analysis** follows the quant handbook in `docs/reference/`, using two years of daily closes:

- Log returns and lagged, sign, rolling-mean and weekday features. Every rolling window is shifted by one day, and a unit test proves that no feature uses data from the day it forecasts.
- Four models: AR(1), XGBoost (depth 3), an online Passive-Aggressive learner, and a stacked blend with non-negative weights.
- Validation: a 75/25 time-ordered split, expanding and rolling walk-forward, and a comparison with 1,000 random up/down strategies.
- Headline numbers are out-of-sample. In-sample numbers are labeled as such.
- Output is information only: direction, a bounded strength score, risk numbers and plain caveats. It never gives sizes, orders or buy/sell instructions.

**Quant** runs the same pipeline at the interval you choose (1 minute, 1 hour, 4 hours or daily), following [`docs/quant_trading_comprehensive_guide.pdf`](docs/quant_trading_comprehensive_guide.pdf):

- Intraday bars use a time-of-session feature in place of the weekday. The 1-minute strategy is flat overnight.
- Position = tanh(forecast ÷ (0.1 × bar volatility)), as a share of your maximum. Every change in position pays half a round trip; the round trip is the live Alpaca spread when keys are set, otherwise 10 bps.
- The action is flat unless every check passes: it beats 95% of random strategies, walk-forward expected value is positive, it is profitable after costs, the live record isn't failing, and the signal is strong enough.
- **Is it working?** Every signal is saved when its bar closes and scored when the next one closes. The live hit rate and expected value are compared with the range the backtest predicts for that many signals. Below the range, the verdict is "Not working" and the action stays flat. "Track every bar" keeps recording in the background with no page open (up to `QUANT_MAX_WATCHES` pairs).
- It places no orders. The sizes and actions are a model's output, for information.

Details: [`/methodology`](web/app/methodology/page.tsx) and [`docs/REFACTOR_PLAN.md`](docs/REFACTOR_PLAN.md).

## Data source and its limits

- Prices for outliers, analysis, and Quant come from **Yahoo Finance** through `yfinance`, an unofficial library. Data can be delayed, missing or wrong, and there is no service guarantee.
- **Top of book only.** With free Alpaca keys (`ALPACA_API_KEY`, `ALPACA_SECRET_KEY`), the Microstructure panel shows the live best bid and ask from the IEX exchange: mid-price, spread and imbalance. IEX is one venue, so its spread is wider than the whole market's. There is no full depth (level 2). Without keys the panel says "Not available with current data source"; nothing is estimated. The same keys place paper orders when the bot is on, and supply minute bars when `LEAD_LAG_SOURCE=alpaca`.
- **Polygon** (optional) supplies bars for the moving-average bot strategy and, by default, lead-lag research. The free plan is 5 requests a minute and delayed by about 15 minutes.
- **Daily bars on the analysis page.** Its forecast horizon is one trading day. The Quant page also uses Yahoo's 1-minute (last 7 sessions) and hourly (last 2 years) bars; 1-minute bars arrive up to a minute late.
- Stocks with less than about a year of history get price and risk only, with a note that the models need more data.

## Paper bot and lead-lag research

The paper bot trades an Alpaca **paper** account, never a live one. `api/services/paper_broker.py` checks that the trading URL is exactly `https://paper-api.alpaca.markets` before it does anything, and again before every request. Any other URL, including a look-alike or a live endpoint, stops the bot before it starts.

### 1. Get free keys

- **Alpaca paper keys.** Sign up at [alpaca.markets](https://alpaca.markets), switch to your **Paper** account in the dashboard, and generate API keys there. Paper keys come with a simulated balance, free real-time IEX data, and historical minute bars.
- **Polygon key (optional).** Sign up at [polygon.io](https://polygon.io) and copy your key from the dashboard. The free plan allows 5 requests a minute, has minute bars (not second bars), and its data is delayed by 15 minutes. You can skip Polygon: set `LEAD_LAG_SOURCE=alpaca` and research uses Alpaca's minute bars instead, which allow about 200 requests a minute. The moving-average strategy needs Polygon.

### 2. Set up `.env`

Add these lines to `.env` in the repo root. Never commit this file; it's in `.gitignore`.

```bash
ALPACA_API_KEY=your-paper-key-id
ALPACA_SECRET_KEY=your-paper-secret
POLYGON_API_KEY=your-polygon-key        # optional; skip if LEAD_LAG_SOURCE=alpaca
BOT_ENABLED=true
BOT_ADMIN_TOKEN=a-long-random-string     # needed whenever anyone but you can reach the API
LEAD_LAG_SOURCE=alpaca                   # or polygon (default)
```

Restart the API after you change `.env`. The Bot page tells you if the bot is off, the token is missing, or the trading URL isn't the paper URL. On Cloud Run, set the same names on the `billions-api` service (see [Deploy](#deploy)); a local `.env` file is not uploaded.

### 3. Choose a strategy

**Moving-average crossover.** Every `poll_seconds`, the bot loads Polygon bars (daily or 1-minute) for each ticker. It buys when the short average crosses above the long one, and sells the whole position when it crosses back below. Only a fresh cross counts, so starting the bot halfway through a trend doesn't buy late. On Polygon's free plan the bars are 15 minutes old.

**Lead-lag.** Some tickers react a few seconds after a related one, such as NVDA and AMD, or SPY and QQQ. The bot listens to every trade on a WebSocket stream and keeps a few seconds of prices for each ticker. When the leader moves more than the threshold within the lookback window and the follower hasn't moved yet, the bot buys the follower on an up move. On a down move it skips, or shorts only if you turn shorting on. It exits on take profit, on stop loss, or when the max wait passes without the follower following. After each leader move, a follower stays quiet for the cooldown, so one move counts once.

- **Stream.** The default is Alpaca's free real-time IEX stream. IEX is one exchange and a small share of all volume, so some moves show up late or not at all. Polygon's stream is used only when `POLYGON_REALTIME=true`, for a paid real-time plan. With Polygon's delayed free plan the bot refuses to start the stream and logs a warning, because a 15-minute delay makes a seconds-level strategy meaningless.
- **Timings.** Every lead-lag trade records when the leader's move crossed the threshold, when the bot decided, when the order went out, and when it filled. The Bot page shows leader-to-signal, signal-to-fill, and leader-to-fill times, plus slippage: the fill compared with the follower's price at the signal. If slippage and delay eat most of your take profit, the edge isn't real for you.

**Risk limits for both strategies.** Each entry is sized in whole shares to stay under the dollar limit per position. Entries stop when the daily order limit is reached; exits are still allowed, so a position is never stuck open. The **kill switch** blocks every new order first, then stops the bot and cancels open orders. Tick "Also close every position" to close your positions too. The bot trades only while the market is open, using `api/services/market_calendar.py`.

Paper fills aren't real fills. Alpaca fills paper orders against the quote, with no place in the queue and no market impact, so real results would be worse.

### 4. Use the Lead-Lag Analysis page

- **Analyze a pair.** Enter any two stocks or ETFs. The page measures who leads and by how long (cross-correlation at each lag), accuracy with its sample count, the average follower move, the average lag, accuracy over the last 5, 20, and 60 trading days, an out-of-sample check, and a confidence label.
- **Best pairs today.** A sortable, filterable table of the latest research. Select a pair to see the leader and follower moves around signal events, accuracy over time, and how long the follower took. **Use on the Bot page** opens the Bot page with that pair's lead-lag settings filled in.
- **Watchlist and candidates.** Add the pairs you care about. Suggested candidates are sector peers, ETFs with their top holdings, and index ETFs that stand in for index futures (SPY, QQQ, IWM, and DIA for ES, NQ, RTY, and YM), which the free plans don't include. The suggestion list is kept by hand in `api/services/lead_lag_research.py`, because the symbol list in `universe.py` has no sector data.
- **Scans.** Before a scan starts, the page tells you how many requests it needs and how long that takes at your plan's rate limit. The 45 suggested pairs use 40 tickers, about 80 requests on Polygon's free plan, so a full scan takes about 16 minutes there and a few minutes on Alpaca.
- **Daily research.** With `LEAD_LAG_SCHEDULER=true` (the default), the API rescans your watchlist and the candidates 20 minutes after each trading day's close and stores the results under that date, so you can see each pair's history. Set it to `false` to turn it off.

### What the accuracy percentage means, and its limits

"68% (412 events)" means: in the history tested, the leader moved at least the threshold within the lookback 412 times, and in 68% of those the follower's move over the next max wait went the same way. A coin flip scores about 50%. The page always shows the count, because 68% of 20 events is weak evidence and 68% of 2,000 is strong evidence.

- **Out of sample.** Which ticker leads is decided on the older 70% of days only. Accuracy is then measured on the newest 30% with that same choice. A pair that scores well on the older days and close to 50% on the recent ones was a pattern in hindsight.
- **Confidence.** *Strong* needs at least 60% overall, in sample, and out of sample, 100 or more events, a less than 1% chance of being luck, and similar accuracy over the last 5, 20, and 60 days. *Moderate* needs at least 55% overall and out of sample, with less than a 5% chance of luck. *Not enough data* means fewer than 30 events, or fewer than 10 on the recent days.
- **Resolution.** On minute bars, a lead of a few seconds is invisible: both moves land in the same minute. Minute-bar research tells you whether a relationship exists at the scale of minutes. Testing a seconds-level lead needs second bars (`POLYGON_SECOND_BARS=true` on a paid plan), or the bot's own timing log.
- **It's about direction, not profit.** A follower that goes your way by 0.01% counts as a hit, but it won't cover the spread. Check the average follower move against your costs.
- **The past isn't the future.** Relationships between instruments change, and other traders find the same ones. Accuracy measured on past prices doesn't guarantee future results.

### 5. Run the bot

```bash
pnpm dev                              # API on :8000, web on :3000
# open http://localhost:3000/lead-lag  research a pair, then select "Use on the Bot page"
# open http://localhost:3000/bot       check the settings and risk limits, then select "Start the bot"
```

Run the bot on your own computer, or on a server with exactly one always-on instance. On Cloud Run, set the minimum and maximum instances to 1. Otherwise the instance can shut down when it's idle, and each copy would run its own bot.

## Run it locally

You need Node 20+, pnpm 9+ and Python 3.12+.

```bash
pnpm install     # root dev tools
pnpm setup       # web dependencies + Python venv in .venv
pnpm dev         # API on http://localhost:8000, web on http://localhost:3000
```

Or, with Docker: `docker compose up`.

The first outlier scan takes about 2 minutes after the API starts. Until then, the page says there is no data yet.

## Environment variables

Backend (`.env` in the repo root; see [`.env.example`](.env.example)):

| Name | Default | Purpose |
|---|---|---|
| `CORS_ORIGINS` | `http://localhost:3000,http://127.0.0.1:3000` | Allowed frontend origins, comma-separated |
| `DATABASE_URL` | `sqlite:///./data/billions.db` | Outlier cache. A PostgreSQL URL also works. |
| `OUTLIER_SCHEDULER` | `true` | Run the background outlier scan |
| `REFRESH_INTERVAL_MINUTES` | `30` | Scan interval while the market is open |
| `RATE_LIMIT_DEFAULT` / `RATE_LIMIT_ANALYSIS` | `120/minute` / `20/minute` | Per-IP limits |
| `ALPHA_VANTAGE_API_KEY` | empty | Optional second source for the symbol list |
| `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` | empty | Optional. Live best bid/ask, paper bot orders, and Alpaca minute bars for lead-lag |
| `QUANT_SCHEDULER` | `true` | Record signals for tracked Quant pairs in the background |
| `QUANT_MAX_WATCHES` | `20` | Most (ticker, interval) pairs that can be tracked |
| `ALPACA_TRADING_URL` | `https://paper-api.alpaca.markets` | Paper trading endpoint. Any other value stops the bot from starting. |
| `POLYGON_API_KEY` | empty | Optional. Polygon bars for the moving-average strategy and lead-lag research |
| `POLYGON_SECOND_BARS` / `POLYGON_REALTIME` | `false` / `false` | Turn on only if your Polygon plan includes second bars or real-time data |
| `BOT_ENABLED` | `false` | Turn on the paper bot endpoints and page |
| `BOT_ADMIN_TOKEN` | empty | When set, every bot request needs it (the Bot page asks for it) |
| `BOT_MAX_POSITION_DOLLARS` / `BOT_MAX_ORDERS_PER_DAY` | `1000` / `20` | Default risk limits; you can change them per run on the Bot page |
| `LEAD_LAG_SCHEDULER` | `true` | Rescan the lead-lag watchlist and candidates after each trading day's close |
| `LEAD_LAG_SOURCE` | `polygon` | Research data: `polygon` or `alpaca` |
| `LEAD_LAG_DAYS` | `60` | Trading days of history per pair |

Frontend (`web/.env.local`; see [`web/.env.example`](web/.env.example)):

| Name | Purpose |
|---|---|
| `NEXT_PUBLIC_API_URL` | Backend URL as the browser sees it |
| `API_URL` | Optional backend URL for the Next.js server (defaults to the above) |
| `NEXT_PUBLIC_SITE_URL` | Public site URL, for canonical links and link previews |

## Tests and checks

```bash
pnpm test                        # pytest (271 tests) + vitest (22 tests)
pnpm lint                        # flake8 + eslint
pnpm --dir web typecheck
pnpm --dir web test:e2e          # Playwright: outliers -> analysis, desktop and phone, against a mock API
```

CI (`.github/workflows/ci.yml`) runs backend lint and tests, then frontend lint, types, tests and build, then the E2E suite.

Lighthouse on the production build scores 95+ on mobile and 100 on desktop for Performance, Accessibility, Best Practices and SEO. See [`docs/lighthouse/README.md`](docs/lighthouse/README.md).

## Deploy

### Google Cloud Run (current)

The live app runs as two Cloud Run services in project `billions-510913`, region `us-central1`:

| Service | Image | URL |
|---|---|---|
| `billions-web` | `…/billions/web:latest` | https://billions-web-106830243219.us-central1.run.app |
| `billions-api` | `…/billions/api:latest` | https://billions-api-106830243219.us-central1.run.app |

Legacy hostnames (`https://billions-*-6kgocug7kq-uc.a.run.app`) still resolve; keep both forms in `CORS_ORIGINS`.

Images live in Artifact Registry:

`us-central1-docker.pkg.dev/billions-510913/billions/{api,web}:latest`

Build configs: [`cloudbuild.api.yaml`](cloudbuild.api.yaml) (repo root) and [`web/cloudbuild.yaml`](web/cloudbuild.yaml). Ignore files: [`.gcloudignore`](.gcloudignore) and [`web/.gcloudignore`](web/.gcloudignore).

#### 1. One-time project setup

```bash
gcloud config set project billions-510913
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
gcloud artifacts repositories create billions \
  --repository-format=docker --location=us-central1 \
  --project billions-510913
gcloud auth configure-docker us-central1-docker.pkg.dev
```

You need the Cloud Run Admin, Cloud Build Editor, and Artifact Registry Writer roles (Owner is enough).

#### 2. Build and push images

Prefer Cloud Build (no local Docker daemon). From the repo root:

```bash
# API (~2 minutes)
gcloud builds submit . --config cloudbuild.api.yaml --project billions-510913

# Web (~3 minutes). _API_URL is baked into the Next.js client bundle.
gcloud builds submit web --config web/cloudbuild.yaml --project billions-510913 \
  --substitutions=_API_URL=https://billions-api-106830243219.us-central1.run.app
```

Or build locally and push:

```bash
docker build --platform linux/amd64 -f api/Dockerfile \
  -t us-central1-docker.pkg.dev/billions-510913/billions/api:latest .
docker push us-central1-docker.pkg.dev/billions-510913/billions/api:latest

cd web
docker build --platform linux/amd64 \
  --build-arg NEXT_PUBLIC_API_URL=https://billions-api-106830243219.us-central1.run.app \
  -t us-central1-docker.pkg.dev/billions-510913/billions/web:latest .
docker push us-central1-docker.pkg.dev/billions-510913/billions/web:latest
cd ..
```

#### 3. First deploy

Deploy the API first, then the web app with that URL.

```bash
gcloud run deploy billions-api \
  --image us-central1-docker.pkg.dev/billions-510913/billions/api:latest \
  --project billions-510913 --region us-central1 \
  --allow-unauthenticated \
  --cpu 1 --memory 2Gi --timeout 300 \
  --min-instances 0 --max-instances 20 \
  --cpu-throttling \
  --set-env-vars "^@^CORS_ORIGINS=https://billions-web-106830243219.us-central1.run.app,https://billions-web-6kgocug7kq-uc.a.run.app,http://localhost:3000"

gcloud run deploy billions-web \
  --image us-central1-docker.pkg.dev/billions-510913/billions/web:latest \
  --project billions-510913 --region us-central1 \
  --allow-unauthenticated \
  --port 8080 --cpu 1 --memory 512Mi \
  --set-env-vars "API_URL=https://billions-api-106830243219.us-central1.run.app"
```

`^@^` lets `CORS_ORIGINS` use commas. After the web service exists, update CORS if the console also shows the `*-uc.a.run.app` URL.

#### 4. Production environment variables

Match your local `.env`, but set them on the Cloud Run service. Never put real keys in the README or in git.

**API (`billions-api`)** — current production shape:

| Name | Example / notes |
|---|---|
| `CORS_ORIGINS` | Both web URLs + `http://localhost:3000`, comma-separated |
| `DEBUG` | `false` |
| `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` | Paper keys from [alpaca.markets](https://alpaca.markets) (IEX quotes, paper orders, Alpaca minute bars) |
| `POLYGON_API_KEY` | Optional. Moving-average strategy and Polygon research |
| `BOT_ENABLED` | `true` to turn on `/bot` and the bot API |
| `BOT_ADMIN_TOKEN` | Set a long random string whenever the API is public; the Bot page asks for it |
| `LEAD_LAG_SOURCE` | `alpaca` (no Polygon rate limit) or `polygon` (default) |
| `LEAD_LAG_SCHEDULER` | `true` (default) for the post-close research rescan |
| `OUTLIER_SCHEDULER` / `QUANT_SCHEDULER` | `true` (defaults) |

Update without rebuilding:

```bash
gcloud run services update billions-api \
  --project billions-510913 --region us-central1 \
  --update-env-vars "^@^CORS_ORIGINS=https://billions-web-106830243219.us-central1.run.app,https://billions-web-6kgocug7kq-uc.a.run.app,http://localhost:3000@BOT_ENABLED=true@LEAD_LAG_SOURCE=alpaca@DEBUG=false@ALPACA_API_KEY=YOUR_KEY@ALPACA_SECRET_KEY=YOUR_SECRET@POLYGON_API_KEY=YOUR_POLYGON_KEY@BOT_ADMIN_TOKEN=YOUR_LONG_TOKEN"
```

**Web (`billions-web`)**

| Name | Purpose |
|---|---|
| `API_URL` | Backend URL for Next.js server-side fetches |
| `NEXT_PUBLIC_API_URL` | Set at **image build** time via `_API_URL` / `--build-arg` (not a Cloud Run env var) |
| `NEXT_PUBLIC_SITE_URL` | Optional. Public site URL for canonical links and Open Graph |

#### 5. Redeploy (code only)

Keeps the service's existing scaling and env vars:

```bash
gcloud builds submit . --config cloudbuild.api.yaml --project billions-510913
gcloud builds submit web --config web/cloudbuild.yaml --project billions-510913 \
  --substitutions=_API_URL=https://billions-api-106830243219.us-central1.run.app

gcloud run deploy billions-api \
  --image us-central1-docker.pkg.dev/billions-510913/billions/api:latest \
  --project billions-510913 --region us-central1

gcloud run deploy billions-web \
  --image us-central1-docker.pkg.dev/billions-510913/billions/web:latest \
  --project billions-510913 --region us-central1
```

#### 6. Scaling and cost

| Mode | API flags | When to use |
|---|---|---|
| **Request-based (current)** | `--min-instances 0 --max-instances 20 --cpu-throttling` | Cheapest for browsing. CPU stops between requests. Background outlier / Quant / lead-lag jobs and the paper bot pause while scaled to zero. |
| **Always-on bot** | `--min-instances 1 --max-instances 1 --no-cpu-throttling` | One instance stays up with full CPU. Needed for a live paper bot and reliable background scans. Roughly $50–70 / month for 1 CPU + 2 GiB. |

Switch in the console (Cloud Run → service → Edit & deploy new revision → Scaling / Billing) or with `gcloud run services update`. Two API instances would each run their own bot and their own SQLite cache.

The SQLite cache lives on the container disk. It is rebuilt by the first scan after each deploy or cold start.

#### 7. Smoke test

```bash
curl -s https://billions-api-106830243219.us-central1.run.app/health
# {"status":"healthy","service":"BILLIONS API","version":"2.0.0"}

curl -s -o /dev/null -w "%{http_code}\n" https://billions-web-106830243219.us-central1.run.app/
curl -s -o /dev/null -w "%{http_code}\n" https://billions-web-106830243219.us-central1.run.app/methodology
curl -s -D - -o /dev/null \
  -H "Origin: https://billions-web-106830243219.us-central1.run.app" \
  https://billions-api-106830243219.us-central1.run.app/health
# expect access-control-allow-origin matching the site
```

### Other hosts

**Backend: Railway or Render.** Both build [`api/Dockerfile`](api/Dockerfile) (about 860 MB image, about 170 MB of memory at rest).

- *Railway:* create a project from this repo. `railway.json` selects the Dockerfile and the `/health` check. Set `CORS_ORIGINS` to your frontend URL.
- *Render:* create a Blueprint from [`render.yaml`](render.yaml), then set `CORS_ORIGINS`.
- Copy the same API env vars from the Cloud Run table above (Alpaca, Polygon, bot, lead-lag).

**Frontend: Vercel.** Import the repo, set **Root Directory** to `web`, and set `NEXT_PUBLIC_API_URL` (the backend URL) and `NEXT_PUBLIC_SITE_URL` (the Vercel URL). [`web/vercel.json`](web/vercel.json) pins the install and build commands.

## Project structure

```
api/                 FastAPI backend
  routers/           outliers.py, analysis.py, quant.py, quotes.py, bot.py, lead_lag.py
  services/          outlier_engine.py, outliers.py (scheduler + read model), quant_analysis.py,
                     analysis.py (cache), prices.py, universe.py, market_calendar.py,
                     polygon.py, alpaca_data.py, throttle.py, paper_broker.py, lead_lag.py,
                     lead_lag_research.py, bot/ (strategies, risk, streams, runner)
  tests/             pytest suite
web/                 Next.js app (App Router, TypeScript, Tailwind v4)
  DESIGN.md          design system (Ferrari-based), showroom layer and copy rules
  app/               home, outliers/[strategy], analysis/[ticker], quant, lead-lag, bot,
                     methodology, offline
  components/        charts (hand-written SVG), analysis, home (hero art), ui, logo
  e2e/               Playwright tests, mock API and recorded fixtures
cloudbuild.api.yaml  Cloud Build: API image to Artifact Registry
web/cloudbuild.yaml  Cloud Build: web image (NEXT_PUBLIC_API_URL substitution)
docs/                refactor plan, reference PDFs, screenshots, Lighthouse results
scripts/             one-command dev helpers
```

## Licence

MIT. See [`LICENSE`](LICENSE). General Sans is used under the ITF Free Font License. It is downloaded from Fontshare at build time and not stored in this repository.
