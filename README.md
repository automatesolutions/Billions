# BILLIONS

BILLIONS finds stocks that are moving unusually far from the pack, and explains each one with a plain, tested quant analysis.

It is a read-only web app. **Information only. Not financial advice. This tool does not place trades.**

> This README is being rewritten as part of the `refactor/outliers-only` work. See `docs/REFACTOR_PLAN.md`.

## Pages

| URL | What it shows |
|---|---|
| `/outliers/{scalp,swing,longterm}` | Stocks whose returns sit more than 2 standard deviations from the rest |
| `/analysis/{TICKER}` | Signal, edge, risk and model validation for one stock |
| `/methodology` | How the numbers are made, and their limits |

## Run it locally

You need Node 20+, pnpm 9+ and Python 3.12+.

```bash
pnpm install     # root tools
pnpm setup       # frontend deps + Python venv in .venv
pnpm dev         # API on :8000, web on :3000
```

Or with Docker: `docker compose up`.

Open http://localhost:3000.

## Environment variables

Backend (`.env` in the repo root, see `.env.example`):

| Name | Default | Purpose |
|---|---|---|
| `CORS_ORIGINS` | `http://localhost:3000,http://127.0.0.1:3000` | Allowed frontend origins, comma-separated |
| `DATABASE_URL` | `sqlite:///./data/billions.db` | Outlier cache |
| `ALPHA_VANTAGE_API_KEY` | empty | Optional |

Frontend (`web/.env.local`, see `web/.env.example`):

| Name | Purpose |
|---|---|
| `NEXT_PUBLIC_API_URL` | Backend base URL |
| `NEXT_PUBLIC_SITE_URL` | Public site URL for link previews |

## Tests

```bash
pnpm test        # pytest + vitest
pnpm lint        # flake8 + eslint
```

## Deploy

- **Frontend:** Vercel. Set the project's Root Directory to `web` and set `NEXT_PUBLIC_API_URL`.
- **Backend:** Railway (`railway.json`) or Render (`render.yaml`). Both build `api/Dockerfile`. Set `CORS_ORIGINS` to your Vercel URL.

## Licence

MIT. See `LICENSE`.
