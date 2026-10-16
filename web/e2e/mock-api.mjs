// Minimal stand-in for the BILLIONS API, serving recorded responses for E2E tests.
import { readFileSync } from 'node:fs';
import { createServer } from 'node:http';

const fixture = (name) => JSON.parse(readFileSync(new URL(`./fixtures/${name}`, import.meta.url), 'utf8'));
const outliers = fixture('outliers-swing.json');
const analysis = fixture('analysis-AAPL.json');
const quant = fixture('quant-AAPL.json');
const port = Number(process.env.MOCK_API_PORT || 8010);
const watched = new Set();

createServer((req, res) => {
  const url = new URL(req.url, 'http://localhost');
  const send = (status, body) => {
    res.writeHead(status, {
      'Content-Type': 'application/json',
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, POST, DELETE',
    });
    res.end(JSON.stringify(body));
  };
  let m;
  if (req.method === 'OPTIONS') return send(204, {});
  if (url.pathname === '/health') return send(200, { status: 'healthy' });
  if ((m = url.pathname.match(/^\/api\/v1\/outliers\/(scalp|swing|longterm)$/))) return send(200, { ...outliers, strategy: m[1] });
  if ((m = url.pathname.match(/^\/api\/v1\/analysis\/([A-Z0-9.-]+)$/i))) {
    const ticker = m[1].toUpperCase();
    if (ticker === 'NOPE') return send(404, { detail: `No price history found for ${ticker}.` });
    return send(200, { ...analysis, ticker });
  }
  if (url.pathname === '/api/v1/quant/watching') return send(200, { watching: [{ ticker: 'AAPL', interval: '1h' }], max: 20 });
  if ((m = url.pathname.match(/^\/api\/v1\/quant\/([A-Z0-9.-]+)\/watch$/i))) {
    const key = `${m[1].toUpperCase()}:${url.searchParams.get('interval') || '1h'}`;
    if (req.method === 'DELETE') watched.delete(key);
    else watched.add(key);
    return send(200, { watching: watched.has(key) });
  }
  if ((m = url.pathname.match(/^\/api\/v1\/quant\/([A-Z0-9.-]+)$/i))) {
    const ticker = m[1].toUpperCase();
    const interval = url.searchParams.get('interval') || '1h';
    return send(200, {
      ...quant,
      ticker,
      interval,
      watching: watched.has(`${ticker}:${interval}`),
      last_update_at: new Date(Date.now() - 1800e3).toISOString(),
      next_update_at: new Date(Date.now() + 1800e3).toISOString(),
    });
  }
  if ((m = url.pathname.match(/^\/api\/v1\/quotes\/([A-Z0-9.-]+)$/i))) {
    const ticker = m[1].toUpperCase();
    return send(200, {
      ticker, available: true, source: 'Alpaca', feed: 'iex', market_open: false,
      bid: 99.98, ask: 100.02, bid_size: 300, ask_size: 100, mid: 100, spread: 0.04, spread_bps: 4,
      imbalance: 0.5, microprice: 100.01, quoted_at: '2026-10-15T19:59:58Z', age_seconds: 60,
    });
  }
  send(404, { detail: 'Not found' });
}).listen(port, () => console.log(`mock API on :${port}`));
