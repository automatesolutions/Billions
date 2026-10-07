// Minimal stand-in for the BILLIONS API, serving recorded responses for E2E tests.
import { readFileSync } from 'node:fs';
import { createServer } from 'node:http';

const fixture = (name) => JSON.parse(readFileSync(new URL(`./fixtures/${name}`, import.meta.url), 'utf8'));
const outliers = fixture('outliers-swing.json');
const analysis = fixture('analysis-AAPL.json');
const port = Number(process.env.MOCK_API_PORT || 8010);

createServer((req, res) => {
  const url = new URL(req.url, 'http://localhost');
  const send = (status, body) => {
    res.writeHead(status, { 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*' });
    res.end(JSON.stringify(body));
  };
  let m;
  if (url.pathname === '/health') return send(200, { status: 'healthy' });
  if ((m = url.pathname.match(/^\/api\/v1\/outliers\/(scalp|swing|longterm)$/))) return send(200, { ...outliers, strategy: m[1] });
  if ((m = url.pathname.match(/^\/api\/v1\/analysis\/([A-Z0-9.-]+)$/i))) {
    const ticker = m[1].toUpperCase();
    if (ticker === 'NOPE') return send(404, { detail: `No price history found for ${ticker}.` });
    return send(200, { ...analysis, ticker });
  }
  send(404, { detail: 'Not found' });
}).listen(port, () => console.log(`mock API on :${port}`));
