/**
 * Decorative hero art: about 260 stocks clustered as "the pack", the 1σ and 2σ rings,
 * and one Rosso Corsa outlier far outside them. Seeded, so server and client render the same.
 */

const W = 1200;
const H = 720;
const CX = 640;
const CY = 400;
const SX = 120;
const SY = 70;

function mulberry32(seed: number) {
  return () => {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function pack(count: number) {
  const rand = mulberry32(20261007);
  const pts: { x: number; y: number; r: number; o: number }[] = [];
  while (pts.length < count) {
    // Box–Muller: a normal cloud, like returns.
    const u = Math.max(rand(), 1e-9);
    const v = rand();
    const mag = Math.sqrt(-2 * Math.log(u));
    const zx = mag * Math.cos(2 * Math.PI * v);
    const zy = mag * Math.sin(2 * Math.PI * v);
    if (Math.hypot(zx, zy) > 2.2) continue;
    pts.push({
      x: CX + zx * SX,
      y: CY + zy * SY,
      r: 1.6 + rand() * 2.2,
      o: 0.16 + (1 - Math.min(Math.hypot(zx, zy) / 2.2, 1)) * 0.34,
    });
  }
  return pts;
}

const POINTS = pack(260);
const OUT = { x: 1020, y: 150 };

export function PackField({ className }: { className?: string }) {
  return (
    <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="xMidYMid slice" aria-hidden className={className}>
      <defs>
        <radialGradient id="pf-glow">
          <stop offset="0%" className="[stop-color:var(--color-primary)]" stopOpacity="0.85" />
          <stop offset="100%" className="[stop-color:var(--color-primary)]" stopOpacity="0" />
        </radialGradient>
        <radialGradient id="pf-pack">
          <stop offset="0%" className="[stop-color:var(--color-ink)]" stopOpacity="0.06" />
          <stop offset="100%" className="[stop-color:var(--color-ink)]" stopOpacity="0" />
        </radialGradient>
      </defs>

      <ellipse cx={CX} cy={CY} rx={SX * 3} ry={SY * 3} fill="url(#pf-pack)" />

      {/* σ rings */}
      <ellipse cx={CX} cy={CY} rx={SX} ry={SY} fill="none" className="stroke-muted" strokeWidth="1" opacity="0.5" />
      <ellipse cx={CX} cy={CY} rx={SX * 2} ry={SY * 2} fill="none" className="stroke-muted" strokeWidth="1" strokeDasharray="4 8" />
      <text x={CX + SX * 2 + 12} y={CY + 4} className="fill-muted text-caption">
        2σ
      </text>

      {POINTS.map((p, i) => (
        <circle key={i} cx={p.x.toFixed(1)} cy={p.y.toFixed(1)} r={p.r.toFixed(1)} className="fill-ink" opacity={p.o.toFixed(2)} />
      ))}

      {/* The outlier */}
      <line x1={CX} y1={CY} x2={OUT.x} y2={OUT.y} className="stroke-primary" strokeWidth="1" strokeDasharray="2 6" opacity="0.7" />
      <circle cx={OUT.x} cy={OUT.y} r="90" fill="url(#pf-glow)" opacity="0.55" />
      <circle cx={OUT.x} cy={OUT.y} r="22" fill="none" className="stroke-primary" strokeWidth="1" opacity="0.6" />
      <circle cx={OUT.x} cy={OUT.y} r="8" className="fill-primary" />
    </svg>
  );
}
