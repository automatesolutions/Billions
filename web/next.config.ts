import path from "node:path";
import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  poweredByHeader: false,
  // The repo root has its own pnpm lockfile (dev tooling); the app lives here.
  outputFileTracingRoot: path.join(__dirname),
};

export default nextConfig;
