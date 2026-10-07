import path from "node:path";
import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  poweredByHeader: false,
  // The repo root has its own pnpm lockfile (dev tooling); the app lives here.
  outputFileTracingRoot: path.join(__dirname),
  // Put <title>, description and Open Graph tags in <head> for every visitor, not only known bots.
  // Our generateMetadata only reads URL params, so this never waits on data.
  htmlLimitedBots: /.*/,
};

export default nextConfig;
