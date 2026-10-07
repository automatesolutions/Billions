// Runs the venv Python with the given arguments from the repo root.
import { spawnSync } from "node:child_process";
import { rootDir, venvPython } from "./venv.mjs";

if (!venvPython) {
  console.error("No .venv found. Run `pnpm setup` first.");
  process.exit(1);
}
const result = spawnSync(venvPython, process.argv.slice(2), { cwd: rootDir, stdio: "inherit" });
process.exit(result.status ?? 1);
