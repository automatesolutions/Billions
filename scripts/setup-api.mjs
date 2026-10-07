// Creates ./.venv and installs the backend dependencies.
import { spawnSync } from "node:child_process";
import { rootDir, systemPython } from "./venv.mjs";

const run = (cmd, args) => {
  const r = spawnSync(cmd, args, { cwd: rootDir, stdio: "inherit" });
  if (r.status !== 0) process.exit(r.status ?? 1);
};

run(systemPython, ["-m", "venv", ".venv"]);
const { venvPython } = await import("./venv.mjs?after-venv");
run(venvPython, ["-m", "pip", "install", "-r", "api/requirements-dev.txt"]);
