// Starts the FastAPI backend with auto-reload on http://localhost:8000.
import { spawn } from "node:child_process";
import { rootDir, venvPython } from "./venv.mjs";

if (!venvPython) {
  console.error("No .venv found. Run `pnpm setup` first.");
  process.exit(1);
}
const args = ["-m", "uvicorn", "api.main:app", "--reload", "--reload-dir", "api", "--port", "8000"];
spawn(venvPython, args, { cwd: rootDir, stdio: "inherit" }).on("exit", (code) => process.exit(code ?? 0));
