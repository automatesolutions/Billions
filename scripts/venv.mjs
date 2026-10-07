// Finds the Python interpreter in ./.venv, on Windows or Unix.
import { existsSync } from "node:fs";
import { join } from "node:path";

const root = new URL("..", import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, "$1");
const candidates = [join(root, ".venv", "Scripts", "python.exe"), join(root, ".venv", "bin", "python")];

export const rootDir = root;
export const venvPython = candidates.find((p) => existsSync(p));
export const systemPython = process.platform === "win32" ? "python" : "python3";
