// Downloads General Sans (variable, woff2) from Fontshare into app/fonts/.
//
// The ITF Free Font License allows self-hosting on our own site but not
// redistributing the font files through a public repository, so the file is
// fetched at install/build time and git-ignored instead of committed.
import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const target = join(dirname(fileURLToPath(import.meta.url)), "..", "app", "fonts", "GeneralSans-Variable.woff2");
const CSS_URL = "https://api.fontshare.com/v2/css?f[]=general-sans@1&display=swap";

if (existsSync(target)) process.exit(0);

const css = await (await fetch(CSS_URL)).text();
const match = css.match(/url\('([^']+\.woff2)'\)/);
if (!match) throw new Error("General Sans woff2 URL not found in Fontshare CSS");

const url = match[1].startsWith("//") ? `https:${match[1]}` : match[1];
const response = await fetch(url);
if (!response.ok) throw new Error(`Font download failed: HTTP ${response.status}`);

mkdirSync(dirname(target), { recursive: true });
writeFileSync(target, Buffer.from(await response.arrayBuffer()));
console.log("General Sans downloaded from Fontshare");
