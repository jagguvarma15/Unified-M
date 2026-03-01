import { readdirSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";
import { gzipSync } from "node:zlib";

const DIST_DIR = new URL("../dist", import.meta.url).pathname;
const ASSETS_DIR = join(DIST_DIR, "assets");

const BUDGETS = {
  jsTotalGzipKb: 380,
  cssTotalGzipKb: 120,
  maxSingleJsGzipKb: 230,
};

const files = readdirSync(ASSETS_DIR);
const jsFiles = files.filter((f) => f.endsWith(".js"));
const cssFiles = files.filter((f) => f.endsWith(".css"));

function gzipKb(filePath) {
  const content = readFileSync(filePath);
  return gzipSync(content).length / 1024;
}

function sum(values) {
  return values.reduce((acc, n) => acc + n, 0);
}

const jsGzipSizes = jsFiles.map((f) => gzipKb(join(ASSETS_DIR, f)));
const cssGzipSizes = cssFiles.map((f) => gzipKb(join(ASSETS_DIR, f)));

const jsTotal = sum(jsGzipSizes);
const cssTotal = sum(cssGzipSizes);
const maxSingleJs = jsGzipSizes.length ? Math.max(...jsGzipSizes) : 0;

const rows = [
  ["JS total gzip", `${jsTotal.toFixed(1)} KB`, `<= ${BUDGETS.jsTotalGzipKb} KB`],
  ["CSS total gzip", `${cssTotal.toFixed(1)} KB`, `<= ${BUDGETS.cssTotalGzipKb} KB`],
  ["Max JS chunk gzip", `${maxSingleJs.toFixed(1)} KB`, `<= ${BUDGETS.maxSingleJsGzipKb} KB`],
];

console.log("Bundle budget report");
for (const [name, value, budget] of rows) {
  console.log(`- ${name}: ${value} (budget ${budget})`);
}

const failed =
  jsTotal > BUDGETS.jsTotalGzipKb ||
  cssTotal > BUDGETS.cssTotalGzipKb ||
  maxSingleJs > BUDGETS.maxSingleJsGzipKb;

statSync(DIST_DIR);

if (failed) {
  console.error("Bundle budget check failed.");
  process.exit(1);
}

console.log("Bundle budget check passed.");
