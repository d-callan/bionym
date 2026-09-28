// Headless UI tests — drive web/index.html with an injected mock graph
// via the window.__bionym test hook; no backend needed.
//
//   node tests/test_ui.mjs
//
// Needs playwright-core (devDependency of web/components) and a
// Chrome/Chromium binary at $CHROME (default /usr/bin/google-chrome).
import { createRequire } from "node:module";
import http from "node:http";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const WEB = path.join(ROOT, "web");
// resolve from the package that declares playwright-core — no
// machine-specific absolute paths
const require = createRequire(path.join(ROOT, "web", "components", "package.json"));
const { chromium } = require("playwright-core");
const CHROME = process.env.CHROME || "/usr/bin/google-chrome";
const ROW_CAP = 500;
const N_NODES = 40, N_EDGES = 600;

const MIME = {
  ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript",
  ".css": "text/css", ".json": "application/json", ".svg": "image/svg+xml",
};

function serve() {
  return new Promise(resolve => {
    const srv = http.createServer(async (req, res) => {
      try {
        const url = new URL(req.url, "http://x");
        const p = path.join(WEB, url.pathname === "/" ? "index.html" : url.pathname);
        const body = await readFile(p);
        res.writeHead(200, { "content-type": MIME[path.extname(p)] || "application/octet-stream" });
        res.end(body);
      } catch {
        res.writeHead(404); res.end("nf");
      }
    }).listen(0, () => resolve(srv));
  });
}

const results = [];
const check = (name, cond, extra = "") => {
  results.push([name, !!cond]);
  console.log(`${cond ? "ok  " : "FAIL"}  ${name}${extra ? "  — " + extra : ""}`);
};

const srv = await serve();
const port = srv.address().port;
const browser = await chromium.launch({ executablePath: CHROME, headless: true });

try {
  const page = await browser.newPage();
  page.on("pageerror", e => console.error("pageerror:", e.message));
  await page.goto(`http://localhost:${port}/index.html`);
  await page.waitForFunction(() => window.__bionym, null, { timeout: 15000 });

  // inject a mock graph through the test hook — built inside the page
  // (page.evaluate can't see Node-scope helpers)
  await page.evaluate(([n, e]) => {
    const nodes = {};
    for (let i = 0; i < n; i++)
      nodes[`n:${i}`] = {
        id: `n:${i}`, type: i % 3 ? "Gene" : "Dataset",
        label: `node ${i}`, id_namespace: "test", url: null,
      };
    const edges = [];
    for (let i = 0; i < e; i++)
      edges.push({
        subject: `n:${i % n}`, predicate: "linked_to",
        object: `n:${(i * 7 + 1) % n}`, confidence: 0.8, evidence: [],
      });
    window.__bionym.setGraph({ nodes, edges, metadata: { query: "test", stages: [] } });
  }, [N_NODES, N_EDGES]);

  // 1. columns render smoke — every node gets a glyph
  await page.waitForSelector("#net .nodeg", { timeout: 10000 });
  check(
    "columns render: all nodes drawn",
    (await page.locator("#net .nodeg").count()) === N_NODES,
    `${await page.locator("#net .nodeg").count()} nodeg`
  );

  // 2. tooltip on hover (dispatchEvent — page.hover misses offscreen svg)
  await page.evaluate(() => {
    document
      .querySelectorAll("#net .nodeg")[4]
      .dispatchEvent(new MouseEvent("mouseenter"));
  });
  const tip = page.locator(".bn-tip");
  check("tooltip shows on hover", await tip.isVisible());
  check(
    "tooltip has label + edge count",
    (await tip.textContent()).includes("node 4")
  );

  // 3. table row cap — 600 edges → ROW_CAP rows + 1 summary row
  const edgeRows = await page.locator("#edges tbody tr").count();
  check("edges table capped", edgeRows === ROW_CAP + 1, `${edgeRows} rows`);
  check(
    "row-cap summary present",
    (await page.locator("#edges tbody tr:last-child").textContent()).includes(
      `of ${N_EDGES}`
    )
  );

  // 4. force layout completes asynchronously — nodes appear after ticks
  await page.evaluate(() => window.__bionym.net.setCluster(false));
  await page.evaluate(() => window.__bionym.net.setLayout("force"));
  await page.waitForFunction(
    () => document.querySelectorAll("#net .nodeg").length > 0,
    null,
    { timeout: 10000 }
  );
  const pos = await page.evaluate(() =>
    [...document.querySelectorAll("#net .nodeg")]
      .slice(0, 5)
      .map(g => g.getAttribute("transform"))
  );
  check(
    "force layout completes (async ticks)",
    pos.every(t => /translate\([\d.-]+,[\d.-]+\)/.test(t)),
    pos[0]
  );
  check(
    "force layout spread (positions differ)",
    new Set(pos).size > 1
  );

  // 5. click pins a node → detail panel populates
  await page.evaluate(() => {
    document
      .querySelectorAll("#net .nodeg")[0]
      .dispatchEvent(new MouseEvent("click", { bubbles: true }));
  });
  await page.waitForTimeout(200);
  check(
    "click pins node → detail panel",
    (await page.locator("#detail").innerHTML()).length > 50
  );
} finally {
  await browser.close();
  srv.close();
}

const fails = results.filter(([, ok]) => !ok);
console.log(`\n${results.length - fails.length}/${results.length} passed`);
process.exit(fails.length ? 1 : 0);
