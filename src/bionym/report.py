"""Self-contained HTML report: summary tables + D3 multi-partite network.

Nodes are laid out in one column per node type (deterministic order),
edges drawn as curved links with confidence mapped to opacity/width.
D3 is loaded from CDN; the graph JSON is embedded inline.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

from .graph import KnowledgeGraph

# Column order for the multi-partite layout.
COLUMN_ORDER = [
    "IdType", "Gene", "Organism", "Assembly",
    "OrthologGroup", "Domain", "GOTerm", "Pathway",
    "Dataset", "Condition",
]

_COLORS = {
    "IdType": "#8d99ae", "Gene": "#2b6cb0", "Organism": "#2f855a",
    "Assembly": "#b7791f", "OrthologGroup": "#6b46c1", "Domain": "#c05621",
    "GOTerm": "#2c7a7b", "Pathway": "#97266d", "Dataset": "#4a5568",
    "Condition": "#9b2c2c",
}


def render_html(graph: KnowledgeGraph) -> str:
    nodes = [n.model_dump(mode="json") for n in graph.nodes.values()]
    edges = [e.model_dump(mode="json") for e in graph.edges]
    meta = graph.metadata
    usage = (meta.get("jev_usage") or {}).get("total") or {}

    graph_json = json.dumps({"nodes": nodes, "edges": edges})
    columns = json.dumps(COLUMN_ORDER)
    colors = json.dumps(_COLORS)

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>BioNym: {html.escape(str(meta.get('input','')))}</title>
<script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
<style>
 body {{ font-family: system-ui, sans-serif; margin: 2rem; color: #1a202c; }}
 h1 {{ font-size: 1.4rem; }} h2 {{ font-size: 1.1rem; margin-top: 2rem; }}
 table {{ border-collapse: collapse; font-size: 0.85rem; }}
 th, td {{ border: 1px solid #cbd5e0; padding: 3px 8px; text-align: left; }}
 th {{ background: #edf2f7; }}
 .meta {{ color: #4a5568; font-size: 0.85rem; }}
 svg {{ border: 1px solid #e2e8f0; }}
 .node {{ stroke: #fff; stroke-width: 1px; }}
 .nodelabel {{ font-size: 9px; }}
 .link {{ fill: none; }}
 .colhead {{ font-size: 11px; font-weight: 600; fill: #4a5568; }}
 .tfilter {{ font-size: .8rem; padding: .2rem .5rem; border: 1px solid #cbd5e0; border-radius: 5px; font-weight: normal; }}
 th {{ cursor: pointer; user-select: none; }}
 th:hover {{ background: #e2e8f0; }}
</style></head><body>
<h1>BioNym report: <code>{html.escape(str(meta.get('input','')))}</code></h1>
<p class="meta">stages: {html.escape(', '.join(meta.get('stages', [])))} &middot;
 {len(nodes)} nodes, {len(edges)} edges &middot;
 JEV tokens: {usage.get('input_tokens', 0)} in / {usage.get('output_tokens', 0)} out
 ({usage.get('calls', 0)} calls)</p>
<p class="meta">Confidence scores are judged by
 <a href="https://typesafe.ai" target="_blank" rel="noopener">JEV (TypeSafe.ai)</a>
 — every edge carries a model-judged confidence and its evidence.</p>

<h2>Network</h2>
<svg id="net" width="1100" height="600"></svg>

<h2>Edges <input class="tfilter" data-for="edges" placeholder="filter…"></h2>
<table id="edges"><thead><tr><th>subject</th><th>predicate</th><th>object</th><th>conf</th><th>ev</th></tr></thead><tbody></tbody></table>

<h2>Nodes <input class="tfilter" data-for="nodes" placeholder="filter…"></h2>
<table id="nodes"><thead><tr><th>id</th><th>type</th><th>label</th><th>namespace</th></tr></thead><tbody></tbody></table>

<script>
const graph = {graph_json};
const COLS = {columns};
const COLORS = {colors};
const svg = d3.select("#net"), W = +svg.attr("width"), H = +svg.attr("height");

const byType = d3.group(graph.nodes, d => d.type);
const cols = COLS.filter(t => byType.has(t));
const x = d3.scalePoint(cols, [60, W - 60]);
const pos = {{}};
cols.forEach(t => {{
  const ns = byType.get(t);
  const y = d3.scalePoint(d3.range(ns.length), [40, H - 30]);
  ns.forEach((n, i) => pos[n.id] = {{x: x(t), y: y(i)}});
}});

svg.selectAll(".colhead").data(cols).join("text")
  .attr("class", "colhead").attr("x", d => x(d)).attr("y", 18)
  .attr("text-anchor", "middle").text(d => d);

svg.selectAll(".link").data(graph.edges).join("path")
  .attr("class", "link")
  .attr("d", e => {{
    const s = pos[e.subject], t = pos[e.object];
    if (!s || !t) return "";
    const mx = (s.x + t.x) / 2;
    return `M${{s.x}},${{s.y}} C${{mx}},${{s.y}} ${{mx}},${{t.y}} ${{t.x}},${{t.y}}`;
  }})
  .attr("stroke", "#718096")
  .attr("stroke-opacity", e => 0.15 + 0.85 * e.confidence)
  .attr("stroke-width", e => 0.5 + 2.5 * e.confidence);

const nodeG = svg.selectAll(".nodeg").data(graph.nodes).join("g")
  .attr("transform", d => `translate(${{pos[d.id].x}},${{pos[d.id].y}})`)
  .style("cursor", d => d.url ? "pointer" : "default")
  .on("click", (e, d) => {{ if (d.url) window.open(d.url, "_blank"); }});
nodeG.append("circle").attr("class", "node")
  .attr("r", 5).attr("fill", d => COLORS[d.type] || "#999");
nodeG.append("text").attr("class", "nodelabel").attr("x", 7).attr("y", 3)
  .text(d => (d.label || d.id).slice(0, 24));
nodeG.append("title")
  .text(d => `${{d.id}} (${{d.type}})${{d.url ? " — click to open" : ""}}`);

// -- sortable + filterable tables --------------------------------------
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({{"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}}[c]));
const link = n => n.url ? `<a href="${{esc(n.url)}}" target="_blank" rel="noopener">${{esc(n.label||n.id)}}</a>` : esc(n.label||n.id);
const byId = Object.fromEntries(graph.nodes.map(n => [n.id, n]));
const cell = id => byId[id] ? link(byId[id]) : esc(id);

function sortable(tableId, rows, cols, initKey = null, initDir = 1) {{
  const tbl = document.getElementById(tableId);
  const tbody = tbl.querySelector("tbody");
  const ths = [...tbl.querySelectorAll("th")];
  const state = {{key: initKey, dir: initDir, filter: ""}};
  const fi = document.querySelector(`.tfilter[data-for="${{tableId}}"]`);
  if (fi) fi.addEventListener("input", e => {{ state.filter = e.target.value.toLowerCase(); draw(); }});
  ths.forEach((th, i) => th.addEventListener("click", () => {{
    if (state.key === i) state.dir *= -1; else {{ state.key = i; state.dir = 1; }}
    draw();
  }}));
  function draw() {{
    let rs = rows.filter(r => !state.filter || cols.some(c => String(c.val(r) ?? "").toLowerCase().includes(state.filter)));
    if (state.key !== null) {{
      const c = cols[state.key];
      rs = rs.slice().sort((a, b) => {{
        const x = c.val(a), y = c.val(b);
        return (typeof x === "number" && typeof y === "number" ? x - y : String(x ?? "").localeCompare(String(y ?? ""))) * state.dir;
      }});
    }}
    tbody.innerHTML = rs.map(r => "<tr>" + cols.map(c => `<td class="${{c.cls || ""}}">${{c.render(r)}}</td>`).join("") + "</tr>").join("");
    ths.forEach((th, i) => th.textContent = cols[i].label + (state.key === i ? (state.dir > 0 ? " ▲" : " ▼") : ""));
  }}
  draw();
}}

sortable("edges", graph.edges, [
  {{label:"subject", val:e=>e.subject, render:e=>cell(e.subject)}},
  {{label:"predicate", val:e=>e.predicate, render:e=>esc(e.predicate)}},
  {{label:"object", val:e=>e.object, render:e=>cell(e.object)}},
  {{label:"conf", val:e=>e.confidence, render:e=>e.confidence.toFixed(2), cls:"conf"}},
  {{label:"ev", val:e=>(e.evidence||[]).length, render:e=>(e.evidence||[]).length}},
], 3, -1);
sortable("nodes", graph.nodes, [
  {{label:"id", val:n=>n.id, render:n=>esc(n.id)}},
  {{label:"type", val:n=>n.type, render:n=>esc(n.type)}},
  {{label:"label", val:n=>n.label||n.id, render:n=>link(n)}},
  {{label:"namespace", val:n=>n.id_namespace, render:n=>esc(n.id_namespace)}},
], 1, 1);
</script>
</body></html>"""


def write_report(graph: KnowledgeGraph, path: str | Path) -> Path:
    p = Path(path)
    p.write_text(render_html(graph))
    return p
