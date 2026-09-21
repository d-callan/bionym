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

    node_rows = "".join(
        f"<tr><td>{html.escape(n['id'])}</td><td>{n['type']}</td>"
        f"<td>{html.escape(n.get('label') or '')}</td>"
        f"<td>{html.escape(n.get('id_namespace') or '')}</td></tr>"
        for n in sorted(nodes, key=lambda x: (x["type"], x["id"]))
    )
    edge_rows = "".join(
        f"<tr><td>{html.escape(e['subject'])}</td>"
        f"<td>{html.escape(e['predicate'])}</td>"
        f"<td>{html.escape(e['object'])}</td>"
        f"<td>{e['confidence']:.2f}</td>"
        f"<td>{len(e.get('evidence') or [])}</td></tr>"
        for e in sorted(edges, key=lambda x: -x["confidence"])
    )

    graph_json = json.dumps({"nodes": nodes, "edges": edges})
    columns = json.dumps(COLUMN_ORDER)
    colors = json.dumps(_COLORS)

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>idresolver: {html.escape(str(meta.get('input','')))}</title>
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
</style></head><body>
<h1>idresolver report: <code>{html.escape(str(meta.get('input','')))}</code></h1>
<p class="meta">stages: {html.escape(', '.join(meta.get('stages', [])))} &middot;
 {len(nodes)} nodes, {len(edges)} edges &middot;
 JEV tokens: {usage.get('input_tokens', 0)} in / {usage.get('output_tokens', 0)} out
 ({usage.get('calls', 0)} calls)</p>

<h2>Network</h2>
<svg id="net" width="1100" height="600"></svg>

<h2>Edges (by confidence)</h2>
<table><tr><th>subject</th><th>predicate</th><th>object</th><th>conf</th><th>ev</th></tr>
{edge_rows}</table>

<h2>Nodes</h2>
<table><tr><th>id</th><th>type</th><th>label</th><th>namespace</th></tr>
{node_rows}</table>

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

const node = svg.selectAll(".node").data(graph.nodes).join("circle")
  .attr("class", "node")
  .attr("cx", d => pos[d.id].x).attr("cy", d => pos[d.id].y)
  .attr("r", 5).attr("fill", d => COLORS[d.type] || "#999");
node.append("title").text(d => `${{d.id}} (${{d.type}})`);

svg.selectAll(".nodelabel").data(graph.nodes).join("text")
  .attr("class", "nodelabel")
  .attr("x", d => pos[d.id].x + 7).attr("y", d => pos[d.id].y + 3)
  .text(d => (d.label || d.id).slice(0, 24));
</script>
</body></html>"""


def write_report(graph: KnowledgeGraph, path: str | Path) -> Path:
    p = Path(path)
    p.write_text(render_html(graph))
    return p
