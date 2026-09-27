"""Self-contained HTML report — renders a graph JSON with the same
@bionym/components modules the web UI uses, fully inlined so the output
file has zero network dependencies (safe for Galaxy datasets, offline
viewing, email attachments).

The JS bundle + CSS + driver live in ``bionym/_static/`` and are shipped
as package data. Regenerate them from ``web/components`` with:

    cd web/components && npm run build:report
"""

from __future__ import annotations

import html
import json
from functools import lru_cache
from pathlib import Path

from .graph import KnowledgeGraph


@lru_cache
def _assets() -> dict[str, str]:
    """Load the inlined web assets (bundle JS, component CSS, driver JS)."""
    d = Path(__file__).parent / "_static"
    return {
        "bundle": (d / "components.iife.js").read_text(),
        "css": (d / "components.css").read_text(),
        "driver": (d / "report.js").read_text(),
    }


_TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8"><title>BioNym: __TITLE__</title>
<style>
body { font-family: system-ui, sans-serif; margin: 0; color: #1a202c; background: #f7fafc; }
main { padding: 0 2rem 3rem; }
h1 { font-size: 1.3rem; margin: 0 0 .4rem; }
h2 { font-size: 1.05rem; margin: 1.6rem 0 .5rem; }
header { padding: 1.2rem 2rem; border-bottom: 1px solid #cbd5e0; background: #fff; }
.meta { color: #4a5568; font-size: .85rem; margin: .3rem 0; }
.meta a { color: #2b6cb0; }
svg.bn-net, svg.bn-ov { background: #fff; border: 1px solid #cbd5e0; border-radius: 8px; width: 100%; height: auto; }
table { border-collapse: collapse; font-size: .82rem; background: #fff; }
th, td { border: 1px solid #cbd5e0; padding: 3px 8px; text-align: left; }
th { background: #edf2f7; position: sticky; top: 0; cursor: pointer; user-select: none; }
th:hover { background: #e2e8f0; }
.tablewrap { max-height: 22rem; overflow: auto; border-radius: 8px; }
.conf { font-variant-numeric: tabular-nums; }
.tfilter { font-size: .8rem; padding: .2rem .5rem; border: 1px solid #cbd5e0; border-radius: 5px; font-weight: normal; }
h2 label, h2 button, h2 select { font-weight: normal; margin-left: .8rem; }
#layout, #stacksel { font-size: .8rem; padding: .15rem .3rem; }
#resetzoom, #zin, #zout { font-size: .75rem; padding: .2rem .6rem; margin-left: .5rem;
  background: #edf2f7; color: #1a202c; border: 1px solid #cbd5e0; border-radius: 6px; cursor: pointer; }
#zin, #zout { width: 1.8rem; }
.hint { font-size: .72rem; font-weight: normal; color: #4a5568; margin-left: .6rem; }
details#summary { margin-top: 1.4rem; background: #fff; border: 1px solid #cbd5e0; border-radius: 8px; padding: .6rem 1rem; }
details#summary > summary { cursor: pointer; font-size: 1.05rem; font-weight: 600; }
details#summary ul { margin: .6rem 0 .2rem; padding-left: 1.2rem; font-size: .88rem; }
details#summary li { margin-bottom: .35rem; }
.sumconf { color: #4a5568; font-size: .78rem; }
#detail { position: fixed; right: 0; top: 0; bottom: 0; width: 24rem; overflow-y: auto;
  background: #fff; border-left: 1px solid #cbd5e0; box-shadow: -4px 0 12px rgba(0,0,0,.08);
  padding: 1rem 1.2rem; font-size: .85rem; z-index: 10; }
#detail h3 { margin: 0 0 .3rem; font-size: 1rem; }
__COMPONENT_CSS__
</style></head><body>
<header><h1>BioNym report: <code>__INPUT__</code></h1></header>
<main>
<p class="meta">stages: __STAGES__ &middot; __NODES__ nodes, __NEDGES__ edges &middot;
 JEV tokens: __JEV__</p>
<p class="meta">Confidence scores are judged by
 <a href="https://typesafe.ai" target="_blank" rel="noopener">JEV (TypeSafe.ai)</a>
 — every edge carries a model-judged confidence and its evidence.</p>

<details id="summary" hidden><summary>Gene summary</summary><ul id="summarylist"></ul></details>

<h2>Network
 <select id="layout"><option value="columns">columns</option><option value="force">force</option></select>
 <label id="clustwrap" hidden><input type="checkbox" id="clustergroup" checked> cluster</label>
 <button id="resetzoom" title="zoom to fit">fit</button><button id="zin">+</button><button id="zout">−</button>
 <span class="hint">click a node for details · scroll/drag to zoom/pan</span></h2>
<div id="net"></div>

<h2>Overview <select id="stacksel"><option value="off">flat</option><option value="subject">stack: subjects</option><option value="object">stack: objects</option></select></h2>
<div id="ov"></div>

<h2>Edges <input class="tfilter" data-for="edges" placeholder="filter…"></h2>
<div class="tablewrap"><table id="edges"><thead><tr><th>subject</th><th>predicate</th><th>object</th><th>conf</th><th>ev</th></tr></thead><tbody></tbody></table></div>

<h2>Nodes <input class="tfilter" data-for="nodes" placeholder="filter…"></h2>
<div class="tablewrap"><table id="nodes"><thead><tr><th>id</th><th>type</th><th>label</th><th>namespace</th></tr></thead><tbody></tbody></table></div>
</main>

<aside id="detail" hidden></aside>

<script>__BUNDLE__</script>
<script>window.__GRAPH__ = __GRAPH_JSON__;</script>
<script>__DRIVER__</script>
</body></html>"""


def render_html(graph: KnowledgeGraph) -> str:
    graph_json = graph.model_dump_json()
    # keep literal "<" out of the inline JSON so embedded strings can't
    # terminate the <script> element early or open a comment ("<!--")
    graph_json = graph_json.replace("<", "\\u003c")

    meta = graph.metadata
    usage = (meta.get("jev_usage") or {}).get("total") or {}
    jev = (
        f"{usage.get('input_tokens', 0)} in / {usage.get('output_tokens', 0)} out"
        f" ({usage.get('calls', 0)} calls)"
    )
    assets = _assets()
    return (
        _TEMPLATE
        .replace("__TITLE__", html.escape(str(meta.get("input", ""))))
        .replace("__INPUT__", html.escape(str(meta.get("input", ""))))
        .replace("__STAGES__", html.escape(", ".join(meta.get("stages", []))))
        .replace("__NODES__", str(len(graph.nodes)))
        .replace("__NEDGES__", str(len(graph.edges)))
        .replace("__JEV__", jev)
        .replace("__COMPONENT_CSS__", assets["css"])
        .replace("__BUNDLE__", assets["bundle"])
        .replace("__GRAPH_JSON__", graph_json)
        .replace("__DRIVER__", assets["driver"])
    )


def write_report(graph: KnowledgeGraph, path: str | Path) -> Path:
    p = Path(path)
    p.write_text(render_html(graph))
    return p
