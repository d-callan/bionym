# @bionym/components

Standalone vanilla JS/D3 components for rendering
[bionym](https://github.com/d-callan/bionym) knowledge graphs.
Feed them `/api/resolve` JSON; they own their internal state, interaction,
and legend — you own the page layout, controls, and styling.

## Install

```bash
npm install @bionym/components d3
```

No build step required — plain ES modules. For a no-bundler page, use an
import map (see `example.html`).

## Components

### `BionymNetwork(el, graph, opts)`

Interactive SVG network in two layouts — `"columns"` (multi-partite by
node type) or `"force"` (force-directed, with semantic-zoom clustering).

```js
import { BionymNetwork } from "@bionym/components";
import "@bionym/components/styles.css";

const net = BionymNetwork(el, graph, {
  layout: "columns",      // "columns" | "force"
  width: 1400, height: 700, // viewBox dims; svg scales to el width
  minConfidence: 0,       // drop edges below threshold (keeps query root)
  cluster: true,          // force layout: merge connected nodes into pie glyphs when zoomed out
  legend: true,           // type-color legend under the svg
  interactive: true,      // pan/zoom/hover/select — set false for a static render
  palette: { Gene: "#0066cc" },  // merged over TYPE_PALETTE — see Palettes
  onSelect: (node, ctx) => {},  // click-to-pin; ctx = {nodes, edges, byId, pos, mode}
  onHover: (node) => {},
});

net.setLayout("force");       // re-render in the other layout
net.setCluster(false);        // toggle clustering
net.setMinConfidence(0.4);    // like the min-conf slider
net.zoomBy(1.4);              // consumer-wired zoom buttons
net.fit();                    // zoom to fit all nodes
net.resetView();              // zoom to identity
net.update(newGraph);         // new data (resets pan/zoom)
net.selected();               // currently pinned node or null
net.destroy();
```

Pan/zoom are handled inside the component (scroll/drag on the svg) —
there is deliberately **no built-in button chrome**; wire `zoomBy`/`fit`
to your own controls.

### `BionymOverview(el, graph, opts)`

Bar chart of node counts by type. `stackBy: "subject" | "object"` stacks
each bar by the predicates of edges incident on that side.

```js
const chart = BionymOverview(el, graph, {
  height: 130,
  stackBy: "off",
  onBarClick: (type) => myTable.filter({ type }),
  onSegmentClick: (type, predicate) => myTable.filter({ type, predicate }),
});
chart.setStackBy("object");
chart.update(newGraph);
chart.destroy();
```

`width` defaults to the container's width and redraws on resize; pass an
explicit `width`/`height` for a fixed size.

### `BionymDetail(el, opts)`

Evidence-trail panel for a selected node. Drive it from `onSelect`:

```js
const detail = BionymDetail(el, {
  onResolve: node => navigateTo(`/resolve/${resolveId(node)}`),
});
const net = BionymNetwork(netEl, graph, {
  onSelect: (node, ctx) => node ? detail.show(node, ctx) : detail.hide(),
});
```

Positioning is yours (the stylesheet only styles the box).

## Data helpers

For tables or your own visualizations, the JSON → tabular helpers are
exported from the root:

```js
import { nodesToRows, edgesToRows, typeCounts } from "@bionym/components";

nodesToRows(graph);  // [{id, type, label, namespace, url, degree}, ...]
edgesToRows(graph);  // [{subject, subjectLabel, predicate, object, objectLabel, confidence}, ...]
typeCounts(graph);   // [{type, count}, ...] — powers the overview chart
```

Also: `parseGraph` (dict→array normalization), `filterByConfidence`,
`predicateColors`, `resolveId`, `TYPE_ORDER`, and the palette constants below.

## Palettes

Palettes ship as their own module — drop in yours or extend ours:

```js
import { TYPE_PALETTE, PREDICATE_PALETTE, BAR_COLOR, mergePalette } from "@bionym/components/palettes";
// (also re-exported from the package root)

// full replacement — unknown types fall back to FALLBACK_COLOR
BionymNetwork(el, graph, { palette: myPalette });

// extend ours — TYPE_PALETTE is a plain {Type: "#hex"} map
BionymNetwork(el, graph, { palette: { ...TYPE_PALETTE, Gene: "#0066cc" } });

// overview chart has its own knobs
BionymOverview(el, graph, {
  barColor: BAR_COLOR,
  predicatePalette: PREDICATE_PALETTE,  // stacked-segment colors
  predicateOther: "#a0aec0",            // tail color
});
```

`palette` options are **merged over the defaults**, so partial palettes are
fine — unspecified types keep ours.

## Styling

All component classes are `bn-*` scoped (`styles.css`). Theme via CSS
variables on the container: `--bn-ink`, `--bn-muted`, `--bn-line`,
`--bn-bg` — or pass `palette` to components to re-color node types.

## Try it

Serve the repo root and open `web/components/example.html`:

```bash
cd /path/to/idresolver && python -m http.server 8080 -d web
# → http://localhost:8080/components/example.html
```

It loads a fixture graph (`example-graph.json`) through an import map —
no npm install needed.
