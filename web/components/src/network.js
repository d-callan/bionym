/**
 * BioNym network component — renders a bionym graph as an interactive
 * multi-partite ("columns") or force-directed ("force") SVG network.
 *
 *   import { BionymNetwork } from "@bionym/components";
 *   const net = BionymNetwork(el, graph, opts);
 *   net.zoomBy(1.4); net.fit(); net.setLayout("force"); net.destroy();
 *
 * Pan/zoom live inside the component; control chrome (buttons, sliders)
 * is the consumer's — everything here is a method or option.
 */
import * as d3 from "d3";
import { TYPE_ORDER, esc, ellipsize, parseGraph, filterByConfidence } from "./data.js";
import { TYPE_PALETTE, FALLBACK_COLOR, mergePalette } from "./palettes.js";

const ZMIN = 0.05, ZMAX = 6; // low floor: force layouts can span many viewport widths
const CLUSTER_R = 44;        // screen px: edges shorter than this can merge endpoints
const CLUSTER_EXT = 110;     // screen px: max extent of a merged cluster — guards
                             // against chains of short edges → one giant glyph

// Node positions: {id: {x, y}}. "columns" = multi-partite by type;
// "force" = synchronous d3.forceSimulation (cloned data — forceLink
// mutates links, which would corrupt consumer-side edge data).
function layoutNodes(nodes, edges, mode, W, H) {
  if (mode === "force") {
    const simNodes = nodes.map(n => ({ ...n }));
    const simLinks = edges.map(e => ({ source: e.subject, target: e.object }));
    const sim = d3.forceSimulation(simNodes)
      .force("link", d3.forceLink(simLinks).id(d => d.id).distance(70))
      .force("charge", d3.forceManyBody().strength(-180))
      .force("center", d3.forceCenter(W / 2, H / 2))
      .force("collide", d3.forceCollide(16))
      .stop();
    for (let i = 0; i < 300; i++) sim.tick();
    return [Object.fromEntries(simNodes.map(n => [n.id, { x: n.x, y: n.y }])), []];
  }
  const byType = d3.group(nodes, d => d.type);
  // known types in TYPE_ORDER, then any unknown types appended so a new
  // NodeType can never leave nodes unpositioned
  const cols = TYPE_ORDER.filter(t => byType.has(t))
    .concat([...byType.keys()].filter(t => !TYPE_ORDER.includes(t)));
  const x = d3.scalePoint(cols, [90, W - 90]);
  const pos = {};
  cols.forEach(t => {
    const ns = byType.get(t);
    const y = d3.scalePoint(d3.range(ns.length), [45, H - 30]);
    ns.forEach((n, i) => pos[n.id] = { x: x(t), y: y(i) });
  });
  return [pos, cols];
}

// Connectivity-aware clustering: union-find over edges whose *screen* length
// is below CLUSTER_R, shortest first. Purely spatial neighbors that share no
// edge are never merged — a pie glyph always means "these nodes are connected".
function clusterPoints(nodes, pos, edges, t) {
  const pts = new Map(nodes.map(n => [n.id, { ...n, x: pos[n.id].x, y: pos[n.id].y }]));
  const parent = new Map(), box = new Map();
  nodes.forEach(n => {
    parent.set(n.id, n.id);
    const [sx, sy] = t.apply([pos[n.id].x, pos[n.id].y]);
    box.set(n.id, [sx, sy, sx, sy]); // screen bbox of the component rooted here
  });
  const find = x => { while (parent.get(x) !== x) { parent.set(x, parent.get(parent.get(x))); x = parent.get(x); } return x; };
  edges.map(e => {
    const s = pos[e.subject], o = pos[e.object];
    if (!s || !o) return null;
    const [sx, sy] = t.apply([s.x, s.y]), [ox, oy] = t.apply([o.x, o.y]);
    return { d: Math.hypot(sx - ox, sy - oy), s: e.subject, o: e.object };
  }).filter(e => e && e.d < CLUSTER_R).sort((a, b) => a.d - b.d)
    .forEach(({ s, o }) => {
      const rs = find(s), ro = find(o);
      if (rs === ro) return;
      const a = box.get(rs), b = box.get(ro);
      const nb = [Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.max(a[2], b[2]), Math.max(a[3], b[3])];
      if (Math.max(nb[2] - nb[0], nb[3] - nb[1]) > CLUSTER_EXT) return;
      parent.set(rs, ro); box.set(ro, nb);
    });
  const groups = new Map();
  nodes.forEach(n => { const r = find(n.id); (groups.get(r) || groups.set(r, []).get(r)).push(pts.get(n.id)); });
  return [...groups.entries()].map(([r, members]) =>
    ({ key: members.length > 1 ? `c:${r}` : `s:${members[0].id}`, members }));
}

function nodeTipHtml(n, edges) {
  const inc = edges.filter(e => e.subject === n.id || e.object === n.id);
  const preds = [...d3.rollup(inc, v => v.length, e => e.predicate).entries()].sort((a, b) => b[1] - a[1]);
  return `<div class="bn-t-title">${esc(n.label || n.id)}</div>` +
    `<div class="bn-t-sub">${esc(n.id)} · ${esc(n.type)} · click for details</div>` +
    (inc.length ? `<div class="bn-t-sub">${inc.length} edges · ${preds.map(([p, c]) => `${esc(p)}×${c}`).join(" ")}</div>` : "");
}

function clusterTipHtml(c, edges) {
  if (c.count === 1) return nodeTipHtml(c.members[0], edges);
  const types = [...d3.rollup(c.members, v => v.length, m => m.type).entries()].sort((a, b) => b[1] - a[1]);
  const names = c.members.slice(0, 5).map(m => esc(m.label || m.id));
  return `<div class="bn-t-title">${c.count} nodes — click to zoom in</div>` +
    `<div class="bn-t-sub">${types.map(([ty, n]) => `${esc(ty)}×${n}`).join(" ")}</div>` +
    `<div class="bn-t-sub">${names.join(", ")}${c.count > 5 ? ", …" : ""}</div>`;
}

/**
 * @param {HTMLElement} el   container — component creates an svg + tooltip + legend inside
 * @param {object} graph     bionym graph JSON (raw API shape or normalized)
 * @param {object} [opts]
 *   layout:        "columns" | "force"            (default "columns")
 *   width/height:  viewBox dims                   (default 1400×700; svg scales to el)
 *   minConfidence: drop edges below this          (default 0)
 *   cluster:       semantic-zoom clustering in force mode (default true)
 *   legend:        render a type-color legend     (default true)
 *   interactive:   bind pan/zoom/hover/click      (default true)
 *   palette:       {type: hex} — merged over TYPE_PALETTE; import
 *                  TYPE_PALETTE from "@bionym/components/palettes" to extend ours
 *   onSelect:      (node|null, {edges, byId}) — click-to-pin; consumer renders details
 *   onHover:       (node|null)
 * @returns instance {update, setLayout, setCluster, setMinConfidence,
 *                    zoomBy, fit, resetView, selected, destroy}
 */
export function BionymNetwork(el, graph, opts = {}) {
  const o = {
    layout: "columns", width: 1400, height: 700,
    minConfidence: 0, cluster: true, legend: true, interactive: true,
    palette: {}, onSelect: null, onHover: null,
    ...opts,
  };
  const pal = mergePalette(TYPE_PALETTE, o.palette);
  const colorOf = t => pal[t] || FALLBACK_COLOR;

  el.classList.add("bn-net-wrap");
  const svg = d3.select(el).append("svg")
    .attr("class", "bn-net")
    .attr("viewBox", `0 0 ${o.width} ${o.height}`)
    .attr("preserveAspectRatio", "xMidYMid meet");
  const tipEl = document.createElement("div");
  tipEl.className = "bn-tip"; tipEl.style.display = "none";
  el.appendChild(tipEl);
  const legendEl = document.createElement("div");
  legendEl.className = "bn-legend";
  if (o.legend) el.appendChild(legendEl);

  // instance state
  let fullGraph = null;   // parsed, unfiltered — refresh() re-filters from here
  let data = null;        // filtered {nodes, edges, byId, pos, mode, labelAlways}
  let selected = null;    // pinned node id — ego highlight survives mouseleave
  let sceneQueued = false;
  const W = o.width, H = o.height;

  const zoom = d3.zoom().scaleExtent([ZMIN, ZMAX]).on("zoom", e => {
    svg.select(".viewport").attr("transform", e.transform);
    if (data && data.mode === "force" && o.cluster) queueScene();
  });
  if (o.interactive) svg.call(zoom);

  // data-space bounds → transform centering them in the viewBox
  const fitTransform = (x0, y0, x1, y1, pad = 0.85, maxK = ZMAX) => {
    const k = Math.max(ZMIN, Math.min(maxK,
      pad * Math.min(W / Math.max(20, x1 - x0), H / Math.max(20, y1 - y0))));
    return d3.zoomIdentity
      .translate(W / 2 - k * (x0 + x1) / 2, H / 2 - k * (y0 + y1) / 2).scale(k);
  };

  const showTip = (ev, html) => {
    tipEl.innerHTML = html; tipEl.style.display = "block";
    const r = tipEl.getBoundingClientRect();
    tipEl.style.left = Math.min(ev.clientX + 14, innerWidth - r.width - 10) + "px";
    tipEl.style.top = Math.min(ev.clientY + 12, innerHeight - r.height - 10) + "px";
  };
  const hideTip = () => tipEl.style.display = "none";

  // screen coords of a data-space point, for keyboard-focus tooltips
  const focusTipPoint = (x, y) => {
    const tt = d3.zoomTransform(svg.node());
    const [sx, sy] = tt.apply([x, y]);
    const sr = svg.node().getBoundingClientRect();
    return { clientX: sr.left + sx * sr.width / W, clientY: sr.top + sy * sr.height / H };
  };

  function queueScene() {
    if (sceneQueued) return;
    sceneQueued = true;
    requestAnimationFrame(() => { sceneQueued = false; if (data) drawScene(); });
  }

  function drawScene() {
    const { nodes, edges, pos, mode, byId, labelAlways } = data;
    const vp = svg.select(".viewport");
    const t = d3.zoomTransform(svg.node());
    vp.select(".links").selectAll("*").remove();
    vp.select(".nodes").selectAll("*").remove();

    const neighbors = new Map();
    edges.forEach(e => {
      (neighbors.get(e.subject) || neighbors.set(e.subject, new Set()).get(e.subject)).add(e.object);
      (neighbors.get(e.object) || neighbors.set(e.object, new Set()).get(e.object)).add(e.subject);
    });
    const nb = id => neighbors.get(id) || new Set();

    if (o.interactive) svg.on("click.bg", ev => {
      if (ev.target.closest && ev.target.closest(".nodeg")) return;
      if (selected !== null) { selected = null; o.onSelect?.(null, data); drawScene(); }
    });

    if (mode === "force" && o.cluster) { drawClustered(vp, nodes, edges, pos, byId, t, nb); return; }

    const linkSel = vp.select(".links").selectAll(".link").data(edges).join("path")
      .attr("class", "link")
      .attr("d", e => {
        const s = pos[e.subject], ob = pos[e.object];
        if (!s || !ob) return "";
        const mx = (s.x + ob.x) / 2;
        return `M${s.x},${s.y} C${mx},${s.y} ${mx},${ob.y} ${ob.x},${ob.y}`;
      })
      .attr("stroke", "#718096")
      .attr("stroke-opacity", e => 0.12 + 0.88 * e.confidence)
      .attr("stroke-width", e => 0.5 + 2.5 * e.confidence);

    function applyEgo(id) {
      const nbs = nb(id);
      nodeG.classed("dim", m => m.id !== id && !nbs.has(m.id));
      linkSel.classed("link-dim", l => l.subject !== id && l.object !== id)
        .classed("link-hi", l => l.subject === id || l.object === id);
      nodeG.select(".nodelabel").classed("on", m => m.id === id || nbs.has(m.id) || labelAlways.has(m.id));
    }
    function clearEgo() {
      nodeG.classed("dim", false);
      linkSel.classed("link-dim", false).classed("link-hi", false);
      nodeG.select(".nodelabel").classed("on", m => labelAlways.has(m.id));
    }
    const restoreSel = () => { selected && byId[selected] ? applyEgo(selected) : clearEgo(); };
    const pin = n => {
      selected = selected === n.id ? null : n.id;
      o.onSelect?.(selected ? n : null, data);
    };

    const handlers = o.interactive ? g => g
      .attr("tabindex", 0).attr("role", "button")
      .attr("aria-label", n => `${n.label || n.id} (${n.type})`)
      .style("cursor", "pointer")
      .on("click", (e, n) => pin(n))
      .on("keydown", (e, n) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pin(n); } })
      .on("mouseenter", (e, n) => { applyEgo(n.id); o.onHover?.(n); showTip(e, nodeTipHtml(n, edges)); })
      .on("mousemove", (e, n) => showTip(e, nodeTipHtml(n, edges)))
      .on("mouseleave", () => { restoreSel(); hideTip(); o.onHover?.(null); })
      .on("focus", (e, n) => { const p = pos[n.id]; showTip(focusTipPoint(p.x, p.y), nodeTipHtml(n, edges)); applyEgo(n.id); })
      .on("blur", () => { hideTip(); restoreSel(); })
      : g => g;

    const nodeG = vp.select(".nodes").selectAll(".nodeg")
      .data(nodes, n => n.id).join("g")
      .attr("class", "nodeg")
      .attr("transform", n => `translate(${pos[n.id].x},${pos[n.id].y})`)
      .call(handlers);
    nodeG.append("circle").attr("class", "node").attr("r", 5)
      .attr("fill", n => colorOf(n.type));
    nodeG.append("text").attr("class", "nodelabel").attr("x", 7).attr("y", 3)
      .classed("on", n => labelAlways.has(n.id))
      .text(n => ellipsize(n.label || n.id));
    restoreSel();
  }

  function zoomToFit(members) {
    const x0 = d3.min(members, m => m.x), x1 = d3.max(members, m => m.x);
    const y0 = d3.min(members, m => m.y), y1 = d3.max(members, m => m.y);
    svg.transition().duration(450).call(zoom.transform, fitTransform(x0, y0, x1, y1));
  }

  // Clustered rendering: connectivity clusters → pie glyphs (slices = type
  // mix, radius ∝ sqrt(count)), one aggregated edge per cluster-pair.
  function drawClustered(vp, nodes, edges, pos, byId, t, nb) {
    const clusters = clusterPoints(nodes, pos, edges, t);
    clusters.forEach((c, i) => {
      c.i = i; c.count = c.members.length;
      c.x = d3.mean(c.members, m => m.x);
      c.y = d3.mean(c.members, m => m.y);
    });
    const memberCl = new Map();
    clusters.forEach(c => c.members.forEach(m => memberCl.set(m.id, c)));

    const adj = new Map(), agg = new Map();
    edges.forEach(e => {
      const a = memberCl.get(e.subject), b = memberCl.get(e.object);
      if (!a || !b || a === b) return;
      (adj.get(a) || adj.set(a, new Set()).get(a)).add(b);
      (adj.get(b) || adj.set(b, new Set()).get(b)).add(a);
      const key = a.i < b.i ? a.i + "|" + b.i : b.i + "|" + a.i;
      const A = agg.get(key) || { a, b, n: 0, conf: 0, preds: {} };
      A.n++; A.conf += e.confidence; A.preds[e.predicate] = (A.preds[e.predicate] || 0) + 1;
      agg.set(key, A);
    });
    const links = [...agg.values()];

    const linkSel = vp.select(".links").selectAll(".link").data(links).join("path")
      .attr("class", "link")
      .attr("d", L => `M${L.a.x},${L.a.y} C${(L.a.x + L.b.x) / 2},${L.a.y} ${(L.a.x + L.b.x) / 2},${L.b.y} ${L.b.x},${L.b.y}`)
      .attr("stroke", "#718096")
      .attr("stroke-opacity", L => 0.12 + 0.88 * (L.conf / L.n))
      .attr("stroke-width", L => 0.8 + 1.6 * Math.sqrt(L.n));
    linkSel.append("title").text(L => `${L.n} edge${L.n > 1 ? "s" : ""} · ` +
      Object.entries(L.preds).sort((x, y) => y[1] - x[1]).map(([p, c]) => `${p}×${c}`).join(" "));

    const r = d3.scaleSqrt().domain([1, Math.max(2, d3.max(clusters, c => c.count))]).range([5, 26]);
    const pie = d3.pie().value(d => d[1]), arc = d3.arc().innerRadius(0);

    function applyC(c) {
      const nbs = adj.get(c) || new Set();
      cg.classed("dim", m => m !== c && !nbs.has(m));
      linkSel.classed("link-dim", L => L.a !== c && L.b !== c)
        .classed("link-hi", L => L.a === c || L.b === c);
      cg.select(".nodelabel").classed("on", m => m.count > 1 || m === c || nbs.has(m));
    }
    function clearC() {
      cg.classed("dim", false);
      linkSel.classed("link-dim", false).classed("link-hi", false);
      cg.select(".nodelabel").classed("on", m => m.count > 1);
    }
    const restoreC = () => { const s = selected ? memberCl.get(selected) : null; s ? applyC(s) : clearC(); };
    const activate = c => {
      hideTip();
      if (c.count === 1) {
        const m = c.members[0];
        selected = selected === m.id ? null : m.id;
        o.onSelect?.(selected ? m : null, data);
      } else zoomToFit(c.members);
    };

    const handlers = o.interactive ? g => g
      .attr("tabindex", 0).attr("role", "button")
      .attr("aria-label", c => c.count > 1
        ? `cluster of ${c.count} nodes`
        : `${c.members[0].label || c.members[0].id} (${c.members[0].type})`)
      .style("cursor", "pointer")
      .on("click", (e, c) => activate(c))
      .on("keydown", (e, c) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); activate(c); } })
      .on("mouseenter", (e, c) => { applyC(c); showTip(e, clusterTipHtml(c, edges)); })
      .on("mousemove", (e, c) => showTip(e, clusterTipHtml(c, edges)))
      .on("mouseleave", () => { restoreC(); hideTip(); })
      .on("focus", (e, c) => { applyC(c); showTip(focusTipPoint(c.x, c.y), clusterTipHtml(c, edges)); })
      .on("blur", () => { restoreC(); hideTip(); })
      : g => g;

    const cg = vp.select(".nodes").selectAll(".nodeg")
      .data(clusters, c => c.key).join("g")
      .attr("class", c => "nodeg" + (c.count > 1 ? " cluster" : ""))
      .attr("transform", c => `translate(${c.x},${c.y})`)
      .call(handlers);

    cg.each(function (c) {
      const g = d3.select(this);
      if (c.count > 1) {
        const arcs = pie([...d3.rollup(c.members, v => v.length, m => m.type).entries()]);
        g.selectAll("path.slice").data(arcs).join("path").attr("class", "slice")
          .attr("d", arc.outerRadius(r(c.count)))
          .attr("fill", d => colorOf(d.data[0]))
          .attr("stroke", "#fff").attr("stroke-width", 1);
        g.append("text").attr("class", "nodelabel on").attr("text-anchor", "middle")
          .attr("dy", "0.35em").attr("font-weight", "600").text(c.count);
      } else {
        const m = c.members[0];
        g.append("circle").attr("class", "node").attr("r", 5).attr("fill", colorOf(m.type));
        g.append("text").attr("class", "nodelabel").attr("x", 7).attr("y", 3).text(ellipsize(m.label || m.id));
      }
    });
    restoreC();
  }

  function renderGraph(g) {
    const parsed = parseGraph(g);
    fullGraph = parsed;
    const filtered = o.minConfidence > 0
      ? filterByConfidence(parsed, o.minConfidence, parsed.metadata.query)
      : parsed;
    const { nodes, edges, byId } = filtered;
    const mode = o.layout;

    svg.selectAll("*").remove();
    const [pos, cols] = layoutNodes(nodes, edges, mode, W, H);

    const vp = svg.append("g").attr("class", "viewport");
    const t0 = d3.zoomTransform(svg.node());
    // fresh graph (identity transform): fit force layouts to node bounds —
    // they routinely spill outside the viewBox
    if (mode === "force" && nodes.length && t0.k === 1 && !t0.x && !t0.y) {
      const xs = nodes.map(n => pos[n.id].x), ys = nodes.map(n => pos[n.id].y);
      svg.call(zoom.transform, fitTransform(d3.min(xs), d3.min(ys), d3.max(xs), d3.max(ys), 0.9, 1.5));
    }
    vp.attr("transform", d3.zoomTransform(svg.node()));

    if (mode === "columns") {
      const x = d3.scalePoint(cols, [90, W - 90]);
      vp.append("g").selectAll(".colhead").data(cols).join("text")
        .attr("class", "colhead").attr("x", d => x(d)).attr("y", 20)
        .attr("text-anchor", "middle").text(d => d);
    }
    vp.append("g").attr("class", "links");
    vp.append("g").attr("class", "nodes");

    // Adaptive labels: label every node only when its column has room for
    // 9px text; dense columns get labels on hover/click instead.
    const labelAlways = new Set();
    if (mode === "columns") {
      const byT = d3.group(nodes, n => n.type);
      byT.forEach(ns => { if (ns.length && (H - 75) / ns.length >= 12) ns.forEach(n => labelAlways.add(n.id)); });
    }

    selected = null;
    data = { nodes, edges, pos, mode, byId, labelAlways };
    drawScene();

    if (o.legend) {
      const typesPresent = [...new Set(nodes.map(n => n.type))];
      legendEl.innerHTML = typesPresent
        .map(t => `<span class="leg"><i style="background:${colorOf(t)}"></i>${esc(t)}</span>`)
        .join("");
      legendEl.hidden = !nodes.length;
    }
  }

  if (graph) renderGraph(graph);

  return {
    /** Replace the graph (resets pan/zoom to auto-fit). */
    update(g) {
      svg.call(zoom.transform, d3.zoomIdentity);
      renderGraph(g);
    },
    /** Re-render the current graph (e.g. after option changes). */
    refresh() { if (fullGraph) renderGraph(fullGraph); },
    setLayout(m) { o.layout = m; this.refresh(); },
    setCluster(v) { o.cluster = v; if (data) drawScene(); },
    setMinConfidence(v) { o.minConfidence = v; this.refresh(); },
    zoomBy(k) { svg.transition().duration(200).call(zoom.scaleBy, k); },
    /** Zoom so all nodes fit the viewport. */
    fit() {
      if (data && data.nodes.length) {
        const xs = data.nodes.map(n => data.pos[n.id].x), ys = data.nodes.map(n => data.pos[n.id].y);
        svg.transition().duration(300)
          .call(zoom.transform, fitTransform(d3.min(xs), d3.min(ys), d3.max(xs), d3.max(ys), 0.9, 1.5));
      } else this.resetView();
    },
    resetView() { svg.transition().duration(300).call(zoom.transform, d3.zoomIdentity); },
    selected() { return selected && data ? data.byId[selected] : null; },
    destroy() { svg.on(".zoom", null); el.innerHTML = ""; el.classList.remove("bn-net-wrap"); },
  };
}
