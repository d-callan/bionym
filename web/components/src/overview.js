/**
 * BioNym overview bar chart — node counts by type. Flat bars by default;
 * stackBy "subject"/"object" stacks each bar by the predicates of edges
 * incident on that side of the type.
 *
 *   const chart = BionymOverview(el, graph, {
 *     stackBy: "object",
 *     onBarClick: type => myFilter.set("type", type),
 *   });
 */
import * as d3 from "d3";
import { TYPE_ORDER, parseGraph, predicateColors } from "./data.js";
import { BAR_COLOR, PREDICATE_PALETTE, PREDICATE_OTHER } from "./palettes.js";

/**
 * @param {HTMLElement} el   container — component creates an svg inside
 * @param {object} graph     bionym graph JSON (raw or normalized)
 * @param {object} [opts]
 *   width:      px — null/undefined → fill container (clientWidth)
 *   height:     px (default 130)
 *   stackBy:    "off" | "subject" | "object"  (default "off")
 *   barColor:   fill for flat bars (default BAR_COLOR from palettes)
 *   predicatePalette: array of segment colors (default PREDICATE_PALETTE)
 *   predicateOther:   tail color for predicates past the palette length
 *   onBarClick: (nodeType, event) — click a bar to e.g. filter a table
 *   onSegmentClick: (nodeType, predicate, event) — stacked mode only
 * @returns {update, setStackBy, destroy}
 */
export function BionymOverview(el, graph, opts = {}) {
  const o = {
    width: null, height: 130, stackBy: "off",
    barColor: BAR_COLOR, predicatePalette: PREDICATE_PALETTE,
    predicateOther: PREDICATE_OTHER,
    onBarClick: null, onSegmentClick: null,
    ...opts,
  };
  el.classList.add("bn-ov-wrap");
  const svg = d3.select(el).append("svg").attr("class", "bn-ov");
  let data = null;

  function draw() {
    if (!data) return;
    const { nodes, edges, byId } = data;
    svg.selectAll("*").remove();
    el.hidden = !nodes.length;
    if (!nodes.length) return;

    const W = o.width || Math.max(600, el.clientWidth || 900);
    const H = o.height, m = { t: 28, r: 10, b: 30, l: 36 };
    svg.attr("viewBox", `0 0 ${W} ${H}`);

    const byType = d3.group(nodes, n => n.type);
    const types = TYPE_ORDER.filter(ty => byType.has(ty))
      .concat([...byType.keys()].filter(ty => !TYPE_ORDER.includes(ty)));
    const x = d3.scaleBand(types, [m.l, W - m.r]).padding(0.25);
    const pc = predicateColors(edges, o.predicatePalette, o.predicateOther);

    let stacks = null, yMax;
    if (o.stackBy === "off") {
      yMax = d3.max(types, ty => byType.get(ty).length) || 1;
    } else {
      const acc = new Map(types.map(ty => [ty, new Map()]));
      edges.forEach(e => {
        const n = byId[o.stackBy === "subject" ? e.subject : e.object];
        if (!n || !acc.has(n.type)) return;
        acc.get(n.type).set(e.predicate, (acc.get(n.type).get(e.predicate) || 0) + 1);
      });
      stacks = new Map(types.map(ty => {
        let y0 = 0;
        return [ty, [...acc.get(ty).entries()].sort((a, b) => b[1] - a[1])
          .map(([p, c]) => ({ pred: p, v0: y0, v1: (y0 += c) }))];
      }));
      yMax = d3.max(types, ty => d3.sum([...acc.get(ty).values()])) || 1;
    }
    const y = d3.scaleLinear([0, yMax], [H - m.b, m.t]);
    svg.append("g").attr("transform", `translate(${m.l},0)`)
      .call(d3.axisLeft(y).ticks(4).tickSize(0))
      .call(g => g.select(".domain").remove());

    const bg = svg.selectAll(".bar").data(types).join("g")
      .attr("class", "bar")
      .attr("transform", ty => `translate(${x(ty)},0)`);
    if (o.onBarClick || o.onSegmentClick) {
      bg.style("cursor", "pointer")
        .on("click", (ev, ty) => o.onBarClick && o.onBarClick(ty, ev));
    }
    if (o.stackBy === "off") {
      bg.append("rect").attr("y", ty => y(byType.get(ty).length))
        .attr("height", ty => H - m.b - y(byType.get(ty).length))
        .attr("width", x.bandwidth()).attr("fill", o.barColor)
        .append("title").text(ty => `${ty}: ${byType.get(ty).length} nodes`);
      bg.append("text").attr("x", x.bandwidth() / 2).attr("text-anchor", "middle")
        .attr("y", ty => y(byType.get(ty).length) - 3).text(ty => byType.get(ty).length);
    } else {
      bg.selectAll("rect.seg").data(ty => stacks.get(ty)).join("rect")
        .attr("class", "seg")
        .attr("y", s => y(s.v1)).attr("height", s => y(s.v0) - y(s.v1))
        .attr("width", x.bandwidth()).attr("fill", s => pc.color(s.pred))
        .on("click", o.onSegmentClick
          ? (ev, s) => { ev.stopPropagation(); o.onSegmentClick(d3.select(ev.target.parentNode).datum(), s.pred, ev); }
          : null)
        .append("title").text(s => `${s.pred} ×${s.v1 - s.v0}`);
      // predicate legend (only for predicates actually stacked)
      const vis = new Set(); stacks.forEach(segs => segs.forEach(s => vis.add(s.pred)));
      const items = pc.ranked.filter(([p]) => vis.has(p)).slice(0, 8).map(([p]) => p);
      if ([...vis].some(p => pc.color(p) === o.predicateOther)) items.push("other");
      const leg = svg.append("g").attr("transform", `translate(${m.l + 6},6)`);
      items.forEach((p, i) => {
        const lx = (i % 5) * 115, ly = Math.floor(i / 5) * 12;
        leg.append("rect").attr("x", lx).attr("y", ly).attr("width", 8).attr("height", 8)
          .attr("fill", p === "other" ? o.predicateOther : pc.color(p));
        leg.append("text").attr("x", lx + 11).attr("y", ly + 7).text(p);
      });
    }
    bg.append("text").attr("x", x.bandwidth() / 2).attr("y", H - m.b + 12)
      .attr("text-anchor", "middle").text(ty => ty);
  }

  // auto-width redraws need a resize observer on the container
  let ro = null;
  if (!o.width && typeof ResizeObserver !== "undefined") {
    ro = new ResizeObserver(() => draw());
    ro.observe(el);
  }

  if (graph) { data = parseGraph(graph); draw(); }

  return {
    update(g) { data = parseGraph(g); draw(); },
    setStackBy(side) { o.stackBy = side; draw(); },
    destroy() { ro?.disconnect(); svg.remove(); el.classList.remove("bn-ov-wrap"); },
  };
}
