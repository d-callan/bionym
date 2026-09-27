/**
 * Pure data helpers for bionym graph JSON — no DOM, no D3.
 * Everything here accepts either the raw API shape
 *   {nodes: {id: node}, edges: [edge], metadata: {...}}
 * or an already-normalized {nodes: [node], edges: [edge]}.
 */

import { PREDICATE_PALETTE, PREDICATE_OTHER } from "./palettes.js";

/** Node types in display order (column order for the multi-partite layout). */
export const TYPE_ORDER = [
  "IdType", "Gene", "Organism", "Assembly", "OrthologGroup", "Transcript",
  "Protein", "Domain", "GOTerm", "Pathway", "Dataset", "Condition",
  "Publication",
];

// Palettes live in ./palettes.js (also importable as "@bionym/components/palettes").
// Re-exported here and from ./index.js for convenience.
export {
  TYPE_PALETTE as PALETTE,
  PREDICATE_PALETTE as PRED_PALETTE,
  PREDICATE_OTHER as PRED_OTHER,
  FALLBACK_COLOR,
} from "./palettes.js";

export const esc = s =>
  String(s ?? "").replace(/[&<>"]/g, c =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

export const ellipsize = (s, n = 26) =>
  s.length > n ? s.slice(0, n - 1) + "…" : s;

/**
 * Normalize raw graph JSON to {nodes: [...], edges: [...], metadata}.
 * Tolerates nodes as dict (API shape) or array.
 */
export function parseGraph(g) {
  const nodes = Array.isArray(g.nodes) ? g.nodes : Object.values(g.nodes || {});
  const edges = g.edges || [];
  const byId = Object.fromEntries(nodes.map(n => [n.id, n]));
  return { nodes, edges, metadata: g.metadata || {}, byId };
}

/**
 * Drop edges below `minConfidence`, then nodes with no remaining edges.
 * `rootId` (usually metadata.query) is always kept. Mirrors the backend's
 * KnowledgeGraph.filter_by_confidence.
 */
export function filterByConfidence(graph, minConfidence, rootId) {
  const { nodes, edges, metadata } = parseGraph(graph);
  const keepEdges = edges.filter(e => e.confidence >= minConfidence);
  const keep = new Set(keepEdges.flatMap(e => [e.subject, e.object]));
  const root = rootId || metadata.query;
  if (root) keep.add(root);
  const keepNodes = nodes.filter(n => keep.has(n.id));
  return { nodes: keepNodes, edges: keepEdges, metadata, byId: Object.fromEntries(keepNodes.map(n => [n.id, n])) };
}

/** Rows for a nodes table: one flat object per node. */
export function nodesToRows(graph) {
  return parseGraph(graph).nodes.map(n => ({
    id: n.id,
    type: n.type,
    label: n.label || n.id,
    namespace: n.id_namespace,
    url: n.url,
    degree: n.degree,
  }));
}

/** Rows for an edges table. subjectLabel/objectLabel resolved via byId. */
export function edgesToRows(graph) {
  const { edges, byId } = parseGraph(graph);
  return edges.map(e => ({
    subject: e.subject,
    subjectLabel: byId[e.subject]?.label || e.subject,
    predicate: e.predicate,
    object: e.object,
    objectLabel: byId[e.object]?.label || e.object,
    confidence: e.confidence,
  }));
}

/** [{type, count}] in TYPE_ORDER (unknown types appended). Feeds the
 *  overview chart or a consumer's own chart. */
export function typeCounts(graph) {
  const { nodes } = parseGraph(graph);
  const byType = new Map();
  nodes.forEach(n => byType.set(n.type, (byType.get(n.type) || 0) + 1));
  const ordered = TYPE_ORDER.filter(t => byType.has(t))
    .concat([...byType.keys()].filter(t => !TYPE_ORDER.includes(t)));
  return ordered.map(t => ({ type: t, count: byType.get(t) }));
}

/**
 * Rank predicates by edge count and assign colors from `palette`
 * (default PREDICATE_PALETTE). Returns {color(pred), ranked: [[pred, count], ...]}.
 */
export function predicateColors(edges, palette = PREDICATE_PALETTE, other = PREDICATE_OTHER) {
  const counts = new Map();
  edges.forEach(e => counts.set(e.predicate, (counts.get(e.predicate) || 0) + 1));
  const ranked = [...counts.entries()].sort((a, b) => b[1] - a[1]);
  const m = new Map(ranked.slice(0, palette.length).map((d, i) => [d[0], palette[i]]));
  return { color: p => m.get(p) || other, ranked };
}

/**
 * Best resolvable identifier for a node — the value a "resolve →" action
 * would feed back into /api/resolve (e.g. follow an ortholog when the
 * query gene has no downstream data).
 */
export function resolveId(n) {
  if (n.attrs && n.attrs.resolve_id) return String(n.attrs.resolve_id);
  if (n.attrs && n.attrs.gene_id) return String(n.attrs.gene_id);
  const rest = n.id.includes(":") ? n.id.split(":").slice(1).join(":") : n.id;
  return rest || n.id;
}
