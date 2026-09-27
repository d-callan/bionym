/**
 * BioNym detail panel — shows a node's identity + incident-edge evidence
 * trail. Ship it separately so consumers can mount it wherever they like:
 *
 *   const detail = BionymDetail(el, {
 *     onResolve: node => fetch(`/api/resolve?identifier=${resolveId(node)}`),
 *   });
 *   const net = BionymNetwork(netEl, graph, {
 *     onSelect: (node, ctx) => node ? detail.show(node, ctx) : detail.hide(),
 *   });
 */
import { esc } from "./data.js";

/**
 * @param {HTMLElement} el  the panel element — component fills its innerHTML
 * @param {object} [opts]
 *   onResolve: (node) — renders a "resolve →" action for Gene nodes when set
 * @returns {show(node, ctx), hide(), destroy()}
 *   ctx is the second arg BionymNetwork passes to onSelect: {edges, byId}
 */
export function BionymDetail(el, opts = {}) {
  const o = { onResolve: null, ...opts };
  el.classList.add("bn-detail");
  el.hidden = true;

  return {
    show(n, { edges = [], byId = {} } = {}) {
      const inc = edges.filter(e => e.subject === n.id || e.object === n.id);
      el.innerHTML =
        `<button class="bn-d-close" title="close">✕</button>` +
        `<h3>${esc(n.label || n.id)}</h3>` +
        `<div class="bn-d-sub">${esc(n.id)} · ${esc(n.type)}` +
        `${n.id_namespace ? " · " + esc(n.id_namespace) : ""}</div><ul>` +
        inc.map(e => {
          const other = e.subject === n.id ? e.object : e.subject;
          const on = byId[other];
          return `<li>${e.subject === n.id ? "→" : "←"} ${esc(e.predicate)} ` +
            `${esc(on ? on.label || other : other)} ` +
            `<span class="bn-conf">${e.confidence.toFixed(2)}</span></li>`;
        }).join("") + `</ul><div class="bn-d-actions">` +
        (n.url ? `<a href="${esc(n.url)}" target="_blank" rel="noopener">source ↗</a>` : "") +
        (o.onResolve && n.type === "Gene" ? `<a href="#" class="bn-d-resolve">resolve →</a>` : "") +
        `</div>`;
      el.hidden = false;
      el.querySelector(".bn-d-close").onclick = () => { el.hidden = true; };
      const r = el.querySelector(".bn-d-resolve");
      if (r) r.onclick = e => { e.preventDefault(); o.onResolve(n); };
    },
    hide() { el.hidden = true; },
    visible() { return !el.hidden; },
    destroy() { el.innerHTML = ""; el.hidden = true; el.classList.remove("bn-detail"); },
  };
}
