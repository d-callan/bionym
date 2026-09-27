/**
 * Driver script for `bionym resolve --report` HTML output.
 * Inlined into the generated page after the IIFE bundle and graph JSON.
 * Expects: window.BionymComponents (bundle), window.__GRAPH__ (JSON).
 */
(function () {
  const { BionymNetwork, BionymOverview, BionymDetail, parseGraph, esc } =
    window.BionymComponents;
  const graph = window.__GRAPH__;
  const $ = s => document.querySelector(s);

  const g = parseGraph(graph);
  const byId = g.byId;

  // collapsible summary block (only when the resolver ran summarize)
  const sum = g.metadata.summary;
  const det = $("#summary");
  det.hidden = !sum || !sum.length;
  if (!det.hidden) {
    $("#summarylist").innerHTML = sum
      .map(s => `<li>${esc(s.claim)} <span class="sumconf">${(+s.confidence).toFixed(2)}</span></li>`)
      .join("");
  }

  const detail = BionymDetail($("#detail"));
  const net = BionymNetwork($("#net"), graph, {
    onSelect: (n, ctx) => (n ? detail.show(n, ctx) : detail.hide()),
  });
  const ov = BionymOverview($("#ov"), graph);

  // ---- control chrome → component methods ----------------------------------
  $("#resetzoom").addEventListener("click", () => net.fit());
  $("#zin").addEventListener("click", () => net.zoomBy(1.4));
  $("#zout").addEventListener("click", () => net.zoomBy(1 / 1.4));
  $("#layout").addEventListener("change", () => {
    $("#clustwrap").hidden = $("#layout").value !== "force";
    net.setLayout($("#layout").value);
  });
  $("#clustergroup").addEventListener("change", () => net.setCluster($("#clustergroup").checked));
  $("#stacksel").addEventListener("change", () => ov.setStackBy($("#stacksel").value));

  // ---- sortable + filterable tables ----------------------------------------
  const link = n => n.url
    ? `<a href="${esc(n.url)}" target="_blank" rel="noopener">${esc(n.label || n.id)}</a>`
    : esc(n.label || n.id);
  const cell = id => byId[id] ? link(byId[id]) : esc(id);

  function sortable(tableId, rows, cols, initKey = null, initDir = 1) {
    const tbl = document.getElementById(tableId);
    const tbody = tbl.querySelector("tbody");
    const ths = [...tbl.querySelectorAll("th")];
    const state = { key: initKey, dir: initDir, filter: "" };
    const fi = document.querySelector(`.tfilter[data-for="${tableId}"]`);
    if (fi) fi.oninput = e => { state.filter = e.target.value.toLowerCase(); draw(); };
    ths.forEach((th, i) => th.onclick = () => {
      if (state.key === i) state.dir *= -1; else { state.key = i; state.dir = 1; }
      draw();
    });
    function draw() {
      let rs = rows.filter(r => !state.filter || cols.some(c => String(c.val(r) ?? "").toLowerCase().includes(state.filter)));
      if (state.key !== null) {
        const c = cols[state.key];
        rs = rs.slice().sort((a, b) => {
          const x = c.val(a), y = c.val(b);
          return (typeof x === "number" && typeof y === "number" ? x - y
            : String(x ?? "").localeCompare(String(y ?? ""))) * state.dir;
        });
      }
      tbody.innerHTML = rs.map(r => "<tr>" +
        cols.map(c => `<td class="${c.cls || ""}">${c.render(r)}</td>`).join("") +
        "</tr>").join("");
      ths.forEach((th, i) => th.textContent =
        cols[i].label + (state.key === i ? (state.dir > 0 ? " ▲" : " ▼") : ""));
    }
    draw();
  }

  sortable("edges", g.edges, [
    { label: "subject", val: e => e.subject, render: e => cell(e.subject) },
    { label: "predicate", val: e => e.predicate, render: e => esc(e.predicate) },
    { label: "object", val: e => e.object, render: e => cell(e.object) },
    { label: "conf", val: e => e.confidence, render: e => e.confidence.toFixed(2), cls: "conf" },
    { label: "ev", val: e => (e.evidence || []).length, render: e => (e.evidence || []).length },
  ], 3, -1);
  sortable("nodes", g.nodes, [
    { label: "id", val: n => n.id, render: n => esc(n.id) },
    { label: "type", val: n => n.type, render: n => esc(n.type) },
    { label: "label", val: n => n.label || n.id, render: n => link(n) },
    { label: "namespace", val: n => n.id_namespace, render: n => esc(n.id_namespace) },
  ], 1, 1);
})();
