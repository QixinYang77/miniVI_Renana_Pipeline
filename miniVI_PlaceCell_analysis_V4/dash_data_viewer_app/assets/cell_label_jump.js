(function () {
  if (window.__traceViewerCellLabelJumpInit) {
    return;
  }
  window.__traceViewerCellLabelJumpInit = true;

  function normalizeLabel(text) {
    return String(text || "")
      .replace(/<br\s*\/?>/gi, "")
      .replace(/\s+/g, "")
      .trim();
  }

  function jumpRows(graph) {
    if (!graph) {
      return [];
    }
    var meta = null;
    if (graph._fullLayout && graph._fullLayout.meta) {
      meta = graph._fullLayout.meta;
    } else if (graph.layout && graph.layout.meta) {
      meta = graph.layout.meta;
    }
    return meta && Array.isArray(meta.cell_jump_rows) ? meta.cell_jump_rows : [];
  }

  function rowForTick(graph, tickText) {
    var rows = jumpRows(graph);
    if (!rows.length) {
      return null;
    }
    var key = normalizeLabel(tickText.textContent);
    for (var i = 0; i < rows.length; i += 1) {
      if (normalizeLabel(rows[i].label) === key) {
        return rows[i];
      }
    }

    var ticks = Array.prototype.slice.call(graph.querySelectorAll(".ytick text"));
    var idx = ticks.indexOf(tickText);
    if (idx >= 0 && idx < rows.length) {
      return rows[idx];
    }
    return null;
  }

  function sendJump(row) {
    if (!row || !window.dash_clientside || !window.dash_clientside.set_props) {
      return false;
    }
    window.dash_clientside.set_props("cell-jump-request", {
      data: {
        animal_id: row.animal_id,
        cell_idx: row.cell_idx,
        nonce: Date.now() + Math.random(),
      },
    });
    return true;
  }

  document.addEventListener(
    "click",
    function (event) {
      var tickText = event.target && event.target.closest && event.target.closest(".ytick text");
      if (!tickText) {
        return;
      }
      var graph = tickText.closest(".js-plotly-plot");
      var row = rowForTick(graph, tickText);
      if (!row) {
        return;
      }
      if (sendJump(row)) {
        event.preventDefault();
        event.stopPropagation();
      }
    },
    true
  );

  document.addEventListener(
    "mouseover",
    function (event) {
      var tickText = event.target && event.target.closest && event.target.closest(".ytick text");
      if (!tickText) {
        return;
      }
      var graph = tickText.closest(".js-plotly-plot");
      if (!rowForTick(graph, tickText)) {
        return;
      }
      tickText.style.cursor = "pointer";
      tickText.setAttribute("title", "Open this cell in Single cell view");
    },
    true
  );
})();
