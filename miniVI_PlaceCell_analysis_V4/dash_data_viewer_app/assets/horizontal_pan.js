(function () {
  if (window.__traceViewerHorizontalPanInit) {
    return;
  }
  window.__traceViewerHorizontalPanInit = true;

  function finiteNumber(value) {
    var num = Number(value);
    return Number.isFinite(num) ? num : null;
  }

  function currentRange(axis) {
    if (!axis) {
      return null;
    }
    var range = axis.range || axis._range || null;
    if (!range || range.length < 2) {
      return null;
    }
    var lo = finiteNumber(range[0]);
    var hi = finiteNumber(range[1]);
    if (lo === null || hi === null || hi <= lo) {
      return null;
    }
    return [lo, hi];
  }

  function axisBounds(axis, range) {
    var lo = finiteNumber(axis.minallowed);
    var hi = finiteNumber(axis.maxallowed);
    if (lo === null) {
      lo = finiteNumber(axis._minallowed);
    }
    if (hi === null) {
      hi = finiteNumber(axis._maxallowed);
    }
    if (lo === null) {
      lo = range[0];
    }
    if (hi === null) {
      hi = range[1];
    }
    if (hi < lo) {
      var tmp = lo;
      lo = hi;
      hi = tmp;
    }
    return [lo, hi];
  }

  function plotWidth(axis, graph) {
    var width = finiteNumber(axis._length);
    if (width && width > 0) {
      return width;
    }
    var rect = graph.getBoundingClientRect();
    return Math.max(1, rect.width || 1);
  }

  document.addEventListener(
    "wheel",
    function (event) {
      if (event.defaultPrevented) {
        return;
      }
      if (event.ctrlKey || event.metaKey) {
        return;
      }
      var dx = finiteNumber(event.deltaX) || 0;
      var dy = finiteNumber(event.deltaY) || 0;
      if (Math.abs(dx) < 1 || Math.abs(dx) < Math.abs(dy) * 0.8) {
        return;
      }

      var graph = event.target && event.target.closest && event.target.closest(".js-plotly-plot");
      if (!graph || !graph._fullLayout || !window.Plotly) {
        return;
      }
      var axis = graph._fullLayout.xaxis;
      var range = currentRange(axis);
      if (!range) {
        return;
      }

      var bounds = axisBounds(axis, range);
      var span = range[1] - range[0];
      var maxSpan = bounds[1] - bounds[0];
      if (span <= 0 || maxSpan <= 0 || span >= maxSpan - 1e-12) {
        return;
      }

      var shift = (dx / plotWidth(axis, graph)) * span;
      var nextLo = range[0] + shift;
      var nextHi = range[1] + shift;

      if (nextLo < bounds[0]) {
        nextLo = bounds[0];
        nextHi = bounds[0] + span;
      }
      if (nextHi > bounds[1]) {
        nextHi = bounds[1];
        nextLo = bounds[1] - span;
      }
      if (Math.abs(nextLo - range[0]) < 1e-12 && Math.abs(nextHi - range[1]) < 1e-12) {
        return;
      }

      event.preventDefault();
      window.Plotly.relayout(graph, {
        "xaxis.range[0]": nextLo,
        "xaxis.range[1]": nextHi,
      });
    },
    { capture: true, passive: false }
  );
})();
