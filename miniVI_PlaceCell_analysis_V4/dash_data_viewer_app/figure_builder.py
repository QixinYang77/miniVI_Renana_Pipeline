"""Plotly figure builders for the Dash trace viewer."""

from __future__ import annotations

import html as html_lib
import json
from pathlib import Path
from typing import Any

import numpy as np
import plotly
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
from plotly.utils import PlotlyJSONEncoder

try:
    from .data_io import (
        AnimalData,
        bad_mask_for_cell,
        compute_snr_values,
        default_data_root,
        discover_animals,
        dict_intervals,
        get_cell_vector,
        load_animal,
        mask_to_intervals,
    )
except ImportError:  # pragma: no cover - direct script imports
    from data_io import (
        AnimalData,
        bad_mask_for_cell,
        compute_snr_values,
        default_data_root,
        discover_animals,
        dict_intervals,
        get_cell_vector,
        load_animal,
        mask_to_intervals,
    )


TRACE_COLOR = "#111111"
VM_COLOR = "#6A4C93"
SS_COLOR = "#026C80"
CS_COLOR = "#EE9B00"
CB_FILL = "rgba(238, 155, 0, 0.20)"
PLATEAU_FILL = "rgba(214, 40, 40, 0.24)"
BAD_FILL = "rgba(120, 120, 120, 0.28)"
SESSION_LINE = "rgba(0, 0, 0, 0.35)"
TRACE_SCALE = 0.62
TRACE_HALF_HEIGHT = TRACE_SCALE / 2.0
TRACE_PAD = 0.12
RASTER_PAD = 0.12
RASTER_HEIGHT = 0.24
ROW_GAP = 0.36
PLOT_PAD = 0.08
DEFAULT_HTML_TRACE_MAX_POINTS = 20_000
DEFAULT_ROW_HEIGHT_PX = 112
DEFAULT_STACKED_ROW_HEIGHT_PX = 135
MIN_ROW_HEIGHT_PX = 60
MAX_ROW_HEIGHT_PX = 180
CATEGORY_HTML_FILENAMES = {
    "CS+ PLC": "pooled_place_cells_CSplus_PLC_trace_viewer.html",
    "CS- PLC": "pooled_place_cells_CSminus_PLC_trace_viewer.html",
}
POOLED_HTML_WHEEL_PAN_SCRIPT = r"""
(function () {
  var graph = document.getElementById("{plot_id}");
  if (!graph || graph.__pooledWheelPanInit) {
    return;
  }
  graph.__pooledWheelPanInit = true;

  function finiteNumber(value) {
    var num = Number(value);
    return Number.isFinite(num) ? num : null;
  }

  function axisRange(axis) {
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

  function axisPixels(axis, fallbackPixels) {
    var px = finiteNumber(axis && axis._length);
    if (px !== null && px > 0) {
      return px;
    }
    return Math.max(1, fallbackPixels || 1);
  }

  function pan(axisName, deltaPixels, fallbackPixels, direction, update) {
    var layout = graph._fullLayout || {};
    var axis = layout[axisName];
    var range = axisRange(axis);
    if (!range) {
      return false;
    }
    var bounds = axisBounds(axis, range);
    var span = range[1] - range[0];
    var maxSpan = bounds[1] - bounds[0];
    if (span <= 0 || maxSpan <= 0 || span >= maxSpan - 1e-12) {
      return false;
    }

    var shift = direction * (deltaPixels / axisPixels(axis, fallbackPixels)) * span;
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
      return false;
    }
    update[axisName + ".range[0]"] = nextLo;
    update[axisName + ".range[1]"] = nextHi;
    return true;
  }

  graph.addEventListener(
    "wheel",
    function (event) {
      if (event.defaultPrevented || event.ctrlKey || event.metaKey || !window.Plotly || !graph._fullLayout) {
        return;
      }
      var dx = finiteNumber(event.deltaX) || 0;
      var dy = finiteNumber(event.deltaY) || 0;
      var absX = Math.abs(dx);
      var absY = Math.abs(dy);
      if (absX < 1 && absY < 1) {
        return;
      }

      var rect = graph.getBoundingClientRect();
      var update = {};
      var changed = false;
      if (absX >= absY * 0.8 && absX >= 1) {
        changed = pan("xaxis", dx, rect.width, 1, update);
      } else if (absY >= 1) {
        changed = pan("yaxis", dy, rect.height, -1, update);
      }
      if (!changed) {
        return;
      }
      event.preventDefault();
      window.Plotly.relayout(graph, update);
    },
    { passive: false }
  );
})();
"""
POOLED_HTML_CONFIG = {
    "scrollZoom": False,
    "doubleClick": "reset",
    "displaylogo": False,
    "responsive": True,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
}


def _clean_row_height_px(row_height_px: float | int | None, fallback: int = DEFAULT_ROW_HEIGHT_PX) -> int:
    try:
        value = float(row_height_px)
    except Exception:
        value = float(fallback)
    if not np.isfinite(value):
        value = float(fallback)
    value = min(MAX_ROW_HEIGHT_PX, max(MIN_ROW_HEIGHT_PX, value))
    return int(round(value))


def _plotlyjs_include_html(include_plotlyjs: bool | str) -> str:
    if include_plotlyjs is True:
        return f"<script>{get_plotlyjs()}</script>"
    if include_plotlyjs in {"cdn", "cdn-v3"}:
        version = html_lib.escape(str(plotly.__version__))
        return f'<script src="https://cdn.plot.ly/plotly-{version}.min.js"></script>'
    if isinstance(include_plotlyjs, str) and include_plotlyjs.startswith(("http://", "https://")):
        return f'<script src="{html_lib.escape(include_plotlyjs, quote=True)}"></script>'
    return ""


def _safe_json(value: Any) -> str:
    return json.dumps(value, cls=PlotlyJSONEncoder, separators=(",", ":")).replace("</", "<\\/")


def _segment_windows(max_duration_s: float, segment_duration_s: float | None) -> list[tuple[float, float]]:
    max_duration_s = max(0.0, float(max_duration_s))
    if segment_duration_s is None:
        return [(0.0, max_duration_s)]
    duration_s = float(segment_duration_s)
    if (not np.isfinite(duration_s)) or duration_s <= 0 or max_duration_s <= 0:
        return [(0.0, max_duration_s)]

    windows: list[tuple[float, float]] = []
    start_s = 0.0
    while start_s < max_duration_s - 1e-9:
        end_s = min(max_duration_s, start_s + duration_s)
        windows.append((start_s, end_s))
        start_s += duration_s
    return windows or [(0.0, max_duration_s)]


def _format_segment_start_minute(seconds: float) -> str:
    minutes = max(0.0, float(seconds)) / 60.0
    if abs(minutes - round(minutes)) < 1e-9:
        return str(int(round(minutes)))
    return f"{minutes:g}"


def _write_segmented_pooled_html(
    *,
    output_path: Path,
    segments: list[dict[str, Any]],
    include_plotlyjs: bool | str,
    title: str,
) -> None:
    title_html = html_lib.escape(str(title))
    plotlyjs = _plotlyjs_include_html(include_plotlyjs)
    config_json = _safe_json(POOLED_HTML_CONFIG)
    plot_id = "pooled-place-cell-segment-plot"
    wheel_script = POOLED_HTML_WHEEL_PAN_SCRIPT.replace("{plot_id}", plot_id)
    segment_meta: list[dict[str, Any]] = []
    segment_data_tags: list[str] = []
    for idx, seg in enumerate(segments):
        data_id = f"segment-data-{idx}"
        segment_meta.append(
            {
                "label": seg["label"],
                "start_label": seg["start_label"],
                "start_s": seg["start_s"],
                "end_s": seg["end_s"],
                "data_id": data_id,
            }
        )
        segment_data_tags.append(
            f'<script id="{data_id}" type="application/json">{_safe_json(seg["figure"])}</script>'
        )
    segment_meta_json = _safe_json(segment_meta)

    if len(segments) <= 15:
        marks = "".join(
            f'<option value="{idx}" label="{html_lib.escape(str(seg["start_label"]))}"></option>'
            for idx, seg in enumerate(segments)
        )
    else:
        marks = (
            f'<option value="0" label="{html_lib.escape(str(segments[0]["start_label"]))}"></option>'
            f'<option value="{len(segments) - 1}" label="{html_lib.escape(str(segments[-1]["start_label"]))}"></option>'
        )

    html_text = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title_html}</title>
  {plotlyjs}
  <style>
    body {{
      margin: 0;
      font-family: Arial, sans-serif;
      background: #fff;
      color: #111;
    }}
    #segment-controls {{
      position: sticky;
      top: 0;
      z-index: 10;
      display: grid;
      grid-template-columns: auto minmax(220px, 1fr) auto;
      gap: 10px;
      align-items: center;
      padding: 8px 12px;
      border-bottom: 1px solid #ddd;
      background: rgba(255, 255, 255, 0.96);
      font-size: 13px;
    }}
    #segment-slider {{
      width: 100%;
    }}
    #segment-label {{
      white-space: nowrap;
      color: #444;
      font-size: 12px;
    }}
  </style>
</head>
<body>
  <div id="segment-controls">
    <div><strong>Start minute</strong></div>
    <input id="segment-slider" type="range" min="0" max="{max(0, len(segments) - 1)}" step="1" value="0" list="segment-marks">
    <div id="segment-label"></div>
    <datalist id="segment-marks">{marks}</datalist>
  </div>
  <div id="{plot_id}"></div>
  <div id="render-status" style="padding: 8px 12px; color: #555; font-size: 12px;">Loading first segment...</div>
  {"".join(segment_data_tags)}
  <script>
    const SEGMENTS = {segment_meta_json};
    const CONFIG = {config_json};
    const PLOT_ID = "{plot_id}";
    const slider = document.getElementById("segment-slider");
    const label = document.getElementById("segment-label");
    const status = document.getElementById("render-status");
    const cache = new Map();
    let xRangeGuardInstalled = false;
    let xRangeNormalizing = false;

    function setStatus(message, isError) {{
      if (!status) return;
      status.textContent = message || "";
      status.style.color = isError ? "#a00000" : "#555";
      status.style.display = message ? "block" : "none";
    }}

    function normalizeIncreasingXRange() {{
      const graph = document.getElementById(PLOT_ID);
      if (!graph || !window.Plotly || !graph._fullLayout || xRangeNormalizing) {{
        return Promise.resolve();
      }}
      const axis = graph._fullLayout.xaxis || {{}};
      const range = axis.range || axis._range;
      if (!range || range.length < 2) {{
        return Promise.resolve();
      }}
      const x0 = Number(range[0]);
      const x1 = Number(range[1]);
      if (!Number.isFinite(x0) || !Number.isFinite(x1) || x0 <= x1) {{
        return Promise.resolve();
      }}
      xRangeNormalizing = true;
      return Plotly.relayout(graph, {{
        "xaxis.range[0]": x1,
        "xaxis.range[1]": x0,
      }}).finally(function () {{
        xRangeNormalizing = false;
      }});
    }}

    function installIncreasingXRangeGuard() {{
      const graph = document.getElementById(PLOT_ID);
      if (!graph || xRangeGuardInstalled || typeof graph.on !== "function") {{
        return;
      }}
      xRangeGuardInstalled = true;
      graph.on("plotly_relayout", function () {{
        if (xRangeNormalizing) {{
          return;
        }}
        window.setTimeout(normalizeIncreasingXRange, 0);
      }});
    }}

    function readFigure(segment) {{
      if (cache.has(segment.data_id)) {{
        return cache.get(segment.data_id);
      }}
      const node = document.getElementById(segment.data_id);
      if (!node) {{
        throw new Error("Missing embedded segment data: " + segment.data_id);
      }}
      const figure = JSON.parse(node.textContent);
      cache.set(segment.data_id, figure);
      return figure;
    }}

    function renderSegment(index) {{
      const idx = Math.max(0, Math.min(SEGMENTS.length - 1, Number(index) || 0));
      const segment = SEGMENTS[idx];
      slider.value = String(idx);
      label.textContent = `Segment ${{idx + 1}}/${{SEGMENTS.length}} | ${{segment.label}}`;
      if (!window.Plotly) {{
        setStatus("Plotly did not load. If this file was exported with include_plotlyjs='cdn', connect to the internet or regenerate with include_plotlyjs=True.", true);
        return Promise.resolve();
      }}
      setStatus("Rendering segment " + (idx + 1) + "/" + SEGMENTS.length + "...", false);
      let figure;
      try {{
        figure = readFigure(segment) || {{}};
      }} catch (err) {{
        setStatus("Could not read segment data: " + err.message, true);
        return Promise.resolve();
      }}
      return Plotly.react(PLOT_ID, figure.data || [], figure.layout || {{}}, CONFIG)
        .then(function () {{
          installIncreasingXRangeGuard();
          return normalizeIncreasingXRange();
        }})
        .then(function () {{
          setStatus("", false);
        }})
        .catch(function (err) {{
          setStatus("Plotly render failed: " + err.message, true);
        }});
    }}

    slider.addEventListener("input", function () {{
      renderSegment(slider.value);
    }});

    renderSegment(0);
  </script>
  <script>{wheel_script}</script>
</body>
</html>
"""
    output_path.write_text(html_text, encoding="utf-8")


def _robust_center_scale(*arrays: np.ndarray) -> tuple[float, float]:
    finite_chunks = []
    for arr in arrays:
        vals = np.asarray(arr, dtype=float).reshape(-1)
        vals = vals[np.isfinite(vals)]
        if vals.size > 0:
            finite_chunks.append(vals)
    if not finite_chunks:
        return 0.0, 1.0
    vals = np.concatenate(finite_chunks)
    center = float(np.nanmedian(vals))
    p1, p99 = np.nanpercentile(vals, [1, 99])
    scale = float(p99 - p1)
    if (not np.isfinite(scale)) or scale <= 1e-12:
        scale = float(np.nanstd(vals) * 6.0)
    if (not np.isfinite(scale)) or scale <= 1e-12:
        scale = 1.0
    return center, scale


def _scaled(values: np.ndarray, center: float, scale: float, offset: float) -> np.ndarray:
    return ((np.asarray(values, dtype=float) - center) / scale) * TRACE_SCALE + float(offset)


def _minmax_downsample_trace(
    values: np.ndarray,
    frame_rate: float,
    max_points: int | None,
) -> tuple[np.ndarray | None, np.ndarray, np.ndarray | None]:
    y = np.asarray(values, dtype=float).reshape(-1)
    if max_points is None:
        return None, y, None
    max_points = int(max_points)
    if max_points <= 0 or y.size <= max_points:
        return None, y, None

    n_bins = max(1, max_points // 2)
    edges = np.linspace(0, y.size, n_bins + 1, dtype=int)
    x_out: list[float] = []
    y_out: list[float] = []
    idx_out: list[int] = []

    for start, end in zip(edges[:-1], edges[1:]):
        if end <= start:
            continue
        seg = y[start:end]
        finite = np.flatnonzero(np.isfinite(seg))
        if finite.size == 0:
            mid = start + (end - start) // 2
            x_out.append(mid / float(frame_rate))
            y_out.append(np.nan)
            idx_out.append(mid)
            continue

        finite_vals = seg[finite]
        idx_min = int(finite[int(np.argmin(finite_vals))])
        idx_max = int(finite[int(np.argmax(finite_vals))])
        for local_idx in sorted({idx_min, idx_max}):
            frame_idx = start + local_idx
            x_out.append(frame_idx / float(frame_rate))
            y_out.append(float(y[frame_idx]))
            idx_out.append(frame_idx)

    return np.asarray(x_out, dtype=float), np.asarray(y_out, dtype=float), np.asarray(idx_out, dtype=int)


def _finite_min_max(*arrays: np.ndarray) -> tuple[float, float]:
    lo = np.inf
    hi = -np.inf
    for arr in arrays:
        vals = np.asarray(arr, dtype=float).reshape(-1)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        lo = min(lo, float(np.min(vals)))
        hi = max(hi, float(np.max(vals)))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return -TRACE_HALF_HEIGHT, TRACE_HALF_HEIGHT
    return lo, hi


def _row_geometry(trace_y: np.ndarray, vm_y: np.ndarray) -> tuple[float, float, float, float]:
    trace_min, trace_max = _finite_min_max(trace_y, vm_y)
    raster_y0 = trace_max + RASTER_PAD
    raster_y1 = raster_y0 + RASTER_HEIGHT
    row_bottom = trace_min - TRACE_PAD
    row_top = raster_y1 + TRACE_PAD
    return row_bottom, row_top, raster_y0, raster_y1


def _stack_offsets(bounds: list[tuple[float, float]]) -> tuple[list[float], float, float]:
    if not bounds:
        return [], -1.0, 1.0

    offsets = [0.0 for _ in bounds]
    cursor = 0.0
    for idx in range(len(bounds) - 1, -1, -1):
        row_bottom, row_top = bounds[idx]
        offsets[idx] = cursor - row_bottom
        cursor += (row_top - row_bottom) + ROW_GAP

    bottoms = [offset + row_bottom for offset, (row_bottom, _) in zip(offsets, bounds)]
    tops = [offset + row_top for offset, (_, row_top) in zip(offsets, bounds)]
    return offsets, min(bottoms) - PLOT_PAD, max(tops) + PLOT_PAD


def _hover_customdata(start: int, end: int, frame_rate: float, snr_values: np.ndarray) -> np.ndarray:
    frames = np.arange(int(start), int(end), dtype=float)
    absolute_time_s = frames / float(frame_rate)
    snr_slice = np.asarray(snr_values[int(start) : int(end)], dtype=float)
    return np.column_stack((absolute_time_s, snr_slice))


def _snr_hover_values(start: int, end: int, snr_values: np.ndarray) -> np.ndarray:
    return np.asarray(snr_values[int(start) : int(end)], dtype=float)


def _cell_category(animal: AnimalData, cell_idx: int) -> str:
    categories = getattr(animal, "plc_categories", [])
    if 0 <= int(cell_idx) < len(categories):
        return str(categories[int(cell_idx)])
    return "Unknown"


def _pooled_place_cell_number_lookup(
    animals: list[AnimalData],
    category_filter: set[str] | None = None,
    *,
    cell_selection: str = "place",
) -> dict[tuple[str, int], str]:
    category_filter_set = {str(category) for category in category_filter} if category_filter is not None else None
    lookup: dict[tuple[str, int], str] = {}
    for animal in animals:
        for cell_idx in range(animal.n_cells):
            if not _cell_matches_selection(animal, cell_idx, cell_selection):
                continue
            category = _cell_category(animal, cell_idx)
            if category_filter_set is not None and category not in category_filter_set:
                continue
            lookup[(animal.animal_id, int(cell_idx))] = str(len(lookup) + 1)
    return lookup


def _cell_matches_selection(animal: AnimalData, cell_idx: int, cell_selection: str) -> bool:
    normalized = str(cell_selection).strip().lower().replace("-", "_").replace(" ", "_")
    place_mask = getattr(animal, "place_cell_mask", [])
    is_place_cell = bool(cell_idx < len(place_mask) and place_mask[cell_idx])
    if normalized in {"place", "place_cell", "place_cells", "plc", "plcs"}:
        return is_place_cell
    if normalized in {"non_place", "non_place_cell", "non_place_cells", "non_plc", "non_plcs", "nonplace"}:
        return not is_place_cell
    if normalized in {"all", "all_cells", "any"}:
        return True
    raise ValueError(f"Unknown cell_selection: {cell_selection!r}")


def _cell_selection_title(cell_selection: str) -> str:
    normalized = str(cell_selection).strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {"place", "place_cell", "place_cells", "plc", "plcs"}:
        return "place cells"
    if normalized in {"non_place", "non_place_cell", "non_place_cells", "non_plc", "non_plcs", "nonplace"}:
        return "non-place cells"
    if normalized in {"all", "all_cells", "any"}:
        return "cells"
    raise ValueError(f"Unknown cell_selection: {cell_selection!r}")


def _cell_label(animal: AnimalData, cell_idx: int) -> str:
    category = _cell_category(animal, cell_idx)
    if category in {"CS+ PLC", "CS- PLC"}:
        return f"Cell {int(cell_idx) + 1} ({category})"
    return f"Cell {int(cell_idx) + 1}"


def _short_animal_label(animal_id: str) -> str:
    parts = str(animal_id).split("_")
    for idx, part in enumerate(parts):
        if part.startswith("pAce"):
            if idx + 1 < len(parts) and not parts[idx + 1].isdigit():
                return "_".join(parts[idx : idx + 2])
            return part
    return str(animal_id)


def _spike_tick_trace(
    *,
    spikes: np.ndarray,
    frame_rate: float,
    y0: float,
    y1: float,
    name: str,
    color: str,
    visible: bool = True,
) -> go.Scattergl | None:
    spks = np.asarray(spikes, dtype=int).reshape(-1)
    if spks.size == 0:
        return None
    x = np.empty(spks.size * 3, dtype=float)
    y = np.empty(spks.size * 3, dtype=float)
    t = spks.astype(float) / float(frame_rate)
    x[0::3] = t
    x[1::3] = t
    x[2::3] = np.nan
    y[0::3] = y0
    y[1::3] = y1
    y[2::3] = np.nan
    return go.Scattergl(
        x=x,
        y=y,
        mode="lines",
        line={"color": color, "width": 1},
        name=name,
        hoverinfo="skip",
        visible=visible,
        showlegend=False,
    )


def _shape(x0: float, x1: float, y0: float, y1: float, color: str, layer: str = "below") -> dict[str, Any]:
    return {
        "type": "rect",
        "xref": "x",
        "yref": "y",
        "x0": float(x0),
        "x1": float(x1),
        "y0": float(y0),
        "y1": float(y1),
        "fillcolor": color,
        "line": {"width": 0},
        "layer": layer,
    }


def _add_interval_shapes(
    shapes: list[dict[str, Any]],
    intervals: list[tuple[int, int]],
    *,
    frame_rate: float,
    y0: float,
    y1: float,
    color: str,
    row_start: int = 0,
    row_end: int | None = None,
    absolute_time: bool = False,
) -> None:
    for start, end in intervals:
        s = int(start)
        e = int(end)
        if row_end is not None:
            s = max(s, int(row_start))
            e = min(e, int(row_end))
            if e <= s:
                continue
            if absolute_time:
                x0 = s / float(frame_rate)
                x1 = e / float(frame_rate)
            else:
                x0 = (s - int(row_start)) / float(frame_rate)
                x1 = (e - int(row_start)) / float(frame_rate)
        else:
            x0 = s / float(frame_rate)
            x1 = e / float(frame_rate)
        shapes.append(_shape(x0, x1, y0, y1, color))


def _extend_interval_polygons(
    x_values: list[float | None],
    y_values: list[float | None],
    intervals: list[tuple[int, int]],
    *,
    frame_rate: float,
    y0: float,
    y1: float,
    row_start: int = 0,
    row_end: int | None = None,
    absolute_time: bool = False,
) -> None:
    for start, end in intervals:
        s = int(start)
        e = int(end)
        if row_end is not None:
            s = max(s, int(row_start))
            e = min(e, int(row_end))
            if e <= s:
                continue
            if absolute_time:
                x0 = s / float(frame_rate)
                x1 = e / float(frame_rate)
            else:
                x0 = (s - int(row_start)) / float(frame_rate)
                x1 = (e - int(row_start)) / float(frame_rate)
        else:
            x0 = s / float(frame_rate)
            x1 = e / float(frame_rate)
        x_values.extend([x0, x0, x1, x1, x0, None])
        y_values.extend([y0, y1, y1, y0, y0, None])


def _add_interval_layer_trace(
    fig: go.Figure,
    *,
    x_values: list[float | None],
    y_values: list[float | None],
    name: str,
    color: str,
    visible: bool,
    showlegend: bool = True,
) -> None:
    if not x_values:
        return
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=y_values,
            mode="lines",
            fill="toself",
            fillcolor=color,
            line={"color": color, "width": 0},
            name=name,
            visible=True if visible else "legendonly",
            hoverinfo="skip",
            legendgroup=name,
            showlegend=bool(showlegend),
        )
    )


def _add_session_shapes(
    shapes: list[dict[str, Any]],
    animal: AnimalData,
    y0: float,
    y1: float,
    *,
    row_start: int = 0,
    row_end: int | None = None,
    absolute_time: bool = False,
) -> None:
    for frame in animal.session_start_frames[1:]:
        frame = int(frame)
        if row_end is not None:
            if not int(row_start) <= frame < int(row_end):
                continue
            if absolute_time:
                x = frame / float(animal.frame_rate)
            else:
                x = (frame - int(row_start)) / float(animal.frame_rate)
        else:
            if not 0 <= frame < animal.n_frames:
                continue
            x = frame / float(animal.frame_rate)
        shapes.append(
            {
                "type": "line",
                "xref": "x",
                "yref": "y",
                "x0": x,
                "x1": x,
                "y0": y0,
                "y1": y1,
                "line": {"color": SESSION_LINE, "width": 1, "dash": "dot"},
                "layer": "above",
            }
        )


def build_stacked_figure(
    animal: AnimalData,
    snr_threshold: float,
    min_good_minutes: float,
    *,
    cell_indices: list[int] | np.ndarray | None = None,
    time_window_s: tuple[float, float] | None = None,
    row_height_px: float | int | None = DEFAULT_STACKED_ROW_HEIGHT_PX,
) -> tuple[go.Figure, list[dict[str, Any]]]:
    fig = go.Figure()
    row_height = _clean_row_height_px(row_height_px)
    shapes: list[dict[str, Any]] = []
    stats_rows: list[dict[str, Any]] = []
    tickvals: list[float] = []
    ticktext: list[str] = []
    cell_jump_rows: list[dict[str, Any]] = []
    cell_click_curves: dict[str, dict[str, Any]] = {}
    cell_rows: list[dict[str, Any]] = []
    bounds: list[tuple[float, float]] = []

    if cell_indices is None:
        display_cells = list(range(animal.n_cells))
    else:
        display_cells = [int(i) for i in cell_indices if 0 <= int(i) < animal.n_cells]

    if time_window_s is None:
        frame_start = 0
        frame_end = animal.n_frames
    else:
        start_s, end_s = float(time_window_s[0]), float(time_window_s[1])
        frame_start = max(0, int(np.floor(start_s * float(animal.frame_rate))))
        frame_end = min(animal.n_frames, int(np.ceil(end_s * float(animal.frame_rate))))

    if not display_cells:
        fig.update_layout(
            title=f"{animal.animal_id}: no cells selected",
            height=520,
            margin={"l": 80, "r": 24, "t": 48, "b": 42},
            plot_bgcolor="white",
            paper_bgcolor="white",
            xaxis={"visible": False},
            yaxis={"visible": False},
            annotations=[
                {
                    "text": "No cells match the current filter.",
                    "xref": "paper",
                    "yref": "paper",
                    "x": 0.5,
                    "y": 0.5,
                    "showarrow": False,
                    "font": {"size": 14},
                }
            ],
        )
        return fig, []

    if frame_end <= frame_start:
        fig.update_layout(
            title=f"{animal.animal_id}: empty time window",
            height=520,
            margin={"l": 80, "r": 24, "t": 48, "b": 42},
            plot_bgcolor="white",
            paper_bgcolor="white",
            xaxis={"visible": False},
            yaxis={"visible": False},
            annotations=[
                {
                    "text": "No frames match the current time window.",
                    "xref": "paper",
                    "yref": "paper",
                    "x": 0.5,
                    "y": 0.5,
                    "showarrow": False,
                    "font": {"size": 14},
                }
            ],
        )
        return fig, []

    for cell_idx in display_cells:
        trace = get_cell_vector(animal.traces, cell_idx, animal.n_frames)
        vm = get_cell_vector(animal.vms, cell_idx, animal.n_frames)
        center, scale = _robust_center_scale(trace, vm)
        trace_y = _scaled(trace[frame_start:frame_end], center, scale, 0.0)
        vm_y = _scaled(vm[frame_start:frame_end], center, scale, 0.0)
        row_bottom, row_top, raster_y0, raster_y1 = _row_geometry(trace_y, vm_y)
        bounds.append((row_bottom, row_top))

        bad_mask, stats = bad_mask_for_cell(animal, cell_idx, snr_threshold, min_good_minutes)
        snr_values = compute_snr_values(animal, cell_idx)
        stats_rows.append(stats)
        cell_rows.append(
            {
                "cell_idx": cell_idx,
                "frame_start": frame_start,
                "frame_end": frame_end,
                "trace_y": trace_y,
                "vm_y": vm_y,
                "snr_values": snr_values,
                "bad_intervals": mask_to_intervals(bad_mask),
                "cb_intervals": dict_intervals(animal.complex_bursts[cell_idx], animal.n_frames),
                "plateau_intervals": dict_intervals(animal.plateaus[cell_idx], animal.n_frames),
                "row_bottom": row_bottom,
                "row_top": row_top,
                "raster_y0": raster_y0,
                "raster_y1": raster_y1,
            }
        )

    offsets, plot_y0, plot_y1 = _stack_offsets(bounds)

    for item, offset in zip(cell_rows, offsets):
        cell_idx = int(item["cell_idx"])
        item_start = int(item["frame_start"])
        item_end = int(item["frame_end"])
        tickvals.append(offset)
        label = _cell_label(animal, cell_idx)
        ticktext.append(label)
        cell_jump_rows.append({"label": label, "animal_id": animal.animal_id, "cell_idx": cell_idx})
        y0 = offset + float(item["row_bottom"])
        y1 = offset + float(item["row_top"])
        snr_hover = _snr_hover_values(item_start, item_end, item["snr_values"])
        _add_interval_shapes(
            shapes,
            item["bad_intervals"],
            frame_rate=animal.frame_rate,
            y0=y0,
            y1=y1,
            color=BAD_FILL,
            row_start=item_start,
            row_end=item_end,
            absolute_time=True,
        )
        _add_interval_shapes(
            shapes,
            item["cb_intervals"],
            frame_rate=animal.frame_rate,
            y0=y0,
            y1=y1,
            color=CB_FILL,
            row_start=item_start,
            row_end=item_end,
            absolute_time=True,
        )
        _add_interval_shapes(
            shapes,
            item["plateau_intervals"],
            frame_rate=animal.frame_rate,
            y0=y0,
            y1=y1,
            color=PLATEAU_FILL,
            row_start=item_start,
            row_end=item_end,
            absolute_time=True,
        )

        target = {"animal_id": animal.animal_id, "cell_idx": cell_idx}
        curve_idx = len(fig.data)
        fig.add_trace(
            go.Scattergl(
                y=item["trace_y"] + offset,
                x0=item_start / float(animal.frame_rate),
                dx=1.0 / float(animal.frame_rate),
                mode="lines",
                line={"color": TRACE_COLOR, "width": 0.7},
                name=f"Cell {cell_idx + 1} trace",
                hovertemplate=(
                    f"Cell {cell_idx + 1} (idx {cell_idx})"
                    f"<br>{_cell_category(animal, cell_idx)}"
                    "<br>t=%{x:.3f} s"
                    "<br>SNR=%{customdata:.3f}<extra></extra>"
                ),
                customdata=snr_hover,
                showlegend=False,
            )
        )
        cell_click_curves[str(curve_idx)] = target
        curve_idx = len(fig.data)
        fig.add_trace(
            go.Scattergl(
                y=item["vm_y"] + offset,
                x0=item_start / float(animal.frame_rate),
                dx=1.0 / float(animal.frame_rate),
                mode="lines",
                line={"color": VM_COLOR, "width": 0.6},
                name=f"Cell {cell_idx + 1} Vm",
                hovertemplate=(
                    f"Cell {cell_idx + 1} Vm (idx {cell_idx})"
                    f"<br>{_cell_category(animal, cell_idx)}"
                    "<br>t=%{x:.3f} s<extra></extra>"
                ),
                showlegend=False,
            )
        )
        cell_click_curves[str(curve_idx)] = target

        ss_trace = _spike_tick_trace(
            spikes=animal.refined_ss[cell_idx][
                (animal.refined_ss[cell_idx] >= item_start) & (animal.refined_ss[cell_idx] < item_end)
            ],
            frame_rate=animal.frame_rate,
            y0=offset + float(item["raster_y0"]),
            y1=offset + float(item["raster_y1"]),
            name=f"Cell {cell_idx + 1} SS",
            color=SS_COLOR,
        )
        cs_trace = _spike_tick_trace(
            spikes=animal.all_cs_spikes[cell_idx][
                (animal.all_cs_spikes[cell_idx] >= item_start) & (animal.all_cs_spikes[cell_idx] < item_end)
            ],
            frame_rate=animal.frame_rate,
            y0=offset + float(item["raster_y0"]),
            y1=offset + float(item["raster_y1"]),
            name=f"Cell {cell_idx + 1} CS",
            color=CS_COLOR,
        )
        if ss_trace is not None:
            cell_click_curves[str(len(fig.data))] = target
            fig.add_trace(ss_trace)
        if cs_trace is not None:
            cell_click_curves[str(len(fig.data))] = target
            fig.add_trace(cs_trace)

    _add_session_shapes(shapes, animal, plot_y0, plot_y1, row_start=frame_start, row_end=frame_end, absolute_time=True)
    x_start_s = frame_start / float(animal.frame_rate)
    x_end_s = frame_end / float(animal.frame_rate)
    if time_window_s is None:
        title = f"{animal.animal_id}: {len(display_cells)} cells stacked"
        xaxis_title = "Absolute time (s)"
    else:
        title = (
            f"{animal.animal_id}: {len(display_cells)} cells stacked, "
            f"{frame_start / float(animal.frame_rate):g}-{frame_end / float(animal.frame_rate):g} s"
        )
        xaxis_title = "Absolute time (s)"
    fig.update_layout(
        title=title,
        height=max(560, row_height * max(1, len(display_cells))),
        margin={"l": 80, "r": 24, "t": 48, "b": 42},
        plot_bgcolor="white",
        paper_bgcolor="white",
        shapes=shapes,
        xaxis={
            "title": xaxis_title,
            "range": [x_start_s, x_end_s],
            "minallowed": x_start_s,
            "maxallowed": x_end_s,
            "showgrid": True,
            "gridcolor": "#eeeeee",
        },
        yaxis={
            "title": "Neuron",
            "tickmode": "array",
            "tickvals": tickvals,
            "ticktext": ticktext,
            "range": [plot_y0, plot_y1],
            "minallowed": plot_y0,
            "maxallowed": plot_y1,
            "fixedrange": True,
            "showgrid": False,
            "zeroline": False,
        },
        dragmode="zoom",
        hovermode="closest",
        meta={"cell_jump_rows": cell_jump_rows, "cell_click_curves": cell_click_curves},
        uirevision=f"{animal.animal_id}-stacked-{frame_start}-{frame_end}",
    )
    return fig, stats_rows


def build_pooled_place_cells_figure(
    animals: list[AnimalData],
    snr_threshold: float,
    min_good_minutes: float,
    *,
    trace_max_points: int | None = None,
    time_window_s: tuple[float, float] | None = None,
    category_filter: set[str] | None = None,
    optional_layers_visible: bool = True,
    bad_epochs_visible: bool | None = None,
    row_height_px: float | int | None = DEFAULT_STACKED_ROW_HEIGHT_PX,
    show_legend: bool = True,
    y_label_mode: str = "identity",
    y_label_lookup: dict[tuple[str, int], str] | None = None,
    cell_selection: str = "place",
    cell_key_filter: set[tuple[str, int]] | None = None,
) -> tuple[go.Figure, list[dict[str, Any]]]:
    fig = go.Figure()
    row_height = _clean_row_height_px(row_height_px)
    show_legend = bool(show_legend)
    shapes: list[dict[str, Any]] = []
    interval_layers = {
        "Bad SNR": {"x": [], "y": [], "color": BAD_FILL},
        "Complex bursts": {"x": [], "y": [], "color": CB_FILL},
        "Plateaus": {"x": [], "y": [], "color": PLATEAU_FILL},
    }
    stats_rows: list[dict[str, Any]] = []
    tickvals: list[float] = []
    ticktext: list[str] = []
    cell_jump_rows: list[dict[str, Any]] = []
    cell_click_curves: dict[str, dict[str, Any]] = {}
    cell_rows: list[dict[str, Any]] = []
    bounds: list[tuple[float, float]] = []
    category_filter_set = {str(category) for category in category_filter} if category_filter is not None else None
    filter_label = ", ".join(sorted(category_filter_set)) if category_filter_set else None
    selection_title = _cell_selection_title(cell_selection)
    normalized_cell_key_filter = (
        {(str(animal_id), int(cell_idx)) for animal_id, cell_idx in cell_key_filter}
        if cell_key_filter is not None
        else None
    )

    for animal in animals:
        if time_window_s is None:
            frame_start = 0
            frame_end = animal.n_frames
        else:
            start_s, end_s = float(time_window_s[0]), float(time_window_s[1])
            frame_start = max(0, int(np.floor(start_s * float(animal.frame_rate))))
            frame_end = min(animal.n_frames, int(np.ceil(end_s * float(animal.frame_rate))))
        if frame_end <= frame_start:
            continue

        for cell_idx in range(animal.n_cells):
            if not _cell_matches_selection(animal, cell_idx, cell_selection):
                continue
            cell_key = (animal.animal_id, int(cell_idx))
            if normalized_cell_key_filter is not None and cell_key not in normalized_cell_key_filter:
                continue
            category = _cell_category(animal, cell_idx)
            if category_filter_set is not None and category not in category_filter_set:
                continue
            trace = get_cell_vector(animal.traces, cell_idx, animal.n_frames)
            vm = get_cell_vector(animal.vms, cell_idx, animal.n_frames)
            center, scale = _robust_center_scale(trace, vm)
            trace_y = _scaled(trace[frame_start:frame_end], center, scale, 0.0)
            vm_y = _scaled(vm[frame_start:frame_end], center, scale, 0.0)
            row_bottom, row_top, raster_y0, raster_y1 = _row_geometry(trace_y, vm_y)
            bounds.append((row_bottom, row_top))

            bad_mask, stats = bad_mask_for_cell(animal, cell_idx, snr_threshold, min_good_minutes)
            snr_values = compute_snr_values(animal, cell_idx)
            stats = dict(stats)
            stats["animal_id"] = animal.animal_id
            stats["category"] = category
            stats["cell_idx"] = int(cell_idx)
            stats_rows.append(stats)
            cell_rows.append(
                {
                    "animal": animal,
                    "cell_idx": cell_idx,
                    "category": category,
                    "frame_start": frame_start,
                    "frame_end": frame_end,
                    "trace_y": trace_y,
                    "vm_y": vm_y,
                    "snr_values": snr_values,
                    "bad_intervals": mask_to_intervals(bad_mask),
                    "cb_intervals": dict_intervals(animal.complex_bursts[cell_idx], animal.n_frames),
                    "plateau_intervals": dict_intervals(animal.plateaus[cell_idx], animal.n_frames),
                    "row_bottom": row_bottom,
                    "row_top": row_top,
                    "raster_y0": raster_y0,
                    "raster_y1": raster_y1,
                }
            )

    if not cell_rows:
        empty_label = f"{filter_label} " if filter_label else ""
        fig.update_layout(
            title=f"All animals: no {empty_label}{selection_title} selected",
            height=520,
            margin={"l": 130, "r": 24, "t": 48, "b": 42},
            plot_bgcolor="white",
            paper_bgcolor="white",
            xaxis={"visible": False},
            yaxis={"visible": False},
            annotations=[
                {
                    "text": f"No {empty_label}{selection_title} found across loaded animals.",
                    "xref": "paper",
                    "yref": "paper",
                    "x": 0.5,
                    "y": 0.5,
                    "showarrow": False,
                    "font": {"size": 14},
                }
            ],
        )
        return fig, []

    offsets, plot_y0, plot_y1 = _stack_offsets(bounds)
    x_axis_start_s = np.inf
    x_axis_end_s = 0.0
    vm_legend_added = False

    for display_idx, (item, offset) in enumerate(zip(cell_rows, offsets), start=1):
        animal = item["animal"]
        cell_idx = int(item["cell_idx"])
        frame_start = int(item["frame_start"])
        frame_end = int(item["frame_end"])
        category = str(item["category"])
        animal_label = _short_animal_label(animal.animal_id)
        x_start_s = frame_start / float(animal.frame_rate)
        x_end_s = frame_end / float(animal.frame_rate)
        x_axis_start_s = min(x_axis_start_s, x_start_s)
        x_axis_end_s = max(x_axis_end_s, x_end_s)
        row_y0 = offset + float(item["row_bottom"])
        row_y1 = offset + float(item["row_top"])

        tickvals.append(offset)
        number_label_mode = str(y_label_mode) == "number"
        if str(y_label_mode) == "number":
            label = str((y_label_lookup or {}).get((animal.animal_id, cell_idx), str(display_idx)))
        else:
            label = f"{animal_label}<br>Cell {cell_idx + 1} ({category})"
        ticktext.append(label)
        if number_label_mode:
            cell_display_label = f"Cell {label}"
            cell_jump_rows.append({"label": label})
            target = {"label": label}
        else:
            cell_display_label = f"{animal_label} Cell {cell_idx + 1}"
            cell_jump_rows.append({"label": label, "animal_id": animal.animal_id, "cell_idx": cell_idx})
            target = {"animal_id": animal.animal_id, "cell_idx": cell_idx}

        _extend_interval_polygons(
            interval_layers["Bad SNR"]["x"],
            interval_layers["Bad SNR"]["y"],
            item["bad_intervals"],
            frame_rate=animal.frame_rate,
            y0=row_y0,
            y1=row_y1,
            row_start=frame_start,
            row_end=frame_end,
            absolute_time=True,
        )
        _extend_interval_polygons(
            interval_layers["Complex bursts"]["x"],
            interval_layers["Complex bursts"]["y"],
            item["cb_intervals"],
            frame_rate=animal.frame_rate,
            y0=row_y0,
            y1=row_y1,
            row_start=frame_start,
            row_end=frame_end,
            absolute_time=True,
        )
        _extend_interval_polygons(
            interval_layers["Plateaus"]["x"],
            interval_layers["Plateaus"]["y"],
            item["plateau_intervals"],
            frame_rate=animal.frame_rate,
            y0=row_y0,
            y1=row_y1,
            row_start=frame_start,
            row_end=frame_end,
            absolute_time=True,
        )
        for frame in animal.session_start_frames[1:]:
            if frame_start <= int(frame) < frame_end:
                x = int(frame) / float(animal.frame_rate)
                shapes.append(
                    {
                        "type": "line",
                        "xref": "x",
                        "yref": "y",
                        "x0": x,
                        "x1": x,
                        "y0": row_y0,
                        "y1": row_y1,
                        "line": {"color": SESSION_LINE, "width": 1, "dash": "dot"},
                        "layer": "above",
                    }
                )

        snr_hover = _snr_hover_values(frame_start, frame_end, item["snr_values"])
        trace_x, trace_y, trace_idx = _minmax_downsample_trace(item["trace_y"] + offset, animal.frame_rate, trace_max_points)
        trace_hover = snr_hover if trace_idx is None else snr_hover[trace_idx]
        trace_kwargs: dict[str, Any] = {"y": trace_y}
        if trace_x is None:
            trace_kwargs.update({"x0": x_start_s, "dx": 1.0 / float(animal.frame_rate)})
        else:
            trace_kwargs["x"] = trace_x + x_start_s
        curve_idx = len(fig.data)
        fig.add_trace(
            go.Scattergl(
                **trace_kwargs,
                mode="lines",
                line={"color": TRACE_COLOR, "width": 0.7},
                name=f"{cell_display_label} trace",
                hovertemplate=(
                    f"{cell_display_label}"
                    f"<br>{category}"
                    "<br>t=%{x:.3f} s"
                    "<br>SNR=%{customdata:.3f}<extra></extra>"
                ),
                customdata=trace_hover,
                showlegend=False,
            )
        )
        cell_click_curves[str(curve_idx)] = target

        vm_x, vm_y, _ = _minmax_downsample_trace(item["vm_y"] + offset, animal.frame_rate, trace_max_points)
        vm_kwargs: dict[str, Any] = {"y": vm_y}
        if vm_x is None:
            vm_kwargs.update({"x0": x_start_s, "dx": 1.0 / float(animal.frame_rate)})
        else:
            vm_kwargs["x"] = vm_x + x_start_s
        curve_idx = len(fig.data)
        fig.add_trace(
            go.Scattergl(
                **vm_kwargs,
                mode="lines",
                line={"color": VM_COLOR, "width": 0.6},
                name=f"{cell_display_label} Vm",
                visible=True if (not show_legend or optional_layers_visible) else "legendonly",
                hovertemplate=(
                    f"{cell_display_label} Vm"
                    f"<br>{category}"
                    "<br>t=%{x:.3f} s<extra></extra>"
                ),
                legendgroup="Vm",
                showlegend=show_legend and not vm_legend_added,
            )
        )
        cell_click_curves[str(curve_idx)] = target
        vm_legend_added = True

        ss_trace = _spike_tick_trace_row(
            spikes=animal.refined_ss[cell_idx],
            frame_rate=animal.frame_rate,
            start=frame_start,
            end=frame_end,
            y0=offset + float(item["raster_y0"]),
            y1=offset + float(item["raster_y1"]),
            name=f"{cell_display_label} SS",
            color=SS_COLOR,
            absolute_time=True,
        )
        cs_trace = _spike_tick_trace_row(
            spikes=animal.all_cs_spikes[cell_idx],
            frame_rate=animal.frame_rate,
            start=frame_start,
            end=frame_end,
            y0=offset + float(item["raster_y0"]),
            y1=offset + float(item["raster_y1"]),
            name=f"{cell_display_label} CS",
            color=CS_COLOR,
            absolute_time=True,
        )
        if ss_trace is not None:
            cell_click_curves[str(len(fig.data))] = target
            fig.add_trace(ss_trace)
        if cs_trace is not None:
            cell_click_curves[str(len(fig.data))] = target
            fig.add_trace(cs_trace)

    bad_layer_visible = bool(optional_layers_visible) if bad_epochs_visible is None else bool(bad_epochs_visible)
    if not show_legend:
        bad_layer_visible = True
    for name, layer in interval_layers.items():
        _add_interval_layer_trace(
            fig,
            x_values=layer["x"],
            y_values=layer["y"],
            name=name,
            color=str(layer["color"]),
            visible=True if not show_legend else (bad_layer_visible if name == "Bad SNR" else optional_layers_visible),
            showlegend=show_legend,
        )

    if time_window_s is None:
        label = f" {filter_label}" if filter_label else ""
        title = f"All animals: {len(cell_rows)}{label} {selection_title} stacked"
        xaxis_title = "Absolute time (s)"
    else:
        label = f" {filter_label}" if filter_label else ""
        title = f"All animals: {len(cell_rows)}{label} {selection_title} stacked, {time_window_s[0]:g}-{time_window_s[1]:g} s"
        xaxis_title = "Absolute time (s)"

    if not np.isfinite(x_axis_start_s):
        x_axis_start_s = 0.0

    fig.update_layout(
        title=title,
        height=max(700, row_height * max(1, len(cell_rows))),
        margin={"l": 190, "r": 24, "t": 48, "b": 42},
        plot_bgcolor="white",
        paper_bgcolor="white",
        shapes=shapes,
        xaxis={
            "title": xaxis_title,
            "range": [x_axis_start_s, x_axis_end_s],
            "minallowed": x_axis_start_s,
            "maxallowed": x_axis_end_s,
            "showgrid": True,
            "gridcolor": "#eeeeee",
        },
        yaxis={
            "title": "Animal / neuron",
            "tickmode": "array",
            "tickvals": tickvals,
            "ticktext": ticktext,
            "range": [plot_y0, plot_y1],
            "minallowed": plot_y0,
            "maxallowed": plot_y1,
            "fixedrange": True,
            "showgrid": False,
            "zeroline": False,
        },
        dragmode="zoom",
        hovermode="closest",
        showlegend=show_legend,
        legend={"groupclick": "togglegroup"},
        meta={"cell_jump_rows": cell_jump_rows, "cell_click_curves": cell_click_curves},
        uirevision=f"all-animals-{str(cell_selection)}",
    )
    return fig, stats_rows


def plot_pooled_place_cells_all_animals_html(
    data_root: str | Path | None = None,
    output_html: str | Path | None = None,
    *,
    animal_ids: list[str] | None = None,
    snr_threshold: float = 3.5,
    min_good_minutes: float = 5.0,
    trace_max_points: int | None = None,
    category_filter: set[str] | None = None,
    optional_layers_visible: bool = True,
    bad_epochs_visible: bool | None = None,
    include_plotlyjs: bool | str = True,
    segment_duration_s: float | None = None,
    show_legend: bool = True,
    y_label_mode: str = "identity",
    cell_selection: str = "place",
    cell_key_filter: set[tuple[str, int]] | None = None,
) -> tuple[go.Figure, list[dict[str, Any]], Path]:
    """Build and save a standalone pooled place-cell Plotly HTML figure.

    This uses the same raw pooled-place-cell renderer as the Dash view. It loads
    every animal under ``data_root`` unless ``animal_ids`` is provided.
    """
    root = Path(data_root or default_data_root()).expanduser().resolve()
    ids = list(animal_ids) if animal_ids is not None else discover_animals(root)
    animals = [load_animal(root, animal_id) for animal_id in ids]
    y_label_lookup = (
        _pooled_place_cell_number_lookup(animals, category_filter, cell_selection=cell_selection)
        if str(y_label_mode) == "number"
        else None
    )

    if output_html is None:
        output_path = root.parent / "figures" / "CKII_pooled" / "pooled_place_cells_trace_viewer.html"
    else:
        output_path = Path(output_html).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    max_duration_s = max((a.n_frames / float(a.frame_rate) for a in animals), default=0.0)
    windows = _segment_windows(max_duration_s, segment_duration_s)
    if segment_duration_s is not None and len(windows) > 1:
        segments: list[dict[str, Any]] = []
        first_fig: go.Figure | None = None
        first_stats: list[dict[str, Any]] = []
        for idx, window in enumerate(windows):
            fig, stats_rows = build_pooled_place_cells_figure(
                animals,
                snr_threshold=float(snr_threshold),
                min_good_minutes=float(min_good_minutes),
                trace_max_points=trace_max_points,
                time_window_s=window,
                category_filter=category_filter,
                optional_layers_visible=optional_layers_visible,
                bad_epochs_visible=bad_epochs_visible,
                show_legend=show_legend,
                y_label_mode=y_label_mode,
                y_label_lookup=y_label_lookup,
                cell_selection=cell_selection,
                cell_key_filter=cell_key_filter,
            )
            if first_fig is None:
                first_fig = fig
                first_stats = stats_rows
            start_s, end_s = window
            segments.append(
                {
                    "label": f"{start_s:g}-{end_s:g} s",
                    "start_label": _format_segment_start_minute(start_s),
                    "start_s": start_s,
                    "end_s": end_s,
                    "figure": fig.to_plotly_json(),
                }
            )
        _write_segmented_pooled_html(
            output_path=output_path,
            segments=segments,
            include_plotlyjs=include_plotlyjs,
            title=first_fig.layout.title.text if first_fig and first_fig.layout.title.text else f"Pooled {_cell_selection_title(cell_selection)}",
        )
        return first_fig or empty_figure(f"No pooled {_cell_selection_title(cell_selection)} found."), first_stats, output_path

    fig, stats_rows = build_pooled_place_cells_figure(
        animals,
        snr_threshold=float(snr_threshold),
        min_good_minutes=float(min_good_minutes),
        trace_max_points=trace_max_points,
        category_filter=category_filter,
        optional_layers_visible=optional_layers_visible,
        bad_epochs_visible=bad_epochs_visible,
        show_legend=show_legend,
        y_label_mode=y_label_mode,
        y_label_lookup=y_label_lookup,
        cell_selection=cell_selection,
        cell_key_filter=cell_key_filter,
    )

    fig.write_html(
        str(output_path),
        include_plotlyjs=include_plotlyjs,
        full_html=True,
        config=POOLED_HTML_CONFIG,
        post_script=POOLED_HTML_WHEEL_PAN_SCRIPT,
    )
    return fig, stats_rows, output_path


def plot_pooled_place_cells_by_category_html(
    data_root: str | Path | None = None,
    output_dir: str | Path | None = None,
    *,
    animal_ids: list[str] | None = None,
    categories: tuple[str, ...] = ("CS+ PLC", "CS- PLC"),
    snr_threshold: float = 3.5,
    min_good_minutes: float = 5.0,
    trace_max_points: int | None = None,
    optional_layers_visible: bool = False,
    bad_epochs_visible: bool = True,
    include_plotlyjs: bool | str = "cdn",
    segment_duration_s: float | None = None,
    show_legend: bool = True,
    y_label_mode: str = "identity",
    cell_selection: str = "place",
) -> dict[str, dict[str, Any]]:
    """Save one full-duration raw pooled-place-cell HTML per PLC category.

    ``trace_max_points=None`` preserves every frame. Passing an integer is an
    explicit opt-in to min/max visual compression.
    """
    root = Path(data_root or default_data_root()).expanduser().resolve()
    ids = list(animal_ids) if animal_ids is not None else discover_animals(root)
    animals = [load_animal(root, animal_id) for animal_id in ids]

    if output_dir is None:
        output_path = root.parent / "figures" / "CKII_pooled"
    else:
        output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, dict[str, Any]] = {}
    max_duration_s = max((a.n_frames / float(a.frame_rate) for a in animals), default=0.0)
    windows = _segment_windows(max_duration_s, segment_duration_s)
    for category in categories:
        y_label_lookup = (
            _pooled_place_cell_number_lookup(animals, {category}, cell_selection=cell_selection)
            if str(y_label_mode) == "number"
            else None
        )
        html_path = output_path / CATEGORY_HTML_FILENAMES.get(
            category,
            f"pooled_place_cells_{category.replace(' ', '_').replace('+', 'plus').replace('-', 'minus')}.html",
        )
        if segment_duration_s is not None and len(windows) > 1:
            segments: list[dict[str, Any]] = []
            first_fig: go.Figure | None = None
            first_stats: list[dict[str, Any]] = []
            for idx, window in enumerate(windows):
                fig, stats_rows = build_pooled_place_cells_figure(
                    animals,
                    snr_threshold=float(snr_threshold),
                    min_good_minutes=float(min_good_minutes),
                    trace_max_points=trace_max_points,
                    time_window_s=window,
                    category_filter={category},
                    optional_layers_visible=optional_layers_visible,
                    bad_epochs_visible=bad_epochs_visible,
                    show_legend=show_legend,
                    y_label_mode=y_label_mode,
                    y_label_lookup=y_label_lookup,
                    cell_selection=cell_selection,
                )
                if first_fig is None:
                    first_fig = fig
                    first_stats = stats_rows
                start_s, end_s = window
                segments.append(
                    {
                        "label": f"{start_s:g}-{end_s:g} s",
                        "start_label": _format_segment_start_minute(start_s),
                        "start_s": start_s,
                        "end_s": end_s,
                        "figure": fig.to_plotly_json(),
                    }
                )
            _write_segmented_pooled_html(
                output_path=html_path,
                segments=segments,
                include_plotlyjs=include_plotlyjs,
                title=first_fig.layout.title.text if first_fig and first_fig.layout.title.text else f"{category} pooled {_cell_selection_title(cell_selection)}",
            )
            stats_rows = first_stats
            n_cells = len(first_stats)
        else:
            fig, stats_rows = build_pooled_place_cells_figure(
                animals,
                snr_threshold=float(snr_threshold),
                min_good_minutes=float(min_good_minutes),
                trace_max_points=trace_max_points,
                category_filter={category},
                optional_layers_visible=optional_layers_visible,
                bad_epochs_visible=bad_epochs_visible,
                show_legend=show_legend,
                y_label_mode=y_label_mode,
                y_label_lookup=y_label_lookup,
                cell_selection=cell_selection,
            )
            fig.write_html(
                str(html_path),
                include_plotlyjs=include_plotlyjs,
                full_html=True,
                config=POOLED_HTML_CONFIG,
                post_script=POOLED_HTML_WHEEL_PAN_SCRIPT,
            )
            n_cells = len(stats_rows)
        outputs[category] = {
            "path": html_path,
            "stats": stats_rows,
            "n_cells": n_cells,
            "n_segments": len(windows) if segment_duration_s is not None else 1,
        }
    return outputs


def _spike_tick_trace_row(
    *,
    spikes: np.ndarray,
    frame_rate: float,
    start: int,
    end: int,
    y0: float,
    y1: float,
    name: str,
    color: str,
    absolute_time: bool = False,
) -> go.Scattergl | None:
    spks = np.asarray(spikes, dtype=int).reshape(-1)
    if spks.size == 0:
        return None
    left = int(np.searchsorted(spks, start, side="left"))
    right = int(np.searchsorted(spks, end, side="left"))
    spks = spks[left:right]
    if spks.size == 0:
        return None
    if absolute_time:
        rel = spks.astype(float) / float(frame_rate)
    else:
        rel = (spks - int(start)).astype(float) / float(frame_rate)
    x = np.empty(spks.size * 3, dtype=float)
    y = np.empty(spks.size * 3, dtype=float)
    x[0::3] = rel
    x[1::3] = rel
    x[2::3] = np.nan
    y[0::3] = y0
    y[1::3] = y1
    y[2::3] = np.nan
    return go.Scattergl(
        x=x,
        y=y,
        mode="lines",
        line={"color": color, "width": 1},
        name=name,
        hoverinfo="skip",
        showlegend=False,
    )


def build_single_cell_figure(
    animal: AnimalData,
    cell_idx: int,
    snr_threshold: float,
    min_good_minutes: float,
    row_duration_s: float,
    *,
    row_start_idx: int = 0,
    row_stop_idx: int | None = None,
    frame_start: int | None = None,
    frame_end: int | None = None,
    row_height_px: float | int | None = DEFAULT_ROW_HEIGHT_PX,
) -> tuple[go.Figure, list[dict[str, Any]]]:
    cell_idx = int(np.clip(int(cell_idx), 0, max(0, animal.n_cells - 1)))
    row_height = _clean_row_height_px(row_height_px)
    row_duration_s = max(0.5, float(row_duration_s))
    row_frames = max(1, int(round(row_duration_s * float(animal.frame_rate))))
    base_start = 0 if frame_start is None else int(np.clip(int(frame_start), 0, animal.n_frames))
    base_end = animal.n_frames if frame_end is None else int(np.clip(int(frame_end), base_start, animal.n_frames))
    n_rows = int(np.ceil(max(0, base_end - base_start) / float(row_frames)))
    row_start_idx = int(np.clip(int(row_start_idx), 0, max(0, n_rows)))
    if row_stop_idx is None:
        row_stop_idx = n_rows
    row_stop_idx = int(np.clip(int(row_stop_idx), row_start_idx, max(row_start_idx, n_rows)))

    fig = go.Figure()
    shapes: list[dict[str, Any]] = []
    tickvals: list[float] = []
    ticktext: list[str] = []

    trace = get_cell_vector(animal.traces, cell_idx, animal.n_frames)
    vm = get_cell_vector(animal.vms, cell_idx, animal.n_frames)
    center, scale = _robust_center_scale(trace, vm)
    bad_mask, stats = bad_mask_for_cell(animal, cell_idx, snr_threshold, min_good_minutes)
    snr_values = compute_snr_values(animal, cell_idx)
    bad_intervals = mask_to_intervals(bad_mask)
    cb_intervals = dict_intervals(animal.complex_bursts[cell_idx], animal.n_frames)
    plateau_intervals = dict_intervals(animal.plateaus[cell_idx], animal.n_frames)
    row_items: list[dict[str, Any]] = []
    bounds: list[tuple[float, float]] = []

    for row_idx in range(row_start_idx, row_stop_idx):
        start = base_start + row_idx * row_frames
        end = min(base_end, start + row_frames)
        if end <= start:
            continue
        trace_y = _scaled(trace[start:end], center, scale, 0.0)
        vm_y = _scaled(vm[start:end], center, scale, 0.0)
        row_bottom, row_top, raster_y0, raster_y1 = _row_geometry(trace_y, vm_y)
        bounds.append((row_bottom, row_top))
        row_items.append(
            {
                "row_idx": row_idx,
                "start": start,
                "end": end,
                "trace_y": trace_y,
                "vm_y": vm_y,
                "row_bottom": row_bottom,
                "row_top": row_top,
                "raster_y0": raster_y0,
                "raster_y1": raster_y1,
            }
        )

    offsets, plot_y0, plot_y1 = _stack_offsets(bounds)

    for item, offset in zip(row_items, offsets):
        row_idx = int(item["row_idx"])
        start = int(item["start"])
        end = int(item["end"])
        tickvals.append(offset)
        ticktext.append(f"{start / animal.frame_rate:.0f}-{end / animal.frame_rate:.0f}s")
        y0 = offset + float(item["row_bottom"])
        y1 = offset + float(item["row_top"])
        _add_interval_shapes(
            shapes,
            bad_intervals,
            frame_rate=animal.frame_rate,
            y0=y0,
            y1=y1,
            color=BAD_FILL,
            row_start=start,
            row_end=end,
        )
        _add_interval_shapes(
            shapes,
            cb_intervals,
            frame_rate=animal.frame_rate,
            y0=y0,
            y1=y1,
            color=CB_FILL,
            row_start=start,
            row_end=end,
        )
        _add_interval_shapes(
            shapes,
            plateau_intervals,
            frame_rate=animal.frame_rate,
            y0=y0,
            y1=y1,
            color=PLATEAU_FILL,
            row_start=start,
            row_end=end,
        )
        hover_data = _hover_customdata(start, end, animal.frame_rate, snr_values)
        category = _cell_category(animal, cell_idx)

        fig.add_trace(
            go.Scattergl(
                y=item["trace_y"] + offset,
                x0=0,
                dx=1.0 / float(animal.frame_rate),
                mode="lines",
                line={"color": TRACE_COLOR, "width": 0.7},
                name=f"Row {row_idx + 1} trace",
                customdata=hover_data,
                hovertemplate=(
                    f"Cell {cell_idx + 1} (idx {cell_idx}) row {row_idx + 1}"
                    f"<br>{category}"
                    "<br>t=%{customdata[0]:.3f} s"
                    "<br>SNR=%{customdata[1]:.3f}<extra></extra>"
                ),
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scattergl(
                y=item["vm_y"] + offset,
                x0=0,
                dx=1.0 / float(animal.frame_rate),
                mode="lines",
                line={"color": VM_COLOR, "width": 0.6},
                name=f"Row {row_idx + 1} Vm",
                customdata=hover_data,
                hovertemplate=(
                    f"Cell {cell_idx + 1} Vm (idx {cell_idx}) row {row_idx + 1}"
                    f"<br>{category}"
                    "<br>t=%{customdata[0]:.3f} s"
                    "<br>SNR=%{customdata[1]:.3f}<extra></extra>"
                ),
                showlegend=False,
            )
        )
        ss_trace = _spike_tick_trace_row(
            spikes=animal.refined_ss[cell_idx],
            frame_rate=animal.frame_rate,
            start=start,
            end=end,
            y0=offset + float(item["raster_y0"]),
            y1=offset + float(item["raster_y1"]),
            name=f"Row {row_idx + 1} SS",
            color=SS_COLOR,
        )
        cs_trace = _spike_tick_trace_row(
            spikes=animal.all_cs_spikes[cell_idx],
            frame_rate=animal.frame_rate,
            start=start,
            end=end,
            y0=offset + float(item["raster_y0"]),
            y1=offset + float(item["raster_y1"]),
            name=f"Row {row_idx + 1} CS",
            color=CS_COLOR,
        )
        if ss_trace is not None:
            fig.add_trace(ss_trace)
        if cs_trace is not None:
            fig.add_trace(cs_trace)

    if row_items:
        display_start_s = row_items[0]["start"] / float(animal.frame_rate)
        display_end_s = row_items[-1]["end"] / float(animal.frame_rate)
        title = (
            f"{animal.animal_id}: Cell {cell_idx + 1}, "
            f"{display_start_s:.0f}-{display_end_s:.0f} s, {row_duration_s:g} s rows"
        )
    else:
        title = f"{animal.animal_id}: Cell {cell_idx + 1}, no rows to display"

    fig.update_layout(
        title=title,
        height=max(520, row_height * max(1, len(row_items))),
        margin={"l": 92, "r": 24, "t": 48, "b": 42},
        plot_bgcolor="white",
        paper_bgcolor="white",
        shapes=shapes,
        xaxis={
            "title": "Time within row (s)",
            "range": [0, row_duration_s],
            "minallowed": 0,
            "maxallowed": row_duration_s,
            "showgrid": True,
            "gridcolor": "#eeeeee",
        },
        yaxis={
            "title": "Absolute time row",
            "tickmode": "array",
            "tickvals": tickvals,
            "ticktext": ticktext,
            "range": [plot_y0, plot_y1],
            "minallowed": plot_y0,
            "maxallowed": plot_y1,
            "fixedrange": True,
            "showgrid": False,
            "zeroline": False,
        },
        dragmode="zoom",
        hovermode="closest",
        uirevision=f"{animal.animal_id}-cell-{cell_idx}-rows-{row_duration_s:g}-{base_start}-{base_end}-{row_start_idx}-{row_stop_idx}",
    )
    return fig, [stats]


def build_single_cell_figures(
    animal: AnimalData,
    cell_idx: int,
    snr_threshold: float,
    min_good_minutes: float,
    row_duration_s: float,
    *,
    max_rows_per_figure: int = 60,
    time_window_s: tuple[float, float] | None = None,
    row_height_px: float | int | None = DEFAULT_ROW_HEIGHT_PX,
) -> tuple[list[go.Figure], list[dict[str, Any]]]:
    if time_window_s is None:
        frame_start = 0
        frame_end = animal.n_frames
    else:
        start_s, end_s = float(time_window_s[0]), float(time_window_s[1])
        frame_start = max(0, int(np.floor(start_s * float(animal.frame_rate))))
        frame_end = min(animal.n_frames, int(np.ceil(end_s * float(animal.frame_rate))))

    row_duration_s = max(0.5, float(row_duration_s))
    row_frames = max(1, int(round(row_duration_s * float(animal.frame_rate))))
    n_rows = int(np.ceil(max(0, frame_end - frame_start) / float(row_frames)))
    max_rows_per_figure = max(1, int(max_rows_per_figure))

    figs: list[go.Figure] = []
    stats_rows: list[dict[str, Any]] = []
    for row_start in range(0, n_rows, max_rows_per_figure):
        row_stop = min(n_rows, row_start + max_rows_per_figure)
        fig, stats = build_single_cell_figure(
            animal,
            cell_idx,
            snr_threshold,
            min_good_minutes,
            row_duration_s,
            row_start_idx=row_start,
            row_stop_idx=row_stop,
            frame_start=frame_start,
            frame_end=frame_end,
            row_height_px=row_height_px,
        )
        figs.append(fig)
        if not stats_rows:
            stats_rows = stats
    if not figs:
        figs = [empty_figure(f"{animal.animal_id}: Cell {int(cell_idx) + 1}, no frames to display")]
    return figs, stats_rows


def build_single_cell_continuous_figure(
    animal: AnimalData,
    cell_idx: int,
    snr_threshold: float,
    min_good_minutes: float,
    *,
    time_window_s: tuple[float, float] | None = None,
) -> tuple[go.Figure, list[dict[str, Any]]]:
    cell_idx = int(np.clip(int(cell_idx), 0, max(0, animal.n_cells - 1)))
    if time_window_s is None:
        frame_start = 0
        frame_end = animal.n_frames
    else:
        start_s, end_s = float(time_window_s[0]), float(time_window_s[1])
        frame_start = max(0, int(np.floor(start_s * float(animal.frame_rate))))
        frame_end = min(animal.n_frames, int(np.ceil(end_s * float(animal.frame_rate))))

    if frame_end <= frame_start:
        return empty_figure(f"{animal.animal_id}: Cell {cell_idx + 1}, no frames in selected window"), []

    trace = get_cell_vector(animal.traces, cell_idx, animal.n_frames)
    vm = get_cell_vector(animal.vms, cell_idx, animal.n_frames)
    center, scale = _robust_center_scale(trace, vm)
    trace_y = _scaled(trace[frame_start:frame_end], center, scale, 0.0)
    vm_y = _scaled(vm[frame_start:frame_end], center, scale, 0.0)
    row_bottom, row_top, raster_y0, raster_y1 = _row_geometry(trace_y, vm_y)

    bad_mask, stats = bad_mask_for_cell(animal, cell_idx, snr_threshold, min_good_minutes)
    snr_values = compute_snr_values(animal, cell_idx)
    bad_intervals = mask_to_intervals(bad_mask)
    cb_intervals = dict_intervals(animal.complex_bursts[cell_idx], animal.n_frames)
    plateau_intervals = dict_intervals(animal.plateaus[cell_idx], animal.n_frames)
    hover_data = _hover_customdata(frame_start, frame_end, animal.frame_rate, snr_values)
    category = _cell_category(animal, cell_idx)

    fig = go.Figure()
    shapes: list[dict[str, Any]] = []
    _add_interval_shapes(
        shapes,
        bad_intervals,
        frame_rate=animal.frame_rate,
        y0=row_bottom,
        y1=row_top,
        color=BAD_FILL,
        row_start=frame_start,
        row_end=frame_end,
        absolute_time=True,
    )
    _add_interval_shapes(
        shapes,
        cb_intervals,
        frame_rate=animal.frame_rate,
        y0=row_bottom,
        y1=row_top,
        color=CB_FILL,
        row_start=frame_start,
        row_end=frame_end,
        absolute_time=True,
    )
    _add_interval_shapes(
        shapes,
        plateau_intervals,
        frame_rate=animal.frame_rate,
        y0=row_bottom,
        y1=row_top,
        color=PLATEAU_FILL,
        row_start=frame_start,
        row_end=frame_end,
        absolute_time=True,
    )
    _add_session_shapes(shapes, animal, row_bottom, row_top, row_start=frame_start, row_end=frame_end, absolute_time=True)

    fig.add_trace(
        go.Scattergl(
            y=trace_y,
            x0=frame_start / float(animal.frame_rate),
            dx=1.0 / float(animal.frame_rate),
            mode="lines",
            line={"color": TRACE_COLOR, "width": 0.8},
            name=f"Cell {cell_idx + 1} trace",
            customdata=hover_data,
            hovertemplate=(
                f"Cell {cell_idx + 1} (idx {cell_idx})"
                f"<br>{category}"
                "<br>t=%{customdata[0]:.3f} s"
                "<br>SNR=%{customdata[1]:.3f}<extra></extra>"
            ),
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scattergl(
            y=vm_y,
            x0=frame_start / float(animal.frame_rate),
            dx=1.0 / float(animal.frame_rate),
            mode="lines",
            line={"color": VM_COLOR, "width": 0.7},
            name=f"Cell {cell_idx + 1} Vm",
            customdata=hover_data,
            hovertemplate=(
                f"Cell {cell_idx + 1} Vm (idx {cell_idx})"
                f"<br>{category}"
                "<br>t=%{customdata[0]:.3f} s"
                "<br>SNR=%{customdata[1]:.3f}<extra></extra>"
            ),
            showlegend=False,
        )
    )

    ss_trace = _spike_tick_trace_row(
        spikes=animal.refined_ss[cell_idx],
        frame_rate=animal.frame_rate,
        start=frame_start,
        end=frame_end,
        y0=raster_y0,
        y1=raster_y1,
        name=f"Cell {cell_idx + 1} SS",
        color=SS_COLOR,
        absolute_time=True,
    )
    cs_trace = _spike_tick_trace_row(
        spikes=animal.all_cs_spikes[cell_idx],
        frame_rate=animal.frame_rate,
        start=frame_start,
        end=frame_end,
        y0=raster_y0,
        y1=raster_y1,
        name=f"Cell {cell_idx + 1} CS",
        color=CS_COLOR,
        absolute_time=True,
    )
    if ss_trace is not None:
        fig.add_trace(ss_trace)
    if cs_trace is not None:
        fig.add_trace(cs_trace)

    display_start_s = frame_start / float(animal.frame_rate)
    display_end_s = frame_end / float(animal.frame_rate)
    fig.update_layout(
        title=f"{animal.animal_id}: Cell {cell_idx + 1}, {display_start_s:.0f}-{display_end_s:.0f} s",
        height=620,
        margin={"l": 64, "r": 24, "t": 48, "b": 42},
        plot_bgcolor="white",
        paper_bgcolor="white",
        shapes=shapes,
        xaxis={
            "title": "Absolute time (s)",
            "range": [display_start_s, display_end_s],
            "minallowed": display_start_s,
            "maxallowed": display_end_s,
            "showgrid": True,
            "gridcolor": "#eeeeee",
        },
        yaxis={
            "title": "Signal",
            "range": [row_bottom - PLOT_PAD, row_top + PLOT_PAD],
            "showgrid": False,
            "zeroline": False,
        },
        dragmode="zoom",
        hovermode="closest",
        uirevision=f"{animal.animal_id}-cell-{cell_idx}-continuous-{frame_start}-{frame_end}",
    )
    return fig, [stats]


def empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        height=520,
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis={"visible": False},
        yaxis={"visible": False},
        annotations=[
            {
                "text": message,
                "xref": "paper",
                "yref": "paper",
                "x": 0.5,
                "y": 0.5,
                "showarrow": False,
                "font": {"size": 14},
            }
        ],
    )
    return fig
