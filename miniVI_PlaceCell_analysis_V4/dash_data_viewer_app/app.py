"""Dash GUI for viewing CKII merged aligned neural traces.

Usage
-----
    python dash_data_viewer_app/app.py --data-root miniVI_PlaceCell_analysis_V4/data --port 8053
"""

from __future__ import annotations

import argparse
from functools import lru_cache
import os
from pathlib import Path
import sys
import threading
import webbrowser

_THIS_DIR = Path(__file__).resolve().parent
_ANALYSIS_ROOT = _THIS_DIR.parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
if str(_ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS_ROOT))

from dash import ALL, Dash, Input, Output, State, ctx, dcc, html  # noqa: E402
from dash.exceptions import PreventUpdate  # noqa: E402

try:
    from .data_io import default_data_root, discover_animals, load_animal  # noqa: E402
    from .figure_builder import build_pooled_place_cells_figure, build_single_cell_figures, build_stacked_figure, empty_figure  # noqa: E402
except ImportError:  # pragma: no cover - direct script execution
    from data_io import default_data_root, discover_animals, load_animal
    from figure_builder import build_pooled_place_cells_figure, build_single_cell_figures, build_stacked_figure, empty_figure


CARD_STYLE = {
    "border": "1px solid #ddd",
    "padding": "8px",
    "borderRadius": "6px",
    "backgroundColor": "white",
    "marginBottom": "8px",
}
LABEL_STYLE = {"fontSize": "12px", "marginBottom": "2px", "fontWeight": 600}
INPUT_STYLE = {"width": "100%", "fontSize": "12px"}
BUTTON_STYLE = {
    "height": "34px",
    "fontSize": "12px",
    "border": "1px solid #bbb",
    "borderRadius": "4px",
    "backgroundColor": "#f8f8f8",
    "cursor": "pointer",
}
SECTION_HEADER = {"marginTop": "4px", "marginBottom": "6px", "fontSize": "13px", "fontWeight": 600}
GRAPH_CONFIG = {
    "scrollZoom": False,
    "doubleClick": "reset",
    "displaylogo": False,
    "responsive": True,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
}
SINGLE_ROWS_PER_PANEL = 60
DEFAULT_SNR_THRESHOLD = 3.5
DEFAULT_WINDOW_DURATION_S = 120.0
DEFAULT_ROW_HEIGHT_PX = 112
MIN_ROW_HEIGHT_PX = 60
MAX_ROW_HEIGHT_PX = 180
SINGLE_PANEL_TARGET_HEIGHT_PX = SINGLE_ROWS_PER_PANEL * DEFAULT_ROW_HEIGHT_PX


def _safe_float(value, fallback: float) -> float:
    try:
        out = float(value)
    except Exception:
        return float(fallback)
    if not (out == out):
        return float(fallback)
    return out


def _safe_int(value, fallback: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(fallback)


def _safe_row_height(value) -> int:
    row_height = _safe_float(value, DEFAULT_ROW_HEIGHT_PX)
    row_height = min(MAX_ROW_HEIGHT_PX, max(MIN_ROW_HEIGHT_PX, row_height))
    return int(round(row_height))


def _number_row(
    label: str,
    input_id: str,
    value: float,
    *,
    step: float,
    min_value: float | None = None,
    disabled: bool = False,
):
    kwargs = {
        "id": input_id,
        "type": "number",
        "value": value,
        "step": step,
        "debounce": True,
        "disabled": disabled,
        "style": INPUT_STYLE,
    }
    if min_value is not None:
        kwargs["min"] = min_value
    return html.Div(
        [
            html.Label(label, style=LABEL_STYLE),
            dcc.Input(**kwargs),
        ],
        style={"marginBottom": "6px"},
    )


def _make_options(values: list[str]) -> list[dict[str, str]]:
    return [{"label": v, "value": v} for v in values]


def _place_cell_indices(animal) -> list[int]:
    mask = getattr(animal, "place_cell_mask", [])
    return [i for i in range(animal.n_cells) if i < len(mask) and bool(mask[i])]


def _display_cell_indices(animal, view_mode: str) -> list[int]:
    if view_mode == "place_cells":
        return _place_cell_indices(animal)
    return list(range(animal.n_cells))


def _category_for_cell(animal, cell_idx: int) -> str:
    categories = getattr(animal, "plc_categories", [])
    if 0 <= int(cell_idx) < len(categories):
        return str(categories[int(cell_idx)])
    return "Unknown"


def _cell_option_label(animal, cell_idx: int) -> str:
    category = _category_for_cell(animal, cell_idx)
    if category in {"CS+ PLC", "CS- PLC"}:
        return f"Cell {cell_idx + 1} (idx {cell_idx}, {category})"
    return f"Cell {cell_idx + 1} (idx {cell_idx}, Non-PLC)"


def _category_counts(animal) -> tuple[int, int, int]:
    place_mask = getattr(animal, "place_cell_mask", [])
    csplus_mask = getattr(animal, "csplus_plc_mask", [])
    n_place = sum(1 for i in range(animal.n_cells) if i < len(place_mask) and bool(place_mask[i]))
    n_csplus = sum(1 for i in range(animal.n_cells) if i < len(csplus_mask) and bool(csplus_mask[i]))
    n_csminus = max(0, int(n_place) - int(n_csplus))
    return int(n_place), int(n_csplus), int(n_csminus)


def _format_minute_number(seconds: float) -> str:
    minutes = max(0.0, float(seconds)) / 60.0
    if abs(minutes - round(minutes)) < 1e-9:
        return str(int(round(minutes)))
    return f"{minutes:g}"


def _time_window_from_segment(segment_value, duration_value, max_duration_s: float) -> tuple[tuple[float, float] | None, str]:
    max_duration_s = max(0.0, float(max_duration_s))
    segment_idx = max(0, _safe_int(segment_value, 0))
    duration_s = _safe_float(duration_value, DEFAULT_WINDOW_DURATION_S)
    if duration_s <= 0 or max_duration_s <= 0:
        return None, "full duration"
    duration_s = max(0.5, float(duration_s))
    max_segment = max(0, int((max_duration_s - 1e-9) // duration_s))
    segment_idx = min(segment_idx, max_segment)
    start_s = min(max_duration_s, segment_idx * duration_s)
    end_s = min(max_duration_s, start_s + max(0.5, float(duration_s)))
    if end_s <= start_s:
        start_s = max(0.0, max_duration_s - max(0.5, float(duration_s)))
        end_s = max_duration_s
    return (start_s, end_s), f"segment {segment_idx + 1}/{max_segment + 1} ({start_s:g}-{end_s:g} s)"


def _segment_slider_props(duration_value, max_duration_s: float, current_segment) -> tuple[int, int, int, dict[int, str], bool]:
    duration_s = _safe_float(duration_value, DEFAULT_WINDOW_DURATION_S)
    max_duration_s = max(0.0, float(max_duration_s))
    if duration_s <= 0 or max_duration_s <= 0:
        return 0, 1, 0, {0: "full"}, True

    duration_s = max(0.5, float(duration_s))
    max_segment = max(0, int((max_duration_s - 1e-9) // duration_s))
    value = min(max(0, _safe_int(current_segment, 0)), max_segment)
    if max_segment == 0:
        return 0, 1, 0, {0: "0"}, True

    if max_segment <= 14:
        marks = {idx: _format_minute_number(idx * duration_s) for idx in range(max_segment + 1)}
    else:
        marks = {
            0: "0",
            max_segment: _format_minute_number(max_segment * duration_s),
        }
    return max_segment, 1, value, marks, False


def create_app(*, data_root: str | Path | None = None) -> Dash:
    root = Path(data_root or default_data_root()).expanduser().resolve()
    app = Dash(__name__)
    app.title = "Trace Data Viewer"

    animals = discover_animals(root)
    default_animal = animals[0] if animals else None

    @lru_cache(maxsize=8)
    def _load_cached(animal_id: str):
        return load_animal(root, animal_id)

    def _load_all_animals():
        return [_load_cached(animal_id) for animal_id in animals]

    app.layout = html.Div(
        style={
            "display": "grid",
            "gridTemplateColumns": "auto 1fr",
            "height": "100vh",
            "fontFamily": "Arial, sans-serif",
            "fontSize": "12px",
            "gap": "8px",
            "padding": "8px",
            "boxSizing": "border-box",
        },
        children=[
            html.Div(
                style={
                    "width": "330px",
                    "minWidth": "280px",
                    "maxWidth": "70vw",
                    "resize": "horizontal",
                    "overflow": "auto",
                    "boxSizing": "border-box",
                    "backgroundColor": "#f7f7f7",
                    "border": "1px solid #ddd",
                    "borderRadius": "6px",
                    "padding": "10px",
                },
                children=[
                    dcc.Store(id="cell-jump-request", data=None),
                    html.Div(
                        style=CARD_STYLE,
                        children=[
                            html.Div("Dataset", style=SECTION_HEADER),
                            html.Label("Animal", style=LABEL_STYLE),
                            dcc.Dropdown(
                                id="animal-dropdown",
                                options=_make_options(animals),
                                value=default_animal,
                                clearable=False,
                                style={"fontSize": "12px"},
                            ),
                            html.Div(
                                f"Data root: {root}",
                                style={"fontSize": "10px", "marginTop": "6px", "color": "#666", "wordBreak": "break-all"},
                            ),
                        ],
                    ),
                    html.Div(
                        style=CARD_STYLE,
                        children=[
                            html.Div("View", style=SECTION_HEADER),
                            dcc.Tabs(
                                id="view-tabs",
                                value="stacked",
                                children=[
                                    dcc.Tab(label="All cells stacked", value="stacked"),
                                    dcc.Tab(label="All place cells", value="place_cells"),
                                    dcc.Tab(label="Pooled place cells", value="pooled_place_cells"),
                                    dcc.Tab(label="Single cell", value="single"),
                                ],
                                style={"fontSize": "12px"},
                            ),
                            html.Div(style={"height": "8px"}),
                            html.Label("Cell", style=LABEL_STYLE),
                            html.Div(
                                style={
                                    "display": "grid",
                                    "gridTemplateColumns": "54px minmax(0, 1fr) 54px",
                                    "gap": "6px",
                                    "alignItems": "center",
                                },
                                children=[
                                    html.Button("Prev", id="prev-cell", n_clicks=0, disabled=True, style=BUTTON_STYLE),
                                    dcc.Dropdown(id="cell-dropdown", clearable=False, style={"fontSize": "12px"}),
                                    html.Button("Next", id="next-cell", n_clicks=0, disabled=True, style=BUTTON_STYLE),
                                ],
                            ),
                        ],
                    ),
                    html.Div(
                        style=CARD_STYLE,
                        children=[
                            html.Div("SNR Mask", style=SECTION_HEADER),
                            _number_row("SNR threshold", "snr-threshold", DEFAULT_SNR_THRESHOLD, step=0.1, min_value=0.0),
                            _number_row("Min good minutes", "min-good-minutes", 5.0, step=0.5, min_value=0.0),
                        ],
                    ),
                    html.Div(
                        style=CARD_STYLE,
                        children=[
                            html.Div("Single Cell Rows", style=SECTION_HEADER),
                            _number_row("Row duration (s)", "row-duration", 10.0, step=1.0, min_value=0.5, disabled=True),
                        ],
                    ),
                    html.Div(
                        style=CARD_STYLE,
                        children=[
                            html.Div("Display", style=SECTION_HEADER),
                            html.Label("Row height (px)", style=LABEL_STYLE),
                            dcc.Slider(
                                id="row-height",
                                min=MIN_ROW_HEIGHT_PX,
                                max=MAX_ROW_HEIGHT_PX,
                                step=4,
                                value=DEFAULT_ROW_HEIGHT_PX,
                                marks={
                                    MIN_ROW_HEIGHT_PX: str(MIN_ROW_HEIGHT_PX),
                                    DEFAULT_ROW_HEIGHT_PX: str(DEFAULT_ROW_HEIGHT_PX),
                                    MAX_ROW_HEIGHT_PX: str(MAX_ROW_HEIGHT_PX),
                                },
                                included=False,
                                updatemode="mouseup",
                                tooltip={"placement": "bottom", "always_visible": False},
                            ),
                        ],
                    ),
                    html.Div(
                        style=CARD_STYLE,
                        children=[
                            html.Div("Time Window", style=SECTION_HEADER),
                            _number_row("Duration (s)", "window-duration", DEFAULT_WINDOW_DURATION_S, step=10.0, min_value=0.0),
                            html.Label("Start minute", style=LABEL_STYLE),
                            dcc.Slider(
                                id="window-segment",
                                min=0,
                                max=0,
                                step=1,
                                value=0,
                                marks={0: "0"},
                                included=False,
                                updatemode="mouseup",
                                tooltip={"placement": "bottom", "always_visible": False},
                            ),
                        ],
                    ),
                    html.Div(
                        style=CARD_STYLE,
                        children=[
                            html.Div("Legend", style=SECTION_HEADER),
                            html.Div("Black: trace", style={"marginBottom": "3px"}),
                            html.Div("Purple: subthreshold Vm", style={"marginBottom": "3px", "color": "#6A4C93"}),
                            html.Div("Teal ticks: simple spikes", style={"marginBottom": "3px", "color": "#026C80"}),
                            html.Div("Orange ticks/spans: complex spikes/bursts", style={"marginBottom": "3px", "color": "#B36A00"}),
                            html.Div("Red spans: plateaus", style={"marginBottom": "3px", "color": "#D62828"}),
                            html.Div("Gray spans: bad SNR/invalid behavior frames", style={"marginBottom": "3px", "color": "#666"}),
                            html.Div("Place-cell labels: CS+ PLC / CS- PLC", style={"color": "#555"}),
                        ],
                    ),
                    html.Div(id="metadata-text", style={"fontSize": "11px", "color": "#555", "lineHeight": "1.35"}),
                    html.Div(id="render-status", style={"fontSize": "11px", "color": "#555", "lineHeight": "1.35", "marginTop": "6px"}),
                ],
            ),
            html.Div(
                style={"height": "100%", "overflow": "auto", "border": "1px solid #ddd", "borderRadius": "6px"},
                children=[
                    dcc.Loading(
                        id="plot-loading",
                        type="circle",
                        children=html.Div(
                            id="plot-container",
                            children=[
                                dcc.Graph(
                                    id={"type": "trace-graph", "index": 0},
                                    figure=empty_figure("Select an animal to load traces."),
                                    config=GRAPH_CONFIG,
                                    style={"width": "100%"},
                                )
                            ],
                        ),
                    )
                ],
            ),
        ],
    )

    @app.callback(
        Output("cell-dropdown", "options"),
        Output("cell-dropdown", "value"),
        Output("metadata-text", "children"),
        Input("animal-dropdown", "value"),
        Input("view-tabs", "value"),
        Input("prev-cell", "n_clicks"),
        Input("next-cell", "n_clicks"),
        State("cell-dropdown", "value"),
    )
    def _update_cell_options(animal_id, view_mode, prev_clicks, next_clicks, current_cell):
        if not animal_id:
            return [], None, "No animals found."
        view_mode = str(view_mode)
        if view_mode == "pooled_place_cells":
            loaded_animals = _load_all_animals()
            n_cells = sum(a.n_cells for a in loaded_animals)
            n_place_cells = n_csplus = n_csminus = 0
            for a in loaded_animals:
                pc, csp, csm = _category_counts(a)
                n_place_cells += pc
                n_csplus += csp
                n_csminus += csm
            status = (
                f"All animals pooled: {len(loaded_animals)} animals, {n_cells} cells. "
                f"Place cells: {n_place_cells}/{n_cells} "
                f"(CS+ PLC: {n_csplus}, CS- PLC: {n_csminus}). "
                "The animal dropdown is ignored in this view."
            )
            return [], None, status

        animal = _load_cached(str(animal_id))
        display_cells = _display_cell_indices(animal, view_mode)
        options = [{"label": _cell_option_label(animal, i), "value": i} for i in display_cells]
        value = _safe_int(current_cell, display_cells[0] if display_cells else 0)
        if display_cells and ctx.triggered_id in {"prev-cell", "next-cell"} and view_mode == "single":
            if value not in display_cells:
                value = display_cells[0]
            idx = display_cells.index(value)
            step = -1 if ctx.triggered_id == "prev-cell" else 1
            value = display_cells[(idx + step) % len(display_cells)]
        elif value not in display_cells:
            value = display_cells[0] if display_cells else None
        duration_min = animal.n_frames / animal.frame_rate / 60.0
        n_place_cells, n_csplus, n_csminus = _category_counts(animal)
        mode_msg = " All place cells view." if view_mode == "place_cells" else ""
        status = (
            f"{animal.animal_id}: {animal.n_cells} cells, {animal.n_frames:,} frames, "
            f"{duration_min:.1f} min, {animal.frame_rate:g} Hz. "
            f"Place cells: {n_place_cells}/{animal.n_cells} "
            f"(CS+ PLC: {n_csplus}, CS- PLC: {n_csminus}) from {animal.place_cell_source}.{mode_msg} "
            f"Loaded from {animal.path.name}."
        )
        return options, value, status

    @app.callback(
        Output("animal-dropdown", "value"),
        Output("view-tabs", "value"),
        Output("cell-dropdown", "value", allow_duplicate=True),
        Input("cell-jump-request", "data"),
        Input({"type": "trace-graph", "index": ALL}, "clickData"),
        State({"type": "trace-graph", "index": ALL}, "figure"),
        State("view-tabs", "value"),
        prevent_initial_call=True,
    )
    def _jump_to_single_cell(request, click_data_list, figure_list, current_view):
        target = None
        if ctx.triggered_id == "cell-jump-request":
            target = request if isinstance(request, dict) else None
        elif isinstance(ctx.triggered_id, dict) and ctx.triggered_id.get("type") == "trace-graph":
            if str(current_view) not in {"stacked", "place_cells", "pooled_place_cells"}:
                raise PreventUpdate
            graph_idx = _safe_int(ctx.triggered_id.get("index"), 0)
            if graph_idx < 0 or graph_idx >= len(click_data_list or []) or graph_idx >= len(figure_list or []):
                raise PreventUpdate
            click_data = (click_data_list or [])[graph_idx]
            figure = (figure_list or [])[graph_idx]
            points = click_data.get("points", []) if isinstance(click_data, dict) else []
            if not points:
                raise PreventUpdate
            curve_number = str(points[0].get("curveNumber", ""))
            meta = ((figure or {}).get("layout") or {}).get("meta") or {}
            target = (meta.get("cell_click_curves") or {}).get(curve_number)
        if not isinstance(target, dict):
            raise PreventUpdate

        animal_id = str(target.get("animal_id", ""))
        if animal_id not in animals:
            raise PreventUpdate
        animal = _load_cached(animal_id)
        cell_idx = _safe_int(target.get("cell_idx"), 0)
        cell_idx = min(max(0, cell_idx), max(0, animal.n_cells - 1))
        return animal_id, "single", cell_idx

    @app.callback(
        Output("row-duration", "disabled"),
        Output("window-duration", "disabled"),
        Output("prev-cell", "disabled"),
        Output("next-cell", "disabled"),
        Input("view-tabs", "value"),
    )
    def _update_mode_controls(view_mode):
        is_single_cell = str(view_mode) == "single"
        return (not is_single_cell, is_single_cell, not is_single_cell, not is_single_cell)

    @app.callback(
        Output("window-segment", "max"),
        Output("window-segment", "step"),
        Output("window-segment", "value"),
        Output("window-segment", "marks"),
        Output("window-segment", "disabled"),
        Input("animal-dropdown", "value"),
        Input("view-tabs", "value"),
        Input("window-duration", "value"),
        State("window-segment", "value"),
    )
    def _update_segment_slider(animal_id, view_mode, window_duration, current_segment):
        if not animal_id:
            return 0, 1, 0, {0: "full"}, True
        if str(view_mode) == "single":
            return 0, 1, 0, {0: "full"}, True
        if str(view_mode) == "pooled_place_cells":
            loaded_animals = _load_all_animals()
            max_duration_s = max((a.n_frames / float(a.frame_rate) for a in loaded_animals), default=0.0)
        else:
            animal = _load_cached(str(animal_id))
            max_duration_s = animal.n_frames / float(animal.frame_rate)
        return _segment_slider_props(window_duration, max_duration_s, current_segment)

    @app.callback(
        Output("plot-container", "children"),
        Output("render-status", "children"),
        Input("animal-dropdown", "value"),
        Input("view-tabs", "value"),
        Input("cell-dropdown", "value"),
        Input("snr-threshold", "value"),
        Input("min-good-minutes", "value"),
        Input("row-duration", "value"),
        Input("row-height", "value"),
        Input("window-segment", "value"),
        Input("window-duration", "value"),
    )
    def _render(
        animal_id,
        view_mode,
        cell_idx,
        snr_threshold,
        min_good_minutes,
        row_duration,
        row_height,
        window_segment,
        window_duration,
    ):
        if not animal_id:
            raise PreventUpdate
        animal = _load_cached(str(animal_id))
        snr_threshold = max(0.0, _safe_float(snr_threshold, DEFAULT_SNR_THRESHOLD))
        min_good_minutes = max(0.0, _safe_float(min_good_minutes, 5.0))
        row_duration = max(0.5, _safe_float(row_duration, 10.0))
        row_height = _safe_row_height(row_height)
        view_mode = str(view_mode)
        display_cells = _display_cell_indices(animal, view_mode)
        max_duration_s = animal.n_frames / float(animal.frame_rate)
        time_window, window_label = _time_window_from_segment(window_segment, window_duration, max_duration_s)

        if view_mode == "pooled_place_cells":
            loaded_animals = _load_all_animals()
            pooled_max_duration_s = max((a.n_frames / float(a.frame_rate) for a in loaded_animals), default=max_duration_s)
            time_window, window_label = _time_window_from_segment(window_segment, window_duration, pooled_max_duration_s)
            fig, stats_rows = build_pooled_place_cells_figure(
                loaded_animals,
                snr_threshold,
                min_good_minutes,
                time_window_s=time_window,
                row_height_px=row_height,
                optional_layers_visible=True,
                bad_epochs_visible=True,
                show_legend=False,
            )
            children = [
                dcc.Graph(
                    id={"type": "trace-graph", "index": 0},
                    figure=fig,
                    config=GRAPH_CONFIG,
                    style={"width": "100%"},
                )
            ]
            n_cells = sum(a.n_cells for a in loaded_animals)
            n_place_cells = n_csplus = n_csminus = 0
            for a in loaded_animals:
                pc, csp, csm = _category_counts(a)
                n_place_cells += pc
                n_csplus += csp
                n_csminus += csm
            removed = sum(1 for row in stats_rows if row.get("removed_by_min_good_minutes", False))
            eligible = sum(1 for row in stats_rows if row.get("eligible_cell", False))
            status = (
                f"All animals pooled: rendered {len(stats_rows)} place cells from {len(loaded_animals)} animals. "
                f"Window={window_label}; row height={row_height}px; "
                f"SNR threshold={snr_threshold:g}, min good minutes={min_good_minutes:g}. "
                f"Eligible in view: {eligible}/{len(stats_rows)}; removed by min-good rule: {removed}. "
                f"Place cells: {n_place_cells}/{n_cells} total (CS+ PLC: {n_csplus}, CS- PLC: {n_csminus}). Full resolution."
            )
            return children, status

        if view_mode == "single":
            display_cells = list(range(animal.n_cells))
            cell_idx = _safe_int(cell_idx, display_cells[0] if display_cells else 0)
            if cell_idx not in display_cells:
                cell_idx = display_cells[0] if display_cells else None
            if cell_idx is None:
                figs = [empty_figure("No cells found for this animal.")]
                stats_rows = []
            else:
                max_rows_per_panel = max(1, int(SINGLE_PANEL_TARGET_HEIGHT_PX // max(1, row_height)))
                figs, stats_rows = build_single_cell_figures(
                    animal,
                    cell_idx,
                    snr_threshold,
                    min_good_minutes,
                    row_duration,
                    max_rows_per_figure=max_rows_per_panel,
                    time_window_s=None,
                    row_height_px=row_height,
                )
            children = [
                dcc.Graph(
                    id={"type": "trace-graph", "index": i},
                    figure=fig,
                    config=GRAPH_CONFIG,
                    style={"width": "100%"},
                )
                for i, fig in enumerate(figs)
            ]
            full_minutes = animal.n_frames / float(animal.frame_rate) / 60.0
            window_label = f"full recording ({full_minutes:.1f} min)"
            panel_note = (
                f" Single-cell view shows the full {full_minutes:.1f} min recording as {row_duration:g} s rows "
                f"across {len(figs)} panel(s); full resolution."
            )
        else:
            if view_mode == "place_cells" and not display_cells:
                fig = empty_figure("No place cells found for this animal.")
                stats_rows = []
            else:
                fig, stats_rows = build_stacked_figure(
                    animal,
                    snr_threshold,
                    min_good_minutes,
                    cell_indices=display_cells,
                    time_window_s=time_window,
                    row_height_px=row_height,
                )
            children = [
                dcc.Graph(
                    id={"type": "trace-graph", "index": 0},
                    figure=fig,
                    config=GRAPH_CONFIG,
                    style={"width": "100%"},
                )
            ]
            panel_note = " Full resolution."

        removed = sum(1 for row in stats_rows if row.get("removed_by_min_good_minutes", False))
        eligible = sum(1 for row in stats_rows if row.get("eligible_cell", False))
        n_place_cells, n_csplus, n_csminus = _category_counts(animal)
        view_label = "all place cells" if view_mode == "place_cells" else view_mode
        if view_mode == "single":
            category_note = f" Place cells: {n_place_cells} total (CS+ PLC: {n_csplus}, CS- PLC: {n_csminus}); selected cell: {cell_idx + 1 if cell_idx is not None else 'none'}."
        else:
            category_note = f" Place cells: {n_place_cells} total (CS+ PLC: {n_csplus}, CS- PLC: {n_csminus}); shown: {len(display_cells)}."
        status = (
            f"{animal.animal_id}: rendered {view_label}. Window={window_label}; row height={row_height}px; "
            f"SNR threshold={snr_threshold:g}, "
            f"min good minutes={min_good_minutes:g}. Eligible in view: {eligible}/{len(stats_rows)}; "
            f"removed by min-good rule: {removed}.{category_note}{panel_note}"
        )
        return children, status

    return app


def _browser_host(host: str) -> str:
    h = str(host).strip().lower()
    if h in {"0.0.0.0", "::", ""}:
        return "127.0.0.1"
    return host


def _open_browser_later(host: str, port: int, delay_s: float = 1.0) -> None:
    url = f"http://{_browser_host(host)}:{int(port)}"
    timer = threading.Timer(delay_s, lambda: webbrowser.open(url))
    timer.daemon = True
    timer.start()


def run_from_notebook(data_root: str | Path | None = None, *, host: str = "127.0.0.1", port: int = 8053, debug: bool = False):
    app = create_app(data_root=data_root)
    _open_browser_later(host, port)
    app.run(host=host, port=port, debug=debug)


def main() -> None:
    parser = argparse.ArgumentParser(description="Dash CKII trace data viewer")
    parser.add_argument("--data-root", type=str, default=str(default_data_root()))
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8053)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    app = create_app(data_root=args.data_root)
    if not args.no_browser:
        _open_browser_later(args.host, args.port)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
