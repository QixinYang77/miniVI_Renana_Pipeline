import argparse
import math
import threading
import webbrowser

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import ALL, Dash, Input, Output, State, ctx, dcc, html
from dash.exceptions import PreventUpdate

try:
    from plotly_resampler import FigureResampler, register_plotly_resampler

    _HAS_PLOTLY_RESAMPLER = True
except Exception:
    FigureResampler = None
    register_plotly_resampler = None
    _HAS_PLOTLY_RESAMPLER = False

from demix_io import (
    load_stitched_bundle,
    roi_centroid,
    roi_contours,
    save_annotations_compatible,
)
from signal_ops import highpass_filter_trace, make_bad_mask
from trace_store import TraceStore


def _safe_float(v, fallback):
    try:
        return float(v)
    except Exception:
        return float(fallback)


def _sanitize_epochs(epochs, eps=1e-9):
    parsed = []
    for ep in (epochs or []):
        try:
            lo = float(min(float(ep[0]), float(ep[1])))
            hi = float(max(float(ep[0]), float(ep[1])))
        except Exception:
            continue
        if (not np.isfinite(lo)) or (not np.isfinite(hi)) or hi <= lo:
            continue
        parsed.append((lo, hi))
    if not parsed:
        return []
    parsed.sort(key=lambda z: z[0])
    merged = [parsed[0]]
    for lo, hi in parsed[1:]:
        p_lo, p_hi = merged[-1]
        if lo <= p_hi + eps:
            merged[-1] = (p_lo, max(p_hi, hi))
        else:
            merged.append((lo, hi))
    return merged


def _build_params_per_cell(n_cells, frame_rate, saved_params_dict):
    base = {
        i: {
            "hp_hz": 20.0,
            "mad_k": 4.5,
            "min_isi_ms": max(2.0, 1000.0 / max(1.0, float(frame_rate))),
        }
        for i in range(int(n_cells))
    }
    if isinstance(saved_params_dict, dict):
        for k, v in saved_params_dict.items():
            try:
                ki = int(k)
            except Exception:
                continue
            if ki in base and isinstance(v, dict):
                base[ki].update(v)
    return base


def _find_spike_peaks_simple(y, threshold):
    arr = np.asarray(y, dtype=np.float32).reshape(-1)
    if arr.size < 3:
        return np.array([], dtype=int)
    finite = np.isfinite(arr)
    if np.count_nonzero(finite) < 3:
        return np.array([], dtype=int)
    mid = arr[1:-1]
    is_peak = (
        np.isfinite(mid)
        & (mid > threshold)
        & (mid > arr[:-2])
        & (mid >= arr[2:])
    )
    return np.flatnonzero(is_peak).astype(int) + 1


def make_image_figure(bundle, session_idx, kind, cell_indices=None):
    rois = bundle["ROIs"]
    colors = bundle["colors"]
    if cell_indices is None:
        idx_draw = range(len(rois))
    else:
        idx_draw = [int(i) for i in cell_indices if 0 <= int(i) < len(rois)]

    if kind == "raw":
        img = bundle["mean_img_raw_list"][session_idx]
        cs = "gray"
    elif kind == "mc":
        img = bundle["mean_img_mc_list"][session_idx]
        cs = "gray"
    elif kind == "mc_denoised":
        img = bundle["mean_img_mc_denoised_list"][session_idx]
        cs = "gray"
    elif kind == "weights":
        img = bundle["weights_img_list"][session_idx]
        cs = "gray"
    else:
        raise ValueError(f"Unknown image kind: {kind}")

    # Build full ROI hover label map from all cells so hover remains informative
    # even if some cells are deselected from contour display.
    hover_labels = np.full(img.shape, "No cell", dtype=object)
    for ci, mask in enumerate(rois):
        try:
            hover_labels[np.asarray(mask, dtype=bool)] = f"Cell {ci + 1}"
        except Exception:
            continue

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=img,
            colorscale=cs,
            showscale=False,
            text=hover_labels,
            hoverongaps=False,
            hovertemplate="%{text}<br>x=%{x:.1f}, y=%{y:.1f}<extra></extra>",
        )
    )
    for i in idx_draw:
        mask = rois[i]
        for contour in roi_contours(mask):
            fig.add_trace(
                go.Scatter(
                    x=contour[:, 1],
                    y=contour[:, 0],
                    mode="lines",
                    line={"width": 1, "color": colors[i]},
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
        cy, cx = roi_centroid(mask)
        # Move label slightly upward for readability on dense ROIs.
        label_y = max(float(cy) - 2.0, 0.0)
        fig.add_trace(
            go.Scatter(
                x=[cx],
                y=[label_y],
                mode="text",
                text=[str(i + 1)],
                textfont={"color": colors[i], "size": 11},
                hoverinfo="skip",
                showlegend=False,
            )
        )

    h, w = img.shape
    fig.update_xaxes(
        visible=False,
        range=[-0.5, float(w) - 0.5],
        fixedrange=False,
        showgrid=False,
        zeroline=False,
    )
    fig.update_yaxes(
        visible=False,
        range=[float(h) - 0.5, -0.5],
        fixedrange=False,
        showgrid=False,
        zeroline=False,
        scaleanchor="x",
        scaleratio=1,
    )
    fig.update_layout(
        margin={"l": 0, "r": 0, "t": 0, "b": 0, "pad": 0},
        autosize=True,
        dragmode="zoom",
        template="plotly_white",
    )
    return fig


def build_images_grid(bundle, cell_indices=None):
    cards = []
    kinds = ["raw", "mc", "mc_denoised", "weights"]
    draw_cells = None if cell_indices is None else [int(i) for i in cell_indices]

    for si in range(bundle["n_sessions"]):
        row_items = []
        for kind in kinds:
            fig = make_image_figure(bundle, si, kind, cell_indices=draw_cells)
            row_items.append(
                html.Div(
                    dcc.Graph(
                        id={"type": "mean-image", "session": si, "kind": kind},
                        figure=fig,
                        config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d"]},
                        style={"height": "150px"},
                    ),
                    style={
                        "border": "1px solid #ddd",
                        "padding": "2px",
                        "borderRadius": "6px",
                        "backgroundColor": "white",
                    },
                )
            )

        cards.append(
            html.Div(
                [
                    html.Div(
                        f"Session {si + 1} - {bundle['labels_list'][si]}",
                        style={"fontWeight": 600, "margin": "0 0 2px 0", "fontSize": "13px"},
                    ),
                    html.Div(
                        row_items,
                        style={
                            "display": "grid",
                            "gridTemplateColumns": "repeat(4, minmax(180px, 1fr))",
                            "gap": "4px",
                        },
                    ),
                ],
                style={"marginBottom": "4px"},
            )
        )
    return cards


def create_app(bundle):
    trace_store = TraceStore(bundle)
    n_cells = int(bundle["N"])
    t0 = float(bundle["t"][0])
    t1 = float(bundle["t"][-1])
    total_span = max(1.0, t1 - t0)
    default_window_s = max(1.0, total_span / 3.0)
    n_segments = max(1, math.ceil(total_span / default_window_s))
    default_source = (
        "weighted_mc_denoised_traces"
        if "weighted_mc_denoised_traces" in bundle["sources_raw"]
        else list(bundle["sources_raw"].keys())[0]
    )
    img_h, img_w = bundle["mean_img_mc_list"][0].shape
    roi_centroids_xy = []
    for mask in bundle["ROIs"]:
        cy, cx = roi_centroid(mask)
        roi_centroids_xy.append([float(cx), float(cy)])

    app = Dash(__name__)
    app.title = "Demix Overview (Dash)"
    resampler_registered = False
    if _HAS_PLOTLY_RESAMPLER:
        try:
            register_plotly_resampler(mode="auto")
            resampler_registered = True
        except Exception:
            try:
                register_plotly_resampler(app)
                resampler_registered = True
            except Exception:
                resampler_registered = False

    app.layout = html.Div(
        [
            dcc.Store(id="bad-epochs-store", data=_sanitize_epochs(bundle["initial_bad_epochs"])),
            dcc.Store(id="bad-span-mode-store", data=False),
            dcc.Store(id="pending-span-store", data=None),
            dcc.Store(id="remove-baseline-store", data=False),
            dcc.Store(id="motion-regression-store", data=False),
            dcc.Store(id="show-motion-fit-store", data=False),
            dcc.Store(id="motion-fit-refresh-store", data=0),
            dcc.Store(id="full-trace-mode-store", data=False),
            dcc.Store(id="hovered-cell-store", data=None),
            dcc.Store(id="sync-highlight-held-store", data=False),
            dcc.Store(id="roi-centroids-store", data=roi_centroids_xy),
            dcc.Store(id="image-shape-store", data={"h": int(img_h), "w": int(img_w)}),
            dcc.Store(id="show-spikes-store", data=False),
            dcc.Store(
                id="spikes-store",
                data=[list(map(int, s)) for s in (bundle["initial_spikes_verified_array"] or [])],
            ),
            dcc.Store(
                id="spike-params-store",
                data=_build_params_per_cell(
                    n_cells,
                    bundle["frame_rate"],
                    bundle["initial_params_per_cell"],
                ),
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H4("Segment", style={"marginTop": 0, "marginBottom": "6px"}),
                                    dcc.Slider(
                                        id="window-start-slider",
                                        min=1,
                                        max=n_segments,
                                        value=1,
                                        step=1,
                                        tooltip={"always_visible": False, "placement": "bottom"},
                                    ),
                                    html.Button(
                                        "Full trace mode: OFF",
                                        id="toggle-full-trace-mode-btn",
                                        n_clicks=0,
                                        style={"width": "100%", "marginTop": "8px"},
                                    ),
                                    html.Div("Window (s)", style={"marginTop": "8px"}),
                                    dcc.Input(
                                        id="window-seconds-input",
                                        type="number",
                                        min=1,
                                        step=1,
                                        value=default_window_s,
                                        style={"width": "100%"},
                                    ),
                                ],
                                style={
                                    "border": "1px solid #ddd",
                                    "padding": "8px",
                                    "borderRadius": "6px",
                                    "backgroundColor": "white",
                                    "marginBottom": "8px",
                                },
                            ),
                            html.Div(
                                [
                                    html.H4("Overview Image", style={"marginTop": 0, "marginBottom": "6px"}),
                                    html.Div("Session"),
                                    dcc.Dropdown(
                                        id="image-session-dropdown",
                                        options=[
                                            {"label": f"{i + 1}: {bundle['labels_list'][i]}", "value": i}
                                            for i in range(bundle["n_sessions"])
                                        ],
                                        value=0,
                                        clearable=False,
                                    ),
                                    html.Div("Image type", style={"marginTop": "6px"}),
                                    dcc.Dropdown(
                                        id="image-kind-dropdown",
                                        options=[
                                            {"label": "Mean Image - Raw", "value": "raw"},
                                            {"label": "Mean Image - MC", "value": "mc"},
                                            {"label": "MC Denoised", "value": "mc_denoised"},
                                            {"label": "Max Weight Map", "value": "weights"},
                                        ],
                                        value="weights",
                                        clearable=False,
                                    ),
                                    html.Div(
                                        [
                                            dcc.Graph(
                                                id="mean-image-single",
                                                figure=make_image_figure(
                                                    bundle,
                                                    session_idx=0,
                                                    kind="weights",
                                                    cell_indices=bundle["initial_good_cells"],
                                                ),
                                                config={
                                                    "displaylogo": False,
                                                    "displayModeBar": True,
                                                    "modeBarButtonsToRemove": ["lasso2d"],
                                                    "scrollZoom": True,
                                                },
                                                style={"height": "320px", "width": "100%"},
                                            ),
                                            dcc.Graph(
                                                id="mean-image-highlight-overlay",
                                                figure={
                                                    "data": [],
                                                    "layout": {
                                                        "margin": {"l": 0, "r": 0, "t": 0, "b": 0, "pad": 0},
                                                        "xaxis": {
                                                            "visible": False,
                                                            "range": [-0.5, float(img_w) - 0.5],
                                                            "showgrid": False,
                                                            "zeroline": False,
                                                        },
                                                        "yaxis": {
                                                            "visible": False,
                                                            "range": [float(img_h) - 0.5, -0.5],
                                                            "showgrid": False,
                                                            "zeroline": False,
                                                            "scaleanchor": "x",
                                                            "scaleratio": 1,
                                                        },
                                                        "paper_bgcolor": "rgba(0,0,0,0)",
                                                        "plot_bgcolor": "rgba(0,0,0,0)",
                                                        "dragmode": False,
                                                    },
                                                },
                                                config={
                                                    "displaylogo": False,
                                                    "displayModeBar": False,
                                                    "staticPlot": True,
                                                },
                                                style={
                                                    "height": "320px",
                                                    "width": "100%",
                                                    "position": "absolute",
                                                    "top": 0,
                                                    "left": 0,
                                                    "pointerEvents": "none",
                                                },
                                            ),
                                        ],
                                        style={"position": "relative", "height": "320px", "width": "100%"},
                                    ),
                                    html.Div(
                                        "Hold S while hovering a trace to highlight that cell on the image",
                                        style={"fontSize": "11px", "padding": "2px 0", "color": "#111"},
                                    ),
                                    html.Div(
                                        id="image-hover-readout",
                                        style={"fontSize": "12px", "padding": "2px 0", "color": "#111"},
                                    ),
                                ],
                                style={
                                    "border": "1px solid #ddd",
                                    "padding": "8px",
                                    "borderRadius": "6px",
                                    "backgroundColor": "white",
                                    "marginBottom": "8px",
                                },
                            ),
                            html.H3("Controls", style={"marginTop": 0}),
                            html.Div("Trace source"),
                            dcc.Dropdown(
                                id="trace-source-dropdown",
                                options=[{"label": k, "value": k} for k in bundle["sources_raw"].keys()],
                                value=default_source,
                                clearable=False,
                            ),
                            html.Div("Normalization", style={"marginTop": "8px"}),
                            dcc.Dropdown(
                                id="norm-dropdown",
                                options=[
                                    {"label": "z-score", "value": "zscore"},
                                    {"label": "0-1", "value": "0-1"},
                                ],
                                value="0-1",
                                clearable=False,
                            ),
                            html.H4("Cells", style={"marginTop": "12px", "marginBottom": "6px"}),
                            html.Div(
                                [
                                    html.Button("Select All", id="select-all-cells-btn", n_clicks=0),
                                    html.Button(
                                        "Load Saved Good",
                                        id="restore-good-cells-btn",
                                        n_clicks=0,
                                        style={"marginLeft": "6px"},
                                    ),
                                ]
                            ),
                            html.Div(
                                dcc.Checklist(
                                    id="cell-checklist",
                                    options=[
                                        {"label": f"Cell {i + 1}", "value": i}
                                        for i in range(n_cells)
                                    ],
                                    value=bundle["initial_good_cells"],
                                    labelStyle={"display": "block", "padding": "2px 0"},
                                ),
                                style={
                                    "maxHeight": "180px",
                                    "overflowY": "auto",
                                    "border": "1px solid #ddd",
                                    "padding": "6px",
                                },
                            ),
                            html.H4("Bad Epochs", style={"marginTop": "12px", "marginBottom": "6px"}),
                            html.Button("Bad-span mode: OFF", id="toggle-bad-span-btn", n_clicks=0),
                            html.Button(
                                "",
                                id="hotkey-toggle-bad-span-btn",
                                n_clicks=0,
                                style={"display": "none"},
                            ),
                            html.Button(
                                "",
                                id="hotkey-confirm-bad-span-btn",
                                n_clicks=0,
                                style={"display": "none"},
                            ),
                            html.Button(
                                "",
                                id="hotkey-cancel-pending-btn",
                                n_clicks=0,
                                style={"display": "none"},
                            ),
                            html.Button(
                                "",
                                id="hotkey-sync-highlight-down-btn",
                                n_clicks=0,
                                style={"display": "none"},
                            ),
                            html.Button(
                                "",
                                id="hotkey-sync-highlight-up-btn",
                                n_clicks=0,
                                style={"display": "none"},
                            ),
                            html.Div(
                                "Shortcuts: B toggle mode, Enter confirm selection, Esc cancel pending",
                                style={"fontSize": "11px"},
                            ),
                            html.Button(
                                "Confirm Selection",
                                id="confirm-bad-span-btn",
                                n_clicks=0,
                                disabled=True,
                                style={"marginLeft": "6px"},
                            ),
                            html.Div(
                                id="pending-span-text",
                                style={"marginTop": "6px", "fontSize": "12px"},
                            ),
                            html.Div(
                                id="bad-epoch-feedback",
                                style={"marginTop": "6px", "fontSize": "12px"},
                            ),
                            html.Div(
                                id="bad-epoch-list",
                                style={
                                    "maxHeight": "160px",
                                    "overflowY": "auto",
                                    "border": "1px solid #ddd",
                                    "padding": "6px",
                                    "marginTop": "6px",
                                },
                            ),
                            html.Button(
                                "Clear All Bad Epochs",
                                id="clear-bad-epochs-btn",
                                n_clicks=0,
                                style={"width": "100%", "marginTop": "6px"},
                            ),
                            html.H4("Baseline", style={"marginTop": "12px", "marginBottom": "6px"}),
                            html.Div("Baseline window (s)"),
                            dcc.Input(
                                id="baseline-window-input",
                                type="number",
                                min=1,
                                step=1,
                                value=30,
                                style={"width": "100%"},
                            ),
                            dcc.Checklist(
                                id="display-baseline-check",
                                options=[{"label": "Display baseline", "value": "show"}],
                                value=[],
                                labelStyle={"display": "block", "padding": "4px 0"},
                                style={"marginTop": "4px"},
                            ),
                            html.Button(
                                "Remove baseline drift: OFF",
                                id="toggle-remove-baseline-btn",
                                n_clicks=0,
                                style={"width": "100%"},
                            ),
                            html.Div("Display high-pass (Hz)", style={"marginTop": "12px"}),
                            dcc.Checklist(
                                id="highpass-enable-check",
                                options=[{"label": "Enable high-pass (display only)", "value": "on"}],
                                value=[],
                                labelStyle={"display": "block", "padding": "2px 0"},
                            ),
                            dcc.Input(
                                id="highpass-cutoff-input",
                                type="number",
                                min=0.1,
                                step=0.5,
                                value=20.0,
                                style={"width": "100%"},
                            ),
                            html.H4("Motion Regression", style={"marginTop": "12px", "marginBottom": "6px"}),
                            html.Button(
                                "Display fitted motion noise",
                                id="toggle-motion-fit-btn",
                                n_clicks=0,
                                style={"width": "100%", "marginTop": "6px"},
                            ),
                            html.Button(
                                "Remove motion noise",
                                id="toggle-motion-regression-btn",
                                n_clicks=0,
                                style={"width": "100%"},
                            ),
                            html.Div("Pre-motion baseline window (s, 0=off)", style={"marginTop": "6px"}),
                            dcc.Input(
                                id="motion-prebaseline-window-input",
                                type="number",
                                min=0,
                                step=1,
                                value=30,
                                style={"width": "100%"},
                            ),
                            html.Div("Chunk frames", style={"marginTop": "6px"}),
                            dcc.Input(
                                id="motion-chunk-frames-input",
                                type="number",
                                min=1,
                                step=100,
                                value=5000,
                                style={"width": "100%"},
                            ),
                            html.Div("Overlap frames", style={"marginTop": "6px"}),
                            dcc.Input(
                                id="motion-overlap-frames-input",
                                type="number",
                                min=0,
                                step=50,
                                value=0,
                                style={"width": "100%"},
                            ),
                            html.Div("Ridge lambda (0=OLS)", style={"marginTop": "6px"}),
                            dcc.Input(
                                id="motion-ridge-lambda-input",
                                type="number",
                                min=0,
                                step=0.01,
                                value=0.0,
                                style={"width": "100%"},
                            ),
                            html.Div("Motion components (Ub)", style={"marginTop": "6px"}),
                            dcc.Input(
                                id="motion-n-components-input",
                                type="number",
                                min=1,
                                step=1,
                                value=5,
                                style={"width": "100%"},
                            ),
                            html.Div("Shift smoothing frames (0=off)", style={"marginTop": "6px"}),
                            dcc.Input(
                                id="motion-shift-smooth-input",
                                type="number",
                                min=0,
                                step=1,
                                value=0,
                                style={"width": "100%"},
                            ),
                            html.H4("Spike Detection", style={"marginTop": "12px", "marginBottom": "6px"}),
                            html.Div("MAD threshold multiplier"),
                            dcc.Input(
                                id="spike-mad-k-input",
                                type="number",
                                min=0.5,
                                step=0.1,
                                value=4.5,
                                style={"width": "100%"},
                            ),
                            html.Button(
                                "Spike Detection",
                                id="run-spike-detect-btn",
                                n_clicks=0,
                                style={"width": "100%", "marginTop": "6px"},
                            ),
                            html.Button(
                                "Display spikes: OFF",
                                id="clear-spikes-btn",
                                n_clicks=0,
                                style={"width": "100%", "marginTop": "6px"},
                            ),
                            html.Div(
                                id="spike-detect-status",
                                style={"marginTop": "6px", "fontSize": "12px"},
                            ),
                            html.H4("Save", style={"marginTop": "12px", "marginBottom": "6px"}),
                            dcc.Input(
                                id="save-filename-input",
                                type="text",
                                value="volpy_demix_results_annotated.pickle",
                                style={"width": "100%"},
                            ),
                            html.Button(
                                "Save Annotations",
                                id="save-btn",
                                n_clicks=0,
                                style={"width": "100%", "marginTop": "6px"},
                            ),
                            html.Div(
                                id="save-status",
                                style={"marginTop": "6px", "fontSize": "12px"},
                            ),
                        ],
                        style={
                            "padding": "10px",
                            "backgroundColor": "#f7f7f7",
                            "border": "1px solid #ddd",
                            "borderRadius": "6px",
                            "overflow": "auto",
                            "width": "320px",
                            "minWidth": "260px",
                            "maxWidth": "70vw",
                            "resize": "horizontal",
                            "boxSizing": "border-box",
                        },
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    dcc.Graph(
                                        id="trace-graph",
                                        config={
                                            "displaylogo": False,
                                            "doubleClick": "reset",
                                            "modeBarButtonsToRemove": ["lasso2d"],
                                        },
                                        responsive=True,
                                        style={"height": "560px", "minHeight": "420px"},
                                    ),
                                    html.Div(
                                        id="window-label",
                                        style={"fontSize": "12px", "padding": "2px 0"},
                                    ),
                                    html.Div(
                                        id="hover-readout",
                                        style={"fontSize": "12px", "padding": "0 0 2px 0", "color": "#111"},
                                    ),
                                ],
                                style={
                                    "display": "grid",
                                    "gridTemplateRows": "auto auto auto",
                                    "overflowY": "visible",
                                },
                            ),
                        ],
                        style={
                            "padding": "6px 4px",
                            "display": "grid",
                            "gridTemplateRows": "auto",
                            "gap": "8px",
                            "overflowY": "auto",
                            "overflowX": "hidden",
                            "minWidth": 0,
                        },
                    ),
                ],
                style={
                    "height": "100vh",
                    "display": "grid",
                    "gridTemplateColumns": "auto 1fr",
                    "gap": "8px",
                    "padding": "8px",
                    "boxSizing": "border-box",
                },
            ),
        ],
        style={"height": "100vh", "fontFamily": "Arial, sans-serif"},
    )

    @app.callback(
        Output("window-start-slider", "max"),
        Output("window-start-slider", "step"),
        Output("window-start-slider", "value"),
        Output("window-start-slider", "marks"),
        Input("window-seconds-input", "value"),
        State("window-start-slider", "value"),
    )
    def sync_slider_limits(window_s, cur_val):
        w = max(1.0, _safe_float(window_s, default_window_s))
        n_seg = max(1, math.ceil(total_span / w))
        val = min(max(1, int(_safe_float(cur_val, 1))), n_seg)
        if n_seg <= 15:
            marks = {i: str(i) for i in range(1, n_seg + 1)}
        else:
            marks = {1: "1", n_seg: str(n_seg)}
        return n_seg, 1, val, marks

    @app.callback(
        Output("full-trace-mode-store", "data"),
        Output("toggle-full-trace-mode-btn", "children"),
        Output("window-seconds-input", "disabled"),
        Input("toggle-full-trace-mode-btn", "n_clicks"),
        State("full-trace-mode-store", "data"),
        prevent_initial_call=True,
    )
    def toggle_full_trace_mode(_n_clicks, enabled):
        mode = not bool(enabled)
        if mode:
            return True, "Full trace mode: ON", True
        return False, "Full trace mode: OFF", False

    @app.callback(
        Output("window-seconds-input", "value", allow_duplicate=True),
        Output("window-start-slider", "value", allow_duplicate=True),
        Input("full-trace-mode-store", "data"),
        prevent_initial_call=True,
    )
    def apply_full_trace_mode(enabled):
        if bool(enabled):
            return float(total_span), 1
        return float(default_window_s), 1

    @app.callback(
        Output("mean-image-single", "figure"),
        Input("image-session-dropdown", "value"),
        Input("image-kind-dropdown", "value"),
        Input("cell-checklist", "value"),
    )
    def update_single_mean_image(session_idx, kind, selected_cells):
        si = int(_safe_float(session_idx, 0))
        if si < 0 or si >= bundle["n_sessions"]:
            si = 0
        kind = str(kind or "weights")
        if kind not in {"raw", "mc", "mc_denoised", "weights"}:
            kind = "weights"
        cells = [int(i) for i in (selected_cells or []) if 0 <= int(i) < n_cells]
        return make_image_figure(bundle, si, kind, cells)

    @app.callback(
        Output("image-hover-readout", "children"),
        Input("mean-image-single", "hoverData"),
    )
    def show_image_hover_info(hover_data):
        if not hover_data or "points" not in hover_data or not hover_data["points"]:
            return "Image hover: move cursor over mean image"
        p = hover_data["points"][0]
        label = str(p.get("text", "No cell"))
        try:
            x = float(p.get("x", np.nan))
            y = float(p.get("y", np.nan))
            return f"Image hover: {label} at x={x:.1f}, y={y:.1f}"
        except Exception:
            return f"Image hover: {label}"

    @app.callback(
        Output("cell-checklist", "value"),
        Input("select-all-cells-btn", "n_clicks"),
        Input("restore-good-cells-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def set_cell_selection(_sel_all, _restore):
        trig = ctx.triggered_id
        if trig == "select-all-cells-btn":
            return list(range(n_cells))
        if trig == "restore-good-cells-btn":
            return list(bundle["initial_good_cells"])
        raise PreventUpdate

    @app.callback(
        Output("show-spikes-store", "data"),
        Output("clear-spikes-btn", "children"),
        Input("clear-spikes-btn", "n_clicks"),
        State("show-spikes-store", "data"),
        prevent_initial_call=True,
    )
    def toggle_spike_display(_n_clicks, enabled):
        mode = not bool(enabled)
        return mode, f"Display spikes: {'ON' if mode else 'OFF'}"

    @app.callback(
        Output("spikes-store", "data"),
        Output("spike-params-store", "data"),
        Output("spike-detect-status", "children"),
        Input("run-spike-detect-btn", "n_clicks"),
        State("trace-source-dropdown", "value"),
        State("spike-mad-k-input", "value"),
        State("cell-checklist", "value"),
        State("bad-epochs-store", "data"),
        State("spikes-store", "data"),
        State("spike-params-store", "data"),
        prevent_initial_call=True,
    )
    def run_spike_detection(
        _n_clicks,
        source_key,
        mad_k_val,
        selected_cells,
        bad_epochs,
        spikes_store,
        params_store,
    ):
        source_key = source_key or default_source
        if source_key not in bundle["sources_raw"]:
            raise PreventUpdate
        mad_k = max(0.1, _safe_float(mad_k_val, 4.5))
        hp_hz = 20.0

        cells_target = sorted(set(int(i) for i in (selected_cells or []) if 0 <= int(i) < n_cells))
        if len(cells_target) == 0:
            cells_target = list(range(n_cells))

        out_spikes = [list(map(int, s)) for s in (spikes_store or [[] for _ in range(n_cells)])]
        if len(out_spikes) != n_cells:
            out_spikes = [[] for _ in range(n_cells)]

        out_params = params_store if isinstance(params_store, dict) else {}
        if not isinstance(out_params, dict):
            out_params = {}

        source = bundle["sources_raw"][source_key]
        t_axis = bundle["t"]
        bad_mask_global = make_bad_mask(t_axis, _sanitize_epochs(bad_epochs or []))

        total_spikes = 0
        for ci in cells_target:
            trace = source[:, ci].astype(np.float32, copy=False)
            hp = highpass_filter_trace(
                trace,
                bundle["frame_rate"],
                cutoff_hz=hp_hz,
                bad_mask=bad_mask_global,
            )
            if np.any(bad_mask_global):
                hp = hp.copy()
                hp[bad_mask_global] = np.nan

            med = np.nanmedian(hp)
            mad = np.nanmedian(np.abs(hp - med))
            sigma = 1.4826 * mad
            if (not np.isfinite(med)) or (not np.isfinite(sigma)) or sigma <= 0:
                peaks = np.array([], dtype=int)
            else:
                thr = med + mad_k * sigma
                peaks = _find_spike_peaks_simple(hp, thr)
                if np.any(bad_mask_global) and peaks.size > 0:
                    peaks = peaks[~bad_mask_global[peaks]]

            out_spikes[ci] = list(map(int, peaks.tolist()))
            total_spikes += len(out_spikes[ci])
            cur = out_params.get(ci, out_params.get(str(ci), {}))
            if not isinstance(cur, dict):
                cur = {}
            cur = dict(cur)
            cur.update(
                {
                    "mad_k": float(mad_k),
                    "hp_hz": float(hp_hz),
                    "source_key": str(source_key),
                }
            )
            out_params[int(ci)] = cur
            if str(ci) in out_params:
                del out_params[str(ci)]

        msg = (
            f"Spike detection done for {len(cells_target)} cell(s) "
            f"with MAD x {mad_k:.2f} (HP {hp_hz:.1f} Hz). Total spikes: {total_spikes}."
        )
        return out_spikes, out_params, msg

    app.clientside_callback(
        """
        function(hoverData, selectedCells, displayBaselineVals, removeBaselineDrift, hpVals) {
            const emptyMsg = "Hover: move cursor over a cell trace";
            const emptyOut = [emptyMsg, null];
            if (!hoverData || !hoverData.points || hoverData.points.length === 0) {
                return emptyOut;
            }
            const p = hoverData.points[0];
            const tVal = Number(p && p.x);
            if (!Number.isFinite(tVal)) {
                return emptyOut;
            }
            if (p && p.customdata !== undefined && p.customdata !== null && p.customdata !== "") {
                const cellId = Number(p.customdata);
                if (Number.isFinite(cellId)) {
                    const cellIdx = Math.max(0, Math.trunc(cellId) - 1);
                    return [`Hover: Cell ${Math.trunc(cellId)} at t=${tVal.toFixed(3)}`, cellIdx];
                }
            }
            const curve = Number(p && p.curveNumber);
            const cells = (selectedCells || [])
                .map(v => Number(v))
                .filter(v => Number.isFinite(v) && v >= 0)
                .sort((a, b) => a - b);
            if (!Number.isFinite(curve) || cells.length === 0) {
                return emptyOut;
            }
            const hpEnabled = Array.isArray(hpVals) && hpVals.indexOf("on") !== -1;
            const remove = !!removeBaselineDrift;
            const showBaselineRequested = Array.isArray(displayBaselineVals)
                && displayBaselineVals.indexOf("show") !== -1
                && !remove;
            const showBaseline = showBaselineRequested && !hpEnabled;
            const tracesPerCell = showBaseline ? 2 : 1;
            const idx = Math.floor(curve / tracesPerCell);
            if (idx < 0 || idx >= cells.length) {
                return emptyOut;
            }
            return [`Hover: Cell ${cells[idx] + 1} at t=${tVal.toFixed(3)}`, cells[idx]];
        }
        """,
        Output("hover-readout", "children"),
        Output("hovered-cell-store", "data"),
        Input("trace-graph", "hoverData"),
        Input("cell-checklist", "value"),
        Input("display-baseline-check", "value"),
        Input("remove-baseline-store", "data"),
        Input("highpass-enable-check", "value"),
    )

    app.clientside_callback(
        """
        function(downClicks, upClicks, curVal) {
            const ctx = dash_clientside.callback_context;
            if (!ctx || !ctx.triggered || ctx.triggered.length === 0) {
                return !!curVal;
            }
            const trig = String(ctx.triggered[0].prop_id || "");
            if (trig.indexOf("hotkey-sync-highlight-down-btn") === 0) {
                return true;
            }
            if (trig.indexOf("hotkey-sync-highlight-up-btn") === 0) {
                return false;
            }
            return !!curVal;
        }
        """,
        Output("sync-highlight-held-store", "data"),
        Input("hotkey-sync-highlight-down-btn", "n_clicks"),
        Input("hotkey-sync-highlight-up-btn", "n_clicks"),
        State("sync-highlight-held-store", "data"),
        prevent_initial_call=True,
    )

    app.clientside_callback(
        """
        function(syncHeld, hoveredCell, centroids, imageShape, relayoutData) {
            const shape = imageShape || {};
            const h = Number(shape.h || 1);
            const w = Number(shape.w || 1);
            let xRange = [-0.5, w - 0.5];
            let yRange = [h - 0.5, -0.5];

            if (relayoutData) {
                if (Array.isArray(relayoutData["xaxis.range"])) {
                    xRange = relayoutData["xaxis.range"];
                } else if (
                    relayoutData["xaxis.range[0]"] !== undefined &&
                    relayoutData["xaxis.range[1]"] !== undefined
                ) {
                    xRange = [relayoutData["xaxis.range[0]"], relayoutData["xaxis.range[1]"]];
                }
                if (Array.isArray(relayoutData["yaxis.range"])) {
                    yRange = relayoutData["yaxis.range"];
                } else if (
                    relayoutData["yaxis.range[0]"] !== undefined &&
                    relayoutData["yaxis.range[1]"] !== undefined
                ) {
                    yRange = [relayoutData["yaxis.range[0]"], relayoutData["yaxis.range[1]"]];
                }
            }

            const out = {
                data: [],
                layout: {
                    margin: {l: 0, r: 0, t: 0, b: 0, pad: 0},
                    xaxis: {
                        visible: false,
                        range: xRange,
                        showgrid: false,
                        zeroline: false
                    },
                    yaxis: {
                        visible: false,
                        range: yRange,
                        showgrid: false,
                        zeroline: false,
                        scaleanchor: "x",
                        scaleratio: 1
                    },
                    paper_bgcolor: "rgba(0,0,0,0)",
                    plot_bgcolor: "rgba(0,0,0,0)",
                    dragmode: false
                }
            };

            if (!syncHeld) {
                return out;
            }
            const idx = Number(hoveredCell);
            if (!Number.isFinite(idx) || idx < 0) {
                return out;
            }
            const c = (centroids || [])[Math.trunc(idx)];
            if (!Array.isArray(c) || c.length < 2) {
                return out;
            }
            const cx = Number(c[0]);
            const cy = Number(c[1]);
            if (!Number.isFinite(cx) || !Number.isFinite(cy)) {
                return out;
            }
            out.data.push({
                type: "scatter",
                mode: "markers+text",
                x: [cx],
                y: [cy],
                marker: {
                    size: 20,
                    color: "rgba(0,0,0,0)",
                    line: {color: "#FFD400", width: 3},
                    symbol: "circle-open"
                },
                text: [String(Math.trunc(idx) + 1)],
                textposition: "top center",
                textfont: {size: 12, color: "#FFD400"},
                hoverinfo: "skip",
                showlegend: false
            });
            return out;
        }
        """,
        Output("mean-image-highlight-overlay", "figure"),
        Input("sync-highlight-held-store", "data"),
        Input("hovered-cell-store", "data"),
        Input("roi-centroids-store", "data"),
        Input("image-shape-store", "data"),
        Input("mean-image-single", "relayoutData"),
    )

    @app.callback(
        Output("bad-span-mode-store", "data"),
        Output("toggle-bad-span-btn", "children"),
        Output("pending-span-store", "data", allow_duplicate=True),
        Input("toggle-bad-span-btn", "n_clicks"),
        Input("hotkey-toggle-bad-span-btn", "n_clicks"),
        State("bad-span-mode-store", "data"),
        prevent_initial_call=True,
    )
    def toggle_bad_span_mode(_n_clicks, _hotkey_clicks, enabled):
        mode = not bool(enabled)
        if mode:
            return True, "Bad-span mode: ON (drag-select, then Confirm Selection)", None
        return False, "Bad-span mode: OFF", None

    @app.callback(
        Output("pending-span-store", "data"),
        Input("trace-graph", "selectedData"),
        State("bad-span-mode-store", "data"),
        prevent_initial_call=True,
    )
    def capture_pending_span(selected_data, bad_mode):
        if not bool(bad_mode) or not selected_data:
            raise PreventUpdate

        # Use the selection box range (always present for box-select)
        rng = selected_data.get("range", {})
        x_range = rng.get("x", [])
        if len(x_range) >= 2:
            lo = float(min(x_range))
            hi = float(max(x_range))
        else:
            # Fallback: extract from selected points
            points = selected_data.get("points", [])
            xs = []
            for p in points:
                if "x" not in p:
                    continue
                try:
                    xs.append(float(p["x"]))
                except Exception:
                    continue
            if len(xs) < 2:
                raise PreventUpdate
            lo = float(min(xs))
            hi = float(max(xs))

        if hi <= lo:
            raise PreventUpdate

        return {"lo": lo, "hi": hi}

    @app.callback(
        Output("pending-span-text", "children"),
        Output("confirm-bad-span-btn", "disabled"),
        Input("pending-span-store", "data"),
    )
    def render_pending_span(pending_span):
        if not pending_span:
            return "No pending selection.", True
        lo = float(pending_span["lo"])
        hi = float(pending_span["hi"])
        return f"Pending selection: {lo:.3f} to {hi:.3f}. Click Confirm Selection.", False

    @app.callback(
        Output("bad-epochs-store", "data"),
        Output("pending-span-store", "data", allow_duplicate=True),
        Output("bad-epoch-feedback", "children"),
        Input("confirm-bad-span-btn", "n_clicks"),
        Input("hotkey-confirm-bad-span-btn", "n_clicks"),
        Input("clear-bad-epochs-btn", "n_clicks"),
        Input({"type": "delete-epoch-btn", "index": ALL}, "n_clicks"),
        State("pending-span-store", "data"),
        State("bad-epochs-store", "data"),
        prevent_initial_call=True,
    )
    def update_bad_epochs(_confirm, _confirm_hotkey, _clear, _del_clicks, pending_span, bad_epochs):
        trig = ctx.triggered_id
        epochs = _sanitize_epochs(bad_epochs or [])

        if trig == "clear-bad-epochs-btn":
            return [], None, "Cleared all bad epochs."

        if isinstance(trig, dict) and trig.get("type") == "delete-epoch-btn":
            idx = int(trig.get("index", -1))
            if 0 <= idx < len(epochs):
                del epochs[idx]
                return epochs, pending_span, f"Deleted bad epoch #{idx + 1}."
            raise PreventUpdate

        if trig in {"confirm-bad-span-btn", "hotkey-confirm-bad-span-btn"}:
            if not pending_span:
                return epochs, None, "No pending selection to confirm."
            lo = float(min(pending_span["lo"], pending_span["hi"]))
            hi = float(max(pending_span["lo"], pending_span["hi"]))
            if hi <= lo:
                return epochs, None, "Invalid selection range."
            epochs.append((lo, hi))
            epochs = _sanitize_epochs(epochs)
            return epochs, None, f"Added bad epoch: {lo:.3f} to {hi:.3f}"

        raise PreventUpdate

    @app.callback(
        Output("pending-span-store", "data", allow_duplicate=True),
        Output("bad-epoch-feedback", "children", allow_duplicate=True),
        Input("hotkey-cancel-pending-btn", "n_clicks"),
        State("pending-span-store", "data"),
        prevent_initial_call=True,
    )
    def cancel_pending_span_hotkey(_n_clicks, pending_span):
        if not _n_clicks:
            raise PreventUpdate
        if not pending_span:
            return None, "No pending selection to cancel."
        return None, "Cancelled pending selection."

    @app.callback(
        Output("bad-span-mode-store", "data", allow_duplicate=True),
        Output("toggle-bad-span-btn", "children", allow_duplicate=True),
        Input("confirm-bad-span-btn", "n_clicks"),
        Input("hotkey-confirm-bad-span-btn", "n_clicks"),
        State("pending-span-store", "data"),
        prevent_initial_call=True,
    )
    def auto_disable_bad_span_mode(_confirm, _confirm_hotkey, pending_span):
        if not pending_span:
            raise PreventUpdate
        return False, "Bad-span mode: OFF"

    @app.callback(
        Output("bad-epoch-list", "children"),
        Input("bad-epochs-store", "data"),
    )
    def render_bad_epoch_list(bad_epochs):
        epochs = _sanitize_epochs(bad_epochs or [])
        if not epochs:
            return html.Div("No bad epochs.")
        rows = []
        for i, (a, b) in enumerate(epochs):
            lo = min(float(a), float(b))
            hi = max(float(a), float(b))
            rows.append(
                html.Div(
                    [
                        html.Span(f"[{i + 1}] {lo:.3f} - {hi:.3f}", style={"fontSize": "12px"}),
                        html.Button(
                            "Delete",
                            id={"type": "delete-epoch-btn", "index": i},
                            n_clicks=0,
                            style={"marginLeft": "8px", "fontSize": "11px"},
                        ),
                    ],
                    style={"display": "flex", "justifyContent": "space-between", "marginBottom": "4px"},
                )
            )
        return rows

    @app.callback(
        Output("remove-baseline-store", "data"),
        Output("toggle-remove-baseline-btn", "children"),
        Input("toggle-remove-baseline-btn", "n_clicks"),
        State("remove-baseline-store", "data"),
        prevent_initial_call=True,
    )
    def toggle_remove_baseline(_n_clicks, enabled):
        mode = not bool(enabled)
        if mode:
            return True, "Remove baseline drift: ON"
        return False, "Remove baseline drift: OFF"

    @app.callback(
        Output("motion-regression-store", "data"),
        Output("show-motion-fit-store", "data"),
        Output("motion-fit-refresh-store", "data"),
        Output("toggle-motion-regression-btn", "children"),
        Output("toggle-motion-fit-btn", "children"),
        Input("toggle-motion-fit-btn", "n_clicks"),
        Input("toggle-motion-regression-btn", "n_clicks"),
        State("motion-regression-store", "data"),
        State("motion-fit-refresh-store", "data"),
        prevent_initial_call=True,
    )
    def handle_motion_buttons(_show_clicks, _remove_clicks, motion_enabled, refresh_nonce):
        trig = ctx.triggered_id
        nonce = int(_safe_float(refresh_nonce, 0))
        if trig == "toggle-motion-fit-btn":
            # Always recompute fitted noise from raw traces with current params.
            return (
                False,  # preview mode: no subtraction applied
                True,   # show fitted noise
                nonce + 1,
                "Remove motion noise: OFF",
                "Display fitted motion noise (shown)",
            )
        if trig == "toggle-motion-regression-btn":
            # Toggle subtraction on/off. OFF must restore original traces.
            next_mode = not bool(motion_enabled)
            if next_mode:
                return (
                    True,
                    False,
                    nonce,
                    "Remove motion noise: ON",
                    "Display fitted motion noise",
                )
            return (
                False,
                False,
                nonce,
                "Remove motion noise: OFF",
                "Display fitted motion noise",
            )
        raise PreventUpdate

    @app.callback(
        Output("trace-graph", "figure"),
        Output("window-label", "children"),
        Output("trace-graph", "style"),
        Input("trace-source-dropdown", "value"),
        Input("cell-checklist", "value"),
        Input("window-start-slider", "value"),
        Input("window-seconds-input", "value"),
        Input("norm-dropdown", "value"),
        Input("highpass-enable-check", "value"),
        Input("highpass-cutoff-input", "value"),
        Input("bad-epochs-store", "data"),
        Input("bad-span-mode-store", "data"),
        Input("pending-span-store", "data"),
        Input("baseline-window-input", "value"),
        Input("display-baseline-check", "value"),
        Input("remove-baseline-store", "data"),
        Input("motion-regression-store", "data"),
        Input("motion-prebaseline-window-input", "value"),
        Input("motion-chunk-frames-input", "value"),
        Input("motion-overlap-frames-input", "value"),
        Input("motion-ridge-lambda-input", "value"),
        Input("motion-n-components-input", "value"),
        Input("motion-shift-smooth-input", "value"),
        Input("show-motion-fit-store", "data"),
        Input("motion-fit-refresh-store", "data"),
        Input("full-trace-mode-store", "data"),
        Input("spikes-store", "data"),
        Input("show-spikes-store", "data"),
    )
    def build_trace_figure(
        source_key,
        selected_cells,
        start_t,
        window_s,
        norm_mode,
        highpass_enable_vals,
        highpass_cutoff_hz,
        bad_epochs,
        bad_span_mode,
        pending_span,
        baseline_window_s,
        display_baseline_vals,
        remove_baseline_drift,
        motion_regression_enabled,
        motion_prebaseline_window_s,
        motion_chunk_frames,
        motion_overlap_frames,
        motion_ridge_lambda,
        motion_n_components,
        motion_shift_smooth_frames,
        show_motion_fit,
        motion_fit_refresh_nonce,
        full_trace_mode,
        spikes_store,
        show_spikes_store,
    ):
        source_key = source_key or default_source
        if source_key not in bundle["sources_raw"]:
            raise PreventUpdate
        cells = sorted(set(int(i) for i in (selected_cells or []) if 0 <= int(i) < n_cells))
        w = max(1.0, _safe_float(window_s, default_window_s))
        n_seg_total = max(1, math.ceil(total_span / w))
        segment = max(1, int(_safe_float(start_t, 1)))
        st0 = t0 + (segment - 1) * w
        hp_enabled = "on" in (highpass_enable_vals or [])
        hp_cutoff = max(0.1, _safe_float(highpass_cutoff_hz, 20.0))
        baseline_window_s = max(1.0, _safe_float(baseline_window_s, 30.0))
        remove_baseline_drift = bool(remove_baseline_drift)
        motion_regression_enabled = bool(motion_regression_enabled)
        motion_prebaseline_window_s = max(0.0, _safe_float(motion_prebaseline_window_s, 30.0))
        motion_chunk_frames = max(1, int(_safe_float(motion_chunk_frames, 5000)))
        motion_overlap_frames = max(0, int(_safe_float(motion_overlap_frames, 0)))
        motion_ridge_lambda = max(0.0, _safe_float(motion_ridge_lambda, 0.0))
        motion_n_components = max(1, int(_safe_float(motion_n_components, 5)))
        motion_shift_smooth_frames = max(0, int(_safe_float(motion_shift_smooth_frames, 0)))
        show_motion_fit = bool(show_motion_fit)
        motion_fit_refresh_nonce = int(_safe_float(motion_fit_refresh_nonce, 0))
        if motion_overlap_frames >= motion_chunk_frames:
            motion_overlap_frames = max(0, motion_chunk_frames - 1)
        full_trace_mode = bool(full_trace_mode)
        show_baseline_requested = ("show" in (display_baseline_vals or [])) and (not remove_baseline_drift)
        show_baseline = show_baseline_requested and (not hp_enabled)
        fit_baseline = bool(show_baseline_requested or remove_baseline_drift)
        use_resampler = bool(
            full_trace_mode and n_seg_total == 1 and _HAS_PLOTLY_RESAMPLER and resampler_registered
        )
        resampler_target = max(20000, int(round(30.0 * float(bundle["frame_rate"]))))

        bad_epochs_clean = _sanitize_epochs(bad_epochs or [])
        spikes_by_cell = [[] for _ in range(n_cells)]
        if isinstance(spikes_store, (list, tuple)):
            for ci in range(min(n_cells, len(spikes_store))):
                try:
                    spikes_by_cell[ci] = list(map(int, spikes_store[ci]))
                except Exception:
                    spikes_by_cell[ci] = []
        data = trace_store.get_window_data(
            source_key=source_key,
            cells=cells,
            start_t=st0,
            window_s=w,
            bad_epochs=bad_epochs_clean,
            normalize=norm_mode,
            baseline_cfg={"enabled": False},
            baseline_display_cfg={
                "fit": fit_baseline,
                "show": show_baseline,
                "remove": remove_baseline_drift,
                "percentile": 50.0,
                "window_s": baseline_window_s,
                "smooth": "median",
            },
            highpass_cfg={"enabled": hp_enabled, "cutoff_hz": hp_cutoff},
            motion_regression_cfg={
                "enabled": motion_regression_enabled,
                "pre_baseline_window_s": motion_prebaseline_window_s,
                "chunk_frames": motion_chunk_frames,
                "overlap_frames": motion_overlap_frames,
                "ridge_lambda": motion_ridge_lambda,
                "n_components": motion_n_components,
                "shift_smooth_frames": motion_shift_smooth_frames,
                "show_fitted": show_motion_fit,
                "refresh_nonce": motion_fit_refresh_nonce,
            },
            target_points=0,
            hide_bad_epochs=True,
        )

        base_fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            row_heights=[0.92, 0.08],
            vertical_spacing=0.02,
        )
        if use_resampler:
            try:
                fig = FigureResampler(base_fig, default_n_shown_samples=resampler_target)
            except Exception:
                fig = base_fig
                use_resampler = False
        else:
            fig = base_fig
        y_step = 1.0
        # Lift y-axis cell labels slightly so they align higher on each trace band.
        label_lift = 0.22 * y_step

        yticks = []
        ylabels = []
        spikes_visible = 0
        n_traces = len(data["traces"])
        for idx, tr in enumerate(data["traces"]):
            cidx = int(tr["cell"])
            # Keep Cell 1 at the top without flipping waveform orientation.
            yoff = (n_traces - 1 - idx) * y_step
            yplot = tr["y"] + yoff
            cell_trace = go.Scattergl(
                x=tr["x"],
                y=yplot,
                mode="lines",
                line={"width": 1, "color": bundle["colors"][cidx]},
                name=f"Cell {cidx + 1}",
                customdata=np.full(len(tr["x"]), cidx + 1, dtype=int),
                hovertemplate="Cell %{customdata}<br>Time=%{x:.3f}<extra></extra>",
                showlegend=False,
            )
            if use_resampler:
                try:
                    fig.add_trace(
                        cell_trace,
                        row=1,
                        col=1,
                        hf_x=tr["x"],
                        hf_y=yplot,
                        max_n_samples=resampler_target,
                    )
                except Exception:
                    use_resampler = False
                    fig.add_trace(cell_trace, row=1, col=1)
            else:
                fig.add_trace(cell_trace, row=1, col=1)

            # Overlay spike markers (global indices mapped into current display window).
            if bool(show_spikes_store) and cidx < len(spikes_by_cell) and len(spikes_by_cell[cidx]) > 0:
                spk_idx = np.asarray(spikes_by_cell[cidx], dtype=int)
                spk_idx = spk_idx[(spk_idx >= 0) & (spk_idx < bundle["t"].shape[0])]
                if spk_idx.size > 0:
                    spk_idx = spk_idx[(spk_idx >= int(data["i0"])) & (spk_idx < int(data["i1"]))]
                    if spk_idx.size > 0:
                        spk_t = bundle["t"][spk_idx].astype(np.float32, copy=False)
                        x_src = np.asarray(tr["x"], dtype=np.float32)
                        y_src = np.asarray(tr["y"], dtype=np.float32)
                        valid = np.isfinite(x_src) & np.isfinite(y_src)
                        if np.count_nonzero(valid) >= 2:
                            x_valid = x_src[valid]
                            y_valid = y_src[valid]
                            xmin = float(x_valid[0])
                            xmax = float(x_valid[-1])
                            in_rng = (spk_t >= xmin) & (spk_t <= xmax)
                            if np.any(in_rng):
                                spk_t_plot = spk_t[in_rng]
                                spk_y_plot = np.interp(spk_t_plot, x_valid, y_valid).astype(np.float32) + yoff
                                spikes_visible += int(spk_t_plot.size)
                                fig.add_trace(
                                    go.Scattergl(
                                        x=spk_t_plot,
                                        y=spk_y_plot,
                                        mode="markers",
                                        marker={
                                            "size": 6,
                                            "symbol": "circle",
                                            "color": bundle["colors"][cidx],
                                            "line": {"color": "#000000", "width": 0.7},
                                        },
                                        customdata=np.full(spk_t_plot.size, cidx + 1, dtype=int),
                                        hovertemplate=(
                                            "Spike<br>Cell %{customdata}<br>Time=%{x:.3f}"
                                            "<extra></extra>"
                                        ),
                                        name=f"Spikes {cidx + 1}",
                                        showlegend=False,
                                    ),
                                    row=1,
                                    col=1,
                                )

            if show_baseline and tr.get("baseline_y") is not None:
                bx = tr.get("baseline_x", tr["x"])
                baseline_trace = go.Scattergl(
                    x=bx,
                    y=tr["baseline_y"] + yoff,
                    mode="lines",
                    line={"width": 1.3, "color": "#7A7A7A"},
                    name=f"Baseline {cidx + 1}",
                    hoverinfo="skip",
                    showlegend=False,
                )
                if use_resampler:
                    try:
                        fig.add_trace(
                            baseline_trace,
                            row=1,
                            col=1,
                            hf_x=bx,
                            hf_y=tr["baseline_y"] + yoff,
                            max_n_samples=resampler_target,
                        )
                    except Exception:
                        use_resampler = False
                        fig.add_trace(baseline_trace, row=1, col=1)
                else:
                    fig.add_trace(baseline_trace, row=1, col=1)
            if show_motion_fit and tr.get("motion_fit_y") is not None:
                mx = tr.get("motion_fit_x", tr["x"])
                motion_fit_trace = go.Scattergl(
                    x=mx,
                    y=tr["motion_fit_y"] + yoff,
                    mode="lines",
                    line={"width": 2.2, "color": "#555555"},
                    name=f"Motion fit {cidx + 1}",
                    hoverinfo="skip",
                    showlegend=False,
                )
                if use_resampler:
                    try:
                        fig.add_trace(
                            motion_fit_trace,
                            row=1,
                            col=1,
                            hf_x=mx,
                            hf_y=tr["motion_fit_y"] + yoff,
                            max_n_samples=resampler_target,
                        )
                    except Exception:
                        use_resampler = False
                        fig.add_trace(motion_fit_trace, row=1, col=1)
                else:
                    fig.add_trace(motion_fit_trace, row=1, col=1)
            yticks.append(yoff + label_lift)
            ylabels.append(f"Cell {cidx + 1}")


        t_lo = float(data["t0"])
        t_hi = float(data["t1"])
        i0 = int(data["i0"])
        i1 = int(data["i1"])
        t_seg = bundle["t"][i0:i1]
        sd_seg = bundle["shift_distances"][i0:i1]
        bad_mask_local = data.get("bad_mask_local", None)
        if bad_mask_local is not None and len(bad_mask_local) == len(t_seg):
            keep = ~bad_mask_local
            if np.any(keep):
                t_seg = t_seg[keep]
                sd_seg = sd_seg[keep]
        shift_trace = go.Scattergl(
            x=t_seg,
            y=sd_seg,
            mode="lines",
            line={"width": 1.5, "color": "#FF0054"},
            name="Shift",
            showlegend=False,
            hoverinfo="skip",
        )
        if use_resampler:
            try:
                fig.add_trace(
                    shift_trace,
                    row=2,
                    col=1,
                    hf_x=t_seg,
                    hf_y=sd_seg,
                    max_n_samples=resampler_target,
                )
            except Exception:
                use_resampler = False
                fig.add_trace(shift_trace, row=2, col=1)
        else:
            fig.add_trace(shift_trace, row=2, col=1)

        for bt in bundle["session_boundary_times"]:
            if t_lo <= bt <= t_hi:
                fig.add_vline(x=bt, line_dash="dot", line_width=1.2, line_color="#00A8E8", row=1, col=1)
                fig.add_vline(x=bt, line_dash="dot", line_width=1.2, line_color="#00A8E8", row=2, col=1)

        # Confirmed bad epochs — red shaded regions
        for ep in bad_epochs_clean:
            ep_lo = float(min(float(ep[0]), float(ep[1])))
            ep_hi = float(max(float(ep[0]), float(ep[1])))
            if ep_hi >= t_lo and ep_lo <= t_hi:
                for row in [1, 2]:
                    fig.add_vrect(
                        x0=ep_lo, x1=ep_hi,
                        fillcolor="red", opacity=0.15,
                        line_width=1, line_color="red", line_dash="dot",
                        row=row, col=1,
                    )

        # Pending selection — orange highlight
        if bool(bad_span_mode) and pending_span:
            ps_lo = float(min(pending_span["lo"], pending_span["hi"]))
            ps_hi = float(max(pending_span["lo"], pending_span["hi"]))
            if ps_hi >= t_lo and ps_lo <= t_hi:
                for row in [1, 2]:
                    fig.add_vrect(
                        x0=ps_lo, x1=ps_hi,
                        fillcolor="orange", opacity=0.2,
                        line_width=1, line_color="orange",
                        row=row, col=1,
                    )

        fig.update_layout(
            template="plotly_white",
            autosize=True,
            margin={"l": 70, "r": 10, "t": 10, "b": 30},
            hovermode="closest",
            spikedistance=-1,
            dragmode="select" if bool(bad_span_mode) else "zoom",
            uirevision=f"{source_key}:{','.join(str(x) for x in cells)}",
        )
        fig.update_xaxes(
            range=[t_lo, t_hi],
            showspikes=True,
            spikesnap="cursor",
            spikemode="across",
            spikedash="dash",
            spikecolor="#555555",
            spikethickness=1,
            row=1,
            col=1,
        )
        fig.update_xaxes(
            title_text=bundle["time_label"],
            range=[t_lo, t_hi],
            showspikes=True,
            spikesnap="cursor",
            spikemode="across",
            spikedash="dash",
            spikecolor="#555555",
            spikethickness=1,
            row=2,
            col=1,
        )
        fig.update_yaxes(
            title_text="Cells",
            tickmode="array",
            tickvals=yticks,
            ticktext=ylabels,
            row=1,
            col=1,
        )
        fig.update_yaxes(title_text="Shift", row=2, col=1)

        n_cells_sel = max(1, len(cells))
        graph_h = int(min(2600, max(520, 120 + 44 * n_cells_sel)))
        graph_style = {"height": f"{graph_h}px", "minHeight": "420px"}
        mode_txt = "ON (drag-select, then Confirm)" if bad_span_mode else "OFF"
        unit = bundle["time_label"].replace("Time ", "").replace("(", "").replace(")", "")
        lbl = (
            f"Segment {segment}/{n_seg_total} | {t_lo:.3f} - {t_hi:.3f} {unit}"
            f" | Cells: {len(cells)} | Bad epochs: {len(bad_epochs_clean)} | Bad-span: {mode_txt}"
        )
        if remove_baseline_drift:
            lbl += f" | Baseline: removed ({baseline_window_s:.0f}s median)"
        elif show_baseline:
            lbl += f" | Baseline: shown ({baseline_window_s:.0f}s median)"
        elif show_baseline_requested and hp_enabled:
            lbl += " | Baseline: hidden (high-pass ON)"
        if hp_enabled:
            lbl += f" | High-pass: {hp_cutoff:.1f} Hz"
        lbl += f" | Motion regression: {data.get('motion_info', 'OFF')}"
        if show_motion_fit:
            lbl += " | Fitted noise: shown"
        else:
            lbl += " | Fitted noise: hidden"
        if full_trace_mode:
            lbl += " | Full trace mode: ON"
            if use_resampler:
                lbl += " | Resampler: ON"
            else:
                lbl += " | Resampler: OFF"
        if bool(show_spikes_store):
            lbl += f" | Spikes shown: {spikes_visible}"
        else:
            lbl += " | Spikes shown: OFF"
        if len(cells) == 0:
            lbl += " | No cells selected"
        return fig, lbl, graph_style

    @app.callback(
        Output("save-status", "children"),
        Input("save-btn", "n_clicks"),
        State("save-filename-input", "value"),
        State("bad-epochs-store", "data"),
        State("cell-checklist", "value"),
        State("spikes-store", "data"),
        State("spike-params-store", "data"),
        prevent_initial_call=True,
    )
    def save_annotations(_n_clicks, filename, bad_epochs, selected_cells, spikes_store, params_store):
        try:
            spikes_verified_array = [[] for _ in range(n_cells)]
            if isinstance(spikes_store, (list, tuple)):
                for ci in range(min(n_cells, len(spikes_store))):
                    try:
                        spikes_verified_array[ci] = list(map(int, spikes_store[ci]))
                    except Exception:
                        spikes_verified_array[ci] = []

            params_per_cell = params_store if isinstance(params_store, dict) else {}
            paths = save_annotations_compatible(
                bundle=bundle,
                filename=(filename or "volpy_demix_results_annotated.pickle").strip(),
                bad_epochs=_sanitize_epochs(bad_epochs or []),
                good_cells=list(selected_cells or []),
                spikes_verified_array=spikes_verified_array,
                params_per_cell=params_per_cell,
                baseline_cfg=bundle["initial_baseline_cfg"],
            )
            if not paths:
                return "No session had an on-disk source path to save to."
            return f"Saved {len(paths)} file(s). Example: {paths[0]}"
        except Exception as exc:
            return f"Save failed: {exc}"

    return app


def parse_args():
    parser = argparse.ArgumentParser(description="Dash overview for miniVI demix results.")
    parser.add_argument(
        "paths",
        nargs="*",
        help="One or more volpy_demix_results*.pickle paths.",
    )
    parser.add_argument(
        "--main-folder",
        type=str,
        default="",
        help="Folder containing session subfolders ending with 'good'.",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not auto-open the app URL in a browser.",
    )
    return parser.parse_args()


def _browser_host(host):
    h = str(host).strip().lower()
    if h in {"0.0.0.0", "::", ""}:
        return "127.0.0.1"
    return host


def _open_browser_later(host, port, delay_s=1.0):
    url = f"http://{_browser_host(host)}:{int(port)}"
    timer = threading.Timer(delay_s, lambda: webbrowser.open(url))
    timer.daemon = True
    timer.start()


def main():
    args = parse_args()
    bundle = load_stitched_bundle(paths=args.paths, main_folder=args.main_folder)
    app = create_app(bundle)
    if not bool(args.no_browser):
        _open_browser_later(args.host, args.port, delay_s=1.0)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
