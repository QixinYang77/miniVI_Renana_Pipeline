"""Dash CS Detection Tuning App.

Interactive parameter tuning for complex-spike detection, with step-by-step
diagnostic figures displayed inline.

Usage
-----
    python app.py /path/to/main_data_folder [--port 8051] [--debug]
"""

import argparse
import os
import sys
import threading
import webbrowser

# ---------------------------------------------------------------------------
# Ensure the package root is importable so ``from utils...`` works inside
# cs_pipeline / data_io.
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_THIS_DIR)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

from dash import Dash, Input, Output, State, ctx, dcc, html, no_update  # noqa: E402
from dash.exceptions import PreventUpdate  # noqa: E402
from flask import abort, send_file  # noqa: E402

from cs_pipeline import ROUND1_DEFAULTS, ROUND2_DEFAULTS, get_default_params  # noqa: E402
import cs_pipeline  # noqa: E402
from data_io import load_cs_bundle, save_merged_results  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(v, fallback):
    try:
        return float(v)
    except Exception:
        return float(fallback)


def _safe_int(v, fallback):
    try:
        return int(v)
    except Exception:
        return int(fallback)


def _parse_threshold(text):
    """Parse comma-separated threshold string into float or list of floats."""
    parts = [s.strip() for s in str(text).split(",") if s.strip()]
    vals = []
    for p in parts:
        try:
            vals.append(float(p))
        except ValueError:
            pass
    if len(vals) == 1:
        return vals[0]
    return vals if vals else 0.5


def _parse_optional_float(v, fallback=None):
    """Parse optional float; accepts None/'none'/'null'/'nan' as None."""
    if v is None:
        return fallback
    s = str(v).strip().lower()
    if s in {"", "none", "null", "nan"}:
        return None
    try:
        return float(v)
    except Exception:
        return fallback


# ---------------------------------------------------------------------------
# Styling constants (matching dash_overview_app)
# ---------------------------------------------------------------------------

CARD_STYLE = {
    "border": "1px solid #ddd",
    "padding": "8px",
    "borderRadius": "6px",
    "backgroundColor": "white",
    "marginBottom": "8px",
}

LABEL_STYLE = {"fontSize": "12px", "marginBottom": "2px", "fontWeight": 600}

INPUT_STYLE = {"width": "100%", "fontSize": "12px"}

BTN_STYLE = {"width": "100%", "marginTop": "6px", "fontSize": "12px"}

PDF_IFRAME_STYLE = {
    "width": "100%",
    "height": "980px",
    "border": "1px solid #ddd",
    "borderRadius": "6px",
    "marginBottom": "8px",
}

SECTION_HEADER = {"marginTop": "4px", "marginBottom": "6px", "fontSize": "13px", "fontWeight": 600}


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------

def _param_row(label, input_id, default, input_type="number", step=None):
    """Create a compact label + input row."""
    kwargs = {"id": input_id, "type": input_type, "value": default, "style": INPUT_STYLE}
    if step is not None:
        kwargs["step"] = step
    if input_type == "text":
        kwargs.pop("type")
        return html.Div([
            html.Label(label, style=LABEL_STYLE),
            dcc.Input(**kwargs),
        ], style={"marginBottom": "4px"})
    return html.Div([
        html.Label(label, style=LABEL_STYLE),
        dcc.Input(**kwargs),
    ], style={"marginBottom": "4px"})


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(bundle):
    app = Dash(__name__)
    app.title = "CS Detection Tuning"

    n_cells = bundle["n_cells"]
    frame_rate = bundle["frame_rate"]
    session_start_frames = bundle["session_start_frames"]

    # -- Server-side mutable state (single-user local app) --
    cell_status = {}        # int -> "round1" | "round2" | "processed" | "removed"
    cell_params = {}        # int -> dict of params
    round1_cache = {}       # int -> round1 results dict
    round2_cache = {}       # int -> round2 results dict
    final_pdf_cache = {}    # int -> saved final pdf path
    removed_cells = set()

    @app.server.route("/final-pdf/<int:cell_idx>")
    def serve_final_pdf(cell_idx):
        cell_idx = int(cell_idx)
        pdf_path = final_pdf_cache.get(cell_idx)
        if not pdf_path:
            candidate = os.path.join(
                bundle["main_data_folder"], "SNR_figures", f"cell_{cell_idx}_burst_detection.pdf"
            )
            if os.path.isfile(candidate):
                pdf_path = candidate
                final_pdf_cache[cell_idx] = pdf_path
        if not pdf_path or (not os.path.isfile(pdf_path)):
            abort(404)
        return send_file(pdf_path, mimetype="application/pdf", as_attachment=False)

    # Pre-populate from existing results if available
    existing = bundle.get("existing_results")
    if existing is not None:
        _load_existing(existing, n_cells, cell_status, round2_cache, cell_params, removed_cells)

    # -- Build dropdown options --
    def _cell_options():
        opts = []
        for i in range(n_cells):
            s = cell_status.get(i, "")
            if i in removed_cells:
                label = f"Cell {i}  \u2717"
            elif s == "processed":
                label = f"Cell {i}  \u2713"
            elif s == "round2":
                label = f"Cell {i}  [R2]"
            elif s == "round1":
                label = f"Cell {i}  [R1]"
            else:
                label = f"Cell {i}"
            opts.append({"label": label, "value": i})
        return opts

    def _status_summary():
        n_proc = sum(1 for s in cell_status.values() if s == "processed")
        n_rem = len(removed_cells)
        return f"Processed: {n_proc}/{n_cells}  |  Removed: {n_rem}/{n_cells}"

    # -- Layout --
    app.layout = html.Div(
        style={
            "display": "grid",
            "gridTemplateColumns": "auto 1fr",
            "height": "100vh",
            "fontFamily": "Arial, sans-serif",
            "fontSize": "12px",
            "gap": "8px",
            "padding": "8px",
        },
        children=[
            # ========== LEFT SIDEBAR ==========
            html.Div(
                style={
                    "width": "320px",
                    "minWidth": "260px",
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
                    # Cell selector
                    html.Div(style=CARD_STYLE, children=[
                        html.Div("Cell", style=SECTION_HEADER),
                        dcc.Dropdown(
                            id="cell-dropdown",
                            options=_cell_options(),
                            value=0,
                            clearable=False,
                            style={"fontSize": "12px"},
                        ),
                        html.Div(id="status-summary", children=_status_summary(),
                                 style={"fontSize": "11px", "marginTop": "4px", "color": "#666"}),
                    ]),

                    # Round 1 parameters
                    html.Div(style=CARD_STYLE, children=[
                        html.Div("Round 1 Parameters", style=SECTION_HEADER),
                        _param_row("f_hp_CS (Hz)", "r1-f_hp_CS", ROUND1_DEFAULTS["f_hp_CS"], step=0.5),
                        _param_row("simple_threshold_SS (MAD)", "r1-simple_threshold_SS", ROUND1_DEFAULTS["simple_threshold_SS"], step=0.5),
                        _param_row("f_hp (Hz)", "r1-f_hp", ROUND1_DEFAULTS["f_hp"], step=1),
                        dcc.Checklist(
                            id="r1-separate-by-sessions",
                            options=[{"label": "separate_by_sessions", "value": "on"}],
                            value=["on"] if ROUND1_DEFAULTS.get("separate_by_sessions", False) else [],
                            inputStyle={"marginRight": "6px"},
                            style={"fontSize": "12px", "marginTop": "2px"},
                        ),
                    ]),

                    # Round 2 parameters
                    html.Div(style=CARD_STYLE, children=[
                        html.Div("Round 2 Parameters", style=SECTION_HEADER),
                        html.Div("Detection Thresholds", style={"fontSize": "11px", "fontWeight": 600, "marginTop": "2px", "marginBottom": "4px", "color": "#444"}),
                        _param_row("simple_threshold (MAD)", "r2-simple_threshold", ROUND2_DEFAULTS["simple_threshold"], step=0.5),
                        _param_row("SS_height_cap", "r2-SS_height_cap", ROUND2_DEFAULTS["SS_height_cap"], step=0.05),
                        _param_row("complex_spike_threshold", "r2-complex_spike_threshold",
                                   ", ".join(str(v) for v in ROUND2_DEFAULTS["complex_spike_threshold"]),
                                   input_type="text"),
                        _param_row("cb_amp_threshold", "r2-cb_amp_threshold", ROUND2_DEFAULTS["cb_amp_threshold"], step=0.05),
                        _param_row("cb_duration_threshold (ms)", "r2-cb_duration_threshold", ROUND2_DEFAULTS["cb_duration_threshold"], step=5),
                        _param_row("min_num_spikes", "r2-min_num_spikes", ROUND2_DEFAULTS["min_num_spikes"], step=1),
                        html.Div("Baseline Subtraction", style={"fontSize": "11px", "fontWeight": 600, "marginTop": "6px", "marginBottom": "4px", "color": "#444"}),
                        dcc.Checklist(
                            id="r2-baseline_subtract",
                            options=[{"label": "baseline_subtract", "value": "on"}],
                            value=["on"] if ROUND2_DEFAULTS.get("baseline_subtract", False) else [],
                            inputStyle={"marginRight": "6px"},
                            style={"fontSize": "12px", "marginTop": "2px", "marginBottom": "4px"},
                        ),
                        _param_row("baseline_window_ms", "r2-baseline_window_ms", ROUND2_DEFAULTS["baseline_window_ms"], step=5),
                        _param_row("baseline_percentile (or None)", "r2-baseline_percentile",
                                   str(ROUND2_DEFAULTS["baseline_percentile"]), input_type="text"),
                        html.Div("Burst Windowing", style={"fontSize": "11px", "fontWeight": 600, "marginTop": "6px", "marginBottom": "4px", "color": "#444"}),
                        _param_row("vm_crossing_threshold", "r2-vm_crossing_threshold", ROUND2_DEFAULTS["vm_crossing_threshold"], step=0.05),
                        html.Div("Merge / Grouping", style={"fontSize": "11px", "fontWeight": 600, "marginTop": "6px", "marginBottom": "4px", "color": "#444"}),
                        _param_row("merge_SS_ms", "r2-merge_SS_ms", ROUND2_DEFAULTS["merge_SS_ms"], step=1),
                        _param_row("merge_CB_ms", "r2-merge_CB_ms", ROUND2_DEFAULTS["merge_CB_ms"], step=1),
                        html.Div("Plateau / Filter", style={"fontSize": "11px", "fontWeight": 600, "marginTop": "6px", "marginBottom": "4px", "color": "#444"}),
                        _param_row("plateau_amp_threshold", "r2-plateau_amp_threshold", ROUND2_DEFAULTS["plateau_amp_threshold"], step=0.05),
                        _param_row("plateau_duration_threshold (ms)", "r2-plateau_duration_threshold", ROUND2_DEFAULTS["plateau_duration_threshold"], step=10),
                        _param_row("plateau_kernel_ms", "r2-plateau_kernel_ms", ROUND2_DEFAULTS["plateau_kernel_ms"], step=10),
                        _param_row("plateau_score_min_ms", "r2-plateau_score_min_ms", ROUND2_DEFAULTS["plateau_score_min_ms"], step=10),
                        _param_row("isi_threshold_ms", "r2-isi_threshold_ms", ROUND2_DEFAULTS["isi_threshold_ms"], step=5),
                        _param_row("highpass (Hz)", "r2-highpass", ROUND2_DEFAULTS["highpass"], step=0.5),
                        _param_row("median_window", "r2-median_window", ROUND2_DEFAULTS["median_window"], step=2),
                    ]),

                    # Actions
                    html.Div(style=CARD_STYLE, children=[
                        html.Div("Actions", style=SECTION_HEADER),
                        html.Button("Run Detection", id="run-detect-btn",
                                    style={**BTN_STYLE, "backgroundColor": "#2196F3", "color": "white", "border": "none"}),
                        html.Button("Mark as Removed", id="toggle-removed-btn",
                                    style={**BTN_STYLE, "backgroundColor": "#f44336", "color": "white", "border": "none"}),
                        html.Button("Save All Results", id="save-btn",
                                    style={**BTN_STYLE, "backgroundColor": "#9C27B0", "color": "white", "border": "none"}),
                        html.Div(id="action-status", style={"fontSize": "11px", "marginTop": "6px", "color": "#333"}),
                    ]),
                ],
            ),

            # ========== RIGHT CONTENT ==========
            html.Div(
                style={"overflowY": "auto", "padding": "4px 0"},
                children=[
                    html.Div("Final Detection PDF", style={**SECTION_HEADER, "fontSize": "14px"}),
                    dcc.Loading(
                        id="loading-detect",
                        type="circle",
                        children=html.Div(id="final-pdf-view"),
                    ),
                ],
            ),

            # Hidden store to trigger UI refreshes
            dcc.Store(id="refresh-trigger", data=0),
        ],
    )

    # ==================================================================
    # Callbacks
    # ==================================================================

    # --- Cell selection: populate params, show cached figures, update buttons ---
    @app.callback(
        Output("r1-f_hp_CS", "value"),
        Output("r1-simple_threshold_SS", "value"),
        Output("r1-f_hp", "value"),
        Output("r1-separate-by-sessions", "value"),
        Output("r2-simple_threshold", "value"),
        Output("r2-SS_height_cap", "value"),
        Output("r2-complex_spike_threshold", "value"),
        Output("r2-cb_amp_threshold", "value"),
        Output("r2-cb_duration_threshold", "value"),
        Output("r2-min_num_spikes", "value"),
        Output("r2-baseline_subtract", "value"),
        Output("r2-baseline_window_ms", "value"),
        Output("r2-baseline_percentile", "value"),
        Output("r2-vm_crossing_threshold", "value"),
        Output("r2-merge_SS_ms", "value"),
        Output("r2-merge_CB_ms", "value"),
        Output("r2-plateau_amp_threshold", "value"),
        Output("r2-plateau_duration_threshold", "value"),
        Output("r2-plateau_kernel_ms", "value"),
        Output("r2-plateau_score_min_ms", "value"),
        Output("r2-isi_threshold_ms", "value"),
        Output("r2-highpass", "value"),
        Output("r2-median_window", "value"),
        Output("final-pdf-view", "children"),
        Output("toggle-removed-btn", "children"),
        Input("cell-dropdown", "value"),
        Input("refresh-trigger", "data"),
        prevent_initial_call=False,
    )
    def on_cell_selected(cell_idx, _trigger):
        if cell_idx is None:
            raise PreventUpdate

        defaults = get_default_params()
        p = cell_params.get(cell_idx, defaults)

        # Param values
        r1_fhp_cs = p.get("f_hp_CS", defaults["f_hp_CS"])
        r1_ss_thr = p.get("simple_threshold_SS", defaults["simple_threshold_SS"])
        r1_fhp = p.get("f_hp", defaults["f_hp"])
        r1_sep = p.get("separate_by_sessions", defaults.get("separate_by_sessions", False))
        r1_sep_value = ["on"] if bool(r1_sep) else []
        r2_thr = p.get("simple_threshold", defaults["simple_threshold"])
        r2_sshc = p.get("SS_height_cap", defaults["SS_height_cap"])
        r2_cst = p.get("complex_spike_threshold", defaults["complex_spike_threshold"])
        if isinstance(r2_cst, (list, tuple)):
            r2_cst_str = ", ".join(str(v) for v in r2_cst)
        else:
            r2_cst_str = str(r2_cst)
        r2_cb_amp = p.get("cb_amp_threshold", defaults["cb_amp_threshold"])
        r2_cb_dur = p.get("cb_duration_threshold", defaults["cb_duration_threshold"])
        r2_min_spk = p.get("min_num_spikes", defaults["min_num_spikes"])
        r2_bl_sub = bool(p.get("baseline_subtract", defaults["baseline_subtract"]))
        r2_bl_sub_val = ["on"] if r2_bl_sub else []
        r2_bl_win = p.get("baseline_window_ms", defaults["baseline_window_ms"])
        r2_bl_pct = p.get("baseline_percentile", defaults["baseline_percentile"])
        r2_bl_pct_str = "None" if r2_bl_pct is None else str(r2_bl_pct)
        r2_vm_crossing = p.get("vm_crossing_threshold", defaults["vm_crossing_threshold"])
        r2_merge_ss = p.get("merge_SS_ms", defaults["merge_SS_ms"])
        r2_merge_cb = p.get("merge_CB_ms", defaults["merge_CB_ms"])
        r2_plat_amp = p.get("plateau_amp_threshold", defaults["plateau_amp_threshold"])
        r2_plat_dur = p.get("plateau_duration_threshold", defaults["plateau_duration_threshold"])
        r2_plat_ker = p.get("plateau_kernel_ms", defaults["plateau_kernel_ms"])
        r2_plat_scr = p.get("plateau_score_min_ms", defaults["plateau_score_min_ms"])
        r2_isi = p.get("isi_threshold_ms", defaults["isi_threshold_ms"])
        r2_hp = p.get("highpass", defaults["highpass"])
        r2_mw = p.get("median_window", defaults["median_window"])

        # Cached final PDF
        final_pdf = final_pdf_cache.get(cell_idx)
        if (not final_pdf) or (not os.path.isfile(final_pdf)):
            candidate = os.path.join(bundle["main_data_folder"], "SNR_figures", f"cell_{cell_idx}_burst_detection.pdf")
            if os.path.isfile(candidate):
                final_pdf = candidate
                final_pdf_cache[cell_idx] = candidate
        final_view = _render_pdf_view(cell_idx, final_pdf)

        removed_label = "Unmark Removed" if cell_idx in removed_cells else "Mark as Removed"

        return (
            r1_fhp_cs, r1_ss_thr, r1_fhp, r1_sep_value,
            r2_thr, r2_sshc, r2_cst_str,
            r2_cb_amp, r2_cb_dur,
            r2_min_spk, r2_bl_sub_val, r2_bl_win, r2_bl_pct_str, r2_vm_crossing, r2_merge_ss, r2_merge_cb,
            r2_plat_amp, r2_plat_dur, r2_plat_ker, r2_plat_scr,
            r2_isi, r2_hp, r2_mw,
            final_view, removed_label,
        )

    # --- Run Detection (Round 1 + Round 2 + final PDF) ---
    @app.callback(
        Output("final-pdf-view", "children", allow_duplicate=True),
        Output("cell-dropdown", "options", allow_duplicate=True),
        Output("status-summary", "children", allow_duplicate=True),
        Output("action-status", "children", allow_duplicate=True),
        Input("run-detect-btn", "n_clicks"),
        State("cell-dropdown", "value"),
        State("r1-f_hp_CS", "value"),
        State("r1-simple_threshold_SS", "value"),
        State("r1-f_hp", "value"),
        State("r1-separate-by-sessions", "value"),
        State("r2-simple_threshold", "value"),
        State("r2-SS_height_cap", "value"),
        State("r2-complex_spike_threshold", "value"),
        State("r2-cb_amp_threshold", "value"),
        State("r2-cb_duration_threshold", "value"),
        State("r2-min_num_spikes", "value"),
        State("r2-baseline_subtract", "value"),
        State("r2-baseline_window_ms", "value"),
        State("r2-baseline_percentile", "value"),
        State("r2-vm_crossing_threshold", "value"),
        State("r2-merge_SS_ms", "value"),
        State("r2-merge_CB_ms", "value"),
        State("r2-plateau_amp_threshold", "value"),
        State("r2-plateau_duration_threshold", "value"),
        State("r2-plateau_kernel_ms", "value"),
        State("r2-plateau_score_min_ms", "value"),
        State("r2-isi_threshold_ms", "value"),
        State("r2-highpass", "value"),
        State("r2-median_window", "value"),
        prevent_initial_call=True,
    )
    def on_run_detection(
        n_clicks, cell_idx,
        r1_f_hp_CS, r1_simple_threshold_SS, r1_f_hp, r1_separate_by_sessions_val,
        r2_simple_threshold, r2_SS_height_cap, r2_complex_spike_threshold_str,
        r2_cb_amp, r2_cb_dur, r2_min_num_spikes, r2_baseline_subtract_val,
        r2_baseline_window_ms, r2_baseline_percentile, r2_vm_crossing_threshold, r2_merge_SS_ms, r2_merge_CB_ms,
        r2_plat_amp, r2_plat_dur, r2_plat_kern, r2_plat_score,
        r2_isi, r2_highpass, r2_median_window,
    ):
        if cell_idx is None:
            raise PreventUpdate

        params = get_default_params()
        params["f_hp_CS"] = _safe_float(r1_f_hp_CS, ROUND1_DEFAULTS["f_hp_CS"])
        params["simple_threshold_SS"] = _safe_float(r1_simple_threshold_SS, ROUND1_DEFAULTS["simple_threshold_SS"])
        params["f_hp"] = _safe_float(r1_f_hp, ROUND1_DEFAULTS["f_hp"])
        params["separate_by_sessions"] = bool(r1_separate_by_sessions_val and "on" in r1_separate_by_sessions_val)
        params["simple_threshold"] = _safe_float(r2_simple_threshold, ROUND2_DEFAULTS["simple_threshold"])
        params["SS_height_cap"] = _safe_float(r2_SS_height_cap, ROUND2_DEFAULTS["SS_height_cap"])
        params["complex_spike_threshold"] = _parse_threshold(r2_complex_spike_threshold_str)
        params["cb_amp_threshold"] = _safe_float(r2_cb_amp, ROUND2_DEFAULTS["cb_amp_threshold"])
        params["cb_duration_threshold"] = _safe_int(r2_cb_dur, ROUND2_DEFAULTS["cb_duration_threshold"])
        params["min_num_spikes"] = _safe_int(r2_min_num_spikes, ROUND2_DEFAULTS["min_num_spikes"])
        params["baseline_subtract"] = bool(r2_baseline_subtract_val and "on" in r2_baseline_subtract_val)
        params["baseline_window_ms"] = _safe_int(r2_baseline_window_ms, ROUND2_DEFAULTS["baseline_window_ms"])
        params["baseline_percentile"] = _parse_optional_float(
            r2_baseline_percentile, ROUND2_DEFAULTS["baseline_percentile"]
        )
        params["vm_crossing_threshold"] = _safe_float(
            r2_vm_crossing_threshold, ROUND2_DEFAULTS["vm_crossing_threshold"]
        )
        params["merge_SS_ms"] = _safe_int(r2_merge_SS_ms, ROUND2_DEFAULTS["merge_SS_ms"])
        params["merge_CB_ms"] = _safe_int(r2_merge_CB_ms, ROUND2_DEFAULTS["merge_CB_ms"])
        params["plateau_amp_threshold"] = _safe_float(r2_plat_amp, ROUND2_DEFAULTS["plateau_amp_threshold"])
        params["plateau_duration_threshold"] = _safe_int(r2_plat_dur, ROUND2_DEFAULTS["plateau_duration_threshold"])
        params["plateau_kernel_ms"] = _safe_int(r2_plat_kern, ROUND2_DEFAULTS["plateau_kernel_ms"])
        params["plateau_score_min_ms"] = _safe_int(r2_plat_score, ROUND2_DEFAULTS["plateau_score_min_ms"])
        params["isi_threshold_ms"] = _safe_int(r2_isi, ROUND2_DEFAULTS["isi_threshold_ms"])
        params["highpass"] = _safe_float(r2_highpass, ROUND2_DEFAULTS["highpass"])
        params["median_window"] = _safe_int(r2_median_window, ROUND2_DEFAULTS["median_window"])

        trace_raw = bundle["traces"][cell_idx, :].copy()
        spike_idx = bundle["spikes_volpy"][cell_idx].copy()

        r1_results, _ = cs_pipeline.run_round1(
            trace_raw, spike_idx, frame_rate, session_start_frames, params,
            cell_idx=cell_idx, output_folder=bundle["main_data_folder"],
        )
        r2_results, _ = cs_pipeline.run_round2(
            r1_results, frame_rate, session_start_frames, params,
            cell_idx=cell_idx, output_folder=bundle["main_data_folder"],
        )
        final_pdf_path = cs_pipeline.generate_pdf(
            r1_results, r2_results, frame_rate, session_start_frames, cell_idx, bundle["main_data_folder"]
        )

        round1_cache[cell_idx] = r1_results
        round2_cache[cell_idx] = r2_results
        final_pdf_cache[cell_idx] = final_pdf_path
        cell_params[cell_idx] = params
        cell_status[cell_idx] = "processed"
        removed_cells.discard(cell_idx)

        return (
            _render_pdf_view(cell_idx, final_pdf_path),
            _cell_options(),
            _status_summary(),
            f"Detection complete for Cell {cell_idx}. Final PDF saved: {final_pdf_path}",
        )

    # --- Toggle removed ---
    @app.callback(
        Output("cell-dropdown", "options", allow_duplicate=True),
        Output("status-summary", "children", allow_duplicate=True),
        Output("toggle-removed-btn", "children", allow_duplicate=True),
        Output("action-status", "children", allow_duplicate=True),
        Output("final-pdf-view", "children", allow_duplicate=True),
        Input("toggle-removed-btn", "n_clicks"),
        State("cell-dropdown", "value"),
        prevent_initial_call=True,
    )
    def on_toggle_removed(n_clicks, cell_idx):
        if cell_idx is None:
            raise PreventUpdate

        if cell_idx in removed_cells:
            removed_cells.discard(cell_idx)
            cell_status.pop(cell_idx, None)
            btn_label = "Mark as Removed"
            msg = f"Cell {cell_idx} restored."
        else:
            removed_cells.add(cell_idx)
            cell_status[cell_idx] = "removed"
            round1_cache.pop(cell_idx, None)
            round2_cache.pop(cell_idx, None)
            final_pdf_cache.pop(cell_idx, None)
            btn_label = "Unmark Removed"
            msg = f"Cell {cell_idx} marked as removed."

        return (
            _cell_options(),
            _status_summary(),
            btn_label,
            msg,
            [],
        )

    # --- Save all ---
    @app.callback(
        Output("action-status", "children", allow_duplicate=True),
        Input("save-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def on_save(n_clicks):
        processed = {
            idx: res for idx, res in round2_cache.items()
            if idx not in removed_cells
        }
        n_expected = n_cells - len(removed_cells)
        n_done = len(processed)

        if n_done == 0:
            return "Nothing to save — run detection for at least one cell first."

        save_path = save_merged_results(bundle, processed, cell_params, removed_cells)

        if n_done < n_expected:
            missing = [i for i in range(n_cells) if i not in removed_cells and i not in processed]
            return f"Saved {n_done}/{n_expected} cells to {save_path}. Missing: {missing}"
        return f"Saved all {n_done} cells to {save_path}"

    return app


# ---------------------------------------------------------------------------
# Helpers used by callbacks
# ---------------------------------------------------------------------------

def _render_pdf_view(cell_idx, pdf_path):
    """Render a saved final PDF for a cell."""
    if not pdf_path or (not os.path.isfile(pdf_path)):
        return [html.Div("No results yet.", style={"color": "#999", "fontStyle": "italic"})]
    mtime = int(os.path.getmtime(pdf_path))
    pdf_url = f"/final-pdf/{int(cell_idx)}?v={mtime}"
    return [
        html.Div(f"Saved PDF: {pdf_path}", style={"fontSize": "11px", "color": "#555", "marginBottom": "6px"}),
        html.Iframe(src=pdf_url, style=PDF_IFRAME_STYLE),
    ]


def _load_existing(existing, n_cells, cell_status, round2_cache, cell_params, removed_cells):
    """Load pre-existing results from a previously saved merged_neural_CS.pkl."""
    # Reconstruct per-cell results
    for idx in range(n_cells):
        cb = (existing.get("complex_bursts_dicts") or [None] * n_cells)[idx]
        if cb is None:
            continue  # cell was not processed
        round2_cache[idx] = {
            "complex_bursts_dict_vm": cb,
            "refined_SS": (existing.get("refined_SS") or [None] * n_cells)[idx],
            "all_CS_spikes": (existing.get("all_CS_spikes") or [None] * n_cells)[idx],
            "all_spikes": (existing.get("all_spikes") or [None] * n_cells)[idx],
            "spike_heights_interpolated": (existing.get("spike_heights_interpolated") or [None] * n_cells)[idx],
            "SNR_interpolated": (existing.get("SNR_interpolated") or [None] * n_cells)[idx],
            "trace_SNR_interpolated": (existing.get("traces_SNR_interpolated") or [None] * n_cells)[idx],
            "Vm": (existing.get("Vm_SNR_interpolated") or [None] * n_cells)[idx],
            "burst_metrics": (existing.get("burst_metrics") or [None] * n_cells)[idx],
            "plateaus_dict": (existing.get("plateaus_dicts") or [None] * n_cells)[idx],
        }
        cell_status[idx] = "processed"

    # Load per-cell params
    saved_params = existing.get("params_by_cell") or {}
    for idx, p in saved_params.items():
        cell_params[int(idx)] = p

    # Load removed cells
    saved_removed = existing.get("removed_cells") or []
    for idx in saved_removed:
        removed_cells.add(int(idx))
        cell_status[int(idx)] = "removed"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

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


def run_from_notebook(main_data_folder, traces, traces_raw, spikes, spikes_volpy,
                      frame_rate, session_start_frames, subfolders,
                      weights_all, ROIs_all, mean_images, mean_images_raw,
                      port=8051, **kwargs):
    """Launch the app from a Jupyter notebook using already-loaded data.

    Example::

        from dash_cs_detection_app.app import run_from_notebook
        run_from_notebook(
            main_data_folder, traces, traces_raw, spikes, spikes_volpy,
            frame_rate, session_start_frames, subfolders,
            weights_all, ROIs_all, mean_images, mean_images_raw,
        )
    """
    bundle = {
        "main_data_folder": main_data_folder,
        "subfolders": subfolders,
        "traces": traces,
        "traces_raw": traces_raw,
        "spikes": spikes,
        "spikes_volpy": spikes_volpy,
        "frame_rate": frame_rate,
        "session_start_frames": session_start_frames,
        "weights_all": weights_all,
        "ROIs_all": ROIs_all,
        "mean_images": mean_images,
        "mean_images_raw": mean_images_raw,
        "n_cells": traces.shape[0],
        "existing_results": None,
    }
    # Check for existing results
    existing_pkl = os.path.join(main_data_folder, "merged_neural_CS.pkl")
    if os.path.isfile(existing_pkl):
        import pickle
        with open(existing_pkl, "rb") as f:
            bundle["existing_results"] = pickle.load(f)

    app = create_app(bundle)
    _open_browser_later("127.0.0.1", port)
    app.run(host="127.0.0.1", port=port, debug=False)


def main():
    parser = argparse.ArgumentParser(description="Dash CS Detection Tuning App")
    parser.add_argument("main_data_folder", help="Path to main data folder containing -good subfolders")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8051)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    bundle = load_cs_bundle(args.main_data_folder)
    app = create_app(bundle)

    if not args.no_browser:
        _open_browser_later(args.host, args.port)

    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
