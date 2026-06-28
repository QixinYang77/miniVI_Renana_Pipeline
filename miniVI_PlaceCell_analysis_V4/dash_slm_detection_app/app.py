"""Dash GUI for tuning SLM spike-detection parameters."""

from __future__ import annotations

import argparse
import os
import sys
import threading
import webbrowser
from pathlib import Path

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_THIS_DIR)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

from dash import Dash, Input, Output, State, dcc, html  # noqa: E402
from dash.exceptions import PreventUpdate  # noqa: E402
from flask import abort, send_file  # noqa: E402

from data_io import load_bundle_pickle, load_saved_results, save_slm_results  # noqa: E402
from pipeline import get_default_params, run_detection  # noqa: E402


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


def _parse_optional_int(v, fallback=None):
    if v is None:
        return fallback
    s = str(v).strip().lower()
    if s in {"", "none", "null", "nan"}:
        return None
    try:
        return int(v)
    except Exception:
        return fallback


def _parse_optional_float(v, fallback=None):
    if v is None:
        return fallback
    s = str(v).strip().lower()
    if s in {"", "none", "null", "nan"}:
        return None
    try:
        return float(v)
    except Exception:
        return fallback


def _parse_threshold(text):
    parts = [s.strip() for s in str(text).split(",") if s.strip()]
    vals = []
    for part in parts:
        try:
            vals.append(float(part))
        except ValueError:
            pass
    if len(vals) == 1:
        return vals[0]
    return vals if vals else [0.7, 0.6]


def _param_row(label, input_id, default, input_type="number", step=None):
    kwargs = {"id": input_id, "value": default, "style": INPUT_STYLE}
    if input_type != "text":
        kwargs["type"] = input_type
    if step is not None:
        kwargs["step"] = step
    return html.Div(
        [
            html.Label(label, style=LABEL_STYLE),
            dcc.Input(**kwargs),
        ],
        style={"marginBottom": "4px"},
    )


def _render_pdf_view(flat_index, pdf_path):
    if not pdf_path or (not os.path.isfile(pdf_path)):
        return [html.Div("No results yet.", style={"color": "#999", "fontStyle": "italic"})]
    mtime = int(os.path.getmtime(pdf_path))
    pdf_url = f"/final-pdf/{int(flat_index)}?v={mtime}"
    return [
        html.Div(f"Saved PDF: {pdf_path}", style={"fontSize": "11px", "color": "#555", "marginBottom": "6px"}),
        html.Iframe(src=pdf_url, style=PDF_IFRAME_STYLE),
    ]


def create_app(bundle, *, results_path, pdf_output_dir):
    app = Dash(__name__)
    app.title = "SLM Spike Detection Tuning"

    ordered_cells = bundle["ordered_cells"]
    trace_lookup = bundle["trace_lookup"]
    n_cells = bundle["n_cells_total"]
    sampling_rate_hz = bundle["sampling_rate_hz"]

    results_by_cell_key = {}
    params_by_cell_key = {}
    removed_cell_keys = set()
    pdf_cache = {}

    existing = load_saved_results(results_path)
    if existing is not None:
        results_by_cell_key.update(existing.get("results_by_cell_key") or {})
        params_by_cell_key.update(existing.get("params_by_cell_key") or {})
        removed_cell_keys.update(existing.get("removed_cell_keys") or [])
        for cell_key, result in results_by_cell_key.items():
            pdf_path = result.get("diagnostic_pdf_path")
            if pdf_path:
                pdf_cache[cell_key] = pdf_path

    def _cell_meta(flat_index):
        return ordered_cells[int(flat_index)]

    def _cell_options():
        options = []
        for flat_index, meta in enumerate(ordered_cells):
            cell_key = meta["cell_key"]
            if cell_key in removed_cell_keys:
                suffix = "  \u2717"
            elif cell_key in results_by_cell_key:
                suffix = "  \u2713"
            else:
                suffix = ""
            label = f"{meta['condition']} cell {meta['condition_cell_index'] + 1}: {meta['folder']}{suffix}"
            options.append({"label": label, "value": flat_index})
        return options

    def _status_summary():
        processed = sum(
            1 for meta in ordered_cells
            if meta["cell_key"] in results_by_cell_key and meta["cell_key"] not in removed_cell_keys
        )
        removed = len(removed_cell_keys)
        return f"Processed: {processed}/{n_cells}  |  Removed: {removed}/{n_cells}"

    @app.server.route("/final-pdf/<int:flat_index>")
    def serve_final_pdf(flat_index):
        try:
            meta = _cell_meta(flat_index)
        except Exception:
            abort(404)
        cell_key = meta["cell_key"]
        pdf_path = pdf_cache.get(cell_key)
        if not pdf_path:
            result = results_by_cell_key.get(cell_key) or {}
            pdf_path = result.get("diagnostic_pdf_path")
        if not pdf_path or (not os.path.isfile(pdf_path)):
            abort(404)
        return send_file(pdf_path, mimetype="application/pdf", as_attachment=False)

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
            html.Div(
                style={
                    "width": "360px",
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
                    html.Div(
                        style=CARD_STYLE,
                        children=[
                            html.Div("Cell", style=SECTION_HEADER),
                            dcc.Dropdown(
                                id="cell-dropdown",
                                options=_cell_options(),
                                value=0 if ordered_cells else None,
                                clearable=False,
                                style={"fontSize": "12px"},
                            ),
                            html.Div(
                                id="status-summary",
                                children=_status_summary(),
                                style={"fontSize": "11px", "marginTop": "4px", "color": "#666"},
                            ),
                        ],
                    ),
                    html.Div(
                        style=CARD_STYLE,
                        children=[
                            html.Div("Detection Parameters", style=SECTION_HEADER),
                            _param_row("baseline_window_s", "baseline-window-s", 10.0, step=0.5),
                            _param_row("pnorm_CS", "pnorm-cs", 0.5, step=0.05),
                            _param_row("process_window_CS", "process-window-cs", 300000, step=1000),
                            _param_row("simple_threshold_round1_CB", "simple-threshold-round1-cb", 50.0, step=0.5),
                            _param_row("simple_threshold_round2_CB", "simple-threshold-round2-cb", 4.0, step=0.1),
                            _param_row("pnorm_SS", "pnorm-ss", 0.5, step=0.05),
                            _param_row("process_window_SS", "process-window-ss", 300000, step=1000),
                            _param_row("simple_threshold_SS", "simple-threshold-ss", 5.0, step=0.1),
                            _param_row("f_hp", "f-hp", 20.0, step=0.5),
                            _param_row("f_hp_CS", "f-hp-cs", 1.0, step=0.5),
                            _param_row("SS_height_cap", "ss-height-cap", 0.7, step=0.05),
                            _param_row("complex_spike_threshold", "complex-spike-threshold", "0.7, 0.6", input_type="text"),
                            _param_row("highpass", "highpass", 2.0, step=0.5),
                            _param_row("median_window", "median-window", 11, step=2),
                            _param_row("cb_amp_threshold", "cb-amp-threshold", 0.4, step=0.05),
                            _param_row("cb_duration_threshold", "cb-duration-threshold", 20, step=1),
                            _param_row("min_num_spikes", "min-num-spikes", 2, step=1),
                            _param_row("plateau_amp_threshold", "plateau-amp-threshold", 0.8, step=0.05),
                            _param_row("plateau_duration_threshold", "plateau-duration-threshold", 100, step=5),
                            _param_row("plateau_kernel_ms", "plateau-kernel-ms", 100, step=5),
                            _param_row("plateau_score_min_ms", "plateau-score-min-ms", 80, step=5),
                            _param_row("isi_threshold_ms", "isi-threshold-ms", 20, step=1),
                            _param_row("baseline_subtract", "baseline-subtract", "False", input_type="text"),
                            _param_row("baseline_window_ms", "baseline-window-ms", 20, step=1),
                            _param_row("baseline_percentile", "baseline-percentile", "None", input_type="text"),
                            _param_row("vm_crossing_threshold", "vm-crossing-threshold", 0.1, step=0.05),
                            _param_row("merge_SS_ms", "merge-ss-ms", "None", input_type="text"),
                            _param_row("merge_CB_ms", "merge-cb-ms", "None", input_type="text"),
                        ],
                    ),
                    html.Div(
                        style=CARD_STYLE,
                        children=[
                            html.Div("Actions", style=SECTION_HEADER),
                            html.Button(
                                "Run Detection",
                                id="run-detect-btn",
                                style={**BTN_STYLE, "backgroundColor": "#2196F3", "color": "white", "border": "none"},
                            ),
                            html.Button(
                                "Mark as Removed",
                                id="toggle-removed-btn",
                                style={**BTN_STYLE, "backgroundColor": "#f44336", "color": "white", "border": "none"},
                            ),
                            html.Button(
                                "Save All Results",
                                id="save-btn",
                                style={**BTN_STYLE, "backgroundColor": "#9C27B0", "color": "white", "border": "none"},
                            ),
                            html.Div(id="action-status", style={"fontSize": "11px", "marginTop": "6px", "color": "#333"}),
                            html.Div(
                                f"Sampling rate: {sampling_rate_hz} Hz",
                                style={"fontSize": "11px", "marginTop": "6px", "color": "#666"},
                            ),
                        ],
                    ),
                ],
            ),
            html.Div(
                style={"overflowY": "auto", "padding": "4px 0"},
                children=[
                    html.Div("Final Detection PDF", style={**SECTION_HEADER, "fontSize": "14px"}),
                    dcc.Loading(id="loading-detect", type="circle", children=html.Div(id="final-pdf-view")),
                ],
            ),
        ],
    )

    @app.callback(
        Output("baseline-window-s", "value"),
        Output("pnorm-cs", "value"),
        Output("process-window-cs", "value"),
        Output("simple-threshold-round1-cb", "value"),
        Output("simple-threshold-round2-cb", "value"),
        Output("pnorm-ss", "value"),
        Output("process-window-ss", "value"),
        Output("simple-threshold-ss", "value"),
        Output("f-hp", "value"),
        Output("f-hp-cs", "value"),
        Output("ss-height-cap", "value"),
        Output("complex-spike-threshold", "value"),
        Output("highpass", "value"),
        Output("median-window", "value"),
        Output("cb-amp-threshold", "value"),
        Output("cb-duration-threshold", "value"),
        Output("min-num-spikes", "value"),
        Output("plateau-amp-threshold", "value"),
        Output("plateau-duration-threshold", "value"),
        Output("plateau-kernel-ms", "value"),
        Output("plateau-score-min-ms", "value"),
        Output("isi-threshold-ms", "value"),
        Output("baseline-subtract", "value"),
        Output("baseline-window-ms", "value"),
        Output("baseline-percentile", "value"),
        Output("vm-crossing-threshold", "value"),
        Output("merge-ss-ms", "value"),
        Output("merge-cb-ms", "value"),
        Output("final-pdf-view", "children"),
        Output("toggle-removed-btn", "children"),
        Input("cell-dropdown", "value"),
        prevent_initial_call=False,
    )
    def on_cell_selected(flat_index):
        if flat_index is None:
            raise PreventUpdate

        meta = _cell_meta(flat_index)
        cell_key = meta["cell_key"]
        defaults = get_default_params()
        params = params_by_cell_key.get(cell_key, defaults)
        thresholds = params.get("complex_spike_threshold", defaults["complex_spike_threshold"])
        if isinstance(thresholds, (list, tuple)):
            thresholds_str = ", ".join(str(v) for v in thresholds)
        else:
            thresholds_str = str(thresholds)

        pdf_path = pdf_cache.get(cell_key)
        if not pdf_path:
            pdf_path = (results_by_cell_key.get(cell_key) or {}).get("diagnostic_pdf_path")
        removed_label = "Unmark Removed" if cell_key in removed_cell_keys else "Mark as Removed"

        return (
            params.get("baseline_window_s", defaults["baseline_window_s"]),
            params.get("pnorm_CS", defaults["pnorm_CS"]),
            params.get("process_window_CS", defaults["process_window_CS"]),
            params.get("simple_threshold_round1_CB", defaults["simple_threshold_round1_CB"]),
            params.get("simple_threshold_round2_CB", defaults["simple_threshold_round2_CB"]),
            params.get("pnorm_SS", defaults["pnorm_SS"]),
            params.get("process_window_SS", defaults["process_window_SS"]),
            params.get("simple_threshold_SS", defaults["simple_threshold_SS"]),
            params.get("f_hp", defaults["f_hp"]),
            params.get("f_hp_CS", defaults["f_hp_CS"]),
            params.get("SS_height_cap", defaults["SS_height_cap"]),
            thresholds_str,
            params.get("highpass", defaults["highpass"]),
            params.get("median_window", defaults["median_window"]),
            params.get("cb_amp_threshold", defaults["cb_amp_threshold"]),
            params.get("cb_duration_threshold", defaults["cb_duration_threshold"]),
            params.get("min_num_spikes", defaults["min_num_spikes"]),
            params.get("plateau_amp_threshold", defaults["plateau_amp_threshold"]),
            params.get("plateau_duration_threshold", defaults["plateau_duration_threshold"]),
            params.get("plateau_kernel_ms", defaults["plateau_kernel_ms"]),
            params.get("plateau_score_min_ms", defaults["plateau_score_min_ms"]),
            params.get("isi_threshold_ms", defaults["isi_threshold_ms"]),
            str(params.get("baseline_subtract", defaults["baseline_subtract"])),
            params.get("baseline_window_ms", defaults["baseline_window_ms"]),
            "None" if params.get("baseline_percentile", defaults["baseline_percentile"]) is None else str(params.get("baseline_percentile", defaults["baseline_percentile"])),
            params.get("vm_crossing_threshold", defaults["vm_crossing_threshold"]),
            "None" if params.get("merge_SS_ms", defaults["merge_SS_ms"]) is None else str(params.get("merge_SS_ms", defaults["merge_SS_ms"])),
            "None" if params.get("merge_CB_ms", defaults["merge_CB_ms"]) is None else str(params.get("merge_CB_ms", defaults["merge_CB_ms"])),
            _render_pdf_view(flat_index, pdf_path),
            removed_label,
        )

    @app.callback(
        Output("final-pdf-view", "children", allow_duplicate=True),
        Output("cell-dropdown", "options", allow_duplicate=True),
        Output("status-summary", "children", allow_duplicate=True),
        Output("action-status", "children", allow_duplicate=True),
        Input("run-detect-btn", "n_clicks"),
        State("cell-dropdown", "value"),
        State("baseline-window-s", "value"),
        State("pnorm-cs", "value"),
        State("process-window-cs", "value"),
        State("simple-threshold-round1-cb", "value"),
        State("simple-threshold-round2-cb", "value"),
        State("pnorm-ss", "value"),
        State("process-window-ss", "value"),
        State("simple-threshold-ss", "value"),
        State("f-hp", "value"),
        State("f-hp-cs", "value"),
        State("ss-height-cap", "value"),
        State("complex-spike-threshold", "value"),
        State("highpass", "value"),
        State("median-window", "value"),
        State("cb-amp-threshold", "value"),
        State("cb-duration-threshold", "value"),
        State("min-num-spikes", "value"),
        State("plateau-amp-threshold", "value"),
        State("plateau-duration-threshold", "value"),
        State("plateau-kernel-ms", "value"),
        State("plateau-score-min-ms", "value"),
        State("isi-threshold-ms", "value"),
        State("baseline-subtract", "value"),
        State("baseline-window-ms", "value"),
        State("baseline-percentile", "value"),
        State("vm-crossing-threshold", "value"),
        State("merge-ss-ms", "value"),
        State("merge-cb-ms", "value"),
        prevent_initial_call=True,
    )
    def on_run_detection(
        _n_clicks,
        flat_index,
        baseline_window_s,
        pnorm_cs,
        process_window_cs,
        simple_threshold_round1_cb,
        simple_threshold_round2_cb,
        pnorm_ss,
        process_window_ss,
        simple_threshold_ss,
        f_hp,
        f_hp_cs,
        ss_height_cap,
        complex_spike_threshold,
        highpass,
        median_window,
        cb_amp_threshold,
        cb_duration_threshold,
        min_num_spikes,
        plateau_amp_threshold,
        plateau_duration_threshold,
        plateau_kernel_ms,
        plateau_score_min_ms,
        isi_threshold_ms,
        baseline_subtract,
        baseline_window_ms,
        baseline_percentile,
        vm_crossing_threshold,
        merge_ss_ms,
        merge_cb_ms,
    ):
        if flat_index is None:
            raise PreventUpdate

        meta = _cell_meta(flat_index)
        cell_key = meta["cell_key"]
        defaults = get_default_params()
        params = get_default_params()
        params["baseline_window_s"] = _safe_float(baseline_window_s, defaults["baseline_window_s"])
        params["pnorm_CS"] = _safe_float(pnorm_cs, defaults["pnorm_CS"])
        params["process_window_CS"] = _safe_int(process_window_cs, defaults["process_window_CS"])
        params["simple_threshold_round1_CB"] = _safe_float(simple_threshold_round1_cb, defaults["simple_threshold_round1_CB"])
        params["simple_threshold_round2_CB"] = _safe_float(simple_threshold_round2_cb, defaults["simple_threshold_round2_CB"])
        params["pnorm_SS"] = _safe_float(pnorm_ss, defaults["pnorm_SS"])
        params["process_window_SS"] = _safe_int(process_window_ss, defaults["process_window_SS"])
        params["simple_threshold_SS"] = _safe_float(simple_threshold_ss, defaults["simple_threshold_SS"])
        params["f_hp"] = _safe_float(f_hp, defaults["f_hp"])
        params["f_hp_CS"] = _safe_float(f_hp_cs, defaults["f_hp_CS"])
        params["SS_height_cap"] = _safe_float(ss_height_cap, defaults["SS_height_cap"])
        params["complex_spike_threshold"] = _parse_threshold(complex_spike_threshold)
        params["highpass"] = _safe_float(highpass, defaults["highpass"])
        params["median_window"] = _safe_int(median_window, defaults["median_window"])
        params["cb_amp_threshold"] = _safe_float(cb_amp_threshold, defaults["cb_amp_threshold"])
        params["cb_duration_threshold"] = _safe_int(cb_duration_threshold, defaults["cb_duration_threshold"])
        params["min_num_spikes"] = _safe_int(min_num_spikes, defaults["min_num_spikes"])
        params["plateau_amp_threshold"] = _safe_float(plateau_amp_threshold, defaults["plateau_amp_threshold"])
        params["plateau_duration_threshold"] = _safe_int(plateau_duration_threshold, defaults["plateau_duration_threshold"])
        params["plateau_kernel_ms"] = _safe_int(plateau_kernel_ms, defaults["plateau_kernel_ms"])
        params["plateau_score_min_ms"] = _safe_int(plateau_score_min_ms, defaults["plateau_score_min_ms"])
        params["isi_threshold_ms"] = _safe_int(isi_threshold_ms, defaults["isi_threshold_ms"])
        params["baseline_subtract"] = str(baseline_subtract).strip().lower() == "true"
        params["baseline_window_ms"] = _safe_int(baseline_window_ms, defaults["baseline_window_ms"])
        params["baseline_percentile"] = _parse_optional_float(baseline_percentile, defaults["baseline_percentile"])
        params["vm_crossing_threshold"] = _safe_float(vm_crossing_threshold, defaults["vm_crossing_threshold"])
        params["merge_SS_ms"] = _parse_optional_int(merge_ss_ms, defaults["merge_SS_ms"])
        params["merge_CB_ms"] = _parse_optional_int(merge_cb_ms, defaults["merge_CB_ms"])

        trace = trace_lookup[cell_key]
        result = run_detection(
            trace,
            sampling_rate_hz,
            params,
            cell_key=cell_key,
            pdf_output_dir=pdf_output_dir,
        )
        results_by_cell_key[cell_key] = result
        params_by_cell_key[cell_key] = params
        removed_cell_keys.discard(cell_key)
        if result.get("diagnostic_pdf_path"):
            pdf_cache[cell_key] = result["diagnostic_pdf_path"]

        return (
            _render_pdf_view(flat_index, result.get("diagnostic_pdf_path")),
            _cell_options(),
            _status_summary(),
            f"Detection complete for {meta['condition']} cell {meta['condition_cell_index'] + 1}: {meta['folder']}",
        )

    @app.callback(
        Output("cell-dropdown", "options", allow_duplicate=True),
        Output("status-summary", "children", allow_duplicate=True),
        Output("toggle-removed-btn", "children", allow_duplicate=True),
        Output("action-status", "children", allow_duplicate=True),
        Input("toggle-removed-btn", "n_clicks"),
        State("cell-dropdown", "value"),
        prevent_initial_call=True,
    )
    def on_toggle_removed(_n_clicks, flat_index):
        if flat_index is None:
            raise PreventUpdate
        meta = _cell_meta(flat_index)
        cell_key = meta["cell_key"]
        if cell_key in removed_cell_keys:
            removed_cell_keys.discard(cell_key)
            msg = f"Restored {meta['folder']}."
            label = "Mark as Removed"
        else:
            removed_cell_keys.add(cell_key)
            msg = f"Marked {meta['folder']} as removed."
            label = "Unmark Removed"
        return _cell_options(), _status_summary(), label, msg

    @app.callback(
        Output("action-status", "children", allow_duplicate=True),
        Input("save-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def on_save(_n_clicks):
        save_path = save_slm_results(
            bundle,
            results_by_cell_key,
            params_by_cell_key,
            removed_cell_keys,
            results_path,
        )
        missing = [
            meta["folder"]
            for meta in ordered_cells
            if meta["cell_key"] not in removed_cell_keys and meta["cell_key"] not in results_by_cell_key
        ]
        if missing:
            return f"Saved results to {save_path}. Missing unprocessed cells: {missing}"
        return f"Saved all SLM GUI results to {save_path}"

    return app


def _browser_host(host):
    host = str(host).strip().lower()
    if host in {"0.0.0.0", "::", ""}:
        return "127.0.0.1"
    return host


def _open_browser_later(host, port, delay_s=1.0):
    url = f"http://{_browser_host(host)}:{int(port)}"
    timer = threading.Timer(delay_s, lambda: webbrowser.open(url))
    timer.daemon = True
    timer.start()


def main():
    parser = argparse.ArgumentParser(description="Dash SLM spike-detection tuning app")
    parser.add_argument("bundle_pickle", help="Path to the preloaded SLM bundle pickle")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8052)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--results-path", type=str, default="")
    args = parser.parse_args()

    bundle_path = Path(args.bundle_pickle).expanduser().resolve()
    bundle = load_bundle_pickle(bundle_path)

    if args.results_path:
        results_path = Path(args.results_path).expanduser().resolve()
    else:
        results_path = bundle_path.with_name("SLM_spike_detection_results.pkl")
    pdf_output_dir = results_path.parent / "SLM_SNR_figures"

    app = create_app(
        bundle,
        results_path=str(results_path),
        pdf_output_dir=str(pdf_output_dir),
    )

    if not args.no_browser:
        _open_browser_later(args.host, args.port)

    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
