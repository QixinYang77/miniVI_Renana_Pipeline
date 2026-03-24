"""CS detection pipeline wrappers for the Dash app.

Wraps the detection functions from ``spike_detection`` into two clean
round functions plus PDF generation for diagnostics.
"""

import os
import subprocess
import sys
from contextlib import nullcontext

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import median_filter

# Add utils directory directly to sys.path (no __init__.py in utils/)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_UTILS_DIR = os.path.join(os.path.dirname(_THIS_DIR), "utils")
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
if _UTILS_DIR not in sys.path:
    sys.path.insert(0, _UTILS_DIR)

from fig_capture import close_all, pipeline_context  # noqa: E402

from spike_detection import (  # noqa: E402
    complex_bursts_detection,
    detect_bursts_from_vm,
    detect_complex_spikes,
    interpolate_nan_segment,
    plot_burst_metrics_pdf,
    plot_trace_with_bursts_pdf,
    refine_all_spikes,
    refine_single_spikes,
    spike_height_calculation2,
)

from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402


# ---------------------------------------------------------------------------
# Default parameters (matching the notebook batch cell)
# ---------------------------------------------------------------------------

ROUND1_DEFAULTS = {
    "pnorm_CS": 0.25,
    "process_window_CS": 300000,
    "f_hp_CS": 1.0,
    "pnorm_SS": 0.25,
    "process_window_SS": 300000,
    "simple_threshold_SS": 6.0,
    "f_hp": 20.0,
    "separate_by_sessions": False,
}

ROUND2_DEFAULTS = {
    "simple_threshold": 6.0,
    "SS_height_cap": 0.6,
    "complex_spike_threshold": [0.8, 0.6],
    "highpass": 1.0,
    "median_window": 11,
    "cb_amp_threshold": 0.6,
    "cb_duration_threshold": 20,
    "min_num_spikes": 2,
    "plateau_amp_threshold": 0.8,
    "plateau_duration_threshold": 100,
    "plateau_kernel_ms": 100,
    "plateau_score_min_ms": 80,
    "isi_threshold_ms": 20,
    "baseline_subtract": False,
    "baseline_window_ms": 20,
    "baseline_percentile": 10,
    "vm_crossing_threshold": 0.1,
    "merge_SS_ms": 20,
    "merge_CB_ms": 5,
}


def get_default_params():
    """Return a merged dict of all default parameters."""
    d = {}
    d.update(ROUND1_DEFAULTS)
    d.update(ROUND2_DEFAULTS)
    return d


def _append_open_figures_to_pdf(pdf):
    """Append all currently open Matplotlib figures to a PdfPages object."""
    for fig_num in plt.get_fignums():
        fig = plt.figure(fig_num)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Round 1
# ---------------------------------------------------------------------------

def run_round1(trace_raw, spike_idx, frame_rate, session_start_frames, params, *, cell_idx=None, output_folder=None):
    """Run Round 1: baseline subtraction + rough spike height estimation.

    Returns
    -------
    results : dict
        Intermediate results needed by Round 2.
    figures : list[dict]
        Empty list (Round 1 is delivered as a saved PDF for display).
    """
    trace_idx = trace_raw.copy()
    separate_by_sessions = bool(params.get("separate_by_sessions", ROUND1_DEFAULTS["separate_by_sessions"]))
    session_frames = session_start_frames if separate_by_sessions else None
    round1_pdf_path = None
    save_round1_pdf = bool(output_folder is not None and cell_idx is not None)
    if save_round1_pdf:
        figure_folder = os.path.join(output_folder, "SNR_figures")
        os.makedirs(figure_folder, exist_ok=True)
        round1_pdf_path = os.path.join(figure_folder, f"cell_{int(cell_idx)}_round1_diagnostics.pdf")
        if os.path.exists(round1_pdf_path):
            os.remove(round1_pdf_path)

    # Step 0: baseline subtraction
    baseline_window = int(frame_rate * 20)
    nan_mask = np.isnan(trace_idx)
    trace_baseline = median_filter(interpolate_nan_segment(trace_idx), size=baseline_window)
    trace_idx = trace_idx - trace_baseline
    trace_idx[nan_mask] = np.nan

    with pipeline_context():
        pdf_ctx = PdfPages(round1_pdf_path) if save_round1_pdf else nullcontext(None)
        with pdf_ctx as pdf:
            # Step 1: complex_bursts_detection (very high threshold -> almost no CB)
            close_all()
            complex_bursts_dict, _segment_bounds = complex_bursts_detection(
                trace_idx, spike_idx, frame_rate,
                pnorm=params.get("pnorm_CS", ROUND1_DEFAULTS["pnorm_CS"]),
                process_window=params.get("process_window_CS", ROUND1_DEFAULTS["process_window_CS"]),
                CB_detection_method="simple",
                simple_threshold=50,
                separate_by_sessions=separate_by_sessions,
                session_start_frames=session_frames,
                f_hp=params.get("f_hp_CS", ROUND1_DEFAULTS["f_hp_CS"]),
                plotflag=save_round1_pdf,
            )
            if save_round1_pdf and pdf is not None:
                _append_open_figures_to_pdf(pdf)

            # Step 2: refine_single_spikes
            close_all()
            refined_SS, trace_noCS = refine_single_spikes(
                trace_idx, spike_idx, complex_bursts_dict, frame_rate,
                process_window=params.get("process_window_SS", ROUND1_DEFAULTS["process_window_SS"]),
                pnorm=params.get("pnorm_SS", ROUND1_DEFAULTS["pnorm_SS"]),
                f_hp=params.get("f_hp", ROUND1_DEFAULTS["f_hp"]),
                min_spikes=10,
                SS_detection_method="simple",
                simple_threshold_SS=params.get("simple_threshold_SS", ROUND1_DEFAULTS["simple_threshold_SS"]),
                separate_by_sessions=separate_by_sessions,
                session_start_frames=session_frames,
                plotflag=save_round1_pdf,
            )
            if save_round1_pdf and pdf is not None:
                _append_open_figures_to_pdf(pdf)

            spike_idx = refined_SS

            # Step 3: spike_height_calculation2
            close_all()
            spike_heights_interpolated, SNR_interpolated = spike_height_calculation2(
                refined_SS, trace_idx,
                complex_bursts_dict["trace_mf"],
                trace_noCS, frame_rate,
                plotflag=save_round1_pdf,
                session_start_frames=session_frames,
                pdf=None,
            )
            if save_round1_pdf and pdf is not None:
                _append_open_figures_to_pdf(pdf)

    figures = []

    results = {
        "trace_idx": trace_idx,
        "spike_idx": spike_idx,
        "complex_bursts_dict": complex_bursts_dict,
        "refined_SS": refined_SS,
        "trace_noCS": trace_noCS,
        "spike_heights_interpolated": spike_heights_interpolated,
        "SNR_interpolated": SNR_interpolated,
        "separate_by_sessions": separate_by_sessions,
        "round1_pdf_path": round1_pdf_path,
    }
    return results, figures


# ---------------------------------------------------------------------------
# Round 2
# ---------------------------------------------------------------------------

def run_round2(round1_results, frame_rate, session_start_frames, params, *, cell_idx=None, output_folder=None):
    """Run Round 2: normalized CS detection + burst classification.

    Returns
    -------
    results : dict
        Final per-cell results.
    figures : list[dict]
        Empty list (Round 2 is delivered as a saved PDF for display).
    """
    trace_idx = round1_results["trace_idx"].copy()
    spike_idx = round1_results["spike_idx"].copy()
    spike_heights_interpolated = round1_results["spike_heights_interpolated"]
    SNR_interpolated = round1_results["SNR_interpolated"]
    separate_by_sessions = bool(
        params.get(
            "separate_by_sessions",
            round1_results.get("separate_by_sessions", ROUND1_DEFAULTS["separate_by_sessions"]),
        )
    )
    session_frames = session_start_frames if separate_by_sessions else None
    round2_pdf_path = None
    save_round2_pdf = bool(output_folder is not None and cell_idx is not None)
    if save_round2_pdf:
        figure_folder = os.path.join(output_folder, "SNR_figures")
        os.makedirs(figure_folder, exist_ok=True)
        round2_pdf_path = os.path.join(figure_folder, f"cell_{int(cell_idx)}_round2_diagnostics.pdf")
        if os.path.exists(round2_pdf_path):
            os.remove(round2_pdf_path)

    # Normalize by spike height
    trace_idx = trace_idx / spike_heights_interpolated

    with pipeline_context():
        pdf_ctx = PdfPages(round2_pdf_path) if save_round2_pdf else nullcontext(None)
        with pdf_ctx as pdf:
            # Step 4: complex_bursts_detection (real threshold)
            close_all()
            complex_bursts_dict, segment_bounds = complex_bursts_detection(
                trace_idx, spike_idx, frame_rate,
                pnorm=params.get("pnorm_CS", ROUND1_DEFAULTS["pnorm_CS"]),
                process_window=params.get("process_window_CS", ROUND1_DEFAULTS["process_window_CS"]),
                CB_detection_method="simple",
                simple_threshold=params.get("simple_threshold", ROUND2_DEFAULTS["simple_threshold"]),
                separate_by_sessions=separate_by_sessions,
                session_start_frames=session_frames,
                f_hp=params.get("f_hp_CS", ROUND1_DEFAULTS["f_hp_CS"]),
                plotflag=save_round2_pdf,
            )
            if save_round2_pdf and pdf is not None:
                _append_open_figures_to_pdf(pdf)

            # Step 5: refine_single_spikes
            close_all()
            refined_SS, trace_noCS = refine_single_spikes(
                trace_idx, spike_idx, complex_bursts_dict, frame_rate,
                process_window=params.get("process_window_SS", ROUND1_DEFAULTS["process_window_SS"]),
                pnorm=params.get("pnorm_SS", ROUND1_DEFAULTS["pnorm_SS"]),
                f_hp=params.get("f_hp", ROUND1_DEFAULTS["f_hp"]),
                min_spikes=10,
                SS_detection_method="simple",
                simple_threshold_SS=params.get("simple_threshold_SS", ROUND1_DEFAULTS["simple_threshold_SS"]),
                SS_height_cap=params.get("SS_height_cap", ROUND2_DEFAULTS["SS_height_cap"]),
                separate_by_sessions=separate_by_sessions,
                session_start_frames=session_frames,
                plotflag=save_round2_pdf,
            )
            if save_round2_pdf and pdf is not None:
                _append_open_figures_to_pdf(pdf)

            # Step 6: detect_complex_spikes
            close_all()
            cs_threshold = params.get("complex_spike_threshold", ROUND2_DEFAULTS["complex_spike_threshold"])
            CS_spikes, rejected_cs_spikes, rejected_cs_heights = detect_complex_spikes(
                trace_idx, complex_bursts_dict,
                np.ones_like(trace_idx),
                threshold=cs_threshold,
                plotflag=save_round2_pdf,
                return_rejected=True,
            )
            if save_round2_pdf and pdf is not None:
                _append_open_figures_to_pdf(pdf)

    # Step 7: refine_all_spikes (no plotting)
    complex_bursts_dict, refined_SS, all_CS_spikes, all_spikes = refine_all_spikes(
        complex_bursts_dict, CS_spikes, refined_SS,
    )

    # Spikes rejected from CS classification can be kept as SS only if
    # they pass SS_height_cap; otherwise they are removed from all spikes.
    ss_height_cap = params.get("SS_height_cap", ROUND2_DEFAULTS["SS_height_cap"])
    if rejected_cs_spikes.size > 0:
        if ss_height_cap is None:
            rescued_ss = rejected_cs_spikes
        else:
            rescued_ss = rejected_cs_spikes[rejected_cs_heights >= float(ss_height_cap)]

        if rescued_ss.size > 0:
            refined_SS = np.sort(np.unique(np.concatenate([np.asarray(refined_SS, dtype=np.int64), rescued_ss])))
            all_CS_spikes = np.asarray(all_CS_spikes, dtype=np.int64)
            if all_CS_spikes.size > 0:
                refined_SS = refined_SS[~np.isin(refined_SS, all_CS_spikes)]

            if refined_SS.size > 0 and all_CS_spikes.size > 0:
                all_spikes = np.sort(np.concatenate([refined_SS, all_CS_spikes]))
            elif refined_SS.size > 0:
                all_spikes = np.sort(refined_SS)
            else:
                all_spikes = np.sort(all_CS_spikes)

    # Step 8: detect_bursts_from_vm (results only, no PDF)
    (
        simple_spikes_final, complex_spikes_final, all_spikes_final,
        trace_SNR_interpolated, Vm, burst_metrics,
        complex_bursts_dict_vm, plateaus_dict,
    ) = detect_bursts_from_vm(
        trace_idx,
        np.ones_like(trace_idx),
        complex_bursts_dict,
        all_spikes,
        frame_rate,
        highpass=params.get("highpass", ROUND2_DEFAULTS["highpass"]),
        median_window=params.get("median_window", ROUND2_DEFAULTS["median_window"]),
        cb_amp_threshold=params.get("cb_amp_threshold", ROUND2_DEFAULTS["cb_amp_threshold"]),
        cb_duration_threshold=params.get("cb_duration_threshold", ROUND2_DEFAULTS["cb_duration_threshold"]),
        isi_threshold_ms=params.get("isi_threshold_ms", ROUND2_DEFAULTS["isi_threshold_ms"]),
        baseline_subtract=params.get("baseline_subtract", ROUND2_DEFAULTS["baseline_subtract"]),
        baseline_window_ms=params.get("baseline_window_ms", ROUND2_DEFAULTS["baseline_window_ms"]),
        baseline_percentile=params.get("baseline_percentile", ROUND2_DEFAULTS["baseline_percentile"]),
        vm_crossing_threshold=params.get("vm_crossing_threshold", ROUND2_DEFAULTS["vm_crossing_threshold"]),
        min_num_spikes=params.get("min_num_spikes", ROUND2_DEFAULTS["min_num_spikes"]),
        merge_SS_ms=params.get("merge_SS_ms", ROUND2_DEFAULTS["merge_SS_ms"]),
        merge_CB_ms=params.get("merge_CB_ms", ROUND2_DEFAULTS["merge_CB_ms"]),
        plateau_amp_threshold=params.get("plateau_amp_threshold", ROUND2_DEFAULTS["plateau_amp_threshold"]),
        plateau_duration_threshold=params.get("plateau_duration_threshold", ROUND2_DEFAULTS["plateau_duration_threshold"]),
        plateau_kernel_ms=params.get("plateau_kernel_ms", ROUND2_DEFAULTS["plateau_kernel_ms"]),
        plateau_score_min_ms=params.get("plateau_score_min_ms", ROUND2_DEFAULTS["plateau_score_min_ms"]),
        plotflag=False,
        pdf=None,
    )

    results = {
        "trace_idx_normalized": trace_idx,
        "complex_bursts_dict_vm": complex_bursts_dict_vm,
        "refined_SS": simple_spikes_final,
        "all_CS_spikes": complex_spikes_final,
        "all_spikes": all_spikes_final,
        "spike_heights_interpolated": spike_heights_interpolated,
        "SNR_interpolated": SNR_interpolated,
        "trace_SNR_interpolated": trace_SNR_interpolated,
        "Vm": Vm,
        "burst_metrics": burst_metrics,
        "plateaus_dict": plateaus_dict,
        "round2_pdf_path": round2_pdf_path,
    }
    figures = []
    return results, figures


# ---------------------------------------------------------------------------
# PDF generation
# ---------------------------------------------------------------------------

def generate_pdf(round1_results, round2_results, frame_rate, session_start_frames,
                 cell_idx, output_folder):
    """Generate a full diagnostic PDF for a cell and auto-open it.

    Returns the PDF path.
    """
    figure_folder = os.path.join(output_folder, "SNR_figures")
    os.makedirs(figure_folder, exist_ok=True)
    pdf_path = os.path.join(figure_folder, f"cell_{cell_idx}_burst_detection.pdf")

    if os.path.exists(pdf_path):
        os.remove(pdf_path)

    with pipeline_context():
        close_all()

        with PdfPages(pdf_path) as pdf:
            # Spike height page
            spike_height_calculation2(
                round1_results["refined_SS"],
                round1_results["trace_idx"],
                round1_results["complex_bursts_dict"]["trace_mf"],
                round1_results["trace_noCS"],
                frame_rate,
                plotflag=True,
                session_start_frames=session_start_frames,
                pdf=pdf,
            )
            close_all()

            # Trace segments with bursts
            r2 = round2_results
            plot_trace_with_bursts_pdf(
                r2["trace_SNR_interpolated"],
                r2["Vm"],
                r2["refined_SS"],
                r2["all_CS_spikes"],
                r2["complex_bursts_dict_vm"],
                frame_rate,
                segment_duration=5,
                rows_per_page=100,
                pdf=pdf,
                plateaus_dict=r2["plateaus_dict"],
            )
            close_all()

            # Burst metrics
            plot_burst_metrics_pdf(r2["burst_metrics"], pdf=pdf, figsize=(6, 3))
            close_all()

    # Auto-open on macOS
    try:
        subprocess.run(["open", pdf_path])
    except Exception as exc:
        print(f"Could not auto-open PDF: {exc}")

    return pdf_path
