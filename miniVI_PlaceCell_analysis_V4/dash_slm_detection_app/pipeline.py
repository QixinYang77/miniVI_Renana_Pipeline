"""SLM spike-detection pipeline for the tuning app."""

from __future__ import annotations

import os
import sys
from contextlib import nullcontext

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from scipy.ndimage import median_filter

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_THIS_DIR)
_UTILS_DIR = os.path.join(_PARENT_DIR, "utils")
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)
if _UTILS_DIR not in sys.path:
    sys.path.insert(0, _UTILS_DIR)

from dash_cs_detection_app.fig_capture import close_all, pipeline_context  # noqa: E402
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


SLM_DEFAULTS = {
    "baseline_window_s": 10.0,
    "pnorm_CS": 0.5,
    "process_window_CS": 300000,
    "simple_threshold_round1_CB": 50.0,
    "simple_threshold_round2_CB": 4.0,
    "pnorm_SS": 0.5,
    "process_window_SS": 300000,
    "simple_threshold_SS": 5.0,
    "f_hp": 20.0,
    "f_hp_CS": 1.0,
    "SS_height_cap": 0.7,
    "complex_spike_threshold": [0.7, 0.6],
    "highpass": 2.0,
    "median_window": 11,
    "cb_amp_threshold": 0.4,
    "cb_duration_threshold": 20,
    "min_num_spikes": 2,
    "plateau_amp_threshold": 0.8,
    "plateau_duration_threshold": 100,
    "plateau_kernel_ms": 100,
    "plateau_score_min_ms": 80,
    "isi_threshold_ms": 20,
    "baseline_subtract": False,
    "baseline_window_ms": 20,
    "baseline_percentile": None,
    "vm_crossing_threshold": 0.1,
    "merge_SS_ms": None,
    "merge_CB_ms": None,
}


def get_default_params():
    """Return a fresh copy of the SLM tuning defaults."""
    defaults = dict(SLM_DEFAULTS)
    defaults["complex_spike_threshold"] = list(SLM_DEFAULTS["complex_spike_threshold"])
    return defaults


def _append_open_figures_to_pdf(pdf):
    for fig_num in plt.get_fignums():
        fig = plt.figure(fig_num)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def _safe_cell_name(cell_key):
    return str(cell_key).replace("::", "__").replace("/", "_")


def run_detection(trace, sampling_rate_hz, params, *, cell_key=None, pdf_output_dir=None):
    """Run the SLM spike-detection flow and optionally write diagnostics."""
    trace = np.asarray(trace, dtype=float).copy()
    nan_mask = np.isnan(trace)
    initial_spikes = np.array([], dtype=np.int64)

    baseline_window = max(1, int(round(params["baseline_window_s"] * sampling_rate_hz)))
    if baseline_window % 2 == 0:
        baseline_window += 1

    trace_baseline = median_filter(interpolate_nan_segment(trace.copy()), size=baseline_window)
    trace_baseline_subtracted = trace - trace_baseline
    trace_baseline_subtracted[nan_mask] = np.nan

    diagnostic_pdf_path = None
    save_pdf = bool(pdf_output_dir and cell_key)
    if save_pdf:
        os.makedirs(pdf_output_dir, exist_ok=True)
        diagnostic_pdf_path = os.path.join(
            pdf_output_dir,
            f"{_safe_cell_name(cell_key)}_burst_detection.pdf",
        )
        if os.path.exists(diagnostic_pdf_path):
            os.remove(diagnostic_pdf_path)

    with pipeline_context():
        pdf_ctx = PdfPages(diagnostic_pdf_path) if save_pdf else nullcontext(None)
        with pdf_ctx as pdf:
            close_all()
            complex_bursts_round1, trace_mf_round1, *_ = complex_bursts_detection(
                trace_baseline_subtracted,
                initial_spikes,
                sampling_rate_hz,
                pnorm=params["pnorm_CS"],
                process_window=params["process_window_CS"],
                plotflag=save_pdf,
                CB_detection_method="simple",
                simple_threshold=params["simple_threshold_round1_CB"],
                separate_by_sessions=False,
                session_start_frames=None,
                f_hp=params["f_hp_CS"],
            )
            if save_pdf and pdf is not None:
                _append_open_figures_to_pdf(pdf)

            close_all()
            refined_ss_round1, trace_noCS_round1 = refine_single_spikes(
                trace_baseline_subtracted,
                initial_spikes,
                complex_bursts_round1,
                sampling_rate_hz,
                process_window=params["process_window_SS"],
                pnorm=params["pnorm_SS"],
                f_hp=params["f_hp"],
                min_spikes=10,
                plotflag=save_pdf,
                separate_by_sessions=False,
                session_start_frames=None,
                SS_detection_method="simple",
                simple_threshold_SS=params["simple_threshold_SS"],
            )
            if save_pdf and pdf is not None:
                _append_open_figures_to_pdf(pdf)

            close_all()
            spike_heights_interpolated, SNR_interpolated = spike_height_calculation2(
                refined_ss_round1,
                trace_baseline_subtracted,
                trace_mf_round1,
                trace_noCS_round1,
                sampling_rate_hz,
                plotflag=save_pdf,
                session_start_frames=None,
                pdf=pdf,
            )
            if save_pdf and pdf is not None:
                _append_open_figures_to_pdf(pdf)

            with np.errstate(divide="ignore", invalid="ignore"):
                trace_snr_input = trace_baseline_subtracted / spike_heights_interpolated
            trace_snr_input[~np.isfinite(trace_snr_input)] = np.nan

            close_all()
            complex_bursts_round2, *_ = complex_bursts_detection(
                trace_snr_input,
                refined_ss_round1,
                sampling_rate_hz,
                pnorm=params["pnorm_CS"],
                process_window=params["process_window_CS"],
                plotflag=save_pdf,
                CB_detection_method="simple",
                simple_threshold=params["simple_threshold_round2_CB"],
                separate_by_sessions=False,
                session_start_frames=[0],
                f_hp=params["f_hp_CS"],
            )
            if save_pdf and pdf is not None:
                _append_open_figures_to_pdf(pdf)

            close_all()
            refined_ss_round2, _ = refine_single_spikes(
                trace_snr_input,
                refined_ss_round1,
                complex_bursts_round2,
                sampling_rate_hz,
                process_window=params["process_window_SS"],
                pnorm=params["pnorm_SS"],
                f_hp=params["f_hp"],
                min_spikes=10,
                plotflag=save_pdf,
                separate_by_sessions=False,
                session_start_frames=[0],
                SS_detection_method="simple",
                simple_threshold_SS=params["simple_threshold_SS"],
                SS_height_cap=params["SS_height_cap"],
            )
            if save_pdf and pdf is not None:
                _append_open_figures_to_pdf(pdf)

            close_all()
            CS_spikes = detect_complex_spikes(
                trace_snr_input,
                complex_bursts_round2,
                np.ones_like(trace_snr_input),
                threshold=params["complex_spike_threshold"],
                plotflag=save_pdf,
            )
            if save_pdf and pdf is not None:
                _append_open_figures_to_pdf(pdf)

            complex_bursts_round2, refined_ss_round2, all_CS_spikes, all_spikes = refine_all_spikes(
                complex_bursts_round2,
                CS_spikes,
                refined_ss_round2,
            )

            (
                simple_spikes_final,
                complex_spikes_final,
                all_spikes_final,
                trace_SNR_interpolated,
                Vm,
                burst_metrics,
                complex_bursts_dict_vm,
                plateaus_dict,
            ) = detect_bursts_from_vm(
                trace_snr_input,
                np.ones_like(trace_snr_input),
                complex_bursts_round2,
                all_spikes,
                sampling_rate_hz,
                highpass=params["highpass"],
                median_window=params["median_window"],
                cb_amp_threshold=params["cb_amp_threshold"],
                cb_duration_threshold=params["cb_duration_threshold"],
                isi_threshold_ms=params["isi_threshold_ms"],
                baseline_subtract=params["baseline_subtract"],
                baseline_window_ms=params["baseline_window_ms"],
                baseline_percentile=params["baseline_percentile"],
                vm_crossing_threshold=params["vm_crossing_threshold"],
                min_num_spikes=params["min_num_spikes"],
                merge_SS_ms=params["merge_SS_ms"],
                merge_CB_ms=params["merge_CB_ms"],
                plateau_amp_threshold=params["plateau_amp_threshold"],
                plateau_duration_threshold=params["plateau_duration_threshold"],
                plateau_kernel_ms=params["plateau_kernel_ms"],
                plateau_score_min_ms=params["plateau_score_min_ms"],
                plotflag=False,
            )

            if save_pdf and pdf is not None:
                close_all()
                plot_trace_with_bursts_pdf(
                    trace_SNR_interpolated,
                    Vm,
                    simple_spikes_final,
                    complex_spikes_final,
                    complex_bursts_dict_vm,
                    sampling_rate_hz,
                    segment_duration=5,
                    rows_per_page=100,
                    pdf=pdf,
                    plateaus_dict=plateaus_dict,
                )
                close_all()
                plot_burst_metrics_pdf(burst_metrics, pdf=pdf, figsize=(6, 3))
                close_all()

    return {
        "refined_SS_round1": refined_ss_round1,
        "spike_heights_interpolated": spike_heights_interpolated,
        "SNR_interpolated": SNR_interpolated,
        "trace_SNR_interpolated": trace_SNR_interpolated,
        "simple_spikes": simple_spikes_final,
        "complex_spikes": complex_spikes_final,
        "all_CS_spikes": complex_spikes_final,
        "all_spikes": all_spikes_final,
        "Vm": Vm,
        "burst_metrics": burst_metrics,
        "complex_bursts_dict": complex_bursts_dict_vm,
        "plateaus_dict": plateaus_dict,
        "trace_idx_normalized": trace_snr_input,
        "diagnostic_pdf_path": diagnostic_pdf_path,
    }

