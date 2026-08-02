"""Adapter from saved ALI results to the interactive demixing dashboard."""

from __future__ import annotations

from pathlib import Path
import pickle

import numpy as np


def write_ali_dashboard_bundle(
    output_path: Path,
    *,
    ali_result,
    mean_image_raw: np.ndarray,
    mean_image_motion_corrected: np.ndarray,
    mean_image_motion_projected: np.ndarray,
    registration_shifts: np.ndarray,
    frame_rate_hz: float,
    source_movie_path: Path,
    ali_numeric_result_path: Path,
    support_fraction: float = 0.20,
) -> Path:
    """Write one ALI result in ``dash_overview_app`` bundle format.

    The dashboard is algorithm-agnostic once traces, spatial weights, ROI
    supports, stage images, shifts, and event frames are provided. ALI traces
    remain explicitly named ``ali_trace`` in the source selector.
    """

    if not 0 < support_fraction < 1:
        raise ValueError("support_fraction must lie between zero and one")
    traces = np.asarray(ali_result.traces, dtype=np.float32)
    detection_traces = np.asarray(
        ali_result.detection_traces,
        dtype=np.float32,
    )
    weights = np.asarray(ali_result.footprints, dtype=np.float32)
    if traces.ndim != 2:
        raise ValueError("ALI traces must have shape (components, time)")
    if detection_traces.shape != traces.shape:
        raise ValueError("ALI detection and full-band traces must align")
    if weights.ndim != 3 or weights.shape[0] != traces.shape[0]:
        raise ValueError("ALI footprints must have shape (components, y, x)")

    image_shape = tuple(weights.shape[1:])
    stage_images = [
        np.asarray(mean_image_raw, dtype=np.float32),
        np.asarray(mean_image_motion_corrected, dtype=np.float32),
        np.asarray(mean_image_motion_projected, dtype=np.float32),
    ]
    if any(image.shape != image_shape for image in stage_images):
        raise ValueError("All stage mean images must match ALI footprints")

    shifts = np.asarray(registration_shifts, dtype=np.float32)
    if shifts.shape != (traces.shape[1], 2):
        raise ValueError(
            "registration_shifts must have shape (trace frames, 2)"
        )
    positive = np.maximum(weights, 0)
    peaks = np.max(positive, axis=(1, 2), keepdims=True)
    supports = positive >= support_fraction * np.maximum(peaks, 1e-8)
    if np.any(supports.reshape(len(supports), -1).sum(axis=1) == 0):
        raise ValueError("At least one ALI support mask is empty")

    events = [
        np.asarray(frames, dtype=np.int64).reshape(-1).tolist()
        for frames in ali_result.assigned_spike_frames
    ]
    if len(events) != traces.shape[0]:
        raise ValueError("ALI event lists must match component count")

    bundle = {
        "algorithm": "Activity Localization Imaging via pyALI",
        "mean_img_raw": stage_images[0],
        "mean_img_mc": stage_images[1],
        "mean_img_mc_denoised": stage_images[2],
        "weights": weights,
        "ROIs": supports,
        "raw_traces": None,
        "mc_traces": traces,
        "mc_denoised_traces": traces,
        "ali_trace": traces,
        "ali_detection_trace": detection_traces,
        "shift_distances": np.linalg.norm(shifts, axis=1).astype(np.float32),
        "reg_shifts": shifts,
        "frame_rate": float(frame_rate_hz),
        "input_movie": str(Path(source_movie_path).resolve()),
        "ali_numeric_result": str(Path(ali_numeric_result_path).resolve()),
        "preferred_trace_source": "ali_trace",
        "spikes": events,
        "spikes_verified_array": events,
        "show_spikes_by_default": True,
        "cell_numbers": np.arange(1, traces.shape[0] + 1),
        "good_cells": list(range(traces.shape[0])),
        "image_labels": {
            "raw": "Raw mean — representative window",
            "mc": "After NoRMCorre",
            "mc_denoised": "After motion-artifact projection",
            "weights": "Maximum ALI spatial weight",
        },
        "ali_support_fraction": float(support_fraction),
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.temporary")
    with temporary.open("wb") as handle:
        pickle.dump(bundle, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(output_path)
    return output_path.resolve()

