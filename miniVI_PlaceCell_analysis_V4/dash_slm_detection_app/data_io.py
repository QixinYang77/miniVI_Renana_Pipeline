"""Data loading and persistence helpers for the SLM tuning app."""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np
from scipy.io import loadmat


def interpolate_trace(trace, n_bad_frames=0):
    """Mask the initial frames and linearly interpolate NaNs."""
    interpolated = np.asarray(trace, dtype=float).copy()
    interpolated[:n_bad_frames] = np.nan
    nan_mask = np.isnan(interpolated)
    if not nan_mask.any():
        return interpolated

    valid_idx = np.flatnonzero(~nan_mask)
    if valid_idx.size == 0:
        raise ValueError("Trace contains no valid samples after masking bad frames.")

    interpolated[nan_mask] = np.interp(
        np.flatnonzero(nan_mask),
        valid_idx,
        interpolated[valid_idx],
    )
    return interpolated


def build_slm_bundle(slm_root, sampling_rate_hz=500, initial_bad_frames=50):
    """Load SLM traces and return a notebook/app bundle."""
    slm_root = Path(slm_root).expanduser().resolve()
    if not slm_root.is_dir():
        raise FileNotFoundError(f"SLM root does not exist: {slm_root}")

    trace_lists = {"1x": [], "20x": []}
    trace_info_by_condition = {"1x": [], "20x": []}
    ordered_cells = []
    global_cell_index = 0

    for folder in sorted(slm_root.iterdir()):
        if not folder.is_dir():
            continue

        folder_name = folder.name
        if folder_name.endswith("_1x"):
            condition = "1x"
        elif folder_name.endswith("_20x"):
            condition = "20x"
        else:
            continue

        mat_files = sorted(
            path for path in folder.glob("*.mat")
            if not path.name.startswith("._")
        )
        if not mat_files:
            continue

        mat_path = mat_files[0]
        mat_data = loadmat(mat_path)
        if "intens" not in mat_data:
            raise KeyError(f"'intens' trace not found in {mat_path}")

        trace = np.asarray(mat_data["intens"], dtype=float).squeeze()
        trace = interpolate_trace(trace, n_bad_frames=initial_bad_frames)
        condition_cell_index = len(trace_lists[condition])
        cell_key = f"{condition}::{folder_name}"

        trace_lists[condition].append(trace)
        info = {
            "cell_key": cell_key,
            "condition": condition,
            "folder": folder_name,
            "mat_file": mat_path.name,
            "condition_cell_index": condition_cell_index,
            "global_cell_index": global_cell_index,
        }
        trace_info_by_condition[condition].append(info)
        ordered_cells.append(info.copy())
        global_cell_index += 1

    traces_by_condition = {}
    trace_lookup = {}
    for condition in ("1x", "20x"):
        if trace_lists[condition]:
            traces = np.stack(trace_lists[condition])
        else:
            traces = np.empty((0, 0), dtype=float)
        traces_by_condition[condition] = traces
        for info in trace_info_by_condition[condition]:
            trace_lookup[info["cell_key"]] = traces[info["condition_cell_index"]]

    return {
        "slm_root": str(slm_root),
        "sampling_rate_hz": int(sampling_rate_hz),
        "initial_bad_frames": int(initial_bad_frames),
        "traces_by_condition": traces_by_condition,
        "trace_info_by_condition": trace_info_by_condition,
        "ordered_cells": ordered_cells,
        "trace_lookup": trace_lookup,
        "n_cells_total": len(ordered_cells),
    }


def save_bundle_pickle(bundle, bundle_path):
    """Persist a preloaded SLM bundle for the launcher/app handoff."""
    bundle_path = Path(bundle_path).expanduser().resolve()
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    with bundle_path.open("wb") as f:
        pickle.dump(bundle, f)
    return str(bundle_path)


def load_bundle_pickle(bundle_path):
    """Load a prebuilt SLM bundle from disk."""
    bundle_path = Path(bundle_path).expanduser().resolve()
    with bundle_path.open("rb") as f:
        return pickle.load(f)


def load_saved_results(results_path):
    """Load previously saved SLM GUI results if available."""
    results_path = Path(results_path).expanduser().resolve()
    if not results_path.is_file():
        return None
    with results_path.open("rb") as f:
        return pickle.load(f)


def save_slm_results(bundle, results_by_cell_key, params_by_cell_key, removed_cell_keys, save_path):
    """Save full SLM GUI outputs for later notebook use."""
    save_path = Path(save_path).expanduser().resolve()
    save_path.parent.mkdir(parents=True, exist_ok=True)

    serializable_results = {}
    for cell_key, result in results_by_cell_key.items():
        serializable = dict(result)
        serializable_results[cell_key] = serializable

    save_data = {
        "slm_root": bundle["slm_root"],
        "sampling_rate_hz": bundle["sampling_rate_hz"],
        "initial_bad_frames": bundle["initial_bad_frames"],
        "n_cells_total": bundle["n_cells_total"],
        "ordered_cells": bundle["ordered_cells"],
        "trace_info_by_condition": bundle["trace_info_by_condition"],
        "results_by_cell_key": serializable_results,
        "params_by_cell_key": params_by_cell_key,
        "removed_cell_keys": sorted(removed_cell_keys),
    }

    with save_path.open("wb") as f:
        pickle.dump(save_data, f)
    return str(save_path)

