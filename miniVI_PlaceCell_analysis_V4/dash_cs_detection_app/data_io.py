"""Load concatenated demix data and save CS detection results."""

import glob
import os
import pickle
import sys

import numpy as np

# Add the utils directory directly to sys.path (no __init__.py in utils/)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_UTILS_DIR = os.path.join(os.path.dirname(_THIS_DIR), "utils")
if _UTILS_DIR not in sys.path:
    sys.path.insert(0, _UTILS_DIR)

from preprocess_neural import (
    load_concat_raw_manifest,
    read_Volpy_demix_results_data,
    split_concatenated_volpy_results,
)


def load_cs_bundle(main_data_folder):
    """Load concatenated demix results and return a bundle dict.

    Mirrors the notebook cells: subfolders discovery, concat demix loading,
    per-session splitting, trace concatenation.

    If ``merged_neural_CS.pkl`` exists, also loads existing results/params
    so the app can resume a previous session.
    """
    # --- Discover -good subfolders ---
    subfolders = sorted(
        os.path.join(main_data_folder, f)
        for f in os.listdir(main_data_folder)
        if os.path.isdir(os.path.join(main_data_folder, f)) and f.endswith("-good")
    )

    # --- Load concatenated demix ---
    concat_root = os.path.join(main_data_folder, "output_concat")
    concat_manifest_path = os.path.join(concat_root, "concat_raw_manifest.json")

    manifest = load_concat_raw_manifest(concat_manifest_path)
    session_edges_frames = manifest.get("session_edges_frames", None)

    candidates = []
    candidates += glob.glob(os.path.join(concat_root, "output_final", "*", "volpy_demix_results_annotated.pickle"))
    candidates += glob.glob(os.path.join(concat_root, "output_final", "*", "volpy_demix_results.pickle"))
    candidates += glob.glob(os.path.join(concat_root, "volpy_demix_results_annotated.pickle"))
    candidates += glob.glob(os.path.join(concat_root, "volpy_demix_results.pickle"))

    if not candidates:
        raise FileNotFoundError(f"No volpy demix results found under: {concat_root}")

    concat_demix_path = candidates[0]
    print("Using concatenated demix:", concat_demix_path)

    with open(concat_demix_path, "rb") as f:
        concat_results = pickle.load(f)

    split_sessions = split_concatenated_volpy_results(
        concat_results,
        session_edges_frames,
        frame_rate_hz=manifest.get("frame_rate_hz", None),
    )

    session_folders_manifest = [
        s.get("session_good_folder_local", "")
        for s in manifest.get("sessions_local", [])
    ]
    if (
        any(not p for p in session_folders_manifest)
        or len(session_folders_manifest) != len(split_sessions)
    ):
        print("[WARN] Could not map manifest sessions; falling back to subfolders order.")
        session_folders_manifest = list(subfolders)[: len(split_sessions)]

    demix_by_session = dict(zip(session_folders_manifest, split_sessions))

    # Ensure subfolders follow concatenation order
    subfolders = [sf for sf in session_folders_manifest if sf in subfolders] + [
        sf for sf in subfolders if sf not in session_folders_manifest
    ]

    # --- Load and concatenate neural data ---
    traces_all, traces_raw_all = [], []
    spikes_all, spikes_volpy_all = [], []
    weights_all, ROIs_all = [], []
    mean_images, mean_images_raw = [], []
    session_start_frames = []
    traces_length = 0

    for data_folder in subfolders:
        print(f"Loading session: {data_folder}")
        (
            traces_sess, traces_raw_sess,
            spikes_sess, spikes_volpy_sess,
            weights, ROIs, mean_image, mean_image_raw, _metadata,
        ) = read_Volpy_demix_results_data(demix_by_session[data_folder], plotflag=False)

        session_start_frames.append(traces_length)
        traces_all.append(traces_sess)
        traces_raw_all.append(traces_raw_sess)
        spikes_all.append(spikes_sess + traces_length)
        spikes_volpy_all.append(spikes_volpy_sess + traces_length)
        weights_all.append(weights)
        ROIs_all.append(ROIs)
        mean_images.append(mean_image)
        mean_images_raw.append(mean_image_raw)
        traces_length += traces_sess.shape[1]

    traces = np.concatenate(traces_all, axis=1)
    traces_raw_shape = traces_raw_all[0].shape
    if traces_raw_shape[0] < traces_raw_shape[1]:
        traces_raw = np.concatenate(traces_raw_all, axis=1)
    else:
        traces_raw = np.concatenate(traces_raw_all, axis=0)

    spikes_all_np = np.array(spikes_all, dtype=object)
    spikes = [
        np.concatenate(spikes_all_np[:, n]) for n in range(spikes_all_np.shape[1])
    ]

    spikes_volpy_all_np = np.array(spikes_volpy_all, dtype=object)
    spikes_volpy = [
        np.concatenate(spikes_volpy_all_np[:, n])
        for n in range(spikes_volpy_all_np.shape[1])
    ]

    frame_rate = int(manifest.get("frame_rate_hz", 500))

    print(f"Traces shape: {traces.shape}, frame_rate: {frame_rate} Hz")
    print(f"Session start frames: {session_start_frames}")

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
    }

    # --- Load existing results if available ---
    existing_pkl = os.path.join(main_data_folder, "merged_neural_CS.pkl")
    if os.path.isfile(existing_pkl):
        print(f"Loading existing results from: {existing_pkl}")
        with open(existing_pkl, "rb") as f:
            existing = pickle.load(f)
        bundle["existing_results"] = existing
    else:
        bundle["existing_results"] = None

    return bundle


def save_merged_results(bundle, cell_results, cell_params, removed_cells):
    """Save merged_neural_CS.pkl in the same format as the notebook.

    Parameters
    ----------
    bundle : dict
        The bundle from ``load_cs_bundle``.
    cell_results : dict[int, dict]
        Per-cell computed results (from ``cs_pipeline``).
    cell_params : dict[int, dict]
        Per-cell parameter overrides.
    removed_cells : set[int]
        Cell indices marked as removed (bad SNR).

    Returns
    -------
    str
        Path to the saved pickle.
    """
    n_cells = bundle["n_cells"]

    def _init():
        return [None] * n_cells

    all_complex_bursts_dicts = _init()
    all_refined_SS = _init()
    all_CS_spikes = _init()
    all_spikes = _init()
    all_spike_heights = _init()
    all_SNR = _init()
    all_traces_SNR = _init()
    all_Vm = _init()
    all_burst_metrics = _init()
    all_plateaus = _init()

    for idx, res in cell_results.items():
        if idx in removed_cells:
            continue
        all_complex_bursts_dicts[idx] = res["complex_bursts_dict_vm"]
        all_refined_SS[idx] = res["refined_SS"]
        all_CS_spikes[idx] = res["all_CS_spikes"]
        all_spikes[idx] = res["all_spikes"]
        all_spike_heights[idx] = res["spike_heights_interpolated"]
        all_SNR[idx] = res["SNR_interpolated"]
        all_traces_SNR[idx] = res["trace_SNR_interpolated"]
        all_Vm[idx] = res["Vm"]
        all_burst_metrics[idx] = res["burst_metrics"]
        all_plateaus[idx] = res["plateaus_dict"]

    save_data = {
        "traces": bundle["traces"],
        "traces_raw": bundle["traces_raw"],
        "spikes": bundle["spikes"],
        "spikes_volpy": bundle["spikes_volpy"],
        "frame_rate": bundle["frame_rate"],
        "session_start_frames": bundle["session_start_frames"],
        "subfolders": bundle["subfolders"],
        "weights_all": bundle["weights_all"],
        "ROIs_all": bundle["ROIs_all"],
        "mean_images": bundle["mean_images"],
        "mean_images_raw": bundle["mean_images_raw"],
        "complex_bursts_dicts": all_complex_bursts_dicts,
        "refined_SS": all_refined_SS,
        "all_CS_spikes": all_CS_spikes,
        "all_spikes": all_spikes,
        "spike_heights_interpolated": all_spike_heights,
        "SNR_interpolated": all_SNR,
        "traces_SNR_interpolated": all_traces_SNR,
        "Vm_SNR_interpolated": all_Vm,
        "burst_metrics": all_burst_metrics,
        "plateaus_dicts": all_plateaus,
        "params_by_cell": cell_params,
        "removed_cells": sorted(removed_cells),
    }

    save_path = os.path.join(bundle["main_data_folder"], "merged_neural_CS.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(save_data, f)

    print(f"Saved to: {save_path}")
    return save_path
