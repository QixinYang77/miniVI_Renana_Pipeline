import numpy as np
import re
import pprint
import matplotlib.pyplot as plt
import os

import h5py
from scipy.interpolate import interp1d
import pandas as pd
import matplotlib.cm as cm
import numpy.ma as ma
from tqdm import tqdm
import cv2
import pickle
import glob
import json
from scipy.ndimage import uniform_filter1d
from scipy import signal
from scipy import stats
import logging
from scipy.ndimage import median_filter
from scipy.optimize import curve_fit

def _find_parent_good_folder(path: str) -> str:
    """
    Walk up from `path` until we find a folder ending with '-good'.
    Returns '' if not found.
    """
    if not path:
        return ""
    p = os.path.abspath(path)
    while True:
        if os.path.basename(p).endswith("-good") and os.path.isdir(p):
            return p
        parent = os.path.dirname(p)
        if parent == p:
            return ""
        p = parent


def load_concat_raw_manifest(manifest_path: str, *, cluster_prefix="/ems/elsc-labs/adam-y", local_prefix="/Volumes/adam-lab"):
    """
    Load concat_raw_manifest.json produced by utils_concat/concat_raw_sessions.py.
    Also provides local-mapped raw paths when manifest was created on the cluster.
    """
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    # Build a local-mapped view (does not modify the original dict)
    sessions_local = []
    for s in manifest.get("sessions", []):
        raw_path = s.get("raw_path", "")
        raw_local = raw_path.replace(cluster_prefix, local_prefix) if raw_path else raw_path
        s2 = dict(s)
        s2["raw_path_local"] = raw_local
        s2["session_good_folder_local"] = _find_parent_good_folder(raw_local)
        sessions_local.append(s2)

    out = dict(manifest)
    out["sessions_local"] = sessions_local
    return out


def _slice_2d_time(arr, s0, s1, T_expected):
    A = np.asarray(arr)
    if A.ndim != 2:
        return arr
    if A.shape[0] == T_expected:
        return A[s0:s1, :]
    if A.shape[1] == T_expected:
        return A[:, s0:s1]
    return arr


def _slice_1d_time(arr, s0, s1, T_expected):
    A = np.asarray(arr)
    if A.ndim == 1 and A.shape[0] == T_expected:
        return A[s0:s1]
    if A.ndim == 2 and A.shape[0] == T_expected:
        return A[s0:s1, :]
    return arr


def split_concatenated_volpy_results(concat_results: dict, session_edges_frames, *, frame_rate_hz=None):
    """
    Split a concatenated volpy_demix_results(_annotated).pickle dict into per-session dicts.

    - Uses `session_edges_frames` (length n_sessions+1) computed *after* per-session truncation.
    - Converts spikes/spikes_verified_array to session-local frame indices.
    - Splits bad_epochs (if present) to session-local times.
    """
    edges = np.asarray(session_edges_frames, dtype=int).reshape(-1)
    if edges.size < 2:
        raise ValueError("session_edges_frames must have length >= 2")

    # Determine concatenated length T from a reliable key.
    T = None
    for k in ["weighted_mc_denoised_traces", "mc_denoised_traces", "volpy_trace", "shift_distances"]:
        if k in concat_results and concat_results[k] is not None:
            A = np.asarray(concat_results[k])
            if A.ndim == 2:
                # One dimension is time, but we don't know orientation; pick the larger.
                T = int(max(A.shape[0], A.shape[1]))
                break
            if A.ndim == 1:
                T = int(A.shape[0])
                break
    if T is None:
        raise ValueError("Could not infer concatenated T from results keys.")

    if int(edges[-1]) > int(T):
        raise ValueError("session_edges_frames end (%d) > inferred T (%d)" % (int(edges[-1]), int(T)))

    fr = None
    if frame_rate_hz is not None:
        fr = float(frame_rate_hz)
    else:
        if "frame_rate" in concat_results:
            try:
                fr = float(concat_results["frame_rate"])
            except Exception:
                fr = None

    keys_2d = [
        "volpy_trace",
        "volpy_sub_trace",
        "raw_traces",
        "mc_traces",
        "mc_denoised_traces",
        "weighted_mc_traces",
        "weighted_mc_denoised_traces",
    ]
    keys_1d = ["shift_distances"]
    keys_2d_timefirst = ["reg_shifts"]  # (T,2) typically

    out_sessions = []
    for si in range(int(edges.size) - 1):
        s0 = int(edges[si])
        s1 = int(edges[si + 1])
        sess = dict(concat_results)

        # Slice numeric time-series arrays
        for k in keys_2d:
            if k in sess and sess[k] is not None:
                sess[k] = _slice_2d_time(sess[k], s0, s1, T_expected=T)
        for k in keys_1d:
            if k in sess and sess[k] is not None:
                sess[k] = _slice_1d_time(sess[k], s0, s1, T_expected=T)
        for k in keys_2d_timefirst:
            if k in sess and sess[k] is not None:
                sess[k] = _slice_1d_time(sess[k], s0, s1, T_expected=T)

        # Spikes: convert to session-local
        for spikes_key in ["spikes", "spikes_verified_array"]:
            if spikes_key in sess and sess[spikes_key] is not None:
                sp = sess[spikes_key]
                sp_out = []
                try:
                    for row in sp:
                        rr = np.asarray(row).astype(int).reshape(-1)
                        rr = rr[(rr >= s0) & (rr < s1)] - s0
                        sp_out.append(rr.tolist())
                    sess[spikes_key] = sp_out
                except Exception:
                    pass

        # bad_epochs: session-localize if possible (stored in seconds in SpikeDetectionAPP_Concat)
        if "bad_epochs" in sess and sess["bad_epochs"] and fr and fr > 0:
            try:
                t0 = float(s0) / float(fr)
                t1 = float(s1) / float(fr)
                out_be = []
                for e0, e1 in sess["bad_epochs"]:
                    lo, hi = min(float(e0), float(e1)), max(float(e0), float(e1))
                    # intersect with session window
                    lo2, hi2 = max(lo, t0), min(hi, t1)
                    if hi2 > lo2:
                        out_be.append((lo2 - t0, hi2 - t0))
                sess["bad_epochs"] = out_be
            except Exception:
                pass

        # Record concat provenance
        sess["concat_session_index"] = int(si)
        sess["concat_frame_offset"] = int(s0)
        sess["concat_frame_range"] = (int(s0), int(s1))
        out_sessions.append(sess)

    return out_sessions


def read_Volpy_demix_results_data(data, plotflag=False):
    """
    Same outputs as read_Volpy_demix_results_file(), but from an already-loaded pickle dict.
    """
    traces = data['weighted_mc_denoised_traces']
    traces_raw = data.get('weighted_mc_traces', None)
    if traces_raw is None:
        traces_raw = data.get('mc_traces', None)

    print("Original traces shape:", np.shape(traces))
    if np.shape(traces)[0] > np.shape(traces)[1]:
        traces = traces.T

    spikes = data.get('spikes_verified_array', None)
    spikes_volpy = data.get('spikes', None)
    good_cells = np.array(data.get('good_cells', np.arange(traces.shape[0])))

    weights = data['weights'][good_cells, :, :]
    ROIs = data['ROIs'][good_cells]
    mean_image_raw = data.get('mean_img_raw', None)
    mean_image = data.get('mean_img_mc', None)

    traces = traces[good_cells]

    if spikes is None:
        spikes = [[] for _ in range(int(traces.shape[0]))]
    spikes = [np.array(spikes[i]) for i in good_cells]
    spikes = np.array(spikes, dtype=object)

    if spikes_volpy is None:
        spikes_volpy = [[] for _ in range(int(traces.shape[0]))]
    spikes_volpy = [np.array(spikes_volpy[i]) for i in good_cells]
    spikes_volpy = np.array(spikes_volpy, dtype=object)

    metadata = data.get("metadata", {}) if isinstance(data, dict) else {}

    if plotflag:
        print("Traces shape:", traces.shape)
        print("Spikes shape:", spikes.shape)
        fig, axes = plt.subplots(traces.shape[0], 1, figsize=(10, 6), sharex=True, sharey=True)
        for i, trace, spks in zip(range(traces.shape[0]), traces, spikes):
            axes[i].plot(trace, label=f'Cell {i}',linewidth=0.5)
            if len(spks):
                axes[i].scatter(spks, trace[spks], color='red', s=5, label='Spikes')
            axes[i].axis('off')
        plt.tight_layout()
        plt.show()

    return traces, traces_raw, spikes, spikes_volpy, weights, ROIs, mean_image, mean_image_raw, metadata


def read_Volpy_demix_results_file_full(data_folder):
    print("Reading VolPy demix results from:", data_folder)
    sub_folder = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, f)) and 'TS' not in f]
    sub_folder.sort()
    sub_folder = sub_folder[0]
    output_file_path = os.path.join(sub_folder,'output','volpy_demix_results_annotated.pickle')
    file = open(output_file_path, 'rb')
    data = pickle.load(file)

    # also load the metadata
    metadata_files = glob.glob(os.path.join(sub_folder, 'output','*metadata.pkl'))
    with open(metadata_files[0], 'rb') as f:
        metadata = pickle.load(f)

    return data, metadata

def read_Volpy_demix_results_file(data_folder, plotflag=False):
    print("Reading VolPy demix results from:", data_folder)
    sub_folder = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, f)) and 'TS' not in f]
    sub_folder.sort()
    sub_folder = sub_folder[0]
    output_file_path = os.path.join(sub_folder,'output','volpy_demix_results_annotated.pickle')
    file = open(output_file_path, 'rb')
    data = pickle.load(file)
    traces = data['weighted_mc_denoised_traces']
    traces_raw = data['weighted_mc_traces']
    print("Original traces shape:", traces.shape)
    if np.shape(traces)[0] > np.shape(traces)[1]:
        traces = traces.T

    spikes = data['spikes_verified_array']
    spikes_volpy = data['spikes']
    good_cells = np.array(data['good_cells'])
    weights = data['weights'][good_cells,:,:]
    ROIs = data['ROIs'][good_cells]
    print(data.keys())
    mean_image_raw = data['mean_img_raw']
    mean_image = data['mean_img_mc']
    traces = traces[good_cells]
    spikes = [np.array(spikes[i]) for i in good_cells]
    spikes = np.array(spikes, dtype=object)
    spikes_volpy = [np.array(spikes_volpy[i]) for i in good_cells]
    spikes_volpy = np.array(spikes_volpy, dtype=object)

    metadata = read_metadata_file(data_folder)
    if plotflag:
        print("Traces shape:", traces.shape)
        print("Spikes shape:", spikes.shape)
        fig, axes = plt.subplots(traces.shape[0], 1, figsize=(10, 6), sharex=True, sharey=True)
        for i, trace, spks in zip(range(traces.shape[0]), traces, spikes):
            axes[i].plot(trace, label=f'Cell {i}',linewidth=0.5)    
            axes[i].scatter(spks, trace[spks], color='red', s=5, label='Spikes')
            # remove axes
            axes[i].axis('off')
        # tight layout
        plt.tight_layout()
        plt.show()
    return traces, traces_raw, spikes, spikes_volpy, weights, ROIs, mean_image, mean_image_raw, metadata

def read_metadata_file(data_folder):
    sub_folder = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, f)) and 'TS' not in f]
    sub_folder.sort()
    sub_folder = sub_folder[0]
    metadata_files = glob.glob(os.path.join(sub_folder, 'output','*.pkl'))
    with open(metadata_files[0], 'rb') as f:
        metadata = pickle.load(f)
    print("Metadata loaded from:", metadata_files[0])
    return metadata

def apply_boxcar_filter(data, window_size):
    """
    Apply a boxcar filter to the data.
    
    Args:
        data (np.array): The data to be filtered.
        window_size (int): The size of the boxcar window.
        
    Returns:
        np.array: The filtered data.
    """
    return uniform_filter1d(data, size=window_size)

# Detect epochs where speed is above the threshold and duration is at least 1 second
def detect_speed_epochs(speed, threshold, min_duration=50):
    """
    Detect epochs where speed is above a certain threshold and duration is at least min_duration.
    
    Args:
        speed (np.array): The speed of the mouse.
        threshold (float): The speed threshold.
        min_duration (int): Minimum duration in milliseconds for an epoch to be considered valid.
        
    Returns:
        list: A list of tuples, each containing the start and end indices of a detected epoch.
    """
    above_threshold = speed > threshold
    epochs = []
    
    start = None
    for i in range(len(above_threshold)):
        if above_threshold[i] and start is None:
            start = i  # Start of a new epoch
        elif not above_threshold[i] and start is not None:
            if i - start >= min_duration:  # Check if the epoch is long enough
                epochs.append((start, i - 1))
            start = None  # Reset start for the next epoch
    
    # Check if we ended with an open epoch
    if start is not None and len(speed) - start >= min_duration:
        epochs.append((start, len(speed) - 1))
    
    return epochs

# align behavior and neural data

def align_neural_behav_data(thorsync_data, positions, hd_angles, traces, traces_raw, spikes, spikes_volpy, ratio_width, ratio_height, metadata, frames_truncated=500, truncate_neural=True, speed_smooth_window=100, remove_speed_outliers=True, plotflag=False):
    # note hd_angles is in degrees

    # frames_truncated = metadata['frames_truncated']
    # print(f"Frames truncated from the beginning: {frames_truncated}")
    
    ts_behav = thorsync_data['flir_frame_times']
    ts_neural = thorsync_data['frameout2_peak_times']
    
    behav_data_length = positions.shape[0]
    neural_data_length = traces.shape[1]
    x = positions[:, 0]  # x positions
    y = positions[:, 1]  # y positions

    if len(ts_behav) != behav_data_length:
        print(f"Warning: The length of the behavior timestamps ({len(ts_behav)}) does not match the number of behavior frames ({behav_data_length}).")
        # remake ts_behav by set ts_bahav[0] as first, then add 50 
        ts_behav = np.arange(0, behav_data_length * 50, 50) + ts_behav[0]
        print(f"ts_behav length is now {len(ts_behav)} and starts at {ts_behav[0]:.2f} ms, ends at {ts_behav[-1]:.2f} ms")
    
    ts_behav = ts_behav[:behav_data_length] # behavior data stops as soon as the number of frames reaches, this now correspond to positions 
    ts_neural = ts_neural[frames_truncated:neural_data_length+ frames_truncated] # this now corresponds to traces
    ts_neural_orig = ts_neural.copy() 
    

    neural_lag = ts_neural[-1] - ts_behav[-1]
    print(f"Neural data ends at {ts_neural[-1]:.2f} ms, behavior data ends at {ts_behav[-1]:.2f} ms")


    if neural_lag > 0:
        print(f"There are {neural_lag:.2f} ms of neural data without the behaviral data at the end.")
        if truncate_neural:
            original_length = len(ts_neural)
            # remove all neural data after the last behavior frame
            ts_neural = ts_neural[ts_neural <= ts_behav[-1]]
            traces = traces[:, :len(ts_neural)]
            traces_raw = traces_raw[:, :len(ts_neural)]
            if len(spikes) > 0:
                for i in range(len(spikes)):
                    spikes[i] = spikes[i][spikes[i] < len(ts_neural)]
            if len(spikes_volpy) > 0:
                for i in range(len(spikes_volpy)):
                    spikes_volpy[i] = spikes_volpy[i][spikes_volpy[i] < len(ts_neural)]
            

            print(f"Neural data truncated from {original_length} to {len(ts_neural)} frames.")

    # Now interpolate behavior data to match neural timestamps
    x_interp = interp1d(ts_behav, x, bounds_error=False, fill_value='extrapolate')
    y_interp = interp1d(ts_behav, y, bounds_error=False, fill_value='extrapolate')
    hd_angles_interp = interp1d(ts_behav, hd_angles, bounds_error=False, fill_value='extrapolate')
    x_neural = x_interp(ts_neural) 
    y_neural = y_interp(ts_neural) 
    hd_angles_neural = hd_angles_interp(ts_neural)
    # wrap hd_angles to 0-360
    hd_angles_neural = hd_angles_neural % 360

    speed = np.sqrt(np.diff(x_neural, prepend=x_neural[0])**2 + np.diff(y_neural, prepend=y_neural[0])**2) / np.mean(np.diff(ts_neural))*1000  # cm/s
    speed_smoothed = apply_boxcar_filter(speed, speed_smooth_window)
    
    # remove outliers in speed
    if remove_speed_outliers:
        speed_threshold = 50
        remove_window = 10
        epochs_outliers = detect_speed_epochs(speed_smoothed, speed_threshold, min_duration=1)
        for start, end in epochs_outliers:
            x_neural[start-remove_window:end+remove_window] = np.nan
            y_neural[start-remove_window:end+remove_window] = np.nan
        x_neural = pd.Series(x_neural).interpolate(method='nearest').to_numpy()
        y_neural = pd.Series(y_neural).interpolate(method='nearest').to_numpy()
        speed = np.sqrt(np.diff(x_neural, prepend=x_neural[0])**2 + np.diff(y_neural, prepend=y_neural[0])**2) / np.mean(np.diff(ts_neural))*1000  # cm/s
        speed_smoothed = apply_boxcar_filter(speed, speed_smooth_window)

    frame_rate = 1000/np.mean(np.diff(ts_neural))

    # round frame_rate to 0 decimal places
    frame_rate = round(frame_rate)

    if plotflag:
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        axes[0].plot(ts_neural, x_neural, label='Interpolated X Position', linewidth=0.5)
        axes[0].plot(ts_neural, y_neural, label='Interpolated Y Position', linewidth=0.5)
        axes[0].plot(ts_neural, speed, label='Speed (cm/s)', linewidth=0.5)
        axes[0].plot(ts_neural, speed_smoothed, label='Speed (cm/s)', linewidth=0.5)

        axes[0].set_ylabel('Position (cm)')
        axes[0].legend()

        for i, trace, spks in zip(range(traces.shape[0]), traces, spikes):
            # normalize trace to 0-1 range
            trace = (trace - np.min(trace)) / (np.max(trace) - np.min(trace))
            axes[1].plot(ts_neural, trace+i, label=f'Cell {i}', linewidth=0.5)
            axes[1].scatter(ts_neural[spks], trace[spks]+i, color='red', s=2, label='Spikes' if i == 0 else None)
            
        speed_epochs = detect_speed_epochs(speed_smoothed, 5)
        # Plot the speed epochs
        for start, end in speed_epochs:
            start = ts_neural[start]
            end = ts_neural[end]
            axes[1].axvspan(start, end, color='yellow', alpha=0.3, label='Speed Epochs' if start == speed_epochs[0][0] else "")
            axes[0].axvspan(start, end, color='yellow', alpha=0.3, label='Speed Epochs' if start == speed_epochs[0][0] else "")
            
    frames_orig_behav = np.arange(behav_data_length)
    frames_orig_neural = np.arange(neural_data_length)

    
    frames_behav = frames_orig_behav[(ts_behav >= ts_neural[0]) & (ts_behav <= ts_neural[-1])]
    frames_neural = frames_orig_neural[(ts_neural_orig >= ts_neural[0]) & (ts_neural_orig <= ts_neural[-1])]
    ts_behav = ts_behav[(ts_behav >= ts_neural[0]) & (ts_behav <= ts_neural[-1])] 
    
    aligned_data = {
        'ts_neural': ts_neural,
        'x_neural': x_neural,
        'y_neural': y_neural,
        'hd_angles_neural': hd_angles_neural,
        'traces': traces,
        'traces_raw': traces_raw,
        'spikes': spikes,
        'spikes_volpy': spikes_volpy,
        'speed': speed,
        'speed_smoothed': speed_smoothed,
        'frame_rate': frame_rate,
        'ts_behav': ts_behav,
        'frames_behav': frames_behav,
        'frames_neural': frames_neural
    }
    
    return aligned_data


def _normalize_per_cell_list(data, n_cells):
    if data is None:
        return [None] * int(n_cells)
    seq = list(data)
    if len(seq) < n_cells:
        seq = seq + [None] * (int(n_cells) - len(seq))
    elif len(seq) > n_cells:
        seq = seq[:int(n_cells)]
    return seq


def _index_map_array(indices, inv_map):
    arr = np.asarray(indices, dtype=np.int64).reshape(-1)
    if arr.size == 0:
        return np.array([], dtype=np.int64)
    valid = (arr >= 0) & (arr < inv_map.size)
    mapped = np.full(arr.shape, -1, dtype=np.int64)
    mapped[valid] = inv_map[arr[valid]]
    mapped = mapped[mapped >= 0]
    if mapped.size == 0:
        return np.array([], dtype=np.int64)
    return np.sort(np.unique(mapped))


def _remap_spike_list_per_cell(data, inv_map, n_cells):
    seq = _normalize_per_cell_list(data, n_cells)
    out = []
    for x in seq:
        if x is None:
            out.append(np.array([], dtype=np.int64))
            continue
        arr = np.asarray(x)
        if arr.dtype == bool:
            arr = np.flatnonzero(arr)
        out.append(_index_map_array(arr, inv_map))
    return out


def _subset_frame_vector(vec, keep_idx, n_src):
    if vec is None:
        return None
    arr = np.asarray(vec)
    if arr.ndim == 0:
        return vec
    if arr.shape[0] == n_src:
        return arr[keep_idx]
    if arr.shape[0] == keep_idx.size:
        return arr
    if arr.shape[0] > keep_idx.size:
        return arr[:keep_idx.size]
    return arr


def _remap_complex_bursts_dict_one(d, inv_map, keep_idx, n_src):
    if d is None:
        return None
    out = dict(d)

    for k in ["trace_mf", "trace_lp", "trace_bl_subtracted", "fitted_baseline", "trace"]:
        if k in out:
            out[k] = _subset_frame_vector(out.get(k), keep_idx, n_src)

    starts_raw = np.asarray(out.get("starts", []), dtype=np.int64).reshape(-1)
    ends_raw = np.asarray(out.get("ends", []), dtype=np.int64).reshape(-1)
    n_bursts = min(starts_raw.size, ends_raw.size)
    starts_raw = starts_raw[:n_bursts]
    ends_raw = ends_raw[:n_bursts]

    # Keep burst-wise correspondence (do not use unique/sort on starts/ends pairs).
    starts_tmp = np.full(n_bursts, -1, dtype=np.int64)
    ends_tmp = np.full(n_bursts, -1, dtype=np.int64)
    valid_s = (starts_raw >= 0) & (starts_raw < inv_map.size)
    valid_e = (ends_raw >= 0) & (ends_raw < inv_map.size)
    starts_tmp[valid_s] = inv_map[starts_raw[valid_s]]
    ends_tmp[valid_e] = inv_map[ends_raw[valid_e]]
    keep_burst = (starts_tmp >= 0) & (ends_tmp >= 0) & (ends_tmp >= starts_tmp)

    out["starts"] = starts_tmp[keep_burst]
    out["ends"] = ends_tmp[keep_burst]

    burst_keys_same_len = ["durations_ms", "amplitudes", "baselines", "peaks"]
    for k in burst_keys_same_len:
        if k in out:
            a = np.asarray(out.get(k)).reshape(-1)
            if a.size >= n_bursts:
                out[k] = a[:n_bursts][keep_burst]

    for k in ["complex_bursts", "locs"]:
        if k in out:
            a = np.asarray(out.get(k), dtype=np.int64).reshape(-1)
            if a.size >= n_bursts:
                tmp = np.full(n_bursts, -1, dtype=np.int64)
                a = a[:n_bursts]
                valid = (a >= 0) & (a < inv_map.size)
                tmp[valid] = inv_map[a[valid]]
                out[k] = tmp[keep_burst]
            else:
                out[k] = _index_map_array(a, inv_map)

    return out


def _remap_plateaus_dict_one(d, inv_map, keep_idx, n_src):
    if d is None:
        return None
    out = dict(d)
    starts = np.asarray(out.get("starts", []), dtype=np.int64).reshape(-1)
    ends = np.asarray(out.get("ends", []), dtype=np.int64).reshape(-1)
    n_pl = min(starts.size, ends.size)
    starts = starts[:n_pl]
    ends = ends[:n_pl]

    starts_tmp = np.full(n_pl, -1, dtype=np.int64)
    ends_tmp = np.full(n_pl, -1, dtype=np.int64)
    valid_s = (starts >= 0) & (starts < inv_map.size)
    valid_e = (ends >= 0) & (ends < inv_map.size)
    starts_tmp[valid_s] = inv_map[starts[valid_s]]
    ends_tmp[valid_e] = inv_map[ends[valid_e]]
    keep_pl = (starts_tmp >= 0) & (ends_tmp >= 0) & (ends_tmp >= starts_tmp)

    out["starts"] = starts_tmp[keep_pl]
    out["ends"] = ends_tmp[keep_pl]

    for k in ["durations_ms", "amplitudes", "baselines", "peaks", "n_spikes"]:
        if k in out:
            a = np.asarray(out.get(k)).reshape(-1)
            if a.size >= n_pl:
                out[k] = a[:n_pl][keep_pl]

    if "locs" in out:
        a = np.asarray(out.get("locs"), dtype=np.int64).reshape(-1)
        if a.size >= n_pl:
            a = a[:n_pl]
            tmp = np.full(n_pl, -1, dtype=np.int64)
            valid = (a >= 0) & (a < inv_map.size)
            tmp[valid] = inv_map[a[valid]]
            out["locs"] = tmp[keep_pl]
        else:
            out["locs"] = _index_map_array(a, inv_map)

    spk_idx = out.get("spike_indices", None)
    if spk_idx is not None:
        spk_idx = list(spk_idx)
        remapped = []
        for i in range(min(len(spk_idx), n_pl)):
            if not keep_pl[i]:
                continue
            remapped.append(_index_map_array(spk_idx[i], inv_map))
        out["spike_indices"] = remapped
        out["n_spikes"] = np.asarray([len(x) for x in remapped], dtype=np.int64)
    else:
        out["spike_indices"] = []

    return out


def _remap_burst_metrics_one(metrics, inv_map, frame_rate, all_spikes_mapped):
    if metrics is None:
        return None
    all_spikes_mapped = np.asarray(all_spikes_mapped, dtype=np.int64)
    out = []
    for m in list(metrics):
        if not isinstance(m, dict):
            continue
        s = m.get("start", None)
        e = m.get("end", None)
        if s is None or e is None:
            continue
        s = int(s)
        e = int(e)
        if s < 0 or e < 0 or s >= inv_map.size or e >= inv_map.size:
            continue
        s2 = int(inv_map[s])
        e2 = int(inv_map[e])
        if s2 < 0 or e2 < 0 or e2 < s2:
            continue

        m2 = dict(m)
        m2["start"] = s2
        m2["end"] = e2
        if frame_rate is not None and frame_rate > 0:
            m2["duration_ms"] = (e2 - s2 + 1) * 1000.0 / float(frame_rate)
        if all_spikes_mapped.size > 0:
            m2["n_spikes"] = int(np.sum((all_spikes_mapped >= s2) & (all_spikes_mapped <= e2)))
        out.append(m2)
    return out


def align_neural_behav_data_withCS(
    thorsync_data,
    positions,
    hd_angles,
    traces,
    traces_raw,
    spikes,
    spikes_volpy,
    ratio_width,
    ratio_height,
    metadata,
    cs_session_data=None,
    frames_truncated=500,
    truncate_neural=True,
    speed_smooth_window=100,
    remove_speed_outliers=True,
    plotflag=False,
):
    """
    Align neural/behavior data and apply the same frame truncation to CS outputs.

    `cs_session_data` is expected to be session-local (same frame basis as `traces`
    before behavior alignment). Any frames dropped during alignment are dropped
    consistently from spikes, bursts, SNR traces, and burst metrics.
    """
    n_src = int(np.asarray(traces).shape[1])
    aligned = align_neural_behav_data(
        thorsync_data,
        positions,
        hd_angles,
        traces,
        traces_raw,
        spikes,
        spikes_volpy,
        ratio_width,
        ratio_height,
        metadata,
        frames_truncated=frames_truncated,
        truncate_neural=truncate_neural,
        speed_smooth_window=speed_smooth_window,
        remove_speed_outliers=remove_speed_outliers,
        plotflag=plotflag,
    )

    n_cells = int(np.asarray(aligned["traces"]).shape[0])
    n_keep = int(np.asarray(aligned["traces"]).shape[1])
    keep_idx = np.asarray(aligned.get("frames_neural", np.arange(n_keep)), dtype=np.int64).reshape(-1)
    if keep_idx.size != n_keep:
        keep_idx = np.arange(n_keep, dtype=np.int64)
    keep_valid = (keep_idx >= 0) & (keep_idx < n_src)
    if not np.all(keep_valid):
        keep_idx = keep_idx[keep_valid]
    if keep_idx.size != n_keep:
        # Fallback to direct 0..n_keep-1 map when frames_neural is malformed.
        keep_idx = np.arange(n_keep, dtype=np.int64)

    inv_map = np.full(n_src, -1, dtype=np.int64)
    inv_map[keep_idx] = np.arange(keep_idx.size, dtype=np.int64)

    if cs_session_data is None:
        aligned["complex_bursts_dicts"] = [None] * n_cells
        aligned["refined_SS"] = [np.asarray(s, dtype=np.int64) for s in aligned["spikes"]]
        aligned["all_CS_spikes"] = [np.array([], dtype=np.int64) for _ in range(n_cells)]
        aligned["all_spikes"] = [np.asarray(s, dtype=np.int64) for s in aligned["spikes"]]
        aligned["spike_heights_interpolated"] = [None] * n_cells
        aligned["SNR_interpolated"] = [None] * n_cells
        aligned["traces_SNR_interpolated"] = [None] * n_cells
        aligned["Vm_SNR_interpolated"] = [None] * n_cells
        aligned["burst_metrics"] = [None] * n_cells
        aligned["plateaus_dicts"] = [None] * n_cells
        return aligned

    cs = dict(cs_session_data)

    refined_SS = _remap_spike_list_per_cell(cs.get("refined_SS"), inv_map, n_cells)
    all_CS_spikes = _remap_spike_list_per_cell(cs.get("all_CS_spikes"), inv_map, n_cells)
    all_spikes = _remap_spike_list_per_cell(cs.get("all_spikes"), inv_map, n_cells)

    if cs.get("all_spikes") is None:
        all_spikes = []
        for ss, cs_sp in zip(refined_SS, all_CS_spikes):
            if ss.size > 0 and cs_sp.size > 0:
                all_spikes.append(np.sort(np.unique(np.concatenate([ss, cs_sp]))))
            elif ss.size > 0:
                all_spikes.append(ss.copy())
            else:
                all_spikes.append(cs_sp.copy())

    def _remap_vectors(key):
        seq = _normalize_per_cell_list(cs.get(key), n_cells)
        return [_subset_frame_vector(v, keep_idx, n_src) for v in seq]

    complex_bursts_dicts = _normalize_per_cell_list(cs.get("complex_bursts_dicts"), n_cells)
    complex_bursts_dicts = [
        _remap_complex_bursts_dict_one(d, inv_map, keep_idx, n_src) for d in complex_bursts_dicts
    ]

    plateaus_dicts = _normalize_per_cell_list(cs.get("plateaus_dicts"), n_cells)
    plateaus_dicts = [
        _remap_plateaus_dict_one(d, inv_map, keep_idx, n_src) for d in plateaus_dicts
    ]

    burst_metrics_in = _normalize_per_cell_list(cs.get("burst_metrics"), n_cells)
    burst_metrics = [
        _remap_burst_metrics_one(burst_metrics_in[i], inv_map, aligned.get("frame_rate", None), all_spikes[i])
        for i in range(n_cells)
    ]

    aligned["complex_bursts_dicts"] = complex_bursts_dicts
    aligned["refined_SS"] = refined_SS
    aligned["all_CS_spikes"] = all_CS_spikes
    aligned["all_spikes"] = all_spikes
    aligned["spike_heights_interpolated"] = _remap_vectors("spike_heights_interpolated")
    aligned["SNR_interpolated"] = _remap_vectors("SNR_interpolated")
    aligned["traces_SNR_interpolated"] = _remap_vectors("traces_SNR_interpolated")
    aligned["Vm_SNR_interpolated"] = _remap_vectors("Vm_SNR_interpolated")
    aligned["burst_metrics"] = burst_metrics
    aligned["plateaus_dicts"] = plateaus_dicts
    return aligned


def spike_map(spikes, x_neural, y_neural, speed_smoothed, speed_threshold = 5):

    mouse_positions_at_spikes = []
    valid_indices = np.where(speed_smoothed > speed_threshold)[0]

    for i, spike_frames in enumerate(spikes):
        spike_frames = spike_frames[np.isin(spike_frames, valid_indices)]
        # Interpolate the mouse position at the spike times
        x_spike = x_neural[spike_frames]
        y_spike = y_neural[spike_frames]
        mouse_positions_at_spikes.append((x_spike, y_spike))
        
    return mouse_positions_at_spikes


def compute_time_varying_snr_from_trace(
    trace,
    spks,
    complex_burst_dict=None,
    sampling_rate_hz=500.0,
    isi_threshold_ms=20.0,
    spike_baseline_points=3,
    spike_remove_points=3,
    baseline_window_seconds=10.0,
    min_points_per_baseline_window=20,
    plot_single_cell=False,
):
    """
    Compute time-varying SNR for one cell trace.

    Workflow:
    1) Categorize spikes into isolated single spikes and bursts (by ISI threshold).
    2) Use isolated spikes + first spike of each burst to estimate spike height and fit
       an exponential decay model over time.
     3) Remove +/- `spike_remove_points` around all spikes and exclude complex-burst
         intervals, compute baseline noise (std) every `baseline_window_seconds`, and
         fit an exponential increase model.
    4) Compute time-varying SNR = fitted spike-height / fitted baseline-noise.
    """

    tr = np.asarray(trace, dtype=float).reshape(-1)
    n_samples = tr.size
    if n_samples == 0:
        raise ValueError("trace is empty")

    fs = float(sampling_rate_hz)
    if fs <= 0:
        raise ValueError("sampling_rate_hz must be > 0")

    spk_arr = np.asarray(spks)
    if spk_arr.dtype == bool:
        spk_arr = np.flatnonzero(spk_arr)
    spk_arr = np.asarray(spk_arr, dtype=int).reshape(-1)
    spk_arr = spk_arr[(spk_arr >= 0) & (spk_arr < n_samples)]
    if spk_arr.size > 0:
        spk_arr = np.sort(np.unique(spk_arr))

    isi_thresh_frames = max(1, int(round((float(isi_threshold_ms) / 1000.0) * fs)))

    spike_groups = []
    if spk_arr.size > 0:
        group_start = 0
        for i in range(1, spk_arr.size):
            if (spk_arr[i] - spk_arr[i - 1]) > isi_thresh_frames:
                spike_groups.append(spk_arr[group_start:i])
                group_start = i
        spike_groups.append(spk_arr[group_start:])

    isolated_spikes = []
    burst_first_spikes = []
    bursts = []
    for g in spike_groups:
        if g.size == 1:
            isolated_spikes.append(int(g[0]))
        else:
            bursts.append(g.astype(int))
            burst_first_spikes.append(int(g[0]))

    isolated_spikes = np.asarray(isolated_spikes, dtype=int)
    burst_first_spikes = np.asarray(burst_first_spikes, dtype=int)
    selected_spikes = np.sort(np.unique(np.concatenate([isolated_spikes, burst_first_spikes]))) if (
        isolated_spikes.size + burst_first_spikes.size
    ) > 0 else np.array([], dtype=int)

    def _exp_decay(t, a, tau, c):
        return a * np.exp(-t / tau) + c

    def _exp_increase(t, a, tau, c):
        return c + a * (1.0 - np.exp(-t / tau))

    t_sec_full = np.arange(n_samples, dtype=float) / fs

    # ---------- spike height (measured + fitted decay) ----------
    sel_heights = np.full(selected_spikes.shape, np.nan, dtype=float)
    pre_pts = int(max(1, spike_baseline_points))
    for i, s in enumerate(selected_spikes):
        s0 = max(0, int(s) - pre_pts)
        if s0 >= int(s):
            continue
        local_min = np.nanmin(tr[s0:int(s)])
        sel_heights[i] = tr[int(s)] - local_min

    valid_h = np.isfinite(sel_heights)
    fit_params_spike = np.array([np.nan, np.nan, np.nan], dtype=float)
    spike_height_fit = np.full(n_samples, np.nan, dtype=float)

    if np.any(valid_h):
        t_h = selected_spikes[valid_h].astype(float) / fs
        y_h = sel_heights[valid_h]

        if y_h.size >= 3 and np.nanmax(y_h) > np.nanmin(y_h):
            a0 = float(max(1e-9, np.nanmax(y_h) - np.nanmin(y_h)))
            tau0 = float(max(1e-3, (t_h[-1] - t_h[0]) / 2.0 if t_h.size > 1 else 1.0))
            c0 = float(np.nanmin(y_h))
            try:
                popt, _ = curve_fit(
                    _exp_decay,
                    t_h,
                    y_h,
                    p0=[a0, tau0, c0],
                    bounds=([0.0, 1e-6, -np.inf], [np.inf, np.inf, np.inf]),
                    maxfev=20000,
                )
                fit_params_spike = np.asarray(popt, dtype=float)
                spike_height_fit = _exp_decay(t_sec_full, *fit_params_spike)
            except Exception:
                med = float(np.nanmedian(y_h))
                fit_params_spike = np.array([0.0, np.inf, med], dtype=float)
                spike_height_fit = np.full(n_samples, med, dtype=float)
        else:
            med = float(np.nanmedian(y_h))
            fit_params_spike = np.array([0.0, np.inf, med], dtype=float)
            spike_height_fit = np.full(n_samples, med, dtype=float)

    # ---------- baseline noise (measured + fitted increase) ----------
    mask_keep = np.ones(n_samples, dtype=bool)
    rm_pts = int(max(0, spike_remove_points))
    for s in spk_arr:
        lo = max(0, int(s) - rm_pts)
        hi = min(n_samples, int(s) + rm_pts + 1)
        mask_keep[lo:hi] = False

    burst_dict_for_mask = complex_burst_dict

    burst_intervals = []
    if isinstance(burst_dict_for_mask, dict):
        starts = np.asarray(burst_dict_for_mask.get("starts", []), dtype=int).reshape(-1)
        ends = np.asarray(burst_dict_for_mask.get("ends", []), dtype=int).reshape(-1)
        n_b = min(starts.size, ends.size)
        if n_b > 0:
            starts = starts[:n_b]
            ends = ends[:n_b]
            for s_b, e_b in zip(starts, ends):
                lo_b = int(min(s_b, e_b))
                hi_b = int(max(s_b, e_b))
                if hi_b < 0 or lo_b >= n_samples:
                    continue
                lo_b = max(0, lo_b)
                hi_b = min(n_samples - 1, hi_b)
                mask_keep[lo_b:hi_b + 1] = False
                burst_intervals.append((lo_b, hi_b))

    spike_removed_trace = tr.copy()
    spike_removed_trace[~mask_keep] = np.nan

    win = int(max(1, round(float(baseline_window_seconds) * fs)))
    t_baseline = []
    baseline_std = []
    for start in range(0, n_samples, win):
        end = min(n_samples, start + win)
        seg = tr[start:end][mask_keep[start:end]]
        if seg.size >= int(min_points_per_baseline_window):
            baseline_std.append(float(np.nanstd(seg, ddof=0)))
            t_baseline.append(((start + end) / 2.0) / fs)
        else:
            baseline_std.append(np.nan)
            t_baseline.append(((start + end) / 2.0) / fs)

    t_baseline = np.asarray(t_baseline, dtype=float)
    baseline_std = np.asarray(baseline_std, dtype=float)

    valid_b = np.isfinite(baseline_std)
    fit_params_baseline = np.array([np.nan, np.nan, np.nan], dtype=float)
    baseline_noise_fit = np.full(n_samples, np.nan, dtype=float)

    if np.any(valid_b):
        t_b = t_baseline[valid_b]
        y_b = baseline_std[valid_b]

        if y_b.size >= 3 and np.nanmax(y_b) > np.nanmin(y_b):
            a0 = float(max(1e-9, np.nanmax(y_b) - np.nanmin(y_b)))
            tau0 = float(max(1e-3, (t_b[-1] - t_b[0]) / 2.0 if t_b.size > 1 else 1.0))
            c0 = float(np.nanmin(y_b))
            try:
                popt, _ = curve_fit(
                    _exp_increase,
                    t_b,
                    y_b,
                    p0=[a0, tau0, c0],
                    bounds=([0.0, 1e-6, 0.0], [np.inf, np.inf, np.inf]),
                    maxfev=20000,
                )
                fit_params_baseline = np.asarray(popt, dtype=float)
                baseline_noise_fit = _exp_increase(t_sec_full, *fit_params_baseline)
            except Exception:
                med = float(np.nanmedian(y_b))
                fit_params_baseline = np.array([0.0, np.inf, med], dtype=float)
                baseline_noise_fit = np.full(n_samples, med, dtype=float)
        else:
            med = float(np.nanmedian(y_b))
            fit_params_baseline = np.array([0.0, np.inf, med], dtype=float)
            baseline_noise_fit = np.full(n_samples, med, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        snr_t = spike_height_fit / baseline_noise_fit
    snr_t[~np.isfinite(snr_t)] = np.nan

    if plot_single_cell:
        fig, axes = plt.subplots(3, 1, figsize=(10, 5), sharex=True)

        axes[0].plot(t_sec_full, tr, color="black", linewidth=0.8, label="trace")
        if spk_arr.size > 0:
            axes[0].scatter(
                t_sec_full[spk_arr],
                tr[spk_arr],
                s=8,
                color="tab:red",
                alpha=0.8,
                label="spikes",
            )
        axes[0].plot(
            t_sec_full,
            spike_height_fit,
            color="tab:blue",
            linewidth=1.2,
            label="fitted spike height",
        )
        if np.any(valid_h):
            axes[0].scatter(
                selected_spikes[valid_h] / fs,
                sel_heights[valid_h],
                s=14,
                color="tab:blue",
                alpha=0.8,
                label="measured spike height",
            )
        axes[0].set_title("Trace + spikes + fitted time-varying spike height")
        axes[0].set_ylabel("signal")
        axes[0].legend(loc="upper right", fontsize=8)

        if t_baseline.size > 0:
            axes[1].scatter(
                t_baseline,
                baseline_std,
                s=16,
                color="tab:orange",
                alpha=0.8,
                label="baseline std (10s windows)",
            )
        axes[1].plot(
            t_sec_full,
            baseline_noise_fit,
            color="tab:green",
            linewidth=1.2,
            label="fitted baseline noise",
        )
        axes[1].set_ylabel("noise std")
        axes[1].set_title("Baseline noise: measured vs fitted")
        axes[1].legend(loc="upper right", fontsize=8)

        axes[2].plot(t_sec_full, snr_t, color="tab:purple", linewidth=1.2)
        axes[2].set_title("Time-varying SNR")
        axes[2].set_ylabel("SNR")
        axes[2].set_xlabel("Time (s)")

        plt.tight_layout()
        plt.show()

    return {
        "sampling_rate_hz": fs,
        "isi_threshold_ms": float(isi_threshold_ms),
        "isolated_single_spikes": isolated_spikes,
        "burst_first_spikes": burst_first_spikes,
        "burst_groups": bursts,
        "selected_spikes_for_height": selected_spikes,
        "selected_spike_heights": sel_heights,
        "spike_height_fit": spike_height_fit,
        "spike_height_fit_params": {
            "a": float(fit_params_spike[0]) if np.isfinite(fit_params_spike[0]) else np.nan,
            "tau": float(fit_params_spike[1]) if np.isfinite(fit_params_spike[1]) else np.nan,
            "c": float(fit_params_spike[2]) if np.isfinite(fit_params_spike[2]) else np.nan,
        },
        "spike_removed_trace": spike_removed_trace,
        "excluded_complex_burst_intervals": burst_intervals,
        "baseline_window_times_s": t_baseline,
        "baseline_std": baseline_std,
        "baseline_noise_fit": baseline_noise_fit,
        "baseline_noise_fit_params": {
            "a": float(fit_params_baseline[0]) if np.isfinite(fit_params_baseline[0]) else np.nan,
            "tau": float(fit_params_baseline[1]) if np.isfinite(fit_params_baseline[1]) else np.nan,
            "c": float(fit_params_baseline[2]) if np.isfinite(fit_params_baseline[2]) else np.nan,
        },
        "snr_time_varying": snr_t,
        "time_seconds": t_sec_full,
    }
