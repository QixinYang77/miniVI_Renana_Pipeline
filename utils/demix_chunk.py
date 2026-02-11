"""
Run VolPy demixing on a single video chunk.

This is a memory-efficient version of demix_fullROI.py that works on chunked videos.

Usage:
    python demix_chunk.py <chunk_folder>/movReg_denoised.tif

Outputs:
    - chunk_results.pickle (in the same chunk folder)
"""

import glob
import numpy as np
import os

# CaImAn imports OpenCV (cv2) on import; on some cluster nodes libGL is missing and cv2 fails to load.
# VolPy demixing in this script does not require OpenCV, so we fall back to a small stub to allow CaImAn to import.
import sys
import types

try:
    import cv2  # noqa: F401
except Exception as e:
    msg = str(e)
    if "libGL.so.1" in msg or "libGL" in msg:
        cv2_stub = types.ModuleType("cv2")

        def _noop(*_args, **_kwargs):
            return None

        cv2_stub.setNumThreads = _noop

        class _OCL(object):
            @staticmethod
            def setUseOpenCL(_flag):
                return None

        cv2_stub.ocl = _OCL()
        cv2_stub.getBuildInformation = lambda: "cv2 stub (libGL missing)"
        sys.modules["cv2"] = cv2_stub
        print(
            "[WARN] cv2 failed to import (libGL missing). Injected a stub cv2 module so CaImAn can import. "
            "If a later step needs OpenCV, install libGL / headless OpenCV.",
            flush=True,
        )
    else:
        raise

import caiman as cm
from caiman.source_extraction.volpy import utils
from caiman.source_extraction.volpy.volparams import volparams
from caiman.source_extraction.volpy.volpy import VOLPY
import tifffile
import pickle
import logging
import argparse
from scipy.ndimage import binary_dilation, label, median_filter
from scipy.sparse.linalg import svds

# ----------------- CLI ARGUMENT -----------------
parser = argparse.ArgumentParser(description='Run VolPy + demixing on a chunked TIF movie file.')
parser.add_argument('input_file', type=str, help='Path to the chunk movie file (e.g., chunk_0/movReg_denoised.tif)')
args = parser.parse_args()

fname = args.input_file
fnames = [fname]

chunk_folder = os.path.dirname(fname)
print("Chunk folder:", chunk_folder)

# ----------------- LOAD CHUNK INFO -----------------
chunk_info_file = os.path.join(chunk_folder, 'chunk_info.pkl')
with open(chunk_info_file, 'rb') as f:
    chunk_info = pickle.load(f)

chunk_idx = chunk_info['chunk_idx']
original_roi_indices = chunk_info['roi_indices']
bbox = chunk_info['bbox']

print(f"Processing chunk {chunk_idx}")
print(f"Original ROI indices: {original_roi_indices}")
print(f"Bounding box: {bbox}")

# ----------------- ROI LOADING -----------------
ROI_file = os.path.join(chunk_folder, 'manual_ROIs.npz')
print("Loading ROIs from:", ROI_file)
with np.load(ROI_file) as data:
    ROIs = data["ROIs"].astype(bool)

# Ensure ROIs has shape (N, H, W)
if ROIs.ndim == 2:
    ROIs = ROIs[np.newaxis, ...]
print("ROIs shape:", ROIs.shape)

fr = 500  # frame rate (Hz)

# ----------------- CLUSTER SETUP -----------------
c, dview, n_processes = cm.cluster.setup_cluster(
    backend='local', n_processes=None, single_thread=False
)

# ----------------- VOLPY PARAMS -----------------
opts_dict = {
    'fr': fr,
}
opts = volparams(params_dict=opts_dict)

# parameters for trace denoising and spike extraction
ROIselect = ROIs
index = list(range(len(ROIselect)))
weights_init = None

context_size = 12
censor_size = 8
nPC_bg = 8
visualize_ROI = False
flip_signal = False
hp_freq_pb = 3
hp_freq = 20
clip = 100
threshold_method = 'adaptive_threshold'
min_spikes = 10
pnorm = 0.5
threshold = 5
do_plot = False
ridge_bg = 0.01
sub_freq = 20
weight_update = 'ridge'
n_iter = 3
method = 'spikepursuit'
distance = 200
template_size = 20

opts_dict = {
    'fnames': fnames,
    'fr': fr,
    'ROIs': ROIselect,
    'index': index,
    'weights': weights_init,
    'context_size': context_size,
    'censor_size': censor_size,
    'nPC_bg': nPC_bg,
    'visualize_ROI': visualize_ROI,
    'flip_signal': flip_signal,
    'hp_freq_pb': hp_freq_pb,
    'hp_freq': hp_freq,
    'clip': clip,
    'threshold_method': threshold_method,
    'min_spikes': min_spikes,
    'pnorm': pnorm,
    'threshold': threshold,
    'do_plot': do_plot,
    'ridge_bg': ridge_bg,
    'sub_freq': sub_freq,
    'weight_update': weight_update,
    'n_iter': n_iter,
    'method': method,
    'distance': distance,
    'template_size': template_size,
}

opts.change_params(params_dict=opts_dict)

# ----------------- RUN VOLPY -----------------
vpy = VOLPY(n_processes=n_processes, dview=dview, params=opts)
vpy.fit(n_processes=n_processes, dview=dview)

# ----------------- VIDEO PATHS -----------------
raw_video_files = glob.glob(os.path.join(chunk_folder, '*movie.tif'))
mc_video_file = os.path.join(chunk_folder, 'movReg.tif')
mc_denoised_video_file = fname

has_raw = len(raw_video_files) > 0
has_mc = os.path.exists(mc_video_file)

if has_raw:
    raw_video_file = raw_video_files[0]
    print("Raw video:", raw_video_file)
if has_mc:
    print("MC video:", mc_video_file)
print("MC denoised video:", mc_denoised_video_file)

# ----------------- HELPER FUNCTIONS -----------------
def compute_mean_traces(video_file, ROIs, *, n_frames=None):
    """Compute simple mean traces (per ROI) from a video."""
    key = None
    if n_frames is not None:
        key = slice(0, int(n_frames))
    video = tifffile.imread(video_file, key=key) if key is not None else tifffile.imread(video_file)
    T = int(video.shape[0])
    traces = []
    for roi in ROIs:
        trace = np.mean(video[:, roi > 0], axis=1)
        traces.append(trace)
    traces = np.asarray(traces)
    return traces


def compute_weighted_traces(video_file, weights, *, n_frames=None):
    """Compute demixed traces using VolPy weights."""
    key = None
    if n_frames is not None:
        key = slice(0, int(n_frames))
    video = tifffile.imread(video_file, key=key) if key is not None else tifffile.imread(video_file)
    T, H, W = video.shape
    P = H * W

    if weights.ndim == 3:
        if weights.shape[1:] == (H, W):
            W_flat = weights.reshape(weights.shape[0], P)
        elif weights.shape[0:2] == (H, W):
            W_flat = np.moveaxis(weights, -1, 0).reshape(-1, P)
        else:
            raise ValueError(f"3D weights shape {weights.shape} incompatible with video {video.shape}")
    elif weights.ndim == 2:
        if weights.shape[0] == P:
            W_flat = weights.T
        elif weights.shape[1] == P:
            W_flat = weights
        else:
            raise ValueError(f"2D weights shape {weights.shape} incompatible with H*W = {P}")
    else:
        raise ValueError(f"Unsupported weights.ndim = {weights.ndim}")

    video_flat = video.reshape(T, P)
    traces = W_flat @ video_flat.T
    return traces


def _disk_footprint(radius):
    r = int(max(0, int(radius)))
    if r <= 0:
        return np.ones((1, 1), dtype=bool)
    yy, xx = np.ogrid[-r : r + 1, -r : r + 1]
    return (xx * xx + yy * yy) <= (r * r)


def _largest_connected_component(mask):
    m = np.asarray(mask, dtype=bool)
    if not np.any(m):
        return m
    lbl, n = label(m)
    if n <= 1:
        return m
    areas = np.bincount(lbl.ravel())
    areas[0] = 0
    keep = int(np.argmax(areas))
    return lbl == keep


def _weights_to_cell_maps(weights, n_cells, h, w):
    """Convert VolPy weights into (N, H, W) maps when possible."""
    wt = np.asarray(weights)
    p = int(h * w)
    n_cells = int(n_cells)

    if wt.ndim == 3:
        if wt.shape[0] == n_cells and wt.shape[1] == h and wt.shape[2] == w:
            return wt.astype(np.float32, copy=False)
        if wt.shape[0] == h and wt.shape[1] == w and wt.shape[2] == n_cells:
            return np.moveaxis(wt, -1, 0).astype(np.float32, copy=False)
        return None

    if wt.ndim == 2:
        if wt.shape[0] == n_cells and wt.shape[1] == p:
            return wt.reshape(n_cells, h, w).astype(np.float32, copy=False)
        if wt.shape[1] == n_cells and wt.shape[0] == p:
            return wt.T.reshape(n_cells, h, w).astype(np.float32, copy=False)
        return None

    if wt.ndim == 1 and n_cells == 1 and wt.size == p:
        return wt.reshape(1, h, w).astype(np.float32, copy=False)

    return None


def _build_signal_mask_from_weight(
    weight_map,
    roi_fallback,
    *,
    min_pixels=20,
    percentiles=(70.0, 60.0, 50.0),
):
    """Adaptive positive-weight mask with ROI fallback."""
    wm = np.asarray(weight_map, dtype=np.float32)
    roi_fb = np.asarray(roi_fallback, dtype=bool)
    pos = np.isfinite(wm) & (wm > 0)

    if np.any(pos):
        vals = wm[pos]
        for pct in percentiles:
            thr = float(np.percentile(vals, float(pct)))
            cand = np.isfinite(wm) & (wm >= thr) & (wm > 0)
            cand = _largest_connected_component(cand)
            if int(np.count_nonzero(cand)) >= int(min_pixels):
                return cand, True
        # If positive-weight support exists but is sparse, keep largest positive component.
        cand = _largest_connected_component(pos)
        if np.any(cand):
            return cand, True

    return roi_fb, False


def clean_weighted_traces_background(
    weighted_traces,
    video_file,
    rois,
    weights,
    fr,
    context_size,
    censor_size,
    nPC_bg,
    ridge_bg,
    *,
    median_window_s=30.0,
):
    """VolPy-style background denoising for weighted traces.

    Fit background on a 30s median-smoothed trace, subtract fitted background from
    the original (non-filtered) weighted trace.
    """
    wtr = np.asarray(weighted_traces, dtype=np.float32)
    if wtr.ndim == 1:
        wtr = wtr[np.newaxis, :]

    rois = np.asarray(rois).astype(bool)
    if rois.ndim == 2:
        rois = rois[np.newaxis, ...]
    if rois.ndim != 3:
        raise ValueError(f"Unexpected ROIs shape: {rois.shape}")

    n_cells, t_len_full = int(wtr.shape[0]), int(wtr.shape[1])
    if n_cells != int(rois.shape[0]):
        raise ValueError(
            f"weighted_traces n_cells ({n_cells}) does not match rois ({rois.shape[0]})"
        )
    h, w = int(rois.shape[1]), int(rois.shape[2])
    weight_maps = _weights_to_cell_maps(weights, n_cells=n_cells, h=h, w=w)
    if weight_maps is None:
        print(
            "[WARN] Could not map VolPy weights to (N,H,W); using ROI masks for background cleaning.",
            flush=True,
        )

    key = slice(0, int(t_len_full))
    video = tifffile.imread(video_file, key=key).astype(np.float32, copy=False)
    if video.ndim != 3:
        raise ValueError(f"Expected 3D movie in {video_file}, got shape {video.shape}")
    t_use = int(min(int(video.shape[0]), t_len_full))
    if int(video.shape[0]) != t_len_full:
        print(
            f"[WARN] weighted trace length ({t_len_full}) vs video frames ({video.shape[0]}) mismatch; "
            f"using first {t_use} frames for cleaning and leaving remaining samples unchanged.",
            flush=True,
        )
    if t_use < 2:
        print("[WARN] Too few frames for background cleaning; returning original traces.", flush=True)
        return wtr.astype(np.float32, copy=False)
    wtr_fit = wtr[:, :t_use]
    video = video[:t_use]

    cleaned = wtr.astype(np.float32, copy=True)
    ctx = int(max(1, int(context_size)))
    censor = int(max(0, int(censor_size)))
    npc = int(max(1, int(nPC_bg)))
    alpha = float(max(0.0, float(nPC_bg) * float(ridge_bg)))

    win = int(round(float(median_window_s) * float(fr)))
    win = max(3, win)
    if win % 2 == 0:
        win += 1

    ctx_foot = np.ones((ctx, ctx), dtype=bool)
    censor_foot = _disk_footprint(censor)
    n_weight_masks = 0

    for ci in range(n_cells):
        t_raw = wtr_fit[ci].astype(np.float32, copy=False)
        t_fit = median_filter(t_raw, size=int(win)).astype(np.float32, copy=False)

        roi = rois[ci]
        if not np.any(roi):
            print(f"[WARN] Cell {ci}: empty ROI; keeping original weighted trace.", flush=True)
            continue

        if weight_maps is not None:
            signal_mask, used_weight = _build_signal_mask_from_weight(
                weight_maps[ci],
                roi,
                min_pixels=20,
                percentiles=(70.0, 60.0, 50.0),
            )
            if used_weight:
                n_weight_masks += 1
        else:
            signal_mask = roi

        bwexp = binary_dilation(signal_mask, structure=ctx_foot)
        xinds = np.where(np.any(bwexp, axis=1))[0]
        yinds = np.where(np.any(bwexp, axis=0))[0]
        if xinds.size == 0 or yinds.size == 0:
            print(f"[WARN] Cell {ci}: empty context crop; keeping original weighted trace.", flush=True)
            continue

        x0, x1 = int(xinds[0]), int(xinds[-1])
        y0, y1 = int(yinds[0]), int(yinds[-1])
        sig_crop = signal_mask[x0 : x1 + 1, y0 : y1 + 1]
        notbw = ~binary_dilation(sig_crop, structure=censor_foot)
        if not np.any(notbw):
            print(
                f"[WARN] Cell {ci}: no background pixels after censoring; keeping original weighted trace.",
                flush=True,
            )
            continue

        data_bg = video[:, x0 : x1 + 1, y0 : y1 + 1][:, notbw]
        if data_bg.ndim != 2:
            data_bg = data_bg.reshape(int(video.shape[0]), -1)
        p_bg = int(data_bg.shape[1])
        if p_bg < (npc + 1):
            print(
                f"[WARN] Cell {ci}: too few bg pixels ({p_bg}) for nPC_bg={npc}; keeping original weighted trace.",
                flush=True,
            )
            continue
        if int(data_bg.shape[0]) <= npc:
            print(
                f"[WARN] Cell {ci}: too few frames ({data_bg.shape[0]}) for nPC_bg={npc}; keeping original weighted trace.",
                flush=True,
            )
            continue

        data_bg = data_bg.astype(np.float32, copy=False)
        data_bg = data_bg - np.mean(data_bg, axis=0, keepdims=True).astype(np.float32, copy=False)

        k = int(min(npc, min(data_bg.shape) - 1))
        if k < 1:
            print(f"[WARN] Cell {ci}: invalid SVD rank for background; keeping original weighted trace.", flush=True)
            continue

        try:
            Ub, _, _ = svds(data_bg, k=k)
            Ub = Ub.astype(np.float32, copy=False)
            if Ub.shape[0] != t_use:
                print(
                    f"[WARN] Cell {ci}: Ub/time mismatch ({Ub.shape[0]} vs {t_use}); keeping original weighted trace.",
                    flush=True,
                )
                continue
            UtU = np.matmul(Ub.T, Ub).astype(np.float64, copy=False)
            if alpha > 0.0:
                UtU += np.eye(int(UtU.shape[0]), dtype=np.float64) * float(alpha)
            Uty = np.matmul(Ub.T.astype(np.float64, copy=False), t_fit.astype(np.float64, copy=False))
            try:
                coef = np.linalg.solve(UtU, Uty)
            except np.linalg.LinAlgError:
                coef = np.linalg.lstsq(UtU, Uty, rcond=None)[0]
            bg_hat = np.matmul(Ub.astype(np.float64, copy=False), coef).astype(np.float32, copy=False)
            cleaned[ci, :t_use] = (t_raw - bg_hat).astype(np.float32, copy=False)
        except Exception as e:
            print(
                f"[WARN] Cell {ci}: background denoising failed ({e!r}); keeping original weighted trace.",
                flush=True,
            )
            continue

    print(
        f"[INFO] Background cleaner used weight-derived support for {n_weight_masks}/{n_cells} cells.",
        flush=True,
    )
    return cleaned.astype(np.float32, copy=False)


def calculate_mean_image(video_file, *, n_frames=None):
    key = None
    if n_frames is not None:
        key = slice(0, int(n_frames))
    video = tifffile.imread(video_file, key=key) if key is not None else tifffile.imread(video_file)
    mean_image = np.mean(video, axis=0)
    return mean_image

# ----------------- EXTRACT VOLPY OUTPUTS -----------------
volpy_trace = vpy.estimates['t']
volpy_sub_trace = vpy.estimates['t_sub']
spikes = vpy.estimates['spikes']
snr = vpy.estimates['snr']
weights = vpy.estimates['weights']
mean_im = vpy.estimates['mean_im']

print("volpy_trace shape (raw):", np.shape(volpy_trace))
print("volpy_sub_trace shape (raw):", np.shape(volpy_sub_trace))
print("spikes shape (raw):", np.shape(spikes))
print("weights shape (raw):", np.shape(weights))
print("SNR (raw):", snr)

# Handle single neuron case
if ROIselect.shape[0] == 1:
    if volpy_trace.ndim == 1:
        volpy_trace = volpy_trace[np.newaxis, :]
    if volpy_sub_trace.ndim == 1:
        volpy_sub_trace = volpy_sub_trace[np.newaxis, :]
    if spikes.ndim == 1:
        spikes = spikes[np.newaxis, :]
    if weights.ndim == 1:
        weights = weights[np.newaxis, :]
    snr = np.atleast_1d(snr)

if mean_im.ndim == 3:
    mean_im = mean_im.squeeze()

print("volpy_trace shape (fixed):", np.shape(volpy_trace))
print("volpy_sub_trace shape (fixed):", np.shape(volpy_sub_trace))
print("spikes shape (fixed):", np.shape(spikes))
print("weights shape (fixed):", np.shape(weights))

# ----------------- COMPUTE TRACES -----------------
T_ref = int(volpy_trace.shape[-1])
mc_denoised_traces = compute_mean_traces(mc_denoised_video_file, ROIselect, n_frames=T_ref)
weighted_mc_denoised_traces = compute_weighted_traces(mc_denoised_video_file, weights, n_frames=T_ref)
weighted_mc_denoised_traces_cleaned = clean_weighted_traces_background(
    weighted_mc_denoised_traces,
    mc_denoised_video_file,
    ROIselect,
    weights,
    fr,
    context_size,
    censor_size,
    nPC_bg,
    ridge_bg,
    median_window_s=30.0,
)

if has_raw:
    raw_traces = compute_mean_traces(raw_video_file, ROIselect, n_frames=T_ref)
else:
    raw_traces = None

if has_mc:
    mc_traces = compute_mean_traces(mc_video_file, ROIselect, n_frames=T_ref)
    weighted_mc_traces = compute_weighted_traces(mc_video_file, weights, n_frames=T_ref)
else:
    # If the chunk folder doesn't include movReg.tif (e.g., when we only chunk the denoised movie),
    # fall back to using the denoised movie for "mc" traces so downstream concat/viewers still work.
    mc_traces = mc_denoised_traces
    weighted_mc_traces = weighted_mc_denoised_traces
    print(
        "[WARN] movReg.tif not found in chunk folder; using denoised movie as fallback for mc_traces.",
        flush=True,
    )

print("mc_denoised_traces shape:", mc_denoised_traces.shape)
print("weighted_mc_denoised_traces shape:", weighted_mc_denoised_traces.shape)
print(
    "weighted_mc_denoised_traces_cleaned shape:",
    weighted_mc_denoised_traces_cleaned.shape,
)

# ----------------- MEAN IMAGES -----------------
mean_image_mc_denoised = calculate_mean_image(mc_denoised_video_file, n_frames=T_ref)

if has_raw:
    mean_image_raw = calculate_mean_image(raw_video_file, n_frames=T_ref)
else:
    mean_image_raw = None

if has_mc:
    mean_image_mc = calculate_mean_image(mc_video_file, n_frames=T_ref)
else:
    mean_image_mc = mean_image_mc_denoised

# ----------------- MOTION CORRECTION SHIFTS -----------------
motion_correction_params_path = os.path.join(chunk_folder, 'motion_correction_params.npy')
if os.path.exists(motion_correction_params_path):
    motion_correction_params = np.load(motion_correction_params_path, allow_pickle=True).item()
    reg_shifts = motion_correction_params['reg_shifts']
    shift_distances = np.linalg.norm(reg_shifts, axis=1)
else:
    reg_shifts = None
    shift_distances = None

# ----------------- SAVE CHUNK RESULTS -----------------
output_file = os.path.join(chunk_folder, 'chunk_results.pickle')
with open(output_file, 'wb') as file:
    pickle.dump({
        # Chunk metadata
        'chunk_idx': chunk_idx,
        'original_roi_indices': original_roi_indices,
        'bbox': bbox,
        # VolPy results
        'volpy_trace': volpy_trace,
        'volpy_sub_trace': volpy_sub_trace,
        'spikes': spikes,
        'snr': snr,
        'weights': weights,
        # Mean images (chunk coordinates)
        'mean_img_raw': mean_image_raw,
        'mean_img_mc': mean_image_mc,
        'mean_img_mc_denoised': mean_image_mc_denoised,
        # Traces
        'raw_traces': raw_traces,
        'mc_traces': mc_traces,
        'mc_denoised_traces': mc_denoised_traces,
        'weighted_mc_traces': weighted_mc_traces,
        'weighted_mc_denoised_traces': weighted_mc_denoised_traces,
        'weighted_mc_denoised_traces_cleaned': weighted_mc_denoised_traces_cleaned,
        # ROIs (chunk coordinates)
        'ROIs': ROIselect,
        # Motion correction
        'reg_shifts': reg_shifts,
        'shift_distances': shift_distances,
        'mc_fallback_to_denoised': (not has_mc),
    }, file)

print("Done! Chunk results saved to:", output_file)

# Stop the cluster cleanly
try:
    if "dview" in locals():
        cm.stop_server(dview=dview)
except Exception:
    try:
        if "dview" in locals() and hasattr(cm, "cluster") and hasattr(cm.cluster, "stop_server"):
            cm.cluster.stop_server(dview=dview)
    except Exception:
        pass
