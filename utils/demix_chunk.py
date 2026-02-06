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
from scipy.ndimage import median_filter

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
clip = 500
threshold_method = 'adaptive_threshold'
min_spikes = 20
pnorm = 0.5
threshold = 5
do_plot = False
ridge_bg = 0.01
sub_freq = 20
weight_update = 'ridge'
n_iter = 3
method = 'spikepursuit'
distance = 200
template_size = 10

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
