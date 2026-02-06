"""
Concatenate chunk results back into a single volpy_demix_results.pickle file.

This script reads all chunk_results.pickle files and combines them,
mapping ROI results back to their original indices.

Usage:
    python concat_chunk_results.py <data_folder>

Where data_folder contains the 'chunks' subdirectory with chunk results.

Outputs:
    - volpy_demix_results.pickle (in data_folder, same format as demix_fullROI.py)
"""

import argparse
import os
import glob
import numpy as np
import tifffile
import pickle


def _tiff_n_pages(path):
    with tifffile.TiffFile(path) as tif:
        return int(len(tif.pages))


def _iter_tiff_chunks(path, *, chunk_frames=2000, max_frames=None):
    n_pages = _tiff_n_pages(path)
    if max_frames is not None:
        n_pages = min(int(n_pages), int(max_frames))
    for start in range(0, n_pages, int(chunk_frames)):
        end = min(start + int(chunk_frames), n_pages)
        arr = tifffile.imread(path, key=slice(int(start), int(end)))
        if arr.ndim != 3:
            raise ValueError("Expected 3D movie chunk from %s, got %s" % (path, arr.shape))
        yield int(start), int(end), arr


def calculate_mean_image_stream(video_file, *, n_frames=None, chunk_frames=2000):
    """Compute mean image without loading the full movie into RAM."""
    acc = None
    count = 0
    for s0, s1, chunk in _iter_tiff_chunks(video_file, chunk_frames=chunk_frames, max_frames=n_frames):
        if acc is None:
            acc = np.zeros(chunk.shape[1:], dtype=np.float64)
        acc += np.sum(chunk.astype(np.float64), axis=0)
        count += int(chunk.shape[0])
    if acc is None or count <= 0:
        raise ValueError("No frames found for mean image: %s" % video_file)
    return (acc / float(count)).astype(np.float32)


def compute_mean_traces_stream(video_file, ROIs, *, n_frames=None, chunk_frames=2000):
    """Compute mean traces (N, T) from a movie and full-frame ROI masks, streaming over time."""
    ROIs = np.asarray(ROIs).astype(bool)
    if ROIs.ndim == 2:
        ROIs = ROIs[None, ...]
    if ROIs.ndim != 3:
        raise ValueError("Unexpected ROIs shape: %s" % (ROIs.shape,))
    n_rois = int(ROIs.shape[0])
    H, W = int(ROIs.shape[1]), int(ROIs.shape[2])
    P = int(H * W)

    # Determine target length
    n_pages = _tiff_n_pages(video_file)
    T_use = int(min(n_pages, int(n_frames) if n_frames is not None else n_pages))
    traces = np.full((n_rois, T_use), np.nan, dtype=np.float32)

    roi_idx = []
    for i in range(n_rois):
        idx = np.flatnonzero(ROIs[i].reshape(-1))
        roi_idx.append(idx)

    for s0, s1, chunk in _iter_tiff_chunks(video_file, chunk_frames=chunk_frames, max_frames=T_use):
        if chunk.shape[1] != H or chunk.shape[2] != W:
            raise ValueError("ROI dims (%d,%d) don't match video (%d,%d) in %s" % (H, W, chunk.shape[1], chunk.shape[2], video_file))
        flat = chunk.reshape(int(chunk.shape[0]), P)
        for i in range(n_rois):
            idx = roi_idx[i]
            if idx.size == 0:
                continue
            # mean across ROI pixels -> (Tchunk,)
            traces[i, s0:s1] = np.mean(flat[:, idx], axis=1).astype(np.float32, copy=False)

    return traces


def compute_weighted_traces_stream(video_file, weights, *, n_frames=None, chunk_frames=2000, roi_block=32):
    """Compute weighted traces (N, T) from a movie and full-frame weights, streaming over time."""
    Wt = np.asarray(weights, dtype=np.float32)
    if Wt.ndim != 3:
        raise ValueError("Expected weights (N,H,W), got %s" % (Wt.shape,))
    n_rois = int(Wt.shape[0])
    H, W = int(Wt.shape[1]), int(Wt.shape[2])
    P = int(H * W)

    n_pages = _tiff_n_pages(video_file)
    T_use = int(min(n_pages, int(n_frames) if n_frames is not None else n_pages))
    traces = np.full((n_rois, T_use), np.nan, dtype=np.float32)

    for s0, s1, chunk in _iter_tiff_chunks(video_file, chunk_frames=chunk_frames, max_frames=T_use):
        if chunk.shape[1] != H or chunk.shape[2] != W:
            raise ValueError("weights dims (%d,%d) don't match video (%d,%d) in %s" % (H, W, chunk.shape[1], chunk.shape[2], video_file))
        flat = chunk.reshape(int(chunk.shape[0]), P).astype(np.float32, copy=False)
        # Process ROIs in blocks to limit peak memory
        for r0 in range(0, n_rois, int(roi_block)):
            r1 = min(r0 + int(roi_block), n_rois)
            Wblk = Wt[r0:r1].reshape(int(r1 - r0), P).T  # (P, B)
            out = np.dot(flat, Wblk)  # (Tchunk, B)
            traces[r0:r1, s0:s1] = out.T.astype(np.float32, copy=False)

    return traces


def main():
    parser = argparse.ArgumentParser(description='Concatenate chunk results.')
    parser.add_argument('data_folder', type=str, help='Path to data folder containing chunks/')
    parser.add_argument(
        '--trace-chunk-frames',
        type=int,
        default=2000,
        help='Frames per read chunk when recomputing traces/mean images from full movies.',
    )
    parser.add_argument(
        '--roi-block',
        type=int,
        default=32,
        help='ROIs per block when computing weighted traces from full movies.',
    )
    args = parser.parse_args()
    
    data_folder = args.data_folder
    chunks_dir = os.path.join(data_folder, 'chunks')
    
    print(f"Data folder: {data_folder}")
    print(f"Chunks directory: {chunks_dir}")
    
    # Load chunk metadata
    metadata_file = os.path.join(chunks_dir, 'chunk_metadata.pkl')
    with open(metadata_file, 'rb') as f:
        chunk_metadata = pickle.load(f)
    
    n_rois_total = chunk_metadata['n_rois_total']
    original_shape = chunk_metadata['original_shape']
    T, H, W = original_shape
    
    print(f"Total ROIs: {n_rois_total}")
    print(f"Original video shape: {original_shape}")
    
    # Load original ROIs for reference
    ROI_file = os.path.join(data_folder, 'manual_ROIs.npz')
    with np.load(ROI_file) as data:
        original_ROIs = data["ROIs"].astype(bool)
    if original_ROIs.ndim == 2:
        original_ROIs = original_ROIs[np.newaxis, ...]
    
    # Initialize output arrays
    # We'll determine T from the first chunk's VolPy trace length (this is the time axis used by demixing).
    T_actual = None
    
    # Collect all chunk results
    chunk_results_list = []
    for chunk_info in chunk_metadata['chunks']:
        chunk_idx = chunk_info['chunk_idx']
        chunk_dir = chunk_info['chunk_dir']
        
        results_file = os.path.join(chunk_dir, 'chunk_results.pickle')
        if not os.path.exists(results_file):
            print(f"WARNING: Missing results for chunk {chunk_idx}: {results_file}")
            continue
        
        with open(results_file, 'rb') as f:
            chunk_results = pickle.load(f)
        
        chunk_results_list.append(chunk_results)
        
        if T_actual is None:
            T_actual = int(chunk_results["volpy_trace"].shape[-1])
            print(f"Detected T = {T_actual} frames")
    
    if not chunk_results_list:
        raise RuntimeError("No chunk results found!")
    
    # Initialize combined arrays
    volpy_trace = np.zeros((n_rois_total, T_actual))
    volpy_sub_trace = np.zeros((n_rois_total, T_actual))
    spikes = np.empty(n_rois_total, dtype=object)
    snr = np.zeros(n_rois_total)
    weights = np.zeros((n_rois_total, H, W))
    
    raw_traces = np.zeros((n_rois_total, T_actual))
    mc_traces = np.zeros((n_rois_total, T_actual))
    mc_denoised_traces = np.zeros((n_rois_total, T_actual))
    weighted_mc_traces = np.zeros((n_rois_total, T_actual))
    weighted_mc_denoised_traces = np.zeros((n_rois_total, T_actual))
    
    has_raw = False
    has_mc = False
    
    def _assign_trace(dst, orig_idx, src_1d, *, label, chunk_idx):
        if src_1d is None:
            return
        src = np.asarray(src_1d)
        if src.ndim != 1:
            src = src.reshape(-1)
        if src.shape[0] == T_actual:
            dst[orig_idx] = src
            return
        if src.shape[0] > T_actual:
            print(
                f"[WARN] {label} length mismatch in chunk {chunk_idx}: {src.shape[0]} > {T_actual}. Truncating.",
                flush=True,
            )
            dst[orig_idx] = src[:T_actual]
            return
        print(
            f"[WARN] {label} length mismatch in chunk {chunk_idx}: {src.shape[0]} < {T_actual}. Padding zeros.",
            flush=True,
        )
        dst[orig_idx, : src.shape[0]] = src

    # Merge chunk results
    for chunk_results in chunk_results_list:
        chunk_idx = chunk_results['chunk_idx']
        original_roi_indices = chunk_results['original_roi_indices']
        bbox = chunk_results['bbox']
        y_min, y_max, x_min, x_max = bbox
        
        print(f"Merging chunk {chunk_idx} with ROIs {original_roi_indices}")
        
        # Map local chunk indices to original indices
        for local_idx, orig_idx in enumerate(original_roi_indices):
            # Traces (direct copy, same time axis)
            _assign_trace(volpy_trace, orig_idx, chunk_results['volpy_trace'][local_idx], label='volpy_trace', chunk_idx=chunk_idx)
            _assign_trace(volpy_sub_trace, orig_idx, chunk_results['volpy_sub_trace'][local_idx], label='volpy_sub_trace', chunk_idx=chunk_idx)
            snr[orig_idx] = chunk_results['snr'][local_idx]
            
            # Spikes (array of spike times)
            if isinstance(chunk_results['spikes'], np.ndarray):
                if chunk_results['spikes'].dtype == object:
                    spikes[orig_idx] = chunk_results['spikes'][local_idx]
                else:
                    # spikes might be 2D array where each row is spike times
                    spikes[orig_idx] = chunk_results['spikes'][local_idx]
            else:
                spikes[orig_idx] = chunk_results['spikes'][local_idx]
            
            # Weights - need to place chunk weights into full-size array
            chunk_weights = chunk_results['weights']
            if chunk_weights.ndim == 3:
                # (N, chunk_H, chunk_W) -> place at bbox location
                weights[orig_idx, y_min:y_max, x_min:x_max] = chunk_weights[local_idx]
            elif chunk_weights.ndim == 2:
                # (chunk_H*chunk_W, N) or (N, chunk_H*chunk_W)
                chunk_H = y_max - y_min
                chunk_W = x_max - x_min
                if chunk_weights.shape[0] == chunk_H * chunk_W:
                    # (P, N)
                    w = chunk_weights[:, local_idx].reshape(chunk_H, chunk_W)
                else:
                    # (N, P)
                    w = chunk_weights[local_idx].reshape(chunk_H, chunk_W)
                weights[orig_idx, y_min:y_max, x_min:x_max] = w
            
            # Mean and weighted traces
            _assign_trace(mc_denoised_traces, orig_idx, chunk_results['mc_denoised_traces'][local_idx], label='mc_denoised_traces', chunk_idx=chunk_idx)
            _assign_trace(weighted_mc_denoised_traces, orig_idx, chunk_results['weighted_mc_denoised_traces'][local_idx], label='weighted_mc_denoised_traces', chunk_idx=chunk_idx)
            
            if chunk_results['raw_traces'] is not None:
                has_raw = True
                _assign_trace(raw_traces, orig_idx, chunk_results['raw_traces'][local_idx], label='raw_traces', chunk_idx=chunk_idx)
            
            if chunk_results['mc_traces'] is not None:
                has_mc = True
                _assign_trace(mc_traces, orig_idx, chunk_results['mc_traces'][local_idx], label='mc_traces', chunk_idx=chunk_idx)
                _assign_trace(weighted_mc_traces, orig_idx, chunk_results['weighted_mc_traces'][local_idx], label='weighted_mc_traces', chunk_idx=chunk_idx)
    
    # Load mean images from original videos (not from chunks, for consistency)
    print("\nLoading mean images from original videos...")
    
    raw_video_files = glob.glob(os.path.join(data_folder, '*movie.tif'))
    mc_video_file = os.path.join(data_folder, 'movReg.tif')
    # Prefer masked-denoised if present, otherwise fall back to the legacy movReg_denoised.tif.
    mc_denoised_video_file = os.path.join(data_folder, 'movReg_denoised_masked.tif')
    if not os.path.exists(mc_denoised_video_file):
        mc_denoised_video_file = os.path.join(data_folder, 'movReg_denoised.tif')
    
    if os.path.exists(mc_denoised_video_file):
        mean_image_mc_denoised = calculate_mean_image_stream(
            mc_denoised_video_file,
            n_frames=T_actual,
            chunk_frames=int(args.trace_chunk_frames),
        )
        print(f"Loaded mean image from {mc_denoised_video_file}")
    else:
        mean_image_mc_denoised = np.zeros((H, W))
        print(f"WARNING: Missing denoised movie for mean image: {mc_denoised_video_file}")
    
    if raw_video_files:
        mean_image_raw = calculate_mean_image_stream(
            raw_video_files[0],
            n_frames=T_actual,
            chunk_frames=int(args.trace_chunk_frames),
        )
        print(f"Loaded mean image from {raw_video_files[0]}")
    else:
        mean_image_raw = np.zeros((H, W))
    
    if os.path.exists(mc_video_file):
        mean_image_mc = calculate_mean_image_stream(
            mc_video_file,
            n_frames=T_actual,
            chunk_frames=int(args.trace_chunk_frames),
        )
        print(f"Loaded mean image from {mc_video_file}")
    else:
        mean_image_mc = np.zeros((H, W))

    # If chunking only produced denoised chunks, raw/mc traces may be missing from chunk results.
    # Recompute them once from the full movies so downstream QC/viewer has these signals.
    if (not has_raw) and raw_video_files:
        try:
            print("[INFO] Recomputing raw_traces from full raw movie (streaming)...", flush=True)
            raw_traces = compute_mean_traces_stream(
                raw_video_files[0],
                original_ROIs,
                n_frames=T_actual,
                chunk_frames=int(args.trace_chunk_frames),
            )
            has_raw = True
            print("[INFO] raw_traces computed: %s" % (raw_traces.shape,), flush=True)
        except Exception as e:
            print("[WARN] Failed to recompute raw_traces: %r" % (e,), flush=True)

    if (not has_mc) and os.path.exists(mc_video_file):
        try:
            print("[INFO] Recomputing mc_traces and weighted_mc_traces from full movReg.tif (streaming)...", flush=True)
            mc_traces = compute_mean_traces_stream(
                mc_video_file,
                original_ROIs,
                n_frames=T_actual,
                chunk_frames=int(args.trace_chunk_frames),
            )
            weighted_mc_traces = compute_weighted_traces_stream(
                mc_video_file,
                weights,
                n_frames=T_actual,
                chunk_frames=int(args.trace_chunk_frames),
                roi_block=int(args.roi_block),
            )
            has_mc = True
            print("[INFO] mc_traces computed: %s" % (mc_traces.shape,), flush=True)
        except Exception as e:
            print("[WARN] Failed to recompute mc traces: %r" % (e,), flush=True)
    
    # Load motion correction params
    motion_correction_params_path = os.path.join(data_folder, 'motion_correction_params.npy')
    if os.path.exists(motion_correction_params_path):
        motion_correction_params = np.load(motion_correction_params_path, allow_pickle=True).item()
        reg_shifts = motion_correction_params['reg_shifts']
        shift_distances = np.linalg.norm(reg_shifts, axis=1)
    else:
        reg_shifts = None
        shift_distances = None
    
    # Save combined results (same format as demix_fullROI.py)
    output_file = os.path.join(data_folder, 'volpy_demix_results.pickle')
    with open(output_file, 'wb') as file:
        pickle.dump({
            'volpy_trace': volpy_trace,
            'volpy_sub_trace': volpy_sub_trace,
            'spikes': spikes,
            'snr': snr,
            'mean_img_raw': mean_image_raw,
            'mean_img_mc': mean_image_mc,
            'mean_img_mc_denoised': mean_image_mc_denoised,
            'raw_traces': raw_traces,
            'mc_traces': mc_traces,
            'mc_denoised_traces': mc_denoised_traces,
            'weighted_mc_traces': weighted_mc_traces,
            'weighted_mc_denoised_traces': weighted_mc_denoised_traces,
            'weights': weights,
            'ROIs': original_ROIs,
            'reg_shifts': reg_shifts,
            'shift_distances': shift_distances
        }, file)
    
    print(f"\n{'='*60}")
    print("CONCATENATION COMPLETE")
    print(f"{'='*60}")
    print(f"Combined {len(chunk_results_list)} chunks into {n_rois_total} ROIs")
    print(f"Results saved to: {output_file}")
    
    # Clean up chunk files
    import shutil
    try:
        real_chunks = os.path.realpath(chunks_dir)
        real_data = os.path.realpath(data_folder)
        if os.path.isdir(chunks_dir) and os.path.basename(real_chunks) == "chunks" and real_chunks.startswith(real_data + os.sep):
            shutil.rmtree(chunks_dir)
            print(f"Cleaned up chunks directory: {chunks_dir}")
        else:
            print(f"[WARN] Refusing to delete non-standard chunks directory: {chunks_dir}", flush=True)
    except Exception as e:
        print(f"[WARN] Failed to cleanup chunks directory {chunks_dir}: {e!r}", flush=True)


if __name__ == '__main__':
    main()
