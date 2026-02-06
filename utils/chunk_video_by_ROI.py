"""
Chunk video by ROI for memory-efficient VolPy processing.

This script splits a video and ROI masks into smaller chunks,
each containing one or more ROIs with spatial dilation/padding.

Usage:
    python chunk_video_by_ROI.py <input_file> [--dilation 50] [--group_distance 30]

Outputs:
    - chunk_0/chunk_video.tif, chunk_ROI.npz, chunk_info.pkl
    - chunk_1/...
    - ...
    - chunk_metadata.pkl (contains mapping info for reconstruction)
"""

import argparse
import os
import glob
import numpy as np
import tifffile
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.ndimage import binary_dilation, generate_binary_structure
from scipy.spatial.distance import cdist


def get_roi_bbox(roi_mask, dilation=10):
    """Get bounding box for a ROI with dilation padding."""
    ys, xs = np.where(roi_mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()
    
    # Add dilation padding
    y_min = max(0, y_min - dilation)
    y_max = y_max + dilation
    x_min = max(0, x_min - dilation)
    x_max = x_max + dilation
    
    return (y_min, y_max, x_min, x_max)


def get_roi_centroid(roi_mask):
    """Get centroid of a ROI."""
    ys, xs = np.where(roi_mask > 0)
    if len(xs) == 0:
        return None
    return (ys.mean(), xs.mean())


def group_nearby_rois(ROIs, max_distance=30):
    """
    Group ROIs that are close to each other.
    Returns list of lists, where each inner list contains indices of ROIs in a group.
    """
    n_rois = ROIs.shape[0]
    
    # Get centroids
    centroids = []
    for i in range(n_rois):
        c = get_roi_centroid(ROIs[i])
        if c is not None:
            centroids.append(c)
        else:
            centroids.append((0, 0))  # fallback
    
    centroids = np.array(centroids)
    
    if n_rois == 1:
        return [[0]]
    
    # Compute pairwise distances
    distances = cdist(centroids, centroids)
    
    # Simple greedy grouping
    visited = set()
    groups = []
    
    for i in range(n_rois):
        if i in visited:
            continue
        
        group = [i]
        visited.add(i)
        
        # Find all ROIs within max_distance
        for j in range(i + 1, n_rois):
            if j not in visited and distances[i, j] <= max_distance:
                group.append(j)
                visited.add(j)
        
        groups.append(group)
    
    return groups


def merge_bboxes(bboxes, H, W):
    """Merge multiple bounding boxes into one."""
    y_min = min(b[0] for b in bboxes)
    y_max = max(b[1] for b in bboxes)
    x_min = min(b[2] for b in bboxes)
    x_max = max(b[3] for b in bboxes)
    
    # Clip to image bounds
    y_min = max(0, y_min)
    y_max = min(H, y_max)
    x_min = max(0, x_min)
    x_max = min(W, x_max)
    
    return (y_min, y_max, x_min, x_max)


def save_chunk_visualization(mean_image, ROIs, chunk_metadata, output_path):
    """
    Save a visualization showing the mean image, ROIs, and bounding boxes.
    """
    # Generate distinct colors for each chunk
    n_chunks = len(chunk_metadata['chunks'])
    cmap = plt.cm.get_cmap('tab20', max(n_chunks, 1))
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Show mean image
    vmax = np.percentile(mean_image, 99)
    ax.imshow(mean_image, cmap='gray', vmin=0, vmax=vmax)
    
    # Draw each chunk's bounding box and ROIs
    for chunk_info in chunk_metadata['chunks']:
        chunk_idx = chunk_info['chunk_idx']
        roi_indices = chunk_info['roi_indices']
        bbox = chunk_info['bbox']
        y_min, y_max, x_min, x_max = bbox
        
        color = cmap(chunk_idx)
        
        # Draw bounding box
        rect = patches.Rectangle(
            (x_min, y_min), x_max - x_min, y_max - y_min,
            linewidth=2, edgecolor=color, facecolor='none',
            linestyle='--', label=f'Chunk {chunk_idx}'
        )
        ax.add_patch(rect)
        
        # Draw ROI contours and labels
        for roi_idx in roi_indices:
            roi_mask = ROIs[roi_idx]
            # Draw contour
            ax.contour(roi_mask, levels=[0.5], colors=[color], linewidths=1.5)
            
            # Add ROI label at centroid
            ys, xs = np.where(roi_mask > 0)
            if len(xs) > 0:
                cx, cy = xs.mean(), ys.mean()
                ax.text(cx, cy, str(roi_idx), color='white', fontsize=8,
                       ha='center', va='center', fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.7))
        
        # Add chunk label at top-left of bbox
        ax.text(x_min + 2, y_min + 2, f'C{chunk_idx}', color=color, fontsize=10,
               ha='left', va='top', fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.5))
    
    ax.set_title(f"Chunk Visualization: {n_chunks} chunks, {chunk_metadata['n_rois_total']} ROIs\n"
                 f"Dilation: {chunk_metadata['dilation']}px", fontsize=12)
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved chunk visualization to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Chunk video by ROI for memory-efficient processing.')
    parser.add_argument('input_file', type=str, help='Path to the input movie file (e.g., movReg_denoised.tif)')
    parser.add_argument('--dilation', type=int, default=10, help='Dilation/padding around each ROI in pixels')
    parser.add_argument('--group_distance', type=int, default=0, help='Max distance to group nearby ROIs (0 = no grouping)')
    parser.add_argument(
        '--include_mc',
        action='store_true',
        help='Also chunk movReg.tif if present (default: off).',
    )
    parser.add_argument(
        '--include_raw',
        action='store_true',
        help='Also chunk the first *movie.tif found if present (default: off).',
    )
    args = parser.parse_args()
    
    fname = args.input_file
    data_folder = os.path.dirname(fname)
    dilation = args.dilation
    group_distance = args.group_distance
    include_mc = bool(args.include_mc)
    include_raw = bool(args.include_raw)
    
    print(f"Input file: {fname}")
    print(f"Data folder: {data_folder}")
    print(f"Dilation: {dilation} pixels")
    print(f"Group distance: {group_distance} pixels")
    print(f"Include MC video (movReg.tif): {include_mc}")
    print(f"Include raw video (*movie.tif): {include_raw}")
    
    # Load ROIs
    ROI_file = os.path.join(data_folder, 'manual_ROIs.npz')
    print(f"Loading ROIs from: {ROI_file}")
    with np.load(ROI_file) as data:
        ROIs = data["ROIs"].astype(bool)
    
    if ROIs.ndim == 2:
        ROIs = ROIs[np.newaxis, ...]
    
    n_rois = ROIs.shape[0]
    H, W = ROIs.shape[1], ROIs.shape[2]
    print(f"ROIs shape: {ROIs.shape}, {n_rois} ROIs")
    
    # Load video to get dimensions. Use page indexing (not series metadata) so we don't accidentally
    # read only the first series when the TIFF was written via chunked appends.
    print(f"Reading video: {fname}")
    with tifffile.TiffFile(fname) as tif:
        n_pages = len(tif.pages)
        if n_pages < 1:
            raise ValueError(f"No pages found in: {fname}")
        frame0 = tif.pages[0].asarray()
        if frame0.ndim != 2:
            raise ValueError(f"Expected 2D frames in {fname}, got {frame0.shape}")
        vid_H, vid_W = frame0.shape

    video = tifffile.imread(fname, key=slice(0, int(n_pages)))
    T = int(video.shape[0])
    print(f"Video shape: {video.shape} (pages={n_pages})")
    
    assert H == vid_H and W == vid_W, f"ROI dims ({H}, {W}) don't match video ({vid_H}, {vid_W})"
    
    # Group ROIs if requested
    if group_distance > 0:
        roi_groups = group_nearby_rois(ROIs, max_distance=group_distance)
        print(f"Grouped {n_rois} ROIs into {len(roi_groups)} groups")
    else:
        # Each ROI is its own group
        roi_groups = [[i] for i in range(n_rois)]
        print(f"Processing {n_rois} ROIs independently")
    
    # Create chunks directory
    chunks_dir = os.path.join(data_folder, 'chunks')
    os.makedirs(chunks_dir, exist_ok=True)
    
    # Optionally load other videos to chunk (can be very slow on network storage).
    raw_video_files = []
    raw_video = None
    mc_video = None

    if include_raw:
        raw_video_files = glob.glob(os.path.join(data_folder, '*movie.tif'))
        if raw_video_files:
            raw_path = raw_video_files[0]
            with tifffile.TiffFile(raw_path) as tif:
                raw_pages = len(tif.pages)
            raw_video = tifffile.imread(raw_path, key=slice(0, min(int(raw_pages), int(T))))
            if raw_video.shape[0] != T:
                print(
                    f"[WARN] Raw video frames ({raw_video.shape[0]}) != MC-denoised frames ({T}). "
                    f"Truncated raw to {raw_video.shape[0]} frames.",
                    flush=True,
                )
            print(f"Loaded raw video: {raw_path}")
        else:
            print("[WARN] include_raw was set but no *movie.tif was found.", flush=True)

    if include_mc:
        mc_video_file = os.path.join(data_folder, 'movReg.tif')
        if os.path.exists(mc_video_file):
            with tifffile.TiffFile(mc_video_file) as tif:
                mc_pages = len(tif.pages)
            mc_video = tifffile.imread(mc_video_file, key=slice(0, min(int(mc_pages), int(T))))
            if mc_video.shape[0] != T:
                print(
                    f"[WARN] MC video frames ({mc_video.shape[0]}) != MC-denoised frames ({T}). "
                    f"Truncated MC to {mc_video.shape[0]} frames.",
                    flush=True,
                )
            print(f"Loaded MC video: {mc_video_file}")
        else:
            print("[WARN] include_mc was set but movReg.tif was not found.", flush=True)
    
    mc_denoised_video = video  # Already loaded
    
    # Process each group
    chunk_metadata = {
        'n_chunks': len(roi_groups),
        'n_rois_total': n_rois,
        'original_shape': (T, H, W),
        'dilation': dilation,
        'group_distance': group_distance,
        'chunks': []
    }
    
    for chunk_idx, roi_indices in enumerate(roi_groups):
        print(f"\n--- Processing chunk {chunk_idx}/{len(roi_groups)-1} with ROIs {roi_indices} ---")
        
        # Get bounding boxes for all ROIs in this group
        bboxes = []
        for roi_idx in roi_indices:
            bbox = get_roi_bbox(ROIs[roi_idx], dilation=dilation)
            if bbox is not None:
                bboxes.append(bbox)
        
        if not bboxes:
            print(f"  Warning: No valid ROIs in chunk {chunk_idx}, skipping")
            continue
        
        # Merge bboxes
        merged_bbox = merge_bboxes(bboxes, H, W)
        y_min, y_max, x_min, x_max = merged_bbox
        print(f"  Bounding box: y=[{y_min}:{y_max}], x=[{x_min}:{x_max}]")
        print(f"  Chunk size: {y_max - y_min} x {x_max - x_min} pixels")
        
        # Create chunk directory
        chunk_dir = os.path.join(chunks_dir, f'chunk_{chunk_idx}')
        os.makedirs(chunk_dir, exist_ok=True)
        
        # Extract and save cropped ROIs (translated to chunk coordinates)
        chunk_rois = []
        local_roi_indices = []
        for roi_idx in roi_indices:
            cropped_roi = ROIs[roi_idx, y_min:y_max, x_min:x_max]
            chunk_rois.append(cropped_roi)
            local_roi_indices.append(roi_idx)
        
        chunk_rois = np.array(chunk_rois)
        chunk_roi_file = os.path.join(chunk_dir, 'manual_ROIs.npz')
        np.savez(chunk_roi_file, ROIs=chunk_rois)
        print(f"  Saved chunk ROIs: {chunk_roi_file}, shape={chunk_rois.shape}")
        
        # Extract and save cropped videos
        # MC denoised (required)
        chunk_mc_denoised = mc_denoised_video[:, y_min:y_max, x_min:x_max]
        chunk_mc_denoised_file = os.path.join(chunk_dir, 'movReg_denoised.tif')
        tifffile.imsave(chunk_mc_denoised_file, chunk_mc_denoised, bigtiff=True)
        print(f"  Saved chunk MC denoised: {chunk_mc_denoised_file}")
        
        # MC (optional)
        if include_mc and (mc_video is not None):
            chunk_mc = mc_video[:, y_min:y_max, x_min:x_max]
            chunk_mc_file = os.path.join(chunk_dir, 'movReg.tif')
            tifffile.imsave(chunk_mc_file, chunk_mc, bigtiff=True)
            print(f"  Saved chunk MC: {chunk_mc_file}")
        
        # Raw (optional)
        if include_raw and (raw_video is not None):
            chunk_raw = raw_video[:, y_min:y_max, x_min:x_max]
            # Use same naming pattern as original
            raw_name = os.path.basename(raw_video_files[0])
            chunk_raw_file = os.path.join(chunk_dir, raw_name)
            tifffile.imsave(chunk_raw_file, chunk_raw, bigtiff=True)
            print(f"  Saved chunk raw: {chunk_raw_file}")
        
        # Copy motion correction params if exists
        mc_params_file = os.path.join(data_folder, 'motion_correction_params.npy')
        if os.path.exists(mc_params_file):
            import shutil
            shutil.copy(mc_params_file, os.path.join(chunk_dir, 'motion_correction_params.npy'))
        
        # Save chunk info
        chunk_info = {
            'chunk_idx': chunk_idx,
            'roi_indices': roi_indices,  # Original indices in full ROI array
            'bbox': merged_bbox,
            'chunk_shape': chunk_rois.shape[1:],
        }
        chunk_info_file = os.path.join(chunk_dir, 'chunk_info.pkl')
        with open(chunk_info_file, 'wb') as f:
            pickle.dump(chunk_info, f)
        
        chunk_metadata['chunks'].append({
            'chunk_idx': chunk_idx,
            'chunk_dir': chunk_dir,
            'roi_indices': roi_indices,
            'bbox': merged_bbox,
            'n_rois': len(roi_indices),
        })
    
    # Save global metadata
    metadata_file = os.path.join(chunks_dir, 'chunk_metadata.pkl')
    with open(metadata_file, 'wb') as f:
        pickle.dump(chunk_metadata, f)
    print(f"\nSaved chunk metadata to: {metadata_file}")
    
    # Save visualization of chunks
    mean_image = np.mean(mc_denoised_video, axis=0)
    viz_path = os.path.join(data_folder, 'chunk_visualization.png')
    save_chunk_visualization(mean_image, ROIs, chunk_metadata, viz_path)
    
    # Print summary
    print("\n" + "="*60)
    print("CHUNKING COMPLETE")
    print("="*60)
    print(f"Total chunks created: {len(chunk_metadata['chunks'])}")
    print(f"Chunks directory: {chunks_dir}")
    print("\nTo process chunks, submit jobs for each chunk_*/movReg_denoised.tif")


if __name__ == '__main__':
    main()
