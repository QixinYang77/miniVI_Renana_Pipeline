#!/usr/bin/env python3
"""
Generate head-direction per-cell egocentric summary figures for cells tuned at a selected pass threshold in any spike type.

Uses three completed run folders:
  - head_all_spike
  - head_simple_spike
  - head_complex_spike

Selection rule:
  include a cell if pass_<threshold> is True in at least one of the three runs.

Plot behavior:
  - Row 1 uses all-spike fit (when available)
  - Row 2 uses simple-spike fit for reference/arrows/curve
  - Row 3 uses complex-spike fit for reference/arrows/curve

Output:
  <output-dir>/per_cell_summary/<category>/*.svg
  plus CSV manifests.
"""

import argparse
import copy
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

# --------------- path surgery (same as other HPC scripts) ---------------
HERE = Path(__file__).parent.parent.resolve()
_top_repo = str(HERE.parent)
sys.path[:] = [p for p in sys.path if os.path.normpath(p) != os.path.normpath(_top_repo)]
sys.path.insert(0, str(HERE))
os.environ['PYTHONPATH'] = str(HERE)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import utils.placecell_core as core
from utils.placecell_pipeline import (
    AnalysisParams,
    PlaceCellParams,
    PFTraversalParams,
    PooledParams,
    CachePolicy,
    PipelineConfig,
    EgocentricSummaryPlotParams,
    _load_merged_data,
    _prepare_native_analysis_context,
    _load_spatial_analysis_by_idx,
    _passes_egocentric_category_gate,
    _extract_egocentric_plot_timeseries,
    _plot_egocentric_per_cell_summary_figure,
    _filter_egocentric_summary_row_by_valid_bins,
    _filter_egocentric_summary_row_for_tuning_decision,
    _resolve_plot_fit_params,
    _compute_spatial_arrow_fields,
    _compute_empirical_direction_lookup_all_bins,
    _compute_placecell_style_preferred_nonpreferred_maps,
    _make_arena_edges,
    _digitize_with_upper_edge_inclusive,
)
from utils.spatial_heatmaps import classify_spatial_cells


ANIMALS = [
    'CKII_pAce21_PR_20250806',
    'CKII_pAce38_PX_20251126',
    'CKII_pAce45_PX_20260118',
    'CKII_pAce47_PX_20260128',
    'CKII_pAce46_PR_20260222',
    'CKII_pAce50_PRL_20260317',
]


def build_config(data_root, figures_root):
    return PipelineConfig(
        project_root=HERE,
        data_root=Path(data_root),
        figures_root=Path(figures_root),
        notebooks_root=HERE / 'notebooks_PCs',
        animals=ANIMALS,
        analysis=AnalysisParams(
            speed_threshold=3.0, speed_threshold_quiet=0.5, min_duration_s=0.25,
            merge_gap_s=0.0, kernel_size=51, snr_threshold=3.0, min_good_minutes=5.0,
            theta_freqs=(4.0, 8.0), slow_freqs=2.0,
        ),
        place_cell=PlaceCellParams(
            bin_size=1.5, place_field_threshold=0.35, min_component_peak_ratio=0.45,
            split_multi_peak_fields=True, split_secondary_peak_ratio=0.6,
            split_secondary_peak_min_separation_cm=6.0, min_peak_rate=0.5,
            max_field_area_ratio=0.5, min_field_bins=10, min_pf_reliability=0.2,
            min_pf_traversals=5, pf_reliability_dilation_bins=3,
            pf_reliability_dilation_shape='disk', smooth_sigma=1.5, min_occupancy_s=0.001,
            occ_smooth_sigma=1.5, num_shuffles=1000, random_seed=42,
            ss_shape_min_separation_ms=14.0, trim_sparse_top_row_for_analysis=True,
            trim_sparse_top_row_for_plotting=True, sparse_top_row_nonocc_frac_threshold=0.8,
        ),
        traversal=PFTraversalParams(
            center_by_pf_position=True, pf_component_selection='peak_rate',
            min_duration_ms=100.0, min_distance_cm=5.0, traversal_merge_gap_s=2.0,
            clear_traversal=False, session_indices=(0, 1), pf_center_window_sec=10.0,
            min_traversals=10, firing_rate_bin_ms=100.0, firing_rate_smooth_ms=50.0,
            subtract_pre_traversal_baseline=False, mask_non_traversal_pf=True,
            max_pf_distance_cm=8.0, plateau_min_duration_ms=100.0,
        ),
        pooled=PooledParams(
            cb_num_threshold=5, cs_peak_rate_threshold=0.5, run_psd_sections=True,
            cs_plc_only=True, psd_speed_threshold=3, psd_chunk_s=2.0, psd_nperseg_s=1.0,
            psd_noverlap_frac=0.5, simple_event_window_ms=80.0, simple_event_min_gap_ms=50.0,
            min_chunk_valid_fraction=1.0, max_freq=100.0, normalize_psd=True,
            norm_freq_range=(20.0, 100.0),
        ),
        cache=CachePolicy(
            force_recompute=False, validate_only=False, save_executed_notebooks=False,
        ),
    )


def _coerce_bool(v):
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, float, np.integer, np.floating)):
        if not np.isfinite(float(v)):
            return False
        return bool(v)
    s = str(v).strip().lower()
    if s in ('nan', 'na', 'none', ''):
        return False
    return s in ('true', '1', 'yes')


def load_npz_summary_lookup(results_dir, manifest):
    """Build summary lookup dict from per-cell .npz files.

    Returns {(category, animal_id, cell_idx): dict} for successful cells.
    """
    lookup = {}
    n_found = 0
    n_missing = 0
    n_skipped = 0

    for manifest_idx, cell_info in enumerate(manifest):
        npz_path = results_dir / f'cell_{manifest_idx:04d}.npz'
        if not npz_path.exists():
            n_missing += 1
            continue

        try:
            data = dict(np.load(npz_path, allow_pickle=True))
        except Exception as exc:
            print(f'  [WARN] Failed to load {npz_path}: {exc}')
            n_missing += 1
            continue

        status = str(data.get('status', ''))
        if status != 'success':
            n_skipped += 1
            continue

        row = {}
        for key, val in data.items():
            if key == 'status':
                continue
            v = val.item() if hasattr(val, 'item') and getattr(val, 'ndim', 1) == 0 else val
            row[key] = v

        cat = str(row.get('category', cell_info['category']))
        animal = str(row.get('animal_id', cell_info['animal_id']))
        cidx = int(row.get('cell_idx', cell_info['cell_idx']))
        lookup[(cat, animal, cidx)] = row
        n_found += 1

    print(f'NPZ lookup ({results_dir.parent.name}): {n_found} success, {n_skipped} skipped, {n_missing} missing')
    return lookup


def _safe_float(v, default=np.nan):
    try:
        out = float(v)
    except Exception:
        return float(default)
    if not np.isfinite(out):
        return float(default)
    return float(out)


def _sanitize_split_source(value):
    src = str(value).strip().lower()
    return src if src in {'fit', 'empirical'} else 'fit'


def _sanitize_stats_pref_reference(value):
    tok = str(value).strip().lower()
    if tok in {'matching_metric', 'per_metric', 'metric', 'matching'}:
        return 'matching_metric'
    return 'all'


def _paired_means_in_region(pref_map, nonpref_map, region_mask, *, joint_valid_bins=True):
    pref = np.asarray(pref_map, dtype=float)
    nonpref = np.asarray(nonpref_map, dtype=float)
    region = np.asarray(region_mask, dtype=bool)
    if pref.shape != region.shape or nonpref.shape != region.shape:
        return np.nan, np.nan, 0, 0

    if bool(joint_valid_bins):
        keep = region & np.isfinite(pref) & np.isfinite(nonpref)
        n = int(np.sum(keep))
        if n <= 0:
            return np.nan, np.nan, 0, 0
        return float(np.nanmean(pref[keep])), float(np.nanmean(nonpref[keep])), int(n), int(n)

    keep_pref = region & np.isfinite(pref)
    keep_nonpref = region & np.isfinite(nonpref)
    n_pref = int(np.sum(keep_pref))
    n_nonpref = int(np.sum(keep_nonpref))
    pref_mean = float(np.nanmean(pref[keep_pref])) if n_pref > 0 else np.nan
    nonpref_mean = float(np.nanmean(nonpref[keep_nonpref])) if n_nonpref > 0 else np.nan
    return pref_mean, nonpref_mean, int(n_pref), int(n_nonpref)


def _build_source_edges_for_mask(analysis, source_shape):
    nx, ny = int(source_shape[0]), int(source_shape[1])
    if nx <= 0 or ny <= 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    params = analysis.get('params', {}) if isinstance(analysis, dict) else {}
    if not isinstance(params, dict):
        params = {}
    width = _safe_float(params.get('width_real', 35.5), default=35.5)
    height = _safe_float(params.get('height_real', 20.0), default=20.0)
    bin_size = _safe_float(params.get('bin_size', np.nan), default=np.nan)

    if np.isfinite(bin_size) and bin_size > 0:
        x_edges = np.asarray(_make_arena_edges(float(width), float(bin_size)), dtype=float)
        y_edges = np.asarray(_make_arena_edges(float(height), float(bin_size)), dtype=float)
    else:
        x_edges = np.linspace(0.0, float(width), int(nx) + 1, dtype=float)
        y_edges = np.linspace(0.0, float(height), int(ny) + 1, dtype=float)

    if x_edges.size != int(nx) + 1:
        x_edges = np.linspace(0.0, float(width), int(nx) + 1, dtype=float)
    if y_edges.size != int(ny) + 1:
        y_edges = np.linspace(0.0, float(height), int(ny) + 1, dtype=float)
    return np.asarray(x_edges, dtype=float), np.asarray(y_edges, dtype=float)


def _resample_mask_to_target_grid(
    source_mask,
    source_x_edges,
    source_y_edges,
    target_x_edges,
    target_y_edges,
    min_overlap_fraction=0.0,
):
    src = np.asarray(source_mask, dtype=bool)
    tx = np.asarray(target_x_edges, dtype=float)
    ty = np.asarray(target_y_edges, dtype=float)
    sx = np.asarray(source_x_edges, dtype=float)
    sy = np.asarray(source_y_edges, dtype=float)
    if src.ndim != 2 or tx.size < 2 or ty.size < 2 or sx.size < 2 or sy.size < 2:
        return np.zeros((max(tx.size - 1, 0), max(ty.size - 1, 0)), dtype=bool)
    if src.shape != (sx.size - 1, sy.size - 1):
        return np.zeros((tx.size - 1, ty.size - 1), dtype=bool)
    thr = _safe_float(min_overlap_fraction, default=0.0)
    if (not np.isfinite(thr)) or thr < 0:
        thr = 0.0
    thr = float(np.clip(thr, 0.0, 1.0))
    out = np.zeros((int(tx.size - 1), int(ty.size - 1)), dtype=bool)
    sx0 = np.asarray(sx[:-1], dtype=float)
    sx1 = np.asarray(sx[1:], dtype=float)
    sy0 = np.asarray(sy[:-1], dtype=float)
    sy1 = np.asarray(sy[1:], dtype=float)
    for ix in range(int(tx.size - 1)):
        tx0 = float(tx[ix])
        tx1 = float(tx[ix + 1])
        x_overlap = np.where((sx0 < tx1) & (sx1 > tx0))[0]
        if x_overlap.size == 0:
            continue
        for iy in range(int(ty.size - 1)):
            ty0 = float(ty[iy])
            ty1 = float(ty[iy + 1])
            y_overlap = np.where((sy0 < ty1) & (sy1 > ty0))[0]
            if y_overlap.size == 0:
                continue
            target_area = float(max(0.0, tx1 - tx0) * max(0.0, ty1 - ty0))
            if target_area <= 0:
                continue
            overlap_area = 0.0
            for sx_i in x_overlap:
                x0 = max(float(sx0[int(sx_i)]), tx0)
                x1 = min(float(sx1[int(sx_i)]), tx1)
                ox = float(x1 - x0)
                if ox <= 0:
                    continue
                for sy_i in y_overlap:
                    if not bool(src[int(sx_i), int(sy_i)]):
                        continue
                    y0 = max(float(sy0[int(sy_i)]), ty0)
                    y1 = min(float(sy1[int(sy_i)]), ty1)
                    oy = float(y1 - y0)
                    if oy <= 0:
                        continue
                    overlap_area += float(ox * oy)
            if (overlap_area / target_area) >= thr:
                out[ix, iy] = True
    return np.asarray(out, dtype=bool)


def _resample_map_to_target_grid(source_map, source_x_edges, source_y_edges, target_x_edges, target_y_edges):
    src = np.asarray(source_map, dtype=float)
    tx = np.asarray(target_x_edges, dtype=float)
    ty = np.asarray(target_y_edges, dtype=float)
    sx = np.asarray(source_x_edges, dtype=float)
    sy = np.asarray(source_y_edges, dtype=float)
    out = np.full((max(tx.size - 1, 0), max(ty.size - 1, 0)), np.nan, dtype=float)
    if src.ndim != 2 or tx.size < 2 or ty.size < 2 or sx.size < 2 or sy.size < 2:
        return out
    if src.shape != (sx.size - 1, sy.size - 1):
        return out

    x_centers = 0.5 * (tx[:-1] + tx[1:])
    y_centers = 0.5 * (ty[:-1] + ty[1:])
    xx, yy = np.meshgrid(x_centers, y_centers, indexing='ij')
    xi = _digitize_with_upper_edge_inclusive(np.asarray(xx, dtype=float).reshape(-1), sx)
    yi = _digitize_with_upper_edge_inclusive(np.asarray(yy, dtype=float).reshape(-1), sy)
    keep = (
        (xi >= 0) & (yi >= 0)
        & (xi < src.shape[0]) & (yi < src.shape[1])
    )
    if np.any(keep):
        out_flat = out.reshape(-1)
        out_flat[keep] = src[xi[keep].astype(int), yi[keep].astype(int)]
    return np.asarray(out, dtype=float)


def _get_undilated_pf_component_masks(analysis):
    if not isinstance(analysis, dict):
        return None
    components = list(analysis.get('place_field_components', []) or [])
    fallback = np.asarray(analysis.get('place_field_mask', []), dtype=bool)
    if fallback.ndim != 2:
        fallback = np.array([], dtype=bool)

    shape = None
    if fallback.ndim == 2 and fallback.size > 0:
        shape = fallback.shape
    for comp in components:
        mask = np.asarray((comp or {}).get('mask', []), dtype=bool)
        if mask.ndim == 2 and mask.size > 0:
            shape = mask.shape
            break
    if shape is None:
        return None

    primary = np.zeros(shape, dtype=bool)
    secondary = np.zeros(shape, dtype=bool)
    if len(components) >= 1:
        m0 = np.asarray((components[0] or {}).get('mask', []), dtype=bool)
        if m0.shape == shape:
            primary = np.asarray(m0, dtype=bool)
    if len(components) >= 2:
        m1 = np.asarray((components[1] or {}).get('mask', []), dtype=bool)
        if m1.shape == shape:
            secondary = np.asarray(m1, dtype=bool)
    if (not np.any(primary)) and fallback.shape == shape:
        primary = np.asarray(fallback, dtype=bool)
    combined = np.asarray(primary | secondary, dtype=bool)
    return {
        'primary': np.asarray(primary, dtype=bool),
        'secondary': np.asarray(secondary, dtype=bool),
        'combined': np.asarray(combined, dtype=bool),
        'source_shape': shape,
    }


def _mean_in_region(rate_map, region_mask):
    arr = np.asarray(rate_map, dtype=float)
    region = np.asarray(region_mask, dtype=bool)
    if arr.shape != region.shape:
        return np.nan, 0
    keep = region & np.isfinite(arr)
    n = int(np.sum(keep))
    if n <= 0:
        return np.nan, 0
    return float(np.nanmean(arr[keep])), int(n)


def _compute_frame_velocity_components(x_frames, y_frames, speed_frames):
    x_arr = np.asarray(x_frames, dtype=float).reshape(-1)
    y_arr = np.asarray(y_frames, dtype=float).reshape(-1)
    spd = np.asarray(speed_frames, dtype=float).reshape(-1)
    n = int(x_arr.size)
    vx = np.full(n, np.nan, dtype=float)
    vy = np.full(n, np.nan, dtype=float)
    if n <= 0 or y_arr.size != n or spd.size != n:
        return vx, vy
    dx = np.gradient(x_arr)
    dy = np.gradient(y_arr)
    ang = np.arctan2(dy, dx)
    vx = spd * np.cos(ang)
    vy = spd * np.sin(ang)
    bad = (~np.isfinite(vx)) | (~np.isfinite(vy))
    vx = np.asarray(vx, dtype=float)
    vy = np.asarray(vy, dtype=float)
    vx[bad] = np.nan
    vy[bad] = np.nan
    return np.asarray(vx, dtype=float), np.asarray(vy, dtype=float)


def _hd_vel_correlation_in_region(hd_x_map, hd_y_map, vel_x_map, vel_y_map, region_mask, method='normalized_dot'):
    hx = np.asarray(hd_x_map, dtype=float)
    hy = np.asarray(hd_y_map, dtype=float)
    vx = np.asarray(vel_x_map, dtype=float)
    vy = np.asarray(vel_y_map, dtype=float)
    region = np.asarray(region_mask, dtype=bool)
    if not (hx.shape == hy.shape == vx.shape == vy.shape == region.shape):
        return np.nan, 0
    hd_mag = np.sqrt((hx ** 2) + (hy ** 2))
    vel_mag = np.sqrt((vx ** 2) + (vy ** 2))
    keep = (
        region
        & np.isfinite(hx) & np.isfinite(hy)
        & np.isfinite(vx) & np.isfinite(vy)
        & (hd_mag > 0)
        & (vel_mag > 0)
    )
    n = int(np.sum(keep))
    if n <= 0:
        return np.nan, 0
    dot = (hx * vx) + (hy * vy)
    method_norm = str(method).strip().lower()
    if method_norm == 'mean_raw_dot':
        return float(np.nanmean(dot[keep])), int(n)
    denom = np.sum(hd_mag[keep] * vel_mag[keep])
    if (not np.isfinite(denom)) or denom <= 0:
        return np.nan, int(n)
    return float(np.sum(dot[keep]) / float(denom)), int(n)


def _wrap_angle_to_pi_local(angle):
    arr = np.asarray(angle, dtype=float)
    return (arr + np.pi) % (2.0 * np.pi) - np.pi


def _compute_fit_reference_direction_map(fit_info, x_edges, y_edges):
    tx = np.asarray(x_edges, dtype=float)
    ty = np.asarray(y_edges, dtype=float)
    nx = int(max(tx.size - 1, 0))
    ny = int(max(ty.size - 1, 0))
    out = np.full((nx, ny), np.nan, dtype=float)
    if nx <= 0 or ny <= 0:
        return out, False
    if not isinstance(fit_info, dict):
        return out, False
    x_ref = _safe_float(fit_info.get('best_x_ref', np.nan), default=np.nan)
    y_ref = _safe_float(fit_info.get('best_y_ref', np.nan), default=np.nan)
    theta = _safe_float(fit_info.get('best_theta', np.nan), default=np.nan)
    if (not np.isfinite(x_ref)) or (not np.isfinite(y_ref)) or (not np.isfinite(theta)):
        return out, False
    x_centers = 0.5 * (tx[:-1] + tx[1:])
    y_centers = 0.5 * (ty[:-1] + ty[1:])
    xx, yy = np.meshgrid(x_centers, y_centers, indexing='ij')
    alpha = np.arctan2(float(y_ref) - yy, float(x_ref) - xx)
    return _wrap_angle_to_pi_local(alpha - float(theta)), True


def _compute_split_frame_masks_for_target_grid(
    *,
    data,
    fit_info,
    params,
    preferred_lookup_specs,
    split_source,
    preferred_half_width_deg,
    target_x_edges,
    target_y_edges,
):
    tx = np.asarray(target_x_edges, dtype=float)
    ty = np.asarray(target_y_edges, dtype=float)
    empty_masks = {
        'all': {'preferred': np.array([], dtype=bool), 'nonpreferred': np.array([], dtype=bool)},
        'ss': {'preferred': np.array([], dtype=bool), 'nonpreferred': np.array([], dtype=bool)},
        'cs': {'preferred': np.array([], dtype=bool), 'nonpreferred': np.array([], dtype=bool)},
    }
    out = {
        'ok': False,
        'reason': 'unknown',
        'valid_frames': np.array([], dtype=bool),
        'moving_mask_frames': np.array([], dtype=bool),
        'speed_smoothed_frames': np.array([], dtype=float),
        'masks': empty_masks,
    }
    if not isinstance(data, dict):
        out['reason'] = 'invalid_data'
        return out
    x_arr = np.asarray(data.get('x_frames', np.array([])), dtype=float).reshape(-1)
    y_arr = np.asarray(data.get('y_frames', np.array([])), dtype=float).reshape(-1)
    speed_arr = np.asarray(data.get('speed_frames', np.array([])), dtype=float).reshape(-1)
    dir_arr = _wrap_angle_to_pi_local(np.asarray(data.get('dir_frames', np.array([])), dtype=float).reshape(-1))
    bad_arr = np.asarray(data.get('bad_mask_frames', np.array([])), dtype=bool).reshape(-1)
    n_frames = int(x_arr.size)
    if n_frames <= 0:
        out['reason'] = 'no_frames'
        return out
    if any(arr.size != n_frames for arr in (y_arr, speed_arr, dir_arr)):
        out['reason'] = 'length_mismatch'
        return out
    if bad_arr.size != n_frames:
        bad_arr = np.zeros(n_frames, dtype=bool)

    frame_rate = _safe_float(data.get('frame_rate', np.nan), default=np.nan)
    if (not np.isfinite(frame_rate)) or frame_rate <= 0:
        out['reason'] = 'invalid_frame_rate'
        return out

    pcfg = data.get('placecell_map_params', {})
    if not isinstance(pcfg, dict):
        pcfg = {}
    kernel_size = int(max(1, int(pcfg.get('kernel_size', int(getattr(params, 'pc_kernel_size', 51))))))
    filter_type = str(pcfg.get('filter_type', str(getattr(params, 'pc_filter_type', 'median'))))
    speed_threshold = _safe_float(
        pcfg.get('speed_threshold', getattr(params, 'pc_speed_threshold_cm_s', np.nan)),
        default=getattr(params, 'pc_speed_threshold_cm_s', 3.0),
    )
    min_duration_s = _safe_float(
        pcfg.get('min_duration_s', getattr(params, 'pc_min_duration_s', np.nan)),
        default=getattr(params, 'pc_min_duration_s', 0.25),
    )
    merge_gap_s = _safe_float(
        pcfg.get('merge_gap_s', getattr(params, 'pc_merge_gap_s', np.nan)),
        default=getattr(params, 'pc_merge_gap_s', 0.0),
    )

    valid_frames = (
        np.isfinite(x_arr)
        & np.isfinite(y_arr)
        & np.isfinite(speed_arr)
        & (~bad_arr)
    )
    first_n_minutes = getattr(params, 'first_n_minutes', None)
    if first_n_minutes is not None:
        cutoff = int(np.floor(float(first_n_minutes) * 60.0 * float(frame_rate)))
        cutoff = max(1, min(n_frames, cutoff))
        if cutoff < n_frames:
            valid_frames[cutoff:] = False

    speed_for_epochs = np.asarray(speed_arr, dtype=float).copy()
    speed_for_epochs[~valid_frames] = np.nan
    try:
        speed_smooth, _, moving_idx = core._compute_moving_epochs(
            speed_for_epochs,
            float(frame_rate),
            kernel_size=int(kernel_size),
            filter_type=str(filter_type),
            speed_threshold=float(speed_threshold),
            min_duration_s=float(min_duration_s),
            merge_gap_s=float(merge_gap_s),
        )
    except Exception:
        out['reason'] = 'moving_epoch_failed'
        return out
    moving_idx = np.asarray(moving_idx, dtype=int).reshape(-1)
    if moving_idx.size > 0:
        moving_idx = moving_idx[(moving_idx >= 0) & (moving_idx < n_frames)]
        moving_idx = moving_idx[valid_frames[moving_idx]]
    if moving_idx.size <= 0:
        out['reason'] = 'no_moving_frames'
        return out

    fit_dir_map, fit_has_ref = _compute_fit_reference_direction_map(fit_info, tx, ty)
    split_source = _sanitize_split_source(split_source)
    if split_source == 'fit' and (not fit_has_ref):
        out['reason'] = 'missing_fit_reference'
        return out

    x_mv = np.asarray(x_arr[moving_idx], dtype=float)
    y_mv = np.asarray(y_arr[moving_idx], dtype=float)
    dir_mv = np.asarray(dir_arr[moving_idx], dtype=float)
    finite_mv = np.isfinite(dir_mv) & np.isfinite(x_mv) & np.isfinite(y_mv)
    pref_half_width_rad = float(np.deg2rad(float(preferred_half_width_deg)))

    def _resolve_lookup_spec(key):
        fallback = (np.asarray(tx, dtype=float), np.asarray(ty, dtype=float), np.asarray(fit_dir_map, dtype=float))
        if split_source != 'empirical' or not isinstance(preferred_lookup_specs, dict):
            return fallback
        spec = preferred_lookup_specs.get(str(key), {})
        if not isinstance(spec, dict):
            return fallback
        x_lookup = np.asarray(spec.get('x_edges', np.array([])), dtype=float)
        y_lookup = np.asarray(spec.get('y_edges', np.array([])), dtype=float)
        dir_lookup = np.asarray(spec.get('dir_map', np.array([])), dtype=float)
        if x_lookup.size < 2 or y_lookup.size < 2:
            return fallback
        nx_lookup = int(x_lookup.size - 1)
        ny_lookup = int(y_lookup.size - 1)
        if dir_lookup.shape != (nx_lookup, ny_lookup):
            return fallback
        return np.asarray(x_lookup, dtype=float), np.asarray(y_lookup, dtype=float), np.asarray(dir_lookup, dtype=float)

    def _build_split_masks_from_lookup(x_lookup, y_lookup, dir_lookup):
        pref_mask_local = np.zeros(n_frames, dtype=bool)
        nonpref_mask_local = np.zeros(n_frames, dtype=bool)
        xi_mv = _digitize_with_upper_edge_inclusive(x_mv, x_lookup)
        yi_mv = _digitize_with_upper_edge_inclusive(y_mv, y_lookup)
        in_bounds = (
            (xi_mv >= 0)
            & (yi_mv >= 0)
            & (xi_mv < int(x_lookup.size - 1))
            & (yi_mv < int(y_lookup.size - 1))
        )
        keep_mv = in_bounds & finite_mv
        if not np.any(keep_mv):
            return pref_mask_local, nonpref_mask_local
        idx_use = moving_idx[keep_mv]
        ii = xi_mv[keep_mv].astype(int)
        jj = yi_mv[keep_mv].astype(int)
        dirs = dir_mv[keep_mv]
        ref = np.asarray(dir_lookup, dtype=float)[ii, jj]
        has_ref = np.isfinite(ref)
        if not np.any(has_ref):
            return pref_mask_local, nonpref_mask_local
        idx_use = idx_use[has_ref]
        dirs = dirs[has_ref]
        ref = ref[has_ref]
        delta = _wrap_angle_to_pi_local(dirs - ref)
        is_pref = np.abs(delta) <= (pref_half_width_rad + 1e-12)
        if np.any(is_pref):
            pref_mask_local[np.asarray(idx_use[is_pref], dtype=int)] = True
        if np.any(~is_pref):
            nonpref_mask_local[np.asarray(idx_use[~is_pref], dtype=int)] = True
        return pref_mask_local, nonpref_mask_local

    masks = {}
    for key in ('all', 'ss', 'cs'):
        x_lookup, y_lookup, dir_lookup = _resolve_lookup_spec(key)
        pref_mask, nonpref_mask = _build_split_masks_from_lookup(x_lookup, y_lookup, dir_lookup)
        masks[key] = {
            'preferred': np.asarray(pref_mask, dtype=bool),
            'nonpreferred': np.asarray(nonpref_mask, dtype=bool),
        }
    out['ok'] = True
    out['reason'] = 'ok'
    out['valid_frames'] = np.asarray(valid_frames, dtype=bool)
    moving_mask = np.zeros(n_frames, dtype=bool)
    if moving_idx.size > 0:
        moving_mask[np.asarray(moving_idx, dtype=int)] = True
    moving_mask &= np.asarray(valid_frames, dtype=bool)
    out['moving_mask_frames'] = np.asarray(moving_mask, dtype=bool)
    speed_smooth_arr = np.asarray(speed_smooth, dtype=float).reshape(-1)
    if speed_smooth_arr.size != n_frames:
        speed_smooth_arr = np.asarray(speed_arr, dtype=float).reshape(-1)
    out['speed_smoothed_frames'] = np.asarray(speed_smooth_arr, dtype=float)
    out['masks'] = masks
    return out


def _compute_split_spike_mrl_map(
    *,
    x_frames,
    y_frames,
    dir_frames,
    spike_frames,
    split_mask,
    valid_frames,
    x_edges,
    y_edges,
):
    x_arr = np.asarray(x_frames, dtype=float).reshape(-1)
    y_arr = np.asarray(y_frames, dtype=float).reshape(-1)
    dir_arr = _wrap_angle_to_pi_local(np.asarray(dir_frames, dtype=float).reshape(-1))
    split = np.asarray(split_mask, dtype=bool).reshape(-1)
    valid = np.asarray(valid_frames, dtype=bool).reshape(-1)
    tx = np.asarray(x_edges, dtype=float)
    ty = np.asarray(y_edges, dtype=float)
    nx = int(max(tx.size - 1, 0))
    ny = int(max(ty.size - 1, 0))
    out = np.full((nx, ny), np.nan, dtype=float)
    n_frames = int(x_arr.size)
    if nx <= 0 or ny <= 0 or n_frames <= 0:
        return out
    if any(arr.size != n_frames for arr in (y_arr, dir_arr, split, valid)):
        return out
    spk = np.asarray(spike_frames, dtype=int).reshape(-1)
    if spk.size <= 0:
        return out
    spk = spk[(spk >= 0) & (spk < n_frames)]
    if spk.size <= 0:
        return out
    spk = np.unique(spk)
    keep = valid[spk] & split[spk]
    if not np.any(keep):
        return out
    spk = spk[keep]
    xs = np.asarray(x_arr[spk], dtype=float)
    ys = np.asarray(y_arr[spk], dtype=float)
    ds = np.asarray(dir_arr[spk], dtype=float)
    xi = _digitize_with_upper_edge_inclusive(xs, tx)
    yi = _digitize_with_upper_edge_inclusive(ys, ty)
    in_bounds = (
        np.isfinite(xs)
        & np.isfinite(ys)
        & np.isfinite(ds)
        & (xi >= 0) & (yi >= 0)
        & (xi < nx) & (yi < ny)
    )
    if not np.any(in_bounds):
        return out
    ii = xi[in_bounds].astype(int)
    jj = yi[in_bounds].astype(int)
    ds = ds[in_bounds]
    flat = ii * ny + jj
    cos_sum = np.bincount(flat, weights=np.cos(ds), minlength=nx * ny).astype(float)
    sin_sum = np.bincount(flat, weights=np.sin(ds), minlength=nx * ny).astype(float)
    cnt = np.bincount(flat, minlength=nx * ny).astype(float)
    keep_bins = cnt > 0
    if not np.any(keep_bins):
        return out
    vec_mag = np.sqrt((cos_sum[keep_bins] ** 2) + (sin_sum[keep_bins] ** 2))
    out_flat = out.reshape(-1)
    out_flat[keep_bins] = vec_mag / cnt[keep_bins]
    return np.asarray(out, dtype=float)


def _compute_split_occupancy_time_map(
    *,
    x_frames,
    y_frames,
    split_mask,
    valid_frames,
    x_edges,
    y_edges,
    frame_rate,
):
    x_arr = np.asarray(x_frames, dtype=float).reshape(-1)
    y_arr = np.asarray(y_frames, dtype=float).reshape(-1)
    split = np.asarray(split_mask, dtype=bool).reshape(-1)
    valid = np.asarray(valid_frames, dtype=bool).reshape(-1)
    tx = np.asarray(x_edges, dtype=float)
    ty = np.asarray(y_edges, dtype=float)
    nx = int(max(tx.size - 1, 0))
    ny = int(max(ty.size - 1, 0))
    out = np.zeros((nx, ny), dtype=float)
    if nx <= 0 or ny <= 0:
        return out
    n_frames = int(x_arr.size)
    if n_frames <= 0 or y_arr.size != n_frames or split.size != n_frames or valid.size != n_frames:
        return out
    if (not np.isfinite(float(frame_rate))) or float(frame_rate) <= 0:
        return out
    keep = split & valid & np.isfinite(x_arr) & np.isfinite(y_arr)
    if not np.any(keep):
        return out
    counts, _, _ = np.histogram2d(x_arr[keep], y_arr[keep], bins=[tx, ty])
    out = np.asarray(counts, dtype=float) / float(frame_rate)
    return np.asarray(out, dtype=float)


def _compute_split_framewise_mean_map(
    *,
    x_frames,
    y_frames,
    value_frames,
    split_mask,
    valid_frames,
    x_edges,
    y_edges,
):
    x_arr = np.asarray(x_frames, dtype=float).reshape(-1)
    y_arr = np.asarray(y_frames, dtype=float).reshape(-1)
    val_arr = np.asarray(value_frames, dtype=float).reshape(-1)
    split = np.asarray(split_mask, dtype=bool).reshape(-1)
    valid = np.asarray(valid_frames, dtype=bool).reshape(-1)
    tx = np.asarray(x_edges, dtype=float)
    ty = np.asarray(y_edges, dtype=float)
    nx = int(max(tx.size - 1, 0))
    ny = int(max(ty.size - 1, 0))
    out = np.full((nx, ny), np.nan, dtype=float)
    if nx <= 0 or ny <= 0:
        return out
    n_frames = int(x_arr.size)
    if (
        n_frames <= 0
        or y_arr.size != n_frames
        or val_arr.size != n_frames
        or split.size != n_frames
        or valid.size != n_frames
    ):
        return out
    keep = split & valid & np.isfinite(x_arr) & np.isfinite(y_arr) & np.isfinite(val_arr)
    if not np.any(keep):
        return out
    counts, _, _ = np.histogram2d(x_arr[keep], y_arr[keep], bins=[tx, ty])
    sums, _, _ = np.histogram2d(x_arr[keep], y_arr[keep], bins=[tx, ty], weights=val_arr[keep])
    counts = np.asarray(counts, dtype=float)
    sums = np.asarray(sums, dtype=float)
    good = counts > 0
    if np.any(good):
        out = np.asarray(out, dtype=float)
        out[good] = sums[good] / counts[good]
    return np.asarray(out, dtype=float)


def _compute_split_heading_vector_map(
    *,
    x_frames,
    y_frames,
    dir_frames,
    split_mask,
    valid_frames,
    x_edges,
    y_edges,
):
    x_arr = np.asarray(x_frames, dtype=float).reshape(-1)
    y_arr = np.asarray(y_frames, dtype=float).reshape(-1)
    dir_arr = _wrap_angle_to_pi_local(np.asarray(dir_frames, dtype=float).reshape(-1))
    split = np.asarray(split_mask, dtype=bool).reshape(-1)
    valid = np.asarray(valid_frames, dtype=bool).reshape(-1)
    tx = np.asarray(x_edges, dtype=float)
    ty = np.asarray(y_edges, dtype=float)
    nx = int(max(tx.size - 1, 0))
    ny = int(max(ty.size - 1, 0))
    vx_out = np.full((nx, ny), np.nan, dtype=float)
    vy_out = np.full((nx, ny), np.nan, dtype=float)
    mrl_out = np.full((nx, ny), np.nan, dtype=float)
    if nx <= 0 or ny <= 0:
        return vx_out, vy_out, mrl_out
    n_frames = int(x_arr.size)
    if (
        n_frames <= 0
        or y_arr.size != n_frames
        or dir_arr.size != n_frames
        or split.size != n_frames
        or valid.size != n_frames
    ):
        return vx_out, vy_out, mrl_out
    keep = split & valid & np.isfinite(x_arr) & np.isfinite(y_arr) & np.isfinite(dir_arr)
    if not np.any(keep):
        return vx_out, vy_out, mrl_out
    c = np.cos(dir_arr[keep])
    s = np.sin(dir_arr[keep])
    counts, _, _ = np.histogram2d(x_arr[keep], y_arr[keep], bins=[tx, ty])
    sum_c, _, _ = np.histogram2d(x_arr[keep], y_arr[keep], bins=[tx, ty], weights=c)
    sum_s, _, _ = np.histogram2d(x_arr[keep], y_arr[keep], bins=[tx, ty], weights=s)
    counts = np.asarray(counts, dtype=float)
    sum_c = np.asarray(sum_c, dtype=float)
    sum_s = np.asarray(sum_s, dtype=float)
    good = counts > 0
    if np.any(good):
        vx_out = np.asarray(vx_out, dtype=float)
        vy_out = np.asarray(vy_out, dtype=float)
        mrl_out = np.asarray(mrl_out, dtype=float)
        vx_out[good] = sum_c[good] / counts[good]
        vy_out[good] = sum_s[good] / counts[good]
        mrl_out[good] = np.sqrt((vx_out[good] ** 2) + (vy_out[good] ** 2))
    return np.asarray(vx_out, dtype=float), np.asarray(vy_out, dtype=float), np.asarray(mrl_out, dtype=float)


def _compute_value_heading_mrl_map(
    *,
    x_frames,
    y_frames,
    dir_frames,
    value_frames,
    valid_frames,
    moving_frames,
    x_edges,
    y_edges,
    frame_rate,
    n_angle_bins,
    occupancy_threshold_s,
    min_occupied_angle_bins,
    clip_negative_values=False,
):
    x_arr = np.asarray(x_frames, dtype=float).reshape(-1)
    y_arr = np.asarray(y_frames, dtype=float).reshape(-1)
    dir_arr = _wrap_angle_to_pi_local(np.asarray(dir_frames, dtype=float).reshape(-1))
    val_arr = np.asarray(value_frames, dtype=float).reshape(-1)
    valid = np.asarray(valid_frames, dtype=bool).reshape(-1)
    moving = np.asarray(moving_frames, dtype=bool).reshape(-1)
    tx = np.asarray(x_edges, dtype=float)
    ty = np.asarray(y_edges, dtype=float)
    nx = int(max(tx.size - 1, 0))
    ny = int(max(ty.size - 1, 0))
    out = np.full((nx, ny), np.nan, dtype=float)
    if nx <= 0 or ny <= 0:
        return out
    n_frames = int(x_arr.size)
    if (
        n_frames <= 0
        or y_arr.size != n_frames
        or dir_arr.size != n_frames
        or val_arr.size != n_frames
        or valid.size != n_frames
        or moving.size != n_frames
    ):
        return out
    fr = _safe_float(frame_rate, default=np.nan)
    if (not np.isfinite(fr)) or fr <= 0:
        return out
    n_ang = int(max(2, int(n_angle_bins)))
    occ_thr_s = _safe_float(occupancy_threshold_s, default=0.0)
    if (not np.isfinite(occ_thr_s)) or occ_thr_s < 0:
        occ_thr_s = 0.0
    min_occ_bins = int(max(1, int(min_occupied_angle_bins)))

    keep = (
        valid
        & moving
        & np.isfinite(x_arr)
        & np.isfinite(y_arr)
        & np.isfinite(dir_arr)
        & np.isfinite(val_arr)
    )
    if not np.any(keep):
        return out
    xs = np.asarray(x_arr[keep], dtype=float)
    ys = np.asarray(y_arr[keep], dtype=float)
    ds = np.asarray(dir_arr[keep], dtype=float)
    vs = np.asarray(val_arr[keep], dtype=float)
    xi = _digitize_with_upper_edge_inclusive(xs, tx)
    yi = _digitize_with_upper_edge_inclusive(ys, ty)
    angle_edges = np.linspace(-np.pi, np.pi, n_ang + 1, dtype=float)
    angle_centers = 0.5 * (angle_edges[:-1] + angle_edges[1:])
    ai = np.searchsorted(angle_edges, ds, side='right') - 1
    ai = np.mod(ai, n_ang)
    in_bounds = (
        (xi >= 0)
        & (yi >= 0)
        & (xi < nx)
        & (yi < ny)
        & (ai >= 0)
        & (ai < n_ang)
    )
    if not np.any(in_bounds):
        return out
    ii = xi[in_bounds].astype(int)
    jj = yi[in_bounds].astype(int)
    kk = ai[in_bounds].astype(int)
    vals = np.asarray(vs[in_bounds], dtype=float)

    flat_idx = (ii * ny + jj) * n_ang + kk
    n_total = int(nx * ny * n_ang)
    cnt = np.bincount(flat_idx, minlength=n_total).astype(float).reshape((nx, ny, n_ang))
    val_sum = np.bincount(flat_idx, weights=vals, minlength=n_total).astype(float).reshape((nx, ny, n_ang))
    with np.errstate(divide='ignore', invalid='ignore'):
        mean_val = val_sum / cnt
    mean_val[cnt <= 0] = np.nan
    occ_s = cnt / float(fr)

    for ix in range(nx):
        for iy in range(ny):
            occ_curve = np.asarray(occ_s[ix, iy, :], dtype=float)
            val_curve = np.asarray(mean_val[ix, iy, :], dtype=float)
            good = np.isfinite(val_curve) & np.isfinite(occ_curve) & (occ_curve >= float(occ_thr_s))
            if int(np.sum(good)) < int(min_occ_bins):
                continue
            w = np.asarray(val_curve[good], dtype=float)
            if bool(clip_negative_values):
                w = np.maximum(w, 0.0)
            ang = np.asarray(angle_centers[good], dtype=float)
            finite = np.isfinite(w) & np.isfinite(ang)
            if not np.any(finite):
                continue
            w = w[finite]
            ang = ang[finite]
            if bool(clip_negative_values):
                w = np.maximum(w, 0.0)
            if w.size <= 0:
                continue
            mean_w = float(np.nanmean(w))
            if (not np.isfinite(mean_w)) or mean_w <= 0:
                continue
            w_norm = w / mean_w
            keep_w = np.isfinite(w_norm) & (w_norm > 0) & np.isfinite(ang)
            if not np.any(keep_w):
                continue
            w_use = np.asarray(w_norm[keep_w], dtype=float)
            ang_use = np.asarray(ang[keep_w], dtype=float)
            denom = float(np.sum(w_use))
            if (not np.isfinite(denom)) or denom <= 0:
                continue
            vec = np.sum(w_use * np.exp(1j * ang_use)) / denom
            out[ix, iy] = float(np.clip(np.abs(vec), 0.0, 1.0))
    return np.asarray(out, dtype=float)


def _compute_pf_split_stats_rows(
    *,
    analysis,
    plot_data,
    row_all,
    row_ss,
    row_cs,
    params,
    split_source,
    stats_pref_reference,
    stats_joint_valid_bins,
    category,
    animal_id,
    cell_idx,
    any_pass_threshold,
    pass_100_any=False,
    hd_vel_corr_method='normalized_dot',
):
    hd_vel_corr_method = str(hd_vel_corr_method).strip().lower()
    if hd_vel_corr_method not in {'normalized_dot', 'mean_raw_dot'}:
        hd_vel_corr_method = 'normalized_dot'
    stats_joint_valid_bins = bool(stats_joint_valid_bins)
    pass_100_any = bool(pass_100_any)
    is_place_cell = bool((analysis or {}).get('is_place_cell', False))
    pf_masks = _get_undilated_pf_component_masks(analysis) if is_place_cell else None

    local_all = plot_data.get('local_tuning_all')
    local_ss = plot_data.get('local_tuning_ss')
    local_cs = plot_data.get('local_tuning_cs')
    fit_all = _resolve_plot_fit_params(local_tuning=local_all, summary_row=row_all if isinstance(row_all, dict) else {})
    fit_ss = _resolve_plot_fit_params(local_tuning=local_ss, summary_row=row_ss if isinstance(row_ss, dict) else row_all)
    fit_cs = _resolve_plot_fit_params(local_tuning=local_cs, summary_row=row_cs if isinstance(row_cs, dict) else row_all)

    arrow_all = _compute_spatial_arrow_fields(local_tuning=local_all, fit_info=fit_all, params=params)
    arrow_ss = _compute_spatial_arrow_fields(local_tuning=local_ss, fit_info=fit_ss, params=params)
    arrow_cs = _compute_spatial_arrow_fields(local_tuning=local_cs, fit_info=fit_cs, params=params)
    emp_lookup_all = _compute_empirical_direction_lookup_all_bins(local_tuning=local_all if isinstance(local_all, dict) else {})
    emp_lookup_ss = _compute_empirical_direction_lookup_all_bins(local_tuning=local_ss if isinstance(local_ss, dict) else {})
    emp_lookup_cs = _compute_empirical_direction_lookup_all_bins(local_tuning=local_cs if isinstance(local_cs, dict) else {})

    def _lookup_spec_from_empirical_or_fallback(emp_lookup: dict, fallback_lookup: dict) -> dict:
        fb = fallback_lookup if isinstance(fallback_lookup, dict) else {}
        valid_mask_col8 = np.asarray(fb.get('arrow_valid_mask', np.array([])), dtype=bool)

        def _apply_col8_valid_mask(dir_map_in: np.ndarray) -> np.ndarray:
            dir_map_out = np.asarray(dir_map_in, dtype=float).copy()
            if (
                str(split_source) == 'empirical'
                and valid_mask_col8.ndim == 2
                and dir_map_out.ndim == 2
                and valid_mask_col8.shape == dir_map_out.shape
            ):
                dir_map_out[~valid_mask_col8] = np.nan
            return np.asarray(dir_map_out, dtype=float)

        x_emp = np.asarray(emp_lookup.get('x_edges', np.array([])), dtype=float) if isinstance(emp_lookup, dict) else np.array([], dtype=float)
        y_emp = np.asarray(emp_lookup.get('y_edges', np.array([])), dtype=float) if isinstance(emp_lookup, dict) else np.array([], dtype=float)
        d_emp = np.asarray(emp_lookup.get('dir_map', np.array([])), dtype=float) if isinstance(emp_lookup, dict) else np.array([], dtype=float)
        if (
            x_emp.size >= 2
            and y_emp.size >= 2
            and d_emp.shape == (int(x_emp.size - 1), int(y_emp.size - 1))
        ):
            return {
                'x_edges': np.asarray(x_emp, dtype=float),
                'y_edges': np.asarray(y_emp, dtype=float),
                'dir_map': _apply_col8_valid_mask(np.asarray(d_emp, dtype=float)),
            }
        return {
            'x_edges': np.asarray(fb.get('x_edges', np.array([])), dtype=float),
            'y_edges': np.asarray(fb.get('y_edges', np.array([])), dtype=float),
            'dir_map': _apply_col8_valid_mask(np.asarray(fb.get('psi_emp_map', np.array([])), dtype=float)),
        }

    stats_pref_reference = _sanitize_stats_pref_reference(stats_pref_reference)
    if stats_pref_reference == 'matching_metric':
        ss_emp_lookup = emp_lookup_ss
        cs_emp_lookup = emp_lookup_cs
        ss_lookup = arrow_ss
        cs_lookup = arrow_cs
    else:
        # Default: use all-spike preferred/non-preferred definition for every metric.
        ss_emp_lookup = emp_lookup_all
        cs_emp_lookup = emp_lookup_all
        ss_lookup = arrow_all
        cs_lookup = arrow_all
    preferred_lookup_specs = {
        'all': _lookup_spec_from_empirical_or_fallback(emp_lookup_all, arrow_all),
        'ss': _lookup_spec_from_empirical_or_fallback(ss_emp_lookup, ss_lookup),
        'cs': _lookup_spec_from_empirical_or_fallback(cs_emp_lookup, cs_lookup),
    }

    split_source = _sanitize_split_source(split_source)
    stats_data = dict(plot_data)
    pcfg = dict(plot_data.get('placecell_map_params', {})) if isinstance(plot_data.get('placecell_map_params', {}), dict) else {}
    pcfg['smooth_sigma'] = 0.0
    pcfg['occ_smooth_sigma'] = 0.0
    stats_data['placecell_map_params'] = pcfg
    stats_params = copy.deepcopy(params)
    stats_params.pc_smooth_sigma = 0.0
    stats_params.pc_occ_smooth_sigma = 0.0

    split_maps = _compute_placecell_style_preferred_nonpreferred_maps(
        data=stats_data,
        fit_info=fit_all,
        params=stats_params,
        preferred_angle_source=split_source,
        preferred_lookup_specs=preferred_lookup_specs,
        preferred_half_width_deg=float(getattr(params, 'split_preferred_half_width_deg', 50.0)),
    )
    if not bool(split_maps.get('ok', False)):
        return []

    tx = np.asarray(split_maps.get('x_edges', np.array([])), dtype=float)
    ty = np.asarray(split_maps.get('y_edges', np.array([])), dtype=float)
    if tx.size < 2 or ty.size < 2:
        return []
    target_shape = (int(tx.size - 1), int(ty.size - 1))
    region_masks = {'all_bins': np.ones(target_shape, dtype=bool)}
    pf_mask_mode_label = 'non_place_all_bins'
    if is_place_cell and isinstance(pf_masks, dict):
        source_shape = tuple(int(v) for v in pf_masks['source_shape'])
        sx, sy = _build_source_edges_for_mask(analysis, source_shape)
        if sx.size >= 2 and sy.size >= 2:
            pf_overlap_thr = _safe_float(getattr(params, 'pf_overlay_area_threshold', 0.25), default=0.25)
            if (not np.isfinite(pf_overlap_thr)) or pf_overlap_thr < 0:
                pf_overlap_thr = 0.25
            pf_overlap_thr = float(np.clip(pf_overlap_thr, 0.0, 1.0))
            region_masks = {
                'primary': _resample_mask_to_target_grid(
                    pf_masks['primary'], sx, sy, tx, ty, min_overlap_fraction=pf_overlap_thr
                ),
                'secondary': _resample_mask_to_target_grid(
                    pf_masks['secondary'], sx, sy, tx, ty, min_overlap_fraction=pf_overlap_thr
                ),
                'combined': _resample_mask_to_target_grid(
                    pf_masks['combined'], sx, sy, tx, ty, min_overlap_fraction=pf_overlap_thr
                ),
            }
            region_masks['outside_combined'] = np.asarray(~np.asarray(region_masks['combined'], dtype=bool), dtype=bool)
            region_masks['all_bins'] = np.ones_like(np.asarray(region_masks['combined'], dtype=bool), dtype=bool)
            pf_mask_mode_label = 'components_undilated'
    metric_map_keys = {
        'all': 'all',
        'ss': 'ss',
        'cs': 'cs',
        'theta': 'theta',
        'slow': 'slow',
    }
    split_bin_size_cm = _safe_float(getattr(params, 'split_map_bin_size_cm', np.nan), default=np.nan)
    all_mrl_overall_map = _resample_map_to_target_grid(
        source_map=np.asarray(arrow_all.get('mrl_emp_map', np.array([])), dtype=float),
        source_x_edges=np.asarray(arrow_all.get('x_edges', np.array([])), dtype=float),
        source_y_edges=np.asarray(arrow_all.get('y_edges', np.array([])), dtype=float),
        target_x_edges=tx,
        target_y_edges=ty,
    )
    ss_mrl_overall_map = _resample_map_to_target_grid(
        source_map=np.asarray(arrow_ss.get('mrl_emp_map', np.array([])), dtype=float),
        source_x_edges=np.asarray(arrow_ss.get('x_edges', np.array([])), dtype=float),
        source_y_edges=np.asarray(arrow_ss.get('y_edges', np.array([])), dtype=float),
        target_x_edges=tx,
        target_y_edges=ty,
    )
    cs_mrl_overall_map = _resample_map_to_target_grid(
        source_map=np.asarray(arrow_cs.get('mrl_emp_map', np.array([])), dtype=float),
        source_x_edges=np.asarray(arrow_cs.get('x_edges', np.array([])), dtype=float),
        source_y_edges=np.asarray(arrow_cs.get('y_edges', np.array([])), dtype=float),
        target_x_edges=tx,
        target_y_edges=ty,
    )
    def _resample_emp_fit_vector_maps(arrow_fields):
        def _resample_vector(psi_key, mrl_key):
            psi_src = np.asarray((arrow_fields or {}).get(psi_key, np.array([])), dtype=float)
            mrl_src = np.asarray((arrow_fields or {}).get(mrl_key, np.array([])), dtype=float)
            vx_src = np.cos(psi_src) * np.clip(mrl_src, 0.0, 1.0)
            vy_src = np.sin(psi_src) * np.clip(mrl_src, 0.0, 1.0)
            vx_map = _resample_map_to_target_grid(
                source_map=np.asarray(vx_src, dtype=float),
                source_x_edges=np.asarray((arrow_fields or {}).get('x_edges', np.array([])), dtype=float),
                source_y_edges=np.asarray((arrow_fields or {}).get('y_edges', np.array([])), dtype=float),
                target_x_edges=tx,
                target_y_edges=ty,
            )
            vy_map = _resample_map_to_target_grid(
                source_map=np.asarray(vy_src, dtype=float),
                source_x_edges=np.asarray((arrow_fields or {}).get('x_edges', np.array([])), dtype=float),
                source_y_edges=np.asarray((arrow_fields or {}).get('y_edges', np.array([])), dtype=float),
                target_x_edges=tx,
                target_y_edges=ty,
            )
            if np.asarray(vx_map, dtype=float).shape != target_shape:
                vx_map = np.full(target_shape, np.nan, dtype=float)
            if np.asarray(vy_map, dtype=float).shape != target_shape:
                vy_map = np.full(target_shape, np.nan, dtype=float)
            return np.asarray(vx_map, dtype=float), np.asarray(vy_map, dtype=float)

        emp_x_map, emp_y_map = _resample_vector('psi_emp_map', 'mrl_emp_map')
        fit_x_map, fit_y_map = _resample_vector('psi_fit_map', 'mrl_fit_map')
        return emp_x_map, emp_y_map, fit_x_map, fit_y_map

    hd_emp_x_map, hd_emp_y_map, hd_fit_x_map_all, hd_fit_y_map_all = _resample_emp_fit_vector_maps(arrow_all)
    hd_emp_x_map_ss, hd_emp_y_map_ss, hd_fit_x_map_ss, hd_fit_y_map_ss = _resample_emp_fit_vector_maps(arrow_ss)
    hd_emp_x_map_cs, hd_emp_y_map_cs, hd_fit_x_map_cs, hd_fit_y_map_cs = _resample_emp_fit_vector_maps(arrow_cs)
    all_pref_occ_map = np.full(target_shape, np.nan, dtype=float)
    all_nonpref_occ_map = np.full(target_shape, np.nan, dtype=float)
    all_pref_speed_map = np.full(target_shape, np.nan, dtype=float)
    all_nonpref_speed_map = np.full(target_shape, np.nan, dtype=float)
    theta_mrl_overall_map = np.full(target_shape, np.nan, dtype=float)
    slow_mrl_overall_map = np.full(target_shape, np.nan, dtype=float)
    vel_x_map = np.full(target_shape, np.nan, dtype=float)
    vel_y_map = np.full(target_shape, np.nan, dtype=float)
    beh_hd_x_map = np.full(target_shape, np.nan, dtype=float)
    beh_hd_y_map = np.full(target_shape, np.nan, dtype=float)
    split_masks_payload = _compute_split_frame_masks_for_target_grid(
        data=stats_data,
        fit_info=fit_all,
        params=stats_params,
        preferred_lookup_specs=preferred_lookup_specs,
        split_source=str(split_source),
        preferred_half_width_deg=float(getattr(params, 'split_preferred_half_width_deg', 50.0)),
        target_x_edges=tx,
        target_y_edges=ty,
    )
    if bool(split_masks_payload.get('ok', False)):
        x_frames = np.asarray(stats_data.get('x_frames', np.array([])), dtype=float).reshape(-1)
        y_frames = np.asarray(stats_data.get('y_frames', np.array([])), dtype=float).reshape(-1)
        n_frames = int(x_frames.size)
        speed_frames_raw = np.asarray(stats_data.get('speed_frames', np.array([])), dtype=float).reshape(-1)
        speed_frames_sm = np.asarray(
            split_masks_payload.get('speed_smoothed_frames', np.array([])),
            dtype=float,
        ).reshape(-1)
        speed_frames = (
            speed_frames_sm
            if speed_frames_sm.size == n_frames
            else speed_frames_raw
        )
        dir_frames = np.asarray(stats_data.get('dir_frames', np.array([])), dtype=float).reshape(-1)
        valid_frames = np.asarray(split_masks_payload.get('valid_frames', np.array([])), dtype=bool).reshape(-1)
        if valid_frames.size != n_frames:
            valid_frames = np.ones(n_frames, dtype=bool)
        moving_mask = np.asarray(split_masks_payload.get('moving_mask_frames', np.array([])), dtype=bool).reshape(-1)
        if moving_mask.size != n_frames:
            moving_mask = np.zeros(n_frames, dtype=bool)
        masks = split_masks_payload.get('masks', {}) if isinstance(split_masks_payload.get('masks', {}), dict) else {}
        all_masks = masks.get('all', {}) if isinstance(masks.get('all', {}), dict) else {}
        ss_masks = masks.get('ss', {}) if isinstance(masks.get('ss', {}), dict) else {}
        cs_masks = masks.get('cs', {}) if isinstance(masks.get('cs', {}), dict) else {}
        frame_rate = _safe_float(stats_data.get('frame_rate', np.nan), default=np.nan)
        split_occ_thr = _safe_float(getattr(params, 'occupancy_threshold_split_s', np.nan), default=np.nan)
        if (not np.isfinite(split_occ_thr)) or split_occ_thr < 0:
            split_occ_thr = _safe_float(getattr(params, 'occupancy_threshold_s', np.nan), default=0.0)
        if (not np.isfinite(split_occ_thr)) or split_occ_thr < 0:
            split_occ_thr = 0.0
        min_occ_ang_bins = int(max(1, int(_safe_float(getattr(params, 'min_occupied_angle_bins', 1), default=1))))
        n_angle_bins = int(max(2, int(_safe_float(getattr(params, 'n_angle_bins', 10), default=10))))
        all_pref_occ_map = _compute_split_occupancy_time_map(
            x_frames=x_frames,
            y_frames=y_frames,
            split_mask=np.asarray(all_masks.get('preferred', np.zeros(n_frames, dtype=bool)), dtype=bool),
            valid_frames=valid_frames,
            x_edges=tx,
            y_edges=ty,
            frame_rate=frame_rate,
        )
        all_nonpref_occ_map = _compute_split_occupancy_time_map(
            x_frames=x_frames,
            y_frames=y_frames,
            split_mask=np.asarray(all_masks.get('nonpreferred', np.zeros(n_frames, dtype=bool)), dtype=bool),
            valid_frames=valid_frames,
            x_edges=tx,
            y_edges=ty,
            frame_rate=frame_rate,
        )
        all_pref_speed_map = _compute_split_framewise_mean_map(
            x_frames=x_frames,
            y_frames=y_frames,
            value_frames=speed_frames,
            split_mask=np.asarray(all_masks.get('preferred', np.zeros(n_frames, dtype=bool)), dtype=bool),
            valid_frames=valid_frames,
            x_edges=tx,
            y_edges=ty,
        )
        all_nonpref_speed_map = _compute_split_framewise_mean_map(
            x_frames=x_frames,
            y_frames=y_frames,
            value_frames=speed_frames,
            split_mask=np.asarray(all_masks.get('nonpreferred', np.zeros(n_frames, dtype=bool)), dtype=bool),
            valid_frames=valid_frames,
            x_edges=tx,
            y_edges=ty,
        )
        vx_frames, vy_frames = _compute_frame_velocity_components(
            x_frames=x_frames,
            y_frames=y_frames,
            speed_frames=speed_frames,
        )
        moving_use = np.asarray(moving_mask, dtype=bool)
        vel_x_map = _compute_split_framewise_mean_map(
            x_frames=x_frames,
            y_frames=y_frames,
            value_frames=vx_frames,
            split_mask=np.asarray(moving_use, dtype=bool),
            valid_frames=valid_frames,
            x_edges=tx,
            y_edges=ty,
        )
        vel_y_map = _compute_split_framewise_mean_map(
            x_frames=x_frames,
            y_frames=y_frames,
            value_frames=vy_frames,
            split_mask=np.asarray(moving_use, dtype=bool),
            valid_frames=valid_frames,
            x_edges=tx,
            y_edges=ty,
        )
        beh_hd_x_map, beh_hd_y_map, _ = _compute_split_heading_vector_map(
            x_frames=x_frames,
            y_frames=y_frames,
            dir_frames=dir_frames,
            split_mask=np.asarray(moving_use, dtype=bool),
            valid_frames=valid_frames,
            x_edges=tx,
            y_edges=ty,
        )
        theta_mrl_overall_map = _compute_value_heading_mrl_map(
            x_frames=x_frames,
            y_frames=y_frames,
            dir_frames=dir_frames,
            value_frames=np.asarray(stats_data.get('theta_amp_frames', np.array([])), dtype=float).reshape(-1),
            valid_frames=valid_frames,
            moving_frames=moving_mask,
            x_edges=tx,
            y_edges=ty,
            frame_rate=frame_rate,
            n_angle_bins=n_angle_bins,
            occupancy_threshold_s=split_occ_thr,
            min_occupied_angle_bins=min_occ_ang_bins,
            clip_negative_values=False,
        )
        slow_mrl_overall_map = _compute_value_heading_mrl_map(
            x_frames=x_frames,
            y_frames=y_frames,
            dir_frames=dir_frames,
            value_frames=np.asarray(stats_data.get('slow_vm_frames', np.array([])), dtype=float).reshape(-1),
            valid_frames=valid_frames,
            moving_frames=moving_mask,
            x_edges=tx,
            y_edges=ty,
            frame_rate=frame_rate,
            n_angle_bins=n_angle_bins,
            occupancy_threshold_s=split_occ_thr,
            min_occupied_angle_bins=min_occ_ang_bins,
            clip_negative_values=True,
        )

    all_pref_valid = np.isfinite(np.asarray(split_maps.get('all', {}).get('preferred_map', np.array([])), dtype=float))
    all_nonpref_valid = np.isfinite(np.asarray(split_maps.get('all', {}).get('nonpreferred_map', np.array([])), dtype=float))
    if all_pref_valid.shape == all_pref_occ_map.shape:
        all_pref_occ_map = np.asarray(all_pref_occ_map, dtype=float)
        all_pref_occ_map[~all_pref_valid] = np.nan
    if all_nonpref_valid.shape == all_nonpref_occ_map.shape:
        all_nonpref_occ_map = np.asarray(all_nonpref_occ_map, dtype=float)
        all_nonpref_occ_map[~all_nonpref_valid] = np.nan
    if all_pref_valid.shape == all_pref_speed_map.shape:
        all_pref_speed_map = np.asarray(all_pref_speed_map, dtype=float)
        all_pref_speed_map[~all_pref_valid] = np.nan
    if all_nonpref_valid.shape == all_nonpref_speed_map.shape:
        all_nonpref_speed_map = np.asarray(all_nonpref_speed_map, dtype=float)
        all_nonpref_speed_map[~all_nonpref_valid] = np.nan
    ss_pref_valid = np.isfinite(np.asarray(split_maps.get('ss', {}).get('preferred_map', np.array([])), dtype=float))
    ss_nonpref_valid = np.isfinite(np.asarray(split_maps.get('ss', {}).get('nonpreferred_map', np.array([])), dtype=float))
    cs_pref_valid = np.isfinite(np.asarray(split_maps.get('cs', {}).get('preferred_map', np.array([])), dtype=float))
    cs_nonpref_valid = np.isfinite(np.asarray(split_maps.get('cs', {}).get('nonpreferred_map', np.array([])), dtype=float))

    corr_valid_mask = np.asarray(all_pref_valid, dtype=bool) | np.asarray(all_nonpref_valid, dtype=bool)
    corr_valid_mask_ss = np.asarray(ss_pref_valid, dtype=bool) | np.asarray(ss_nonpref_valid, dtype=bool)
    corr_valid_mask_cs = np.asarray(cs_pref_valid, dtype=bool) | np.asarray(cs_nonpref_valid, dtype=bool)

    def _mask_to_valid(arr, valid_mask):
        out = np.asarray(arr, dtype=float)
        if out.shape == np.asarray(valid_mask, dtype=bool).shape:
            out = out.copy()
            out[~np.asarray(valid_mask, dtype=bool)] = np.nan
        return np.asarray(out, dtype=float)

    if corr_valid_mask.shape == hd_emp_x_map.shape:
        hd_emp_x_map = _mask_to_valid(hd_emp_x_map, corr_valid_mask)
    if corr_valid_mask.shape == hd_emp_y_map.shape:
        hd_emp_y_map = _mask_to_valid(hd_emp_y_map, corr_valid_mask)
    if corr_valid_mask.shape == vel_x_map.shape:
        vel_x_map = _mask_to_valid(vel_x_map, corr_valid_mask)
    if corr_valid_mask.shape == vel_y_map.shape:
        vel_y_map = _mask_to_valid(vel_y_map, corr_valid_mask)
    if corr_valid_mask.shape == beh_hd_x_map.shape:
        beh_hd_x_map = _mask_to_valid(beh_hd_x_map, corr_valid_mask)
    if corr_valid_mask.shape == beh_hd_y_map.shape:
        beh_hd_y_map = _mask_to_valid(beh_hd_y_map, corr_valid_mask)

    hd_fit_x_map_all = _mask_to_valid(hd_fit_x_map_all, corr_valid_mask)
    hd_fit_y_map_all = _mask_to_valid(hd_fit_y_map_all, corr_valid_mask)
    hd_emp_x_map_ss = _mask_to_valid(hd_emp_x_map_ss, corr_valid_mask_ss)
    hd_emp_y_map_ss = _mask_to_valid(hd_emp_y_map_ss, corr_valid_mask_ss)
    hd_fit_x_map_ss = _mask_to_valid(hd_fit_x_map_ss, corr_valid_mask_ss)
    hd_fit_y_map_ss = _mask_to_valid(hd_fit_y_map_ss, corr_valid_mask_ss)
    hd_emp_x_map_cs = _mask_to_valid(hd_emp_x_map_cs, corr_valid_mask_cs)
    hd_emp_y_map_cs = _mask_to_valid(hd_emp_y_map_cs, corr_valid_mask_cs)
    hd_fit_x_map_cs = _mask_to_valid(hd_fit_x_map_cs, corr_valid_mask_cs)
    hd_fit_y_map_cs = _mask_to_valid(hd_fit_y_map_cs, corr_valid_mask_cs)
    out_rows = []
    for region_name, region_mask in region_masks.items():
        n_region_bins = int(np.sum(np.asarray(region_mask, dtype=bool)))
        for metric_name, split_key in metric_map_keys.items():
            metric_payload = split_maps.get(split_key, {})
            pref_map = np.asarray(metric_payload.get('preferred_map', np.array([])), dtype=float)
            nonpref_map = np.asarray(metric_payload.get('nonpreferred_map', np.array([])), dtype=float)
            pref_mean, nonpref_mean, n_pref, n_nonpref = _paired_means_in_region(
                pref_map,
                nonpref_map,
                region_mask,
                joint_valid_bins=stats_joint_valid_bins,
            )
            if np.isfinite(pref_mean) and np.isfinite(nonpref_mean):
                delta = float(pref_mean - nonpref_mean)
            else:
                delta = np.nan
            out_rows.append({
                'animal_id': str(animal_id),
                'cell_idx': int(cell_idx),
                'cell_num': int(cell_idx) + 1,
                'category': str(category),
                'any_pass_threshold': int(any_pass_threshold),
                'region': str(region_name),
                'metric': str(metric_name),
                'preferred_mean': float(pref_mean) if np.isfinite(pref_mean) else np.nan,
                'nonpreferred_mean': float(nonpref_mean) if np.isfinite(nonpref_mean) else np.nan,
                'delta_pref_minus_nonpref': float(delta) if np.isfinite(delta) else np.nan,
                'n_pref_bins': int(n_pref),
                    'n_nonpref_bins': int(n_nonpref),
                    'n_region_bins': int(n_region_bins),
                    'is_place_cell': bool(is_place_cell),
                    'pass_100_any': bool(pass_100_any),
                    'split_source': str(split_source),
                    'stats_pref_reference': str(stats_pref_reference),
                    'stats_joint_valid_bins': bool(stats_joint_valid_bins),
                    'split_bin_size_cm': float(split_bin_size_cm) if np.isfinite(split_bin_size_cm) else np.nan,
                    'stats_smooth_sigma_cm': 0.0,
                    'stats_occ_smooth_sigma_cm': 0.0,
                    'pf_mask_mode': str(pf_mask_mode_label),
                    'hd_vel_corr_method': str(hd_vel_corr_method),
                })
        occ_pref_mean, occ_nonpref_mean, occ_n_pref, occ_n_nonpref = _paired_means_in_region(
            all_pref_occ_map,
            all_nonpref_occ_map,
            region_mask,
            joint_valid_bins=stats_joint_valid_bins,
        )
        occ_delta = (
            float(occ_pref_mean - occ_nonpref_mean)
            if (np.isfinite(occ_pref_mean) and np.isfinite(occ_nonpref_mean))
            else np.nan
        )
        out_rows.append({
            'animal_id': str(animal_id),
            'cell_idx': int(cell_idx),
            'cell_num': int(cell_idx) + 1,
            'category': str(category),
            'any_pass_threshold': int(any_pass_threshold),
            'region': str(region_name),
            'metric': 'occupancy_time',
            'preferred_mean': float(occ_pref_mean) if np.isfinite(occ_pref_mean) else np.nan,
            'nonpreferred_mean': float(occ_nonpref_mean) if np.isfinite(occ_nonpref_mean) else np.nan,
            'delta_pref_minus_nonpref': float(occ_delta) if np.isfinite(occ_delta) else np.nan,
            'n_pref_bins': int(occ_n_pref),
            'n_nonpref_bins': int(occ_n_nonpref),
            'n_region_bins': int(n_region_bins),
            'is_place_cell': bool(is_place_cell),
            'pass_100_any': bool(pass_100_any),
            'split_source': str(split_source),
            'stats_pref_reference': str(stats_pref_reference),
            'stats_joint_valid_bins': bool(stats_joint_valid_bins),
            'split_bin_size_cm': float(split_bin_size_cm) if np.isfinite(split_bin_size_cm) else np.nan,
            'stats_smooth_sigma_cm': 0.0,
            'stats_occ_smooth_sigma_cm': 0.0,
            'pf_mask_mode': str(pf_mask_mode_label),
            'hd_vel_corr_method': str(hd_vel_corr_method),
        })
        speed_pref_mean, speed_nonpref_mean, speed_n_pref, speed_n_nonpref = _paired_means_in_region(
            all_pref_speed_map,
            all_nonpref_speed_map,
            region_mask,
            joint_valid_bins=stats_joint_valid_bins,
        )
        speed_delta = (
            float(speed_pref_mean - speed_nonpref_mean)
            if (np.isfinite(speed_pref_mean) and np.isfinite(speed_nonpref_mean))
            else np.nan
        )
        out_rows.append({
            'animal_id': str(animal_id),
            'cell_idx': int(cell_idx),
            'cell_num': int(cell_idx) + 1,
            'category': str(category),
            'any_pass_threshold': int(any_pass_threshold),
            'region': str(region_name),
            'metric': 'speed',
            'preferred_mean': float(speed_pref_mean) if np.isfinite(speed_pref_mean) else np.nan,
            'nonpreferred_mean': float(speed_nonpref_mean) if np.isfinite(speed_nonpref_mean) else np.nan,
            'delta_pref_minus_nonpref': float(speed_delta) if np.isfinite(speed_delta) else np.nan,
            'n_pref_bins': int(speed_n_pref),
            'n_nonpref_bins': int(speed_n_nonpref),
            'n_region_bins': int(n_region_bins),
            'is_place_cell': bool(is_place_cell),
            'pass_100_any': bool(pass_100_any),
            'split_source': str(split_source),
            'stats_pref_reference': str(stats_pref_reference),
            'stats_joint_valid_bins': bool(stats_joint_valid_bins),
            'split_bin_size_cm': float(split_bin_size_cm) if np.isfinite(split_bin_size_cm) else np.nan,
            'stats_smooth_sigma_cm': 0.0,
            'stats_occ_smooth_sigma_cm': 0.0,
            'pf_mask_mode': str(pf_mask_mode_label),
            'hd_vel_corr_method': str(hd_vel_corr_method),
        })
        corr_val, corr_n = _hd_vel_correlation_in_region(
            hd_x_map=hd_emp_x_map,
            hd_y_map=hd_emp_y_map,
            vel_x_map=vel_x_map,
            vel_y_map=vel_y_map,
            region_mask=region_mask,
            method=str(hd_vel_corr_method),
        )
        out_rows.append({
            'animal_id': str(animal_id),
            'cell_idx': int(cell_idx),
            'cell_num': int(cell_idx) + 1,
            'category': str(category),
            'any_pass_threshold': int(any_pass_threshold),
            'region': str(region_name),
            'metric': 'hd_vel_corr',
            # Store a single overall value in both columns for long-format compatibility.
            'preferred_mean': float(corr_val) if np.isfinite(corr_val) else np.nan,
            'nonpreferred_mean': float(corr_val) if np.isfinite(corr_val) else np.nan,
            'delta_pref_minus_nonpref': 0.0 if np.isfinite(corr_val) else np.nan,
            'n_pref_bins': int(corr_n),
            'n_nonpref_bins': int(corr_n),
            'n_region_bins': int(n_region_bins),
            'is_place_cell': bool(is_place_cell),
            'pass_100_any': bool(pass_100_any),
            'split_source': str(split_source),
            'stats_pref_reference': str(stats_pref_reference),
            'stats_joint_valid_bins': bool(stats_joint_valid_bins),
            'split_bin_size_cm': float(split_bin_size_cm) if np.isfinite(split_bin_size_cm) else np.nan,
            'stats_smooth_sigma_cm': 0.0,
            'stats_occ_smooth_sigma_cm': 0.0,
            'pf_mask_mode': str(pf_mask_mode_label),
            'hd_vel_corr_method': str(hd_vel_corr_method),
        })
        hd_pref_corr_val, hd_pref_corr_n = _hd_vel_correlation_in_region(
            hd_x_map=hd_emp_x_map,
            hd_y_map=hd_emp_y_map,
            vel_x_map=beh_hd_x_map,
            vel_y_map=beh_hd_y_map,
            region_mask=region_mask,
            method='mean_raw_dot',
        )
        out_rows.append({
            'animal_id': str(animal_id),
            'cell_idx': int(cell_idx),
            'cell_num': int(cell_idx) + 1,
            'category': str(category),
            'any_pass_threshold': int(any_pass_threshold),
            'region': str(region_name),
            'metric': 'hd_pref_neural_vs_behavior',
            # Store single-value correlation in both columns for long-format compatibility.
            'preferred_mean': float(hd_pref_corr_val) if np.isfinite(hd_pref_corr_val) else np.nan,
            'nonpreferred_mean': float(hd_pref_corr_val) if np.isfinite(hd_pref_corr_val) else np.nan,
            'delta_pref_minus_nonpref': 0.0 if np.isfinite(hd_pref_corr_val) else np.nan,
            'n_pref_bins': int(hd_pref_corr_n),
            'n_nonpref_bins': int(hd_pref_corr_n),
            'n_region_bins': int(n_region_bins),
            'is_place_cell': bool(is_place_cell),
            'pass_100_any': bool(pass_100_any),
            'split_source': str(split_source),
            'stats_pref_reference': str(stats_pref_reference),
            'stats_joint_valid_bins': bool(stats_joint_valid_bins),
            'split_bin_size_cm': float(split_bin_size_cm) if np.isfinite(split_bin_size_cm) else np.nan,
            'stats_smooth_sigma_cm': 0.0,
            'stats_occ_smooth_sigma_cm': 0.0,
            'pf_mask_mode': str(pf_mask_mode_label),
            'hd_vel_corr_method': 'mean_raw_dot',
        })
        eb_empfit_corr_method = 'normalized_dot'
        for metric_name, emp_x_local, emp_y_local, fit_x_local, fit_y_local in (
            ('eb_empfit_corr_all', hd_emp_x_map, hd_emp_y_map, hd_fit_x_map_all, hd_fit_y_map_all),
            ('eb_empfit_corr_ss', hd_emp_x_map_ss, hd_emp_y_map_ss, hd_fit_x_map_ss, hd_fit_y_map_ss),
            ('eb_empfit_corr_cs', hd_emp_x_map_cs, hd_emp_y_map_cs, hd_fit_x_map_cs, hd_fit_y_map_cs),
        ):
            eb_corr_val, eb_corr_n = _hd_vel_correlation_in_region(
                hd_x_map=emp_x_local,
                hd_y_map=emp_y_local,
                vel_x_map=fit_x_local,
                vel_y_map=fit_y_local,
                region_mask=region_mask,
                method=str(eb_empfit_corr_method),
            )
            out_rows.append({
                'animal_id': str(animal_id),
                'cell_idx': int(cell_idx),
                'cell_num': int(cell_idx) + 1,
                'category': str(category),
                'any_pass_threshold': int(any_pass_threshold),
                'region': str(region_name),
                'metric': str(metric_name),
                'preferred_mean': float(eb_corr_val) if np.isfinite(eb_corr_val) else np.nan,
                'nonpreferred_mean': float(eb_corr_val) if np.isfinite(eb_corr_val) else np.nan,
                'delta_pref_minus_nonpref': 0.0 if np.isfinite(eb_corr_val) else np.nan,
                'n_pref_bins': int(eb_corr_n),
                'n_nonpref_bins': int(eb_corr_n),
                'n_region_bins': int(n_region_bins),
                'is_place_cell': bool(is_place_cell),
                'pass_100_any': bool(pass_100_any),
                'split_source': str(split_source),
                'stats_pref_reference': str(stats_pref_reference),
                'stats_joint_valid_bins': bool(stats_joint_valid_bins),
                'split_bin_size_cm': float(split_bin_size_cm) if np.isfinite(split_bin_size_cm) else np.nan,
                'stats_smooth_sigma_cm': 0.0,
                'stats_occ_smooth_sigma_cm': 0.0,
                'pf_mask_mode': str(pf_mask_mode_label),
                'hd_vel_corr_method': str(eb_empfit_corr_method),
            })
        for metric_name, overall_map in (
            ('all_mrl_overall', all_mrl_overall_map),
            ('ss_mrl_overall', ss_mrl_overall_map),
            ('cs_mrl_overall', cs_mrl_overall_map),
            ('theta_mrl_overall', theta_mrl_overall_map),
            ('slow_mrl_overall', slow_mrl_overall_map),
        ):
            overall_mean, n_overall = _mean_in_region(overall_map, region_mask)
            out_rows.append({
                'animal_id': str(animal_id),
                'cell_idx': int(cell_idx),
                'cell_num': int(cell_idx) + 1,
                'category': str(category),
                'any_pass_threshold': int(any_pass_threshold),
                'region': str(region_name),
                'metric': str(metric_name),
                # Store a single overall value in both columns for compatibility with long-format schema.
                'preferred_mean': float(overall_mean) if np.isfinite(overall_mean) else np.nan,
                'nonpreferred_mean': float(overall_mean) if np.isfinite(overall_mean) else np.nan,
                'delta_pref_minus_nonpref': 0.0 if np.isfinite(overall_mean) else np.nan,
                'n_pref_bins': int(n_overall),
                'n_nonpref_bins': int(n_overall),
                'n_region_bins': int(n_region_bins),
                'is_place_cell': bool(is_place_cell),
                'pass_100_any': bool(pass_100_any),
                'split_source': str(split_source),
                'stats_pref_reference': str(stats_pref_reference),
                'stats_joint_valid_bins': bool(stats_joint_valid_bins),
                'split_bin_size_cm': float(split_bin_size_cm) if np.isfinite(split_bin_size_cm) else np.nan,
                'stats_smooth_sigma_cm': 0.0,
                'stats_occ_smooth_sigma_cm': 0.0,
                'pf_mask_mode': str(pf_mask_mode_label),
                'hd_vel_corr_method': str(hd_vel_corr_method),
            })
    for row in out_rows:
        row.setdefault('selected_in_pass_any_folder', False)
    return out_rows


def _build_preferred_lookup_specs_for_stats(
    *,
    local_all,
    local_ss,
    local_cs,
    arrow_all,
    arrow_ss,
    arrow_cs,
    stats_pref_reference: str,
    split_source: str,
):
    stats_pref_reference = _sanitize_stats_pref_reference(stats_pref_reference)
    split_source = _sanitize_split_source(split_source)
    emp_lookup_all = _compute_empirical_direction_lookup_all_bins(local_tuning=local_all if isinstance(local_all, dict) else {})
    emp_lookup_ss = _compute_empirical_direction_lookup_all_bins(local_tuning=local_ss if isinstance(local_ss, dict) else {})
    emp_lookup_cs = _compute_empirical_direction_lookup_all_bins(local_tuning=local_cs if isinstance(local_cs, dict) else {})

    if stats_pref_reference == 'matching_metric':
        ss_emp_lookup = emp_lookup_ss
        cs_emp_lookup = emp_lookup_cs
        ss_lookup = arrow_ss
        cs_lookup = arrow_cs
    else:
        ss_emp_lookup = emp_lookup_all
        cs_emp_lookup = emp_lookup_all
        ss_lookup = arrow_all
        cs_lookup = arrow_all

    def _lookup_spec_from_empirical_or_fallback(emp_lookup: dict, fallback_lookup: dict) -> dict:
        fb = fallback_lookup if isinstance(fallback_lookup, dict) else {}
        valid_mask_col8 = np.asarray(fb.get('arrow_valid_mask', np.array([])), dtype=bool)

        def _apply_col8_valid_mask(dir_map_in: np.ndarray) -> np.ndarray:
            dir_map_out = np.asarray(dir_map_in, dtype=float).copy()
            if (
                str(split_source) == 'empirical'
                and valid_mask_col8.ndim == 2
                and dir_map_out.ndim == 2
                and valid_mask_col8.shape == dir_map_out.shape
            ):
                dir_map_out[~valid_mask_col8] = np.nan
            return np.asarray(dir_map_out, dtype=float)

        x_emp = np.asarray(emp_lookup.get('x_edges', np.array([])), dtype=float) if isinstance(emp_lookup, dict) else np.array([], dtype=float)
        y_emp = np.asarray(emp_lookup.get('y_edges', np.array([])), dtype=float) if isinstance(emp_lookup, dict) else np.array([], dtype=float)
        d_emp = np.asarray(emp_lookup.get('dir_map', np.array([])), dtype=float) if isinstance(emp_lookup, dict) else np.array([], dtype=float)
        if (
            x_emp.size >= 2
            and y_emp.size >= 2
            and d_emp.shape == (int(x_emp.size - 1), int(y_emp.size - 1))
        ):
            return {
                'x_edges': np.asarray(x_emp, dtype=float),
                'y_edges': np.asarray(y_emp, dtype=float),
                'dir_map': _apply_col8_valid_mask(np.asarray(d_emp, dtype=float)),
            }
        return {
            'x_edges': np.asarray(fb.get('x_edges', np.array([])), dtype=float),
            'y_edges': np.asarray(fb.get('y_edges', np.array([])), dtype=float),
            'dir_map': _apply_col8_valid_mask(np.asarray(fb.get('psi_emp_map', np.array([])), dtype=float)),
        }

    return {
        'all': _lookup_spec_from_empirical_or_fallback(emp_lookup_all, arrow_all),
        'ss': _lookup_spec_from_empirical_or_fallback(ss_emp_lookup, ss_lookup),
        'cs': _lookup_spec_from_empirical_or_fallback(cs_emp_lookup, cs_lookup),
    }


def _compute_shuffle_null_payload_for_plot(
    *,
    plot_data,
    row_all,
    row_ss,
    row_cs,
    params,
    split_source: str,
    stats_pref_reference: str,
    n_shuffles: int,
    rng_seed: int,
):
    out = {
        'ok': False,
        'reason': 'unknown',
        'n_shuffles': int(max(1, int(n_shuffles))),
        'rows': [],
    }
    if not isinstance(plot_data, dict):
        out['reason'] = 'invalid_plot_data'
        return out

    local_all = plot_data.get('local_tuning_all')
    local_ss = plot_data.get('local_tuning_ss')
    local_cs = plot_data.get('local_tuning_cs')
    fit_all = _resolve_plot_fit_params(local_tuning=local_all, summary_row=row_all if isinstance(row_all, dict) else {})
    fit_ss = _resolve_plot_fit_params(local_tuning=local_ss, summary_row=row_ss if isinstance(row_ss, dict) else row_all)
    fit_cs = _resolve_plot_fit_params(local_tuning=local_cs, summary_row=row_cs if isinstance(row_cs, dict) else row_all)
    arrow_all = _compute_spatial_arrow_fields(local_tuning=local_all, fit_info=fit_all, params=params)
    arrow_ss = _compute_spatial_arrow_fields(local_tuning=local_ss, fit_info=fit_ss, params=params)
    arrow_cs = _compute_spatial_arrow_fields(local_tuning=local_cs, fit_info=fit_cs, params=params)

    split_source = _sanitize_split_source(split_source)
    preferred_lookup_specs = _build_preferred_lookup_specs_for_stats(
        local_all=local_all,
        local_ss=local_ss,
        local_cs=local_cs,
        arrow_all=arrow_all,
        arrow_ss=arrow_ss,
        arrow_cs=arrow_cs,
        stats_pref_reference=stats_pref_reference,
        split_source=split_source,
    )

    stats_data = dict(plot_data)
    pcfg = dict(plot_data.get('placecell_map_params', {})) if isinstance(plot_data.get('placecell_map_params', {}), dict) else {}
    # Match PF-split stats basis (unsmoothed maps for scalar summaries).
    pcfg['smooth_sigma'] = 0.0
    pcfg['occ_smooth_sigma'] = 0.0
    stats_data['placecell_map_params'] = pcfg
    stats_params = copy.deepcopy(params)
    stats_params.pc_smooth_sigma = 0.0
    stats_params.pc_occ_smooth_sigma = 0.0

    split_maps = _compute_placecell_style_preferred_nonpreferred_maps(
        data=stats_data,
        fit_info=fit_all,
        params=stats_params,
        preferred_angle_source=split_source,
        preferred_lookup_specs=preferred_lookup_specs,
        preferred_half_width_deg=float(getattr(params, 'split_preferred_half_width_deg', 50.0)),
        apply_min_occupancy_mask=True,
    )
    if not bool(split_maps.get('ok', False)):
        out['reason'] = f"split_maps_failed({split_maps.get('reason', 'unknown')})"
        return out

    tx = np.asarray(split_maps.get('x_edges', np.array([])), dtype=float)
    ty = np.asarray(split_maps.get('y_edges', np.array([])), dtype=float)
    nx = int(max(tx.size - 1, 0))
    ny = int(max(ty.size - 1, 0))
    if nx <= 0 or ny <= 0:
        out['reason'] = 'invalid_target_edges'
        return out

    split_masks_payload = _compute_split_frame_masks_for_target_grid(
        data=stats_data,
        fit_info=fit_all,
        params=stats_params,
        preferred_lookup_specs=preferred_lookup_specs,
        split_source=split_source,
        preferred_half_width_deg=float(getattr(params, 'split_preferred_half_width_deg', 50.0)),
        target_x_edges=tx,
        target_y_edges=ty,
    )
    if not bool(split_masks_payload.get('ok', False)):
        out['reason'] = f"split_masks_failed({split_masks_payload.get('reason', 'unknown')})"
        return out

    x_arr = np.asarray(stats_data.get('x_frames', np.array([])), dtype=float).reshape(-1)
    y_arr = np.asarray(stats_data.get('y_frames', np.array([])), dtype=float).reshape(-1)
    dir_arr = _wrap_angle_to_pi_local(np.asarray(stats_data.get('dir_frames', np.array([])), dtype=float).reshape(-1))
    theta_vals = np.asarray(stats_data.get('theta_amp_frames', np.array([])), dtype=float).reshape(-1)
    slow_vals = np.asarray(stats_data.get('slow_vm_frames', np.array([])), dtype=float).reshape(-1)
    raw_all = np.asarray(stats_data.get('raw_all_spike_frames', np.array([])), dtype=int).reshape(-1)
    raw_ss = np.asarray(stats_data.get('raw_ss_spike_frames', np.array([])), dtype=int).reshape(-1)
    raw_cs = np.asarray(stats_data.get('raw_cs_spike_frames', np.array([])), dtype=int).reshape(-1)
    valid_frames = np.asarray(split_masks_payload.get('valid_frames', np.array([])), dtype=bool).reshape(-1)
    moving_mask = np.asarray(split_masks_payload.get('moving_mask_frames', np.array([])), dtype=bool).reshape(-1)
    n_frames = int(x_arr.size)
    if (
        n_frames <= 0
        or any(arr.size != n_frames for arr in (y_arr, dir_arr, theta_vals, slow_vals, valid_frames, moving_mask))
    ):
        out['reason'] = 'length_mismatch'
        return out

    frame_rate = _safe_float(stats_data.get('frame_rate', np.nan), default=np.nan)
    if (not np.isfinite(frame_rate)) or frame_rate <= 0:
        out['reason'] = 'invalid_frame_rate'
        return out
    split_occ_thr = _safe_float(getattr(params, 'occupancy_threshold_split_s', np.nan), default=np.nan)
    if (not np.isfinite(split_occ_thr)) or split_occ_thr < 0:
        split_occ_thr = _safe_float(getattr(params, 'occupancy_threshold_s', np.nan), default=0.1)
    if (not np.isfinite(split_occ_thr)) or split_occ_thr < 0:
        split_occ_thr = 0.1

    xi = _digitize_with_upper_edge_inclusive(x_arr, tx)
    yi = _digitize_with_upper_edge_inclusive(y_arr, ty)
    in_bounds = (
        np.isfinite(x_arr) & np.isfinite(y_arr)
        & (xi >= 0) & (yi >= 0)
        & (xi < nx) & (yi < ny)
    )
    frame_bin_flat = np.full(n_frames, -1, dtype=int)
    if np.any(in_bounds):
        frame_bin_flat[in_bounds] = (xi[in_bounds].astype(int) * ny + yi[in_bounds].astype(int))
    n_bins = int(nx * ny)

    def _prepare_spike_frames(raw_frames: np.ndarray) -> np.ndarray:
        spk = np.asarray(raw_frames, dtype=int).reshape(-1)
        if spk.size <= 0:
            return np.array([], dtype=int)
        spk = spk[(spk >= 0) & (spk < n_frames)]
        if spk.size <= 0:
            return np.array([], dtype=int)
        spk = np.unique(spk)
        spk = spk[np.asarray(valid_frames[spk], dtype=bool)]
        spk = spk[np.asarray(frame_bin_flat[spk] >= 0, dtype=bool)]
        return np.asarray(spk, dtype=int)

    spk_all = _prepare_spike_frames(raw_all)
    spk_ss = _prepare_spike_frames(raw_ss)
    spk_cs = _prepare_spike_frames(raw_cs)
    pref_half_width_rad = float(np.deg2rad(float(getattr(params, 'split_preferred_half_width_deg', 50.0))))

    fit_dir_tx, fit_has_ref = _compute_fit_reference_direction_map(fit_all, tx, ty)

    def _resolve_lookup_spec(key: str):
        fallback = (np.asarray(tx, dtype=float), np.asarray(ty, dtype=float), np.asarray(fit_dir_tx, dtype=float))
        if split_source != 'empirical' or not isinstance(preferred_lookup_specs, dict):
            return fallback
        spec = preferred_lookup_specs.get(str(key), {})
        if not isinstance(spec, dict):
            return fallback
        x_lookup = np.asarray(spec.get('x_edges', np.array([])), dtype=float)
        y_lookup = np.asarray(spec.get('y_edges', np.array([])), dtype=float)
        dir_lookup = np.asarray(spec.get('dir_map', np.array([])), dtype=float)
        if x_lookup.size < 2 or y_lookup.size < 2:
            return fallback
        if dir_lookup.shape != (int(x_lookup.size - 1), int(y_lookup.size - 1)):
            return fallback
        return np.asarray(x_lookup, dtype=float), np.asarray(y_lookup, dtype=float), np.asarray(dir_lookup, dtype=float)

    def _build_key_shuffle_info(key: str):
        x_lookup, y_lookup, dir_lookup = _resolve_lookup_spec(key)
        if (
            split_source == 'fit'
            and (not bool(fit_has_ref))
            and np.all(~np.isfinite(np.asarray(dir_lookup, dtype=float)))
        ):
            return {
                'idx': np.array([], dtype=int),
                'dir': np.array([], dtype=float),
                'lookup_flat': np.array([], dtype=int),
                'ref_base': np.array([], dtype=float),
                'n_lookup_bins': 0,
            }
        xi_lk = _digitize_with_upper_edge_inclusive(x_arr, x_lookup)
        yi_lk = _digitize_with_upper_edge_inclusive(y_arr, y_lookup)
        in_bounds_lk = (
            (xi_lk >= 0) & (yi_lk >= 0)
            & (xi_lk < int(x_lookup.size - 1))
            & (yi_lk < int(y_lookup.size - 1))
        )
        eligible = (
            np.asarray(valid_frames, dtype=bool)
            & np.asarray(moving_mask, dtype=bool)
            & np.isfinite(dir_arr)
            & np.isfinite(x_arr)
            & np.isfinite(y_arr)
            & in_bounds_lk
        )
        idx = np.where(eligible)[0].astype(int)
        if idx.size <= 0:
            return {
                'idx': np.array([], dtype=int),
                'dir': np.array([], dtype=float),
                'lookup_flat': np.array([], dtype=int),
                'ref_base': np.array([], dtype=float),
                'n_lookup_bins': int(max(0, (x_lookup.size - 1) * (y_lookup.size - 1))),
            }
        ii = xi_lk[idx].astype(int)
        jj = yi_lk[idx].astype(int)
        ref = np.asarray(dir_lookup, dtype=float)[ii, jj]
        has_ref = np.isfinite(ref)
        idx = idx[has_ref]
        if idx.size <= 0:
            return {
                'idx': np.array([], dtype=int),
                'dir': np.array([], dtype=float),
                'lookup_flat': np.array([], dtype=int),
                'ref_base': np.array([], dtype=float),
                'n_lookup_bins': int(max(0, (x_lookup.size - 1) * (y_lookup.size - 1))),
            }
        ii = xi_lk[idx].astype(int)
        jj = yi_lk[idx].astype(int)
        lookup_flat = (ii * int(y_lookup.size - 1) + jj).astype(int)
        ref = np.asarray(dir_lookup, dtype=float)[ii, jj]
        return {
            'idx': np.asarray(idx, dtype=int),
            'dir': np.asarray(dir_arr[idx], dtype=float),
            'lookup_flat': np.asarray(lookup_flat, dtype=int),
            'ref_base': np.asarray(ref, dtype=float),
            'n_lookup_bins': int(max(0, (x_lookup.size - 1) * (y_lookup.size - 1))),
        }

    key_info = {
        'all': _build_key_shuffle_info('all'),
        'ss': _build_key_shuffle_info('ss'),
        'cs': _build_key_shuffle_info('cs'),
    }

    def _mask_from_key_info(info: dict, *, random_delta: bool, rng: np.random.Generator | None = None) -> np.ndarray:
        m = np.zeros(n_frames, dtype=bool)
        idx = np.asarray(info.get('idx', np.array([])), dtype=int)
        if idx.size <= 0:
            return m
        dirs = np.asarray(info.get('dir', np.array([])), dtype=float)
        ref_base = np.asarray(info.get('ref_base', np.array([])), dtype=float)
        lookup_flat = np.asarray(info.get('lookup_flat', np.array([])), dtype=int)
        if random_delta:
            n_lk = int(max(0, int(info.get('n_lookup_bins', 0))))
            if n_lk <= 0 or rng is None:
                return m
            delta = rng.uniform(-np.pi, np.pi, size=n_lk)
            ref = _wrap_angle_to_pi_local(ref_base + delta[lookup_flat])
        else:
            ref = np.asarray(ref_base, dtype=float)
        delta_ang = _wrap_angle_to_pi_local(dirs - ref)
        is_pref = np.abs(delta_ang) <= (pref_half_width_rad + 1e-12)
        if np.any(is_pref):
            m[idx[is_pref]] = True
        return m

    def _metric_pref_mean(metric_name: str, pref_mask: np.ndarray) -> tuple[float, int]:
        mask = np.asarray(pref_mask, dtype=bool).reshape(-1)
        if mask.size != n_frames:
            return np.nan, 0
        usable_frames = mask & np.asarray(valid_frames, dtype=bool) & (frame_bin_flat >= 0)
        if not np.any(usable_frames):
            return np.nan, 0
        pref_occ = np.bincount(
            frame_bin_flat[usable_frames].astype(int),
            minlength=n_bins,
        ).astype(float) / float(frame_rate)
        occ_ok = np.isfinite(pref_occ) & (pref_occ >= float(split_occ_thr))
        if str(metric_name) in {'all', 'ss', 'cs'}:
            spk = spk_all if str(metric_name) == 'all' else (spk_ss if str(metric_name) == 'ss' else spk_cs)
            if spk.size > 0:
                spk_use = spk[np.asarray(mask[spk], dtype=bool)]
            else:
                spk_use = np.array([], dtype=int)
            spk_counts = np.zeros(n_bins, dtype=float)
            if spk_use.size > 0:
                spk_bins = frame_bin_flat[spk_use]
                spk_bins = spk_bins[spk_bins >= 0]
                if spk_bins.size > 0:
                    spk_counts = np.bincount(spk_bins.astype(int), minlength=n_bins).astype(float)
            rate = np.full(n_bins, np.nan, dtype=float)
            with np.errstate(invalid='ignore', divide='ignore'):
                rate[occ_ok] = spk_counts[occ_ok] / pref_occ[occ_ok]
            keep = np.isfinite(rate)
            if np.sum(keep) <= 0:
                return np.nan, 0
            return float(np.nanmean(rate[keep])), int(np.sum(keep))
        vals = theta_vals if str(metric_name) == 'theta' else slow_vals
        finite_vals = usable_frames & np.isfinite(vals)
        if not np.any(finite_vals):
            return np.nan, 0
        cnt = np.bincount(frame_bin_flat[finite_vals].astype(int), minlength=n_bins).astype(float)
        sums = np.bincount(frame_bin_flat[finite_vals].astype(int), weights=np.asarray(vals[finite_vals], dtype=float), minlength=n_bins).astype(float)
        mean_map = np.full(n_bins, np.nan, dtype=float)
        has_cnt = cnt > 0
        with np.errstate(invalid='ignore', divide='ignore'):
            mean_map[has_cnt] = sums[has_cnt] / cnt[has_cnt]
        mean_map[~occ_ok] = np.nan
        keep = np.isfinite(mean_map)
        if np.sum(keep) <= 0:
            return np.nan, 0
        return float(np.nanmean(mean_map[keep])), int(np.sum(keep))

    emp_masks = split_masks_payload.get('masks', {}) if isinstance(split_masks_payload.get('masks', {}), dict) else {}
    emp_pref_all = np.asarray((emp_masks.get('all', {}) or {}).get('preferred', np.zeros(n_frames, dtype=bool)), dtype=bool)
    emp_nonpref_all = np.asarray((emp_masks.get('all', {}) or {}).get('nonpreferred', np.zeros(n_frames, dtype=bool)), dtype=bool)
    emp_pref_ss = np.asarray((emp_masks.get('ss', {}) or {}).get('preferred', np.zeros(n_frames, dtype=bool)), dtype=bool)
    emp_nonpref_ss = np.asarray((emp_masks.get('ss', {}) or {}).get('nonpreferred', np.zeros(n_frames, dtype=bool)), dtype=bool)
    emp_pref_cs = np.asarray((emp_masks.get('cs', {}) or {}).get('preferred', np.zeros(n_frames, dtype=bool)), dtype=bool)
    emp_nonpref_cs = np.asarray((emp_masks.get('cs', {}) or {}).get('nonpreferred', np.zeros(n_frames, dtype=bool)), dtype=bool)

    row_defs = [
        ('all', 'All', 'all', emp_pref_all, emp_nonpref_all),
        ('ss', 'SS', 'ss', emp_pref_ss, emp_nonpref_ss),
        ('cs', 'CS', 'cs', emp_pref_cs, emp_nonpref_cs),
        ('theta', 'Theta', 'all', emp_pref_all, emp_nonpref_all),
        ('slow', 'Slow Vm', 'all', emp_pref_all, emp_nonpref_all),
    ]

    n_shuf = int(max(1, int(n_shuffles)))
    rng = np.random.default_rng(int(max(0, int(rng_seed))))
    shuffle_pref = {k: np.full(n_shuf, np.nan, dtype=float) for k, *_ in row_defs}

    key_pref_masks_per_shuffle = {'all': None, 'ss': None, 'cs': None}
    for s_idx in range(n_shuf):
        for key in ('all', 'ss', 'cs'):
            key_pref_masks_per_shuffle[key] = _mask_from_key_info(key_info[key], random_delta=True, rng=rng)
        for metric_key, _label, key_for_shuffle, _emp_pref_mask, _emp_nonpref_mask in row_defs:
            mean_pref, _ = _metric_pref_mean(metric_key, key_pref_masks_per_shuffle[str(key_for_shuffle)])
            shuffle_pref[metric_key][s_idx] = float(mean_pref) if np.isfinite(mean_pref) else np.nan

    rows_payload = []
    for metric_key, label, _key_for_shuffle, emp_pref_mask, emp_nonpref_mask in row_defs:
        emp_pref_mean, emp_pref_n = _metric_pref_mean(metric_key, emp_pref_mask)
        emp_nonpref_mean, emp_nonpref_n = _metric_pref_mean(metric_key, emp_nonpref_mask)
        rows_payload.append(
            {
                'metric': str(metric_key),
                'label': str(label),
                'shuffle_preferred': np.asarray(shuffle_pref[metric_key], dtype=float),
                'empirical_preferred': float(emp_pref_mean) if np.isfinite(emp_pref_mean) else np.nan,
                'empirical_nonpreferred': float(emp_nonpref_mean) if np.isfinite(emp_nonpref_mean) else np.nan,
                'empirical_preferred_n_bins': int(emp_pref_n),
                'empirical_nonpreferred_n_bins': int(emp_nonpref_n),
            }
        )

    out['ok'] = True
    out['reason'] = 'ok'
    out['n_shuffles'] = int(n_shuf)
    out['split_source'] = str(split_source)
    out['stats_pref_reference'] = str(stats_pref_reference)
    out['preferred_half_width_deg'] = float(getattr(params, 'split_preferred_half_width_deg', 50.0))
    out['rows'] = rows_payload
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Generate per-cell summaries for union(pass_<threshold> across 3 head spike types)'
    )
    parser.add_argument('--base-dir', type=str, required=True,
                        help='Root folder containing head_all_spike/head_simple_spike/head_complex_spike')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output dir for this comparison analysis')
    parser.add_argument('--manifest-dir', type=str, default=None,
                        help='Directory containing manifest.json. Defaults to --base-dir')
    parser.add_argument('--all-run', type=str, default='head_all_spike')
    parser.add_argument('--ss-run', type=str, default='head_simple_spike')
    parser.add_argument('--cs-run', type=str, default='head_complex_spike')
    parser.add_argument('--data-root', type=str, default=str(HERE / 'data'))
    parser.add_argument('--spatial-cache-root', type=str, default=None,
                        help='Root containing per-animal spatial_analysis_full.pkl files. '
                             'Defaults to --data-root')
    parser.add_argument('--figures-root', type=str, default=str(HERE / 'figures'))
    parser.add_argument('--direction-mode', type=str, default='head', choices=['head', 'travel'])
    parser.add_argument('--first-n-minutes', type=float, default=10.0)
    parser.add_argument('--split-map-bin-size-cm', type=float, default=2.5)
    parser.add_argument(
        '--occupancy-threshold-s',
        type=float,
        default=0.2,
        help='Occupancy threshold (seconds) for non-split egocentric maps and placecell maps.',
    )
    parser.add_argument(
        '--occupancy-threshold-split-s',
        type=float,
        default=0.1,
        help='Column 5/6 split-map occupancy threshold (seconds) for pref/non-pref bin validity.',
    )
    parser.add_argument(
        '--valid-bin-min-occupied-angle-bins',
        type=int,
        default=3,
        help='Spatial-bin validity criterion: minimum occupied angle bins (strictly greater than this value).',
    )
    parser.add_argument(
        '--valid-bin-min-mean-rate-hz',
        type=float,
        default=0.5,
        help='Spatial-bin validity criterion: minimum mean firing rate (Hz).',
    )
    parser.add_argument(
        '--valid-bin-min-spikes',
        type=int,
        default=2,
        help='Spatial-bin validity criterion: minimum spike count per spatial bin.',
    )
    parser.add_argument(
        '--tuning-min-valid-bins',
        type=int,
        default=5,
        help='Minimum valid spatial bins required for tuning/pass decisions.',
    )
    parser.add_argument(
        '--split-preferred-half-width-deg',
        type=float,
        default=50.0,
        help='Half-width (degrees) of preferred-angle window for pref/non-pref split.',
    )
    parser.add_argument(
        '--pf-overlay-area-threshold',
        type=float,
        default=0.25,
        help='PF coarse-bin interpolation threshold: overlap area ratio in [0,1].',
    )
    parser.add_argument(
        '--split-preferred-angle-source',
        type=str,
        default='empirical',
        choices=['fit', 'empirical'],
        help='Preferred/non-preferred split source for columns 5/6: fit(red) or empirical(blue).',
    )
    parser.add_argument(
        '--stats-pref-reference',
        type=str,
        default='all',
        choices=['all', 'matching_metric'],
        help='Shared preferred reference for column 5/6 SS/CS split maps and PF stats: '
             'all-spike (all) or per-metric (matching_metric).',
    )
    parser.add_argument(
        '--stats-joint-valid-bins',
        type=str,
        default='true',
        choices=['true', 'false'],
        help='When true, pref/non-pref stats use only bins valid in both maps.',
    )
    parser.add_argument(
        '--hd-vel-corr-method',
        type=str,
        default='normalized_dot',
        choices=['normalized_dot', 'mean_raw_dot'],
        help='Binwise HD(empirical) vs velocity correlation metric.',
    )
    parser.add_argument(
        '--col3-emp-arrow-length-scale',
        type=float,
        default=1.0,
        help='Scale factor for empirical (blue) arrow length in column 3 spatial maps.',
    )
    parser.add_argument(
        '--binpolar-render-style',
        type=str,
        default='fan',
        choices=['fan', 'line', 'curve'],
        help='Render style for right-side mini-polar panels: fan or line (curve alias for line).',
    )
    parser.add_argument(
        '--any-pass-threshold',
        type=int,
        default=99,
        choices=[95, 99, 100],
        help='Select cells passing this threshold in any run.',
    )
    parser.add_argument('--save-formats', nargs='+', default=['svg'])
    args = parser.parse_args(argv)

    import pandas as pd

    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir)
    manifest_dir = Path(args.manifest_dir) if args.manifest_dir else base_dir
    any_pass_threshold = int(args.any_pass_threshold)
    tuning_min_valid_bins = int(max(1, int(args.tuning_min_valid_bins)))
    stats_pref_reference = _sanitize_stats_pref_reference(args.stats_pref_reference)
    occupancy_threshold_s = float(max(0.0, float(args.occupancy_threshold_s)))
    occupancy_threshold_split_s = float(max(0.0, float(args.occupancy_threshold_split_s)))
    valid_bin_min_spikes = int(max(0, int(args.valid_bin_min_spikes)))
    col3_emp_arrow_length_scale = float(args.col3_emp_arrow_length_scale)
    if (not np.isfinite(col3_emp_arrow_length_scale)) or col3_emp_arrow_length_scale <= 0:
        col3_emp_arrow_length_scale = 1.0
    binpolar_render_style = str(args.binpolar_render_style).strip().lower()
    if binpolar_render_style == 'curve':
        binpolar_render_style = 'line'
    if binpolar_render_style not in {'fan', 'line'}:
        binpolar_render_style = 'fan'
    any_pass_key = f'pass_{any_pass_threshold}'
    any_pass_suffix = f'any{any_pass_threshold}'

    run_all = base_dir / args.all_run
    run_ss = base_dir / args.ss_run
    run_cs = base_dir / args.cs_run
    for p in [run_all, run_ss, run_cs]:
        if not p.exists():
            raise FileNotFoundError(f'Missing run folder: {p}')

    config = build_config(args.data_root, args.figures_root)
    spatial_cache_root = Path(args.spatial_cache_root) if args.spatial_cache_root else Path(args.data_root)

    manifest_path = manifest_dir / 'manifest.json'
    with open(manifest_path) as f:
        manifest = json.load(f)
    print(f'Loaded manifest: {len(manifest)} cells')

    lookup_all = load_npz_summary_lookup(run_all / 'per_cell_results', manifest)
    lookup_ss = load_npz_summary_lookup(run_ss / 'per_cell_results', manifest)
    lookup_cs = load_npz_summary_lookup(run_cs / 'per_cell_results', manifest)

    params = EgocentricSummaryPlotParams(
        categories=tuple(sorted(set(m['category'] for m in manifest))),
        first_n_minutes=args.first_n_minutes,
        direction_mode=args.direction_mode,
        time_bin_s=0.1,
        arena_size_cm=(35.5, 20.0),
        speed_min_cm_s=3.0,
        speed_max_cm_s=60.0,
        local_spatial_bin_cm=5.0,
        n_angle_bins=10,
        occupancy_threshold_s=float(occupancy_threshold_s),
        occupancy_threshold_split_s=float(occupancy_threshold_split_s),
        min_occupied_angle_bins=int(args.valid_bin_min_occupied_angle_bins),
        min_mean_rate_hz=float(args.valid_bin_min_mean_rate_hz),
        min_spikes_per_bin=int(valid_bin_min_spikes),
        only_plot_spikes_in_valid_spatial_bins=False,
        show_empirical_fit_curve=True,
        show_spatial_map_with_fitted_arrows=True,
        curve_polar=False,
        split_maps_placecell_style=True,
        split_preferred_angle_source=str(args.split_preferred_angle_source),
        split_preferred_reference_mode=str(stats_pref_reference),
        split_preferred_half_width_deg=float(args.split_preferred_half_width_deg),
        pf_overlay_area_threshold=float(args.pf_overlay_area_threshold),
        split_map_bin_size_cm=float(args.split_map_bin_size_cm),
        pc_bin_size_cm=1.5,
        pc_smooth_sigma=2.5,
        pc_occ_smooth_sigma=2.5,
        pc_min_occupancy_s=float(occupancy_threshold_s),
        pc_use_smoothed_occ_mask=False,
        pc_kernel_size=51,
        pc_filter_type='boxcar',
        pc_speed_threshold_cm_s=3.0,
        pc_min_duration_s=0.25,
        pc_merge_gap_s=0.0,
        travel_smooth_window=5,
        travel_min_step=0.0,
        theta_freqs=(4.0, 8.0),
        slow_freqs=2.0,
        theta_slow_speed_threshold=3.0,
        theta_slow_kernel_size=51,
        theta_slow_min_duration_s=0.25,
        theta_slow_merge_gap_s=0.0,
        hd_vel_corr_method=str(args.hd_vel_corr_method),
        col3_emp_arrow_length_scale=float(col3_emp_arrow_length_scale),
        binpolar_render_style=str(binpolar_render_style),
        show_shuffle_null_column=True,
        shuffle_null_n=1000,
        save_formats=tuple(args.save_formats),
        clear_output=False,
    )

    per_cell_root = output_dir / 'per_cell_summary'
    if per_cell_root.exists():
        shutil.rmtree(per_cell_root)
        print(f'Cleared previous output folder: {per_cell_root}')
    per_cell_root.mkdir(parents=True, exist_ok=True)

    print('\n--- Loading spatial data ---')
    spatial_data = classify_spatial_cells(
        data_folder=str(spatial_cache_root),
        folders=config.animals,
        cb_num_threshold=config.pooled.cb_num_threshold,
        cs_peak_rate_threshold=config.pooled.cs_peak_rate_threshold,
        snr_threshold=config.analysis.snr_threshold,
    )
    print(f'Using spatial cache root: {spatial_cache_root}')

    cells_by_animal = defaultdict(list)
    for cell_info in manifest:
        cells_by_animal[cell_info['animal_id']].append(cell_info)

    manifest_rows = []
    skip_rows = []
    pf_split_rows = []
    counts = {
        'attempted': 0,
        'selected_any_threshold': 0,
        'plotted': 0,
        'plotted_pass': 0,
        'plotted_fail': 0,
        'skipped': 0,
        'filtered_false_positive': 0,
        'filtered_low_real_mrl_all': 0,
    }

    for animal_id, cells in cells_by_animal.items():
        print(f'\n=== {animal_id} ({len(cells)} manifest cells) ===')
        animal_dir = config.data_root / animal_id
        spatial_cache_animal_dir = spatial_cache_root / animal_id

        try:
            merged = _load_merged_data(animal_dir)
            ctx = _prepare_native_analysis_context(merged, config)
            spatial_by_idx = _load_spatial_analysis_by_idx(spatial_cache_animal_dir)
        except Exception as exc:
            print(f'  [ERROR] Failed to load data: {exc}')
            for cell_info in cells:
                skip_rows.append({
                    'category': cell_info['category'],
                    'animal_id': animal_id,
                    'cell_idx': cell_info['cell_idx'],
                    'reason': f'data_load_failed({exc})',
                })
            counts['skipped'] += len(cells)
            continue

        for cell_info in cells:
            category = str(cell_info['category'])
            cell_idx = int(cell_info['cell_idx'])
            counts['attempted'] += 1

            key = (category, str(animal_id), int(cell_idx))
            row_all = lookup_all.get(key)
            row_ss = lookup_ss.get(key)
            row_cs = lookup_cs.get(key)
            row_all = _filter_egocentric_summary_row_by_valid_bins(row_all, min_valid_bins=tuning_min_valid_bins)
            row_ss = _filter_egocentric_summary_row_by_valid_bins(row_ss, min_valid_bins=tuning_min_valid_bins)
            row_cs = _filter_egocentric_summary_row_by_valid_bins(row_cs, min_valid_bins=tuning_min_valid_bins)

            analysis = spatial_by_idx.get(cell_idx)
            if not isinstance(analysis, dict):
                skip_rows.append({
                    'category': category, 'animal_id': animal_id,
                    'cell_idx': cell_idx, 'reason': 'missing_spatial_analysis',
                })
                counts['skipped'] += 1
                continue
            # Force split-map smoothing parameters from this run config, rather than
            # inheriting precomputed analysis defaults from spatial_analysis_full.pkl.
            analysis_for_plot = dict(analysis)
            analysis_params_for_plot = (
                dict(analysis.get('params', {}))
                if isinstance(analysis.get('params', {}), dict)
                else {}
            )
            analysis_params_for_plot['smooth_sigma'] = float(params.pc_smooth_sigma)
            analysis_params_for_plot['occ_smooth_sigma'] = float(params.pc_occ_smooth_sigma)
            analysis_for_plot['params'] = analysis_params_for_plot

            pass_gate, gate_reason = _passes_egocentric_category_gate(
                category=category, analysis=analysis,
            )
            if not pass_gate:
                skip_rows.append({
                    'category': category, 'animal_id': animal_id,
                    'cell_idx': cell_idx, 'reason': gate_reason or 'category_gate_failed',
                })
                counts['skipped'] += 1
                continue

            if cell_idx >= int(ctx.get('n_cells', 0)):
                skip_rows.append({
                    'category': category, 'animal_id': animal_id,
                    'cell_idx': cell_idx, 'reason': 'cell_idx_out_of_range',
                })
                counts['skipped'] += 1
                continue

            eligible = np.asarray(ctx.get('eligible_cells', np.array([])), dtype=bool)
            if cell_idx >= eligible.size or not bool(eligible[cell_idx]):
                skip_rows.append({
                    'category': category, 'animal_id': animal_id,
                    'cell_idx': cell_idx, 'reason': 'cell_not_eligible',
                })
                counts['skipped'] += 1
                continue

            # Primary row drives row-1/all-spike title text and default fit fallback.
            row_primary = None
            if isinstance(row_all, dict):
                row_primary = dict(row_all)
            elif isinstance(row_ss, dict):
                row_primary = dict(row_ss)
            elif isinstance(row_cs, dict):
                row_primary = dict(row_cs)
            else:
                row_primary = {
                    'category': category,
                    'animal_id': animal_id,
                    'cell_idx': int(cell_idx),
                }

            bad_mask = np.asarray(ctx['bad_masks'][cell_idx], dtype=bool)

            try:
                plot_data = _extract_egocentric_plot_timeseries(
                    ctx=ctx,
                    cell_idx=int(cell_idx),
                    bad_mask=bad_mask,
                    analysis=analysis_for_plot,
                    params=params,
                )
                row_all = _filter_egocentric_summary_row_for_tuning_decision(
                    row_all,
                    local_tuning=plot_data.get('local_tuning_all'),
                    min_valid_bins=tuning_min_valid_bins,
                )
                row_ss = _filter_egocentric_summary_row_for_tuning_decision(
                    row_ss,
                    local_tuning=plot_data.get('local_tuning_ss'),
                    min_valid_bins=tuning_min_valid_bins,
                )
                row_cs = _filter_egocentric_summary_row_for_tuning_decision(
                    row_cs,
                    local_tuning=plot_data.get('local_tuning_cs'),
                    min_valid_bins=tuning_min_valid_bins,
                )
                pass95_all = _coerce_bool((row_all or {}).get('pass_95', False))
                pass95_ss = _coerce_bool((row_ss or {}).get('pass_95', False))
                pass95_cs = _coerce_bool((row_cs or {}).get('pass_95', False))
                pass99_all = _coerce_bool((row_all or {}).get('pass_99', False))
                pass99_ss = _coerce_bool((row_ss or {}).get('pass_99', False))
                pass99_cs = _coerce_bool((row_cs or {}).get('pass_99', False))
                pass100_all = _coerce_bool((row_all or {}).get('pass_100', False))
                pass100_ss = _coerce_bool((row_ss or {}).get('pass_100', False))
                pass100_cs = _coerce_bool((row_cs or {}).get('pass_100', False))

                pass_by_thr_all = {95: pass95_all, 99: pass99_all, 100: pass100_all}
                pass_by_thr_ss = {95: pass95_ss, 99: pass99_ss, 100: pass100_ss}
                pass_by_thr_cs = {95: pass95_cs, 99: pass99_cs, 100: pass100_cs}
                pass_thr_all = bool(pass_by_thr_all.get(any_pass_threshold, False))
                pass_thr_ss = bool(pass_by_thr_ss.get(any_pass_threshold, False))
                pass_thr_cs = bool(pass_by_thr_cs.get(any_pass_threshold, False))
                pass_thr_any = bool(pass_thr_all or pass_thr_ss or pass_thr_cs)
                pass95_any = bool(pass95_all or pass95_ss or pass95_cs)
                pass99_any = bool(pass99_all or pass99_ss or pass99_cs)
                pass100_any = bool(pass100_all or pass100_ss or pass100_cs)
                real_mrl_all = np.nan
                if isinstance(row_all, dict):
                    try:
                        real_mrl_all = float(row_all.get('real_mrl', np.nan))
                    except Exception:
                        real_mrl_all = np.nan
                pf_row_start = int(len(pf_split_rows))
                pf_row_end = int(pf_row_start)
                try:
                    _cell_pf_rows = _compute_pf_split_stats_rows(
                        analysis=analysis,
                        plot_data=plot_data,
                        row_all=row_all,
                        row_ss=row_ss,
                        row_cs=row_cs,
                        params=params,
                        split_source=str(args.split_preferred_angle_source),
                        stats_pref_reference=str(stats_pref_reference),
                        stats_joint_valid_bins=_coerce_bool(str(args.stats_joint_valid_bins)),
                        category=category,
                        animal_id=animal_id,
                        cell_idx=cell_idx,
                        any_pass_threshold=int(any_pass_threshold),
                        pass_100_any=bool(pass100_any),
                        hd_vel_corr_method=str(args.hd_vel_corr_method),
                    )
                    for _r in _cell_pf_rows:
                        _r['selected_in_pass_any_folder'] = False
                    pf_split_rows.extend(_cell_pf_rows)
                    pf_row_end = int(len(pf_split_rows))
                except Exception as exc_stats:
                    print(
                        f'  [WARN] PF split stats failed for {animal_id} cell {cell_idx + 1}: '
                        f'{type(exc_stats).__name__}: {exc_stats}'
                    )
                plot_group = f'pass_{any_pass_suffix}' if bool(pass_thr_any) else f'fail_{any_pass_suffix}'
                if not bool(pass_thr_any):
                    # Only keep non-passing CS+/CS- place cells in fail-group output.
                    if str(category) not in {'CSplus', 'CSminus'}:
                        continue
                    if not bool((analysis or {}).get('is_place_cell', False)):
                        continue
                else:
                    counts['selected_any_threshold'] += 1
                if bool(pass_thr_any) and np.isfinite(real_mrl_all) and (real_mrl_all < 0.2):
                    skip_rows.append({
                        'category': category,
                        'animal_id': animal_id,
                        'cell_idx': cell_idx,
                        'reason': f'low_real_mrl_all_lt0p2(real_mrl_all={real_mrl_all:.3f})',
                    })
                    counts['filtered_low_real_mrl_all'] += 1
                    counts['skipped'] += 1
                    print(
                        f'  [SKIP] {category} / cell {cell_idx + 1}: '
                        f'low real_mrl_all ({real_mrl_all:.3f} < 0.200)'
                    )
                    continue
                row_primary['pass_95'] = bool(pass95_any)
                row_primary['pass_99'] = bool(pass99_any)
                row_primary['pass_100'] = bool(pass100_any)
                if bool(pass_thr_any) and bool(getattr(params, 'show_shuffle_null_column', False)):
                    try:
                        seed_src = f"{animal_id}|{int(cell_idx)}|{int(any_pass_threshold)}|shuffle_null"
                        seed_bytes = hashlib.sha256(seed_src.encode("utf-8")).digest()
                        shuffle_seed = int.from_bytes(seed_bytes[:8], byteorder="little", signed=False) % (2**31 - 1)
                        if shuffle_seed <= 0:
                            shuffle_seed = 1
                        shuffle_payload = _compute_shuffle_null_payload_for_plot(
                            plot_data=plot_data,
                            row_all=row_all,
                            row_ss=row_ss,
                            row_cs=row_cs,
                            params=params,
                            split_source=str(args.split_preferred_angle_source),
                            stats_pref_reference=str(stats_pref_reference),
                            n_shuffles=int(max(1, int(getattr(params, 'shuffle_null_n', 1000)))),
                            rng_seed=int(shuffle_seed),
                        )
                        plot_data['shuffle_null_column_payload'] = dict(shuffle_payload)
                    except Exception as exc_shuffle:
                        plot_data['shuffle_null_column_payload'] = {
                            'ok': False,
                            'reason': f'shuffle_payload_failed({type(exc_shuffle).__name__}: {exc_shuffle})',
                            'rows': [],
                            'n_shuffles': int(max(1, int(getattr(params, 'shuffle_null_n', 1000)))),
                        }
                else:
                    plot_data.pop('shuffle_null_column_payload', None)

                cat_dir = per_cell_root / plot_group / category
                cat_dir.mkdir(parents=True, exist_ok=True)
                out_base = cat_dir / f'{animal_id}_cell{cell_idx + 1:03d}_egocentric_summary_{any_pass_suffix}_3spike'
                plot_meta = _plot_egocentric_per_cell_summary_figure(
                    category=category,
                    animal_id=str(animal_id),
                    cell_idx=int(cell_idx),
                    mode=str(args.direction_mode),
                    data=plot_data,
                    summary_row=row_primary,
                    summary_row_all=row_all,
                    summary_row_ss=row_ss,
                    summary_row_cs=row_cs,
                    params=params,
                    out_base=out_base,
                )
                plt.close('all')

                green_all = int(plot_meta.get('green_ring_count_all', 0))
                green_ss = int(plot_meta.get('green_ring_count_ss', 0))
                green_cs = int(plot_meta.get('green_ring_count_cs', 0))
                if bool(pass_thr_any) and max(green_all, green_ss, green_cs) < 3:
                    for saved in list(plot_meta.get('saved_paths', [])):
                        try:
                            Path(saved).unlink(missing_ok=True)
                        except Exception:
                            pass
                    skip_rows.append({
                        'category': category,
                        'animal_id': animal_id,
                        'cell_idx': cell_idx,
                        'reason': f'false_positive_green_rings_lt3(all={green_all},ss={green_ss},cs={green_cs})',
                    })
                    counts['filtered_false_positive'] += 1
                    counts['skipped'] += 1
                    print(
                        f'  [SKIP] {category} / cell {cell_idx + 1}: '
                        f'false_positive (green rings all/ss/cs={green_all}/{green_ss}/{green_cs})'
                    )
                    continue

                counts['plotted'] += 1
                if bool(pass_thr_any):
                    counts['plotted_pass'] += 1
                    if pf_row_end > pf_row_start:
                        for _ri in range(int(pf_row_start), int(pf_row_end)):
                            pf_split_rows[_ri]['selected_in_pass_any_folder'] = True
                else:
                    counts['plotted_fail'] += 1
                manifest_rows.append({
                    'category': category,
                    'animal_id': animal_id,
                    'cell_idx': cell_idx,
                    'cell_num': cell_idx + 1,
                    'any_pass_threshold': int(any_pass_threshold),
                    'plot_group': str(plot_group),
                    'selected_by_any_threshold': bool(pass_thr_any),
                    f'selected_by_any_pass{any_pass_threshold}': bool(pass_thr_any),
                    f'pass{any_pass_threshold}_all': bool(pass_thr_all),
                    f'pass{any_pass_threshold}_ss': bool(pass_thr_ss),
                    f'pass{any_pass_threshold}_cs': bool(pass_thr_cs),
                    'pass99_all': bool(pass99_all),
                    'pass99_ss': bool(pass99_ss),
                    'pass99_cs': bool(pass99_cs),
                    'pass100_all': bool(pass100_all),
                    'pass100_ss': bool(pass100_ss),
                    'pass100_cs': bool(pass100_cs),
                    'real_mrl_all': float(real_mrl_all) if np.isfinite(real_mrl_all) else np.nan,
                    'valid_bin_mean_mrl_all': plot_meta.get('valid_bin_mean_mrl_all', np.nan),
                    'valid_bin_mean_mrl_ss': plot_meta.get('valid_bin_mean_mrl_ss', np.nan),
                    'valid_bin_mean_mrl_cs': plot_meta.get('valid_bin_mean_mrl_cs', np.nan),
                    'valid_bin_mrl_n_all': plot_meta.get('valid_bin_mrl_n_all', np.nan),
                    'valid_bin_mrl_n_ss': plot_meta.get('valid_bin_mrl_n_ss', np.nan),
                    'valid_bin_mrl_n_cs': plot_meta.get('valid_bin_mrl_n_cs', np.nan),
                    'hd_vel_corr_method': str(plot_meta.get('hd_vel_corr_method', str(args.hd_vel_corr_method))),
                    'hd_vel_corr_all': plot_meta.get('hd_vel_corr_all', np.nan),
                    'hd_vel_corr_ss': plot_meta.get('hd_vel_corr_ss', np.nan),
                    'hd_vel_corr_cs': plot_meta.get('hd_vel_corr_cs', np.nan),
                    'hd_vel_corr_n_all': plot_meta.get('hd_vel_corr_n_all', np.nan),
                    'hd_vel_corr_n_ss': plot_meta.get('hd_vel_corr_n_ss', np.nan),
                    'hd_vel_corr_n_cs': plot_meta.get('hd_vel_corr_n_cs', np.nan),
                    'has_best_reference': bool(plot_meta.get('has_best_reference', False)),
                    'n_subplots': int(plot_meta.get('n_subplots', 0)),
                    'real_mrl_primary': plot_meta.get('real_mrl', np.nan),
                    'pass_95_any': str(plot_meta.get('pass_95', '')),
                    'pass_99_any': str(plot_meta.get('pass_99', '')),
                    'pass_100_any': str(plot_meta.get('pass_100', '')),
                    'green_rings_all': int(green_all),
                    'green_rings_ss': int(green_ss),
                    'green_rings_cs': int(green_cs),
                    'stats_pref_reference': str(stats_pref_reference),
                    'saved_paths': ';'.join(list(plot_meta.get('saved_paths', []))),
                })
                print(
                    f'  [OK] [{plot_group}] {category} / cell {cell_idx + 1} '
                    f'({any_pass_key}: all={pass_thr_all}, ss={pass_thr_ss}, cs={pass_thr_cs})'
                )

            except Exception as exc:
                skip_rows.append({
                    'category': category, 'animal_id': animal_id,
                    'cell_idx': cell_idx,
                    'reason': f'plot_failed({type(exc).__name__}: {exc})',
                })
                counts['skipped'] += 1
                print(f'  [SKIP] {category} / cell {cell_idx + 1}: {exc}')
                continue

    manifest_columns = [
        'category',
        'animal_id',
        'cell_idx',
        'cell_num',
        'any_pass_threshold',
        'plot_group',
        'selected_by_any_threshold',
        f'selected_by_any_pass{any_pass_threshold}',
        f'pass{any_pass_threshold}_all',
        f'pass{any_pass_threshold}_ss',
        f'pass{any_pass_threshold}_cs',
        'pass99_all',
        'pass99_ss',
        'pass99_cs',
        'pass100_all',
        'pass100_ss',
        'pass100_cs',
        'real_mrl_all',
        'valid_bin_mean_mrl_all',
        'valid_bin_mean_mrl_ss',
        'valid_bin_mean_mrl_cs',
        'valid_bin_mrl_n_all',
        'valid_bin_mrl_n_ss',
        'valid_bin_mrl_n_cs',
        'hd_vel_corr_method',
        'hd_vel_corr_all',
        'hd_vel_corr_ss',
        'hd_vel_corr_cs',
        'hd_vel_corr_n_all',
        'hd_vel_corr_n_ss',
        'hd_vel_corr_n_cs',
        'has_best_reference',
        'n_subplots',
        'real_mrl_primary',
        'pass_95_any',
        'pass_99_any',
        'pass_100_any',
        'green_rings_all',
        'green_rings_ss',
        'green_rings_cs',
        'stats_pref_reference',
        'saved_paths',
    ]
    skip_columns = ['category', 'animal_id', 'cell_idx', 'reason']
    manifest_df = pd.DataFrame(manifest_rows, columns=manifest_columns)
    skip_df = pd.DataFrame(skip_rows, columns=skip_columns)
    pf_split_columns = [
        'animal_id',
        'cell_idx',
        'cell_num',
        'category',
        'any_pass_threshold',
        'region',
        'metric',
        'preferred_mean',
        'nonpreferred_mean',
        'delta_pref_minus_nonpref',
        'n_pref_bins',
        'n_nonpref_bins',
        'n_region_bins',
        'is_place_cell',
        'pass_100_any',
        'selected_in_pass_any_folder',
        'split_source',
        'stats_pref_reference',
        'stats_joint_valid_bins',
        'split_bin_size_cm',
        'stats_smooth_sigma_cm',
        'stats_occ_smooth_sigma_cm',
        'pf_mask_mode',
        'hd_vel_corr_method',
    ]
    pf_split_df = pd.DataFrame(pf_split_rows, columns=pf_split_columns)
    manifest_csv = per_cell_root / f'egocentric_per_cell_plot_manifest_{any_pass_suffix}_3spike.csv'
    skip_csv = per_cell_root / f'egocentric_per_cell_plot_skipped_{any_pass_suffix}_3spike.csv'
    pf_split_csv = per_cell_root / f'egocentric_pf_split_stats_{any_pass_suffix}_3spike.csv'
    manifest_df.to_csv(manifest_csv, index=False)
    skip_df.to_csv(skip_csv, index=False)
    pf_split_df.to_csv(pf_split_csv, index=False)

    print(f'\n{"=" * 60}')
    print(f'Attempted manifest cells:          {counts["attempted"]}')
    print(f'Selected by any {any_pass_key} (3 runs): {counts["selected_any_threshold"]}')
    print(f'Filtered low all real_mrl<0.2:    {counts["filtered_low_real_mrl_all"]}')
    print(f'Filtered false positives:         {counts["filtered_false_positive"]}')
    print(f'Plotted:                          {counts["plotted"]}')
    print(f'  pass group plotted:             {counts["plotted_pass"]}')
    print(f'  fail group plotted:             {counts["plotted_fail"]}')
    print(f'Skipped after selection:          {counts["skipped"]}')
    print(f'Manifest:                         {manifest_csv}')
    print(f'Skipped:                          {skip_csv}')
    print(f'PF split stats (all categories):  {pf_split_csv} (rows={len(pf_split_df)})')


if __name__ == '__main__':
    main()
