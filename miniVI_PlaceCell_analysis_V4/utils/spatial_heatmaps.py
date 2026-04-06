"""Spatial heatmap plotting extracted from pooled notebook.

This module contains the core per-cell spatial heatmap renderer and category
aggregation logic, decoupled from notebook execution.
"""

from __future__ import annotations

import os
import glob
import pickle
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def get_deleted_cells(data_folder, folders, snr_threshold=3.5, bad_frac_threshold=0.9):
    """
    Identify cells that are truly deleted due to bad SNR.
    A cell is considered deleted if >90% of its frames have SNR < threshold.
    
    Returns: set of (session, cell_idx) tuples
    """
    deleted_set = set()
    for folder in folders:
        merged_path = os.path.join(data_folder, folder, 'merged_aligned_data_CS.pkl')
        if os.path.exists(merged_path):
            with open(merged_path, 'rb') as f:
                merged_data = pickle.load(f)
            
            SNR_interpolated = merged_data.get('SNR_interpolated', [])
            for cell_idx in range(len(SNR_interpolated)):
                snr_vals = np.asarray(SNR_interpolated[cell_idx])
                if snr_vals.ndim == 0:
                    deleted_set.add((folder, cell_idx))
                    continue
                
                bad_mask = snr_vals < snr_threshold
                bad_frac = np.mean(bad_mask)
                
                if bad_frac > bad_frac_threshold:
                    deleted_set.add((folder, cell_idx))
    return deleted_set


def get_deleted_cells_with_fallback(data_folder, folders, snr_threshold=3.5, bad_frac_threshold=0.9):
    """Same logic as get_deleted_cells, but supports merged_aligned_data.pkl fallback."""
    deleted_set = set()
    for folder in folders:
        merged_path = os.path.join(data_folder, folder, 'merged_aligned_data_CS.pkl')
        if not os.path.exists(merged_path):
            merged_path = os.path.join(data_folder, folder, 'merged_aligned_data.pkl')
        if os.path.exists(merged_path):
            with open(merged_path, 'rb') as f:
                merged_data = pickle.load(f)
            SNR_interpolated = merged_data.get('SNR_interpolated', [])
            for cell_idx in range(len(SNR_interpolated)):
                snr_vals = np.asarray(SNR_interpolated[cell_idx])
                if snr_vals.ndim == 0:
                    deleted_set.add((folder, cell_idx))
                    continue
                bad_mask = snr_vals < snr_threshold
                bad_frac = np.mean(bad_mask)
                if bad_frac > bad_frac_threshold:
                    deleted_set.add((folder, cell_idx))
    return deleted_set


def compute_removed_frame_stats_with_fallback(
    data_folder,
    folders,
    snr_threshold=3.5,
    min_good_minutes=5.0,
):
    """Compute per-cell removed-frame stats using time-varying SNR masks.

    Returns
    -------
    dict
        Mapping (folder, cell_idx) -> stats dict
    """
    from utils.placecell_pipeline import _compute_bad_masks

    out = {}
    for folder in folders:
        merged_path = os.path.join(data_folder, folder, "merged_aligned_data.pkl")
        if not os.path.exists(merged_path):
            merged_path = os.path.join(data_folder, folder, "merged_aligned_data_CS.pkl")
        if not os.path.exists(merged_path):
            continue

        with open(merged_path, "rb") as f:
            merged_data = pickle.load(f)

        try:
            bad_masks = _compute_bad_masks(
                merged_data,
                snr_threshold=float(snr_threshold),
                min_good_minutes=float(min_good_minutes),
            )
        except Exception:
            continue

        x = np.asarray(merged_data.get("x_neural", []), dtype=float)
        y = np.asarray(merged_data.get("y_neural", np.full_like(x, np.nan)), dtype=float)
        speed = np.asarray(merged_data.get("speed", np.full_like(x, np.nan)), dtype=float)
        n_frames = int(len(x))
        pos_nan_mask = (~np.isfinite(x)) | (~np.isfinite(y)) | (~np.isfinite(speed))

        if bad_masks.ndim != 2 or bad_masks.shape[1] != n_frames:
            continue

        for cell_idx in range(int(bad_masks.shape[0])):
            bad_mask = np.asarray(bad_masks[cell_idx], dtype=bool)
            n_removed_total = int(np.sum(bad_mask))
            n_removed_pos_nan = int(np.sum(bad_mask & pos_nan_mask))
            n_removed_snr_only = int(np.sum(bad_mask & (~pos_nan_mask)))
            n_kept_total = int(n_frames - n_removed_total)
            pct_removed_total = (100.0 * n_removed_total / n_frames) if n_frames > 0 else np.nan
            pct_removed_snr_only = (100.0 * n_removed_snr_only / n_frames) if n_frames > 0 else np.nan

            out[(folder, cell_idx)] = {
                "n_frames_total": n_frames,
                "n_frames_kept_total": n_kept_total,
                "n_removed_frames_total": n_removed_total,
                "n_removed_frames_snr_only": n_removed_snr_only,
                "n_removed_frames_pos_nan": n_removed_pos_nan,
                "pct_removed_frames_total": pct_removed_total,
                "pct_removed_frames_snr_only": pct_removed_snr_only,
            }
    return out


def load_pooled_spatial_data(data_folder, folders):
    """Load spatial_analysis_full.pkl from specified animal folders.
    Returns list of (folder_name, cell_data) tuples for filtering.
    """
    all_cells = []
    data_folder_abs = os.path.abspath(str(data_folder))
    for folder in folders:
        spatial_path = os.path.join(data_folder, folder, 'spatial_analysis_full.pkl')
        if os.path.exists(spatial_path):
            with open(spatial_path, 'rb') as f:
                cells = pickle.load(f)
                for cell in cells:
                    cell['session'] = folder  # Add session info for filtering
                    cell['data_folder'] = data_folder_abs
                all_cells.extend(cells)
                print(f"Loaded {len(cells)} cells from {folder}")
        else:
            print(f"Missing: {spatial_path}")
    print(f"\nTotal: {len(all_cells)} cells loaded")
    return all_cells


def compute_cb_in_pf_counts(data_folder, folders):
    """Count complex bursts in PF during locomotion from spatial_analysis_full payload."""
    cb_in_pf_counts = {}
    for folder in folders:
        spatial_path = os.path.join(data_folder, folder, 'spatial_analysis_full.pkl')
        if not os.path.exists(spatial_path):
            continue
        with open(spatial_path, 'rb') as f:
            spatial_cells = pickle.load(f)
        for cell in spatial_cells:
            cell_idx = cell['cell_idx']
            spike_shapes = cell.get('spike_shapes')
            if spike_shapes and 'complex' in spike_shapes:
                cb = spike_shapes['complex']['shapes']
                n_in = len(cb.get('run_in', []))
            else:
                n_in = 0
            cb_in_pf_counts[(folder, cell_idx)] = n_in
    return cb_in_pf_counts


@dataclass
class SpatialCategoryData:
    valid_spatial_cells: list[dict[str, Any]]
    plcs_csplus: list[dict[str, Any]]
    plcs_csminus: list[dict[str, Any]]
    non_plcs: list[dict[str, Any]]
    deleted_cells: set[tuple[str, int]]
    cb_in_pf_counts: dict[tuple[str, int], int]


def is_csplus_place_cell(
    *,
    is_place_cell: bool,
    n_cb_in_pf: int,
    cs_peak_rate: float,
    cb_num_threshold: int = 10,
    cs_peak_rate_threshold: float = 0.5,
) -> bool:
    return (
        bool(is_place_cell)
        and int(n_cb_in_pf) >= int(cb_num_threshold)
        and np.isfinite(float(cs_peak_rate))
        and float(cs_peak_rate) > float(cs_peak_rate_threshold)
    )


def classify_spatial_cells(
    data_folder,
    folders,
    cb_num_threshold=10,
    cs_peak_rate_threshold=0.5,
    snr_threshold=3.5,
    bad_frac_threshold=0.9,
):
    deleted_cells = get_deleted_cells_with_fallback(
        data_folder, folders, snr_threshold=snr_threshold, bad_frac_threshold=bad_frac_threshold
    )
    cb_in_pf_counts = compute_cb_in_pf_counts(data_folder, folders)
    removed_stats = compute_removed_frame_stats_with_fallback(
        data_folder,
        folders,
        snr_threshold=snr_threshold,
        min_good_minutes=5.0,
    )

    all_spatial_cells = load_pooled_spatial_data(data_folder, folders)
    valid_spatial_cells = [
        cell for cell in all_spatial_cells if (cell['session'], cell['cell_idx']) not in deleted_cells
    ]

    for cell in valid_spatial_cells:
        n_cb = cb_in_pf_counts.get((cell['session'], cell['cell_idx']), 0)
        cs_peak_rate = cell.get('cs_peak_rate', np.nan)
        cell['n_cb_in_pf'] = n_cb
        cell['is_cs_plc'] = is_csplus_place_cell(
            is_place_cell=bool(cell.get('is_place_cell', False)),
            n_cb_in_pf=int(n_cb),
            cs_peak_rate=float(cs_peak_rate),
            cb_num_threshold=int(cb_num_threshold),
            cs_peak_rate_threshold=float(cs_peak_rate_threshold),
        )
        rm_stats = removed_stats.get((cell['session'], cell['cell_idx']))
        if isinstance(rm_stats, dict):
            cell.update(rm_stats)

    plcs_csplus, plcs_csminus, non_plcs = [], [], []
    for cell in valid_spatial_cells:
        if cell['is_place_cell']:
            if cell['is_cs_plc']:
                plcs_csplus.append(cell)
            else:
                plcs_csminus.append(cell)
        else:
            non_plcs.append(cell)

    return SpatialCategoryData(
        valid_spatial_cells=valid_spatial_cells,
        plcs_csplus=plcs_csplus,
        plcs_csminus=plcs_csminus,
        non_plcs=non_plcs,
        deleted_cells=deleted_cells,
        cb_in_pf_counts=cb_in_pf_counts,
    )


def compute_global_theta_slow_vlims(cells):
    theta_mins, theta_maxs = [], []
    slow_abs_maxs = []
    for cell in cells:
        theta_map = cell.get('theta_map', None)
        if theta_map is not None and np.any(np.isfinite(theta_map)):
            theta_mins.append(np.nanmin(theta_map))
            theta_maxs.append(np.nanmax(theta_map))

        slow_map = cell.get('slow_map', None)
        if slow_map is not None and np.any(np.isfinite(slow_map)):
            slow_abs_maxs.append(np.nanmax(np.abs(slow_map)))

    theta_vlim = (min(theta_mins), max(theta_maxs)) if (theta_mins and theta_maxs) else None
    if slow_abs_maxs:
        m = max(slow_abs_maxs)
        slow_vlim = (-m, m)
    else:
        slow_vlim = None
    return theta_vlim, slow_vlim


def plot_celltype_distribution_pie(
    plcs_csplus,
    plcs_csminus,
    non_plcs,
    save_path=None,
    fig_size=(1, 1.8),
):
    """Plot CS+ PLC / CS- PLC / Non-PLC distribution pie chart."""
    fig, ax = plt.subplots(figsize=fig_size)
    sizes = [len(plcs_csplus), len(plcs_csminus), len(non_plcs)]
    labels = ['CS+ PLCs', 'CS- PLCs', 'Non-PLCs']
    # Match category background hues used in pooled plots, with stronger opacity in pie.
    colors = ['#FFF3E0', '#E3F2FD', '#888888']  # CS+ bg tint, CS- bg tint, gray
    explode = (0.02, 0.02, 0.02)

    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        colors=colors,
        explode=explode,
        autopct=lambda pct: f'{int(round(pct/100.*sum(sizes)))}\n({pct:.1f}%)',
        startangle=0,
        textprops={'fontsize': 6, 'fontname': 'Arial'},
    )

    for autotext in autotexts:
        autotext.set_fontsize(5)
        autotext.set_fontname('Arial')

    # Use higher alpha than background spans so CS+/CS- categories remain clear.
    if len(wedges) >= 2:
        wedges[0].set_alpha(1.0)  # CS+ PLC
        wedges[1].set_alpha(1.0)  # CS- PLC
    if len(wedges) >= 3:
        wedges[2].set_alpha(0.6)  # Non-PLC

    ax.axis('equal')
    if save_path:
        fig.savefig(save_path, dpi=300)
    return fig, ax


def _extract_cell_event_entry(entries: Any, cell_idx: int) -> dict[str, Any] | None:
    if isinstance(entries, (list, tuple)):
        idx = int(cell_idx)
        if 0 <= idx < len(entries) and isinstance(entries[idx], dict):
            return entries[idx]
    elif isinstance(entries, dict):
        return entries
    return None


def _sanitize_and_merge_intervals(
    starts: np.ndarray,
    ends: np.ndarray,
    n_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    starts = np.asarray(starts, dtype=int).ravel()
    ends = np.asarray(ends, dtype=int).ravel()
    n = int(min(starts.size, ends.size))
    if n <= 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    starts = starts[:n]
    ends = ends[:n]
    swap = ends < starts
    if np.any(swap):
        temp = starts.copy()
        starts[swap] = ends[swap]
        ends[swap] = temp[swap]

    if int(n_frames) > 0:
        starts = np.clip(starts, 0, int(n_frames) - 1)
        ends = np.clip(ends, 0, int(n_frames) - 1)
    valid = ends >= starts
    starts = starts[valid]
    ends = ends[valid]
    if starts.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    order = np.argsort(starts, kind="mergesort")
    starts = starts[order]
    ends = ends[order]

    merged_starts = [int(starts[0])]
    merged_ends = [int(ends[0])]
    for s, e in zip(starts[1:], ends[1:]):
        s_i = int(s)
        e_i = int(e)
        if s_i <= merged_ends[-1] + 1:
            merged_ends[-1] = max(merged_ends[-1], e_i)
        else:
            merged_starts.append(s_i)
            merged_ends.append(e_i)
    return np.asarray(merged_starts, dtype=int), np.asarray(merged_ends, dtype=int)


def _build_plateau_intervals_from_merged(
    merged_data: dict[str, Any] | None,
    *,
    cell_idx: int,
    include_long_cb_as_plateau: bool,
    cb_min_duration_ms: float,
    n_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(merged_data, dict):
        return np.array([], dtype=int), np.array([], dtype=int)

    p_entry = _extract_cell_event_entry(merged_data.get("plateaus_dicts"), int(cell_idx))
    p_starts = np.asarray(p_entry.get("starts", []), dtype=int) if p_entry is not None else np.array([], dtype=int)
    p_ends = np.asarray(p_entry.get("ends", []), dtype=int) if p_entry is not None else np.array([], dtype=int)

    cb_starts = np.array([], dtype=int)
    cb_ends = np.array([], dtype=int)
    if bool(include_long_cb_as_plateau):
        thr = float(cb_min_duration_ms)
        if (not np.isfinite(thr)) or thr <= 0:
            raise ValueError("plateau_cb_min_duration_ms must be a finite number > 0.")
        cb_entry = _extract_cell_event_entry(merged_data.get("complex_bursts_dicts"), int(cell_idx))
        if cb_entry is not None:
            starts_all = np.asarray(cb_entry.get("starts", []), dtype=int).ravel()
            ends_all = np.asarray(cb_entry.get("ends", []), dtype=int).ravel()
            n_cb = int(min(starts_all.size, ends_all.size))
            if n_cb > 0:
                starts_all = starts_all[:n_cb]
                ends_all = ends_all[:n_cb]
                durs = np.asarray(cb_entry.get("durations_ms", []), dtype=float).ravel()
                if durs.size == n_cb:
                    durs_ms = durs
                else:
                    frame_rate = float(merged_data.get("frame_rate", np.nan))
                    if np.isfinite(frame_rate) and frame_rate > 0:
                        durs_ms = (ends_all - starts_all + 1).astype(float) / frame_rate * 1000.0
                    else:
                        durs_ms = np.full(n_cb, np.nan, dtype=float)
                keep = np.isfinite(durs_ms) & (durs_ms >= thr)
                cb_starts = starts_all[keep]
                cb_ends = ends_all[keep]

    starts = np.concatenate([p_starts, cb_starts]) if (p_starts.size or cb_starts.size) else np.array([], dtype=int)
    ends = np.concatenate([p_ends, cb_ends]) if (p_ends.size or cb_ends.size) else np.array([], dtype=int)
    return _sanitize_and_merge_intervals(starts, ends, int(n_frames))


def _infer_cell_map_shape(cell: dict[str, Any]) -> tuple[int, int]:
    for key in (
        "rate_map",
        "ss_norm_map",
        "cs_norm_map",
        "theta_map",
        "slow_map",
        "place_field_mask",
    ):
        arr = cell.get(key, None)
        if arr is None:
            continue
        arr_np = np.asarray(arr)
        if arr_np.ndim == 2 and arr_np.shape[0] > 0 and arr_np.shape[1] > 0:
            return int(arr_np.shape[0]), int(arr_np.shape[1])

    params = cell.get("params", {})
    width_real = float(params.get("width_real", 35.5))
    height_real = float(params.get("height_real", 20.0))
    bin_size = float(params.get("bin_size", 1.5))
    if (not np.isfinite(bin_size)) or bin_size <= 0:
        bin_size = 1.5
    n_x = max(1, int(np.round(width_real / bin_size)))
    n_y = max(1, int(np.round(height_real / bin_size)))
    return n_x, n_y


def _resolve_merged_data_for_cell(
    cell: dict[str, Any],
    *,
    plateau_data_folder: str | None,
    merged_cache: dict[str, dict[str, Any] | None],
    warned_sessions: set[str],
) -> dict[str, Any] | None:
    session = str(cell.get("session", "")).strip()
    if len(session) == 0:
        return None

    data_root = plateau_data_folder if plateau_data_folder is not None else cell.get("data_folder", None)
    if not isinstance(data_root, str) or len(data_root.strip()) == 0:
        if session not in warned_sessions:
            print(f"Plateau map warning: missing data folder for session '{session}'.")
            warned_sessions.add(session)
        return None

    key = f"{os.path.abspath(data_root)}::{session}"
    if key in merged_cache:
        return merged_cache[key]

    merged_path = os.path.join(data_root, session, "merged_aligned_data.pkl")
    if not os.path.exists(merged_path):
        merged_path = os.path.join(data_root, session, "merged_aligned_data_CS.pkl")
    if not os.path.exists(merged_path):
        if session not in warned_sessions:
            print(f"Plateau map warning: missing merged data for session '{session}'.")
            warned_sessions.add(session)
        merged_cache[key] = None
        return None

    try:
        with open(merged_path, "rb") as f:
            merged = pickle.load(f)
    except Exception:
        if session not in warned_sessions:
            print(f"Plateau map warning: failed loading merged data for session '{session}'.")
            warned_sessions.add(session)
        merged_cache[key] = None
        return None

    merged_cache[key] = merged if isinstance(merged, dict) else None
    return merged_cache[key]


def _compute_plateau_occurrence_maps_for_cell(
    cell: dict[str, Any],
    *,
    merged_data: dict[str, Any] | None,
    include_long_cb_as_plateau: bool,
    cb_min_duration_ms: float,
    speed_threshold: float,
) -> dict[str, np.ndarray]:
    n_x, n_y = _infer_cell_map_shape(cell)
    out_all = np.zeros((n_x, n_y), dtype=float)
    out_moving = np.zeros((n_x, n_y), dtype=float)
    out_resting = np.zeros((n_x, n_y), dtype=float)

    if not isinstance(merged_data, dict):
        return {"all": out_all, "moving": out_moving, "resting": out_resting}

    x = np.asarray(merged_data.get("x_neural", []), dtype=float).reshape(-1)
    y = np.asarray(merged_data.get("y_neural", []), dtype=float).reshape(-1)
    speed = np.asarray(merged_data.get("speed", np.full_like(x, np.nan)), dtype=float).reshape(-1)
    n_frames = int(min(x.size, y.size, speed.size))
    if n_frames <= 0:
        return {"all": out_all, "moving": out_moving, "resting": out_resting}

    x = x[:n_frames]
    y = y[:n_frames]
    speed = speed[:n_frames]

    starts, ends = _build_plateau_intervals_from_merged(
        merged_data,
        cell_idx=int(cell.get("cell_idx", -1)),
        include_long_cb_as_plateau=bool(include_long_cb_as_plateau),
        cb_min_duration_ms=float(cb_min_duration_ms),
        n_frames=n_frames,
    )
    if starts.size == 0:
        return {"all": out_all, "moving": out_moving, "resting": out_resting}

    params = cell.get("params", {})
    width_real = float(params.get("width_real", 35.5))
    height_real = float(params.get("height_real", 20.0))
    if (not np.isfinite(width_real)) or width_real <= 0:
        width_real = 35.5
    if (not np.isfinite(height_real)) or height_real <= 0:
        height_real = 20.0
    x_edges = np.linspace(0.0, width_real, n_x + 1, dtype=float)
    y_edges = np.linspace(0.0, height_real, n_y + 1, dtype=float)

    def _accumulate_unique(mask: np.ndarray, xi: np.ndarray, yi: np.ndarray, target: np.ndarray) -> None:
        if not np.any(mask):
            return
        keys = np.unique(xi[mask] * n_y + yi[mask])
        target.reshape(-1)[keys] += 1.0

    thr = float(speed_threshold)
    for s, e in zip(starts, ends):
        s_i = int(s)
        e_i = int(e)
        if e_i < s_i:
            continue
        seg_slice = slice(s_i, e_i + 1)
        x_seg = x[seg_slice]
        y_seg = y[seg_slice]
        sp_seg = speed[seg_slice]

        xi = np.searchsorted(x_edges, x_seg, side="right") - 1
        yi = np.searchsorted(y_edges, y_seg, side="right") - 1
        x_right_edge = np.isfinite(x_seg) & np.isclose(x_seg, x_edges[-1], atol=1e-9, rtol=0.0)
        y_right_edge = np.isfinite(y_seg) & np.isclose(y_seg, y_edges[-1], atol=1e-9, rtol=0.0)
        xi[x_right_edge] = int(n_x - 1)
        yi[y_right_edge] = int(n_y - 1)

        valid_xy = np.isfinite(x_seg) & np.isfinite(y_seg)
        in_bounds = (xi >= 0) & (yi >= 0) & (xi < n_x) & (yi < n_y)
        valid = valid_xy & in_bounds
        if not np.any(valid):
            continue

        moving = valid & np.isfinite(sp_seg) & (sp_seg >= thr)
        resting = valid & np.isfinite(sp_seg) & (sp_seg < thr)
        _accumulate_unique(valid, xi, yi, out_all)
        _accumulate_unique(moving, xi, yi, out_moving)
        _accumulate_unique(resting, xi, yi, out_resting)

    return {"all": out_all, "moving": out_moving, "resting": out_resting}


def plot_selected_cells_figure(
    cell_groups,
    group_names=None,
    subplot_width=0.25,
    gap_ratio=0.1,
    theta_vlim=None,
    slow_vlim=None,
    save_path=None,
    show_scale_bar=True,
    pf_only_place_cells=True,
    plot_putative_PF=False,
    show_place_cell_star=True,
    show_significance_marker=True,
    show_theta_slow_corr_text=False,
    plot_spike_shapes=True,
    plot_spike_shapes_overall=True,
    min_shapes_per_condition=None,
    min_shapes_per_condition_ss=5,
    min_shapes_per_condition_cb=3,
    spike_shape_state='run',
    show_shape_counts=False,
    shape_ylim=None,
    print_removed_frames=True,
    include_plateau=True,
    plateau_state_mode='split',
    plateau_include_long_cb_as_plateau=True,
    plateau_cb_min_duration_ms=200.0,
    plateau_speed_threshold=3.0,
    plateau_data_folder=None,
):
    """
    Plot spatial heatmaps for selected cells in a grid format.
    
    Parameters:
    -----------
    cell_groups : list of lists
        Either a list of cell groups [[group1_cells], [group2_cells], ...] for multi-group figure,
        or a single list of cells [cell1, cell2, ...] which will be treated as one group.
    group_names : list of str, optional
        Names for each group (for display/saving). If None, uses generic names.
    subplot_width : float
        Width of each map subplot in inches (default 0.25)
    gap_ratio : float
        Width of gap columns relative to cell columns (only used for multi-group)
    theta_vlim : list/tuple or None
        [vmin, vmax] for theta map. If None, uses auto limits.
    slow_vlim : list/tuple or None
        [vmin, vmax] for slow Vm map. If None, uses auto limits.
    save_path : str, optional
        Path to save the figure. If None, doesn't save.
    show_scale_bar : bool
        Whether to show scale bar on first column (default True)
    pf_only_place_cells : bool
        If True (default), only plot PF contours for cells that are place cells (is_place_cell=True).
        If False, plot PF contours based on individual spike type criteria.
    plot_putative_PF : bool
        If True, plot dashed PF contours for putative place fields (significant but not place cell).
        If False (default), only plot solid PF contours for confirmed place cells.
    show_place_cell_star : bool
        If True (default), show ★ marker for confirmed place cells on rate maps.
        If False, hide the star markers.
    show_significance_marker : bool
        If True (default), show significance markers (*, **, ***) based on p-value.
        If False, hide the significance markers.
    show_theta_slow_corr_text : bool
        If True, display theta/slow correlation numbers below those maps (default False).
    plot_spike_shapes : bool
        If True (default), add 2 rows at the bottom with normalized spike/burst shapes.
    plot_spike_shapes_overall : bool
        If True (default), rows are Overall SS and Overall CB (combined in/out PF).
        If False, rows are In-PF and Out-PF (legacy behavior).
    min_shapes_per_condition : int or None
        Backward-compatible alias: if set, uses the same minimum for SS and CB.
    min_shapes_per_condition_ss : int
        Minimum number of SS waveforms required (default 10).
    min_shapes_per_condition_cb : int
        Minimum number of CB waveforms required (default 5).
    spike_shape_state : {'run', 'rest'}
        Which state to plot for the 2 spike-shape rows (default 'run').
    show_shape_counts : bool
        If True, display n=X counts for SS and CB waveforms on each shape row.
    shape_ylim : tuple or None
        (ymin, ymax) for spike shape rows. If None, auto-computed from data.
    print_removed_frames : bool
        If True (default), print removed-frame stats per cell (SNR-only and total)
        when available in `spatial_analysis_full.pkl`.
    include_plateau : bool
        If True (default), add plateau occurrence heatmap row(s) using merged data.
    plateau_state_mode : {'split', 'all', 'moving', 'resting'}
        Plateau row mode. 'split' adds both moving and resting rows.
    plateau_include_long_cb_as_plateau : bool
        If True (default), include long complex bursts as plateau intervals.
    plateau_cb_min_duration_ms : float
        Minimum CB duration (ms) to include as plateau when enabled (default 200 ms).
    plateau_speed_threshold : float
        Speed threshold for moving/resting split: moving >= threshold, resting < threshold.
    plateau_data_folder : str or None
        Optional root folder containing per-session merged files. If None, uses per-cell
        `data_folder` (added by `load_pooled_spatial_data`).
    
    Returns:
    --------
    fig : matplotlib Figure
    """
    from matplotlib.gridspec import GridSpec
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    
    # Normalize input: if cell_groups is a flat list of cells, wrap it
    if len(cell_groups) > 0 and isinstance(cell_groups[0], dict):
        cell_groups = [cell_groups]  # Single group
    
    n_groups = len(cell_groups)
    cells_per_group = [len(g) for g in cell_groups]
    total_cells = sum(cells_per_group)
    
    if total_cells == 0:
        print("No cells to plot!")
        return None
    
    # Flatten cells list for easy indexing
    all_cells = []
    for g in cell_groups:
        all_cells.extend(g)
    
    # Determine if we need gaps (only for multiple groups)
    use_gaps = n_groups > 1
    
    # Backward compatibility: allow a single threshold for both SS and CB.
    if min_shapes_per_condition is not None:
        if (min_shapes_per_condition_ss, min_shapes_per_condition_cb) == (10, 5):
            min_shapes_per_condition_ss = int(min_shapes_per_condition)
            min_shapes_per_condition_cb = int(min_shapes_per_condition)
    min_shapes_per_condition_ss = int(min_shapes_per_condition_ss)
    min_shapes_per_condition_cb = int(min_shapes_per_condition_cb)
    include_plateau = bool(include_plateau)
    plateau_state_mode = str(plateau_state_mode).strip().lower()
    valid_plateau_modes = {'split', 'all', 'moving', 'resting'}
    if plateau_state_mode not in valid_plateau_modes:
        raise ValueError(
            "plateau_state_mode must be one of {'split', 'all', 'moving', 'resting'}."
        )
    if (not np.isfinite(float(plateau_cb_min_duration_ms))) or float(plateau_cb_min_duration_ms) <= 0:
        raise ValueError("plateau_cb_min_duration_ms must be a finite number > 0.")
    if not np.isfinite(float(plateau_speed_threshold)):
        raise ValueError("plateau_speed_threshold must be a finite number.")
    if plateau_data_folder is not None:
        plateau_data_folder = os.path.abspath(str(plateau_data_folder))

    plot_spike_shapes_overall = bool(plot_spike_shapes_overall)
    if not bool(plot_spike_shapes):
        plot_spike_shapes_overall = False

    base_rows = 6  # trajectory, rate map, SS, CS, theta, slow
    plateau_row_modes: list[tuple[str, str]] = []
    if include_plateau:
        if plateau_state_mode == 'split':
            plateau_row_modes = [('moving', 'Plateau (moving)'), ('resting', 'Plateau (resting)')]
        elif plateau_state_mode == 'all':
            plateau_row_modes = [('all', 'Plateau')]
        elif plateau_state_mode == 'moving':
            plateau_row_modes = [('moving', 'Plateau')]
        else:
            plateau_row_modes = [('resting', 'Plateau')]
    n_plateau_rows = len(plateau_row_modes)
    map_rows = base_rows + n_plateau_rows
    plot_plateau_shapes = bool(include_plateau and plot_spike_shapes)
    n_shape_rows = (2 if plot_spike_shapes else 0) + (1 if plot_plateau_shapes else 0)
    n_rows = map_rows + n_shape_rows
    
    # Calculate width ratios
    if use_gaps:
        # With gaps between groups
        width_ratios = []
        col_to_cell = {}
        gap_cols = []
        cell_idx = 0
        col_idx = 0
        for g_idx, n_cells in enumerate(cells_per_group):
            for c in range(n_cells):
                width_ratios.append(1)
                col_to_cell[col_idx] = cell_idx
                col_idx += 1
                cell_idx += 1
            if g_idx < n_groups - 1:  # Add gap after each group except last
                width_ratios.append(gap_ratio)
                gap_cols.append(col_idx)
                col_idx += 1
    else:
        # No gaps - simple grid
        width_ratios = [1] * total_cells
        col_to_cell = {i: i for i in range(total_cells)}
        gap_cols = []
    
    n_cols = len(width_ratios)
    
    # Find the last/first data column for colorbars/labels
    last_data_col = max(col_to_cell.keys())
    first_data_col = min(col_to_cell.keys())
    
    # Row indices for optional spike-shape panels
    plateau_row_by_mode = {
        mode: (base_rows + idx) for idx, (mode, _) in enumerate(plateau_row_modes)
    }
    shape_row_top = map_rows
    shape_row_bottom = map_rows + 1
    plateau_shape_row = map_rows + 2
    
    # Get arena dimensions from first cell
    params = all_cells[0]['params']
    width_real = params['width_real']
    height_real = params['height_real']
    arena_aspect = height_real / width_real
    
    # Calculate figure dimensions dynamically
    left_margin = 0.4
    right_margin = 0.3
    top_margin = 0.25
    bottom_margin = 0.1
    
    total_width_units = float(sum(width_ratios))
    cell_width = float(subplot_width)
    cell_height = cell_width * arena_aspect
    
    row_gap = 0.1
    fig_width = left_margin + right_margin + total_width_units * cell_width
    fig_height = n_rows * cell_height + (n_rows - 1) * row_gap + top_margin + bottom_margin
    
    # Create figure
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = GridSpec(n_rows, n_cols, figure=fig, width_ratios=width_ratios,
                  left=left_margin/fig_width, right=1-right_margin/fig_width,
                  top=1-top_margin/fig_height, bottom=bottom_margin/fig_height,
                  wspace=0.05, hspace=0.15)
    
    # Create axes for data columns only (skip gap columns)
    axes_grid = {}
    
    # Base rows (maps + optional plateau rows)
    for row in range(map_rows):
        for col in range(n_cols):
            if col in gap_cols:
                continue
            axes_grid[(row, col)] = fig.add_subplot(gs[row, col])
    
    # Optional spike-shape rows (row-specific shared axes across columns)
    if plot_spike_shapes:
        anchor_ss = fig.add_subplot(gs[shape_row_top, first_data_col])
        anchor_cb = fig.add_subplot(gs[shape_row_bottom, first_data_col])
        axes_grid[(shape_row_top, first_data_col)] = anchor_ss
        axes_grid[(shape_row_bottom, first_data_col)] = anchor_cb

        for col in range(n_cols):
            if col in gap_cols or col == first_data_col:
                continue
            axes_grid[(shape_row_top, col)] = fig.add_subplot(
                gs[shape_row_top, col], sharex=anchor_ss, sharey=anchor_ss
            )
            axes_grid[(shape_row_bottom, col)] = fig.add_subplot(
                gs[shape_row_bottom, col], sharex=anchor_cb, sharey=anchor_cb
            )

    # Optional plateau-shape row (all plateau traces, no averaging)
    if plot_plateau_shapes:
        anchor_plateau = fig.add_subplot(gs[plateau_shape_row, first_data_col])
        axes_grid[(plateau_shape_row, first_data_col)] = anchor_plateau
        for col in range(n_cols):
            if col in gap_cols or col == first_data_col:
                continue
            axes_grid[(plateau_shape_row, col)] = fig.add_subplot(
                gs[plateau_shape_row, col], sharex=anchor_plateau, sharey=anchor_plateau
            )
    
    # Define colors and styling
    cmap = 'magma'
    slow_cmap = 'coolwarm'
    extent = (0, width_real, 0, height_real)
    simple_spike_color = "#026C80"
    complex_spike_color = "#EE9B00"
    ss_contour_color = "#026C80"
    
    def _style_map_axis(ax):
        ax.set_xlim(0, width_real)
        ax.set_ylim(0, height_real)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    
    def _plot_pf_contour(ax, pf_mask, color, linewidth=0.6, linestyle='solid', alpha=1.0):
        if pf_mask is None or not np.any(pf_mask):
            return
        padded_mask = np.pad(pf_mask.astype(float), pad_width=1, mode='constant', constant_values=0)
        bin_size = width_real / pf_mask.shape[0]
        padded_extent = (-bin_size, width_real + bin_size, -bin_size, height_real + bin_size)
        cs = ax.contour(padded_mask.T, levels=[0.5], colors=color, linewidths=linewidth,
                        linestyles=linestyle, extent=padded_extent, origin="lower",
                        alpha=alpha)
    
    def _get_sig_marker(p_val):
        if p_val < 0.001: return "***"
        elif p_val < 0.01: return "**"
        elif p_val < 0.05: return "*"
        else: return ""
    
    def _add_colorbar(ax, im, ticks=None, ticklabels=None):
        cax = inset_axes(ax, width="10%", height="100%", loc="center right",
                         bbox_to_anchor=(0.12, 0.0, 1, 1), bbox_transform=ax.transAxes, borderpad=0)
        cbar = ax.figure.colorbar(im, cax=cax)
        if ticks is not None and ticklabels is not None:
            cbar.set_ticks(ticks)
            cbar.set_ticklabels(ticklabels)
        cbar.ax.yaxis.set_ticks_position('right')
        cbar.ax.yaxis.set_label_position('right')
        cbar.ax.tick_params(labelsize=5, width=0.4, direction='out', right=True, left=False, labelright=True, labelleft=False)
        return cbar

    plateau_maps_by_cell: dict[int, dict[str, np.ndarray]] = {}
    merged_data_by_cell: dict[int, dict[str, Any] | None] = {}
    if include_plateau:
        merged_cache: dict[str, dict[str, Any] | None] = {}
        warned_sessions: set[str] = set()
        for flat_idx, cell in enumerate(all_cells):
            merged_data = _resolve_merged_data_for_cell(
                cell,
                plateau_data_folder=plateau_data_folder,
                merged_cache=merged_cache,
                warned_sessions=warned_sessions,
            )
            merged_data_by_cell[flat_idx] = merged_data
            plateau_maps_by_cell[flat_idx] = _compute_plateau_occurrence_maps_for_cell(
                cell,
                merged_data=merged_data,
                include_long_cb_as_plateau=bool(plateau_include_long_cb_as_plateau),
                cb_min_duration_ms=float(plateau_cb_min_duration_ms),
                speed_threshold=float(plateau_speed_threshold),
            )

    plateau_shape_traces_by_cell: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
    plateau_shape_xlim: tuple[float, float] | None = None
    plateau_shape_ylim: tuple[float, float] | None = None
    if plot_plateau_shapes:
        plateau_shape_pre_ms = 40.0
        plateau_shape_post_ms = 20.0
        x_min_ms = np.inf
        x_max_ms = -np.inf
        y_min_ps = np.inf
        y_max_ps = -np.inf

        for flat_idx, cell in enumerate(all_cells):
            merged_data = merged_data_by_cell.get(flat_idx)
            traces_for_cell: list[tuple[np.ndarray, np.ndarray]] = []
            if isinstance(merged_data, dict):
                src = merged_data.get(
                    "traces_SNR_interpolated",
                    merged_data.get("traces", []),
                )
                ci = int(cell.get("cell_idx", -1))
                trace = np.array([], dtype=float)
                if isinstance(src, (list, tuple, np.ndarray)) and 0 <= ci < len(src):
                    trace = np.asarray(src[ci], dtype=float).reshape(-1)
                if trace.size > 0:
                    starts, ends = _build_plateau_intervals_from_merged(
                        merged_data,
                        cell_idx=ci,
                        include_long_cb_as_plateau=bool(plateau_include_long_cb_as_plateau),
                        cb_min_duration_ms=float(plateau_cb_min_duration_ms),
                        n_frames=int(trace.size),
                    )
                    frame_rate_ps = float(merged_data.get("frame_rate", np.nan))
                    use_ms = np.isfinite(frame_rate_ps) and frame_rate_ps > 0
                    if use_ms:
                        pre_frames = max(0, int(np.ceil(float(plateau_shape_pre_ms) / 1000.0 * float(frame_rate_ps))))
                        post_frames = max(0, int(np.ceil(float(plateau_shape_post_ms) / 1000.0 * float(frame_rate_ps))))
                    else:
                        pre_frames = int(np.ceil(float(plateau_shape_pre_ms)))
                        post_frames = int(np.ceil(float(plateau_shape_post_ms)))
                    for s, e in zip(starts, ends):
                        s_i = int(s)
                        e_i = int(e)
                        if e_i < s_i:
                            continue
                        seg_start = max(0, s_i - pre_frames)
                        seg_end = min(int(trace.size) - 1, e_i + post_frames)
                        if seg_end < seg_start:
                            continue
                        seg = np.asarray(trace[seg_start:seg_end + 1], dtype=float)
                        if seg.size == 0 or not np.any(np.isfinite(seg)):
                            continue
                        if use_ms:
                            frame_idx = np.arange(seg_start, seg_end + 1, dtype=float)
                            x_ms = (frame_idx - float(s_i)) / float(frame_rate_ps) * 1000.0
                        else:
                            frame_idx = np.arange(seg_start, seg_end + 1, dtype=float)
                            x_ms = frame_idx - float(s_i)
                        traces_for_cell.append((x_ms, seg))
                        if x_ms.size:
                            x_min_ms = min(x_min_ms, float(np.nanmin(x_ms)))
                            x_max_ms = max(x_max_ms, float(np.nanmax(x_ms)))
                        finite_seg = seg[np.isfinite(seg)]
                        if finite_seg.size:
                            y_min_ps = min(y_min_ps, float(np.nanmin(finite_seg)))
                            y_max_ps = max(y_max_ps, float(np.nanmax(finite_seg)))
            plateau_shape_traces_by_cell[flat_idx] = traces_for_cell

        # Fixed display window for plateau-shape row.
        plateau_shape_xlim = (0.0, 250.0)
        if np.isfinite(y_min_ps) and np.isfinite(y_max_ps):
            if y_max_ps == y_min_ps:
                eps = 1e-3
                plateau_shape_ylim = (y_min_ps - eps, y_max_ps + eps)
            else:
                plateau_shape_ylim = (y_min_ps, y_max_ps)
        else:
            plateau_shape_ylim = (-0.2, 1.2)

    # Ensure plateau-shape row uses shared limits (all columns share x/y with anchor).
    if plot_plateau_shapes:
        anchor_plateau_ax = axes_grid.get((plateau_shape_row, first_data_col))
        if anchor_plateau_ax is not None:
            if plateau_shape_xlim is not None:
                anchor_plateau_ax.set_xlim(*plateau_shape_xlim)
            if plateau_shape_ylim is not None:
                anchor_plateau_ax.set_ylim(*plateau_shape_ylim)
    
    # Prepare shared axes and global limits for spike-shape rows
    shape_xlim = None
    _shape_ylim_user = shape_ylim  # preserve user override
    shape_ylim = None
    shape_gap_ms = None
    shape_ss_x_scale = None
    shape_cb_x_scale = None
    shape_ss_x_start = None
    shape_ss_x_end = None
    shape_cb_x_start = None
    shape_xlim_ss_full = None
    shape_xlim_cb_full = None
    ss_shape_row_ymin = None
    if plot_spike_shapes:
        ss_x_min = np.inf
        ss_x_max = -np.inf
        cb_x_min = np.inf
        cb_x_max = -np.inf
        y_min = np.inf
        y_max = -np.inf
        ss_y_min = np.inf
        
        if spike_shape_state not in ('run', 'rest'):
            raise ValueError("spike_shape_state must be 'run' or 'rest'")
        
        def _gather_waves(spike_shapes, spike_type, pf_key):
            info = spike_shapes.get(spike_type) if isinstance(spike_shapes, dict) else None
            if not info:
                return np.array([]), []
            time_ms = np.asarray(info.get('time_ms', []), dtype=float)
            shapes = info.get('shapes', {})
            in_key = f"{spike_shape_state}_in"
            out_key = f"{spike_shape_state}_out"
            if pf_key == 'in':
                waves = list(shapes.get(in_key, []))
            else:
                waves = list(shapes.get(out_key, []))
            return time_ms, waves
        
        def _mean_sem(waves):
            if len(waves) == 0:
                return None, None
            arr = np.vstack(waves)
            mean = np.nanmean(arr, axis=0)
            n_eff = np.sum(np.isfinite(arr), axis=0)
            std = np.nanstd(arr, axis=0, ddof=0)
            with np.errstate(divide='ignore', invalid='ignore'):
                sem = np.where(n_eff > 0, std / np.sqrt(n_eff), np.nan)
            return mean, sem
        
        def _min_req(spike_type):
            return (
                min_shapes_per_condition_ss
                if spike_type == 'simple'
                else min_shapes_per_condition_cb
            )
        
        def _normalize_wave(time_ms, wave):
            time_ms = np.asarray(time_ms, dtype=float)
            wave = np.asarray(wave, dtype=float)
            if time_ms.size == 0 or wave.size != time_ms.size:
                return None
            pre_mask = time_ms < 0
            if not np.any(pre_mask):
                return None
            baseline = np.nanmean(wave[pre_mask])
            if not np.isfinite(baseline):
                return None
            idx0 = int(np.nanargmin(np.abs(time_ms)))
            if not np.isfinite(wave[idx0]):
                return None
            height = wave[idx0] - baseline
            if not np.isfinite(height) or abs(height) < 1e-12:
                return None
            return (wave - baseline) / height
        
        def _normalize_waves(time_ms, waves):
            out = []
            for w in waves:
                w_n = _normalize_wave(time_ms, w)
                if w_n is None:
                    continue
                out.append(w_n)
            return out
        
        for c in all_cells:
            spike_shapes = c.get('spike_shapes')
            if not spike_shapes:
                continue
            for spike_type in ('simple', 'complex'):
                info = spike_shapes.get(spike_type) if isinstance(spike_shapes, dict) else None
                if info:
                    time_ms = np.asarray(info.get('time_ms', []), dtype=float)
                    if time_ms.size:
                        if spike_type == 'simple':
                            ss_x_min = min(ss_x_min, float(np.nanmin(time_ms)))
                            ss_x_max = max(ss_x_max, float(np.nanmax(time_ms)))
                        else:
                            cb_x_min = min(cb_x_min, float(np.nanmin(time_ms)))
                            cb_x_max = max(cb_x_max, float(np.nanmax(time_ms)))
                for pf_key in ('in', 'out'):
                    time_ms, waves = _gather_waves(spike_shapes, spike_type, pf_key)
                    norm_waves = _normalize_waves(time_ms, waves)
                    if len(norm_waves) == 0:
                        continue
                    mean, sem = _mean_sem(norm_waves)
                    if mean is None:
                        continue
                    low = mean
                    high = mean
                    if np.any(np.isfinite(low)):
                        low_min = float(np.nanmin(low))
                        y_min = min(y_min, low_min)
                        if spike_type == 'simple':
                            ss_y_min = min(ss_y_min, low_min)
                    if np.any(np.isfinite(high)):
                        y_max = max(y_max, float(np.nanmax(high)))
        
        if np.isfinite(ss_x_min) and np.isfinite(ss_x_max):
            shape_time_xlim_ss = (ss_x_min, ss_x_max)
        else:
            shape_time_xlim_ss = (-20.0, 20.0)
        if np.isfinite(cb_x_min) and np.isfinite(cb_x_max):
            shape_time_xlim_cb = (cb_x_min, cb_x_max)
        else:
            shape_time_xlim_cb = (-20.0, 80.0)
        shape_xlim_ss_full = (float(shape_time_xlim_ss[0]), float(shape_time_xlim_ss[1]))
        shape_xlim_cb_full = (float(shape_time_xlim_cb[0]), float(shape_time_xlim_cb[1]))
        shape_gap_ms = 5.0
        # Scale SS in time (x-axis) by 5; CB keeps its native x-scale.
        shape_ss_x_scale = 5.0
        shape_cb_x_scale = 1.0
        # Concatenate SS then CB with a 5 ms gap (in CB-scale units).
        shape_ss_x_start = 0.0
        shape_ss_x_end = float((shape_time_xlim_ss[1] - shape_time_xlim_ss[0]) * shape_ss_x_scale)
        shape_cb_x_start = float(shape_ss_x_end + shape_gap_ms * shape_cb_x_scale)
        shape_xlim = (
            float(shape_ss_x_start),
            float(shape_cb_x_start + (shape_time_xlim_cb[1] - shape_time_xlim_cb[0]) * shape_cb_x_scale),
        )
        if np.isfinite(y_min) and np.isfinite(y_max):
            if y_max == y_min:
                eps = 1e-3
                _shape_ylim_auto = (y_min - eps, y_max + eps)
            else:
                # Use global min/max (no extra padding) to keep the rows compact.
                _shape_ylim_auto = (y_min, y_max)
        else:
            _shape_ylim_auto = (-0.2, 1.2)
        # Clamp the lower bound to keep the rows visually compact while preserving baseline ~0.
        _shape_ylim_auto = (max(_shape_ylim_auto[0], -0.2), _shape_ylim_auto[1])
        # Use user-supplied limits if provided, otherwise use auto
        if _shape_ylim_user is not None:
            shape_ylim = _shape_ylim_user
            ss_shape_row_ymin = -0.15
        else:
            shape_ylim = _shape_ylim_auto
            if np.isfinite(ss_y_min):
                ss_shape_row_ymin = -0.15
            else:
                ss_shape_row_ymin = float(shape_ylim[0])
        
        # Shared axes are established via `sharex/sharey` when creating subplots.
        if plot_spike_shapes_overall:
            axes_grid[(shape_row_top, first_data_col)].set_xlim(*shape_xlim_ss_full)
            ss_y0 = -0.155
            ss_y1 = 1.1 if 1.1 > ss_y0 else (ss_y0 + 1e-3)
            axes_grid[(shape_row_top, first_data_col)].set_ylim(ss_y0, ss_y1)
            axes_grid[(shape_row_bottom, first_data_col)].set_xlim(*shape_xlim_cb_full)
            axes_grid[(shape_row_bottom, first_data_col)].set_ylim(*shape_ylim)
        else:
            axes_grid[(shape_row_top, first_data_col)].set_xlim(*shape_xlim)
            axes_grid[(shape_row_top, first_data_col)].set_ylim(*shape_ylim)
            axes_grid[(shape_row_bottom, first_data_col)].set_xlim(*shape_xlim)
            axes_grid[(shape_row_bottom, first_data_col)].set_ylim(*shape_ylim)
    
    # Plot each cell
    for display_col, cell_idx in col_to_cell.items():
        cell = all_cells[cell_idx]
        is_last_column = (display_col == last_data_col)
        is_first_column = (display_col == first_data_col)
        
        # Parse animal name
        animal_id = cell['animal_id']
        animal_short = animal_id.split('_')[1] if '_' in animal_id else animal_id
        cell_num = cell['cell_idx'] + 1
        is_place_cell = cell.get('is_place_cell', False)
        is_place_cell_ss = cell.get('is_place_cell_ss', False)
        is_place_cell_cs = cell.get('is_place_cell_cs', False)
        if print_removed_frames:
            n_total = cell.get('n_frames_total', None)
            n_removed_snr = cell.get('n_removed_frames_snr_only', None)
            n_removed_total = cell.get('n_removed_frames_total', None)
            pct_snr = cell.get('pct_removed_frames_snr_only', None)
            pct_total = cell.get('pct_removed_frames_total', None)
            if (
                isinstance(n_total, (int, np.integer))
                and isinstance(n_removed_snr, (int, np.integer))
                and isinstance(n_removed_total, (int, np.integer))
                and int(n_total) > 0
            ):
                pct_snr_str = f"{float(pct_snr):.2f}%" if np.isfinite(pct_snr) else "nan"
                pct_total_str = f"{float(pct_total):.2f}%" if np.isfinite(pct_total) else "nan"
                print(
                    f"{animal_short} Cell {cell_num}: "
                    f"removed_snr={int(n_removed_snr)}/{int(n_total)} ({pct_snr_str}), "
                    f"removed_total={int(n_removed_total)}/{int(n_total)} ({pct_total_str})"
                )
            else:
                print(f"{animal_short} Cell {cell_num}: removed-frame stats unavailable in this cache.")
        
        # Row 0: Trajectory
        ax_traj = axes_grid[(0, display_col)]
        x_traj, y_traj = cell['x_traj'], cell['y_traj']
        ax_traj.plot(x_traj, y_traj, color="gray", linewidth=0.3, alpha=0.5)
        ss_x = cell.get('ss_spikes_x', cell['spikes_x'])
        ss_y = cell.get('ss_spikes_y', cell['spikes_y'])
        if len(ss_x) > 0:
            ax_traj.scatter(ss_x, ss_y, s=0.5, color=simple_spike_color, alpha=0.4, linewidths=0, zorder=2, rasterized=True)
        cs_x = cell.get('cs_spikes_x', np.array([]))
        cs_y = cell.get('cs_spikes_y', np.array([]))
        if len(cs_x) > 0:
            ax_traj.scatter(cs_x, cs_y, s=0.5, color=complex_spike_color, alpha=0.4, linewidths=0, zorder=3, rasterized=True)
        _style_map_axis(ax_traj)
        ax_traj.set_title(f"{animal_short}\nCell {cell_num}", fontsize=5, fontname="Arial", pad=2)
        
        # Legend (trajectory row, rightmost column)
        if is_last_column:
            leg_ax = inset_axes(
                ax_traj,
                width="20%",
                height="55%",
                loc="center right",
                bbox_to_anchor=(0.27, 0.0, 1, 1),
                bbox_transform=ax_traj.transAxes,
                borderpad=0,
            )
            leg_ax.set_axis_off()
            leg_ax.set_xlim(0, 1)
            leg_ax.set_ylim(0, 1)
            leg_ax.scatter([0.25], [0.7], s=12, color=simple_spike_color, clip_on=False)
            leg_ax.text(0.75, 0.7, "SS", va="center", ha="left", fontsize=4, fontname="Arial")
            leg_ax.scatter([0.25], [0.3], s=12, color=complex_spike_color, clip_on=False)
            leg_ax.text(0.75, 0.3, "CB", va="center", ha="left", fontsize=4, fontname="Arial")
        
        # Add 10 cm scale bar on first column only
        if is_first_column and show_scale_bar:
            scale_bar_length = 10  # cm
            x_start = 1
            y_pos = -1
            ax_traj.plot([x_start, x_start + scale_bar_length], [y_pos, y_pos], 
                        color='black', linewidth=1.5, solid_capstyle='butt', clip_on=False)
            ax_traj.text(x_start + scale_bar_length/2, y_pos - 1.5, '10 cm', 
                        ha='center', va='top', fontsize=5, fontname='Arial', clip_on=False)
        
        # Row 1: Rate map (all spikes)
        ax_rate = axes_grid[(1, display_col)]
        rate_map = cell['rate_map']
        peak_rate = cell['peak_rate']
        p_val = cell.get('p_value', 1.0)
        sig_mark = _get_sig_marker(p_val)
        im_rate = None
        if rate_map is not None:
            masked_map = ma.masked_where(np.isnan(rate_map), rate_map)
            im_rate = ax_rate.imshow(masked_map.T, origin="lower", extent=extent, cmap=cmap,
                          interpolation="nearest", vmin=0,
                          vmax=peak_rate if np.isfinite(peak_rate) and peak_rate > 0 else None)
            # Plot PF contour: solid if place cell (★), dashed if significant but not place cell (*)
            pf_mask = cell['place_field_mask']
            if pf_mask is None or not np.any(pf_mask):
                print(f"{animal_short} Cell {cell_num}: All spikes PF mask is empty")
            elif is_place_cell:
                _plot_pf_contour(ax_rate, pf_mask, "magenta", linestyle='solid')
            elif p_val < 0.05 and not pf_only_place_cells and plot_putative_PF:
                _plot_pf_contour(ax_rate, pf_mask, "magenta", linestyle='dashed')
        _style_map_axis(ax_rate)
        rate_str = f"{peak_rate:.1f}" if np.isfinite(peak_rate) else "N/A"
        display_sig_mark = sig_mark if show_significance_marker else ""
        label_str = f"{display_sig_mark} {rate_str} Hz".strip()
        ax_rate.text(1.0, -0.02, label_str, transform=ax_rate.transAxes,
                     ha="right", va="top", fontsize=4, fontname="Arial")
        # Add star marker for place cells (renders as vector path in Illustrator)
        if is_place_cell and show_place_cell_star:
            ax_rate.plot(0.03, -0.06, marker='*', markersize=4, color='black',
                        transform=ax_rate.transAxes, clip_on=False)
        if is_last_column and im_rate is not None:
            _add_colorbar(ax_rate, im_rate, ticks=[0, im_rate.get_clim()[1]], ticklabels=["0", "max"])
        
        # Row 2: SS normalized map
        ax_ss = axes_grid[(2, display_col)]
        ss_norm_map = cell['ss_norm_map']
        ss_p_val = cell.get('ss_p_value', 1.0)
        ss_sig_mark = _get_sig_marker(ss_p_val)
        ss_mask = cell.get('ss_place_field_mask', None)
        im_ss = None
        if ss_norm_map is not None:
            ss_masked = ma.masked_where(np.isnan(ss_norm_map), ss_norm_map)
            im_ss = ax_ss.imshow(ss_masked.T, origin="lower", extent=extent, cmap=cmap,
                        interpolation="nearest", vmin=0, vmax=1)
            # Plot SS contour: solid if SS place cell, dashed if PF exists but not place cell
            if ss_mask is None or not np.any(ss_mask):
                print(f"{animal_short} Cell {cell_num}: SS PF mask is empty")
            elif pf_only_place_cells and not is_place_cell:
                pass  # Skip SS PF if cell is not an all-spike place cell
            elif is_place_cell_ss:
                _plot_pf_contour(ax_ss, ss_mask, ss_contour_color, linestyle='solid')
            elif plot_putative_PF:
                _plot_pf_contour(ax_ss, ss_mask, ss_contour_color, linestyle='dashed')
        _style_map_axis(ax_ss)
        ss_peak = cell['ss_peak_rate']
        ss_str = f"{ss_peak:.1f}" if np.isfinite(ss_peak) else "N/A"
        # Add star and significance for SS (star only if is_place_cell_ss)
        ss_display_sig = ss_sig_mark if show_significance_marker else ""
        ss_label_str = f"{ss_display_sig} {ss_str} Hz".strip()
        ax_ss.text(1.0, -0.02, ss_label_str, transform=ax_ss.transAxes,
                   ha="right", va="top", fontsize=4, fontname="Arial")
        # Add star marker for SS place cells
        if is_place_cell_ss and show_place_cell_star:
            ax_ss.plot(0.03, -0.06, marker='*', markersize=4, color='black',
                      transform=ax_ss.transAxes, clip_on=False)
        if is_last_column and im_ss is not None:
            _add_colorbar(ax_ss, im_ss)
        
        # Row 3: CS normalized map
        ax_cs = axes_grid[(3, display_col)]
        cs_norm_map = cell['cs_norm_map']
        cs_p_val = cell.get('cs_p_value', 1.0)
        cs_sig_mark = _get_sig_marker(cs_p_val)
        cs_mask = cell.get('cs_place_field_mask', None)
        im_cs = None
        if cs_norm_map is not None:
            cs_masked = ma.masked_where(np.isnan(cs_norm_map), cs_norm_map)
            im_cs = ax_cs.imshow(cs_masked.T, origin="lower", extent=extent, cmap=cmap,
                        interpolation="nearest", vmin=0, vmax=1)
            # Plot CS contour: solid if CS place cell, dashed if PF exists but not place cell
            if cs_mask is None or not np.any(cs_mask):
                print(f"{animal_short} Cell {cell_num}: CS PF mask is empty")
            elif pf_only_place_cells and not is_place_cell:
                pass  # Skip CS PF if cell is not an all-spike place cell
            elif is_place_cell_cs:
                _plot_pf_contour(ax_cs, cs_mask, complex_spike_color, linestyle='solid')
            elif plot_putative_PF:
                _plot_pf_contour(ax_cs, cs_mask, complex_spike_color, linestyle='dashed')
        _style_map_axis(ax_cs)
        cs_peak = cell['cs_peak_rate']
        cs_str = f"{cs_peak:.1f}" if np.isfinite(cs_peak) else "N/A"
        # Add star and significance for CS (star only if is_place_cell_cs)
        cs_display_sig = cs_sig_mark if show_significance_marker else ""
        cs_label_str = f"{cs_display_sig} {cs_str} Hz".strip()
        ax_cs.text(1.0, -0.02, cs_label_str, transform=ax_cs.transAxes,
                   ha="right", va="top", fontsize=4, fontname="Arial")
        # Add star marker for CS place cells
        if is_place_cell_cs and show_place_cell_star:
            ax_cs.plot(0.03, -0.06, marker='*', markersize=4, color='black',
                      transform=ax_cs.transAxes, clip_on=False)
        if is_last_column and im_cs is not None:
            _add_colorbar(ax_cs, im_cs)
        
        # Row 4: Theta amplitude map
        ax_theta = axes_grid[(4, display_col)]
        theta_map = cell.get('theta_map', None)
        im_theta = None
        if theta_map is not None and np.any(np.isfinite(theta_map)):
            theta_masked = ma.masked_where(np.isnan(theta_map), theta_map)
            theta_vmin, theta_vmax = theta_vlim if theta_vlim else (np.nanmin(theta_map), np.nanmax(theta_map))
            im_theta = ax_theta.imshow(theta_masked.T, origin="lower", extent=extent, cmap=cmap,
                           interpolation="nearest", vmin=theta_vmin, vmax=theta_vmax)
        _style_map_axis(ax_theta)
        if show_theta_slow_corr_text:
            theta_corr_all = cell.get('theta_corr_all', np.nan)
            theta_corr_ss = cell.get('theta_corr_ss', np.nan)
            theta_corr_cs = cell.get('theta_corr_cs', np.nan)
            corr_strs = []
            if np.isfinite(theta_corr_all): corr_strs.append((f"{theta_corr_all:.2f}", "black"))
            if np.isfinite(theta_corr_ss): corr_strs.append((f"{theta_corr_ss:.2f}", simple_spike_color))
            if np.isfinite(theta_corr_cs): corr_strs.append((f"{theta_corr_cs:.2f}", complex_spike_color))
            x_positions = [0.25, 0.5, 0.75] if len(corr_strs) == 3 else ([0.33, 0.67] if len(corr_strs) == 2 else [0.5])
            for i, (corr_str, color) in enumerate(corr_strs):
                ax_theta.text(x_positions[i], -0.02, corr_str, transform=ax_theta.transAxes,
                             ha="center", va="top", fontsize=4, fontname="Arial", color=color)
        if is_last_column and im_theta is not None:
            _add_colorbar(ax_theta, im_theta)
        
        # Row 5: Slow Vm map
        ax_slow = axes_grid[(5, display_col)]
        slow_map = cell.get('slow_map', None)
        im_slow = None
        if slow_map is not None and np.any(np.isfinite(slow_map)):
            slow_masked = ma.masked_where(np.isnan(slow_map), slow_map)
            slow_vmin, slow_vmax = slow_vlim if slow_vlim else (-np.nanmax(np.abs(slow_map)), np.nanmax(np.abs(slow_map)))
            im_slow = ax_slow.imshow(slow_masked.T, origin="lower", extent=extent, cmap=slow_cmap,
                          interpolation="nearest", vmin=slow_vmin, vmax=slow_vmax)
        _style_map_axis(ax_slow)
        if show_theta_slow_corr_text:
            slow_corr_all = cell.get('slow_corr_all', np.nan)
            slow_corr_ss = cell.get('slow_corr_ss', np.nan)
            slow_corr_cs = cell.get('slow_corr_cs', np.nan)
            slow_corr_strs = []
            if np.isfinite(slow_corr_all): slow_corr_strs.append((f"{slow_corr_all:.2f}", "black"))
            if np.isfinite(slow_corr_ss): slow_corr_strs.append((f"{slow_corr_ss:.2f}", simple_spike_color))
            if np.isfinite(slow_corr_cs): slow_corr_strs.append((f"{slow_corr_cs:.2f}", complex_spike_color))
            slow_x_positions = [0.25, 0.5, 0.75] if len(slow_corr_strs) == 3 else ([0.33, 0.67] if len(slow_corr_strs) == 2 else [0.5])
            for i, (corr_str, color) in enumerate(slow_corr_strs):
                ax_slow.text(slow_x_positions[i], -0.02, corr_str, transform=ax_slow.transAxes,
                            ha="center", va="top", fontsize=4, fontname="Arial", color=color)
        if is_last_column and im_slow is not None:
            _add_colorbar(ax_slow, im_slow)

        # Optional plateau occurrence map row(s)
        if include_plateau:
            plateau_maps = plateau_maps_by_cell.get(cell_idx, {})
            for mode, _label in plateau_row_modes:
                row_idx = plateau_row_by_mode[mode]
                ax_plateau = axes_grid[(row_idx, display_col)]
                plateau_map = plateau_maps.get(mode, None)
                im_plateau = None
                plateau_max_occ = np.nan
                if plateau_map is not None and np.any(np.isfinite(plateau_map)):
                    plateau_masked = ma.masked_where(np.isnan(plateau_map), plateau_map)
                    plateau_max_occ = float(np.nanmax(plateau_map))
                    vmax = plateau_max_occ if np.isfinite(plateau_max_occ) and plateau_max_occ > 0 else 1.0
                    im_plateau = ax_plateau.imshow(
                        plateau_masked.T,
                        origin="lower",
                        extent=extent,
                        cmap="Reds",
                        interpolation="nearest",
                        vmin=0.0,
                        vmax=vmax,
                    )
                _style_map_axis(ax_plateau)
                # Overlay the same magenta place field contour as the rate map row
                pf_mask = cell['place_field_mask']
                if pf_mask is not None and np.any(pf_mask):
                    if is_place_cell:
                        _plot_pf_contour(ax_plateau, pf_mask, "magenta", linewidth=0.3, linestyle='solid', alpha=0.6)
                    elif p_val < 0.05 and not pf_only_place_cells and plot_putative_PF:
                        _plot_pf_contour(ax_plateau, pf_mask, "magenta", linewidth=0.3, linestyle='dashed', alpha=0.6)
                occ_text = f"max {plateau_max_occ:g}" if np.isfinite(plateau_max_occ) else "max N/A"
                ax_plateau.text(
                    1.0,
                    -0.02,
                    occ_text,
                    transform=ax_plateau.transAxes,
                    ha="right",
                    va="top",
                    fontsize=4,
                    fontname="Arial",
                )
                if is_last_column and im_plateau is not None:
                    vmax = float(im_plateau.get_clim()[1])
                    _add_colorbar(ax_plateau, im_plateau, ticks=[0.0, vmax], ticklabels=["0", "max"])

        # Optional spike-shape rows
        if plot_spike_shapes:
            spike_shapes = cell.get('spike_shapes')

            if plot_spike_shapes_overall:
                def _plot_shapes_axis_overall(ax, spike_type):
                    ax.set_facecolor('white')
                    axis_xlim = shape_xlim_ss_full if spike_type == 'simple' else shape_xlim_cb_full
                    tmin, tmax = shape_time_xlim_ss if spike_type == 'simple' else shape_time_xlim_cb
                    color = simple_spike_color if spike_type == 'simple' else complex_spike_color
                    if spike_type == 'simple':
                        y0 = -0.15
                        y1 = 1.1
                        if y1 <= y0:
                            y1 = y0 + 1e-3
                        axis_ylim = (y0, y1)
                    else:
                        axis_ylim = shape_ylim
                    ax.set_xlim(*axis_xlim)
                    ax.set_ylim(*axis_ylim)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    for spine in ax.spines.values():
                        spine.set_visible(False)

                    if not spike_shapes:
                        return

                    info = spike_shapes.get(spike_type) if isinstance(spike_shapes, dict) else None
                    if not info:
                        return
                    time_ms_full = np.asarray(info.get('time_ms', []), dtype=float)
                    shapes = info.get('shapes', {})
                    keys = [f"{spike_shape_state}_in", f"{spike_shape_state}_out"]
                    waves = []
                    for k in keys:
                        waves.extend(list(shapes.get(k, [])))
                    if time_ms_full.size == 0 or len(waves) == 0:
                        return

                    tmask = (time_ms_full >= tmin) & (time_ms_full <= tmax)
                    if not np.any(tmask):
                        return
                    time_ms = time_ms_full[tmask]
                    x = time_ms
                    norm_waves = []
                    for w in waves:
                        w_arr = np.asarray(w, dtype=float)
                        if w_arr.size != time_ms_full.size:
                            continue
                        w_n = _normalize_wave(time_ms_full, w_arr)
                        if w_n is None:
                            continue
                        norm_waves.append(w_n)
                        ax.plot(x, w_n[tmask], color=color, alpha=0.1, linewidth=0.1, rasterized=True)

                    if len(norm_waves) >= _min_req(spike_type):
                        mean, _ = _mean_sem(norm_waves)
                        if mean is not None:
                            ax.plot(x, mean[tmask], color=color, alpha=1.0, linewidth=1.0)

                    if show_shape_counts:
                        x_mid = (x[0] + x[-1]) / 2
                        y_top = axis_ylim[1] - 0.08 * (axis_ylim[1] - axis_ylim[0])
                        ax.text(x_mid, y_top, f'n={len(norm_waves)}',
                                ha='center', va='top', fontsize=3.5, fontname='Arial',
                                color=color, alpha=0.8)

                    y_line_low = max(axis_ylim[0], min(0.0, axis_ylim[1]))
                    y_line_high = max(axis_ylim[0], min(1.0, axis_ylim[1]))
                    y_line_low, y_line_high = sorted((y_line_low, y_line_high))
                    if np.isfinite(y_line_low) and np.isfinite(y_line_high) and y_line_high > y_line_low:
                        ax.plot([0.0, 0.0], [y_line_low, y_line_high], color='black', linestyle='--', linewidth=0.5, alpha=0.35)

                ax_ss_shape = axes_grid[(shape_row_top, display_col)]
                ax_cb_shape = axes_grid[(shape_row_bottom, display_col)]
                _plot_shapes_axis_overall(ax_ss_shape, 'simple')
                _plot_shapes_axis_overall(ax_cb_shape, 'complex')

                if is_last_column:
                    leg_ax = inset_axes(
                        ax_ss_shape,
                        width="16%",
                        height="55%",
                        loc="center right",
                        bbox_to_anchor=(0.12, 0.0, 1, 1),
                        bbox_transform=ax_ss_shape.transAxes,
                        borderpad=0,
                    )
                    leg_ax.set_axis_off()
                    leg_ax.set_xlim(0, 1)
                    leg_ax.set_ylim(0, 1)
                    leg_ax.plot([0.0, 0.6], [0.7, 0.7], color=simple_spike_color, linewidth=1.2)
                    leg_ax.text(0.7, 0.7, "SS", va="center", ha="left", fontsize=4, fontname="Arial")
                    leg_ax.plot([0.0, 0.6], [0.3, 0.3], color=complex_spike_color, linewidth=1.2)
                    leg_ax.text(0.7, 0.3, "CB", va="center", ha="left", fontsize=4, fontname="Arial")

                if is_first_column:
                    def _place_bar(target_ax, bar_len_x, label, y_frac=0.08):
                        xlims = target_ax.get_xlim()
                        ylims = target_ax.get_ylim()
                        seg_x0, seg_x1 = float(xlims[0]), float(xlims[1])
                        y_span = float(ylims[1] - ylims[0]) if ylims[1] != ylims[0] else 1.0
                        y0 = float(ylims[0] + float(y_frac) * y_span)
                        seg_span = float(seg_x1 - seg_x0) if seg_x1 != seg_x0 else 1.0
                        if not np.isfinite(seg_span) or seg_span <= 0:
                            return
                        bar_len_x = float(min(bar_len_x, 0.8 * seg_span))
                        x_left = float(seg_x0 + 0.08 * seg_span)
                        x_right = float(seg_x1 - 0.08 * seg_span)
                        if x_right <= x_left:
                            x_left, x_right = seg_x0, seg_x1
                        x0 = x_left
                        x1 = x0 + bar_len_x
                        if x1 > x_right:
                            x1 = x_right
                            x0 = x1 - bar_len_x
                        target_ax.plot([x0, x1], [y0, y0], color='black', linewidth=0.8, solid_capstyle='butt')
                        target_ax.text((x0 + x1) / 2, y0 - 0.06 * y_span, label, ha='center', va='top', fontsize=4, fontname='Arial')

                    _place_bar(ax_ss_shape, 10.0, '10 ms', y_frac=0.22)
                    _place_bar(ax_cb_shape, 50.0, '50 ms', y_frac=0.08)

            else:
                def _plot_shapes_axis_split(ax, pf_key, no_pf=False):
                    bg_color = "#FFE6F2" if pf_key == 'in' else "#F0F0F0"
                    ax.set_facecolor(bg_color)
                    ax.set_xlim(*shape_xlim)
                    ax.set_ylim(*shape_ylim)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    for spine in ax.spines.values():
                        spine.set_visible(False)
                    if no_pf and pf_key == 'in':
                        ax.set_facecolor('white')
                        return
                    if not spike_shapes:
                        return

                    def _plot_block(spike_type, color):
                        info = spike_shapes.get(spike_type) if isinstance(spike_shapes, dict) else None
                        if not info:
                            return
                        time_ms_full = np.asarray(info.get('time_ms', []), dtype=float)
                        shapes = info.get('shapes', {})
                        if no_pf and pf_key == 'out':
                            keys = [f"{spike_shape_state}_in", f"{spike_shape_state}_out"]
                        else:
                            keys = [f"{spike_shape_state}_in" if pf_key == 'in' else f"{spike_shape_state}_out"]
                        waves = []
                        for k in keys:
                            waves.extend(list(shapes.get(k, [])))
                        if time_ms_full.size == 0 or len(waves) == 0:
                            return
                        if spike_type == 'simple':
                            tmin, tmax = shape_time_xlim_ss
                            x_start = shape_ss_x_start
                            x_scale = shape_ss_x_scale
                        else:
                            tmin, tmax = shape_time_xlim_cb
                            x_start = shape_cb_x_start
                            x_scale = shape_cb_x_scale
                        tmask = (time_ms_full >= tmin) & (time_ms_full <= tmax)
                        if not np.any(tmask):
                            return
                        time_ms = time_ms_full[tmask]
                        x = (time_ms - tmin) * x_scale + x_start
                        norm_waves = []
                        for w in waves:
                            w_arr = np.asarray(w, dtype=float)
                            if w_arr.size != time_ms_full.size:
                                continue
                            w_n = _normalize_wave(time_ms_full, w_arr)
                            if w_n is None:
                                continue
                            norm_waves.append(w_n)
                            ax.plot(x, w_n[tmask], color=color, alpha=0.1, linewidth=0.1, rasterized=True)

                        if len(norm_waves) >= _min_req(spike_type):
                            mean, _ = _mean_sem(norm_waves)
                            if mean is not None:
                                ax.plot(x, mean[tmask], color=color, alpha=1.0, linewidth=1.0)

                        if show_shape_counts:
                            x_mid = (x[0] + x[-1]) / 2
                            y_top = shape_ylim[1] - 0.08 * (shape_ylim[1] - shape_ylim[0])
                            ax.text(x_mid, y_top, f'n={len(norm_waves)}',
                                    ha='center', va='top', fontsize=3.5, fontname='Arial',
                                    color=color, alpha=0.8)

                    _plot_block('simple', simple_spike_color)
                    _plot_block('complex', complex_spike_color)

                    y_line_low = max(shape_ylim[0], min(0.0, shape_ylim[1]))
                    y_line_high = max(shape_ylim[0], min(1.0, shape_ylim[1]))
                    y_line_low, y_line_high = sorted((y_line_low, y_line_high))
                    if np.isfinite(y_line_low) and np.isfinite(y_line_high) and y_line_high > y_line_low:
                        x0_ss = (0.0 - shape_time_xlim_ss[0]) * shape_ss_x_scale + shape_ss_x_start
                        x0_cb = (0.0 - shape_time_xlim_cb[0]) * shape_cb_x_scale + shape_cb_x_start
                        ax.plot([x0_ss, x0_ss], [y_line_low, y_line_high], color='black', linestyle='--', linewidth=0.5, alpha=0.35)
                        ax.plot([x0_cb, x0_cb], [y_line_low, y_line_high], color='black', linestyle='--', linewidth=0.5, alpha=0.35)

                pf_mask_all = cell.get('place_field_mask', None)
                no_pf = (not bool(is_place_cell)) or (pf_mask_all is None) or (not np.any(np.asarray(pf_mask_all)))
                ax_in_shape = axes_grid[(shape_row_top, display_col)]
                ax_out_shape = axes_grid[(shape_row_bottom, display_col)]
                _plot_shapes_axis_split(ax_in_shape, 'in', no_pf=no_pf)
                _plot_shapes_axis_split(ax_out_shape, 'out', no_pf=no_pf)

                if is_last_column:
                    target_ax = ax_out_shape if no_pf else ax_in_shape
                    leg_ax = inset_axes(
                        target_ax,
                        width="16%",
                        height="55%",
                        loc="center right",
                        bbox_to_anchor=(0.12, 0.0, 1, 1),
                        bbox_transform=target_ax.transAxes,
                        borderpad=0,
                    )
                    leg_ax.set_axis_off()
                    leg_ax.set_xlim(0, 1)
                    leg_ax.set_ylim(0, 1)
                    leg_ax.plot([0.0, 0.6], [0.7, 0.7], color=simple_spike_color, linewidth=1.2)
                    leg_ax.text(0.7, 0.7, "SS", va="center", ha="left", fontsize=4, fontname="Arial")
                    leg_ax.plot([0.0, 0.6], [0.3, 0.3], color=complex_spike_color, linewidth=1.2)
                    leg_ax.text(0.7, 0.3, "CB", va="center", ha="left", fontsize=4, fontname="Arial")

                if is_first_column:
                    y_span = float(shape_ylim[1] - shape_ylim[0]) if shape_ylim[1] != shape_ylim[0] else 1.0
                    y0 = float(shape_ylim[0] + 0.08 * y_span)

                    def _place_bar(seg_x0, seg_x1, bar_len_x, label):
                        seg_x0 = float(seg_x0)
                        seg_x1 = float(seg_x1)
                        seg_span = float(seg_x1 - seg_x0) if seg_x1 != seg_x0 else 1.0
                        if not np.isfinite(seg_span) or seg_span <= 0:
                            return
                        bar_len_x = float(min(bar_len_x, 0.8 * seg_span))
                        x_left = float(seg_x0 + 0.08 * seg_span)
                        x_right = float(seg_x1 - 0.08 * seg_span)
                        if x_right <= x_left:
                            x_left, x_right = seg_x0, seg_x1
                        x0 = x_left
                        x1 = x0 + bar_len_x
                        if x1 > x_right:
                            x1 = x_right
                            x0 = x1 - bar_len_x
                        ax_out_shape.plot([x0, x1], [y0, y0], color='black', linewidth=0.8, solid_capstyle='butt')
                        ax_out_shape.text((x0 + x1) / 2, y0 - 0.06 * y_span, label, ha='center', va='top', fontsize=4, fontname='Arial')

                    _place_bar(shape_ss_x_start, shape_ss_x_end, 10.0 * shape_ss_x_scale, '10 ms')
                    _place_bar(shape_cb_x_start, shape_xlim[1], 50.0 * shape_cb_x_scale, '50 ms')

        if plot_plateau_shapes:
            ax_plateau_shape = axes_grid[(plateau_shape_row, display_col)]
            ax_plateau_shape.set_facecolor('white')
            if plateau_shape_xlim is not None:
                ax_plateau_shape.set_xlim(*plateau_shape_xlim)
            if plateau_shape_ylim is not None:
                ax_plateau_shape.set_ylim(*plateau_shape_ylim)
            ax_plateau_shape.set_xticks([])
            ax_plateau_shape.set_yticks([])
            for spine in ax_plateau_shape.spines.values():
                spine.set_visible(False)

            for x_ms, seg in plateau_shape_traces_by_cell.get(cell_idx, []):
                if x_ms.size == 0 or seg.size != x_ms.size:
                    continue
                ax_plateau_shape.plot(
                    x_ms,
                    seg,
                    color='red',
                    alpha=0.5,
                    linewidth=0.3,
                    rasterized=True,
                )
            if is_first_column:
                xlims = ax_plateau_shape.get_xlim()
                ylims = ax_plateau_shape.get_ylim()
                x0_lim, x1_lim = float(xlims[0]), float(xlims[1])
                y0_lim, y1_lim = float(ylims[0]), float(ylims[1])
                x_span = float(x1_lim - x0_lim) if x1_lim != x0_lim else 1.0
                y_span = float(y1_lim - y0_lim) if y1_lim != y0_lim else 1.0
                if np.isfinite(x_span) and x_span > 0:
                    bar_len_ms = float(min(100.0, 0.8 * x_span))
                    x_left = float(x0_lim + 0.08 * x_span)
                    x_right = float(x1_lim - 0.08 * x_span)
                    if x_right <= x_left:
                        x_left, x_right = x0_lim, x1_lim
                    x_bar0 = x_left
                    x_bar1 = x_bar0 + bar_len_ms
                    if x_bar1 > x_right:
                        x_bar1 = x_right
                        x_bar0 = x_bar1 - bar_len_ms
                    y_bar = float(y0_lim + 0.08 * y_span)
                    ax_plateau_shape.plot([x_bar0, x_bar1], [y_bar, y_bar], color='black', linewidth=0.8, solid_capstyle='butt')
                    ax_plateau_shape.text((x_bar0 + x_bar1) / 2, y_bar - 0.06 * y_span, '100 ms',
                                          ha='center', va='top', fontsize=4, fontname='Arial')
    # Add row labels on first column
    row_labels = ['Trajectory', 'All spikes', 'SS', 'CS', 'Theta', 'Slow Vm']
    for _, label in plateau_row_modes:
        row_labels.append(label)
    if plot_spike_shapes:
        if plot_spike_shapes_overall:
            row_labels.extend(['SS shape', 'CB shape'])
        else:
            row_labels.extend(['In-PF shapes', 'Out-PF shapes'])
    if plot_plateau_shapes:
        row_labels.append('All plateaus')
    for row_idx, label in enumerate(row_labels):
        axes_grid[(row_idx, first_data_col)].text(-0.15, 0.5, label, 
            transform=axes_grid[(row_idx, first_data_col)].transAxes,
            ha="right", va="center", fontsize=5, fontname="Arial", rotation=90)
    
    # Save figure
    if save_path:
        plt.rcParams["svg.fonttype"] = "none"
        fig.savefig(save_path, dpi=300)
        print(f"Saved: {save_path}")
    
    plt.show()
    return fig


def _render_spatial_heatmap_chunks(cells, base_filename, **plot_kwargs):
    cells = list(cells)
    n_cells = len(cells)
    if n_cells == 0:
        print(f"No cells to plot for {base_filename}")
        return []

    max_per_fig = int(globals().get('SPATIAL_MAX_CELLS_PER_FIGURE', 10))
    max_per_fig = max(1, max_per_fig)
    n_chunks = (n_cells + max_per_fig - 1) // max_per_fig

    out_paths = []
    root, ext = os.path.splitext(base_filename)
    for i in range(n_chunks):
        chunk = cells[i * max_per_fig:(i + 1) * max_per_fig]
        if n_chunks == 1:
            filename = base_filename
        else:
            filename = f"{root}_part{i+1:02d}{ext}"
        out_path = os.path.join(figure_save_folder, filename)
        print(f"Rendering {filename}: cells={len(chunk)}")
        plot_selected_cells_figure(
            cell_groups=chunk,
            subplot_width=float(globals().get('SPATIAL_SUBPLOT_WIDTH', 0.25)),
            save_path=out_path,
            **plot_kwargs,
        )
        out_paths.append(out_path)
    return out_paths


def render_spatial_heatmap_chunks(cells, figure_save_folder, base_filename, max_cells_per_figure=10, subplot_width=0.35, **plot_kwargs):
    """Explicit chunk renderer without globals()."""
    cells = list(cells)
    n_cells = len(cells)
    if n_cells == 0:
        print(f"No cells to plot for {base_filename}")
        return []

    max_per_fig = max(1, int(max_cells_per_figure))
    n_chunks = (n_cells + max_per_fig - 1) // max_per_fig

    out_paths = []
    root, ext = os.path.splitext(base_filename)
    for i in range(n_chunks):
        chunk = cells[i * max_per_fig:(i + 1) * max_per_fig]
        filename = base_filename if n_chunks == 1 else f"{root}_part{i+1:02d}{ext}"
        out_path = os.path.join(figure_save_folder, filename)
        print(f"Rendering {filename}: cells={len(chunk)}")
        plot_selected_cells_figure(
            cell_groups=chunk,
            subplot_width=float(subplot_width),
            save_path=out_path,
            **plot_kwargs,
        )
        out_paths.append(out_path)
    return out_paths
