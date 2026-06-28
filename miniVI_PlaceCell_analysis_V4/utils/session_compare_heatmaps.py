"""Session-compare spatial heatmaps for Unified CKII workflow.

This module ports the heatmap workflow from
PooledFigure_CKII_Stats_V2_SpatialSessionCompare.ipynb into reusable functions.
"""

from __future__ import annotations

import copy
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import pandas as pd

try:
    from scipy import stats as scipy_stats
except Exception:  # pragma: no cover
    scipy_stats = None
try:
    from scipy import ndimage as scipy_ndimage
except Exception:  # pragma: no cover
    scipy_ndimage = None

from utils.placecell_pipeline import (
    PipelineConfig,
    _clean_behavior_speed_outliers_for_cache,
    _get_spike_positions_on_traj,
    _load_merged_data,
    _normalize_two_session_split_mode,
    _prepare_native_analysis_context,
    _resolve_merged_data_path,
    _run_place_cell_analysis_native,
    _validate_two_session_split_window_minutes,
)
from utils import placecell_core as core
from utils.spatial_heatmaps import cell_has_cs_place_field, is_csplus_place_cell
from utils.spatial_heatmaps import (
    _build_plateau_intervals_from_merged,
    _compute_plateau_occurrence_maps_for_cell,
)


@dataclass
class SessionCompareParams:
    panel_mode: str = "s1_s2"  # "s1_s2" or "combined_s1_s2"
    cache_version: str = "v2"
    rebuild_cache: bool = False
    max_cells_per_figure: int = 10
    missing_s2_policy: str = "na_panel"
    occupancy_spearman_threshold: float = -0.5
    apply_occupancy_dataset_filter: bool = False
    enforce_s1s2_min_peak_rate_filter: bool = True
    s1s2_min_peak_rate_hz: float = 0.5
    clean_heatmap: bool = True
    heatmap_similarity_metric: str = "pearson"
    weighted: bool = False
    baseline_subtraction_cosine: bool = True
    s1s2_min_occupancy_per_bin_s: float = 0.5
    plot_spike_shapes: bool = True
    plot_spike_shapes_overall: bool = True
    plot_spike_shapes_in_field: bool = True
    plot_spike_shapes_out_field: bool = True
    plot_PF_combined: bool = True
    include_plateau: bool = True
    plateau_state_mode: str = "all"
    plateau_include_long_cb_as_plateau: bool = False
    plateau_cb_min_duration_ms: float = 200.0
    plateau_speed_threshold: float = 3.0
    two_session_split_mode: str = "recorded_sessions"
    two_session_split_window_minutes: float | None = None
    align_to_distance_normalized_exports: bool = False
    distance_normalized_export_dir: str | None = None
    distance_normalized_min_trials_per_session: int = 4
    distance_normalized_pf_rank: int = 1
    selected_theta_vlim: tuple[float, float] | None = None
    selected_slow_vlim: tuple[float, float] | None = None
    quiet_epoch_mode: str = "strict_low_speed"  # "strict_low_speed" or "non_moving"
    stats_min_included_minutes_per_session: float = 4.0


# Globals consumed by the ported renderer function.
ENFORCE_S1S2_MIN_PEAK_RATE_FILTER = True
S1S2_MIN_PEAK_RATE_HZ = 0.5
MIN_VALID_BINS_FOR_CORR_WEIGHTED = 20
MIN_EFFECTIVE_BINS_WEIGHTED = 20
MIN_OCCUPANCY_WEIGHT = 0.5
MIN_VALID_BINS_FOR_DISTANCE = 20

_SESSION_PAYLOAD_BY_DATASET: dict[str, dict[str, Any]] = {}
SESSION_COMPARE_CACHE_SCHEMA = "session_compare_heatmaps_v10"


def _normalize_quiet_epoch_mode(mode: str) -> str:
    mode_norm = str(mode).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "strict": "strict_low_speed",
        "strict_low_speed": "strict_low_speed",
        "low_speed": "strict_low_speed",
        "speed_threshold": "strict_low_speed",
        "non_moving": "non_moving",
        "not_moving": "non_moving",
        "not_loco": "non_moving",
        "not_locomotion": "non_moving",
    }
    if mode_norm not in aliases:
        raise ValueError("quiet_epoch_mode must be one of {'strict_low_speed', 'non_moving'}.")
    return aliases[mode_norm]


def _cache_path(dataset_dir: Path, cache_version: str) -> Path:
    return dataset_dir / f"spatial_session_cache_{cache_version}.pkl"


def _valid_cache_obj(
    cache_obj: Any,
    dataset_id: str,
    cache_version: str,
    *,
    split_mode: str | None = None,
    split_window_minutes: float | None = None,
    quiet_epoch_mode: str | None = None,
    speed_threshold_quiet: float | None = None,
) -> bool:
    if not isinstance(cache_obj, dict):
        return False
    if cache_obj.get("cache_schema") != SESSION_COMPARE_CACHE_SCHEMA:
        return False
    if cache_obj.get("cache_version") != cache_version:
        return False
    if cache_obj.get("dataset_id") != dataset_id:
        return False
    if split_mode is not None:
        try:
            expected_mode = _normalize_two_session_split_mode(split_mode)
            got_mode = _normalize_two_session_split_mode(cache_obj.get("split_mode", "recorded_sessions"))
        except Exception:
            return False
        if got_mode != expected_mode:
            return False
        expected_window = _validate_two_session_split_window_minutes(expected_mode, split_window_minutes)
        got_window_raw = cache_obj.get("split_window_minutes", None)
        if expected_mode == "time_windows":
            try:
                got_window = float(got_window_raw)
            except (TypeError, ValueError):
                return False
            if not np.isclose(float(got_window), float(expected_window), rtol=0.0, atol=1e-9):
                return False
    if quiet_epoch_mode is not None:
        try:
            expected_quiet_mode = _normalize_quiet_epoch_mode(quiet_epoch_mode)
            got_quiet_mode = _normalize_quiet_epoch_mode(cache_obj.get("quiet_epoch_mode", "non_moving"))
        except Exception:
            return False
        if got_quiet_mode != expected_quiet_mode:
            return False
        if expected_quiet_mode == "strict_low_speed" and speed_threshold_quiet is not None:
            try:
                got_threshold = float(cache_obj.get("speed_threshold_quiet", np.nan))
            except (TypeError, ValueError):
                return False
            if not np.isclose(float(got_threshold), float(speed_threshold_quiet), rtol=0.0, atol=1e-9):
                return False
    if "cells" not in cache_obj or "frame_ranges" not in cache_obj or "has_session2" not in cache_obj:
        return False
    # Require minimally compatible per-cell payloads for rendering.
    cells = cache_obj.get("cells", {})
    if not isinstance(cells, dict):
        return False
    for by_cond in cells.values():
        if not isinstance(by_cond, dict):
            continue
        for cond in ("session1", "session2"):
            c = by_cond.get(cond, None)
            if not isinstance(c, dict):
                continue
            required = ("rate_map", "ss_norm_map", "cs_norm_map", "theta_map", "slow_map", "spike_shapes")
            if not all(k in c for k in required):
                return False
    return True


def _rank_reference_pf_components(
    components: Any,
    fallback_mask: np.ndarray | None,
    rate_map: np.ndarray | None = None,
    max_components: int = 2,
) -> list[np.ndarray]:
    def _safe_float(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return np.nan

    clean: list[dict[str, Any]] = []
    fallback_shape = None
    if isinstance(fallback_mask, np.ndarray) and fallback_mask.ndim == 2:
        fallback_shape = fallback_mask.shape

    if isinstance(components, (list, tuple)):
        for rank_idx, comp in enumerate(components):
            arr = None
            peak_rate = np.nan
            if isinstance(comp, dict):
                arr_raw = comp.get("mask", None)
                if arr_raw is not None:
                    arr = np.asarray(arr_raw, dtype=bool)
                peak_rate = _safe_float(comp.get("peak_rate", np.nan))
            else:
                arr = np.asarray(comp, dtype=bool)
            if arr is None or arr.ndim != 2:
                continue
            if fallback_shape is not None and arr.shape != fallback_shape:
                continue
            if not np.any(arr):
                continue
            if (not np.isfinite(peak_rate)) and isinstance(rate_map, np.ndarray) and rate_map.shape == arr.shape:
                vals = np.asarray(rate_map, dtype=float)[arr]
                if vals.size > 0 and np.any(np.isfinite(vals)):
                    peak_rate = float(np.nanmax(vals))
            clean.append(
                {
                    "mask": np.asarray(arr, dtype=bool),
                    "peak_rate": peak_rate,
                    "source_rank": int(rank_idx),
                    "area_bins": int(np.sum(arr)),
                }
            )

    if len(clean) == 0 and isinstance(fallback_mask, np.ndarray) and fallback_mask.ndim == 2 and np.any(fallback_mask):
        mask = np.asarray(fallback_mask, dtype=bool)
        if scipy_ndimage is not None:
            structure = np.ones((3, 3), dtype=int)
            labeled, n_comp = scipy_ndimage.label(mask, structure=structure)
            for comp_idx in range(1, int(n_comp) + 1):
                comp_mask = np.asarray(labeled == comp_idx, dtype=bool)
                if not np.any(comp_mask):
                    continue
                peak_rate = np.nan
                if isinstance(rate_map, np.ndarray) and rate_map.shape == mask.shape:
                    vals = np.asarray(rate_map, dtype=float)[comp_mask]
                    if vals.size > 0 and np.any(np.isfinite(vals)):
                        peak_rate = float(np.nanmax(vals))
                clean.append(
                    {
                        "mask": comp_mask,
                        "peak_rate": peak_rate,
                        "source_rank": int(comp_idx - 1),
                        "area_bins": int(np.sum(comp_mask)),
                    }
                )
        else:
            clean.append(
                {
                    "mask": mask,
                    "peak_rate": np.nan,
                    "source_rank": 0,
                    "area_bins": int(np.sum(mask)),
                }
            )

    if len(clean) == 0:
        return []

    finite_peak_items = [item for item in clean if np.isfinite(item["peak_rate"])]
    if len(finite_peak_items) > 0:
        primary = min(
            finite_peak_items,
            key=lambda item: (-float(item["peak_rate"]), -int(item["area_bins"]), int(item["source_rank"])),
        )
        primary_id = id(primary)
        remaining = [item for item in clean if id(item) != primary_id]
        remaining.sort(
            key=lambda item: (
                -int(item["area_bins"]),
                -float(item["peak_rate"]) if np.isfinite(item["peak_rate"]) else np.inf,
                int(item["source_rank"]),
            )
        )
        clean = [primary] + remaining
    else:
        clean.sort(key=lambda item: (-int(item["area_bins"]), int(item["source_rank"])))

    return [item["mask"] for item in clean[:max(0, int(max_components))]]


def _compute_session_ranges(
    merged_data: dict[str, Any],
    *,
    split_mode: str = "recorded_sessions",
    split_window_minutes: float | None = None,
) -> tuple[dict[str, tuple[int, int] | None], bool]:
    n_frames = int(len(merged_data.get("x_neural", [])))
    if n_frames <= 0 and "traces" in merged_data:
        try:
            traces = np.asarray(merged_data.get("traces"))
            if traces.ndim == 2:
                n_frames = int(traces.shape[1])
        except Exception:
            pass

    mode = _normalize_two_session_split_mode(split_mode)
    window_min = _validate_two_session_split_window_minutes(mode, split_window_minutes)
    if mode == "time_windows":
        window_sec = float(window_min) * 60.0
        frame_rate = float(merged_data.get("frame_rate", np.nan))
        if (not np.isfinite(frame_rate)) or frame_rate <= 0:
            raise ValueError("merged_data frame_rate must be a finite number > 0 for time_windows split.")
        s1_end = int(np.ceil(window_sec * frame_rate))
        s2_start = s1_end
        s2_end = int(np.ceil(2.0 * window_sec * frame_rate))

        s1_end = max(0, min(int(s1_end), n_frames))
        s2_start = max(0, min(int(s2_start), n_frames))
        s2_end = max(s2_start, min(int(s2_end), n_frames))
        has_session2 = bool(s2_end > s2_start)
        ranges = {
            "combined": (0, n_frames),
            "session1": (0, s1_end),
            "session2": (s2_start, s2_end) if has_session2 else None,
        }
        return ranges, has_session2

    starts = sorted({int(v) for v in merged_data.get("session_start_frames", [0]) if int(v) >= 0})
    if not starts:
        starts = [0]
    if starts[0] != 0:
        starts = [0] + starts
    has_session2 = len(starts) > 1 and int(starts[1]) < n_frames
    ranges = {
        "combined": (0, n_frames),
        "session1": (0, int(starts[1])) if has_session2 else (0, n_frames),
        "session2": (int(starts[1]), n_frames) if has_session2 else None,
    }
    return ranges, bool(has_session2)


def _load_distance_normalized_allowed_keys(
    config: PipelineConfig,
    params: SessionCompareParams,
) -> dict[str, set[tuple[str, int]]]:
    if not bool(params.align_to_distance_normalized_exports):
        return {}

    if params.distance_normalized_export_dir is None:
        export_root = config.figures_root / "CKII_pooled" / "pf_distance_centered_2sessions_average_exports"
    else:
        export_root = Path(params.distance_normalized_export_dir)

    pf_rank = int(params.distance_normalized_pf_rank)
    if pf_rank <= 0:
        raise ValueError("distance_normalized_pf_rank must be >= 1.")
    min_trials = int(params.distance_normalized_min_trials_per_session)
    if min_trials < 0:
        raise ValueError("distance_normalized_min_trials_per_session must be >= 0.")

    out: dict[str, set[tuple[str, int]]] = {}
    for category in ("CSplus", "CSminus"):
        csv_path = export_root / category / "session_averages_pre_post_index.csv"
        if not csv_path.exists():
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        col_s1 = f"n_trials_pf{pf_rank}_s1"
        col_s2 = f"n_trials_pf{pf_rank}_s2"
        required = {"animal_id", "cell_idx", col_s1, col_s2}
        if not required.issubset(set(df.columns)):
            continue
        sub = df[
            pd.to_numeric(df[col_s1], errors="coerce").fillna(0) >= int(min_trials)
        ]
        sub = sub[
            pd.to_numeric(sub[col_s2], errors="coerce").fillna(0) >= int(min_trials)
        ]
        allowed = {
            (str(row["animal_id"]), int(row["cell_idx"]))
            for _, row in sub[["animal_id", "cell_idx"]].drop_duplicates().iterrows()
        }
        out[category] = allowed
    return out


def _clip_spike_list(spike_list: Any, s0: int, s1: int) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    if not isinstance(spike_list, (list, tuple)):
        return out
    for sp in spike_list:
        arr = np.asarray(sp, dtype=int)
        arr = arr[(arr >= int(s0)) & (arr < int(s1))] - int(s0)
        out.append(np.asarray(arr, dtype=int))
    return out


def _clip_vec_list(vlist: Any, s0: int, s1: int) -> list[Any]:
    out: list[Any] = []
    if not isinstance(vlist, (list, tuple)):
        return out
    for v in vlist:
        if v is None:
            out.append(None)
            continue
        try:
            arr = np.asarray(v)
            out.append(arr[int(s0):int(s1)])
        except Exception:
            out.append(v)
    return out


def _clip_complex_bursts_dicts(dicts_in: Any, s0: int, s1: int) -> list[dict[str, Any]]:
    if not isinstance(dicts_in, (list, tuple)):
        return []
    out: list[dict[str, Any]] = []
    for item in dicts_in:
        if not isinstance(item, dict):
            out.append({})
            continue
        starts = np.asarray(item.get("starts", []), dtype=int)
        ends = np.asarray(item.get("ends", []), dtype=int)
        n = min(len(starts), len(ends))
        starts = starts[:n]
        ends = ends[:n]
        keep = (ends >= int(s0)) & (starts < int(s1)) if n > 0 else np.zeros((0,), dtype=bool)
        clipped: dict[str, Any] = {}
        for k, v in item.items():
            if isinstance(v, np.ndarray):
                if v.ndim == 1 and v.size == n:
                    clipped[k] = v[:n][keep]
                else:
                    clipped[k] = v
            elif isinstance(v, list):
                arr = None
                try:
                    arr = np.asarray(v)
                except Exception:
                    arr = None
                if arr is not None and arr.ndim == 1 and arr.size == n:
                    clipped[k] = arr[:n][keep]
                else:
                    clipped[k] = v
            else:
                clipped[k] = v
        clipped["starts"] = np.asarray(clipped.get("starts", []), dtype=int) - int(s0)
        clipped["ends"] = np.asarray(clipped.get("ends", []), dtype=int) - int(s0)
        out.append(clipped)
    return out


def _clip_burst_metrics_for_slice(metrics_in: Any, s0: int, s1: int, n_cells: int) -> list[Any]:
    """Clip per-cell burst metric entries to a frame slice and make frames slice-relative."""
    s0 = int(s0)
    s1 = int(s1)
    n_cells = int(n_cells)
    out: list[Any] = [None for _ in range(max(n_cells, 0))]
    if not isinstance(metrics_in, (list, tuple)) or n_cells <= 0:
        return out

    def _clip_one_cell(cell_metrics: Any) -> list[dict[str, Any]] | None:
        if not isinstance(cell_metrics, (list, tuple)):
            return None
        clipped_entries: list[dict[str, Any]] = []
        for entry in cell_metrics:
            if not isinstance(entry, dict):
                continue
            try:
                start = int(entry.get("start", -1))
            except Exception:
                continue
            if start < s0 or start >= s1:
                continue
            entry_out = dict(entry)
            entry_out["start"] = start - s0
            if "end" in entry_out:
                try:
                    entry_out["end"] = int(entry_out["end"]) - s0
                except Exception:
                    pass
            clipped_entries.append(entry_out)
        return clipped_entries

    per_cell_like = len(metrics_in) == n_cells and all(
        (item is None) or isinstance(item, (list, tuple)) for item in metrics_in
    )
    if per_cell_like:
        for idx in range(n_cells):
            out[idx] = _clip_one_cell(metrics_in[idx])
        return out

    # Fallback for a single-cell list-of-dicts shape; keep it on cell 0 only.
    if metrics_in and isinstance(metrics_in[0], dict):
        out[0] = _clip_one_cell(metrics_in)
    return out


def _slice_merged_data(loaded: dict[str, Any], s0: int, s1: int) -> dict[str, Any]:
    s0 = int(s0)
    s1 = int(s1)
    sliced: dict[str, Any] = {}

    n_cells = int(len(loaded.get("spikes", [])))

    # Frame-wise arrays.
    for k in ["speed", "ts_neural", "x_neural", "y_neural", "hd_angles_neural"]:
        if k in loaded:
            sliced[k] = np.asarray(loaded[k])[s0:s1]

    traces = np.asarray(loaded.get("traces"))
    sliced["traces"] = traces[:, s0:s1]

    for k in ["frame_rate", "frame_width", "frame_height"]:
        if k in loaded:
            sliced[k] = loaded[k]
    if loaded.get("manual_refined_source", False):
        sliced["manual_refined_source"] = True
    if "manual_refined_bad_masks" in loaded:
        masks = np.asarray(loaded["manual_refined_bad_masks"], dtype=bool)
        if masks.ndim == 2:
            sliced["manual_refined_bad_masks"] = masks[:, s0:s1]
    if "manual_refined_bad_mask_stats" in loaded:
        sliced["manual_refined_bad_mask_stats"] = loaded["manual_refined_bad_mask_stats"]

    all_spikes = _clip_spike_list(loaded.get("all_spikes", loaded.get("spikes", [])), s0, s1)
    sliced["spikes"] = all_spikes
    sliced["all_spikes"] = all_spikes
    sliced["refined_SS"] = _clip_spike_list(loaded.get("refined_SS", [np.array([], dtype=int) for _ in range(n_cells)]), s0, s1)
    sliced["all_CS_spikes"] = _clip_spike_list(loaded.get("all_CS_spikes", [np.array([], dtype=int) for _ in range(n_cells)]), s0, s1)

    for k in [
        "spike_heights_interpolated",
        "SNR_interpolated",
        "traces_SNR_interpolated",
        "Vm_SNR_interpolated",
        "plateau_traces_normalized",
        "plateau_Vm_normalized",
    ]:
        if k in loaded:
            sliced[k] = _clip_vec_list(loaded[k], s0, s1)

    sliced["complex_bursts_dicts"] = _clip_complex_bursts_dicts(loaded.get("complex_bursts_dicts", []), s0, s1)
    sliced["plateaus_dicts"] = _clip_complex_bursts_dicts(loaded.get("plateaus_dicts", []), s0, s1)
    sliced["burst_metrics"] = _clip_burst_metrics_for_slice(loaded.get("burst_metrics", []), s0, s1, n_cells)
    sliced["session_start_frames"] = [0]

    return sliced


def _analysis_to_spatial_entry_from_ctx(
    analysis: dict[str, Any],
    dataset_id: str,
    cell_idx: int,
    ctx: dict[str, Any],
    reference_pf_mask: np.ndarray | None = None,
    reference_pf_components: list[np.ndarray] | None = None,
    quiet_epoch_mode: str = "strict_low_speed",
    speed_threshold_quiet: float = 0.5,
    quiet_kernel_size: int = 51,
    quiet_min_duration_s: float = 0.25,
    quiet_merge_gap_s: float = 0.0,
) -> dict[str, Any]:
    n_frames_total = int(ctx["n_frames"])
    moving_idx = np.asarray(analysis.get("moving_indices", []), dtype=int)
    x_traj = np.asarray(analysis.get("x_traj", np.array([], dtype=float)), dtype=float)
    y_traj = np.asarray(analysis.get("y_traj", np.array([], dtype=float)), dtype=float)

    ss_spikes_x, ss_spikes_y = _get_spike_positions_on_traj(
        ctx["refined_ss"][cell_idx] if cell_idx < len(ctx["refined_ss"]) else None,
        moving_idx,
        x_traj,
        y_traj,
        n_frames_total,
    )
    cs_spikes_x, cs_spikes_y = _get_spike_positions_on_traj(
        ctx["all_cs_spikes"][cell_idx] if cell_idx < len(ctx["all_cs_spikes"]) else None,
        moving_idx,
        x_traj,
        y_traj,
        n_frames_total,
    )

    bad_mask = np.asarray(ctx["bad_masks"][cell_idx], dtype=bool)
    if bad_mask.shape[0] != n_frames_total:
        bad_mask = np.zeros(n_frames_total, dtype=bool)
    removed_total = int(np.sum(bad_mask))
    kept_total = int(n_frames_total - removed_total)
    pct_removed_total = (100.0 * removed_total / n_frames_total) if n_frames_total > 0 else np.nan
    frame_rate = float(ctx["frame_rate"])
    total_minutes = float(n_frames_total) / frame_rate / 60.0 if frame_rate > 0 else np.nan
    included_minutes = float(kept_total) / frame_rate / 60.0 if frame_rate > 0 else np.nan

    def _sizes_or_zero(value: Any) -> list[float]:
        if isinstance(value, (list, tuple, np.ndarray)) and len(value) > 0:
            return list(value)
        return [0]

    def _compute_reference_pf_inout_fields(pf_mask: np.ndarray | None, prefix: str = "") -> dict[str, float]:
        keys = [
            f"{prefix}{name}_inout_{field}"
            for name in ("all", "ss", "cs")
            for field in ("loco_in", "loco_out", "loco_ratio", "quiet_in", "quiet_out", "quiet_ratio")
        ]
        fields = {key: np.nan for key in keys}
        if pf_mask is None:
            return fields
        try:
            pf_mask_ref = np.asarray(pf_mask, dtype=bool)
        except Exception:
            return fields
        if pf_mask_ref.ndim != 2 or not np.any(pf_mask_ref):
            return fields

        valid_frames, moving_mask, quiet_mask = _reference_state_masks()
        if not np.any(valid_frames):
            return fields

        params = analysis.get("params", {}) if isinstance(analysis, dict) else {}
        width_real = float(params.get("width_real", 35.5))
        height_real = float(params.get("height_real", 20.0))
        bin_size = float(params.get("bin_size", 1.5))
        bins = [
            np.arange(0, width_real + bin_size, bin_size),
            np.arange(0, height_real + bin_size, bin_size),
        ]
        spike_sources = _reference_spike_sources()
        for name, spk in spike_sources.items():
            inout = core.compute_in_out_ratio(
                spk,
                np.asarray(ctx["x_neural"], dtype=float),
                np.asarray(ctx["y_neural"], dtype=float),
                pf_mask_ref,
                bins,
                moving_mask,
                quiet_mask,
                float(ctx["frame_rate"]),
            )
            for field, value in inout.items():
                fields[f"{prefix}{name}_inout_{field}"] = value
        return fields

    quiet_epoch_mode = _normalize_quiet_epoch_mode(quiet_epoch_mode)
    state_masks_cache: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    burst_events_cache: np.ndarray | None = None

    def _reference_state_masks() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        nonlocal state_masks_cache
        if state_masks_cache is not None:
            return state_masks_cache
        x_full = np.asarray(ctx["x_neural"], dtype=float)
        y_full = np.asarray(ctx["y_neural"], dtype=float)
        speed = np.asarray(ctx.get("speed", np.full(n_frames_total, np.nan)), dtype=float)
        valid_frames = np.isfinite(x_full) & np.isfinite(y_full) & np.isfinite(speed) & (~bad_mask)
        moving_mask = np.zeros(n_frames_total, dtype=bool)
        moving_idx_local = analysis.get("moving_indices", [])
        if moving_idx_local is not None and len(moving_idx_local) > 0:
            moving_idx_local = np.asarray(moving_idx_local, dtype=int)
            moving_idx_local = moving_idx_local[
                (moving_idx_local >= 0) & (moving_idx_local < n_frames_total)
            ]
            moving_mask[moving_idx_local] = True
        if not np.any(moving_mask):
            speed_for_epochs = speed.copy()
            speed_for_epochs[~valid_frames] = np.nan
            _, _, moving_idx_local = core._compute_moving_epochs(
                speed_for_epochs,
                float(ctx["frame_rate"]),
                kernel_size=51,
                filter_type="boxcar",
                speed_threshold=3.0,
                min_duration_s=0.25,
                merge_gap_s=0.0,
            )
            if len(moving_idx_local) > 0:
                moving_mask[np.asarray(moving_idx_local, dtype=int)] = True
        moving_mask &= valid_frames
        if quiet_epoch_mode == "strict_low_speed":
            quiet_mask = np.zeros(n_frames_total, dtype=bool)
            speed_for_epochs = speed.copy()
            speed_for_epochs[~valid_frames] = np.nan
            _, _, quiet_idx = core._compute_quiet_epochs(
                speed_for_epochs,
                float(ctx["frame_rate"]),
                kernel_size=int(quiet_kernel_size),
                filter_type="boxcar",
                speed_threshold_quiet=float(speed_threshold_quiet),
                min_duration_s=float(quiet_min_duration_s),
                merge_gap_s=float(quiet_merge_gap_s),
            )
            if len(quiet_idx) > 0:
                quiet_idx = np.asarray(quiet_idx, dtype=int)
                quiet_idx = quiet_idx[(quiet_idx >= 0) & (quiet_idx < n_frames_total)]
                quiet_mask[quiet_idx] = True
            quiet_mask &= valid_frames & (~moving_mask)
        else:
            quiet_mask = valid_frames & (~moving_mask)
        state_masks_cache = (valid_frames, moving_mask, quiet_mask)
        return state_masks_cache

    def _reference_spike_sources() -> dict[str, np.ndarray]:
        return {
            "all": ctx["spikes"][cell_idx] if cell_idx < len(ctx["spikes"]) else np.array([], dtype=int),
            "ss": ctx["refined_ss"][cell_idx] if cell_idx < len(ctx["refined_ss"]) else np.array([], dtype=int),
            "cs": ctx["all_cs_spikes"][cell_idx] if cell_idx < len(ctx["all_cs_spikes"]) else np.array([], dtype=int),
        }

    def _reference_complex_burst_events() -> np.ndarray:
        nonlocal burst_events_cache
        if burst_events_cache is not None:
            return burst_events_cache
        bursts = None
        complex_bursts_dicts = ctx.get("complex_bursts_dicts", [])
        if isinstance(complex_bursts_dicts, (list, tuple)):
            if cell_idx < len(complex_bursts_dicts):
                bursts = complex_bursts_dicts[cell_idx]
        elif isinstance(complex_bursts_dicts, dict):
            bursts = complex_bursts_dicts
        if not isinstance(bursts, dict):
            burst_events_cache = np.array([], dtype=int)
            return burst_events_cache

        starts = np.asarray(bursts.get("starts", []), dtype=int).reshape(-1)
        ends = np.asarray(bursts.get("ends", []), dtype=int).reshape(-1)
        cs_spikes = np.asarray(
            ctx["all_cs_spikes"][cell_idx] if cell_idx < len(ctx["all_cs_spikes"]) else np.array([], dtype=int),
            dtype=int,
        ).reshape(-1)
        cs_spikes = np.sort(cs_spikes[(cs_spikes >= 0) & (cs_spikes < n_frames_total)])

        burst_events: list[int] = []
        n_intervals = min(starts.size, ends.size)
        if n_intervals > 0 and cs_spikes.size > 0:
            for start_idx, end_idx in zip(starts[:n_intervals], ends[:n_intervals]):
                start_i = int(start_idx)
                end_i = int(end_idx)
                if end_i < start_i:
                    continue
                in_burst = cs_spikes[(cs_spikes >= start_i) & (cs_spikes <= end_i)]
                if in_burst.size > 0:
                    burst_events.append(int(in_burst[0]))

        if not burst_events:
            for key in ("complex_bursts", "locs", "starts"):
                fallback = np.asarray(bursts.get(key, []), dtype=int).reshape(-1)
                if fallback.size > 0:
                    burst_events = [int(v) for v in fallback]
                    break

        events = np.asarray(burst_events, dtype=int).reshape(-1)
        events = events[(events >= 0) & (events < n_frames_total)]
        burst_events_cache = np.unique(events)
        return burst_events_cache

    def _complex_burst_rate_for_mask(mask: np.ndarray, burst_events: np.ndarray) -> tuple[float, float, float]:
        frame_rate = float(ctx["frame_rate"])
        mask = np.asarray(mask, dtype=bool)
        time_s = float(np.sum(mask) / frame_rate) if frame_rate > 0 else np.nan
        count = float(np.sum(mask[burst_events])) if burst_events.size > 0 else 0.0
        rate = (count / time_s) if np.isfinite(time_s) and time_s > 0 else np.nan
        return rate, count, time_s

    def _compute_reference_cb_rate_fields(
        pf_mask: np.ndarray | None,
        in_label: str,
        out_label: str,
    ) -> dict[str, float]:
        fields = {
            f"complex_burst_event_rate_loco_{in_label}": np.nan,
            f"complex_burst_event_rate_loco_{out_label}": np.nan,
            f"complex_burst_event_rate_quiet_{in_label}": np.nan,
            f"complex_burst_event_rate_quiet_{out_label}": np.nan,
            f"complex_burst_event_count_loco_{in_label}": np.nan,
            f"complex_burst_event_count_loco_{out_label}": np.nan,
            f"complex_burst_event_count_quiet_{in_label}": np.nan,
            f"complex_burst_event_count_quiet_{out_label}": np.nan,
            f"complex_burst_time_loco_{in_label}": np.nan,
            f"complex_burst_time_loco_{out_label}": np.nan,
            f"complex_burst_time_quiet_{in_label}": np.nan,
            f"complex_burst_time_quiet_{out_label}": np.nan,
        }
        if pf_mask is None:
            return fields
        try:
            pf_mask_ref = np.asarray(pf_mask, dtype=bool)
        except Exception:
            return fields
        if pf_mask_ref.ndim != 2 or not np.any(pf_mask_ref):
            return fields

        valid_frames, moving_mask, quiet_mask = _reference_state_masks()
        if not np.any(valid_frames):
            return fields

        params = analysis.get("params", {}) if isinstance(analysis, dict) else {}
        width_real = float(params.get("width_real", 35.5))
        height_real = float(params.get("height_real", 20.0))
        bin_size = float(params.get("bin_size", 1.5))
        bins = [
            np.arange(0, width_real + bin_size, bin_size),
            np.arange(0, height_real + bin_size, bin_size),
        ]
        inside_pf = core.positions_in_place_field(
            np.asarray(ctx["x_neural"], dtype=float),
            np.asarray(ctx["y_neural"], dtype=float),
            bins,
            pf_mask_ref,
        )
        burst_events = _reference_complex_burst_events()
        masks = {
            ("loco", in_label): moving_mask & inside_pf & valid_frames,
            ("loco", out_label): moving_mask & (~inside_pf) & valid_frames,
            ("quiet", in_label): quiet_mask & inside_pf & valid_frames,
            ("quiet", out_label): quiet_mask & (~inside_pf) & valid_frames,
        }
        for (state, label), mask in masks.items():
            rate, count, time_s = _complex_burst_rate_for_mask(mask, burst_events)
            fields[f"complex_burst_event_rate_{state}_{label}"] = rate
            fields[f"complex_burst_event_count_{state}_{label}"] = count
            fields[f"complex_burst_time_{state}_{label}"] = time_s
        return fields

    def _reference_cb_rate_fields() -> dict[str, float]:
        valid_frames, moving_mask, quiet_mask = _reference_state_masks()
        burst_events = _reference_complex_burst_events()
        fields: dict[str, float] = {}
        for state, mask in (("loco", moving_mask & valid_frames), ("quiet", quiet_mask & valid_frames)):
            rate, count, time_s = _complex_burst_rate_for_mask(mask, burst_events)
            fields[f"complex_burst_event_rate_{state}"] = rate
            fields[f"complex_burst_event_count_{state}"] = count
            fields[f"complex_burst_time_{state}"] = time_s

        fields.update(_compute_reference_cb_rate_fields(reference_pf_mask, "in_pf", "out_pf"))
        components = reference_pf_components if isinstance(reference_pf_components, (list, tuple)) else []
        for pf_rank in (1, 2):
            comp_mask = components[pf_rank - 1] if len(components) >= pf_rank else None
            fields.update(_compute_reference_cb_rate_fields(comp_mask, f"in_pf{pf_rank}", f"out_pf{pf_rank}"))
        return fields

    def _reference_quiet_rate_fields() -> dict[str, float]:
        _, _, quiet_mask = _reference_state_masks()
        quiet_time_s = float(np.sum(quiet_mask) / float(ctx["frame_rate"]))
        fields = {f"{name}_rate_quiet": np.nan for name in ("all", "ss", "cs")}
        for name, spk in _reference_spike_sources().items():
            spk = np.asarray(spk, dtype=int)
            spk = spk[(spk >= 0) & (spk < n_frames_total)]
            n_quiet = int(np.sum(quiet_mask[spk])) if spk.size > 0 else 0
            fields[f"{name}_rate_quiet"] = n_quiet / quiet_time_s if quiet_time_s > 0 else np.nan
        return fields

    def _reference_loco_rate_fields() -> dict[str, float]:
        _, moving_mask, _ = _reference_state_masks()
        loco_time_s = float(np.sum(moving_mask) / float(ctx["frame_rate"]))
        fields = {f"{name}_rate_loco": np.nan for name in ("all", "ss", "cs")}
        for name, spk in _reference_spike_sources().items():
            spk = np.asarray(spk, dtype=int)
            spk = spk[(spk >= 0) & (spk < n_frames_total)]
            n_loco = int(np.sum(moving_mask[spk])) if spk.size > 0 else 0
            fields[f"{name}_rate_loco"] = n_loco / loco_time_s if loco_time_s > 0 else np.nan
        return fields

    def _reference_pf_inout_fields() -> dict[str, float]:
        fields = _compute_reference_pf_inout_fields(reference_pf_mask)
        components = reference_pf_components if isinstance(reference_pf_components, (list, tuple)) else []
        for pf_rank in (1, 2):
            comp_mask = components[pf_rank - 1] if len(components) >= pf_rank else None
            fields.update(_compute_reference_pf_inout_fields(comp_mask, prefix=f"pf{pf_rank}_"))
        fields["n_combined_reference_pfs"] = int(len(components))
        return fields

    out = {
        "animal_id": dataset_id,
        "session": dataset_id,
        "cell_idx": int(cell_idx),
        "is_place_cell": bool(analysis.get("is_place_cell", False)),
        "is_place_cell_ss": bool(analysis.get("is_place_cell_ss", False)),
        "is_place_cell_cs": bool(analysis.get("is_place_cell_cs", False)),
        "si": analysis.get("si", np.nan),
        "p_value": analysis.get("p_value", np.nan),
        "ss_si": analysis.get("ss_si", np.nan),
        "ss_p_value": analysis.get("ss_p_value", np.nan),
        "cs_si": analysis.get("cs_si", np.nan),
        "cs_p_value": analysis.get("cs_p_value", np.nan),
        "peak_rate": analysis.get("peak_rate", np.nan),
        "ss_peak_rate": analysis.get("ss_peak_rate", np.nan),
        "cs_peak_rate": analysis.get("cs_peak_rate", np.nan),
        "peak_rate_all": analysis.get("peak_rate", np.nan),
        "peak_rate_ss": analysis.get("ss_peak_rate", np.nan),
        "peak_rate_cs": analysis.get("cs_peak_rate", np.nan),
        "rate_map": analysis.get("rate_map", None),
        "ss_rate_map": analysis.get("ss_rate_map", None),
        "cs_rate_map": analysis.get("cs_rate_map", None),
        "ss_norm_map": analysis.get("ss_norm_map", None),
        "cs_norm_map": analysis.get("cs_norm_map", None),
        "occupancy": analysis.get("occupancy", None),
        "place_field_mask": analysis.get("place_field_mask", None),
        "place_field_components": analysis.get("place_field_components", []),
        "ss_place_field_mask": analysis.get("ss_place_field_mask", None),
        "cs_place_field_mask": analysis.get("cs_place_field_mask", None),
        "pf_sizes": analysis.get("pf_sizes", []),
        "ss_pf_sizes": analysis.get("ss_pf_sizes", []),
        "cs_pf_sizes": analysis.get("cs_pf_sizes", []),
        "place_field_sizes_cm2": _sizes_or_zero(analysis.get("pf_sizes", [])),
        "place_field_sizes_cm2_ss": _sizes_or_zero(analysis.get("ss_pf_sizes", [])),
        "place_field_sizes_cm2_cs": _sizes_or_zero(analysis.get("cs_pf_sizes", [])),
        "n_place_fields": analysis.get("n_place_fields", 0),
        "n_ss_place_fields": analysis.get("n_ss_place_fields", 0),
        "n_cs_place_fields": analysis.get("n_cs_place_fields", 0),
        "n_place_fields_ss": analysis.get("n_ss_place_fields", 0),
        "n_place_fields_cs": analysis.get("n_cs_place_fields", 0),
        "field_area": analysis.get("field_area", np.nan),
        "ss_field_area": analysis.get("ss_field_area", np.nan),
        "cs_field_area": analysis.get("cs_field_area", np.nan),
        "total_pf_size_cm2": analysis.get("field_area", np.nan),
        "total_pf_size_cm2_ss": analysis.get("ss_field_area", np.nan),
        "total_pf_size_cm2_cs": analysis.get("cs_field_area", np.nan),
        "si_bits_per_spike": analysis.get("si_bits_per_spike", np.nan),
        "si_bits_per_spike_ss": analysis.get("si_bits_per_spike_ss", np.nan),
        "si_bits_per_spike_cs": analysis.get("si_bits_per_spike_cs", np.nan),
        "coherence_all": analysis.get("coherence_all", np.nan),
        "coherence_ss": analysis.get("coherence_ss", np.nan),
        "coherence_cs": analysis.get("coherence_cs", np.nan),
        "sparsity_all": analysis.get("sparsity_all", np.nan),
        "sparsity_ss": analysis.get("sparsity_ss", np.nan),
        "sparsity_cs": analysis.get("sparsity_cs", np.nan),
        "x_traj": x_traj,
        "y_traj": y_traj,
        "spikes_x": analysis.get("spikes_x", np.array([])),
        "spikes_y": analysis.get("spikes_y", np.array([])),
        "ss_spikes_x": ss_spikes_x,
        "ss_spikes_y": ss_spikes_y,
        "cs_spikes_x": cs_spikes_x,
        "cs_spikes_y": cs_spikes_y,
        "theta_corr_all": analysis.get("theta_corr_all", np.nan),
        "theta_corr_ss": analysis.get("theta_corr_ss", np.nan),
        "theta_corr_cs": analysis.get("theta_corr_cs", np.nan),
        "slow_corr_all": analysis.get("slow_corr_all", np.nan),
        "slow_corr_ss": analysis.get("slow_corr_ss", np.nan),
        "slow_corr_cs": analysis.get("slow_corr_cs", np.nan),
        "theta_map": analysis.get("theta_map", None),
        "slow_map": analysis.get("slow_map", None),
        "spike_shapes": analysis.get("spike_shapes", None),
        "burst_metrics": analysis.get("burst_metrics", None),
        "spike_burst_rate_metrics": analysis.get("spike_burst_rate_metrics", None),
        "params": analysis.get("params", {}),
        "frame_rate": frame_rate,
        "n_frames_total": n_frames_total,
        "n_frames_kept_total": kept_total,
        "total_minutes": total_minutes,
        "included_minutes": included_minutes,
        "n_removed_frames_total": removed_total,
        "pct_removed_frames_total": pct_removed_total,
        "quiet_epoch_mode": quiet_epoch_mode,
        "speed_threshold_quiet": float(speed_threshold_quiet),
    }
    out.update(_reference_pf_inout_fields())
    out.update(_reference_cb_rate_fields())
    out.update(_reference_loco_rate_fields())
    out.update(_reference_quiet_rate_fields())
    out["fr_loco_all_allpf_all"] = out.get("all_rate_loco", np.nan)
    out["fr_loco_all_allpf_ss"] = out.get("ss_rate_loco", np.nan)
    out["fr_loco_all_allpf_cs"] = out.get("cs_rate_loco", np.nan)
    out["fr_in_allpf_all"] = out.get("all_inout_loco_in", np.nan)
    out["fr_in_allpf_ss"] = out.get("ss_inout_loco_in", np.nan)
    out["fr_in_allpf_cs"] = out.get("cs_inout_loco_in", np.nan)
    out["fr_out_allpf_all"] = out.get("all_inout_loco_out", np.nan)
    out["fr_out_allpf_ss"] = out.get("ss_inout_loco_out", np.nan)
    out["fr_out_allpf_cs"] = out.get("cs_inout_loco_out", np.nan)
    out["fr_quiet_all_allpf_all"] = out.get("all_rate_quiet", np.nan)
    out["fr_quiet_all_allpf_ss"] = out.get("ss_rate_quiet", np.nan)
    out["fr_quiet_all_allpf_cs"] = out.get("cs_rate_quiet", np.nan)
    out["fr_quiet_in_allpf_all"] = out.get("all_inout_quiet_in", np.nan)
    out["fr_quiet_in_allpf_ss"] = out.get("ss_inout_quiet_in", np.nan)
    out["fr_quiet_in_allpf_cs"] = out.get("cs_inout_quiet_in", np.nan)
    out["fr_quiet_out_allpf_all"] = out.get("all_inout_quiet_out", np.nan)
    out["fr_quiet_out_allpf_ss"] = out.get("ss_inout_quiet_out", np.nan)
    out["fr_quiet_out_allpf_cs"] = out.get("cs_inout_quiet_out", np.nan)
    for pf_rank in (1, 2):
        out[f"fr_in_pf{pf_rank}_all"] = out.get(f"pf{pf_rank}_all_inout_loco_in", np.nan)
        out[f"fr_in_pf{pf_rank}_ss"] = out.get(f"pf{pf_rank}_ss_inout_loco_in", np.nan)
        out[f"fr_in_pf{pf_rank}_cs"] = out.get(f"pf{pf_rank}_cs_inout_loco_in", np.nan)
    return out


def _run_spatial_for_slice(
    sliced: dict[str, Any],
    dataset_id: str,
    config: PipelineConfig,
    eligible_cell_indices: set[int] | None = None,
    reference_pf_masks: dict[int, np.ndarray] | None = None,
    reference_pf_components: dict[int, list[np.ndarray]] | None = None,
    quiet_epoch_mode: str = "strict_low_speed",
) -> list[dict[str, Any]]:
    slice_config = copy.deepcopy(config)
    slice_config.analysis.min_good_minutes = 0.0
    ctx = _prepare_native_analysis_context(sliced, slice_config)
    if eligible_cell_indices is not None:
        eligible_mask = np.zeros(int(ctx["n_cells"]), dtype=bool)
        for idx in eligible_cell_indices:
            if 0 <= int(idx) < int(ctx["n_cells"]):
                eligible_mask[int(idx)] = True
        ctx["eligible_cells"] = np.asarray(ctx["eligible_cells"], dtype=bool) & eligible_mask
    outputs = _run_place_cell_analysis_native(dataset_id, slice_config, ctx)

    out_cells: list[dict[str, Any]] = []
    for cell_idx, analysis in enumerate(outputs):
        if not bool(ctx["eligible_cells"][cell_idx]):
            continue
        if not isinstance(analysis, dict):
            continue
        ref_pf_mask = None
        if isinstance(reference_pf_masks, dict):
            ref_pf_mask = reference_pf_masks.get(int(cell_idx), None)
        ref_pf_components = None
        if isinstance(reference_pf_components, dict):
            ref_pf_components = reference_pf_components.get(int(cell_idx), None)
        out_cells.append(
            _analysis_to_spatial_entry_from_ctx(
                analysis,
                dataset_id,
                cell_idx,
                ctx,
                reference_pf_mask=ref_pf_mask,
                reference_pf_components=ref_pf_components,
                quiet_epoch_mode=quiet_epoch_mode,
                speed_threshold_quiet=float(slice_config.analysis.speed_threshold_quiet),
                quiet_kernel_size=int(slice_config.analysis.kernel_size),
                quiet_min_duration_s=float(slice_config.analysis.min_duration_s),
                quiet_merge_gap_s=float(slice_config.analysis.merge_gap_s),
            )
        )
    return out_cells


def _count_cb_run_in(cell: dict[str, Any]) -> int:
    if not isinstance(cell, dict):
        return 0
    spike_shapes = cell.get("spike_shapes")
    if isinstance(spike_shapes, dict) and "complex" in spike_shapes:
        shapes = spike_shapes.get("complex", {}).get("shapes", {})
        return int(len(shapes.get("run_in", [])))
    return 0


def _split_session_plc_label(
    cell: Any,
    *,
    cb_num_threshold: int,
    cs_peak_rate_threshold: float,
    cs_plc_definition_mode: str = "legacy",
) -> str:
    if not isinstance(cell, dict) or bool(cell.get("is_na_panel", False)):
        return "missing"

    is_place_cell = bool(cell.get("is_place_cell", False))
    try:
        cs_peak_rate = float(cell.get("cs_peak_rate", np.nan))
    except (TypeError, ValueError):
        cs_peak_rate = np.nan

    is_csplus = is_csplus_place_cell(
        is_place_cell=is_place_cell,
        n_cb_in_pf=_count_cb_run_in(cell),
        cs_peak_rate=cs_peak_rate,
        cb_num_threshold=int(cb_num_threshold),
        cs_peak_rate_threshold=float(cs_peak_rate_threshold),
        has_cs_place_field=cell_has_cs_place_field(cell),
        cs_plc_definition_mode=cs_plc_definition_mode,
    )

    if is_csplus:
        return "CS+ PLC"
    if is_place_cell:
        return "CS- PLC"
    return "Non-PLC"


def _normalize_map_to_own_peak(rate_map: Any) -> Any:
    if not isinstance(rate_map, np.ndarray):
        return None
    if rate_map.size == 0 or (not np.any(np.isfinite(rate_map))):
        return np.full_like(rate_map, np.nan, dtype=float)
    peak = float(np.nanmax(rate_map))
    if (not np.isfinite(peak)) or peak <= 0:
        out = np.array(rate_map, dtype=float, copy=True)
        out[np.isfinite(out)] = 0.0
        return out
    return np.asarray(rate_map, dtype=float) / peak


def _renormalize_ss_cs_maps_per_condition(cell: dict[str, Any]) -> None:
    ss_norm = _normalize_map_to_own_peak(cell.get("ss_rate_map", None))
    cs_norm = _normalize_map_to_own_peak(cell.get("cs_rate_map", None))
    if ss_norm is not None:
        cell["ss_norm_map"] = ss_norm
    if cs_norm is not None:
        cell["cs_norm_map"] = cs_norm


def _make_na_cell(ref_cell: dict[str, Any], condition_label: str) -> dict[str, Any]:
    cell = copy.deepcopy(ref_cell)
    cell["condition_label"] = condition_label + " (N/A)"
    cell["is_na_panel"] = True

    map_keys = ["rate_map", "ss_norm_map", "cs_norm_map", "theta_map", "slow_map", "occupancy"]
    mask_keys = ["place_field_mask", "ss_place_field_mask", "cs_place_field_mask"]

    shape = None
    for mk in ["rate_map", "ss_norm_map", "cs_norm_map", "theta_map", "slow_map"]:
        m = cell.get(mk, None)
        if isinstance(m, np.ndarray) and m.ndim == 2:
            shape = m.shape
            break
    if shape is None:
        shape = (24, 14)

    for mk in map_keys:
        cell[mk] = np.full(shape, np.nan, dtype=float)
    for mk in mask_keys:
        cell[mk] = np.zeros(shape, dtype=bool)

    cell["x_traj"] = np.array([])
    cell["y_traj"] = np.array([])
    cell["spikes_x"] = np.array([])
    cell["spikes_y"] = np.array([])
    cell["ss_spikes_x"] = np.array([])
    cell["ss_spikes_y"] = np.array([])
    cell["cs_spikes_x"] = np.array([])
    cell["cs_spikes_y"] = np.array([])
    cell["peak_rate"] = np.nan
    cell["spike_shapes"] = None
    return cell


def _extract_dataset_occupancy_pair(payload: dict[str, Any]) -> tuple[np.ndarray | None, np.ndarray | None]:
    cells = payload.get("cells", {})
    if not isinstance(cells, dict) or len(cells) == 0:
        return None, None

    occ_s1 = None
    occ_s2 = None
    for _cell_idx, by_cond in cells.items():
        if not isinstance(by_cond, dict):
            continue
        if occ_s1 is None:
            c1 = by_cond.get("session1", None)
            if isinstance(c1, dict):
                m1 = c1.get("occupancy", None)
                if isinstance(m1, np.ndarray) and m1.ndim == 2:
                    occ_s1 = np.asarray(m1, dtype=float)
        if occ_s2 is None:
            c2 = by_cond.get("session2", None)
            if isinstance(c2, dict):
                m2 = c2.get("occupancy", None)
                if isinstance(m2, np.ndarray) and m2.ndim == 2:
                    occ_s2 = np.asarray(m2, dtype=float)
        if occ_s1 is not None and occ_s2 is not None:
            break
    return occ_s1, occ_s2


def _spearman_nan_safe(m1: Any, m2: Any, min_valid_bins: int = 20) -> tuple[float, int]:
    if not isinstance(m1, np.ndarray) or not isinstance(m2, np.ndarray):
        return np.nan, 0
    if m1.shape != m2.shape:
        return np.nan, 0
    a = np.asarray(m1, dtype=float).ravel()
    b = np.asarray(m2, dtype=float).ravel()
    valid = np.isfinite(a) & np.isfinite(b)
    n = int(np.sum(valid))
    if n < int(min_valid_bins):
        return np.nan, n
    x = a[valid]
    y = b[valid]
    if np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return np.nan, n
    if scipy_stats is not None:
        try:
            rho, _ = scipy_stats.spearmanr(x, y)
            return (float(rho) if np.isfinite(rho) else np.nan), n
        except Exception:
            return np.nan, n
    rx = pd.Series(x).rank(method="average").to_numpy(dtype=float)
    ry = pd.Series(y).rank(method="average").to_numpy(dtype=float)
    if np.nanstd(rx) == 0 or np.nanstd(ry) == 0:
        return np.nan, n
    rho = float(np.corrcoef(rx, ry)[0, 1])
    return (rho if np.isfinite(rho) else np.nan), n


def _get_condition_cell(dataset_id: str, cell_idx: int, cond: str) -> dict[str, Any] | None:
    payload = _SESSION_PAYLOAD_BY_DATASET.get(str(dataset_id), None)
    if payload is None:
        return None
    by_cell = payload.get("cells", {})
    by_cond = by_cell.get(int(cell_idx), {})
    if not isinstance(by_cond, dict):
        return None
    c = by_cond.get(cond, None)
    return c if isinstance(c, dict) else None


def _nan_peak_from_map(map_arr: Any) -> float:
    if not isinstance(map_arr, np.ndarray):
        return np.nan
    if map_arr.size == 0 or (not np.any(np.isfinite(map_arr))):
        return np.nan
    return float(np.nanmax(map_arr))


def _session_peak_for_map(cell: dict[str, Any], map_key: str = "rate_map") -> float:
    if not isinstance(cell, dict):
        return np.nan
    if map_key == "ss_norm_map":
        try:
            v = float(cell.get("ss_peak_rate", np.nan))
            if np.isfinite(v):
                return v
        except Exception:
            pass
        return _nan_peak_from_map(cell.get("ss_rate_map", None))
    if map_key == "cs_norm_map":
        try:
            v = float(cell.get("cs_peak_rate", np.nan))
            if np.isfinite(v):
                return v
        except Exception:
            pass
        return _nan_peak_from_map(cell.get("cs_rate_map", None))
    try:
        v = float(cell.get("peak_rate", np.nan))
        if np.isfinite(v):
            return v
    except Exception:
        pass
    return _nan_peak_from_map(cell.get("rate_map", None))


def _passes_s1s2_peak_threshold(dataset_id: str, cell_idx: int, map_key: str = "rate_map", threshold: float | None = None) -> bool:
    if threshold is None:
        threshold = float(S1S2_MIN_PEAK_RATE_HZ)
    c1 = _get_condition_cell(dataset_id, int(cell_idx), "session1")
    c2 = _get_condition_cell(dataset_id, int(cell_idx), "session2")
    if c1 is None or c2 is None:
        return False
    p1 = _session_peak_for_map(c1, map_key=map_key)
    p2 = _session_peak_for_map(c2, map_key=map_key)
    finite_peaks = np.asarray([p1, p2], dtype=float)
    finite_peaks = finite_peaks[np.isfinite(finite_peaks)]
    return bool(finite_peaks.size > 0 and np.nanmax(finite_peaks) >= float(threshold))


def build_session_compare_payloads(config: PipelineConfig, params: SessionCompareParams) -> dict[str, Any]:
    quiet_epoch_mode = _normalize_quiet_epoch_mode(params.quiet_epoch_mode)
    dataset_registry: list[dict[str, Any]] = []
    for dataset_id in config.animals:
        dataset_dir = config.data_root / dataset_id
        spatial_path = dataset_dir / "spatial_analysis_full.pkl"
        try:
            merged_path = _resolve_merged_data_path(dataset_dir, config)
        except FileNotFoundError:
            merged_path = dataset_dir / str(config.merged_data_filename)

        has_required = spatial_path.exists() and merged_path.exists()
        if not has_required:
            continue

        merged_tmp = _load_merged_data(dataset_dir, config)
        merged_tmp = _clean_behavior_speed_outliers_for_cache(
            merged_tmp,
            config,
            animal_id=dataset_id,
        )
        frame_ranges, has_session2 = _compute_session_ranges(
            merged_tmp,
            split_mode=params.two_session_split_mode,
            split_window_minutes=params.two_session_split_window_minutes,
        )

        dataset_registry.append(
            {
                "dataset_id": dataset_id,
                "dataset_dir": dataset_dir,
                "spatial_path": spatial_path,
                "merged_path": merged_path,
                "has_required_outputs": has_required,
                "has_session2": has_session2,
                "frame_ranges": frame_ranges,
            }
        )

    session_payload_by_dataset: dict[str, dict[str, Any]] = {}

    for ds in dataset_registry:
        dataset_id = ds["dataset_id"]
        dataset_dir = ds["dataset_dir"]
        cache_path = _cache_path(dataset_dir, params.cache_version)

        cache_obj = None
        if (not params.rebuild_cache) and cache_path.exists():
            try:
                with cache_path.open("rb") as f:
                    loaded_cache = pickle.load(f)
                cache_mtime = float(cache_path.stat().st_mtime)
                spatial_mtime = float(ds["spatial_path"].stat().st_mtime) if ds["spatial_path"].exists() else -np.inf
                merged_mtime = float(ds["merged_path"].stat().st_mtime) if ds["merged_path"].exists() else -np.inf
                is_fresh = cache_mtime >= max(spatial_mtime, merged_mtime)
                is_valid = _valid_cache_obj(
                    loaded_cache,
                    dataset_id,
                    params.cache_version,
                    split_mode=params.two_session_split_mode,
                    split_window_minutes=params.two_session_split_window_minutes,
                    quiet_epoch_mode=quiet_epoch_mode,
                    speed_threshold_quiet=float(config.analysis.speed_threshold_quiet),
                )
                cached_merged_path = str(loaded_cache.get("generated_from", {}).get("merged_path", ""))
                same_source = cached_merged_path == str(ds["merged_path"])
                if is_valid and is_fresh and same_source:
                    cache_obj = loaded_cache
                    print(f"Loaded session-compare cache: {cache_path}")
                elif is_valid and not same_source:
                    print(
                        f"Session-compare cache source changed for {dataset_id}; "
                        "rebuilding because merged-data filename changed."
                    )
                elif is_valid and (not is_fresh):
                    print(
                        f"Session-compare cache is stale for {dataset_id}; "
                        "rebuilding because source data changed."
                    )
            except Exception as exc:
                print(f"Cache load failed for {dataset_id}: {exc}")

        if cache_obj is None:
            merged_data = _load_merged_data(dataset_dir, config)
            merged_data = _clean_behavior_speed_outliers_for_cache(
                merged_data,
                config,
                animal_id=dataset_id,
            )
            ranges, has_session2 = _compute_session_ranges(
                merged_data,
                split_mode=params.two_session_split_mode,
                split_window_minutes=params.two_session_split_window_minutes,
            )

            cache_obj = {
                "cache_schema": SESSION_COMPARE_CACHE_SCHEMA,
                "cache_version": params.cache_version,
                "dataset_id": dataset_id,
                "split_mode": _normalize_two_session_split_mode(params.two_session_split_mode),
                "split_window_minutes": _validate_two_session_split_window_minutes(
                    params.two_session_split_mode,
                    params.two_session_split_window_minutes,
                ),
                "quiet_epoch_mode": quiet_epoch_mode,
                "speed_threshold_quiet": float(config.analysis.speed_threshold_quiet),
                "has_session2": bool(has_session2),
                "frame_ranges": {
                    "session1": list(ranges["session1"]) if ranges["session1"] is not None else None,
                    "session2": list(ranges["session2"]) if ranges["session2"] is not None else None,
                },
                "cells": {},
                "generated_from": {
                    "merged_path": str(ds["merged_path"]),
                    "behavior_speed_outlier_cleaning_stats": merged_data.get(
                        "behavior_speed_outlier_cleaning_stats",
                        None,
                    ),
                },
            }

            with ds["spatial_path"].open("rb") as f:
                combined_cells = pickle.load(f)
            combined_cell_indices: set[int] = set()
            combined_pf_masks: dict[int, np.ndarray] = {}
            combined_pf_components: dict[int, list[np.ndarray]] = {}
            for c in combined_cells:
                if not isinstance(c, dict) or "cell_idx" not in c:
                    continue
                c2 = copy.deepcopy(c)
                c2["session"] = dataset_id
                c2["animal_id"] = dataset_id
                idx = int(c2["cell_idx"])
                combined_cell_indices.add(idx)
                pf_mask = c2.get("place_field_mask", None)
                if isinstance(pf_mask, np.ndarray) and pf_mask.ndim == 2 and np.any(pf_mask):
                    combined_pf_masks[idx] = np.asarray(pf_mask, dtype=bool)
                ranked_components = _rank_reference_pf_components(
                    c2.get("place_field_components", []),
                    combined_pf_masks.get(idx, None),
                    c2.get("rate_map", None),
                    max_components=2,
                )
                if len(ranked_components) > 0:
                    combined_pf_components[idx] = ranked_components
                cache_obj["cells"].setdefault(idx, {})["combined"] = c2

            s1_0, s1_1 = ranges["session1"]
            sliced_s1 = _slice_merged_data(merged_data, s1_0, s1_1)
            session1_cells = _run_spatial_for_slice(
                sliced_s1,
                dataset_id,
                config,
                eligible_cell_indices=combined_cell_indices,
                reference_pf_masks=combined_pf_masks,
                reference_pf_components=combined_pf_components,
                quiet_epoch_mode=quiet_epoch_mode,
            )
            for c in session1_cells:
                idx = int(c["cell_idx"])
                cache_obj["cells"].setdefault(idx, {})["session1"] = c

            if has_session2 and ranges["session2"] is not None:
                s2_0, s2_1 = ranges["session2"]
                sliced_s2 = _slice_merged_data(merged_data, s2_0, s2_1)
                session2_cells = _run_spatial_for_slice(
                    sliced_s2,
                    dataset_id,
                    config,
                    eligible_cell_indices=combined_cell_indices,
                    reference_pf_masks=combined_pf_masks,
                    reference_pf_components=combined_pf_components,
                    quiet_epoch_mode=quiet_epoch_mode,
                )
                for c in session2_cells:
                    idx = int(c["cell_idx"])
                    cache_obj["cells"].setdefault(idx, {})["session2"] = c

            with cache_path.open("wb") as f:
                pickle.dump(cache_obj, f)
            print(f"Saved session-compare cache: {cache_path}")

        session_payload_by_dataset[dataset_id] = cache_obj

    return {
        "dataset_registry": dataset_registry,
        "session_payload_by_dataset": session_payload_by_dataset,
    }


def assemble_session_compare_groups(
    config: PipelineConfig,
    params: SessionCompareParams,
    payloads: dict[str, Any],
    spatial_data: Any | None = None,
) -> dict[str, Any]:
    session_payload_by_dataset = dict(payloads.get("session_payload_by_dataset", {}))

    combined_cells: list[dict[str, Any]] = []
    for dataset_id, payload in session_payload_by_dataset.items():
        by_cell = payload.get("cells", {})
        for cell_idx, by_cond in by_cell.items():
            if not isinstance(by_cond, dict):
                continue
            c = by_cond.get("combined", None)
            if isinstance(c, dict):
                c2 = copy.deepcopy(c)
                c2["session"] = dataset_id
                c2["animal_id"] = dataset_id
                combined_cells.append(c2)

    def _keys_from_cells(cells: Any) -> list[tuple[str, int]]:
        out: list[tuple[str, int]] = []
        if not isinstance(cells, (list, tuple)):
            return out
        for c in cells:
            if not isinstance(c, dict):
                continue
            try:
                key = (str(c.get("session", c.get("animal_id", ""))), int(c.get("cell_idx", -1)))
            except Exception:
                continue
            if key[0] and key[1] >= 0:
                out.append(key)
        return sorted(set(out))

    category_source = "session_compare_combined_cache"
    if spatial_data is not None:
        csplus_keys = _keys_from_cells(getattr(spatial_data, "plcs_csplus", []))
        csminus_keys = _keys_from_cells(getattr(spatial_data, "plcs_csminus", []))
        nonpc_keys = _keys_from_cells(getattr(spatial_data, "non_plcs", []))
        category_source = "spatial_data"
    else:
        cell_labels: dict[tuple[str, int], str] = {}
        for c in combined_cells:
            key = (str(c.get("session", "")), int(c.get("cell_idx", -1)))
            if key[1] < 0:
                continue
            is_pc = bool(c.get("is_place_cell", False))
            if is_pc:
                n_cb = _count_cb_run_in(c)
                is_csplus = is_csplus_place_cell(
                    is_place_cell=True,
                    n_cb_in_pf=int(n_cb),
                    cs_peak_rate=float(c.get("cs_peak_rate", np.nan)),
                    cb_num_threshold=int(config.pooled.cb_num_threshold),
                    cs_peak_rate_threshold=float(config.pooled.cs_peak_rate_threshold),
                    has_cs_place_field=cell_has_cs_place_field(c),
                    cs_plc_definition_mode=str(config.pooled.cs_plc_definition_mode),
                )
                cell_labels[key] = "csplus" if is_csplus else "csminus"

        csplus_keys = sorted([k for k, v in cell_labels.items() if v == "csplus"])
        csminus_keys = sorted([k for k, v in cell_labels.items() if v == "csminus"])
        nonpc_keys = sorted([
            (str(c.get("session", "")), int(c.get("cell_idx", -1)))
            for c in combined_cells
            if not bool(c.get("is_place_cell", False)) and int(c.get("cell_idx", -1)) >= 0
        ])

    category_counts_before_filters = {
        "CSplus": int(len(csplus_keys)),
        "CSminus": int(len(csminus_keys)),
        "non-PLC": int(len(nonpc_keys)),
    }

    aligned_key_sets = _load_distance_normalized_allowed_keys(config, params)
    if len(aligned_key_sets) > 0:
        csplus_allowed = aligned_key_sets.get("CSplus", None)
        csminus_allowed = aligned_key_sets.get("CSminus", None)
        if isinstance(csplus_allowed, set):
            csplus_keys = sorted([k for k in csplus_keys if k in csplus_allowed])
        if isinstance(csminus_allowed, set):
            csminus_keys = sorted([k for k in csminus_keys if k in csminus_allowed])

    # Dataset occupancy filter (Session1 vs Session2 Spearman).
    occupancy_rows: list[dict[str, Any]] = []
    for dataset_id, payload in session_payload_by_dataset.items():
        occ1, occ2 = _extract_dataset_occupancy_pair(payload)
        if isinstance(occ1, np.ndarray) and isinstance(occ2, np.ndarray):
            sim, n_valid = _spearman_nan_safe(occ1, occ2, min_valid_bins=20)
        else:
            sim, n_valid = np.nan, 0
        occupancy_rows.append(
            {
                "dataset_id": str(dataset_id),
                "occupancy_spearman": sim,
                "n_valid_bins": int(n_valid),
            }
        )
    occupancy_df = pd.DataFrame(occupancy_rows)

    all_dataset_ids = set(str(k) for k in session_payload_by_dataset.keys())
    if params.apply_occupancy_dataset_filter:
        allowed_dataset_ids = set(
            occupancy_df[
                np.isfinite(occupancy_df["occupancy_spearman"])
                & (occupancy_df["occupancy_spearman"] > float(params.occupancy_spearman_threshold))
            ]["dataset_id"].astype(str).tolist()
        )
    else:
        allowed_dataset_ids = set(all_dataset_ids)

    excluded_dataset_ids = sorted(all_dataset_ids - allowed_dataset_ids)
    if params.apply_occupancy_dataset_filter:
        session_payload_by_dataset = {
            k: v for k, v in session_payload_by_dataset.items() if str(k) in allowed_dataset_ids
        }

    def _filter_keys(keys_in: list[tuple[str, int]]) -> list[tuple[str, int]]:
        return sorted([k for k in keys_in if str(k[0]) in allowed_dataset_ids])

    csplus_keys = _filter_keys(csplus_keys)
    csminus_keys = _filter_keys(csminus_keys)
    nonpc_keys = _filter_keys(nonpc_keys)

    global _SESSION_PAYLOAD_BY_DATASET
    _SESSION_PAYLOAD_BY_DATASET = session_payload_by_dataset

    global ENFORCE_S1S2_MIN_PEAK_RATE_FILTER, S1S2_MIN_PEAK_RATE_HZ
    ENFORCE_S1S2_MIN_PEAK_RATE_FILTER = bool(params.enforce_s1s2_min_peak_rate_filter)
    S1S2_MIN_PEAK_RATE_HZ = float(params.s1s2_min_peak_rate_hz)

    conditions = ("session1", "session2") if params.panel_mode == "s1_s2" else ("combined", "session1", "session2")
    condition_labels = {"combined": "Combined", "session1": "Session 1", "session2": "Session 2"}

    def _prepare_groups(cell_keys: list[tuple[str, int]]) -> tuple[list[list[dict[str, Any]]], int]:
        rows: list[list[dict[str, Any]]] = []
        missing_s2_rows = 0
        for dataset_id, cell_idx in cell_keys:
            ref_combined = _get_condition_cell(dataset_id, cell_idx, "combined")
            if ref_combined is None:
                continue

            row_cells: list[dict[str, Any]] = []
            for cond in conditions:
                c = _get_condition_cell(dataset_id, int(cell_idx), cond)
                if c is None:
                    if cond == "session2" and params.missing_s2_policy == "na_panel":
                        c = _make_na_cell(ref_combined, condition_labels[cond])
                        missing_s2_rows += 1
                    else:
                        row_cells = []
                        break
                else:
                    c = copy.deepcopy(c)
                    c["condition_label"] = condition_labels[cond]
                    c["is_na_panel"] = bool(c.get("is_na_panel", False))
                    _renormalize_ss_cs_maps_per_condition(c)
                row_cells.append(c)

            if len(row_cells) == len(conditions):
                rows.append(row_cells)
        return rows, missing_s2_rows

    csplus_groups, csplus_missing_s2 = _prepare_groups(csplus_keys)
    csminus_groups, csminus_missing_s2 = _prepare_groups(csminus_keys)
    nonpc_groups, nonpc_missing_s2 = _prepare_groups(nonpc_keys)

    all_plot_cells = [c for grp in (csplus_groups + csminus_groups + nonpc_groups) for c in grp]
    theta_vals = []
    slow_abs_vals = []
    for c in all_plot_cells:
        tmap = c.get("theta_map", None)
        smap = c.get("slow_map", None)
        if isinstance(tmap, np.ndarray) and np.any(np.isfinite(tmap)):
            theta_vals.append((float(np.nanmin(tmap)), float(np.nanmax(tmap))))
        if isinstance(smap, np.ndarray) and np.any(np.isfinite(smap)):
            slow_abs_vals.append(float(np.nanmax(np.abs(smap))))

    auto_theta_vlim = (min(v[0] for v in theta_vals), max(v[1] for v in theta_vals)) if theta_vals else None
    auto_slow_vlim = (-max(slow_abs_vals), max(slow_abs_vals)) if slow_abs_vals else None

    selected_theta_vlim = params.selected_theta_vlim if params.selected_theta_vlim is not None else auto_theta_vlim
    selected_slow_vlim = params.selected_slow_vlim if params.selected_slow_vlim is not None else auto_slow_vlim

    return {
        "session_payload_by_dataset": session_payload_by_dataset,
        "occupancy_similarity_df": occupancy_df,
        "allowed_dataset_ids": sorted(allowed_dataset_ids),
        "excluded_dataset_ids": excluded_dataset_ids,
        "distance_normalized_alignment": {
            "enabled": bool(params.align_to_distance_normalized_exports),
            "export_dir": (
                str(config.figures_root / "CKII_pooled" / "pf_distance_centered_2sessions_average_exports")
                if params.distance_normalized_export_dir is None
                else str(params.distance_normalized_export_dir)
            ),
            "min_trials_per_session": int(params.distance_normalized_min_trials_per_session),
            "pf_rank": int(params.distance_normalized_pf_rank),
        },
        "category_source": category_source,
        "category_counts_before_filters": category_counts_before_filters,
        "csplus_keys": csplus_keys,
        "csminus_keys": csminus_keys,
        "nonpc_keys": nonpc_keys,
        "csplus_groups": csplus_groups,
        "csminus_groups": csminus_groups,
        "nonpc_groups": nonpc_groups,
        "csplus_missing_s2": int(csplus_missing_s2),
        "csminus_missing_s2": int(csminus_missing_s2),
        "nonpc_missing_s2": int(nonpc_missing_s2),
        "selected_theta_vlim": selected_theta_vlim,
        "selected_slow_vlim": selected_slow_vlim,
        "conditions": conditions,
    }


def _session_cell_to_4panel_row(
    cell: dict[str, Any],
    combined_cell: dict[str, Any],
    *,
    is_cs_plc: bool,
    condition: str,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "session": str(combined_cell.get("session", combined_cell.get("animal_id", ""))),
        "animal_id": str(combined_cell.get("animal_id", combined_cell.get("session", ""))),
        "cell_idx": int(combined_cell.get("cell_idx", cell.get("cell_idx", -1))),
        "condition": str(condition),
        "is_place_cell": True,
        "is_cs_plc": bool(is_cs_plc),
    }
    passthrough_cols = [
        "peak_rate_all",
        "peak_rate_ss",
        "peak_rate_cs",
        "place_field_sizes_cm2",
        "place_field_sizes_cm2_ss",
        "place_field_sizes_cm2_cs",
        "coherence_all",
        "coherence_ss",
        "coherence_cs",
        "sparsity_all",
        "sparsity_ss",
        "sparsity_cs",
        "si_bits_per_spike",
        "si_bits_per_spike_ss",
        "si_bits_per_spike_cs",
        "all_inout_loco_in",
        "all_inout_loco_out",
        "ss_inout_loco_in",
        "ss_inout_loco_out",
        "cs_inout_loco_in",
        "cs_inout_loco_out",
        "fr_loco_all_allpf_all",
        "fr_loco_all_allpf_ss",
        "fr_loco_all_allpf_cs",
        "fr_in_allpf_all",
        "fr_in_allpf_ss",
        "fr_in_allpf_cs",
        "fr_out_allpf_all",
        "fr_out_allpf_ss",
        "fr_out_allpf_cs",
        "fr_quiet_all_allpf_all",
        "fr_quiet_all_allpf_ss",
        "fr_quiet_all_allpf_cs",
        "fr_quiet_in_allpf_all",
        "fr_quiet_in_allpf_ss",
        "fr_quiet_in_allpf_cs",
        "fr_quiet_out_allpf_all",
        "fr_quiet_out_allpf_ss",
        "fr_quiet_out_allpf_cs",
        "complex_burst_event_rate_loco",
        "complex_burst_event_rate_quiet",
        "complex_burst_event_rate_loco_in_pf",
        "complex_burst_event_rate_loco_out_pf",
        "complex_burst_event_rate_quiet_in_pf",
        "complex_burst_event_rate_quiet_out_pf",
        "complex_burst_event_rate_loco_in_pf1",
        "complex_burst_event_rate_loco_in_pf2",
        "all_rate_loco",
        "ss_rate_loco",
        "cs_rate_loco",
        "all_rate_quiet",
        "ss_rate_quiet",
        "cs_rate_quiet",
        "all_inout_quiet_in",
        "all_inout_quiet_out",
        "ss_inout_quiet_in",
        "ss_inout_quiet_out",
        "cs_inout_quiet_in",
        "cs_inout_quiet_out",
        "fr_in_pf1_all",
        "fr_in_pf1_ss",
        "fr_in_pf1_cs",
        "fr_in_pf2_all",
        "fr_in_pf2_ss",
        "fr_in_pf2_cs",
        "n_combined_reference_pfs",
        "frame_rate",
        "n_frames_total",
        "n_frames_kept_total",
        "total_minutes",
        "included_minutes",
        "n_removed_frames_total",
        "pct_removed_frames_total",
    ]
    for col in passthrough_cols:
        row[col] = cell.get(col, np.nan)

    row["peak_rate_all"] = cell.get("peak_rate_all", cell.get("peak_rate", row["peak_rate_all"]))
    row["peak_rate_ss"] = cell.get("peak_rate_ss", cell.get("ss_peak_rate", row["peak_rate_ss"]))
    row["peak_rate_cs"] = cell.get("peak_rate_cs", cell.get("cs_peak_rate", row["peak_rate_cs"]))
    row["place_field_sizes_cm2"] = cell.get("place_field_sizes_cm2", cell.get("pf_sizes", row["place_field_sizes_cm2"]))
    row["place_field_sizes_cm2_ss"] = cell.get("place_field_sizes_cm2_ss", cell.get("ss_pf_sizes", row["place_field_sizes_cm2_ss"]))
    row["place_field_sizes_cm2_cs"] = cell.get("place_field_sizes_cm2_cs", cell.get("cs_pf_sizes", row["place_field_sizes_cm2_cs"]))
    row["fr_loco_all_allpf_all"] = cell.get("fr_loco_all_allpf_all", cell.get("all_rate_loco", row["fr_loco_all_allpf_all"]))
    row["fr_loco_all_allpf_ss"] = cell.get("fr_loco_all_allpf_ss", cell.get("ss_rate_loco", row["fr_loco_all_allpf_ss"]))
    row["fr_loco_all_allpf_cs"] = cell.get("fr_loco_all_allpf_cs", cell.get("cs_rate_loco", row["fr_loco_all_allpf_cs"]))
    row["fr_in_allpf_all"] = cell.get("fr_in_allpf_all", cell.get("all_inout_loco_in", row["fr_in_allpf_all"]))
    row["fr_in_allpf_ss"] = cell.get("fr_in_allpf_ss", cell.get("ss_inout_loco_in", row["fr_in_allpf_ss"]))
    row["fr_in_allpf_cs"] = cell.get("fr_in_allpf_cs", cell.get("cs_inout_loco_in", row["fr_in_allpf_cs"]))
    row["fr_out_allpf_all"] = cell.get("fr_out_allpf_all", cell.get("all_inout_loco_out", row["fr_out_allpf_all"]))
    row["fr_out_allpf_ss"] = cell.get("fr_out_allpf_ss", cell.get("ss_inout_loco_out", row["fr_out_allpf_ss"]))
    row["fr_out_allpf_cs"] = cell.get("fr_out_allpf_cs", cell.get("cs_inout_loco_out", row["fr_out_allpf_cs"]))
    row["fr_quiet_all_allpf_all"] = cell.get("fr_quiet_all_allpf_all", cell.get("all_rate_quiet", row["fr_quiet_all_allpf_all"]))
    row["fr_quiet_all_allpf_ss"] = cell.get("fr_quiet_all_allpf_ss", cell.get("ss_rate_quiet", row["fr_quiet_all_allpf_ss"]))
    row["fr_quiet_all_allpf_cs"] = cell.get("fr_quiet_all_allpf_cs", cell.get("cs_rate_quiet", row["fr_quiet_all_allpf_cs"]))
    row["fr_quiet_in_allpf_all"] = cell.get("fr_quiet_in_allpf_all", cell.get("all_inout_quiet_in", row["fr_quiet_in_allpf_all"]))
    row["fr_quiet_in_allpf_ss"] = cell.get("fr_quiet_in_allpf_ss", cell.get("ss_inout_quiet_in", row["fr_quiet_in_allpf_ss"]))
    row["fr_quiet_in_allpf_cs"] = cell.get("fr_quiet_in_allpf_cs", cell.get("cs_inout_quiet_in", row["fr_quiet_in_allpf_cs"]))
    row["fr_quiet_out_allpf_all"] = cell.get("fr_quiet_out_allpf_all", cell.get("all_inout_quiet_out", row["fr_quiet_out_allpf_all"]))
    row["fr_quiet_out_allpf_ss"] = cell.get("fr_quiet_out_allpf_ss", cell.get("ss_inout_quiet_out", row["fr_quiet_out_allpf_ss"]))
    row["fr_quiet_out_allpf_cs"] = cell.get("fr_quiet_out_allpf_cs", cell.get("cs_inout_quiet_out", row["fr_quiet_out_allpf_cs"]))
    for pf_rank in (1, 2):
        row[f"fr_in_pf{pf_rank}_all"] = cell.get(
            f"fr_in_pf{pf_rank}_all",
            cell.get(f"pf{pf_rank}_all_inout_loco_in", row[f"fr_in_pf{pf_rank}_all"]),
        )
        row[f"fr_in_pf{pf_rank}_ss"] = cell.get(
            f"fr_in_pf{pf_rank}_ss",
            cell.get(f"pf{pf_rank}_ss_inout_loco_in", row[f"fr_in_pf{pf_rank}_ss"]),
        )
        row[f"fr_in_pf{pf_rank}_cs"] = cell.get(
            f"fr_in_pf{pf_rank}_cs",
            cell.get(f"pf{pf_rank}_cs_inout_loco_in", row[f"fr_in_pf{pf_rank}_cs"]),
        )
    return row


def build_session_split_4panel_tables(
    config: PipelineConfig,
    params: SessionCompareParams,
    payloads: dict[str, Any],
    groups_payload: dict[str, Any],
) -> dict[str, dict[str, pd.DataFrame]]:
    """Build session-specific CS+/CS- tables for the split 4-panel summary."""
    _ = config, params  # Kept in the public signature to mirror session-compare assembly calls.
    session_payload_by_dataset = dict(payloads.get("session_payload_by_dataset", {}))
    class_keys = {
        "csplus": list(groups_payload.get("csplus_keys", [])),
        "csminus": list(groups_payload.get("csminus_keys", [])),
    }
    min_stats_minutes = _stats_min_included_minutes(params)
    eligible_stats_keys: dict[str, list[tuple[str, int]]] = {"csplus": [], "csminus": []}
    stats_filter_summary: dict[str, dict[str, int | float]] = {}
    for class_name, keys in class_keys.items():
        n_missing_or_invalid = 0
        n_too_short = 0
        for dataset_id, cell_idx in keys:
            payload = session_payload_by_dataset.get(str(dataset_id), {})
            by_cell = payload.get("cells", {}) if isinstance(payload, dict) else {}
            by_cond = by_cell.get(int(cell_idx), {}) if isinstance(by_cell, dict) else {}
            if not isinstance(by_cond, dict):
                n_missing_or_invalid += 1
                continue
            s1 = by_cond.get("session1", None)
            s2 = by_cond.get("session2", None)
            if not isinstance(s1, dict) or not isinstance(s2, dict) or bool(s2.get("is_na_panel", False)):
                n_missing_or_invalid += 1
                continue
            if not _passes_stats_included_minutes(s1, s2, min_stats_minutes):
                n_too_short += 1
                continue
            eligible_stats_keys[class_name].append((str(dataset_id), int(cell_idx)))
        stats_filter_summary[class_name] = {
            "min_included_minutes_per_session": float(min_stats_minutes),
            "n_input": int(len(keys)),
            "n_included": int(len(eligible_stats_keys[class_name])),
            "n_missing_or_invalid_session": int(n_missing_or_invalid),
            "n_below_min_included_minutes": int(n_too_short),
        }

    groups_payload["stats_min_included_minutes_per_session"] = float(min_stats_minutes)
    groups_payload["stats_filter_summary"] = stats_filter_summary
    out: dict[str, dict[str, pd.DataFrame]] = {}

    for condition in ("session1", "session2"):
        rows_by_class: dict[str, list[dict[str, Any]]] = {"csplus": [], "csminus": []}
        for class_name, keys in eligible_stats_keys.items():
            is_cs_plc = class_name == "csplus"
            for dataset_id, cell_idx in keys:
                payload = session_payload_by_dataset.get(str(dataset_id), {})
                by_cell = payload.get("cells", {}) if isinstance(payload, dict) else {}
                by_cond = by_cell.get(int(cell_idx), {}) if isinstance(by_cell, dict) else {}
                if not isinstance(by_cond, dict):
                    continue
                combined_cell = by_cond.get("combined", None)
                session_cell = by_cond.get(condition, None)
                if not isinstance(combined_cell, dict) or not isinstance(session_cell, dict):
                    continue
                if bool(session_cell.get("is_na_panel", False)):
                    continue
                rows_by_class[class_name].append(
                    _session_cell_to_4panel_row(
                        session_cell,
                        combined_cell,
                        is_cs_plc=is_cs_plc,
                        condition=condition,
                    )
                )

        out[condition] = {
            "df_cs_plc": pd.DataFrame(rows_by_class["csplus"]),
            "df_non_cs_plc": pd.DataFrame(rows_by_class["csminus"]),
        }

    return out


def summarize_session_split_plc_classification(
    config: PipelineConfig,
    groups_payload: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    """Summarize whether combined-session PLC classes persist in split sessions."""

    def _condition_cell(group: list[dict[str, Any]], label: str) -> dict[str, Any] | None:
        for cell in group:
            if isinstance(cell, dict) and str(cell.get("condition_label", "")) == label:
                return cell
        return None

    def _row_from_group(combined_label: str, group: list[dict[str, Any]]) -> dict[str, Any]:
        s1 = _condition_cell(group, "Session 1")
        s2 = _condition_cell(group, "Session 2")
        ref = s1 if isinstance(s1, dict) else s2
        if not isinstance(ref, dict):
            ref = next((cell for cell in group if isinstance(cell, dict)), {})

        return {
            "combined_group": combined_label,
            "animal_id": str(ref.get("animal_id", ref.get("session", ""))) if isinstance(ref, dict) else "",
            "session": str(ref.get("session", ref.get("animal_id", ""))) if isinstance(ref, dict) else "",
            "cell_idx": int(ref.get("cell_idx", -1)) if isinstance(ref, dict) else -1,
            "session1_label": _split_session_plc_label(
                s1,
                cb_num_threshold=int(config.pooled.cb_num_threshold),
                cs_peak_rate_threshold=float(config.pooled.cs_peak_rate_threshold),
                cs_plc_definition_mode=str(config.pooled.cs_plc_definition_mode),
            ),
            "session2_label": _split_session_plc_label(
                s2,
                cb_num_threshold=int(config.pooled.cb_num_threshold),
                cs_peak_rate_threshold=float(config.pooled.cs_peak_rate_threshold),
                cs_plc_definition_mode=str(config.pooled.cs_plc_definition_mode),
            ),
            "session1_is_place_cell": bool(s1.get("is_place_cell", False)) if isinstance(s1, dict) else False,
            "session2_is_place_cell": bool(s2.get("is_place_cell", False)) if isinstance(s2, dict) else False,
            "session1_n_cb_run_in": _count_cb_run_in(s1) if isinstance(s1, dict) else 0,
            "session2_n_cb_run_in": _count_cb_run_in(s2) if isinstance(s2, dict) else 0,
            "session1_cs_peak_rate": s1.get("cs_peak_rate", np.nan) if isinstance(s1, dict) else np.nan,
            "session2_cs_peak_rate": s2.get("cs_peak_rate", np.nan) if isinstance(s2, dict) else np.nan,
        }

    group_specs = [
        ("combined CS+ PLC", "CS+ PLC", list(groups_payload.get("csplus_groups", []))),
        ("combined CS- PLC", "CS- PLC", list(groups_payload.get("csminus_groups", []))),
        ("combined Non-PLC", "Non-PLC", list(groups_payload.get("nonpc_groups", []))),
    ]

    per_cell_rows: list[dict[str, Any]] = []
    for combined_label, _, groups in group_specs:
        for group in groups:
            if isinstance(group, list):
                per_cell_rows.append(_row_from_group(combined_label, group))

    per_cell_df = pd.DataFrame(per_cell_rows)
    if per_cell_df.empty:
        summary_df = pd.DataFrame(
            columns=[
                "combined_group",
                "n_cells",
                "session1_CS+_PLC",
                "session1_CS-_PLC",
                "session1_Non-PLC",
                "session1_missing",
                "session2_CS+_PLC",
                "session2_CS-_PLC",
                "session2_Non-PLC",
                "session2_missing",
                "same_label_both_sessions",
                "becomes_split_session_PLC",
            ]
        )
        return {
            "per_cell_df": per_cell_df,
            "summary_df": summary_df,
            "nonplc_becomes_plc_df": per_cell_df.copy(),
        }

    summary_rows: list[dict[str, Any]] = []
    for combined_label, expected_label, _ in group_specs:
        sub = per_cell_df[per_cell_df["combined_group"] == combined_label]
        s1_counts = sub["session1_label"].value_counts()
        s2_counts = sub["session2_label"].value_counts()
        is_split_session_plc = (
            sub["session1_label"].isin(["CS+ PLC", "CS- PLC"])
            | sub["session2_label"].isin(["CS+ PLC", "CS- PLC"])
        )
        summary_rows.append(
            {
                "combined_group": combined_label,
                "n_cells": int(len(sub)),
                "session1_CS+_PLC": int(s1_counts.get("CS+ PLC", 0)),
                "session1_CS-_PLC": int(s1_counts.get("CS- PLC", 0)),
                "session1_Non-PLC": int(s1_counts.get("Non-PLC", 0)),
                "session1_missing": int(s1_counts.get("missing", 0)),
                "session2_CS+_PLC": int(s2_counts.get("CS+ PLC", 0)),
                "session2_CS-_PLC": int(s2_counts.get("CS- PLC", 0)),
                "session2_Non-PLC": int(s2_counts.get("Non-PLC", 0)),
                "session2_missing": int(s2_counts.get("missing", 0)),
                "same_label_both_sessions": int(
                    (
                        (sub["session1_label"] == expected_label)
                        & (sub["session2_label"] == expected_label)
                    ).sum()
                ),
                "becomes_split_session_PLC": int(
                    (
                        (sub["combined_group"] == "combined Non-PLC")
                        & is_split_session_plc
                    ).sum()
                ),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    nonplc_becomes_plc_df = per_cell_df[
        (per_cell_df["combined_group"] == "combined Non-PLC")
        & (
            per_cell_df["session1_label"].isin(["CS+ PLC", "CS- PLC"])
            | per_cell_df["session2_label"].isin(["CS+ PLC", "CS- PLC"])
        )
    ].copy()

    return {
        "per_cell_df": per_cell_df,
        "summary_df": summary_df,
        "nonplc_becomes_plc_df": nonplc_becomes_plc_df,
    }


def build_session_compare_analysis(
    config: PipelineConfig,
    params: SessionCompareParams,
    spatial_data: Any | None = None,
) -> dict[str, Any]:
    """Build all non-rendered session-compare analysis outputs in one call."""
    payloads = build_session_compare_payloads(config, params)
    groups = assemble_session_compare_groups(config, params, payloads, spatial_data=spatial_data)
    split_4panel_tables = build_session_split_4panel_tables(
        config=config,
        params=params,
        payloads=payloads,
        groups_payload=groups,
    )
    split_classification = summarize_session_split_plc_classification(config, groups)
    groups["session_split_classification"] = split_classification
    groups["session_split_classification_summary_df"] = split_classification["summary_df"]
    groups["nonplc_becomes_plc_df"] = split_classification["nonplc_becomes_plc_df"]
    similarity_stats_by_metric = compute_session_compare_all_similarity_stats(groups, params)
    groups["s1s2_similarity_stats_by_metric"] = similarity_stats_by_metric

    return {
        "payloads": payloads,
        "groups": groups,
        "session_split_4panel_tables": split_4panel_tables,
        "session_split_classification": split_classification,
        "session_split_classification_summary_df": split_classification["summary_df"],
        "nonplc_becomes_plc_df": split_classification["nonplc_becomes_plc_df"],
        "s1s2_similarity_stats_by_metric": similarity_stats_by_metric,
    }


def _condition_index_from_label(cell: dict[str, Any]) -> int | None:
    lbl = str(cell.get("condition_label", "")).lower()
    if "combined" in lbl:
        return 0
    if "session 1" in lbl:
        return 1
    if "session 2" in lbl:
        return 2
    return None


def _s1s2_map_values_and_weights(
    map1: Any,
    map2: Any,
    occ1: Any | None = None,
    occ2: Any | None = None,
    min_valid_bins: int = 20,
    min_eff_bins: int = 20,
    min_occ: float = 0.0,
    weighted: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    if not isinstance(map1, np.ndarray) or not isinstance(map2, np.ndarray):
        return np.array([], dtype=float), np.array([], dtype=float), None
    if map1.shape != map2.shape:
        return np.array([], dtype=float), np.array([], dtype=float), None

    x = np.asarray(map1, dtype=float).ravel()
    y = np.asarray(map2, dtype=float).ravel()
    valid = np.isfinite(x) & np.isfinite(y)
    w = None
    try:
        min_occ_val = float(min_occ)
    except (TypeError, ValueError):
        min_occ_val = 0.0
    if not np.isfinite(min_occ_val) or min_occ_val < 0:
        min_occ_val = 0.0
    occupancy_required = bool(weighted) or min_occ_val > 0.0
    if occupancy_required:
        if not isinstance(occ1, np.ndarray) or not isinstance(occ2, np.ndarray):
            return np.array([], dtype=float), np.array([], dtype=float), None
        if map1.shape != occ1.shape or map1.shape != occ2.shape:
            return np.array([], dtype=float), np.array([], dtype=float), None
        w = np.minimum(np.asarray(occ1, dtype=float).ravel(), np.asarray(occ2, dtype=float).ravel())
        occ_valid = np.isfinite(w)
        if min_occ_val > 0.0:
            occ_valid &= w >= min_occ_val
        elif bool(weighted):
            occ_valid &= w > 0.0
        valid &= occ_valid
    if int(np.sum(valid)) < int(min_valid_bins):
        return np.array([], dtype=float), np.array([], dtype=float), None

    x = x[valid]
    y = y[valid]
    if not bool(weighted):
        return x, y, None

    w = np.asarray(w[valid], dtype=float)
    sw = float(np.sum(w))
    if sw <= 0:
        return np.array([], dtype=float), np.array([], dtype=float), None
    sw2 = float(np.sum(w * w))
    n_eff = (sw * sw / sw2) if sw2 > 0 else 0.0
    if n_eff < float(min_eff_bins):
        return np.array([], dtype=float), np.array([], dtype=float), None
    return x, y, w


def _weighted_pearson_from_vectors(x: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> float:
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size < 3 or y.size < 3 or x.size != y.size:
        return np.nan
    if w is None:
        if np.nanstd(x) == 0 or np.nanstd(y) == 0:
            return np.nan
        r = float(np.corrcoef(x, y)[0, 1])
        return r if np.isfinite(r) else np.nan

    w = np.asarray(w, dtype=float).ravel()
    if w.size != x.size or not np.any(np.isfinite(w)):
        return np.nan
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    if int(np.sum(valid)) < 3:
        return np.nan
    x = x[valid]
    y = y[valid]
    w = w[valid]
    sw = float(np.sum(w))
    if sw <= 0:
        return np.nan

    mx = float(np.sum(w * x) / sw)
    my = float(np.sum(w * y) / sw)
    dx = x - mx
    dy = y - my

    den_x = float(np.sum(w * dx * dx))
    den_y = float(np.sum(w * dy * dy))
    den = np.sqrt(den_x * den_y)
    if den <= 0:
        return np.nan

    return float(np.sum(w * dx * dy) / den)


def _pearson_s1s2_map_corr(
    map1: Any,
    map2: Any,
    occ1: Any | None = None,
    occ2: Any | None = None,
    min_valid_bins: int = 20,
    min_eff_bins: int = 20,
    min_occ: float = 0.0,
    weighted: bool = False,
) -> float:
    x, y, w = _s1s2_map_values_and_weights(
        map1,
        map2,
        occ1,
        occ2,
        min_valid_bins=min_valid_bins,
        min_eff_bins=min_eff_bins,
        min_occ=min_occ,
        weighted=weighted,
    )
    return _weighted_pearson_from_vectors(x, y, w)


def _weighted_s1s2_map_corr(
    map1: Any,
    map2: Any,
    occ1: Any,
    occ2: Any,
    min_valid_bins: int = 20,
    min_eff_bins: int = 20,
    min_occ: float = 0.0,
) -> float:
    return _pearson_s1s2_map_corr(
        map1,
        map2,
        occ1,
        occ2,
        min_valid_bins=min_valid_bins,
        min_eff_bins=min_eff_bins,
        min_occ=min_occ,
        weighted=True,
    )


def _spearman_s1s2_map_corr(
    map1: Any,
    map2: Any,
    occ1: Any | None = None,
    occ2: Any | None = None,
    min_valid_bins: int = 20,
    min_eff_bins: int = 20,
    min_occ: float = 0.0,
    weighted: bool = False,
) -> float:
    x, y, w = _s1s2_map_values_and_weights(
        map1,
        map2,
        occ1,
        occ2,
        min_valid_bins=min_valid_bins,
        min_eff_bins=min_eff_bins,
        min_occ=min_occ,
        weighted=weighted,
    )
    if x.size < 3:
        return np.nan
    if np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return np.nan

    rx = pd.Series(x).rank(method="average").to_numpy(dtype=float)
    ry = pd.Series(y).rank(method="average").to_numpy(dtype=float)
    if np.nanstd(rx) == 0 or np.nanstd(ry) == 0:
        return np.nan
    if bool(weighted):
        return _weighted_pearson_from_vectors(rx, ry, w)

    if scipy_stats is not None:
        try:
            rho, _p = scipy_stats.spearmanr(x, y)
            return float(rho) if np.isfinite(rho) else np.nan
        except Exception:
            return np.nan

    rho = float(np.corrcoef(rx, ry)[0, 1])
    return rho if np.isfinite(rho) else np.nan


def _cosine_s1s2_map_corr(
    map1: Any,
    map2: Any,
    occ1: Any | None = None,
    occ2: Any | None = None,
    min_valid_bins: int = 20,
    min_eff_bins: int = 20,
    min_occ: float = 0.0,
    weighted: bool = False,
    center: bool = False,
) -> float:
    x, y, w = _s1s2_map_values_and_weights(
        map1,
        map2,
        occ1,
        occ2,
        min_valid_bins=min_valid_bins,
        min_eff_bins=min_eff_bins,
        min_occ=min_occ,
        weighted=weighted,
    )
    if x.size < 3:
        return np.nan

    if bool(center):
        if bool(weighted):
            if w is None:
                return np.nan
            sw = float(np.sum(w))
            if sw <= 0:
                return np.nan
            x = x - float(np.sum(w * x) / sw)
            y = y - float(np.sum(w * y) / sw)
        else:
            x = x - float(np.nanmean(x))
            y = y - float(np.nanmean(y))

    if bool(weighted):
        if w is None:
            return np.nan
        num = float(np.sum(w * x * y))
        den_x = float(np.sum(w * x * x))
        den_y = float(np.sum(w * y * y))
        den = np.sqrt(den_x * den_y)
        if den <= 0:
            return np.nan
        return float(np.clip(num / den, -1.0, 1.0))

    nx = float(np.linalg.norm(x))
    ny = float(np.linalg.norm(y))
    if nx <= 0 or ny <= 0:
        return np.nan
    c = float(np.dot(x, y) / (nx * ny))
    return float(np.clip(c, -1.0, 1.0))


def _normalize_session_compare_similarity_metric(metric: Any) -> tuple[str, str]:
    metric_raw = str(metric).strip().lower()
    if metric_raw in ("weighted_pearson", "weighted-r", "weighted_r", "weighted", "wr", "pearson", "r"):
        return "pearson", "r"
    if metric_raw in ("spearman", "spearman_rho", "rho"):
        return "spearman", "rho"
    if metric_raw in ("cosine", "cosine_similarity", "cos"):
        return "cosine", "cos"
    print(f"Unknown heatmap_similarity_metric='{metric}', fallback to pearson")
    return "pearson", "r"


def _similarity_label_with_weight(metric_label: str, weighted: bool) -> str:
    return f"weighted {metric_label}" if bool(weighted) else str(metric_label)


def _find_session1_session2_cells(group: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    idx_s1 = None
    idx_s2 = None
    for idx_cond, cell in enumerate(group):
        if not isinstance(cell, dict):
            continue
        cond_idx = _condition_index_from_label(cell)
        if cond_idx == 1:
            idx_s1 = idx_cond
        elif cond_idx == 2:
            idx_s2 = idx_cond

    if idx_s1 is None or idx_s2 is None:
        if len(group) == 2:
            idx_s1 = 0 if idx_s1 is None else idx_s1
            idx_s2 = 1 if idx_s2 is None else idx_s2
        elif len(group) >= 3:
            idx_s1 = 1 if idx_s1 is None else idx_s1
            idx_s2 = 2 if idx_s2 is None else idx_s2

    c1 = group[idx_s1] if idx_s1 is not None and 0 <= idx_s1 < len(group) else None
    c2 = group[idx_s2] if idx_s2 is not None and 0 <= idx_s2 < len(group) else None
    return (c1 if isinstance(c1, dict) else None), (c2 if isinstance(c2, dict) else None)


def _cell_included_minutes(cell: dict[str, Any] | None) -> float:
    if not isinstance(cell, dict):
        return np.nan
    for key in ("included_minutes", "included_time_min"):
        try:
            value = float(cell.get(key, np.nan))
        except (TypeError, ValueError):
            value = np.nan
        if np.isfinite(value):
            return value

    try:
        kept_frames = float(cell.get("n_frames_kept_total", np.nan))
        frame_rate = float(cell.get("frame_rate", np.nan))
    except (TypeError, ValueError):
        kept_frames = np.nan
        frame_rate = np.nan
    if np.isfinite(kept_frames) and np.isfinite(frame_rate) and frame_rate > 0:
        return kept_frames / frame_rate / 60.0
    return np.nan


def _stats_min_included_minutes(params: SessionCompareParams) -> float:
    try:
        value = float(getattr(params, "stats_min_included_minutes_per_session", 4.0))
    except (TypeError, ValueError):
        value = 4.0
    return value if np.isfinite(value) else 4.0


def _passes_stats_included_minutes(
    c1: dict[str, Any] | None,
    c2: dict[str, Any] | None,
    min_minutes: float,
) -> bool:
    if min_minutes <= 0:
        return True
    m1 = _cell_included_minutes(c1)
    m2 = _cell_included_minutes(c2)
    return bool(np.isfinite(m1) and np.isfinite(m2) and m1 >= float(min_minutes) and m2 >= float(min_minutes))


def _cb_condition_label(condition: str) -> str:
    labels = {
        "running": "Running",
        "run_in": "Run in PF",
        "run_out": "Run out PF",
        "resting": "Resting",
        "rest_in": "Rest in PF",
        "rest_out": "Rest out PF",
    }
    return labels.get(str(condition), str(condition))


def _session_cb_metric_specs(plateau_threshold_ms: float) -> list[tuple[str, str]]:
    return [
        ("n_spikes", "Spks./burst"),
        ("peak_amp", "Peak amp"),
        ("duration_ms", "Duration (ms)"),
        ("auc", "AUC"),
        ("burst_rate_hz", "CB rate (Hz)"),
        ("burst_prob", "CB prob."),
        ("plateau_pct", f"%CB > {int(plateau_threshold_ms)} ms"),
    ]


def _session_cb_sig_label(p_val: float) -> str:
    try:
        p_val = float(p_val)
    except Exception:
        return ""
    if not np.isfinite(p_val):
        return ""
    if p_val < 0.001:
        return "***"
    if p_val < 0.01:
        return "**"
    if p_val < 0.05:
        return "*"
    return "n.s."


def _session_cb_paired_test(s1_vals: Any, s2_vals: Any) -> dict[str, Any]:
    s1 = np.asarray(s1_vals, dtype=float).reshape(-1)
    s2 = np.asarray(s2_vals, dtype=float).reshape(-1)
    n = min(s1.size, s2.size)
    s1 = s1[:n]
    s2 = s2[:n]
    valid = np.isfinite(s1) & np.isfinite(s2)
    s1 = s1[valid]
    s2 = s2[valid]
    out = {
        "test": "n/a",
        "statistic": np.nan,
        "p_value": np.nan,
        "shapiro_p_diff": np.nan,
        "n_pairs": int(s1.size),
        "reason": "",
    }
    if s1.size < 3:
        out["reason"] = "fewer than three paired finite cells"
        return out
    if scipy_stats is None:
        out["reason"] = "scipy unavailable"
        return out
    diffs = s2 - s1
    if np.allclose(diffs, 0.0, rtol=0.0, atol=0.0):
        out.update({"test": "Paired t-test", "statistic": 0.0, "p_value": 1.0, "reason": "all paired differences are zero"})
        return out
    try:
        shapiro_p = float(scipy_stats.shapiro(diffs).pvalue)
    except Exception:
        shapiro_p = np.nan
    out["shapiro_p_diff"] = shapiro_p
    use_ttest = bool(np.isfinite(shapiro_p) and shapiro_p >= 0.05)
    if use_ttest:
        out["test"] = "Paired t-test"
        try:
            stat_raw, p_raw = scipy_stats.ttest_rel(s1, s2, nan_policy="omit")
        except Exception as exc:
            out["reason"] = str(exc)
            return out
    else:
        out["test"] = "Wilcoxon signed-rank"
        try:
            stat_raw, p_raw = scipy_stats.wilcoxon(s1, s2, alternative="two-sided")
        except Exception as exc:
            out["reason"] = str(exc)
            return out
    try:
        out["statistic"] = float(stat_raw)
    except Exception:
        out["statistic"] = np.nan
    try:
        out["p_value"] = float(p_raw)
    except Exception:
        out["p_value"] = np.nan
    if not np.isfinite(out["p_value"]) and not out["reason"]:
        out["reason"] = "test returned non-finite p-value"
    return out


def _session_cb_bursts_for_condition(cell: dict[str, Any], condition: str) -> list[dict[str, Any]]:
    if not isinstance(cell, dict):
        return []
    condition_key = str(condition)
    aggregate_components = {
        "running": ("run_in", "run_out"),
        "run": ("run_in", "run_out"),
        "loco": ("run_in", "run_out"),
        "resting": ("rest_in", "rest_out"),
        "rest": ("rest_in", "rest_out"),
        "quiet": ("rest_in", "rest_out"),
    }
    burst_metrics = cell.get("burst_metrics", None)
    complex_by_condition = burst_metrics.get("complex", None) if isinstance(burst_metrics, dict) else None
    if str(condition_key).lower() in aggregate_components:
        bursts_out: list[dict[str, Any]] = []
        if isinstance(complex_by_condition, dict):
            for component in aggregate_components[str(condition_key).lower()]:
                bursts = complex_by_condition.get(component, [])
                if isinstance(bursts, (list, tuple)):
                    bursts_out.extend([b for b in bursts if isinstance(b, dict)])
        return bursts_out
    bursts = complex_by_condition.get(condition_key, []) if isinstance(complex_by_condition, dict) else []
    if not isinstance(bursts, (list, tuple)):
        return []
    return [b for b in bursts if isinstance(b, dict)]


def _session_cb_direct_rate_metric_value(
    cell: dict[str, Any],
    condition: str,
    metric_key: str,
) -> tuple[float, str]:
    if metric_key not in {"burst_rate_hz", "burst_prob"}:
        return np.nan, "not_rate_metric"
    sbrm = cell.get("spike_burst_rate_metrics", None) if isinstance(cell, dict) else None
    source_key = "burst_rate" if metric_key == "burst_rate_hz" else "burst_prob"
    source = sbrm.get(source_key, None) if isinstance(sbrm, dict) else None
    if not isinstance(source, dict):
        return np.nan, "missing_spike_burst_rate_metrics"
    aliases_by_condition = {
        "running": ("running", "run", "loco"),
        "run": ("running", "run", "loco"),
        "loco": ("running", "run", "loco"),
        "resting": ("resting", "rest", "quiet"),
        "rest": ("resting", "rest", "quiet"),
        "quiet": ("resting", "rest", "quiet"),
    }
    aliases = aliases_by_condition.get(str(condition).strip().lower(), (str(condition),))
    for key in aliases:
        try:
            value = float(source.get(str(key), np.nan))
        except Exception:
            value = np.nan
        if np.isfinite(value):
            return value, f"direct:{source_key}.{key}"
    return np.nan, "missing_direct_condition_metric"


def _session_cb_aggregate_rate_metric_value(
    cell: dict[str, Any],
    condition: str,
    metric_key: str,
) -> tuple[float, str]:
    condition_norm = str(condition).strip().lower()
    if condition_norm in {"running", "run", "loco"}:
        state = "loco"
    elif condition_norm in {"resting", "rest", "quiet"}:
        state = "quiet"
    else:
        return np.nan, "not_aggregate_condition"

    if metric_key == "burst_rate_hz":
        for key in (f"complex_burst_event_rate_{state}",):
            try:
                value = float(cell.get(key, np.nan))
            except Exception:
                value = np.nan
            if np.isfinite(value):
                return value, f"aggregate:{key}"
        try:
            count = float(cell.get(f"complex_burst_event_count_{state}", np.nan))
            time_s = float(cell.get(f"complex_burst_time_{state}", np.nan))
        except Exception:
            count, time_s = np.nan, np.nan
        if np.isfinite(count) and np.isfinite(time_s) and time_s > 0:
            return float(count / time_s), f"aggregate:count/time:{state}"
        return np.nan, "missing_aggregate_rate_fields"

    if metric_key != "burst_prob":
        return np.nan, "not_rate_metric"

    # Prefer a direct aggregate burst probability if a newer cache stores it.
    direct, source = _session_cb_direct_rate_metric_value(cell, condition, metric_key)
    if np.isfinite(direct):
        return direct, source

    try:
        cb_count = float(cell.get(f"complex_burst_event_count_{state}", np.nan))
    except Exception:
        cb_count = np.nan
    if not np.isfinite(cb_count):
        count_parts = []
        for pf_label in ("in_pf", "out_pf"):
            try:
                count_parts.append(float(cell.get(f"complex_burst_event_count_{state}_{pf_label}", np.nan)))
            except Exception:
                count_parts.append(np.nan)
        finite_counts = np.asarray(count_parts, dtype=float)
        if np.any(np.isfinite(finite_counts)):
            cb_count = float(np.nansum(finite_counts))

    ss_count_parts = []
    for pf_label, inout_label in (("in_pf", "in"), ("out_pf", "out")):
        try:
            ss_rate = float(cell.get(f"ss_inout_{state}_{inout_label}", np.nan))
            time_s = float(cell.get(f"complex_burst_time_{state}_{pf_label}", np.nan))
        except Exception:
            ss_rate, time_s = np.nan, np.nan
        if np.isfinite(ss_rate) and np.isfinite(time_s) and time_s >= 0:
            ss_count_parts.append(ss_rate * time_s)
    ss_count = float(np.nansum(ss_count_parts)) if len(ss_count_parts) > 0 else np.nan
    if np.isfinite(cb_count) and np.isfinite(ss_count):
        total = cb_count + ss_count
        if total > 0:
            return float(cb_count / total), f"aggregate:cb_count/(ss_count+cb_count):{state}"
    return np.nan, "missing_aggregate_probability_fields"


def _session_cb_metric_value(
    cell: dict[str, Any],
    condition: str,
    metric_key: str,
    *,
    min_bursts_per_condition: int,
    plateau_threshold_ms: float,
) -> tuple[float, int, str]:
    bursts = _session_cb_bursts_for_condition(cell, condition)
    n_bursts = int(len(bursts))
    if n_bursts < int(min_bursts_per_condition):
        return np.nan, n_bursts, "below_min_bursts"

    if metric_key in {"n_spikes", "peak_amp", "duration_ms", "auc"}:
        vals = np.asarray([b.get(metric_key, np.nan) for b in bursts], dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size < int(min_bursts_per_condition):
            return np.nan, n_bursts, "below_min_finite_burst_metrics"
        return float(np.nanmean(vals)), n_bursts, "burst_metric_mean"

    if metric_key == "plateau_pct":
        durations = np.asarray([b.get("duration_ms", np.nan) for b in bursts], dtype=float)
        durations = durations[np.isfinite(durations)]
        if durations.size == 0:
            return np.nan, n_bursts, "missing_burst_durations"
        pct = 100.0 * float(np.sum(durations > float(plateau_threshold_ms))) / float(durations.size)
        return pct, n_bursts, "burst_duration_fraction"

    if metric_key in {"burst_rate_hz", "burst_prob"}:
        aggregate_value, aggregate_source = _session_cb_aggregate_rate_metric_value(cell, condition, metric_key)
        if np.isfinite(aggregate_value):
            return float(aggregate_value), n_bursts, aggregate_source
        value, source = _session_cb_direct_rate_metric_value(cell, condition, metric_key)
        return (float(value) if np.isfinite(value) else np.nan), n_bursts, source

    return np.nan, n_bursts, "unknown_metric"


def _session_cb_find_labeled_cell(group: list[dict[str, Any]], label: str) -> dict[str, Any] | None:
    label = str(label)
    for cell in group:
        if isinstance(cell, dict) and str(cell.get("condition_label", "")).startswith(label):
            return cell
    return None


def plot_session_compare_complex_burst_metrics(
    session_compare_groups: dict[str, Any],
    session_compare_params: SessionCompareParams,
    save_path: str | os.PathLike[str],
    conditions: tuple[str, ...] = ("running", "run_in", "run_out", "resting", "rest_in", "rest_out"),
    min_bursts_per_condition: int = 3,
    plateau_threshold_ms: float = 100.0,
    fig_width: float = 7.0,
    fig_height: float = 6.8,
    save_csv: bool = True,
) -> dict[str, Any]:
    """Compare CS+ complex-burst metrics between S1 and S2 for each PF/state condition."""
    metric_specs = _session_cb_metric_specs(float(plateau_threshold_ms))
    min_stats_minutes = _stats_min_included_minutes(session_compare_params)

    value_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    for group in list(session_compare_groups.get("csplus_groups", [])):
        if not isinstance(group, list):
            continue
        s1 = _session_cb_find_labeled_cell(group, "Session 1")
        s2 = _session_cb_find_labeled_cell(group, "Session 2")
        if not isinstance(s1, dict) or not isinstance(s2, dict) or bool(s2.get("is_na_panel", False)):
            skipped_rows.append({"reason": "missing valid S1 or S2 panel"})
            continue
        if not _passes_stats_included_minutes(s1, s2, min_stats_minutes):
            skipped_rows.append(
                {
                    "animal_id": str(s1.get("animal_id", s1.get("session", ""))),
                    "session": str(s1.get("session", s1.get("animal_id", ""))),
                    "cell_idx": int(s1.get("cell_idx", -1)),
                    "reason": "below stats_min_included_minutes_per_session",
                    "included_minutes_s1": _cell_included_minutes(s1),
                    "included_minutes_s2": _cell_included_minutes(s2),
                }
            )
            continue

        animal_id = str(s1.get("animal_id", s1.get("session", "")))
        session_name = str(s1.get("session", s1.get("animal_id", animal_id)))
        cell_idx = int(s1.get("cell_idx", -1))
        for condition in conditions:
            for metric_key, metric_label in metric_specs:
                s1_value, s1_count, s1_value_source = _session_cb_metric_value(
                    s1,
                    condition,
                    metric_key,
                    min_bursts_per_condition=int(min_bursts_per_condition),
                    plateau_threshold_ms=float(plateau_threshold_ms),
                )
                s2_value, s2_count, s2_value_source = _session_cb_metric_value(
                    s2,
                    condition,
                    metric_key,
                    min_bursts_per_condition=int(min_bursts_per_condition),
                    plateau_threshold_ms=float(plateau_threshold_ms),
                )
                if not (np.isfinite(s1_value) and np.isfinite(s2_value)):
                    continue
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio = float(s2_value / s1_value) if np.isfinite(s1_value) and s1_value != 0 else np.nan
                value_rows.append(
                    {
                        "animal_id": animal_id,
                        "session": session_name,
                        "cell_idx": cell_idx,
                        "cell_num": cell_idx + 1 if cell_idx >= 0 else np.nan,
                        "condition": str(condition),
                        "condition_label": _cb_condition_label(str(condition)),
                        "metric": metric_key,
                        "metric_label": metric_label,
                        "session1_value": float(s1_value),
                        "session2_value": float(s2_value),
                        "delta_s2_minus_s1": float(s2_value - s1_value),
                        "ratio_s2_s1": ratio,
                        "n_bursts_s1": int(s1_count),
                        "n_bursts_s2": int(s2_count),
                        "session1_value_source": str(s1_value_source),
                        "session2_value_source": str(s2_value_source),
                        "included_minutes_s1": _cell_included_minutes(s1),
                        "included_minutes_s2": _cell_included_minutes(s2),
                        "min_bursts_per_condition": int(min_bursts_per_condition),
                        "plateau_threshold_ms": float(plateau_threshold_ms),
                        "stats_min_included_minutes_per_session": float(min_stats_minutes),
                    }
                )

    values_df = pd.DataFrame(value_rows)
    if values_df.empty:
        values_df = pd.DataFrame(
            columns=[
                "animal_id",
                "session",
                "cell_idx",
                "cell_num",
                "condition",
                "condition_label",
                "metric",
                "metric_label",
                "session1_value",
                "session2_value",
                "delta_s2_minus_s1",
                "ratio_s2_s1",
                "n_bursts_s1",
                "n_bursts_s2",
                "session1_value_source",
                "session2_value_source",
                "included_minutes_s1",
                "included_minutes_s2",
                "min_bursts_per_condition",
                "plateau_threshold_ms",
                "stats_min_included_minutes_per_session",
            ]
        )
    skipped_df = pd.DataFrame(skipped_rows)

    summary_rows: list[dict[str, Any]] = []
    stats_rows: list[dict[str, Any]] = []
    for condition in conditions:
        for metric_key, metric_label in metric_specs:
            sub = values_df[
                (values_df["condition"] == str(condition))
                & (values_df["metric"] == metric_key)
            ].copy()
            for session_key, value_col in (("session1", "session1_value"), ("session2", "session2_value")):
                vals = pd.to_numeric(sub[value_col], errors="coerce").to_numpy(dtype=float) if value_col in sub else np.array([], dtype=float)
                vals = vals[np.isfinite(vals)]
                n_vals = int(vals.size)
                sd = float(np.nanstd(vals, ddof=1)) if n_vals > 1 else np.nan
                summary_rows.append(
                    {
                        "condition": str(condition),
                        "condition_label": _cb_condition_label(str(condition)),
                        "metric": metric_key,
                        "metric_label": metric_label,
                        "session_phase": session_key,
                        "n": n_vals,
                        "mean": float(np.nanmean(vals)) if n_vals > 0 else np.nan,
                        "sem": float(sd / np.sqrt(n_vals)) if n_vals > 1 and np.isfinite(sd) else np.nan,
                        "sd": sd,
                        "median": float(np.nanmedian(vals)) if n_vals > 0 else np.nan,
                        "min": float(np.nanmin(vals)) if n_vals > 0 else np.nan,
                        "max": float(np.nanmax(vals)) if n_vals > 0 else np.nan,
                    }
                )

            s1_vals = pd.to_numeric(sub["session1_value"], errors="coerce").to_numpy(dtype=float) if "session1_value" in sub else np.array([], dtype=float)
            s2_vals = pd.to_numeric(sub["session2_value"], errors="coerce").to_numpy(dtype=float) if "session2_value" in sub else np.array([], dtype=float)
            test_res = _session_cb_paired_test(s1_vals, s2_vals)
            stats_rows.append(
                {
                    "condition": str(condition),
                    "condition_label": _cb_condition_label(str(condition)),
                    "metric": metric_key,
                    "metric_label": metric_label,
                    "comparison": "Session 1 vs Session 2",
                    "test": test_res["test"],
                    "statistic": test_res["statistic"],
                    "p_value": test_res["p_value"],
                    "significance": _session_cb_sig_label(test_res["p_value"]),
                    "n_pairs": int(test_res["n_pairs"]),
                    "shapiro_p_diff": test_res["shapiro_p_diff"],
                    "reason": test_res["reason"],
                }
            )
    summary_df = pd.DataFrame(summary_rows)
    stats_df = pd.DataFrame(stats_rows)

    fig, axes = plt.subplots(
        len(conditions),
        len(metric_specs),
        figsize=(float(fig_width), float(fig_height)),
        sharex=False,
        sharey=False,
        squeeze=False,
    )

    def _draw_panel(ax: Any, s1_vals: np.ndarray, s2_vals: np.ndarray, stat_row: pd.Series | None) -> None:
        s1_vals = np.asarray(s1_vals, dtype=float).reshape(-1)
        s2_vals = np.asarray(s2_vals, dtype=float).reshape(-1)
        valid = np.isfinite(s1_vals) & np.isfinite(s2_vals)
        s1_vals = s1_vals[valid]
        s2_vals = s2_vals[valid]
        data = [s1_vals, s2_vals]
        colors = ["#4C78A8", "#F58518"]
        for pos, vals, color in zip([1, 2], data, colors):
            if vals.size > 0:
                vp = ax.violinplot([vals], positions=[pos], showmedians=True, showextrema=False, widths=0.55)
                body = vp["bodies"][0]
                body.set_facecolor(color)
                body.set_edgecolor("none")
                body.set_alpha(0.35)
                if "cmedians" in vp:
                    vp["cmedians"].set_color("#1F77B4")
                    vp["cmedians"].set_linewidth(0.8)

        if s1_vals.size > 0:
            jitter = np.linspace(-0.055, 0.055, s1_vals.size) if s1_vals.size > 1 else np.array([0.0])
            for idx in range(s1_vals.size):
                ax.plot([1 + jitter[idx], 2 + jitter[idx]], [s1_vals[idx], s2_vals[idx]], color="black", alpha=0.25, linewidth=0.45, zorder=1)
            ax.scatter(np.full(s1_vals.size, 1.0) + jitter, s1_vals, s=5, color="black", alpha=0.6, linewidths=0, zorder=2)
            ax.scatter(np.full(s2_vals.size, 2.0) + jitter, s2_vals, s=5, color="black", alpha=0.6, linewidths=0, zorder=2)

        finite_vals = np.concatenate([s1_vals[np.isfinite(s1_vals)], s2_vals[np.isfinite(s2_vals)]])
        if finite_vals.size > 0:
            y_min = float(np.nanmin(finite_vals))
            y_max = float(np.nanmax(finite_vals))
            y_span = y_max - y_min
            if not np.isfinite(y_span) or y_span <= 0:
                y_span = max(abs(y_max), 1.0) * 0.2
            y_low = y_min - 0.12 * y_span
            y_high = y_max + 0.30 * y_span
            if stat_row is not None and int(stat_row.get("n_pairs", 0)) >= 3:
                y = y_max + 0.09 * y_span
                h = 0.03 * y_span
                ax.plot([1, 1, 2, 2], [y, y + h, y + h, y], color="black", linewidth=0.6, clip_on=False)
                ax.text(
                    1.5,
                    y + h + 0.02 * y_span,
                    str(stat_row.get("significance", "")),
                    ha="center",
                    va="bottom",
                    fontsize=5,
                    fontname="Arial",
                    clip_on=False,
                )
                y_high = max(y_high, y + h + 0.16 * y_span)
            ax.set_ylim(y_low, y_high)
        ax.set_xlim(0.55, 2.45)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["S1", "S2"], fontsize=5, fontname="Arial")
        ax.tick_params(axis="both", labelsize=5, width=0.5, length=1.75, direction="in")
        ax.text(0.96, 0.94, f"n={s1_vals.size}", transform=ax.transAxes, ha="right", va="top", fontsize=5, fontname="Arial")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)

    for row_idx, condition in enumerate(conditions):
        for col_idx, (metric_key, metric_label) in enumerate(metric_specs):
            ax = axes[row_idx, col_idx]
            if row_idx == 0:
                ax.set_title(metric_label, fontsize=6, fontname="Arial")
            if col_idx == 0:
                ax.set_ylabel(_cb_condition_label(str(condition)), fontsize=6, fontname="Arial")
            sub = values_df[
                (values_df["condition"] == str(condition))
                & (values_df["metric"] == metric_key)
            ].copy()
            stat_sub = stats_df[
                (stats_df["condition"] == str(condition))
                & (stats_df["metric"] == metric_key)
            ]
            stat_row = stat_sub.iloc[0] if len(stat_sub) > 0 else None
            s1_vals = pd.to_numeric(sub["session1_value"], errors="coerce").to_numpy(dtype=float) if "session1_value" in sub else np.array([], dtype=float)
            s2_vals = pd.to_numeric(sub["session2_value"], errors="coerce").to_numpy(dtype=float) if "session2_value" in sub else np.array([], dtype=float)
            _draw_panel(ax, s1_vals, s2_vals, stat_row)

    fig.tight_layout(w_pad=0.45, h_pad=0.55)
    figure_path = str(save_path) if save_path is not None else None
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.rcParams["svg.fonttype"] = "none"
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    values_csv = summary_csv = stats_csv = skipped_csv = None
    if save_csv and save_path:
        save_path = Path(save_path)
        values_csv = str(save_path.with_name(f"{save_path.stem}_values.csv"))
        summary_csv = str(save_path.with_name(f"{save_path.stem}_summary.csv"))
        stats_csv = str(save_path.with_name(f"{save_path.stem}_stats.csv"))
        skipped_csv = str(save_path.with_name(f"{save_path.stem}_skipped.csv"))
        values_df.to_csv(values_csv, index=False)
        summary_df.to_csv(summary_csv, index=False)
        stats_df.to_csv(stats_csv, index=False)
        skipped_df.to_csv(skipped_csv, index=False)

    return {
        "fig": fig,
        "values_df": values_df,
        "summary_df": summary_df,
        "stats_df": stats_df,
        "skipped_df": skipped_df,
        "figure_path": figure_path,
        "values_csv": values_csv,
        "summary_csv": summary_csv,
        "stats_csv": stats_csv,
        "skipped_csv": skipped_csv,
    }


def _passes_s1s2_peak_threshold_from_cells(
    c1: dict[str, Any],
    c2: dict[str, Any],
    map_key: str,
    threshold: float,
) -> bool:
    p1 = _session_peak_for_map(c1, map_key=map_key)
    p2 = _session_peak_for_map(c2, map_key=map_key)
    finite_peaks = np.asarray([p1, p2], dtype=float)
    finite_peaks = finite_peaks[np.isfinite(finite_peaks)]
    return bool(finite_peaks.size > 0 and np.nanmax(finite_peaks) >= float(threshold))


def _compute_s1s2_map_similarity(
    c1: dict[str, Any],
    c2: dict[str, Any],
    map_key: str,
    *,
    metric_mode: str,
    enforce_peak_filter: bool,
    peak_threshold_hz: float,
    min_occupancy_per_bin_s: float | None = None,
    weighted: bool = False,
    baseline_subtraction_cosine: bool = True,
) -> float:
    if bool(enforce_peak_filter) and not _passes_s1s2_peak_threshold_from_cells(
        c1,
        c2,
        map_key=map_key,
        threshold=float(peak_threshold_hz),
    ):
        return np.nan

    min_valid_bins_w = int(globals().get("MIN_VALID_BINS_FOR_CORR_WEIGHTED", 20))
    min_eff_bins_w = int(globals().get("MIN_EFFECTIVE_BINS_WEIGHTED", 20))
    if min_occupancy_per_bin_s is None:
        min_occ_w = float(globals().get("MIN_OCCUPANCY_WEIGHT", 0.5))
    else:
        min_occ_w = float(min_occupancy_per_bin_s)
    if not np.isfinite(min_occ_w) or min_occ_w < 0:
        min_occ_w = 0.0
    min_valid_bins_u = int(globals().get("MIN_VALID_BINS_FOR_DISTANCE", min_valid_bins_w))

    if metric_mode == "pearson":
        return _pearson_s1s2_map_corr(
            c1.get(map_key, None),
            c2.get(map_key, None),
            c1.get("occupancy", None),
            c2.get("occupancy", None),
            min_valid_bins=min_valid_bins_w,
            min_eff_bins=min_eff_bins_w,
            min_occ=min_occ_w,
            weighted=bool(weighted),
        )
    if metric_mode == "spearman":
        return _spearman_s1s2_map_corr(
            c1.get(map_key, None),
            c2.get(map_key, None),
            c1.get("occupancy", None),
            c2.get("occupancy", None),
            min_valid_bins=min_valid_bins_w if bool(weighted) else min_valid_bins_u,
            min_eff_bins=min_eff_bins_w,
            min_occ=min_occ_w,
            weighted=bool(weighted),
        )
    return _cosine_s1s2_map_corr(
        c1.get(map_key, None),
        c2.get(map_key, None),
        c1.get("occupancy", None),
        c2.get("occupancy", None),
        min_valid_bins=min_valid_bins_w if bool(weighted) else min_valid_bins_u,
        min_eff_bins=min_eff_bins_w,
        min_occ=min_occ_w,
        weighted=bool(weighted),
        center=bool(baseline_subtraction_cosine),
    )


def _normalize_final_subplot_csminus_metric(metric: str | None) -> tuple[str, str, str]:
    metric_norm = str(metric or "all_spike").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "all": "all_spike",
        "all_spikes": "all_spike",
        "all_spike": "all_spike",
        "rate": "all_spike",
        "rate_map": "all_spike",
        "ss": "ss",
        "simple_spike": "ss",
        "simple_spikes": "ss",
        "ss_map": "ss",
    }
    metric_key = aliases.get(metric_norm)
    if metric_key is None:
        raise ValueError("final_subplot_csminus_metric must be 'all_spike' or 'ss'.")
    if metric_key == "ss":
        return metric_key, "CS- SS", "CS-\nSS"
    return metric_key, "CS- PLC", "CS-\nPLC"


def compute_session_compare_correlation_stats(
    groups_payload: dict[str, Any],
    params: SessionCompareParams,
    *,
    similarity_metric: str | None = None,
    weighted: bool | None = None,
    baseline_subtraction_cosine: bool | None = None,
    final_subplot_csminus_metric: str = "all_spike",
) -> dict[str, Any]:
    """Compute S1-vs-S2 map similarity tables without plotting."""
    metric_source = (
        getattr(params, "heatmap_similarity_metric", "pearson")
        if similarity_metric is None
        else similarity_metric
    )
    metric_mode, metric_label = _normalize_session_compare_similarity_metric(metric_source)
    final_csminus_metric, final_csminus_label, final_csminus_tick_label = _normalize_final_subplot_csminus_metric(
        final_subplot_csminus_metric
    )
    weighted_similarity = bool(getattr(params, "weighted", False) if weighted is None else weighted)
    baseline_cosine = bool(
        getattr(params, "baseline_subtraction_cosine", True)
        if baseline_subtraction_cosine is None
        else baseline_subtraction_cosine
    )
    display_metric_label = _similarity_label_with_weight(metric_label, weighted_similarity)
    enforce_peak_filter = bool(getattr(params, "enforce_s1s2_min_peak_rate_filter", True))
    peak_threshold_hz = float(getattr(params, "s1s2_min_peak_rate_hz", S1S2_MIN_PEAK_RATE_HZ))
    min_occupancy_per_bin_s = float(getattr(params, "s1s2_min_occupancy_per_bin_s", MIN_OCCUPANCY_WEIGHT))
    if not np.isfinite(min_occupancy_per_bin_s) or min_occupancy_per_bin_s < 0:
        min_occupancy_per_bin_s = 0.0
    min_stats_minutes = _stats_min_included_minutes(params)

    metric_specs = [
        ("all_spike", "All spikes", "rate_map"),
        ("ss", "SS", "ss_norm_map"),
        ("cs", "CS", "cs_norm_map"),
        ("theta", "Theta", "theta_map"),
        ("slow_vm", "Slow Vm", "slow_map"),
    ]
    group_specs = [
        ("CS+ PLC", "csplus_groups", "#D81B60"),
        ("CS- PLC", "csminus_groups", "#1F77B4"),
        ("Non-PLC", "nonpc_groups", "#8A8A8A"),
    ]

    value_rows: list[dict[str, Any]] = []
    for group_label, group_key, _color in group_specs:
        for group in list(groups_payload.get(group_key, [])):
            if not isinstance(group, list):
                continue
            c1, c2 = _find_session1_session2_cells(group)
            if c1 is None or c2 is None or bool(c2.get("is_na_panel", False)):
                continue
            if not _passes_stats_included_minutes(c1, c2, min_stats_minutes):
                continue
            animal_id = str(c1.get("animal_id", c1.get("session", "")))
            cell_idx = int(c1.get("cell_idx", -1))
            included_minutes_s1 = _cell_included_minutes(c1)
            included_minutes_s2 = _cell_included_minutes(c2)
            for metric_key, metric_title, map_key in metric_specs:
                sim_val = _compute_s1s2_map_similarity(
                    c1,
                    c2,
                    map_key,
                    metric_mode=metric_mode,
                    enforce_peak_filter=enforce_peak_filter,
                    peak_threshold_hz=peak_threshold_hz,
                    min_occupancy_per_bin_s=min_occupancy_per_bin_s,
                    weighted=weighted_similarity,
                    baseline_subtraction_cosine=baseline_cosine,
                )
                if not np.isfinite(sim_val):
                    continue
                value_rows.append(
                    {
                        "metric": metric_key,
                        "metric_label": metric_title,
                        "map_key": map_key,
                        "group": group_label,
                        "animal_id": animal_id,
                        "session": animal_id,
                        "cell_idx": cell_idx,
                        "cell_num": cell_idx + 1 if cell_idx >= 0 else np.nan,
                        "included_minutes_s1": included_minutes_s1,
                        "included_minutes_s2": included_minutes_s2,
                        "stats_min_included_minutes_per_session": float(min_stats_minutes),
                        "similarity_metric": metric_mode,
                        "similarity_label": display_metric_label,
                        "weighted": weighted_similarity,
                        "baseline_subtraction_cosine": baseline_cosine,
                        "s1s2_min_occupancy_per_bin_s": float(min_occupancy_per_bin_s),
                        "r": float(sim_val),
                    }
                )

    values_df = pd.DataFrame(value_rows)
    if values_df.empty:
        values_df = pd.DataFrame(
            columns=[
                "metric",
                "metric_label",
                "map_key",
                "group",
                "animal_id",
                "session",
                "cell_idx",
                "cell_num",
                "included_minutes_s1",
                "included_minutes_s2",
                "stats_min_included_minutes_per_session",
                "similarity_metric",
                "similarity_label",
                "weighted",
                "baseline_subtraction_cosine",
                "s1s2_min_occupancy_per_bin_s",
                "r",
            ]
        )

    paired_csplus_rows: list[dict[str, Any]] = []
    if not values_df.empty:
        csplus_ss_cs = values_df[
            (values_df["group"] == "CS+ PLC")
            & (values_df["metric"].isin(["ss", "cs"]))
        ].copy()
        if not csplus_ss_cs.empty:
            pivot = csplus_ss_cs.pivot_table(
                index=["animal_id", "cell_idx", "cell_num"],
                columns="metric",
                values="r",
                aggfunc="first",
            ).reset_index()
            for _, row in pivot.iterrows():
                ss_r = float(row.get("ss", np.nan))
                cs_r = float(row.get("cs", np.nan))
                if not (np.isfinite(ss_r) and np.isfinite(cs_r)):
                    continue
                paired_csplus_rows.append(
                    {
                        "animal_id": str(row.get("animal_id", "")),
                        "session": str(row.get("animal_id", "")),
                        "cell_idx": int(row.get("cell_idx", -1)),
                        "cell_num": int(row.get("cell_num", -1)) if np.isfinite(row.get("cell_num", np.nan)) else np.nan,
                        "group": "CS+ PLC",
                        "ss_r": ss_r,
                        "cs_r": cs_r,
                        "delta_cs_minus_ss": float(cs_r - ss_r),
                        "similarity_metric": metric_mode,
                        "similarity_label": display_metric_label,
                        "weighted": weighted_similarity,
                        "baseline_subtraction_cosine": baseline_cosine,
                        "s1s2_min_occupancy_per_bin_s": float(min_occupancy_per_bin_s),
                    }
                )
    paired_df = pd.DataFrame(paired_csplus_rows)
    if paired_df.empty:
        paired_df = pd.DataFrame(
            columns=[
                "animal_id",
                "session",
                "cell_idx",
                "cell_num",
                "group",
                "ss_r",
                "cs_r",
                "delta_cs_minus_ss",
                "similarity_metric",
                "similarity_label",
                "weighted",
                "baseline_subtraction_cosine",
                "s1s2_min_occupancy_per_bin_s",
            ]
        )

    summary_rows: list[dict[str, Any]] = []
    for metric_key, metric_title, _map_key in metric_specs:
        for group_label, _group_key, _color in group_specs:
            vals = values_df[
                (values_df["metric"] == metric_key)
                & (values_df["group"] == group_label)
            ]["r"].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            n = int(vals.size)
            mean = float(np.nanmean(vals)) if n > 0 else np.nan
            sd = float(np.nanstd(vals, ddof=1)) if n > 1 else np.nan
            sem = float(sd / np.sqrt(n)) if n > 1 and np.isfinite(sd) else np.nan
            summary_rows.append(
                {
                    "metric": metric_key,
                    "metric_label": metric_title,
                    "group": group_label,
                    "n": n,
                    "mean": mean,
                    "sem": sem,
                    "sd": sd,
                    "median": float(np.nanmedian(vals)) if n > 0 else np.nan,
                    "min": float(np.nanmin(vals)) if n > 0 else np.nan,
                    "max": float(np.nanmax(vals)) if n > 0 else np.nan,
                    "similarity_metric": metric_mode,
                    "similarity_label": display_metric_label,
                    "weighted": weighted_similarity,
                    "baseline_subtraction_cosine": baseline_cosine,
                    "s1s2_min_occupancy_per_bin_s": float(min_occupancy_per_bin_s),
                }
            )
    summary_df = pd.DataFrame(summary_rows)

    def _p_to_sig_text(p_val: float) -> str:
        if not np.isfinite(p_val):
            return "n.s."
        if p_val < 1e-4:
            return "****"
        if p_val < 1e-3:
            return "***"
        if p_val < 1e-2:
            return "**"
        if p_val < 5e-2:
            return "*"
        return "n.s."

    def _finite_1d(vals: Any) -> np.ndarray:
        arr = np.asarray(vals, dtype=float).ravel()
        return arr[np.isfinite(arr)]

    def _safe_test_float(val: Any) -> float:
        try:
            val_f = float(val)
        except Exception:
            return np.nan
        return val_f if np.isfinite(val_f) else np.nan

    def _shapiro_p(vals: np.ndarray) -> float:
        vals = _finite_1d(vals)
        if scipy_stats is None or vals.size < 3:
            return np.nan
        try:
            p_raw = scipy_stats.shapiro(vals).pvalue
        except Exception:
            return np.nan
        return _safe_test_float(p_raw)

    def _unpaired_parametric_first(vals1: np.ndarray, vals2: np.ndarray) -> dict[str, Any]:
        vals1 = _finite_1d(vals1)
        vals2 = _finite_1d(vals2)
        result = {
            "test": "n/a",
            "statistic": np.nan,
            "p_value": np.nan,
            "shapiro_p_group1": np.nan,
            "shapiro_p_group2": np.nan,
            "reason": "",
        }
        if vals1.size < 3 or vals2.size < 3:
            result["reason"] = "fewer than three finite values in one or both groups"
            return result
        if scipy_stats is None:
            result["reason"] = "scipy unavailable"
            return result

        result["shapiro_p_group1"] = _shapiro_p(vals1)
        result["shapiro_p_group2"] = _shapiro_p(vals2)
        use_ttest = (
            np.isfinite(result["shapiro_p_group1"])
            and np.isfinite(result["shapiro_p_group2"])
            and result["shapiro_p_group1"] >= 0.05
            and result["shapiro_p_group2"] >= 0.05
        )
        if use_ttest:
            result["test"] = "Unpaired t-test"
            try:
                stat_raw, p_raw = scipy_stats.ttest_ind(vals1, vals2)
            except Exception as exc:
                result["reason"] = str(exc)
                return result
        else:
            result["test"] = "Mann-Whitney U"
            try:
                stat_raw, p_raw = scipy_stats.mannwhitneyu(vals1, vals2, alternative="two-sided")
            except Exception as exc:
                result["reason"] = str(exc)
                return result

        result["statistic"] = _safe_test_float(stat_raw)
        result["p_value"] = _safe_test_float(p_raw)
        if not np.isfinite(result["p_value"]) and not result["reason"]:
            result["reason"] = "test returned non-finite p-value"
        return result

    def _paired_parametric_first(vals1: np.ndarray, vals2: np.ndarray) -> dict[str, Any]:
        vals1 = np.asarray(vals1, dtype=float).ravel()
        vals2 = np.asarray(vals2, dtype=float).ravel()
        n_pairs = min(vals1.size, vals2.size)
        vals1 = vals1[:n_pairs]
        vals2 = vals2[:n_pairs]
        valid = np.isfinite(vals1) & np.isfinite(vals2)
        vals1 = vals1[valid]
        vals2 = vals2[valid]
        result = {
            "test": "n/a",
            "statistic": np.nan,
            "p_value": np.nan,
            "shapiro_p_diff": np.nan,
            "n_pairs": int(vals1.size),
            "reason": "",
        }
        if vals1.size < 3:
            result["reason"] = "fewer than three paired finite cells"
            return result
        if scipy_stats is None:
            result["reason"] = "scipy unavailable"
            return result

        diffs = vals1 - vals2
        result["shapiro_p_diff"] = _shapiro_p(diffs)
        if np.allclose(diffs, 0.0, rtol=0.0, atol=0.0):
            result["test"] = "Paired t-test"
            result["statistic"] = 0.0
            result["p_value"] = 1.0
            result["reason"] = "all paired differences are zero"
            return result

        use_ttest = (
            np.isfinite(result["shapiro_p_diff"])
            and result["shapiro_p_diff"] >= 0.05
        )
        if use_ttest:
            result["test"] = "Paired t-test"
            try:
                stat_raw, p_raw = scipy_stats.ttest_rel(vals1, vals2, nan_policy="omit")
            except Exception as exc:
                result["reason"] = str(exc)
                return result
        else:
            result["test"] = "Wilcoxon signed-rank"
            try:
                stat_raw, p_raw = scipy_stats.wilcoxon(vals1, vals2, alternative="two-sided")
            except Exception as exc:
                result["reason"] = str(exc)
                return result

        result["statistic"] = _safe_test_float(stat_raw)
        result["p_value"] = _safe_test_float(p_raw)
        if not np.isfinite(result["p_value"]) and not result["reason"]:
            result["reason"] = "test returned non-finite p-value"
        return result

    pairwise_specs = [
        (0, 1, "CS+ PLC", "CS- PLC"),
        (0, 2, "CS+ PLC", "Non-PLC"),
        (1, 2, "CS- PLC", "Non-PLC"),
    ]

    pairwise_rows: list[dict[str, Any]] = []
    for metric_key, metric_title, _map_key in metric_specs:
        vals_by_group: dict[str, np.ndarray] = {}
        for group_label, _group_key, _color in group_specs:
            vals = values_df[
                (values_df["metric"] == metric_key)
                & (values_df["group"] == group_label)
            ]["r"].to_numpy(dtype=float)
            vals_by_group[group_label] = vals[np.isfinite(vals)]

        for group1_idx, group2_idx, group1_label, group2_label in pairwise_specs:
            vals1 = vals_by_group.get(group1_label, np.array([], dtype=float))
            vals2 = vals_by_group.get(group2_label, np.array([], dtype=float))
            test_res = _unpaired_parametric_first(vals1, vals2)
            stat = test_res["statistic"]
            p_val = test_res["p_value"]

            pairwise_rows.append(
                {
                    "metric": metric_key,
                    "metric_label": metric_title,
                    "comparison": f"{group1_label} vs {group2_label}",
                    "group1": group1_label,
                    "group2": group2_label,
                    "group1_index": int(group1_idx),
                    "group2_index": int(group2_idx),
                    "test": test_res["test"],
                    "statistic": stat,
                    "p_value": p_val,
                    "significance": _p_to_sig_text(p_val),
                    "n_group1": int(vals1.size),
                    "n_group2": int(vals2.size),
                    "shapiro_p_group1": test_res["shapiro_p_group1"],
                    "shapiro_p_group2": test_res["shapiro_p_group2"],
                    "reason": test_res["reason"],
                    "similarity_metric": metric_mode,
                    "similarity_label": display_metric_label,
                    "weighted": weighted_similarity,
                    "baseline_subtraction_cosine": baseline_cosine,
                    "s1s2_min_occupancy_per_bin_s": float(min_occupancy_per_bin_s),
                }
            )
    pairwise_df = pd.DataFrame(pairwise_rows)

    paired_n = int(len(paired_df))
    ss_paired_vals = paired_df["ss_r"].to_numpy(dtype=float) if "ss_r" in paired_df.columns else np.array([], dtype=float)
    cs_paired_vals = paired_df["cs_r"].to_numpy(dtype=float) if "cs_r" in paired_df.columns else np.array([], dtype=float)
    paired_valid = np.isfinite(ss_paired_vals) & np.isfinite(cs_paired_vals)
    ss_paired_vals = ss_paired_vals[paired_valid]
    cs_paired_vals = cs_paired_vals[paired_valid]
    paired_res = _paired_parametric_first(ss_paired_vals, cs_paired_vals)
    paired_n = int(paired_res["n_pairs"])
    paired_stats_df = pd.DataFrame(
        [
            {
                "comparison": "CS+ PLC SS vs CS",
                "group": "CS+ PLC",
                "metric1": "ss",
                "metric2": "cs",
                "test": paired_res["test"],
                "statistic": paired_res["statistic"],
                "p_value": paired_res["p_value"],
                "significance": _p_to_sig_text(paired_res["p_value"]),
                "n_pairs": paired_n,
                "shapiro_p": paired_res["shapiro_p_diff"],
                "shapiro_p_diff": paired_res["shapiro_p_diff"],
                "reason": paired_res["reason"],
                "similarity_metric": metric_mode,
                "similarity_label": display_metric_label,
                "weighted": weighted_similarity,
                "baseline_subtraction_cosine": baseline_cosine,
                "s1s2_min_occupancy_per_bin_s": float(min_occupancy_per_bin_s),
            }
        ]
    )

    csminus_all_vals = values_df[
        (values_df["metric"] == "all_spike")
        & (values_df["group"] == "CS- PLC")
    ]["r"].to_numpy(dtype=float)
    csminus_all_vals = csminus_all_vals[np.isfinite(csminus_all_vals)]
    csminus_final_vals = values_df[
        (values_df["metric"] == final_csminus_metric)
        & (values_df["group"] == "CS- PLC")
    ]["r"].to_numpy(dtype=float)
    csminus_final_vals = csminus_final_vals[np.isfinite(csminus_final_vals)]
    combined_final_specs = [
        ("CS+ SS", "CS+ PLC", "ss", ss_paired_vals, "#026C80"),
        ("CS+ CS", "CS+ PLC", "cs", cs_paired_vals, "#EE9B00"),
        (final_csminus_label, "CS- PLC", final_csminus_metric, csminus_final_vals, "#1F77B4"),
    ]
    combined_final_rows: list[dict[str, Any]] = []
    for group_label, source_group, source_metric, vals, color in combined_final_specs:
        vals = _finite_1d(vals)
        for val in vals:
            combined_final_rows.append(
                {
                    "panel_group": group_label,
                    "source_group": source_group,
                    "source_metric": source_metric,
                    "color": color,
                    "r": float(val),
                    "similarity_metric": metric_mode,
                    "similarity_label": display_metric_label,
                    "weighted": weighted_similarity,
                    "baseline_subtraction_cosine": baseline_cosine,
                    "s1s2_min_occupancy_per_bin_s": float(min_occupancy_per_bin_s),
                }
            )
    combined_final_df = pd.DataFrame(combined_final_rows)
    if combined_final_df.empty:
        combined_final_df = pd.DataFrame(
            columns=[
                "panel_group",
                "source_group",
                "source_metric",
                "color",
                "r",
                "similarity_metric",
                "similarity_label",
                "weighted",
                "baseline_subtraction_cosine",
                "s1s2_min_occupancy_per_bin_s",
            ]
        )

    combined_final_pairwise_rows: list[dict[str, Any]] = []
    final_pairwise_specs = [
        (0, 1, "CS+ SS", "CS+ CS", "paired"),
        (0, 2, "CS+ SS", final_csminus_label, "unpaired"),
        (1, 2, "CS+ CS", final_csminus_label, "unpaired"),
    ]
    final_vals_by_group = {
        label: _finite_1d(vals)
        for label, _source_group, _source_metric, vals, _color in combined_final_specs
    }
    for group1_idx, group2_idx, group1_label, group2_label, comparison_kind in final_pairwise_specs:
        if comparison_kind == "paired":
            test_res = paired_res
            stat = test_res["statistic"]
            p_val = test_res["p_value"]
            row_extra = {
                "n_pairs": int(test_res.get("n_pairs", 0)),
                "n_group1": int(test_res.get("n_pairs", 0)),
                "n_group2": int(test_res.get("n_pairs", 0)),
                "shapiro_p": test_res.get("shapiro_p_diff", np.nan),
                "shapiro_p_diff": test_res.get("shapiro_p_diff", np.nan),
                "shapiro_p_group1": np.nan,
                "shapiro_p_group2": np.nan,
            }
        else:
            vals1 = final_vals_by_group.get(group1_label, np.array([], dtype=float))
            vals2 = final_vals_by_group.get(group2_label, np.array([], dtype=float))
            test_res = _unpaired_parametric_first(vals1, vals2)
            stat = test_res["statistic"]
            p_val = test_res["p_value"]
            row_extra = {
                "n_pairs": np.nan,
                "n_group1": int(vals1.size),
                "n_group2": int(vals2.size),
                "shapiro_p": np.nan,
                "shapiro_p_diff": np.nan,
                "shapiro_p_group1": test_res.get("shapiro_p_group1", np.nan),
                "shapiro_p_group2": test_res.get("shapiro_p_group2", np.nan),
            }
        combined_final_pairwise_rows.append(
            {
                "comparison": f"{group1_label} vs {group2_label}",
                "group1": group1_label,
                "group2": group2_label,
                "group1_index": int(group1_idx),
                "group2_index": int(group2_idx),
                "comparison_kind": comparison_kind,
                "test": test_res["test"],
                "statistic": stat,
                "p_value": p_val,
                "significance": _p_to_sig_text(p_val),
                "reason": test_res["reason"],
                "similarity_metric": metric_mode,
                "similarity_label": display_metric_label,
                "weighted": weighted_similarity,
                "baseline_subtraction_cosine": baseline_cosine,
                "s1s2_min_occupancy_per_bin_s": float(min_occupancy_per_bin_s),
                **row_extra,
            }
        )
    combined_final_pairwise_df = pd.DataFrame(combined_final_pairwise_rows)

    return {
        "metric_specs": metric_specs,
        "group_specs": group_specs,
        "values_df": values_df,
        "summary_df": summary_df,
        "pairwise_df": pairwise_df,
        "paired_df": paired_df,
        "paired_stats_df": paired_stats_df,
        "combined_final_df": combined_final_df,
        "combined_final_pairwise_df": combined_final_pairwise_df,
        "combined_final_specs": combined_final_specs,
        "combined_final_pairwise_specs": final_pairwise_specs,
        "pairwise_specs": pairwise_specs,
        "paired_n": paired_n,
        "ss_paired_vals": ss_paired_vals,
        "cs_paired_vals": cs_paired_vals,
        "csminus_all_vals": csminus_all_vals,
        "csminus_final_vals": csminus_final_vals,
        "combined_final_csminus_metric": final_csminus_metric,
        "combined_final_csminus_label": final_csminus_label,
        "combined_final_csminus_tick_label": final_csminus_tick_label,
        "similarity_metric": metric_mode,
        "similarity_label": display_metric_label,
        "weighted": weighted_similarity,
        "baseline_subtraction_cosine": baseline_cosine,
        "s1s2_min_occupancy_per_bin_s": float(min_occupancy_per_bin_s),
        "theta_cosine_mean_centered": bool(metric_mode == "cosine" and baseline_cosine),
        "cosine_baseline_subtracted": bool(metric_mode == "cosine" and baseline_cosine),
    }


def compute_session_compare_all_similarity_stats(
    groups_payload: dict[str, Any],
    params: SessionCompareParams,
    *,
    similarity_metrics: tuple[str, ...] | list[str] | None = None,
    baseline_subtraction_cosine: bool | None = None,
    final_subplot_csminus_metric: str = "all_spike",
) -> dict[str, dict[str, Any]]:
    """Compute session-compare similarity tables for all requested metric modes."""
    if similarity_metrics is None:
        similarity_metrics = ("pearson", "spearman", "cosine")
    out: dict[str, dict[str, Any]] = {}
    for metric in similarity_metrics:
        metric_mode, _metric_label = _normalize_session_compare_similarity_metric(metric)
        if metric_mode in out:
            continue
        out[metric_mode] = compute_session_compare_correlation_stats(
            groups_payload,
            params,
            similarity_metric=metric_mode,
            baseline_subtraction_cosine=baseline_subtraction_cosine,
            final_subplot_csminus_metric=final_subplot_csminus_metric,
        )
    return out


def plot_session_compare_correlation_stats(
    groups_payload: dict[str, Any],
    params: SessionCompareParams,
    *,
    figure_save_folder: str | os.PathLike[str],
    save_name: str = "SessionCompare_S1S2_CorrelationStats.svg",
    show_plot: bool = True,
    save_csv: bool = True,
    figsize: tuple[float, float] = (8.4, 2.2),
    random_seed: int = 42,
    similarity_metric: str | None = None,
    precomputed_stats: dict[str, Any] | None = None,
    baseline_subtraction_cosine: bool | None = None,
    final_subplot_csminus_metric: str = "all_spike",
) -> dict[str, Any]:
    """Plot S1-vs-S2 map correlation summaries for CS+ PLCs, CS- PLCs, and non-PLCs."""
    final_csminus_metric, final_csminus_label, final_csminus_tick_label = _normalize_final_subplot_csminus_metric(
        final_subplot_csminus_metric
    )
    target_baseline_cosine = bool(
        getattr(params, "baseline_subtraction_cosine", True)
        if baseline_subtraction_cosine is None
        else baseline_subtraction_cosine
    )
    stats_payload = (
        dict(precomputed_stats)
        if isinstance(precomputed_stats, dict)
        else compute_session_compare_correlation_stats(
            groups_payload,
            params,
            similarity_metric=similarity_metric,
            baseline_subtraction_cosine=target_baseline_cosine,
            final_subplot_csminus_metric=final_csminus_metric,
        )
    )
    metric_specs = list(stats_payload.get("metric_specs", []))
    group_specs = list(stats_payload.get("group_specs", []))
    values_df = stats_payload.get("values_df", pd.DataFrame()).copy()
    summary_df = stats_payload.get("summary_df", pd.DataFrame()).copy()
    pairwise_df = stats_payload.get("pairwise_df", pd.DataFrame()).copy()
    paired_df = stats_payload.get("paired_df", pd.DataFrame()).copy()
    paired_stats_df = stats_payload.get("paired_stats_df", pd.DataFrame()).copy()
    pairwise_specs = list(
        stats_payload.get(
            "pairwise_specs",
            [
                (0, 1, "CS+ PLC", "CS- PLC"),
                (0, 2, "CS+ PLC", "Non-PLC"),
                (1, 2, "CS- PLC", "Non-PLC"),
            ],
        )
    )
    paired_n = int(stats_payload.get("paired_n", len(paired_df)))
    ss_paired_vals = np.asarray(stats_payload.get("ss_paired_vals", []), dtype=float).ravel()
    cs_paired_vals = np.asarray(stats_payload.get("cs_paired_vals", []), dtype=float).ravel()
    metric_mode = str(stats_payload.get("similarity_metric", "pearson"))
    metric_label = str(stats_payload.get("similarity_label", "r"))

    target_min_occ = float(getattr(params, "s1s2_min_occupancy_per_bin_s", MIN_OCCUPANCY_WEIGHT))
    if not np.isfinite(target_min_occ) or target_min_occ < 0:
        target_min_occ = 0.0
    payload_min_occ = stats_payload.get("s1s2_min_occupancy_per_bin_s", np.nan)
    try:
        payload_min_occ = float(payload_min_occ)
    except (TypeError, ValueError):
        payload_min_occ = np.nan
    if (not np.isfinite(payload_min_occ)) and isinstance(values_df, pd.DataFrame) and "s1s2_min_occupancy_per_bin_s" in values_df.columns and len(values_df) > 0:
        try:
            payload_min_occ = float(values_df["s1s2_min_occupancy_per_bin_s"].dropna().iloc[0])
        except Exception:
            payload_min_occ = np.nan
    needs_threshold_refresh = bool(groups_payload) and (
        (not np.isfinite(payload_min_occ))
        or (not np.isclose(float(payload_min_occ), float(target_min_occ), rtol=0.0, atol=1e-12))
    )
    needs_theta_cosine_refresh = (
        bool(groups_payload)
        and str(metric_mode).strip().lower() == "cosine"
        and target_baseline_cosine
        and not bool(stats_payload.get("theta_cosine_mean_centered", False))
    )
    payload_baseline_cosine = bool(stats_payload.get("cosine_baseline_subtracted", False))
    needs_baseline_cosine_refresh = (
        bool(groups_payload)
        and str(metric_mode).strip().lower() == "cosine"
        and payload_baseline_cosine != target_baseline_cosine
    )
    payload_final_csminus_metric = str(stats_payload.get("combined_final_csminus_metric", "all_spike"))
    needs_final_csminus_refresh = bool(groups_payload) and payload_final_csminus_metric != final_csminus_metric
    if (
        "combined_final_df" not in stats_payload
        or "combined_final_pairwise_df" not in stats_payload
        or needs_threshold_refresh
        or needs_theta_cosine_refresh
        or needs_baseline_cosine_refresh
        or needs_final_csminus_refresh
    ):
        stats_payload = compute_session_compare_correlation_stats(
            groups_payload,
            params,
            similarity_metric=metric_mode if metric_mode else similarity_metric,
            weighted=bool(stats_payload.get("weighted", getattr(params, "weighted", False))),
            baseline_subtraction_cosine=target_baseline_cosine,
            final_subplot_csminus_metric=final_csminus_metric,
        )
        metric_specs = list(stats_payload.get("metric_specs", []))
        group_specs = list(stats_payload.get("group_specs", []))
        values_df = stats_payload.get("values_df", pd.DataFrame()).copy()
        summary_df = stats_payload.get("summary_df", pd.DataFrame()).copy()
        pairwise_df = stats_payload.get("pairwise_df", pd.DataFrame()).copy()
        paired_df = stats_payload.get("paired_df", pd.DataFrame()).copy()
        paired_stats_df = stats_payload.get("paired_stats_df", pd.DataFrame()).copy()
        pairwise_specs = list(
            stats_payload.get(
                "pairwise_specs",
                [
                    (0, 1, "CS+ PLC", "CS- PLC"),
                    (0, 2, "CS+ PLC", "Non-PLC"),
                    (1, 2, "CS- PLC", "Non-PLC"),
                ],
            )
        )
        paired_n = int(stats_payload.get("paired_n", len(paired_df)))
        ss_paired_vals = np.asarray(stats_payload.get("ss_paired_vals", []), dtype=float).ravel()
        cs_paired_vals = np.asarray(stats_payload.get("cs_paired_vals", []), dtype=float).ravel()
        metric_mode = str(stats_payload.get("similarity_metric", "pearson"))
        metric_label = str(stats_payload.get("similarity_label", "r"))

    combined_final_df = stats_payload.get("combined_final_df", pd.DataFrame()).copy()
    combined_final_pairwise_df = stats_payload.get("combined_final_pairwise_df", pd.DataFrame()).copy()
    final_csminus_label = str(stats_payload.get("combined_final_csminus_label", final_csminus_label))
    final_csminus_tick_label = str(stats_payload.get("combined_final_csminus_tick_label", final_csminus_tick_label))
    combined_final_pairwise_specs = list(
        stats_payload.get(
            "combined_final_pairwise_specs",
            [
                (0, 1, "CS+ SS", "CS+ CS", "paired"),
                (0, 2, "CS+ SS", final_csminus_label, "unpaired"),
                (1, 2, "CS+ CS", final_csminus_label, "unpaired"),
            ],
        )
    )
    combined_final_group_specs = [
        ("CS+ SS", "#026C80", "CS+\nSS"),
        ("CS+ CS", "#EE9B00", "CS+\nCS"),
        (final_csminus_label, "#1F77B4", final_csminus_tick_label),
    ]

    fig, axes = plt.subplots(1, len(metric_specs) + 2, figsize=figsize, sharey=False)
    axes = np.atleast_1d(axes)
    rng = np.random.default_rng(int(random_seed))
    x = np.arange(len(group_specs), dtype=float)
    colors = [spec[2] for spec in group_specs]
    group_tick_labels = ["CS+\nPLC", "CS-\nPLC", "Non-\nPLC"]
    draw_zero_line = str(metric_mode).strip().lower() != "cosine"

    def _panel_axis_layout(value_arrays: list[np.ndarray], n_brackets: int) -> tuple[float, float, float, float, float]:
        finite_chunks = []
        for vals in value_arrays:
            vals_arr = np.asarray(vals, dtype=float).ravel()
            vals_arr = vals_arr[np.isfinite(vals_arr)]
            if vals_arr.size > 0:
                finite_chunks.append(vals_arr)
        if finite_chunks:
            all_vals = np.concatenate(finite_chunks)
            data_min = float(np.nanmin(all_vals))
            data_max = float(np.nanmax(all_vals))
        else:
            data_min = 0.0 if not draw_zero_line else -0.1
            data_max = 1.0 if str(metric_mode).strip().lower() == "cosine" else 0.1
        if draw_zero_line:
            data_min = min(data_min, 0.0)
            data_max = max(data_max, 0.0)
        data_span = max(float(data_max - data_min), 0.05)
        lower = data_min - 0.12 * data_span
        bracket_h = max(0.035 * data_span, 0.008)
        bracket_gap = max(0.14 * data_span, 0.035)
        bracket_y0 = data_max + 0.12 * data_span
        upper = data_max + 0.18 * data_span
        if n_brackets > 0:
            upper = max(
                upper,
                bracket_y0 + max(0, n_brackets - 1) * bracket_gap + bracket_h + 0.08 * data_span,
            )
        if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
            lower, upper = (-0.1, 1.1) if str(metric_mode).strip().lower() == "cosine" else (-0.1, 0.1)
        return float(lower), float(upper), float(bracket_y0), float(bracket_gap), float(bracket_h)

    shared_y_lims: list[tuple[float, float]] = []
    shared_y_axes: list[Any] = []
    shared_metric_keys = {"all_spike", "ss", "cs"}

    for ax, (metric_key, metric_title, _map_key) in zip(axes[:len(metric_specs)], metric_specs):
        group_ns_for_ticks: list[int] = []
        panel_vals_by_group: list[np.ndarray] = []
        for group_idx, (group_label, _group_key, _color) in enumerate(group_specs):
            vals = values_df[
                (values_df["metric"] == metric_key)
                & (values_df["group"] == group_label)
            ]["r"].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            panel_vals_by_group.append(vals)
            group_ns_for_ticks.append(int(vals.size))
            if vals.size == 0:
                continue
            bp = ax.boxplot(
                [vals],
                positions=[x[group_idx]],
                widths=0.52,
                patch_artist=True,
                showfliers=False,
                boxprops={"facecolor": colors[group_idx], "edgecolor": "black", "linewidth": 0.6},
                medianprops={"color": "black", "linewidth": 0.8},
                whiskerprops={"color": "black", "linewidth": 0.6},
                capprops={"color": "black", "linewidth": 0.6},
                zorder=2,
            )
            for patch in bp.get("boxes", []):
                patch.set_alpha(0.75)
            jitter = rng.uniform(-0.09, 0.09, size=vals.size)
            ax.scatter(
                np.full(vals.size, x[group_idx]) + jitter,
                vals,
                s=8,
                color="black",
                alpha=0.6,
                linewidths=0,
                zorder=3,
            )

        panel_ymin, panel_ymax, bracket_y0, bracket_gap, bracket_h = _panel_axis_layout(
            panel_vals_by_group,
            len(pairwise_specs),
        )
        for bracket_idx, (group1_idx, group2_idx, group1_label, group2_label) in enumerate(pairwise_specs):
            p_sub = pairwise_df[
                (pairwise_df["metric"] == metric_key)
                & (pairwise_df["group1"] == group1_label)
                & (pairwise_df["group2"] == group2_label)
            ]
            sig_text = str(p_sub["significance"].iloc[0]) if len(p_sub) > 0 else "n/a"
            y = bracket_y0 + bracket_idx * bracket_gap
            x1 = x[group1_idx]
            x2 = x[group2_idx]
            ax.plot([x1, x1, x2, x2], [y, y + bracket_h, y + bracket_h, y], color="black", linewidth=0.5)
            ax.text(
                0.5 * (x1 + x2),
                y + bracket_h + 0.01,
                sig_text,
                ha="center",
                va="bottom",
                fontsize=4.5,
                fontname="Arial",
            )
        if draw_zero_line:
            ax.axhline(0, color="black", linewidth=0.4, zorder=1)
        ax.set_title(metric_title, fontsize=6, fontname="Arial")
        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"{label}\nn={n}" for label, n in zip(group_tick_labels, group_ns_for_ticks)],
            fontsize=5,
            fontname="Arial",
        )
        ax.set_ylim(panel_ymin, panel_ymax)
        if str(metric_key) in shared_metric_keys:
            shared_y_lims.append((panel_ymin, panel_ymax))
            shared_y_axes.append(ax)
        ax.tick_params(axis="both", labelsize=5, width=0.5, length=1.75, direction="in")
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    ax_pair = axes[len(metric_specs)]
    pair_x = np.array([0.0, 1.0], dtype=float)
    pair_colors = ["#026C80", "#EE9B00"]
    pair_labels = ["SS", "CS"]
    pair_vals = [ss_paired_vals, cs_paired_vals]
    pair_ymin, pair_ymax, pair_bracket_y, _pair_bracket_gap, pair_bracket_h = _panel_axis_layout(pair_vals, 1)
    for pair_idx, vals in enumerate(pair_vals):
        if vals.size == 0:
            continue
        bp = ax_pair.boxplot(
            [vals],
            positions=[pair_x[pair_idx]],
            widths=0.45,
            patch_artist=True,
            showfliers=False,
            boxprops={"facecolor": pair_colors[pair_idx], "edgecolor": "black", "linewidth": 0.6},
            medianprops={"color": "black", "linewidth": 0.8},
            whiskerprops={"color": "black", "linewidth": 0.6},
            capprops={"color": "black", "linewidth": 0.6},
            zorder=2,
        )
        for patch in bp.get("boxes", []):
            patch.set_alpha(0.75)
        jitter = rng.uniform(-0.055, 0.055, size=vals.size)
        ax_pair.scatter(
            np.full(vals.size, pair_x[pair_idx]) + jitter,
            vals,
            s=8,
            color="black",
            alpha=0.6,
            linewidths=0,
            zorder=3,
        )
    for ss_val, cs_val in zip(ss_paired_vals, cs_paired_vals):
        ax_pair.plot(pair_x, [ss_val, cs_val], color="black", alpha=0.25, linewidth=0.5, zorder=1)
    ax_pair.plot(
        [pair_x[0], pair_x[0], pair_x[1], pair_x[1]],
        [pair_bracket_y, pair_bracket_y + pair_bracket_h, pair_bracket_y + pair_bracket_h, pair_bracket_y],
        color="black",
        linewidth=0.5,
    )
    ax_pair.text(
        0.5 * (pair_x[0] + pair_x[1]),
        pair_bracket_y + pair_bracket_h + 0.02 * max(pair_ymax - pair_ymin, 0.05),
        str(paired_stats_df["significance"].iloc[0]) if len(paired_stats_df) > 0 else "n/a",
        ha="center",
        va="bottom",
        fontsize=4.5,
        fontname="Arial",
    )
    if draw_zero_line:
        ax_pair.axhline(0, color="black", linewidth=0.4, zorder=1)
    ax_pair.set_title("CS+ PLC\nSS vs CS", fontsize=6, fontname="Arial")
    ax_pair.set_xticks(pair_x)
    ax_pair.set_xticklabels([f"{label}\nn={paired_n}" for label in pair_labels], fontsize=5, fontname="Arial")
    ax_pair.set_ylim(pair_ymin, pair_ymax)
    shared_y_lims.append((pair_ymin, pair_ymax))
    shared_y_axes.append(ax_pair)
    ax_pair.tick_params(axis="both", labelsize=5, width=0.5, length=1.75, direction="in")
    for spine in ax_pair.spines.values():
        spine.set_linewidth(0.5)
    ax_pair.spines["top"].set_visible(False)
    ax_pair.spines["right"].set_visible(False)

    ax_final = axes[-1]
    final_x = np.arange(len(combined_final_group_specs), dtype=float)
    final_group_ns_for_ticks: list[int] = []
    final_vals_by_group: list[np.ndarray] = []
    for final_idx, (panel_group, color, _tick_label) in enumerate(combined_final_group_specs):
        vals = combined_final_df[
            combined_final_df.get("panel_group", pd.Series(dtype=object)) == panel_group
        ].get("r", pd.Series(dtype=float)).to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        final_vals_by_group.append(vals)
        final_group_ns_for_ticks.append(int(vals.size))
        if vals.size == 0:
            continue
        bp = ax_final.boxplot(
            [vals],
            positions=[final_x[final_idx]],
            widths=0.52,
            patch_artist=True,
            showfliers=False,
            boxprops={"facecolor": color, "edgecolor": "black", "linewidth": 0.6},
            medianprops={"color": "black", "linewidth": 0.8},
            whiskerprops={"color": "black", "linewidth": 0.6},
            capprops={"color": "black", "linewidth": 0.6},
            zorder=2,
        )
        for patch in bp.get("boxes", []):
            patch.set_alpha(0.75)
        jitter = rng.uniform(-0.09, 0.09, size=vals.size)
        ax_final.scatter(
            np.full(vals.size, final_x[final_idx]) + jitter,
            vals,
            s=8,
            color="black",
            alpha=0.6,
            linewidths=0,
            zorder=3,
        )

    final_ymin, final_ymax, bracket_y0, bracket_gap, bracket_h = _panel_axis_layout(
        final_vals_by_group,
        len(combined_final_pairwise_specs),
    )
    final_group_to_x = {
        panel_group: final_x[idx]
        for idx, (panel_group, _color, _tick_label) in enumerate(combined_final_group_specs)
    }
    if "CS+ SS" in final_group_to_x and "CS+ CS" in final_group_to_x:
        paired_x = [final_group_to_x["CS+ SS"], final_group_to_x["CS+ CS"]]
        for ss_val, cs_val in zip(ss_paired_vals, cs_paired_vals):
            if np.isfinite(ss_val) and np.isfinite(cs_val):
                ax_final.plot(
                    paired_x,
                    [float(ss_val), float(cs_val)],
                    color="black",
                    alpha=0.25,
                    linewidth=0.5,
                    zorder=1,
                )
    for bracket_idx, (_group1_idx, _group2_idx, group1_label, group2_label, _kind) in enumerate(combined_final_pairwise_specs):
        if group1_label not in final_group_to_x or group2_label not in final_group_to_x:
            continue
        p_sub = combined_final_pairwise_df[
            (combined_final_pairwise_df.get("group1", pd.Series(dtype=object)) == group1_label)
            & (combined_final_pairwise_df.get("group2", pd.Series(dtype=object)) == group2_label)
        ]
        sig_text = str(p_sub["significance"].iloc[0]) if len(p_sub) > 0 else "n/a"
        y = bracket_y0 + bracket_idx * bracket_gap
        x1 = final_group_to_x[group1_label]
        x2 = final_group_to_x[group2_label]
        ax_final.plot([x1, x1, x2, x2], [y, y + bracket_h, y + bracket_h, y], color="black", linewidth=0.5)
        ax_final.text(
            0.5 * (x1 + x2),
            y + bracket_h + 0.01,
            sig_text,
            ha="center",
            va="bottom",
            fontsize=4.5,
            fontname="Arial",
        )
    if draw_zero_line:
        ax_final.axhline(0, color="black", linewidth=0.4, zorder=1)
    ax_final.set_title(f"CS+ SS/CS\nvs {final_csminus_label}", fontsize=6, fontname="Arial")
    ax_final.set_xticks(final_x)
    ax_final.set_xticklabels(
        [
            f"{tick_label}\nn={n}"
            for (_panel_group, _color, tick_label), n in zip(
                combined_final_group_specs,
                final_group_ns_for_ticks,
            )
        ],
        fontsize=5,
        fontname="Arial",
    )
    ax_final.set_ylim(final_ymin, final_ymax)
    ax_final.tick_params(axis="both", labelsize=5, width=0.5, length=1.75, direction="in")
    for spine in ax_final.spines.values():
        spine.set_linewidth(0.5)
    ax_final.spines["top"].set_visible(False)
    ax_final.spines["right"].set_visible(False)
    if shared_y_lims:
        shared_ymin = min(lim[0] for lim in shared_y_lims)
        shared_ymax = max(lim[1] for lim in shared_y_lims)
        for ax in shared_y_axes:
            ax.set_ylim(shared_ymin, shared_ymax)
        for ax in shared_y_axes[1:]:
            ax.tick_params(labelleft=False)
        ax_final.tick_params(labelleft=True)
    axes[0].set_ylabel(f"S1 vs S2 {metric_label}", fontsize=6, fontname="Arial")
    fig.tight_layout(w_pad=0.5)

    figure_path = None
    values_csv = None
    summary_csv = None
    pairwise_csv = None
    paired_csv = None
    paired_stats_csv = None
    combined_final_csv = None
    combined_final_pairwise_csv = None
    figure_dir = Path(figure_save_folder)
    figure_dir.mkdir(parents=True, exist_ok=True)
    if save_name:
        figure_path = str(figure_dir / str(save_name))
        plt.rcParams["svg.fonttype"] = "none"
        fig.savefig(figure_path, dpi=300)
        print(f"Saved: {figure_path}")
    if save_csv:
        stem = Path(str(save_name)).stem if save_name else "SessionCompare_S1S2_CorrelationStats"
        values_csv = str(figure_dir / f"{stem}_values.csv")
        summary_csv = str(figure_dir / f"{stem}_summary.csv")
        pairwise_csv = str(figure_dir / f"{stem}_pairwise.csv")
        paired_csv = str(figure_dir / f"{stem}_csplus_ss_cs_paired_values.csv")
        paired_stats_csv = str(figure_dir / f"{stem}_csplus_ss_cs_paired_stats.csv")
        combined_final_csv = str(figure_dir / f"{stem}_csplus_ss_cs_csminus_combined_values.csv")
        combined_final_pairwise_csv = str(figure_dir / f"{stem}_csplus_ss_cs_csminus_combined_pairwise.csv")
        values_df.to_csv(values_csv, index=False)
        summary_df.to_csv(summary_csv, index=False)
        pairwise_df.to_csv(pairwise_csv, index=False)
        paired_df.to_csv(paired_csv, index=False)
        paired_stats_df.to_csv(paired_stats_csv, index=False)
        combined_final_df.to_csv(combined_final_csv, index=False)
        combined_final_pairwise_df.to_csv(combined_final_pairwise_csv, index=False)

    if show_plot:
        plt.show()

    return {
        "fig": fig,
        "values_df": values_df,
        "summary_df": summary_df,
        "pairwise_df": pairwise_df,
        "paired_df": paired_df,
        "paired_stats_df": paired_stats_df,
        "combined_final_df": combined_final_df,
        "combined_final_pairwise_df": combined_final_pairwise_df,
        "combined_final_csminus_metric": final_csminus_metric,
        "combined_final_csminus_label": final_csminus_label,
        "combined_final_csminus_tick_label": final_csminus_tick_label,
        "baseline_subtraction_cosine": bool(target_baseline_cosine),
        "figure_path": figure_path,
        "values_csv": values_csv,
        "summary_csv": summary_csv,
        "pairwise_csv": pairwise_csv,
        "paired_csv": paired_csv,
        "paired_stats_csv": paired_stats_csv,
        "combined_final_csv": combined_final_csv,
        "combined_final_pairwise_csv": combined_final_pairwise_csv,
    }


def render_session_compare_heatmaps(
    config: PipelineConfig,
    params: SessionCompareParams,
    groups_payload: dict[str, Any],
    figure_save_folder: str | os.PathLike[str] | None = None,
    show_occupancy_spearman: bool = True,
    show_occupancy_heatmap: bool = True,
    similarity_metric: str | None = "pearson",
) -> dict[str, Any]:
    if figure_save_folder is None:
        figure_save_folder = config.figures_root / "CKII_pooled"
    else:
        figure_save_folder = Path(figure_save_folder)
    figure_save_folder.mkdir(parents=True, exist_ok=True)

    similarity_metric_source = params.heatmap_similarity_metric if similarity_metric is None else similarity_metric
    similarity_metric_mode, similarity_metric_label = _normalize_session_compare_similarity_metric(similarity_metric_source)
    similarity_metric_label = _similarity_label_with_weight(similarity_metric_label, bool(params.weighted))

    panel_suffix = "combined_s1_s2" if params.panel_mode == "combined_s1_s2" else "s1_s2"
    session_split_classification = groups_payload.get("session_split_classification", None)
    if not isinstance(session_split_classification, dict):
        session_split_classification = summarize_session_split_plc_classification(config, groups_payload)

    occupancy_spearman_by_dataset: dict[str, float] = {}
    occupancy_df = groups_payload.get("occupancy_similarity_df", None)
    if isinstance(occupancy_df, pd.DataFrame) and {"dataset_id", "occupancy_spearman"}.issubset(occupancy_df.columns):
        for _, row in occupancy_df.iterrows():
            dataset_id = str(row.get("dataset_id", ""))
            if not dataset_id:
                continue
            try:
                occupancy_spearman_by_dataset[dataset_id] = float(row.get("occupancy_spearman", np.nan))
            except (TypeError, ValueError):
                occupancy_spearman_by_dataset[dataset_id] = np.nan

    def _chunk_groups(groups: list[list[dict[str, Any]]], max_cells: int):
        if max_cells <= 0:
            yield 1, groups
            return
        n = len(groups)
        if n == 0:
            return
        part = 1
        for i in range(0, n, max_cells):
            yield part, groups[i:i + max_cells]
            part += 1

    def _render_group_set(groups: list[list[dict[str, Any]]], label: str, filename_base: str) -> list[str]:
        n_cells = len(groups)
        if n_cells == 0:
            print(f"Skipping {label}: no cells")
            return []

        chunks = list(_chunk_groups(groups, int(params.max_cells_per_figure)))
        n_parts = len(chunks)
        print(f"Plotting {label} comparison (n={n_cells}, {len(groups_payload.get('conditions', []))} conditions each)")

        saved_paths: list[str] = []
        for part_idx, chunk in chunks:
            out_name = f"{filename_base}.svg" if n_parts == 1 else f"{filename_base}_part{part_idx:02d}.svg"
            out_path = figure_save_folder / out_name
            plot_putative_pf = str(label).strip().lower() != "non-plc"
            print(f"  Rendering {label} part {part_idx}/{n_parts}: n={len(chunk)} -> {out_name}")
            _ = plot_session_compare_selected_cells_figure(
                cell_groups=chunk,
                theta_vlim=groups_payload.get("selected_theta_vlim", None),
                slow_vlim=groups_payload.get("selected_slow_vlim", None),
                save_path=str(out_path),
                plot_putative_PF=plot_putative_pf,
                pf_only_place_cells=False,
                show_place_cell_star=True,
                show_significance_marker=True,
                overlay_similarity_metric=str(similarity_metric_mode),
                clean_heatmap=bool(params.clean_heatmap),
                plot_spike_shapes=bool(params.plot_spike_shapes),
                plot_spike_shapes_overall=bool(params.plot_spike_shapes_overall),
                plot_spike_shapes_in_field=bool(params.plot_spike_shapes_in_field),
                plot_spike_shapes_out_field=bool(params.plot_spike_shapes_out_field),
                plot_PF_combined=bool(params.plot_PF_combined),
                include_plateau=bool(params.include_plateau),
                plateau_state_mode=str(params.plateau_state_mode),
                plateau_include_long_cb_as_plateau=bool(params.plateau_include_long_cb_as_plateau),
                plateau_cb_min_duration_ms=float(params.plateau_cb_min_duration_ms),
                plateau_speed_threshold=float(params.plateau_speed_threshold),
                plateau_data_folder=str(config.data_root),
                two_session_split_mode=params.two_session_split_mode,
                two_session_split_window_minutes=params.two_session_split_window_minutes,
                show_split_session_plc_star=True,
                split_session_cb_num_threshold=int(config.pooled.cb_num_threshold),
                split_session_cs_peak_rate_threshold=float(config.pooled.cs_peak_rate_threshold),
                split_session_cs_plc_definition_mode=str(config.pooled.cs_plc_definition_mode),
                behavior_cleaning_config=config,
                show_shape_counts=False,
                show_occupancy_spearman=bool(show_occupancy_spearman),
                show_occupancy_heatmap=bool(show_occupancy_heatmap),
                occupancy_spearman_by_dataset=occupancy_spearman_by_dataset,
                weighted=bool(params.weighted),
                baseline_subtraction_cosine=bool(getattr(params, "baseline_subtraction_cosine", True)),
                min_occupancy_per_bin_s=float(getattr(params, "s1s2_min_occupancy_per_bin_s", MIN_OCCUPANCY_WEIGHT)),
                subplot_width=0.5,
            )
            saved_paths.append(str(out_path))
        return saved_paths

    saved_csplus = _render_group_set(groups_payload.get("csplus_groups", []), "CS+", f"spatial_compare_csplus_{panel_suffix}")
    saved_csminus = _render_group_set(groups_payload.get("csminus_groups", []), "CS-", f"spatial_compare_csminus_{panel_suffix}")
    saved_nonplc = _render_group_set(groups_payload.get("nonpc_groups", []), "Non-PLC", f"spatial_compare_nonplc_{panel_suffix}")

    summary = {
        "figure_save_folder": str(figure_save_folder),
        "panel_mode": params.panel_mode,
        "saved_csplus": saved_csplus,
        "saved_csminus": saved_csminus,
        "saved_nonplc": saved_nonplc,
        "n_csplus_rows": int(len(groups_payload.get("csplus_groups", []))),
        "n_csminus_rows": int(len(groups_payload.get("csminus_groups", []))),
        "n_nonplc_rows": int(len(groups_payload.get("nonpc_groups", []))),
        "n_csplus_missing_s2": int(groups_payload.get("csplus_missing_s2", 0)),
        "n_csminus_missing_s2": int(groups_payload.get("csminus_missing_s2", 0)),
        "n_nonplc_missing_s2": int(groups_payload.get("nonpc_missing_s2", 0)),
        "session_split_classification": session_split_classification,
        "session_split_classification_summary_df": session_split_classification["summary_df"],
        "nonplc_becomes_plc_df": session_split_classification["nonplc_becomes_plc_df"],
        "show_occupancy_spearman": bool(show_occupancy_spearman),
        "show_occupancy_heatmap": bool(show_occupancy_heatmap),
        "similarity_metric": str(similarity_metric_mode),
        "similarity_label": str(similarity_metric_label),
        "weighted": bool(params.weighted),
        "baseline_subtraction_cosine": bool(getattr(params, "baseline_subtraction_cosine", True)),
        "s1s2_min_occupancy_per_bin_s": float(getattr(params, "s1s2_min_occupancy_per_bin_s", MIN_OCCUPANCY_WEIGHT)),
    }
    return summary


def run_session_compare_heatmaps(
    config: PipelineConfig,
    params: SessionCompareParams,
    spatial_data: Any | None = None,
) -> dict[str, Any]:
    analysis = build_session_compare_analysis(config, params, spatial_data=spatial_data)
    payloads = analysis["payloads"]
    groups = analysis["groups"]
    rendered = render_session_compare_heatmaps(config, params, groups)

    out = {
        "payloads": payloads,
        "groups": groups,
        "session_split_4panel_tables": analysis["session_split_4panel_tables"],
        "session_split_classification": analysis["session_split_classification"],
        "render": rendered,
    }
    return out


def plot_session_compare_selected_cells_figure(
    cell_groups,
    group_names=None,
    subplot_width=0.5,
    gap_ratio=0.1,
    theta_vlim=None,
    slow_vlim=None,
    save_path=None,
    show_scale_bar=True,
    pf_only_place_cells=True,
    plot_putative_PF=False,
    show_place_cell_star=True,
    show_significance_marker=True,
    show_weighted_map_r=True,
    overlay_similarity_metric='pearson',
    weighted=False,
    baseline_subtraction_cosine=True,
    min_occupancy_per_bin_s=0.5,
    clean_heatmap=True,
    plot_spike_shapes=True,
    plot_spike_shapes_overall=True,
    plot_spike_shapes_in_field=True,
    plot_spike_shapes_out_field=True,
    plot_PF_combined=True,
    min_shapes_per_condition=None,
    min_shapes_per_condition_ss=5,
    min_shapes_per_condition_cb=3,
    spike_shape_state='run',
    show_shape_counts=False,
    shape_ylim=None,
    include_plateau=True,
    plateau_state_mode='all',
    plateau_include_long_cb_as_plateau=False,
    plateau_cb_min_duration_ms=200.0,
    plateau_speed_threshold=3.0,
    plateau_data_folder=None,
    two_session_split_mode='recorded_sessions',
    two_session_split_window_minutes=None,
    show_split_session_plc_star=False,
    split_session_cb_num_threshold=10,
    split_session_cs_peak_rate_threshold=0.5,
    split_session_cs_plc_definition_mode="legacy",
    behavior_cleaning_config=None,
    show_occupancy_spearman=True,
    show_occupancy_heatmap=True,
    occupancy_spearman_by_dataset=None,
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
        Width of each map subplot (inches). Fixed to 0.5 by default.
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
    show_weighted_map_r : bool
        If True (default), show Session1-vs-Session2 similarity above maps.
    overlay_similarity_metric : str
        Similarity metric for S1-vs-S2 map overlays.
        Options: 'pearson'/'weighted_pearson' (default), 'spearman', 'cosine'.
    weighted : bool
        If True, weight Pearson, Spearman, and cosine similarities by the minimum
        S1/S2 occupancy per spatial bin.
    clean_heatmap : bool
        If True (default), draw a clean heatmap: keep only max firing-rate labels and weighted r text.
        Hide star/significance markers and theta/slow numeric annotations.
    plot_spike_shapes : bool
        Master switch for waveform rows.
    plot_spike_shapes_overall : bool
        If True (default), show two overall waveform rows (SS and CB; each uses in+out combined),
        and hide split In-PF / Out-PF rows.
    plot_spike_shapes_in_field : bool
        If True, include the In-PF waveform row (used when overall=False).
    plot_spike_shapes_out_field : bool
        If True, include the Out-PF waveform row (used when overall=False).
    plot_PF_combined : bool
        If True (default), PF contours are taken from the combined-session analysis
        for the same animal/cell and overlaid on all condition panels.
        If False, each condition uses its own PF contours.
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
    show_split_session_plc_star : bool
        If True, display colored CS+/CS- PLC star labels beside the all-spike peak-rate text.
    split_session_cb_num_threshold : int
        CB run-in threshold for split-session CS+ PLC marker classification.
    split_session_cs_peak_rate_threshold : float
        CS peak-rate threshold for split-session CS+ PLC marker classification.
    show_occupancy_spearman : bool
        If True, display per-cell S1/S2 occupancy Spearman above the first row.
    show_occupancy_heatmap : bool
        If True, display a first row with the plotted cell/session occupancy heatmap.
    occupancy_spearman_by_dataset : dict or None
        Mapping from dataset id to occupancy Spearman value; used only as a fallback
        when per-cell Session 1/Session 2 occupancy maps are unavailable.
    
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

    show_significance_marker_effective = bool(show_significance_marker) and (not bool(clean_heatmap))
    show_place_cell_star_effective = bool(show_place_cell_star) and (not bool(clean_heatmap))
    show_theta_slow_numbers = not bool(clean_heatmap)
    show_occupancy_heatmap = bool(show_occupancy_heatmap)
    show_occupancy_spearman_effective = bool(show_occupancy_spearman)

    plot_spike_shapes_overall = bool(plot_spike_shapes_overall)
    plot_spike_shapes_in_field = bool(plot_spike_shapes_in_field)
    plot_spike_shapes_out_field = bool(plot_spike_shapes_out_field)
    if not bool(plot_spike_shapes):
        plot_spike_shapes_overall = False
        plot_spike_shapes_in_field = False
        plot_spike_shapes_out_field = False
    if plot_spike_shapes_overall:
        plot_spike_shapes_in_field = False
        plot_spike_shapes_out_field = False
    plot_spike_shapes_any = bool(plot_spike_shapes_overall or plot_spike_shapes_in_field or plot_spike_shapes_out_field)

    include_plateau = bool(include_plateau)
    plateau_state_mode = str(plateau_state_mode).strip().lower()
    valid_plateau_modes = {'split', 'all', 'moving', 'resting'}
    if plateau_state_mode not in valid_plateau_modes:
        raise ValueError("plateau_state_mode must be one of {'split', 'all', 'moving', 'resting'}.")
    if (not np.isfinite(float(plateau_cb_min_duration_ms))) or float(plateau_cb_min_duration_ms) <= 0:
        raise ValueError("plateau_cb_min_duration_ms must be a finite number > 0.")
    if not np.isfinite(float(plateau_speed_threshold)):
        raise ValueError("plateau_speed_threshold must be a finite number.")
    if plateau_data_folder is not None:
        plateau_data_folder = os.path.abspath(str(plateau_data_folder))
    two_session_split_mode = _normalize_two_session_split_mode(two_session_split_mode)
    two_session_split_window_minutes = _validate_two_session_split_window_minutes(
        two_session_split_mode,
        two_session_split_window_minutes,
    )
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
    plot_plateau_shapes = bool(include_plateau and plot_spike_shapes)

    if plot_spike_shapes_overall:
        n_shape_rows = 2  # separate rows: overall SS and overall CB
    else:
        n_shape_rows = int(plot_spike_shapes_in_field) + int(plot_spike_shapes_out_field)

    occupancy_row = 0 if show_occupancy_heatmap else None
    row_offset = 1 if show_occupancy_heatmap else 0
    trajectory_row = row_offset + 0
    rate_row = row_offset + 1
    ss_row = row_offset + 2
    cs_row = row_offset + 3
    theta_row = row_offset + 4
    slow_row = row_offset + 5
    base_rows = row_offset + 6  # optional occupancy, trajectory, rate map, SS, CS, theta, slow
    map_rows = base_rows + n_plateau_rows
    if plot_plateau_shapes:
        n_shape_rows += 1
    n_rows = map_rows + n_shape_rows
    
    # Calculate width ratios
    group_col_indices = {}
    if use_gaps:
        # With gaps between groups
        width_ratios = []
        col_to_cell = {}
        gap_cols = []
        cell_idx = 0
        col_idx = 0
        for g_idx, n_cells in enumerate(cells_per_group):
            group_cols = []
            for _ in range(n_cells):
                width_ratios.append(1)
                col_to_cell[col_idx] = cell_idx
                group_cols.append(col_idx)
                col_idx += 1
                cell_idx += 1
            group_col_indices[g_idx] = group_cols
            if g_idx < n_groups - 1:  # Add gap after each group except last
                width_ratios.append(gap_ratio)
                gap_cols.append(col_idx)
                col_idx += 1
    else:
        # No gaps - simple grid
        width_ratios = [1] * total_cells
        col_to_cell = {i: i for i in range(total_cells)}
        gap_cols = []

        start_col = 0
        for g_idx, n_cells in enumerate(cells_per_group):
            group_col_indices[g_idx] = list(range(start_col, start_col + n_cells))
            start_col += n_cells
    
    n_cols = len(width_ratios)
    
    # Find the last/first data column for colorbars/labels
    last_data_col = max(col_to_cell.keys())
    first_data_col = min(col_to_cell.keys())
    
    # Row indices for optional spike-shape panels
    shape_rows = []
    if plot_spike_shapes_overall:
        shape_rows.append(('overall_ss', map_rows + len(shape_rows)))
        shape_rows.append(('overall_cb', map_rows + len(shape_rows)))
    else:
        if plot_spike_shapes_in_field:
            shape_rows.append(('in', map_rows + len(shape_rows)))
        if plot_spike_shapes_out_field:
            shape_rows.append(('out', map_rows + len(shape_rows)))
    shape_row_lookup = {k: r for k, r in shape_rows}
    shape_row_overall_ss = shape_row_lookup.get('overall_ss', None)
    shape_row_overall_cb = shape_row_lookup.get('overall_cb', None)
    shape_row_in = shape_row_lookup.get('in', None)
    shape_row_out = shape_row_lookup.get('out', None)
    shape_anchor_row = shape_rows[0][1] if len(shape_rows) > 0 else None
    plateau_row_by_mode = {
        mode: (base_rows + idx) for idx, (mode, _) in enumerate(plateau_row_modes)
    }
    plateau_shape_row = map_rows + len(shape_rows) if plot_plateau_shapes else None
    
    # Get arena dimensions from first cell
    params = all_cells[0]['params']
    width_real = params['width_real']
    height_real = params['height_real']
    arena_aspect = height_real / width_real
    
    # Calculate figure dimensions from fixed per-subplot width
    left_margin = 0.4
    right_margin = 0.3
    top_margin = 0.37 if show_occupancy_spearman_effective else 0.25
    bottom_margin = 0.1

    total_width_units = float(sum(width_ratios))
    cell_width = float(subplot_width)
    cell_height = cell_width * arena_aspect

    row_gap = 0.14
    fig_width = left_margin + right_margin + total_width_units * cell_width
    fig_height = n_rows * cell_height + (n_rows - 1) * row_gap + top_margin + bottom_margin

    # Create figure
    fig = plt.figure(figsize=(fig_width, fig_height))
    hspace = (row_gap / cell_height) if cell_height > 0 else 0.15
    gs = GridSpec(n_rows, n_cols, figure=fig, width_ratios=width_ratios,
                  left=left_margin/fig_width, right=1-right_margin/fig_width,
                  top=1-top_margin/fig_height, bottom=bottom_margin/fig_height,
                  wspace=0.0, hspace=hspace)
    
    # Create axes for data columns only (skip gap columns)
    axes_grid = {}
    
    # Base rows (maps + optional plateau rows)
    for row in range(map_rows):
        for col in range(n_cols):
            if col in gap_cols:
                continue
            axes_grid[(row, col)] = fig.add_subplot(gs[row, col])
    
    # Optional spike-shape rows (share x/y within each shape row across columns)
    if plot_spike_shapes_any:
        row_anchor_axes = {}
        for _pf_key, row_idx in shape_rows:
            row_anchor_axes[row_idx] = fig.add_subplot(gs[row_idx, first_data_col])
            axes_grid[(row_idx, first_data_col)] = row_anchor_axes[row_idx]

        for col in range(n_cols):
            if col in gap_cols or col == first_data_col:
                continue
            for _pf_key, row_idx in shape_rows:
                anchor_ax = row_anchor_axes[row_idx]
                axes_grid[(row_idx, col)] = fig.add_subplot(
                    gs[row_idx, col], sharex=anchor_ax, sharey=anchor_ax
                )

    # Optional plateau-shape row (all plateau traces, no averaging)
    if plot_plateau_shapes and plateau_shape_row is not None:
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
    csplus_plc_color = "#D81B60"
    csminus_plc_color = "#1F77B4"
    
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
        ax.contour(padded_mask.T, levels=[0.5], colors=color, linewidths=linewidth,
                   linestyles=linestyle, extent=padded_extent, origin="lower", alpha=alpha)

    def _included_minutes_for_title(cell: dict[str, Any]) -> float:
        if not isinstance(cell, dict):
            return np.nan
        for key in ("included_minutes", "included_time_min"):
            try:
                value = float(cell.get(key, np.nan))
            except (TypeError, ValueError):
                value = np.nan
            if np.isfinite(value):
                return value

        try:
            kept_frames = float(cell.get("n_frames_kept_total", np.nan))
            frame_rate = float(cell.get("frame_rate", np.nan))
        except (TypeError, ValueError):
            kept_frames = np.nan
            frame_rate = np.nan
        if np.isfinite(kept_frames) and np.isfinite(frame_rate) and frame_rate > 0:
            return kept_frames / frame_rate / 60.0

        label = str(cell.get("condition_label", "")).lower()
        if (
            _normalize_two_session_split_mode(two_session_split_mode) == "time_windows"
            and ("session 1" in label or "session 2" in label)
            and two_session_split_window_minutes is not None
        ):
            try:
                return float(two_session_split_window_minutes)
            except (TypeError, ValueError):
                return np.nan
        return np.nan

    def _condition_label_for_title(cell: dict[str, Any]) -> str:
        condition_label = str(cell.get("condition_label", "")).strip()
        if not condition_label:
            return ""
        if "n/a" in condition_label.lower():
            return condition_label
        minutes = _included_minutes_for_title(cell)
        if not np.isfinite(minutes):
            return condition_label
        return f"{condition_label} ({int(round(float(minutes)))} min)"

    def _cell_title(cell: dict[str, Any], animal_short: str, cell_num: int) -> str:
        title_str = f"{animal_short}\nCell {cell_num}"
        condition_label = _condition_label_for_title(cell)
        if condition_label:
            title_str += f"\n{condition_label}"
        return title_str

    def _dataset_id_from_group(g_cells) -> str:
        if not isinstance(g_cells, (list, tuple)):
            return ""
        for c in g_cells:
            if not isinstance(c, dict):
                continue
            dataset_id = str(c.get("animal_id", c.get("session", ""))).strip()
            if dataset_id:
                return dataset_id
        return ""

    def _occupancy_spearman_from_group(g_cells) -> float:
        if not isinstance(g_cells, (list, tuple)):
            return np.nan
        c1 = None
        c2 = None
        for c in g_cells:
            if not isinstance(c, dict):
                continue
            cond_idx = _condition_index_from_label(c)
            if cond_idx == 1:
                c1 = c
            elif cond_idx == 2:
                c2 = c
        if (c1 is None or c2 is None) and len(g_cells) == 2:
            c1 = g_cells[0] if isinstance(g_cells[0], dict) else c1
            c2 = g_cells[1] if isinstance(g_cells[1], dict) else c2
        elif (c1 is None or c2 is None) and len(g_cells) >= 3:
            c1 = g_cells[1] if isinstance(g_cells[1], dict) else c1
            c2 = g_cells[2] if isinstance(g_cells[2], dict) else c2
        if not isinstance(c1, dict) or not isinstance(c2, dict) or bool(c2.get('is_na_panel', False)):
            return np.nan
        val, _n_valid = _spearman_nan_safe(
            c1.get('occupancy', None),
            c2.get('occupancy', None),
            min_valid_bins=int(globals().get('MIN_VALID_BINS_FOR_DISTANCE', 20)),
        )
        return float(val) if np.isfinite(val) else np.nan

    def _draw_occupancy_spearman_labels() -> None:
        if not show_occupancy_spearman_effective:
            return
        for g_idx, g_cells in enumerate(cell_groups):
            dataset_id = _dataset_id_from_group(g_cells)
            val = _occupancy_spearman_from_group(g_cells)
            if (not np.isfinite(val)) and isinstance(occupancy_spearman_by_dataset, dict) and dataset_id:
                val = occupancy_spearman_by_dataset.get(dataset_id, np.nan)
            text = (
                f"occupancy_spearman = {float(val):.2f}"
                if np.isfinite(val)
                else "occupancy_spearman = n/a"
            )
            label_row = 0
            cols = [col for col in group_col_indices.get(g_idx, []) if (label_row, col) in axes_grid]
            if not cols:
                continue
            bboxes = [axes_grid[(label_row, col)].get_position() for col in cols]
            x0 = min(bb.x0 for bb in bboxes)
            x1 = max(bb.x1 for bb in bboxes)
            y1 = max(bb.y1 for bb in bboxes)
            fig.text(
                0.5 * (x0 + x1),
                min(0.995, y1 + 0.045),
                text,
                ha="center",
                va="bottom",
                fontsize=4.5,
                fontname="Arial",
            )

    def _split_session_plc_marker(cell: dict[str, Any]) -> tuple[str, str] | None:
        label = _split_session_plc_label(
            cell,
            cb_num_threshold=int(split_session_cb_num_threshold),
            cs_peak_rate_threshold=float(split_session_cs_peak_rate_threshold),
            cs_plc_definition_mode=str(split_session_cs_plc_definition_mode),
        )
        if label == "CS+ PLC":
            return "CS+", csplus_plc_color
        if label == "CS- PLC":
            return "CS-", csminus_plc_color
        return None

    def _draw_split_session_plc_marker(ax, cell: dict[str, Any]) -> None:
        if not bool(show_split_session_plc_star):
            return
        marker = _split_session_plc_marker(cell)
        if marker is None:
            return
        marker_label, marker_color = marker
        ax.plot(
            0.02,
            -0.055,
            marker='*',
            markersize=4.5,
            color=marker_color,
            transform=ax.transAxes,
            clip_on=False,
            zorder=5,
        )
        ax.text(
            0.105,
            -0.02,
            marker_label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=4,
            fontname="Arial",
            color=marker_color,
            clip_on=False,
        )

    def _sanitize_pf_components(components, fallback_mask, rate_map=None, max_components=2):
        clean = []
        fallback_shape = None
        if fallback_mask is not None and isinstance(fallback_mask, np.ndarray) and fallback_mask.ndim == 2:
            fallback_shape = fallback_mask.shape

        if isinstance(components, (list, tuple)):
            for rank_idx, comp in enumerate(components):
                arr = None
                peak_rate = np.nan
                if isinstance(comp, dict):
                    arr_raw = comp.get("mask", None)
                    if arr_raw is not None:
                        arr = np.asarray(arr_raw, dtype=bool)
                    peak_rate = float(comp.get("peak_rate", np.nan))
                else:
                    arr = np.asarray(comp, dtype=bool)
                if arr is None or arr.ndim != 2:
                    continue
                if fallback_shape is not None and arr.shape != fallback_shape:
                    continue
                if not np.any(arr):
                    continue
                if (not np.isfinite(peak_rate)) and isinstance(rate_map, np.ndarray) and rate_map.shape == arr.shape:
                    vals = np.asarray(rate_map, dtype=float)[arr]
                    if vals.size > 0 and np.any(np.isfinite(vals)):
                        peak_rate = float(np.nanmax(vals))
                clean.append({
                    "mask": np.asarray(arr, dtype=bool),
                    "peak_rate": peak_rate,
                    "source_rank": int(rank_idx),
                    "area_bins": int(np.sum(arr)),
                })

        if len(clean) > 0:
            # Match the distance-defined pipeline: PF1 by peak, secondary PFs by area.
            finite_peak_items = [item for item in clean if np.isfinite(item["peak_rate"])]
            if len(finite_peak_items) > 0:
                primary = min(
                    finite_peak_items,
                    key=lambda item: (-float(item["peak_rate"]), -int(item["area_bins"]), int(item["source_rank"])),
                )
                primary_id = id(primary)
                remaining = [item for item in clean if id(item) != primary_id]
                remaining.sort(
                    key=lambda item: (
                        -int(item["area_bins"]),
                        -float(item["peak_rate"]) if np.isfinite(item["peak_rate"]) else np.inf,
                        int(item["source_rank"]),
                    )
                )
                clean = [primary] + remaining
            else:
                clean.sort(key=lambda item: (-int(item["area_bins"]), int(item["source_rank"])))
            return [item["mask"] for item in clean[:max(0, int(max_components))]]

        if isinstance(fallback_mask, np.ndarray) and fallback_mask.ndim == 2 and np.any(fallback_mask):
            mask = np.asarray(fallback_mask, dtype=bool)
            # Fallback for older caches lacking ranked components.
            if scipy_ndimage is not None:
                structure = np.ones((3, 3), dtype=int)
                labeled, n_comp = scipy_ndimage.label(mask, structure=structure)
                if int(n_comp) > 0:
                    comps = [(labeled == i) for i in range(1, int(n_comp) + 1)]
                    if isinstance(rate_map, np.ndarray) and rate_map.shape == mask.shape:
                        def _comp_peak(comp_mask):
                            vals = np.asarray(rate_map, dtype=float)
                            vv = vals[comp_mask]
                            return float(np.nanmax(vv)) if vv.size > 0 and np.any(np.isfinite(vv)) else -np.inf
                        comps = sorted(comps, key=_comp_peak, reverse=True)
                    else:
                        comps = sorted(comps, key=lambda c: int(np.sum(c)), reverse=True)
                    return [np.asarray(c, dtype=bool) for c in comps[:max(0, int(max_components))] if np.any(c)]
            return [mask]

        return []

    def _plot_pf_components(ax, components, linewidth=0.6, linestyle='solid', alpha=1.0):
        if not isinstance(components, (list, tuple)) or len(components) == 0:
            return
        for i, comp in enumerate(list(components)[:2]):
            color = "magenta" if i == 0 else "cyan"
            _plot_pf_contour(ax, comp, color, linewidth=linewidth, linestyle=linestyle, alpha=alpha)
    
    def _get_sig_marker(p_val):
        if p_val < 0.001: return "***"
        elif p_val < 0.01: return "**"
        elif p_val < 0.05: return "*"
        else: return ""
    
    def _add_colorbar(ax, im, ticks=None, ticklabels=None):
        cax = inset_axes(ax, width="5%", height="100%", loc="center right",
                         bbox_to_anchor=(0.12, 0.0, 1, 1), bbox_transform=ax.transAxes, borderpad=0)
        cbar = ax.figure.colorbar(im, cax=cax)
        if ticks is not None and ticklabels is not None:
            cbar.set_ticks(ticks)
            cbar.set_ticklabels(ticklabels)
        cbar.ax.tick_params(labelsize=5)
        return cbar
    
    # Prepare shared axes and global limits for spike-shape rows
    shape_xlim = None
    _shape_ylim_user = shape_ylim  # preserve user override
    shape_ylim = None
    shape_ylim_ss = None
    shape_ylim_cb = None
    shape_gap_ms = None
    shape_ss_x_scale = None
    shape_cb_x_scale = None
    shape_ss_x_start = None
    shape_ss_x_end = None
    shape_cb_x_start = None
    shape_xlim_ss_full = None
    shape_xlim_cb_full = None
    if plot_spike_shapes_any:
        ss_x_min = np.inf
        ss_x_max = -np.inf
        cb_x_min = np.inf
        cb_x_max = -np.inf
        y_min = np.inf
        y_max = -np.inf
        ss_y_min = np.inf
        ss_y_max = -np.inf
        cb_y_min = np.inf
        cb_y_max = -np.inf
        
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
                        lo = float(np.nanmin(low))
                        y_min = min(y_min, lo)
                        if spike_type == 'simple':
                            ss_y_min = min(ss_y_min, lo)
                        else:
                            cb_y_min = min(cb_y_min, lo)
                    if np.any(np.isfinite(high)):
                        hi = float(np.nanmax(high))
                        y_max = max(y_max, hi)
                        if spike_type == 'simple':
                            ss_y_max = max(ss_y_max, hi)
                        else:
                            cb_y_max = max(cb_y_max, hi)
        
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
        def _auto_ylim(vmin, vmax, fallback=(-0.2, 1.2)):
            if np.isfinite(vmin) and np.isfinite(vmax):
                if vmax == vmin:
                    eps = 1e-3
                    return (vmin - eps, vmax + eps)
                return (vmin, vmax)
            return fallback

        _shape_ylim_auto = _auto_ylim(y_min, y_max)
        _shape_ylim_ss_auto = _auto_ylim(ss_y_min, ss_y_max, fallback=_shape_ylim_auto)
        _shape_ylim_cb_auto = _auto_ylim(cb_y_min, cb_y_max, fallback=_shape_ylim_auto)

        # Clamp the lower bound to keep rows compact; only SS gets fixed top at 1.1.
        _shape_ylim_auto = (max(_shape_ylim_auto[0], -0.3), _shape_ylim_auto[1])
        _shape_ylim_ss_auto = (max(_shape_ylim_ss_auto[0], -0.3), 1.1)
        _shape_ylim_cb_auto = (max(_shape_ylim_cb_auto[0], -0.5), _shape_ylim_cb_auto[1])

        # Use user-supplied limits if provided; otherwise use auto per-row limits.
        if _shape_ylim_user is not None:
            shape_ylim = _shape_ylim_user
            shape_ylim_ss = _shape_ylim_user
            shape_ylim_cb = _shape_ylim_user
        else:
            shape_ylim = _shape_ylim_auto
            shape_ylim_ss = _shape_ylim_ss_auto
            shape_ylim_cb = _shape_ylim_cb_auto
        
        # Shared axes are established via `sharex/sharey` when creating subplots.
        if shape_anchor_row is not None:
            axes_grid[(shape_anchor_row, first_data_col)].set_xlim(*shape_xlim)
            axes_grid[(shape_anchor_row, first_data_col)].set_ylim(*shape_ylim)
    
    def _condition_index_from_label(cell):
        lbl = str(cell.get('condition_label', '')).lower()
        if 'combined' in lbl:
            return 0
        if 'session 1' in lbl:
            return 1
        if 'session 2' in lbl:
            return 2
        return None

    def _weighted_map_corr(map1, map2, occ1, occ2, min_valid_bins=20, min_eff_bins=20, min_occ=0.0):
        if not isinstance(map1, np.ndarray) or not isinstance(map2, np.ndarray):
            return np.nan
        if not isinstance(occ1, np.ndarray) or not isinstance(occ2, np.ndarray):
            return np.nan
        if map1.shape != map2.shape or map1.shape != occ1.shape or map1.shape != occ2.shape:
            return np.nan

        x = np.asarray(map1, dtype=float).ravel()
        y = np.asarray(map2, dtype=float).ravel()
        w = np.minimum(np.asarray(occ1, dtype=float).ravel(), np.asarray(occ2, dtype=float).ravel())

        valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > float(min_occ))
        if int(np.sum(valid)) < int(min_valid_bins):
            return np.nan

        x = x[valid]
        y = y[valid]
        w = w[valid]
        sw = float(np.sum(w))
        if sw <= 0:
            return np.nan

        mx = float(np.sum(w * x) / sw)
        my = float(np.sum(w * y) / sw)
        dx = x - mx
        dy = y - my

        den_x = float(np.sum(w * dx * dx))
        den_y = float(np.sum(w * dy * dy))
        den = np.sqrt(den_x * den_y)
        if den <= 0:
            return np.nan

        sw2 = float(np.sum(w * w))
        n_eff = (sw * sw / sw2) if sw2 > 0 else 0.0
        if n_eff < float(min_eff_bins):
            return np.nan

        return float(np.sum(w * dx * dy) / den)

    def _spearman_map_corr(map1, map2, min_valid_bins=20):
        if not isinstance(map1, np.ndarray) or not isinstance(map2, np.ndarray):
            return np.nan
        if map1.shape != map2.shape:
            return np.nan

        a = np.asarray(map1, dtype=float).ravel()
        b = np.asarray(map2, dtype=float).ravel()
        valid = np.isfinite(a) & np.isfinite(b)
        if int(np.sum(valid)) < int(min_valid_bins):
            return np.nan

        x = a[valid]
        y = b[valid]
        if np.nanstd(x) == 0 or np.nanstd(y) == 0:
            return np.nan

        scipy_stats_local = globals().get('scipy_stats', None)
        if scipy_stats_local is not None:
            try:
                rho, _p = scipy_stats_local.spearmanr(x, y)
                return float(rho) if np.isfinite(rho) else np.nan
            except Exception:
                return np.nan

        rx = pd.Series(x).rank(method='average').to_numpy(dtype=float)
        ry = pd.Series(y).rank(method='average').to_numpy(dtype=float)
        if np.nanstd(rx) == 0 or np.nanstd(ry) == 0:
            return np.nan
        return float(np.corrcoef(rx, ry)[0, 1])

    def _cosine_map_corr(map1, map2, min_valid_bins=20):
        if not isinstance(map1, np.ndarray) or not isinstance(map2, np.ndarray):
            return np.nan
        if map1.shape != map2.shape:
            return np.nan

        a = np.asarray(map1, dtype=float).ravel()
        b = np.asarray(map2, dtype=float).ravel()
        valid = np.isfinite(a) & np.isfinite(b)
        if int(np.sum(valid)) < int(min_valid_bins):
            return np.nan

        x = a[valid]
        y = b[valid]
        nx = float(np.linalg.norm(x))
        ny = float(np.linalg.norm(y))
        if nx <= 0 or ny <= 0:
            return np.nan

        c = float(np.dot(x, y) / (nx * ny))
        return float(np.clip(c, -1.0, 1.0))

    weighted_r_by_s2_col = {}
    if show_weighted_map_r:
        min_valid_bins_w = int(globals().get('MIN_VALID_BINS_FOR_CORR_WEIGHTED', 20))
        min_eff_bins_w = int(globals().get('MIN_EFFECTIVE_BINS_WEIGHTED', 20))
        try:
            min_occ_w = float(min_occupancy_per_bin_s)
        except (TypeError, ValueError):
            min_occ_w = float(globals().get('MIN_OCCUPANCY_WEIGHT', 0.5))
        if not np.isfinite(min_occ_w) or min_occ_w < 0:
            min_occ_w = 0.0
        min_valid_bins_u = int(globals().get('MIN_VALID_BINS_FOR_DISTANCE', min_valid_bins_w))

        metric_raw = str(overlay_similarity_metric).strip().lower()
        if metric_raw in ('weighted_pearson', 'weighted-r', 'weighted_r', 'weighted', 'wr', 'pearson', 'r'):
            metric_mode = 'pearson'
            metric_label = 'r'
        elif metric_raw in ('spearman', 'spearman_rho', 'rho'):
            metric_mode = 'spearman'
            metric_label = 'rho'
        elif metric_raw in ('cosine', 'cosine_similarity', 'cos'):
            metric_mode = 'cosine'
            metric_label = 'cos'
        else:
            metric_mode = 'pearson'
            metric_label = 'r'
            print(f"Unknown overlay_similarity_metric='{overlay_similarity_metric}', fallback to pearson")
        metric_label = _similarity_label_with_weight(metric_label, bool(weighted))

        for g_idx, g_cells in enumerate(cell_groups):
            group_cols = group_col_indices.get(g_idx, [])
            if len(g_cells) != len(group_cols):
                continue

            idx_s1 = None
            idx_s2 = None
            for idx_cond, c in enumerate(g_cells):
                cond_idx = _condition_index_from_label(c)
                if cond_idx == 1:
                    idx_s1 = idx_cond
                elif cond_idx == 2:
                    idx_s2 = idx_cond

            # Fallback ordering when labels are missing.
            if idx_s1 is None or idx_s2 is None:
                if len(g_cells) == 2:
                    idx_s1 = 0 if idx_s1 is None else idx_s1
                    idx_s2 = 1 if idx_s2 is None else idx_s2
                elif len(g_cells) >= 3:
                    idx_s1 = 1 if idx_s1 is None else idx_s1
                    idx_s2 = 2 if idx_s2 is None else idx_s2

            if idx_s1 is None or idx_s2 is None:
                continue

            c1 = g_cells[idx_s1]
            c2 = g_cells[idx_s2]
            if bool(c2.get('is_na_panel', False)):
                continue

            map_r = {}
            for map_key in ('rate_map', 'ss_norm_map', 'cs_norm_map', 'theta_map', 'slow_map'):
                sim_val = _compute_s1s2_map_similarity(
                    c1,
                    c2,
                    map_key,
                    metric_mode=metric_mode,
                    enforce_peak_filter=bool(globals().get('ENFORCE_S1S2_MIN_PEAK_RATE_FILTER', True)),
                    peak_threshold_hz=float(globals().get('S1S2_MIN_PEAK_RATE_HZ', 0.5)),
                    min_occupancy_per_bin_s=min_occ_w,
                    weighted=bool(weighted),
                    baseline_subtraction_cosine=bool(baseline_subtraction_cosine),
                )

                map_r[map_key] = f"{metric_label} = {sim_val:.2f}" if np.isfinite(sim_val) else f"{metric_label} = n/a"

            occ_sim_val, _occ_n = _spearman_nan_safe(
                c1.get('occupancy', None),
                c2.get('occupancy', None),
                min_valid_bins=int(min_valid_bins_u),
            )
            map_r['occupancy'] = (
                f"occupancy_spearman = {float(occ_sim_val):.2f}"
                if np.isfinite(occ_sim_val)
                else "occupancy_spearman = n/a"
            )

            weighted_r_by_s2_col[group_cols[idx_s2]] = {
                'map_r': map_r,
                's1_col': group_cols[idx_s1],
                's2_col': group_cols[idx_s2],
            }

    def _draw_between_s1s2_text(row_idx, s2_col, text_str, fontsize=5):
        if not text_str:
            return
        entry = weighted_r_by_s2_col.get(s2_col, {})
        s1_col = entry.get('s1_col', None)
        if (
            (s1_col is not None)
            and ((row_idx, s1_col) in axes_grid)
            and ((row_idx, s2_col) in axes_grid)
        ):
            bb1 = axes_grid[(row_idx, s1_col)].get_position()
            bb2 = axes_grid[(row_idx, s2_col)].get_position()
            x_mid = 0.5 * (bb1.x1 + bb2.x0)
            y_top = max(bb1.y1, bb2.y1) + 0.003
            fig.text(x_mid, y_top, text_str, ha='center', va='bottom', fontsize=fontsize, fontname='Arial')
        else:
            axes_grid[(row_idx, s2_col)].text(
                0.5, 1.02, text_str,
                transform=axes_grid[(row_idx, s2_col)].transAxes,
                ha='center', va='bottom', fontsize=fontsize, fontname='Arial'
            )

    def _condition_key_from_cell(cell: dict[str, Any]) -> str:
        lbl = str(cell.get('condition_label', '')).lower()
        if 'combined' in lbl:
            return 'combined'
        if 'session 1' in lbl:
            return 'session1'
        if 'session 2' in lbl:
            return 'session2'
        return 'combined'

    merged_cache: dict[tuple[str, str, str], dict[str, Any] | None] = {}
    warned_plateau_sessions: set[str] = set()

    def _resolve_condition_merged_data(cell: dict[str, Any]) -> dict[str, Any] | None:
        if bool(cell.get('is_na_panel', False)):
            return None
        animal_id = str(cell.get('animal_id', cell.get('session', ''))).strip()
        if len(animal_id) == 0:
            return None
        data_root = plateau_data_folder if plateau_data_folder is not None else cell.get('data_folder', None)
        if not isinstance(data_root, str) or len(data_root.strip()) == 0:
            if animal_id not in warned_plateau_sessions:
                print(f"Plateau map warning: missing data folder for session '{animal_id}'.")
                warned_plateau_sessions.add(animal_id)
            return None

        cond = _condition_key_from_cell(cell)
        key = (os.path.abspath(data_root), animal_id, cond)
        if key in merged_cache:
            return merged_cache[key]

        try:
            full = _load_merged_data(Path(data_root) / animal_id, behavior_cleaning_config)
            if behavior_cleaning_config is not None:
                full = _clean_behavior_speed_outliers_for_cache(
                    full,
                    behavior_cleaning_config,
                    animal_id=animal_id,
                )
        except Exception:
            if animal_id not in warned_plateau_sessions:
                print(f"Plateau map warning: failed loading merged data for session '{animal_id}'.")
                warned_plateau_sessions.add(animal_id)
            merged_cache[key] = None
            return None

        if cond == 'combined':
            merged_cache[key] = full
            return full

        ranges, _has_session2 = _compute_session_ranges(
            full,
            split_mode=two_session_split_mode,
            split_window_minutes=two_session_split_window_minutes,
        )
        frame_range = ranges.get(cond, None)
        if frame_range is None:
            merged_cache[key] = None
            return None
        s0, s1 = frame_range
        merged_cache[key] = _slice_merged_data(full, int(s0), int(s1))
        return merged_cache[key]

    plateau_maps_by_cell: dict[int, dict[str, np.ndarray]] = {}
    merged_data_by_cell: dict[int, dict[str, Any] | None] = {}
    if include_plateau:
        for flat_idx, cell in enumerate(all_cells):
            merged_data = _resolve_condition_merged_data(cell)
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
                        finite_seg = seg[np.isfinite(seg)]
                        if finite_seg.size:
                            y_min_ps = min(y_min_ps, float(np.nanmin(finite_seg)))
                            y_max_ps = max(y_max_ps, float(np.nanmax(finite_seg)))
            plateau_shape_traces_by_cell[flat_idx] = traces_for_cell

        plateau_shape_xlim = (0.0, 250.0)
        if np.isfinite(y_min_ps) and np.isfinite(y_max_ps):
            if y_max_ps == y_min_ps:
                eps = 1e-3
                plateau_shape_ylim = (y_min_ps - eps, y_max_ps + eps)
            else:
                plateau_shape_ylim = (y_min_ps, y_max_ps)
        else:
            plateau_shape_ylim = (-0.2, 1.2)

        if plateau_shape_row is not None:
            anchor_plateau_ax = axes_grid.get((plateau_shape_row, first_data_col))
            if anchor_plateau_ax is not None:
                if plateau_shape_xlim is not None:
                    anchor_plateau_ax.set_xlim(*plateau_shape_xlim)
                if plateau_shape_ylim is not None:
                    anchor_plateau_ax.set_ylim(*plateau_shape_ylim)

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

        # Optional contour source override: use combined-session PF masks across panels.
        combined_cell_for_pf = None
        if bool(plot_PF_combined):
            try:
                combined_cell_for_pf = _get_condition_cell(animal_id, int(cell['cell_idx']), "combined")
            except Exception:
                combined_cell_for_pf = None

        if isinstance(combined_cell_for_pf, dict):
            pf_mask_for_plot = combined_cell_for_pf.get('place_field_mask', cell.get('place_field_mask', None))
            pf_components_for_plot = _sanitize_pf_components(
                combined_cell_for_pf.get('place_field_components', []),
                np.asarray(pf_mask_for_plot, dtype=bool) if pf_mask_for_plot is not None else None,
                rate_map=np.asarray(combined_cell_for_pf.get('rate_map', cell.get('rate_map', None)), dtype=float)
                if combined_cell_for_pf.get('rate_map', cell.get('rate_map', None)) is not None else None,
            )
            # Under combined-PF mode, force SS/CS panels to use the same all-spike PF contour.
            ss_mask_for_plot = pf_mask_for_plot
            cs_mask_for_plot = pf_mask_for_plot
            ss_components_for_plot = pf_components_for_plot
            cs_components_for_plot = pf_components_for_plot
        else:
            pf_mask_for_plot = cell.get('place_field_mask', None)
            pf_components_for_plot = _sanitize_pf_components(
                cell.get('place_field_components', []),
                np.asarray(pf_mask_for_plot, dtype=bool) if pf_mask_for_plot is not None else None,
                rate_map=np.asarray(cell.get('rate_map', None), dtype=float) if cell.get('rate_map', None) is not None else None,
            )
            ss_mask_for_plot = cell.get('ss_place_field_mask', None)
            cs_mask_for_plot = cell.get('cs_place_field_mask', None)
            ss_components_for_plot = []
            cs_components_for_plot = []

        # When plotting combined PF contours, use one shared contour color across panels.
        ss_pf_contour_color = "magenta" if bool(plot_PF_combined) else ss_contour_color
        cs_pf_contour_color = "magenta" if bool(plot_PF_combined) else complex_spike_color

        if cell.get("is_na_panel", False):
            for row_idx in range(map_rows):
                ax_na = axes_grid[(row_idx, display_col)]
                _style_map_axis(ax_na)
                ax_na.text(0.5, 0.5, "N/A", transform=ax_na.transAxes,
                           ha="center", va="center", fontsize=6, color="gray")
            if plot_spike_shapes_any:
                for _pf_key, row_idx in shape_rows:
                    ax_na = axes_grid[(row_idx, display_col)]
                    ax_na.text(0.5, 0.5, "N/A", transform=ax_na.transAxes,
                               ha="center", va="center", fontsize=6, color="gray")
                    ax_na.set_xticks([])
                    ax_na.set_yticks([])
                    for spine in ax_na.spines.values():
                        spine.set_visible(False)
            if plot_plateau_shapes and plateau_shape_row is not None:
                ax_na = axes_grid[(plateau_shape_row, display_col)]
                ax_na.text(0.5, 0.5, "N/A", transform=ax_na.transAxes,
                           ha="center", va="center", fontsize=6, color="gray")
                ax_na.set_xticks([])
                ax_na.set_yticks([])
                for spine in ax_na.spines.values():
                    spine.set_visible(False)
            axes_grid[(0, display_col)].set_title(
                _cell_title(cell, animal_short, cell_num),
                fontsize=5,
                fontname="Arial",
                pad=2,
            )
            continue

        if show_occupancy_heatmap and occupancy_row is not None:
            ax_occ = axes_grid[(occupancy_row, display_col)]
            occ_map = cell.get('occupancy', None)
            im_occ = None
            occ_max = np.nan
            if isinstance(occ_map, np.ndarray) and occ_map.ndim == 2 and np.any(np.isfinite(occ_map)):
                occ_arr = np.asarray(occ_map, dtype=float)
                occ_masked = ma.masked_where(~np.isfinite(occ_arr), occ_arr)
                occ_max = float(np.nanmax(occ_arr))
                vmax = occ_max if np.isfinite(occ_max) and occ_max > 0 else None
                im_occ = ax_occ.imshow(
                    occ_masked.T,
                    origin="lower",
                    extent=extent,
                    cmap="Greys",
                    interpolation="nearest",
                    vmin=0,
                    vmax=vmax,
                )
            _style_map_axis(ax_occ)
            ax_occ.set_title(_cell_title(cell, animal_short, cell_num), fontsize=5, fontname="Arial", pad=2)
            occ_text = f"max {occ_max:.1f}s" if np.isfinite(occ_max) else "max N/A"
            ax_occ.text(1.0, -0.02, occ_text, transform=ax_occ.transAxes,
                        ha="right", va="top", fontsize=4, fontname="Arial")
            if display_col in weighted_r_by_s2_col:
                _draw_between_s1s2_text(
                    occupancy_row,
                    display_col,
                    weighted_r_by_s2_col[display_col].get('map_r', {}).get('occupancy', ''),
                    fontsize=4.5,
                )
            if is_last_column and im_occ is not None:
                _add_colorbar(ax_occ, im_occ, ticks=[0, im_occ.get_clim()[1]], ticklabels=["0", "max"])

        # Trajectory
        ax_traj = axes_grid[(trajectory_row, display_col)]
        x_traj, y_traj = cell['x_traj'], cell['y_traj']
        ax_traj.plot(x_traj, y_traj, color="gray", linewidth=0.3, alpha=0.5, rasterized=True)
        ss_x = cell.get('ss_spikes_x', cell['spikes_x'])
        ss_y = cell.get('ss_spikes_y', cell['spikes_y'])
        if len(ss_x) > 0:
            ax_traj.scatter(ss_x, ss_y, s=2, color=simple_spike_color, alpha=0.6, linewidths=0, zorder=2, rasterized=True)
        cs_x = cell.get('cs_spikes_x', np.array([]))
        cs_y = cell.get('cs_spikes_y', np.array([]))
        if len(cs_x) > 0:
            ax_traj.scatter(cs_x, cs_y, s=2, color=complex_spike_color, alpha=0.6, linewidths=0, zorder=3, rasterized=True)
        _style_map_axis(ax_traj)
        if not show_occupancy_heatmap:
            ax_traj.set_title(_cell_title(cell, animal_short, cell_num), fontsize=5, fontname="Arial", pad=2)
        
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
        
        # Rate map (all spikes)
        ax_rate = axes_grid[(rate_row, display_col)]
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
            # Plot PF contour (always solid when shown).
            pf_mask = pf_mask_for_plot
            if pf_mask is None or not np.any(pf_mask):
                print(f"{animal_short} Cell {cell_num}: All spikes PF mask is empty")
            elif is_place_cell:
                if bool(plot_PF_combined):
                    _plot_pf_components(ax_rate, pf_components_for_plot, linestyle='solid')
                else:
                    _plot_pf_contour(ax_rate, pf_mask, "magenta", linestyle='solid')
            elif (not pf_only_place_cells) and plot_putative_PF:
                if bool(plot_PF_combined):
                    _plot_pf_components(ax_rate, pf_components_for_plot, linestyle='solid')
                else:
                    _plot_pf_contour(ax_rate, pf_mask, "magenta", linestyle='solid')
        _style_map_axis(ax_rate)
        rate_str = f"{peak_rate:.1f}" if np.isfinite(peak_rate) else "N/A"
        display_sig_mark = sig_mark if show_significance_marker_effective else ""
        label_str = f"{display_sig_mark} {rate_str} Hz".strip()
        ax_rate.text(1.0, -0.02, label_str, transform=ax_rate.transAxes,
                     ha="right", va="top", fontsize=4, fontname="Arial")
        _draw_split_session_plc_marker(ax_rate, cell)
        # Add star marker for place cells (renders as vector path in Illustrator)
        if is_place_cell and show_place_cell_star_effective:
            ax_rate.plot(0.03, -0.06, marker='*', markersize=4, color='black',
                        transform=ax_rate.transAxes, clip_on=False)
        if display_col in weighted_r_by_s2_col:
            _draw_between_s1s2_text(
                rate_row,
                display_col,
                weighted_r_by_s2_col[display_col].get('map_r', {}).get('rate_map', ''),
                fontsize=5,
            )
        if is_last_column and im_rate is not None:
            _add_colorbar(ax_rate, im_rate, ticks=[0, im_rate.get_clim()[1]], ticklabels=["0", "max"])
        
        # SS normalized map
        ax_ss = axes_grid[(ss_row, display_col)]
        ss_norm_map = cell['ss_norm_map']
        ss_p_val = cell.get('ss_p_value', 1.0)
        ss_sig_mark = _get_sig_marker(ss_p_val)
        ss_mask = ss_mask_for_plot
        im_ss = None
        if ss_norm_map is not None:
            ss_masked = ma.masked_where(np.isnan(ss_norm_map), ss_norm_map)
            im_ss = ax_ss.imshow(ss_masked.T, origin="lower", extent=extent, cmap=cmap,
                        interpolation="nearest", vmin=0, vmax=1)
            # Plot SS contour (always solid when shown).
            if ss_mask is None or not np.any(ss_mask):
                print(f"{animal_short} Cell {cell_num}: SS PF mask is empty")
            elif pf_only_place_cells and not is_place_cell:
                pass  # Skip SS PF if cell is not an all-spike place cell
            elif is_place_cell_ss:
                if bool(plot_PF_combined):
                    _plot_pf_components(ax_ss, ss_components_for_plot, linestyle='solid')
                else:
                    _plot_pf_contour(ax_ss, ss_mask, ss_pf_contour_color, linestyle='solid')
            elif plot_putative_PF:
                if bool(plot_PF_combined):
                    _plot_pf_components(ax_ss, ss_components_for_plot, linestyle='solid')
                else:
                    _plot_pf_contour(ax_ss, ss_mask, ss_pf_contour_color, linestyle='solid')
        _style_map_axis(ax_ss)
        ss_peak = cell['ss_peak_rate']
        ss_str = f"{ss_peak:.1f}" if np.isfinite(ss_peak) else "N/A"
        # Add star and significance for SS (star only if is_place_cell_ss)
        ss_display_sig = ss_sig_mark if show_significance_marker_effective else ""
        ss_label_str = f"{ss_display_sig} {ss_str} Hz".strip()
        ax_ss.text(1.0, -0.02, ss_label_str, transform=ax_ss.transAxes,
                   ha="right", va="top", fontsize=4, fontname="Arial")
        # Add star marker for SS place cells
        if is_place_cell_ss and show_place_cell_star_effective:
            ax_ss.plot(0.03, -0.06, marker='*', markersize=4, color='black',
                      transform=ax_ss.transAxes, clip_on=False)
        if display_col in weighted_r_by_s2_col:
            _draw_between_s1s2_text(
                ss_row,
                display_col,
                weighted_r_by_s2_col[display_col].get('map_r', {}).get('ss_norm_map', ''),
                fontsize=5,
            )
        if is_last_column and im_ss is not None:
            _add_colorbar(ax_ss, im_ss)
        
        # CS normalized map
        ax_cs = axes_grid[(cs_row, display_col)]
        cs_norm_map = cell['cs_norm_map']
        cs_p_val = cell.get('cs_p_value', 1.0)
        cs_sig_mark = _get_sig_marker(cs_p_val)
        cs_mask = cs_mask_for_plot
        im_cs = None
        if cs_norm_map is not None:
            cs_masked = ma.masked_where(np.isnan(cs_norm_map), cs_norm_map)
            im_cs = ax_cs.imshow(cs_masked.T, origin="lower", extent=extent, cmap=cmap,
                        interpolation="nearest", vmin=0, vmax=1)
            # Plot CS contour (always solid when shown).
            if cs_mask is None or not np.any(cs_mask):
                print(f"{animal_short} Cell {cell_num}: CS PF mask is empty")
            elif pf_only_place_cells and not is_place_cell:
                pass  # Skip CS PF if cell is not an all-spike place cell
            elif is_place_cell_cs:
                if bool(plot_PF_combined):
                    _plot_pf_components(ax_cs, cs_components_for_plot, linestyle='solid')
                else:
                    _plot_pf_contour(ax_cs, cs_mask, cs_pf_contour_color, linestyle='solid')
            elif plot_putative_PF:
                if bool(plot_PF_combined):
                    _plot_pf_components(ax_cs, cs_components_for_plot, linestyle='solid')
                else:
                    _plot_pf_contour(ax_cs, cs_mask, cs_pf_contour_color, linestyle='solid')
        _style_map_axis(ax_cs)
        cs_peak = cell['cs_peak_rate']
        cs_str = f"{cs_peak:.1f}" if np.isfinite(cs_peak) else "N/A"
        # Add star and significance for CS (star only if is_place_cell_cs)
        cs_display_sig = cs_sig_mark if show_significance_marker_effective else ""
        cs_label_str = f"{cs_display_sig} {cs_str} Hz".strip()
        ax_cs.text(1.0, -0.02, cs_label_str, transform=ax_cs.transAxes,
                   ha="right", va="top", fontsize=4, fontname="Arial")
        # Add star marker for CS place cells
        if is_place_cell_cs and show_place_cell_star_effective:
            ax_cs.plot(0.03, -0.06, marker='*', markersize=4, color='black',
                      transform=ax_cs.transAxes, clip_on=False)
        if display_col in weighted_r_by_s2_col:
            _draw_between_s1s2_text(
                cs_row,
                display_col,
                weighted_r_by_s2_col[display_col].get('map_r', {}).get('cs_norm_map', ''),
                fontsize=5,
            )
        if is_last_column and im_cs is not None:
            _add_colorbar(ax_cs, im_cs)
        
        # Theta amplitude map
        ax_theta = axes_grid[(theta_row, display_col)]
        theta_map = cell.get('theta_map', None)
        im_theta = None
        if theta_map is not None and np.any(np.isfinite(theta_map)):
            theta_masked = ma.masked_where(np.isnan(theta_map), theta_map)
            theta_vmin, theta_vmax = theta_vlim if theta_vlim else (np.nanmin(theta_map), np.nanmax(theta_map))
            im_theta = ax_theta.imshow(theta_masked.T, origin="lower", extent=extent, cmap=cmap,
                           interpolation="nearest", vmin=theta_vmin, vmax=theta_vmax)
        _style_map_axis(ax_theta)
        if show_theta_slow_numbers:
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
        if display_col in weighted_r_by_s2_col:
            _draw_between_s1s2_text(
                theta_row,
                display_col,
                weighted_r_by_s2_col[display_col].get('map_r', {}).get('theta_map', ''),
                fontsize=5,
            )
        if is_last_column and im_theta is not None:
            _add_colorbar(ax_theta, im_theta)
        
        # Slow Vm map
        ax_slow = axes_grid[(slow_row, display_col)]
        slow_map = cell.get('slow_map', None)
        im_slow = None
        if slow_map is not None and np.any(np.isfinite(slow_map)):
            slow_masked = ma.masked_where(np.isnan(slow_map), slow_map)
            slow_vmin, slow_vmax = slow_vlim if slow_vlim else (-np.nanmax(np.abs(slow_map)), np.nanmax(np.abs(slow_map)))
            im_slow = ax_slow.imshow(slow_masked.T, origin="lower", extent=extent, cmap=slow_cmap,
                          interpolation="nearest", vmin=slow_vmin, vmax=slow_vmax)
        _style_map_axis(ax_slow)
        if show_theta_slow_numbers:
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
        if display_col in weighted_r_by_s2_col:
            _draw_between_s1s2_text(
                slow_row,
                display_col,
                weighted_r_by_s2_col[display_col].get('map_r', {}).get('slow_map', ''),
                fontsize=5,
            )
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
                if pf_mask_for_plot is not None and np.any(pf_mask_for_plot):
                    if is_place_cell:
                        if bool(plot_PF_combined):
                            _plot_pf_components(ax_plateau, pf_components_for_plot, linewidth=0.3, linestyle='solid', alpha=0.6)
                        else:
                            _plot_pf_contour(ax_plateau, pf_mask_for_plot, "magenta", linewidth=0.3, linestyle='solid', alpha=0.6)
                    elif (not pf_only_place_cells) and plot_putative_PF:
                        if bool(plot_PF_combined):
                            _plot_pf_components(ax_plateau, pf_components_for_plot, linewidth=0.3, linestyle='solid', alpha=0.6)
                        else:
                            _plot_pf_contour(ax_plateau, pf_mask_for_plot, "magenta", linewidth=0.3, linestyle='solid', alpha=0.6)
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
        
        # Row 6-7 (optional): Spike shapes (In-PF vs Out-PF)
        if plot_spike_shapes_any:
            spike_shapes = cell.get('spike_shapes')
            
            def _plot_shapes_axis(ax, pf_key, no_pf=False, spike_type_filter=None, no_background=False):
                full_span_mode = bool(plot_spike_shapes_overall) and (spike_type_filter in ('simple', 'complex'))
                if no_background:
                    ax.set_facecolor('white')
                else:
                    bg_color = "#FFE6F2" if pf_key == 'in' else ("#F0F0F0" if pf_key == 'out' else "#F7F7F7")
                    ax.set_facecolor(bg_color)
                if full_span_mode and spike_type_filter == 'simple' and (shape_xlim_ss_full is not None):
                    axis_xlim = shape_xlim_ss_full
                elif full_span_mode and spike_type_filter == 'complex' and (shape_xlim_cb_full is not None):
                    axis_xlim = shape_xlim_cb_full
                else:
                    axis_xlim = shape_xlim

                if full_span_mode and spike_type_filter == 'simple' and (shape_ylim_ss is not None):
                    axis_ylim = shape_ylim_ss
                elif full_span_mode and spike_type_filter == 'complex' and (shape_ylim_cb is not None):
                    axis_ylim = shape_ylim_cb
                else:
                    axis_ylim = shape_ylim

                ax.set_xlim(*axis_xlim)
                ax.set_ylim(*axis_ylim)
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                if no_pf and pf_key == 'in':
                    # Non-PLCs: no PF separation (leave In-PF row empty)
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
                    if pf_key.startswith('overall'):
                        keys = [f"{spike_shape_state}_in", f"{spike_shape_state}_out"]
                    elif no_pf and pf_key == 'out':
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
                    if full_span_mode and spike_type_filter == spike_type:
                        x = time_ms
                    else:
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
                        y_top = axis_ylim[1] - 0.08 * (axis_ylim[1] - axis_ylim[0])
                        ax.text(x_mid, y_top, f'n={len(norm_waves)}',
                                ha='center', va='top', fontsize=3.5, fontname='Arial',
                                color=color, alpha=0.8)

                if spike_type_filter == 'simple':
                    _plot_block('simple', simple_spike_color)
                elif spike_type_filter == 'complex':
                    _plot_block('complex', complex_spike_color)
                else:
                    _plot_block('simple', simple_spike_color)
                    _plot_block('complex', complex_spike_color)
                
                # Reference line at t=0 for plotted segment(s)
                y_line_low = max(axis_ylim[0], min(0.0, axis_ylim[1]))
                y_line_high = max(axis_ylim[0], min(1.0, axis_ylim[1]))
                y_line_low, y_line_high = sorted((y_line_low, y_line_high))
                if np.isfinite(y_line_low) and np.isfinite(y_line_high) and y_line_high > y_line_low:
                    if full_span_mode:
                        ax.plot([0.0, 0.0], [y_line_low, y_line_high], color='black', linestyle='--', linewidth=0.5, alpha=0.35)
                    else:
                        if spike_type_filter in (None, 'simple'):
                            x0_ss = (0.0 - shape_time_xlim_ss[0]) * shape_ss_x_scale + shape_ss_x_start
                            ax.plot([x0_ss, x0_ss], [y_line_low, y_line_high], color='black', linestyle='--', linewidth=0.5, alpha=0.35)
                        if spike_type_filter in (None, 'complex'):
                            x0_cb = (0.0 - shape_time_xlim_cb[0]) * shape_cb_x_scale + shape_cb_x_start
                            ax.plot([x0_cb, x0_cb], [y_line_low, y_line_high], color='black', linestyle='--', linewidth=0.5, alpha=0.35)
            
            pf_mask_all = cell.get('place_field_mask', None)
            # Lenient In-PF gating: use PF mask availability (solid or dashed contours),
            # not strict place-cell status.
            try:
                has_pf_mask = (pf_mask_all is not None) and bool(np.any(np.asarray(pf_mask_all)))
            except Exception:
                has_pf_mask = False
            no_pf = not has_pf_mask
            ax_overall_ss_shape = None
            ax_overall_cb_shape = None
            ax_in_shape = None
            ax_out_shape = None
            if plot_spike_shapes_overall:
                if shape_row_overall_ss is not None:
                    ax_overall_ss_shape = axes_grid[(shape_row_overall_ss, display_col)]
                    _plot_shapes_axis(ax_overall_ss_shape, 'overall_ss', no_pf=False, spike_type_filter='simple', no_background=True)
                if shape_row_overall_cb is not None:
                    ax_overall_cb_shape = axes_grid[(shape_row_overall_cb, display_col)]
                    _plot_shapes_axis(ax_overall_cb_shape, 'overall_cb', no_pf=False, spike_type_filter='complex', no_background=True)
            else:
                if plot_spike_shapes_in_field and (shape_row_in is not None):
                    ax_in_shape = axes_grid[(shape_row_in, display_col)]
                    _plot_shapes_axis(ax_in_shape, 'in', no_pf=no_pf)
                if plot_spike_shapes_out_field and (shape_row_out is not None):
                    ax_out_shape = axes_grid[(shape_row_out, display_col)]
                    _plot_shapes_axis(ax_out_shape, 'out', no_pf=no_pf)
            
            # Legend (first available spike-shape row, rightmost column)
            if is_last_column:
                target_ax = None
                if ax_overall_ss_shape is not None:
                    target_ax = ax_overall_ss_shape
                elif ax_overall_cb_shape is not None:
                    target_ax = ax_overall_cb_shape
                elif no_pf and (ax_out_shape is not None):
                    target_ax = ax_out_shape
                elif ax_in_shape is not None:
                    target_ax = ax_in_shape
                else:
                    target_ax = ax_out_shape
                if target_ax is not None:
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
            
            # Horizontal scale bars on first column: SS row gets 10 ms, CB row gets 50 ms.
            if is_first_column:
                def _place_bar(target_ax, seg_x0, seg_x1, bar_len_x, label):
                    if target_ax is None:
                        return
                    ylims = target_ax.get_ylim()
                    y_span = float(ylims[1] - ylims[0]) if ylims[1] != ylims[0] else 1.0
                    y0 = float(ylims[0] + 0.08 * y_span)
                    seg_x0 = float(seg_x0)
                    seg_x1 = float(seg_x1)
                    seg_span = float(seg_x1 - seg_x0) if seg_x1 != seg_x0 else 1.0
                    if (not np.isfinite(seg_span)) or seg_span <= 0:
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

                if plot_spike_shapes_overall:
                    _place_bar(ax_overall_ss_shape, shape_xlim_ss_full[0], shape_xlim_ss_full[1], 10.0, '10 ms')
                    _place_bar(ax_overall_cb_shape, shape_xlim_cb_full[0], shape_xlim_cb_full[1], 50.0, '50 ms')
                else:
                    target_scale_ax = ax_out_shape if ax_out_shape is not None else ax_in_shape
                    _place_bar(target_scale_ax, shape_ss_x_start, shape_ss_x_end, 10.0 * shape_ss_x_scale, '10 ms')
                    _place_bar(target_scale_ax, shape_cb_x_start, shape_xlim[1], 50.0 * shape_cb_x_scale, '50 ms')

        if plot_plateau_shapes and plateau_shape_row is not None:
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
                if x_ms.size == 0 or seg.size == 0:
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
                x_span = float(xlims[1] - xlims[0]) if xlims[1] != xlims[0] else 1.0
                y_span = float(ylims[1] - ylims[0]) if ylims[1] != ylims[0] else 1.0
                x_bar0 = float(xlims[0] + 0.08 * x_span)
                x_bar1 = min(float(x_bar0 + 100.0), float(xlims[1] - 0.08 * x_span))
                if x_bar1 > x_bar0:
                    y_bar = float(ylims[0] + 0.08 * y_span)
                    ax_plateau_shape.plot([x_bar0, x_bar1], [y_bar, y_bar], color='black', linewidth=0.8, solid_capstyle='butt')
                    ax_plateau_shape.text((x_bar0 + x_bar1) / 2, y_bar - 0.06 * y_span, '100 ms',
                                          ha='center', va='top', fontsize=4, fontname='Arial')
    
    # Add row labels on first column
    row_labels = []
    if show_occupancy_heatmap:
        row_labels.append('Occupancy')
    row_labels.extend(['Trajectory', 'All spikes', 'SS', 'CS', 'Theta', 'Slow Vm'])
    for _, label in plateau_row_modes:
        row_labels.append(label)
    if plot_spike_shapes_overall:
        row_labels.extend(['SS shapes', 'CB shapes'])
    else:
        if plot_spike_shapes_in_field:
            row_labels.append('In-PF shapes')
        if plot_spike_shapes_out_field:
            row_labels.append('Out-PF shapes')
    if plot_plateau_shapes:
        row_labels.append('All plateaus')
    for row_idx, label in enumerate(row_labels):
        axes_grid[(row_idx, first_data_col)].text(-0.15, 0.5, label, 
            transform=axes_grid[(row_idx, first_data_col)].transAxes,
            ha="right", va="center", fontsize=5, fontname="Arial", rotation=90)
    _draw_occupancy_spearman_labels()
    
    # Save figure
    if save_path:
        plt.rcParams["svg.fonttype"] = "none"
        fig.savefig(save_path, dpi=300)
        print(f"Saved: {save_path}")
    
    plt.show()
    return fig

print("Defined plot_selected_cells_figure() function for session comparison")
