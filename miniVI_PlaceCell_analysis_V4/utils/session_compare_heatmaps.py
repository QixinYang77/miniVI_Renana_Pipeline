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
    _get_spike_positions_on_traj,
    _load_merged_data,
    _prepare_native_analysis_context,
    _run_place_cell_analysis_native,
)
from utils.spatial_heatmaps import is_csplus_place_cell


@dataclass
class SessionCompareParams:
    panel_mode: str = "s1_s2"  # "s1_s2" or "combined_s1_s2"
    cache_version: str = "v2"
    rebuild_cache: bool = False
    max_cells_per_figure: int = 10
    missing_s2_policy: str = "na_panel"
    occupancy_spearman_threshold: float = -0.5
    apply_occupancy_dataset_filter: bool = True
    enforce_s1s2_min_peak_rate_filter: bool = True
    s1s2_min_peak_rate_hz: float = 0.5
    clean_heatmap: bool = True
    heatmap_similarity_metric: str = "weighted_pearson"
    plot_spike_shapes: bool = True
    plot_spike_shapes_overall: bool = True
    plot_spike_shapes_in_field: bool = True
    plot_spike_shapes_out_field: bool = True
    plot_PF_combined: bool = True
    selected_theta_vlim: tuple[float, float] | None = None
    selected_slow_vlim: tuple[float, float] | None = None


# Globals consumed by the ported renderer function.
ENFORCE_S1S2_MIN_PEAK_RATE_FILTER = True
S1S2_MIN_PEAK_RATE_HZ = 0.5
MIN_VALID_BINS_FOR_CORR_WEIGHTED = 20
MIN_EFFECTIVE_BINS_WEIGHTED = 20
MIN_OCCUPANCY_WEIGHT = 0.0
MIN_VALID_BINS_FOR_DISTANCE = 20

_SESSION_PAYLOAD_BY_DATASET: dict[str, dict[str, Any]] = {}
SESSION_COMPARE_CACHE_SCHEMA = "session_compare_heatmaps_v2"


def _cache_path(dataset_dir: Path, cache_version: str) -> Path:
    return dataset_dir / f"spatial_session_cache_{cache_version}.pkl"


def _valid_cache_obj(cache_obj: Any, dataset_id: str, cache_version: str) -> bool:
    if not isinstance(cache_obj, dict):
        return False
    if cache_obj.get("cache_schema") != SESSION_COMPARE_CACHE_SCHEMA:
        return False
    if cache_obj.get("cache_version") != cache_version:
        return False
    if cache_obj.get("dataset_id") != dataset_id:
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


def _compute_session_ranges(merged_data: dict[str, Any]) -> tuple[dict[str, tuple[int, int] | None], bool]:
    n_frames = int(len(merged_data.get("x_neural", [])))
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

    all_spikes = _clip_spike_list(loaded.get("all_spikes", loaded.get("spikes", [])), s0, s1)
    sliced["spikes"] = all_spikes
    sliced["all_spikes"] = all_spikes
    sliced["refined_SS"] = _clip_spike_list(loaded.get("refined_SS", [np.array([], dtype=int) for _ in range(n_cells)]), s0, s1)
    sliced["all_CS_spikes"] = _clip_spike_list(loaded.get("all_CS_spikes", [np.array([], dtype=int) for _ in range(n_cells)]), s0, s1)

    for k in ["spike_heights_interpolated", "SNR_interpolated", "traces_SNR_interpolated", "Vm_SNR_interpolated"]:
        if k in loaded:
            sliced[k] = _clip_vec_list(loaded[k], s0, s1)

    sliced["complex_bursts_dicts"] = _clip_complex_bursts_dicts(loaded.get("complex_bursts_dicts", []), s0, s1)
    sliced["plateaus_dicts"] = None
    sliced["burst_metrics"] = [None for _ in range(n_cells)]
    sliced["session_start_frames"] = [0]

    return sliced


def _analysis_to_spatial_entry_from_ctx(
    analysis: dict[str, Any],
    dataset_id: str,
    cell_idx: int,
    ctx: dict[str, Any],
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
        "n_place_fields": analysis.get("n_place_fields", 0),
        "n_ss_place_fields": analysis.get("n_ss_place_fields", 0),
        "n_cs_place_fields": analysis.get("n_cs_place_fields", 0),
        "field_area": analysis.get("field_area", np.nan),
        "ss_field_area": analysis.get("ss_field_area", np.nan),
        "cs_field_area": analysis.get("cs_field_area", np.nan),
        "si_bits_per_spike": analysis.get("si_bits_per_spike", np.nan),
        "si_bits_per_spike_ss": analysis.get("si_bits_per_spike_ss", np.nan),
        "si_bits_per_spike_cs": analysis.get("si_bits_per_spike_cs", np.nan),
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
        "n_frames_total": n_frames_total,
        "n_frames_kept_total": kept_total,
        "n_removed_frames_total": removed_total,
        "pct_removed_frames_total": pct_removed_total,
    }
    return out


def _run_spatial_for_slice(
    sliced: dict[str, Any],
    dataset_id: str,
    config: PipelineConfig,
) -> list[dict[str, Any]]:
    ctx = _prepare_native_analysis_context(sliced, config)
    outputs = _run_place_cell_analysis_native(dataset_id, config, ctx)

    out_cells: list[dict[str, Any]] = []
    for cell_idx, analysis in enumerate(outputs):
        if not bool(ctx["eligible_cells"][cell_idx]):
            continue
        if not isinstance(analysis, dict):
            continue
        out_cells.append(_analysis_to_spatial_entry_from_ctx(analysis, dataset_id, cell_idx, ctx))
    return out_cells


def _count_cb_run_in(cell: dict[str, Any]) -> int:
    spike_shapes = cell.get("spike_shapes")
    if isinstance(spike_shapes, dict) and "complex" in spike_shapes:
        shapes = spike_shapes.get("complex", {}).get("shapes", {})
        return int(len(shapes.get("run_in", [])))
    return 0


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
    return bool(np.isfinite(p1) and np.isfinite(p2) and (p1 > float(threshold)) and (p2 > float(threshold)))


def build_session_compare_payloads(config: PipelineConfig, params: SessionCompareParams) -> dict[str, Any]:
    dataset_registry: list[dict[str, Any]] = []
    for dataset_id in config.animals:
        dataset_dir = config.data_root / dataset_id
        spatial_path = dataset_dir / "spatial_analysis_full.pkl"
        merged_primary = dataset_dir / "merged_aligned_data.pkl"
        merged_fallback = dataset_dir / "merged_aligned_data_CS.pkl"
        merged_path = merged_primary if merged_primary.exists() else merged_fallback

        has_required = spatial_path.exists() and merged_path.exists()
        if not has_required:
            continue

        merged_tmp = _load_merged_data(dataset_dir)
        frame_ranges, has_session2 = _compute_session_ranges(merged_tmp)

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
                if _valid_cache_obj(loaded_cache, dataset_id, params.cache_version):
                    cache_obj = loaded_cache
                    print(f"Loaded session-compare cache: {cache_path}")
            except Exception as exc:
                print(f"Cache load failed for {dataset_id}: {exc}")

        if cache_obj is None:
            merged_data = _load_merged_data(dataset_dir)
            ranges, has_session2 = _compute_session_ranges(merged_data)

            cache_obj = {
                "cache_schema": SESSION_COMPARE_CACHE_SCHEMA,
                "cache_version": params.cache_version,
                "dataset_id": dataset_id,
                "has_session2": bool(has_session2),
                "frame_ranges": {
                    "session1": list(ranges["session1"]) if ranges["session1"] is not None else None,
                    "session2": list(ranges["session2"]) if ranges["session2"] is not None else None,
                },
                "cells": {},
                "generated_from": {"merged_path": str(ds["merged_path"])},
            }

            with ds["spatial_path"].open("rb") as f:
                combined_cells = pickle.load(f)
            for c in combined_cells:
                if not isinstance(c, dict) or "cell_idx" not in c:
                    continue
                c2 = copy.deepcopy(c)
                c2["session"] = dataset_id
                c2["animal_id"] = dataset_id
                idx = int(c2["cell_idx"])
                cache_obj["cells"].setdefault(idx, {})["combined"] = c2

            s1_0, s1_1 = ranges["session1"]
            sliced_s1 = _slice_merged_data(merged_data, s1_0, s1_1)
            session1_cells = _run_spatial_for_slice(sliced_s1, dataset_id, config)
            for c in session1_cells:
                idx = int(c["cell_idx"])
                cache_obj["cells"].setdefault(idx, {})["session1"] = c

            if has_session2 and ranges["session2"] is not None:
                s2_0, s2_1 = ranges["session2"]
                sliced_s2 = _slice_merged_data(merged_data, s2_0, s2_1)
                session2_cells = _run_spatial_for_slice(sliced_s2, dataset_id, config)
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
            )
            cell_labels[key] = "csplus" if is_csplus else "csminus"

    csplus_keys = sorted([k for k, v in cell_labels.items() if v == "csplus"])
    csminus_keys = sorted([k for k, v in cell_labels.items() if v == "csminus"])
    nonpc_keys = sorted([
        (str(c.get("session", "")), int(c.get("cell_idx", -1)))
        for c in combined_cells
        if not bool(c.get("is_place_cell", False)) and int(c.get("cell_idx", -1)) >= 0
    ])

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


def render_session_compare_heatmaps(
    config: PipelineConfig,
    params: SessionCompareParams,
    groups_payload: dict[str, Any],
) -> dict[str, Any]:
    figure_save_folder = config.figures_root / "CKII_pooled"
    figure_save_folder.mkdir(parents=True, exist_ok=True)

    panel_suffix = "combined_s1_s2" if params.panel_mode == "combined_s1_s2" else "s1_s2"

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
            print(f"  Rendering {label} part {part_idx}/{n_parts}: n={len(chunk)} -> {out_name}")
            _ = plot_session_compare_selected_cells_figure(
                cell_groups=chunk,
                theta_vlim=groups_payload.get("selected_theta_vlim", None),
                slow_vlim=groups_payload.get("selected_slow_vlim", None),
                save_path=str(out_path),
                plot_putative_PF=True,
                pf_only_place_cells=False,
                show_place_cell_star=True,
                show_significance_marker=True,
                overlay_similarity_metric=str(params.heatmap_similarity_metric),
                clean_heatmap=bool(params.clean_heatmap),
                plot_spike_shapes=bool(params.plot_spike_shapes),
                plot_spike_shapes_overall=bool(params.plot_spike_shapes_overall),
                plot_spike_shapes_in_field=bool(params.plot_spike_shapes_in_field),
                plot_spike_shapes_out_field=bool(params.plot_spike_shapes_out_field),
                plot_PF_combined=bool(params.plot_PF_combined),
                show_shape_counts=False,
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
    }
    return summary


def run_session_compare_heatmaps(
    config: PipelineConfig,
    params: SessionCompareParams,
) -> dict[str, Any]:
    payloads = build_session_compare_payloads(config, params)
    groups = assemble_session_compare_groups(config, params, payloads)
    rendered = render_session_compare_heatmaps(config, params, groups)

    out = {
        "payloads": payloads,
        "groups": groups,
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
    overlay_similarity_metric='weighted_pearson',
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
        Options: 'weighted_pearson' (default), 'spearman', 'cosine'.
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
    if plot_spike_shapes_overall:
        n_shape_rows = 2  # separate rows: overall SS and overall CB
    else:
        n_shape_rows = int(plot_spike_shapes_in_field) + int(plot_spike_shapes_out_field)

    base_rows = 6  # trajectory, rate map, SS, CS, theta, slow
    n_rows = base_rows + n_shape_rows
    
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
        shape_rows.append(('overall_ss', base_rows + len(shape_rows)))
        shape_rows.append(('overall_cb', base_rows + len(shape_rows)))
    else:
        if plot_spike_shapes_in_field:
            shape_rows.append(('in', base_rows + len(shape_rows)))
        if plot_spike_shapes_out_field:
            shape_rows.append(('out', base_rows + len(shape_rows)))
    shape_row_lookup = {k: r for k, r in shape_rows}
    shape_row_overall_ss = shape_row_lookup.get('overall_ss', None)
    shape_row_overall_cb = shape_row_lookup.get('overall_cb', None)
    shape_row_in = shape_row_lookup.get('in', None)
    shape_row_out = shape_row_lookup.get('out', None)
    shape_anchor_row = shape_rows[0][1] if len(shape_rows) > 0 else None
    
    # Get arena dimensions from first cell
    params = all_cells[0]['params']
    width_real = params['width_real']
    height_real = params['height_real']
    arena_aspect = height_real / width_real
    
    # Calculate figure dimensions from fixed per-subplot width
    left_margin = 0.4
    right_margin = 0.3
    top_margin = 0.25
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
    
    # Base rows (maps)
    for row in range(base_rows):
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
    
    def _plot_pf_contour(ax, pf_mask, color, linewidth=0.6, linestyle='solid'):
        if pf_mask is None or not np.any(pf_mask):
            return
        padded_mask = np.pad(pf_mask.astype(float), pad_width=1, mode='constant', constant_values=0)
        bin_size = width_real / pf_mask.shape[0]
        padded_extent = (-bin_size, width_real + bin_size, -bin_size, height_real + bin_size)
        ax.contour(padded_mask.T, levels=[0.5], colors=color, linewidths=linewidth,
                   linestyles=linestyle, extent=padded_extent, origin="lower")

    def _sanitize_pf_components(components, fallback_mask, rate_map=None):
        clean = []
        if isinstance(components, (list, tuple)):
            for comp in components:
                arr = np.asarray(comp, dtype=bool)
                if arr.ndim != 2:
                    continue
                if fallback_mask is not None and isinstance(fallback_mask, np.ndarray) and fallback_mask.ndim == 2:
                    if arr.shape != fallback_mask.shape:
                        continue
                if np.any(arr):
                    clean.append(arr)
        if len(clean) == 0 and isinstance(fallback_mask, np.ndarray) and fallback_mask.ndim == 2 and np.any(fallback_mask):
            mask = np.asarray(fallback_mask, dtype=bool)
            # Fallback for old caches: split connected PF components from merged mask.
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
                    clean = [np.asarray(c, dtype=bool) for c in comps if np.any(c)]
            if len(clean) == 0:
                clean = [mask]
        return clean

    def _plot_pf_components(ax, components, linewidth=0.6, linestyle='solid'):
        if not isinstance(components, (list, tuple)) or len(components) == 0:
            return
        for i, comp in enumerate(components):
            color = "magenta" if i == 0 else "cyan"
            _plot_pf_contour(ax, comp, color, linewidth=linewidth, linestyle=linestyle)
    
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
        min_occ_w = float(globals().get('MIN_OCCUPANCY_WEIGHT', 0.0))
        min_valid_bins_u = int(globals().get('MIN_VALID_BINS_FOR_DISTANCE', min_valid_bins_w))

        metric_raw = str(overlay_similarity_metric).strip().lower()
        if metric_raw in ('weighted_pearson', 'weighted-r', 'weighted_r', 'weighted', 'wr', 'pearson', 'r'):
            metric_mode = 'weighted_pearson'
            metric_label = 'r'
        elif metric_raw in ('spearman', 'spearman_rho', 'rho'):
            metric_mode = 'spearman'
            metric_label = 'rho'
        elif metric_raw in ('cosine', 'cosine_similarity', 'cos'):
            metric_mode = 'cosine'
            metric_label = 'cos'
        else:
            metric_mode = 'weighted_pearson'
            metric_label = 'r'
            print(f"Unknown overlay_similarity_metric='{overlay_similarity_metric}', fallback to weighted_pearson")

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
                if bool(globals().get('ENFORCE_S1S2_MIN_PEAK_RATE_FILTER', True)):
                    thr = float(globals().get('S1S2_MIN_PEAK_RATE_HZ', 0.5))
                    if not _passes_s1s2_peak_threshold(c1.get('animal_id', ''), int(c1.get('cell_idx', -1)), map_key=map_key, threshold=thr):
                        map_r[map_key] = f"{metric_label} = n/a"
                        continue

                if metric_mode == 'weighted_pearson':
                    sim_val = _weighted_map_corr(
                        c1.get(map_key, None),
                        c2.get(map_key, None),
                        c1.get('occupancy', None),
                        c2.get('occupancy', None),
                        min_valid_bins=min_valid_bins_w,
                        min_eff_bins=min_eff_bins_w,
                        min_occ=min_occ_w,
                    )
                elif metric_mode == 'spearman':
                    sim_val = _spearman_map_corr(
                        c1.get(map_key, None),
                        c2.get(map_key, None),
                        min_valid_bins=min_valid_bins_u,
                    )
                else:
                    sim_val = _cosine_map_corr(
                        c1.get(map_key, None),
                        c2.get(map_key, None),
                        min_valid_bins=min_valid_bins_u,
                    )

                map_r[map_key] = f"{metric_label} = {sim_val:.2f}" if np.isfinite(sim_val) else f"{metric_label} = n/a"

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
            for row_idx in range(base_rows):
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
            condition_label = cell.get("condition_label", "")
            title_str = f"{animal_short}\nCell {cell_num}"
            if condition_label:
                title_str += f"\n{condition_label}"
            axes_grid[(0, display_col)].set_title(title_str, fontsize=5, fontname="Arial", pad=2)
            continue
        
        # Row 0: Trajectory
        ax_traj = axes_grid[(0, display_col)]
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
        condition_label = cell.get("condition_label", "")
        title_str = f"{animal_short}\nCell {cell_num}"
        if condition_label:
            title_str += f"\n{condition_label}"
        ax_traj.set_title(title_str, fontsize=5, fontname="Arial", pad=2)
        
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
        # Add star marker for place cells (renders as vector path in Illustrator)
        if is_place_cell and show_place_cell_star_effective:
            ax_rate.plot(0.03, -0.06, marker='*', markersize=4, color='black',
                        transform=ax_rate.transAxes, clip_on=False)
        if display_col in weighted_r_by_s2_col:
            _draw_between_s1s2_text(
                1,
                display_col,
                weighted_r_by_s2_col[display_col].get('map_r', {}).get('rate_map', ''),
                fontsize=5,
            )
        if is_last_column and im_rate is not None:
            _add_colorbar(ax_rate, im_rate, ticks=[0, im_rate.get_clim()[1]], ticklabels=["0", "max"])
        
        # Row 2: SS normalized map
        ax_ss = axes_grid[(2, display_col)]
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
                2,
                display_col,
                weighted_r_by_s2_col[display_col].get('map_r', {}).get('ss_norm_map', ''),
                fontsize=5,
            )
        if is_last_column and im_ss is not None:
            _add_colorbar(ax_ss, im_ss)
        
        # Row 3: CS normalized map
        ax_cs = axes_grid[(3, display_col)]
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
                3,
                display_col,
                weighted_r_by_s2_col[display_col].get('map_r', {}).get('cs_norm_map', ''),
                fontsize=5,
            )
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
                4,
                display_col,
                weighted_r_by_s2_col[display_col].get('map_r', {}).get('theta_map', ''),
                fontsize=5,
            )
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
                5,
                display_col,
                weighted_r_by_s2_col[display_col].get('map_r', {}).get('slow_map', ''),
                fontsize=5,
            )
        if is_last_column and im_slow is not None:
            _add_colorbar(ax_slow, im_slow)
        
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
    
    # Add row labels on first column
    row_labels = ['Trajectory', 'All spikes', 'SS', 'CS', 'Theta', 'Slow Vm']
    if plot_spike_shapes_overall:
        row_labels.extend(['SS shapes', 'CB shapes'])
    else:
        if plot_spike_shapes_in_field:
            row_labels.append('In-PF shapes')
        if plot_spike_shapes_out_field:
            row_labels.append('Out-PF shapes')
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

print("Defined plot_selected_cells_figure() function for session comparison")
