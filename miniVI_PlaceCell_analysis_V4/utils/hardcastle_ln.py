"""Hardcastle-style LN model utilities adapted for CKII data.

This module ports the core ideas from the MATLAB implementation in
GiocomoLab/ln-model-of-mec-neurons:
- one-hot position / direction / speed predictors
- penalized Poisson model with exponential nonlinearity
- k-fold cross-validation
- forward model selection based on held-out LLH improvement

Adaptations for this project:
- theta phase is omitted
- the arena can be rectangular
- NaN / bad-frame handling and moving-epoch restriction follow the local pipeline
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import optimize, signal, sparse, stats
from scipy.ndimage import gaussian_filter

from utils.placecell_core import _compute_moving_epochs


LN_MODEL_SPECS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("PHS", ("P", "H", "S")),
    ("PH", ("P", "H")),
    ("PS", ("P", "S")),
    ("HS", ("H", "S")),
    ("P", ("P",)),
    ("H", ("H",)),
    ("S", ("S",)),
)


@dataclass(frozen=True)
class LNGroupData:
    name: str
    X: sparse.csr_matrix
    bin_index: np.ndarray
    n_bins: int
    penalty: sparse.csr_matrix
    centers: np.ndarray
    shape: tuple[int, ...]


def _gaussian_kernel(radius: int = 4, sigma: float = 2.0) -> np.ndarray:
    x = np.arange(-int(radius), int(radius) + 1, dtype=float)
    kernel = np.exp(-(x**2) / (2.0 * float(sigma) ** 2))
    kernel_sum = float(np.sum(kernel))
    if kernel_sum <= 0:
        return np.array([1.0], dtype=float)
    return kernel / kernel_sum


def _smooth_rate(fr_hz: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    fr_hz = np.asarray(fr_hz, dtype=float)
    if fr_hz.size == 0 or kernel.size <= 1:
        return fr_hz.copy()
    return np.convolve(fr_hz, kernel, mode="same")


def _pearson_corr_safe(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    if np.sum(finite) < 2:
        return np.nan
    x_use = x[finite]
    y_use = y[finite]
    if np.allclose(np.nanstd(x_use), 0.0) or np.allclose(np.nanstd(y_use), 0.0):
        return np.nan
    return float(np.corrcoef(x_use, y_use)[0, 1])


def _llh_increase_bits_per_spike(counts: np.ndarray, pred_counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=float)
    pred_counts = np.asarray(pred_counts, dtype=float)
    total_spikes = float(np.sum(counts))
    if total_spikes <= 0:
        return np.nan
    mean_count = float(np.mean(counts))
    if not np.isfinite(mean_count) or mean_count <= 0:
        return np.nan
    pred = np.clip(pred_counts, 1e-12, None)
    baseline = np.full_like(pred, mean_count, dtype=float)
    nll_model = float(np.sum(pred - counts * np.log(pred)))
    nll_mean = float(np.sum(baseline - counts * np.log(baseline)))
    return float((nll_mean - nll_model) / total_spikes / np.log(2.0))


def _wilcoxon_right(x: np.ndarray, y: np.ndarray | None = None) -> float:
    x = np.asarray(x, dtype=float)
    if y is None:
        finite = np.isfinite(x)
        x_use = x[finite]
        if x_use.size == 0:
            return np.nan
        if np.allclose(x_use, 0.0):
            return 1.0
        try:
            return float(
                stats.wilcoxon(x_use, alternative="greater", zero_method="wilcox").pvalue
            )
        except Exception:
            return np.nan

    y = np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x_use = x[finite]
    y_use = y[finite]
    if x_use.size == 0:
        return np.nan
    diff = x_use - y_use
    if np.allclose(diff, 0.0):
        return 1.0
    try:
        return float(
            stats.wilcoxon(x_use, y_use, alternative="greater", zero_method="wilcox").pvalue
        )
    except Exception:
        return np.nan


def _session_bounds_from_starts(
    session_start_frames: list[int] | np.ndarray | None,
    n_frames: int,
) -> list[tuple[int, int]]:
    if session_start_frames is None:
        starts = [0]
    else:
        starts = sorted({int(v) for v in np.asarray(session_start_frames).ravel() if int(v) >= 0})
        if not starts:
            starts = [0]
        if starts[0] != 0:
            starts = [0] + starts
    starts = [s for s in starts if s < int(n_frames)]
    if not starts:
        starts = [0]
    ends = starts[1:] + [int(n_frames)]
    return [(int(s), int(e)) for s, e in zip(starts, ends) if int(e) > int(s)]


def _compute_travel_direction_deg(
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    smooth_window: int = 5,
    min_step: float = 0.0,
) -> np.ndarray:
    x_vals = np.asarray(x_vals, dtype=float)
    y_vals = np.asarray(y_vals, dtype=float)
    out = np.full(x_vals.shape, np.nan, dtype=float)
    finite = np.isfinite(x_vals) & np.isfinite(y_vals)
    if np.sum(finite) < 2:
        return out
    idx = np.arange(x_vals.size)
    valid_idx = idx[finite]
    x_interp = np.interp(idx, valid_idx, x_vals[finite])
    y_interp = np.interp(idx, valid_idx, y_vals[finite])
    dx = np.gradient(x_interp)
    dy = np.gradient(y_interp)
    sw = int(max(1, smooth_window))
    if sw > 1:
        kernel = np.ones(sw, dtype=float) / float(sw)
        dx = np.convolve(dx, kernel, mode="same")
        dy = np.convolve(dy, kernel, mode="same")
    step = np.hypot(dx, dy)
    ang = np.degrees(np.arctan2(dy, dx)) % 360.0
    ang[step <= float(min_step)] = np.nan
    ang[~finite] = np.nan
    out[:] = ang
    return out


def _prepare_binned_ln_data(
    *,
    spike_frames: np.ndarray,
    x_cm: np.ndarray,
    y_cm: np.ndarray,
    speed: np.ndarray,
    hd_deg: np.ndarray,
    frame_rate: float,
    session_start_frames: list[int] | np.ndarray | None,
    selected_session_indices: tuple[int, ...] | None,
    bin_size_s: float,
    moving_only: bool,
    moving_speed_threshold: float,
    moving_kernel_size: int,
    moving_filter_type: str,
    moving_min_duration_s: float,
    moving_merge_gap_s: float,
    bad_mask: np.ndarray | None = None,
    trace_for_nan: np.ndarray | None = None,
    direction_predictor: str = "head",
    travel_smooth_window: int = 5,
    travel_min_step: float = 0.0,
    require_full_valid_bin: bool = True,
) -> dict[str, Any]:
    x_cm = np.asarray(x_cm, dtype=float)
    y_cm = np.asarray(y_cm, dtype=float)
    speed = np.asarray(speed, dtype=float)
    hd_deg = np.asarray(hd_deg, dtype=float)
    n_frames = int(x_cm.size)
    if any(arr.size != n_frames for arr in (y_cm, speed, hd_deg)):
        raise ValueError("Behavior arrays must have matching lengths.")
    if frame_rate <= 0:
        raise ValueError("frame_rate must be > 0.")

    direction_norm = str(direction_predictor).strip().lower()
    if direction_norm == "travel":
        direction_deg = _compute_travel_direction_deg(
            x_cm,
            y_cm,
            smooth_window=int(travel_smooth_window),
            min_step=float(travel_min_step),
        )
    elif direction_norm == "head":
        direction_deg = np.mod(hd_deg, 360.0)
    else:
        raise ValueError("direction_predictor must be 'head' or 'travel'.")

    if trace_for_nan is not None:
        trace_valid = np.isfinite(np.asarray(trace_for_nan, dtype=float))
        if trace_valid.size != n_frames:
            raise ValueError("trace_for_nan must match frame count.")
    else:
        trace_valid = np.ones(n_frames, dtype=bool)

    bad_frame_mask = np.zeros(n_frames, dtype=bool)
    if bad_mask is not None:
        bad_arr = np.asarray(bad_mask)
        if bad_arr.dtype == bool and bad_arr.size == n_frames:
            bad_frame_mask = bad_arr.copy()
        else:
            bad_idx = np.asarray(bad_arr, dtype=int)
            bad_idx = bad_idx[(bad_idx >= 0) & (bad_idx < n_frames)]
            bad_frame_mask[bad_idx] = True

    valid_frames = (
        np.isfinite(x_cm)
        & np.isfinite(y_cm)
        & np.isfinite(speed)
        & np.isfinite(direction_deg)
        & trace_valid
        & (~bad_frame_mask)
    )

    moving_mask = np.ones(n_frames, dtype=bool)
    speed_smooth_frame = np.asarray(speed, dtype=float).copy()
    if moving_only:
        speed_for_epochs = np.asarray(speed, dtype=float).copy()
        speed_for_epochs[~valid_frames] = np.nan
        speed_smooth_frame, _, moving_idx = _compute_moving_epochs(
            speed_for_epochs,
            float(frame_rate),
            kernel_size=int(moving_kernel_size),
            filter_type=str(moving_filter_type),
            speed_threshold=float(moving_speed_threshold),
            min_duration_s=float(moving_min_duration_s),
            merge_gap_s=float(moving_merge_gap_s),
        )
        moving_mask = np.zeros(n_frames, dtype=bool)
        if len(moving_idx) > 0:
            moving_idx = np.asarray(moving_idx, dtype=int)
            moving_idx = moving_idx[(moving_idx >= 0) & (moving_idx < n_frames)]
            moving_mask[moving_idx] = True

    session_bounds = _session_bounds_from_starts(session_start_frames, n_frames)
    if selected_session_indices is None:
        selected_session_indices = tuple(range(len(session_bounds)))
    selected_session_indices = tuple(
        idx for idx in selected_session_indices if 0 <= int(idx) < len(session_bounds)
    )
    if len(selected_session_indices) == 0:
        raise ValueError("No selected sessions available for LN model.")

    selected_frame_mask = np.zeros(n_frames, dtype=bool)
    for sess_idx in selected_session_indices:
        start, end = session_bounds[int(sess_idx)]
        selected_frame_mask[int(start):int(end)] = True

    analysis_frame_mask = valid_frames & selected_frame_mask
    if moving_only:
        analysis_frame_mask &= moving_mask

    spike_frames = np.asarray(spike_frames, dtype=int)
    spike_frames = spike_frames[(spike_frames >= 0) & (spike_frames < n_frames)]
    spike_frames = spike_frames[analysis_frame_mask[spike_frames]]

    bin_size_frames = max(1, int(round(float(bin_size_s) * float(frame_rate))))
    dt = float(bin_size_frames) / float(frame_rate)

    rows: list[dict[str, Any]] = []
    spike_counts_by_frame = np.zeros(n_frames, dtype=int)
    if spike_frames.size > 0:
        np.add.at(spike_counts_by_frame, spike_frames, 1)

    for sess_idx in selected_session_indices:
        start, end = session_bounds[int(sess_idx)]
        edges = np.arange(int(start), int(end) + 1, int(bin_size_frames))
        if edges[-1] != int(end):
            edges = np.append(edges, int(end))
        for b_start, b_end in zip(edges[:-1], edges[1:]):
            b_start = int(b_start)
            b_end = int(b_end)
            n_bin_frames = int(b_end - b_start)
            valid_slice = analysis_frame_mask[b_start:b_end]
            n_valid = int(np.sum(valid_slice))
            if require_full_valid_bin:
                if n_bin_frames != int(bin_size_frames) or n_valid != n_bin_frames:
                    continue
            elif n_valid <= 0:
                continue
            rows.append(
                {
                    "session_idx": int(sess_idx),
                    "frame_start": b_start,
                    "frame_end": b_end,
                    "x_cm": float(np.nanmean(x_cm[b_start:b_end][valid_slice])),
                    "y_cm": float(np.nanmean(y_cm[b_start:b_end][valid_slice])),
                    "speed_cm_s": float(np.nanmean(speed[b_start:b_end][valid_slice])),
                    "direction_deg": float(np.nanmean(direction_deg[b_start:b_end][valid_slice])),
                    "spike_count": int(np.sum(spike_counts_by_frame[b_start:b_end])),
                    "n_valid_frames": n_valid,
                    "time_s": 0.5 * (float(b_start + b_end) / float(frame_rate)),
                }
            )

    if len(rows) == 0:
        raise ValueError("No valid LN bins after filtering.")

    out: dict[str, Any] = {k: np.asarray([row[k] for row in rows]) for k in rows[0].keys()}
    out["dt"] = float(dt)
    out["analysis_frame_mask"] = np.asarray(analysis_frame_mask, dtype=bool)
    out["session_bounds"] = session_bounds
    out["selected_session_indices"] = tuple(int(v) for v in selected_session_indices)
    out["speed_smooth_frame"] = np.asarray(speed_smooth_frame, dtype=float)
    out["moving_mask"] = np.asarray(moving_mask, dtype=bool)
    return out


def _nearest_linear_bin(values: np.ndarray, centers: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float).ravel()
    centers = np.asarray(centers, dtype=float).ravel()
    idx = np.argmin(np.abs(values[:, None] - centers[None, :]), axis=1)
    return np.asarray(idx, dtype=int)


def _nearest_circular_bin_deg(values: np.ndarray, centers_deg: np.ndarray) -> np.ndarray:
    vals = np.mod(np.asarray(values, dtype=float).ravel(), 360.0)
    ctr = np.mod(np.asarray(centers_deg, dtype=float).ravel(), 360.0)
    diffs = np.abs(((vals[:, None] - ctr[None, :] + 180.0) % 360.0) - 180.0)
    return np.asarray(np.argmin(diffs, axis=1), dtype=int)


def _build_position_group(
    x_cm: np.ndarray,
    y_cm: np.ndarray,
    n_x_bins: int,
    n_y_bins: int,
    arena_size_cm: tuple[float, float],
) -> LNGroupData:
    width_cm = float(arena_size_cm[0])
    height_cm = float(arena_size_cm[1])
    x_centers = np.linspace(width_cm / (2.0 * n_x_bins), width_cm - width_cm / (2.0 * n_x_bins), n_x_bins)
    y_centers = np.linspace(height_cm / (2.0 * n_y_bins), height_cm - height_cm / (2.0 * n_y_bins), n_y_bins)
    x_idx = _nearest_linear_bin(x_cm, x_centers)
    y_idx = _nearest_linear_bin(y_cm, y_centers)
    bin_index = np.asarray(y_idx * int(n_x_bins) + x_idx, dtype=int)
    n_obs = int(bin_index.size)
    row = np.arange(n_obs, dtype=int)
    col = bin_index
    data = np.ones(n_obs, dtype=float)
    X = sparse.csr_matrix((data, (row, col)), shape=(n_obs, int(n_x_bins) * int(n_y_bins)))
    return LNGroupData(
        name="P",
        X=X,
        bin_index=bin_index,
        n_bins=int(n_x_bins) * int(n_y_bins),
        penalty=_position_penalty_2d(int(n_x_bins), int(n_y_bins)),
        centers=np.vstack(np.meshgrid(x_centers, y_centers, indexing="xy")).reshape(2, -1).T,
        shape=(int(n_y_bins), int(n_x_bins)),
    )


def _build_direction_group(direction_deg: np.ndarray, n_bins: int) -> LNGroupData:
    centers = np.linspace(180.0 / n_bins, 360.0 - 180.0 / n_bins, int(n_bins))
    bin_index = _nearest_circular_bin_deg(direction_deg, centers)
    n_obs = int(bin_index.size)
    row = np.arange(n_obs, dtype=int)
    data = np.ones(n_obs, dtype=float)
    X = sparse.csr_matrix((data, (row, bin_index)), shape=(n_obs, int(n_bins)))
    return LNGroupData(
        name="H",
        X=X,
        bin_index=bin_index,
        n_bins=int(n_bins),
        penalty=_roughness_penalty_1d(int(n_bins), circular=True),
        centers=centers,
        shape=(int(n_bins),),
    )


def _build_speed_group(speed_cm_s: np.ndarray, n_bins: int, max_speed_cm_s: float) -> LNGroupData:
    speed_clip = np.clip(np.asarray(speed_cm_s, dtype=float), 0.0, float(max_speed_cm_s))
    centers = np.linspace(
        float(max_speed_cm_s) / (2.0 * n_bins),
        float(max_speed_cm_s) - float(max_speed_cm_s) / (2.0 * n_bins),
        int(n_bins),
    )
    bin_index = _nearest_linear_bin(speed_clip, centers)
    n_obs = int(bin_index.size)
    row = np.arange(n_obs, dtype=int)
    data = np.ones(n_obs, dtype=float)
    X = sparse.csr_matrix((data, (row, bin_index)), shape=(n_obs, int(n_bins)))
    return LNGroupData(
        name="S",
        X=X,
        bin_index=bin_index,
        n_bins=int(n_bins),
        penalty=_roughness_penalty_1d(int(n_bins), circular=False),
        centers=centers,
        shape=(int(n_bins),),
    )


def _filter_binned_by_position_occupancy(
    binned: dict[str, Any],
    *,
    position_n_bins: tuple[int, int],
    arena_size_cm: tuple[float, float],
    min_position_bin_occupancy_s: float,
) -> dict[str, Any]:
    threshold_s = float(min_position_bin_occupancy_s)
    if threshold_s <= 0:
        binned["position_occupancy_filter"] = {
            "applied": False,
            "min_position_bin_occupancy_s": float(threshold_s),
            "n_bins_before": int(np.asarray(binned["spike_count"]).size),
            "n_bins_after": int(np.asarray(binned["spike_count"]).size),
            "n_position_bins_kept": np.nan,
            "n_position_bins_total": np.nan,
        }
        return binned

    pos_group = _build_position_group(
        np.asarray(binned["x_cm"], dtype=float),
        np.asarray(binned["y_cm"], dtype=float),
        int(position_n_bins[0]),
        int(position_n_bins[1]),
        arena_size_cm=tuple(arena_size_cm),
    )
    dt = float(binned["dt"])
    occ_counts = np.bincount(pos_group.bin_index, minlength=int(pos_group.n_bins))
    occ_s = occ_counts.astype(float) * dt
    keep_pos_bins = occ_s >= threshold_s
    keep_rows = np.asarray(keep_pos_bins[pos_group.bin_index], dtype=bool)
    n_rows_before = int(keep_rows.size)
    if not np.any(keep_rows):
        raise ValueError("No LN bins remain after applying the position occupancy threshold.")

    out = {}
    for key, value in binned.items():
        if isinstance(value, np.ndarray) and value.ndim >= 1 and value.shape[0] == n_rows_before:
            out[key] = value[keep_rows]
        else:
            out[key] = value
    out["position_occupancy_filter"] = {
        "applied": True,
        "min_position_bin_occupancy_s": float(threshold_s),
        "n_bins_before": int(n_rows_before),
        "n_bins_after": int(np.sum(keep_rows)),
        "n_position_bins_kept": int(np.sum(keep_pos_bins)),
        "n_position_bins_total": int(pos_group.n_bins),
    }
    return out


def _roughness_penalty_1d(n_bins: int, circular: bool) -> sparse.csr_matrix:
    if n_bins <= 1:
        return sparse.csr_matrix((n_bins, n_bins), dtype=float)
    d = sparse.diags(
        diagonals=[-np.ones(n_bins - 1), np.ones(n_bins - 1)],
        offsets=[0, 1],
        shape=(n_bins - 1, n_bins),
        format="csr",
    )
    dd = (d.T @ d).tocsr()
    if circular:
        dd = dd.toarray()
        dd[0, :] = np.roll(dd[1, :], -1)
        dd[-1, :] = np.roll(dd[-2, :], 1)
        return sparse.csr_matrix(dd)
    return dd


def _position_penalty_2d(n_x_bins: int, n_y_bins: int) -> sparse.csr_matrix:
    if n_x_bins <= 1 and n_y_bins <= 1:
        return sparse.csr_matrix((1, 1), dtype=float)
    dd_x = _roughness_penalty_1d(n_x_bins, circular=False)
    dd_y = _roughness_penalty_1d(n_y_bins, circular=False)
    eye_x = sparse.eye(n_x_bins, format="csr")
    eye_y = sparse.eye(n_y_bins, format="csr")
    return (sparse.kron(eye_y, dd_x) + sparse.kron(dd_y, eye_x)).tocsr()


def _build_model_design(
    groups: dict[str, LNGroupData],
    included_groups: tuple[str, ...],
    roughness_weights: dict[str, float],
) -> dict[str, Any]:
    matrices = []
    penalty_blocks = []
    slices: dict[str, slice] = {}
    start = 0
    for group_name in included_groups:
        group = groups[group_name]
        matrices.append(group.X)
        penalty_blocks.append(float(roughness_weights[group_name]) * group.penalty)
        stop = start + int(group.n_bins)
        slices[group_name] = slice(start, stop)
        start = stop
    X = sparse.hstack(matrices, format="csr") if len(matrices) > 1 else matrices[0]
    penalty = sparse.block_diag(penalty_blocks, format="csr") if len(penalty_blocks) > 1 else penalty_blocks[0]
    return {
        "X": X.tocsr(),
        "penalty": penalty.tocsr(),
        "slices": slices,
        "included_groups": tuple(included_groups),
        "n_params": int(start),
    }


def _penalized_poisson_objective(
    params: np.ndarray,
    X: sparse.csr_matrix,
    counts: np.ndarray,
    penalty: sparse.csr_matrix,
) -> tuple[float, np.ndarray]:
    params = np.asarray(params, dtype=float)
    counts = np.asarray(counts, dtype=float)
    u = np.asarray(X @ params, dtype=float).ravel()
    u_safe = np.clip(u, -50.0, 50.0)
    rate = np.exp(u_safe)
    penalty_vec = penalty @ params
    f = float(np.sum(rate - counts * u_safe) + 0.5 * np.dot(params, penalty_vec))
    grad = np.asarray(X.T @ (rate - counts), dtype=float).ravel() + np.asarray(penalty_vec, dtype=float).ravel()
    return f, grad


def _fit_penalized_poisson(
    X: sparse.csr_matrix,
    counts: np.ndarray,
    penalty: sparse.csr_matrix,
    init_param: np.ndarray,
    maxiter: int,
) -> dict[str, Any]:
    counts = np.asarray(counts, dtype=float)
    init_param = np.asarray(init_param, dtype=float)

    def _fun(x: np.ndarray) -> tuple[float, np.ndarray]:
        return _penalized_poisson_objective(x, X, counts, penalty)

    res = optimize.minimize(
        fun=lambda x: _fun(x)[0],
        x0=init_param,
        jac=lambda x: _fun(x)[1],
        method="L-BFGS-B",
        options={"maxiter": int(maxiter), "disp": False},
    )
    if not np.all(np.isfinite(res.x)):
        raise RuntimeError("LN optimizer returned non-finite parameters.")
    return {
        "params": np.asarray(res.x, dtype=float),
        "success": bool(res.success),
        "status": int(getattr(res, "status", 0)),
        "message": str(getattr(res, "message", "")),
        "fun": float(getattr(res, "fun", np.nan)),
        "nit": int(getattr(res, "nit", 0)),
    }


def _compute_fit_metrics(
    *,
    X: sparse.csr_matrix,
    counts: np.ndarray,
    dt: float,
    smooth_kernel: np.ndarray,
    params: np.ndarray,
) -> dict[str, float]:
    counts = np.asarray(counts, dtype=float)
    pred_counts = np.exp(np.clip(np.asarray(X @ params, dtype=float).ravel(), -50.0, 50.0))
    fr = counts / float(dt)
    pred_fr = pred_counts / float(dt)
    smooth_fr = _smooth_rate(fr, smooth_kernel)
    smooth_pred_fr = _smooth_rate(pred_fr, smooth_kernel)
    sse = float(np.nansum((smooth_pred_fr - smooth_fr) ** 2))
    sst = float(np.nansum((smooth_fr - np.nanmean(smooth_fr)) ** 2))
    var_explained = np.nan
    if np.isfinite(sst) and sst > 0:
        var_explained = float(1.0 - (sse / sst))
    correlation = _pearson_corr_safe(smooth_fr, smooth_pred_fr)
    mse = float(np.nanmean((smooth_pred_fr - smooth_fr) ** 2))
    llh_bits_spike = _llh_increase_bits_per_spike(counts, pred_counts)
    return {
        "var_explained": var_explained,
        "correlation": correlation,
        "llh_bits_spike": llh_bits_spike,
        "mse": mse,
        "n_spikes": float(np.sum(counts)),
        "n_timebins": int(counts.size),
    }


def _make_cv_test_indices(n_obs: int, num_folds: int, section_repeats: int) -> list[np.ndarray]:
    sections = int(num_folds) * int(section_repeats)
    if n_obs < max(2, sections):
        raise ValueError("Not enough LN bins for the requested cross-validation scheme.")
    edges = np.round(np.linspace(0, n_obs, sections + 1)).astype(int)
    fold_indices: list[np.ndarray] = []
    for k in range(int(num_folds)):
        parts = []
        for rep in range(int(section_repeats)):
            start = int(edges[k + rep * int(num_folds)])
            end = int(edges[k + rep * int(num_folds) + 1])
            if end > start:
                parts.append(np.arange(start, end, dtype=int))
        if len(parts) == 0:
            fold_indices.append(np.array([], dtype=int))
        else:
            fold_indices.append(np.concatenate(parts))
    return fold_indices


def _cross_validate_model(
    *,
    model_name: str,
    design: dict[str, Any],
    counts: np.ndarray,
    dt: float,
    smooth_kernel: np.ndarray,
    num_folds: int,
    section_repeats: int,
    maxiter: int,
    random_seed: int | None,
) -> dict[str, Any]:
    X = design["X"]
    penalty = design["penalty"]
    counts = np.asarray(counts, dtype=float)
    test_indices_by_fold = _make_cv_test_indices(int(counts.size), int(num_folds), int(section_repeats))
    test_fit_rows = []
    train_fit_rows = []
    param_mat = []
    fold_records = []
    rng = np.random.default_rng(random_seed)
    init_param = 1e-3 * rng.standard_normal(int(design["n_params"]))

    for fold_idx, test_idx in enumerate(test_indices_by_fold):
        if test_idx.size == 0:
            raise ValueError(f"LN fold {fold_idx + 1} has no test samples.")
        train_mask = np.ones(int(counts.size), dtype=bool)
        train_mask[test_idx] = False
        if np.sum(train_mask) <= 0:
            raise ValueError("LN cross-validation produced an empty training split.")
        X_train = X[train_mask]
        y_train = counts[train_mask]
        X_test = X[test_idx]
        y_test = counts[test_idx]

        fit_out = _fit_penalized_poisson(
            X_train,
            y_train,
            penalty,
            init_param=init_param,
            maxiter=maxiter,
        )
        params = np.asarray(fit_out["params"], dtype=float)
        init_param = params.copy()

        train_metrics = _compute_fit_metrics(
            X=X_train,
            counts=y_train,
            dt=dt,
            smooth_kernel=smooth_kernel,
            params=params,
        )
        test_metrics = _compute_fit_metrics(
            X=X_test,
            counts=y_test,
            dt=dt,
            smooth_kernel=smooth_kernel,
            params=params,
        )
        train_fit_rows.append(train_metrics)
        test_fit_rows.append(test_metrics)
        param_mat.append(params)
        fold_records.append(
            {
                "model_name": model_name,
                "fold_idx": int(fold_idx) + 1,
                "optimizer_success": bool(fit_out["success"]),
                "optimizer_status": int(fit_out["status"]),
                "optimizer_message": str(fit_out["message"]),
                "train_llh_bits_spike": train_metrics["llh_bits_spike"],
                "test_llh_bits_spike": test_metrics["llh_bits_spike"],
                "train_var_explained": train_metrics["var_explained"],
                "test_var_explained": test_metrics["var_explained"],
                "train_correlation": train_metrics["correlation"],
                "test_correlation": test_metrics["correlation"],
            }
        )

    param_mean = np.nanmean(np.vstack(param_mat), axis=0)
    full_fit = _fit_penalized_poisson(
        X,
        counts,
        penalty,
        init_param=param_mean,
        maxiter=maxiter,
    )
    full_metrics = _compute_fit_metrics(
        X=X,
        counts=counts,
        dt=dt,
        smooth_kernel=smooth_kernel,
        params=np.asarray(full_fit["params"], dtype=float),
    )
    return {
        "model_name": model_name,
        "included_groups": tuple(design["included_groups"]),
        "test_fit": test_fit_rows,
        "train_fit": train_fit_rows,
        "param_mean": np.asarray(param_mean, dtype=float),
        "param_full": np.asarray(full_fit["params"], dtype=float),
        "full_fit_metrics": full_metrics,
        "fold_records": fold_records,
        "mean_test_llh_bits_spike": float(np.nanmean([r["llh_bits_spike"] for r in test_fit_rows])),
        "sem_test_llh_bits_spike": float(stats.sem([r["llh_bits_spike"] for r in test_fit_rows], nan_policy="omit")),
        "mean_test_var_explained": float(np.nanmean([r["var_explained"] for r in test_fit_rows])),
        "mean_test_correlation": float(np.nanmean([r["correlation"] for r in test_fit_rows])),
        "optimizer_success_full": bool(full_fit["success"]),
        "optimizer_message_full": str(full_fit["message"]),
    }


def _select_best_ln_model(
    model_results: dict[str, dict[str, Any]],
    forward_alpha: float,
    baseline_alpha: float,
) -> dict[str, Any]:
    llh_map = {
        name: np.asarray([row["llh_bits_spike"] for row in out["test_fit"]], dtype=float)
        for name, out in model_results.items()
    }
    single_candidates = ("P", "H", "S")
    top1 = max(single_candidates, key=lambda name: np.nanmean(llh_map[name]))
    double_candidates = {
        "P": ("PH", "PS"),
        "H": ("PH", "HS"),
        "S": ("PS", "HS"),
    }[top1]
    top2 = max(double_candidates, key=lambda name: np.nanmean(llh_map[name]))
    top3 = "PHS"

    p_12 = _wilcoxon_right(llh_map[top2], llh_map[top1])
    p_23 = _wilcoxon_right(llh_map[top3], llh_map[top2])

    if np.isfinite(p_12) and p_12 < float(forward_alpha):
        if np.isfinite(p_23) and p_23 < float(forward_alpha):
            selected = top3
        else:
            selected = top2
    else:
        selected = top1

    baseline_p = _wilcoxon_right(llh_map[selected])
    if np.isfinite(baseline_p) and baseline_p > float(baseline_alpha):
        selected = None

    return {
        "best_single_model": top1,
        "best_double_model": top2,
        "full_model": top3,
        "selected_model": selected,
        "p_12": p_12,
        "p_23": p_23,
        "baseline_p": baseline_p,
        "forward_alpha": float(forward_alpha),
        "baseline_alpha": float(baseline_alpha),
    }


def _nanmean_by_index(values: np.ndarray, indices: np.ndarray, n_bins: int) -> np.ndarray:
    out = np.full(int(n_bins), np.nan, dtype=float)
    values = np.asarray(values, dtype=float)
    indices = np.asarray(indices, dtype=int)
    for bin_idx in range(int(n_bins)):
        in_bin = indices == int(bin_idx)
        if np.any(in_bin):
            out[bin_idx] = float(np.nanmean(values[in_bin]))
    return out


def _fill_nan_with_neighbor_mean_2d(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr, dtype=float).copy()
    if out.ndim != 2:
        return out
    nan_mask = ~np.isfinite(out)
    if not np.any(nan_mask):
        return out
    n_rows, n_cols = out.shape
    nan_idx = np.argwhere(nan_mask)
    for row_idx, col_idx in nan_idx:
        vals = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr = min(max(int(row_idx) + dr, 0), n_rows - 1)
                cc = min(max(int(col_idx) + dc, 0), n_cols - 1)
                val = out[rr, cc]
                if np.isfinite(val):
                    vals.append(float(val))
        if vals:
            out[int(row_idx), int(col_idx)] = float(np.mean(vals))
    return out


def _compute_empirical_tuning(
    *,
    counts: np.ndarray,
    dt: float,
    groups: dict[str, LNGroupData],
    position_shape: tuple[int, int],
    position_smooth_sigma: float,
    smooth_kernel: np.ndarray,
) -> dict[str, Any]:
    fr_hz = np.asarray(counts, dtype=float) / float(dt)
    smooth_fr = _smooth_rate(fr_hz, smooth_kernel)
    pos_curve = _nanmean_by_index(smooth_fr, groups["P"].bin_index, groups["P"].n_bins)
    pos_curve = pos_curve.reshape(position_shape)
    pos_curve = _fill_nan_with_neighbor_mean_2d(pos_curve)
    if float(position_smooth_sigma) > 0:
        pos_curve = gaussian_filter(pos_curve, sigma=float(position_smooth_sigma), mode="nearest")
    hd_curve = _nanmean_by_index(smooth_fr, groups["H"].bin_index, groups["H"].n_bins)
    speed_curve = _nanmean_by_index(smooth_fr, groups["S"].bin_index, groups["S"].n_bins)
    return {
        "position_hz": pos_curve,
        "direction_hz": hd_curve,
        "speed_hz": speed_curve,
        "smoothed_fr_hz": smooth_fr,
    }


def _compute_model_profiles(
    *,
    groups: dict[str, LNGroupData],
    model_design: dict[str, Any],
    params: np.ndarray,
    dt: float,
) -> dict[str, Any]:
    params = np.asarray(params, dtype=float)
    included = tuple(model_design["included_groups"])
    slices = dict(model_design["slices"])

    u_by_group: dict[str, np.ndarray] = {}
    for group_name in included:
        sl = slices[group_name]
        u_by_group[group_name] = np.asarray(groups[group_name].X @ params[sl], dtype=float).ravel()

    total_u = np.zeros(groups["P"].X.shape[0], dtype=float)
    for group_name in included:
        total_u += u_by_group[group_name]

    profiles: dict[str, Any] = {}
    for group_name in ("P", "H", "S"):
        group = groups[group_name]
        if group_name in included:
            sl = slices[group_name]
            group_param = params[sl]
            other_u = total_u - u_by_group[group_name]
            prof = np.zeros(group.n_bins, dtype=float)
            for bin_idx in range(group.n_bins):
                prof[bin_idx] = float(
                    np.mean(np.exp(np.clip(other_u + float(group_param[bin_idx]), -50.0, 50.0)))
                    / float(dt)
                )
        else:
            prof = np.full(group.n_bins, float(np.mean(np.exp(np.clip(total_u, -50.0, 50.0))) / float(dt)))
        profiles[group_name] = prof

    return {
        "position_hz": profiles["P"].reshape(groups["P"].shape),
        "direction_hz": profiles["H"],
        "speed_hz": profiles["S"],
        "pred_rate_hz": np.exp(np.clip(total_u, -50.0, 50.0)) / float(dt),
    }


def fit_hardcastle_ln_model(
    *,
    spike_frames: np.ndarray,
    x_cm: np.ndarray,
    y_cm: np.ndarray,
    speed: np.ndarray,
    hd_deg: np.ndarray,
    frame_rate: float,
    session_start_frames: list[int] | np.ndarray | None = None,
    selected_session_indices: tuple[int, ...] | None = None,
    bad_mask: np.ndarray | None = None,
    trace_for_nan: np.ndarray | None = None,
    bin_size_s: float = 0.02,
    moving_only: bool = True,
    moving_speed_threshold: float = 3.0,
    moving_kernel_size: int = 51,
    moving_filter_type: str = "boxcar",
    moving_min_duration_s: float = 0.25,
    moving_merge_gap_s: float = 0.0,
    direction_predictor: str = "head",
    travel_smooth_window: int = 5,
    travel_min_step: float = 0.0,
    require_full_valid_bin: bool = True,
    arena_size_cm: tuple[float, float] = (35.5, 20.0),
    position_n_bins: tuple[int, int] = (20, 12),
    min_position_bin_occupancy_s: float = 0.0,
    n_direction_bins: int = 18,
    n_speed_bins: int = 10,
    max_speed_cm_s: float = 50.0,
    roughness_position: float = 8.0,
    roughness_direction: float = 50.0,
    roughness_speed: float = 50.0,
    cv_num_folds: int = 10,
    cv_section_repeats: int = 5,
    selection_alpha: float = 0.05,
    baseline_alpha: float | None = None,
    smoothing_sigma_bins: float = 2.0,
    smoothing_radius_bins: int = 4,
    optimizer_maxiter: int = 400,
    position_smooth_sigma: float = 0.5,
    random_seed: int | None = None,
    min_total_spikes: int = 10,
) -> dict[str, Any]:
    """Fit Hardcastle-style P/H/S LN models for one cell."""
    binned = _prepare_binned_ln_data(
        spike_frames=spike_frames,
        x_cm=x_cm,
        y_cm=y_cm,
        speed=speed,
        hd_deg=hd_deg,
        frame_rate=float(frame_rate),
        session_start_frames=session_start_frames,
        selected_session_indices=selected_session_indices,
        bin_size_s=float(bin_size_s),
        moving_only=bool(moving_only),
        moving_speed_threshold=float(moving_speed_threshold),
        moving_kernel_size=int(moving_kernel_size),
        moving_filter_type=str(moving_filter_type),
        moving_min_duration_s=float(moving_min_duration_s),
        moving_merge_gap_s=float(moving_merge_gap_s),
        bad_mask=bad_mask,
        trace_for_nan=trace_for_nan,
        direction_predictor=str(direction_predictor),
        travel_smooth_window=int(travel_smooth_window),
        travel_min_step=float(travel_min_step),
        require_full_valid_bin=bool(require_full_valid_bin),
    )
    binned = _filter_binned_by_position_occupancy(
        binned,
        position_n_bins=tuple(position_n_bins),
        arena_size_cm=tuple(arena_size_cm),
        min_position_bin_occupancy_s=float(min_position_bin_occupancy_s),
    )

    counts = np.asarray(binned["spike_count"], dtype=float)
    if float(np.sum(counts)) < int(min_total_spikes):
        raise ValueError("Insufficient spikes after LN binning and masking.")

    groups = {
        "P": _build_position_group(
            binned["x_cm"],
            binned["y_cm"],
            int(position_n_bins[0]),
            int(position_n_bins[1]),
            arena_size_cm=tuple(arena_size_cm),
        ),
        "H": _build_direction_group(
            binned["direction_deg"],
            int(n_direction_bins),
        ),
        "S": _build_speed_group(
            binned["speed_cm_s"],
            int(n_speed_bins),
            float(max_speed_cm_s),
        ),
    }
    roughness_weights = {
        "P": float(roughness_position),
        "H": float(roughness_direction),
        "S": float(roughness_speed),
    }
    smooth_kernel = _gaussian_kernel(
        radius=int(smoothing_radius_bins),
        sigma=float(smoothing_sigma_bins),
    )

    model_results: dict[str, dict[str, Any]] = {}
    for model_name, included_groups in LN_MODEL_SPECS:
        design = _build_model_design(groups, included_groups, roughness_weights)
        model_results[model_name] = {
            **_cross_validate_model(
                model_name=model_name,
                design=design,
                counts=counts,
                dt=float(binned["dt"]),
                smooth_kernel=smooth_kernel,
                num_folds=int(cv_num_folds),
                section_repeats=int(cv_section_repeats),
                maxiter=int(optimizer_maxiter),
                random_seed=None if random_seed is None else int(random_seed) + len(model_results),
            ),
            "design": design,
        }

    if baseline_alpha is None:
        baseline_alpha = float(selection_alpha)
    selection = _select_best_ln_model(
        model_results,
        forward_alpha=float(selection_alpha),
        baseline_alpha=float(baseline_alpha),
    )
    empirical_tuning = _compute_empirical_tuning(
        counts=counts,
        dt=float(binned["dt"]),
        groups=groups,
        position_shape=groups["P"].shape,
        position_smooth_sigma=float(position_smooth_sigma),
        smooth_kernel=smooth_kernel,
    )
    full_profiles = _compute_model_profiles(
        groups=groups,
        model_design=model_results["PHS"]["design"],
        params=np.asarray(model_results["PHS"]["param_full"], dtype=float),
        dt=float(binned["dt"]),
    )
    selected_profiles = None
    selected_model = selection.get("selected_model")
    if isinstance(selected_model, str):
        selected_profiles = _compute_model_profiles(
            groups=groups,
            model_design=model_results[selected_model]["design"],
            params=np.asarray(model_results[selected_model]["param_full"], dtype=float),
            dt=float(binned["dt"]),
        )

    return {
        "params": {
            "bin_size_s": float(bin_size_s),
            "moving_only": bool(moving_only),
            "moving_speed_threshold": float(moving_speed_threshold),
            "require_full_valid_bin": bool(require_full_valid_bin),
            "direction_predictor": str(direction_predictor).strip().lower(),
            "arena_size_cm": tuple(float(v) for v in arena_size_cm),
            "position_n_bins": tuple(int(v) for v in position_n_bins),
            "min_position_bin_occupancy_s": float(min_position_bin_occupancy_s),
            "n_direction_bins": int(n_direction_bins),
            "n_speed_bins": int(n_speed_bins),
            "max_speed_cm_s": float(max_speed_cm_s),
            "cv_num_folds": int(cv_num_folds),
            "cv_section_repeats": int(cv_section_repeats),
            "selection_alpha": float(selection_alpha),
            "baseline_alpha": float(baseline_alpha),
        },
        "binned": binned,
        "groups": groups,
        "models": model_results,
        "selection": selection,
        "empirical_tuning": empirical_tuning,
        "full_model_profiles": full_profiles,
        "selected_model_profiles": selected_profiles,
        "n_bins": int(counts.size),
        "n_spikes_total": int(np.sum(counts)),
        "dt": float(binned["dt"]),
        "position_occupancy_filter": dict(binned.get("position_occupancy_filter", {})),
    }
