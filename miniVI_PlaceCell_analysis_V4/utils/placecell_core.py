import os
import copy
import glob
import math
import cv2
import numpy as np
import numpy.ma as ma
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import statsmodels.api as sm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.ndimage import gaussian_filter, label, binary_dilation, maximum_filter
from scipy.ndimage import median_filter
from scipy.ndimage import uniform_filter1d, gaussian_filter1d
from scipy import signal
from utils.spatial_analysis_func import calculate_spatial_information
from scipy.signal import medfilt
from utils.spike_detection import (
    interpolate_nan_segment,
    detect_burst_SS,
    lowpass_filter,
)
from utils.spatial_analysis_func import bandpass_filter

plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 6,
        "axes.labelsize": 6,
        "axes.titlesize": 6,
        "xtick.labelsize": 5,
        "ytick.labelsize": 5,
        "xtick.major.size": 1.75,
        "ytick.major.size": 1.75,
        "xtick.minor.size": 1.0,
        "ytick.minor.size": 1.0,
        "legend.fontsize": 5,
        "axes.linewidth": 0.5,
        "svg.fonttype": "none",
    }
)


def fit_behavior_glm_statsmodels(
    spike_frames,
    x_cm,
    y_cm,
    speed,
    hd_deg,
    frame_rate,
    session_start_frames=None,
    accel=None,
    trace_for_nan=None,
    bad_mask=None,
    bin_size_s=0.02,
    arena_size_cm=(35.5, 20.0),
    rbf_grid=(7, 4),
    rbf_sigma_cm=4.5,
    standardize=True,
    include_session=True,
    include_speed_state=False,
    speed_threshold=5.0,
    moving_only=False,
    spike_smooth_sigma_bins=1.0,
    include_travel_dir=True,
    include_accel=True,
    direction_predictor=None,
    selected_session_indices=None,
    moving_kernel_size=51,
    moving_filter_type="boxcar",
    moving_min_duration_s=0.25,
    moving_merge_gap_s=0.0,
    require_full_valid_bin=False,
    include_pos_hd=False,
    include_pos_speed=False,
    include_pos_state=False,
    include_pos_travel_dir=False,
    smooth_behavior=True,
    use_regularization=False,
    reg_alpha=1.0,
    reg_l1_wt=0.0,
    min_total_spikes=None,
):
    """
    Fit a Poisson GLM (statsmodels) for one neuron's spikes against behavior.

    Parameters
    ----------
    bad_mask : array-like, optional
        Boolean array of shape (n_frames,) where True indicates bad frames
        (e.g., low SNR) that should be excluded from the analysis.
    moving_only : bool, optional
        If True, only include bins where speed >= speed_threshold (moving state).
        Default is False.
    direction_predictor : {"head", "travel", "both"} or None, optional
        Explicit directional predictor selection. If None, preserve legacy behavior:
        include head direction always, and include travel direction when
        ``include_travel_dir=True``.
    selected_session_indices : tuple[int, ...] or None, optional
        If provided, keep only bins from these session indices before fitting.

    Returns a dict with the fitted result, design matrix, and metadata.
    """
    x_cm = np.asarray(x_cm, dtype=float)
    y_cm = np.asarray(y_cm, dtype=float)
    speed = np.asarray(speed, dtype=float)
    hd_deg = np.asarray(hd_deg, dtype=float)
    n_frames = len(x_cm)

    if accel is None:
        accel = np.gradient(speed) * float(frame_rate)
    accel = np.asarray(accel, dtype=float)
    if trace_for_nan is not None:
        trace_for_nan = np.asarray(trace_for_nan, dtype=float)
    if bad_mask is not None:
        bad_mask = np.asarray(bad_mask, dtype=bool)
        if len(bad_mask) != n_frames:
            raise ValueError(
                f"bad_mask length ({len(bad_mask)}) must match n_frames ({n_frames})"
            )

    bin_size_frames = max(1, int(round(bin_size_s * frame_rate)))
    bin_size_s = bin_size_frames / float(frame_rate)
    if session_start_frames is None or len(session_start_frames) == 0:
        session_start_frames = [0]
    session_start_frames = sorted(set(int(s) for s in session_start_frames))
    session_start_frames = [s for s in session_start_frames if s < n_frames]
    session_ends = session_start_frames[1:] + [n_frames]

    if selected_session_indices is not None:
        selected_session_indices = tuple(sorted({int(i) for i in selected_session_indices if int(i) >= 0}))
    else:
        selected_session_indices = None

    if direction_predictor is None:
        resolved_direction_predictor = "both" if include_travel_dir else "head"
    else:
        resolved_direction_predictor = str(direction_predictor).strip().lower()
    if resolved_direction_predictor not in {"head", "travel", "both"}:
        raise ValueError(
            f"Unsupported direction_predictor '{direction_predictor}'. "
            "Use 'head', 'travel', or 'both'."
        )

    trace_valid_mask = np.ones(n_frames, dtype=bool)
    if trace_for_nan is not None:
        trace_valid_mask = np.isfinite(trace_for_nan)

    bad_frame_mask = np.zeros(n_frames, dtype=bool)
    if bad_mask is not None:
        bad_frame_mask = bad_mask.copy()

    hd_rad = np.deg2rad(hd_deg)
    dx = np.gradient(x_cm)
    dy = np.gradient(y_cm)
    travel_rad = np.arctan2(dy, dx)

    frame_valid_mask = np.isfinite(x_cm) & np.isfinite(y_cm) & np.isfinite(speed)
    if include_accel:
        frame_valid_mask &= np.isfinite(accel)
    if resolved_direction_predictor in {"head", "both"}:
        frame_valid_mask &= np.isfinite(hd_rad)
    if resolved_direction_predictor in {"travel", "both"}:
        frame_valid_mask &= np.isfinite(travel_rad)
    frame_valid_mask &= trace_valid_mask
    frame_valid_mask &= ~bad_frame_mask

    analysis_frame_mask = frame_valid_mask.copy()
    moving_frame_mask = np.ones(n_frames, dtype=bool)
    speed_smooth_frame = np.asarray(speed, dtype=float).copy()
    if moving_only:
        speed_for_epochs = np.asarray(speed, dtype=float).copy()
        speed_for_epochs[~frame_valid_mask] = np.nan
        speed_smooth_frame, _, moving_idx = _compute_moving_epochs(
            speed_for_epochs,
            frame_rate,
            kernel_size=int(moving_kernel_size),
            filter_type=str(moving_filter_type),
            speed_threshold=float(speed_threshold),
            min_duration_s=float(moving_min_duration_s),
            merge_gap_s=float(moving_merge_gap_s),
        )
        moving_frame_mask = np.zeros(n_frames, dtype=bool)
        if len(moving_idx) > 0:
            moving_idx = np.asarray(moving_idx, dtype=int)
            moving_idx = moving_idx[(moving_idx >= 0) & (moving_idx < n_frames)]
            moving_idx = moving_idx[frame_valid_mask[moving_idx]]
            moving_frame_mask[moving_idx] = True
        analysis_frame_mask &= moving_frame_mask

    spike_frames = np.asarray(spike_frames, dtype=int)
    spike_frames = spike_frames[(spike_frames >= 0) & (spike_frames < n_frames)]
    if spike_frames.size > 0:
        spike_frames = spike_frames[analysis_frame_mask[spike_frames]]

    selected_session_id_set = set(selected_session_indices) if selected_session_indices is not None else None

    bin_edges = []
    bin_session_ids = []
    for sess_id, (start, end) in enumerate(zip(session_start_frames, session_ends)):
        if selected_session_id_set is not None and sess_id not in selected_session_id_set:
            continue
        edges = np.arange(start, end + 1, bin_size_frames)
        if edges[-1] != end:
            edges = np.append(edges, end)
        for i in range(len(edges) - 1):
            bin_edges.append((edges[i], edges[i + 1]))
            bin_session_ids.append(sess_id)

    if len(bin_edges) == 0:
        raise ValueError("No bins available after applying selected_session_indices.")

    bin_session_ids = np.asarray(bin_session_ids, dtype=int)
    bin_counts = np.zeros(len(bin_edges), dtype=float)
    x_bin = np.full(len(bin_edges), np.nan)
    y_bin = np.full(len(bin_edges), np.nan)
    speed_bin = np.full(len(bin_edges), np.nan)
    accel_bin = np.full(len(bin_edges), np.nan)
    sin_hd_bin = np.full(len(bin_edges), np.nan)
    cos_hd_bin = np.full(len(bin_edges), np.nan)
    sin_travel_bin = np.full(len(bin_edges), np.nan)
    cos_travel_bin = np.full(len(bin_edges), np.nan)
    hd_deg_bin = np.full(len(bin_edges), np.nan)
    travel_deg_bin = np.full(len(bin_edges), np.nan)

    bin_valid = np.ones(len(bin_edges), dtype=bool)
    bin_frame_count = np.zeros(len(bin_edges), dtype=int)
    bin_valid_frame_count = np.zeros(len(bin_edges), dtype=int)
    bin_exposure_s = np.zeros(len(bin_edges), dtype=float)
    for i, (b_start, b_end) in enumerate(bin_edges):
        n_bin_frames = int(b_end - b_start)
        bin_frame_count[i] = n_bin_frames
        valid_slice = analysis_frame_mask[b_start:b_end]
        n_valid_frames = int(np.sum(valid_slice))
        bin_valid_frame_count[i] = n_valid_frames
        bin_exposure_s[i] = n_valid_frames / float(frame_rate)
        if require_full_valid_bin:
            if n_bin_frames != int(bin_size_frames) or n_valid_frames != n_bin_frames:
                bin_valid[i] = False
                continue
        elif n_valid_frames <= 0:
            bin_valid[i] = False
            continue

        mask = (spike_frames >= b_start) & (spike_frames < b_end)
        bin_counts[i] = np.sum(mask)

        xs = x_cm[b_start:b_end][valid_slice]
        ys = y_cm[b_start:b_end][valid_slice]
        sp = speed[b_start:b_end][valid_slice]
        ac = accel[b_start:b_end][valid_slice]
        hd = hd_rad[b_start:b_end][valid_slice]
        tr = travel_rad[b_start:b_end][valid_slice]

        x_bin[i] = np.nanmean(xs) if xs.size else np.nan
        y_bin[i] = np.nanmean(ys) if ys.size else np.nan
        speed_bin[i] = np.nanmean(sp) if sp.size else np.nan
        accel_bin[i] = np.nanmean(ac) if ac.size else np.nan
        hd_finite = np.isfinite(hd)
        tr_finite = np.isfinite(tr)
        if np.any(hd_finite):
            sin_hd_bin[i] = np.mean(np.sin(hd[hd_finite]))
            cos_hd_bin[i] = np.mean(np.cos(hd[hd_finite]))
            hd_deg_bin[i] = np.rad2deg(np.arctan2(sin_hd_bin[i], cos_hd_bin[i]))
        if np.any(tr_finite):
            sin_travel_bin[i] = np.mean(np.sin(tr[tr_finite]))
            cos_travel_bin[i] = np.mean(np.cos(tr[tr_finite]))
            travel_deg_bin[i] = np.rad2deg(np.arctan2(sin_travel_bin[i], cos_travel_bin[i]))

    def _smooth_with_mask(values, mask, window_bins):
        arr = np.asarray(values, dtype=float)
        if window_bins <= 0:
            return arr.copy()
        window_bins = int(max(1, window_bins))
        kernel = np.ones(window_bins, dtype=float)
        kernel /= np.sum(kernel)
        arr_filled = np.where(mask, arr, 0.0)
        weights = np.where(mask & np.isfinite(arr), 1.0, 0.0)
        num = np.convolve(arr_filled, kernel, mode="same")
        den = np.convolve(weights, kernel, mode="same")
        out = np.full_like(arr, np.nan, dtype=float)
        valid = den > 0
        out[valid] = num[valid] / den[valid]
        return out

    y_raw = bin_counts.astype(float)
    if spike_smooth_sigma_bins > 0:
        y_model = _smooth_with_mask(y_raw, bin_valid, spike_smooth_sigma_bins)
    else:
        y_model = y_raw.copy()
    exposure_model = bin_exposure_s.copy()

    x_model = x_bin.copy()
    y_model_pos = y_bin.copy()
    speed_model = speed_bin.copy()
    accel_model = accel_bin.copy()
    sin_hd_model = sin_hd_bin.copy()
    cos_hd_model = cos_hd_bin.copy()
    sin_travel_model = sin_travel_bin.copy()
    cos_travel_model = cos_travel_bin.copy()
    hd_deg_model = hd_deg_bin.copy()
    travel_deg_model = travel_deg_bin.copy()

    if smooth_behavior and spike_smooth_sigma_bins > 0:
        x_model = _smooth_with_mask(x_model, bin_valid, spike_smooth_sigma_bins)
        y_model_pos = _smooth_with_mask(y_model_pos, bin_valid, spike_smooth_sigma_bins)
        speed_model = _smooth_with_mask(speed_model, bin_valid, spike_smooth_sigma_bins)
        accel_model = _smooth_with_mask(accel_model, bin_valid, spike_smooth_sigma_bins)
        sin_hd_model = _smooth_with_mask(sin_hd_model, bin_valid, spike_smooth_sigma_bins)
        cos_hd_model = _smooth_with_mask(cos_hd_model, bin_valid, spike_smooth_sigma_bins)
        sin_travel_model = _smooth_with_mask(sin_travel_model, bin_valid, spike_smooth_sigma_bins)
        cos_travel_model = _smooth_with_mask(cos_travel_model, bin_valid, spike_smooth_sigma_bins)
        hd_deg_model = _smooth_with_mask(hd_deg_model, bin_valid, spike_smooth_sigma_bins)
        travel_deg_model = _smooth_with_mask(
            travel_deg_model, bin_valid, spike_smooth_sigma_bins
        )

    arena_w, arena_h = arena_size_cm
    grid_x = np.linspace(0, arena_w, rbf_grid[0])
    grid_y = np.linspace(0, arena_h, rbf_grid[1])
    centers = np.array([(gx, gy) for gx in grid_x for gy in grid_y], dtype=float)

    rbf_features = np.full((len(bin_edges), len(centers)), np.nan)
    for i, (xb, yb) in enumerate(zip(x_model, y_model_pos)):
        if not (np.isfinite(xb) and np.isfinite(yb)):
            continue
        d2 = (centers[:, 0] - xb) ** 2 + (centers[:, 1] - yb) ** 2
        rbf_features[i, :] = np.exp(-d2 / (2.0 * rbf_sigma_cm ** 2))

    if include_session:
        session_dummies = pd.get_dummies(bin_session_ids, drop_first=True).to_numpy()
    else:
        session_dummies = np.zeros((len(bin_edges), 0), dtype=float)

    speed_feat = speed_model.copy()
    accel_feat = accel_model.copy()
    moving_bin = speed_model >= speed_threshold
    quiet_bin = ~moving_bin
    speed_moving = speed_feat * moving_bin.astype(float)

    feature_arrays = []
    feature_names = []
    feature_is_continuous = []

    def _append_feature_block(block, names, *, continuous):
        block_arr = np.asarray(block, dtype=float)
        if block_arr.ndim == 1:
            block_arr = block_arr[:, None]
        if block_arr.shape[1] != len(names):
            raise ValueError("Feature block/name mismatch while building GLM design matrix.")
        if block_arr.shape[1] == 0:
            return
        feature_arrays.append(block_arr)
        feature_names.extend(list(names))
        feature_is_continuous.extend([bool(continuous)] * block_arr.shape[1])

    _append_feature_block(
        rbf_features,
        [f"pos_rbf_{k}" for k in range(len(centers))],
        continuous=True,
    )
    if resolved_direction_predictor in {"head", "both"}:
        _append_feature_block(sin_hd_model, ["hd_sin"], continuous=True)
        _append_feature_block(cos_hd_model, ["hd_cos"], continuous=True)
    if resolved_direction_predictor in {"travel", "both"}:
        _append_feature_block(sin_travel_model, ["travel_sin"], continuous=True)
        _append_feature_block(cos_travel_model, ["travel_cos"], continuous=True)

    if include_speed_state:
        _append_feature_block(quiet_bin.astype(float), ["quiet"], continuous=False)
        _append_feature_block(speed_moving, ["speed_moving"], continuous=True)
    else:
        _append_feature_block(speed_feat, ["speed"], continuous=True)

    if include_accel:
        _append_feature_block(accel_feat, ["accel"], continuous=True)
    if session_dummies.shape[1] > 0:
        _append_feature_block(
            session_dummies,
            [f"session_{k + 1}" for k in range(session_dummies.shape[1])],
            continuous=False,
        )

    interaction_blocks = []
    interaction_names = []
    if include_pos_hd:
        interaction_blocks.append(rbf_features * sin_hd_model[:, None])
        interaction_blocks.append(rbf_features * cos_hd_model[:, None])
        interaction_names.extend([f"pos_x_hd_sin_{k}" for k in range(len(centers))])
        interaction_names.extend([f"pos_x_hd_cos_{k}" for k in range(len(centers))])
    if include_pos_speed:
        speed_for_interaction = speed_moving if include_speed_state else speed_feat
        interaction_blocks.append(rbf_features * speed_for_interaction[:, None])
        interaction_names.extend([f"pos_x_speed_{k}" for k in range(len(centers))])
    if include_pos_state and include_speed_state:
        interaction_blocks.append(rbf_features * quiet_bin.astype(float)[:, None])
        interaction_names.extend([f"pos_x_state_{k}" for k in range(len(centers))])
    if include_pos_travel_dir:
        interaction_blocks.append(rbf_features * sin_travel_model[:, None])
        interaction_blocks.append(rbf_features * cos_travel_model[:, None])
        interaction_names.extend([f"pos_x_travel_sin_{k}" for k in range(len(centers))])
        interaction_names.extend([f"pos_x_travel_cos_{k}" for k in range(len(centers))])

    if interaction_blocks:
        _append_feature_block(
            np.hstack(interaction_blocks),
            interaction_names,
            continuous=True,
        )

    X = np.hstack(feature_arrays)
    valid_mask = np.isfinite(y_model)
    valid_mask &= np.all(np.isfinite(X), axis=1)
    valid_mask &= bin_valid
    valid_mask &= np.isfinite(exposure_model) & (exposure_model > 0)

    standardization_stats = {}
    if standardize and np.any(valid_mask):
        for col_idx, name in enumerate(feature_names):
            if not feature_is_continuous[col_idx]:
                standardization_stats[name] = {
                    "mean": np.nan,
                    "std": np.nan,
                    "applied": False,
                }
                continue
            col = X[:, col_idx]
            mu = float(np.nanmean(col[valid_mask]))
            sigma = float(np.nanstd(col[valid_mask]))
            applied = bool(np.isfinite(mu) and np.isfinite(sigma) and sigma > 0)
            if applied:
                X[:, col_idx] = (col - mu) / sigma
            standardization_stats[name] = {
                "mean": mu,
                "std": sigma,
                "applied": applied,
            }
    else:
        for col_idx, name in enumerate(feature_names):
            standardization_stats[name] = {
                "mean": np.nan,
                "std": np.nan,
                "applied": False,
            }

    valid_mask = np.isfinite(y_model)
    valid_mask &= np.all(np.isfinite(X), axis=1)
    valid_mask &= bin_valid
    valid_mask &= np.isfinite(exposure_model) & (exposure_model > 0)
    X_valid = X[valid_mask]
    y_valid = y_model[valid_mask]
    exposure_valid = exposure_model[valid_mask]

    def _compute_deviance(family, y, mu):
        return float(family.deviance(y, mu))

    # Check for sufficient valid data
    n_valid = np.sum(valid_mask)
    n_features = X_valid.shape[1] + 1  # +1 for constant
    if n_valid < n_features + 5:
        raise ValueError(
            f"Insufficient valid bins for GLM fitting: {n_valid} valid bins, "
            f"but need at least {n_features + 5} (features + buffer). "
            f"Try reducing bin_size_s, relaxing bad_mask, or using less covariates."
        )
    
    # Check for any remaining NaN/inf values
    if not np.all(np.isfinite(X_valid)):
        raise ValueError("X_valid contains NaN or inf values after filtering.")
    if not np.all(np.isfinite(y_valid)):
        raise ValueError("y_valid contains NaN or inf values after filtering.")
    if not np.all(np.isfinite(exposure_valid)) or np.any(exposure_valid <= 0):
        raise ValueError("Exposure contains invalid or non-positive values after filtering.")

    n_spikes_total = int(np.sum(y_raw[valid_mask])) if np.any(valid_mask) else 0
    if min_total_spikes is not None and n_spikes_total < int(min_total_spikes):
        raise ValueError(
            "Insufficient spikes after frame-level NaN/bad-frame exclusion: "
            f"{n_spikes_total} < {int(min_total_spikes)}."
        )

    X_with_const = sm.add_constant(X_valid, has_constant="add")
    model = sm.GLM(
        y_valid,
        X_with_const,
        family=sm.families.Poisson(),
        exposure=exposure_valid,
    )
    if use_regularization:
        result = model.fit_regularized(alpha=reg_alpha, L1_wt=reg_l1_wt)
    else:
        result = model.fit()
    mu = result.fittedvalues
    deviance = _compute_deviance(model.family, y_valid, mu)
    null_rate_hz = float(np.sum(y_valid) / np.sum(exposure_valid))
    mu_null = exposure_valid * null_rate_hz
    null_deviance = _compute_deviance(model.family, y_valid, mu_null)
    pseudo_r2 = 1.0 - (deviance / null_deviance) if null_deviance else np.nan

    bin_centers = np.array([(s + e) * 0.5 for s, e in bin_edges], dtype=float)
    bin_centers_s = bin_centers / float(frame_rate)
    resolved_session_indices_out = tuple(sorted({int(v) for v in np.unique(bin_session_ids)}))

    return {
        "model": model,
        "result": result,
        "deviance": deviance,
        "null_deviance": null_deviance,
        "pseudo_r2": pseudo_r2,
        "X": X_valid,
        "y": y_valid,
        "y_raw": y_raw,
        "y_smooth": y_model,
        "exposure_s": exposure_valid,
        "bin_exposure_s": exposure_model,
        "bin_frame_count": bin_frame_count,
        "bin_valid_frame_count": bin_valid_frame_count,
        "base_frame_valid_mask": frame_valid_mask,
        "frame_valid_mask": analysis_frame_mask,
        "moving_frame_mask": moving_frame_mask,
        "speed_smooth_frame": speed_smooth_frame,
        "spike_frames_used": spike_frames,
        "fitted_count": np.asarray(mu, dtype=float),
        "fitted_rate_hz": np.asarray(mu, dtype=float) / exposure_valid,
        "feature_names": ["const"] + feature_names,
        "bin_edges": bin_edges,
        "bin_session_ids": bin_session_ids,
        "valid_mask": valid_mask,
        "bin_size_frames": bin_size_frames,
        "bin_size_s": bin_size_s,
        "frame_rate": float(frame_rate),
        "bin_centers_s": bin_centers_s,
        "rbf_centers": centers,
        "rbf_sigma_cm": float(rbf_sigma_cm),
        "arena_size_cm": tuple(arena_size_cm),
        "rbf_grid": tuple(rbf_grid),
        "use_regularization": use_regularization,
        "reg_alpha": reg_alpha,
        "reg_l1_wt": reg_l1_wt,
        "direction_predictor": resolved_direction_predictor,
        "selected_session_indices": resolved_session_indices_out,
        "n_valid_bins": int(n_valid),
        "n_spikes_total": int(n_spikes_total),
        "require_full_valid_bin": bool(require_full_valid_bin),
        "standardization_stats": standardization_stats,
        "behavior_bins": {
            "x_cm": x_model,
            "y_cm": y_model_pos,
            "speed": speed_model,
            "accel": accel_model,
            "hd_deg": hd_deg_model,
            "hd_sin": sin_hd_model,
            "hd_cos": cos_hd_model,
            "travel_deg": travel_deg_model,
            "travel_sin": sin_travel_model,
            "travel_cos": cos_travel_model,
        },
    }


def plot_observed_vs_fitted_rate(
    glm_out,
    ax=None,
    max_points=2000,
    voltage_trace=None,
    frame_rate=None,
    spike_frames=None,
    behavior_traces=None,
    behavior_keys=None,
    plot_behaviors=False,
):
    """
    Plot observed binned firing rate vs fitted rate from the GLM.
    Optionally include a voltage trace subplot with shared x-axis.
    """
    bin_size_s = float(glm_out["bin_size_s"])
    fitted = np.asarray(glm_out["result"].fittedvalues, dtype=float)
    exposure_valid = np.asarray(
        glm_out.get("exposure_s", np.full_like(fitted, glm_out["bin_size_s"], dtype=float)),
        dtype=float,
    )

    bin_edges = glm_out.get("bin_edges")
    valid_mask = glm_out.get("valid_mask")
    frame_rate = glm_out.get("frame_rate")
    if bin_edges is not None and valid_mask is not None:
        y_full = np.asarray(glm_out.get("y_raw", glm_out["y"]), dtype=float)
        y = y_full[valid_mask]
    else:
        y = np.asarray(glm_out["y"], dtype=float)
    obs_rate = y / exposure_valid
    fit_rate = fitted / exposure_valid

    if bin_edges is not None and valid_mask is not None and frame_rate is not None:
        bin_centers = np.array([(s + e) * 0.5 for s, e in bin_edges], dtype=float)
        t = bin_centers[valid_mask] / float(frame_rate)
    else:
        t = np.arange(0, len(obs_rate)) * bin_size_s

    n = len(obs_rate)
    if len(t) != n:
        n = min(len(t), n)
        t = t[:n]
        obs_rate = obs_rate[:n]
        fit_rate = fit_rate[:n]
    if n > max_points:
        step = max(1, n // max_points)
        obs_rate = obs_rate[::step]
        fit_rate = fit_rate[::step]
        t = t[::step]

    behavior_bins = None
    if plot_behaviors:
        behavior_bins = glm_out.get("behavior_bins")

    if voltage_trace is not None or behavior_bins:
        voltage_trace = np.asarray(voltage_trace, dtype=float)
        if frame_rate is None:
            raise ValueError("frame_rate is required when plotting voltage_trace.")
        if ax is not None:
            raise ValueError("ax must be None when plotting multiple subplots.")
        n_beh = 0
        if behavior_bins:
            if behavior_keys is None:
                behavior_keys = list(behavior_bins.keys())
            n_beh = len(behavior_keys)
        n_rows = 1 + (1 if voltage_trace is not None else 0) + n_beh
        fig, axes = plt.subplots(
            n_rows,
            1,
            sharex=True,
            figsize=(6, 1.6 * n_rows),
            gridspec_kw={"height_ratios": [1] * n_rows},
        )
        ax_rate = axes[0]
        ax_rate.plot(t, obs_rate, color="black", linewidth=0.6, label="Observed")
        ax_rate.plot(t, fit_rate, color="#d95f02", linewidth=0.8, alpha=0.9, label="Fitted")
        ax_rate.set_ylabel("Rate (spk/s)")
        ax_rate.legend(frameon=False, fontsize=6)
        ax_rate.spines["top"].set_visible(False)
        ax_rate.spines["right"].set_visible(False)
        row = 1
        if voltage_trace is not None:
            t_full = np.arange(len(voltage_trace)) / float(frame_rate)
            ax_trace = axes[row]
            ax_trace.plot(t_full, voltage_trace, color="#4c78a8", linewidth=0.6)
            if spike_frames is not None:
                spike_frames = np.asarray(spike_frames, dtype=int)
                spike_frames = spike_frames[(spike_frames >= 0) & (spike_frames < len(voltage_trace))]
                spike_times = spike_frames / float(frame_rate)
                spike_vals = voltage_trace[spike_frames]
                ax_trace.scatter(
                    spike_times,
                    spike_vals,
                    s=8,
                    color="#d62728",
                    zorder=3,
                    label="spikes",
                )
            ax_trace.set_ylabel("Trace")
            ax_trace.spines["top"].set_visible(False)
            ax_trace.spines["right"].set_visible(False)
            row += 1
        if behavior_bins:
            for key in behavior_keys:
                if key not in behavior_bins:
                    continue
                beh = behavior_bins[key]
                if valid_mask is not None and len(beh) == len(valid_mask):
                    beh = beh[valid_mask]
                if len(beh) != len(t):
                    n = min(len(beh), len(t))
                    beh = beh[:n]
                    t_plot = t[:n]
                else:
                    t_plot = t
                ax_beh = axes[row]
                ax_beh.plot(t_plot, beh, linewidth=0.6, color="#4c78a8")
                ax_beh.set_ylabel(key)
                ax_beh.spines["top"].set_visible(False)
                ax_beh.spines["right"].set_visible(False)
                row += 1
        axes[-1].set_xlabel("Time (s)")
        return axes

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 2.5))
    ax.plot(t, obs_rate, color="black", linewidth=0.6, label="Observed")
    ax.plot(t, fit_rate, color="#d95f02", linewidth=0.8, alpha=0.9, label="Fitted")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Rate (spk/s)")
    ax.legend(frameon=False, fontsize=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax


def plot_predicted_spatial_map(
    glm_out,
    grid_res=(60, 35),
    ax=None,
    title="Predicted rate map",
    occupancy_map=None,
    mask_unvisited=True,
    min_occupancy_bins=1.0,
):
    """
    Plot predicted firing rate across arena using position basis only.
    Other covariates are held at their mean values.
    """
    rate_map, arena_w, arena_h = predict_spatial_rate_map(
        glm_out,
        grid_res=grid_res,
        occupancy_map=occupancy_map,
        mask_unvisited=mask_unvisited,
        min_occupancy_bins=min_occupancy_bins,
    )

    if ax is None:
        _, ax = plt.subplots(figsize=(3.8, 2.2))
    im = ax.imshow(
        rate_map,
        origin="lower",
        extent=(0, arena_w, 0, arena_h),
        aspect="auto",
        cmap="magma",
    )
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="spk/s")
    return ax, rate_map


def _glm_valid_position_occupancy_map(glm_out, grid_res=(60, 35)):
    """
    Build an occupancy/support map from the GLM's own valid binned positions.

    Returns a map with the same orientation as ``predict_spatial_rate_map``:
    shape (n_y_bins, n_x_bins).
    """
    behavior_bins = glm_out.get("behavior_bins", {}) or {}
    x_bins = np.asarray(behavior_bins.get("x_cm", []), dtype=float)
    y_bins = np.asarray(behavior_bins.get("y_cm", []), dtype=float)
    if x_bins.size == 0 or y_bins.size == 0 or x_bins.shape != y_bins.shape:
        return None

    valid_mask = np.asarray(
        glm_out.get("valid_mask", np.ones_like(x_bins, dtype=bool)),
        dtype=bool,
    )
    if valid_mask.shape != x_bins.shape:
        valid_mask = np.ones_like(x_bins, dtype=bool)

    keep = valid_mask & np.isfinite(x_bins) & np.isfinite(y_bins)
    if not np.any(keep):
        return None

    arena_w, arena_h = glm_out.get("arena_size_cm", (35.5, 20.0))
    occ, _, _ = np.histogram2d(
        x_bins[keep],
        y_bins[keep],
        bins=[grid_res[0], grid_res[1]],
        range=[[0.0, float(arena_w)], [0.0, float(arena_h)]],
    )
    return occ.T


def predict_spatial_rate_map(
    glm_out,
    grid_res=(60, 35),
    occupancy_map=None,
    mask_unvisited=True,
    min_occupancy_bins=1.0,
):
    """
    Predict firing rate map across the arena using position basis only.
    Returns (rate_map, arena_w, arena_h).
    """
    centers = glm_out.get("rbf_centers")
    sigma = glm_out.get("rbf_sigma_cm")
    arena_w, arena_h = glm_out.get("arena_size_cm", (35.5, 20.0))
    feature_names = glm_out["feature_names"]

    if centers is None or sigma is None:
        raise ValueError("GLM output missing RBF center or sigma info.")

    standardization_stats = glm_out.get("standardization_stats", {}) or {}

    xs = np.linspace(0, arena_w, grid_res[0])
    ys = np.linspace(0, arena_h, grid_res[1])
    grid_x, grid_y = np.meshgrid(xs, ys)
    grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    d2 = (
        (grid_points[:, 0][:, None] - centers[:, 0][None, :]) ** 2
        + (grid_points[:, 1][:, None] - centers[:, 1][None, :]) ** 2
    )
    rbf = np.exp(-d2 / (2.0 * sigma ** 2))

    X_mean = np.nanmean(glm_out["X"], axis=0)
    full = np.tile(X_mean, (rbf.shape[0], 1))

    for i, name in enumerate(feature_names[1:]):
        if name.startswith("pos_rbf_"):
            rbf_idx = int(name.split("_")[-1])
            rbf_vals = rbf[:, rbf_idx]
            stat_info = standardization_stats.get(name, {})
            if bool(stat_info.get("applied", False)):
                mu = float(stat_info.get("mean", np.nan))
                sigma_std = float(stat_info.get("std", np.nan))
                if np.isfinite(mu) and np.isfinite(sigma_std) and sigma_std > 0:
                    rbf_vals = (rbf_vals - mu) / sigma_std
            full[:, i] = rbf_vals
        elif name.startswith("session_"):
            full[:, i] = 0.0

    X_grid = sm.add_constant(full, has_constant="add")
    eta = X_grid @ glm_out["result"].params
    rate = np.exp(eta)
    rate_map = rate.reshape(grid_res[1], grid_res[0])
    if occupancy_map is None and mask_unvisited:
        occupancy_map = _glm_valid_position_occupancy_map(glm_out, grid_res=grid_res)
    if occupancy_map is not None:
        occ = np.asarray(occupancy_map)
        if occ.shape != rate_map.shape:
            if occ.T.shape == rate_map.shape:
                occ = occ.T
            else:
                occ = None
        if occ is not None:
            rate_map = rate_map.copy()
            rate_map[~np.isfinite(occ) | (occ < float(min_occupancy_bins))] = np.nan
    return rate_map, arena_w, arena_h


def plot_spatial_rate_map_style(
    rate_map,
    width_real,
    height_real,
    ax,
    title="",
    place_field_mask=None,
    plot_pf_contours=False,
):
    """
    Plot a rate map with the same style as analyze_place_cell_single_moving.
    """
    extent = (0, width_real, 0, height_real)
    masked_map = ma.masked_where(np.isnan(rate_map), rate_map)
    cmap = plt.get_cmap("jet").copy()
    cmap.set_bad(color="white")
    im = ax.imshow(
        masked_map.T,
        origin="lower",
        extent=extent,
        cmap=cmap,
        interpolation="nearest",
    )
    if plot_pf_contours and place_field_mask is not None and np.any(place_field_mask):
        padded_mask = np.zeros(
            (place_field_mask.shape[0] + 2, place_field_mask.shape[1] + 2), dtype=bool
        )
        padded_mask[1:-1, 1:-1] = place_field_mask
        bin_x = (extent[1] - extent[0]) / place_field_mask.shape[0]
        bin_y = (extent[3] - extent[2]) / place_field_mask.shape[1]
        padded_extent = (
            extent[0] - bin_x,
            extent[1] + bin_x,
            extent[2] - bin_y,
            extent[3] + bin_y,
        )
        ax.contour(
            padded_mask.T,
            levels=[0.5],
            colors="magenta",
            linewidths=1.2,
            extent=padded_extent,
            origin="lower",
        )
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=6, fontname="Arial")
    ax.axis("off")
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.add_patch(
        Rectangle(
            (extent[0], extent[2]),
            extent[1] - extent[0],
            extent[3] - extent[2],
            linewidth=1.0,
            edgecolor="black",
            facecolor="none",
            zorder=3,
        )
    )

    cax = inset_axes(
        ax,
        width="8%",
        height="100%",
        loc="lower left",
        bbox_to_anchor=(1.02, 0.0, 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )
    cbar = ax.figure.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=5)
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontname("Arial")

    scale_len_cm = 10
    x0 = 0.95 - (scale_len_cm / width_real) * 0.9
    x1 = 0.95
    y0 = -0.08
    ax.plot(
        [x0, x1],
        [y0, y0],
        transform=ax.transAxes,
        color="black",
        linewidth=1.0,
        clip_on=False,
    )
    ax.text(
        (x0 + x1) / 2,
        y0 - 0.04,
        "10 cm",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=6,
        color="black",
    )
    return ax


def plot_observed_and_predicted_spatial_maps(
    pc_out,
    glm_out,
    bin_size=None,
    axes=None,
    plot_pf_contours=False,
    grid_res=(60, 35),
    mask_unvisited=True,
    min_occupancy_bins=1.0,
):
    """
    Plot observed and GLM-predicted spatial maps side by side with matching style.
    """
    width_real = pc_out["params"]["width_real"]
    height_real = pc_out["params"]["height_real"]
    pred_map, _, _ = predict_spatial_rate_map(
        glm_out,
        grid_res=grid_res,
        mask_unvisited=mask_unvisited,
        min_occupancy_bins=min_occupancy_bins,
    )

    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 2.4), sharex=True, sharey=True)
    ax_obs, ax_pred = axes

    plot_spatial_rate_map_style(
        pc_out["rate_map"],
        width_real,
        height_real,
        ax_obs,
        title="Observed",
        place_field_mask=pc_out.get("place_field_mask"),
        plot_pf_contours=plot_pf_contours,
    )
    masked_pred = ma.masked_where(np.isnan(pred_map), pred_map)
    cmap = plt.get_cmap("jet").copy()
    cmap.set_bad(color="white")
    im_pred = ax_pred.imshow(
        masked_pred,
        origin="lower",
        extent=(0, width_real, 0, height_real),
        cmap=cmap,
        interpolation="nearest",
    )
    ax_pred.set_aspect("equal", adjustable="box")
    ax_pred.set_title("Predicted", fontsize=6, fontname="Arial")
    ax_pred.axis("off")
    ax_pred.set_xlim(0, width_real)
    ax_pred.set_ylim(0, height_real)
    ax_pred.add_patch(
        Rectangle(
            (0, 0),
            width_real,
            height_real,
            linewidth=1.0,
            edgecolor="black",
            facecolor="none",
            zorder=3,
        )
    )
    cax = inset_axes(
        ax_pred,
        width="8%",
        height="100%",
        loc="lower left",
        bbox_to_anchor=(1.02, 0.0, 1, 1),
        bbox_transform=ax_pred.transAxes,
        borderpad=0,
    )
    cbar = ax_pred.figure.colorbar(im_pred, cax=cax)
    cbar.ax.tick_params(labelsize=5)
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontname("Arial")
    return axes, pred_map


def drop_one_glm_statsmodels(glm_out, groups=None):
    """
    Drop-one covariate analysis for a fitted GLM.

    Returns a DataFrame with deviance increases relative to the full model.
    """
    X = glm_out["X"]
    y = glm_out["y"]
    exposure = np.asarray(
        glm_out.get("exposure_s", np.full_like(y, glm_out.get("bin_size_s", 1.0), dtype=float)),
        dtype=float,
    )
    feature_names = glm_out["feature_names"][1:]

    if groups is None:
        groups = {
            "position": [n for n in feature_names if n.startswith("pos_rbf_")],
            "head_direction": [n for n in feature_names if n in {"hd_sin", "hd_cos"}],
            "travel_direction": [n for n in feature_names if n in {"travel_sin", "travel_cos"}],
            "speed": [n for n in feature_names if n in {"speed", "speed_moving"}],
            "state": [n for n in feature_names if n == "quiet"],
            "accel": ["accel"],
            "session": [n for n in feature_names if n.startswith("session_")],
            "pos_x_hd": [n for n in feature_names if n.startswith("pos_x_hd_")],
            "pos_x_speed": [n for n in feature_names if n.startswith("pos_x_speed_")],
            "pos_x_state": [n for n in feature_names if n.startswith("pos_x_state_")],
            "pos_x_travel_dir": [
                n for n in feature_names if n.startswith("pos_x_travel_")
            ],
        }

    result = glm_out["result"]
    if "deviance" in glm_out:
        full_deviance = glm_out["deviance"]
    elif hasattr(result, "deviance"):
        full_deviance = result.deviance
    else:
        mu = result.fittedvalues
        full_deviance = float(result.model.family.deviance(y, mu))
    if "null_deviance" in glm_out:
        null_deviance = glm_out["null_deviance"]
    elif hasattr(result, "null_deviance"):
        null_deviance = result.null_deviance
    else:
        mu_null = np.full_like(y, np.mean(y))
        null_deviance = float(result.model.family.deviance(y, mu_null))
    pseudo_r2_full = 1.0 - (full_deviance / null_deviance) if null_deviance else np.nan

    results = []
    for group_name, group_feats in groups.items():
        if not group_feats:
            continue
        drop_idx = [i for i, n in enumerate(feature_names) if n in set(group_feats)]
        if len(drop_idx) == 0:
            continue
        keep_idx = [i for i in range(len(feature_names)) if i not in set(drop_idx)]
        if len(keep_idx) == 0:
            continue
        X_drop = X[:, keep_idx]
        finite_mask = np.isfinite(y) & np.all(np.isfinite(X_drop), axis=1)
        finite_mask &= np.isfinite(exposure) & (exposure > 0)
        if not np.any(finite_mask):
            continue
        y_fit = y[finite_mask]
        X_fit = X_drop[finite_mask]
        exposure_fit = exposure[finite_mask]
        X_fit = sm.add_constant(X_fit, has_constant="add")
        model = sm.GLM(y_fit, X_fit, family=sm.families.Poisson(), exposure=exposure_fit)
        try:
            if glm_out.get("use_regularization"):
                res = model.fit_regularized(
                    alpha=glm_out.get("reg_alpha", 1.0),
                    L1_wt=glm_out.get("reg_l1_wt", 0.0),
                )
                mu = res.fittedvalues
                deviance = float(model.family.deviance(y_fit, mu))
            else:
                res = model.fit()
                deviance = res.deviance
        except Exception:
            deviance = np.nan
        pseudo_r2_drop = 1.0 - (deviance / null_deviance) if null_deviance else np.nan
        results.append(
            {
                "covariate": group_name,
                "deviance": deviance,
                "delta_deviance": deviance - full_deviance,
                "pseudo_r2": pseudo_r2_drop,
                "delta_pseudo_r2": pseudo_r2_full - pseudo_r2_drop,
                "n_params": X_drop.shape[1],
            }
        )

    df = pd.DataFrame(results).sort_values("delta_deviance", ascending=False)
    return df


def shuffle_glm_drop_one_statsmodels(
    spike_frames,
    x_cm,
    y_cm,
    speed,
    hd_deg,
    frame_rate,
    session_start_frames=None,
    accel=None,
    trace_for_nan=None,
    bad_mask=None,
    n_shuffles=200,
    min_shift_s=5.0,
    groups=None,
    random_state=None,
    **glm_kwargs,
):
    """
    Shuffle spike times (circular shift within sessions) to assess drop-one GLM significance.

    Returns a dict with the fitted GLM, observed drop-one table, and shuffle distributions.
    """
    glm_kwargs = dict(glm_kwargs)
    glm_kwargs.pop("session_start_frames", None)
    glm_kwargs.pop("accel", None)
    glm_kwargs.pop("trace_for_nan", None)
    glm_kwargs.pop("bad_mask", None)

    n_frames = len(x_cm)
    if session_start_frames is None or len(session_start_frames) == 0:
        session_start_frames = [0]
    session_start_frames = sorted(set(int(s) for s in session_start_frames))
    session_start_frames = [s for s in session_start_frames if s < n_frames]
    session_ends = session_start_frames[1:] + [n_frames]
    session_bounds = list(zip(session_start_frames, session_ends))

    rng = np.random.default_rng(random_state)
    min_shift_frames = max(1, int(round(min_shift_s * frame_rate)))

    def _shuffle_spike_frames(frames):
        frames = np.asarray(frames, dtype=int)
        frames = frames[(frames >= 0) & (frames < n_frames)]
        shuffled = []
        for start, end in session_bounds:
            sess_len = end - start
            if sess_len <= 1:
                continue
            sess_frames = frames[(frames >= start) & (frames < end)] - start
            if sess_frames.size == 0:
                continue
            if sess_len > 2 * min_shift_frames:
                shift = rng.integers(
                    min_shift_frames, sess_len - min_shift_frames + 1
                )
            else:
                shift = rng.integers(1, sess_len)
            shuffled.append(((sess_frames + shift) % sess_len) + start)
        if not shuffled:
            return frames.copy()
        return np.sort(np.concatenate(shuffled))

    glm_out = fit_behavior_glm_statsmodels(
        spike_frames,
        x_cm,
        y_cm,
        speed,
        hd_deg,
        frame_rate=frame_rate,
        session_start_frames=session_start_frames,
        accel=accel,
        trace_for_nan=trace_for_nan,
        bad_mask=bad_mask,
        **glm_kwargs,
    )
    drop_one_df = drop_one_glm_statsmodels(glm_out, groups=groups).reset_index(drop=True)
    covariates = drop_one_df["covariate"].tolist()
    shuffle_dist = {cov: [] for cov in covariates}

    for _ in range(int(n_shuffles)):
        shuf_spikes = _shuffle_spike_frames(spike_frames)
        glm_shuf = fit_behavior_glm_statsmodels(
            shuf_spikes,
            x_cm,
            y_cm,
            speed,
            hd_deg,
            frame_rate=frame_rate,
            session_start_frames=session_start_frames,
            accel=accel,
            trace_for_nan=trace_for_nan,
            bad_mask=bad_mask,
            **glm_kwargs,
        )
        shuf_df = drop_one_glm_statsmodels(glm_shuf, groups=groups)
        shuf_map = dict(zip(shuf_df["covariate"], shuf_df["delta_pseudo_r2"]))
        for cov in covariates:
            shuffle_dist[cov].append(shuf_map.get(cov, np.nan))

    p_vals = []
    shuffle_means = []
    shuffle_stds = []
    for cov in covariates:
        null_vals = np.asarray(shuffle_dist[cov], dtype=float)
        null_vals = null_vals[np.isfinite(null_vals)]
        if null_vals.size == 0:
            p_vals.append(np.nan)
            shuffle_means.append(np.nan)
            shuffle_stds.append(np.nan)
            continue
        obs = float(
            drop_one_df.loc[drop_one_df["covariate"] == cov, "delta_pseudo_r2"].iloc[0]
        )
        p_val = (np.sum(null_vals >= obs) + 1.0) / (len(null_vals) + 1.0)
        p_vals.append(p_val)
        shuffle_means.append(float(np.mean(null_vals)))
        shuffle_stds.append(float(np.std(null_vals)))

    drop_one_df = drop_one_df.copy()
    drop_one_df["shuffle_mean"] = shuffle_means
    drop_one_df["shuffle_std"] = shuffle_stds
    drop_one_df["shuffle_p"] = p_vals

    return {
        "glm_out": glm_out,
        "drop_one_df": drop_one_df,
        "shuffle_dist": shuffle_dist,
        "n_shuffles": int(n_shuffles),
        "min_shift_s": float(min_shift_s),
    }


def plot_drop_one_deviance(drop_one_df, ax=None, title="Drop-one covariate"):
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 2.6))
    ax.bar(
        drop_one_df["covariate"],
        drop_one_df["delta_deviance"],
        color="#4c78a8",
    )
    ax.set_ylabel("Deviance increase")
    ax.set_title(title)
    ax.axhline(0, color="black", linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelrotation=30)
    return ax


def plot_drop_one_metrics(drop_one_df, axes=None, title="Drop-one covariate"):
    """
    Plot delta deviance and delta pseudo-R2 stacked vertically.
    """
    if axes is None:
        fig, axes = plt.subplots(2, 1, figsize=(4, 4.8), sharex=True)
    ax_dev, ax_r2 = axes

    ax_dev.bar(
        drop_one_df["covariate"],
        drop_one_df["delta_deviance"],
        color="#4c78a8",
    )
    ax_dev.set_ylabel("Deviance increase")
    ax_dev.set_title(title)
    ax_dev.axhline(0, color="black", linewidth=0.7)
    ax_dev.spines["top"].set_visible(False)
    ax_dev.spines["right"].set_visible(False)

    ax_r2.bar(
        drop_one_df["covariate"],
        drop_one_df["delta_pseudo_r2"],
        color="#f58518",
    )
    ax_r2.set_ylabel("Delta pseudo-R2")
    ax_r2.axhline(0, color="black", linewidth=0.7)
    ax_r2.spines["top"].set_visible(False)
    ax_r2.spines["right"].set_visible(False)
    ax_r2.tick_params(axis="x", labelrotation=30)
    return axes


def glm_coef_table(glm_out):
    """
    Return a DataFrame of coefficients for regularized or unregularized GLM.
    """
    result = glm_out["result"]
    names = glm_out.get("feature_names")
    if names is None:
        names = [f"x{i}" for i in range(len(result.params))]
    return pd.DataFrame({"term": names, "coef": result.params})


def load_behavior_frame(subfolders, frame_idx=100):
    behav_video_dir = os.path.join(subfolders[0], 'output_with_head_direction2')
    mp4_path = os.path.join(subfolders[0], 'output_with_head_direction2.mp4')

    cap = cv2.VideoCapture(mp4_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame_bgr = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Failed to read frame {frame_idx} from behavior video")

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return frame_rgb


def plot_annotated_behavior_frame(video_file, frame_idx=100, ax=None, return_frame=True):
    folder_path = os.path.dirname(video_file)
    csv_matches = glob.glob(os.path.join(folder_path, "*behavior_refined.csv"))
    if not csv_matches:
        csv_matches = glob.glob(os.path.join(folder_path, "*filtered_head_HNlogic_clean.csv"))
    if not csv_matches:
        raise FileNotFoundError(f"No *behavior_refined.csv found in {folder_path}")
    df = pd.read_csv(csv_matches[0])

    head_angle = pd.to_numeric(df["head_direction_rad"], errors="coerce").to_numpy()
    best_x = pd.to_numeric(df["head_best_x"], errors="coerce").to_numpy()
    best_y = pd.to_numeric(df["head_best_y"], errors="coerce").to_numpy()

    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_file}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Failed to read frame {frame_idx} from {video_file}")
    if frame_idx >= len(best_x):
        raise IndexError("Frame index exceeds tracking data length")

    arrow_length = 100
    dot_radius = 25
    dot_color = (0, 255, 0)
    arrow_color = (0, 0, 255)
    thickness = 10

    x = best_x[frame_idx]
    y = best_y[frame_idx]
    ang = head_angle[frame_idx]
    if np.isfinite(x) and np.isfinite(y):
        cx, cy = int(x), int(y)
        cv2.circle(frame, (cx, cy), dot_radius, dot_color, -1)
        if np.isfinite(ang):
            dx = arrow_length * math.cos(ang)
            dy = arrow_length * math.sin(ang)
            end_x = int(cx + dx)
            end_y = int(cy + dy)
            cv2.arrowedLine(
                frame,
                (cx, cy),
                (end_x, end_y),
                arrow_color,
                thickness,
                tipLength=0.3,
            )

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(frame_rgb)
    ax.axis("off")
    legend_dot_size = 6
    legend_arrow_size = 6
    legend_line_width = 1.2
    dot_handle = Line2D(
        [0],
        [0],
        marker="o",
        color=(0, 1, 0),
        linestyle="None",
        markersize=legend_dot_size,
        label="head position",
    )
    arrow_handle = Line2D(
        [0, 1],
        [0, 0],
        marker=">",
        color=(1, 0, 0),
        linestyle="-",
        linewidth=legend_line_width,
        markersize=legend_arrow_size,
        markevery=[1],
        label="head direction",
    )
    # ax.legend(
    #     handles=[dot_handle, arrow_handle],
    #     loc="upper right",
    #     fontsize=7,
    #     prop={"family": "Arial", "size": 7},
    #     labelcolor="white",
    #     handlelength=1.5,
    #     handletextpad=0.6,
    #     frameon=False,
    # )
    if return_frame:
        return frame_rgb
    return None


def plot_trajectory_with_spikes(
    x_neural,
    y_neural,
    spikes,
    speed,
    frame_rate,
    ax=None,
    kernel_size=21,
    filter_type="median",
    speed_threshold=2,
    min_duration_s=0.5,
    merge_gap_s=1.0,
    speed_mode="above",  # "above" for moving, "below" for resting
    max_speed=50,  # Exclude frames with speed above this value (abnormal)
):
    """
    Plot trajectory with spike locations overlay.
    
    Parameters
    ----------
    speed_mode : str, optional
        "above" to show spikes when speed >= threshold (moving periods)
        "below" to show spikes when speed < threshold (resting periods)
        Default is "above".
    max_speed : float, optional
        Maximum speed threshold. Frames with speed above this value are
        considered abnormal and excluded from trajectory and spike plotting.
        Default is 50 cm/s.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))

    x_neural = np.asarray(x_neural)
    y_neural = np.asarray(y_neural)
    speed = np.asarray(speed, dtype=float)
    
    # Exclude abnormal speed frames
    abnormal_speed = speed > max_speed
    valid_frames = (~np.isnan(x_neural)) & (~np.isnan(y_neural)) & (~np.isnan(speed)) & (~abnormal_speed)
    
    speed_for_epochs = speed.copy()
    speed_for_epochs[~valid_frames] = np.nan
    speed_smooth, moving_epochs, moving_idx = _compute_moving_epochs(
        speed_for_epochs,
        frame_rate,
        kernel_size=kernel_size,
        filter_type=filter_type,
        speed_threshold=speed_threshold,
        min_duration_s=min_duration_s,
        merge_gap_s=merge_gap_s,
    )
    moving_idx = moving_idx[valid_frames[moving_idx]]
    total_frames = len(x_neural)
    moving_mask = np.zeros(total_frames, dtype=bool)
    moving_mask[moving_idx] = True
    quiet_idx = np.where(valid_frames & (~moving_mask))[0]
    
    # Compute epoch indices based on speed_mode
    if speed_mode == "below":
        # Get all valid frame indices that are NOT in moving_idx (i.e., resting)
        all_valid_idx = np.where(valid_frames)[0]
        epoch_idx = np.setdiff1d(all_valid_idx, moving_idx, assume_unique=False)
    else:
        # Default: use moving indices
        epoch_idx = moving_idx

    ax.plot(
        x_neural[valid_frames],
        y_neural[valid_frames],
        color="gray",
        linewidth=0.6,
        alpha=0.6,
    )

    num_cells = len(spikes)
    if num_cells <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, num_cells))
    elif num_cells <= 20:
        colors = plt.cm.tab20(np.linspace(0, 1, num_cells))
    else:
        colors = plt.cm.hsv(np.linspace(0, 1, num_cells, endpoint=False))

    for i, spike_idx in enumerate(spikes):
        if len(spike_idx) == 0:
            continue
        spike_idx = np.asarray(spike_idx, dtype=int)
        spike_idx = spike_idx[(spike_idx >= 0) & (spike_idx < len(x_neural))]
        spike_idx = spike_idx[valid_frames[spike_idx]]
        if spike_idx.size == 0:
            continue
        spike_idx = np.intersect1d(spike_idx, epoch_idx, assume_unique=False)
        if spike_idx.size == 0:
            continue
        ax.scatter(
            x_neural[spike_idx],
            y_neural[spike_idx],
            s=6,
            color=colors[i],
            alpha=0.9,
            linewidths=0,
        )

    legend_handles = []
    for i in range(num_cells):
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color=colors[i],
                linestyle="None",
                markersize=4,
                label=f"Cell {i + 1}",
            )
        )
    ax.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(0.95, 0.5),
        fontsize=5,
        frameon=False,
        ncol=1,
    )

    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    
    # Add 10 cm scale bar at lower left, outside the trajectory
    x_min, x_max = np.nanmin(x_neural[valid_frames]), np.nanmax(x_neural[valid_frames])
    y_min, y_max = np.nanmin(y_neural[valid_frames]), np.nanmax(y_neural[valid_frames])
    scale_bar_length = 10  # 10 cm
    # Position scale bar below and to the left of the trajectory
    scale_x_start = x_min
    scale_x_end = x_min + scale_bar_length
    scale_y = y_min - (y_max - y_min) * 0.05  # 5% below the trajectory
    ax.plot([scale_x_start, scale_x_end], [scale_y, scale_y],
            color='black', linewidth=2, solid_capstyle='butt', clip_on=False)
    ax.text((scale_x_start + scale_x_end) / 2, scale_y - (y_max - y_min) * 0.02, '10 cm',
            fontsize=6, ha='center', va='top', fontname='Arial')
    
    return ax


def plot_speed_with_moving_epochs(
    speed,
    frame_rate,
    kernel_size=21,
    filter_type="boxcar",
    ax=None,
    speed_threshold=2,
    min_duration_s=0.5,
    merge_gap_s=1.0,
):
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 2))

    speed = np.asarray(speed, dtype=float)
    speed_clean = speed.copy()
    speed_clean[speed_clean > 50] = np.nan
    valid_mask = np.isfinite(speed_clean)
    # if np.any(valid_mask):
    #     idx = np.arange(speed_clean.size)
    #     speed_clean = np.interp(idx, idx[valid_mask], speed_clean[valid_mask])

    speed_smooth, moving_epochs, _ = _compute_moving_epochs(
        speed_clean,
        frame_rate,
        kernel_size=kernel_size,
        filter_type=filter_type,
        speed_threshold=speed_threshold,
        min_duration_s=min_duration_s,
        merge_gap_s=merge_gap_s,
    )

    time_axis = np.arange(len(speed_smooth)) / frame_rate
    ax.plot(time_axis, speed_clean, color="gray", linewidth=0.5)
    ax.plot(time_axis, speed_smooth, color="black", linewidth=1)
    ax.axhline(
        speed_threshold,
        color="black",
        linestyle="--",
        linewidth=0.8,
    )
    for start, end in moving_epochs:
        ax.axvspan(
            start / frame_rate,
            end / frame_rate,
            color="yellow",
            alpha=0.3,
        )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Speed (cm/s)")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    return ax, speed_smooth, moving_epochs


def _compute_quiet_epochs(
    speed,
    frame_rate,
    kernel_size=21,
    filter_type="median",
    speed_threshold_quiet=1,
    min_duration_s=0.5,
    merge_gap_s=1.0,
):
    """Compute quiet epochs based on smoothed speed <= speed_threshold_quiet."""
    speed = np.asarray(speed, dtype=float)
    if filter_type == "median":
        if kernel_size % 2 == 0:
            kernel_size += 1
        speed_smooth = medfilt(speed, kernel_size=kernel_size)
    elif filter_type == "boxcar":
        kernel_size = int(kernel_size)
        if kernel_size < 1:
            raise ValueError("kernel_size must be >= 1 for boxcar filter")
        kernel = np.ones(kernel_size, dtype=float) / kernel_size
        speed_smooth = np.convolve(speed, kernel, mode="same")
    else:
        raise ValueError("filter_type must be 'median' or 'boxcar'")

    quiet_mask = speed_smooth <= speed_threshold_quiet
    min_frames = int(round(min_duration_s * frame_rate))
    merge_gap = int(round(merge_gap_s * frame_rate))

    starts = []
    ends = []
    in_run = False
    for i, is_quiet in enumerate(quiet_mask):
        if is_quiet and not in_run:
            starts.append(i)
            in_run = True
        elif not is_quiet and in_run:
            ends.append(i - 1)
            in_run = False
    if in_run:
        ends.append(len(quiet_mask) - 1)

    epochs = []
    for start, end in zip(starts, ends):
        if (end - start + 1) >= min_frames:
            epochs.append((start, end))

    merged_epochs = []
    for start, end in epochs:
        if not merged_epochs:
            merged_epochs.append([start, end])
            continue
        prev_start, prev_end = merged_epochs[-1]
        if (start - prev_end - 1) < merge_gap:
            merged_epochs[-1][1] = end
        else:
            merged_epochs.append([start, end])

    quiet_idx = []
    for start, end in merged_epochs:
        quiet_idx.append(np.arange(start, end + 1, dtype=int))
    if quiet_idx:
        quiet_idx = np.concatenate(quiet_idx)
    else:
        quiet_idx = np.array([], dtype=int)

    return speed_smooth, merged_epochs, quiet_idx


def plot_speed_with_quiet_epochs(
    speed,
    frame_rate,
    kernel_size=21,
    filter_type="boxcar",
    ax=None,
    speed_threshold_quiet=1,
    min_duration_s=0.5,
    merge_gap_s=1.0,
    plot_lines=True,
    plot_threshold_line=True,
    shade_color="lightgray",
    shade_alpha=0.35,
):
    """Plot speed and highlight quiet epochs (speed <= speed_threshold_quiet) in gray."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 2))

    speed = np.asarray(speed, dtype=float)
    speed_clean = speed.copy()
    speed_clean[speed_clean > 50] = np.nan

    speed_smooth, quiet_epochs, _ = _compute_quiet_epochs(
        speed_clean,
        frame_rate,
        kernel_size=kernel_size,
        filter_type=filter_type,
        speed_threshold_quiet=speed_threshold_quiet,
        min_duration_s=min_duration_s,
        merge_gap_s=merge_gap_s,
    )

    time_axis = np.arange(len(speed_smooth)) / frame_rate
    if plot_lines:
        ax.plot(time_axis, speed_clean, color="gray", linewidth=0.5)
        ax.plot(time_axis, speed_smooth, color="black", linewidth=1)

    if plot_threshold_line:
        ax.axhline(
            speed_threshold_quiet,
            color="gray",
            linestyle="--",
            linewidth=0.8,
        )

    for start, end in quiet_epochs:
        ax.axvspan(
            start / frame_rate,
            end / frame_rate,
            color=shade_color,
            alpha=shade_alpha,
        )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Speed (cm/s)")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    return ax, speed_smooth, quiet_epochs


def analyze_place_cell_single_moving(
    x_neural,
    y_neural,
    spikes,
    speed,
    frame_rate,
    cell_idx,
    axes=None,
    width_real=35.5,
    height_real=20,
    bin_size=1.5,
    smooth_sigma=2.0,
    num_shuffles=1000,
    place_field_threshold=0.4,
    min_component_peak_ratio=0.4,
    split_multi_peak_fields=True,
    split_secondary_peak_ratio=0.8,
    split_secondary_peak_min_separation_cm=15.0,
    min_field_bins=5,
    kernel_size=21,
    filter_type="boxcar",
    speed_threshold=2,
    min_duration_s=0.5,
    merge_gap_s=1.0,
    plot_pf_contours=False,
    max_field_area_ratio=2.0 / 3.0,
    min_occupancy_s=1.0,
    occ_smooth_sigma=2.0,
    use_smoothed_occ_mask=False,
    trim_sparse_top_row_for_analysis=True,
    trim_sparse_top_row_for_plotting=True,
    sparse_top_row_nonocc_frac_threshold=0.8,
    bad_timepoints=None,
    random_seed=0,
    plot_sub=False,
    traces=None,
    theta_freqs=(4, 8),
    slow_freqs=2,
    separate_spikes=False,
    refined_SS=None,
    all_CS_spikes=None,
    complex_bursts_dicts=None,
    simple_spike_color="#026C80",
    complex_spike_color="#EE9B00",
    display_cell_num=None,
    is_last_column=False,
    is_first_column=False,
    display_PF_SS_CS=False,
    min_peak_rate=0.5,
    min_pf_firing_traversals=5,
    pf_firing_traversal_distance_window_cm=15.0,
    pf_firing_traversal_detection_window_cm=8.0,
    pf_firing_traversal_distance_bin_cm=1.5,
    pf_firing_traversal_distance_mode="euclidean_to_peak",
    pf_firing_traversal_center_vicinity_min_cm=1,
    pf_firing_traversal_center_vicinity_max_cm=5,
    pf_firing_traversal_resting_speed_threshold=0.5,
    pf_firing_traversal_merge_gap_s=2.0,
    pf_firing_traversal_exclude_trials_with_bad_frames=True,
    pf_reliability_dilation_bins=3,
    pf_reliability_dilation_shape="disk",
    save_spike_shapes=False,
    ss_shape_pre_ms=20.0,
    ss_shape_post_ms=20.0,
    cb_shape_pre_ms=20.0,
    cb_shape_post_ms=150.0,
    ss_shape_min_separation_ms=20.0,
    spike_shape_median_filter_size=11,
    spike_shape_baseline_frames=3,
    save_burst_metrics=False,
    burst_metrics=None,
    include_simple_bursts_metrics=False,
    save_spike_burst_rate_metrics=False,
    min_cb_bursts_per_condition=5,
):
    x_neural = np.asarray(x_neural)
    y_neural = np.asarray(y_neural)
    speed = np.asarray(speed, dtype=float)
    pf_reliability_dilation_bins = int(pf_reliability_dilation_bins)
    if pf_reliability_dilation_bins < 0:
        raise ValueError("pf_reliability_dilation_bins must be >= 0.")
    pf_reliability_dilation_shape = str(pf_reliability_dilation_shape).strip().lower()
    if pf_reliability_dilation_shape not in {"disk", "square", "manhattan"}:
        raise ValueError(
            "Unknown pf_reliability_dilation_shape="
            f"{pf_reliability_dilation_shape!r}. Use 'disk', 'square', or 'manhattan'."
        )
    valid_frames = (~np.isnan(x_neural)) & (~np.isnan(y_neural)) & (~np.isnan(speed))
    if bad_timepoints is not None:
        bad_mask = np.asarray(bad_timepoints)
        if bad_mask.dtype != bool:
            if bad_mask.ndim == 1 and bad_mask.size == len(x_neural):
                bad_mask = bad_mask.astype(bool)
            else:
                bad_mask = np.asarray(bad_mask, dtype=int)
                bad_mask = bad_mask[(bad_mask >= 0) & (bad_mask < len(x_neural))]
                bad_bool = np.zeros(len(x_neural), dtype=bool)
                bad_bool[bad_mask] = True
                bad_mask = bad_bool
        if bad_mask.shape[0] != len(x_neural):
            raise ValueError("bad_timepoints must match x_neural length or be index list.")
        valid_frames &= ~bad_mask
    speed_for_epochs = speed.copy()
    speed_for_epochs[~valid_frames] = np.nan

    speed_smooth, moving_epochs, moving_idx = _compute_moving_epochs(
        speed_for_epochs,
        frame_rate,
        kernel_size=kernel_size,
        filter_type=filter_type,
        speed_threshold=speed_threshold,
        min_duration_s=min_duration_s,
        merge_gap_s=merge_gap_s,
    )

    moving_idx = moving_idx[valid_frames[moving_idx]]
    total_frames = len(x_neural)
    moving_mask = np.zeros(total_frames, dtype=bool)
    moving_mask[moving_idx] = True
    quiet_idx = np.where(valid_frames & (~moving_mask))[0]

    bins = [
        np.arange(0, width_real + bin_size, bin_size),
        np.arange(0, height_real + bin_size, bin_size),
    ]
    extent = (0, width_real, 0, height_real)

    x_sub = x_neural[moving_idx]
    y_sub = y_neural[moving_idx]
    occ_counts, _, _ = np.histogram2d(x_sub, y_sub, bins=bins)
    occ_map_raw = occ_counts / frame_rate  # Non-smoothed occupancy in seconds
    occ_map = gaussian_filter(occ_map_raw, sigma=occ_smooth_sigma, mode="constant")

    # Mask for bins with low occupancy (< min_occupancy_s seconds)
    # Use smoothed or raw occupancy based on parameter
    if use_smoothed_occ_mask:
        low_occ_mask = occ_map < min_occupancy_s
    else:
        low_occ_mask = occ_map_raw < min_occupancy_s

    # Optional top-edge row trim based on NON-smoothed occupancy.
    # Decision metric: fraction of bins in top y-row with occ_map_raw < min_occupancy_s.
    top_row_nonocc_fraction_raw = np.nan
    top_row_trimmed_analysis = False
    top_row_trimmed_plotting = False
    trim_mask_analysis = np.zeros_like(low_occ_mask, dtype=bool)
    trim_mask_plotting = np.zeros_like(low_occ_mask, dtype=bool)
    if low_occ_mask.ndim == 2 and low_occ_mask.shape[1] > 0:
        raw_nonocc_mask = occ_map_raw < float(min_occupancy_s)
        top_row_nonocc_fraction_raw = float(np.mean(raw_nonocc_mask[:, -1]))
        trim_top = bool(
            np.isfinite(top_row_nonocc_fraction_raw)
            and (top_row_nonocc_fraction_raw >= float(sparse_top_row_nonocc_frac_threshold))
        )
        if trim_top:
            if bool(trim_sparse_top_row_for_analysis):
                trim_mask_analysis[:, -1] = True
                top_row_trimmed_analysis = True
            if bool(trim_sparse_top_row_for_plotting):
                trim_mask_plotting[:, -1] = True
                top_row_trimmed_plotting = True

    # Analysis occupancy mask candidate.
    low_occ_mask_analysis = np.asarray(low_occ_mask, dtype=bool).copy()
    if np.any(trim_mask_analysis):
        low_occ_mask_analysis |= np.asarray(trim_mask_analysis, dtype=bool)

    # Only enforce occupancy exclusion on the most outer spatial bins.
    # Internal low-occupancy holes are allowed inside PFs.
    low_occ_border_mask_analysis = np.zeros_like(low_occ_mask_analysis, dtype=bool)
    if low_occ_border_mask_analysis.ndim == 2 and low_occ_border_mask_analysis.size > 0:
        low_occ_border_mask_analysis[0, :] = True
        low_occ_border_mask_analysis[-1, :] = True
        low_occ_border_mask_analysis[:, 0] = True
        low_occ_border_mask_analysis[:, -1] = True
    low_occ_border_mask_analysis &= low_occ_mask_analysis

    cell_spikes = np.asarray(spikes[cell_idx], dtype=int)
    cell_spikes = cell_spikes[(cell_spikes >= 0) & (cell_spikes < total_frames)]
    cell_spikes = cell_spikes[valid_frames[cell_spikes]]
    binary_train_full = np.zeros(total_frames, dtype=bool)
    binary_train_full[cell_spikes] = True
    binary_train_sub = binary_train_full[moving_idx]
    spikes_in_sub_idx = np.where(binary_train_sub)[0]
    x_spk = x_sub[spikes_in_sub_idx]
    y_spk = y_sub[spikes_in_sub_idx]

    spike_map, _, _ = np.histogram2d(x_spk, y_spk, bins=bins)
    with np.errstate(divide="ignore", invalid="ignore"):
        raw_map = spike_map / occ_map
        raw_map[np.isnan(raw_map)] = 0
        raw_map[np.isinf(raw_map)] = 0

    # Smoothed map for place field detection:
    # retain legacy zero-occupancy masking, and optionally trim top row for analysis.
    smooth_map_for_pf_base = gaussian_filter(raw_map, sigma=smooth_sigma, mode="constant")
    smooth_map_for_pf_base[occ_map == 0] = np.nan
    smooth_map_for_pf = smooth_map_for_pf_base.copy()
    if np.any(trim_mask_analysis):
        smooth_map_for_pf[trim_mask_analysis] = np.nan

    # Smoothed map for display:
    # occupancy-based mask plus optional top-row plotting trim.
    low_occ_mask_plot = np.asarray(low_occ_mask, dtype=bool).copy()
    if np.any(trim_mask_plotting):
        low_occ_mask_plot |= np.asarray(trim_mask_plotting, dtype=bool)
    smooth_map = smooth_map_for_pf_base.copy()
    smooth_map[low_occ_mask_plot] = np.nan

    def _spike_positions(spike_times):
        if spike_times is None:
            return np.array([]), np.array([])
        spike_times = np.asarray(spike_times, dtype=int)
        spike_times = spike_times[(spike_times >= 0) & (spike_times < total_frames)]
        spike_times = spike_times[valid_frames[spike_times]]
        if spike_times.size == 0:
            return np.array([]), np.array([])
        binary_full = np.zeros(total_frames, dtype=bool)
        binary_full[spike_times] = True
        binary_sub = binary_full[moving_idx]
        spikes_in_sub_idx = np.where(binary_sub)[0]
        if spikes_in_sub_idx.size == 0:
            return np.array([]), np.array([])
        return x_sub[spikes_in_sub_idx], y_sub[spikes_in_sub_idx]

    def _rate_map_for_spikes(spike_times):
        if spike_times is None:
            return None
        spike_times = np.asarray(spike_times, dtype=int)
        spike_times = spike_times[(spike_times >= 0) & (spike_times < total_frames)]
        spike_times = spike_times[valid_frames[spike_times]]
        if spike_times.size == 0:
            empty_map = np.zeros_like(smooth_map)
            empty_map[low_occ_mask_plot] = np.nan
            return empty_map
        binary_full = np.zeros(total_frames, dtype=bool)
        binary_full[spike_times] = True
        binary_sub = binary_full[moving_idx]
        spikes_in_sub_idx = np.where(binary_sub)[0]
        if spikes_in_sub_idx.size == 0:
            empty_map = np.zeros_like(smooth_map)
            empty_map[low_occ_mask_plot] = np.nan
            return empty_map
        x_spk_local = x_sub[spikes_in_sub_idx]
        y_spk_local = y_sub[spikes_in_sub_idx]
        spike_map_local, _, _ = np.histogram2d(x_spk_local, y_spk_local, bins=bins)
        with np.errstate(divide="ignore", invalid="ignore"):
            raw_local = spike_map_local / occ_map
            raw_local[np.isnan(raw_local)] = 0
            raw_local[np.isinf(raw_local)] = 0
        smooth_local = gaussian_filter(raw_local, sigma=smooth_sigma, mode="constant")
        smooth_local[low_occ_mask_plot] = np.nan
        return smooth_local

    ss_rate_map = _rate_map_for_spikes(
        refined_SS[cell_idx] if refined_SS is not None else None
    )
    cs_rate_map = _rate_map_for_spikes(
        all_CS_spikes[cell_idx] if all_CS_spikes is not None else None
    )

    # Helper to get raw (unsmoothed) rate map for SI calculation
    def _raw_rate_map_for_spikes(spike_times):
        if spike_times is None:
            return None, np.array([])
        spike_times = np.asarray(spike_times, dtype=int)
        spike_times = spike_times[(spike_times >= 0) & (spike_times < total_frames)]
        spike_times = spike_times[valid_frames[spike_times]]
        if spike_times.size == 0:
            return np.zeros_like(raw_map), np.array([])
        binary_full = np.zeros(total_frames, dtype=bool)
        binary_full[spike_times] = True
        binary_sub = binary_full[moving_idx]
        spikes_in_sub_idx_local = np.where(binary_sub)[0]
        if spikes_in_sub_idx_local.size == 0:
            return np.zeros_like(raw_map), np.array([])
        x_spk_local = x_sub[spikes_in_sub_idx_local]
        y_spk_local = y_sub[spikes_in_sub_idx_local]
        spike_map_local, _, _ = np.histogram2d(x_spk_local, y_spk_local, bins=bins)
        with np.errstate(divide="ignore", invalid="ignore"):
            raw_local = spike_map_local / occ_map
            raw_local[np.isnan(raw_local)] = 0
            raw_local[np.isinf(raw_local)] = 0
        return raw_local, binary_sub

    # Get raw rate maps and binary trains for SS and CS
    ss_raw_map, ss_binary_sub = _raw_rate_map_for_spikes(
        refined_SS[cell_idx] if refined_SS is not None else None
    )
    cs_raw_map, cs_binary_sub = _raw_rate_map_for_spikes(
        all_CS_spikes[cell_idx] if all_CS_spikes is not None else None
    )

    # Calculate peak rates for SS and CS (0 if no spikes or all NaN)
    ss_peak_rate = 0.0
    cs_peak_rate = 0.0
    if ss_rate_map is not None:
        ss_max = np.nanmax(ss_rate_map)
        ss_peak_rate = ss_max if np.isfinite(ss_max) else 0.0
    if cs_rate_map is not None:
        cs_max = np.nanmax(cs_rate_map)
        cs_peak_rate = cs_max if np.isfinite(cs_max) else 0.0

    # Calculate spatial information for SS
    ss_si = np.nan
    ss_p_val = np.nan
    if ss_raw_map is not None and ss_binary_sub.size > 0:
        ss_si = calculate_spatial_information(ss_raw_map, occ_map)
        ss_shuffled_si_values = []
        ss_spikes_in_sub = np.sum(ss_binary_sub)
        if ss_spikes_in_sub > 0:
            rng_ss = np.random.default_rng(random_seed + 1)
            for _ in range(num_shuffles):
                shift = int(rng_ss.integers(1, len(ss_binary_sub)))
                shuf_bool = np.roll(ss_binary_sub, shift=shift)
                shuf_x = x_sub[shuf_bool]
                shuf_y = y_sub[shuf_bool]
                shuf_spk_map, _, _ = np.histogram2d(shuf_x, shuf_y, bins=bins)
                with np.errstate(divide="ignore", invalid="ignore"):
                    shuf_rate = shuf_spk_map / occ_map
                    shuf_rate[np.isnan(shuf_rate)] = 0
                ss_shuffled_si_values.append(calculate_spatial_information(shuf_rate, occ_map))
            ss_shuffled_si_arr = np.array(ss_shuffled_si_values)
            ss_p_val = np.sum(ss_shuffled_si_arr >= ss_si) / num_shuffles
        else:
            ss_p_val = 1.0

    # Calculate spatial information for CS
    cs_si = np.nan
    cs_p_val = np.nan
    if cs_raw_map is not None and cs_binary_sub.size > 0:
        cs_si = calculate_spatial_information(cs_raw_map, occ_map)
        cs_shuffled_si_values = []
        cs_spikes_in_sub = np.sum(cs_binary_sub)
        if cs_spikes_in_sub > 0:
            rng_cs = np.random.default_rng(random_seed + 2)
            for _ in range(num_shuffles):
                shift = int(rng_cs.integers(1, len(cs_binary_sub)))
                shuf_bool = np.roll(cs_binary_sub, shift=shift)
                shuf_x = x_sub[shuf_bool]
                shuf_y = y_sub[shuf_bool]
                shuf_spk_map, _, _ = np.histogram2d(shuf_x, shuf_y, bins=bins)
                with np.errstate(divide="ignore", invalid="ignore"):
                    shuf_rate = shuf_spk_map / occ_map
                    shuf_rate[np.isnan(shuf_rate)] = 0
                cs_shuffled_si_values.append(calculate_spatial_information(shuf_rate, occ_map))
            cs_shuffled_si_arr = np.array(cs_shuffled_si_values)
            cs_p_val = np.sum(cs_shuffled_si_arr >= cs_si) / num_shuffles
        else:
            cs_p_val = 1.0

    x_quiet = x_neural[quiet_idx]
    y_quiet = y_neural[quiet_idx]
    occ_counts_quiet, _, _ = np.histogram2d(x_quiet, y_quiet, bins=bins)
    occ_map_quiet = occ_counts_quiet / frame_rate
    occ_map_quiet = gaussian_filter(occ_map_quiet, sigma=2, mode="constant")
    binary_train_quiet = binary_train_full[quiet_idx]
    spikes_in_quiet_idx = np.where(binary_train_quiet)[0]
    x_spk_quiet = x_quiet[spikes_in_quiet_idx]
    y_spk_quiet = y_quiet[spikes_in_quiet_idx]
    spike_map_quiet, _, _ = np.histogram2d(x_spk_quiet, y_spk_quiet, bins=bins)
    with np.errstate(divide="ignore", invalid="ignore"):
        raw_map_quiet = spike_map_quiet / occ_map_quiet
        raw_map_quiet[np.isnan(raw_map_quiet)] = 0
        raw_map_quiet[np.isinf(raw_map_quiet)] = 0
    smooth_map_quiet = gaussian_filter(raw_map_quiet, sigma=smooth_sigma, mode="constant")
    smooth_map_quiet[occ_map_quiet == 0] = np.nan

    # Peak rate for all spikes (0 if no spikes or all NaN)
    peak_rate_raw = np.nanmax(smooth_map_for_pf)
    peak_rate = peak_rate_raw if np.isfinite(peak_rate_raw) else 0.0
    ss_norm_map = None
    cs_norm_map = None
    if np.isfinite(peak_rate) and peak_rate > 0:
        if ss_rate_map is not None:
            ss_norm_map = ss_rate_map / peak_rate
        if cs_rate_map is not None:
            cs_norm_map = cs_rate_map / peak_rate
    
    # ========================================================================
    # Place field detection for All spikes
    # Keep ALL valid components (each must have >= min_field_bins bins)
    # ========================================================================
    place_field_mask = np.zeros_like(smooth_map_for_pf, dtype=bool)
    raw_field_mask = np.zeros_like(smooth_map_for_pf, dtype=bool)
    place_field_components = []  # List of individual place field info (sorted by size)
    
    arena_area = width_real * height_real
    max_field_area = max_field_area_ratio * arena_area
    
    if not (np.isnan(peak_rate) or peak_rate == 0):
        threshold_val = place_field_threshold * peak_rate
        with np.errstate(invalid="ignore"):
            raw_field_mask = smooth_map_for_pf > threshold_val
        raw_field_mask &= ~low_occ_border_mask_analysis
        place_field_mask, place_field_components = _keep_all_valid_components(
            raw_field_mask,
            smooth_map_for_pf,
            min_peak_ratio=min_component_peak_ratio,
            split_multi_peak_fields=split_multi_peak_fields,
            split_secondary_peak_ratio=split_secondary_peak_ratio,
            split_secondary_peak_min_separation_cm=split_secondary_peak_min_separation_cm,
            bin_size_cm=bin_size,
            min_bins=min_field_bins,
        )
    
    # If peak rate is below min_peak_rate, set place field to empty (All spikes)
    if not (np.isfinite(peak_rate) and peak_rate >= min_peak_rate):
        place_field_mask = np.zeros_like(smooth_map_for_pf, dtype=bool)
        place_field_components = []

    # Compute the all-spike SI shuffle before the expensive PF traversal gate.
    # The traversal gate can only remove candidate fields, so cells failing SI,
    # peak-rate, or basic PF detection cannot become place cells later.
    si = calculate_spatial_information(raw_map, occ_map)
    shuffled_si_values = []
    if len(spikes_in_sub_idx) > 0:
        rng = np.random.default_rng(random_seed)
        for _ in range(num_shuffles):
            shift = int(rng.integers(1, len(binary_train_sub)))
            shuf_bool = np.roll(binary_train_sub, shift=shift)
            shuf_x = x_sub[shuf_bool]
            shuf_y = y_sub[shuf_bool]
            shuf_spk_map, _, _ = np.histogram2d(shuf_x, shuf_y, bins=bins)
            with np.errstate(divide="ignore", invalid="ignore"):
                shuf_rate = shuf_spk_map / occ_map
                shuf_rate[np.isnan(shuf_rate)] = 0
            shuffled_si_values.append(calculate_spatial_information(shuf_rate, occ_map))
    else:
        shuffled_si_values = [0] * num_shuffles

    shuffled_si_arr = np.array(shuffled_si_values)
    p_val = np.sum(shuffled_si_arr >= si) / num_shuffles

    # ------------------------------------------------------------------------
    # Optional PF-component filtering by firing traversals (All spikes):
    # keep only PF components with spikes in at least min_pf_firing_traversals
    # traversals. The reliability fraction is still computed for reporting:
    #   reliability = (# traversals with >=1 spike) / (total traversals)
    # ------------------------------------------------------------------------
    pf_component_filter_stats = []
    min_firing_traversals_required = (
        None if min_pf_firing_traversals is None else int(min_pf_firing_traversals)
    )
    component_firing_traversal_filter_enabled = (
        (min_firing_traversals_required is not None)
        and (min_firing_traversals_required > 0)
    )
    reliability_dilation_structure = (
        _build_pf_dilation_structure(
            pf_reliability_dilation_bins, pf_reliability_dilation_shape
        )
        if component_firing_traversal_filter_enabled and pf_reliability_dilation_bins > 0
        else None
    )
    candidate_is_place_cell_for_pf_filter = (
        (p_val < 0.05)
        and bool(place_field_components)
        and (np.isfinite(peak_rate) and peak_rate >= min_peak_rate)
    )
    apply_pf_component_filter = (
        bool(place_field_components)
        and component_firing_traversal_filter_enabled
        and bool(candidate_is_place_cell_for_pf_filter)
    )
    if apply_pf_component_filter:
        cell_spikes_sorted = np.sort(cell_spikes)
        kept_components = []

        for comp_idx, comp in enumerate(place_field_components):
            comp_mask = np.asarray(comp.get("mask"), dtype=bool)
            comp_mask_eval = comp_mask
            reliability_eval_expanded = False
            if reliability_dilation_structure is not None and comp_mask.size > 0 and np.any(comp_mask):
                comp_mask_eval = binary_dilation(comp_mask, structure=reliability_dilation_structure)
                reliability_eval_expanded = True
            reliability_eval_size_bins = int(np.sum(comp_mask_eval)) if comp_mask_eval.size > 0 else 0
            traversal_stats = _compute_distance_defined_pf_firing_traversal_stats(
                x_neural=x_neural,
                y_neural=y_neural,
                speed=speed,
                frame_rate=frame_rate,
                event_spikes=cell_spikes_sorted,
                place_field_mask=comp_mask_eval,
                rate_map=smooth_map_for_pf,
                width_real=width_real,
                height_real=height_real,
                bin_size=bin_size,
                bad_timepoints=bad_timepoints,
                moving_speed_threshold=float(speed_threshold),
                moving_kernel_size=int(kernel_size),
                moving_min_duration_s=float(min_duration_s),
                moving_merge_gap_s=float(pf_firing_traversal_merge_gap_s),
                distance_window_cm=float(pf_firing_traversal_distance_window_cm),
                detection_window_cm=float(pf_firing_traversal_detection_window_cm),
                distance_bin_cm=float(pf_firing_traversal_distance_bin_cm),
                distance_mode=str(pf_firing_traversal_distance_mode),
                center_vicinity_min_cm=int(pf_firing_traversal_center_vicinity_min_cm),
                center_vicinity_max_cm=int(pf_firing_traversal_center_vicinity_max_cm),
                resting_speed_threshold=float(pf_firing_traversal_resting_speed_threshold),
                exclude_trials_with_bad_frames=bool(pf_firing_traversal_exclude_trials_with_bad_frames),
            )

            n_total = int(traversal_stats["n_trials"])
            n_reliable = int(traversal_stats["n_firing_traversals"])
            reliability = traversal_stats["reliability"]

            passes_component = n_reliable >= min_firing_traversals_required
            comp_with_stats = dict(comp)
            comp_with_stats["component_index"] = int(comp_idx)
            comp_with_stats["n_traversals_all"] = n_total
            comp_with_stats["n_reliable_traversals_all"] = int(n_reliable)
            comp_with_stats["n_firing_traversals_all"] = int(n_reliable)
            comp_with_stats["reliability_all"] = reliability
            comp_with_stats["min_pf_firing_traversals"] = int(min_firing_traversals_required)
            comp_with_stats["passes_firing_traversal_filter"] = bool(passes_component)
            comp_with_stats["passes_reliability_filter"] = bool(passes_component)
            comp_with_stats["firing_traversal_filter_spike_type"] = "all"
            comp_with_stats["firing_traversal_detection"] = "distance_defined_iterative"
            comp_with_stats["reliability_eval_expanded"] = bool(reliability_eval_expanded)
            comp_with_stats["reliability_eval_size_bins"] = reliability_eval_size_bins
            comp_with_stats["reliability_eval_radius_bins"] = int(pf_reliability_dilation_bins)
            comp_with_stats["reliability_eval_shape"] = pf_reliability_dilation_shape

            pf_component_filter_stats.append(
                {
                    "component_index": int(comp_idx),
                    "size_bins": int(comp.get("size", 0)),
                    "peak_rate": float(comp.get("peak_rate", np.nan)),
                    "n_traversals_all": n_total,
                    "n_reliable_traversals_all": int(n_reliable),
                    "n_firing_traversals_all": int(n_reliable),
                    "reliability_all": reliability,
                    "min_pf_firing_traversals": int(min_firing_traversals_required),
                    "passes_firing_traversal_filter": bool(passes_component),
                    "passes_reliability_filter": bool(passes_component),
                    "firing_traversal_filter_spike_type": "all",
                    "firing_traversal_detection": "distance_defined_iterative",
                    "reliability_eval_expanded": bool(reliability_eval_expanded),
                    "reliability_eval_size_bins": reliability_eval_size_bins,
                    "reliability_eval_radius_bins": int(pf_reliability_dilation_bins),
                    "reliability_eval_shape": pf_reliability_dilation_shape,
                }
            )
            if passes_component:
                kept_components.append(comp_with_stats)

        place_field_components = kept_components
        place_field_mask = np.zeros_like(smooth_map_for_pf, dtype=bool)
        for comp in place_field_components:
            place_field_mask |= np.asarray(comp.get("mask"), dtype=bool)

    # Final all-spike place-cell status after any firing-traversal PF filtering.
    # SS/CS traversal validation is only meaningful for cells that remain PLCs.
    field_area = np.sum(place_field_mask) * (bin_size ** 2)
    n_place_fields = len(place_field_components)
    is_place_cell = (
        (p_val < 0.05) and
        (n_place_fields > 0) and
        (field_area <= max_field_area) and
        (np.isfinite(peak_rate) and peak_rate >= min_peak_rate)
    )

    def _filter_event_place_field_components(
        components,
        event_spike_times,
        rate_map_for_pf,
        label,
    ):
        filter_stats = []
        if not (components and component_firing_traversal_filter_enabled):
            return list(components or []), filter_stats

        event_spikes = (
            np.asarray(event_spike_times, dtype=int)
            if event_spike_times is not None
            else np.array([], dtype=int)
        )
        event_spikes = event_spikes[(event_spikes >= 0) & (event_spikes < total_frames)]
        event_spikes = event_spikes[valid_frames[event_spikes]]
        event_spikes_sorted = np.sort(event_spikes)

        kept_components = []
        label = str(label)
        traversals_key = f"n_traversals_{label}"
        reliable_key = f"n_reliable_traversals_{label}"
        firing_key = f"n_firing_traversals_{label}"
        reliability_key = f"reliability_{label}"

        for comp_idx, comp in enumerate(components):
            comp_mask = np.asarray(comp.get("mask"), dtype=bool)
            comp_mask_eval = comp_mask
            reliability_eval_expanded = False
            if reliability_dilation_structure is not None and comp_mask.size > 0 and np.any(comp_mask):
                comp_mask_eval = binary_dilation(comp_mask, structure=reliability_dilation_structure)
                reliability_eval_expanded = True
            reliability_eval_size_bins = int(np.sum(comp_mask_eval)) if comp_mask_eval.size > 0 else 0
            traversal_stats = _compute_distance_defined_pf_firing_traversal_stats(
                x_neural=x_neural,
                y_neural=y_neural,
                speed=speed,
                frame_rate=frame_rate,
                event_spikes=event_spikes_sorted,
                place_field_mask=comp_mask_eval,
                rate_map=rate_map_for_pf,
                width_real=width_real,
                height_real=height_real,
                bin_size=bin_size,
                bad_timepoints=bad_timepoints,
                moving_speed_threshold=float(speed_threshold),
                moving_kernel_size=int(kernel_size),
                moving_min_duration_s=float(min_duration_s),
                moving_merge_gap_s=float(pf_firing_traversal_merge_gap_s),
                distance_window_cm=float(pf_firing_traversal_distance_window_cm),
                detection_window_cm=float(pf_firing_traversal_detection_window_cm),
                distance_bin_cm=float(pf_firing_traversal_distance_bin_cm),
                distance_mode=str(pf_firing_traversal_distance_mode),
                center_vicinity_min_cm=int(pf_firing_traversal_center_vicinity_min_cm),
                center_vicinity_max_cm=int(pf_firing_traversal_center_vicinity_max_cm),
                resting_speed_threshold=float(pf_firing_traversal_resting_speed_threshold),
                exclude_trials_with_bad_frames=bool(pf_firing_traversal_exclude_trials_with_bad_frames),
            )

            n_total = int(traversal_stats["n_trials"])
            n_reliable = int(traversal_stats["n_firing_traversals"])
            reliability = traversal_stats["reliability"]

            passes_component = n_reliable >= min_firing_traversals_required
            comp_with_stats = dict(comp)
            comp_with_stats["component_index"] = int(comp_idx)
            comp_with_stats[traversals_key] = n_total
            comp_with_stats[reliable_key] = int(n_reliable)
            comp_with_stats[firing_key] = int(n_reliable)
            comp_with_stats[reliability_key] = reliability
            comp_with_stats["min_pf_firing_traversals"] = int(min_firing_traversals_required)
            comp_with_stats["passes_firing_traversal_filter"] = bool(passes_component)
            comp_with_stats["passes_reliability_filter"] = bool(passes_component)
            comp_with_stats["firing_traversal_filter_spike_type"] = label
            comp_with_stats["firing_traversal_detection"] = "distance_defined_iterative"
            comp_with_stats["reliability_filter_spike_type"] = label
            comp_with_stats["reliability_eval_expanded"] = bool(reliability_eval_expanded)
            comp_with_stats["reliability_eval_size_bins"] = reliability_eval_size_bins
            comp_with_stats["reliability_eval_radius_bins"] = int(pf_reliability_dilation_bins)
            comp_with_stats["reliability_eval_shape"] = pf_reliability_dilation_shape

            filter_stats.append(
                {
                    "component_index": int(comp_idx),
                    "spike_type": label,
                    "size_bins": int(comp.get("size", 0)),
                    "peak_rate": float(comp.get("peak_rate", np.nan)),
                    traversals_key: n_total,
                    reliable_key: int(n_reliable),
                    firing_key: int(n_reliable),
                    reliability_key: reliability,
                    "min_pf_firing_traversals": int(min_firing_traversals_required),
                    "passes_firing_traversal_filter": bool(passes_component),
                    "passes_reliability_filter": bool(passes_component),
                    "firing_traversal_filter_spike_type": label,
                    "firing_traversal_detection": "distance_defined_iterative",
                    "reliability_eval_expanded": bool(reliability_eval_expanded),
                    "reliability_eval_size_bins": reliability_eval_size_bins,
                    "reliability_eval_radius_bins": int(pf_reliability_dilation_bins),
                    "reliability_eval_shape": pf_reliability_dilation_shape,
                }
            )
            if passes_component:
                kept_components.append(comp_with_stats)

        return kept_components, filter_stats

    def _mask_from_components(components, template):
        component_mask = np.zeros_like(template, dtype=bool)
        for comp in components or []:
            mask = np.asarray(comp.get("mask"), dtype=bool)
            if mask.shape == component_mask.shape:
                component_mask |= mask
        return component_mask

    # Calculate place field masks for SS and CS separately
    # These are always computed (not just for display) since they're needed for is_place_cell_ss/cs
    ss_place_field_mask = np.zeros_like(smooth_map_for_pf, dtype=bool)
    cs_place_field_mask = np.zeros_like(smooth_map_for_pf, dtype=bool)
    ss_place_field_components = []
    cs_place_field_components = []
    ss_pf_component_filter_stats = []
    cs_pf_component_filter_stats = []
    ss_smooth_for_pf = None
    cs_smooth_for_pf = None
    
    # SS place field: computed from SS rate map, then component-filtered by
    # minimum firing traversals only for final all-spike PLCs or SS PLC candidates.
    if ss_rate_map is not None:
        ss_peak = np.nanmax(ss_rate_map)
        if np.isfinite(ss_peak) and ss_peak > 0:
            ss_threshold_val = place_field_threshold * ss_peak
            ss_smooth_for_pf = gaussian_filter(ss_raw_map, sigma=smooth_sigma, mode="constant")
            ss_smooth_for_pf[occ_map == 0] = np.nan
            if np.any(trim_mask_analysis):
                ss_smooth_for_pf[trim_mask_analysis] = np.nan
            with np.errstate(invalid="ignore"):
                ss_raw_field_mask = ss_smooth_for_pf > ss_threshold_val
            ss_raw_field_mask &= ~low_occ_border_mask_analysis
            ss_place_field_mask_candidate, ss_place_field_components_candidate = _keep_all_valid_components(
                ss_raw_field_mask,
                ss_smooth_for_pf,
                min_peak_ratio=min_component_peak_ratio,
                split_multi_peak_fields=split_multi_peak_fields,
                split_secondary_peak_ratio=split_secondary_peak_ratio,
                split_secondary_peak_min_separation_cm=split_secondary_peak_min_separation_cm,
                bin_size_cm=bin_size,
                min_bins=min_field_bins,
            )
            # Always use the computed mask (no area limit for SS/CS masks)
            ss_place_field_mask = ss_place_field_mask_candidate
            ss_place_field_components = ss_place_field_components_candidate
    
    # If SS peak rate is below min_peak_rate, set SS place field to empty
    if not (np.isfinite(ss_peak_rate) and ss_peak_rate >= min_peak_rate):
        ss_place_field_mask = np.zeros_like(smooth_map_for_pf, dtype=bool)
        ss_place_field_components = []
    elif (
        ss_place_field_components
        and (
            is_place_cell
            or (
                (ss_p_val < 0.05)
                and (np.isfinite(ss_peak_rate) and ss_peak_rate >= min_peak_rate)
            )
        )
    ):
        ss_spike_times_for_reliability = (
            refined_SS[cell_idx]
            if refined_SS is not None and cell_idx < len(refined_SS)
            else None
        )
        ss_place_field_components, ss_pf_component_filter_stats = _filter_event_place_field_components(
            ss_place_field_components,
            ss_spike_times_for_reliability,
            ss_smooth_for_pf if ss_smooth_for_pf is not None else ss_rate_map,
            "ss",
        )
        ss_place_field_mask = _mask_from_components(ss_place_field_components, smooth_map_for_pf)
    
    # CS place field: computed from CS rate map, then component-filtered by
    # minimum firing traversals only for final all-spike PLCs or CS PLC candidates.
    if cs_rate_map is not None:
        cs_peak = np.nanmax(cs_rate_map)
        if np.isfinite(cs_peak) and cs_peak > 0:
            cs_threshold_val = place_field_threshold * cs_peak
            cs_smooth_for_pf = gaussian_filter(cs_raw_map, sigma=smooth_sigma, mode="constant")
            cs_smooth_for_pf[occ_map == 0] = np.nan
            if np.any(trim_mask_analysis):
                cs_smooth_for_pf[trim_mask_analysis] = np.nan
            with np.errstate(invalid="ignore"):
                cs_raw_field_mask = cs_smooth_for_pf > cs_threshold_val
            cs_raw_field_mask &= ~low_occ_border_mask_analysis
            cs_place_field_mask_candidate, cs_place_field_components_candidate = _keep_all_valid_components(
                cs_raw_field_mask,
                cs_smooth_for_pf,
                min_peak_ratio=min_component_peak_ratio,
                split_multi_peak_fields=split_multi_peak_fields,
                split_secondary_peak_ratio=split_secondary_peak_ratio,
                split_secondary_peak_min_separation_cm=split_secondary_peak_min_separation_cm,
                bin_size_cm=bin_size,
                min_bins=min_field_bins,
            )
            # Always use the computed mask (no area limit for SS/CS masks)
            cs_place_field_mask = cs_place_field_mask_candidate
            cs_place_field_components = cs_place_field_components_candidate
    
    # If CS peak rate is below min_peak_rate, set CS place field to empty
    if not (np.isfinite(cs_peak_rate) and cs_peak_rate >= min_peak_rate):
        cs_place_field_mask = np.zeros_like(smooth_map_for_pf, dtype=bool)
        cs_place_field_components = []
    elif (
        cs_place_field_components
        and (
            is_place_cell
            or (
                (cs_p_val < 0.05)
                and (np.isfinite(cs_peak_rate) and cs_peak_rate >= min_peak_rate)
            )
        )
    ):
        cs_spike_times_for_reliability = (
            all_CS_spikes[cell_idx]
            if all_CS_spikes is not None and cell_idx < len(all_CS_spikes)
            else None
        )
        cs_place_field_components, cs_pf_component_filter_stats = _filter_event_place_field_components(
            cs_place_field_components,
            cs_spike_times_for_reliability,
            cs_smooth_for_pf if cs_smooth_for_pf is not None else cs_rate_map,
            "cs",
        )
        cs_place_field_mask = _mask_from_components(cs_place_field_components, smooth_map_for_pf)

    # Field area is the SUMMED area of all valid place field components
    field_area = np.sum(place_field_mask) * (bin_size ** 2)
    n_place_fields = len(place_field_components)
    
    # Place cell criteria:
    # 1. Significant spatial information (p < 0.05)
    # 2. At least one valid place field component remains
    # 3. SUMMED field area within limit
    # 4. Peak firing rate >= min_peak_rate
    is_place_cell = (
        (p_val < 0.05) and 
        (n_place_fields > 0) and
        (field_area <= max_field_area) and 
        (np.isfinite(peak_rate) and peak_rate >= min_peak_rate)
    )
    
    # Extract place field sizes and peak rates (sorted by peak rate, highest first)
    # pf_sizes[0] is the primary (strongest) place field, pf_sizes[1] is secondary, etc.
    # These are in real units (cm^2)
    # If no place fields detected, use [0] instead of [] so downstream code gets 0 instead of NaN
    pf_sizes = [comp['size'] * (bin_size ** 2) for comp in place_field_components] if place_field_components else [0]
    pf_peak_rates = [comp['peak_rate'] for comp in place_field_components]

    # Determine if cell is a place cell based on SS or CS only
    # Same criteria as is_place_cell: significant SI (p < 0.05), field area <= max_field_area, peak rate >= min_peak_rate
    ss_field_area = np.sum(ss_place_field_mask) * (bin_size ** 2)
    cs_field_area = np.sum(cs_place_field_mask) * (bin_size ** 2)

    # Extract SS/CS place field sizes and peak rates (sorted by peak rate, highest first)
    # If no place fields detected, use [0] instead of [] so downstream code gets 0 instead of NaN
    ss_pf_sizes = [comp['size'] * (bin_size ** 2) for comp in ss_place_field_components] if ss_place_field_components else [0]
    cs_pf_sizes = [comp['size'] * (bin_size ** 2) for comp in cs_place_field_components] if cs_place_field_components else [0]
    ss_pf_peak_rates = [comp['peak_rate'] for comp in ss_place_field_components]
    cs_pf_peak_rates = [comp['peak_rate'] for comp in cs_place_field_components]
    n_ss_place_fields = len(ss_place_field_components)
    n_cs_place_fields = len(cs_place_field_components)

    is_place_cell_ss = (
        (ss_p_val < 0.05) and
        (n_ss_place_fields > 0) and
        (ss_field_area <= max_field_area) and  # Field area within limit (same as is_place_cell)
        (np.isfinite(ss_peak_rate) and ss_peak_rate >= min_peak_rate)
    )
    is_place_cell_cs = (
        (cs_p_val < 0.05) and
        (n_cs_place_fields > 0) and
        (cs_field_area <= max_field_area) and  # Field area within limit (same as is_place_cell)
        (np.isfinite(cs_peak_rate) and cs_peak_rate >= min_peak_rate)
    )

    # Spatial information in bits/spike for all, SS, CS
    si_bits_per_spike = calculate_spatial_information_bits_per_spike(smooth_map, occ_map)
    si_bits_per_spike_ss = calculate_spatial_information_bits_per_spike(ss_rate_map, occ_map) if ss_rate_map is not None else np.nan
    si_bits_per_spike_cs = calculate_spatial_information_bits_per_spike(cs_rate_map, occ_map) if cs_rate_map is not None else np.nan
    coherence_all = calculate_rate_map_coherence(smooth_map)
    coherence_ss = calculate_rate_map_coherence(ss_rate_map) if ss_rate_map is not None else np.nan
    coherence_cs = calculate_rate_map_coherence(cs_rate_map) if cs_rate_map is not None else np.nan
    sparsity_all = calculate_rate_map_sparsity(smooth_map, occ_map)
    sparsity_ss = calculate_rate_map_sparsity(ss_rate_map, occ_map) if ss_rate_map is not None else np.nan
    sparsity_cs = calculate_rate_map_sparsity(cs_rate_map, occ_map) if cs_rate_map is not None else np.nan

    # Calculate theta/slow maps and correlations if traces are provided
    theta_map = None
    slow_map = None
    theta_corr_all = np.nan
    theta_corr_ss = np.nan
    theta_corr_cs = np.nan
    slow_corr_all = np.nan
    slow_corr_ss = np.nan
    slow_corr_cs = np.nan
    if traces is not None:
        trace = np.asarray(traces[cell_idx], dtype=float)
        trace = interpolate_nan_segment(trace)
        theta_vm = bandpass_filter(
            trace, theta_freqs[0], theta_freqs[1], frame_rate, order=5
        )
        slow_vm = lowpass_filter(trace, slow_freqs, frame_rate, order=5)
        theta_amp = np.abs(signal.hilbert(theta_vm))

        def _mean_map(values):
            if x_sub.size == 0:
                return np.full(
                    (len(bins[0]) - 1, len(bins[1]) - 1), np.nan, dtype=float
                )
            counts, _, _ = np.histogram2d(x_sub, y_sub, bins=bins)
            sums, _, _ = np.histogram2d(x_sub, y_sub, bins=bins, weights=values)
            if smooth_sigma and smooth_sigma > 0:
                counts = gaussian_filter(counts, sigma=smooth_sigma, mode="constant")
                sums = gaussian_filter(sums, sigma=smooth_sigma, mode="constant")
            with np.errstate(divide="ignore", invalid="ignore"):
                mean_map = sums / counts
            mean_map[counts == 0] = np.nan
            mean_map[low_occ_mask_plot] = np.nan
            return mean_map

        theta_map = _mean_map(theta_amp[moving_idx])
        slow_map = _mean_map(slow_vm[moving_idx])

        def _map_correlation(map1, map2):
            """Calculate Pearson correlation between two maps, ignoring NaN bins."""
            if map1 is None or map2 is None:
                return np.nan
            valid = np.isfinite(map1) & np.isfinite(map2)
            if np.sum(valid) < 3:
                return np.nan
            v1 = map1[valid]
            v2 = map2[valid]
            if np.std(v1) == 0 or np.std(v2) == 0:
                return np.nan
            return np.corrcoef(v1, v2)[0, 1]

        # Correlations with theta map
        theta_corr_all = _map_correlation(theta_map, smooth_map)
        theta_corr_ss = _map_correlation(theta_map, ss_rate_map)
        theta_corr_cs = _map_correlation(theta_map, cs_rate_map)

        # Correlations with slow map
        slow_corr_all = _map_correlation(slow_map, smooth_map)
        slow_corr_ss = _map_correlation(slow_map, ss_rate_map)
        slow_corr_cs = _map_correlation(slow_map, cs_rate_map)

    num_cells = len(spikes)
    if num_cells <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, num_cells))
    elif num_cells <= 20:
        colors = plt.cm.tab20(np.linspace(0, 1, num_cells))
    else:
        colors = plt.cm.hsv(np.linspace(0, 1, num_cells, endpoint=False))
    cell_color = colors[cell_idx]

    x_all_valid = x_neural[valid_frames]
    y_all_valid = y_neural[valid_frames]
    ss_x = np.array([])
    ss_y = np.array([])
    cs_x = np.array([])
    cs_y = np.array([])
    if separate_spikes:
        ss_times = refined_SS[cell_idx] if refined_SS is not None else None
        cs_times = all_CS_spikes[cell_idx] if all_CS_spikes is not None else None
        ss_x, ss_y = _spike_positions(ss_times)
        cs_x, cs_y = _spike_positions(cs_times)

    spike_shapes = None
    if save_spike_shapes:
        if traces is None:
            raise ValueError("save_spike_shapes=True requires `traces`.")
        if refined_SS is None:
            raise ValueError("save_spike_shapes=True requires `refined_SS`.")

        trace = np.asarray(traces[cell_idx], dtype=float)
        if trace.size != total_frames:
            raise ValueError(
                f"traces[{cell_idx}] length ({trace.size}) must match x_neural length ({total_frames})."
            )
        trace_mf = median_filter(trace, size=int(spike_shape_median_filter_size))

        inside_pf = _positions_in_place_field(x_neural, y_neural, bins, place_field_mask)

        def _normalize_window(spike_idx, pre_frames, post_frames):
            if spike_idx - pre_frames < 0 or spike_idx + post_frames >= len(trace):
                return None
            baseline_region = trace_mf[
                max(0, spike_idx - int(spike_shape_baseline_frames)) : spike_idx
            ]
            baseline = np.min(baseline_region) if baseline_region.size > 0 else 0.0
            height = trace[spike_idx] - baseline
            if not np.isfinite(height) or height == 0:
                return None
            window = trace[spike_idx - pre_frames : spike_idx + post_frames + 1]
            if window.size != (pre_frames + post_frames + 1):
                return None
            return (window - baseline) / height

        def _collect_shapes(event_indices, pre_ms, post_ms):
            pre_frames = int(round((float(pre_ms) / 1000.0) * frame_rate))
            post_frames = int(round((float(post_ms) / 1000.0) * frame_rate))
            time_ms = (np.arange(-pre_frames, post_frames + 1) / frame_rate) * 1000.0

            shapes = {"run_in": [], "run_out": [], "rest_in": [], "rest_out": []}
            indices = {"run_in": [], "run_out": [], "rest_in": [], "rest_out": []}

            event_indices = np.asarray(event_indices, dtype=int)
            event_indices = event_indices[(event_indices >= 0) & (event_indices < total_frames)]
            for spike_idx in event_indices:
                if not valid_frames[spike_idx]:
                    continue
                window = _normalize_window(spike_idx, pre_frames, post_frames)
                if window is None:
                    continue
                state_key = "run" if moving_mask[spike_idx] else "rest"
                pf_key = "in" if inside_pf[spike_idx] else "out"
                key = f"{state_key}_{pf_key}"
                shapes[key].append(window)
                indices[key].append(int(spike_idx))

            return time_ms, shapes, indices

        ss_event_indices = np.asarray(refined_SS[cell_idx], dtype=int)
        ss_event_indices = ss_event_indices[(ss_event_indices >= 0) & (ss_event_indices < total_frames)]
        ss_event_indices = np.unique(ss_event_indices)
        ss_event_indices = np.sort(ss_event_indices)

        if ss_shape_min_separation_ms is not None:
            min_sep_frames = int(round((float(ss_shape_min_separation_ms) / 1000.0) * frame_rate))
            if min_sep_frames > 0 and ss_event_indices.size > 1:
                diffs = np.diff(ss_event_indices)
                prev_dist = np.r_[np.inf, diffs]
                next_dist = np.r_[diffs, np.inf]
                keep = np.minimum(prev_dist, next_dist) >= min_sep_frames
                ss_event_indices = ss_event_indices[keep]

        burst_event_indices = np.array([], dtype=int)
        if complex_bursts_dicts is not None:
            bursts = None
            if isinstance(complex_bursts_dicts, (list, tuple)):
                if cell_idx < len(complex_bursts_dicts):
                    bursts = complex_bursts_dicts[cell_idx]
            else:
                bursts = complex_bursts_dicts
            if bursts is not None:
                starts = np.asarray(bursts.get("starts", []), dtype=int)
                ends = np.asarray(bursts.get("ends", []), dtype=int)
                if (
                    all_CS_spikes is not None
                    and cell_idx < len(all_CS_spikes)
                    and starts.size > 0
                    and ends.size > 0
                ):
                    cs_spikes = np.asarray(all_CS_spikes[cell_idx], dtype=int)
                    cs_spikes = cs_spikes[(cs_spikes >= 0) & (cs_spikes < total_frames)]
                    burst_first = []
                    for start_idx, end_idx in zip(starts, ends):
                        in_burst = cs_spikes[(cs_spikes >= start_idx) & (cs_spikes <= end_idx)]
                        if in_burst.size:
                            burst_first.append(int(in_burst[0]))
                    burst_event_indices = np.asarray(burst_first, dtype=int)
                elif starts.size > 0:
                    burst_event_indices = starts
                else:
                    burst_event_indices = np.asarray(bursts.get("complex_bursts", []), dtype=int)

        time_ms_ss, ss_shapes, ss_indices = _collect_shapes(
            ss_event_indices, ss_shape_pre_ms, ss_shape_post_ms
        )
        time_ms_cb, cb_shapes, cb_indices = _collect_shapes(
            burst_event_indices, cb_shape_pre_ms, cb_shape_post_ms
        )

        spike_shapes = {
            "simple": {
                "time_ms": time_ms_ss,
                "shapes": ss_shapes,
                "indices": ss_indices,
            },
            "complex": {
                "time_ms": time_ms_cb,
                "shapes": cb_shapes,
                "indices": cb_indices,
            },
            "params": {
                "ss_shape_pre_ms": float(ss_shape_pre_ms),
                "ss_shape_post_ms": float(ss_shape_post_ms),
                "cb_shape_pre_ms": float(cb_shape_pre_ms),
                "cb_shape_post_ms": float(cb_shape_post_ms),
                "ss_shape_min_separation_ms": None if ss_shape_min_separation_ms is None else float(ss_shape_min_separation_ms),
                "spike_shape_median_filter_size": int(spike_shape_median_filter_size),
                "spike_shape_baseline_frames": int(spike_shape_baseline_frames),
            },
        }

    burst_metrics_by_condition = None
    if save_burst_metrics:
        inside_pf = np.zeros(total_frames, dtype=bool)
        if place_field_mask is not None and np.asarray(place_field_mask).size:
            inside_pf = _positions_in_place_field(x_neural, y_neural, bins, place_field_mask)

        burst_metrics_by_condition = {
            "complex": {"run_in": [], "run_out": [], "rest_in": [], "rest_out": []},
            "simple": {"run_in": [], "run_out": [], "rest_in": [], "rest_out": []},
            "params": {
                "include_simple_bursts_metrics": bool(include_simple_bursts_metrics),
            },
        }

        def _add_burst_entry(burst_type, entry):
            start_idx = int(entry.get("start", -1))
            if start_idx < 0 or start_idx >= total_frames:
                return
            if not valid_frames[start_idx]:
                return
            state = "run" if moving_mask[start_idx] else "rest"
            pf_key = "in" if inside_pf[start_idx] else "out"
            cond = f"{state}_{pf_key}"
            if cond not in burst_metrics_by_condition[burst_type]:
                return
            entry = dict(entry)
            entry["cell_id"] = int(cell_idx)
            entry["condition"] = cond
            entry["state"] = state
            entry["pf"] = pf_key
            burst_metrics_by_condition[burst_type][cond].append(entry)

        cell_burst_metrics = None
        if burst_metrics is not None:
            if isinstance(burst_metrics, (list, tuple)) and burst_metrics:
                if isinstance(burst_metrics[0], dict):
                    cell_burst_metrics = burst_metrics
                elif isinstance(burst_metrics[0], (list, tuple)):
                    cell_burst_metrics = burst_metrics[cell_idx] if cell_idx < len(burst_metrics) else []
                else:
                    cell_burst_metrics = burst_metrics
            else:
                cell_burst_metrics = burst_metrics

        if cell_burst_metrics is not None:
            for burst in cell_burst_metrics:
                if not isinstance(burst, dict):
                    continue
                is_complex = bool(burst.get("is_complex", False))
                is_single = bool(burst.get("is_single", False))
                if is_complex:
                    _add_burst_entry("complex", burst)
                elif include_simple_bursts_metrics and (not is_single):
                    _add_burst_entry("simple", burst)
        else:
            # Fallback: build complex-burst metrics from `complex_bursts_dicts` + trace
            bursts = None
            if complex_bursts_dicts is not None:
                if isinstance(complex_bursts_dicts, (list, tuple)):
                    if cell_idx < len(complex_bursts_dicts):
                        bursts = complex_bursts_dicts[cell_idx]
                else:
                    bursts = complex_bursts_dicts

            if bursts is not None:
                starts = np.asarray(bursts.get("starts", []), dtype=int)
                ends = np.asarray(bursts.get("ends", []), dtype=int)
                amplitudes = np.asarray(bursts.get("amplitudes", []), dtype=float)
                durations_ms = np.asarray(bursts.get("durations_ms", []), dtype=float)
                baselines = np.asarray(bursts.get("baselines", []), dtype=float)

                trace_for_auc = None
                if "trace_mf" in bursts:
                    trace_for_auc = np.asarray(bursts.get("trace_mf"), dtype=float)
                elif traces is not None:
                    trace_for_auc = median_filter(
                        np.asarray(traces[cell_idx], dtype=float), size=int(spike_shape_median_filter_size)
                    )
                if trace_for_auc is None or trace_for_auc.size != total_frames:
                    trace_for_auc = None

                cs_spikes = None
                if all_CS_spikes is not None and cell_idx < len(all_CS_spikes):
                    cs_spikes = np.asarray(all_CS_spikes[cell_idx], dtype=int)
                    cs_spikes = cs_spikes[(cs_spikes >= 0) & (cs_spikes < total_frames)]
                    cs_spikes = np.sort(cs_spikes)

                n_bursts = int(min(starts.size, ends.size))
                for i in range(n_bursts):
                    start = int(starts[i])
                    end = int(ends[i])
                    if start < 0 or start >= total_frames or end < 0 or end >= total_frames:
                        continue
                    if end < start:
                        continue
                    if not valid_frames[start]:
                        continue

                    baseline = float(baselines[i]) if (i < baselines.size and np.isfinite(baselines[i])) else 0.0

                    if i < amplitudes.size and np.isfinite(amplitudes[i]):
                        peak_amp = float(amplitudes[i])
                    elif trace_for_auc is not None:
                        window = trace_for_auc[start : end + 1]
                        window = np.where(np.isfinite(window), window, 0.0)
                        peak_amp = float(np.nanmax(window - baseline)) if window.size else np.nan
                    else:
                        peak_amp = np.nan

                    if i < durations_ms.size and np.isfinite(durations_ms[i]):
                        duration_ms = float(durations_ms[i])
                    else:
                        duration_ms = float((end - start + 1) * 1000.0 / frame_rate)

                    if trace_for_auc is not None:
                        window = trace_for_auc[start : end + 1]
                        window = np.where(np.isfinite(window), window, 0.0)
                        auc = float(np.trapz(np.clip(window - baseline, 0, None), dx=1.0 / frame_rate))
                    else:
                        auc = np.nan

                    if cs_spikes is not None:
                        n_spikes = int(np.sum((cs_spikes >= start) & (cs_spikes <= end)))
                    else:
                        n_spikes = np.nan

                    _add_burst_entry(
                        "complex",
                        {
                            "start": start,
                            "end": end,
                            "n_spikes": n_spikes,
                            "peak_amp": peak_amp,
                            "baseline": baseline,
                            "duration_ms": duration_ms,
                            "auc": auc,
                            "is_complex": True,
                            "is_single": False,
                        },
                    )

    spike_burst_rate_metrics = None
    if save_spike_burst_rate_metrics:
        # Per-condition rates and burst probability (run/rest × in/out PF).
        conditions = ("run_in", "run_out", "rest_in", "rest_out")
        spike_burst_rate_metrics = {
            "conditions": list(conditions),
            "simple_rate": {c: np.nan for c in conditions},
            "burst_rate": {c: np.nan for c in conditions},
            "burst_prob": {c: np.nan for c in conditions},
            "simple_counts": {c: 0 for c in conditions},
            "burst_counts": {c: 0 for c in conditions},
            "time_s": {c: np.nan for c in conditions},
            "params": {
                "frame_rate": float(frame_rate),
                "speed_threshold": float(speed_threshold),
                "min_cb_bursts_per_condition": int(min_cb_bursts_per_condition),
            },
        }

        if is_place_cell and place_field_mask is not None and np.asarray(place_field_mask).size and np.any(place_field_mask):
            inside_pf_all = _positions_in_place_field(x_neural, y_neural, bins, place_field_mask)

            # Event indices (frame indices)
            simple_events = np.array([], dtype=int)
            if refined_SS is not None and cell_idx < len(refined_SS):
                simple_events = np.asarray(refined_SS[cell_idx], dtype=int)
                simple_events = simple_events[(simple_events >= 0) & (simple_events < total_frames)]

            burst_events = np.array([], dtype=int)
            bursts = None
            if complex_bursts_dicts is not None:
                if isinstance(complex_bursts_dicts, (list, tuple)):
                    if cell_idx < len(complex_bursts_dicts):
                        bursts = complex_bursts_dicts[cell_idx]
                else:
                    bursts = complex_bursts_dicts
            if isinstance(bursts, dict):
                starts = np.asarray(bursts.get("starts", []), dtype=int)
                ends = np.asarray(bursts.get("ends", []), dtype=int)
                cs_spikes = None
                if all_CS_spikes is not None and cell_idx < len(all_CS_spikes):
                    cs_spikes = np.asarray(all_CS_spikes[cell_idx], dtype=int)
                    cs_spikes = cs_spikes[(cs_spikes >= 0) & (cs_spikes < total_frames)]
                    cs_spikes = np.sort(cs_spikes)
                if cs_spikes is not None and starts.size and ends.size:
                    burst_first = []
                    for start_idx, end_idx in zip(starts, ends):
                        in_burst = cs_spikes[(cs_spikes >= start_idx) & (cs_spikes <= end_idx)]
                        if in_burst.size == 0:
                            continue
                        burst_first.append(int(in_burst[0]))
                    burst_events = np.asarray(burst_first, dtype=int)
                else:
                    burst_events = np.asarray(bursts.get("complex_bursts", []), dtype=int)
                burst_events = burst_events[(burst_events >= 0) & (burst_events < total_frames)]

            cond_masks = {
                "run_in": moving_mask & inside_pf_all & valid_frames,
                "run_out": moving_mask & (~inside_pf_all) & valid_frames,
                "rest_in": (~moving_mask) & inside_pf_all & valid_frames,
                "rest_out": (~moving_mask) & (~inside_pf_all) & valid_frames,
            }

            for cond in conditions:
                mask = cond_masks[cond]
                time_s = float(np.sum(mask) / frame_rate) if frame_rate > 0 else np.nan
                spike_burst_rate_metrics["time_s"][cond] = time_s
                if not np.isfinite(time_s) or time_s <= 0:
                    continue

                simple_count = int(np.sum(mask[simple_events])) if simple_events.size else 0
                burst_count = int(np.sum(mask[burst_events])) if burst_events.size else 0
                total_events = simple_count + burst_count

                spike_burst_rate_metrics["simple_counts"][cond] = simple_count
                spike_burst_rate_metrics["burst_counts"][cond] = burst_count
                spike_burst_rate_metrics["simple_rate"][cond] = float(simple_count / time_s)
                spike_burst_rate_metrics["burst_rate"][cond] = float(burst_count / time_s)
                spike_burst_rate_metrics["burst_prob"][cond] = (
                    float(burst_count / total_events) if total_events > 0 else np.nan
                )

    results = {
        "cell_id": cell_idx,
        "is_place_cell": is_place_cell,
        "si": si,
        "p_value": p_val,
        "shuffle_dist": shuffled_si_arr,
        "rate_map": smooth_map,
        "ss_rate_map": ss_rate_map,
        "cs_rate_map": cs_rate_map,
        "ss_norm_map": ss_norm_map,
        "cs_norm_map": cs_norm_map,
        "ss_peak_rate": ss_peak_rate,
        "cs_peak_rate": cs_peak_rate,
        "ss_si": ss_si,
        "ss_p_value": ss_p_val,
        "cs_si": cs_si,
        "cs_p_value": cs_p_val,
        "is_place_cell_ss": is_place_cell_ss,
        "is_place_cell_cs": is_place_cell_cs,
        "si_bits_per_spike": si_bits_per_spike,
        "si_bits_per_spike_ss": si_bits_per_spike_ss,
        "si_bits_per_spike_cs": si_bits_per_spike_cs,
        "coherence_all": coherence_all,
        "coherence_ss": coherence_ss,
        "coherence_cs": coherence_cs,
        "sparsity_all": sparsity_all,
        "sparsity_ss": sparsity_ss,
        "sparsity_cs": sparsity_cs,
        "theta_corr_all": theta_corr_all,
        "theta_corr_ss": theta_corr_ss,
        "theta_corr_cs": theta_corr_cs,
        "slow_corr_all": slow_corr_all,
        "slow_corr_ss": slow_corr_ss,
        "slow_corr_cs": slow_corr_cs,
        "theta_map": theta_map,
        "slow_map": slow_map,
        "rate_map_quiet": smooth_map_quiet,
        "occupancy": occ_map,
        "occupancy_quiet": occ_map_quiet,
        "place_field_mask": place_field_mask,
        "ss_place_field_mask": ss_place_field_mask,
        "cs_place_field_mask": cs_place_field_mask,
        "place_field_components": place_field_components,
        "pf_component_filter_stats": pf_component_filter_stats,
        "ss_place_field_components": ss_place_field_components,
        "cs_place_field_components": cs_place_field_components,
        "ss_pf_component_filter_stats": ss_pf_component_filter_stats,
        "cs_pf_component_filter_stats": cs_pf_component_filter_stats,
        "n_place_fields": n_place_fields,
        "n_ss_place_fields": n_ss_place_fields,
        "n_cs_place_fields": n_cs_place_fields,
        "pf_sizes": pf_sizes,  # List of sizes in cm^2, sorted by peak rate (primary first)
        "ss_pf_sizes": ss_pf_sizes,
        "cs_pf_sizes": cs_pf_sizes,
        "pf_peak_rates": pf_peak_rates,  # Peak rate within each PF, sorted (primary first)
        "ss_pf_peak_rates": ss_pf_peak_rates,
        "cs_pf_peak_rates": cs_pf_peak_rates,
        "field_area": field_area,  # Total (summed) field area in cm^2
        "ss_field_area": ss_field_area,
        "cs_field_area": cs_field_area,
        "peak_rate": peak_rate,
        "x_traj": x_sub,
        "y_traj": y_sub,
        "spikes_x": x_spk,
        "spikes_y": y_spk,
        "moving_epochs": moving_epochs,
        "moving_indices": moving_idx,
        "plot_data": {
            "extent": extent,
            "low_occ_mask": low_occ_mask_plot,
            "top_row_nonocc_fraction_raw": top_row_nonocc_fraction_raw,
            "top_row_trimmed_analysis": bool(top_row_trimmed_analysis),
            "top_row_trimmed_plotting": bool(top_row_trimmed_plotting),
            "x_all_valid": x_all_valid,
            "y_all_valid": y_all_valid,
            "cell_color": cell_color,
            "ss_spikes_x": ss_x,
            "ss_spikes_y": ss_y,
            "cs_spikes_x": cs_x,
            "cs_spikes_y": cs_y,
        },
        "plot_params": {
            "plot_pf_contours": plot_pf_contours,
            "plot_sub": plot_sub,
            "separate_spikes": separate_spikes,
            "simple_spike_color": simple_spike_color,
            "complex_spike_color": complex_spike_color,
            "display_cell_num": display_cell_num,
            "is_last_column": is_last_column,
            "is_first_column": is_first_column,
            "display_PF_SS_CS": display_PF_SS_CS,
            "width_real": width_real,
            "height_real": height_real,
        },
        "spike_shapes": spike_shapes,
        "burst_metrics": burst_metrics_by_condition,
        "spike_burst_rate_metrics": spike_burst_rate_metrics,
        "params": {
            "width_real": width_real,
            "height_real": height_real,
            "bin_size": bin_size,
            "smooth_sigma": smooth_sigma,
            "num_shuffles": num_shuffles,
            "place_field_threshold": place_field_threshold,
            "min_component_peak_ratio": float(min_component_peak_ratio),
            "split_multi_peak_fields": bool(split_multi_peak_fields),
            "split_secondary_peak_ratio": float(split_secondary_peak_ratio),
            "split_secondary_peak_min_separation_cm": float(split_secondary_peak_min_separation_cm),
            "min_field_bins": min_field_bins,
            "kernel_size": kernel_size,
            "filter_type": filter_type,
            "speed_threshold": speed_threshold,
            "min_duration_s": min_duration_s,
            "merge_gap_s": merge_gap_s,
            "max_field_area_ratio": max_field_area_ratio,
            "min_occupancy_s": float(min_occupancy_s),
            "occ_smooth_sigma": float(occ_smooth_sigma),
            "use_smoothed_occ_mask": bool(use_smoothed_occ_mask),
            "trim_sparse_top_row_for_analysis": bool(trim_sparse_top_row_for_analysis),
            "trim_sparse_top_row_for_plotting": bool(trim_sparse_top_row_for_plotting),
            "sparse_top_row_nonocc_frac_threshold": float(sparse_top_row_nonocc_frac_threshold),
            "top_row_nonocc_fraction_raw": top_row_nonocc_fraction_raw,
            "top_row_trimmed_analysis": bool(top_row_trimmed_analysis),
            "top_row_trimmed_plotting": bool(top_row_trimmed_plotting),
            "min_peak_rate": min_peak_rate,
            "min_pf_firing_traversals": (
                None
                if min_firing_traversals_required is None
                else int(min_firing_traversals_required)
            ),
            "pf_firing_traversal_filter_scope": "candidate_place_cells_only",
            "pf_firing_traversal_all_filter_applied": bool(apply_pf_component_filter),
            "pf_firing_traversal_detection": "distance_defined_iterative",
            "pf_firing_traversal_distance_window_cm": float(pf_firing_traversal_distance_window_cm),
            "pf_firing_traversal_detection_window_cm": float(pf_firing_traversal_detection_window_cm),
            "pf_firing_traversal_distance_bin_cm": float(pf_firing_traversal_distance_bin_cm),
            "pf_firing_traversal_distance_mode": str(pf_firing_traversal_distance_mode),
            "pf_firing_traversal_center_vicinity_min_cm": int(pf_firing_traversal_center_vicinity_min_cm),
            "pf_firing_traversal_center_vicinity_max_cm": int(pf_firing_traversal_center_vicinity_max_cm),
            "pf_firing_traversal_resting_speed_threshold": float(pf_firing_traversal_resting_speed_threshold),
            "pf_firing_traversal_merge_gap_s": float(pf_firing_traversal_merge_gap_s),
            "pf_firing_traversal_exclude_trials_with_bad_frames": bool(pf_firing_traversal_exclude_trials_with_bad_frames),
            "pf_reliability_dilation_bins": int(pf_reliability_dilation_bins),
            "pf_reliability_dilation_shape": pf_reliability_dilation_shape,
            "random_seed": random_seed,
            "save_spike_shapes": bool(save_spike_shapes),
            "ss_shape_pre_ms": float(ss_shape_pre_ms),
            "ss_shape_post_ms": float(ss_shape_post_ms),
            "cb_shape_pre_ms": float(cb_shape_pre_ms),
            "cb_shape_post_ms": float(cb_shape_post_ms),
            "ss_shape_min_separation_ms": None if ss_shape_min_separation_ms is None else float(ss_shape_min_separation_ms),
            "spike_shape_median_filter_size": int(spike_shape_median_filter_size),
            "spike_shape_baseline_frames": int(spike_shape_baseline_frames),
            "save_burst_metrics": bool(save_burst_metrics),
            "include_simple_bursts_metrics": bool(include_simple_bursts_metrics),
            "save_spike_burst_rate_metrics": bool(save_spike_burst_rate_metrics),
            "min_cb_bursts_per_condition": int(min_cb_bursts_per_condition),
        },
    }

    if axes is not None:
        plot_place_cell_single_moving_results(results, axes=axes)

    return results


def plot_place_cell_single_moving_results(results, axes):
    """
    Plot the same figure panels as `analyze_place_cell_single_moving`, using its returned results.

    Parameters
    ----------
    results : dict
        Output dict returned by `analyze_place_cell_single_moving`.
    axes : list[matplotlib.axes.Axes]
        Axes to draw on (same expected order/length as in `analyze_place_cell_single_moving`).
    """
    plot_data = results.get("plot_data") or {}
    plot_params = results.get("plot_params") or {}

    separate_spikes = bool(plot_params.get("separate_spikes", False))
    plot_sub = bool(plot_params.get("plot_sub", False))
    plot_pf_contours = bool(plot_params.get("plot_pf_contours", False))
    display_PF_SS_CS = bool(plot_params.get("display_PF_SS_CS", False))
    simple_spike_color = plot_params.get("simple_spike_color", "#026C80")
    complex_spike_color = plot_params.get("complex_spike_color", "#EE9B00")
    display_cell_num = plot_params.get("display_cell_num", None)
    is_last_column = bool(plot_params.get("is_last_column", False))
    is_first_column = bool(plot_params.get("is_first_column", False))
    width_real = float(plot_params.get("width_real", results.get("params", {}).get("width_real", 35.5)))
    height_real = float(plot_params.get("height_real", results.get("params", {}).get("height_real", 20.0)))

    extent = plot_data.get("extent", (0, width_real, 0, height_real))
    low_occ_mask = plot_data.get("low_occ_mask", None)
    x_all_valid = plot_data.get("x_all_valid", None)
    y_all_valid = plot_data.get("y_all_valid", None)
    cell_color = plot_data.get("cell_color", None)

    if low_occ_mask is None or x_all_valid is None or y_all_valid is None or cell_color is None:
        raise ValueError(
            "results is missing `plot_data` required for plotting; re-run "
            "`analyze_place_cell_single_moving` with the updated code."
        )

    expected_axes = 2
    if separate_spikes:
        expected_axes += 2
    if plot_sub:
        expected_axes += 2
    if len(axes) != expected_axes:
        raise ValueError(f"axes must contain exactly {expected_axes} axes")

    ax_idx = 0
    ax_traj = axes[ax_idx]
    ax_idx += 1
    ax_rate = axes[ax_idx]
    ax_idx += 1
    ax_ss = None
    ax_cs = None
    if separate_spikes:
        ax_ss = axes[ax_idx]
        ax_cs = axes[ax_idx + 1]
        ax_idx += 2
    if plot_sub:
        ax_theta = axes[ax_idx]
        ax_slow = axes[ax_idx + 1]

    smooth_map = results.get("rate_map", None)
    peak_rate = results.get("peak_rate", np.nan)
    p_val = results.get("p_value", np.nan)
    is_place_cell = bool(results.get("is_place_cell", False))
    place_field_mask = results.get("place_field_mask", None)
    occ_map = results.get("occupancy", None)

    if smooth_map is None or place_field_mask is None or occ_map is None:
        raise ValueError("results is missing required map fields for plotting.")

    def _plot_pf_contour(ax, mask=None, color="magenta"):
        pf_mask = mask if mask is not None else place_field_mask
        if not (plot_pf_contours and np.any(pf_mask)):
            return
        padded_mask = np.zeros((pf_mask.shape[0] + 2, pf_mask.shape[1] + 2), dtype=bool)
        padded_mask[1:-1, 1:-1] = pf_mask
        bin_x = (extent[1] - extent[0]) / pf_mask.shape[0]
        bin_y = (extent[3] - extent[2]) / pf_mask.shape[1]
        padded_extent = (
            extent[0] - bin_x,
            extent[1] + bin_x,
            extent[2] - bin_y,
            extent[3] + bin_y,
        )
        ax.contour(
            padded_mask.T,
            levels=[0.5],
            colors=color,
            linewidths=1.2,
            extent=padded_extent,
            origin="lower",
        )

    def _plot_pf_contour_ss_cs(ax, mask, color, is_pc=None):
        """Plot place field contour for SS/CS with custom mask and color.

        Contour is always plotted if the mask has any place field bins,
        regardless of place cell classification.
        """
        if not np.any(mask):
            return
        padded_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=bool)
        padded_mask[1:-1, 1:-1] = mask
        bin_x = (extent[1] - extent[0]) / mask.shape[0]
        bin_y = (extent[3] - extent[2]) / mask.shape[1]
        padded_extent = (
            extent[0] - bin_x,
            extent[1] + bin_x,
            extent[2] - bin_y,
            extent[3] + bin_y,
        )
        ax.contour(
            padded_mask.T,
            levels=[0.5],
            colors=color,
            linewidths=1.2,
            extent=padded_extent,
            origin="lower",
        )

    def _style_map_axis(ax):
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.add_patch(
            Rectangle(
                (extent[0], extent[2]),
                extent[1] - extent[0],
                extent[3] - extent[2],
                linewidth=1.0,
                edgecolor="black",
                facecolor="none",
                zorder=3,
            )
        )

    masked_map = ma.masked_where(np.isnan(smooth_map), smooth_map)
    cmap = plt.get_cmap("jet").copy()
    cmap.set_bad(color="white")
    im = ax_rate.imshow(
        masked_map.T,
        origin="lower",
        extent=extent,
        cmap=cmap,
        interpolation="nearest",
        vmin=0,
        vmax=peak_rate if np.isfinite(peak_rate) and peak_rate > 0 else None,
    )
    _plot_pf_contour(ax_rate)
    _style_map_axis(ax_rate)

    if p_val < 0.001:
        sig_mark = "***"
    elif p_val < 0.01:
        sig_mark = "**"
    elif p_val < 0.05:
        sig_mark = "*"
    else:
        sig_mark = ""
    max_rate_str = f"{peak_rate:.1f}" if np.isfinite(peak_rate) else "N/A"
    pc_star = "★" if is_place_cell else ""
    ax_rate.text(
        1.0,
        -0.05,
        f"{pc_star}{sig_mark} {max_rate_str} Hz"
        if (sig_mark or pc_star)
        else f"{max_rate_str} Hz",
        transform=ax_rate.transAxes,
        ha="right",
        va="top",
        fontsize=5,
        fontname="Arial",
    )

    if is_last_column:
        cax = inset_axes(
            ax_rate,
            width="5%",
            height="100%",
            loc="center right",
            bbox_to_anchor=(0.12, 0.0, 1, 1),
            bbox_transform=ax_rate.transAxes,
            borderpad=0,
        )
        cbar = ax_rate.figure.colorbar(im, cax=cax)
        cbar.set_ticks([0, im.get_clim()[1]])
        cbar.set_ticklabels(["0", "max"])
        cbar.ax.tick_params(labelsize=5)
        for tick in cbar.ax.get_yticklabels():
            tick.set_fontname("Arial")

    if separate_spikes:
        ss_map_use = results.get("ss_norm_map", None)
        cs_map_use = results.get("cs_norm_map", None)
        if ss_map_use is None:
            ss_map_use = np.full_like(smooth_map, np.nan)
        if cs_map_use is None:
            cs_map_use = np.full_like(smooth_map, np.nan)

        ss_p_val = results.get("ss_p_value", np.nan)
        cs_p_val = results.get("cs_p_value", np.nan)
        ss_peak_rate = results.get("ss_peak_rate", np.nan)
        cs_peak_rate = results.get("cs_peak_rate", np.nan)
        is_place_cell_ss = bool(results.get("is_place_cell_ss", False))
        is_place_cell_cs = bool(results.get("is_place_cell_cs", False))
        ss_place_field_mask = results.get("ss_place_field_mask", None)
        cs_place_field_mask = results.get("cs_place_field_mask", None)
        if ss_place_field_mask is None or cs_place_field_mask is None:
            raise ValueError("results is missing SS/CS place field masks for plotting.")

        ss_masked = ma.masked_where(np.isnan(ss_map_use), ss_map_use)
        im_ss = ax_ss.imshow(
            ss_masked.T,
            origin="lower",
            extent=extent,
            cmap=cmap,
            interpolation="nearest",
            vmin=0,
            vmax=1,
        )
        if not display_PF_SS_CS:
            _plot_pf_contour(ax_ss)
        _plot_pf_contour_ss_cs(ax_ss, ss_place_field_mask, simple_spike_color)
        _style_map_axis(ax_ss)

        if ss_p_val < 0.001:
            ss_sig_mark = "***"
        elif ss_p_val < 0.01:
            ss_sig_mark = "**"
        elif ss_p_val < 0.05:
            ss_sig_mark = "*"
        else:
            ss_sig_mark = ""
        ss_max_rate_str = f"{ss_peak_rate:.1f}" if np.isfinite(ss_peak_rate) else "N/A"
        ss_pc_star = "★" if is_place_cell_ss else ""
        ax_ss.text(
            1.0,
            -0.05,
            f"{ss_pc_star}{ss_sig_mark} {ss_max_rate_str} Hz"
            if (ss_sig_mark or ss_pc_star)
            else f"{ss_max_rate_str} Hz",
            transform=ax_ss.transAxes,
            ha="right",
            va="top",
            fontsize=5,
            fontname="Arial",
        )
        if is_last_column:
            cax_ss = inset_axes(
                ax_ss,
                width="5%",
                height="100%",
                loc="center right",
                bbox_to_anchor=(0.12, 0.0, 1, 1),
                bbox_transform=ax_ss.transAxes,
                borderpad=0,
            )
            cbar_ss = ax_ss.figure.colorbar(im_ss, cax=cax_ss)
            cbar_ss.ax.tick_params(labelsize=5)
            for tick in cbar_ss.ax.get_yticklabels():
                tick.set_fontname("Arial")

        cs_masked = ma.masked_where(np.isnan(cs_map_use), cs_map_use)
        im_cs = ax_cs.imshow(
            cs_masked.T,
            origin="lower",
            extent=extent,
            cmap=cmap,
            interpolation="nearest",
            vmin=0,
            vmax=1,
        )
        if not display_PF_SS_CS:
            _plot_pf_contour(ax_cs)
        _plot_pf_contour_ss_cs(
            ax_cs, cs_place_field_mask, complex_spike_color
        )
        _style_map_axis(ax_cs)

        if cs_p_val < 0.001:
            cs_sig_mark = "***"
        elif cs_p_val < 0.01:
            cs_sig_mark = "**"
        elif cs_p_val < 0.05:
            cs_sig_mark = "*"
        else:
            cs_sig_mark = ""
        cs_max_rate_str = f"{cs_peak_rate:.1f}" if np.isfinite(cs_peak_rate) else "N/A"
        cs_pc_star = "★" if is_place_cell_cs else ""
        ax_cs.text(
            1.0,
            -0.05,
            f"{cs_pc_star}{cs_sig_mark} {cs_max_rate_str} Hz"
            if (cs_sig_mark or cs_pc_star)
            else f"{cs_max_rate_str} Hz",
            transform=ax_cs.transAxes,
            ha="right",
            va="top",
            fontsize=5,
            fontname="Arial",
        )
        if is_last_column:
            cax_cs = inset_axes(
                ax_cs,
                width="5%",
                height="100%",
                loc="center right",
                bbox_to_anchor=(0.12, 0.0, 1, 1),
                bbox_transform=ax_cs.transAxes,
                borderpad=0,
            )
            cbar_cs = ax_cs.figure.colorbar(im_cs, cax=cax_cs)
            cbar_cs.ax.tick_params(labelsize=5)
            for tick in cbar_cs.ax.get_yticklabels():
                tick.set_fontname("Arial")

    if plot_sub:
        theta_map = results.get("theta_map", None)
        slow_map = results.get("slow_map", None)
        if theta_map is None or slow_map is None:
            raise ValueError(
                "theta/slow maps are missing; re-run `analyze_place_cell_single_moving` "
                "with `traces` (and `plot_sub=True`) to populate them."
            )

        theta_corr_all = results.get("theta_corr_all", np.nan)
        theta_corr_ss = results.get("theta_corr_ss", np.nan)
        theta_corr_cs = results.get("theta_corr_cs", np.nan)
        slow_corr_all = results.get("slow_corr_all", np.nan)
        slow_corr_ss = results.get("slow_corr_ss", np.nan)
        slow_corr_cs = results.get("slow_corr_cs", np.nan)

        if np.any(np.isfinite(theta_map)):
            theta_vmin = np.nanmin(theta_map)
            theta_vmax = np.nanmax(theta_map)
        else:
            theta_vmin = np.nan
            theta_vmax = np.nan

        if np.any(np.isfinite(slow_map)):
            slow_vabs = np.nanmax(np.abs(slow_map))
            if np.isfinite(slow_vabs) and slow_vabs > 0:
                slow_vmin = -slow_vabs
                slow_vmax = slow_vabs
            else:
                slow_vmin = np.nanmin(slow_map)
                slow_vmax = np.nanmax(slow_map)
        else:
            slow_vmin = np.nan
            slow_vmax = np.nan

        theta_cmap = plt.get_cmap("jet").copy()
        theta_cmap.set_bad(color="white")
        theta_masked = ma.masked_where(np.isnan(theta_map), theta_map)
        im_theta = ax_theta.imshow(
            theta_masked.T,
            origin="lower",
            extent=extent,
            cmap=theta_cmap,
            interpolation="nearest",
            vmin=theta_vmin if np.isfinite(theta_vmin) else None,
            vmax=theta_vmax if np.isfinite(theta_vmax) else None,
        )
        _plot_pf_contour(ax_theta)
        _style_map_axis(ax_theta)
        corr_strs = []
        if np.isfinite(theta_corr_all):
            corr_strs.append((f"{theta_corr_all:.2f}", "black"))
        if separate_spikes:
            if np.isfinite(theta_corr_ss):
                corr_strs.append((f"{theta_corr_ss:.2f}", simple_spike_color))
            if np.isfinite(theta_corr_cs):
                corr_strs.append((f"{theta_corr_cs:.2f}", complex_spike_color))
        x_positions = (
            [0.25, 0.5, 0.75]
            if len(corr_strs) == 3
            else ([0.33, 0.67] if len(corr_strs) == 2 else [0.5])
        )
        for i, (corr_str, color) in enumerate(corr_strs):
            ax_theta.text(
                x_positions[i],
                -0.05,
                corr_str,
                transform=ax_theta.transAxes,
                ha="center",
                va="top",
                fontsize=5,
                fontname="Arial",
                color=color,
            )
        if is_last_column:
            cax_theta = inset_axes(
                ax_theta,
                width="5%",
                height="100%",
                loc="center right",
                bbox_to_anchor=(0.12, 0.0, 1, 1),
                bbox_transform=ax_theta.transAxes,
                borderpad=0,
            )
            cbar_theta = ax_theta.figure.colorbar(im_theta, cax=cax_theta)
            cbar_theta.ax.tick_params(labelsize=5)
            for tick in cbar_theta.ax.get_yticklabels():
                tick.set_fontname("Arial")

        slow_cmap = plt.get_cmap("bwr").copy()
        slow_cmap.set_bad(color="white")
        slow_masked = ma.masked_where(np.isnan(slow_map), slow_map)
        im_slow = ax_slow.imshow(
            slow_masked.T,
            origin="lower",
            extent=extent,
            cmap=slow_cmap,
            interpolation="nearest",
            vmin=slow_vmin if np.isfinite(slow_vmin) else None,
            vmax=slow_vmax if np.isfinite(slow_vmax) else None,
        )
        _plot_pf_contour(ax_slow)
        _style_map_axis(ax_slow)
        slow_corr_strs = []
        if np.isfinite(slow_corr_all):
            slow_corr_strs.append((f"{slow_corr_all:.2f}", "black"))
        if separate_spikes:
            if np.isfinite(slow_corr_ss):
                slow_corr_strs.append((f"{slow_corr_ss:.2f}", simple_spike_color))
            if np.isfinite(slow_corr_cs):
                slow_corr_strs.append((f"{slow_corr_cs:.2f}", complex_spike_color))
        slow_x_positions = (
            [0.25, 0.5, 0.75]
            if len(slow_corr_strs) == 3
            else ([0.33, 0.67] if len(slow_corr_strs) == 2 else [0.5])
        )
        for i, (corr_str, color) in enumerate(slow_corr_strs):
            ax_slow.text(
                slow_x_positions[i],
                -0.05,
                corr_str,
                transform=ax_slow.transAxes,
                ha="center",
                va="top",
                fontsize=5,
                fontname="Arial",
                color=color,
            )
        if is_last_column:
            cax_slow = inset_axes(
                ax_slow,
                width="5%",
                height="100%",
                loc="center right",
                bbox_to_anchor=(0.12, 0.0, 1, 1),
                bbox_transform=ax_slow.transAxes,
                borderpad=0,
            )
            cbar_slow = ax_slow.figure.colorbar(im_slow, cax=cax_slow)
            cbar_slow.ax.tick_params(labelsize=5)
            for tick in cbar_slow.ax.get_yticklabels():
                tick.set_fontname("Arial")

    occ_background = np.ones_like(occ_map, dtype=float)
    occ_background[low_occ_mask] = np.nan
    bg_cmap = plt.get_cmap("Greys").copy()
    bg_cmap.set_bad(color="white")
    ax_traj.imshow(
        occ_background.T,
        origin="lower",
        extent=extent,
        cmap=bg_cmap,
        interpolation="nearest",
        vmin=0,
        vmax=1,
        alpha=0.3,
    )

    ax_traj.plot(x_all_valid, y_all_valid, color="gray", linewidth=0.6, alpha=0.6)
    if separate_spikes:
        ss_x = plot_data.get("ss_spikes_x", np.array([]))
        ss_y = plot_data.get("ss_spikes_y", np.array([]))
        cs_x = plot_data.get("cs_spikes_x", np.array([]))
        cs_y = plot_data.get("cs_spikes_y", np.array([]))
        ax_traj.scatter(
            ss_x,
            ss_y,
            s=2,
            color=simple_spike_color,
            alpha=0.6,
            linewidths=0,
        )
        ax_traj.scatter(
            cs_x,
            cs_y,
            s=2,
            color=complex_spike_color,
            alpha=0.6,
            linewidths=0,
        )
        if is_last_column:
            legend_handles = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="none",
                    markerfacecolor=simple_spike_color,
                    markeredgecolor="none",
                    markersize=4,
                    label="SS",
                ),
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="none",
                    markerfacecolor=complex_spike_color,
                    markeredgecolor="none",
                    markersize=4,
                    label="CS",
                ),
            ]
            ax_traj.legend(
                handles=legend_handles,
                loc="upper left",
                bbox_to_anchor=(1.02, 1.0),
                frameon=False,
                fontsize=6,
                handletextpad=0.05,
                borderaxespad=0,
            )
    else:
        x_spk = results.get("spikes_x", np.array([]))
        y_spk = results.get("spikes_y", np.array([]))
        ax_traj.scatter(
            x_spk,
            y_spk,
            s=2,
            color=cell_color,
            alpha=0.5,
            linewidths=0,
        )

    ax_rate.set_box_aspect(height_real / width_real)
    ax_traj.set_box_aspect(height_real / width_real)
    ax_traj.set_aspect("equal", adjustable="box")
    ax_traj.set_xlim(extent[0], extent[1])
    ax_traj.set_ylim(extent[2], extent[3])
    ax_traj.axis("off")
    ax_traj.add_patch(
        Rectangle(
            (extent[0], extent[2]),
            extent[1] - extent[0],
            extent[3] - extent[2],
            linewidth=1.0,
            edgecolor="black",
            facecolor="none",
            zorder=3,
        )
    )

    title_num = display_cell_num if display_cell_num is not None else (results.get("cell_id", 0) + 1)
    ax_traj.set_title(f"cell {title_num}", fontsize=6, fontname="Arial")

    if is_first_column:
        scale_len_cm = 10
        x0 = 0.95 - (scale_len_cm / width_real) * 0.9
        x1 = 0.95
        y0 = -0.08
        ax_traj.plot(
            [x0, x1],
            [y0, y0],
            transform=ax_traj.transAxes,
            color="black",
            linewidth=1.0,
            clip_on=False,
        )
        ax_traj.text(
            (x0 + x1) / 2,
            y0 - 0.04,
            "10 cm",
            transform=ax_traj.transAxes,
            ha="center",
            va="top",
            fontsize=6,
            color="black",
        )


def _compute_hd_info_bits_per_spike(
    spike_counts,
    occ_time_s,
    visited_mask=None,
    prior_alpha=0.0,
    prior_beta_s=0.0,
):
    """Compute directional information (bits/spike) from angle-binned spike/occupancy data."""
    spk = np.asarray(spike_counts, dtype=float).ravel()
    occ = np.asarray(occ_time_s, dtype=float).ravel()
    if spk.size == 0 or occ.size == 0 or spk.size != occ.size:
        return np.nan

    spk = np.clip(np.nan_to_num(spk, nan=0.0), 0.0, None)
    occ = np.clip(np.nan_to_num(occ, nan=0.0), 0.0, None)

    if visited_mask is None:
        keep = occ > 0
    else:
        keep = np.asarray(visited_mask, dtype=bool).ravel()
        if keep.size != occ.size:
            keep = occ > 0
        else:
            keep = keep & (occ > 0)
    if not np.any(keep):
        return np.nan

    spk_k = spk[keep] + float(prior_alpha)
    occ_k = occ[keep] + float(prior_beta_s)
    valid = np.isfinite(spk_k) & np.isfinite(occ_k) & (occ_k > 0)
    if not np.any(valid):
        return np.nan
    spk_k = spk_k[valid]
    occ_k = occ_k[valid]

    occ_sum = float(np.sum(occ_k))
    if occ_sum <= 0:
        return np.nan
    p_k = occ_k / occ_sum

    rate_k = np.divide(
        spk_k,
        occ_k,
        out=np.zeros_like(spk_k, dtype=float),
        where=occ_k > 0,
    )
    mean_rate = float(np.sum(p_k * rate_k))
    if not np.isfinite(mean_rate) or mean_rate <= 0:
        return 0.0

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.divide(rate_k, mean_rate)
    valid_ratio = np.isfinite(ratio) & (ratio > 0)
    if not np.any(valid_ratio):
        return 0.0

    info = np.sum(
        p_k[valid_ratio] * ratio[valid_ratio] * np.log2(ratio[valid_ratio])
    )
    return float(info) if np.isfinite(info) else np.nan


def _run_hd_multinomial_null(
    observed_info_bits,
    n_spikes_total,
    occ_counts_prior,
    occ_time_for_rate,
    visited_mask,
    n_shuffles=1000,
    prior_alpha=0.0,
    prior_beta_s=0.0,
    smooth_sigma=0.0,
    rng=None,
):
    """Run occupancy-prior multinomial null and return mean null info + p-value."""
    n_spikes_total = int(n_spikes_total)
    if n_spikes_total <= 0:
        return np.nan, np.nan, np.array([], dtype=float)

    occ_prior = np.asarray(occ_counts_prior, dtype=float).ravel()
    occ_time = np.asarray(occ_time_for_rate, dtype=float).ravel()
    if occ_prior.size == 0 or occ_prior.size != occ_time.size:
        return np.nan, np.nan, np.array([], dtype=float)

    vis = np.asarray(visited_mask, dtype=bool).ravel()
    if vis.size != occ_prior.size:
        vis = occ_prior > 0
    prior_mask = vis & np.isfinite(occ_prior) & (occ_prior > 0)
    if not np.any(prior_mask):
        return np.nan, np.nan, np.array([], dtype=float)

    p = np.zeros_like(occ_prior, dtype=float)
    denom = float(np.sum(occ_prior[prior_mask]))
    if denom <= 0:
        return np.nan, np.nan, np.array([], dtype=float)
    p[prior_mask] = occ_prior[prior_mask] / denom

    n_shuffles = int(n_shuffles)
    if n_shuffles < 1:
        return np.nan, np.nan, np.array([], dtype=float)

    if rng is None:
        rng = np.random.default_rng()

    null_vals = np.full(n_shuffles, np.nan, dtype=float)
    for b in range(n_shuffles):
        shuf_counts = rng.multinomial(n_spikes_total, p)
        shuf_counts_rate = np.asarray(shuf_counts, dtype=float)
        if smooth_sigma is not None and float(smooth_sigma) > 0:
            shuf_counts_rate = gaussian_filter1d(
                shuf_counts_rate,
                sigma=float(smooth_sigma),
                mode="wrap",
            )
        null_vals[b] = _compute_hd_info_bits_per_spike(
            shuf_counts_rate,
            occ_time,
            visited_mask=vis,
            prior_alpha=prior_alpha,
            prior_beta_s=prior_beta_s,
        )

    null_vals = null_vals[np.isfinite(null_vals)]
    if null_vals.size == 0:
        return np.nan, np.nan, np.array([], dtype=float)

    null_mean = float(np.mean(null_vals))
    if np.isfinite(observed_info_bits):
        p_val = float(
            (1.0 + np.sum(null_vals >= float(observed_info_bits)))
            / (float(null_vals.size) + 1.0)
        )
    else:
        p_val = np.nan
    return null_mean, p_val, null_vals


def _normalize_session_bounds_frames(session_start_frames, n_frames):
    """Return sorted non-empty session bounds covering the recording."""
    n_frames = int(n_frames)
    if n_frames <= 0:
        return []

    if session_start_frames is None:
        starts = [0]
    else:
        starts = []
        for val in np.asarray(session_start_frames).ravel():
            try:
                sval = int(val)
            except Exception:
                continue
            if 0 <= sval < n_frames:
                starts.append(sval)
        starts = sorted(set(starts))
        if len(starts) == 0 or starts[0] != 0:
            starts = [0] + starts

    ends = starts[1:] + [n_frames]
    bounds = []
    for start, end in zip(starts, ends):
        start_i = int(start)
        end_i = int(end)
        if end_i > start_i:
            bounds.append((start_i, end_i))
    return bounds


def _sample_circular_shift_lag(n_frames, min_shift_frames, rng):
    """Sample a non-zero circular lag, honoring a minimum shift when feasible."""
    n_frames = int(n_frames)
    if n_frames <= 1:
        return 0

    min_shift_frames = max(0, int(min_shift_frames))
    lo = max(1, min_shift_frames)
    hi = min(n_frames - 1, n_frames - min_shift_frames)
    if lo <= hi:
        return int(rng.integers(lo, hi + 1))
    return int(rng.integers(1, n_frames))


def _circular_time_shift_spike_frames(
    spike_frames,
    n_frames,
    session_bounds=None,
    min_shift_frames=0,
    rng=None,
    sessionwise=True,
    allowed_frame_mask=None,
):
    """Circularly shift spikes within each session or across the full recording."""
    n_frames = int(n_frames)
    spike_frames = np.asarray(spike_frames, dtype=int).ravel()
    spike_frames = spike_frames[(spike_frames >= 0) & (spike_frames < n_frames)]
    if spike_frames.size == 0 or n_frames <= 1:
        return spike_frames.copy()

    if rng is None:
        rng = np.random.default_rng()

    if allowed_frame_mask is not None:
        allowed_frame_mask = np.asarray(allowed_frame_mask, dtype=bool).ravel()
        if allowed_frame_mask.size != n_frames:
            raise ValueError("allowed_frame_mask must match n_frames.")
        spike_frames = spike_frames[allowed_frame_mask[spike_frames]]
        if spike_frames.size == 0:
            return spike_frames.copy()

    if sessionwise:
        bounds = session_bounds
        if bounds is None:
            bounds = _normalize_session_bounds_frames(None, n_frames)
    else:
        bounds = [(0, n_frames)]

    shifted = spike_frames.copy()
    for start, end in bounds:
        start_i = int(start)
        end_i = int(end)
        sess_len = int(end_i - start_i)
        if sess_len <= 1:
            continue
        in_sess = (spike_frames >= start_i) & (spike_frames < end_i)
        if not np.any(in_sess):
            continue

        if allowed_frame_mask is None:
            lag = _sample_circular_shift_lag(
                sess_len,
                min_shift_frames=min_shift_frames,
                rng=rng,
            )
            shifted[in_sess] = start_i + (
                (spike_frames[in_sess] - start_i + lag) % sess_len
            )
            continue

        allowed_idx = np.flatnonzero(allowed_frame_mask[start_i:end_i]) + start_i
        if allowed_idx.size <= 1:
            continue

        sess_spikes = spike_frames[in_sess]
        spike_pos = np.searchsorted(allowed_idx, sess_spikes)
        pos_valid = (
            (spike_pos >= 0)
            & (spike_pos < allowed_idx.size)
            & (allowed_idx[spike_pos] == sess_spikes)
        )
        if not np.any(pos_valid):
            continue
        lag = _sample_circular_shift_lag(
            allowed_idx.size,
            min_shift_frames=min_shift_frames,
            rng=rng,
        )
        shifted_vals = allowed_idx[(spike_pos[pos_valid] + lag) % allowed_idx.size]
        shifted_idx = np.where(in_sess)[0][pos_valid]
        shifted[shifted_idx] = shifted_vals

    return shifted


def _p_to_star_count(p_val, thresholds=(0.05, 0.01, 0.001)):
    """Map p-value to star count: *, **, ***."""
    if not np.isfinite(p_val):
        return 0
    thr = list(thresholds) if thresholds is not None else [0.05, 0.01, 0.001]
    count = 0
    for t in thr:
        try:
            tt = float(t)
        except Exception:
            continue
        if p_val < tt:
            count += 1
    return int(count)


def _star_text_from_count(star_count):
    count = int(star_count)
    if count <= 0:
        return ""
    return "*" * min(count, 3)


def analyze_hd_vector_field_single_moving(
    x_neural,
    y_neural,
    hd_angles_neural,
    spikes,
    speed,
    frame_rate,
    cell_idx,
    axes=None,
    width_real=35.5,
    height_real=20,
    bin_size=1.5,
    kernel_size=21,
    filter_type="boxcar",
    speed_threshold=2,
    min_duration_s=0.5,
    merge_gap_s=1.0,
    bad_timepoints=None,
    refined_SS=None,
    all_CS_spikes=None,
    min_spikes_per_bin=20,
    min_visited_angle_bins=2,
    min_visited_angle_separation_deg=0.0,
    min_occ_frames_per_angle=2,
    hd_count_smooth_sigma=1.0,
    min_pref_vector_strength=0.0,
    preferred_angle_mode="peak",
    pref_angle_std_threshold=None,
    hd_bin_size_deg=30,
    direction_mode="head",
    travel_smooth_window=5,
    travel_min_step=0.0,
    display_cell_num=None,
    is_first_column=False,
    is_last_column=False,
    quiver_scale=None,
    cmap="viridis",
    use_normalized_rate=False,
    enable_hd_info_shuffle_gate=False,
    hd_info_n_shuffles=1000,
    hd_info_alpha=0.05,
    hd_info_gate_rule="p_and_above_null_mean",
    hd_info_random_seed=None,
    hd_info_star_thresholds=(0.05, 0.01, 0.001),
    hd_info_prior_alpha=0.0,
    hd_info_prior_beta_s=0.0,
    hd_info_use_smoothed_counts=True,
    hd_info_null_mode="timeshift",
    hd_time_shift_min_s=10.0,
    hd_time_shift_sessionwise=True,
    session_start_frames=None,
):
    """Analyze spatial head-direction vector fields for one cell during locomotion."""
    x_neural = np.asarray(x_neural, dtype=float)
    y_neural = np.asarray(y_neural, dtype=float)
    speed = np.asarray(speed, dtype=float)
    hd_angles_neural = np.asarray(hd_angles_neural, dtype=float)

    n_frames = len(x_neural)
    if (
        len(y_neural) != n_frames
        or len(speed) != n_frames
        or len(hd_angles_neural) != n_frames
    ):
        raise ValueError(
            "x_neural, y_neural, speed, and hd_angles_neural must have matching lengths."
        )

    direction_mode = str(direction_mode).lower()
    if direction_mode not in {"head", "travel"}:
        raise ValueError("direction_mode must be 'head' or 'travel'.")
    session_bounds = _normalize_session_bounds_frames(session_start_frames, n_frames)

    def _compute_travel_direction_deg(x_vals, y_vals, smooth_window=5, min_step=0.0):
        x_vals = np.asarray(x_vals, dtype=float)
        y_vals = np.asarray(y_vals, dtype=float)
        n = len(x_vals)
        out = np.full(n, np.nan, dtype=float)
        finite = np.isfinite(x_vals) & np.isfinite(y_vals)
        if np.sum(finite) < 2:
            return out
        idx = np.arange(n)
        valid_idx = idx[finite]
        x_interp = np.interp(idx, valid_idx, x_vals[finite])
        y_interp = np.interp(idx, valid_idx, y_vals[finite])
        dx = np.gradient(x_interp)
        dy = np.gradient(y_interp)
        sw = int(max(1, smooth_window))
        if sw > 1:
            dx = uniform_filter1d(dx, size=sw, mode="nearest")
            dy = uniform_filter1d(dy, size=sw, mode="nearest")
        step = np.hypot(dx, dy)
        ang = np.rad2deg(np.arctan2(dy, dx)) % 360.0
        ang[step <= float(min_step)] = np.nan
        ang[~finite] = np.nan
        out[:] = ang
        return out

    if direction_mode == "travel":
        direction_angles = _compute_travel_direction_deg(
            x_neural,
            y_neural,
            smooth_window=travel_smooth_window,
            min_step=travel_min_step,
        )
    else:
        direction_angles = np.mod(hd_angles_neural, 360.0)

    valid_frames = (
        np.isfinite(x_neural)
        & np.isfinite(y_neural)
        & np.isfinite(speed)
        & np.isfinite(direction_angles)
    )

    if bad_timepoints is not None:
        bad_mask = np.asarray(bad_timepoints)
        if bad_mask.dtype != bool:
            if bad_mask.ndim == 1 and bad_mask.size == n_frames:
                bad_mask = bad_mask.astype(bool)
            else:
                bad_idx = np.asarray(bad_mask, dtype=int)
                bad_idx = bad_idx[(bad_idx >= 0) & (bad_idx < n_frames)]
                bad_bool = np.zeros(n_frames, dtype=bool)
                bad_bool[bad_idx] = True
                bad_mask = bad_bool
        if bad_mask.shape[0] != n_frames:
            raise ValueError(
                "bad_timepoints must match x_neural length or be an index list."
            )
        valid_frames &= ~bad_mask

    speed_for_epochs = speed.copy()
    speed_for_epochs[~valid_frames] = np.nan
    _, moving_epochs, moving_idx = _compute_moving_epochs(
        speed_for_epochs,
        frame_rate,
        kernel_size=kernel_size,
        filter_type=filter_type,
        speed_threshold=speed_threshold,
        min_duration_s=min_duration_s,
        merge_gap_s=merge_gap_s,
    )

    moving_idx = moving_idx[valid_frames[moving_idx]]
    moving_mask = np.zeros(n_frames, dtype=bool)
    moving_mask[moving_idx] = True

    bins = [
        np.arange(0, width_real + bin_size, bin_size),
        np.arange(0, height_real + bin_size, bin_size),
    ]
    extent = (0, width_real, 0, height_real)
    nx = len(bins[0]) - 1
    ny = len(bins[1]) - 1
    x_centers = 0.5 * (bins[0][:-1] + bins[0][1:])
    y_centers = 0.5 * (bins[1][:-1] + bins[1][1:])

    x_traj = x_neural[moving_idx]
    y_traj = y_neural[moving_idx]
    dir_traj = np.mod(direction_angles[moving_idx], 360.0)
    x_bin_traj = np.digitize(x_traj, bins[0]) - 1
    y_bin_traj = np.digitize(y_traj, bins[1]) - 1
    x_bin_traj = np.clip(x_bin_traj, 0, nx - 1)
    y_bin_traj = np.clip(y_bin_traj, 0, ny - 1)

    hd_bin_size_deg = float(hd_bin_size_deg)
    if hd_bin_size_deg <= 0:
        raise ValueError("hd_bin_size_deg must be > 0.")
    min_visited_angle_bins = int(min_visited_angle_bins)
    if min_visited_angle_bins < 1:
        raise ValueError("min_visited_angle_bins must be >= 1.")
    min_visited_angle_separation_deg = float(min_visited_angle_separation_deg)
    if min_visited_angle_separation_deg < 0 or min_visited_angle_separation_deg > 180:
        raise ValueError("min_visited_angle_separation_deg must be in [0, 180].")
    min_occ_frames_per_angle = int(min_occ_frames_per_angle)
    if min_occ_frames_per_angle < 1:
        raise ValueError("min_occ_frames_per_angle must be >= 1.")
    hd_count_smooth_sigma = float(hd_count_smooth_sigma)
    if hd_count_smooth_sigma < 0:
        raise ValueError("hd_count_smooth_sigma must be >= 0.")
    min_pref_vector_strength = float(min_pref_vector_strength)
    if min_pref_vector_strength < 0 or min_pref_vector_strength > 1:
        raise ValueError("min_pref_vector_strength must be in [0, 1].")
    preferred_angle_mode = str(preferred_angle_mode).lower()
    if preferred_angle_mode not in {"peak", "circular_mean", "peak_cluster_mean"}:
        raise ValueError(
            "preferred_angle_mode must be 'peak', 'circular_mean', or "
            "'peak_cluster_mean'."
        )
    enable_hd_info_shuffle_gate = bool(enable_hd_info_shuffle_gate)
    hd_info_n_shuffles = int(hd_info_n_shuffles)
    if hd_info_n_shuffles < 1:
        raise ValueError("hd_info_n_shuffles must be >= 1.")
    hd_info_alpha = float(hd_info_alpha)
    if hd_info_alpha <= 0 or hd_info_alpha >= 1:
        raise ValueError("hd_info_alpha must be in (0, 1).")
    hd_info_gate_rule = str(hd_info_gate_rule).lower()
    if hd_info_gate_rule != "p_and_above_null_mean":
        raise ValueError(
            "hd_info_gate_rule must be 'p_and_above_null_mean'."
        )
    hd_info_prior_alpha = float(hd_info_prior_alpha)
    if hd_info_prior_alpha < 0:
        raise ValueError("hd_info_prior_alpha must be >= 0.")
    hd_info_prior_beta_s = float(hd_info_prior_beta_s)
    if hd_info_prior_beta_s < 0:
        raise ValueError("hd_info_prior_beta_s must be >= 0.")
    hd_info_use_smoothed_counts = bool(hd_info_use_smoothed_counts)
    hd_info_null_mode = str(hd_info_null_mode).strip().lower()
    if hd_info_null_mode not in {"timeshift", "multinomial"}:
        raise ValueError("hd_info_null_mode must be 'timeshift' or 'multinomial'.")
    hd_time_shift_min_s = float(hd_time_shift_min_s)
    if hd_time_shift_min_s < 0:
        raise ValueError("hd_time_shift_min_s must be >= 0.")
    hd_time_shift_sessionwise = bool(hd_time_shift_sessionwise)
    hd_info_star_thresholds = tuple(hd_info_star_thresholds)
    if len(hd_info_star_thresholds) < 1:
        raise ValueError("hd_info_star_thresholds must contain at least one threshold.")
    hd_time_shift_min_frames = max(
        0,
        int(round(float(frame_rate) * float(hd_time_shift_min_s))),
    )

    rng_base = (
        np.random.default_rng(int(hd_info_random_seed))
        if hd_info_random_seed is not None
        else np.random.default_rng()
    )
    n_hd_bins_float = 360.0 / hd_bin_size_deg
    if not np.isclose(n_hd_bins_float, round(n_hd_bins_float), atol=1e-9):
        raise ValueError(
            "hd_bin_size_deg must evenly divide 360 for circular binning."
        )
    if not np.isclose(90.0 / hd_bin_size_deg, round(90.0 / hd_bin_size_deg), atol=1e-9):
        raise ValueError(
            "hd_bin_size_deg must evenly divide 90 so 0/90/180/270 are HD bin centers."
        )
    n_hd_bins = int(round(n_hd_bins_float))
    if n_hd_bins < 1:
        raise ValueError("Invalid HD binning; choose hd_bin_size_deg <= 360.")
    hd_bin_edges = np.linspace(0.0, 360.0, n_hd_bins + 1)
    # Center bins at 0 deg (then +hd_bin_size_deg increments).
    hd_bin_centers_deg = (np.arange(n_hd_bins, dtype=float) * hd_bin_size_deg) % 360.0
    hd_bin_centers_rad = np.deg2rad(hd_bin_centers_deg)
    hd_hist_shift_deg = 0.5 * hd_bin_size_deg

    def _hist_hd(angles_deg):
        angles_deg = np.asarray(angles_deg, dtype=float)
        if angles_deg.size == 0:
            return np.zeros(n_hd_bins, dtype=int)
        angles_shifted = (angles_deg + hd_hist_shift_deg) % 360.0
        counts, _ = np.histogram(angles_shifted, bins=hd_bin_edges)
        return counts.astype(int)

    def _sanitize_spike_times(spike_times, apply_analysis_mask=True):
        if spike_times is None:
            return np.array([], dtype=int)
        spike_times = np.asarray(spike_times, dtype=int)
        spike_times = spike_times[(spike_times >= 0) & (spike_times < n_frames)]
        if spike_times.size == 0 or not apply_analysis_mask:
            return spike_times
        return spike_times[valid_frames[spike_times] & moving_mask[spike_times]]

    def _find_circular_true_clusters(mask_1d):
        """Find contiguous True clusters on a circular 1D boolean array."""
        mask_1d = np.asarray(mask_1d, dtype=bool).ravel()
        n = mask_1d.size
        if n == 0:
            return []
        true_idx = np.where(mask_1d)[0]
        if true_idx.size == 0:
            return []

        clusters = [[int(true_idx[0])]]
        for idx in true_idx[1:]:
            idx = int(idx)
            if idx == clusters[-1][-1] + 1:
                clusters[-1].append(idx)
            else:
                clusters.append([idx])

        # Merge wrap-around clusters (end and start touching on circular axis).
        if len(clusters) > 1 and clusters[0][0] == 0 and clusters[-1][-1] == (n - 1):
            merged = clusters[-1] + clusters[0]
            clusters = [merged] + clusters[1:-1]

        return [np.asarray(c, dtype=int) for c in clusters]

    def _smooth_hd_counts_if_needed(counts_arr):
        counts_arr = np.asarray(counts_arr, dtype=float)
        if hd_count_smooth_sigma > 0:
            return gaussian_filter1d(
                counts_arr,
                sigma=hd_count_smooth_sigma,
                axis=-1,
                mode="wrap",
            )
        return counts_arr.copy()

    analysis_frame_mask = valid_frames & moving_mask

    def _compute_shifted_spike_hd_maps(spike_times_for_shift, rng_local):
        shifted_spike_times = _circular_time_shift_spike_frames(
            spike_times_for_shift,
            n_frames=n_frames,
            session_bounds=session_bounds,
            min_shift_frames=hd_time_shift_min_frames,
            rng=rng_local,
            sessionwise=hd_time_shift_sessionwise,
            allowed_frame_mask=analysis_frame_mask,
        )
        shifted_spike_times = _sanitize_spike_times(
            shifted_spike_times,
            apply_analysis_mask=True,
        )
        spike_count_map = np.zeros((nx, ny), dtype=int)
        spike_counts_map = np.zeros((nx, ny, n_hd_bins), dtype=int)
        if shifted_spike_times.size == 0:
            return shifted_spike_times, spike_count_map, spike_counts_map

        x_spk_shift = x_neural[shifted_spike_times]
        y_spk_shift = y_neural[shifted_spike_times]
        dir_spk_shift = np.mod(direction_angles[shifted_spike_times], 360.0)
        x_bin_shift = np.digitize(x_spk_shift, bins[0]) - 1
        y_bin_shift = np.digitize(y_spk_shift, bins[1]) - 1
        x_bin_shift = np.clip(x_bin_shift, 0, nx - 1)
        y_bin_shift = np.clip(y_bin_shift, 0, ny - 1)
        hd_bin_shift = np.digitize(
            (dir_spk_shift + hd_hist_shift_deg) % 360.0,
            hd_bin_edges,
        ) - 1
        hd_bin_shift = np.mod(hd_bin_shift, n_hd_bins)

        np.add.at(spike_count_map, (x_bin_shift, y_bin_shift), 1)
        np.add.at(spike_counts_map, (x_bin_shift, y_bin_shift, hd_bin_shift), 1)
        return shifted_spike_times, spike_count_map, spike_counts_map

    def _run_time_shift_hd_null(
        spike_times_for_shift,
        spike_count_valid_mask,
        visited_mask_map,
        occ_time_for_info_map,
        hd_info_obs_map,
        global_visited_mask,
        global_occ_time_for_info,
        global_hd_info_obs,
        rng_local,
    ):
        eligible_mask = np.asarray(spike_count_valid_mask, dtype=bool) & np.isfinite(
            hd_info_obs_map
        )
        null_sum_map = np.zeros((nx, ny), dtype=float)
        null_valid_count_map = np.zeros((nx, ny), dtype=int)
        null_ge_count_map = np.zeros((nx, ny), dtype=int)
        global_null_vals = []

        for _ in range(hd_info_n_shuffles):
            _, shift_count_map, shift_spike_counts_map = _compute_shifted_spike_hd_maps(
                spike_times_for_shift,
                rng_local=rng_local,
            )

            if hd_info_use_smoothed_counts:
                shift_spk_for_info = _smooth_hd_counts_if_needed(shift_spike_counts_map)
            else:
                shift_spk_for_info = np.asarray(shift_spike_counts_map, dtype=float)

            eligible_idx = np.argwhere(eligible_mask)
            for ix, iy in eligible_idx:
                null_info = _compute_hd_info_bits_per_spike(
                    shift_spk_for_info[ix, iy, :],
                    occ_time_for_info_map[ix, iy, :],
                    visited_mask=visited_mask_map[ix, iy, :],
                    prior_alpha=hd_info_prior_alpha,
                    prior_beta_s=hd_info_prior_beta_s,
                )
                if not np.isfinite(null_info):
                    continue
                null_sum_map[ix, iy] += float(null_info)
                null_valid_count_map[ix, iy] += 1
                if null_info >= float(hd_info_obs_map[ix, iy]):
                    null_ge_count_map[ix, iy] += 1

            global_shift_spk_counts = np.nansum(shift_spike_counts_map, axis=(0, 1))
            if hd_info_use_smoothed_counts:
                global_shift_spk_for_info = _smooth_hd_counts_if_needed(
                    global_shift_spk_counts
                )
            else:
                global_shift_spk_for_info = np.asarray(
                    global_shift_spk_counts,
                    dtype=float,
                )
            global_null_info = _compute_hd_info_bits_per_spike(
                global_shift_spk_for_info,
                global_occ_time_for_info,
                visited_mask=global_visited_mask,
                prior_alpha=hd_info_prior_alpha,
                prior_beta_s=hd_info_prior_beta_s,
            )
            if np.isfinite(global_null_info):
                global_null_vals.append(float(global_null_info))

        null_mean_map = np.full((nx, ny), np.nan, dtype=float)
        valid_map = null_valid_count_map > 0
        null_mean_map[valid_map] = (
            null_sum_map[valid_map] / null_valid_count_map[valid_map]
        )

        p_val_map = np.full((nx, ny), np.nan, dtype=float)
        p_valid = valid_map & np.isfinite(hd_info_obs_map)
        p_val_map[p_valid] = (
            1.0 + null_ge_count_map[p_valid]
        ) / (1.0 + null_valid_count_map[p_valid].astype(float))

        global_null_vals = np.asarray(global_null_vals, dtype=float)
        global_null_vals = global_null_vals[np.isfinite(global_null_vals)]
        if global_null_vals.size > 0 and np.isfinite(global_hd_info_obs):
            global_null_mean = float(np.mean(global_null_vals))
            global_p = float(
                (1.0 + np.sum(global_null_vals >= float(global_hd_info_obs)))
                / (float(global_null_vals.size) + 1.0)
            )
        else:
            global_null_mean = np.nan
            global_p = np.nan

        return {
            "null_mean_map": null_mean_map,
            "p_val_map": p_val_map,
            "global_null_mean": global_null_mean,
            "global_p": global_p,
        }

    def _compute_vector_maps(spike_times, rng_local=None):
        spike_count_map = np.zeros((nx, ny), dtype=int)
        valid_bin_mask = np.zeros((nx, ny), dtype=bool)
        spike_count_valid_mask = np.zeros((nx, ny), dtype=bool)
        visited_angle_bin_count_map = np.zeros((nx, ny), dtype=int)
        visited_angle_max_sep_deg_map = np.full((nx, ny), np.nan, dtype=float)
        visited_angle_valid_mask = np.zeros((nx, ny), dtype=bool)
        pref_angle_pass_mask = np.zeros((nx, ny), dtype=bool)
        pref_angle_threshold_map = np.full((nx, ny), np.nan, dtype=float)
        pref_hd_deg_map = np.full((nx, ny), np.nan, dtype=float)
        pref_hd_deg_weighted_map = np.full((nx, ny), np.nan, dtype=float)
        pref_vector_strength_map = np.full((nx, ny), np.nan, dtype=float)
        mean_visited_rate_hz_map = np.full((nx, ny), np.nan, dtype=float)
        pref_rate_hz_map = np.full((nx, ny), np.nan, dtype=float)
        pref_rate_norm_map = np.full((nx, ny), np.nan, dtype=float)
        hd_info_obs_bits_per_spike_map = np.full((nx, ny), np.nan, dtype=float)
        hd_info_null_mean_bits_per_spike_map = np.full((nx, ny), np.nan, dtype=float)
        hd_info_p_value_map = np.full((nx, ny), np.nan, dtype=float)
        hd_info_sig_mask = np.zeros((nx, ny), dtype=bool)
        hd_info_star_count_map = np.zeros((nx, ny), dtype=int)
        hd_info_star_text_map = np.full((nx, ny), "", dtype=object)
        u_map = np.full((nx, ny), np.nan, dtype=float)
        v_map = np.full((nx, ny), np.nan, dtype=float)
        u_norm_map = np.full((nx, ny), np.nan, dtype=float)
        v_norm_map = np.full((nx, ny), np.nan, dtype=float)
        occ_counts_hd_map = np.zeros((nx, ny, n_hd_bins), dtype=int)
        spike_counts_hd_map = np.zeros((nx, ny, n_hd_bins), dtype=int)
        occ_time_hd_map = np.zeros((nx, ny, n_hd_bins), dtype=float)
        visited_mask_hd_map = np.zeros((nx, ny, n_hd_bins), dtype=bool)
        occ_time_for_info_map = np.zeros((nx, ny, n_hd_bins), dtype=float)
        rate_hd_map = np.full((nx, ny, n_hd_bins), np.nan, dtype=float)

        spike_times_raw = _sanitize_spike_times(spike_times, apply_analysis_mask=False)
        spike_times = _sanitize_spike_times(spike_times_raw, apply_analysis_mask=True)
        if spike_times.size > 0:
            x_spk = x_neural[spike_times]
            y_spk = y_neural[spike_times]
            dir_spk = np.mod(direction_angles[spike_times], 360.0)
            x_bin_spk = np.digitize(x_spk, bins[0]) - 1
            y_bin_spk = np.digitize(y_spk, bins[1]) - 1
            x_bin_spk = np.clip(x_bin_spk, 0, nx - 1)
            y_bin_spk = np.clip(y_bin_spk, 0, ny - 1)

            spike_count_map, _, _ = np.histogram2d(x_spk, y_spk, bins=bins)
            spike_count_map = spike_count_map.astype(int)
        else:
            x_spk = np.array([], dtype=float)
            y_spk = np.array([], dtype=float)
            dir_spk = np.array([], dtype=float)
            x_bin_spk = np.array([], dtype=int)
            y_bin_spk = np.array([], dtype=int)

        for ix in range(nx):
            for iy in range(ny):
                occ_in_bin = (x_bin_traj == ix) & (y_bin_traj == iy)
                dir_occ_bin = dir_traj[occ_in_bin]
                if dir_occ_bin.size > 0 and frame_rate > 0:
                    occ_counts = _hist_hd(dir_occ_bin)
                    occ_time = occ_counts / float(frame_rate)
                else:
                    occ_counts = np.zeros(n_hd_bins, dtype=int)
                    occ_time = np.zeros(n_hd_bins, dtype=float)

                occ_counts_hd_map[ix, iy, :] = occ_counts
                occ_time_hd_map[ix, iy, :] = occ_time

                in_bin = (x_bin_spk == ix) & (y_bin_spk == iy)
                n_spikes_bin = int(np.sum(in_bin))
                if n_spikes_bin > int(min_spikes_per_bin):
                    dir_spk_bin = dir_spk[in_bin]
                    spk_counts_hd = _hist_hd(dir_spk_bin)
                else:
                    dir_spk_bin = np.array([], dtype=float)
                    spk_counts_hd = np.zeros(n_hd_bins, dtype=int)

                spike_counts_hd_map[ix, iy, :] = spk_counts_hd
                occ_counts_for_rate = np.asarray(occ_counts, dtype=float)
                spk_counts_for_rate = np.asarray(spk_counts_hd, dtype=float)
                if hd_count_smooth_sigma > 0:
                    occ_counts_for_rate = gaussian_filter1d(
                        occ_counts_for_rate,
                        sigma=hd_count_smooth_sigma,
                        mode="wrap",
                    )
                    spk_counts_for_rate = gaussian_filter1d(
                        spk_counts_for_rate,
                        sigma=hd_count_smooth_sigma,
                        mode="wrap",
                    )
                occ_time_for_rate = (
                    occ_counts_for_rate / float(frame_rate)
                    if frame_rate > 0
                    else np.zeros(n_hd_bins, dtype=float)
                )
                occ_valid_for_rate = occ_counts >= int(min_occ_frames_per_angle)
                with np.errstate(divide="ignore", invalid="ignore"):
                    rate_hd = np.divide(
                        spk_counts_for_rate,
                        occ_time_for_rate,
                        out=np.zeros(n_hd_bins, dtype=float),
                        where=(occ_time_for_rate > 0) & occ_valid_for_rate,
                    )
                rate_hd_map[ix, iy, :] = rate_hd

                if n_hd_bins > 0:
                    visited_mask = occ_counts >= int(min_occ_frames_per_angle)
                else:
                    visited_mask = np.array([], dtype=bool)
                visited_mask_hd_map[ix, iy, :] = visited_mask
                n_visited_angle_bins = int(np.sum(visited_mask))
                visited_angle_bin_count_map[ix, iy] = n_visited_angle_bins
                if n_visited_angle_bins >= 2:
                    visited_centers_deg = hd_bin_centers_deg[visited_mask]
                    diffs = np.abs(
                        (
                            visited_centers_deg[:, None]
                            - visited_centers_deg[None, :]
                            + 180.0
                        )
                        % 360.0
                        - 180.0
                    )
                    max_sep_deg = float(np.nanmax(diffs))
                else:
                    max_sep_deg = 0.0
                visited_angle_max_sep_deg_map[ix, iy] = max_sep_deg
                sep_valid = (
                    max_sep_deg >= float(min_visited_angle_separation_deg)
                    if min_visited_angle_separation_deg > 0
                    else True
                )
                visited_angle_valid = (
                    n_visited_angle_bins >= int(min_visited_angle_bins)
                    and sep_valid
                )
                visited_angle_valid_mask[ix, iy] = bool(visited_angle_valid)
                if visited_mask.size > 0 and np.any(visited_mask):
                    mean_visited_rate = float(np.nanmean(rate_hd[visited_mask]))
                    std_visited_rate = float(np.nanstd(rate_hd[visited_mask]))
                else:
                    mean_visited_rate = 0.0
                    std_visited_rate = 0.0
                if not np.isfinite(mean_visited_rate):
                    mean_visited_rate = 0.0
                if not np.isfinite(std_visited_rate):
                    std_visited_rate = 0.0
                mean_visited_rate_hz_map[ix, iy] = mean_visited_rate

                weights_pref = np.asarray(rate_hd, dtype=float)
                if weights_pref.size > 0:
                    weights_pref = np.where(visited_mask, weights_pref, 0.0)
                    weights_pref = np.clip(weights_pref, 0.0, None)
                sum_w = float(np.nansum(weights_pref))
                vec_x = 0.0
                vec_y = 0.0
                if sum_w > 0 and weights_pref.size == hd_bin_centers_rad.size:
                    vec_x = float(
                        np.nansum(weights_pref * np.cos(hd_bin_centers_rad))
                    )
                    vec_y = float(
                        np.nansum(weights_pref * np.sin(hd_bin_centers_rad))
                    )
                    pref_vector_strength = float(np.hypot(vec_x, vec_y) / sum_w)
                    preferred_dir_weighted_deg = (
                        np.rad2deg(np.arctan2(vec_y, vec_x)) % 360.0
                    )
                else:
                    pref_vector_strength = 0.0
                    preferred_dir_weighted_deg = np.nan
                pref_hd_deg_weighted_map[ix, iy] = preferred_dir_weighted_deg
                pref_vector_strength_map[ix, iy] = pref_vector_strength

                if n_spikes_bin > int(min_spikes_per_bin):
                    if hd_info_use_smoothed_counts:
                        occ_counts_for_info = np.asarray(occ_counts_for_rate, dtype=float)
                        spk_counts_for_info = np.asarray(spk_counts_for_rate, dtype=float)
                        null_smooth_sigma = float(hd_count_smooth_sigma)
                    else:
                        occ_counts_for_info = np.asarray(occ_counts, dtype=float)
                        spk_counts_for_info = np.asarray(spk_counts_hd, dtype=float)
                        null_smooth_sigma = 0.0
                    occ_time_for_info = (
                        occ_counts_for_info / float(frame_rate)
                        if frame_rate > 0
                        else np.zeros(n_hd_bins, dtype=float)
                    )
                    occ_time_for_info_map[ix, iy, :] = occ_time_for_info
                    hd_info_obs = _compute_hd_info_bits_per_spike(
                        spk_counts_for_info,
                        occ_time_for_info,
                        visited_mask=visited_mask,
                        prior_alpha=hd_info_prior_alpha,
                        prior_beta_s=hd_info_prior_beta_s,
                    )
                    hd_info_obs_bits_per_spike_map[ix, iy] = hd_info_obs
                    if hd_info_null_mode == "multinomial":
                        hd_info_null_mean, hd_info_p, _ = _run_hd_multinomial_null(
                            observed_info_bits=hd_info_obs,
                            n_spikes_total=n_spikes_bin,
                            occ_counts_prior=occ_counts_for_info,
                            occ_time_for_rate=occ_time_for_info,
                            visited_mask=visited_mask,
                            n_shuffles=hd_info_n_shuffles,
                            prior_alpha=hd_info_prior_alpha,
                            prior_beta_s=hd_info_prior_beta_s,
                            smooth_sigma=null_smooth_sigma,
                            rng=rng_local,
                        )
                        hd_info_null_mean_bits_per_spike_map[ix, iy] = hd_info_null_mean
                        hd_info_p_value_map[ix, iy] = hd_info_p
                        hd_info_sig = bool(
                            np.isfinite(hd_info_obs)
                            and np.isfinite(hd_info_null_mean)
                            and np.isfinite(hd_info_p)
                            and (hd_info_p < float(hd_info_alpha))
                            and (hd_info_obs > hd_info_null_mean)
                        )
                        hd_info_sig_mask[ix, iy] = hd_info_sig
                        star_count = _p_to_star_count(
                            hd_info_p,
                            thresholds=hd_info_star_thresholds,
                        )
                        hd_info_star_count_map[ix, iy] = star_count
                        hd_info_star_text_map[ix, iy] = _star_text_from_count(star_count)

                if n_spikes_bin <= int(min_spikes_per_bin):
                    continue

                spike_count_valid_mask[ix, iy] = True

                pref_idx = -1
                preferred_dir_deg = np.nan
                if np.any(visited_mask):
                    visited_idx = np.where(visited_mask)[0]
                    if preferred_angle_mode == "peak":
                        visited_rates = np.asarray(rate_hd[visited_idx], dtype=float)
                        if visited_rates.size > 0 and np.any(np.isfinite(visited_rates)):
                            peak_local_i = int(np.nanargmax(visited_rates))
                            pref_idx = int(visited_idx[peak_local_i])
                            preferred_dir_deg = float(hd_bin_centers_deg[pref_idx])
                    elif preferred_angle_mode == "peak_cluster_mean":
                        pos_rate_mask = (
                            visited_mask
                            & np.isfinite(rate_hd)
                            & (rate_hd > 0)
                        )
                        clusters = _find_circular_true_clusters(pos_rate_mask)
                        if len(clusters) == 0:
                            visited_rates = np.asarray(rate_hd[visited_idx], dtype=float)
                            if visited_rates.size > 0 and np.any(np.isfinite(visited_rates)):
                                peak_local_i = int(np.nanargmax(visited_rates))
                                pref_idx = int(visited_idx[peak_local_i])
                                preferred_dir_deg = float(hd_bin_centers_deg[pref_idx])
                        else:
                            best_cluster = None
                            best_peak = -np.inf
                            for cluster_idx in clusters:
                                cluster_rates = np.asarray(rate_hd[cluster_idx], dtype=float)
                                if cluster_rates.size == 0 or not np.any(np.isfinite(cluster_rates)):
                                    cluster_peak = -np.inf
                                else:
                                    cluster_peak = float(np.nanmax(cluster_rates))
                                if cluster_peak > best_peak:
                                    best_peak = cluster_peak
                                    best_cluster = cluster_idx

                            if best_cluster is not None and best_cluster.size > 0:
                                cluster_rates = np.asarray(rate_hd[best_cluster], dtype=float)
                                cluster_rates = np.clip(np.nan_to_num(cluster_rates, nan=0.0), 0.0, None)
                                cluster_centers_rad = hd_bin_centers_rad[best_cluster]
                                cluster_centers_deg = hd_bin_centers_deg[best_cluster]

                                if float(np.nansum(cluster_rates)) > 0:
                                    vec_x_c = float(
                                        np.nansum(cluster_rates * np.cos(cluster_centers_rad))
                                    )
                                    vec_y_c = float(
                                        np.nansum(cluster_rates * np.sin(cluster_centers_rad))
                                    )
                                    preferred_dir_deg = (
                                        np.rad2deg(np.arctan2(vec_y_c, vec_x_c)) % 360.0
                                    )
                                else:
                                    preferred_dir_deg = float(np.nanmean(cluster_centers_deg))

                                circ_diff = np.abs(
                                    (
                                        cluster_centers_deg
                                        - preferred_dir_deg
                                        + 180.0
                                    )
                                    % 360.0
                                    - 180.0
                                )
                                pref_idx = int(best_cluster[int(np.argmin(circ_diff))])
                    else:  # "circular_mean"
                        if sum_w > 0 and weights_pref.size == hd_bin_centers_rad.size:
                            preferred_dir_cont_deg = (
                                np.rad2deg(np.arctan2(vec_y, vec_x)) % 360.0
                            )
                            visited_centers_deg = hd_bin_centers_deg[visited_idx]
                            circ_diff = np.abs(
                                (
                                    visited_centers_deg
                                    - preferred_dir_cont_deg
                                    + 180.0
                                )
                                % 360.0
                                - 180.0
                            )
                            nearest_i = int(np.argmin(circ_diff))
                            pref_idx = int(visited_idx[nearest_i])
                            preferred_dir_deg = float(visited_centers_deg[nearest_i])
                pref_hd_deg_map[ix, iy] = preferred_dir_deg

                if not np.isfinite(preferred_dir_deg):
                    pref_angle_pass_mask[ix, iy] = False
                    pref_angle_threshold_map[ix, iy] = np.nan
                    valid_bin_mask[ix, iy] = False
                    pref_rate_hz_map[ix, iy] = 0.0
                    pref_rate_norm_map[ix, iy] = 0.0
                    u_map[ix, iy] = 0.0
                    v_map[ix, iy] = 0.0
                    u_norm_map[ix, iy] = 0.0
                    v_norm_map[ix, iy] = 0.0
                    continue

                if pref_idx < 0:
                    circ_diff_pref = np.abs(
                        (
                            hd_bin_centers_deg
                            - preferred_dir_deg
                            + 180.0
                        )
                        % 360.0
                        - 180.0
                    )
                    pref_idx = int(np.argmin(circ_diff_pref))
                pref_rate = float(rate_hd[pref_idx]) if n_hd_bins > 0 else 0.0
                if not np.isfinite(pref_rate):
                    pref_rate = 0.0
                if pref_rate <= 0:
                    pref_angle_pass_mask[ix, iy] = False
                    pref_angle_threshold_map[ix, iy] = np.nan
                    valid_bin_mask[ix, iy] = False
                    pref_rate_hz_map[ix, iy] = 0.0
                    pref_rate_norm_map[ix, iy] = 0.0
                    u_map[ix, iy] = 0.0
                    v_map[ix, iy] = 0.0
                    u_norm_map[ix, iy] = 0.0
                    v_norm_map[ix, iy] = 0.0
                    continue
                pref_rate_hz_map[ix, iy] = pref_rate

                if pref_angle_std_threshold is None:
                    pref_angle_pass = True
                    pref_angle_threshold = np.nan
                else:
                    pref_angle_threshold = (
                        mean_visited_rate
                        + float(pref_angle_std_threshold) * std_visited_rate
                    )
                    pref_angle_pass = (
                        np.isfinite(pref_rate)
                        and np.isfinite(pref_angle_threshold)
                        and (pref_rate > pref_angle_threshold)
                    )
                pref_angle_pass = bool(pref_angle_pass) and (
                    float(pref_vector_strength) >= float(min_pref_vector_strength)
                )
                pref_angle_pass_mask[ix, iy] = bool(pref_angle_pass)
                pref_angle_threshold_map[ix, iy] = pref_angle_threshold
                valid_bin_mask[ix, iy] = bool(pref_angle_pass and visited_angle_valid)
                if enable_hd_info_shuffle_gate:
                    valid_bin_mask[ix, iy] = bool(
                        valid_bin_mask[ix, iy] and hd_info_sig_mask[ix, iy]
                    )

                if mean_visited_rate > 0:
                    pref_rate_norm = pref_rate / mean_visited_rate
                else:
                    pref_rate_norm = 0.0
                if not np.isfinite(pref_rate_norm):
                    pref_rate_norm = 0.0
                pref_rate_norm_map[ix, iy] = pref_rate_norm

                arrow_theta_deg = preferred_dir_weighted_deg
                arrow_len = float(pref_vector_strength)
                if not np.isfinite(arrow_theta_deg):
                    arrow_theta_deg = preferred_dir_deg
                if not np.isfinite(arrow_len) or arrow_len < 0:
                    arrow_len = 0.0
                arrow_len = float(np.clip(arrow_len, 0.0, 1.0))

                pref_rad = np.deg2rad(arrow_theta_deg)
                u_map[ix, iy] = arrow_len * np.cos(pref_rad)
                v_map[ix, iy] = arrow_len * np.sin(pref_rad)
                u_norm_map[ix, iy] = arrow_len * np.cos(pref_rad)
                v_norm_map[ix, iy] = arrow_len * np.sin(pref_rad)

        global_occ_counts_curve = np.nansum(occ_counts_hd_map, axis=(0, 1))
        global_spike_counts_curve = np.nansum(spike_counts_hd_map, axis=(0, 1))
        global_visited_mask = global_occ_counts_curve >= int(min_occ_frames_per_angle)
        if hd_info_use_smoothed_counts:
            global_occ_counts_for_info = _smooth_hd_counts_if_needed(global_occ_counts_curve)
            global_spike_counts_for_info = _smooth_hd_counts_if_needed(global_spike_counts_curve)
        else:
            global_occ_counts_for_info = np.asarray(global_occ_counts_curve, dtype=float)
            global_spike_counts_for_info = np.asarray(global_spike_counts_curve, dtype=float)
        global_occ_time_for_info = (
            global_occ_counts_for_info / float(frame_rate)
            if frame_rate > 0
            else np.zeros(n_hd_bins, dtype=float)
        )
        global_hd_info_obs = np.nan
        global_hd_info_null_mean = np.nan
        global_hd_info_p = np.nan
        global_hd_info_sig = False
        global_hd_info_star_count = 0
        global_hd_info_star_text = ""
        global_n_spikes_total = int(np.nansum(global_spike_counts_curve))
        if global_n_spikes_total > int(min_spikes_per_bin):
            global_hd_info_obs = _compute_hd_info_bits_per_spike(
                global_spike_counts_for_info,
                global_occ_time_for_info,
                visited_mask=global_visited_mask,
                prior_alpha=hd_info_prior_alpha,
                prior_beta_s=hd_info_prior_beta_s,
            )

        if hd_info_null_mode == "timeshift":
            time_shift_out = _run_time_shift_hd_null(
                spike_times_for_shift=spike_times,
                spike_count_valid_mask=spike_count_valid_mask,
                visited_mask_map=visited_mask_hd_map,
                occ_time_for_info_map=occ_time_for_info_map,
                hd_info_obs_map=hd_info_obs_bits_per_spike_map,
                global_visited_mask=global_visited_mask,
                global_occ_time_for_info=global_occ_time_for_info,
                global_hd_info_obs=global_hd_info_obs,
                rng_local=rng_local,
            )
            hd_info_null_mean_bits_per_spike_map[:] = time_shift_out["null_mean_map"]
            hd_info_p_value_map[:] = time_shift_out["p_val_map"]
            hd_info_sig_mask[:] = (
                np.isfinite(hd_info_obs_bits_per_spike_map)
                & np.isfinite(hd_info_null_mean_bits_per_spike_map)
                & np.isfinite(hd_info_p_value_map)
                & (hd_info_p_value_map < float(hd_info_alpha))
                & (hd_info_obs_bits_per_spike_map > hd_info_null_mean_bits_per_spike_map)
            )
            star_fn = np.vectorize(
                lambda p: _p_to_star_count(
                    p,
                    thresholds=hd_info_star_thresholds,
                ),
                otypes=[int],
            )
            hd_info_star_count_map[:] = np.where(
                np.isfinite(hd_info_p_value_map),
                star_fn(hd_info_p_value_map),
                0,
            )
            text_fn = np.vectorize(_star_text_from_count, otypes=[object])
            hd_info_star_text_map[:] = text_fn(hd_info_star_count_map)
            global_hd_info_null_mean = time_shift_out["global_null_mean"]
            global_hd_info_p = time_shift_out["global_p"]
        elif global_n_spikes_total > int(min_spikes_per_bin):
            global_null_smooth_sigma = float(hd_count_smooth_sigma) if hd_info_use_smoothed_counts else 0.0
            global_hd_info_null_mean, global_hd_info_p, _ = _run_hd_multinomial_null(
                observed_info_bits=global_hd_info_obs,
                n_spikes_total=global_n_spikes_total,
                occ_counts_prior=global_occ_counts_for_info,
                occ_time_for_rate=global_occ_time_for_info,
                visited_mask=global_visited_mask,
                n_shuffles=hd_info_n_shuffles,
                prior_alpha=hd_info_prior_alpha,
                prior_beta_s=hd_info_prior_beta_s,
                smooth_sigma=global_null_smooth_sigma,
                rng=rng_local,
            )

        if global_n_spikes_total > int(min_spikes_per_bin):
            global_hd_info_sig = bool(
                np.isfinite(global_hd_info_obs)
                and np.isfinite(global_hd_info_null_mean)
                and np.isfinite(global_hd_info_p)
                and (global_hd_info_p < float(hd_info_alpha))
                and (global_hd_info_obs > global_hd_info_null_mean)
            )
            global_hd_info_star_count = _p_to_star_count(
                global_hd_info_p,
                thresholds=hd_info_star_thresholds,
            )
            global_hd_info_star_text = _star_text_from_count(global_hd_info_star_count)

        return {
            "spike_count_map": spike_count_map,
            "valid_bin_mask": valid_bin_mask,
            "spike_count_valid_mask": spike_count_valid_mask,
            "visited_angle_bin_count_map": visited_angle_bin_count_map,
            "visited_angle_max_sep_deg_map": visited_angle_max_sep_deg_map,
            "visited_angle_valid_mask": visited_angle_valid_mask,
            "pref_angle_pass_mask": pref_angle_pass_mask,
            "pref_angle_threshold_map": pref_angle_threshold_map,
            "pref_hd_deg_map": pref_hd_deg_map,
            "pref_hd_deg_weighted_map": pref_hd_deg_weighted_map,
            "pref_dir_deg_map": pref_hd_deg_map.copy(),
            "pref_vector_strength_map": pref_vector_strength_map,
            "mean_visited_rate_hz_map": mean_visited_rate_hz_map,
            "pref_rate_hz_map": pref_rate_hz_map,
            "pref_rate_norm_map": pref_rate_norm_map,
            "hd_info_obs_bits_per_spike_map": hd_info_obs_bits_per_spike_map,
            "hd_info_null_mean_bits_per_spike_map": hd_info_null_mean_bits_per_spike_map,
            "hd_info_p_value_map": hd_info_p_value_map,
            "hd_info_sig_mask": hd_info_sig_mask,
            "hd_info_star_count_map": hd_info_star_count_map,
            "hd_info_star_text_map": hd_info_star_text_map,
            "u_map": u_map,
            "v_map": v_map,
            "u_norm_map": u_norm_map,
            "v_norm_map": v_norm_map,
            "occ_counts_hd_map": occ_counts_hd_map,
            "spike_counts_hd_map": spike_counts_hd_map,
            "occ_time_hd_map": occ_time_hd_map,
            "rate_hd_map": rate_hd_map,
            "visited_mask_hd_map": visited_mask_hd_map,
            "occ_time_for_info_map": occ_time_for_info_map,
            "global_occ_counts_curve": np.asarray(global_occ_counts_curve, dtype=float),
            "global_spike_counts_curve": np.asarray(global_spike_counts_curve, dtype=float),
            "global_visited_mask": np.asarray(global_visited_mask, dtype=bool),
            "global_occ_time_for_info": np.asarray(global_occ_time_for_info, dtype=float),
            "global_hd_info_obs_bits_per_spike": global_hd_info_obs,
            "global_hd_info_null_mean_bits_per_spike": global_hd_info_null_mean,
            "global_hd_info_p_value": global_hd_info_p,
            "global_hd_info_sig": bool(global_hd_info_sig),
            "global_hd_info_star_count": int(global_hd_info_star_count),
            "global_hd_info_star_text": str(global_hd_info_star_text),
            "hd_info_null_mode": hd_info_null_mode,
        }

    all_spike_times = spikes[cell_idx] if cell_idx < len(spikes) else []
    simple_spike_times = (
        refined_SS[cell_idx]
        if (refined_SS is not None and cell_idx < len(refined_SS))
        else np.array([], dtype=int)
    )
    complex_spike_times = (
        all_CS_spikes[cell_idx]
        if (all_CS_spikes is not None and cell_idx < len(all_CS_spikes))
        else np.array([], dtype=int)
    )

    rng_all = np.random.default_rng(rng_base.integers(0, 2**63 - 1))
    rng_simple = np.random.default_rng(rng_base.integers(0, 2**63 - 1))
    rng_complex = np.random.default_rng(rng_base.integers(0, 2**63 - 1))
    types = {
        "all": _compute_vector_maps(all_spike_times, rng_local=rng_all),
        "simple": _compute_vector_maps(simple_spike_times, rng_local=rng_simple),
        "complex": _compute_vector_maps(complex_spike_times, rng_local=rng_complex),
    }

    results = {
        "cell_id": int(cell_idx),
        "moving_epochs": moving_epochs,
        "moving_indices": moving_idx,
        "x_traj": x_traj,
        "y_traj": y_traj,
        "bins": bins,
        "extent": extent,
        "x_centers": x_centers,
        "y_centers": y_centers,
        "types": types,
        "plot_params": {
            "display_cell_num": display_cell_num,
            "is_first_column": bool(is_first_column),
            "is_last_column": bool(is_last_column),
            "quiver_scale": quiver_scale,
            "cmap": cmap,
            "use_normalized_rate": bool(use_normalized_rate),
        },
        "params": {
            "width_real": float(width_real),
            "height_real": float(height_real),
            "bin_size": float(bin_size),
            "kernel_size": int(kernel_size),
            "filter_type": filter_type,
            "speed_threshold": float(speed_threshold),
            "min_duration_s": float(min_duration_s),
            "merge_gap_s": float(merge_gap_s),
            "min_spikes_per_bin": int(min_spikes_per_bin),
            "min_visited_angle_bins": int(min_visited_angle_bins),
            "min_visited_angle_separation_deg": float(min_visited_angle_separation_deg),
            "min_occ_frames_per_angle": int(min_occ_frames_per_angle),
            "hd_count_smooth_sigma": float(hd_count_smooth_sigma),
            "min_pref_vector_strength": float(min_pref_vector_strength),
            "preferred_angle_mode": preferred_angle_mode,
            "pref_angle_std_threshold": None if pref_angle_std_threshold is None else float(pref_angle_std_threshold),
            "hd_bin_size_deg": float(hd_bin_size_deg),
            "hd_bin_edges_deg": np.asarray(hd_bin_edges, dtype=float),
            "hd_bin_centers_deg": np.asarray(
                hd_bin_centers_deg, dtype=float
            ),
            "direction_mode": direction_mode,
            "travel_smooth_window": int(travel_smooth_window),
            "travel_min_step": float(travel_min_step),
            "use_normalized_rate": bool(use_normalized_rate),
            "enable_hd_info_shuffle_gate": bool(enable_hd_info_shuffle_gate),
            "hd_info_n_shuffles": int(hd_info_n_shuffles),
            "hd_info_alpha": float(hd_info_alpha),
            "hd_info_gate_rule": hd_info_gate_rule,
            "hd_info_random_seed": (
                None if hd_info_random_seed is None else int(hd_info_random_seed)
            ),
            "hd_info_star_thresholds": tuple(float(x) for x in hd_info_star_thresholds),
            "hd_info_prior_alpha": float(hd_info_prior_alpha),
            "hd_info_prior_beta_s": float(hd_info_prior_beta_s),
            "hd_info_use_smoothed_counts": bool(hd_info_use_smoothed_counts),
            "hd_info_null_mode": hd_info_null_mode,
            "hd_time_shift_min_s": float(hd_time_shift_min_s),
            "hd_time_shift_sessionwise": bool(hd_time_shift_sessionwise),
            "session_start_frames": [int(start) for start, _ in session_bounds],
        },
    }

    if axes is not None:
        plot_hd_vector_field_single_moving_results(
            results,
            axes=axes,
            quiver_scale=quiver_scale,
            cmap=cmap,
            show_trajectory=False,
            use_normalized_rate=use_normalized_rate,
        )

    return results


def plot_hd_vector_field_single_moving_results(
    results,
    axes,
    quiver_scale=None,
    cmap="viridis",
    rate_vmin=0.0,
    rate_vmax=None,
    show_trajectory=False,
    use_normalized_rate=False,
    place_field_contours=None,
    max_arrow_length=None,
    require_shuffle_pass=False,
):
    """Plot 3-row head-direction vector fields for one cell."""
    if len(axes) != 3:
        raise ValueError("axes must contain exactly 3 axes (all/simple/complex).")

    extent = results["extent"]
    x_centers = np.asarray(results["x_centers"], dtype=float)
    y_centers = np.asarray(results["y_centers"], dtype=float)
    Xc, Yc = np.meshgrid(x_centers, y_centers, indexing="ij")
    types = results["types"]
    plot_params = results.get("plot_params", {})
    display_cell_num = plot_params.get("display_cell_num", None)
    is_first_column = bool(plot_params.get("is_first_column", False))
    width_real = float(results.get("params", {}).get("width_real", 35.5))
    bin_size = float(results.get("params", {}).get("bin_size", 1.5))
    title_num = display_cell_num if display_cell_num is not None else (results["cell_id"] + 1)
    rate_key = "pref_rate_norm_map" if use_normalized_rate else "pref_rate_hz_map"
    u_key = "u_norm_map" if use_normalized_rate else "u_map"
    v_key = "v_norm_map" if use_normalized_rate else "v_map"

    if rate_vmax is None:
        local_max = 0.0
        for key in ("all", "simple", "complex"):
            u_vals = np.asarray(types[key][u_key], dtype=float)
            v_vals = np.asarray(types[key][v_key], dtype=float)
            values = np.hypot(u_vals, v_vals)
            finite = values[np.isfinite(values)]
            if finite.size > 0:
                local_max = max(local_max, float(np.nanmax(finite)))
        rate_vmax = local_max if local_max > 0 else 1.0
    if max_arrow_length is not None and np.isfinite(max_arrow_length) and max_arrow_length > 0:
        rate_vmax = min(float(rate_vmax), float(max_arrow_length))

    if quiver_scale is None:
        # Robust auto-scale: use percentile to avoid one outlier shrinking all arrows.
        valid_rates = []
        for key in ("all", "simple", "complex"):
            u_vals = np.asarray(types[key][u_key], dtype=float)
            v_vals = np.asarray(types[key][v_key], dtype=float)
            arr = np.hypot(u_vals, v_vals)
            vals = arr[np.isfinite(arr) & (arr > 0)]
            if max_arrow_length is not None and np.isfinite(max_arrow_length) and max_arrow_length > 0 and vals.size > 0:
                vals = np.minimum(vals, float(max_arrow_length))
            if vals.size > 0:
                valid_rates.append(vals)
        if valid_rates:
            ref_rate = float(np.nanpercentile(np.concatenate(valid_rates), 90))
            if not np.isfinite(ref_rate) or ref_rate <= 0:
                ref_rate = float(rate_vmax)
        else:
            ref_rate = float(rate_vmax)
        if not np.isfinite(ref_rate) or ref_rate <= 0:
            ref_rate = 1.0
        quiver_scale = ref_rate / (0.45 * bin_size) if bin_size > 0 else 1.0

    row_info = [
        ("all", "All spikes"),
        ("simple", "Simple spikes"),
        ("complex", "Complex spikes"),
    ]

    mappables = {"all": None, "simple": None, "complex": None}

    def _plot_pf_contour(ax, contour_info):
        if contour_info is None:
            return
        mask = np.asarray(contour_info.get("mask", []), dtype=bool)
        if mask.size == 0 or not np.any(mask):
            return
        c_width = float(contour_info.get("width_real", extent[1] - extent[0]))
        c_height = float(contour_info.get("height_real", extent[3] - extent[2]))
        color = contour_info.get("color", "magenta")
        # Pad by one bin on each side so contour lines trace outer edges cleanly.
        padded_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=bool)
        padded_mask[1:-1, 1:-1] = mask
        bin_x = c_width / mask.shape[0]
        bin_y = c_height / mask.shape[1]
        padded_extent = (-bin_x, c_width + bin_x, -bin_y, c_height + bin_y)
        ax.contour(
            padded_mask.T,
            levels=[0.5],
            colors=color,
            linewidths=0.5,
            alpha=0.5,
            extent=padded_extent,
            origin="lower",
        )

    for row_idx, (type_key, row_label) in enumerate(row_info):
        ax = axes[row_idx]
        if show_trajectory:
            ax.plot(results["x_traj"], results["y_traj"], color="#cfcfcf", alpha=0.6, linewidth=0.5)

        type_data = types[type_key]
        rate_raw = np.asarray(type_data[rate_key], dtype=float)
        u_raw = np.asarray(type_data[u_key], dtype=float)
        v_raw = np.asarray(type_data[v_key], dtype=float)
        arrow_mag_raw = np.hypot(u_raw, v_raw)

        rate_plot = rate_raw.copy()
        u_plot = u_raw.copy()
        v_plot = v_raw.copy()
        arrow_mag_plot = arrow_mag_raw.copy()
        if max_arrow_length is not None and np.isfinite(max_arrow_length) and max_arrow_length > 0:
            max_len = float(max_arrow_length)
            clip_ratio = np.ones_like(arrow_mag_plot, dtype=float)
            over = np.isfinite(arrow_mag_plot) & (arrow_mag_plot > max_len)
            clip_ratio[over] = max_len / arrow_mag_plot[over]
            arrow_mag_plot[over] = max_len
            u_plot = u_plot * clip_ratio
            v_plot = v_plot * clip_ratio
            rate_plot[over] = np.minimum(rate_plot[over], max_len)

        valid_mask = (
            np.asarray(type_data["valid_bin_mask"], dtype=bool)
            & np.isfinite(u_plot)
            & np.isfinite(v_plot)
            & np.isfinite(arrow_mag_plot)
            & (arrow_mag_plot > 0)
        )
        if require_shuffle_pass and ("hd_info_sig_mask" in type_data):
            valid_mask &= np.asarray(type_data["hd_info_sig_mask"], dtype=bool)

        contour_info = None
        if isinstance(place_field_contours, dict):
            contour_info = place_field_contours.get(type_key)
        _plot_pf_contour(ax, contour_info)

        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.add_patch(
            Rectangle(
                (extent[0], extent[2]),
                extent[1] - extent[0],
                extent[3] - extent[2],
                linewidth=1.0,
                edgecolor="black",
                facecolor="none",
                zorder=3,
            )
        )

        if is_first_column:
            ax.text(
                -0.08,
                0.5,
                row_label,
                transform=ax.transAxes,
                rotation=90,
                va="center",
                ha="right",
                fontsize=6,
                fontname="Arial",
            )
        if row_idx == 0:
            ax.set_title(f"Cell {title_num}", fontsize=6, fontname="Arial")

        # Draw arrows last so they always sit above contours/frame.
        if np.any(valid_mask):
            ax.quiver(
                Xc[valid_mask],
                Yc[valid_mask],
                u_plot[valid_mask],
                v_plot[valid_mask],
                color="black",
                angles="xy",
                scale_units="xy",
                scale=quiver_scale,
                width=0.0045,
                headwidth=3.0,
                headlength=3.4,
                headaxislength=3.2,
                minshaft=1.0,
                pivot="tail",
                zorder=10,
            )

    return mappables


def plot_hd_vector_fields_all_cells_moving(
    x_neural,
    y_neural,
    hd_angles_neural,
    spikes,
    speed,
    frame_rate,
    refined_SS=None,
    all_CS_spikes=None,
    pc_output_all=None,
    cell_order=None,
    width_real=35.5,
    height_real=20,
    bin_size=1.5,
    kernel_size=21,
    filter_type="boxcar",
    speed_threshold=2,
    min_duration_s=0.5,
    merge_gap_s=1.0,
    bad_timepoints_all=None,
    min_spikes_per_bin=20,
    min_visited_angle_bins=2,
    min_visited_angle_separation_deg=0.0,
    min_occ_frames_per_angle=2,
    hd_count_smooth_sigma=1.0,
    min_pref_vector_strength=0.0,
    preferred_angle_mode="peak",
    pref_angle_std_threshold=None,
    hd_bin_size_deg=30,
    direction_mode="head",
    travel_smooth_window=5,
    travel_min_step=0.0,
    cmap="viridis",
    quiver_scale=None,
    add_colorbar=True,
    show_trajectory=False,
    use_normalized_rate=False,
    max_arrow_length=None,
    max_arrow_std_factor=3.0,
    enable_hd_info_shuffle_gate=False,
    hd_info_n_shuffles=1000,
    hd_info_alpha=0.05,
    hd_info_gate_rule="p_and_above_null_mean",
    hd_info_random_seed=None,
    hd_info_star_thresholds=(0.05, 0.01, 0.001),
    hd_info_prior_alpha=0.0,
    hd_info_prior_beta_s=0.0,
    hd_info_use_smoothed_counts=True,
    hd_info_null_mode="timeshift",
    hd_time_shift_min_s=10.0,
    hd_time_shift_sessionwise=True,
    require_shuffle_pass=True,
    figsize_per_col=1.1,
    fig_height=3.3,
    session_start_frames=None,
):
    """Compute and plot HD vector fields for all cells in a 3 x N layout."""
    n_cells = len(spikes)
    if cell_order is None:
        cell_order = list(range(n_cells))
    else:
        cell_order = list(cell_order)

    n_cols = len(cell_order)
    if n_cols == 0:
        raise ValueError("cell_order must contain at least one cell index.")

    vector_output_all = [None] * n_cells
    per_cell_outputs = []

    for col_idx, cell_idx in enumerate(cell_order):
        if bad_timepoints_all is None:
            bad_tp = None
        elif isinstance(bad_timepoints_all, (list, tuple, np.ndarray)) and len(bad_timepoints_all) == n_cells:
            bad_tp = bad_timepoints_all[cell_idx]
        else:
            bad_tp = bad_timepoints_all

        out = analyze_hd_vector_field_single_moving(
            x_neural,
            y_neural,
            hd_angles_neural,
            spikes,
            speed,
            frame_rate,
            cell_idx=cell_idx,
            axes=None,
            width_real=width_real,
            height_real=height_real,
            bin_size=bin_size,
            kernel_size=kernel_size,
            filter_type=filter_type,
            speed_threshold=speed_threshold,
            min_duration_s=min_duration_s,
            merge_gap_s=merge_gap_s,
            bad_timepoints=bad_tp,
            refined_SS=refined_SS,
            all_CS_spikes=all_CS_spikes,
            min_spikes_per_bin=min_spikes_per_bin,
            min_visited_angle_bins=min_visited_angle_bins,
            min_visited_angle_separation_deg=min_visited_angle_separation_deg,
            min_occ_frames_per_angle=min_occ_frames_per_angle,
            hd_count_smooth_sigma=hd_count_smooth_sigma,
            min_pref_vector_strength=min_pref_vector_strength,
            preferred_angle_mode=preferred_angle_mode,
            pref_angle_std_threshold=pref_angle_std_threshold,
            hd_bin_size_deg=hd_bin_size_deg,
            direction_mode=direction_mode,
            travel_smooth_window=travel_smooth_window,
            travel_min_step=travel_min_step,
            display_cell_num=cell_idx + 1,
            is_first_column=(col_idx == 0),
            is_last_column=(col_idx == (n_cols - 1)),
            quiver_scale=quiver_scale,
            cmap=cmap,
            use_normalized_rate=use_normalized_rate,
            enable_hd_info_shuffle_gate=enable_hd_info_shuffle_gate,
            hd_info_n_shuffles=hd_info_n_shuffles,
            hd_info_alpha=hd_info_alpha,
            hd_info_gate_rule=hd_info_gate_rule,
            hd_info_random_seed=hd_info_random_seed,
            hd_info_star_thresholds=hd_info_star_thresholds,
            hd_info_prior_alpha=hd_info_prior_alpha,
            hd_info_prior_beta_s=hd_info_prior_beta_s,
            hd_info_use_smoothed_counts=hd_info_use_smoothed_counts,
            hd_info_null_mode=hd_info_null_mode,
            hd_time_shift_min_s=hd_time_shift_min_s,
            hd_time_shift_sessionwise=hd_time_shift_sessionwise,
            session_start_frames=session_start_frames,
        )
        vector_output_all[cell_idx] = out
        per_cell_outputs.append((cell_idx, out))

    global_rate_max = 0.0
    displayed_rates_all = []
    positive_rates = []
    rate_key = "pref_rate_norm_map" if use_normalized_rate else "pref_rate_hz_map"
    u_key = "u_norm_map" if use_normalized_rate else "u_map"
    v_key = "v_norm_map" if use_normalized_rate else "v_map"
    for _, out in per_cell_outputs:
        for key in ("all", "simple", "complex"):
            u_vals = np.asarray(out["types"][key][u_key], dtype=float)
            v_vals = np.asarray(out["types"][key][v_key], dtype=float)
            values = np.hypot(u_vals, v_vals)
            finite = values[np.isfinite(values)]
            if finite.size > 0:
                global_rate_max = max(global_rate_max, float(np.nanmax(finite)))
                displayed_rates_all.append(finite)
                pos = finite[finite > 0]
                if pos.size > 0:
                    positive_rates.append(pos)
    if global_rate_max <= 0:
        global_rate_max = 1.0

    cap_arrow_length = None
    pf_cap_mode = max_arrow_length.lower() if isinstance(max_arrow_length, str) else None
    use_pf_based_cap = pf_cap_mode in {"pf_peak", "pf_mean"}
    if use_pf_based_cap:
        pf_rate_values = []
        for cell_idx, out in per_cell_outputs:
            if pc_output_all is None or cell_idx >= len(pc_output_all):
                continue
            analysis = pc_output_all[cell_idx]
            if not isinstance(analysis, dict):
                continue

            params = analysis.get("params", {})
            pf_width = float(params.get("width_real", width_real))
            pf_height = float(params.get("height_real", height_real))
            row_to_mask = {
                "all": analysis.get("place_field_mask"),
                "simple": analysis.get("ss_place_field_mask"),
                "complex": analysis.get("cs_place_field_mask"),
            }

            x_centers = np.asarray(out.get("x_centers", []), dtype=float)
            y_centers = np.asarray(out.get("y_centers", []), dtype=float)
            if x_centers.size == 0 or y_centers.size == 0:
                continue
            Xc, Yc = np.meshgrid(x_centers, y_centers, indexing="ij")
            x_flat = Xc.ravel()
            y_flat = Yc.ravel()

            for type_key, pf_mask in row_to_mask.items():
                pf_mask = np.asarray(pf_mask, dtype=bool)
                if pf_mask.size == 0 or not np.any(pf_mask):
                    continue

                pf_bins = [
                    np.linspace(0.0, pf_width, pf_mask.shape[0] + 1),
                    np.linspace(0.0, pf_height, pf_mask.shape[1] + 1),
                ]
                in_pf_flat = _positions_in_place_field(x_flat, y_flat, pf_bins, pf_mask)
                in_pf = in_pf_flat.reshape(Xc.shape)

                type_data = out["types"][type_key]
                u_vals = np.asarray(type_data[u_key], dtype=float)
                v_vals = np.asarray(type_data[v_key], dtype=float)
                rates = np.hypot(u_vals, v_vals)
                valid_arrows = np.asarray(type_data["valid_bin_mask"], dtype=bool)
                use_mask = in_pf & valid_arrows & np.isfinite(rates)
                if np.any(use_mask):
                    pf_rate_values.append(np.asarray(rates[use_mask], dtype=float))

        if pf_rate_values:
            pf_rate_concat = np.concatenate(pf_rate_values)
            if pf_cap_mode == "pf_peak":
                cap_val = float(np.nanmax(pf_rate_concat))
            else:  # "pf_mean"
                cap_val = float(np.nanmean(pf_rate_concat))
            if np.isfinite(cap_val) and cap_val > 0:
                cap_arrow_length = cap_val
    elif max_arrow_length is not None:
        cap_val = float(max_arrow_length)
        if np.isfinite(cap_val) and cap_val > 0:
            cap_arrow_length = cap_val
    elif (
        max_arrow_std_factor is not None
        and displayed_rates_all
    ):
        # Auto-cap from all finite displayed spatial bins (raw or normalized mode).
        all_rates = np.concatenate(displayed_rates_all)
        rate_mean = float(np.nanmean(all_rates))
        rate_std = float(np.nanstd(all_rates))
        candidate = rate_mean + float(max_arrow_std_factor) * rate_std
        if np.isfinite(candidate) and candidate > 0:
            cap_arrow_length = candidate

    if cap_arrow_length is not None:
        global_rate_max = min(float(global_rate_max), float(cap_arrow_length))

    quiver_scale_use = quiver_scale
    if quiver_scale_use is None:
        if positive_rates:
            scale_rates = np.concatenate(positive_rates)
            if cap_arrow_length is not None:
                scale_rates = np.minimum(scale_rates, float(cap_arrow_length))
            ref_rate = float(np.nanpercentile(scale_rates, 90))
            if not np.isfinite(ref_rate) or ref_rate <= 0:
                ref_rate = float(global_rate_max)
        else:
            ref_rate = float(global_rate_max)
        if not np.isfinite(ref_rate) or ref_rate <= 0:
            ref_rate = 1.0
        quiver_scale_use = ref_rate / (0.45 * float(bin_size)) if bin_size > 0 else 1.0
        if not np.isfinite(quiver_scale_use) or quiver_scale_use <= 0:
            quiver_scale_use = 1.0

    fig_width = max(2.0, float(figsize_per_col) * n_cols)
    fig, axes = plt.subplots(
        3,
        n_cols,
        figsize=(fig_width, fig_height),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    row_mappables = {"all": None, "simple": None, "complex": None}
    for col_idx, (_, out) in enumerate(per_cell_outputs):
        place_field_contours = None
        if pc_output_all is not None and col_idx < len(per_cell_outputs):
            cell_idx = per_cell_outputs[col_idx][0]
            if cell_idx < len(pc_output_all):
                analysis = pc_output_all[cell_idx]
            else:
                analysis = None
            if isinstance(analysis, dict):
                params = analysis.get("params", {})
                plot_params = analysis.get("plot_params", {})
                pf_width = float(params.get("width_real", width_real))
                pf_height = float(params.get("height_real", height_real))
                simple_color = plot_params.get("simple_spike_color", "#026C80")
                complex_color = plot_params.get("complex_spike_color", "#EE9B00")
                place_field_contours = {
                    "all": {
                        "mask": analysis.get("place_field_mask"),
                        "width_real": pf_width,
                        "height_real": pf_height,
                        "color": "magenta",
                    },
                    "simple": {
                        "mask": analysis.get("ss_place_field_mask"),
                        "width_real": pf_width,
                        "height_real": pf_height,
                        "color": simple_color,
                    },
                    "complex": {
                        "mask": analysis.get("cs_place_field_mask"),
                        "width_real": pf_width,
                        "height_real": pf_height,
                        "color": complex_color,
                    },
                }

        mappables = plot_hd_vector_field_single_moving_results(
            out,
            axes[:, col_idx],
            quiver_scale=quiver_scale_use,
            cmap=cmap,
            rate_vmin=0.0,
            rate_vmax=global_rate_max,
            show_trajectory=show_trajectory,
            use_normalized_rate=use_normalized_rate,
            place_field_contours=place_field_contours,
            max_arrow_length=cap_arrow_length,
            require_shuffle_pass=require_shuffle_pass,
        )
        for key in row_mappables.keys():
            if mappables.get(key) is not None:
                row_mappables[key] = mappables[key]

    if add_colorbar and n_cols > 0 and any(v is not None for v in row_mappables.values()):
        norm = mcolors.Normalize(vmin=0.0, vmax=global_rate_max)
        row_keys = ["all", "simple", "complex"]
        for row_idx, row_key in enumerate(row_keys):
            cax = inset_axes(
                axes[row_idx, -1],
                width="5%",
                height="100%",
                loc="center right",
                bbox_to_anchor=(0.12, 0.0, 1, 1),
                bbox_transform=axes[row_idx, -1].transAxes,
                borderpad=0,
            )
            mappable = row_mappables[row_key]
            if mappable is None:
                mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            cbar = fig.colorbar(mappable, cax=cax)
            cbar.ax.tick_params(labelsize=5)
            cbar.set_label("Pref/mean" if use_normalized_rate else "Hz", fontsize=5)
            for tick in cbar.ax.get_yticklabels():
                tick.set_fontname("Arial")

    return vector_output_all, fig, axes


def _circular_gaussian_smooth_curve(curve, sigma=1.0):
    """Circularly smooth a 1D curve while respecting NaN values."""
    arr = np.asarray(curve, dtype=float)
    if arr.ndim != 1:
        raise ValueError("curve must be 1D.")
    if arr.size == 0:
        return arr.copy()
    if sigma is None or float(sigma) <= 0:
        return arr.copy()

    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.full_like(arr, np.nan, dtype=float)

    filled = np.where(finite, arr, 0.0)
    weights = finite.astype(float)
    smooth_vals = gaussian_filter1d(filled, sigma=float(sigma), mode="wrap")
    smooth_weights = gaussian_filter1d(weights, sigma=float(sigma), mode="wrap")
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.divide(
            smooth_vals,
            smooth_weights,
            out=np.full_like(smooth_vals, np.nan, dtype=float),
            where=smooth_weights > 1e-8,
        )
    return out


def _compute_vector_bin_pf_overlap_mask(vector_bins, pf_mask, pf_width, pf_height):
    """Map PF mask to vector bins using an any-overlap rule."""
    x_edges = np.asarray(vector_bins[0], dtype=float)
    y_edges = np.asarray(vector_bins[1], dtype=float)
    nx = len(x_edges) - 1
    ny = len(y_edges) - 1

    overlap_mask = np.zeros((nx, ny), dtype=bool)
    pf_mask = np.asarray(pf_mask, dtype=bool)
    if pf_mask.size == 0 or not np.any(pf_mask):
        return overlap_mask

    if pf_mask.ndim != 2:
        return overlap_mask

    npx, npy = pf_mask.shape
    pf_width = float(pf_width)
    pf_height = float(pf_height)
    if pf_width <= 0 or pf_height <= 0:
        return overlap_mask

    for ix in range(nx):
        x0, x1 = float(x_edges[ix]), float(x_edges[ix + 1])
        px0 = int(np.floor((x0 / pf_width) * npx))
        px1 = int(np.ceil((x1 / pf_width) * npx) - 1)
        px0 = int(np.clip(px0, 0, npx - 1))
        px1 = int(np.clip(px1, 0, npx - 1))
        if px1 < px0:
            continue

        for iy in range(ny):
            y0, y1 = float(y_edges[iy]), float(y_edges[iy + 1])
            py0 = int(np.floor((y0 / pf_height) * npy))
            py1 = int(np.ceil((y1 / pf_height) * npy) - 1)
            py0 = int(np.clip(py0, 0, npy - 1))
            py1 = int(np.clip(py1, 0, npy - 1))
            if py1 < py0:
                continue

            overlap_mask[ix, iy] = bool(np.any(pf_mask[px0 : px1 + 1, py0 : py1 + 1]))

    return overlap_mask


def _build_hd_place_field_contours_and_overlap(
    pc_output_all,
    cell_idx,
    vector_bins,
    width_real,
    height_real,
):
    """Build PF contour config and per-type overlap masks for vector bins."""
    nx = len(vector_bins[0]) - 1
    ny = len(vector_bins[1]) - 1
    empty_overlap = {
        "all": np.zeros((nx, ny), dtype=bool),
        "simple": np.zeros((nx, ny), dtype=bool),
        "complex": np.zeros((nx, ny), dtype=bool),
    }

    if pc_output_all is None or cell_idx >= len(pc_output_all):
        return None, empty_overlap

    analysis = pc_output_all[cell_idx]
    if not isinstance(analysis, dict):
        return None, empty_overlap

    params = analysis.get("params", {})
    plot_params = analysis.get("plot_params", {})
    pf_width = float(params.get("width_real", width_real))
    pf_height = float(params.get("height_real", height_real))
    simple_color = plot_params.get("simple_spike_color", "#026C80")
    complex_color = plot_params.get("complex_spike_color", "#EE9B00")

    place_field_contours = {
        "all": {
            "mask": analysis.get("place_field_mask"),
            "width_real": pf_width,
            "height_real": pf_height,
            "color": "magenta",
        },
        "simple": {
            "mask": analysis.get("ss_place_field_mask"),
            "width_real": pf_width,
            "height_real": pf_height,
            "color": simple_color,
        },
        "complex": {
            "mask": analysis.get("cs_place_field_mask"),
            "width_real": pf_width,
            "height_real": pf_height,
            "color": complex_color,
        },
    }

    overlap_masks = {}
    for type_key, contour_info in place_field_contours.items():
        overlap_masks[type_key] = _compute_vector_bin_pf_overlap_mask(
            vector_bins,
            contour_info.get("mask"),
            contour_info.get("width_real", width_real),
            contour_info.get("height_real", height_real),
        )

    return place_field_contours, overlap_masks


def plot_hd_vector_fields_single_cell_moving_with_bin_polar(
    x_neural,
    y_neural,
    hd_angles_neural,
    spikes,
    speed,
    frame_rate,
    cell_idx,
    refined_SS=None,
    all_CS_spikes=None,
    pc_output_all=None,
    width_real=35.5,
    height_real=20,
    bin_size=1.5,
    kernel_size=21,
    filter_type="boxcar",
    speed_threshold=2,
    min_duration_s=0.5,
    merge_gap_s=1.0,
    bad_timepoints=None,
    min_spikes_per_bin=20,
    min_visited_angle_bins=2,
    min_visited_angle_separation_deg=0.0,
    min_occ_frames_per_angle=2,
    hd_count_smooth_sigma=1.0,
    min_pref_vector_strength=0.0,
    preferred_angle_mode="peak",
    pref_angle_std_threshold=None,
    hd_bin_size_deg=30,
    direction_mode="head",
    travel_smooth_window=5,
    travel_min_step=0.0,
    cmap="viridis",
    quiver_scale=None,
    show_trajectory=False,
    use_normalized_rate=False,
    max_arrow_length=None,
    max_arrow_std_factor=3.0,
    polar_smooth_sigma=1.0,
    pf_bg_alpha=0.1,
    tuning_curve_color="green",
    tuning_style="line",
    mini_pref_arrow_length_mode="vector_strength",
    mini_pref_arrow_length_min=0.15,
    mini_pref_arrow_length_max=0.95,
    occupancy_style="line",
    show_hd_info_stars=True,
    enable_hd_info_shuffle_gate=False,
    hd_info_n_shuffles=1000,
    hd_info_alpha=0.05,
    hd_info_gate_rule="p_and_above_null_mean",
    hd_info_random_seed=None,
    hd_info_star_thresholds=(0.05, 0.01, 0.001),
    hd_info_prior_alpha=0.0,
    hd_info_prior_beta_s=0.0,
    hd_info_use_smoothed_counts=True,
    hd_info_null_mode="timeshift",
    hd_time_shift_min_s=10.0,
    hd_time_shift_sessionwise=True,
    vector_require_shuffle_pass=True,
    right_panel_width_mode="auto",
    figsize_base=(6.5, 5.0),
    save_path=None,
    return_data=True,
    session_start_frames=None,
):
    """
    Plot one cell as 3 rows with whole-arena polar tuning, vector maps,
    and a per-bin polar grid for each row.
    """
    occupancy_style = str(occupancy_style).lower()
    if occupancy_style not in {"line", "fan", "fan+line"}:
        raise ValueError(
            "occupancy_style must be one of: 'line', 'fan', 'fan+line'."
        )
    tuning_style = str(tuning_style).lower()
    if tuning_style not in {"line", "fan", "fan+line"}:
        raise ValueError(
            "tuning_style must be one of: 'line', 'fan', 'fan+line'."
        )
    try:
        tuning_curve_rgb = np.asarray(mcolors.to_rgb(tuning_curve_color), dtype=float)
        tuning_curve_color_use = tuple(tuning_curve_rgb.tolist())
    except ValueError:
        raise ValueError("Invalid tuning_curve_color.")
    mini_arrow_color = tuple(np.clip(tuning_curve_rgb * 0.7, 0.0, 1.0).tolist())

    mini_pref_arrow_length_mode = str(mini_pref_arrow_length_mode).lower()
    if mini_pref_arrow_length_mode not in {"fixed", "vector_strength", "selectivity", "tuning_si"}:
        raise ValueError(
            "mini_pref_arrow_length_mode must be one of: "
            "'fixed', 'vector_strength', 'selectivity', 'tuning_si'."
        )
    mini_pref_arrow_length_min = float(mini_pref_arrow_length_min)
    mini_pref_arrow_length_max = float(mini_pref_arrow_length_max)
    if not np.isfinite(mini_pref_arrow_length_min) or not np.isfinite(mini_pref_arrow_length_max):
        raise ValueError("mini_pref_arrow_length_min/max must be finite.")
    if mini_pref_arrow_length_min < 0:
        mini_pref_arrow_length_min = 0.0
    if mini_pref_arrow_length_max > 1:
        mini_pref_arrow_length_max = 1.0
    if mini_pref_arrow_length_min > mini_pref_arrow_length_max:
        mini_pref_arrow_length_min, mini_pref_arrow_length_max = (
            mini_pref_arrow_length_max,
            mini_pref_arrow_length_min,
        )

    results = analyze_hd_vector_field_single_moving(
        x_neural,
        y_neural,
        hd_angles_neural,
        spikes,
        speed,
        frame_rate,
        cell_idx=cell_idx,
        axes=None,
        width_real=width_real,
        height_real=height_real,
        bin_size=bin_size,
        kernel_size=kernel_size,
        filter_type=filter_type,
        speed_threshold=speed_threshold,
        min_duration_s=min_duration_s,
        merge_gap_s=merge_gap_s,
        bad_timepoints=bad_timepoints,
        refined_SS=refined_SS,
        all_CS_spikes=all_CS_spikes,
        min_spikes_per_bin=min_spikes_per_bin,
        min_visited_angle_bins=min_visited_angle_bins,
        min_visited_angle_separation_deg=min_visited_angle_separation_deg,
        min_occ_frames_per_angle=min_occ_frames_per_angle,
        hd_count_smooth_sigma=hd_count_smooth_sigma,
        min_pref_vector_strength=min_pref_vector_strength,
        preferred_angle_mode=preferred_angle_mode,
        pref_angle_std_threshold=pref_angle_std_threshold,
        hd_bin_size_deg=hd_bin_size_deg,
        direction_mode=direction_mode,
        travel_smooth_window=travel_smooth_window,
        travel_min_step=travel_min_step,
        display_cell_num=cell_idx + 1,
        is_first_column=True,
        is_last_column=True,
        quiver_scale=quiver_scale,
        cmap=cmap,
        use_normalized_rate=use_normalized_rate,
        enable_hd_info_shuffle_gate=enable_hd_info_shuffle_gate,
        hd_info_n_shuffles=hd_info_n_shuffles,
        hd_info_alpha=hd_info_alpha,
        hd_info_gate_rule=hd_info_gate_rule,
        hd_info_random_seed=hd_info_random_seed,
        hd_info_star_thresholds=hd_info_star_thresholds,
        hd_info_prior_alpha=hd_info_prior_alpha,
        hd_info_prior_beta_s=hd_info_prior_beta_s,
        hd_info_use_smoothed_counts=hd_info_use_smoothed_counts,
        hd_info_null_mode=hd_info_null_mode,
        hd_time_shift_min_s=hd_time_shift_min_s,
        hd_time_shift_sessionwise=hd_time_shift_sessionwise,
        session_start_frames=session_start_frames,
    )

    x_centers = np.asarray(results.get("x_centers", []), dtype=float)
    y_centers = np.asarray(results.get("y_centers", []), dtype=float)
    nx = x_centers.size
    ny = y_centers.size
    if nx == 0 or ny == 0:
        raise ValueError("No spatial bins available for plotting.")

    bins = results.get("bins")
    if bins is None or len(bins) != 2:
        raise ValueError("Missing spatial bins in analysis results.")

    place_field_contours, pf_overlap_masks = _build_hd_place_field_contours_and_overlap(
        pc_output_all,
        cell_idx,
        bins,
        width_real,
        height_real,
    )

    rate_key = "pref_rate_norm_map" if use_normalized_rate else "pref_rate_hz_map"
    displayed_rates_all = []
    positive_rates = []
    global_rate_max = 0.0
    for type_key in ("all", "simple", "complex"):
        rates = np.asarray(results["types"][type_key][rate_key], dtype=float)
        finite = rates[np.isfinite(rates)]
        if finite.size > 0:
            displayed_rates_all.append(finite)
            global_rate_max = max(global_rate_max, float(np.nanmax(finite)))
            pos = finite[finite > 0]
            if pos.size > 0:
                positive_rates.append(pos)
    if global_rate_max <= 0:
        global_rate_max = 1.0

    cap_arrow_length = None
    pf_cap_mode = max_arrow_length.lower() if isinstance(max_arrow_length, str) else None
    use_pf_based_cap = pf_cap_mode in {"pf_peak", "pf_mean"}
    if use_pf_based_cap and place_field_contours is not None:
        pf_rate_values = []
        for type_key in ("all", "simple", "complex"):
            type_data = results["types"][type_key]
            rates = np.asarray(type_data[rate_key], dtype=float)
            valid_arrows = np.asarray(type_data["valid_bin_mask"], dtype=bool)
            in_pf = np.asarray(pf_overlap_masks.get(type_key), dtype=bool)
            use_mask = in_pf & valid_arrows & np.isfinite(rates)
            if np.any(use_mask):
                pf_rate_values.append(np.asarray(rates[use_mask], dtype=float))
        if pf_rate_values:
            pf_rate_concat = np.concatenate(pf_rate_values)
            if pf_cap_mode == "pf_peak":
                cap_val = float(np.nanmax(pf_rate_concat))
            else:
                cap_val = float(np.nanmean(pf_rate_concat))
            if np.isfinite(cap_val) and cap_val > 0:
                cap_arrow_length = cap_val
    elif max_arrow_length is not None:
        cap_val = float(max_arrow_length)
        if np.isfinite(cap_val) and cap_val > 0:
            cap_arrow_length = cap_val
    elif max_arrow_std_factor is not None and displayed_rates_all:
        all_rates = np.concatenate(displayed_rates_all)
        rate_mean = float(np.nanmean(all_rates))
        rate_std = float(np.nanstd(all_rates))
        candidate = rate_mean + float(max_arrow_std_factor) * rate_std
        if np.isfinite(candidate) and candidate > 0:
            cap_arrow_length = candidate

    if cap_arrow_length is not None:
        global_rate_max = min(float(global_rate_max), float(cap_arrow_length))

    quiver_scale_use = quiver_scale
    if quiver_scale_use is None:
        if positive_rates:
            scale_rates = np.concatenate(positive_rates)
            if cap_arrow_length is not None:
                scale_rates = np.minimum(scale_rates, float(cap_arrow_length))
            ref_rate = float(np.nanpercentile(scale_rates, 90))
            if not np.isfinite(ref_rate) or ref_rate <= 0:
                ref_rate = float(global_rate_max)
        else:
            ref_rate = float(global_rate_max)
        if not np.isfinite(ref_rate) or ref_rate <= 0:
            ref_rate = 1.0
        quiver_scale_use = ref_rate / (0.45 * float(bin_size)) if bin_size > 0 else 1.0
        if not np.isfinite(quiver_scale_use) or quiver_scale_use <= 0:
            quiver_scale_use = 1.0

    def _prepare_polar_curve_summary(occ_counts_curve, occ_time_curve, tuning_raw_curve):
        occ_counts_curve = np.asarray(occ_counts_curve, dtype=float)
        occ_time_curve = np.asarray(occ_time_curve, dtype=float)
        tuning_raw_curve = np.asarray(tuning_raw_curve, dtype=float)

        if occ_counts_curve.shape != (hd_theta.size,):
            occ_counts_curve = np.zeros(hd_theta.size, dtype=float)
        if occ_time_curve.shape != (hd_theta.size,):
            occ_time_curve = np.zeros(hd_theta.size, dtype=float)
        if tuning_raw_curve.shape != (hd_theta.size,):
            tuning_raw_curve = np.full(hd_theta.size, np.nan, dtype=float)

        visited_angle_mask = occ_counts_curve >= float(min_occ_frames_per_angle)
        occ_curve_smooth = _circular_gaussian_smooth_curve(
            occ_counts_curve,
            sigma=polar_smooth_sigma,
        )
        tuning_curve_smooth = _circular_gaussian_smooth_curve(
            tuning_raw_curve,
            sigma=polar_smooth_sigma,
        )
        tuning_curve_smooth = np.asarray(tuning_curve_smooth, dtype=float)
        tuning_curve_smooth[~visited_angle_mask] = np.nan

        if np.any(np.isfinite(occ_curve_smooth)):
            occ_curve_smooth = np.clip(np.nan_to_num(occ_curve_smooth, nan=0.0), 0.0, None)
            occ_max = float(np.nanmax(occ_curve_smooth))
            if np.isfinite(occ_max) and occ_max > 0:
                occ_curve_norm = occ_curve_smooth / occ_max
            else:
                occ_curve_norm = np.zeros_like(occ_curve_smooth)
        else:
            occ_curve_norm = np.zeros_like(occ_counts_curve, dtype=float)

        tuning_curve_norm = np.full_like(tuning_curve_smooth, np.nan, dtype=float)
        tuning_max_rate_hz = np.nan
        tuning_selectivity_index = np.nan
        if np.any(np.isfinite(tuning_curve_smooth)):
            tuning_curve_use = np.clip(np.nan_to_num(tuning_curve_smooth, nan=0.0), 0.0, None)
            tuning_max = float(np.nanmax(tuning_curve_use))
            tuning_min = float(np.nanmin(tuning_curve_use))
            if np.isfinite(tuning_max) and tuning_max > 0:
                tuning_curve_norm = tuning_curve_use / tuning_max
                tuning_max_rate_hz = tuning_max
                denom = tuning_max + max(tuning_min, 0.0)
                if np.isfinite(denom) and denom > 0:
                    tuning_selectivity_index = (tuning_max - max(tuning_min, 0.0)) / denom
                else:
                    tuning_selectivity_index = 0.0
            else:
                tuning_curve_norm = np.zeros_like(tuning_curve_use)
                tuning_max_rate_hz = 0.0
                tuning_selectivity_index = 0.0

        if np.any(np.isfinite(occ_time_curve)):
            occ_max_time_s = float(np.nanmax(occ_time_curve))
        else:
            occ_max_time_s = np.nan

        return {
            "occ_counts_curve": occ_counts_curve,
            "occ_time_curve": occ_time_curve,
            "rate_hd_curve": tuning_raw_curve,
            "visited_angle_mask": visited_angle_mask,
            "occ_curve_smooth": occ_curve_smooth,
            "tuning_curve_smooth": tuning_curve_smooth,
            "occ_curve_norm": occ_curve_norm,
            "tuning_curve_norm": tuning_curve_norm,
            "tuning_max_rate_hz": tuning_max_rate_hz,
            "tuning_selectivity_index": tuning_selectivity_index,
            "occ_max_time_s": occ_max_time_s,
        }

    def _find_circular_true_clusters_local(mask_1d):
        mask_1d = np.asarray(mask_1d, dtype=bool).ravel()
        n = mask_1d.size
        if n == 0:
            return []
        true_idx = np.where(mask_1d)[0]
        if true_idx.size == 0:
            return []

        clusters = [[int(true_idx[0])]]
        for idx in true_idx[1:]:
            idx = int(idx)
            if idx == clusters[-1][-1] + 1:
                clusters[-1].append(idx)
            else:
                clusters.append([idx])

        if len(clusters) > 1 and clusters[0][0] == 0 and clusters[-1][-1] == (n - 1):
            merged = clusters[-1] + clusters[0]
            clusters = [merged] + clusters[1:-1]

        return [np.asarray(c, dtype=int) for c in clusters]

    def _summarize_overall_polar(type_key, type_data, rng_local):
        occ_counts_curve = np.asarray(
            type_data.get("global_occ_counts_curve", []),
            dtype=float,
        )
        spike_counts_curve = np.asarray(
            type_data.get("global_spike_counts_curve", []),
            dtype=float,
        )
        if occ_counts_curve.ndim != 1 or occ_counts_curve.size != hd_theta.size:
            occ_counts_map = np.asarray(type_data.get("occ_counts_hd_map", []), dtype=float)
            if occ_counts_map.ndim == 3:
                occ_counts_curve = np.nansum(occ_counts_map, axis=(0, 1))
            else:
                occ_counts_curve = np.zeros(hd_theta.size, dtype=float)
        if spike_counts_curve.ndim != 1 or spike_counts_curve.size != hd_theta.size:
            spike_counts_map = np.asarray(type_data.get("spike_counts_hd_map", []), dtype=float)
            if spike_counts_map.ndim == 3:
                spike_counts_curve = np.nansum(spike_counts_map, axis=(0, 1))
            else:
                spike_counts_curve = np.zeros(hd_theta.size, dtype=float)

        if occ_counts_curve.shape != (hd_theta.size,):
            occ_counts_curve = np.zeros(hd_theta.size, dtype=float)
        if spike_counts_curve.shape != (hd_theta.size,):
            spike_counts_curve = np.zeros(hd_theta.size, dtype=float)

        n_spikes_total = int(np.nansum(spike_counts_curve))
        occ_counts_for_rate = np.asarray(occ_counts_curve, dtype=float)
        spk_counts_for_rate = np.asarray(spike_counts_curve, dtype=float)
        if hd_count_smooth_sigma > 0:
            occ_counts_for_rate = gaussian_filter1d(
                occ_counts_for_rate,
                sigma=hd_count_smooth_sigma,
                mode="wrap",
            )
            spk_counts_for_rate = gaussian_filter1d(
                spk_counts_for_rate,
                sigma=hd_count_smooth_sigma,
                mode="wrap",
            )
        occ_time_for_rate = (
            occ_counts_for_rate / float(frame_rate)
            if frame_rate > 0
            else np.zeros(hd_theta.size, dtype=float)
        )
        visited_mask = occ_counts_curve >= int(min_occ_frames_per_angle)
        with np.errstate(divide="ignore", invalid="ignore"):
            rate_hd_curve = np.divide(
                spk_counts_for_rate,
                occ_time_for_rate,
                out=np.zeros(hd_theta.size, dtype=float),
                where=(occ_time_for_rate > 0) & visited_mask,
            )

        n_visited_angle_bins = int(np.sum(visited_mask))
        if n_visited_angle_bins >= 2:
            visited_centers_deg = hd_centers_deg[visited_mask]
            diffs = np.abs(
                (
                    visited_centers_deg[:, None]
                    - visited_centers_deg[None, :]
                    + 180.0
                )
                % 360.0
                - 180.0
            )
            max_sep_deg = float(np.nanmax(diffs))
        else:
            max_sep_deg = 0.0
        sep_valid = (
            max_sep_deg >= float(min_visited_angle_separation_deg)
            if min_visited_angle_separation_deg > 0
            else True
        )
        visited_angle_valid = (
            n_visited_angle_bins >= int(min_visited_angle_bins)
            and sep_valid
        )

        if np.any(visited_mask):
            mean_visited_rate = float(np.nanmean(rate_hd_curve[visited_mask]))
            std_visited_rate = float(np.nanstd(rate_hd_curve[visited_mask]))
        else:
            mean_visited_rate = 0.0
            std_visited_rate = 0.0
        if not np.isfinite(mean_visited_rate):
            mean_visited_rate = 0.0
        if not np.isfinite(std_visited_rate):
            std_visited_rate = 0.0

        weights_pref = np.asarray(rate_hd_curve, dtype=float)
        if weights_pref.size > 0:
            weights_pref = np.where(visited_mask, weights_pref, 0.0)
            weights_pref = np.clip(weights_pref, 0.0, None)
        sum_w = float(np.nansum(weights_pref))
        if sum_w > 0 and weights_pref.size == hd_theta.size:
            vec_x = float(np.nansum(weights_pref * np.cos(hd_theta)))
            vec_y = float(np.nansum(weights_pref * np.sin(hd_theta)))
            pref_vector_strength = float(np.hypot(vec_x, vec_y) / sum_w)
            preferred_dir_weighted_deg = np.rad2deg(np.arctan2(vec_y, vec_x)) % 360.0
        else:
            vec_x = 0.0
            vec_y = 0.0
            pref_vector_strength = 0.0
            preferred_dir_weighted_deg = np.nan

        hd_info_obs = type_data.get("global_hd_info_obs_bits_per_spike", np.nan)
        hd_info_null_mean = type_data.get(
            "global_hd_info_null_mean_bits_per_spike",
            np.nan,
        )
        hd_info_p = type_data.get("global_hd_info_p_value", np.nan)
        hd_info_sig = bool(type_data.get("global_hd_info_sig", False))
        hd_info_star_count = int(type_data.get("global_hd_info_star_count", 0))
        hd_info_star_text = str(type_data.get("global_hd_info_star_text", ""))
        if (
            n_spikes_total > int(min_spikes_per_bin)
            and not np.isfinite(hd_info_obs)
        ):
            if hd_info_use_smoothed_counts:
                occ_counts_for_info = np.asarray(occ_counts_for_rate, dtype=float)
                spk_counts_for_info = np.asarray(spk_counts_for_rate, dtype=float)
                null_smooth_sigma = float(hd_count_smooth_sigma)
            else:
                occ_counts_for_info = np.asarray(occ_counts_curve, dtype=float)
                spk_counts_for_info = np.asarray(spike_counts_curve, dtype=float)
                null_smooth_sigma = 0.0
            occ_time_for_info = (
                occ_counts_for_info / float(frame_rate)
                if frame_rate > 0
                else np.zeros(hd_theta.size, dtype=float)
            )
            hd_info_obs = _compute_hd_info_bits_per_spike(
                spk_counts_for_info,
                occ_time_for_info,
                visited_mask=visited_mask,
                prior_alpha=hd_info_prior_alpha,
                prior_beta_s=hd_info_prior_beta_s,
            )
            hd_info_null_mean, hd_info_p, _ = _run_hd_multinomial_null(
                observed_info_bits=hd_info_obs,
                n_spikes_total=n_spikes_total,
                occ_counts_prior=occ_counts_for_info,
                occ_time_for_rate=occ_time_for_info,
                visited_mask=visited_mask,
                n_shuffles=hd_info_n_shuffles,
                prior_alpha=hd_info_prior_alpha,
                prior_beta_s=hd_info_prior_beta_s,
                smooth_sigma=null_smooth_sigma,
                rng=rng_local,
            )
            hd_info_sig = bool(
                np.isfinite(hd_info_obs)
                and np.isfinite(hd_info_null_mean)
                and np.isfinite(hd_info_p)
                and (hd_info_p < float(hd_info_alpha))
                and (hd_info_obs > hd_info_null_mean)
            )
            hd_info_star_count = _p_to_star_count(
                hd_info_p,
                thresholds=hd_info_star_thresholds,
            )
            hd_info_star_text = _star_text_from_count(hd_info_star_count)

        spike_count_valid = bool(n_spikes_total > int(min_spikes_per_bin))
        pref_idx = -1
        preferred_dir_deg = np.nan
        if np.any(visited_mask):
            visited_idx = np.where(visited_mask)[0]
            if preferred_angle_mode == "peak":
                visited_rates = np.asarray(rate_hd_curve[visited_idx], dtype=float)
                if visited_rates.size > 0 and np.any(np.isfinite(visited_rates)):
                    peak_local_i = int(np.nanargmax(visited_rates))
                    pref_idx = int(visited_idx[peak_local_i])
                    preferred_dir_deg = float(hd_centers_deg[pref_idx])
            elif preferred_angle_mode == "peak_cluster_mean":
                pos_rate_mask = visited_mask & np.isfinite(rate_hd_curve) & (rate_hd_curve > 0)
                clusters = _find_circular_true_clusters_local(pos_rate_mask)
                if len(clusters) == 0:
                    visited_rates = np.asarray(rate_hd_curve[visited_idx], dtype=float)
                    if visited_rates.size > 0 and np.any(np.isfinite(visited_rates)):
                        peak_local_i = int(np.nanargmax(visited_rates))
                        pref_idx = int(visited_idx[peak_local_i])
                        preferred_dir_deg = float(hd_centers_deg[pref_idx])
                else:
                    best_cluster = None
                    best_peak = -np.inf
                    for cluster_idx in clusters:
                        cluster_rates = np.asarray(rate_hd_curve[cluster_idx], dtype=float)
                        if cluster_rates.size == 0 or not np.any(np.isfinite(cluster_rates)):
                            cluster_peak = -np.inf
                        else:
                            cluster_peak = float(np.nanmax(cluster_rates))
                        if cluster_peak > best_peak:
                            best_peak = cluster_peak
                            best_cluster = cluster_idx
                    if best_cluster is not None and best_cluster.size > 0:
                        cluster_rates = np.asarray(rate_hd_curve[best_cluster], dtype=float)
                        cluster_rates = np.clip(np.nan_to_num(cluster_rates, nan=0.0), 0.0, None)
                        cluster_centers_rad = hd_theta[best_cluster]
                        cluster_centers_deg = hd_centers_deg[best_cluster]
                        if float(np.nansum(cluster_rates)) > 0:
                            vec_x_c = float(np.nansum(cluster_rates * np.cos(cluster_centers_rad)))
                            vec_y_c = float(np.nansum(cluster_rates * np.sin(cluster_centers_rad)))
                            preferred_dir_deg = np.rad2deg(np.arctan2(vec_y_c, vec_x_c)) % 360.0
                        else:
                            preferred_dir_deg = float(np.nanmean(cluster_centers_deg))
                        circ_diff = np.abs(
                            (cluster_centers_deg - preferred_dir_deg + 180.0) % 360.0 - 180.0
                        )
                        pref_idx = int(best_cluster[int(np.argmin(circ_diff))])
            else:
                if sum_w > 0 and weights_pref.size == hd_theta.size:
                    preferred_dir_cont_deg = np.rad2deg(np.arctan2(vec_y, vec_x)) % 360.0
                    visited_centers_deg = hd_centers_deg[visited_idx]
                    circ_diff = np.abs(
                        (visited_centers_deg - preferred_dir_cont_deg + 180.0) % 360.0 - 180.0
                    )
                    nearest_i = int(np.argmin(circ_diff))
                    pref_idx = int(visited_idx[nearest_i])
                    preferred_dir_deg = float(visited_centers_deg[nearest_i])

        if pref_idx < 0 and np.isfinite(preferred_dir_deg):
            circ_diff_pref = np.abs(
                (hd_centers_deg - preferred_dir_deg + 180.0) % 360.0 - 180.0
            )
            pref_idx = int(np.argmin(circ_diff_pref))

        pref_rate_hz = 0.0
        pref_angle_pass = False
        pref_angle_threshold = np.nan
        if np.isfinite(preferred_dir_deg) and pref_idx >= 0:
            pref_rate_hz = float(rate_hd_curve[pref_idx]) if hd_theta.size > 0 else 0.0
            if np.isfinite(pref_rate_hz) and pref_rate_hz > 0:
                if pref_angle_std_threshold is None:
                    pref_angle_pass = True
                else:
                    pref_angle_threshold = (
                        mean_visited_rate + float(pref_angle_std_threshold) * std_visited_rate
                    )
                    pref_angle_pass = (
                        np.isfinite(pref_rate_hz)
                        and np.isfinite(pref_angle_threshold)
                        and (pref_rate_hz > pref_angle_threshold)
                    )
                pref_angle_pass = bool(pref_angle_pass) and (
                    float(pref_vector_strength) >= float(min_pref_vector_strength)
                )

        valid_panel = bool(spike_count_valid and visited_angle_valid and pref_angle_pass)
        if enable_hd_info_shuffle_gate:
            valid_panel = bool(valid_panel and hd_info_sig)

        summary = _prepare_polar_curve_summary(
            occ_counts_curve,
            occ_counts_curve / float(frame_rate) if frame_rate > 0 else np.zeros(hd_theta.size, dtype=float),
            rate_hd_curve,
        )
        summary.update(
            {
                "type_key": str(type_key),
                "n_spikes_total": int(n_spikes_total),
                "visited_angle_bin_count": int(n_visited_angle_bins),
                "visited_angle_max_sep_deg": float(max_sep_deg),
                "spike_count_valid": bool(spike_count_valid),
                "visited_angle_valid": bool(visited_angle_valid),
                "pref_angle_pass": bool(pref_angle_pass),
                "pref_angle_threshold": pref_angle_threshold,
                "pref_hd_deg": preferred_dir_deg,
                "pref_hd_deg_weighted": preferred_dir_weighted_deg,
                "pref_vector_strength": float(pref_vector_strength),
                "pref_rate_hz": float(pref_rate_hz) if np.isfinite(pref_rate_hz) else np.nan,
                "mean_visited_rate_hz": float(mean_visited_rate),
                "valid_panel": bool(valid_panel),
                "hd_info_obs_bits_per_spike": hd_info_obs,
                "hd_info_null_mean_bits_per_spike": hd_info_null_mean,
                "hd_info_p_value": hd_info_p,
                "hd_info_sig": bool(hd_info_sig),
                "hd_info_star_count": int(hd_info_star_count),
                "hd_info_star_text": str(hd_info_star_text),
            }
        )
        return summary

    fig_w_base, fig_h_base = figsize_base
    fig_height = max(4.0, float(fig_h_base))
    overall_polar_ratio = float(np.clip(float(height_real) / max(float(width_real), 1e-9), 0.35, 1.0))
    if str(right_panel_width_mode).lower() == "auto":
        # Make tiny polar panels close to square: width per bin ~= height per bin.
        row_h = fig_height / 3.0
        overall_polar_w_target = row_h
        left_w_target = row_h * (float(width_real) / max(float(height_real), 1e-6))
        right_w_target = row_h * (float(nx) / max(float(ny), 1.0))
        right_ratio = max(right_w_target / max(left_w_target, 1e-6), 0.2)
        fig_width = max(4.8, overall_polar_w_target + left_w_target + right_w_target + 1.4)
    else:
        try:
            right_ratio = float(right_panel_width_mode)
        except Exception:
            right_ratio = 2.0
        if not np.isfinite(right_ratio) or right_ratio <= 0:
            right_ratio = 2.0
        fig_width = max(4.8, float(fig_w_base) * (1.0 + overall_polar_ratio + 0.5 * right_ratio))

    fig = plt.figure(figsize=(fig_width, fig_height))
    outer = fig.add_gridspec(
        3,
        3,
        width_ratios=[overall_polar_ratio, 1.0, right_ratio],
        left=0.03,
        right=0.99,
        top=0.96,
        bottom=0.04,
        wspace=0.02,
        hspace=0.18,
    )

    row_info = [
        ("all", "All spikes"),
        ("simple", "Simple spikes"),
        ("complex", "Complex spikes"),
    ]
    axes_overall = np.empty(3, dtype=object)
    axes_main = np.empty(3, dtype=object)
    axes_polar = {}
    fig_aspect_h_over_w = fig_height / max(fig_width, 1e-9)
    for row_idx, (type_key, _) in enumerate(row_info):
        axes_overall[row_idx] = fig.add_subplot(outer[row_idx, 0], projection="polar")
        axes_main[row_idx] = fig.add_subplot(outer[row_idx, 1])
        right_bbox = outer[row_idx, 2].get_position(fig)
        # Build physically square tiles: width_frac and height_frac must account
        # for figure aspect because figure coordinates are normalized separately
        # along x and y.
        tile_h = min(
            right_bbox.height / max(ny, 1),
            right_bbox.width / max(nx * fig_aspect_h_over_w, 1e-9),
        )
        tile_w = tile_h * fig_aspect_h_over_w
        grid_w = tile_w * nx
        grid_h = tile_h * ny
        x0 = right_bbox.x0 + 0.5 * (right_bbox.width - grid_w)
        y0 = right_bbox.y0 + 0.5 * (right_bbox.height - grid_h)
        axes_grid = np.empty((ny, nx), dtype=object)
        for iy in range(ny):
            for ix in range(nx):
                axp = fig.add_axes(
                    [
                        x0 + ix * tile_w,
                        y0 + iy * tile_h,
                        tile_w,
                        tile_h,
                    ],
                    projection="polar",
                )
                axes_grid[iy, ix] = axp
        axes_polar[type_key] = axes_grid

    hd_centers_deg = np.asarray(
        results.get("params", {}).get(
            "hd_bin_centers_deg",
            0.5
            * (
                np.asarray(results.get("params", {}).get("hd_bin_edges_deg", [0.0, 360.0]))[:-1]
                + np.asarray(results.get("params", {}).get("hd_bin_edges_deg", [0.0, 360.0]))[1:]
            ),
        ),
        dtype=float,
    )
    hd_theta = np.deg2rad(hd_centers_deg)
    hd_theta_loop = np.concatenate([hd_theta, hd_theta[:1]]) if hd_theta.size else np.array([0.0])
    hd_edges_deg = np.asarray(
        results.get("params", {}).get("hd_bin_edges_deg", []),
        dtype=float,
    )
    if hd_edges_deg.size == (hd_theta.size + 1):
        hd_bin_width_rad = np.deg2rad(np.diff(hd_edges_deg))
    else:
        default_w = (2.0 * np.pi / max(hd_theta.size, 1))
        hd_bin_width_rad = np.full(hd_theta.shape, default_w, dtype=float)

    polar_data = {}
    overall_polar_data = {}
    for type_key, _ in row_info:
        type_data = results["types"][type_key]
        occ_raw = np.asarray(type_data.get("occ_counts_hd_map", []), dtype=float)
        occ_time_raw = np.asarray(type_data.get("occ_time_hd_map", []), dtype=float)
        tuning_raw = np.asarray(type_data.get("rate_hd_map", []), dtype=float)
        if occ_raw.shape != tuning_raw.shape or occ_raw.ndim != 3:
            occ_raw = np.zeros((nx, ny, hd_theta.size), dtype=float)
            occ_time_raw = np.zeros((nx, ny, hd_theta.size), dtype=float)
            tuning_raw = np.full((nx, ny, hd_theta.size), np.nan, dtype=float)
        if occ_time_raw.shape != tuning_raw.shape:
            occ_time_raw = np.zeros((nx, ny, hd_theta.size), dtype=float)

        occ_sm = np.full_like(occ_raw, np.nan, dtype=float)
        tuning_sm = np.full_like(tuning_raw, np.nan, dtype=float)
        for ix in range(nx):
            for iy in range(ny):
                occ_sm[ix, iy, :] = _circular_gaussian_smooth_curve(
                    occ_raw[ix, iy, :],
                    sigma=polar_smooth_sigma,
                )
                tuning_curve_sm = _circular_gaussian_smooth_curve(
                    tuning_raw[ix, iy, :],
                    sigma=polar_smooth_sigma,
                )
                # Keep tuning constrained to truly visited angle bins so smoothing
                # does not leak non-zero values into unvisited directions.
                visited_mask_curve = (
                    np.asarray(occ_raw[ix, iy, :], dtype=float)
                    >= float(min_occ_frames_per_angle)
                )
                tuning_curve_sm = np.asarray(tuning_curve_sm, dtype=float)
                tuning_curve_sm[~visited_mask_curve] = np.nan
                tuning_sm[ix, iy, :] = tuning_curve_sm

        tuning_curve_norm = np.full_like(tuning_sm, np.nan, dtype=float)
        occ_curve_norm = np.full_like(occ_sm, np.nan, dtype=float)
        tuning_max_rate_map = np.full((nx, ny), np.nan, dtype=float)
        tuning_selectivity_index_map = np.full((nx, ny), np.nan, dtype=float)
        occ_max_time_s_map = np.full((nx, ny), np.nan, dtype=float)
        for ix in range(nx):
            for iy in range(ny):
                tune_curve = np.asarray(tuning_sm[ix, iy, :], dtype=float)
                occ_curve = np.asarray(occ_sm[ix, iy, :], dtype=float)
                occ_time_curve = np.asarray(occ_time_raw[ix, iy, :], dtype=float)

                if tune_curve.size > 0 and np.any(np.isfinite(tune_curve)):
                    tune_curve = np.clip(np.nan_to_num(tune_curve, nan=0.0), 0.0, None)
                    tune_max = float(np.nanmax(tune_curve))
                    tune_min = float(np.nanmin(tune_curve))
                    if np.isfinite(tune_max) and tune_max > 0:
                        tuning_curve_norm[ix, iy, :] = tune_curve / tune_max
                        tuning_max_rate_map[ix, iy] = tune_max
                        denom = tune_max + max(tune_min, 0.0)
                        if np.isfinite(denom) and denom > 0:
                            tuning_selectivity_index_map[ix, iy] = (
                                (tune_max - max(tune_min, 0.0)) / denom
                            )
                        else:
                            tuning_selectivity_index_map[ix, iy] = 0.0
                    else:
                        tuning_curve_norm[ix, iy, :] = np.zeros_like(tune_curve)
                        tuning_max_rate_map[ix, iy] = 0.0
                        tuning_selectivity_index_map[ix, iy] = 0.0

                if occ_curve.size > 0 and np.any(np.isfinite(occ_curve)):
                    occ_curve = np.clip(np.nan_to_num(occ_curve, nan=0.0), 0.0, None)
                    occ_max = float(np.nanmax(occ_curve))
                    if np.isfinite(occ_max) and occ_max > 0:
                        occ_curve_norm[ix, iy, :] = occ_curve / occ_max
                    else:
                        occ_curve_norm[ix, iy, :] = np.zeros_like(occ_curve)
                if occ_time_curve.size > 0 and np.any(np.isfinite(occ_time_curve)):
                    occ_max_time_s_map[ix, iy] = float(np.nanmax(occ_time_curve))
                else:
                    occ_max_time_s_map[ix, iy] = 0.0

        pref_hd_deg_map = np.asarray(type_data.get("pref_hd_deg_map"), dtype=float)
        if pref_hd_deg_map.shape != (nx, ny):
            pref_hd_deg_map = np.full((nx, ny), np.nan, dtype=float)
        pref_hd_deg_weighted_map = np.asarray(
            type_data.get("pref_hd_deg_weighted_map"),
            dtype=float,
        )
        if pref_hd_deg_weighted_map.shape != (nx, ny):
            pref_hd_deg_weighted_map = pref_hd_deg_map.copy()
        pref_vector_strength_map = np.asarray(
            type_data.get("pref_vector_strength_map"),
            dtype=float,
        )
        if pref_vector_strength_map.shape != (nx, ny):
            pref_vector_strength_map = np.full((nx, ny), np.nan, dtype=float)
        hd_info_star_text_map = np.asarray(
            type_data.get("hd_info_star_text_map"),
            dtype=object,
        )
        if hd_info_star_text_map.shape != (nx, ny):
            hd_info_star_text_map = np.full((nx, ny), "", dtype=object)
        hd_info_star_count_map = np.asarray(
            type_data.get("hd_info_star_count_map"),
            dtype=int,
        )
        if hd_info_star_count_map.shape != (nx, ny):
            hd_info_star_count_map = np.zeros((nx, ny), dtype=int)
        hd_info_p_value_map = np.asarray(
            type_data.get("hd_info_p_value_map"),
            dtype=float,
        )
        if hd_info_p_value_map.shape != (nx, ny):
            hd_info_p_value_map = np.full((nx, ny), np.nan, dtype=float)
        hd_info_sig_mask = np.asarray(
            type_data.get("hd_info_sig_mask"),
            dtype=bool,
        )
        if hd_info_sig_mask.shape != (nx, ny):
            hd_info_sig_mask = np.zeros((nx, ny), dtype=bool)

        polar_data[type_key] = {
            "occ_counts_hd_map": occ_raw,
            "occ_time_hd_map": occ_time_raw,
            "rate_hd_map": tuning_raw,
            "occ_curve_smooth_map": occ_sm,
            "tuning_curve_smooth_map": tuning_sm,
            "occ_curve_norm_map": occ_curve_norm,
            "tuning_curve_norm_map": tuning_curve_norm,
            "tuning_max_rate_hz_map": tuning_max_rate_map,
            "tuning_selectivity_index_map": tuning_selectivity_index_map,
            "occ_max_time_s_map": occ_max_time_s_map,
            "pref_hd_deg_map": pref_hd_deg_map,
            "pref_hd_deg_weighted_map": pref_hd_deg_weighted_map,
            "pref_vector_strength_map": pref_vector_strength_map,
            "hd_info_star_text_map": hd_info_star_text_map,
            "hd_info_star_count_map": hd_info_star_count_map,
            "hd_info_p_value_map": hd_info_p_value_map,
            "hd_info_sig_mask": hd_info_sig_mask,
            "spike_count_valid_mask": np.asarray(type_data.get("spike_count_valid_mask"), dtype=bool),
            "visited_angle_valid_mask": np.asarray(type_data.get("visited_angle_valid_mask"), dtype=bool),
            "pref_angle_pass_mask": np.asarray(type_data.get("pref_angle_pass_mask"), dtype=bool),
            "valid_bin_mask": np.asarray(type_data.get("valid_bin_mask"), dtype=bool),
        }

        seed_base = results.get("params", {}).get("hd_info_random_seed", None)
        if seed_base is None:
            seed_base = int(results.get("cell_id", 0)) + 1
        type_seed_offsets = {"all": 101, "simple": 211, "complex": 307}
        rng_overall = np.random.default_rng(int(seed_base) + int(type_seed_offsets.get(type_key, 0)))
        overall_polar_data[type_key] = _summarize_overall_polar(
            type_key,
            type_data,
            rng_local=rng_overall,
        )

    def _mini_arrow_metric(vector_strength_val, selectivity_val):
        if mini_pref_arrow_length_mode == "fixed":
            metric = 1.0
        elif mini_pref_arrow_length_mode == "vector_strength":
            metric = vector_strength_val
        else:  # "selectivity" / "tuning_si"
            metric = selectivity_val
        if not np.isfinite(metric):
            metric = 0.0
        return float(np.clip(metric, 0.0, 1.0))

    # Force the left vector-map arrows to match mini-arrow definition exactly:
    # same preferred angle and same length mapping.
    results_for_vector_plot = copy.deepcopy(results)
    results_for_vector_plot["plot_params"]["is_first_column"] = False
    for type_key, _ in row_info:
        pdata = polar_data[type_key]
        pref_hd_deg_map = np.asarray(pdata["pref_hd_deg_weighted_map"], dtype=float)
        pref_strength_map = np.asarray(pdata["pref_vector_strength_map"], dtype=float)
        tuning_si_map = np.asarray(pdata["tuning_selectivity_index_map"], dtype=float)
        spike_ok = np.asarray(pdata["spike_count_valid_mask"], dtype=bool)
        visited_ok = np.asarray(pdata["visited_angle_valid_mask"], dtype=bool)
        if vector_require_shuffle_pass:
            sig_ok = np.asarray(pdata["hd_info_sig_mask"], dtype=bool)
        else:
            sig_ok = np.ones_like(spike_ok, dtype=bool)
        valid_mask = spike_ok & visited_ok & sig_ok

        arrow_len_map = np.full_like(pref_hd_deg_map, np.nan, dtype=float)
        u_map_mini = np.full_like(pref_hd_deg_map, np.nan, dtype=float)
        v_map_mini = np.full_like(pref_hd_deg_map, np.nan, dtype=float)
        for ix in range(nx):
            for iy in range(ny):
                if not valid_mask[ix, iy]:
                    continue
                theta_deg = pref_hd_deg_map[ix, iy]
                if not np.isfinite(theta_deg):
                    continue
                length_metric = _mini_arrow_metric(
                    pref_strength_map[ix, iy],
                    tuning_si_map[ix, iy],
                )
                arrow_r = float(
                    mini_pref_arrow_length_min
                    + (mini_pref_arrow_length_max - mini_pref_arrow_length_min)
                    * length_metric
                )
                theta_rad = np.deg2rad(theta_deg % 360.0)
                arrow_len_map[ix, iy] = arrow_r
                u_map_mini[ix, iy] = arrow_r * np.cos(theta_rad)
                v_map_mini[ix, iy] = arrow_r * np.sin(theta_rad)

        results_for_vector_plot["types"][type_key]["pref_rate_hz_map"] = arrow_len_map.copy()
        results_for_vector_plot["types"][type_key]["pref_rate_norm_map"] = arrow_len_map.copy()
        results_for_vector_plot["types"][type_key]["u_map"] = u_map_mini.copy()
        results_for_vector_plot["types"][type_key]["v_map"] = v_map_mini.copy()
        results_for_vector_plot["types"][type_key]["u_norm_map"] = u_map_mini.copy()
        results_for_vector_plot["types"][type_key]["v_norm_map"] = v_map_mini.copy()
        results_for_vector_plot["types"][type_key]["valid_bin_mask"] = valid_mask.copy()

    quiver_scale_mini = quiver_scale
    if quiver_scale_mini is None:
        ref_len = float(max(mini_pref_arrow_length_max, 1e-6))
        quiver_scale_mini = ref_len / (0.45 * float(bin_size)) if bin_size > 0 else 1.0

    plot_hd_vector_field_single_moving_results(
        results_for_vector_plot,
        axes_main,
        quiver_scale=quiver_scale_mini,
        cmap=cmap,
        rate_vmin=0.0,
        rate_vmax=float(max(mini_pref_arrow_length_max, 1.0)),
        show_trajectory=show_trajectory,
        use_normalized_rate=False,
        place_field_contours=place_field_contours,
        max_arrow_length=None,
        require_shuffle_pass=vector_require_shuffle_pass,
    )

    for row_idx, (type_key, row_label) in enumerate(row_info):
        axp = axes_overall[row_idx]
        pdata = overall_polar_data[type_key]
        occ_curve = np.asarray(pdata["occ_curve_norm"], dtype=float)
        occ_counts_curve = np.asarray(pdata["occ_counts_curve"], dtype=float)
        visited_angle_mask = np.asarray(pdata["visited_angle_mask"], dtype=bool)
        if occupancy_style in {"fan", "fan+line"} and hd_theta.size > 0:
            if occ_curve.size > 0 and np.any(np.isfinite(occ_curve)):
                fan_heights = np.clip(np.nan_to_num(occ_curve, nan=0.0), 0.0, 1.0)
            else:
                fan_heights = np.zeros(hd_theta.size, dtype=float)
            fan_heights = np.where(visited_angle_mask, fan_heights, 0.0)
            fan_colors = [
                (0.50, 0.50, 0.50, float(0.18 + 0.45 * h))
                for h in fan_heights
            ]
            axp.bar(
                hd_theta,
                fan_heights,
                width=hd_bin_width_rad,
                bottom=0.0,
                align="center",
                color=fan_colors,
                edgecolor="none",
                linewidth=0.0,
                zorder=1,
            )
        if occupancy_style in {"line", "fan+line"}:
            if occ_curve.size > 0 and np.any(np.isfinite(occ_curve)):
                occ_curve_plot = np.clip(np.nan_to_num(occ_curve, nan=0.0), 0.0, 1.0)
                occ_plot_loop = np.concatenate([occ_curve_plot, occ_curve_plot[:1]])
                axp.plot(hd_theta_loop, occ_plot_loop, color="#7a7a7a", linewidth=0.35)

        show_tuning = bool(pdata["spike_count_valid"]) and bool(pdata["visited_angle_valid"])
        if show_tuning:
            tune_curve = np.asarray(pdata["tuning_curve_norm"], dtype=float)
            if tune_curve.size > 0 and np.any(np.isfinite(tune_curve)):
                tune_curve = np.clip(np.nan_to_num(tune_curve, nan=0.0), 0.0, 1.0)
                if tuning_style in {"fan", "fan+line"}:
                    tuning_fan_colors = [
                        (
                            tuning_curve_color_use[0],
                            tuning_curve_color_use[1],
                            tuning_curve_color_use[2],
                            float(0.10 + 0.50 * h),
                        )
                        for h in tune_curve
                    ]
                    axp.bar(
                        hd_theta,
                        tune_curve,
                        width=hd_bin_width_rad,
                        bottom=0.0,
                        align="center",
                        color=tuning_fan_colors,
                        edgecolor="none",
                        linewidth=0.0,
                        zorder=4,
                    )
                if tuning_style in {"line", "fan+line"}:
                    tune_loop = np.concatenate([tune_curve, tune_curve[:1]])
                    axp.plot(
                        hd_theta_loop,
                        tune_loop,
                        color=tuning_curve_color_use,
                        linewidth=0.5,
                        zorder=5,
                    )
                theta_pref_deg = pdata.get("pref_hd_deg_weighted", np.nan)
                if not np.isfinite(theta_pref_deg):
                    theta_pref_deg = pdata.get("pref_hd_deg", np.nan)
                if np.isfinite(theta_pref_deg):
                    theta_pref = np.deg2rad(float(theta_pref_deg) % 360.0)
                    length_metric = _mini_arrow_metric(
                        pdata.get("pref_vector_strength", np.nan),
                        pdata.get("tuning_selectivity_index", np.nan),
                    )
                    arrow_r = float(
                        mini_pref_arrow_length_min
                        + (mini_pref_arrow_length_max - mini_pref_arrow_length_min)
                        * length_metric
                    )
                    ann = axp.annotate(
                        "",
                        xy=(theta_pref, arrow_r),
                        xytext=(theta_pref, 0.0),
                        arrowprops=dict(
                            arrowstyle="->",
                            color=mini_arrow_color,
                            linewidth=0.5,
                            mutation_scale=2.2,
                            shrinkA=0.0,
                            shrinkB=0.0,
                        ),
                        zorder=12,
                    )
                    ann.set_clip_on(False)
                max_rate_val = pdata.get("tuning_max_rate_hz", np.nan)
                if np.isfinite(max_rate_val):
                    rate_label = f"{float(max_rate_val):.1f}Hz"
                    if show_hd_info_stars:
                        star_txt_local = str(pdata.get("hd_info_star_text", ""))
                        if star_txt_local:
                            rate_label = f"{rate_label} {star_txt_local}"
                    axp.text(
                        0.02,
                        0.98,
                        rate_label,
                        transform=axp.transAxes,
                        ha="left",
                        va="top",
                        fontsize=3.2,
                        color=tuning_curve_color_use,
                    )
        max_occ_time_val = pdata.get("occ_max_time_s", np.nan)
        if np.isfinite(max_occ_time_val):
            axp.text(
                0.02,
                0.02,
                f"{float(max_occ_time_val):.2f}s",
                transform=axp.transAxes,
                ha="left",
                va="bottom",
                fontsize=3.2,
                color="#666666",
            )
        axp.scatter(
            [0.0],
            [0.0],
            s=0.8,
            c="black",
            alpha=0.8,
            linewidths=0.0,
            edgecolors="none",
            zorder=13,
        )
        axp.set_ylim(0.0, 1.0)
        axp.grid(False)
        axp.set_xticks([])
        axp.set_yticks([])
        axp.spines["polar"].set_linewidth(0.25)
        axp.spines["polar"].set_color((0.0, 0.0, 0.0, 0.2))
        axp.text(
            -0.22,
            0.5,
            row_label,
            transform=axp.transAxes,
            rotation=90,
            va="center",
            ha="right",
            fontsize=6,
            fontname="Arial",
        )
        if row_idx == 0:
            axp.set_title("Whole arena", fontsize=6, fontname="Arial")

    for type_key, _ in row_info:
        axes_grid = axes_polar[type_key]
        pdata = polar_data[type_key]
        occ_norm = pdata["occ_curve_norm_map"]
        occ_raw = pdata["occ_counts_hd_map"]
        tuning_norm = pdata["tuning_curve_norm_map"]
        tuning_max_rate = pdata["tuning_max_rate_hz_map"]
        tuning_si = pdata["tuning_selectivity_index_map"]
        occ_max_time_s = pdata["occ_max_time_s_map"]
        pref_hd_deg_map = pdata["pref_hd_deg_weighted_map"]
        pref_strength_map = pdata["pref_vector_strength_map"]
        star_text_map = pdata["hd_info_star_text_map"]
        spike_ok = pdata["spike_count_valid_mask"]
        visited_ok = pdata["visited_angle_valid_mask"]
        overlap_mask = np.asarray(pf_overlap_masks.get(type_key), dtype=bool)
        contour_color = (
            place_field_contours.get(type_key, {}).get("color", "magenta")
            if isinstance(place_field_contours, dict)
            else "magenta"
        )

        for ix in range(nx):
            for iy in range(ny):
                axp = axes_grid[iy, ix]

                if overlap_mask.shape == (nx, ny) and overlap_mask[ix, iy]:
                    axp.set_facecolor(mcolors.to_rgba(contour_color, alpha=float(pf_bg_alpha)))
                else:
                    axp.set_facecolor((1.0, 1.0, 1.0, 1.0))

                occ_curve = occ_norm[ix, iy, :]
                occ_counts_curve = np.asarray(occ_raw[ix, iy, :], dtype=float)
                visited_angle_mask = occ_counts_curve >= float(min_occ_frames_per_angle)
                if occupancy_style in {"fan", "fan+line"} and hd_theta.size > 0:
                    # Fan encodes occupancy magnitude (normalized 0-1) for visited bins.
                    if occ_curve.size > 0 and np.any(np.isfinite(occ_curve)):
                        fan_heights = np.clip(np.nan_to_num(occ_curve, nan=0.0), 0.0, 1.0)
                    else:
                        fan_heights = np.zeros(hd_theta.size, dtype=float)
                    fan_heights = np.where(visited_angle_mask, fan_heights, 0.0)
                    fan_colors = [
                        (0.50, 0.50, 0.50, float(0.18 + 0.45 * h))
                        for h in fan_heights
                    ]
                    axp.bar(
                        hd_theta,
                        fan_heights,
                        width=hd_bin_width_rad,
                        bottom=0.0,
                        align="center",
                        color=fan_colors,
                        edgecolor="none",
                        linewidth=0.0,
                        zorder=1,
                    )
                if occupancy_style in {"line", "fan+line"}:
                    if occ_curve.size > 0 and np.any(np.isfinite(occ_curve)):
                        occ_curve = np.clip(np.nan_to_num(occ_curve, nan=0.0), 0.0, 1.0)
                        occ_plot_loop = np.concatenate([occ_curve, occ_curve[:1]])
                        axp.plot(hd_theta_loop, occ_plot_loop, color="#7a7a7a", linewidth=0.35)

                show_tuning = bool(spike_ok[ix, iy]) and bool(visited_ok[ix, iy])
                if show_tuning:
                    tune_curve = tuning_norm[ix, iy, :]
                    if tune_curve.size > 0 and np.any(np.isfinite(tune_curve)):
                        tune_curve = np.clip(np.nan_to_num(tune_curve, nan=0.0), 0.0, 1.0)
                        if tuning_style in {"fan", "fan+line"}:
                            tuning_fan_colors = [
                                (
                                    tuning_curve_color_use[0],
                                    tuning_curve_color_use[1],
                                    tuning_curve_color_use[2],
                                    float(0.10 + 0.50 * h),
                                )
                                for h in tune_curve
                            ]
                            axp.bar(
                                hd_theta,
                                tune_curve,
                                width=hd_bin_width_rad,
                                bottom=0.0,
                                align="center",
                                color=tuning_fan_colors,
                                edgecolor="none",
                                linewidth=0.0,
                                zorder=4,
                            )
                        if tuning_style in {"line", "fan+line"}:
                            tune_loop = np.concatenate([tune_curve, tune_curve[:1]])
                            axp.plot(
                                hd_theta_loop,
                                tune_loop,
                                color=tuning_curve_color_use,
                                linewidth=0.45,
                                zorder=5,
                            )
                        theta_pref_deg = pref_hd_deg_map[ix, iy]
                        if np.isfinite(theta_pref_deg):
                            theta_pref = np.deg2rad(theta_pref_deg % 360.0)
                            length_metric = _mini_arrow_metric(
                                pref_strength_map[ix, iy],
                                tuning_si[ix, iy],
                            )
                            arrow_r = float(
                                mini_pref_arrow_length_min
                                + (mini_pref_arrow_length_max - mini_pref_arrow_length_min)
                                * length_metric
                            )
                            ann = axp.annotate(
                                "",
                                xy=(theta_pref, arrow_r),
                                xytext=(theta_pref, 0.0),
                                arrowprops=dict(
                                    arrowstyle="->",
                                    color=mini_arrow_color,
                                    linewidth=0.45,
                                    mutation_scale=2.0,
                                    shrinkA=0.0,
                                    shrinkB=0.0,
                                ),
                                zorder=12,
                            )
                            ann.set_clip_on(False)
                        max_rate_val = tuning_max_rate[ix, iy]
                        if np.isfinite(max_rate_val):
                            rate_label = f"{max_rate_val:.1f}Hz"
                            if show_hd_info_stars:
                                star_txt_local = str(star_text_map[ix, iy])
                                if star_txt_local:
                                    rate_label = f"{rate_label} {star_txt_local}"
                            axp.text(
                                0.02,
                                0.98,
                                rate_label,
                                transform=axp.transAxes,
                                ha="left",
                                va="top",
                                fontsize=2.8,
                                color=tuning_curve_color_use,
                            )
                max_occ_time_val = occ_max_time_s[ix, iy]
                if np.isfinite(max_occ_time_val):
                    axp.text(
                        0.02,
                        0.02,
                        f"{max_occ_time_val:.2f}s",
                        transform=axp.transAxes,
                        ha="left",
                        va="bottom",
                        fontsize=2.8,
                        color="#666666",
                    )
                axp.scatter(
                    [0.0],
                    [0.0],
                    s=0.6,
                    c="black",
                    alpha=0.8,
                    linewidths=0.0,
                    edgecolors="none",
                    zorder=13,
                )
                axp.set_ylim(0.0, 1.0)
                axp.grid(False)
                axp.set_xticks([])
                axp.set_yticks([])
                axp.spines["polar"].set_linewidth(0.2)
                axp.spines["polar"].set_color((0.0, 0.0, 0.0, 0.2))

    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")

    result_dict = None
    if return_data:
        result_dict = {
            "cell_id": int(cell_idx),
            "analysis": results,
            "hd_bin_centers_deg": hd_centers_deg,
            "pf_overlap_masks": pf_overlap_masks,
            "overall_polar_data": overall_polar_data,
            "polar_data": polar_data,
            "max_arrow_length_used": cap_arrow_length,
            "quiver_scale_used": quiver_scale_mini,
            "mini_pref_arrow_length_mode": mini_pref_arrow_length_mode,
            "mini_pref_arrow_length_min": mini_pref_arrow_length_min,
            "mini_pref_arrow_length_max": mini_pref_arrow_length_max,
            "tuning_curve_color": tuning_curve_color_use,
            "mini_pref_arrow_color": mini_arrow_color,
            "tuning_style": tuning_style,
            "show_hd_info_stars": bool(show_hd_info_stars),
        }

    return fig, axes_main, axes_polar, result_dict


def plot_hd_vector_fields_all_cells_moving_with_bin_polar(
    x_neural,
    y_neural,
    hd_angles_neural,
    spikes,
    speed,
    frame_rate,
    refined_SS=None,
    all_CS_spikes=None,
    pc_output_all=None,
    cell_order=None,
    width_real=35.5,
    height_real=20,
    bin_size=1.5,
    kernel_size=21,
    filter_type="boxcar",
    speed_threshold=2,
    min_duration_s=0.5,
    merge_gap_s=1.0,
    bad_timepoints_all=None,
    min_spikes_per_bin=20,
    min_visited_angle_bins=2,
    min_visited_angle_separation_deg=0.0,
    min_occ_frames_per_angle=2,
    hd_count_smooth_sigma=1.0,
    min_pref_vector_strength=0.0,
    preferred_angle_mode="peak",
    pref_angle_std_threshold=None,
    hd_bin_size_deg=30,
    direction_mode="head",
    travel_smooth_window=5,
    travel_min_step=0.0,
    cmap="viridis",
    quiver_scale=None,
    show_trajectory=False,
    use_normalized_rate=False,
    max_arrow_length=None,
    max_arrow_std_factor=3.0,
    polar_smooth_sigma=1.0,
    pf_bg_alpha=0.1,
    tuning_curve_color="green",
    tuning_style="line",
    mini_pref_arrow_length_mode="vector_strength",
    mini_pref_arrow_length_min=0.15,
    mini_pref_arrow_length_max=0.95,
    occupancy_style="line",
    show_hd_info_stars=True,
    enable_hd_info_shuffle_gate=False,
    hd_info_n_shuffles=1000,
    hd_info_alpha=0.05,
    hd_info_gate_rule="p_and_above_null_mean",
    hd_info_random_seed=None,
    hd_info_star_thresholds=(0.05, 0.01, 0.001),
    hd_info_prior_alpha=0.0,
    hd_info_prior_beta_s=0.0,
    hd_info_use_smoothed_counts=True,
    hd_info_null_mode="timeshift",
    hd_time_shift_min_s=10.0,
    hd_time_shift_sessionwise=True,
    vector_require_shuffle_pass=True,
    right_panel_width_mode="auto",
    figsize_base=(6.5, 5.0),
    save_dir=None,
    save_prefix="hd_vector_field_polar",
    save_ext="svg",
    close_figures=False,
    return_data=True,
    session_start_frames=None,
):
    """
    Convenience wrapper: generate one vector+polar figure per cell.
    """
    n_cells = len(spikes)
    if cell_order is None:
        cell_order = list(range(n_cells))
    else:
        cell_order = list(cell_order)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    outputs = []
    for cell_idx in cell_order:
        if bad_timepoints_all is None:
            bad_tp = None
        elif (
            isinstance(bad_timepoints_all, (list, tuple, np.ndarray))
            and len(bad_timepoints_all) == n_cells
        ):
            bad_tp = bad_timepoints_all[cell_idx]
        else:
            bad_tp = bad_timepoints_all

        save_path = None
        if save_dir is not None:
            ext = str(save_ext).lstrip(".")
            save_path = os.path.join(
                save_dir,
                f"{save_prefix}_cell{int(cell_idx) + 1}.{ext}",
            )

        fig, axes_main, axes_polar, result_dict = (
            plot_hd_vector_fields_single_cell_moving_with_bin_polar(
                x_neural,
                y_neural,
                hd_angles_neural,
                spikes,
                speed,
                frame_rate,
                cell_idx=cell_idx,
                refined_SS=refined_SS,
                all_CS_spikes=all_CS_spikes,
                pc_output_all=pc_output_all,
                width_real=width_real,
                height_real=height_real,
                bin_size=bin_size,
                kernel_size=kernel_size,
                filter_type=filter_type,
                speed_threshold=speed_threshold,
                min_duration_s=min_duration_s,
                merge_gap_s=merge_gap_s,
                bad_timepoints=bad_tp,
                min_spikes_per_bin=min_spikes_per_bin,
                min_visited_angle_bins=min_visited_angle_bins,
                min_visited_angle_separation_deg=min_visited_angle_separation_deg,
                min_occ_frames_per_angle=min_occ_frames_per_angle,
                hd_count_smooth_sigma=hd_count_smooth_sigma,
                min_pref_vector_strength=min_pref_vector_strength,
                preferred_angle_mode=preferred_angle_mode,
                pref_angle_std_threshold=pref_angle_std_threshold,
                hd_bin_size_deg=hd_bin_size_deg,
                direction_mode=direction_mode,
                travel_smooth_window=travel_smooth_window,
                travel_min_step=travel_min_step,
                cmap=cmap,
                quiver_scale=quiver_scale,
                show_trajectory=show_trajectory,
                use_normalized_rate=use_normalized_rate,
                max_arrow_length=max_arrow_length,
                max_arrow_std_factor=max_arrow_std_factor,
                polar_smooth_sigma=polar_smooth_sigma,
                pf_bg_alpha=pf_bg_alpha,
                tuning_curve_color=tuning_curve_color,
                tuning_style=tuning_style,
                mini_pref_arrow_length_mode=mini_pref_arrow_length_mode,
                mini_pref_arrow_length_min=mini_pref_arrow_length_min,
                mini_pref_arrow_length_max=mini_pref_arrow_length_max,
                occupancy_style=occupancy_style,
                show_hd_info_stars=show_hd_info_stars,
                enable_hd_info_shuffle_gate=enable_hd_info_shuffle_gate,
                hd_info_n_shuffles=hd_info_n_shuffles,
                hd_info_alpha=hd_info_alpha,
                hd_info_gate_rule=hd_info_gate_rule,
                hd_info_random_seed=hd_info_random_seed,
                hd_info_star_thresholds=hd_info_star_thresholds,
                hd_info_prior_alpha=hd_info_prior_alpha,
                hd_info_prior_beta_s=hd_info_prior_beta_s,
                hd_info_use_smoothed_counts=hd_info_use_smoothed_counts,
                hd_info_null_mode=hd_info_null_mode,
                hd_time_shift_min_s=hd_time_shift_min_s,
                hd_time_shift_sessionwise=hd_time_shift_sessionwise,
                vector_require_shuffle_pass=vector_require_shuffle_pass,
                right_panel_width_mode=right_panel_width_mode,
                figsize_base=figsize_base,
                save_path=save_path,
                return_data=return_data,
                session_start_frames=session_start_frames,
            )
        )

        outputs.append(
            {
                "cell_id": int(cell_idx),
                "save_path": save_path,
                "fig": fig,
                "axes_main": axes_main,
                "axes_polar": axes_polar,
                "result": result_dict,
            }
        )

        if close_figures:
            plt.close(fig)

    return outputs


def _merge_segments(starts, ends, max_gap_frames):
    if len(starts) == 0:
        return [], []
    merged_starts = [starts[0]]
    merged_ends = [ends[0]]
    for i in range(1, len(starts)):
        if starts[i] - merged_ends[-1] <= max_gap_frames:
            merged_ends[-1] = ends[i]
        else:
            merged_starts.append(starts[i])
            merged_ends.append(ends[i])
    return merged_starts, merged_ends


def _keep_largest_component(mask, min_bins=None):
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return mask
    labeled_array, num_features = label(mask)
    if num_features == 0:
        return mask
    component_sizes = [(i, np.sum(labeled_array == i)) for i in range(1, num_features + 1)]
    component_sizes.sort(key=lambda x: x[1], reverse=True)
    largest_idx, largest_size = component_sizes[0]
    if min_bins is not None and largest_size < min_bins:
        return np.zeros_like(mask, dtype=bool)
    return labeled_array == largest_idx


def _keep_peak_component(mask, peak_map, min_bins=None):
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return mask
    labeled_array, num_features = label(mask)
    if num_features == 0:
        return mask
    peak_map = np.asarray(peak_map)
    best_label = None
    best_peak = -np.inf
    for label_idx in range(1, num_features + 1):
        component = labeled_array == label_idx
        size = np.sum(component)
        if min_bins is not None and size < min_bins:
            continue
        if np.any(component):
            comp_peak = np.nanmax(peak_map[component])
        else:
            comp_peak = -np.inf
        if not np.isfinite(comp_peak):
            comp_peak = -np.inf
        if comp_peak > best_peak:
            best_peak = comp_peak
            best_label = label_idx
    if best_label is None:
        return np.zeros_like(mask, dtype=bool)
    return labeled_array == best_label


def _select_pf_component(mask, peak_map=None, min_bins=None, strategy="peak_rate"):
    """
    Select one PF component from a multi-component mask.

    Parameters
    ----------
    mask : ndarray (bool)
        Candidate PF mask.
    peak_map : ndarray or None
        Map used to evaluate per-component peak values when strategy is peak-based.
    min_bins : int or None
        Minimum component size (in bins).
    strategy : str
        One of:
        - "peak_rate" (default): component with highest peak firing rate
        - "largest_area": component with largest area (number of bins)
    """
    if strategy is None:
        strategy = "peak_rate"
    strategy = str(strategy).strip().lower()

    if strategy in {"peak_rate", "highest_rate", "highest_firing_rate", "peak"}:
        if peak_map is None:
            # Fallback for safety; callers should normally provide peak_map.
            return _keep_largest_component(mask, min_bins=min_bins)
        return _keep_peak_component(mask, peak_map, min_bins=min_bins)
    if strategy in {"largest_area", "largest", "area"}:
        return _keep_largest_component(mask, min_bins=min_bins)

    raise ValueError(
        f"Unknown pf_component_selection={strategy!r}. "
        "Use 'peak_rate' or 'largest_area'."
    )


def _find_component_local_peak_candidates(component_mask, peak_map):
    component_mask = np.asarray(component_mask, dtype=bool)
    peak_map = np.asarray(peak_map, dtype=float)
    if not np.any(component_mask):
        return []

    valid_vals = np.where(component_mask & np.isfinite(peak_map), peak_map, -np.inf)
    local_max_mask = component_mask & np.isfinite(peak_map)
    if np.any(local_max_mask):
        neigh_max = maximum_filter(valid_vals, size=3, mode="constant", cval=-np.inf)
        local_max_mask &= valid_vals >= neigh_max
    else:
        return []

    labeled_max, n_max = label(local_max_mask)
    peaks = []
    for peak_label in range(1, n_max + 1):
        plateau = labeled_max == peak_label
        if not np.any(plateau):
            continue
        plateau_vals = np.where(plateau, peak_map, -np.inf)
        flat_idx = int(np.nanargmax(plateau_vals))
        coord = tuple(int(v) for v in np.unravel_index(flat_idx, peak_map.shape))
        peak_val = float(peak_map[coord])
        if not np.isfinite(peak_val):
            continue
        peaks.append({
            "coord": coord,
            "flat_idx": flat_idx,
            "peak_rate": peak_val,
        })
    peaks.sort(key=lambda p: p["peak_rate"], reverse=True)
    return peaks


def _assign_component_bins_to_selected_peaks(component_mask, peak_map, selected_peaks):
    component_mask = np.asarray(component_mask, dtype=bool)
    peak_map = np.asarray(peak_map, dtype=float)
    if not np.any(component_mask) or len(selected_peaks) != 2:
        return np.zeros_like(component_mask, dtype=int)

    n_cols = int(component_mask.shape[1])
    component_coords = np.argwhere(component_mask)
    component_flat = [int(r * n_cols + c) for r, c in component_coords]
    value_map = np.asarray(peak_map, dtype=float)
    memo_dest: dict[int, int] = {}

    def _neighbors(flat_idx: int):
        r = flat_idx // n_cols
        c = flat_idx % n_cols
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr = r + dr
                cc = c + dc
                if 0 <= rr < component_mask.shape[0] and 0 <= cc < component_mask.shape[1] and component_mask[rr, cc]:
                    yield int(rr * n_cols + cc)

    def _ascend(flat_idx: int) -> int:
        cached = memo_dest.get(flat_idx)
        if cached is not None:
            return cached
        best_idx = int(flat_idx)
        best_val = float(value_map.flat[flat_idx])
        for n_idx in _neighbors(flat_idx):
            n_val = float(value_map.flat[n_idx])
            if not np.isfinite(n_val):
                continue
            if (n_val > best_val + 1e-12) or (abs(n_val - best_val) <= 1e-12 and n_idx < best_idx):
                best_idx = int(n_idx)
                best_val = float(n_val)
        if best_idx == flat_idx:
            memo_dest[flat_idx] = int(flat_idx)
        else:
            memo_dest[flat_idx] = _ascend(best_idx)
        return memo_dest[flat_idx]

    selected_flat = [int(p["flat_idx"]) for p in selected_peaks]
    selected_coords = [np.asarray(p["coord"], dtype=float) for p in selected_peaks]
    assigned = np.zeros_like(component_mask, dtype=int)

    for flat_idx in component_flat:
        dest = _ascend(int(flat_idx))
        if dest in selected_flat:
            label_idx = int(selected_flat.index(dest)) + 1
        else:
            dest_coord = np.asarray([dest // n_cols, dest % n_cols], dtype=float)
            dists = [float(np.linalg.norm(dest_coord - sc)) for sc in selected_coords]
            label_idx = 1 if dists[0] <= dists[1] else 2
        r = flat_idx // n_cols
        c = flat_idx % n_cols
        assigned[r, c] = int(label_idx)

    return assigned


def _split_pf_component_on_secondary_peak(
    component_mask,
    peak_map,
    *,
    min_bins=None,
    peak_ratio=0.8,
    min_separation_cm=15.0,
    bin_size_cm=1.5,
):
    component_mask = np.asarray(component_mask, dtype=bool)
    peak_map = np.asarray(peak_map, dtype=float)
    if not np.any(component_mask):
        return []

    peak_candidates = _find_component_local_peak_candidates(component_mask, peak_map)
    if len(peak_candidates) < 2:
        return [component_mask]

    primary = peak_candidates[0]
    primary_peak = float(primary["peak_rate"])
    if not np.isfinite(primary_peak) or primary_peak <= 0:
        return [component_mask]

    selected_secondary = None
    for candidate in peak_candidates[1:]:
        cand_peak = float(candidate["peak_rate"])
        if (not np.isfinite(cand_peak)) or cand_peak < float(peak_ratio) * primary_peak:
            continue
        dr = float(candidate["coord"][0] - primary["coord"][0]) * float(bin_size_cm)
        dc = float(candidate["coord"][1] - primary["coord"][1]) * float(bin_size_cm)
        peak_dist_cm = float(np.hypot(dr, dc))
        if peak_dist_cm >= float(min_separation_cm):
            selected_secondary = candidate
            break

    if selected_secondary is None:
        return [component_mask]

    assigned = _assign_component_bins_to_selected_peaks(
        component_mask,
        peak_map,
        [primary, selected_secondary],
    )
    min_bins_int = None if min_bins is None else int(min_bins)
    split_components = []
    for label_idx in (1, 2):
        subcomponent = assigned == label_idx
        if not np.any(subcomponent):
            return [component_mask]
        sub_size = int(np.sum(subcomponent))
        if min_bins_int is not None and sub_size < min_bins_int:
            return [component_mask]
        split_components.append(subcomponent)
    return split_components


def _keep_all_valid_components(
    mask,
    peak_map,
    min_peak_ratio=0.4,
    split_multi_peak_fields=True,
    split_secondary_peak_ratio=0.8,
    split_secondary_peak_min_separation_cm=15.0,
    bin_size_cm=1.5,
    min_bins=None,
):
    """
    Keep all connected components that meet minimum size requirement.
    
    Returns:
        combined_mask: Boolean mask with all valid components
        component_info: List of dicts with 'mask', 'size', 'peak_rate' for each valid component,
                       sorted by peak firing rate (highest first) to determine primacy
    """
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return mask, []
    
    labeled_array, num_features = label(mask)
    if num_features == 0:
        return mask, []
    
    peak_map = np.asarray(peak_map)
    global_peak = np.nanmax(peak_map) if peak_map.size > 0 else np.nan
    valid_components = []
    
    for label_idx in range(1, num_features + 1):
        component = labeled_array == label_idx
        subcomponents = [component]
        if bool(split_multi_peak_fields):
            subcomponents = _split_pf_component_on_secondary_peak(
                component,
                peak_map,
                min_bins=min_bins,
                peak_ratio=split_secondary_peak_ratio,
                min_separation_cm=split_secondary_peak_min_separation_cm,
                bin_size_cm=bin_size_cm,
            )
        for split_idx, subcomponent in enumerate(subcomponents, start=1):
            size = int(np.sum(subcomponent))

            # Skip components that are too small
            if min_bins is not None and size < min_bins:
                continue

            # Get peak rate in this component
            if np.any(subcomponent):
                comp_peak = np.nanmax(peak_map[subcomponent])
            else:
                comp_peak = np.nan
            if (
                np.isfinite(float(min_peak_ratio))
                and np.isfinite(global_peak)
                and global_peak > 0
                and (not np.isfinite(comp_peak) or float(comp_peak) < float(min_peak_ratio) * float(global_peak))
            ):
                continue

            valid_components.append({
                'mask': subcomponent,
                'size': size,
                'peak_rate': comp_peak,
                'label': label_idx,
                'split_rank': split_idx,
                'was_split': bool(len(subcomponents) > 1),
            })
    
    if not valid_components:
        return np.zeros_like(mask, dtype=bool), []
    
    # Sort by peak firing rate (highest first) to determine primacy
    # Use -inf for non-finite values so they sort to the end
    valid_components.sort(key=lambda x: x['peak_rate'] if np.isfinite(x['peak_rate']) else -np.inf, reverse=True)
    
    # Create combined mask of all valid components
    combined_mask = np.zeros_like(mask, dtype=bool)
    for comp in valid_components:
        combined_mask |= comp['mask']
    
    return combined_mask, valid_components


def _build_pf_dilation_structure(radius_bins, shape="disk"):
    radius_bins = int(radius_bins)
    if radius_bins < 0:
        raise ValueError("radius_bins must be >= 0.")

    shape = str(shape).strip().lower()
    if shape not in {"disk", "square", "manhattan"}:
        raise ValueError(
            f"Unknown shape={shape!r}. Use 'disk', 'square', or 'manhattan'."
        )

    side = (2 * radius_bins) + 1
    if shape == "square":
        return np.ones((side, side), dtype=bool)

    coords = np.arange(-radius_bins, radius_bins + 1, dtype=int)
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    if shape == "disk":
        return (xx * xx + yy * yy) <= (radius_bins * radius_bins)
    return (np.abs(xx) + np.abs(yy)) <= radius_bins


def _normalize_bad_frame_mask_for_pf_filter(bad_timepoints, n_frames):
    if bad_timepoints is None:
        return None
    bad_mask = np.asarray(bad_timepoints)
    if bad_mask.dtype != bool:
        if bad_mask.ndim == 1 and bad_mask.size == int(n_frames):
            bad_mask = bad_mask.astype(bool)
        else:
            bad_idx = np.asarray(bad_mask, dtype=int)
            bad_idx = bad_idx[(bad_idx >= 0) & (bad_idx < int(n_frames))]
            out = np.zeros(int(n_frames), dtype=bool)
            out[bad_idx] = True
            bad_mask = out
    if bad_mask.shape[0] != int(n_frames):
        return None
    return np.asarray(bad_mask, dtype=bool)


def _build_symmetric_distance_axis_for_pf_filter(distance_window_cm, distance_bin_cm):
    window_req = float(distance_window_cm)
    bin_cm = float(distance_bin_cm)
    if window_req <= 0 or bin_cm <= 0:
        raise ValueError("distance_window_cm and distance_bin_cm must be > 0.")
    n_half_bins = max(1, int(np.ceil(window_req / bin_cm - 1e-12)))
    window_eff = float(n_half_bins) * bin_cm
    distance_rel = np.arange(-n_half_bins, n_half_bins + 1, dtype=float) * bin_cm
    distance_rel[np.abs(distance_rel) < 1e-12] = 0.0
    return window_req, window_eff, distance_rel


def _normalize_distance_mode_for_pf_filter(raw):
    mode = str(raw if raw is not None else "").strip().lower()
    if mode in {"cumulative_path", "cumulative", "path", "path_length", "travel", "traveled_distance"}:
        return "cumulative_path"
    if mode in {"euclidean_to_peak", "euclidean", "peak", "straight_to_peak"}:
        return "euclidean_to_peak"
    raise ValueError(
        f"Unsupported distance_mode {raw!r}. Use 'cumulative_path' or 'euclidean_to_peak'."
    )


def _compute_signed_distance_from_center_for_pf_filter(
    x_seg,
    y_seg,
    *,
    center_local,
    pf_peak_xy,
    distance_mode,
):
    x_seg = np.asarray(x_seg, dtype=float)
    y_seg = np.asarray(y_seg, dtype=float)
    if x_seg.size == 0 or y_seg.size == 0 or x_seg.size != y_seg.size:
        return None
    center_local = int(center_local)
    if center_local < 0 or center_local >= int(x_seg.size):
        return None
    if not (np.all(np.isfinite(x_seg)) and np.all(np.isfinite(y_seg))):
        return None

    mode = _normalize_distance_mode_for_pf_filter(distance_mode)
    if mode == "cumulative_path":
        if x_seg.size == 1:
            return np.array([0.0], dtype=float)
        step_cm = np.sqrt(np.diff(x_seg) ** 2 + np.diff(y_seg) ** 2)
        cumulative_cm = np.concatenate([np.array([0.0], dtype=float), np.cumsum(step_cm, dtype=float)])
        signed_distance = cumulative_cm - float(cumulative_cm[center_local])
        signed_distance[np.abs(signed_distance) < 1e-12] = 0.0
        return signed_distance

    pf_x, pf_y = pf_peak_xy
    signed_distance = np.sqrt((x_seg - float(pf_x)) ** 2 + (y_seg - float(pf_y)) ** 2)
    signed_distance = np.asarray(signed_distance, dtype=float)
    signed_distance[:center_local] *= -1.0
    signed_distance[center_local] = 0.0
    signed_distance[np.abs(signed_distance) < 1e-12] = 0.0
    return signed_distance


def _find_distance_defined_trials_iterative_for_pf_filter(
    dist_to_peak,
    valid_mask,
    moving_mask,
    *,
    trial_span_cm,
    detection_span_cm,
    vicinity_min_cm,
    vicinity_max_cm,
):
    n = int(len(dist_to_peak))
    if n == 0:
        return []
    dist_arr = np.asarray(dist_to_peak, dtype=float).reshape(-1)
    valid_arr = np.asarray(valid_mask, dtype=bool).reshape(-1)
    moving_arr = np.asarray(moving_mask, dtype=bool).reshape(-1)
    if dist_arr.size != n or valid_arr.size != n or moving_arr.size != n:
        raise ValueError("dist_to_peak, valid_mask, and moving_mask must have matching lengths.")

    trials = []
    taken = np.zeros(n, dtype=bool)
    valid_dist_mask = valid_arr & np.isfinite(dist_arr)
    moving_valid_mask = valid_dist_mask & moving_arr
    if not np.any(moving_valid_mask):
        return []
    moving_valid_indices = np.flatnonzero(moving_valid_mask)

    def _build_reach_arrays(span_cm):
        reach_mask = valid_dist_mask & (dist_arr >= float(span_cm))
        left_reach = np.full(n, -1, dtype=int)
        right_reach = np.full(n, -1, dtype=int)
        last_idx = -1
        for i in range(n):
            if reach_mask[i]:
                last_idx = int(i)
            left_reach[i] = int(last_idx)
        next_idx = -1
        for i in range(n - 1, -1, -1):
            if reach_mask[i]:
                next_idx = int(i)
            right_reach[i] = int(next_idx)
        return left_reach, right_reach

    left_detect_reach, right_detect_reach = _build_reach_arrays(float(detection_span_cm))
    left_trial_reach, right_trial_reach = _build_reach_arrays(float(trial_span_cm))

    def _find_left(center_idx, left_reach):
        center_idx = int(center_idx)
        if center_idx <= 0:
            return None
        out = int(left_reach[center_idx - 1])
        return None if out < 0 else out

    def _find_right(center_idx, right_reach):
        center_idx = int(center_idx)
        if center_idx >= (n - 1):
            return None
        out = int(right_reach[center_idx + 1])
        return None if out < 0 else out

    for vicinity_cm in range(int(vicinity_min_cm), int(vicinity_max_cm) + 1):
        candidate_indices = np.flatnonzero(moving_valid_mask & (dist_arr <= float(vicinity_cm)))
        if candidate_indices.size == 0:
            continue
        pointer = 0
        pos = 0
        n_candidates = int(candidate_indices.size)
        while pos < n_candidates:
            center_idx = int(candidate_indices[pos])
            if center_idx < pointer or bool(taken[center_idx]):
                pos += 1
                continue
            candidate_center_idx = int(center_idx)
            detect_start_idx = _find_left(center_idx, left_detect_reach)
            detect_end_idx = _find_right(center_idx, right_detect_reach)
            if (
                detect_start_idx is None
                or detect_end_idx is None
                or detect_end_idx <= detect_start_idx
            ):
                pointer = max(int(pointer), int(candidate_center_idx) + 1, int(center_idx) + 1)
                pos += 1
                continue
            start_idx = _find_left(center_idx, left_trial_reach)
            end_idx = _find_right(center_idx, right_trial_reach)
            if start_idx is None or end_idx is None or end_idx <= start_idx:
                pointer = max(int(pointer), int(candidate_center_idx) + 1, int(center_idx) + 1)
                pos += 1
                continue

            left_pos = int(np.searchsorted(moving_valid_indices, int(start_idx), side="left"))
            right_pos = int(np.searchsorted(moving_valid_indices, int(end_idx), side="right"))
            trial_candidates = moving_valid_indices[left_pos:right_pos]
            if trial_candidates.size > 0 and np.any(taken[trial_candidates]):
                trial_candidates = trial_candidates[~taken[trial_candidates]]
            if trial_candidates.size == 0:
                pointer = max(int(pointer), int(candidate_center_idx) + 1, int(center_idx) + 1)
                pos += 1
                continue
            trial_dists = np.asarray(dist_arr[trial_candidates], dtype=float)
            center_idx = int(trial_candidates[int(np.nanargmin(trial_dists))])
            detect_start_idx = _find_left(center_idx, left_detect_reach)
            detect_end_idx = _find_right(center_idx, right_detect_reach)
            if (
                detect_start_idx is None
                or detect_end_idx is None
                or detect_end_idx <= detect_start_idx
            ):
                pointer = max(int(pointer), int(candidate_center_idx) + 1, int(center_idx) + 1)
                pos += 1
                continue
            start_idx = _find_left(center_idx, left_trial_reach)
            end_idx = _find_right(center_idx, right_trial_reach)
            if start_idx is None or end_idx is None or end_idx <= start_idx:
                pointer = max(int(pointer), int(candidate_center_idx) + 1, int(center_idx) + 1)
                pos += 1
                continue
            end_exclusive = int(end_idx + 1)
            if np.any(taken[start_idx:end_exclusive]):
                pointer = max(int(pointer), int(candidate_center_idx) + 1, int(center_idx) + 1)
                pos += 1
                continue
            trials.append(
                {
                    "epoch_start": int(start_idx),
                    "epoch_end": int(end_exclusive),
                    "center_idx": int(center_idx),
                    "center_vicinity_cm": float(vicinity_cm),
                    "detect_start": int(detect_start_idx),
                    "detect_end": int(detect_end_idx + 1),
                }
            )
            taken[start_idx:end_exclusive] = True
            pointer = int(end_exclusive)
            pos = int(np.searchsorted(candidate_indices, int(pointer), side="left"))
    if len(trials) > 1:
        trials = sorted(
            trials,
            key=lambda t: (int(t["epoch_start"]), int(t["epoch_end"]), int(t["center_idx"])),
        )
    return trials


def _compute_pf_peak_xy_from_mask_for_pf_filter(place_field_mask, rate_map, pf_bins):
    if place_field_mask is None or not np.any(place_field_mask):
        return None
    if rate_map is None:
        return None
    masked_rate = np.where(place_field_mask, rate_map, np.nan)
    if not np.any(np.isfinite(masked_rate)):
        return None
    x_centers = (pf_bins[0][:-1] + pf_bins[0][1:]) / 2
    y_centers = (pf_bins[1][:-1] + pf_bins[1][1:]) / 2
    peak_idx = np.nanargmax(masked_rate)
    peak_row, peak_col = np.unravel_index(peak_idx, masked_rate.shape)
    return float(x_centers[peak_row]), float(y_centers[peak_col])


def _compute_distance_defined_pf_firing_traversal_stats(
    *,
    x_neural,
    y_neural,
    speed,
    frame_rate,
    event_spikes,
    place_field_mask,
    rate_map,
    width_real,
    height_real,
    bin_size,
    bad_timepoints=None,
    moving_speed_threshold=3.0,
    moving_kernel_size=51,
    moving_min_duration_s=0.25,
    moving_merge_gap_s=2.0,
    distance_window_cm=15.0,
    detection_window_cm=8.0,
    distance_bin_cm=1.5,
    distance_mode="euclidean_to_peak",
    center_vicinity_min_cm=1,
    center_vicinity_max_cm=5,
    resting_speed_threshold=0.5,
    exclude_trials_with_bad_frames=True,
):
    x_arr = np.asarray(x_neural, dtype=float)
    y_arr = np.asarray(y_neural, dtype=float)
    speed_arr = np.asarray(speed, dtype=float)
    n_frames = int(min(x_arr.size, y_arr.size, speed_arr.size))
    if n_frames <= 0:
        return {"n_trials": 0, "n_firing_traversals": 0, "reliability": np.nan}
    x_arr = x_arr[:n_frames]
    y_arr = y_arr[:n_frames]
    speed_arr = speed_arr[:n_frames]

    distance_window_req, distance_window_eff, distance_rel = _build_symmetric_distance_axis_for_pf_filter(
        distance_window_cm,
        distance_bin_cm,
    )
    detection_window_eff = min(float(detection_window_cm), float(distance_window_eff))
    if (not np.isfinite(detection_window_eff)) or detection_window_eff <= 0:
        raise ValueError("detection_window_cm must be a finite number > 0.")
    distance_window_eff = float(distance_window_eff)
    distance_bin_cm = float(distance_bin_cm)
    bin_edges = np.concatenate(
        [
            distance_rel - 0.5 * distance_bin_cm,
            np.array([distance_rel[-1] + 0.5 * distance_bin_cm], dtype=float),
        ]
    )

    pf_bins = [
        np.arange(0, float(width_real) + float(bin_size), float(bin_size)),
        np.arange(0, float(height_real) + float(bin_size), float(bin_size)),
    ]
    pf_mask = np.asarray(place_field_mask, dtype=bool)
    pf_peak_xy = _compute_pf_peak_xy_from_mask_for_pf_filter(pf_mask, rate_map, pf_bins)
    if pf_peak_xy is None:
        return {"n_trials": 0, "n_firing_traversals": 0, "reliability": np.nan}
    pf_x, pf_y = pf_peak_xy

    bad_mask = _normalize_bad_frame_mask_for_pf_filter(bad_timepoints, n_frames)
    valid_frame_mask = np.isfinite(x_arr) & np.isfinite(y_arr) & np.isfinite(speed_arr)
    if bad_mask is not None:
        valid_frame_mask &= ~bad_mask

    moving_kernel_size = max(3, int(moving_kernel_size))
    if moving_kernel_size % 2 == 0:
        moving_kernel_size += 1
    _, _, moving_idx_raw = _compute_moving_epochs(
        speed=speed_arr,
        frame_rate=float(frame_rate),
        kernel_size=int(moving_kernel_size),
        filter_type="median",
        speed_threshold=float(moving_speed_threshold),
        min_duration_s=float(max(0.0, moving_min_duration_s)),
        merge_gap_s=float(max(0.0, moving_merge_gap_s)),
    )
    moving_mask = np.zeros(n_frames, dtype=bool)
    moving_idx_raw = np.asarray(moving_idx_raw, dtype=int)
    moving_idx_raw = moving_idx_raw[(moving_idx_raw >= 0) & (moving_idx_raw < n_frames)]
    moving_mask[moving_idx_raw] = True
    moving_mask &= valid_frame_mask

    dist_to_peak = np.full(n_frames, np.nan, dtype=float)
    valid_xy = np.isfinite(x_arr) & np.isfinite(y_arr)
    dist_to_peak[valid_xy] = np.sqrt(
        (x_arr[valid_xy] - float(pf_x)) ** 2 + (y_arr[valid_xy] - float(pf_y)) ** 2
    )

    trials = _find_distance_defined_trials_iterative_for_pf_filter(
        dist_to_peak=dist_to_peak,
        valid_mask=valid_frame_mask,
        moving_mask=moving_mask,
        trial_span_cm=float(distance_window_eff),
        detection_span_cm=float(detection_window_eff),
        vicinity_min_cm=int(center_vicinity_min_cm),
        vicinity_max_cm=int(center_vicinity_max_cm),
    )
    if len(trials) == 0:
        return {"n_trials": 0, "n_firing_traversals": 0, "reliability": np.nan}

    event_spikes = np.unique(np.asarray(event_spikes, dtype=int))
    event_spikes = event_spikes[(event_spikes >= 0) & (event_spikes < n_frames)]
    if bad_mask is not None and event_spikes.size > 0:
        event_spikes = event_spikes[~bad_mask[event_spikes]]
    event_spikes = np.sort(event_spikes)

    n_trials = 0
    n_firing = 0
    seen_epochs = set()
    for tr in trials:
        trial_start = int(tr["epoch_start"])
        trial_end = int(tr["epoch_end"])
        center_idx = int(tr["center_idx"])
        if trial_end <= trial_start or center_idx < trial_start or center_idx >= trial_end:
            continue
        if bool(exclude_trials_with_bad_frames) and bad_mask is not None and np.any(bad_mask[trial_start:trial_end]):
            continue

        x_seg_full = x_arr[trial_start:trial_end]
        y_seg_full = y_arr[trial_start:trial_end]
        center_local_full = int(center_idx - trial_start)
        signed_distance_full = _compute_signed_distance_from_center_for_pf_filter(
            x_seg_full,
            y_seg_full,
            center_local=center_local_full,
            pf_peak_xy=pf_peak_xy,
            distance_mode=distance_mode,
        )
        if (
            signed_distance_full is None
            or signed_distance_full.size == 0
            or not np.all(np.isfinite(signed_distance_full))
        ):
            continue
        left_candidates = np.where(signed_distance_full[:center_local_full + 1] <= -distance_window_eff)[0]
        right_candidates = np.where(signed_distance_full[center_local_full:] >= distance_window_eff)[0]
        if left_candidates.size == 0 or right_candidates.size == 0:
            continue
        start_local = int(left_candidates[-1])
        end_local = int(center_local_full + right_candidates[0])
        if end_local <= start_local:
            continue
        start = int(trial_start + start_local)
        end = int(start + (end_local - start_local + 1))
        center_local = int(center_local_full - start_local)
        signed_distance = np.asarray(signed_distance_full[start_local:end_local + 1], dtype=float)
        if end <= start or center_local < 0 or center_local >= signed_distance.size:
            continue

        speed_seg = speed_arr[start:end]
        rest_mask = np.isfinite(speed_seg) & (speed_seg < float(resting_speed_threshold))
        valid_local = valid_frame_mask[start:end] & np.isfinite(signed_distance)
        x_seg = x_arr[start:end]
        y_seg = y_arr[start:end]
        in_pf_local = _positions_in_place_field(x_seg, y_seg, pf_bins, pf_mask)
        if not np.any(in_pf_local & valid_local):
            continue
        bin_idx = np.digitize(signed_distance, bin_edges) - 1
        bin_valid = (bin_idx >= 0) & (bin_idx < int(distance_rel.size))
        include_local = valid_local & bin_valid & (~rest_mask)
        if not np.any(include_local):
            continue
        display_epoch_key = (int(start), int(end))
        if display_epoch_key in seen_epochs:
            continue
        seen_epochs.add(display_epoch_key)

        # Trial discovery mirrors generate_distance_defined_trials_and_dataset,
        # while the PF gate counts firing during the actual PF occupancy frames.
        firing_local = include_local & in_pf_local
        frame_to_bin = np.full(end - start, -1, dtype=int)
        frame_to_bin[firing_local] = bin_idx[firing_local]
        n_trials += 1
        if event_spikes.size > 0:
            left = int(np.searchsorted(event_spikes, start, side="left"))
            right = int(np.searchsorted(event_spikes, end, side="left"))
            spike_local = event_spikes[left:right] - start
            spike_local = spike_local[(spike_local >= 0) & (spike_local < (end - start))]
            if spike_local.size > 0 and np.any(frame_to_bin[spike_local] >= 0):
                n_firing += 1

    reliability = (float(n_firing) / float(n_trials)) if n_trials > 0 else np.nan
    return {
        "n_trials": int(n_trials),
        "n_firing_traversals": int(n_firing),
        "reliability": reliability,
        "distance_window_cm_requested": float(distance_window_req),
        "distance_window_cm_effective": float(distance_window_eff),
        "detection_window_cm_effective": float(detection_window_eff),
    }


def _positions_in_place_field(x_vals, y_vals, pf_bins, place_field_mask):
    x_idx = np.digitize(x_vals, pf_bins[0]) - 1
    y_idx = np.digitize(y_vals, pf_bins[1]) - 1
    x_idx = np.clip(x_idx, 0, place_field_mask.shape[0] - 1)
    y_idx = np.clip(y_idx, 0, place_field_mask.shape[1] - 1)
    valid = (~np.isnan(x_vals)) & (~np.isnan(y_vals))
    in_pf = np.zeros_like(valid, dtype=bool)
    in_pf[valid] = place_field_mask[x_idx[valid], y_idx[valid]]
    return in_pf


def positions_in_place_field(x_vals, y_vals, pf_bins, place_field_mask):
    return _positions_in_place_field(x_vals, y_vals, pf_bins, place_field_mask)


def _safe_zscore(trace, do_zscore):
    trace = np.asarray(trace, dtype=float)
    if not do_zscore:
        return trace
    mean = np.nanmean(trace)
    std = np.nanstd(trace)
    if not np.isfinite(std) or std == 0:
        return trace - mean
    return (trace - mean) / std


def _calc_path_length(x_vals, y_vals):
    dx = np.diff(x_vals)
    dy = np.diff(y_vals)
    valid = np.isfinite(dx) & np.isfinite(dy)
    if not np.any(valid):
        return 0.0
    return np.sum(np.sqrt(dx[valid] ** 2 + dy[valid] ** 2))


def _classify_traversal_direction(
    x_vals,
    y_vals,
    center_xy,
    min_angle_rad=1e-3,
    min_consistency=0.6,
    min_center_radius_cm=1.0,
):
    # Binary direction labeling only: always return "cw" or "ccw".
    if center_xy is None:
        return "cw"
    x_vals = np.asarray(x_vals, dtype=float)
    y_vals = np.asarray(y_vals, dtype=float)
    valid = np.isfinite(x_vals) & np.isfinite(y_vals)
    if np.sum(valid) < 2:
        return "cw"
    cx, cy = center_xy
    xv = x_vals[valid]
    yv = y_vals[valid]

    # Ignore samples too close to PF center, where angular direction is unstable.
    radius = np.hypot(xv - cx, yv - cy)
    stable = radius >= float(min_center_radius_cm)
    if np.sum(stable) < 2:
        # Fallback on coarse displacement direction when center-locked angle is unstable.
        dx = xv[-1] - xv[0]
        dy = yv[-1] - yv[0]
        if np.abs(dx) >= np.abs(dy):
            return "cw" if dx >= 0 else "ccw"
        return "cw" if dy >= 0 else "ccw"
    xs = xv[stable]
    ys = yv[stable]

    angles = np.arctan2(ys - cy, xs - cx)
    if angles.size < 2:
        return "cw"
    dtheta = np.arctan2(np.sin(np.diff(angles)), np.cos(np.diff(angles)))
    if dtheta.size == 0:
        return "cw"

    pos_turn = float(np.nansum(dtheta[dtheta > 0]))
    neg_turn = float(np.nansum(-dtheta[dtheta < 0]))
    gross_turn = pos_turn + neg_turn
    if gross_turn < float(min_angle_rad):
        # Too little angular change: fallback to displacement direction.
        dx = xs[-1] - xs[0]
        dy = ys[-1] - ys[0]
        if np.abs(dx) >= np.abs(dy):
            return "cw" if dx >= 0 else "ccw"
        return "cw" if dy >= 0 else "ccw"
    net_turn = pos_turn - neg_turn
    consistency = abs(net_turn) / (gross_turn + 1e-12)

    # Out-and-back / mixed-direction trajectories: split logic should usually resolve;
    # when still mixed, fall back to net-turn sign.
    if consistency < float(min_consistency):
        return "cw" if net_turn >= 0 else "ccw"

    start_angle = angles[0]
    end_angle = angles[-1]
    delta = np.arctan2(np.sin(end_angle - start_angle), np.cos(end_angle - start_angle))
    if np.abs(delta) < min_angle_rad:
        return "cw" if net_turn >= 0 else "ccw"
    return "cw" if delta > 0 else "ccw"


def _classify_traversal_direction_global(
    x_vals,
    y_vals,
    arena_center_xy,
    min_step_cm=0.1,
    min_total_path_cm=2.0,
    min_turn_score=1e-2,
):
    """
    Classify traversal direction using the overall trajectory around arena center.

    Returns binary labels:
      - "ccw" for positive global turn score
      - "cw"  for negative global turn score
    Falls back to coarse displacement sign when rotational evidence is weak.
    """
    x_vals = np.asarray(x_vals, dtype=float)
    y_vals = np.asarray(y_vals, dtype=float)
    valid = np.isfinite(x_vals) & np.isfinite(y_vals)
    if np.sum(valid) < 2:
        return "cw"

    xv = x_vals[valid]
    yv = y_vals[valid]
    if arena_center_xy is None or len(arena_center_xy) != 2:
        cx = float(np.nanmean(xv))
        cy = float(np.nanmean(yv))
    else:
        cx = float(arena_center_xy[0])
        cy = float(arena_center_xy[1])
        if not (np.isfinite(cx) and np.isfinite(cy)):
            cx = float(np.nanmean(xv))
            cy = float(np.nanmean(yv))

    dx = np.diff(xv)
    dy = np.diff(yv)
    step = np.hypot(dx, dy)
    move_mask = np.isfinite(step) & (step >= float(min_step_cm))
    if not np.any(move_mask):
        dxf = xv[-1] - xv[0]
        dyf = yv[-1] - yv[0]
        if np.abs(dxf) >= np.abs(dyf):
            return "cw" if dxf >= 0 else "ccw"
        return "cw" if dyf >= 0 else "ccw"

    rx = xv[:-1] - cx
    ry = yv[:-1] - cy
    cross = rx * dy - ry * dx
    turn_score = float(np.nansum(cross[move_mask]))
    total_path = float(np.nansum(step[move_mask]))
    if not np.isfinite(turn_score):
        turn_score = 0.0

    if total_path >= float(min_total_path_cm) and np.abs(turn_score) >= float(min_turn_score):
        return "ccw" if turn_score > 0 else "cw"

    dxf = xv[-1] - xv[0]
    dyf = yv[-1] - yv[0]
    if np.abs(dxf) >= np.abs(dyf):
        return "cw" if dxf >= 0 else "ccw"
    return "cw" if dyf >= 0 else "ccw"


def _classify_traversal_direction_pf_entry_exit(
    x_vals,
    y_vals,
    pf_peak_xy,
    entry_exit_window_frames=5,
    min_center_radius_cm=1.0,
    min_angle_deg=12.0,
):
    """
    Classify traversal direction using PF peak as reference and entry/exit geometry.

    Direction is determined from the signed angular change between:
      - mean position over first valid entry window
      - mean position over last valid exit window
    around the PF peak.
    """
    if pf_peak_xy is None:
        return "cw"
    x_vals = np.asarray(x_vals, dtype=float)
    y_vals = np.asarray(y_vals, dtype=float)
    valid = np.isfinite(x_vals) & np.isfinite(y_vals)
    if np.sum(valid) < 2:
        return "cw"

    cx = float(pf_peak_xy[0])
    cy = float(pf_peak_xy[1])
    if not (np.isfinite(cx) and np.isfinite(cy)):
        return "cw"

    idx = np.flatnonzero(valid)
    n_win = max(1, int(entry_exit_window_frames))
    n_win = min(n_win, int(idx.size))
    entry_idx = idx[:n_win]
    exit_idx = idx[-n_win:]

    entry_xy = np.array(
        [
            float(np.nanmean(x_vals[entry_idx])),
            float(np.nanmean(y_vals[entry_idx])),
        ],
        dtype=float,
    )
    exit_xy = np.array(
        [
            float(np.nanmean(x_vals[exit_idx])),
            float(np.nanmean(y_vals[exit_idx])),
        ],
        dtype=float,
    )
    v_in = entry_xy - np.array([cx, cy], dtype=float)
    v_out = exit_xy - np.array([cx, cy], dtype=float)
    r_in = float(np.hypot(v_in[0], v_in[1]))
    r_out = float(np.hypot(v_out[0], v_out[1]))
    min_r = float(min_center_radius_cm)
    min_angle_rad = float(np.deg2rad(float(min_angle_deg)))

    if np.isfinite(r_in) and np.isfinite(r_out) and r_in >= min_r and r_out >= min_r:
        ang_in = float(np.arctan2(v_in[1], v_in[0]))
        ang_out = float(np.arctan2(v_out[1], v_out[0]))
        delta = float(np.arctan2(np.sin(ang_out - ang_in), np.cos(ang_out - ang_in)))
        if np.abs(delta) >= min_angle_rad:
            # Keep historical sign convention used in local PF classifier.
            return "cw" if delta > 0 else "ccw"

    xv = x_vals[valid]
    yv = y_vals[valid]
    radius = np.hypot(xv - cx, yv - cy)
    stable = radius >= min_r
    if np.sum(stable) >= 2:
        angles = np.arctan2(yv[stable] - cy, xv[stable] - cx)
        delta_stable = float(
            np.arctan2(np.sin(angles[-1] - angles[0]), np.cos(angles[-1] - angles[0]))
        )
        if np.abs(delta_stable) >= (0.5 * min_angle_rad):
            return "cw" if delta_stable > 0 else "ccw"
        dtheta = np.arctan2(np.sin(np.diff(angles)), np.cos(np.diff(angles)))
        if dtheta.size > 0:
            net_turn = float(np.nansum(dtheta))
            if np.isfinite(net_turn) and np.abs(net_turn) >= (0.5 * min_angle_rad):
                return "cw" if net_turn >= 0 else "ccw"

    dxf = float(xv[-1] - xv[0])
    dyf = float(yv[-1] - yv[0])
    if np.abs(dxf) >= np.abs(dyf):
        return "cw" if dxf >= 0 else "ccw"
    return "cw" if dyf >= 0 else "ccw"


def find_place_field_traversals(
    x_neural,
    y_neural,
    spikes,
    speed,
    frame_rate,
    cell_idx,
    speed_threshold=2,
    min_duration_ms=500,
    min_distance_cm=5,
    max_pf_distance_cm=None,
    width_real=35.5,
    height_real=20,
    bin_size=1.5,
    smooth_sigma=2.0,
    place_field_threshold=0.4,
    min_field_bins=5,
    speed_smooth_kernel=51,
    merge_gap_s=1.0,
    clear_traversal=False,
    verbose=True,
    analysis=None,
    return_traversal_types=False,
    pf_component_selection="peak_rate",
    bad_timepoints=None,
):
    """
    Find traversal epochs through a cell's place field.

    Returns traversal epochs (start, end), place field mask, smoothed rate map, and bins.
    Epochs are inclusive of start and exclusive of end.
    If analysis is provided, its place-field mask and bins are reused.
    If return_traversal_types is True, also returns a list of traversal types ("cw"/"ccw"/"unknown").
    If bad_timepoints is provided, those frames are excluded from traversal detection.
    """
    x_neural = np.asarray(x_neural)
    y_neural = np.asarray(y_neural)
    speed = np.asarray(speed, dtype=float)
    total_frames = len(x_neural)

    if analysis is not None and not analysis.get("is_place_cell", True):
        if verbose:
            print(f"Warning: cell {cell_idx} is not a place cell; no traversals computed.")
        params = analysis.get("params", {})
        width_real = params.get("width_real", width_real)
        height_real = params.get("height_real", height_real)
        bin_size = params.get("bin_size", bin_size)
        bins = [
            np.arange(0, width_real + bin_size, bin_size),
            np.arange(0, height_real + bin_size, bin_size),
        ]
        empty_mask = np.zeros((0, 0), dtype=bool)
        empty_map = np.full((0, 0), np.nan)
        if return_traversal_types:
            return [], empty_mask, empty_map, bins, []
        return [], empty_mask, empty_map, bins

    if analysis is not None:
        params = analysis.get("params", {})
        width_real = params.get("width_real", width_real)
        height_real = params.get("height_real", height_real)
        bin_size = params.get("bin_size", bin_size)
        bins = [
            np.arange(0, width_real + bin_size, bin_size),
            np.arange(0, height_real + bin_size, bin_size),
        ]
        place_field_mask = np.asarray(analysis.get("place_field_mask", []), dtype=bool)
        smooth_map = analysis.get("rate_map", np.full_like(place_field_mask, np.nan))
        peak_rate = analysis.get("peak_rate", np.nan)
        if not np.isfinite(peak_rate) and smooth_map.size > 0:
            peak_rate = np.nanmax(smooth_map)
        if place_field_mask.size > 0:
            place_field_mask = _select_pf_component(
                place_field_mask,
                smooth_map,
                min_bins=min_field_bins,
                strategy=pf_component_selection,
            )
    else:
        bins = [
            np.arange(0, width_real + bin_size, bin_size),
            np.arange(0, height_real + bin_size, bin_size),
        ]

        valid_frames = (~np.isnan(x_neural)) & (~np.isnan(y_neural)) & (~np.isnan(speed))
        idx_run = np.where((speed >= speed_threshold) & valid_frames)[0]

        x_run = x_neural[idx_run]
        y_run = y_neural[idx_run]

        occ_counts, _, _ = np.histogram2d(x_run, y_run, bins=bins)
        occ_map = occ_counts / frame_rate
        occ_map = gaussian_filter(occ_map, sigma=smooth_sigma, mode="constant")

        cell_spikes = np.asarray(spikes[cell_idx], dtype=int)
        cell_spikes = cell_spikes[(cell_spikes >= 0) & (cell_spikes < total_frames)]
        binary_train_full = np.zeros(total_frames, dtype=bool)
        binary_train_full[cell_spikes] = True
        binary_train_run = binary_train_full[idx_run]
        spikes_in_run = np.where(binary_train_run)[0]

        x_spk = x_run[spikes_in_run]
        y_spk = y_run[spikes_in_run]
        spike_map, _, _ = np.histogram2d(x_spk, y_spk, bins=bins)

        with np.errstate(divide="ignore", invalid="ignore"):
            raw_map = spike_map / occ_map
            raw_map[np.isnan(raw_map)] = 0
            raw_map[np.isinf(raw_map)] = 0

        smooth_map = gaussian_filter(raw_map, sigma=smooth_sigma, mode="constant")
        smooth_map[occ_map == 0] = np.nan

        peak_rate = np.nanmax(smooth_map)
        place_field_mask = np.zeros_like(smooth_map, dtype=bool)
        if not (np.isnan(peak_rate) or peak_rate == 0):
            threshold_val = place_field_threshold * peak_rate
            with np.errstate(invalid="ignore"):
                raw_field_mask = smooth_map > threshold_val

            place_field_mask = _select_pf_component(
                raw_field_mask,
                smooth_map,
                min_bins=min_field_bins,
                strategy=pf_component_selection,
            )

    valid_frames = (~np.isnan(x_neural)) & (~np.isnan(y_neural)) & (~np.isnan(speed))
    if bad_timepoints is not None:
        bad_mask = np.asarray(bad_timepoints)
        if bad_mask.dtype != bool:
            if bad_mask.ndim == 1 and bad_mask.size == len(x_neural):
                bad_mask = bad_mask.astype(bool)
            else:
                bad_mask = np.asarray(bad_mask, dtype=int)
                bad_mask = bad_mask[(bad_mask >= 0) & (bad_mask < len(x_neural))]
                bad_bool = np.zeros(len(x_neural), dtype=bool)
                bad_bool[bad_mask] = True
                bad_mask = bad_bool
        if bad_mask.shape[0] != len(x_neural):
            raise ValueError("bad_timepoints must match x_neural length or be index list.")
        valid_frames &= ~bad_mask

    in_place_field = _positions_in_place_field(x_neural, y_neural, bins, place_field_mask)

    if speed_smooth_kernel % 2 == 0:
        speed_smooth_kernel += 1
    speed_smooth = median_filter(speed, size=speed_smooth_kernel)
    traversing = in_place_field & (speed_smooth > speed_threshold) & valid_frames

    diff = np.diff(np.concatenate([[0], traversing.astype(int), [0]]))
    raw_starts = np.where(diff == 1)[0]
    raw_ends = np.where(diff == -1)[0]
    raw_epochs = list(zip(raw_starts, raw_ends))

    merge_gap_frames = int(round(merge_gap_s * frame_rate))
    merged_epochs = []
    for start, end in raw_epochs:
        if not merged_epochs:
            merged_epochs.append([start, end])
            continue
        prev_start, prev_end = merged_epochs[-1]
        if (start - prev_end) <= merge_gap_frames:
            merged_epochs[-1][1] = end
        else:
            merged_epochs.append([start, end])

    trimmed_epochs = []
    for start, end in merged_epochs:
        in_pf_window = in_place_field[start:end]
        if not np.any(in_pf_window):
            continue
        first_in = start + int(np.argmax(in_pf_window))

        last_exit = first_in + 1
        in_field_now = in_place_field[first_in]
        for i in range(first_in, end):
            if in_place_field[i]:
                in_field_now = True
                last_exit = i + 1
            elif in_field_now:
                look_end = min(i + merge_gap_frames, end)
                if np.any(in_place_field[i:look_end]):
                    continue
                break
            in_field_now = False

        if last_exit > first_in:
            trimmed_epochs.append((first_in, last_exit))

    min_duration_frames = int(min_duration_ms / 1000 * frame_rate)

    # PF center (used for optional turnaround splitting and traversal direction type labels).
    center_xy = None
    if np.any(place_field_mask):
        x_centers = (bins[0][:-1] + bins[0][1:]) / 2
        y_centers = (bins[1][:-1] + bins[1][1:]) / 2
        mask_indices = np.argwhere(place_field_mask)
        if mask_indices.size > 0:
            pf_x = x_centers[mask_indices[:, 0]]
            pf_y = y_centers[mask_indices[:, 1]]
            center_xy = (np.nanmean(pf_x), np.nanmean(pf_y))

    # Compute PF peak position for max_pf_distance_cm filtering
    pf_peak_xy = None
    if max_pf_distance_cm is not None and np.any(place_field_mask):
        x_centers = (bins[0][:-1] + bins[0][1:]) / 2
        y_centers = (bins[1][:-1] + bins[1][1:]) / 2
        masked_rate = np.where(place_field_mask, smooth_map, np.nan)
        if np.any(np.isfinite(masked_rate)):
            peak_idx = np.nanargmax(masked_rate)
            peak_row, peak_col = np.unravel_index(peak_idx, masked_rate.shape)
            pf_peak_xy = (x_centers[peak_row], y_centers[peak_col])

    traversal_epochs = []
    for start, end in trimmed_epochs:
        if (end - start) < min_duration_frames:
            continue
        if _calc_path_length(x_neural[start:end], y_neural[start:end]) < min_distance_cm:
            continue
        # Filter by max distance to PF peak
        if max_pf_distance_cm is not None and pf_peak_xy is not None:
            x_traj = x_neural[start:end]
            y_traj = y_neural[start:end]
            valid_pos = (~np.isnan(x_traj)) & (~np.isnan(y_traj))
            if np.any(valid_pos):
                dx = x_traj[valid_pos] - pf_peak_xy[0]
                dy = y_traj[valid_pos] - pf_peak_xy[1]
                min_dist = np.min(np.sqrt(dx**2 + dy**2))
                if min_dist > max_pf_distance_cm:
                    continue
        if clear_traversal:
            pre_idx = start - 1
            post_idx = end
            if pre_idx < 0 or post_idx >= len(in_place_field):
                continue
            if not (valid_frames[pre_idx] and valid_frames[post_idx]):
                continue
            if in_place_field[pre_idx] or in_place_field[post_idx]:
                continue
        traversal_epochs.append((start, end))

    traversal_types = []
    if return_traversal_types:
        for start, end in traversal_epochs:
            traversal_types.append(
                _classify_traversal_direction(x_neural[start:end], y_neural[start:end], center_xy)
            )

    if verbose:
        max_dist_str = f", max PF distance: {max_pf_distance_cm} cm" if max_pf_distance_cm is not None else ""
        print(
            f"Found {len(traversal_epochs)} traversal epochs "
            f"(min duration: {min_duration_ms} ms, min distance: {min_distance_cm} cm{max_dist_str})"
        )
        print(
            f"Place field has {np.sum(place_field_mask)} bins, "
            f"peak rate: {peak_rate:.2f} Hz"
        )

    if return_traversal_types:
        return traversal_epochs, place_field_mask, smooth_map, bins, traversal_types
    return traversal_epochs, place_field_mask, smooth_map, bins


def find_outside_place_field_epochs(
    x_neural,
    y_neural,
    speed,
    frame_rate,
    place_field_mask,
    pf_bins,
    speed_threshold=2,
    min_duration_ms=500,
    min_distance_cm=5,
    pf_buffer_sec=2.0,
    speed_smooth_kernel=51,
    merge_gap_s=1.0,
    verbose=True,
):
    """
    Find epochs where the animal is moving continuously OUTSIDE the place field.

    Parameters
    ----------
    x_neural, y_neural : array
        Position coordinates at neural frame rate
    speed : array
        Speed at neural frame rate
    frame_rate : float
        Neural frame rate (Hz)
    place_field_mask : 2D bool array
        The place field mask
    pf_bins : list of arrays
        [x_bins, y_bins] from place field analysis
    speed_threshold : float
        Minimum speed (cm/s) to consider "moving"
    min_duration_ms : float
        Minimum duration of an outside-PF epoch (ms)
    min_distance_cm : float
        Minimum distance traveled during the epoch (cm)
    pf_buffer_sec : float
        Buffer time (s) before and after the epoch where animal must also be outside PF
    speed_smooth_kernel : int
        Kernel size for median filtering speed
    merge_gap_s : float
        Gap (s) within which to merge consecutive outside-PF epochs
    verbose : bool
        Print summary

    Returns
    -------
    outside_pf_epochs : list of (start, end) tuples
        Epochs where the animal is outside the PF continuously
    """
    x_neural = np.asarray(x_neural)
    y_neural = np.asarray(y_neural)
    speed = np.asarray(speed, dtype=float)
    total_frames = len(x_neural)

    if place_field_mask is None or not np.any(place_field_mask):
        if verbose:
            print("No place field mask provided; returning empty list.")
        return []

    valid_frames = (~np.isnan(x_neural)) & (~np.isnan(y_neural)) & (~np.isnan(speed))

    # Get positions inside place field
    in_place_field = _positions_in_place_field(x_neural, y_neural, pf_bins, place_field_mask)

    # Smooth speed
    if speed_smooth_kernel % 2 == 0:
        speed_smooth_kernel += 1
    speed_smooth = median_filter(speed, size=speed_smooth_kernel)

    # Moving and outside the place field
    outside_pf_moving = (~in_place_field) & (speed_smooth > speed_threshold) & valid_frames

    # Find continuous epochs of outside-PF moving
    diff = np.diff(np.concatenate([[0], outside_pf_moving.astype(int), [0]]))
    raw_starts = np.where(diff == 1)[0]
    raw_ends = np.where(diff == -1)[0]
    raw_epochs = list(zip(raw_starts, raw_ends))

    # Merge nearby epochs
    merge_gap_frames = int(round(merge_gap_s * frame_rate))
    merged_epochs = []
    for start, end in raw_epochs:
        if not merged_epochs:
            merged_epochs.append([start, end])
            continue
        prev_start, prev_end = merged_epochs[-1]
        if (start - prev_end) <= merge_gap_frames:
            merged_epochs[-1][1] = end
        else:
            merged_epochs.append([start, end])

    # Filter by duration and distance
    min_duration_frames = int(min_duration_ms / 1000 * frame_rate)
    buffer_frames = int(round(pf_buffer_sec * frame_rate))

    outside_pf_epochs = []
    for start, end in merged_epochs:
        # Check duration
        if (end - start) < min_duration_frames:
            continue
        # Check distance
        if _calc_path_length(x_neural[start:end], y_neural[start:end]) < min_distance_cm:
            continue
        # Check buffer before epoch: animal must be outside PF
        pre_start = max(0, start - buffer_frames)
        if np.any(in_place_field[pre_start:start]):
            continue
        # Check buffer after epoch: animal must be outside PF
        post_end = min(total_frames, end + buffer_frames)
        if np.any(in_place_field[end:post_end]):
            continue
        # Also ensure the epoch itself is continuously outside PF
        if np.any(in_place_field[start:end]):
            continue

        outside_pf_epochs.append((start, end))

    if verbose:
        print(
            f"Found {len(outside_pf_epochs)} outside-PF epochs "
            f"(min duration: {min_duration_ms} ms, min distance: {min_distance_cm} cm, "
            f"buffer: {pf_buffer_sec} s)"
        )

    return outside_pf_epochs


def compute_epoch_centered_averages(
    theta_vm,
    slow_vm,
    epochs,
    frame_rate,
    window_sec=2.0,
    zscore_traces=False,
    ifr=None,
    ifr_ss=None,
    ifr_cs=None,
    bad_timepoints=None,
):
    """
    Compute epoch-centered averages for theta, slow, and optionally firing rate traces.

    Centers on the midpoint of each epoch.

    Parameters
    ----------
    theta_vm, slow_vm : array
        Theta and slow Vm traces
    epochs : list of (start, end) tuples
        Epochs to center on
    frame_rate : float
        Neural frame rate (Hz)
    window_sec : float
        Window (s) before and after center point
    zscore_traces : bool
        Whether to z-score traces
    ifr, ifr_ss, ifr_cs : array or None
        Instantaneous firing rate arrays (total, SS, CS)
    bad_timepoints : array or None
        Boolean mask of bad timepoints to skip

    Returns
    -------
    dict with keys:
        - time_rel : time array relative to center
        - theta_mean, theta_sem : mean and SEM of theta amplitude
        - slow_mean, slow_sem : mean and SEM of slow Vm
        - rate_mean, rate_sem : mean and SEM of firing rate (if ifr provided)
        - ss_rate_mean, ss_rate_sem : mean and SEM of SS rate (if ifr_ss provided)
        - cs_rate_mean, cs_rate_sem : mean and SEM of CS rate (if ifr_cs provided)
        - ss_pct_mean, ss_pct_sem, cs_pct_mean, cs_pct_sem : mean and SEM of percentages
        - n_epochs : number of valid epochs used
    """
    window_frames = int(round(window_sec * frame_rate))
    avg_len = window_frames * 2 + 1
    time_rel = np.arange(-window_frames, window_frames + 1) / frame_rate

    theta_source = _safe_zscore(theta_vm, zscore_traces)
    theta_amp = np.abs(signal.hilbert(theta_source))
    slow_source = _safe_zscore(slow_vm, zscore_traces)

    n_epochs = len(epochs)
    theta_stack = np.full((n_epochs, avg_len), np.nan)
    slow_stack = np.full((n_epochs, avg_len), np.nan)
    rate_stack = np.full((n_epochs, avg_len), np.nan) if ifr is not None else None
    ss_rate_stack = np.full((n_epochs, avg_len), np.nan) if ifr_ss is not None else None
    cs_rate_stack = np.full((n_epochs, avg_len), np.nan) if ifr_cs is not None else None
    ss_pct_stack = (
        np.full((n_epochs, avg_len), np.nan) if (ifr_ss is not None and ifr_cs is not None) else None
    )
    cs_pct_stack = (
        np.full((n_epochs, avg_len), np.nan) if (ifr_ss is not None and ifr_cs is not None) else None
    )

    bad_mask = None
    if bad_timepoints is not None:
        bad_mask = np.asarray(bad_timepoints)
        if bad_mask.dtype != bool:
            bad_bool = np.zeros(len(theta_source), dtype=bool)
            bad_mask = np.asarray(bad_mask, dtype=int)
            bad_mask = bad_mask[(bad_mask >= 0) & (bad_mask < len(theta_source))]
            bad_bool[bad_mask] = True
            bad_mask = bad_bool

    # Baseline windows (2s to 1s before epoch start)
    baseline_start_offset = int(round(2.0 * frame_rate))
    baseline_end_offset = int(round(1.0 * frame_rate))

    valid_count = 0
    for i, (epoch_start, epoch_end) in enumerate(epochs):
        # Center on midpoint of epoch
        center_idx = (epoch_start + epoch_end) // 2

        # Check for bad timepoints in window
        start_idx = max(center_idx - window_frames, 0)
        end_idx = min(center_idx + window_frames + 1, len(theta_source))
        if bad_mask is not None and np.any(bad_mask[start_idx:end_idx]):
            continue

        # Compute baseline
        base_start = max(epoch_start - baseline_start_offset, 0)
        base_end = max(epoch_start - baseline_end_offset, 0)
        baseline_theta = np.nan
        baseline_slow = np.nan
        if base_end > base_start:
            baseline_theta = np.nanmean(theta_amp[base_start:base_end])
            baseline_slow = np.nanmean(slow_source[base_start:base_end])

        # Extract and baseline-subtract
        if end_idx <= start_idx:
            continue

        # Compute where in the output array to write
        # left_pad: how many frames we're missing on the left (due to clipping at 0)
        # right_pad: how many frames we're missing on the right (due to clipping at len)
        left_pad = max(0, -(center_idx - window_frames))  # frames clipped on left
        right_pad = max(0, (center_idx + window_frames + 1) - len(theta_source))  # frames clipped on right
        write_start = left_pad
        write_end = avg_len - right_pad
        
        # Length of segment we actually extracted
        seg_len = end_idx - start_idx
        
        # Sanity check: segment length should match write range
        expected_len = write_end - write_start
        if seg_len != expected_len:
            # Skip this epoch if there's a mismatch (shouldn't happen, but safety check)
            continue

        theta_segment = theta_amp[start_idx:end_idx].copy()
        slow_segment = slow_source[start_idx:end_idx].copy()
        if np.isfinite(baseline_theta):
            theta_segment = theta_segment - baseline_theta
        if np.isfinite(baseline_slow):
            slow_segment = slow_segment - baseline_slow

        theta_stack[i, write_start:write_end] = theta_segment
        slow_stack[i, write_start:write_end] = slow_segment

        if ifr is not None:
            rate_stack[i, write_start:write_end] = ifr[start_idx:end_idx]
        if ifr_ss is not None:
            ss_rate_stack[i, write_start:write_end] = ifr_ss[start_idx:end_idx]
        if ifr_cs is not None:
            cs_rate_stack[i, write_start:write_end] = ifr_cs[start_idx:end_idx]
        if ifr_ss is not None and ifr_cs is not None:
            ss_seg = ifr_ss[start_idx:end_idx]
            cs_seg = ifr_cs[start_idx:end_idx]
            denom = ss_seg + cs_seg
            valid = np.isfinite(denom) & (denom > 0)
            ss_pct = np.where(valid, 100.0 * ss_seg / denom, np.nan)
            cs_pct = np.where(valid, 100.0 * cs_seg / denom, np.nan)
            ss_pct_stack[i, write_start:write_end] = ss_pct
            cs_pct_stack[i, write_start:write_end] = cs_pct

        valid_count += 1

    # Compute means and SEMs
    def _mean_sem(stack):
        if stack is None:
            return None, None
        mean = np.nanmean(stack, axis=0)
        count = np.sum(np.isfinite(stack), axis=0)
        std = np.nanstd(stack, axis=0, ddof=0)
        sem = np.where(count > 0, std / np.sqrt(count), np.nan)
        return mean, sem

    theta_mean, theta_sem = _mean_sem(theta_stack)
    slow_mean, slow_sem = _mean_sem(slow_stack)
    rate_mean, rate_sem = _mean_sem(rate_stack)
    ss_rate_mean, ss_rate_sem = _mean_sem(ss_rate_stack)
    cs_rate_mean, cs_rate_sem = _mean_sem(cs_rate_stack)
    ss_pct_mean, ss_pct_sem = _mean_sem(ss_pct_stack)
    cs_pct_mean, cs_pct_sem = _mean_sem(cs_pct_stack)

    return {
        "time_rel": time_rel,
        "theta_mean": theta_mean,
        "theta_sem": theta_sem,
        "slow_mean": slow_mean,
        "slow_sem": slow_sem,
        "rate_mean": rate_mean,
        "rate_sem": rate_sem,
        "ss_rate_mean": ss_rate_mean,
        "ss_rate_sem": ss_rate_sem,
        "cs_rate_mean": cs_rate_mean,
        "cs_rate_sem": cs_rate_sem,
        "ss_pct_mean": ss_pct_mean,
        "ss_pct_sem": ss_pct_sem,
        "cs_pct_mean": cs_pct_mean,
        "cs_pct_sem": cs_pct_sem,
        "n_epochs": valid_count,
    }


def plot_place_field_traversal_trials(
    trace,
    theta_vm,
    slow_vm,
    speed,
    x_neural,
    y_neural,
    traversal_epochs,
    place_field_mask,
    pf_bins,
    frame_rate,
    cell_idx=None,
    padding_sec=2.0,
    zscore_traces=False,
    speed_smooth_kernel=51,
    resting_speed_threshold=5,
    rest_merge_gap_s=0.5,
    traversal_types=None,
    clockwise=False,
    counterclockwise=False,
    session_start_frames=None,
    color_by_session=False,
    session_indices=None,
    bad_timepoints=None,
    show_rest_patches=True,
    theta_freqs=None,
    slow_freqs=None,
    refined_SS=None,
    all_CS_spikes=None,
    all_spikes=None,
    firing_rate_bin_ms=100,
    firing_rate_smooth_ms=20,
    firing_rate_color="#555555",
    firing_rate_alpha=0.6,
    firing_rate_linewidth=0.5,
    show_firing_rate=False,
    simple_spike_color="#026C80",
    complex_spike_color="#EE9B00",
    spike_tick_size=4,
    pf_peak_xy=None,
    pf_center_window_sec=2.0,
    plot_pf_centered_average=True,
    show=True,
    return_pf_centered=False,
    normalize_slow_vm=False,
    normalize_slow_baseline=False,
    plot_pf_centered_rate=True,
    axes=None,
    apply_layout=True,
    return_pf_centered_sem=False,
    return_spike_rate_means=False,
):
    trace_z = _safe_zscore(trace, zscore_traces)
    theta_z = _safe_zscore(theta_vm, zscore_traces)
    slow_z = _safe_zscore(slow_vm, zscore_traces)
    slow_plot = slow_vm.copy() if normalize_slow_vm else slow_vm
    if normalize_slow_vm:
        mean_slow = np.nanmean(slow_vm)
        std_slow = np.nanstd(slow_vm)
        if np.isfinite(std_slow) and std_slow != 0:
            slow_plot = (slow_vm - mean_slow) / std_slow
        else:
            slow_plot = slow_vm - mean_slow

    if speed_smooth_kernel % 2 == 0:
        speed_smooth_kernel += 1
    speed_smooth = median_filter(speed, size=speed_smooth_kernel)

    padding_frames = int(padding_sec * frame_rate)

    # Initialize epoch-type pairs to maintain mapping through filtering
    if traversal_types is not None and len(traversal_types) == len(traversal_epochs):
        epochs_with_types = list(zip(traversal_epochs, traversal_types))
    else:
        # Default to "unknown" if types not provided
        epochs_with_types = [(epoch, "unknown") for epoch in traversal_epochs]

    # Filter by clockwise/counterclockwise if requested
    if clockwise ^ counterclockwise:
        if traversal_types is None or len(traversal_types) != len(traversal_epochs):
            print("Traversal types not provided or length mismatch; plotting all traversals.")
        else:
            target = "cw" if clockwise else "ccw"
            epochs_with_types = [
                (epoch, ttype)
                for epoch, ttype in epochs_with_types
                if ttype == target
            ]

    # Filter by session indices
    if session_indices is not None:
        if session_start_frames is None:
            print("session_start_frames not provided; plotting all sessions.")
        else:
            session_start_frames = np.asarray(session_start_frames, dtype=int)
            session_start_frames = session_start_frames[np.argsort(session_start_frames)]
            if np.isscalar(session_indices):
                session_indices = [int(session_indices)]
            session_indices = [int(i) for i in session_indices]
            epochs_filtered = []
            for (epoch_start, epoch_end), ttype in epochs_with_types:
                session_idx = np.searchsorted(session_start_frames, epoch_start, side="right") - 1
                if session_idx in session_indices:
                    epochs_filtered.append(((epoch_start, epoch_end), ttype))
            epochs_with_types = epochs_filtered

    bad_mask = None
    if bad_timepoints is not None:
        bad_mask = np.asarray(bad_timepoints)
        if bad_mask.dtype != bool:
            if bad_mask.ndim == 1 and bad_mask.size == len(trace_z):
                bad_mask = bad_mask.astype(bool)
            else:
                bad_mask = np.asarray(bad_mask, dtype=int)
                bad_mask = bad_mask[(bad_mask >= 0) & (bad_mask < len(trace_z))]
                bad_bool = np.zeros(len(trace_z), dtype=bool)
                bad_bool[bad_mask] = True
                bad_mask = bad_bool
        if bad_mask.shape[0] != len(trace_z):
            raise ValueError("bad_timepoints must match trace length or be index list.")

    # Filter by bad timepoints and extract valid epochs with their types
    valid_epochs = []
    valid_traversal_types = []
    for (epoch_start, epoch_end), ttype in epochs_with_types:
        start_idx = epoch_start - padding_frames
        end_idx = epoch_end + padding_frames
        if start_idx < 0 or end_idx > len(trace_z):
            continue
        if bad_mask is not None and np.any(bad_mask[start_idx:end_idx]):
            continue
        valid_epochs.append((epoch_start, epoch_end))
        valid_traversal_types.append(ttype)

    n_trials = len(valid_epochs)
    if n_trials == 0:
        print("No valid traversal epochs with sufficient padding")
        if return_pf_centered and return_spike_rate_means and return_pf_centered_sem:
            return (
                None,
                None,
                [],
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )
        if return_pf_centered and return_spike_rate_means:
            return None, None, [], None, None, None, None, None, None, None
        if return_pf_centered and return_pf_centered_sem:
            return None, None, [], None, None, None, None, None, None
        if return_pf_centered:
            return None, None, [], None, None, None
        return None, None, []

    trace_mins, trace_maxs = [], []
    theta_mins, theta_maxs = [], []
    slow_mins, slow_maxs = [], []
    for epoch_start, epoch_end in valid_epochs:
        start_idx = epoch_start - padding_frames
        end_idx = epoch_end + padding_frames
        trace_mins.append(np.nanmin(trace_z[start_idx:end_idx]))
        trace_maxs.append(np.nanmax(trace_z[start_idx:end_idx]))
        theta_mins.append(np.nanmin(theta_z[start_idx:end_idx]))
        theta_maxs.append(np.nanmax(theta_z[start_idx:end_idx]))
        slow_mins.append(np.nanmin(slow_z[start_idx:end_idx]))
        slow_maxs.append(np.nanmax(slow_z[start_idx:end_idx]))

    trace_min_global = min(trace_mins)
    theta_max_global = max(theta_maxs)
    theta_min_global = min(theta_mins)
    slow_max_global = max(slow_maxs)
    slow_min_global = min(slow_mins)

    trace_range = max(trace_maxs) - min(trace_mins)
    theta_range = theta_max_global - theta_min_global
    slow_range = slow_max_global - slow_min_global

    # Spacing between the stacked signals in the same axis.
    # Keep slow Vm close to the trace, and theta below slow.
    pad_trace_slow = 0.005 * max(trace_range, slow_range)
    pad_slow_theta = 0.05 * max(slow_range, theta_range)

    # We plot:
    #   trace at 0
    #   slow_vm at -offset1
    #   theta_vm at -(offset1 + offset2)
    offset1 = slow_max_global - trace_min_global + pad_trace_slow
    offset2 = theta_max_global - slow_min_global + pad_slow_theta

    max_epoch_duration = max((e - s) for s, e in valid_epochs)
    global_x_min = -padding_sec
    global_x_max = max_epoch_duration / frame_rate + padding_sec

    def _get_spike_indices(spike_source):
        if spike_source is None:
            return None
        if (
            cell_idx is not None
            and isinstance(spike_source, (list, tuple))
            and len(spike_source) > 0
            and isinstance(spike_source[0], (list, np.ndarray))
        ):
            return np.asarray(spike_source[cell_idx], dtype=int)
        return np.asarray(spike_source, dtype=int)

    simple_spikes = _get_spike_indices(refined_SS)
    complex_spikes = _get_spike_indices(all_CS_spikes)
    all_spike_indices = _get_spike_indices(all_spikes)

    def _compute_ifr(spike_indices):
        if spike_indices is None or len(spike_indices) == 0:
            return None
        spike_indices = spike_indices[(spike_indices >= 0) & (spike_indices < len(trace_z))]
        if spike_indices.size == 0:
            return None
        spike_train = np.zeros(len(trace_z), dtype=float)
        spike_train[spike_indices] = 1.0
        bin_frames = max(1, int(round(firing_rate_bin_ms / 1000 * frame_rate)))
        bin_kernel = np.ones(bin_frames, dtype=float) / bin_frames
        ifr_local = np.convolve(spike_train, bin_kernel, mode="same") * frame_rate
        smooth_frames = max(1, int(round(firing_rate_smooth_ms / 1000 * frame_rate)))
        if smooth_frames > 1:
            smooth_kernel = np.ones(smooth_frames, dtype=float) / smooth_frames
            ifr_local = np.convolve(ifr_local, smooth_kernel, mode="same")
        return ifr_local

    ifr = None
    if (
        (show_firing_rate or (plot_pf_centered_average and plot_pf_centered_rate))
        and all_spike_indices is not None
        and len(all_spike_indices) > 0
    ):
        ifr = _compute_ifr(all_spike_indices)

    ifr_ss = None
    ifr_cs = None
    if (
        return_spike_rate_means
        and plot_pf_centered_average
        and plot_pf_centered_rate
        and pf_peak_xy is not None
    ):
        ifr_ss = _compute_ifr(simple_spikes)
        ifr_cs = _compute_ifr(complex_spikes)

    include_pf_rate = (
        plot_pf_centered_average and pf_peak_xy is not None and plot_pf_centered_rate and ifr is not None
    )
    include_pf_ss_rate = (
        return_spike_rate_means
        and plot_pf_centered_average
        and pf_peak_xy is not None
        and plot_pf_centered_rate
        and ifr_ss is not None
    )
    include_pf_cs_rate = (
        return_spike_rate_means
        and plot_pf_centered_average
        and pf_peak_xy is not None
        and plot_pf_centered_rate
        and ifr_cs is not None
    )
    include_pf_ss_cs_rate = include_pf_ss_rate or include_pf_cs_rate
    include_pf_pct = (
        return_spike_rate_means
        and plot_pf_centered_average
        and pf_peak_xy is not None
        and plot_pf_centered_rate
        and ifr_ss is not None
        and ifr_cs is not None
    )
    pf_rate_rows = int(include_pf_rate) + int(include_pf_ss_cs_rate) + int(include_pf_pct)
    include_pf_panel = plot_pf_centered_average and pf_peak_xy is not None
    extra_rows = (pf_rate_rows + 2) if include_pf_panel else 0
    pf_heights = ([1.1] * pf_rate_rows + [1.4, 1.4]) if include_pf_panel else []
    # Keep a visible separator row between trial-by-trial rows and grand-average rows.
    separator_height = 0.35 if extra_rows else 0.15
    height_ratios = [1] * n_trials + [separator_height] + (pf_heights if extra_rows else [])
    if axes is None:
        fig, axes = plt.subplots(
            n_trials + 1 + extra_rows,
            2,
            figsize=(6, 0.6 * n_trials + 0.3 + (1.4 * extra_rows)),
            gridspec_kw={
                "width_ratios": [5, 1],
                "hspace": 0,
                "wspace": 0.01,
                "height_ratios": height_ratios,
            },
        )
    else:
        axes = np.asarray(axes)
        fig = axes[0, 0].figure

    merge_gap_frames = int(rest_merge_gap_s * frame_rate)
    session_colors = None
    epoch_sessions = None
    if color_by_session and session_start_frames is not None:
        session_start_frames = np.asarray(session_start_frames, dtype=int)
        session_start_frames = session_start_frames[np.argsort(session_start_frames)]
        num_sessions = len(session_start_frames)
        if num_sessions > 0:
            base_colors = [
                "#000000",
                "#e41a1c",
                "#377eb8",
                "#4daf4a",
                "#984ea3",
                "#ff7f00",
                "#ffff33",
                "#a65628",
                "#f781bf",
                "#999999",
            ]
            if num_sessions <= len(base_colors):
                session_colors = base_colors[:num_sessions]
            else:
                extra = plt.cm.hsv(
                    np.linspace(0, 1, num_sessions - len(base_colors), endpoint=False)
                )
                session_colors = base_colors + [tuple(c) for c in extra]
            epoch_sessions = [
                np.searchsorted(session_start_frames, s, side="right") - 1
                for s, _ in valid_epochs
            ]

    pf_center_indices = []
    for trial_idx, (epoch_start, epoch_end) in enumerate(valid_epochs):
        ax_trace = axes[trial_idx, 0]
        ax_traj = axes[trial_idx, 1]

        start_idx = epoch_start - padding_frames
        end_idx = epoch_end + padding_frames
        time_rel = (np.arange(start_idx, end_idx) - epoch_start) / frame_rate

        trace_window = trace_z[start_idx:end_idx]
        theta_window = theta_z[start_idx:end_idx]
        slow_source = slow_z if zscore_traces else slow_plot
        slow_window = slow_source[start_idx:end_idx]
        if normalize_slow_baseline:
            baseline = np.nan
            if epoch_start > start_idx:
                baseline = np.nanmean(slow_source[start_idx:epoch_start])
            if np.isfinite(baseline):
                slow_window = slow_window - baseline

        epoch_duration = (epoch_end - epoch_start) / frame_rate
        ax_trace.axvspan(0, epoch_duration, alpha=0.2, color="green", zorder=0)

        if ifr is not None and show_firing_rate:
            ifr_window = ifr[start_idx:end_idx]
            trace_min = np.nanmin(trace_window)
            trace_max = np.nanmax(trace_window)
            trace_range = trace_max - trace_min
            if not np.isfinite(trace_range) or trace_range == 0:
                trace_range = 1.0
            ifr_min = np.nanmin(ifr_window)
            ifr_max = np.nanmax(ifr_window)
            ifr_range = ifr_max - ifr_min
            if not np.isfinite(ifr_range) or ifr_range == 0:
                ifr_range = 1.0
            ifr_norm = (ifr_window - ifr_min) / ifr_range
            ifr_scaled = trace_min + ifr_norm * trace_range
            ax_trace.plot(
                time_rel,
                ifr_scaled,
                color=firing_rate_color,
                linewidth=firing_rate_linewidth,
                alpha=firing_rate_alpha,
                zorder=1,
            )

        trace_range_local = np.nanmax(trace_window) - np.nanmin(trace_window)
        if not np.isfinite(trace_range_local) or trace_range_local == 0:
            trace_range_local = 0.1
        spike_tick_y = np.nanmax(trace_window) + 0.05 * trace_range_local

        if simple_spikes is not None:
            ss_mask = (simple_spikes >= start_idx) & (simple_spikes < end_idx)
            ss_times = (simple_spikes[ss_mask] - epoch_start) / frame_rate
            if ss_times.size > 0:
                ax_trace.plot(
                    ss_times,
                    np.full_like(ss_times, spike_tick_y, dtype=float),
                    linestyle="None",
                    marker="|",
                    markersize=spike_tick_size,
                    markeredgewidth=0.5,
                    color=simple_spike_color,
                    zorder=3,
                )
        if complex_spikes is not None:
            cs_mask = (complex_spikes >= start_idx) & (complex_spikes < end_idx)
            cs_times = (complex_spikes[cs_mask] - epoch_start) / frame_rate
            if cs_times.size > 0:
                ax_trace.plot(
                    cs_times,
                    np.full_like(cs_times, spike_tick_y, dtype=float),
                    linestyle="None",
                    marker="|",
                    markersize=spike_tick_size,
                    markeredgewidth=0.5,
                    color=complex_spike_color,
                    zorder=3,
                )

        x_traj_full = x_neural[start_idx:end_idx]
        y_traj_full = y_neural[start_idx:end_idx]
        in_pf_window = _positions_in_place_field(
            x_traj_full, y_traj_full, pf_bins, place_field_mask
        )
        speed_window_raw = speed[start_idx:end_idx]
        resting_in_pf = in_pf_window & (speed_window_raw < resting_speed_threshold)

        pre_frames = padding_frames
        post_start_idx = epoch_end - start_idx

        if show_rest_patches:
            pre_resting = resting_in_pf[:pre_frames]
            if np.any(pre_resting):
                diff = np.diff(np.concatenate([[0], pre_resting.astype(int), [0]]))
                starts = np.where(diff == 1)[0]
                ends = np.where(diff == -1)[0]
                starts, ends = _merge_segments(starts, ends, merge_gap_frames)
                for s, e in zip(starts, ends):
                    t_start = time_rel[s]
                    t_end = time_rel[min(e, pre_frames - 1)]
                    ax_trace.axvspan(t_start, t_end, alpha=0.2, color="gold", zorder=0)

            post_resting = resting_in_pf[post_start_idx:]
            if np.any(post_resting):
                diff = np.diff(np.concatenate([[0], post_resting.astype(int), [0]]))
                starts = np.where(diff == 1)[0]
                ends = np.where(diff == -1)[0]
                starts, ends = _merge_segments(starts, ends, merge_gap_frames)
                for s, e in zip(starts, ends):
                    t_start = time_rel[post_start_idx + s]
                    t_end = time_rel[min(post_start_idx + e, len(time_rel) - 1)]
                    ax_trace.axvspan(t_start, t_end, alpha=0.2, color="gold", zorder=0)

        ax_trace.plot(
            time_rel,
            trace_window,
            color="black",
            linewidth=0.25,
            label="Trace",
            zorder=2,
        )
        # Local offsets per trial so slow Vm sits a fixed amount below the trace.
        # Target: slow_max is ~0.05 below trace_min (in the plotted units).
        slow_trace_gap = 0.0
        trace_min_local = np.nanmin(trace_window) if np.any(np.isfinite(trace_window)) else 0.0
        slow_max_local = np.nanmax(slow_window) if np.any(np.isfinite(slow_window)) else 0.0
        slow_min_local = np.nanmin(slow_window) if np.any(np.isfinite(slow_window)) else 0.0
        theta_max_local = np.nanmax(theta_window) if np.any(np.isfinite(theta_window)) else 0.0
        offset1 = max(0.0, float(slow_max_local - trace_min_local + slow_trace_gap))
        offset2 = max(0.0, float(theta_max_local - slow_min_local + pad_slow_theta))
        theta_label = "Theta (4-8 Hz)"
        if theta_freqs is not None and len(theta_freqs) == 2:
            theta_label = f"Theta ({theta_freqs[0]}-{theta_freqs[1]} Hz)"
        slow_label = "Slow (<=2 Hz)"
        if slow_freqs is not None:
            if np.isscalar(slow_freqs):
                slow_label = f"Slow (<= {slow_freqs} Hz)"
            elif len(slow_freqs) == 2:
                slow_label = f"Slow ({slow_freqs[0]}-{slow_freqs[1]} Hz)"
        ax_trace.plot(
            time_rel,
            slow_window - offset1,
            color="red",
            linewidth=0.5,
            label=slow_label,
            zorder=2,
        )
        ax_trace.plot(
            time_rel,
            theta_window - offset1 - offset2,
            color="blue",
            linewidth=0.5,
            label=theta_label,
            zorder=2,
        )

        x_data_max = time_rel[-1]
        tick_len = 0.08
        ax_trace.plot(
            [x_data_max - tick_len, x_data_max],
            [0, 0],
            color="black",
            linewidth=0.5,
            alpha=0.5,
        )
        ax_trace.plot(
            [x_data_max - tick_len, x_data_max],
            [-offset1, -offset1],
            color="red",
            linewidth=0.5,
            alpha=0.5,
        )
        ax_trace.plot(
            [x_data_max - tick_len, x_data_max],
            [-offset1 - offset2, -offset1 - offset2],
            color="blue",
            linewidth=0.5,
            alpha=0.5,
        )

        trial_color = "black"
        if session_colors is not None and epoch_sessions is not None:
            session_idx = epoch_sessions[trial_idx]
            if 0 <= session_idx < len(session_colors):
                trial_color = session_colors[session_idx]
        # Build trial label with direction type
        trial_type = valid_traversal_types[trial_idx] if trial_idx < len(valid_traversal_types) else "?"
        type_label = _format_trial_type_label(trial_type)
        trial_label = f"{trial_idx + 1}" + (
            f" {type_label}" if type_label and not bool(trajectory_direction_label_top) else ""
        )
        ax_trace.text(
            time_rel[0] - 0.2,
            trace_window[0] + 0.1,
            trial_label,
            fontsize=6,
            ha="left",
            va="bottom",
            fontweight="bold",
            color=trial_color,
        )
        ax_trace.text(
            -0.6,
            0.3,
            f"{epoch_start / frame_rate:.1f}s",
            transform=ax_trace.get_xaxis_transform(),
            ha="left",
            va="bottom",
            fontsize=6,
            fontname="Arial",
            color="black",
        )
        # ax_trace.text(
        #     epoch_duration + 0.65,
        #     0.05,
        #     f"{epoch_end / frame_rate:.1f}s",
        #     transform=ax_trace.get_xaxis_transform(),
        #     ha="right",
        #     va="bottom",
        #     fontsize=5,
        #     fontname="Arial",
        #     color="black",
        # )

        ax_trace.spines["top"].set_visible(False)
        ax_trace.spines["right"].set_visible(False)
        ax_trace.spines["left"].set_visible(False)
        ax_trace.spines["bottom"].set_visible(False)
        ax_trace.set_yticks([])
        ax_trace.set_xticks([])
        ax_trace.set_xlim([global_x_min, global_x_max])

        if trial_idx == 0:
            handles, labels = ax_trace.get_legend_handles_labels()
            if refined_SS is not None:
                handles.append(
                    Line2D(
                        [0],
                        [0],
                        marker="|",
                        color=simple_spike_color,
                        linestyle="None",
                        markersize=spike_tick_size,
                        markeredgewidth=0.5,
                        label="Simple spikes",
                    )
                )
                labels.append("Simple spikes")
            if all_CS_spikes is not None:
                handles.append(
                    Line2D(
                        [0],
                        [0],
                        marker="|",
                        color=complex_spike_color,
                        linestyle="None",
                        markersize=spike_tick_size,
                        markeredgewidth=0.5,
                        label="Complex spikes",
                    )
                )
                labels.append("Complex spikes")
            handle_map = dict(zip(labels, handles))
            ordered = []
            for label in ["Trace", slow_label, theta_label]:
                if label in handle_map:
                    ordered.append((label, handle_map[label]))
            if refined_SS is not None:
                ordered.append(
                    (
                        "Simple spikes",
                        Line2D(
                            [0],
                            [0],
                            marker="|",
                            color=simple_spike_color,
                            linestyle="None",
                            markersize=spike_tick_size,
                            markeredgewidth=0.5,
                        ),
                    )
                )
            if all_CS_spikes is not None:
                ordered.append(
                    (
                        "Complex spikes",
                        Line2D(
                            [0],
                            [0],
                            marker="|",
                            color=complex_spike_color,
                            linestyle="None",
                            markersize=spike_tick_size,
                            markeredgewidth=0.5,
                        ),
                    )
                )
            legend_labels = [label for label, _ in ordered]
            legend_handles = [handle for _, handle in ordered]
            legend = ax_trace.legend(
                legend_handles,
                legend_labels,
                loc="lower left",
                bbox_to_anchor=(0.0, 1.02),
                fontsize=5,
                frameon=False,
                ncol=3,
                borderaxespad=0,
            )
            if cell_idx is not None:
                direction_label = "all"
                if clockwise ^ counterclockwise:
                    direction_label = "cw" if clockwise else "ccw"
                title_y = 1.08
                try:
                    fig = ax_trace.figure
                    fig.canvas.draw()
                    legend_bbox = legend.get_window_extent(fig.canvas.get_renderer())
                    legend_bbox_axes = legend_bbox.transformed(ax_trace.transAxes.inverted())
                    title_y = legend_bbox_axes.y1 + 0.05
                except Exception:
                    title_y = 1.08
                ax_trace.set_title(
                    f"Cell {cell_idx} - Place Field Traversals ({direction_label}, n={n_trials})",
                    fontsize=6,
                    pad=0,
                    y=title_y,
                )
        x_traj = x_neural[start_idx:end_idx]
        y_traj = y_neural[start_idx:end_idx]
        post_start = epoch_end - start_idx

        ax_traj.plot(
            x_traj[:pre_frames], y_traj[:pre_frames], color="#87CEEB", linewidth=0.6, alpha=0.7
        )

        x_epoch = x_neural[epoch_start:epoch_end]
        y_epoch = y_neural[epoch_start:epoch_end]
        ax_traj.plot(x_epoch, y_epoch, color="green", linewidth=1.2)

        ax_traj.plot(
            x_traj[post_start:],
            y_traj[post_start:],
            color="#FFB6C1",
            linewidth=0.6,
            alpha=0.7,
        )

        pf_center_idx = None
        if pf_peak_xy is not None and np.all(np.isfinite(pf_peak_xy)):
            ax_traj.scatter(
                pf_peak_xy[0],
                pf_peak_xy[1],
                color="magenta",
                s=20,
                marker="*",
                zorder=6,
                label="PF peak" if trial_idx == 0 else None,
            )
            x_epoch = x_neural[epoch_start:epoch_end]
            y_epoch = y_neural[epoch_start:epoch_end]
            valid_epoch = (~np.isnan(x_epoch)) & (~np.isnan(y_epoch))
            if np.any(valid_epoch):
                dx = x_epoch[valid_epoch] - pf_peak_xy[0]
                dy = y_epoch[valid_epoch] - pf_peak_xy[1]
                dist = np.sqrt(dx**2 + dy**2)
                closest_idx = np.argmin(dist)
                valid_indices = np.where(valid_epoch)[0]
                pf_center_idx = epoch_start + valid_indices[closest_idx]
                cx = x_epoch[valid_epoch][closest_idx]
                cy = y_epoch[valid_epoch][closest_idx]
                ax_traj.scatter(
                    cx,
                    cy,
                    color="yellow",
                    edgecolor="black",
                    linewidth=0.3,
                    s=18,
                    marker="*",
                    zorder=7,
                    label="Closest to PF peak" if trial_idx == 0 else None,
                )
                ax_trace.axvline(
                    (pf_center_idx - epoch_start) / frame_rate,
                    color="yellow",
                    linestyle="--",
                    linewidth=0.6,
                    zorder=1,
                )
        pf_center_indices.append(pf_center_idx)

        valid_epoch_idx = np.where(np.isfinite(x_epoch) & np.isfinite(y_epoch))[0]
        if valid_epoch_idx.size >= 2:
            # Entry arrow: first valid movement direction.
            i0, i1 = int(valid_epoch_idx[0]), int(valid_epoch_idx[1])
            dx0 = float(x_epoch[i1] - x_epoch[i0])
            dy0 = float(y_epoch[i1] - y_epoch[i0])
            n0 = float(np.hypot(dx0, dy0))
            if n0 > 0:
                ux0, uy0 = dx0 / n0, dy0 / n0
                arrow_len0 = min(1.0, max(0.25, 0.35 * n0))
                sx0, sy0 = float(x_epoch[i0]), float(y_epoch[i0])
                hx0, hy0 = sx0 + ux0 * arrow_len0, sy0 + uy0 * arrow_len0
                ax_traj.annotate(
                    "",
                    xy=(hx0, hy0),
                    xytext=(sx0, sy0),
                    arrowprops=dict(arrowstyle="-|>", color="blue", lw=0.5, mutation_scale=6),
                    zorder=6,
                )

            # Exit arrow: last valid movement direction ending at exit.
            j0, j1 = int(valid_epoch_idx[-2]), int(valid_epoch_idx[-1])
            dx1 = float(x_epoch[j1] - x_epoch[j0])
            dy1 = float(y_epoch[j1] - y_epoch[j0])
            n1 = float(np.hypot(dx1, dy1))
            if n1 > 0:
                ux1, uy1 = dx1 / n1, dy1 / n1
                arrow_len1 = min(1.0, max(0.25, 0.35 * n1))
                ex1, ey1 = float(x_epoch[j1]), float(y_epoch[j1])
                tx1, ty1 = ex1 - ux1 * arrow_len1, ey1 - uy1 * arrow_len1
                ax_traj.annotate(
                    "",
                    xy=(ex1, ey1),
                    xytext=(tx1, ty1),
                    arrowprops=dict(arrowstyle="-|>", color="red", lw=0.5, mutation_scale=6),
                    zorder=6,
                )

        if np.any(place_field_mask):
            padded_mask = np.zeros(
                (place_field_mask.shape[0] + 2, place_field_mask.shape[1] + 2), dtype=bool
            )
            padded_mask[1:-1, 1:-1] = place_field_mask
            extent = (pf_bins[0][0], pf_bins[0][-1], pf_bins[1][0], pf_bins[1][-1])
            bin_x = (extent[1] - extent[0]) / place_field_mask.shape[0]
            bin_y = (extent[3] - extent[2]) / place_field_mask.shape[1]
            padded_extent = (
                extent[0] - bin_x,
                extent[1] + bin_x,
                extent[2] - bin_y,
                extent[3] + bin_y,
            )
            ax_traj.contour(
                padded_mask.T,
                levels=[0.5],
                colors="magenta",
                linewidths=0.8,
                extent=padded_extent,
                origin="lower",
            )

        if trial_idx == 0:
            handles, labels = ax_traj.get_legend_handles_labels()
            if isinstance(place_field_contours, (list, tuple)) and len(place_field_contours) > 0:
                from matplotlib.lines import Line2D

                added_pf_labels = set(labels)
                for contour_idx, contour_info in enumerate(place_field_contours, start=1):
                    if not isinstance(contour_info, dict):
                        continue
                    contour_mask = np.asarray(contour_info.get("mask", []), dtype=bool)
                    if contour_mask.size == 0 or not np.any(contour_mask):
                        continue
                    contour_color = str(contour_info.get("color", "magenta"))
                    pf_label = f"PF{contour_idx}"
                    if pf_label in added_pf_labels:
                        continue
                    handles.append(
                        Line2D([0], [0], color=contour_color, linewidth=float(contour_info.get("linewidth", 0.8)))
                    )
                    labels.append(pf_label)
                    added_pf_labels.add(pf_label)
            ax_traj.legend(
                handles,
                labels,
                loc="lower left",
                bbox_to_anchor=(0.0, 1.02),
                fontsize=5,
                frameon=False,
                handletextpad=0.1,
                borderpad=0.0,
                borderaxespad=0,
            )

        ax_traj.set_xlim([pf_bins[0][0], pf_bins[0][-1]])
        ax_traj.set_ylim([pf_bins[1][0], pf_bins[1][-1]])
        ax_traj.set_aspect("equal")
        ax_traj.axis("off")

    ax_scale = axes[n_trials, 0]
    ax_scale_traj = axes[n_trials, 1]
    ax_scale.set_xlim([global_x_min, global_x_max])

    scale_x_start = global_x_min + 0.2
    scale_x_end = scale_x_start + 1.0
    ax_scale.plot(
        [scale_x_start, scale_x_end],
        [0.5, 0.5],
        color="black",
        linewidth=1.0,
        solid_capstyle="butt",
    )
    ax_scale.text(
        (scale_x_start + scale_x_end) / 2,
        0.1,
        "1 s",
        fontsize=6,
        ha="center",
        va="top",
    )
    _add_vertical_scale_bar(ax_scale, 0.6, "1 spk", x_pos=0.92, align="left")

    ax_scale.set_ylim([0, 1])
    ax_scale.axis("off")
    ax_scale_traj.axis("off")

    if apply_layout:
        # Use deterministic subplot spacing (avoid tight_layout here).
        # tight_layout can collapse rows in this figure, especially with many
        # stacked trial rows plus added summary rows, causing visual overlap.
        hspace_val = 0.12 if extra_rows else 0.0
        plt.subplots_adjust(hspace=hspace_val, wspace=0.01, top=0.9, bottom=0.06)
    if show:
        plt.show()

    pf_theta_mean = None
    pf_slow_mean = None
    pf_rate_mean = None
    pf_ss_rate_mean = None
    pf_cs_rate_mean = None
    pf_ss_pct_mean = None
    pf_cs_pct_mean = None
    pf_theta_sem = None
    pf_slow_sem = None
    pf_rate_sem = None
    pf_ss_rate_sem = None
    pf_cs_rate_sem = None
    pf_ss_pct_sem = None
    pf_cs_pct_sem = None
    if plot_pf_centered_average and pf_peak_xy is not None:
        window_frames = int(round(pf_center_window_sec * frame_rate))
        if window_frames > 0:
            avg_len = window_frames * 2 + 1
            time_rel = np.arange(-window_frames, window_frames + 1) / frame_rate
            theta_source = theta_z if zscore_traces else theta_vm
            theta_amp = np.abs(signal.hilbert(theta_source))
            slow_source = slow_z if zscore_traces else slow_plot
            theta_stack = np.full((n_trials, avg_len), np.nan)
            slow_stack = np.full((n_trials, avg_len), np.nan)
            rate_stack = np.full((n_trials, avg_len), np.nan) if include_pf_rate else None
            ss_rate_stack = np.full((n_trials, avg_len), np.nan) if include_pf_ss_rate else None
            cs_rate_stack = np.full((n_trials, avg_len), np.nan) if include_pf_cs_rate else None
            ss_pct_stack = np.full((n_trials, avg_len), np.nan) if include_pf_pct else None
            cs_pct_stack = np.full((n_trials, avg_len), np.nan) if include_pf_pct else None
            baseline_start_offset = int(round(2.0 * frame_rate))
            baseline_end_offset = int(round(1.0 * frame_rate))
            for i, (center_idx, (epoch_start, _)) in enumerate(
                zip(pf_center_indices, valid_epochs)
            ):
                if center_idx is None:
                    continue
                base_start = max(epoch_start - baseline_start_offset, 0)
                base_end = max(epoch_start - baseline_end_offset, 0)
                baseline_theta = np.nan
                baseline_slow = np.nan
                if base_end > base_start:
                    baseline_theta = np.nanmean(theta_amp[base_start:base_end])
                    baseline_slow = np.nanmean(slow_source[base_start:base_end])
                start_idx = max(center_idx - window_frames, 0)
                end_idx = min(center_idx + window_frames + 1, len(theta_source))
                if end_idx <= start_idx:
                    continue
                left_pad = start_idx - (center_idx - window_frames)
                right_pad = (center_idx + window_frames + 1) - end_idx
                write_start = max(0, left_pad)
                write_end = avg_len - max(0, right_pad)
                theta_segment = theta_amp[start_idx:end_idx]
                slow_segment = slow_source[start_idx:end_idx]
                if np.isfinite(baseline_theta):
                    theta_segment = theta_segment - baseline_theta
                if np.isfinite(baseline_slow):
                    slow_segment = slow_segment - baseline_slow
                theta_stack[i, write_start:write_end] = theta_segment
                slow_stack[i, write_start:write_end] = slow_segment
                if include_pf_rate and ifr is not None:
                    rate_stack[i, write_start:write_end] = ifr[start_idx:end_idx]
                if include_pf_ss_rate and ifr_ss is not None:
                    ss_rate_stack[i, write_start:write_end] = ifr_ss[start_idx:end_idx]
                if include_pf_cs_rate and ifr_cs is not None:
                    cs_rate_stack[i, write_start:write_end] = ifr_cs[start_idx:end_idx]
                if include_pf_pct and ifr_ss is not None and ifr_cs is not None:
                    ss_seg = ifr_ss[start_idx:end_idx]
                    cs_seg = ifr_cs[start_idx:end_idx]
                    denom = ss_seg + cs_seg
                    valid = np.isfinite(denom) & (denom > 0)
                    ss_pct = np.where(valid, 100.0 * ss_seg / denom, np.nan)
                    cs_pct = np.where(valid, 100.0 * cs_seg / denom, np.nan)
                    ss_pct_stack[i, write_start:write_end] = ss_pct
                    cs_pct_stack[i, write_start:write_end] = cs_pct

            pf_theta_mean = np.nanmean(theta_stack, axis=0)
            pf_slow_mean = np.nanmean(slow_stack, axis=0)
            pf_rate_mean = np.nanmean(rate_stack, axis=0) if include_pf_rate else None
            pf_ss_rate_mean = np.nanmean(ss_rate_stack, axis=0) if include_pf_ss_rate else None
            pf_cs_rate_mean = np.nanmean(cs_rate_stack, axis=0) if include_pf_cs_rate else None
            pf_ss_pct_mean = np.nanmean(ss_pct_stack, axis=0) if include_pf_pct else None
            pf_cs_pct_mean = np.nanmean(cs_pct_stack, axis=0) if include_pf_pct else None
            count_theta = np.sum(np.isfinite(theta_stack), axis=0)
            count_slow = np.sum(np.isfinite(slow_stack), axis=0)
            std_theta = np.nanstd(theta_stack, axis=0, ddof=0)
            std_slow = np.nanstd(slow_stack, axis=0, ddof=0)
            pf_theta_sem = np.where(count_theta > 0, std_theta / np.sqrt(count_theta), np.nan)
            pf_slow_sem = np.where(count_slow > 0, std_slow / np.sqrt(count_slow), np.nan)
            if include_pf_rate and rate_stack is not None:
                count_rate = np.sum(np.isfinite(rate_stack), axis=0)
                std_rate = np.nanstd(rate_stack, axis=0, ddof=0)
                pf_rate_sem = np.where(count_rate > 0, std_rate / np.sqrt(count_rate), np.nan)
            if include_pf_ss_rate and ss_rate_stack is not None:
                count_ss_rate = np.sum(np.isfinite(ss_rate_stack), axis=0)
                std_ss_rate = np.nanstd(ss_rate_stack, axis=0, ddof=0)
                pf_ss_rate_sem = np.where(
                    count_ss_rate > 0, std_ss_rate / np.sqrt(count_ss_rate), np.nan
                )
            if include_pf_cs_rate and cs_rate_stack is not None:
                count_cs_rate = np.sum(np.isfinite(cs_rate_stack), axis=0)
                std_cs_rate = np.nanstd(cs_rate_stack, axis=0, ddof=0)
                pf_cs_rate_sem = np.where(
                    count_cs_rate > 0, std_cs_rate / np.sqrt(count_cs_rate), np.nan
                )
            if include_pf_pct and ss_pct_stack is not None and cs_pct_stack is not None:
                count_ss_pct = np.sum(np.isfinite(ss_pct_stack), axis=0)
                std_ss_pct = np.nanstd(ss_pct_stack, axis=0, ddof=0)
                pf_ss_pct_sem = np.where(
                    count_ss_pct > 0, std_ss_pct / np.sqrt(count_ss_pct), np.nan
                )
                count_cs_pct = np.sum(np.isfinite(cs_pct_stack), axis=0)
                std_cs_pct = np.nanstd(cs_pct_stack, axis=0, ddof=0)
                pf_cs_pct_sem = np.where(
                    count_cs_pct > 0, std_cs_pct / np.sqrt(count_cs_pct), np.nan
                )

            pf_rows = []
            if include_pf_rate:
                pf_rows.append("rate")
            if include_pf_ss_cs_rate:
                pf_rows.append("ss_cs_rate")
            if include_pf_pct:
                pf_rows.append("ss_cs_pct")
            pf_rows.extend(["theta", "slow"])

            row_start = n_trials + 1
            pf_axes_left = [axes[row_start + i, 0] for i in range(len(pf_rows))]
            pf_axes_right = [axes[row_start + i, 1] for i in range(len(pf_rows))]
            row_axes = {name: pf_axes_left[i] for i, name in enumerate(pf_rows)}
            # Keep default GridSpec geometry so PF averages are guaranteed to sit in rows
            # below the last trial (simple row order, no manual axis repositioning).

            for ax in pf_axes_right:
                ax.axis("off")

            if include_pf_rate and pf_rate_mean is not None:
                ax_pf_rate = row_axes["rate"]
                ax_pf_rate.plot(time_rel, pf_rate_mean, color=firing_rate_color, linewidth=0.7)
                ax_pf_rate.axvline(0, color="black", linewidth=0.5, alpha=0.5)
                ax_pf_rate.set_ylabel("FR\n(Hz)", fontsize=6, fontname="Arial")
                ax_pf_rate.spines["top"].set_visible(False)
                ax_pf_rate.spines["right"].set_visible(False)
                ax_pf_rate.tick_params(labelsize=5)
                ax_pf_rate.tick_params(labelbottom=False)

            if include_pf_ss_cs_rate:
                ax_pf_ss_cs = row_axes["ss_cs_rate"]
                if pf_ss_rate_mean is not None:
                    ax_pf_ss_cs.plot(
                        time_rel, pf_ss_rate_mean, color=simple_spike_color, linewidth=0.7
                    )
                if pf_cs_rate_mean is not None:
                    ax_pf_ss_cs.plot(
                        time_rel, pf_cs_rate_mean, color=complex_spike_color, linewidth=0.7
                    )
                ax_pf_ss_cs.axvline(0, color="black", linewidth=0.5, alpha=0.5)
                ax_pf_ss_cs.set_ylabel("SS/CS FR\n(Hz)", fontsize=6, fontname="Arial")
                ax_pf_ss_cs.spines["top"].set_visible(False)
                ax_pf_ss_cs.spines["right"].set_visible(False)
                ax_pf_ss_cs.tick_params(labelsize=5)
                ax_pf_ss_cs.tick_params(labelbottom=False)

            if include_pf_pct:
                ax_pf_pct = row_axes["ss_cs_pct"]
                if pf_ss_pct_mean is not None:
                    ax_pf_pct.plot(
                        time_rel, pf_ss_pct_mean, color=simple_spike_color, linewidth=0.7
                    )
                if pf_cs_pct_mean is not None:
                    ax_pf_pct.plot(
                        time_rel, pf_cs_pct_mean, color=complex_spike_color, linewidth=0.7
                    )
                ax_pf_pct.axvline(0, color="black", linewidth=0.5, alpha=0.5)
                ax_pf_pct.set_ylabel("SS/CS %", fontsize=6, fontname="Arial")
                ax_pf_pct.spines["top"].set_visible(False)
                ax_pf_pct.spines["right"].set_visible(False)
                ax_pf_pct.tick_params(labelsize=5)
                ax_pf_pct.tick_params(labelbottom=False)

            ax_pf_theta = row_axes["theta"]
            ax_pf_slow = row_axes["slow"]
            ax_pf_theta.plot(time_rel, pf_theta_mean, color="blue", linewidth=0.7)
            ax_pf_slow.plot(time_rel, pf_slow_mean, color="red", linewidth=0.7)
            ax_pf_theta.axvline(0, color="black", linewidth=0.5, alpha=0.5)
            ax_pf_slow.axvline(0, color="black", linewidth=0.5, alpha=0.5)
            ax_pf_theta.set_ylabel("$\\theta$ amp.\n(spk. height)", fontsize=6, fontname="Arial")
            ax_pf_slow.set_ylabel("Slow Vm\n(spk. height)", fontsize=6, fontname="Arial")
            ax_pf_slow.set_xlabel("Time from PF peak (s)", fontsize=6, fontname="Arial")
            ax_pf_theta.tick_params(labelbottom=False)
            ax_pf_theta.spines["top"].set_visible(False)
            ax_pf_theta.spines["right"].set_visible(False)
            ax_pf_slow.spines["top"].set_visible(False)
            ax_pf_slow.spines["right"].set_visible(False)
            ax_pf_theta.tick_params(labelsize=5)
            ax_pf_slow.tick_params(labelsize=5)

            for ax in pf_axes_left:
                ax.set_xlim([time_rel[0], time_rel[-1]])

    print(f"Plotted {n_trials} traversal trials")
    if return_pf_centered and return_spike_rate_means and return_pf_centered_sem:
        return (
            fig,
            axes,
            valid_epochs,
            pf_theta_mean,
            pf_slow_mean,
            pf_rate_mean,
            pf_ss_rate_mean,
            pf_cs_rate_mean,
            pf_ss_pct_mean,
            pf_cs_pct_mean,
            pf_theta_sem,
            pf_slow_sem,
            pf_rate_sem,
            pf_ss_rate_sem,
            pf_cs_rate_sem,
            pf_ss_pct_sem,
            pf_cs_pct_sem,
        )
    if return_pf_centered and return_spike_rate_means:
        return (
            fig,
            axes,
            valid_epochs,
            pf_theta_mean,
            pf_slow_mean,
            pf_rate_mean,
            pf_ss_rate_mean,
            pf_cs_rate_mean,
            pf_ss_pct_mean,
            pf_cs_pct_mean,
        )
    if return_pf_centered and return_pf_centered_sem:
        return (
            fig,
            axes,
            valid_epochs,
            pf_theta_mean,
            pf_slow_mean,
            pf_rate_mean,
            pf_theta_sem,
            pf_slow_sem,
            pf_rate_sem,
        )
    if return_pf_centered:
        return fig, axes, valid_epochs, pf_theta_mean, pf_slow_mean, pf_rate_mean
    return fig, axes, valid_epochs


def plot_place_field_traversal_trials_with_cb_example(
    trace,
    theta_vm,
    slow_vm,
    speed,
    x_neural,
    y_neural,
    traversal_epochs,
    place_field_mask,
    pf_bins,
    frame_rate,
    complex_bursts_dicts,
    cell_idx,
    burst_pre_ms=100,
    burst_post_ms=100,
    padding_sec=2.0,
    zscore_traces=False,
    bad_timepoints=None,
    show=True,
    return_pf_centered=False,
    return_spike_rate_means=False,
    **kwargs,
):
    bad_timepoints = kwargs.pop("bad_timepoints", bad_timepoints)
    want_spike_rate_means = return_pf_centered and return_spike_rate_means
    if return_pf_centered:
        if want_spike_rate_means:
            (
                fig,
                axes,
                valid_epochs,
                theta_mean,
                slow_mean,
                rate_mean,
                ss_rate_mean,
                cs_rate_mean,
                ss_pct_mean,
                cs_pct_mean,
            ) = plot_place_field_traversal_trials(
                trace,
                theta_vm,
                slow_vm,
                speed,
                x_neural,
                y_neural,
                traversal_epochs,
                place_field_mask,
                pf_bins,
                frame_rate,
                cell_idx=cell_idx,
                padding_sec=padding_sec,
                zscore_traces=zscore_traces,
                bad_timepoints=bad_timepoints,
                show=False,
                return_pf_centered=True,
                return_spike_rate_means=True,
                **kwargs,
            )
        else:
            fig, axes, valid_epochs, theta_mean, slow_mean, rate_mean = plot_place_field_traversal_trials(
                trace,
                theta_vm,
                slow_vm,
                speed,
                x_neural,
                y_neural,
                traversal_epochs,
                place_field_mask,
                pf_bins,
                frame_rate,
                cell_idx=cell_idx,
                padding_sec=padding_sec,
                zscore_traces=zscore_traces,
                bad_timepoints=bad_timepoints,
                show=False,
                return_pf_centered=True,
                **kwargs,
            )
    else:
        fig, axes, valid_epochs = plot_place_field_traversal_trials(
            trace,
            theta_vm,
            slow_vm,
            speed,
            x_neural,
            y_neural,
            traversal_epochs,
            place_field_mask,
            pf_bins,
            frame_rate,
            cell_idx=cell_idx,
            padding_sec=padding_sec,
            zscore_traces=zscore_traces,
            bad_timepoints=bad_timepoints,
            show=False,
            return_pf_centered=False,
            **kwargs,
        )
    if fig is None:
        if return_pf_centered:
            if want_spike_rate_means:
                return None, None, [], None, None, None, None, None, None, None
            return None, None, [], None, None, None
        return None, None, []

    if complex_bursts_dicts is None or cell_idx is None:
        if show:
            plt.show()
        if return_pf_centered:
            if want_spike_rate_means:
                return (
                    fig,
                    axes,
                    valid_epochs,
                    theta_mean,
                    slow_mean,
                    rate_mean,
                    ss_rate_mean,
                    cs_rate_mean,
                    ss_pct_mean,
                    cs_pct_mean,
                )
            return fig, axes, valid_epochs, theta_mean, slow_mean, rate_mean
        return fig, axes, valid_epochs

    if isinstance(complex_bursts_dicts, (list, tuple)):
        if cell_idx >= len(complex_bursts_dicts):
            if show:
                plt.show()
            if return_pf_centered:
                if want_spike_rate_means:
                    return (
                        fig,
                        axes,
                        valid_epochs,
                        theta_mean,
                        slow_mean,
                        rate_mean,
                        ss_rate_mean,
                        cs_rate_mean,
                        ss_pct_mean,
                        cs_pct_mean,
                    )
                return fig, axes, valid_epochs, theta_mean, slow_mean, rate_mean
            return fig, axes, valid_epochs
        bursts = complex_bursts_dicts[cell_idx]
    else:
        bursts = complex_bursts_dicts

    trace_arr = np.asarray(trace, dtype=float)
    trace_z = _safe_zscore(trace_arr, zscore_traces)
    starts = np.asarray(bursts.get("starts", []), dtype=int)
    ends = np.asarray(bursts.get("ends", []), dtype=int)
    amplitudes = np.asarray(bursts.get("amplitudes", []), dtype=float)

    if starts.size > 0 and ends.size > 0:
        if amplitudes.size != starts.size:
            amp_fallback = np.full(starts.shape, np.nan, dtype=float)
            for i, start_idx in enumerate(starts):
                if 0 <= start_idx < len(trace_arr):
                    if start_idx > 0:
                        base = np.nanmin(trace_arr[max(0, start_idx - 3) : start_idx])
                    else:
                        base = trace_arr[start_idx]
                    if not np.isfinite(base):
                        base = trace_arr[start_idx]
                    amp_fallback[i] = trace_arr[start_idx] - base
            amplitudes = amp_fallback

    n_trials = len(valid_epochs)
    if n_trials == 0:
        if show:
            plt.show()
        if return_pf_centered:
            if want_spike_rate_means:
                return (
                    fig,
                    axes,
                    valid_epochs,
                    theta_mean,
                    slow_mean,
                    rate_mean,
                    ss_rate_mean,
                    cs_rate_mean,
                    ss_pct_mean,
                    cs_pct_mean,
                )
            return fig, axes, valid_epochs, theta_mean, slow_mean, rate_mean
        return fig, axes, valid_epochs

    pos_trace = axes[0, 0].get_position()
    pos_traj = axes[0, 1].get_position()
    left = pos_trace.x0
    right = pos_traj.x1
    total_width = right - left
    traj_width = pos_traj.width
    if total_width > 0 and traj_width > 0:
        scale = total_width / (total_width + traj_width)
    else:
        scale = 1.0
    new_col_width = traj_width * scale

    for ax in fig.axes:
        pos = ax.get_position()
        new_x0 = left + (pos.x0 - left) * scale
        new_width = pos.width * scale
        ax.set_position([new_x0, pos.y0, new_width, pos.height])

    new_col_x0 = right - new_col_width

    padding_frames = int(padding_sec * frame_rate)
    pre_frames = int(round(burst_pre_ms / 1000 * frame_rate))
    post_frames = int(round(burst_post_ms / 1000 * frame_rate))

    def _add_time_scale_bar(ax, bar_ms, label):
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        if not (
            np.isfinite(x_min)
            and np.isfinite(x_max)
            and np.isfinite(y_min)
            and np.isfinite(y_max)
        ):
            return
        if x_max <= x_min or y_max <= y_min:
            return
        x_range = x_max - x_min
        if bar_ms >= x_range:
            return
        y_range = y_max - y_min if y_max != y_min else 1.0
        x0 = x_min + 0.05 * x_range
        x1 = x0 + bar_ms
        right_limit = x_max - 0.05 * x_range
        if x1 > right_limit:
            x1 = right_limit
            x0 = x1 - bar_ms
        y_bar = y_min + 0.05 * y_range
        ax.plot(
            [x0, x1],
            [y_bar, y_bar],
            color="black",
            linewidth=1.0,
            solid_capstyle="butt",
        )
        ax.text(
            (x0 + x1) / 2,
            y_bar - 0.05 * y_range,
            label,
            fontsize=6,
            fontname="Arial",
            ha="center",
            va="top",
        )

    for trial_idx, (epoch_start, epoch_end) in enumerate(valid_epochs):
        ax_trace = axes[trial_idx, 0]
        ax_traj = axes[trial_idx, 1]
        pos = ax_traj.get_position()
        ax_example = fig.add_axes([new_col_x0, pos.y0, new_col_width, pos.height])

        chosen_idx = None
        if starts.size > 0 and ends.size > 0:
            in_traversal = np.where((starts >= epoch_start) & (starts <= epoch_end))[0]
            if in_traversal.size > 0:
                cand_amp = amplitudes[in_traversal] if amplitudes.size == starts.size else None
                if cand_amp is not None and np.any(np.isfinite(cand_amp)):
                    chosen_idx = in_traversal[np.nanargmax(cand_amp)]
                else:
                    chosen_idx = in_traversal[0]

        if chosen_idx is None:
            ax_example.text(
                0.5,
                0.5,
                "No burst",
                transform=ax_example.transAxes,
                ha="center",
                va="center",
                fontsize=6,
                fontname="Arial",
            )
            ax_example.axis("off")
            continue

        burst_start = int(starts[chosen_idx])
        burst_end = int(ends[chosen_idx]) if chosen_idx < len(ends) else burst_start
        if burst_end < burst_start:
            burst_end = burst_start

        t_start = (burst_start - epoch_start) / frame_rate
        t_end = (burst_end - epoch_start) / frame_rate
        if t_end <= t_start:
            t_end = t_start + (1.0 / frame_rate)
        start_idx = epoch_start - padding_frames
        end_idx = epoch_end + padding_frames
        if start_idx < 0:
            start_idx = 0
        if end_idx > len(trace_z):
            end_idx = len(trace_z)
        trace_window = trace_z[start_idx:end_idx]
        trace_min = np.nanmin(trace_window) if trace_window.size > 0 else np.nan
        trace_max = np.nanmax(trace_window) if trace_window.size > 0 else np.nan
        if not np.isfinite(trace_min) or not np.isfinite(trace_max) or trace_max == trace_min:
            trace_min, trace_max = ax_trace.get_ylim()
        rect = Rectangle(
            (t_start, trace_min),
            t_end - t_start,
            trace_max - trace_min,
            fill=False,
            edgecolor="gray",
            linestyle="--",
            linewidth=0.8,
            zorder=4,
        )
        ax_trace.add_patch(rect)

        win_start = max(burst_start - pre_frames, 0)
        win_end = min(burst_end + post_frames + 1, len(trace_arr))
        if win_end <= win_start:
            ax_example.text(
                0.5,
                0.5,
                "No burst",
                transform=ax_example.transAxes,
                ha="center",
                va="center",
                fontsize=6,
                fontname="Arial",
            )
            ax_example.axis("off")
            continue

        t_ms = (np.arange(win_start, win_end) - burst_start) / frame_rate * 1000
        ax_example.plot(t_ms, trace_arr[win_start:win_end], color="black", linewidth=0.6)
        ax_example.set_xlim([t_ms[0], t_ms[-1]])
        _add_time_scale_bar(ax_example, 50.0, "50 ms")
        ax_example.set_xticks([])
        ax_example.set_yticks([])
        ax_example.spines["top"].set_visible(False)
        ax_example.spines["right"].set_visible(False)
        ax_example.spines["left"].set_visible(False)
        ax_example.spines["bottom"].set_visible(False)

    if show:
        plt.show()

    if return_pf_centered:
        if want_spike_rate_means:
            return (
                fig,
                axes,
                valid_epochs,
                theta_mean,
                slow_mean,
                rate_mean,
                ss_rate_mean,
                cs_rate_mean,
                ss_pct_mean,
                cs_pct_mean,
            )
        return fig, axes, valid_epochs, theta_mean, slow_mean, rate_mean
    return fig, axes, valid_epochs


def plot_place_field_traversal_trials_centered_by_max_rate(
    trace,
    theta_vm,
    slow_vm,
    speed,
    x_neural,
    y_neural,
    traversal_epochs,
    place_field_mask,
    pf_bins,
    frame_rate,
    cell_idx=None,
    padding_sec=2.0,
    zscore_traces=False,
    speed_smooth_kernel=51,
    resting_speed_threshold=5,
    rest_merge_gap_s=0.5,
    traversal_types=None,
    clockwise=False,
    counterclockwise=False,
    session_start_frames=None,
    color_by_session=False,
    session_indices=None,
    bad_timepoints=None,
    show_rest_patches=True,
    rest_patch_scope="prepost",
    min_rest_patch_duration_s=0.0,
    rest_patch_color="gold",
    rest_patch_alpha=0.2,
    shade_traversal_epoch=True,
    gradient_traversal_trajectory=False,
    trajectory_rasterized=False,
    trajectory_direction_label_top=False,
    trajectory_direction_label_fontsize=6,
    theta_freqs=None,
    slow_freqs=None,
    slow_trace_gap=1.0,
    theta_slow_pad_factor=0.05,
    sharey=False,
    refined_SS=None,
    all_CS_spikes=None,
    all_spikes=None,
    firing_rate_bin_ms=100,
    firing_rate_smooth_ms=20,
    firing_rate_color="#555555",
    firing_rate_alpha=0.6,
    firing_rate_linewidth=0.5,
    show_firing_rate=False,
    simple_spike_color="#026C80",
    complex_spike_color="#EE9B00",
    spike_tick_size=4,
    pf_peak_xy=None,
    pf_center_window_sec=2.0,
    plot_pf_centered_average=True,
    show=True,
    return_pf_centered=False,
    normalize_slow_vm=False,
    normalize_slow_baseline=False,
    plot_pf_centered_rate=True,
    axes=None,
    apply_layout=True,
    return_pf_centered_sem=False,
    return_spike_rate_means=False,
    center_by_pf_position=False,
    max_pf_distance_cm=None,
    **kwargs,
):
    """
    Similar to plot_place_field_traversal_trials, but centers by maximum firing rate
    within the PF traversal instead of closest position to PF peak.
    Trials without spikes in the PF traversal are excluded from the average.

    If center_by_pf_position=True, instead of centering by max firing rate, centers
    by the timepoint closest to the average PF peak position (pf_peak_xy), but still
    excludes traversals without spikes from the average calculation.

    max_pf_distance_cm : float or None
        Maximum allowed distance (cm) from PF peak at closest approach for a trial to be included.
        If None, no distance limit is applied. Only used when center_by_pf_position=True.
    sharey : bool
        If True, enforce shared y-limits across trial trace axes and keep slow/theta
        offsets consistent across trials within the figure.
    """
    slow_trace_gap = float(slow_trace_gap)
    theta_slow_pad_factor = float(theta_slow_pad_factor)
    sharey = bool(sharey)
    trace_ymax_from_simple_spikes = bool(kwargs.pop("trace_ymax_from_simple_spikes", False))
    trace_ymax_simple_spike_scale = float(kwargs.pop("trace_ymax_simple_spike_scale", 1.05))
    if slow_trace_gap < 0:
        raise ValueError("slow_trace_gap must be >= 0.")
    if theta_slow_pad_factor < 0:
        raise ValueError("theta_slow_pad_factor must be >= 0.")
    if not np.isfinite(trace_ymax_simple_spike_scale) or trace_ymax_simple_spike_scale <= 0:
        raise ValueError("trace_ymax_simple_spike_scale must be > 0.")
    place_field_contours = kwargs.pop("place_field_contours", None)
    center_position_label = kwargs.pop("center_position_label", None)
    pf_peak_label = kwargs.pop("pf_peak_label", "PF peak")
    pf_peak_color = kwargs.pop("pf_peak_color", "magenta")
    max_plot_trial_duration_s = kwargs.pop("max_plot_trial_duration_s", None)
    drop_epochs_with_bad_timepoints = bool(kwargs.pop("drop_epochs_with_bad_timepoints", True))
    trajectory_color_values_by_trial = kwargs.pop("trajectory_color_values_by_trial", None)
    trajectory_color_cmap = kwargs.pop("trajectory_color_cmap", None)
    trajectory_color_vmin = kwargs.pop("trajectory_color_vmin", None)
    trajectory_color_vmax = kwargs.pop("trajectory_color_vmax", None)
    trial_type_label_map = kwargs.pop("trial_type_label_map", None)
    trajectory_spatial_bin_hd_overlays_by_trial = kwargs.pop("trajectory_spatial_bin_hd_overlays_by_trial", None)
    trajectory_spatial_bin_hd_overlay_scale_cm = float(kwargs.pop("trajectory_spatial_bin_hd_overlay_scale_cm", 2.5))
    trajectory_spatial_bin_hd_overlay_valid_color = str(kwargs.pop("trajectory_spatial_bin_hd_overlay_valid_color", "#4D4D4D"))
    trajectory_spatial_bin_hd_overlay_green_color = str(kwargs.pop("trajectory_spatial_bin_hd_overlay_green_color", "#1A9C3D"))
    trajectory_spatial_bin_hd_overlay_alpha = float(kwargs.pop("trajectory_spatial_bin_hd_overlay_alpha", 0.9))
    pf_occupancy_mask = kwargs.pop("pf_occupancy_mask", None)
    show_pf_centered_pct = bool(kwargs.pop("show_pf_centered_pct", True))
    pf_mask_shape_ref = np.asarray(place_field_mask, dtype=bool).shape
    if pf_occupancy_mask is None:
        pf_occupancy_mask = np.asarray(place_field_mask, dtype=bool)
    else:
        pf_occupancy_mask = np.asarray(pf_occupancy_mask, dtype=bool)
        if pf_occupancy_mask.shape != pf_mask_shape_ref:
            pf_occupancy_mask = np.asarray(place_field_mask, dtype=bool)
    show_pf_prepost_patches = bool(kwargs.pop("show_pf_prepost_patches", False))
    pf_prepost_patch_alpha = float(kwargs.pop("pf_prepost_patch_alpha", 0.05))
    pf_prepost_patch_defs = kwargs.pop("pf_prepost_patch_defs", None)
    pf_prepost_patch_scope = str(kwargs.pop("pf_prepost_patch_scope", "prepost")).strip().lower()
    pf_prepost_running_only = bool(kwargs.pop("pf_prepost_running_only", True))
    rest_patch_require_in_pf = bool(kwargs.pop("rest_patch_require_in_pf", True))
    if pf_prepost_patch_alpha < 0 or pf_prepost_patch_alpha > 1:
        raise ValueError("pf_prepost_patch_alpha must be in [0, 1].")
    if pf_prepost_patch_scope not in {"prepost", "full_window"}:
        raise ValueError("pf_prepost_patch_scope must be one of: prepost, full_window.")

    pf_prepost_masks: list[tuple[np.ndarray, str]] = []
    if show_pf_prepost_patches:
        if isinstance(pf_prepost_patch_defs, (list, tuple)):
            for item in pf_prepost_patch_defs:
                if not isinstance(item, dict):
                    continue
                mask_i = np.asarray(item.get("mask", []), dtype=bool)
                if mask_i.size == 0 or mask_i.shape != pf_mask_shape_ref or not np.any(mask_i):
                    continue
                color_i = str(item.get("color", "magenta"))
                pf_prepost_masks.append((mask_i, color_i))
        if len(pf_prepost_masks) == 0:
            # Fallback to the current PF mask if explicit PF1/PF2 masks were not provided.
            base_mask = np.asarray(place_field_mask, dtype=bool)
            if base_mask.size > 0 and np.any(base_mask):
                pf_prepost_masks = [(base_mask, "magenta")]

    def _draw_pf_contour(
        ax,
        mask,
        *,
        color="magenta",
        linewidth=0.8,
        linestyle="solid",
        alpha=1.0,
    ):
        pf_mask_local = np.asarray(mask, dtype=bool)
        if pf_mask_local.size == 0 or not np.any(pf_mask_local):
            return
        if pf_mask_local.shape != pf_mask_shape_ref:
            return
        padded_mask = np.zeros((pf_mask_local.shape[0] + 2, pf_mask_local.shape[1] + 2), dtype=bool)
        padded_mask[1:-1, 1:-1] = pf_mask_local
        extent = (pf_bins[0][0], pf_bins[0][-1], pf_bins[1][0], pf_bins[1][-1])
        bin_x = (extent[1] - extent[0]) / pf_mask_local.shape[0]
        bin_y = (extent[3] - extent[2]) / pf_mask_local.shape[1]
        padded_extent = (
            extent[0] - bin_x,
            extent[1] + bin_x,
            extent[2] - bin_y,
            extent[3] + bin_y,
        )
        ax.contour(
            padded_mask.T,
            levels=[0.5],
            colors=[color],
            linewidths=float(linewidth),
            linestyles=str(linestyle),
            alpha=float(alpha),
            extent=padded_extent,
            origin="lower",
        )

    def _add_vertical_scale_bar(ax, bar_height, label, x_pos=0.92, align="left"):
        y_min, y_max = ax.get_ylim()
        if not (np.isfinite(y_min) and np.isfinite(y_max)):
            return
        if y_max <= y_min:
            return
        y_span = y_max - y_min
        y0 = y_min + 0.1 * y_span
        if y0 + bar_height > y_max:
            y0 = y_max - bar_height
        ax.plot(
            [x_pos, x_pos],
            [y0, y0 + bar_height],
            transform=ax.get_yaxis_transform(),
            color="black",
            linewidth=1.0,
            clip_on=False,
        )
        ax.text(
            x_pos + 0.01 if align == "left" else x_pos - 0.01,
            y0 + bar_height / 2.0,
            label,
            transform=ax.get_yaxis_transform(),
            ha=align,
            va="center",
            fontsize=6,
            fontname="Arial",
            color="black",
        )

    def _format_trial_type_label(raw_type):
        key = str(raw_type).strip().lower()
        if isinstance(trial_type_label_map, dict) and key in trial_type_label_map:
            return str(trial_type_label_map.get(key, ""))
        if key in {"cw", "ccw"}:
            return key.upper()
        if key in {"p", "np"}:
            return key.upper()
        return ""

    trace_z = _safe_zscore(trace, zscore_traces)
    theta_z = _safe_zscore(theta_vm, zscore_traces)
    slow_z = _safe_zscore(slow_vm, zscore_traces)
    slow_plot = slow_vm.copy() if normalize_slow_vm else slow_vm
    if normalize_slow_vm:
        mean_slow = np.nanmean(slow_vm)
        std_slow = np.nanstd(slow_vm)
        if np.isfinite(std_slow) and std_slow != 0:
            slow_plot = (slow_vm - mean_slow) / std_slow
        else:
            slow_plot = slow_vm - mean_slow

    if speed_smooth_kernel % 2 == 0:
        speed_smooth_kernel += 1
    speed_smooth = median_filter(speed, size=speed_smooth_kernel)

    padding_frames = int(padding_sec * frame_rate)

    # Initialize epoch-type pairs to maintain mapping through filtering
    if traversal_types is not None and len(traversal_types) == len(traversal_epochs):
        epochs_with_types = list(zip(traversal_epochs, traversal_types))
    else:
        # Default to "unknown" if types not provided
        epochs_with_types = [(epoch, "unknown") for epoch in traversal_epochs]

    # Filter by clockwise/counterclockwise if requested
    if clockwise ^ counterclockwise:
        if traversal_types is None or len(traversal_types) != len(traversal_epochs):
            print("Traversal types not provided or length mismatch; plotting all traversals.")
        else:
            target = "cw" if clockwise else "ccw"
            epochs_with_types = [
                (epoch, ttype)
                for epoch, ttype in epochs_with_types
                if ttype == target
            ]

    # Filter by session indices
    if session_indices is not None:
        if session_start_frames is None:
            print("session_start_frames not provided; plotting all sessions.")
        else:
            session_start_frames = np.asarray(session_start_frames, dtype=int)
            session_start_frames = session_start_frames[np.argsort(session_start_frames)]
            if np.isscalar(session_indices):
                session_indices = [int(session_indices)]
            session_indices = [int(i) for i in session_indices]
            epochs_filtered = []
            for (epoch_start, epoch_end), ttype in epochs_with_types:
                session_idx = np.searchsorted(session_start_frames, epoch_start, side="right") - 1
                if session_idx in session_indices:
                    epochs_filtered.append(((epoch_start, epoch_end), ttype))
            epochs_with_types = epochs_filtered

    bad_mask = None
    if bad_timepoints is not None:
        bad_mask = np.asarray(bad_timepoints)
        if bad_mask.dtype != bool:
            if bad_mask.ndim == 1 and bad_mask.size == len(trace_z):
                bad_mask = bad_mask.astype(bool)
            else:
                bad_mask = np.asarray(bad_mask, dtype=int)
                bad_mask = bad_mask[(bad_mask >= 0) & (bad_mask < len(trace_z))]
                bad_bool = np.zeros(len(trace_z), dtype=bool)
                bad_bool[bad_mask] = True
                bad_mask = bad_bool
        if bad_mask.shape[0] != len(trace_z):
            raise ValueError("bad_timepoints must match trace length or be index list.")

    # Filter by bad timepoints and extract valid epochs with their types
    valid_epochs = []
    valid_traversal_types = []
    for (epoch_start, epoch_end), ttype in epochs_with_types:
        start_idx = epoch_start - padding_frames
        end_idx = epoch_end + padding_frames
        if start_idx < 0 or end_idx > len(trace_z):
            continue
        if (
            bool(drop_epochs_with_bad_timepoints)
            and bad_mask is not None
            and np.any(bad_mask[start_idx:end_idx])
        ):
            continue
        valid_epochs.append((epoch_start, epoch_end))
        valid_traversal_types.append(ttype)

    n_trials = len(valid_epochs)
    if n_trials == 0:
        print("No valid traversal epochs with sufficient padding")
        if return_pf_centered and return_spike_rate_means and return_pf_centered_sem:
            return (
                None,
                None,
                [],
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )
        if return_pf_centered and return_spike_rate_means:
            return None, None, [], None, None, None, None, None, None, None
        if return_pf_centered and return_pf_centered_sem:
            return None, None, [], None, None, None, None, None, None
        if return_pf_centered:
            return None, None, [], None, None, None
        return None, None, []

    trace_mins, trace_maxs = [], []
    theta_mins, theta_maxs = [], []
    slow_mins, slow_maxs = [], []
    for epoch_start, epoch_end in valid_epochs:
        start_idx = epoch_start - padding_frames
        end_idx = epoch_end + padding_frames
        trace_mins.append(np.nanmin(trace_z[start_idx:end_idx]))
        trace_maxs.append(np.nanmax(trace_z[start_idx:end_idx]))
        theta_mins.append(np.nanmin(theta_z[start_idx:end_idx]))
        theta_maxs.append(np.nanmax(theta_z[start_idx:end_idx]))
        slow_mins.append(np.nanmin(slow_z[start_idx:end_idx]))
        slow_maxs.append(np.nanmax(slow_z[start_idx:end_idx]))

    trace_min_global = min(trace_mins)
    trace_max_global = max(trace_maxs)
    theta_max_global = max(theta_maxs)
    theta_min_global = min(theta_mins)
    slow_max_global = max(slow_maxs)
    slow_min_global = min(slow_mins)

    trace_range = trace_max_global - trace_min_global
    theta_range = theta_max_global - theta_min_global
    slow_range = slow_max_global - slow_min_global

    trace_plot_trial_mask = np.ones(n_trials, dtype=bool)
    fixed_trace_xlim_s = None
    fixed_trace_xlim_frames = None
    if max_plot_trial_duration_s is not None:
        max_plot_trial_duration_s = float(max_plot_trial_duration_s)
        if np.isfinite(max_plot_trial_duration_s) and max_plot_trial_duration_s > 0:
            fixed_trace_xlim_s = float(max_plot_trial_duration_s)
            fixed_trace_xlim_frames = max(1, int(round(fixed_trace_xlim_s * float(frame_rate))))
    max_epoch_duration = max((e - s) for s, e in valid_epochs)
    global_x_min = -padding_sec
    global_x_max = (
        fixed_trace_xlim_s if fixed_trace_xlim_s is not None else (max_epoch_duration / frame_rate + padding_sec)
    )

    def _get_trace_plot_indices(epoch_start, epoch_end):
        start_idx = int(max(0, epoch_start - padding_frames))
        end_idx = int(min(len(trace_z), epoch_end + padding_frames))
        visible_end_idx = end_idx
        if fixed_trace_xlim_frames is not None:
            visible_end_idx = min(visible_end_idx, int(epoch_start + fixed_trace_xlim_frames))
        if visible_end_idx <= start_idx:
            visible_end_idx = min(end_idx, start_idx + 1)
        return start_idx, end_idx, visible_end_idx

    def _get_spike_indices(spike_source):
        if spike_source is None:
            return None
        if (
            cell_idx is not None
            and isinstance(spike_source, (list, tuple))
            and len(spike_source) > 0
            and isinstance(spike_source[0], (list, np.ndarray))
        ):
            return np.asarray(spike_source[cell_idx], dtype=int)
        return np.asarray(spike_source, dtype=int)

    simple_spikes = _get_spike_indices(refined_SS)
    complex_spikes = _get_spike_indices(all_CS_spikes)
    all_spike_indices = _get_spike_indices(all_spikes)

    simple_spike_trace_ymax_global = None
    simple_spike_theta_ymax_global = None
    if trace_ymax_from_simple_spikes and simple_spikes is not None and len(simple_spikes) > 0:
        ss_vals_all = []
        theta_ss_vals_all = []
        for keep_i, (epoch_start, epoch_end) in zip(trace_plot_trial_mask.tolist(), valid_epochs):
            if not keep_i:
                continue
            start_idx, _, visible_end_idx = _get_trace_plot_indices(epoch_start, epoch_end)
            ss_idx = np.asarray(simple_spikes, dtype=int)
            ss_idx = ss_idx[(ss_idx >= start_idx) & (ss_idx < visible_end_idx)]
            if bad_mask is not None and ss_idx.size > 0:
                ss_idx = ss_idx[~bad_mask[ss_idx]]
            if ss_idx.size == 0:
                continue
            ss_vals = np.asarray(trace_z[ss_idx], dtype=float)
            ss_vals = ss_vals[np.isfinite(ss_vals)]
            if ss_vals.size > 0:
                ss_vals_all.append(ss_vals)
            theta_ss_vals = np.asarray(theta_z[ss_idx], dtype=float)
            theta_ss_vals = theta_ss_vals[np.isfinite(theta_ss_vals)]
            if theta_ss_vals.size > 0:
                theta_ss_vals_all.append(theta_ss_vals)
        if len(ss_vals_all) > 0:
            ss_max_global = float(np.nanmax(np.concatenate(ss_vals_all)))
            candidate_ymax = float(ss_max_global * trace_ymax_simple_spike_scale)
            if np.isfinite(candidate_ymax):
                simple_spike_trace_ymax_global = candidate_ymax
        if len(theta_ss_vals_all) > 0:
            theta_ss_max_global = float(np.nanmax(np.concatenate(theta_ss_vals_all)))
            candidate_theta_ymax = float(theta_ss_max_global * trace_ymax_simple_spike_scale)
            if np.isfinite(candidate_theta_ymax):
                simple_spike_theta_ymax_global = candidate_theta_ymax

    trace_display_max_global = trace_max_global
    if (
        trace_ymax_from_simple_spikes
        and simple_spike_trace_ymax_global is not None
        and np.isfinite(simple_spike_trace_ymax_global)
        and simple_spike_trace_ymax_global > trace_min_global
    ):
        trace_display_max_global = min(trace_display_max_global, float(simple_spike_trace_ymax_global))

    theta_display_max_global = theta_max_global
    if (
        trace_ymax_from_simple_spikes
        and simple_spike_theta_ymax_global is not None
        and np.isfinite(simple_spike_theta_ymax_global)
        and simple_spike_theta_ymax_global > theta_min_global
    ):
        theta_display_max_global = min(theta_display_max_global, float(simple_spike_theta_ymax_global))

    theta_display_range = theta_display_max_global - theta_min_global

    # Spacing between stacked signals within the same axis.
    # Show slow Vm right below the trace, and theta below slow.
    pad_slow_theta = theta_slow_pad_factor * max(
        slow_range if np.isfinite(slow_range) and slow_range > 0 else 1.0,
        theta_display_range if np.isfinite(theta_display_range) and theta_display_range > 0 else 1.0,
    )
    offset1_shared = slow_trace_gap
    offset2_shared = max(0.0, float(theta_display_max_global - slow_min_global + pad_slow_theta))

    def _compute_channel_display_stats(signal_window, display_max_ceiling=None):
        signal_window_arr = np.asarray(signal_window, dtype=float)
        finite_signal = signal_window_arr[np.isfinite(signal_window_arr)]
        if finite_signal.size > 0:
            signal_min_local = float(np.nanmin(finite_signal))
            signal_max_local = float(np.nanmax(finite_signal))
        else:
            signal_min_local = 0.0
            signal_max_local = 0.0

        signal_display_max = signal_max_local
        if (
            display_max_ceiling is not None
            and np.isfinite(display_max_ceiling)
            and display_max_ceiling > signal_min_local
        ):
            signal_display_max = min(signal_display_max, float(display_max_ceiling))

        if not np.isfinite(signal_display_max):
            signal_display_max = signal_max_local
        if signal_display_max < signal_min_local:
            signal_display_max = signal_max_local

        if np.isfinite(signal_display_max) and signal_display_max < signal_max_local:
            signal_window_ylim = np.minimum(signal_window_arr, signal_display_max)
        else:
            signal_window_ylim = signal_window_arr

        return {
            "signal_min": signal_min_local,
            "signal_display_max": signal_display_max,
            "signal_window_ylim": signal_window_ylim,
        }

    def _compute_trace_display_stats(trace_window):
        trace_display_stats = _compute_channel_display_stats(
            trace_window,
            (
                simple_spike_trace_ymax_global
                if trace_ymax_from_simple_spikes
                else None
            ),
        )
        trace_min_local = float(trace_display_stats["signal_min"])
        trace_display_max = float(trace_display_stats["signal_display_max"])
        trace_window_ylim = np.asarray(trace_display_stats["signal_window_ylim"], dtype=float)

        trace_range_local = trace_display_max - trace_min_local
        if not np.isfinite(trace_range_local) or trace_range_local <= 0:
            trace_range_local = 0.1
        spike_tick_y_local = trace_display_max + 0.05 * trace_range_local
        return {
            "trace_min": trace_min_local,
            "trace_display_max": trace_display_max,
            "trace_window_ylim": trace_window_ylim,
            "spike_tick_y": float(spike_tick_y_local),
        }

    def _compute_ifr(spike_indices):
        if spike_indices is None or len(spike_indices) == 0:
            return None
        spike_indices = spike_indices[(spike_indices >= 0) & (spike_indices < len(trace_z))]
        if spike_indices.size == 0:
            return None
        spike_train = np.zeros(len(trace_z), dtype=float)
        spike_train[spike_indices] = 1.0
        bin_frames = max(1, int(round(firing_rate_bin_ms / 1000 * frame_rate)))
        bin_kernel = np.ones(bin_frames, dtype=float) / bin_frames
        ifr_local = np.convolve(spike_train, bin_kernel, mode="same") * frame_rate
        smooth_frames = max(1, int(round(firing_rate_smooth_ms / 1000 * frame_rate)))
        if smooth_frames > 1:
            smooth_kernel = np.ones(smooth_frames, dtype=float) / smooth_frames
            ifr_local = np.convolve(ifr_local, smooth_kernel, mode="same")
        return ifr_local

    ifr = None
    if (
        (show_firing_rate or (plot_pf_centered_average and plot_pf_centered_rate))
        and all_spike_indices is not None
        and len(all_spike_indices) > 0
    ):
        ifr = _compute_ifr(all_spike_indices)
    ifr_valid = ifr.copy() if ifr is not None else None
    if bad_mask is not None and ifr_valid is not None:
        ifr_valid[bad_mask] = np.nan

    ifr_ss = None
    ifr_cs = None
    if (
        return_spike_rate_means
        and plot_pf_centered_average
        and plot_pf_centered_rate
        and pf_peak_xy is not None
    ):
        ifr_ss = _compute_ifr(simple_spikes)
        ifr_cs = _compute_ifr(complex_spikes)
    ifr_ss_valid = ifr_ss.copy() if ifr_ss is not None else None
    ifr_cs_valid = ifr_cs.copy() if ifr_cs is not None else None
    if bad_mask is not None:
        if ifr_ss_valid is not None:
            ifr_ss_valid[bad_mask] = np.nan
        if ifr_cs_valid is not None:
            ifr_cs_valid[bad_mask] = np.nan

    include_pf_rate = (
        plot_pf_centered_average and pf_peak_xy is not None and plot_pf_centered_rate and ifr is not None
    )
    include_pf_ss_rate = (
        return_spike_rate_means
        and plot_pf_centered_average
        and pf_peak_xy is not None
        and plot_pf_centered_rate
        and ifr_ss is not None
    )
    include_pf_cs_rate = (
        return_spike_rate_means
        and plot_pf_centered_average
        and pf_peak_xy is not None
        and plot_pf_centered_rate
        and ifr_cs is not None
    )
    include_pf_ss_cs_rate = include_pf_ss_rate or include_pf_cs_rate
    include_pf_pct = (
        bool(show_pf_centered_pct)
        and
        return_spike_rate_means
        and plot_pf_centered_average
        and pf_peak_xy is not None
        and plot_pf_centered_rate
        and ifr_ss is not None
        and ifr_cs is not None
    )
    pf_rate_rows = int(include_pf_rate) + int(include_pf_ss_cs_rate) + int(include_pf_pct)
    include_pf_panel = plot_pf_centered_average and pf_peak_xy is not None
    extra_rows = (pf_rate_rows + 2) if include_pf_panel else 0
    pf_heights = ([1.1] * pf_rate_rows + [1.4, 1.4]) if include_pf_panel else []
    scale_row_height = 1.0
    height_ratios = [1] * n_trials + [scale_row_height] + (pf_heights if extra_rows else [])
    if axes is None:
        fig, axes = plt.subplots(
            n_trials + 1 + extra_rows,
            2,
            figsize=(6, 0.6 * n_trials + 0.3 + (1.4 * extra_rows)),
            gridspec_kw={
                "width_ratios": [5, 1],
                "hspace": 0,
                "wspace": 0.01,
                "height_ratios": height_ratios,
            },
        )
    else:
        axes = np.asarray(axes)
        fig = axes[0, 0].figure

    merge_gap_frames = int(rest_merge_gap_s * frame_rate)
    session_colors = None
    epoch_sessions = None
    if color_by_session and session_start_frames is not None:
        session_start_frames = np.asarray(session_start_frames, dtype=int)
        session_start_frames = session_start_frames[np.argsort(session_start_frames)]
        num_sessions = len(session_start_frames)
        if num_sessions > 0:
            base_colors = [
                "#000000",
                "#e41a1c",
                "#377eb8",
                "#4daf4a",
                "#984ea3",
                "#ff7f00",
                "#ffff33",
                "#a65628",
                "#f781bf",
                "#999999",
            ]
            if num_sessions <= len(base_colors):
                session_colors = base_colors[:num_sessions]
            else:
                extra = plt.cm.hsv(
                    np.linspace(0, 1, num_sessions - len(base_colors), endpoint=False)
                )
                session_colors = base_colors + [tuple(c) for c in extra]
            epoch_sessions = [
                np.searchsorted(session_start_frames, s, side="right") - 1
                for s, _ in valid_epochs
            ]

    # Compute center indices for each trial
    # If center_by_pf_position=True, use closest position to PF peak (works even without spikes)
    # Otherwise, use max firing rate position (requires spikes)
    max_rate_center_indices = []
    trials_with_spikes = []  # Track which trials have spikes within PF traversal
    for trial_idx, (epoch_start, epoch_end) in enumerate(valid_epochs):
        center_idx = None
        has_spikes = False

        # Check if there are any spikes within the traversal epoch
        if all_spike_indices is not None:
            spikes_in_epoch = (all_spike_indices >= epoch_start) & (all_spike_indices < epoch_end)
            has_spikes = np.any(spikes_in_epoch)

        if center_by_pf_position and pf_peak_xy is not None and np.all(np.isfinite(pf_peak_xy)):
            # Center by closest position to PF peak (works even without spikes)
            x_epoch = x_neural[epoch_start:epoch_end]
            y_epoch = y_neural[epoch_start:epoch_end]
            valid_epoch = (~np.isnan(x_epoch)) & (~np.isnan(y_epoch))
            if bad_mask is not None:
                valid_epoch = valid_epoch & (~bad_mask[epoch_start:epoch_end])
            if np.any(valid_epoch):
                dx = x_epoch[valid_epoch] - pf_peak_xy[0]
                dy = y_epoch[valid_epoch] - pf_peak_xy[1]
                dist = np.sqrt(dx**2 + dy**2)
                min_dist = np.min(dist)
                # Check if closest point is within allowed distance
                if max_pf_distance_cm is not None and min_dist > max_pf_distance_cm:
                    center_idx = None  # Exclude this trial
                else:
                    closest_idx = np.argmin(dist)
                    valid_indices = np.where(valid_epoch)[0]
                    center_idx = epoch_start + valid_indices[closest_idx]
        elif has_spikes and ifr_valid is not None:
            # Center by max firing rate within the traversal epoch (requires spikes)
            ifr_segment = ifr_valid[epoch_start:epoch_end]
            if len(ifr_segment) > 0 and np.any(np.isfinite(ifr_segment)):
                max_rel_idx = np.nanargmax(ifr_segment)
                center_idx = epoch_start + max_rel_idx

        max_rate_center_indices.append(center_idx)
        trials_with_spikes.append(has_spikes)

    shared_trace_ylim = None
    if sharey and n_trials > 0:
        ymins = []
        ymaxs = []
        for keep_i, (epoch_start, epoch_end) in zip(trace_plot_trial_mask.tolist(), valid_epochs):
            if not keep_i:
                continue
            start_idx, _, visible_end_idx = _get_trace_plot_indices(epoch_start, epoch_end)
            trace_window = trace_z[start_idx:visible_end_idx]
            trace_display_stats = _compute_trace_display_stats(trace_window)
            trace_window_ylim = np.asarray(trace_display_stats["trace_window_ylim"], dtype=float)
            theta_window = theta_z[start_idx:visible_end_idx]
            theta_display_stats = _compute_channel_display_stats(
                theta_window,
                (
                    simple_spike_theta_ymax_global
                    if trace_ymax_from_simple_spikes
                    else None
                ),
            )
            theta_window_ylim = np.asarray(theta_display_stats["signal_window_ylim"], dtype=float)
            slow_source = slow_z if zscore_traces else slow_plot
            slow_window = slow_source[start_idx:visible_end_idx]
            if normalize_slow_baseline:
                baseline = np.nan
                if epoch_start > start_idx:
                    baseline = np.nanmean(slow_source[start_idx:epoch_start])
                if np.isfinite(baseline):
                    slow_window = slow_window - baseline

            spike_tick_y = float(trace_display_stats["spike_tick_y"])

            trial_vals = np.concatenate(
                [
                    np.ravel(trace_window_ylim),
                    np.ravel(slow_window - offset1_shared),
                    np.ravel(theta_window_ylim - offset1_shared - offset2_shared),
                    np.asarray([spike_tick_y, 0.0, -offset1_shared, -offset1_shared - offset2_shared], dtype=float),
                ]
            )
            trial_vals = trial_vals[np.isfinite(trial_vals)]
            if trial_vals.size == 0:
                continue
            ymins.append(float(np.nanmin(trial_vals)))
            ymaxs.append(float(np.nanmax(trial_vals)))
        if len(ymins) > 0 and len(ymaxs) > 0:
            y0 = float(np.min(ymins))
            y1 = float(np.max(ymaxs))
            span = y1 - y0
            pad = 0.03 * span if np.isfinite(span) and span > 0 else 0.1
            shared_trace_ylim = (y0 - pad, y1 + pad)

    for trial_idx, (epoch_start, epoch_end) in enumerate(valid_epochs):
        ax_trace = axes[trial_idx, 0]
        ax_traj = axes[trial_idx, 1]

        start_idx, end_idx, visible_end_idx = _get_trace_plot_indices(epoch_start, epoch_end)
        time_rel = (np.arange(start_idx, end_idx) - epoch_start) / frame_rate
        plot_trace_trial = bool(trace_plot_trial_mask[trial_idx]) if trial_idx < len(trace_plot_trial_mask) else True

        trace_window = trace_z[start_idx:end_idx]
        theta_window = theta_z[start_idx:end_idx]
        trace_window_visible = trace_z[start_idx:visible_end_idx]
        theta_window_visible = theta_z[start_idx:visible_end_idx]
        trace_display_stats = _compute_trace_display_stats(trace_window_visible)
        trace_window_ylim = np.asarray(trace_display_stats["trace_window_ylim"], dtype=float)
        theta_display_stats = _compute_channel_display_stats(
            theta_window_visible,
            (
                simple_spike_theta_ymax_global
                if trace_ymax_from_simple_spikes
                else None
            ),
        )
        theta_window_ylim = np.asarray(theta_display_stats["signal_window_ylim"], dtype=float)
        slow_source = slow_z if zscore_traces else slow_plot
        slow_window = slow_source[start_idx:end_idx]
        slow_window_visible = slow_window[: max(0, visible_end_idx - start_idx)]
        if normalize_slow_baseline:
            baseline = np.nan
            if epoch_start > start_idx:
                baseline = np.nanmean(slow_source[start_idx:epoch_start])
            if np.isfinite(baseline):
                slow_window = slow_window - baseline
                slow_window_visible = slow_window_visible - baseline

        epoch_duration = (epoch_end - epoch_start) / frame_rate
        if plot_trace_trial and shade_traversal_epoch:
            ax_trace.axvspan(0, epoch_duration, alpha=0.2, color="green", zorder=0)

        if plot_trace_trial and ifr_valid is not None and show_firing_rate:
            ifr_window = ifr_valid[start_idx:end_idx]
            trace_min = float(trace_display_stats["trace_min"])
            trace_max = float(trace_display_stats["trace_display_max"])
            trace_range = trace_max - trace_min
            if not np.isfinite(trace_range) or trace_range == 0:
                trace_range = 1.0
            ifr_min = np.nanmin(ifr_window)
            ifr_max = np.nanmax(ifr_window)
            ifr_range = ifr_max - ifr_min
            if not np.isfinite(ifr_range) or ifr_range == 0:
                ifr_range = 1.0
            ifr_norm = (ifr_window - ifr_min) / ifr_range
            ifr_scaled = trace_min + ifr_norm * trace_range
            ax_trace.plot(
                time_rel,
                ifr_scaled,
                color=firing_rate_color,
                linewidth=firing_rate_linewidth,
                alpha=firing_rate_alpha,
                zorder=1,
            )

        spike_tick_y = float(trace_display_stats["spike_tick_y"])

        if plot_trace_trial and simple_spikes is not None:
            ss_mask = (simple_spikes >= start_idx) & (simple_spikes < end_idx)
            ss_times = (simple_spikes[ss_mask] - epoch_start) / frame_rate
            if ss_times.size > 0:
                ax_trace.plot(
                    ss_times,
                    np.full_like(ss_times, spike_tick_y, dtype=float),
                    linestyle="None",
                    marker="|",
                    markersize=spike_tick_size,
                    markeredgewidth=0.5,
                    color=simple_spike_color,
                    zorder=3,
                )
        if plot_trace_trial and complex_spikes is not None:
            cs_mask = (complex_spikes >= start_idx) & (complex_spikes < end_idx)
            cs_times = (complex_spikes[cs_mask] - epoch_start) / frame_rate
            if cs_times.size > 0:
                ax_trace.plot(
                    cs_times,
                    np.full_like(cs_times, spike_tick_y, dtype=float),
                    linestyle="None",
                    marker="|",
                    markersize=spike_tick_size,
                    markeredgewidth=0.5,
                    color=complex_spike_color,
                    zorder=3,
                )

        x_traj_full = x_neural[start_idx:end_idx]
        y_traj_full = y_neural[start_idx:end_idx]
        in_pf_window = _positions_in_place_field(
            x_traj_full, y_traj_full, pf_bins, place_field_mask
        )
        speed_window_raw = speed[start_idx:end_idx]
        resting_mask = np.isfinite(speed_window_raw) & (speed_window_raw < resting_speed_threshold)
        if bool(rest_patch_require_in_pf):
            resting_mask = resting_mask & in_pf_window

        pre_frames = padding_frames
        post_start_idx = epoch_end - start_idx

        def _shade_mask_segments(mask_local: np.ndarray, color_local: str, *, offset: int = 0) -> None:
            if not np.any(mask_local):
                return
            diff = np.diff(np.concatenate([[0], mask_local.astype(int), [0]]))
            seg_starts = np.where(diff == 1)[0]
            seg_ends = np.where(diff == -1)[0]
            seg_starts, seg_ends = _merge_segments(seg_starts, seg_ends, merge_gap_frames)
            for s, e in zip(seg_starts, seg_ends):
                s_abs = int(offset + s)
                e_abs = int(min(offset + e, len(time_rel) - 1))
                if e_abs <= s_abs:
                    continue
                t_start = time_rel[s_abs]
                t_end = time_rel[e_abs]
                ax_trace.axvspan(
                    t_start,
                    t_end,
                    alpha=pf_prepost_patch_alpha,
                    color=color_local,
                    zorder=0,
                )

        if plot_trace_trial and show_pf_prepost_patches and len(pf_prepost_masks) > 0:
            for pf_mask_i, pf_color_i in pf_prepost_masks:
                in_pf_i = _positions_in_place_field(x_traj_full, y_traj_full, pf_bins, pf_mask_i)
                if bool(pf_prepost_running_only):
                    traversing_i = in_pf_i & np.isfinite(speed_window_raw) & (
                        speed_window_raw >= resting_speed_threshold
                    )
                else:
                    traversing_i = in_pf_i.astype(bool)
                if str(pf_prepost_patch_scope) == "full_window":
                    _shade_mask_segments(traversing_i, pf_color_i, offset=0)
                else:
                    pre_trav = traversing_i[:pre_frames]
                    _shade_mask_segments(pre_trav, pf_color_i, offset=0)
                    post_trav = traversing_i[post_start_idx:]
                    _shade_mask_segments(post_trav, pf_color_i, offset=post_start_idx)

        if plot_trace_trial and show_rest_patches:
            min_rest_frames = max(1, int(round(float(min_rest_patch_duration_s) * float(frame_rate))))

            def _shade_rest_segments(mask_local: np.ndarray, offset: int = 0) -> None:
                if not np.any(mask_local):
                    return
                diff = np.diff(np.concatenate([[0], mask_local.astype(int), [0]]))
                starts = np.where(diff == 1)[0]
                ends = np.where(diff == -1)[0]
                starts, ends = _merge_segments(starts, ends, merge_gap_frames)
                for s, e in zip(starts, ends):
                    dur_frames = int(e - s)
                    if dur_frames < min_rest_frames:
                        continue
                    s_abs = int(offset + s)
                    e_abs = int(min(offset + e, len(time_rel) - 1))
                    if e_abs <= s_abs:
                        continue
                    t_start = time_rel[s_abs]
                    t_end = time_rel[e_abs]
                    ax_trace.axvspan(
                        t_start,
                        t_end,
                        alpha=rest_patch_alpha,
                        color=rest_patch_color,
                        zorder=0,
                    )

            if str(rest_patch_scope).strip().lower() == "full_window":
                _shade_rest_segments(resting_mask, 0)
            else:
                _shade_rest_segments(resting_mask[:pre_frames], 0)
                _shade_rest_segments(resting_mask[post_start_idx:], post_start_idx)

        if plot_trace_trial:
            ax_trace.plot(
                time_rel,
                trace_window,
                color="black",
                linewidth=0.5,
                label="Trace",
                zorder=2,
            )
        if sharey:
            offset1 = offset1_shared
            offset2 = offset2_shared
        else:
            slow_min_local = np.nanmin(slow_window_visible) if np.any(np.isfinite(slow_window_visible)) else 0.0
            theta_max_local = float(theta_display_stats["signal_display_max"])
            offset1 = slow_trace_gap
            offset2 = max(0.0, float(theta_max_local - slow_min_local + pad_slow_theta))
        theta_label = "Theta (4-8 Hz)"
        if theta_freqs is not None and len(theta_freqs) == 2:
            theta_label = f"Theta ({theta_freqs[0]}-{theta_freqs[1]} Hz)"
        slow_label = "Slow (<=2 Hz)"
        if slow_freqs is not None:
            if np.isscalar(slow_freqs):
                slow_label = f"Slow (<= {slow_freqs} Hz)"
            elif len(slow_freqs) == 2:
                slow_label = f"Slow ({slow_freqs[0]}-{slow_freqs[1]} Hz)"
        if plot_trace_trial:
            ax_trace.plot(
                time_rel,
                slow_window - offset1,
                color="red",
                linewidth=0.5,
                label=slow_label,
                zorder=2,
            )
            ax_trace.plot(
                time_rel,
                theta_window - offset1 - offset2,
                color="blue",
                linewidth=0.5,
                label=theta_label,
                zorder=2,
            )

        if plot_trace_trial:
            x_data_max = min(time_rel[-1], global_x_max)
            tick_len = 0.08
            ax_trace.plot(
                [x_data_max - tick_len, x_data_max],
                [0, 0],
                color="black",
                linewidth=0.5,
                alpha=0.5,
            )
            ax_trace.plot(
                [x_data_max - tick_len, x_data_max],
                [-offset1, -offset1],
                color="red",
                linewidth=0.5,
                alpha=0.5,
            )
            ax_trace.plot(
                [x_data_max - tick_len, x_data_max],
                [-offset1 - offset2, -offset1 - offset2],
                color="blue",
                linewidth=0.5,
                alpha=0.5,
            )

        trial_color = "black"
        if session_colors is not None and epoch_sessions is not None:
            session_idx = epoch_sessions[trial_idx]
            if 0 <= session_idx < len(session_colors):
                trial_color = session_colors[session_idx]
        # Build trial label with direction type
        trial_type = valid_traversal_types[trial_idx] if trial_idx < len(valid_traversal_types) else "?"
        type_label = _format_trial_type_label(trial_type)
        trial_label = f"{trial_idx + 1}" + (
            f" {type_label}" if type_label and not bool(trajectory_direction_label_top) else ""
        )
        ax_trace.text(
            time_rel[0] - 0.2,
            trace_window[0] + 0.1,
            trial_label,
            fontsize=6,
            ha="left",
            va="bottom",
            fontweight="bold",
            color=trial_color,
        )
        ax_trace.text(
            -0.6,
            0.3,
            f"{epoch_start / frame_rate:.1f}s",
            transform=ax_trace.get_xaxis_transform(),
            ha="left",
            va="bottom",
            fontsize=6,
            fontname="Arial",
            color="black",
        )

        ax_trace.spines["top"].set_visible(False)
        ax_trace.spines["right"].set_visible(False)
        ax_trace.spines["left"].set_visible(False)
        ax_trace.spines["bottom"].set_visible(False)
        ax_trace.set_yticks([])
        ax_trace.set_xticks([])
        ax_trace.set_xlim([global_x_min, global_x_max])
        if sharey and shared_trace_ylim is not None:
            ax_trace.set_ylim(shared_trace_ylim[0], shared_trace_ylim[1])
        elif trace_ymax_from_simple_spikes:
            trial_vals = np.concatenate(
                [
                    np.ravel(trace_window_ylim),
                    np.ravel(slow_window_visible - offset1),
                    np.ravel(theta_window_ylim - offset1 - offset2),
                    np.asarray([spike_tick_y, 0.0, -offset1, -offset1 - offset2], dtype=float),
                ]
            )
            trial_vals = trial_vals[np.isfinite(trial_vals)]
            if trial_vals.size > 0:
                y0 = float(np.nanmin(trial_vals))
                y1 = float(np.nanmax(trial_vals))
                span = y1 - y0
                pad = 0.03 * span if np.isfinite(span) and span > 0 else 0.1
                ax_trace.set_ylim(y0 - pad, y1 + pad)

        if trial_idx == 0:
            handles, labels = ax_trace.get_legend_handles_labels()
            if refined_SS is not None:
                handles.append(
                    Line2D(
                        [0],
                        [0],
                        marker="|",
                        color=simple_spike_color,
                        linestyle="None",
                        markersize=spike_tick_size,
                        markeredgewidth=0.5,
                        label="Simple spikes",
                    )
                )
                labels.append("Simple spikes")
            if all_CS_spikes is not None:
                handles.append(
                    Line2D(
                        [0],
                        [0],
                        marker="|",
                        color=complex_spike_color,
                        linestyle="None",
                        markersize=spike_tick_size,
                        markeredgewidth=0.5,
                        label="Complex spikes",
                    )
                )
                labels.append("Complex spikes")
            handle_map = dict(zip(labels, handles))
            ordered = []
            for label in ["Trace", slow_label, theta_label]:
                if label in handle_map:
                    ordered.append((label, handle_map[label]))
            if refined_SS is not None:
                ordered.append(
                    (
                        "Simple spikes",
                        Line2D(
                            [0],
                            [0],
                            marker="|",
                            color=simple_spike_color,
                            linestyle="None",
                            markersize=spike_tick_size,
                            markeredgewidth=0.5,
                        ),
                    )
                )
            if all_CS_spikes is not None:
                ordered.append(
                    (
                        "Complex spikes",
                        Line2D(
                            [0],
                            [0],
                            marker="|",
                            color=complex_spike_color,
                            linestyle="None",
                            markersize=spike_tick_size,
                            markeredgewidth=0.5,
                        ),
                    )
                )
            legend_labels = [label for label, _ in ordered]
            legend_handles = [handle for _, handle in ordered]
            legend = ax_trace.legend(
                legend_handles,
                legend_labels,
                loc="lower left",
                bbox_to_anchor=(0.0, 1.02),
                fontsize=5,
                frameon=False,
                ncol=3,
                borderaxespad=0,
            )
            if cell_idx is not None:
                direction_label = "all"
                if clockwise ^ counterclockwise:
                    direction_label = "cw" if clockwise else "ccw"
                title_y = 1.08
                try:
                    fig = ax_trace.figure
                    fig.canvas.draw()
                    legend_bbox = legend.get_window_extent(fig.canvas.get_renderer())
                    legend_bbox_axes = legend_bbox.transformed(ax_trace.transAxes.inverted())
                    title_y = legend_bbox_axes.y1 + 0.05
                except Exception:
                    title_y = 1.08
                center_method = "PF Position" if center_by_pf_position else "Max Rate"
                ax_trace.set_title(
                    f"Cell {cell_idx} - Place Field Traversals ({direction_label}, n={n_trials}) [{center_method} Centered]",
                    fontsize=6,
                    pad=0,
                    y=title_y,
                )
        x_traj = x_neural[start_idx:end_idx]
        y_traj = y_neural[start_idx:end_idx]
        post_start = epoch_end - start_idx

        ax_traj.plot(
            x_traj[:pre_frames],
            y_traj[:pre_frames],
            color="#87CEEB",
            linewidth=0.6,
            alpha=0.7,
            rasterized=bool(trajectory_rasterized),
        )

        x_epoch = x_neural[epoch_start:epoch_end]
        y_epoch = y_neural[epoch_start:epoch_end]
        color_vals_trial = None
        if isinstance(trajectory_color_values_by_trial, (list, tuple)) and trial_idx < len(trajectory_color_values_by_trial):
            try:
                cand_vals = np.asarray(trajectory_color_values_by_trial[trial_idx], dtype=float).reshape(-1)
            except Exception:
                cand_vals = None
            if cand_vals is not None and cand_vals.size == len(x_epoch):
                color_vals_trial = cand_vals
        if color_vals_trial is not None:
            from matplotlib.collections import LineCollection

            x_epoch_arr = np.asarray(x_epoch, dtype=float)
            y_epoch_arr = np.asarray(y_epoch, dtype=float)
            val_epoch_arr = np.asarray(color_vals_trial, dtype=float)
            valid_xy = np.isfinite(x_epoch_arr) & np.isfinite(y_epoch_arr) & np.isfinite(val_epoch_arr)
            valid_pairs = valid_xy[:-1] & valid_xy[1:]
            if np.any(valid_pairs):
                segs = np.stack(
                    [
                        np.column_stack([x_epoch_arr[:-1][valid_pairs], y_epoch_arr[:-1][valid_pairs]]),
                        np.column_stack([x_epoch_arr[1:][valid_pairs], y_epoch_arr[1:][valid_pairs]]),
                    ],
                    axis=1,
                )
                seg_vals = 0.5 * (val_epoch_arr[:-1][valid_pairs] + val_epoch_arr[1:][valid_pairs])
                cmap_obj = plt.get_cmap(str(trajectory_color_cmap or "vanimo"))
                norm = mcolors.Normalize(
                    vmin=float(-1.0 if trajectory_color_vmin is None else trajectory_color_vmin),
                    vmax=float(1.0 if trajectory_color_vmax is None else trajectory_color_vmax),
                )
                lc = LineCollection(
                    segs,
                    cmap=cmap_obj,
                    norm=norm,
                    linewidths=1.2,
                    zorder=3,
                    capstyle="round",
                    joinstyle="round",
                )
                lc.set_array(seg_vals)
                lc.set_rasterized(bool(trajectory_rasterized))
                ax_traj.add_collection(lc)
            else:
                ax_traj.plot(
                    x_epoch,
                    y_epoch,
                    color="green",
                    linewidth=1.2,
                    rasterized=bool(trajectory_rasterized),
                )
        elif bool(gradient_traversal_trajectory):
            from matplotlib.collections import LineCollection

            x_epoch_arr = np.asarray(x_epoch, dtype=float)
            y_epoch_arr = np.asarray(y_epoch, dtype=float)
            valid_xy = np.isfinite(x_epoch_arr) & np.isfinite(y_epoch_arr)
            if np.sum(valid_xy) >= 2:
                pts = np.column_stack([x_epoch_arr[valid_xy], y_epoch_arr[valid_xy]])
                segs = np.stack([pts[:-1], pts[1:]], axis=1)
                n_seg = segs.shape[0]
                seg_palette = np.array(
                    [
                        [0.82, 0.94, 0.82, 1.0],
                        [0.60, 0.84, 0.60, 1.0],
                        [0.32, 0.68, 0.32, 1.0],
                        [0.00, 0.50, 0.00, 1.0],
                    ],
                    dtype=float,
                )
                if n_seg <= 1:
                    seg_colors = seg_palette[[-1]]
                else:
                    seg_bins = np.floor(np.linspace(0, 4, n_seg, endpoint=False)).astype(int)
                    seg_bins = np.clip(seg_bins, 0, 3)
                    seg_colors = seg_palette[seg_bins]
                lc = LineCollection(
                    segs,
                    colors=seg_colors,
                    linewidths=1.2,
                    zorder=3,
                    capstyle="round",
                    joinstyle="round",
                )
                lc.set_rasterized(bool(trajectory_rasterized))
                ax_traj.add_collection(lc)
            else:
                ax_traj.plot(
                    x_epoch,
                    y_epoch,
                    color="green",
                    linewidth=1.2,
                    rasterized=bool(trajectory_rasterized),
                )
        else:
            ax_traj.plot(
                x_epoch,
                y_epoch,
                color="green",
                linewidth=1.2,
                rasterized=bool(trajectory_rasterized),
            )

        ax_traj.plot(
            x_traj[post_start:],
            y_traj[post_start:],
            color="#FFB6C1",
            linewidth=0.6,
            alpha=0.7,
            rasterized=bool(trajectory_rasterized),
        )

        # Mark PF peak if provided
        if pf_peak_xy is not None and np.all(np.isfinite(pf_peak_xy)):
            ax_traj.scatter(
                pf_peak_xy[0],
                pf_peak_xy[1],
                color=pf_peak_color,
                s=20,
                marker="*",
                zorder=6,
                label=str(pf_peak_label) if trial_idx == 0 else None,
            )

        # Mark center position and add vertical line in trace plot
        center_idx = max_rate_center_indices[trial_idx]
        if center_idx is not None:
            # Get position at center index
            if epoch_start <= center_idx < epoch_end:
                mx = x_neural[center_idx]
                my = y_neural[center_idx]
                center_label = (
                    str(center_position_label)
                    if center_by_pf_position and center_position_label is not None
                    else ("Closest to PF" if center_by_pf_position else "Max FR position")
                )
                ax_traj.scatter(
                    mx,
                    my,
                    color="yellow",
                    edgecolor="black",
                    linewidth=0.3,
                    s=18,
                    marker="*",
                    zorder=7,
                    label=center_label if trial_idx == 0 else None,
                )
                # Add vertical line in trace plot at center time
                ax_trace.axvline(
                    (center_idx - epoch_start) / frame_rate,
                    color="yellow",
                    linestyle="--",
                    linewidth=0.6,
                    zorder=1,
                )

        if (
            isinstance(trajectory_spatial_bin_hd_overlays_by_trial, (list, tuple))
            and trial_idx < len(trajectory_spatial_bin_hd_overlays_by_trial)
        ):
            overlay_obj = trajectory_spatial_bin_hd_overlays_by_trial[trial_idx]
            if isinstance(overlay_obj, dict):
                ov_x = np.asarray(overlay_obj.get("x", np.array([])), dtype=float).reshape(-1)
                ov_y = np.asarray(overlay_obj.get("y", np.array([])), dtype=float).reshape(-1)
                ov_ang = np.asarray(overlay_obj.get("angle", np.array([])), dtype=float).reshape(-1)
                ov_green = np.asarray(overlay_obj.get("is_green", np.array([])), dtype=bool).reshape(-1)
                n_ov = int(min(ov_x.size, ov_y.size, ov_ang.size))
                if n_ov > 0:
                    if ov_green.size < n_ov:
                        ov_green = np.pad(
                            ov_green,
                            (0, n_ov - ov_green.size),
                            mode="constant",
                            constant_values=False,
                        )
                    for kk in range(n_ov):
                        x0 = float(ov_x[kk])
                        y0 = float(ov_y[kk])
                        a0 = float(ov_ang[kk])
                        if not (np.isfinite(x0) and np.isfinite(y0) and np.isfinite(a0)):
                            continue
                        is_green = bool(ov_green[kk])
                        arrow_col = "#0B3D91"
                        # Keep arrows visible but compact (~3x smaller than prior setting).
                        scale = float(max(0.25, trajectory_spatial_bin_hd_overlay_scale_cm / 3.0))
                        dx = float(np.cos(a0) * scale)
                        dy = float(np.sin(a0) * scale)
                        from matplotlib.patches import FancyArrowPatch

                        arrow = FancyArrowPatch(
                            (x0, y0),
                            (x0 + dx, y0 + dy),
                            arrowstyle="->",
                            mutation_scale=(5.8 if is_green else 5.2),
                            linewidth=(0.8 if is_green else 0.7),
                            color=arrow_col,
                            facecolor="none",
                            edgecolor=arrow_col,
                            alpha=float(np.clip(trajectory_spatial_bin_hd_overlay_alpha, 0.0, 1.0)),
                            zorder=(12 if is_green else 11),
                        )
                        arrow.set_clip_on(True)
                        ax_traj.add_patch(arrow)

        avg_n = 5

        def _mean_xy(i0, i1):
            i0 = int(max(0, i0))
            i1 = int(min(len(x_neural), i1))
            if i1 <= i0:
                return None
            xs = np.asarray(x_neural[i0:i1], dtype=float)
            ys = np.asarray(y_neural[i0:i1], dtype=float)
            valid = np.isfinite(xs) & np.isfinite(ys)
            if not np.any(valid):
                return None
            return float(np.nanmean(xs[valid])), float(np.nanmean(ys[valid]))

        def _draw_dir_arrow(p_before, p_after, color):
            if p_before is None or p_after is None:
                return
            dx = float(p_after[0] - p_before[0])
            dy = float(p_after[1] - p_before[1])
            n = float(np.hypot(dx, dy))
            if n <= 0:
                return
            ux, uy = dx / n, dy / n
            midx = 0.5 * (float(p_before[0]) + float(p_after[0]))
            midy = 0.5 * (float(p_before[1]) + float(p_after[1]))
            # Longer arrows for clearer direction visualization.
            arrow_len = min(1.8, max(0.5, 0.8 * n))
            x0, y0 = midx - 0.5 * arrow_len * ux, midy - 0.5 * arrow_len * uy
            x1, y1 = midx + 0.5 * arrow_len * ux, midy + 0.5 * arrow_len * uy
            ax_traj.annotate(
                "",
                xy=(x1, y1),
                xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=0.2, mutation_scale=6),
                zorder=6,
            )

        # Entry direction: average 5 frames before vs 5 frames after entry.
        entry_before = _mean_xy(epoch_start - avg_n, epoch_start)
        entry_after = _mean_xy(epoch_start, epoch_start + avg_n)
        _draw_dir_arrow(entry_before, entry_after, "blue")

        # Exit direction: average 5 frames before vs 5 frames after exit.
        exit_before = _mean_xy(epoch_end - avg_n, epoch_end)
        exit_after = _mean_xy(epoch_end, epoch_end + avg_n)
        _draw_dir_arrow(exit_before, exit_after, "red")

        if trial_idx == 0:
            ax_traj.legend(
                loc="lower left",
                bbox_to_anchor=(0.0, 1.02),
                fontsize=5,
                frameon=False,
                handletextpad=0.1,
                borderpad=0.0,
                borderaxespad=0,
            )

        if isinstance(place_field_contours, (list, tuple)) and len(place_field_contours) > 0:
            for contour_info in place_field_contours:
                if not isinstance(contour_info, dict):
                    continue
                _draw_pf_contour(
                    ax_traj,
                    contour_info.get("mask"),
                    color=contour_info.get("color", "magenta"),
                    linewidth=contour_info.get("linewidth", 0.8),
                    linestyle=contour_info.get("linestyle", "solid"),
                    alpha=contour_info.get("alpha", 1.0),
                )
        else:
            _draw_pf_contour(ax_traj, place_field_mask, color="magenta", linewidth=0.8)

        if bool(trajectory_direction_label_top) and type_label:
            arena_x0, arena_x1 = float(pf_bins[0][0]), float(pf_bins[0][-1])
            arena_y0, arena_y1 = float(pf_bins[1][0]), float(pf_bins[1][-1])
            ax_traj.text(
                0.5 * (arena_x0 + arena_x1),
                0.5 * (arena_y0 + arena_y1),
                type_label,
                ha="center",
                va="center",
                fontsize=float(trajectory_direction_label_fontsize),
                fontname="Arial",
                fontweight="bold",
                color="black",
                zorder=20,
                bbox=dict(
                    boxstyle="round,pad=0.08",
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.7,
                ),
            )

        ax_traj.set_xlim([pf_bins[0][0], pf_bins[0][-1]])
        ax_traj.set_ylim([pf_bins[1][0], pf_bins[1][-1]])
        arena_x0, arena_x1 = float(pf_bins[0][0]), float(pf_bins[0][-1])
        arena_y0, arena_y1 = float(pf_bins[1][0]), float(pf_bins[1][-1])
        ax_traj.add_patch(
            Rectangle(
                (arena_x0, arena_y0),
                arena_x1 - arena_x0,
                arena_y1 - arena_y0,
                fill=False,
                edgecolor="black",
                linewidth=0.25,
                zorder=9,
            )
        )
        ax_traj.set_aspect("equal")
        ax_traj.axis("off")

    ax_scale = axes[n_trials, 0]
    ax_scale_traj = axes[n_trials, 1]
    if n_trials > 0:
        ref_trace_ax = axes[0, 0]
        try:
            ax_scale.sharex(ref_trace_ax)
            ax_scale.sharey(ref_trace_ax)
        except Exception:
            pass
        ax_scale.set_xlim(ref_trace_ax.get_xlim())
        ax_scale.set_ylim(ref_trace_ax.get_ylim())
    else:
        ax_scale.set_xlim([global_x_min, global_x_max])
        ax_scale.set_ylim((0.0, 1.0))
    y_min_scale, y_max_scale = ax_scale.get_ylim()
    y_span_scale = y_max_scale - y_min_scale if y_max_scale > y_min_scale else 1.0

    scale_x_start = global_x_min + 0.2
    scale_x_end = scale_x_start + 1.0
    bar_y = 0.0
    ax_scale.plot(
        [scale_x_start, scale_x_end],
        [bar_y, bar_y],
        color="black",
        linewidth=1.0,
        solid_capstyle="butt",
    )
    vertical_bar_x = scale_x_start - 0.02
    ax_scale.plot(
        [vertical_bar_x, vertical_bar_x],
        [0.0, 1.0],
        color="black",
        linewidth=1.0,
        solid_capstyle="butt",
        clip_on=False,
    )
    ax_scale.text(
        vertical_bar_x - 0.04,
        0.5,
        "1 spk",
        fontsize=6,
        ha="right",
        va="center",
    )
    ax_scale.text(
        (scale_x_start + scale_x_end) / 2,
        bar_y - 0.05 * y_span_scale,
        "1 s",
        fontsize=6,
        ha="center",
        va="top",
    )
    ax_scale.axis("off")
    ax_scale_traj.axis("off")

    if apply_layout:
        # Deterministic spacing; tight_layout can collapse rows for large n_trials.
        hspace_val = 0.12 if extra_rows else 0.0
        plt.subplots_adjust(hspace=hspace_val, wspace=0.01, top=0.9, bottom=0.06)
    if show:
        plt.show()

    pf_theta_mean = None
    pf_slow_mean = None
    pf_rate_mean = None
    pf_ss_rate_mean = None
    pf_cs_rate_mean = None
    pf_ss_pct_mean = None
    pf_cs_pct_mean = None
    pf_theta_sem = None
    pf_slow_sem = None
    pf_rate_sem = None
    pf_ss_rate_sem = None
    pf_cs_rate_sem = None
    pf_ss_pct_sem = None
    pf_cs_pct_sem = None
    if plot_pf_centered_average and pf_peak_xy is not None:
        window_frames = int(round(pf_center_window_sec * frame_rate))
        if window_frames > 0:
            avg_len = window_frames * 2 + 1
            time_rel = np.arange(-window_frames, window_frames + 1) / frame_rate
            theta_source = theta_z if zscore_traces else theta_vm
            theta_amp = np.abs(signal.hilbert(theta_source))
            slow_source = slow_z if zscore_traces else slow_plot
            theta_amp_valid = theta_amp.copy()
            slow_source_valid = slow_source.copy()
            if bad_mask is not None:
                theta_amp_valid[bad_mask] = np.nan
                slow_source_valid[bad_mask] = np.nan
            
            # Count trials with spikes for proper array sizing
            n_valid_trials = sum(trials_with_spikes)
            
            theta_stack = np.full((n_valid_trials, avg_len), np.nan)
            slow_stack = np.full((n_valid_trials, avg_len), np.nan)
            rate_stack = np.full((n_valid_trials, avg_len), np.nan) if include_pf_rate else None
            ss_rate_stack = np.full((n_valid_trials, avg_len), np.nan) if include_pf_ss_rate else None
            cs_rate_stack = np.full((n_valid_trials, avg_len), np.nan) if include_pf_cs_rate else None
            ss_pct_stack = np.full((n_valid_trials, avg_len), np.nan) if include_pf_pct else None
            cs_pct_stack = np.full((n_valid_trials, avg_len), np.nan) if include_pf_pct else None
            baseline_start_offset = int(round(2.0 * frame_rate))
            baseline_end_offset = int(round(1.0 * frame_rate))
            
            valid_trial_idx = 0
            for i, (center_idx, (epoch_start, _)) in enumerate(
                zip(max_rate_center_indices, valid_epochs)
            ):
                # Skip trials without spikes
                if not trials_with_spikes[i]:
                    continue
                if center_idx is None:
                    valid_trial_idx += 1
                    continue
                    
                base_start = max(epoch_start - baseline_start_offset, 0)
                base_end = max(epoch_start - baseline_end_offset, 0)
                baseline_theta = np.nan
                baseline_slow = np.nan
                if base_end > base_start:
                    baseline_theta = np.nanmean(theta_amp_valid[base_start:base_end])
                    baseline_slow = np.nanmean(slow_source_valid[base_start:base_end])
                start_idx = max(center_idx - window_frames, 0)
                end_idx = min(center_idx + window_frames + 1, len(theta_source))
                if end_idx <= start_idx:
                    valid_trial_idx += 1
                    continue
                left_pad = start_idx - (center_idx - window_frames)
                right_pad = (center_idx + window_frames + 1) - end_idx
                write_start = max(0, left_pad)
                write_end = avg_len - max(0, right_pad)
                theta_segment = theta_amp_valid[start_idx:end_idx]
                slow_segment = slow_source_valid[start_idx:end_idx]
                if np.isfinite(baseline_theta):
                    theta_segment = theta_segment - baseline_theta
                if np.isfinite(baseline_slow):
                    slow_segment = slow_segment - baseline_slow
                theta_stack[valid_trial_idx, write_start:write_end] = theta_segment
                slow_stack[valid_trial_idx, write_start:write_end] = slow_segment
                if include_pf_rate and ifr_valid is not None:
                    rate_stack[valid_trial_idx, write_start:write_end] = ifr_valid[start_idx:end_idx]
                if include_pf_ss_rate and ifr_ss_valid is not None:
                    ss_rate_stack[valid_trial_idx, write_start:write_end] = ifr_ss_valid[start_idx:end_idx]
                if include_pf_cs_rate and ifr_cs_valid is not None:
                    cs_rate_stack[valid_trial_idx, write_start:write_end] = ifr_cs_valid[start_idx:end_idx]
                if include_pf_pct and ifr_ss_valid is not None and ifr_cs_valid is not None:
                    ss_seg = ifr_ss_valid[start_idx:end_idx]
                    cs_seg = ifr_cs_valid[start_idx:end_idx]
                    denom = ss_seg + cs_seg
                    valid = np.isfinite(denom) & (denom > 0)
                    ss_pct = np.where(valid, 100.0 * ss_seg / denom, np.nan)
                    cs_pct = np.where(valid, 100.0 * cs_seg / denom, np.nan)
                    ss_pct_stack[valid_trial_idx, write_start:write_end] = ss_pct
                    cs_pct_stack[valid_trial_idx, write_start:write_end] = cs_pct
                valid_trial_idx += 1

            pf_theta_mean = np.nanmean(theta_stack, axis=0)
            pf_slow_mean = np.nanmean(slow_stack, axis=0)
            pf_rate_mean = np.nanmean(rate_stack, axis=0) if include_pf_rate else None
            pf_ss_rate_mean = np.nanmean(ss_rate_stack, axis=0) if include_pf_ss_rate else None
            pf_cs_rate_mean = np.nanmean(cs_rate_stack, axis=0) if include_pf_cs_rate else None
            pf_ss_pct_mean = np.nanmean(ss_pct_stack, axis=0) if include_pf_pct else None
            pf_cs_pct_mean = np.nanmean(cs_pct_stack, axis=0) if include_pf_pct else None
            count_theta = np.sum(np.isfinite(theta_stack), axis=0)
            count_slow = np.sum(np.isfinite(slow_stack), axis=0)
            std_theta = np.nanstd(theta_stack, axis=0, ddof=0)
            std_slow = np.nanstd(slow_stack, axis=0, ddof=0)
            pf_theta_sem = np.where(count_theta > 0, std_theta / np.sqrt(count_theta), np.nan)
            pf_slow_sem = np.where(count_slow > 0, std_slow / np.sqrt(count_slow), np.nan)
            if include_pf_rate and rate_stack is not None:
                count_rate = np.sum(np.isfinite(rate_stack), axis=0)
                std_rate = np.nanstd(rate_stack, axis=0, ddof=0)
                pf_rate_sem = np.where(count_rate > 0, std_rate / np.sqrt(count_rate), np.nan)
            if include_pf_ss_rate and ss_rate_stack is not None:
                count_ss_rate = np.sum(np.isfinite(ss_rate_stack), axis=0)
                std_ss_rate = np.nanstd(ss_rate_stack, axis=0, ddof=0)
                pf_ss_rate_sem = np.where(
                    count_ss_rate > 0, std_ss_rate / np.sqrt(count_ss_rate), np.nan
                )
            if include_pf_cs_rate and cs_rate_stack is not None:
                count_cs_rate = np.sum(np.isfinite(cs_rate_stack), axis=0)
                std_cs_rate = np.nanstd(cs_rate_stack, axis=0, ddof=0)
                pf_cs_rate_sem = np.where(
                    count_cs_rate > 0, std_cs_rate / np.sqrt(count_cs_rate), np.nan
                )
            if include_pf_pct and ss_pct_stack is not None and cs_pct_stack is not None:
                count_ss_pct = np.sum(np.isfinite(ss_pct_stack), axis=0)
                std_ss_pct = np.nanstd(ss_pct_stack, axis=0, ddof=0)
                pf_ss_pct_sem = np.where(
                    count_ss_pct > 0, std_ss_pct / np.sqrt(count_ss_pct), np.nan
                )
                count_cs_pct = np.sum(np.isfinite(cs_pct_stack), axis=0)
                std_cs_pct = np.nanstd(cs_pct_stack, axis=0, ddof=0)
                pf_cs_pct_sem = np.where(
                    count_cs_pct > 0, std_cs_pct / np.sqrt(count_cs_pct), np.nan
                )

            pf_rows = []
            if include_pf_rate:
                pf_rows.append("rate")
            if include_pf_ss_cs_rate:
                pf_rows.append("ss_cs_rate")
            if include_pf_pct:
                pf_rows.append("ss_cs_pct")
            pf_rows.extend(["theta", "slow"])

            row_start = n_trials + 1
            pf_axes_left = [axes[row_start + i, 0] for i in range(len(pf_rows))]
            pf_axes_right = [axes[row_start + i, 1] for i in range(len(pf_rows))]
            row_axes = {name: pf_axes_left[i] for i, name in enumerate(pf_rows)}

            # Keep default GridSpec row order. This guarantees PF average rows are
            # below trial-by-trial rows even when n_trials is large.

            for ax in pf_axes_right:
                ax.axis("off")

            if include_pf_rate and pf_rate_mean is not None:
                ax_pf_rate = row_axes["rate"]
                ax_pf_rate.plot(time_rel, pf_rate_mean, color=firing_rate_color, linewidth=0.7)
                ax_pf_rate.axvline(0, color="black", linewidth=0.5, alpha=0.5)
                ax_pf_rate.set_ylabel("FR\n(Hz)", fontsize=6, fontname="Arial")
                ax_pf_rate.spines["top"].set_visible(False)
                ax_pf_rate.spines["right"].set_visible(False)
                ax_pf_rate.tick_params(labelsize=5)
                ax_pf_rate.tick_params(labelbottom=False)

            if include_pf_ss_cs_rate:
                ax_pf_ss_cs = row_axes["ss_cs_rate"]
                if pf_ss_rate_mean is not None:
                    ax_pf_ss_cs.plot(
                        time_rel, pf_ss_rate_mean, color=simple_spike_color, linewidth=0.7
                    )
                if pf_cs_rate_mean is not None:
                    ax_pf_ss_cs.plot(
                        time_rel, pf_cs_rate_mean, color=complex_spike_color, linewidth=0.7
                    )
                ax_pf_ss_cs.axvline(0, color="black", linewidth=0.5, alpha=0.5)
                ax_pf_ss_cs.set_ylabel("SS/CS FR\n(Hz)", fontsize=6, fontname="Arial")
                ax_pf_ss_cs.spines["top"].set_visible(False)
                ax_pf_ss_cs.spines["right"].set_visible(False)
                ax_pf_ss_cs.tick_params(labelsize=5)
                ax_pf_ss_cs.tick_params(labelbottom=False)

            if include_pf_pct:
                ax_pf_pct = row_axes["ss_cs_pct"]
                if pf_ss_pct_mean is not None:
                    ax_pf_pct.plot(
                        time_rel, pf_ss_pct_mean, color=simple_spike_color, linewidth=0.7
                    )
                if pf_cs_pct_mean is not None:
                    ax_pf_pct.plot(
                        time_rel, pf_cs_pct_mean, color=complex_spike_color, linewidth=0.7
                    )
                ax_pf_pct.axvline(0, color="black", linewidth=0.5, alpha=0.5)
                ax_pf_pct.set_ylabel("SS/CS %", fontsize=6, fontname="Arial")
                ax_pf_pct.spines["top"].set_visible(False)
                ax_pf_pct.spines["right"].set_visible(False)
                ax_pf_pct.tick_params(labelsize=5)
                ax_pf_pct.tick_params(labelbottom=False)

            ax_pf_theta = row_axes["theta"]
            ax_pf_slow = row_axes["slow"]
            ax_pf_theta.plot(time_rel, pf_theta_mean, color="blue", linewidth=0.7)
            ax_pf_slow.plot(time_rel, pf_slow_mean, color="red", linewidth=0.7)
            ax_pf_theta.axvline(0, color="black", linewidth=0.5, alpha=0.5)
            ax_pf_slow.axvline(0, color="black", linewidth=0.5, alpha=0.5)
            ax_pf_theta.set_ylabel("$\\theta$ amp.\n(spk. height)", fontsize=6, fontname="Arial")
            ax_pf_slow.set_ylabel("Slow Vm\n(spk. height)", fontsize=6, fontname="Arial")
            ax_pf_slow.set_xlabel("Time from max FR (s)", fontsize=6, fontname="Arial")
            ax_pf_theta.tick_params(labelbottom=False)
            ax_pf_theta.spines["top"].set_visible(False)
            ax_pf_theta.spines["right"].set_visible(False)
            ax_pf_slow.spines["top"].set_visible(False)
            ax_pf_slow.spines["right"].set_visible(False)
            ax_pf_theta.tick_params(labelsize=5)
            ax_pf_slow.tick_params(labelsize=5)

            for ax in pf_axes_left:
                ax.set_xlim([time_rel[0], time_rel[-1]])

    n_with_spikes = sum(trials_with_spikes)
    print(f"Plotted {n_trials} traversal trials ({n_with_spikes} with spikes in PF, used for average)")
    if return_pf_centered and return_spike_rate_means and return_pf_centered_sem:
        return (
            fig,
            axes,
            valid_epochs,
            pf_theta_mean,
            pf_slow_mean,
            pf_rate_mean,
            pf_ss_rate_mean,
            pf_cs_rate_mean,
            pf_ss_pct_mean,
            pf_cs_pct_mean,
            pf_theta_sem,
            pf_slow_sem,
            pf_rate_sem,
            pf_ss_rate_sem,
            pf_cs_rate_sem,
            pf_ss_pct_sem,
            pf_cs_pct_sem,
        )
    if return_pf_centered and return_spike_rate_means:
        return (
            fig,
            axes,
            valid_epochs,
            pf_theta_mean,
            pf_slow_mean,
            pf_rate_mean,
            pf_ss_rate_mean,
            pf_cs_rate_mean,
            pf_ss_pct_mean,
            pf_cs_pct_mean,
        )
    if return_pf_centered and return_pf_centered_sem:
        return (
            fig,
            axes,
            valid_epochs,
            pf_theta_mean,
            pf_slow_mean,
            pf_rate_mean,
            pf_theta_sem,
            pf_slow_sem,
            pf_rate_sem,
        )
    if return_pf_centered:
        return fig, axes, valid_epochs, pf_theta_mean, pf_slow_mean, pf_rate_mean
    return fig, axes, valid_epochs


def plot_place_field_traversal_trials_with_cb_example_centered_by_max_rate(
    trace,
    theta_vm,
    slow_vm,
    speed,
    x_neural,
    y_neural,
    traversal_epochs,
    place_field_mask,
    pf_bins,
    frame_rate,
    complex_bursts_dicts,
    cell_idx,
    burst_pre_ms=100,
    burst_post_ms=100,
    padding_sec=2.0,
    zscore_traces=False,
    bad_timepoints=None,
    show=True,
    return_pf_centered=False,
    return_spike_rate_means=False,
    return_trial_firing_rates=False,
    center_by_pf_position=False,
    **kwargs,
):
    """
    Similar to plot_place_field_traversal_trials_with_cb_example, but centers by maximum
    firing rate within the PF traversal instead of closest position to PF peak.
    Trials without spikes in the PF traversal are excluded from the average.

    If return_trial_firing_rates is True, an extra dict is appended to the return tuple
    containing per-trial traversal start times and firing rates (all spikes, SS, CS),
    plus optional PF-centered (around max-rate) firing rate traces when spike indices
    are provided via all_spikes/refined_SS/all_CS_spikes.
    
    If center_by_pf_position=True, instead of centering by max firing rate, centers
    by the timepoint closest to the average PF peak position (pf_peak_xy), but still
    excludes traversals without spikes from the average calculation.
    """
    bad_timepoints = kwargs.pop("bad_timepoints", bad_timepoints)
    pf_occupancy_mask = kwargs.get("pf_occupancy_mask", None)
    pf_mask_shape_ref = np.asarray(place_field_mask, dtype=bool).shape
    if pf_occupancy_mask is None:
        pf_occupancy_mask = np.asarray(place_field_mask, dtype=bool)
    else:
        pf_occupancy_mask = np.asarray(pf_occupancy_mask, dtype=bool)
        if pf_occupancy_mask.shape != pf_mask_shape_ref:
            pf_occupancy_mask = np.asarray(place_field_mask, dtype=bool)
    want_spike_rate_means = return_pf_centered and return_spike_rate_means

    def _build_trial_firing_rate_data(valid_epochs_local):
        n_trials_local = len(valid_epochs_local)
        epoch_start_frames = np.asarray([s for s, _ in valid_epochs_local], dtype=int)
        epoch_end_frames = np.asarray([e for _, e in valid_epochs_local], dtype=int)
        duration_frames = epoch_end_frames - epoch_start_frames
        duration_sec = duration_frames.astype(float) / float(frame_rate)
        epoch_start_sec = epoch_start_frames.astype(float) / float(frame_rate)
        epoch_end_sec = epoch_end_frames.astype(float) / float(frame_rate)

        def _get_spike_indices_local(spike_source):
            if spike_source is None:
                return None
            if (
                cell_idx is not None
                and isinstance(spike_source, (list, tuple))
                and len(spike_source) > 0
                and isinstance(spike_source[0], (list, np.ndarray))
            ):
                if cell_idx >= len(spike_source):
                    return None
                return np.asarray(spike_source[cell_idx], dtype=int)
            return np.asarray(spike_source, dtype=int)

        def _count_spikes_in_epochs(spike_indices, starts, ends):
            if spike_indices is None or len(spike_indices) == 0:
                return np.zeros(starts.shape[0], dtype=int)
            spike_indices = np.asarray(spike_indices, dtype=int)
            spike_indices = spike_indices[(spike_indices >= 0) & (spike_indices < len(trace))]
            if spike_indices.size == 0:
                return np.zeros(starts.shape[0], dtype=int)
            spike_sorted = np.sort(spike_indices)
            left = np.searchsorted(spike_sorted, starts, side="left")
            right = np.searchsorted(spike_sorted, ends, side="left")
            return (right - left).astype(int)

        all_spike_indices_local = _get_spike_indices_local(kwargs.get("all_spikes", None))
        ss_spike_indices_local = _get_spike_indices_local(kwargs.get("refined_SS", None))
        cs_spike_indices_local = _get_spike_indices_local(kwargs.get("all_CS_spikes", None))

        n_all = _count_spikes_in_epochs(all_spike_indices_local, epoch_start_frames, epoch_end_frames)
        n_ss = _count_spikes_in_epochs(ss_spike_indices_local, epoch_start_frames, epoch_end_frames)
        n_cs = _count_spikes_in_epochs(cs_spike_indices_local, epoch_start_frames, epoch_end_frames)

        with np.errstate(divide="ignore", invalid="ignore"):
            fr_all = np.where(duration_sec > 0, n_all / duration_sec, np.nan)
            fr_ss = np.where(duration_sec > 0, n_ss / duration_sec, np.nan)
            fr_cs = np.where(duration_sec > 0, n_cs / duration_sec, np.nan)

        trial_data = {
            "epoch_start_frames": epoch_start_frames,
            "epoch_end_frames": epoch_end_frames,
            "epoch_start_sec": epoch_start_sec,
            "epoch_end_sec": epoch_end_sec,
            "duration_sec": duration_sec,
            "n_spikes_all": n_all,
            "n_spikes_ss": n_ss,
            "n_spikes_cs": n_cs,
            "fr_all_hz": fr_all,
            "fr_ss_hz": fr_ss,
            "fr_cs_hz": fr_cs,
        }

        firing_rate_bin_ms = kwargs.get("firing_rate_bin_ms", 100)
        firing_rate_smooth_ms = kwargs.get("firing_rate_smooth_ms", 20)
        pf_center_window_sec_local = kwargs.get("pf_center_window_sec", padding_sec)
        window_frames = int(round(float(pf_center_window_sec_local) * float(frame_rate)))
        if window_frames <= 0 or n_trials_local == 0:
            return trial_data

        avg_len = window_frames * 2 + 1
        time_rel_centered_sec = np.arange(-window_frames, window_frames + 1, dtype=float) / float(frame_rate)

        def _compute_ifr(spike_indices):
            if spike_indices is None or len(spike_indices) == 0:
                return None
            spike_indices = np.asarray(spike_indices, dtype=int)
            spike_indices = spike_indices[(spike_indices >= 0) & (spike_indices < len(trace))]
            if spike_indices.size == 0:
                return None
            spike_train = np.zeros(len(trace), dtype=float)
            spike_train[spike_indices] = 1.0
            bin_frames = max(1, int(round(firing_rate_bin_ms / 1000 * frame_rate)))
            bin_kernel = np.ones(bin_frames, dtype=float) / bin_frames
            ifr_local = np.convolve(spike_train, bin_kernel, mode="same") * frame_rate
            smooth_frames = max(1, int(round(firing_rate_smooth_ms / 1000 * frame_rate)))
            if smooth_frames > 1:
                smooth_kernel = np.ones(smooth_frames, dtype=float) / smooth_frames
                ifr_local = np.convolve(ifr_local, smooth_kernel, mode="same")
            return ifr_local

        ifr_all = _compute_ifr(all_spike_indices_local)
        ifr_ss = _compute_ifr(ss_spike_indices_local)
        ifr_cs = _compute_ifr(cs_spike_indices_local)

        pf_peak_xy_local = kwargs.get("pf_peak_xy", None)
        max_pf_distance_cm_local = kwargs.get("max_pf_distance_cm", None)
        max_rate_center_indices_local = []
        for epoch_start, epoch_end in valid_epochs_local:
            center_idx = None
            has_spikes = False
            if all_spike_indices_local is not None:
                spikes_in_epoch = (all_spike_indices_local >= epoch_start) & (
                    all_spike_indices_local < epoch_end
                )
                has_spikes = np.any(spikes_in_epoch)

            if center_by_pf_position and pf_peak_xy_local is not None and np.all(np.isfinite(pf_peak_xy_local)):
                # Center by closest position to PF peak (works even without spikes)
                x_epoch = x_neural[epoch_start:epoch_end]
                y_epoch = y_neural[epoch_start:epoch_end]
                valid_epoch = (~np.isnan(x_epoch)) & (~np.isnan(y_epoch))
                if np.any(valid_epoch):
                    dx = x_epoch[valid_epoch] - pf_peak_xy_local[0]
                    dy = y_epoch[valid_epoch] - pf_peak_xy_local[1]
                    dist = np.sqrt(dx**2 + dy**2)
                    min_dist = np.min(dist)
                    # Check if closest point is within allowed distance
                    if max_pf_distance_cm_local is not None and min_dist > max_pf_distance_cm_local:
                        center_idx = None  # Exclude this trial
                    else:
                        closest_idx = np.argmin(dist)
                        valid_indices = np.where(valid_epoch)[0]
                        center_idx = int(epoch_start + valid_indices[closest_idx])
            elif has_spikes and ifr_all is not None:
                # Center by max firing rate (requires spikes)
                ifr_segment = ifr_all[epoch_start:epoch_end]
                if len(ifr_segment) > 0 and np.any(np.isfinite(ifr_segment)):
                    max_rel_idx = int(np.nanargmax(ifr_segment))
                    center_idx = int(epoch_start + max_rel_idx)
            max_rate_center_indices_local.append(center_idx)

        def _extract_centered_trials(ifr_signal, centers):
            if ifr_signal is None:
                return None
            out = np.full((n_trials_local, avg_len), np.nan, dtype=float)
            for i, center_idx in enumerate(centers):
                if center_idx is None:
                    continue
                start_idx = max(center_idx - window_frames, 0)
                end_idx = min(center_idx + window_frames + 1, len(ifr_signal))
                if end_idx <= start_idx:
                    continue
                left_pad = start_idx - (center_idx - window_frames)
                right_pad = (center_idx + window_frames + 1) - end_idx
                write_start = max(0, left_pad)
                write_end = avg_len - max(0, right_pad)
                out[i, write_start:write_end] = ifr_signal[start_idx:end_idx]
            return out

        centers_arr = np.asarray(
            [(-1 if c is None else int(c)) for c in max_rate_center_indices_local], dtype=int
        )
        with np.errstate(invalid="ignore"):
            center_sec = np.where(centers_arr >= 0, centers_arr.astype(float) / float(frame_rate), np.nan)
            center_rel_sec = np.where(
                centers_arr >= 0,
                (centers_arr.astype(float) - epoch_start_frames.astype(float)) / float(frame_rate),
                np.nan,
            )

        trial_data.update(
            {
                "center_frames": centers_arr,
                "center_sec": center_sec,
                "center_rel_sec": center_rel_sec,
                "time_rel_centered_sec": time_rel_centered_sec,
                "ifr_all_centered_hz": _extract_centered_trials(ifr_all, max_rate_center_indices_local),
                "ifr_ss_centered_hz": _extract_centered_trials(ifr_ss, max_rate_center_indices_local),
                "ifr_cs_centered_hz": _extract_centered_trials(ifr_cs, max_rate_center_indices_local),
            }
        )
        return trial_data

    if return_pf_centered:
        if want_spike_rate_means:
            (
                fig,
                axes,
                valid_epochs,
                theta_mean,
                slow_mean,
                rate_mean,
                ss_rate_mean,
                cs_rate_mean,
                ss_pct_mean,
                cs_pct_mean,
            ) = plot_place_field_traversal_trials_centered_by_max_rate(
                trace,
                theta_vm,
                slow_vm,
                speed,
                x_neural,
                y_neural,
                traversal_epochs,
                place_field_mask,
                pf_bins,
                frame_rate,
                cell_idx=cell_idx,
                padding_sec=padding_sec,
                zscore_traces=zscore_traces,
                bad_timepoints=bad_timepoints,
                show=False,
                return_pf_centered=True,
                return_spike_rate_means=True,
                center_by_pf_position=center_by_pf_position,
                **kwargs,
            )
        else:
            fig, axes, valid_epochs, theta_mean, slow_mean, rate_mean = plot_place_field_traversal_trials_centered_by_max_rate(
                trace,
                theta_vm,
                slow_vm,
                speed,
                x_neural,
                y_neural,
                traversal_epochs,
                place_field_mask,
                pf_bins,
                frame_rate,
                cell_idx=cell_idx,
                padding_sec=padding_sec,
                zscore_traces=zscore_traces,
                bad_timepoints=bad_timepoints,
                show=False,
                return_pf_centered=True,
                center_by_pf_position=center_by_pf_position,
                **kwargs,
            )
    else:
        fig, axes, valid_epochs = plot_place_field_traversal_trials_centered_by_max_rate(
            trace,
            theta_vm,
            slow_vm,
            speed,
            x_neural,
            y_neural,
            traversal_epochs,
            place_field_mask,
            pf_bins,
            frame_rate,
            cell_idx=cell_idx,
            padding_sec=padding_sec,
            zscore_traces=zscore_traces,
            bad_timepoints=bad_timepoints,
            show=False,
            return_pf_centered=False,
            return_spike_rate_means=bool(return_spike_rate_means),
            center_by_pf_position=center_by_pf_position,
            **kwargs,
        )
    if fig is None:
        trial_data = _build_trial_firing_rate_data([]) if return_trial_firing_rates else None
        if return_pf_centered:
            if want_spike_rate_means:
                if return_trial_firing_rates:
                    return None, None, [], None, None, None, None, None, None, None, trial_data
                return None, None, [], None, None, None, None, None, None, None
            if return_trial_firing_rates:
                return None, None, [], None, None, None, trial_data
            return None, None, [], None, None, None
        if return_trial_firing_rates:
            return None, None, [], trial_data
        return None, None, []

    trial_data = _build_trial_firing_rate_data(valid_epochs) if return_trial_firing_rates else None

    if complex_bursts_dicts is None or cell_idx is None:
        if show:
            plt.show()
        if return_pf_centered:
            if want_spike_rate_means:
                out = (
                    fig,
                    axes,
                    valid_epochs,
                    theta_mean,
                    slow_mean,
                    rate_mean,
                    ss_rate_mean,
                    cs_rate_mean,
                    ss_pct_mean,
                    cs_pct_mean,
                )
                if return_trial_firing_rates:
                    return (*out, trial_data)
                return out
            out = (fig, axes, valid_epochs, theta_mean, slow_mean, rate_mean)
            if return_trial_firing_rates:
                return (*out, trial_data)
            return out
        out = (fig, axes, valid_epochs)
        if return_trial_firing_rates:
            return (*out, trial_data)
        return out

    if isinstance(complex_bursts_dicts, (list, tuple)):
        if cell_idx >= len(complex_bursts_dicts):
            if show:
                plt.show()
            if return_pf_centered:
                if want_spike_rate_means:
                    out = (
                        fig,
                        axes,
                        valid_epochs,
                        theta_mean,
                        slow_mean,
                        rate_mean,
                        ss_rate_mean,
                        cs_rate_mean,
                        ss_pct_mean,
                        cs_pct_mean,
                    )
                    if return_trial_firing_rates:
                        return (*out, trial_data)
                    return out
                out = (fig, axes, valid_epochs, theta_mean, slow_mean, rate_mean)
                if return_trial_firing_rates:
                    return (*out, trial_data)
                return out
            out = (fig, axes, valid_epochs)
            if return_trial_firing_rates:
                return (*out, trial_data)
            return out
        bursts = complex_bursts_dicts[cell_idx]
    else:
        bursts = complex_bursts_dicts

    trace_arr = np.asarray(trace, dtype=float)
    trace_z = _safe_zscore(trace_arr, zscore_traces)
    starts = np.asarray(bursts.get("starts", []), dtype=int)
    ends = np.asarray(bursts.get("ends", []), dtype=int)
    amplitudes = np.asarray(bursts.get("amplitudes", []), dtype=float)

    # Extract plateau data from kwargs
    _plateaus_dicts_local = kwargs.get("plateaus_dicts", None)
    plat_starts = np.array([], dtype=int)
    plat_ends = np.array([], dtype=int)
    if _plateaus_dicts_local is not None and cell_idx is not None:
        if isinstance(_plateaus_dicts_local, (list, tuple)) and cell_idx < len(_plateaus_dicts_local):
            _plat_dict = _plateaus_dicts_local[cell_idx]
        elif isinstance(_plateaus_dicts_local, dict):
            _plat_dict = _plateaus_dicts_local
        else:
            _plat_dict = None
        if _plat_dict is not None:
            plat_starts = np.asarray(_plat_dict.get("starts", []), dtype=int)
            plat_ends = np.asarray(_plat_dict.get("ends", []), dtype=int)

    if starts.size > 0 and ends.size > 0:
        if amplitudes.size != starts.size:
            amp_fallback = np.full(starts.shape, np.nan, dtype=float)
            for i, start_idx in enumerate(starts):
                if 0 <= start_idx < len(trace_arr):
                    if start_idx > 0:
                        base = np.nanmin(trace_arr[max(0, start_idx - 3) : start_idx])
                    else:
                        base = trace_arr[start_idx]
                    if not np.isfinite(base):
                        base = trace_arr[start_idx]
                    amp_fallback[i] = trace_arr[start_idx] - base
            amplitudes = amp_fallback

    n_trials = len(valid_epochs)
    if n_trials == 0:
        if show:
            plt.show()
        if return_pf_centered:
            if want_spike_rate_means:
                out = (
                    fig,
                    axes,
                    valid_epochs,
                    theta_mean,
                    slow_mean,
                    rate_mean,
                    ss_rate_mean,
                    cs_rate_mean,
                    ss_pct_mean,
                    cs_pct_mean,
                )
                if return_trial_firing_rates:
                    return (*out, trial_data)
                return out
            out = (fig, axes, valid_epochs, theta_mean, slow_mean, rate_mean)
            if return_trial_firing_rates:
                return (*out, trial_data)
            return out
        out = (fig, axes, valid_epochs)
        if return_trial_firing_rates:
            return (*out, trial_data)
        return out

    pos_trace = axes[0, 0].get_position()
    pos_traj = axes[0, 1].get_position()
    left = pos_trace.x0
    right = pos_traj.x1
    total_width = right - left
    traj_width = pos_traj.width
    if total_width > 0 and traj_width > 0:
        scale = total_width / (total_width + traj_width)
    else:
        scale = 1.0
    new_col_width = traj_width * scale

    for ax in fig.axes:
        pos = ax.get_position()
        new_x0 = left + (pos.x0 - left) * scale
        new_width = pos.width * scale
        ax.set_position([new_x0, pos.y0, new_width, pos.height])

    new_col_x0 = right - new_col_width
    trajectory_colorbar = bool(kwargs.get("trajectory_colorbar", False))
    trajectory_colorbar_ticks = kwargs.get("trajectory_colorbar_ticks", (-1.0, 0.0, 1.0))
    trajectory_colorbar_label = kwargs.get("trajectory_colorbar_label", None)
    first_example_ax = None
    example_axes: list[Any] = []
    example_trace_extents: list[tuple[float, float]] = []

    padding_frames = int(padding_sec * frame_rate)
    pre_frames = int(round(burst_pre_ms / 1000 * frame_rate))
    post_frames = int(round(burst_post_ms / 1000 * frame_rate))

    # Compute max firing rate center indices for each trial (needed for finding closest burst)
    def _get_spike_indices_local(spike_source):
        if spike_source is None:
            return None
        if (
            cell_idx is not None
            and isinstance(spike_source, (list, tuple))
            and len(spike_source) > 0
            and isinstance(spike_source[0], (list, np.ndarray))
        ):
            return np.asarray(spike_source[cell_idx], dtype=int)
        return np.asarray(spike_source, dtype=int)

    all_spikes_local = kwargs.get("all_spikes", None)
    all_spike_indices = _get_spike_indices_local(all_spikes_local)

    bad_mask = None
    if bad_timepoints is not None:
        bad_mask = np.asarray(bad_timepoints)
        if bad_mask.dtype != bool:
            if bad_mask.ndim == 1 and bad_mask.size == len(trace):
                bad_mask = bad_mask.astype(bool)
            else:
                bad_idx = np.asarray(bad_mask, dtype=int).reshape(-1)
                bad_idx = bad_idx[(bad_idx >= 0) & (bad_idx < len(trace))]
                bad_mask_bool = np.zeros(len(trace), dtype=bool)
                bad_mask_bool[bad_idx] = True
                bad_mask = bad_mask_bool
        if bad_mask.shape[0] != len(trace):
            raise ValueError("bad_timepoints must match trace length or be index list.")
    
    # Compute instantaneous firing rate if we have spike data
    ifr_local = None
    if all_spike_indices is not None and len(all_spike_indices) > 0:
        firing_rate_bin_ms = kwargs.get("firing_rate_bin_ms", 100)
        firing_rate_smooth_ms = kwargs.get("firing_rate_smooth_ms", 20)
        valid_spikes = all_spike_indices[(all_spike_indices >= 0) & (all_spike_indices < len(trace))]
        if valid_spikes.size > 0:
            spike_train = np.zeros(len(trace), dtype=float)
            spike_train[valid_spikes] = 1.0
            bin_frames = max(1, int(round(firing_rate_bin_ms / 1000 * frame_rate)))
            bin_kernel = np.ones(bin_frames, dtype=float) / bin_frames
            ifr_local = np.convolve(spike_train, bin_kernel, mode="same") * frame_rate
            smooth_frames = max(1, int(round(firing_rate_smooth_ms / 1000 * frame_rate)))
            if smooth_frames > 1:
                smooth_kernel = np.ones(smooth_frames, dtype=float) / smooth_frames
                ifr_local = np.convolve(ifr_local, smooth_kernel, mode="same")

    # Compute center indices for each trial (for burst example selection)
    pf_peak_xy_local = kwargs.get("pf_peak_xy", None)
    max_pf_distance_cm_local = kwargs.get("max_pf_distance_cm", None)
    max_rate_center_indices = []
    for epoch_start, epoch_end in valid_epochs:
        max_rate_idx = None
        has_spikes = False
        if all_spike_indices is not None:
            spikes_in_epoch = (all_spike_indices >= epoch_start) & (all_spike_indices < epoch_end)
            has_spikes = np.any(spikes_in_epoch)

        if center_by_pf_position and pf_peak_xy_local is not None and np.all(np.isfinite(pf_peak_xy_local)):
            # Center by closest position to PF peak (works even without spikes)
            x_epoch = x_neural[epoch_start:epoch_end]
            y_epoch = y_neural[epoch_start:epoch_end]
            valid_epoch = (~np.isnan(x_epoch)) & (~np.isnan(y_epoch))
            if np.any(valid_epoch):
                dx = x_epoch[valid_epoch] - pf_peak_xy_local[0]
                dy = y_epoch[valid_epoch] - pf_peak_xy_local[1]
                dist = np.sqrt(dx**2 + dy**2)
                min_dist = np.min(dist)
                # Check if closest point is within allowed distance
                if max_pf_distance_cm_local is not None and min_dist > max_pf_distance_cm_local:
                    max_rate_idx = None  # Exclude this trial
                else:
                    closest_idx = np.argmin(dist)
                    valid_indices = np.where(valid_epoch)[0]
                    max_rate_idx = epoch_start + valid_indices[closest_idx]
        elif has_spikes and ifr_local is not None:
            # Center by max firing rate (requires spikes)
            ifr_segment = ifr_local[epoch_start:epoch_end]
            if len(ifr_segment) > 0 and np.any(np.isfinite(ifr_segment)):
                max_rel_idx = np.nanargmax(ifr_segment)
                max_rate_idx = epoch_start + max_rel_idx
        max_rate_center_indices.append(max_rate_idx)

    def _add_time_scale_bar(ax, bar_ms, label):
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        if not (
            np.isfinite(x_min)
            and np.isfinite(x_max)
            and np.isfinite(y_min)
            and np.isfinite(y_max)
        ):
            return
        if x_max <= x_min or y_max <= y_min:
            return
        x_range = x_max - x_min
        if bar_ms >= x_range:
            return
        y_range = y_max - y_min if y_max != y_min else 1.0
        x0 = x_min + 0.05 * x_range
        x1 = x0 + bar_ms
        right_limit = x_max - 0.05 * x_range
        if x1 > right_limit:
            x1 = right_limit
            x0 = x1 - bar_ms
        y_bar = y_min + 0.05 * y_range
        ax.plot(
            [x0, x1],
            [y_bar, y_bar],
            color="black",
            linewidth=1.0,
            solid_capstyle="butt",
        )
        ax.text(
            (x0 + x1) / 2,
            y_bar - 0.05 * y_range,
            label,
            fontsize=6,
            fontname="Arial",
            ha="center",
            va="top",
        )

    def _add_example_y_scale_bar(ax, y0: float = 0.0, y1: float = 1.0) -> None:
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        if not (
            np.isfinite(x_min)
            and np.isfinite(x_max)
            and np.isfinite(y_min)
            and np.isfinite(y_max)
        ):
            return
        if x_max <= x_min or y_max <= y_min:
            return
        y_low = float(min(y0, y1))
        y_high = float(max(y0, y1))
        if y_low < y_min or y_high > y_max:
            return
        x_range = x_max - x_min
        x_bar = x_min + 0.08 * x_range
        ax.plot(
            [x_bar, x_bar],
            [y0, y1],
            color="black",
            linewidth=1.0,
            solid_capstyle="butt",
            clip_on=False,
        )
        ax.text(
            x_bar - 0.03 * x_range,
            y0,
            "0",
            fontsize=6,
            fontname="Arial",
            ha="right",
            va="center",
        )
        ax.text(
            x_bar - 0.03 * x_range,
            y1,
            "1",
            fontsize=6,
            fontname="Arial",
            ha="right",
            va="center",
        )

    for trial_idx, (epoch_start, epoch_end) in enumerate(valid_epochs):
        ax_trace = axes[trial_idx, 0]
        ax_traj = axes[trial_idx, 1]
        pos = ax_traj.get_position()
        ax_example = fig.add_axes([new_col_x0, pos.y0, new_col_width, pos.height])
        if trial_idx == 0:
            first_example_ax = ax_example

        # Shade plateau regions in red
        if plat_starts.size > 0 and plat_ends.size > 0:
            vis_start = epoch_start - padding_frames
            vis_end = epoch_end + padding_frames
            for ps, pe in zip(plat_starts, plat_ends):
                if pe >= vis_start and ps <= vis_end:
                    pt_start = (max(ps, vis_start) - epoch_start) / frame_rate
                    pt_end = (min(pe, vis_end) - epoch_start) / frame_rate
                    ax_trace.axvspan(pt_start, pt_end, color='red', alpha=0.50, zorder=0)

        chosen_idx = None
        chosen_start = None
        chosen_end = None
        chosen_is_plateau = False
        chosen_is_spike_fallback = False
        max_rate_idx = max_rate_center_indices[trial_idx]
        target_idx = None
        # Primary reference timepoint:
        # max slow Vm inside PF during this traversal, with fallback to max-rate center.
        x_epoch = x_neural[epoch_start:epoch_end]
        y_epoch = y_neural[epoch_start:epoch_end]
        in_pf_epoch = _positions_in_place_field(x_epoch, y_epoch, pf_bins, pf_occupancy_mask)
        if np.any(in_pf_epoch):
            slow_epoch = np.asarray(slow_vm[epoch_start:epoch_end], dtype=float)
            valid_slow_pf = in_pf_epoch & np.isfinite(slow_epoch)
            if np.any(valid_slow_pf):
                valid_rel = np.where(valid_slow_pf)[0]
                rel_peak = valid_rel[np.nanargmax(slow_epoch[valid_slow_pf])]
                target_idx = int(epoch_start + rel_peak)
        if target_idx is None and max_rate_idx is not None:
            target_idx = int(max_rate_idx)

        # Plateau takes priority over complex burst for third-column example.
        n_plat = int(min(plat_starts.size, plat_ends.size))
        if n_plat > 0:
            p_starts = plat_starts[:n_plat]
            p_ends = plat_ends[:n_plat]
            in_plat = np.where((p_starts >= epoch_start) & (p_starts <= epoch_end))[0]
            if in_plat.size > 0:
                if target_idx is not None:
                    p_centers = 0.5 * (p_starts[in_plat].astype(float) + p_ends[in_plat].astype(float))
                    chosen_local = int(np.argmin(np.abs(p_centers - float(target_idx))))
                    chosen_idx = int(in_plat[chosen_local])
                else:
                    p_durs = (p_ends[in_plat] - p_starts[in_plat]).astype(float)
                    if np.any(np.isfinite(p_durs)):
                        chosen_idx = int(in_plat[int(np.nanargmax(p_durs))])
                    else:
                        chosen_idx = int(in_plat[0])
                chosen_start = int(p_starts[chosen_idx])
                chosen_end = int(p_ends[chosen_idx])
                chosen_is_plateau = True

        # Fallback to burst if no plateau was selected.
        if (not chosen_is_plateau) and starts.size > 0 and ends.size > 0:
            in_traversal = np.where((starts >= epoch_start) & (starts <= epoch_end))[0]
            if in_traversal.size > 0:
                if target_idx is not None:
                    burst_starts_in_traversal = starts[in_traversal].astype(float)
                    if ends.size == starts.size:
                        burst_ends_in_traversal = ends[in_traversal].astype(float)
                    else:
                        burst_ends_in_traversal = burst_starts_in_traversal
                    burst_centers_in_traversal = 0.5 * (
                        burst_starts_in_traversal + burst_ends_in_traversal
                    )
                    distances = np.abs(burst_centers_in_traversal - float(target_idx))
                    closest_burst_local_idx = int(np.argmin(distances))
                    chosen_idx = int(in_traversal[closest_burst_local_idx])
                else:
                    cand_amp = amplitudes[in_traversal] if amplitudes.size == starts.size else None
                    if cand_amp is not None and np.any(np.isfinite(cand_amp)):
                        chosen_idx = int(in_traversal[np.nanargmax(cand_amp)])
                    else:
                        chosen_idx = int(in_traversal[0])
                chosen_start = int(starts[chosen_idx])
                chosen_end = int(ends[chosen_idx]) if chosen_idx < len(ends) else chosen_start

        # Final fallback: if there is no plateau/CB example, zoom around the
        # PF-restricted all-spike firing-rate maximum for this traversal.
        if chosen_start is None or chosen_end is None:
            fallback_center_idx = None
            has_spikes_in_epoch = False
            if all_spike_indices is not None and len(all_spike_indices) > 0:
                spikes_in_epoch = (all_spike_indices >= epoch_start) & (all_spike_indices < epoch_end)
                has_spikes_in_epoch = bool(np.any(spikes_in_epoch))
            if has_spikes_in_epoch and ifr_local is not None and np.any(in_pf_epoch):
                ifr_epoch = np.asarray(ifr_local[epoch_start:epoch_end], dtype=float)
                fallback_valid = np.asarray(in_pf_epoch, dtype=bool) & np.isfinite(ifr_epoch)
                if bad_mask is not None:
                    fallback_valid &= (~bad_mask[epoch_start:epoch_end])
                if np.any(fallback_valid):
                    valid_rel = np.where(fallback_valid)[0]
                    fallback_rel = int(valid_rel[np.nanargmax(ifr_epoch[fallback_valid])])
                    fallback_center_idx = int(epoch_start + fallback_rel)
            if fallback_center_idx is not None:
                chosen_start = fallback_center_idx
                chosen_end = fallback_center_idx
                chosen_is_spike_fallback = True

        if chosen_start is None or chosen_end is None:
            ax_example.text(
                0.5,
                0.5,
                "No event",
                transform=ax_example.transAxes,
                ha="center",
                va="center",
                fontsize=6,
                fontname="Arial",
            )
            ax_example.axis("off")
            continue

        burst_start = int(chosen_start)
        burst_end = int(chosen_end)
        if burst_end < burst_start:
            burst_end = burst_start

        t_start = (burst_start - epoch_start) / frame_rate
        t_end = (burst_end - epoch_start) / frame_rate
        if t_end <= t_start:
            t_end = t_start + (1.0 / frame_rate)
        start_idx = epoch_start - padding_frames
        end_idx = epoch_end + padding_frames
        if start_idx < 0:
            start_idx = 0
        if end_idx > len(trace_z):
            end_idx = len(trace_z)
        trace_window = trace_z[start_idx:end_idx]
        trace_min = np.nanmin(trace_window) if trace_window.size > 0 else np.nan
        trace_max = np.nanmax(trace_window) if trace_window.size > 0 else np.nan
        if not np.isfinite(trace_min) or not np.isfinite(trace_max) or trace_max == trace_min:
            trace_min, trace_max = ax_trace.get_ylim()
        if chosen_is_spike_fallback:
            box_start_idx = max(burst_start - pre_frames, start_idx)
            box_end_idx = min(burst_end + post_frames + 1, end_idx)
            box_t_start = (box_start_idx - epoch_start) / frame_rate
            box_t_end = (box_end_idx - epoch_start) / frame_rate
            if box_t_end <= box_t_start:
                box_t_end = box_t_start + (1.0 / frame_rate)
            rect = Rectangle(
                (box_t_start, trace_min),
                box_t_end - box_t_start,
                trace_max - trace_min,
                fill=False,
                edgecolor="gray",
                linestyle="--",
                linewidth=0.25,
                zorder=4,
            )
            ax_trace.add_patch(rect)
        else:
            rect = Rectangle(
                (t_start, trace_min),
                t_end - t_start,
                trace_max - trace_min,
                fill=False,
                edgecolor=("red" if chosen_is_plateau else "gray"),
                linestyle="--",
                linewidth=0.25,
                zorder=4,
            )
            ax_trace.add_patch(rect)

        win_start = max(burst_start - pre_frames, 0)
        win_end = min(burst_end + post_frames + 1, len(trace_arr))
        if win_end <= win_start:
            ax_example.text(
                0.5,
                0.5,
                "No burst",
                transform=ax_example.transAxes,
                ha="center",
                va="center",
                fontsize=6,
                fontname="Arial",
            )
            ax_example.axis("off")
            continue

        t_ms = (np.arange(win_start, win_end) - burst_start) / frame_rate * 1000
        example_trace = np.asarray(trace_arr[win_start:win_end], dtype=float)
        ax_example.plot(t_ms, example_trace, color="black", linewidth=0.6)
        finite_example = example_trace[np.isfinite(example_trace)]
        if finite_example.size > 0:
            example_trace_extents.append(
                (float(np.nanmin(finite_example)), float(np.nanmax(finite_example)))
            )
            example_axes.append(ax_example)
        ax_example.set_xlim([t_ms[0], t_ms[-1]])
        _add_time_scale_bar(ax_example, 50.0, "50 ms")
        ax_example.set_xticks([])
        ax_example.set_yticks([])
        ax_example.spines["top"].set_visible(False)
        ax_example.spines["right"].set_visible(False)
        ax_example.spines["left"].set_visible(False)
        ax_example.spines["bottom"].set_visible(False)

    if len(example_axes) > 0 and len(example_trace_extents) > 0:
        y_min_shared = float(np.nanmin([lo for lo, _ in example_trace_extents]))
        y_max_shared = float(np.nanmax([hi for _, hi in example_trace_extents]))
        y_min_shared = min(y_min_shared, 0.0)
        y_max_shared = max(y_max_shared, 1.0)
        if not np.isfinite(y_min_shared) or not np.isfinite(y_max_shared):
            y_min_shared, y_max_shared = 0.0, 1.0
        if y_max_shared <= y_min_shared:
            y_max_shared = y_min_shared + 1.0
        y_pad = 0.05 * (y_max_shared - y_min_shared)
        y_lim_shared = (y_min_shared - y_pad, y_max_shared + y_pad)
        for ax_example in example_axes:
            ax_example.set_ylim(y_lim_shared)
        if first_example_ax in example_axes:
            _add_example_y_scale_bar(first_example_ax, 0.0, 1.0)

    if bool(trajectory_colorbar) and first_example_ax is not None:
        cmap_obj = plt.get_cmap(str(kwargs.get("trajectory_color_cmap", "vanimo")))
        norm = mcolors.Normalize(
            vmin=float(kwargs.get("trajectory_color_vmin", -1.0)),
            vmax=float(kwargs.get("trajectory_color_vmax", 1.0)),
        )
        cax_pos = first_example_ax.get_position()
        cax = fig.add_axes([cax_pos.x1 + 0.01, cax_pos.y0, 0.012, cax_pos.height])
        sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_ticks(list(trajectory_colorbar_ticks))
        cbar.ax.tick_params(labelsize=5)
        if trajectory_colorbar_label:
            cbar.set_label(str(trajectory_colorbar_label), fontsize=6, fontname="Arial")

    if show:
        plt.show()

    if return_pf_centered:
        if want_spike_rate_means:
            out = (
                fig,
                axes,
                valid_epochs,
                theta_mean,
                slow_mean,
                rate_mean,
                ss_rate_mean,
                cs_rate_mean,
                ss_pct_mean,
                cs_pct_mean,
            )
            if return_trial_firing_rates:
                return (*out, trial_data)
            return out
        out = (fig, axes, valid_epochs, theta_mean, slow_mean, rate_mean)
        if return_trial_firing_rates:
            return (*out, trial_data)
        return out
    out = (fig, axes, valid_epochs)
    if return_trial_firing_rates:
        return (*out, trial_data)
    return out


def plot_place_field_traversal_trials_with_cb_example_by_direction(
    trace,
    theta_vm,
    slow_vm,
    speed,
    x_neural,
    y_neural,
    traversal_epochs,
    place_field_mask,
    pf_bins,
    frame_rate,
    complex_bursts_dicts,
    cell_idx,
    burst_pre_ms=100,
    burst_post_ms=100,
    padding_sec=2.0,
    zscore_traces=False,
    show=True,
    return_pf_centered=False,
    traversal_types=None,
    return_trial_rates=False,
    return_pf_centered_sem=False,
    bad_timepoints=None,
    **kwargs,
):
    bad_timepoints = kwargs.pop("bad_timepoints", bad_timepoints)
    def _get_spike_indices(spike_source):
        if spike_source is None:
            return None
        if (
            cell_idx is not None
            and isinstance(spike_source, (list, tuple))
            and len(spike_source) > 0
            and isinstance(spike_source[0], (list, np.ndarray))
        ):
            return np.asarray(spike_source[cell_idx], dtype=int)
        return np.asarray(spike_source, dtype=int)

    def _filter_epochs(clockwise, counterclockwise):
        epochs_to_plot = traversal_epochs
        if clockwise ^ counterclockwise:
            if traversal_types is None or len(traversal_types) != len(traversal_epochs):
                print("Traversal types not provided or length mismatch; plotting all traversals.")
            else:
                target = "cw" if clockwise else "ccw"
                epochs_to_plot = [
                    epoch
                    for epoch, traversal_type in zip(traversal_epochs, traversal_types)
                    if traversal_type == target
                ]
        if session_indices is not None:
            if session_start_frames is None:
                print("session_start_frames not provided; plotting all sessions.")
            else:
                frames = np.asarray(session_start_frames, dtype=int)
                frames = frames[np.argsort(frames)]
                indices = session_indices
                if np.isscalar(indices):
                    indices = [int(indices)]
                indices = [int(i) for i in indices]
                epochs_filtered = []
                for epoch_start, epoch_end in epochs_to_plot:
                    session_idx = np.searchsorted(frames, epoch_start, side="right") - 1
                    if session_idx in indices:
                        epochs_filtered.append((epoch_start, epoch_end))
                epochs_to_plot = epochs_filtered

        valid = []
        for epoch_start, epoch_end in epochs_to_plot:
            start_idx = epoch_start - padding_frames
            end_idx = epoch_end + padding_frames
            if start_idx < 0 or end_idx >= trace_len:
                continue
            if bad_mask is not None and np.any(bad_mask[start_idx:end_idx]):
                continue
            valid.append((epoch_start, epoch_end))
        return valid

    def _build_axes(gs, n_rows):
        axes = np.empty((n_rows, 2), dtype=object)
        burst_axes = []
        for row_idx in range(n_rows):
            axes[row_idx, 0] = fig.add_subplot(gs[row_idx, 0])
            axes[row_idx, 1] = fig.add_subplot(gs[row_idx, 1])
            burst_axes.append(fig.add_subplot(gs[row_idx, 2]))
        axes_full = np.empty((n_rows, 3), dtype=object)
        axes_full[:, :2] = axes
        axes_full[:, 2] = burst_axes
        return axes, burst_axes, axes_full

    def _plot_cb_examples(axes, burst_axes, valid_epochs):
        if complex_bursts_dicts is None or cell_idx is None:
            return

        if isinstance(complex_bursts_dicts, (list, tuple)):
            if cell_idx >= len(complex_bursts_dicts):
                return
            bursts = complex_bursts_dicts[cell_idx]
        else:
            bursts = complex_bursts_dicts

        trace_arr = np.asarray(trace, dtype=float)
        trace_z = _safe_zscore(trace_arr, zscore_traces)
        starts = np.asarray(bursts.get("starts", []), dtype=int)
        ends = np.asarray(bursts.get("ends", []), dtype=int)
        amplitudes = np.asarray(bursts.get("amplitudes", []), dtype=float)

        if starts.size > 0 and ends.size > 0:
            if amplitudes.size != starts.size:
                amp_fallback = np.full(starts.shape, np.nan, dtype=float)
                for i, start_idx in enumerate(starts):
                    if 0 <= start_idx < len(trace_arr):
                        if start_idx > 0:
                            base = np.nanmin(trace_arr[max(0, start_idx - 3) : start_idx])
                        else:
                            base = trace_arr[start_idx]
                        if not np.isfinite(base):
                            base = trace_arr[start_idx]
                        amp_fallback[i] = trace_arr[start_idx] - base
                amplitudes = amp_fallback

        pre_frames = int(round(burst_pre_ms / 1000 * frame_rate))
        post_frames = int(round(burst_post_ms / 1000 * frame_rate))

        def _add_time_scale_bar(ax, bar_ms, label):
            x_min, x_max = ax.get_xlim()
            y_min, y_max = ax.get_ylim()
            if not (
                np.isfinite(x_min)
                and np.isfinite(x_max)
                and np.isfinite(y_min)
                and np.isfinite(y_max)
            ):
                return
            if x_max <= x_min or y_max <= y_min:
                return
            x_range = x_max - x_min
            if bar_ms >= x_range:
                return
            y_range = y_max - y_min if y_max != y_min else 1.0
            x0 = x_min + 0.05 * x_range
            x1 = x0 + bar_ms
            right_limit = x_max - 0.05 * x_range
            if x1 > right_limit:
                x1 = right_limit
                x0 = x1 - bar_ms
            y_bar = y_min + 0.05 * y_range
            ax.plot(
                [x0, x1],
                [y_bar, y_bar],
                color="black",
                linewidth=1.0,
                solid_capstyle="butt",
            )
            ax.text(
                (x0 + x1) / 2,
                y_bar - 0.05 * y_range,
                label,
                fontsize=6,
                fontname="Arial",
                ha="center",
                va="top",
            )

        for trial_idx, (epoch_start, epoch_end) in enumerate(valid_epochs):
            if trial_idx >= len(burst_axes):
                break
            ax_trace = axes[trial_idx, 0]
            ax_example = burst_axes[trial_idx]

            chosen_idx = None
            if starts.size > 0 and ends.size > 0:
                in_traversal = np.where((starts >= epoch_start) & (starts <= epoch_end))[0]
                if in_traversal.size > 0:
                    cand_amp = amplitudes[in_traversal] if amplitudes.size == starts.size else None
                    if cand_amp is not None and np.any(np.isfinite(cand_amp)):
                        chosen_idx = in_traversal[np.nanargmax(cand_amp)]
                    else:
                        chosen_idx = in_traversal[0]

            if chosen_idx is None:
                ax_example.text(
                    0.5,
                    0.5,
                    "No burst",
                    transform=ax_example.transAxes,
                    ha="center",
                    va="center",
                    fontsize=6,
                    fontname="Arial",
                )
                ax_example.axis("off")
                continue

            burst_start = int(starts[chosen_idx])
            burst_end = int(ends[chosen_idx]) if chosen_idx < len(ends) else burst_start
            if burst_end < burst_start:
                burst_end = burst_start

            t_start = (burst_start - epoch_start) / frame_rate
            t_end = (burst_end - epoch_start) / frame_rate
            if t_end <= t_start:
                t_end = t_start + (1.0 / frame_rate)
            start_idx = epoch_start - padding_frames
            end_idx = epoch_end + padding_frames
            if start_idx < 0:
                start_idx = 0
            if end_idx > len(trace_z):
                end_idx = len(trace_z)
            trace_window = trace_z[start_idx:end_idx]
            trace_min = np.nanmin(trace_window) if trace_window.size > 0 else np.nan
            trace_max = np.nanmax(trace_window) if trace_window.size > 0 else np.nan
            if not np.isfinite(trace_min) or not np.isfinite(trace_max) or trace_max == trace_min:
                trace_min, trace_max = ax_trace.get_ylim()
            rect = Rectangle(
                (t_start, trace_min),
                t_end - t_start,
                trace_max - trace_min,
                fill=False,
                edgecolor="gray",
                linestyle="--",
                linewidth=0.8,
                zorder=4,
            )
            ax_trace.add_patch(rect)

            win_start = max(burst_start - pre_frames, 0)
            win_end = min(burst_end + post_frames + 1, len(trace_arr))
            if win_end <= win_start:
                ax_example.text(
                    0.5,
                    0.5,
                    "No burst",
                    transform=ax_example.transAxes,
                    ha="center",
                    va="center",
                    fontsize=6,
                    fontname="Arial",
                )
                ax_example.axis("off")
                continue

            t_ms = (np.arange(win_start, win_end) - burst_start) / frame_rate * 1000
            ax_example.plot(t_ms, trace_arr[win_start:win_end], color="black", linewidth=0.6)
            ax_example.set_xlim([t_ms[0], t_ms[-1]])
            _add_time_scale_bar(ax_example, 50.0, "50 ms")
            ax_example.set_xticks([])
            ax_example.set_yticks([])
            ax_example.spines["top"].set_visible(False)
            ax_example.spines["right"].set_visible(False)
            ax_example.spines["left"].set_visible(False)
            ax_example.spines["bottom"].set_visible(False)

    def _compute_trial_rates(valid_epochs, ss_indices, cb_starts):
        entry_times = []
        ss_rates = []
        cb_rates = []
        for epoch_start, epoch_end in valid_epochs:
            duration = (epoch_end - epoch_start) / frame_rate
            if duration <= 0:
                entry_times.append(epoch_start / frame_rate)
                ss_rates.append(np.nan)
                cb_rates.append(np.nan)
                continue
            entry_times.append(epoch_start / frame_rate)
            if ss_indices is None:
                ss_rates.append(np.nan)
            else:
                ss_count = np.sum((ss_indices >= epoch_start) & (ss_indices < epoch_end))
                ss_rates.append(ss_count / duration)
            if cb_starts is None:
                cb_rates.append(np.nan)
            else:
                cb_count = np.sum((cb_starts >= epoch_start) & (cb_starts < epoch_end))
                cb_rates.append(cb_count / duration)
        return {
            "entry_time": np.asarray(entry_times, dtype=float),
            "ss_rate": np.asarray(ss_rates, dtype=float),
            "cb_rate": np.asarray(cb_rates, dtype=float),
        }

    padding_frames = int(padding_sec * frame_rate)
    trace_len = len(trace)
    bad_mask = None
    if bad_timepoints is not None:
        bad_mask = np.asarray(bad_timepoints)
        if bad_mask.dtype != bool:
            if bad_mask.ndim == 1 and bad_mask.size == trace_len:
                bad_mask = bad_mask.astype(bool)
            else:
                bad_mask = np.asarray(bad_mask, dtype=int)
                bad_mask = bad_mask[(bad_mask >= 0) & (bad_mask < trace_len)]
                bad_bool = np.zeros(trace_len, dtype=bool)
                bad_bool[bad_mask] = True
                bad_mask = bad_bool
        if bad_mask.shape[0] != trace_len:
            raise ValueError("bad_timepoints must match trace length or be index list.")
    session_start_frames = kwargs.get("session_start_frames")
    session_indices = kwargs.get("session_indices")
    pf_peak_xy = kwargs.get("pf_peak_xy")
    plot_pf_centered_average = kwargs.get("plot_pf_centered_average", True)
    plot_pf_centered_rate = kwargs.get("plot_pf_centered_rate", True)
    all_spikes = kwargs.get("all_spikes")
    kwargs = dict(kwargs)
    kwargs.pop("traversal_types", None)
    kwargs.pop("clockwise", None)
    kwargs.pop("counterclockwise", None)

    include_pf_rate = False
    all_spike_indices = _get_spike_indices(all_spikes)
    if (
        plot_pf_centered_average
        and pf_peak_xy is not None
        and plot_pf_centered_rate
        and all_spike_indices is not None
        and len(all_spike_indices) > 0
    ):
        include_pf_rate = True

    extra_rows = 3 if include_pf_rate else (2 if plot_pf_centered_average and pf_peak_xy is not None else 0)
    pf_heights = [1.1, 1.4, 1.4] if include_pf_rate else [1.4, 1.4]

    valid_cw = _filter_epochs(clockwise=True, counterclockwise=False)
    valid_ccw = _filter_epochs(clockwise=False, counterclockwise=True)

    n_trials_cw = len(valid_cw)
    n_trials_ccw = len(valid_ccw)
    n_rows_cw = n_trials_cw + 1 + extra_rows
    n_rows_ccw = n_trials_ccw + 1 + extra_rows
    height_ratios_cw = [1] * n_trials_cw + [0.15] + (pf_heights if extra_rows else [])
    height_ratios_ccw = [1] * n_trials_ccw + [0.15] + (pf_heights if extra_rows else [])

    n_trials_max = max(n_trials_cw, n_trials_ccw, 1)
    fig_height = 0.6 * n_trials_max + 0.3 + (1.4 * extra_rows)
    fig = plt.figure(figsize=(12, fig_height))
    outer = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.06)

    gs_cw = outer[0].subgridspec(
        max(n_rows_cw, 1),
        3,
        width_ratios=[5, 1, 1],
        hspace=0,
        wspace=0.01,
        height_ratios=height_ratios_cw or [1],
    )
    gs_ccw = outer[1].subgridspec(
        max(n_rows_ccw, 1),
        3,
        width_ratios=[5, 1, 1],
        hspace=0,
        wspace=0.01,
        height_ratios=height_ratios_ccw or [1],
    )

    axes_cw, burst_axes_cw, axes_cw_full = _build_axes(gs_cw, max(n_rows_cw, 1))
    axes_ccw, burst_axes_ccw, axes_ccw_full = _build_axes(gs_ccw, max(n_rows_ccw, 1))

    theta_mean_cw = slow_mean_cw = rate_mean_cw = None
    theta_mean_ccw = slow_mean_ccw = rate_mean_ccw = None
    trial_rates = {"cw": None, "ccw": None}

    theta_sem_cw = slow_sem_cw = rate_sem_cw = None
    theta_sem_ccw = slow_sem_ccw = rate_sem_ccw = None

    if n_trials_cw > 0:
        if return_pf_centered_sem:
            (
                _,
                _,
                valid_cw,
                theta_mean_cw,
                slow_mean_cw,
                rate_mean_cw,
                theta_sem_cw,
                slow_sem_cw,
                rate_sem_cw,
            ) = plot_place_field_traversal_trials(
                trace,
                theta_vm,
                slow_vm,
                speed,
                x_neural,
                y_neural,
                traversal_epochs,
                place_field_mask,
                pf_bins,
                frame_rate,
                cell_idx=cell_idx,
                padding_sec=padding_sec,
                zscore_traces=zscore_traces,
                bad_timepoints=bad_timepoints,
                show=False,
                return_pf_centered=True,
                return_pf_centered_sem=True,
                axes=axes_cw,
                apply_layout=False,
                clockwise=True,
                counterclockwise=False,
                traversal_types=traversal_types,
                **kwargs,
            )
        else:
            _, _, valid_cw, theta_mean_cw, slow_mean_cw, rate_mean_cw = plot_place_field_traversal_trials(
            trace,
            theta_vm,
            slow_vm,
            speed,
            x_neural,
            y_neural,
            traversal_epochs,
            place_field_mask,
            pf_bins,
            frame_rate,
            cell_idx=cell_idx,
            padding_sec=padding_sec,
            zscore_traces=zscore_traces,
            bad_timepoints=bad_timepoints,
            show=False,
            return_pf_centered=True,
            axes=axes_cw,
            apply_layout=False,
            clockwise=True,
            counterclockwise=False,
            traversal_types=traversal_types,
                **kwargs,
            )
        _plot_cb_examples(axes_cw, burst_axes_cw, valid_cw)
    else:
        for ax in axes_cw_full.ravel():
            ax.axis("off")
        axes_cw_full[0, 0].text(
            0.5,
            0.5,
            "No CW traversals",
            transform=axes_cw_full[0, 0].transAxes,
            ha="center",
            va="center",
            fontsize=6,
            fontname="Arial",
        )

    if n_trials_ccw > 0:
        if return_pf_centered_sem:
            (
                _,
                _,
                valid_ccw,
                theta_mean_ccw,
                slow_mean_ccw,
                rate_mean_ccw,
                theta_sem_ccw,
                slow_sem_ccw,
                rate_sem_ccw,
            ) = plot_place_field_traversal_trials(
                trace,
                theta_vm,
                slow_vm,
                speed,
                x_neural,
                y_neural,
                traversal_epochs,
                place_field_mask,
                pf_bins,
                frame_rate,
                cell_idx=cell_idx,
                padding_sec=padding_sec,
                zscore_traces=zscore_traces,
                bad_timepoints=bad_timepoints,
                show=False,
                return_pf_centered=True,
                return_pf_centered_sem=True,
                axes=axes_ccw,
                apply_layout=False,
                clockwise=False,
                counterclockwise=True,
                traversal_types=traversal_types,
                **kwargs,
            )
        else:
            _, _, valid_ccw, theta_mean_ccw, slow_mean_ccw, rate_mean_ccw = plot_place_field_traversal_trials(
            trace,
            theta_vm,
            slow_vm,
            speed,
            x_neural,
            y_neural,
            traversal_epochs,
            place_field_mask,
            pf_bins,
            frame_rate,
            cell_idx=cell_idx,
            padding_sec=padding_sec,
            zscore_traces=zscore_traces,
            bad_timepoints=bad_timepoints,
            show=False,
            return_pf_centered=True,
            axes=axes_ccw,
            apply_layout=False,
            clockwise=False,
            counterclockwise=True,
            traversal_types=traversal_types,
                **kwargs,
            )
        _plot_cb_examples(axes_ccw, burst_axes_ccw, valid_ccw)
    else:
        for ax in axes_ccw_full.ravel():
            ax.axis("off")
        axes_ccw_full[0, 0].text(
            0.5,
            0.5,
            "No CCW traversals",
            transform=axes_ccw_full[0, 0].transAxes,
            ha="center",
            va="center",
            fontsize=6,
            fontname="Arial",
        )

    for ax_list, n_trials in ((burst_axes_cw, n_trials_cw), (burst_axes_ccw, n_trials_ccw)):
        for idx in range(len(ax_list)):
            if idx >= n_trials:
                ax_list[idx].axis("off")

    if return_trial_rates:
        ss_indices = _get_spike_indices(kwargs.get("refined_SS"))
        cb_starts = None
        if complex_bursts_dicts is not None:
            if isinstance(complex_bursts_dicts, (list, tuple)):
                if cell_idx is not None and cell_idx < len(complex_bursts_dicts):
                    cb_starts = np.asarray(
                        complex_bursts_dicts[cell_idx].get("starts", []), dtype=int
                    )
            else:
                cb_starts = np.asarray(complex_bursts_dicts.get("starts", []), dtype=int)
        trial_rates["cw"] = _compute_trial_rates(valid_cw, ss_indices, cb_starts)
        trial_rates["ccw"] = _compute_trial_rates(valid_ccw, ss_indices, cb_starts)

    if show:
        plt.show()

    valid_epochs = {"cw": valid_cw, "ccw": valid_ccw}
    theta_mean = {"cw": theta_mean_cw, "ccw": theta_mean_ccw}
    slow_mean = {"cw": slow_mean_cw, "ccw": slow_mean_ccw}
    rate_mean = {"cw": rate_mean_cw, "ccw": rate_mean_ccw}
    theta_sem = {"cw": theta_sem_cw, "ccw": theta_sem_ccw}
    slow_sem = {"cw": slow_sem_cw, "ccw": slow_sem_ccw}
    rate_sem = {"cw": rate_sem_cw, "ccw": rate_sem_ccw}
    axes_out = {"cw": axes_cw_full, "ccw": axes_ccw_full}

    if return_pf_centered and return_trial_rates and return_pf_centered_sem:
        return (
            fig,
            axes_out,
            valid_epochs,
            theta_mean,
            slow_mean,
            rate_mean,
            theta_sem,
            slow_sem,
            rate_sem,
            trial_rates,
        )
    if return_pf_centered and return_pf_centered_sem:
        return fig, axes_out, valid_epochs, theta_mean, slow_mean, rate_mean, theta_sem, slow_sem, rate_sem
    if return_pf_centered and return_trial_rates:
        return fig, axes_out, valid_epochs, theta_mean, slow_mean, rate_mean, trial_rates
    if return_pf_centered:
        return fig, axes_out, valid_epochs, theta_mean, slow_mean, rate_mean
    if return_trial_rates:
        return fig, axes_out, valid_epochs, trial_rates
    return fig, axes_out, valid_epochs


def plot_place_field_traversal_trials_with_average(
    trace,
    theta_vm,
    slow_vm,
    speed,
    x_neural,
    y_neural,
    traversal_epochs,
    place_field_mask,
    pf_bins,
    frame_rate,
    cell_idx=None,
    padding_sec=2.0,
    zscore_traces=False,
    speed_smooth_kernel=51,
    resting_speed_threshold=5,
    rest_merge_gap_s=0.5,
    traversal_types=None,
    clockwise=False,
    counterclockwise=False,
    session_start_frames=None,
    color_by_session=False,
    session_indices=None,
    show_rest_patches=True,
    theta_freqs=None,
    slow_freqs=None,
    **kwargs,
):
    def _add_vertical_scale_bar(ax, bar_height, label, x_pos=-0.02, align="right"):
        y_min, y_max = ax.get_ylim()
        if not (np.isfinite(y_min) and np.isfinite(y_max)):
            return
        if y_max <= y_min:
            return
        y0 = y_min + 0.7 * (y_max - y_min)
        if y0 + bar_height > y_max:
            y0 = y_max - bar_height
        ax.plot(
            [x_pos, x_pos],
            [y0, y0 + bar_height],
            transform=ax.get_yaxis_transform(),
            color="black",
            linewidth=1.0,
            clip_on=False,
        )
        ax.text(
            x_pos - 0.005 if align == "right" else x_pos + 0.005,
            y0 + bar_height / 2,
            label,
            transform=ax.get_yaxis_transform(),
            ha=align,
            va="center",
            fontsize=6,
            fontname="Arial",
            color="black",
        )

    trace_z = _safe_zscore(trace, zscore_traces)
    theta_z = _safe_zscore(theta_vm, zscore_traces)
    slow_z = _safe_zscore(slow_vm, zscore_traces)
    pf_occupancy_mask = kwargs.pop("pf_occupancy_mask", None)
    pf_mask_shape_ref = np.asarray(place_field_mask, dtype=bool).shape
    if pf_occupancy_mask is None:
        pf_occupancy_mask = np.asarray(place_field_mask, dtype=bool)
    else:
        pf_occupancy_mask = np.asarray(pf_occupancy_mask, dtype=bool)
        if pf_occupancy_mask.shape != pf_mask_shape_ref:
            pf_occupancy_mask = np.asarray(place_field_mask, dtype=bool)

    if speed_smooth_kernel % 2 == 0:
        speed_smooth_kernel += 1
    speed_smooth = median_filter(speed, size=speed_smooth_kernel)

    padding_frames = int(padding_sec * frame_rate)
    epochs_to_plot = traversal_epochs
    if clockwise ^ counterclockwise:
        if traversal_types is None or len(traversal_types) != len(traversal_epochs):
            print("Traversal types not provided or length mismatch; plotting all traversals.")
        else:
            target = "cw" if clockwise else "ccw"
            epochs_to_plot = [
                epoch
                for epoch, traversal_type in zip(traversal_epochs, traversal_types)
                if traversal_type == target
            ]
    if session_indices is not None:
        if session_start_frames is None:
            print("session_start_frames not provided; plotting all sessions.")
        else:
            session_start_frames = np.asarray(session_start_frames, dtype=int)
            session_start_frames = session_start_frames[np.argsort(session_start_frames)]
            if np.isscalar(session_indices):
                session_indices = [int(session_indices)]
            session_indices = [int(i) for i in session_indices]
            epochs_filtered = []
            for epoch_start, epoch_end in epochs_to_plot:
                session_idx = np.searchsorted(session_start_frames, epoch_start, side="right") - 1
                if session_idx in session_indices:
                    epochs_filtered.append((epoch_start, epoch_end))
            epochs_to_plot = epochs_filtered

    valid_epochs = [
        (s, e)
        for s, e in epochs_to_plot
        if s - padding_frames >= 0 and e + padding_frames < len(trace_z)
    ]

    n_trials = len(valid_epochs)
    if n_trials == 0:
        print("No valid traversal epochs with sufficient padding")
        return None, None, []

    trace_mins, trace_maxs = [], []
    theta_mins, theta_maxs = [], []
    slow_mins, slow_maxs = [], []
    for epoch_start, epoch_end in valid_epochs:
        start_idx = epoch_start - padding_frames
        end_idx = epoch_end + padding_frames
        trace_mins.append(np.nanmin(trace_z[start_idx:end_idx]))
        trace_maxs.append(np.nanmax(trace_z[start_idx:end_idx]))
        theta_mins.append(np.nanmin(theta_z[start_idx:end_idx]))
        theta_maxs.append(np.nanmax(theta_z[start_idx:end_idx]))
        slow_mins.append(np.nanmin(slow_z[start_idx:end_idx]))
        slow_maxs.append(np.nanmax(slow_z[start_idx:end_idx]))

    trace_min_global = min(trace_mins)
    theta_max_global = max(theta_maxs)
    theta_min_global = min(theta_mins)
    slow_max_global = max(slow_maxs)
    slow_min_global = min(slow_mins)

    trace_range = max(trace_maxs) - min(trace_mins)
    theta_range = theta_max_global - theta_min_global
    slow_range = slow_max_global - slow_min_global

    pad_trace_slow = 0.005 * (trace_range if np.isfinite(trace_range) and trace_range > 0 else max(1.0, slow_range))
    pad_slow_theta = 0.05 * max(slow_range if np.isfinite(slow_range) and slow_range > 0 else 1.0, theta_range if np.isfinite(theta_range) and theta_range > 0 else 1.0)

    offset1 = slow_max_global - trace_min_global + pad_trace_slow
    offset2 = theta_max_global - slow_min_global + pad_slow_theta

    max_epoch_duration = max((e - s) for s, e in valid_epochs)
    global_x_min = -padding_sec
    global_x_max = max_epoch_duration / frame_rate + padding_sec

    height_ratios = [1] * n_trials + [0.15, 0.4, 0.4]
    fig, axes = plt.subplots(
        n_trials + 3,
        2,
        figsize=(6, 0.6 * n_trials + 1.3),
        gridspec_kw={
            "width_ratios": [5, 1],
            "hspace": 0,
            "wspace": 0.01,
            "height_ratios": height_ratios,
        },
    )

    merge_gap_frames = int(rest_merge_gap_s * frame_rate)
    session_colors = None
    epoch_sessions = None
    if color_by_session and session_start_frames is not None:
        session_start_frames = np.asarray(session_start_frames, dtype=int)
        session_start_frames = session_start_frames[np.argsort(session_start_frames)]
        num_sessions = len(session_start_frames)
        if num_sessions > 0:
            base_colors = [
                "#000000",
                "#e41a1c",
                "#377eb8",
                "#4daf4a",
                "#984ea3",
                "#ff7f00",
                "#ffff33",
                "#a65628",
                "#f781bf",
                "#999999",
            ]
            if num_sessions <= len(base_colors):
                session_colors = base_colors[:num_sessions]
            else:
                extra = plt.cm.hsv(
                    np.linspace(0, 1, num_sessions - len(base_colors), endpoint=False)
                )
                session_colors = base_colors + [tuple(c) for c in extra]
            epoch_sessions = [
                np.searchsorted(session_start_frames, s, side="right") - 1
                for s, _ in valid_epochs
            ]

    for trial_idx, (epoch_start, epoch_end) in enumerate(valid_epochs):
        ax_trace = axes[trial_idx, 0]
        ax_traj = axes[trial_idx, 1]

        start_idx = epoch_start - padding_frames
        end_idx = epoch_end + padding_frames
        time_rel = (np.arange(start_idx, end_idx) - epoch_start) / frame_rate

        trace_window = trace_z[start_idx:end_idx]
        theta_window = theta_z[start_idx:end_idx]
        slow_window = slow_z[start_idx:end_idx]

        epoch_duration = (epoch_end - epoch_start) / frame_rate
        ax_trace.axvspan(0, epoch_duration, alpha=0.2, color="green", zorder=0)

        x_traj_full = x_neural[start_idx:end_idx]
        y_traj_full = y_neural[start_idx:end_idx]
        in_pf_window = _positions_in_place_field(
            x_traj_full, y_traj_full, pf_bins, pf_occupancy_mask
        )
        speed_window_raw = speed[start_idx:end_idx]
        resting_in_pf = in_pf_window & (speed_window_raw < resting_speed_threshold)

        pre_frames = padding_frames
        post_start_idx = epoch_end - start_idx

        if show_rest_patches:
            pre_resting = resting_in_pf[:pre_frames]
            if np.any(pre_resting):
                diff = np.diff(np.concatenate([[0], pre_resting.astype(int), [0]]))
                starts = np.where(diff == 1)[0]
                ends = np.where(diff == -1)[0]
                starts, ends = _merge_segments(starts, ends, merge_gap_frames)
                for s, e in zip(starts, ends):
                    t_start = time_rel[s]
                    t_end = time_rel[min(e, pre_frames - 1)]
                    ax_trace.axvspan(t_start, t_end, alpha=0.2, color="gold", zorder=0)

            post_resting = resting_in_pf[post_start_idx:]
            if np.any(post_resting):
                diff = np.diff(np.concatenate([[0], post_resting.astype(int), [0]]))
                starts = np.where(diff == 1)[0]
                ends = np.where(diff == -1)[0]
                starts, ends = _merge_segments(starts, ends, merge_gap_frames)
                for s, e in zip(starts, ends):
                    t_start = time_rel[post_start_idx + s]
                    t_end = time_rel[min(post_start_idx + e, len(time_rel) - 1)]
                    ax_trace.axvspan(t_start, t_end, alpha=0.2, color="gold", zorder=0)

        ax_trace.plot(
            time_rel,
            trace_window,
            color="black",
            linewidth=0.5,
            label="Trace",
            zorder=2,
        )
        slow_trace_gap = 0.0
        trace_min_local = np.nanmin(trace_window) if np.any(np.isfinite(trace_window)) else 0.0
        slow_max_local = np.nanmax(slow_window) if np.any(np.isfinite(slow_window)) else 0.0
        slow_min_local = np.nanmin(slow_window) if np.any(np.isfinite(slow_window)) else 0.0
        theta_max_local = np.nanmax(theta_window) if np.any(np.isfinite(theta_window)) else 0.0
        offset1 = max(0.0, float(slow_max_local - trace_min_local + slow_trace_gap))
        offset2 = max(0.0, float(theta_max_local - slow_min_local + pad_slow_theta))
        theta_label = "Theta (4-8 Hz)"
        if theta_freqs is not None and len(theta_freqs) == 2:
            theta_label = f"Theta ({theta_freqs[0]}-{theta_freqs[1]} Hz)"
        slow_label = "Slow (<=2 Hz)"
        if slow_freqs is not None:
            if np.isscalar(slow_freqs):
                slow_label = f"Slow (<= {slow_freqs} Hz)"
            elif len(slow_freqs) == 2:
                slow_label = f"Slow ({slow_freqs[0]}-{slow_freqs[1]} Hz)"
        ax_trace.plot(
            time_rel,
            slow_window - offset1,
            color="red",
            linewidth=0.5,
            label=slow_label,
            zorder=2,
        )
        ax_trace.plot(
            time_rel,
            theta_window - offset1 - offset2,
            color="blue",
            linewidth=0.5,
            label=theta_label,
            zorder=2,
        )

        x_data_max = time_rel[-1]
        tick_len = 0.08
        ax_trace.plot(
            [x_data_max - tick_len, x_data_max],
            [0, 0],
            color="black",
            linewidth=0.5,
            alpha=0.5,
        )
        ax_trace.plot(
            [x_data_max - tick_len, x_data_max],
            [-offset1, -offset1],
            color="red",
            linewidth=0.5,
            alpha=0.5,
        )
        ax_trace.plot(
            [x_data_max - tick_len, x_data_max],
            [-offset1 - offset2, -offset1 - offset2],
            color="blue",
            linewidth=0.5,
            alpha=0.5,
        )

        trial_color = "black"
        if session_colors is not None and epoch_sessions is not None:
            session_idx = epoch_sessions[trial_idx]
            if 0 <= session_idx < len(session_colors):
                trial_color = session_colors[session_idx]
        # Build trial label with direction type
        trial_type = valid_traversal_types[trial_idx] if trial_idx < len(valid_traversal_types) else "?"
        type_key = str(trial_type).strip().lower()
        type_label = type_key.upper() if type_key in ("cw", "ccw", "p", "np") else ""
        trial_label = f"{trial_idx + 1}" + (
            f" {type_label}" if type_label and not bool(trajectory_direction_label_top) else ""
        )
        ax_trace.text(
            time_rel[0] - 0.2,
            trace_window[0] + 0.1,
            trial_label,
            fontsize=6,
            ha="left",
            va="bottom",
            fontweight="bold",
            color=trial_color,
        )
        ax_trace.text(
            -0.6,
            0.3,
            f"{epoch_start / frame_rate:.1f}s",
            transform=ax_trace.get_xaxis_transform(),
            ha="left",
            va="bottom",
            fontsize=6,
            fontname="Arial",
            color="black",
        )

        ax_trace.spines["top"].set_visible(False)
        ax_trace.spines["right"].set_visible(False)
        ax_trace.spines["left"].set_visible(False)
        ax_trace.spines["bottom"].set_visible(False)
        ax_trace.set_yticks([])
        ax_trace.set_xticks([])
        ax_trace.set_xlim([global_x_min, global_x_max])

        if trial_idx == 0:
            ax_trace.legend(
                loc="lower left",
                bbox_to_anchor=(0.0, 1.02),
                fontsize=5,
                frameon=False,
                ncol=3,
                borderaxespad=0,
            )
            if cell_idx is not None:
                direction_label = "all"
                if clockwise ^ counterclockwise:
                    direction_label = "cw" if clockwise else "ccw"
                ax_trace.set_title(
                    f"Cell {cell_idx} - Place Field Traversals ({direction_label}, n={n_trials})",
                    fontsize=6,
                    pad=10,
                )
        x_traj = x_neural[start_idx:end_idx]
        y_traj = y_neural[start_idx:end_idx]
        post_start = epoch_end - start_idx

        ax_traj.plot(
            x_traj[:pre_frames], y_traj[:pre_frames], color="#87CEEB", linewidth=0.6, alpha=0.7
        )

        x_epoch = x_neural[epoch_start:epoch_end]
        y_epoch = y_neural[epoch_start:epoch_end]
        ax_traj.plot(x_epoch, y_epoch, color="green", linewidth=1.2)

        ax_traj.plot(
            x_traj[post_start:],
            y_traj[post_start:],
            color="#FFB6C1",
            linewidth=0.6,
            alpha=0.7,
        )

        if len(x_epoch) > 0:
            ax_traj.scatter(
                x_epoch[0],
                y_epoch[0],
                color="blue",
                s=12,
                zorder=5,
                marker="o",
                label="Start",
            )
            ax_traj.scatter(
                x_epoch[-1],
                y_epoch[-1],
                color="red",
                s=12,
                zorder=5,
                marker="s",
                label="End",
            )

        if trial_idx == 0:
            ax_traj.legend(
                loc="lower left",
                bbox_to_anchor=(0.0, 1.02),
                fontsize=5,
                frameon=False,
                handletextpad=0.1,
                borderpad=0.0,
                borderaxespad=0,
            )

        if np.any(place_field_mask):
            padded_mask = np.zeros(
                (place_field_mask.shape[0] + 2, place_field_mask.shape[1] + 2), dtype=bool
            )
            padded_mask[1:-1, 1:-1] = place_field_mask
            extent = (pf_bins[0][0], pf_bins[0][-1], pf_bins[1][0], pf_bins[1][-1])
            bin_x = (extent[1] - extent[0]) / place_field_mask.shape[0]
            bin_y = (extent[3] - extent[2]) / place_field_mask.shape[1]
            padded_extent = (
                extent[0] - bin_x,
                extent[1] + bin_x,
                extent[2] - bin_y,
                extent[3] + bin_y,
            )
            ax_traj.contour(
                padded_mask.T,
                levels=[0.5],
                colors="magenta",
                linewidths=0.8,
                extent=padded_extent,
                origin="lower",
            )

        if bool(trajectory_direction_label_top) and type_label:
            arena_x0, arena_x1 = float(pf_bins[0][0]), float(pf_bins[0][-1])
            arena_y0, arena_y1 = float(pf_bins[1][0]), float(pf_bins[1][-1])
            ax_traj.text(
                0.5 * (arena_x0 + arena_x1),
                0.5 * (arena_y0 + arena_y1),
                type_label,
                ha="center",
                va="center",
                fontsize=5,
                fontname="Arial",
                fontweight="bold",
                color="black",
                zorder=20,
                bbox=dict(boxstyle="round,pad=0.08", facecolor="white", edgecolor="none", alpha=0.7),
            )

        ax_traj.set_xlim([pf_bins[0][0], pf_bins[0][-1]])
        ax_traj.set_ylim([pf_bins[1][0], pf_bins[1][-1]])
        ax_traj.set_aspect("equal")
        ax_traj.axis("off")

    ax_scale = axes[n_trials, 0]
    ax_scale_traj = axes[n_trials, 1]
    ax_scale.axis("off")
    ax_scale_traj.axis("off")

    avg_len = int(max_epoch_duration + 2 * padding_frames)
    time_avg = (np.arange(avg_len) - padding_frames) / frame_rate
    theta_stack = np.full((n_trials, avg_len), np.nan)
    slow_stack = np.full((n_trials, avg_len), np.nan)
    for i, (epoch_start, epoch_end) in enumerate(valid_epochs):
        window_start = epoch_start - padding_frames
        window_end = epoch_end + padding_frames
        theta_window = theta_z[window_start:window_end]
        slow_window = slow_z[window_start:window_end]
        window_len = min(len(theta_window), avg_len)
        if window_len <= 0:
            continue
        theta_stack[i, :window_len] = theta_window[:window_len]
        slow_stack[i, :window_len] = slow_window[:window_len]

    theta_mean = np.nanmean(theta_stack, axis=0)
    slow_mean = np.nanmean(slow_stack, axis=0)

    ax_avg_theta = axes[n_trials + 1, 0]
    ax_avg_slow = axes[n_trials + 2, 0]
    axes[n_trials + 1, 1].axis("off")
    axes[n_trials + 2, 1].axis("off")

    ax_avg_theta.plot(time_avg, theta_mean, color="blue", linewidth=0.5)
    ax_avg_slow.plot(time_avg, slow_mean, color="red", linewidth=0.5)

    ax_avg_theta.set_xlim([global_x_min, global_x_max])
    ax_avg_slow.set_xlim([global_x_min, global_x_max])
    ax_avg_theta.axis("off")
    ax_avg_slow.axis("off")
    _add_vertical_scale_bar(ax_avg_theta, 0.1, "0.1 spk")
    _add_vertical_scale_bar(ax_avg_slow, 0.1, "0.1 spk")

    scale_x_start = global_x_min + 0.2
    scale_x_end = scale_x_start + 1.0
    ax_avg_slow.plot(
        [scale_x_start, scale_x_end],
        [0.1, 0.1],
        transform=ax_avg_slow.get_xaxis_transform(),
        color="black",
        linewidth=1.0,
        solid_capstyle="butt",
        clip_on=False,
    )
    ax_avg_slow.text(
        (scale_x_start + scale_x_end) / 2,
        0.02,
        "1 s",
        transform=ax_avg_slow.get_xaxis_transform(),
        fontsize=6,
        ha="center",
        va="top",
    )
    _add_vertical_scale_bar(ax_scale, 0.6, "1 spk", x_pos=0.92, align="left")

    axes[1, 1].axis("off")
    axes[1, 3].axis("off")
    plt.tight_layout()
    plt.subplots_adjust(hspace=0, wspace=0.01, top=0.9)
    #plt.show()

    print(f"Plotted {n_trials} traversal trials")
    return fig, axes, valid_epochs, theta_mean, slow_mean


def plot_theta_slow_maps_by_state(
    x_neural,
    y_neural,
    speed,
    frame_rate,
    traces,
    analysis,
    cell_idx,
    theta_freqs=(4, 8),
    slow_freqs=2,
    speed_threshold=2,
    kernel_size=21,
    filter_type="boxcar",
    min_duration_s=0.5,
    merge_gap_s=1.0,
    smooth_sigma=2.0,
    bad_timepoints=None,
    save_dir=None,
    show=True,
):
    x_neural = np.asarray(x_neural)
    y_neural = np.asarray(y_neural)
    speed = np.asarray(speed, dtype=float)

    if analysis is None or not analysis.get("is_place_cell", False):
        return None, None, None

    def _get_bins_and_mask(analysis):
        place_field_mask = np.asarray(analysis.get("place_field_mask", []), dtype=bool)
        if place_field_mask.size == 0:
            return None, None
        bins = analysis.get("bins")
        if bins is None:
            params = analysis.get("params", {})
            width_real = params.get("width_real")
            height_real = params.get("height_real")
            bin_size = params.get("bin_size")
            if width_real is None or height_real is None or bin_size is None:
                return None, None
            bins = [
                np.arange(0, width_real + bin_size, bin_size),
                np.arange(0, height_real + bin_size, bin_size),
            ]
        bins = [np.asarray(b) for b in bins]
        return bins, place_field_mask

    def _apply_bad_mask(valid_frames, cell_idx):
        if bad_timepoints is None:
            return valid_frames
        bad_mask = bad_timepoints
        if isinstance(bad_timepoints, (list, tuple)):
            if len(bad_timepoints) > cell_idx:
                bad_mask = bad_timepoints[cell_idx]
        else:
            bad_mask = np.asarray(bad_timepoints)
            if bad_mask.ndim >= 2:
                if bad_mask.shape[0] == len(valid_frames) and bad_mask.shape[1] > cell_idx:
                    bad_mask = bad_mask[:, cell_idx]
                elif bad_mask.shape[1] == len(valid_frames) and bad_mask.shape[0] > cell_idx:
                    bad_mask = bad_mask[cell_idx]
                elif bad_mask.shape[0] > cell_idx:
                    bad_mask = bad_mask[cell_idx]
        bad_mask = np.asarray(bad_mask)
        if bad_mask.ndim > 1:
            bad_mask = np.squeeze(bad_mask)
        if bad_mask.dtype != bool:
            if bad_mask.ndim == 1 and bad_mask.size == len(valid_frames):
                bad_mask = bad_mask.astype(bool)
            else:
                bad_mask = np.asarray(bad_mask, dtype=int)
                bad_mask = bad_mask[(bad_mask >= 0) & (bad_mask < len(valid_frames))]
                bad_bool = np.zeros(len(valid_frames), dtype=bool)
                bad_bool[bad_mask] = True
                bad_mask = bad_bool
        if bad_mask.shape[0] != len(valid_frames):
            raise ValueError("bad_timepoints must match trace length or be index list.")
        return valid_frames & (~bad_mask)

    def _mean_map(x_vals, y_vals, values, bins):
        if x_vals.size == 0:
            return np.full((len(bins[0]) - 1, len(bins[1]) - 1), np.nan)
        counts, _, _ = np.histogram2d(x_vals, y_vals, bins=bins)
        sums, _, _ = np.histogram2d(x_vals, y_vals, bins=bins, weights=values)
        if smooth_sigma and smooth_sigma > 0:
            counts = gaussian_filter(counts, sigma=smooth_sigma, mode="constant")
            sums = gaussian_filter(sums, sigma=smooth_sigma, mode="constant")
        with np.errstate(divide="ignore", invalid="ignore"):
            mean_map = sums / counts
        mean_map[counts == 0] = np.nan
        return mean_map

    def _plot_pf_contour(ax, place_field_mask, bins):
        if place_field_mask is None or not np.any(place_field_mask):
            return
        extent = (bins[0][0], bins[0][-1], bins[1][0], bins[1][-1])
        padded_mask = np.zeros(
            (place_field_mask.shape[0] + 2, place_field_mask.shape[1] + 2), dtype=bool
        )
        padded_mask[1:-1, 1:-1] = place_field_mask
        bin_x = (extent[1] - extent[0]) / place_field_mask.shape[0]
        bin_y = (extent[3] - extent[2]) / place_field_mask.shape[1]
        padded_extent = (
            extent[0] - bin_x,
            extent[1] + bin_x,
            extent[2] - bin_y,
            extent[3] + bin_y,
        )
        ax.contour(
            padded_mask.T,
            levels=[0.5],
            colors="magenta",
            linewidths=0.8,
            extent=padded_extent,
            origin="lower",
        )

    bins, place_field_mask = _get_bins_and_mask(analysis)
    if bins is None or place_field_mask is None:
        return None, None, None

    def _as_map(data):
        if data is None:
            return None
        arr = np.asarray(data, dtype=float)
        if arr.shape != place_field_mask.shape:
            return None
        return arr

    def _blank_map():
        return np.full_like(place_field_mask, np.nan, dtype=float)

    rate_move_map = _as_map(analysis.get("rate_map"))
    rate_quiet_map = _as_map(analysis.get("rate_map_quiet"))
    if rate_move_map is None:
        rate_move_map = _blank_map()
    if rate_quiet_map is None:
        rate_quiet_map = _blank_map()

    trace = np.asarray(traces[cell_idx], dtype=float)
    trace = interpolate_nan_segment(trace)
    theta_vm = bandpass_filter(trace, theta_freqs[0], theta_freqs[1], frame_rate, order=5)
    slow_vm = lowpass_filter(trace, slow_freqs, frame_rate, order=5)
    theta_amp = np.abs(signal.hilbert(theta_vm))

    valid_frames = (~np.isnan(x_neural)) & (~np.isnan(y_neural)) & (~np.isnan(speed))
    valid_frames = _apply_bad_mask(valid_frames, cell_idx)
    speed_for_epochs = speed.copy()
    speed_for_epochs[~valid_frames] = np.nan

    _, _, moving_idx = _compute_moving_epochs(
        speed_for_epochs,
        frame_rate,
        kernel_size=kernel_size,
        filter_type=filter_type,
        speed_threshold=speed_threshold,
        min_duration_s=min_duration_s,
        merge_gap_s=merge_gap_s,
    )
    moving_idx = moving_idx[valid_frames[moving_idx]]
    moving_mask = np.zeros(len(speed), dtype=bool)
    moving_mask[moving_idx] = True
    quiet_mask = valid_frames & (~moving_mask)

    theta_move_map = _mean_map(
        x_neural[moving_mask],
        y_neural[moving_mask],
        theta_amp[moving_mask],
        bins,
    )
    theta_quiet_map = _mean_map(
        x_neural[quiet_mask],
        y_neural[quiet_mask],
        theta_amp[quiet_mask],
        bins,
    )
    slow_move_map = _mean_map(
        x_neural[moving_mask],
        y_neural[moving_mask],
        slow_vm[moving_mask],
        bins,
    )
    slow_quiet_map = _mean_map(
        x_neural[quiet_mask],
        y_neural[quiet_mask],
        slow_vm[quiet_mask],
        bins,
    )

    theta_vmax = np.nanmax([theta_move_map, theta_quiet_map])
    theta_vmin = np.nanmin([theta_move_map, theta_quiet_map])
    if np.any(np.isfinite(rate_move_map)) or np.any(np.isfinite(rate_quiet_map)):
        rate_vmax = np.nanmax([rate_move_map, rate_quiet_map])
        rate_vmin = 0.0 if np.isfinite(rate_vmax) else np.nan
    else:
        rate_vmax = np.nan
        rate_vmin = np.nan
    slow_vabs = np.nanmax(
        [
            np.nanmax(np.abs(slow_move_map)),
            np.nanmax(np.abs(slow_quiet_map)),
        ]
    )
    if np.isfinite(slow_vabs) and slow_vabs > 0:
        slow_vmin = -slow_vabs
        slow_vmax = slow_vabs
    else:
        slow_vmax = np.nanmax([slow_move_map, slow_quiet_map])
        slow_vmin = np.nanmin([slow_move_map, slow_quiet_map])

    fig, axes = plt.subplots(3, 2, figsize=(5, 3))
    extent = (bins[0][0], bins[0][-1], bins[1][0], bins[1][-1])

    def _plot_map(ax, data, title, vmin, vmax, cmap):
        masked = ma.masked_where(np.isnan(data), data)
        im = ax.imshow(
            masked.T,
            origin="lower",
            extent=extent,
            cmap=cmap,
            interpolation="nearest",
            vmin=vmin if np.isfinite(vmin) else None,
            vmax=vmax if np.isfinite(vmax) else None,
        )
        _plot_pf_contour(ax, place_field_mask, bins)
        ax.set_title(title, fontsize=6, fontname="Arial")
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes

        cax = inset_axes(
            ax,
            width="4%",
            height="80%",
            loc="lower left",
            bbox_to_anchor=(1.02, 0.1, 1, 1),
            bbox_transform=ax.transAxes,
            borderpad=0,
        )
        cbar = ax.figure.colorbar(im, cax=cax)
        cbar.ax.tick_params(labelsize=5)
        for label in cbar.ax.get_yticklabels():
            label.set_fontname("Arial")
        if not np.any(np.isfinite(data)):
            ax.text(
                0.5,
                0.5,
                "No data",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=6,
                fontname="Arial",
            )

    _plot_map(
        axes[0, 0],
        rate_move_map,
        "FR (move)",
        rate_vmin,
        rate_vmax,
        "jet",
    )
    _plot_map(
        axes[0, 1],
        rate_quiet_map,
        "FR (quiet)",
        rate_vmin,
        rate_vmax,
        "jet",
    )
    _plot_map(
        axes[1, 0],
        theta_move_map,
        "Theta amp (move)",
        theta_vmin,
        theta_vmax,
        "jet",
    )
    _plot_map(
        axes[1, 1],
        theta_quiet_map,
        "Theta amp (quiet)",
        theta_vmin,
        theta_vmax,
        "jet",
    )
    _plot_map(
        axes[2, 0],
        slow_move_map,
        "Slow Vm (move)",
        slow_vmin,
        slow_vmax,
        "bwr",
    )
    _plot_map(
        axes[2, 1],
        slow_quiet_map,
        "Slow Vm (quiet)",
        slow_vmin,
        slow_vmax,
        "bwr",
    )

    fig.suptitle(f"Cell {cell_idx}", fontsize=7, fontname="Arial", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if show:
        plt.show()

    maps = {
        "rate_move": rate_move_map,
        "rate_quiet": rate_quiet_map,
        "theta_move": theta_move_map,
        "theta_quiet": theta_quiet_map,
        "slow_move": slow_move_map,
        "slow_quiet": slow_quiet_map,
    }
    return fig, axes, maps


def get_subthreshold_activity(
    traces,
    all_spikes,
    all_CS_spikes,
    refined_SS,
    complex_bursts_dicts,
    spike_heights_interpolated,
    ts_neural,
    frame_rate,
    idx,
    mask_duration=1000,
    burst_isi_threshold=14,
    burst_window_ms=40,
    median_filter_size=11,
    theta_freqs=(4, 12),
    slow_freqs=2,
):
    trace = traces[idx].copy()
    spike = all_spikes[idx]
    cs_spike = all_CS_spikes[idx]
    ss_spike = refined_SS[idx]
    cs_starts = complex_bursts_dicts[idx]["starts"]
    cs_ends = complex_bursts_dicts[idx]["ends"]
    spk_heights = spike_heights_interpolated[idx]
    ts = ts_neural

    if mask_duration is not None:
        trace[:mask_duration] = np.nan

    trace = interpolate_nan_segment(trace)
    trace = trace / spk_heights
    single_spikes, bursts, burst_event_time = detect_burst_SS(
        spike, fr=frame_rate, burst_isi_threshold=burst_isi_threshold
    )
    single_spikes = np.asarray(single_spikes, dtype=np.int64)
    burst_event_time = np.asarray(burst_event_time, dtype=np.int64)
    simple_spikes = np.sort(np.concatenate([single_spikes, burst_event_time]).astype(np.int64))

    trace_sub = trace.copy()
    n = len(trace_sub)
    fr = frame_rate

    for t in single_spikes:
        if 0 <= t < n:
            trace_sub[max(0, t - 1) : min(n, t + 2)] = np.nan

    burst_window = int(burst_window_ms * fr / 1000)
    for burst_start in burst_event_time:
        if not (0 <= burst_start < n):
            continue
        start_idx = max(0, burst_start - burst_window // 2)
        end_idx = min(n, burst_start + burst_window // 2)
        local = trace_sub[start_idx:end_idx]
        if len(local) == 0:
            continue
        local_pks = signal.find_peaks(local)[0]
        for pk in local_pks:
            trace_sub[start_idx + pk] = np.nan

    trace_sub = interpolate_nan_segment(trace_sub)
    trace_sub = median_filter(trace_sub, size=median_filter_size)

    trace_sub_noCS = trace_sub.copy()
    for start, end in zip(cs_starts, cs_ends):
        trace_sub_noCS[max(0, start - 1) : min(n, end + 2)] = np.nan
    trace_sub_noCS = interpolate_nan_segment(trace_sub_noCS)

    theta_Vm = bandpass_filter(trace_sub, theta_freqs[0], theta_freqs[1], fs=fr, order=3)
    theta_Vm_noCS = bandpass_filter(
        trace_sub_noCS, theta_freqs[0], theta_freqs[1], fs=fr, order=3
    )

    slow_Vm = lowpass_filter(trace_sub, slow_freqs, fs=fr, order=3)
    slow_Vm_noCS = lowpass_filter(trace_sub_noCS, slow_freqs, fs=fr, order=3)

    return {
        "idx": idx,
        "ts": ts,
        "trace": trace,
        "trace_sub": trace_sub,
        "trace_sub_noCS": trace_sub_noCS,
        "theta_Vm": theta_Vm,
        "theta_Vm_noCS": theta_Vm_noCS,
        "slow_Vm": slow_Vm,
        "slow_Vm_noCS": slow_Vm_noCS,
        "theta_freqs": theta_freqs,
        "slow_freqs": slow_freqs,
        "single_spikes": single_spikes,
        "bursts": bursts,
        "burst_event_time": burst_event_time,
        "simple_spikes": simple_spikes,
        "spike": spike,
        "cs_spike": cs_spike,
        "ss_spike": ss_spike,
        "cs_starts": cs_starts,
        "cs_ends": cs_ends,
    }


def compute_theta_amp_by_pf(
    x_neural,
    y_neural,
    speed,
    frame_rate,
    spikes,
    trace,
    cell_idx,
    analysis=None,
    width_real=35.5,
    height_real=20,
    bin_size=1.5,
    smooth_sigma=2.0,
    num_shuffles=1000,
    place_field_threshold=0.4,
    min_field_bins=5,
    kernel_size=21,
    filter_type="boxcar",
    speed_threshold=2,
    min_duration_s=0.5,
    merge_gap_s=1.0,
    theta_freqs=(4, 8),
    theta_order=3,
    bad_timepoints=None,
):
    x_neural = np.asarray(x_neural)
    y_neural = np.asarray(y_neural)
    speed = np.asarray(speed, dtype=float)
    trace = np.asarray(trace, dtype=float)
    if np.any(~np.isfinite(trace)) and np.any(np.isfinite(trace)):
        idx = np.arange(len(trace))
        valid = np.isfinite(trace)
        trace = trace.copy()
        trace[~valid] = np.interp(idx[~valid], idx[valid], trace[valid])

    if analysis is None:
        analysis = analyze_place_cell_single_moving(
            x_neural,
            y_neural,
            spikes,
            speed,
            frame_rate,
            cell_idx,
            axes=None,
            width_real=width_real,
            height_real=height_real,
            bin_size=bin_size,
            smooth_sigma=smooth_sigma,
            num_shuffles=num_shuffles,
            place_field_threshold=place_field_threshold,
            min_field_bins=min_field_bins,
            kernel_size=kernel_size,
            filter_type=filter_type,
            speed_threshold=speed_threshold,
            min_duration_s=min_duration_s,
            merge_gap_s=merge_gap_s,
            bad_timepoints=bad_timepoints,
        )

    valid_frames = (~np.isnan(x_neural)) & (~np.isnan(y_neural)) & (~np.isnan(speed))
    moving_mask = np.zeros(len(x_neural), dtype=bool)
    moving_idx = analysis["moving_indices"]
    moving_mask[moving_idx] = True

    bins = [
        np.arange(0, width_real + bin_size, bin_size),
        np.arange(0, height_real + bin_size, bin_size),
    ]
    xbin = np.digitize(x_neural, bins[0]) - 1
    ybin = np.digitize(y_neural, bins[1]) - 1
    in_bounds = (
        (xbin >= 0)
        & (xbin < analysis["place_field_mask"].shape[0])
        & (ybin >= 0)
        & (ybin < analysis["place_field_mask"].shape[1])
    )
    inside_pf = np.zeros(len(x_neural), dtype=bool)
    inside_pf[in_bounds] = analysis["place_field_mask"][xbin[in_bounds], ybin[in_bounds]]

    theta_signal = bandpass_filter(
        trace, theta_freqs[0], theta_freqs[1], fs=frame_rate, order=theta_order
    )
    theta_amp = np.abs(signal.hilbert(theta_signal))

    run_mask = moving_mask & valid_frames
    rest_mask = (~moving_mask) & valid_frames

    def mean_or_nan(mask):
        if not np.any(mask):
            return np.nan
        return np.nanmean(theta_amp[mask])

    if not analysis["is_place_cell"]:
        return {
            "cell_idx": cell_idx,
            "is_place_cell": False,
            "p_value": analysis["p_value"],
            "si": analysis["si"],
            "theta_amp": {
                "run_in": np.nan,
                "run_out": np.nan,
                "rest_in": np.nan,
                "rest_out": np.nan,
            },
        }

    run_in = run_mask & inside_pf
    run_out = run_mask & (~inside_pf)
    rest_in = rest_mask & inside_pf
    rest_out = rest_mask & (~inside_pf)

    return {
        "cell_idx": cell_idx,
        "is_place_cell": True,
        "p_value": analysis["p_value"],
        "si": analysis["si"],
        "theta_amp": {
            "run_in": mean_or_nan(run_in),
            "run_out": mean_or_nan(run_out),
            "rest_in": mean_or_nan(rest_in),
            "rest_out": mean_or_nan(rest_out),
        },
    }


def compute_theta_amp_by_state(
    speed,
    frame_rate,
    trace,
    cell_idx,
    kernel_size=21,
    filter_type="boxcar",
    speed_threshold=2,
    min_duration_s=0.5,
    merge_gap_s=1.0,
    theta_freqs=(4, 8),
    theta_order=3,
    bad_timepoints=None,
):
    """
    Compute theta amplitude during moving vs resting states, ignoring place field.
    """
    speed = np.asarray(speed, dtype=float)
    trace = np.asarray(trace, dtype=float)
    if np.any(~np.isfinite(trace)) and np.any(np.isfinite(trace)):
        idx = np.arange(len(trace))
        valid = np.isfinite(trace)
        trace = trace.copy()
        trace[~valid] = np.interp(idx[~valid], idx[valid], trace[valid])

    valid_frames = np.isfinite(speed) & np.isfinite(trace)
    speed_for_epochs = speed.copy()
    speed_for_epochs[~valid_frames] = np.nan

    _, moving_epochs, moving_idx = _compute_moving_epochs(
        speed_for_epochs,
        frame_rate,
        kernel_size=kernel_size,
        filter_type=filter_type,
        speed_threshold=speed_threshold,
        min_duration_s=min_duration_s,
        merge_gap_s=merge_gap_s,
    )

    moving_mask = np.zeros(len(speed), dtype=bool)
    if len(moving_idx) > 0:
        moving_mask[moving_idx] = True

    theta_signal = bandpass_filter(
        trace, theta_freqs[0], theta_freqs[1], fs=frame_rate, order=theta_order
    )
    theta_amp = np.abs(signal.hilbert(theta_signal))

    run_mask = moving_mask & valid_frames
    rest_mask = (~moving_mask) & valid_frames

    def mean_or_nan(mask):
        if not np.any(mask):
            return np.nan
        return np.nanmean(theta_amp[mask])

    return {
        "cell_idx": cell_idx,
        "theta_amp": {
            "run": mean_or_nan(run_mask),
            "rest": mean_or_nan(rest_mask),
        },
        "moving_epochs": moving_epochs,
        "moving_indices": moving_idx,
    }


def compute_psd_by_state(
    speed,
    frame_rate,
    trace,
    cell_idx,
    kernel_size=21,
    filter_type="boxcar",
    speed_threshold=2,
    min_duration_s=0.5,
    merge_gap_s=1.0,
    chunk_s=2.0,
    nperseg_s=1.0,
    noverlap_frac=0.5,
    bad_timepoints=None,
):
    """
    Compute Welch PSD for moving vs resting by chunking epochs into fixed windows.
    """
    speed = np.asarray(speed, dtype=float)
    trace = np.asarray(trace, dtype=float)

    speed_valid = np.isfinite(speed)
    trace_valid = np.isfinite(trace)
    valid_frames = speed_valid.copy()
    if bad_timepoints is not None:
        bad_mask = np.asarray(bad_timepoints)
        if bad_mask.dtype != bool:
            if bad_mask.ndim == 1 and bad_mask.size == len(speed):
                bad_mask = bad_mask.astype(bool)
            else:
                bad_mask = np.asarray(bad_mask, dtype=int)
                bad_mask = bad_mask[(bad_mask >= 0) & (bad_mask < len(speed))]
                bad_bool = np.zeros(len(speed), dtype=bool)
                bad_bool[bad_mask] = True
                bad_mask = bad_bool
        if bad_mask.shape[0] != len(speed):
            raise ValueError("bad_timepoints must match speed length or be index list.")
        valid_frames &= ~bad_mask
    psd_valid = valid_frames & trace_valid
    speed_for_epochs = speed.copy()
    speed_for_epochs[~valid_frames] = np.nan

    _, _, moving_idx = _compute_moving_epochs(
        speed_for_epochs,
        frame_rate,
        kernel_size=kernel_size,
        filter_type=filter_type,
        speed_threshold=speed_threshold,
        min_duration_s=min_duration_s,
        merge_gap_s=merge_gap_s,
    )

    moving_mask = np.zeros(len(speed), dtype=bool)
    if len(moving_idx) > 0:
        moving_mask[moving_idx] = True
    moving_mask &= valid_frames
    moving_mask &= psd_valid
    rest_mask = (~moving_mask) & psd_valid

    def _mask_to_epochs(mask):
        epochs = []
        in_epoch = False
        start = 0
        for i, val in enumerate(mask):
            if val and not in_epoch:
                start = i
                in_epoch = True
            elif not val and in_epoch:
                epochs.append((start, i - 1))
                in_epoch = False
        if in_epoch:
            epochs.append((start, len(mask) - 1))
        return epochs

    def _epoch_psd(epoch_list):
        chunk_frames = int(round(chunk_s * frame_rate))
        if chunk_frames <= 0:
            return None, []
        nperseg = int(round(nperseg_s * frame_rate)) if nperseg_s else chunk_frames
        nperseg = max(1, min(nperseg, chunk_frames))
        noverlap = int(round(nperseg * noverlap_frac))
        noverlap = min(noverlap, nperseg - 1)

        psd_list = []
        freqs_out = None
        for start, end in epoch_list:
            if (end - start + 1) < chunk_frames:
                continue
            for s in range(start, end - chunk_frames + 2, chunk_frames):
                seg = trace[s : s + chunk_frames]
                if not np.all(np.isfinite(seg)):
                    continue
                freqs, pxx = signal.welch(
                    seg,
                    fs=frame_rate,
                    nperseg=nperseg,
                    noverlap=noverlap,
                    detrend="constant",
                )
                freqs_out = freqs
                psd_list.append(pxx)
        return freqs_out, psd_list

    run_epochs = _mask_to_epochs(moving_mask)
    rest_epochs = _mask_to_epochs(rest_mask)

    freqs_run, psd_run_list = _epoch_psd(run_epochs)
    freqs_rest, psd_rest_list = _epoch_psd(rest_epochs)

    freqs = freqs_run if freqs_run is not None else freqs_rest
    psd_run = np.nanmean(psd_run_list, axis=0) if psd_run_list else None
    psd_rest = np.nanmean(psd_rest_list, axis=0) if psd_rest_list else None

    return {
        "cell_idx": cell_idx,
        "freqs": freqs,
        "psd_run": psd_run,
        "psd_rest": psd_rest,
        "psd_run_chunks": psd_run_list,
        "psd_rest_chunks": psd_rest_list,
        "n_chunks_run": len(psd_run_list),
        "n_chunks_rest": len(psd_rest_list),
    }

def compute_vm_spectrogram(
    traces,
    cell_idx,
    frame_rate,
    window_size=2000,
    step_ms=100.0,
    fmin=1.0,
    fmax=100.0,
    detrend="constant",
    scaling="density",
    mode="psd",
    interpolate_nans=True,
):
    """
    Compute power spectrogram for one cell's Vm using a sliding Hann window.

    Window size is in samples; step is in ms. Frequency output can be limited
    with fmin/fmax. Returns empty arrays if the trace is shorter than the
    window size.
    """
    if window_size <= 0:
        raise ValueError("window_size must be positive.")
    if step_ms <= 0:
        raise ValueError("step_ms must be positive.")
    if frame_rate <= 0:
        raise ValueError("frame_rate must be positive.")

    trace = np.asarray(traces[cell_idx], dtype=float)
    if trace.ndim != 1:
        raise ValueError("Vm trace must be 1D.")
    if interpolate_nans:
        trace = interpolate_nan_segment(trace)
    if not np.any(np.isfinite(trace)):
        raise ValueError("Vm trace has no finite samples.")
    if trace.size < window_size:
        return {
            "cell_idx": cell_idx,
            "freqs": np.array([]),
            "times": np.array([]),
            "power": np.empty((0, 0)),
            "window_size": window_size,
            "step_ms": step_ms,
            "fmin": fmin,
            "fmax": fmax,
        }

    step_samples = int(round(step_ms * frame_rate / 1000.0))
    step_samples = max(1, step_samples)
    noverlap = window_size - step_samples
    if noverlap < 0:
        noverlap = 0

    freqs, times, power = signal.spectrogram(
        trace,
        fs=frame_rate,
        window="hann",
        nperseg=window_size,
        noverlap=noverlap,
        nfft=window_size,
        detrend=detrend,
        scaling=scaling,
        mode=mode,
    )

    if fmin is not None or fmax is not None:
        if fmin is None:
            fmin = -np.inf
        if fmax is None:
            fmax = np.inf
        freq_mask = (freqs >= fmin) & (freqs <= fmax)
        freqs = freqs[freq_mask]
        power = power[freq_mask, :]

    return {
        "cell_idx": cell_idx,
        "freqs": freqs,
        "times": times,
        "power": power,
        "window_size": window_size,
        "step_ms": step_ms,
        "fmin": fmin,
        "fmax": fmax,
    }

def compute_vm_spectrogram2(
    traces,
    cell_idx,
    frame_rate,
    fmin=1.0,
    fmax=100.0,
    n_freqs_lo=100,
    n_freqs_hi=20,
    freq_split=20.0,
    w=5.0,
    step_ms=20.0,
    interpolate_nans=True,
):
    """
    Compute power spectrogram for one cell's Vm using a Morlet wavelet transform.

    Uses scipy.signal.cwt with morlet2 wavelets. Frequency bins are dense
    below freq_split and sparse above. The result is downsampled in time
    according to step_ms.

    Parameters
    ----------
    traces : array-like
        List/array of Vm traces (one per cell).
    cell_idx : int
        Index of the cell to analyse.
    frame_rate : float
        Sampling rate in Hz.
    fmin, fmax : float
        Frequency range (Hz) for the spectrogram.
    n_freqs_lo : int
        Number of frequency bins in [fmin, freq_split).
    n_freqs_hi : int
        Number of frequency bins in [freq_split, fmax].
    freq_split : float
        Boundary (Hz) between dense and sparse regions.
    w : float
        Morlet wavelet omega0 parameter (number of cycles). Higher values
        give better frequency resolution at the cost of time resolution.
    step_ms : float
        Time step (ms) for downsampling the output along the time axis.
    interpolate_nans : bool
        If True, interpolate NaN segments before computing.

    Returns
    -------
    dict with keys: cell_idx, freqs, times, power, fmin, fmax, step_ms, w.
    """
    if frame_rate <= 0:
        raise ValueError("frame_rate must be positive.")
    if step_ms <= 0:
        raise ValueError("step_ms must be positive.")

    trace = np.asarray(traces[cell_idx], dtype=float)
    if trace.ndim != 1:
        raise ValueError("Vm trace must be 1D.")
    if interpolate_nans:
        trace = interpolate_nan_segment(trace)
    if not np.any(np.isfinite(trace)):
        raise ValueError("Vm trace has no finite samples.")

    # Dense sampling below freq_split, sparse above
    freqs_lo = np.linspace(fmin, freq_split, n_freqs_lo, endpoint=False)
    freqs_hi = np.linspace(freq_split, fmax, n_freqs_hi, endpoint=True)
    freqs = np.concatenate([freqs_lo, freqs_hi])

    # Morlet CWT via fftconvolve (scipy.signal.cwt was removed in SciPy 1.12)
    N = trace.size
    power = np.empty((len(freqs), N))
    for i, freq in enumerate(freqs):
        width = w * frame_rate / (2.0 * np.pi * freq)
        M = int(10 * width + 1)
        t = np.arange(0, M) - (M - 1.0) / 2
        wavelet = np.exp(1j * w * t / width) * np.exp(-0.5 * (t / width) ** 2)
        wavelet /= np.sqrt(width)  # L2 normalisation
        conv = signal.fftconvolve(trace, np.conj(wavelet[::-1]), mode="same")
        power[i] = np.abs(conv) ** 2

    # Full-resolution time axis
    times_full = np.arange(trace.size) / frame_rate

    # Downsample in time
    step_samples = max(1, int(round(step_ms * frame_rate / 1000.0)))
    time_idx = np.arange(0, trace.size, step_samples)
    times = times_full[time_idx]
    power = power[:, time_idx]

    return {
        "cell_idx": cell_idx,
        "freqs": freqs,
        "times": times,
        "power": power,
        "fmin": fmin,
        "fmax": fmax,
        "step_ms": step_ms,
        "w": w,
    }


def plot_spike_shapes_by_pf(
    x_neural,
    y_neural,
    speed,
    frame_rate,
    traces,
    refined_SS,
    complex_bursts_dicts,
    all_CS_spikes=None,
    analysis_list=None,
    spikes=None,
    cell_indices=None,
    width_real=35.5,
    height_real=20,
    bin_size=1.5,
    smooth_sigma=2.0,
    num_shuffles=1000,
    place_field_threshold=0.4,
    min_field_bins=5,
    kernel_size=21,
    filter_type="boxcar",
    speed_threshold=2,
    min_duration_s=0.5,
    merge_gap_s=1.0,
    pre_ms=10,
    post_ms=100,
    median_filter_size=11,
    require_place_cell=True,
    plot_sem=True,
    show=True,
    return_data=False,
):
    x_neural = np.asarray(x_neural)
    y_neural = np.asarray(y_neural)
    speed = np.asarray(speed, dtype=float)

    if analysis_list is None and spikes is None:
        raise ValueError("analysis_list or spikes must be provided")

    if cell_indices is None:
        cell_indices = range(len(traces))

    pre_frames = int(round(pre_ms / 1000 * frame_rate))
    post_frames = int(round(post_ms / 1000 * frame_rate))
    window_len = pre_frames + post_frames + 1
    time_ms = (np.arange(-pre_frames, post_frames + 1) / frame_rate) * 1000.0

    shapes = {
        "simple": {"run_in": [], "run_out": [], "rest_in": [], "rest_out": []},
        "complex": {"run_in": [], "run_out": [], "rest_in": [], "rest_out": []},
    }

    def _normalize_window(trace, trace_mf, spike_idx):
        if spike_idx - pre_frames < 0 or spike_idx + post_frames >= len(trace):
            return None
        baseline_region = trace_mf[max(0, spike_idx - 3) : spike_idx]
        baseline = np.min(baseline_region) if baseline_region.size > 0 else 0.0
        height = trace[spike_idx] - baseline
        if not np.isfinite(height) or height == 0:
            return None
        window = trace[spike_idx - pre_frames : spike_idx + post_frames + 1]
        if window.size != window_len:
            return None
        return (window - baseline) / height

    for cell_idx in cell_indices:
        if cell_idx >= len(traces):
            continue
        trace = np.asarray(traces[cell_idx], dtype=float)
        if trace.size == 0:
            continue
        trace_mf = median_filter(trace, size=median_filter_size)

        analysis = None
        if analysis_list is not None and cell_idx < len(analysis_list):
            analysis = analysis_list[cell_idx]
        if analysis is None:
            analysis = analyze_place_cell_single_moving(
                x_neural,
                y_neural,
                spikes,
                speed,
                frame_rate,
                cell_idx,
                axes=None,
                width_real=width_real,
                height_real=height_real,
                bin_size=bin_size,
                smooth_sigma=smooth_sigma,
                num_shuffles=num_shuffles,
                place_field_threshold=place_field_threshold,
                min_field_bins=min_field_bins,
                kernel_size=kernel_size,
                filter_type=filter_type,
                speed_threshold=speed_threshold,
                min_duration_s=min_duration_s,
                merge_gap_s=merge_gap_s,
            )

        if require_place_cell and not analysis.get("is_place_cell", False):
            continue

        place_field_mask = np.asarray(analysis.get("place_field_mask", []), dtype=bool)
        if place_field_mask.size == 0 or not np.any(place_field_mask):
            continue

        params = analysis.get("params", {})
        width_real_local = params.get("width_real", width_real)
        height_real_local = params.get("height_real", height_real)
        bin_size_local = params.get("bin_size", bin_size)
        bins = [
            np.arange(0, width_real_local + bin_size_local, bin_size_local),
            np.arange(0, height_real_local + bin_size_local, bin_size_local),
        ]

        inside_pf = _positions_in_place_field(x_neural, y_neural, bins, place_field_mask)

        moving_mask = np.zeros(len(speed), dtype=bool)
        moving_idx = analysis.get("moving_indices")
        if moving_idx is None or len(moving_idx) == 0:
            _, _, moving_idx = _compute_moving_epochs(
                speed,
                frame_rate,
                kernel_size=kernel_size,
                filter_type=filter_type,
                speed_threshold=speed_threshold,
                min_duration_s=min_duration_s,
                merge_gap_s=merge_gap_s,
            )
        if moving_idx is not None and len(moving_idx) > 0:
            moving_mask[moving_idx] = True

        simple_spikes = np.array([], dtype=int)
        if cell_idx < len(refined_SS):
            simple_spikes = np.asarray(refined_SS[cell_idx], dtype=int)

        burst_spikes = np.array([], dtype=int)
        if cell_idx < len(complex_bursts_dicts):
            starts = np.asarray(complex_bursts_dicts[cell_idx].get("starts", []), dtype=int)
            ends = np.asarray(complex_bursts_dicts[cell_idx].get("ends", []), dtype=int)
            if all_CS_spikes is not None and cell_idx < len(all_CS_spikes):
                cs_spikes = np.asarray(all_CS_spikes[cell_idx], dtype=int)
                cs_spikes = cs_spikes[(cs_spikes >= 0) & (cs_spikes < len(trace))]
                if cs_spikes.size > 0 and starts.size > 0 and ends.size > 0:
                    burst_first = []
                    for start_idx, end_idx in zip(starts, ends):
                        in_burst = cs_spikes[(cs_spikes >= start_idx) & (cs_spikes <= end_idx)]
                        if in_burst.size == 0:
                            continue
                        burst_first.append(in_burst[0])
                    burst_spikes = np.asarray(burst_first, dtype=int)
            else:
                burst_spikes = np.asarray(
                    complex_bursts_dicts[cell_idx].get("complex_bursts", []), dtype=int
                )

        def _add_waveforms(event_indices, target):
            for spike_idx in event_indices:
                if spike_idx < 0 or spike_idx >= len(trace):
                    continue
                if (
                    not np.isfinite(x_neural[spike_idx])
                    or not np.isfinite(y_neural[spike_idx])
                    or not np.isfinite(speed[spike_idx])
                ):
                    continue
                window = _normalize_window(trace, trace_mf, spike_idx)
                if window is None:
                    continue
                if moving_mask[spike_idx]:
                    if inside_pf[spike_idx]:
                        target["run_in"].append(window)
                    else:
                        target["run_out"].append(window)
                else:
                    if inside_pf[spike_idx]:
                        target["rest_in"].append(window)
                    else:
                        target["rest_out"].append(window)

        _add_waveforms(simple_spikes, shapes["simple"])
        _add_waveforms(burst_spikes, shapes["complex"])

    def _mean_sem(stack):
        if len(stack) == 0:
            return None, None
        arr = np.vstack(stack)
        mean = np.nanmean(arr, axis=0)
        count = np.sum(np.isfinite(arr), axis=0)
        std = np.nanstd(arr, axis=0, ddof=0)
        sem = np.where(count > 0, std / np.sqrt(count), np.nan)
        return mean, sem

    fig, axes = plt.subplots(2, 2, figsize=(3, 2), sharex=True, sharey=True)
    state_map = [("run", "Locomotion"), ("rest", "Quiet")]
    type_map = [("simple", "Simple spikes"), ("complex", "Complex bursts")]
    color_map = {"in": "magenta", "out": "gray"}

    y_min_global = np.inf
    y_max_global = -np.inf
    for state_key, _ in state_map:
        for spike_type, _ in type_map:
            for pf_key in ("in", "out"):
                key = f"{state_key}_{pf_key}"
                mean, sem = _mean_sem(shapes[spike_type][key])
                if mean is None:
                    continue
                local_min = np.nanmin(mean)
                local_max = np.nanmax(mean)
                if plot_sem and sem is not None:
                    local_min = np.nanmin(mean - sem)
                    local_max = np.nanmax(mean + sem)
                if np.isfinite(local_min):
                    y_min_global = min(y_min_global, local_min)
                if np.isfinite(local_max):
                    y_max_global = max(y_max_global, local_max)
    if not np.isfinite(y_min_global) or not np.isfinite(y_max_global):
        y_min_global = 0.0
        y_max_global = 1.0
    y_min_global = min(y_min_global, 0.0)
    y_max_global = max(y_max_global, 1.0)

    for row_idx, (state_key, row_label) in enumerate(state_map):
        for col_idx, (spike_type, col_label) in enumerate(type_map):
            ax = axes[row_idx, col_idx]
            pre_candidates = []
            for pf_key, pf_label in [("in", "In PF"), ("out", "Out PF")]:
                key = f"{state_key}_{pf_key}"
                mean, sem = _mean_sem(shapes[spike_type][key])
                if mean is None:
                    continue
                if row_idx == 0 and col_idx == 0:
                    label = f"{pf_label} (n={len(shapes[spike_type][key])})"
                else:
                    label = f"n={len(shapes[spike_type][key])}"
                ax.plot(
                    time_ms,
                    mean,
                    color=color_map[pf_key],
                    linewidth=0.8,
                    label=label,
                )
                pre_mask = time_ms < 0
                if np.any(pre_mask):
                    pre_candidates.append(mean[pre_mask])
                if plot_sem and sem is not None:
                    ax.fill_between(
                        time_ms,
                        mean - sem,
                        mean + sem,
                        color=color_map[pf_key],
                        alpha=0.2,
                        linewidth=0,
                    )
                    if np.any(pre_mask):
                        pre_candidates.append((mean - sem)[pre_mask])
                        pre_candidates.append((mean + sem)[pre_mask])
            ax.set_xlim(time_ms[0], time_ms[-1])
            ax.set_ylim(y_min_global, y_max_global)

            if pre_candidates:
                pre_stack = np.hstack(pre_candidates)
                y_pre_min = np.nanmin(pre_stack)
                if not np.isfinite(y_pre_min):
                    y_pre_min = y_min_global
            else:
                y_pre_min = y_min_global
            y_line_low = min(y_pre_min, 1.0)
            y_line_high = max(y_min_global, 1.0)
            ax.plot(
                [0, 0],
                [y_line_low, y_line_high],
                color="black",
                linestyle="--",
                linewidth=0.5,
                alpha=0.6,
            )
            if row_idx == 0 and col_idx == 0:
                x_range = time_ms[-1] - time_ms[0]
                x0 = time_ms[0] + 0.05 * x_range
                x1 = x0 + 10.0
                if x1 > time_ms[-1]:
                    x1 = time_ms[-1] - 0.05 * x_range
                    x0 = x1 - 10.0
                y_bar = y_min_global + 0.05 * (
                    y_max_global - y_min_global if y_max_global != y_min_global else 1.0
                )
                y_range = y_max_global - y_min_global if y_max_global != y_min_global else 1.0
                ax.plot(
                    [x0, x0],
                    [0, 1],
                    color="black",
                    linewidth=1.0,
                    solid_capstyle="butt",
                )
                ax.plot(
                    [x0, x1],
                    [y_bar, y_bar],
                    color="black",
                    linewidth=1.0,
                    solid_capstyle="butt",
                )
                ax.text(
                    x0 - 0.5,
                    0.5,
                    "1 spk",
                    fontsize=6,
                    fontname="Arial",
                    ha="right",
                    va="center",
                    rotation=90,
                )
                ax.text(
                    (x0 + x1) / 2,
                    y_bar - 0.05 * y_range,
                    "10 ms",
                    fontsize=6,
                    fontname="Arial",
                    ha="center",
                    va="top",
                )
            ax.set_title(
                f"{col_label} - {row_label}",
                fontsize=6,
                fontname="Arial",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.legend(fontsize=5, frameon=False)

    plt.tight_layout()
    if show:
        plt.show()

    data = {"time_ms": time_ms, "shapes": shapes}
    if return_data:
        return fig, axes, data
    return fig, axes


def plot_spike_shapes_by_pf_and_direction(
    x_neural,
    y_neural,
    speed,
    frame_rate,
    traces,
    refined_SS,
    complex_bursts_dicts,
    all_CS_spikes=None,
    analysis_list=None,
    spikes=None,
    cell_indices=None,
    width_real=35.5,
    height_real=20,
    bin_size=1.5,
    smooth_sigma=2.0,
    num_shuffles=1000,
    place_field_threshold=0.4,
    min_field_bins=5,
    kernel_size=21,
    filter_type="boxcar",
    speed_threshold=2,
    min_duration_s=0.5,
    merge_gap_s=1.0,
    pre_ms=10,
    post_ms=100,
    median_filter_size=11,
    require_place_cell=True,
    plot_sem=True,
    show=True,
    return_data=False,
):
    x_neural = np.asarray(x_neural)
    y_neural = np.asarray(y_neural)
    speed = np.asarray(speed, dtype=float)

    if analysis_list is None and spikes is None:
        raise ValueError("analysis_list or spikes must be provided")

    if cell_indices is None:
        cell_indices = range(len(traces))

    pre_frames = int(round(pre_ms / 1000 * frame_rate))
    post_frames = int(round(post_ms / 1000 * frame_rate))
    window_len = pre_frames + post_frames + 1
    time_ms = (np.arange(-pre_frames, post_frames + 1) / frame_rate) * 1000.0

    shapes = {
        "simple": {
            "run_cw_in": [],
            "run_cw_out": [],
            "run_ccw_in": [],
            "run_ccw_out": [],
            "rest_in": [],
            "rest_out": [],
        },
        "complex": {
            "run_cw_in": [],
            "run_cw_out": [],
            "run_ccw_in": [],
            "run_ccw_out": [],
            "rest_in": [],
            "rest_out": [],
        },
    }

    def _normalize_window(trace, trace_mf, spike_idx):
        if spike_idx - pre_frames < 0 or spike_idx + post_frames >= len(trace):
            return None
        baseline_region = trace_mf[max(0, spike_idx - 3) : spike_idx]
        baseline = np.min(baseline_region) if baseline_region.size > 0 else 0.0
        height = trace[spike_idx] - baseline
        if not np.isfinite(height) or height == 0:
            return None
        window = trace[spike_idx - pre_frames : spike_idx + post_frames + 1]
        if window.size != window_len:
            return None
        return (window - baseline) / height

    for cell_idx in cell_indices:
        if cell_idx >= len(traces):
            continue
        trace = np.asarray(traces[cell_idx], dtype=float)
        if trace.size == 0:
            continue
        trace_mf = median_filter(trace, size=median_filter_size)

        analysis = None
        if analysis_list is not None and cell_idx < len(analysis_list):
            analysis = analysis_list[cell_idx]
        if analysis is None:
            analysis = analyze_place_cell_single_moving(
                x_neural,
                y_neural,
                spikes,
                speed,
                frame_rate,
                cell_idx,
                axes=None,
                width_real=width_real,
                height_real=height_real,
                bin_size=bin_size,
                smooth_sigma=smooth_sigma,
                num_shuffles=num_shuffles,
                place_field_threshold=place_field_threshold,
                min_field_bins=min_field_bins,
                kernel_size=kernel_size,
                filter_type=filter_type,
                speed_threshold=speed_threshold,
                min_duration_s=min_duration_s,
                merge_gap_s=merge_gap_s,
            )

        if require_place_cell and not analysis.get("is_place_cell", False):
            continue

        place_field_mask = np.asarray(analysis.get("place_field_mask", []), dtype=bool)
        if place_field_mask.size == 0 or not np.any(place_field_mask):
            continue

        params = analysis.get("params", {})
        width_real_local = params.get("width_real", width_real)
        height_real_local = params.get("height_real", height_real)
        bin_size_local = params.get("bin_size", bin_size)
        bins = [
            np.arange(0, width_real_local + bin_size_local, bin_size_local),
            np.arange(0, height_real_local + bin_size_local, bin_size_local),
        ]

        inside_pf = _positions_in_place_field(x_neural, y_neural, bins, place_field_mask)

        moving_mask = np.zeros(len(speed), dtype=bool)
        moving_idx = analysis.get("moving_indices")
        if moving_idx is None or len(moving_idx) == 0:
            _, _, moving_idx = _compute_moving_epochs(
                speed,
                frame_rate,
                kernel_size=kernel_size,
                filter_type=filter_type,
                speed_threshold=speed_threshold,
                min_duration_s=min_duration_s,
                merge_gap_s=merge_gap_s,
            )
        if moving_idx is not None and len(moving_idx) > 0:
            moving_mask[moving_idx] = True

        traversal_type_by_frame = np.full(len(speed), None, dtype=object)
        traversal_epochs, _, _, _, traversal_types = find_place_field_traversals(
            x_neural,
            y_neural,
            spikes if spikes is not None else refined_SS,
            speed,
            frame_rate,
            cell_idx,
            speed_threshold=speed_threshold,
            min_duration_ms=int(round(min_duration_s * 1000)),
            merge_gap_s=merge_gap_s,
            analysis=analysis,
            return_traversal_types=True,
            verbose=False,
        )
        for (start, end), traversal_type in zip(traversal_epochs, traversal_types):
            if traversal_type in ("cw", "ccw"):
                traversal_type_by_frame[start:end] = traversal_type

        simple_spikes = np.array([], dtype=int)
        if cell_idx < len(refined_SS):
            simple_spikes = np.asarray(refined_SS[cell_idx], dtype=int)

        burst_spikes = np.array([], dtype=int)
        if cell_idx < len(complex_bursts_dicts):
            starts = np.asarray(complex_bursts_dicts[cell_idx].get("starts", []), dtype=int)
            ends = np.asarray(complex_bursts_dicts[cell_idx].get("ends", []), dtype=int)
            if all_CS_spikes is not None and cell_idx < len(all_CS_spikes):
                cs_spikes = np.asarray(all_CS_spikes[cell_idx], dtype=int)
                cs_spikes = cs_spikes[(cs_spikes >= 0) & (cs_spikes < len(trace))]
                if cs_spikes.size > 0 and starts.size > 0 and ends.size > 0:
                    burst_first = []
                    for start_idx, end_idx in zip(starts, ends):
                        in_burst = cs_spikes[(cs_spikes >= start_idx) & (cs_spikes <= end_idx)]
                        if in_burst.size == 0:
                            continue
                        burst_first.append(in_burst[0])
                    burst_spikes = np.asarray(burst_first, dtype=int)
            else:
                burst_spikes = np.asarray(
                    complex_bursts_dicts[cell_idx].get("complex_bursts", []), dtype=int
                )

        def _add_waveforms(event_indices, target):
            for spike_idx in event_indices:
                if spike_idx < 0 or spike_idx >= len(trace):
                    continue
                if (
                    not np.isfinite(x_neural[spike_idx])
                    or not np.isfinite(y_neural[spike_idx])
                    or not np.isfinite(speed[spike_idx])
                ):
                    continue
                window = _normalize_window(trace, trace_mf, spike_idx)
                if window is None:
                    continue
                if moving_mask[spike_idx]:
                    direction = traversal_type_by_frame[spike_idx]
                    if direction not in ("cw", "ccw"):
                        continue
                    if inside_pf[spike_idx]:
                        target[f"run_{direction}_in"].append(window)
                    else:
                        target[f"run_{direction}_out"].append(window)
                else:
                    if inside_pf[spike_idx]:
                        target["rest_in"].append(window)
                    else:
                        target["rest_out"].append(window)

        _add_waveforms(simple_spikes, shapes["simple"])
        _add_waveforms(burst_spikes, shapes["complex"])

    def _mean_sem(stack):
        if len(stack) == 0:
            return None, None
        arr = np.vstack(stack)
        mean = np.nanmean(arr, axis=0)
        count = np.sum(np.isfinite(arr), axis=0)
        std = np.nanstd(arr, axis=0, ddof=0)
        sem = np.where(count > 0, std / np.sqrt(count), np.nan)
        return mean, sem

    fig, axes = plt.subplots(2, 4, figsize=(6, 2), sharex=True, sharey=True)

    color_map = {"in": "magenta", "out": "gray"}

    y_min_global = np.inf
    y_max_global = -np.inf
    for spike_type in ("simple", "complex"):
        for pf_key in ("in", "out"):
            for state_key in ("run_cw", "run_ccw", "rest"):
                key = f"{state_key}_{pf_key}"
                mean, sem = _mean_sem(shapes[spike_type][key])
                if mean is None:
                    continue
                local_min = np.nanmin(mean)
                local_max = np.nanmax(mean)
                if plot_sem and sem is not None:
                    local_min = np.nanmin(mean - sem)
                    local_max = np.nanmax(mean + sem)
                y_min_global = min(y_min_global, local_min)
                y_max_global = max(y_max_global, local_max)

    if not np.isfinite(y_min_global) or not np.isfinite(y_max_global):
        y_min_global = 0.0
        y_max_global = 1.0
    y_min_global = min(y_min_global, 0.0)
    y_max_global = max(y_max_global, 1.0)

    axis_specs = [
        (0, 0, "simple", "run_cw", "SS-CW"),
        (0, 1, "simple", "run_ccw", "SS-CCW"),
        (0, 2, "complex", "run_cw", "CB-CW"),
        (0, 3, "complex", "run_ccw", "CB-CCW"),
        (1, 0, "simple", "rest", "Simple spikes - Quiet"),
        (1, 2, "complex", "rest", "Complex bursts - Quiet"),
    ]

    for row_idx, col_idx, spike_type, state_key, title in axis_specs:
        ax = axes[row_idx, col_idx]
        plotted = False
        pre_candidates = []
        for pf_key, pf_label in (("in", "In PF"), ("out", "Out PF")):
            key = f"{state_key}_{pf_key}"
            mean, sem = _mean_sem(shapes[spike_type][key])
            if mean is None:
                continue
            if row_idx == 0 and col_idx == 0:
                label = f"{pf_label} (n={len(shapes[spike_type][key])})"
            else:
                label = f"n={len(shapes[spike_type][key])}"
            ax.plot(
                time_ms,
                mean,
                color=color_map[pf_key],
                linewidth=0.8,
                label=label,
            )
            if plot_sem and sem is not None:
                ax.fill_between(
                    time_ms,
                    mean - sem,
                    mean + sem,
                    color=color_map[pf_key],
                    alpha=0.2,
                    linewidth=0,
                )
            pre_mask = time_ms < 0
            if np.any(pre_mask):
                pre_candidates.append(mean[pre_mask])
                if sem is not None:
                    pre_candidates.append((mean - sem)[pre_mask])
                    pre_candidates.append((mean + sem)[pre_mask])
            plotted = True

        if pre_candidates:
            pre_stack = np.hstack(pre_candidates)
            y_pre_min = np.nanmin(pre_stack)
            if not np.isfinite(y_pre_min):
                y_pre_min = y_min_global
        else:
            y_pre_min = y_min_global
        y_line_low = min(y_pre_min, 1.0)
        y_line_high = max(y_min_global, 1.0)
        ax.set_xlim(time_ms[0], time_ms[-1])
        ax.set_ylim(y_min_global, y_max_global)
        ax.plot(
            [0, 0],
            [y_line_low, y_line_high],
            color="black",
            linestyle="--",
            linewidth=0.5,
            alpha=0.6,
        )
        if row_idx == 0 and col_idx == 0:
            x_range = time_ms[-1] - time_ms[0]
            x0 = time_ms[0] + 0.05 * x_range
            x1 = x0 + 10.0
            if x1 > time_ms[-1]:
                x1 = time_ms[-1] - 0.05 * x_range
                x0 = x1 - 10.0
            y_bar = y_min_global + 0.05 * (
                y_max_global - y_min_global if y_max_global != y_min_global else 1.0
            )
            y_range = y_max_global - y_min_global if y_max_global != y_min_global else 1.0
            ax.plot(
                [x0, x0],
                [0, 1],
                color="black",
                linewidth=1.0,
                solid_capstyle="butt",
            )
            ax.plot(
                [x0, x1],
                [y_bar, y_bar],
                color="black",
                linewidth=1.0,
                solid_capstyle="butt",
            )
            ax.text(
                x0 - 0.5,
                0.5,
                "1 spk",
                fontsize=6,
                fontname="Arial",
                ha="right",
                va="center",
                rotation=90,
            )
            ax.text(
                (x0 + x1) / 2,
                y_bar - 0.05 * y_range,
                "10 ms",
                fontsize=6,
                fontname="Arial",
                ha="center",
                va="top",
            )
        ax.set_title(
            title,
            fontsize=6,
            fontname="Arial",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        if plotted:
            ax.legend(fontsize=5, frameon=False)
        else:
            ax.text(
                0.5,
                0.5,
                "No data",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=6,
                fontname="Arial",
            )

    plt.tight_layout()
    if show:
        plt.show()

    if return_data:
        return fig, axes, {"time_ms": time_ms, "shapes": shapes}
    return fig, axes


def plot_spike_shape_grand_average_by_direction(
    time_ms,
    grand_stack,
    color_map=None,
    fig_size=(6, 2),
    show=True,
    output_path=None,
):
    if color_map is None:
        color_map = {"in": "magenta", "out": "gray"}

    fig, axes = plt.subplots(2, 4, figsize=fig_size, sharex=True, sharey=True)

    type_map = [("simple", "Simple spikes"), ("complex", "Complex bursts")]

    y_min_global = np.inf
    y_max_global = -np.inf
    for spike_type, _ in type_map:
        for pf_key in ("in", "out"):
            for state_key in ("run_cw", "run_ccw", "rest"):
                cond = f"{state_key}_{pf_key}"
                stack = grand_stack.get(spike_type, {}).get(cond, [])
                if not stack:
                    continue
                arr = np.vstack(stack)
                valid_rows = np.any(np.isfinite(arr), axis=1)
                if not np.any(valid_rows):
                    continue
                arr = arr[valid_rows]
                mean_wave = np.nanmean(arr, axis=0)
                std_wave = np.nanstd(arr, axis=0, ddof=0)
                count = np.sum(np.isfinite(arr), axis=0)
                sem_wave = np.where(count > 0, std_wave / np.sqrt(count), np.nan)
                if np.any(np.isfinite(mean_wave)):
                    y_min_global = min(y_min_global, np.nanmin(mean_wave - sem_wave))
                    y_max_global = max(y_max_global, np.nanmax(mean_wave + sem_wave))
    if not np.isfinite(y_min_global) or not np.isfinite(y_max_global):
        y_min_global = 0.0
        y_max_global = 1.0
    y_min_global = min(y_min_global, 0.0)
    y_max_global = max(y_max_global, 1.0)

    axis_specs = [
        (0, 0, "simple", "run_cw", "SS-CW"),
        (0, 1, "simple", "run_ccw", "SS-CCW"),
        (0, 2, "complex", "run_cw", "CB-CW"),
        (0, 3, "complex", "run_ccw", "CB-CCW"),
        (1, 0, "simple", "rest", "Simple spikes - Quiet"),
        (1, 2, "complex", "rest", "Complex bursts - Quiet"),
    ]

    for row_idx, col_idx, spike_type, state_key, title in axis_specs:
        ax = axes[row_idx, col_idx]
        plotted = False
        pre_candidates = []
        for pf_key, pf_label in (("in", "In PF"), ("out", "Out PF")):
            cond = f"{state_key}_{pf_key}"
            stack = grand_stack.get(spike_type, {}).get(cond, [])
            if not stack:
                continue
            arr = np.vstack(stack)
            valid_rows = np.any(np.isfinite(arr), axis=1)
            if not np.any(valid_rows):
                continue
            arr = arr[valid_rows]
            mean_wave = np.nanmean(arr, axis=0)
            std_wave = np.nanstd(arr, axis=0, ddof=0)
            count = np.sum(np.isfinite(arr), axis=0)
            sem_wave = np.where(count > 0, std_wave / np.sqrt(count), np.nan)
            if row_idx == 0 and col_idx == 0:
                label = f"{pf_label} (n={int(np.sum(valid_rows))})"
            else:
                label = f"n={int(np.sum(valid_rows))}"
            ax.plot(
                time_ms,
                mean_wave,
                color=color_map[pf_key],
                linewidth=0.8,
                label=label,
            )
            if np.any(np.isfinite(sem_wave)):
                ax.fill_between(
                    time_ms,
                    mean_wave - sem_wave,
                    mean_wave + sem_wave,
                    color=color_map[pf_key],
                    alpha=0.2,
                    linewidth=0,
                )
            pre_mask = time_ms < 0
            if np.any(pre_mask):
                pre_candidates.append(mean_wave[pre_mask])
                pre_candidates.append((mean_wave - sem_wave)[pre_mask])
                pre_candidates.append((mean_wave + sem_wave)[pre_mask])
            plotted = True

        if pre_candidates:
            pre_stack = np.hstack(pre_candidates)
            y_pre_min = np.nanmin(pre_stack)
            if not np.isfinite(y_pre_min):
                y_pre_min = y_min_global
        else:
            y_pre_min = y_min_global
        y_line_low = min(y_pre_min, 1.0)
        y_line_high = max(y_min_global, 1.0)
        ax.set_xlim(time_ms[0], time_ms[-1])
        ax.set_ylim(y_min_global, y_max_global)
        ax.plot(
            [0, 0],
            [y_line_low, y_line_high],
            color="black",
            linestyle="--",
            linewidth=0.5,
            alpha=0.6,
        )
        if row_idx == 0 and col_idx == 0:
            x_range = time_ms[-1] - time_ms[0]
            x0 = time_ms[0] + 0.05 * x_range
            x1 = x0 + 10.0
            if x1 > time_ms[-1]:
                x1 = time_ms[-1] - 0.05 * x_range
                x0 = x1 - 10.0
            y_bar = y_min_global + 0.05 * (
                y_max_global - y_min_global if y_max_global != y_min_global else 1.0
            )
            y_range = y_max_global - y_min_global if y_max_global != y_min_global else 1.0
            ax.plot(
                [x0, x0],
                [0, 1],
                color="black",
                linewidth=1.0,
                solid_capstyle="butt",
            )
            ax.plot(
                [x0, x1],
                [y_bar, y_bar],
                color="black",
                linewidth=1.0,
                solid_capstyle="butt",
            )
            ax.text(
                x0 - 0.5,
                0.5,
                "1 spk",
                fontsize=6,
                fontname="Arial",
                ha="right",
                va="center",
                rotation=90,
            )
            ax.text(
                (x0 + x1) / 2,
                y_bar - 0.05 * y_range,
                "10 ms",
                fontsize=6,
                fontname="Arial",
                ha="center",
                va="top",
            )
        ax.set_title(
            title,
            fontsize=6,
            fontname="Arial",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        if plotted:
            ax.legend(fontsize=5, frameon=False)
        else:
            ax.text(
                0.5,
                0.5,
                "No data",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=6,
                fontname="Arial",
            )

    axes[1, 1].axis("off")
    axes[1, 3].axis("off")

    plt.tight_layout()
    if output_path is not None:
        plt.rcParams["svg.fonttype"] = "none"
        fig.savefig(output_path, dpi=300)
    if show:
        plt.show()
    return fig, axes


def plot_spike_shape_grand_average(
    time_ms,
    grand_stack,
    color_map=None,
    fig_size=(3, 2),
    show=True,
    output_path=None,
):
    """
    Plot grand-average spike/burst shapes for 4 conditions (run/rest × in/out PF).

    Parameters
    ----------
    time_ms : array-like or dict
        If array-like, uses the same time base for both simple and complex shapes.
        If dict, must contain keys {'simple', 'complex'} mapping to separate time bases.
    grand_stack : dict
        Must contain `grand_stack['simple'][cond]` and `grand_stack['complex'][cond]`, where each is
        a list of per-cell waveforms (each waveform is a 1D array matching that type's time base).
    """
    if color_map is None:
        color_map = {"in": "magenta", "out": "gray"}

    time_ms_by_type = time_ms
    multi_time = isinstance(time_ms_by_type, dict)
    if multi_time:
        if not (("simple" in time_ms_by_type) and ("complex" in time_ms_by_type)):
            raise ValueError("time_ms dict must contain keys {'simple', 'complex'}")
        fig, axes = plt.subplots(2, 2, figsize=fig_size, sharex=False, sharey=True)
    else:
        fig, axes = plt.subplots(2, 2, figsize=fig_size, sharex=True, sharey=True)

    state_map = [("run", "Locomotion"), ("rest", "Quiet")]
    type_map = [("simple", "Simple spikes"), ("complex", "Complex bursts")]

    y_min_global = np.inf
    y_max_global = -np.inf
    for state_key, _ in state_map:
        for spike_type, _ in type_map:
            for pf_key in ("in", "out"):
                cond = f"{state_key}_{pf_key}"
                stack = grand_stack.get(spike_type, {}).get(cond, [])
                if not stack:
                    continue
                arr = np.vstack(stack)
                valid_rows = np.any(np.isfinite(arr), axis=1)
                if not np.any(valid_rows):
                    continue
                mean_wave = np.nanmean(arr, axis=0)
                if np.any(np.isfinite(mean_wave)):
                    y_min_global = min(y_min_global, np.nanmin(mean_wave))
                    y_max_global = max(y_max_global, np.nanmax(mean_wave))
    if not np.isfinite(y_min_global) or not np.isfinite(y_max_global):
        y_min_global = 0.0
        y_max_global = 1.0
    y_min_global = min(y_min_global, 0.0)
    y_max_global = max(y_max_global, 1.0)

    for row_idx, (state_key, state_label) in enumerate(state_map):
        for col_idx, (spike_type, type_label) in enumerate(type_map):
            local_time_ms = time_ms_by_type[spike_type] if multi_time else time_ms_by_type
            ax = axes[row_idx, col_idx]
            plotted = False
            pre_candidates = []
            for pf_key, pf_label in (("in", "In PF"), ("out", "Out PF")):
                cond = f"{state_key}_{pf_key}"
                stack = grand_stack.get(spike_type, {}).get(cond, [])
                if not stack:
                    continue
                arr = np.vstack(stack)
                valid_rows = np.any(np.isfinite(arr), axis=1)
                if not np.any(valid_rows):
                    continue
                mean_wave = np.nanmean(arr, axis=0)
                sem_wave = np.nanstd(arr, axis=0) / np.sqrt(arr.shape[0])
                n_cells = int(np.sum(valid_rows))
                if row_idx == 0 and col_idx == 0:
                    label = f"{pf_label} (n={n_cells})"
                else:
                    label = f"n={n_cells}"
                # Plot SEM shading
                ax.fill_between(
                    local_time_ms,
                    mean_wave - sem_wave,
                    mean_wave + sem_wave,
                    color=color_map[pf_key],
                    alpha=0.2,
                    linewidth=0,
                )
                ax.plot(
                    local_time_ms,
                    mean_wave,
                    color=color_map[pf_key],
                    linewidth=0.8,
                    label=label,
                )
                pre_mask = local_time_ms < 0
                if np.any(pre_mask):
                    pre_candidates.append(mean_wave[pre_mask])
                plotted = True

            if pre_candidates:
                pre_stack = np.hstack(pre_candidates)
                y_pre_min = np.nanmin(pre_stack)
                if not np.isfinite(y_pre_min):
                    y_pre_min = y_min_global
            else:
                y_pre_min = y_min_global
            y_line_low = min(y_pre_min, 1.0)
            y_line_high = max(y_min_global, 1.0)
            ax.set_xlim(local_time_ms[0], local_time_ms[-1])
            ax.set_ylim(y_min_global, y_max_global)
            ax.plot(
                [0, 0],
                [y_line_low, y_line_high],
                color="black",
                linestyle="--",
                linewidth=0.5,
                alpha=0.6,
            )
            if row_idx == 0 and col_idx == 0:
                x_range = local_time_ms[-1] - local_time_ms[0]
                x0 = local_time_ms[0] + 0.05 * x_range
                x1 = x0 + 10.0
                if x1 > local_time_ms[-1]:
                    x1 = local_time_ms[-1] - 0.05 * x_range
                    x0 = x1 - 10.0
                y_bar = y_min_global + 0.05 * (
                    y_max_global - y_min_global if y_max_global != y_min_global else 1.0
                )
                y_range = y_max_global - y_min_global if y_max_global != y_min_global else 1.0
                ax.plot(
                    [x0, x0],
                    [0, 1],
                    color="black",
                    linewidth=1.0,
                    solid_capstyle="butt",
                )
                ax.plot(
                    [x0, x1],
                    [y_bar, y_bar],
                    color="black",
                    linewidth=1.0,
                    solid_capstyle="butt",
                )
                ax.text(
                    x0 - 0.5,
                    0.5,
                    "1 spk",
                    fontsize=6,
                    fontname="Arial",
                    ha="right",
                    va="center",
                    rotation=90,
                )
                ax.text(
                    (x0 + x1) / 2,
                    y_bar - 0.05 * y_range,
                    "10 ms",
                    fontsize=6,
                    fontname="Arial",
                    ha="center",
                    va="top",
                )
            ax.set_title(
                f"{type_label} - {state_label}",
                fontsize=6,
                fontname="Arial",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            if plotted:
                ax.legend(fontsize=5, frameon=False)
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=6,
                    fontname="Arial",
                )

    plt.tight_layout()
    if output_path is not None:
        plt.rcParams["svg.fonttype"] = "none"
        fig.savefig(output_path, dpi=300)
    if show:
        plt.show()
    return fig, axes

def _compute_moving_epochs(
    speed,
    frame_rate,
    kernel_size=21,
    filter_type="median",
    speed_threshold=2,
    min_duration_s=0.5,
    merge_gap_s=1.0,
    speed_max = 60
):
    speed = np.asarray(speed, dtype=float)
    speed[speed>speed_max] = np.nan
    # interpolate missing speed
    speed = interpolate_nan_segment(speed)
    if filter_type == "median":
        if kernel_size % 2 == 0:
            kernel_size += 1
        speed_smooth = medfilt(speed, kernel_size=kernel_size)
    elif filter_type == "boxcar":
        kernel_size = int(kernel_size)
        if kernel_size < 1:
            raise ValueError("kernel_size must be >= 1 for boxcar filter")
        kernel = np.ones(kernel_size, dtype=float) / kernel_size
        speed_smooth = np.convolve(speed, kernel, mode="same")
    else:
        raise ValueError("filter_type must be 'median' or 'boxcar'")

    moving_mask = speed_smooth > speed_threshold
    min_frames = int(round(min_duration_s * frame_rate))
    merge_gap = int(round(merge_gap_s * frame_rate))

    starts = []
    ends = []
    in_run = False
    for i, is_moving in enumerate(moving_mask):
        if is_moving and not in_run:
            starts.append(i)
            in_run = True
        elif not is_moving and in_run:
            ends.append(i - 1)
            in_run = False
    if in_run:
        ends.append(len(moving_mask) - 1)

    epochs = []
    for start, end in zip(starts, ends):
        if (end - start + 1) >= min_frames:
            epochs.append((start, end))

    merged_epochs = []
    for start, end in epochs:
        if not merged_epochs:
            merged_epochs.append([start, end])
            continue
        prev_start, prev_end = merged_epochs[-1]
        if (start - prev_end - 1) < merge_gap:
            merged_epochs[-1][1] = end
        else:
            merged_epochs.append([start, end])

    moving_idx = []
    for start, end in merged_epochs:
        moving_idx.append(np.arange(start, end + 1, dtype=int))
    if moving_idx:
        moving_idx = np.concatenate(moving_idx)
    else:
        moving_idx = np.array([], dtype=int)

    return speed_smooth, merged_epochs, moving_idx


# ============================================================
# Statistics Collection Helper Functions
# ============================================================

def calculate_spatial_information_bits_per_spike(firing_rate_map, occupancy_map):
    """Calculates spatial information in bits per spike.
    
    Parameters
    ----------
    firing_rate_map : ndarray
        2D array of firing rates per spatial bin.
    occupancy_map : ndarray
        2D array of occupancy (time spent) per spatial bin.
    
    Returns
    -------
    float
        Spatial information in bits/spike.
    """
    firing_rate_map = np.asarray(firing_rate_map, dtype=float)
    occupancy_map = np.asarray(occupancy_map, dtype=float)
    
    # Mask out NaN values and zero occupancy bins
    valid_mask = np.isfinite(firing_rate_map) & np.isfinite(occupancy_map) & (occupancy_map > 0)
    if not np.any(valid_mask):
        return np.nan
    
    rate = firing_rate_map[valid_mask]
    occ = occupancy_map[valid_mask]
    
    epsilon = 1e-15
    occupancy_prob = occ / (np.sum(occ) + epsilon)
    mean_firing_rate = np.sum(rate * occupancy_prob)
    
    if mean_firing_rate <= epsilon:
        return 0.0
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rate_ratio = rate / mean_firing_rate
        log_term = np.log2(rate_ratio + epsilon)
    
    info_per_bin = occupancy_prob * rate_ratio * log_term
    valid_info = info_per_bin[np.isfinite(info_per_bin)]
    
    return np.sum(valid_info) if valid_info.size > 0 else np.nan


def calculate_rate_map_coherence(rate_map):
    """Compute Fisher-z-transformed spatial coherence from a 2D rate map."""
    rate_map = np.asarray(rate_map, dtype=float)
    if rate_map.ndim != 2 or rate_map.size == 0:
        return np.nan

    center_vals = []
    neighbor_means = []
    n_rows, n_cols = rate_map.shape

    for row in range(n_rows):
        for col in range(n_cols):
            center = rate_map[row, col]
            if not np.isfinite(center):
                continue

            row_start = max(0, row - 1)
            row_end = min(n_rows, row + 2)
            col_start = max(0, col - 1)
            col_end = min(n_cols, col + 2)
            window = np.asarray(rate_map[row_start:row_end, col_start:col_end], dtype=float)
            neighbor_mask = np.ones(window.shape, dtype=bool)
            neighbor_mask[row - row_start, col - col_start] = False
            neighbor_vals = window[neighbor_mask & np.isfinite(window)]
            if neighbor_vals.size == 0:
                continue

            center_vals.append(float(center))
            neighbor_means.append(float(np.mean(neighbor_vals)))

    if len(center_vals) < 3:
        return np.nan

    center_arr = np.asarray(center_vals, dtype=float)
    neighbor_arr = np.asarray(neighbor_means, dtype=float)
    if np.std(center_arr) == 0 or np.std(neighbor_arr) == 0:
        return np.nan

    corr = np.corrcoef(center_arr, neighbor_arr)[0, 1]
    if not np.isfinite(corr):
        return np.nan

    eps = 1e-6
    corr = float(np.clip(corr, -1.0 + eps, 1.0 - eps))
    return float(np.arctanh(corr))


def calculate_rate_map_sparsity(rate_map, occupancy_map):
    """Compute occupancy-corrected sparsity from a 2D rate map."""
    rate_map = np.asarray(rate_map, dtype=float)
    occupancy_map = np.asarray(occupancy_map, dtype=float)
    if rate_map.shape != occupancy_map.shape or rate_map.ndim != 2 or rate_map.size == 0:
        return np.nan

    valid_mask = np.isfinite(rate_map) & np.isfinite(occupancy_map) & (occupancy_map > 0)
    if not np.any(valid_mask):
        return np.nan

    rates = rate_map[valid_mask]
    occupancy = occupancy_map[valid_mask]
    occupancy_sum = float(np.sum(occupancy))
    if occupancy_sum <= 0:
        return np.nan

    occ_prob = occupancy / occupancy_sum
    weighted_mean = float(np.sum(occ_prob * rates))
    denom = float(np.sum(occ_prob * (rates ** 2)))
    if denom <= 0 or not np.isfinite(denom):
        return np.nan

    return float((weighted_mean ** 2) / denom)


def get_place_field_sizes_cm(place_field_mask, bin_size_cm):
    """Get sizes (in cm²) of each place field.
    
    Parameters
    ----------
    place_field_mask : ndarray
        2D boolean array indicating place field bins.
    bin_size_cm : float
        Size of each spatial bin in cm.
    
    Returns
    -------
    list
        List of place field sizes in cm².
    """
    labeled_array, num_features = label(place_field_mask)
    return [np.sum(labeled_array == i) * bin_size_cm ** 2 for i in range(1, num_features + 1)]


def compute_in_out_ratio(spike_times, x_neural, y_neural, place_field_mask, bins, 
                         moving_mask, quiet_mask, frame_rate):
    """Compute in/out firing rate ratio for spikes.
    
    Parameters
    ----------
    spike_times : array-like
        Spike frame indices.
    x_neural, y_neural : ndarray
        Position coordinates.
    place_field_mask : ndarray
        2D boolean array indicating place field bins.
    bins : list
        [x_bins, y_bins] for spatial binning.
    moving_mask, quiet_mask : ndarray
        Boolean masks for locomotion and quiet states.
    frame_rate : float
        Sampling rate in Hz.
    
    Returns
    -------
    dict
        Dictionary with keys: loco_in, loco_out, loco_ratio, quiet_in, quiet_out, quiet_ratio.
    """
    if spike_times is None or len(spike_times) == 0:
        return {k: np.nan for k in ['loco_in', 'loco_out', 'loco_ratio', 'quiet_in', 'quiet_out', 'quiet_ratio']}
    
    spike_times = np.asarray(spike_times, dtype=int)
    n_frames = len(x_neural)
    spike_times = spike_times[(spike_times >= 0) & (spike_times < n_frames)]
    
    xbin = np.digitize(x_neural, bins[0]) - 1
    ybin = np.digitize(y_neural, bins[1]) - 1
    in_bounds = (xbin >= 0) & (xbin < place_field_mask.shape[0]) & (ybin >= 0) & (ybin < place_field_mask.shape[1])
    inside_pf = np.zeros(n_frames, dtype=bool)
    inside_pf[in_bounds] = place_field_mask[xbin[in_bounds], ybin[in_bounds]]
    
    results = {}
    for state_name, state_mask in [('loco', moving_mask), ('quiet', quiet_mask)]:
        time_in = np.sum(state_mask & inside_pf) / frame_rate
        time_out = np.sum(state_mask & ~inside_pf) / frame_rate
        spikes_in_state = spike_times[state_mask[spike_times]]
        n_in = np.sum(inside_pf[spikes_in_state])
        n_out = np.sum(~inside_pf[spikes_in_state])
        rate_in = n_in / time_in if time_in > 0 else np.nan
        rate_out = n_out / time_out if time_out > 0 else np.nan
        results[f'{state_name}_in'], results[f'{state_name}_out'] = rate_in, rate_out
        results[f'{state_name}_ratio'] = rate_in / rate_out if rate_out > 0 else np.nan
    return results


def compute_theta_slow_in_out(trace, x_neural, y_neural, place_field_mask, bins, 
                               moving_mask, quiet_mask, frame_rate, theta_freqs, slow_freqs):
    """Compute theta amplitude and slow Vm inside/outside place field.
    
    Parameters
    ----------
    trace : array-like
        Voltage trace.
    x_neural, y_neural : ndarray
        Position coordinates.
    place_field_mask : ndarray
        2D boolean array indicating place field bins.
    bins : list
        [x_bins, y_bins] for spatial binning.
    moving_mask, quiet_mask : ndarray
        Boolean masks for locomotion and quiet states.
    frame_rate : float
        Sampling rate in Hz.
    theta_freqs : tuple
        (low, high) frequency bounds for theta band.
    slow_freqs : float
        Cutoff frequency for slow Vm.
    
    Returns
    -------
    dict
        Dictionary with theta and slow Vm values inside/outside place field.
    """
    trace = np.asarray(trace, dtype=float)
    if np.all(np.isnan(trace)):
        return {k: np.nan for k in ['theta_loco_in', 'theta_loco_out', 'theta_quiet_in', 'theta_quiet_out',
                                     'slow_loco_in', 'slow_loco_out', 'slow_quiet_in', 'slow_quiet_out']}
    
    trace = interpolate_nan_segment(trace)
    theta_vm = bandpass_filter(trace, theta_freqs[0], theta_freqs[1], frame_rate, order=5)
    slow_vm = lowpass_filter(trace, slow_freqs, frame_rate, order=5)
    theta_amp = np.abs(signal.hilbert(theta_vm))
    
    n_frames = len(x_neural)
    xbin = np.digitize(x_neural, bins[0]) - 1
    ybin = np.digitize(y_neural, bins[1]) - 1
    in_bounds = (xbin >= 0) & (xbin < place_field_mask.shape[0]) & (ybin >= 0) & (ybin < place_field_mask.shape[1])
    inside_pf = np.zeros(n_frames, dtype=bool)
    inside_pf[in_bounds] = place_field_mask[xbin[in_bounds], ybin[in_bounds]]
    
    results = {}
    for state_name, state_mask in [('loco', moving_mask), ('quiet', quiet_mask)]:
        mask_in, mask_out = state_mask & inside_pf, state_mask & ~inside_pf
        results[f'theta_{state_name}_in'] = np.nanmean(theta_amp[mask_in]) if np.any(mask_in) else np.nan
        results[f'theta_{state_name}_out'] = np.nanmean(theta_amp[mask_out]) if np.any(mask_out) else np.nan
        results[f'slow_{state_name}_in'] = np.nanmean(slow_vm[mask_in]) if np.any(mask_in) else np.nan
        results[f'slow_{state_name}_out'] = np.nanmean(slow_vm[mask_out]) if np.any(mask_out) else np.nan
    return results


# =============================================================================
# Arena Linearization Functions (Rounded Corridor)
# =============================================================================

def linearize_position_rounded_corridor(
    x, y,
    width=35.5, height=20.0,
    corner_radius=5.0,
    corridor_width=5.0,
    origin='center'
):
    """
    Convert 2D position to linearized 1D position along a rounded-corner corridor.

    The corridor follows the perimeter of a rounded rectangle. Points are projected
    to the nearest wall segment (straight edges or corner arcs).

    Parameters
    ----------
    x, y : array-like
        2D position coordinates.
    width, height : float
        Arena dimensions in cm.
    corner_radius : float
        Radius of rounded corners in cm.
    corridor_width : float
        Width of the corridor from the wall (for determining if point is in corridor).
    origin : str
        Coordinate system origin: 'center' (0,0 at center) or 'corner' (0,0 at bottom-left).

    Returns
    -------
    dict with keys:
        'linear_pos' : ndarray
            Linearized position (cm along perimeter, starting from bottom-center, going CW).
        'distance_to_wall' : ndarray
            Distance from each point to the nearest wall.
        'in_corridor' : ndarray (bool)
            Whether each point is within the corridor.
        'perimeter' : float
            Total perimeter length.
        'segment_type' : ndarray (str)
            Which segment each point projects to ('bottom', 'right', 'top', 'left',
            'corner_br', 'corner_tr', 'corner_tl', 'corner_bl').
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Convert to center-origin coordinates if needed
    if origin == 'corner':
        x = x - width / 2
        y = y - height / 2

    # Inner rectangle (the "hub" for corners)
    iw = width - 2 * corner_radius  # inner width
    ih = height - 2 * corner_radius  # inner height
    half_iw = iw / 2
    half_ih = ih / 2
    half_w = width / 2
    half_h = height / 2

    # Perimeter segments (clockwise from bottom-center)
    # Bottom: from (0, -h/2) to (iw/2, -h/2), length = iw/2
    # Corner BR: quarter circle at (iw/2, -ih/2), angle -90 to 0, arc = pi*R/2
    # Right: from (w/2, -ih/2) to (w/2, ih/2), length = ih
    # Corner TR: quarter circle at (iw/2, ih/2), angle 0 to 90, arc = pi*R/2
    # Top: from (iw/2, h/2) to (-iw/2, h/2), length = iw
    # Corner TL: quarter circle at (-iw/2, ih/2), angle 90 to 180, arc = pi*R/2
    # Left: from (-w/2, ih/2) to (-w/2, -ih/2), length = ih
    # Corner BL: quarter circle at (-iw/2, -ih/2), angle 180 to 270, arc = pi*R/2
    # Bottom (second half): from (-iw/2, -h/2) to (0, -h/2), length = iw/2

    arc_len = np.pi * corner_radius / 2
    straight_bottom_half = iw / 2
    straight_right = ih
    straight_top = iw
    straight_left = ih

    # Cumulative lengths for each segment start
    seg_starts = {
        'bottom_right': 0,
        'corner_br': straight_bottom_half,
        'right': straight_bottom_half + arc_len,
        'corner_tr': straight_bottom_half + arc_len + straight_right,
        'top': straight_bottom_half + 2 * arc_len + straight_right,
        'corner_tl': straight_bottom_half + 2 * arc_len + straight_right + straight_top,
        'left': straight_bottom_half + 3 * arc_len + straight_right + straight_top,
        'corner_bl': straight_bottom_half + 3 * arc_len + 2 * straight_right + straight_top,
        'bottom_left': straight_bottom_half + 4 * arc_len + 2 * straight_right + straight_top,
    }
    perimeter = 2 * iw + 2 * ih + 4 * arc_len

    # Corner centers
    corner_centers = {
        'br': (half_iw, -half_ih),
        'tr': (half_iw, half_ih),
        'tl': (-half_iw, half_ih),
        'bl': (-half_iw, -half_ih),
    }

    n = len(x)
    linear_pos = np.zeros(n, dtype=float)
    distance_to_wall = np.zeros(n, dtype=float)
    segment_type = np.empty(n, dtype=object)

    for i in range(n):
        xi, yi = x[i], y[i]

        if not np.isfinite(xi) or not np.isfinite(yi):
            linear_pos[i] = np.nan
            distance_to_wall[i] = np.nan
            segment_type[i] = 'nan'
            continue

        # Clamp point to inner rectangle to determine projection zone
        cx = np.clip(xi, -half_iw, half_iw)
        cy = np.clip(yi, -half_ih, half_ih)

        # Vector from clamped point to actual point
        dx = xi - cx
        dy = yi - cy

        # Determine which zone the point is in
        if cx == xi and cy == yi:
            # Point is inside inner rectangle - project to nearest wall
            # Find distances to all four walls
            d_bottom = yi - (-half_h)
            d_top = half_h - yi
            d_left = xi - (-half_w)
            d_right = half_w - xi

            min_dist = min(d_bottom, d_top, d_left, d_right)

            if min_dist == d_bottom:
                # Project to bottom wall
                distance_to_wall[i] = d_bottom
                if xi >= 0:
                    linear_pos[i] = xi
                    segment_type[i] = 'bottom_right'
                else:
                    linear_pos[i] = perimeter + xi  # wrap around
                    segment_type[i] = 'bottom_left'
            elif min_dist == d_right:
                # Project to right wall
                distance_to_wall[i] = d_right
                linear_pos[i] = seg_starts['right'] + (yi + half_ih)
                segment_type[i] = 'right'
            elif min_dist == d_top:
                # Project to top wall
                distance_to_wall[i] = d_top
                linear_pos[i] = seg_starts['top'] + (half_iw - xi)
                segment_type[i] = 'top'
            else:
                # Project to left wall
                distance_to_wall[i] = d_left
                linear_pos[i] = seg_starts['left'] + (half_ih - yi)
                segment_type[i] = 'left'

        elif abs(dx) < 1e-9 and abs(dy) < 1e-9:
            # Exactly on inner rectangle boundary - shouldn't happen but handle it
            distance_to_wall[i] = corner_radius
            linear_pos[i] = 0
            segment_type[i] = 'boundary'

        else:
            # Point is outside inner rectangle - determine which wall/corner
            dist = np.sqrt(dx**2 + dy**2)

            if cx == half_iw and cy > -half_ih and cy < half_ih:
                # Right straight wall zone (between corners)
                distance_to_wall[i] = half_w - xi
                linear_pos[i] = seg_starts['right'] + (yi + half_ih)
                segment_type[i] = 'right'

            elif cx == -half_iw and cy > -half_ih and cy < half_ih:
                # Left straight wall zone
                distance_to_wall[i] = xi + half_w
                linear_pos[i] = seg_starts['left'] + (half_ih - yi)
                segment_type[i] = 'left'

            elif cy == half_ih and cx > -half_iw and cx < half_iw:
                # Top straight wall zone
                distance_to_wall[i] = half_h - yi
                linear_pos[i] = seg_starts['top'] + (half_iw - xi)
                segment_type[i] = 'top'

            elif cy == -half_ih and cx > -half_iw and cx < half_iw:
                # Bottom straight wall zone
                distance_to_wall[i] = yi + half_h
                if xi >= 0:
                    linear_pos[i] = xi
                    segment_type[i] = 'bottom_right'
                else:
                    linear_pos[i] = seg_starts['bottom_left'] + (xi + half_iw)
                    segment_type[i] = 'bottom_left'

            else:
                # Corner zone - find which corner
                if cx >= half_iw and cy <= -half_ih:
                    # Bottom-right corner
                    cc = corner_centers['br']
                    angle = np.arctan2(yi - cc[1], xi - cc[0])  # -pi/2 to 0
                    angle_normalized = angle + np.pi / 2  # 0 to pi/2
                    distance_to_wall[i] = corner_radius - dist
                    linear_pos[i] = seg_starts['corner_br'] + angle_normalized * corner_radius
                    segment_type[i] = 'corner_br'

                elif cx >= half_iw and cy >= half_ih:
                    # Top-right corner
                    cc = corner_centers['tr']
                    angle = np.arctan2(yi - cc[1], xi - cc[0])  # 0 to pi/2
                    distance_to_wall[i] = corner_radius - dist
                    linear_pos[i] = seg_starts['corner_tr'] + angle * corner_radius
                    segment_type[i] = 'corner_tr'

                elif cx <= -half_iw and cy >= half_ih:
                    # Top-left corner
                    cc = corner_centers['tl']
                    angle = np.arctan2(yi - cc[1], xi - cc[0])  # pi/2 to pi
                    angle_normalized = angle - np.pi / 2  # 0 to pi/2
                    distance_to_wall[i] = corner_radius - dist
                    linear_pos[i] = seg_starts['corner_tl'] + angle_normalized * corner_radius
                    segment_type[i] = 'corner_tl'

                else:
                    # Bottom-left corner
                    cc = corner_centers['bl']
                    angle = np.arctan2(yi - cc[1], xi - cc[0])  # -pi to -pi/2
                    angle_normalized = angle + np.pi  # 0 to pi/2
                    distance_to_wall[i] = corner_radius - dist
                    linear_pos[i] = seg_starts['corner_bl'] + angle_normalized * corner_radius
                    segment_type[i] = 'corner_bl'

    # Ensure linear_pos is in [0, perimeter)
    linear_pos = np.mod(linear_pos, perimeter)

    # Compute in_corridor
    in_corridor = distance_to_wall <= corridor_width

    return {
        'linear_pos': linear_pos,
        'distance_to_wall': distance_to_wall,
        'in_corridor': in_corridor,
        'perimeter': perimeter,
        'segment_type': segment_type,
        'params': {
            'width': width,
            'height': height,
            'corner_radius': corner_radius,
            'corridor_width': corridor_width,
        }
    }


def compute_linearization_bins(
    width=35.5, height=20.0,
    corner_radius=5.0,
    bin_size=1.5,
):
    """
    Compute bin edges and centers for linearized corridor.

    Parameters
    ----------
    width, height : float
        Arena dimensions in cm.
    corner_radius : float
        Radius of rounded corners in cm.
    bin_size : float
        Bin size in cm.

    Returns
    -------
    dict with keys:
        'bin_edges' : ndarray
            Bin edges along the perimeter.
        'bin_centers' : ndarray
            Bin centers along the perimeter.
        'perimeter' : float
            Total perimeter length.
        'n_bins' : int
            Number of bins.
    """
    iw = width - 2 * corner_radius
    ih = height - 2 * corner_radius
    arc_len = np.pi * corner_radius / 2
    perimeter = 2 * iw + 2 * ih + 4 * arc_len

    bin_edges = np.arange(0, perimeter + bin_size, bin_size)
    bin_edges[-1] = perimeter  # Ensure last edge is exactly perimeter
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    return {
        'bin_edges': bin_edges,
        'bin_centers': bin_centers,
        'perimeter': perimeter,
        'n_bins': len(bin_centers),
    }


def linear_pos_to_2d(
    linear_pos,
    width=35.5, height=20.0,
    corner_radius=5.0,
    wall_offset=0.0,
):
    """
    Convert linearized position back to 2D coordinates on the wall.

    Parameters
    ----------
    linear_pos : array-like
        Linearized position along the perimeter.
    width, height : float
        Arena dimensions in cm.
    corner_radius : float
        Radius of rounded corners in cm.
    wall_offset : float
        Offset from wall (positive = towards center). Default 0 = on wall.

    Returns
    -------
    x, y : ndarray
        2D coordinates (center-origin).
    """
    linear_pos = np.asarray(linear_pos, dtype=float)

    iw = width - 2 * corner_radius
    ih = height - 2 * corner_radius
    half_iw = iw / 2
    half_ih = ih / 2
    half_w = width / 2
    half_h = height / 2

    arc_len = np.pi * corner_radius / 2

    # Segment boundaries
    seg_starts = {
        'bottom_right': 0,
        'corner_br': half_iw,
        'right': half_iw + arc_len,
        'corner_tr': half_iw + arc_len + ih,
        'top': half_iw + 2 * arc_len + ih,
        'corner_tl': half_iw + 2 * arc_len + ih + iw,
        'left': half_iw + 3 * arc_len + ih + iw,
        'corner_bl': half_iw + 3 * arc_len + 2 * ih + iw,
        'bottom_left': half_iw + 4 * arc_len + 2 * ih + iw,
    }
    perimeter = 2 * iw + 2 * ih + 4 * arc_len

    # Corner centers
    corner_centers = {
        'br': (half_iw, -half_ih),
        'tr': (half_iw, half_ih),
        'tl': (-half_iw, half_ih),
        'bl': (-half_iw, -half_ih),
    }

    # Wrap linear_pos to [0, perimeter)
    linear_pos = np.mod(linear_pos, perimeter)

    n = len(linear_pos) if hasattr(linear_pos, '__len__') else 1
    if n == 1:
        linear_pos = np.array([linear_pos])

    x = np.zeros(n, dtype=float)
    y = np.zeros(n, dtype=float)

    wall_r = corner_radius - wall_offset

    for i in range(n):
        lp = linear_pos[i]

        if not np.isfinite(lp):
            x[i], y[i] = np.nan, np.nan
            continue

        if lp < seg_starts['corner_br']:
            # Bottom right segment
            x[i] = lp
            y[i] = -half_h + wall_offset

        elif lp < seg_starts['right']:
            # Bottom-right corner
            angle = (lp - seg_starts['corner_br']) / corner_radius - np.pi / 2
            cc = corner_centers['br']
            x[i] = cc[0] + wall_r * np.cos(angle)
            y[i] = cc[1] + wall_r * np.sin(angle)

        elif lp < seg_starts['corner_tr']:
            # Right segment
            x[i] = half_w - wall_offset
            y[i] = -half_ih + (lp - seg_starts['right'])

        elif lp < seg_starts['top']:
            # Top-right corner
            angle = (lp - seg_starts['corner_tr']) / corner_radius
            cc = corner_centers['tr']
            x[i] = cc[0] + wall_r * np.cos(angle)
            y[i] = cc[1] + wall_r * np.sin(angle)

        elif lp < seg_starts['corner_tl']:
            # Top segment
            x[i] = half_iw - (lp - seg_starts['top'])
            y[i] = half_h - wall_offset

        elif lp < seg_starts['left']:
            # Top-left corner
            angle = (lp - seg_starts['corner_tl']) / corner_radius + np.pi / 2
            cc = corner_centers['tl']
            x[i] = cc[0] + wall_r * np.cos(angle)
            y[i] = cc[1] + wall_r * np.sin(angle)

        elif lp < seg_starts['corner_bl']:
            # Left segment
            x[i] = -half_w + wall_offset
            y[i] = half_ih - (lp - seg_starts['left'])

        elif lp < seg_starts['bottom_left']:
            # Bottom-left corner
            angle = (lp - seg_starts['corner_bl']) / corner_radius + np.pi
            cc = corner_centers['bl']
            x[i] = cc[0] + wall_r * np.cos(angle)
            y[i] = cc[1] + wall_r * np.sin(angle)

        else:
            # Bottom left segment
            x[i] = -half_iw + (lp - seg_starts['bottom_left'])
            y[i] = -half_h + wall_offset

    return x, y


def plot_linearization_bin_map(
    width=35.5, height=20.0,
    corner_radius=5.0,
    corridor_width=5.0,
    bin_size=1.5,
    ax=None,
    cmap='hsv',
    show_bin_numbers=False,
    alpha=0.6,
    bin_values=None,
    vmin=None,
    vmax=None,
    colorbar=True,
    colorbar_label=None,
):
    """
    Visualize how each linearized bin maps to the 2D arena.

    Parameters
    ----------
    width, height : float
        Arena dimensions in cm.
    corner_radius : float
        Radius of rounded corners in cm.
    corridor_width : float
        Width of the corridor from the wall.
    bin_size : float
        Bin size for linearization.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    cmap : str
        Colormap name.
    show_bin_numbers : bool
        Whether to show bin numbers.
    alpha : float
        Alpha for bin patches.

    Returns
    -------
    fig, ax
    """
    import matplotlib.patches as mpatches
    from matplotlib.collections import PatchCollection

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    # Get bins
    bins_info = compute_linearization_bins(width, height, corner_radius, bin_size)
    bin_edges = bins_info['bin_edges']
    n_bins = bins_info['n_bins']
    perimeter = bins_info['perimeter']

    # Create colormap
    cmap_obj = plt.cm.get_cmap(cmap)
    if bin_values is not None:
        bin_values = np.asarray(bin_values, dtype=float)
        _vmin = vmin if vmin is not None else np.nanmin(bin_values)
        _vmax = vmax if vmax is not None else np.nanmax(bin_values)
        norm = plt.Normalize(_vmin, _vmax)
        colors = [cmap_obj(norm(bin_values[i])) if i < len(bin_values) else (0.5, 0.5, 0.5, 1) for i in range(n_bins)]
    else:
        norm = plt.Normalize(0, perimeter)
        colors = [cmap_obj(i / n_bins) for i in range(n_bins)]

    half_w = width / 2
    half_h = height / 2

    # Draw arena outline as rectangle
    rect = mpatches.Rectangle(
        (-half_w, -half_h), width, height,
        linewidth=2, edgecolor='black', facecolor='white', zorder=0
    )
    ax.add_patch(rect)

    # Helper: project corner-zone positions to the rectangle boundary
    iw = width - 2 * corner_radius
    ih = height - 2 * corner_radius
    half_iw = iw / 2
    half_ih = ih / 2
    arc_len = np.pi * corner_radius / 2
    _seg_starts = {
        'corner_br': half_iw,
        'right': half_iw + arc_len,
        'corner_tr': half_iw + arc_len + ih,
        'top': half_iw + 2 * arc_len + ih,
        'corner_tl': half_iw + 2 * arc_len + ih + iw,
        'left': half_iw + 3 * arc_len + ih + iw,
        'corner_bl': half_iw + 3 * arc_len + 2 * ih + iw,
        'bottom_left': half_iw + 4 * arc_len + 2 * ih + iw,
    }
    _corner_info = {
        'br': {'center': (half_iw, -half_ih), 'angle_start': -np.pi / 2, 'seg_start': _seg_starts['corner_br']},
        'tr': {'center': (half_iw, half_ih), 'angle_start': 0, 'seg_start': _seg_starts['corner_tr']},
        'tl': {'center': (-half_iw, half_ih), 'angle_start': np.pi / 2, 'seg_start': _seg_starts['corner_tl']},
        'bl': {'center': (-half_iw, -half_ih), 'angle_start': np.pi, 'seg_start': _seg_starts['corner_bl']},
    }

    def _project_to_rect(lp_arr):
        """Project linear positions to rectangle boundary (extends corners outward)."""
        lp_arr = np.mod(np.asarray(lp_arr, dtype=float), perimeter)
        # Get normal wall points first
        x_w, y_w = linear_pos_to_2d(lp_arr, width, height, corner_radius, wall_offset=0)
        if corner_radius <= 0:
            return x_w, y_w
        # For corner-zone points, project radially to the rectangle
        for key, info in _corner_info.items():
            cx, cy = info['center']
            seg_s = info['seg_start']
            seg_e = seg_s + arc_len
            for i in range(len(lp_arr)):
                lp = lp_arr[i]
                if lp >= seg_s and lp < seg_e:
                    angle = info['angle_start'] + (lp - seg_s) / corner_radius
                    cos_a, sin_a = np.cos(angle), np.sin(angle)
                    # Find intersection with rectangle boundary
                    t_vals = []
                    if abs(cos_a) > 1e-12:
                        wall_x = half_w if cos_a > 0 else -half_w
                        t_vals.append((wall_x - cx) / cos_a)
                    if abs(sin_a) > 1e-12:
                        wall_y = half_h if sin_a > 0 else -half_h
                        t_vals.append((wall_y - cy) / sin_a)
                    t = min(t for t in t_vals if t > 0)
                    x_w[i] = cx + t * cos_a
                    y_w[i] = cy + t * sin_a
        return x_w, y_w

    # Draw each bin as a colored region
    patches = []
    for i in range(n_bins):
        lp_start = bin_edges[i]
        lp_end = bin_edges[i + 1]

        # Sample points along this bin
        n_samples = max(3, int((lp_end - lp_start) / 0.5) + 1)
        lp_samples = np.linspace(lp_start, lp_end, n_samples)

        # Outer boundary: projected to rectangle at corners
        x_wall, y_wall = _project_to_rect(lp_samples)
        # Inner boundary: follows the rounded corridor
        x_inner, y_inner = linear_pos_to_2d(lp_samples, width, height, corner_radius, wall_offset=corridor_width)

        # For corner bins, add the rectangle corner point to close the polygon properly
        for key, info in _corner_info.items():
            seg_s = info['seg_start']
            seg_e = seg_s + arc_len
            if lp_start < seg_e and lp_end > seg_s:
                # This bin spans part of this corner — add corner vertex
                cx, cy = info['center']
                # Rectangle corner point
                corner_x = half_w if cx > 0 else -half_w
                corner_y = half_h if cy > 0 else -half_h
                # Check if the corner point is within the bin's angular range
                mid_angle = info['angle_start'] + np.pi / 4  # 45 deg = hits corner
                mid_lp = seg_s + (np.pi / 4) * corner_radius
                if lp_start <= mid_lp <= lp_end:
                    # Insert corner point at the right position in the wall array
                    frac = (mid_lp - lp_start) / (lp_end - lp_start)
                    idx = max(1, min(len(x_wall) - 1, int(frac * len(x_wall))))
                    x_wall = np.insert(x_wall, idx, corner_x)
                    y_wall = np.insert(y_wall, idx, corner_y)

        # Create polygon for this bin (wall to inner, then back)
        verts = np.concatenate([
            np.column_stack([x_wall, y_wall]),
            np.column_stack([x_inner[::-1], y_inner[::-1]])
        ])

        poly = mpatches.Polygon(verts, closed=True)
        patches.append(poly)

    # Add all patches
    pc = PatchCollection(patches, alpha=alpha, edgecolor='none')
    pc.set_facecolors(colors)
    ax.add_collection(pc)

    # Draw corridor boundary (inner edge)
    lp_dense = np.linspace(0, perimeter, 500)
    x_inner, y_inner = linear_pos_to_2d(lp_dense, width, height, corner_radius, wall_offset=corridor_width)
    ax.plot(x_inner, y_inner, 'k--', linewidth=1, alpha=0.5, label='Corridor boundary')

    # Add bin numbers if requested
    if show_bin_numbers:
        bin_centers = bins_info['bin_centers']
        x_mid, y_mid = linear_pos_to_2d(bin_centers, width, height, corner_radius, wall_offset=corridor_width/2)
        for i in range(n_bins):
            if np.isfinite(x_mid[i]) and np.isfinite(y_mid[i]):
                ax.text(x_mid[i], y_mid[i], str(i), fontsize=5, ha='center', va='center')

    ax.set_xlim(-half_w - 2, half_w + 2)
    ax.set_ylim(-half_h - 4, half_h + 2)
    ax.set_aspect('equal')
    ax.set_xlabel('X (cm)')
    ax.set_ylabel('Y (cm)')
    ax.set_title(f'Linearization Bin Map (W={corridor_width}cm, bin={bin_size}cm)\n'
                 f'{n_bins} bins, perimeter={perimeter:.1f}cm')

    # Add colorbar
    if colorbar:
        sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
        sm.set_array([])
        _cb_label = colorbar_label if colorbar_label is not None else (
            'Linear position (cm)' if bin_values is None else '')
        cbar = plt.colorbar(sm, ax=ax, label=_cb_label, shrink=0.8)

    return fig, ax


def plot_corridor_with_trajectory(
    x_neural, y_neural, speed, frame_rate,
    width=35.5, height=20.0,
    corner_radius=5.0,
    corridor_width=5.0,
    speed_threshold=3.0,
    kernel_size=51,
    filter_type='boxcar',
    min_duration_s=0.25,
    merge_gap_s=0.0,
    ax=None,
    trajectory_alpha=0.3,
    trajectory_linewidth=0.5,
    moving_color='#1f77b4',
    stationary_color='#cccccc',
    origin='corner',
    show_stats=True,
):
    """
    Plot the corridor with trajectory overlay and compute corridor occupancy.

    Parameters
    ----------
    x_neural, y_neural : ndarray
        Position coordinates.
    speed : ndarray
        Speed array.
    frame_rate : float
        Sampling rate in Hz.
    width, height : float
        Arena dimensions in cm.
    corner_radius : float
        Radius of rounded corners in cm.
    corridor_width : float
        Width of the corridor from the wall.
    speed_threshold : float
        Speed threshold for moving epochs.
    kernel_size : int
        Kernel size for speed smoothing.
    filter_type : str
        Filter type for speed smoothing.
    min_duration_s : float
        Minimum duration for moving epochs.
    merge_gap_s : float
        Gap to merge moving epochs.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    trajectory_alpha : float
        Alpha for trajectory lines.
    trajectory_linewidth : float
        Linewidth for trajectory.
    moving_color : str
        Color for moving segments.
    stationary_color : str
        Color for stationary segments.
    origin : str
        Coordinate origin: 'center' or 'corner'.
    show_stats : bool
        Whether to show statistics on plot.

    Returns
    -------
    dict with keys:
        'fig', 'ax' : matplotlib objects
        'corridor_time_moving' : float
            Time (s) spent in corridor during moving epochs.
        'total_time_moving' : float
            Total time (s) in moving epochs.
        'corridor_pct_moving' : float
            Percentage of moving time spent in corridor.
        'in_corridor' : ndarray (bool)
            Boolean mask for in-corridor frames.
        'moving_mask' : ndarray (bool)
            Boolean mask for moving frames.
    """
    import matplotlib.patches as mpatches

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    x = np.asarray(x_neural, dtype=float)
    y = np.asarray(y_neural, dtype=float)

    # Convert to center coordinates
    if origin == 'corner':
        x_center = x - width / 2
        y_center = y - height / 2
    else:
        x_center = x.copy()
        y_center = y.copy()

    half_w = width / 2
    half_h = height / 2

    # Draw arena as simple rectangle
    rect = mpatches.Rectangle(
        (-half_w, -half_h), width, height,
        linewidth=2, edgecolor='black', facecolor='#f0f0f0', zorder=0
    )
    ax.add_patch(rect)

    # Draw corridor region (shaded) — simple rectangular band
    # Outer boundary = arena walls, inner boundary = inset rectangle
    inner_half_w = half_w - corridor_width
    inner_half_h = half_h - corridor_width
    # Corridor = area between outer and inner rectangles
    outer_verts = np.array([
        [-half_w, -half_h], [half_w, -half_h], [half_w, half_h], [-half_w, half_h], [-half_w, -half_h]
    ])
    inner_verts = np.array([
        [-inner_half_w, -inner_half_h], [inner_half_w, -inner_half_h],
        [inner_half_w, inner_half_h], [-inner_half_w, inner_half_h], [-inner_half_w, -inner_half_h]
    ])
    from matplotlib.path import Path as MplPath
    outer_codes = [MplPath.MOVETO] + [MplPath.LINETO] * 3 + [MplPath.CLOSEPOLY]
    inner_codes = [MplPath.MOVETO] + [MplPath.LINETO] * 3 + [MplPath.CLOSEPOLY]
    corridor_path = MplPath(
        np.concatenate([outer_verts, inner_verts[::-1]]),
        np.concatenate([outer_codes, inner_codes])
    )
    corridor_patch = mpatches.PathPatch(corridor_path, facecolor='#cce5ff',
                                         edgecolor='none', alpha=0.5, zorder=1)
    ax.add_patch(corridor_patch)

    # Draw corridor inner boundary
    ax.plot(
        [-inner_half_w, inner_half_w, inner_half_w, -inner_half_w, -inner_half_w],
        [-inner_half_h, -inner_half_h, inner_half_h, inner_half_h, -inner_half_h],
        'b--', linewidth=1, alpha=0.7, label=f'Corridor ({corridor_width}cm)', zorder=2
    )

    # Compute moving epochs
    _, _, moving_idx = _compute_moving_epochs(
        speed, frame_rate,
        kernel_size=kernel_size,
        filter_type=filter_type,
        speed_threshold=speed_threshold,
        min_duration_s=min_duration_s,
        merge_gap_s=merge_gap_s
    )

    moving_mask = np.zeros(len(x), dtype=bool)
    if moving_idx is not None and len(moving_idx) > 0:
        moving_mask[np.asarray(moving_idx, dtype=int)] = True

    # Compute in_corridor using simple rectangular distance to nearest wall
    dist_to_wall = np.minimum(
        np.minimum(x_center - (-half_w), half_w - x_center),
        np.minimum(y_center - (-half_h), half_h - y_center)
    )
    in_corridor = dist_to_wall <= corridor_width

    # Plot trajectory
    # Stationary segments
    stat_mask = ~moving_mask
    ax.plot(x_center[stat_mask], y_center[stat_mask], '.',
            color=stationary_color, markersize=0.5, alpha=trajectory_alpha,
            zorder=3, label='Stationary')

    # Moving segments
    ax.plot(x_center[moving_mask], y_center[moving_mask], '.',
            color=moving_color, markersize=0.5, alpha=trajectory_alpha,
            zorder=4, label='Moving')

    # Compute statistics
    total_time_moving = np.sum(moving_mask) / frame_rate
    corridor_time_moving = np.sum(moving_mask & in_corridor) / frame_rate
    corridor_pct_moving = 100 * corridor_time_moving / total_time_moving if total_time_moving > 0 else 0

    # Add stats text
    if show_stats:
        stats_text = (f'Moving time in corridor: {corridor_time_moving:.1f}s / {total_time_moving:.1f}s '
                     f'({corridor_pct_moving:.1f}%)')
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=8,
                verticalalignment='top', fontname='Arial',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlim(-half_w - 1, half_w + 1)
    ax.set_ylim(-half_h - 1, half_h + 1)
    ax.set_aspect('equal')
    ax.set_xlabel('X (cm)')
    ax.set_ylabel('Y (cm)')
    ax.set_title(f'Trajectory in Corridor (W={corridor_width}cm)')
    ax.legend(loc='lower right', fontsize=7)

    return {
        'fig': fig,
        'ax': ax,
        'corridor_time_moving': corridor_time_moving,
        'total_time_moving': total_time_moving,
        'corridor_pct_moving': corridor_pct_moving,
        'in_corridor': in_corridor,
        'moving_mask': moving_mask,
        'dist_to_wall': dist_to_wall,
    }


def compute_running_direction(
    centered_pos,
    perimeter,
    smooth_window=5,
):
    """
    Compute running direction (CW/CCW) at each frame based on velocity of centered position.

    Parameters
    ----------
    centered_pos : ndarray
        Position centered at PF peak (0 = PF center).
    perimeter : float
        Total perimeter length.
    smooth_window : int
        Window size for smoothing the velocity.

    Returns
    -------
    dict with keys:
        'direction' : ndarray
            +1 = CCW (position increasing), -1 = CW (position decreasing), 0 = stationary/invalid
        'velocity' : ndarray
            Smoothed velocity of centered position.
        'valid_mask' : ndarray (bool)
            True for frames with valid direction.
    """
    from scipy.ndimage import uniform_filter1d

    centered_pos = np.asarray(centered_pos, dtype=float)
    n_frames = len(centered_pos)

    # Track valid frames (non-NaN)
    valid_mask = np.isfinite(centered_pos)

    # Interpolate NaN values for gradient computation
    pos_interp = centered_pos.copy()
    if not np.all(valid_mask):
        # Linear interpolation for NaN values
        valid_idx = np.where(valid_mask)[0]
        if len(valid_idx) >= 2:
            pos_interp = np.interp(
                np.arange(n_frames),
                valid_idx,
                centered_pos[valid_idx]
            )
        else:
            # Not enough valid points, return zeros
            return {
                'direction': np.zeros(n_frames, dtype=int),
                'velocity': np.zeros(n_frames, dtype=float),
                'valid_mask': valid_mask,
            }

    # Compute velocity using gradient (centered differences)
    velocity = np.gradient(pos_interp)

    # Handle wrap-around: if |velocity| > perimeter/2, it's a wrap-around artifact
    velocity[velocity > perimeter / 2] -= perimeter
    velocity[velocity < -perimeter / 2] += perimeter

    # Smooth to reduce noise
    if smooth_window > 1:
        velocity_smooth = uniform_filter1d(velocity, smooth_window, mode='nearest')
    else:
        velocity_smooth = velocity.copy()

    # Direction: +1 = CCW (increasing), -1 = CW (decreasing)
    direction = np.sign(velocity_smooth).astype(int)

    # Set direction to 0 for originally invalid frames
    direction[~valid_mask] = 0

    return {
        'direction': direction,
        'velocity': velocity_smooth,
        'valid_mask': valid_mask,
    }


def detect_valid_pf_passes(
    centered_pos,
    approach_threshold=5.0,
    exit_threshold=None,
    min_frames=10,
):
    """
    Detect valid passes through the place field region.

    A valid pass is defined as an approach that reaches within approach_threshold
    of the PF center (position 0).

    Parameters
    ----------
    centered_pos : ndarray
        Position centered at PF peak (0 = PF center).
    approach_threshold : float
        Maximum distance from center to count as a valid approach (default 5 cm).
    exit_threshold : float, optional
        Distance from center that defines "outside" the PF region.
        If None, uses 3 * approach_threshold.
    min_frames : int
        Minimum number of frames for a valid pass.

    Returns
    -------
    dict with keys:
        'valid_mask' : ndarray (bool)
            True for frames that are part of a valid pass.
        'pass_info' : list of dict
            Each dict contains: 'start', 'end', 'direction', 'min_dist'
    """
    centered_pos = np.asarray(centered_pos, dtype=float)
    n_frames = len(centered_pos)

    if exit_threshold is None:
        exit_threshold = 3 * approach_threshold

    valid_mask = np.zeros(n_frames, dtype=bool)
    pass_info = []

    # State machine to detect passes
    in_approach = False
    approach_start = 0
    min_dist_in_approach = np.inf

    for i in range(n_frames):
        if not np.isfinite(centered_pos[i]):
            continue

        dist = np.abs(centered_pos[i])

        if not in_approach:
            # Start approach when entering the exit_threshold zone
            if dist < exit_threshold:
                in_approach = True
                approach_start = i
                min_dist_in_approach = dist
        else:
            # Update minimum distance
            if dist < min_dist_in_approach:
                min_dist_in_approach = dist

            # End approach when exiting the exit_threshold zone
            if dist >= exit_threshold:
                # Check if this was a valid pass (reached within approach_threshold)
                if min_dist_in_approach <= approach_threshold and (i - approach_start) >= min_frames:
                    # Mark all frames in this pass as valid
                    valid_mask[approach_start:i] = True

                    # Determine direction based on entry/exit positions
                    entry_pos = centered_pos[approach_start]
                    exit_pos = centered_pos[i-1] if i > 0 else centered_pos[i]
                    if entry_pos < 0 and exit_pos > 0:
                        direction = 'ccw'
                    elif entry_pos > 0 and exit_pos < 0:
                        direction = 'cw'
                    elif entry_pos < 0 and exit_pos < 0:
                        direction = 'turn_neg'  # Turn-around from negative side
                    else:
                        direction = 'turn_pos'  # Turn-around from positive side

                    pass_info.append({
                        'start': approach_start,
                        'end': i,
                        'direction': direction,
                        'min_dist': min_dist_in_approach,
                        'entry_pos': entry_pos,
                        'exit_pos': exit_pos,
                    })

                in_approach = False
                min_dist_in_approach = np.inf

    # Handle case where recording ends during an approach
    if in_approach and min_dist_in_approach <= approach_threshold:
        valid_mask[approach_start:n_frames] = True
        entry_pos = centered_pos[approach_start]
        pass_info.append({
            'start': approach_start,
            'end': n_frames,
            'direction': 'incomplete',
            'min_dist': min_dist_in_approach,
            'entry_pos': entry_pos,
            'exit_pos': np.nan,
        })

    return {
        'valid_mask': valid_mask,
        'pass_info': pass_info,
    }


def detect_pf_traversals(
    centered_pos,
    speed=None,
    core_radius=5.0,
    buffer_width=10.0,
    min_speed=2.0,
    reversal_threshold=5.0,
    verbose=False,
):
    """
    Detect CW and CCW traversals using a 3-Zone State Machine.

    Zones (for core_radius=5, buffer_width=10):
        Left Buffer:  [-15, -5)
        Core:         [-5, 5]
        Right Buffer: (5, 15]

    A traversal is valid when:
    1. Animal enters a buffer zone (determines direction: left=CW, right=CCW)
    2. Animal enters the core zone
    3. Animal exits the core (to either side)

    CW:  Left_Buffer -> Core -> exit (any side)
    CCW: Right_Buffer -> Core -> exit (any side)

    U-turns are detected when:
    - Animal reverses direction by more than reversal_threshold
    - Animal re-enters core after exiting to buffer

    Parameters
    ----------
    centered_pos : ndarray
        Position centered at PF peak (0 = PF center).
    speed : ndarray, optional
        Speed array for filtering by minimum speed.
    core_radius : float
        Radius of the core zone around center. Default 5 cm.
    buffer_width : float
        Width of buffer zones on each side. Default 10 cm.
    min_speed : float
        Minimum average speed for a valid traversal. Default 2 cm/s.
        Set to 0 to disable speed filtering.
    reversal_threshold : float
        Distance the animal must move backward to trigger U-turn detection.
        Default 5 cm. Set to a large value to disable reversal detection.
    verbose : bool
        If True, print debug info about traversals.

    Returns
    -------
    dict with keys:
        'valid_mask' : ndarray (bool)
            True for frames that are part of a valid traversal.
        'pass_info' : list of dict
            Each dict contains: 'start', 'end', 'direction'
    """
    centered_pos = np.asarray(centered_pos, dtype=float)
    n_frames = len(centered_pos)

    if speed is not None:
        speed = np.asarray(speed, dtype=float)
    else:
        speed = np.ones(n_frames)  # No speed filtering

    # Define zone boundaries
    b_left_outer = -(core_radius + buffer_width)   # e.g., -15
    b_left_inner = -core_radius                     # e.g., -5
    b_right_inner = core_radius                     # e.g., +5
    b_right_outer = core_radius + buffer_width      # e.g., +15

    valid_mask = np.zeros(n_frames, dtype=bool)
    pass_info = []

    # State tracking
    # 0 = Reset/Outside
    # 1 = In Left Buffer (Potential CW start)
    # 2 = In Right Buffer (Potential CCW start)
    # 3 = CW confirmed, in core
    # 4 = CCW confirmed, in core
    # 5 = CW confirmed, exited core to buffer (pass ends if re-entering core)
    # 6 = CCW confirmed, exited core to buffer (pass ends if re-entering core)
    current_state = 0
    start_idx = 0
    exit_core_idx = 0  # Track when we exited core (for potential new pass on U-turn)

    # For reversal detection: track furthest progress in expected direction
    # CW expects position to increase (left to right), CCW expects decrease
    max_pos_reached = -np.inf  # For CW passes
    min_pos_reached = np.inf   # For CCW passes
    peak_idx = 0  # Frame where max/min was reached

    def finalize_pass(end_idx, direction, exit_reason, local_start_idx=None):
        """Helper to finalize a confirmed traversal."""
        nonlocal start_idx
        s_idx = local_start_idx if local_start_idx is not None else start_idx
        segment_speed = np.nanmean(speed[s_idx:end_idx])
        if segment_speed >= min_speed:
            valid_mask[s_idx:end_idx] = True
            pass_info.append({
                'start': s_idx,
                'end': end_idx,
                'direction': direction,
                'avg_speed': segment_speed,
            })
            if verbose:
                print(f"  {direction.upper()} traversal: frames {s_idx}-{end_idx-1}, exit={exit_reason}, speed={segment_speed:.1f}")
            return True
        elif verbose:
            print(f"  {direction.upper()} traversal REJECTED (low speed): frames {s_idx}-{end_idx-1}, speed={segment_speed:.1f}")
        return False

    if verbose:
        print(f"  Zone boundaries: Left[{b_left_outer:.0f},{b_left_inner:.0f}), Core[{b_left_inner:.0f},{b_right_inner:.0f}], Right({b_right_inner:.0f},{b_right_outer:.0f}]")

    # Track previous zone for U-turn detection
    prev_in_left_buffer = False
    prev_in_right_buffer = False

    for i in range(n_frames):
        pos = centered_pos[i]

        # Handle NaN - finalize any confirmed traversal, then reset
        if not np.isfinite(pos):
            if current_state in (3, 4, 5, 6):
                direction = 'cw' if current_state in (3, 5) else 'ccw'
                finalize_pass(i, direction, "NaN")
            current_state = 0
            prev_in_left_buffer = False
            prev_in_right_buffer = False
            continue

        # Check if outside outer boundaries
        if pos < b_left_outer or pos > b_right_outer:
            if current_state in (3, 4, 5, 6):
                direction = 'cw' if current_state in (3, 5) else 'ccw'
                exit_side = "left" if pos < b_left_outer else "right"
                finalize_pass(i, direction, exit_side)
            current_state = 0
            prev_in_left_buffer = False
            prev_in_right_buffer = False
            continue

        # Determine current zone
        in_core = b_left_inner <= pos <= b_right_inner
        in_left_buffer = b_left_outer <= pos < b_left_inner
        in_right_buffer = b_right_inner < pos <= b_right_outer

        # --- STATE MACHINE ---

        if current_state == 0:
            # Check for entries into buffers
            if in_left_buffer:
                current_state = 1  # Entered Left Buffer
                start_idx = i
            elif in_right_buffer:
                current_state = 2  # Entered Right Buffer
                start_idx = i

        elif current_state == 1:  # In Left Buffer (waiting to enter Core)
            if in_core:  # Entered Core - CW confirmed!
                current_state = 3
                max_pos_reached = pos  # Initialize progress tracking
                peak_idx = i

        elif current_state == 2:  # In Right Buffer (waiting to enter Core)
            if in_core:  # Entered Core - CCW confirmed!
                current_state = 4
                min_pos_reached = pos  # Initialize progress tracking
                peak_idx = i

        elif current_state == 3:  # CW confirmed, in core
            # CW expects position to increase (left to right)
            if pos > max_pos_reached:
                max_pos_reached = pos
                peak_idx = i
            # Check for reversal (moved back significantly)
            elif pos < max_pos_reached - reversal_threshold:
                # Reversal detected! End CW pass at peak, start CCW
                finalize_pass(peak_idx + 1, 'cw', "reversal")
                current_state = 4  # Start CCW tracking
                start_idx = peak_idx
                min_pos_reached = pos
                peak_idx = i
                if verbose:
                    print(f"    Reversal in core: CW ended at {peak_idx}, starting CCW")
            elif not in_core:  # Exited core to a buffer
                current_state = 5
                exit_core_idx = i  # Track when we exited for potential U-turn

        elif current_state == 4:  # CCW confirmed, in core
            # CCW expects position to decrease (right to left)
            if pos < min_pos_reached:
                min_pos_reached = pos
                peak_idx = i
            # Check for reversal (moved back significantly)
            elif pos > min_pos_reached + reversal_threshold:
                # Reversal detected! End CCW pass at peak, start CW
                finalize_pass(peak_idx + 1, 'ccw', "reversal")
                current_state = 3  # Start CW tracking
                start_idx = peak_idx
                max_pos_reached = pos
                peak_idx = i
                if verbose:
                    print(f"    Reversal in core: CCW ended at {peak_idx}, starting CW")
            elif not in_core:  # Exited core to a buffer
                current_state = 6
                exit_core_idx = i  # Track when we exited for potential U-turn

        elif current_state == 5:  # CW, exited core, in buffer
            # Continue tracking progress for reversal detection
            if pos > max_pos_reached:
                max_pos_reached = pos
                peak_idx = i
            # Check for reversal in buffer
            elif pos < max_pos_reached - reversal_threshold:
                # Reversal detected in buffer
                finalize_pass(peak_idx + 1, 'cw', "reversal")
                current_state = 4  # Start CCW tracking
                start_idx = peak_idx
                min_pos_reached = pos
                peak_idx = i
                if verbose:
                    print(f"    Reversal in buffer: CW ended at {peak_idx}, starting CCW")
            elif in_core:  # Re-entered core - U-turn detected!
                # End the CW pass at exit_core_idx (not including the buffer wandering)
                finalize_pass(exit_core_idx, 'cw', "U-turn")
                # Start tracking the U-turn segment as a potential new pass
                # The turn segment starts from when we exited core (buffer entry)
                if prev_in_right_buffer:
                    # Was in right buffer before re-entering -> CCW direction
                    current_state = 4  # CCW, in core
                    start_idx = exit_core_idx
                    min_pos_reached = pos
                    peak_idx = i
                    if verbose:
                        print(f"    U-turn: starting new CCW tracking from frame {exit_core_idx}")
                elif prev_in_left_buffer:
                    # Was in left buffer before re-entering -> CW direction
                    current_state = 3  # CW, in core
                    start_idx = exit_core_idx
                    max_pos_reached = pos
                    peak_idx = i
                    if verbose:
                        print(f"    U-turn: starting new CW tracking from frame {exit_core_idx}")
                else:
                    current_state = 0  # Shouldn't happen, but reset
            # Otherwise stay in state 5, waiting to leave outer boundary

        elif current_state == 6:  # CCW, exited core, in buffer
            # Continue tracking progress for reversal detection
            if pos < min_pos_reached:
                min_pos_reached = pos
                peak_idx = i
            # Check for reversal in buffer
            elif pos > min_pos_reached + reversal_threshold:
                # Reversal detected in buffer
                finalize_pass(peak_idx + 1, 'ccw', "reversal")
                current_state = 3  # Start CW tracking
                start_idx = peak_idx
                max_pos_reached = pos
                peak_idx = i
                if verbose:
                    print(f"    Reversal in buffer: CCW ended at {peak_idx}, starting CW")
            elif in_core:  # Re-entered core - U-turn detected!
                # End the CCW pass at exit_core_idx (not including the buffer wandering)
                finalize_pass(exit_core_idx, 'ccw', "U-turn")
                # Start tracking the U-turn segment as a potential new pass
                if prev_in_left_buffer:
                    # Was in left buffer before re-entering -> CW direction
                    current_state = 3  # CW, in core
                    start_idx = exit_core_idx
                    max_pos_reached = pos
                    peak_idx = i
                    if verbose:
                        print(f"    U-turn: starting new CW tracking from frame {exit_core_idx}")
                elif prev_in_right_buffer:
                    # Was in right buffer before re-entering -> CCW direction
                    current_state = 4  # CCW, in core
                    start_idx = exit_core_idx
                    min_pos_reached = pos
                    peak_idx = i
                    if verbose:
                        print(f"    U-turn: starting new CCW tracking from frame {exit_core_idx}")
                else:
                    current_state = 0  # Shouldn't happen, but reset
            # Otherwise stay in state 6, waiting to leave outer boundary

        # Update previous zone tracking
        prev_in_left_buffer = in_left_buffer
        prev_in_right_buffer = in_right_buffer

    if verbose:
        n_cw = sum(1 for p in pass_info if p['direction'] == 'cw')
        n_ccw = sum(1 for p in pass_info if p['direction'] == 'ccw')
        print(f"  Total: {n_cw} CW, {n_ccw} CCW traversals")

    return {
        'valid_mask': valid_mask,
        'pass_info': pass_info,
    }


def compute_direction_mask(
    centered_pos,
    perimeter,
    direction='both',
    approach_threshold=5.0,
    smooth_window=5,
):
    """
    Compute a mask for frames in a specific running direction that made valid PF passes.

    Parameters
    ----------
    centered_pos : ndarray
        Position centered at PF peak (0 = PF center).
    perimeter : float
        Total perimeter length.
    direction : str
        'cw', 'ccw', or 'both'.
    approach_threshold : float
        Maximum distance from center to count as a valid approach.
    smooth_window : int
        Window for smoothing velocity when computing direction.

    Returns
    -------
    ndarray (bool)
        Mask for frames matching the specified direction and valid passes.
    """
    centered_pos = np.asarray(centered_pos, dtype=float)
    n_frames = len(centered_pos)

    # Get valid passes
    pass_result = detect_valid_pf_passes(centered_pos, approach_threshold=approach_threshold)
    valid_mask = pass_result['valid_mask']

    if direction == 'both':
        return valid_mask

    # Get running direction
    dir_result = compute_running_direction(centered_pos, perimeter, smooth_window)
    frame_direction = dir_result['direction']

    # Combine: valid pass AND correct direction
    if direction == 'ccw':
        dir_mask = frame_direction > 0
    elif direction == 'cw':
        dir_mask = frame_direction < 0
    else:
        dir_mask = np.ones(n_frames, dtype=bool)

    return valid_mask & dir_mask


def compute_linearized_rate_map(
    x_neural, y_neural, spike_times, speed, frame_rate,
    width=35.5, height=20.0,
    corner_radius=5.0,
    corridor_width=5.0,
    bin_size=1.5,
    speed_threshold=3.0,
    kernel_size=51,
    filter_type='boxcar',
    min_duration_s=0.25,
    merge_gap_s=0.0,
    smooth_sigma=None,
    origin='corner',
    direction_mask=None,
):
    """
    Compute linearized firing rate map along the corridor.

    Parameters
    ----------
    x_neural, y_neural : ndarray
        Position coordinates.
    spike_times : array-like
        Spike frame indices.
    speed : ndarray
        Speed array.
    frame_rate : float
        Sampling rate in Hz.
    width, height : float
        Arena dimensions in cm.
    corner_radius : float
        Radius of rounded corners in cm.
    corridor_width : float
        Width of the corridor from the wall.
    bin_size : float
        Bin size for linearization.
    speed_threshold : float
        Speed threshold for moving epochs.
    smooth_sigma : float, optional
        Gaussian smoothing sigma (in bins). If None, no smoothing.
    origin : str
        Coordinate origin: 'center' or 'corner'.
    direction_mask : ndarray (bool), optional
        Additional mask for direction filtering (e.g., CW or CCW only).
        If None, uses all frames.

    Returns
    -------
    dict with keys:
        'rate_map' : ndarray
            Firing rate in each linearized bin.
        'occupancy' : ndarray
            Time spent in each bin (s).
        'spike_counts' : ndarray
            Number of spikes in each bin.
        'bin_centers' : ndarray
            Bin centers (linear position).
        'bin_edges' : ndarray
            Bin edges.
    """
    from scipy.ndimage import gaussian_filter1d

    x = np.asarray(x_neural, dtype=float)
    y = np.asarray(y_neural, dtype=float)
    spike_times = np.asarray(spike_times, dtype=int)

    # Convert to center coordinates
    if origin == 'corner':
        x_center = x - width / 2
        y_center = y - height / 2
    else:
        x_center = x.copy()
        y_center = y.copy()

    # Compute linearized positions
    lin_result = linearize_position_rounded_corridor(
        x_center, y_center, width, height, corner_radius, corridor_width, origin='center'
    )
    linear_pos = lin_result['linear_pos']
    in_corridor = lin_result['in_corridor']

    # Get bins
    bins_info = compute_linearization_bins(width, height, corner_radius, bin_size)
    bin_edges = bins_info['bin_edges']
    bin_centers = bins_info['bin_centers']
    n_bins = bins_info['n_bins']

    # Compute moving epochs
    _, _, moving_idx = _compute_moving_epochs(
        speed, frame_rate,
        kernel_size=kernel_size,
        filter_type=filter_type,
        speed_threshold=speed_threshold,
        min_duration_s=min_duration_s,
        merge_gap_s=merge_gap_s
    )

    moving_mask = np.zeros(len(x), dtype=bool)
    if moving_idx is not None and len(moving_idx) > 0:
        moving_mask[np.asarray(moving_idx, dtype=int)] = True

    # Only use moving & in-corridor frames
    valid_mask = moving_mask & in_corridor & np.isfinite(linear_pos)

    # Apply direction mask if provided
    if direction_mask is not None:
        direction_mask = np.asarray(direction_mask, dtype=bool)
        if len(direction_mask) == len(valid_mask):
            valid_mask = valid_mask & direction_mask

    # Compute occupancy
    bin_idx = np.digitize(linear_pos, bin_edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    occupancy = np.zeros(n_bins, dtype=float)
    for i in range(n_bins):
        occupancy[i] = np.sum(valid_mask & (bin_idx == i)) / frame_rate

    # Compute spike counts
    spike_counts = np.zeros(n_bins, dtype=int)
    valid_spikes = spike_times[(spike_times >= 0) & (spike_times < len(x))]
    valid_spikes = valid_spikes[valid_mask[valid_spikes]]

    for sp in valid_spikes:
        bi = bin_idx[sp]
        spike_counts[bi] += 1

    # Compute rate map
    with np.errstate(divide='ignore', invalid='ignore'):
        rate_map = np.where(occupancy > 0, spike_counts / occupancy, np.nan)

    # Smooth if requested
    if smooth_sigma is not None and smooth_sigma > 0:
        # Handle NaN values for smoothing
        valid = np.isfinite(rate_map)
        if np.any(valid):
            rate_filled = np.where(valid, rate_map, 0)
            weights = valid.astype(float)
            rate_smoothed = gaussian_filter1d(rate_filled, smooth_sigma, mode='wrap')
            weights_smoothed = gaussian_filter1d(weights, smooth_sigma, mode='wrap')
            rate_map = np.where(weights_smoothed > 0, rate_smoothed / weights_smoothed, np.nan)

    return {
        'rate_map': rate_map,
        'occupancy': occupancy,
        'spike_counts': spike_counts,
        'bin_centers': bin_centers,
        'bin_edges': bin_edges,
        'n_bins': n_bins,
        'perimeter': bins_info['perimeter'],
    }


def compute_linearized_pf_peak(
    x_neural, y_neural, spike_times, speed, frame_rate,
    width=35.5, height=20.0,
    corner_radius=5.0,
    corridor_width=5.0,
    bin_size=1.5,
    speed_threshold=3.0,
    kernel_size=51,
    filter_type='boxcar',
    min_duration_s=0.25,
    merge_gap_s=0.0,
    smooth_sigma=2,
    origin='corner',
):
    """
    Find the place field peak in linearized coordinates.

    Parameters
    ----------
    x_neural, y_neural : ndarray
        Position coordinates.
    spike_times : array-like
        Spike frame indices.
    speed : ndarray
        Speed array.
    frame_rate : float
        Sampling rate in Hz.
    width, height : float
        Arena dimensions in cm.
    corner_radius : float
        Radius of rounded corners in cm.
    corridor_width : float
        Width of the corridor from the wall.
    bin_size : float
        Bin size for linearization.
    speed_threshold : float
        Speed threshold for moving epochs.
    smooth_sigma : float
        Gaussian smoothing sigma (in bins).
    origin : str
        Coordinate origin: 'center' or 'corner'.

    Returns
    -------
    dict with keys:
        'peak_linear_pos' : float
            Linearized position of the place field peak (cm).
        'peak_bin_idx' : int
            Bin index of the peak.
        'peak_rate' : float
            Firing rate at the peak (Hz).
        'rate_map' : ndarray
            Full linearized rate map.
        'bin_centers' : ndarray
            Bin centers.
        'perimeter' : float
            Total perimeter length.
    """
    # Compute linearized rate map
    lin_rate = compute_linearized_rate_map(
        x_neural, y_neural, spike_times, speed, frame_rate,
        width=width, height=height,
        corner_radius=corner_radius,
        corridor_width=corridor_width,
        bin_size=bin_size,
        speed_threshold=speed_threshold,
        kernel_size=kernel_size,
        filter_type=filter_type,
        min_duration_s=min_duration_s,
        merge_gap_s=merge_gap_s,
        smooth_sigma=smooth_sigma,
        origin=origin,
    )

    rate_map = lin_rate['rate_map']
    bin_centers = lin_rate['bin_centers']

    # Find peak (ignoring NaN)
    if np.all(np.isnan(rate_map)):
        return {
            'peak_linear_pos': np.nan,
            'peak_bin_idx': -1,
            'peak_rate': np.nan,
            'rate_map': rate_map,
            'bin_centers': bin_centers,
            'perimeter': lin_rate['perimeter'],
        }

    peak_bin_idx = np.nanargmax(rate_map)
    peak_linear_pos = bin_centers[peak_bin_idx]
    peak_rate = rate_map[peak_bin_idx]

    return {
        'peak_linear_pos': peak_linear_pos,
        'peak_bin_idx': peak_bin_idx,
        'peak_rate': peak_rate,
        'rate_map': rate_map,
        'bin_centers': bin_centers,
        'perimeter': lin_rate['perimeter'],
        'occupancy': lin_rate['occupancy'],
        'spike_counts': lin_rate['spike_counts'],
    }


def center_linear_position(linear_pos, center_pos, perimeter):
    """
    Center linearized positions so that center_pos becomes 0.

    Handles wrap-around for circular coordinates.

    Parameters
    ----------
    linear_pos : array-like
        Linearized positions (0 to perimeter).
    center_pos : float
        Position to center at (will become 0).
    perimeter : float
        Total perimeter length.

    Returns
    -------
    centered_pos : ndarray
        Centered positions ranging from -perimeter/2 to +perimeter/2.
    """
    linear_pos = np.asarray(linear_pos, dtype=float)
    centered = linear_pos - center_pos

    # Wrap to [-perimeter/2, +perimeter/2]
    centered = np.mod(centered + perimeter / 2, perimeter) - perimeter / 2

    return centered


def compute_linearized_subthreshold_maps(
    x_neural, y_neural, trace, speed, frame_rate,
    width=35.5, height=20.0,
    corner_radius=5.0,
    corridor_width=5.0,
    bin_size=1.5,
    speed_threshold=3.0,
    kernel_size=51,
    filter_type='boxcar',
    min_duration_s=0.25,
    merge_gap_s=0.0,
    theta_freqs=(4, 8),
    slow_freqs=2,
    smooth_sigma=2,
    origin='corner',
    direction_mask=None,
):
    """
    Compute linearized theta amplitude and slow Vm maps along the corridor.

    Parameters
    ----------
    x_neural, y_neural : ndarray
        Position coordinates.
    trace : ndarray
        Voltage trace for this cell.
    speed : ndarray
        Speed array.
    frame_rate : float
        Sampling rate in Hz.
    width, height : float
        Arena dimensions in cm.
    corner_radius : float
        Radius of rounded corners in cm.
    corridor_width : float
        Width of the corridor from the wall.
    bin_size : float
        Bin size for linearization.
    speed_threshold : float
        Speed threshold for moving epochs.
    theta_freqs : tuple
        (low, high) frequency bounds for theta band.
    slow_freqs : float
        Cutoff frequency for slow Vm.
    smooth_sigma : float
        Gaussian smoothing sigma (in bins).
    origin : str
        Coordinate origin: 'center' or 'corner'.
    direction_mask : ndarray (bool), optional
        Additional mask for direction filtering (e.g., CW or CCW only).
        If None, uses all frames.

    Returns
    -------
    dict with keys:
        'theta_map' : ndarray - Mean theta amplitude per bin
        'slow_map' : ndarray - Mean slow Vm per bin
        'bin_centers' : ndarray
        'perimeter' : float
    """
    from scipy.ndimage import gaussian_filter1d

    x = np.asarray(x_neural, dtype=float)
    y = np.asarray(y_neural, dtype=float)
    trace = np.asarray(trace, dtype=float)

    # Interpolate NaN in trace
    trace = interpolate_nan_segment(trace)

    # Compute theta and slow Vm
    theta_vm = bandpass_filter(trace, theta_freqs[0], theta_freqs[1], frame_rate, order=5)
    slow_vm = lowpass_filter(trace, slow_freqs, frame_rate, order=5)
    theta_amp = np.abs(signal.hilbert(theta_vm))

    # Convert to center coordinates
    if origin == 'corner':
        x_center = x - width / 2
        y_center = y - height / 2
    else:
        x_center = x.copy()
        y_center = y.copy()

    # Compute linearized positions
    lin_result = linearize_position_rounded_corridor(
        x_center, y_center, width, height, corner_radius, corridor_width, origin='center'
    )
    linear_pos = lin_result['linear_pos']
    in_corridor = lin_result['in_corridor']

    # Get bins
    bins_info = compute_linearization_bins(width, height, corner_radius, bin_size)
    bin_edges = bins_info['bin_edges']
    bin_centers = bins_info['bin_centers']
    n_bins = bins_info['n_bins']

    # Compute moving epochs
    _, _, moving_idx = _compute_moving_epochs(
        speed, frame_rate,
        kernel_size=kernel_size,
        filter_type=filter_type,
        speed_threshold=speed_threshold,
        min_duration_s=min_duration_s,
        merge_gap_s=merge_gap_s
    )

    moving_mask = np.zeros(len(x), dtype=bool)
    if moving_idx is not None and len(moving_idx) > 0:
        moving_mask[np.asarray(moving_idx, dtype=int)] = True

    # Only use moving & in-corridor frames
    valid_mask = moving_mask & in_corridor & np.isfinite(linear_pos) & np.isfinite(theta_amp)

    # Apply direction mask if provided
    if direction_mask is not None:
        direction_mask = np.asarray(direction_mask, dtype=bool)
        if len(direction_mask) == len(valid_mask):
            valid_mask = valid_mask & direction_mask

    # Compute bin indices
    bin_idx = np.digitize(linear_pos, bin_edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    # Accumulate theta and slow Vm per bin
    theta_sum = np.zeros(n_bins, dtype=float)
    slow_sum = np.zeros(n_bins, dtype=float)
    counts = np.zeros(n_bins, dtype=int)

    valid_indices = np.where(valid_mask)[0]
    for idx in valid_indices:
        bi = bin_idx[idx]
        theta_sum[bi] += theta_amp[idx]
        slow_sum[bi] += slow_vm[idx]
        counts[bi] += 1

    # Compute means
    with np.errstate(divide='ignore', invalid='ignore'):
        theta_map = np.where(counts > 0, theta_sum / counts, np.nan)
        slow_map = np.where(counts > 0, slow_sum / counts, np.nan)

    # Smooth if requested
    if smooth_sigma is not None and smooth_sigma > 0:
        for arr in [theta_map, slow_map]:
            valid = np.isfinite(arr)
            if np.any(valid):
                arr_filled = np.where(valid, arr, 0)
                weights = valid.astype(float)
                arr_smoothed = gaussian_filter1d(arr_filled, smooth_sigma, mode='wrap')
                weights_smoothed = gaussian_filter1d(weights, smooth_sigma, mode='wrap')
                arr[:] = np.where(weights_smoothed > 0, arr_smoothed / weights_smoothed, np.nan)

    return {
        'theta_map': theta_map,
        'slow_map': slow_map,
        'bin_centers': bin_centers,
        'perimeter': bins_info['perimeter'],
        'n_bins': n_bins,
    }


def plot_linearized_pf_raster(
    x_neural, y_neural, speed, frame_rate,
    spike_times_ss, spike_times_cs,
    pf_peak_linear_pos, perimeter,
    width=35.5, height=20.0,
    corner_radius=5.0,
    corridor_width=5.0,
    speed_threshold=3.0,
    kernel_size=51,
    filter_type='boxcar',
    min_duration_s=0.25,
    merge_gap_s=0.0,
    moving_only=True,
    origin='corner',
    ax=None,
    trajectory_color='#cccccc',
    trajectory_alpha=0.3,
    trajectory_linewidth=0.5,
    ss_color='#026C80',
    cs_color='#EE9B00',
    spike_marker='|',
    spike_size=3,
    spike_alpha=0.8,
    title=None,
    show_colorbar=False,
    time_unit='s',
    plateau_bursts=None,
    plateau_color='red',
    plateau_linewidth=2,
    plateau_alpha=0.8,
    direction_mask=None,
    center_line_color='red',
):
    """
    Plot linearized position vs time raster with trajectory and spikes.

    X-axis: Linearized position centered at place field peak (0 = PF peak).
    Y-axis: Time.
    Gray line: Animal trajectory.
    Colored markers: Simple spikes (SS) and complex spikes (CS).
    Red segments: Plateau bursts (>100ms complex bursts).

    Parameters
    ----------
    x_neural, y_neural : ndarray
        Position coordinates.
    speed : ndarray
        Speed array.
    frame_rate : float
        Sampling rate in Hz.
    spike_times_ss : array-like
        Simple spike frame indices.
    spike_times_cs : array-like
        Complex spike frame indices.
    pf_peak_linear_pos : float
        Linearized position of the place field peak (for centering).
    perimeter : float
        Total perimeter length.
    width, height : float
        Arena dimensions in cm.
    corner_radius : float
        Radius of rounded corners in cm.
    corridor_width : float
        Width of the corridor from the wall.
    speed_threshold : float
        Speed threshold for moving epochs.
    moving_only : bool
        If True, only show trajectory during moving epochs.
    origin : str
        Coordinate origin: 'center' or 'corner'.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    trajectory_color : str
        Color for trajectory.
    trajectory_alpha : float
        Alpha for trajectory.
    ss_color, cs_color : str
        Colors for simple and complex spikes.
    spike_marker : str
        Marker style for spikes.
    spike_size : float
        Marker size for spikes.
    title : str, optional
        Plot title.
    time_unit : str
        Time unit for y-axis ('s' for seconds, 'min' for minutes).
    plateau_bursts : dict, optional
        Dict with 'starts', 'ends' arrays for plateau bursts to display.
    plateau_color : str
        Color for plateau burst segments.
    plateau_linewidth : float
        Line width for plateau burst segments.
    direction_mask : ndarray (bool), optional
        Additional mask for direction filtering (e.g., CW or CCW only).
        If None, uses all frames (subject to moving_only).

    Returns
    -------
    dict with keys:
        'fig', 'ax' : matplotlib objects
        'linear_pos_centered' : ndarray
            Centered linearized positions for all frames.
        'time_axis' : ndarray
            Time axis.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 10))
    else:
        fig = ax.figure

    x = np.asarray(x_neural, dtype=float)
    y = np.asarray(y_neural, dtype=float)
    n_frames = len(x)

    # Convert to center coordinates for linearization
    if origin == 'corner':
        x_center = x - width / 2
        y_center = y - height / 2
    else:
        x_center = x.copy()
        y_center = y.copy()

    # Compute linearized positions
    lin_result = linearize_position_rounded_corridor(
        x_center, y_center, width, height, corner_radius, corridor_width, origin='center'
    )
    linear_pos = lin_result['linear_pos']

    # Center at PF peak
    linear_pos_centered = center_linear_position(linear_pos, pf_peak_linear_pos, perimeter)

    # Time axis
    time_axis = np.arange(n_frames) / frame_rate
    if time_unit == 'min':
        time_axis = time_axis / 60.0

    # Compute moving mask
    _, _, moving_idx = _compute_moving_epochs(
        speed, frame_rate,
        kernel_size=kernel_size,
        filter_type=filter_type,
        speed_threshold=speed_threshold,
        min_duration_s=min_duration_s,
        merge_gap_s=merge_gap_s
    )
    moving_mask = np.zeros(n_frames, dtype=bool)
    if moving_idx is not None and len(moving_idx) > 0:
        moving_mask[np.asarray(moving_idx, dtype=int)] = True

    # Determine which frames to plot
    if moving_only:
        plot_mask = moving_mask & np.isfinite(linear_pos_centered)
    else:
        plot_mask = np.isfinite(linear_pos_centered)

    # Apply direction mask if provided
    if direction_mask is not None:
        direction_mask = np.asarray(direction_mask, dtype=bool)
        if len(direction_mask) == len(plot_mask):
            plot_mask = plot_mask & direction_mask

    # Plot trajectory as scatter (more efficient for large datasets)
    ax.scatter(
        linear_pos_centered[plot_mask],
        time_axis[plot_mask],
        c=trajectory_color,
        s=0.5,
        alpha=trajectory_alpha,
        rasterized=True,
        label='Trajectory'
    )

    # Plot plateau bursts as line segments showing spatial extent
    n_plateaus = 0
    if plateau_bursts is not None:
        starts = np.asarray(plateau_bursts.get('starts', []), dtype=int)
        ends = np.asarray(plateau_bursts.get('ends', []), dtype=int)

        for start_frame, end_frame in zip(starts, ends):
            if start_frame < 0 or end_frame >= n_frames:
                continue
            if moving_only and not np.any(moving_mask[start_frame:end_frame+1]):
                continue
            # Apply direction mask to plateaus
            if direction_mask is not None and not np.any(direction_mask[start_frame:end_frame+1]):
                continue

            # Get positions and times for this burst
            burst_frames = np.arange(start_frame, end_frame + 1)
            burst_pos = linear_pos_centered[burst_frames]
            burst_time = time_axis[burst_frames]

            # Only plot if we have valid positions
            valid = np.isfinite(burst_pos)
            if np.sum(valid) >= 2:
                # Handle wrap-around: detect large jumps and split into segments
                burst_pos_valid = burst_pos[valid]
                burst_time_valid = burst_time[valid]

                # Detect wrap-around points (jumps > half perimeter)
                pos_diff = np.diff(burst_pos_valid)
                wrap_threshold = perimeter / 3  # Large jump indicates wrap-around
                wrap_points = np.where(np.abs(pos_diff) > wrap_threshold)[0]

                if len(wrap_points) == 0:
                    # No wrap-around, plot as single segment
                    ax.plot(
                        burst_pos_valid,
                        burst_time_valid,
                        color=plateau_color,
                        linewidth=plateau_linewidth,
                        alpha=plateau_alpha,
                        solid_capstyle='round',
                        zorder=5
                    )
                else:
                    # Split at wrap-around points and plot each segment
                    segment_starts = [0] + list(wrap_points + 1)
                    segment_ends = list(wrap_points + 1) + [len(burst_pos_valid)]

                    for seg_start, seg_end in zip(segment_starts, segment_ends):
                        if seg_end - seg_start >= 2:
                            ax.plot(
                                burst_pos_valid[seg_start:seg_end],
                                burst_time_valid[seg_start:seg_end],
                                color=plateau_color,
                                linewidth=plateau_linewidth,
                                alpha=plateau_alpha,
                                solid_capstyle='round',
                                zorder=5
                            )

                n_plateaus += 1

    # Plot simple spikes
    spike_times_ss = np.asarray(spike_times_ss, dtype=int)
    valid_ss = spike_times_ss[(spike_times_ss >= 0) & (spike_times_ss < n_frames)]
    if moving_only:
        valid_ss = valid_ss[moving_mask[valid_ss]]
    valid_ss = valid_ss[np.isfinite(linear_pos_centered[valid_ss])]
    # Apply direction mask to spikes
    if direction_mask is not None and len(valid_ss) > 0:
        valid_ss = valid_ss[direction_mask[valid_ss]]

    if len(valid_ss) > 0:
        ax.scatter(
            linear_pos_centered[valid_ss],
            time_axis[valid_ss],
            c=ss_color,
            s=spike_size,
            marker=spike_marker,
            alpha=spike_alpha,
            label=f'SS (n={len(valid_ss)})',
            zorder=3
        )

    # Plot complex spikes
    spike_times_cs = np.asarray(spike_times_cs, dtype=int)
    valid_cs = spike_times_cs[(spike_times_cs >= 0) & (spike_times_cs < n_frames)]
    if moving_only:
        valid_cs = valid_cs[moving_mask[valid_cs]]
    valid_cs = valid_cs[np.isfinite(linear_pos_centered[valid_cs])]
    # Apply direction mask to spikes
    if direction_mask is not None and len(valid_cs) > 0:
        valid_cs = valid_cs[direction_mask[valid_cs]]

    if len(valid_cs) > 0:
        ax.scatter(
            linear_pos_centered[valid_cs],
            time_axis[valid_cs],
            c=cs_color,
            s=spike_size * 2,
            marker=spike_marker,
            alpha=spike_alpha,
            label=f'CS (n={len(valid_cs)})',
            zorder=4
        )

    # Formatting
    _cl_label = 'PF peak' if center_line_color != 'gray' else 'Max rate'
    ax.axvline(0, color=center_line_color, linestyle='--', linewidth=1, alpha=0.7, label=_cl_label)
    ax.set_xlabel('Linearized position (cm, centered at PF peak)')
    ylabel = 'Time (min)' if time_unit == 'min' else 'Time (s)'
    ax.set_ylabel(ylabel)
    ax.set_xlim(-perimeter / 2, perimeter / 2)
    ax.set_ylim(0, time_axis[-1])
    ax.invert_yaxis()  # Time flows downward

    # Update legend to include plateaus if present
    if n_plateaus > 0:
        # Add a dummy line for legend
        ax.plot([], [], color=plateau_color, linewidth=plateau_linewidth,
                label=f'Plateau (n={n_plateaus})')

    ax.legend(loc='lower right', fontsize=7, markerscale=2)

    if title:
        ax.set_title(title)

    return {
        'fig': fig,
        'ax': ax,
        'linear_pos_centered': linear_pos_centered,
        'time_axis': time_axis,
        'moving_mask': moving_mask,
        'n_plateaus': n_plateaus,
    }


def plot_linearized_pf_raster_for_cell(
    cell_idx,
    x_neural, y_neural, speed, frame_rate,
    refined_SS, all_CS_spikes, all_spikes,
    PC_output_all,
    traces=None,
    complex_bursts_dicts=None,
    width=35.5, height=20.0,
    corner_radius=5.0,
    corridor_width=5.0,
    bin_size=1.5,
    speed_threshold=3.0,
    kernel_size=51,
    filter_type='boxcar',
    min_duration_s=0.25,
    merge_gap_s=0.0,
    smooth_sigma=2,
    moving_only=True,
    origin='corner',
    ax=None,
    figsize=(10, 16),
    trajectory_color='#cccccc',
    trajectory_alpha=0.3,
    ss_color='#026C80',
    cs_color='#EE9B00',
    plateau_color='red',
    spike_size=3,
    time_unit='min',
    show_rate_map=True,
    show_subthreshold=True,
    theta_freqs=(4, 8),
    slow_freqs=2,
    plateau_min_duration_ms=100,
    plateaus_dicts=None,
):
    """
    Plot linearized PF raster for a single cell, with SS/CS rate maps and subthreshold activity.

    Parameters
    ----------
    cell_idx : int
        Cell index.
    x_neural, y_neural : ndarray
        Position coordinates.
    speed : ndarray
        Speed array.
    frame_rate : float
        Sampling rate in Hz.
    refined_SS : list
        List of simple spike times for each cell.
    all_CS_spikes : list
        List of complex spike times for each cell.
    all_spikes : list
        List of all spike times for each cell.
    PC_output_all : list
        Place cell analysis output for each cell.
    traces : list, optional
        List of voltage traces for each cell (for theta/slow Vm computation).
    complex_bursts_dicts : list, optional
        List of complex burst dicts for each cell (with 'starts', 'ends', 'durations_ms').
    width, height : float
        Arena dimensions in cm.
    corner_radius : float
        Radius of rounded corners in cm.
    corridor_width : float
        Width of the corridor from the wall.
    bin_size : float
        Bin size for linearization.
    speed_threshold : float
        Speed threshold for moving epochs.
    smooth_sigma : float
        Gaussian smoothing sigma (in bins) for rate map.
    moving_only : bool
        If True, only show trajectory during moving epochs.
    origin : str
        Coordinate origin: 'center' or 'corner'.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on (if None, creates figure with subplots).
    figsize : tuple
        Figure size.
    show_rate_map : bool
        If True, show linearized rate map subplot.
    show_subthreshold : bool
        If True, show theta and slow Vm subplots (requires traces).
    theta_freqs : tuple
        (low, high) frequency bounds for theta band.
    slow_freqs : float
        Cutoff frequency for slow Vm.
    plateau_min_duration_ms : float
        Minimum duration (ms) for a complex burst to be considered a plateau.

    Returns
    -------
    dict with keys:
        'fig', 'axes' : matplotlib objects
        'pf_peak_info' : dict with PF peak information
        'raster_result' : dict with raster plot results
        'ss_rate_map' : dict with SS rate map info
        'cs_rate_map' : dict with CS rate map info
        'subthreshold_maps' : dict with theta/slow Vm maps (if computed)
    """
    # Check if cell is a place cell
    analysis = PC_output_all[cell_idx]
    is_place_cell = analysis.get('is_place_cell', False)

    # Get spike times
    spike_times_all = np.asarray(all_spikes[cell_idx], dtype=int)
    spike_times_ss = np.asarray(refined_SS[cell_idx], dtype=int)
    spike_times_cs = np.asarray(all_CS_spikes[cell_idx], dtype=int)

    # Extract plateau bursts
    plateau_bursts = None
    if plateaus_dicts is not None and cell_idx < len(plateaus_dicts):
        _plat = plateaus_dicts[cell_idx]
        if _plat is not None:
            _ps = np.asarray(_plat.get('starts', []), dtype=int)
            _pe = np.asarray(_plat.get('ends', []), dtype=int)
            if _ps.size > 0:
                plateau_bursts = {'starts': _ps, 'ends': _pe}
    elif complex_bursts_dicts is not None and cell_idx < len(complex_bursts_dicts):
        bursts = complex_bursts_dicts[cell_idx]
        cb_starts = np.asarray(bursts.get('starts', []), dtype=int)
        cb_ends = np.asarray(bursts.get('ends', []), dtype=int)
        cb_durs = np.asarray(bursts.get('durations_ms', []), dtype=float)
        plateau_mask = cb_durs >= plateau_min_duration_ms
        if np.any(plateau_mask):
            plateau_bursts = {
                'starts': cb_starts[plateau_mask],
                'ends': cb_ends[plateau_mask],
            }

    # Choose spike times for PF peak: use SS spikes if cell is not an
    # all-spikes place cell but is an SS place cell
    is_place_cell_ss = analysis.get('is_place_cell_ss', False)
    is_place_cell_cs = analysis.get('is_place_cell_cs', False)
    is_any_place_cell = is_place_cell or is_place_cell_ss or is_place_cell_cs
    if is_place_cell:
        pf_spikes = spike_times_all
    elif is_place_cell_ss:
        pf_spikes = spike_times_ss
    else:
        pf_spikes = spike_times_all

    # Compute linearized PF peak
    pf_peak_info = compute_linearized_pf_peak(
        x_neural, y_neural, pf_spikes, speed, frame_rate,
        width=width, height=height,
        corner_radius=corner_radius,
        corridor_width=corridor_width,
        bin_size=bin_size,
        speed_threshold=speed_threshold,
        kernel_size=kernel_size,
        filter_type=filter_type,
        min_duration_s=min_duration_s,
        merge_gap_s=merge_gap_s,
        smooth_sigma=smooth_sigma,
        origin=origin,
    )

    # Compute separate SS and CS rate maps
    ss_rate_info = compute_linearized_rate_map(
        x_neural, y_neural, spike_times_ss, speed, frame_rate,
        width=width, height=height,
        corner_radius=corner_radius,
        corridor_width=corridor_width,
        bin_size=bin_size,
        speed_threshold=speed_threshold,
        kernel_size=kernel_size,
        filter_type=filter_type,
        min_duration_s=min_duration_s,
        merge_gap_s=merge_gap_s,
        smooth_sigma=smooth_sigma,
        origin=origin,
    )

    cs_rate_info = compute_linearized_rate_map(
        x_neural, y_neural, spike_times_cs, speed, frame_rate,
        width=width, height=height,
        corner_radius=corner_radius,
        corridor_width=corridor_width,
        bin_size=bin_size,
        speed_threshold=speed_threshold,
        kernel_size=kernel_size,
        filter_type=filter_type,
        min_duration_s=min_duration_s,
        merge_gap_s=merge_gap_s,
        smooth_sigma=smooth_sigma,
        origin=origin,
    )

    # Compute subthreshold maps if traces provided
    subthreshold_maps = None
    if show_subthreshold and traces is not None and cell_idx < len(traces):
        subthreshold_maps = compute_linearized_subthreshold_maps(
            x_neural, y_neural, traces[cell_idx], speed, frame_rate,
            width=width, height=height,
            corner_radius=corner_radius,
            corridor_width=corridor_width,
            bin_size=bin_size,
            speed_threshold=speed_threshold,
            kernel_size=kernel_size,
            filter_type=filter_type,
            min_duration_s=min_duration_s,
            merge_gap_s=merge_gap_s,
            theta_freqs=theta_freqs,
            slow_freqs=slow_freqs,
            smooth_sigma=smooth_sigma,
            origin=origin,
        )

    # Determine number of subplots
    n_subplots = 1  # raster
    if show_rate_map:
        n_subplots += 1
    if show_subthreshold and subthreshold_maps is not None:
        n_subplots += 2  # theta + slow

    # Create figure
    if ax is None:
        if n_subplots > 1:
            # Height ratios: raster gets most space
            height_ratios = [4] + [1] * (n_subplots - 1)
            fig, axes = plt.subplots(n_subplots, 1, figsize=figsize,
                                      gridspec_kw={'height_ratios': height_ratios},
                                      sharex=True)
            ax_raster = axes[0]
        else:
            fig, ax_raster = plt.subplots(figsize=figsize)
            axes = [ax_raster]
    else:
        fig = ax.figure
        ax_raster = ax
        axes = [ax]

    # Plot raster
    pf_peak_pos = pf_peak_info['peak_linear_pos']
    perimeter = pf_peak_info['perimeter']

    if np.isnan(pf_peak_pos):
        ax_raster.text(0.5, 0.5, 'No valid PF peak', transform=ax_raster.transAxes,
                       ha='center', va='center', fontsize=12)
        return {
            'fig': fig,
            'axes': axes,
            'pf_peak_info': pf_peak_info,
            'raster_result': None,
            'ss_rate_map': ss_rate_info,
            'cs_rate_map': cs_rate_info,
            'subthreshold_maps': subthreshold_maps,
        }

    raster_result = plot_linearized_pf_raster(
        x_neural, y_neural, speed, frame_rate,
        spike_times_ss, spike_times_cs,
        pf_peak_pos, perimeter,
        width=width, height=height,
        corner_radius=corner_radius,
        corridor_width=corridor_width,
        speed_threshold=speed_threshold,
        kernel_size=kernel_size,
        filter_type=filter_type,
        min_duration_s=min_duration_s,
        merge_gap_s=merge_gap_s,
        moving_only=moving_only,
        origin=origin,
        ax=ax_raster,
        trajectory_color=trajectory_color,
        trajectory_alpha=trajectory_alpha,
        ss_color=ss_color,
        cs_color=cs_color,
        spike_size=spike_size,
        time_unit=time_unit,
        title=f'Cell {cell_idx + 1}' + (' (Place Cell)' if is_place_cell else ''),
        plateau_bursts=plateau_bursts,
        plateau_color=plateau_color,
        center_line_color='magenta' if is_any_place_cell else 'gray',
    )

    # Remove x-label from raster (shared with subplots below)
    if n_subplots > 1:
        ax_raster.set_xlabel('')

    # Get bin centers and sort index for all lower subplots
    bin_centers = pf_peak_info['bin_centers']
    bin_centers_centered = center_linear_position(bin_centers, pf_peak_pos, perimeter)
    sort_idx = np.argsort(bin_centers_centered)
    bin_centers_sorted = bin_centers_centered[sort_idx]

    subplot_idx = 1  # Track which subplot we're on

    # Plot rate maps
    if show_rate_map and n_subplots > 1:
        ax_rate = axes[subplot_idx]
        subplot_idx += 1

        # Get SS and CS rate maps (sorted)
        ss_rate_sorted = ss_rate_info['rate_map'][sort_idx]
        cs_rate_sorted = cs_rate_info['rate_map'][sort_idx]

        # Plot SS rate map
        ax_rate.fill_between(bin_centers_sorted, 0, ss_rate_sorted,
                              alpha=0.4, color=ss_color, label='SS')
        ax_rate.plot(bin_centers_sorted, ss_rate_sorted, color=ss_color, linewidth=1)

        # Plot CS rate map
        ax_rate.fill_between(bin_centers_sorted, 0, cs_rate_sorted,
                              alpha=0.4, color=cs_color, label='CS')
        ax_rate.plot(bin_centers_sorted, cs_rate_sorted, color=cs_color, linewidth=1)

        # Mark PF peak
        _cl_color = 'magenta' if is_any_place_cell else 'gray'
        ax_rate.axvline(0, color=_cl_color, linestyle='--', linewidth=1, alpha=0.7)

        ax_rate.set_ylabel('Rate (Hz)')
        ax_rate.set_xlim(-perimeter / 2, perimeter / 2)
        ax_rate.legend(loc='upper right', fontsize=7)

        # Add peak rate info
        ss_peak = np.nanmax(ss_rate_sorted) if np.any(np.isfinite(ss_rate_sorted)) else 0
        cs_peak = np.nanmax(cs_rate_sorted) if np.any(np.isfinite(cs_rate_sorted)) else 0
        ax_rate.set_title(f'SS peak: {ss_peak:.2f} Hz, CS peak: {cs_peak:.2f} Hz', fontsize=9)

    # Plot theta amplitude
    if show_subthreshold and subthreshold_maps is not None and n_subplots > subplot_idx:
        ax_theta = axes[subplot_idx]
        subplot_idx += 1

        theta_sorted = subthreshold_maps['theta_map'][sort_idx]
        ax_theta.fill_between(bin_centers_sorted, 0, theta_sorted,
                               alpha=0.4, color='blue')
        ax_theta.plot(bin_centers_sorted, theta_sorted, color='blue', linewidth=1)
        ax_theta.axvline(0, color=_cl_color, linestyle='--', linewidth=1, alpha=0.7)
        ax_theta.set_ylabel('θ amp\n(a.u.)')
        ax_theta.set_xlim(-perimeter / 2, perimeter / 2)

        theta_peak = np.nanmax(theta_sorted) if np.any(np.isfinite(theta_sorted)) else 0
        ax_theta.set_title(f'Theta ({theta_freqs[0]}-{theta_freqs[1]} Hz), peak: {theta_peak:.3f}', fontsize=9)

    # Plot slow Vm
    if show_subthreshold and subthreshold_maps is not None and n_subplots > subplot_idx:
        ax_slow = axes[subplot_idx]
        subplot_idx += 1

        slow_sorted = subthreshold_maps['slow_map'][sort_idx]
        ax_slow.fill_between(bin_centers_sorted, np.nanmin(slow_sorted) if np.any(np.isfinite(slow_sorted)) else 0,
                              slow_sorted, alpha=0.4, color='red')
        ax_slow.plot(bin_centers_sorted, slow_sorted, color='red', linewidth=1)
        ax_slow.axvline(0, color=_cl_color, linestyle='--', linewidth=1, alpha=0.7)
        ax_slow.set_ylabel('Slow Vm\n(a.u.)')
        ax_slow.set_xlabel('Linearized position (cm, centered at PF peak)')
        ax_slow.set_xlim(-perimeter / 2, perimeter / 2)

        slow_range = np.nanmax(slow_sorted) - np.nanmin(slow_sorted) if np.any(np.isfinite(slow_sorted)) else 0
        ax_slow.set_title(f'Slow Vm (<{slow_freqs} Hz), range: {slow_range:.3f}', fontsize=9)

    plt.tight_layout()

    return {
        'fig': fig,
        'axes': axes,
        'pf_peak_info': pf_peak_info,
        'raster_result': raster_result,
        'ss_rate_map': ss_rate_info,
        'cs_rate_map': cs_rate_info,
        'subthreshold_maps': subthreshold_maps,
    }


def plot_linearized_pf_rasters_all_cells(
    x_neural, y_neural, speed, frame_rate,
    refined_SS, all_CS_spikes, all_spikes,
    PC_output_all,
    save_dir,
    traces=None,
    complex_bursts_dicts=None,
    width=35.5, height=20.0,
    corner_radius=5.0,
    corridor_width=5.0,
    bin_size=1.5,
    speed_threshold=3.0,
    kernel_size=51,
    filter_type='boxcar',
    min_duration_s=0.25,
    merge_gap_s=0.0,
    smooth_sigma=2,
    moving_only=True,
    origin='corner',
    figsize=(10, 16),
    trajectory_color='#cccccc',
    trajectory_alpha=0.3,
    ss_color='#026C80',
    cs_color='#EE9B00',
    plateau_color='red',
    spike_size=3,
    time_unit='min',
    show_rate_map=True,
    show_subthreshold=True,
    theta_freqs=(4, 8),
    slow_freqs=2,
    plateau_min_duration_ms=100,
    plateaus_dicts=None,
    place_cells_only=True,
    show_plots=False,
):
    """
    Generate and save linearized PF raster plots for all (place) cells.

    Parameters
    ----------
    x_neural, y_neural : ndarray
        Position coordinates.
    speed : ndarray
        Speed array.
    frame_rate : float
        Sampling rate in Hz.
    refined_SS : list
        List of simple spike times for each cell.
    all_CS_spikes : list
        List of complex spike times for each cell.
    all_spikes : list
        List of all spike times for each cell.
    PC_output_all : list
        Place cell analysis output for each cell.
    save_dir : str
        Directory to save figures.
    traces : list, optional
        List of voltage traces for each cell (for theta/slow Vm computation).
    complex_bursts_dicts : list, optional
        List of complex burst dicts for each cell.
    width, height : float
        Arena dimensions in cm.
    corner_radius : float
        Radius of rounded corners in cm.
    corridor_width : float
        Width of the corridor from the wall.
    bin_size : float
        Bin size for linearization.
    speed_threshold : float
        Speed threshold for moving epochs.
    smooth_sigma : float
        Gaussian smoothing sigma (in bins) for rate map.
    moving_only : bool
        If True, only show trajectory during moving epochs.
    show_subthreshold : bool
        If True, show theta and slow Vm subplots (requires traces).
    theta_freqs : tuple
        (low, high) frequency bounds for theta band.
    slow_freqs : float
        Cutoff frequency for slow Vm.
    plateau_min_duration_ms : float
        Minimum duration (ms) for a complex burst to be considered a plateau.
    place_cells_only : bool
        If True, only process place cells.
    show_plots : bool
        If True, display plots.

    Returns
    -------
    dict with keys:
        'n_cells_processed' : int
        'cell_indices' : list
        'pf_peak_infos' : list of dicts
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    n_cells = len(PC_output_all)
    cell_indices = []
    pf_peak_infos = []

    for cell_idx in range(n_cells):
        analysis = PC_output_all[cell_idx]
        is_place_cell = analysis.get('is_place_cell', False)

        if place_cells_only and not is_place_cell:
            continue

        print(f'Processing Cell {cell_idx + 1}...')

        result = plot_linearized_pf_raster_for_cell(
            cell_idx,
            x_neural, y_neural, speed, frame_rate,
            refined_SS, all_CS_spikes, all_spikes,
            PC_output_all,
            traces=traces,
            complex_bursts_dicts=complex_bursts_dicts,
            width=width, height=height,
            corner_radius=corner_radius,
            corridor_width=corridor_width,
            bin_size=bin_size,
            speed_threshold=speed_threshold,
            kernel_size=kernel_size,
            filter_type=filter_type,
            min_duration_s=min_duration_s,
            merge_gap_s=merge_gap_s,
            smooth_sigma=smooth_sigma,
            moving_only=moving_only,
            origin=origin,
            figsize=figsize,
            trajectory_color=trajectory_color,
            trajectory_alpha=trajectory_alpha,
            ss_color=ss_color,
            cs_color=cs_color,
            plateau_color=plateau_color,
            spike_size=spike_size,
            time_unit=time_unit,
            show_rate_map=show_rate_map,
            show_subthreshold=show_subthreshold,
            theta_freqs=theta_freqs,
            slow_freqs=slow_freqs,
            plateau_min_duration_ms=plateau_min_duration_ms,
            plateaus_dicts=plateaus_dicts,
        )

        fig = result['fig']
        pf_peak_info = result['pf_peak_info']

        # Save figure
        out_path = os.path.join(save_dir, f'Cell{cell_idx + 1}_linearized_raster.svg')
        plt.rcParams['svg.fonttype'] = 'none'
        fig.savefig(out_path, dpi=300)

        if not show_plots:
            plt.close(fig)

        print(f'  Saved to {out_path}')
        print(f'  PF peak: {pf_peak_info["peak_linear_pos"]:.1f} cm, '
              f'rate: {pf_peak_info["peak_rate"]:.2f} Hz')

        cell_indices.append(cell_idx)
        pf_peak_infos.append(pf_peak_info)

    print(f'\nProcessed {len(cell_indices)} cells')
    print(f'Figures saved to: {save_dir}')

    return {
        'n_cells_processed': len(cell_indices),
        'cell_indices': cell_indices,
        'pf_peak_infos': pf_peak_infos,
    }


def plot_linearized_pf_raster_directional(
    cell_idx,
    x_neural, y_neural, speed, frame_rate,
    refined_SS, all_CS_spikes, all_spikes,
    PC_output_all,
    traces=None,
    complex_bursts_dicts=None,
    width=35.5, height=20.0,
    corner_radius=5.0,
    corridor_width=5.0,
    bin_size=1.5,
    speed_threshold=3.0,
    kernel_size=51,
    filter_type='boxcar',
    min_duration_s=0.25,
    merge_gap_s=0.0,
    smooth_sigma=2,
    moving_only=True,
    origin='corner',
    figsize=(18, 16),
    trajectory_color='#cccccc',
    trajectory_alpha=0.3,
    ss_color='#026C80',
    cs_color='#EE9B00',
    plateau_color='red',
    spike_size=3,
    time_unit='min',
    show_rate_map=True,
    show_subthreshold=True,
    theta_freqs=(4, 8),
    slow_freqs=2,
    plateau_min_duration_ms=100,
    plateaus_dicts=None,
    core_radius=5.0,
    buffer_width=10.0,
    min_speed=2.0,
    reversal_threshold=5.0,
    normalize_slow_vm=True,
    baseline_distance=10.0,
    verbose=False,
    delete_plateau_spikes=False,
):
    """
    Plot linearized PF raster with 3 columns: All, CW, CCW directions.

    Traversal Detection Parameters (3-Zone State Machine)
    ------------------------------------------------------
    Zones: Left Buffer -> Core -> Right Buffer
    CW:  Left Buffer [-15,-5) -> Core [-5,5] -> Right Buffer (5,15]
    CCW: Right Buffer -> Core -> Left Buffer

    core_radius : float
        Radius of the core zone around PF center. Default 5 cm.
    buffer_width : float
        Width of buffer zones on each side. Default 10 cm.
    min_speed : float
        Minimum average speed for a valid traversal. Default 2 cm/s.
    reversal_threshold : float
        Distance to move backward to trigger U-turn detection. Default 5 cm.

    Parameters
    ----------
    cell_idx : int
        Cell index.
    x_neural, y_neural : ndarray
        Position coordinates.
    speed : ndarray
        Speed array.
    frame_rate : float
        Sampling rate in Hz.
    refined_SS : list
        List of simple spike times for each cell.
    all_CS_spikes : list
        List of complex spike times for each cell.
    all_spikes : list
        List of all spike times for each cell.
    PC_output_all : list
        Place cell analysis output for each cell.
    traces : list, optional
        List of voltage traces for each cell (for theta/slow Vm computation).
    complex_bursts_dicts : list, optional
        List of complex burst dicts for each cell.
    approach_threshold : float
        Maximum distance from PF center to count as a valid pass (default 5 cm).
    direction_smooth_window : int
        Window for smoothing velocity when computing direction.
    (other parameters same as plot_linearized_pf_raster_for_cell)

    Returns
    -------
    dict with figure, axes, and data for each direction.
    """
    # Check if cell is a place cell
    analysis = PC_output_all[cell_idx]
    is_place_cell = analysis.get('is_place_cell', False)
    _do_delete_plateau_cs = delete_plateau_spikes  # local flag for this function

    # Get spike times
    spike_times_all = np.asarray(all_spikes[cell_idx], dtype=int)
    spike_times_ss = np.asarray(refined_SS[cell_idx], dtype=int)
    spike_times_cs = np.asarray(all_CS_spikes[cell_idx], dtype=int)

    # Extract plateau bursts
    plateau_bursts = None
    if plateaus_dicts is not None and cell_idx < len(plateaus_dicts):
        _plat = plateaus_dicts[cell_idx]
        if _plat is not None:
            _ps = np.asarray(_plat.get('starts', []), dtype=int)
            _pe = np.asarray(_plat.get('ends', []), dtype=int)
            if _ps.size > 0:
                plateau_bursts = {'starts': _ps, 'ends': _pe}
    elif complex_bursts_dicts is not None and cell_idx < len(complex_bursts_dicts):
        bursts = complex_bursts_dicts[cell_idx]
        cb_starts = np.asarray(bursts.get('starts', []), dtype=int)
        cb_ends = np.asarray(bursts.get('ends', []), dtype=int)
        cb_durs = np.asarray(bursts.get('durations_ms', []), dtype=float)
        plateau_mask = cb_durs >= plateau_min_duration_ms
        if np.any(plateau_mask):
            plateau_bursts = {
                'starts': cb_starts[plateau_mask],
                'ends': cb_ends[plateau_mask],
            }

    # Remove CS spikes that fall within plateau windows
    if _do_delete_plateau_cs and plateau_bursts is not None and spike_times_cs.size > 0:
        p_starts = plateau_bursts['starts']
        p_ends = plateau_bursts['ends']
        in_plateau = np.zeros(len(spike_times_cs), dtype=bool)
        for ps, pe in zip(p_starts, p_ends):
            in_plateau |= (spike_times_cs >= ps) & (spike_times_cs <= pe)
        n_removed = np.sum(in_plateau)
        spike_times_cs = spike_times_cs[~in_plateau]
        if verbose:
            print(f"  delete_plateau_spikes: removed {n_removed} CS from plateaus, "
                  f"{len(spike_times_cs)} CS remaining")

    # Choose spike times for PF peak: use SS spikes if cell is not an
    # all-spikes place cell but is an SS place cell
    is_place_cell_ss = analysis.get('is_place_cell_ss', False)
    is_place_cell_cs = analysis.get('is_place_cell_cs', False)
    is_any_place_cell = is_place_cell or is_place_cell_ss or is_place_cell_cs
    if is_place_cell:
        pf_spikes = spike_times_all
    elif is_place_cell_ss:
        pf_spikes = spike_times_ss
    else:
        pf_spikes = spike_times_all

    # Compute linearized PF peak
    pf_peak_info = compute_linearized_pf_peak(
        x_neural, y_neural, pf_spikes, speed, frame_rate,
        width=width, height=height,
        corner_radius=corner_radius,
        corridor_width=corridor_width,
        bin_size=bin_size,
        speed_threshold=speed_threshold,
        kernel_size=kernel_size,
        filter_type=filter_type,
        min_duration_s=min_duration_s,
        merge_gap_s=merge_gap_s,
        smooth_sigma=smooth_sigma,
        origin=origin,
    )

    pf_peak_pos = pf_peak_info['peak_linear_pos']
    perimeter = pf_peak_info['perimeter']
    bin_centers = pf_peak_info['bin_centers']

    if np.isnan(pf_peak_pos):
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No valid PF peak', transform=ax.transAxes,
                ha='center', va='center', fontsize=12)
        return {
            'fig': fig,
            'axes': [[ax]],
            'pf_peak_info': pf_peak_info,
            'direction_results': None,
        }

    # Compute centered position for direction detection
    # First linearize all positions
    if origin == 'corner':
        x_center = np.asarray(x_neural) - width / 2
        y_center = np.asarray(y_neural) - height / 2
    else:
        x_center = np.asarray(x_neural, dtype=float)
        y_center = np.asarray(y_neural, dtype=float)

    lin_result = linearize_position_rounded_corridor(
        x_center, y_center, width, height, corner_radius, corridor_width, origin='center'
    )
    linear_pos = lin_result['linear_pos']

    # Center at PF peak
    linear_pos_centered = center_linear_position(linear_pos, pf_peak_pos, perimeter)

    # Detect traversals using 3-zone state machine
    pass_result = detect_pf_traversals(
        linear_pos_centered,
        speed=speed,
        core_radius=core_radius,
        buffer_width=buffer_width,
        min_speed=min_speed,
        reversal_threshold=reversal_threshold,
        verbose=verbose,
    )
    valid_pass_mask = pass_result['valid_mask']
    pass_info = pass_result['pass_info']

    # Create direction masks based on complete passes (not frame-by-frame velocity)
    # This avoids issues with velocity jitter around 0
    n_frames = len(linear_pos_centered)
    mask_cw = np.zeros(n_frames, dtype=bool)
    mask_ccw = np.zeros(n_frames, dtype=bool)

    # Mark frames from each pass with its direction
    for p in pass_info:
        start_idx = p['start']
        end_idx = p['end']
        if p['direction'] == 'cw':
            mask_cw[start_idx:end_idx] = True
        elif p['direction'] == 'ccw':
            mask_ccw[start_idx:end_idx] = True

    # Count passes and frames by direction
    n_cw = sum(1 for p in pass_info if p['direction'] == 'cw')
    n_ccw = sum(1 for p in pass_info if p['direction'] == 'ccw')
    n_cw_frames = np.sum(mask_cw)
    n_ccw_frames = np.sum(mask_ccw)

    # Determine subplot layout
    n_rows = 1  # raster
    if show_rate_map:
        n_rows += 1
    if show_subthreshold and traces is not None:
        n_rows += 2  # theta + slow

    # Create figure with 3 columns
    height_ratios = [4] + [1] * (n_rows - 1)
    fig, axes = plt.subplots(n_rows, 3, figsize=figsize,
                             gridspec_kw={'height_ratios': height_ratios},
                             sharex='col', sharey='row')

    if n_rows == 1:
        axes = axes.reshape(1, 3)

    direction_labels = ['All', 'CW', 'CCW']
    direction_masks_for_maps = [None, mask_cw, mask_ccw]  # None = all moving (no additional filter)
    # Use pass counts for display
    n_total_passes = n_cw + n_ccw
    direction_pass_counts = [n_total_passes, n_cw, n_ccw]

    # Compute bin centers centered at PF peak
    bin_centers_centered = center_linear_position(bin_centers, pf_peak_pos, perimeter)
    sort_idx = np.argsort(bin_centers_centered)
    bin_centers_sorted = bin_centers_centered[sort_idx]

    # --- Determine place field region from 2D PF mask ---
    pf_left_boundary = None
    pf_right_boundary = None
    has_pf_region = False

    if normalize_slow_vm:
        pf_mask_2d = analysis.get('place_field_mask', None)
        if pf_mask_2d is not None and np.any(pf_mask_2d):
            # Reconstruct 2D bin edges from analysis params
            params_2d = analysis.get('params', {})
            bin_size_2d = params_2d.get('bin_size', 1.5)
            width_2d = params_2d.get('width_real', width)
            height_2d = params_2d.get('height_real', height)

            x_edges_2d = np.arange(0, width_2d + bin_size_2d, bin_size_2d)
            y_edges_2d = np.arange(0, height_2d + bin_size_2d, bin_size_2d)
            x_centers_2d = (x_edges_2d[:-1] + x_edges_2d[1:]) / 2
            y_centers_2d = (y_edges_2d[:-1] + y_edges_2d[1:]) / 2

            # Get (x, y) centers of all PF bins
            pf_ix, pf_iy = np.where(pf_mask_2d)
            pf_x = x_centers_2d[pf_ix]
            pf_y = y_centers_2d[pf_iy]

            # Convert to center-origin for linearization
            if origin == 'corner':
                pf_x_c = pf_x - width / 2
                pf_y_c = pf_y - height / 2
            else:
                pf_x_c = pf_x.copy()
                pf_y_c = pf_y.copy()

            # Linearize the PF bin centers
            pf_lin = linearize_position_rounded_corridor(
                pf_x_c, pf_y_c, width, height,
                corner_radius, corridor_width, origin='center'
            )
            pf_lpos = pf_lin['linear_pos']
            pf_in_corr = pf_lin['in_corridor']
            valid_pf = pf_in_corr & np.isfinite(pf_lpos)

            if np.any(valid_pf):
                pf_centered = center_linear_position(
                    pf_lpos[valid_pf], pf_peak_pos, perimeter)
                pf_left_boundary = np.min(pf_centered) - bin_size_2d / 2
                pf_right_boundary = np.max(pf_centered) + bin_size_2d / 2
                has_pf_region = True

    # --- Precompute per-traversal normalized slow Vm ---
    normalized_slow_vm = None
    if normalize_slow_vm and has_pf_region and traces is not None and cell_idx < len(traces):
        from scipy.ndimage import gaussian_filter1d

        trace_arr = np.asarray(traces[cell_idx], dtype=float)
        trace_nan_mask = np.isnan(trace_arr)
        trace_arr = interpolate_nan_segment(trace_arr)
        slow_vm_signal = lowpass_filter(trace_arr, slow_freqs, frame_rate, order=5)
        slow_vm_signal[trace_nan_mask] = np.nan

        normalized_slow_vm = np.full(n_frames, np.nan)
        baseline_side_per_pass = []  # 'left', 'right', or None per traversal
        for p in pass_info:
            trav_frames = np.arange(p['start'], p['end'])
            trav_centered = linear_pos_centered[trav_frames]

            # Compute baseline from BOTH sides of the PF, take the minimum
            # Handle circular wrapping: baseline region may cross ±perimeter/2
            half_p = perimeter / 2

            # Left baseline: region just before PF left boundary
            left_lo = pf_left_boundary - baseline_distance
            left_hi = pf_left_boundary
            if left_lo < -half_p:
                bl_mask_left = ((trav_centered >= left_lo + perimeter) |
                                (trav_centered < left_hi))
            else:
                bl_mask_left = ((trav_centered >= left_lo) &
                                (trav_centered < left_hi))

            # Right baseline: region just after PF right boundary
            right_lo = pf_right_boundary
            right_hi = pf_right_boundary + baseline_distance
            if right_hi > half_p:
                bl_mask_right = ((trav_centered > right_lo) |
                                 (trav_centered <= right_hi - perimeter))
            else:
                bl_mask_right = ((trav_centered > right_lo) &
                                 (trav_centered <= right_hi))

            left_frames = trav_frames[bl_mask_left]
            right_frames = trav_frames[bl_mask_right]
            bl_left = np.nanmean(slow_vm_signal[left_frames]) if len(left_frames) > 0 else np.nan
            bl_right = np.nanmean(slow_vm_signal[right_frames]) if len(right_frames) > 0 else np.nan
            baseline_val = np.nanmin([bl_left, bl_right])

            # Track which side was chosen
            if np.isnan(bl_left) and np.isnan(bl_right):
                baseline_side_per_pass.append(None)
            elif np.isnan(bl_right) or (np.isfinite(bl_left) and bl_left <= bl_right):
                baseline_side_per_pass.append('left')
            else:
                baseline_side_per_pass.append('right')

            normalized_slow_vm[trav_frames] = slow_vm_signal[trav_frames] - baseline_val

        # Precompute binning info for normalized slow Vm maps
        bins_info_norm = compute_linearization_bins(width, height, corner_radius, bin_size)
        bin_edges_norm = bins_info_norm['bin_edges']
        n_bins_norm = bins_info_norm['n_bins']
        bin_idx_norm = np.digitize(linear_pos, bin_edges_norm) - 1
        bin_idx_norm = np.clip(bin_idx_norm, 0, n_bins_norm - 1)

    # Helper: shade baseline region on an axis (handles circular wrapping)
    # side: 'left', 'right', or 'both'
    def _shade_baseline(ax, pf_left, pf_right, bl_dist, peri, side='both',
                        color='gray', alpha=0.08):
        half = peri / 2
        if side in ('left', 'both'):
            left_lo = pf_left - bl_dist
            left_hi = pf_left
            if left_lo < -half:
                ax.axvspan(-half, left_hi, alpha=alpha, color=color, zorder=0)
                ax.axvspan(left_lo + peri, half, alpha=alpha, color=color, zorder=0)
            else:
                ax.axvspan(left_lo, left_hi, alpha=alpha, color=color, zorder=0)
        if side in ('right', 'both'):
            right_lo = pf_right
            right_hi = pf_right + bl_dist
            if right_hi > half:
                ax.axvspan(right_lo, half, alpha=alpha, color=color, zorder=0)
                ax.axvspan(-half, right_hi - peri, alpha=alpha, color=color, zorder=0)
            else:
                ax.axvspan(right_lo, right_hi, alpha=alpha, color=color, zorder=0)

    # Determine dominant baseline side per direction for shading
    _bl_side_by_dir = {}  # 'all', 'cw', 'ccw' -> 'left'/'right'/'both'
    if normalize_slow_vm and has_pf_region and len(pass_info) > 0:
        for dir_key in ('all', 'cw', 'ccw'):
            sides = [s for p, s in zip(pass_info, baseline_side_per_pass)
                     if s is not None and (dir_key == 'all' or p['direction'] == dir_key)]
            n_left = sides.count('left')
            n_right = sides.count('right')
            if n_left > 0 and n_right == 0:
                _bl_side_by_dir[dir_key] = 'left'
            elif n_right > 0 and n_left == 0:
                _bl_side_by_dir[dir_key] = 'right'
            elif n_left > 0 and n_right > 0:
                _bl_side_by_dir[dir_key] = 'both'
            else:
                _bl_side_by_dir[dir_key] = 'both'

    direction_results = {}

    for col_idx, (label, dir_mask, n_passes) in enumerate(zip(direction_labels, direction_masks_for_maps, direction_pass_counts)):
        ax_raster = axes[0, col_idx]

        # Plot raster
        raster_result = plot_linearized_pf_raster(
            x_neural, y_neural, speed, frame_rate,
            spike_times_ss, spike_times_cs,
            pf_peak_pos, perimeter,
            width=width, height=height,
            corner_radius=corner_radius,
            corridor_width=corridor_width,
            speed_threshold=speed_threshold,
            kernel_size=kernel_size,
            filter_type=filter_type,
            min_duration_s=min_duration_s,
            merge_gap_s=merge_gap_s,
            moving_only=moving_only,
            origin=origin,
            ax=ax_raster,
            trajectory_color=trajectory_color,
            trajectory_alpha=trajectory_alpha,
            ss_color=ss_color,
            cs_color=cs_color,
            spike_size=spike_size,
            time_unit=time_unit,
            title=f'{label} (n={n_passes})' if col_idx > 0 else f'Cell {cell_idx + 1} - {label}' + (' (PC)' if is_place_cell else ''),
            plateau_bursts=plateau_bursts,
            plateau_color=plateau_color,
            direction_mask=dir_mask,  # Filter raster by direction
            center_line_color='magenta' if is_any_place_cell else 'gray',
        )

        # Shade place field region on raster
        if has_pf_region:
            ax_raster.axvspan(pf_left_boundary, pf_right_boundary,
                              alpha=0.10, color='red', zorder=0)
            if normalize_slow_vm:
                _shade_baseline(ax_raster, pf_left_boundary, pf_right_boundary,
                                baseline_distance, perimeter,
                                side=_bl_side_by_dir.get(label.lower(), 'both'))

        # Remove x-label from raster (shared with subplots below)
        if n_rows > 1:
            ax_raster.set_xlabel('')

        row_idx = 1

        # Compute and plot rate maps
        if show_rate_map:
            ax_rate = axes[row_idx, col_idx]
            row_idx += 1

            # SS rate map
            ss_rate_info = compute_linearized_rate_map(
                x_neural, y_neural, spike_times_ss, speed, frame_rate,
                width=width, height=height,
                corner_radius=corner_radius,
                corridor_width=corridor_width,
                bin_size=bin_size,
                speed_threshold=speed_threshold,
                kernel_size=kernel_size,
                filter_type=filter_type,
                min_duration_s=min_duration_s,
                merge_gap_s=merge_gap_s,
                smooth_sigma=smooth_sigma,
                origin=origin,
                direction_mask=dir_mask,
            )

            # CS rate map
            cs_rate_info = compute_linearized_rate_map(
                x_neural, y_neural, spike_times_cs, speed, frame_rate,
                width=width, height=height,
                corner_radius=corner_radius,
                corridor_width=corridor_width,
                bin_size=bin_size,
                speed_threshold=speed_threshold,
                kernel_size=kernel_size,
                filter_type=filter_type,
                min_duration_s=min_duration_s,
                merge_gap_s=merge_gap_s,
                smooth_sigma=smooth_sigma,
                origin=origin,
                direction_mask=dir_mask,
            )

            ss_rate_sorted = ss_rate_info['rate_map'][sort_idx]
            cs_rate_sorted = cs_rate_info['rate_map'][sort_idx]

            ax_rate.fill_between(bin_centers_sorted, 0, ss_rate_sorted,
                                 alpha=0.4, color=ss_color, label='SS')
            ax_rate.plot(bin_centers_sorted, ss_rate_sorted, color=ss_color, linewidth=1)
            ax_rate.fill_between(bin_centers_sorted, 0, cs_rate_sorted,
                                 alpha=0.4, color=cs_color, label='CS')
            ax_rate.plot(bin_centers_sorted, cs_rate_sorted, color=cs_color, linewidth=1)
            _cl_color = 'magenta' if is_any_place_cell else 'gray'
            ax_rate.axvline(0, color=_cl_color, linestyle='--', linewidth=1, alpha=0.7)
            ax_rate.set_xlim(-perimeter / 2, perimeter / 2)
            if has_pf_region:
                ax_rate.axvspan(pf_left_boundary, pf_right_boundary,
                                alpha=0.10, color='red', zorder=0)
                if normalize_slow_vm:
                    _shade_baseline(ax_rate, pf_left_boundary, pf_right_boundary,
                                    baseline_distance, perimeter,
                                    side=_bl_side_by_dir.get(label.lower(), 'both'))
            if col_idx == 0:
                ax_rate.set_ylabel('Rate (Hz)')
            if col_idx == 2:
                ax_rate.legend(loc='lower right', fontsize=7)

        # Compute and plot subthreshold maps
        if show_subthreshold and traces is not None and cell_idx < len(traces):
            subthreshold_maps = compute_linearized_subthreshold_maps(
                x_neural, y_neural, traces[cell_idx], speed, frame_rate,
                width=width, height=height,
                corner_radius=corner_radius,
                corridor_width=corridor_width,
                bin_size=bin_size,
                speed_threshold=speed_threshold,
                kernel_size=kernel_size,
                filter_type=filter_type,
                min_duration_s=min_duration_s,
                merge_gap_s=merge_gap_s,
                theta_freqs=theta_freqs,
                slow_freqs=slow_freqs,
                smooth_sigma=smooth_sigma,
                origin=origin,
                direction_mask=dir_mask,
            )

            # Theta plot
            ax_theta = axes[row_idx, col_idx]
            row_idx += 1
            theta_sorted = subthreshold_maps['theta_map'][sort_idx]
            ax_theta.fill_between(bin_centers_sorted, 0, theta_sorted,
                                  alpha=0.4, color='blue')
            ax_theta.plot(bin_centers_sorted, theta_sorted, color='blue', linewidth=1)
            ax_theta.axvline(0, color=_cl_color, linestyle='--', linewidth=1, alpha=0.7)
            ax_theta.set_xlim(-perimeter / 2, perimeter / 2)
            if has_pf_region:
                ax_theta.axvspan(pf_left_boundary, pf_right_boundary,
                                 alpha=0.10, color='red', zorder=0)
                if normalize_slow_vm:
                    _shade_baseline(ax_theta, pf_left_boundary, pf_right_boundary,
                                    baseline_distance, perimeter,
                                    side=_bl_side_by_dir.get(label.lower(), 'both'))
            if col_idx == 0:
                ax_theta.set_ylabel('Theta amp')

            # Slow Vm plot
            ax_slow = axes[row_idx, col_idx]

            # Use normalized slow Vm if available, otherwise use raw
            if normalized_slow_vm is not None:
                # Bin normalized slow Vm for this direction
                if dir_mask is not None:
                    valid_norm = (dir_mask &
                                  np.isfinite(normalized_slow_vm) &
                                  np.isfinite(linear_pos))
                else:
                    valid_norm = ((mask_cw | mask_ccw) &
                                  np.isfinite(normalized_slow_vm) &
                                  np.isfinite(linear_pos))

                norm_slow_sum = np.zeros(n_bins_norm, dtype=float)
                norm_slow_count = np.zeros(n_bins_norm, dtype=int)
                for idx in np.where(valid_norm)[0]:
                    bi = bin_idx_norm[idx]
                    norm_slow_sum[bi] += normalized_slow_vm[idx]
                    norm_slow_count[bi] += 1

                with np.errstate(divide='ignore', invalid='ignore'):
                    norm_slow_map = np.where(
                        norm_slow_count > 0,
                        norm_slow_sum / norm_slow_count, np.nan)

                if smooth_sigma is not None and smooth_sigma > 0:
                    valid_s = np.isfinite(norm_slow_map)
                    if np.any(valid_s):
                        arr_filled = np.where(valid_s, norm_slow_map, 0)
                        weights = valid_s.astype(float)
                        arr_sm = gaussian_filter1d(arr_filled, smooth_sigma,
                                                   mode='wrap')
                        w_sm = gaussian_filter1d(weights, smooth_sigma,
                                                 mode='wrap')
                        norm_slow_map = np.where(
                            w_sm > 0, arr_sm / w_sm, np.nan)

                slow_sorted = norm_slow_map[sort_idx]
            else:
                slow_sorted = subthreshold_maps['slow_map'][sort_idx]

            ax_slow.plot(bin_centers_sorted, slow_sorted,
                         color='red', linewidth=1)
            if normalized_slow_vm is not None:
                ax_slow.axhline(0, color='gray', linestyle='--',
                                linewidth=0.8, alpha=0.7)
            ax_slow.axvline(0, color=_cl_color, linestyle='--',
                            linewidth=1, alpha=0.7)
            ax_slow.set_xlim(-perimeter / 2, perimeter / 2)
            if has_pf_region:
                ax_slow.axvspan(pf_left_boundary, pf_right_boundary,
                                alpha=0.10, color='red', zorder=0)
                if normalize_slow_vm:
                    _shade_baseline(ax_slow, pf_left_boundary, pf_right_boundary,
                                    baseline_distance, perimeter,
                                    side=_bl_side_by_dir.get(label.lower(), 'both'))
            if col_idx == 0:
                ax_slow.set_ylabel(
                    'Slow Vm (norm)' if normalized_slow_vm is not None
                    else 'Slow Vm')
            ax_slow.set_xlabel('Position rel. to PF peak (cm)')

        direction_results[label.lower()] = {
            'raster_result': raster_result,
            'n_passes': n_passes,
            'direction_mask': dir_mask,
        }

    plt.tight_layout()

    return {
        'fig': fig,
        'axes': axes,
        'pf_peak_info': pf_peak_info,
        'direction_results': direction_results,
        'pass_info': pass_info,
        'n_cw_passes': n_cw,
        'n_ccw_passes': n_ccw,
        'n_cw_frames': n_cw_frames,
        'n_ccw_frames': n_ccw_frames,
        'pf_region': {
            'has_pf_region': has_pf_region,
            'pf_left_boundary': pf_left_boundary,
            'pf_right_boundary': pf_right_boundary,
        } if has_pf_region else None,
    }
