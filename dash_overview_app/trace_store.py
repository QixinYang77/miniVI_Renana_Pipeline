import hashlib
import numpy as np

from signal_ops import (
    make_bad_mask,
    baseline_correct_trace,
    highpass_filter_trace,
    minmax_envelope,
    motion_regress_traces_chunked,
)


def _epochs_hash(epochs):
    if not epochs:
        return "none"
    parts = [f"{float(a):.6f}:{float(b):.6f}" for a, b in epochs]
    raw = "|".join(parts).encode("utf-8")
    return hashlib.md5(raw).hexdigest()


class TraceStore:
    def __init__(self, bundle):
        self.bundle = bundle
        self.t = np.asarray(bundle["t"], dtype=np.float32)
        self.sources = bundle["sources_raw"]
        self.shift_xy = bundle.get("shift_xy", None)
        self.frame_rate = float(bundle["frame_rate"])
        self._cache = {}
        self._motion_cache = {}

    def _window_indices(self, start_t, window_s):
        t0 = float(max(self.t[0], start_t))
        t1 = float(min(self.t[-1], t0 + max(0.1, float(window_s))))
        i0 = int(np.searchsorted(self.t, t0, side="left"))
        i1 = int(np.searchsorted(self.t, t1, side="right"))
        i0 = max(0, min(i0, self.t.size))
        i1 = max(i0 + 1, min(i1, self.t.size))
        return i0, i1

    def get_window_data(
        self,
        source_key,
        cells,
        start_t,
        window_s,
        bad_epochs=None,
        normalize="zscore",
        baseline_cfg=None,
        baseline_display_cfg=None,
        highpass_cfg=None,
        motion_regression_cfg=None,
        target_points=0,
        hide_bad_epochs=True,
    ):
        baseline_cfg = baseline_cfg or {"enabled": False}
        baseline_display_cfg = baseline_display_cfg or {}
        highpass_cfg = highpass_cfg or {}
        motion_regression_cfg = motion_regression_cfg or {"enabled": False}
        bad_epochs = bad_epochs or []
        source = self.sources[source_key]

        i0, i1 = self._window_indices(start_t=start_t, window_s=window_s)
        t_win = self.t[i0:i1]
        bad_mask_local = make_bad_mask(t_win, bad_epochs)
        epochs_key = _epochs_hash(bad_epochs)

        base_sig = (
            bool(baseline_cfg.get("enabled", False)),
            float(baseline_cfg.get("percentile", 50.0)),
            float(baseline_cfg.get("window_s", 60.0)),
            str(baseline_cfg.get("smooth", "median")),
        )
        base_disp_sig = (
            bool(baseline_display_cfg.get("fit", False)),
            bool(baseline_display_cfg.get("remove", False)),
            float(baseline_display_cfg.get("percentile", 50.0)),
            float(baseline_display_cfg.get("window_s", 30.0)),
            str(baseline_display_cfg.get("smooth", "median")),
        )
        hp_sig = (
            bool(highpass_cfg.get("enabled", False)),
            float(highpass_cfg.get("cutoff_hz", 20.0)),
        )
        motion_sig = (
            bool(motion_regression_cfg.get("enabled", False)),
            float(motion_regression_cfg.get("pre_baseline_window_s", 30.0)),
            int(motion_regression_cfg.get("chunk_frames", 5000)),
            int(motion_regression_cfg.get("overlap_frames", 0)),
            float(motion_regression_cfg.get("ridge_lambda", 0.0)),
            int(motion_regression_cfg.get("n_components", 5)),
            int(motion_regression_cfg.get("shift_smooth_frames", 0)),
            bool(motion_regression_cfg.get("show_fitted", False)),
            int(motion_regression_cfg.get("refresh_nonce", 0)),
            bool(self.shift_xy is not None),
        )
        key = (
            source_key,
            tuple(int(c) for c in cells),
            int(i0),
            int(i1),
            str(normalize),
            int(max(0, target_points)),
            bool(hide_bad_epochs),
            epochs_key,
            base_sig,
            base_disp_sig,
            hp_sig,
            motion_sig,
        )
        if key in self._cache:
            return self._cache[key]

        fit_display_baseline = bool(baseline_display_cfg.get("fit", False))
        remove_baseline_drift = bool(baseline_display_cfg.get("remove", False))
        display_percentile = float(baseline_display_cfg.get("percentile", 50.0))
        display_window_s = float(baseline_display_cfg.get("window_s", 30.0))
        display_smooth = str(baseline_display_cfg.get("smooth", "median"))
        hp_enabled = bool(highpass_cfg.get("enabled", False))
        hp_cutoff = float(highpass_cfg.get("cutoff_hz", 20.0))
        motion_enabled = bool(motion_regression_cfg.get("enabled", False))
        motion_prebaseline_window_s = float(
            max(0.0, float(motion_regression_cfg.get("pre_baseline_window_s", 30.0)))
        )
        motion_chunk = int(max(1, int(motion_regression_cfg.get("chunk_frames", 5000))))
        motion_overlap = int(max(0, int(motion_regression_cfg.get("overlap_frames", 0))))
        motion_ridge_lambda = float(max(0.0, float(motion_regression_cfg.get("ridge_lambda", 0.0))))
        motion_n_components = int(max(1, int(motion_regression_cfg.get("n_components", 5))))
        motion_shift_smooth = int(max(0, int(motion_regression_cfg.get("shift_smooth_frames", 0))))
        motion_show_fitted = bool(motion_regression_cfg.get("show_fitted", False))
        if motion_overlap >= motion_chunk:
            motion_overlap = max(0, motion_chunk - 1)

        use_saved_baseline_cfg = bool(baseline_cfg.get("enabled", False)) and not fit_display_baseline

        # Optional motion regression (display-only): mirrors chunk+overlap logic from
        # motion_regression_masked.py but applied to full trace matrix and then sliced for view.
        source_full_raw = source.astype(np.float32, copy=False)
        source_win_raw = source_full_raw[i0:i1, :].astype(np.float32, copy=True)
        source_win_denoised = None
        source_win = source_win_raw.copy()
        motion_applied = False
        motion_info = "OFF"
        compute_motion = bool(motion_enabled or motion_show_fitted)
        if compute_motion:
            if self.shift_xy is not None:
                sxy = np.asarray(self.shift_xy, dtype=np.float32)
                if sxy.ndim == 2 and sxy.shape[1] >= 2 and sxy.shape[0] >= source_full_raw.shape[0]:
                    sxy_full = sxy[: source_full_raw.shape[0], :2]
                    valid_rows = np.isfinite(sxy_full[:, 0]) & np.isfinite(sxy_full[:, 1])
                    if int(np.count_nonzero(valid_rows)) >= 3:
                        refresh_nonce = int(motion_regression_cfg.get("refresh_nonce", 0))
                        motion_cache_key = (
                            source_key,
                            epochs_key,
                            float(motion_prebaseline_window_s),
                            int(motion_chunk),
                            int(motion_overlap),
                            float(motion_ridge_lambda),
                            int(motion_n_components),
                            int(motion_shift_smooth),
                            bool(motion_show_fitted),
                            int(refresh_nonce),
                        )
                        cached = self._motion_cache.get(motion_cache_key, None)
                        if cached is None:
                            bad_mask_global = make_bad_mask(self.t, bad_epochs)
                            source_full_fit = source_full_raw.astype(np.float32, copy=True)
                            if motion_prebaseline_window_s > 0:
                                for ci_all in range(source_full_fit.shape[1]):
                                    source_full_fit[:, ci_all], _ = baseline_correct_trace(
                                        source_full_fit[:, ci_all],
                                        self.frame_rate,
                                        percentile=50.0,
                                        window_s=motion_prebaseline_window_s,
                                        smooth="median",
                                        bad_mask=bad_mask_global,
                                    )

                            source_full_denoised_fit = motion_regress_traces_chunked(
                                source_full_fit,
                                sxy_full,
                                chunk_frames=motion_chunk,
                                overlap_frames=motion_overlap,
                                ridge_lambda=motion_ridge_lambda,
                                n_components=motion_n_components,
                                shift_smooth_frames=motion_shift_smooth,
                                bad_mask=bad_mask_global,
                            )
                            motion_term_full = (source_full_fit - source_full_denoised_fit).astype(
                                np.float32, copy=False
                            )
                            source_full_denoised = (source_full_raw - motion_term_full).astype(
                                np.float32, copy=False
                            )
                            cached = {
                                "denoised": source_full_denoised.astype(np.float32, copy=False),
                                "motion_term": motion_term_full,
                            }
                            self._motion_cache[motion_cache_key] = cached
                            if len(self._motion_cache) > 6:
                                self._motion_cache = dict(list(self._motion_cache.items())[-4:])

                        source_full_denoised = cached["denoised"]
                        source_win_denoised = source_full_denoised[i0:i1, :].astype(np.float32, copy=False)
                        if motion_enabled:
                            source_win = source_win_denoised
                            motion_applied = True
                            motion_info = (
                                f"ON (pre-base={motion_prebaseline_window_s:g}s, chunk={motion_chunk}, "
                                f"overlap={motion_overlap}, PCs={motion_n_components}, ridge={motion_ridge_lambda:g}, "
                                f"smooth={motion_shift_smooth})"
                            )
                        else:
                            motion_info = (
                                f"PREVIEW (pre-base={motion_prebaseline_window_s:g}s, chunk={motion_chunk}, "
                                f"overlap={motion_overlap}, PCs={motion_n_components}, ridge={motion_ridge_lambda:g}, "
                                f"smooth={motion_shift_smooth})"
                            )
                    else:
                        motion_info = "OFF (reg_shifts unavailable)"
                else:
                    motion_info = "OFF (reg_shifts unavailable)"
            else:
                motion_info = "OFF (reg_shifts unavailable)"

        traces = []
        for ci in cells:
            y_raw = source_win[:, int(ci)].astype(np.float32, copy=True)
            y_work = y_raw.copy()
            baseline_raw = None

            def _apply_pre_norm_pipeline(y_in, with_baseline=False):
                y_proc = np.asarray(y_in, dtype=np.float32).copy()
                baseline_local = None
                if fit_display_baseline:
                    y_corr, baseline_local = baseline_correct_trace(
                        y_proc,
                        self.frame_rate,
                        percentile=display_percentile,
                        window_s=display_window_s,
                        smooth=display_smooth,
                        bad_mask=bad_mask_local,
                    )
                    if remove_baseline_drift:
                        y_proc = y_corr
                elif use_saved_baseline_cfg:
                    y_proc, _ = baseline_correct_trace(
                        y_proc,
                        self.frame_rate,
                        percentile=float(baseline_cfg.get("percentile", 50.0)),
                        window_s=float(baseline_cfg.get("window_s", 60.0)),
                        smooth=str(baseline_cfg.get("smooth", "median")),
                        bad_mask=bad_mask_local,
                    )

                if hp_enabled:
                    y_proc = highpass_filter_trace(
                        y_proc,
                        self.frame_rate,
                        cutoff_hz=hp_cutoff,
                        bad_mask=bad_mask_local,
                    )

                if np.any(bad_mask_local):
                    y_proc[bad_mask_local] = np.nan
                    if baseline_local is not None:
                        baseline_local = baseline_local.astype(np.float32, copy=False)
                        baseline_local[bad_mask_local] = np.nan

                if not with_baseline:
                    baseline_local = None
                return y_proc, baseline_local

            y_work, baseline_raw = _apply_pre_norm_pipeline(y_raw, with_baseline=True)

            motion_fit_raw = None
            if motion_show_fitted and (source_win_denoised is not None):
                y_ref_raw = source_win_raw[:, int(ci)].astype(np.float32, copy=True)
                y_den_raw = source_win_denoised[:, int(ci)].astype(np.float32, copy=True)
                y_ref_proc, _ = _apply_pre_norm_pipeline(y_ref_raw, with_baseline=False)
                y_den_proc, _ = _apply_pre_norm_pipeline(y_den_raw, with_baseline=False)
                motion_fit_raw = (y_ref_proc - y_den_proc).astype(np.float32, copy=False)

            if normalize == "0-1":
                mn = np.nanmin(y_work)
                mx = np.nanmax(y_work)
                if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
                    y = np.zeros_like(y_work, dtype=np.float32)

                    def _norm_arr_full(arr):
                        return np.zeros_like(arr, dtype=np.float32)

                    def _norm_component(arr):
                        return np.zeros_like(arr, dtype=np.float32)

                else:
                    scale = mx - mn
                    y = (y_work - mn) / scale

                    def _norm_arr_full(arr):
                        return (arr - mn) / scale

                    def _norm_component(arr):
                        return arr / scale

            else:
                m = np.nanmean(y_work)
                s = np.nanstd(y_work)
                denom = s if s > 0 else 1.0
                y = (y_work - m) / denom

                def _norm_arr_full(arr):
                    return (arr - m) / denom

                def _norm_component(arr):
                    return arr / denom

            baseline_plot = None
            if (
                baseline_raw is not None
                and (not remove_baseline_drift)
                and (not hp_enabled)
            ):
                baseline_plot = _norm_arr_full(baseline_raw).astype(np.float32, copy=False)

            motion_fit_plot = None
            if motion_fit_raw is not None:
                motion_fit_plot = _norm_arr_full(motion_fit_raw).astype(np.float32, copy=False)

            if np.any(bad_mask_local):
                if hide_bad_epochs:
                    keep = ~bad_mask_local
                    if not np.any(keep):
                        x_plot = t_win
                        y_plot = y
                        baseline_plot_out = baseline_plot
                        motion_fit_plot_out = motion_fit_plot
                    else:
                        x_plot = t_win[keep]
                        y_plot = y[keep]
                        baseline_plot_out = baseline_plot[keep] if baseline_plot is not None else None
                        motion_fit_plot_out = (
                            motion_fit_plot[keep] if motion_fit_plot is not None else None
                        )
                else:
                    # y already has NaN in bad regions (preserved through normalization)
                    x_plot = t_win
                    y_plot = y
                    baseline_plot_out = baseline_plot
                    motion_fit_plot_out = motion_fit_plot
            else:
                x_plot = t_win
                y_plot = y
                baseline_plot_out = baseline_plot
                motion_fit_plot_out = motion_fit_plot

            baseline_x_plot = x_plot
            if int(target_points) > 0 and baseline_plot_out is None:
                x_plot, y_plot = minmax_envelope(x_plot, y_plot, target_points=int(target_points))
                baseline_x_plot = x_plot

            traces.append(
                {
                    "cell": int(ci),
                    "x": x_plot,
                    "y": y_plot,
                    "baseline_x": baseline_x_plot if baseline_plot_out is not None else None,
                    "baseline_y": baseline_plot_out,
                    "motion_fit_x": x_plot if motion_fit_plot_out is not None else None,
                    "motion_fit_y": motion_fit_plot_out,
                }
            )

        out = {
            "i0": i0,
            "i1": i1,
            "t0": float(t_win[0]) if t_win.size else 0.0,
            "t1": float(t_win[-1]) if t_win.size else 0.0,
            "traces": traces,
            "bad_mask_local": bad_mask_local,
            "motion_applied": motion_applied,
            "motion_info": motion_info,
        }
        self._cache[key] = out
        if len(self._cache) > 300:
            self._cache = dict(list(self._cache.items())[-200:])
        return out
