import hashlib
import numpy as np

from signal_ops import make_bad_mask, baseline_correct_trace, highpass_filter_trace, minmax_envelope


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
        self.frame_rate = float(bundle["frame_rate"])
        self._cache = {}

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
        target_points=0,
        hide_bad_epochs=True,
    ):
        baseline_cfg = baseline_cfg or {"enabled": False}
        baseline_display_cfg = baseline_display_cfg or {}
        highpass_cfg = highpass_cfg or {}
        bad_epochs = bad_epochs or []
        source = self.sources[source_key]

        i0, i1 = self._window_indices(start_t=start_t, window_s=window_s)
        t_win = self.t[i0:i1]
        bad_mask_local = make_bad_mask(t_win, bad_epochs)

        base_sig = (
            bool(baseline_cfg.get("enabled", False)),
            float(baseline_cfg.get("percentile", 5.0)),
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
        key = (
            source_key,
            tuple(int(c) for c in cells),
            int(i0),
            int(i1),
            str(normalize),
            int(max(0, target_points)),
            bool(hide_bad_epochs),
            _epochs_hash(bad_epochs),
            base_sig,
            base_disp_sig,
            hp_sig,
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

        use_saved_baseline_cfg = bool(baseline_cfg.get("enabled", False)) and not fit_display_baseline

        traces = []
        for ci in cells:
            y_raw = source[i0:i1, int(ci)].astype(np.float32, copy=True)
            y_work = y_raw.copy()
            baseline_raw = None

            if fit_display_baseline:
                y_corr, baseline_raw = baseline_correct_trace(
                    y_raw,
                    self.frame_rate,
                    percentile=display_percentile,
                    window_s=display_window_s,
                    smooth=display_smooth,
                    bad_mask=bad_mask_local,
                )
                if remove_baseline_drift:
                    y_work = y_corr
            elif use_saved_baseline_cfg:
                y_work, _ = baseline_correct_trace(
                    y_work,
                    self.frame_rate,
                    percentile=float(baseline_cfg.get("percentile", 5.0)),
                    window_s=float(baseline_cfg.get("window_s", 60.0)),
                    smooth=str(baseline_cfg.get("smooth", "median")),
                    bad_mask=bad_mask_local,
                )

            if hp_enabled:
                y_work = highpass_filter_trace(
                    y_work,
                    self.frame_rate,
                    cutoff_hz=hp_cutoff,
                    bad_mask=bad_mask_local,
                )

            # NaN bad epochs BEFORE normalization so noise doesn't
            # dominate the z-score / 0-1 scaling.
            if np.any(bad_mask_local):
                y_work[bad_mask_local] = np.nan

            if normalize == "0-1":
                mn = np.nanmin(y_work)
                mx = np.nanmax(y_work)
                if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
                    y = np.zeros_like(y_work, dtype=np.float32)

                    def _norm_arr(arr):
                        return np.zeros_like(arr, dtype=np.float32)

                else:
                    scale = mx - mn
                    y = (y_work - mn) / scale

                    def _norm_arr(arr):
                        return (arr - mn) / scale

            else:
                m = np.nanmean(y_work)
                s = np.nanstd(y_work)
                denom = s if s > 0 else 1.0
                y = (y_work - m) / denom

                def _norm_arr(arr):
                    return (arr - m) / denom

            baseline_plot = None
            if (
                baseline_raw is not None
                and (not remove_baseline_drift)
                and (not hp_enabled)
            ):
                baseline_plot = baseline_raw.astype(np.float32, copy=True)
                if np.any(bad_mask_local):
                    baseline_plot[bad_mask_local] = np.nan
                baseline_plot = _norm_arr(baseline_plot).astype(np.float32, copy=False)

            if np.any(bad_mask_local):
                if hide_bad_epochs:
                    keep = ~bad_mask_local
                    if not np.any(keep):
                        x_plot = t_win
                        y_plot = y
                        baseline_plot_out = baseline_plot
                    else:
                        x_plot = t_win[keep]
                        y_plot = y[keep]
                        baseline_plot_out = baseline_plot[keep] if baseline_plot is not None else None
                else:
                    # y already has NaN in bad regions (preserved through normalization)
                    x_plot = t_win
                    y_plot = y
                    baseline_plot_out = baseline_plot
            else:
                x_plot = t_win
                y_plot = y
                baseline_plot_out = baseline_plot

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
                }
            )

        out = {
            "i0": i0,
            "i1": i1,
            "t0": float(t_win[0]) if t_win.size else 0.0,
            "t1": float(t_win[-1]) if t_win.size else 0.0,
            "traces": traces,
            "bad_mask_local": bad_mask_local,
        }
        self._cache[key] = out
        if len(self._cache) > 300:
            self._cache = dict(list(self._cache.items())[-200:])
        return out
