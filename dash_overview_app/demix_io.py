import os
import pickle
import numpy as np

try:
    from skimage.measure import find_contours

    _HAS_SKI = True
except Exception:
    _HAS_SKI = False

from plotly.colors import qualitative as qual

from signal_ops import make_bad_mask, baseline_correct_trace


TRACE_KEYS_ALL = [
    "raw_traces",
    "mc_traces",
    "mc_denoised_traces",
    "weighted_mc_traces",
    "weighted_mc_denoised_traces",
    "weighted_mc_denoised_traces_cleaned",
    "ali_trace",
    "ali_detection_trace",
    "volpy_trace",
    "volpy_sub_trace",
]


def discover_batch_pickles(main_folder):
    if not main_folder or not os.path.isdir(main_folder):
        return []
    found = []
    for root, _, files in os.walk(main_folder):
        base = os.path.basename(root).lower()
        if not base.endswith("good"):
            continue
        if "volpy_demix_results_annotated.pickle" in files:
            path = os.path.join(root, "volpy_demix_results_annotated.pickle")
        elif "volpy_demix_results.pickle" in files:
            path = os.path.join(root, "volpy_demix_results.pickle")
        else:
            continue
        found.append((os.path.basename(root), path))
    found.sort(key=lambda x: x[0])
    return found


def _load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _get_session_length(result_obj, n_cells):
    for k in TRACE_KEYS_ALL:
        arr = result_obj.get(k, None)
        if arr is None:
            continue
        a2 = np.asarray(arr)
        if a2.ndim != 2:
            continue
        if a2.shape[0] == n_cells and a2.shape[1] != n_cells:
            return int(a2.shape[1])
        if a2.shape[1] == n_cells:
            return int(a2.shape[0])
        if a2.shape[0] == n_cells and a2.shape[1] == n_cells:
            return int(a2.shape[1])
    sd = np.asarray(result_obj["shift_distances"]).ravel()
    return int(sd.size) if sd.size > 0 else 0


def _as_2d_tn(arr, n_cells, t_target):
    if arr is None:
        return np.full((t_target, n_cells), np.nan, dtype=np.float32)
    a2 = np.asarray(arr)
    if a2.ndim != 2:
        return np.full((t_target, n_cells), np.nan, dtype=np.float32)
    if a2.shape[1] == n_cells:
        pass
    elif a2.shape[0] == n_cells:
        a2 = a2.T
    else:
        return np.full((t_target, n_cells), np.nan, dtype=np.float32)
    if a2.shape[0] > t_target:
        a2 = a2[:t_target, :]
    elif a2.shape[0] < t_target:
        pad = np.full((t_target - a2.shape[0], n_cells), np.nan, dtype=np.float32)
        a2 = np.concatenate([a2, pad], axis=0)
    return a2.astype(np.float32, copy=False)


def palette_colors(n):
    # High-contrast, saturated palette to avoid low-visibility light tones.
    base = [
        "#E6194B",  # red
        "#3CB44B",  # green
        "#4363D8",  # blue
        "#F58231",  # orange
        "#911EB4",  # purple
        "#F032E6",  # magenta
        "#008080",  # teal
        "#800000",  # maroon
        "#9A6324",  # brown
        "#000075",  # navy
        "#808000",  # olive
        "#46F0F0",  # cyan
        "#DCBEFF",  # lavender (still high contrast on line width=1)
        "#FF1493",  # deep pink
        "#00A8E8",  # vivid sky
        "#2ECC40",  # lime green
        "#FF4136",  # vivid red-orange
        "#7FDBFF",  # bright light blue
        "#B10DC9",  # strong violet
        "#FF851B",  # tangerine
        "#39CCCC",  # turquoise
        "#01FF70",  # neon green
        "#85144B",  # wine
        "#3D9970",  # sea green
    ]
    return (base * ((n // len(base)) + 1))[:n]


def roi_contours(mask):
    if _HAS_SKI:
        return find_contours(mask.astype(float), 0.5)
    r = mask.astype(bool)
    up = np.zeros_like(r)
    up[1:] = r[:-1]
    dn = np.zeros_like(r)
    dn[:-1] = r[1:]
    lf = np.zeros_like(r)
    lf[:, 1:] = r[:, :-1]
    rt = np.zeros_like(r)
    rt[:, :-1] = r[:, 1:]
    edge = r & ~(up & dn & lf & rt)
    ys, xs = np.where(edge)
    return [np.vstack([ys, xs]).T] if len(xs) else []


def roi_centroid(mask):
    ys, xs = np.nonzero(mask)
    return (ys.mean(), xs.mean()) if xs.size > 0 else (0.0, 0.0)


def load_stitched_bundle(paths=None, main_folder=None, frame_rate_override=None):
    paths = [p for p in (paths or []) if p and os.path.exists(p)]
    labels = []

    if not paths and main_folder:
        discovered = discover_batch_pickles(main_folder)
        paths = [p for _, p in discovered]
        labels = [lbl for lbl, _ in discovered]

    if not paths:
        raise ValueError("No valid pickle paths found.")

    results_list = [_load_pickle(p) for p in paths]
    if not labels:
        labels = [os.path.basename(os.path.dirname(p)) or os.path.basename(p) for p in paths]

    required_core = [
        "mean_img_raw",
        "mean_img_mc",
        "weights",
        "ROIs",
        "shift_distances",
        "mc_traces",
    ]
    for si, r in enumerate(results_list):
        for k in required_core:
            if k not in r:
                raise KeyError(f"session {si + 1} ({labels[si]}): missing '{k}'")

    base0 = results_list[0]
    rois = list(base0["ROIs"])
    n_cells = len(rois)
    n_sessions = len(results_list)

    session_lengths = [_get_session_length(r, n_cells) for r in results_list]
    if any(x == 0 for x in session_lengths):
        raise ValueError("Unable to infer session lengths from traces / shift_distances.")

    session_edges = np.concatenate(([0], np.cumsum(session_lengths))).astype(int)
    t_total = int(np.sum(session_lengths))

    frame_rate = float(frame_rate_override) if frame_rate_override else float(base0.get("frame_rate", 500.0))
    time_is_seconds = bool(frame_rate > 0)
    t_axis = (
        (np.arange(t_total, dtype=np.float32) / float(frame_rate))
        if time_is_seconds
        else np.arange(t_total, dtype=np.float32)
    )
    time_label = "Time (s)" if time_is_seconds else "Samples"
    session_boundary_times = []
    for i in range(1, n_sessions):
        if time_is_seconds:
            session_boundary_times.append(float(session_edges[i]) / float(frame_rate))
        else:
            session_boundary_times.append(float(session_edges[i]))

    sources_raw = {}
    for k in TRACE_KEYS_ALL:
        if any(r.get(k, None) is not None for r in results_list):
            per_sess = [
                _as_2d_tn(r.get(k, None), n_cells, t_s)
                for r, t_s in zip(results_list, session_lengths)
            ]
            sources_raw[k] = np.concatenate(per_sess, axis=0).astype(np.float32, copy=False)

    shift_concat = []
    shift_xy_concat = []
    has_shift_xy = False
    for r, pth, t_s in zip(results_list, paths, session_lengths):
        sd = np.asarray(r["shift_distances"], dtype=np.float32).ravel()
        if sd.size > t_s:
            sd = sd[:t_s]
        elif sd.size < t_s:
            sd = np.concatenate([sd, np.full((t_s - sd.size,), np.nan, dtype=np.float32)])
        shift_concat.append(sd)

        reg_xy = None
        if "reg_shifts" in r:
            reg_xy = np.asarray(r["reg_shifts"], dtype=np.float32)
        else:
            mc_params = os.path.join(os.path.dirname(pth), "motion_correction_params.npy")
            if os.path.exists(mc_params):
                try:
                    mc_obj = np.load(mc_params, allow_pickle=True).item()
                    if isinstance(mc_obj, dict) and "reg_shifts" in mc_obj:
                        reg_xy = np.asarray(mc_obj["reg_shifts"], dtype=np.float32)
                except Exception:
                    reg_xy = None

        if reg_xy is not None and reg_xy.ndim == 2 and reg_xy.shape[1] >= 2:
            xy = reg_xy[:, :2]
            if xy.shape[0] > t_s:
                xy = xy[:t_s, :]
            elif xy.shape[0] < t_s:
                pad = np.full((t_s - xy.shape[0], 2), np.nan, dtype=np.float32)
                xy = np.concatenate([xy, pad], axis=0)
            has_shift_xy = True
        else:
            xy = np.full((t_s, 2), np.nan, dtype=np.float32)
        shift_xy_concat.append(xy.astype(np.float32, copy=False))
    shift_distances = np.concatenate(shift_concat).astype(np.float32, copy=False)
    shift_xy = np.concatenate(shift_xy_concat, axis=0).astype(np.float32, copy=False)

    mean_img_raw_list = [np.asarray(r["mean_img_raw"], dtype=np.float32) for r in results_list]
    mean_img_mc_list = [np.asarray(r["mean_img_mc"], dtype=np.float32) for r in results_list]
    mean_img_mc_denoised_list = [
        np.asarray(r.get("mean_img_mc_denoised", r["mean_img_mc"]), dtype=np.float32)
        for r in results_list
    ]

    h, w = mean_img_mc_list[0].shape
    weights_img_list = []
    for r in results_list:
        wt = np.asarray(r["weights"], dtype=np.float32)
        if wt.ndim == 2:
            w_img = wt.max(axis=0).reshape(h, w)
        elif wt.ndim == 3:
            w_img = wt.max(axis=0)
        else:
            raise ValueError("weights shape not supported.")
        weights_img_list.append(np.asarray(w_img, dtype=np.float32))

    saved_good = base0.get("good_cells", None)
    if saved_good is not None and len(saved_good) > 0:
        try:
            initial_good_cells = sorted(set(int(i) for i in saved_good if 0 <= int(i) < n_cells))
        except Exception:
            initial_good_cells = list(range(n_cells))
    else:
        initial_good_cells = list(range(n_cells))

    saved_params = base0.get("spike_detection_params_per_cell", {})

    stitched_spikes = [[] for _ in range(n_cells)]
    has_stitched_spikes = False
    stitched_bad_epochs = []
    offset_samples = 0
    for i_sess, r in enumerate(results_list):
        sess_len = session_lengths[i_sess]
        if "spikes_verified_array" in r:
            has_stitched_spikes = True
            for i in range(n_cells):
                local = r["spikes_verified_array"][i]
                stitched_spikes[i].extend([int(x) + offset_samples for x in local])
        local_be = r.get("bad_epochs", [])
        if local_be:
            time_offset = (offset_samples / frame_rate) if time_is_seconds else offset_samples
            for e0, e1 in local_be:
                stitched_bad_epochs.append((float(e0) + time_offset, float(e1) + time_offset))
        offset_samples += sess_len

    if n_sessions > 1:
        sv = stitched_spikes if has_stitched_spikes else None
    else:
        sv = base0.get("spikes_verified_array", None)
    if sv is None and n_sessions == 1:
        native_spikes = base0.get("spikes", None)
        if (
            isinstance(native_spikes, (list, tuple, np.ndarray))
            and len(native_spikes) == n_cells
        ):
            sv = [
                np.asarray(cell_spikes, dtype=int).reshape(-1).tolist()
                for cell_spikes in native_spikes
            ]
    if sv is None:
        sv_dict = base0.get("spikes_verified", {})
        sv = [[] for _ in range(n_cells)]
        if isinstance(sv_dict, dict):
            for k, v in sv_dict.items():
                try:
                    sv[int(k)] = list(map(int, v))
                except Exception:
                    continue
    if not isinstance(sv, (list, tuple)) or len(sv) != n_cells:
        sv = [[] for _ in range(n_cells)]
    spikes_verified_array = [list(map(int, s)) for s in sv]

    baseline_cfg = base0.get("baseline_correction_params", None)
    if not isinstance(baseline_cfg, dict):
        baseline_cfg = {"enabled": False}

    default_image_labels = {
        "raw": "Mean Image - Raw",
        "mc": "Mean Image - MC",
        "mc_denoised": "MC Denoised",
        "weights": "Max Weight Map",
    }
    saved_image_labels = base0.get("image_labels", {})
    if isinstance(saved_image_labels, dict):
        image_labels = {
            key: str(saved_image_labels.get(key, value))
            for key, value in default_image_labels.items()
        }
    else:
        image_labels = default_image_labels

    return {
        "results_list": results_list,
        "paths_list": paths,
        "labels_list": labels,
        "n_sessions": n_sessions,
        "session_lengths": session_lengths,
        "session_edges_frames": session_edges,
        "session_boundary_times": session_boundary_times,
        "frame_rate": frame_rate,
        "time_is_seconds": time_is_seconds,
        "time_label": time_label,
        "t": t_axis,
        "T": t_total,
        "N": n_cells,
        "ROIs": rois,
        "colors": palette_colors(n_cells),
        "trace_keys_all": list(TRACE_KEYS_ALL),
        "sources_raw": sources_raw,
        "shift_distances": shift_distances,
        "shift_xy": shift_xy if has_shift_xy else None,
        "mean_img_raw_list": mean_img_raw_list,
        "mean_img_mc_list": mean_img_mc_list,
        "mean_img_mc_denoised_list": mean_img_mc_denoised_list,
        "weights_img_list": weights_img_list,
        "initial_bad_epochs": list(stitched_bad_epochs),
        "initial_good_cells": list(initial_good_cells),
        "initial_spikes_verified_array": spikes_verified_array,
        "initial_params_per_cell": saved_params,
        "initial_baseline_cfg": baseline_cfg,
        "preferred_trace_source": base0.get(
            "preferred_trace_source",
            None,
        ),
        "initial_show_spikes": bool(
            base0.get("show_spikes_by_default", False)
        ),
        "image_labels": image_labels,
    }


def save_annotations_compatible(
    bundle,
    filename,
    bad_epochs,
    good_cells,
    spikes_verified_array,
    params_per_cell,
    baseline_cfg=None,
):
    if not filename:
        raise ValueError("Filename is required.")

    frame_rate = float(bundle["frame_rate"])
    time_is_seconds = bool(bundle["time_is_seconds"])
    n_cells = int(bundle["N"])
    session_edges = bundle["session_edges_frames"]
    paths_list = bundle["paths_list"]
    results_list = bundle["results_list"]
    sources_raw = bundle["sources_raw"]
    trace_keys_all = bundle["trace_keys_all"]

    baseline_cfg = baseline_cfg or {"enabled": False}
    baseline_enabled = bool(baseline_cfg.get("enabled", False))
    baseline_percentile = float(baseline_cfg.get("percentile", 50.0))
    baseline_window_s = float(baseline_cfg.get("window_s", 60.0))
    baseline_smooth = str(baseline_cfg.get("smooth", "median"))

    saved_paths = []

    def _local_epochs(s_idx):
        s0 = int(session_edges[s_idx])
        s1 = int(session_edges[s_idx + 1])
        t0g, t1g = (
            (float(s0) / frame_rate, float(s1) / frame_rate)
            if time_is_seconds
            else (float(s0), float(s1))
        )
        loc = []
        for b0, b1 in (bad_epochs or []):
            lo = max(min(float(b0), float(b1)), t0g)
            hi = min(max(float(b0), float(b1)), t1g)
            if hi > lo:
                loc.append((lo - t0g, hi - t0g))
        return loc

    for s in range(len(results_list)):
        if s >= len(paths_list) or not paths_list[s]:
            continue
        base = dict(results_list[s])
        s0 = int(session_edges[s])
        s1 = int(session_edges[s + 1])
        ep = _local_epochs(s)

        local_t = (
            np.arange(s1 - s0, dtype=np.float32) / frame_rate
            if time_is_seconds
            else np.arange(s1 - s0, dtype=np.float32)
        )
        mask_l = make_bad_mask(local_t, ep)

        for k in trace_keys_all:
            if k not in sources_raw:
                continue

            dat_processed = sources_raw[k][s0:s1, :].astype(np.float32, copy=True)

            if baseline_enabled:
                for ci in range(dat_processed.shape[1]):
                    dat_processed[:, ci], _ = baseline_correct_trace(
                        dat_processed[:, ci],
                        frame_rate,
                        percentile=baseline_percentile,
                        window_s=baseline_window_s,
                        smooth=baseline_smooth,
                        bad_mask=mask_l,
                    )

            if np.any(mask_l):
                dat_processed[mask_l, :] = np.nan

            orig_val = base.get(k)
            if hasattr(orig_val, "shape") and len(orig_val.shape) == 2:
                if orig_val.shape[1] == n_cells and orig_val.shape[0] != n_cells:
                    base[k] = dat_processed.T
                else:
                    base[k] = dat_processed
            else:
                base[k] = dat_processed

        spk_local = []
        for i in range(n_cells):
            g = np.asarray(spikes_verified_array[i], dtype=int)
            valid = g[(g >= s0) & (g < s1)]
            spk_local.append((valid - s0).tolist())

        base["spikes_verified_array"] = spk_local
        base["bad_epochs"] = ep
        base["good_cells"] = sorted(set(int(i) for i in good_cells if 0 <= int(i) < n_cells))
        base["spike_detection_params_per_cell"] = params_per_cell

        if baseline_enabled:
            base["baseline_correction_params"] = {
                "enabled": True,
                "percentile": baseline_percentile,
                "window_s": baseline_window_s,
                "smooth": baseline_smooth,
            }

        out_path = os.path.join(os.path.dirname(paths_list[s]), filename)
        with open(out_path, "wb") as f:
            pickle.dump(base, f)
        saved_paths.append(out_path)

    return saved_paths
