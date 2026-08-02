import argparse
import logging
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tifffile

import caiman as cm
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.volpy import utils as volpy_utils
from caiman.source_extraction.volpy.volparams import volparams
from caiman.summary_images import local_correlations_movie_offline

_repo_root = str(Path(__file__).resolve().parents[1])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from utils.minivi_io import downsample_video_local_mean, load_experiment_metadata, load_raw, load_ROI


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step1: motion correction (and optional ROI detection).")
    parser.add_argument("data_file", help="Path to input .raw file")
    parser.add_argument(
        "--factor",
        nargs=2,
        type=int,
        default=(1, 1),
        metavar=("H", "W"),
        help="Spatial downsample factor (H W). If not (1,1), downsamples movie and ROIs.",
    )
    parser.add_argument("--frames-to-truncate", type=int, default=500)
    parser.add_argument(
        "--overwrite-output-folder",
        dest="overwrite_output_folder",
        action="store_true",
        default=True,
        help="Overwrite outputs in the output folder before running (default: True).",
    )
    parser.add_argument(
        "--no-overwrite-output-folder",
        dest="overwrite_output_folder",
        action="store_false",
        help="Do not delete existing outputs in the output folder.",
    )
    parser.add_argument(
        "--disable-initial-dark-frame-fix",
        action="store_true",
        help="Disable fixing abnormally-dark initial frames by replacing them with the first good frame.",
    )
    parser.add_argument(
        "--initial-dark-max-seconds",
        type=float,
        default=1.0,
        help="Inspect at most this many seconds from the start for dark-frame fixing (default: 1.0).",
    )
    parser.add_argument(
        "--initial-dark-rel-threshold",
        type=float,
        default=0.1,
        help="Frames with mean intensity < threshold * reference are considered dark (default: 0.5).",
    )

    parser.add_argument("--output-subdir", default="output_final", help="Subfolder under data folder for outputs.")
    parser.add_argument(
        "--run-tag",
        default=None,
        help="Optional run identifier. If set, output goes to <data_folder>/<output_subdir>/<run_tag>/ (otherwise <data_folder>/<output_subdir>/).",
    )

    parser.add_argument(
        "--preserve-sub1hz",
        action="store_true",
        help="Disable drift/motion regressions that can attenuate slow (<1 Hz) signals.",
    )
    parser.add_argument(
        "--disable-additive-regression",
        action="store_true",
        help="Skip additive pattern-drift regression (alpha_t * P subtraction).",
    )
    parser.add_argument(
        "--alpha-smoothing",
        default="median",
        choices=("median", "exp_decay", "none"),
        help="How to model alpha_t for additive regression. 'exp_decay' fits a single exponential decay to alphas_raw.",
    )
    parser.add_argument(
        "--alpha-median-ms",
        type=float,
        default=5000,
        help="Median-filter window length for alpha_t when --alpha-smoothing=median.",
    )
    parser.add_argument(
        "--alpha-exp-min-r2",
        type=float,
        default=0.0,
        help="Minimum R^2 for accepting exp_decay fit; otherwise fallback behavior is used.",
    )
    parser.add_argument(
        "--alpha-exp-fallback",
        default="disabled",
        choices=("disabled", "median"),
        help="Fallback if exp_decay fit fails/poor: 'disabled' uses no additive correction; 'median' uses median-filtered alpha_t.",
    )
    parser.add_argument(
        "--enable-motion-regression",
        action="store_true",
        help="Enable pixelwise motion-regression denoising (default: disabled).",
    )
    parser.add_argument(
        "--disable-motion-regression",
        action="store_true",
        help="Skip pixelwise motion-regression denoising.",
    )
    parser.add_argument(
        "--motion-reg-chunk-size",
        type=int,
        default=30000,
        help="Chunk size (frames) for motion-regression denoising fits (default: 30000). Larger preserves slower signals.",
    )
    parser.add_argument(
        "--no-alpha-plot",
        action="store_true",
        help="Disable saving an alpha(t) diagnostic plot (alphas_raw vs alphas).",
    )
    parser.add_argument(
        "--keep-baseline-for-correlation",
        action="store_true",
        help="Do not remove baseline in local_correlations_movie_offline (ROI detection only).",
    )
    parser.add_argument(
        "--skip-roi-detection",
        action="store_true",
        help="Skip Mask R-CNN ROI detection (motion correction outputs only).",
    )
    parser.add_argument(
        "--no-previews",
        action="store_true",
        help="Disable saving side-by-side comparison preview tif(s).",
    )
    parser.add_argument(
        "--preview-seconds",
        type=float,
        default=120.0,
        help="Duration in seconds for side-by-side preview clip(s) (default: 120.0).",
    )
    parser.add_argument(
        "--preview-fps",
        type=float,
        default=50.0,
        help="Target frame rate (Hz) for preview clip(s) via temporal subsampling (default: 50.0).",
    )
    parser.add_argument(
        "--preview-stride",
        type=int,
        default=0,
        help="Temporal stride for preview clip(s). If 0, uses --preview-fps to choose a stride (default: 0).",
    )
    parser.add_argument(
        "--apply-shifts-to-raw",
        action="store_true",
        help="Also apply the computed motion shifts to the (unpreprocessed) movie tif saved as <filename>_movie.tif.",
    )
    parser.add_argument(
        "--keep-caiman-mmaps",
        action="store_true",
        help="Keep CaImAn-generated .mmap files (default: auto-delete after outputs are saved).",
    )
    parser.add_argument(
        "--niter-rig",
        type=int,
        default=1,
        help=(
            "Number of rigid NoRMCorre template-refinement iterations "
            "(default: 1; the full-session ALI notebook requests 2)."
        ),
    )
    parser.add_argument("--n-processes", type=int, default=None, help="Processes for CaImAn cluster (default: auto).")
    return parser.parse_args()


def to_float32(Y):
    Y = np.asarray(Y)
    if Y.dtype != np.float32:
        Y = Y.astype(np.float32, copy=False)
    return Y


def estimate_static_pattern(Y, method="median", percentile=30, sigma_px=3.0):
    from scipy.ndimage import gaussian_filter

    if method == "median":
        P = np.median(Y, axis=0)
    elif method == "percentile":
        P = np.percentile(Y, percentile, axis=0)
    else:
        raise ValueError("method must be 'median' or 'percentile'")

    if sigma_px and sigma_px > 0:
        P = gaussian_filter(P, sigma=sigma_px)
    return P.astype(np.float32)


def _as_path_list(x) -> List[str]:
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return [os.fspath(p) for p in x if p]
    return [os.fspath(x)]


def _cleanup_files(paths: List[str], *, label: str = "file") -> None:
    for p in paths:
        if not p:
            continue
        try:
            if os.path.exists(p):
                os.remove(p)
                print(f"[INFO] Deleted {label}: {os.path.abspath(p)}", flush=True)
        except Exception as e:
            print(f"[WARN] Failed to delete {label} {p!r}: {e!r}", flush=True)


def regress_pattern_drift(
    Y,
    P,
    smooth_alpha=None,
    fs_hz=500,
    median_ms=500,
    *,
    return_raw=False,
    return_fit_info=False,
    exp_min_r2=0.0,
    exp_fallback="disabled",
):
    from scipy.signal import medfilt

    T, H, W = Y.shape
    P_vec = P.reshape(-1).astype(np.float32)
    Y2D = Y.reshape(T, -1)

    denom = np.dot(P_vec, P_vec) + 1e-12
    alphas_raw = (Y2D @ P_vec) / denom

    fit_info = None

    if smooth_alpha in (None, "none"):
        alphas = alphas_raw.astype(np.float32)
    elif smooth_alpha == "median":
        if fs_hz is None:
            raise ValueError("fs_hz must be provided when smooth_alpha='median'")
        k = int(round((median_ms / 1000.0) * fs_hz))
        if k < 3:
            k = 3
        if k % 2 == 0:
            k += 1
        alphas = medfilt(alphas_raw, kernel_size=k).astype(np.float32)
    elif smooth_alpha == "exp_decay":
        if fs_hz is None:
            raise ValueError("fs_hz must be provided when smooth_alpha='exp_decay'")

        from scipy.optimize import least_squares

        if exp_fallback not in ("disabled", "median"):
            raise ValueError("exp_fallback must be 'disabled' or 'median'")

        y = alphas_raw.astype(np.float64)
        if not np.all(np.isfinite(y)):
            alphas = alphas_raw.astype(np.float32)
        else:
            t = (np.arange(T, dtype=np.float64) / float(fs_hz)).astype(np.float64)
            duration_s = float(t[-1] - t[0]) if T > 1 else 0.0

            p5, p95 = np.percentile(y, [5, 95])
            data_range = float(p95 - p5)
            if not np.isfinite(data_range) or data_range <= 0:
                alphas = alphas_raw.astype(np.float32)
            else:
                tail_n = max(1, int(round(0.1 * T)))
                c0 = float(np.median(y[-tail_n:]))
                a0 = float(y[0] - c0)
                tau_min = max(0.1, 3.0 / float(fs_hz))
                tau_max = max(tau_min * 10.0, max(1.0, duration_s) * 10.0)
                tau0 = min(max(1.0, duration_s / 3.0), tau_max)

                a_bound = 5.0 * data_range
                c_low = float(np.min(y) - 5.0 * data_range)
                c_high = float(np.max(y) + 5.0 * data_range)

                def _resid(p):
                    a, tau, c = p
                    return (a * np.exp(-t / tau) + c) - y

                try:
                    res = least_squares(
                        _resid,
                        x0=np.array([a0, tau0, c0], dtype=np.float64),
                        bounds=(
                            np.array([-a_bound, tau_min, c_low], dtype=np.float64),
                            np.array([a_bound, tau_max, c_high], dtype=np.float64),
                        ),
                        loss="soft_l1",
                        f_scale=max(1e-12, 0.25 * data_range),
                        max_nfev=2000,
                    )
                    a_hat, tau_hat, c_hat = (float(v) for v in res.x)
                    y_fit = (a_hat * np.exp(-t / tau_hat) + c_hat).astype(np.float32)

                    resid = (y.astype(np.float32) - y_fit).astype(np.float32)
                    var_y = float(np.var(y))
                    r2 = 1.0 - (float(np.var(resid)) / var_y) if var_y > 0 else 0.0

                    fit_info = {
                        "model": "exp_decay",
                        "success": bool(res.success),
                        "cost": float(res.cost),
                        "r2": float(r2),
                        "a": a_hat,
                        "tau_s": tau_hat,
                        "c": c_hat,
                        "tau_min_s": float(tau_min),
                        "tau_max_s": float(tau_max),
                    }

                    if (not res.success) or (not np.isfinite(r2)) or (r2 < float(exp_min_r2)):
                        if fit_info is not None:
                            fit_info["fallback"] = exp_fallback

                        if exp_fallback == "median":
                            k = int(round((median_ms / 1000.0) * fs_hz))
                            if k < 3:
                                k = 3
                            if k % 2 == 0:
                                k += 1
                            alphas = medfilt(alphas_raw, kernel_size=k).astype(np.float32)
                        else:
                            alphas = np.zeros_like(alphas_raw, dtype=np.float32)
                    else:
                        alphas = y_fit
                except Exception as e:
                    if exp_fallback == "median":
                        k = int(round((median_ms / 1000.0) * fs_hz))
                        if k < 3:
                            k = 3
                        if k % 2 == 0:
                            k += 1
                        alphas = medfilt(alphas_raw, kernel_size=k).astype(np.float32)
                    else:
                        alphas = np.zeros_like(alphas_raw, dtype=np.float32)
                    fit_info = {
                        "model": "exp_decay",
                        "success": False,
                        "fallback": exp_fallback,
                        "error": repr(e),
                    }
    else:
        raise ValueError("smooth_alpha must be one of: None, 'none', 'median', 'exp_decay'")

    Y_addcorr = (Y2D - alphas[:, None] * P_vec[None, :]).reshape(T, H, W).astype(np.float32)

    if return_raw and return_fit_info:
        return Y_addcorr, alphas, alphas_raw.astype(np.float32), fit_info
    if return_raw:
        return Y_addcorr, alphas, alphas_raw.astype(np.float32)
    if return_fit_info:
        return Y_addcorr, alphas, fit_info
    return Y_addcorr, alphas


def flat_field_correct(Y, P, dark=None, eps_rel=1e-3):
    Y = to_float32(Y)
    P = to_float32(P)
    if dark is None:
        D = np.percentile(Y, 1, axis=0).astype(np.float32)
    else:
        D = to_float32(dark)

    eps = np.float32(np.percentile(P, 1) * eps_rel)
    scale = np.float32(np.median(P))

    Y_corr = (Y - D[None, :, :]) / (P[None, :, :] + eps)
    Y_corr *= scale
    return Y_corr.astype(np.float32)


def preprocess_voltage_movie(
    movie_3d,
    *,
    flatfield_method="median",
    flatfield_percentile=30,
    blur_sigma_px=0,
    do_additive_regression=True,
    fs_hz=500.0,
    alpha_smoothing="median",
    alpha_median_ms=5000,
    exp_min_r2=0.0,
    exp_fallback="disabled",
    store_raw_alphas=True,
):
    Y = to_float32(movie_3d)

    P = estimate_static_pattern(
        Y,
        method=flatfield_method if flatfield_method in ("median", "percentile") else "median",
        percentile=flatfield_percentile,
        sigma_px=blur_sigma_px,
    )

    alphas_raw = None
    alpha_fit_info = None
    if do_additive_regression:
        if store_raw_alphas:
            Y_addcorr, alphas, alphas_raw, alpha_fit_info = regress_pattern_drift(
                Y,
                P,
                smooth_alpha=alpha_smoothing,
                fs_hz=fs_hz,
                median_ms=alpha_median_ms,
                return_raw=True,
                return_fit_info=True,
                exp_min_r2=exp_min_r2,
                exp_fallback=exp_fallback,
            )
        else:
            Y_addcorr, alphas, alpha_fit_info = regress_pattern_drift(
                Y,
                P,
                smooth_alpha=alpha_smoothing,
                fs_hz=fs_hz,
                median_ms=alpha_median_ms,
                return_raw=False,
                return_fit_info=True,
                exp_min_r2=exp_min_r2,
                exp_fallback=exp_fallback,
            )
    else:
        Y_addcorr, alphas = Y, None

    Y_ff = flat_field_correct(Y_addcorr, P, dark=None, eps_rel=1e-3)

    return {
        "P_static": P,
        "alphas": alphas,
        "alphas_raw": alphas_raw,
        "alpha_fit_info": alpha_fit_info,
        "Y_addcorr": Y_addcorr,
        "Y_flatfield": Y_ff,
    }


def main() -> None:
    args = _parse_args()
    factor = tuple(args.factor)
    import gc

    t0 = time.perf_counter()
    t_last = t0

    def _step(msg: str) -> None:
        nonlocal t_last
        now = time.perf_counter()
        dt = now - t_last
        total = now - t0
        print(f"[STEP] {msg} (+{dt:.2f}s, total {total:.2f}s)", flush=True)
        t_last = now

    _step("Start")

    data_folder = os.path.dirname(args.data_file)
    filename = os.path.splitext(os.path.basename(args.data_file))[0]

    if os.path.isabs(str(args.output_subdir)):
        raise ValueError("--output-subdir must be a relative folder name, got: {!r}".format(args.output_subdir))
    if str(args.output_subdir).strip() in ("", ".", ".."):
        raise ValueError(
            "--output-subdir must not be empty / '.' / '..' (refusing to write into the raw data folder)"
        )

    base_output_folder = os.path.join(data_folder, args.output_subdir)
    output_folder = os.path.join(base_output_folder, args.run_tag) if args.run_tag else base_output_folder
    os.makedirs(output_folder, exist_ok=True)

    if bool(args.overwrite_output_folder):
        data_folder_real = os.path.realpath(data_folder)
        output_folder_real = os.path.realpath(output_folder)
        if output_folder_real == data_folder_real:
            raise RuntimeError(
                "Refusing to clear output folder because it resolves to the raw data folder: {!r}".format(
                    output_folder_real
                )
            )
        if not output_folder_real.startswith(data_folder_real + os.sep):
            raise RuntimeError(
                "Refusing to clear output folder outside the raw data folder.\n"
                "data_folder={!r}\noutput_folder={!r}".format(data_folder_real, output_folder_real)
            )

        _step("Clear output folder")
        # Preserve manual ROI annotations if you keep them in the output folder.
        keep = {"manual_ROIs.npz"}
        for name in os.listdir(output_folder):
            if name in keep:
                continue
            path = os.path.join(output_folder, name)
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
            except FileNotFoundError:
                pass
        print(f"[INFO] Cleared output folder (kept: {sorted(keep)}): {output_folder}", flush=True)

    print("Data folder:", data_folder, flush=True)
    print("Filename:", filename, flush=True)
    print("Output folder:", output_folder, flush=True)
    print("Downsample factor:", factor, flush=True)

    save_previews = not bool(args.no_previews)
    if args.preview_stride < 0:
        raise ValueError("--preview-stride must be >= 0")

    raw_movie_tif_path = os.path.join(output_folder, f"{filename}_movie.tif")
    raw_rois_tif_path = os.path.join(output_folder, f"{filename}_ROIs.tif")
    ds_movie_tif_path = os.path.join(output_folder, f"{filename}_movie_ds{factor[0]}x{factor[1]}.tif")
    ds_rois_tif_path = os.path.join(output_folder, f"{filename}_ROIs_ds{factor[0]}x{factor[1]}.tif")

    _step("Read Experiment.xml metadata")
    frame_rate_hz, orig_width, orig_height = load_experiment_metadata(args.data_file)

    preview_frames = int(round(float(args.preview_seconds) * float(frame_rate_hz)))
    if preview_frames < 1:
        preview_frames = 1

    preview_stride = int(args.preview_stride)
    if preview_stride == 0:
        if args.preview_fps is None or float(args.preview_fps) <= 0:
            preview_stride = 1
        else:
            preview_stride = max(1, int(round(float(frame_rate_hz) / float(args.preview_fps))))

    def _percentiles_for_viewing(paths: List[str]) -> Tuple[float, float]:
        samples = []
        for path in paths:
            if not path or (not os.path.exists(path)):
                continue
            try:
                with tifffile.TiffFile(path) as tif:
                    n = len(tif.pages)
                    if n <= 0:
                        continue
                    last = min(n - 1, preview_frames - 1)
                    idxs = np.linspace(0, last, num=min(16, last + 1), dtype=int)
                    for i in idxs:
                        if i % preview_stride != 0:
                            continue
                        frame = tif.pages[int(i)].asarray()
                        frame = np.asarray(frame, dtype=np.float32)
                        step = max(1, int(max(frame.shape) // 256))
                        samples.append(frame[::step, ::step])
            except Exception:
                continue

        if not samples:
            return 0.0, 1.0
        s = np.stack(samples, axis=0)
        lo, hi = (float(v) for v in np.percentile(s, [1, 99]))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return 0.0, 1.0
        return lo, hi

    def _to_uint16_for_viewing(arr: np.ndarray, *, lo: float, hi: float) -> np.ndarray:
        x = np.asarray(arr)
        x = x.astype(np.float32, copy=False)
        if x.size == 0:
            return x.astype(np.uint16)

        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return np.zeros_like(x, dtype=np.uint16)

        y = (x - lo) / (hi - lo)
        y = np.clip(y, 0.0, 1.0)
        return (y * 65535.0).astype(np.uint16)

    def _save_side_by_side_preview(columns: List[Tuple[str, str]], out_path: str) -> None:
        """
        Save a side-by-side preview tif with columns concatenated along width.

        Args:
            columns: list of (label, tif_path). Missing files are skipped.
            out_path: output tif path.
        """
        if not save_previews:
            return

        if os.path.exists(out_path):
            try:
                if os.path.getsize(out_path) < 1024:
                    os.remove(out_path)
                else:
                    return
            except OSError:
                return

        existing = [(label, p) for (label, p) in columns if p and os.path.exists(p)]
        if len(existing) < 2:
            return

        lo, hi = _percentiles_for_viewing([p for (_label, p) in existing])
        sources = [(label, p) for (label, p) in existing]

        try:
            tiffs = [tifffile.TiffFile(p) for (_, p) in sources]
        except Exception as e:
            print(f"[WARN] Failed opening preview sources: {e!r}")
            return

        tmp_out = out_path + ".tmp"
        memmap_path = out_path + ".memmap"
        mm = None
        try:
            lengths = [len(t.pages) for t in tiffs]
            if not lengths or min(lengths) < 1:
                return

            n_frames = min(min(lengths), preview_frames)

            first_frames = [t.pages[0].asarray() for t in tiffs]
            h_min = min(f.shape[0] for f in first_frames)
            w_min = min(f.shape[1] for f in first_frames)

            if os.path.exists(tmp_out):
                try:
                    os.remove(tmp_out)
                except OSError:
                    pass
            if os.path.exists(memmap_path):
                try:
                    os.remove(memmap_path)
                except OSError:
                    pass

            # Build the preview as an on-disk memmap (so we can still write via tifffile.imsave
            # without holding the full preview stack in RAM).
            out_count = (n_frames + preview_stride - 1) // preview_stride
            n_cols = len(tiffs)
            mm = np.memmap(
                memmap_path,
                dtype=np.uint16,
                mode="w+",
                shape=(int(out_count), int(h_min), int(w_min * n_cols)),
            )

            out_idx = 0
            for i in range(0, n_frames, preview_stride):
                col_start = 0
                for (_label, _p), tif in zip(sources, tiffs):
                    arr = tif.pages[int(i)].asarray()
                    arr = arr[:h_min, :w_min]
                    mm[out_idx, :, col_start : col_start + w_min] = _to_uint16_for_viewing(arr, lo=lo, hi=hi)
                    col_start += w_min
                out_idx += 1

            mm.flush()
            tifffile.imsave(tmp_out, mm, bigtiff=True)
            del mm
            mm = None
            try:
                os.remove(memmap_path)
            except OSError:
                pass

            os.replace(tmp_out, out_path)

            labels = [s[0] for s in sources]
            print(
                f"[INFO] Saved side-by-side preview: {out_path} "
                f"(cols: {', '.join(labels)}; seconds={args.preview_seconds}; fps~{frame_rate_hz/preview_stride:.2f}; lo={lo:.3g} hi={hi:.3g})"
            )
        except Exception as e:
            print(f"[WARN] Failed to save side-by-side preview {out_path}: {e!r}")
            try:
                for p in (tmp_out, memmap_path):
                    if p and os.path.exists(p):
                        os.remove(p)
            except Exception:
                pass
        finally:
            if mm is not None:
                try:
                    del mm
                except Exception:
                    pass
            for t in tiffs:
                try:
                    t.close()
                except Exception:
                    pass

    def _atomic_tiff_save(path: str, arr: np.ndarray) -> None:
        tmp = path + ".tmp"
        tifffile.imsave(tmp, arr, bigtiff=True)
        os.replace(tmp, path)

    def _fix_initial_dark_frames_inplace(movie: np.ndarray) -> dict:
        """
        Detect consecutive abnormally-dark frames at the beginning and replace them
        with the first non-dark frame (in-place).

        Returns a dict with info about the fix.
        """
        if bool(args.disable_initial_dark_frame_fix):
            return {"enabled": False, "n_fixed": 0}
        if movie.ndim != 3:
            return {"enabled": True, "n_fixed": 0, "reason": "movie not 3D"}

        T = int(movie.shape[0])
        if T < 2:
            return {"enabled": True, "n_fixed": 0, "reason": "too few frames"}

        max_check = int(round(float(args.initial_dark_max_seconds) * float(frame_rate_hz)))
        max_check = max(1, min(T, max_check))

        rel_thr = float(args.initial_dark_rel_threshold)
        if not np.isfinite(rel_thr) or rel_thr <= 0:
            rel_thr = 0.5

        # Per-frame mean for initial segment (vectorized).
        m0 = movie[:max_check].reshape(max_check, -1).mean(axis=1).astype(np.float64)

        # Reference intensity from *all* frames after the initial window.
        # (User-facing rationale: compare first window vs mean of the rest of the movie.)
        start = int(max_check)
        if start >= T:
            return {"enabled": True, "n_fixed": 0, "reason": "no later frames"}

        m_rest = movie[start:].reshape(T - start, -1).mean(axis=1).astype(np.float64)
        ref_mean = float(np.mean(m_rest))
        ref_std = float(np.std(m_rest))

        if (not np.isfinite(ref_mean)) or ref_mean <= 0:
            return {"enabled": True, "n_fixed": 0, "reason": "bad reference mean", "ref_mean": ref_mean}

        # Threshold: initial frames must be substantially below the mean of the rest.
        thr = float(rel_thr * ref_mean)

        n_bad = 0
        for v in m0:
            if np.isfinite(v) and (v < thr):
                n_bad += 1
            else:
                break

        if n_bad <= 0 or n_bad >= T:
            return {
                "enabled": True,
                "n_fixed": 0,
                "ref_mean": ref_mean,
                "ref_std": ref_std,
                "thr": thr,
                "ref_window_start": start,
            }

        movie[:n_bad] = movie[n_bad]
        return {
            "enabled": True,
            "n_fixed": int(n_bad),
            "first_good": int(n_bad),
            "ref_mean": ref_mean,
            "ref_std": ref_std,
            "thr": thr,
            "ref_window_start": start,
        }

    raw_movie_3d = None
    if os.path.exists(raw_movie_tif_path):
        _step("Load existing raw movie tif")
        raw_movie_3d = tifffile.imread(raw_movie_tif_path)
        if raw_movie_3d.ndim != 3 or raw_movie_3d.shape[1] != orig_height or raw_movie_3d.shape[2] != orig_width:
            print(
                f"[WARN] Existing {raw_movie_tif_path} has shape {getattr(raw_movie_3d, 'shape', None)}; "
                f"expected (*, {orig_height}, {orig_width}). Regenerating from raw.",
                flush=True,
            )
            raw_movie_3d = None

    if raw_movie_3d is None:
        _step("Load raw movie from .raw")
        raw_movie_3d, _fr, _w, _h = load_raw(args.data_file, frames_to_truncate=args.frames_to_truncate)
        if _fr is not None:
            frame_rate_hz = _fr

    _step("Fix initial dark frames (if needed)")
    dark_fix_info = _fix_initial_dark_frames_inplace(raw_movie_3d)
    if dark_fix_info.get("enabled") and dark_fix_info.get("n_fixed", 0) > 0:
        print(
            f"[INFO] Fixed {dark_fix_info['n_fixed']} initial dark frames by copying frame "
            f"{dark_fix_info['first_good']} (thr={dark_fix_info.get('thr'):.3g})"
        )

    if os.path.exists(raw_rois_tif_path):
        _step("Load existing raw ROIs tif")
        ROIs_raw = tifffile.imread(raw_rois_tif_path).astype(bool)
        if ROIs_raw.ndim != 3 or ROIs_raw.shape[1] != orig_height or ROIs_raw.shape[2] != orig_width:
            print(
                f"[WARN] Existing {raw_rois_tif_path} has shape {getattr(ROIs_raw, 'shape', None)}; "
                f"expected (N, {orig_height}, {orig_width}). Regenerating from ROIs.xaml.",
                flush=True,
            )
            ROIs_raw = None
    else:
        ROIs_raw = None

    if ROIs_raw is None:
        _step("Load ROIs from ROIs.xaml")
        ROIs_raw = load_ROI(data_folder, orig_width, orig_height)

    # Persist raw (non-downsampled) movie/ROIs. These always reflect original HxW.
    if (not os.path.exists(raw_movie_tif_path)) or (dark_fix_info.get("n_fixed", 0) > 0):
        _step("Save raw movie tif")
        if not os.path.exists(raw_movie_tif_path):
            tifffile.imsave(raw_movie_tif_path, raw_movie_3d, bigtiff=True)
        else:
            _atomic_tiff_save(raw_movie_tif_path, raw_movie_3d)

    if not os.path.exists(raw_rois_tif_path):
        _step("Save raw ROIs tif")
        tifffile.imsave(raw_rois_tif_path, ROIs_raw.astype(np.uint8), bigtiff=True)

    # Choose working movie/ROIs for downstream steps.
    if factor == (1, 1):
        movie_3d = raw_movie_3d
        ROIs = ROIs_raw
    else:
        _step("Prepare downsampled working movie/ROIs")
        if os.path.exists(ds_movie_tif_path):
            movie_3d = tifffile.imread(ds_movie_tif_path)
        else:
            movie_3d = downsample_video_local_mean(raw_movie_3d, factor)
            tifffile.imsave(ds_movie_tif_path, movie_3d, bigtiff=True)

        if os.path.exists(ds_rois_tif_path):
            ROIs = tifffile.imread(ds_rois_tif_path).astype(bool)
        else:
            ROIs = downsample_video_local_mean(ROIs_raw.astype(np.uint8), factor).astype(bool)
            tifffile.imsave(ds_rois_tif_path, ROIs.astype(np.uint8), bigtiff=True)

    height = movie_3d.shape[1]
    width = movie_3d.shape[2]

    _step("Save movie/ROIs/mean image")
    metadata = {
        "frameRate": frame_rate_hz,
        "width": width,
        "height": height,
        "downsample_factor": factor,
        "original_width": orig_width,
        "original_height": orig_height,
        "frames_truncated": args.frames_to_truncate,
        "run_tag": args.run_tag,
        "output_folder": output_folder,
        "overwrite_output_folder": bool(args.overwrite_output_folder),
        "preserve_sub1hz": bool(args.preserve_sub1hz),
        "alpha_smoothing": str(args.alpha_smoothing),
        "alpha_median_ms": float(args.alpha_median_ms),
        "alpha_exp_min_r2": float(args.alpha_exp_min_r2),
        "alpha_exp_fallback": str(args.alpha_exp_fallback),
        "apply_shifts_to_raw": bool(args.apply_shifts_to_raw),
        "initial_dark_frame_fix": dark_fix_info,
        "motion_reg_chunk_size": int(args.motion_reg_chunk_size),
        "motion_regression_enabled": bool(args.enable_motion_regression)
        and not (args.preserve_sub1hz or args.disable_motion_regression),
        "normcorre_niter_rig": int(args.niter_rig),
    }

    del ROIs
    del ROIs_raw
    mean_image_raw_path = os.path.join(output_folder, "mean_image_raw.tif")
    if not os.path.exists(mean_image_raw_path):
        mean_image_raw = np.mean(movie_3d, axis=0)
        tifffile.imsave(mean_image_raw_path, mean_image_raw, bigtiff=True)
        del mean_image_raw

    # Preprocess for motion correction (optional)
    _step("Preprocess for motion correction")
    do_additive = not (args.preserve_sub1hz or args.disable_additive_regression)
    out = preprocess_voltage_movie(
        movie_3d,
        flatfield_method="median",
        blur_sigma_px=0,
        do_additive_regression=do_additive,
        fs_hz=float(frame_rate_hz),
        alpha_smoothing=str(args.alpha_smoothing),
        alpha_median_ms=float(args.alpha_median_ms),
        exp_min_r2=float(args.alpha_exp_min_r2),
        exp_fallback=str(args.alpha_exp_fallback),
        store_raw_alphas=True,
    )

    if (not args.no_alpha_plot) and do_additive and (out.get("alphas_raw") is not None) and (out.get("alphas") is not None):
        _step("Save alpha(t) diagnostic plot")
        try:
            alphas_raw = np.asarray(out["alphas_raw"], dtype=np.float64)
            alphas = np.asarray(out["alphas"], dtype=np.float64)
            t_s = np.arange(alphas.shape[0], dtype=np.float64) / float(frame_rate_hz)

            max_points = 20000
            step = max(1, int(np.ceil(alphas.shape[0] / max_points)))
            idx = slice(None, None, step)

            fig = plt.figure(figsize=(12, 6))
            ax = fig.add_subplot(2, 1, 1)
            ax.plot(t_s[idx], alphas_raw[idx], lw=0.8, alpha=0.6, label="alphas_raw")
            ax.plot(t_s[idx], alphas[idx], lw=1.0, label="alphas (used)")

            info = out.get("alpha_fit_info") or {}
            title = "alpha(t) fit"
            if isinstance(info, dict) and info:
                model = info.get("model")
                r2 = info.get("r2")
                fallback = info.get("fallback")
                parts = [str(model)] if model else []
                if r2 is not None:
                    parts.append(f"R2={float(r2):.3f}")
                if fallback:
                    parts.append(f"fallback={fallback}")
                if parts:
                    title += " [" + ", ".join(parts) + "]"
            ax.set_title(title)
            ax.set_xlabel("time (s)")
            ax.set_ylabel("alpha")
            ax.legend(loc="best", frameon=False)

            ax2 = fig.add_subplot(2, 1, 2)
            # show residual (what is removed by smoothing/fitting)
            resid = alphas_raw - alphas
            ax2.plot(t_s[idx], resid[idx], lw=0.8, alpha=0.8, color="tab:red", label="alphas_raw - alphas")
            ax2.set_xlabel("time (s)")
            ax2.set_ylabel("residual")
            ax2.legend(loc="best", frameon=False)

            fig.tight_layout()
            fig_path = os.path.join(output_folder, f"{filename}_alpha_fit.png")
            fig.savefig(fig_path, dpi=150)
            plt.close(fig)
            print(f"[INFO] Saved alpha plot: {fig_path}", flush=True)
        except Exception as e:
            print(f"[WARN] Failed to save alpha plot: {e!r}", flush=True)

    _step("Save preprocessed tif")
    preprocessed_fname = os.path.join(output_folder, f"{filename}_preprocessed.tif")
    tifffile.imsave(preprocessed_fname, out["Y_flatfield"], bigtiff=True)
    del out
    del movie_3d
    gc.collect()

    # Motion correction
    _step("Set up CaImAn motion correction")
    fnames = [preprocessed_fname]
    fr = int(np.round(frame_rate_hz))

    opts_dict = {
        "fnames": fnames,
        "fr": fr,
        "pw_rigid": False,
        "max_shifts": (10, 10),
        "gSig_filt": (3, 3),
        "strides": (48, 48),
        "overlaps": (24, 24),
        "max_deviation_rigid": 3,
        "border_nan": "copy",
        "niter_rig": int(args.niter_rig),
    }

    opts = volparams(params_dict=opts_dict)
    if "dview" in locals():
        cm.stop_server(dview=dview)
    c, dview, n_processes = cm.cluster.setup_cluster(
        backend="local", n_processes=args.n_processes, single_thread=False
    )

    mc = MotionCorrect(fnames, dview=dview, **opts.get_group("motion"))
    _step("Run motion correction")
    mc.motion_correct(save_movie=True)
    caiman_mmaps_to_cleanup: List[str] = []
    caiman_mmaps_to_cleanup.extend(_as_path_list(getattr(mc, "mmap_file", None)))

    _step("Save motion correction outputs")
    m_rig = cm.load(mc.mmap_file)
    tifffile.imsave(os.path.join(output_folder, "movReg.tif"), m_rig, bigtiff=True)
    np.save(os.path.join(output_folder, "motion_correction_params.npy"), {"reg_shifts": mc.shifts_rig})
    mean_image_mc = cm.summary_images.mean_image(mc.mmap_file[0], window=1000, dview=dview)
    tifffile.imsave(os.path.join(output_folder, "mean_image_mc.tif"), mean_image_mc, bigtiff=True)

    if args.apply_shifts_to_raw:
        _step("Apply shifts to raw movie")
        raw_mmap = mc.apply_shifts_movie([raw_movie_tif_path], save_memmap=True)
        caiman_mmaps_to_cleanup.extend(_as_path_list(raw_mmap))
        m_raw_rig = cm.load(raw_mmap)
        tifffile.imsave(os.path.join(output_folder, f"{filename}_movie_mc.tif"), m_raw_rig, bigtiff=True)
        del m_raw_rig
        del raw_mmap
        gc.collect()

    # Optional motion-regression denoising
    do_motion_regression_denoising = bool(args.enable_motion_regression) and not (
        args.preserve_sub1hz or args.disable_motion_regression
    )
    if do_motion_regression_denoising:
        _step("Run motion-regression denoising")
        from sklearn.linear_model import LinearRegression

        if args.motion_reg_chunk_size < 1:
            raise ValueError("--motion-reg-chunk-size must be >= 1")

        motion_shift = np.array(mc.shifts_rig)
        motion_shiftx = motion_shift[:, 0] - np.mean(motion_shift[:, 0])
        motion_shifty = motion_shift[:, 1] - np.mean(motion_shift[:, 1])
        regressors = np.column_stack(
            [
                motion_shiftx,
                motion_shifty,
                motion_shiftx**2,
                motion_shifty**2,
                motion_shiftx * motion_shifty,
            ]
        )
        n_frames, h, w = m_rig.shape
        m_rig_denoised = np.empty_like(m_rig)
        chunk_size = int(args.motion_reg_chunk_size)
        for i in range(h):
            for j in range(w):
                y = m_rig[:, i, j]
                y_denoised = np.zeros_like(y)
                for start in range(0, n_frames, chunk_size):
                    end = min(start + chunk_size, n_frames)
                    reg_chunk = regressors[start:end, :]
                    y_chunk = y[start:end]
                    model = LinearRegression().fit(reg_chunk, y_chunk)
                    # Preserve baseline within each chunk: only subtract the motion-correlated component,
                    # not the fitted intercept. This avoids chunk-boundary "steps" in the output.
                    y_pred = model.predict(reg_chunk)
                    y_motion = y_pred - model.intercept_
                    y_denoised[start:end] = y_chunk - y_motion
                m_rig_denoised[:, i, j] = y_denoised
        _step("Save movReg_denoised.tif")
        tifffile.imsave(os.path.join(output_folder, "movReg_denoised.tif"), m_rig_denoised, bigtiff=True)
        del m_rig_denoised
        del regressors
        del motion_shift
        del motion_shiftx
        del motion_shifty
        gc.collect()
    else:
        print("Skipping motion-regression denoising (preserving slow signals).")

    if not args.skip_roi_detection:
        _step("ROI detection (Mask R-CNN)")
        gaussian_blur = False
        Cn = np.nanmax(
            local_correlations_movie_offline(
                mc.mmap_file[0],
                fr=fr,
                window=fr * 4,
                stride=fr * 4,
                winSize_baseline=fr,
                remove_baseline=not args.keep_baseline_for_correlation,
                gaussian_blur=gaussian_blur,
                dview=dview,
            ),
            axis=0,
        )
        del Cn

        img = mean_image_mc
        img = (img - np.mean(img)) / np.std(img)
        summary_images = np.stack([img, img, img], axis=0).astype(np.float32)

        weights_path = cm.utils.utils.download_model("mask_rcnn")
        ROIs_mrcnn = volpy_utils.mrcnn_inference(
            img=summary_images.transpose([1, 2, 0]),
            size_range=[10, 22],
            weights_path=weights_path,
            display_result=False,
        )

        tifffile.imsave(os.path.join(output_folder, "ROIs_MRCNN.tif"), ROIs_mrcnn.astype(np.uint8), bigtiff=True)
        plt.figure()
        plt.imshow(img, cmap="gray")
        plt.contour(ROIs_mrcnn.sum(axis=0), colors="r")
        plt.title("Detected ROIs")
        plt.savefig(os.path.join(output_folder, "Detected_ROIs.png"))
        plt.close()
        del ROIs_mrcnn
        del summary_images
        gc.collect()

    _step("Write run metadata")
    metadata_path = os.path.join(output_folder, "run_metadata.npy")
    np.save(metadata_path, metadata)

    # Side-by-side comparison previews for quick debugging
    _step("Write side-by-side preview(s)")
    _save_side_by_side_preview(
        [
            ("preprocessed", preprocessed_fname),
            ("movReg", os.path.join(output_folder, "movReg.tif")),
        ],
        os.path.join(output_folder, f"{filename}_preprocessed_vs_movReg_preview.tif"),
    )

    raw_mc_path = os.path.join(output_folder, f"{filename}_movie_mc.tif")
    _save_side_by_side_preview(
        [("raw", raw_movie_tif_path), ("raw_mc", raw_mc_path)],
        os.path.join(output_folder, f"{filename}_raw_vs_raw_mc_preview.tif"),
    )

    del m_rig
    del mean_image_mc
    gc.collect()
    if "dview" in locals():
        cm.stop_server(dview=dview)

    if not bool(args.keep_caiman_mmaps):
        _step("Clean up CaImAn mmap files")
        _cleanup_files(caiman_mmaps_to_cleanup, label="CaImAn mmap")

    _step("Done")
    print("Job finished.")


if __name__ == "__main__":
    main()
