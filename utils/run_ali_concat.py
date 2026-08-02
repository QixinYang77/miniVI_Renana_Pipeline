"""Run all-pixel motion projection and full-session ALI on cluster output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
import tifffile


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from utils.ali_dashboard_utils import write_ali_dashboard_bundle
from utils.ali_motion_projection import write_motion_projected_tiff
from utils.ali_pipeline_utils import (
    load_ali_results,
    mean_image_from_window,
    plot_ali_results,
    run_pyali,
    save_ali_results,
    write_ali_median_highpass_movie,
)


UPSTREAM_PYALI_COMMIT = "a348549f1601686c9c4e3bd5fbb7e832c2958502"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the selected positive-going ALI parameters on every frame "
            "of one concatenated, motion-corrected movie."
        )
    )
    parser.add_argument(
        "output_folder",
        help="Folder containing movReg.tif and motion_correction_params.npy",
    )
    parser.add_argument("--source-name", default="movReg.tif")
    parser.add_argument(
        "--projection-name",
        default="movReg_motion_projected.tif",
    )
    parser.add_argument(
        "--projection-summary-name",
        default="movReg_motion_projection_summary.npz",
    )
    parser.add_argument(
        "--roi-tif",
        required=True,
        help=(
            "Automatic SLM mask TIFF written by DS_motion_correction.py; "
            "these masks constrain the ALI search but are not cell ROIs"
        ),
    )
    parser.add_argument("--frame-rate", type=float, default=500.0)
    parser.add_argument(
        "--pyali-root",
        default=str(REPOSITORY_ROOT / "third_party" / "pyALI"),
    )
    parser.add_argument(
        "--results-subdir",
        default=(
            "ali_full_coarse4p0_sep8_assign5_minspikes4_"
            "radius8p5_smooth0p75"
        ),
    )
    parser.add_argument("--motion-trend-seconds", type=float, default=4.0)
    parser.add_argument("--motion-jitter-seconds", type=float, default=0.010)
    parser.add_argument("--motion-epoch-seconds", type=float, default=10.0)
    parser.add_argument("--motion-row-block", type=int, default=4)
    parser.add_argument("--median-window-ms", type=float, default=50.0)
    parser.add_argument("--search-dilation-px", type=int, default=8)
    parser.add_argument("--coarse-threshold-sd", type=float, default=4.0)
    parser.add_argument("--minimum-segment-voxels", type=int, default=5)
    parser.add_argument("--spatial-sigma-px", type=float, default=2.5)
    parser.add_argument("--spatial-radius-px", type=int, default=5)
    parser.add_argument("--ali-map-sigma", type=float, default=1.0)
    parser.add_argument("--ali-map-radius", type=int, default=3)
    parser.add_argument("--ali-peak-threshold", type=float, default=0.30)
    parser.add_argument(
        "--center-separation-px",
        type=float,
        default=8.0,
        help="Minimum separation in camera pixels",
    )
    parser.add_argument("--assignment-radius-px", type=float, default=5.0)
    parser.add_argument("--minimum-cluster-spikes", type=int, default=4)
    parser.add_argument("--footprint-radius-px", type=float, default=8.5)
    parser.add_argument(
        "--footprint-smoothing-sigma-px",
        type=float,
        default=0.75,
    )
    parser.add_argument("--coarse-chunk-frames", type=int, default=1_000)
    parser.add_argument("--trace-chunk-frames", type=int, default=2_000)
    parser.add_argument(
        "--maximum-coarse-events",
        type=int,
        default=0,
        help="Safety ceiling; zero means unlimited and is recommended for full movies",
    )
    parser.add_argument(
        "--overwrite-motion-projection",
        action="store_true",
    )
    parser.add_argument("--overwrite-ali", action="store_true")
    return parser.parse_args()


def common_valid_fov_mask_from_shifts(
    shifts: np.ndarray,
    *,
    image_shape: tuple[int, int],
    neighbour_radius: int = 1,
) -> tuple[np.ndarray, dict[str, object]]:
    """Return pixels observed in every rigidly registered frame."""

    shifts = np.asarray(shifts, dtype=np.float64)
    if shifts.ndim != 2 or shifts.shape[1] != 2:
        raise ValueError("registration shifts must have shape (frames, 2)")
    if not np.all(np.isfinite(shifts)):
        raise ValueError("registration shifts contain non-finite values")
    vertical = shifts[:, 0]
    horizontal = shifts[:, 1]
    top_filled = int(np.ceil(max(0.0, float(vertical.max()))))
    bottom_filled = int(np.ceil(max(0.0, float(-vertical.min()))))
    left_filled = int(np.ceil(max(0.0, float(horizontal.max()))))
    right_filled = int(np.ceil(max(0.0, float(-horizontal.min()))))
    top = top_filled + (neighbour_radius if top_filled else 0)
    bottom = bottom_filled + (neighbour_radius if bottom_filled else 0)
    left = left_filled + (neighbour_radius if left_filled else 0)
    right = right_filled + (neighbour_radius if right_filled else 0)
    height, width = image_shape
    if top + bottom >= height or left + right >= width:
        raise ValueError("Registration shifts remove the complete image")
    mask = np.zeros((height, width), dtype=bool)
    mask[top : height - bottom, left : width - right] = True
    bounds = {
        "top_excluded_px": top,
        "bottom_excluded_px": bottom,
        "left_excluded_px": left,
        "right_excluded_px": right,
        "neighbour_radius_px": neighbour_radius,
        "valid_shape": [height - top - bottom, width - left - right],
    }
    return mask, bounds


def _load_registration_shifts(path: Path, n_frames: int) -> np.ndarray:
    parameters = np.load(path, allow_pickle=True).item()
    if "reg_shifts" not in parameters:
        raise KeyError(f"Expected reg_shifts in {path}")
    shifts = np.asarray(parameters["reg_shifts"], dtype=np.float32)
    if shifts.ndim != 2 or shifts.shape[1] < 2:
        raise ValueError(f"Unexpected registration-shift shape: {shifts.shape}")
    shifts = shifts[:, :2]
    if shifts.shape[0] != n_frames:
        raise ValueError(
            f"Registration shifts contain {shifts.shape[0]:,} frames, "
            f"but the movie contains {n_frames:,}"
        )
    return shifts


def _load_slm_masks(path: Path, image_shape: tuple[int, int]) -> np.ndarray:
    masks = np.asarray(tifffile.imread(path), dtype=bool)
    if masks.ndim == 2:
        masks = masks[None, ...]
    if masks.ndim != 3 or masks.shape[1:] != image_shape:
        raise ValueError(
            f"SLM masks have shape {masks.shape}, expected (masks, {image_shape})"
        )
    if np.any(masks.reshape(len(masks), -1).sum(axis=1) == 0):
        raise ValueError("At least one SLM mask is empty")
    return masks


def _load_mean(path: Path, expected_shape: tuple[int, int]) -> np.ndarray:
    image = np.asarray(tifffile.imread(path), dtype=np.float32)
    if image.shape != expected_shape:
        raise ValueError(
            f"Mean image {path} has shape {image.shape}, expected {expected_shape}"
        )
    return image


def main() -> None:
    args = _parse_args()
    output_folder = Path(args.output_folder).resolve()
    source_path = output_folder / args.source_name
    shifts_path = output_folder / "motion_correction_params.npy"
    roi_path = Path(args.roi_tif)
    if not roi_path.is_absolute():
        roi_path = output_folder / roi_path
    pyali_root = Path(args.pyali_root).resolve()
    for required in (source_path, shifts_path, roi_path, pyali_root):
        if not required.exists():
            raise FileNotFoundError(required)

    source = tifffile.memmap(source_path, mode="r")
    if source.ndim != 3:
        raise ValueError("Motion-corrected movie must have shape (time, y, x)")
    n_frames, height, width = source.shape
    image_shape = (height, width)
    frame_rate_hz = float(args.frame_rate)
    if frame_rate_hz <= 0:
        raise ValueError("--frame-rate must be positive")
    shifts = _load_registration_shifts(shifts_path, n_frames)
    slm_masks = _load_slm_masks(roi_path, image_shape)
    valid_fov_mask, valid_fov_bounds = common_valid_fov_mask_from_shifts(
        shifts,
        image_shape=image_shape,
        neighbour_radius=1,
    )
    search_mask = ndimage.binary_dilation(
        np.any(slm_masks, axis=0),
        iterations=int(args.search_dilation_px),
    ) & valid_fov_mask
    if not np.any(search_mask):
        raise ValueError("The valid, dilated SLM search mask is empty")

    results_root = output_folder / args.results_subdir
    results_root.mkdir(parents=True, exist_ok=True)
    projection_path = output_folder / args.projection_name
    projection_summary_path = output_folder / args.projection_summary_name
    projection_path, projection_summary = write_motion_projected_tiff(
        source_path,
        shifts,
        frame_rate_hz=frame_rate_hz,
        trend_smoothing_seconds=float(args.motion_trend_seconds),
        jitter_smoothing_seconds=float(args.motion_jitter_seconds),
        epoch_seconds=float(args.motion_epoch_seconds),
        output_path=projection_path,
        summary_path=projection_summary_path,
        preserve_epoch_mean=True,
        spatial_row_block=int(args.motion_row_block),
        overwrite=bool(args.overwrite_motion_projection),
    )

    median_window_frames = int(
        round(float(args.median_window_ms) * frame_rate_hz / 1_000.0)
    )
    median_window_frames = max(3, median_window_frames)
    if median_window_frames % 2 == 0:
        median_window_frames += 1
    highpass_path = results_root / (
        f"motion_projected_median_hp_{median_window_frames}f.tif"
    )
    highpass_metadata_path = highpass_path.with_suffix(".json")
    highpass_path, highpass_cache_hit = write_ali_median_highpass_movie(
        projection_path,
        highpass_path,
        highpass_metadata_path,
        start_frame=0,
        n_frames=n_frames,
        window_frames=median_window_frames,
        chunk_frames=int(args.coarse_chunk_frames),
        overwrite=bool(args.overwrite_ali),
    )

    result_directory = results_root / "result"
    numeric_path = result_directory / "pyali_results.npz"
    if numeric_path.exists() and not args.overwrite_ali:
        ali_result = load_ali_results(numeric_path)
        result_source = "compatible saved result"
        ali_noise_sd = None
    else:
        maximum_coarse_events = (
            None
            if int(args.maximum_coarse_events) <= 0
            else int(args.maximum_coarse_events)
        )
        ali_upsampling_factor = 4
        peak_min_distance = int(
            round(float(args.center_separation_px) * ali_upsampling_factor)
        )
        ali_result, ali_noise_sd = run_pyali(
            highpass_path,
            pyali_root,
            valid_fov_mask=valid_fov_mask,
            search_mask=search_mask,
            polarity=1,
            spatial_sigma_px=float(args.spatial_sigma_px),
            spatial_radius_px=int(args.spatial_radius_px),
            coarse_threshold_sd=float(args.coarse_threshold_sd),
            minimum_segment_voxels=int(args.minimum_segment_voxels),
            coarse_chunk_frames=int(args.coarse_chunk_frames),
            coarse_cache_path=results_root / "coarse_events.npz",
            overwrite_coarse_cache=bool(args.overwrite_ali),
            maximum_coarse_events=maximum_coarse_events,
            n_svd_components=25,
            svd_random_seed=7,
            fine_n_pixels=77,
            fine_radius_px=9.9,
            ali_upsampling_factor=ali_upsampling_factor,
            ali_smoothing_sigma=float(args.ali_map_sigma),
            ali_smoothing_radius=int(args.ali_map_radius),
            ali_peak_threshold=float(args.ali_peak_threshold),
            ali_peak_min_distance=peak_min_distance,
            cluster_assignment_radius_px=float(args.assignment_radius_px),
            minimum_cluster_spikes=int(args.minimum_cluster_spikes),
            cluster_boundary_distance_px=2.0,
            footprint_radius_px=float(args.footprint_radius_px),
            footprint_smoothing_sigma_px=float(
                args.footprint_smoothing_sigma_px
            ),
            trace_movie_path=projection_path,
            trace_start_frame=0,
            trace_chunk_frames=int(args.trace_chunk_frames),
        )
        parameters = {
            "workflow": "full concatenated cluster ALI",
            "n_frames": int(n_frames),
            "duration_seconds": float(n_frames / frame_rate_hz),
            "frame_rate_hz": frame_rate_hz,
            "polarity": "positive-going",
            "median_window_ms_requested": float(args.median_window_ms),
            "median_window_frames_actual": median_window_frames,
            "coarse_threshold_sd": float(args.coarse_threshold_sd),
            "ali_smoothing_sigma": float(args.ali_map_sigma),
            "ali_peak_threshold": float(args.ali_peak_threshold),
            "center_separation_px": float(args.center_separation_px),
            "cluster_assignment_radius_px": float(args.assignment_radius_px),
            "minimum_cluster_spikes": int(args.minimum_cluster_spikes),
            "footprint_radius_px": float(args.footprint_radius_px),
            "footprint_smoothing_sigma_px": float(
                args.footprint_smoothing_sigma_px
            ),
            "maximum_coarse_events": maximum_coarse_events,
            "search_dilation_px": int(args.search_dilation_px),
            "valid_fov_bounds": valid_fov_bounds,
            "manual_cell_rois_used": False,
            "slm_masks_are_search_guide_only": True,
            "saved_traces_are_full_band": True,
        }
        numeric_path, _ = save_ali_results(
            ali_result,
            result_directory,
            source_movie_path=projection_path,
            highpass_movie_path=highpass_path,
            valid_fov_mask=valid_fov_mask,
            search_mask=search_mask,
            pyali_commit=UPSTREAM_PYALI_COMMIT,
            parameters=parameters,
        )
        result_source = "fresh ALI run"

    projected_mean = mean_image_from_window(
        projection_path,
        start_frame=0,
        n_frames=n_frames,
        chunk_frames=int(args.trace_chunk_frames),
    )
    raw_mean = _load_mean(output_folder / "mean_image_raw.tif", image_shape)
    mc_mean = _load_mean(output_folder / "mean_image_mc.tif", image_shape)
    dashboard_path = write_ali_dashboard_bundle(
        result_directory / "ali_dashboard_bundle.pickle",
        ali_result=ali_result,
        mean_image_raw=raw_mean,
        mean_image_motion_corrected=mc_mean,
        mean_image_motion_projected=projected_mean,
        registration_shifts=shifts,
        frame_rate_hz=frame_rate_hz,
        source_movie_path=projection_path,
        ali_numeric_result_path=numeric_path,
        support_fraction=0.20,
    )

    qc_figure, _ = plot_ali_results(
        projected_mean,
        ali_result,
        valid_fov_mask=valid_fov_mask,
        search_mask=search_mask,
        frame_rate_hz=frame_rate_hz,
        trace_seconds=min(60.0, n_frames / frame_rate_hz),
        trace_source_label=(
            "Full-band motion-projected ALI traces; center separation 8 px, "
            "footprint radius 8.5 px, sigma 0.75 px"
        ),
        trace_high_pass_hz=1.0,
        trace_high_pass_order=3,
        ali_map_display_gamma=0.5,
        ali_map_display_upper_percentile=99.5,
    )
    qc_path = result_directory / "ali_qc_first60s.png"
    qc_figure.savefig(qc_path, dpi=200, bbox_inches="tight")
    plt.close(qc_figure)

    run_summary = {
        "source_movie": str(source_path),
        "motion_projected_movie": str(projection_path),
        "motion_projection_cache_hit": bool(projection_summary.cache_hit),
        "highpass_movie": str(highpass_path),
        "highpass_cache_hit": bool(highpass_cache_hit),
        "result_source": result_source,
        "numeric_result": str(numeric_path),
        "dashboard_bundle": str(dashboard_path),
        "qc_figure": str(qc_path),
        "n_frames": int(n_frames),
        "duration_seconds": float(n_frames / frame_rate_hz),
        "n_localized_events": int(len(ali_result.coarse_spikes)),
        "n_components": int(len(ali_result.footprints)),
        "n_assigned_events": int(
            sum(len(values) for values in ali_result.assigned_spike_frames)
        ),
        "manual_cell_rois_used": False,
    }
    summary_path = result_directory / "ali_run_summary.json"
    summary_path.write_text(json.dumps(run_summary, indent=2) + "\n")
    print(json.dumps(run_summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
