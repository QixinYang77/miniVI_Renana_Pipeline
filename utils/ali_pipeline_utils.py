"""Out-of-core adapter for the upstream pyALI Activity Localization Imaging code.

The GPL-3.0 pyALI implementation remains isolated under ``third_party/pyALI``.
This module provides dataset I/O, chunked coarse-event detection, boundary
masking, result serialization, and QC plotting around the upstream scientific
functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
import numpy as np
from scipy import ndimage, signal
from scipy.sparse.linalg import eigs
from skimage.feature import peak_local_max
from skimage.measure import find_contours
import tifffile


@dataclass
class ALIResult:
    """Activity Localization Imaging outputs in full-camera coordinates."""

    coarse_spikes: np.ndarray
    coarse_peak_values: np.ndarray
    fine_spike_locations: np.ndarray
    ali_map: np.ndarray
    cluster_centers: np.ndarray
    cluster_indices: np.ndarray
    footprints: np.ndarray
    detection_least_squares_traces: np.ndarray
    detection_traces: np.ndarray
    least_squares_traces: np.ndarray
    traces: np.ndarray
    assigned_spike_frames: list[np.ndarray]
    component_metrics: list[dict[str, float | int]]


def _source_signature(path: Path) -> dict[str, object]:
    stat = Path(path).stat()
    return {
        "path": str(Path(path).resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _array_digest(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def _validate_mask(
    mask: np.ndarray,
    image_shape: tuple[int, int],
    *,
    name: str,
) -> np.ndarray:
    values = np.asarray(mask, dtype=bool)
    if values.shape != image_shape:
        raise ValueError(
            f"{name} has shape {values.shape}, expected {image_shape}"
        )
    return values.copy()


def _load_pyali_functions(pyali_root: Path) -> dict[str, object]:
    functions_directory = Path(pyali_root) / "python" / "functions"
    if not functions_directory.exists():
        raise FileNotFoundError(
            f"pyALI functions were not found: {functions_directory}"
        )
    if str(functions_directory) not in sys.path:
        sys.path.insert(0, str(functions_directory))
    names = [
        "ali_assign_cluster",
        "ali_denoising",
        "ali_density_map",
        "ali_fp_support",
        "ali_spk_fine",
    ]
    return {
        name: getattr(importlib.import_module(name), name)
        for name in names
    }


def write_ali_median_highpass_movie(
    source_path: Path,
    output_path: Path,
    metadata_path: Path,
    *,
    start_frame: int,
    n_frames: int,
    window_frames: int,
    chunk_frames: int = 1_000,
    overwrite: bool = False,
) -> tuple[Path, bool]:
    """Write a pixelwise short-median residual as a float32 BigTIFF."""

    source_path = Path(source_path)
    output_path = Path(output_path)
    metadata_path = Path(metadata_path)
    source = tifffile.memmap(source_path, mode="r")
    if source.ndim != 3:
        raise ValueError("Source TIFF must have shape (time, y, x)")
    stop_frame = start_frame + n_frames
    if start_frame < 0 or stop_frame > source.shape[0]:
        raise ValueError("Requested ALI window lies outside the source movie")
    if window_frames < 3 or window_frames % 2 == 0:
        raise ValueError("window_frames must be an odd integer of at least 3")
    signature = {
        "version": 1,
        "algorithm": "pyali_pixelwise_median_highpass",
        "source": _source_signature(source_path),
        "start_frame": int(start_frame),
        "n_frames": int(n_frames),
        "window_frames": int(window_frames),
    }
    expected_shape = (n_frames, *source.shape[1:])
    if output_path.exists() and metadata_path.exists() and not overwrite:
        metadata = json.loads(metadata_path.read_text())
        existing = tifffile.memmap(output_path, mode="r")
        if (
            metadata == signature
            and existing.shape == expected_shape
            and existing.dtype == np.float32
        ):
            return output_path, True
        raise FileExistsError(
            "Existing ALI high-pass cache is incompatible; set overwrite=True"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    target = tifffile.memmap(
        output_path,
        shape=expected_shape,
        dtype=np.float32,
        bigtiff=True,
    )
    radius = window_frames // 2
    for output_start in range(0, n_frames, chunk_frames):
        output_stop = min(output_start + chunk_frames, n_frames)
        read_start = max(0, output_start - radius)
        read_stop = min(n_frames, output_stop + radius)
        block = np.asarray(
            source[
                start_frame + read_start : start_frame + read_stop
            ],
            dtype=np.float32,
        )
        baseline = ndimage.median_filter(
            block,
            size=(window_frames, 1, 1),
            mode="nearest",
        )
        core_start = output_start - read_start
        core_stop = core_start + (output_stop - output_start)
        target[output_start:output_stop] = (
            block[core_start:core_stop] - baseline[core_start:core_stop]
        )
        target.flush()
        print(
            f"ALI temporal median high-pass "
            f"{output_stop:>6,} / {n_frames:,} frames"
        )
    del target
    metadata_path.write_text(json.dumps(signature, indent=2))
    return output_path, False


def spatial_mean_downsample(
    values: np.ndarray,
    *,
    factor: int = 2,
) -> np.ndarray:
    """Average non-overlapping spatial blocks over the last two axes."""

    array = np.asarray(values)
    if array.ndim < 2:
        raise ValueError("values must have at least two dimensions")
    if factor < 1:
        raise ValueError("factor must be at least one")
    height, width = array.shape[-2:]
    if height % factor or width % factor:
        raise ValueError(
            f"Spatial shape {(height, width)} is not divisible by {factor}"
        )
    output_height = height // factor
    output_width = width // factor
    reshaped = array.reshape(
        *array.shape[:-2],
        output_height,
        factor,
        output_width,
        factor,
    )
    return reshaped.mean(axis=(-3, -1), dtype=np.float64).astype(
        np.float32
    )


def spatial_max_downsample(
    values: np.ndarray,
    *,
    factor: int = 2,
) -> np.ndarray:
    """Take the maximum of non-overlapping spatial blocks."""

    array = np.asarray(values)
    if array.ndim < 2:
        raise ValueError("values must have at least two dimensions")
    if factor < 1:
        raise ValueError("factor must be at least one")
    height, width = array.shape[-2:]
    if height % factor or width % factor:
        raise ValueError(
            f"Spatial shape {(height, width)} is not divisible by {factor}"
        )
    reshaped = array.reshape(
        *array.shape[:-2],
        height // factor,
        factor,
        width // factor,
        factor,
    )
    return reshaped.max(axis=(-3, -1)).astype(np.float32)


def spatial_downsample_binary_mask(
    mask: np.ndarray,
    *,
    factor: int = 2,
    reduction: str = "any",
) -> np.ndarray:
    """Downsample a binary mask with conservative ``any`` or ``all`` blocks."""

    values = np.asarray(mask, dtype=bool)
    if values.ndim != 2:
        raise ValueError("mask must be two-dimensional")
    if factor < 1:
        raise ValueError("factor must be at least one")
    height, width = values.shape
    if height % factor or width % factor:
        raise ValueError(
            f"Mask shape {(height, width)} is not divisible by {factor}"
        )
    blocks = values.reshape(
        height // factor,
        factor,
        width // factor,
        factor,
    )
    if reduction == "any":
        return blocks.any(axis=(1, 3))
    if reduction == "all":
        return blocks.all(axis=(1, 3))
    raise ValueError("reduction must be 'any' or 'all'")


def write_spatially_binned_movie(
    source_path: Path,
    output_path: Path,
    metadata_path: Path,
    *,
    start_frame: int,
    n_frames: int,
    factor: int = 2,
    reduction: str = "mean",
    chunk_frames: int = 1_000,
    overwrite: bool = False,
) -> tuple[Path, bool]:
    """Write a spatial block-mean or block-maximum movie out of core."""

    source_path = Path(source_path)
    output_path = Path(output_path)
    metadata_path = Path(metadata_path)
    source = tifffile.memmap(source_path, mode="r")
    if source.ndim != 3:
        raise ValueError("Source TIFF must have shape (time, y, x)")
    stop_frame = start_frame + n_frames
    if start_frame < 0 or n_frames < 1 or stop_frame > source.shape[0]:
        raise ValueError("Requested binning window lies outside the movie")
    if factor < 1:
        raise ValueError("factor must be at least one")
    if reduction not in {"mean", "max"}:
        raise ValueError("reduction must be 'mean' or 'max'")
    height, width = source.shape[1:]
    if height % factor or width % factor:
        raise ValueError(
            f"Source shape {(height, width)} is not divisible by {factor}"
        )
    expected_shape = (
        n_frames,
        height // factor,
        width // factor,
    )
    signature = {
        "version": 1,
        "algorithm": f"nonoverlapping_spatial_block_{reduction}",
        "source": _source_signature(source_path),
        "start_frame": int(start_frame),
        "n_frames": int(n_frames),
        "factor": int(factor),
        "input_shape": [int(height), int(width)],
        "output_shape": list(expected_shape[1:]),
        "output_dtype": "float32",
    }
    if output_path.exists() and metadata_path.exists() and not overwrite:
        metadata = json.loads(metadata_path.read_text())
        existing = tifffile.memmap(output_path, mode="r")
        if (
            metadata == signature
            and existing.shape == expected_shape
            and existing.dtype == np.float32
        ):
            return output_path, True
        raise FileExistsError(
            "Existing spatial-binning cache is incompatible; "
            "set overwrite=True"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    target = tifffile.memmap(
        output_path,
        shape=expected_shape,
        dtype=np.float32,
        bigtiff=True,
    )
    for output_start in range(0, n_frames, chunk_frames):
        output_stop = min(output_start + chunk_frames, n_frames)
        block = np.asarray(
            source[
                start_frame + output_start : start_frame + output_stop
            ],
            dtype=np.float32,
        )
        if reduction == "mean":
            pooled = spatial_mean_downsample(block, factor=factor)
        else:
            pooled = spatial_max_downsample(block, factor=factor)
        target[output_start:output_stop] = pooled
        target.flush()
        print(
            f"Spatial {factor}x{factor} {reduction} pooling "
            f"{output_stop:>6,} / {n_frames:,} frames"
        )
    del target
    metadata_path.write_text(json.dumps(signature, indent=2))
    return output_path, False


def mean_image_from_window(
    movie_path: Path,
    *,
    start_frame: int,
    n_frames: int,
    chunk_frames: int = 1_000,
) -> np.ndarray:
    """Calculate a float64-accumulated mean without loading the movie."""

    movie = tifffile.memmap(movie_path, mode="r")
    stop_frame = min(start_frame + n_frames, movie.shape[0])
    if start_frame < 0 or stop_frame <= start_frame:
        raise ValueError("Invalid mean-image window")
    accumulation = np.zeros(movie.shape[1:], dtype=np.float64)
    count = 0
    for start in range(start_frame, stop_frame, chunk_frames):
        stop = min(start + chunk_frames, stop_frame)
        chunk = np.asarray(movie[start:stop], dtype=np.float32)
        accumulation += chunk.sum(axis=0, dtype=np.float64)
        count += stop - start
    return (accumulation / count).astype(np.float32)


def temporal_standard_deviation_image(
    movie_path: Path,
    *,
    start_frame: int = 0,
    n_frames: int | None = None,
    chunk_frames: int = 1_000,
) -> np.ndarray:
    """Calculate each pixel's temporal SD with float64 chunked moments."""

    movie = tifffile.memmap(movie_path, mode="r")
    if movie.ndim != 3:
        raise ValueError("Movie TIFF must have shape (time, y, x)")
    if n_frames is None:
        n_frames = movie.shape[0] - start_frame
    stop_frame = start_frame + n_frames
    if start_frame < 0 or n_frames < 1 or stop_frame > movie.shape[0]:
        raise ValueError("Requested SD window lies outside the movie")
    running_sum = np.zeros(movie.shape[1:], dtype=np.float64)
    running_square = np.zeros(movie.shape[1:], dtype=np.float64)
    count = 0
    for start in range(start_frame, stop_frame, chunk_frames):
        stop = min(start + chunk_frames, stop_frame)
        chunk = np.asarray(movie[start:stop], dtype=np.float32)
        running_sum += chunk.sum(axis=0, dtype=np.float64)
        running_square += np.square(
            chunk,
            dtype=np.float64,
        ).sum(axis=0)
        count += stop - start
    mean = running_sum / count
    variance = np.maximum(running_square / count - mean**2, 0)
    return np.sqrt(variance).astype(np.float32)


def _spatial_filter_frames(
    frames: np.ndarray,
    *,
    sigma_px: float,
    radius_px: int,
) -> np.ndarray:
    return ndimage.gaussian_filter(
        frames,
        sigma=(0, sigma_px, sigma_px),
        radius=(0, radius_px, radius_px),
        mode="reflect",
    ).astype(np.float32)


def _deterministic_ali_denoising(
    frames_y_x_time: np.ndarray,
    n_components: int,
    *,
    random_seed: int,
) -> np.ndarray:
    """Reproduce upstream ALI SVD with an explicit ARPACK start vector."""

    frames = np.asarray(frames_y_x_time)
    height, width, n_frames = frames.shape
    data = frames.reshape(height * width, n_frames)
    rng = np.random.default_rng(random_seed)
    if data.shape[1] < data.shape[0]:
        covariance = data.T @ data
        eigenvalues, temporal = eigs(
            covariance,
            n_components,
            v0=rng.standard_normal(covariance.shape[0]),
        )
        singular_values = np.sqrt(np.diag(eigenvalues))
        spatial = data @ temporal @ np.linalg.inv(singular_values)
    else:
        covariance = data @ data.T
        eigenvalues, spatial = eigs(
            covariance,
            n_components,
            v0=rng.standard_normal(covariance.shape[0]),
        )
        singular_values = np.sqrt(np.diag(eigenvalues))
        temporal = (
            np.linalg.inv(singular_values) @ spatial.T @ data
        ).T
    reconstruction = (
        np.real(spatial)
        @ np.real(singular_values)
        @ np.real(temporal).T
    )
    return reconstruction.reshape(height, width, n_frames)


def detect_coarse_spikes_chunked(
    highpass_movie_path: Path,
    *,
    search_mask: np.ndarray,
    polarity: int = 1,
    spatial_sigma_px: float = 1.8,
    spatial_radius_px: int = 2,
    threshold_sd: float = 5.0,
    minimum_segment_voxels: int = 5,
    chunk_frames: int = 1_000,
    overlap_frames: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run pyALI-style coarse 3-D event detection in temporal chunks.

    The Python port estimates per-pixel noise with standard deviation after
    spatial Gaussian filtering. Chunk overlap preserves components crossing a
    chunk boundary; a component is emitted only by the chunk containing its
    peak frame.
    """

    if polarity not in (-1, 1):
        raise ValueError("polarity must be +1 or -1")
    movie = tifffile.memmap(highpass_movie_path, mode="r")
    n_frames, height, width = movie.shape
    search = _validate_mask(
        search_mask,
        (height, width),
        name="search_mask",
    )

    running_sum = np.zeros((height, width), dtype=np.float64)
    running_square = np.zeros((height, width), dtype=np.float64)
    count = 0
    for start in range(0, n_frames, chunk_frames):
        stop = min(start + chunk_frames, n_frames)
        frames = polarity * np.asarray(
            movie[start:stop],
            dtype=np.float32,
        )
        filtered = _spatial_filter_frames(
            frames,
            sigma_px=spatial_sigma_px,
            radius_px=spatial_radius_px,
        )
        running_sum += filtered.sum(axis=0, dtype=np.float64)
        running_square += np.square(
            filtered,
            dtype=np.float64,
        ).sum(axis=0)
        count += stop - start
    mean = running_sum / count
    variance = np.maximum(running_square / count - mean**2, 0)
    noise_sd = np.sqrt(variance).astype(np.float32)
    noise_sd[~search] = np.inf

    structure = np.ones((3, 3, 3), dtype=np.uint8)
    spike_records: list[tuple[int, int, int, float]] = []
    for core_start in range(0, n_frames, chunk_frames):
        core_stop = min(core_start + chunk_frames, n_frames)
        read_start = max(0, core_start - overlap_frames)
        read_stop = min(n_frames, core_stop + overlap_frames)
        frames = polarity * np.asarray(
            movie[read_start:read_stop],
            dtype=np.float32,
        )
        filtered = _spatial_filter_frames(
            frames,
            sigma_px=spatial_sigma_px,
            radius_px=spatial_radius_px,
        )
        thresholded = filtered > (
            threshold_sd * noise_sd[None, :, :]
        )
        thresholded[:, ~search] = False
        labels, n_labels = ndimage.label(
            thresholded,
            structure=structure,
        )
        if n_labels:
            counts = np.bincount(labels.ravel())
            label_ids = np.flatnonzero(
                counts >= minimum_segment_voxels
            )
            label_ids = label_ids[label_ids != 0]
            if label_ids.size:
                positions = ndimage.maximum_position(
                    filtered,
                    labels=labels,
                    index=label_ids.tolist(),
                )
                for temporal, row, column in positions:
                    frame = int(read_start + temporal)
                    if core_start <= frame < core_stop:
                        spike_records.append(
                            (
                                int(row),
                                int(column),
                                frame,
                                float(filtered[temporal, row, column]),
                            )
                        )
        print(
            f"ALI coarse detection "
            f"{core_stop:>6,} / {n_frames:,} frames"
        )

    if not spike_records:
        return (
            np.zeros((0, 3), dtype=np.int64),
            np.zeros(0, dtype=np.float32),
            noise_sd,
        )
    spike_records.sort(key=lambda values: (values[2], values[0], values[1]))
    spikes = np.asarray(
        [values[:3] for values in spike_records],
        dtype=np.int64,
    )
    peaks = np.asarray(
        [values[3] for values in spike_records],
        dtype=np.float32,
    )
    return spikes, peaks, noise_sd


def _extract_traces_chunked(
    movie_path: Path,
    footprints: np.ndarray,
    *,
    valid_fov_mask: np.ndarray,
    start_frame: int = 0,
    n_frames: int | None = None,
    ridge: float = 1e-6,
    chunk_frames: int = 1_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply pyALI's trace-decomposition equations out of core."""

    movie = tifffile.memmap(movie_path, mode="r")
    total_frames, height, width = movie.shape
    if n_frames is None:
        n_frames = total_frames - start_frame
    stop_frame = start_frame + n_frames
    if start_frame < 0 or n_frames < 1 or stop_frame > total_frames:
        raise ValueError("Requested trace window lies outside the movie")
    valid = _validate_mask(
        valid_fov_mask,
        (height, width),
        name="valid_fov_mask",
    )
    footprint_values = np.asarray(footprints, dtype=np.float64).copy()
    footprint_values[:, ~valid] = 0
    footprint_matrix = footprint_values.reshape(
        len(footprints),
        -1,
    ).T
    gram = footprint_matrix.T @ footprint_matrix
    diagonal = np.diag(gram).copy()
    if np.any(diagonal <= 0):
        raise ValueError("At least one ALI footprint has zero energy")
    projection = np.empty(
        (len(footprints), n_frames),
        dtype=np.float64,
    )
    for relative_start in range(0, n_frames, chunk_frames):
        relative_stop = min(relative_start + chunk_frames, n_frames)
        frames = np.asarray(
            movie[
                start_frame + relative_start : start_frame + relative_stop
            ],
            dtype=np.float32,
        )
        projection[:, relative_start:relative_stop] = (
            footprint_matrix.T
            @ frames.reshape(relative_stop - relative_start, -1).T
        )
        print(
            f"ALI trace decomposition "
            f"{relative_stop:>6,} / {n_frames:,} frames"
        )
    trace_ls = np.linalg.solve(
        gram + ridge * np.eye(len(footprints)),
        projection,
    )
    trace_new = trace_ls.copy()
    trace_nonnegative = np.maximum(0, trace_new)
    for component in range(len(footprints)):
        trace_new[component] = (
            trace_nonnegative[component]
            + (
                projection[component]
                - gram[component] @ trace_nonnegative
            )
            / diagonal[component]
        )
    trace_new -= np.median(trace_new, axis=1, keepdims=True)
    trace_ls -= np.median(trace_ls, axis=1, keepdims=True)
    return trace_ls.astype(np.float32), trace_new.astype(np.float32)


def estimate_ali_footprints(
    highpass_movie_path: Path,
    pyali_root: Path,
    *,
    cluster_centers: np.ndarray,
    assigned_spike_frames: list[np.ndarray],
    valid_fov_mask: np.ndarray,
    footprint_radius_px: float,
    footprint_smoothing_sigma_px: float = 0.0,
) -> np.ndarray:
    """Re-estimate ALI footprints while keeping detections fixed.

    Each footprint is the mean high-pass movie frame at the events already
    assigned to one ALI component. The upstream circular support constraint
    and optional spatial Gaussian smoothing are then applied exactly as in
    :func:`run_pyali`. This makes footprint-shape sweeps inexpensive because
    coarse detection, fine localization, and clustering are not repeated.
    """

    if footprint_radius_px <= 0:
        raise ValueError("footprint_radius_px must be positive")
    if footprint_smoothing_sigma_px < 0:
        raise ValueError(
            "footprint_smoothing_sigma_px must be non-negative"
        )
    centers = np.asarray(cluster_centers, dtype=np.float32)
    if centers.ndim != 2 or centers.shape[1] != 2:
        raise ValueError("cluster_centers must have shape (components, 2)")
    if len(assigned_spike_frames) != len(centers):
        raise ValueError(
            "assigned_spike_frames must match the number of centers"
        )

    movie = tifffile.memmap(highpass_movie_path, mode="r")
    if movie.ndim != 3:
        raise ValueError("High-pass TIFF must have shape (time, y, x)")
    n_frames, height, width = movie.shape
    valid = _validate_mask(
        valid_fov_mask,
        (height, width),
        name="valid_fov_mask",
    )
    footprint_y_x_cluster = np.zeros(
        (height, width, len(centers)),
        dtype=np.float32,
    )
    for cluster, frames in enumerate(assigned_spike_frames):
        frame_indices = np.asarray(frames, dtype=np.int64)
        if frame_indices.size == 0:
            raise ValueError(
                f"Component {cluster + 1} has no assigned spike frames"
            )
        if frame_indices.min() < 0 or frame_indices.max() >= n_frames:
            raise ValueError(
                f"Component {cluster + 1} has out-of-range spike frames"
            )
        footprint_y_x_cluster[:, :, cluster] = np.asarray(
            movie[frame_indices],
            dtype=np.float32,
        ).mean(axis=0)

    upstream = _load_pyali_functions(Path(pyali_root))
    supported = upstream["ali_fp_support"](
        footprint_y_x_cluster,
        centers.T,
        footprint_radius_px,
    )[0]
    footprints = supported.transpose(2, 0, 1).astype(np.float32)
    footprints[:, ~valid] = 0

    if footprint_smoothing_sigma_px > 0:
        row_grid, column_grid = np.ogrid[:height, :width]
        support_masks = np.empty(
            (len(centers), height, width),
            dtype=bool,
        )
        for cluster, center in enumerate(centers):
            support_masks[cluster] = (
                (row_grid + 1 - center[0]) ** 2
                + (column_grid + 1 - center[1]) ** 2
                <= footprint_radius_px**2
            )
        footprints = ndimage.gaussian_filter(
            footprints,
            sigma=(
                0,
                footprint_smoothing_sigma_px,
                footprint_smoothing_sigma_px,
            ),
            mode="nearest",
        ).astype(np.float32)
        footprints[~support_masks] = 0
        footprints[:, ~valid] = 0

    return footprints


def update_ali_result_footprints(
    result: ALIResult,
    footprints: np.ndarray,
    highpass_movie_path: Path,
    *,
    valid_fov_mask: np.ndarray,
    trace_movie_path: Path | None = None,
    trace_start_frame: int = 0,
    trace_chunk_frames: int = 1_000,
) -> ALIResult:
    """Replace footprints and re-extract all corresponding ALI traces."""

    values = np.asarray(footprints, dtype=np.float32)
    if values.shape != result.footprints.shape:
        raise ValueError(
            f"footprints have shape {values.shape}, "
            f"expected {result.footprints.shape}"
        )
    detection_trace_ls, detection_traces = _extract_traces_chunked(
        highpass_movie_path,
        values,
        valid_fov_mask=valid_fov_mask,
        chunk_frames=trace_chunk_frames,
    )
    if trace_movie_path is None:
        trace_ls = detection_trace_ls.copy()
        traces = detection_traces.copy()
    else:
        trace_ls, traces = _extract_traces_chunked(
            trace_movie_path,
            values,
            valid_fov_mask=valid_fov_mask,
            start_frame=trace_start_frame,
            n_frames=detection_traces.shape[1],
            chunk_frames=trace_chunk_frames,
        )

    metrics: list[dict[str, float | int]] = []
    for cluster, (footprint, trace, frames) in enumerate(
        zip(values, detection_traces, result.assigned_spike_frames),
        start=1,
    ):
        peak = float(np.max(np.abs(footprint)))
        support = np.abs(footprint) >= 0.20 * max(peak, 1e-8)
        sigma = float(
            np.median(np.abs(trace - np.median(trace))) / 0.67448975
        )
        metrics.append(
            {
                "cell": cluster,
                "assigned_spikes": int(len(frames)),
                "footprint_area_pixels": int(support.sum()),
                "detection_trace_noise_sd": sigma,
            }
        )

    return ALIResult(
        coarse_spikes=result.coarse_spikes.copy(),
        coarse_peak_values=result.coarse_peak_values.copy(),
        fine_spike_locations=result.fine_spike_locations.copy(),
        ali_map=result.ali_map.copy(),
        cluster_centers=result.cluster_centers.copy(),
        cluster_indices=result.cluster_indices.copy(),
        footprints=values,
        detection_least_squares_traces=detection_trace_ls,
        detection_traces=detection_traces,
        least_squares_traces=trace_ls,
        traces=traces,
        assigned_spike_frames=[
            np.asarray(frames, dtype=np.int64).copy()
            for frames in result.assigned_spike_frames
        ],
        component_metrics=metrics,
    )


def run_pyali(
    highpass_movie_path: Path,
    pyali_root: Path,
    *,
    valid_fov_mask: np.ndarray,
    search_mask: np.ndarray,
    polarity: int = 1,
    spatial_sigma_px: float = 1.8,
    spatial_radius_px: int = 2,
    coarse_threshold_sd: float = 5.0,
    minimum_segment_voxels: int = 5,
    coarse_chunk_frames: int = 1_000,
    coarse_cache_path: Path | None = None,
    overwrite_coarse_cache: bool = False,
    maximum_coarse_events: int | None = None,
    n_svd_components: int = 25,
    svd_random_seed: int | None = None,
    fine_n_pixels: int = 15,
    fine_radius_px: float = 4.0,
    ali_upsampling_factor: int = 4,
    ali_smoothing_sigma: float = 0.7,
    ali_smoothing_radius: int = 2,
    ali_peak_threshold: float = 2.0,
    ali_peak_min_distance: int = 2,
    cluster_assignment_radius_px: float = 1.5,
    minimum_cluster_spikes: int = 2,
    cluster_boundary_distance_px: float = 2.0,
    footprint_radius_px: float = 10.0,
    footprint_smoothing_sigma_px: float = 0.0,
    trace_movie_path: Path | None = None,
    trace_start_frame: int = 0,
    trace_chunk_frames: int = 1_000,
) -> tuple[ALIResult, np.ndarray]:
    """Run upstream pyALI localization with a boundary-safe local adapter."""

    upstream = _load_pyali_functions(pyali_root)
    movie = tifffile.memmap(highpass_movie_path, mode="r")
    n_frames, height, width = movie.shape
    valid = _validate_mask(
        valid_fov_mask,
        (height, width),
        name="valid_fov_mask",
    )
    search = _validate_mask(
        search_mask,
        (height, width),
        name="search_mask",
    )
    search &= valid
    if footprint_radius_px <= 0:
        raise ValueError("footprint_radius_px must be positive")
    if footprint_smoothing_sigma_px < 0:
        raise ValueError(
            "footprint_smoothing_sigma_px must be non-negative"
        )

    coarse_signature = {
        "version": 1,
        "highpass_movie": _source_signature(highpass_movie_path),
        "search_mask_sha256": _array_digest(search),
        "polarity": int(polarity),
        "spatial_sigma_px": float(spatial_sigma_px),
        "spatial_radius_px": int(spatial_radius_px),
        "coarse_threshold_sd": float(coarse_threshold_sd),
        "minimum_segment_voxels": int(minimum_segment_voxels),
    }
    coarse_cache_hit = False
    if coarse_cache_path is not None:
        coarse_cache_path = Path(coarse_cache_path)
    if (
        coarse_cache_path is not None
        and coarse_cache_path.exists()
        and not overwrite_coarse_cache
    ):
        with np.load(coarse_cache_path, allow_pickle=False) as cached:
            cached_signature = json.loads(
                str(cached["signature_json"].item())
            )
            if cached_signature == coarse_signature:
                coarse_spikes = np.asarray(
                    cached["coarse_spikes"],
                    dtype=np.int64,
                )
                coarse_peaks = np.asarray(
                    cached["coarse_peak_values"],
                    dtype=np.float32,
                )
                noise_sd = np.asarray(
                    cached["noise_sd"],
                    dtype=np.float32,
                )
                coarse_cache_hit = True
            else:
                raise FileExistsError(
                    "Existing coarse-event cache is incompatible; set "
                    "overwrite_coarse_cache=True"
                )
    if not coarse_cache_hit:
        coarse_spikes, coarse_peaks, noise_sd = (
            detect_coarse_spikes_chunked(
                highpass_movie_path,
                search_mask=search,
                polarity=polarity,
                spatial_sigma_px=spatial_sigma_px,
                spatial_radius_px=spatial_radius_px,
                threshold_sd=coarse_threshold_sd,
                minimum_segment_voxels=minimum_segment_voxels,
                chunk_frames=coarse_chunk_frames,
            )
        )
        if coarse_cache_path is not None:
            coarse_cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                coarse_cache_path,
                signature_json=np.asarray(
                    json.dumps(coarse_signature, sort_keys=True)
                ),
                coarse_spikes=coarse_spikes,
                coarse_peak_values=coarse_peaks,
                noise_sd=noise_sd,
            )
    else:
        print(f"Reusing pyALI coarse-event cache: {coarse_cache_path}")
    if (
        maximum_coarse_events is not None
        and len(coarse_spikes) > maximum_coarse_events
    ):
        raise RuntimeError(
            f"pyALI found {len(coarse_spikes):,} coarse events, exceeding "
            f"the configured safety limit of {maximum_coarse_events:,}; "
            "the coarse cache was retained, but SVD and demixing were skipped"
        )
    if len(coarse_spikes) < 2:
        raise RuntimeError(
            "pyALI found fewer than two coarse events; inspect polarity and "
            "coarse_threshold_sd"
        )

    candidate_frames = np.asarray(
        movie[coarse_spikes[:, 2]],
        dtype=np.float32,
    ).transpose(1, 2, 0)
    candidate_frames[~valid] = 0
    adaptive_rank = min(
        n_svd_components,
        candidate_frames.shape[2] - 1,
        height * width - 1,
    )
    if adaptive_rank < 1:
        raise RuntimeError("Not enough candidate frames for pyALI SVD")
    if svd_random_seed is None:
        denoised_frames = upstream["ali_denoising"](
            candidate_frames,
            adaptive_rank,
        )[0]
    else:
        denoised_frames = _deterministic_ali_denoising(
            candidate_frames,
            adaptive_rank,
            random_seed=svd_random_seed,
        )
    fine_locations = upstream["ali_spk_fine"](
        polarity * denoised_frames,
        fine_n_pixels,
        fine_radius_px,
        coarse_spikes[:, :2],
    )[0]
    del candidate_frames, denoised_frames

    rounded = np.rint(fine_locations).astype(np.int64)
    finite = np.all(np.isfinite(fine_locations), axis=1)
    in_bounds = (
        (rounded[:, 0] >= 0)
        & (rounded[:, 0] < height)
        & (rounded[:, 1] >= 0)
        & (rounded[:, 1] < width)
    )
    accepted = finite & in_bounds
    accepted_indices = np.flatnonzero(accepted)
    accepted[accepted_indices] &= search[
        rounded[accepted_indices, 0],
        rounded[accepted_indices, 1],
    ]
    coarse_spikes = coarse_spikes[accepted]
    coarse_peaks = coarse_peaks[accepted]
    fine_locations = fine_locations[accepted]
    if len(coarse_spikes) < 2:
        raise RuntimeError("Fewer than two fine-localized events were valid")

    count_map, centers = upstream["ali_density_map"](
        fine_locations,
        (height, width),
        ali_upsampling_factor,
    )
    ali_map = ndimage.gaussian_filter(
        count_map,
        sigma=ali_smoothing_sigma,
        radius=ali_smoothing_radius,
    )
    peaks = peak_local_max(
        ali_map,
        threshold_abs=ali_peak_threshold,
        min_distance=ali_peak_min_distance,
    )
    if len(peaks) == 0:
        raise RuntimeError(
            "No ALI-map peaks crossed ali_peak_threshold"
        )
    cluster_centers_2_by_n = np.asarray(
        [
            centers[0][peaks[:, 0]],
            centers[1][peaks[:, 1]],
        ],
        dtype=np.float64,
    )
    center_rows = np.clip(
        np.rint(cluster_centers_2_by_n[0]).astype(int),
        0,
        height - 1,
    )
    center_columns = np.clip(
        np.rint(cluster_centers_2_by_n[1]).astype(int),
        0,
        width - 1,
    )
    distance_to_boundary = ndimage.distance_transform_edt(search)
    center_valid = (
        search[center_rows, center_columns]
        & (
            distance_to_boundary[center_rows, center_columns]
            > cluster_boundary_distance_px
        )
    )
    cluster_centers_2_by_n = cluster_centers_2_by_n[:, center_valid]
    if cluster_centers_2_by_n.shape[1] == 0:
        raise RuntimeError("All ALI peaks were boundary-adjacent")

    cluster_indices = upstream["ali_assign_cluster"](
        fine_locations,
        cluster_centers_2_by_n,
        cluster_assignment_radius_px,
    )
    counts = np.asarray(
        [
            np.count_nonzero(cluster_indices == index + 1)
            for index in range(cluster_centers_2_by_n.shape[1])
        ]
    )
    keep = counts >= minimum_cluster_spikes
    cluster_centers_2_by_n = cluster_centers_2_by_n[:, keep]
    if cluster_centers_2_by_n.shape[1] == 0:
        raise RuntimeError(
            "No ALI clusters retained the minimum assigned spike count"
        )
    cluster_indices = upstream["ali_assign_cluster"](
        fine_locations,
        cluster_centers_2_by_n,
        cluster_assignment_radius_px,
    )

    n_clusters = cluster_centers_2_by_n.shape[1]
    footprint_y_x_cluster = np.zeros(
        (height, width, n_clusters),
        dtype=np.float32,
    )
    assigned_spike_frames: list[np.ndarray] = []
    for cluster in range(n_clusters):
        event_indices = np.flatnonzero(cluster_indices == cluster + 1)
        frames = coarse_spikes[event_indices, 2]
        assigned_spike_frames.append(
            np.sort(frames.astype(np.int64))
        )
        footprint_y_x_cluster[:, :, cluster] = np.asarray(
            movie[frames],
            dtype=np.float32,
        ).mean(axis=0)
    supported_footprints = upstream["ali_fp_support"](
        footprint_y_x_cluster,
        cluster_centers_2_by_n,
        footprint_radius_px,
    )[0]
    footprints = supported_footprints.transpose(2, 0, 1).astype(
        np.float32
    )
    footprints[:, ~valid] = 0
    if footprint_smoothing_sigma_px > 0:
        row_grid, column_grid = np.ogrid[:height, :width]
        support_masks = np.empty(
            (n_clusters, height, width),
            dtype=bool,
        )
        for cluster, center in enumerate(
            cluster_centers_2_by_n.T
        ):
            support_masks[cluster] = (
                (row_grid + 1 - center[0]) ** 2
                + (column_grid + 1 - center[1]) ** 2
                <= footprint_radius_px**2
            )
        footprints = ndimage.gaussian_filter(
            footprints,
            sigma=(
                0,
                footprint_smoothing_sigma_px,
                footprint_smoothing_sigma_px,
            ),
            mode="nearest",
        ).astype(np.float32)
        footprints[~support_masks] = 0
        footprints[:, ~valid] = 0

    detection_trace_ls, detection_traces = _extract_traces_chunked(
        highpass_movie_path,
        footprints,
        valid_fov_mask=valid,
        chunk_frames=trace_chunk_frames,
    )
    if trace_movie_path is None:
        trace_ls = detection_trace_ls.copy()
        traces = detection_traces.copy()
    else:
        trace_ls, traces = _extract_traces_chunked(
            trace_movie_path,
            footprints,
            valid_fov_mask=valid,
            start_frame=trace_start_frame,
            n_frames=n_frames,
            chunk_frames=trace_chunk_frames,
        )
    cluster_centers = cluster_centers_2_by_n.T.astype(np.float32)
    metrics: list[dict[str, float | int]] = []
    for cluster, (footprint, trace, frames) in enumerate(
        zip(footprints, detection_traces, assigned_spike_frames),
        start=1,
    ):
        peak = float(np.max(np.abs(footprint)))
        support = np.abs(footprint) >= 0.20 * peak
        sigma = float(
            np.median(np.abs(trace - np.median(trace))) / 0.67448975
        )
        event_values = trace[frames]
        metrics.append(
            {
                "cell": cluster,
                "center_row": float(cluster_centers[cluster - 1, 0]),
                "center_column": float(
                    cluster_centers[cluster - 1, 1]
                ),
                "assigned_spikes": int(len(frames)),
                "footprint_area_pixels": int(support.sum()),
                "median_event_snr": float(
                    np.median(event_values) / max(sigma, 1e-8)
                ),
            }
        )
    return (
        ALIResult(
            coarse_spikes=coarse_spikes,
            coarse_peak_values=coarse_peaks,
            fine_spike_locations=fine_locations.astype(np.float32),
            ali_map=ali_map.astype(np.float32),
            cluster_centers=cluster_centers,
            cluster_indices=cluster_indices.astype(np.int64),
            footprints=footprints,
            detection_least_squares_traces=detection_trace_ls,
            detection_traces=detection_traces,
            least_squares_traces=trace_ls,
            traces=traces,
            assigned_spike_frames=assigned_spike_frames,
            component_metrics=metrics,
        ),
        noise_sd,
    )


def save_ali_results(
    result: ALIResult,
    output_directory: Path,
    *,
    source_movie_path: Path,
    highpass_movie_path: Path,
    valid_fov_mask: np.ndarray,
    search_mask: np.ndarray,
    pyali_commit: str,
    parameters: dict[str, object],
) -> tuple[Path, Path]:
    """Save portable ALI numeric arrays and provenance metadata."""

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    numeric_path = output_directory / "pyali_results.npz"
    metadata_path = output_directory / "pyali_metadata.json"
    spike_offsets = np.zeros(
        len(result.assigned_spike_frames) + 1,
        dtype=np.int64,
    )
    spike_offsets[1:] = np.cumsum(
        [len(values) for values in result.assigned_spike_frames]
    )
    assigned_frames = (
        np.concatenate(result.assigned_spike_frames)
        if result.assigned_spike_frames
        else np.zeros(0, dtype=np.int64)
    )
    np.savez_compressed(
        numeric_path,
        coarse_spikes=result.coarse_spikes,
        coarse_peak_values=result.coarse_peak_values,
        fine_spike_locations=result.fine_spike_locations,
        ali_map=result.ali_map,
        cluster_centers=result.cluster_centers,
        cluster_indices=result.cluster_indices,
        footprints=result.footprints,
        detection_least_squares_traces=(
            result.detection_least_squares_traces
        ),
        detection_traces=result.detection_traces,
        least_squares_traces=result.least_squares_traces,
        traces=result.traces,
        assigned_spike_frames=assigned_frames,
        assigned_spike_offsets=spike_offsets,
        valid_fov_mask=np.asarray(valid_fov_mask, dtype=bool),
        search_mask=np.asarray(search_mask, dtype=bool),
        cell_numbers=np.arange(1, len(result.footprints) + 1),
    )
    metadata = {
        "format_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "algorithm": "Activity Localization Imaging via pyALI",
        "upstream_repository": "https://github.com/spinaldynamicslab/pyALI",
        "upstream_commit": pyali_commit,
        "upstream_license": "GPL-3.0",
        "source_movie": _source_signature(source_movie_path),
        "highpass_movie": _source_signature(highpass_movie_path),
        "footprint_sha256": _array_digest(result.footprints),
        "n_coarse_spikes": int(len(result.coarse_spikes)),
        "n_clusters": int(len(result.footprints)),
        "component_metrics": result.component_metrics,
        "parameters": parameters,
        "numeric_results_path": str(numeric_path.resolve()),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))
    return numeric_path, metadata_path


def load_ali_results(
    numeric_path: Path,
    metadata_path: Path | None = None,
) -> ALIResult:
    """Load a result written by :func:`save_ali_results`."""

    numeric_path = Path(numeric_path)
    if metadata_path is None:
        metadata_path = numeric_path.with_name("pyali_metadata.json")
    metadata_path = Path(metadata_path)
    metadata = (
        json.loads(metadata_path.read_text())
        if metadata_path.exists()
        else {}
    )
    with np.load(numeric_path, allow_pickle=False) as saved:
        packed_frames = np.asarray(
            saved["assigned_spike_frames"],
            dtype=np.int64,
        )
        offsets = np.asarray(
            saved["assigned_spike_offsets"],
            dtype=np.int64,
        )
        assigned_frames = [
            packed_frames[offsets[index] : offsets[index + 1]].copy()
            for index in range(len(offsets) - 1)
        ]
        metrics = metadata.get("component_metrics", [])
        if len(metrics) != len(assigned_frames):
            metrics = [
                {
                    "cell": index + 1,
                    "assigned_spikes": int(len(frames)),
                }
                for index, frames in enumerate(assigned_frames)
            ]
        return ALIResult(
            coarse_spikes=np.asarray(saved["coarse_spikes"]),
            coarse_peak_values=np.asarray(saved["coarse_peak_values"]),
            fine_spike_locations=np.asarray(
                saved["fine_spike_locations"]
            ),
            ali_map=np.asarray(saved["ali_map"]),
            cluster_centers=np.asarray(saved["cluster_centers"]),
            cluster_indices=np.asarray(saved["cluster_indices"]),
            footprints=np.asarray(saved["footprints"]),
            detection_least_squares_traces=np.asarray(
                saved["detection_least_squares_traces"]
            ),
            detection_traces=np.asarray(saved["detection_traces"]),
            least_squares_traces=np.asarray(
                saved["least_squares_traces"]
            ),
            traces=np.asarray(saved["traces"]),
            assigned_spike_frames=assigned_frames,
            component_metrics=metrics,
        )


def plot_ali_results(
    mean_image: np.ndarray,
    result: ALIResult,
    *,
    valid_fov_mask: np.ndarray,
    search_mask: np.ndarray,
    frame_rate_hz: float,
    trace_seconds: float = 10.0,
    trace_source_label: str = "demixed source-movie traces",
    trace_high_pass_hz: float | None = None,
    trace_high_pass_order: int = 3,
    ali_map_display_gamma: float | None = 0.5,
    ali_map_display_upper_percentile: float = 99.5,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot localization events, ALI map, footprints, and extracted traces.

    The optional high-pass is applied only to the displayed trace copy. It
    does not modify ``result.traces`` or any serialized ALI result.
    ``ali_map_display_gamma`` applies a display-only power-law transform to
    reveal weaker event-density peaks without changing ``result.ali_map``.
    """

    mean_image = np.asarray(mean_image, dtype=np.float32)
    valid = _validate_mask(
        valid_fov_mask,
        mean_image.shape,
        name="valid_fov_mask",
    )
    search = _validate_mask(
        search_mask,
        mean_image.shape,
        name="search_mask",
    )
    display_mean = mean_image.copy()
    display_mean[~valid] = float(np.min(display_mean[valid]))
    low, high = np.percentile(display_mean[np.isfinite(display_mean)], [1, 99])

    fig = plt.figure(figsize=(8.0, 8.0), constrained_layout=True)
    grid = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 1.5])
    spike_axis = fig.add_subplot(grid[0, 0])
    map_axis = fig.add_subplot(grid[0, 1])
    footprint_axis = fig.add_subplot(grid[1, 0])
    weight_axis = fig.add_subplot(grid[1, 1])
    trace_axis = fig.add_subplot(grid[2, :])

    spike_axis.imshow(display_mean, cmap="gray", vmin=low, vmax=high)
    spike_axis.scatter(
        result.fine_spike_locations[:, 1],
        result.fine_spike_locations[:, 0],
        s=2,
        c="#D55E00",
        alpha=0.55,
        linewidths=0,
    )
    spike_axis.contour(
        search.astype(float),
        levels=[0.5],
        colors=["#56B4E9"],
        linewidths=0.4,
    )
    spike_axis.set_title(
        f"Fine-localized candidate events (n={len(result.coarse_spikes)})"
    )

    ali_map = np.asarray(result.ali_map, dtype=np.float32)
    map_norm = None
    map_display_label = "linear"
    if ali_map_display_gamma is not None:
        gamma = float(ali_map_display_gamma)
        upper_percentile = float(ali_map_display_upper_percentile)
        if gamma <= 0:
            raise ValueError("ali_map_display_gamma must be positive")
        if not 0 < upper_percentile <= 100:
            raise ValueError(
                "ali_map_display_upper_percentile must lie in (0, 100]"
            )
        positive_density = ali_map[np.isfinite(ali_map) & (ali_map > 0)]
        display_maximum = (
            float(np.percentile(positive_density, upper_percentile))
            if positive_density.size
            else 1.0
        )
        map_norm = PowerNorm(
            gamma=gamma,
            vmin=0,
            vmax=max(display_maximum, np.finfo(np.float32).eps),
            clip=True,
        )
        map_display_label = (
            f"power γ={gamma:g}, "
            f"positive p{upper_percentile:g} clip"
        )
    map_axis.imshow(
        ali_map,
        cmap="magma",
        norm=map_norm,
        extent=(0, mean_image.shape[1], mean_image.shape[0], 0),
        aspect="equal",
    )
    map_axis.scatter(
        result.cluster_centers[:, 1],
        result.cluster_centers[:, 0],
        s=18,
        facecolors="none",
        edgecolors="#00BFC4",
        linewidths=0.8,
    )
    map_axis.set_title(
        "ALI event-density map and detected peaks\n"
        f"display: {map_display_label}"
    )

    footprint_axis.imshow(display_mean, cmap="gray", vmin=low, vmax=high)
    colors = plt.cm.tab20(
        np.linspace(0, 1, max(1, len(result.footprints)))
    )
    footprint_sum = np.zeros_like(mean_image, dtype=np.float32)
    for cell, (footprint, center, color) in enumerate(
        zip(result.footprints, result.cluster_centers, colors),
        start=1,
    ):
        positive = np.maximum(footprint, 0)
        scale = max(float(positive.max()), 1e-8)
        footprint_sum += positive / scale
        for contour in find_contours(positive, 0.20 * scale):
            footprint_axis.plot(
                contour[:, 1],
                contour[:, 0],
                color=color,
                linewidth=0.8,
            )
        footprint_axis.text(
            center[1],
            center[0],
            str(cell),
            color=color,
            fontsize=7,
            fontweight="bold",
            ha="center",
            va="center",
        )
    footprint_axis.set_title(
        f"ALI footprints (n={len(result.footprints)})"
    )
    weight_axis.imshow(footprint_sum, cmap="magma")
    weight_axis.set_title("Sum of normalized positive footprint weights")

    display_traces = result.traces.astype(np.float64)
    trace_filter_label = ""
    if trace_high_pass_hz is not None:
        cutoff_hz = float(trace_high_pass_hz)
        if not 0 < cutoff_hz < frame_rate_hz / 2:
            raise ValueError(
                "trace_high_pass_hz must lie between zero and Nyquist"
            )
        if trace_high_pass_order < 1:
            raise ValueError("trace_high_pass_order must be at least one")
        sos = signal.butter(
            trace_high_pass_order,
            cutoff_hz,
            btype="highpass",
            fs=frame_rate_hz,
            output="sos",
        )
        display_traces = signal.sosfiltfilt(
            sos,
            display_traces,
            axis=1,
        )
        trace_filter_label = (
            f", {cutoff_hz:g} Hz zero-phase high-pass"
        )

    n_display_frames = min(
        result.traces.shape[1],
        int(round(trace_seconds * frame_rate_hz)),
    )
    time_s = np.arange(n_display_frames) / frame_rate_hz
    traces = display_traces[:, :n_display_frames]
    trace_low = np.min(traces, axis=1, keepdims=True)
    trace_high = np.max(traces, axis=1, keepdims=True)
    trace_range = trace_high - trace_low
    normalized_traces = np.divide(
        traces - trace_low,
        trace_range,
        out=np.zeros_like(traces),
        where=trace_range > 0,
    )
    trace_spacing = 1.08
    trace_offsets = np.arange(len(traces), dtype=np.float64) * trace_spacing
    for cell, (trace, frames, color) in enumerate(
        zip(normalized_traces, result.assigned_spike_frames, colors),
        start=1,
    ):
        offset = trace_offsets[cell - 1]
        trace_axis.plot(
            time_s,
            trace + offset,
            color=color,
            linewidth=0.45,
        )
        displayed_events = frames[frames < n_display_frames]
        trace_axis.scatter(
            displayed_events / frame_rate_hz,
            trace[displayed_events] + offset,
            s=5,
            color="#D55E00",
            marker="|",
        )
    displayed_duration_s = n_display_frames / frame_rate_hz
    trace_axis.set_xlim(0, displayed_duration_s)
    trace_axis.set_ylim(
        -0.10,
        trace_offsets[-1] + 1.10,
    )
    trace_axis.set_yticks(trace_offsets + 0.5)
    trace_axis.set_yticklabels(
        [str(cell) for cell in range(1, len(traces) + 1)]
    )
    trace_axis.spines[["top", "right", "left"]].set_visible(False)
    trace_axis.tick_params(axis="y", length=0, pad=2)
    trace_axis.set_xlabel("Time (s)")
    trace_axis.set_ylabel("ALI component")
    trace_axis.set_title(
        f"{trace_source_label}{trace_filter_label}, "
        "independently min–max normalized to 0–1; "
        "red ticks mark localized events"
    )

    for axis in (spike_axis, map_axis, footprint_axis, weight_axis):
        axis.set_xticks([])
        axis.set_yticks([])
    return fig, np.asarray(
        [
            spike_axis,
            map_axis,
            footprint_axis,
            weight_axis,
            trace_axis,
        ],
        dtype=object,
    )

