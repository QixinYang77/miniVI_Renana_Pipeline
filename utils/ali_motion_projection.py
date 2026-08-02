"""Out-of-core, all-pixel motion-artifact projection for cluster ALI runs."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import ndimage
import tifffile


MOTION_REGRESSOR_NAMES = ("dY", "dX", "dY2", "dX2", "dY*dX")


@dataclass(frozen=True)
class MotionProjectionSummary:
    """Compact diagnostic arrays saved beside the projected movie."""

    filtered_shifts: np.ndarray
    coefficient_maps: np.ndarray
    epoch_bounds: np.ndarray
    removed_std_image: np.ndarray
    variance_explained_image: np.ndarray
    cache_hit: bool = False


def _array_digest(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def _source_signature(path: Path) -> dict[str, object]:
    stat = Path(path).stat()
    return {
        "path": str(Path(path).resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def build_motion_regressors(
    shifts: np.ndarray,
    *,
    frame_rate_hz: float,
    trend_smoothing_seconds: float,
    jitter_smoothing_seconds: float,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Build centered dY/dX, quadratic, and interaction regressors.

    This follows the motion-artifact projection used in the local ALI
    illustration: subtract a slow shift trajectory, lightly smooth the
    residual jitter, then construct five motion terms.
    """

    shifts = np.asarray(shifts, dtype=np.float64)
    if shifts.ndim != 2 or shifts.shape[1] != 2:
        raise ValueError("shifts must have shape (time, 2)")
    if not np.all(np.isfinite(shifts)):
        raise ValueError("shifts contain non-finite values")
    if frame_rate_hz <= 0:
        raise ValueError("frame_rate_hz must be positive")
    if trend_smoothing_seconds <= 0 or jitter_smoothing_seconds <= 0:
        raise ValueError("motion smoothing durations must be positive")

    trend_frames = max(
        2,
        int(round(trend_smoothing_seconds * frame_rate_hz)),
    )
    jitter_frames = max(
        1,
        int(round(jitter_smoothing_seconds * frame_rate_hz)),
    )
    centered = shifts - shifts.mean(axis=0, keepdims=True)
    slow_trajectory = ndimage.uniform_filter1d(
        centered,
        size=trend_frames,
        axis=0,
        mode="nearest",
    )
    filtered = centered - slow_trajectory
    filtered = ndimage.uniform_filter1d(
        filtered,
        size=jitter_frames,
        axis=0,
        mode="nearest",
    )
    d_y = filtered[:, 0]
    d_x = filtered[:, 1]
    regressors = np.column_stack(
        [d_y, d_x, d_y**2, d_x**2, d_y * d_x]
    )
    return regressors.astype(np.float32), MOTION_REGRESSOR_NAMES


def _signature(
    source_path: Path,
    shifts: np.ndarray,
    *,
    frame_rate_hz: float,
    trend_smoothing_seconds: float,
    jitter_smoothing_seconds: float,
    epoch_seconds: float,
    preserve_epoch_mean: bool,
) -> dict[str, object]:
    return {
        "version": 1,
        "algorithm": "all_pixel_motion_artifact_projection_v1",
        "source": _source_signature(source_path),
        "shifts_sha256": _array_digest(
            np.asarray(shifts, dtype=np.float32)
        ),
        "frame_rate_hz": float(frame_rate_hz),
        "trend_smoothing_seconds": float(trend_smoothing_seconds),
        "jitter_smoothing_seconds": float(jitter_smoothing_seconds),
        "epoch_seconds": float(epoch_seconds),
        "preserve_epoch_mean": bool(preserve_epoch_mean),
        "regressor_names": list(MOTION_REGRESSOR_NAMES),
    }


def _load_summary(
    summary_path: Path,
    expected_signature: dict[str, object],
) -> MotionProjectionSummary | None:
    if not summary_path.exists():
        return None
    try:
        with np.load(summary_path, allow_pickle=False) as saved:
            signature = json.loads(str(saved["signature_json"].item()))
            if signature != expected_signature:
                return None
            return MotionProjectionSummary(
                filtered_shifts=np.asarray(saved["filtered_shifts"]),
                coefficient_maps=np.asarray(saved["coefficient_maps"]),
                epoch_bounds=np.asarray(saved["epoch_bounds"]),
                removed_std_image=np.asarray(saved["removed_std_image"]),
                variance_explained_image=np.asarray(
                    saved["variance_explained_image"]
                ),
                cache_hit=True,
            )
    except (KeyError, OSError, ValueError, json.JSONDecodeError):
        return None


def _save_summary(
    summary_path: Path,
    summary: MotionProjectionSummary,
    signature: dict[str, object],
) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = summary_path.with_name(f".{summary_path.name}.temporary.npz")
    np.savez_compressed(
        temporary,
        signature_json=np.asarray(json.dumps(signature, sort_keys=True)),
        filtered_shifts=summary.filtered_shifts,
        coefficient_maps=summary.coefficient_maps,
        epoch_bounds=summary.epoch_bounds,
        removed_std_image=summary.removed_std_image,
        variance_explained_image=summary.variance_explained_image,
    )
    temporary.replace(summary_path)


def write_motion_projected_tiff(
    source_path: Path,
    shifts: np.ndarray,
    *,
    frame_rate_hz: float,
    trend_smoothing_seconds: float,
    jitter_smoothing_seconds: float,
    epoch_seconds: float,
    output_path: Path,
    summary_path: Path,
    preserve_epoch_mean: bool = True,
    spatial_row_block: int = 4,
    overwrite: bool = False,
) -> tuple[Path, MotionProjectionSummary]:
    """Fit and subtract motion-linked signal independently at every pixel.

    No manual or cell ROI is used. The movie is processed by temporal epoch
    and small spatial row blocks, so memory use is bounded for full sessions.
    """

    source_path = Path(source_path)
    output_path = Path(output_path)
    summary_path = Path(summary_path)
    source = tifffile.memmap(source_path, mode="r")
    if source.ndim != 3:
        raise ValueError("source TIFF must have shape (time, y, x)")
    n_frames, height, width = source.shape
    shifts = np.asarray(shifts, dtype=np.float32)
    if shifts.shape != (n_frames, 2):
        raise ValueError(f"Expected shifts {(n_frames, 2)}, got {shifts.shape}")
    if epoch_seconds <= 0:
        raise ValueError("epoch_seconds must be positive")
    if spatial_row_block < 1:
        raise ValueError("spatial_row_block must be positive")

    signature = _signature(
        source_path,
        shifts,
        frame_rate_hz=frame_rate_hz,
        trend_smoothing_seconds=trend_smoothing_seconds,
        jitter_smoothing_seconds=jitter_smoothing_seconds,
        epoch_seconds=epoch_seconds,
        preserve_epoch_mean=preserve_epoch_mean,
    )
    if output_path.exists() and summary_path.exists() and not overwrite:
        existing = tifffile.memmap(output_path, mode="r")
        cached = _load_summary(summary_path, signature)
        if (
            existing.shape == source.shape
            and existing.dtype == np.float32
            and cached is not None
        ):
            print(f"Reusing motion-projected TIFF: {output_path}", flush=True)
            return output_path.resolve(), cached
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            "Existing motion-projection output is incompatible; use "
            f"--overwrite-motion-projection: {output_path}"
        )

    regressors, _ = build_motion_regressors(
        shifts,
        frame_rate_hz=frame_rate_hz,
        trend_smoothing_seconds=trend_smoothing_seconds,
        jitter_smoothing_seconds=jitter_smoothing_seconds,
    )
    epoch_frames = max(2, int(round(epoch_seconds * frame_rate_hz)))
    epoch_bounds = np.asarray(
        [
            (start, min(start + epoch_frames, n_frames))
            for start in range(0, n_frames, epoch_frames)
        ],
        dtype=np.int64,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    target = tifffile.memmap(
        output_path,
        shape=source.shape,
        dtype=np.float32,
        bigtiff=True,
    )
    coefficient_maps = np.zeros(
        (len(epoch_bounds), len(MOTION_REGRESSOR_NAMES), height, width),
        dtype=np.float32,
    )
    before_ss = np.zeros((height, width), dtype=np.float64)
    after_ss = np.zeros((height, width), dtype=np.float64)

    for epoch_index, (epoch_start, epoch_stop) in enumerate(epoch_bounds):
        epoch_start = int(epoch_start)
        epoch_stop = int(epoch_stop)
        epoch_regressors = regressors[epoch_start:epoch_stop].astype(
            np.float64
        )
        epoch_regressors -= epoch_regressors.mean(axis=0, keepdims=True)
        design = np.column_stack(
            [
                np.ones(epoch_stop - epoch_start, dtype=np.float64),
                epoch_regressors,
            ]
        )
        pseudoinverse = np.linalg.pinv(design, rcond=1e-10)

        for row_start in range(0, height, spatial_row_block):
            row_stop = min(row_start + spatial_row_block, height)
            block_height = row_stop - row_start
            input_block = np.asarray(
                source[epoch_start:epoch_stop, row_start:row_stop, :],
                dtype=np.float64,
            ).reshape(epoch_stop - epoch_start, -1)
            coefficients = pseudoinverse @ input_block
            fitted_motion = epoch_regressors @ coefficients[1:]
            if preserve_epoch_mean:
                output_block = input_block - fitted_motion
            else:
                output_block = input_block - (
                    coefficients[0][None, :] + fitted_motion
                )
            target[epoch_start:epoch_stop, row_start:row_stop, :] = (
                output_block.reshape(
                    epoch_stop - epoch_start,
                    block_height,
                    width,
                ).astype(np.float32)
            )
            coefficient_maps[
                epoch_index, :, row_start:row_stop, :
            ] = coefficients[1:].reshape(
                len(MOTION_REGRESSOR_NAMES),
                block_height,
                width,
            ).astype(np.float32)

            input_centered = input_block - input_block.mean(
                axis=0,
                keepdims=True,
            )
            output_centered = output_block - output_block.mean(
                axis=0,
                keepdims=True,
            )
            before_ss[row_start:row_stop] += np.sum(
                input_centered**2,
                axis=0,
            ).reshape(block_height, width)
            after_ss[row_start:row_stop] += np.sum(
                output_centered**2,
                axis=0,
            ).reshape(block_height, width)

        target.flush()
        print(
            f"Motion projection epoch {epoch_index + 1:,}/{len(epoch_bounds):,}: "
            f"frames {epoch_start:,}-{epoch_stop - 1:,}",
            flush=True,
        )

    explained_ss = np.maximum(before_ss - after_ss, 0.0)
    removed_std_image = np.sqrt(explained_ss / n_frames).astype(np.float32)
    variance_explained_image = np.divide(
        explained_ss,
        before_ss,
        out=np.zeros_like(explained_ss),
        where=before_ss > 0,
    )
    variance_explained_image = np.clip(
        variance_explained_image,
        0.0,
        1.0,
    ).astype(np.float32)
    summary = MotionProjectionSummary(
        filtered_shifts=regressors[:, :2],
        coefficient_maps=coefficient_maps,
        epoch_bounds=epoch_bounds,
        removed_std_image=removed_std_image,
        variance_explained_image=variance_explained_image,
    )
    target.flush()
    del target
    _save_summary(summary_path, summary, signature)
    return output_path.resolve(), summary
