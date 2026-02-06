import argparse
import math
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import tifffile


def _load_motion_shifts(output_folder: Path) -> np.ndarray:
    params_path = output_folder / "motion_correction_params.npy"
    if not params_path.exists():
        raise FileNotFoundError(f"Missing motion correction params: {params_path}")
    params = np.load(str(params_path), allow_pickle=True).item()
    if "reg_shifts" not in params:
        raise KeyError(f"Expected 'reg_shifts' in {params_path}")
    shifts = np.asarray(params["reg_shifts"], dtype=np.float32)
    if shifts.ndim != 2 or shifts.shape[1] < 2:
        raise ValueError(f"Unexpected reg_shifts shape: {shifts.shape}")
    return shifts[:, :2]


def _load_union_mask(output_folder: Path, roi_npz: str, dilate_px: int) -> np.ndarray:
    roi_path = output_folder / roi_npz
    if not roi_path.exists():
        raise FileNotFoundError(f"Missing ROI masks: {roi_path}")
    with np.load(str(roi_path), allow_pickle=True) as d:
        if "ROIs" not in d:
            raise KeyError(f"Expected key 'ROIs' in {roi_path}")
        rois = np.asarray(d["ROIs"]).astype(bool)
    if rois.ndim == 2:
        rois = rois[None, ...]
    if rois.ndim != 3:
        raise ValueError(f"Unexpected ROIs array shape: {rois.shape}")
    mask = np.any(rois, axis=0)

    if dilate_px and dilate_px > 0:
        try:
            from scipy.ndimage import binary_dilation

            mask = binary_dilation(mask, iterations=int(dilate_px))
        except Exception as e:
            print(f"[WARN] Could not dilate mask (scipy missing?): {e!r}", flush=True)

    return mask.astype(bool)


def _resolve_output_folder(arg_path: str) -> Path:
    p = Path(arg_path)
    if p.is_dir():
        return p
    # If given a movie path, use its parent.
    if p.name.endswith(".tif") or p.name.endswith(".tiff"):
        return p.parent
    raise ValueError(f"Expected an output folder or a .tif path, got: {arg_path!r}")


def _read_movie_chunk(path: Path, start: int, end: int) -> np.ndarray:
    # tifffile.imread supports key slicing
    return tifffile.imread(str(path), key=slice(int(start), int(end)))


def _safe_cast(arr: np.ndarray, dtype: np.dtype) -> np.ndarray:
    if np.issubdtype(dtype, np.floating):
        return arr.astype(dtype, copy=False)
    info = np.iinfo(dtype)
    return np.clip(np.rint(arr), info.min, info.max).astype(dtype, copy=False)


def run_masked_motion_regression(
    output_folder: Path,
    *,
    in_name: str,
    out_name: str,
    roi_npz: str,
    dilate_px: int,
    chunk_frames: int,
    overlap_frames: int,
    pixel_block: int,
    zero_outside_mask: bool,
) -> Path:
    output_folder = Path(output_folder)
    in_path = output_folder / in_name
    if not in_path.exists():
        raise FileNotFoundError(f"Missing input movie: {in_path}")

    shifts = _load_motion_shifts(output_folder)
    mask = _load_union_mask(output_folder, roi_npz=roi_npz, dilate_px=dilate_px)

    with tifffile.TiffFile(str(in_path)) as tif:
        pages = tif.pages
        n_frames = len(pages)
        if n_frames < 1:
            raise ValueError(f"No frames found in {in_path}")
        first = pages[0].asarray()
        if first.ndim != 2:
            raise ValueError(f"Expected 2D frames, got {first.shape} in {in_path}")
        h, w = first.shape
        in_dtype = first.dtype

    if mask.shape != (h, w):
        raise ValueError(f"ROI mask shape {mask.shape} does not match movie {(h, w)}")
    if shifts.shape[0] < n_frames:
        raise ValueError(f"reg_shifts has {shifts.shape[0]} frames but movie has {n_frames}")

    # Build regressors from shifts (same as DS_motion_correction.py)
    motion_shiftx = shifts[:n_frames, 0] - float(np.mean(shifts[:n_frames, 0]))
    motion_shifty = shifts[:n_frames, 1] - float(np.mean(shifts[:n_frames, 1]))
    regressors = np.column_stack(
        [
            motion_shiftx,
            motion_shifty,
            motion_shiftx**2,
            motion_shifty**2,
            motion_shiftx * motion_shifty,
        ]
    ).astype(np.float32)

    mask_flat_idx = np.flatnonzero(mask.reshape(-1))
    n_mask_px = int(mask_flat_idx.size)
    if n_mask_px == 0:
        raise ValueError(f"Mask is empty (no ROI pixels) in {output_folder}")

    out_path = output_folder / out_name
    print(f"[INFO] Input:  {in_path}", flush=True)
    print(f"[INFO] Output: {out_path}", flush=True)
    print(f"[INFO] Frames: {n_frames}, shape: {h}x{w}, dtype: {in_dtype}", flush=True)
    print(f"[INFO] Mask pixels: {n_mask_px} (dilate_px={dilate_px}, zero_outside={zero_outside_mask})", flush=True)
    print(f"[INFO] chunk_frames={chunk_frames}, overlap_frames={overlap_frames}, pixel_block={pixel_block}", flush=True)

    if chunk_frames < 1:
        raise ValueError("--chunk-frames must be >= 1")
    if overlap_frames < 0:
        raise ValueError("--overlap-frames must be >= 0")
    if overlap_frames >= chunk_frames:
        raise ValueError("--overlap-frames must be < --chunk-frames")
    if pixel_block < 1:
        raise ValueError("--pixel-block must be >= 1")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    def _append_tiff(arr3d: np.ndarray, append: bool) -> None:
        out_cast = _safe_cast(arr3d, in_dtype)
        # Important: when appending chunked writes, tifffile's default ImageDescription metadata can encode
        # the first chunk's shape (e.g., [300000, H, W]). That may cause later readers (tifffile.imread)
        # to only load the *first* series/chunk. Disable metadata so the file is treated as a plain page stack.
        tifffile.imsave(
            str(out_path),
            out_cast,
            bigtiff=True,
            append=bool(append),
            metadata=None,
        )

    def _denoise_chunk(start: int, end: int) -> np.ndarray:
        X = regressors[start:end, :]  # (T, k)
        T = int(X.shape[0])
        X_aug = np.concatenate([np.ones((T, 1), dtype=np.float32), X], axis=1)  # (T, k+1)

        mov = _read_movie_chunk(in_path, start, end)
        if mov.ndim != 3:
            raise ValueError(f"Expected 3D chunk, got {mov.shape} from {in_path}")

        mov_f = mov.astype(np.float32, copy=False)
        out_chunk = np.zeros_like(mov_f) if zero_outside_mask else mov_f.copy()

        # Fill only masked pixels, blockwise (avoid huge T×M matrices).
        flat = mov_f.reshape(T, -1)
        out_flat = out_chunk.reshape(T, -1)
        for b0 in range(0, n_mask_px, int(pixel_block)):
            b1 = min(b0 + int(pixel_block), n_mask_px)
            idx = mask_flat_idx[b0:b1]
            Y = flat[:, idx]  # (T, B)
            # Solve least squares with intercept: Y ≈ a + X @ beta
            # We subtract only X @ beta (keep intercept) to reduce baseline steps.
            coef = np.linalg.lstsq(X_aug, Y, rcond=None)[0]  # (k+1, B)
            beta = coef[1:, :]  # (k, B)
            Y_denoised = Y - (X @ beta)
            out_flat[:, idx] = Y_denoised

        return out_chunk

    overlap = int(overlap_frames)
    if overlap <= 0:
        # Simple, non-overlapping chunks.
        wrote_any = False
        for start in range(0, n_frames, int(chunk_frames)):
            end = min(start + int(chunk_frames), n_frames)
            out_chunk = _denoise_chunk(start, end)
            _append_tiff(out_chunk, append=wrote_any)
            wrote_any = True
            print(f"[INFO] Wrote frames {start}:{end}", flush=True)
        return out_path

    # Overlap+blend: consecutive chunks overlap in time and are crossfaded to avoid discontinuities.
    stride = int(chunk_frames) - overlap
    if stride < 1:
        raise ValueError("Invalid overlap: --chunk-frames - --overlap-frames must be >= 1")

    prev_tail = None  # type: Optional[np.ndarray]
    wrote_any = False
    start = 0
    while start < n_frames:
        end = min(start + int(chunk_frames), n_frames)
        out_chunk = _denoise_chunk(start, end)
        Tchunk = int(end - start)

        # If this is the only chunk, just write it.
        if start == 0 and end >= n_frames:
            _append_tiff(out_chunk, append=False)
            print(f"[INFO] Wrote frames {start}:{end}", flush=True)
            break

        eff_ov = int(min(overlap, Tchunk))
        if eff_ov <= 0:
            _append_tiff(out_chunk, append=wrote_any)
            wrote_any = True
            print(f"[INFO] Wrote frames {start}:{end}", flush=True)
            start += stride
            prev_tail = None
            continue

        if start == 0:
            # Write all but the tail overlap; hold tail for blending with next chunk.
            head_len = Tchunk - eff_ov
            if head_len > 0:
                _append_tiff(out_chunk[:head_len], append=False)
                wrote_any = True
                print(f"[INFO] Wrote frames {start}:{start + head_len}", flush=True)
            prev_tail = out_chunk[head_len:]
            start += stride
            continue

        # Blend overlap with previous tail (same frames).
        if prev_tail is None:
            prev_tail = out_chunk[:eff_ov]
        prev_len = int(prev_tail.shape[0])
        eff_ov2 = int(min(eff_ov, prev_len))

        if eff_ov2 > 0:
            curr_ov = out_chunk[:eff_ov2]
            w = np.linspace(0.0, 1.0, eff_ov2, dtype=np.float32)[:, None, None]
            blended = (1.0 - w) * prev_tail[-eff_ov2:] + w * curr_ov
            _append_tiff(blended, append=wrote_any)
            wrote_any = True
            print(f"[INFO] Wrote blended frames {start}:{start + eff_ov2}", flush=True)

        # Last chunk: write everything after overlap and exit.
        if end >= n_frames:
            if Tchunk > eff_ov2:
                _append_tiff(out_chunk[eff_ov2:], append=True)
                print(f"[INFO] Wrote frames {start + eff_ov2}:{end}", flush=True)
            break

        # Middle: write the non-overlap middle, keep tail for next blend.
        mid_start = eff_ov2
        mid_end = Tchunk - eff_ov
        if mid_end > mid_start:
            _append_tiff(out_chunk[mid_start:mid_end], append=True)
            print(f"[INFO] Wrote frames {start + mid_start}:{start + mid_end}", flush=True)

        prev_tail = out_chunk[mid_end:]
        start += stride

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Masked motion-regression denoising for movReg.tif using manual_ROIs.npz."
    )
    parser.add_argument(
        "path",
        help="Output folder (contains movReg.tif, motion_correction_params.npy, manual_ROIs.npz) or a movReg.tif path.",
    )
    parser.add_argument("--in-name", default="movReg.tif", help="Input movie filename in the folder.")
    parser.add_argument(
        "--out-name", default="movReg_denoised_masked.tif", help="Output movie filename to write in the folder."
    )
    parser.add_argument("--roi-npz", default="manual_ROIs.npz", help="ROI .npz file (expects key ROIs).")
    parser.add_argument("--dilate-px", type=int, default=0, help="Dilate the union ROI mask by this many pixels.")
    parser.add_argument(
        "--chunk-frames",
        type=int,
        default=5000,
        help="Number of frames per regression chunk (smaller reduces memory).",
    )
    parser.add_argument(
        "--overlap-frames",
        type=int,
        default=0,
        help="Overlap (in frames) between consecutive chunks; blended to reduce chunk-edge discontinuities (default: 0).",
    )
    parser.add_argument(
        "--pixel-block",
        type=int,
        default=5000,
        help="Number of masked pixels to solve per block (smaller reduces memory).",
    )
    parser.add_argument(
        "--keep-outside-mask",
        action="store_true",
        help="Keep pixels outside the mask unchanged (default: set outside-mask pixels to 0).",
    )
    args = parser.parse_args()

    folder = _resolve_output_folder(args.path)
    run_masked_motion_regression(
        folder,
        in_name=str(args.in_name),
        out_name=str(args.out_name),
        roi_npz=str(args.roi_npz),
        dilate_px=int(args.dilate_px),
        chunk_frames=int(args.chunk_frames),
        overlap_frames=int(args.overlap_frames),
        pixel_block=int(args.pixel_block),
        zero_outside_mask=not bool(args.keep_outside_mask),
    )


if __name__ == "__main__":
    main()
