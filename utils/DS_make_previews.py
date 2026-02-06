import argparse
import glob
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tifffile


def _percentiles_for_viewing(
    paths: List[str],
    *,
    start_frame: int,
    preview_frames: int,
    preview_stride: int,
    max_samples_per_movie: int = 16,
) -> Tuple[float, float]:
    samples = []
    for path in paths:
        if not path or (not os.path.exists(path)):
            continue
        try:
            with tifffile.TiffFile(path) as tif:
                n = len(tif.pages)
                if n <= 0:
                    continue
                first = max(0, int(start_frame))
                last = min(n - 1, first + preview_frames - 1)
                idxs = np.linspace(0, last, num=min(max_samples_per_movie, last + 1), dtype=int)
                for i in idxs:
                    if i < first:
                        continue
                    if i % preview_stride != 0:
                        continue
                    frame = tif.pages[int(i)].asarray()
                    frame = np.asarray(frame, dtype=np.float32)
                    # Spatially subsample large frames for cheaper percentile estimation.
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


def _save_side_by_side_preview(
    *,
    columns: List[Tuple[str, str]],
    out_path: str,
    start_frame: int,
    preview_frames: int,
    preview_stride: int,
    force: bool,
    verbose: bool,
) -> None:
    existing = [(label, p) for (label, p) in columns if p and os.path.exists(p)]
    if len(existing) < 2:
        return

    if os.path.exists(out_path) and (not force):
        try:
            if os.path.getsize(out_path) >= 1024:
                return
        except OSError:
            return

    lo, hi = _percentiles_for_viewing(
        [p for (_label, p) in existing],
        start_frame=int(start_frame),
        preview_frames=preview_frames,
        preview_stride=preview_stride,
    )

    sources: List[Tuple[str, str]] = [(label, p) for (label, p) in existing]

    try:
        tiffs = [tifffile.TiffFile(p) for (_label, p) in sources]
    except Exception as e:
        if verbose:
            print(f"[WARN] Failed opening preview sources for {out_path}: {e!r}", flush=True)
        return

    tmp_out = out_path + ".tmp"
    memmap_path = out_path + ".memmap"
    mm = None
    try:
        lengths = [len(t.pages) for t in tiffs]
        if not lengths or min(lengths) < 1:
            return
        start = max(0, int(start_frame))
        available = min(lengths) - start
        if available < 1:
            return
        n_frames = min(int(available), int(preview_frames))
        if n_frames < 1:
            return

        first_frames = [t.pages[start].asarray() for t in tiffs]
        h_min = min(f.shape[0] for f in first_frames)
        w_min = min(f.shape[1] for f in first_frames)

        for p in (tmp_out, memmap_path):
            if os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass

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
                arr = tif.pages[start + int(i)].asarray()
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

        if verbose:
            labels = [s[0] for s in sources]
            print(
                f"[INFO] Saved side-by-side preview: {out_path} (cols: {', '.join(labels)}; frames={out_count}; lo={lo:.3g} hi={hi:.3g})",
                flush=True,
            )
    except Exception as e:
        if verbose:
            print(f"[WARN] Failed to save side-by-side preview {out_path}: {e!r}", flush=True)
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


def _guess_output_folder(path: str) -> Optional[str]:
    if not path:
        return None
    if os.path.isdir(path):
        # If it already looks like an output folder.
        if any(os.path.exists(os.path.join(path, f)) for f in ("movReg.tif", "mean_image_mc.tif")):
            return path
        # If user passed the parent data folder.
        candidate = os.path.join(path, "output_final")
        if os.path.isdir(candidate):
            return candidate
    return None


def _find_bases(output_folder: str) -> List[str]:
    bases = []
    for p in glob.glob(os.path.join(output_folder, "*_preprocessed.tif")):
        base = os.path.basename(p)
        if base.endswith("_preprocessed.tif"):
            bases.append(base[: -len("_preprocessed.tif")])
    bases.sort()
    return bases


def _load_metadata(output_folder: str) -> Dict[str, Any]:
    meta_path = os.path.join(output_folder, "run_metadata.npy")
    if not os.path.exists(meta_path):
        return {}
    try:
        meta = np.load(meta_path, allow_pickle=True).item()
        return meta if isinstance(meta, dict) else {}
    except Exception:
        return {}


def _detect_initial_dark_frames_from_tif(
    path: str,
    *,
    frame_rate_hz: Optional[float],
    max_seconds: float,
    rel_threshold: float,
    max_check_frames_cap: int = 2000,
) -> int:
    if not path or (not os.path.exists(path)):
        return 0

    try:
        with tifffile.TiffFile(path) as tif:
            n = len(tif.pages)
            if n < 2:
                return 0

            fr = frame_rate_hz
            if fr is None or (not np.isfinite(fr)) or fr <= 0:
                max_check = min(n, 200)
            else:
                max_check = int(round(float(max_seconds) * float(fr)))
                max_check = max(1, min(n, min(int(max_check_frames_cap), max_check)))

            rel_thr = float(rel_threshold)
            if not np.isfinite(rel_thr) or rel_thr <= 0:
                rel_thr = 0.5

            start = min(max_check, n - 1)
            n_ref = min(64, n - start)
            idxs = np.linspace(start, n - 1, num=n_ref, dtype=int)
            m_ref = np.array([np.asarray(tif.pages[int(i)].asarray(), dtype=np.float32).mean() for i in idxs])
            ref = float(np.median(m_ref))
            mad = float(np.median(np.abs(m_ref - ref))) + 1e-12

            thr_rel = rel_thr * ref
            thr_mad = ref - 6.0 * mad
            thr = float(min(thr_rel, thr_mad))

            n_bad = 0
            for i in range(max_check):
                m = float(np.asarray(tif.pages[int(i)].asarray(), dtype=np.float32).mean())
                if np.isfinite(m) and (m < thr):
                    n_bad += 1
                else:
                    break
            return int(n_bad)
    except Exception:
        return 0


def _combine_preview_tifs(
    *,
    top_path: str,
    bottom_path: str,
    out_path: str,
    force: bool,
    verbose: bool,
) -> None:
    """
    Combine two preview stacks into a single stack:
      output frame = [top; bottom] (vertical concatenation), with zero padding to max width.
    """
    if not (top_path and bottom_path):
        return
    if (not os.path.exists(top_path)) or (not os.path.exists(bottom_path)):
        return

    if os.path.exists(out_path) and (not force):
        try:
            if os.path.getsize(out_path) >= 1024:
                return
        except OSError:
            return

    tmp_out = out_path + ".tmp"
    memmap_path = out_path + ".memmap"
    mm = None
    t_top = None
    t_bottom = None
    try:
        t_top = tifffile.TiffFile(top_path)
        t_bottom = tifffile.TiffFile(bottom_path)

        n = min(len(t_top.pages), len(t_bottom.pages))
        if n < 1:
            return

        top0 = np.asarray(t_top.pages[0].asarray())
        bot0 = np.asarray(t_bottom.pages[0].asarray())
        if top0.ndim != 2 or bot0.ndim != 2:
            return

        h_top, w_top = top0.shape
        h_bot, w_bot = bot0.shape
        h_out = int(h_top + h_bot)
        w_out = int(max(w_top, w_bot))

        for p in (tmp_out, memmap_path):
            if os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass

        mm = np.memmap(memmap_path, dtype=np.uint16, mode="w+", shape=(int(n), h_out, w_out))

        for i in range(n):
            top = np.asarray(t_top.pages[i].asarray())
            bot = np.asarray(t_bottom.pages[i].asarray())
            top = top[:h_top, :w_top]
            bot = bot[:h_bot, :w_bot]
            if top.dtype != np.uint16:
                top = top.astype(np.uint16, copy=False)
            if bot.dtype != np.uint16:
                bot = bot.astype(np.uint16, copy=False)

            mm[i, :, :] = 0
            mm[i, 0:h_top, 0:w_top] = top
            mm[i, h_top : h_top + h_bot, 0:w_bot] = bot

        mm.flush()
        tifffile.imsave(tmp_out, mm, bigtiff=True)
        del mm
        mm = None
        try:
            os.remove(memmap_path)
        except OSError:
            pass
        os.replace(tmp_out, out_path)

        if verbose:
            print(f"[INFO] Saved combined preview: {out_path} (frames={n})", flush=True)
    except Exception as e:
        if verbose:
            print(f"[WARN] Failed to combine previews into {out_path}: {e!r}", flush=True)
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
        for t in (t_top, t_bottom):
            if t is None:
                continue
            try:
                t.close()
            except Exception:
                pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 1.5: make side-by-side preview videos from Step1 outputs.")
    parser.add_argument("folders", nargs="+", help="Output folder(s) (or parent folder containing output_final/).")
    parser.add_argument("--preview-seconds", type=float, default=120.0)
    parser.add_argument("--preview-fps", type=float, default=50.0)
    parser.add_argument(
        "--preview-stride",
        type=int,
        default=0,
        help="If 0, compute stride from preview-fps and the movie frameRate if available; else use this stride.",
    )
    parser.add_argument(
        "--preview-start",
        type=int,
        default=None,
        help="Start frame index for previews. If set, overrides auto dark-frame skipping.",
    )
    parser.add_argument(
        "--combine-four",
        action="store_true",
        help="Also create a 2x2 combined preview: top row raw|raw_mc, bottom row preprocessed|movReg.",
    )
    parser.add_argument(
        "--movreg-denoised",
        action="store_true",
        help="Also create a side-by-side preview: movReg | movReg_denoised (if movReg_denoised.tif exists).",
    )
    parser.add_argument(
        "--movreg-denoised-masked",
        action="store_true",
        help="Also create a side-by-side preview: movReg | movReg_denoised_masked (if movReg_denoised_masked.tif exists).",
    )
    parser.add_argument(
        "--only-movreg",
        action="store_true",
        help="Only create movReg comparison previews (skip raw/preprocessed previews).",
    )
    parser.add_argument(
        "--keep-initial-dark",
        action="store_true",
        help="Do not skip initial dark frames; preview starts at frame 0.",
    )
    parser.add_argument("--initial-dark-max-seconds", type=float, default=2.0)
    parser.add_argument("--initial-dark-rel-threshold", type=float, default=0.5)
    parser.add_argument("--force", action="store_true", help="Overwrite existing previews.")
    parser.add_argument("--quiet", action="store_true", help="Less logging.")
    args = parser.parse_args()

    verbose = not bool(args.quiet)
    t0 = time.perf_counter()

    for folder in args.folders:
        out_dir = _guess_output_folder(folder)
        if out_dir is None:
            if verbose:
                print(f"[WARN] Not an output folder (and no output_final/ found): {folder}", flush=True)
            continue

        meta = _load_metadata(out_dir)
        source_fr = None
        try:
            if "frameRate" in meta:
                source_fr = float(meta["frameRate"])
        except Exception:
            source_fr = None

        bases = _find_bases(out_dir)
        if not bases:
            if verbose:
                print(f"[WARN] No *_preprocessed.tif found in: {out_dir}", flush=True)
            continue

        if verbose:
            print(f"[INFO] Processing {out_dir} (bases: {len(bases)})", flush=True)

        for base in bases:
            movie_tif = os.path.join(out_dir, f"{base}_movie.tif")
            preprocessed_tif = os.path.join(out_dir, f"{base}_preprocessed.tif")
            raw_mc_tif = os.path.join(out_dir, f"{base}_movie_mc.tif")
            movreg_tif = os.path.join(out_dir, "movReg.tif")
            movreg_denoised_tif = os.path.join(out_dir, "movReg_denoised.tif")
            movreg_denoised_masked_tif = os.path.join(out_dir, "movReg_denoised_masked.tif")

            # Determine stride. Prefer metadata frameRate if present.
            stride = int(args.preview_stride)
            if stride < 0:
                stride = 1
            if stride == 0:
                fr = source_fr
                if fr is None or (not np.isfinite(fr)) or fr <= 0:
                    stride = 1
                else:
                    stride = max(1, int(round(fr / float(args.preview_fps)))) if float(args.preview_fps) > 0 else 1

            fr_for_len = source_fr
            if fr_for_len is None or (not np.isfinite(fr_for_len)) or fr_for_len <= 0:
                fr_for_len = float(args.preview_fps) * float(stride)
            preview_frames = int(round(float(args.preview_seconds) * float(fr_for_len)))
            if preview_frames < 1:
                preview_frames = 1

            start_frame = 0
            if args.preview_start is not None:
                start_frame = max(0, int(args.preview_start))
            elif not bool(args.keep_initial_dark):
                fix = meta.get("initial_dark_frame_fix", {}) if isinstance(meta, dict) else {}
                if isinstance(fix, dict) and int(fix.get("n_fixed", 0) or 0) > 0:
                    start_frame = int(fix.get("first_good", fix.get("n_fixed", 0)) or 0)
                else:
                    start_frame = _detect_initial_dark_frames_from_tif(
                        movie_tif,
                        frame_rate_hz=source_fr,
                        max_seconds=float(args.initial_dark_max_seconds),
                        rel_threshold=float(args.initial_dark_rel_threshold),
                    )
            if verbose and start_frame > 0:
                print(
                    f"[INFO] Preview starts at frame {start_frame} (skip initial dark frames). "
                    f"stride={stride} preview_frames={preview_frames}",
                    flush=True,
                )

            if not bool(args.only_movreg):
                _save_side_by_side_preview(
                    columns=[("preprocessed", preprocessed_tif), ("movReg", movreg_tif)],
                    out_path=os.path.join(out_dir, f"{base}_preprocessed_vs_movReg_preview.tif"),
                    start_frame=start_frame,
                    preview_frames=preview_frames,
                    preview_stride=stride,
                    force=bool(args.force),
                    verbose=verbose,
                )

                _save_side_by_side_preview(
                    columns=[("raw", movie_tif), ("raw_mc", raw_mc_tif)],
                    out_path=os.path.join(out_dir, f"{base}_raw_vs_raw_mc_preview.tif"),
                    start_frame=start_frame,
                    preview_frames=preview_frames,
                    preview_stride=stride,
                    force=bool(args.force),
                    verbose=verbose,
                )

                if bool(args.combine_four):
                    top = os.path.join(out_dir, f"{base}_raw_vs_raw_mc_preview.tif")
                    bottom = os.path.join(out_dir, f"{base}_preprocessed_vs_movReg_preview.tif")
                    _combine_preview_tifs(
                        top_path=top,
                        bottom_path=bottom,
                        out_path=os.path.join(out_dir, f"{base}_4way_preview.tif"),
                        force=bool(args.force),
                        verbose=verbose,
                    )

            if bool(args.movreg_denoised):
                _save_side_by_side_preview(
                    columns=[("movReg", movreg_tif), ("movReg_denoised", movreg_denoised_tif)],
                    out_path=os.path.join(out_dir, f"{base}_movReg_vs_denoised_preview.tif"),
                    start_frame=start_frame,
                    preview_frames=preview_frames,
                    preview_stride=stride,
                    force=bool(args.force),
                    verbose=verbose,
                )

            if bool(args.movreg_denoised_masked):
                _save_side_by_side_preview(
                    columns=[("movReg", movreg_tif), ("movReg_denoised_masked", movreg_denoised_masked_tif)],
                    out_path=os.path.join(out_dir, f"{base}_movReg_vs_denoised_masked_preview.tif"),
                    start_frame=start_frame,
                    preview_frames=preview_frames,
                    preview_stride=stride,
                    force=bool(args.force),
                    verbose=verbose,
                )

    if verbose:
        print(f"[INFO] Done (elapsed {time.perf_counter() - t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
