import argparse
import glob
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import tifffile


def _session_output_dir(raw_path: str, output_subdir: str, run_tag: Optional[str]) -> str:
    out_dir = os.path.join(os.path.dirname(raw_path), output_subdir)
    if run_tag:
        out_dir = os.path.join(out_dir, run_tag)
    return out_dir


def _pick_single(paths: Sequence[str], *, label: str) -> str:
    paths = [p for p in paths if p]
    if not paths:
        raise FileNotFoundError("No candidates found for: {}".format(label))
    if len(paths) == 1:
        return paths[0]
    paths = sorted(paths)
    raise RuntimeError("Ambiguous {} ({} matches): {}".format(label, len(paths), paths))


def _find_raw_movie_tif(output_dir: str) -> str:
    cands = glob.glob(os.path.join(output_dir, "*_movie.tif"))
    cands = [
        p
        for p in cands
        if (not p.endswith("_movie_mc.tif"))
        and (not p.endswith("_preprocessed.tif"))
        and (not os.path.basename(p).startswith("._"))
    ]
    return _pick_single(cands, label="raw movie tif in {}".format(output_dir))


def _tif_n_frames(path: str) -> int:
    with tifffile.TiffFile(path) as tif:
        return len(tif.pages)


def _tif_frame_shape_dtype(path: str) -> Tuple[Tuple[int, int], np.dtype]:
    with tifffile.TiffFile(path) as tif:
        arr0 = tif.pages[0].asarray()
    if arr0.ndim != 2:
        raise ValueError("Expected 2D frames in {}, got {}".format(path, arr0.shape))
    return (int(arr0.shape[0]), int(arr0.shape[1])), arr0.dtype


def _concat_tifs(
    in_paths: Sequence[str],
    out_path: str,
    *,
    chunk_frames: int,
    verbose: bool = True,
) -> List[int]:
    in_paths = list(in_paths)
    if not in_paths:
        raise ValueError("No input tifs provided")

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(out_path):
        os.remove(out_path)

    ref_shape, ref_dtype = _tif_frame_shape_dtype(in_paths[0])
    edges = [0]
    written = 0

    for p in in_paths:
        shape, dtype = _tif_frame_shape_dtype(p)
        if shape != ref_shape:
            raise ValueError("Shape mismatch: {} is {}, expected {}".format(p, shape, ref_shape))
        if dtype != ref_dtype:
            raise ValueError("Dtype mismatch: {} is {}, expected {}".format(p, dtype, ref_dtype))

        n = _tif_n_frames(p)
        if verbose:
            print("[INFO] Concat {} frames from {}".format(n, p), flush=True)

        for start in range(0, n, int(chunk_frames)):
            end = min(start + int(chunk_frames), n)
            arr = tifffile.imread(p, key=slice(int(start), int(end)))
            tifffile.imsave(out_path, arr, bigtiff=True, append=(written > 0))
            written += int(arr.shape[0])

        edges.append(written)

    if verbose:
        print("[INFO] Wrote {} total frames to {}".format(written, out_path), flush=True)

    return edges


def _load_manual_rois_npz(path: str) -> np.ndarray:
    with np.load(path, allow_pickle=True) as d:
        if "ROIs" not in d:
            raise KeyError("Expected key 'ROIs' in {}".format(path))
        rois = np.asarray(d["ROIs"]).astype(bool)
    if rois.ndim == 2:
        rois = rois[None, ...]
    if rois.ndim != 3:
        raise ValueError("Unexpected ROIs array shape {} in {}".format(rois.shape, path))
    return rois


def _copy_first_manual_rois(session_dirs: Sequence[str], out_dir: str, *, verbose: bool = True) -> None:
    src_paths = [os.path.join(d, "manual_ROIs.npz") for d in session_dirs]
    src0 = _pick_single([p for p in src_paths if os.path.exists(p)], label="manual_ROIs.npz")
    rois0 = _load_manual_rois_npz(src0)

    for p in src_paths:
        if not os.path.exists(p):
            continue
        rois = _load_manual_rois_npz(p)
        if rois.shape != rois0.shape or (not np.array_equal(rois, rois0)):
            print("[WARN] manual_ROIs.npz differs across sessions; using the first one: {}".format(src0), flush=True)
            break

    dst = os.path.join(out_dir, "manual_ROIs.npz")
    os.makedirs(out_dir, exist_ok=True)
    shutil.copy(src0, dst)
    if verbose:
        print("[INFO] Copied manual ROIs: {} -> {}".format(src0, dst), flush=True)

    meta0 = os.path.join(os.path.dirname(src0), "manual_ROIs_meta.json")
    if os.path.exists(meta0):
        shutil.copy(meta0, os.path.join(out_dir, "manual_ROIs_meta.json"))


def _concat_motion_params(session_dirs: Sequence[str], out_dir: str, *, verbose: bool = True) -> None:
    shifts = []
    for d in session_dirs:
        p = os.path.join(d, "motion_correction_params.npy")
        if not os.path.exists(p):
            print("[WARN] Missing motion_correction_params.npy in {}".format(d), flush=True)
            continue
        obj = np.load(p, allow_pickle=True).item()
        if "reg_shifts" not in obj:
            print("[WARN] Missing reg_shifts in {}".format(p), flush=True)
            continue
        shifts.append(np.asarray(obj["reg_shifts"]))

    if not shifts:
        return

    reg_shifts = np.concatenate(shifts, axis=0)
    out_path = os.path.join(out_dir, "motion_correction_params.npy")
    np.save(out_path, {"reg_shifts": reg_shifts})
    if verbose:
        print("[INFO] Saved concatenated motion shifts: {}".format(out_path), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build an output_concat folder by concatenating multiple session outputs (output_final)."
    )
    parser.add_argument("--out-dir", required=True, help="Destination output folder (e.g., .../Awake/output_concat).")
    parser.add_argument(
        "--raw",
        action="append",
        default=[],
        help="Raw .raw file path for each session (used to locate its output folder). Repeat for each session.",
    )
    parser.add_argument("--output-subdir", default="output_final", help="Per-session output folder name (default: output_final).")
    parser.add_argument("--run-tag", default=None, help="Optional run tag subfolder under output-subdir.")
    parser.add_argument(
        "--chunk-frames",
        type=int,
        default=5000,
        help="Frames per read/write chunk when concatenating TIFFs (default: 5000).",
    )
    parser.add_argument(
        "--include-raw-movie",
        action="store_true",
        help="Also concatenate the raw movie tif (*_movie.tif) into output_concat.",
    )
    parser.add_argument(
        "--include-movreg",
        action="store_true",
        default=True,
        help="Concatenate movReg.tif into output_concat (default: True).",
    )
    parser.add_argument(
        "--include-masked-denoised",
        action="store_true",
        default=True,
        help="Concatenate movReg_denoised_masked.tif into output_concat (default: True).",
    )
    args = parser.parse_args()

    if not args.raw:
        raise ValueError("Provide at least one --raw path.")

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    session_dirs = [_session_output_dir(p, args.output_subdir, args.run_tag) for p in args.raw]
    missing = [d for d in session_dirs if not os.path.isdir(d)]
    if missing:
        raise FileNotFoundError("Missing session output folder(s): {}".format(missing))

    if args.include_raw_movie:
        raw_movie_paths = [_find_raw_movie_tif(d) for d in session_dirs]
        raw_name = os.path.basename(raw_movie_paths[0])
        raw_out = os.path.join(out_dir, raw_name)
        raw_edges = _concat_tifs(raw_movie_paths, raw_out, chunk_frames=int(args.chunk_frames))
    else:
        raw_edges = None

    movreg_edges = None
    if bool(args.include_movreg):
        movreg_paths = [_pick_single([os.path.join(d, "movReg.tif")], label="movReg.tif") for d in session_dirs]
        movreg_paths = [p for p in movreg_paths if os.path.exists(p)]
        if len(movreg_paths) != len(session_dirs):
            raise FileNotFoundError("Missing movReg.tif in some session folders.")
        movreg_out = os.path.join(out_dir, "movReg.tif")
        movreg_edges = _concat_tifs(movreg_paths, movreg_out, chunk_frames=int(args.chunk_frames))

    masked_edges = None
    if bool(args.include_masked_denoised):
        masked_paths = [_pick_single([os.path.join(d, "movReg_denoised_masked.tif")], label="movReg_denoised_masked.tif") for d in session_dirs]
        masked_paths = [p for p in masked_paths if os.path.exists(p)]
        if len(masked_paths) != len(session_dirs):
            raise FileNotFoundError("Missing movReg_denoised_masked.tif in some session folders.")
        masked_out = os.path.join(out_dir, "movReg_denoised_masked.tif")
        masked_edges = _concat_tifs(masked_paths, masked_out, chunk_frames=int(args.chunk_frames))

    # Manual ROIs (required for demix chunking)
    _copy_first_manual_rois(session_dirs, out_dir)

    # Motion shifts (optional but useful)
    _concat_motion_params(session_dirs, out_dir)

    # Session edges + manifest
    edges = masked_edges or movreg_edges or raw_edges
    if edges is not None:
        np.save(os.path.join(out_dir, "session_edges_frames.npy"), np.asarray(edges, dtype=np.int64))

    manifest: Dict[str, object] = {
        "out_dir": out_dir,
        "session_output_dirs": session_dirs,
        "raw_paths": list(args.raw),
        "output_subdir": str(args.output_subdir),
        "run_tag": args.run_tag,
        "chunk_frames": int(args.chunk_frames),
        "edges_source": "masked" if masked_edges is not None else ("movReg" if movreg_edges is not None else ("raw" if raw_edges is not None else None)),
        "session_edges_frames": edges,
        "movies": {
            "raw_movie": bool(args.include_raw_movie),
            "movReg": bool(args.include_movreg),
            "movReg_denoised_masked": bool(args.include_masked_denoised),
        },
    }
    with open(os.path.join(out_dir, "concat_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print("[INFO] Done. Output folder: {}".format(out_dir), flush=True)


if __name__ == "__main__":
    main()

