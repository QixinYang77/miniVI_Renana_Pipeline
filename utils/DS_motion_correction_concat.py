"""
Concatenate multiple session .raw files into a single dataset folder and run motion correction on the concatenated movie.

This is a wrapper around:
  - utils_concat/concat_raw_sessions.py  (builds output_concat/Image_001_001.raw + copies Experiment.xml/ROIs.xaml)
  - utils/DS_motion_correction.py        (runs CaImAn motion correction on the concatenated raw)

Typical usage on the cluster (via run_python_job.sh):
  python utils/DS_motion_correction_concat.py \\
    --out-dir /ems/.../Awake/output_concat \\
    --raw /ems/.../2-good/.../Image_001_001.raw \\
    --raw /ems/.../3-good/.../Image_001_001.raw \\
    --frames-to-truncate-per-session 500 \\
    --mc-output-subdir output_final \\
    --mc-run-tag concat_mc \\
    --factor 1 1 \\
    --skip-roi-detection
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional


def _run_motion_correction(
    *,
    concat_raw_path: str,
    mc_output_subdir: str,
    mc_run_tag: Optional[str],
    factor_hw: List[int],
    frames_to_truncate: int,
    overwrite_output_folder: bool,
    skip_roi_detection: bool,
    extra_mc_args: List[str],
) -> None:
    # Import lazily so concatenation can run even if CaImAn isn't available in the local environment.
    import utils.DS_motion_correction as dsmc

    argv = [
        "DS_motion_correction.py",
        str(concat_raw_path),
        "--frames-to-truncate",
        str(int(frames_to_truncate)),
        "--output-subdir",
        str(mc_output_subdir),
        "--factor",
        str(int(factor_hw[0])),
        str(int(factor_hw[1])),
    ]
    if mc_run_tag:
        argv += ["--run-tag", str(mc_run_tag)]
    if skip_roi_detection:
        argv += ["--skip-roi-detection"]

    if overwrite_output_folder:
        argv += ["--overwrite-output-folder"]
    else:
        argv += ["--no-overwrite-output-folder"]

    if extra_mc_args:
        argv += list(extra_mc_args)

    prev_argv = sys.argv
    try:
        sys.argv = argv
        dsmc.main()
    finally:
        sys.argv = prev_argv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Concatenate multiple .raw files into output_concat and run motion correction on the concatenated raw."
    )
    # Concatenation args
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Destination folder (e.g., .../Awake/output_concat). Will contain Image_001_001.raw and metadata files.",
    )
    parser.add_argument(
        "--raw",
        action="append",
        default=[],
        help="Raw .raw file path for each session. Repeat for each session.",
    )
    parser.add_argument(
        "--out-raw-name",
        default="Image_001_001.raw",
        help="Output raw filename (default: Image_001_001.raw).",
    )
    parser.add_argument(
        "--frames-to-truncate-per-session",
        type=int,
        default=500,
        help="Frames to drop from the start of EACH session before concatenation (default: 500).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output raw in out-dir.")
    parser.add_argument(
        "--copy-metadata-from",
        default="first",
        help="Where to copy Experiment.xml / ROIs.xaml from: 'first' or a path to a session folder/raw (default: first).",
    )

    # Motion correction args (subset; other DS_motion_correction args can be forwarded via --mc-args)
    parser.add_argument("--mc-output-subdir", default="output_final", help="Output subfolder under out-dir (default: output_final).")
    parser.add_argument("--mc-run-tag", default="concat_mc", help="Optional run tag under mc-output-subdir (default: concat_mc). Use '' to disable.")
    parser.add_argument(
        "--factor",
        nargs=2,
        type=int,
        default=(1, 1),
        metavar=("H", "W"),
        help="Spatial downsample factor (H W) for motion correction (default: 1 1).",
    )
    parser.add_argument(
        "--mc-frames-to-truncate",
        type=int,
        default=0,
        help="Frames to truncate for the concatenated raw inside DS_motion_correction (default: 0; truncation already handled per-session).",
    )
    parser.add_argument(
        "--no-overwrite-output-folder",
        dest="overwrite_output_folder",
        action="store_false",
        default=True,
        help="Do not clear the motion-correction output folder before running.",
    )
    parser.add_argument(
        "--skip-roi-detection",
        action="store_true",
        help="Skip Mask R-CNN ROI detection inside DS_motion_correction.",
    )
    parser.add_argument(
        "--mc-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Additional args forwarded to utils/DS_motion_correction.py (after a '--'). Example: --mc-args --preserve-sub1hz",
    )

    args = parser.parse_args()

    out_dir = os.path.abspath(str(args.out_dir))
    raw_paths = list(args.raw or [])
    if not raw_paths:
        raise ValueError("Provide at least one --raw path.")

    mc_run_tag = str(args.mc_run_tag)
    if mc_run_tag.strip() == "":
        mc_run_tag = None

    # 1) Concatenate raw sessions into out_dir/out_raw_name and copy metadata.
    from utils_concat.concat_raw_sessions import concat_raw_sessions

    print("[STEP] Concatenate raw sessions", flush=True)
    concat_raw_sessions(
        raw_paths=raw_paths,
        out_dir=out_dir,
        out_raw_name=str(args.out_raw_name),
        frames_to_truncate_per_session=int(args.frames_to_truncate_per_session),
        overwrite=bool(args.overwrite),
        copy_metadata_from=str(args.copy_metadata_from),
        experiment_xml_name="Experiment.xml",
        rois_xaml_name="ROIs.xaml",
    )

    concat_raw_path = str(Path(out_dir) / str(args.out_raw_name))
    if not os.path.exists(concat_raw_path):
        raise FileNotFoundError("Concatenated raw not found after concat step: {}".format(concat_raw_path))

    # 2) Run motion correction on the concatenated raw.
    print("[STEP] Motion correction on concatenated raw", flush=True)
    extra_mc_args = list(args.mc_args or [])
    if extra_mc_args and extra_mc_args[0] == "--":
        extra_mc_args = extra_mc_args[1:]

    _run_motion_correction(
        concat_raw_path=concat_raw_path,
        mc_output_subdir=str(args.mc_output_subdir),
        mc_run_tag=mc_run_tag,
        factor_hw=list(args.factor),
        frames_to_truncate=int(args.mc_frames_to_truncate),
        overwrite_output_folder=bool(args.overwrite_output_folder),
        skip_roi_detection=bool(args.skip_roi_detection),
        extra_mc_args=extra_mc_args,
    )


if __name__ == "__main__":
    main()

