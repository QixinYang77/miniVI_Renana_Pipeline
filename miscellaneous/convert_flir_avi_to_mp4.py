#!/usr/bin/env python3
"""Convert a single FLIR AVI video into a compressed MP4."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-video", required=True, help="Path to the source AVI file.")
    parser.add_argument("--output-video", required=True, help="Path to the destination MP4 file.")
    parser.add_argument(
        "--ffmpeg-binary",
        default="ffmpeg",
        help="ffmpeg executable to use on the cluster.",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=17,
        help="libx264 constant rate factor. Lower is higher quality.",
    )
    parser.add_argument(
        "--preset",
        default="slow",
        help="libx264 preset. Slower presets usually compress better at the same quality.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the destination MP4 if it already exists.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    ffmpeg_path = shutil.which(args.ffmpeg_binary)
    if ffmpeg_path is None:
        raise FileNotFoundError(
            f"Could not find ffmpeg executable: {args.ffmpeg_binary!r}. "
            "Install ffmpeg in the active conda environment or expose it on PATH."
        )

    input_video = Path(args.input_video).expanduser().resolve()
    output_video = Path(args.output_video).expanduser().resolve()
    output_video.parent.mkdir(parents=True, exist_ok=True)

    command = [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "info",
        "-y" if args.overwrite else "-n",
        "-i",
        str(input_video),
        "-map",
        "0:v:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        args.preset,
        "-crf",
        str(args.crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_video),
    ]

    print("Running:", " ".join(command))
    subprocess.run(command, check=True)
    print(f"Saved MP4 to: {output_video}")


if __name__ == "__main__":
    main()
