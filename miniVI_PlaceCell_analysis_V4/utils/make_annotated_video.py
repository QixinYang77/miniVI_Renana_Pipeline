"""
Standalone script to generate head-direction annotated behavior videos.

Usage:
    python make_annotated_video.py <data_folder> [<data_folder2> ...]
        [--thr-head 0.6] [--thr-nose 0.6] [--filter-kernel 1]
        [--short-gap 5] [--interpolate-nan] [--include-body]
"""

import argparse
import sys
import os
import glob

# Add parent dir so we can import from utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.preprocess_behavior import read_DLC_csv_file, refine_headpositions, make_anotated_behav_video


def main():
    parser = argparse.ArgumentParser(description="Generate head-direction annotated behavior videos.")
    parser.add_argument("data_folders", nargs="+", help="One or more session folders containing DLC csv + .avi video")
    parser.add_argument("--thr-head", type=float, default=0.6, help="Likelihood threshold for head (default: 0.6)")
    parser.add_argument("--thr-nose", type=float, default=0.6, help="Likelihood threshold for nose (default: 0.6)")
    parser.add_argument("--filter-kernel", type=int, default=1, help="Median filter kernel size (default: 1)")
    parser.add_argument("--short-gap", type=int, default=5, help="Max frames for short-gap interpolation (default: 5)")
    parser.add_argument("--interpolate-nan", action="store_true", default=False,
                        help="Interpolate remaining NaN positions (default: False)")
    parser.add_argument("--include-body", action="store_true", default=False,
                        help="Include body parts in fallback (default: False)")
    args = parser.parse_args()

    for data_folder in args.data_folders:
        print(f"\n{'='*60}")
        print(f"Processing: {data_folder}")
        print(f"{'='*60}")

        df = read_DLC_csv_file(data_folder)

        # Check if boundary_box.json exists to decide whether to pass data_folder
        boundary_file = os.path.join(data_folder, "boundary_box.json")
        kw = {}
        if os.path.exists(boundary_file):
            kw["data_folder"] = data_folder

        result = refine_headpositions(
            df,
            thr_likelihood_head=args.thr_head,
            thr_likelihood_nose=args.thr_nose,
            filter_kernel_size=args.filter_kernel,
            short_gap_max_frames=args.short_gap,
            interpolate_remaining_nan=args.interpolate_nan,
            include_body=args.include_body,
            **kw,
        )

        # Unpack depending on whether data_folder was passed
        if isinstance(result, tuple):
            df_refined = result[0]
        else:
            df_refined = result

        video_files = glob.glob(os.path.join(data_folder, "*.avi"))
        if not video_files:
            print(f"[WARN] No .avi file found in {data_folder}, skipping.")
            continue

        video_file = video_files[0]
        make_anotated_behav_video(video_file, df_refined)

    print("\nDone.")


if __name__ == "__main__":
    main()
