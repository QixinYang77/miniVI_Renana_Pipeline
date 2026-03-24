"""
Standalone script to generate a concat demo video from merged CS-refined data.

Uses merged_aligned_data_CS.pkl and grabs behavior videos from each session subfolder.

Usage:
    python make_demo_video_concat_job.py <main_data_folder>
        [--cells 0 2 3 4]
        [--speed-threshold 3] [--output-fps 20] [--neural-step 25]
        [--width-real 35.5] [--height-real 20.0] [--display-window 1000]
        [--output-name place_cell_activity_video.mp4]
"""

import argparse
import sys
import os

# Add parent dir so we can import from utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.demo_video_with_neural import make_demo_video_concat


def main():
    parser = argparse.ArgumentParser(description="Generate concat demo video from merged CS-refined data.")
    parser.add_argument("main_data_folder", help="Top-level folder containing merged_aligned_data_CS.pkl")
    parser.add_argument("--cells", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6],
                        help="Cell indices to display (default: 1 2 3 4 5 6)")
    parser.add_argument("--speed-threshold", type=float, default=3, help="Min speed cm/s (default: 3)")
    parser.add_argument("--output-fps", type=int, default=20, help="Output video FPS (default: 20)")
    parser.add_argument("--neural-step", type=int, default=25, help="Neural downsample step (default: 25)")
    parser.add_argument("--width-real", type=float, default=35.5, help="Arena width in cm (default: 35.5)")
    parser.add_argument("--height-real", type=float, default=20.0, help="Arena height in cm (default: 20.0)")
    parser.add_argument("--display-window", type=int, default=1000, help="Trace window in samples (default: 1000)")
    parser.add_argument("--output-name", type=str, default="place_cell_activity_video.mp4",
                        help="Output filename (default: place_cell_activity_video.mp4)")
    args = parser.parse_args()

    print(f"Main data folder: {args.main_data_folder}")
    print(f"Cells: {args.cells}")

    make_demo_video_concat(
        main_data_folder=args.main_data_folder,
        cells_display=args.cells,
        speed_threshold=args.speed_threshold,
        output_fps=args.output_fps,
        neural_step=args.neural_step,
        width_real=args.width_real,
        height_real=args.height_real,
        display_window=args.display_window,
        output_name=args.output_name,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
