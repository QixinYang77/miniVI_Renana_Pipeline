"""
Standalone script to generate demo videos with behavior + neural traces.

Usage:
    python make_demo_video_job.py <data_folder> <behavior_video_path>
        [--cells 1 2 3 4 5 6]
        [--speed-threshold 3] [--output-fps 50] [--neural-step 10]
        [--width-real 35.5] [--height-real 20.0] [--display-window 1000]
        [--output-name place_cell_activity_video.mp4]
"""

import argparse
import sys
import os

# Add parent dir so we can import from utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.demo_video_with_neural import make_demo_video


def main():
    parser = argparse.ArgumentParser(description="Generate demo video with behavior + neural traces.")
    parser.add_argument("data_folder", help="Session folder containing aligned_data.pkl")
    parser.add_argument("behavior_video_path", help="Path to the behavior .avi video")
    parser.add_argument("--cells", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6],
                        help="Cell indices to display (default: 1 2 3 4 5 6)")
    parser.add_argument("--speed-threshold", type=float, default=3, help="Min speed cm/s (default: 3)")
    parser.add_argument("--output-fps", type=int, default=50, help="Output video FPS (default: 50)")
    parser.add_argument("--neural-step", type=int, default=10, help="Neural downsample step (default: 10)")
    parser.add_argument("--width-real", type=float, default=35.5, help="Arena width in cm (default: 35.5)")
    parser.add_argument("--height-real", type=float, default=20.0, help="Arena height in cm (default: 20.0)")
    parser.add_argument("--display-window", type=int, default=1000, help="Trace window in samples (default: 1000)")
    parser.add_argument("--output-name", type=str, default="place_cell_activity_video.mp4",
                        help="Output filename (default: place_cell_activity_video.mp4)")
    args = parser.parse_args()

    print(f"Data folder: {args.data_folder}")
    print(f"Behavior video: {args.behavior_video_path}")
    print(f"Cells: {args.cells}")

    make_demo_video(
        data_folder=args.data_folder,
        cells_display=args.cells,
        behavior_video_path=args.behavior_video_path,
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
