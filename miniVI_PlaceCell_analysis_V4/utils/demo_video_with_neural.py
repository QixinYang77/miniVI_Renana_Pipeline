"""Generate a demo video with behavior + neural traces side by side."""

import io
import json
import os

import cv2
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import sys
from tqdm import tqdm

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from preprocess_neural import detect_speed_epochs


class _NumpyCompatUnpickler(pickle.Unpickler):
    """Handle pickles saved with numpy 2.x on numpy 1.x."""
    def find_class(self, module, name):
        if module.startswith('numpy._core'):
            module = module.replace('numpy._core', 'numpy.core')
        return super().find_class(module, name)


def read_video_to_grayscale_array(video_path):
    """Read an MP4 video into a 3D NumPy array in grayscale."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    try:
        video_array = np.zeros((n_frames, height, width), dtype=np.uint8)
        preallocated = True
    except MemoryError:
        print("Video too large for preallocation; falling back to list accumulation.")
        frames = []
        preallocated = False

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if preallocated:
            if frame_idx < n_frames:
                video_array[frame_idx] = gray_frame
        else:
            frames.append(gray_frame)
        frame_idx += 1

    cap.release()

    if not preallocated:
        video_array = np.stack(frames, axis=0)
    if frame_idx < n_frames:
        video_array = video_array[:frame_idx]

    return video_array


def canvas_to_bgr(fig):
    """Render a Matplotlib figure to a NumPy BGR image (cv2-friendly)."""
    buf = io.BytesIO()
    fig.savefig(buf, format='raw', dpi=fig.dpi)
    buf.seek(0)
    w, h = fig.canvas.get_width_height()
    rgba = np.frombuffer(buf.getvalue(), dtype=np.uint8).reshape((h, w, 4))
    return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)


def remove_mad_outliers(data, time, threshold=20):
    """Remove outliers based on MAD score."""
    data = np.asarray(data, dtype=float)
    data_median = np.nanmedian(data)
    abs_dev = np.abs(data - data_median)
    mad = np.nanmedian(abs_dev)

    finite = np.isfinite(data)
    if mad == 0:
        keep = finite & (data == data_median)
    else:
        keep = finite & ((abs_dev / mad) <= threshold)

    return data[keep], time[keep]


def make_demo_video(data_folder, cells_display, behavior_video_path,
                    speed_threshold=3, output_fps=50, neural_step=10,
                    width_real=35.5, height_real=20.0, display_window=1000,
                    output_name='place_cell_activity_video_concat.mp4',
                    max_frames=None):
    """Generate a demo video with behavior + neural traces.

    Parameters
    ----------
    data_folder : str
        Session folder containing ``aligned_data.pkl``.
    cells_display : list[int]
        Which cell indices to display.
    behavior_video_path : str
        Full path to the behavior .mp4 video.
    speed_threshold : float
        Minimum speed (cm/s) for valid locomotion epochs.
    output_fps : int
        FPS of the output video.
    neural_step : int
        Step through neural samples (500 Hz / 10 = 50 fps).
    width_real, height_real : float
        Physical arena dimensions in cm.
    display_window : int
        Number of neural samples visible in the trace panel.
    output_name : str
        Output filename (written inside ``data_folder``).

    Returns
    -------
    str
        Path to the generated video.
    """
    # Load aligned data
    aligned_data_path = os.path.join(data_folder, 'aligned_data.pkl')
    if not os.path.exists(aligned_data_path):
        raise FileNotFoundError(f"aligned_data.pkl not found in {data_folder}")

    with open(aligned_data_path, 'rb') as f:
        loaded_data = _NumpyCompatUnpickler(f).load()
    print(f"Loaded aligned data from {aligned_data_path}")

    ts_neural = loaded_data['ts_neural']
    x_neural = loaded_data['x_neural']
    y_neural = loaded_data['y_neural']
    traces = loaded_data['traces']
    spikes = loaded_data['spikes']
    speed = loaded_data['speed']
    frame_rate = loaded_data['frame_rate']
    ts_behav = loaded_data['ts_behav']
    frames_behav = loaded_data['frames_behav']
    frame_width = loaded_data['frame_width']
    frame_height = loaded_data['frame_height']

    print(f"ts_neural range: {ts_neural[0]:.1f} - {ts_neural[-1]:.1f} ms, len={len(ts_neural)}")
    print(f"ts_behav range: {ts_behav[0]:.1f} - {ts_behav[-1]:.1f} ms, len={len(ts_behav)}")
    print(f"frames_behav range: {frames_behav[0]} - {frames_behav[-1]}, len={len(frames_behav)}")

    # Read behavior video
    print(f"Reading video from: {behavior_video_path}")
    behav_video = read_video_to_grayscale_array(behavior_video_path)
    print(f"behav_video shape: {behav_video.shape}")

    # Convert x_neural, y_neural from cm back to pixel coordinates
    bbox_path = os.path.join(data_folder, 'boundary_box.json')
    if os.path.exists(bbox_path):
        with open(bbox_path) as f:
            bbox = json.load(f)['boundary_box']
        x_min_px, y_min_px, x_max_px, y_max_px = bbox
        print(f"Loaded boundary box: x=[{x_min_px:.0f}, {x_max_px:.0f}], y=[{y_min_px:.0f}, {y_max_px:.0f}]")
    else:
        x_min_px, y_min_px = 0, 0
        x_max_px, y_max_px = float(frame_width), float(frame_height)
        print("No boundary_box.json found, using full frame as extent")

    x_neural_px = x_neural / width_real * (x_max_px - x_min_px) + x_min_px
    y_neural_px = y_neural / height_real * (y_max_px - y_min_px) + y_min_px

    # Precompute
    valid_indices = np.where(speed >= speed_threshold)[0]
    speed_epochs = detect_speed_epochs(speed, speed_threshold)

    spikes_valid = []
    xy_spikes = []
    for i in cells_display:
        spike_frames = spikes[i]
        spike_frames = spike_frames[np.isin(spike_frames, valid_indices)]
        spikes_valid.append(np.sort(spike_frames))
        xy_spikes.append(np.column_stack((x_neural_px[spike_frames], y_neural_px[spike_frames])))

    behav_inds = np.clip(np.searchsorted(ts_behav, ts_neural, side='left'), 0, len(ts_behav) - 1)

    # Switch to Agg for rendering
    prev_backend = matplotlib.get_backend()
    matplotlib.use('Agg')

    try:
        num_cells = len(cells_display)
        colors = cm.jet(np.linspace(0, 1, num_cells))

        fig, axes = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [2, 1]})
        ts_neural_display = ts_neural - ts_neural[0]

        # Right panel: static traces + events + speed epochs
        for k, i in enumerate(cells_display):
            trace = traces[i]
            trace_valid, ts_valid = remove_mad_outliers(trace, ts_neural, threshold=20)
            trace_interp = pd.Series(trace_valid, index=ts_valid).reindex(ts_neural).interpolate(method='nearest').bfill().ffill().to_numpy()
            tmin, tmax = np.nanmin(trace_interp), np.nanmax(trace_interp)
            if tmax != tmin:
                trace_norm = (trace_interp - tmin) / (tmax - tmin)
            else:
                trace_norm = np.zeros_like(trace_interp)
            axes[1].plot(ts_neural_display, trace_norm + k, alpha=0.7, linewidth=0.5, color=colors[k])
            if len(spikes[i]) > 0:
                axes[1].eventplot(ts_neural_display[spikes[i]], lineoffsets=k + 1 - 0.1,
                                  linelengths=0.1, colors='red', alpha=0.5)

        for start, end in speed_epochs:
            axes[1].axvspan(ts_neural_display[start], ts_neural_display[end], color='yellow', alpha=0.3)
        axes[1].axis('off')

        # Left panel: behavior video + mouse + spike positions
        # (re-created each frame in the loop below)
        axes[0].axis('off')

        vline = axes[1].axvline(0, color='k', linestyle='--')
        dt = np.median(np.diff(ts_neural_display))
        half_window = (display_window / 2.0) * dt
        fig.tight_layout()

        # Video writer
        fig.canvas.draw()
        canvas_width, canvas_height = fig.canvas.get_width_height()

        output_video_path = os.path.join(data_folder, output_name)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_video_path, fourcc, output_fps, (canvas_width, canvas_height))
        if not video_writer.isOpened():
            raise RuntimeError(f"Failed to open VideoWriter at: {output_video_path}")

        # Rendering loop
        print(f"Generating video... Output: {output_video_path}")
        frame_indices = range(0, len(ts_neural), neural_step)
        if max_frames is not None:
            frame_indices = list(frame_indices)[:max_frames]
        for frame_count, neural_ind in enumerate(tqdm(frame_indices, desc="Rendering frames")):
            behav_ind = behav_inds[neural_ind]
            frame_behav_temp = int(np.clip(frames_behav[behav_ind], 0, behav_video.shape[0] - 1))
            if frame_count < 5:
                print(f"  frame {frame_count}: neural_ind={neural_ind}, behav_ind={behav_ind}, "
                      f"frame_behav_temp={frame_behav_temp}, pixel_mean={behav_video[frame_behav_temp].mean():.1f}")

            # Redraw left panel from scratch each frame
            axes[0].cla()
            axes[0].imshow(behav_video[frame_behav_temp], cmap='gray', vmin=0, vmax=255,
                           aspect='auto', origin='lower')
            axes[0].plot(x_neural_px[neural_ind], y_neural_px[neural_ind], 'ro', markersize=10)
            for k, i in enumerate(cells_display):
                num_points = np.searchsorted(spikes_valid[k], neural_ind + 1)
                if num_points > 0:
                    axes[0].scatter(xy_spikes[k][:num_points, 0], xy_spikes[k][:num_points, 1],
                                    c=[colors[k]], s=10, alpha=0.8)
            axes[0].axis('off')

            current_time = ts_neural_display[neural_ind]
            vline.set_xdata([current_time])
            axes[1].set_xlim(current_time - half_window, current_time + half_window)

            frame_bgr = canvas_to_bgr(fig)
            if (frame_bgr.shape[1], frame_bgr.shape[0]) != (canvas_width, canvas_height):
                frame_bgr = cv2.resize(frame_bgr, (canvas_width, canvas_height), interpolation=cv2.INTER_AREA)
            video_writer.write(frame_bgr)

        video_writer.release()
        plt.close(fig)
        print(f"Video saved to: {output_video_path}")

    finally:
        # Restore previous backend
        try:
            matplotlib.use(prev_backend)
        except Exception:
            pass

    return output_video_path


def make_demo_video_concat(main_data_folder, cells_display,
                           speed_threshold=3, output_fps=20, neural_step=25,
                           width_real=35.5, height_real=20.0, display_window=1000,
                           output_name='place_cell_activity_video.mp4',
                           max_frames=None):
    """Generate a demo video from merged CS-refined data across multiple sessions.

    Loads ``merged_aligned_data_CS.pkl`` from *main_data_folder*, then grabs
    each session's behavior video (.avi) and per-session alignment info
    (``aligned_data.pkl``) from the session subfolders listed in the merged data.

    Parameters
    ----------
    main_data_folder : str
        Top-level folder containing ``merged_aligned_data_CS.pkl``.
    cells_display : list[int]
        Which cell indices to display.
    speed_threshold, output_fps, neural_step, width_real, height_real,
    display_window, output_name, max_frames :
        Same as :func:`make_demo_video`.

    Returns
    -------
    str
        Path to the generated video.
    """
    import glob as _glob

    # ------------------------------------------------------------------
    # 1. Load merged data
    # ------------------------------------------------------------------
    merged_path = os.path.join(main_data_folder, 'merged_aligned_data_CS.pkl')
    if not os.path.exists(merged_path):
        merged_path = os.path.join(main_data_folder, 'merged_aligned_data.pkl')
    with open(merged_path, 'rb') as f:
        data = _NumpyCompatUnpickler(f).load()
    print(f"Loaded merged data from {merged_path}")

    ts_neural = data['ts_neural']
    x_neural = data['x_neural']
    y_neural = data['y_neural']
    traces = data['traces_SNR_interpolated']
    spikes = data['all_spikes']
    speed = data['speed']
    frame_rate = data['frame_rate']
    subfolders_raw = data['subfolders']
    session_start_frames = data['session_start_frames']

    # Remap subfolder paths: the pickle may store absolute paths from a
    # different machine (e.g. /Volumes/adam-lab/...).  Re-anchor them
    # relative to main_data_folder so the code works on any mount point.
    subfolders = [
        os.path.join(main_data_folder, os.path.basename(sf))
        for sf in subfolders_raw
    ]

    n_total = len(ts_neural)
    n_sessions = len(subfolders)
    session_ends = list(session_start_frames[1:]) + [n_total]

    print(f"Total neural frames: {n_total}, sessions: {n_sessions}")
    for s in range(n_sessions):
        print(f"  Session {s}: frames [{session_start_frames[s]}:{session_ends[s]}] ({session_ends[s] - session_start_frames[s]} frames)")

    # ------------------------------------------------------------------
    # 2. Load per-session behavior videos + alignment info
    # ------------------------------------------------------------------
    print("Loading per-session behavior data and videos...")
    behav_videos = []
    session_video_frames = []  # per-session array: neural_local_idx -> video frame idx

    for s, sf in enumerate(subfolders):
        print(f"  Session {s}: {sf}")

        # Per-session aligned data for ts_behav, frames_behav
        aligned_path = os.path.join(sf, 'aligned_data.pkl')
        with open(aligned_path, 'rb') as f:
            sess_data = _NumpyCompatUnpickler(f).load()

        ts_behav_sess = sess_data['ts_behav']
        frames_behav_sess = sess_data['frames_behav']
        ts_neural_sess = sess_data['ts_neural']

        # Find and load behavior video (.avi)
        avi_files = sorted(_glob.glob(os.path.join(sf, '*.avi')))
        if not avi_files:
            raise FileNotFoundError(f"No .avi found in {sf}")

        print(f"    Video: {os.path.basename(avi_files[0])}")
        behav_video = read_video_to_grayscale_array(avi_files[0])
        behav_videos.append(behav_video)
        print(f"    Shape: {behav_video.shape}")

        # Map neural frames -> behavior video frames for this session
        n_sess = session_ends[s] - session_start_frames[s]
        behav_inds_sess = np.clip(
            np.searchsorted(ts_behav_sess, ts_neural_sess[:n_sess], side='left'),
            0, len(ts_behav_sess) - 1,
        )
        vid_frames_sess = np.clip(
            frames_behav_sess[behav_inds_sess].astype(int),
            0, behav_video.shape[0] - 1,
        )
        session_video_frames.append(vid_frames_sess)

    # ------------------------------------------------------------------
    # 3. Build merged lookup arrays
    # ------------------------------------------------------------------
    session_map = np.zeros(n_total, dtype=int)
    behav_frame_map = np.zeros(n_total, dtype=int)
    for s in range(n_sessions):
        start, end = session_start_frames[s], session_ends[s]
        session_map[start:end] = s
        behav_frame_map[start:end] = session_video_frames[s]

    # ------------------------------------------------------------------
    # 4. Convert positions cm -> pixels (using first session's boundary box)
    # ------------------------------------------------------------------
    bbox_path = os.path.join(subfolders[0], 'boundary_box.json')
    if os.path.exists(bbox_path):
        with open(bbox_path) as f:
            bbox = json.load(f)['boundary_box']
        x_min_px, y_min_px, x_max_px, y_max_px = bbox
        print(f"Boundary box: x=[{x_min_px:.0f}, {x_max_px:.0f}], y=[{y_min_px:.0f}, {y_max_px:.0f}]")
    else:
        x_min_px, y_min_px = 0, 0
        x_max_px = float(behav_videos[0].shape[2])
        y_max_px = float(behav_videos[0].shape[1])
        print("No boundary_box.json found, using full frame")

    x_neural_px = x_neural / width_real * (x_max_px - x_min_px) + x_min_px
    y_neural_px = y_neural / height_real * (y_max_px - y_min_px) + y_min_px

    # ------------------------------------------------------------------
    # 5. Precompute spike data
    # ------------------------------------------------------------------
    valid_indices = np.where(speed >= speed_threshold)[0]
    speed_epochs = detect_speed_epochs(speed, speed_threshold)

    spikes_valid = []
    xy_spikes = []
    for i in cells_display:
        spike_frames = spikes[i]
        spike_frames = spike_frames[np.isin(spike_frames, valid_indices)]
        spikes_valid.append(np.sort(spike_frames))
        xy_spikes.append(np.column_stack((x_neural_px[spike_frames], y_neural_px[spike_frames])))

    # ------------------------------------------------------------------
    # 6. Render video
    # ------------------------------------------------------------------
    prev_backend = matplotlib.get_backend()
    matplotlib.use('Agg')

    try:
        num_cells = len(cells_display)
        colors = cm.jet(np.linspace(0, 1, num_cells))

        fig, axes = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [2, 1]})
        ts_neural_display = ts_neural - ts_neural[0]

        # Right panel: static traces + events + speed epochs
        for k, i in enumerate(cells_display):
            trace = traces[i]
            trace_valid, ts_valid = remove_mad_outliers(trace, ts_neural, threshold=20)
            trace_interp = pd.Series(trace_valid, index=ts_valid).reindex(ts_neural).interpolate(method='nearest').bfill().ffill().to_numpy()
            tmin, tmax = np.nanmin(trace_interp), np.nanmax(trace_interp)
            if tmax != tmin:
                trace_norm = (trace_interp - tmin) / (tmax - tmin)
            else:
                trace_norm = np.zeros_like(trace_interp)
            axes[1].plot(ts_neural_display, trace_norm + k, alpha=0.7, linewidth=0.5, color=colors[k])
            if len(spikes[i]) > 0:
                axes[1].eventplot(ts_neural_display[spikes[i]], lineoffsets=k + 1 - 0.1,
                                  linelengths=0.1, colors='red', alpha=0.5)

        for start, end in speed_epochs:
            axes[1].axvspan(ts_neural_display[start], ts_neural_display[end], color='yellow', alpha=0.3)
        axes[1].axis('off')
        axes[0].axis('off')

        vline = axes[1].axvline(0, color='k', linestyle='--')
        dt = np.median(np.diff(ts_neural_display))
        half_window = (display_window / 2.0) * dt
        fig.tight_layout()

        # Video writer
        fig.canvas.draw()
        canvas_width, canvas_height = fig.canvas.get_width_height()

        output_video_path = os.path.join(main_data_folder, output_name)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_video_path, fourcc, output_fps, (canvas_width, canvas_height))
        if not video_writer.isOpened():
            raise RuntimeError(f"Failed to open VideoWriter at: {output_video_path}")

        # Rendering loop
        print(f"Generating video... Output: {output_video_path}")
        frame_indices = range(0, n_total, neural_step)
        if max_frames is not None:
            frame_indices = list(frame_indices)[:max_frames]

        prev_session = -1
        for frame_count, neural_ind in enumerate(tqdm(frame_indices, desc="Rendering frames")):
            s = session_map[neural_ind]
            vid_frame_idx = behav_frame_map[neural_ind]

            if s != prev_session:
                print(f"  Switched to session {s} at neural_ind={neural_ind}")
                prev_session = s

            # Redraw left panel
            axes[0].cla()
            axes[0].imshow(behav_videos[s][vid_frame_idx], cmap='gray', vmin=0, vmax=255,
                           aspect='auto', origin='lower')
            axes[0].plot(x_neural_px[neural_ind], y_neural_px[neural_ind], 'ro', markersize=10)
            for k, i in enumerate(cells_display):
                num_points = np.searchsorted(spikes_valid[k], neural_ind + 1)
                if num_points > 0:
                    axes[0].scatter(xy_spikes[k][:num_points, 0], xy_spikes[k][:num_points, 1],
                                    c=[colors[k]], s=10, alpha=0.8)
            axes[0].axis('off')

            current_time = ts_neural_display[neural_ind]
            vline.set_xdata([current_time])
            axes[1].set_xlim(current_time - half_window, current_time + half_window)

            frame_bgr = canvas_to_bgr(fig)
            if (frame_bgr.shape[1], frame_bgr.shape[0]) != (canvas_width, canvas_height):
                frame_bgr = cv2.resize(frame_bgr, (canvas_width, canvas_height), interpolation=cv2.INTER_AREA)
            video_writer.write(frame_bgr)

        video_writer.release()
        plt.close(fig)
        print(f"Video saved to: {output_video_path}")

    finally:
        try:
            matplotlib.use(prev_backend)
        except Exception:
            pass

    return output_video_path
