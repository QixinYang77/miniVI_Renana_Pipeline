import numpy as np
import re
import pprint
import matplotlib.pyplot as plt
import os

import h5py
from scipy.interpolate import interp1d
import pandas as pd
import matplotlib.cm as cm
import numpy.ma as ma
from tqdm import tqdm
import cv2
import pickle
import glob
import json
from scipy.signal import medfilt
import math

def get_behav_video_dimensions(data_folder, width_real=35.5, height_real=20, sample_frame_idx=0):
    DLC_video = [f for f in os.listdir(data_folder) if f.endswith('.avi')][0]
    video_path = os.path.join(data_folder, DLC_video)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video file: {video_path}")

    frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps    = cap.get(cv2.CAP_PROP_FPS)
    video_n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # read one frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(sample_frame_idx))
    ret, frame_bgr = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Could not read frame {sample_frame_idx} from {video_path}")

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    print(f"Video dimensions: {frame_width}x{frame_height}, FPS: {video_fps}, Total frames: {video_n_frames}")
    ratio_width  = frame_width  / width_real
    ratio_height = frame_height / height_real
    print(f"pixel to cm ratio={ratio_width:.2f} (width), {ratio_height:.2f} (height)")

    return frame_width, frame_height, ratio_width, ratio_height, video_fps, video_n_frames, frame_rgb

def read_DLC_csv_file(data_folder):
    DLC_csv_files = [f for f in os.listdir(data_folder) if f.startswith('Flir') and f.endswith('.csv')]
    # chose the latest file that contains "filtered" in its name
    filtered_files = [f for f in DLC_csv_files if "filtered" in f]
    if filtered_files:
        print(f"Using filtered DLC file: {max(filtered_files)}")
        DLC_csv_file = os.path.join(data_folder, max(filtered_files))
        df = pd.read_csv(DLC_csv_file, header=[0, 1, 2])
        return df
    else:
        print("No filtered DLC file found. Using unfiltered DLC file.")


def refine_headpositions(df, thr_likelihood_head=0.5, thr_likelihood_nose=0.5, filter_kernel_size=3,
                         data_folder=None, width_real=35.5, height_real=20,
                         short_gap_max_frames=3, interpolate_remaining_nan=False,
                         include_body=False):
    # Load boundary box if data_folder is provided
    boundary_box = None
    if data_folder is not None:
        boundary_file = os.path.join(data_folder, "boundary_box.json")
        with open(boundary_file, "r", encoding="utf-8") as f:
            boundary_data = json.load(f)
        boundary_box = tuple(boundary_data.get("boundary_box", ()))
        if len(boundary_box) != 4:
            print(f"Warning: boundary_box.json in {data_folder} is malformed; skipping clipping.")
            boundary_box = None

    # Identify scorer (first level of columns that is not "scorer")
    scorer = [c[0] for c in df.columns if c[0] != "scorer"][0]

    # Candidate body parts for refined head positions
    bodyparts = ['Nose', 'Head_pre', 'Head', 'Head_post', 'Neck']
    if include_body:
        bodyparts += ['Body_pre', 'Body', 'Body_post']

    # Priority groups for fallback search
    priority_groups = [
        ['Head'],                  # highest priority
        ['Head_pre', 'Head_post'], # around head
        ['Nose', 'Neck'],          # tip or base
    ]
    if include_body:
        priority_groups += [
            ['Body_pre'],              # just after neck
            ['Body'],                  # mid-body
            ['Body_post'],             # rear body
        ]

    # Build x, y, likelihood matrices: shape (n_frames, n_bodyparts)
    lik_mat = np.stack(
        [df[(scorer, bp, 'likelihood')].to_numpy() for bp in bodyparts],
        axis=1
    )
    x_mat = np.stack(
        [df[(scorer, bp, 'x')].to_numpy() for bp in bodyparts],
        axis=1
    )
    y_mat = np.stack(
        [df[(scorer, bp, 'y')].to_numpy() for bp in bodyparts],
        axis=1
    )

    n_frames, n_bp = lik_mat.shape
    idx_head = bodyparts.index('Head')
    idx_nose = bodyparts.index('Nose')

    # ==============================
    # 1b. Short-gap interpolation: for each body part, if it drops below
    #     threshold for <= short_gap_max_frames, linearly interpolate x/y
    #     through the gap and restore likelihood so the fallback logic
    #     can still use this body part.
    # ==============================
    thr_min = min(thr_likelihood_head, thr_likelihood_nose)
    for bp_i in range(n_bp):
        lik_col = lik_mat[:, bp_i]
        good = lik_col >= thr_min
        # Find runs of bad frames
        t = 0
        while t < n_frames:
            if not good[t]:
                # Start of a bad run
                gap_start = t
                while t < n_frames and not good[t]:
                    t += 1
                gap_end = t  # exclusive
                gap_len = gap_end - gap_start
                # Only interpolate if gap is short and bounded by good frames on both sides
                if gap_len <= short_gap_max_frames and gap_start > 0 and gap_end < n_frames:
                    for coord_mat in (x_mat, y_mat):
                        v_before = coord_mat[gap_start - 1, bp_i]
                        v_after  = coord_mat[gap_end, bp_i]
                        for k in range(gap_len):
                            alpha = (k + 1) / (gap_len + 1)
                            coord_mat[gap_start + k, bp_i] = v_before + alpha * (v_after - v_before)
                    # Set likelihood to threshold so these frames pass the check
                    lik_mat[gap_start:gap_end, bp_i] = thr_min
            else:
                t += 1

    def pick_index_for_frame(lik_row, thr, forbidden_idx=None):
        """
        Fallback chooser:
        - traverse priority_groups
        - within each group, keep indices with likelihood >= thr
        - among them choose the one with highest likelihood
        - optionally exclude forbidden_idx (e.g. best when picking second-best)
        Returns index 0..n_bp-1 or None.
        """
        for group in priority_groups:
            idxs = [bodyparts.index(bp) for bp in group]
            if forbidden_idx is not None:
                idxs = [i for i in idxs if i != forbidden_idx]
            if not idxs:
                continue

            liks = lik_row[idxs]
            mask = liks >= thr
            if np.any(mask):
                cand_idxs = np.array(idxs)[mask]
                best_local = cand_idxs[np.argmax(lik_row[cand_idxs])]
                return best_local
        return None


    # ==============================
    # 2. Best & second-best positions with new logic
    # ==============================
    best_idx = np.full(n_frames, -1, dtype=int)
    second_idx = np.full(n_frames, -1, dtype=int)

    for t in range(n_frames):
        lik_row = lik_mat[t]

        # ----- BEST -----
        if lik_row[idx_head] >= thr_likelihood_head:
            # Head is good enough
            best_idx[t] = idx_head
        else:
            # Fallback search with threshold 0.5
            idx = pick_index_for_frame(lik_row, thr=thr_likelihood_head, forbidden_idx=None)
            best_idx[t] = idx if idx is not None else -1

        # ----- SECOND-BEST -----
        # Prefer Nose if good enough and not already used as best
        if (lik_row[idx_nose] >= thr_likelihood_nose) and (best_idx[t] != idx_nose):
            second_idx[t] = idx_nose
        else:
            forbid = best_idx[t] if best_idx[t] != -1 else None
            idx2 = pick_index_for_frame(lik_row, thr=thr_likelihood_nose, forbidden_idx=forbid)
            second_idx[t] = idx2 if idx2 is not None else -1


    n_best_nan = np.sum(best_idx == -1)
    n_second_nan = np.sum(second_idx == -1)
    print(f"Frames with no valid best position: {n_best_nan}/{n_frames} ({100*n_best_nan/n_frames:.1f}%)")
    print(f"Frames with no valid second position: {n_second_nan}/{n_frames} ({100*n_second_nan/n_frames:.1f}%)")

    # Convert indices to coordinates & labels (raw, possibly NaN)
    best_x = np.full(n_frames, np.nan)
    best_y = np.full(n_frames, np.nan)
    best_label = np.full(n_frames, None, dtype=object)

    mask_best = best_idx != -1
    best_x[mask_best] = x_mat[mask_best, best_idx[mask_best]]
    best_y[mask_best] = y_mat[mask_best, best_idx[mask_best]]
    best_label[mask_best] = [bodyparts[i] for i in best_idx[mask_best]]

    second_x = np.full(n_frames, np.nan)
    second_y = np.full(n_frames, np.nan)
    second_label = np.full(n_frames, None, dtype=object)

    mask_second = second_idx != -1
    second_x[mask_second] = x_mat[mask_second, second_idx[mask_second]]
    second_y[mask_second] = y_mat[mask_second, second_idx[mask_second]]
    second_label[mask_second] = [bodyparts[i] for i in second_idx[mask_second]]

    # ==============================
    # 3. Decide front vs back (anterior–posterior ordering)
    # ==============================
    front_order = {
        'Body_post': -3,
        'Body': -2,
        'Body_pre': -1,
        'Neck': 0,
        'Head_post': 1,
        'Head': 2,
        'Head_pre': 3,
        'Nose': 4,
    }

    front_x = np.full(n_frames, np.nan)
    front_y = np.full(n_frames, np.nan)
    back_x  = np.full(n_frames, np.nan)
    back_y  = np.full(n_frames, np.nan)
    front_label = np.full(n_frames, None, dtype=object)
    back_label  = np.full(n_frames, None, dtype=object)

    for t in range(n_frames):
        lbl1 = best_label[t]
        lbl2 = second_label[t]
        x1, y1 = best_x[t], best_y[t]
        x2, y2 = second_x[t], second_y[t]

        if (lbl1 is None) or (lbl2 is None):
            continue
        if (lbl1 not in front_order) or (lbl2 not in front_order):
            continue
        if np.isnan(x1) or np.isnan(y1) or np.isnan(x2) or np.isnan(y2):
            continue

        s1 = front_order[lbl1]
        s2 = front_order[lbl2]

        if s1 == s2:
            # ambiguous, keep NaN
            continue

        if s1 > s2:
            # best point is more frontal
            front_x[t], front_y[t] = x1, y1
            back_x[t],  back_y[t]  = x2, y2
            front_label[t], back_label[t] = lbl1, lbl2
        else:
            # second-best point is more frontal
            front_x[t], front_y[t] = x2, y2
            back_x[t],  back_y[t]  = x1, y1
            front_label[t], back_label[t] = lbl2, lbl1


    # ==============================
    # 4. NaN interpolation + median filter smoothing (same as before)
    # ==============================
    def interpolate_nan_1d(arr):
        """Linear interpolate NaNs in 1D, filling both ends."""
        s = pd.Series(arr)
        return s.interpolate(method='linear', limit_direction='both').to_numpy()

    # interpolate remaining NaN (frames where all fallback failed)
    if interpolate_remaining_nan:
        best_x_f   = interpolate_nan_1d(best_x)
        best_y_f   = interpolate_nan_1d(best_y)
        second_x_f = interpolate_nan_1d(second_x)
        second_y_f = interpolate_nan_1d(second_y)

        front_x_f  = interpolate_nan_1d(front_x)
        front_y_f  = interpolate_nan_1d(front_y)
        back_x_f   = interpolate_nan_1d(back_x)
        back_y_f   = interpolate_nan_1d(back_y)
    else:
        best_x_f   = best_x.copy()
        best_y_f   = best_y.copy()
        second_x_f = second_x.copy()
        second_y_f = second_y.copy()

        front_x_f  = front_x.copy()
        front_y_f  = front_y.copy()
        back_x_f   = back_x.copy()
        back_y_f   = back_y.copy()

    # Clip interpolated positions to arena boundary
    if boundary_box is not None:
        bx_min, by_min, bx_max, by_max = boundary_box
        best_x_f   = np.clip(best_x_f,   bx_min, bx_max)
        best_y_f   = np.clip(best_y_f,   by_min, by_max)
        second_x_f = np.clip(second_x_f, bx_min, bx_max)
        second_y_f = np.clip(second_y_f, by_min, by_max)
        front_x_f  = np.clip(front_x_f,  bx_min, bx_max)
        front_y_f  = np.clip(front_y_f,  by_min, by_max)
        back_x_f   = np.clip(back_x_f,   bx_min, bx_max)
        back_y_f   = np.clip(back_y_f,   by_min, by_max)

    # median filter (slight smoothing), preserving NaN positions
    k = filter_kernel_size  # odd kernel size

    def medfilt_preserve_nan(arr, kernel_size):
        nan_mask = np.isnan(arr)
        if np.all(nan_mask):
            return arr.copy()
        # Fill NaN temporarily for medfilt, then restore
        arr_filled = arr.copy()
        arr_filled[nan_mask] = 0
        result = medfilt(arr_filled, kernel_size=kernel_size)
        result[nan_mask] = np.nan
        return result

    best_x_s   = medfilt_preserve_nan(best_x_f,   kernel_size=k)
    best_y_s   = medfilt_preserve_nan(best_y_f,   kernel_size=k)
    second_x_s = medfilt_preserve_nan(second_x_f, kernel_size=k)
    second_y_s = medfilt_preserve_nan(second_y_f, kernel_size=k)

    front_x_s  = medfilt_preserve_nan(front_x_f,  kernel_size=k)
    front_y_s  = medfilt_preserve_nan(front_y_f,  kernel_size=k)
    back_x_s   = medfilt_preserve_nan(back_x_f,   kernel_size=k)
    back_y_s   = medfilt_preserve_nan(back_y_f,   kernel_size=k)

    # ==============================
    # 5. Head direction from smoothed front/back (back → front)
    # ==============================
    dx_s = front_x_s - back_x_s
    dy_s = front_y_s - back_y_s          # image coordinates: y increases downward

    head_dir_rad_s = np.arctan2(dy_s, dx_s)   # back → front
    head_dir_deg_s = np.rad2deg(head_dir_rad_s)

    # ==============================
    # 6. Write everything back into df (flat columns)
    # ==============================
    df['head_best_x'] = best_x_s
    df['head_best_y'] = best_y_s
    df['head_best_label'] = best_label    # labels not smoothed, but OK

    df['head_second_x'] = second_x_s
    df['head_second_y'] = second_y_s
    df['head_second_label'] = second_label

    df['head_front_x'] = front_x_s
    df['head_front_y'] = front_y_s
    df['head_back_x']  = back_x_s
    df['head_back_y']  = back_y_s
    df['head_front_label'] = front_label
    df['head_back_label']  = back_label

    df['head_direction_rad'] = head_dir_rad_s
    df['head_direction_deg'] = head_dir_deg_s

    if boundary_box is not None:
        x_cm, y_cm = normalize_positions(best_x_s, best_y_s, boundary_box, width_real, height_real)
        positions = np.column_stack((x_cm, y_cm))
        return df, x_cm, y_cm, positions

    return df

def make_anotated_behav_video(video_file, df):
    folder_path = os.path.dirname(video_file)
    output_mp4 = os.path.join(folder_path, "output_with_head_direction2.mp4")
    head_angle = df['head_direction_rad'].to_numpy()
    best_x = df['head_best_x'].to_numpy()
    best_y = df['head_best_y'].to_numpy()
    # 2. Load video
    # --------------------------------------------------------
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_file}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames_tracking = len(best_x)
    n_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video frames: {n_frames_video}, Tracking frames: {n_frames_tracking}")

    # --------------------------------------------------------
    # 3. Prepare MP4 writer
    # --------------------------------------------------------
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_mp4, fourcc, fps, (width, height))

    # --------------------------------------------------------
    # 4. Iterate through frames and draw overlays
    # --------------------------------------------------------
    arrow_length = 40    # pixels; adjust if needed
    dot_radius = 4
    dot_color = (0, 255, 0)      # green
    arrow_color = (0, 0, 255)    # red
    thickness = 2

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx >= n_frames_tracking:
            break

        x = best_x[frame_idx]
        y = best_y[frame_idx]
        ang = head_angle[frame_idx]

        # only draw if valid coordinates
        if not np.isnan(x) and not np.isnan(y):
            cx, cy = int(x), int(y)

            # Draw best head position
            cv2.circle(frame, (cx, cy), dot_radius, dot_color, -1)

            # Draw arrow (back → front)
            if not np.isnan(ang):
                dx = arrow_length * math.cos(ang)
                dy = arrow_length * math.sin(ang)

                # NOTE: y increases downward in images → keep sign as is
                end_x = int(cx + dx)
                end_y = int(cy + dy)

                cv2.arrowedLine(
                    frame,
                    (cx, cy),
                    (end_x, end_y),
                    arrow_color,
                    thickness,
                    tipLength=0.3
                )

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()

    print(f"Saved MP4 with head overlay → {output_mp4}")


def normalize_positions(x, y, boundary, width_real, height_real):
    """
    Normalize positions to real-world coordinates (cm).
    Maps boundary box to [0, width_real] x [0, height_real].
    """
    x_min, y_min, x_max, y_max = boundary
    
    # First clip to boundary
    x_clipped = np.clip(x, x_min, x_max)
    y_clipped = np.clip(y, y_min, y_max)
    
    # Normalize to [0, 1] within boundary
    x_norm = (x_clipped - x_min) / (x_max - x_min)
    y_norm = (y_clipped - y_min) / (y_max - y_min)
    
    # Scale to real-world coordinates
    x_cm = x_norm * width_real
    y_cm = y_norm * height_real
    
    return x_cm, y_cm

def get_normalized_head_positions(df, data_folder, width_real=35.5, height_real=20):
    x = df['head_best_x'].to_numpy()
    y = df['head_best_y'].to_numpy()

    boundary_file = os.path.join(data_folder, "boundary_box.json")

    with open(boundary_file, "r", encoding="utf-8") as f:
        boundary_data = json.load(f)
    boundary_box = tuple(boundary_data.get("boundary_box", ()))
    if len(boundary_box) != 4:
        print(f"Warning: boundary_box.json in {data_folder} is malformed; returning raw pixel positions.")
    x_cm, y_cm = normalize_positions(x, y, boundary_box, width_real, height_real)
    positions = np.column_stack((x_cm, y_cm))

    return x_cm, y_cm, positions

