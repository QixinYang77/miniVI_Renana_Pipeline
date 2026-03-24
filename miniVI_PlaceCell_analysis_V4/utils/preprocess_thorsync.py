from scipy.signal import find_peaks
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

def detect_flir_events(flir_signal, time_vector):
    """
    Detects falling edge events in a FLIR signal.

    Args:
        flir_signal (np.array): The voltage signal for the behavior camera (FLIR).
        time_vector (np.array): The time vector corresponding to the signal.

    Returns:
        tuple: A tuple containing:
            - flir_frame_times (np.array): Timestamps of the FLIR frames (falling edges).
            - flir_falling_edge_indices (np.array): The indices of the detected falling edges.
    """
    # We look for a drop from a high voltage to a low voltage.
    # A simple threshold can distinguish the two states.
    flir_threshold = 1.  # Based on your plot, ~3.0 is high, <0.5 is low.
    
    # Find indices where the signal goes from high to low.
    flir_falling_edge_indices = np.where((flir_signal[:-1] > flir_threshold) & (flir_signal[1:] < flir_threshold))[0]
    
    # Get the corresponding time for each falling edge. We add 1 to the index
    # to get the time at the *end* of the step, as requested.
    flir_frame_times = time_vector[flir_falling_edge_indices + 1]
    
    return flir_frame_times, flir_falling_edge_indices


def detect_frameout2_events(frameout2_signal, time_vector):
    """
    Detects peak events in a FrameOut2 signal.

    Args:
        frameout2_signal (np.array): The voltage signal for the sCMOS camera (FrameOut2).
        time_vector (np.array): The time vector corresponding to the signal.

    Returns:
        tuple: A tuple containing:
            - frameout2_peak_times (np.array): Timestamps of the FrameOut2 peaks.
            - frameout2_peak_indices (np.array): The indices of the detected peaks.
    """
    # We use SciPy's find_peaks function. It's robust and easy to use.
    # We'll set a height threshold to make sure we only get the real peaks.
    frameout2_threshold = 0.5 # Based on your plot, peaks are at 1.0.

    frameout2_peak_indices, _ = find_peaks(frameout2_signal, height=frameout2_threshold)
    
    # Get the corresponding time for each peak
    frameout2_peak_times = time_vector[frameout2_peak_indices]

    return frameout2_peak_times, frameout2_peak_indices

def read_thorsync_file(data_folder, plotflag=False, load_all=False):
    """
    Optimized version of read_thorsync_file.
    
    Args:
        data_folder (str): Path to the data folder.
        plotflag (bool): If True, plots the synchronization signals.
        load_all (bool): If True, loads all auxiliary channels (Dig488, LEDdig, etc.). 
                         If False (default), loads only alignment channels (FLIR, FrameOut).
    """
    TS_folders = [f for f in os.listdir(data_folder) if f.startswith('TS')]
    if not TS_folders:
        raise FileNotFoundError("No TS folder found in the data folder.")
    TS_folder = os.path.join(data_folder, TS_folders[0])
    
    TS_files = [f for f in os.listdir(TS_folder) if f.endswith('.h5')]
    if not TS_files:
        raise FileNotFoundError("No .h5 file found in the TS folder.")
    TS_file = os.path.join(TS_folder, TS_files[0])

    with h5py.File(TS_file, 'r') as f:
        # print(list(f.keys())) # Optional: Uncomment to debug keys

        # 1. Load ONLY essential alignment channels first
        FLIR = np.array(f['AI']['FLIR']).squeeze()
        
        # ThorSync sampling rate
        thorsync_sr = 30000 
        time_vector_TS = np.arange(len(FLIR)) / thorsync_sr * 1000

        # 2. Handle FrameOut / FrameCounter logic
        if 'CI' in f and 'FrameCounter' in f['CI']:
            # fast path: Counter Input exists
            FrameCounter = np.array(f['CI']['FrameCounter']).squeeze()
            FrameOut2 = np.diff(FrameCounter)
            FrameOut2 = np.insert(FrameOut2, 0, 0)
            FrameOut = np.array(f['DI']['FrameOut']).squeeze() # Load this anyway for completeness
        else:
            print("CI group not found. Reconstructing FrameOut2 from FrameOut...")
            FrameOut = np.array(f['DI']['FrameOut']).squeeze()
            
            frame_threshold = 1.0  # Threshold to detect FrameOut pulses !!! Change this if you see issues !!!
            
            # OPTIMIZATION: Fast rising edge detection (0 -> 1) using diff
            # This is much faster than find_peaks for clean digital signals
            # Indices where signal goes from low to high
            frame_out_edges = np.where(np.diff(FrameOut) > frame_threshold)[0] + 1
            
            if len(frame_out_edges) > 1:
                # Calculate typical frame interval (dframe) directly from edges
                dframe_samples = np.round(np.min(np.unique(np.diff(frame_out_edges))))
                
                ind_start = np.where(FrameOut > 0.5)[0][0]
                ind_end = len(FLIR) - 100
                
                FrameOut2 = np.zeros_like(FrameOut)
                # Vectorized assignment is faster than loops
                indices_to_set = np.arange(ind_start, ind_end + 1, int(dframe_samples))
                # Ensure we don't go out of bounds
                indices_to_set = indices_to_set[indices_to_set < len(FrameOut2)]
                FrameOut2[indices_to_set] = 1

                plt.figure()
                plt.plot(time_vector_TS, FrameOut)
                plt.scatter(time_vector_TS[indices_to_set], FrameOut2[indices_to_set], color='red', label='Reconstructed FrameOut2 Peaks')

            else:
                print("Warning: Not enough FrameOut events to reconstruct FrameOut2.")
                FrameOut2 = np.zeros_like(FrameOut)

        # 3. OPTIMIZATION: Fast Event Detection (Replace detect_flir_events & detect_frameout2_events)
        
        # FLIR: Falling edge detection (High -> Low)
        # Assuming threshold ~1.0V
        flir_falling_edge_indices = np.where((FLIR[:-1] > 1.0) & (FLIR[1:] < 1.0))[0]
        flir_frame_times = time_vector_TS[flir_falling_edge_indices + 1]

        # FrameOut2: Peak detection
        # Since FrameOut2 is 0 or 1 (or similar spikes), we can just look for value > 0.5
        # If it is a reconstructed impulse train (0s and 1s), simple threshold works
        frameout2_peak_indices = np.where(FrameOut2 > 0.5)[0]
        frameout2_peak_times = time_vector_TS[frameout2_peak_indices]

        # 4. Load auxiliary data ONLY if requested
        results = {
            'FLIR': FLIR,
            'FrameOut': FrameOut,
            'FrameOut2': FrameOut2,
            'time_vector_TS': time_vector_TS,
            'flir_frame_times': flir_frame_times,
            'flir_falling_edge_indices': flir_falling_edge_indices,
            'frameout2_peak_times': frameout2_peak_times,
            'frameout2_peak_indices': frameout2_peak_indices
        }

        if load_all:
            # Load the heavy stuff only now
            results['Dig488'] = np.array(f['AI']['488Dig']).squeeze()
            results['analog488'] = np.array(f['AI']['488analog']).squeeze()
            results['LEDdig'] = np.array(f['AI']['LEDdig']).squeeze()
            results['FrameIn'] = np.array(f['DI']['FrameIn']).squeeze()
            results['Trigger_line7'] = np.array(f['DI']['Trigger_line7']).squeeze()
            results['FitHz'] = np.array(f['Freq']['FitHz']).squeeze()
            results['Hz'] = np.array(f['Freq']['Hz']).squeeze()

    # Plotting logic (unchanged, but using the computed values)
    if plotflag:
        fig, axes = plt.subplots(2, 1, figsize=(8, 4), sharex=True)
        axes[0].plot(time_vector_TS, FLIR, label='FLIR Signal')
        axes[0].axhline(y=1.0, color='r', linestyle='--', label='Threshold')
        
        # event plot flir_frame_times
        # Using vlines is often faster for visualization than iterating axvline
        axes[0].vlines(flir_frame_times, 0, np.max(FLIR), colors='orange', linestyles='--', linewidth=0.5)

        axes[1].plot(time_vector_TS, FrameOut2, label='FrameOut2 Signal')
        axes[1].eventplot(frameout2_peak_times, color='green', lineoffsets=0)
        
        plt.tight_layout()
        plt.show()
        
        # QC Plot
        if len(flir_frame_times) > 1 and len(frameout2_peak_times) > 1:
            fig, axes = plt.subplots(2, 1, figsize=(8, 4))
            axes[0].plot(np.diff(flir_frame_times), label='FLIR Frame Times Difference')    
            axes[1].plot(np.diff(frameout2_peak_times), label='FrameOut2 Peak Times Difference')
            axes[0].set_ylim(0, 100)
            axes[0].legend()
            axes[1].legend()
            plt.tight_layout()
            plt.show()
    
    print('Total FLIR frames detected:', len(flir_frame_times))
    print('Total FrameOut2 peaks detected:', len(frameout2_peak_times))
    
    return results