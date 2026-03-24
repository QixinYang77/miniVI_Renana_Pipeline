import os
import numpy as np
from scipy import signal, stats
import cv2
import logging
from scipy.signal import butter, filtfilt
from scipy.ndimage import median_filter
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.backends.backend_pdf import PdfPages


def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a
def highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def lowpass_filter(data, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def whitened_matched_filter(data, locs, window):
    """
    Function for using whitened matched filter to the original signal for better
    SNR. Use welch method to approximate the spectral density of the signal.
    Rescale the signal in frequency domain. After scaling, convolve the signal with
    peak-triggered-average to make spikes more prominent.
    
    Args:
        data: 1-d array
            input signal

        locs: 1-d array
            spike times

        window: 1-d array
            window with size of temporal filter

    Returns:
        datafilt: 1-d array
            signal processed after whitened matched filter
    
    """
    N = np.ceil(np.log2(len(data)))
    censor = np.zeros(len(data))
    censor[locs] = 1
    censor = np.int16(np.convolve(censor.flatten(), np.ones([1, len(window)]).flatten(), 'same'))
    censor = (censor < 0.5)
    noise = data[censor]

    _, pxx = signal.welch(noise, fs=2 * np.pi, window=signal.get_window('hamming', 1000), nfft=2 ** N, detrend=False,
                          nperseg=1000)
    Nf2 = np.concatenate([pxx, np.flipud(pxx[1:-1])])
    scaling_vector = 1 / np.sqrt(Nf2)

    cc = np.pad(data.copy(),(0,int(2**N-len(data))),'constant')    
    dd = (cv2.dft(cc,flags=cv2.DFT_SCALE+cv2.DFT_COMPLEX_OUTPUT)[:,0,:]*scaling_vector[:,np.newaxis])[:,np.newaxis,:]
    dataScaled = cv2.idft(dd)[:,0,0]
    PTDscaled = dataScaled[(locs[:, np.newaxis] + window)]
    PTAscaled = np.mean(PTDscaled, 0)
    datafilt = np.convolve(dataScaled, np.flipud(PTAscaled), 'same')
    datafilt = datafilt[:len(data)]
    return datafilt


def get_subthreshold_activity(trace, spikes, delete_before=1, delete_after=1, median_window=21):
    """
    Remove spike-adjacent samples, interpolate NaNs, then median filter to estimate subthreshold activity.
    """
    trace = np.asarray(trace, dtype=float).copy()
    spikes = np.asarray(spikes, dtype=np.int64)
    n = len(trace)
    if n == 0:
        return trace

    delete_before = max(0, int(delete_before))
    delete_after = max(0, int(delete_after))

    for t in spikes:
        if 0 <= t < n:
            start = max(0, t - delete_before)
            end = min(n, t + delete_after + 1)
            trace[start:end] = np.nan

    nans = np.isnan(trace)
    if np.any(nans) and np.any(~nans):
        not_nans = ~nans
        trace = np.interp(np.arange(n), np.arange(n)[not_nans], trace[not_nans])

    if median_window is None:
        return trace
    median_window = int(median_window)
    if median_window < 1:
        return trace

    return median_filter(trace, size=median_window)


def detect_bursts_from_vm(trace_idx, spike_heights_interpolated, complex_bursts_dict, all_spikes, fr,
                          highpass=1.0, median_window=11, cb_amp_threshold=0.3, cb_duration_threshold=20,
                          isi_threshold_ms=20, baseline_subtract=False, baseline_window_ms=20,
                          baseline_percentile=10, min_num_spikes=2):
    """
    Detect bursts from subthreshold Vm and compute burst metrics.

    Returns:
        simple_spikes_all, complex_spikes, all_spikes, trace_SNR_interpolated, Vm, burst_metrics, complex_bursts_dict_vm
    """
    trace_idx = np.asarray(trace_idx, dtype=float)
    spike_heights_interpolated = np.asarray(spike_heights_interpolated, dtype=float)
    all_spikes = np.asarray(all_spikes, dtype=np.int64)
    n = len(trace_idx)

    if n == 0:
        empty_dict = {
            'complex_bursts': np.array([], dtype=np.int64),
            'starts': np.array([], dtype=np.int64),
            'ends': np.array([], dtype=np.int64),
            'durations_ms': np.array([], dtype=np.int64),
            'amplitudes': np.array([], dtype=float),
            'baselines': np.array([], dtype=float),
            'locs': np.array([], dtype=np.int64),
            'peaks': np.array([], dtype=float),
            'trace_mf': np.array([], dtype=float),
        }
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64), all_spikes, np.array([]), np.array([]), [], empty_dict

    with np.errstate(divide='ignore', invalid='ignore'):
        trace_snr_interpolated = trace_idx / spike_heights_interpolated
    trace_snr_interpolated_unfiltered = trace_snr_interpolated.copy()
    Vm_unfiltered = get_subthreshold_activity(trace_snr_interpolated_unfiltered, all_spikes, median_window=median_window)

    trace_snr_interpolated[~np.isfinite(trace_snr_interpolated)] = np.nan
    trace_snr_interpolated = interpolate_nan_segment(trace_snr_interpolated)
    if highpass and highpass > 0:
        trace_snr_interpolated = highpass_filter(trace_snr_interpolated, fs=fr, cutoff=highpass, order=5)

    trace_snr_interpolated_noCS = trace_snr_interpolated.copy()
    for start, end in zip(complex_bursts_dict['starts'], complex_bursts_dict['ends']):
        trace_snr_interpolated_noCS[start:end + 1] = np.nan
    trace_snr_interpolated_noCS = interpolate_nan_segment(trace_snr_interpolated_noCS)

    Vm = get_subthreshold_activity(trace_snr_interpolated, all_spikes, median_window=median_window)
    _ = get_subthreshold_activity(trace_snr_interpolated_noCS, all_spikes, median_window=median_window)

    isi_threshold_frames = int(np.round(isi_threshold_ms * fr / 1000))
    baseline_window_frames = int(np.round(baseline_window_ms * fr / 1000))
    baseline_window_frames = max(1, baseline_window_frames)

    def _local_baseline(start_idx):
        if not baseline_subtract:
            return 0.0
        base_start = max(0, int(start_idx) - baseline_window_frames)
        base_end = max(0, int(start_idx))
        region = Vm[base_start:base_end]
        if region.size == 0:
            return 0.0
        region = region[np.isfinite(region)]
        if region.size == 0:
            return 0.0
        if baseline_percentile is None:
            return float(np.nanmin(region))
        return float(np.nanpercentile(region, baseline_percentile))
    spike_times = all_spikes[(all_spikes >= 0) & (all_spikes < len(Vm))]
    spike_times = np.sort(spike_times)

    bursts = []
    single_spikes = []
    if len(spike_times) > 0:
        current = [spike_times[0]]
        for spk in spike_times[1:]:
            if spk - current[-1] <= isi_threshold_frames:
                current.append(spk)
            else:
                if len(current) > 1:
                    bursts.append(np.array(current, dtype=np.int64))
                else:
                    single_spikes.append(current[0])
                current = [spk]
        if len(current) > 1:
            bursts.append(np.array(current, dtype=np.int64))
        else:
            single_spikes.append(current[0])
    single_spikes = np.array(single_spikes, dtype=np.int64)

    complex_bursts = []
    simple_bursts = []
    complex_spikes = []
    simple_burst_spikes = []
    burst_metrics = []

    def _safe_nanargmin(arr):
        if not np.any(np.isfinite(arr)):
            return None
        return int(np.nanargmin(arr))

    prev_end = -1
    for i, burst in enumerate(bursts):
        first = int(burst[0])
        last = int(burst[-1])

        first_spike_idx = np.searchsorted(spike_times, first)
        prev_spike = spike_times[first_spike_idx - 1] if first_spike_idx > 0 else None
        next_spike_idx = np.searchsorted(spike_times, last, side='right')
        next_spike = spike_times[next_spike_idx] if next_spike_idx < len(spike_times) else None

        start_search_start = (prev_spike + 1) if prev_spike is not None else 0
        start_search_end = first
        end_search_start = last
        end_search_end = (next_spike - 1) if next_spike is not None else (len(Vm) - 1)

        start_search_end = max(start_search_start, start_search_end)
        end_search_end = max(end_search_start, end_search_end)

        start_window = Vm[start_search_start:start_search_end + 1]
        if len(start_window) == 0:
            start = first
        else:
            valid = np.isfinite(start_window)
            crossing_idx = np.where(valid & (start_window <= 0))[0]
            if len(crossing_idx) > 0:
                start = start_search_start + crossing_idx[-1] + 1
            else:
                min_idx = _safe_nanargmin(start_window)
                start = start_search_start + min_idx if min_idx is not None else first
            start = min(start, first)

        end_window = Vm[end_search_start:end_search_end + 1]
        if len(end_window) == 0:
            end = last
        else:
            valid = np.isfinite(end_window)
            crossing_idx = np.where(valid & (end_window <= 0))[0]
            if len(crossing_idx) > 0:
                end = end_search_start + crossing_idx[0] - 1
            else:
                min_idx = _safe_nanargmin(end_window)
                end = end_search_start + min_idx if min_idx is not None else last
            end = max(end, last)

        if start <= prev_end:
            start = prev_end + 1
        next_burst_first = int(bursts[i + 1][0]) if i + 1 < len(bursts) else None
        if next_burst_first is not None and end >= next_burst_first:
            end = next_burst_first - 1
        if end < start:
            start = first
            end = max(last, start)

        prev_end = end

        window = Vm[start:end + 1]
        window = np.where(np.isfinite(window), window, 0.0)
        peak_raw = np.nanmax(window) if len(window) > 0 else np.nan
        baseline = _local_baseline(start)
        peak_amp = peak_raw - baseline if baseline_subtract else peak_raw
        duration_ms = (end - start + 1) * 1000 / fr
        auc = np.trapz(np.clip(window, 0, None), dx=1 / fr) if len(window) > 0 else 0.0

        meets_criteria = len(burst) >= min_num_spikes and peak_amp >= cb_amp_threshold and duration_ms >= cb_duration_threshold
        burst_metrics.append({
            'start': start,
            'end': end,
            'n_spikes': len(burst),
            'peak_amp': peak_amp,
            'baseline': baseline,
            'duration_ms': duration_ms,
            'auc': auc,
            'is_complex': meets_criteria,
            'is_single': False,
        })

        if meets_criteria:
            complex_bursts.append((start, end))
            complex_spikes.extend(burst.tolist())
        else:
            simple_bursts.append((start, end))
            simple_burst_spikes.extend(burst.tolist())

    complex_spikes = np.sort(np.array(complex_spikes, dtype=np.int64))
    simple_burst_spikes = np.array(simple_burst_spikes, dtype=np.int64)
    if len(simple_burst_spikes) > 0 and len(single_spikes) > 0:
        simple_spikes_all = np.sort(np.concatenate([simple_burst_spikes, single_spikes]))
    elif len(simple_burst_spikes) > 0:
        simple_spikes_all = np.sort(simple_burst_spikes)
    else:
        simple_spikes_all = np.sort(single_spikes)

    all_burst_bounds = sorted(complex_bursts + simple_bursts, key=lambda x: x[0])

    def _nearest_burst_bounds(spk):
        prev_end_local = -1
        next_start_local = len(Vm)
        for s, e in all_burst_bounds:
            if e < spk:
                prev_end_local = max(prev_end_local, e)
            elif s > spk:
                next_start_local = min(next_start_local, s)
                break
            else:
                prev_end_local = max(prev_end_local, e)
        return prev_end_local, next_start_local

    for spk in single_spikes:
        spk = int(spk)
        if spk < 0 or spk >= len(Vm):
            continue

        spk_idx = np.searchsorted(spike_times, spk)
        prev_spike = spike_times[spk_idx - 1] if spk_idx > 0 else None
        next_spike = spike_times[spk_idx + 1] if spk_idx + 1 < len(spike_times) else None

        start_search_start = (prev_spike + 1) if prev_spike is not None else 0
        start_search_end = spk
        end_search_start = spk
        end_search_end = (next_spike - 1) if next_spike is not None else (len(Vm) - 1)

        start_search_end = max(start_search_start, start_search_end)
        end_search_end = max(end_search_start, end_search_end)

        start_window = Vm[start_search_start:start_search_end + 1]
        if len(start_window) == 0:
            start = spk
        else:
            valid = np.isfinite(start_window)
            crossing_idx = np.where(valid & (start_window <= 0))[0]
            if len(crossing_idx) > 0:
                start = start_search_start + crossing_idx[-1] + 1
            else:
                min_idx = _safe_nanargmin(start_window)
                start = start_search_start + min_idx if min_idx is not None else spk
            start = min(start, spk)

        end_window = Vm[end_search_start:end_search_end + 1]
        if len(end_window) == 0:
            end = spk
        else:
            valid = np.isfinite(end_window)
            crossing_idx = np.where(valid & (end_window <= 0))[0]
            if len(crossing_idx) > 0:
                end = end_search_start + crossing_idx[0] - 1
            else:
                min_idx = _safe_nanargmin(end_window)
                end = end_search_start + min_idx if min_idx is not None else spk
            end = max(end, spk)

        prev_burst_end, next_burst_start = _nearest_burst_bounds(spk)
        if start <= prev_burst_end:
            start = prev_burst_end + 1
        if next_burst_start < len(Vm) and end >= next_burst_start:
            end = next_burst_start - 1
        if end < start:
            start = spk
            end = spk

        window = Vm[start:end + 1]
        window = np.where(np.isfinite(window), window, 0.0)
        peak_raw = np.nanmax(window) if len(window) > 0 else np.nan
        baseline = _local_baseline(start)
        peak_amp = peak_raw - baseline if baseline_subtract else peak_raw
        duration_ms = (end - start + 1) * 1000 / fr
        auc = np.trapz(np.clip(window, 0, None), dx=1 / fr) if len(window) > 0 else 0.0

        burst_metrics.append({
            'start': start,
            'end': end,
            'n_spikes': 1,
            'peak_amp': peak_amp,
            'baseline': baseline,
            'duration_ms': duration_ms,
            'auc': auc,
            'is_complex': False,
            'is_single': True,
        })

    complex_entries = [b for b in burst_metrics if b.get('is_complex')]
    starts = np.array([b['start'] for b in complex_entries], dtype=np.int64)
    ends = np.array([b['end'] for b in complex_entries], dtype=np.int64)
    durations_ms = np.array([b['duration_ms'] for b in complex_entries], dtype=np.int64)
    amplitudes = np.array([b['peak_amp'] for b in complex_entries], dtype=float)
    baselines = np.array([b.get('baseline', 0.0) for b in complex_entries], dtype=float)
    locs = []
    peaks = []
    for start, end in zip(starts, ends):
        seg = Vm[start:end + 1]
        if len(seg) == 0 or not np.any(np.isfinite(seg)):
            locs.append(start)
            peaks.append(np.nan)
            continue
        rel_idx = int(np.nanargmax(seg))
        loc = start + rel_idx
        locs.append(loc)
        peaks.append(Vm[loc])
    locs = np.array(locs, dtype=np.int64)
    peaks = np.array(peaks, dtype=float)

    complex_bursts_dict_vm = {
        'complex_bursts': locs.copy(),
        'starts': starts,
        'ends': ends,
        'durations_ms': durations_ms,
        'amplitudes': amplitudes,
        'baselines': baselines,
        'locs': locs,
        'peaks': peaks,
        'trace_mf': Vm.copy(),
    }

    return simple_spikes_all, complex_spikes, all_spikes, trace_snr_interpolated_unfiltered, Vm_unfiltered, burst_metrics, complex_bursts_dict_vm


def adaptive_thresh(pks, clip, pnorm=0.5, min_spikes=10):
    """ Adaptive threshold method for deciding threshold given heights of all peaks.

    Args:
        pks: 1-d array
            height of all peaks

        clip: int
            maximum number of spikes for producing templates

        pnorm: float, between 0 and 1, default is 0.5
            a variable deciding the amount of spikes chosen for adaptive threshold method
            
        min_spikes: int
            minimal number of spikes to be detected

    Returns:
        thresh: float
            threshold for choosing spikes

        falsePosRate: float
            possibility of misclassify noise as real spikes

        detectionRate: float
            possibility of real spikes being detected

        low_spikes: boolean
            true if number of spikes is smaller than minimal value
    """
    # find median of the kernel density estimation of peak heights
    spread = np.array([pks.min(), pks.max()])
    spread = spread + np.diff(spread) * np.array([-0.05, 0.05])
    low_spikes = False
    pts = np.linspace(spread[0], spread[1], 2001)
    kde = stats.gaussian_kde(pks)
    f = kde(pts)    
    xi = pts
    center = np.where(xi > np.median(pks))[0][0]

    fmodel = np.concatenate([f[0:center + 1], np.flipud(f[0:center])])
    if len(fmodel) < len(f):
        fmodel = np.append(fmodel, np.ones(len(f) - len(fmodel)) * min(fmodel))
    else:
        fmodel = fmodel[0:len(f)]

    # adjust the model so it doesn't exceed the data:
    csf = np.cumsum(f) / np.sum(f)
    csmodel = np.cumsum(fmodel) / np.max([np.sum(f), np.sum(fmodel)])
    lastpt = np.where(np.logical_and(csf[0:-1] > csmodel[0:-1] + np.spacing(1), csf[1:] < csmodel[1:]))[0]
    if not lastpt.size:
        lastpt = center
    else:
        lastpt = lastpt[0]
    fmodel[0:lastpt + 1] = f[0:lastpt + 1]
    fmodel[lastpt:] = np.minimum(fmodel[lastpt:], f[lastpt:])

    # find threshold
    csf = np.cumsum(f)
    csmodel = np.cumsum(fmodel)
    csf2 = csf[-1] - csf
    csmodel2 = csmodel[-1] - csmodel
    obj = csf2 ** pnorm - csmodel2 ** pnorm
    maxind = np.argmax(obj)
    thresh = xi[maxind]

    if np.sum(pks > thresh) < min_spikes:
        low_spikes = True
        logging.warning(f'Few spikes were detected. Adjusting threshold to take {min_spikes} largest spikes')
        thresh = np.percentile(pks, 100 * (1 - min_spikes / len(pks)))
    elif ((np.sum(pks > thresh) > clip) & (clip > 0)):
        logging.warning(f'Selecting top {clip} spikes for template')
        thresh = np.percentile(pks, 100 * (1 - clip / len(pks)))

    ix = np.argmin(np.abs(xi - thresh))
    falsePosRate = csmodel2[ix] / csf2[ix]
    detectionRate = (csf2[ix] - csmodel2[ix]) / np.max(csf2 - csmodel2)
    return thresh, falsePosRate, detectionRate, low_spikes


def interpolate_nan_segment(segment):
    """Linearly interpolate NaNs within a 1D segment. Leaves all-NaN segments unchanged."""
    segment = np.asarray(segment, dtype=float)
    if segment.ndim != 1:
        raise ValueError("segment must be 1D")
    nans = np.isnan(segment)
    if not np.any(nans):
        return segment
    valid = ~nans
    if not np.any(valid):
        # all values are NaN; return as-is
        return segment
    segment[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(valid), segment[valid])
    return segment


def split_trace_into_segments(trace, process_window, interpolate_nan=False):
    """
    Break a 1D trace into consecutive segments of length process_window.

    Args:
        trace (np.ndarray): 1D input trace.
        process_window (int): Desired window length for each segment (in samples/frames).
        interpolate_nan (bool): Whether to linearly interpolate NaNs inside each segment.

    Returns:
        segments (list[np.ndarray]): List of trace segments (last one may be shorter).
        segment_bounds (list[tuple]): List of (start_idx, end_idx) pairs for each segment.
    """
    if process_window <= 0:
        raise ValueError("process_window must be a positive integer")
    trace = np.asarray(trace)
    n = int(np.ceil(len(trace) / process_window))
    segments = []
    segment_bounds = []
    for i in range(n):
        start = i * process_window
        end = min((i + 1) * process_window, len(trace))
        segment = trace[start:end]
        if interpolate_nan:
            segment = interpolate_nan_segment(segment)
        segments.append(segment)
        segment_bounds.append((start, end))
    return segments, segment_bounds

# plot overall spike shape for all cells
def plot_spike_shape(trace, spike_time, fr, prewindow=10, postwindow=10, isolated=True, plotaxis=None):
    # trace is a 1D numpy array
    # spike_time is a list of spike times (indices)
    # fr is the frame rate in Hz
    # prewindow and postwindow are in milliseconds

    trace_raw = trace.copy()
    # interpolate NaN values
    nans = np.isnan(trace)
    if np.any(nans):
        not_nans = ~nans
        trace_interpolated = np.interp(np.arange(len(trace)), np.arange(len(trace))[not_nans], trace[not_nans])
    else:
        trace_interpolated = trace.copy()

    trace_interpolated = highpass_filter(trace_interpolated, cutoff=1, fs=fr, order=3)
    trace = (trace_interpolated - np.mean(trace_interpolated)) / np.std(trace_interpolated)

    # set NaN back based on original trace
    trace[np.isnan(trace_raw)] = np.nan


    # convert prewindow and postwindow from milliseconds to frames
    prewindow = int(np.ceil(prewindow * fr / 1000))
    postwindow = int(np.ceil(postwindow * fr / 1000))
    if isolated:
        # only keep spikes if prewindow before it and postwindow after it have no other spikes
        isolated_spike_time = []
        for t in spike_time:
            if all([(t - dt) not in spike_time for dt in range(1, prewindow + 1)]) and all([(t + dt) not in spike_time for dt in range(1, postwindow + 1)]):
                isolated_spike_time.append(t)
        spike_time = isolated_spike_time


    num_spikes = len(spike_time)
    spike_shapes = np.zeros((num_spikes, prewindow + postwindow + 1))
    for i, t in enumerate(spike_time):
        if t - prewindow >= 0 and t + postwindow + 1 < len(trace):
            # if there is no nan values in the window
            if not np.any(np.isnan(trace[t - prewindow:t + postwindow + 1])):
                baseline = np.mean(trace[t - prewindow:t])
                spike_shape_temp = trace[t - prewindow:t + postwindow + 1]
                spike_height = trace[t] - np.min(trace[t-3:t])
                #spike_shapes[i, :] = (spike_shape_temp - baseline) / spike_height
                spike_shapes[i, :] = spike_shape_temp

    mean_spike_shape = np.mean(spike_shapes, axis=0)

    if plotaxis is not None:
        t_plot = np.arange(-prewindow, postwindow + 1)
        # convert t_plot to milliseconds
        t_plot = t_plot * 1000 / int(fr) 
        
        for i in range(num_spikes):
            plotaxis.plot(t_plot, spike_shapes[i, :], color='gray', alpha=0.3, linewidth=0.5)
        
        plotaxis.plot(t_plot, mean_spike_shape)
        
        # tick every 20 ms
        xticks = np.arange(-prewindow * 1000 / fr, (postwindow + 1) * 1000 / fr, 20)
        plotaxis.set_xticks(xticks)
    
    return spike_shapes, mean_spike_shape

def detect_burst_SS(spike_times, fr, burst_isi_threshold=14):
    """
    Detect single spikes (SS) and bursts based on ISI threshold.
    
    Parameters:
    -----------
    spike_times : list or array
        Spike times as frame indices
    fr : float
        Frame rate in Hz
    burst_isi_threshold : float
        ISI threshold in seconds (default 14 ms = 0.014 s)
        Spikes with ISI <= threshold are considered part of a burst
    
    Returns:
    --------
    single_spikes : list
        List of single spike times (frame indices)
    bursts : list of lists
        List of bursts, where each burst is a list of spike times (frame indices)
    burst_event_times : list
        List of burst event times (first spike of each burst, frame indices)
    """
    
    if len(spike_times) == 0:
        return [], [], []
    
    spike_times = np.array(spike_times)
    spike_times = np.sort(spike_times)
    
    # Convert ISI threshold from seconds to frames
    isi_threshold_frames = burst_isi_threshold * fr / 1000  # convert ms to s
    
    single_spikes = []
    bursts = []
    burst_event_times = []
    
    # Track which spikes have been assigned
    i = 0
    while i < len(spike_times):
        current_spike = spike_times[i]
        
        # Check if this spike starts a burst
        if i < len(spike_times) - 1:
            next_isi = spike_times[i + 1] - current_spike
            
            if next_isi <= isi_threshold_frames:
                # Start of a burst - collect all spikes in the burst
                burst = [current_spike]
                j = i + 1
                
                while j < len(spike_times):
                    isi = spike_times[j] - spike_times[j - 1]
                    if isi <= isi_threshold_frames:
                        burst.append(spike_times[j])
                        j += 1
                    else:
                        break
                
                bursts.append(burst)
                burst_event_times.append(burst[0])  # First spike as burst event time
                i = j  # Move to next unassigned spike
            else:
                # Single spike (next spike is too far)
                single_spikes.append(current_spike)
                i += 1
        else:
            # Last spike - check if previous was part of burst
            single_spikes.append(current_spike)
            i += 1
    
    single_spikes = np.array(single_spikes).astype(np.int64)
    burst_event_times = np.array(burst_event_times).astype(np.int64)
    return single_spikes, bursts, burst_event_times

def plot_trace_with_spikes(trace, spikes, ax, trace_color='k', spike_color='r', plot_trace=True, spike_offset=0, spike_size=3):
    if plot_trace:
        ax.plot(trace, color=trace_color, label='Neural Trace', linewidth=0.5)
    ax.scatter(spikes, trace[spikes]+spike_offset, color=spike_color, label='Detected Spikes', s=spike_size)
    # remove ax
    ax.set_axis_off()
    # compact ax
    ax.margins(x=0)


def spike_refinement(trace, spike_times, frame_rate, process_window=60, window_length=5, pnorm=0.5, min_spikes=10):
    """
    Refine spike times using matched filtering and adaptive thresholding.
    
    Parameters:
    -----------
    trace : 1D array
        Neural trace
    spike_times : list or array
        Initial spike times (frame indices)
    frame_rate : float
        Frame rate in Hz
    process_window : int
        Length of the processing window in seconds (default 60 s)
    window_length : float
        Length of the window for matched filtering in ms (default 5 ms)
    pnorm : float
        Parameter for adaptive thresholding (default 0.5)
    min_spikes : int
        Minimum number of spikes to detect (default 10)
    
    Returns:
    --------
    refined_spike_times : array
        Refined spike times (frame indices)
    """
    if len(spike_times) == 0:
        # we do a coarse spike detection first
        spike_times = signal.find_peaks(trace, height=None, distance=2)[0]

    # Remove spikes when trace is NaN
    spike_times = np.array([t for t in spike_times if not np.isnan(trace[t])])
    
    process_window = int(process_window *frame_rate) # in frames

    print(f'Trace length: {trace.shape[0]} frames')
    # number of segments
    n_segments = int(np.ceil(trace.shape[0] / process_window))

    # merge the last 2 segments if the last segment is less than half of process_window
    segment_bounds = []
    for segment_idx in range(n_segments):
        start_frame = segment_idx * process_window
        end_frame = min((segment_idx + 1) * process_window, trace.shape[0])
        segment_bounds.append([start_frame, end_frame])
    if (segment_bounds[-1][1] - segment_bounds[-1][0]) < process_window / 2 and n_segments > 1:
        segment_bounds[-2][1] = segment_bounds[-1][1]
        segment_bounds.pop()
        n_segments -= 1

    print(f'Processing trace in {n_segments} segments')

    refined_spike_times = []
    trace_spk_filt = []
    thresholds = []

    # print each segment length
    for segment_idx in range(n_segments):
        start_frame = segment_bounds[segment_idx][0]
        end_frame = segment_bounds[segment_idx][1]

        print(f'Segment {segment_idx}: {start_frame} to {end_frame} ({end_frame - start_frame} frames)')

        trace_segment = trace[start_frame:end_frame]
        spike_times_segment = spike_times[(spike_times >= start_frame) & (spike_times < end_frame)] - start_frame
        # remove spikes if trace_segment has NaNs at those timepoints
        valid_spike_times_segment = [t for t in spike_times_segment if not np.isnan(trace_segment[t])]
        spike_times_segment = np.array(valid_spike_times_segment)
        trace_segment = interpolate_nan_segment(trace_segment)

        refined_spike_times_segment, trace_segment_spk_filt, thresh = spike_refinement_segment(trace_segment, spike_times_segment, frame_rate, window_length, pnorm, min_spikes)
        refined_spike_times_segment = refined_spike_times_segment + start_frame
        refined_spike_times.extend(refined_spike_times_segment.tolist())
        trace_spk_filt.extend(trace_segment_spk_filt.tolist())
        thresholds.append(thresh)

    refined_spike_times = np.array(refined_spike_times).astype(np.int64)

    return refined_spike_times, np.array(trace_spk_filt), np.array(thresholds), segment_bounds

def spike_refinement_segment(trace_segment, spike_times_segment, frame_rate, window_length=5, pnorm=0.5, min_spikes=10, plot_ax=None):
    """
    Refine spike times in a trace segment using matched filtering and adaptive thresholding.
    
    Parameters:
    -----------
    trace_segment : 1D array
        Neural trace segment
    spike_times_segment : list or array
        Initial spike times in the segment (frame indices)
    frame_rate : float
        Frame rate in Hz
    window_length : float
        Length of the window for matched filtering in ms (default 5 ms)
    pnorm : float
        Parameter for adaptive thresholding (default 0.5)
    min_spikes : int
        Minimum number of spikes to detect (default 10)
    
    Returns:
    --------
    refined_spike_times_segment : array
        Refined spike times in the segment (frame indices)
    """

    # extrace single spikes
    single_spikes, bursts, burst_event_times = detect_burst_SS(spike_times_segment, frame_rate, burst_isi_threshold=14)
    print(f"Number of single spikes: {len(single_spikes)}")
    print(f"Number of bursts: {len(bursts)}")

    single_spikes_shapes, mean_single_spikes_shape = plot_spike_shape(trace_segment, single_spikes, frame_rate, prewindow=window_length, postwindow=window_length, isolated=False, plotaxis=plot_ax)
    burst_shapes, mean_burst_shape = plot_spike_shape(trace_segment, burst_event_times, frame_rate, prewindow=window_length, postwindow=window_length, isolated=False, plotaxis=plot_ax)

    if len(single_spikes) < min_spikes:
        print("Not enough single spikes for refinement, adjusting min_spikes")
        min_spikes = len(single_spikes)

    # Handle case when there are no single spikes at all
    if len(single_spikes) == 0:
        print("No single spikes found in this segment, returning empty results")
        return np.array([]).astype(np.int64), np.zeros(len(trace_segment)), 0.0

    # select top 10 single spikes based on amplitude (time of spike - 1 timepoint before spike)
    SS_amplitudes = []  
    for spk_time in single_spikes:
        if spk_time > 0 and spk_time < len(trace_segment):
            amplitude = trace_segment[spk_time] - trace_segment[spk_time - 1]
            SS_amplitudes.append((spk_time, amplitude))
    SS_amplitudes.sort(key=lambda x: x[1], reverse=True)
    top_single_spikes = [spk[0] for spk in SS_amplitudes[:min_spikes]]
    top_single_spikes = np.array(top_single_spikes).astype(np.int64)  # Ensure integer type
    top_single_spikes_shapes, top_mean_single_spikes_shape = plot_spike_shape(trace_segment, top_single_spikes, frame_rate, prewindow=window_length, postwindow=window_length, isolated=False, plotaxis=plot_ax)

    window_length_frame = int(np.ceil(window_length * frame_rate / 1000))
    window = np.int64(np.arange(-window_length_frame, window_length_frame + 1, 1))
    trace_segment_mf = median_filter(trace_segment, 21)
    trace_segment_spk = trace_segment-trace_segment_mf
    trace_segment_spk_filt = whitened_matched_filter(trace_segment_spk, top_single_spikes , window)
    trace_segment_spk_filt = trace_segment_spk_filt - np.median(trace_segment_spk_filt)

    pks = trace_segment_spk_filt[signal.find_peaks(trace_segment_spk_filt, height=None, distance=2)[0]]
    thresh, falsePosRate, detectionRate, low_spikes = adaptive_thresh(pks, clip=0, pnorm=pnorm, min_spikes=min_spikes)  # clip=0 means no clipping
    refined_spike_times_segment = signal.find_peaks(trace_segment_spk_filt, height=thresh, distance=2)[0]

    return refined_spike_times_segment, trace_segment_spk_filt, thresh


def complex_bursts_detection(trace, spike_times, frame_rate, process_window=60, cutoff_freq=20, pnorm=0.25, clip=100, min_spikes=10, window_length=100, plotflag=False):
    """
    Detect complex spikes (CS) in a neural trace by breaking it into segments.
    
    Parameters:
    -----------
    trace : 1D array
        Neural trace
    spike_times : list or array
        Initial spike times (frame indices)
    frame_rate : float
        Frame rate in Hz
    process_window : int
        Length of the processing window in seconds (default 60 s)
    cutoff_freq : float
        Cutoff frequency for lowpass filter in Hz (default 20 Hz)
    pnorm : float
        Parameter for adaptive thresholding (default 0.25)
    clip : int
        Maximum number of spikes for producing templates (default 100)
    min_spikes : int
        Minimum number of spikes to detect (default 10)
    window_length : float
        Length of the window for matched filtering in ms (default 40 ms)
    plotflag : bool
        Whether to plot the results (default False)
    
    Returns:
    --------
    complex_bursts_dict : dict
        Dictionary containing complex burst information with keys:
        - 'complex_bursts': array of complex burst peak times
        - 'starts': array of complex burst start times
        - 'ends': array of complex burst end times
        - 'durations_ms': array of complex burst durations in ms
        - 'amplitudes': array of complex burst amplitudes
        - 'baselines': array of complex burst baselines
        - 'locs': array of complex burst peak locations
        - 'peaks': array of complex burst peak values
    segment_bounds : list
        List of (start_frame, end_frame) tuples for each segment
    """
    if len(spike_times) == 0:
        # Return empty dict if no spikes
        return {
            'complex_bursts': np.array([]).astype(np.int64),
            'starts': np.array([]).astype(np.int64),
            'ends': np.array([]).astype(np.int64),
            'durations_ms': np.array([]).astype(np.int64),
            'amplitudes': np.array([]),
            'baselines': np.array([]),
            'locs': np.array([]).astype(np.int64),
            'peaks': np.array([])
        }, []

    spike_times = np.array(spike_times)
    # Remove spikes when trace is NaN
    spike_times = np.array([t for t in spike_times if not np.isnan(trace[t])])
    
    process_window_frames = int(process_window * frame_rate)  # in frames

    print(f'Trace length: {trace.shape[0]} frames')
    # number of segments
    n_segments = int(np.ceil(trace.shape[0] / process_window_frames))

    # merge the last 2 segments if the last segment is less than half of process_window
    segment_bounds = []
    for segment_idx in range(n_segments):
        start_frame = segment_idx * process_window_frames
        end_frame = min((segment_idx + 1) * process_window_frames, trace.shape[0])
        segment_bounds.append([start_frame, end_frame])
    if (segment_bounds[-1][1] - segment_bounds[-1][0]) < process_window_frames / 2 and n_segments > 1:
        segment_bounds[-2][1] = segment_bounds[-1][1]
        segment_bounds.pop()
        n_segments -= 1

    print(f'Processing trace in {n_segments} segments for CS detection')

    # Initialize lists to collect results from all segments
    all_complex_bursts = []
    all_starts = []
    all_ends = []
    all_durations_ms = []
    all_amplitudes = []
    all_baselines = []
    all_locs = []
    all_peaks = []
    all_trace_mf = []
    all_trace_filt = []
    all_trace_filt_wmf = []
    all_thresholds = []

    for segment_idx in range(n_segments):
        start_frame = segment_bounds[segment_idx][0]
        end_frame = segment_bounds[segment_idx][1]
        segment_len = end_frame - start_frame

        print(f'Segment {segment_idx}: {start_frame} to {end_frame} ({end_frame - start_frame} frames)')

        trace_segment = trace[start_frame:end_frame].copy()
        spike_times_segment = spike_times[(spike_times >= start_frame) & (spike_times < end_frame)] - start_frame
        
        # Skip segment if no spikes
        if len(spike_times_segment) == 0:
            print(f'  No spikes in segment {segment_idx}, skipping')
            all_trace_mf.extend([np.nan] * segment_len)
            all_trace_filt.extend([np.nan] * segment_len)
            all_trace_filt_wmf.extend([np.nan] * segment_len)
            all_thresholds.append(np.nan)
            continue
        
        # Remove spikes if trace_segment has NaNs at those timepoints
        valid_spike_times_segment = [t for t in spike_times_segment if not np.isnan(trace_segment[t])]
        spike_times_segment = np.array(valid_spike_times_segment)
        
        if len(spike_times_segment) == 0:
            print(f'  No valid spikes in segment {segment_idx} after NaN removal, skipping')
            all_trace_mf.extend([np.nan] * segment_len)
            all_trace_filt.extend([np.nan] * segment_len)
            all_trace_filt_wmf.extend([np.nan] * segment_len)
            all_thresholds.append(np.nan)
            continue

        # Run CS detection on segment
        try:
            complex_bursts_dict_segment, trace_mf_segment, trace_filt_segment, trace_filt_wmf_segment, _, thresh_segment = complex_bursts_detection_segment(
                trace_segment, spike_times_segment, frame_rate, 
                cutoff_freq=cutoff_freq, pnorm=pnorm, clip=clip, 
                min_spikes=min_spikes, window_length=window_length, plotflag=False
            )
        except Exception as e:
            print(f'  Error in segment {segment_idx}: {e}')
            all_trace_mf.extend([np.nan] * segment_len)
            all_trace_filt.extend([np.nan] * segment_len)
            all_trace_filt_wmf.extend([np.nan] * segment_len)
            all_thresholds.append(np.nan)
            continue

        if len(trace_mf_segment) != segment_len:
            trace_mf_segment = np.pad(trace_mf_segment, (0, max(0, segment_len - len(trace_mf_segment))), constant_values=np.nan)[:segment_len]
        if len(trace_filt_segment) != segment_len:
            trace_filt_segment = np.pad(trace_filt_segment, (0, max(0, segment_len - len(trace_filt_segment))), constant_values=np.nan)[:segment_len]
        if len(trace_filt_wmf_segment) != segment_len:
            trace_filt_wmf_segment = np.pad(trace_filt_wmf_segment, (0, max(0, segment_len - len(trace_filt_wmf_segment))), constant_values=np.nan)[:segment_len]

        # Store filtered traces for plotting
        all_trace_mf.extend(trace_mf_segment.tolist())
        all_trace_filt.extend(trace_filt_segment.tolist())
        all_trace_filt_wmf.extend(trace_filt_wmf_segment.tolist())
        all_thresholds.append(thresh_segment)

        # Offset indices back to original trace coordinates
        if len(complex_bursts_dict_segment['complex_bursts']) > 0:
            all_complex_bursts.extend((complex_bursts_dict_segment['complex_bursts'] + start_frame).tolist())
            all_starts.extend((complex_bursts_dict_segment['starts'] + start_frame).tolist())
            all_ends.extend((complex_bursts_dict_segment['ends'] + start_frame).tolist())
            all_durations_ms.extend(complex_bursts_dict_segment['durations_ms'].tolist())
            all_amplitudes.extend(complex_bursts_dict_segment['amplitudes'].tolist())
            all_baselines.extend(complex_bursts_dict_segment['baselines'].tolist())
            all_locs.extend((complex_bursts_dict_segment['locs'] + start_frame).tolist())
            all_peaks.extend(complex_bursts_dict_segment['peaks'].tolist())
            
            print(f'  Found {len(complex_bursts_dict_segment["complex_bursts"])} complex spikes in segment {segment_idx}')

    # Combine all results
    complex_bursts_dict = {
        'complex_bursts': np.array(all_complex_bursts).astype(np.int64),
        'starts': np.array(all_starts).astype(np.int64),
        'ends': np.array(all_ends).astype(np.int64),
        'durations_ms': np.array(all_durations_ms).astype(np.int64),
        'amplitudes': np.array(all_amplitudes),
        'baselines': np.array(all_baselines),
        'locs': np.array(all_locs).astype(np.int64),
        'peaks': np.array(all_peaks),
        'trace_mf': np.array(all_trace_mf),
        'trace_filt': np.array(all_trace_filt),
        'trace_filt_wmf': np.array(all_trace_filt_wmf)
    }

    print(f'Total complex spikes detected: {len(complex_bursts_dict["complex_bursts"])}')

    if plotflag:
        # Interpolate NaN values in trace for plotting
        trace_plot = trace.copy()
        nans = np.isnan(trace_plot)
        if np.any(nans):
            not_nans = ~nans
            trace_plot = np.interp(np.arange(len(trace_plot)), np.arange(len(trace_plot))[not_nans], trace_plot[not_nans])
        
        # Apply median filter for visualization
        trace_mf = median_filter(trace_plot, size=11)
        
        fig, ax = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
        
        ax[0].plot(trace_plot, label='Original Trace', linewidth=0.5, alpha=0.7)
        ax[0].plot(trace_mf, label='Median Filtered Trace', color='black', linewidth=1)
        
        # Highlight complex spike regions
        for start, end in zip(complex_bursts_dict['starts'], complex_bursts_dict['ends']):
            ax[0].axvspan(start, end, color='yellow', alpha=0.3)
        
        # Plot amplitude lines from baseline to peak
        for loc, peak, baseline in zip(complex_bursts_dict['locs'], complex_bursts_dict['peaks'], complex_bursts_dict['baselines']):
            ax[0].plot([loc, loc], [baseline, peak], color='red', linestyle='-', linewidth=1)
        
        # Mark segment boundaries
        for i, (start, end) in enumerate(segment_bounds):
            ax[0].axvline(x=start, color='blue', linestyle='--', linewidth=0.5, alpha=0.5)
            ax[0].text(start, ax[0].get_ylim()[1], f'Seg {i}', fontsize=8, color='blue', alpha=0.7)
        
        ax[0].set_xlabel('Frame')
        ax[0].set_ylabel('Amplitude')
        #ax[0].set_title(f'Complex Spike Detection (n={len(complex_bursts_dict["complex_bursts"])})')
        ax[0].legend(loc='upper right', fontsize=8)

        # Plot trace_filt (lowpass filtered trace)
        ax[1].plot(all_trace_filt, label='Lowpass Filtered Trace', color='orange', linewidth=0.5)
        ax[1].set_ylabel('Amplitude')
        ax[1].legend(loc='upper right', fontsize=8)
        
        # Plot trace_filt_wmf (whitened matched filter output)
        ax[2].plot(all_trace_filt_wmf, label='Whitened Matched Filter Output', color='green', linewidth=0.5)
        # Plot threshold lines for each segment
        for i, (bounds, thresh) in enumerate(zip(segment_bounds, all_thresholds)):
            if not np.isnan(thresh):
                ax[2].hlines(y=thresh, xmin=bounds[0], xmax=bounds[1], color='red', linestyle='--', linewidth=1)
        ax[2].set_xlabel('Frame')
        ax[2].set_ylabel('Amplitude')
        ax[2].legend(loc='upper right', fontsize=8)

        # remove axes and compact layout
        for a in ax:
            a.set_axis_off()
            a.margins(x=0)

        plt.tight_layout()
        plt.show()

    return complex_bursts_dict, segment_bounds


# def complex_bursts_detection_segment_old(trace_idx, spike_idx, fr, cutoff_freq=20, pnorm=0.25, clip=100, min_spikes=10, window_length=40, plotflag=False):
#     """
#     Detect complex spikes (CS) in a neural trace segment.
#     Parameters:
#     -----------
#     trace_idx : 1D array
#         Neural trace segment
#     spike_idx : list or array
#         Initial spike times in the segment (frame indices)
#     fr : float
#         Frame rate in Hz
#     cutoff_freq : float
#         Cutoff frequency for lowpass filter in Hz (default 20 Hz)
#     pnorm : float
#         Parameter for adaptive thresholding (default 0.25)
#     clip : int
#         Maximum number of spikes for producing templates (default 100)
#     min_spikes : int
#         Minimum number of spikes to detect (default 10)
#     window_length : float
#         Length of the window for matched filtering in ms (default 40 ms)
#     """

#     single_spikes, bursts, burst_event_time = detect_burst_SS(spike_idx, fr=fr, burst_isi_threshold=14)
#     # simple spikes are single_spikes combined with first spike within bursts
#     simple_spikes = np.sort(np.concatenate([single_spikes, burst_event_time]))

#     # Interpolate NaN values in trace_idx
#     nans = np.isnan(trace_idx)
#     if np.any(nans):
#         not_nans = ~nans
#         # Use linear interpolation, with extrapolation for NaNs at beginning/end
#         trace_idx = np.interp(np.arange(len(trace_idx)), np.arange(len(trace_idx))[not_nans], trace_idx[not_nans])

#     trace_idx_sub = trace_idx.copy()

#     # remove traces around single_spikes (1 points before and after)
#     for t in single_spikes:
#         trace_idx_sub[max(0, t-1):t+2] = np.nan

#     # for each burst, within a local window of 20ms, delete all local spikes
#     burst_window = int(40 * fr / 1000)  # 20 ms window in frames
#     for burst_start in burst_event_time:
#         start_idx = max(0, burst_start - burst_window // 2)
#         end_idx = min(len(trace_idx_sub), burst_start + burst_window // 2)
#         # detect local peaks within this window
#         local_segment = trace_idx_sub[start_idx:end_idx]
#         local_pks = signal.find_peaks(local_segment)[0]
#         # set these local peaks to NaN
#         for pk in local_pks:
#             trace_idx_sub[start_idx + pk] = np.nan

#     # linear interpolation for NaN values in trace_idx_sub
#     nans = np.isnan(trace_idx_sub)
#     if np.any(nans):
#         not_nans = ~nans
#         trace_idx_sub = np.interp(np.arange(len(trace_idx_sub)), np.arange(len(trace_idx_sub))[not_nans], trace_idx_sub[not_nans])

#     # Apply median filter of size 11
#     trace_idx_mf = median_filter(trace_idx_sub, size=11) # This is the Vm_sub!

#     # Apply lowpass filter
#     trace_filt = lowpass_filter(trace_idx_mf, cutoff=cutoff_freq, fs=fr, order=3)
#     trace_filt = trace_filt - np.median(trace_filt)

#     ##### CS detection #####
#     window_length = int(np.ceil(window_length * fr / 1000))
#     complex_burst_baseline_window = 10 # in samples
#     trough_search_window = int(100 * fr / 1000)  # 50 ms window to search for troughs

#     pks = trace_filt[signal.find_peaks(trace_filt, height=None, distance=20)[0]]
#     thresh, _, _, low_spikes = adaptive_thresh(pks, clip, 0.25, min_spikes) # low pnorm to be harsh first
#     locs = signal.find_peaks(trace_filt, height=thresh)[0]

#     window = np.int64(np.arange(-window_length, window_length + 1, 1))
#     PTD = trace_filt[(locs[:, np.newaxis] + window)]
#     PTA = np.median(PTD, 0)
#     PTA = PTA - np.min(PTA)
#     templates = PTA

#     trace_filt_wmf = whitened_matched_filter(trace_filt, locs, window)
#     trace_filt_wmf = trace_filt_wmf - np.median(trace_filt_wmf)

#     pks = trace_filt_wmf[signal.find_peaks(trace_filt_wmf, height=None, distance=20)[0]]
#     thresh, falsePosRate, detectionRate, low_spikes = adaptive_thresh(pks, clip=0, pnorm=pnorm, min_spikes=10)  # clip=0 means no clipping
#     complex_bursts_init = signal.find_peaks(trace_filt_wmf, height=thresh, distance=20)[0]

#     # detect start and end of complex spikes by finding the troughs around each spike
#     complex_burst_starts = []
#     complex_burst_ends = []
#     complex_bursts_durations = []
#     complex_bursts_amplitudes = []
#     complex_bursts_baselines = []
#     complex_bursts_locs = []
#     complex_bursts_peaks = []

#     complex_bursts = []
#     prev_complex_burst_end = 0  # Track end of previous complex burst

#     for spike in complex_bursts_init:

#         # Find trough (minimum) before spike in trace_filt within 50ms window
#         # But search start must be after the end of the previous complex burst
#         search_start = max(prev_complex_burst_end, spike - trough_search_window)
#         trough_region_before = trace_filt[search_start:spike]
#         if len(trough_region_before) == 0:
#             complex_burst_starts_temp = spike
#         else:
#             # Take the global minimum within the search window
#             trough_idx = np.argmin(trough_region_before)
#             complex_burst_starts_temp = search_start + trough_idx
        
#         # Find trough (local minimum) immediately after spike in trace_filt
#         search_end = min(len(trace_filt), spike + trough_search_window)
#         trough_region_after = trace_filt[spike:search_end]
#         if len(trough_region_after) == 0:
#             complex_burst_ends_temp = spike
#         else:
#             # Find all local minima in the region after spike
#             local_mins_after = signal.argrelmin(trough_region_after)[0]
#             if len(local_mins_after) > 0:
#                 # Take the first local minimum (closest to spike)
#                 complex_burst_ends_temp = spike + local_mins_after[0]
#             else:
#                 # If no local minimum found, take the global minimum
#                 trough_idx = np.argmin(trough_region_after)
#                 complex_burst_ends_temp = spike + trough_idx

#         # take the minimum of datapoints before the start as baseline
#         baseline_start_region = trace_idx_mf[max(0, complex_burst_starts_temp-complex_burst_baseline_window):complex_burst_starts_temp]
#         if len(baseline_start_region) == 0:
#             baseline_start = 0
#         else:
#             baseline_start = np.min(baseline_start_region)

#         baseline = baseline_start
        
#         # Take the time of the maximum within the burst as the complex burst location
#         pk_idx_within_burst = np.argmax(trace_idx_mf[complex_burst_starts_temp:complex_burst_ends_temp+1])

#         peak_value = trace_idx_mf[complex_burst_starts_temp + pk_idx_within_burst]
#         amplitude = peak_value - baseline

#         # there must be a spike_idx present between start and end
#         if np.any((spike_idx >= complex_burst_starts_temp) & (spike_idx <= complex_burst_ends_temp)):
#             complex_bursts.append(spike)
#             complex_burst_starts.append(complex_burst_starts_temp)
#             complex_burst_ends.append(complex_burst_ends_temp)
#             complex_bursts_durations.append(int((complex_burst_ends_temp - complex_burst_starts_temp)*1000/fr))  # in ms
#             complex_bursts_amplitudes.append(amplitude)
#             complex_bursts_baselines.append(baseline_start)
#             complex_bursts_locs.append(complex_burst_starts_temp + pk_idx_within_burst)
#             complex_bursts_peaks.append(peak_value)
#             # Update previous burst end for next iteration
#             prev_complex_burst_end = complex_burst_ends_temp

#     # return complex bursts as a dictionary
#     complex_bursts_dict = {
#         'complex_bursts': np.array(complex_bursts).astype(np.int64),
#         'starts': np.array(complex_burst_starts).astype(np.int64),
#         'ends': np.array(complex_burst_ends).astype(np.int64),
#         'durations_ms': np.array(complex_bursts_durations).astype(np.int64),
#         'amplitudes': np.array(complex_bursts_amplitudes),
#         'baselines': np.array(complex_bursts_baselines),
#         'locs': np.array(complex_bursts_locs).astype(np.int64),
#         'peaks': np.array(complex_bursts_peaks)
#     }

#     if plotflag:
#         fig, axes = plt.subplots(3, 1, figsize=(8,6), sharex=True)

#         axes[0].plot(trace_idx, label='Original Trace', linewidth=0.5)
#         axes[0].plot(trace_idx_mf, label='Median Filtered Trace', color='black', linewidth=1)
#         #axes[0].plot(complex_bursts, trace_idx_mf[complex_bursts], "x", label='Detected Spikes', color='red')
#         for start, end in zip(complex_burst_starts, complex_burst_ends):
#             axes[0].axvspan(start, end, color='yellow', alpha=0.3)

#         # plot at the loc, a line from peak to baseline
#         for loc, peak, baseline in zip(complex_bursts_locs, complex_bursts_peaks, complex_bursts_baselines):
#             axes[0].plot([loc, loc], [baseline, peak], color='red', linestyle='-', linewidth=1)

#         axes[1].plot(trace_filt, label='Whitened Matched Filter Output', color='orange')
#         # plot pks
#         #axes[1].plot(complex_bursts, trace_filt[complex_bursts], "x", label='Detected Peaks', color='red')

#         axes[2].plot(trace_filt_wmf, label='Whitened Matched Filter Output after WMF', color='green')
#         # plot pks
#         #axes[2].plot(complex_bursts, trace_filt_wmf[complex_bursts], "x", label='Detected Peaks', color='red')
#         # plot threshold lines
#         axes[2].axhline(y=thresh, color='red', linestyle='--', label='Detection Threshold')

#     return complex_bursts_dict, trace_idx_mf,trace_filt, trace_filt_wmf, templates, thresh


def complex_bursts_detection_segment(
    trace_idx,
    spike_idx,
    fr,
    cutoff_freq=20,
    pnorm=0.25,
    clip=100,
    min_spikes=10,
    window_length=40,           # ms, template half-width for WMF window
    plotflag=False,
    # --- trough bracketing params ---
    trough_distance_ms=3,       # allow close troughs (helps splitting)
    trough_prom_frac=0.01,      # smaller = more troughs detected
    min_cs_dur_ms=15,           # reject ultra-short segments
    max_cs_dur_ms=250,          # hard cap to avoid runaway merges
):
    """
    Detect complex bursts (CS) within a trace segment.

    Pipeline (same spirit as your original):
      1) Construct Vm_sub by removing sharp spikes from trace_idx, interpolate gaps,
         then median filter (trace_idx_mf).
      2) Lowpass filter trace_idx_mf -> trace_filt (CS envelope-ish) and detect initial peaks.
      3) Build WMF template from trace_filt, compute whitened matched-filter output trace_filt_wmf,
         detect suprathreshold candidate CS peaks: complex_bursts_init.
      4) For each candidate peak, bracket start/end by nearest troughs in trace_filt.
      5) IMPORTANT: If a bracket contains multiple WMF peaks, split it at the deepest trough(s)
         between peaks so bursts don’t merge.

    Returns:
      complex_bursts_dict, trace_idx_mf, trace_filt, trace_filt_wmf, templates, thresh
    """

    # -----------------------------
    # Helpers
    # -----------------------------
    def _interp_nan_1d(x):
        x = np.asarray(x, dtype=float)
        nans = np.isnan(x)
        if not np.any(nans):
            return x
        ok = ~nans
        if not np.any(ok):
            return x  # all NaN
        x[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(ok), x[ok])
        return x

    def _find_troughs(x, distance=1, prominence=None):
        # troughs are peaks of the inverted signal
        troughs, props = signal.find_peaks(-x, distance=distance, prominence=prominence)
        return troughs.astype(np.int64), props

    def _bracket_by_troughs(trace_filt_local, spike, troughs, prev_end=0,
                            min_dur_frames=1, max_dur_frames=None):
        troughs = np.asarray(troughs, dtype=np.int64)

        left = troughs[troughs < spike]
        left = left[left >= prev_end]
        right = troughs[troughs > spike]

        if len(left) == 0:
            start = max(prev_end, spike - 1)
        else:
            start = int(left[-1])

        if len(right) == 0:
            end = int(spike)
        else:
            end = int(right[0])

        if end <= start:
            end = start + 1

        if max_dur_frames is not None and (end - start) > max_dur_frames:
            end = start + int(max_dur_frames)

        if (end - start) < min_dur_frames:
            end = start + int(min_dur_frames)

        start = max(0, start)
        end = min(len(trace_filt_local) - 1, end)
        if end <= start:
            end = min(start + 1, len(trace_filt_local) - 1)

        return start, end

    def _deepest_trough_between(troughs, trace_filt_local, a, b):
        # choose trough in (a,b) with minimum trace_filt value
        cand = troughs[(troughs > a) & (troughs < b)]
        if len(cand) == 0:
            return (a + b) // 2
        return int(cand[np.argmin(trace_filt_local[cand])])

    # -----------------------------
    # Input sanitation
    # -----------------------------
    trace_idx = np.asarray(trace_idx, dtype=float).copy()
    spike_idx = np.asarray(spike_idx, dtype=np.int64).copy()
    spike_idx = np.sort(spike_idx)

    if trace_idx.ndim != 1:
        raise ValueError("trace_idx must be 1D")

    n = len(trace_idx)
    if n == 0:
        empty = {
            'complex_bursts': np.array([], dtype=np.int64),
            'starts': np.array([], dtype=np.int64),
            'ends': np.array([], dtype=np.int64),
            'durations_ms': np.array([], dtype=np.int64),
            'amplitudes': np.array([], dtype=float),
            'baselines': np.array([], dtype=float),
            'locs': np.array([], dtype=np.int64),
            'peaks': np.array([], dtype=float),
        }
        return empty, trace_idx, trace_idx, trace_idx, np.array([], dtype=float), np.nan

    # -----------------------------
    # 1) Define simple spikes to suppress when estimating Vm_sub
    # -----------------------------
    single_spikes, bursts, burst_event_time = detect_burst_SS(spike_idx, fr=fr, burst_isi_threshold=14)
    single_spikes = np.asarray(single_spikes, dtype=np.int64)
    burst_event_time = np.asarray(burst_event_time, dtype=np.int64)
    simple_spikes = np.sort(np.concatenate([single_spikes, burst_event_time]).astype(np.int64))

    # -----------------------------
    # 2) Interpolate NaNs in raw trace
    # -----------------------------
    trace_idx = _interp_nan_1d(trace_idx)
    if np.all(np.isnan(trace_idx)):
        empty = {
            'complex_bursts': np.array([], dtype=np.int64),
            'starts': np.array([], dtype=np.int64),
            'ends': np.array([], dtype=np.int64),
            'durations_ms': np.array([], dtype=np.int64),
            'amplitudes': np.array([], dtype=float),
            'baselines': np.array([], dtype=float),
            'locs': np.array([], dtype=np.int64),
            'peaks': np.array([], dtype=float),
        }
        return empty, trace_idx, trace_idx, trace_idx, np.array([], dtype=float), np.nan

    # -----------------------------
    # 3) Vm_sub construction: remove sharp spikes then median filter
    # -----------------------------
    trace_idx_sub = trace_idx.copy()

    # Remove around isolated single spikes (±1 sample)
    for t in single_spikes:
        if 0 <= t < n:
            trace_idx_sub[max(0, t - 1):min(n, t + 2)] = np.nan

    # Remove local peaks around burst starts (within 40 ms total window)
    burst_window = int(40 * fr / 1000)
    for burst_start in burst_event_time:
        if not (0 <= burst_start < n):
            continue
        start_idx = max(0, burst_start - burst_window // 2)
        end_idx = min(n, burst_start + burst_window // 2)
        local = trace_idx_sub[start_idx:end_idx]
        if len(local) == 0:
            continue
        local_pks = signal.find_peaks(local)[0]
        for pk in local_pks:
            trace_idx_sub[start_idx + pk] = np.nan

    trace_idx_sub = _interp_nan_1d(trace_idx_sub)

    # Vm_sub
    trace_idx_mf = median_filter(trace_idx_sub, size=11)

    # -----------------------------
    # 4) Lowpass -> trace_filt
    # -----------------------------
    trace_filt = lowpass_filter(trace_idx_mf, cutoff=cutoff_freq, fs=fr, order=3)
    trace_filt = trace_filt - np.median(trace_filt)

    # -----------------------------
    # 5) Initial peak detection on trace_filt (for template seeding)
    # -----------------------------
    init_dist = max(1, int(20 * fr / 1000))
    init_peaks = signal.find_peaks(trace_filt, height=None, distance=init_dist)[0]
    if len(init_peaks) == 0:
        empty = {
            'complex_bursts': np.array([], dtype=np.int64),
            'starts': np.array([], dtype=np.int64),
            'ends': np.array([], dtype=np.int64),
            'durations_ms': np.array([], dtype=np.int64),
            'amplitudes': np.array([], dtype=float),
            'baselines': np.array([], dtype=float),
            'locs': np.array([], dtype=np.int64),
            'peaks': np.array([], dtype=float),
        }
        return empty, trace_idx_mf, trace_filt, trace_filt, np.array([], dtype=float), np.nan

    pks0 = trace_filt[init_peaks]
    thresh0, _, _, _ = adaptive_thresh(pks0, clip=clip, pnorm=0.25, min_spikes=min_spikes)
    locs = signal.find_peaks(trace_filt, height=thresh0, distance=init_dist)[0].astype(np.int64)

    if len(locs) == 0:
        empty = {
            'complex_bursts': np.array([], dtype=np.int64),
            'starts': np.array([], dtype=np.int64),
            'ends': np.array([], dtype=np.int64),
            'durations_ms': np.array([], dtype=np.int64),
            'amplitudes': np.array([], dtype=float),
            'baselines': np.array([], dtype=float),
            'locs': np.array([], dtype=np.int64),
            'peaks': np.array([], dtype=float),
        }
        return empty, trace_idx_mf, trace_filt, trace_filt, np.array([], dtype=float), np.nan

    # -----------------------------
    # 6) WMF on trace_filt using PTA template window
    # -----------------------------
    wl_frames = int(np.ceil(window_length * fr / 1000))
    window = np.int64(np.arange(-wl_frames, wl_frames + 1, 1))

    # Keep only locs where indexing is valid
    valid_locs = locs[(locs + window[0] >= 0) & (locs + window[-1] < n)]
    if len(valid_locs) == 0:
        valid_locs = locs  # fallback

    PTD = trace_filt[(valid_locs[:, None] + window)]
    PTA = np.median(PTD, axis=0)
    PTA = PTA - np.min(PTA)
    templates = PTA

    trace_filt_wmf = whitened_matched_filter(trace_filt, valid_locs, window)
    trace_filt_wmf = trace_filt_wmf - np.median(trace_filt_wmf)

    # -----------------------------
    # 7) Detect suprathreshold peaks on WMF output
    # -----------------------------
    final_dist = max(1, int(20 * fr / 1000))
    wmf_peaks = signal.find_peaks(trace_filt_wmf, height=None, distance=final_dist)[0]
    if len(wmf_peaks) == 0:
        empty = {
            'complex_bursts': np.array([], dtype=np.int64),
            'starts': np.array([], dtype=np.int64),
            'ends': np.array([], dtype=np.int64),
            'durations_ms': np.array([], dtype=np.int64),
            'amplitudes': np.array([], dtype=float),
            'baselines': np.array([], dtype=float),
            'locs': np.array([], dtype=np.int64),
            'peaks': np.array([], dtype=float),
        }
        return empty, trace_idx_mf, trace_filt, trace_filt_wmf, templates, np.nan

    pks = trace_filt_wmf[wmf_peaks]
    thresh, _, _, _ = adaptive_thresh(pks, clip=0, pnorm=pnorm, min_spikes=min_spikes)
    complex_bursts_init = signal.find_peaks(trace_filt_wmf, height=thresh, distance=final_dist)[0].astype(np.int64)
    complex_bursts_init = np.sort(complex_bursts_init)

    if len(complex_bursts_init) == 0:
        empty = {
            'complex_bursts': np.array([], dtype=np.int64),
            'starts': np.array([], dtype=np.int64),
            'ends': np.array([], dtype=np.int64),
            'durations_ms': np.array([], dtype=np.int64),
            'amplitudes': np.array([], dtype=float),
            'baselines': np.array([], dtype=float),
            'locs': np.array([], dtype=np.int64),
            'peaks': np.array([], dtype=float),
        }
        return empty, trace_idx_mf, trace_filt, trace_filt_wmf, templates, thresh

    # -----------------------------
    # 8) Find troughs in trace_filt for bracketing + splitting
    # -----------------------------
    trough_distance = max(1, int(trough_distance_ms * fr / 1000))
    trough_prom = float(np.std(trace_filt) * trough_prom_frac) if trough_prom_frac is not None else None
    troughs, _ = _find_troughs(trace_filt, distance=trough_distance, prominence=trough_prom)
    troughs = np.sort(troughs)

    min_dur_frames = max(1, int(min_cs_dur_ms * fr / 1000))
    max_dur_frames = int(max_cs_dur_ms * fr / 1000) if max_cs_dur_ms is not None else None

    # -----------------------------
    # 9) Bracket each WMF peak; if a bracket contains >1 peak, split at deepest troughs
    # -----------------------------
    events = []
    prev_end = 0

    for pk in complex_bursts_init:
        start, end = _bracket_by_troughs(
            trace_filt_local=trace_filt,
            spike=int(pk),
            troughs=troughs,
            prev_end=int(prev_end),
            min_dur_frames=min_dur_frames,
            max_dur_frames=max_dur_frames
        )

        # peaks inside this bracket
        pks_in = complex_bursts_init[(complex_bursts_init >= start) & (complex_bursts_init <= end)]

        if len(pks_in) <= 1:
            events.append((start, end, int(pk)))
            prev_end = end
            continue

        # Split: cut between consecutive peaks at deepest trough
        cutpoints = []
        for a, b in zip(pks_in[:-1], pks_in[1:]):
            cutpoints.append(_deepest_trough_between(troughs, trace_filt, int(a), int(b)))
        cutpoints = np.asarray(cutpoints, dtype=np.int64)

        bounds = [start] + cutpoints.tolist() + [end]
        for i, pk_i in enumerate(pks_in):
            s_i = int(bounds[i])
            e_i = int(bounds[i + 1])
            if e_i <= s_i:
                continue
            if (e_i - s_i) < min_dur_frames:
                continue
            # also apply max cap locally, just in case
            if max_dur_frames is not None and (e_i - s_i) > max_dur_frames:
                e_i = s_i + max_dur_frames
            events.append((s_i, e_i, int(pk_i)))

        prev_end = end

    # -----------------------------
    # 10) Compute features per event (baseline/amp/loc/peak) on trace_idx_mf
    # -----------------------------
    complex_burst_baseline_window = 10  # samples
    complex_bursts = []
    starts = []
    ends = []
    durations_ms = []
    amplitudes = []
    baselines = []
    locs = []
    peaks_out = []

    for start_idx, end_idx, pk in events:
        start_idx = max(0, int(start_idx))
        end_idx = min(n - 1, int(end_idx))
        if end_idx <= start_idx:
            continue

        # baseline from mf trace just before start
        baseline_region = trace_idx_mf[max(0, start_idx - complex_burst_baseline_window):start_idx]
        baseline = float(np.min(baseline_region)) if len(baseline_region) else 0.0

        seg = trace_idx_mf[start_idx:end_idx + 1]
        if len(seg) == 0:
            continue

        pk_rel = int(np.argmax(seg))
        loc = start_idx + pk_rel
        peak_value = float(trace_idx_mf[loc])
        amp = peak_value - baseline
        dur_ms = int((end_idx - start_idx) * 1000 / fr)

        # require at least one raw spike within [start,end]
        if np.any((spike_idx >= start_idx) & (spike_idx <= end_idx)):
            complex_bursts.append(int(pk))     # WMF peak index
            starts.append(int(start_idx))
            ends.append(int(end_idx))
            durations_ms.append(int(dur_ms))
            amplitudes.append(float(amp))
            baselines.append(float(baseline))
            locs.append(int(loc))
            peaks_out.append(float(peak_value))

    complex_bursts_dict = {
        'complex_bursts': np.asarray(complex_bursts, dtype=np.int64),
        'starts': np.asarray(starts, dtype=np.int64),
        'ends': np.asarray(ends, dtype=np.int64),
        'durations_ms': np.asarray(durations_ms, dtype=np.int64),
        'amplitudes': np.asarray(amplitudes, dtype=float),
        'baselines': np.asarray(baselines, dtype=float),
        'locs': np.asarray(locs, dtype=np.int64),
        'peaks': np.asarray(peaks_out, dtype=float),
    }

    # -----------------------------
    # 11) Optional plotting
    # -----------------------------
    if plotflag:
        fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)

        axes[0].plot(trace_idx, linewidth=0.5, alpha=0.5, label='trace_idx')
        axes[0].plot(trace_idx_mf, linewidth=1.0, label='trace_idx_mf (median)')
        for s, e in zip(complex_bursts_dict['starts'], complex_bursts_dict['ends']):
            axes[0].axvspan(s, e, alpha=0.25)
        for loc, pk_val, base in zip(complex_bursts_dict['locs'], complex_bursts_dict['peaks'], complex_bursts_dict['baselines']):
            axes[0].plot([loc, loc], [base, pk_val], linewidth=1)

        axes[0].legend(fontsize=8)
        axes[0].set_ylabel('Vm')

        axes[1].plot(trace_filt, linewidth=0.8, label='trace_filt (LP Vm_sub)')
        if len(troughs) > 0:
            axes[1].plot(troughs, trace_filt[troughs], 'v', markersize=4, alpha=0.7, label='troughs')
        axes[1].legend(fontsize=8)
        axes[1].set_ylabel('LP')

        axes[2].plot(trace_filt_wmf, linewidth=0.8, label='trace_filt_wmf')
        axes[2].axhline(thresh, linestyle='--', linewidth=1, label=f'thresh={thresh:.3g}')
        axes[2].plot(complex_bursts_init, trace_filt_wmf[complex_bursts_init], 'x', alpha=0.8, label='WMF peaks')
        axes[2].legend(fontsize=8)
        axes[2].set_xlabel('Frame')
        axes[2].set_ylabel('WMF')

        plt.tight_layout()
        plt.show()

    return complex_bursts_dict, trace_idx_mf, trace_filt, trace_filt_wmf, templates, thresh

def refine_single_spikes(trace, spike_time, complex_bursts_dict, frame_rate, process_window=300, pnorm=0.25, min_spikes=10, plotflag=False):
    # Refine spike detection with a trace whose complex bursts were removed
    trace_noCS = trace.copy()

    complex_burst_starts = complex_bursts_dict['starts']
    complex_burst_ends = complex_bursts_dict['ends']

    # detect single spikes
    single_spikes, _, _ = detect_burst_SS(spike_time, fr=frame_rate, burst_isi_threshold=14)
    # remove spikes that are within complex bursts
    single_spikes = np.array([spk for spk in single_spikes if not np.any((spk >= complex_burst_starts) & (spk <= complex_burst_ends))])

    # set periods where there is CS to NaN
    for start, end in zip(complex_burst_starts, complex_burst_ends):
        trace_noCS[start:end+1] = np.nan

    # Interpolate NaN values to fill in the gaps
    nans = np.isnan(trace_noCS)
    if np.any(nans) and np.any(~nans):
        not_nans = ~nans
        trace_noCS = np.interp(np.arange(len(trace_noCS)), 
                                    np.arange(len(trace_noCS))[not_nans], 
                                    trace_noCS[not_nans])
    
    # high-pass filter trace_noCS 
    trace_noCS = highpass_filter(trace_noCS, cutoff=20, fs=frame_rate, order=3)
        
    refined_SS, _, _, _ = spike_refinement(trace_noCS, single_spikes, frame_rate, process_window=process_window, pnorm=pnorm, min_spikes=min_spikes)

    # Note refined_SS contains isolated single spikes and simple bursts (No CS)

    if plotflag:
        # plot trace with refined single spikes overlaid, and CS regions shaded
        fig, ax = plt.subplots(figsize=(8,2))
        ax.plot(trace_noCS, label='Original Trace', linewidth=0.5)
        #ax.plot(trace_noCS, label='Trace with CS Removed', color='gray', linewidth=0.5)
        ax.scatter(refined_SS, trace[refined_SS], color='red', label='Refined Single Spikes', s=5)
        for start, end in zip(complex_burst_starts, complex_burst_ends):
            ax.axvspan(start, end, color='yellow', alpha=0.3)
        # remove axis
        ax.set_axis_off()
        ax.margins(x=0)
        plt.show()

    return refined_SS, trace_noCS

def spike_height_calculation(refined_SS, trace_idx, trace_idx_mf, trace_noCS, fr, moving_window_size=20, overlap=10, plotflag=False):
    
    single_spikes, bursts, burst_event_time = detect_burst_SS(refined_SS, fr=fr, burst_isi_threshold=14)
    # simple spikes are single_spikes combined with first spike within bursts, 
    # only for getting the spike height, not the real definition of simple spikes
    simple_spikes = np.sort(np.concatenate([single_spikes, burst_event_time]))

    trace_nospike = trace_noCS.copy()
    # set periods around refined_SS (1 points before and after) to NaN
    for t in refined_SS:
        trace_nospike[max(0, t-1):t+2] = np.nan

    # high-pass filter trace_nospike above 20Hz
    # trace_nospike = interpolate_nan_segment(trace_nospike)
    # trace_nospike = highpass_filter(trace_nospike, cutoff=20, fs=fr, order=5)

    # Calculate the spike height of simple spikes (single_spikes + first spike of bursts)
    simple_spike_heights = []
    for ss in simple_spikes:
        # height is the peak value minus the baseline just before the spike
        baseline_region = trace_idx_mf[max(0, ss-3):ss] # Note we take the baseline from the median filtered trace
        if len(baseline_region) == 0:
            baseline = 0
        else:
            baseline = np.min(baseline_region)
        peak_value = trace_idx[ss]
        height = peak_value - baseline
        simple_spike_heights.append(height)

    simple_spike_heights = np.array(simple_spike_heights)
    simple_spikes_arr = np.array(simple_spikes)

    # Calculate a moving average of simple spike heights
    window_size = int(moving_window_size * fr)  # window in frames
    step_size = int(overlap * fr)    # step size in frames

    moving_avg_heights_simple = []
    moving_avg_times_simple = []
    moving_avg_baseline_noise = []

    # print window size and step size
    print(f"Window size: {window_size}, Step size: {step_size}")

    for start in range(0, len(trace_idx) - window_size + 1, step_size):
        end = start + window_size
        mask = (simple_spikes_arr >= start) & (simple_spikes_arr < end)
        spikes_in_window = simple_spike_heights[mask]
        
        if len(spikes_in_window) > 0:
            avg_height = np.mean(spikes_in_window)
        else:
            avg_height = np.nan

        baseline_std = np.nanstd(trace_nospike[start:end])
        moving_avg_baseline_noise.append(baseline_std)
        
        moving_avg_heights_simple.append(avg_height)
        moving_avg_times_simple.append(start + window_size // 2)

    moving_avg_heights_simple = np.array(moving_avg_heights_simple)
    moving_avg_times_simple = np.array(moving_avg_times_simple)
    moving_avg_baseline_noise = np.array(moving_avg_baseline_noise)

    # Interpolate NaN values in moving_avg_heights_simple
    nans = np.isnan(moving_avg_heights_simple)
    if np.any(nans) and np.any(~nans):
        not_nans = ~nans
        moving_avg_heights_simple = np.interp(np.arange(len(moving_avg_heights_simple)), 
                                            np.arange(len(moving_avg_heights_simple))[not_nans], 
                                            moving_avg_heights_simple[not_nans])

    # Interpolate NaN values in moving_avg_baseline_noise
    nans_noise = np.isnan(moving_avg_baseline_noise)
    if np.any(nans_noise) and np.any(~nans_noise):
        not_nans_noise = ~nans_noise
        moving_avg_baseline_noise = np.interp(np.arange(len(moving_avg_baseline_noise)), 
                                              np.arange(len(moving_avg_baseline_noise))[not_nans_noise], 
                                              moving_avg_baseline_noise[not_nans_noise])

    # Convert times to seconds
    moving_avg_times_simple_sec = moving_avg_times_simple / fr

    # Linear regression for spike heights
    linear_coeffs_simple = np.polyfit(moving_avg_times_simple_sec, moving_avg_heights_simple, 1)
    linear_fit_simple = np.polyval(linear_coeffs_simple, moving_avg_times_simple_sec)
    linear_slope_simple = linear_coeffs_simple[0]

    # Exponential decay fit for spike heights
    def exp_decay(t, a, b, c):
        return a * np.exp(-b * t) + c
    
    p0 = [moving_avg_heights_simple[0] - moving_avg_heights_simple[-1], 0.01, moving_avg_heights_simple[-1]]
    try:
        popt_simple, pcov_simple = curve_fit(exp_decay, moving_avg_times_simple_sec, moving_avg_heights_simple, p0=p0, maxfev=5000)
        exp_fit_simple = exp_decay(moving_avg_times_simple_sec, *popt_simple)
        decay_coeff_simple = popt_simple[1]
        exp_fit_success_simple = True
    except:
        exp_fit_success_simple = False
        decay_coeff_simple = np.nan
        exp_fit_simple = moving_avg_heights_simple  # fallback to raw values

    # Linear regression for baseline noise
    linear_coeffs_noise = np.polyfit(moving_avg_times_simple_sec, moving_avg_baseline_noise, 1)
    linear_fit_noise = np.polyval(linear_coeffs_noise, moving_avg_times_simple_sec)
    linear_slope_noise = linear_coeffs_noise[0]

    # Exponential decay fit for baseline noise
    p0_noise = [moving_avg_baseline_noise[0] - moving_avg_baseline_noise[-1], 0.01, moving_avg_baseline_noise[-1]]
    try:
        popt_noise, pcov_noise = curve_fit(exp_decay, moving_avg_times_simple_sec, moving_avg_baseline_noise, p0=p0_noise, maxfev=5000)
        exp_fit_noise = exp_decay(moving_avg_times_simple_sec, *popt_noise)
        decay_coeff_noise = popt_noise[1]
        exp_fit_success_noise = True
    except:
        exp_fit_success_noise = False
        decay_coeff_noise = np.nan
        exp_fit_noise = moving_avg_baseline_noise  # fallback to raw values

    # Calculate SNR using interpolated (fitted) spike height divided by interpolated baseline noise
    SNR_simple = exp_fit_simple / exp_fit_noise

    # Linear regression for SNR
    linear_coeffs_snr = np.polyfit(moving_avg_times_simple_sec, SNR_simple, 1)
    linear_fit_snr = np.polyval(linear_coeffs_snr, moving_avg_times_simple_sec)
    linear_slope_snr = linear_coeffs_snr[0]

    # Exponential decay fit for SNR
    p0_snr = [SNR_simple[0] - SNR_simple[-1], 0.01, SNR_simple[-1]]
    try:
        popt_snr, pcov_snr = curve_fit(exp_decay, moving_avg_times_simple_sec, SNR_simple, p0=p0_snr, maxfev=5000)
        exp_fit_snr = exp_decay(moving_avg_times_simple_sec, *popt_snr)
        decay_coeff_snr = popt_snr[1]
        exp_fit_success_snr = True
    except:
        exp_fit_success_snr = False
        decay_coeff_snr = np.nan

    # Plot with 3 subplots
    if plotflag:
        fig, axes = plt.subplots(3, 1, figsize=(6, 7), sharex=True)

        # Subplot 1: Spike Heights
        axes[0].plot(simple_spikes_arr / fr, simple_spike_heights, 'o', color='gray', alpha=0.3, markersize=3, label='Simple Spikes')
        axes[0].plot(moving_avg_times_simple_sec, moving_avg_heights_simple, '-', color='blue', linewidth=2, label='Moving Average')
        axes[0].plot(moving_avg_times_simple_sec, linear_fit_simple, '--', color='green', linewidth=2, label=f'Linear (slope={linear_slope_simple:.4f})')
        if exp_fit_success_simple:
            axes[0].plot(moving_avg_times_simple_sec, exp_fit_simple, '--', color='red', linewidth=2, label=f'Exp decay (τ={1/decay_coeff_simple:.2f}s)')
        axes[0].set_ylabel('Spike Height')
        axes[0].set_title('single + first burst spike')
        axes[0].legend(fontsize=7, loc='upper right')

        # Subplot 2: Baseline Noise
        axes[1].plot(moving_avg_times_simple_sec, moving_avg_baseline_noise, '-', color='orange', linewidth=2, label='Baseline Noise')
        axes[1].plot(moving_avg_times_simple_sec, linear_fit_noise, '--', color='green', linewidth=2, label=f'Linear (slope={linear_slope_noise:.4f})')
        if exp_fit_success_noise:
            axes[1].plot(moving_avg_times_simple_sec, exp_fit_noise, '--', color='red', linewidth=2, label=f'Exp decay (τ={1/decay_coeff_noise:.2f}s)')
        axes[1].set_ylabel('Baseline Noise (std)')
        axes[1].legend(fontsize=7, loc='upper right')

        # Subplot 3: SNR
        axes[2].plot(moving_avg_times_simple_sec, SNR_simple, '-', color='purple', linewidth=2, label='SNR')
        axes[2].plot(moving_avg_times_simple_sec, linear_fit_snr, '--', color='green', linewidth=2, label=f'Linear (slope={linear_slope_snr:.4f})')
        if exp_fit_success_snr:
            axes[2].plot(moving_avg_times_simple_sec, exp_fit_snr, '--', color='red', linewidth=2, label=f'Exp decay (τ={1/decay_coeff_snr:.2f}s)')
        axes[2].set_xlabel('Time (s)')
        axes[2].set_ylabel('SNR')
        axes[2].legend(fontsize=7, loc='upper right')

        plt.tight_layout()
        plt.show()

    print(f"Simple Spikes - Linear slope: {linear_slope_simple:.6f}")
    if exp_fit_success_simple:
        print(f"Simple Spikes - Exponential decay coefficient (b): {decay_coeff_simple:.6f}")
        print(f"Simple Spikes - Time constant (τ = 1/b): {1/decay_coeff_simple:.2f} seconds")
    print(f"Baseline Noise - Linear slope: {linear_slope_noise:.6f}")
    if exp_fit_success_noise:
        print(f"Baseline Noise - Exponential decay coefficient (b): {decay_coeff_noise:.6f}")
        print(f"Baseline Noise - Time constant (τ = 1/b): {1/decay_coeff_noise:.2f} seconds")
    print(f"SNR - Linear slope: {linear_slope_snr:.6f}")
    if exp_fit_success_snr:
        print(f"SNR - Exponential decay coefficient (b): {decay_coeff_snr:.6f}")
        print(f"SNR - Time constant (τ = 1/b): {1/decay_coeff_snr:.2f} seconds")

    # interpolate spike heights at each timepoint of the trace from the fitted exponential curve
    #spike_heights_interpolated = exp_decay(np.arange(len(trace_idx)) / fr, *popt_simple) if exp_fit_success_simple else np.full(len(trace_idx), np.nan) 

    # interpolate spike heights at each timepoint of the trace from the fitted linear curve
    spike_heights_interpolated = np.polyval(linear_coeffs_simple, np.arange(len(trace_idx)) / fr)

    # interpolate SNR at each timepoint of the trace from the fitted linear curve
    SNR_interpolated = np.polyval(linear_coeffs_snr, np.arange(len(trace_idx)) / fr)

    return spike_heights_interpolated, SNR_interpolated

def spike_height_calculation2(refined_SS, trace_idx, trace_idx_mf, trace_noCS, fr, moving_window_size=20, overlap=10, plotflag=False, save_path=None, figure_folder=None, session_start_frames=None, pdf=None):
    refined_SS = np.asarray(refined_SS, dtype=np.int64)
    trace_idx = np.asarray(trace_idx, dtype=float)
    trace_idx_mf = np.asarray(trace_idx_mf, dtype=float)
    trace_noCS = np.asarray(trace_noCS, dtype=float)
    n = len(trace_idx)

    if n == 0:
        return np.array([]), np.array([])

    trace_nospike = trace_noCS.copy()
    # set periods around refined_SS (1 points before and after) to NaN
    for t in refined_SS:
        if 0 <= t < n:
            trace_nospike[max(0, t - 1):min(n, t + 2)] = np.nan

    # session boundaries
    if session_start_frames is None or len(session_start_frames) == 0:
        session_starts = [0]
    else:
        session_starts = sorted({int(s) for s in session_start_frames if 0 <= int(s) < n})
        if 0 not in session_starts:
            session_starts = [0] + session_starts
    session_bounds = []
    for i, start in enumerate(session_starts):
        end = session_starts[i + 1] if i + 1 < len(session_starts) else n
        if end > start:
            session_bounds.append((start, end))
    if len(session_bounds) == 0:
        session_bounds = [(0, n)]

    merge_frames = session_starts[1:]

    # Fit exponential decay directly on simple spike heights
    def exp_decay(t, a, b, c):
        return a * np.exp(-b * t) + c

    # Calculate baseline noise with moving windows (same as original)
    window_size = int(moving_window_size * fr)  # window in frames
    step_size = int(overlap * fr)    # step size in frames
    print(f"Window size: {window_size}, Step size: {step_size}")

    spike_heights_interpolated = np.full(n, np.nan)
    SNR_interpolated = np.full(n, np.nan)

    if plotflag:
        fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
        time_axis_sec = np.arange(n) / fr
        axes[0].plot(time_axis_sec, trace_idx, color='k', linewidth=0.5, label='Trace')
        colors = plt.cm.tab10.colors
    else:
        axes = None
        colors = None

    all_simple_spikes = []
    all_baseline_noise = []
    all_snr = []

    for session_idx, (start, end) in enumerate(session_bounds):
        session_label = f"Session {session_idx + 1}"
        color = colors[session_idx % len(colors)] if plotflag else None

        refined_SS_session = refined_SS[(refined_SS >= start) & (refined_SS < end)]
        single_spikes, _, burst_event_time = detect_burst_SS(refined_SS_session, fr=fr, burst_isi_threshold=14)
        if len(single_spikes) == 0 and len(burst_event_time) == 0:
            simple_spikes = np.array([], dtype=np.int64)
        else:
            simple_spikes = np.sort(np.concatenate([single_spikes, burst_event_time])).astype(np.int64)
        all_simple_spikes.append(simple_spikes)

        # Calculate spike heights for this session
        simple_spike_heights = []
        simple_spike_times_sec = []
        for ss in simple_spikes:
            if ss < 0 or ss >= n:
                continue
            if np.isnan(trace_idx[ss]):
                continue
            baseline_region = trace_idx_mf[max(0, ss - 3):ss]
            baseline = np.min(baseline_region) if len(baseline_region) else 0
            peak_value = trace_idx[ss]
            simple_spike_heights.append(peak_value - baseline)
            simple_spike_times_sec.append((ss - start) / fr)

        simple_spike_heights = np.array(simple_spike_heights)
        simple_spike_times_sec = np.array(simple_spike_times_sec)

        exp_fit_success_simple = False
        decay_coeff_simple = np.nan
        popt_simple = None
        if len(simple_spike_heights) >= 2:
            p0 = [simple_spike_heights[0] - simple_spike_heights[-1], 0.01, simple_spike_heights[-1]]
            try:
                popt_simple, _ = curve_fit(exp_decay, simple_spike_times_sec, simple_spike_heights, p0=p0, maxfev=5000)
                decay_coeff_simple = popt_simple[1]
                exp_fit_success_simple = True
            except Exception:
                exp_fit_success_simple = False

        if len(simple_spike_heights) >= 2:
            linear_coeffs_simple = np.polyfit(simple_spike_times_sec, simple_spike_heights, 1)
            linear_slope_simple = linear_coeffs_simple[0]
        elif len(simple_spike_heights) == 1:
            linear_coeffs_simple = np.array([0.0, simple_spike_heights[0]])
            linear_slope_simple = 0.0
        else:
            linear_coeffs_simple = np.array([np.nan, np.nan])
            linear_slope_simple = np.nan

        # Baseline noise per session
        session_len = end - start
        moving_avg_baseline_noise = []
        moving_avg_times_noise = []

        if window_size <= 0 or step_size <= 0 or session_len < window_size:
            baseline_std = np.nanstd(trace_nospike[start:end])
            moving_avg_baseline_noise.append(baseline_std)
            moving_avg_times_noise.append(start + session_len // 2)
        else:
            for win_start in range(start, end - window_size + 1, step_size):
                win_end = win_start + window_size
                baseline_std = np.nanstd(trace_nospike[win_start:win_end])
                moving_avg_baseline_noise.append(baseline_std)
                moving_avg_times_noise.append(win_start + window_size // 2)

        moving_avg_baseline_noise = np.array(moving_avg_baseline_noise)
        moving_avg_times_noise = np.array(moving_avg_times_noise)
        all_baseline_noise.append(moving_avg_baseline_noise)

        nans_noise = np.isnan(moving_avg_baseline_noise)
        if np.any(nans_noise) and np.any(~nans_noise):
            not_nans_noise = ~nans_noise
            moving_avg_baseline_noise = np.interp(np.arange(len(moving_avg_baseline_noise)),
                                                  np.arange(len(moving_avg_baseline_noise))[not_nans_noise],
                                                  moving_avg_baseline_noise[not_nans_noise])

        moving_avg_times_noise_sec = (moving_avg_times_noise - start) / fr
        moving_avg_times_noise_abs_sec = moving_avg_times_noise / fr

        if len(moving_avg_times_noise_sec) >= 2 and np.any(~np.isnan(moving_avg_baseline_noise)):
            linear_coeffs_noise = np.polyfit(moving_avg_times_noise_sec, moving_avg_baseline_noise, 1)
            linear_fit_noise = np.polyval(linear_coeffs_noise, moving_avg_times_noise_sec)
            linear_slope_noise = linear_coeffs_noise[0]
        else:
            linear_coeffs_noise = np.array([np.nan, np.nan])
            linear_fit_noise = moving_avg_baseline_noise
            linear_slope_noise = np.nan

        exp_fit_success_noise = False
        decay_coeff_noise = np.nan
        popt_noise = None
        if len(moving_avg_times_noise_sec) >= 2 and np.any(~np.isnan(moving_avg_baseline_noise)):
            p0_noise = [moving_avg_baseline_noise[0] - moving_avg_baseline_noise[-1], 0.01, moving_avg_baseline_noise[-1]]
            try:
                popt_noise, _ = curve_fit(exp_decay, moving_avg_times_noise_sec, moving_avg_baseline_noise, p0=p0_noise, maxfev=5000)
                exp_fit_noise = exp_decay(moving_avg_times_noise_sec, *popt_noise)
                decay_coeff_noise = popt_noise[1]
                exp_fit_success_noise = True
            except Exception:
                exp_fit_success_noise = False
                exp_fit_noise = moving_avg_baseline_noise
        else:
            exp_fit_noise = moving_avg_baseline_noise

        if len(moving_avg_times_noise_sec) > 0:
            if exp_fit_success_simple and popt_simple is not None:
                simple_fit_at_noise_times = exp_decay(moving_avg_times_noise_sec, *popt_simple)
            else:
                simple_fit_at_noise_times = np.polyval(linear_coeffs_simple, moving_avg_times_noise_sec)
        else:
            simple_fit_at_noise_times = np.array([])

        SNR_simple = simple_fit_at_noise_times / exp_fit_noise if len(exp_fit_noise) > 0 else np.array([])
        all_snr.append(SNR_simple)

        if len(moving_avg_times_noise_sec) >= 2 and len(SNR_simple) == len(moving_avg_times_noise_sec):
            linear_coeffs_snr = np.polyfit(moving_avg_times_noise_sec, SNR_simple, 1)
            linear_fit_snr = np.polyval(linear_coeffs_snr, moving_avg_times_noise_sec)
            linear_slope_snr = linear_coeffs_snr[0]
        else:
            linear_coeffs_snr = np.array([np.nan, np.nan])
            linear_fit_snr = SNR_simple
            linear_slope_snr = np.nan

        exp_fit_success_snr = False
        decay_coeff_snr = np.nan
        popt_snr = None
        if len(moving_avg_times_noise_sec) >= 2 and len(SNR_simple) == len(moving_avg_times_noise_sec):
            p0_snr = [SNR_simple[0] - SNR_simple[-1], 0.01, SNR_simple[-1]]
            try:
                popt_snr, _ = curve_fit(exp_decay, moving_avg_times_noise_sec, SNR_simple, p0=p0_snr, maxfev=5000)
                exp_fit_snr = exp_decay(moving_avg_times_noise_sec, *popt_snr)
                decay_coeff_snr = popt_snr[1]
                exp_fit_success_snr = True
            except Exception:
                exp_fit_success_snr = False
        else:
            exp_fit_snr = SNR_simple

        session_time_rel = (np.arange(start, end) - start) / fr
        session_time_abs = np.arange(start, end) / fr

        if exp_fit_success_simple and popt_simple is not None:
            spike_heights_interpolated[start:end] = exp_decay(session_time_rel, *popt_simple)
        elif len(simple_spike_heights) == 1:
            spike_heights_interpolated[start:end] = simple_spike_heights[0]
        else:
            spike_heights_interpolated[start:end] = np.polyval(linear_coeffs_simple, session_time_rel)

        if np.any(~np.isnan(linear_coeffs_snr)):
            SNR_interpolated[start:end] = np.polyval(linear_coeffs_snr, session_time_rel)
        elif len(SNR_simple) == 1:
            SNR_interpolated[start:end] = SNR_simple[0]

        print(f"{session_label} - Simple Spikes Linear slope: {linear_slope_simple:.6f}")
        if exp_fit_success_simple:
            print(f"{session_label} - Simple Spikes Exp decay (b): {decay_coeff_simple:.6f}")
            print(f"{session_label} - Simple Spikes Tau (1/b): {1/decay_coeff_simple:.2f} seconds")
        print(f"{session_label} - Baseline Noise Linear slope: {linear_slope_noise:.6f}")
        if exp_fit_success_noise:
            print(f"{session_label} - Baseline Noise Exp decay (b): {decay_coeff_noise:.6f}")
            print(f"{session_label} - Baseline Noise Tau (1/b): {1/decay_coeff_noise:.2f} seconds")
        print(f"{session_label} - SNR Linear slope: {linear_slope_snr:.6f}")
        if exp_fit_success_snr:
            print(f"{session_label} - SNR Exp decay (b): {decay_coeff_snr:.6f}")
            print(f"{session_label} - SNR Tau (1/b): {1/decay_coeff_snr:.2f} seconds")

        if plotflag:
            if exp_fit_success_simple and popt_simple is not None:
                axes[0].plot(session_time_abs, exp_decay(session_time_rel, *popt_simple), color=color, linewidth=1.5, label=f'Exp Fit ({session_label})')
            if len(moving_avg_times_noise_abs_sec) > 0:
                axes[1].plot(moving_avg_times_noise_abs_sec, moving_avg_baseline_noise, '-', color=color, linewidth=1.5, label=f'Baseline Noise ({session_label})')
                axes[1].plot(moving_avg_times_noise_abs_sec, linear_fit_noise, '--', color=color, linewidth=1.0, alpha=0.6, label=f'Linear ({session_label})')
                if exp_fit_success_noise and popt_noise is not None:
                    axes[1].plot(session_time_abs, exp_decay(session_time_rel, *popt_noise), ':', color=color, linewidth=1.5, label=f'Exp Fit ({session_label})')
            if len(moving_avg_times_noise_abs_sec) > 0 and len(SNR_simple) > 0:
                axes[2].plot(moving_avg_times_noise_abs_sec, SNR_simple, '-', color=color, linewidth=1.5, label=f'SNR ({session_label})')
                axes[2].plot(moving_avg_times_noise_abs_sec, linear_fit_snr, '--', color=color, linewidth=1.0, alpha=0.6, label=f'Linear ({session_label})')
                if exp_fit_success_snr and popt_snr is not None:
                    axes[2].plot(session_time_abs, exp_decay(session_time_rel, *popt_snr), ':', color=color, linewidth=1.5, label=f'Exp Fit ({session_label})')

    if plotflag:
        if len(all_simple_spikes) > 0:
            simple_spikes_arr = np.sort(np.concatenate(all_simple_spikes)) if any(len(s) > 0 for s in all_simple_spikes) else np.array([], dtype=np.int64)
        else:
            simple_spikes_arr = np.array([], dtype=np.int64)
        if len(simple_spikes_arr) > 0:
            axes[0].scatter(simple_spikes_arr / fr, trace_idx[simple_spikes_arr], color='red', s=3, label='Simple Spikes')

        for i, boundary in enumerate(merge_frames):
            x_line = boundary / fr
            for ax in axes:
                ax.axvline(x_line, color='blue', linestyle='--', linewidth=1, alpha=0.5, label='Session boundary' if i == 0 else None)

        axes[0].set_ylabel('Amplitude')
        axes[1].set_ylabel('Baseline Noise (std)')
        axes[2].set_ylabel('SNR')
        axes[2].set_xlabel('Time (s)')

        def _tight_ylim(ax, data):
            data = np.asarray(data, dtype=float)
            data = data[np.isfinite(data)]
            if data.size == 0:
                return
            y_min = np.min(data)
            y_max = np.max(data)
            if y_min == y_max:
                y_min -= 1.0
                y_max += 1.0
            ax.set_ylim(y_min, y_max)
            ax.margins(y=0)

        time_axis_sec = np.arange(n) / fr
        if len(time_axis_sec) > 0:
            for ax in axes:
                ax.set_xlim(time_axis_sec[0], time_axis_sec[-1])
                ax.margins(x=0)
            _tight_ylim(axes[0], trace_idx)
            noise_vals = np.concatenate([vals for vals in all_baseline_noise if vals.size > 0]) if any(vals.size > 0 for vals in all_baseline_noise) else np.array([])
            snr_vals = np.concatenate([vals for vals in all_snr if vals.size > 0]) if any(vals.size > 0 for vals in all_snr) else np.array([])
            _tight_ylim(axes[1], noise_vals)
            _tight_ylim(axes[2], snr_vals)

        axes[0].legend(fontsize=7, loc='upper right')
        axes[1].legend(fontsize=7, loc='upper right')
        axes[2].legend(fontsize=7, loc='upper right')

        plt.tight_layout()
        if save_path is None and figure_folder is not None:
            os.makedirs(figure_folder, exist_ok=True)
            save_path = os.path.join(figure_folder, "spike_height_calculation2.pdf")
        if pdf is not None:
            pdf.savefig(fig, bbox_inches='tight')
        if save_path:
            fig.savefig(save_path, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")
        plt.show()

    return spike_heights_interpolated, SNR_interpolated

def detect_complex_spikes(trace, complex_bursts_dict, spike_heights_interpolated, threshold=0.25, plotflag=False):
    """
    Detect individual spikes within each complex burst (CS).
    
    For each CS window, find local peaks, calculate their heights relative to 
    the interpolated spike height, and return spikes that cross the threshold.
    
    Parameters:
    -----------
    trace : 1D array
        Original neural trace
    complex_bursts_dict : dict
        Dictionary containing complex burst information with 'starts' and 'ends' keys
    spike_heights_interpolated : 1D array
        Interpolated spike heights at each timepoint (same length as trace)
    threshold : float
        Threshold for spike detection (default 0.25). Local spike height divided by
        interpolated spike height must exceed this value.
    plotflag : bool
        Whether to plot the results (default False)
    
    Returns:
    --------
    CS_spikes : list of lists
        Each inner list contains spike times (frame indices) within a complex spike.
        The outer list corresponds to each CS in order.
    """
    complex_burst_starts = complex_bursts_dict['starts']
    complex_burst_ends = complex_bursts_dict['ends']
    
    CS_spikes = []

    trace_mf = complex_bursts_dict['trace_mf']
    
    for start, end in zip(complex_burst_starts, complex_burst_ends):
        # Extract the trace segment for this CS
        cs_segment = trace[start:end+1]
        
        # Find local peaks within this CS window
        local_peaks, _ = signal.find_peaks(cs_segment)
        
        spikes_in_cs = []
        
        for pk in local_peaks:
            # Convert to absolute index
            abs_pk = start + pk
            
            # Calculate local spike height
            # Baseline: minimum of 3 points before the peak, use median filtered trace
            baseline_start = max(0, abs_pk - 2)
            baseline_region = trace_mf[baseline_start:abs_pk]
            if len(baseline_region) == 0:
                baseline = trace_mf[abs_pk]
            else:
                baseline = np.min(baseline_region)
            
            peak_value = trace[abs_pk]
            local_spike_height = peak_value - baseline
            
            # Get mean interpolated spike height within this CS window
            mean_spike_height = np.nanmean(spike_heights_interpolated[start:end+1])
            
            # Avoid division by zero or nan
            if np.isnan(mean_spike_height) or mean_spike_height == 0:
                continue
            
            # Calculate ratio
            ratio = local_spike_height / mean_spike_height
            
            # Check if ratio crosses threshold
            if ratio >= threshold:
                spikes_in_cs.append(abs_pk)
        
        CS_spikes.append(spikes_in_cs)
    
    if plotflag:
        fig, ax = plt.subplots(figsize=(8, 3))
        
        # Plot trace
        ax.plot(trace, color='gray', linewidth=0.5, label='Trace')
        
        # Highlight CS regions
        for start, end in zip(complex_burst_starts, complex_burst_ends):
            ax.axvspan(start, end, color='yellow', alpha=0.3)
        
        # Plot detected spikes within CS
        all_cs_spikes = [spk for cs in CS_spikes for spk in cs]
        if len(all_cs_spikes) > 0:
            all_cs_spikes = np.array(all_cs_spikes)
            ax.scatter(all_cs_spikes, trace[all_cs_spikes], color='red', s=20, zorder=5, label='CS Spikes')
        
        ax.set_xlabel('Frame')
        ax.set_ylabel('Amplitude')
        ax.set_title(f'Complex Spikes Detection (threshold={threshold})')
        ax.legend(loc='upper right', fontsize=8)
        ax.margins(x=0)
        
        plt.tight_layout()
        plt.show()
    
    return CS_spikes

def refine_all_spikes(complex_bursts_dict, CS_spikes, refined_SS):
    # Post-process CS_spikes: 
    # - If a CS burst has only 1 spike, move it to refined_SS and remove the burst
    # - If a CS burst has 0 spikes, just remove the burst

    # Create lists to track which bursts to keep and spikes to move
    refined_SS = np.asarray(refined_SS, dtype=np.int64)
    original_burst_count = len(CS_spikes)
    valid_CS_spikes = []
    spikes_to_move_to_SS = []
    valid_burst_indices = []

    for i, cs_burst_spikes in enumerate(CS_spikes):
        n_spikes = len(cs_burst_spikes)
        if n_spikes > 1:
            # Keep this burst and its spikes
            valid_CS_spikes.append(cs_burst_spikes)
            valid_burst_indices.append(i)
        elif n_spikes == 1:
            # Move the single spike to refined_SS
            spikes_to_move_to_SS.extend(cs_burst_spikes)
        # If n_spikes == 0, just skip (don't add to valid lists)

    # Update CS_spikes
    CS_spikes = valid_CS_spikes

    # Update complex_bursts_dict by keeping only valid bursts
    starts = np.asarray(complex_bursts_dict['starts'])
    ends = np.asarray(complex_bursts_dict['ends'])
    locs = np.asarray(complex_bursts_dict['locs'])
    peaks = np.asarray(complex_bursts_dict['peaks'])
    baselines = np.asarray(complex_bursts_dict['baselines'])
    durations_ms = np.asarray(complex_bursts_dict['durations_ms'])
    amplitudes = np.asarray(complex_bursts_dict['amplitudes'])
    if len(valid_burst_indices) > 0:
        complex_bursts_dict['starts'] = starts[valid_burst_indices]
        complex_bursts_dict['ends'] = ends[valid_burst_indices]
        complex_bursts_dict['locs'] = locs[valid_burst_indices]
        complex_bursts_dict['peaks'] = peaks[valid_burst_indices]
        complex_bursts_dict['baselines'] = baselines[valid_burst_indices]
        complex_bursts_dict['durations_ms'] = durations_ms[valid_burst_indices]
        complex_bursts_dict['amplitudes'] = amplitudes[valid_burst_indices]
    else:
        complex_bursts_dict['starts'] = starts[:0]
        complex_bursts_dict['ends'] = ends[:0]
        complex_bursts_dict['locs'] = locs[:0]
        complex_bursts_dict['peaks'] = peaks[:0]
        complex_bursts_dict['baselines'] = baselines[:0]
        complex_bursts_dict['durations_ms'] = durations_ms[:0]
        complex_bursts_dict['amplitudes'] = amplitudes[:0]

    # Add single spikes from removed bursts to refined_SS
    if len(spikes_to_move_to_SS) > 0:
        refined_SS = np.sort(np.concatenate([refined_SS, np.array(spikes_to_move_to_SS)]))

    removed_count = original_burst_count - len(valid_CS_spikes)
    print(f"Removed {removed_count} invalid CS bursts")
    print(f"Moved {len(spikes_to_move_to_SS)} spikes from single-spike bursts to refined_SS")
    print(f"Remaining CS bursts: {len(CS_spikes)}")

    if len(CS_spikes) > 0 and any(len(cs) > 0 for cs in CS_spikes):
        all_CS_spikes = np.sort(np.concatenate(CS_spikes))
    else:
        all_CS_spikes = np.array([], dtype=np.int64)

    if len(refined_SS) > 0 or len(all_CS_spikes) > 0:
        all_spikes = np.sort(np.concatenate([refined_SS, all_CS_spikes]))
    else:
        all_spikes = np.array([], dtype=np.int64)
    return complex_bursts_dict, refined_SS, all_CS_spikes, all_spikes

# def refine_complex_bursts(trace, spike_heights_interpolated)



def plot_trace_with_spikes_pdf(trace, refined_SS, CS_spikes, complex_bursts_dict, fr, 
                                segment_duration=10, rows_per_page=20, save_path=None):
    """
    Plot trace with spikes using matplotlib, breaking into segments row by row.
    Saves as multi-page PDF if needed.
    
    Parameters:
    -----------
    trace : 1D array
        Neural trace
    refined_SS : array
        Single spike times (frame indices)
    CS_spikes : list of lists
        Complex spike times for each CS burst
    complex_bursts_dict : dict
        Dictionary with 'starts' and 'ends' keys
    fr : float
        Frame rate in Hz
    segment_duration : float
        Duration of each segment in seconds (default 10)
    rows_per_page : int
        Number of rows (segments) per PDF page (default 20)
    save_path : str
        Path to save PDF file (default None)
    """
    # Flatten CS_spikes
    all_CS_spikes = np.sort(np.concatenate(CS_spikes)) if len(CS_spikes) > 0 and any(len(cs) > 0 for cs in CS_spikes) else np.array([])
    
    # Calculate segment parameters
    segment_frames = int(segment_duration * fr)
    n_segments = int(np.ceil(len(trace) / segment_frames))
    
    # Get y-axis range for shared y
    y_min = np.nanmin(trace)
    y_max = np.nanmax(trace)
    y_margin = 0.05 * (y_max - y_min)
    y_range = [y_min - y_margin, y_max + y_margin]
    
    complex_burst_starts = complex_bursts_dict['starts']
    complex_burst_ends = complex_bursts_dict['ends']
    
    # Calculate number of pages
    n_pages = int(np.ceil(n_segments / rows_per_page))
    
    if save_path:
        pdf = PdfPages(save_path)
    
    for page_idx in range(n_pages):
        # Calculate which segments are on this page
        start_seg = page_idx * rows_per_page
        end_seg = min((page_idx + 1) * rows_per_page, n_segments)
        n_rows_this_page = end_seg - start_seg
        
        # Create figure for this page - taller rows (0.8 inch per row instead of 0.5)
        fig, axes = plt.subplots(n_rows_this_page, 1, figsize=(12, 0.8 * n_rows_this_page), sharex=False, sharey=True)
        if n_rows_this_page == 1:
            axes = [axes]
        
        for row_idx, seg_idx in enumerate(range(start_seg, end_seg)):
            ax = axes[row_idx]
            
            start_frame = seg_idx * segment_frames
            end_frame = min((seg_idx + 1) * segment_frames, len(trace))
            
            start_time = start_frame / fr
            end_time = end_frame / fr
            
            # Get trace segment
            trace_segment = trace[start_frame:end_frame]
            x_frames = np.arange(start_frame, end_frame)
            
            # Plot trace
            ax.plot(x_frames, trace_segment, color='gray', linewidth=0.5, label='Trace' if seg_idx == 0 else None)
            
            # Highlight CS regions
            for cs_start, cs_end in zip(complex_burst_starts, complex_burst_ends):
                if cs_end >= start_frame and cs_start <= end_frame:
                    # Clamp to segment boundaries
                    rect_start = max(cs_start, start_frame)
                    rect_end = min(cs_end, end_frame)
                    ax.axvspan(rect_start, rect_end, color='yellow', alpha=0.3, linewidth=0)
                    # Add vertical lines at start and end
                    # if cs_start >= start_frame and cs_start <= end_frame:
                    #     ax.axvline(x=cs_start, color='orange', linewidth=1.5)
                    # if cs_end >= start_frame and cs_end <= end_frame:
                    #     ax.axvline(x=cs_end, color='orange', linewidth=1.5)
            
            # Plot refined_SS (single spikes) - blue
            ss_in_segment = refined_SS[(refined_SS >= start_frame) & (refined_SS < end_frame)]
            if len(ss_in_segment) > 0:
                ax.plot(ss_in_segment, trace[ss_in_segment], 'o', color='blue', markersize=1, 
                        label='Simple Spikes' if seg_idx == 0 else None)
            
            # Plot CS_spikes (complex spike components) - red
            cs_in_segment = all_CS_spikes[(all_CS_spikes >= start_frame) & (all_CS_spikes < end_frame)]
            if len(cs_in_segment) > 0:
                cs_in_segment = cs_in_segment.astype(int)
                ax.plot(cs_in_segment, trace[cs_in_segment], 'o', color='red', markersize=1,
                        label='CS Spikes' if seg_idx == 0 else None)
            
            # Set axis properties
            ax.set_ylim(y_range)
            ax.set_xlim(start_frame, end_frame)
            ax.set_yticks([])
            ax.set_xticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            
            # Add time labels
            ax.text(start_frame, y_range[1], f'{start_time:.1f}s', fontsize=8, 
                    ha='left', va='top')
            ax.text(end_frame, y_range[1], f'{end_time:.1f}s', fontsize=8, 
                    ha='right', va='top')
        
        # Add legend on first page only
        if page_idx == 0:
            # Create legend handles manually
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color='gray', linewidth=0.5, label='Trace'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=5, label='Single Spikes'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=5, label='CS Spikes'),
                Line2D([0], [0], color='yellow', linewidth=8, alpha=0.5, label='CS Regions'),
            ]
            # Place legend inside the first subplot at the top
            axes[0].legend(handles=legend_elements, loc='upper right', ncol=4, fontsize=7, framealpha=0.8)
        
        # Reduce vertical gaps between subplots
        plt.subplots_adjust(hspace=0.02, top=0.98, bottom=0.02)
        
        if save_path:
            pdf.savefig(fig, bbox_inches='tight')
        
        plt.close(fig)
    
    if save_path:
        pdf.close()
        print(f"Saved to {save_path}")
    
    print(f"Created {n_pages} pages with {n_segments} segments total")


def plot_trace_with_bursts_pdf(trace_snr, vm, simple_spikes, complex_spikes, complex_bursts, fr,
                               segment_duration=10, rows_per_page=20, save_path=None, pdf=None,
                               simple_spike_color='#026C80', complex_spike_color='#EE9B00'):
    """
    Plot trace_snr and vm with simple/complex spikes and complex burst regions.
    Saves as multi-page PDF if needed.

    Parameters:
    -----------
    trace_snr : 1D array
        SNR-normalized trace
    vm : 1D array
        Subthreshold trace
    simple_spikes : array-like
        Simple spike times (frame indices)
    complex_spikes : array-like
        Complex spike times (frame indices)
    complex_bursts : list or dict
        List of (start, end) tuples or dict with 'starts'/'ends'
    fr : float
        Frame rate in Hz
    segment_duration : float
        Duration of each segment in seconds (default 10)
    rows_per_page : int
        Number of rows (segments) per PDF page (default 20)
    save_path : str
        Path to save PDF file (default None)
    """
    trace_snr = np.asarray(trace_snr, dtype=float)
    vm = np.asarray(vm, dtype=float)
    n = len(trace_snr)

    simple_spikes = np.asarray(simple_spikes, dtype=np.int64)
    complex_spikes = np.asarray(complex_spikes, dtype=np.int64)

    if isinstance(complex_bursts, dict):
        burst_starts = np.asarray(complex_bursts.get('starts', []), dtype=np.int64)
        burst_ends = np.asarray(complex_bursts.get('ends', []), dtype=np.int64)
    else:
        burst_starts = np.array([b[0] for b in complex_bursts], dtype=np.int64) if len(complex_bursts) > 0 else np.array([], dtype=np.int64)
        burst_ends = np.array([b[1] for b in complex_bursts], dtype=np.int64) if len(complex_bursts) > 0 else np.array([], dtype=np.int64)

    segment_frames = int(segment_duration * fr)
    n_segments = int(np.ceil(n / segment_frames))
    n_pages = int(np.ceil(n_segments / rows_per_page))

    y_min = np.nanmin([np.nanmin(trace_snr), np.nanmin(vm)])
    y_max = np.nanmax([np.nanmax(trace_snr), np.nanmax(vm)])
    y_margin = 0.05 * (y_max - y_min) if y_max > y_min else 1.0
    y_range = [y_min - y_margin, y_max + y_margin]

    owns_pdf = False
    if pdf is None and save_path:
        pdf = PdfPages(save_path)
        owns_pdf = True

    for page_idx in range(n_pages):
        start_seg = page_idx * rows_per_page
        end_seg = min((page_idx + 1) * rows_per_page, n_segments)
        n_rows_this_page = end_seg - start_seg

        fig, axes = plt.subplots(n_rows_this_page, 1, figsize=(12, 0.8 * n_rows_this_page), sharex=False, sharey=True)
        if n_rows_this_page == 1:
            axes = [axes]

        for row_idx, seg_idx in enumerate(range(start_seg, end_seg)):
            ax = axes[row_idx]
            start_frame = seg_idx * segment_frames
            end_frame = min((seg_idx + 1) * segment_frames, n)

            start_time = start_frame / fr
            end_time = end_frame / fr

            x_frames = np.arange(start_frame, end_frame)
            trace_segment = trace_snr[start_frame:end_frame]
            vm_segment = vm[start_frame:end_frame]

            ax.plot(x_frames, trace_segment, color='gray', linewidth=0.5, label='Trace' if seg_idx == 0 else None)
            ax.plot(x_frames, vm_segment, color='black', linewidth=0.8, label='Vm' if seg_idx == 0 else None)

            for cs_start, cs_end in zip(burst_starts, burst_ends):
                if cs_end >= start_frame and cs_start <= end_frame:
                    rect_start = max(cs_start, start_frame)
                    rect_end = min(cs_end, end_frame)
                    ax.axvspan(rect_start, rect_end, color='yellow', alpha=0.3, linewidth=0)

            ss_in_segment = simple_spikes[(simple_spikes >= start_frame) & (simple_spikes < end_frame)]
            if len(ss_in_segment) > 0:
                ax.plot(ss_in_segment, trace_snr[ss_in_segment], 'o', color=simple_spike_color, markersize=1,
                        label='Simple Spikes' if seg_idx == 0 else None)

            cs_in_segment = complex_spikes[(complex_spikes >= start_frame) & (complex_spikes < end_frame)]
            if len(cs_in_segment) > 0:
                ax.plot(cs_in_segment, trace_snr[cs_in_segment], 'o', color=complex_spike_color, markersize=1,
                        label='Complex Spikes' if seg_idx == 0 else None)

            ax.set_ylim(y_range)
            ax.set_xlim(start_frame, end_frame)
            ax.set_yticks([])
            ax.set_xticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)

            ax.text(start_frame, y_range[1], f'{start_time:.1f}s', fontsize=8, ha='left', va='top')
            ax.text(end_frame, y_range[1], f'{end_time:.1f}s', fontsize=8, ha='right', va='top')

        if page_idx == 0:
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color='gray', linewidth=0.5, label='Trace'),
                Line2D([0], [0], color='black', linewidth=0.8, label='Vm'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor=simple_spike_color, markersize=5, label='Simple Spikes'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor=complex_spike_color, markersize=5, label='Complex Spikes'),
                Line2D([0], [0], color='yellow', linewidth=8, alpha=0.5, label='Complex Bursts'),
            ]
            axes[0].legend(handles=legend_elements, loc='upper right', ncol=3, fontsize=7, framealpha=0.8)

        plt.subplots_adjust(hspace=0.02, top=0.98, bottom=0.02)

        if pdf is not None:
            pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    if owns_pdf:
        pdf.close()
        print(f"Saved to {save_path}")

    print(f"Created {n_pages} pages with {n_segments} segments total")


def plot_burst_metrics_pdf(burst_metrics, pdf=None, save_path=None, figsize=(6, 3),
                           simple_spike_color='#026C80', complex_spike_color='#EE9B00'):
    """
    Plot burst amplitude/duration and AUC vs spike count with mean ± s.e.m. lines.
    """
    burst_n_spikes = np.array([b['n_spikes'] for b in burst_metrics], dtype=float)
    burst_amp = np.array([b['peak_amp'] for b in burst_metrics], dtype=float)
    burst_dur = np.array([b['duration_ms'] for b in burst_metrics], dtype=float)
    burst_auc = np.array([b['auc'] for b in burst_metrics], dtype=float)
    burst_is_complex = np.array([b['is_complex'] for b in burst_metrics], dtype=bool)

    def _mean_sem(vals):
        vals = np.asarray(vals, dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            return np.nan, np.nan
        return np.nanmean(vals), np.nanstd(vals) / np.sqrt(len(vals))

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    is_single = np.array([b.get('is_single', False) for b in burst_metrics], dtype=bool)
    non_single_mask = ~is_single
    if len(burst_amp) > 0:
        axes[0].scatter(burst_dur[(~burst_is_complex) & non_single_mask], burst_amp[(~burst_is_complex) & non_single_mask],
                        color=simple_spike_color, s=5, alpha=0.5, label='Simple Bursts')
        axes[0].scatter(burst_dur[burst_is_complex & non_single_mask], burst_amp[burst_is_complex & non_single_mask],
                        color=complex_spike_color, s=5, alpha=0.5, label='Complex Bursts')
    axes[0].set_xlabel('Duration (ms)')
    axes[0].set_ylabel('Peak amplitude')
    axes[0].legend(fontsize=7, loc='upper right')

    if len(burst_auc) > 0:
        axes[1].scatter(burst_n_spikes[~burst_is_complex], burst_auc[~burst_is_complex], color=simple_spike_color, s=5, alpha=0.5, label='Simple Bursts + Single Spikes')
        axes[1].scatter(burst_n_spikes[burst_is_complex], burst_auc[burst_is_complex], color=complex_spike_color, s=5, alpha=0.5, label='Complex Bursts')

    unique_counts = np.unique(burst_n_spikes[np.isfinite(burst_n_spikes)])
    means_simple = []
    sems_simple = []
    means_complex = []
    sems_complex = []
    means_all = []
    sems_all = []
    for count in unique_counts:
        simple_vals = burst_auc[(burst_n_spikes == count) & (~burst_is_complex)]
        complex_vals = burst_auc[(burst_n_spikes == count) & (burst_is_complex)]
        all_vals = burst_auc[burst_n_spikes == count]
        mean_s, sem_s = _mean_sem(simple_vals)
        mean_c, sem_c = _mean_sem(complex_vals)
        mean_a, sem_a = _mean_sem(all_vals)
        means_simple.append(mean_s)
        sems_simple.append(sem_s)
        means_complex.append(mean_c)
        sems_complex.append(sem_c)
        means_all.append(mean_a)
        sems_all.append(sem_a)

    unique_counts = np.array(unique_counts, dtype=float)
    means_simple = np.array(means_simple, dtype=float)
    sems_simple = np.array(sems_simple, dtype=float)
    means_complex = np.array(means_complex, dtype=float)
    sems_complex = np.array(sems_complex, dtype=float)
    means_all = np.array(means_all, dtype=float)
    sems_all = np.array(sems_all, dtype=float)

    if len(unique_counts) > 0:
        axes[1].errorbar(unique_counts, means_simple, yerr=sems_simple, color=simple_spike_color, linestyle='-', marker='o', markersize=4, capsize=3, label='Simple mean ± s.e.m.')
        axes[1].errorbar(unique_counts, means_complex, yerr=sems_complex, color=complex_spike_color, linestyle='-', marker='o', markersize=4, capsize=3, label='Complex mean ± s.e.m.')
        axes[1].errorbar(unique_counts, means_all, yerr=sems_all, color='black', linestyle='-', marker='o', markersize=4, capsize=3, label='All mean ± s.e.m.')

    axes[1].set_xlabel('Number of spikes')
    axes[1].set_ylabel('AUC')
    axes[1].legend(fontsize=7, loc='upper right')

    plt.tight_layout()

    owns_pdf = False
    if pdf is None and save_path:
        pdf = PdfPages(save_path)
        owns_pdf = True

    if pdf is not None:
        pdf.savefig(fig, bbox_inches='tight')

    plt.close(fig)

    if owns_pdf:
        pdf.close()
        print(f"Saved to {save_path}")
