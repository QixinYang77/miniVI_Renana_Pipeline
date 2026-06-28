"""Manual all-spike detection primitives."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy import signal
from scipy.ndimage import median_filter

_THIS_DIR = Path(__file__).resolve().parent
_ANALYSIS_ROOT = _THIS_DIR.parent
_UTILS_DIR = _ANALYSIS_ROOT / "utils"
if str(_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_UTILS_DIR))

from spike_detection import highpass_filter, interpolate_nan_segment  # noqa: E402

DEFAULT_BASELINE_WINDOW_S = 10.0
DEFAULT_SEGMENT_DURATION_S = 120.0
DEFAULT_HIGHPASS_HZ = 50.0
DEFAULT_THRESHOLD_MAD = 5.0
DEFAULT_SPIKE_BASELINE_REMOVE_ENABLED = False
DEFAULT_SPIKE_BASELINE_WINDOW_MS = 51.0
DEFAULT_FIND_PEAK_DISTANCE = 2
DEFAULT_CB_BASELINE_WINDOW_S = 1.0
DEFAULT_CB_REMOVE_SPIKES_FOR_VM = False
DEFAULT_VM_MEDIAN_WINDOW_MS = 21.0
DEFAULT_VM_CROSSING_THRESHOLD = 0.1
DEFAULT_CB_AMP_THRESHOLD = 0.6
DEFAULT_CB_DURATION_THRESHOLD_MS = 20.0
DEFAULT_CB_MIN_SPIKES = 0
DEFAULT_CB_ISI_THRESHOLD_MS = 20.0
DEFAULT_CB_REQUIRE_MIN_ISI = False
DEFAULT_CB_SPIKE_HEIGHT_MIN_ISOLATED_SPIKES = 5
DEFAULT_CB_INCLUDE_FIRST_BURST_SPIKE_FOR_SPIKE_HEIGHT = False
DEFAULT_CB_REFINE_ONSET = True
DEFAULT_CB_ONSET_THRESHOLD = 0.05
DEFAULT_CB_OFFSET_THRESHOLD = 0.05
DEFAULT_CB_MAX_ONSET_LEAD_MS = 200.0
DEFAULT_SECOND_ROUND_SS_THRESHOLD_MAD = 5.0
DEFAULT_SECOND_ROUND_CS_THRESHOLD_MAD = 4.0
DEFAULT_SECOND_ROUND_REFINE_SIMPLE_SPIKES = False
DEFAULT_SECOND_ROUND_SIMPLE_SPIKE_MIN_HEIGHT_FRACTION = 0.8
DEFAULT_SECOND_ROUND_CB_AMP_THRESHOLD = 0.6
DEFAULT_SECOND_ROUND_CB_MIN_SPIKES = 2
DEFAULT_SNR_SPIKE_MASK_MS = 3.0
DEFAULT_PLATEAU_BASELINE_WINDOW_S = 10.0
DEFAULT_PLATEAU_VM_MEDIAN_WINDOW_MS = 21.0
DEFAULT_PLATEAU_VM_CROSSING_THRESHOLD = 0.1
DEFAULT_PLATEAU_ONSET_THRESHOLD = 0.05
DEFAULT_PLATEAU_OFFSET_THRESHOLD = 0.05
DEFAULT_PLATEAU_AMP_THRESHOLD = 0.6
DEFAULT_PLATEAU_DURATION_THRESHOLD_MS = 100.0
DEFAULT_PLATEAU_MIN_SPIKES = 0
DEFAULT_PLATEAU_PEAK_FRACTION_THRESHOLD = 0.8
DEFAULT_PLATEAU_PEAK_FRACTION_DURATION_MS = 20.0


def _safe_positive(value: Any, fallback: float, *, min_value: float) -> float:
    try:
        out = float(value)
    except Exception:
        return float(fallback)
    if not np.isfinite(out) or out < min_value:
        return float(fallback)
    return float(out)


def normalize_params(
    *,
    baseline_window_s: Any = DEFAULT_BASELINE_WINDOW_S,
    segment_duration_s: Any = DEFAULT_SEGMENT_DURATION_S,
    highpass_hz: Any = DEFAULT_HIGHPASS_HZ,
    threshold_mad: Any = DEFAULT_THRESHOLD_MAD,
    spike_baseline_remove_enabled: Any = DEFAULT_SPIKE_BASELINE_REMOVE_ENABLED,
    spike_baseline_window_ms: Any = DEFAULT_SPIKE_BASELINE_WINDOW_MS,
) -> dict[str, float]:
    return {
        "baseline_window_s": _safe_positive(baseline_window_s, DEFAULT_BASELINE_WINDOW_S, min_value=0.001),
        "segment_duration_s": _safe_positive(segment_duration_s, DEFAULT_SEGMENT_DURATION_S, min_value=0.5),
        "highpass_hz": _safe_positive(highpass_hz, DEFAULT_HIGHPASS_HZ, min_value=0.0),
        "threshold_mad": _safe_positive(threshold_mad, DEFAULT_THRESHOLD_MAD, min_value=0.0),
        "spike_baseline_remove_enabled": _safe_bool(
            spike_baseline_remove_enabled,
            DEFAULT_SPIKE_BASELINE_REMOVE_ENABLED,
        ),
        "spike_baseline_window_ms": _safe_positive(
            spike_baseline_window_ms,
            DEFAULT_SPIKE_BASELINE_WINDOW_MS,
            min_value=0.001,
        ),
    }


def normalize_cb_params(
    *,
    cb_baseline_window_s: Any = DEFAULT_CB_BASELINE_WINDOW_S,
    remove_spikes_for_vm: Any = DEFAULT_CB_REMOVE_SPIKES_FOR_VM,
    vm_median_window_ms: Any = DEFAULT_VM_MEDIAN_WINDOW_MS,
    vm_crossing_threshold: Any = DEFAULT_VM_CROSSING_THRESHOLD,
    cb_amp_threshold: Any = DEFAULT_CB_AMP_THRESHOLD,
    cb_duration_threshold_ms: Any = DEFAULT_CB_DURATION_THRESHOLD_MS,
    cb_min_spikes: Any = DEFAULT_CB_MIN_SPIKES,
    cb_isi_threshold_ms: Any = DEFAULT_CB_ISI_THRESHOLD_MS,
    cb_require_min_isi: Any = DEFAULT_CB_REQUIRE_MIN_ISI,
    spike_height_min_isolated_spikes: Any = DEFAULT_CB_SPIKE_HEIGHT_MIN_ISOLATED_SPIKES,
    include_first_burst_spike_for_spike_height: Any = DEFAULT_CB_INCLUDE_FIRST_BURST_SPIKE_FOR_SPIKE_HEIGHT,
    refine_cb_onset: Any = DEFAULT_CB_REFINE_ONSET,
    cb_onset_threshold: Any = DEFAULT_CB_ONSET_THRESHOLD,
    cb_offset_threshold: Any = DEFAULT_CB_OFFSET_THRESHOLD,
    cb_max_onset_lead_ms: Any = DEFAULT_CB_MAX_ONSET_LEAD_MS,
) -> dict[str, float]:
    min_spikes = _safe_positive(cb_min_spikes, DEFAULT_CB_MIN_SPIKES, min_value=0.0)
    min_height_spikes = _safe_positive(
        spike_height_min_isolated_spikes,
        DEFAULT_CB_SPIKE_HEIGHT_MIN_ISOLATED_SPIKES,
        min_value=0.0,
    )
    return {
        "cb_baseline_window_s": _safe_positive(cb_baseline_window_s, DEFAULT_CB_BASELINE_WINDOW_S, min_value=0.001),
        "remove_spikes_for_vm": _safe_bool(remove_spikes_for_vm, DEFAULT_CB_REMOVE_SPIKES_FOR_VM),
        "vm_median_window_ms": _safe_positive(vm_median_window_ms, DEFAULT_VM_MEDIAN_WINDOW_MS, min_value=0.001),
        "vm_crossing_threshold": float(_safe_float(vm_crossing_threshold, DEFAULT_VM_CROSSING_THRESHOLD)),
        "cb_amp_threshold": float(_safe_float(cb_amp_threshold, DEFAULT_CB_AMP_THRESHOLD)),
        "cb_duration_threshold_ms": _safe_positive(
            cb_duration_threshold_ms,
            DEFAULT_CB_DURATION_THRESHOLD_MS,
            min_value=0.0,
        ),
        "cb_min_spikes": float(max(0, int(round(min_spikes)))),
        "cb_isi_threshold_ms": _safe_positive(cb_isi_threshold_ms, DEFAULT_CB_ISI_THRESHOLD_MS, min_value=0.0),
        "cb_require_min_isi": _safe_bool(cb_require_min_isi, DEFAULT_CB_REQUIRE_MIN_ISI),
        "spike_height_min_isolated_spikes": float(max(0, int(round(min_height_spikes)))),
        "include_first_burst_spike_for_spike_height": _safe_bool(
            include_first_burst_spike_for_spike_height,
            DEFAULT_CB_INCLUDE_FIRST_BURST_SPIKE_FOR_SPIKE_HEIGHT,
        ),
        "refine_cb_onset": _safe_bool(refine_cb_onset, DEFAULT_CB_REFINE_ONSET),
        "cb_onset_threshold": float(_safe_float(cb_onset_threshold, DEFAULT_CB_ONSET_THRESHOLD)),
        "cb_offset_threshold": float(_safe_float(cb_offset_threshold, DEFAULT_CB_OFFSET_THRESHOLD)),
        "cb_max_onset_lead_ms": _safe_positive(
            cb_max_onset_lead_ms,
            DEFAULT_CB_MAX_ONSET_LEAD_MS,
            min_value=0.0,
        ),
    }


def normalize_second_round_params(
    *,
    ss_threshold_mad: Any = DEFAULT_SECOND_ROUND_SS_THRESHOLD_MAD,
    cs_threshold_mad: Any = DEFAULT_SECOND_ROUND_CS_THRESHOLD_MAD,
    refine_simple_spikes_by_height: Any = DEFAULT_SECOND_ROUND_REFINE_SIMPLE_SPIKES,
    simple_spike_min_height_fraction: Any = DEFAULT_SECOND_ROUND_SIMPLE_SPIKE_MIN_HEIGHT_FRACTION,
) -> dict[str, float]:
    return {
        "ss_threshold_mad": _safe_positive(
            ss_threshold_mad,
            DEFAULT_SECOND_ROUND_SS_THRESHOLD_MAD,
            min_value=0.0,
        ),
        "cs_threshold_mad": _safe_positive(
            cs_threshold_mad,
            DEFAULT_SECOND_ROUND_CS_THRESHOLD_MAD,
            min_value=0.0,
        ),
        "refine_simple_spikes_by_height": _safe_bool(
            refine_simple_spikes_by_height,
            DEFAULT_SECOND_ROUND_REFINE_SIMPLE_SPIKES,
        ),
        "simple_spike_min_height_fraction": _safe_positive(
            simple_spike_min_height_fraction,
            DEFAULT_SECOND_ROUND_SIMPLE_SPIKE_MIN_HEIGHT_FRACTION,
            min_value=0.0,
        ),
    }


def normalize_second_round_cb_params(
    *,
    cb_amp_threshold: Any = DEFAULT_SECOND_ROUND_CB_AMP_THRESHOLD,
    cb_min_spikes: Any = DEFAULT_SECOND_ROUND_CB_MIN_SPIKES,
) -> dict[str, float]:
    min_spikes = _safe_positive(cb_min_spikes, DEFAULT_SECOND_ROUND_CB_MIN_SPIKES, min_value=0.0)
    return {
        "cb_amp_threshold": float(_safe_float(cb_amp_threshold, DEFAULT_SECOND_ROUND_CB_AMP_THRESHOLD)),
        "cb_min_spikes": float(max(0, int(round(min_spikes)))),
    }


def normalize_plateau_params(
    *,
    plateau_baseline_window_s: Any = DEFAULT_PLATEAU_BASELINE_WINDOW_S,
    plateau_vm_median_window_ms: Any = DEFAULT_PLATEAU_VM_MEDIAN_WINDOW_MS,
    plateau_vm_crossing_threshold: Any = DEFAULT_PLATEAU_VM_CROSSING_THRESHOLD,
    plateau_onset_threshold: Any = DEFAULT_PLATEAU_ONSET_THRESHOLD,
    plateau_offset_threshold: Any = DEFAULT_PLATEAU_OFFSET_THRESHOLD,
    plateau_amp_threshold: Any = DEFAULT_PLATEAU_AMP_THRESHOLD,
    plateau_duration_threshold_ms: Any = DEFAULT_PLATEAU_DURATION_THRESHOLD_MS,
    plateau_min_spikes: Any = DEFAULT_PLATEAU_MIN_SPIKES,
    plateau_peak_fraction_threshold: Any = DEFAULT_PLATEAU_PEAK_FRACTION_THRESHOLD,
    plateau_peak_fraction_duration_ms: Any = DEFAULT_PLATEAU_PEAK_FRACTION_DURATION_MS,
) -> dict[str, float]:
    min_spikes = _safe_positive(plateau_min_spikes, DEFAULT_PLATEAU_MIN_SPIKES, min_value=0.0)
    return {
        "plateau_baseline_window_s": _safe_positive(
            plateau_baseline_window_s,
            DEFAULT_PLATEAU_BASELINE_WINDOW_S,
            min_value=0.001,
        ),
        "plateau_vm_median_window_ms": _safe_positive(
            plateau_vm_median_window_ms,
            DEFAULT_PLATEAU_VM_MEDIAN_WINDOW_MS,
            min_value=0.001,
        ),
        "plateau_vm_crossing_threshold": float(
            _safe_float(plateau_vm_crossing_threshold, DEFAULT_PLATEAU_VM_CROSSING_THRESHOLD)
        ),
        "plateau_onset_threshold": float(_safe_float(plateau_onset_threshold, DEFAULT_PLATEAU_ONSET_THRESHOLD)),
        "plateau_offset_threshold": float(_safe_float(plateau_offset_threshold, DEFAULT_PLATEAU_OFFSET_THRESHOLD)),
        "plateau_amp_threshold": float(_safe_float(plateau_amp_threshold, DEFAULT_PLATEAU_AMP_THRESHOLD)),
        "plateau_duration_threshold_ms": _safe_positive(
            plateau_duration_threshold_ms,
            DEFAULT_PLATEAU_DURATION_THRESHOLD_MS,
            min_value=0.0,
        ),
        "plateau_min_spikes": float(max(0, int(round(min_spikes)))),
        "plateau_peak_fraction_threshold": _safe_positive(
            plateau_peak_fraction_threshold,
            DEFAULT_PLATEAU_PEAK_FRACTION_THRESHOLD,
            min_value=0.0,
        ),
        "plateau_peak_fraction_duration_ms": _safe_positive(
            plateau_peak_fraction_duration_ms,
            DEFAULT_PLATEAU_PEAK_FRACTION_DURATION_MS,
            min_value=0.0,
        ),
    }


def _safe_float(value: Any, fallback: float) -> float:
    try:
        out = float(value)
    except Exception:
        return float(fallback)
    if not np.isfinite(out):
        return float(fallback)
    return float(out)


def _safe_bool(value: Any, fallback: bool = False) -> bool:
    if isinstance(value, (list, tuple, set)):
        return any(_safe_bool(item, False) for item in value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "enabled"}
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    try:
        return bool(int(value))
    except Exception:
        return bool(fallback)


def _serialize_params(params: dict[str, float]) -> dict[str, float | bool]:
    return {
        key: bool(value) if key == "spike_baseline_remove_enabled" else float(value)
        for key, value in params.items()
    }


def _serialize_cb_params(params: dict[str, float]) -> dict[str, float]:
    clean = normalize_cb_params(**params)
    clean["cb_min_spikes"] = int(round(clean["cb_min_spikes"]))
    return {
        key: (
            bool(value)
            if key in {
                "remove_spikes_for_vm",
                "cb_require_min_isi",
                "include_first_burst_spike_for_spike_height",
                "refine_cb_onset",
            }
            else (
                int(value)
                if key in {"cb_min_spikes", "spike_height_min_isolated_spikes"}
                else float(value)
            )
        )
        for key, value in clean.items()
    }


def _serialize_second_round_params(params: dict[str, float]) -> dict[str, float]:
    clean = normalize_second_round_params(**params)
    return {
        key: (
            bool(value)
            if key == "refine_simple_spikes_by_height"
            else float(value)
        )
        for key, value in clean.items()
    }


def _serialize_second_round_cb_params(params: dict[str, float]) -> dict[str, float | int]:
    clean = normalize_second_round_cb_params(**params)
    return {
        "cb_amp_threshold": float(clean["cb_amp_threshold"]),
        "cb_min_spikes": int(round(clean["cb_min_spikes"])),
    }


def _serialize_plateau_params(params: dict[str, float]) -> dict[str, float | int]:
    clean = normalize_plateau_params(**params)
    return {
        "plateau_baseline_window_s": float(clean["plateau_baseline_window_s"]),
        "plateau_vm_median_window_ms": float(clean["plateau_vm_median_window_ms"]),
        "plateau_vm_crossing_threshold": float(clean["plateau_vm_crossing_threshold"]),
        "plateau_onset_threshold": float(clean["plateau_onset_threshold"]),
        "plateau_offset_threshold": float(clean["plateau_offset_threshold"]),
        "plateau_amp_threshold": float(clean["plateau_amp_threshold"]),
        "plateau_duration_threshold_ms": float(clean["plateau_duration_threshold_ms"]),
        "plateau_min_spikes": int(round(clean["plateau_min_spikes"])),
        "plateau_peak_fraction_threshold": float(clean["plateau_peak_fraction_threshold"]),
        "plateau_peak_fraction_duration_ms": float(clean["plateau_peak_fraction_duration_ms"]),
    }


def baseline_subtract_trace(trace_idx: np.ndarray, frame_rate: float, baseline_window_s: float) -> np.ndarray:
    trace_idx = np.asarray(trace_idx, dtype=float).reshape(-1)
    if trace_idx.size == 0:
        return trace_idx.copy()
    baseline_window = max(1, int(round(float(frame_rate) * float(baseline_window_s))))
    nan_mask = np.isnan(trace_idx)
    trace_interp = interpolate_nan_segment(trace_idx.copy())
    trace_baseline = median_filter(trace_interp, size=baseline_window)
    trace = trace_idx - trace_baseline
    trace[nan_mask] = np.nan
    return trace


def subtract_median_baseline_ms(trace_idx: np.ndarray, frame_rate: float, baseline_window_ms: float) -> np.ndarray:
    trace_idx = np.asarray(trace_idx, dtype=float).reshape(-1)
    if trace_idx.size == 0:
        return trace_idx.copy()
    baseline_window = max(1, int(round(float(frame_rate) * float(baseline_window_ms) / 1000.0)))
    nan_mask = np.isnan(trace_idx)
    trace_interp = interpolate_nan_segment(trace_idx.copy())
    trace_baseline = median_filter(trace_interp, size=baseline_window)
    trace = trace_idx - trace_baseline
    trace[nan_mask] = np.nan
    return trace


def highpass_trace(trace: np.ndarray, frame_rate: float, highpass_hz: float) -> np.ndarray:
    trace = np.asarray(trace, dtype=float).reshape(-1)
    if trace.size == 0:
        return trace.copy()
    if highpass_hz is None or float(highpass_hz) <= 0:
        return trace.copy()

    nan_mask = np.isnan(trace)
    finite_count = int(np.sum(np.isfinite(trace)))
    if finite_count < 32:
        return trace.copy()

    trace_interp = interpolate_nan_segment(trace.copy())
    try:
        trace_hp = highpass_filter(trace_interp, cutoff=float(highpass_hz), fs=float(frame_rate), order=3)
    except Exception:
        trace_hp = trace.copy()
    trace_hp[nan_mask] = np.nan
    return trace_hp


def segment_bounds(n_frames: int, frame_rate: float, segment_duration_s: float) -> list[tuple[int, int]]:
    n_frames = int(n_frames)
    if n_frames <= 0:
        return []
    frames_per_segment = max(1, int(round(float(frame_rate) * float(segment_duration_s))))
    bounds: list[tuple[int, int]] = []
    start = 0
    while start < n_frames:
        end = min(n_frames, start + frames_per_segment)
        bounds.append((int(start), int(end)))
        start = end
    return bounds


def default_thresholds_for_segments(trace_hp: np.ndarray, bounds: list[tuple[int, int]], threshold_mad: float) -> np.ndarray:
    trace_hp = np.asarray(trace_hp, dtype=float)
    thresholds = np.full(len(bounds), np.nan, dtype=float)
    for idx, (start, end) in enumerate(bounds):
        seg = trace_hp[int(start):int(end)]
        finite = seg[np.isfinite(seg)]
        if finite.size == 0:
            thresholds[idx] = np.nan
            continue
        median = float(np.nanmedian(finite))
        mad = float(np.nanmedian(np.abs(finite - median)))
        thresholds[idx] = median + float(threshold_mad) * mad
    return thresholds


def sanitize_thresholds(
    thresholds: Any,
    defaults: np.ndarray,
) -> np.ndarray:
    defaults = np.asarray(defaults, dtype=float).reshape(-1)
    if thresholds is None:
        return defaults.copy()
    arr = np.asarray(thresholds, dtype=float).reshape(-1)
    out = defaults.copy()
    n = min(out.size, arr.size)
    if n > 0:
        valid = np.isfinite(arr[:n])
        idx = np.flatnonzero(valid)
        out[idx] = arr[:n][idx]
    return out


def detect_spikes_from_thresholds(
    trace_hp: np.ndarray,
    bounds: list[tuple[int, int]],
    thresholds: np.ndarray,
    *,
    distance: int = DEFAULT_FIND_PEAK_DISTANCE,
) -> np.ndarray:
    trace_hp = np.asarray(trace_hp, dtype=float)
    thresholds = np.asarray(thresholds, dtype=float)
    spikes: list[np.ndarray] = []
    for seg_idx, (start, end) in enumerate(bounds):
        if seg_idx >= thresholds.size or not np.isfinite(thresholds[seg_idx]):
            continue
        seg = trace_hp[int(start):int(end)]
        if seg.size == 0:
            continue
        finite = np.isfinite(seg)
        if not np.any(finite):
            continue
        detect_seg = seg.copy()
        detect_seg[~finite] = -np.inf
        local_peaks, _ = signal.find_peaks(
            detect_seg,
            height=float(thresholds[seg_idx]),
            distance=max(1, int(distance)),
        )
        if local_peaks.size > 0:
            spikes.append(local_peaks.astype(np.int64) + int(start))
    if not spikes:
        return np.array([], dtype=np.int64)
    return np.sort(np.unique(np.concatenate(spikes).astype(np.int64)))


def _median_window_frames(window_ms: float, frame_rate: float) -> int:
    frames = max(1, int(round(float(window_ms) * float(frame_rate) / 1000.0)))
    if frames % 2 == 0:
        frames += 1
    return frames


def _subthreshold_vm_segment(
    trace_seg: np.ndarray,
    local_spikes: np.ndarray,
    median_window_frames: int,
    *,
    remove_spikes: bool = False,
) -> np.ndarray:
    vm = np.asarray(trace_seg, dtype=float).copy()
    n = vm.size
    if remove_spikes:
        for spk in np.asarray(local_spikes, dtype=np.int64).reshape(-1):
            if 0 <= int(spk) < n:
                start = max(0, int(spk) - 1)
                end = min(n, int(spk) + 2)
                vm[start:end] = np.nan
    if np.any(np.isnan(vm)) and np.any(np.isfinite(vm)):
        vm = interpolate_nan_segment(vm)
    if median_window_frames > 1 and vm.size > 0:
        vm = median_filter(vm, size=int(median_window_frames))
    return vm


def _average_spike_height(
    trace_seg: np.ndarray,
    local_spikes: np.ndarray,
    baseline_points: int = 3,
) -> float:
    heights_arr = _spike_heights_from_trace_baseline(
        trace_seg,
        local_spikes,
        baseline_points=baseline_points,
    )
    heights = heights_arr[np.isfinite(heights_arr) & (heights_arr > 0)]
    if heights.size == 0:
        return float("nan")
    return float(np.nanmean(heights))


def _spike_heights_from_trace_baseline(
    trace_seg: np.ndarray,
    local_spikes: np.ndarray,
    baseline_points: int = 3,
) -> np.ndarray:
    trace_seg = np.asarray(trace_seg, dtype=float)
    local_spikes = np.asarray(local_spikes, dtype=np.int64).reshape(-1)
    heights = np.full(local_spikes.shape, np.nan, dtype=float)
    for idx, spk in enumerate(local_spikes):
        spk = int(spk)
        if spk <= 0 or spk >= trace_seg.size or not np.isfinite(trace_seg[spk]):
            continue
        pre = trace_seg[max(0, spk - int(baseline_points)):spk]
        pre = pre[np.isfinite(pre)]
        if pre.size == 0:
            continue
        height = float(trace_seg[spk] - np.nanmin(pre))
        if np.isfinite(height):
            heights[idx] = height
    return heights


def _isolated_spikes(local_spikes: np.ndarray, frame_rate: float, isi_threshold_ms: float) -> np.ndarray:
    local_spikes = np.sort(np.unique(np.asarray(local_spikes, dtype=np.int64).reshape(-1)))
    if local_spikes.size <= 1:
        return local_spikes
    isi_threshold_frames = float(isi_threshold_ms) * float(frame_rate) / 1000.0
    previous_isi = np.full(local_spikes.size, np.inf, dtype=float)
    next_isi = np.full(local_spikes.size, np.inf, dtype=float)
    diffs = np.diff(local_spikes).astype(float)
    previous_isi[1:] = diffs
    next_isi[:-1] = diffs
    is_isolated = (previous_isi > isi_threshold_frames) & (next_isi > isi_threshold_frames)
    return local_spikes[is_isolated]


def _spikes_for_height_estimation(
    local_spikes: np.ndarray,
    frame_rate: float,
    isi_threshold_ms: float,
    *,
    include_first_burst_spike: bool = False,
) -> np.ndarray:
    local_spikes = np.sort(np.unique(np.asarray(local_spikes, dtype=np.int64).reshape(-1)))
    if local_spikes.size <= 1:
        return local_spikes
    isi_threshold_frames = float(isi_threshold_ms) * float(frame_rate) / 1000.0
    diffs = np.diff(local_spikes).astype(float)
    previous_isi = np.full(local_spikes.size, np.inf, dtype=float)
    next_isi = np.full(local_spikes.size, np.inf, dtype=float)
    previous_isi[1:] = diffs
    next_isi[:-1] = diffs
    is_isolated = (previous_isi > isi_threshold_frames) & (next_isi > isi_threshold_frames)
    if not include_first_burst_spike:
        return local_spikes[is_isolated]
    is_first_burst_spike = (previous_isi > isi_threshold_frames) & (next_isi <= isi_threshold_frames)
    return local_spikes[is_isolated | is_first_burst_spike]


def calculate_display_vm(
    trace: np.ndarray,
    segment_bounds_in: list[tuple[int, int]],
    spikes: np.ndarray,
    frame_rate: float,
    vm_median_window_ms: float = DEFAULT_VM_MEDIAN_WINDOW_MS,
    remove_spikes_for_vm: bool = DEFAULT_CB_REMOVE_SPIKES_FOR_VM,
) -> np.ndarray:
    """Calculate subthreshold Vm in the same units as the provided display trace."""
    trace = np.asarray(trace, dtype=float).reshape(-1)
    spikes = np.sort(np.unique(np.asarray(spikes, dtype=np.int64).reshape(-1)))
    spikes = spikes[(spikes >= 0) & (spikes < trace.size)]
    median_frames = _median_window_frames(vm_median_window_ms, frame_rate)
    vm_trace = np.full(trace.shape, np.nan, dtype=float)
    for start, end in segment_bounds_in:
        start = int(start)
        end = int(end)
        if end <= start:
            continue
        seg = trace[start:end]
        local_spikes = spikes[(spikes >= start) & (spikes < end)] - start
        vm = _subthreshold_vm_segment(
            seg,
            local_spikes,
            median_frames,
            remove_spikes=remove_spikes_for_vm,
        )
        vm[~np.isfinite(seg)] = np.nan
        vm_trace[start:end] = vm
    return vm_trace


def _contiguous_true_windows(mask: np.ndarray) -> list[tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    if mask.size == 0:
        return []
    diff = np.diff(mask.astype(np.int8))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1
    if mask[0]:
        starts = np.concatenate([[0], starts])
    if mask[-1]:
        ends = np.concatenate([ends, [mask.size]])
    windows: list[tuple[int, int]] = []
    si = 0
    ei = 0
    while si < len(starts) and ei < len(ends):
        start = int(starts[si])
        while ei < len(ends) and int(ends[ei]) <= start:
            ei += 1
        if ei >= len(ends):
            break
        end = int(ends[ei]) - 1
        if end >= start:
            windows.append((start, end))
        si += 1
        ei += 1
    return windows


def _min_isi_ms(spikes: np.ndarray, frame_rate: float) -> float:
    spikes = np.sort(np.asarray(spikes, dtype=np.int64).reshape(-1))
    if spikes.size < 2:
        return float("inf")
    isi = np.diff(spikes).astype(float) * 1000.0 / float(frame_rate)
    if isi.size == 0:
        return float("inf")
    return float(np.nanmin(isi))


def _merge_overlapping_windows(windows: list[tuple[int, int]]) -> list[tuple[int, int]]:
    clean: list[tuple[int, int]] = []
    for start, end in windows:
        s = int(min(start, end))
        e = int(max(start, end))
        if e >= s:
            clean.append((s, e))
    if not clean:
        return []

    clean.sort(key=lambda item: (item[0], item[1]))
    merged: list[list[int]] = [[clean[0][0], clean[0][1]]]
    for start, end in clean[1:]:
        last = merged[-1]
        if int(start) <= int(last[1]):
            last[1] = max(int(last[1]), int(end))
        else:
            merged.append([int(start), int(end)])
    return [(int(start), int(end)) for start, end in merged]


def _refine_cb_start_from_slope(
    vm_norm: np.ndarray,
    local_start: int,
    local_end: int,
    frame_rate: float,
    *,
    onset_threshold: float,
    max_onset_lead_ms: float,
) -> int:
    vm_norm = np.asarray(vm_norm, dtype=float).reshape(-1)
    if vm_norm.size == 0:
        return int(local_start)
    local_start = max(0, min(int(local_start), vm_norm.size - 1))
    local_end = max(local_start, min(int(local_end), vm_norm.size - 1))
    if local_end <= local_start:
        return int(local_start)

    window = vm_norm[local_start : local_end + 1]
    finite_window = np.where(np.isfinite(window), window, -np.inf)
    if not np.any(np.isfinite(finite_window)):
        return int(local_start)

    peak_frame = int(local_start + np.nanargmax(finite_window))
    slope_stop = peak_frame if peak_frame > local_start else local_end
    if slope_stop <= local_start:
        return int(local_start)

    dvm = np.diff(vm_norm)
    slope_slice = dvm[local_start:slope_stop]
    if slope_slice.size == 0:
        return int(local_start)
    slope_slice = np.where(np.isfinite(slope_slice), slope_slice, -np.inf)
    if not np.any(np.isfinite(slope_slice)):
        return int(local_start)

    max_slope_offset = int(np.nanargmax(slope_slice))
    max_slope_value = float(slope_slice[max_slope_offset])
    if not np.isfinite(max_slope_value) or max_slope_value <= 0:
        return int(local_start)
    max_slope_frame = int(local_start + max_slope_offset + 1)

    max_lead_frames = int(round(float(max_onset_lead_ms) * float(frame_rate) / 1000.0))
    search_start = max(0, max_slope_frame - max(0, max_lead_frames))
    search_end = max(local_start, max_slope_frame)
    search = vm_norm[search_start : search_end + 1]
    finite_search = np.isfinite(search)
    below = finite_search & (search <= float(onset_threshold))
    if np.any(below):
        refined_start = int(search_start + np.flatnonzero(below)[-1] + 1)
    else:
        refined_start = int(search_start)

    refined_start = max(int(local_start), refined_start)
    refined_start = min(refined_start, int(max_slope_frame), int(local_end))
    return int(refined_start)


def _refine_cb_end_from_slope(
    vm_norm: np.ndarray,
    local_start: int,
    local_end: int,
    frame_rate: float,
    *,
    offset_threshold: float,
    max_offset_lag_ms: float,
) -> int:
    vm_norm = np.asarray(vm_norm, dtype=float).reshape(-1)
    if vm_norm.size == 0:
        return int(local_end)
    local_start = max(0, min(int(local_start), vm_norm.size - 1))
    local_end = max(local_start, min(int(local_end), vm_norm.size - 1))
    if local_end <= local_start:
        return int(local_end)

    window = vm_norm[local_start : local_end + 1]
    finite_window = np.where(np.isfinite(window), window, -np.inf)
    if not np.any(np.isfinite(finite_window)):
        return int(local_end)

    peak_frame = int(local_start + np.nanargmax(finite_window))
    max_lag_frames = int(round(float(max_offset_lag_ms) * float(frame_rate) / 1000.0))
    search_stop = min(vm_norm.size - 1, local_end + max(0, max_lag_frames))
    if search_stop <= peak_frame:
        return int(local_end)

    dvm = np.diff(vm_norm)
    slope_slice = dvm[peak_frame:search_stop]
    if slope_slice.size == 0:
        return int(local_end)
    slope_slice = np.where(np.isfinite(slope_slice), slope_slice, np.inf)
    if not np.any(np.isfinite(slope_slice)):
        return int(local_end)

    min_slope_offset = int(np.nanargmin(slope_slice))
    min_slope_value = float(slope_slice[min_slope_offset])
    if not np.isfinite(min_slope_value) or min_slope_value >= 0:
        return int(local_end)
    min_slope_frame = int(peak_frame + min_slope_offset + 1)

    search = vm_norm[min_slope_frame : search_stop + 1]
    finite_search = np.isfinite(search)
    below = finite_search & (search <= float(offset_threshold))
    if not np.any(below):
        return int(local_end)

    refined_end = int(min_slope_frame + np.flatnonzero(below)[0])
    refined_end = max(int(local_end), refined_end)
    refined_end = min(refined_end, int(search_stop), vm_norm.size - 1)
    return int(refined_end)


def _refine_plateau_end_from_slope(
    vm_norm: np.ndarray,
    local_start: int,
    local_end: int,
    frame_rate: float,
    *,
    offset_threshold: float,
    crossing_threshold: float,
    max_offset_lag_ms: float,
) -> int:
    vm_norm = np.asarray(vm_norm, dtype=float).reshape(-1)
    if vm_norm.size == 0:
        return int(local_end)
    local_start = max(0, min(int(local_start), vm_norm.size - 1))
    local_end = max(local_start, min(int(local_end), vm_norm.size - 1))
    if local_end <= local_start:
        return int(local_end)

    window = vm_norm[local_start : local_end + 1]
    finite_window = np.where(np.isfinite(window), window, -np.inf)
    if not np.any(np.isfinite(finite_window)):
        return int(local_end)

    peak_frame = int(local_start + np.nanargmax(finite_window))
    max_lag_frames = int(round(float(max_offset_lag_ms) * float(frame_rate) / 1000.0))
    lag_stop = min(vm_norm.size - 1, local_end + max(0, max_lag_frames))
    if lag_stop <= peak_frame:
        return int(local_end)

    post_peak = vm_norm[peak_frame + 1 : lag_stop + 1]
    finite_post = np.isfinite(post_peak)
    below_crossing = finite_post & (post_peak <= float(crossing_threshold))
    if np.any(below_crossing):
        search_stop = int(peak_frame + 1 + np.flatnonzero(below_crossing)[0])
    else:
        search_stop = int(local_end)
    search_stop = max(int(local_end), min(search_stop, lag_stop))
    if search_stop <= peak_frame:
        return int(local_end)

    dvm = np.diff(vm_norm)
    slope_slice = dvm[peak_frame:search_stop]
    if slope_slice.size == 0:
        return int(min(local_end, search_stop))
    slope_slice = np.where(np.isfinite(slope_slice), slope_slice, np.inf)
    if not np.any(np.isfinite(slope_slice)):
        return int(min(local_end, search_stop))

    min_slope_offset = int(np.nanargmin(slope_slice))
    min_slope_value = float(slope_slice[min_slope_offset])
    if not np.isfinite(min_slope_value) or min_slope_value >= 0:
        return int(search_stop)
    min_slope_frame = int(peak_frame + min_slope_offset + 1)

    search = vm_norm[min_slope_frame : search_stop + 1]
    finite_search = np.isfinite(search)
    below_offset = finite_search & (search <= float(offset_threshold))
    if np.any(below_offset):
        refined_end = int(min_slope_frame + np.flatnonzero(below_offset)[0])
    else:
        refined_end = int(search_stop)
    refined_end = max(int(local_start), refined_end)
    refined_end = min(refined_end, int(search_stop), vm_norm.size - 1)
    return int(refined_end)


def _complex_burst_mask(n_frames: int, complex_bursts: dict[str, Any] | None) -> np.ndarray:
    mask = np.zeros(int(n_frames), dtype=bool)
    if not isinstance(complex_bursts, dict):
        return mask
    try:
        starts = np.asarray(complex_bursts.get("starts", []), dtype=np.int64).reshape(-1)
        ends = np.asarray(complex_bursts.get("ends", []), dtype=np.int64).reshape(-1)
    except Exception:
        return mask
    n = min(starts.size, ends.size)
    for start, end in zip(starts[:n], ends[:n]):
        s = max(0, int(min(start, end)))
        e = min(int(n_frames), int(max(start, end)) + 1)
        if e > s:
            mask[s:e] = True
    return mask


def spikes_in_windows(spikes: np.ndarray, windows: dict[str, Any] | None, n_frames: int) -> np.ndarray:
    spikes = np.sort(np.unique(np.asarray(spikes, dtype=np.int64).reshape(-1)))
    spikes = spikes[(spikes >= 0) & (spikes < int(n_frames))]
    if spikes.size == 0 or not isinstance(windows, dict):
        return np.array([], dtype=np.int64)
    mask = _complex_burst_mask(int(n_frames), windows)
    return spikes[mask[spikes]]


def _manual_period_mask(n_frames: int, periods: list[dict[str, Any]] | None) -> np.ndarray:
    mask = np.zeros(int(n_frames), dtype=bool)
    if not isinstance(periods, list):
        return mask
    for period in periods:
        if not isinstance(period, dict):
            continue
        try:
            start = int(period.get("start_frame", period.get("start")))
            end = int(period.get("end_frame", period.get("end")))
        except Exception:
            continue
        s = max(0, int(min(start, end)))
        e = min(int(n_frames), int(max(start, end)))
        if e > s:
            mask[s:e] = True
    return mask


def calculate_segment_snr(
    normalized_noise_trace: np.ndarray,
    segment_bounds_in: list[tuple[int, int]],
    spikes: np.ndarray,
    frame_rate: float,
    *,
    complex_bursts: dict[str, Any] | None = None,
    failed_min_spikes_windows: dict[str, Any] | None = None,
    manual_exclusion_periods: list[dict[str, Any]] | None = None,
    spike_mask_ms: float = DEFAULT_SNR_SPIKE_MASK_MS,
) -> dict[str, Any]:
    """Compute segment SNR from spike-height-normalized baseline residual noise.

    The input should be trace minus its median-filtered Vm/baseline, divided by
    the segment spike height. Since one spike-height unit is the numerator, SNR
    is 1 / std(residual baseline noise).
    """
    trace = np.asarray(normalized_noise_trace, dtype=float).reshape(-1)
    n_frames = trace.size
    mask = ~np.isfinite(trace)
    mask |= _complex_burst_mask(n_frames, complex_bursts)
    mask |= _complex_burst_mask(n_frames, failed_min_spikes_windows)
    mask |= _manual_period_mask(n_frames, manual_exclusion_periods)

    spike_radius = max(0, int(round(float(spike_mask_ms) * float(frame_rate) / 1000.0)))
    spike_arr = np.asarray(spikes, dtype=np.int64).reshape(-1)
    spike_arr = spike_arr[(spike_arr >= 0) & (spike_arr < n_frames)]
    for spike in np.unique(spike_arr):
        start = max(0, int(spike) - spike_radius)
        end = min(n_frames, int(spike) + spike_radius + 1)
        mask[start:end] = True

    segment_snr = np.full(len(segment_bounds_in), np.nan, dtype=float)
    segment_noise = np.full(len(segment_bounds_in), np.nan, dtype=float)
    segment_baseline_counts = np.zeros(len(segment_bounds_in), dtype=np.int64)
    for seg_idx, (start, end) in enumerate(segment_bounds_in):
        start = max(0, int(start))
        end = min(n_frames, int(end))
        if end <= start:
            continue
        keep = ~mask[start:end]
        values = trace[start:end][keep]
        values = values[np.isfinite(values)]
        segment_baseline_counts[seg_idx] = int(values.size)
        if values.size < 2:
            continue
        noise = float(np.nanstd(values, ddof=0))
        segment_noise[seg_idx] = noise
        if np.isfinite(noise) and noise > 1e-12:
            segment_snr[seg_idx] = 1.0 / noise

    finite_snr = segment_snr[np.isfinite(segment_snr)]
    overall_snr = float(np.nanmean(finite_snr)) if finite_snr.size else float("nan")
    return {
        "segment_snr": segment_snr,
        "segment_noise": segment_noise,
        "segment_baseline_counts": segment_baseline_counts,
        "overall_snr": overall_snr,
        "spike_mask_ms": float(spike_mask_ms),
    }


def _refine_simple_spikes_by_height(
    trace: np.ndarray | None,
    segment_bounds_in: list[tuple[int, int]],
    simple_spikes: np.ndarray,
    segment_spike_heights: np.ndarray | None,
    min_height_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    if trace is None or segment_spike_heights is None:
        return simple_spikes, np.array([], dtype=np.int64)
    trace = np.asarray(trace, dtype=float).reshape(-1)
    segment_heights = np.asarray(segment_spike_heights, dtype=float).reshape(-1)
    simple_spikes = np.sort(np.unique(np.asarray(simple_spikes, dtype=np.int64).reshape(-1)))
    if trace.size == 0 or simple_spikes.size == 0:
        return simple_spikes, np.array([], dtype=np.int64)

    min_fraction = max(0.0, float(min_height_fraction))
    kept: list[np.ndarray] = []
    removed: list[np.ndarray] = []
    for seg_idx, (start, end) in enumerate(segment_bounds_in):
        start = max(0, int(start))
        end = min(trace.size, int(end))
        if end <= start:
            continue
        in_segment = simple_spikes[(simple_spikes >= start) & (simple_spikes < end)]
        if in_segment.size == 0:
            continue
        if seg_idx >= segment_heights.size or not np.isfinite(segment_heights[seg_idx]) or segment_heights[seg_idx] <= 0:
            kept.append(in_segment)
            continue
        local_spikes = in_segment - start
        heights = _spike_heights_from_trace_baseline(
            trace[start:end],
            local_spikes,
            baseline_points=3,
        )
        threshold = float(min_fraction) * float(segment_heights[seg_idx])
        valid_height = np.isfinite(heights)
        keep_mask = (~valid_height) | (heights >= threshold)
        if np.any(keep_mask):
            kept.append(in_segment[keep_mask])
        if np.any(~keep_mask):
            removed.append(in_segment[~keep_mask])

    refined = (
        np.sort(np.unique(np.concatenate(kept).astype(np.int64)))
        if kept
        else np.array([], dtype=np.int64)
    )
    removed_arr = (
        np.sort(np.unique(np.concatenate(removed).astype(np.int64)))
        if removed
        else np.array([], dtype=np.int64)
    )
    return refined, removed_arr


def refine_spikes_by_height(
    trace: np.ndarray | None,
    segment_bounds_in: list[tuple[int, int]],
    candidate_spikes: np.ndarray,
    segment_spike_heights: np.ndarray | None,
    min_height_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    return _refine_simple_spikes_by_height(
        trace,
        segment_bounds_in,
        candidate_spikes,
        segment_spike_heights,
        min_height_fraction,
    )


def second_round_segment_thresholds(
    trace_hp: np.ndarray,
    segment_bounds_in: list[tuple[int, int]],
    stats_exclusion_windows: dict[str, Any] | None,
    second_round_params: dict[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    trace_hp = np.asarray(trace_hp, dtype=float).reshape(-1)
    params = normalize_second_round_params(**(second_round_params or {}))
    exclusion_mask = _complex_burst_mask(trace_hp.size, stats_exclusion_windows)
    ss_thresholds = np.full(len(segment_bounds_in), np.nan, dtype=float)
    cs_thresholds = np.full(len(segment_bounds_in), np.nan, dtype=float)

    for seg_idx, (start, end) in enumerate(segment_bounds_in):
        start = int(start)
        end = int(end)
        if end <= start:
            continue
        seg = trace_hp[start:end]
        seg_excluded = exclusion_mask[start:end]
        finite = np.isfinite(seg)
        stats_values = seg[finite & ~seg_excluded]
        if stats_values.size == 0:
            stats_values = seg[finite]
        if stats_values.size == 0:
            continue
        median = float(np.nanmedian(stats_values))
        mad = float(np.nanmedian(np.abs(stats_values - median)))
        ss_thresholds[seg_idx] = median + float(params["ss_threshold_mad"]) * mad
        cs_thresholds[seg_idx] = median + float(params["cs_threshold_mad"]) * mad
    return ss_thresholds, cs_thresholds


def detect_spikes_in_windows_from_thresholds(
    trace_hp: np.ndarray,
    segment_bounds_in: list[tuple[int, int]],
    windows: dict[str, Any] | None,
    thresholds: np.ndarray,
    *,
    distance: int = DEFAULT_FIND_PEAK_DISTANCE,
) -> np.ndarray:
    trace_hp = np.asarray(trace_hp, dtype=float).reshape(-1)
    thresholds = np.asarray(thresholds, dtype=float).reshape(-1)
    window_mask = _complex_burst_mask(trace_hp.size, windows)
    spikes: list[np.ndarray] = []
    for seg_idx, (start, end) in enumerate(segment_bounds_in):
        start = int(start)
        end = int(end)
        if end <= start or seg_idx >= thresholds.size or not np.isfinite(thresholds[seg_idx]):
            continue
        seg = trace_hp[start:end]
        seg_window = window_mask[start:end]
        finite = np.isfinite(seg)
        if not np.any(finite & seg_window):
            continue
        detect_seg = seg.copy()
        detect_seg[~finite | ~seg_window] = -np.inf
        local_peaks, _ = signal.find_peaks(
            detect_seg,
            height=float(thresholds[seg_idx]),
            distance=max(1, int(distance)),
        )
        if local_peaks.size:
            spikes.append(local_peaks.astype(np.int64) + start)
    if not spikes:
        return np.array([], dtype=np.int64)
    return np.sort(np.unique(np.concatenate(spikes).astype(np.int64)))


def detect_second_round_spikes(
    trace_hp: np.ndarray,
    segment_bounds_in: list[tuple[int, int]],
    complex_bursts: dict[str, Any] | None,
    second_round_params: dict[str, float] | None = None,
    *,
    trace: np.ndarray | None = None,
    segment_spike_heights: np.ndarray | None = None,
    distance: int = DEFAULT_FIND_PEAK_DISTANCE,
) -> dict[str, Any]:
    """Detect/refine SS candidates before the second-round CB and CS passes.

    The provided CB windows are used for median/MAD statistics only. SS
    candidates are detected anywhere finite, then final true CB windows remove
    their spikes from the SS class later.
    """
    trace_hp = np.asarray(trace_hp, dtype=float).reshape(-1)
    params = normalize_second_round_params(**(second_round_params or {}))
    ss_spikes: list[np.ndarray] = []
    ss_thresholds, cs_thresholds = second_round_segment_thresholds(
        trace_hp,
        segment_bounds_in,
        complex_bursts,
        params,
    )

    for seg_idx, (start, end) in enumerate(segment_bounds_in):
        start = int(start)
        end = int(end)
        if end <= start or seg_idx >= ss_thresholds.size or not np.isfinite(ss_thresholds[seg_idx]):
            continue
        seg = trace_hp[start:end]
        finite = np.isfinite(seg)

        detect_ss = seg.copy()
        detect_ss[~finite] = -np.inf
        ss_local, _ = signal.find_peaks(
            detect_ss,
            height=float(ss_thresholds[seg_idx]),
            distance=max(1, int(distance)),
        )
        if ss_local.size:
            ss_spikes.append(ss_local.astype(np.int64) + start)

    simple_spikes = (
        np.sort(np.unique(np.concatenate(ss_spikes).astype(np.int64)))
        if ss_spikes
        else np.array([], dtype=np.int64)
    )
    complex_spikes = np.array([], dtype=np.int64)
    removed_simple_spikes = np.array([], dtype=np.int64)
    if bool(params["refine_simple_spikes_by_height"]):
        refinement_candidates = simple_spikes
        _kept_refined, removed_simple_spikes = _refine_simple_spikes_by_height(
            trace,
            segment_bounds_in,
            refinement_candidates,
            segment_spike_heights,
            float(params["simple_spike_min_height_fraction"]),
        )
        if removed_simple_spikes.size:
            simple_spikes = simple_spikes[~np.isin(simple_spikes, removed_simple_spikes)]
    all_spikes = np.sort(np.unique(np.concatenate([simple_spikes, complex_spikes]).astype(np.int64)))
    return {
        "detection_order": "ss_all_refine_cb_true_cs",
        "second_round_params": _serialize_second_round_params(params),
        "spikes": all_spikes,
        "cb_input_spikes": simple_spikes,
        "simple_spikes": simple_spikes,
        "complex_spikes": complex_spikes,
        "removed_simple_spikes": removed_simple_spikes,
        "ss_thresholds": ss_thresholds,
        "cs_thresholds": cs_thresholds,
    }


def detect_complex_bursts_segmented(
    trace: np.ndarray,
    segment_bounds_in: list[tuple[int, int]],
    spikes: np.ndarray,
    frame_rate: float,
    cb_params: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Classify all detected spikes into simple/complex spikes using segment-local Vm windows."""
    trace = np.asarray(trace, dtype=float).reshape(-1)
    spikes = np.sort(np.unique(np.asarray(spikes, dtype=np.int64).reshape(-1)))
    spikes = spikes[(spikes >= 0) & (spikes < trace.size)]
    params = normalize_cb_params(**(cb_params or {}))
    trace = baseline_subtract_trace(trace, frame_rate, params["cb_baseline_window_s"])
    remove_spikes_for_vm = bool(params["remove_spikes_for_vm"])
    median_frames = _median_window_frames(params["vm_median_window_ms"], frame_rate)
    crossing_thr = float(params["vm_crossing_threshold"])
    amp_thr = float(params["cb_amp_threshold"])
    dur_thr_ms = float(params["cb_duration_threshold_ms"])
    min_spikes = int(round(params["cb_min_spikes"]))
    isi_thr_ms = float(params["cb_isi_threshold_ms"])
    require_min_isi = bool(params["cb_require_min_isi"])
    spike_height_min_isolated = int(round(params["spike_height_min_isolated_spikes"]))
    include_first_burst_spike_for_height = bool(params["include_first_burst_spike_for_spike_height"])
    refine_cb_onset = bool(params["refine_cb_onset"])
    onset_thr = float(params["cb_onset_threshold"])
    offset_thr = float(params["cb_offset_threshold"])
    max_onset_lead_ms = float(params["cb_max_onset_lead_ms"])

    simple_spike_set = set(int(v) for v in spikes)
    complex_spike_set: set[int] = set()
    starts: list[int] = []
    ends: list[int] = []
    locs: list[int] = []
    peaks: list[float] = []
    amplitudes: list[float] = []
    durations_ms: list[float] = []
    n_spikes_list: list[int] = []
    min_isi_list: list[float] = []
    segment_indices: list[int] = []
    failed_starts: list[int] = []
    failed_ends: list[int] = []
    failed_locs: list[int] = []
    failed_peaks: list[float] = []
    failed_amplitudes: list[float] = []
    failed_durations_ms: list[float] = []
    failed_n_spikes_list: list[int] = []
    failed_min_isi_list: list[float] = []
    failed_segment_indices: list[int] = []
    segment_spike_heights = np.full(len(segment_bounds_in), np.nan, dtype=float)
    segment_spike_height_counts = np.zeros(len(segment_bounds_in), dtype=np.int64)
    failed_min_spikes_after_amp_duration_by_segment = np.zeros(len(segment_bounds_in), dtype=np.int64)
    segment_vm_medians = np.full(len(segment_bounds_in), np.nan, dtype=float)
    vm_trace = np.full(trace.shape, np.nan, dtype=float)
    last_valid_spike_height = float("nan")
    segment_local_spikes: list[np.ndarray] = [
        np.array([], dtype=np.int64) for _segment in segment_bounds_in
    ]

    for seg_idx, (start, end) in enumerate(segment_bounds_in):
        start = int(start)
        end = int(end)
        if end <= start:
            continue
        seg = trace[start:end]
        local_spikes = spikes[(spikes >= start) & (spikes < end)] - start
        segment_local_spikes[seg_idx] = local_spikes.astype(np.int64, copy=False)

        vm = _subthreshold_vm_segment(
            seg,
            local_spikes,
            median_frames,
            remove_spikes=remove_spikes_for_vm,
        )
        vm[~np.isfinite(seg)] = np.nan
        vm_trace[start:end] = vm
        finite_vm = vm[np.isfinite(vm)]
        if finite_vm.size == 0:
            finite_seg = seg[np.isfinite(seg)]
            if finite_seg.size:
                segment_vm_medians[seg_idx] = float(np.nanmedian(finite_seg))
        else:
            segment_vm_medians[seg_idx] = float(np.nanmedian(finite_vm))

        height_spikes = _spikes_for_height_estimation(
            local_spikes,
            frame_rate,
            isi_thr_ms,
            include_first_burst_spike=include_first_burst_spike_for_height,
        )
        segment_spike_height_counts[seg_idx] = int(height_spikes.size)
        candidate_height = (
            _average_spike_height(seg, height_spikes)
            if height_spikes.size > spike_height_min_isolated
            else float("nan")
        )
        if np.isfinite(candidate_height) and candidate_height > 0:
            avg_height = candidate_height
            last_valid_spike_height = candidate_height
        else:
            avg_height = last_valid_spike_height
        segment_spike_heights[seg_idx] = avg_height

    valid_heights = segment_spike_heights[np.isfinite(segment_spike_heights) & (segment_spike_heights > 0)]
    if valid_heights.size:
        fallback_height = float(np.nanmedian(valid_heights))
    else:
        finite_trace = trace[np.isfinite(trace)]
        fallback_height = float(np.nanpercentile(np.abs(finite_trace - np.nanmedian(finite_trace)), 95)) if finite_trace.size else 1.0
    if not np.isfinite(fallback_height) or fallback_height <= 1e-12:
        fallback_height = 1.0
    missing_heights = ~np.isfinite(segment_spike_heights) | (segment_spike_heights <= 0)
    segment_spike_heights[missing_heights] = fallback_height

    for seg_idx, (start, end) in enumerate(segment_bounds_in):
        start = int(start)
        end = int(end)
        if end <= start:
            continue
        avg_height = float(segment_spike_heights[seg_idx])
        if not np.isfinite(avg_height) or avg_height <= 0:
            continue
        vm = vm_trace[start:end]
        vm_median = float(segment_vm_medians[seg_idx]) if seg_idx < segment_vm_medians.size else float("nan")
        if not np.isfinite(vm_median):
            finite_vm = vm[np.isfinite(vm)]
            if finite_vm.size:
                vm_median = float(np.nanmedian(finite_vm))
            else:
                continue
        local_spikes = segment_local_spikes[seg_idx]
        vm_norm = (vm - vm_median) / avg_height
        vm_finite = np.where(np.isfinite(vm_norm), vm_norm, 0.0)
        above = vm_finite > crossing_thr

        candidate_windows: list[tuple[int, int]] = []
        for local_start, local_end in _contiguous_true_windows(above):
            refined_start = (
                _refine_cb_start_from_slope(
                    vm_norm,
                    local_start,
                    local_end,
                    frame_rate,
                    onset_threshold=onset_thr,
                    max_onset_lead_ms=max_onset_lead_ms,
                )
                if refine_cb_onset
                else int(local_start)
            )
            refined_end = (
                _refine_cb_end_from_slope(
                    vm_norm,
                    refined_start,
                    local_end,
                    frame_rate,
                    offset_threshold=offset_thr,
                    max_offset_lag_ms=max_onset_lead_ms,
                )
                if refine_cb_onset
                else int(local_end)
            )
            candidate_windows.append((int(refined_start), int(refined_end)))

        for refined_start, refined_end in _merge_overlapping_windows(candidate_windows):
            in_window = local_spikes[(local_spikes >= refined_start) & (local_spikes <= refined_end)]
            n_spikes = int(in_window.size)
            window = vm_finite[refined_start:refined_end + 1]
            peak_amp = float(np.nanmax(window)) if window.size else float("nan")
            duration_ms = float(refined_end - refined_start + 1) * 1000.0 / float(frame_rate)
            passes_amp_duration = (
                np.isfinite(peak_amp)
                and peak_amp >= amp_thr
                and duration_ms >= dur_thr_ms
            )
            min_isi = _min_isi_ms(in_window, frame_rate)
            if passes_amp_duration and n_spikes < min_spikes:
                failed_min_spikes_after_amp_duration_by_segment[seg_idx] += 1
                global_failed_start = start + int(refined_start)
                global_failed_end = start + int(refined_end)
                if window.size and np.any(np.isfinite(window)):
                    failed_local_loc = int(refined_start + np.nanargmax(window))
                    failed_peak_value = float(vm_norm[failed_local_loc])
                else:
                    failed_local_loc = int(refined_start)
                    failed_peak_value = float("nan")
                failed_starts.append(global_failed_start)
                failed_ends.append(global_failed_end)
                failed_locs.append(start + failed_local_loc)
                failed_peaks.append(failed_peak_value)
                failed_amplitudes.append(peak_amp)
                failed_durations_ms.append(duration_ms)
                failed_n_spikes_list.append(n_spikes)
                failed_min_isi_list.append(min_isi)
                failed_segment_indices.append(int(seg_idx))
            is_complex = (
                n_spikes >= min_spikes
                and passes_amp_duration
                and (not require_min_isi or min_isi <= isi_thr_ms)
            )
            if not is_complex:
                continue

            global_start = start + int(refined_start)
            global_end = start + int(refined_end)
            global_spikes = start + in_window.astype(np.int64)
            complex_spike_set.update(int(v) for v in global_spikes)
            if window.size and np.any(np.isfinite(window)):
                local_loc = int(refined_start + np.nanargmax(window))
                peak_value = float(vm_norm[local_loc])
            else:
                local_loc = int(refined_start)
                peak_value = float("nan")
            starts.append(global_start)
            ends.append(global_end)
            locs.append(start + local_loc)
            peaks.append(peak_value)
            amplitudes.append(peak_amp)
            durations_ms.append(duration_ms)
            n_spikes_list.append(n_spikes)
            min_isi_list.append(min_isi)
            segment_indices.append(int(seg_idx))

    simple_spike_set.difference_update(complex_spike_set)
    simple_spikes = np.array(sorted(simple_spike_set), dtype=np.int64)
    complex_spikes = np.array(sorted(complex_spike_set), dtype=np.int64)
    complex_bursts = {
        "starts": np.asarray(starts, dtype=np.int64),
        "ends": np.asarray(ends, dtype=np.int64),
        "locs": np.asarray(locs, dtype=np.int64),
        "peaks": np.asarray(peaks, dtype=float),
        "amplitudes": np.asarray(amplitudes, dtype=float),
        "durations_ms": np.asarray(durations_ms, dtype=float),
        "n_spikes": np.asarray(n_spikes_list, dtype=np.int64),
        "min_isi_ms": np.asarray(min_isi_list, dtype=float),
        "segment_indices": np.asarray(segment_indices, dtype=np.int64),
    }
    failed_min_spikes_after_amp_duration_windows = {
        "starts": np.asarray(failed_starts, dtype=np.int64),
        "ends": np.asarray(failed_ends, dtype=np.int64),
        "locs": np.asarray(failed_locs, dtype=np.int64),
        "peaks": np.asarray(failed_peaks, dtype=float),
        "amplitudes": np.asarray(failed_amplitudes, dtype=float),
        "durations_ms": np.asarray(failed_durations_ms, dtype=float),
        "n_spikes": np.asarray(failed_n_spikes_list, dtype=np.int64),
        "min_isi_ms": np.asarray(failed_min_isi_list, dtype=float),
        "segment_indices": np.asarray(failed_segment_indices, dtype=np.int64),
    }

    trace_spike_height_normalized = np.full(trace.shape, np.nan, dtype=float)
    vm_spike_height_normalized = np.full(trace.shape, np.nan, dtype=float)
    for seg_idx, (start, end) in enumerate(segment_bounds_in):
        start = int(start)
        end = int(end)
        if end <= start:
            continue
        seg = trace[start:end]
        height = float(segment_spike_heights[seg_idx]) if seg_idx < segment_spike_heights.size else float("nan")
        if not np.isfinite(height) or height <= 0:
            height = fallback_height
        center = float(segment_vm_medians[seg_idx]) if seg_idx < segment_vm_medians.size else float("nan")
        if not np.isfinite(center):
            finite_seg = seg[np.isfinite(seg)]
            center = float(np.nanmedian(finite_seg)) if finite_seg.size else 0.0
        trace_spike_height_normalized[start:end] = (seg - center) / height
        vm_spike_height_normalized[start:end] = (vm_trace[start:end] - center) / height

    return {
        "cb_params": _serialize_cb_params(params),
        "simple_spikes": simple_spikes,
        "complex_spikes": complex_spikes,
        "complex_bursts": complex_bursts,
        "failed_min_spikes_after_amp_duration": int(np.sum(failed_min_spikes_after_amp_duration_by_segment)),
        "failed_min_spikes_after_amp_duration_by_segment": failed_min_spikes_after_amp_duration_by_segment,
        "failed_min_spikes_after_amp_duration_windows": failed_min_spikes_after_amp_duration_windows,
        "segment_spike_heights": segment_spike_heights,
        "segment_spike_height_counts": segment_spike_height_counts,
        "vm_trace": vm_trace,
        "vm_spike_height_normalized": vm_spike_height_normalized,
        "trace_spike_height_normalized": trace_spike_height_normalized,
    }


def detect_plateaus_segmented(
    trace: np.ndarray,
    segment_bounds_in: list[tuple[int, int]],
    spikes: np.ndarray,
    frame_rate: float,
    plateau_params: dict[str, float] | None = None,
    onset_refinement_params: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Detect plateau depolarizations from a long-baseline-corrected trace."""
    original_trace = np.asarray(trace, dtype=float).reshape(-1)
    spikes = np.sort(np.unique(np.asarray(spikes, dtype=np.int64).reshape(-1)))
    spikes = spikes[(spikes >= 0) & (spikes < original_trace.size)]
    params = normalize_plateau_params(**(plateau_params or {}))
    onset_params = (
        normalize_cb_params(**onset_refinement_params)
        if isinstance(onset_refinement_params, dict)
        else None
    )
    refine_plateau_onset = bool(onset_params["refine_cb_onset"]) if onset_params is not None else False
    onset_thr = float(params["plateau_onset_threshold"])
    offset_thr = float(params["plateau_offset_threshold"])
    max_onset_lead_ms = (
        float(onset_params["cb_max_onset_lead_ms"]) if onset_params is not None else DEFAULT_CB_MAX_ONSET_LEAD_MS
    )
    plateau_trace = baseline_subtract_trace(
        original_trace,
        frame_rate,
        params["plateau_baseline_window_s"],
    )
    median_frames = _median_window_frames(params["plateau_vm_median_window_ms"], frame_rate)
    crossing_thr = float(params["plateau_vm_crossing_threshold"])
    amp_thr = float(params["plateau_amp_threshold"])
    dur_thr_ms = float(params["plateau_duration_threshold_ms"])
    min_spikes = int(round(params["plateau_min_spikes"]))
    peak_fraction_thr = float(params["plateau_peak_fraction_threshold"])
    peak_fraction_duration_ms = float(params["plateau_peak_fraction_duration_ms"])
    peak_fraction_required_frames = (
        int(np.ceil(peak_fraction_duration_ms * float(frame_rate) / 1000.0))
        if peak_fraction_duration_ms > 0
        else 0
    )

    segment_spike_heights = np.full(len(segment_bounds_in), np.nan, dtype=float)
    segment_spike_height_counts = np.zeros(len(segment_bounds_in), dtype=np.int64)
    segment_vm_medians = np.full(len(segment_bounds_in), np.nan, dtype=float)
    vm_trace = np.full(plateau_trace.shape, np.nan, dtype=float)
    segment_local_spikes: list[np.ndarray] = [
        np.array([], dtype=np.int64) for _segment in segment_bounds_in
    ]
    last_valid_spike_height = float("nan")

    for seg_idx, (start, end) in enumerate(segment_bounds_in):
        start = int(start)
        end = int(end)
        if end <= start:
            continue
        seg = plateau_trace[start:end]
        local_spikes = spikes[(spikes >= start) & (spikes < end)] - start
        segment_local_spikes[seg_idx] = local_spikes.astype(np.int64, copy=False)
        vm = _subthreshold_vm_segment(seg, local_spikes, median_frames, remove_spikes=False)
        vm[~np.isfinite(seg)] = np.nan
        vm_trace[start:end] = vm

        finite_vm = vm[np.isfinite(vm)]
        if finite_vm.size:
            segment_vm_medians[seg_idx] = float(np.nanmedian(finite_vm))
        else:
            finite_seg = seg[np.isfinite(seg)]
            if finite_seg.size:
                segment_vm_medians[seg_idx] = float(np.nanmedian(finite_seg))

        height_spikes = local_spikes
        segment_spike_height_counts[seg_idx] = int(height_spikes.size)
        candidate_height = _average_spike_height(seg, height_spikes) if height_spikes.size else float("nan")
        if np.isfinite(candidate_height) and candidate_height > 0:
            avg_height = candidate_height
            last_valid_spike_height = candidate_height
        else:
            avg_height = last_valid_spike_height
        segment_spike_heights[seg_idx] = avg_height

    valid_heights = segment_spike_heights[np.isfinite(segment_spike_heights) & (segment_spike_heights > 0)]
    if valid_heights.size:
        fallback_height = float(np.nanmedian(valid_heights))
    else:
        finite_trace = plateau_trace[np.isfinite(plateau_trace)]
        fallback_height = (
            float(np.nanpercentile(np.abs(finite_trace - np.nanmedian(finite_trace)), 95))
            if finite_trace.size
            else 1.0
        )
    if not np.isfinite(fallback_height) or fallback_height <= 1e-12:
        fallback_height = 1.0
    missing_heights = ~np.isfinite(segment_spike_heights) | (segment_spike_heights <= 0)
    segment_spike_heights[missing_heights] = fallback_height

    starts: list[int] = []
    ends: list[int] = []
    locs: list[int] = []
    peaks: list[float] = []
    amplitudes: list[float] = []
    durations_ms: list[float] = []
    peak_fraction_durations_ms: list[float] = []
    n_spikes_list: list[int] = []
    segment_indices: list[int] = []

    trace_spike_height_normalized = np.full(plateau_trace.shape, np.nan, dtype=float)
    vm_spike_height_normalized = np.full(plateau_trace.shape, np.nan, dtype=float)
    for seg_idx, (start, end) in enumerate(segment_bounds_in):
        start = int(start)
        end = int(end)
        if end <= start:
            continue
        seg = plateau_trace[start:end]
        height = float(segment_spike_heights[seg_idx]) if seg_idx < segment_spike_heights.size else fallback_height
        if not np.isfinite(height) or height <= 0:
            height = fallback_height
        center = float(segment_vm_medians[seg_idx]) if seg_idx < segment_vm_medians.size else float("nan")
        if not np.isfinite(center):
            finite_seg = seg[np.isfinite(seg)]
            center = float(np.nanmedian(finite_seg)) if finite_seg.size else 0.0

        vm = vm_trace[start:end]
        trace_spike_height_normalized[start:end] = (seg - center) / height
        vm_norm = (vm - center) / height
        vm_spike_height_normalized[start:end] = vm_norm
        vm_finite = np.where(np.isfinite(vm_norm), vm_norm, 0.0)
        above = vm_finite > crossing_thr
        local_spikes = segment_local_spikes[seg_idx]

        candidate_windows: list[tuple[int, int]] = []
        for local_start, local_end in _contiguous_true_windows(above):
            refined_start = (
                _refine_cb_start_from_slope(
                    vm_norm,
                    local_start,
                    local_end,
                    frame_rate,
                    onset_threshold=onset_thr,
                    max_onset_lead_ms=max_onset_lead_ms,
                )
                if refine_plateau_onset
                else int(local_start)
            )
            refined_end = (
                _refine_plateau_end_from_slope(
                    vm_norm,
                    refined_start,
                    local_end,
                    frame_rate,
                    offset_threshold=offset_thr,
                    crossing_threshold=crossing_thr,
                    max_offset_lag_ms=max_onset_lead_ms,
                )
                if refine_plateau_onset
                else int(local_end)
            )
            candidate_windows.append((int(refined_start), int(refined_end)))

        for refined_start, refined_end in _merge_overlapping_windows(candidate_windows):
            in_window = local_spikes[(local_spikes >= refined_start) & (local_spikes <= refined_end)]
            n_spikes = int(in_window.size)
            window = vm_finite[refined_start:refined_end + 1]
            peak_amp = float(np.nanmax(window)) if window.size else float("nan")
            duration_ms = float(refined_end - refined_start + 1) * 1000.0 / float(frame_rate)
            peak_fraction_run_frames = 0
            if np.isfinite(peak_amp) and window.size:
                peak_fraction_level = peak_amp * peak_fraction_thr
                peak_fraction_mask = np.asarray(window >= peak_fraction_level, dtype=bool)
                for run_start, run_end in _contiguous_true_windows(peak_fraction_mask):
                    peak_fraction_run_frames = max(peak_fraction_run_frames, int(run_end - run_start + 1))
            peak_fraction_run_ms = float(peak_fraction_run_frames) * 1000.0 / float(frame_rate)
            if not (
                np.isfinite(peak_amp)
                and peak_amp >= amp_thr
                and duration_ms >= dur_thr_ms
                and n_spikes >= min_spikes
                and peak_fraction_run_frames >= peak_fraction_required_frames
            ):
                continue
            if window.size and np.any(np.isfinite(window)):
                local_loc = int(refined_start + np.nanargmax(window))
                peak_value = float(vm_norm[local_loc])
            else:
                local_loc = int(refined_start)
                peak_value = float("nan")
            starts.append(start + int(refined_start))
            ends.append(start + int(refined_end))
            locs.append(start + local_loc)
            peaks.append(peak_value)
            amplitudes.append(peak_amp)
            durations_ms.append(duration_ms)
            peak_fraction_durations_ms.append(peak_fraction_run_ms)
            n_spikes_list.append(n_spikes)
            segment_indices.append(int(seg_idx))

    plateaus = {
        "starts": np.asarray(starts, dtype=np.int64),
        "ends": np.asarray(ends, dtype=np.int64),
        "locs": np.asarray(locs, dtype=np.int64),
        "peaks": np.asarray(peaks, dtype=float),
        "amplitudes": np.asarray(amplitudes, dtype=float),
        "durations_ms": np.asarray(durations_ms, dtype=float),
        "peak_fraction_durations_ms": np.asarray(peak_fraction_durations_ms, dtype=float),
        "n_spikes": np.asarray(n_spikes_list, dtype=np.int64),
        "segment_indices": np.asarray(segment_indices, dtype=np.int64),
    }
    return {
        "plateau_params": _serialize_plateau_params(params),
        "plateaus": plateaus,
        "segment_spike_heights": segment_spike_heights,
        "segment_spike_height_counts": segment_spike_height_counts,
        "vm_trace": vm_trace,
        "vm_spike_height_normalized": vm_spike_height_normalized,
        "trace_spike_height_normalized": trace_spike_height_normalized,
        "plateau_trace": plateau_trace,
    }


def prepare_detection_inputs(
    trace_idx: np.ndarray,
    frame_rate: float,
    params: dict[str, float],
) -> dict[str, Any]:
    params = normalize_params(**params)
    trace = baseline_subtract_trace(trace_idx, frame_rate, params["baseline_window_s"])
    trace_for_highpass = (
        subtract_median_baseline_ms(trace, frame_rate, params["spike_baseline_window_ms"])
        if params["spike_baseline_remove_enabled"]
        else trace
    )
    trace_hp = highpass_trace(trace_for_highpass, frame_rate, params["highpass_hz"])
    bounds = segment_bounds(len(trace), frame_rate, params["segment_duration_s"])
    defaults = default_thresholds_for_segments(trace_hp, bounds, params["threshold_mad"])
    return {
        "params": _serialize_params(params),
        "trace": trace,
        "trace_for_highpass": trace_for_highpass,
        "trace_hp": trace_hp,
        "segment_bounds": bounds,
        "default_thresholds": defaults,
    }


def run_detection(
    trace_idx: np.ndarray,
    frame_rate: float,
    params: dict[str, float],
    thresholds: Any = None,
) -> dict[str, Any]:
    prepared = prepare_detection_inputs(trace_idx, frame_rate, params)
    clean_thresholds = sanitize_thresholds(thresholds, prepared["default_thresholds"])
    spikes = detect_spikes_from_thresholds(
        prepared["trace_hp"],
        prepared["segment_bounds"],
        clean_thresholds,
    )
    prepared["thresholds"] = clean_thresholds
    prepared["spikes"] = spikes
    return prepared


def _serialize_complex_bursts(complex_bursts: dict[str, Any] | None) -> dict[str, list[Any]]:
    if not isinstance(complex_bursts, dict):
        return {
            "starts": [],
            "ends": [],
            "locs": [],
            "peaks": [],
            "amplitudes": [],
            "durations_ms": [],
            "n_spikes": [],
            "min_isi_ms": [],
            "segment_indices": [],
        }
    out: dict[str, list[Any]] = {}
    for key in ("starts", "ends", "locs", "n_spikes", "segment_indices"):
        values = []
        try:
            raw = np.asarray(complex_bursts.get(key, []), dtype=object).reshape(-1)
        except Exception:
            raw = []
        for item in raw:
            try:
                numeric = float(item)
            except Exception:
                continue
            if np.isfinite(numeric):
                values.append(int(round(numeric)))
        out[key] = values
    for key in ("peaks", "amplitudes", "durations_ms", "min_isi_ms", "peak_fraction_durations_ms"):
        values = []
        try:
            raw = np.asarray(complex_bursts.get(key, []), dtype=object).reshape(-1)
        except Exception:
            raw = []
        for item in raw:
            try:
                numeric = float(item)
            except Exception:
                values.append(float("nan"))
                continue
            values.append(float(numeric) if np.isfinite(numeric) else float("nan"))
        out[key] = values
    return out


def _serialize_manual_exclusion_periods(periods: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    if not isinstance(periods, list):
        return []
    out: list[dict[str, Any]] = []
    for period in periods:
        if not isinstance(period, dict):
            continue
        try:
            start = int(period.get("start_frame", period.get("start")))
            end = int(period.get("end_frame", period.get("end")))
        except Exception:
            continue
        if end < start:
            start, end = end, start
        if end <= start:
            continue
        item: dict[str, Any] = {
            "start_frame": int(start),
            "end_frame": int(end),
        }
        for key in ("start_s", "end_s", "duration_s"):
            if key not in period:
                continue
            try:
                value = float(period[key])
            except Exception:
                continue
            if np.isfinite(value):
                item[key] = value
        if "segment_index" in period:
            try:
                item["segment_index"] = int(period["segment_index"])
            except Exception:
                pass
        out.append(item)
    return out


def make_result_payload(
    *,
    animal_id: str,
    cell_idx: int,
    frame_rate: float,
    source_file: str | None = None,
    source_filename: str | None = None,
    params: dict[str, float],
    segment_bounds_in: list[tuple[int, int]],
    thresholds: np.ndarray,
    spikes: np.ndarray,
    cb_params: dict[str, float] | None = None,
    first_round_cb_params: dict[str, float] | None = None,
    simple_spikes: np.ndarray | None = None,
    complex_spikes: np.ndarray | None = None,
    complex_bursts: dict[str, Any] | None = None,
    first_round_complex_bursts: dict[str, Any] | None = None,
    second_round: dict[str, Any] | None = None,
    second_round_params: dict[str, float] | None = None,
    second_round_ss_thresholds: np.ndarray | None = None,
    second_round_cs_thresholds: np.ndarray | None = None,
    second_round_cb_params: dict[str, float] | None = None,
    second_round_cb: dict[str, Any] | None = None,
    plateau_params: dict[str, float] | None = None,
    plateau_result: dict[str, Any] | None = None,
    plateaus: dict[str, Any] | None = None,
    manual_exclusion_periods: list[dict[str, Any]] | None = None,
    failed_min_spikes_after_amp_duration: int | None = None,
    failed_min_spikes_after_amp_duration_by_segment: np.ndarray | None = None,
    failed_min_spikes_after_amp_duration_windows: dict[str, Any] | None = None,
    first_round_failed_min_spikes_after_amp_duration: int | None = None,
    first_round_failed_min_spikes_after_amp_duration_by_segment: np.ndarray | None = None,
    first_round_failed_min_spikes_after_amp_duration_windows: dict[str, Any] | None = None,
    segment_spike_heights: np.ndarray | None = None,
    segment_spike_height_counts: np.ndarray | None = None,
    segment_snr: np.ndarray | None = None,
    segment_snr_noise: np.ndarray | None = None,
    segment_snr_baseline_counts: np.ndarray | None = None,
    overall_snr: float | None = None,
    snr_spike_mask_ms: float | None = None,
    snr_acceptable_until_s: float | None = None,
) -> dict[str, Any]:
    return {
        "source_file": None if source_file is None else str(source_file),
        "source_filename": None if source_filename is None else str(source_filename),
        "animal_id": str(animal_id),
        "cell_idx": int(cell_idx),
        "frame_rate": float(frame_rate),
        "trace_key": "traces",
        "snr_acceptable_until_s": None if snr_acceptable_until_s is None else float(snr_acceptable_until_s),
        "params": _serialize_params(normalize_params(**params)),
        "segment_bounds": [[int(s), int(e)] for s, e in segment_bounds_in],
        "thresholds": [float(v) if np.isfinite(v) else np.nan for v in np.asarray(thresholds, dtype=float)],
        "manual_exclusion_periods": _serialize_manual_exclusion_periods(manual_exclusion_periods),
        "spikes": [int(v) for v in np.asarray(spikes, dtype=np.int64).reshape(-1)],
        "cb_params": _serialize_cb_params(cb_params or {}),
        "first_round_cb_params": _serialize_cb_params(first_round_cb_params or cb_params or {}),
        "simple_spikes": [int(v) for v in np.asarray(simple_spikes if simple_spikes is not None else [], dtype=np.int64).reshape(-1)],
        "complex_spikes": [int(v) for v in np.asarray(complex_spikes if complex_spikes is not None else [], dtype=np.int64).reshape(-1)],
        "complex_bursts": _serialize_complex_bursts(complex_bursts),
        "first_round_complex_bursts": _serialize_complex_bursts(first_round_complex_bursts or complex_bursts),
        "second_round": second_round if isinstance(second_round, dict) else None,
        "second_round_params": _serialize_second_round_params(second_round_params or {}),
        "second_round_ss_thresholds": [
            float(v) if np.isfinite(v) else np.nan
            for v in np.asarray(second_round_ss_thresholds if second_round_ss_thresholds is not None else [], dtype=float).reshape(-1)
        ],
        "second_round_cs_thresholds": [
            float(v) if np.isfinite(v) else np.nan
            for v in np.asarray(second_round_cs_thresholds if second_round_cs_thresholds is not None else [], dtype=float).reshape(-1)
        ],
        "second_round_cb_params": _serialize_second_round_cb_params(second_round_cb_params or {}),
        "second_round_cb": second_round_cb if isinstance(second_round_cb, dict) else None,
        "plateau_params": _serialize_plateau_params(plateau_params or {}),
        "plateau_result": plateau_result if isinstance(plateau_result, dict) else None,
        "plateaus": _serialize_complex_bursts(plateaus),
        "failed_min_spikes_after_amp_duration": int(failed_min_spikes_after_amp_duration or 0),
        "failed_min_spikes_after_amp_duration_by_segment": [
            int(v)
            for v in np.asarray(
                failed_min_spikes_after_amp_duration_by_segment
                if failed_min_spikes_after_amp_duration_by_segment is not None
                else [],
                dtype=np.int64,
            ).reshape(-1)
        ],
        "failed_min_spikes_after_amp_duration_windows": _serialize_complex_bursts(
            failed_min_spikes_after_amp_duration_windows
        ),
        "first_round_failed_min_spikes_after_amp_duration": int(
            first_round_failed_min_spikes_after_amp_duration
            if first_round_failed_min_spikes_after_amp_duration is not None
            else failed_min_spikes_after_amp_duration or 0
        ),
        "first_round_failed_min_spikes_after_amp_duration_by_segment": [
            int(v)
            for v in np.asarray(
                first_round_failed_min_spikes_after_amp_duration_by_segment
                if first_round_failed_min_spikes_after_amp_duration_by_segment is not None
                else failed_min_spikes_after_amp_duration_by_segment
                if failed_min_spikes_after_amp_duration_by_segment is not None
                else [],
                dtype=np.int64,
            ).reshape(-1)
        ],
        "first_round_failed_min_spikes_after_amp_duration_windows": _serialize_complex_bursts(
            first_round_failed_min_spikes_after_amp_duration_windows
            if first_round_failed_min_spikes_after_amp_duration_windows is not None
            else failed_min_spikes_after_amp_duration_windows
        ),
        "segment_spike_heights": [
            float(v) if np.isfinite(v) else np.nan
            for v in np.asarray(segment_spike_heights if segment_spike_heights is not None else [], dtype=float).reshape(-1)
        ],
        "segment_spike_height_counts": [
            int(v)
            for v in np.asarray(segment_spike_height_counts if segment_spike_height_counts is not None else [], dtype=np.int64).reshape(-1)
        ],
        "segment_snr": [
            float(v) if np.isfinite(v) else np.nan
            for v in np.asarray(segment_snr if segment_snr is not None else [], dtype=float).reshape(-1)
        ],
        "segment_snr_noise": [
            float(v) if np.isfinite(v) else np.nan
            for v in np.asarray(segment_snr_noise if segment_snr_noise is not None else [], dtype=float).reshape(-1)
        ],
        "segment_snr_baseline_counts": [
            int(v)
            for v in np.asarray(segment_snr_baseline_counts if segment_snr_baseline_counts is not None else [], dtype=np.int64).reshape(-1)
        ],
        "overall_snr": (
            float(overall_snr)
            if overall_snr is not None and np.isfinite(float(overall_snr))
            else np.nan
        ),
        "snr_spike_mask_ms": (
            float(snr_spike_mask_ms)
            if snr_spike_mask_ms is not None and np.isfinite(float(snr_spike_mask_ms))
            else float(DEFAULT_SNR_SPIKE_MASK_MS)
        ),
    }
