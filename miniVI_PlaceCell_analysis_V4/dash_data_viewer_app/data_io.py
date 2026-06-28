"""Data loading and SNR mask helpers for the Dash trace viewer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
import sys
from typing import Any

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_ANALYSIS_ROOT = _THIS_DIR.parent
_UTILS_DIR = _ANALYSIS_ROOT / "utils"
if str(_ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS_ROOT))
if str(_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_UTILS_DIR))

from preprocess_neural import compute_time_varying_snr_from_trace  # noqa: E402
from spatial_heatmaps import cell_has_cs_place_field, is_csplus_place_cell  # noqa: E402

CSPLUS_CB_NUM_THRESHOLD = 5
CSPLUS_CS_PEAK_RATE_THRESHOLD = 0.5
CSPLUS_DEFINITION_MODE = "cs_place_field"


@dataclass
class AnimalData:
    animal_id: str
    path: Path
    merged: dict[str, Any]
    frame_rate: float
    n_frames: int
    n_cells: int
    traces: Any
    vms: Any
    all_spikes: list[np.ndarray]
    refined_ss: list[np.ndarray]
    all_cs_spikes: list[np.ndarray]
    complex_bursts: list[Any]
    plateaus: list[Any]
    pos_nan_mask: np.ndarray
    session_start_frames: list[int]
    place_cell_mask: np.ndarray
    csplus_plc_mask: np.ndarray
    plc_categories: list[str]
    cb_in_pf_counts: np.ndarray
    cs_peak_rates: np.ndarray
    place_cell_source: str
    snr_cache: dict[int, np.ndarray]


def default_data_root() -> Path:
    return _ANALYSIS_ROOT / "data"


def discover_animals(data_root: str | Path) -> list[str]:
    root = Path(data_root).expanduser().resolve()
    if not root.exists():
        return []
    animals: list[str] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if (child / "merged_aligned_data.pkl").exists() or (child / "merged_aligned_data_CS.pkl").exists():
            animals.append(child.name)
    return animals


def merged_path_for_animal(data_root: str | Path, animal_id: str) -> Path:
    animal_dir = Path(data_root).expanduser().resolve() / str(animal_id)
    primary = animal_dir / "merged_aligned_data.pkl"
    fallback = animal_dir / "merged_aligned_data_CS.pkl"
    if primary.exists():
        return primary
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Missing merged aligned data for {animal_id}: {primary} or {fallback}")


def _as_per_cell_arrays(value: Any, n_cells: int) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for cell_idx in range(n_cells):
        item = []
        if isinstance(value, (list, tuple)) and cell_idx < len(value):
            item = value[cell_idx]
        out.append(np.asarray(item, dtype=int).reshape(-1))
    return out


def _as_per_cell_entries(value: Any, n_cells: int) -> list[Any]:
    if isinstance(value, (list, tuple)):
        return [value[i] if i < len(value) else None for i in range(n_cells)]
    return [None for _ in range(n_cells)]


def _get_trace_source(merged: dict[str, Any], key: str, fallback: Any) -> Any:
    value = merged.get(key, None)
    if value is None:
        return fallback
    return value


def _set_place_cell_flag(mask: np.ndarray, cell_idx: Any, value: Any) -> None:
    try:
        idx = int(cell_idx)
    except Exception:
        return
    if 0 <= idx < mask.size:
        mask[idx] = bool(value)


def _count_cb_run_in(entry: dict[str, Any]) -> int:
    spike_shapes = entry.get("spike_shapes")
    if isinstance(spike_shapes, dict):
        complex_payload = spike_shapes.get("complex", {})
        if isinstance(complex_payload, dict):
            shapes = complex_payload.get("shapes", {})
            if isinstance(shapes, dict):
                return int(len(shapes.get("run_in", [])))
    return 0


def _is_csplus_plc(
    is_place_cell: bool,
    n_cb_in_pf: int,
    cs_peak_rate: float,
    *,
    has_cs_place_field: bool,
) -> bool:
    return is_csplus_place_cell(
        is_place_cell=bool(is_place_cell),
        n_cb_in_pf=int(n_cb_in_pf),
        cs_peak_rate=float(cs_peak_rate),
        cb_num_threshold=int(CSPLUS_CB_NUM_THRESHOLD),
        cs_peak_rate_threshold=float(CSPLUS_CS_PEAK_RATE_THRESHOLD),
        has_cs_place_field=bool(has_cs_place_field),
        cs_plc_definition_mode=CSPLUS_DEFINITION_MODE,
    )


def _load_place_cell_metadata(animal_dir: Path, n_cells: int) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray, np.ndarray, str]:
    mask = np.zeros(int(n_cells), dtype=bool)
    csplus_mask = np.zeros(int(n_cells), dtype=bool)
    categories = ["Non-PLC" for _ in range(int(n_cells))]
    cb_counts = np.zeros(int(n_cells), dtype=int)
    cs_peak_rates = np.full(int(n_cells), np.nan, dtype=float)

    spatial_path = animal_dir / "spatial_analysis_full.pkl"
    if spatial_path.exists():
        try:
            with spatial_path.open("rb") as f:
                spatial = pickle.load(f)
            if isinstance(spatial, list):
                found = False
                for pos, entry in enumerate(spatial):
                    if not isinstance(entry, dict) or "is_place_cell" not in entry:
                        continue
                    try:
                        idx = int(entry.get("cell_idx", pos))
                    except Exception:
                        continue
                    if not 0 <= idx < int(n_cells):
                        continue
                    is_pc = bool(entry.get("is_place_cell", False))
                    n_cb = _count_cb_run_in(entry)
                    try:
                        cs_peak_rate = float(entry.get("cs_peak_rate", np.nan))
                    except Exception:
                        cs_peak_rate = np.nan
                    is_csplus = _is_csplus_plc(
                        is_pc,
                        n_cb,
                        cs_peak_rate,
                        has_cs_place_field=cell_has_cs_place_field(entry),
                    )
                    mask[idx] = is_pc
                    csplus_mask[idx] = is_csplus
                    categories[idx] = "CS+ PLC" if is_csplus else ("CS- PLC" if is_pc else "Non-PLC")
                    cb_counts[idx] = int(n_cb)
                    cs_peak_rates[idx] = cs_peak_rate
                    found = True
                if found:
                    return mask, csplus_mask, categories, cb_counts, cs_peak_rates, spatial_path.name
        except Exception:
            pass

    pooled_path = animal_dir / "pooled_stats.pkl"
    if pooled_path.exists():
        try:
            with pooled_path.open("rb") as f:
                pooled = pickle.load(f)
            if hasattr(pooled, "iterrows") and "is_place_cell" in getattr(pooled, "columns", []):
                found = False
                for pos, row in pooled.iterrows():
                    try:
                        idx = int(row.get("cell_idx", pos))
                    except Exception:
                        continue
                    if not 0 <= idx < int(n_cells):
                        continue
                    is_pc = bool(row.get("is_place_cell", False))
                    n_cb = int(row.get("n_cb_in_pf", 0)) if "n_cb_in_pf" in getattr(pooled, "columns", []) else 0
                    try:
                        cs_peak_rate = float(row.get("cs_peak_rate", np.nan))
                    except Exception:
                        cs_peak_rate = np.nan
                    is_csplus = _is_csplus_plc(
                        is_pc,
                        n_cb,
                        cs_peak_rate,
                        has_cs_place_field=cell_has_cs_place_field(row.to_dict()),
                    )
                    mask[idx] = is_pc
                    csplus_mask[idx] = is_csplus
                    categories[idx] = "CS+ PLC" if is_csplus else ("CS- PLC" if is_pc else "Non-PLC")
                    cb_counts[idx] = int(n_cb)
                    cs_peak_rates[idx] = cs_peak_rate
                    found = True
                if found:
                    return mask, csplus_mask, categories, cb_counts, cs_peak_rates, pooled_path.name
        except Exception:
            pass

    return mask, csplus_mask, categories, cb_counts, cs_peak_rates, "none"


def load_animal(data_root: str | Path, animal_id: str) -> AnimalData:
    path = merged_path_for_animal(data_root, animal_id)
    animal_dir = path.parent
    with path.open("rb") as f:
        merged = pickle.load(f)

    required = ["x_neural", "speed", "spikes", "frame_rate"]
    missing = [key for key in required if key not in merged]
    if missing:
        raise KeyError(f"{animal_id}: merged data is missing required keys: {missing}")

    x_neural = np.asarray(merged["x_neural"], dtype=float).reshape(-1)
    y_neural = np.asarray(merged.get("y_neural", np.full_like(x_neural, np.nan)), dtype=float).reshape(-1)
    speed = np.asarray(merged.get("speed", np.full_like(x_neural, np.nan)), dtype=float).reshape(-1)
    n_frames = int(x_neural.size)
    frame_rate = float(merged.get("frame_rate", np.nan))
    if (not np.isfinite(frame_rate)) or frame_rate <= 0:
        frame_rate = 1.0

    n_cells = int(len(merged["spikes"]))
    traces = _get_trace_source(merged, "traces_SNR_interpolated", merged.get("traces", []))
    vms = _get_trace_source(merged, "Vm_SNR_interpolated", traces)
    all_spikes = _as_per_cell_arrays(merged.get("all_spikes", merged.get("spikes", [])), n_cells)
    refined_ss = _as_per_cell_arrays(merged.get("refined_SS", []), n_cells)
    all_cs_spikes = _as_per_cell_arrays(merged.get("all_CS_spikes", []), n_cells)
    complex_bursts = _as_per_cell_entries(merged.get("complex_bursts_dicts", []), n_cells)
    plateaus = _as_per_cell_entries(merged.get("plateaus_dicts", []), n_cells)
    session_start_frames = [int(v) for v in np.asarray(merged.get("session_start_frames", [0]), dtype=int).reshape(-1)]
    if not session_start_frames:
        session_start_frames = [0]

    pos_nan_mask = (~np.isfinite(x_neural)) | (~np.isfinite(y_neural)) | (~np.isfinite(speed))
    if pos_nan_mask.size != n_frames:
        pos_nan_mask = np.ones(n_frames, dtype=bool)
    (
        place_cell_mask,
        csplus_plc_mask,
        plc_categories,
        cb_in_pf_counts,
        cs_peak_rates,
        place_cell_source,
    ) = _load_place_cell_metadata(animal_dir, n_cells)

    return AnimalData(
        animal_id=str(animal_id),
        path=path,
        merged=merged,
        frame_rate=frame_rate,
        n_frames=n_frames,
        n_cells=n_cells,
        traces=traces,
        vms=vms,
        all_spikes=all_spikes,
        refined_ss=refined_ss,
        all_cs_spikes=all_cs_spikes,
        complex_bursts=complex_bursts,
        plateaus=plateaus,
        pos_nan_mask=pos_nan_mask,
        session_start_frames=session_start_frames,
        place_cell_mask=place_cell_mask,
        csplus_plc_mask=csplus_plc_mask,
        plc_categories=plc_categories,
        cb_in_pf_counts=cb_in_pf_counts,
        cs_peak_rates=cs_peak_rates,
        place_cell_source=place_cell_source,
        snr_cache={},
    )


def get_cell_vector(source: Any, cell_idx: int, n_frames: int) -> np.ndarray:
    if isinstance(source, np.ndarray) and source.ndim >= 2 and cell_idx < source.shape[0]:
        arr = np.asarray(source[cell_idx], dtype=float).reshape(-1)
    elif isinstance(source, (list, tuple)) and cell_idx < len(source):
        arr = np.asarray(source[cell_idx], dtype=float).reshape(-1)
    else:
        arr = np.full(n_frames, np.nan, dtype=float)

    if arr.size == n_frames:
        return arr
    out = np.full(n_frames, np.nan, dtype=float)
    n = min(n_frames, arr.size)
    if n > 0:
        out[:n] = arr[:n]
    return out


def compute_snr_values(animal: AnimalData, cell_idx: int) -> np.ndarray:
    cell_idx = int(cell_idx)
    cached = animal.snr_cache.get(cell_idx)
    if cached is not None:
        return cached

    trace = get_cell_vector(animal.traces, cell_idx, animal.n_frames)
    spks = animal.all_spikes[cell_idx] if cell_idx < len(animal.all_spikes) else np.array([], dtype=int)
    cb = animal.complex_bursts[cell_idx] if cell_idx < len(animal.complex_bursts) else None
    if trace.ndim != 1 or trace.size != animal.n_frames:
        snr_vals = np.full(animal.n_frames, np.nan, dtype=float)
    else:
        snr_res = compute_time_varying_snr_from_trace(
            trace=trace,
            spks=spks,
            complex_burst_dict=cb,
            sampling_rate_hz=animal.frame_rate,
            isi_threshold_ms=20,
            spike_baseline_points=3,
            spike_remove_points=3,
            baseline_window_seconds=10,
            plot_single_cell=False,
        )
        snr_vals = np.asarray(snr_res.get("snr_time_varying", []), dtype=float).reshape(-1)
        if snr_vals.size != animal.n_frames:
            raise RuntimeError(
                f"{animal.animal_id} cell {cell_idx}: invalid SNR length {snr_vals.size}, expected {animal.n_frames}"
            )

    animal.snr_cache[cell_idx] = snr_vals
    return snr_vals


def source_trace_bad_mask(animal: AnimalData, cell_idx: int) -> np.ndarray:
    """Recover manually excluded/interpolated frames from non-interpolated trace sources."""
    out = np.zeros(int(animal.n_frames), dtype=bool)
    for key in ("traces", "SNR_interpolated", "spike_heights_interpolated"):
        source = animal.merged.get(key, None)
        if source is None:
            continue
        arr = None
        try:
            if isinstance(source, (list, tuple)):
                if int(cell_idx) < len(source):
                    arr = np.asarray(source[int(cell_idx)], dtype=float).reshape(-1)
            else:
                data = np.asarray(source, dtype=float)
                if data.ndim == 1:
                    arr = data
                elif data.ndim == 2:
                    if data.shape[0] > int(cell_idx) and data.shape[1] == int(animal.n_frames):
                        arr = data[int(cell_idx), :]
                    elif data.shape[1] > int(cell_idx) and data.shape[0] == int(animal.n_frames):
                        arr = data[:, int(cell_idx)]
        except Exception:
            arr = None
        if arr is None or arr.size != int(animal.n_frames):
            continue
        out |= ~np.isfinite(arr)
    return out


def bad_mask_for_cell(animal: AnimalData, cell_idx: int, snr_threshold: float, min_good_minutes: float) -> tuple[np.ndarray, dict[str, Any]]:
    snr_vals = compute_snr_values(animal, cell_idx)
    source_bad = source_trace_bad_mask(animal, cell_idx)
    good_snr = np.isfinite(snr_vals) & (snr_vals >= float(snr_threshold))
    n_good_before = int(np.sum(good_snr))
    good_minutes = float(n_good_before) / float(animal.frame_rate) / 60.0
    removed_by_min_minutes = bool(good_minutes < float(min_good_minutes))
    bad_mask = (~good_snr) | animal.pos_nan_mask | source_bad
    bad_mask = np.asarray(bad_mask, dtype=bool)
    removed_source_trace = int(np.sum(bad_mask & source_bad))
    removed_snr_only = int(np.sum(bad_mask & (~animal.pos_nan_mask) & (~source_bad)))
    stats = {
        "cell_idx": int(cell_idx),
        "n_good_frames_before_min_minutes": n_good_before,
        "good_minutes_before_min_minutes": good_minutes,
        "removed_by_min_good_minutes": removed_by_min_minutes,
        "eligible_cell": bool((not removed_by_min_minutes) and np.any(~bad_mask)),
        "n_removed_frames_total": int(np.sum(bad_mask)),
        "n_removed_frames_source_trace_bad": removed_source_trace,
        "n_removed_frames_snr_only": removed_snr_only,
        "pct_removed_frames_total": (100.0 * float(np.sum(bad_mask)) / float(animal.n_frames)) if animal.n_frames > 0 else np.nan,
        "pct_removed_frames_source_trace_bad": (100.0 * float(removed_source_trace) / float(animal.n_frames)) if animal.n_frames > 0 else np.nan,
    }
    return bad_mask, stats


def mask_to_intervals(mask: np.ndarray) -> list[tuple[int, int]]:
    arr = np.asarray(mask, dtype=bool).reshape(-1)
    if arr.size == 0 or not np.any(arr):
        return []
    padded = np.concatenate([[False], arr, [False]])
    diff = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(diff == 1)
    ends_excl = np.flatnonzero(diff == -1)
    return [(int(s), int(e)) for s, e in zip(starts, ends_excl) if e > s]


def dict_intervals(entry: Any, n_frames: int) -> list[tuple[int, int]]:
    if not isinstance(entry, dict):
        return []
    starts = np.asarray(entry.get("starts", []), dtype=int).reshape(-1)
    ends = np.asarray(entry.get("ends", []), dtype=int).reshape(-1)
    n = min(starts.size, ends.size)
    out: list[tuple[int, int]] = []
    for s, e in zip(starts[:n], ends[:n]):
        lo = int(min(s, e))
        hi = int(max(s, e)) + 1
        if hi <= 0 or lo >= n_frames:
            continue
        out.append((max(0, lo), min(n_frames, hi)))
    return out
