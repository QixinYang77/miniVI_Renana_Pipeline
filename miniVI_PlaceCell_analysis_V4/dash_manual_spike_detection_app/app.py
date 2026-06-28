"""Dash GUI for manual all-spike threshold tuning.

Usage
-----
    python dash_manual_spike_detection_app/app.py --data-root miniVI_PlaceCell_analysis_V4/data --port 8054
"""

from __future__ import annotations

import argparse
from functools import lru_cache
import json
from pathlib import Path
import re
import sys
import threading
import webbrowser
from typing import Any

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_ANALYSIS_ROOT = _THIS_DIR.parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
if str(_ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS_ROOT))

from dash import ALL, Dash, Input, Output, State, ctx, dcc, html  # noqa: E402
from dash.exceptions import PreventUpdate  # noqa: E402

try:
    from .data_io import (  # noqa: E402
        SOURCE_FILENAME,
        TRACE_KEY,
        default_data_root,
        discover_animals,
        get_saved_cell,
        load_animal,
        load_sidecar,
        save_cell_result,
    )
    from .figure_builder import (  # noqa: E402
        COMPLEX_SPIKE_COLOR,
        DEFAULT_ROW_HEIGHT_PX,
        MAX_ROW_HEIGHT_PX,
        MIN_ROW_HEIGHT_PX,
        OVERALL_SNR_COLOR,
        THRESHOLD_COLOR,
        build_detection_figures,
        build_saved_review_figures,
        build_waveform_shape_figure,
        empty_figure,
    )
    from .pipeline import (  # noqa: E402
        DEFAULT_BASELINE_WINDOW_S,
        DEFAULT_CB_AMP_THRESHOLD,
        DEFAULT_CB_BASELINE_WINDOW_S,
        DEFAULT_CB_DURATION_THRESHOLD_MS,
        DEFAULT_CB_INCLUDE_FIRST_BURST_SPIKE_FOR_SPIKE_HEIGHT,
        DEFAULT_CB_ISI_THRESHOLD_MS,
        DEFAULT_CB_MAX_ONSET_LEAD_MS,
        DEFAULT_CB_MIN_SPIKES,
        DEFAULT_CB_OFFSET_THRESHOLD,
        DEFAULT_CB_ONSET_THRESHOLD,
        DEFAULT_CB_REFINE_ONSET,
        DEFAULT_CB_REQUIRE_MIN_ISI,
        DEFAULT_CB_REMOVE_SPIKES_FOR_VM,
        DEFAULT_CB_SPIKE_HEIGHT_MIN_ISOLATED_SPIKES,
        DEFAULT_HIGHPASS_HZ,
        DEFAULT_PLATEAU_AMP_THRESHOLD,
        DEFAULT_PLATEAU_BASELINE_WINDOW_S,
        DEFAULT_PLATEAU_DURATION_THRESHOLD_MS,
        DEFAULT_PLATEAU_MIN_SPIKES,
        DEFAULT_PLATEAU_OFFSET_THRESHOLD,
        DEFAULT_PLATEAU_ONSET_THRESHOLD,
        DEFAULT_PLATEAU_PEAK_FRACTION_DURATION_MS,
        DEFAULT_PLATEAU_PEAK_FRACTION_THRESHOLD,
        DEFAULT_PLATEAU_VM_CROSSING_THRESHOLD,
        DEFAULT_PLATEAU_VM_MEDIAN_WINDOW_MS,
        DEFAULT_SEGMENT_DURATION_S,
        DEFAULT_SECOND_ROUND_CB_AMP_THRESHOLD,
        DEFAULT_SECOND_ROUND_CB_MIN_SPIKES,
        DEFAULT_SECOND_ROUND_CS_THRESHOLD_MAD,
        DEFAULT_SECOND_ROUND_REFINE_SIMPLE_SPIKES,
        DEFAULT_SECOND_ROUND_SIMPLE_SPIKE_MIN_HEIGHT_FRACTION,
        DEFAULT_SECOND_ROUND_SS_THRESHOLD_MAD,
        DEFAULT_SPIKE_BASELINE_REMOVE_ENABLED,
        DEFAULT_SPIKE_BASELINE_WINDOW_MS,
        DEFAULT_THRESHOLD_MAD,
        DEFAULT_VM_CROSSING_THRESHOLD,
        DEFAULT_VM_MEDIAN_WINDOW_MS,
        calculate_display_vm,
        calculate_segment_snr,
        default_thresholds_for_segments,
        detect_complex_bursts_segmented,
        detect_plateaus_segmented,
        detect_second_round_spikes,
        detect_spikes_in_windows_from_thresholds,
        detect_spikes_from_thresholds,
        make_result_payload,
        normalize_cb_params,
        normalize_params,
        normalize_plateau_params,
        normalize_second_round_cb_params,
        normalize_second_round_params,
        prepare_detection_inputs,
        sanitize_thresholds,
        second_round_segment_thresholds,
    )
except ImportError:  # pragma: no cover - direct script execution
    from data_io import (
        SOURCE_FILENAME,
        TRACE_KEY,
        default_data_root,
        discover_animals,
        get_saved_cell,
        load_animal,
        load_sidecar,
        save_cell_result,
    )
    from figure_builder import (
        COMPLEX_SPIKE_COLOR,
        DEFAULT_ROW_HEIGHT_PX,
        MAX_ROW_HEIGHT_PX,
        MIN_ROW_HEIGHT_PX,
        OVERALL_SNR_COLOR,
        THRESHOLD_COLOR,
        build_detection_figures,
        build_saved_review_figures,
        build_waveform_shape_figure,
        empty_figure,
    )
    from pipeline import (
        DEFAULT_BASELINE_WINDOW_S,
        DEFAULT_CB_AMP_THRESHOLD,
        DEFAULT_CB_BASELINE_WINDOW_S,
        DEFAULT_CB_DURATION_THRESHOLD_MS,
        DEFAULT_CB_INCLUDE_FIRST_BURST_SPIKE_FOR_SPIKE_HEIGHT,
        DEFAULT_CB_ISI_THRESHOLD_MS,
        DEFAULT_CB_MAX_ONSET_LEAD_MS,
        DEFAULT_CB_MIN_SPIKES,
        DEFAULT_CB_OFFSET_THRESHOLD,
        DEFAULT_CB_ONSET_THRESHOLD,
        DEFAULT_CB_REFINE_ONSET,
        DEFAULT_CB_REQUIRE_MIN_ISI,
        DEFAULT_CB_REMOVE_SPIKES_FOR_VM,
        DEFAULT_CB_SPIKE_HEIGHT_MIN_ISOLATED_SPIKES,
        DEFAULT_HIGHPASS_HZ,
        DEFAULT_PLATEAU_AMP_THRESHOLD,
        DEFAULT_PLATEAU_BASELINE_WINDOW_S,
        DEFAULT_PLATEAU_DURATION_THRESHOLD_MS,
        DEFAULT_PLATEAU_MIN_SPIKES,
        DEFAULT_PLATEAU_OFFSET_THRESHOLD,
        DEFAULT_PLATEAU_ONSET_THRESHOLD,
        DEFAULT_PLATEAU_PEAK_FRACTION_DURATION_MS,
        DEFAULT_PLATEAU_PEAK_FRACTION_THRESHOLD,
        DEFAULT_PLATEAU_VM_CROSSING_THRESHOLD,
        DEFAULT_PLATEAU_VM_MEDIAN_WINDOW_MS,
        DEFAULT_SEGMENT_DURATION_S,
        DEFAULT_SECOND_ROUND_CB_AMP_THRESHOLD,
        DEFAULT_SECOND_ROUND_CB_MIN_SPIKES,
        DEFAULT_SECOND_ROUND_CS_THRESHOLD_MAD,
        DEFAULT_SECOND_ROUND_REFINE_SIMPLE_SPIKES,
        DEFAULT_SECOND_ROUND_SIMPLE_SPIKE_MIN_HEIGHT_FRACTION,
        DEFAULT_SECOND_ROUND_SS_THRESHOLD_MAD,
        DEFAULT_SPIKE_BASELINE_REMOVE_ENABLED,
        DEFAULT_SPIKE_BASELINE_WINDOW_MS,
        DEFAULT_THRESHOLD_MAD,
        DEFAULT_VM_CROSSING_THRESHOLD,
        DEFAULT_VM_MEDIAN_WINDOW_MS,
        calculate_display_vm,
        calculate_segment_snr,
        default_thresholds_for_segments,
        detect_complex_bursts_segmented,
        detect_plateaus_segmented,
        detect_second_round_spikes,
        detect_spikes_in_windows_from_thresholds,
        detect_spikes_from_thresholds,
        make_result_payload,
        normalize_cb_params,
        normalize_params,
        normalize_plateau_params,
        normalize_second_round_cb_params,
        normalize_second_round_params,
        prepare_detection_inputs,
        sanitize_thresholds,
        second_round_segment_thresholds,
    )


CARD_STYLE = {
    "border": "1px solid #ddd",
    "padding": "8px",
    "borderRadius": "6px",
    "backgroundColor": "white",
    "marginBottom": "8px",
}
LABEL_STYLE = {"fontSize": "12px", "marginBottom": "2px", "fontWeight": 600}
INPUT_STYLE = {"width": "100%", "fontSize": "12px"}
BUTTON_STYLE = {
    "height": "34px",
    "fontSize": "12px",
    "border": "1px solid #bbb",
    "borderRadius": "4px",
    "backgroundColor": "#f8f8f8",
    "cursor": "pointer",
}
PRIMARY_BUTTON_STYLE = {
    **BUTTON_STYLE,
    "backgroundColor": "#222",
    "border": "1px solid #222",
    "color": "white",
}
SECTION_HEADER = {"marginTop": "4px", "marginBottom": "6px", "fontSize": "13px", "fontWeight": 600}
STATUS_STYLE = {"fontSize": "11px", "color": "#666", "lineHeight": "1.35", "marginTop": "6px"}
FAILED_CB_TEXT_COLOR = "#7B2CBF"

LEFT_GRAPH_CONFIG = {
    "scrollZoom": False,
    "doubleClick": "reset",
    "displaylogo": False,
    "responsive": True,
    "editable": True,
    "edits": {"shapePosition": True},
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d", "pan2d"],
}
RIGHT_GRAPH_CONFIG = {
    "scrollZoom": False,
    "doubleClick": "reset",
    "displaylogo": False,
    "responsive": True,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d", "pan2d"],
}
REVIEW_GRAPH_CONFIG = {
    "scrollZoom": False,
    "doubleClick": "reset",
    "displaylogo": False,
    "responsive": True,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d", "pan2d"],
}
TAB_PANEL_VISIBLE_STYLE = {
    "minWidth": 0,
    "minHeight": 0,
    "height": "100%",
    "display": "grid",
    "gridTemplateRows": "auto 1fr",
    "gap": "6px",
}
TAB_PANEL_HIDDEN_STYLE = {
    **TAB_PANEL_VISIBLE_STYLE,
    "display": "none",
}
RIGHT_PANEL_MODE_NORMALIZED = "spike_height_normalized"
RIGHT_PANEL_MODE_PLATEAU = "plateau_normalized"
RIGHT_PANEL_MODE_TRACE = "trace"
RIGHT_PANEL_MODE_DEFAULT = RIGHT_PANEL_MODE_NORMALIZED
SECOND_ROUND_DETECTION_ORDER = "ss_all_refine_cb_true_cs"
SAVED_REVIEW_ROW_DURATION_S = 10.0


def _number_row(
    label: str,
    input_id: str,
    value: float,
    *,
    step: float | str,
    min_value: float | None = None,
    disabled: bool = False,
):
    kwargs = {
        "id": input_id,
        "type": "number",
        "value": value,
        "step": step,
        "debounce": False,
        "disabled": disabled,
        "style": INPUT_STYLE,
    }
    if min_value is not None:
        kwargs["min"] = min_value
    return html.Div(
        [
            html.Label(label, style=LABEL_STYLE),
            dcc.Input(**kwargs),
        ],
        style={"marginBottom": "6px"},
    )


def _animal_options(animals: list[str]) -> list[dict[str, str]]:
    return [{"label": animal_id, "value": animal_id} for animal_id in animals]


def _safe_int(value: Any, fallback: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(fallback)


def _safe_float(value: Any, fallback: float) -> float:
    try:
        out = float(value)
    except Exception:
        return float(fallback)
    if not np.isfinite(out):
        return float(fallback)
    return float(out)


def _as_float_array(values: Any) -> np.ndarray:
    try:
        raw = np.asarray(values, dtype=object).reshape(-1)
    except Exception:
        return np.array([], dtype=float)
    out = np.full(raw.size, np.nan, dtype=float)
    for idx, item in enumerate(raw):
        try:
            value = float(item)
        except Exception:
            continue
        if np.isfinite(value):
            out[idx] = value
    return out


def _safe_bool(value: Any, fallback: bool = False) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on", "enabled"}
    if value is None:
        return bool(fallback)
    return bool(value)


def _clean_row_height(value: Any) -> int:
    row_height = _safe_float(value, DEFAULT_ROW_HEIGHT_PX)
    row_height = min(MAX_ROW_HEIGHT_PX, max(MIN_ROW_HEIGHT_PX, row_height))
    return int(round(row_height))


def _clean_right_panel_mode(value: Any) -> str:
    if value == RIGHT_PANEL_MODE_TRACE:
        return RIGHT_PANEL_MODE_TRACE
    if value == RIGHT_PANEL_MODE_PLATEAU:
        return RIGHT_PANEL_MODE_PLATEAU
    return RIGHT_PANEL_MODE_NORMALIZED


def _right_panel_name(mode: str) -> str:
    if mode == RIGHT_PANEL_MODE_TRACE:
        return "Trace"
    if mode == RIGHT_PANEL_MODE_PLATEAU:
        return "Plateau-normalized trace"
    return "Spike-normalized trace"


def _format_snr_value(value: Any) -> str:
    snr = _safe_float(value, float("nan"))
    if not np.isfinite(snr):
        return "n/a"
    if abs(snr) >= 100:
        return f"{snr:.0f}"
    if abs(snr) >= 10:
        return f"{snr:.1f}"
    return f"{snr:.2f}"


def _normalize_gui_defaults(gui_defaults: dict[str, Any] | None = None) -> dict[str, Any]:
    raw = dict(gui_defaults or {})
    raw_params = raw.get("params", {})
    raw_cb_params = raw.get("cb_params", {})
    raw_second_round = raw.get("second_round_params", {})
    raw_second_round_cb = raw.get("second_round_cb_params", {})
    raw_plateau = raw.get("plateau_params", {})
    raw_display = raw.get("display", {})
    if not isinstance(raw_params, dict):
        raw_params = {}
    if not isinstance(raw_cb_params, dict):
        raw_cb_params = {}
    if not isinstance(raw_second_round, dict):
        raw_second_round = {}
    if not isinstance(raw_second_round_cb, dict):
        raw_second_round_cb = {}
    if not isinstance(raw_plateau, dict):
        raw_plateau = {}
    if not isinstance(raw_display, dict):
        raw_display = {}

    params = normalize_params(
        baseline_window_s=raw.get("baseline_window_s", raw_params.get("baseline_window_s", DEFAULT_BASELINE_WINDOW_S)),
        segment_duration_s=raw.get("segment_duration_s", raw_params.get("segment_duration_s", DEFAULT_SEGMENT_DURATION_S)),
        highpass_hz=raw.get("highpass_hz", raw_params.get("highpass_hz", DEFAULT_HIGHPASS_HZ)),
        threshold_mad=raw.get("threshold_mad", raw_params.get("threshold_mad", DEFAULT_THRESHOLD_MAD)),
        spike_baseline_remove_enabled=raw.get(
            "spike_baseline_remove_enabled",
            raw_params.get("spike_baseline_remove_enabled", DEFAULT_SPIKE_BASELINE_REMOVE_ENABLED),
        ),
        spike_baseline_window_ms=raw.get(
            "spike_baseline_window_ms",
            raw_params.get("spike_baseline_window_ms", DEFAULT_SPIKE_BASELINE_WINDOW_MS),
        ),
    )
    cb_params = normalize_cb_params(
        cb_baseline_window_s=raw.get(
            "cb_baseline_window_s",
            raw_cb_params.get("cb_baseline_window_s", DEFAULT_CB_BASELINE_WINDOW_S),
        ),
        remove_spikes_for_vm=raw.get(
            "remove_spikes_for_vm",
            raw_cb_params.get("remove_spikes_for_vm", DEFAULT_CB_REMOVE_SPIKES_FOR_VM),
        ),
        vm_median_window_ms=raw.get(
            "vm_median_window_ms",
            raw_cb_params.get("vm_median_window_ms", DEFAULT_VM_MEDIAN_WINDOW_MS),
        ),
        vm_crossing_threshold=raw.get(
            "vm_crossing_threshold",
            raw_cb_params.get("vm_crossing_threshold", DEFAULT_VM_CROSSING_THRESHOLD),
        ),
        cb_amp_threshold=raw.get(
            "cb_amp_threshold",
            raw_cb_params.get("cb_amp_threshold", DEFAULT_CB_AMP_THRESHOLD),
        ),
        cb_duration_threshold_ms=raw.get(
            "cb_duration_threshold_ms",
            raw_cb_params.get("cb_duration_threshold_ms", DEFAULT_CB_DURATION_THRESHOLD_MS),
        ),
        cb_min_spikes=raw.get("cb_min_spikes", raw_cb_params.get("cb_min_spikes", DEFAULT_CB_MIN_SPIKES)),
        cb_isi_threshold_ms=raw.get(
            "cb_isi_threshold_ms",
            raw_cb_params.get("cb_isi_threshold_ms", DEFAULT_CB_ISI_THRESHOLD_MS),
        ),
        cb_require_min_isi=raw.get(
            "cb_require_min_isi",
            raw_cb_params.get("cb_require_min_isi", DEFAULT_CB_REQUIRE_MIN_ISI),
        ),
        spike_height_min_isolated_spikes=raw.get(
            "spike_height_min_isolated_spikes",
            raw_cb_params.get(
                "spike_height_min_isolated_spikes",
                DEFAULT_CB_SPIKE_HEIGHT_MIN_ISOLATED_SPIKES,
            ),
        ),
        include_first_burst_spike_for_spike_height=raw.get(
            "include_first_burst_spike_for_spike_height",
            raw_cb_params.get(
                "include_first_burst_spike_for_spike_height",
                DEFAULT_CB_INCLUDE_FIRST_BURST_SPIKE_FOR_SPIKE_HEIGHT,
            ),
        ),
        refine_cb_onset=raw.get(
            "refine_cb_onset",
            raw_cb_params.get("refine_cb_onset", DEFAULT_CB_REFINE_ONSET),
        ),
        cb_onset_threshold=raw.get(
            "cb_onset_threshold",
            raw_cb_params.get("cb_onset_threshold", DEFAULT_CB_ONSET_THRESHOLD),
        ),
        cb_offset_threshold=raw.get(
            "cb_offset_threshold",
            raw_cb_params.get("cb_offset_threshold", DEFAULT_CB_OFFSET_THRESHOLD),
        ),
        cb_max_onset_lead_ms=raw.get(
            "cb_max_onset_lead_ms",
            raw_cb_params.get("cb_max_onset_lead_ms", DEFAULT_CB_MAX_ONSET_LEAD_MS),
        ),
    )
    second_round_params = normalize_second_round_params(
        ss_threshold_mad=raw.get(
            "second_round_ss_threshold_mad",
            raw_second_round.get("ss_threshold_mad", DEFAULT_SECOND_ROUND_SS_THRESHOLD_MAD),
        ),
        cs_threshold_mad=raw.get(
            "second_round_cs_threshold_mad",
            raw_second_round.get("cs_threshold_mad", DEFAULT_SECOND_ROUND_CS_THRESHOLD_MAD),
        ),
        refine_simple_spikes_by_height=raw.get(
            "second_round_refine_simple_spikes_by_height",
            raw_second_round.get("refine_simple_spikes_by_height", DEFAULT_SECOND_ROUND_REFINE_SIMPLE_SPIKES),
        ),
        simple_spike_min_height_fraction=raw.get(
            "second_round_simple_spike_min_height_fraction",
            raw_second_round.get(
                "simple_spike_min_height_fraction",
                DEFAULT_SECOND_ROUND_SIMPLE_SPIKE_MIN_HEIGHT_FRACTION,
            ),
        ),
    )
    second_round_cb_params = normalize_second_round_cb_params(
        cb_amp_threshold=raw.get(
            "second_round_cb_amp_threshold",
            raw_second_round_cb.get("cb_amp_threshold", DEFAULT_SECOND_ROUND_CB_AMP_THRESHOLD),
        ),
        cb_min_spikes=raw.get(
            "second_round_cb_min_spikes",
            raw_second_round_cb.get("cb_min_spikes", DEFAULT_SECOND_ROUND_CB_MIN_SPIKES),
        ),
    )
    plateau_params = normalize_plateau_params(
        plateau_baseline_window_s=raw.get(
            "plateau_baseline_window_s",
            raw_plateau.get("plateau_baseline_window_s", DEFAULT_PLATEAU_BASELINE_WINDOW_S),
        ),
        plateau_vm_median_window_ms=raw.get(
            "plateau_vm_median_window_ms",
            raw_plateau.get("plateau_vm_median_window_ms", DEFAULT_PLATEAU_VM_MEDIAN_WINDOW_MS),
        ),
        plateau_vm_crossing_threshold=raw.get(
            "plateau_vm_crossing_threshold",
            raw_plateau.get("plateau_vm_crossing_threshold", DEFAULT_PLATEAU_VM_CROSSING_THRESHOLD),
        ),
        plateau_onset_threshold=raw.get(
            "plateau_onset_threshold",
            raw_plateau.get("plateau_onset_threshold", DEFAULT_PLATEAU_ONSET_THRESHOLD),
        ),
        plateau_offset_threshold=raw.get(
            "plateau_offset_threshold",
            raw_plateau.get("plateau_offset_threshold", DEFAULT_PLATEAU_OFFSET_THRESHOLD),
        ),
        plateau_amp_threshold=raw.get(
            "plateau_amp_threshold",
            raw_plateau.get("plateau_amp_threshold", DEFAULT_PLATEAU_AMP_THRESHOLD),
        ),
        plateau_duration_threshold_ms=raw.get(
            "plateau_duration_threshold_ms",
            raw_plateau.get("plateau_duration_threshold_ms", DEFAULT_PLATEAU_DURATION_THRESHOLD_MS),
        ),
        plateau_min_spikes=raw.get(
            "plateau_min_spikes",
            raw_plateau.get("plateau_min_spikes", DEFAULT_PLATEAU_MIN_SPIKES),
        ),
        plateau_peak_fraction_threshold=raw.get(
            "plateau_peak_fraction_threshold",
            raw_plateau.get("plateau_peak_fraction_threshold", DEFAULT_PLATEAU_PEAK_FRACTION_THRESHOLD),
        ),
        plateau_peak_fraction_duration_ms=raw.get(
            "plateau_peak_fraction_duration_ms",
            raw_plateau.get("plateau_peak_fraction_duration_ms", DEFAULT_PLATEAU_PEAK_FRACTION_DURATION_MS),
        ),
    )
    row_height_px = _clean_row_height(raw.get("segment_height_px", raw_display.get("segment_height_px", DEFAULT_ROW_HEIGHT_PX)))
    right_panel_mode = _clean_right_panel_mode(
        raw.get("right_panel_mode", raw_display.get("right_panel_mode", RIGHT_PANEL_MODE_DEFAULT))
    )
    use_saved_cell_parameters = _safe_bool(raw.get("use_saved_cell_parameters", True), True)
    return {
        "params": params,
        "cb_params": cb_params,
        "second_round_params": second_round_params,
        "second_round_cb_params": second_round_cb_params,
        "plateau_params": plateau_params,
        "segment_height_px": int(row_height_px),
        "right_panel_mode": right_panel_mode,
        "use_saved_cell_parameters": use_saved_cell_parameters,
    }


def _load_gui_defaults_json(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    defaults_path = Path(path).expanduser()
    if not defaults_path.is_file():
        raise FileNotFoundError(f"Cannot find GUI defaults JSON: {defaults_path}")
    with defaults_path.open("r") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"GUI defaults JSON must contain an object: {defaults_path}")
    return data


def _spike_baseline_checklist_value(enabled: Any) -> list[str]:
    return ["enabled"] if normalize_params(spike_baseline_remove_enabled=enabled)["spike_baseline_remove_enabled"] else []


def _spike_baseline_enabled(value: Any) -> bool:
    return bool(normalize_params(spike_baseline_remove_enabled=value)["spike_baseline_remove_enabled"])


def _cb_remove_spikes_checklist_value(enabled: Any) -> list[str]:
    return ["enabled"] if normalize_cb_params(remove_spikes_for_vm=enabled)["remove_spikes_for_vm"] else []


def _cb_require_min_isi_checklist_value(enabled: Any) -> list[str]:
    return ["enabled"] if normalize_cb_params(cb_require_min_isi=enabled)["cb_require_min_isi"] else []


def _cb_include_first_burst_spike_checklist_value(enabled: Any) -> list[str]:
    return (
        ["enabled"]
        if normalize_cb_params(include_first_burst_spike_for_spike_height=enabled)[
            "include_first_burst_spike_for_spike_height"
        ]
        else []
    )


def _cb_refine_onset_checklist_value(enabled: Any) -> list[str]:
    return ["enabled"] if normalize_cb_params(refine_cb_onset=enabled)["refine_cb_onset"] else []


def _cb_require_min_isi_enabled(value: Any) -> bool:
    return bool(normalize_cb_params(cb_require_min_isi=value)["cb_require_min_isi"])


def _second_round_refine_simple_spikes_checklist_value(enabled: Any) -> list[str]:
    return (
        ["enabled"]
        if normalize_second_round_params(refine_simple_spikes_by_height=enabled)[
            "refine_simple_spikes_by_height"
        ]
        else []
    )


def _second_round_refine_simple_spikes_enabled(value: Any) -> bool:
    return bool(
        normalize_second_round_params(refine_simple_spikes_by_height=value)[
            "refine_simple_spikes_by_height"
        ]
    )


def _second_round_full_cb_params(base_cb_params: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    base = normalize_cb_params(**base_cb_params)
    clean_overrides = normalize_second_round_cb_params(**overrides)
    base["cb_amp_threshold"] = clean_overrides["cb_amp_threshold"]
    base["cb_min_spikes"] = clean_overrides["cb_min_spikes"]
    return normalize_cb_params(**base)


def _max_time_s(bundle) -> float:
    return float(bundle.n_frames) / float(bundle.frame_rate)


def _clean_snr_until(value: Any, max_time_s: float) -> float:
    max_time_s = max(0.0, float(max_time_s))
    out = _safe_float(value, max_time_s)
    return float(min(max_time_s, max(0.0, out)))


def _saved_snr_until(saved_cell: dict[str, Any] | None, max_time_s: float) -> float:
    if isinstance(saved_cell, dict) and "snr_acceptable_until_s" in saved_cell:
        return _clean_snr_until(saved_cell.get("snr_acceptable_until_s"), max_time_s)
    return float(max_time_s)


def _resolve_render_snr_until(
    value: Any,
    max_time_s: float,
    *,
    saved_cell: dict[str, Any] | None,
    current_store: dict[str, Any] | None,
    animal_id: str,
    cell_idx: int,
    triggered_id: Any,
) -> float:
    if value is None:
        return float(max_time_s)
    if (
        _safe_float(value, np.nan) == 0.0
        and triggered_id in (None, "animal-dropdown", "cell-dropdown")
        and not _store_cell_matches(current_store, animal_id, cell_idx)
        and not (isinstance(saved_cell, dict) and "snr_acceptable_until_s" in saved_cell)
    ):
        return float(max_time_s)
    return _clean_snr_until(value, max_time_s)


def _store_cell_matches(store: dict[str, Any] | None, animal_id: str, cell_idx: int) -> bool:
    if not isinstance(store, dict):
        return False
    if store.get("animal_id") != str(animal_id):
        return False
    return _safe_int(store.get("cell_idx"), -1) == int(cell_idx)


def _cell_cache_key(animal_id: str, cell_idx: int) -> str:
    return f"{str(animal_id)}::{int(cell_idx)}"


def _cached_session_cell(cache: dict[str, Any] | None, animal_id: str, cell_idx: int) -> dict[str, Any] | None:
    if not isinstance(cache, dict):
        return None
    item = cache.get(_cell_cache_key(animal_id, cell_idx))
    return item if _store_cell_matches(item, animal_id, cell_idx) else None


def _update_session_cell_cache(cache: dict[str, Any] | None, store: dict[str, Any] | None) -> dict[str, Any]:
    out = dict(cache) if isinstance(cache, dict) else {}
    if not isinstance(store, dict):
        return out
    animal_id = store.get("animal_id")
    cell_idx = _safe_int(store.get("cell_idx"), -1)
    if not animal_id or cell_idx < 0:
        return out
    out[_cell_cache_key(str(animal_id), cell_idx)] = dict(store)
    return out


def _button_was_triggered(triggered_id: Any, component_id: str) -> bool:
    if triggered_id == component_id:
        return True
    try:
        triggered_props = getattr(ctx, "triggered_prop_ids", {}) or {}
    except Exception:
        triggered_props = {}
    return f"{component_id}.n_clicks" in triggered_props


def _segment_index_for_interval(start_frame: int, end_frame: int, bounds: list[tuple[int, int]]) -> int | None:
    for idx, (seg_start, seg_end) in enumerate(bounds):
        if int(start_frame) >= int(seg_start) and int(end_frame) <= int(seg_end):
            return int(idx)
    return None


def _compact_manual_exclusions(
    periods: Any,
    *,
    n_frames: int | None = None,
    frame_rate: float | None = None,
    segment_bounds_in: list[tuple[int, int]] | None = None,
) -> list[dict[str, Any]]:
    if not isinstance(periods, (list, tuple)):
        return []
    raw_periods: list[tuple[int, int]] = []
    for period in periods:
        if isinstance(period, dict):
            start_raw = period.get("start_frame", period.get("start"))
            end_raw = period.get("end_frame", period.get("end"))
        elif isinstance(period, (list, tuple)) and len(period) >= 2:
            start_raw, end_raw = period[0], period[1]
        else:
            continue
        start = _safe_int(start_raw, -1)
        end = _safe_int(end_raw, -1)
        if start < 0 or end < 0:
            continue
        if end < start:
            start, end = end, start
        if n_frames is not None:
            n = max(0, int(n_frames))
            start = max(0, min(n, start))
            end = max(0, min(n, end))
        if end > start:
            raw_periods.append((int(start), int(end)))
    if not raw_periods:
        return []

    merged: list[list[int]] = []
    for start, end in sorted(raw_periods):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)

    out: list[dict[str, Any]] = []
    bounds = segment_bounds_in or []
    for start, end in merged:
        item: dict[str, Any] = {"start_frame": int(start), "end_frame": int(end)}
        if frame_rate is not None and np.isfinite(float(frame_rate)) and float(frame_rate) > 0:
            item["start_s"] = float(start) / float(frame_rate)
            item["end_s"] = float(end) / float(frame_rate)
            item["duration_s"] = float(end - start) / float(frame_rate)
        segment_idx = _segment_index_for_interval(start, end, bounds) if bounds else None
        if segment_idx is not None:
            item["segment_index"] = int(segment_idx)
        out.append(item)
    return out


def _manual_exclusion_mask(periods: Any, n_frames: int) -> np.ndarray:
    mask = np.zeros(max(0, int(n_frames)), dtype=bool)
    for period in _compact_manual_exclusions(periods, n_frames=mask.size):
        mask[int(period["start_frame"]):int(period["end_frame"])] = True
    return mask


def _json_safe_scalar(value: Any) -> Any:
    try:
        numeric = float(value)
    except Exception:
        return value
    if not np.isfinite(numeric):
        return None
    if float(numeric).is_integer():
        return int(round(numeric))
    return float(numeric)


def _snr_cutoff_frame(snr_until_s: Any, n_frames: int, frame_rate: float) -> int:
    if not np.isfinite(float(frame_rate)) or float(frame_rate) <= 0:
        return int(max(0, n_frames))
    max_time_s = float(n_frames) / float(frame_rate)
    snr_until = _clean_snr_until(snr_until_s, max_time_s)
    return int(np.clip(np.floor(snr_until * float(frame_rate)), 0, int(n_frames)))


def _filter_frames_before_cutoff(frames: Any, cutoff_frame: int, n_frames: int) -> np.ndarray:
    values = np.sort(np.unique(np.asarray(frames, dtype=np.int64).reshape(-1)))
    return values[(values >= 0) & (values < int(n_frames)) & (values < int(cutoff_frame))]


def _filter_event_windows_before_cutoff(events: Any, cutoff_frame: int) -> dict[str, list[Any]]:
    if not isinstance(events, dict):
        return {}
    starts = _as_int_list(events.get("starts", []))
    ends = _as_int_list(events.get("ends", []))
    keep_indices = [
        idx
        for idx, (start, end) in enumerate(zip(starts, ends))
        if int(start) >= 0 and int(end) >= int(start) and int(end) < int(cutoff_frame)
    ]
    out: dict[str, list[Any]] = {}
    for key, values in events.items():
        try:
            arr = np.asarray(values, dtype=object).reshape(-1)
        except Exception:
            out[key] = []
            continue
        if arr.size == len(starts):
            out[key] = [_json_safe_scalar(arr[idx]) for idx in keep_indices if idx < arr.size]
        else:
            out[key] = [_json_safe_scalar(item) for item in arr]
    for key in ("starts", "ends", "locs", "peaks", "amplitudes", "durations_ms", "n_spikes", "min_isi_ms", "segment_indices"):
        out.setdefault(key, [])
    return out


def _saved_review_segment_bounds(
    saved_cell: dict[str, Any] | None,
    *,
    n_frames: int,
    fallback_bounds: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    raw_bounds = saved_cell.get("segment_bounds", []) if isinstance(saved_cell, dict) else []
    bounds: list[tuple[int, int]] = []
    for item in raw_bounds:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        start = max(0, min(int(n_frames), _safe_int(item[0], -1)))
        end = max(0, min(int(n_frames), _safe_int(item[1], -1)))
        if end > start:
            bounds.append((int(start), int(end)))
    return bounds if bounds else list(fallback_bounds)


def _fixed_duration_bounds(n_frames: int, frame_rate: float, duration_s: float) -> list[tuple[int, int]]:
    n_frames = max(0, int(n_frames))
    if n_frames <= 0:
        return []
    frame_rate = max(1e-12, float(frame_rate))
    row_frames = max(1, int(round(float(duration_s) * frame_rate)))
    return [
        (int(start), int(min(n_frames, start + row_frames)))
        for start in range(0, n_frames, row_frames)
    ]


def _add_exclusion_from_zoom(
    periods: Any,
    *,
    segment_idx: Any,
    x_range: list[float] | tuple[float, float] | None,
    segment_bounds_in: list[tuple[int, int]],
    frame_rate: float,
    n_frames: int,
) -> list[dict[str, Any]]:
    out = _compact_manual_exclusions(periods, n_frames=n_frames, frame_rate=frame_rate, segment_bounds_in=segment_bounds_in)
    if not segment_bounds_in:
        return out
    idx = max(0, min(len(segment_bounds_in) - 1, _safe_int(segment_idx, 0)))
    seg_start, seg_end = segment_bounds_in[idx]
    row_duration_s = max(0.0, float(seg_end - seg_start) / float(frame_rate))
    if x_range is None:
        lo, hi = 0.0, row_duration_s
    else:
        try:
            lo = float(x_range[0])
            hi = float(x_range[1])
        except Exception:
            lo, hi = 0.0, row_duration_s
    if hi < lo:
        lo, hi = hi, lo
    lo = max(0.0, min(row_duration_s, lo))
    hi = max(0.0, min(row_duration_s, hi))
    if hi <= lo:
        return out
    start_frame = int(seg_start) + int(np.floor(lo * float(frame_rate)))
    end_frame = int(seg_start) + int(np.ceil(hi * float(frame_rate)))
    end_frame = min(int(seg_end), max(start_frame + 1, end_frame))
    out.append({"start_frame": int(start_frame), "end_frame": int(end_frame)})
    return _compact_manual_exclusions(out, n_frames=n_frames, frame_rate=frame_rate, segment_bounds_in=segment_bounds_in)


def _manual_exclusion_total_s(periods: Any, frame_rate: float) -> float:
    if not np.isfinite(float(frame_rate)) or float(frame_rate) <= 0:
        return 0.0
    total_frames = sum(
        int(period["end_frame"]) - int(period["start_frame"])
        for period in _compact_manual_exclusions(periods)
    )
    return float(total_frames) / float(frame_rate)


def _manual_exclusion_label(period: dict[str, Any], idx: int) -> str:
    segment_text = (
        f"segment {int(period['segment_index'])}"
        if "segment_index" in period
        else "multi-segment"
    )
    if "start_s" in period and "end_s" in period:
        return (
            f"{idx}: {period['start_s']:.3f}-{period['end_s']:.3f}s "
            f"({segment_text}, {period['duration_s']:.3f}s)"
        )
    return f"{idx}: frames {period['start_frame']}-{period['end_frame']} ({segment_text})"


def _manual_exclusion_rows(current_store: dict[str, Any] | None) -> list[Any]:
    if not isinstance(current_store, dict):
        return [html.Div("No exclusions.", style={"fontSize": "11px", "color": "#777"})]
    frame_rate = _safe_float(current_store.get("frame_rate"), 1.0)
    bounds = [
        (int(v[0]), int(v[1]))
        for v in current_store.get("segment_bounds", [])
        if isinstance(v, (list, tuple)) and len(v) >= 2
    ]
    n_frames = max((end for _start, end in bounds), default=None)
    periods = _compact_manual_exclusions(
        current_store.get("manual_exclusion_periods", []),
        n_frames=n_frames,
        frame_rate=frame_rate,
        segment_bounds_in=bounds,
    )
    if not periods:
        return [html.Div("No exclusions.", style={"fontSize": "11px", "color": "#777"})]
    rows: list[Any] = []
    for idx, period in enumerate(periods):
        rows.append(
            html.Div(
                style={
                    "display": "grid",
                    "gridTemplateColumns": "1fr auto",
                    "gap": "6px",
                    "alignItems": "center",
                    "padding": "4px 0",
                    "borderTop": "1px solid #eee" if idx else "none",
                },
                children=[
                    html.Div(
                        _manual_exclusion_label(period, idx),
                        style={"fontSize": "11px", "lineHeight": "1.25", "color": "#333"},
                    ),
                    html.Button(
                        "x",
                        id={"type": "cancel-manual-exclusion", "index": int(idx)},
                        n_clicks=0,
                        style={
                            "height": "22px",
                            "width": "22px",
                            "padding": "0",
                            "border": "1px solid #C62828",
                            "borderRadius": "50%",
                            "backgroundColor": "#C62828",
                            "color": "white",
                            "fontSize": "14px",
                            "fontWeight": 700,
                            "lineHeight": "18px",
                            "cursor": "pointer",
                        },
                    ),
                ],
            )
        )
    return rows


def _params_equal(left: dict[str, Any] | None, right: dict[str, Any] | None) -> bool:
    if not isinstance(left, dict) or not isinstance(right, dict):
        return False
    try:
        a = normalize_params(**left)
        b = normalize_params(**right)
    except Exception:
        return False
    for key in (
        "baseline_window_s",
        "segment_duration_s",
        "highpass_hz",
        "threshold_mad",
        "spike_baseline_window_ms",
    ):
        if not np.isclose(float(a[key]), float(b[key]), rtol=0.0, atol=1e-9):
            return False
    if bool(a["spike_baseline_remove_enabled"]) != bool(b["spike_baseline_remove_enabled"]):
        return False
    return True


def _cb_params_equal(left: dict[str, Any] | None, right: dict[str, Any] | None) -> bool:
    if not isinstance(left, dict) or not isinstance(right, dict):
        return False
    try:
        a = normalize_cb_params(**left)
        b = normalize_cb_params(**right)
    except Exception:
        return False
    for key in (
        "cb_baseline_window_s",
        "vm_median_window_ms",
        "vm_crossing_threshold",
        "cb_amp_threshold",
        "cb_duration_threshold_ms",
        "cb_min_spikes",
        "cb_isi_threshold_ms",
        "cb_require_min_isi",
        "spike_height_min_isolated_spikes",
        "cb_onset_threshold",
        "cb_offset_threshold",
        "cb_max_onset_lead_ms",
    ):
        if not np.isclose(float(a[key]), float(b[key]), rtol=0.0, atol=1e-9):
            return False
    if bool(a["remove_spikes_for_vm"]) != bool(b["remove_spikes_for_vm"]):
        return False
    if bool(a["cb_require_min_isi"]) != bool(b["cb_require_min_isi"]):
        return False
    if bool(a["include_first_burst_spike_for_spike_height"]) != bool(b["include_first_burst_spike_for_spike_height"]):
        return False
    if bool(a["refine_cb_onset"]) != bool(b["refine_cb_onset"]):
        return False
    return True


def _second_round_params_equal(left: dict[str, Any] | None, right: dict[str, Any] | None) -> bool:
    if not isinstance(left, dict) or not isinstance(right, dict):
        return False
    try:
        a = normalize_second_round_params(**left)
        b = normalize_second_round_params(**right)
    except Exception:
        return False
    for key in ("ss_threshold_mad", "cs_threshold_mad", "simple_spike_min_height_fraction"):
        if not np.isclose(float(a[key]), float(b[key]), rtol=0.0, atol=1e-9):
            return False
    if bool(a["refine_simple_spikes_by_height"]) != bool(b["refine_simple_spikes_by_height"]):
        return False
    return True


def _second_round_cb_params_equal(left: dict[str, Any] | None, right: dict[str, Any] | None) -> bool:
    if not isinstance(left, dict) or not isinstance(right, dict):
        return False
    try:
        a = normalize_second_round_cb_params(**left)
        b = normalize_second_round_cb_params(**right)
    except Exception:
        return False
    if not np.isclose(float(a["cb_amp_threshold"]), float(b["cb_amp_threshold"]), rtol=0.0, atol=1e-9):
        return False
    return int(round(a["cb_min_spikes"])) == int(round(b["cb_min_spikes"]))


def _as_int_list(values: Any) -> list[int]:
    try:
        return [int(v) for v in np.asarray(values, dtype=np.int64).reshape(-1)]
    except Exception:
        return []


def _first_round_source_signature(
    *,
    params: dict[str, Any],
    cb_params: dict[str, Any],
    thresholds: np.ndarray,
    complex_bursts: dict[str, Any] | None,
    failed_min_spikes_windows: dict[str, Any] | None = None,
    manual_exclusion_periods: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    bursts = complex_bursts if isinstance(complex_bursts, dict) else {}
    failed = failed_min_spikes_windows if isinstance(failed_min_spikes_windows, dict) else {}
    threshold_list = [
        float(v) if np.isfinite(v) else None
        for v in np.asarray(thresholds, dtype=float).reshape(-1)
    ]
    return {
        "params": normalize_params(**params),
        "cb_params": normalize_cb_params(**cb_params),
        "thresholds": threshold_list,
        "cb_starts": _as_int_list(bursts.get("starts", [])),
        "cb_ends": _as_int_list(bursts.get("ends", [])),
        "failed_cb_starts": _as_int_list(failed.get("starts", [])),
        "failed_cb_ends": _as_int_list(failed.get("ends", [])),
        "manual_exclusion_periods": [
            {
                "start_frame": int(period["start_frame"]),
                "end_frame": int(period["end_frame"]),
            }
            for period in _compact_manual_exclusions(manual_exclusion_periods)
        ],
    }


def _second_round_cb_source_signature(second_round: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(second_round, dict):
        return None
    return {
        "second_round_source": second_round.get("source_signature"),
        "second_round_params": second_round.get("second_round_params"),
        "detection_order": second_round.get("detection_order"),
        "spikes": _as_int_list(second_round.get("cb_input_spikes", second_round.get("spikes", []))),
    }


def _second_round_cb_input_spikes(second_round: dict[str, Any] | None) -> np.ndarray:
    if not isinstance(second_round, dict):
        return np.array([], dtype=np.int64)
    values = second_round.get("cb_input_spikes", second_round.get("spikes", []))
    return np.asarray(_as_int_list(values), dtype=np.int64)


def _second_round_cb_source_matches(
    second_round_cb: dict[str, Any] | None,
    source_signature: dict[str, Any] | None,
    second_round_cb_params: dict[str, float],
    *,
    check_params: bool = True,
) -> bool:
    if not isinstance(second_round_cb, dict) or source_signature is None:
        return False
    if check_params and not _second_round_cb_params_equal(second_round_cb.get("second_round_cb_params"), second_round_cb_params):
        return False
    return second_round_cb.get("source_signature") == source_signature


def _compact_complex_bursts(complex_bursts: dict[str, Any]) -> dict[str, list[Any]]:
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
    int_keys = {"starts", "ends", "locs", "n_spikes", "segment_indices"}
    out: dict[str, list[Any]] = {}
    for key, value in complex_bursts.items():
        try:
            values = np.asarray(value, dtype=object).reshape(-1)
        except Exception:
            out[key] = []
            continue
        cleaned: list[Any] = []
        for item in values:
            try:
                numeric = float(item)
            except Exception:
                if key not in int_keys:
                    cleaned.append(None)
                continue
            if not np.isfinite(numeric):
                if key not in int_keys:
                    cleaned.append(None)
                continue
            cleaned.append(int(round(numeric)) if key in int_keys else float(numeric))
        out[key] = cleaned
    return out


def _combine_cb_windows(*window_dicts: dict[str, Any] | None) -> dict[str, list[int]]:
    """Return merged CB-style windows using only starts/ends."""
    windows: list[tuple[int, int]] = []
    for window_dict in window_dicts:
        if not isinstance(window_dict, dict):
            continue
        starts = _as_int_list(window_dict.get("starts", []))
        ends = _as_int_list(window_dict.get("ends", []))
        for start, end in zip(starts, ends):
            s = int(min(start, end))
            e = int(max(start, end))
            if e >= s:
                windows.append((s, e))
    ordered: list[tuple[int, int]] = []
    for start, end in sorted(windows, key=lambda item: (item[0], item[1])):
        if not ordered or int(start) > int(ordered[-1][1]):
            ordered.append((int(start), int(end)))
        else:
            prev_start, prev_end = ordered[-1]
            ordered[-1] = (int(prev_start), max(int(prev_end), int(end)))
    return {
        "starts": [start for start, _end in ordered],
        "ends": [end for _start, end in ordered],
    }


def _window_mask_from_store(windows: dict[str, Any] | None, n_frames: int) -> np.ndarray:
    mask = np.zeros(int(n_frames), dtype=bool)
    if not isinstance(windows, dict):
        return mask
    starts = _as_int_list(windows.get("starts", []))
    ends = _as_int_list(windows.get("ends", []))
    for start, end in zip(starts, ends):
        s = max(0, int(min(start, end)))
        e = min(int(n_frames), int(max(start, end)) + 1)
        if e > s:
            mask[s:e] = True
    return mask


def _min_isi_ms_from_spikes(spikes: np.ndarray, frame_rate: float) -> float:
    spikes = np.sort(np.asarray(spikes, dtype=np.int64).reshape(-1))
    if spikes.size < 2:
        return float("inf")
    isi = np.diff(spikes).astype(float) * 1000.0 / float(frame_rate)
    return float(np.nanmin(isi)) if isi.size else float("inf")


def _cb_window_records(windows: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(windows, dict):
        return []
    starts = _as_int_list(windows.get("starts", []))
    ends = _as_int_list(windows.get("ends", []))
    n = min(len(starts), len(ends))
    records: list[dict[str, Any]] = []
    for idx in range(n):
        record: dict[str, Any] = {
            "starts": int(starts[idx]),
            "ends": int(ends[idx]),
        }
        for key in ("locs", "n_spikes", "segment_indices"):
            values = _as_int_list(windows.get(key, []))
            if idx < len(values):
                record[key] = int(values[idx])
            elif key == "locs":
                record[key] = int(starts[idx])
            elif key == "segment_indices":
                record[key] = -1
            else:
                record[key] = 0
        for key in ("peaks", "amplitudes", "durations_ms", "min_isi_ms"):
            try:
                values = np.asarray(windows.get(key, []), dtype=float).reshape(-1)
                value = float(values[idx]) if idx < values.size and np.isfinite(values[idx]) else float("nan")
            except Exception:
                value = float("nan")
            record[key] = value
        records.append(record)
    return records


def _records_to_cb_windows(records: list[dict[str, Any]]) -> dict[str, list[Any]]:
    ordered = sorted(records, key=lambda item: (int(item["starts"]), int(item["ends"])))
    int_keys = ("starts", "ends", "locs", "n_spikes", "segment_indices")
    float_keys = ("peaks", "amplitudes", "durations_ms", "min_isi_ms")
    out: dict[str, list[Any]] = {}
    for key in int_keys:
        out[key] = [int(record.get(key, 0)) for record in ordered]
    for key in float_keys:
        values: list[Any] = []
        for record in ordered:
            value = _safe_float(record.get(key), float("nan"))
            values.append(float(value) if np.isfinite(value) else None)
        out[key] = values
    return out


def _finalize_cb_result_from_cs(
    cb_result: dict[str, Any],
    candidate_complex_spikes: np.ndarray,
    frame_rate: float,
    cb_params: dict[str, Any],
    n_segments: int,
) -> dict[str, Any]:
    params = normalize_cb_params(**cb_params)
    min_spikes_required = int(round(params["cb_min_spikes"]))
    require_min_isi = bool(params["cb_require_min_isi"])
    isi_threshold_ms = float(params["cb_isi_threshold_ms"])
    candidate_spikes = np.sort(np.unique(np.asarray(candidate_complex_spikes, dtype=np.int64).reshape(-1)))

    records_by_window: dict[tuple[int, int], dict[str, Any]] = {}
    for record in _cb_window_records(cb_result.get("complex_bursts")) + _cb_window_records(
        cb_result.get("failed_min_spikes_after_amp_duration_windows")
    ):
        start = int(min(record["starts"], record["ends"]))
        end = int(max(record["starts"], record["ends"]))
        records_by_window.setdefault((start, end), {**record, "starts": start, "ends": end})

    finalized: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    failed_by_segment = np.zeros(int(n_segments), dtype=np.int64)
    for (start, end), record in sorted(records_by_window.items()):
        in_window = candidate_spikes[(candidate_spikes >= start) & (candidate_spikes <= end)]
        n_spikes = int(in_window.size)
        min_isi = _min_isi_ms_from_spikes(in_window, frame_rate)
        updated = dict(record)
        updated["n_spikes"] = n_spikes
        updated["min_isi_ms"] = min_isi
        if n_spikes > 0:
            updated["locs"] = int(in_window[0])
        is_true_cb = n_spikes >= min_spikes_required and (not require_min_isi or min_isi <= isi_threshold_ms)
        if is_true_cb:
            finalized.append(updated)
        else:
            failed.append(updated)
            seg_idx = _safe_int(updated.get("segment_indices"), -1)
            if 0 <= seg_idx < failed_by_segment.size:
                failed_by_segment[seg_idx] += 1

    out = dict(cb_result)
    out["complex_bursts"] = _records_to_cb_windows(finalized)
    out["failed_min_spikes_after_amp_duration"] = int(len(failed))
    out["failed_min_spikes_after_amp_duration_by_segment"] = failed_by_segment
    out["failed_min_spikes_after_amp_duration_windows"] = _records_to_cb_windows(failed)
    return out


def _apply_final_second_round_classification(
    second_round_result: dict[str, Any],
    second_round_cb_calc: dict[str, Any],
    candidate_complex_spikes: np.ndarray,
    second_round_cb_input_spikes: np.ndarray,
    n_frames: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rewrite final SS/CS lists from the finalized CB windows."""
    final_true_cb_mask = _window_mask_from_store(
        second_round_cb_calc.get("complex_bursts"),
        int(n_frames),
    )
    candidate_complex_spikes = np.sort(
        np.unique(np.asarray(candidate_complex_spikes, dtype=np.int64).reshape(-1))
    )
    candidate_complex_spikes = candidate_complex_spikes[
        (candidate_complex_spikes >= 0) & (candidate_complex_spikes < int(n_frames))
    ]
    second_round_cb_input_spikes = np.sort(
        np.unique(np.asarray(second_round_cb_input_spikes, dtype=np.int64).reshape(-1))
    )
    second_round_cb_input_spikes = second_round_cb_input_spikes[
        (second_round_cb_input_spikes >= 0) & (second_round_cb_input_spikes < int(n_frames))
    ]
    final_complex_spikes = candidate_complex_spikes[
        final_true_cb_mask[candidate_complex_spikes]
    ]
    final_simple_spikes = second_round_cb_input_spikes[
        ~final_true_cb_mask[second_round_cb_input_spikes]
    ]
    final_spikes = np.sort(
        np.unique(np.concatenate([final_simple_spikes, final_complex_spikes]).astype(np.int64))
    )
    second_round_result["simple_spikes"] = [int(v) for v in final_simple_spikes]
    second_round_result["complex_spikes"] = [int(v) for v in final_complex_spikes]
    second_round_result["spikes"] = [int(v) for v in final_spikes]
    return final_simple_spikes, final_complex_spikes, final_spikes


def _compact_cb_detection_result(
    cb_result: dict[str, Any],
    *,
    source_signature: dict[str, Any] | None = None,
    second_round_cb_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out = {
        "cb_params": cb_result["cb_params"],
        "simple_spikes": [int(v) for v in np.asarray(cb_result["simple_spikes"], dtype=np.int64).reshape(-1)],
        "complex_spikes": [int(v) for v in np.asarray(cb_result["complex_spikes"], dtype=np.int64).reshape(-1)],
        "complex_bursts": _compact_complex_bursts(cb_result["complex_bursts"]),
        "failed_min_spikes_after_amp_duration": int(cb_result.get("failed_min_spikes_after_amp_duration", 0)),
        "failed_min_spikes_after_amp_duration_by_segment": [
            int(v)
            for v in np.asarray(
                cb_result.get("failed_min_spikes_after_amp_duration_by_segment", []),
                dtype=np.int64,
            ).reshape(-1)
        ],
        "failed_min_spikes_after_amp_duration_windows": _compact_complex_bursts(
            cb_result.get("failed_min_spikes_after_amp_duration_windows", {})
        ),
        "segment_spike_heights": [
            float(v) if np.isfinite(v) else None
            for v in np.asarray(cb_result["segment_spike_heights"], dtype=float).reshape(-1)
        ],
        "segment_spike_height_counts": [
            int(v)
            for v in np.asarray(cb_result["segment_spike_height_counts"], dtype=np.int64).reshape(-1)
        ],
    }
    if source_signature is not None:
        out["source_signature"] = source_signature
    if second_round_cb_params is not None:
        out["second_round_cb_params"] = normalize_second_round_cb_params(**second_round_cb_params)
    return out


def _compact_plateau_detection_result(plateau_result: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(plateau_result, dict):
        return None
    return {
        "plateau_params": normalize_plateau_params(**plateau_result.get("plateau_params", {})),
        "plateaus": _compact_complex_bursts(plateau_result.get("plateaus", {})),
        "segment_spike_heights": [
            float(v) if np.isfinite(v) else None
            for v in np.asarray(plateau_result.get("segment_spike_heights", []), dtype=float).reshape(-1)
        ],
        "segment_spike_height_counts": [
            int(v)
            for v in np.asarray(
                plateau_result.get("segment_spike_height_counts", []),
                dtype=np.int64,
            ).reshape(-1)
        ],
    }


def _plateau_params_equal(left: dict[str, Any] | None, right: dict[str, Any] | None) -> bool:
    if not isinstance(left, dict) or not isinstance(right, dict):
        return False
    try:
        a = normalize_plateau_params(**left)
        b = normalize_plateau_params(**right)
    except Exception:
        return False
    for key in (
        "plateau_baseline_window_s",
        "plateau_vm_median_window_ms",
        "plateau_vm_crossing_threshold",
        "plateau_onset_threshold",
        "plateau_offset_threshold",
        "plateau_amp_threshold",
        "plateau_duration_threshold_ms",
        "plateau_min_spikes",
        "plateau_peak_fraction_threshold",
        "plateau_peak_fraction_duration_ms",
    ):
        if not np.isclose(float(a[key]), float(b[key]), rtol=0.0, atol=1e-9):
            return False
    return True


def _second_round_source_matches(
    second_round: dict[str, Any] | None,
    source_signature: dict[str, Any],
    second_round_params: dict[str, float],
    *,
    check_params: bool = True,
) -> bool:
    if not isinstance(second_round, dict):
        return False
    if second_round.get("detection_order") != SECOND_ROUND_DETECTION_ORDER:
        return False
    if check_params and not _second_round_params_equal(second_round.get("second_round_params"), second_round_params):
        return False
    return second_round.get("source_signature") == source_signature


def _saved_second_round_result(
    saved_cell: dict[str, Any] | None,
    *,
    source_signature: dict[str, Any],
    second_round_params: dict[str, float],
    n_segments: int,
) -> dict[str, Any] | None:
    if not isinstance(saved_cell, dict):
        return None
    saved_second_round = saved_cell.get("second_round")
    if _second_round_source_matches(saved_second_round, source_signature, second_round_params, check_params=True):
        return saved_second_round
    return None


def _as_threshold_list(values: Any) -> list[float] | None:
    if values is None:
        return None
    try:
        arr = np.asarray(values, dtype=float).reshape(-1)
    except Exception:
        return None
    return [float(v) if np.isfinite(v) else float("nan") for v in arr]


def _json_safe_saved_cell(saved_cell: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(saved_cell, dict):
        return None
    out = dict(saved_cell)
    thresholds = _as_threshold_list(out.get("thresholds"))
    if thresholds is not None:
        out["thresholds"] = [float(v) if np.isfinite(v) else None for v in thresholds]
    if "manual_exclusion_periods" in out:
        out["manual_exclusion_periods"] = _compact_manual_exclusions(out.get("manual_exclusion_periods", []))
    if "spikes" in out:
        out["spikes"] = [int(v) for v in np.asarray(out["spikes"], dtype=np.int64).reshape(-1)]
    for spike_key in ("simple_spikes", "complex_spikes"):
        if spike_key in out:
            out[spike_key] = [int(v) for v in np.asarray(out[spike_key], dtype=np.int64).reshape(-1)]
    for burst_key in ("complex_bursts", "first_round_complex_bursts"):
        if burst_key in out and isinstance(out[burst_key], dict):
            out[burst_key] = _compact_complex_bursts(out[burst_key])
    if "segment_spike_heights" in out:
        out["segment_spike_heights"] = [
            float(v) if np.isfinite(v) else None
            for v in np.asarray(out["segment_spike_heights"], dtype=float).reshape(-1)
        ]
    if "segment_spike_height_counts" in out:
        out["segment_spike_height_counts"] = [
            int(v)
            for v in np.asarray(out["segment_spike_height_counts"], dtype=np.int64).reshape(-1)
        ]
    for key in ("segment_snr", "segment_snr_noise"):
        if key in out:
            out[key] = [
                float(v) if np.isfinite(v) else None
                for v in np.asarray(out[key], dtype=float).reshape(-1)
            ]
    if "segment_snr_baseline_counts" in out:
        out["segment_snr_baseline_counts"] = [
            int(v)
            for v in np.asarray(out["segment_snr_baseline_counts"], dtype=np.int64).reshape(-1)
        ]
    if "overall_snr" in out:
        overall_snr = _safe_float(out.get("overall_snr"), float("nan"))
        out["overall_snr"] = float(overall_snr) if np.isfinite(overall_snr) else None
    if "snr_spike_mask_ms" in out:
        out["snr_spike_mask_ms"] = _safe_float(out.get("snr_spike_mask_ms"), 3.0)
    for key in ("failed_min_spikes_after_amp_duration", "first_round_failed_min_spikes_after_amp_duration"):
        if key in out:
            out[key] = int(out.get(key) or 0)
    for key in (
        "failed_min_spikes_after_amp_duration_by_segment",
        "first_round_failed_min_spikes_after_amp_duration_by_segment",
    ):
        if key in out:
            out[key] = [int(v) for v in np.asarray(out[key], dtype=np.int64).reshape(-1)]
    for key in (
        "failed_min_spikes_after_amp_duration_windows",
        "first_round_failed_min_spikes_after_amp_duration_windows",
    ):
        if key in out and isinstance(out[key], dict):
            out[key] = _compact_complex_bursts(out[key])
    if "segment_bounds" in out:
        out["segment_bounds"] = [[int(s), int(e)] for s, e in out["segment_bounds"]]
    if "params" in out and isinstance(out["params"], dict):
        out["params"] = normalize_params(**out["params"])
    if "cb_params" in out and isinstance(out["cb_params"], dict):
        out["cb_params"] = normalize_cb_params(**out["cb_params"])
    if "first_round_cb_params" in out and isinstance(out["first_round_cb_params"], dict):
        out["first_round_cb_params"] = normalize_cb_params(**out["first_round_cb_params"])
    if "second_round_params" in out and isinstance(out["second_round_params"], dict):
        out["second_round_params"] = normalize_second_round_params(**out["second_round_params"])
    if "second_round" in out and isinstance(out["second_round"], dict):
        second_round = dict(out["second_round"])
        if "second_round_params" in second_round and isinstance(second_round["second_round_params"], dict):
            second_round["second_round_params"] = normalize_second_round_params(**second_round["second_round_params"])
        for spike_key in ("spikes", "cb_input_spikes", "simple_spikes", "complex_spikes", "removed_simple_spikes"):
            if spike_key in second_round:
                second_round[spike_key] = [int(v) for v in np.asarray(second_round[spike_key], dtype=np.int64).reshape(-1)]
        for threshold_key in ("ss_thresholds", "cs_thresholds"):
            if threshold_key in second_round:
                second_round[threshold_key] = [
                    float(v) if np.isfinite(v) else None
                    for v in np.asarray(second_round[threshold_key], dtype=float).reshape(-1)
                ]
        out["second_round"] = second_round
    if "second_round_cb_params" in out and isinstance(out["second_round_cb_params"], dict):
        out["second_round_cb_params"] = normalize_second_round_cb_params(**out["second_round_cb_params"])
    if "second_round_cb" in out and isinstance(out["second_round_cb"], dict):
        second_cb = dict(out["second_round_cb"])
        if "second_round_cb_params" in second_cb and isinstance(second_cb["second_round_cb_params"], dict):
            second_cb["second_round_cb_params"] = normalize_second_round_cb_params(**second_cb["second_round_cb_params"])
        if "complex_bursts" in second_cb and isinstance(second_cb["complex_bursts"], dict):
            second_cb["complex_bursts"] = _compact_complex_bursts(second_cb["complex_bursts"])
        if "failed_min_spikes_after_amp_duration" in second_cb:
            second_cb["failed_min_spikes_after_amp_duration"] = int(
                second_cb.get("failed_min_spikes_after_amp_duration") or 0
            )
        if "failed_min_spikes_after_amp_duration_by_segment" in second_cb:
            second_cb["failed_min_spikes_after_amp_duration_by_segment"] = [
                int(v)
                for v in np.asarray(
                    second_cb["failed_min_spikes_after_amp_duration_by_segment"],
                    dtype=np.int64,
                ).reshape(-1)
            ]
        if (
            "failed_min_spikes_after_amp_duration_windows" in second_cb
            and isinstance(second_cb["failed_min_spikes_after_amp_duration_windows"], dict)
        ):
            second_cb["failed_min_spikes_after_amp_duration_windows"] = _compact_complex_bursts(
                second_cb["failed_min_spikes_after_amp_duration_windows"]
            )
        for spike_key in ("simple_spikes", "complex_spikes"):
            if spike_key in second_cb:
                second_cb[spike_key] = [int(v) for v in np.asarray(second_cb[spike_key], dtype=np.int64).reshape(-1)]
        out["second_round_cb"] = second_cb
    if "plateau_params" in out and isinstance(out["plateau_params"], dict):
        out["plateau_params"] = normalize_plateau_params(**out["plateau_params"])
    if "plateaus" in out and isinstance(out["plateaus"], dict):
        out["plateaus"] = _compact_complex_bursts(out["plateaus"])
    if "plateau_result" in out and isinstance(out["plateau_result"], dict):
        out["plateau_result"] = _compact_plateau_detection_result(out["plateau_result"])
    for key in ("second_round_ss_thresholds", "second_round_cs_thresholds"):
        if key in out:
            out[key] = [
                float(v) if np.isfinite(v) else None
                for v in np.asarray(out[key], dtype=float).reshape(-1)
            ]
    return out


def _store_identity_matches(store: dict[str, Any] | None, animal_id: str, cell_idx: int, params: dict[str, float]) -> bool:
    if not isinstance(store, dict):
        return False
    if store.get("animal_id") != str(animal_id):
        return False
    if _safe_int(store.get("cell_idx"), -1) != int(cell_idx):
        return False
    return _params_equal(store.get("params"), params)


def _store_ready_for_save(
    store: dict[str, Any] | None,
    *,
    animal_id: str,
    cell_idx: int,
    params: dict[str, Any],
    cb_params: dict[str, Any],
    second_round_params: dict[str, Any] | None = None,
    second_round_cb_params: dict[str, Any] | None = None,
    plateau_params: dict[str, Any] | None = None,
    snr_acceptable_until_s: Any,
    max_time_s: float,
    run_second_round_clicks: Any = None,
) -> tuple[bool, str]:
    if not _store_identity_matches(store, animal_id, cell_idx, params):
        return False, "Current detection state is stale; render the cell again before saving."
    if not isinstance(store, dict):
        return False, "Current detection state is stale; render the cell again before saving."
    current_second_round_clicks = _safe_int(run_second_round_clicks, 0)
    store_second_round_clicks = _safe_int(store.get("run_second_round_n_clicks_seen"), 0)
    if current_second_round_clicks > store_second_round_clicks:
        return False, "2nd-round spike detection is still rendering; wait for the plot/status to update before saving."
    store_cb_params = store.get("first_round_cb_params", store.get("cb_params"))
    if not _cb_params_equal(store_cb_params, cb_params):
        return False, "Current CB detection state is stale; render the cell again before saving."
    if isinstance(store.get("second_round"), dict):
        if second_round_params is not None and not _second_round_params_equal(
            store.get("second_round_params"),
            second_round_params,
        ):
            return False, "Typed 2nd-round spike parameters are pending Run; run 2nd round again before saving."
        if isinstance(store.get("second_round_cb"), dict) and second_round_cb_params is not None:
            if not _second_round_cb_params_equal(
                store.get("second_round_cb_params"),
                second_round_cb_params,
            ):
                return False, "Typed 2nd-round CB parameters are pending Run; run 2nd round again before saving."
    if plateau_params is not None and not _plateau_params_equal(store.get("plateau_params"), plateau_params):
        return False, "Current plateau detection state is stale; render the cell again before saving."
    current_snr = _clean_snr_until(snr_acceptable_until_s, max_time_s)
    store_snr = _clean_snr_until(store.get("snr_acceptable_until_s", max_time_s), max_time_s)
    if not np.isclose(current_snr, store_snr, rtol=0.0, atol=1e-6):
        return False, "Current SNR cutoff is stale; render the cell again before saving."
    return True, ""


def _extract_shape_y_updates(relayout_data: dict[str, Any] | None) -> dict[int, list[float]]:
    updates: dict[int, list[float]] = {}
    if not isinstance(relayout_data, dict):
        return updates

    def _path_y_values(path_value: Any) -> list[float]:
        if not isinstance(path_value, str):
            return []
        nums = [float(v) for v in re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", path_value)]
        if len(nums) < 4:
            return []
        return [nums[idx] for idx in range(1, len(nums), 2) if np.isfinite(nums[idx])]

    for key, value in relayout_data.items():
        if not isinstance(key, str) or not key.startswith("shapes["):
            continue
        try:
            shape_idx = int(key.split("[", 1)[1].split("]", 1)[0])
        except Exception:
            continue
        if key.endswith(".y0") or key.endswith(".y1"):
            try:
                y_val = float(value)
            except Exception:
                continue
            updates.setdefault(shape_idx, []).append(y_val)
        elif key.endswith(".path"):
            y_vals = _path_y_values(value)
            if y_vals:
                updates.setdefault(shape_idx, []).extend(y_vals)
        elif isinstance(value, dict):
            for y_key in ("y0", "y1"):
                if y_key not in value:
                    continue
                try:
                    y_val = float(value[y_key])
                except Exception:
                    continue
                updates.setdefault(shape_idx, []).append(y_val)
            y_vals = _path_y_values(value.get("path"))
            if y_vals:
                updates.setdefault(shape_idx, []).extend(y_vals)
    return updates


def _apply_shape_drag(
    relayout_data: dict[str, Any] | None,
    thresholds: np.ndarray,
    current_store: dict[str, Any] | None,
) -> np.ndarray:
    updates = _extract_shape_y_updates(relayout_data)
    if not updates or not isinstance(current_store, dict):
        return thresholds

    row_offsets = np.asarray(current_store.get("row_offsets", []), dtype=float)
    shape_to_segment = current_store.get("shape_to_segment", [])
    center = _safe_float(current_store.get("threshold_center"), 0.0)
    scale = _safe_float(current_store.get("threshold_scale"), 1.0)
    trace_scale = _safe_float(current_store.get("trace_scale"), 1.0)
    if abs(trace_scale) <= 1e-12:
        trace_scale = 1.0
    out = np.asarray(thresholds, dtype=float).copy()
    for shape_idx, y_values in updates.items():
        if shape_idx < len(shape_to_segment):
            seg_idx = _safe_int(shape_to_segment[shape_idx], shape_idx)
        else:
            seg_idx = shape_idx
        if seg_idx < 0 or seg_idx >= out.size or seg_idx >= row_offsets.size:
            continue
        finite_y = [float(v) for v in y_values if np.isfinite(float(v))]
        if not finite_y:
            continue
        scaled_y = float(np.mean(finite_y) - row_offsets[seg_idx])
        out[seg_idx] = float((scaled_y / trace_scale) * scale + center)
    return out


def _relayout_has_x_reset(relayout_data: dict[str, Any] | None) -> bool:
    if not isinstance(relayout_data, dict):
        return False
    return bool(relayout_data.get("xaxis.autorange")) or "xaxis.range" in relayout_data and relayout_data.get("xaxis.range") is None


def _extract_x_range(relayout_data: dict[str, Any] | None) -> tuple[bool, list[float] | None]:
    if not isinstance(relayout_data, dict):
        return False, None
    if _relayout_has_x_reset(relayout_data):
        return True, None
    if "xaxis.range" in relayout_data:
        value = relayout_data.get("xaxis.range")
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            lo = _safe_float(value[0], np.nan)
            hi = _safe_float(value[1], np.nan)
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                return True, [lo, hi]
    if "xaxis.range[0]" in relayout_data and "xaxis.range[1]" in relayout_data:
        lo = _safe_float(relayout_data.get("xaxis.range[0]"), np.nan)
        hi = _safe_float(relayout_data.get("xaxis.range[1]"), np.nan)
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            return True, [lo, hi]
    return False, None


def _select_x_range(
    *,
    triggered_id: str | None,
    hp_relayout_data: dict[str, Any] | None,
    trace_relayout_data: dict[str, Any] | None,
    current_store: dict[str, Any] | None,
    animal_id: str,
    cell_idx: int,
    params: dict[str, float],
) -> list[float] | None:
    if triggered_id in {"hp-graph", "trace-graph"}:
        relayout_data = hp_relayout_data if triggered_id == "hp-graph" else trace_relayout_data
        has_x_event, x_range = _extract_x_range(relayout_data)
        if has_x_event:
            return x_range
    if _store_identity_matches(current_store, animal_id, cell_idx, params):
        stored = current_store.get("x_range")
        if isinstance(stored, (list, tuple)) and len(stored) >= 2:
            lo = _safe_float(stored[0], np.nan)
            hi = _safe_float(stored[1], np.nan)
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                return [lo, hi]
    return None


def _cell_options(bundle, saved_payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    cells = (saved_payload or {}).get("cells", {})
    saved = set()
    if isinstance(cells, dict):
        for key in cells:
            try:
                saved.add(int(key))
            except Exception:
                pass
    options = []
    for idx in range(bundle.n_cells):
        suffix = " [saved]" if idx in saved else ""
        options.append({"label": f"Cell {idx + 1}{suffix}", "value": idx})
    return options


def _cell_label(cell_idx: int | str) -> str:
    return f"Cell {_safe_int(cell_idx, -1) + 1}"


def _cell_text(cell_idx: int | str) -> str:
    return f"cell {_safe_int(cell_idx, -1) + 1}"


def _saved_status_text(saved_cell: dict[str, Any] | None, cell_idx: int, *, session_cached: bool = False) -> str:
    cell_label = _cell_label(cell_idx)
    if not saved_cell:
        return f"{cell_label} has no saved manual thresholds."
    spikes = saved_cell.get("spikes", [])
    complex_spikes = saved_cell.get("complex_spikes", [])
    complex_bursts = saved_cell.get("complex_bursts", {})
    cb_count = len(complex_bursts.get("starts", [])) if isinstance(complex_bursts, dict) else 0
    plateaus = saved_cell.get("plateaus", {})
    plateau_count = len(plateaus.get("starts", [])) if isinstance(plateaus, dict) else 0
    thresholds = saved_cell.get("thresholds", [])
    exclusions = _compact_manual_exclusions(saved_cell.get("manual_exclusion_periods", []))
    source = "session state" if session_cached else "saved"
    return (
        f"{cell_label} {source}: {len(spikes)} spikes, {len(complex_spikes)} complex spikes, "
        f"{cb_count} CBs, {plateau_count} plateaus, {len(thresholds)} segment thresholds, "
        f"{len(exclusions)} exclusions."
    )


def create_app(*, data_root: str | Path | None = None, gui_defaults: dict[str, Any] | None = None) -> Dash:
    root = Path(data_root or default_data_root()).expanduser().resolve()
    defaults = _normalize_gui_defaults(gui_defaults)
    default_params = defaults["params"]
    default_cb_params = defaults["cb_params"]
    default_second_round_params = defaults["second_round_params"]
    default_second_round_cb_params = defaults["second_round_cb_params"]
    default_plateau_params = defaults["plateau_params"]
    default_segment_height_px = int(defaults["segment_height_px"])
    default_right_panel_mode = str(defaults["right_panel_mode"])
    default_use_saved_cell_parameters = bool(defaults["use_saved_cell_parameters"])
    app = Dash(__name__)
    app.title = "Manual Spike Detection"

    animals = discover_animals(root)
    default_animal = animals[0] if animals else None

    @lru_cache(maxsize=8)
    def _load_cached(animal_id: str):
        return load_animal(root, animal_id)

    @lru_cache(maxsize=24)
    def _prepare_cached(
        animal_id: str,
        cell_idx: int,
        baseline_window_s: float,
        segment_duration_s: float,
        highpass_hz: float,
        threshold_mad: float,
        spike_baseline_remove_enabled: bool,
        spike_baseline_window_ms: float,
    ):
        bundle = _load_cached(str(animal_id))
        params = normalize_params(
            baseline_window_s=baseline_window_s,
            segment_duration_s=segment_duration_s,
            highpass_hz=highpass_hz,
            threshold_mad=threshold_mad,
            spike_baseline_remove_enabled=spike_baseline_remove_enabled,
            spike_baseline_window_ms=spike_baseline_window_ms,
        )
        return prepare_detection_inputs(bundle.traces[int(cell_idx), :], bundle.frame_rate, params)

    app.layout = html.Div(
        style={
            "display": "grid",
            "gridTemplateColumns": "auto minmax(0, 1fr)",
            "height": "100vh",
            "fontFamily": "Arial, sans-serif",
            "fontSize": "12px",
            "gap": "8px",
            "padding": "8px",
            "boxSizing": "border-box",
            "backgroundColor": "#fafafa",
            "overflow": "hidden",
        },
        children=[
            html.Div(
                style={
                    "width": "330px",
                    "minWidth": "280px",
                    "maxWidth": "70vw",
                    "resize": "horizontal",
                    "overflow": "auto",
                    "boxSizing": "border-box",
                    "backgroundColor": "#f7f7f7",
                    "border": "1px solid #ddd",
                    "borderRadius": "6px",
                    "padding": "10px",
                    "height": "100%",
                },
                children=[
                    dcc.Store(id="saved-cell-store", data=None),
                    dcc.Store(id="detection-store", data=None),
                    dcc.Store(id="session-cell-cache-store", data={}),
                    dcc.Store(id="save-refresh-store", data=0),
                    html.Div(
                        style=CARD_STYLE,
                        children=[
                            html.Div("Dataset", style=SECTION_HEADER),
                            html.Label("Animal", style=LABEL_STYLE),
                            dcc.Dropdown(
                                id="animal-dropdown",
                                options=_animal_options(animals),
                                value=default_animal,
                                clearable=False,
                                style={"fontSize": "12px"},
                            ),
                            html.Div(id="animal-info", style=STATUS_STYLE),
                            html.Label("Cell", style={**LABEL_STYLE, "marginTop": "8px"}),
                            dcc.Dropdown(
                                id="cell-dropdown",
                                options=[],
                                value=None,
                                clearable=False,
                                style={"fontSize": "12px"},
                            ),
                            html.Div(id="cell-info", style=STATUS_STYLE),
                        ],
                    ),
                    html.Div(
                        style=CARD_STYLE,
                        children=[
                            html.Div("Actions", style=SECTION_HEADER),
                            _number_row("Acceptable SNR until (s)", "snr-acceptable-until-s", None, step="any", min_value=0),
                            html.Button("Reset thresholds", id="reset-thresholds", n_clicks=0, style=BUTTON_STYLE),
                            _number_row("Exclusion segment", "manual-exclusion-segment", 0, step=1, min_value=0),
                            html.Div(
                                style={"marginTop": "4px"},
                                children=[
                                    html.Button("Add zoom exclusion", id="add-manual-exclusion", n_clicks=0, style=BUTTON_STYLE),
                                ],
                            ),
                            html.Div(id="manual-exclusion-list", style={"marginTop": "6px"}),
                            html.Button("Clear exclusions", id="clear-manual-exclusions", n_clicks=0, style={**BUTTON_STYLE, "marginTop": "6px"}),
                            html.Div(id="manual-exclusion-status", style=STATUS_STYLE),
                            html.Button("Save current cell", id="save-cell", n_clicks=0, style={**PRIMARY_BUTTON_STYLE, "marginTop": "8px"}),
                            html.Button(
                                "Review saved cell",
                                id="review-saved-cell",
                                n_clicks=0,
                                disabled=True,
                                style={**BUTTON_STYLE, "marginTop": "8px", "width": "100%"},
                            ),
                            html.Button(
                                "Show waveform shapes",
                                id="show-waveform-shapes",
                                n_clicks=0,
                                disabled=True,
                                style={**BUTTON_STYLE, "marginTop": "8px", "width": "100%"},
                            ),
                            html.Div(id="save-status", style=STATUS_STYLE),
                        ],
                    ),
                    html.Div(
                        style=CARD_STYLE,
                        children=[
                            html.Div("Parameters", style=SECTION_HEADER),
                            _number_row("Baseline window (s)", "baseline-window-s", default_params["baseline_window_s"], step="any", min_value=0.001),
                            _number_row("Segment duration (s)", "segment-duration-s", default_params["segment_duration_s"], step="any", min_value=0.5),
                            dcc.Checklist(
                                id="cb-include-first-burst-spike-height",
                                options=[
                                    {"label": "Include first burst spike for spike height", "value": "enabled"},
                                ],
                                value=_cb_include_first_burst_spike_checklist_value(
                                    default_cb_params["include_first_burst_spike_for_spike_height"]
                                ),
                                inputStyle={"marginRight": "5px"},
                                labelStyle={"display": "block", "marginBottom": "6px", "fontSize": "12px"},
                            ),
                        ],
                    ),
                    html.Div(
                        style=CARD_STYLE,
                        children=[
                            html.Div("1st round all spike detection", style=SECTION_HEADER),
                            dcc.Checklist(
                                id="spike-baseline-remove-enabled",
                                options=[
                                    {"label": "Remove baseline before high-pass", "value": "enabled"},
                                ],
                                value=_spike_baseline_checklist_value(default_params["spike_baseline_remove_enabled"]),
                                inputStyle={"marginRight": "5px"},
                                labelStyle={"display": "block", "marginBottom": "6px", "fontSize": "12px"},
                            ),
                            _number_row(
                                "Baseline window (ms)",
                                "spike-baseline-window-ms",
                                default_params["spike_baseline_window_ms"],
                                step="any",
                                min_value=0.001,
                                disabled=not default_params["spike_baseline_remove_enabled"],
                            ),
                            _number_row("High-pass cutoff (Hz)", "highpass-hz", default_params["highpass_hz"], step="any", min_value=0),
                            _number_row("Initial threshold (MAD)", "threshold-mad", default_params["threshold_mad"], step="any", min_value=0),
                        ],
                    ),
                    html.Div(
                        style=CARD_STYLE,
                        children=[
                            html.Div("Complex Bursts", style=SECTION_HEADER),
                            _number_row("CB baseline window (s)", "cb-baseline-window-s", default_cb_params["cb_baseline_window_s"], step="any", min_value=0.001),
                            dcc.Checklist(
                                id="cb-remove-spikes-for-vm",
                                options=[
                                    {"label": "Remove spikes for Vm calculation", "value": "enabled"},
                                ],
                                value=_cb_remove_spikes_checklist_value(default_cb_params["remove_spikes_for_vm"]),
                                inputStyle={"marginRight": "5px"},
                                labelStyle={"display": "block", "marginBottom": "6px", "fontSize": "12px"},
                            ),
                            _number_row("Vm median window (ms)", "cb-vm-median-window-ms", default_cb_params["vm_median_window_ms"], step="any", min_value=0.001),
                            _number_row("Vm crossing threshold", "cb-vm-crossing-threshold", default_cb_params["vm_crossing_threshold"], step="any"),
                            dcc.Checklist(
                                id="cb-refine-onset",
                                options=[
                                    {"label": "Refine CB/plateau onset and offset from Vm slopes", "value": "enabled"},
                                ],
                                value=_cb_refine_onset_checklist_value(default_cb_params["refine_cb_onset"]),
                                inputStyle={"marginRight": "5px"},
                                labelStyle={"display": "block", "marginBottom": "6px", "fontSize": "12px"},
                            ),
                            _number_row("CB onset threshold", "cb-onset-threshold", default_cb_params["cb_onset_threshold"], step="any"),
                            _number_row("CB offset threshold", "cb-offset-threshold", default_cb_params["cb_offset_threshold"], step="any"),
                            _number_row("CB max onset lead / offset lag (ms)", "cb-max-onset-lead-ms", default_cb_params["cb_max_onset_lead_ms"], step="any", min_value=0),
                            _number_row("CB amplitude threshold", "cb-amp-threshold", default_cb_params["cb_amp_threshold"], step="any"),
                            _number_row("CB duration threshold (ms)", "cb-duration-threshold-ms", default_cb_params["cb_duration_threshold_ms"], step="any", min_value=0),
                            _number_row("CB min spikes", "cb-min-spikes", default_cb_params["cb_min_spikes"], step=1, min_value=0),
                            dcc.Checklist(
                                id="cb-require-min-isi",
                                options=[
                                    {"label": "Require min ISI threshold", "value": "enabled"},
                                ],
                                value=_cb_require_min_isi_checklist_value(default_cb_params["cb_require_min_isi"]),
                                inputStyle={"marginRight": "5px"},
                                labelStyle={"display": "block", "marginBottom": "6px", "fontSize": "12px"},
                            ),
                            _number_row(
                                "CB ISI threshold (ms)",
                                "cb-isi-threshold-ms",
                                default_cb_params["cb_isi_threshold_ms"],
                                step="any",
                                min_value=0,
                                disabled=not default_cb_params["cb_require_min_isi"],
                            ),
                            _number_row("Spike height min isolated spikes", "cb-spike-height-min-isolated-spikes", default_cb_params["spike_height_min_isolated_spikes"], step=1, min_value=0),
                        ],
                    ),
                    html.Div(
                        style=CARD_STYLE,
                        children=[
                            html.Div("2nd round spike detection", style=SECTION_HEADER),
                            _number_row("SS threshold (MAD)", "second-round-ss-threshold-mad", default_second_round_params["ss_threshold_mad"], step="any", min_value=0),
                            _number_row("CS threshold in CB (MAD)", "second-round-cs-threshold-mad", default_second_round_params["cs_threshold_mad"], step="any", min_value=0),
                            dcc.Checklist(
                                id="second-round-refine-simple-spikes",
                                options=[
                                    {"label": "Reject small simple spikes", "value": "enabled"},
                                ],
                                value=_second_round_refine_simple_spikes_checklist_value(
                                    default_second_round_params["refine_simple_spikes_by_height"]
                                ),
                                inputStyle={"marginRight": "5px"},
                                labelStyle={"display": "block", "marginBottom": "6px", "fontSize": "12px"},
                            ),
                            _number_row(
                                "SS min height fraction",
                                "second-round-simple-spike-min-height-fraction",
                                default_second_round_params["simple_spike_min_height_fraction"],
                                step="any",
                                min_value=0,
                                disabled=not default_second_round_params["refine_simple_spikes_by_height"],
                            ),
                            _number_row(
                                "CB amplitude threshold",
                                "second-round-cb-amp-threshold",
                                default_second_round_cb_params["cb_amp_threshold"],
                                step="any",
                            ),
                            _number_row(
                                "CB min spikes",
                                "second-round-cb-min-spikes",
                                default_second_round_cb_params["cb_min_spikes"],
                                step=1,
                                min_value=0,
                            ),
                            html.Button(
                                "Run 2nd round",
                                id="run-second-round",
                                n_clicks=0,
                                disabled=True,
                                style={**BUTTON_STYLE, "marginTop": "2px"},
                            ),
                        ],
                    ),
                    html.Div(
                        style=CARD_STYLE,
                        children=[
                            html.Div("Plateau detection", style=SECTION_HEADER),
                            _number_row(
                                "Plateau baseline window (s)",
                                "plateau-baseline-window-s",
                                default_plateau_params["plateau_baseline_window_s"],
                                step="any",
                                min_value=0.001,
                            ),
                            _number_row(
                                "Plateau Vm median window (ms)",
                                "plateau-vm-median-window-ms",
                                default_plateau_params["plateau_vm_median_window_ms"],
                                step="any",
                                min_value=0.001,
                            ),
                            _number_row(
                                "Plateau Vm crossing threshold",
                                "plateau-vm-crossing-threshold",
                                default_plateau_params["plateau_vm_crossing_threshold"],
                                step="any",
                            ),
                            _number_row(
                                "Plateau onset threshold",
                                "plateau-onset-threshold",
                                default_plateau_params["plateau_onset_threshold"],
                                step="any",
                            ),
                            _number_row(
                                "Plateau offset threshold",
                                "plateau-offset-threshold",
                                default_plateau_params["plateau_offset_threshold"],
                                step="any",
                            ),
                            _number_row(
                                "Plateau amplitude threshold",
                                "plateau-amp-threshold",
                                default_plateau_params["plateau_amp_threshold"],
                                step="any",
                            ),
                            _number_row(
                                "Plateau peak fraction",
                                "plateau-peak-fraction-threshold",
                                default_plateau_params["plateau_peak_fraction_threshold"],
                                step="any",
                                min_value=0,
                            ),
                            _number_row(
                                "Peak fraction duration (ms)",
                                "plateau-peak-fraction-duration-ms",
                                default_plateau_params["plateau_peak_fraction_duration_ms"],
                                step="any",
                                min_value=0,
                            ),
                            _number_row(
                                "Plateau duration threshold (ms)",
                                "plateau-duration-threshold-ms",
                                default_plateau_params["plateau_duration_threshold_ms"],
                                step="any",
                                min_value=0,
                            ),
                            _number_row(
                                "Plateau min spikes",
                                "plateau-min-spikes",
                                default_plateau_params["plateau_min_spikes"],
                                step=1,
                                min_value=0,
                            ),
                        ],
                    ),
                    html.Div(
                        style=CARD_STYLE,
                        children=[
                            html.Div("Display", style=SECTION_HEADER),
                            html.Label("Right panel signal", style=LABEL_STYLE),
                            dcc.RadioItems(
                                id="right-panel-mode",
                                options=[
                                    {"label": "Spike-normalized trace + Vm", "value": RIGHT_PANEL_MODE_NORMALIZED},
                                    {"label": "Plateau-normalized trace + Vm", "value": RIGHT_PANEL_MODE_PLATEAU},
                                    {"label": "Raw trace + Vm", "value": RIGHT_PANEL_MODE_TRACE},
                                ],
                                value=default_right_panel_mode,
                                inputStyle={"marginRight": "5px"},
                                labelStyle={"display": "block", "marginBottom": "3px", "fontSize": "12px"},
                                style={"marginBottom": "8px"},
                            ),
                            html.Label("Segment height (px)", style=LABEL_STYLE),
                            dcc.Slider(
                                id="segment-height-px",
                                min=MIN_ROW_HEIGHT_PX,
                                max=MAX_ROW_HEIGHT_PX,
                                step=4,
                                value=default_segment_height_px,
                                marks={
                                    MIN_ROW_HEIGHT_PX: str(MIN_ROW_HEIGHT_PX),
                                    default_segment_height_px: str(default_segment_height_px),
                                    MAX_ROW_HEIGHT_PX: str(MAX_ROW_HEIGHT_PX),
                                },
                                included=False,
                                updatemode="mouseup",
                            ),
                        ],
                    ),
                ],
            ),
            html.Div(
                style={
                    "minWidth": 0,
                    "minHeight": 0,
                    "height": "100%",
                    "display": "grid",
                    "gridTemplateRows": "auto 1fr",
                    "gap": "6px",
                },
                children=[
                    dcc.Tabs(
                        id="plot-view-tabs",
                        value="edit",
                        children=[
                            dcc.Tab(label="Current detection", value="edit"),
                            dcc.Tab(label="Saved review", value="review"),
                            dcc.Tab(label="Waveform shapes", value="waveforms"),
                        ],
                    ),
                    html.Div(
                        style={
                            "minWidth": 0,
                            "minHeight": 0,
                            "height": "100%",
                            "position": "relative",
                        },
                        children=[
                            html.Div(
                                id="edit-tab-content",
                                style=TAB_PANEL_VISIBLE_STYLE,
                                children=[
                                    html.Div(id="detection-status", style={"fontSize": "12px", "color": "#444", "lineHeight": "1.35"}),
                                    html.Div(
                                        id="shared-scroll",
                                        style={
                                            "overflowY": "auto",
                                            "overflowX": "hidden",
                                            "height": "100%",
                                            "minHeight": 0,
                                            "border": "1px solid #ddd",
                                            "borderRadius": "6px",
                                            "backgroundColor": "white",
                                        },
                                        children=[
                                            html.Div(
                                                style={
                                                    "display": "grid",
                                                    "gridTemplateColumns": "minmax(360px, 1fr) minmax(360px, 1fr)",
                                                    "gap": "8px",
                                                    "alignItems": "start",
                                                    "padding": "8px",
                                                },
                                                children=[
                                                    dcc.Graph(id="hp-graph", figure=empty_figure("Select an animal and cell."), config=LEFT_GRAPH_CONFIG),
                                                    dcc.Graph(id="trace-graph", figure=empty_figure("Select an animal and cell."), config=RIGHT_GRAPH_CONFIG),
                                                ],
                                            )
                                        ],
                                    ),
                                ],
                            ),
                            html.Div(
                                id="review-tab-content",
                                style=TAB_PANEL_HIDDEN_STYLE,
                                children=[
                                    html.Div(
                                        style={
                                            "display": "grid",
                                            "gridTemplateColumns": "1fr auto",
                                            "gap": "8px",
                                            "alignItems": "center",
                                        },
                                        children=[
                                            html.Div(id="review-status", style={"fontSize": "12px", "color": "#444", "lineHeight": "1.35"}),
                                            html.Button(
                                                "Back to detection",
                                                id="back-to-edit-tab",
                                                n_clicks=0,
                                                style=BUTTON_STYLE,
                                            ),
                                        ],
                                    ),
                                    html.Div(
                                        id="review-scroll",
                                        style={
                                            "overflow": "auto",
                                            "minHeight": 0,
                                            "height": "100%",
                                            "border": "1px solid #ddd",
                                            "borderRadius": "6px",
                                            "backgroundColor": "white",
                                            "padding": "8px",
                                            "boxSizing": "border-box",
                                        },
                                        children=[html.Div(id="review-graphs")],
                                    ),
                                ],
                            ),
                            html.Div(
                                id="waveform-tab-content",
                                style=TAB_PANEL_HIDDEN_STYLE,
                                children=[
                                    html.Div(
                                        style={
                                            "display": "grid",
                                            "gridTemplateColumns": "1fr auto",
                                            "gap": "8px",
                                            "alignItems": "center",
                                        },
                                        children=[
                                            html.Div(id="waveform-status", style={"fontSize": "12px", "color": "#444", "lineHeight": "1.35"}),
                                            html.Button(
                                                "Back to detection",
                                                id="back-to-edit-tab-waveforms",
                                                n_clicks=0,
                                                style=BUTTON_STYLE,
                                            ),
                                        ],
                                    ),
                                    html.Div(
                                        id="waveform-scroll",
                                        style={
                                            "overflow": "auto",
                                            "minHeight": 0,
                                            "height": "100%",
                                            "border": "1px solid #ddd",
                                            "borderRadius": "6px",
                                            "backgroundColor": "white",
                                            "padding": "8px",
                                            "boxSizing": "border-box",
                                        },
                                        children=[html.Div(id="waveform-graphs")],
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )

    @app.callback(
        Output("cell-dropdown", "options"),
        Output("cell-dropdown", "value"),
        Output("animal-info", "children"),
        Input("animal-dropdown", "value"),
        Input("save-refresh-store", "data"),
        State("cell-dropdown", "value"),
    )
    def refresh_cell_dropdown(animal_id, _refresh_count, current_cell):
        if not animal_id:
            return [], None, f"No CKII folders with {SOURCE_FILENAME} found under {root}."
        try:
            bundle = _load_cached(str(animal_id))
            sidecar = load_sidecar(bundle.animal_dir)
        except Exception as exc:
            return [], None, f"Could not load {animal_id}: {exc}"
        options = _cell_options(bundle, sidecar)
        if current_cell is None:
            value = None
        else:
            value = _safe_int(current_cell, -1)
            if value < 0 or value >= bundle.n_cells:
                value = None
        duration_min = bundle.n_frames / float(bundle.frame_rate) / 60.0
        info = (
            f"{bundle.n_cells} cells, {bundle.n_frames} frames, {duration_min:.1f} min, "
            f"{bundle.frame_rate:g} Hz. Source: {SOURCE_FILENAME}."
        )
        return options, value, info

    @app.callback(
        Output("spike-baseline-window-ms", "disabled"),
        Input("spike-baseline-remove-enabled", "value"),
    )
    def toggle_spike_baseline_window(enabled_value):
        return not _spike_baseline_enabled(enabled_value)

    @app.callback(
        Output("cb-isi-threshold-ms", "disabled"),
        Input("cb-require-min-isi", "value"),
    )
    def toggle_cb_isi_threshold(enabled_value):
        return not _cb_require_min_isi_enabled(enabled_value)

    @app.callback(
        Output("second-round-simple-spike-min-height-fraction", "disabled"),
        Input("second-round-refine-simple-spikes", "value"),
    )
    def toggle_second_round_simple_spike_height_fraction(enabled_value):
        return not _second_round_refine_simple_spikes_enabled(enabled_value)

    @app.callback(
        Output("run-second-round", "disabled"),
        Input("detection-store", "data"),
    )
    def toggle_second_round_button(current_store):
        return not (isinstance(current_store, dict) and "spikes" in current_store and "complex_bursts" in current_store)

    @app.callback(
        Output("review-saved-cell", "disabled"),
        Output("show-waveform-shapes", "disabled"),
        Input("animal-dropdown", "value"),
        Input("cell-dropdown", "value"),
        Input("save-refresh-store", "data"),
    )
    def toggle_saved_result_buttons(animal_id, cell_idx, _refresh_count):
        if not animal_id or cell_idx is None:
            return True, True
        try:
            bundle = _load_cached(str(animal_id))
            saved_cell = get_saved_cell(load_sidecar(bundle.animal_dir), int(cell_idx))
        except Exception:
            saved_cell = None
        disabled = not isinstance(saved_cell, dict)
        return disabled, disabled

    @app.callback(
        Output("plot-view-tabs", "value"),
        Input("review-saved-cell", "n_clicks"),
        Input("show-waveform-shapes", "n_clicks"),
        Input("back-to-edit-tab", "n_clicks"),
        Input("back-to-edit-tab-waveforms", "n_clicks"),
        State("plot-view-tabs", "value"),
        prevent_initial_call=True,
    )
    def switch_plot_tab(_review_clicks, _waveform_clicks, _back_clicks, _back_waveform_clicks, current_tab):
        triggered = ctx.triggered_id
        if triggered == "review-saved-cell":
            return "review"
        if triggered == "show-waveform-shapes":
            return "waveforms"
        if triggered in ("back-to-edit-tab", "back-to-edit-tab-waveforms"):
            return "edit"
        return current_tab or "edit"

    @app.callback(
        Output("edit-tab-content", "style"),
        Output("review-tab-content", "style"),
        Output("waveform-tab-content", "style"),
        Input("plot-view-tabs", "value"),
    )
    def update_plot_tab_visibility(active_tab):
        if active_tab == "review":
            return TAB_PANEL_HIDDEN_STYLE, TAB_PANEL_VISIBLE_STYLE, TAB_PANEL_HIDDEN_STYLE
        if active_tab == "waveforms":
            return TAB_PANEL_HIDDEN_STYLE, TAB_PANEL_HIDDEN_STYLE, TAB_PANEL_VISIBLE_STYLE
        return TAB_PANEL_VISIBLE_STYLE, TAB_PANEL_HIDDEN_STYLE, TAB_PANEL_HIDDEN_STYLE

    @app.callback(
        Output("review-status", "children"),
        Output("review-graphs", "children"),
        Input("plot-view-tabs", "value"),
        Input("animal-dropdown", "value"),
        Input("cell-dropdown", "value"),
        Input("save-refresh-store", "data"),
    )
    def render_saved_review(active_tab, animal_id, cell_idx, _refresh_count):
        if active_tab != "review":
            return "", []
        if not animal_id or cell_idx is None:
            fig = empty_figure("Select an animal and cell.")
            return "Select an animal and cell to review saved data.", [
                dcc.Graph(figure=fig, config=REVIEW_GRAPH_CONFIG)
            ]

        cell_idx = int(cell_idx)
        cell_label = _cell_label(cell_idx)
        cell_text = _cell_text(cell_idx)
        try:
            bundle = _load_cached(str(animal_id))
            if cell_idx < 0 or cell_idx >= bundle.n_cells:
                raise IndexError(f"{cell_label} is outside Cell 1-{bundle.n_cells}.")
            saved_cell = get_saved_cell(load_sidecar(bundle.animal_dir), cell_idx)
        except Exception as exc:
            fig = empty_figure(f"Could not load saved review: {exc}")
            return f"Could not load saved review: {exc}", [dcc.Graph(figure=fig, config=REVIEW_GRAPH_CONFIG)]

        if not isinstance(saved_cell, dict):
            fig = empty_figure("No saved result for this cell.")
            return f"No saved result for {animal_id} {cell_text}.", [
                dcc.Graph(figure=fig, config=REVIEW_GRAPH_CONFIG)
            ]

        try:
            params = normalize_params(**saved_cell.get("params", {}))
            prepared = prepare_detection_inputs(bundle.traces[cell_idx, :], bundle.frame_rate, params)
            analysis_segment_bounds = _saved_review_segment_bounds(
                saved_cell,
                n_frames=bundle.n_frames,
                fallback_bounds=prepared["segment_bounds"],
            )
            review_segment_bounds = _fixed_duration_bounds(
                bundle.n_frames,
                bundle.frame_rate,
                SAVED_REVIEW_ROW_DURATION_S,
            )
            manual_exclusion_periods = _compact_manual_exclusions(
                saved_cell.get("manual_exclusion_periods", []),
                n_frames=bundle.n_frames,
                frame_rate=bundle.frame_rate,
                segment_bounds_in=review_segment_bounds,
            )
            analysis_trace = np.asarray(prepared["trace"], dtype=float).copy()
            exclusion_mask = _manual_exclusion_mask(manual_exclusion_periods, bundle.n_frames)
            analysis_trace[exclusion_mask] = np.nan

            final_spikes = np.sort(
                np.unique(np.asarray(saved_cell.get("spikes", []), dtype=np.int64).reshape(-1))
            )
            final_spikes = final_spikes[(final_spikes >= 0) & (final_spikes < bundle.n_frames)]
            simple_spikes = np.sort(
                np.unique(np.asarray(saved_cell.get("simple_spikes", []), dtype=np.int64).reshape(-1))
            )
            complex_spikes = np.sort(
                np.unique(np.asarray(saved_cell.get("complex_spikes", []), dtype=np.int64).reshape(-1))
            )
            if simple_spikes.size == 0 and complex_spikes.size == 0 and final_spikes.size:
                simple_spikes = final_spikes

            cb_params = normalize_cb_params(**saved_cell.get("cb_params", {}))
            normalized = detect_complex_bursts_segmented(
                analysis_trace,
                analysis_segment_bounds,
                final_spikes,
                bundle.frame_rate,
                cb_params,
            )
            figures = build_saved_review_figures(
                trace=normalized["trace_spike_height_normalized"],
                vm_trace=normalized["vm_spike_height_normalized"],
                segment_bounds=review_segment_bounds,
                simple_spikes=simple_spikes,
                complex_spikes=complex_spikes,
                complex_bursts=saved_cell.get("complex_bursts"),
                failed_min_spikes_windows=saved_cell.get("failed_min_spikes_after_amp_duration_windows"),
                plateaus=saved_cell.get("plateaus"),
                manual_exclusion_periods=manual_exclusion_periods,
                frame_rate=bundle.frame_rate,
                animal_id=str(animal_id),
                cell_idx=cell_idx,
                row_height_px=default_segment_height_px,
                snr_acceptable_until_s=_saved_snr_until(saved_cell, _max_time_s(bundle)),
                segment_snr=None,
                overall_snr=saved_cell.get("overall_snr"),
            )
        except Exception as exc:
            fig = empty_figure(f"Could not build saved review: {exc}")
            return f"Could not build saved review for {animal_id} {cell_text}: {exc}", [
                dcc.Graph(figure=fig, config=REVIEW_GRAPH_CONFIG)
            ]

        cb_count = len(saved_cell.get("complex_bursts", {}).get("starts", [])) if isinstance(saved_cell.get("complex_bursts"), dict) else 0
        putative_count = (
            len(saved_cell.get("failed_min_spikes_after_amp_duration_windows", {}).get("starts", []))
            if isinstance(saved_cell.get("failed_min_spikes_after_amp_duration_windows"), dict)
            else 0
        )
        plateau_count = len(saved_cell.get("plateaus", {}).get("starts", [])) if isinstance(saved_cell.get("plateaus"), dict) else 0
        status = [
            f"Saved only: {animal_id} {cell_text}; {len(final_spikes)} spikes "
            f"({len(simple_spikes)} SS, {len(complex_spikes)} CS); "
            f"{SAVED_REVIEW_ROW_DURATION_S:g}s review rows; ",
            html.Span(f"{cb_count} CBs", style={"color": COMPLEX_SPIKE_COLOR, "fontWeight": 600}),
            "; ",
            html.Span(f"{putative_count} putative CBs", style={"color": FAILED_CB_TEXT_COLOR, "fontWeight": 600}),
            "; ",
            html.Span(f"{plateau_count} plateaus", style={"color": THRESHOLD_COLOR, "fontWeight": 600}),
            "; ",
            html.Span(
                f"overall SNR={_format_snr_value(saved_cell.get('overall_snr'))}",
                style={"color": OVERALL_SNR_COLOR, "fontWeight": 600},
            ),
            ".",
        ]
        graph_nodes = [
            dcc.Graph(
                id={"type": "saved-review-graph", "index": int(idx)},
                figure=fig,
                config=REVIEW_GRAPH_CONFIG,
                style={"marginBottom": "8px"},
            )
            for idx, fig in enumerate(figures)
        ]
        return status, graph_nodes

    @app.callback(
        Output("waveform-status", "children"),
        Output("waveform-graphs", "children"),
        Input("plot-view-tabs", "value"),
        Input("animal-dropdown", "value"),
        Input("cell-dropdown", "value"),
        Input("save-refresh-store", "data"),
    )
    def render_waveform_shapes(active_tab, animal_id, cell_idx, _refresh_count):
        if active_tab != "waveforms":
            return "", []
        if not animal_id or cell_idx is None:
            fig = empty_figure("Select an animal and cell.")
            return "Select an animal and cell to view waveform shapes.", [
                dcc.Graph(figure=fig, config=REVIEW_GRAPH_CONFIG)
            ]

        cell_idx = int(cell_idx)
        cell_label = _cell_label(cell_idx)
        cell_text = _cell_text(cell_idx)
        try:
            bundle = _load_cached(str(animal_id))
            if cell_idx < 0 or cell_idx >= bundle.n_cells:
                raise IndexError(f"{cell_label} is outside Cell 1-{bundle.n_cells}.")
            saved_cell = get_saved_cell(load_sidecar(bundle.animal_dir), cell_idx)
        except Exception as exc:
            fig = empty_figure(f"Could not load saved waveform data: {exc}")
            return f"Could not load saved waveform data: {exc}", [
                dcc.Graph(figure=fig, config=REVIEW_GRAPH_CONFIG)
            ]

        if not isinstance(saved_cell, dict):
            fig = empty_figure("No saved result for this cell.")
            return f"No saved result for {animal_id} {cell_text}.", [
                dcc.Graph(figure=fig, config=REVIEW_GRAPH_CONFIG)
            ]

        try:
            params = normalize_params(**saved_cell.get("params", {}))
            prepared = prepare_detection_inputs(bundle.traces[cell_idx, :], bundle.frame_rate, params)
            analysis_segment_bounds = _saved_review_segment_bounds(
                saved_cell,
                n_frames=bundle.n_frames,
                fallback_bounds=prepared["segment_bounds"],
            )
            manual_exclusion_periods = _compact_manual_exclusions(
                saved_cell.get("manual_exclusion_periods", []),
                n_frames=bundle.n_frames,
                frame_rate=bundle.frame_rate,
                segment_bounds_in=analysis_segment_bounds,
            )
            analysis_trace = np.asarray(prepared["trace"], dtype=float).copy()
            exclusion_mask = _manual_exclusion_mask(manual_exclusion_periods, bundle.n_frames)
            analysis_trace[exclusion_mask] = np.nan
            snr_until = _saved_snr_until(saved_cell, _max_time_s(bundle))
            snr_cutoff_frame = _snr_cutoff_frame(snr_until, bundle.n_frames, bundle.frame_rate)
            if snr_cutoff_frame < bundle.n_frames:
                analysis_trace[snr_cutoff_frame:] = np.nan

            final_spikes = _filter_frames_before_cutoff(
                saved_cell.get("spikes", []),
                snr_cutoff_frame,
                bundle.n_frames,
            )
            simple_spikes = _filter_frames_before_cutoff(
                saved_cell.get("simple_spikes", []),
                snr_cutoff_frame,
                bundle.n_frames,
            )
            complex_spikes = _filter_frames_before_cutoff(
                saved_cell.get("complex_spikes", []),
                snr_cutoff_frame,
                bundle.n_frames,
            )
            if simple_spikes.size == 0 and complex_spikes.size == 0 and final_spikes.size:
                simple_spikes = final_spikes
            filtered_complex_bursts = _filter_event_windows_before_cutoff(
                saved_cell.get("complex_bursts"),
                snr_cutoff_frame,
            )
            filtered_plateaus = _filter_event_windows_before_cutoff(
                saved_cell.get("plateaus"),
                snr_cutoff_frame,
            )

            cb_params = normalize_cb_params(**saved_cell.get("cb_params", {}))
            normalized = detect_complex_bursts_segmented(
                analysis_trace,
                analysis_segment_bounds,
                final_spikes,
                bundle.frame_rate,
                cb_params,
            )

            plateau_trace = None
            plateau_vm_trace = None
            if isinstance(saved_cell.get("plateau_result"), dict) or isinstance(saved_cell.get("plateaus"), dict):
                plateau_params = normalize_plateau_params(**saved_cell.get("plateau_params", {}))
                plateau_normalized = detect_plateaus_segmented(
                    analysis_trace,
                    analysis_segment_bounds,
                    final_spikes,
                    bundle.frame_rate,
                    plateau_params,
                    onset_refinement_params=normalized["cb_params"],
                )
                plateau_trace = plateau_normalized["trace_spike_height_normalized"]
                plateau_vm_trace = plateau_normalized["vm_spike_height_normalized"]

            fig, stats = build_waveform_shape_figure(
                spike_trace=normalized["trace_spike_height_normalized"],
                spike_vm_trace=normalized["vm_spike_height_normalized"],
                plateau_trace=plateau_trace,
                plateau_vm_trace=plateau_vm_trace,
                simple_spikes=simple_spikes,
                complex_spikes=complex_spikes,
                complex_bursts=filtered_complex_bursts,
                plateaus=filtered_plateaus,
                frame_rate=bundle.frame_rate,
                animal_id=str(animal_id),
                cell_idx=cell_idx,
            )
        except Exception as exc:
            fig = empty_figure(f"Could not build waveform shapes: {exc}")
            return f"Could not build waveform shapes for {animal_id} {cell_text}: {exc}", [
                dcc.Graph(figure=fig, config=REVIEW_GRAPH_CONFIG)
            ]

        max_time_s = _max_time_s(bundle)
        snr_note = "full recording" if np.isclose(snr_until, max_time_s, rtol=0.0, atol=1e-6) else f"{snr_until:g}s"
        status = [
            f"Saved only: {animal_id} {cell_text}; ",
            html.Span(
                f"{stats['simple_spike_snippets']}/{stats['simple_spikes']} SS snippets",
                style={"color": "#026C80", "fontWeight": 600},
            ),
            "; ",
            html.Span(
                f"{stats['complex_burst_snippets']}/{stats['complex_bursts']} CB snippets",
                style={"color": COMPLEX_SPIKE_COLOR, "fontWeight": 600},
            ),
            "; ",
            html.Span(
                f"{stats['plateau_snippets']}/{stats['plateaus']} plateau snippets",
                style={"color": THRESHOLD_COLOR, "fontWeight": 600},
            ),
            f"; acceptable SNR until={snr_note}",
            (
                f". Windows: SS {stats['ss_window_ms'][0]:g} to {stats['ss_window_ms'][1]:g} ms, "
                f"CB {stats['cb_window_ms'][0]:g} to {stats['cb_window_ms'][1]:g} ms, "
                f"plateau {stats['plateau_window_ms'][0]:g} to {stats['plateau_window_ms'][1]:g} ms."
            ),
        ]
        return status, [
            dcc.Graph(
                id="waveform-shapes-graph",
                figure=fig,
                config=REVIEW_GRAPH_CONFIG,
                style={"minWidth": "760px"},
            )
        ]

    @app.callback(
        Output("manual-exclusion-status", "children"),
        Input("detection-store", "data"),
    )
    def update_manual_exclusion_status(current_store):
        if not isinstance(current_store, dict):
            return "Excluded: 0 periods."
        periods = _compact_manual_exclusions(current_store.get("manual_exclusion_periods", []))
        total_s = _manual_exclusion_total_s(periods, _safe_float(current_store.get("frame_rate"), 1.0))
        return f"Excluded: {len(periods)} periods, {total_s:g}s total."

    @app.callback(
        Output("manual-exclusion-list", "children"),
        Input("detection-store", "data"),
    )
    def update_manual_exclusion_list(current_store):
        return _manual_exclusion_rows(current_store)

    @app.callback(
        Output("session-cell-cache-store", "data"),
        Input("detection-store", "data"),
        State("session-cell-cache-store", "data"),
    )
    def update_session_cell_cache(current_store, session_cache):
        return _update_session_cell_cache(session_cache, current_store)

    @app.callback(
        Output("baseline-window-s", "value"),
        Output("segment-duration-s", "value"),
        Output("highpass-hz", "value"),
        Output("threshold-mad", "value"),
        Output("spike-baseline-remove-enabled", "value"),
        Output("spike-baseline-window-ms", "value"),
        Output("cb-baseline-window-s", "value"),
        Output("cb-remove-spikes-for-vm", "value"),
        Output("cb-vm-median-window-ms", "value"),
        Output("cb-vm-crossing-threshold", "value"),
        Output("cb-refine-onset", "value"),
        Output("cb-onset-threshold", "value"),
        Output("cb-offset-threshold", "value"),
        Output("cb-max-onset-lead-ms", "value"),
        Output("cb-amp-threshold", "value"),
        Output("cb-duration-threshold-ms", "value"),
        Output("cb-min-spikes", "value"),
        Output("cb-require-min-isi", "value"),
        Output("cb-isi-threshold-ms", "value"),
        Output("cb-spike-height-min-isolated-spikes", "value"),
        Output("cb-include-first-burst-spike-height", "value"),
        Output("second-round-ss-threshold-mad", "value"),
        Output("second-round-cs-threshold-mad", "value"),
        Output("second-round-refine-simple-spikes", "value"),
        Output("second-round-simple-spike-min-height-fraction", "value"),
        Output("second-round-cb-amp-threshold", "value"),
        Output("second-round-cb-min-spikes", "value"),
        Output("plateau-baseline-window-s", "value"),
        Output("plateau-vm-median-window-ms", "value"),
        Output("plateau-vm-crossing-threshold", "value"),
        Output("plateau-onset-threshold", "value"),
        Output("plateau-offset-threshold", "value"),
        Output("plateau-amp-threshold", "value"),
        Output("plateau-peak-fraction-threshold", "value"),
        Output("plateau-peak-fraction-duration-ms", "value"),
        Output("plateau-duration-threshold-ms", "value"),
        Output("plateau-min-spikes", "value"),
        Output("snr-acceptable-until-s", "value"),
        Output("snr-acceptable-until-s", "max"),
        Output("saved-cell-store", "data"),
        Output("cell-info", "children"),
        Input("animal-dropdown", "value"),
        Input("cell-dropdown", "value"),
        Input("save-refresh-store", "data"),
        State("session-cell-cache-store", "data"),
        State("detection-store", "data"),
    )
    def load_cell_settings(animal_id, cell_idx, _refresh_count, session_cache, current_store):
        if not animal_id or cell_idx is None:
            return (
                default_params["baseline_window_s"],
                default_params["segment_duration_s"],
                default_params["highpass_hz"],
                default_params["threshold_mad"],
                _spike_baseline_checklist_value(default_params["spike_baseline_remove_enabled"]),
                default_params["spike_baseline_window_ms"],
                default_cb_params["cb_baseline_window_s"],
                _cb_remove_spikes_checklist_value(default_cb_params["remove_spikes_for_vm"]),
                default_cb_params["vm_median_window_ms"],
                default_cb_params["vm_crossing_threshold"],
                _cb_refine_onset_checklist_value(default_cb_params["refine_cb_onset"]),
                default_cb_params["cb_onset_threshold"],
                default_cb_params["cb_offset_threshold"],
                default_cb_params["cb_max_onset_lead_ms"],
                default_cb_params["cb_amp_threshold"],
                default_cb_params["cb_duration_threshold_ms"],
                int(round(default_cb_params["cb_min_spikes"])),
                _cb_require_min_isi_checklist_value(default_cb_params["cb_require_min_isi"]),
                default_cb_params["cb_isi_threshold_ms"],
                int(round(default_cb_params["spike_height_min_isolated_spikes"])),
                _cb_include_first_burst_spike_checklist_value(
                    default_cb_params["include_first_burst_spike_for_spike_height"]
                ),
                default_second_round_params["ss_threshold_mad"],
                default_second_round_params["cs_threshold_mad"],
                _second_round_refine_simple_spikes_checklist_value(
                    default_second_round_params["refine_simple_spikes_by_height"]
                ),
                default_second_round_params["simple_spike_min_height_fraction"],
                default_second_round_cb_params["cb_amp_threshold"],
                int(round(default_second_round_cb_params["cb_min_spikes"])),
                default_plateau_params["plateau_baseline_window_s"],
                default_plateau_params["plateau_vm_median_window_ms"],
                default_plateau_params["plateau_vm_crossing_threshold"],
                default_plateau_params["plateau_onset_threshold"],
                default_plateau_params["plateau_offset_threshold"],
                default_plateau_params["plateau_amp_threshold"],
                default_plateau_params["plateau_peak_fraction_threshold"],
                default_plateau_params["plateau_peak_fraction_duration_ms"],
                default_plateau_params["plateau_duration_threshold_ms"],
                int(round(default_plateau_params["plateau_min_spikes"])),
                None,
                0,
                None,
                "No cell selected.",
            )
        try:
            bundle = _load_cached(str(animal_id))
            sidecar_cell = get_saved_cell(load_sidecar(bundle.animal_dir), int(cell_idx))
        except Exception:
            bundle = None
            sidecar_cell = None
        saved_cell = sidecar_cell
        use_cached_cell = False
        use_saved_params_for_cell = bool(saved_cell)
        params = normalize_params(
            **(saved_cell.get("params", {}) if use_saved_params_for_cell else default_params)
        )
        saved_cb_params = (
            saved_cell.get("first_round_cb_params", saved_cell.get("cb_params", {}))
            if use_saved_params_for_cell
            else default_cb_params
        )
        cb_params = normalize_cb_params(**saved_cb_params)
        second_round_params = normalize_second_round_params(
            **(
                saved_cell.get("second_round_params", {})
                if use_saved_params_for_cell
                else default_second_round_params
            )
        )
        second_round_cb_params = normalize_second_round_cb_params(
            **(
                saved_cell.get("second_round_cb_params", {})
                if use_saved_params_for_cell
                else default_second_round_cb_params
            )
        )
        plateau_params = normalize_plateau_params(
            **(
                saved_cell.get("plateau_params", {})
                if use_saved_params_for_cell
                else default_plateau_params
            )
        )
        max_time = _max_time_s(bundle) if bundle is not None else 0.0
        snr_until = _saved_snr_until(saved_cell, max_time)
        saved_store = _json_safe_saved_cell(saved_cell)
        status_text = _saved_status_text(saved_cell, int(cell_idx), session_cached=use_cached_cell)
        return (
            params["baseline_window_s"],
            params["segment_duration_s"],
            params["highpass_hz"],
            params["threshold_mad"],
            _spike_baseline_checklist_value(params["spike_baseline_remove_enabled"]),
            params["spike_baseline_window_ms"],
            cb_params["cb_baseline_window_s"],
            _cb_remove_spikes_checklist_value(cb_params["remove_spikes_for_vm"]),
            cb_params["vm_median_window_ms"],
            cb_params["vm_crossing_threshold"],
            _cb_refine_onset_checklist_value(cb_params["refine_cb_onset"]),
            cb_params["cb_onset_threshold"],
            cb_params["cb_offset_threshold"],
            cb_params["cb_max_onset_lead_ms"],
            cb_params["cb_amp_threshold"],
            cb_params["cb_duration_threshold_ms"],
            int(round(cb_params["cb_min_spikes"])),
            _cb_require_min_isi_checklist_value(cb_params["cb_require_min_isi"]),
            cb_params["cb_isi_threshold_ms"],
            int(round(cb_params["spike_height_min_isolated_spikes"])),
            _cb_include_first_burst_spike_checklist_value(
                cb_params["include_first_burst_spike_for_spike_height"]
            ),
            second_round_params["ss_threshold_mad"],
            second_round_params["cs_threshold_mad"],
            _second_round_refine_simple_spikes_checklist_value(
                second_round_params["refine_simple_spikes_by_height"]
            ),
            second_round_params["simple_spike_min_height_fraction"],
            second_round_cb_params["cb_amp_threshold"],
            int(round(second_round_cb_params["cb_min_spikes"])),
            plateau_params["plateau_baseline_window_s"],
            plateau_params["plateau_vm_median_window_ms"],
            plateau_params["plateau_vm_crossing_threshold"],
            plateau_params["plateau_onset_threshold"],
            plateau_params["plateau_offset_threshold"],
            plateau_params["plateau_amp_threshold"],
            plateau_params["plateau_peak_fraction_threshold"],
            plateau_params["plateau_peak_fraction_duration_ms"],
            plateau_params["plateau_duration_threshold_ms"],
            int(round(plateau_params["plateau_min_spikes"])),
            snr_until,
            max_time,
            saved_store,
            status_text,
        )

    @app.callback(
        Output("hp-graph", "figure"),
        Output("trace-graph", "figure"),
        Output("detection-store", "data"),
        Output("detection-status", "children"),
        Input("animal-dropdown", "value"),
        Input("cell-dropdown", "value"),
        Input("baseline-window-s", "value"),
        Input("segment-duration-s", "value"),
        Input("highpass-hz", "value"),
        Input("threshold-mad", "value"),
        Input("spike-baseline-remove-enabled", "value"),
        Input("spike-baseline-window-ms", "value"),
        Input("cb-baseline-window-s", "value"),
        Input("cb-remove-spikes-for-vm", "value"),
        Input("cb-vm-median-window-ms", "value"),
        Input("cb-vm-crossing-threshold", "value"),
        Input("cb-refine-onset", "value"),
        Input("cb-onset-threshold", "value"),
        Input("cb-offset-threshold", "value"),
        Input("cb-max-onset-lead-ms", "value"),
        Input("cb-amp-threshold", "value"),
        Input("cb-duration-threshold-ms", "value"),
        Input("cb-min-spikes", "value"),
        Input("cb-require-min-isi", "value"),
        Input("cb-isi-threshold-ms", "value"),
        Input("cb-spike-height-min-isolated-spikes", "value"),
        Input("cb-include-first-burst-spike-height", "value"),
        Input("second-round-ss-threshold-mad", "value"),
        Input("second-round-cs-threshold-mad", "value"),
        Input("second-round-refine-simple-spikes", "value"),
        Input("second-round-simple-spike-min-height-fraction", "value"),
        Input("second-round-cb-amp-threshold", "value"),
        Input("second-round-cb-min-spikes", "value"),
        Input("plateau-baseline-window-s", "value"),
        Input("plateau-vm-median-window-ms", "value"),
        Input("plateau-vm-crossing-threshold", "value"),
        Input("plateau-onset-threshold", "value"),
        Input("plateau-offset-threshold", "value"),
        Input("plateau-amp-threshold", "value"),
        Input("plateau-peak-fraction-threshold", "value"),
        Input("plateau-peak-fraction-duration-ms", "value"),
        Input("plateau-duration-threshold-ms", "value"),
        Input("plateau-min-spikes", "value"),
        Input("snr-acceptable-until-s", "value"),
        Input("right-panel-mode", "value"),
        Input("segment-height-px", "value"),
        Input("saved-cell-store", "data"),
        Input("hp-graph", "relayoutData"),
        Input("trace-graph", "relayoutData"),
        Input("reset-thresholds", "n_clicks"),
        Input("run-second-round", "n_clicks"),
        Input("add-manual-exclusion", "n_clicks"),
        Input({"type": "cancel-manual-exclusion", "index": ALL}, "n_clicks"),
        Input("clear-manual-exclusions", "n_clicks"),
        State("manual-exclusion-segment", "value"),
        State("detection-store", "data"),
    )
    def render_detection(
        animal_id,
        cell_idx,
        baseline_window_s,
        segment_duration_s,
        highpass_hz,
        threshold_mad,
        spike_baseline_remove_enabled,
        spike_baseline_window_ms,
        cb_baseline_window_s,
        cb_remove_spikes_for_vm,
        cb_vm_median_window_ms,
        cb_vm_crossing_threshold,
        cb_refine_onset,
        cb_onset_threshold,
        cb_offset_threshold,
        cb_max_onset_lead_ms,
        cb_amp_threshold,
        cb_duration_threshold_ms,
        cb_min_spikes,
        cb_require_min_isi,
        cb_isi_threshold_ms,
        cb_spike_height_min_isolated_spikes,
        cb_include_first_burst_spike_height,
        second_round_ss_threshold_mad,
        second_round_cs_threshold_mad,
        second_round_refine_simple_spikes,
        second_round_simple_spike_min_height_fraction,
        second_round_cb_amp_threshold,
        second_round_cb_min_spikes,
        plateau_baseline_window_s,
        plateau_vm_median_window_ms,
        plateau_vm_crossing_threshold,
        plateau_onset_threshold,
        plateau_offset_threshold,
        plateau_amp_threshold,
        plateau_peak_fraction_threshold,
        plateau_peak_fraction_duration_ms,
        plateau_duration_threshold_ms,
        plateau_min_spikes,
        snr_acceptable_until_s,
        right_panel_mode,
        segment_height_px,
        saved_cell,
        hp_relayout_data,
        trace_relayout_data,
        _reset_clicks,
        _run_second_round_clicks,
        _add_manual_exclusion_clicks,
        cancel_manual_exclusion_clicks,
        _clear_manual_exclusions_clicks,
        manual_exclusion_segment,
        current_store,
    ):
        if not animal_id or cell_idx is None:
            fig = empty_figure("Select an animal and cell.")
            return fig, fig, None, "No cell selected."

        cell_idx = int(cell_idx)
        cell_label = _cell_label(cell_idx)
        cell_text = _cell_text(cell_idx)
        if not _store_cell_matches(saved_cell, str(animal_id), cell_idx):
            saved_cell = None
        triggered = ctx.triggered_id
        params = normalize_params(
            baseline_window_s=baseline_window_s,
            segment_duration_s=segment_duration_s,
            highpass_hz=highpass_hz,
            threshold_mad=threshold_mad,
            spike_baseline_remove_enabled=spike_baseline_remove_enabled,
            spike_baseline_window_ms=spike_baseline_window_ms,
        )
        cb_params = normalize_cb_params(
            cb_baseline_window_s=cb_baseline_window_s,
            remove_spikes_for_vm=cb_remove_spikes_for_vm,
            vm_median_window_ms=cb_vm_median_window_ms,
            vm_crossing_threshold=cb_vm_crossing_threshold,
            refine_cb_onset=cb_refine_onset,
            cb_onset_threshold=cb_onset_threshold,
            cb_offset_threshold=cb_offset_threshold,
            cb_max_onset_lead_ms=cb_max_onset_lead_ms,
            cb_amp_threshold=cb_amp_threshold,
            cb_duration_threshold_ms=cb_duration_threshold_ms,
            cb_min_spikes=cb_min_spikes,
            cb_require_min_isi=cb_require_min_isi,
            cb_isi_threshold_ms=cb_isi_threshold_ms,
            spike_height_min_isolated_spikes=cb_spike_height_min_isolated_spikes,
            include_first_burst_spike_for_spike_height=cb_include_first_burst_spike_height,
        )
        second_round_params = normalize_second_round_params(
            ss_threshold_mad=second_round_ss_threshold_mad,
            cs_threshold_mad=second_round_cs_threshold_mad,
            refine_simple_spikes_by_height=second_round_refine_simple_spikes,
            simple_spike_min_height_fraction=second_round_simple_spike_min_height_fraction,
        )
        second_round_cb_params = normalize_second_round_cb_params(
            cb_amp_threshold=second_round_cb_amp_threshold,
            cb_min_spikes=second_round_cb_min_spikes,
        )
        plateau_params = normalize_plateau_params(
            plateau_baseline_window_s=plateau_baseline_window_s,
            plateau_vm_median_window_ms=plateau_vm_median_window_ms,
            plateau_vm_crossing_threshold=plateau_vm_crossing_threshold,
            plateau_onset_threshold=plateau_onset_threshold,
            plateau_offset_threshold=plateau_offset_threshold,
            plateau_amp_threshold=plateau_amp_threshold,
            plateau_peak_fraction_threshold=plateau_peak_fraction_threshold,
            plateau_peak_fraction_duration_ms=plateau_peak_fraction_duration_ms,
            plateau_duration_threshold_ms=plateau_duration_threshold_ms,
            plateau_min_spikes=plateau_min_spikes,
        )
        if triggered in {"animal-dropdown", "cell-dropdown"} and saved_cell is None:
            params = normalize_params(**default_params)
            cb_params = normalize_cb_params(**default_cb_params)
            second_round_params = normalize_second_round_params(**default_second_round_params)
            second_round_cb_params = normalize_second_round_cb_params(**default_second_round_cb_params)
            plateau_params = normalize_plateau_params(**default_plateau_params)
        try:
            bundle = _load_cached(str(animal_id))
            if cell_idx < 0 or cell_idx >= bundle.n_cells:
                raise IndexError(f"{cell_label} is outside Cell 1-{bundle.n_cells}.")
            prepared = _prepare_cached(
                str(animal_id),
                int(cell_idx),
                float(params["baseline_window_s"]),
                float(params["segment_duration_s"]),
                float(params["highpass_hz"]),
                float(params["threshold_mad"]),
                bool(params["spike_baseline_remove_enabled"]),
                float(params["spike_baseline_window_ms"]),
            )
            max_time = _max_time_s(bundle)
        except Exception as exc:
            fig = empty_figure(f"Could not render cell: {exc}")
            return fig, fig, None, f"Could not render {animal_id} {cell_text}: {exc}"

        run_second_round_triggered = _button_was_triggered(triggered, "run-second-round")
        snr_until = _resolve_render_snr_until(
            snr_acceptable_until_s,
            max_time,
            saved_cell=saved_cell,
            current_store=current_store,
            animal_id=str(animal_id),
            cell_idx=cell_idx,
            triggered_id=triggered,
        )
        if _store_cell_matches(current_store, str(animal_id), cell_idx):
            manual_exclusion_periods = _compact_manual_exclusions(
                current_store.get("manual_exclusion_periods", []),
                n_frames=bundle.n_frames,
                frame_rate=bundle.frame_rate,
                segment_bounds_in=prepared["segment_bounds"],
            )
        elif isinstance(saved_cell, dict):
            manual_exclusion_periods = _compact_manual_exclusions(
                saved_cell.get("manual_exclusion_periods", []),
                n_frames=bundle.n_frames,
                frame_rate=bundle.frame_rate,
                segment_bounds_in=prepared["segment_bounds"],
            )
        else:
            manual_exclusion_periods = []

        if triggered == "add-manual-exclusion":
            current_x_range = (
                current_store.get("x_range")
                if _store_cell_matches(current_store, str(animal_id), cell_idx)
                else None
            )
            manual_exclusion_periods = _add_exclusion_from_zoom(
                manual_exclusion_periods,
                segment_idx=manual_exclusion_segment,
                x_range=current_x_range,
                segment_bounds_in=prepared["segment_bounds"],
                frame_rate=bundle.frame_rate,
                n_frames=bundle.n_frames,
            )
        elif (
            isinstance(triggered, dict)
            and triggered.get("type") == "cancel-manual-exclusion"
            and manual_exclusion_periods
            and any(_safe_int(v, 0) > 0 for v in (cancel_manual_exclusion_clicks or []))
        ):
            remove_idx = _safe_int(triggered.get("index"), -1)
            if 0 <= remove_idx < len(manual_exclusion_periods):
                manual_exclusion_periods = [
                    period
                    for idx, period in enumerate(manual_exclusion_periods)
                    if idx != remove_idx
                ]
                manual_exclusion_periods = _compact_manual_exclusions(
                    manual_exclusion_periods,
                    n_frames=bundle.n_frames,
                    frame_rate=bundle.frame_rate,
                    segment_bounds_in=prepared["segment_bounds"],
                )
        elif triggered == "clear-manual-exclusions":
            manual_exclusion_periods = []

        manual_exclusion_mask = _manual_exclusion_mask(manual_exclusion_periods, bundle.n_frames)
        analysis_trace = np.asarray(prepared["trace"], dtype=float).copy()
        analysis_trace_hp = np.asarray(prepared["trace_hp"], dtype=float).copy()
        analysis_trace[manual_exclusion_mask] = np.nan
        analysis_trace_hp[manual_exclusion_mask] = np.nan
        defaults = default_thresholds_for_segments(analysis_trace_hp, prepared["segment_bounds"], params["threshold_mad"])

        threshold_source = None
        if triggered != "reset-thresholds" and _store_identity_matches(current_store, str(animal_id), cell_idx, params):
            threshold_source = current_store.get("thresholds")
        elif (
            triggered != "reset-thresholds"
            and isinstance(saved_cell, dict)
            and _params_equal(saved_cell.get("params"), params)
        ):
            threshold_source = saved_cell.get("thresholds")

        thresholds = sanitize_thresholds(_as_threshold_list(threshold_source), defaults)
        is_threshold_drag = (
            triggered == "hp-graph"
            and _store_identity_matches(current_store, str(animal_id), cell_idx, params)
            and bool(_extract_shape_y_updates(hp_relayout_data))
        )
        if is_threshold_drag:
            thresholds = _apply_shape_drag(hp_relayout_data, thresholds, current_store)
            thresholds = sanitize_thresholds(thresholds, defaults)

        spikes = detect_spikes_from_thresholds(analysis_trace_hp, prepared["segment_bounds"], thresholds)
        cb_result = detect_complex_bursts_segmented(
            analysis_trace,
            prepared["segment_bounds"],
            spikes,
            bundle.frame_rate,
            cb_params,
        )
        first_round_putative_cb_windows = _combine_cb_windows(
            cb_result["complex_bursts"],
            cb_result.get("failed_min_spikes_after_amp_duration_windows"),
        )
        source_signature = _first_round_source_signature(
            params=prepared["params"],
            cb_params=cb_result["cb_params"],
            thresholds=thresholds,
            complex_bursts=cb_result["complex_bursts"],
            failed_min_spikes_windows=cb_result.get("failed_min_spikes_after_amp_duration_windows"),
            manual_exclusion_periods=manual_exclusion_periods,
        )
        second_round_result = None
        if run_second_round_triggered:
            second_round_calc = detect_second_round_spikes(
                analysis_trace_hp,
                prepared["segment_bounds"],
                cb_result["complex_bursts"],
                second_round_params,
                trace=analysis_trace,
                segment_spike_heights=cb_result["segment_spike_heights"],
            )
            second_round_result = {
                "detection_order": SECOND_ROUND_DETECTION_ORDER,
                "source_signature": source_signature,
                "second_round_params": second_round_calc["second_round_params"],
                "spikes": [int(v) for v in np.asarray(second_round_calc["spikes"], dtype=np.int64).reshape(-1)],
                "cb_input_spikes": [
                    int(v)
                    for v in np.asarray(second_round_calc["cb_input_spikes"], dtype=np.int64).reshape(-1)
                ],
                "simple_spikes": [
                    int(v)
                    for v in np.asarray(second_round_calc["simple_spikes"], dtype=np.int64).reshape(-1)
                ],
                "complex_spikes": [
                    int(v)
                    for v in np.asarray(second_round_calc["complex_spikes"], dtype=np.int64).reshape(-1)
                ],
                "removed_simple_spikes": [
                    int(v)
                    for v in np.asarray(second_round_calc["removed_simple_spikes"], dtype=np.int64).reshape(-1)
                ],
                "ss_thresholds": [
                    float(v) if np.isfinite(v) else None
                    for v in np.asarray(second_round_calc["ss_thresholds"], dtype=float).reshape(-1)
                ],
                "cs_thresholds": [
                    float(v) if np.isfinite(v) else None
                    for v in np.asarray(second_round_calc["cs_thresholds"], dtype=float).reshape(-1)
                ],
            }
        elif _store_identity_matches(current_store, str(animal_id), cell_idx, params) and _second_round_source_matches(
            current_store.get("second_round") if isinstance(current_store, dict) else None,
            source_signature,
            second_round_params,
            check_params=False,
        ):
            second_round_result = current_store.get("second_round")
        else:
            second_round_result = _saved_second_round_result(
                saved_cell,
                source_signature=source_signature,
                second_round_params=second_round_params,
                n_segments=len(prepared["segment_bounds"]),
            )
        second_round_active = isinstance(second_round_result, dict)
        display_spikes = spikes
        display_simple_spikes = cb_result["simple_spikes"]
        display_complex_spikes = cb_result["complex_spikes"]
        if second_round_active:
            display_spikes = np.asarray(second_round_result.get("spikes", []), dtype=np.int64)
            display_simple_spikes = np.asarray(second_round_result.get("simple_spikes", []), dtype=np.int64)
            display_complex_spikes = np.asarray(second_round_result.get("complex_spikes", []), dtype=np.int64)
        active_second_round_params = (
            normalize_second_round_params(**second_round_result.get("second_round_params", {}))
            if second_round_active
            else second_round_params
        )
        second_round_cb_source_signature = _second_round_cb_source_signature(second_round_result)
        second_round_cb_result = None
        second_round_cb_calc = None
        active_second_round_cb_params = second_round_cb_params
        final_candidate_complex_spikes = None
        if run_second_round_triggered and second_round_active:
            second_round_cb_full_params = _second_round_full_cb_params(cb_result["cb_params"], second_round_cb_params)
            second_round_cb_input_spikes = _second_round_cb_input_spikes(second_round_result)
            second_round_cb_calc = detect_complex_bursts_segmented(
                analysis_trace,
                prepared["segment_bounds"],
                second_round_cb_input_spikes,
                bundle.frame_rate,
                second_round_cb_full_params,
            )
            second_round_putative_cb_windows = _combine_cb_windows(
                second_round_cb_calc["complex_bursts"],
                second_round_cb_calc.get("failed_min_spikes_after_amp_duration_windows"),
            )
            final_ss_thresholds, final_cs_thresholds = second_round_segment_thresholds(
                analysis_trace_hp,
                prepared["segment_bounds"],
                second_round_putative_cb_windows,
                active_second_round_params,
            )
            candidate_complex_spikes = detect_spikes_in_windows_from_thresholds(
                analysis_trace_hp,
                prepared["segment_bounds"],
                second_round_putative_cb_windows,
                final_cs_thresholds,
            )
            removed_pre_cb_spikes = np.asarray(
                second_round_result.get("removed_simple_spikes", []),
                dtype=np.int64,
            ).reshape(-1)
            if removed_pre_cb_spikes.size:
                candidate_complex_spikes = candidate_complex_spikes[
                    ~np.isin(candidate_complex_spikes, removed_pre_cb_spikes)
                ]
            second_round_cb_calc = _finalize_cb_result_from_cs(
                second_round_cb_calc,
                candidate_complex_spikes,
                bundle.frame_rate,
                second_round_cb_full_params,
                len(prepared["segment_bounds"]),
            )
            final_candidate_complex_spikes = candidate_complex_spikes
            _apply_final_second_round_classification(
                second_round_result,
                second_round_cb_calc,
                final_candidate_complex_spikes,
                second_round_cb_input_spikes,
                analysis_trace.size,
            )
            second_round_result["ss_thresholds"] = [
                float(v) if np.isfinite(v) else None
                for v in np.asarray(final_ss_thresholds, dtype=float).reshape(-1)
            ]
            second_round_result["cs_thresholds"] = [
                float(v) if np.isfinite(v) else None
                for v in np.asarray(final_cs_thresholds, dtype=float).reshape(-1)
            ]
            second_round_cb_source_signature = _second_round_cb_source_signature(second_round_result)
            second_round_cb_result = _compact_cb_detection_result(
                second_round_cb_calc,
                source_signature=second_round_cb_source_signature,
                second_round_cb_params=second_round_cb_params,
            )
            active_second_round_cb_params = normalize_second_round_cb_params(**second_round_cb_params)
        elif second_round_active and _second_round_cb_source_matches(
            current_store.get("second_round_cb") if isinstance(current_store, dict) else None,
            second_round_cb_source_signature,
            second_round_cb_params,
            check_params=False,
        ):
            second_round_cb_result = current_store.get("second_round_cb")
            active_second_round_cb_params = normalize_second_round_cb_params(
                **second_round_cb_result.get("second_round_cb_params", {})
            )
            second_round_cb_full_params = _second_round_full_cb_params(cb_result["cb_params"], active_second_round_cb_params)
            second_round_cb_calc = detect_complex_bursts_segmented(
                analysis_trace,
                prepared["segment_bounds"],
                _second_round_cb_input_spikes(second_round_result),
                bundle.frame_rate,
                second_round_cb_full_params,
            )
            second_round_cb_calc = _finalize_cb_result_from_cs(
                second_round_cb_calc,
                np.asarray(second_round_result.get("complex_spikes", []), dtype=np.int64),
                bundle.frame_rate,
                second_round_cb_full_params,
                len(prepared["segment_bounds"]),
            )
            final_candidate_complex_spikes = np.asarray(second_round_result.get("complex_spikes", []), dtype=np.int64)
            _apply_final_second_round_classification(
                second_round_result,
                second_round_cb_calc,
                final_candidate_complex_spikes,
                _second_round_cb_input_spikes(second_round_result),
                analysis_trace.size,
            )
            second_round_cb_result = _compact_cb_detection_result(
                second_round_cb_calc,
                source_signature=second_round_cb_source_signature,
                second_round_cb_params=active_second_round_cb_params,
            )
        elif second_round_active and _second_round_cb_source_matches(
            saved_cell.get("second_round_cb") if isinstance(saved_cell, dict) else None,
            second_round_cb_source_signature,
            second_round_cb_params,
            check_params=True,
        ):
            second_round_cb_result = saved_cell.get("second_round_cb")
            active_second_round_cb_params = normalize_second_round_cb_params(
                **second_round_cb_result.get("second_round_cb_params", {})
            )
            second_round_cb_full_params = _second_round_full_cb_params(cb_result["cb_params"], active_second_round_cb_params)
            second_round_cb_calc = detect_complex_bursts_segmented(
                analysis_trace,
                prepared["segment_bounds"],
                _second_round_cb_input_spikes(second_round_result),
                bundle.frame_rate,
                second_round_cb_full_params,
            )
            second_round_cb_calc = _finalize_cb_result_from_cs(
                second_round_cb_calc,
                np.asarray(second_round_result.get("complex_spikes", []), dtype=np.int64),
                bundle.frame_rate,
                second_round_cb_full_params,
                len(prepared["segment_bounds"]),
            )
            final_candidate_complex_spikes = np.asarray(second_round_result.get("complex_spikes", []), dtype=np.int64)
            _apply_final_second_round_classification(
                second_round_result,
                second_round_cb_calc,
                final_candidate_complex_spikes,
                _second_round_cb_input_spikes(second_round_result),
                analysis_trace.size,
            )
            second_round_cb_result = _compact_cb_detection_result(
                second_round_cb_calc,
                source_signature=second_round_cb_source_signature,
                second_round_cb_params=active_second_round_cb_params,
            )
        second_round_cb_active = second_round_cb_calc is not None
        active_cb_result = second_round_cb_calc if second_round_cb_active else cb_result
        if second_round_active:
            display_spikes = np.asarray(second_round_result.get("spikes", []), dtype=np.int64)
            display_simple_spikes = np.asarray(second_round_result.get("simple_spikes", []), dtype=np.int64)
            display_complex_spikes = np.asarray(second_round_result.get("complex_spikes", []), dtype=np.int64)
        plateau_result = None
        if second_round_active:
            plateau_result = detect_plateaus_segmented(
                analysis_trace,
                prepared["segment_bounds"],
                display_spikes,
                bundle.frame_rate,
                plateau_params,
                onset_refinement_params=active_cb_result["cb_params"],
            )
        snr_noise_trace = (
            np.asarray(active_cb_result["trace_spike_height_normalized"], dtype=float)
            - np.asarray(active_cb_result["vm_spike_height_normalized"], dtype=float)
        )
        segment_snr_result = calculate_segment_snr(
            snr_noise_trace,
            prepared["segment_bounds"],
            display_spikes,
            bundle.frame_rate,
            complex_bursts=active_cb_result["complex_bursts"],
            failed_min_spikes_windows=active_cb_result.get("failed_min_spikes_after_amp_duration_windows"),
            manual_exclusion_periods=manual_exclusion_periods,
        )
        same_cell_store = _store_cell_matches(current_store, str(animal_id), cell_idx)
        if run_second_round_triggered and second_round_active:
            run_second_round_seen = _safe_int(_run_second_round_clicks, 0)
        elif same_cell_store:
            run_second_round_seen = _safe_int(current_store.get("run_second_round_n_clicks_seen"), 0)
        else:
            run_second_round_seen = _safe_int(_run_second_round_clicks, 0)
        right_panel_mode_clean = _clean_right_panel_mode(right_panel_mode)
        effective_right_panel_mode = right_panel_mode_clean
        if right_panel_mode_clean == RIGHT_PANEL_MODE_PLATEAU and not isinstance(plateau_result, dict):
            effective_right_panel_mode = RIGHT_PANEL_MODE_NORMALIZED
        if effective_right_panel_mode == RIGHT_PANEL_MODE_PLATEAU and isinstance(plateau_result, dict):
            right_trace = plateau_result["trace_spike_height_normalized"]
            right_vm = plateau_result["vm_spike_height_normalized"]
        elif effective_right_panel_mode == RIGHT_PANEL_MODE_NORMALIZED:
            right_trace = active_cb_result["trace_spike_height_normalized"]
            right_vm = active_cb_result["vm_spike_height_normalized"]
        else:
            right_trace = prepared["trace"]
            right_vm = calculate_display_vm(
                prepared["trace"],
                prepared["segment_bounds"],
                display_spikes,
                bundle.frame_rate,
                active_cb_result["cb_params"]["vm_median_window_ms"],
                active_cb_result["cb_params"]["remove_spikes_for_vm"],
            )
        right_panel_name = _right_panel_name(effective_right_panel_mode)
        x_range = _select_x_range(
            triggered_id=triggered,
            hp_relayout_data=hp_relayout_data,
            trace_relayout_data=trace_relayout_data,
            current_store=current_store,
            animal_id=str(animal_id),
            cell_idx=cell_idx,
            params=params,
        )
        row_height = _clean_row_height(segment_height_px)
        left_fig, right_fig, meta = build_detection_figures(
            trace=prepared["trace"],
            trace_hp=prepared["trace_hp"],
            right_trace=right_trace,
            right_vm_trace=right_vm,
            right_trace_name=right_panel_name,
            right_vm_name="Vm",
            segment_bounds=prepared["segment_bounds"],
            thresholds=thresholds,
            second_round_ss_thresholds=(
                np.asarray(second_round_result.get("ss_thresholds", []), dtype=float)
                if second_round_active
                else None
            ),
            second_round_cs_thresholds=(
                np.asarray(second_round_result.get("cs_thresholds", []), dtype=float)
                if second_round_active
                else None
            ),
            spikes=display_spikes,
            simple_spikes=display_simple_spikes,
            complex_spikes=display_complex_spikes,
            complex_bursts=active_cb_result["complex_bursts"],
            failed_min_spikes_windows=active_cb_result.get("failed_min_spikes_after_amp_duration_windows"),
            plateaus=plateau_result.get("plateaus") if isinstance(plateau_result, dict) else None,
            manual_exclusion_periods=manual_exclusion_periods,
            frame_rate=bundle.frame_rate,
            animal_id=str(animal_id),
            cell_idx=cell_idx,
            row_height_px=row_height,
            snr_acceptable_until_s=snr_until,
            segment_snr=segment_snr_result["segment_snr"],
            overall_snr=segment_snr_result["overall_snr"],
            x_range=x_range,
        )
        store = {
            "animal_id": str(animal_id),
            "cell_idx": int(cell_idx),
            "source_file": str(bundle.source_path),
            "source_filename": bundle.source_path.name,
            "frame_rate": float(bundle.frame_rate),
            "trace_key": TRACE_KEY,
            "right_panel_mode": right_panel_mode_clean,
            "snr_acceptable_until_s": float(snr_until),
            "max_time_s": float(max_time),
            "params": prepared["params"],
            "segment_bounds": [[int(s), int(e)] for s, e in prepared["segment_bounds"]],
            "thresholds": [float(v) if np.isfinite(v) else None for v in thresholds],
            "manual_exclusion_periods": manual_exclusion_periods,
            "spikes": [int(v) for v in np.asarray(display_spikes, dtype=np.int64).reshape(-1)],
            "first_round_spikes": [int(v) for v in np.asarray(spikes, dtype=np.int64).reshape(-1)],
            "cb_params": active_cb_result["cb_params"],
            "first_round_cb_params": cb_result["cb_params"],
            "simple_spikes": [int(v) for v in np.asarray(display_simple_spikes, dtype=np.int64).reshape(-1)],
            "complex_spikes": [int(v) for v in np.asarray(display_complex_spikes, dtype=np.int64).reshape(-1)],
            "first_round_simple_spikes": [int(v) for v in np.asarray(cb_result["simple_spikes"], dtype=np.int64).reshape(-1)],
            "first_round_complex_spikes": [int(v) for v in np.asarray(cb_result["complex_spikes"], dtype=np.int64).reshape(-1)],
            "complex_bursts": _compact_complex_bursts(active_cb_result["complex_bursts"]),
            "first_round_complex_bursts": _compact_complex_bursts(cb_result["complex_bursts"]),
            "first_round_putative_cb_windows": _compact_complex_bursts(first_round_putative_cb_windows),
            "failed_min_spikes_after_amp_duration": int(active_cb_result.get("failed_min_spikes_after_amp_duration", 0)),
            "failed_min_spikes_after_amp_duration_by_segment": [
                int(v)
                for v in np.asarray(
                    active_cb_result.get("failed_min_spikes_after_amp_duration_by_segment", []),
                    dtype=np.int64,
                ).reshape(-1)
            ],
            "failed_min_spikes_after_amp_duration_windows": _compact_complex_bursts(
                active_cb_result.get("failed_min_spikes_after_amp_duration_windows", {})
            ),
            "first_round_failed_min_spikes_after_amp_duration": int(
                cb_result.get("failed_min_spikes_after_amp_duration", 0)
            ),
            "first_round_failed_min_spikes_after_amp_duration_by_segment": [
                int(v)
                for v in np.asarray(
                    cb_result.get("failed_min_spikes_after_amp_duration_by_segment", []),
                    dtype=np.int64,
                ).reshape(-1)
            ],
            "first_round_failed_min_spikes_after_amp_duration_windows": _compact_complex_bursts(
                cb_result.get("failed_min_spikes_after_amp_duration_windows", {})
            ),
            "second_round_params": active_second_round_params,
            "second_round": second_round_result,
            "second_round_detection_order": (
                second_round_result.get("detection_order")
                if isinstance(second_round_result, dict)
                else None
            ),
            "second_round_cb_params": active_second_round_cb_params,
            "second_round_cb": second_round_cb_result,
            "plateau_params": plateau_result.get("plateau_params") if isinstance(plateau_result, dict) else plateau_params,
            "plateau_result": _compact_plateau_detection_result(plateau_result),
            "plateaus": _compact_complex_bursts(
                plateau_result.get("plateaus", {}) if isinstance(plateau_result, dict) else {}
            ),
            "plateau_segment_spike_heights": [
                float(v) if np.isfinite(v) else None
                for v in np.asarray(
                    plateau_result.get("segment_spike_heights", []) if isinstance(plateau_result, dict) else [],
                    dtype=float,
                ).reshape(-1)
            ],
            "plateau_segment_spike_height_counts": [
                int(v)
                for v in np.asarray(
                    plateau_result.get("segment_spike_height_counts", []) if isinstance(plateau_result, dict) else [],
                    dtype=np.int64,
                ).reshape(-1)
            ],
            "segment_spike_heights": [
                float(v) if np.isfinite(v) else None
                for v in np.asarray(active_cb_result["segment_spike_heights"], dtype=float).reshape(-1)
            ],
            "segment_spike_height_counts": [
                int(v)
                for v in np.asarray(active_cb_result["segment_spike_height_counts"], dtype=np.int64).reshape(-1)
            ],
            "segment_snr": [
                float(v) if np.isfinite(v) else None
                for v in np.asarray(segment_snr_result["segment_snr"], dtype=float).reshape(-1)
            ],
            "segment_snr_noise": [
                float(v) if np.isfinite(v) else None
                for v in np.asarray(segment_snr_result["segment_noise"], dtype=float).reshape(-1)
            ],
            "segment_snr_baseline_counts": [
                int(v)
                for v in np.asarray(segment_snr_result["segment_baseline_counts"], dtype=np.int64).reshape(-1)
            ],
            "overall_snr": (
                float(segment_snr_result["overall_snr"])
                if np.isfinite(float(segment_snr_result["overall_snr"]))
                else None
            ),
            "snr_spike_mask_ms": float(segment_snr_result["spike_mask_ms"]),
            "row_offsets": meta.get("row_offsets", []),
            "threshold_center": meta.get("threshold_center", 0.0),
            "threshold_scale": meta.get("threshold_scale", 1.0),
            "trace_scale": meta.get("trace_scale", 1.0),
            "shape_to_segment": meta.get("shape_to_segment", []),
            "x_range": meta.get("x_range"),
            "row_height_px": int(row_height),
            "left_signal_trace_indices": meta.get("left_signal_trace_indices", []),
            "right_signal_trace_indices": meta.get("right_signal_trace_indices", []),
            "left_vm_placeholder_indices": meta.get("left_vm_placeholder_indices", []),
            "right_vm_trace_indices": meta.get("right_vm_trace_indices", []),
            "simple_spike_trace_index": int(meta.get("simple_spike_trace_index", len(prepared["segment_bounds"]))),
            "complex_spike_trace_index": int(meta.get("complex_spike_trace_index", len(prepared["segment_bounds"]) + 1)),
            "spike_trace_index": int(meta.get("spike_trace_index", len(prepared["segment_bounds"]))),
            "run_second_round_n_clicks_seen": int(run_second_round_seen),
        }
        duration_min = bundle.n_frames / float(bundle.frame_rate) / 60.0
        snr_note = "full recording" if np.isclose(snr_until, max_time, rtol=0.0, atol=1e-6) else f"{snr_until:g}s"
        manual_exclusion_note = (
            f"manual exclusions={len(manual_exclusion_periods)} "
            f"({_manual_exclusion_total_s(manual_exclusion_periods, bundle.frame_rate):g}s)"
        )
        failed_min_spike_candidates = int(active_cb_result.get("failed_min_spikes_after_amp_duration", 0))
        plateau_count = (
            len(plateau_result.get("plateaus", {}).get("starts", []))
            if isinstance(plateau_result, dict) and isinstance(plateau_result.get("plateaus"), dict)
            else 0
        )
        if second_round_active:
            second_round_pending = not _second_round_params_equal(active_second_round_params, second_round_params)
            second_round_cb_pending = (
                second_round_cb_active
                and not _second_round_cb_params_equal(active_second_round_cb_params, second_round_cb_params)
            )
            pending_parts = []
            if second_round_pending:
                pending_parts.append("typed 2nd-round thresholds pending Run")
            if second_round_cb_pending:
                pending_parts.append("typed 2nd-round CB parameters pending Run")
            pending_note = "; " + ", ".join(pending_parts) if pending_parts else ""
            round_label = "2nd round spike detection" if second_round_cb_active else "2nd round"
            refined_ss_note = (
                f", removed {len(second_round_result.get('removed_simple_spikes', []))} small SS"
                if active_second_round_params["refine_simple_spikes_by_height"]
                else ""
            )
            round_note = (
                f"{round_label}: {len(display_spikes)} spikes "
                f"({len(display_simple_spikes)} SS, {len(display_complex_spikes)} CS)"
                f"{refined_ss_note}"
                f"{pending_note}"
            )
        else:
            round_note = (
                f"1st round: {len(spikes)} spikes "
                f"({len(cb_result['simple_spikes'])} simple, {len(cb_result['complex_spikes'])} complex); "
                "2nd round not run"
            )
        overall_snr_text = f"overall SNR={_format_snr_value(segment_snr_result['overall_snr'])}"
        status = [
            f"{animal_id} {cell_text}: {round_note}; ",
            html.Span(
                f"{len(active_cb_result['complex_bursts']['starts'])} CBs across {len(prepared['segment_bounds'])} "
                f"segments",
                style={"color": COMPLEX_SPIKE_COLOR, "fontWeight": 600},
            ),
            f" from {duration_min:.1f} min; ",
            html.Span(
            f"putative CB windows passing amp/duration but failing min spikes={failed_min_spike_candidates}",
                style={"color": FAILED_CB_TEXT_COLOR, "fontWeight": 600},
            ),
            "; ",
            html.Span(
                f"{plateau_count} plateaus",
                style={"color": THRESHOLD_COLOR, "fontWeight": 600},
            ),
            "; ",
            html.Span(overall_snr_text, style={"color": OVERALL_SNR_COLOR, "fontWeight": 600}),
            f"; {manual_exclusion_note}; acceptable SNR until={snr_note}.",
        ]
        if is_threshold_drag:
            return left_fig, right_fig, store, status
        return left_fig, right_fig, store, status

    @app.callback(
        Output("save-status", "children"),
        Output("save-refresh-store", "data"),
        Input("save-cell", "n_clicks"),
        State("animal-dropdown", "value"),
        State("cell-dropdown", "value"),
        State("baseline-window-s", "value"),
        State("segment-duration-s", "value"),
        State("highpass-hz", "value"),
        State("threshold-mad", "value"),
        State("spike-baseline-remove-enabled", "value"),
        State("spike-baseline-window-ms", "value"),
        State("cb-baseline-window-s", "value"),
        State("cb-remove-spikes-for-vm", "value"),
        State("cb-vm-median-window-ms", "value"),
        State("cb-vm-crossing-threshold", "value"),
        State("cb-refine-onset", "value"),
        State("cb-onset-threshold", "value"),
        State("cb-offset-threshold", "value"),
        State("cb-max-onset-lead-ms", "value"),
        State("cb-amp-threshold", "value"),
        State("cb-duration-threshold-ms", "value"),
        State("cb-min-spikes", "value"),
        State("cb-require-min-isi", "value"),
        State("cb-isi-threshold-ms", "value"),
        State("cb-spike-height-min-isolated-spikes", "value"),
        State("cb-include-first-burst-spike-height", "value"),
        State("second-round-ss-threshold-mad", "value"),
        State("second-round-cs-threshold-mad", "value"),
        State("second-round-refine-simple-spikes", "value"),
        State("second-round-simple-spike-min-height-fraction", "value"),
        State("second-round-cb-amp-threshold", "value"),
        State("second-round-cb-min-spikes", "value"),
        State("plateau-baseline-window-s", "value"),
        State("plateau-vm-median-window-ms", "value"),
        State("plateau-vm-crossing-threshold", "value"),
        State("plateau-onset-threshold", "value"),
        State("plateau-offset-threshold", "value"),
        State("plateau-amp-threshold", "value"),
        State("plateau-peak-fraction-threshold", "value"),
        State("plateau-peak-fraction-duration-ms", "value"),
        State("plateau-duration-threshold-ms", "value"),
        State("plateau-min-spikes", "value"),
        State("snr-acceptable-until-s", "value"),
        State("run-second-round", "n_clicks"),
        State("detection-store", "data"),
        State("save-refresh-store", "data"),
        prevent_initial_call=True,
    )
    def save_current_cell(
        n_clicks,
        animal_id,
        cell_idx,
        baseline_window_s,
        segment_duration_s,
        highpass_hz,
        threshold_mad,
        spike_baseline_remove_enabled,
        spike_baseline_window_ms,
        cb_baseline_window_s,
        cb_remove_spikes_for_vm,
        cb_vm_median_window_ms,
        cb_vm_crossing_threshold,
        cb_refine_onset,
        cb_onset_threshold,
        cb_offset_threshold,
        cb_max_onset_lead_ms,
        cb_amp_threshold,
        cb_duration_threshold_ms,
        cb_min_spikes,
        cb_require_min_isi,
        cb_isi_threshold_ms,
        cb_spike_height_min_isolated_spikes,
        cb_include_first_burst_spike_height,
        second_round_ss_threshold_mad,
        second_round_cs_threshold_mad,
        second_round_refine_simple_spikes,
        second_round_simple_spike_min_height_fraction,
        second_round_cb_amp_threshold,
        second_round_cb_min_spikes,
        plateau_baseline_window_s,
        plateau_vm_median_window_ms,
        plateau_vm_crossing_threshold,
        plateau_onset_threshold,
        plateau_offset_threshold,
        plateau_amp_threshold,
        plateau_peak_fraction_threshold,
        plateau_peak_fraction_duration_ms,
        plateau_duration_threshold_ms,
        plateau_min_spikes,
        snr_acceptable_until_s,
        run_second_round_clicks,
        current_store,
        refresh_count,
    ):
        if not n_clicks:
            raise PreventUpdate
        if not animal_id or cell_idx is None or not isinstance(current_store, dict):
            return "Nothing to save yet.", refresh_count
        cell_idx = int(cell_idx)
        cell_label = _cell_label(cell_idx)
        params = normalize_params(
            baseline_window_s=baseline_window_s,
            segment_duration_s=segment_duration_s,
            highpass_hz=highpass_hz,
            threshold_mad=threshold_mad,
            spike_baseline_remove_enabled=spike_baseline_remove_enabled,
            spike_baseline_window_ms=spike_baseline_window_ms,
        )
        cb_params = normalize_cb_params(
            cb_baseline_window_s=cb_baseline_window_s,
            remove_spikes_for_vm=cb_remove_spikes_for_vm,
            vm_median_window_ms=cb_vm_median_window_ms,
            vm_crossing_threshold=cb_vm_crossing_threshold,
            refine_cb_onset=cb_refine_onset,
            cb_onset_threshold=cb_onset_threshold,
            cb_offset_threshold=cb_offset_threshold,
            cb_max_onset_lead_ms=cb_max_onset_lead_ms,
            cb_amp_threshold=cb_amp_threshold,
            cb_duration_threshold_ms=cb_duration_threshold_ms,
            cb_min_spikes=cb_min_spikes,
            cb_require_min_isi=cb_require_min_isi,
            cb_isi_threshold_ms=cb_isi_threshold_ms,
            spike_height_min_isolated_spikes=cb_spike_height_min_isolated_spikes,
            include_first_burst_spike_for_spike_height=cb_include_first_burst_spike_height,
        )
        second_round_params = normalize_second_round_params(
            ss_threshold_mad=second_round_ss_threshold_mad,
            cs_threshold_mad=second_round_cs_threshold_mad,
            refine_simple_spikes_by_height=second_round_refine_simple_spikes,
            simple_spike_min_height_fraction=second_round_simple_spike_min_height_fraction,
        )
        second_round_cb_params = normalize_second_round_cb_params(
            cb_amp_threshold=second_round_cb_amp_threshold,
            cb_min_spikes=second_round_cb_min_spikes,
        )
        plateau_params = normalize_plateau_params(
            plateau_baseline_window_s=plateau_baseline_window_s,
            plateau_vm_median_window_ms=plateau_vm_median_window_ms,
            plateau_vm_crossing_threshold=plateau_vm_crossing_threshold,
            plateau_onset_threshold=plateau_onset_threshold,
            plateau_offset_threshold=plateau_offset_threshold,
            plateau_amp_threshold=plateau_amp_threshold,
            plateau_peak_fraction_threshold=plateau_peak_fraction_threshold,
            plateau_peak_fraction_duration_ms=plateau_peak_fraction_duration_ms,
            plateau_duration_threshold_ms=plateau_duration_threshold_ms,
            plateau_min_spikes=plateau_min_spikes,
        )
        try:
            bundle = _load_cached(str(animal_id))
            ready, stale_message = _store_ready_for_save(
                current_store,
                animal_id=str(animal_id),
                cell_idx=cell_idx,
                params=params,
                cb_params=cb_params,
                second_round_params=second_round_params,
                second_round_cb_params=second_round_cb_params,
                plateau_params=plateau_params,
                snr_acceptable_until_s=snr_acceptable_until_s,
                max_time_s=_max_time_s(bundle),
                run_second_round_clicks=run_second_round_clicks,
            )
            if not ready:
                return stale_message, refresh_count
            segment_bounds_for_save = [tuple(v) for v in current_store["segment_bounds"]]
            manual_exclusion_periods = _compact_manual_exclusions(
                current_store.get("manual_exclusion_periods", []),
                n_frames=bundle.n_frames,
                frame_rate=bundle.frame_rate,
                segment_bounds_in=segment_bounds_for_save,
            )
            second_round_store = current_store.get("second_round")
            if isinstance(second_round_store, dict):
                second_round_ss_thresholds = np.asarray(second_round_store.get("ss_thresholds", []), dtype=float)
                second_round_cs_thresholds = np.asarray(second_round_store.get("cs_thresholds", []), dtype=float)
            else:
                second_round_ss_thresholds = np.array([], dtype=float)
                second_round_cs_thresholds = np.array([], dtype=float)
            result = make_result_payload(
                animal_id=str(animal_id),
                cell_idx=cell_idx,
                frame_rate=bundle.frame_rate,
                source_file=str(bundle.source_path),
                source_filename=bundle.source_path.name,
                snr_acceptable_until_s=current_store.get("snr_acceptable_until_s"),
                params=current_store["params"],
                segment_bounds_in=segment_bounds_for_save,
                thresholds=np.asarray(current_store["thresholds"], dtype=float),
                manual_exclusion_periods=manual_exclusion_periods,
                spikes=np.asarray(current_store["spikes"], dtype=np.int64),
                cb_params=current_store.get("cb_params"),
                first_round_cb_params=current_store.get("first_round_cb_params", current_store.get("cb_params")),
                simple_spikes=np.asarray(current_store.get("simple_spikes", []), dtype=np.int64),
                complex_spikes=np.asarray(current_store.get("complex_spikes", []), dtype=np.int64),
                complex_bursts=current_store.get("complex_bursts"),
                first_round_complex_bursts=current_store.get("first_round_complex_bursts", current_store.get("complex_bursts")),
                second_round=current_store.get("second_round"),
                second_round_params=current_store.get("second_round_params"),
                second_round_ss_thresholds=second_round_ss_thresholds,
                second_round_cs_thresholds=second_round_cs_thresholds,
                second_round_cb_params=current_store.get("second_round_cb_params"),
                second_round_cb=current_store.get("second_round_cb"),
                plateau_params=current_store.get("plateau_params"),
                plateau_result=current_store.get("plateau_result"),
                plateaus=current_store.get("plateaus"),
                failed_min_spikes_after_amp_duration=current_store.get("failed_min_spikes_after_amp_duration", 0),
                failed_min_spikes_after_amp_duration_by_segment=np.asarray(
                    current_store.get("failed_min_spikes_after_amp_duration_by_segment", []),
                    dtype=np.int64,
                ),
                failed_min_spikes_after_amp_duration_windows=current_store.get(
                    "failed_min_spikes_after_amp_duration_windows"
                ),
                first_round_failed_min_spikes_after_amp_duration=current_store.get(
                    "first_round_failed_min_spikes_after_amp_duration",
                    current_store.get("failed_min_spikes_after_amp_duration", 0),
                ),
                first_round_failed_min_spikes_after_amp_duration_by_segment=np.asarray(
                    current_store.get(
                        "first_round_failed_min_spikes_after_amp_duration_by_segment",
                        current_store.get("failed_min_spikes_after_amp_duration_by_segment", []),
                    ),
                    dtype=np.int64,
                ),
                first_round_failed_min_spikes_after_amp_duration_windows=current_store.get(
                    "first_round_failed_min_spikes_after_amp_duration_windows",
                    current_store.get("failed_min_spikes_after_amp_duration_windows"),
                ),
                segment_spike_heights=np.asarray(current_store.get("segment_spike_heights", []), dtype=float),
                segment_spike_height_counts=np.asarray(current_store.get("segment_spike_height_counts", []), dtype=np.int64),
                segment_snr=np.asarray(current_store.get("segment_snr", []), dtype=float),
                segment_snr_noise=np.asarray(current_store.get("segment_snr_noise", []), dtype=float),
                segment_snr_baseline_counts=np.asarray(
                    current_store.get("segment_snr_baseline_counts", []),
                    dtype=np.int64,
                ),
                overall_snr=current_store.get("overall_snr"),
                snr_spike_mask_ms=current_store.get("snr_spike_mask_ms"),
            )
            path = save_cell_result(bundle, cell_idx, result)
            saved_cell = get_saved_cell(load_sidecar(bundle.animal_dir), cell_idx)
            saved_exclusion_count = len(
                _compact_manual_exclusions(
                    saved_cell.get("manual_exclusion_periods", []) if isinstance(saved_cell, dict) else [],
                    n_frames=bundle.n_frames,
                    frame_rate=bundle.frame_rate,
                    segment_bounds_in=segment_bounds_for_save,
                )
            )
        except Exception as exc:
            return f"Save failed: {exc}", refresh_count
        saved_second_round = isinstance(result.get("second_round"), dict)
        saved_second_round_cb = isinstance(result.get("second_round_cb"), dict)
        saved_second_round_complete = saved_second_round and saved_second_round_cb
        saved_plateau_count = 0
        if isinstance(result.get("plateaus"), dict):
            saved_plateau_count = len(result["plateaus"].get("starts", []))
        return (
            f"Saved {cell_label} to {path.name} with {saved_exclusion_count} exclusions. "
            f"2nd round spike detection={'yes' if saved_second_round_complete else 'no'}; "
            f"plateaus={saved_plateau_count}.",
            int(refresh_count or 0) + 1,
        )

    return app


def _browser_host(host: str) -> str:
    h = str(host).strip().lower()
    if h in {"0.0.0.0", "::", ""}:
        return "127.0.0.1"
    return host


def _open_browser_later(host: str, port: int, delay_s: float = 1.0) -> None:
    url = f"http://{_browser_host(host)}:{int(port)}"
    timer = threading.Timer(delay_s, lambda: webbrowser.open(url))
    timer.daemon = True
    timer.start()


def run_from_notebook(
    data_root: str | Path | None = None,
    *,
    host: str = "127.0.0.1",
    port: int = 8054,
    debug: bool = False,
    gui_defaults: dict[str, Any] | None = None,
):
    app = create_app(data_root=data_root, gui_defaults=gui_defaults)
    _open_browser_later(host, port)
    app.run(host=host, port=port, debug=debug)


def main() -> None:
    parser = argparse.ArgumentParser(description="Dash manual all-spike detection GUI")
    parser.add_argument("--data-root", type=str, default=str(default_data_root()))
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8054)
    parser.add_argument("--defaults-json", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    app = create_app(data_root=args.data_root, gui_defaults=_load_gui_defaults_json(args.defaults_json))
    if not args.no_browser:
        _open_browser_later(args.host, args.port)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
