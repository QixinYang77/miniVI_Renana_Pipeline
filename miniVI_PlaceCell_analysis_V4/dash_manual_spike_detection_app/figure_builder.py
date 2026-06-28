"""Plotly figure builders for the manual spike detection GUI."""

from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

TRACE_COLOR = "#222222"
HP_COLOR = "#111111"
VM_COLOR = "#8f8f8f"
THRESHOLD_COLOR = "#D62828"
SPIKE_COLOR = "#777777"
SIMPLE_SPIKE_COLOR = "#026C80"
COMPLEX_SPIKE_COLOR = "#EE9B00"
CB_FILL = "rgba(255, 214, 10, 0.24)"
FAILED_CB_FILL = "rgba(123, 44, 191, 0.42)"
PLATEAU_FILL = "rgba(214, 40, 40, 0.52)"
PLATEAU_COLOR = "#D62828"
ZERO_LINE_COLOR = "rgba(255, 255, 255, 0.95)"
SNR_MASK_FILL = "rgba(120, 120, 120, 0.26)"
MANUAL_EXCLUSION_FILL = "rgba(45, 45, 45, 0.46)"
GRID_COLOR = "#eeeeee"
OVERALL_SNR_COLOR = "#7B3F00"
TRACE_SCALE = 0.62
TRACE_PAD = 0.12
RASTER_PAD = 0.12
RASTER_HEIGHT = 0.24
ROW_GAP = 0.36
PLOT_PAD = 0.08
DEFAULT_ROW_HEIGHT_PX = 200
MIN_ROW_HEIGHT_PX = 150
MAX_ROW_HEIGHT_PX = 300
DEFAULT_HIGHPASS_GAIN = 1.0
MIN_HIGHPASS_GAIN = 0.5
MAX_HIGHPASS_GAIN = 8.0


def empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        template="plotly_white",
        annotations=[
            {
                "text": message,
                "xref": "paper",
                "yref": "paper",
                "x": 0.5,
                "y": 0.5,
                "showarrow": False,
                "font": {"size": 13, "color": "#666"},
            }
        ],
        xaxis={"visible": False},
        yaxis={"visible": False},
        height=450,
        margin={"l": 40, "r": 20, "t": 30, "b": 30},
    )
    return fig


def _event_starts(events: dict[str, Any] | None) -> np.ndarray:
    if not isinstance(events, dict):
        return np.array([], dtype=np.int64)
    try:
        starts = np.asarray(events.get("starts", []), dtype=np.int64).reshape(-1)
    except Exception:
        return np.array([], dtype=np.int64)
    return np.sort(np.unique(starts[starts >= 0]))


def _event_ends(events: dict[str, Any] | None) -> np.ndarray:
    if not isinstance(events, dict):
        return np.array([], dtype=np.int64)
    try:
        ends = np.asarray(events.get("ends", []), dtype=np.int64).reshape(-1)
    except Exception:
        return np.array([], dtype=np.int64)
    return ends


def _filter_min_separation(events: np.ndarray, frame_rate: float, min_separation_ms: float | None) -> np.ndarray:
    events = np.sort(np.unique(np.asarray(events, dtype=np.int64).reshape(-1)))
    if events.size <= 1 or min_separation_ms is None:
        return events
    min_sep_frames = int(round(float(min_separation_ms) * float(frame_rate) / 1000.0))
    if min_sep_frames <= 0:
        return events
    diffs = np.diff(events)
    previous = np.r_[np.inf, diffs]
    next_ = np.r_[diffs, np.inf]
    return events[np.minimum(previous, next_) >= min_sep_frames]


def _first_spike_per_burst(
    complex_bursts: dict[str, Any] | None,
    complex_spikes: np.ndarray | None,
    n_frames: int,
) -> np.ndarray:
    starts = _event_starts(complex_bursts)
    ends = _event_ends(complex_bursts)
    if starts.size == 0:
        return np.array([], dtype=np.int64)
    complex_spikes_arr = np.sort(
        np.unique(np.asarray(complex_spikes if complex_spikes is not None else [], dtype=np.int64).reshape(-1))
    )
    complex_spikes_arr = complex_spikes_arr[(complex_spikes_arr >= 0) & (complex_spikes_arr < int(n_frames))]
    event_frames: list[int] = []
    if complex_spikes_arr.size and ends.size:
        for start, end in zip(starts, ends):
            s = int(min(start, end))
            e = int(max(start, end))
            in_burst = complex_spikes_arr[(complex_spikes_arr >= s) & (complex_spikes_arr <= e)]
            if in_burst.size:
                event_frames.append(int(in_burst[0]))
    if not event_frames:
        event_frames = [int(v) for v in starts]
    return np.sort(np.unique(np.asarray(event_frames, dtype=np.int64)))


def _collect_aligned_waveforms(
    trace: np.ndarray,
    event_frames: np.ndarray,
    frame_rate: float,
    *,
    pre_ms: float,
    post_ms: float,
) -> tuple[np.ndarray, np.ndarray]:
    trace = np.asarray(trace, dtype=float).reshape(-1)
    frames = np.sort(np.unique(np.asarray(event_frames, dtype=np.int64).reshape(-1)))
    frames = frames[(frames >= 0) & (frames < trace.size)]
    if trace.size == 0 or frames.size == 0 or not np.isfinite(float(frame_rate)) or float(frame_rate) <= 0:
        return np.array([], dtype=float), np.empty((0, 0), dtype=float)
    pre_frames = max(0, int(round(float(pre_ms) * float(frame_rate) / 1000.0)))
    post_frames = max(0, int(round(float(post_ms) * float(frame_rate) / 1000.0)))
    width = int(pre_frames + post_frames + 1)
    if width <= 0:
        return np.array([], dtype=float), np.empty((0, 0), dtype=float)
    time_ms = (np.arange(width, dtype=float) - float(pre_frames)) * 1000.0 / float(frame_rate)
    snippets: list[np.ndarray] = []
    for frame in frames:
        start = int(frame) - pre_frames
        end = int(frame) + post_frames + 1
        if start < 0 or end > trace.size:
            continue
        snippet = np.asarray(trace[start:end], dtype=float)
        if snippet.size != width or not np.any(np.isfinite(snippet)):
            continue
        snippets.append(snippet)
    if not snippets:
        return time_ms, np.empty((0, width), dtype=float)
    return time_ms, np.vstack(snippets)


def _collect_local_normalized_waveforms(
    trace: np.ndarray,
    event_frames: np.ndarray,
    frame_rate: float,
    *,
    pre_ms: float,
    post_ms: float,
    baseline_frames: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    trace = np.asarray(trace, dtype=float).reshape(-1)
    frames = np.sort(np.unique(np.asarray(event_frames, dtype=np.int64).reshape(-1)))
    frames = frames[(frames >= 0) & (frames < trace.size)]
    if trace.size == 0 or frames.size == 0 or not np.isfinite(float(frame_rate)) or float(frame_rate) <= 0:
        return np.array([], dtype=float), np.empty((0, 0), dtype=float)
    pre_frames = max(0, int(round(float(pre_ms) * float(frame_rate) / 1000.0)))
    post_frames = max(0, int(round(float(post_ms) * float(frame_rate) / 1000.0)))
    width = int(pre_frames + post_frames + 1)
    time_ms = (np.arange(-pre_frames, post_frames + 1, dtype=float) / float(frame_rate)) * 1000.0
    if width <= 0:
        return time_ms, np.empty((0, 0), dtype=float)

    snippets: list[np.ndarray] = []
    for frame in frames:
        frame = int(frame)
        if frame - pre_frames < 0 or frame + post_frames >= trace.size:
            continue
        baseline_region = trace[max(0, frame - int(baseline_frames)):frame]
        baseline_region = baseline_region[np.isfinite(baseline_region)]
        baseline = float(np.nanmin(baseline_region)) if baseline_region.size else 0.0
        height = float(trace[frame] - baseline) if np.isfinite(trace[frame]) else float("nan")
        if not np.isfinite(height) or abs(height) < 1e-12:
            continue
        window = trace[frame - pre_frames:frame + post_frames + 1]
        if window.size != width or not np.any(np.isfinite(window)):
            continue
        snippets.append((window - baseline) / height)
    if not snippets:
        return time_ms, np.empty((0, width), dtype=float)
    return time_ms, np.vstack(snippets)


def _collect_plateau_segments(
    trace: np.ndarray,
    plateaus: dict[str, Any] | None,
    frame_rate: float,
    *,
    pre_ms: float = 40.0,
    post_ms: float = 20.0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    trace = np.asarray(trace, dtype=float).reshape(-1)
    starts = _event_starts(plateaus)
    ends = _event_ends(plateaus)
    if trace.size == 0 or starts.size == 0 or ends.size == 0 or not np.isfinite(float(frame_rate)) or float(frame_rate) <= 0:
        return []
    pre_frames = max(0, int(np.ceil(float(pre_ms) / 1000.0 * float(frame_rate))))
    post_frames = max(0, int(np.ceil(float(post_ms) / 1000.0 * float(frame_rate))))
    segments: list[tuple[np.ndarray, np.ndarray]] = []
    for start, end in zip(starts, ends):
        s = int(start)
        e = int(end)
        if e < s:
            continue
        seg_start = max(0, s - pre_frames)
        seg_end = min(trace.size - 1, e + post_frames)
        if seg_end < seg_start:
            continue
        seg = np.asarray(trace[seg_start:seg_end + 1], dtype=float)
        if seg.size == 0 or not np.any(np.isfinite(seg)):
            continue
        frame_idx = np.arange(seg_start, seg_end + 1, dtype=float)
        x_ms = (frame_idx - float(s)) / float(frame_rate) * 1000.0
        segments.append((x_ms, seg))
    return segments


def _stack_to_line(time_ms: np.ndarray, stack: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if stack.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    x_rows = np.tile(np.append(np.asarray(time_ms, dtype=float), np.nan), (stack.shape[0], 1))
    y_rows = np.concatenate([np.asarray(stack, dtype=float), np.full((stack.shape[0], 1), np.nan)], axis=1)
    return x_rows.reshape(-1), y_rows.reshape(-1)


def _nanmean_or_nan(stack: np.ndarray) -> np.ndarray:
    if stack.size == 0:
        return np.array([], dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.nanmean(stack, axis=0)


def _add_waveform_panel(
    fig: go.Figure,
    *,
    row: int,
    col: int,
    time_ms: np.ndarray,
    trace_stack: np.ndarray,
    trace_name: str,
    trace_color: str,
    trace_overlay_color: str,
    vm_stack: np.ndarray | None = None,
    min_mean_snippets: int = 1,
    showlegend: bool = True,
) -> None:
    fig.add_hline(
        y=0.0,
        line={"color": "rgba(120, 120, 120, 0.55)", "width": 1, "dash": "dash"},
        row=row,
        col=col,
    )
    if trace_stack.size == 0:
        fig.add_annotation(
            text=f"No {trace_name} snippets",
            x=0.5,
            y=0.5,
            showarrow=False,
            font={"size": 12, "color": "#777"},
            row=row,
            col=col,
        )
        return

    x_individual, y_individual = _stack_to_line(time_ms, trace_stack)
    fig.add_trace(
        go.Scattergl(
            x=x_individual,
            y=y_individual,
            mode="lines",
            line={"color": trace_overlay_color, "width": 0.7},
            name=f"{trace_name} snippets",
            showlegend=showlegend,
            hoverinfo="skip",
        ),
        row=row,
        col=col,
    )
    if trace_stack.shape[0] >= int(min_mean_snippets):
        mean_trace = _nanmean_or_nan(trace_stack)
        fig.add_trace(
            go.Scatter(
                x=time_ms,
                y=mean_trace,
                mode="lines",
                line={"color": trace_color, "width": 2.2},
                name=f"{trace_name} mean",
                showlegend=showlegend,
                hovertemplate="time=%{x:.2f} ms<br>mean=%{y:.4g}<extra></extra>",
            ),
            row=row,
            col=col,
        )
    if vm_stack is not None and vm_stack.size:
        mean_vm = _nanmean_or_nan(vm_stack)
        fig.add_trace(
            go.Scatter(
                x=time_ms,
                y=mean_vm,
                mode="lines",
                line={"color": VM_COLOR, "width": 1.8},
                name=f"{trace_name} Vm mean",
                showlegend=showlegend,
                hovertemplate="time=%{x:.2f} ms<br>Vm mean=%{y:.4g}<extra></extra>",
            ),
            row=row,
            col=col,
        )


def build_waveform_shape_figure(
    *,
    spike_trace: np.ndarray,
    spike_vm_trace: np.ndarray,
    plateau_trace: np.ndarray | None,
    plateau_vm_trace: np.ndarray | None,
    simple_spikes: np.ndarray,
    complex_spikes: np.ndarray | None = None,
    complex_bursts: dict[str, Any] | None = None,
    plateaus: dict[str, Any] | None = None,
    frame_rate: float,
    animal_id: str,
    cell_idx: int,
) -> tuple[go.Figure, dict[str, Any]]:
    """Build full-resolution waveform summaries for current live detections."""
    spike_trace = np.asarray(spike_trace, dtype=float).reshape(-1)
    spike_vm_trace = np.asarray(spike_vm_trace, dtype=float).reshape(-1)
    if spike_vm_trace.shape != spike_trace.shape:
        spike_vm_trace = np.full(spike_trace.shape, np.nan, dtype=float)
    plateau_trace_arr = (
        np.asarray(plateau_trace, dtype=float).reshape(-1)
        if plateau_trace is not None
        else np.full(spike_trace.shape, np.nan, dtype=float)
    )
    plateau_vm_arr = (
        np.asarray(plateau_vm_trace, dtype=float).reshape(-1)
        if plateau_vm_trace is not None
        else np.full(plateau_trace_arr.shape, np.nan, dtype=float)
    )
    if plateau_vm_arr.shape != plateau_trace_arr.shape:
        plateau_vm_arr = np.full(plateau_trace_arr.shape, np.nan, dtype=float)

    simple_spikes_arr = np.asarray(simple_spikes if simple_spikes is not None else [], dtype=np.int64).reshape(-1)
    simple_spikes_arr = np.sort(np.unique(simple_spikes_arr[(simple_spikes_arr >= 0) & (simple_spikes_arr < spike_trace.size)]))
    simple_spikes_arr = _filter_min_separation(simple_spikes_arr, frame_rate, 14.0)
    cb_events = _first_spike_per_burst(complex_bursts, complex_spikes, spike_trace.size)
    plateau_starts = _event_starts(plateaus)

    ss_t, ss_stack = _collect_local_normalized_waveforms(
        spike_trace,
        simple_spikes_arr,
        frame_rate,
        pre_ms=10.0,
        post_ms=10.0,
    )
    cb_t, cb_stack = _collect_local_normalized_waveforms(
        spike_trace,
        cb_events,
        frame_rate,
        pre_ms=20.0,
        post_ms=80.0,
    )
    plateau_segments = _collect_plateau_segments(
        plateau_trace_arr,
        plateaus,
        frame_rate,
    )

    fig = make_subplots(
        rows=1,
        cols=3,
        shared_xaxes=False,
        horizontal_spacing=0.075,
        subplot_titles=(
            "Simple spike waveforms",
            "Complex burst waveforms",
            "Plateau waveforms",
        ),
    )
    _add_waveform_panel(
        fig,
        row=1,
        col=1,
        time_ms=ss_t,
        trace_stack=ss_stack,
        trace_name="SS",
        trace_color=SIMPLE_SPIKE_COLOR,
        trace_overlay_color="rgba(2, 108, 128, 0.16)",
        min_mean_snippets=5,
    )
    _add_waveform_panel(
        fig,
        row=1,
        col=2,
        time_ms=cb_t,
        trace_stack=cb_stack,
        trace_name="CB",
        trace_color=COMPLEX_SPIKE_COLOR,
        trace_overlay_color="rgba(238, 155, 0, 0.14)",
        min_mean_snippets=3,
    )
    fig.add_hline(
        y=0.0,
        line={"color": "rgba(120, 120, 120, 0.55)", "width": 1, "dash": "dash"},
        row=1,
        col=3,
    )
    if plateau_segments:
        for idx, (x_ms, seg) in enumerate(plateau_segments):
            fig.add_trace(
                go.Scattergl(
                    x=x_ms,
                    y=seg,
                    mode="lines",
                    line={"color": "rgba(214, 40, 40, 0.50)", "width": 0.9},
                    name="Plateau traces" if idx == 0 else None,
                    showlegend=idx == 0,
                    hovertemplate="time=%{x:.2f} ms<br>value=%{y:.4g}<extra></extra>",
                ),
                row=1,
                col=3,
            )
    else:
        fig.add_annotation(
            text="No plateau snippets",
            x=0.5,
            y=0.5,
            showarrow=False,
            font={"size": 12, "color": "#777"},
            row=1,
            col=3,
        )

    for col, title in (
        (1, "Time from spike peak (ms)"),
        (2, "Time from spike peak (ms)"),
        (3, "Time from plateau onset (ms)"),
    ):
        fig.update_xaxes(
            title_text=title,
            showgrid=False,
            zeroline=False,
            row=1,
            col=col,
        )
        fig.update_yaxes(
            title_text="Spike-height normalized amplitude",
            showgrid=False,
            zeroline=False,
            row=1,
            col=col,
        )
    fig.update_xaxes(range=[-10.0, 10.0], row=1, col=1)
    fig.update_xaxes(range=[-20.0, 80.0], row=1, col=2)
    fig.update_xaxes(range=[0.0, 250.0], row=1, col=3)

    stats = {
        "simple_spikes": int(simple_spikes_arr.size),
        "simple_spike_snippets": int(ss_stack.shape[0]),
        "complex_bursts": int(cb_events.size),
        "complex_burst_snippets": int(cb_stack.shape[0]),
        "plateaus": int(plateau_starts.size),
        "plateau_snippets": int(len(plateau_segments)),
        "ss_window_ms": [-10.0, 10.0],
        "cb_window_ms": [-20.0, 80.0],
        "plateau_window_ms": [0.0, 250.0],
        "plateau_extracted_window_ms": [-40.0, 20.0],
    }
    fig.update_layout(
        template="plotly_white",
        title={
            "text": f"{animal_id}: cell {int(cell_idx) + 1}, waveform shapes",
            "font": {"size": 14},
        },
        height=430,
        margin={"l": 72, "r": 22, "t": 60, "b": 42},
        paper_bgcolor="white",
        plot_bgcolor="white",
        hovermode="closest",
        legend={"orientation": "h", "x": 0, "y": 1.04, "font": {"size": 10}},
        dragmode="zoom",
        uirevision=f"waveform-shapes-{animal_id}-cell-{int(cell_idx)}",
    )
    return fig, stats


def _clean_row_height_px(row_height_px: float | int | None) -> int:
    try:
        value = float(row_height_px)
    except Exception:
        value = float(DEFAULT_ROW_HEIGHT_PX)
    if not np.isfinite(value):
        value = float(DEFAULT_ROW_HEIGHT_PX)
    value = min(MAX_ROW_HEIGHT_PX, max(MIN_ROW_HEIGHT_PX, value))
    return int(round(value))


def _clean_highpass_gain(highpass_gain: float | int | None) -> float:
    try:
        value = float(highpass_gain)
    except Exception:
        value = float(DEFAULT_HIGHPASS_GAIN)
    if not np.isfinite(value):
        value = float(DEFAULT_HIGHPASS_GAIN)
    return float(min(MAX_HIGHPASS_GAIN, max(MIN_HIGHPASS_GAIN, value)))


def _robust_zero_scale(*arrays: np.ndarray) -> float:
    finite_chunks = []
    for arr in arrays:
        vals = np.asarray(arr, dtype=float).reshape(-1)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            finite_chunks.append(vals)
    if not finite_chunks:
        return 1.0
    vals = np.concatenate(finite_chunks)
    scale = float(np.nanpercentile(np.abs(vals), 99))
    if (not np.isfinite(scale)) or scale <= 1e-12:
        scale = float(np.nanstd(vals) * 3.0)
    if (not np.isfinite(scale)) or scale <= 1e-12:
        scale = 1.0
    return scale


def _scaled(values: np.ndarray | float, scale: float) -> np.ndarray | float:
    out = (np.asarray(values, dtype=float) / float(scale)) * TRACE_SCALE
    if np.ndim(values) == 0:
        return float(out)
    return out


def _finite_min_max(*arrays: np.ndarray) -> tuple[float, float]:
    lo = np.inf
    hi = -np.inf
    for arr in arrays:
        vals = np.asarray(arr, dtype=float).reshape(-1)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        lo = min(lo, float(np.min(vals)))
        hi = max(hi, float(np.max(vals)))
    if not np.isfinite(lo) or not np.isfinite(hi):
        half = TRACE_SCALE / 2.0
        return -half, half
    return lo, hi


def _row_geometry(*arrays: np.ndarray) -> tuple[float, float, float, float]:
    trace_min, trace_max = _finite_min_max(*arrays)
    raster_y0 = trace_max + RASTER_PAD
    raster_y1 = raster_y0 + RASTER_HEIGHT
    row_bottom = trace_min - TRACE_PAD
    row_top = raster_y1 + TRACE_PAD
    return row_bottom, row_top, raster_y0, raster_y1


def _stack_offsets(bounds: list[tuple[float, float]]) -> tuple[np.ndarray, float, float]:
    if not bounds:
        return np.array([], dtype=float), -1.0, 1.0
    offsets = np.zeros(len(bounds), dtype=float)
    cursor = 0.0
    for idx in range(len(bounds) - 1, -1, -1):
        row_bottom, row_top = bounds[idx]
        offsets[idx] = cursor - float(row_bottom)
        cursor += (float(row_top) - float(row_bottom)) + ROW_GAP
    bottoms = [float(offset) + float(row_bottom) for offset, (row_bottom, _row_top) in zip(offsets, bounds)]
    tops = [float(offset) + float(row_top) for offset, (_row_bottom, row_top) in zip(offsets, bounds)]
    return offsets, min(bottoms) - PLOT_PAD, max(tops) + PLOT_PAD


def _add_trace_rows(
    fig: go.Figure,
    row_items: list[dict[str, Any]],
    frame_rate: float,
    *,
    y_key: str,
    raw_key: str,
    color: str,
    name: str,
    width: float = 0.7,
    dash: str | None = None,
    use_webgl: bool = True,
    simplify: bool = True,
) -> list[int]:
    trace_indices: list[int] = []
    for row_idx, item in enumerate(row_items):
        y = np.asarray(item[y_key], dtype=float) + float(item["offset"])
        if y.size == 0:
            continue
        start = int(item["start"])
        end = int(item["end"])
        absolute_time = np.arange(start, end, dtype=float) / float(frame_rate)
        raw_values = np.asarray(item[raw_key], dtype=float)
        customdata = np.column_stack([absolute_time, raw_values])
        segment_idx = int(item.get("row_idx", row_idx))
        segment_label = _segment_label(start, end, frame_rate)
        line = {"color": color, "width": float(width)}
        if dash is not None:
            line["dash"] = dash
        if not use_webgl and not simplify:
            line["simplify"] = False
        trace_indices.append(len(fig.data))
        trace_kwargs = dict(
            y=y,
            mode="lines",
            line=line,
            name=name if row_idx == 0 else None,
            showlegend=row_idx == 0,
            customdata=customdata,
            hovertemplate=(
                f"{name} segment {segment_idx} ({segment_label})"
                "<br>absolute time=%{customdata[0]:.3f} s"
                "<br>time in segment=%{x:.3f} s"
                "<br>value=%{customdata[1]:.4g}<extra></extra>"
            ),
        )
        if use_webgl:
            fig.add_trace(go.Scattergl(x0=0, dx=1.0 / float(frame_rate), **trace_kwargs))
        else:
            x = np.arange(y.size, dtype=float) / float(frame_rate)
            fig.add_trace(go.Scatter(x=x, **trace_kwargs))
    return trace_indices


def _spike_marker_arrays(
    spikes: np.ndarray,
    row_items: list[dict[str, Any]],
    frame_rate: float,
) -> tuple[list[float], list[float]]:
    spikes = np.asarray(spikes, dtype=np.int64).reshape(-1)
    xs: list[float] = []
    ys: list[float] = []
    for item in row_items:
        start = int(item["start"])
        end = int(item["end"])
        mask = (spikes >= int(start)) & (spikes < int(end))
        if not np.any(mask):
            continue
        local = (spikes[mask] - int(start)).astype(float) / float(frame_rate)
        y0 = float(item["offset"]) + float(item["raster_y0"])
        y1 = float(item["offset"]) + float(item["raster_y1"])
        for x in local:
            xs.extend([float(x), float(x), np.nan])
            ys.extend([y0, y1, np.nan])
    return xs, ys


def _complex_burst_windows(complex_bursts: dict[str, Any] | None) -> list[tuple[int, int]]:
    if not isinstance(complex_bursts, dict):
        return []
    try:
        starts = np.asarray(complex_bursts.get("starts", []), dtype=np.int64).reshape(-1)
        ends = np.asarray(complex_bursts.get("ends", []), dtype=np.int64).reshape(-1)
    except Exception:
        return []
    n = min(starts.size, ends.size)
    windows: list[tuple[int, int]] = []
    for start, end in zip(starts[:n], ends[:n]):
        s = int(min(start, end))
        e = int(max(start, end))
        windows.append((s, e))
    return windows


def _event_row_indices(events: dict[str, Any] | None, row_items: list[dict[str, Any]]) -> set[int]:
    rows: set[int] = set()
    windows = _complex_burst_windows(events)
    if not windows:
        return rows
    for event_start, event_end in windows:
        event_end_inclusive = int(event_end)
        for item in row_items:
            row_start = int(item["start"])
            row_end = int(item["end"])
            if max(int(event_start), row_start) <= min(event_end_inclusive, row_end - 1):
                rows.add(int(item["row_idx"]))
    return rows


def _segment_tick_label(
    item: dict[str, Any],
    frame_rate: float,
    *,
    segment_snr: float | None = None,
    highlight: bool = False,
) -> str:
    label = f"<b>S{int(item['row_idx'])}</b><br>{_segment_label(int(item['start']), int(item['end']), frame_rate)}"
    if segment_snr is not None:
        label += f"<br>SNR={_format_snr(segment_snr)}"
    if highlight:
        return f"<span style='color:{PLATEAU_COLOR}'>{label}</span>"
    return label


def _cb_shapes_for_rows(
    complex_bursts: dict[str, Any] | None,
    row_items: list[dict[str, Any]],
    frame_rate: float,
    *,
    fillcolor: str = CB_FILL,
) -> list[dict[str, Any]]:
    shapes: list[dict[str, Any]] = []
    windows = _complex_burst_windows(complex_bursts)
    if not windows:
        return shapes
    for cb_start, cb_end in windows:
        for item in row_items:
            row_start = int(item["start"])
            row_end = int(item["end"])
            overlap_start = max(int(cb_start), row_start)
            overlap_end = min(int(cb_end) + 1, row_end)
            if overlap_end <= overlap_start:
                continue
            x0 = (overlap_start - row_start) / float(frame_rate)
            x1 = (overlap_end - row_start) / float(frame_rate)
            y0 = float(item["offset"]) + float(item["row_bottom"])
            y1 = float(item["offset"]) + float(item["raster_y0"]) - 0.03
            if y1 <= y0:
                y1 = float(item["offset"]) + float(item["row_top"])
            shapes.append(
                {
                    "type": "rect",
                    "xref": "x",
                    "yref": "y",
                    "x0": float(x0),
                    "x1": float(x1),
                    "y0": y0,
                    "y1": y1,
                    "fillcolor": fillcolor,
                    "line": {"width": 0},
                    "layer": "below",
                    "editable": False,
                }
            )
    return shapes


def _manual_exclusion_shapes_for_rows(
    periods: list[dict[str, Any]] | None,
    row_items: list[dict[str, Any]],
    frame_rate: float,
) -> list[dict[str, Any]]:
    shapes: list[dict[str, Any]] = []
    if not isinstance(periods, list):
        return shapes
    for period in periods:
        if not isinstance(period, dict):
            continue
        try:
            period_start = int(period.get("start_frame", period.get("start")))
            period_end = int(period.get("end_frame", period.get("end")))
        except Exception:
            continue
        if period_end <= period_start:
            continue
        for item in row_items:
            row_start = int(item["start"])
            row_end = int(item["end"])
            overlap_start = max(period_start, row_start)
            overlap_end = min(period_end, row_end)
            if overlap_end <= overlap_start:
                continue
            x0 = (overlap_start - row_start) / float(frame_rate)
            x1 = (overlap_end - row_start) / float(frame_rate)
            shapes.append(
                {
                    "type": "rect",
                    "xref": "x",
                    "yref": "y",
                    "x0": float(x0),
                    "x1": float(x1),
                    "y0": float(item["offset"]) + float(item["row_bottom"]),
                    "y1": float(item["offset"]) + float(item["row_top"]),
                    "fillcolor": MANUAL_EXCLUSION_FILL,
                    "line": {"width": 0},
                    "layer": "below",
                    "editable": False,
                }
            )
    return shapes


def _zero_line_shapes_for_rows(row_items: list[dict[str, Any]], frame_rate: float) -> list[dict[str, Any]]:
    shapes: list[dict[str, Any]] = []
    for item in row_items:
        x1 = (int(item["end"]) - int(item["start"])) / float(frame_rate)
        y = float(item["offset"])
        shapes.append(
            {
                "type": "line",
                "xref": "x",
                "yref": "y",
                "x0": 0.0,
                "x1": float(x1),
                "y0": y,
                "y1": y,
                "line": {"color": ZERO_LINE_COLOR, "width": 0.7, "dash": "dash"},
                "layer": "above",
                "editable": False,
            }
        )
    return shapes


def _segment_label(start_frame: int, end_frame: int, frame_rate: float) -> str:
    return f"{start_frame / float(frame_rate):.0f}-{end_frame / float(frame_rate):.0f}s"


def _format_snr(value: float | int | None) -> str:
    try:
        snr = float(value)
    except Exception:
        return "n/a"
    if not np.isfinite(snr):
        return "n/a"
    if abs(snr) >= 100:
        return f"{snr:.0f}"
    if abs(snr) >= 10:
        return f"{snr:.1f}"
    return f"{snr:.2f}"


def _clean_snr_until_s(snr_acceptable_until_s: float | int | None, max_time_s: float) -> float:
    max_time_s = max(0.0, float(max_time_s))
    try:
        value = float(snr_acceptable_until_s)
    except Exception:
        value = max_time_s
    if not np.isfinite(value):
        value = max_time_s
    return float(min(max_time_s, max(0.0, value)))


def _clean_x_range(x_range: list[float] | tuple[float, float] | None, max_x: float) -> list[float] | None:
    if x_range is None:
        return None
    try:
        lo = float(x_range[0])
        hi = float(x_range[1])
    except Exception:
        return None
    if not np.isfinite(lo) or not np.isfinite(hi):
        return None
    lo = max(0.0, min(float(max_x), lo))
    hi = max(0.0, min(float(max_x), hi))
    if hi <= lo:
        return None
    return [lo, hi]


def build_detection_figures(
    *,
    trace: np.ndarray,
    trace_hp: np.ndarray,
    right_trace: np.ndarray | None = None,
    right_vm_trace: np.ndarray | None = None,
    right_trace_name: str = "Trace",
    right_vm_name: str = "Vm",
    segment_bounds: list[tuple[int, int]],
    thresholds: np.ndarray,
    second_round_ss_thresholds: np.ndarray | None = None,
    second_round_cs_thresholds: np.ndarray | None = None,
    spikes: np.ndarray,
    simple_spikes: np.ndarray | None = None,
    complex_spikes: np.ndarray | None = None,
    complex_bursts: dict[str, Any] | None = None,
    failed_min_spikes_windows: dict[str, Any] | None = None,
    plateaus: dict[str, Any] | None = None,
    manual_exclusion_periods: list[dict[str, Any]] | None = None,
    frame_rate: float,
    animal_id: str,
    cell_idx: int,
    row_height_px: float | int | None = DEFAULT_ROW_HEIGHT_PX,
    highpass_gain: float | int | None = DEFAULT_HIGHPASS_GAIN,
    snr_acceptable_until_s: float | int | None = None,
    segment_snr: np.ndarray | None = None,
    overall_snr: float | int | None = None,
    x_range: list[float] | tuple[float, float] | None = None,
) -> tuple[go.Figure, go.Figure, dict[str, Any]]:
    if len(segment_bounds) == 0:
        fig = empty_figure("No trace samples found.")
        return fig, fig, {"row_offsets": []}

    trace = np.asarray(trace, dtype=float)
    trace_hp = np.asarray(trace_hp, dtype=float)
    right_trace_arr = trace if right_trace is None else np.asarray(right_trace, dtype=float)
    if right_trace_arr.shape != trace.shape:
        right_trace_arr = trace
    right_vm_arr = None if right_vm_trace is None else np.asarray(right_vm_trace, dtype=float)
    if right_vm_arr is not None and right_vm_arr.shape != trace.shape:
        right_vm_arr = None
    thresholds = np.asarray(thresholds, dtype=float)
    second_round_ss_thresholds_arr = np.asarray(
        second_round_ss_thresholds if second_round_ss_thresholds is not None else [],
        dtype=float,
    ).reshape(-1)
    second_round_cs_thresholds_arr = np.asarray(
        second_round_cs_thresholds if second_round_cs_thresholds is not None else [],
        dtype=float,
    ).reshape(-1)
    segment_snr_arr = np.asarray(segment_snr if segment_snr is not None else [], dtype=float).reshape(-1)
    try:
        overall_snr_value = float(overall_snr)
    except Exception:
        overall_snr_value = float("nan")
    right_trace_scale_raw = _robust_zero_scale(
        right_trace_arr,
        right_vm_arr if right_vm_arr is not None else np.array([]),
    )
    hp_scale_raw = _robust_zero_scale(trace_hp)
    hp_gain = _clean_highpass_gain(highpass_gain)
    row_items: list[dict[str, Any]] = []
    row_bounds: list[tuple[float, float]] = []
    for row_idx, (start, end) in enumerate(segment_bounds):
        start = int(start)
        end = int(end)
        if end <= start:
            continue
        trace_y = np.asarray(_scaled(right_trace_arr[start:end], right_trace_scale_raw), dtype=float)
        vm_y = (
            np.asarray(_scaled(right_vm_arr[start:end], right_trace_scale_raw), dtype=float)
            if right_vm_arr is not None
            else np.array([], dtype=float)
        )
        hp_y = np.asarray(_scaled(trace_hp[start:end], hp_scale_raw), dtype=float) * hp_gain
        # Keep row packing anchored to the right-panel signal.
        # High-pass gain is display-only and should not rescale row geometry.
        row_bottom, row_top, raster_y0, raster_y1 = _row_geometry(trace_y, vm_y)
        row_items.append(
            {
                "row_idx": int(row_idx),
                "start": start,
                "end": end,
                "trace_y": trace_y,
                "vm_y": vm_y,
                "hp_y": hp_y,
                "trace_raw": right_trace_arr[start:end],
                "vm_raw": right_vm_arr[start:end] if right_vm_arr is not None else np.array([], dtype=float),
                "hp_raw": trace_hp[start:end],
                "row_bottom": row_bottom,
                "row_top": row_top,
                "raster_y0": raster_y0,
                "raster_y1": raster_y1,
            }
        )
        row_bounds.append((row_bottom, row_top))

    offsets, plot_y0, plot_y1 = _stack_offsets(row_bounds)
    for item, offset in zip(row_items, offsets):
        item["offset"] = float(offset)
    row_height = _clean_row_height_px(row_height_px)
    height = max(520, int(len(row_items) * row_height + 90))
    tickvals = [float(item["offset"]) for item in row_items]
    plateau_rows = _event_row_indices(plateaus, row_items)
    ticktext = [
        _segment_tick_label(
            item,
            frame_rate,
            highlight=int(item["row_idx"]) in plateau_rows,
        )
        for item in row_items
    ]
    right_ticktext = [
        _segment_tick_label(
            item,
            frame_rate,
            segment_snr=(
                segment_snr_arr[int(item["row_idx"])]
                if int(item["row_idx"]) < segment_snr_arr.size
                else None
            ),
            highlight=int(item["row_idx"]) in plateau_rows,
        )
        for item in row_items
    ]
    max_x = max((int(end) - int(start)) / float(frame_rate) for start, end in segment_bounds)
    max_time_s = max((int(end) for _start, end in segment_bounds), default=0) / float(frame_rate)
    snr_until_s = _clean_snr_until_s(snr_acceptable_until_s, max_time_s)
    xaxis_range = _clean_x_range(x_range, max_x) or [0.0, float(max_x)]
    x_revision = f"{float(xaxis_range[0]):.9g}-{float(xaxis_range[1]):.9g}"

    left = go.Figure()
    right = go.Figure()
    left_signal_trace_indices = _add_trace_rows(
        left,
        row_items,
        frame_rate,
        y_key="hp_y",
        raw_key="hp_raw",
        color=HP_COLOR,
        name="High-pass",
    )
    right_signal_trace_indices = _add_trace_rows(
        right,
        row_items,
        frame_rate,
        y_key="trace_y",
        raw_key="trace_raw",
        color=TRACE_COLOR,
        name=str(right_trace_name),
    )
    right_vm_trace_indices = (
        _add_trace_rows(
            right,
            row_items,
            frame_rate,
            y_key="vm_y",
            raw_key="vm_raw",
            color=VM_COLOR,
            name=str(right_vm_name),
            width=1.1,
        )
        if right_vm_arr is not None
        else []
    )
    left_vm_placeholder_indices: list[int] = []
    for _idx in right_vm_trace_indices:
        left_vm_placeholder_indices.append(len(left.data))
        left.add_trace(go.Scattergl(x=[], y=[], visible=False, showlegend=False, hoverinfo="skip"))

    if simple_spikes is None and complex_spikes is None:
        simple_spikes = spikes
        complex_spikes = np.array([], dtype=np.int64)
    simple_spike_x, simple_spike_y = _spike_marker_arrays(
        np.asarray(simple_spikes if simple_spikes is not None else [], dtype=np.int64),
        row_items,
        frame_rate,
    )
    complex_spike_x, complex_spike_y = _spike_marker_arrays(
        np.asarray(complex_spikes if complex_spikes is not None else [], dtype=np.int64),
        row_items,
        frame_rate,
    )
    simple_spike_trace_index = len(left.data)
    for fig in (left, right):
        fig.add_trace(
            go.Scattergl(
                x=simple_spike_x,
                y=simple_spike_y,
                mode="lines",
                line={"color": SIMPLE_SPIKE_COLOR, "width": 1},
                name="Simple spikes",
                showlegend=True,
                hoverinfo="skip",
            )
        )
    complex_spike_trace_index = len(left.data)
    for fig in (left, right):
        fig.add_trace(
            go.Scattergl(
                x=complex_spike_x,
                y=complex_spike_y,
                mode="lines",
                line={"color": COMPLEX_SPIKE_COLOR, "width": 1.2},
                name="Complex spikes",
                showlegend=True,
                hoverinfo="skip",
            )
        )

    mask_shapes: list[dict[str, Any]] = []
    cutoff_frame = int(np.floor(snr_until_s * float(frame_rate)))
    if cutoff_frame < max((int(end) for _start, end in segment_bounds), default=0):
        for item in row_items:
            start = int(item["start"])
            end = int(item["end"])
            if cutoff_frame >= end:
                continue
            x0 = max(0.0, (max(cutoff_frame, start) - start) / float(frame_rate))
            x1 = (end - start) / float(frame_rate)
            if x1 <= x0:
                continue
            mask_shapes.append(
                {
                    "type": "rect",
                    "xref": "x",
                    "yref": "y",
                    "x0": float(x0),
                    "x1": float(x1),
                    "y0": float(item["offset"]) + float(item["row_bottom"]),
                    "y1": float(item["offset"]) + float(item["row_top"]),
                    "fillcolor": SNR_MASK_FILL,
                    "line": {"width": 0},
                    "layer": "below",
                    "editable": False,
                }
            )

    threshold_shapes = []
    second_round_threshold_shapes = []
    shape_to_segment: list[int] = []
    shape_x_locks: list[list[float]] = []
    for item in row_items:
        row_idx = int(item["row_idx"])
        if row_idx >= thresholds.size or not np.isfinite(thresholds[row_idx]):
            continue
        y = float(item["offset"]) + float(_scaled(float(thresholds[row_idx]), hp_scale_raw)) * hp_gain
        x1 = (int(item["end"]) - int(item["start"])) / float(frame_rate)
        threshold_shapes.append(
            {
                "type": "line",
                "xref": "x",
                "yref": "y",
                "x0": 0,
                "x1": x1,
                "y0": y,
                "y1": y,
                "line": {"color": THRESHOLD_COLOR, "width": 1.3},
                "editable": True,
            }
        )
        shape_to_segment.append(int(row_idx))
        shape_x_locks.append([0.0, float(x1)])
        for threshold_arr, color, dash in (
            (second_round_ss_thresholds_arr, SIMPLE_SPIKE_COLOR, "dot"),
            (second_round_cs_thresholds_arr, COMPLEX_SPIKE_COLOR, "dash"),
        ):
            if row_idx >= threshold_arr.size or not np.isfinite(threshold_arr[row_idx]):
                continue
            y_second = float(item["offset"]) + float(_scaled(float(threshold_arr[row_idx]), hp_scale_raw)) * hp_gain
            second_round_threshold_shapes.append(
                {
                    "type": "line",
                    "xref": "x",
                    "yref": "y",
                    "x0": 0,
                    "x1": x1,
                    "y0": y_second,
                    "y1": y_second,
                    "line": {"color": color, "width": 1.4, "dash": dash},
                    "editable": False,
                }
            )
    failed_cb_shapes = _cb_shapes_for_rows(
        failed_min_spikes_windows,
        row_items,
        frame_rate,
        fillcolor=FAILED_CB_FILL,
    )
    cb_shapes = _cb_shapes_for_rows(complex_bursts, row_items, frame_rate)
    plateau_shapes = _cb_shapes_for_rows(plateaus, row_items, frame_rate, fillcolor=PLATEAU_FILL)
    manual_exclusion_shapes = _manual_exclusion_shapes_for_rows(manual_exclusion_periods, row_items, frame_rate)
    zero_shapes = _zero_line_shapes_for_rows(row_items, frame_rate)
    left_shapes = threshold_shapes + second_round_threshold_shapes + failed_cb_shapes + cb_shapes + mask_shapes + manual_exclusion_shapes + zero_shapes + plateau_shapes
    right_shapes = failed_cb_shapes + cb_shapes + mask_shapes + manual_exclusion_shapes + zero_shapes + plateau_shapes
    left.update_layout(shapes=left_shapes)
    right.update_layout(shapes=right_shapes)
    shape_to_segment.extend([-1 for _shape in second_round_threshold_shapes])
    shape_to_segment.extend([-1 for _shape in failed_cb_shapes])
    shape_to_segment.extend([-1 for _shape in cb_shapes])
    shape_to_segment.extend([-1 for _shape in mask_shapes])
    shape_to_segment.extend([-1 for _shape in manual_exclusion_shapes])
    shape_to_segment.extend([-1 for _shape in zero_shapes])
    shape_to_segment.extend([-1 for _shape in plateau_shapes])

    common_layout = {
        "template": "plotly_white",
        "height": height,
        "margin": {"l": 66, "r": 14, "t": 32, "b": 38},
        "paper_bgcolor": "white",
        "plot_bgcolor": "white",
        "xaxis": {
            "title": "Time within row (s)",
            "range": xaxis_range,
            "autorange": False,
            "minallowed": 0,
            "maxallowed": max_x,
            "showgrid": False,
            "gridcolor": GRID_COLOR,
            "zeroline": False,
        },
        "yaxis": {
            "title": "Absolute time row",
            "range": [float(plot_y0), float(plot_y1)],
            "autorange": False,
            "minallowed": float(plot_y0),
            "maxallowed": float(plot_y1),
            "tickmode": "array",
            "tickvals": tickvals,
            "ticktext": ticktext,
            "fixedrange": True,
            "showgrid": False,
            "zeroline": False,
        },
        "dragmode": "zoom",
        "hovermode": "closest",
        "legend": {"orientation": "h", "x": 0, "y": 1.08, "font": {"size": 10}},
        "uirevision": f"{animal_id}-cell-{int(cell_idx)}-{len(segment_bounds)}-{row_height}-{x_revision}",
        "meta": {"threshold_shape_x": shape_x_locks},
    }
    left.update_layout(title={"text": "High-pass", "font": {"size": 13}}, **common_layout)
    right.update_layout(title={"text": str(right_trace_name), "font": {"size": 13}}, **common_layout)
    right.update_yaxes(ticktext=right_ticktext)
    if np.isfinite(overall_snr_value):
        right.add_annotation(
            text=f"overall SNR={_format_snr(overall_snr_value)}",
            xref="paper",
            yref="paper",
            x=1.0,
            y=1.08,
            xanchor="right",
            yanchor="bottom",
            showarrow=False,
            font={"size": 12, "color": OVERALL_SNR_COLOR},
        )

    meta = {
        "row_offsets": [float(v) for v in offsets],
        "threshold_center": 0.0,
        "threshold_scale": float(hp_scale_raw),
        "right_trace_scale": float(right_trace_scale_raw),
        "trace_scale": float(TRACE_SCALE * hp_gain),
        "highpass_gain": float(hp_gain),
        "plot_y0": float(plot_y0),
        "plot_y1": float(plot_y1),
        "height": int(height),
        "snr_acceptable_until_s": float(snr_until_s),
        "segment_snr": [
            float(v) if np.isfinite(v) else None
            for v in segment_snr_arr.reshape(-1)
        ],
        "overall_snr": float(overall_snr_value) if np.isfinite(overall_snr_value) else None,
        "max_time_s": float(max_time_s),
        "x_range": xaxis_range,
        "max_x": float(max_x),
        "left_signal_trace_indices": [int(v) for v in left_signal_trace_indices],
        "right_signal_trace_indices": [int(v) for v in right_signal_trace_indices],
        "left_vm_placeholder_indices": [int(v) for v in left_vm_placeholder_indices],
        "right_vm_trace_indices": [int(v) for v in right_vm_trace_indices],
        "simple_spike_trace_index": int(simple_spike_trace_index),
        "complex_spike_trace_index": int(complex_spike_trace_index),
        "spike_trace_index": int(simple_spike_trace_index),
        "shape_to_segment": shape_to_segment,
    }
    return left, right, meta


def build_saved_review_figures(
    *,
    trace: np.ndarray,
    vm_trace: np.ndarray,
    segment_bounds: list[tuple[int, int]],
    simple_spikes: np.ndarray | None = None,
    complex_spikes: np.ndarray | None = None,
    complex_bursts: dict[str, Any] | None = None,
    failed_min_spikes_windows: dict[str, Any] | None = None,
    plateaus: dict[str, Any] | None = None,
    manual_exclusion_periods: list[dict[str, Any]] | None = None,
    frame_rate: float,
    animal_id: str,
    cell_idx: int,
    row_height_px: float | int | None = DEFAULT_ROW_HEIGHT_PX,
    snr_acceptable_until_s: float | int | None = None,
    segment_snr: np.ndarray | None = None,
    overall_snr: float | int | None = None,
    max_rows_per_figure: int = 60,
) -> list[go.Figure]:
    """Build read-only saved-cell review figures from saved detections."""
    if len(segment_bounds) == 0:
        return [empty_figure("No saved segment bounds found.")]

    trace = np.asarray(trace, dtype=float).reshape(-1)
    vm_trace = np.asarray(vm_trace, dtype=float).reshape(-1)
    if vm_trace.shape != trace.shape:
        vm_trace = np.full(trace.shape, np.nan, dtype=float)
    if not np.isfinite(float(frame_rate)) or float(frame_rate) <= 0:
        return [empty_figure("Invalid frame rate for saved review.")]

    simple_spikes_arr = np.asarray(simple_spikes if simple_spikes is not None else [], dtype=np.int64).reshape(-1)
    complex_spikes_arr = np.asarray(complex_spikes if complex_spikes is not None else [], dtype=np.int64).reshape(-1)
    segment_snr_arr = np.asarray(segment_snr if segment_snr is not None else [], dtype=float).reshape(-1)
    try:
        overall_snr_value = float(overall_snr) if overall_snr is not None else float("nan")
    except Exception:
        overall_snr_value = float("nan")
    scale = _robust_zero_scale(trace, vm_trace)
    row_height = _clean_row_height_px(row_height_px)
    max_rows_per_figure = max(1, int(max_rows_per_figure))
    max_time_s = max((int(end) for _start, end in segment_bounds), default=0) / float(frame_rate)
    snr_until_s = _clean_snr_until_s(snr_acceptable_until_s, max_time_s)

    figures: list[go.Figure] = []
    for panel_idx, row_start in enumerate(range(0, len(segment_bounds), max_rows_per_figure)):
        chunk_bounds = segment_bounds[row_start : row_start + max_rows_per_figure]
        row_items: list[dict[str, Any]] = []
        row_bounds: list[tuple[float, float]] = []

        for local_idx, (start, end) in enumerate(chunk_bounds):
            row_idx = row_start + local_idx
            start = max(0, int(start))
            end = min(trace.size, int(end))
            if end <= start:
                continue
            trace_y = np.asarray(_scaled(trace[start:end], scale), dtype=float)
            vm_y = np.asarray(_scaled(vm_trace[start:end], scale), dtype=float)
            row_bottom, row_top, raster_y0, raster_y1 = _row_geometry(trace_y, vm_y)
            row_items.append(
                {
                    "row_idx": int(row_idx),
                    "start": start,
                    "end": end,
                    "trace_y": trace_y,
                    "vm_y": vm_y,
                    "trace_raw": trace[start:end],
                    "vm_raw": vm_trace[start:end],
                    "row_bottom": row_bottom,
                    "row_top": row_top,
                    "raster_y0": raster_y0,
                    "raster_y1": raster_y1,
                }
            )
            row_bounds.append((row_bottom, row_top))

        if not row_items:
            figures.append(empty_figure(f"{animal_id}: cell {int(cell_idx) + 1}, no rows in saved review panel {panel_idx + 1}."))
            continue

        offsets, plot_y0, plot_y1 = _stack_offsets(row_bounds)
        for item, offset in zip(row_items, offsets):
            item["offset"] = float(offset)

        fig = go.Figure()
        _add_trace_rows(
            fig,
            row_items,
            frame_rate,
            y_key="trace_y",
            raw_key="trace_raw",
            color=TRACE_COLOR,
            name="Spike-normalized trace",
            use_webgl=False,
            simplify=False,
        )
        _add_trace_rows(
            fig,
            row_items,
            frame_rate,
            y_key="vm_y",
            raw_key="vm_raw",
            color=VM_COLOR,
            name="Vm",
            width=1.1,
            use_webgl=False,
            simplify=False,
        )

        simple_x, simple_y = _spike_marker_arrays(simple_spikes_arr, row_items, frame_rate)
        complex_x, complex_y = _spike_marker_arrays(complex_spikes_arr, row_items, frame_rate)
        fig.add_trace(
            go.Scatter(
                x=simple_x,
                y=simple_y,
                mode="lines",
                line={"color": SIMPLE_SPIKE_COLOR, "width": 1},
                name="Simple spikes",
                showlegend=True,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=complex_x,
                y=complex_y,
                mode="lines",
                line={"color": COMPLEX_SPIKE_COLOR, "width": 1.2},
                name="Complex spikes",
                showlegend=True,
                hoverinfo="skip",
            )
        )

        mask_shapes: list[dict[str, Any]] = []
        cutoff_frame = int(np.floor(snr_until_s * float(frame_rate)))
        max_frame = max((int(end) for _start, end in segment_bounds), default=0)
        if cutoff_frame < max_frame:
            for item in row_items:
                start = int(item["start"])
                end = int(item["end"])
                if cutoff_frame >= end:
                    continue
                x0 = max(0.0, (max(cutoff_frame, start) - start) / float(frame_rate))
                x1 = (end - start) / float(frame_rate)
                if x1 <= x0:
                    continue
                mask_shapes.append(
                    {
                        "type": "rect",
                        "xref": "x",
                        "yref": "y",
                        "x0": float(x0),
                        "x1": float(x1),
                        "y0": float(item["offset"]) + float(item["row_bottom"]),
                        "y1": float(item["offset"]) + float(item["row_top"]),
                        "fillcolor": SNR_MASK_FILL,
                        "line": {"width": 0},
                        "layer": "below",
                        "editable": False,
                    }
                )

        shapes = (
            _cb_shapes_for_rows(failed_min_spikes_windows, row_items, frame_rate, fillcolor=FAILED_CB_FILL)
            + _cb_shapes_for_rows(complex_bursts, row_items, frame_rate, fillcolor=CB_FILL)
            + mask_shapes
            + _manual_exclusion_shapes_for_rows(manual_exclusion_periods, row_items, frame_rate)
            + _zero_line_shapes_for_rows(row_items, frame_rate)
            + _cb_shapes_for_rows(plateaus, row_items, frame_rate, fillcolor=PLATEAU_FILL)
        )
        tickvals = [float(item["offset"]) for item in row_items]
        plateau_rows = _event_row_indices(plateaus, row_items)
        ticktext = []
        for item in row_items:
            ticktext.append(
                _segment_tick_label(
                    item,
                    frame_rate,
                    segment_snr=(
                        segment_snr_arr[int(item["row_idx"])]
                        if segment_snr_arr.size and int(item["row_idx"]) < segment_snr_arr.size
                        else None
                    ),
                    highlight=int(item["row_idx"]) in plateau_rows,
                )
            )
        max_x = max((int(end) - int(start)) / float(frame_rate) for start, end in chunk_bounds)
        height = max(520, int(len(row_items) * row_height + 90))
        display_start_s = row_items[0]["start"] / float(frame_rate)
        display_end_s = row_items[-1]["end"] / float(frame_rate)

        fig.update_layout(
            template="plotly_white",
            title={
                "text": f"{animal_id}: cell {int(cell_idx) + 1}, saved review {display_start_s:.0f}-{display_end_s:.0f}s",
                "font": {"size": 13},
            },
            height=height,
            margin={"l": 74, "r": 18, "t": 38, "b": 38},
            paper_bgcolor="white",
            plot_bgcolor="white",
            shapes=shapes,
            xaxis={
                "title": "Time within row (s)",
                "range": [0.0, float(max_x)],
                "autorange": False,
                "minallowed": 0,
                "maxallowed": float(max_x),
                "showgrid": False,
                "zeroline": False,
            },
            yaxis={
                "title": "Absolute time row",
                "range": [float(plot_y0), float(plot_y1)],
                "autorange": False,
                "minallowed": float(plot_y0),
                "maxallowed": float(plot_y1),
                "tickmode": "array",
                "tickvals": tickvals,
                "ticktext": ticktext,
                "fixedrange": True,
                "showgrid": False,
                "zeroline": False,
            },
            dragmode="zoom",
            hovermode="closest",
            legend={"orientation": "h", "x": 0, "y": 1.08, "font": {"size": 10}},
            uirevision=f"saved-review-{animal_id}-cell-{int(cell_idx)}-{panel_idx}-{len(segment_bounds)}",
        )
        if np.isfinite(overall_snr_value):
            fig.add_annotation(
                text=f"overall SNR={_format_snr(overall_snr_value)}",
                xref="paper",
                yref="paper",
                x=1.0,
                y=1.08,
                xanchor="right",
                yanchor="bottom",
                showarrow=False,
                font={"size": 12, "color": OVERALL_SNR_COLOR},
            )
        figures.append(fig)

    return figures or [empty_figure("No saved rows to display.")]
